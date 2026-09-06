"""Encode a file with a trained SummTransformer -> range-coded bundle.

MTP block scheme (mtp_heads = k), matches PLAN_MTP.md exactly:
  at block start, after processing byte x_t, the incremental stepper's hidden state
  x_hidden yields k CDFs in one shot: head predicts x_{t+1}, extra_heads[i] predicts
  x_{t+2+i}. Encode all k real bytes against those k CDFs, then advance the KV
  caches through the k-1 in-between bytes (teacher forced, one batched stepper
  call) to reach x_{t+k} as the next block's start. k=1 degenerates to plain AR.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import save_bundle
from .codec import quantize_cdf, rc_encode, rc_decode
from .config import TrainConfig
from .summformer import Config as ModelConfig
from .summformer import SummTransformer


@torch.no_grad()
def encode(model: SummTransformer, mcfg: ModelConfig, tcfg: TrainConfig,
          raw_bytes: np.ndarray, out_dir: str) -> None:
    t_wall = time.perf_counter()
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    n_raw = len(raw_bytes)
    if n_raw < 2:
        raise ValueError("file too short to compress (need >= 2 bytes)")
    M = mcfg.mtp_heads

    byte_t = torch.tensor(raw_bytes.astype(np.int64), device=device).unsqueeze(0)  # [1, n_raw]
    stepper, _ = model._make_incremental_stepper(1, device)

    _, x_hidden = stepper(byte_t[:, :1], 0)   # prime with byte[0] (stored raw, not coded)
    x_last = x_hidden[:, -1:, :]

    symbols: list[int] = []
    cdfs: list[np.ndarray] = []
    ce_bits = 0.0   # theoretical entropy under the model's raw (pre-quantization) softmax --
                     # a cheap sanity check against rc_bpb below, computed for free alongside
                     # the CDFs since we already have the logits in hand.
    n_wrong = 0      # argmax byte-prediction accuracy, matches randcompress/compress.py's convention
    pos = 0   # index of the last real byte already fed into the stepper

    t0 = time.perf_counter()
    pbar = tqdm(total=n_raw - 1, desc="collect logits", unit="B")
    while pos < n_raw - 1:
        n_pred = min(M, n_raw - 1 - pos)

        head_logits = [model.head(x_last)[0, 0]]                       # predicts pos+1
        for i in range(n_pred - 1):
            head_logits.append(model.extra_heads[i](x_last)[0, 0])     # predicts pos+2+i

        for j in range(n_pred):
            sym = int(raw_bytes[pos + 1 + j])
            symbols.append(sym)
            logit = head_logits[j]
            cdfs.append(quantize_cdf(logit.float().cpu().numpy()))
            log_p = torch.log_softmax(logit.float(), dim=-1)[sym]
            ce_bits += -log_p.item() / math.log(2)
            if int(logit.argmax().item()) != sym:
                n_wrong += 1

        chunk = byte_t[:, pos + 1: pos + 1 + n_pred]   # teacher-forced real bytes
        _, x_hidden = stepper(chunk, pos + 1)
        x_last = x_hidden[:, -1:, :]
        pos += n_pred
        pbar.update(n_pred)
    pbar.close()
    t_logits = time.perf_counter() - t0
    ce_bpb = ce_bits / n_raw

    symbols_np = np.array(symbols, dtype=np.int32)
    cdfs_np    = np.stack(cdfs).astype(np.int32)

    # ── RC encode ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("  encode...", end=" ", flush=True)
    rc_stream = rc_encode(symbols_np, cdfs_np)
    rc_bytes  = len(rc_stream)
    t_enc     = time.perf_counter() - t0
    print(f"{rc_bytes}B  {t_enc:.2f}s  ({rc_bytes/max(t_enc,1e-9)/1e3:.1f} kB/s)")

    # ── Verify round-trip ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("  verify...", end=" ", flush=True)
    decoded_syms = rc_decode(rc_stream, cdfs_np)
    ok           = bool(np.array_equal(decoded_syms, symbols_np))
    t_dec        = time.perf_counter() - t0
    print(f"{'OK' if ok else 'FAIL'}  {t_dec:.2f}s")
    if not ok:
        raise RuntimeError("RC round-trip verification failed")

    # ── Save bundle ───────────────────────────────────────────────────────────
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    save_bundle(out_dir, model, mcfg, tcfg, rc_stream,
               n_raw_bytes=n_raw, seed_byte=int(raw_bytes[0]))

    t_wall    = time.perf_counter() - t_wall
    tot_bytes = param_bytes + rc_bytes
    ratio     = n_raw / tot_bytes if tot_bytes > 0 else float("inf")
    T_valid   = len(symbols)
    argmax_acc = (T_valid - n_wrong) / max(T_valid, 1)
    print(f"\n[compress]  {t_wall:.1f}s  (logits={t_logits:.1f}s  enc={t_enc:.2f}s  dec={t_dec:.2f}s)")
    print(f"  argmax: {T_valid - n_wrong}/{T_valid} ({argmax_acc:.1%})  "
          f"CE={ce_bpb:.4f}bpb  rc={rc_bytes*8/n_raw:.4f}bpb ({rc_bytes}B)  "
          f"p+rc={ratio:.3f}x ({tot_bytes}B, params={param_bytes}B)")
    print(f"Bundle: {out_dir}/")

    import json
    with open(f"{out_dir}/compression.json", "w") as f:
        json.dump(dict(
            ce_bpb=ce_bpb, rc_bpb=rc_bytes * 8 / n_raw, rc_bytes=rc_bytes,
            param_bytes=param_bytes, total_bytes=tot_bytes, ratio=ratio,
            n_wrong=n_wrong, argmax_acc=argmax_acc, T_valid=T_valid,
            n_raw_bytes=n_raw,
        ), f, indent=2)


def main() -> None:
    import argparse
    from .checkpoint import load_model
    from .tokenizer import load_bytes

    p = argparse.ArgumentParser(description="overfitter compress")
    p.add_argument("--ckpt", required=True, help="dir with model.pt + config.json from train.py")
    p.add_argument("--input", required=True, help="file to compress")
    p.add_argument("--output", required=True, help="output bundle dir")
    args = p.parse_args()

    model, mcfg, tcfg = load_model(args.ckpt)
    raw_bytes = load_bytes(args.input)
    encode(model, mcfg, tcfg, raw_bytes, args.output)


if __name__ == "__main__":
    main()
