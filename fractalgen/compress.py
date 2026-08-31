"""Compress a file with a trained ByteFractalGen -> range-coded bundle.

Each patch_len_list[0]-byte chunk is processed independently (no cross-chunk state): teacher-
forced collect_logits() gives every byte's CDF in the SAME order generate() will later produce
them in, so encode/decode stay consistent. Padding bytes in the last (short) chunk are encoded
too (deterministic, model learns to predict them) and simply trimmed back out on decode.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from overfitter.codec import quantize_cdf, rc_encode, rc_decode
from overfitter.tokenizer import load_bytes

from checkpoint import load_model
from train import make_chunks


@torch.no_grad()
def encode(model, raw_bytes: np.ndarray, out_dir: str) -> None:
    t_wall = time.perf_counter()
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    P0 = model.cfg.patch_len_list[0]
    n_raw = len(raw_bytes)
    chunks = make_chunks(raw_bytes, P0)
    ctx = torch.tensor(chunks.astype(np.int64), device=device)

    symbols_list, cdfs_list = [], []
    ce_bits = 0.0
    n_wrong = 0
    t0 = time.perf_counter()
    for i in tqdm(range(chunks.shape[0]), desc="collect logits", unit="chunk"):
        symbols, logits = model.collect_logits(ctx[i: i + 1])   # [P0], [P0, 256]
        symbols_np = symbols.cpu().numpy().astype(np.int32)
        logp = torch.log_softmax(logits.float(), dim=-1)
        ce_bits += -logp[torch.arange(P0), symbols].sum().item() / math.log(2)
        n_wrong += int((logits.argmax(-1) != symbols).sum().item())
        for j in range(P0):
            cdfs_list.append(quantize_cdf(logits[j].cpu().numpy()))
        symbols_list.append(symbols_np)
    t_logits = time.perf_counter() - t0

    symbols_np = np.concatenate(symbols_list)   # full stream, padding bytes included
    cdfs_np = np.stack(cdfs_list).astype(np.int32)
    ce_bpb = ce_bits / n_raw

    t0 = time.perf_counter()
    print("  encode...", end=" ", flush=True)
    rc_stream = rc_encode(symbols_np, cdfs_np)
    rc_bytes = len(rc_stream)
    t_enc = time.perf_counter() - t0
    print(f"{rc_bytes}B  {t_enc:.2f}s")

    t0 = time.perf_counter()
    print("  verify...", end=" ", flush=True)
    decoded = rc_decode(rc_stream, cdfs_np)
    ok = bool(np.array_equal(decoded, symbols_np))
    t_dec = time.perf_counter() - t0
    print(f"{'OK' if ok else 'FAIL'}  {t_dec:.2f}s")
    if not ok:
        raise RuntimeError("RC round-trip verification failed")

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    import json
    from dataclasses import asdict
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(model.cfg), f, indent=2)
    with open(os.path.join(out_dir, "rc_stream.bin"), "wb") as f:
        f.write(rc_stream)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    T_valid = len(symbols_np)
    argmax_acc = (T_valid - n_wrong) / max(T_valid, 1)
    meta = dict(n_raw_bytes=n_raw, rc_bytes=rc_bytes, param_bytes=param_bytes,
               T_valid=T_valid, n_wrong=n_wrong, argmax_acc=argmax_acc, ce_bpb=ce_bpb)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    t_wall = time.perf_counter() - t_wall
    tot_bytes = param_bytes + rc_bytes
    ratio = n_raw / tot_bytes if tot_bytes > 0 else float("inf")
    print(f"\n[compress]  {t_wall:.1f}s  (logits={t_logits:.1f}s  enc={t_enc:.2f}s  dec={t_dec:.2f}s)")
    print(f"  argmax: {T_valid-n_wrong}/{T_valid} ({argmax_acc:.1%})  "
          f"CE={ce_bpb:.4f}bpb  rc={rc_bytes*8/n_raw:.4f}bpb ({rc_bytes}B)  "
          f"p+rc={ratio:.3f}x ({tot_bytes}B, params={param_bytes}B)")
    print(f"Bundle: {out_dir}/")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model = load_model(args.ckpt)
    raw_bytes = load_bytes(args.input)
    encode(model, raw_bytes, args.output)


if __name__ == "__main__":
    main()
