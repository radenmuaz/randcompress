"""
Encode a file using trained adapter weights → range-coded bundle.

Collects teacher-forced logits chunk-by-chunk, quantizes CDFs, rc_encodes,
verifies round-trip, prints CE bpb / argmax accuracy / timing, saves bundle.
"""
from __future__ import annotations

import hashlib
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .checkpoint import save_bundle
from .codec import quantize_cdf, rc_encode, rc_decode
from .config import ModelConfig, TrainConfig
from .tokenizer import bytes_to_tokens, tokens_to_bytes
from .train import make_chunks


@torch.no_grad()
def encode(model, frozen, adapters, mcfg: ModelConfig, tcfg: TrainConfig,
           raw_bytes: np.ndarray, out_dir: str):
    t_wall = time.perf_counter()
    ib, ob, oh = tcfg.input_bits, tcfg.output_bits, tcfg.output_heads
    V         = 2 ** ob
    out_mask  = (1 << ob) - 1
    n_raw     = len(raw_bytes)
    device    = torch.device("cpu")

    all_inputs, all_targets = make_chunks(raw_bytes, tcfg)
    num_chunks, chunk_size  = all_inputs.shape[:2]
    T_total = num_chunks * chunk_size

    # ── Collect logits step-by-step ───────────────────────────────────────────
    # precompute_weights=True (default): apply HiRA once → fast, more memory
    # precompute_weights=False: recompute per step → slow, less memory
    t0     = time.perf_counter()
    states = model.init_states(1, device)
    logits_list: list[np.ndarray] = []

    if tcfg.precompute_weights:
        ew = model.precompute_eff_weights(frozen, adapters)
        step_fn = lambda tok, st, t: model.step_eff(ew, tok, st, t)
    else:
        step_fn = lambda tok, st, t: model.step(frozen, adapters, tok, st, t)

    pbar = tqdm(total=T_total, desc="collect logits", unit="tok")
    with torch.no_grad():
        for ci in range(num_chunks):
            for pos in range(chunk_size):
                t_global = ci * chunk_size + pos
                tok = torch.tensor([int(all_inputs[ci, pos])], dtype=torch.long)
                logit, states = step_fn(tok, states, t_global)
                logits_list.append(logit[0].float().cpu().numpy())  # [oh, V]
                pbar.update(1)
    pbar.close()
    t_logits = time.perf_counter() - t0

    logits_np = np.stack(logits_list)                                    # [T_total, oh, V]
    tgts_np   = all_targets.reshape(-1, oh)[:, 0].astype(np.int32)      # [T_total]

    # Filter valid (non-pad) positions
    valid    = tgts_np >= 0
    vlog     = logits_np[valid]   # [T_v, oh, V]
    vtgt_raw = tgts_np[valid]     # [T_v]
    T_v      = int(valid.sum())

    # ── CE bpb + argmax accuracy (head 0) ────────────────────────────────────
    lp_h0   = F.log_softmax(torch.tensor(vlog[:, 0, :]), dim=-1).numpy()
    tgt_lp  = lp_h0[np.arange(T_v), vtgt_raw]
    ce_bits = float(-np.sum(tgt_lp) / math.log(2))
    ce_bpb  = ce_bits / n_raw
    n_wrong = int(np.sum(np.argmax(vlog[:, 0, :], axis=-1) != vtgt_raw))
    acc     = (T_v - n_wrong) / max(T_v, 1)

    # ── Build symbol stream (multi-head aware) ────────────────────────────────
    # For oh=1: symbols = vtgt_raw
    # For oh>1, ib=8, ob<8: reconstruct byte then split per head
    if oh == 1:
        symbols = vtgt_raw
    else:
        # all_targets: [num_chunks, S, oh] → [T_total, oh], then filter valid rows
        all_tgts_flat = all_targets.reshape(-1, oh)  # [T_total, oh]
        symbols = all_tgts_flat[valid].ravel().astype(np.int32)  # [T_v * oh]

    # ── Quantize CDFs ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    n_syms = len(symbols)
    cdfs   = np.empty((n_syms, V + 1), dtype=np.int32)

    pbar_cdf = tqdm(total=n_syms, desc="quantize CDFs", unit="tok")
    if oh == 1:
        for i in range(T_v):
            cdfs[i] = quantize_cdf(vlog[i, 0])
            pbar_cdf.update(1)
    else:
        idx = 0
        for i in range(T_v):
            for h in range(oh):
                cdfs[idx] = quantize_cdf(vlog[i, h])
                idx += 1
                pbar_cdf.update(1)
    pbar_cdf.close()
    t_cdf = time.perf_counter() - t0

    # ── RC encode ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"  encode...", end=" ", flush=True)
    rc_stream = rc_encode(symbols, cdfs)
    rc_bytes  = len(rc_stream)
    t_enc     = time.perf_counter() - t0
    print(f"{rc_bytes}B  {t_enc:.2f}s  ({rc_bytes/max(t_enc,1e-9)/1e3:.1f} kB/s)")

    # ── Verify round-trip ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"  verify...", end=" ", flush=True)
    decoded_syms = rc_decode(rc_stream, cdfs)
    ok           = bool(np.all(decoded_syms == symbols))
    t_dec        = time.perf_counter() - t0
    print(f"{'OK' if ok else 'FAIL'}  {t_dec:.2f}s")
    if not ok:
        raise RuntimeError("RC round-trip verification failed")

    # ── Save bundle ───────────────────────────────────────────────────────────
    param_bytes = sum(v.numel() * v.element_size() for v in adapters.values())
    seed_token  = int(all_inputs[0, 0])
    save_bundle(out_dir, adapters, mcfg, tcfg, rc_stream,
                n_raw_bytes=n_raw, seed_token=seed_token,
                T_valid=T_v, n_syms=n_syms)

    # ── Summary ───────────────────────────────────────────────────────────────
    t_wall = time.perf_counter() - t_wall
    tot_bytes = param_bytes + rc_bytes
    ratio     = n_raw / tot_bytes if tot_bytes > 0 else float("inf")
    print(f"\n[compress]  {t_wall:.1f}s  "
          f"(logits={t_logits:.1f}s  cdfs={t_cdf:.1f}s  "
          f"enc={t_enc:.2f}s  dec={t_dec:.2f}s)")
    print(f"  argmax: {T_v-n_wrong}/{T_v} ({acc:.1%})"
          f"  CE={ce_bpb:.4f}bpb"
          f"  rc={rc_bytes*8/n_raw:.4f}bpb ({rc_bytes}B)"
          f"  p+rc={ratio:.3f}x ({tot_bytes}B)")
    print(f"Bundle: {out_dir}/")

    # Save compression stats
    import json
    stats = dict(
        ce_bpb=ce_bpb, rc_bpb=rc_bytes * 8 / n_raw,
        rc_bytes=rc_bytes, param_bytes=param_bytes,
        total_bytes=tot_bytes, ratio=ratio,
        n_wrong=n_wrong, argmax_acc=acc,
        T_valid=T_v, n_raw_bytes=n_raw,
    )
    with open(os.path.join(out_dir, "compression.json"), "w") as f:
        json.dump(stats, f, indent=2)
