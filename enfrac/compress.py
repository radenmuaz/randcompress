"""Compress a file with a trained ByteFractalGen -> range-coded bundle.

Chunks are fully independent (no cross-chunk state), so they're processed in GROUPS of
batch_size via collect_logits()'s batched generate() -- the expensive neural forward passes
batch across a group, while the cheap range-coding arithmetic still runs one symbol at a time,
in the exact interleaved order (all rows' step-i before step-i+1) generate() naturally produces.
decompress.py MUST use the identical batch_size (and thus the identical chunk grouping) to
reproduce the same CDFs -- batched and differently-batched matmuls aren't bit-identical
(floating-point non-associativity), which for range coding is as fatal as a real logit error.
Padding bytes in the last (short) chunk are encoded too (deterministic, model learns to predict
them) and simply trimmed back out on decode.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .checkpoint import load_model, save_model
from .codec import quantize_cdf, rc_encode, rc_decode
from .model import trainable_filter
from .tokenizer import load_bytes
from .train import make_chunks


def encode(model, raw_bytes: np.ndarray, out_dir: str, batch_size: int = 32) -> None:
    t_wall = time.perf_counter()

    P0 = model.cfg.patch_len_list[0]
    n_raw = len(raw_bytes)
    chunks = make_chunks(raw_bytes, P0)
    n_chunks = chunks.shape[0]
    ctx = jnp.asarray(chunks.astype(np.int32))

    symbols_list, logits_list = [], []
    t0 = time.perf_counter()
    for start in tqdm(range(0, n_chunks, batch_size), desc="collect logits", unit="group"):
        end = min(start + batch_size, n_chunks)
        symbols, logits = model.collect_logits(ctx[start:end])   # [ (end-start)*P0 ], [ .., 256]
        symbols_list.append(symbols)
        logits_list.append(logits)
    t_logits = time.perf_counter() - t0

    symbols_t = jnp.concatenate(symbols_list)
    logits_t = jnp.concatenate(logits_list, axis=0)
    symbols_np = np.asarray(symbols_t, dtype=np.int32)   # full stream, padding bytes included
    logits_np = np.asarray(logits_t, dtype=np.float32)

    logp = jax.nn.log_softmax(logits_t.astype(jnp.float32), axis=-1)
    ce_bits = float(-jnp.take_along_axis(logp, symbols_t[:, None], axis=-1).sum()) / math.log(2)
    n_wrong = int((logits_t.argmax(-1) != symbols_t).sum())
    cdfs_np = np.stack([quantize_cdf(logits_np[j]) for j in
                        tqdm(range(logits_np.shape[0]), desc="quantize CDFs", unit="B")]).astype(np.int32)
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

    save_model(out_dir, model)   # writes model.eqx (trainable partition only) + config.json
    with open(os.path.join(out_dir, "rc_stream.bin"), "wb") as f:
        f.write(rc_stream)

    trainable, _ = eqx.partition(model, trainable_filter(model))
    param_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(trainable))
    T_valid = len(symbols_np)
    argmax_acc = (T_valid - n_wrong) / max(T_valid, 1)
    tot_bytes = param_bytes + rc_bytes
    ratio = n_raw / tot_bytes if tot_bytes > 0 else float("inf")
    meta = dict(n_raw_bytes=n_raw, rc_bytes=rc_bytes, param_bytes=param_bytes,
               total_bytes=tot_bytes, ratio=ratio,
               T_valid=T_valid, n_wrong=n_wrong, argmax_acc=argmax_acc, ce_bpb=ce_bpb,
               batch_size=batch_size)   # decompress.py MUST reuse this exact grouping
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    t_wall = time.perf_counter() - t_wall
    kbps = n_raw / max(t_wall, 1e-9) / 1e3
    print(f"\n[compress]  {t_wall:.1f}s  ({kbps:.2f} kB/s)  (logits={t_logits:.1f}s  enc={t_enc:.2f}s  dec={t_dec:.2f}s)")
    print(f"  argmax: {T_valid-n_wrong}/{T_valid} ({argmax_acc:.1%})  "
          f"CE={ce_bpb:.4f}bpb  rc={rc_bytes*8/n_raw:.4f}bpb ({rc_bytes}B)")
    print(f"  model size (params):     {param_bytes:>10,d} B")
    print(f"  rc-coded residual:       {rc_bytes:>10,d} B")
    print(f"  total bundle:            {tot_bytes:>10,d} B")
    print(f"  original file:           {n_raw:>10,d} B")
    verdict = "COMPRESSED" if ratio > 1.0 else "EXPANDED (bundle bigger than original)"
    print(f"  ACTUAL compression ratio: {ratio:.4f}x  [{verdict}]")
    print(f"Bundle: {out_dir}/")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", type=str, default=None,
                   help="python file with a `compress = dict(batch_size=...)` section (same "
                        "file passed to train.py's --config works)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="chunks processed per batched neural forward call -- decompress.py "
                        "reuses this from meta.json automatically. Defaults to --config's "
                        "compress.batch_size, else 32.")
    args = p.parse_args()

    batch_size = args.batch_size
    if batch_size is None and args.config:
        from .config import load_config_file
        mod = load_config_file(args.config)
        batch_size = getattr(mod, "compress", {}).get("batch_size")
    if batch_size is None:
        batch_size = 32

    model = load_model(args.ckpt)
    raw_bytes = load_bytes(args.input)
    encode(model, raw_bytes, args.output, batch_size=batch_size)


if __name__ == "__main__":
    main()
