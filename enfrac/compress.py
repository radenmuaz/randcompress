"""Compress a file with a trained ByteFractalGen -> range-coded bundle.

Level 0 attends across the WHOLE file's timesteps (see model.py's module docstring), so
collect_logits() is called ONCE over the entire level-0 timestep sequence -- there's no
"batch_size" concept, no chunk grouping, no per-chunk state carry, nothing left to disagree about
between compress and decompress. `device` still must match (see decompress.py's module
docstring) -- TPU/GPU/CPU backends compute matmuls differently, which would desync the range
coder.

dtype="float64" (default) uses model.collect_logits_fp64() -- a much faster encode path (batches
ALL level-0 timesteps through the local recursion at once instead of one at a time) that's only
numerically safe in float64: verified empirically that the same batching in float32 flips
~1.6-2.3% of quantized CDF bins relative to the reference, while in float64 that drops to
0/44,544+ mismatches on both random-init and real trained models. dtype is therefore a
correctness invariant exactly like device/batch_size -- recorded in meta.json, read back
automatically, no decode-side override. dtype="float32" falls back to the slower collect_logits()
path if ever needed.
"""
from __future__ import annotations

import json
import math
import os
import time

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


def encode(model, raw_bytes: np.ndarray, out_dir: str, device: str = "cpu", dtype: str = "float64") -> None:
    t_wall = time.perf_counter()

    P0 = model.cfg.patch_len_list[0]
    n_raw = len(raw_bytes)
    chunks = make_chunks(raw_bytes, P0)   # [n_timesteps, P0] -- see make_chunks()'s docstring
    ctx = jnp.asarray(chunks.astype(np.int32))

    t0 = time.perf_counter()
    if dtype == "float64":
        print("  collect logits (fp64 fast path, batched across all timesteps)...", flush=True)
        # `model` stays float32 throughout (that's what gets saved to the bundle below --
        # checkpoints are always float32, see checkpoint.py's load_model docstring); cast a
        # WORKING COPY to float64 just for this computation.
        symbols_t, logits_t = model.cast_dtype(jnp.float64).collect_logits_fp64(ctx)
    else:
        print("  collect logits (level 0 attends across the whole sequence)...", flush=True)
        # Not jit-wrapped: collect_logits() -> generate() calls a symbol_fn from python-level
        # control flow (see decompress.py's symbol_fn comment for why that must stay eager).
        symbols_t, logits_t = model.collect_logits(ctx)
    t_logits = time.perf_counter() - t0

    symbols_np = np.asarray(symbols_t, dtype=np.int32)   # full stream, padding bytes included
    # IMPORTANT: keep logits_t's own dtype here (float64 when dtype="float64") -- downcasting to
    # float32 before quantize_cdf would throw away exactly the precision the fp64 forward pass
    # was computed for, reintroducing the same magnitude of rounding error we're avoiding.
    logits_np = np.asarray(logits_t, dtype=np.float64 if dtype == "float64" else np.float32)

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
               device=device,   # decompress.py MUST reuse this exact backend -- see its
                                  # module docstring: TPU/GPU/CPU matmuls aren't bit-identical
               dtype=dtype)      # decompress.py MUST reuse this exact dtype -- see module docstring
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
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "tpu", "gpu"],
                   help="jax backend for the collect_logits() recursion. Defaults to cpu: that "
                        "recursion is host-dispatch-latency-bound, not FLOP-bound, so routing "
                        "every step through an accelerator's host<->device round trip is pure "
                        "overhead here. Must be set before any other jax call, so this flag is "
                        "read before --ckpt is loaded.")
    p.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"],
                   help="compute dtype for collect_logits -- float64 (default) uses the fast "
                        "collect_logits_fp64() path (only numerically safe in float64, see "
                        "module docstring); float32 falls back to the slower collect_logits(). "
                        "Must be set before any other jax call (enables jax_enable_x64).")
    args = p.parse_args()

    jax.config.update("jax_platform_name", args.device)
    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    model = load_model(args.ckpt)   # always float32 on disk -- see checkpoint.py's load_model
    raw_bytes = load_bytes(args.input)
    encode(model, raw_bytes, args.output, device=args.device, dtype=args.dtype)


if __name__ == "__main__":
    main()
