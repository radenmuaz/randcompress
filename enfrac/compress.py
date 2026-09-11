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
from .train import make_file_chunks, resolve_file_chunk_bytes


def encode(model, raw_bytes: np.ndarray, out_dir: str, device: str = "cpu", dtype: str = "float64",
           file_chunk_bytes: int | None = None, file_chunk_count: int | None = None,
           warn_padding: bool = False, chunk_batch_size: int = 1) -> None:
    """file_chunk_bytes/file_chunk_count: MUST match whatever the checkpoint was actually
    TRAINED with (see train.py's TrainConfig docstring) -- level 0's own sequence length is now
    baked into training (chunk_n_timesteps), so compressing with a different chunking than
    training used feeds level 0 out-of-distribution sequence lengths. Both None (the default)
    resolves to "one chunk = the whole file", matching a model trained without file chunking.
    Each of the n_file_chunks independent, weight-shared chunks (see
    ByteFractalGen.__call__'s docstring) gets its OWN range-coded stream regardless of batching
    below -- batching only ever speeds up the LOGITS computation, never merges chunks' streams.

    chunk_batch_size (default 1, clamped to [1, n_file_chunks]): how many chunks' logits are
    computed together per collect_logits call. 1 = today's behavior EXACTLY (per-chunk
    collect_logits_fp64(), the fp64 fast path). >1 SWITCHES to collect_logits_batched() (always
    float32, no fp64 needed -- see its docstring for why this is safe despite NOT being the fp64
    path: it's the same generate()-based reference computation collect_logits() already uses at
    batch_size=1, just with a bigger independent-chunk batch axis; batch_size is a correctness
    invariant of the same kind as device/dtype -- decompress.py MUST reuse the exact same
    chunk_batch_size, enforced via meta.json, same pattern, no decode-side override). This is
    the dominant real-world speedup: the slow part of compress was never actually the fp64 fast
    path itself, it was collect_logits_fp64's built-in verify_prefix step re-invoking the slow
    generate()-based reference every single chunk (see collect_logits_fp64's docstring) --
    batching collapses that per-chunk dispatch cost by roughly the batch factor."""
    t_wall = time.perf_counter()

    P0 = model.cfg.patch_len_list[0]
    n_raw = len(raw_bytes)
    resolved_chunk_bytes = resolve_file_chunk_bytes(n_raw, P0, file_chunk_bytes, file_chunk_count)
    chunks = make_file_chunks(raw_bytes, P0, resolved_chunk_bytes, warn=warn_padding)
    n_file_chunks, chunk_n_timesteps, _ = chunks.shape
    chunk_batch_size = max(1, min(chunk_batch_size, n_file_chunks))
    print(f"  file_chunk_bytes={resolved_chunk_bytes:,}  n_file_chunks={n_file_chunks}  "
          f"chunk_n_timesteps={chunk_n_timesteps}  chunk_batch_size={chunk_batch_size}", flush=True)

    model64 = model.cast_dtype(jnp.float64) if dtype == "float64" and chunk_batch_size == 1 else None

    t0 = time.perf_counter()
    streams: list[bytes] = []
    stream_lengths: list[int] = []
    total_ce_nats = 0.0
    total_wrong = 0
    total_positions = 0
    desc = f"collect logits + rc-encode ({chunk_batch_size}/batch)"
    for start in tqdm(range(0, n_file_chunks, chunk_batch_size), desc=desc, unit="batch"):
        end = min(start + chunk_batch_size, n_file_chunks)
        B = end - start

        if chunk_batch_size == 1:
            ctx = jnp.asarray(chunks[start].astype(np.int32))   # [chunk_n_timesteps, P0]
            if dtype == "float64":
                symbols_t, logits_t = model64.collect_logits_fp64(ctx)
            else:
                symbols_t, logits_t = model.collect_logits(ctx)
            symbols_batch = symbols_t[None, :]                   # [1, T*P0]
            logits_batch = logits_t[None, :, :]                  # [1, T*P0, 256]
        else:
            ctx = jnp.asarray(chunks[start:end].astype(np.int32))   # [B, chunk_n_timesteps, P0]
            symbols_batch, logits_batch = model.collect_logits_batched(ctx)   # [B,T*P0], [B,T*P0,256]

        for i in range(B):
            symbols_t = symbols_batch[i]
            logits_t = logits_batch[i]
            symbols_np = np.asarray(symbols_t, dtype=np.int32)
            # IMPORTANT: keep logits_t's own dtype here (float64 when dtype="float64" AND
            # chunk_batch_size==1) -- downcasting to float32 before quantize_cdf would throw away
            # exactly the precision the fp64 forward pass was computed for (see module docstring).
            logits_np = np.asarray(logits_t, dtype=np.float64 if (dtype == "float64" and chunk_batch_size == 1) else np.float32)

            logp = jax.nn.log_softmax(logits_t.astype(jnp.float32), axis=-1)
            total_ce_nats += float(-jnp.take_along_axis(logp, symbols_t[:, None], axis=-1).sum())
            total_wrong += int((logits_t.argmax(-1) != symbols_t).sum())
            total_positions += len(symbols_np)

            cdfs_np = np.stack([quantize_cdf(logits_np[j]) for j in range(logits_np.shape[0])]).astype(np.int32)
            rc_stream_c = rc_encode(symbols_np, cdfs_np)

            decoded = rc_decode(rc_stream_c, cdfs_np)
            if not np.array_equal(decoded, symbols_np):
                raise RuntimeError(f"RC round-trip verification failed on chunk {start + i}/{n_file_chunks - 1}")

            streams.append(rc_stream_c)
            stream_lengths.append(len(rc_stream_c))
    t_logits = time.perf_counter() - t0

    rc_blob = b"".join(streams)
    rc_bytes = len(rc_blob)
    ce_bpb = (total_ce_nats / math.log(2)) / n_raw

    save_model(out_dir, model)   # writes model.eqx (trainable partition only) + config.json
    with open(os.path.join(out_dir, "rc_streams.bin"), "wb") as f:
        f.write(rc_blob)

    trainable, _ = eqx.partition(model, trainable_filter(model))
    param_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(trainable))
    argmax_acc = (total_positions - total_wrong) / max(total_positions, 1)
    tot_bytes = param_bytes + rc_bytes
    ratio = n_raw / tot_bytes if tot_bytes > 0 else float("inf")
    meta = dict(n_raw_bytes=n_raw, rc_bytes=rc_bytes, param_bytes=param_bytes,
               total_bytes=tot_bytes, ratio=ratio,
               T_valid=total_positions, n_wrong=total_wrong, argmax_acc=argmax_acc, ce_bpb=ce_bpb,
               device=device,   # decompress.py MUST reuse this exact backend -- see its
                                  # module docstring: TPU/GPU/CPU matmuls aren't bit-identical
               dtype=dtype,      # decompress.py MUST reuse this exact dtype -- see module docstring
               file_chunk_bytes=resolved_chunk_bytes,   # decompress.py MUST reuse this exact
               n_file_chunks=n_file_chunks,              # chunking -- read back automatically,
               chunk_n_timesteps=chunk_n_timesteps,       # same pattern as device/batch_size/dtype
               chunk_batch_size=chunk_batch_size,          # decompress.py MUST reuse this exact
                                                             # batch size -- see collect_logits_batched()
                                                             # docstring for why (same invariant kind)
               stream_lengths=stream_lengths)             # -- byte offsets of each chunk's own
                                                            # stream within rc_streams.bin
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    t_wall = time.perf_counter() - t_wall
    kbps = n_raw / max(t_wall, 1e-9) / 1e3
    print(f"\n[compress]  {t_wall:.1f}s  ({kbps:.2f} kB/s)  (logits+encode={t_logits:.1f}s, "
          f"{n_file_chunks} chunks)")
    print(f"  argmax: {total_positions-total_wrong}/{total_positions} ({argmax_acc:.1%})  "
          f"CE={ce_bpb:.4f}bpb  rc={rc_bytes*8/n_raw:.4f}bpb ({rc_bytes}B)")
    print(f"  model size (params):     {param_bytes:>10,d} B")
    print(f"  rc-coded residual:       {rc_bytes:>10,d} B  ({n_file_chunks} streams)")
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
    p.add_argument("--file_chunk_bytes", type=int, default=None,
                   help="MUST match what the checkpoint was trained with -- by default this is "
                        "auto-read from <ckpt>/meta.json (written by train.py), no need to pass "
                        "it explicitly. Only set this to override that (rarely correct -- see "
                        "encode()'s docstring for why a mismatch feeds level 0 out-of-distribution "
                        "sequence lengths).")
    p.add_argument("--chunk_batch_size", type=int, default=1,
                   help="how many file chunks' logits to compute together per call -- 1 (default) "
                        "= today's behavior exactly (per-chunk, fp64 fast path); up to "
                        "n_file_chunks (max, batches ALL chunks in one call) uses "
                        "collect_logits_batched() instead, the real fix for compress being slow "
                        "with many small chunks (see encode()'s docstring). decompress.py MUST "
                        "use this exact same value -- it's auto-read from meta.json, no separate "
                        "flag there.")
    args = p.parse_args()

    jax.config.update("jax_platform_name", args.device)
    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    model = load_model(args.ckpt)   # always float32 on disk -- see checkpoint.py's load_model
    raw_bytes = load_bytes(args.input)

    file_chunk_bytes = args.file_chunk_bytes
    if file_chunk_bytes is None:
        ckpt_meta_path = os.path.join(args.ckpt, "meta.json")
        if os.path.exists(ckpt_meta_path):
            with open(ckpt_meta_path) as f:
                ckpt_meta = json.load(f)
            file_chunk_bytes = ckpt_meta.get("file_chunk_bytes")   # None if trained pre-chunking

    encode(model, raw_bytes, args.output, device=args.device, dtype=args.dtype,
           file_chunk_bytes=file_chunk_bytes, warn_padding=(args.file_chunk_bytes is not None),
           chunk_batch_size=args.chunk_batch_size)


if __name__ == "__main__":
    main()
