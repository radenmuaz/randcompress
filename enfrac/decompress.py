"""Decompress a range-coded ByteFractalGen bundle -> original file. Level 0 attends across one
file CHUNK's timesteps at a time (see model.py's ByteFractalGen.__call__ docstring / train.py's
make_file_chunks) -- decoding is autoregressive at level 0 by necessity within each chunk (future
bytes genuinely unknown), one call to generate() per BATCH of chunks (chunk_batch_size independent
chunks decoded in LOCKSTEP -- see generate()'s docstring: still fully sequential in time within
each chunk, real autoregression, but chunk_batch_size chunks advance together, each keeping its
OWN independent RCDecoder state -- "stateful between chunks" is handled by simply holding
chunk_batch_size separate decoder objects, one per chunk in the batch), using each chunk's OWN
independent RC stream (see compress.py's encode() docstring for why streams are kept separate
rather than concatenated: chunks never attend across each other, so nothing about compress ever
needed them to share one stream). `device` must match compress.py's choice (TPU/GPU/CPU matmuls
aren't bit-identical to each other, see load_bundle() below); `dtype` must ALSO match (see
compress.py's module docstring) -- float32 vs float64 arithmetic produces different logits, which
would desync the range coder just as fatally as a device/batch_size mismatch. `file_chunk_bytes`/
`n_file_chunks`/`stream_lengths`/`chunk_batch_size` are further invariants of the same kind --
decoding chunk c with the wrong stream, chunk_n_timesteps, or batch_size desyncs that chunk's
range coder exactly like a dtype/device mismatch would (batch_size specifically because
collect_logits_batched()'s logits differ, by ~1e-6, from a DIFFERENT batch size's -- see its
docstring: compress and decompress must agree with EACH OTHER at the exact same batch_size, not
with some reference). All of these are read from meta.json automatically, no CLI override.
"""
from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .codec import quantize_cdf, RCDecoder
from .checkpoint import load_model


def load_bundle(bundle_dir: str):
    # Read meta.json's `device`/`dtype` BEFORE constructing the model -- jax_platform_name/
    # jax_enable_x64 must be set before the first jax call (model construction creates arrays),
    # and compress.py's chosen backend/dtype MUST be reused exactly: TPU/GPU/CPU matmuls aren't
    # bit-identical to each other any more than float32 vs float64 arithmetic is (see module
    # docstring), so a mismatch desyncs the range coder just as fatally as a batch_size mismatch
    # would. No CLI override is exposed here on purpose, mirroring batch_size -- not a knob to
    # hand-tune.
    with open(os.path.join(bundle_dir, "meta.json")) as f:
        meta = json.load(f)
    jax.config.update("jax_platform_name", meta.get("device", "cpu"))
    dtype_str = meta.get("dtype", "float32")
    if dtype_str == "float64":
        jax.config.update("jax_enable_x64", True)

    # REAL BUG found and fixed this session: compress.py's encode() only ever casts its model to
    # float64 (model64) when chunk_batch_size==1 (the fp64 fast path) -- collect_logits_batched()
    # (chunk_batch_size>1) always runs the plain float32 model, even when dtype="float64" (the CLI
    # default). The saved bundle's actual weight dtype therefore depends on BOTH dtype AND
    # chunk_batch_size, not dtype alone -- casting to float64 here whenever dtype_str=="float64"
    # (ignoring chunk_batch_size) silently ran decode with DIFFERENT weights than compress used,
    # desyncing the range coder (32/800 bytes wrong in a caught local repro, not a crash -- this
    # is the same class of bug the device/batch_size invariants exist to prevent).
    n_file_chunks = meta.get("n_file_chunks", 1)
    chunk_batch_size = max(1, min(meta.get("chunk_batch_size", 1), n_file_chunks))
    use_fp64_model = dtype_str == "float64" and chunk_batch_size == 1
    model = load_model(bundle_dir, dtype=jnp.float64 if use_fp64_model else None)
    with open(os.path.join(bundle_dir, "rc_streams.bin"), "rb") as f:
        rc_blob = f.read()
    return model, rc_blob, meta


def decode(bundle_dir: str, output_path: str, verify_path: str | None = None) -> bytes:
    t_wall = time.perf_counter()
    model, rc_blob, meta = load_bundle(bundle_dir)

    n_raw = meta["n_raw_bytes"]
    n_file_chunks = meta["n_file_chunks"]
    chunk_n_timesteps = meta["chunk_n_timesteps"]
    stream_lengths = meta["stream_lengths"]
    chunk_batch_size = max(1, min(meta.get("chunk_batch_size", 1), n_file_chunks))
    assert len(stream_lengths) == n_file_chunks, \
        f"meta.json corrupt: {len(stream_lengths)} stream_lengths but n_file_chunks={n_file_chunks}"

    print(f"Bundle: {bundle_dir}")
    print(f"  n_raw={n_raw}  patch_len_list={model.cfg.patch_len_list}  rc_bytes={meta['rc_bytes']}  "
          f"n_file_chunks={n_file_chunks}  chunk_n_timesteps={chunk_n_timesteps}  "
          f"chunk_batch_size={chunk_batch_size}")

    logits_np_dtype = np.float64 if (meta.get("dtype", "float32") == "float64" and chunk_batch_size == 1) \
        else np.float32   # collect_logits_batched() is always float32 -- see compress.py's encode()

    # Precompute each chunk's own byte-slice of rc_streams.bin (order matches compress.py's loop).
    stream_slices = []
    offset = 0
    for length in stream_lengths:
        stream_slices.append(rc_blob[offset:offset + length])
        offset += length

    t0 = time.perf_counter()
    all_chunks_bytes: list[list[int]] = [None] * n_file_chunks
    desc = f"decompress chunks ({chunk_batch_size}/batch)"
    for start in tqdm(range(0, n_file_chunks, chunk_batch_size), desc=desc, unit="batch"):
        end = min(start + chunk_batch_size, n_file_chunks)
        B = end - start
        decoders = [RCDecoder(stream_slices[c]) for c in range(start, end)]

        def symbol_fn(logits_batch, decoders=decoders) -> list:
            # NOT jit-compatible on purpose: generate() calls this from python-level control flow
            # with real host-side state (each chunk's OWN RCDecoder's low/high/code/pos) that
            # needs CONCRETE values every call -- wrapping the outer generate() call in jit would
            # trace this once with abstract tracers instead of actually decoding, silently
            # corrupting the RC state machine. The state itself lives in C (rc_codec.c's
            # rc_decode_step), not reimplemented in Python -- see codec.py's RCDecoder docstring.
            # IMPORTANT: preserve logits_batch's own dtype here -- downcasting would compute a
            # DIFFERENT CDF than compress.py did, desyncing the range coder (see compress.py's
            # module docstring). Each of the B chunks in this batch uses its OWN decoder -- this
            # IS "handling the stateful case between batch members": B independent decoder
            # objects, not one shared state.
            syms = []
            for b, dec in enumerate(decoders):
                cf = quantize_cdf(np.asarray(logits_batch[b], dtype=logits_np_dtype))
                syms.append(dec.decode_one(cf))
            return syms

        batch_bytes = model.generate(chunk_n_timesteps, symbol_fn, batch_size=B)   # list of B lists
        for i, c in enumerate(range(start, end)):
            all_chunks_bytes[c] = batch_bytes[i]
    t_dec = time.perf_counter() - t0

    all_bytes: list[int] = [b for chunk in all_chunks_bytes for b in chunk]

    raw_out = bytes(all_bytes[:n_raw])
    with open(output_path, "wb") as f:
        f.write(raw_out)
    t_wall = time.perf_counter() - t_wall
    kbps = n_raw / max(t_dec, 1e-9) / 1e3
    print(f"Written {len(raw_out)} bytes -> {output_path}  ({t_wall:.1f}s, decode={t_dec:.1f}s, {kbps:.2f} kB/s)")

    if verify_path:
        with open(verify_path, "rb") as f:
            ref = f.read()
        if raw_out == ref:
            print("Verification: PERFECT MATCH")
        else:
            ref_arr = np.frombuffer(ref, dtype=np.uint8)
            out_arr = np.frombuffer(raw_out, dtype=np.uint8)
            min_len = min(len(ref_arr), len(out_arr))
            n_wrong = int(np.sum(out_arr[:min_len] != ref_arr[:min_len]))
            print(f"Verification: {n_wrong}/{min_len} bytes wrong (len {len(out_arr)} vs {len(ref_arr)})")

    return raw_out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--verify", default=None)
    args = p.parse_args()
    decode(args.bundle, args.output, args.verify)


if __name__ == "__main__":
    main()
