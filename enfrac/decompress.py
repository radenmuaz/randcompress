"""Decompress a range-coded ByteFractalGen bundle -> original file. Level 0 attends across the
whole file's timesteps (see model.py's module docstring) -- decoding is autoregressive at level 0
by necessity (future bytes genuinely unknown), one call to generate() covering all n_timesteps.
`device` must match compress.py's choice (TPU/GPU/CPU matmuls aren't bit-identical to each other,
see load_bundle() below).
"""
from __future__ import annotations

import json
import os
import time

import jax
import numpy as np

from .codec import quantize_cdf, RCDecoder
from .checkpoint import load_model


def load_bundle(bundle_dir: str):
    # Read meta.json's `device` BEFORE constructing the model -- jax_platform_name must be set
    # before the first jax call (model construction creates arrays), and compress.py's chosen
    # backend MUST be reused exactly: TPU/GPU/CPU matmuls aren't bit-identical to each other any
    # more than differently-batched ones are (see module docstring), so a backend mismatch
    # desyncs the range coder just as fatally as a batch_size mismatch would. No CLI override is
    # exposed here on purpose, mirroring batch_size -- this is not a knob to hand-tune.
    with open(os.path.join(bundle_dir, "meta.json")) as f:
        meta = json.load(f)
    jax.config.update("jax_platform_name", meta.get("device", "cpu"))

    model = load_model(bundle_dir)
    with open(os.path.join(bundle_dir, "rc_stream.bin"), "rb") as f:
        rc_stream = f.read()
    return model, rc_stream, meta


def decode(bundle_dir: str, output_path: str, verify_path: str | None = None) -> bytes:
    t_wall = time.perf_counter()
    model, rc_stream, meta = load_bundle(bundle_dir)

    n_raw = meta["n_raw_bytes"]
    P0 = model.cfg.patch_len_list[0]
    n_timesteps = -(-n_raw // P0)   # ceil -- level-0 timesteps, each P0 bytes

    print(f"Bundle: {bundle_dir}")
    print(f"  n_raw={n_raw}  patch_len_list={model.cfg.patch_len_list}  rc_bytes={meta['rc_bytes']}  "
          f"n_timesteps={n_timesteps}")

    decoder = RCDecoder(rc_stream)

    def symbol_fn(logits_batch) -> list:
        # NOT jit-compatible on purpose: generate() calls this from python-level control flow
        # with real host-side state (the RCDecoder's low/high/code/pos) that needs CONCRETE
        # values every call -- wrapping the outer generate() call in jit would trace this once
        # with abstract tracers instead of actually decoding, silently corrupting the RC state
        # machine. The state itself now lives in C (rc_codec.c's rc_decode_step), not
        # reimplemented in Python -- see codec.py's RCDecoder docstring.
        cf = quantize_cdf(np.asarray(logits_batch[0], dtype=np.float32))
        sym = decoder.decode_one(cf)
        return [sym]

    t0 = time.perf_counter()
    all_bytes = model.generate(n_timesteps, symbol_fn)
    t_dec = time.perf_counter() - t0

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
