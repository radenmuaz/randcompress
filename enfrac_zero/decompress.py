"""Decompress a range-coded baseline ByteFractalGen bundle -> original file. Identical scheme to
enfrac/decompress.py (see its module docstring) -- only the checkpoint import differs.
"""
from __future__ import annotations

import json
import os
import time

import jax
import numpy as np
from tqdm import tqdm

from enfrac.codec import quantize_cdf, RC_M
from .checkpoint import load_model


def _rc_init(stream: bytes):
    buf = np.frombuffer(stream, dtype=np.uint8)
    code = 0
    pos = 0
    for _ in range(8):
        code = (code << 8) | (int(buf[pos]) if pos < len(buf) else 0)
        pos += 1
    low = 0
    high = (1 << 64) - 1
    return low, high, code, pos, buf


def _rc_decode_one(low: int, high: int, code: int, pos: int, buf: np.ndarray, cf: np.ndarray, V: int):
    M = RC_M
    rng = high - low + 1
    lo_s, hi_s = 0, V - 1
    while lo_s < hi_s:
        mid = (lo_s + hi_s + 1) // 2
        if low + rng * int(cf[mid]) // M <= code:
            lo_s = mid
        else:
            hi_s = mid - 1
    sym = lo_s
    cum_lo = int(cf[sym])
    cum_hi = int(cf[sym + 1])
    high = low + rng * cum_hi // M - 1
    low = low + rng * cum_lo // M
    while (low >> 56) == (high >> 56):
        low = (low << 8) & 0xFFFFFFFFFFFFFFFF
        high = ((high << 8) | 0xFF) & 0xFFFFFFFFFFFFFFFF
        b = int(buf[pos]) if pos < len(buf) else 0
        code = ((code << 8) | b) & 0xFFFFFFFFFFFFFFFF
        pos += 1
    return sym, low, high, code, pos


def load_bundle(bundle_dir: str):
    # Read meta.json's `device` BEFORE constructing the model -- see enfrac/decompress.py's
    # load_bundle() for the full rationale (backend mismatch desyncs the range coder). No CLI
    # override exposed here on purpose, mirroring batch_size.
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
    n_chunks = -(-n_raw // P0)
    batch_size = meta.get("batch_size", 1)
    V = 256

    print(f"Bundle: {bundle_dir}")
    print(f"  n_raw={n_raw}  patch_len_list={model.cfg.patch_len_list}  rc_bytes={meta['rc_bytes']}  "
          f"batch_size={batch_size}")

    low, high, code, pos, buf = _rc_init(rc_stream)

    def make_symbol_fn(Bg: int):
        def symbol_fn(logits_batch) -> list:
            nonlocal low, high, code, pos
            logits_np = np.asarray(logits_batch, dtype=np.float32)
            syms = []
            for b in range(Bg):
                cf = quantize_cdf(logits_np[b])
                sym, low, high, code, pos = _rc_decode_one(low, high, code, pos, buf, cf, V)
                syms.append(sym)
            return syms
        return symbol_fn

    all_bytes = []
    t0 = time.perf_counter()
    with tqdm(total=n_chunks, desc="decode", unit="chunk") as pbar:
        for start in range(0, n_chunks, batch_size):
            end = min(start + batch_size, n_chunks)
            Bg = end - start
            chunks_bytes = model.generate(make_symbol_fn(Bg), batch_size=Bg)
            for cb in chunks_bytes:
                all_bytes.extend(cb)
            pbar.update(Bg)
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
