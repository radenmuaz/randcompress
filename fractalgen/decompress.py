"""Decompress a range-coded ByteFractalGen bundle -> original file.

Mirrors compress.py's chunk order exactly: for each patch_len_list[0]-byte chunk, model.generate()
recursively produces bytes in the same left-to-right depth-first order collect_logits() teacher-
forced them in, so decoding the SAME range-coded stream in that order reconstructs it exactly.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from overfitter.codec import quantize_cdf, RC_M
from overfitter.decompress import _rc_init, _rc_decode_one
from overfitter.tokenizer import load_bytes

from model import ByteFractalGen, FractalConfig


def load_bundle(bundle_dir: str):
    with open(os.path.join(bundle_dir, "config.json")) as f:
        raw = json.load(f)
    raw["patch_len_list"] = tuple(raw["patch_len_list"])
    model = ByteFractalGen(FractalConfig(**raw))
    state = torch.load(os.path.join(bundle_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    with open(os.path.join(bundle_dir, "rc_stream.bin"), "rb") as f:
        rc_stream = f.read()
    with open(os.path.join(bundle_dir, "meta.json")) as f:
        meta = json.load(f)
    return model, rc_stream, meta


@torch.no_grad()
def decode(bundle_dir: str, output_path: str, verify_path: str | None = None) -> bytes:
    t_wall = time.perf_counter()
    model, rc_stream, meta = load_bundle(bundle_dir)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    n_raw = meta["n_raw_bytes"]
    P0 = model.cfg.patch_len_list[0]
    n_chunks = -(-n_raw // P0)   # ceil
    V = 256

    print(f"Bundle: {bundle_dir}")
    print(f"  n_raw={n_raw}  patch_len_list={model.cfg.patch_len_list}  rc_bytes={meta['rc_bytes']}")

    low, high, code, pos, buf = _rc_init(rc_stream)

    def symbol_fn(logits: torch.Tensor) -> int:
        nonlocal low, high, code, pos
        cf = quantize_cdf(logits.cpu().numpy())
        sym, low, high, code, pos = _rc_decode_one(low, high, code, pos, buf, cf, V)
        return sym

    all_bytes = []
    t0 = time.perf_counter()
    for _ in tqdm(range(n_chunks), desc="decode", unit="chunk"):
        chunk_bytes = model.generate(symbol_fn)   # list of P0 ints
        all_bytes.extend(chunk_bytes)
    t_dec = time.perf_counter() - t0

    raw_out = bytes(all_bytes[:n_raw])
    with open(output_path, "wb") as f:
        f.write(raw_out)
    t_wall = time.perf_counter() - t_wall
    print(f"Written {len(raw_out)} bytes -> {output_path}  ({t_wall:.1f}s, decode={t_dec:.1f}s)")

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
