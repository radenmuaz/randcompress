"""Decode a range-coded bundle -> original file.

Mirrors compress.py's block-MTP loop exactly: same incremental stepper, same
head/extra_heads CDFs. Decoded bytes are fed back in (they equal the true bytes
since RC is lossless), so the decoder reconstructs the identical CDF sequence
the encoder used.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import load_bundle
from .codec import quantize_cdf, RC_M


# ── Pure-Python step-by-step RC state machine ────────────────────────────────
# Mirrors the C rc_codec canonical (low, high) coder exactly (see rc_codec.c).

def _rc_init(stream: bytes):
    buf  = np.frombuffer(stream, dtype=np.uint8)
    code = 0
    pos  = 0
    for _ in range(8):
        code = (code << 8) | (int(buf[pos]) if pos < len(buf) else 0)
        pos += 1
    low  = 0
    high = (1 << 64) - 1
    return low, high, code, pos, buf


def _rc_decode_one(low: int, high: int, code: int, pos: int,
                   buf: np.ndarray, cf: np.ndarray, V: int):
    M   = RC_M
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
    high   = low + rng * cum_hi // M - 1
    low    = low + rng * cum_lo // M

    while (low >> 56) == (high >> 56):
        low  = (low  << 8) & 0xFFFFFFFFFFFFFFFF
        high = ((high << 8) | 0xFF) & 0xFFFFFFFFFFFFFFFF
        b    = int(buf[pos]) if pos < len(buf) else 0
        code = ((code << 8) | b) & 0xFFFFFFFFFFFFFFFF
        pos += 1

    return sym, low, high, code, pos


# ── Main decode function ──────────────────────────────────────────────────────

@torch.no_grad()
def decode(bundle_dir: str, output_path: str, verify_path: str | None = None) -> bytes:
    t_wall = time.perf_counter()
    model, mcfg, tcfg, rc_stream, meta = load_bundle(bundle_dir)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    n_raw     = meta["n_raw_bytes"]
    seed_byte = meta["seed_byte"]
    V         = model.vocab
    M         = mcfg.mtp_heads

    print(f"Bundle: {bundle_dir}")
    print(f"  n_raw={n_raw}  seed={seed_byte}  rc_bytes={meta['rc_bytes']}  mtp_heads={M}")

    stepper, _ = model._make_incremental_stepper(1, device)
    low, high, code, rc_pos, buf = _rc_init(rc_stream)

    decoded = [seed_byte]
    byte_t = torch.tensor([[seed_byte]], dtype=torch.long, device=device)
    _, x_hidden = stepper(byte_t, 0)
    x_last = x_hidden[:, -1:, :]

    pos = 0   # index of the last real (known) byte
    t0 = time.perf_counter()
    pbar = tqdm(total=n_raw - 1, desc="decode", unit="B")
    while pos < n_raw - 1:
        n_pred = min(M, n_raw - 1 - pos)

        head_logits = [model.head(x_last)[0, 0]]
        for i in range(n_pred - 1):
            head_logits.append(model.extra_heads[i](x_last)[0, 0])

        block_bytes = []
        for j in range(n_pred):
            cf = quantize_cdf(head_logits[j].float().cpu().numpy())
            sym, low, high, code, rc_pos = _rc_decode_one(low, high, code, rc_pos, buf, cf, V)
            block_bytes.append(sym)
        decoded.extend(block_bytes)

        chunk = torch.tensor([block_bytes], dtype=torch.long, device=device)
        _, x_hidden = stepper(chunk, pos + 1)
        x_last = x_hidden[:, -1:, :]
        pos += n_pred
        pbar.update(n_pred)
    pbar.close()
    t_dec = time.perf_counter() - t0

    raw_out = bytes(decoded[:n_raw])
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

    p = argparse.ArgumentParser(description="overfitter decompress")
    p.add_argument("--bundle", required=True, help="bundle dir from compress.py")
    p.add_argument("--output", required=True, help="output file path")
    p.add_argument("--verify", default=None, help="original file, to verify against")
    args = p.parse_args()

    decode(args.bundle, args.output, args.verify)


if __name__ == "__main__":
    main()
