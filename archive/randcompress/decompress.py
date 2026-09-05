"""
Decode a range-coded bundle → original file.

Step-by-step AR loop:
  for each position t:
    1. run model.step(cur_tok) → logit → quantize CDF
    2. decode ONE symbol from RC stream using that CDF
    3. feed decoded symbol as next input (it equals the true token since RC is lossless)

This exactly mirrors compress.py's teacher-forced CDF collection, so the CDFs
are identical and the RC decoder recovers the exact original tokens.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import load_bundle
from .codec import quantize_cdf, RC_M, RC_PREC
from .models import get_model
from .tokenizer import bytes_to_tokens, tokens_to_bytes


# ── Pure-Python step-by-step RC state machine ────────────────────────────────
# Mirrors the C rc_codec canonical (low, high) coder exactly.

def _rc_init(stream: bytes):
    buf   = np.frombuffer(stream, dtype=np.uint8)
    code  = 0
    pos   = 0
    for _ in range(8):
        code = (code << 8) | (int(buf[pos]) if pos < len(buf) else 0)
        pos += 1
    low  = 0
    high = (1 << 64) - 1          # UINT64_MAX
    return low, high, code, pos, buf


def _rc_decode_one(low: int, high: int, code: int, pos: int,
                   buf: np.ndarray, cf: np.ndarray, V: int):
    """Decode one symbol from the RC stream. Returns (sym, low, high, code, pos)."""
    M   = RC_M
    rng = high - low + 1

    # Binary search for symbol such that low + rng*cf[sym]/M <= code < low + rng*cf[sym+1]/M
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

    # Normalise: while top byte agrees, shift out and refill
    while (low >> 56) == (high >> 56):
        low  = (low  << 8) & 0xFFFFFFFFFFFFFFFF
        high = ((high << 8) | 0xFF) & 0xFFFFFFFFFFFFFFFF
        b    = int(buf[pos]) if pos < len(buf) else 0
        code = ((code << 8) | b) & 0xFFFFFFFFFFFFFFFF
        pos += 1

    return sym, low, high, code, pos


# ── Main decode function ──────────────────────────────────────────────────────

def decode(bundle_dir: str, output_path: str, verify_path: str | None = None):
    t_wall = time.perf_counter()
    mcfg, tcfg, adapters, rc_stream, meta = load_bundle(bundle_dir)

    n_raw      = meta["n_raw_bytes"]
    T_valid    = meta["T_valid"]
    seed_token = meta["seed_token"]
    oh         = meta.get("output_heads", tcfg.output_heads)
    ob         = meta.get("output_bits",  tcfg.output_bits)
    ib         = meta.get("input_bits",   tcfg.input_bits)
    V          = 2 ** ob
    out_mask   = (1 << ob) - 1

    print(f"Bundle: {bundle_dir}")
    print(f"  n_raw={n_raw}  T_valid={T_valid}  seed={seed_token}")
    print(f"  rc_bytes={meta['rc_bytes']}  ib={ib}  ob={ob}  oh={oh}")

    print("Reconstructing model...")
    model  = get_model(mcfg, tcfg)
    frozen = model.init_frozen(mcfg.seed)
    device = torch.device("cpu")
    states = model.init_states(1, device)
    k      = mcfg.mtp_k

    # Init RC state machine
    low, high, code, pos, buf = _rc_init(rc_stream)

    cur_tok      = torch.tensor([seed_token], dtype=torch.long)
    decoded_syms = []   # list of ints (symbols for each head, each position)

    t0 = time.perf_counter()
    with torch.no_grad():
        pbar = tqdm(total=T_valid, desc="decode", unit="tok")
        for t in range(0, T_valid, k):
            # step_mtp: process cur_tok, return k CDFs + state after 1 RNN step
            logits_k, states = model.step_mtp(frozen, adapters, cur_tok, states, t)
            # logits_k: [1, k, oh, V] — head j covers position t+j

            block_toks = []   # decoded tokens for this block (for state advance)
            n_heads = min(k, T_valid - t)
            for j in range(n_heads):
                syms_at_j = []
                for h in range(oh):
                    cf  = quantize_cdf(logits_k[0, j, h].float().cpu().numpy())
                    sym, low, high, code, pos = _rc_decode_one(
                        low, high, code, pos, buf, cf, V)
                    syms_at_j.append(sym)
                    decoded_syms.append(sym)

                # Reconstruct input token from this position's decoded symbols
                if ib == 8 and ob < 8:
                    byte_val = 0
                    for h in range(oh):
                        byte_val |= (syms_at_j[h] & out_mask) << (h * ob)
                    byte_val &= 0xFF
                    tok_val = int(bytes_to_tokens(np.array([byte_val], np.uint8), ib)[0])
                else:
                    tok_val = syms_at_j[0]
                block_toks.append(tok_val)

            pbar.update(n_heads)

            # Advance state through decoded tokens x_{t+1}..x_{t+k-1}.
            # states already holds state after processing cur_tok (=x_t) from step_mtp.
            # scan_states advances through the next k-1 decoded tokens.
            if len(block_toks) > 1:
                adv = torch.tensor([block_toks[:-1]], dtype=torch.long)
                states = model.scan_states(frozen, adapters, adv, states, t + 1)

            cur_tok = torch.tensor([block_toks[-1]], dtype=torch.long)

        pbar.close()
    t_dec = time.perf_counter() - t0

    # ── Reconstruct bytes ─────────────────────────────────────────────────────
    if oh == 1:
        toks_arr = np.array([seed_token] + decoded_syms, dtype=np.int32)
        raw_out  = tokens_to_bytes(toks_arr, ib)[:n_raw]
    elif ib == 8 and ob < 8:
        syms_arr    = np.array(decoded_syms, dtype=np.uint8).reshape(T_valid, oh)
        pred_bytes  = np.zeros(T_valid, np.uint8)
        for h in range(oh):
            pred_bytes |= (syms_arr[:, h] & out_mask) << (h * ob)
        seed_byte = int(tokens_to_bytes(np.array([seed_token], np.int32), ib)[0])
        raw_out   = np.concatenate([[seed_byte], pred_bytes])[:n_raw].astype(np.uint8)
    else:
        toks_arr = np.array([seed_token] + decoded_syms[:T_valid], dtype=np.int32)
        raw_out  = tokens_to_bytes(toks_arr, ib)[:n_raw]

    # ── Write output ──────────────────────────────────────────────────────────
    with open(output_path, "wb") as f:
        f.write(bytes(raw_out))
    t_wall = time.perf_counter() - t_wall
    print(f"Written {len(raw_out)} bytes → {output_path}  ({t_wall:.1f}s, decode={t_dec:.1f}s)")

    # ── Verify ────────────────────────────────────────────────────────────────
    if verify_path:
        with open(verify_path, "rb") as f:
            ref = np.frombuffer(f.read(), dtype=np.uint8)
        min_len = min(len(raw_out), len(ref))
        n_wrong = int(np.sum(raw_out[:min_len] != ref[:min_len]))
        if n_wrong == 0 and len(raw_out) == len(ref):
            print("Verification: PERFECT MATCH ✓")
        else:
            wrong_idx = np.where(raw_out[:min_len] != ref[:min_len])[0]
            print(f"Verification: {n_wrong}/{min_len} bytes wrong")
            if len(wrong_idx):
                print(f"  first wrong byte at position {int(wrong_idx[0])}")
    return raw_out
