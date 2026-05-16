"""
Decompressor for linear_rnn_srp compressed bundles.

Bundle layout (written by save_compressed):
  ckpt_dir/
    config.json      — full Config dict (architecture + seed)
    params.pkl       — HiRA adapter weights as numpy arrays
    rc_stream.bin    — range-coded bytes
    meta.json        — {n_raw_bytes, T_valid, seed_token, rc_bytes, ...}

Decompression steps
-------------------
1. Load bundle; restore params with a custom unpickler (handles __main__ pickle)
2. Reconstruct frozen base model from config.seed (deterministic)
3. Restore HiRA adapter weights
4. AR decode: for each token t = 0..T_valid-1
     logits = forward_step(model, cur_token, state, t % chunk_size)
     cdf    = quantize_cdf(logits[head=0])
     token  = rc_decode_step(cdf, rc_state)   [pure Python, no JAX]
     cur_token = token
5. Concatenate seed_token + decoded tokens → raw bytes → output file

Usage
-----
  uv run python examples/linear_rnn_srp_decompress.py \\
      --bundle log/linear_rnn_srp/ckpt_last \\
      --output recovered.bin \\
      [--verify datasets/original.txt]
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import os, sys, json, pickle, argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import linear_rnn_srp as _src
from linear_rnn_srp import (
    Config, RC_PREC, DTYPE, POWER_P,
    _quantize_cdf, _parse_stride_map,
    init_xlstm_params, init_randcompress_params,
    init_step_states, _collect_chunk_jit, _reshape_logits,
    bytes_to_tokens, tokens_to_bytes,
)

# Same jit(_quantize_cdf) as the encoder — no vmap, identical XLA, bit-exact.
_quantize_cdf_jit = jax.jit(_quantize_cdf)


# ============================================================================
# Bundle I/O
# ============================================================================

class _SrcUnpickler(pickle.Unpickler):
    """Redirect __main__.* lookups to linear_rnn_srp so params load correctly
    regardless of whether training was run as __main__ or as a module."""
    def find_class(self, module, name):
        if module == '__main__':
            return getattr(_src, name)
        return super().find_class(module, name)


def load_bundle(ckpt_dir):
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg_dict = json.load(f)
    with open(os.path.join(ckpt_dir, "params.pkl"), "rb") as f:
        params_np = _SrcUnpickler(f).load()
    with open(os.path.join(ckpt_dir, "rc_stream.bin"), "rb") as f:
        rc_raw = np.frombuffer(f.read(), dtype=np.uint8).copy()

    # Layer 4: verify rc_stream.bin integrity against stored SHA-256
    if "rc_stream_sha256" in meta:
        import hashlib
        actual_sha = hashlib.sha256(rc_raw.tobytes()).hexdigest()
        if actual_sha != meta["rc_stream_sha256"]:
            raise RuntimeError(
                f"rc_stream.bin is corrupt: SHA-256 mismatch.\n"
                f"  stored:  {meta['rc_stream_sha256']}\n"
                f"  actual:  {actual_sha}\n"
                "The compressed bundle may have been truncated or modified.")

    valid_keys = set(Config._fields)
    cfg_dict   = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    config     = Config(**cfg_dict)
    return meta, config, params_np, rc_raw


# ============================================================================
# Pure-Python range-coder decoder  (no JAX; model forward pass is JIT'd)
# ============================================================================

_M32  = 0xFFFFFFFF
_RC_M = 1 << RC_PREC
_kTop = _RC_M * 256   # flush threshold (LZMA-style: 1<<24)


def _rc_init(buf):
    """Read first 4 bytes into the (low_val, rng) range-coder state."""
    lv = 0
    for i in range(4):
        lv = ((lv << 8) | int(buf[i])) & _M32
    return lv, _RC_M * _RC_M - 1, 4   # low_val, rng, pos


def _rc_decode_sym(lv, rng, pos, buf, cumfreqs):
    """Decode one symbol using the (low_val, rng) range coder.

    Invariant: rng ∈ [M, M²).  rng_step = rng // M ≥ 1 always, so no
    interval collapse.  Returns (sym, new_lv, new_rng, new_pos).
    """
    rng_step   = rng >> RC_PREC               # rng // M, ≥ 1

    # Search on thresholds = rng_step * cf[i] — avoids division imprecision.
    # Cast to int64 BEFORE multiply — cumfreqs is int32, product overflows int32.
    thresholds = np.int64(rng_step) * cumfreqs[:-1].astype(np.int64)
    sym = int(np.searchsorted(thresholds, np.int64(lv), side='right')) - 1
    sym = max(0, min(sym, len(cumfreqs) - 2))

    cum_lo = int(cumfreqs[sym])
    freq   = int(cumfreqs[sym + 1]) - cum_lo

    lv    = (lv - rng_step * cum_lo) & _M32
    rng   = rng_step * freq

    while rng < _kTop:
        rng <<= 8
        byte = int(buf[pos]) if pos < len(buf) else 0
        lv   = ((lv << 8) | byte) & _M32
        pos += 1

    return sym, lv, rng, pos


# ============================================================================
# Main decompression
# ============================================================================

def decompress(ckpt_dir, output_path, verify_path=None):
    meta, config, params_np, rc_raw = load_bundle(ckpt_dir)

    n_raw_bytes = meta["n_raw_bytes"]
    T_valid     = meta["T_valid"]
    seed_token  = meta["seed_token"]
    rc_bytes    = meta["rc_bytes"]

    print(f"Bundle:  {ckpt_dir}")
    print(f"  n_raw={n_raw_bytes}  T_valid={T_valid}  seed={seed_token}  rc_bytes={rc_bytes}")

    # ── set global dtype/power so model functions use correct settings ────────
    _src.DTYPE   = jnp.dtype(config.dtype)
    _src.POWER_P = config.power_p

    key = jr.key(config.seed)
    _, k_xlstm, _ = jr.split(key, 3)

    print("Reconstructing frozen base model from seed...")
    base_xlstm = init_xlstm_params(k_xlstm, config)

    print("Restoring HiRA adapter params...")
    params = jax.tree_util.tree_map(jnp.array, params_np)

    # ── stride / chunk config ─────────────────────────────────────────────────
    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    chunk_size = config.segment_size * (8 // config.input_bits)

    # ── init range coder + model state ───────────────────────────────────────
    rc_lv, rc_rng, rc_pos = _rc_init(rc_raw)  # (low_val, rng, pos)
    model_states = init_step_states(config, batch_size=1)

    decoded   = [seed_token]
    cur_toks  = [seed_token]

    oh, ob = config.output_heads, config.output_bits
    pbar = tqdm(total=T_valid, desc="decode", unit="tok", file=sys.stderr)

    for t in range(T_valid):
        tok_in = jnp.array([cur_toks[-1]], jnp.int32)
        chunk_logits, model_states = _collect_chunk_jit(
            base_xlstm, params, tok_in, model_states,
            1, config.num_heads, config.block_map, stride_map, oh, ob)
        cumfreqs = np.array(_quantize_cdf_jit(chunk_logits[0]), dtype=np.int32)

        sym, rc_lv, rc_rng, rc_pos = _rc_decode_sym(
            rc_lv, rc_rng, rc_pos, rc_raw, cumfreqs)

        decoded.append(sym)
        cur_toks.append(sym)
        pbar.update(1)

    pbar.close()

    # ── convert tokens → bytes ────────────────────────────────────────────────
    raw_out = tokens_to_bytes(np.array(decoded, np.int32), config.input_bits)[:n_raw_bytes]
    print(f"Decoded {len(raw_out)} bytes  (expected {n_raw_bytes})")

    with open(output_path, "wb") as f:
        f.write(bytes(raw_out.tolist()))
    print(f"Written: {output_path}")

    # ── optional verification ─────────────────────────────────────────────────
    if verify_path is not None:
        with open(verify_path, "rb") as f:
            original = np.frombuffer(f.read(), dtype=np.uint8)
        min_len = min(len(raw_out), len(original))
        wrong   = int(np.sum(raw_out[:min_len] != original[:min_len]))
        if wrong == 0 and len(raw_out) == len(original):
            print("Verification: PERFECT MATCH")
        else:
            wrongs = np.where(raw_out[:min_len] != original[:min_len])[0]
            first  = int(wrongs[0]) if len(wrongs) else None
            print(f"Verification: {wrong} bytes wrong  "
                  f"(decoded={len(raw_out)}  original={len(original)})")
            if first is not None:
                print(f"  first wrong byte at position {first}")

    return raw_out


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decompress a linear_rnn_srp compressed bundle")
    parser.add_argument("--bundle",  required=True,
                        help="Compressed bundle directory (ckpt_dir)")
    parser.add_argument("--output",  required=True,
                        help="Output file path for recovered bytes")
    parser.add_argument("--verify",  default=None,
                        help="Original file to check bit-exact recovery")
    args = parser.parse_args()
    decompress(args.bundle, args.output, args.verify)


if __name__ == "__main__":
    main()
