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
     logits = forward_step(model, cur_token, state)   [JAX JIT]
     cdf    = quantize_cdf(logits[head=0])             [JAX JIT]
     token  = canonical (low,high) 64-bit RC step      [pure Python]
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
import examples.old.v1.linear_rnn_srp as _src
from examples.old.v1.linear_rnn_srp import (
    Config, RC_PREC, DTYPE, POWER_P,
    _quantize_cdf, _parse_stride_map,
    init_xlstm_params,
    init_step_states, _collect_chunk_jit,
    bytes_to_tokens, tokens_to_bytes,
)

_quantize_cdf_jit = jax.jit(_quantize_cdf)

_FULL64 = 0xFFFFFFFFFFFFFFFF
_M      = 1 << RC_PREC  # 65536


# ============================================================================
# Bundle I/O
# ============================================================================

class _SrcUnpickler(pickle.Unpickler):
    """Redirect __main__.* lookups to linear_rnn_srp."""
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

    _src.DTYPE   = jnp.dtype(config.dtype)
    _src.POWER_P = config.power_p

    key = jr.key(config.seed)
    _, k_xlstm, _ = jr.split(key, 3)

    print("Reconstructing frozen base model from seed...")
    base_xlstm = init_xlstm_params(k_xlstm, config)

    print("Restoring HiRA adapter params...")
    params = jax.tree_util.tree_map(jnp.array, params_np)

    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    oh, ob     = config.output_heads, config.output_bits
    V          = 1 << ob

    # ── init canonical (low, high) 64-bit range coder ────────────────────────
    # Matches rc_codec.c exactly: 64-bit state, emit/refill when top bytes agree.
    rc_low  = 0
    rc_high = _FULL64
    rc_code = 0
    rc_pos  = 0
    for _ in range(8):
        byte    = int(rc_raw[rc_pos]) if rc_pos < len(rc_raw) else 0
        rc_code = ((rc_code << 8) | byte) & _FULL64
        rc_pos += 1

    model_states = init_step_states(config, batch_size=1)
    decoded      = [seed_token]
    cur_tok      = jnp.array([seed_token], jnp.int32)

    print("AR decoding (JAX JIT forward + Python RC step)...")
    pbar = tqdm(total=T_valid, desc="decode", unit="tok", file=sys.stderr)

    for t in range(T_valid):
        # model forward: one token → logits
        chunk_logits, model_states = _collect_chunk_jit(
            base_xlstm, params, cur_tok, model_states,
            1, config.num_heads, config.block_map, stride_map, oh, ob)
        cf = np.array(_quantize_cdf_jit(chunk_logits[0]), dtype=np.int32)

        # RC decode step: find sym where rc_low + scale(range, cf[sym]) <= rc_code.
        # Uses same boundary formula as rc_codec.c to avoid rounding mismatches.
        rc_range = rc_high - rc_low + 1   # Python int, up to 2^64
        lo_s, hi_s = 0, V - 1
        while lo_s < hi_s:
            mid = (lo_s + hi_s + 1) // 2
            # scale(range, cf[mid]) = range * cf[mid] // M
            if rc_low + rc_range * int(cf[mid]) // _M <= rc_code:
                lo_s = mid
            else:
                hi_s = mid - 1
        sym = lo_s

        # narrow interval (same formula as rc_codec.c: scale = range*f//M)
        cum_lo  = int(cf[sym])
        cum_hi  = int(cf[sym + 1])
        rc_high = rc_low + rc_range * cum_hi // _M - 1
        rc_low  = rc_low + rc_range * cum_lo // _M

        # refill agreed top bytes (Python int arithmetic, no masking needed until emit)
        while (rc_low >> 56) == (rc_high >> 56):
            rc_low  = rc_low  << 8
            rc_high = (rc_high << 8) | 0xFF
            byte    = int(rc_raw[rc_pos]) if rc_pos < len(rc_raw) else 0
            rc_code = (rc_code << 8) | byte
            rc_pos += 1

        decoded.append(sym)
        cur_tok = jnp.array([sym], jnp.int32)
        pbar.update(1)

    pbar.close()

    # ── convert tokens → bytes ────────────────────────────────────────────────
    raw_out = tokens_to_bytes(
        np.array(decoded, dtype=np.int32), config.input_bits)[:n_raw_bytes]
    print(f"Decoded {len(raw_out)} bytes  (expected {n_raw_bytes})")

    with open(output_path, "wb") as f:
        f.write(bytes(raw_out.tolist()))
    print(f"Written: {output_path}")

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
