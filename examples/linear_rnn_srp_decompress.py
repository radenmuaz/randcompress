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

_M32 = 0xFFFFFFFF
_RC_M = 1 << RC_PREC


def _rc_init(buf):
    """Read first 4 bytes into the range-coder state."""
    x = 0
    for i in range(4):
        x = ((x << 8) | int(buf[i])) & _M32
    return 0, _M32, x, 4   # low, high, x, pos


def _rc_decode_sym(low, high, x, pos, buf, cumfreqs):
    """Decode one symbol; return (sym, new_low, new_high, new_x, new_pos).

    Layer-1 defensive checks (Python assertions, active in debug mode):
      x_range  — x must be in [low, high]; violation means RC state is corrupt
      slot_oor — slot must be < M; violation means arithmetic overflow
    These fire before the symbol is returned so the caller gets a clear error
    rather than a silent wrong decode.
    """
    # x ∈ [low, high] invariant — catches RC state corruption / arithmetic overflow
    if low > x or x > high:
        raise RuntimeError(
            f"RC state corrupt at buf_pos={pos}: "
            f"x={x:#010x} not in [{low:#010x}, {high:#010x}]. "
            "Likely cause: CDF mismatch between encoder and decoder paths.")

    rng  = high - low + 1
    slot = ((x - low + 1) * _RC_M - 1) // rng

    if slot >= _RC_M:
        raise RuntimeError(
            f"RC slot overflow at buf_pos={pos}: slot={slot} >= M={_RC_M}. "
            "Arithmetic overflow in range-coder state.")

    sym  = int(np.searchsorted(cumfreqs[:-1], slot, side='right')) - 1
    sym  = max(0, min(sym, len(cumfreqs) - 2))

    cum_lo = int(cumfreqs[sym])
    cum_hi = int(cumfreqs[sym + 1])
    high   = (low + rng * cum_hi // _RC_M - 1) & _M32
    low    = (low + rng * cum_lo // _RC_M)      & _M32

    # refill agreed bytes
    while (low >> 24) == (high >> 24):
        low   = (low  << 8) & _M32
        high  = ((high << 8) | 0xFF) & _M32
        byte  = int(buf[pos]) if pos < len(buf) else 0
        x     = ((x << 8) | byte) & _M32
        pos  += 1

    return sym, low, high, x, pos


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
    rc_low, rc_high, rc_x, rc_pos = _rc_init(rc_raw)
    model_states = init_step_states(config, batch_size=1)

    # ── AR decode loop
    # Uses _collect_chunk_jit(size=1) — same scan body and same weight-hoisting
    # as the encoder's collect_probs, guaranteeing bit-exact logits.
    # CDFs computed via jax.vmap(_quantize_cdf) matching the encoder's batch path.
    decoded   = [seed_token]
    cur_toks  = [seed_token]

    oh, ob = config.output_heads, config.output_bits
    pbar = tqdm(total=T_valid, desc="decode", unit="tok", file=sys.stderr)

    for t in range(T_valid):
        tok_in = jnp.array([cur_toks[-1]], jnp.int32)
        chunk_logits, model_states = _collect_chunk_jit(
            base_xlstm, params, tok_in, model_states,
            1, config.num_heads, config.block_map, stride_map, oh, ob)
        # chunk_logits: [1, V] — match encoder's vmap CDF path exactly
        cumfreqs = np.array(_quantize_cdf_jit(chunk_logits[0]), dtype=np.int32)

        sym, rc_low, rc_high, rc_x, rc_pos = _rc_decode_sym(
            rc_low, rc_high, rc_x, rc_pos, rc_raw, cumfreqs)

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
