"""
Diagnostic: compare teacher-forced vs AR model CDFs.
Finds the first position where CDFs differ (or confirms they match).
Also tests if the Python range decoder can decode the stream using saved CDFs.

Usage:
  uv run python examples/diag_cdf_match.py \
      --bundle log/linear_rnn_srp/ckpt_last \
      --original datasets/juz1.txt \
      --npos 400
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
    Config, RC_PREC, _quantize_cdf, _parse_stride_map,
    init_xlstm_params, init_step_states, _collect_chunk_jit,
    bytes_to_tokens, rc_decode_init,
)
from linear_rnn_srp_decompress import load_bundle, _rc_init, _rc_decode_sym

_RC_M = 1 << RC_PREC
_quantize_cdf_one = jax.jit(lambda x: jax.vmap(_quantize_cdf)(x)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle",   required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--npos",     type=int, default=400)
    args = ap.parse_args()

    meta, config, params_np, rc_raw = load_bundle(args.bundle)
    _src.DTYPE   = jnp.dtype(config.dtype)
    _src.POWER_P = config.power_p

    key = jr.key(config.seed)
    _, k_xlstm, _ = jr.split(key, 3)
    base_xlstm = init_xlstm_params(k_xlstm, config)
    params     = jax.tree_util.tree_map(jnp.array, params_np)

    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    chunk_size = config.segment_size * (8 // config.input_bits)
    oh, ob     = config.output_heads, config.output_bits
    seed_token = meta["seed_token"]
    npos       = min(args.npos, meta["T_valid"])

    # Load original file → tokens
    with open(args.original, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    tokens = np.array(bytes_to_tokens(raw, config.input_bits), np.int32)

    print(f"Comparing first {npos} positions  (seed={seed_token}, chunk={chunk_size})")

    # ── Teacher-forced CDFs (same as encoder's collect_probs) ─────────────────
    print("Computing teacher-forced CDFs...")
    tf_states = init_step_states(config, batch_size=1)
    tf_cdfs   = []
    for t in tqdm(range(npos), desc="teacher-forced", file=sys.stderr):
        tok_in = jnp.array([int(tokens[t])], jnp.int32)
        chunk_logits, tf_states = _collect_chunk_jit(
            base_xlstm, params, tok_in, tf_states,
            1, config.num_heads, config.block_map, stride_map, oh, ob)
        tf_cdfs.append(np.array(_quantize_cdf_one(chunk_logits), np.int32))

    # ── AR CDFs (same as decompressor) ────────────────────────────────────────
    print("Computing AR CDFs...")
    ar_states = init_step_states(config, batch_size=1)
    ar_cdfs   = []
    cur_toks  = [seed_token]
    rc_low, rc_high, rc_x, rc_pos = _rc_init(rc_raw)

    for t in tqdm(range(npos), desc="ar", file=sys.stderr):
        tok_in = jnp.array([cur_toks[-1]], jnp.int32)
        chunk_logits, ar_states = _collect_chunk_jit(
            base_xlstm, params, tok_in, ar_states,
            1, config.num_heads, config.block_map, stride_map, oh, ob)
        cdf = np.array(_quantize_cdf_one(chunk_logits), np.int32)
        ar_cdfs.append(cdf)

        sym, rc_low, rc_high, rc_x, rc_pos = _rc_decode_sym(
            rc_low, rc_high, rc_x, rc_pos, rc_raw, cdf)
        cur_toks.append(sym)

    # ── Compare TF vs AR CDFs ─────────────────────────────────────────────────
    print("\n=== Teacher-forced vs AR CDF comparison ===")
    first_cdf_mismatch = None
    for t in range(npos):
        if not np.array_equal(tf_cdfs[t], ar_cdfs[t]):
            diff = tf_cdfs[t].astype(np.int64) - ar_cdfs[t].astype(np.int64)
            print(f"  MISMATCH at t={t}: max_diff={np.abs(diff).max()}  "
                  f"n_diff={np.sum(diff!=0)}  "
                  f"tf_cdf[tok]={tf_cdfs[t][int(tokens[t+1])]}  "
                  f"ar_cdf[tok]={ar_cdfs[t][int(tokens[t+1])]}")
            first_cdf_mismatch = t
            break
    if first_cdf_mismatch is None:
        print(f"  All {npos} CDFs match exactly between TF and AR paths.")

    # ── Test Python range decoder with TEACHER-FORCED CDFs + carry trace ─────
    print("\n=== Python range decoder with teacher-forced CDFs ===")
    _M32 = 0xFFFFFFFF
    rc_low2, rc_high2, rc_x2, rc_pos2 = _rc_init(rc_raw)
    decoded_tf = [seed_token]
    first_wrong_t = None
    for t in range(npos):
        # save carry before this step
        pre_low, pre_high, pre_x, pre_pos = rc_low2, rc_high2, rc_x2, rc_pos2

        rng  = pre_high - pre_low + 1
        slot = ((pre_x - pre_low + 1) * _RC_M - 1) // rng
        sym, rc_low2, rc_high2, rc_x2, rc_pos2 = _rc_decode_sym(
            pre_low, pre_high, pre_x, pre_pos, rc_raw, tf_cdfs[t])
        decoded_tf.append(sym)

        if first_wrong_t is None and sym != int(tokens[t+1]):
            first_wrong_t = t
            print(f"  First wrong at step t={t}: decoded={sym} expected={tokens[t+1]}")
            print(f"    pre-carry: low={pre_low:#010x} high={pre_high:#010x} "
                  f"x={pre_x:#010x} pos={pre_pos}")
            print(f"    rng={rng:#x}  slot={slot}")
            print(f"    tf_cdfs[t][{sym}]={tf_cdfs[t][sym]}  "
                  f"tf_cdfs[t][{sym+1}]={tf_cdfs[t][sym+1]}")
            print(f"    tf_cdfs[t][{tokens[t+1]}]={tf_cdfs[t][tokens[t+1]]}  "
                  f"tf_cdfs[t][{tokens[t+1]+1}]={tf_cdfs[t][tokens[t+1]+1]}")
            # also show what the JAX formula would give for slot
            import jax.numpy as jnp2
            _U32 = jnp.uint32; _U64 = jnp.uint64
            jax_rng  = _U64(pre_high) - _U64(pre_low) + _U64(1)
            jax_xlp1 = _U32(pre_x) - _U32(pre_low) + _U32(1)
            jax_slot = (_U64(jax_xlp1) * jnp.uint64(1 << RC_PREC) - _U64(1)) // jax_rng
            print(f"    JAX: jax_xlp1={int(jax_xlp1):#x}  jax_slot={int(jax_slot)}")
            print(f"    Python: xlp1={pre_x - pre_low + 1:#x}  slot={slot}")

    wrong_tf = sum(1 for t in range(1, npos+1) if decoded_tf[t] != int(tokens[t]))
    print(f"  Wrong symbols with TF CDFs: {wrong_tf}/{npos}")

    # ── Test Python range decoder with AR CDFs ────────────────────────────────
    print("\n=== Python range decoder with AR CDFs ===")
    wrong_ar = sum(1 for t in range(1, npos+1) if cur_toks[t] != int(tokens[t]))
    print(f"  Wrong symbols with AR CDFs: {wrong_ar}/{npos}")
    if wrong_ar > 0:
        for t in range(1, npos+1):
            if cur_toks[t] != int(tokens[t]):
                print(f"  First wrong at t={t}: decoded={cur_toks[t]} expected={tokens[t]}")
                break


if __name__ == "__main__":
    main()
