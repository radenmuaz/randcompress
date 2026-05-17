"""
Compressor for linear_rnn_srp — takes trained params + input file, writes compressed bundle.

Usage
-----
  uv run python examples/linear_rnn_srp_compress.py \\
      --params  log/juz1/ckpt_last \\
      --input   datasets/juz1.txt \\
      --output  /tmp/juz1_compressed

  # Different RC scheme:
  uv run python examples/linear_rnn_srp_compress.py \\
      --params  log/juz1/ckpt_last \\
      --input   datasets/juz1.txt \\
      --output  /tmp/juz1_u32 \\
      --scheme  canonical-uint32

Schemes:  canonical-uint64 (default) | canonical-uint32 | canonical-uint16
  canonical-uint64 — JAX JIT, 64-bit state, RC_PREC=16  (fastest, most accurate)
  canonical-uint32 — numpy, 32-bit state, RC_PREC=16    (slower, same precision)
  canonical-uint16 — numpy, 16-bit state, RC_PREC=8     (slow, lossy CDF quantization)

Bundle layout written to --output:
  config.json      — architecture + seed (copied from --params)
  params.pkl       — HiRA adapter weights (copied from --params)
  rc_stream.bin    — range-coded byte stream for --input
  meta.json        — {n_raw_bytes, T_valid, seed_token, rc_bytes, rc_scheme, ...}
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import os, sys, json, pickle, argparse, hashlib, time
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import linear_rnn_srp as _src
from linear_rnn_srp import (
    Config, DTYPE, POWER_P,
    _quantize_cdf, _cdf_ok, _parse_stride_map,
    init_xlstm_params,
    init_step_states, _collect_chunk_jit,
    bytes_to_tokens, make_chunks_and_targets,
    rc_encode_c, rc_decode_c,
    load_dataset,
    _RC_SCHEMES,
)

_quantize_cdf_jit = jax.jit(_quantize_cdf)
_cdf_ok_jit       = jax.jit(_cdf_ok)

VALID_SCHEMES = ['canonical-uint64', 'canonical-c'] + list(_RC_SCHEMES.keys())


# ============================================================================
# Bundle I/O
# ============================================================================

class _SrcUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            return getattr(_src, name)
        return super().find_class(module, name)


def load_params_bundle(ckpt_dir):
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg_dict = json.load(f)
    with open(os.path.join(ckpt_dir, "params.pkl"), "rb") as f:
        params_np = _SrcUnpickler(f).load()
    valid_keys = set(Config._fields)
    cfg_dict   = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    return Config(**cfg_dict), params_np


# ============================================================================
# Compress
# ============================================================================

def compress(params_dir, input_path, output_dir, scheme='canonical-uint64'):
    t_total = time.perf_counter()

    print(f"Params:  {params_dir}")
    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Scheme:  {scheme}")

    config, params_np = load_params_bundle(params_dir)
    _src.DTYPE   = jnp.dtype(config.dtype)
    _src.POWER_P = config.power_p

    key = jr.key(config.seed)
    _, k_xlstm, _ = jr.split(key, 3)

    print("Reconstructing frozen base model...")
    base_xlstm = init_xlstm_params(k_xlstm, config)
    print("Restoring HiRA adapter params...")
    params = jax.tree_util.tree_map(jnp.array, params_np)

    raw_bytes = load_dataset(input_path)
    n_raw     = len(raw_bytes)
    print(f"Input:   {n_raw} bytes")

    all_inputs, all_targets = make_chunks_and_targets(raw_bytes, config)
    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    oh, ob     = config.output_heads, config.output_bits
    V          = 1 << ob

    # ── collect teacher-forced logits ─────────────────────────────────────────
    t0 = time.perf_counter()
    num_chunks = all_inputs.shape[0]
    chunk_size = all_inputs.shape[1]
    states     = init_step_states(config, batch_size=1)

    logits_list, tgts_list = [], []
    pbar = tqdm(total=num_chunks * chunk_size, desc="collect logits",
                unit="tok", file=sys.stderr)
    for ci in range(num_chunks):
        for pos in range(chunk_size):
            tok = jnp.array([int(all_inputs[ci, pos])], jnp.int32)
            lf, states = _collect_chunk_jit(
                base_xlstm, params, tok, states,
                1, config.num_heads, config.block_map, stride_map, oh, ob)
            logits_list.append(np.array(lf[0], np.float32))
            tgts_list.append(int(all_targets[ci, pos, 0]))
            pbar.update(1)
    pbar.close()
    t_logits = time.perf_counter() - t0

    logits_np = np.stack(logits_list)
    tgts_np   = np.array(tgts_list, np.int32)
    valid     = tgts_np >= 0
    vlog, vtgt = logits_np[valid], tgts_np[valid]
    T_v       = int(valid.sum())

    # ── CE + accuracy ─────────────────────────────────────────────────────────
    lp       = jax.nn.log_softmax(jnp.array(vlog), axis=-1)
    tgt_lp   = np.array(lp)[np.arange(T_v), vtgt]
    ce_bits  = float(-np.sum(tgt_lp) / np.log(2))
    ce_bpb   = ce_bits / n_raw
    preds    = np.argmax(vlog, axis=-1)
    n_wrong  = int(np.sum(preds != vtgt))
    acc      = (T_v - n_wrong) / max(T_v, 1)

    # ── quantize CDFs ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    cdf_all_ok  = True
    n_bad_cdf   = 0
    cumfreqs_np = np.empty((T_v, V + 1), dtype=np.int32)
    pbar_cdf = tqdm(total=T_v, desc="quantize CDFs", unit="tok",
                    leave=False, file=sys.stderr)
    for i in range(T_v):
        cf = _quantize_cdf_jit(jnp.array(vlog[i]))
        cumfreqs_np[i] = np.array(cf, np.int32)
        if not bool(_cdf_ok_jit(cf)):
            cdf_all_ok = False; n_bad_cdf += 1
        pbar_cdf.update(1)
    pbar_cdf.close()
    if not cdf_all_ok:
        print(f"  [WARN] {n_bad_cdf}/{T_v} CDFs invalid")
    t_cdf = time.perf_counter() - t0

    # ── range encode ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"  encode ({scheme})...", end=" ", flush=True)
    rc_np    = rc_encode_c(vtgt, cumfreqs_np, scheme=scheme)
    rc_bytes = len(rc_np)
    t_enc    = time.perf_counter() - t0
    print(f"{rc_bytes}B  {t_enc:.2f}s  ({rc_bytes/t_enc/1e3:.1f} kB/s)")

    # ── stream size sanity ────────────────────────────────────────────────────
    ce_bytes_exp = ce_bits / 8
    if ce_bytes_exp > 10 and abs(rc_bytes - ce_bytes_exp) / ce_bytes_exp > 0.15:
        print(f"  [WARN] RC anomaly: expected ~{ce_bytes_exp:.0f}B got {rc_bytes}B")

    # ── verify round-trip ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print(f"  verify...", end=" ", flush=True)
    decoded   = rc_decode_c(rc_np, cumfreqs_np, scheme=scheme)
    decode_ok = bool(np.all(decoded == vtgt))
    t_dec     = time.perf_counter() - t0
    print(f"{'OK' if decode_ok else 'FAIL'}  {t_dec:.2f}s")
    if not decode_ok:
        raise RuntimeError("RC round-trip verification failed — bundle not saved.")

    # ── write bundle ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "rc_stream.bin"), "wb") as f:
        f.write(bytes(rc_np.tolist()))

    import shutil
    for fname in ("params.pkl", "config.json"):
        src = os.path.join(params_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

    seed_token       = int(all_inputs[0][0])
    rc_sha256        = hashlib.sha256(rc_np.tobytes()).hexdigest()
    param_bytes      = sum(
        np.prod(p.shape) * np.dtype(_src.DTYPE).itemsize
        for p in jax.tree_util.tree_leaves(params))
    param_disk_bytes = os.path.getsize(os.path.join(output_dir, "params.pkl"))
    meta = dict(
        n_raw_bytes      = int(n_raw),
        T_valid          = int(T_v),
        seed_token       = int(seed_token),
        rc_bytes         = int(rc_bytes),
        rc_stream_sha256 = rc_sha256,
        rc_scheme        = scheme,
        param_bytes      = int(param_bytes),
        param_disk_bytes = int(param_disk_bytes),
        input_bits       = int(config.input_bits),
        output_bits      = int(config.output_bits),
        output_heads     = int(config.output_heads),
    )
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    t_wall    = time.perf_counter() - t_total
    tot_bytes = param_bytes + rc_bytes
    ratio     = n_raw / tot_bytes if tot_bytes > 0 else float('inf')

    print(f"\n[compress]  total={t_wall:.1f}s"
          f"  (logits={t_logits:.1f}s  cdfs={t_cdf:.1f}s"
          f"  enc={t_enc:.2f}s  dec={t_dec:.2f}s)")
    print(f"  argmax: {T_v-n_wrong}/{T_v} correct ({acc:.1%})")
    print(f"  CE={ce_bpb:.4f}bpb  rc={rc_bytes*8/n_raw:.4f}bpb({rc_bytes}B)"
          f"  p+rc={ratio:.3f}x({tot_bytes}B)")
    print(f"Bundle: {output_dir}")
    return meta


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compress a file using a trained linear_rnn_srp bundle")
    parser.add_argument("--params",  required=True,
                        help="Source bundle dir with params.pkl + config.json")
    parser.add_argument("--input",   required=True,
                        help="Input file to compress")
    parser.add_argument("--output",  required=True,
                        help="Output bundle directory")
    parser.add_argument("--scheme",  default="canonical-uint64",
                        choices=VALID_SCHEMES,
                        help="RC scheme: canonical-uint64 (default) | canonical-uint32 | canonical-uint16")
    args = parser.parse_args()
    compress(args.params, args.input, args.output, scheme=args.scheme)


if __name__ == "__main__":
    main()
