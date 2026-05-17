"""
Validate all RC schemes on a trained bundle.
Compresses the original file with each scheme, decompresses, verifies, records timings.
Saves results to --output_dir as JSON + per-scheme bundles.

Usage:
  uv run python examples/validate_rc_schemes.py \\
      --bundle  log/v1_juz1/ckpt_last \\
      --input   datasets/juz1.txt \\
      --output_dir log/v1_juz1/rc_validation
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import os, sys, json, time, argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from linear_rnn_rc_v1 import (
    Config, _load_bundle, init_xlstm_params, init_step_states,
    _collect_chunk_jit, _parse_stride_map, make_chunks_and_targets,
    tokens_to_bytes, _quantize_cdf, rc_encode, rc_decode,
    _decompress_step_init, _decompress_step,
    DTYPE, POWER_P, _RC_SCHEMES,
)
import linear_rnn_rc_v1 as _m

SCHEMES = ['canonical-uint64', 'canonical-c', 'canonical-uint32', 'canonical-uint16']

_qcdf_jit = jax.jit(_quantize_cdf)


def collect_logits_and_cdfs(base_xlstm, params, config, all_inputs, all_targets):
    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    num_chunks = all_inputs.shape[0]; chunk_size = all_inputs.shape[1]
    oh, ob = config.output_heads, config.output_bits; V = 1 << ob
    states = init_step_states(config, batch_size=1)
    logits_list = []; tgts_list = []
    pbar = tqdm(total=num_chunks*chunk_size, desc="collect logits", unit="tok", file=sys.stderr)
    for ci in range(num_chunks):
        for pos in range(chunk_size):
            tok = jnp.array([int(all_inputs[ci, pos])], jnp.int32)
            lf, states = _collect_chunk_jit(base_xlstm, params, tok, states,
                1, config.num_heads, config.block_map, stride_map, oh, ob)
            logits_list.append(np.array(lf[0], np.float32))
            tgts_list.append(int(all_targets[ci, pos, 0]))
            pbar.update(1)
    pbar.close()
    logits_np = np.stack(logits_list); tgts_np = np.array(tgts_list, np.int32)
    valid = tgts_np >= 0; vlog = logits_np[valid]; vtgt = tgts_np[valid]; T_v = int(valid.sum())
    lp = jax.nn.log_softmax(jnp.array(vlog), axis=-1)
    ce_bits = float(-np.sum(np.array(lp)[np.arange(T_v), vtgt]) / np.log(2))
    n_wrong = int(np.sum(np.argmax(vlog, axis=-1) != vtgt))
    cumfreqs = np.empty((T_v, V+1), dtype=np.int32)
    pbar2 = tqdm(total=T_v, desc="quantize CDFs", unit="tok", leave=False, file=sys.stderr)
    for i in range(T_v):
        cumfreqs[i] = np.array(_qcdf_jit(jnp.array(vlog[i])), np.int32); pbar2.update(1)
    pbar2.close()
    return vtgt, cumfreqs, ce_bits, n_wrong, T_v


def validate(bundle_dir, input_path, output_dir, ar_schemes=None):
    import shutil, hashlib

    os.makedirs(output_dir, exist_ok=True)
    meta, config, params_np, rc_raw_orig = _load_bundle(bundle_dir)
    n_raw = meta["n_raw_bytes"]

    _m.DTYPE = jnp.dtype(config.dtype); _m.POWER_P = config.power_p
    key = jr.key(config.seed); _, k_xlstm, _ = jr.split(key, 3)
    print("Reconstructing base model...")
    base_xlstm = init_xlstm_params(k_xlstm, config)
    print("Restoring params...")
    params = jax.tree_util.tree_map(jnp.array, params_np)

    raw_bytes = np.frombuffer(open(input_path, 'rb').read(), dtype=np.uint8).copy()
    all_inputs, all_targets = make_chunks_and_targets(raw_bytes, config)

    print("Collecting logits and CDFs (once, shared across all schemes)...")
    t0 = time.perf_counter()
    vtgt, cumfreqs, ce_bits, n_wrong, T_v = collect_logits_and_cdfs(
        base_xlstm, params, config, all_inputs, all_targets)
    t_collect = time.perf_counter() - t0
    ce_bpb = ce_bits / n_raw; acc = (T_v - n_wrong) / max(T_v, 1)
    print(f"Collected {T_v} tokens in {t_collect:.1f}s  CE={ce_bpb:.4f}bpb  argmax={acc:.1%}")

    results = {}

    for scheme in SCHEMES:
        print(f"\n{'='*60}")
        print(f"Scheme: {scheme}")
        print(f"{'='*60}")

        # ── encode ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            rc_np = rc_encode(vtgt, cumfreqs, scheme=scheme)
            t_enc = time.perf_counter() - t0
            rc_bytes = len(rc_np)
            enc_ok = True
            enc_err = None
        except Exception as e:
            t_enc = time.perf_counter() - t0
            rc_np = None; rc_bytes = 0; enc_ok = False; enc_err = str(e)
            print(f"  encode FAILED: {e}")

        # ── decode (verify) ────────────────────────────────────────────────
        if enc_ok:
            t0 = time.perf_counter()
            try:
                decoded = rc_decode(rc_np, cumfreqs, scheme=scheme)
                t_dec = time.perf_counter() - t0
                round_trip_ok = bool(np.all(decoded == vtgt))
                dec_err = None
            except Exception as e:
                t_dec = time.perf_counter() - t0
                round_trip_ok = False; dec_err = str(e)
                print(f"  decode FAILED: {e}")
        else:
            t_dec = 0.0; round_trip_ok = False; dec_err = enc_err

        # ── AR decompress (full end-to-end) ───────────────────────────────
        do_ar = (ar_schemes is None) or (scheme in ar_schemes)
        if enc_ok and round_trip_ok and do_ar:
            print(f"  AR decompress...", end=" ", flush=True)
            stride_map = _parse_stride_map(config.stride_map, config.num_layers)
            oh, ob = config.output_heads, config.output_bits; V = 1 << ob
            rc_low, rc_high, rc_code, rc_pos = _decompress_step_init(rc_np, scheme=scheme)
            model_states = init_step_states(config, batch_size=1)  # fresh state per scheme
            decoded_full = [meta["seed_token"]]
            cur_tok = jnp.array([meta["seed_token"]], jnp.int32)
            t0_ar = time.perf_counter()
            pbar = tqdm(total=meta["T_valid"], desc="decode", unit="tok", leave=False, file=sys.stderr)
            for _ in range(meta["T_valid"]):
                chunk_logits, model_states = _collect_chunk_jit(base_xlstm, params, cur_tok,
                    model_states, 1, config.num_heads, config.block_map, stride_map, oh, ob)
                cf = np.array(_qcdf_jit(chunk_logits[0]), dtype=np.int32)
                sym, rc_low, rc_high, rc_code, rc_pos = _decompress_step(
                    rc_low, rc_high, rc_code, rc_pos, rc_np, cf, V, scheme=scheme)
                decoded_full.append(sym); cur_tok = jnp.array([sym], jnp.int32); pbar.update(1)
            pbar.close()
            t_ar = time.perf_counter() - t0_ar
            raw_out = tokens_to_bytes(np.array(decoded_full, np.int32), config.input_bits)[:n_raw]
            original = raw_bytes
            ar_ok = bool(len(raw_out)==n_raw and np.all(raw_out==original))
            print(f"{'PERFECT MATCH' if ar_ok else 'MISMATCH'}  {t_ar:.1f}s")
        elif not do_ar:
            t_ar = 0.0; ar_ok = None   # skipped
        else:
            t_ar = 0.0; ar_ok = False

        # ── save bundle ────────────────────────────────────────────────────
        scheme_dir = os.path.join(output_dir, scheme.replace('-', '_'))
        if enc_ok and round_trip_ok:
            os.makedirs(scheme_dir, exist_ok=True)
            with open(os.path.join(scheme_dir, "rc_stream.bin"), "wb") as f:
                f.write(bytes(rc_np.tolist()))
            for fname in ("params.pkl", "config.json"):
                shutil.copy2(os.path.join(bundle_dir, fname), os.path.join(scheme_dir, fname))
            scheme_meta = dict(n_raw_bytes=int(n_raw), T_valid=int(T_v),
                seed_token=int(all_inputs[0][0]), rc_bytes=int(rc_bytes),
                rc_stream_sha256=hashlib.sha256(rc_np.tobytes()).hexdigest(),
                rc_scheme=scheme, input_bits=int(config.input_bits),
                output_bits=int(config.output_bits), output_heads=int(config.output_heads))
            with open(os.path.join(scheme_dir, "meta.json"), "w") as f:
                json.dump(scheme_meta, f, indent=2)

        param_bytes = meta.get("param_bytes", 0)
        tot_bytes = param_bytes + rc_bytes if enc_ok else 0
        ratio = n_raw / tot_bytes if tot_bytes > 0 else 0.0
        bpb = rc_bytes * 8 / n_raw if enc_ok else 0.0

        r = dict(
            scheme       = scheme,
            enc_ok       = enc_ok,
            round_trip_ok= round_trip_ok,
            ar_ok        = ar_ok,
            rc_bytes     = rc_bytes,
            rc_bpb       = round(bpb, 4),
            ce_bpb       = round(ce_bpb, 4),
            ratio        = round(ratio, 4),
            t_enc_s      = round(t_enc, 3),
            t_dec_s      = round(t_dec, 3),
            t_ar_s       = round(t_ar, 1),
            enc_kBps     = round(rc_bytes / t_enc / 1e3, 1) if t_enc > 0 and enc_ok else 0,
            dec_kBps     = round(rc_bytes / t_dec / 1e3, 1) if t_dec > 0 and enc_ok else 0,
            error        = enc_err or dec_err,
            bundle_dir   = scheme_dir if enc_ok and round_trip_ok else None,
        )
        results[scheme] = r

        print(f"  bytes={rc_bytes}  {bpb:.4f}bpb  ratio={ratio:.3f}x"
              f"  enc={t_enc:.3f}s({r['enc_kBps']}kB/s)"
              f"  dec={t_dec:.3f}s({r['dec_kBps']}kB/s)"
              f"  round_trip={'OK' if round_trip_ok else 'FAIL'}"
              f"  ar={'OK' if ar_ok else 'FAIL'}")

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY  input={input_path}  n_raw={n_raw}B  CE={ce_bpb:.4f}bpb  collect={t_collect:.1f}s")
    print(f"{'='*80}")
    print(f"{'scheme':<22} {'bytes':>7} {'bpb':>7} {'ratio':>7} {'enc kB/s':>10} {'dec kB/s':>10} {'rt':>4} {'ar':>4}")
    print(f"{'-'*80}")
    for s, r in results.items():
        ok = 'OK' if r['round_trip_ok'] else 'FAIL'
        ar = 'OK' if r['ar_ok'] else ('SKIP' if r['ar_ok'] is None else 'FAIL')
        print(f"{s:<22} {r['rc_bytes']:>7} {r['rc_bpb']:>7.4f} {r['ratio']:>7.3f}x"
              f" {r['enc_kBps']:>10.1f} {r['dec_kBps']:>10.1f} {ok:>4} {ar:>4}")

    # ── save results JSON ─────────────────────────────────────────────────────
    out_json = os.path.join(output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({"input": input_path, "n_raw": n_raw, "ce_bpb": ce_bpb,
                   "t_collect_s": round(t_collect, 1), "schemes": results}, f, indent=2)
    print(f"\nResults: {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate all RC schemes on a trained bundle")
    parser.add_argument("--bundle",     required=True, help="Trained bundle dir (params.pkl + config.json)")
    parser.add_argument("--input",      required=True, help="Original input file")
    parser.add_argument("--output_dir", required=True, help="Where to save per-scheme bundles and results.json")
    parser.add_argument("--ar_schemes", default=None,
                        help="Comma-separated schemes to run full AR decompress (default: all). "
                             "E.g. canonical-uint64,canonical-c")
    args = parser.parse_args()
    ar_schemes = args.ar_schemes.split(',') if args.ar_schemes else None
    validate(args.bundle, args.input, args.output_dir, ar_schemes=ar_schemes)


if __name__ == "__main__":
    main()
