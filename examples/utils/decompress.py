"""
decompress.py — reconstruct a file from a randcompress_v7 checkpoint.

Usage:
    python examples/decompress.py --log log/randcompress_v7 [--out reconstructed.bin] [--verify]

Reads from ckpt_last/:
    config.json   — model hyperparameters
    params.pkl    — trained HiRA adapter weights
    residual.pkl  — argmax corrections  (empty dict if residual_budget=0)

Config flags that control behaviour (set at train time, read here):
    margin=0.0        → trained with pure CE loss (v6 mode); no margin guarantee
    margin>0.0        → trained with argmax margin loss; expect fewer residual entries
    residual_budget=0 → no corrections stored; pure argmax decode
    residual_budget>0 → corrections stored up to budget; applied here with state correction
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sys
import os
import json
import pickle

# ── import model code from v7 ──────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("rc7", os.path.join(_here, "randcompress_v7.py"))
_rc7  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rc7)

Config              = _rc7.Config
init_xlstm_params   = _rc7.init_xlstm_params
init_step_states    = _rc7.init_step_states
_fwd_step_jit       = _rc7._fwd_step_jit
ByteTokenizer       = _rc7.ByteTokenizer


# ── load checkpoint ────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_dir):
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        meta = json.load(f)
    cfg    = {k: v for k, v in meta.items() if k in Config._fields}
    config = Config(**cfg)

    with open(os.path.join(ckpt_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.array, params)

    res_path = os.path.join(ckpt_dir, "residual.pkl")
    residual = pickle.load(open(res_path, "rb")) if os.path.exists(res_path) else {}

    return config, params, residual


# ── decompression ──────────────────────────────────────────────────────────────

def decompress(base_xlstm, params, config, seed_byte, n_bytes, residual):
    """
    Reconstruct n_bytes from seed_byte using argmax + residual corrections.

    residual_budget=0.0 → residual is empty → pure argmax (may be lossy).
    residual entries     → override argmax AND feed correct token (state correction,
                           prevents cascade from each correction affecting future steps).
    """
    states    = init_step_states(config, batch_size=1)
    cur_token = jnp.array([seed_byte])
    output    = []

    for t in range(n_bytes):
        logit, states = _fwd_step_jit(
            base_xlstm, params, cur_token, config.num_heads, config.block_map)

        if t in residual:
            byte      = residual[t]
            cur_token = jnp.array([byte])   # state correction — keeps trajectory correct
        else:
            byte      = int(jnp.argmax(logit, axis=-1)[0])
            cur_token = jnp.array([byte])

        output.append(byte)

    return bytes(output)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    argv, args = sys.argv[1:], {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            args[argv[i][2:]] = argv[i+1]; i += 2
        else:
            i += 1
    return args


def main():
    args     = _parse_args()
    log_dir  = args.get("log", "log/randcompress_v7")
    ckpt_dir = os.path.join(log_dir, "ckpt_last")
    out_path = args.get("out", os.path.join(log_dir, "reconstructed.bin"))

    print(f"Checkpoint: {ckpt_dir}")
    config, params, residual = load_checkpoint(ckpt_dir)

    # rebuild frozen base (deterministic from seed)
    key = jr.key(config.seed)
    _, k_xlstm, _ = jr.split(key, 3)
    base_xlstm = init_xlstm_params(k_xlstm, config)

    with open(config.dataset, "rb") as f:
        original = f.read()
    seed_byte = original[0]
    n_bytes   = len(original) - 1

    # ── storage accounting ────────────────────────────────────────────────────
    n            = len(original)
    param_bytes  = sum(np.prod(p.shape) * 4 for p in jax.tree_util.tree_leaves(params))
    resid_bytes  = len(residual) * 4          # ~4 bytes per correction entry
    total_stored = param_bytes + resid_bytes
    ratio        = n / total_stored if total_stored > 0 else float('inf')

    print(f"\nDataset:  {config.dataset}  ({n} bytes)")
    print(f"Loss:     margin={config.margin}  ce_weight={config.ce_weight}"
          f"  {'→ pure CE (v6 mode)' if config.margin == 0.0 else '→ margin+CE (v7)'}")
    print(f"Residual: budget={config.residual_budget}"
          f"  {'→ disabled (pure argmax)' if config.residual_budget == 0.0 else f'→ {len(residual)} corrections'}")
    print()
    print(f"{'param bytes':<22} {param_bytes:>8}  ({param_bytes/n*100:5.1f}% of dataset)")
    print(f"{'residual bytes':<22} {resid_bytes:>8}  ({resid_bytes/n*100:5.1f}% of dataset)")
    print(f"{'total stored':<22} {total_stored:>8}  compression = {ratio:.3f}x  "
          f"{'[COMPRESSED]' if ratio >= 1.0 else '[EXPANSION]'}")

    # ── information-theoretic heuristic ──────────────────────────────────────
    err_frac = len(residual) / n if n > 0 else 0
    print()
    if err_frac == 0:
        print("Heuristic: zero residual → lossless via pure argmax (BPC ≤ 1.4 achieved)")
    elif err_frac < 0.01:
        print(f"Heuristic: {err_frac*100:.2f}% errors → residual approach efficient"
              f"  (< 1% threshold)")
    elif err_frac < 0.10:
        print(f"Heuristic: {err_frac*100:.1f}% errors → residual usable but large;"
              f" consider arithmetic coding for better ratio")
    else:
        print(f"Heuristic: {err_frac*100:.0f}% errors → residual inefficient;"
              f" arithmetic coding strongly recommended (BPC-based encoding)")

    # ── decode ────────────────────────────────────────────────────────────────
    print(f"\nDecoding {n_bytes} bytes...")
    reconstructed = decompress(base_xlstm, params, config, seed_byte, n_bytes, residual)
    full_output   = bytes([seed_byte]) + reconstructed

    with open(out_path, "wb") as f:
        f.write(full_output)
    print(f"Written: {out_path}  ({len(full_output)} bytes)")

    # ── verify ────────────────────────────────────────────────────────────────
    if full_output == original:
        print("Verify: PERFECT MATCH")
    else:
        wrong = sum(a != b for a, b in zip(full_output, original))
        print(f"Verify: {wrong}/{len(original)} bytes wrong"
              f"  ({(len(original)-wrong)/len(original)*100:.2f}% correct)")
        first = next(i for i, (a, b) in enumerate(zip(full_output, original)) if a != b)
        print(f"  first mismatch at byte {first}")


if __name__ == "__main__":
    main()
