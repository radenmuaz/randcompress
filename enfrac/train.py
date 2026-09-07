"""Overfit a ByteFractalGen to one file. Level 0 is a real causal AR transformer whose sequence
IS the file's level-0 timesteps (patch_len_list[0]-byte patches, not bytes -- see config.py's
TrainConfig docstring). Training is teacher-forced and FULLY PARALLEL: one causally-masked
attention computation over all n_timesteps at once, exactly like training any GPT-style
transformer on a long sequence -- no recurrence, no per-step state carry, no stop_gradient.
`n_epochs` full forward+backward passes, one gradient update each. remat_time/remat_depth (see
model.py's Trunk.forward_remat) make long sequences memory-feasible via jax.checkpoint --
recompute during backward instead of storing, exact gradients, no truncation.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .config import TrainConfig, build_config, parse_configs
from .model import ByteFractalGen, ModelConfig, trainable_filter
from .tokenizer import load_bytes


class _Tee:
    """stdout/stderr + file, eager flush -- tail -f <log_dir>/train.log to watch a run live."""
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    @property
    def encoding(self):
        return getattr(self.files[0], "encoding", "utf-8")


def make_chunks(raw_bytes: np.ndarray, patch_len: int) -> np.ndarray:
    """Despite the name (kept for compress.py/decompress.py compatibility), this returns the
    level-0 TIMESTEP sequence: [n_timesteps, patch_len] -- n_timesteps = ceil(n_raw/patch_len)
    patches, each patch_len bytes. Not independent "chunks" -- level 0 attends across all of
    them as one sequence (see model.py's module docstring)."""
    n = len(raw_bytes)
    n_timesteps = math.ceil(n / patch_len)
    padded = np.zeros(n_timesteps * patch_len, dtype=np.uint8)
    padded[:n] = raw_bytes
    return padded.reshape(n_timesteps, patch_len)


def timesteps_per_level(model, n_timesteps0: int) -> list[int]:
    """Total patch-count at each level, across the WHOLE file: level 0 = n_timesteps0 (the
    level-0 transformer's own sequence length); level l>=1 = n_timesteps0 * prod(seq_lens[1..l])
    (every level-0 timestep's own seq_lens[l] children, summed over all n_timesteps0 parents) --
    equivalently ceil(n_raw_bytes / patch_len_list[l]), the total number of patch_len_list[l]-byte
    patches in the file. Used to log/report how much work each level actually does."""
    counts = [n_timesteps0]
    cur = n_timesteps0
    for l in range(1, model.n_levels):
        cur = cur * model.seq_lens[l]
        counts.append(cur)
    return counts


def _loss_fn(trainable, static, byte_seq, remat_time, remat_depth):
    m = eqx.combine(trainable, static)
    loss, metrics = m(byte_seq, remat_time, remat_depth)
    return loss, metrics


def train(model, chunks: np.ndarray, lr: float, grad_clip: float, remat_time: str,
          remat_depth: str, n_epochs: int, n_raw: int, filter_spec=None):
    """Generic w.r.t. which ByteFractalGen variant `model` is (main/PEFT or baseline) -- pass
    that variant's own trainable_filter(model) result as `filter_spec`."""
    byte_seq = jnp.asarray(chunks.astype(np.int32))   # [n_timesteps, P0]
    n_timesteps, P0 = byte_seq.shape

    if filter_spec is None:
        filter_spec = trainable_filter(model)
    trainable, static = eqx.partition(model, filter_spec)
    param_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(trainable))

    opt = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, b1=0.9, b2=0.95, weight_decay=0.0),   # fixed lr, no schedule
    )
    opt_state = opt.init(trainable)

    print(f"[fractalgen train] jax devices: {jax.local_devices()}  n_timesteps={n_timesteps}  "
          f"(patch_len_list[0]={P0} bytes/timestep)  remat_time={remat_time}  "
          f"remat_depth={remat_depth}  n_epochs={n_epochs}")

    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(_loss_fn, has_aux=True))

    def log(epoch, loss, acc):
        bpb = loss / math.log(2)
        est_rc_bytes = loss / math.log(2) / 8 * n_raw   # theoretical, no rc_encode() call
        est_total = param_bytes + est_rc_bytes
        est_ratio = n_raw / est_total if est_total > 0 else float("inf")
        print(f"[epoch {epoch}/{n_epochs}]  bpb={bpb:.4f}  acc={acc:.2%}  loss={loss:.4f}  "
              f"est size ~{est_rc_bytes:.0f}B (rc) + {param_bytes}B (params) "
              f"= {est_total:.0f}B  est ratio ~{est_ratio:.4f}x")

    t0 = time.perf_counter()
    pbar = tqdm(range(1, n_epochs + 1), desc="epoch", dynamic_ncols=True)
    for epoch in pbar:
        (loss, metrics), grads = grad_fn(trainable, static, byte_seq, remat_time, remat_depth)
        updates, opt_state = opt.update(grads, opt_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        loss_v, acc_v = float(loss), float(metrics["byte_acc"])
        pbar.set_postfix(loss=f"{loss_v:.4f}", acc=f"{acc_v:.2%}")
        log(epoch, loss_v, acc_v)
    pbar.close()

    dt = time.perf_counter() - t0
    print(f"[fractalgen train] {n_epochs} epochs over {n_timesteps} timesteps  {dt:.1f}s")

    return eqx.combine(trainable, static)


def main() -> None:
    parser = parse_configs(ModelConfig)
    parser.add_argument("--d_model", type=int, default=None, help="broadcast uniformly to every level")
    parser.add_argument("--n_layers", type=int, default=None, help="broadcast uniformly to every level")
    parser.add_argument("--n_heads", type=int, default=None, help="broadcast uniformly to every level")
    parser.add_argument("--mlp_mult", type=int, default=None, help="broadcast uniformly to every level")
    args = parser.parse_args()

    train_overrides = {f.name: getattr(args, f.name, None) for f in dataclasses.fields(TrainConfig)}
    tcfg = build_config(TrainConfig, args.config, ("train", "train_config"), train_overrides)

    patch_len_list = getattr(args, "patch_len_list", None) or ModelConfig.patch_len_list
    n_levels = len(patch_len_list)
    broadcast = {}
    if args.d_model is not None:
        broadcast["d_model_list"] = (args.d_model,) * n_levels
    if args.n_layers is not None:
        broadcast["n_layers_list"] = (args.n_layers,) * n_levels
    if args.n_heads is not None:
        broadcast["n_heads_list"] = (args.n_heads,) * n_levels
    if args.mlp_mult is not None:
        broadcast["mlp_mult_list"] = (args.mlp_mult,) * n_levels

    model_overrides = {f.name: getattr(args, f.name, None) for f in dataclasses.fields(ModelConfig)}
    model_overrides.update(broadcast)
    if tcfg.seed is not None and model_overrides.get("seed") is None:
        model_overrides["seed"] = tcfg.seed
    mcfg = build_config(ModelConfig, args.config, ("model", "model_config"), model_overrides)

    os.makedirs(tcfg.log_dir, exist_ok=True)
    log_file = open(os.path.join(tcfg.log_dir, "train.log"), "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"logging to {os.path.join(tcfg.log_dir, 'train.log')} -- tail -f it")

    raw_bytes = load_bytes(tcfg.dataset)
    print(f"dataset={tcfg.dataset}  n_bytes={len(raw_bytes)}  patch_len_list={mcfg.patch_len_list}")

    model = ByteFractalGen(mcfg)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
    n_trainable = sum(x.size for x in jax.tree_util.tree_leaves(eqx.partition(model, trainable_filter(model))[0]))
    print(f"params={n_params:,} (trainable={n_trainable:,})  use_hira={mcfg.use_hira} "
          f"hira_r={mcfg.hira_r}  seed={mcfg.seed}  seq_lens={model.seq_lens}")

    chunks = make_chunks(raw_bytes, mcfg.patch_len_list[0])

    tsteps = timesteps_per_level(model, chunks.shape[0])
    print("timesteps per level (total patches of that level's size, whole file): " +
          "  ".join(f"level{l}={t:,}" for l, t in enumerate(tsteps)))

    model = train(model, chunks, tcfg.lr, tcfg.grad_clip, tcfg.remat_time, tcfg.remat_depth,
                  tcfg.n_epochs, len(raw_bytes))

    from .checkpoint import save_model
    save_model(tcfg.log_dir, model)
    with open(os.path.join(tcfg.log_dir, "meta.json"), "w") as f:
        import json
        json.dump({"n_raw_bytes": len(raw_bytes), "timesteps_per_level": tsteps}, f)
    print(f"Saved model to {tcfg.log_dir}/")


if __name__ == "__main__":
    main()
