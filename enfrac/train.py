"""Overfit a ByteFractalGen to one file. Each patch_len_list[0]-byte chunk is fully independent
(root_cond is a fixed constant, no state carried across chunks), so training is just: iterate
chunks, forward, backward, AdamW step -- only the trainable partition (HiRA's B, root_cond, and
plain-linear weights when use_hira=False) receives gradients/updates.

Multi-device (TPU pod slice / multi-GPU) data parallelism: when jax.local_device_count() > 1,
train() switches from the single-chunk-per-step eqx.filter_jit path to a pmap'd step that feeds
each device its own `per_device_batch` chunks per training step, computes grads independently
per device, then jax.lax.pmean's them across the `"device"` axis before every replica applies
the identical averaged update -- the standard "replicate params, average grads" data-parallel
recipe. The trainable partition and optimizer state are replicated once (jax.device_put_replicated)
before the loop and live on-device for its whole duration; only chunk indices move host<->device
every step. Frozen leaves (HiRA's W0/A, byte_embed's table) are passed with in_axes=None, so
pmap broadcasts one copy to every device instead of also replicating them as mapped arguments.
With a single device this reduces to the previous per-chunk jit path exactly (no behavior
change -- verified against the single-device runs this package was validated with).

Note: this only parallelizes train.py's hot loop (the actual FLOPs). compress.py/decompress.py
stay single-device -- their cost is dominated by many small host<->device dispatches from
generate()'s python-level autoregressive recursion (bit-exactness requires it, see model.py's
module docstring), not raw compute, so pmap wouldn't help there without a much larger rewrite.
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
    n = len(raw_bytes)
    n_chunks = math.ceil(n / patch_len)
    padded = np.zeros(n_chunks * patch_len, dtype=np.uint8)
    padded[:n] = raw_bytes
    return padded.reshape(n_chunks, patch_len)


def _lr_schedule(lr: float, warmup_steps: int):
    def sched(count):
        return lr * jnp.minimum(1.0, (count + 1) / max(1, warmup_steps))
    return sched


def _loss_fn(trainable, static, chunk):
    m = eqx.combine(trainable, static)
    loss, metrics = m(chunk)
    return loss, metrics


@eqx.filter_jit
def _train_step(model, opt_state, opt, filter_spec, chunk):
    """Single-device path: one (or a locally-batched) chunk per step, no cross-device sync."""
    trainable, static = eqx.partition(model, filter_spec)
    (loss, metrics), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(trainable, static, chunk)
    updates, opt_state = opt.update(grads, opt_state, trainable)
    trainable = eqx.apply_updates(trainable, updates)
    model = eqx.combine(trainable, static)
    return model, opt_state, loss, metrics


def _make_pmap_train_step(opt):
    """Multi-device path: `trainable`/`opt_state` carry a leading device axis (kept resident on
    device across steps via jax.device_put_replicated below); `static`/`filter_spec` are the
    same Python object on every device (in_axes=None -- pmap broadcasts them, no replication of
    frozen arrays needed since eqx.combine only ever reads them, never updates them); `chunk`
    carries a leading device axis, one shard of `per_device_batch` chunks per device."""
    def step(trainable, static, filter_spec, opt_state, chunk):
        (loss, metrics), grads = eqx.filter_value_and_grad(_loss_fn, has_aux=True)(trainable, static, chunk)
        grads = jax.lax.pmean(grads, axis_name="device")
        loss = jax.lax.pmean(loss, axis_name="device")
        metrics = jax.lax.pmean(metrics, axis_name="device")
        updates, opt_state = opt.update(grads, opt_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        return trainable, opt_state, loss, metrics
    return jax.pmap(step, axis_name="device", in_axes=(0, None, None, 0, 0))


def train(model, chunks: np.ndarray, steps: int, lr: float, warmup_steps: int,
          grad_clip: float, log_every: int, n_raw: int, filter_spec=None, per_device_batch: int = 1):
    """Generic w.r.t. which ByteFractalGen variant `model` is (main/PEFT or baseline) -- pass
    that variant's own trainable_filter(model) result as `filter_spec` (defaults to the
    main/PEFT package's trainable_filter, which also happens to work for baseline models since
    its trainable-name set is a superset, but callers should pass their own explicitly).

    Automatically data-parallelizes across jax.local_device_count() devices when > 1 (see module
    docstring) -- each step then consumes n_devices * per_device_batch chunks."""
    ctx = jnp.asarray(chunks.astype(np.int32))   # [n_chunks, P0]
    n_chunks, P0 = ctx.shape

    if filter_spec is None:
        filter_spec = trainable_filter(model)
    trainable0, static = eqx.partition(model, filter_spec)
    param_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(trainable0))

    opt = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=_lr_schedule(lr, warmup_steps), b1=0.9, b2=0.95, weight_decay=0.0),
    )
    opt_state0 = opt.init(trainable0)

    n_devices = jax.local_device_count()
    multi_device = n_devices > 1
    batch_sz = n_devices * per_device_batch if multi_device else 1
    print(f"[fractalgen train] jax devices: {jax.local_devices()}  "
          f"{'data-parallel' if multi_device else 'single-device'}, batch_sz={batch_sz}")

    if multi_device:
        # jax.device_put_replicated is deprecated; broadcasting to a leading (n_devices,) axis
        # and handing that straight to pmap is the current drop-in -- pmap shards it onto the
        # actual devices itself on first call.
        replicate = lambda pytree: jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (n_devices,) + x.shape), pytree)
        trainable = replicate(trainable0)
        opt_state = replicate(opt_state0)
        pmap_step = _make_pmap_train_step(opt)
    else:
        model_state = model   # single eqx.Module carrying both trainable+frozen leaves
        opt_state = opt_state0

    t0 = time.perf_counter()
    epoch_ce_nats = 0.0
    epoch_steps = 0
    epoch = 1
    prev_epoch_idx = 0
    pbar = tqdm(range(1, steps + 1), desc="fractalgen", dynamic_ncols=True)
    for step in pbar:
        idxs = jnp.asarray([((step - 1) * batch_sz + j) % n_chunks for j in range(batch_sz)])
        chunk = ctx[idxs]   # [batch_sz, P0]

        if multi_device:
            chunk = chunk.reshape(n_devices, per_device_batch, P0)
            trainable, opt_state, loss, metrics = pmap_step(trainable, static, filter_spec, opt_state, chunk)
            loss_v = float(loss[0])
            acc = float(metrics["byte_acc"][0])
            bpb = float(metrics["bpb"][0])
        else:
            model_state, opt_state, loss, metrics = _train_step(model_state, opt_state, opt, filter_spec, chunk)
            loss_v = float(loss)
            acc = float(metrics["byte_acc"])
            bpb = float(metrics["bpb"])

        epoch_ce_nats += loss_v * P0 * batch_sz   # loss is mean nats/byte over this step's batch
        epoch_steps += batch_sz

        pbar.set_postfix(loss=f"{loss_v:.4f}", acc=f"{acc:.2%}", bpb=f"{bpb:.3f}")

        if step % log_every == 0 or step == steps:
            print(f"[step {step}/{steps}]  bpb={bpb:.4f}  acc={acc:.2%}  loss={loss_v:.4f}")

        cur_epoch_idx = (step * batch_sz) // n_chunks
        if cur_epoch_idx > prev_epoch_idx or step == steps:
            prev_epoch_idx = cur_epoch_idx
            n_bytes_seen = epoch_steps * P0
            est_ce_bits = epoch_ce_nats / math.log(2)
            est_bpb = est_ce_bits / n_bytes_seen
            est_rc_bytes = est_ce_bits / 8 * (n_raw / n_bytes_seen)
            est_total = param_bytes + est_rc_bytes
            est_ratio = n_raw / est_total if est_total > 0 else float("inf")
            print(f"[epoch ~{epoch}]  CE~{est_bpb:.4f}bpb (theoretical)  "
                  f"est size ~{est_rc_bytes:.0f}B (rc) + {param_bytes}B (params) "
                  f"= {est_total:.0f}B  est ratio ~{est_ratio:.4f}x")
            epoch_ce_nats = 0.0
            epoch_steps = 0
            epoch += 1

    pbar.close()
    dt = time.perf_counter() - t0
    print(f"[fractalgen train] {steps} steps over {n_chunks} chunks  {dt:.1f}s  "
          f"({steps / dt:.1f} steps/s, {steps * batch_sz * P0 / dt / 1e3:.1f} kB/s effective)")

    if multi_device:
        final_trainable = jax.tree_util.tree_map(lambda x: x[0], trainable)
        return eqx.combine(final_trainable, static)
    return model_state


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
    n_levels = len(patch_len_list) - 1
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
    n_chunks = chunks.shape[0]
    log_every = (max(1, round(float(tcfg.log_every) * n_chunks))
                 if "." in tcfg.log_every else int(tcfg.log_every))
    print(f"log_every={tcfg.log_every} -> {log_every} steps "
          f"({'epoch fraction' if '.' in tcfg.log_every else 'raw steps'})")
    if log_every <= 2:
        print(f"WARNING: log_every={log_every} is very frequent (prints nearly every step) "
              f"-- pass a float like 0.1 for 'every 10% of an epoch' if this wasn't intended")

    model = train(model, chunks, tcfg.steps, tcfg.lr, tcfg.warmup_steps, tcfg.grad_clip,
                  log_every, len(raw_bytes), per_device_batch=tcfg.per_device_batch)

    from .checkpoint import save_model
    save_model(tcfg.log_dir, model)
    with open(os.path.join(tcfg.log_dir, "meta.json"), "w") as f:
        import json
        json.dump({"n_raw_bytes": len(raw_bytes)}, f)
    print(f"Saved model to {tcfg.log_dir}/")


if __name__ == "__main__":
    main()
