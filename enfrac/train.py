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
import json
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


def resolve_file_chunk_bytes(n_raw: int, patch_len0: int, file_chunk_bytes: int | None,
                              file_chunk_count: int | None) -> int:
    """Resolves TrainConfig's file_chunk_bytes/file_chunk_count (at most one set -- see
    TrainConfig docstring) to a single concrete file_chunk_bytes value, always a multiple of
    patch_len0 (required so each file chunk cleanly divides into a whole number of level-0
    timesteps -- see make_file_chunks). Both None (the default) resolves to "exactly one file
    chunk containing the whole (padded) file" -- the long-standing B=1 special case, byte-for-byte
    identical to this codebase's behavior before file chunking existed."""
    assert not (file_chunk_bytes is not None and file_chunk_count is not None), \
        "set at most one of file_chunk_bytes/file_chunk_count, not both"
    if file_chunk_count is not None:
        assert file_chunk_count >= 1, f"file_chunk_count must be >=1, got {file_chunk_count}"
        raw_size = -(-n_raw // file_chunk_count)   # ceil
    elif file_chunk_bytes is not None:
        assert file_chunk_bytes >= 1, f"file_chunk_bytes must be >=1, got {file_chunk_bytes}"
        raw_size = file_chunk_bytes
    else:
        raw_size = n_raw   # both unset -> exactly one chunk = the whole file
    resolved = -(-raw_size // patch_len0) * patch_len0   # round UP to a multiple of patch_len0
    return max(resolved, patch_len0)


def make_file_chunks(raw_bytes: np.ndarray, patch_len0: int, file_chunk_bytes: int,
                      warn: bool = True) -> np.ndarray:
    """Splits raw_bytes into independent FILE CHUNKS of file_chunk_bytes each (weight-shared,
    never attending across each other -- see ByteFractalGen.__call__'s docstring), padding the
    FILE (not just trimming) so its length is an exact multiple of file_chunk_bytes -- prints a
    warning naming exactly how many padding bytes were added when the file doesn't divide evenly,
    same spirit as make_chunks' always-silent per-patch padding, but file-chunk padding is loud
    since it can be a much bigger fraction of the last chunk. Returns
    [n_file_chunks, chunk_n_timesteps, patch_len0] int, chunk_n_timesteps = file_chunk_bytes //
    patch_len0 (file_chunk_bytes is guaranteed a multiple of patch_len0 by
    resolve_file_chunk_bytes). decompress.py truncates the padding back off using the real
    n_raw_bytes recorded in meta.json, exactly like it already does for the final level-0 patch's
    own padding today -- no new truncation mechanism needed, just applied one level up too.
    warn=False suppresses the print -- used for the both-None default (file_chunk_bytes resolved
    to "the whole file"), where any padding is just the ordinary, always-silent last-patch
    rounding this codebase has always done, not a new file-chunking concern worth flagging."""
    assert file_chunk_bytes % patch_len0 == 0, \
        f"file_chunk_bytes={file_chunk_bytes} must be a multiple of patch_len_list[0]={patch_len0}"
    n = len(raw_bytes)
    n_file_chunks = -(-n // file_chunk_bytes)   # ceil
    total = n_file_chunks * file_chunk_bytes
    if total != n and warn:
        print(f"WARNING: file size {n:,}B is not a multiple of file_chunk_bytes={file_chunk_bytes:,}B "
              f"-- padding the last chunk with {total - n:,} zero bytes ({n_file_chunks} chunks total). "
              f"decompress truncates this back off automatically via meta.json's n_raw_bytes.")
    padded = np.zeros(total, dtype=np.uint8)
    padded[:n] = raw_bytes
    chunk_n_timesteps = file_chunk_bytes // patch_len0
    return padded.reshape(n_file_chunks, chunk_n_timesteps, patch_len0)


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


def _loss_fn(trainable, static, byte_seq, remat_time, remat_depth, micro_batch):
    m = eqx.combine(trainable, static)
    loss, metrics = m(byte_seq, remat_time, remat_depth, micro_batch)
    return loss, metrics


def build_optimizer(lr: float, grad_clip: float):
    """Factored out of train() so a resume path can reconstruct the IDENTICAL optimizer (same
    clip/adamw hyperparameters) to build a matching opt_state skeleton for deserialization --
    must be called with the same lr/grad_clip the checkpoint was trained with."""
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, b1=0.9, b2=0.95, weight_decay=0.0),   # fixed lr, no schedule
    )


def train(model, chunks: np.ndarray, lr: float, grad_clip: float, remat_time: str,
          remat_depth: str, n_epochs: int, n_raw: int, filter_spec=None, micro_batch: int = 8192,
          chunk_batch_size: int = 16, seed: int = 0, log_dir: str | None = None,
          start_epoch: int = 1, init_opt_state=None, rng: np.random.Generator | None = None,
          on_epoch_end=None):
    """start_epoch/init_opt_state/rng/on_epoch_end: optional RESUME support. Pass all three of
    start_epoch (>1), init_opt_state (an opt_state pytree matching build_optimizer(lr,
    grad_clip).init(trainable)'s structure), and rng (a np.random.Generator carrying forward the
    exact minibatch-sampling stream) together to continue an interrupted run bit-for-bit instead
    of restarting; omit all three (defaults) for a fresh run, unchanged from before. on_epoch_end,
    if given, is called as on_epoch_end(epoch, model, opt_state, rng) after EVERY epoch (both the
    n_file_chunks==1 and >1 branches) with the current (unreplicated) model/opt_state/rng -- the
    caller uses this to checkpoint full resumable state each epoch, not just at the very end."""
    """Generic w.r.t. which ByteFractalGen variant `model` is (main/PEFT or baseline) -- pass
    that variant's own trainable_filter(model) result as `filter_spec`. micro_batch: levels-1+
    jax.lax.scan chunk size, see ByteFractalGen._train_recurse's docstring -- only matters (and
    only needs lowering from the default) once a level's total patch count reaches the millions
    (real enwik8/9 scale); small corpora (fatihah/juz1/quran_uthmani) never hit that regime.

    chunks: [n_file_chunks, chunk_n_timesteps, P0] -- see make_file_chunks. n_file_chunks==1 (the
    default, whole file as one chunk) is a special case handled with the OLD exact behavior/log
    format: one gradient step per epoch over the whole (only) chunk, no separate full-pass (the
    single step's own loss already covers 100% of the file, so a second identical forward pass to
    "evaluate the whole file" would be pure waste). n_file_chunks>1 does real minibatch training:
    each of `steps_per_epoch` steps per epoch samples chunk_batch_size chunk indices WITH
    REPLACEMENT (ordinary SGD minibatching, not epoch-exhaustive shuffling) and takes one gradient
    step on that batch (shape-fixed across all steps -> compiles once); at the END of every epoch,
    a separate full pass (no gradient, B=1 per chunk, same compiled executable reused for every
    chunk since the shape never changes) walks every chunk once, printing that chunk's own bpb and
    accumulating exact sum-of-nats/sum-of-positions across ALL chunks for a true whole-file bpb --
    this is the only place "whole-file bpb" is exact for n_file_chunks>1 (the per-step training
    loss is only a sampled MINIBATCH's aggregate, not the whole file's)."""
    chunks_jnp = jnp.asarray(chunks.astype(np.int32))   # [n_file_chunks, chunk_n_timesteps, P0]
    n_file_chunks, chunk_n_timesteps, P0 = chunks_jnp.shape
    chunk_batch_size = max(1, min(chunk_batch_size, n_file_chunks, 128))
    if rng is None:
        rng = np.random.default_rng(seed)

    if filter_spec is None:
        filter_spec = trainable_filter(model)
    trainable, static = eqx.partition(model, filter_spec)
    param_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(trainable))

    opt = build_optimizer(lr, grad_clip)
    opt_state = init_opt_state if init_opt_state is not None else opt.init(trainable)

    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(_loss_fn, has_aux=True))
    eval_fn = eqx.filter_jit(_loss_fn)   # no grad -- used for the B=1 per-chunk full-pass below

    def size_est(loss):
        bpb = loss / math.log(2)
        est_rc_bytes = bpb / 8 * n_raw   # theoretical, no rc_encode() call
        est_total = param_bytes + est_rc_bytes
        est_ratio = n_raw / est_total if est_total > 0 else float("inf")
        return bpb, est_rc_bytes, est_total, est_ratio

    if n_file_chunks == 1:
        # -- Old exact behavior: one gradient step per epoch over the whole (only) chunk --
        print(f"[fractalgen train] jax devices: {jax.local_devices()}  n_timesteps={chunk_n_timesteps}  "
              f"(patch_len_list[0]={P0} bytes/timestep)  remat_time={remat_time}  "
              f"remat_depth={remat_depth}  micro_batch={micro_batch}  n_epochs={n_epochs}  "
              f"file_chunks=1 (whole file)")
        t0 = time.perf_counter()
        pbar = tqdm(range(start_epoch, n_epochs + 1), desc="epoch", dynamic_ncols=True)
        for epoch in pbar:
            (loss, metrics), grads = grad_fn(trainable, static, chunks_jnp, remat_time, remat_depth, micro_batch)
            updates, opt_state = opt.update(grads, opt_state, trainable)
            trainable = eqx.apply_updates(trainable, updates)
            loss_v, acc_v = float(loss), float(metrics["byte_acc"])
            pbar.set_postfix(loss=f"{loss_v:.4f}", acc=f"{acc_v:.2%}")
            bpb, est_rc_bytes, est_total, est_ratio = size_est(loss_v)
            print(f"[epoch {epoch}/{n_epochs}]  bpb={bpb:.4f}  acc={acc_v:.2%}  loss={loss_v:.4f}  "
                  f"est size ~{est_rc_bytes:.0f}B (rc) + {param_bytes}B (params) "
                  f"= {est_total:.0f}B  est ratio ~{est_ratio:.4f}x")
            if on_epoch_end:
                on_epoch_end(epoch, eqx.combine(trainable, static), opt_state, rng)
        pbar.close()
        dt = time.perf_counter() - t0
        print(f"[fractalgen train] {n_epochs} epochs over {chunk_n_timesteps} timesteps  {dt:.1f}s")
        return eqx.combine(trainable, static)

    # -- Real minibatch training over n_file_chunks independent, weight-shared file chunks --
    # DATA-PARALLEL across every visible device (jax.local_device_count(), e.g. 4 on a v4-8) via
    # eqx.filter_pmap -- see CLAUDE.md's now-corrected note: earlier this project had NO
    # multi-device parallelism at all (single default device only, others idle). Each device gets
    # its own per_device_bs-sized slice of the sampled chunk_batch_size batch, computes its own
    # forward+backward, and gradients/loss are averaged across devices via jax.lax.pmean INSIDE
    # the pmapped step (not a separate host round-trip) before the (replicated, therefore
    # identical-per-device) optimizer update -- standard synchronous data-parallel SGD. trainable/
    # opt_state stay REPLICATED (one physical copy per device, kept in sync by construction since
    # every device applies the identical pmean'd update) for the whole training loop; only
    # collapsed back to a single copy (device 0's, all devices identical by construction) for the
    # per-chunk eval pass and the final returned model, which stay single-device (not the
    # throughput-critical path).
    n_devices = 1 if os.environ.get("NO_PMAP") == "1" else jax.local_device_count()
    per_device_bs = max(1, -(-chunk_batch_size // n_devices))   # ceil
    if per_device_bs * n_devices != chunk_batch_size:
        chunk_batch_size = per_device_bs * n_devices             # round up to a clean multiple
    steps_per_epoch = max(1, -(-n_file_chunks // chunk_batch_size))
    print(f"[fractalgen train] jax devices: {jax.local_devices()}  n_devices={n_devices}  "
          f"n_file_chunks={n_file_chunks}  chunk_n_timesteps={chunk_n_timesteps}  "
          f"(patch_len_list[0]={P0} bytes/timestep)  chunk_batch_size={chunk_batch_size}  "
          f"(per_device={per_device_bs})  steps_per_epoch={steps_per_epoch}  "
          f"remat_time={remat_time}  remat_depth={remat_depth}  micro_batch={micro_batch}  "
          f"n_epochs={n_epochs}")

    def pstep(trainable, opt_state, static, batch, remat_time, remat_depth, micro_batch):
        (loss, metrics), grads = grad_fn(trainable, static, batch, remat_time, remat_depth, micro_batch)
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        acc = jax.lax.pmean(metrics["byte_acc"], axis_name="devices")
        updates, opt_state = opt.update(grads, opt_state, trainable)
        trainable = eqx.apply_updates(trainable, updates)
        return trainable, opt_state, loss, acc

    pmap_step = eqx.filter_pmap(
        pstep, axis_name="devices",
        in_axes=(0, 0, None, 0, None, None, None), out_axes=(0, 0, 0, 0),
    )

    def _replicate(pytree):
        # jax.device_put_replicated is deprecated in this jax version -- broadcasting to a
        # leading n_devices axis is the modern drop-in: jax.pmap places slice i onto device i
        # automatically when the pmapped function is called, no manual per-device placement needed.
        return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_devices,) + x.shape), pytree)

    trainable_r = _replicate(trainable)
    opt_state_r = _replicate(opt_state)   # opt_state (fresh or resumed) is always a single, unreplicated copy

    t0 = time.perf_counter()
    for epoch in range(start_epoch, n_epochs + 1):
        pbar = tqdm(range(1, steps_per_epoch + 1), desc=f"epoch {epoch}/{n_epochs} (steps)", dynamic_ncols=True)
        for step in pbar:
            idx = rng.integers(0, n_file_chunks, size=chunk_batch_size)   # WITH replacement
            batch = chunks_jnp[idx].reshape(n_devices, per_device_bs, chunk_n_timesteps, P0)
            trainable_r, opt_state_r, loss, acc = pmap_step(
                trainable_r, opt_state_r, static, batch, remat_time, remat_depth, micro_batch)
            loss_v, acc_v = float(loss[0]), float(acc[0])   # every device holds the identical pmean'd value
            pbar.set_postfix(loss=f"{loss_v:.4f}", acc=f"{acc_v:.2%}")
        pbar.close()
        # collapse for eval/checkpointing below -- x[0] on a pmap output can still carry that
        # array's PmapSharding (a real bug hit this session: flash attention's Pallas/Mosaic
        # kernel lowers fine inside pmap's own REPLICATED lowering, but a later plain
        # eqx.filter_jit call (eval_fn below) whose inputs still carry PmapSharding metadata makes
        # XLA try to auto-partition that same kernel via the SPMD path instead, which Mosaic
        # doesn't support -- "NotImplementedError: Mosaic kernels cannot be automatically
        # partitioned. Please wrap the call in a shard_map." jax.device_put onto a single device
        # strips that stale sharding, giving eval_fn a genuinely plain, single-device array.
        single_device = jax.devices()[0]
        trainable = jax.tree_util.tree_map(lambda x: jax.device_put(x[0], single_device), trainable_r)
        opt_state = jax.tree_util.tree_map(lambda x: jax.device_put(x[0], single_device), opt_state_r)

        # -- Full pass, every epoch: BATCHED (eval_group chunks per eval_fn call, not B=1 --
        # see CLAUDE.md's audit note: a B=1-per-chunk Python loop over n_file_chunks (thousands at
        # real scale) was measured to cost ~half of every epoch's wall time in pure per-call
        # dispatch overhead, not compute). Groups of eval_group chunks share ONE compiled shape
        # (reused every group, like training's chunk_batch_size shape); a smaller final group (if
        # n_file_chunks isn't a multiple of eval_group) compiles one extra shape, once. ce_nats/
        # positions/correct sums stay EXACT (summing a batch's aggregate == summing its rows'
        # individual aggregates -- no precision loss from batching, only fewer Python-dispatched
        # calls). The tradeoff: chunk_bpbs' resolution is now per-GROUP, not per-chunk (outlier
        # detection is coarser), which is what pays for the wall-clock win.
        eval_group = max(1, min(chunk_batch_size, n_file_chunks))
        chunk_log_every = max(1, -(-n_file_chunks // eval_group) // 20)
        total_ce = total_pos = total_correct = 0.0
        chunk_bpbs = []
        g = 0
        for start in range(0, n_file_chunks, eval_group):
            end = min(start + eval_group, n_file_chunks)
            _, m_c = eval_fn(trainable, static, chunks_jnp[start:end], remat_time, remat_depth, micro_batch)
            ce_c, pos_c, corr_c = float(m_c["ce_nats"]), float(m_c["positions"]), float(m_c["correct"])
            total_ce += ce_c
            total_pos += pos_c
            total_correct += corr_c
            chunk_bpb = (ce_c / pos_c) / math.log(2)
            chunk_bpbs.append(chunk_bpb)
            if g % chunk_log_every == 0 or end == n_file_chunks:
                print(f"[epoch {epoch}/{n_epochs}]  chunks {start}-{end - 1}/{n_file_chunks - 1}  "
                      f"bpb={chunk_bpb:.4f}  acc={corr_c / pos_c:.2%}")
            g += 1
        whole_loss = total_ce / total_pos
        whole_acc = total_correct / total_pos
        bpb, est_rc_bytes, est_total, est_ratio = size_est(whole_loss)
        print(f"[epoch {epoch}/{n_epochs}]  WHOLE FILE  bpb={bpb:.4f}  acc={whole_acc:.2%}  "
              f"loss={whole_loss:.4f}  chunk_bpb_range=[{min(chunk_bpbs):.4f}, {max(chunk_bpbs):.4f}]  "
              f"est size ~{est_rc_bytes:.0f}B (rc) + {param_bytes}B (params) "
              f"= {est_total:.0f}B  est ratio ~{est_ratio:.4f}x")
        if log_dir:
            with open(os.path.join(log_dir, "chunk_bpbs.jsonl"), "a") as jf:
                jf.write(json.dumps({
                    "epoch": epoch, "whole_bpb": round(bpb, 4), "whole_acc": round(whole_acc, 4),
                    "chunk_bpbs": [round(b, 4) for b in chunk_bpbs],
                }) + "\n")
        if on_epoch_end:
            on_epoch_end(epoch, eqx.combine(trainable, static), opt_state, rng)

    dt = time.perf_counter() - t0
    print(f"[fractalgen train] {n_epochs} epochs, {steps_per_epoch} steps/epoch, "
          f"{n_file_chunks} file chunks of {chunk_n_timesteps} timesteps each  {dt:.1f}s")

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

    file_chunk_bytes = resolve_file_chunk_bytes(
        len(raw_bytes), mcfg.patch_len_list[0], tcfg.file_chunk_bytes, tcfg.file_chunk_count)
    chunks = make_file_chunks(raw_bytes, mcfg.patch_len_list[0], file_chunk_bytes,
                               warn=(tcfg.file_chunk_bytes is not None or tcfg.file_chunk_count is not None))
    n_file_chunks, chunk_n_timesteps, _ = chunks.shape
    print(f"file_chunk_bytes={file_chunk_bytes:,}  n_file_chunks={n_file_chunks}  "
          f"chunk_n_timesteps={chunk_n_timesteps}  chunk_batch_size={tcfg.chunk_batch_size}")

    tsteps = timesteps_per_level(model, n_file_chunks * chunk_n_timesteps)
    print("timesteps per level (total patches of that level's size, whole file): " +
          "  ".join(f"level{l}={t:,}" for l, t in enumerate(tsteps)))

    model = train(model, chunks, tcfg.lr, tcfg.grad_clip, tcfg.remat_time, tcfg.remat_depth,
                  tcfg.n_epochs, len(raw_bytes), micro_batch=tcfg.micro_batch,
                  chunk_batch_size=tcfg.chunk_batch_size, seed=tcfg.seed, log_dir=tcfg.log_dir)

    from .checkpoint import save_model
    save_model(tcfg.log_dir, model)
    with open(os.path.join(tcfg.log_dir, "meta.json"), "w") as f:
        json.dump({"n_raw_bytes": len(raw_bytes), "timesteps_per_level": tsteps,
                    "file_chunk_bytes": file_chunk_bytes, "n_file_chunks": n_file_chunks}, f)
    print(f"Saved model to {tcfg.log_dir}/")


if __name__ == "__main__":
    main()
