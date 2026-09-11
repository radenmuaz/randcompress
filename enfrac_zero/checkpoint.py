"""Save/load a baseline ByteFractalGen checkpoint: trainable leaves + config.

Unlike the PEFT/main package, every trainable leaf here is essentially "the whole model" (there's
no frozen adapter base) -- only byte_embed's frozen table is excluded, and it's cheap and fully
deterministic to regenerate (make_byte_embedding() takes no seed dependency)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .model import ByteFractalGen, ModelConfig, trainable_filter


def save_model(ckpt_dir: str, model: ByteFractalGen) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    trainable, _ = eqx.partition(model, trainable_filter(model))
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), trainable)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(asdict(model.cfg), f, indent=2)


def has_resumable_checkpoint(ckpt_dir: str) -> bool:
    """True iff save_full_checkpoint() has written a complete resumable state to ckpt_dir --
    model.eqx/config.json alone (from plain save_model, e.g. a prior FINISHED run) don't count,
    since opt_state/rng/epoch are also required to resume bit-for-bit (see train()'s docstring)."""
    return all(os.path.exists(os.path.join(ckpt_dir, f))
               for f in ("model.eqx", "config.json", "opt_state.eqx", "rng_state.json", "progress.json"))


def save_full_checkpoint(ckpt_dir: str, model: ByteFractalGen, opt_state, rng: np.random.Generator,
                          epoch: int) -> None:
    """Saves everything needed to resume training bit-for-bit: trainable weights (save_model),
    the optimizer state (adamw's m/v moments + clip_by_global_norm's own state), the minibatch
    RNG's exact stream position, and which epoch just finished. Called once per epoch (see
    train()'s on_epoch_end) -- cheap relative to an epoch's own compute, and means an interrupted
    run loses at most one epoch of progress, not the whole thing."""
    save_model(ckpt_dir, model)
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "opt_state.eqx"), opt_state)
    with open(os.path.join(ckpt_dir, "rng_state.json"), "w") as f:
        json.dump(rng.bit_generator.state, f)
    with open(os.path.join(ckpt_dir, "progress.json"), "w") as f:
        json.dump({"epoch": epoch}, f)


def load_full_checkpoint(ckpt_dir: str, lr: float, grad_clip: float):
    """Inverse of save_full_checkpoint(). lr/grad_clip MUST match what the checkpoint was
    trained with -- build_optimizer(lr, grad_clip) is used to construct the opt_state
    deserialization skeleton, and a mismatched lr/grad_clip would silently desync adamw's
    internal step-count-driven bias-correction from the loaded moments. Returns
    (model, opt_state, rng, start_epoch) where start_epoch = the saved epoch + 1 (i.e. train()
    should resume from there, not repeat the last completed epoch)."""
    from enfrac.train import build_optimizer

    model = load_model(ckpt_dir)
    trainable, _ = eqx.partition(model, trainable_filter(model))
    opt = build_optimizer(lr, grad_clip)
    opt_state_skeleton = opt.init(trainable)
    opt_state = eqx.tree_deserialise_leaves(os.path.join(ckpt_dir, "opt_state.eqx"), opt_state_skeleton)

    with open(os.path.join(ckpt_dir, "rng_state.json")) as f:
        state = json.load(f)
    rng = np.random.default_rng()
    rng.bit_generator.state = state

    with open(os.path.join(ckpt_dir, "progress.json")) as f:
        epoch = json.load(f)["epoch"]

    return model, opt_state, rng, epoch + 1


_TUPLE_FIELDS = ("patch_len_list", "d_model_list", "n_layers_list", "n_heads_list", "mlp_mult_list")


def load_model(ckpt_dir: str, dtype=None) -> ByteFractalGen:
    """dtype: if given (e.g. jnp.float64), the loaded model is cast to it via cast_dtype() after
    loading -- see collect_logits_fp64()'s docstring for why dtype is a correctness invariant
    like device/batch_size, not a free knob (compress/decompress must agree). Checkpoints are
    always SAVED in float32 (the dtype training ran in) regardless of this argument -- the
    deserialization skeleton is force-cast to float32 before loading so this works correctly
    even when the caller has jax_enable_x64 on (jax.random's default float dtype follows x64,
    which would otherwise make the skeleton disagree with what's actually on disk)."""
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        raw = json.load(f)
    for k in _TUPLE_FIELDS:
        raw[k] = tuple(raw[k])
    cfg = ModelConfig(**raw)
    skeleton = ByteFractalGen(cfg)
    skeleton = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, skeleton)
    trainable_skeleton, static = eqx.partition(skeleton, trainable_filter(skeleton))
    trainable = eqx.tree_deserialise_leaves(os.path.join(ckpt_dir, "model.eqx"), trainable_skeleton)
    model = eqx.combine(trainable, static)
    if dtype is not None:
        model = model.cast_dtype(dtype)
    return model
