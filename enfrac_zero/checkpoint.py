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

from .model import ByteFractalGen, ModelConfig, trainable_filter


def save_model(ckpt_dir: str, model: ByteFractalGen) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    trainable, _ = eqx.partition(model, trainable_filter(model))
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), trainable)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(asdict(model.cfg), f, indent=2)


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
