"""Save/load a baseline ByteFractalGen checkpoint: trainable leaves + config.

Unlike the PEFT/main package, every trainable leaf here is essentially "the whole model" (there's
no frozen adapter base) -- only byte_embed's frozen table is excluded, and it's cheap and fully
deterministic to regenerate (make_byte_embedding() takes no seed dependency)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import equinox as eqx

from .model import ByteFractalGen, ModelConfig, trainable_filter


def save_model(ckpt_dir: str, model: ByteFractalGen) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    trainable, _ = eqx.partition(model, trainable_filter(model))
    eqx.tree_serialise_leaves(os.path.join(ckpt_dir, "model.eqx"), trainable)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(asdict(model.cfg), f, indent=2)


_TUPLE_FIELDS = ("patch_len_list", "d_model_list", "n_layers_list", "n_heads_list", "mlp_mult_list")


def load_model(ckpt_dir: str) -> ByteFractalGen:
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        raw = json.load(f)
    for k in _TUPLE_FIELDS:
        raw[k] = tuple(raw[k])
    cfg = ModelConfig(**raw)
    skeleton = ByteFractalGen(cfg)
    trainable_skeleton, static = eqx.partition(skeleton, trainable_filter(skeleton))
    trainable = eqx.tree_deserialise_leaves(os.path.join(ckpt_dir, "model.eqx"), trainable_skeleton)
    return eqx.combine(trainable, static)
