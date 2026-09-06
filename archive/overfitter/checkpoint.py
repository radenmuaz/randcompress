"""Save/load a ByteFractalGen checkpoint: full state_dict + config."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import torch

from .model import ByteFractalGen, FractalConfig


def save_model(ckpt_dir: str, model: ByteFractalGen) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(asdict(model.cfg), f, indent=2)


_TUPLE_FIELDS = ("patch_len_list", "d_model_list", "n_layers_list", "n_heads_list", "mlp_mult_list")


def load_model(ckpt_dir: str) -> ByteFractalGen:
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        raw = json.load(f)
    for k in _TUPLE_FIELDS:
        raw[k] = tuple(raw[k])
    model = ByteFractalGen(FractalConfig(**raw))
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    return model
