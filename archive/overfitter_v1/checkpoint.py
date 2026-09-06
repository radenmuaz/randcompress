"""Save/load overfitter bundles: full model weights + config + range-coded stream.

Unlike randcompress/ (frozen base + trained adapter deltas), the whole SummTransformer
is trainable, so the bundle stores the entire state_dict.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

import torch

from .config import TrainConfig
from .summformer import Config as ModelConfig
from .summformer import SummTransformer


def save_model(ckpt_dir: str, model: SummTransformer, mcfg: ModelConfig, tcfg: TrainConfig) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump({"model": asdict(mcfg), "train": asdict(tcfg)}, f, indent=2)


def load_model(ckpt_dir: str) -> tuple[SummTransformer, ModelConfig, TrainConfig]:
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg_json = json.load(f)
    mcfg = ModelConfig(**cfg_json["model"])
    tcfg = TrainConfig(**cfg_json["train"])
    model = SummTransformer(mcfg)
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    return model, mcfg, tcfg


def save_bundle(ckpt_dir: str, model: SummTransformer, mcfg: ModelConfig, tcfg: TrainConfig,
                rc_stream: bytes, n_raw_bytes: int, seed_byte: int) -> None:
    save_model(ckpt_dir, model, mcfg, tcfg)

    with open(os.path.join(ckpt_dir, "rc_stream.bin"), "wb") as f:
        f.write(rc_stream)

    sha = hashlib.sha256(rc_stream).hexdigest()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    meta = {
        "n_raw_bytes":      n_raw_bytes,
        "seed_byte":        int(seed_byte),
        "rc_bytes":         len(rc_stream),
        "rc_stream_sha256": sha,
        "param_bytes":      param_bytes,
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_bundle(ckpt_dir: str):
    model, mcfg, tcfg = load_model(ckpt_dir)

    with open(os.path.join(ckpt_dir, "rc_stream.bin"), "rb") as f:
        rc_stream = f.read()
    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)

    actual = hashlib.sha256(rc_stream).hexdigest()
    if actual != meta["rc_stream_sha256"]:
        raise RuntimeError(
            f"rc_stream.bin corrupt: sha256 mismatch\n"
            f"  expected: {meta['rc_stream_sha256']}\n"
            f"  actual:   {actual}"
        )

    return model, mcfg, tcfg, rc_stream, meta
