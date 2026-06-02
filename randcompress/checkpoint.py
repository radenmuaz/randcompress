"""Save and load parameter bundles."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict

import numpy as np
import torch
from torch import Tensor

from .config import ModelConfig, TrainConfig


def save_checkpoint(ckpt_dir: str, adapters: dict[str, Tensor],
                    mcfg: ModelConfig, tcfg: TrainConfig,
                    seg_idx: int = 0, phase: str = ""):
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {**asdict(mcfg), **asdict(tcfg), "seg_idx": seg_idx, "phase": phase}
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    params_np = {k: v.detach().cpu().numpy() for k, v in adapters.items()}
    with open(os.path.join(ckpt_dir, "params.pkl"), "wb") as f:
        pickle.dump(params_np, f)


def load_checkpoint(ckpt_dir: str) -> tuple[ModelConfig, TrainConfig, dict[str, Tensor]]:
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        meta = json.load(f)
    mc_fields = set(ModelConfig.__dataclass_fields__)
    tc_fields = set(TrainConfig.__dataclass_fields__)
    mcfg = ModelConfig(**{k: v for k, v in meta.items() if k in mc_fields})
    tcfg = TrainConfig(**{k: v for k, v in meta.items() if k in tc_fields})
    with open(os.path.join(ckpt_dir, "params.pkl"), "rb") as f:
        params_np = pickle.load(f)
    adapters = {k: torch.tensor(v) for k, v in params_np.items()}
    return mcfg, tcfg, adapters


def save_bundle(ckpt_dir: str, adapters: dict[str, Tensor],
                mcfg: ModelConfig, tcfg: TrainConfig,
                rc_stream: bytes, n_raw_bytes: int, seed_token: int,
                T_valid: int = 0, n_syms: int = 0):
    os.makedirs(ckpt_dir, exist_ok=True)
    save_checkpoint(ckpt_dir, adapters, mcfg, tcfg)

    rc_path = os.path.join(ckpt_dir, "rc_stream.bin")
    with open(rc_path, "wb") as f:
        f.write(rc_stream)

    sha = hashlib.sha256(rc_stream).hexdigest()
    param_bytes      = sum(v.numel() * v.element_size() for v in adapters.values())
    param_disk_bytes = os.path.getsize(os.path.join(ckpt_dir, "params.pkl"))

    meta = {
        "n_raw_bytes":      n_raw_bytes,
        "T_valid":          T_valid,
        "n_syms":           n_syms if n_syms else T_valid,
        "seed_token":       int(seed_token),
        "rc_bytes":         len(rc_stream),
        "rc_stream_sha256": sha,
        "param_bytes":      param_bytes,
        "param_disk_bytes": param_disk_bytes,
        "input_bits":       tcfg.input_bits,
        "output_bits":      tcfg.output_bits,
        "output_heads":     tcfg.output_heads,
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_bundle(ckpt_dir: str):
    mcfg, tcfg, adapters = load_checkpoint(ckpt_dir)

    rc_path = os.path.join(ckpt_dir, "rc_stream.bin")
    with open(rc_path, "rb") as f:
        rc_stream = f.read()

    with open(os.path.join(ckpt_dir, "meta.json")) as f:
        meta = json.load(f)

    # Verify integrity
    if "rc_stream_sha256" in meta:
        actual = hashlib.sha256(rc_stream).hexdigest()
        if actual != meta["rc_stream_sha256"]:
            raise RuntimeError(
                f"rc_stream.bin corrupt: sha256 mismatch\n"
                f"  expected: {meta['rc_stream_sha256']}\n"
                f"  actual:   {actual}"
            )

    return mcfg, tcfg, adapters, rc_stream, meta
