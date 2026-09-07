"""Config system: dataclass defaults -> optional python-file overrides -> CLI kwarg overrides.

A config file is a plain .py module. It's `exec`'d and searched (in order) for a `model`/
`model_config` value and a `train`/`train_config` value -- each may be either a dict of field
overrides or an already-built ModelConfig/TrainConfig instance (`dataclasses.asdict`'d). Only
fields present in the target dataclass are accepted; anything else raises immediately (typo
protection). CLI kwargs (parsed with defaults of None so "unset" is distinguishable from
"explicitly set to falsy") take precedence over the file, which takes precedence over the
dataclass's own defaults.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Level 0 is a real causal AR transformer whose sequence IS the file's level-0 timesteps --
    a "level-0 timestep" means one PATCH of patch_len_list[0] bytes (fixed across the whole
    sequence), NOT one byte. E.g. quran-uthmani.txt at patch_len_list[0]=1024 has 1329 level-0
    timesteps (ceil(1,359,946 / 1024)), each one a 1024-byte patch -- not 1,359,946 single-byte
    timesteps. Training is teacher-forced and FULLY PARALLEL across those (patch-granularity)
    timesteps -- one causally-masked attention computation, like training any GPT-style
    transformer on a long sequence. No recurrence, no per-step state carry, no stop_gradient
    anywhere.

    remat_time / remat_depth control memory via jax.checkpoint (remat), NOT approximation --
    remat recomputes forward activations during backward instead of storing them, so gradients
    stay exact regardless of how finely either axis is split (unlike TBPTT-style stop_gradient
    truncation, which this deliberately does NOT use). Both are strings, parsed like log_every:
    a float ("0.5") is a FRACTION of that axis per remat group, an int ("512") is an EXACT count
    -- level-0 timesteps (patches, not bytes) for remat_time, transformer layers for remat_depth
    -- "1.0" (default, both axes) means one remat group spanning the whole axis. remat_time
    chunking works by accumulating a running (real, non-detached) KV cache across time-groups --
    each group's queries attend against the full accumulated K/V, exactly reproducing one-shot
    full attention, just computed (and checkpointed) in pieces for memory. remat_depth
    independently checkpoints groups of transformer layers within a Trunk.

    n_epochs: full passes over the whole file (each a single full-sequence forward+backward)."""
    dataset: str = "datasets/juz1.txt"
    log_dir: str = "logs/enfrac/run"
    lr: float = 1e-3                  # fixed, no warmup schedule -- see class docstring
    grad_clip: float = 1.0
    remat_time: str = "1.0"           # fraction OR exact count of level-0 timesteps (patches,
                                        # not bytes) per remat group
    remat_depth: str = "1.0"          # fraction OR exact count of transformer layers per remat group
    n_epochs: int = 5                 # full passes over the whole file
    seed: int = 0


def load_config_file(path: str):
    spec = importlib.util.spec_from_file_location("enfrac_user_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract(mod, names: tuple[str, ...]):
    for name in names:
        if hasattr(mod, name):
            source = getattr(mod, name)
            if dataclasses.is_dataclass(source) and not isinstance(source, type):
                return dataclasses.asdict(source)
            if isinstance(source, dict):
                return dict(source)
            raise TypeError(f"config file attribute {name!r} must be a dict or dataclass instance")
    return {}


def build_config(cls, file_path: str | None, file_names: tuple[str, ...], overrides: dict | None = None):
    """defaults(cls) < file_path's dict/dataclass (first attr in file_names found) < overrides
    (non-None entries only)."""
    values: dict = {}
    if file_path:
        mod = load_config_file(file_path)
        values.update(_extract(mod, file_names))
    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - field_names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} field(s): {sorted(unknown)}")
    return cls(**values)


def add_dataclass_args(parser: argparse.ArgumentParser, cls) -> None:
    """Adds one --<field> per dataclass field, default=None (so build_config can tell 'unset'
    apart from 'explicitly passed'), type inferred from the field's default's type (tuples are
    parsed as comma-separated; bools accept true/false)."""
    for f in dataclasses.fields(cls):
        if f.name in {a.dest for a in parser._actions}:
            continue  # already added (e.g. shared between ModelConfig/TrainConfig)
        default = f.default
        if isinstance(default, bool) or f.type == "bool":
            parser.add_argument(f"--{f.name}", type=lambda x: str(x).lower() != "false", default=None)
        elif isinstance(default, tuple):
            parser.add_argument(f"--{f.name}", type=lambda x: tuple(int(v) for v in x.split(",")), default=None)
        elif isinstance(default, int):
            parser.add_argument(f"--{f.name}", type=int, default=None)
        elif isinstance(default, float):
            parser.add_argument(f"--{f.name}", type=float, default=None)
        else:
            parser.add_argument(f"--{f.name}", type=str, default=None)


def parse_configs(model_cls, model_file_names: tuple[str, ...] = ("model", "model_config")):
    """Standard CLI: --config <path.py> plus one --<field> flag per ModelConfig/TrainConfig
    field. Returns (model_cfg, train_cfg, args) -- `args` carries anything not part of either
    dataclass (e.g. --device, --batch_size for compress/decompress scripts)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                         help="python file defining `model`/`model_config` and/or "
                              "`train`/`train_config` (dict or dataclass instance)")
    add_dataclass_args(parser, model_cls)
    add_dataclass_args(parser, TrainConfig)
    return parser
