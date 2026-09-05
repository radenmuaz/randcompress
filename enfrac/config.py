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
    dataset: str = "datasets/juz1.txt"
    log_dir: str = "logs/enfrac/run"
    steps: int = 3000
    lr: float = 3e-3
    warmup_steps: int = 50
    grad_clip: float = 1.0
    log_every: str = "100"
    seed: int = 0
    per_device_batch: int = 1   # chunks per device per step; multiplied by jax.local_device_count()


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
