"""ModelConfig + TrainConfig with argparse CLI parsing."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, asdict
from typing import Optional


@dataclass
class ModelConfig:
    model:       str   = "msrnn"    # msrnn | transformer | ttt_rnn | deltanet
    d_model:     int   = 64
    num_heads:   int   = 8
    num_layers:  int   = 4          # overridden by len(block_map) for msrnn
    block_map:   str   = "smmm"     # msrnn only: s=sRNN, m=mLSTM
    stride_map:  str   = "1111"     # msrnn only: temporal stride per layer
    lora_r:      int   = 4          # HiRA / LoRA rank
    use_hira:    bool  = True       # False → plain LoRA baseline (same param count)
    rope_scale:  float = 1.0        # transformer only (YaRN linear scale)
    power_p:     int   = 2          # msrnn mLSTM symmetric power map degree
    seed:        int   = 0


@dataclass
class TrainConfig:
    dataset:            str   = "datasets/juz1.txt"
    log_dir:            str   = ""
    input_bits:         int   = 8
    output_bits:        int   = 8
    output_heads:       int   = 1
    vocab_size:         int   = 256
    pad_token:          int   = 0
    segment_size:       int   = 1024
    max_iter_per_phase: int   = 10000
    check_every:        int   = 100
    learning_rate:      float = 1e-2
    weight_decay:       float = 0.0
    grad_clip_norm:     float = 1e2
    optimizer:          str   = "sinkgd"   # sinkgd | adamw | sgd
    sgd_momentum:       float = 0.9
    sinkgd_l:           int   = 5
    use_agd_loss:       bool  = False
    agd_detach_weights: bool  = True
    gen_seed_len:       int   = 16
    residual_budget:    float = 0.0
    dtype:              str   = "float32"
    # compress: precompute HiRA-applied weights once (faster, uses more memory)
    precompute_weights: bool  = True


def _add_dataclass_args(parser: argparse.ArgumentParser, dc_cls):
    for f in fields(dc_cls):
        name = f"--{f.name}"
        default = f.default
        ftype = f.type if isinstance(f.type, type) else type(default)
        if ftype is bool:
            parser.add_argument(name, type=lambda x: x.lower() in ("1","true","yes"),
                                default=default, metavar="BOOL")
        else:
            parser.add_argument(name, type=ftype, default=default)


def parse_configs(argv=None) -> tuple[ModelConfig, TrainConfig]:
    parser = argparse.ArgumentParser(description="randcompress")
    parser.add_argument("--no_hira", action="store_true",
                        help="Use plain LoRA baseline instead of HiRA")
    _add_dataclass_args(parser, ModelConfig)
    _add_dataclass_args(parser, TrainConfig)
    args = parser.parse_args(argv)

    model_fields = {f.name for f in fields(ModelConfig)}
    train_fields = {f.name for f in fields(TrainConfig)}

    md = {f: getattr(args, f) for f in model_fields}
    td = {f: getattr(args, f) for f in train_fields}

    if args.no_hira:
        md["use_hira"] = False

    # Derive num_layers from block_map for msrnn
    if md["model"] == "msrnn":
        md["num_layers"] = len(md["block_map"])
        # Auto-pad stride_map
        sm = md["stride_map"]
        nl = md["num_layers"]
        if len(sm) < nl:
            sm = sm + "1" * (nl - len(sm))
        md["stride_map"] = sm[:nl]

    td["vocab_size"] = 2 ** td["input_bits"]

    return ModelConfig(**md), TrainConfig(**td)
