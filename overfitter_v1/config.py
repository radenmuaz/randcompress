"""Config for overfitter: summformer architecture (Config, from summformer.py) + a small
training-args dataclass for the simple single-file overfit loop."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields

from .summformer import Config as ModelConfig
from .summformer import ROPE_PRESETS


@dataclass
class TrainConfig:
    dataset:      str   = "datasets/surat_al-fatihah.txt"
    log_dir:      str   = "runs/overfitter"
    steps:        int   = 3000
    lr:           float = 3e-3
    weight_decay: float = 0.0
    grad_clip:    float = 1.0
    warmup_steps: int   = 50
    check_every:  int   = 100
    seed:         int   = 0
    device:       str   = ""   # "" = auto (mps > cuda > cpu)
    # adamw      -- plain AdamW over every parameter (baseline/ablation).
    # sinkgd_lm  -- (default) hand-written SinkGD (Sinkhorn-normalized GD, see randcompress/
    #               train.py) for attention/MLP matrix weights; AdamW carved out for
    #               embed/head/extra_heads, since Sinkhorn row/col normalization assumes a
    #               weight matrix with meaningful row/column structure -- a fit for projection
    #               matrices, not an embedding table or a per-symbol readout.
    # sinkgd_all -- SinkGD for every parameter including embed/head, matching what
    #               randcompress/train.py actually does today (no special-casing there).
    optimizer:    str   = "sinkgd_lm"
    sinkgd_l:     int   = 5
    # 0 = whole file as one growing context each step (small files only -- O(L^2) attention).
    # >0 = stream the file in chunks of this many bytes through the incremental KV-cache
    # stepper, RNN-style: state persists across chunks, backprop truncated per chunk (TBPTT).
    # Needed for files too long for a single dense attention pass.
    tbptt_chunk_size: int = 0
    epochs:       int   = 5    # tbptt mode only: full passes over the file


def parse_configs(argv=None) -> tuple[ModelConfig, TrainConfig]:
    p = argparse.ArgumentParser(description="overfitter: overfit a single file with SummTransformer")

    # ── model (SummTransformer Config) ─────────────────────────────────────────
    p.add_argument("--Ks", default="32,32")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--fuse_n_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_kv_heads", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--qk_norm", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--rope_preset", type=str, default="qwen3", choices=list(ROPE_PRESETS))
    p.add_argument("--context_len", type=int, default=4096)
    p.add_argument("--attn_window", type=int, default=None)
    p.add_argument("--fuse_window", type=int, default=None)
    p.add_argument("--mtp_heads", type=int, default=1)
    p.add_argument("--mtp_weight", type=float, default=1.0)
    p.add_argument("--weight_tie", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--share_lm", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--share_fuse", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--pos_scheme", type=str, default="rope", choices=["rope", "none"])
    p.add_argument("--mlp_type", type=str, default="swiglu", choices=["swiglu", "mlp", "none"])
    p.add_argument("--use_bias", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--norm_type", type=str, default="rmsnorm", choices=["rmsnorm", "layernorm"])

    # ── training ─────────────────────────────────────────────────────────────
    for f in fields(TrainConfig):
        ftype = type(f.default)
        if ftype is bool:
            p.add_argument(f"--{f.name}", type=lambda x: x.lower() in ("1", "true", "yes"), default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=ftype, default=f.default)

    args = p.parse_args(argv)
    Ks = tuple(int(x) for x in args.Ks.split(",")) if isinstance(args.Ks, str) else tuple(args.Ks)

    mcfg = ModelConfig(
        Ks=Ks, d_model=args.d_model, n_layers=args.n_layers, fuse_n_layers=args.fuse_n_layers,
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads, head_dim=args.head_dim, qk_norm=args.qk_norm,
        mlp_mult=args.mlp_mult, rope_preset=args.rope_preset, context_len=args.context_len,
        attn_window=args.attn_window, fuse_window=args.fuse_window, input_preset=8,
        mtp_heads=args.mtp_heads, mtp_weight=args.mtp_weight, weight_tie=args.weight_tie,
        share_lm=args.share_lm, share_fuse=args.share_fuse,
        pos_scheme=args.pos_scheme, mlp_type=args.mlp_type,
    )
    tcfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    return mcfg, tcfg
