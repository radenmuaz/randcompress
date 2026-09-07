"""Overfit a baseline (no-HiRA) ByteFractalGen to one file. The training loop itself (full
parallel teacher-forced forward+backward over the whole level-0 timestep sequence each epoch,
remat_time/remat_depth for memory -- see enfrac/train.py's and enfrac/model.py's module
docstrings) is generic w.r.t. which ByteFractalGen variant it drives, so it's reused as-is from
enfrac.train -- only the model class, its ModelConfig (no use_hira/hira_r), and the CLI differ.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

from enfrac.config import TrainConfig, build_config, parse_configs
from enfrac.train import _Tee, make_chunks, timesteps_per_level, train
from enfrac.tokenizer import load_bytes
from .model import ByteFractalGen, ModelConfig


def main() -> None:
    parser = parse_configs(ModelConfig)
    parser.add_argument("--d_model", type=int, default=None, help="broadcast uniformly to every level")
    parser.add_argument("--n_layers", type=int, default=None, help="broadcast uniformly to every level")
    parser.add_argument("--n_heads", type=int, default=None, help="broadcast uniformly to every level")
    parser.add_argument("--mlp_mult", type=int, default=None, help="broadcast uniformly to every level")
    args = parser.parse_args()

    train_overrides = {f.name: getattr(args, f.name, None) for f in dataclasses.fields(TrainConfig)}
    tcfg = build_config(TrainConfig, args.config, ("train", "train_config"), train_overrides)

    patch_len_list = getattr(args, "patch_len_list", None) or ModelConfig.patch_len_list
    n_levels = len(patch_len_list)
    broadcast = {}
    if args.d_model is not None:
        broadcast["d_model_list"] = (args.d_model,) * n_levels
    if args.n_layers is not None:
        broadcast["n_layers_list"] = (args.n_layers,) * n_levels
    if args.n_heads is not None:
        broadcast["n_heads_list"] = (args.n_heads,) * n_levels
    if args.mlp_mult is not None:
        broadcast["mlp_mult_list"] = (args.mlp_mult,) * n_levels

    model_overrides = {f.name: getattr(args, f.name, None) for f in dataclasses.fields(ModelConfig)}
    model_overrides.update(broadcast)
    if tcfg.seed is not None and model_overrides.get("seed") is None:
        model_overrides["seed"] = tcfg.seed
    mcfg = build_config(ModelConfig, args.config, ("model", "model_config"), model_overrides)

    os.makedirs(tcfg.log_dir, exist_ok=True)
    log_file = open(os.path.join(tcfg.log_dir, "train.log"), "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"logging to {os.path.join(tcfg.log_dir, 'train.log')} -- tail -f it")

    raw_bytes = load_bytes(tcfg.dataset)
    print(f"dataset={tcfg.dataset}  n_bytes={len(raw_bytes)}  patch_len_list={mcfg.patch_len_list}")

    import equinox as eqx
    import jax
    from .model import trainable_filter

    model = ByteFractalGen(mcfg)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
    n_trainable = sum(x.size for x in jax.tree_util.tree_leaves(eqx.partition(model, trainable_filter(model))[0]))
    print(f"params={n_params:,} (trainable={n_trainable:,})  seed={mcfg.seed}  seq_lens={model.seq_lens}")

    chunks = make_chunks(raw_bytes, mcfg.patch_len_list[0])

    tsteps = timesteps_per_level(model, chunks.shape[0])
    print("timesteps per level (total patches of that level's size, whole file): " +
          "  ".join(f"level{l}={t:,}" for l, t in enumerate(tsteps)))

    model = train(model, chunks, tcfg.lr, tcfg.grad_clip, tcfg.remat_time, tcfg.remat_depth,
                  tcfg.n_epochs, len(raw_bytes), filter_spec=trainable_filter(model))

    from .checkpoint import save_model
    save_model(tcfg.log_dir, model)
    with open(os.path.join(tcfg.log_dir, "meta.json"), "w") as f:
        json.dump({"n_raw_bytes": len(raw_bytes), "timesteps_per_level": tsteps}, f)
    print(f"Saved model to {tcfg.log_dir}/")


if __name__ == "__main__":
    main()
