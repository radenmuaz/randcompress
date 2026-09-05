"""Overfit a baseline (no-HiRA) ByteFractalGen to one file. The training loop itself (chunking,
the eqx.filter_jit step, AdamW+warmup, epoch-boundary theoretical-bpb logging) is generic w.r.t.
which ByteFractalGen variant it drives, so it's reused as-is from enfrac.train -- only the model
class, its ModelConfig (no use_hira/hira_r), and the CLI differ here.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

from enfrac.config import TrainConfig, build_config, parse_configs
from enfrac.train import _Tee, make_chunks, train
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
    n_levels = len(patch_len_list) - 1
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
    n_chunks = chunks.shape[0]
    log_every = (max(1, round(float(tcfg.log_every) * n_chunks))
                 if "." in tcfg.log_every else int(tcfg.log_every))
    print(f"log_every={tcfg.log_every} -> {log_every} steps "
          f"({'epoch fraction' if '.' in tcfg.log_every else 'raw steps'})")
    if log_every <= 2:
        print(f"WARNING: log_every={log_every} is very frequent (prints nearly every step) "
              f"-- pass a float like 0.1 for 'every 10% of an epoch' if this wasn't intended")

    model = train(model, chunks, tcfg.steps, tcfg.lr, tcfg.warmup_steps, tcfg.grad_clip,
                  log_every, len(raw_bytes), filter_spec=trainable_filter(model),
                  per_device_batch=tcfg.per_device_batch)

    from .checkpoint import save_model
    save_model(tcfg.log_dir, model)
    with open(os.path.join(tcfg.log_dir, "meta.json"), "w") as f:
        json.dump({"n_raw_bytes": len(raw_bytes)}, f)
    print(f"Saved model to {tcfg.log_dir}/")


if __name__ == "__main__":
    main()
