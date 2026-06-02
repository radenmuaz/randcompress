"""
Train Causal Transformer (YaRN RoPE + SwiGLU) with HiRA + SinkGD.

Usage:
    uv run python examples/train_transformer.py
    uv run python examples/train_transformer.py \
        --dataset datasets/surat_al-fatihah.txt \
        --max_iter_per_phase 500 --check_every 50
    uv run python examples/train_transformer.py \
        --d_model 64 --num_layers 4 --rope_scale 1.0 --log_dir runs/transformer_large
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from randcompress.config import parse_configs
from randcompress.models import get_model
from randcompress.train import CurriculumTrainer, _Tee
from randcompress.tokenizer import load_bytes
from randcompress.checkpoint import save_checkpoint


def main():
    argv = sys.argv[1:]
    if "--model" not in argv:
        argv = ["--model", "transformer"] + argv

    mcfg, tcfg = parse_configs(argv)
    script  = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = tcfg.log_dir or os.path.join("runs", script)
    os.makedirs(log_dir, exist_ok=True)

    log_file = open(os.path.join(log_dir, "train.log"), "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"Model : transformer  d={mcfg.d_model}  nh={mcfg.num_heads}"
          f"  layers={mcfg.num_layers}  r={mcfg.lora_r}  rope_scale={mcfg.rope_scale}")
    print(f"Data  : {tcfg.dataset}")
    print(f"Opt   : {tcfg.optimizer}  lr={tcfg.learning_rate}")

    device   = torch.device("cpu")
    model    = get_model(mcfg, tcfg)
    frozen   = model.init_frozen(mcfg.seed)
    adapters = model.init_adapters(mcfg.seed)
    n_train  = sum(v.numel() for v in adapters.values() if v.requires_grad)
    print(f"Trainable params: {n_train/1e3:.1f}K")

    raw_bytes = load_bytes(tcfg.dataset)
    trainer   = CurriculumTrainer(model, frozen, adapters, mcfg, tcfg, device, log_dir)
    trainer.run(raw_bytes)

    save_checkpoint(os.path.join(log_dir, "ckpt_last"), adapters, mcfg, tcfg)
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
