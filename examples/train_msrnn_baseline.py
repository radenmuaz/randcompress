"""
Train MsRNN with plain LoRA baseline (no HiRA Hadamard) + AdamW.

Same param count as HiRA but W = W₀ + B·A instead of W = W₀ + W₀⊙(B·A).
Good for ablation: does the Hadamard structure in HiRA help?

Usage:
    uv run python examples/train_msrnn_baseline.py
    uv run python examples/train_msrnn_baseline.py \
        --dataset datasets/surat_al-fatihah.txt \
        --max_iter_per_phase 500 --check_every 50
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from randcompress.config import parse_configs
from randcompress.models import get_model
from randcompress.train import CurriculumTrainer, _Tee
from randcompress.tokenizer import load_bytes
from randcompress.checkpoint import save_checkpoint


def main():
    # Force no_hira + adamw defaults, but allow CLI overrides
    argv = sys.argv[1:]
    if "--no_hira" not in argv:
        argv = ["--no_hira"] + argv
    if "--optimizer" not in argv:
        argv = ["--optimizer", "adamw"] + argv

    mcfg, tcfg = parse_configs(argv)

    script  = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = tcfg.log_dir or os.path.join("runs", script)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"Model : {mcfg.model}  block_map={mcfg.block_map}  d={mcfg.d_model}"
          f"  r={mcfg.lora_r}  hira={mcfg.use_hira}  (BASELINE: plain LoRA)")
    print(f"Opt   : {tcfg.optimizer}  lr={tcfg.learning_rate}")
    print(f"Data  : {tcfg.dataset}")

    device   = torch.device("cpu")
    model    = get_model(mcfg, tcfg)
    frozen   = model.init_frozen(mcfg.seed)
    adapters = model.init_adapters(mcfg.seed)

    raw_bytes = load_bytes(tcfg.dataset)
    trainer   = CurriculumTrainer(model, frozen, adapters, mcfg, tcfg, device, log_dir)
    trainer.run(raw_bytes)

    save_checkpoint(os.path.join(log_dir, "ckpt_last"), adapters, mcfg, tcfg)
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
