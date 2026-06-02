"""
Train MsRNN with HiRA adapters + SinkGD on a single file.

Default: datasets/juz1.txt, d_model=64, block_map=smmm, lora_r=4, sinkgd.

Usage:
    uv run python examples/train_msrnn.py
    uv run python examples/train_msrnn.py --dataset datasets/surat_al-fatihah.txt \
        --max_iter_per_phase 500 --check_every 50
    uv run python examples/train_msrnn.py --d_model 128 --block_map smmmm \
        --lora_r 8 --log_dir runs/large
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
    mcfg, tcfg = parse_configs()

    script = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = tcfg.log_dir or os.path.join("runs", script)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train.log")
    log_file = open(log_path, "w", buffering=1)
    tee_out  = _Tee(sys.__stdout__, log_file)
    tee_err  = _Tee(sys.__stderr__, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err

    import json
    from dataclasses import asdict
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({**asdict(mcfg), **asdict(tcfg)}, f, indent=2)

    print(f"Model : {mcfg.model}  block_map={mcfg.block_map}  d={mcfg.d_model}"
          f"  nh={mcfg.num_heads}  r={mcfg.lora_r}  hira={mcfg.use_hira}")
    print(f"Data  : {tcfg.dataset}")
    print(f"Opt   : {tcfg.optimizer}  lr={tcfg.learning_rate}")
    print(f"Log   : {log_path}")

    device = torch.device("cpu")
    model  = get_model(mcfg, tcfg)
    frozen  = model.init_frozen(mcfg.seed)
    adapters = model.init_adapters(mcfg.seed)

    n_frozen  = sum(v.numel() for v in frozen.values())
    n_train   = sum(v.numel() for v in adapters.values() if v.requires_grad)
    print(f"Frozen params : {n_frozen/1e6:.3f}M")
    print(f"Trainable params: {n_train/1e6:.3f}M  ({n_train*4/1024:.1f} KB at fp32)")

    raw_bytes = load_bytes(tcfg.dataset)
    print(f"Dataset: {len(raw_bytes)} bytes")

    trainer = CurriculumTrainer(model, frozen, adapters, mcfg, tcfg, device, log_dir)
    trainer.run(raw_bytes)

    save_checkpoint(os.path.join(log_dir, "ckpt_last"), adapters, mcfg, tcfg)
    print(f"\nCheckpoint: {log_dir}/ckpt_last/")

    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
