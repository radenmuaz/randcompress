"""
Train Causal Transformer tuned for ~1M token files.

Config chosen for quran-uthmani.txt (1.36M bytes = 1.36M tokens):

  kv_window=2048, attn_dilation=4, num_layers=4:
    effective context per layer = 1 + (2048-1) × 4 = 8,189
    stacked 4 layers            = 1 + 4 × 2047 × 4 = 32,753 tokens ≈ 32K

  Memory per step (d=64, NH=8, DH=8, 4 layers):
    KV cache = 2 × kv_window × dilation × NH × DH × 4B × num_layers
             = 2 × 2048 × 4 × 8 × 8 × 4 × 4 ≈ 16 MB — easily manageable

  Sinks: 1 zero-score + 1 trainable (prepended, never evicted)

Usage:
    uv run python examples/train_transformer_1m.py
    uv run python examples/train_transformer_1m.py \\
        --dataset datasets/quran-uthmani.txt \\
        --max_iter_per_phase 500 --log_dir runs/transformer_1m
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from randcompress.config import parse_configs
from randcompress.models import get_model
from randcompress.train import CurriculumTrainer, _Tee
from randcompress.tokenizer import load_bytes
from randcompress.checkpoint import save_checkpoint


# Defaults tuned for 1M tokens
_DEFAULTS_1M = [
    "--model",          "transformer",
    "--d_model",        "64",
    "--num_heads",      "8",
    "--num_layers",     "4",
    "--lora_r",         "4",
    "--kv_window",      "2048",    # attention slots per layer
    "--attn_dilation",  "4",       # attend every 4th past position
    "--n_sinks_zero",   "1",       # constant-0 sink (no params)
    "--n_sinks_train",  "1",       # trainable sink
    "--rope_scale",     "4.0",     # YaRN: stretch positions for long context
    "--segment_size",   "1024",
    "--learning_rate",  "1e-2",
    "--optimizer",      "sinkgd",
    "--sinkgd_l",       "5",
]


def main():
    argv = sys.argv[1:]
    # Prepend 1M defaults; CLI overrides take effect because argparse uses last value
    merged = _DEFAULTS_1M + argv

    mcfg, tcfg = parse_configs(merged)
    script  = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = tcfg.log_dir or os.path.join("runs", script)
    os.makedirs(log_dir, exist_ok=True)

    log_file = open(os.path.join(log_dir, "train.log"), "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    d   = mcfg.d_model; W = mcfg.kv_window; dil = mcfg.attn_dilation; L = mcfg.num_layers
    ctx_per_layer = 1 + (W - 1) * dil if W > 0 else float("inf")
    ctx_stacked   = 1 + L * (W - 1) * dil if W > 0 else float("inf")

    print(f"Model : transformer (1M tuned)")
    print(f"  d={d}  nh={mcfg.num_heads}  layers={L}  r={mcfg.lora_r}")
    print(f"  kv_window={W}  dilation={dil}  rope_scale={mcfg.rope_scale}")
    print(f"  sinks: {mcfg.n_sinks_zero} zero + {mcfg.n_sinks_train} trainable")
    print(f"  effective context: {ctx_per_layer:.0f} / layer  →  {ctx_stacked:.0f} stacked")
    print(f"Data  : {tcfg.dataset}   Opt: {tcfg.optimizer}  lr={tcfg.learning_rate}")

    device   = torch.device("cpu")
    model    = get_model(mcfg, tcfg)
    frozen   = model.init_frozen(mcfg.seed)
    adapters = model.init_adapters(mcfg.seed)
    n_train  = sum(v.numel() for v in adapters.values() if v.requires_grad)
    print(f"Trainable params: {n_train/1e3:.1f}K")

    raw_bytes = load_bytes(tcfg.dataset)
    print(f"Dataset: {len(raw_bytes)/1e6:.3f}MB = {len(raw_bytes)} bytes")

    trainer = CurriculumTrainer(model, frozen, adapters, mcfg, tcfg, device, log_dir)
    trainer.run(raw_bytes)

    save_checkpoint(os.path.join(log_dir, "ckpt_last"), adapters, mcfg, tcfg)
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
