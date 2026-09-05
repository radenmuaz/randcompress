"""Overfit a ByteFractalGen to one file. Each patch_len_list[0]-byte chunk is fully independent
(root_cond is a fixed constant, no state carried across chunks -- unlike overfitter/summformer's
streaming design), so training is just: iterate chunks, forward, backward, AdamW step.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from overfitter.tokenizer import load_bytes

from model import ByteFractalGen, FractalConfig


class _Tee:
    """stdout/stderr + file, eager flush -- matches overfitter/train.py's own _Tee."""
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    @property
    def encoding(self):
        return getattr(self.files[0], "encoding", "utf-8")


def make_chunks(raw_bytes: np.ndarray, patch_len: int) -> np.ndarray:
    n = len(raw_bytes)
    n_chunks = math.ceil(n / patch_len)
    padded = np.zeros(n_chunks * patch_len, dtype=np.uint8)
    padded[:n] = raw_bytes
    return padded.reshape(n_chunks, patch_len)


def train(model: ByteFractalGen, chunks: np.ndarray, device: torch.device, steps: int,
         lr: float, warmup_steps: int, grad_clip: float, check_every: int) -> None:
    model.to(device)
    model.train()
    ctx = torch.tensor(chunks.astype(np.int64), device=device)   # [n_chunks, P0]
    n_chunks = ctx.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    t0 = time.perf_counter()
    pbar = tqdm(range(1, steps + 1), desc="fractalgen", dynamic_ncols=True)
    for step in pbar:
        lr_t = lr * min(1.0, step / max(1, warmup_steps))
        for g in opt.param_groups:
            g["lr"] = lr_t

        idx = (step - 1) % n_chunks
        chunk = ctx[idx: idx + 1]
        loss, metrics = model(chunk)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step % check_every == 0 or step == steps:
            acc = metrics["byte_acc"].item()
            bpb = metrics["bpb"].item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}", bpb=f"{bpb:.3f}")

    pbar.close()
    print(f"[fractalgen train] {steps} steps over {n_chunks} chunks  {time.perf_counter()-t0:.1f}s")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--patch_len_list", default="1024,128,16,1")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=2)
    p.add_argument("--qk_norm", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--freeze_byte_embed", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--check_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = open(os.path.join(args.log_dir, "train.log"), "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"logging to {os.path.join(args.log_dir, 'train.log')} -- tail -f it")

    patch_len_list = tuple(int(x) for x in args.patch_len_list.split(","))
    raw_bytes = load_bytes(args.dataset)
    print(f"dataset={args.dataset}  n_bytes={len(raw_bytes)}  patch_len_list={patch_len_list}  device={device}")

    cfg = FractalConfig(patch_len_list=patch_len_list, d_model=args.d_model, n_layers=args.n_layers,
                        n_heads=args.n_heads, mlp_mult=args.mlp_mult, qk_norm=args.qk_norm,
                        freeze_byte_embed=args.freeze_byte_embed)
    model = ByteFractalGen(cfg)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"params={n_params:,}  seq_lens={model.seq_lens}")

    chunks = make_chunks(raw_bytes, patch_len_list[0])
    train(model, chunks, device, args.steps, args.lr, args.warmup_steps, args.grad_clip, args.check_every)

    from checkpoint import save_model
    save_model(args.log_dir, model)
    with open(os.path.join(args.log_dir, "meta.json"), "w") as f:
        import json
        json.dump({"n_raw_bytes": len(raw_bytes)}, f)
    print(f"Saved model to {args.log_dir}/")


if __name__ == "__main__":
    main()
