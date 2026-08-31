"""Overfit a ByteFractalGen to one file. Each patch_len_list[0]-byte chunk is fully independent
(root_cond is a fixed constant, no state carried across chunks), so training is just: iterate
chunks, forward, backward, AdamW step.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from .model import ByteFractalGen, FractalConfig
from .tokenizer import load_bytes


class _Tee:
    """stdout/stderr + file, eager flush -- tail -f <log_dir>/train.log to watch a run live."""
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
         lr: float, warmup_steps: int, grad_clip: float, log_every: int, n_raw: int) -> None:
    model.to(device)
    model.train()
    ctx = torch.tensor(chunks.astype(np.int64), device=device)   # [n_chunks, P0]
    n_chunks, P0 = ctx.shape
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    param_bytes = sum(pp.numel() * pp.element_size() for pp in model.parameters())

    t0 = time.perf_counter()
    epoch_ce_nats = 0.0   # accumulated straight from the training loss already computed each
                           # step -- an estimate of what rc_encode would produce, with no extra
                           # forward passes and no actual range coding run mid-training.
    epoch_steps = 0
    epoch = 1
    pbar = tqdm(range(1, steps + 1), desc="fractalgen", dynamic_ncols=True)
    for step in pbar:
        lr_t = lr * min(1.0, step / max(1, warmup_steps))
        for g in opt.param_groups:
            g["lr"] = lr_t

        idx = (step - 1) % n_chunks
        chunk = ctx[idx: idx + 1]
        loss, metrics = model(chunk)
        epoch_ce_nats += loss.item() * P0   # loss is mean nats/byte over this chunk's P0 bytes
        epoch_steps += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        acc = metrics["byte_acc"].item()
        bpb = metrics["bpb"].item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}", bpb=f"{bpb:.3f}")

        if step % log_every == 0 or step == steps:
            print(f"[step {step}/{steps}]  bpb={bpb:.4f}  acc={acc:.2%}  loss={loss.item():.4f}")

        if step % n_chunks == 0 or step == steps:
            n_bytes_seen = epoch_steps * P0
            est_ce_bits = epoch_ce_nats / math.log(2)
            est_bpb = est_ce_bits / n_bytes_seen
            est_rc_bytes = est_ce_bits / 8 * (n_raw / n_bytes_seen)   # scale partial epoch to full file
            est_total = param_bytes + est_rc_bytes
            est_ratio = n_raw / est_total if est_total > 0 else float("inf")
            print(f"[epoch ~{epoch}]  CE~{est_bpb:.4f}bpb (theoretical)  "
                  f"est size ~{est_rc_bytes:.0f}B (rc) + {param_bytes}B (params) "
                  f"= {est_total:.0f}B  est ratio ~{est_ratio:.4f}x")
            epoch_ce_nats = 0.0
            epoch_steps = 0
            epoch += 1

    pbar.close()
    dt = time.perf_counter() - t0
    print(f"[fractalgen train] {steps} steps over {n_chunks} chunks  {dt:.1f}s  "
          f"({steps/dt:.1f} steps/s, {steps*P0/dt/1e3:.1f} kB/s effective)")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--log_dir", required=True)
    p.add_argument("--patch_len_list", default="1024,128,16,1")
    p.add_argument("--d_model", type=int, default=64, help="broadcast uniformly to every level")
    p.add_argument("--n_layers", type=int, default=4, help="broadcast uniformly to every level")
    p.add_argument("--n_heads", type=int, default=4, help="broadcast uniformly to every level")
    p.add_argument("--mlp_mult", type=int, default=2, help="broadcast uniformly to every level")
    p.add_argument("--byte_embed_dim", type=int, default=256)
    p.add_argument("--share_trunk", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=str, default="100",
                   help="int (e.g. 100) = print current bpb/acc/loss every N steps; float "
                        "(e.g. 0.5) = every that fraction of an epoch (one epoch = n_chunks "
                        "steps), rounded to steps. tqdm postfix always updates every step "
                        "regardless (cheap, no extra forward pass).")
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

    n_levels = len(patch_len_list) - 1
    cfg = FractalConfig(
        patch_len_list=patch_len_list,
        d_model_list=(args.d_model,) * n_levels,
        n_layers_list=(args.n_layers,) * n_levels,
        n_heads_list=(args.n_heads,) * n_levels,
        mlp_mult_list=(args.mlp_mult,) * n_levels,
        byte_embed_dim=args.byte_embed_dim,
        share_trunk=args.share_trunk,
    )
    model = ByteFractalGen(cfg)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"params={n_params:,}  seq_lens={model.seq_lens}")

    chunks = make_chunks(raw_bytes, patch_len_list[0])
    n_chunks = chunks.shape[0]
    log_every = (max(1, round(float(args.log_every) * n_chunks))
                 if "." in args.log_every else int(args.log_every))
    print(f"log_every={args.log_every} -> {log_every} steps "
          f"({'epoch fraction' if '.' in args.log_every else 'raw steps'})")
    if log_every <= 2:
        print(f"WARNING: log_every={log_every} is very frequent (prints nearly every step) "
              f"-- pass a float like 0.1 for 'every 10% of an epoch' if this wasn't intended")

    train(model, chunks, device, args.steps, args.lr, args.warmup_steps, args.grad_clip,
         log_every, len(raw_bytes))

    from .checkpoint import save_model
    save_model(args.log_dir, model)
    with open(os.path.join(args.log_dir, "meta.json"), "w") as f:
        import json
        json.dump({"n_raw_bytes": len(raw_bytes)}, f)
    print(f"Saved model to {args.log_dir}/")


if __name__ == "__main__":
    main()
