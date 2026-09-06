"""Analytical KV-cache memory / compute estimator for a SummTransformer config -- no training
run required. Implements the formulas derived by hand: byte-level + refinement passes are
O(L) (windowed), stage self-attention is O(n_blocks_max^2) (real incremental causal attention,
dominated by the largest/earliest stage), cross-attention is O(L * n_blocks_max) (triangular
sum against a growing key set). See CLAUDE.md's summformer notes for the derivation.

Also flags the "Ks too deep for this file" pathology: a stage whose cum_K exceeds the file
length never fires, and its query backlog in _make_incremental_stepper grows unboundedly for
the entire run -- this is what caused the MPS OOM during the juz1.txt ablation.

Usage:
    uv run python -m overfitter.analyze --Ks 16,16,16 --d_model 24 --n_layers 8 \
        --attn_window 64 --dataset datasets/juz1.txt --epochs 8
    uv run python -m overfitter.analyze --Ks 16,16,16 --d_model 24 --n_layers 8 \
        --attn_window 64 --file_len 44443 --epochs 8
"""
from __future__ import annotations

import argparse
import math
from dataclasses import fields

from .summformer import Config as ModelConfig
from .summformer import SummTransformer, ROPE_PRESETS


def n_blocks_per_stage(Ks: tuple[int, ...], L: int) -> list[float]:
    cum = 1
    out = []
    for k in Ks:
        cum *= k
        out.append(L / cum)
    return out


def analyze(cfg: ModelConfig, L: int, chunk_size: int = 0, epochs: int = 1) -> dict:
    D, H, Lyr, W, mlp = cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.attn_window, cfg.mlp_mult
    n_fuse = len(cfg.Ks)
    nb = n_blocks_per_stage(cfg.Ks, L)
    W_eff = W if W is not None else L   # unbounded byte-level window = full file, for the formula

    # ---- KV-cache memory (fp32, batch=1) ----
    byte_cache_B = 2 * D * W_eff * 4 * Lyr
    refine_cache_B = byte_cache_B * n_fuse   # one refinement pass per fuse stage, same shape
    stage_cache_B = [2 * D * n * 4 * Lyr for n in nb]
    h_hist_B = L * D * 4
    total_cache_B = byte_cache_B + refine_cache_B + sum(stage_cache_B) + h_hist_B

    # ---- Compute (multiply-adds), per epoch over L bytes ----
    per_token_linear = 4 * D * D + 2 * W_eff * D + 6 * D * D * mlp
    flops_linear = L * Lyr * (n_fuse + 1) * per_token_linear
    flops_stage = sum(D * Lyr * n * n for n in nb)
    flops_cross = sum(D * Lyr * (L * n / 2) for n in nb)
    flops_epoch = flops_linear + flops_stage + flops_cross

    # ---- Param count (exact, via instantiation) ----
    n_params = sum(p.numel() for p in SummTransformer(cfg).parameters())

    # ---- Pathology check: does every stage actually fire? ----
    warnings = []
    cum = 1
    for s, k in enumerate(cfg.Ks):
        cum *= k
        if cum > L / 4:
            warnings.append(
                f"stage {s} (cum_K={cum:,}) needs ~{cum:,} bytes for even one firing "
                f"vs file length {L:,} -- {'WILL NEVER FIRE, backlog grows unboundedly' if cum > L else 'fires only a handful of times, marginal'}")
        elif chunk_size and cum > chunk_size:
            # Distinct from starvation: this stage DOES fire, but while it's waiting, byte-level
            # queries pile up in x_in_backlog (up to ~cum bytes) until it does, then that whole
            # backlog is pushed through cross-attn + the refinement pass's byte-level attention
            # in ONE call -- and within a single call, attn_window only bounds the persistent
            # cache, not that call's own Tn x (Tn+window) attention matrix. A real OOM (~7.5GB
            # observed) with cum_K=16384, chunk_size=512, attn_window=4, d_model=24, 4 heads.
            warnings.append(
                f"stage {s} (cum_K={cum:,}) exceeds chunk_size={chunk_size:,} -- its query "
                f"backlog can grow to ~{cum:,} positions before firing, then spikes a single "
                f"O(backlog^2) attention call in the refinement pass. Reduce depth or grow "
                f"chunk_size so every stage's cum_K stays <= chunk_size.")

    balance_Ks0 = math.sqrt(L / W_eff) if W_eff else None

    return dict(
        L=L, n_fuse=n_fuse, n_blocks_max=nb, n_params=n_params,
        byte_cache_B=byte_cache_B, refine_cache_B=refine_cache_B,
        stage_cache_B=stage_cache_B, h_hist_B=h_hist_B, total_cache_B=total_cache_B,
        flops_linear=flops_linear, flops_stage=flops_stage, flops_cross=flops_cross,
        flops_epoch=flops_epoch, flops_total=flops_epoch * epochs,
        epochs=epochs, chunk_size=chunk_size,
        n_steps_per_epoch=math.ceil((L - 1) / chunk_size) if chunk_size else None,
        balance_Ks0=balance_Ks0, warnings=warnings,
    )


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _fmt_flops(n: float) -> str:
    for unit, div in (("GFLOPs", 1e9), ("MFLOPs", 1e6), ("KFLOPs", 1e3)):
        if n >= div:
            return f"{n/div:.2f} {unit}"
    return f"{n:.0f} FLOPs"


def print_report(cfg: ModelConfig, r: dict) -> None:
    print(f"Ks={cfg.Ks}  n_fuse={r['n_fuse']}  d_model={cfg.d_model}  n_layers={cfg.n_layers}  "
          f"n_heads={cfg.n_heads}  attn_window={cfg.attn_window}  mlp_mult={cfg.mlp_mult}")
    print(f"File length L={r['L']:,} bytes" +
          (f"  chunk_size={r['chunk_size']}  steps/epoch={r['n_steps_per_epoch']:,}" if r['chunk_size'] else ""))
    print(f"params={r['n_params']:,}")
    print()
    print(f"n_blocks_max per stage: {['%.1f' % n for n in r['n_blocks_max']]}")
    print()
    print("KV-cache memory (persistent, fp32, batch=1):")
    print(f"  byte-level pass        {_fmt_bytes(r['byte_cache_B']):>10s}")
    print(f"  refinement passes (x{r['n_fuse']})  {_fmt_bytes(r['refine_cache_B']):>10s}")
    for s, b in enumerate(r['stage_cache_B']):
        print(f"  stage {s} self-attn cache {_fmt_bytes(b):>10s}")
    print(f"  h_hist (byte history)  {_fmt_bytes(r['h_hist_B']):>10s}")
    print(f"  {'TOTAL':22s}  {_fmt_bytes(r['total_cache_B']):>10s}")
    print()
    print("Compute per epoch (multiply-adds):")
    print(f"  linear (byte+refine)   {_fmt_flops(r['flops_linear']):>12s}")
    print(f"  stage self-attn (quad) {_fmt_flops(r['flops_stage']):>12s}")
    print(f"  cross-attn              {_fmt_flops(r['flops_cross']):>12s}")
    print(f"  {'TOTAL/epoch':22s}  {_fmt_flops(r['flops_epoch']):>12s}")
    if r['epochs'] > 1:
        print(f"  {'TOTAL x' + str(r['epochs']) + ' epochs':22s}  {_fmt_flops(r['flops_total']):>12s}")
    print()
    if r['balance_Ks0']:
        print(f"Balance heuristic: Ks[0] ~ sqrt(L/attn_window) = {r['balance_Ks0']:.1f}  "
              f"(actual Ks[0]={cfg.Ks[0]})")
    if r['warnings']:
        print("\nWARNINGS:")
        for w in r['warnings']:
            print(f"  - {w}")
    else:
        print("\nNo stage-starvation warnings -- every stage fires enough times to stay bounded.")


def main() -> None:
    p = argparse.ArgumentParser(description="Analytical KV-cache memory/compute estimator")
    p.add_argument("--Ks", default="32,32")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--attn_window", type=int, default=None)
    p.add_argument("--mtp_heads", type=int, default=1)
    p.add_argument("--dataset", type=str, default=None, help="infer L from this file's byte length")
    p.add_argument("--file_len", type=int, default=None, help="or give L directly")
    p.add_argument("--chunk_size", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    if args.file_len is not None:
        L = args.file_len
    elif args.dataset is not None:
        from .tokenizer import load_bytes
        L = len(load_bytes(args.dataset))
    else:
        p.error("need --dataset or --file_len")

    Ks = tuple(int(x) for x in args.Ks.split(","))
    cfg = ModelConfig(Ks=Ks, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                      n_kv_heads=args.n_heads, mlp_mult=args.mlp_mult, attn_window=args.attn_window,
                      input_preset=8, mtp_heads=args.mtp_heads)
    r = analyze(cfg, L, chunk_size=args.chunk_size, epochs=args.epochs)
    print_report(cfg, r)


if __name__ == "__main__":
    main()
