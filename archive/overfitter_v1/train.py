"""Simple overfit training: the whole file as one causal sequence, plain AdamW.

No curriculum/TBPTT, no frozen-base + adapter split, no per-segment state carrying --
SummTransformer already handles long context itself (windowed byte-level attention +
a hierarchical KV-summarization cascade, see Ks/attn_window/fuse_window in Config), so
there's nothing to chunk here: every step is a full recompute over the whole sequence.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from .config import TrainConfig
from .summformer import Config as ModelConfig
from .summformer import SummTransformer, lr_at


class _Tee:
    """stdout/stderr + file, eager flush -- matches randcompress/train.py's own _Tee, so
    `tail -f <log_dir>/train.log` shows tqdm's progress updates live, not just on exit."""
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


# ── SinkGD (Sinkhorn-normalized GD) -- ported from randcompress/train.py, model-agnostic ──

def _sinkhorn_2d(W: torch.Tensor, L: int) -> torch.Tensor:
    m, n = W.shape
    for _ in range(L):
        W = math.sqrt(n) * W / (W.norm(dim=1, keepdim=True) + 1e-8)
        W = math.sqrt(m) * W / (W.norm(dim=0, keepdim=True) + 1e-8)
    return W


def _sinkgd_normalize(g: torch.Tensor, L: int) -> torch.Tensor:
    g = g.float()
    if g.ndim == 0:
        return g
    if g.ndim == 1:
        n = g.shape[0]
        return math.sqrt(n) * g / (g.norm() + 1e-8)
    m = g.shape[0]
    n = g[0].numel()
    return _sinkhorn_2d(g.reshape(m, n), L).reshape(g.shape)


def sinkgd_step(params: dict[str, torch.nn.Parameter], lr: float, weight_decay: float,
                max_norm: float, L: int) -> None:
    grads = [p.grad for p in params.values() if p.grad is not None]
    if not grads:
        return
    gnorm = torch.stack([g.float().norm() ** 2 for g in grads]).sum().sqrt()
    scale = min(1.0, max_norm / (gnorm.item() + 1e-6))
    with torch.no_grad():
        for p in params.values():
            if p.grad is None:
                continue
            g_norm = _sinkgd_normalize(p.grad.float() * scale, L)
            p.sub_(lr * g_norm.to(p.dtype))
            if weight_decay > 0:
                p.sub_(lr * weight_decay * p)


def _split_param_groups(model: SummTransformer) -> tuple[dict, dict]:
    """(readout_params, matrix_params) -- embed/head/extra_heads vs everything else.
    Sinkhorn row/col normalization assumes a weight matrix with meaningful row/column
    structure (a good fit for attention/MLP projections); an embedding table or a
    per-symbol readout doesn't have that same structure, hence the split."""
    readout, matrix = {}, {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("embed.") or name.startswith("head.") or name.startswith("extra_heads."):
            readout[name] = p
        else:
            matrix[name] = p
    return readout, matrix


class MixedOptimizer:
    """optimizer='adamw': plain AdamW over everything. 'sinkgd_all': hand-written SinkGD over
    everything (matches randcompress/train.py's actual default -- no embed/head special-casing
    there today). 'sinkgd_lm' (default): SinkGD for matrix weights, AdamW for embed/head/
    extra_heads. See TrainConfig.optimizer for the full rationale."""
    def __init__(self, model: SummTransformer, tcfg: TrainConfig):
        self.tcfg = tcfg
        readout, matrix = _split_param_groups(model)
        if tcfg.optimizer == "adamw":
            self.adamw_params, self.sinkgd_params = {**readout, **matrix}, {}
        elif tcfg.optimizer == "sinkgd_all":
            self.adamw_params, self.sinkgd_params = {}, {**readout, **matrix}
        elif tcfg.optimizer == "sinkgd_lm":
            self.adamw_params, self.sinkgd_params = readout, matrix
        else:
            raise ValueError(f"unknown optimizer {tcfg.optimizer!r} "
                            f"(expected adamw | sinkgd_lm | sinkgd_all)")

        self.opt = None
        if self.adamw_params:
            self.opt = torch.optim.AdamW(list(self.adamw_params.values()), lr=tcfg.lr,
                                         betas=(0.9, 0.95), weight_decay=tcfg.weight_decay)

    def set_lr(self, lr: float) -> None:
        if self.opt is not None:
            for g in self.opt.param_groups:
                g["lr"] = lr

    def step(self, lr: float) -> None:
        if self.adamw_params:
            torch.nn.utils.clip_grad_norm_(list(self.adamw_params.values()), self.tcfg.grad_clip)
        if self.opt is not None:
            self.opt.step()
        if self.sinkgd_params:
            sinkgd_step(self.sinkgd_params, lr=lr, weight_decay=self.tcfg.weight_decay,
                       max_norm=self.tcfg.grad_clip, L=self.tcfg.sinkgd_l)


def train_overfit(model: SummTransformer, tcfg: TrainConfig, raw_bytes: np.ndarray,
                  device: torch.device) -> SummTransformer:
    model.to(device)
    model.train()
    ctx = torch.tensor(raw_bytes.astype(np.int64), device=device).unsqueeze(0)  # [1, L]

    opt = MixedOptimizer(model, tcfg)

    t0 = time.perf_counter()
    pbar = tqdm(range(1, tcfg.steps + 1), desc="overfit", dynamic_ncols=True)
    for step in pbar:
        lr = lr_at(step, tcfg.warmup_steps, tcfg.lr)
        opt.set_lr(lr)

        loss, metrics = model(ctx)
        model.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(lr)

        if step % tcfg.check_every == 0 or step == tcfg.steps:
            acc = metrics["byte_acc"].item()
            bpb = metrics["bpb"].item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}", bpb=f"{bpb:.3f}")
            if acc >= 1.0:
                pbar.close()
                print(f"[overfit] 100% byte accuracy reached at step {step} "
                      f"({time.perf_counter() - t0:.1f}s)")
                return model

    pbar.close()
    print(f"[overfit] stopped at max steps={tcfg.steps} ({time.perf_counter() - t0:.1f}s) "
          f"-- did not reach 100% accuracy")
    return model


def train_overfit_tbptt(model: SummTransformer, tcfg: TrainConfig, raw_bytes: np.ndarray,
                        device: torch.device) -> SummTransformer:
    """Stream the file through the incremental KV-cache stepper in chunks, RNN-style: state
    (byte-level KV cache + hierarchical summary history) persists across chunks within an
    epoch, backprop is truncated at each chunk boundary (detach_state -- TBPTT depth 1),
    and state resets to None at each epoch boundary (mirrors randcompress/train.py's
    curriculum trainer: "State resets to zeros at each epoch boundary").

    Unlike train_overfit(), this never materializes a dense L x L attention matrix -- the
    byte-level attention cost is bounded by cfg.attn_window (see Attn.forward_incremental's
    cache pruning), and the hierarchical summary stages' cost is bounded by keeping
    cfg.Ks[0] large enough that the number of pooled blocks stays small (that summary
    self-attention recomputes from scratch on every new block, so its cost is cubic in the
    number of blocks seen -- see the comment in summformer.py's _make_incremental_stepper).
    """
    model.to(device)
    model.train()
    ctx = torch.tensor(raw_bytes.astype(np.int64), device=device).unsqueeze(0)  # [1, L]
    L = ctx.shape[1]
    C = tcfg.tbptt_chunk_size
    n_chunks = math.ceil((L - 1) / C)

    opt = MixedOptimizer(model, tcfg)

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    global_step = 0
    total_steps = n_chunks * tcfg.epochs
    t0 = time.perf_counter()
    for epoch in range(1, tcfg.epochs + 1):
        stepper, detach_state = model._make_incremental_stepper(1, device)
        pos = 0
        epoch_ce_bits = 0.0   # accumulated straight from the training loss already computed
                               # each step -- an estimate of what rc_encode would produce, with
                               # no extra forward passes and no actual range coding.
        pbar = tqdm(range(n_chunks), desc=f"epoch {epoch}/{tcfg.epochs}", dynamic_ncols=True)
        for _ in pbar:
            global_step += 1
            end = min(pos + C, L - 1)
            inp = ctx[:, pos:end]        # bytes pos..end-1

            lr = lr_at(global_step, tcfg.warmup_steps, tcfg.lr)
            opt.set_lr(lr)

            logits, _ = stepper(inp, pos)          # [1, Tq, V] -- Tq may exceed inp's length:
            # when a fuse stage hasn't accumulated enough pooled blocks yet, byte-level queries
            # "backlog" (x_in_backlog in _make_incremental_stepper) until it fires, then all
            # backlogged positions are returned together in one call. logits[:, i, :] always
            # predicts the byte at absolute position (pos + Tn - Tq + i + 1), so targets must be
            # sliced from ctx using the actual returned length, not the input chunk length.
            Tq = logits.shape[1]
            tgt = ctx[:, end - Tq + 1: end + 1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            epoch_ce_bits += loss.item() * Tq / math.log(2)   # loss is a mean over Tq positions
            model.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(lr)
            detach_state()

            # MPS's caching allocator fragments badly under many small varying-shape allocations
            # (exactly this loop's pattern: growing/detached KV caches every step) and can OOM
            # well before actual live tensor memory is anywhere near the device limit. Periodic
            # empty_cache() returns unused cached blocks to the pool; cheap relative to a step.
            if device.type == "mps" and global_step % 50 == 0:
                torch.mps.empty_cache()

            if global_step % tcfg.check_every == 0 or global_step == total_steps:
                acc = (logits.argmax(-1) == tgt).float().mean().item()
                bpb = loss.item() / math.log(2)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}", bpb=f"{bpb:.3f}")

            pos = end

        pbar.close()

        # Estimated compression, straight from the training loss -- no rc_encode() call, no
        # separate forward pass. epoch_ce_bits already covers every position in the file
        # exactly once (chunks partition [0, L-1)), so it's directly the CE bpb the trained-
        # so-far model would need for the whole file, same quantity compress.py's ce_bpb
        # reports (just without the RC quantization step).
        est_bpb = epoch_ce_bits / (L - 1)
        est_rc_bytes = epoch_ce_bits / 8
        est_total = param_bytes + est_rc_bytes
        est_ratio = L / est_total if est_total > 0 else float("inf")
        print(f"[epoch {epoch}/{tcfg.epochs}]  CE~{est_bpb:.4f}bpb  "
              f"est size ~{est_rc_bytes:.0f}B (rc) + {param_bytes}B (params) "
              f"= {est_total:.0f}B  est ratio ~{est_ratio:.3f}x")

    print(f"[overfit-tbptt] {tcfg.epochs} epochs x {n_chunks} chunks/epoch "
          f"({total_steps} steps total, chunk={C}B)  {time.perf_counter() - t0:.1f}s")
    return model


def main() -> None:
    from .config import parse_configs
    from .tokenizer import load_bytes

    mcfg, tcfg = parse_configs()
    torch.manual_seed(tcfg.seed)
    device = torch.device(tcfg.device) if tcfg.device else torch.device(
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(tcfg.log_dir, exist_ok=True)
    log_file = open(os.path.join(tcfg.log_dir, "train.log"), "a")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)   # tqdm writes its bar to stderr by default
    print(f"logging to {os.path.join(tcfg.log_dir, 'train.log')} -- tail -f it")

    raw_bytes = load_bytes(tcfg.dataset)
    print(f"dataset={tcfg.dataset}  n_bytes={len(raw_bytes)}  device={device}")

    model = SummTransformer(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Ks={mcfg.Ks}  d_model={mcfg.d_model}  n_layers={mcfg.n_layers}  "
          f"mtp_heads={mcfg.mtp_heads}  params={n_params/1e6:.3f}M")

    if tcfg.tbptt_chunk_size > 0:
        train_overfit_tbptt(model, tcfg, raw_bytes, device)
    else:
        train_overfit(model, tcfg, raw_bytes, device)

    # Sanity check: the incremental KV-cache path (used by compress/decompress, and by
    # train_overfit_tbptt) must be bit-exact with full recompute -- this is what the
    # readout-vs-head fix in _make_incremental_stepper was for. Reusing raw_bytes as
    # "val_data" is fine here since this only samples prompts from it, no generalization
    # is being measured. Skipped for very long files where generate_no_cache's O(L^2)
    # full recompute would itself be infeasible.
    if len(raw_bytes) <= 20000:
        data_t = torch.tensor(raw_bytes.astype(np.int64), device=device)
        result = model.check_kv_cache_consistency(data_t, str(device))
        print(f"check_kv_cache_consistency: match_rate={result['match_rate']:.3f} "
              f"(n_checks={result['n_checks']})")
        if result["match_rate"] < 1.0:
            print("WARNING: incremental stepper diverges from full recompute -- "
                  "compress/decompress will still round-trip losslessly, but predictions "
                  "used for range coding won't match what the model was trained on.")

    from .checkpoint import save_model
    save_model(tcfg.log_dir, model, mcfg, tcfg)
    print(f"Saved model to {tcfg.log_dir}/")


if __name__ == "__main__":
    main()
