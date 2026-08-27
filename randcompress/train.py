"""
Training: curriculum TBPTT, loss functions, and optimizer implementations.

Optimizers:
  sinkgd — Sinkhorn Gradient Descent (default)
  adamw  — torch.optim.AdamW
  sgd    — torch.optim.SGD with momentum
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .config import ModelConfig, TrainConfig
from .models.hira import center_b
from .tokenizer import bytes_to_tokens, tokens_to_bytes, load_bytes


# ── Tee logger (stdout + file, eager flush) ───────────────────────────────────

class _Tee:
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


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_chunks(raw_bytes: np.ndarray, tcfg: TrainConfig, mtp_k: int = 1):
    """Returns (all_inputs [NC, S], all_targets [NC, S, k, oh]) int32 numpy.

    targets[c, s, j, h] = token at position c*S + s + j + 1 (shifted by j+1).
    j=0 is the standard AR target; j>0 are MTP lookahead targets.
    mtp_k=1 (default) gives targets [NC, S, 1, oh] — same data as the old [NC, S, oh].
    """
    ib, ob, oh = tcfg.input_bits, tcfg.output_bits, tcfg.output_heads
    toks       = bytes_to_tokens(raw_bytes, ib)
    chunk_size = tcfg.segment_size * (8 // ib)
    n          = len(toks)
    out_mask   = (1 << ob) - 1
    num_chunks = max(1, math.ceil((n - 1) / chunk_size))

    # extra padding: mtp_k-1 additional positions for lookahead heads
    pad_len = max(0, num_chunks * chunk_size + oh + mtp_k - 1 - n)
    padded  = np.concatenate([toks, np.zeros(pad_len + oh + mtp_k - 1, dtype=np.int32)])

    inputs  = np.stack([padded[i * chunk_size: i * chunk_size + chunk_size]
                        for i in range(num_chunks)]).astype(np.int32)

    tgt_chunks = []
    for i in range(num_chunks):
        tgt = np.zeros((chunk_size, mtp_k, oh), dtype=np.int32)
        for j in range(mtp_k):
            if ib == 8 and ob < 8:
                nxt = padded[i * chunk_size + 1 + j: i * chunk_size + chunk_size + 1 + j]
                for h in range(oh):
                    tgt[:, j, h] = (nxt >> (h * ob)) & out_mask
            else:
                for h in range(oh):
                    tgt[:, j, h] = padded[i * chunk_size + 1 + j + h:
                                          i * chunk_size + chunk_size + 1 + j + h]
        tgt_chunks.append(tgt)

    return inputs, np.stack(tgt_chunks)  # [NC, S, k, oh]


def split_segments(raw_bytes: np.ndarray, segment_size: int) -> list[np.ndarray]:
    segs, n, i = [], len(raw_bytes), 0
    while i < n:
        segs.append(raw_bytes[i: i + segment_size])
        i += segment_size
    return segs


# ── Loss functions ────────────────────────────────────────────────────────────

def cross_entropy_loss(logits: Tensor, targets: Tensor, pad_token: int) -> Tensor:
    """logits: [..., V], targets: [...] int. Masked CE."""
    log_probs    = F.log_softmax(logits, dim=-1)
    tgt_clamped  = targets.clamp(min=0)
    loss_per_tok = -log_probs.gather(-1, tgt_clamped.unsqueeze(-1)).squeeze(-1)
    mask         = (targets != pad_token).float()
    return (loss_per_tok * mask).sum() / (mask.sum() + 1e-8)


def agd_loss(logits: Tensor, targets: Tensor, pad_token: int,
             detach_weights: bool = True) -> Tensor:
    """Arithmetic Gradient Descent loss. Works with any leading batch dims."""
    log_probs_full = F.log_softmax(logits, dim=-1)
    tgt_clamped    = targets.clamp(min=0)
    log_probs      = log_probs_full.gather(-1, tgt_clamped.unsqueeze(-1)).squeeze(-1)

    src       = log_probs.detach() if detach_weights else log_probs
    log_I     = torch.cumsum(src, dim=-1)
    neg_log_I = -log_I
    log_w     = neg_log_I - torch.logsumexp(neg_log_I, dim=-1, keepdim=True)
    weights   = log_w.exp()

    mask = (targets != pad_token).float()
    return (weights * (-log_probs) * mask).sum() / (mask.sum() + 1e-8)


def training_loss(logits: Tensor, targets: Tensor, tcfg: TrainConfig) -> Tensor:
    if tcfg.use_agd_loss:
        return agd_loss(logits, targets, tcfg.pad_token, tcfg.agd_detach_weights)
    return cross_entropy_loss(logits, targets, tcfg.pad_token)


# ── SinkGD optimizer ──────────────────────────────────────────────────────────

def _sinkhorn_2d(W: Tensor, L: int) -> Tensor:
    m, n = W.shape
    for _ in range(L):
        W = math.sqrt(n) * W / (W.norm(dim=1, keepdim=True) + 1e-8)
        W = math.sqrt(m) * W / (W.norm(dim=0, keepdim=True) + 1e-8)
    return W


def _sinkgd_normalize(g: Tensor, L: int) -> Tensor:
    g = g.float()
    if g.ndim == 0:
        return g
    if g.ndim == 1:
        n = g.shape[0]
        return math.sqrt(n) * g / (g.norm() + 1e-8)
    m = g.shape[0]
    n = g[0].numel()
    return _sinkhorn_2d(g.reshape(m, n), L).reshape(g.shape)


def sinkgd_step(
    adapters: dict[str, Tensor],
    grads: dict[str, Tensor],
    lr: float,
    weight_decay: float,
    max_norm: float,
    L: int,
    use_hira: bool,
) -> None:
    """In-place SinkGD parameter update."""
    # Global grad clip
    all_grads = [v for v in grads.values() if v is not None]
    gnorm = torch.stack([g.float().norm() ** 2 for g in all_grads]).sum().sqrt()
    scale = min(1.0, max_norm / (gnorm.item() + 1e-6))

    with torch.no_grad():
        for k, p in adapters.items():
            if not p.requires_grad:
                continue
            g = grads.get(k)
            if g is None:
                continue
            g_scaled = g.float() * scale
            g_norm   = _sinkgd_normalize(g_scaled, L)
            p.sub_(lr * g_norm.to(p.dtype))
            if weight_decay > 0:
                p.sub_(lr * weight_decay * p)

    if use_hira:
        center_b(adapters)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_segments_stateful(model, frozen, adapters, tcfg: TrainConfig,
                           seg_list: list[np.ndarray], device: torch.device):
    """Teacher-forced argmax accuracy per segment, state flowing across chunks."""
    ib, ob, oh = tcfg.input_bits, tcfg.output_bits, tcfg.output_heads
    out_mask   = (1 << ob) - 1
    chunk_size = tcfg.segment_size * (8 // ib)
    results    = []

    for seg_bytes in seg_list:
        seg   = bytes_to_tokens(np.array(seg_bytes, dtype=np.uint8), ib)
        n_tok = len(seg)
        n_b   = len(seg_bytes)
        if n_b < 2:
            results.append((1.0, None))
            continue

        states     = model.init_states(1, device)
        num_chunks = max(1, math.ceil((n_tok - 1) / chunk_size))
        pad_len    = max(0, num_chunks * chunk_size + 1 - n_tok)
        padded     = np.concatenate([seg, np.zeros(pad_len, dtype=np.int32)])

        byte_correct = []
        tok_per_byte = 8 // ib

        for ci in range(num_chunks):
            s    = ci * chunk_size
            inp  = torch.tensor(padded[s: s + chunk_size][None], dtype=torch.long, device=device)
            logits, states = model.forward(frozen, adapters, inp, states)
            # logits: [1, S, k, oh, ov] — use head 0 (AR head) for accuracy
            preds = logits[0, :, 0, :, :].argmax(dim=-1).cpu().numpy()  # [S, oh]

            if ib == 8 and ob < 8:
                pred_bytes = np.zeros(preds.shape[0], np.uint8)
                for h in range(oh):
                    pred_bytes |= (preds[:, h].astype(np.uint8) & out_mask) << (h * ob)
                tgt_bytes = padded[s + 1: s + chunk_size + 1].astype(np.uint8)
            else:
                pred_bytes = tokens_to_bytes(preds[:, 0], ib)
                tgt_bytes  = tokens_to_bytes(padded[s + 1: s + chunk_size + 1], ib)

            valid = min(len(pred_bytes),
                        max(0, n_b - 1 - ci * (chunk_size // tok_per_byte)))
            if valid > 0:
                byte_correct.extend((pred_bytes[:valid] == tgt_bytes[:valid]).tolist())

        total = len(byte_correct)
        acc   = sum(byte_correct) / total if total > 0 else 1.0
        wrong = [i for i, ok in enumerate(byte_correct) if not ok]
        results.append((acc, wrong[0] if wrong else None))

    return results


# ── State utilities ───────────────────────────────────────────────────────────

def _detach_states(states):
    """Recursively detach all tensors in a state structure (list of tensors or tuples)."""
    def _det(x):
        if isinstance(x, torch.Tensor):
            return x.detach()
        if isinstance(x, tuple):
            return tuple(_det(v) for v in x)
        if isinstance(x, list):
            return [_det(v) for v in x]
        return x
    return _det(states)


# ── TBPTT chunk step ──────────────────────────────────────────────────────────

def compute_loss_and_grads(model, frozen, adapters, inp, tgt,
                           states, tcfg: TrainConfig):
    """One TBPTT chunk: forward, loss, backward. Returns (loss, grads, new_states).

    tgt: [B, S, k, oh] — targets for all k MTP heads.
    logits: [B, S, k, oh, ov] from model.forward.
    Loss is averaged over k heads and oh sub-heads.
    """
    for p in adapters.values():
        if p.requires_grad and p.grad is not None:
            p.grad = None

    inp_t = torch.tensor(inp, dtype=torch.long)
    tgt_t = torch.tensor(tgt, dtype=torch.long)

    logits, new_states = model.forward(frozen, adapters, inp_t, states)
    # logits: [B, S, k, oh, ov]
    k  = logits.shape[2]
    oh = tcfg.output_heads

    loss = sum(
        training_loss(logits[:, :, j, h, :], tgt_t[:, :, j, h], tcfg)
        for j in range(k) for h in range(oh)
    ) / (k * oh)

    loss.backward()

    grads = {k: (p.grad.clone() if p.grad is not None else None)
             for k, p in adapters.items() if p.requires_grad}
    return loss.detach().item(), grads, _detach_states(new_states)


# ── Curriculum trainer ────────────────────────────────────────────────────────

class CurriculumTrainer:

    def __init__(self, model, frozen, adapters, mcfg: ModelConfig,
                 tcfg: TrainConfig, device: torch.device, log_dir: str):
        self.model    = model
        self.frozen   = frozen
        self.adapters = adapters
        self.mcfg     = mcfg
        self.tcfg     = tcfg
        self.device   = device
        self.log_dir  = log_dir

        self._build_optimizer()
        self.global_iters = 0

    def _build_optimizer(self):
        tcfg = self.tcfg
        trainable = [p for p in self.adapters.values() if p.requires_grad]
        if tcfg.optimizer == "adamw":
            self.opt = torch.optim.AdamW(
                trainable, lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
        elif tcfg.optimizer == "sgd":
            self.opt = torch.optim.SGD(
                trainable, lr=tcfg.learning_rate,
                momentum=tcfg.sgd_momentum, weight_decay=tcfg.weight_decay)
        else:
            self.opt = None   # SinkGD is manual

    def _opt_step(self, grads):
        tcfg = self.tcfg
        if tcfg.optimizer == "sinkgd":
            sinkgd_step(
                self.adapters, grads,
                lr=tcfg.learning_rate,
                weight_decay=tcfg.weight_decay,
                max_norm=tcfg.grad_clip_norm,
                L=tcfg.sinkgd_l,
                use_hira=self.mcfg.use_hira,
            )
        else:
            # Inject grads into .grad, then clip + step
            for k, p in self.adapters.items():
                if p.requires_grad and k in grads and grads[k] is not None:
                    p.grad = grads[k].to(p.dtype)
            trainable = [p for p in self.adapters.values() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, self.tcfg.grad_clip_norm)
            self.opt.step()
            self.opt.zero_grad()

    def _tbptt_phase(self, phase_name: str, inputs: np.ndarray,
                     targets: np.ndarray, eval_segs: list[np.ndarray]):
        """Single SOLO or COMBINED phase. Returns (success, iters_done, last_accs)."""
        tcfg      = self.tcfg
        n_chunks  = inputs.shape[0]
        chunk_idx = 0
        states    = self.model.init_states(1, self.device)
        last_accs = None
        success   = False

        pbar = tqdm(range(1, tcfg.max_iter_per_phase + 1),
                    desc=phase_name, unit="it", dynamic_ncols=True)

        for it in pbar:
            if chunk_idx == 0:
                states = self.model.init_states(1, self.device)

            inp = inputs[chunk_idx][None, :]    # [1, S]
            tgt = targets[chunk_idx][None, :]   # [1, S, oh]
            chunk_idx = (chunk_idx + 1) % n_chunks

            frozen_states = _detach_states(states)
            loss, grads, states = compute_loss_and_grads(
                self.model, self.frozen, self.adapters,
                inp, tgt, frozen_states, tcfg)

            self._opt_step(grads)
            self.global_iters += 1

            bpb = loss / math.log(2) * (8 // tcfg.input_bits)
            pbar.set_postfix(loss=f"{loss:.4f}", bpb=f"{bpb:.3f}")

            if it % tcfg.check_every == 0 or it == tcfg.max_iter_per_phase:
                accs = eval_segments_stateful(
                    self.model, self.frozen, self.adapters,
                    tcfg, eval_segs, self.device)
                last_accs = accs
                min_acc   = min(a for a, _ in accs)
                fw_str    = next((str(fw) for _, fw in accs if fw is not None), "ok")
                pbar.set_description(f"{phase_name} acc={min_acc:.1%} fw={fw_str}")
                print()
                if all(a == 1.0 for a, _ in accs):
                    success = True
                    pbar.close()
                    break

        else:
            pbar.close()

        return success, it, last_accs

    def run(self, raw_bytes: np.ndarray):
        tcfg      = self.tcfg
        segs      = split_segments(raw_bytes, tcfg.segment_size)
        n_segs    = len(segs)
        completed = []
        failed    = None
        t_start   = time.perf_counter()

        print(f"Segments: {n_segs} × up to {tcfg.segment_size} bytes")
        print(f"Optimizer: {tcfg.optimizer}  lr={tcfg.learning_rate}  "
              f"max_iter/phase={tcfg.max_iter_per_phase}  check_every={tcfg.check_every}")

        k = self.mcfg.mtp_k
        for seg_idx, seg in enumerate(segs):
            t_seg   = time.perf_counter()
            byte_lo = seg_idx * tcfg.segment_size
            print(f"\nSEGMENT {seg_idx+1}/{n_segs}  bytes [{byte_lo}, {byte_lo+len(seg)})")

            inp, tgt = make_chunks(seg, tcfg, mtp_k=k)
            print(f"[SOLO] chunks={inp.shape[0]}")
            ok, iters, accs = self._tbptt_phase(f"solo s{seg_idx+1}", inp, tgt, [seg])

            elapsed = time.perf_counter() - t_seg
            acc0 = accs[0][0] if accs else 0.0
            fw0  = accs[0][1] if accs else None
            if ok:
                print(f"[PASS] solo seg{seg_idx+1}  acc=100%  iters={iters}  {elapsed:.1f}s")
            else:
                print(f"[FAIL] solo seg{seg_idx+1}  acc={acc0:.2%}  "
                      f"first_wrong={fw0}  iters={iters}  {elapsed:.1f}s")
                failed = seg_idx + 1
                break

            completed.append(seg)
            if len(completed) == 1:
                continue

            combined = np.concatenate(completed)
            cinp, ctgt = make_chunks(combined, tcfg, mtp_k=k)
            n_done = len(completed)
            print(f"[COMBINED] segs 1..{n_done}  chunks={cinp.shape[0]}")
            cok, citers, caccs = self._tbptt_phase(
                f"comb 1..{n_done}", cinp, ctgt, completed)

            elapsed = time.perf_counter() - t_seg
            min_acc = min(a for a, _ in caccs) if caccs else 0.0
            if cok:
                print(f"[PASS] comb 1..{n_done}  acc=100%  iters={citers}  {elapsed:.1f}s")
            else:
                print(f"[FAIL] comb 1..{n_done}  min_acc={min_acc:.2%}  "
                      f"iters={citers}  {elapsed:.1f}s")
                failed = seg_idx + 1
                break

        elapsed_total = time.perf_counter() - t_start
        stop = (f"FAIL at segment {failed}" if failed else f"all {n_segs} segments mastered")
        print(f"\n[STOP] {stop}  total_iters={self.global_iters}  elapsed={elapsed_total:.1f}s")
        return failed is None
