"""
HiRA adapter utilities.

HiRA:    W = W₀ + W₀ ⊙ (B·A)   (Hadamard-scaled low-rank update, full-rank effect)
Baseline: W = W₀ + B·A          (plain LoRA delta, same param count)

Both use the same (A, B) parameter pair with identical shapes.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


# ── Application ───────────────────────────────────────────────────────────────

def apply_hira(W0: Tensor, A: Tensor, B: Tensor) -> Tensor:
    """W = W₀ + W₀ ⊙ (B·A)"""
    return W0 + W0 * (B @ A)


def apply_lora(W0: Tensor, A: Tensor, B: Tensor) -> Tensor:
    """W = W₀ + B·A  (plain LoRA baseline)"""
    return W0 + B @ A


def apply_adapter(W0: Tensor, A: Tensor, B: Tensor, use_hira: bool) -> Tensor:
    return apply_hira(W0, A, B) if use_hira else apply_lora(W0, A, B)


# ── Initialisation ────────────────────────────────────────────────────────────

def init_hira_A(d_out: int, d_in: int, r: int, dtype: torch.dtype) -> Tensor:
    """A: [r, d_in] with orthonormal rows (right singular vectors of random matrix)."""
    raw = np.random.randn(r, d_in).astype(np.float64)
    if r <= d_in:
        _, _, Vt = np.linalg.svd(raw, full_matrices=False)
        A = Vt / math.sqrt(d_in)
    else:
        A = raw / math.sqrt(d_in)
    return torch.tensor(A[:r], dtype=dtype)


def init_hira_B(d_out: int, r: int, dtype: torch.dtype) -> Tensor:
    """B: [d_out, r] zero-init (ΔW = 0 at start)."""
    return torch.zeros(d_out, r, dtype=dtype)


def make_adapter_params(
    name_prefix: str,
    d_out: int,
    d_in: int,
    r: int,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Return {prefix.A: Tensor, prefix.B: Tensor} with requires_grad on B only."""
    A = init_hira_A(d_out, d_in, r, dtype)
    B = init_hira_B(d_out, r, dtype)
    return {
        f"{name_prefix}.A": A,              # fixed — registered as buffer
        f"{name_prefix}.B": B.requires_grad_(True),
    }


# ── B centering (SinkGD invariant) ────────────────────────────────────────────

def center_b(adapters: dict[str, Tensor]) -> None:
    """Zero-center columns of every B matrix in-place (no_grad)."""
    with torch.no_grad():
        for k, v in adapters.items():
            if k.endswith(".B"):
                v -= v.mean(dim=0, keepdim=True)
