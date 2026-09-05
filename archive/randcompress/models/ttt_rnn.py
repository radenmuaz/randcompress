"""
TTT-Linear RNN (Test-Time Training, linear variant).

Each layer maintains a hidden weight matrix W ∈ [d, d] as state.
At each step, one gradient descent step on a self-supervised inner loss updates W:

  inner loss: ||W_{t-1} q_t - k_t||²
  gradient:   (W_{t-1} q_t - k_t) q_t^T
  update:     W_t = W_{t-1} - eta * (W_{t-1} q_t - k_t) q_t^T
            = (I - eta * q_t q_t^T) W_{t-1} + eta * k_t q_t^T

Output: h_t = LayerNorm(W_t v_t)   (v_t = value projection of input)

Frozen weights per layer: W_q, W_k, W_v [d_model, d_model], eta (scalar, log-uniform init)
Adapters: HiRA/LoRA on W_q, W_k, W_v.
State: W ∈ [B, d_model, d_model] per layer.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import RandCompressModel
from .hira import apply_adapter, make_adapter_params
from .msrnn import _ortho, _layer_norm


def _init_ttt_frozen(rng: np.random.Generator, d: int, seq_len: int,
                     dtype) -> dict[str, Tensor]:
    rngs = [np.random.default_rng(rng.integers(2**31)) for _ in range(3)]
    W_q  = _ortho(rngs[0], d, d, dtype) / math.sqrt(d)
    W_k  = _ortho(rngs[1], d, d, dtype) / math.sqrt(d)
    W_v  = _ortho(rngs[2], d, d, dtype) / math.sqrt(d)
    # eta: log-uniform in [1/seq_len, 1/1], so effective learning rate decays with context
    tau = float(np.exp(np.random.default_rng(rng.integers(2**31))
                       .uniform(0.0, np.log(max(float(seq_len), 2.0)))))
    eta = torch.tensor(1.0 / tau, dtype=dtype)
    return {
        "W_q":   W_q,
        "W_k":   W_k,
        "W_v":   W_v,
        "eta":   eta,
        "norm_w": torch.ones(d, dtype=dtype),
        "norm_b": torch.zeros(d, dtype=dtype),
    }


def _ttt_step(fw: dict, x_t: Tensor, W: Tensor) -> tuple[Tensor, Tensor]:
    """Single TTT-Linear step. x_t: [B, d]. Returns (out [B, d], W_new)."""
    q = x_t @ fw["W_q"].T   # [B, d]
    k = x_t @ fw["W_k"].T   # [B, d]
    v = x_t @ fw["W_v"].T   # [B, d]

    # Inner prediction and error
    pred  = torch.einsum("bde,be->bd", W, q)      # W_{t-1} q_t  [B, d]
    error = pred - k                               # [B, d]

    # GD update: W_t = W_{t-1} - eta * error ⊗ q^T
    eta   = fw["eta"]
    delta = torch.einsum("bd,be->bde", error, q)  # [B, d, d]
    W_new = W - eta * delta

    # Output: LayerNorm(W_t v_t)
    h = torch.einsum("bde,be->bd", W_new, v)      # [B, d]
    out = _layer_norm(h, fw["norm_w"], fw["norm_b"])
    return out, W_new


def _ttt_scan(fw: dict, x: Tensor, W0: Tensor | None = None):
    """TTT-Linear scan over sequence. x: [B, S, d]. Returns (out [B,S,d], W_T)."""
    B, S, d = x.shape
    if W0 is None:
        W0 = torch.zeros(B, d, d, dtype=x.dtype, device=x.device)

    # Pre-project inputs
    q_all = x @ fw["W_q"].T   # [B, S, d]
    k_all = x @ fw["W_k"].T
    v_all = x @ fw["W_v"].T

    W   = W0
    eta = fw["eta"]
    hs  = []
    for t in range(S):
        q = q_all[:, t, :]
        k = k_all[:, t, :]
        v = v_all[:, t, :]
        pred  = torch.einsum("bde,be->bd", W, q)
        error = pred - k
        delta = torch.einsum("bd,be->bde", error, q)
        W     = W - eta * delta
        h     = torch.einsum("bde,be->bd", W, v)
        hs.append(_layer_norm(h, fw["norm_w"], fw["norm_b"]))

    return torch.stack(hs, dim=1), W


class TTTLinear(RandCompressModel):

    def __init__(self, model_cfg, train_cfg):
        self.mcfg  = model_cfg
        self.tcfg  = train_cfg
        self.dtype = torch.float32 if train_cfg.dtype == "float32" else torch.bfloat16

    def init_frozen(self, seed: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(seed)
        cfg = self.mcfg; tcfg = self.tcfg; d = cfg.d_model; dt = self.dtype
        out: dict[str, Tensor] = {}
        emb_raw = rng.standard_normal((tcfg.vocab_size, d)).astype(np.float32) * 0.01
        out["embedding"] = torch.tensor(emb_raw, dtype=dt)
        for i in range(cfg.num_layers):
            sub = np.random.default_rng(rng.integers(2**31))
            fw  = _init_ttt_frozen(sub, d, tcfg.segment_size, dt)
            for k, v in fw.items():
                out[f"block{i}.{k}"] = v
            out[f"block{i}.res_norm_w"] = torch.ones(d, dtype=dt)
            out[f"block{i}.res_norm_b"] = torch.zeros(d, dtype=dt)
        out["final_norm_w"] = torch.ones(d, dtype=dt)
        out["final_norm_b"] = torch.zeros(d, dtype=dt)
        return out

    def init_adapters(self, seed: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(seed + 1)
        cfg = self.mcfg; tcfg = self.tcfg
        d = cfg.d_model; r = cfg.lora_r; dt = self.dtype
        oh = tcfg.output_heads; ov = 2 ** tcfg.output_bits
        out: dict[str, Tensor] = {}

        out.update(make_adapter_params("emb", tcfg.vocab_size, d, r, dt))
        for i in range(cfg.num_layers):
            for k in ("W_q", "W_k", "W_v"):
                out.update(make_adapter_params(f"block{i}.{k}", d, d, r, dt))

        op_raw = rng.standard_normal((d, oh * ov)).astype(np.float32) * 0.01
        out["output_proj"] = torch.tensor(op_raw, dtype=dt).requires_grad_(True)
        return out

    def _eff_block(self, frozen, adapters, i) -> dict[str, Tensor]:
        cfg = self.mcfg
        fw: dict[str, Tensor] = {}
        for k in ("W_q", "W_k", "W_v"):
            W0 = frozen[f"block{i}.{k}"]
            fw[k] = apply_adapter(W0, adapters[f"block{i}.{k}.A"],
                                  adapters[f"block{i}.{k}.B"], cfg.use_hira)
        for k in ("eta", "norm_w", "norm_b"):
            fw[k] = frozen[f"block{i}.{k}"]
        return fw

    def forward(self, frozen, adapters, tokens: Tensor,
                states: Any) -> tuple[Tensor, Any]:
        cfg = self.mcfg; tcfg = self.tcfg
        oh  = tcfg.output_heads; ov = 2 ** tcfg.output_bits

        emb = apply_adapter(frozen["embedding"], adapters["emb.A"],
                            adapters["emb.B"], cfg.use_hira)
        x   = F.embedding(tokens, emb)   # [B, S, d]
        B, S, d = x.shape

        new_states = []
        for i in range(cfg.num_layers):
            fw  = self._eff_block(frozen, adapters, i)
            W0  = states[i]
            x_n = _layer_norm(x, frozen[f"block{i}.res_norm_w"],
                              frozen[f"block{i}.res_norm_b"])
            cell, W_new = _ttt_scan(fw, x_n, W0)
            x = x + cell
            new_states.append(W_new)

        x      = _layer_norm(x, frozen["final_norm_w"], frozen["final_norm_b"])
        flat   = x @ adapters["output_proj"]
        logits = flat.reshape(B, S, oh, ov)
        return logits, new_states

    def step(self, frozen, adapters, token: Tensor,
             state: Any, t: int) -> tuple[Tensor, Any]:
        cfg = self.mcfg; tcfg = self.tcfg
        oh  = tcfg.output_heads; ov = 2 ** tcfg.output_bits

        emb = apply_adapter(frozen["embedding"], adapters["emb.A"],
                            adapters["emb.B"], cfg.use_hira)
        x   = F.embedding(token, emb)   # [B, d]
        B   = x.shape[0]

        new_state = []
        for i in range(cfg.num_layers):
            fw  = self._eff_block(frozen, adapters, i)
            W   = state[i]
            x_n = _layer_norm(x, frozen[f"block{i}.res_norm_w"],
                              frozen[f"block{i}.res_norm_b"])
            cell, W_new = _ttt_step(fw, x_n, W)
            x = x + cell
            new_state.append(W_new)

        x      = _layer_norm(x, frozen["final_norm_w"], frozen["final_norm_b"])
        flat   = x @ adapters["output_proj"]
        logits = flat.reshape(B, oh, ov)
        return logits, new_state

    def init_states(self, batch_size: int, device: torch.device) -> list:
        d  = self.mcfg.d_model; dt = self.dtype
        return [torch.zeros(batch_size, d, d, dtype=dt, device=device)
                for _ in range(self.mcfg.num_layers)]

    def precompute_eff_weights(self, frozen, adapters) -> dict:
        cfg = self.mcfg; tcfg = self.tcfg; d = cfg.d_model
        ew: dict[str, Tensor] = {}
        ew["embedding"] = apply_adapter(frozen["embedding"], adapters["emb.A"],
                                        adapters["emb.B"], cfg.use_hira)
        for i in range(cfg.num_layers):
            for k in ("W_q", "W_k", "W_v"):
                W0 = frozen[f"block{i}.{k}"]
                ew[f"block{i}.{k}"] = apply_adapter(
                    W0, adapters[f"block{i}.{k}.A"],
                    adapters[f"block{i}.{k}.B"], cfg.use_hira)
            for k in ("eta", "norm_w", "norm_b", "res_norm_w", "res_norm_b"):
                ew[f"block{i}.{k}"] = frozen[f"block{i}.{k}"]
        ew["final_norm_w"] = frozen["final_norm_w"]
        ew["final_norm_b"] = frozen["final_norm_b"]
        ew["output_proj"]  = adapters["output_proj"]
        return ew

    def step_eff(self, ew: dict, token: Tensor,
                 state: Any, t: int) -> tuple[Tensor, Any]:
        cfg = self.mcfg; tcfg = self.tcfg
        oh  = tcfg.output_heads; ov = 2 ** tcfg.output_bits

        x = ew["embedding"][token]   # [B, d]
        B = x.shape[0]

        new_state = []
        for i in range(self.mcfg.num_layers):
            fw  = {k: ew[f"block{i}.{k}"]
                   for k in ("W_q", "W_k", "W_v", "eta", "norm_w", "norm_b")}
            W   = state[i]
            x_n = _layer_norm(x, ew[f"block{i}.res_norm_w"], ew[f"block{i}.res_norm_b"])
            cell, W_new = _ttt_step(fw, x_n, W)
            x = x + cell
            new_state.append(W_new)

        x      = _layer_norm(x, ew["final_norm_w"], ew["final_norm_b"])
        flat   = x @ ew["output_proj"]
        logits = flat.reshape(B, oh, ov)
        return logits, new_state
