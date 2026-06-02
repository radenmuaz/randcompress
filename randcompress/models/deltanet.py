"""
Gated DeltaNet with vector gate.

Cell (per head n):
  k_t = normalize(W_k x_t)        [B, NH, DH]   (unit-norm key)
  v_t = W_v x_t                   [B, NH, DH]
  q_t = W_q x_t                   [B, NH, DH]
  β_t = sigmoid(W_β x_t)          [B, NH, DH]   (vector gate per head element)

  error_t = C_{t-1} k_t - v_t     [B, NH, DH]   (delta rule residual)
  C_t = C_{t-1} - β_t[...,None] * (error_t[...,None] * k_t[...,None,:])
      = C_{t-1} - β_t[:,None] ⊗ error_t ⊗ k_t^T   (outer-product update)
  output = W_o (q_t ⊙ (C_t k_t))  (gated output)  [B, d_model]

Frozen weights: W_q, W_k, W_v, W_β, W_o  all [NH*DH, d_model] (= [d, d])
State: C ∈ [B, NH, DH, DH]
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
from .msrnn import _ortho, _layer_norm, _multihead_layer_norm


def _init_deltanet_frozen(rng: np.random.Generator, d: int, NH: int,
                          seq_len: int, dtype) -> dict[str, Tensor]:
    DH   = d // NH
    rngs = [np.random.default_rng(rng.integers(2**31)) for _ in range(5)]
    def _qkv(g):
        M = np.array(_ortho(g, d, d, dtype))
        return torch.tensor(M.reshape(NH, DH, d) / math.sqrt(d), dtype=dtype)
    return {
        "W_q":   _qkv(rngs[0]),
        "W_k":   _qkv(rngs[1]),
        "W_v":   _qkv(rngs[2]),
        "W_b":   _qkv(rngs[3]),                          # gate weights
        "W_o":   _ortho(rngs[4], d, d, dtype) / math.sqrt(d),
        "skip":  torch.ones(d, dtype=dtype),
        "ln_w":  torch.ones(d, dtype=dtype),
        "ln_b":  torch.zeros(d, dtype=dtype),
    }


def _delta_step(fw: dict, x_t: Tensor, C: Tensor,
                NH: int) -> tuple[Tensor, Tensor]:
    """Single DeltaNet step. x_t: [B, d]. Returns (out [B, d], C_new)."""
    B, d = x_t.shape
    DH   = d // NH

    q = torch.einsum("ndi,bi->bnd", fw["W_q"], x_t)   # [B, NH, DH]
    k = torch.einsum("ndi,bi->bnd", fw["W_k"], x_t)
    v = torch.einsum("ndi,bi->bnd", fw["W_v"], x_t)
    b = torch.sigmoid(torch.einsum("ndi,bi->bnd", fw["W_b"], x_t))  # [B, NH, DH]

    # Normalize key
    k = F.normalize(k, dim=-1)

    # Delta rule: residual = C k - v,  update C by subtracting beta * residual ⊗ k^T
    Ck    = torch.einsum("bnde,bne->bnd", C, k)        # [B, NH, DH]
    error = Ck - v                                      # [B, NH, DH]
    # delta: [B, NH, DH, DH] = beta * error ⊗ k^T
    delta = torch.einsum("bnd,bne->bnde", b * error, k)
    C_new = C - delta

    # Output: q ⊙ (C_new k) → reshape → W_o
    h = torch.einsum("bnde,bne->bnd", C_new, q)        # [B, NH, DH]
    h_flat  = h.reshape(B, d)
    h_out   = h_flat + fw["skip"] * x_t
    h_norm  = _multihead_layer_norm(h_out, fw["ln_w"], fw["ln_b"], NH)
    out     = h_norm @ fw["W_o"].T
    return out, C_new


def _delta_scan(fw: dict, x: Tensor, NH: int,
                C0: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """DeltaNet scan. x: [B, S, d]. Returns (out [B, S, d], C_T)."""
    B, S, d = x.shape
    DH = d // NH
    if C0 is None:
        C0 = torch.zeros(B, NH, DH, DH, dtype=x.dtype, device=x.device)

    # Pre-project inputs
    q_all = torch.einsum("ndi,bsi->bnsd", fw["W_q"], x)
    k_all = torch.einsum("ndi,bsi->bnsd", fw["W_k"], x)
    v_all = torch.einsum("ndi,bsi->bnsd", fw["W_v"], x)
    b_all = torch.sigmoid(torch.einsum("ndi,bsi->bnsd", fw["W_b"], x))
    k_all = F.normalize(k_all, dim=-1)

    C  = C0
    hs = []
    for t in range(S):
        q = q_all[:, :, t, :]
        k = k_all[:, :, t, :]
        v = v_all[:, :, t, :]
        b = b_all[:, :, t, :]
        Ck    = torch.einsum("bnde,bne->bnd", C, k)
        error = Ck - v
        delta = torch.einsum("bnd,bne->bnde", b * error, k)
        C     = C - delta
        h     = torch.einsum("bnde,bne->bnd", C, q)
        hs.append(h)

    hs_t   = torch.stack(hs, dim=2)              # [B, NH, S, DH]
    h_flat = hs_t.permute(0, 2, 1, 3).reshape(B, S, d)
    h_out  = h_flat + fw["skip"] * x
    h_norm = _multihead_layer_norm(h_out, fw["ln_w"], fw["ln_b"], NH)
    out    = h_norm @ fw["W_o"].T
    return out, C


class GatedDeltaNet(RandCompressModel):

    def __init__(self, model_cfg, train_cfg):
        self.mcfg  = model_cfg
        self.tcfg  = train_cfg
        self.dtype = torch.float32 if train_cfg.dtype == "float32" else torch.bfloat16

    def init_frozen(self, seed: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(seed)
        cfg = self.mcfg; tcfg = self.tcfg
        d = cfg.d_model; NH = cfg.num_heads; dt = self.dtype
        out: dict[str, Tensor] = {}
        emb_raw = rng.standard_normal((tcfg.vocab_size, d)).astype(np.float32) * 0.01
        out["embedding"] = torch.tensor(emb_raw, dtype=dt)
        for i in range(cfg.num_layers):
            sub = np.random.default_rng(rng.integers(2**31))
            fw  = _init_deltanet_frozen(sub, d, NH, tcfg.segment_size, dt)
            for k, v in fw.items(): out[f"block{i}.{k}"] = v
            out[f"block{i}.norm_w"] = torch.ones(d, dtype=dt)
            out[f"block{i}.norm_b"] = torch.zeros(d, dtype=dt)
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
            for k in ("W_q", "W_k", "W_v", "W_b", "W_o"):
                out.update(make_adapter_params(f"block{i}.{k}", d, d, r, dt))
        op_raw = rng.standard_normal((d, oh * ov)).astype(np.float32) * 0.01
        out["output_proj"] = torch.tensor(op_raw, dtype=dt).requires_grad_(True)
        return out

    def _eff_block(self, frozen, adapters, i) -> dict[str, Tensor]:
        cfg = self.mcfg; d = cfg.d_model; NH = cfg.num_heads; DH = d // NH
        fw: dict[str, Tensor] = {}
        for k in ("W_q", "W_k", "W_v", "W_b"):
            W0   = frozen[f"block{i}.{k}"]
            flat = apply_adapter(W0.reshape(NH * DH, d),
                                 adapters[f"block{i}.{k}.A"],
                                 adapters[f"block{i}.{k}.B"], cfg.use_hira)
            fw[k] = flat.reshape(W0.shape)
        fw["W_o"] = apply_adapter(frozen[f"block{i}.W_o"],
                                  adapters[f"block{i}.W_o.A"],
                                  adapters[f"block{i}.W_o.B"], cfg.use_hira)
        for k in ("skip", "ln_w", "ln_b"):
            fw[k] = frozen[f"block{i}.{k}"]
        return fw

    def forward(self, frozen, adapters, tokens: Tensor,
                states: Any) -> tuple[Tensor, Any]:
        cfg = self.mcfg; tcfg = self.tcfg
        oh  = tcfg.output_heads; ov = 2 ** tcfg.output_bits; NH = cfg.num_heads

        emb = apply_adapter(frozen["embedding"], adapters["emb.A"],
                            adapters["emb.B"], cfg.use_hira)
        x   = F.embedding(tokens, emb)
        B, S, d = x.shape

        new_states = []
        for i in range(cfg.num_layers):
            fw  = self._eff_block(frozen, adapters, i)
            C0  = states[i]
            x_n = _layer_norm(x, frozen[f"block{i}.norm_w"], frozen[f"block{i}.norm_b"])
            cell, C_new = _delta_scan(fw, x_n, NH, C0)
            x = x + cell
            new_states.append(C_new)

        x      = _layer_norm(x, frozen["final_norm_w"], frozen["final_norm_b"])
        flat   = x @ adapters["output_proj"]
        logits = flat.reshape(B, S, oh, ov)
        return logits, new_states

    def step(self, frozen, adapters, token: Tensor,
             state: Any, t: int) -> tuple[Tensor, Any]:
        cfg = self.mcfg; tcfg = self.tcfg
        oh  = tcfg.output_heads; ov = 2 ** tcfg.output_bits; NH = cfg.num_heads

        emb = apply_adapter(frozen["embedding"], adapters["emb.A"],
                            adapters["emb.B"], cfg.use_hira)
        x   = F.embedding(token, emb)
        B   = x.shape[0]

        new_state = []
        for i in range(cfg.num_layers):
            fw  = self._eff_block(frozen, adapters, i)
            C   = state[i]
            x_n = _layer_norm(x, frozen[f"block{i}.norm_w"], frozen[f"block{i}.norm_b"])
            cell, C_new = _delta_step(fw, x_n, C, NH)
            x = x + cell
            new_state.append(C_new)

        x      = _layer_norm(x, frozen["final_norm_w"], frozen["final_norm_b"])
        flat   = x @ adapters["output_proj"]
        logits = flat.reshape(B, oh, ov)
        return logits, new_state

    def init_states(self, batch_size: int, device: torch.device) -> list:
        cfg = self.mcfg; d = cfg.d_model; NH = cfg.num_heads; DH = d // NH; dt = self.dtype
        return [torch.zeros(batch_size, NH, DH, DH, dtype=dt, device=device)
                for _ in range(cfg.num_layers)]

    def precompute_eff_weights(self, frozen, adapters) -> dict:
        cfg = self.mcfg; d = cfg.d_model; NH = cfg.num_heads; DH = d // NH
        ew: dict[str, Tensor] = {}
        ew["embedding"] = apply_adapter(frozen["embedding"], adapters["emb.A"],
                                        adapters["emb.B"], cfg.use_hira)
        for i in range(cfg.num_layers):
            for k in ("W_q", "W_k", "W_v", "W_b"):
                W0   = frozen[f"block{i}.{k}"]
                flat = apply_adapter(W0.reshape(NH * DH, d),
                                     adapters[f"block{i}.{k}.A"],
                                     adapters[f"block{i}.{k}.B"], cfg.use_hira)
                ew[f"block{i}.{k}"] = flat.reshape(W0.shape)
            ew[f"block{i}.W_o"] = apply_adapter(
                frozen[f"block{i}.W_o"],
                adapters[f"block{i}.W_o.A"],
                adapters[f"block{i}.W_o.B"], cfg.use_hira)
            for k in ("skip", "ln_w", "ln_b", "norm_w", "norm_b"):
                ew[f"block{i}.{k}"] = frozen[f"block{i}.{k}"]
        ew["final_norm_w"] = frozen["final_norm_w"]
        ew["final_norm_b"] = frozen["final_norm_b"]
        ew["output_proj"]  = adapters["output_proj"]
        return ew

    def step_eff(self, ew: dict, token: Tensor,
                 state: Any, t: int) -> tuple[Tensor, Any]:
        cfg = self.mcfg; tcfg = self.tcfg
        oh  = tcfg.output_heads; ov = 2 ** tcfg.output_bits; NH = cfg.num_heads

        x = ew["embedding"][token]
        B = x.shape[0]

        new_state = []
        for i in range(cfg.num_layers):
            fw  = {k: ew[f"block{i}.{k}"]
                   for k in ("W_q", "W_k", "W_v", "W_b", "W_o", "skip", "ln_w", "ln_b")}
            C   = state[i]
            x_n = _layer_norm(x, ew[f"block{i}.norm_w"], ew[f"block{i}.norm_b"])
            cell, C_new = _delta_step(fw, x_n, C, NH)
            x = x + cell
            new_state.append(C_new)

        x      = _layer_norm(x, ew["final_norm_w"], ew["final_norm_b"])
        flat   = x @ ew["output_proj"]
        logits = flat.reshape(B, oh, ov)
        return logits, new_state
