"""
MsRNN — Multi-Scale RNN: gate-free linear mLSTM + whitened sRNN.

Port of examples_old/linear_rnn_argmax.py (v10) to PyTorch.

block_map chars:
  'm' — linear mLSTM: matrix memory C, symmetric power-p kernel, per-head decay alpha
  's' — whitened sRNN: h_t = LayerNorm(W_hh h + W_hx x + b)

Frozen weights fully determine the model given seed + config.
Adapters: HiRA (W = W₀ + W₀⊙BA) or LoRA (W = W₀ + BA), same param count.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import RandCompressModel
from .hira import apply_adapter, make_adapter_params, center_b, init_hira_A, init_hira_B


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ortho(rng: np.random.Generator, n: int, m: int, dtype) -> Tensor:
    raw = rng.standard_normal((max(n, m), min(n, m)))
    Q, _ = np.linalg.qr(raw)
    mat = Q.T
    if n > m:
        extra = rng.standard_normal((n - m, m)) / math.sqrt(m)
        mat = np.concatenate([mat, extra], axis=0)
    return torch.tensor(mat[:n], dtype=dtype)


def _dsym(DH: int, p: int) -> int:
    if p == 1:
        return DH
    return math.comb(DH + p - 1, p)


def _spow(x: Tensor, p: int) -> Tensor:
    """Symmetric degree-p polynomial feature map. [..., D] → [..., D_sym]."""
    if p == 1:
        return x
    D = x.shape[-1]
    i_idx, j_idx = np.tril_indices(D)
    scale = np.where(i_idx == j_idx, 1.0, math.sqrt(2.0)).astype(np.float32)
    scale_t = torch.tensor(scale, dtype=x.dtype, device=x.device)
    return x[..., i_idx] * x[..., j_idx] * scale_t


def _layer_norm(x: Tensor, w: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var  = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (var + eps).sqrt() * w + b


def _multihead_layer_norm(x: Tensor, w: Tensor, b: Tensor,
                          num_heads: int, eps: float = 1e-6) -> Tensor:
    *lead, D = x.shape
    x_r  = x.reshape(*lead, num_heads, D // num_heads)
    mean = x_r.mean(dim=-1, keepdim=True)
    var  = x_r.var(dim=-1, keepdim=True, unbiased=False)
    return ((x_r - mean) / (var + eps).sqrt()).reshape(*lead, D) * w + b


# ── Frozen init ───────────────────────────────────────────────────────────────

def _init_mlstm_frozen(rng: np.random.Generator, d: int, NH: int,
                       seq_len: int, dtype) -> dict[str, Tensor]:
    DH = d // NH
    def qkv(g):
        M = np.array(_ortho(g, d, d, dtype))
        return torch.tensor(M.reshape(NH, DH, d) / math.sqrt(d), dtype=dtype)

    rngs = [np.random.default_rng(rng.integers(2**31)) for _ in range(5)]
    W_q = qkv(rngs[0]); W_k = qkv(rngs[1]); W_v = qkv(rngs[2])
    tau   = np.exp(np.linspace(0.0, np.log(max(float(seq_len), 2.0)), NH))
    alpha = torch.tensor(np.exp(-1.0 / tau), dtype=dtype)
    W_down = _ortho(rngs[3], d, d, dtype) / math.sqrt(d)
    return {
        "W_q": W_q, "W_k": W_k, "W_v": W_v,
        "alpha": alpha,
        "skip":  torch.ones(d, dtype=dtype),
        "ln_w":  torch.ones(d, dtype=dtype),
        "ln_b":  torch.zeros(d, dtype=dtype),
        "W_down": W_down,
    }


def _init_srnn_frozen(rng: np.random.Generator, d: int, dtype) -> dict[str, Tensor]:
    rngs = [np.random.default_rng(rng.integers(2**31)) for _ in range(2)]
    W_hx = _ortho(rngs[0], d, d, dtype) / math.sqrt(d)
    raw_hh = rngs[1].standard_normal((d, d))
    U, _, Vt = np.linalg.svd(raw_hh, full_matrices=False)
    W_hh = torch.tensor(0.95 * (U @ Vt), dtype=dtype)
    return {
        "W_hx": W_hx,
        "W_hh": W_hh,
        "b":    torch.zeros(d, dtype=dtype),
        "ln_w": torch.ones(d, dtype=dtype),
        "ln_b": torch.zeros(d, dtype=dtype),
    }


# ── Forward passes ────────────────────────────────────────────────────────────

def _mlstm_scan(fw: dict, x: Tensor, NH: int, p: int,
                C0: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """mLSTM scan. x: [B, S, d]. Returns (out [B, S, d], C_T).

    Pre-projects Q/K/V for the full sequence, then scans recurrently.
    The Python loop is over time steps only; all per-step ops are batched.
    """
    B, S, d = x.shape
    DH    = d // NH
    D_sym = _dsym(DH, p)
    alpha = fw["alpha"]   # [NH]

    # Project all at once: [B, NH, S, DH]
    q = torch.einsum("ndi,bsi->bnsd", fw["W_q"], x)
    k = torch.einsum("ndi,bsi->bnsd", fw["W_k"], x)
    v = torch.einsum("ndi,bsi->bnsd", fw["W_v"], x)

    # Feature maps for full sequence: [B, NH, S, D_sym]
    phi_q = _spow(q, p)
    phi_k = _spow(k / math.sqrt(DH), p)

    if C0 is None:
        C0 = torch.zeros(B, NH, D_sym, DH, dtype=x.dtype, device=x.device)

    C  = C0
    hs = []
    a  = alpha[:, None, None]   # [NH, 1, 1] for broadcasting
    for t in range(S):
        kv    = torch.einsum("bnd,bne->bnde", phi_k[:, :, t], v[:, :, t])
        C     = a * C + kv
        h_num = torch.einsum("bnd,bnde->bne", phi_q[:, :, t], C)
        hs.append(h_num)

    hs_t   = torch.stack(hs, dim=2)              # [B, NH, S, DH]
    h_flat = hs_t.permute(0, 2, 1, 3).reshape(B, S, d)
    h_out  = h_flat + fw["skip"] * x
    h_norm = _multihead_layer_norm(h_out, fw["ln_w"], fw["ln_b"], NH)
    out    = h_norm @ fw["W_down"].T
    return out, C


def _mlstm_step(fw: dict, x_t: Tensor, C: Tensor,
                NH: int, p: int) -> tuple[Tensor, Tensor]:
    """Single mLSTM step. x_t: [B, d]. Returns (out [B, d], C_new)."""
    B, d = x_t.shape
    DH   = d // NH
    q = torch.einsum("ndi,bi->bnd", fw["W_q"], x_t)
    k = torch.einsum("ndi,bi->bnd", fw["W_k"], x_t)
    v = torch.einsum("ndi,bi->bnd", fw["W_v"], x_t)
    k_s   = k / math.sqrt(DH)
    phi_k = _spow(k_s, p)
    phi_q = _spow(q, p)
    kv    = torch.einsum("bnd,bne->bnde", phi_k, v)
    C_new = fw["alpha"][:, None, None] * C + kv
    h_num = torch.einsum("bnd,bnde->bne", phi_q, C_new)
    h_flat = h_num.reshape(B, d)
    h_out  = h_flat + fw["skip"] * x_t
    h_norm = _multihead_layer_norm(h_out, fw["ln_w"], fw["ln_b"], NH)
    out    = h_norm @ fw["W_down"].T
    return out, C_new


def _srnn_scan(fw: dict, x: Tensor,
               h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """sRNN scan. x: [B, S, d]. Returns (out [B, S, d], h_T).

    Sequential by nature (recurrent). Uses a Python loop but pre-projects
    all x @ W_hx.T up-front to amortize that cost.
    """
    B, S, d = x.shape
    if h0 is None:
        h0 = torch.zeros(B, d, dtype=x.dtype, device=x.device)

    # Pre-project inputs without b (add b inside loop to match _srnn_step order)
    x_proj = x @ fw["W_hx"].T   # [B, S, d]

    h  = h0
    hs = []
    for t in range(S):
        # Matches _srnn_step: (x_t @ W_hx.T + h @ W_hh.T) + b
        pre = x_proj[:, t, :] + h @ fw["W_hh"].T + fw["b"]
        h   = _layer_norm(pre, fw["ln_w"], fw["ln_b"])
        hs.append(h)
    out = torch.stack(hs, dim=1)   # [B, S, d]
    return out, h


def _srnn_step(fw: dict, x_t: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
    """Single sRNN step. x_t: [B, d]. Returns (h_new [B, d], h_new)."""
    pre   = x_t @ fw["W_hx"].T + h @ fw["W_hh"].T + fw["b"]
    h_new = _layer_norm(pre, fw["ln_w"], fw["ln_b"])
    return h_new, h_new


# ── MsRNN model ───────────────────────────────────────────────────────────────

class MsRNN(RandCompressModel):

    def __init__(self, model_cfg, train_cfg):
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.dtype = torch.float32 if train_cfg.dtype == "float32" else torch.bfloat16

    # ── frozen init ───────────────────────────────────────────────────────────

    def init_frozen(self, seed: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(seed)
        cfg = self.mcfg
        tcfg = self.tcfg
        d, NH = cfg.d_model, cfg.num_heads
        dt = self.dtype
        out: dict[str, Tensor] = {}

        # embedding
        emb_raw = rng.standard_normal((tcfg.vocab_size, d)).astype(np.float32) * 0.01
        out["embedding"] = torch.tensor(emb_raw, dtype=dt)

        for i, btype in enumerate(cfg.block_map):
            out[f"block{i}.norm_w"] = torch.ones(d, dtype=dt)
            out[f"block{i}.norm_b"] = torch.zeros(d, dtype=dt)
            sub_rng = np.random.default_rng(rng.integers(2**31))
            if btype == "m":
                fw = _init_mlstm_frozen(sub_rng, d, NH, tcfg.segment_size, dt)
                for k, v in fw.items():
                    out[f"block{i}.mlstm.{k}"] = v
            else:
                fw = _init_srnn_frozen(sub_rng, d, dt)
                for k, v in fw.items():
                    out[f"block{i}.srnn.{k}"] = v

        out["final_norm_w"] = torch.ones(d, dtype=dt)
        out["final_norm_b"] = torch.zeros(d, dtype=dt)
        return out

    # ── adapter init ──────────────────────────────────────────────────────────

    def init_adapters(self, seed: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(seed + 1)
        cfg = self.mcfg
        tcfg = self.tcfg
        d, r = cfg.d_model, cfg.lora_r
        V  = tcfg.vocab_size
        oh = tcfg.output_heads
        ov = 2 ** tcfg.output_bits
        dt = self.dtype
        out: dict[str, Tensor] = {}

        def _add(prefix, d_out, d_in):
            pair = make_adapter_params(prefix, d_out, d_in, r, dt)
            out.update(pair)

        _add("emb", V, d)

        for i, btype in enumerate(cfg.block_map):
            if btype == "m":
                for name in ("W_q", "W_k", "W_v", "W_down"):
                    _add(f"block{i}.mlstm.{name}", d, d)
            else:
                for name in ("W_hx", "W_hh"):
                    _add(f"block{i}.srnn.{name}", d, d)

        # output_proj: always direct param (no HiRA wrapper)
        op_raw = rng.standard_normal((d, oh * ov)).astype(np.float32) * 0.01
        out["output_proj"] = torch.tensor(op_raw, dtype=dt).requires_grad_(True)

        return out

    # ── effective weights ─────────────────────────────────────────────────────

    def _eff_emb(self, frozen, adapters) -> Tensor:
        W0 = frozen["embedding"]
        return apply_adapter(W0, adapters["emb.A"], adapters["emb.B"], self.mcfg.use_hira)

    def _eff_mlstm(self, frozen, adapters, i) -> dict[str, Tensor]:
        fw: dict[str, Tensor] = {}
        for k in ("W_q", "W_k", "W_v", "W_down"):
            W0 = frozen[f"block{i}.mlstm.{k}"]
            NH, DH, d = (self.mcfg.num_heads, self.mcfg.d_model // self.mcfg.num_heads,
                         self.mcfg.d_model)
            flat = apply_adapter(
                W0.reshape(NH * DH, d),
                adapters[f"block{i}.mlstm.{k}.A"],
                adapters[f"block{i}.mlstm.{k}.B"],
                self.mcfg.use_hira,
            )
            fw[k] = flat.reshape(W0.shape)
        for k in ("alpha", "skip", "ln_w", "ln_b"):
            fw[k] = frozen[f"block{i}.mlstm.{k}"]
        return fw

    def _eff_srnn(self, frozen, adapters, i) -> dict[str, Tensor]:
        fw: dict[str, Tensor] = {}
        for k in ("W_hx", "W_hh"):
            W0 = frozen[f"block{i}.srnn.{k}"]
            fw[k] = apply_adapter(
                W0,
                adapters[f"block{i}.srnn.{k}.A"],
                adapters[f"block{i}.srnn.{k}.B"],
                self.mcfg.use_hira,
            )
        for k in ("b", "ln_w", "ln_b"):
            fw[k] = frozen[f"block{i}.srnn.{k}"]
        return fw

    # ── parse stride map ──────────────────────────────────────────────────────

    def _strides(self) -> list[int]:
        sm = self.mcfg.stride_map
        return [int(c) for c in sm[: self.mcfg.num_layers]]

    # ── forward (chunked train) ───────────────────────────────────────────────

    def forward(self, frozen, adapters, tokens: Tensor,
                states: Any) -> tuple[Tensor, Any]:
        cfg   = self.mcfg
        tcfg  = self.tcfg
        oh    = tcfg.output_heads
        ov    = 2 ** tcfg.output_bits
        strides = self._strides()

        emb = self._eff_emb(frozen, adapters)
        x   = F.embedding(tokens, emb)              # [B, S, d]
        B, S, _ = x.shape

        new_states = []
        for i, (btype, stride) in enumerate(zip(cfg.block_map, strides)):
            lstm_state, _last_out = states[i]
            x_n = _layer_norm(x, frozen[f"block{i}.norm_w"], frozen[f"block{i}.norm_b"])

            if stride > 1:
                x_sub = x_n[:, ::stride, :]
                if btype == "m":
                    fw = self._eff_mlstm(frozen, adapters, i)
                    cell_sub, new_lstm = _mlstm_scan(fw, x_sub, cfg.num_heads,
                                                     cfg.power_p, lstm_state)
                else:
                    fw = self._eff_srnn(frozen, adapters, i)
                    cell_sub, new_lstm = _srnn_scan(fw, x_sub, lstm_state)
                cell = cell_sub.repeat_interleave(stride, dim=1)[:, :S, :]
            else:
                if btype == "m":
                    fw = self._eff_mlstm(frozen, adapters, i)
                    cell, new_lstm = _mlstm_scan(fw, x_n, cfg.num_heads,
                                                 cfg.power_p, lstm_state)
                else:
                    fw = self._eff_srnn(frozen, adapters, i)
                    cell, new_lstm = _srnn_scan(fw, x_n, lstm_state)

            new_states.append((new_lstm, cell[:, -1, :]))
            x = x + cell

        x    = _layer_norm(x, frozen["final_norm_w"], frozen["final_norm_b"])
        flat = x @ adapters["output_proj"]           # [B, S, oh*ov]
        logits = flat.reshape(B, S, oh, ov)
        return logits, new_states

    # ── single step (AR decode) ───────────────────────────────────────────────

    def step(self, frozen, adapters, token: Tensor,
             state: Any, t: int) -> tuple[Tensor, Any]:
        cfg     = self.mcfg
        tcfg    = self.tcfg
        oh      = tcfg.output_heads
        ov      = 2 ** tcfg.output_bits
        strides = self._strides()

        emb = self._eff_emb(frozen, adapters)
        x   = F.embedding(token, emb)               # [B, d]

        new_state = []
        for i, (btype, stride) in enumerate(zip(cfg.block_map, strides)):
            lstm_state, last_out = state[i]
            x_n = _layer_norm(x, frozen[f"block{i}.norm_w"], frozen[f"block{i}.norm_b"])

            if stride == 1:
                if btype == "m":
                    fw = self._eff_mlstm(frozen, adapters, i)
                    cell, new_lstm = _mlstm_step(fw, x_n, lstm_state,
                                                 cfg.num_heads, cfg.power_p)
                else:
                    fw = self._eff_srnn(frozen, adapters, i)
                    cell, new_lstm = _srnn_step(fw, x_n, lstm_state)
                new_state.append((new_lstm, cell))
            else:
                if t % stride == 0:
                    if btype == "m":
                        fw = self._eff_mlstm(frozen, adapters, i)
                        cell, new_lstm = _mlstm_step(fw, x_n, lstm_state,
                                                     cfg.num_heads, cfg.power_p)
                    else:
                        fw = self._eff_srnn(frozen, adapters, i)
                        cell, new_lstm = _srnn_step(fw, x_n, lstm_state)
                    new_state.append((new_lstm, cell))
                else:
                    new_state.append((lstm_state, last_out))
                    cell = last_out

            x = x + cell

        x      = _layer_norm(x, frozen["final_norm_w"], frozen["final_norm_b"])
        flat   = x @ adapters["output_proj"]         # [B, oh*ov]
        logits = flat.reshape(token.shape[0], oh, ov)
        return logits, new_state

    # ── precompute effective weights for fast collect ─────────────────────────

    def precompute_eff_weights(self, frozen, adapters) -> dict:
        """Apply HiRA/LoRA once, return all effective weights as a flat dict.
        Called once before the collect_logits loop; eliminates O(T) adapter math.
        """
        cfg = self.mcfg
        ew: dict[str, Tensor] = {}

        # Embedding
        W0 = frozen["embedding"]
        ew["embedding"] = apply_adapter(W0, adapters["emb.A"], adapters["emb.B"],
                                        cfg.use_hira)

        for i, btype in enumerate(cfg.block_map):
            ew[f"block{i}.norm_w"] = frozen[f"block{i}.norm_w"]
            ew[f"block{i}.norm_b"] = frozen[f"block{i}.norm_b"]
            if btype == "m":
                NH, DH, d = cfg.num_heads, cfg.d_model // cfg.num_heads, cfg.d_model
                for k in ("W_q", "W_k", "W_v", "W_down"):
                    W0 = frozen[f"block{i}.mlstm.{k}"]
                    flat = apply_adapter(
                        W0.reshape(NH * DH, d),
                        adapters[f"block{i}.mlstm.{k}.A"],
                        adapters[f"block{i}.mlstm.{k}.B"],
                        cfg.use_hira,
                    )
                    ew[f"block{i}.mlstm.{k}"] = flat.reshape(W0.shape)
                for k in ("alpha", "skip", "ln_w", "ln_b"):
                    ew[f"block{i}.mlstm.{k}"] = frozen[f"block{i}.mlstm.{k}"]
            else:
                for k in ("W_hx", "W_hh"):
                    W0 = frozen[f"block{i}.srnn.{k}"]
                    ew[f"block{i}.srnn.{k}"] = apply_adapter(
                        W0,
                        adapters[f"block{i}.srnn.{k}.A"],
                        adapters[f"block{i}.srnn.{k}.B"],
                        cfg.use_hira,
                    )
                for k in ("b", "ln_w", "ln_b"):
                    ew[f"block{i}.srnn.{k}"] = frozen[f"block{i}.srnn.{k}"]

        ew["final_norm_w"] = frozen["final_norm_w"]
        ew["final_norm_b"] = frozen["final_norm_b"]
        ew["output_proj"]  = adapters["output_proj"]
        return ew

    def step_eff(self, ew: dict, token: Tensor, state: list,
                 t: int) -> tuple[Tensor, list]:
        """Same as step() but uses pre-computed effective weights (no adapter math)."""
        cfg     = self.mcfg
        tcfg    = self.tcfg
        oh      = tcfg.output_heads
        ov      = 2 ** tcfg.output_bits
        strides = self._strides()

        x         = ew["embedding"][token]
        new_state = []

        for i, (btype, stride) in enumerate(zip(cfg.block_map, strides)):
            lstm_state, last_out = state[i]
            x_n = _layer_norm(x, ew[f"block{i}.norm_w"], ew[f"block{i}.norm_b"])

            if btype == "m":
                fw = {k: ew[f"block{i}.mlstm.{k}"]
                      for k in ("W_q", "W_k", "W_v", "W_down",
                                "alpha", "skip", "ln_w", "ln_b")}
            else:
                fw = {k: ew[f"block{i}.srnn.{k}"]
                      for k in ("W_hx", "W_hh", "b", "ln_w", "ln_b")}

            if stride == 1:
                if btype == "m":
                    cell, new_lstm = _mlstm_step(fw, x_n, lstm_state,
                                                 cfg.num_heads, cfg.power_p)
                else:
                    cell, new_lstm = _srnn_step(fw, x_n, lstm_state)
                new_state.append((new_lstm, cell))
            else:
                if t % stride == 0:
                    if btype == "m":
                        cell, new_lstm = _mlstm_step(fw, x_n, lstm_state,
                                                     cfg.num_heads, cfg.power_p)
                    else:
                        cell, new_lstm = _srnn_step(fw, x_n, lstm_state)
                    new_state.append((new_lstm, cell))
                else:
                    new_state.append((lstm_state, last_out))
                    cell = last_out
            x = x + cell

        x      = _layer_norm(x, ew["final_norm_w"], ew["final_norm_b"])
        flat   = x @ ew["output_proj"]
        logits = flat.reshape(token.shape[0], oh, ov)
        return logits, new_state

    # ── state init ────────────────────────────────────────────────────────────

    def init_states(self, batch_size: int, device: torch.device) -> list:
        cfg = self.mcfg
        d, NH = cfg.d_model, cfg.num_heads
        DH    = d // NH
        dt    = self.dtype
        states = []
        for btype in cfg.block_map:
            if btype == "m":
                D_sym = _dsym(DH, cfg.power_p)
                lstm_state = torch.zeros(batch_size, NH, D_sym, DH, dtype=dt, device=device)
            else:
                lstm_state = torch.zeros(batch_size, d, dtype=dt, device=device)
            last_out = torch.zeros(batch_size, d, dtype=dt, device=device)
            states.append((lstm_state, last_out))
        return states
