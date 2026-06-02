"""
Causal Transformer with YaRN RoPE + SwiGLU FFN.

Supports:
  kv_window    — sliding window (N attention slots after dilation; 0 = full cache)
  attn_dilation — attend every d-th past position (1 = contiguous)
  n_sinks_zero  — constant-zero KV sinks (no params, always score exp(0))
  n_sinks_train — trainable KV sinks (learned key and value vectors)

Effective context per layer: 1 + (kv_window - 1) × attn_dilation
Stacked L layers:            1 + L × (kv_window - 1) × attn_dilation

Sinks are prepended to every attention call (training and inference), never
evicted by the sliding window → zero train/test mismatch.

Frozen weights per layer: W_q, W_k, W_v, W_o, W_gate, W_up, W_down,
                          sink_k [n_train, NH, DH], sink_v [n_train, NH, DH]
Adapters: HiRA/LoRA on W_q, W_k, W_v, W_o, W_gate, W_up, W_down + embedding.
State: list of (K_cache, V_cache) per layer; only content tokens are stored.
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


# ── RoPE ─────────────────────────────────────────────────────────────────────

def _rope_freqs(DH: int, rope_scale: float, base: float = 10000.0) -> Tensor:
    i = torch.arange(0, DH, 2, dtype=torch.float32)
    return 1.0 / (rope_scale * (base ** (i / DH)))   # [DH//2]


def _apply_rope(x: Tensor, freqs: Tensor, offset: int = 0) -> Tensor:
    """x: [B, T, NH, DH]. offset: starting position index."""
    B, T, NH, DH = x.shape
    t      = torch.arange(offset, offset + T, dtype=freqs.dtype, device=x.device)
    angles = torch.outer(t, freqs).view(1, T, 1, -1)  # [1, T, 1, DH//2]
    cos, sin = angles.cos(), angles.sin()
    x1, x2  = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(B, T, NH, DH)


def _apply_rope_1(x: Tensor, freqs: Tensor, t: int) -> Tensor:
    """x: [B, NH, DH] single step at position t."""
    angle    = (freqs * t).view(1, 1, -1)   # [1, 1, DH//2]
    cos, sin = angle.cos(), angle.sin()
    x1, x2  = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(x.shape)


# ── Dilated window gather ──────────────────────────────────────────────────────

def _gather_dilated(K: Tensor, V: Tensor,
                    kv_window: int, dilation: int) -> tuple[Tensor, Tensor]:
    """Select past KV with optional sliding window + dilation.

    kv_window: max attention slots (0 = all). dilation: step between positions.
    Reads backwards from the most recent position.
    Returns K_sel, V_sel: [B, W', NH, DH] in chronological order.
    """
    T = K.shape[1]
    if T == 0:
        return K, V

    if dilation <= 1:
        if kv_window > 0:
            return K[:, -kv_window:], V[:, -kv_window:]
        return K, V

    # How many slots to select
    n_slots = kv_window if kv_window > 0 else (T + dilation - 1) // dilation
    # Indices from the end, step backwards by dilation
    end = T - 1
    idx = torch.arange(end, max(-1, end - n_slots * dilation), -dilation,
                       device=K.device)[:n_slots].flip(0)
    return K[:, idx], V[:, idx]


# ── Sink helpers ──────────────────────────────────────────────────────────────

def _build_sinks(fw: dict, n_zero: int, n_train: int,
                 B: int, NH: int, DH: int, dtype, device) -> tuple[list, list]:
    """Return (K_parts, V_parts) lists for concatenation."""
    ks, vs = [], []
    if n_zero > 0:
        z = torch.zeros(B, n_zero, NH, DH, dtype=dtype, device=device)
        ks.append(z); vs.append(z)
    if n_train > 0 and "sink_k" in fw:
        ks.append(fw["sink_k"].unsqueeze(0).expand(B, -1, -1, -1))
        vs.append(fw["sink_v"].unsqueeze(0).expand(B, -1, -1, -1))
    return ks, vs


# ── Frozen init ───────────────────────────────────────────────────────────────

def _init_attn_frozen(rng: np.random.Generator, d: int, NH: int, dtype) -> dict:
    DH   = d // NH
    rngs = [np.random.default_rng(rng.integers(2**31)) for _ in range(4)]
    fw: dict[str, Tensor] = {}
    for name, g in zip(("W_q", "W_k", "W_v"), rngs[:3]):
        fw[name] = torch.tensor(
            np.array(_ortho(g, d, d, dtype)).reshape(NH, DH, d) / math.sqrt(d),
            dtype=dtype)
    fw["W_o"]   = _ortho(rngs[3], d, d, dtype) / math.sqrt(d)
    fw["norm_w"] = torch.ones(d, dtype=dtype)
    fw["norm_b"] = torch.zeros(d, dtype=dtype)
    return fw


def _init_ffn_frozen(rng: np.random.Generator, d: int, ffn_mult: int, dtype) -> dict:
    d_ff = d * ffn_mult
    rngs = [np.random.default_rng(rng.integers(2**31)) for _ in range(3)]
    return {
        "W_gate":  _ortho(rngs[0], d_ff, d, dtype) / math.sqrt(d),
        "W_up":    _ortho(rngs[1], d_ff, d, dtype) / math.sqrt(d),
        "W_down":  _ortho(rngs[2], d, d_ff, dtype) / math.sqrt(d_ff),
        "norm_w":  torch.ones(d, dtype=dtype),
        "norm_b":  torch.zeros(d, dtype=dtype),
    }


# ── Core attention ─────────────────────────────────────────────────────────────

def _attend(Q: Tensor, K_full: Tensor, V_full: Tensor,
            n_prefix: int, S: int, DH: int) -> Tensor:
    """Compute causal attention.

    Q:        [B, NH, S, DH]
    K_full:   [B, NH, n_prefix+S, DH]  (sinks + past cache + chunk)
    V_full:   same
    n_prefix: number of positions (sinks + cache) that are always valid
    Returns:  [B, S, d_model=NH*DH]
    """
    scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(DH)
    # Causal mask on the chunk portion only
    if S > 1:
        chunk_scores = scores[..., n_prefix:]        # [B, NH, S, S]
        inf_mask     = torch.triu(
            torch.full((S, S), float("-inf"), dtype=scores.dtype, device=scores.device),
            diagonal=1)
        scores = torch.cat([scores[..., :n_prefix],
                            chunk_scores + inf_mask], dim=-1)
    weights = scores.softmax(dim=-1)
    ctx     = torch.matmul(weights, V_full)          # [B, NH, S, DH]
    return ctx.permute(0, 2, 1, 3).reshape(Q.shape[0], S, -1)


def _causal_attn_chunk(fw: dict, x: Tensor, NH: int, freqs: Tensor,
                       K_cache: Tensor, V_cache: Tensor,
                       kv_window: int, dilation: int,
                       n_zero: int, n_train: int,
                       T_past_offset: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """Chunked causal attention with sinks + dilated window.
    Returns (out [B,S,d], K_cache_new, V_cache_new).
    K_cache_new stores ALL non-sink positions (window applied at read time).
    """
    B, S, d = x.shape; DH = d // NH

    Q = torch.einsum("nhi,bsi->bsnh", fw["W_q"].reshape(NH, DH, d), x)
    K = torch.einsum("nhi,bsi->bsnh", fw["W_k"].reshape(NH, DH, d), x)
    V = torch.einsum("nhi,bsi->bsnh", fw["W_v"].reshape(NH, DH, d), x)

    Q = _apply_rope(Q, freqs, offset=T_past_offset)
    K = _apply_rope(K, freqs, offset=T_past_offset)

    # Dilated window on past cache
    K_sel, V_sel = _gather_dilated(K_cache, V_cache, kv_window, dilation)

    # Sinks
    sk, sv = _build_sinks(fw, n_zero, n_train, B, NH, DH, x.dtype, x.device)
    n_sink = n_zero + n_train

    # All parts are [B, *, NH, DH] — concat along dim=1, then permute for attn
    K_parts = sk + ([K_sel] if K_sel.shape[1] > 0 else []) + [K]
    V_parts = sv + ([V_sel] if V_sel.shape[1] > 0 else []) + [V]
    K_t = torch.cat(K_parts, dim=1).permute(0, 2, 1, 3)  # [B, NH, total, DH]
    V_t = torch.cat(V_parts, dim=1).permute(0, 2, 1, 3)

    n_prefix = n_sink + K_sel.shape[1]
    ctx = _attend(Q.permute(0, 2, 1, 3), K_t, V_t, n_prefix, S, DH)
    out = ctx @ fw["W_o"].T

    # Update cache: append chunk, then truncate if kv_window set
    K_new_cache = torch.cat([K_cache, K], dim=1)
    V_new_cache = torch.cat([V_cache, V], dim=1)
    if kv_window > 0:
        keep = kv_window * max(dilation, 1)
        K_new_cache = K_new_cache[:, -keep:]
        V_new_cache = V_new_cache[:, -keep:]

    return out, K_new_cache, V_new_cache


def _attn_step(fw: dict, x_t: Tensor, NH: int, freqs: Tensor,
               K_cache: Tensor, V_cache: Tensor, t: int,
               kv_window: int, dilation: int,
               n_zero: int, n_train: int) -> tuple[Tensor, Tensor, Tensor]:
    """Single-step attention with KV cache + sinks + dilated window."""
    B, d = x_t.shape; DH = d // NH

    q = torch.einsum("nhi,bi->bnh", fw["W_q"].reshape(NH, DH, d), x_t)
    k = torch.einsum("nhi,bi->bnh", fw["W_k"].reshape(NH, DH, d), x_t)
    v = torch.einsum("nhi,bi->bnh", fw["W_v"].reshape(NH, DH, d), x_t)

    q = _apply_rope_1(q, freqs, t)
    k = _apply_rope_1(k, freqs, t)

    # Append new k/v to cache
    K_new = torch.cat([K_cache, k.unsqueeze(1)], dim=1)   # [B, T+1, NH, DH]
    V_new = torch.cat([V_cache, v.unsqueeze(1)], dim=1)

    # Dilated window on full updated cache
    K_sel, V_sel = _gather_dilated(K_new, V_new, kv_window, dilation)

    # Sinks
    sk, sv = _build_sinks(fw, n_zero, n_train, B, NH, DH, x_t.dtype, x_t.device)

    K_parts = sk + ([K_sel] if K_sel.shape[1] > 0 else [])
    V_parts = sv + ([V_sel] if V_sel.shape[1] > 0 else [])
    K_t = torch.cat(K_parts, dim=1).permute(0, 2, 1, 3) if K_parts \
          else torch.zeros(B, NH, 0, DH, dtype=x_t.dtype, device=x_t.device)
    V_t = torch.cat(V_parts, dim=1).permute(0, 2, 1, 3) if V_parts \
          else torch.zeros(B, NH, 0, DH, dtype=x_t.dtype, device=x_t.device)

    # q: [B, NH, 1, DH]
    scores  = torch.matmul(q.unsqueeze(2), K_t.transpose(-2, -1)) / math.sqrt(DH)
    weights = scores.softmax(dim=-1)
    ctx     = torch.matmul(weights, V_t).squeeze(2).reshape(B, d)
    out     = ctx @ fw["W_o"].T

    # Truncate cache if needed
    if kv_window > 0:
        keep = kv_window * max(dilation, 1)
        K_new = K_new[:, -keep:]
        V_new = V_new[:, -keep:]

    return out, K_new, V_new


def _swiglu(x: Tensor, fw: dict) -> Tensor:
    return (F.silu(x @ fw["W_gate"].T) * (x @ fw["W_up"].T)) @ fw["W_down"].T


# ── Transformer model ─────────────────────────────────────────────────────────

class CausalTransformer(RandCompressModel):

    def __init__(self, model_cfg, train_cfg):
        self.mcfg     = model_cfg
        self.tcfg     = train_cfg
        self.dtype    = torch.float32 if train_cfg.dtype == "float32" else torch.bfloat16
        self.ffn_mult = 4

    def _freqs(self, device) -> Tensor:
        DH = self.mcfg.d_model // self.mcfg.num_heads
        return _rope_freqs(DH, self.mcfg.rope_scale).to(device=device, dtype=self.dtype)

    def _attn_kwargs(self) -> dict:
        cfg = self.mcfg
        return dict(kv_window=cfg.kv_window, dilation=cfg.attn_dilation,
                    n_zero=cfg.n_sinks_zero, n_train=cfg.n_sinks_train)

    def init_frozen(self, seed: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(seed)
        cfg = self.mcfg; tcfg = self.tcfg
        d = cfg.d_model; NH = cfg.num_heads; DH = d // NH; dt = self.dtype
        out: dict[str, Tensor] = {}

        emb = rng.standard_normal((tcfg.vocab_size, d)).astype(np.float32) * 0.01
        out["embedding"] = torch.tensor(emb, dtype=dt)

        for i in range(cfg.num_layers):
            attn = _init_attn_frozen(np.random.default_rng(rng.integers(2**31)), d, NH, dt)
            ffn  = _init_ffn_frozen(np.random.default_rng(rng.integers(2**31)), d, self.ffn_mult, dt)
            for k, v in attn.items(): out[f"block{i}.attn.{k}"] = v
            for k, v in ffn.items():  out[f"block{i}.ffn.{k}"]  = v

            # Trainable sinks live in frozen (structured random init)
            if cfg.n_sinks_train > 0:
                out[f"block{i}.sink_k"] = torch.zeros(cfg.n_sinks_train, NH, DH, dtype=dt)
                out[f"block{i}.sink_v"] = torch.zeros(cfg.n_sinks_train, NH, DH, dtype=dt)

        out["final_norm_w"] = torch.ones(d, dtype=dt)
        out["final_norm_b"] = torch.zeros(d, dtype=dt)
        return out

    def init_adapters(self, seed: int) -> dict[str, Tensor]:
        rng = np.random.default_rng(seed + 1)
        cfg = self.mcfg; tcfg = self.tcfg
        d = cfg.d_model; r = cfg.lora_r; dt = self.dtype
        d_ff = d * self.ffn_mult
        oh = tcfg.output_heads; ov = 2 ** tcfg.output_bits
        out: dict[str, Tensor] = {}

        out.update(make_adapter_params("emb", tcfg.vocab_size, d, r, dt))
        for i in range(cfg.num_layers):
            for k in ("W_q", "W_k", "W_v", "W_o"):
                out.update(make_adapter_params(f"block{i}.attn.{k}", d, d, r, dt))
            for k in ("W_gate", "W_up"):
                out.update(make_adapter_params(f"block{i}.ffn.{k}", d_ff, d, r, dt))
            out.update(make_adapter_params(f"block{i}.ffn.W_down", d, d_ff, r, dt))

        op = rng.standard_normal((d, oh * ov)).astype(np.float32) * 0.01
        out["output_proj"] = torch.tensor(op, dtype=dt).requires_grad_(True)
        return out

    def _eff_block(self, frozen, adapters, i) -> dict[str, Tensor]:
        cfg = self.mcfg; d = cfg.d_model; NH = cfg.num_heads; DH = d // NH
        fw: dict[str, Tensor] = {}
        for k in ("W_q", "W_k", "W_v"):
            W0   = frozen[f"block{i}.attn.{k}"]
            flat = apply_adapter(W0.reshape(NH * DH, d),
                                 adapters[f"block{i}.attn.{k}.A"],
                                 adapters[f"block{i}.attn.{k}.B"], cfg.use_hira)
            fw[k] = flat.reshape(W0.shape)
        fw["W_o"]   = apply_adapter(frozen[f"block{i}.attn.W_o"],
                                    adapters[f"block{i}.attn.W_o.A"],
                                    adapters[f"block{i}.attn.W_o.B"], cfg.use_hira)
        fw["norm_w"] = frozen[f"block{i}.attn.norm_w"]
        fw["norm_b"] = frozen[f"block{i}.attn.norm_b"]
        for k in ("W_gate", "W_up", "W_down"):
            fw[f"ffn_{k}"] = apply_adapter(frozen[f"block{i}.ffn.{k}"],
                                           adapters[f"block{i}.ffn.{k}.A"],
                                           adapters[f"block{i}.ffn.{k}.B"], cfg.use_hira)
        fw["ffn_norm_w"] = frozen[f"block{i}.ffn.norm_w"]
        fw["ffn_norm_b"] = frozen[f"block{i}.ffn.norm_b"]
        # Trainable sinks (stored in frozen, direct — not HiRA adapted for simplicity)
        if cfg.n_sinks_train > 0:
            fw["sink_k"] = frozen[f"block{i}.sink_k"]
            fw["sink_v"] = frozen[f"block{i}.sink_v"]
        return fw

    def forward(self, frozen, adapters, tokens: Tensor,
                states: Any) -> tuple[Tensor, Any]:
        cfg  = self.mcfg; tcfg = self.tcfg
        oh   = tcfg.output_heads; ov = 2 ** tcfg.output_bits
        NH   = cfg.num_heads; ak = self._attn_kwargs()
        freqs = self._freqs(tokens.device)

        emb = apply_adapter(frozen["embedding"], adapters["emb.A"],
                            adapters["emb.B"], cfg.use_hira)
        x   = F.embedding(tokens, emb)
        B, S, d = x.shape

        new_states = []
        T_past = 0 if states[0][0].shape[1] == 0 else states[0][0].shape[1]

        for i in range(cfg.num_layers):
            fw = self._eff_block(frozen, adapters, i)
            K_c, V_c = states[i]

            x_n = _layer_norm(x, fw["norm_w"], fw["norm_b"])
            attn_out, K_new, V_new = _causal_attn_chunk(
                fw, x_n, NH, freqs, K_c, V_c,
                T_past_offset=T_past, **ak)
            x = x + attn_out

            ffn_fw = {k[4:]: fw[k] for k in fw if k.startswith("ffn_")}
            x_n = _layer_norm(x, ffn_fw["norm_w"], ffn_fw["norm_b"])
            x   = x + _swiglu(x_n, ffn_fw)

            new_states.append((K_new, V_new))

        x      = _layer_norm(x, frozen["final_norm_w"], frozen["final_norm_b"])
        logits = (x @ adapters["output_proj"]).reshape(B, S, oh, ov)
        return logits, new_states

    def step(self, frozen, adapters, token: Tensor,
             state: Any, t: int) -> tuple[Tensor, Any]:
        cfg  = self.mcfg; tcfg = self.tcfg
        oh   = tcfg.output_heads; ov = 2 ** tcfg.output_bits
        NH   = cfg.num_heads; ak = self._attn_kwargs()
        freqs = self._freqs(token.device)

        emb = apply_adapter(frozen["embedding"], adapters["emb.A"],
                            adapters["emb.B"], cfg.use_hira)
        x   = F.embedding(token, emb)
        B   = x.shape[0]

        new_state = []
        for i in range(cfg.num_layers):
            fw = self._eff_block(frozen, adapters, i)
            K_c, V_c = state[i]

            x_n = _layer_norm(x, fw["norm_w"], fw["norm_b"])
            attn_out, K_new, V_new = _attn_step(fw, x_n, NH, freqs, K_c, V_c, t, **ak)
            x   = x + attn_out

            ffn_fw = {k[4:]: fw[k] for k in fw if k.startswith("ffn_")}
            x_n = _layer_norm(x, ffn_fw["norm_w"], ffn_fw["norm_b"])
            x   = x + _swiglu(x_n, ffn_fw)

            new_state.append((K_new, V_new))

        x      = _layer_norm(x, frozen["final_norm_w"], frozen["final_norm_b"])
        logits = (x @ adapters["output_proj"]).reshape(B, oh, ov)
        return logits, new_state

    def init_states(self, batch_size: int, device: torch.device) -> list:
        cfg = self.mcfg; DH = cfg.d_model // cfg.num_heads; dt = self.dtype
        empty = torch.zeros(batch_size, 0, cfg.num_heads, DH, dtype=dt, device=device)
        return [(empty.clone(), empty.clone()) for _ in range(cfg.num_layers)]

    def precompute_eff_weights(self, frozen, adapters) -> dict:
        cfg = self.mcfg; d = cfg.d_model; d_ff = d * self.ffn_mult
        DH = d // cfg.num_heads; NH = cfg.num_heads
        ew: dict[str, Tensor] = {}
        ew["embedding"] = apply_adapter(frozen["embedding"], adapters["emb.A"],
                                        adapters["emb.B"], cfg.use_hira)
        for i in range(cfg.num_layers):
            for k in ("W_q", "W_k", "W_v"):
                W0   = frozen[f"block{i}.attn.{k}"]
                flat = apply_adapter(W0.reshape(NH * DH, d),
                                     adapters[f"block{i}.attn.{k}.A"],
                                     adapters[f"block{i}.attn.{k}.B"], cfg.use_hira)
                ew[f"block{i}.attn.{k}"] = flat.reshape(W0.shape)
            ew[f"block{i}.attn.W_o"] = apply_adapter(
                frozen[f"block{i}.attn.W_o"],
                adapters[f"block{i}.attn.W_o.A"],
                adapters[f"block{i}.attn.W_o.B"], cfg.use_hira)
            for k_name in ("norm_w", "norm_b"):
                ew[f"block{i}.attn.{k_name}"] = frozen[f"block{i}.attn.{k_name}"]
            for k in ("W_gate", "W_up", "W_down"):
                ew[f"block{i}.ffn.{k}"] = apply_adapter(
                    frozen[f"block{i}.ffn.{k}"],
                    adapters[f"block{i}.ffn.{k}.A"],
                    adapters[f"block{i}.ffn.{k}.B"], cfg.use_hira)
            for k_name in ("norm_w", "norm_b"):
                ew[f"block{i}.ffn.{k_name}"] = frozen[f"block{i}.ffn.{k_name}"]
            if cfg.n_sinks_train > 0:
                ew[f"block{i}.sink_k"] = frozen[f"block{i}.sink_k"]
                ew[f"block{i}.sink_v"] = frozen[f"block{i}.sink_v"]
        ew["final_norm_w"] = frozen["final_norm_w"]
        ew["final_norm_b"] = frozen["final_norm_b"]
        ew["output_proj"]  = adapters["output_proj"]
        return ew

    def step_eff(self, ew: dict, token: Tensor,
                 state: Any, t: int) -> tuple[Tensor, Any]:
        cfg  = self.mcfg; tcfg = self.tcfg
        oh   = tcfg.output_heads; ov = 2 ** tcfg.output_bits
        NH   = cfg.num_heads; ak = self._attn_kwargs()
        freqs = self._freqs(token.device)

        x = ew["embedding"][token]; B = x.shape[0]
        new_state = []

        for i in range(cfg.num_layers):
            fw = {k.split(f"block{i}.")[1]: ew[k]
                  for k in ew if k.startswith(f"block{i}.")}
            # Flatten attn/ffn prefixes for helper functions
            fw_a = {k.split("attn.")[1]: fw[k] for k in fw if k.startswith("attn.")}
            fw_a.update({k: fw[k] for k in fw if k.startswith("sink_")})
            fw_f = {k.split("ffn.")[1]: fw[k] for k in fw if k.startswith("ffn.")}

            K_c, V_c = state[i]
            x_n = _layer_norm(x, fw_a["norm_w"], fw_a["norm_b"])
            attn_out, K_new, V_new = _attn_step(fw_a, x_n, NH, freqs, K_c, V_c, t, **ak)
            x   = x + attn_out

            x_n = _layer_norm(x, fw_f["norm_w"], fw_f["norm_b"])
            x   = x + _swiglu(x_n, fw_f)
            new_state.append((K_new, V_new))

        x      = _layer_norm(x, ew["final_norm_w"], ew["final_norm_b"])
        logits = (x @ ew["output_proj"]).reshape(B, oh, ov)
        return logits, new_state
