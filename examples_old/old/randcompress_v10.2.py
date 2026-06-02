"""
randcompress v10.2 — Fully linear cells + growing-window curriculum + exact BPTT

Changes from v10.1:
  - Replace grad_chunk_persistent (truncated BPTT with manual grad accumulation)
    with exact_bptt_step: jax.lax.scan over all chunks in one jax.value_and_grad
    call, with @jax.checkpoint on the scan body (remat) to keep memory bounded.
  - Gradients now flow through all chunk boundaries — true BPTT, not truncated.
  - Training loop becomes a single call per iteration; no Python-level chunk loop.
  - ce_loss returned as scan aux so BPC display is still based on pure CE.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import NamedTuple, Optional
import math
import sys
import time
import os
import functools
from tqdm import tqdm

PAD_TOKEN = 0
DTYPE     = jnp.float32
POWER_P   = 2

class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, s):
        for f in self.files: f.write(s)
    def flush(self):
        for f in self.files: f.flush()


# ============================================================================
# Config
# ============================================================================

class Config(NamedTuple):
    input_bits:    int   = 8
    output_bits:   int   = 8
    output_heads:  int   = 1
    vocab_size:    int   = 256
    # ── architecture ─────────────────────────────────────────────────────────
    d_model:       int   = 128
    segment_size:  int   = 256
    # segment_size:  int   = 128
    num_heads:     int   = 8
    # segment_size:  int   = 4096
    power_p:       int   = 2      # mLSTM attention kernel degree; 1 = v4.4.1 baseline
    # power_p:       int   = 2      # mLSTM attention kernel degree; 1 = v4.4.1 baseline
    seed:          int   = 0
    # stride_map:    str   = "1111"  # one digit per layer; stride-N fires every N tokens
    # block_map:     str   = "mmmm"
    # num_layers:    int   = 8      # always overridden to len(block_map)
    # block_map:     str   = "s"*8
    # stride_map:    str   = "1,4,16,64,256,1024,4096,16384"  # one digit per layer; stride-N fires every N tokens
    # block_map:     str   = "m"*8
    num_layers:    int   = 4      # always overridden to len(block_map)
    stride_map:    str   = "1,8,64,512"  # one digit per layer; stride-N fires every N tokens
    block_map:     str   = "m"*4
    # stride_map:    str   = "1,2,4,8,16,64,256,1024"  # one digit per layer; stride-N fires every N tokens
    # stride_map:    str   = "1,2,4,8,16,32,64"  # one digit per layer; stride-N fires every N tokens
    # stride_map:    str   = "1,1,8,8,32,32,8,8,1,1"  # one digit per layer; stride-N fires every N tokens
    # stride_map:    str   = "1,4,8,16,32,32,16,8,4,1"  # one digit per layer; stride-N fires every N tokens
    # stride_map:    str   = "1,16,32,64,128,128,64,32,16,1"  # one digit per layer; stride-N fires every N tokens
    # stride_map:    str   = "1,64,128,256,512,512,256,128,64,1"  # one digit per layer; stride-N fires every N tokens
    # stride_map:    str   = "11224488"  # one digit per layer; stride-N fires every N tokens
    # num_layers:    int   = 16      # always overridden to len(block_map)
    # block_map:     str   = "s"*16
    # stride_map:    str   = "1,4,64,64,128,128,256,256,512,1024,1024,512,256,256,128,128,64,64,4,1"  # one digit per layer; stride-N fires every N tokens
    lora_r:        int   = 32
    batch_size:    int   = 1
    # ── optimiser ────────────────────────────────────────────────────────────
    # learning_rate: float = 1e-2    # SinkGD: step = √(mn)·lr; output_proj is 64×256 → √=128, so lr≈1/128
    learning_rate: float = 1e-3    # SinkGD: step = √(mn)·lr; output_proj is 64×256 → √=128, so lr≈1/128
    # learning_rate: float = 1.0    # SinkGD: step = √(mn)·lr; output_proj is 64×256 → √=128, so lr≈1/128
    weight_decay:  float = 0.0
    grad_clip_norm:float = 1e6
    sinkgd_l:      int   = 4
    # ── loss ─────────────────────────────────────────────────────────────────
    margin:        float = 0.
    ce_weight:     float = 1.0
    # margin:        float = 1.0
    # ce_weight:     float = 0.1
    # ── TBPTT state drop ─────────────────────────────────────────────────────
    state_reset_prob: float = 1.
    # ── residual / stop ──────────────────────────────────────────────────────
    residual_budget: float = 0.
    target_bpc:    float = 0.0
    # ── generative eval ──────────────────────────────────────────────────────
    gen_seed_len:  int   = 16   # teacher-forced warm-up bytes before eval starts
    # ── misc ─────────────────────────────────────────────────────────────────
    pad_token:     int   = 0
    dtype:         str   = "float32"
    # dtype:         str   = "float16"
    max_iters:     int   = 100000   # kept for compat; curriculum uses max_iter_per_phase
    max_iter_per_phase: int = 100000 # SOLO / COMBINED budget per segment
    check_every:   int   = 1
    # check_every:   int   = 100
    # dataset:       str   = "datasets/surat_al-fatihah.txt"
    dataset:       str   = "datasets/juz1.txt"
    # dataset:       str   = "datasets/quran-uthmani.txt"

_PRESETS = {
    "byte":         dict(input_bits=8, output_bits=8, output_heads=1, vocab_size=256, pad_token=0),
    "mtp_byte4":    dict(input_bits=8, output_bits=8, output_heads=4, vocab_size=256, pad_token=0),
    "mtp_byte8":    dict(input_bits=8, output_bits=8, output_heads=8, vocab_size=256, pad_token=0),
    "nibble":       dict(input_bits=4, output_bits=4, output_heads=1, vocab_size=16,  pad_token=-1),
    "binary":       dict(input_bits=1, output_bits=1, output_heads=1, vocab_size=2,   pad_token=-1),
    "byte2bits":    dict(input_bits=8, output_bits=1, output_heads=8, vocab_size=256, pad_token=-1),
    "byte2nibbles": dict(input_bits=8, output_bits=4, output_heads=2, vocab_size=256, pad_token=-1),
    "mtp_binary":   dict(input_bits=1, output_bits=1, output_heads=8, vocab_size=2,   pad_token=-1),
}

def _parse_stride_map(s: str, n_layers: int) -> tuple:
    """Parse stride_map string to tuple of ints, padding with 1s if short.
    Accepts comma-separated values (e.g. '1,16,32,1') or single-digit chars ('1248').
    """
    if ',' in s:
        strides = [int(x.strip()) for x in s.split(',')]
    else:
        strides = [int(c) for c in s]
    if len(strides) < n_layers:
        strides += [1] * (n_layers - len(strides))
    return tuple(strides[:n_layers])

# ── tokenisation helpers ───────────────────────────────────────────────────

def bytes_to_tokens(raw: np.ndarray, bits: int) -> np.ndarray:
    if bits == 8: return raw.astype(np.int32)
    if bits == 4:
        hi = ((raw >> 4) & 0xF).astype(np.int32)
        lo = (raw & 0xF).astype(np.int32)
        return np.stack([hi, lo], axis=-1).ravel()
    return np.unpackbits(raw).astype(np.int32)

def tokens_to_bytes(toks: np.ndarray, bits: int) -> np.ndarray:
    if bits == 8: return toks.astype(np.uint8)
    if bits == 4:
        hi = (toks[0::2].astype(np.uint8) & 0xF) << 4
        lo = toks[1::2].astype(np.uint8) & 0xF
        return hi | lo
    return np.packbits(toks.astype(np.uint8))

def compute_targets_np(tokens: np.ndarray, config) -> np.ndarray:
    output_mask = (1 << config.output_bits) - 1
    N = len(tokens) - 1
    if config.input_bits == config.output_bits and config.output_heads == 1:
        return tokens[1:N+1].reshape(N, 1)
    if config.input_bits == 8 and config.output_bits < 8:
        next_bytes = tokens[1:N+1].astype(np.int32)
        heads = np.stack(
            [(next_bytes >> (h * config.output_bits)) & output_mask
             for h in range(config.output_heads)],
            axis=-1)
        return heads
    N_mtp = len(tokens) - config.output_heads
    heads = np.stack(
        [tokens[1+h : 1+h+N_mtp] for h in range(config.output_heads)],
        axis=-1)
    return heads

_WARNINGS = []

def validate_config(config, n_dataset_bytes: int):
    warns = []
    ib, ob, oh = config.input_bits, config.output_bits, config.output_heads
    input_vocab  = 2 ** ib
    output_vocab = 2 ** ob
    tokens_per_byte = 8 // ib
    bits_per_step   = ob * oh
    if input_vocab != config.vocab_size:
        warns.append(f"vocab_size={config.vocab_size} ≠ 2**input_bits={input_vocab}")
    if ib not in (1, 4, 8):
        warns.append(f"input_bits={ib} not in {{1,4,8}}")
    if ob not in (1, 4, 8):
        warns.append(f"output_bits={ob} not in {{1,4,8}}")
    if bits_per_step < ib:
        warns.append(f"output_bits*output_heads={bits_per_step} < input_bits={ib}")
    if ib < 8 and config.pad_token == 0:
        warns.append(f"input_bits={ib} sub-byte but pad_token=0 — set pad_token=-1")
    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    max_stride = max(stride_map)
    chunk_size = config.segment_size * (8 // ib)
    if max_stride > 1 and chunk_size % max_stride != 0:
        warns.append(
            f"chunk_size={chunk_size} not divisible by max_stride={max_stride} "
            f"— upsample will be slightly off at chunk boundaries")
    for w in warns:
        print(f"  [CONFIG WARN] {w}")
    return warns


# ============================================================================
# Utilities
# ============================================================================

def logsigmoid(x):
    return -jnp.logaddexp(0.0, -x)

def layer_norm(x, w, b, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * w + b

def multihead_layer_norm(x, w, b, num_heads, eps=1e-6):
    *lead, D = x.shape
    x_r  = x.reshape(*lead, num_heads, D // num_heads)
    mean = x_r.mean(axis=-1, keepdims=True)
    var  = x_r.var(axis=-1, keepdims=True)
    return ((x_r - mean) / jnp.sqrt(var + eps)).reshape(*lead, D) * w + b


# ============================================================================
# Parameter structures
# ============================================================================

class mLSTMParams(NamedTuple):
    W_q:    jnp.ndarray   # [NH, DH, d_model]  query (orthonormal)
    W_k:    jnp.ndarray   # [NH, DH, d_model]  key
    W_v:    jnp.ndarray   # [NH, DH, d_model]  value
    alpha:  jnp.ndarray   # [NH]                per-head decay, frozen, log-uniform
    skip:   jnp.ndarray   # [d_model]           additive skip scale
    ln_w:   jnp.ndarray   # [d_model]           output multihead LayerNorm
    ln_b:   jnp.ndarray
    W_down: jnp.ndarray   # [d_model, d_model]  output projection

class sRNNParams(NamedTuple):
    """Whitened vanilla RNN: h_t = LayerNorm(W_hh h_{t-1} + W_hx x_t + b)."""
    W_hx:   jnp.ndarray   # [H, d_model]  input→hidden (orthonormal)
    W_hh:   jnp.ndarray   # [H, H]        hidden→hidden (near-critical, σ=0.95)
    b:      jnp.ndarray   # [H]           bias (zeros)
    ln_w:   jnp.ndarray   # [H]           LayerNorm weight (ones)
    ln_b:   jnp.ndarray   # [H]           LayerNorm bias (zeros)

class xLSTMBlockParams(NamedTuple):
    norm_w: jnp.ndarray
    norm_b: jnp.ndarray
    mlstm:  Optional[mLSTMParams] = None
    srnn:   Optional[sRNNParams]  = None

class xLSTMParams(NamedTuple):
    embedding: jnp.ndarray
    blocks:    list
    norm_w:    jnp.ndarray
    norm_b:    jnp.ndarray


# ============================================================================
# HiRA structures
# ============================================================================

class HiRAAdapter(NamedTuple):
    A: jnp.ndarray
    B: jnp.ndarray

class mLSTMHiRA(NamedTuple):
    W_q:    HiRAAdapter
    W_k:    HiRAAdapter
    W_v:    HiRAAdapter
    W_down: HiRAAdapter

class sRNNHiRA(NamedTuple):
    W_hx: HiRAAdapter
    W_hh: HiRAAdapter

class BlockHiRA(NamedTuple):
    mlstm: Optional[mLSTMHiRA] = None
    srnn:  Optional[sRNNHiRA]  = None

class RandCompressParams(NamedTuple):
    output_proj: jnp.ndarray
    emb_hira:    HiRAAdapter
    blocks:      list


# ============================================================================
# HiRA application  ΔW = W₀ ⊙ (B·A)
# ============================================================================

def apply_hira(base, hira):
    return base + base * jnp.dot(hira.B, hira.A)

def apply_mlstm_hira(p: mLSTMParams, l: mLSTMHiRA) -> mLSTMParams:
    NH, DH, d = p.W_q.shape
    dm = NH * DH
    def qkv(w, a):
        return apply_hira(w.reshape(dm, d), a).reshape(NH, DH, d)
    return mLSTMParams(
        W_q=qkv(p.W_q, l.W_q), W_k=qkv(p.W_k, l.W_k), W_v=qkv(p.W_v, l.W_v),
        alpha=p.alpha, skip=p.skip, ln_w=p.ln_w, ln_b=p.ln_b,
        W_down=apply_hira(p.W_down, l.W_down),
    )

def apply_srnn_hira(p: sRNNParams, l: sRNNHiRA) -> sRNNParams:
    return sRNNParams(
        W_hx=apply_hira(p.W_hx, l.W_hx),
        W_hh=apply_hira(p.W_hh, l.W_hh),
        b=p.b, ln_w=p.ln_w, ln_b=p.ln_b,
    )

def apply_block_hira(base: xLSTMBlockParams, hira: BlockHiRA) -> xLSTMBlockParams:
    return xLSTMBlockParams(
        norm_w=base.norm_w, norm_b=base.norm_b,
        mlstm=apply_mlstm_hira(base.mlstm, hira.mlstm) if base.mlstm is not None else None,
        srnn =apply_srnn_hira(base.srnn,   hira.srnn)  if base.srnn  is not None else None,
    )

def apply_xlstm_hira(base: xLSTMParams, rc: RandCompressParams) -> xLSTMParams:
    return xLSTMParams(
        embedding=apply_hira(base.embedding, rc.emb_hira),
        blocks=[apply_block_hira(b, l) for b, l in zip(base.blocks, rc.blocks)],
        norm_w=base.norm_w, norm_b=base.norm_b,
    )


# ============================================================================
# Power attention helpers
# ============================================================================

def _dsym(DH, p):
    """Output dimension of the symmetric degree-p feature map of R^DH."""
    if p == 1: return DH
    return math.comb(DH + p - 1, p)

def spow(x, p):
    """Symmetric degree-p polynomial feature map.

    p=1: identity.  p=2: all products x_i*x_j (i<=j), off-diagonal * sqrt(2).
    [..., D] → [..., D_sym] where D_sym = C(D+p-1, p).
    """
    if p == 1:
        return x
    D = x.shape[-1]
    i_idx, j_idx = np.tril_indices(D)
    scale = np.where(i_idx == j_idx, 1.0, np.sqrt(2.0)).astype(np.float32)
    return x[..., i_idx] * x[..., j_idx] * scale


# ============================================================================
# Linear mLSTM layers  — scan-based, no gates, no exp/sigmoid/tanh
# C_t = alpha[:,None,None]*C_{t-1} + phi_k⊗v   (phi = symmetric power map)
# h   = phi_q @ C,  then additive skip + multihead LayerNorm + W_down
# State: C only [B, NH, D_sym, DH].
# ============================================================================

def mlstm_layer_scan_with_init(p: mLSTMParams, x: jnp.ndarray, num_heads: int, state):
    C0 = state
    B, S, d = x.shape
    DH = d // num_heads

    q = jnp.einsum('ndi,bsi->bnsd', p.W_q, x)   # [B, NH, S, DH]
    k = jnp.einsum('ndi,bsi->bnsd', p.W_k, x)
    v = jnp.einsum('ndi,bsi->bnsd', p.W_v, x)

    def step(C, t_in):
        q_t, k_t, v_t = t_in
        k_s   = k_t / jnp.sqrt(jnp.array(DH, k_t.dtype))
        phi_k = spow(k_s, POWER_P)
        phi_q = spow(q_t, POWER_P)
        kv    = jnp.einsum('bnd,bne->bnde', phi_k, v_t)
        C_new = p.alpha[:, None, None] * C + kv
        h_num = jnp.einsum('bnd,bnde->bne', phi_q, C_new)   # [B, NH, DH]
        return C_new, h_num

    C_T, hs = jax.lax.scan(
        step, C0,
        (q.transpose(2,0,1,3), k.transpose(2,0,1,3), v.transpose(2,0,1,3))
    )
    h_flat = hs.transpose(1,2,0,3).reshape(B, S, d)          # [B, S, d_model]
    h_out  = h_flat + p.skip * x
    h_norm = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T), C_T

def mlstm_layer_scan(p: mLSTMParams, x: jnp.ndarray, num_heads: int):
    B, S, d = x.shape
    DH    = d // num_heads
    D_sym = _dsym(DH, POWER_P)
    C0    = jnp.zeros((B, num_heads, D_sym, DH), x.dtype)
    out, _ = mlstm_layer_scan_with_init(p, x, num_heads, C0)
    return out

def mlstm_layer_step(p: mLSTMParams, x_t: jnp.ndarray, state, num_heads: int):
    C = state
    B, d = x_t.shape
    DH   = d // num_heads
    q = jnp.einsum('ndi,bi->bnd', p.W_q, x_t)
    k = jnp.einsum('ndi,bi->bnd', p.W_k, x_t)
    v = jnp.einsum('ndi,bi->bnd', p.W_v, x_t)
    k_s   = k / jnp.sqrt(jnp.array(DH, k.dtype))
    phi_k = spow(k_s, POWER_P)
    phi_q = spow(q,   POWER_P)
    kv    = jnp.einsum('bnd,bne->bnde', phi_k, v)
    C_t   = p.alpha[:, None, None] * C + kv
    h_num = jnp.einsum('bnd,bnde->bne', phi_q, C_t)
    h_flat = h_num.reshape(B, d)
    h_out  = h_flat + p.skip * x_t
    h_norm = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T), C_t


# ============================================================================
# Whitened sRNN layers  — h_t = LayerNorm(W_hh h_{t-1} + W_hx x_t + b)
# Only nonlinearity: LayerNorm. Enforces E[h]=0, Var[h_i]=1 at every step.
# State: h only [B, d_model].
# ============================================================================

def srnn_layer_scan_with_init(p: sRNNParams, x: jnp.ndarray, num_heads: int, state):
    h0 = state
    def step(h, x_t):
        pre  = jnp.dot(x_t, p.W_hx.T) + jnp.dot(h, p.W_hh.T) + p.b
        h_new = layer_norm(pre, p.ln_w, p.ln_b)
        return h_new, h_new
    h_T, ys = jax.lax.scan(step, h0, x.transpose(1, 0, 2))
    return ys.transpose(1, 0, 2), h_T   # [B, S, H], h_T

def srnn_layer_scan(p: sRNNParams, x: jnp.ndarray, num_heads: int):
    B = x.shape[0]
    h0 = jnp.zeros((B, p.W_hx.shape[0]), x.dtype)
    out, _ = srnn_layer_scan_with_init(p, x, num_heads, h0)
    return out

def srnn_layer_step(p: sRNNParams, x_t: jnp.ndarray, state, num_heads: int):
    h = state
    pre = jnp.dot(x_t, p.W_hx.T) + jnp.dot(h, p.W_hh.T) + p.b
    h_t = layer_norm(pre, p.ln_w, p.ln_b)
    return h_t, h_t   # (residual output, new state)



# ============================================================================
# Forward passes
# ============================================================================

def forward_train(base_xlstm, rc, tokens, num_heads, block_map, stride_map):
    xlstm = apply_xlstm_hira(base_xlstm, rc)
    x     = xlstm.embedding[tokens]
    S     = x.shape[1]
    for block, btype, stride in zip(xlstm.blocks, block_map, stride_map):
        x_n = layer_norm(x, block.norm_w, block.norm_b)
        if stride > 1:
            x_sub = x_n[:, ::stride, :]
            cell_out_sub = (mlstm_layer_scan(block.mlstm, x_sub, num_heads)
                            if btype == 'm' else srnn_layer_scan(block.srnn, x_sub, num_heads))
            cell_out = jnp.repeat(cell_out_sub, stride, axis=1)[:, :S, :]
        else:
            cell_out = (mlstm_layer_scan(block.mlstm, x_n, num_heads)
                        if btype == 'm' else srnn_layer_scan(block.srnn, x_n, num_heads))
        x = x + cell_out
    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj)

def forward_train_chunked(base_xlstm, rc, inputs, layer_states, num_heads, block_map, stride_map):
    """
    layer_states: list of (lstm_state, last_out) per layer.
    Returns (logits, new_layer_states).
    """
    xlstm      = apply_xlstm_hira(base_xlstm, rc)
    x          = xlstm.embedding[inputs]
    S          = x.shape[1]
    new_states = []
    for block, btype, stride, state in zip(xlstm.blocks, block_map, stride_map, layer_states):
        lstm_state, _last_out = state
        x_n = layer_norm(x, block.norm_w, block.norm_b)
        if stride > 1:
            x_sub = x_n[:, ::stride, :]
            if btype == 'm':
                cell_out_sub, new_lstm = mlstm_layer_scan_with_init(
                    block.mlstm, x_sub, num_heads, lstm_state)
            else:
                cell_out_sub, new_lstm = srnn_layer_scan_with_init(
                    block.srnn, x_sub, num_heads, lstm_state)
            cell_out = jnp.repeat(cell_out_sub, stride, axis=1)[:, :S, :]
        else:
            if btype == 'm':
                cell_out, new_lstm = mlstm_layer_scan_with_init(
                    block.mlstm, x_n, num_heads, lstm_state)
            else:
                cell_out, new_lstm = srnn_layer_scan_with_init(
                    block.srnn, x_n, num_heads, lstm_state)
        last_out = cell_out[:, -1, :]  # carry to next chunk for non-firing positions
        new_states.append((new_lstm, last_out))
        x = x + cell_out
    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj), new_states

def forward_step(base_xlstm, rc, token, states, num_heads, block_map, stride_map, t):
    """
    states: list of (lstm_state, last_out) per layer.
    t: int32 JAX scalar — global step index for stride-N fire decisions.
    """
    xlstm     = apply_xlstm_hira(base_xlstm, rc)
    x         = xlstm.embedding[token]
    new_states= []
    for block, btype, stride, state in zip(xlstm.blocks, block_map, stride_map, states):
        lstm_state, last_out = state
        x_n = layer_norm(x, block.norm_w, block.norm_b)

        if stride == 1:
            if btype == 'm':
                cell_out, new_lstm = mlstm_layer_step(block.mlstm, x_n, lstm_state, num_heads)
            else:
                cell_out, new_lstm = srnn_layer_step(block.srnn, x_n, lstm_state, num_heads)
            new_states.append((new_lstm, cell_out))
        else:
            # fire if at a stride boundary; otherwise hold last output
            def fire(args):
                x_n_, lstm_state_ = args
                if btype == 'm':
                    return mlstm_layer_step(block.mlstm, x_n_, lstm_state_, num_heads)
                else:
                    return srnn_layer_step(block.srnn, x_n_, lstm_state_, num_heads)

            def hold(args):
                _x_n, lstm_state_ = args
                return last_out, lstm_state_

            cell_out, new_lstm = jax.lax.cond(
                t % stride == 0, fire, hold, (x_n, lstm_state))
            new_states.append((new_lstm, cell_out))

        x = x + cell_out

    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj), new_states

_fwd_step_jit    = jax.jit(forward_step,          static_argnames=["num_heads","block_map","stride_map"])
_fwd_chunked_jit = jax.jit(forward_train_chunked,  static_argnames=["num_heads","block_map","stride_map"])

def _reshape_logits(flat, output_heads, output_bits):
    output_vocab = 2 ** output_bits
    return flat.reshape(*flat.shape[:-1], output_heads, output_vocab)


# ============================================================================
# Exact BPTT: scan over all chunks, gradients flow through all boundaries.
# @jax.checkpoint on scan_body rematerialises activations during backward pass
# so peak memory is O(chunk) rather than O(n_chunks * chunk).
# ============================================================================

@functools.partial(jax.jit, static_argnames=["config"])
def exact_bptt_step(base_xlstm, params, all_inputs, all_targets, config):
    """
    all_inputs:  [num_chunks, chunk_size]      int32
    all_targets: [num_chunks, chunk_size, oh]  int32
    Returns (loss, ce_loss, grads) — all scalars averaged over chunks.
    """
    oh         = config.output_heads
    ob         = config.output_bits
    num_heads  = config.num_heads
    block_map  = config.block_map
    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    margin     = config.margin
    ce_weight  = config.ce_weight

    def loss_fn(p):
        init_states = init_step_states(config, batch_size=1)

        @jax.checkpoint
        def scan_body(states, xs):
            inputs, tgt = xs                        # [chunk_size], [chunk_size, oh]
            logits_flat, new_states = forward_train_chunked(
                base_xlstm, p, inputs[None, :], states, num_heads, block_map, stride_map)
            logits = _reshape_logits(logits_flat, oh, ob)   # [1, S, oh, ov]
            tgt_b  = tgt[None, :]                           # [1, S, oh]
            chunk_loss = sum(
                training_loss(logits[..., h, :], tgt_b[..., h], margin, ce_weight)
                for h in range(oh)
            ) / oh
            chunk_ce = sum(
                cross_entropy_loss(logits[..., h, :], tgt_b[..., h])
                for h in range(oh)
            ) / oh
            return new_states, (chunk_loss, chunk_ce)

        _, (per_chunk_loss, per_chunk_ce) = jax.lax.scan(
            scan_body, init_states, (all_inputs, all_targets))
        return per_chunk_loss.mean(), per_chunk_ce.mean()

    (loss, ce_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, ce_loss, grads


# ============================================================================
# State initialisation  (no conv_buf; includes last_out)
# ============================================================================

def init_step_states(config, batch_size):
    """Returns list of (lstm_state, last_out) per layer."""
    B   = batch_size
    NH  = config.num_heads
    DH  = config.d_model // config.num_heads
    bf  = DTYPE
    states = []
    for btype in config.block_map:
        if btype == 'm':
            D_sym = _dsym(DH, config.power_p)
            lstm_state = jnp.zeros((B, NH, D_sym, DH), bf)   # C only
        else:
            lstm_state = jnp.zeros((B, config.d_model), bf)  # h only
        last_out = jnp.zeros((B, config.d_model), bf)
        states.append((lstm_state, last_out))
    return states


# ============================================================================
# Structured init helpers
# ============================================================================

def _bf(x): return jnp.array(x, dtype=DTYPE)

def _ortho(key, n, m):
    raw = np.array(jr.normal(key, (max(n,m), min(n,m))))
    Q, _ = np.linalg.qr(raw)
    mat  = Q.T
    if n > m:
        extra = np.array(jr.normal(jr.fold_in(key,1),(n-m,m))) / math.sqrt(m)
        mat   = np.concatenate([mat, extra], axis=0)
    return _bf(mat[:n])

def _multiscale_forget_bias(n, seq_len):
    tau = np.exp(np.linspace(0.0, np.log(max(float(seq_len), 2.0)), n))
    f   = np.exp(-1.0 / tau)
    return _bf(np.log(f / (1.0 - f + 1e-9)))


# ============================================================================
# Frozen xLSTM initialisation  (no conv_w)
# ============================================================================

def init_mlstm_params(key, d_model, num_heads, seq_len) -> mLSTMParams:
    d  = d_model
    DH = d // num_heads
    ks = jr.split(key, 5)
    def _qkv(k):
        M = np.array(_ortho(k, d, d))
        return _bf(M.reshape(num_heads, DH, d) / math.sqrt(d))
    W_q = _qkv(ks[0]); W_k = _qkv(ks[1]); W_v = _qkv(ks[2])
    # per-head decay: α_h = exp(-1/τ_h), τ_h log-uniform in [1, seq_len]
    tau   = np.exp(np.linspace(0., np.log(max(float(seq_len), 2.)), num_heads))
    alpha = _bf(np.exp(-1. / tau))
    W_down = _ortho(ks[3], d, d) / math.sqrt(d)
    return mLSTMParams(
        W_q=W_q, W_k=W_k, W_v=W_v, alpha=alpha,
        skip=jnp.ones((d,), DTYPE),
        ln_w=jnp.ones((d,), DTYPE), ln_b=jnp.zeros((d,), DTYPE),
        W_down=W_down,
    )

def init_srnn_params(key, d_model, seq_len) -> sRNNParams:
    H  = d_model
    ks = jr.split(key, 3)
    W_hx = _ortho(ks[0], H, H) / math.sqrt(H)
    raw_hh = np.array(jr.normal(ks[1], (H, H)))
    U, _, Vt = np.linalg.svd(raw_hh, full_matrices=False)
    W_hh = _bf(0.95 * (U @ Vt))              # near-critical: implicit forgetting
    return sRNNParams(
        W_hx=W_hx, W_hh=W_hh,
        b=jnp.zeros((H,), DTYPE),
        ln_w=jnp.ones((H,), DTYPE), ln_b=jnp.zeros((H,), DTYPE),
    )

def init_xlstm_block_params(key, d_model, num_heads,
                             seq_len, btype) -> xLSTMBlockParams:
    ks = jr.split(key, 2)
    return xLSTMBlockParams(
        norm_w=jnp.ones((d_model,), DTYPE),
        norm_b=jnp.zeros((d_model,), DTYPE),
        mlstm=init_mlstm_params(ks[0], d_model, num_heads, seq_len) if btype=='m' else None,
        srnn =init_srnn_params(ks[1], d_model, seq_len)              if btype=='s' else None,
    )

def init_xlstm_params(key, config) -> xLSTMParams:
    ks  = jr.split(key, 2 + config.num_layers)
    emb = _bf(np.array(jr.normal(ks[0], (config.vocab_size, config.d_model))) * 0.01)
    blocks = [
        init_xlstm_block_params(
            ks[2+i], config.d_model, config.num_heads,
            config.segment_size, config.block_map[i]
        )
        for i in range(config.num_layers)
    ]
    return xLSTMParams(
        embedding=emb, blocks=blocks,
        norm_w=jnp.ones((config.d_model,), DTYPE),
        norm_b=jnp.zeros((config.d_model,), DTYPE),
    )


# ============================================================================
# HiRA initialisation  (no ffn adapters)
# ============================================================================

def init_hira_adapter(key, d_out, d_in, r) -> HiRAAdapter:
    raw = np.array(jr.normal(key, (r, d_in)))
    if r <= d_in:
        _, _, Vt = np.linalg.svd(raw, full_matrices=False)
        A = _bf(Vt / math.sqrt(d_in))
    else:
        A = _bf(raw / math.sqrt(d_in))
    return HiRAAdapter(A=A, B=jnp.zeros((d_out, r), DTYPE))

def init_block_hira(key, config, btype) -> BlockHiRA:
    d, r = config.d_model, config.lora_r
    ks = jr.split(key, 4)
    mhira = srnnhira = None
    if btype == 'm':
        mhira = mLSTMHiRA(
            W_q   =init_hira_adapter(ks[0], d, d, r),
            W_k   =init_hira_adapter(ks[1], d, d, r),
            W_v   =init_hira_adapter(ks[2], d, d, r),
            W_down=init_hira_adapter(ks[3], d, d, r),
        )
    else:
        srnnhira = sRNNHiRA(
            W_hx=init_hira_adapter(ks[0], d, d, r),
            W_hh=init_hira_adapter(ks[1], d, d, r),
        )
    return BlockHiRA(mlstm=mhira, srnn=srnnhira)

def init_randcompress_params(key, config) -> RandCompressParams:
    ks = jr.split(key, 2 + config.num_layers)
    out_dim = config.output_heads * (2 ** config.output_bits)
    return RandCompressParams(
        output_proj=_bf(np.array(jr.normal(ks[0], (config.d_model, out_dim))) * 0.01),
        emb_hira   =init_hira_adapter(ks[1], config.vocab_size, config.d_model, config.lora_r),
        blocks     =[init_block_hira(ks[2+i], config, config.block_map[i])
                     for i in range(config.num_layers)],
    )


# ============================================================================
# SinkGD
# ============================================================================

def _sr_sinkhorn_2d(W, L):
    m, n = W.shape
    for _ in range(L):
        W = jnp.sqrt(n) * W / (jnp.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        W = jnp.sqrt(m) * W / (jnp.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    return W

def _sinkgd_normalize(g, L):
    g32 = g.astype(jnp.float32)
    if g32.ndim == 0: return g32
    if g32.ndim == 1:
        n = g32.shape[0]
        return jnp.sqrt(n) * g32 / (jnp.linalg.norm(g32) + 1e-8)
    m = g32.shape[0]
    n = int(np.prod(g32.shape[1:]))
    return _sr_sinkhorn_2d(g32.reshape(m, n), L).reshape(g32.shape)

def sinkgd_init(params):
    return None

@functools.partial(jax.jit, static_argnums=(5,))
def _sinkgd_step(grads, params, lr, weight_decay, max_norm, L):
    leaves = jax.tree_util.tree_leaves(grads)
    gnorm  = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32)**2) for g in leaves))
    scale  = jnp.minimum(1.0, max_norm / (gnorm + 1e-6))
    grads  = jax.tree_util.tree_map(lambda g: g * scale, grads)
    norm_g = jax.tree_util.tree_map(lambda g: _sinkgd_normalize(g, L), grads)
    new_p  = jax.tree_util.tree_map(
        lambda p, g: (p.astype(jnp.float32)
                      - lr * g
                      - lr * weight_decay * p.astype(jnp.float32)).astype(p.dtype),
        params, norm_g,
    )
    return new_p

def _center_hira_b(params):
    """Zero-center columns of every HiRA B matrix.
    Maintains E[Bv]=0 for zero-mean inputs v, keeping the whiteness condition
    (zero-mean activations after act_ln) valid throughout training.
    """
    def fn(x):
        if isinstance(x, HiRAAdapter):
            return HiRAAdapter(A=x.A, B=x.B - x.B.mean(axis=0, keepdims=True))
        return x
    return jax.tree_util.tree_map(fn, params,
                                  is_leaf=lambda x: isinstance(x, HiRAAdapter))

def sinkgd_update(state, grads, params, lr, weight_decay=0.0, max_norm=1.0, L=1):
    new_params = _sinkgd_step(grads, params,
                               jnp.array(lr), jnp.array(weight_decay), jnp.array(max_norm), L)
    new_params = _center_hira_b(new_params)
    return new_params, state


# ============================================================================
# Loss
# ============================================================================

def cross_entropy_loss(logits, targets):
    log_probs   = jax.nn.log_softmax(logits, axis=-1)
    loss_per_tok= -jnp.take_along_axis(log_probs, targets[...,None], axis=-1)[...,0]
    mask        = (targets != PAD_TOKEN).astype(jnp.float32)
    return (loss_per_tok*mask).sum() / (mask.sum()+1e-8)

def argmax_margin_loss(logits, targets, margin=1.0):
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    neg_inf       = jax.nn.one_hot(targets, logits.shape[-1]) * 1e9
    best_wrong    = (logits - neg_inf).max(axis=-1)
    loss_per_tok  = jnp.maximum(0.0, best_wrong - target_logits + margin)
    mask          = (targets != PAD_TOKEN).astype(jnp.float32)
    return (loss_per_tok * mask).sum() / (mask.sum() + 1e-8)

def training_loss(logits, targets, margin=1.0, ce_weight=0.05):
    if margin == 0.0:
        return cross_entropy_loss(logits, targets)
    ce = cross_entropy_loss(logits, targets)
    mg = argmax_margin_loss(logits, targets, margin)
    return mg + ce_weight * ce


# ============================================================================
# Data helpers
# ============================================================================

class ByteTokenizer:
    vocab_size = 256
    @staticmethod
    def encode(text):
        if isinstance(text, str): text = text.encode('utf-8')
        return np.frombuffer(text, dtype=np.uint8)
    @staticmethod
    def decode(tokens):
        return bytes(np.array(tokens, dtype=np.uint8)).decode('utf-8', errors='replace')

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return ByteTokenizer.encode(f.read())

def make_chunks_and_targets(raw_bytes, config):
    toks       = bytes_to_tokens(np.asarray(raw_bytes, dtype=np.uint8), config.input_bits)
    chunk_size = config.segment_size * (8 // config.input_bits)
    n          = len(toks)
    num_chunks = max(1, math.ceil((n - 1) / chunk_size))
    oh         = config.output_heads
    ob         = config.output_bits
    out_mask   = (1 << ob) - 1
    pad_len = num_chunks * chunk_size + oh - n
    pad_len = max(pad_len, 0)
    padded  = np.concatenate([toks, np.zeros(pad_len + oh, dtype=np.int32)])
    all_inputs  = np.stack([padded[i*chunk_size : i*chunk_size + chunk_size]
                             for i in range(num_chunks)]).astype(np.int32)
    tgt_chunks = []
    for i in range(num_chunks):
        chunk_tgts = np.zeros((chunk_size, oh), dtype=np.int32)
        if config.input_bits == 8 and ob < 8:
            next_bytes = padded[i*chunk_size + 1 : i*chunk_size + chunk_size + 1]
            for h in range(oh):
                chunk_tgts[:, h] = (next_bytes >> (h * ob)) & out_mask
        else:
            for h in range(oh):
                chunk_tgts[:, h] = padded[i*chunk_size + 1 + h : i*chunk_size + chunk_size + 1 + h]
        tgt_chunks.append(chunk_tgts)
    all_targets = np.stack(tgt_chunks)
    return jnp.array(all_inputs, dtype=jnp.int32), jnp.array(all_targets, dtype=jnp.int32)


# ============================================================================
# Generative evaluation
# ============================================================================

def eval_generative(base_xlstm, params, config, all_tokens):
    """
    Autoregressive decode with optional warm-up prefix and residual collection.

    gen_seed_len (from config): number of bytes fed teacher-forced before evaluation
      begins. This warms the hidden state so predictions aren't made from a cold
      zero state. Warm-up bytes are filled with correct values in the returned
      `generated` array (always correct, never counted as errors).
    """
    stride_map   = _parse_stride_map(config.stride_map, config.num_layers)
    all_bytes    = np.array(all_tokens, dtype=np.uint8)
    target_bytes = all_bytes[1:]
    n            = len(target_bytes)
    budget_bytes = int(config.residual_budget * n)
    bytes_per_entry = 4
    max_entries  = budget_bytes // bytes_per_entry if budget_bytes > 0 else 0

    ob, oh       = config.output_bits, config.output_heads
    out_mask     = (1 << ob) - 1
    tok_per_byte = 8 // config.input_bits

    seed_bytes = min(config.gen_seed_len, n)   # clamp to available data
    seed_toks  = seed_bytes * tok_per_byte      # warm-up steps in token units

    states     = init_step_states(config, batch_size=1)
    residual   = {}
    input_toks = bytes_to_tokens(all_bytes, config.input_bits)

    # ── warm-up: teacher-forced prefix, state only, no predictions recorded ──
    for i in range(seed_toks):
        tok_i = jnp.array([int(input_toks[i])])
        _, states = _fwd_step_jit(
            base_xlstm, params, tok_i, states, config.num_heads,
            config.block_map, stride_map, jnp.array(i, jnp.int32))

    # ── autoregressive generation from seed_toks onwards ─────────────────────
    # first input is the correct token at position seed_toks (known from prefix)
    cur_token = jnp.array([int(input_toks[seed_toks])])

    if tok_per_byte > 1:
        pred_subtoks = []
        n_pred = (n - seed_bytes) * tok_per_byte
        for i in tqdm(range(n_pred), desc="gen", unit="tok", leave=False, file=sys.stderr):
            t = seed_toks + i
            logit_flat, states = _fwd_step_jit(
                base_xlstm, params, cur_token, states, config.num_heads,
                config.block_map, stride_map, jnp.array(t, jnp.int32))
            logits   = _reshape_logits(logit_flat, oh, ob)
            pred_tok = int(jnp.argmax(logits[0, 0]))
            pred_subtoks.append(pred_tok)
            cur_token = jnp.array([pred_tok])
        gen_bytes_tail = list(tokens_to_bytes(np.array(pred_subtoks, np.int32), config.input_bits))
    else:
        gen_bytes_tail = []
        _pbar = tqdm(range(seed_bytes, n), desc="gen", unit="B", unit_scale=True,
                     leave=False, file=sys.stderr)
        for byte_idx in _pbar:
            t = byte_idx
            logit_flat, states = _fwd_step_jit(
                base_xlstm, params, cur_token, states, config.num_heads,
                config.block_map, stride_map, jnp.array(t, jnp.int32))
            logits = _reshape_logits(logit_flat, oh, ob)

            if config.input_bits == 8 and ob < 8:
                pred_byte = 0
                for h in range(oh):
                    pred_byte |= (int(jnp.argmax(logits[0, h])) & out_mask) << (h * ob)
                pred_byte &= 0xFF
            else:
                pred_tok  = int(jnp.argmax(logits[0, 0]))
                pred_byte = int(tokens_to_bytes(np.array([pred_tok], np.int32), config.input_bits)[0])

            correct_byte = int(target_bytes[byte_idx])

            if pred_byte != correct_byte and len(residual) < max_entries:
                residual[byte_idx] = correct_byte
                correct_tok = int(bytes_to_tokens(np.array([correct_byte], np.uint8), config.input_bits)[0])
                cur_token   = jnp.array([correct_tok])
            else:
                pred_tok_ = int(bytes_to_tokens(np.array([pred_byte], np.uint8), config.input_bits)[0])
                cur_token = jnp.array([pred_tok_])

            gen_bytes_tail.append(pred_byte)
            _pbar.set_postfix(wrong=len(residual))
        _pbar.close()

    # prepend correct seed bytes so `generated` is always length n
    gen_bytes_full = list(target_bytes[:seed_bytes]) + gen_bytes_tail
    generated   = np.array(gen_bytes_full, dtype=np.uint8)
    wrong_mask  = generated[seed_bytes:] != target_bytes[seed_bytes:]
    bytes_wrong = int(np.sum(wrong_mask))
    accuracy    = ((n - seed_bytes) - bytes_wrong) / max(n - seed_bytes, 1)
    wrong_idxs  = np.where(generated != target_bytes)[0]
    first_wrong = int(wrong_idxs[0]) if len(wrong_idxs) > 0 else None
    return generated, bytes_wrong, accuracy, first_wrong, residual


# ============================================================================
# Curriculum helpers
# ============================================================================

def split_segments(raw_bytes, segment_size):
    segs, n, i = [], len(raw_bytes), 0
    while i < n:
        segs.append(raw_bytes[i : i + segment_size])
        i += segment_size
    return segs

def eval_segments_stateful(base_xlstm, params, config, seg_list):
    """Teacher-forced byte accuracy, state flowing across segments (zero-init start).
    Returns list of (accuracy, first_wrong_byte_idx) per segment.
    """
    stride_map   = _parse_stride_map(config.stride_map, config.num_layers)
    tok_per_byte = 8 // config.input_bits
    chunk_size   = config.segment_size * tok_per_byte
    ob, oh       = config.output_bits, config.output_heads
    out_mask     = (1 << ob) - 1
    states       = init_step_states(config, batch_size=1)
    results      = []

    for seg_bytes in seg_list:
        seg   = bytes_to_tokens(np.array(seg_bytes, dtype=np.uint8), config.input_bits)
        n_tok = len(seg)
        n_b   = len(seg_bytes)
        if n_b < 2:
            results.append((1.0, None)); continue

        num_chunks = max(1, math.ceil((n_tok - 1) / chunk_size))
        pad_len    = num_chunks * chunk_size + 1 - n_tok
        padded     = np.concatenate([seg, np.zeros(pad_len, dtype=np.int32)])

        byte_correct = []
        for ci in range(num_chunks):
            s   = ci * chunk_size
            inp = jnp.array(padded[s : s + chunk_size][None], dtype=jnp.int32)
            logits_flat, states = _fwd_chunked_jit(
                base_xlstm, params, inp, states,
                config.num_heads, config.block_map, stride_map)
            logits = _reshape_logits(logits_flat, oh, ob)          # [1,S,oh,ov]
            preds  = np.array(jnp.argmax(logits[0], axis=-1))      # [S, oh]

            if config.input_bits == 8 and ob < 8:
                pred_bytes = np.zeros(preds.shape[0], np.uint8)
                for h in range(oh):
                    pred_bytes |= (preds[:, h].astype(np.uint8) & out_mask) << (h * ob)
                tgt_bytes = padded[s + 1 : s + chunk_size + 1].astype(np.uint8)
            else:
                pred_bytes = tokens_to_bytes(preds[:, 0], config.input_bits)
                tgt_bytes  = tokens_to_bytes(padded[s + 1 : s + chunk_size + 1], config.input_bits)

            valid = min(len(pred_bytes), max(0, n_b - 1 - ci * (chunk_size // tok_per_byte)))
            if valid > 0:
                byte_correct.extend((pred_bytes[:valid] == tgt_bytes[:valid]).tolist())

        total = len(byte_correct)
        acc   = sum(byte_correct) / total if total > 0 else 1.0
        wrong = [i for i, ok in enumerate(byte_correct) if not ok]
        results.append((acc, wrong[0] if wrong else None))

    return results


# ============================================================================
# Checkpoint
# ============================================================================

def save_checkpoint(ckpt_dir, params, config, seg_idx, phase):
    import json, pickle
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {**config._asdict(), "seg_idx": seg_idx, "phase": phase}
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(ckpt_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)

def _save_residual(ckpt_dir, residual: dict):
    import pickle
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "residual.pkl"), "wb") as f:
        pickle.dump(residual, f)

def cosine_lr(step, total_steps, lr_max, lr_min=1e-4, warmup=100):
    warmup_lr = lr_max * jnp.minimum(step/warmup, 1.0)
    t         = jnp.maximum(step-warmup,0) / jnp.maximum(total_steps-warmup,1)
    cos_lr    = lr_min + 0.5*(lr_max-lr_min)*(1.0+jnp.cos(math.pi*t))
    return jnp.where(step < warmup, warmup_lr, cos_lr)


# ============================================================================
# Main
# ============================================================================

def _parse_config():
    import json as _json
    _types = {f: type(v) for f, v in Config()._asdict().items()}
    def _cast(field, s):
        t = _types[field]
        if t is bool:
            return s.lower() in ("1", "true", "yes")
        return t(s)

    cfg  = Config()._asdict()
    argv = sys.argv[1:]

    if "--preset" in argv:
        idx  = argv.index("--preset")
        name = argv[idx + 1]
        if name not in _PRESETS:
            print(f"  [CONFIG WARN] unknown preset '{name}'; known: {list(_PRESETS)}")
        else:
            cfg.update(_PRESETS[name])

    if "--config" in argv:
        idx = argv.index("--config")
        with open(argv[idx + 1]) as f:
            for k, v in _json.load(f).items():
                if k in cfg:
                    cfg[k] = v

    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and argv[i] not in ("--config", "--preset"):
            field = argv[i][2:]
            if field in cfg and i + 1 < len(argv):
                cfg[field] = _cast(field, argv[i + 1])
                i += 2
                continue
        i += 1

    cfg['num_layers'] = len(cfg['block_map'])
    cfg['vocab_size'] = 2 ** cfg['input_bits']

    # auto-extend stride_map to match block_map length (supports comma-separated values)
    parsed = list(_parse_stride_map(cfg['stride_map'], cfg['num_layers']))
    cfg['stride_map'] = ','.join(str(s) for s in parsed)

    config = Config(**cfg)
    lib_defaults = Config()._asdict()
    diff = {k: {"default": lib_defaults[k], "value": cfg[k]}
            for k in cfg if cfg[k] != lib_defaults[k]}
    return config, diff


def main():
    import json as _json
    _script = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join("log", _script)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    config, config_diff = _parse_config()
    global PAD_TOKEN, DTYPE, POWER_P
    PAD_TOKEN = config.pad_token
    DTYPE     = jnp.dtype(config.dtype)
    POWER_P   = config.power_p

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        _json.dump(config._asdict(), f, indent=2)
    with open(os.path.join(log_dir, "config_diff.json"), "w") as f:
        _json.dump(config_diff, f, indent=2)

    if config_diff:
        print("Config overrides (vs defaults):")
        for k, v in config_diff.items():
            print(f"  {k}: {v['default']} → {v['value']}")
    else:
        print("Config: all defaults")

    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    print(f"stride_map: {stride_map}")

    raw_bytes = load_dataset(config.dataset)
    n_total   = len(raw_bytes)

    print(f"\nConfig validation:")
    warns = validate_config(config, n_total)
    if not warns:
        print("  OK")

    tokens       = bytes_to_tokens(raw_bytes, config.input_bits)
    tok_per_byte = 8 // config.input_bits
    print(f"Tokenisation: input_bits={config.input_bits} output_bits={config.output_bits} "
          f"output_heads={config.output_heads}  "
          f"seq={len(tokens)} tokens ({tok_per_byte}× bytes)")
    print(f"Dataset: {config.dataset}  ({n_total} bytes)")

    key = jr.key(config.seed)
    key, k_xlstm, k_rc = jr.split(key, 3)

    print("Initialising frozen xLSTM...")
    base_xlstm = init_xlstm_params(k_xlstm, config)
    frozen_n   = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(base_xlstm))

    print("Initialising HiRA adapters...")
    params    = init_randcompress_params(k_rc, config)
    trainable = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    total          = frozen_n + trainable
    dtype_bytes    = np.dtype(DTYPE).itemsize
    param_bytes      = trainable * dtype_bytes
    residual_bytes   = int(config.residual_budget * n_total)
    total_coded_bytes = param_bytes + residual_bytes
    compress_ratio   = n_total / param_bytes
    compress_ratio_r = n_total / total_coded_bytes
    print(f"{'metric':<28}  {'value':>12}")
    print(f"{'-'*28}  {'-'*12}")
    print(f"{'frozen params':<28}  {frozen_n/1e6:>11.3f}M")
    print(f"{'trainable params':<28}  {trainable/1e6:>11.3f}M")
    print(f"{'total params':<28}  {total/1e6:>11.3f}M")
    print(f"{'trainable / total':<28}  {trainable/total*100:>11.1f}%")
    print(f"{'---':<28}  {'---':>12}")
    print(f"{'dtype':<28}  {jnp.dtype(DTYPE).name:>12}")
    print(f"{'trainable param bytes':<28}  {param_bytes/1e6:>11.3f}M")
    print(f"{'residual budget bytes':<28}  {residual_bytes/1e6:>11.3f}M  ({config.residual_budget*100:.1f}% of data)")
    print(f"{'total coded bytes':<28}  {total_coded_bytes/1e6:>11.3f}M  (params + residual)")
    print(f"{'dataset bytes':<28}  {n_total/1e6:>11.3f}M")
    print(f"{'compression ratio (params)':<28}  {compress_ratio:>11.2f}x")
    print(f"{'compression ratio (+ residual)':<28}  {compress_ratio_r:>11.2f}x")
    print(f"{'compressed?':<28}  {'yes' if compress_ratio_r > 1 else 'no (over-param)':>12}")

    opt_state = sinkgd_init(params)
    lr        = config.learning_rate

    bytes_per_entry = 4
    budget_bytes    = int(config.residual_budget * (n_total - 1))
    max_entries     = budget_bytes // bytes_per_entry

    segs   = split_segments(raw_bytes, config.segment_size)
    n_segs = len(segs)
    print(f"Growing-window curriculum: {n_segs} segments × up to {config.segment_size} bytes")
    print(f"max_iter_per_phase={config.max_iter_per_phase}  check_every={config.check_every}  lr={lr}")
    print(f"Residual budget: {budget_bytes} bytes  ({max_entries} correction entries)")
    print()

    t_start      = time.perf_counter()
    global_iters = 0
    accumulated  = []   # growing list of byte segments

    for seg_idx, seg in enumerate(segs):
        accumulated.append(seg)
        combined = np.concatenate(accumulated)
        comb_inp, comb_tgt = make_chunks_and_targets(combined, config)
        n_chunks  = comb_inp.shape[0]

        byte_lo = seg_idx * config.segment_size
        print(f"\nSEGMENT {seg_idx+1}/{n_segs}  accumulated={len(combined)} bytes  chunks={n_chunks}")

        pbar = tqdm(range(1, config.max_iter_per_phase + 1),
                    desc=f"seg{seg_idx+1}", unit="it", file=sys.stderr, dynamic_ncols=True)
        for it in pbar:
            loss, ce_loss, grads = exact_bptt_step(
                base_xlstm, params, comb_inp, comb_tgt, config)
            params, opt_state = sinkgd_update(
                opt_state, grads, params, lr=lr,
                weight_decay=config.weight_decay, max_norm=config.grad_clip_norm,
                L=config.sinkgd_l)
            global_iters += 1

            bpc = float(ce_loss) / math.log(2)
            pbar.set_postfix(loss=f"{float(loss):.4f}", bpc=f"{bpc:.3f}")

            if it % config.check_every == 0 or it == config.max_iter_per_phase:
                accs    = eval_segments_stateful(base_xlstm, params, config, accumulated)
                min_acc = min(a for a, _ in accs)
                fw_str  = next((str(fw) for _, fw in accs if fw is not None), "ok")
                elapsed = time.perf_counter() - t_start
                print()
                pbar.set_description(f"seg{seg_idx+1} it={global_iters} acc={min_acc:.2%} fw={fw_str} "
                                     f"bpc={bpc:.4f} {elapsed:.0f}s")
                if all(a == 1.0 for a, _ in accs):
                    pbar.close()
                    break

        pbar.close()
        save_checkpoint(os.path.join(log_dir, "ckpt_last"), params, config, seg_idx, "grow")

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    stop_reason   = f"all {n_segs} segments accumulated  total_iters={global_iters}"
    print(f"\n[STOP] {stop_reason}  elapsed={elapsed_total:.1f}s")

    print("\nFinal generative eval...")
    gen_out, gen_wrong, gen_acc, gen_fw, residual = eval_generative(
        base_xlstm, params, config, raw_bytes)
    gen_fw_str  = f"byte {gen_fw}" if gen_fw is not None else "none (perfect)"
    residual_kb = len(residual) * bytes_per_entry / 1024
    print(f"  gen_wrong={gen_wrong}  gen_acc={gen_acc:.4%}  first_wrong={gen_fw_str}")
    print(f"  corrections={len(residual)} entries  {residual_kb:.2f} KB  "
          f"budget={max_entries} entries ({budget_bytes} bytes)")
    print(f"  reconstructable={'YES' if gen_wrong <= max_entries else 'NO'}")

    gen_path = os.path.join(log_dir, "gen_final.txt")
    full_gen  = np.concatenate([[raw_bytes[0]], gen_out])
    with open(gen_path, "w", encoding="utf-8", errors="replace") as gf:
        gf.write(f"stop_reason:  {stop_reason}\n"
                 f"gen_wrong:    {gen_wrong}\n"
                 f"gen_acc:      {gen_acc:.4%}\n"
                 f"first_wrong:  {gen_fw_str}\n"
                 f"corrections:  {len(residual)} entries  {residual_kb:.2f} KB\n"
                 f"budget:       {max_entries} entries  {budget_bytes} bytes\n"
                 f"\n--- target ---\n{ByteTokenizer.decode(raw_bytes)}\n"
                 f"\n--- generated ---\n{ByteTokenizer.decode(full_gen)}\n")

    ckpt_dir = os.path.join(log_dir, "ckpt_last")
    save_checkpoint(ckpt_dir, params, config, 0, "final")
    _save_residual(ckpt_dir, residual)

    print(f"\nLog: {log_path}")
    log_file.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
