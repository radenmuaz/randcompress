"""
randcompress v1.1 with HiRA — Persistent TBPTT (single file)

Changes from v1:
- HiRA scale factor: W_eff = W₀ + hira_scale * W₀ ⊙ (B · A)
- A init scale: 1/sqrt(r) instead of 1/sqrt(d_in)
- output_proj init: 0.1 std instead of 0.01
- configurable adam beta1, beta2, eps, grad_clip per config
- 5 named configs in main(); swap which is active

HiRA update rule: ΔW = hira_scale * W₀ ⊙ (B · A)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import NamedTuple, Optional
import math
import sys
import time
from tqdm import tqdm

PAD_TOKEN = 0   # null byte — not present in UTF-8 Arabic text
# DTYPE = jnp.bfloat16  # single dtype for params/activations; swap to jnp.float32 if needed
DTYPE = jnp.float32  # single dtype for params/activations; swap to jnp.float32 if needed

class _Tee:
    """Write to stdout and a log file simultaneously."""
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files: f.write(s)
    def flush(self):
        for f in self.files: f.flush()


# ============================================================================
# Config
# ============================================================================

class Config(NamedTuple):
    vocab_size: int = 256
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 1024
    num_layers: int = 4
    max_seq_len: int = 512
    seed: int = 0
    conv_kernel: int = 4
    # one char per layer: 'm'=mLSTM, 's'=sLSTM; len must equal num_layers
    block_map: str = "mmmm"
    lora_r: int = 8          # HiRA rank r
    hira_scale: float = 1.0  # multiplier on W₀ ⊙ (B·A); >1 amplifies adapter effect
    batch_size: int = 16
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    grad_accum_k: int = 1  # chunks per update; 1 = standard TBPTT


# ============================================================================
# Utilities
# ============================================================================

def logsigmoid(x):
    return -jnp.logaddexp(0.0, -x)

def layer_norm(x, w, b, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * w + b

def multihead_layer_norm(x, w, b, num_heads, eps=1e-6):
    # x: [..., NH*DH]; normalize each head independently
    *lead, D = x.shape
    x_r = x.reshape(*lead, num_heads, D // num_heads)
    mean = x_r.mean(axis=-1, keepdims=True)
    var = x_r.var(axis=-1, keepdims=True)
    return ((x_r - mean) / jnp.sqrt(var + eps)).reshape(*lead, D) * w + b

def causal_conv1d(x, w):
    # x: [B, S, C], w: [C, K] — depthwise causal conv
    C, K = w.shape
    w = w.astype(x.dtype)  # match dtype (base weights bf16, LoRA-path is f32)
    x_pad = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0)))   # [B, S+K-1, C]
    x_t = x_pad.transpose(0, 2, 1)                       # [B, C, S+K-1]
    w_t = w[:, None, :]                                   # [C, 1, K]
    out = jax.lax.conv_general_dilated(
        x_t, w_t, window_strides=(1,), padding='VALID',
        feature_group_count=C, dimension_numbers=('NCH', 'OIH', 'NCH'),
    )
    return out.transpose(0, 2, 1)                         # [B, S, C]

def causal_conv1d_step(x_t, conv_buf, w):
    # x_t: [B, C], conv_buf: [B, K-1, C] — single-step conv
    w = w.astype(x_t.dtype)
    window = jnp.concatenate([conv_buf, x_t[:, None, :]], axis=1)   # [B, K, C]
    new_buf = window[:, 1:, :]                                        # [B, K-1, C]
    out = jnp.einsum('bkc,ck->bc', window, w)                        # [B, C]
    return out, new_buf

def gated_ffn(ffn_up, ffn_down, x):
    # ffn_up: [2*d_ff, d_model], ffn_down: [d_model, d_ff]
    h = jnp.dot(x, ffn_up.T)
    d_ff = h.shape[-1] // 2
    return jnp.dot(jax.nn.gelu(h[..., :d_ff]) * h[..., d_ff:], ffn_down.T)


# ============================================================================
# mLSTM cell — parallel (training) and recurrent (inference)
# ============================================================================

def _mlstm_cell_parallel_inner(q, k, v, i_preact, f_preact):
    # q, k, v: [B, NH, S, DH]; i_preact, f_preact: [B, NH, S, 1]
    B, NH, S, DH = q.shape
    log_f = jax.nn.log_sigmoid(f_preact)                               # [B, NH, S, 1]
    lf_cs = jnp.concatenate([
        jnp.zeros((B, NH, 1, 1), log_f.dtype),
        jnp.cumsum(log_f, axis=2),
    ], axis=2)                                                          # [B, NH, S+1, 1]
    rep = lf_cs[..., 0]                                                 # [B, NH, S+1]
    log_fg = rep[:, :, 1:, None] - rep[:, :, None, 1:]                 # [B, NH, S, S]
    mask = jnp.tril(jnp.ones((S, S), bool))
    log_fg = jnp.where(mask, log_fg, -jnp.inf)
    log_D = log_fg + i_preact.transpose(0, 1, 3, 2)                    # [B, NH, S, S]
    max_log_D = log_D.max(axis=-1, keepdims=True)                       # [B, NH, S, 1]
    D = jnp.exp(log_D - max_log_D)
    k_s = k / jnp.sqrt(jnp.array(DH, k.dtype))
    qk = jnp.einsum('bnsd,bntd->bnst', q, k_s)
    C = qk * D
    norm = jnp.maximum(jnp.abs(C.sum(axis=-1, keepdims=True)), jnp.exp(-max_log_D))
    h = jnp.einsum('bnst,bntd->bnsd', C / norm, v)                     # [B, NH, S, DH]
    return h, (k_s, v, log_D, max_log_D)  # extra for state extraction

def mlstm_cell_parallel(q, k, v, i_preact, f_preact):
    h, _ = _mlstm_cell_parallel_inner(q, k, v, i_preact, f_preact)
    return h

def mlstm_cell_parallel_with_state(q, k, v, i_preact, f_preact):
    # Returns h [B,NH,S,DH] and final recurrent state (C_T, n_T, m_T).
    # C_T/n_T are the last-row weighted sums — compatible with mlstm_cell_step.
    h, (k_s, v_, log_D, max_log_D) = _mlstm_cell_parallel_inner(q, k, v, i_preact, f_preact)
    # Last row: weights for position S-1 over all predecessors
    m_T = max_log_D[:, :, -1, 0]                                        # [B, NH]
    w_T = jnp.exp(log_D[:, :, -1, :] - m_T[:, :, None])               # [B, NH, S]
    C_T = jnp.einsum('bns,bnsd,bnse->bnde', w_T, k_s, v_)             # [B, NH, DH, DH]
    n_T = jnp.einsum('bns,bnsd->bnd', w_T, k_s)                        # [B, NH, DH]
    return h, (C_T, n_T, m_T)

def mlstm_cell_step(q, k, v, i_preact, f_preact, C_prev, n_prev, m_prev):
    # q, k, v: [B, NH, DH]; i_preact, f_preact: [B, NH]
    DH = q.shape[-1]
    log_f = jax.nn.log_sigmoid(f_preact)
    m_t = jnp.maximum(log_f + m_prev, i_preact)
    f_t = jnp.exp(log_f + m_prev - m_t)
    i_t = jnp.exp(i_preact - m_t)
    k_s = k / jnp.sqrt(jnp.array(DH, k.dtype))
    kv = jnp.einsum('bnd,bne->bnde', k_s, v)
    C_t = f_t[:, :, None, None] * C_prev + i_t[:, :, None, None] * kv
    n_t = f_t[:, :, None] * n_prev + i_t[:, :, None] * k_s
    h_num = jnp.einsum('bnd,bnde->bne', q, C_t)
    denom = jnp.maximum(jnp.abs((q * n_t).sum(axis=-1)), jnp.exp(-m_t))
    h = h_num / (denom[:, :, None] + 1e-8)
    return h, C_t, n_t, m_t


# ============================================================================
# sLSTM cell
# ============================================================================

def slstm_cell(raw, sc, sn, sm):
    # raw: [B, 4*H] (gate preacts + recurrent already summed)
    i_raw, f_raw, z_raw, o_raw = jnp.split(raw, 4, axis=-1)
    log_f = logsigmoid(f_raw)
    m_t = jnp.maximum(i_raw, log_f + sm)
    i_t = jnp.minimum(1.0, jnp.exp(i_raw - m_t))
    f_t = jnp.minimum(1.0, jnp.exp(log_f + sm - m_t))
    o_t = jax.nn.sigmoid(o_raw)
    c_t = f_t * sc + i_t * jnp.tanh(z_raw)
    n_t = f_t * sn + i_t
    y_t = o_t * c_t / (jnp.abs(n_t) + 1e-8)
    return y_t, c_t, n_t, m_t


# ============================================================================
# Parameter structures
# ============================================================================

class mLSTMParams(NamedTuple):
    W_up: jnp.ndarray    # [2*inner_dim, d_model]
    W_q: jnp.ndarray     # [NH, DH, inner_dim]
    W_k: jnp.ndarray     # [NH, DH, inner_dim]
    W_v: jnp.ndarray     # [NH, DH, inner_dim]
    W_i: jnp.ndarray     # [NH, inner_dim]
    W_f: jnp.ndarray     # [NH, inner_dim]
    skip: jnp.ndarray    # [inner_dim]
    ln_w: jnp.ndarray    # [inner_dim]
    ln_b: jnp.ndarray    # [inner_dim]
    W_down: jnp.ndarray  # [d_model, inner_dim]
    conv_w: jnp.ndarray  # [inner_dim, conv_kernel]

class sLSTMParams(NamedTuple):
    W_i: jnp.ndarray     # [H, d_model]  H = NH*DH = d_model
    W_f: jnp.ndarray
    W_z: jnp.ndarray
    W_o: jnp.ndarray
    Ry: jnp.ndarray      # [4*H, H]
    b: jnp.ndarray       # [4*H]
    gn_w: jnp.ndarray    # [H]
    gn_b: jnp.ndarray
    conv_w: jnp.ndarray  # [d_model, conv_kernel]

class xLSTMBlockParams(NamedTuple):
    norm1_w: jnp.ndarray
    norm1_b: jnp.ndarray
    norm2_w: jnp.ndarray
    norm2_b: jnp.ndarray
    ffn_up: jnp.ndarray    # [2*d_ff, d_model]
    ffn_down: jnp.ndarray  # [d_model, d_ff]
    mlstm: Optional[mLSTMParams] = None
    slstm: Optional[sLSTMParams] = None

class xLSTMParams(NamedTuple):
    embedding: jnp.ndarray  # [vocab_size, d_model]
    blocks: list
    norm_w: jnp.ndarray
    norm_b: jnp.ndarray


# ============================================================================
# HiRA structures
# ============================================================================

class HiRAAdapter(NamedTuple):
    A: jnp.ndarray   # [r, d_in]  random init
    B: jnp.ndarray   # [d_out, r] zero init

class mLSTMHiRA(NamedTuple):
    W_up: HiRAAdapter
    W_q: HiRAAdapter
    W_k: HiRAAdapter
    W_v: HiRAAdapter
    W_i: HiRAAdapter   # gate projections — important for mLSTM memorisation
    W_f: HiRAAdapter
    W_down: HiRAAdapter

class sLSTMHiRA(NamedTuple):
    W_i: HiRAAdapter
    W_f: HiRAAdapter
    W_z: HiRAAdapter
    W_o: HiRAAdapter

class BlockHiRA(NamedTuple):
    ffn_up: HiRAAdapter
    ffn_down: HiRAAdapter
    mlstm: Optional[mLSTMHiRA] = None
    slstm: Optional[sLSTMHiRA] = None

class RandCompressParams(NamedTuple):
    output_proj: jnp.ndarray  # [d_model, vocab_size] — no HiRA on head
    emb_hira: HiRAAdapter
    blocks: list              # list of BlockHiRA


# ============================================================================
# HiRA application  (ΔW = W₀ ⊙ (B · A))
# ============================================================================

def apply_hira(base, hira, scale=1.0):
    return base + scale * base * jnp.dot(hira.B, hira.A)

def apply_mlstm_hira(p: mLSTMParams, l: mLSTMHiRA, scale: float) -> mLSTMParams:
    NH, DH, inner_dim = p.W_q.shape
    dm = NH * DH
    def qkv(w, a):
        return apply_hira(w.reshape(dm, inner_dim), a, scale).reshape(NH, DH, inner_dim)
    return mLSTMParams(
        W_up=apply_hira(p.W_up, l.W_up, scale),
        W_q=qkv(p.W_q, l.W_q), W_k=qkv(p.W_k, l.W_k), W_v=qkv(p.W_v, l.W_v),
        W_i=apply_hira(p.W_i, l.W_i, scale), W_f=apply_hira(p.W_f, l.W_f, scale),
        skip=p.skip, ln_w=p.ln_w, ln_b=p.ln_b,
        W_down=apply_hira(p.W_down, l.W_down, scale),
        conv_w=p.conv_w,
    )

def apply_slstm_hira(p: sLSTMParams, l: sLSTMHiRA, scale: float) -> sLSTMParams:
    return sLSTMParams(
        W_i=apply_hira(p.W_i, l.W_i, scale), W_f=apply_hira(p.W_f, l.W_f, scale),
        W_z=apply_hira(p.W_z, l.W_z, scale), W_o=apply_hira(p.W_o, l.W_o, scale),
        Ry=p.Ry, b=p.b, gn_w=p.gn_w, gn_b=p.gn_b, conv_w=p.conv_w,
    )

def apply_block_hira(base: xLSTMBlockParams, hira: BlockHiRA, scale: float) -> xLSTMBlockParams:
    return xLSTMBlockParams(
        norm1_w=base.norm1_w, norm1_b=base.norm1_b,
        norm2_w=base.norm2_w, norm2_b=base.norm2_b,
        ffn_up=apply_hira(base.ffn_up, hira.ffn_up, scale),
        ffn_down=apply_hira(base.ffn_down, hira.ffn_down, scale),
        mlstm=apply_mlstm_hira(base.mlstm, hira.mlstm, scale) if base.mlstm is not None else None,
        slstm=apply_slstm_hira(base.slstm, hira.slstm, scale) if base.slstm is not None else None,
    )

def apply_xlstm_hira(base: xLSTMParams, rc: RandCompressParams, scale: float = 1.0) -> xLSTMParams:
    return xLSTMParams(
        embedding=apply_hira(base.embedding, rc.emb_hira, scale),
        blocks=[apply_block_hira(b, l, scale) for b, l in zip(base.blocks, rc.blocks)],
        norm_w=base.norm_w, norm_b=base.norm_b,
    )


# ============================================================================
# mLSTM layer — parallel (training) and step (inference)
# ============================================================================

def mlstm_layer_parallel(p: mLSTMParams, x: jnp.ndarray, num_heads: int):
    # x: [B, S, d_model]
    B, S, _ = x.shape
    inner_dim = p.W_up.shape[0] // 2
    xz = jnp.dot(x, p.W_up.T)                                # [B, S, 2*inner_dim]
    x_m, z = xz[..., :inner_dim], xz[..., inner_dim:]

    x_conv = causal_conv1d(x_m, p.conv_w)                    # [B, S, inner_dim]
    x_conv_a = jax.nn.silu(x_conv)

    q = jnp.einsum('ndi,bsi->bnsd', p.W_q, x_conv_a)        # [B, NH, S, DH]
    k = jnp.einsum('ndi,bsi->bnsd', p.W_k, x_conv_a)
    v = jnp.einsum('ndi,bsi->bnsd', p.W_v, x_m)             # v from pre-conv

    i_p = jnp.einsum('ni,bsi->bns', p.W_i, x_conv_a)[..., None]  # [B, NH, S, 1]
    f_p = jnp.einsum('ni,bsi->bns', p.W_f, x_conv_a)[..., None]

    h = mlstm_cell_parallel(q, k, v, i_p, f_p)              # [B, NH, S, DH]
    h_flat = h.transpose(0, 2, 1, 3).reshape(B, S, inner_dim)

    h_out = (h_flat + p.skip * x_conv_a) * jax.nn.silu(z)
    h_norm = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T)                       # [B, S, d_model]

def mlstm_layer_parallel_with_state(p: mLSTMParams, x: jnp.ndarray, num_heads: int):
    # Same as mlstm_layer_parallel but also returns (C_T, n_T, m_T, conv_buf).
    B, S, _ = x.shape
    inner_dim = p.W_up.shape[0] // 2
    xz = jnp.dot(x, p.W_up.T)
    x_m, z = xz[..., :inner_dim], xz[..., inner_dim:]

    x_conv = causal_conv1d(x_m, p.conv_w)
    x_conv_a = jax.nn.silu(x_conv)

    q = jnp.einsum('ndi,bsi->bnsd', p.W_q, x_conv_a)
    k = jnp.einsum('ndi,bsi->bnsd', p.W_k, x_conv_a)
    v = jnp.einsum('ndi,bsi->bnsd', p.W_v, x_m)
    i_p = jnp.einsum('ni,bsi->bns', p.W_i, x_conv_a)[..., None]
    f_p = jnp.einsum('ni,bsi->bns', p.W_f, x_conv_a)[..., None]

    h, (C_T, n_T, m_T) = mlstm_cell_parallel_with_state(q, k, v, i_p, f_p)
    h_flat = h.transpose(0, 2, 1, 3).reshape(B, S, inner_dim)
    h_out = (h_flat + p.skip * x_conv_a) * jax.nn.silu(z)
    h_norm = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    out = jnp.dot(h_norm, p.W_down.T)

    K = p.conv_w.shape[1]
    conv_buf = x_m[:, -(K - 1):, :]                         # [B, K-1, inner_dim]
    return out, (C_T, n_T, m_T, conv_buf)

def mlstm_layer_step(p: mLSTMParams, x_t: jnp.ndarray, state, num_heads: int):
    # x_t: [B, d_model]; state: (C, n, m, conv_buf)
    C, n, m, conv_buf = state
    B = x_t.shape[0]
    inner_dim = p.W_up.shape[0] // 2

    xz = jnp.dot(x_t, p.W_up.T)
    x_m, z = xz[..., :inner_dim], xz[..., inner_dim:]

    x_conv, new_buf = causal_conv1d_step(x_m, conv_buf, p.conv_w)
    x_conv_a = jax.nn.silu(x_conv)

    q = jnp.einsum('ndi,bi->bnd', p.W_q, x_conv_a)
    k = jnp.einsum('ndi,bi->bnd', p.W_k, x_conv_a)
    v = jnp.einsum('ndi,bi->bnd', p.W_v, x_m)
    i_p = jnp.einsum('ni,bi->bn', p.W_i, x_conv_a)
    f_p = jnp.einsum('ni,bi->bn', p.W_f, x_conv_a)

    h, C_t, n_t, m_t = mlstm_cell_step(q, k, v, i_p, f_p, C, n, m)
    h_flat = h.reshape(B, inner_dim)

    h_out = (h_flat + p.skip * x_conv_a) * jax.nn.silu(z)
    h_norm = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T), (C_t, n_t, m_t, new_buf)


# ============================================================================
# sLSTM layer — scan (training) and step (inference)
# ============================================================================

def slstm_layer(p: sLSTMParams, x: jnp.ndarray, num_heads: int):
    # x: [B, S, d_model]
    B = x.shape[0]
    H = p.W_i.shape[0]  # NH*DH = d_model

    x_conv = causal_conv1d(x, p.conv_w)
    x_conv_a = jax.nn.silu(x_conv)

    # i, f from conv output; z, o from original x
    gates = jnp.concatenate([
        jnp.dot(x_conv_a, p.W_i.T),
        jnp.dot(x_conv_a, p.W_f.T),
        jnp.dot(x, p.W_z.T),
        jnp.dot(x, p.W_o.T),
    ], axis=-1)  # [B, S, 4H]

    init = tuple(jnp.zeros((B, H), x.dtype) for _ in range(4))  # (y, c, n, m)

    def step(state, gate_t):
        sy, sc, sn, sm = state
        raw = gate_t + jnp.dot(sy, p.Ry.T) + p.b
        y_t, c_t, n_t, m_t = slstm_cell(raw, sc, sn, sm)
        return (y_t, c_t, n_t, m_t), y_t

    _, ys = jax.lax.scan(step, init, gates.transpose(1, 0, 2))
    ys = ys.transpose(1, 0, 2)  # [B, S, H]
    return multihead_layer_norm(ys, p.gn_w, p.gn_b, num_heads)

def slstm_layer_step(p: sLSTMParams, x_t: jnp.ndarray, state, num_heads: int):
    # x_t: [B, d_model]; state: (y, c, n, m, conv_buf)
    sy, sc, sn, sm, conv_buf = state

    x_conv, new_buf = causal_conv1d_step(x_t, conv_buf, p.conv_w)
    x_conv_a = jax.nn.silu(x_conv)

    gate_t = jnp.concatenate([
        jnp.dot(x_conv_a, p.W_i.T),
        jnp.dot(x_conv_a, p.W_f.T),
        jnp.dot(x_t, p.W_z.T),
        jnp.dot(x_t, p.W_o.T),
    ], axis=-1)
    raw = gate_t + jnp.dot(sy, p.Ry.T) + p.b
    y_t, c_t, n_t, m_t = slstm_cell(raw, sc, sn, sm)
    out = multihead_layer_norm(y_t, p.gn_w, p.gn_b, num_heads)
    return out, (y_t, c_t, n_t, m_t, new_buf)


# ============================================================================
# Forward — training (parallel mLSTM) and inference (recurrent step)
# ============================================================================

def forward_train(base_xlstm: xLSTMParams, rc: RandCompressParams,
                  tokens: jnp.ndarray, num_heads: int, block_map: str, hira_scale: float = 1.0):
    xlstm = apply_xlstm_hira(base_xlstm, rc, hira_scale)
    x = xlstm.embedding[tokens]   # [B, S, d_model]

    for block, btype in zip(xlstm.blocks, block_map):
        x_n = layer_norm(x, block.norm1_w, block.norm1_b)
        if btype == 'm':
            x = x + mlstm_layer_parallel(block.mlstm, x_n, num_heads)
        else:
            x = x + slstm_layer(block.slstm, x_n, num_heads)
        x_n = layer_norm(x, block.norm2_w, block.norm2_b)
        x = x + gated_ffn(block.ffn_up, block.ffn_down, x_n)

    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj)   # [B, S, vocab_size]


def forward_train_with_state(base_xlstm: xLSTMParams, rc: RandCompressParams,
                              tokens: jnp.ndarray, num_heads: int, block_map: str,
                              hira_scale: float = 1.0):
    """Like forward_train but also returns final per-layer states for recurrent continuation."""
    xlstm = apply_xlstm_hira(base_xlstm, rc, hira_scale)
    x = xlstm.embedding[tokens]

    states = []
    for block, btype in zip(xlstm.blocks, block_map):
        x_n = layer_norm(x, block.norm1_w, block.norm1_b)
        if btype == 'm':
            cell_out, state = mlstm_layer_parallel_with_state(block.mlstm, x_n, num_heads)
        else:
            # sLSTM: run scan, return final state tuple + empty conv_buf placeholder
            cell_out = slstm_layer(block.slstm, x_n, num_heads)
            # Extract final sLSTM state by running a separate scan that carries state
            state = _slstm_final_state(block.slstm, x_n)
        states.append(state)
        x = x + cell_out
        x_n = layer_norm(x, block.norm2_w, block.norm2_b)
        x = x + gated_ffn(block.ffn_up, block.ffn_down, x_n)

    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj), states


def _slstm_final_state(p: sLSTMParams, x: jnp.ndarray):
    """Run the sLSTM scan and return only the final state (for recurrent continuation)."""
    B = x.shape[0]
    H = p.W_i.shape[0]
    K = p.conv_w.shape[1]

    x_conv = causal_conv1d(x, p.conv_w)
    x_conv_a = jax.nn.silu(x_conv)
    gates = jnp.concatenate([
        jnp.dot(x_conv_a, p.W_i.T), jnp.dot(x_conv_a, p.W_f.T),
        jnp.dot(x, p.W_z.T), jnp.dot(x, p.W_o.T),
    ], axis=-1)

    init = tuple(jnp.zeros((B, H), x.dtype) for _ in range(4))

    def step(state, gate_t):
        sy, sc, sn, sm = state
        raw = gate_t + jnp.dot(sy, p.Ry.T) + p.b
        y_t, c_t, n_t, m_t = slstm_cell(raw, sc, sn, sm)
        return (y_t, c_t, n_t, m_t), None

    (y_T, c_T, n_T, m_T), _ = jax.lax.scan(step, init, gates.transpose(1, 0, 2))
    conv_buf = x[:, -(K - 1):, :]  # [B, K-1, d_model]
    return (y_T, c_T, n_T, m_T, conv_buf)


def init_step_states(config: Config, batch_size: int):
    B, K = batch_size, config.conv_kernel
    NH, DH = config.num_heads, config.d_model // config.num_heads
    bf = DTYPE
    states = []
    for btype in config.block_map:
        if btype == 'm':
            inner = config.d_model
            states.append((
                jnp.zeros((B, NH, DH, DH), bf),    # C
                jnp.zeros((B, NH, DH), bf),          # n
                jnp.zeros((B, NH), bf),              # m
                jnp.zeros((B, K - 1, inner), bf),   # conv_buf
            ))
        else:
            H = config.d_model
            states.append((
                jnp.zeros((B, H), bf),               # y
                jnp.zeros((B, H), bf),               # c
                jnp.zeros((B, H), bf),               # n
                jnp.zeros((B, H), bf),               # m
                jnp.zeros((B, K - 1, config.d_model), bf),  # conv_buf
            ))
    return states


def forward_step(base_xlstm: xLSTMParams, rc: RandCompressParams,
                 token: jnp.ndarray, states: list, num_heads: int, block_map: str,
                 hira_scale: float = 1.0):
    # token: [B] int32
    xlstm = apply_xlstm_hira(base_xlstm, rc, hira_scale)
    x = xlstm.embedding[token]   # [B, d_model]

    new_states = []
    for block, btype, state in zip(xlstm.blocks, block_map, states):
        x_n = layer_norm(x, block.norm1_w, block.norm1_b)
        if btype == 'm':
            cell_out, ns = mlstm_layer_step(block.mlstm, x_n, state, num_heads)
        else:
            cell_out, ns = slstm_layer_step(block.slstm, x_n, state, num_heads)
        new_states.append(ns)
        x = x + cell_out
        x_n = layer_norm(x, block.norm2_w, block.norm2_b)
        x = x + gated_ffn(block.ffn_up, block.ffn_down, x_n)

    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj), new_states   # [B, vocab_size]


# ============================================================================
# Initialization
# ============================================================================

def _bf(x): return x.astype(DTYPE)
def _s(n): return 1.0 / math.sqrt(n)

def init_mlstm_params(key, d_model, num_heads, conv_kernel) -> mLSTMParams:
    inner = d_model
    DH = inner // num_heads
    ks = jr.split(key, 8)
    return mLSTMParams(
        W_up=_bf(jr.normal(ks[0], (2*inner, d_model)) * _s(d_model)),
        W_q=_bf(jr.normal(ks[1], (num_heads, DH, inner)) * _s(inner)),
        W_k=_bf(jr.normal(ks[2], (num_heads, DH, inner)) * _s(inner)),
        W_v=_bf(jr.normal(ks[3], (num_heads, DH, inner)) * _s(inner)),
        W_i=_bf(jr.normal(ks[4], (num_heads, inner)) * _s(inner)),
        W_f=_bf(jr.normal(ks[5], (num_heads, inner)) * _s(inner)),
        skip=jnp.ones((inner,), DTYPE),
        ln_w=jnp.ones((inner,), DTYPE),
        ln_b=jnp.zeros((inner,), DTYPE),
        W_down=_bf(jr.normal(ks[6], (d_model, inner)) * _s(inner)),
        conv_w=_bf(jr.normal(ks[7], (inner, conv_kernel)) * _s(conv_kernel)),
    )

def init_slstm_params(key, d_model, conv_kernel) -> sLSTMParams:
    H = d_model
    ks = jr.split(key, 6)
    return sLSTMParams(
        W_i=_bf(jr.normal(ks[0], (H, d_model)) * _s(d_model)),
        W_f=_bf(jr.normal(ks[1], (H, d_model)) * _s(d_model)),
        W_z=_bf(jr.normal(ks[2], (H, d_model)) * _s(d_model)),
        W_o=_bf(jr.normal(ks[3], (H, d_model)) * _s(d_model)),
        Ry=_bf(jr.normal(ks[4], (4*H, H)) * _s(H)),
        b=jnp.zeros((4*H,), DTYPE),
        gn_w=jnp.ones((H,), DTYPE),
        gn_b=jnp.zeros((H,), DTYPE),
        conv_w=_bf(jr.normal(ks[5], (d_model, conv_kernel)) * _s(conv_kernel)),
    )

def init_xlstm_block_params(key, d_model, num_heads, d_ff, conv_kernel, btype) -> xLSTMBlockParams:
    ks = jr.split(key, 4)
    return xLSTMBlockParams(
        norm1_w=jnp.ones((d_model,), DTYPE),
        norm1_b=jnp.zeros((d_model,), DTYPE),
        norm2_w=jnp.ones((d_model,), DTYPE),
        norm2_b=jnp.zeros((d_model,), DTYPE),
        ffn_up=_bf(jr.normal(ks[2], (2*d_ff, d_model)) * _s(d_model)),
        ffn_down=_bf(jr.normal(ks[3], (d_model, d_ff)) * _s(d_ff)),
        mlstm=init_mlstm_params(ks[0], d_model, num_heads, conv_kernel) if btype == 'm' else None,
        slstm=init_slstm_params(ks[1], d_model, conv_kernel) if btype == 's' else None,
    )

def init_xlstm_params(key, config: Config) -> xLSTMParams:
    ks = jr.split(key, 2 + config.num_layers)
    emb = _bf(jr.normal(ks[0], (config.vocab_size, config.d_model)) * 0.01)
    blocks = [
        init_xlstm_block_params(ks[2+i], config.d_model, config.num_heads,
                                 config.d_ff, config.conv_kernel, config.block_map[i])
        for i in range(config.num_layers)
    ]
    return xLSTMParams(
        embedding=emb, blocks=blocks,
        norm_w=jnp.ones((config.d_model,), DTYPE),
        norm_b=jnp.zeros((config.d_model,), DTYPE),
    )


def init_hira_adapter(key, d_out, d_in, r) -> HiRAAdapter:
    # A scale 1/sqrt(r): keeps B·A magnitude consistent regardless of d_in
    A = jr.normal(key, (r, d_in)) / math.sqrt(r)
    return HiRAAdapter(A=A.astype(DTYPE), B=jnp.zeros((d_out, r), DTYPE))

def init_block_hira(key, config: Config, btype: str) -> BlockHiRA:
    d, d_ff, r = config.d_model, config.d_ff, config.lora_r
    inner = d
    NH = config.num_heads
    ks = jr.split(key, 12)
    ki = 0
    if btype == 'm':
        mhira = mLSTMHiRA(
            W_up=init_hira_adapter(ks[ki],   2*inner, d, r),
            W_q=init_hira_adapter(ks[ki+1],  inner,   inner, r),
            W_k=init_hira_adapter(ks[ki+2],  inner,   inner, r),
            W_v=init_hira_adapter(ks[ki+3],  inner,   inner, r),
            W_i=init_hira_adapter(ks[ki+4],  NH,      inner, r),
            W_f=init_hira_adapter(ks[ki+5],  NH,      inner, r),
            W_down=init_hira_adapter(ks[ki+6], d,     inner, r),
        ); ki += 7
        shira = None
    else:
        shira = sLSTMHiRA(
            W_i=init_hira_adapter(ks[ki], d, d, r),
            W_f=init_hira_adapter(ks[ki+1], d, d, r),
            W_z=init_hira_adapter(ks[ki+2], d, d, r),
            W_o=init_hira_adapter(ks[ki+3], d, d, r),
        ); ki += 4
        mhira = None
    return BlockHiRA(
        ffn_up=init_hira_adapter(ks[ki], 2*d_ff, d, r),
        ffn_down=init_hira_adapter(ks[ki+1], d, d_ff, r),
        mlstm=mhira, slstm=shira,
    )

def init_randcompress_params(key, config: Config) -> RandCompressParams:
    ks = jr.split(key, 2 + config.num_layers)
    return RandCompressParams(
        output_proj=_bf(jr.normal(ks[0], (config.d_model, config.vocab_size)) * 0.1),
        emb_hira=init_hira_adapter(ks[1], config.vocab_size, config.d_model, config.lora_r),
        blocks=[init_block_hira(ks[2+i], config, config.block_map[i]) for i in range(config.num_layers)],
    )


# ============================================================================
# AdamW (hand-written, no optax)
# ============================================================================

class AdamWState(NamedTuple):
    m: any
    v: any
    beta1_power: jnp.ndarray
    beta2_power: jnp.ndarray

def adamw_init(params, beta1=0.9, beta2=0.999) -> AdamWState:
    z = lambda p: jnp.zeros_like(p, jnp.float32)
    return AdamWState(
        m=jax.tree_util.tree_map(z, params),
        v=jax.tree_util.tree_map(z, params),
        beta1_power=jnp.array(beta1, jnp.float32),
        beta2_power=jnp.array(beta2, jnp.float32),
    )

def adamw_update(state, grads, params, lr, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0, max_norm=1.0):
    leaves = jax.tree_util.tree_leaves(grads)
    gnorm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (gnorm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

    m, v, b1p, b2p = state
    b1p = b1p * beta1
    b2p = b2p * beta2
    g32 = jax.tree_util.tree_map(lambda g: g.astype(jnp.float32), grads)
    m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, m, g32)
    v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1 - beta2) * g * g, v, g32)
    bc1, bc2 = 1.0 - b1p, 1.0 - b2p
    new_p = jax.tree_util.tree_map(
        lambda p, m_, v_: (
            p.astype(jnp.float32)
            - lr * (m_ / bc1 / (jnp.sqrt(v_ / bc2) + eps) + weight_decay * p.astype(jnp.float32))
        ).astype(p.dtype),
        params, m, v,
    )
    return new_p, AdamWState(m=m, v=v, beta1_power=b1p, beta2_power=b2p)


# ============================================================================
# Train / eval steps
# ============================================================================

def cross_entropy_loss(logits, targets):
    # Mask PAD_TOKEN positions so padding doesn't contribute to the loss.
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss_per_tok = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    mask = (targets != PAD_TOKEN).astype(jnp.float32)
    return (loss_per_tok * mask).sum() / (mask.sum() + 1e-8)

def cosine_lr(step, total_steps, lr_max, lr_min=1e-4, warmup=100):
    warmup_lr = lr_max * jnp.minimum(step / warmup, 1.0)
    t = jnp.maximum(step - warmup, 0) / jnp.maximum(total_steps - warmup, 1)
    cosine_lr_ = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(math.pi * t))
    return jnp.where(step < warmup, warmup_lr, cosine_lr_)

@jax.jit(static_argnames=["config"])
def train_step(base_xlstm, params, opt_state, batch, config, lr):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    def loss_fn(p):
        logits = forward_train(base_xlstm, p, inputs, config.num_heads, config.block_map,
                               config.hira_scale)
        return cross_entropy_loss(logits, targets)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_params, new_opt = adamw_update(
        opt_state, grads, params,
        lr=lr, beta1=config.adam_beta1, beta2=config.adam_beta2, eps=config.adam_eps,
        weight_decay=config.weight_decay, max_norm=config.grad_clip_norm,
    )
    return new_params, new_opt, loss

@jax.jit(static_argnames=["config"])
def eval_step(base_xlstm, params, batch, config):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = forward_train(base_xlstm, params, inputs, config.num_heads, config.block_map,
                           config.hira_scale)
    return cross_entropy_loss(logits, targets)


# ============================================================================
# Generation (recurrent)
# ============================================================================

_forward_step_jit = jax.jit(forward_step, static_argnames=["num_heads", "block_map", "hira_scale"])

def generate(base_xlstm, params, config, seed_tokens, max_length):
    states = init_step_states(config, batch_size=1)
    token = jnp.array([seed_tokens[0]])
    generated = []
    for _ in range(max_length):
        logit, states = _forward_step_jit(
            base_xlstm, params, token, states, config.num_heads, config.block_map,
            config.hira_scale,
        )
        token = jnp.argmax(logit, axis=-1)
        generated.append(int(token[0]))
    return np.array(generated, dtype=np.uint8)


# ============================================================================
# Data
# ============================================================================

class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text):
        if isinstance(text, str):
            text = text.encode('utf-8')
        return np.frombuffer(text, dtype=np.uint8)

    @staticmethod
    def decode(tokens):
        return bytes(np.array(tokens, dtype=np.uint8)).decode('utf-8', errors='replace')

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return ByteTokenizer.encode(f.read())

def create_batches(tokens, batch_size, seq_len, shuffle=False, key=None):
    seq_len = min(seq_len, len(tokens) - 1)
    if seq_len <= 0:
        return
    num_seq = (len(tokens) - 1) // seq_len
    indices = np.arange(num_seq) * seq_len
    if shuffle and key is not None:
        _, sk = jr.split(key)
        indices = np.array(jr.permutation(sk, indices))
    for i in range(0, len(indices), batch_size):
        batch = [tokens[idx: idx + seq_len + 1] for idx in indices[i: i + batch_size]
                 if len(tokens[idx: idx + seq_len + 1]) == seq_len + 1]
        if batch:
            yield jnp.array(np.stack(batch), dtype=jnp.int32)


# ============================================================================
# Persistent TBPTT helpers
# ============================================================================

def causal_conv1d_with_buf(x, w, buf):
    """Causal conv using an explicit left buffer (last K-1 timesteps from previous chunk)."""
    C, K = w.shape
    w = w.astype(x.dtype)
    x_pad = jnp.concatenate([buf, x], axis=1)   # [B, S+K-1, C]
    x_t = x_pad.transpose(0, 2, 1)
    w_t = w[:, None, :]
    out = jax.lax.conv_general_dilated(
        x_t, w_t, window_strides=(1,), padding='VALID',
        feature_group_count=C, dimension_numbers=('NCH', 'OIH', 'NCH'),
    )
    return out.transpose(0, 2, 1)


def mlstm_cell_parallel_with_init_state(q, k, v, i_preact, f_preact, C0, n0, m0):
    """Parallel mLSTM with non-zero initial state (C0, n0, m0).

    The initial state decays by cumulative forget gates.
    Combined stability = max(seq stability, init weight).
    Returns h [B,NH,S,DH] and final state (C_T, n_T, m_T).
    """
    B, NH, S, DH = q.shape
    log_f = jax.nn.log_sigmoid(f_preact)                           # [B, NH, S, 1]
    lf_cs = jnp.concatenate([
        jnp.zeros((B, NH, 1, 1), log_f.dtype),
        jnp.cumsum(log_f, axis=2),
    ], axis=2)                                                      # [B, NH, S+1, 1]
    rep = lf_cs[..., 0]                                            # [B, NH, S+1]
    log_fg = rep[:, :, 1:, None] - rep[:, :, None, 1:]            # [B, NH, S, S]
    mask = jnp.tril(jnp.ones((S, S), bool))
    log_fg = jnp.where(mask, log_fg, -jnp.inf)
    log_D_seq = log_fg + i_preact.transpose(0, 1, 3, 2)           # [B, NH, S, S]

    # Initial state log-weight at each position t: m0 + cumsum_log_f[t+1]
    log_w_init = m0[:, :, None] + lf_cs[:, :, 1:, 0]             # [B, NH, S]

    max_log_D = jnp.maximum(log_D_seq.max(axis=-1), log_w_init)[:, :, :, None]  # [B,NH,S,1]
    D_seq  = jnp.exp(log_D_seq - max_log_D)                       # [B, NH, S, S]
    d_init = jnp.exp(log_w_init - max_log_D[:, :, :, 0])         # [B, NH, S]

    k_s = k / jnp.sqrt(jnp.array(DH, k.dtype))
    qk   = jnp.einsum('bnsd,bntd->bnst', q, k_s)

    h_num_seq  = jnp.einsum('bnst,bntd->bnsd', qk * D_seq, v)
    h_num_init = jnp.einsum('bnsd,bnde->bnse', q, C0) * d_init[:, :, :, None]
    h_num = h_num_seq + h_num_init

    n_seq  = jnp.einsum('bnst,bntd->bnsd', D_seq, k_s)
    n_init = n0[:, :, None, :] * d_init[:, :, :, None]
    n_full = n_seq + n_init

    qn    = (q * n_full).sum(axis=-1)
    denom = jnp.maximum(jnp.abs(qn), jnp.exp(-max_log_D[:, :, :, 0]))
    h     = h_num / (denom[:, :, :, None] + 1e-8)

    # Final state
    w_T = D_seq[:, :, -1, :]
    d_T = d_init[:, :, -1]
    C_T = jnp.einsum('bns,bnsd,bnse->bnde', w_T, k_s, v) + d_T[:, :, None, None] * C0
    n_T = n_full[:, :, -1, :]
    m_T = max_log_D[:, :, -1, 0]
    return h, (C_T, n_T, m_T)


def mlstm_layer_parallel_with_init(p: mLSTMParams, x: jnp.ndarray, num_heads: int, state):
    """Parallel mLSTM layer with initial state. state = (C0, n0, m0, conv_buf)."""
    C0, n0, m0, conv_buf = state
    B, S, _ = x.shape
    inner_dim = p.W_up.shape[0] // 2
    K = p.conv_w.shape[1]

    xz  = jnp.dot(x, p.W_up.T)
    x_m, z = xz[..., :inner_dim], xz[..., inner_dim:]

    x_conv   = causal_conv1d_with_buf(x_m, p.conv_w, conv_buf)
    x_conv_a = jax.nn.silu(x_conv)

    q   = jnp.einsum('ndi,bsi->bnsd', p.W_q, x_conv_a)
    k   = jnp.einsum('ndi,bsi->bnsd', p.W_k, x_conv_a)
    v   = jnp.einsum('ndi,bsi->bnsd', p.W_v, x_m)
    i_p = jnp.einsum('ni,bsi->bns',   p.W_i, x_conv_a)[..., None]
    f_p = jnp.einsum('ni,bsi->bns',   p.W_f, x_conv_a)[..., None]

    h, (C_T, n_T, m_T) = mlstm_cell_parallel_with_init_state(q, k, v, i_p, f_p, C0, n0, m0)
    h_flat = h.transpose(0, 2, 1, 3).reshape(B, S, inner_dim)
    h_out  = (h_flat + p.skip * x_conv_a) * jax.nn.silu(z)
    h_norm = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    out    = jnp.dot(h_norm, p.W_down.T)

    new_conv_buf = x_m[:, -(K - 1):, :]
    return out, (C_T, n_T, m_T, new_conv_buf)


def slstm_layer_with_init(p: sLSTMParams, x: jnp.ndarray, num_heads: int, state):
    """sLSTM layer with initial state. state = (y0, c0, n0, m0, conv_buf)."""
    y0, c0, n0, m0, conv_buf = state
    K = p.conv_w.shape[1]

    x_conv   = causal_conv1d_with_buf(x, p.conv_w, conv_buf)
    x_conv_a = jax.nn.silu(x_conv)

    gates = jnp.concatenate([
        jnp.dot(x_conv_a, p.W_i.T), jnp.dot(x_conv_a, p.W_f.T),
        jnp.dot(x,        p.W_z.T), jnp.dot(x,        p.W_o.T),
    ], axis=-1)

    def step(state, gate_t):
        sy, sc, sn, sm = state
        raw = gate_t + jnp.dot(sy, p.Ry.T) + p.b
        y_t, c_t, n_t, m_t = slstm_cell(raw, sc, sn, sm)
        return (y_t, c_t, n_t, m_t), y_t

    (y_T, c_T, n_T, m_T), ys = jax.lax.scan(step, (y0, c0, n0, m0), gates.transpose(1, 0, 2))
    out = multihead_layer_norm(ys.transpose(1, 0, 2), p.gn_w, p.gn_b, num_heads)
    return out, (y_T, c_T, n_T, m_T, x[:, -(K - 1):, :])


def forward_train_chunked(base_xlstm, rc, inputs, layer_states, num_heads, block_map,
                          hira_scale: float = 1.0):
    """Forward through one chunk starting from layer_states.
    inputs: [B, S]; returns (logits [B,S,V], new_layer_states).
    """
    xlstm = apply_xlstm_hira(base_xlstm, rc, hira_scale)
    x = xlstm.embedding[inputs]
    new_states = []
    for block, btype, state in zip(xlstm.blocks, block_map, layer_states):
        x_n = layer_norm(x, block.norm1_w, block.norm1_b)
        if btype == 'm':
            cell_out, ns = mlstm_layer_parallel_with_init(block.mlstm, x_n, num_heads, state)
        else:
            cell_out, ns = slstm_layer_with_init(block.slstm, x_n, num_heads, state)
        new_states.append(ns)
        x = x + cell_out
        x_n = layer_norm(x, block.norm2_w, block.norm2_b)
        x = x + gated_ffn(block.ffn_up, block.ffn_down, x_n)
    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj), new_states


# ============================================================================
# Persistent TBPTT train step
# ============================================================================

@jax.jit(static_argnames=["config"])
def train_chunk_persistent(base_xlstm, params, opt_state, chunk_batch, layer_states, config, lr):
    """Train on one chunk. layer_states must be stop_gradient'd before passing.
    Returns (new_params, new_opt, loss, new_layer_states).
    """
    inputs, targets = chunk_batch[:, :-1], chunk_batch[:, 1:]

    def loss_fn(p):
        logits, new_states = forward_train_chunked(
            base_xlstm, p, inputs, layer_states, config.num_heads, config.block_map,
            config.hira_scale,
        )
        return cross_entropy_loss(logits, targets), new_states

    (loss, new_states), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    new_params, new_opt = adamw_update(
        opt_state, grads, params,
        lr=lr, beta1=config.adam_beta1, beta2=config.adam_beta2, eps=config.adam_eps,
        weight_decay=config.weight_decay, max_norm=config.grad_clip_norm,
    )
    return new_params, new_opt, loss, new_states


@jax.jit(static_argnames=["config"])
def grad_chunk_persistent(base_xlstm, params, chunk_batch, layer_states, config):
    """Compute loss + gradients for one chunk without updating params.
    Used by gradient accumulation. Returns (loss, grads, new_layer_states).
    """
    inputs, targets = chunk_batch[:, :-1], chunk_batch[:, 1:]

    def loss_fn(p):
        logits, new_states = forward_train_chunked(
            base_xlstm, p, inputs, layer_states, config.num_heads, config.block_map,
            config.hira_scale,
        )
        return cross_entropy_loss(logits, targets), new_states

    (loss, new_states), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, grads, new_states


# ============================================================================
# Main — Persistent TBPTT
# ============================================================================

def _eval_generative(base_xlstm, params, config, seed_byte, target_np):
    """Greedy generation from seed_byte; compare to target_np (uint8 array).
    Returns (generated_np, bytes_wrong, accuracy, cer, first_wrong_idx).
    first_wrong_idx is the byte index (0-based into target_np) of the first
    mismatch, or None if all bytes are correct.
    """
    n = len(target_np)
    cur_token  = jnp.array([seed_byte])
    states     = init_step_states(config, batch_size=1)
    gen_result = []
    for _ in tqdm(range(n), desc="  gen", unit="tok", leave=False):
        logit, states = _forward_step_jit(
            base_xlstm, params, cur_token, states, config.num_heads, config.block_map,
            config.hira_scale,
        )
        cur_token = jnp.argmax(logit, axis=-1)
        gen_result.append(int(cur_token[0]))
    generated   = np.array(gen_result, dtype=np.uint8)
    wrong_mask  = generated != target_np
    bytes_wrong = int(np.sum(wrong_mask))
    accuracy    = (n - bytes_wrong) / n
    cer         = bytes_wrong / n
    wrong_idxs  = np.where(wrong_mask)[0]
    first_wrong = int(wrong_idxs[0]) if len(wrong_idxs) > 0 else None
    return generated, bytes_wrong, accuracy, cer, first_wrong


def main():
    # =========================================================================
    # Config selection — uncomment exactly one
    # =========================================================================

    # cfg_v1_baseline: mirrors v1 defaults; use as reference / sanity check
    # config = Config(
    #     vocab_size=256, d_model=128, num_heads=8, d_ff=256,
    #     num_layers=4, seed=0, conv_kernel=4,
    #     block_map="msms", lora_r=16, hira_scale=1.0,
    #     batch_size=1, learning_rate=1e-4,
    #     adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
    #     weight_decay=0.0, grad_clip_norm=1.0, grad_accum_k=1,
    # )

    # cfg_faster_adam: higher LR + faster beta2; no other changes
    # config = Config(
    #     vocab_size=256, d_model=128, num_heads=8, d_ff=256,
    #     num_layers=4, seed=0, conv_kernel=4,
    #     block_map="msms", lora_r=16, hira_scale=1.0,
    #     batch_size=1, learning_rate=1e-3,
    #     adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
    #     weight_decay=0.0, grad_clip_norm=5.0, grad_accum_k=1,
    # )

    # cfg_hira_scaled: hira_scale=2 amplifies adapter effect; pairs with lower base LR
    # config = Config(
    #     vocab_size=256, d_model=128, num_heads=8, d_ff=256,
    #     num_layers=4, seed=0, conv_kernel=4,
    #     block_map="mmmm", lora_r=32, hira_scale=2.0,
    #     batch_size=1, learning_rate=5e-4,
    #     adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
    #     weight_decay=0.0, grad_clip_norm=5.0, grad_accum_k=1,
    # )

    # cfg_slstm_only: all sLSTM — train/eval compute identical (both scan);
    # trades mLSTM within-chunk expressiveness for perfect train-eval match
    # config = Config(
    #     vocab_size=256, d_model=128, num_heads=8, d_ff=256,
    #     num_layers=4, seed=0, conv_kernel=4,
    #     block_map="ssss", lora_r=32, hira_scale=2.0,
    #     batch_size=1, learning_rate=1e-3,
    #     adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
    #     weight_decay=0.0, grad_clip_norm=5.0, grad_accum_k=1,
    # )

    # cfg_high_rank: high rank for maximum adapter capacity; aggressive settings
    config = Config(
        vocab_size=256, d_model=128, num_heads=8, d_ff=256,
        num_layers=4, seed=0, conv_kernel=4,
        block_map="mmmm", lora_r=16, hira_scale=2.0,
        batch_size=1, learning_rate=1e-3,
        adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8,
        weight_decay=0.0, grad_clip_norm=5.0, grad_accum_k=1,
    )

    # =========================================================================

    chunk_size = 1024
    num_epochs = 20

    train_log_path = "train_hira_persistent.log"
    eval_log_path  = "eval_final.log"
    train_log_file = open(train_log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, train_log_file)

    dataset_path = "datasets/quran-uthmani.txt"
    val_path     = "datasets/surat_al-fatihah.txt"
    tokens     = load_dataset(dataset_path)
    val_tokens = load_dataset(val_path)
    n_real = len(tokens)
    n_val  = len(val_tokens)
    print(f"Train: {n_real} bytes   Val: {n_val} bytes   chunk_size={chunk_size}")

    key = jr.key(config.seed)
    key, k_xlstm, k_rc = jr.split(key, 3)
    base_xlstm = init_xlstm_params(k_xlstm, config)
    frozen    = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(base_xlstm))
    params    = init_randcompress_params(k_rc, config)
    trainable = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    total     = frozen + trainable
    print(f"Params  — frozen: {frozen/1e6:.3f}M  trainable: {trainable/1e6:.3f}M  total: {total/1e6:.3f}M")
    param_bytes    = trainable * np.dtype(DTYPE).itemsize
    compress_ratio = n_real / param_bytes
    print(f"Ratios  — trainable/total: {trainable/total*100:.1f}%"
          f"  trainable/dataset: {trainable/n_real:.2f} params-per-byte"
          f"  total/dataset: {total/n_real:.2f} params-per-byte"
          f"  ({n_real/1e6:.3f}M bytes dataset)")
    print(f"Compression — dataset {n_real/1e6:.3f}MB / trainable {param_bytes/1e6:.3f}MB"
          f"  = {compress_ratio:.2f}x  ({'compressed' if compress_ratio > 1 else 'over-parameterised'})")
    print(f"Config — block_map={config.block_map}  r={config.lora_r}  hira_scale={config.hira_scale}"
          f"  lr={config.learning_rate}  beta2={config.adam_beta2}  clip={config.grad_clip_norm}")
    opt_state = adamw_init(params, beta1=config.adam_beta1, beta2=config.adam_beta2)

    # Build non-overlapping chunks: [num_chunks, chunk_size+1]
    usable     = (n_real - 1) // chunk_size * chunk_size + 1
    raw        = tokens[:usable]
    num_chunks = (len(raw) - 1) // chunk_size
    chunks     = [raw[i*chunk_size : i*chunk_size + chunk_size + 1] for i in range(num_chunks)]
    print(f"Chunks: {num_chunks}   steps/epoch={num_chunks}")

    # Validation tensors — al-Fatihah fits in one forward pass
    val_inputs_jnp  = jnp.array(val_tokens[:-1][None, :], dtype=jnp.int32)
    val_targets_jnp = jnp.array(val_tokens[1:][None, :],  dtype=jnp.int32)
    val_target_np   = np.array(val_tokens[1:], dtype=np.uint8)
    n_val_pred      = n_val - 1

    total_steps = num_epochs * num_chunks
    global_step = 0
    t_total     = 0.0
    K           = config.grad_accum_k

    print(f"\nPersistent TBPTT for {num_epochs} epochs  grad_accum_k={K}  "
          f"log -> {train_log_path}")
    print("-" * 60)

    for epoch in range(num_epochs):
        states     = init_step_states(config, batch_size=1)
        epoch_loss = 0.0
        recent     = []

        pbar = tqdm(total=num_chunks, desc=f"E{epoch+1:04d}", unit="chunk",
                    dynamic_ncols=True, leave=True)

        for block_start in range(0, num_chunks, K):
            block_end = min(block_start + K, num_chunks)
            k_actual  = block_end - block_start

            accum_grads = None
            block_loss  = 0.0
            t0 = time.perf_counter()

            for ci in range(block_start, block_end):
                lr            = config.learning_rate
                chunk_batch   = jnp.array(chunks[ci][None, :], dtype=jnp.int32)
                frozen_states = jax.lax.stop_gradient(states)

                loss, grads, states = grad_chunk_persistent(
                    base_xlstm, params, chunk_batch, frozen_states, config
                )
                loss_f     = float(loss)
                block_loss += loss_f
                global_step += 1

                accum_grads = (
                    grads if accum_grads is None
                    else jax.tree_util.tree_map(jnp.add, accum_grads, grads)
                )
                pbar.update(1)

            if k_actual > 1:
                accum_grads = jax.tree_util.tree_map(
                    lambda g: g / k_actual, accum_grads
                )
            params, opt_state = adamw_update(
                opt_state, accum_grads, params,
                lr=lr, beta1=config.adam_beta1, beta2=config.adam_beta2,
                eps=config.adam_eps, weight_decay=config.weight_decay,
                max_norm=config.grad_clip_norm,
            )
            t_total += (time.perf_counter() - t0) * 1000

            avg_block_loss = block_loss / k_actual
            epoch_loss    += block_loss
            recent.append(avg_block_loss)
            if len(recent) > 50: recent.pop(0)
            bpc     = math.exp(avg_block_loss) / math.log(2)
            avg_bpc = math.exp(sum(recent) / len(recent)) / math.log(2)
            pbar.set_postfix(loss=f"{avg_block_loss:.4f}", bpc=f"{bpc:.3f}",
                             avg_bpc=f"{avg_bpc:.3f}",
                             ms=f"{t_total/max(global_step,1):.1f}")

        pbar.close()
        avg_loss = epoch_loss / num_chunks
        avg_ms   = t_total / max(global_step, 1)
        bpc      = math.exp(avg_loss) / math.log(2)
        print(f"  epoch {epoch+1:4d}  loss={avg_loss:.5f}  bpc={bpc:.4f}"
              f"  lr={float(lr):.2e}  avg_ms/chunk={avg_ms:.1f}")

        # ---- per-epoch validation on surat al-Fatihah ----
        val_logits = forward_train(base_xlstm, params, val_inputs_jnp,
                                   config.num_heads, config.block_map, config.hira_scale)
        val_loss   = float(cross_entropy_loss(val_logits, val_targets_jnp))
        val_bpc    = math.exp(val_loss) / math.log(2)

        val_gen, val_wrong, val_acc, _, val_first_wrong = _eval_generative(
            base_xlstm, params, config,
            seed_byte=int(val_tokens[0]), target_np=val_target_np,
        )
        val_gen_str   = ByteTokenizer.decode(val_gen)
        val_first_str = (f"byte {val_first_wrong}" if val_first_wrong is not None
                         else "none (perfect)")
        print(f"  [VAL] bpc={val_bpc:.4f}  acc={val_acc*100:.2f}%  "
              f"wrong={val_wrong}/{n_val_pred}  first_wrong={val_first_str}")
        print(f"  [GEN] {val_gen_str}")

        if avg_loss < 0.01:
            print(f"\n  Converged at epoch {epoch+1}")
            break

    # ---- final evaluation on full training dataset ----
    print("\n" + "=" * 60)
    print("Final evaluation on training dataset...")
    print("=" * 60)

    eval_log_file = open(eval_log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, train_log_file, eval_log_file)

    print("Computing train BPC (teacher-forced, chunked)...")
    eval_states     = init_step_states(config, batch_size=1)
    total_eval_loss = 0.0
    for chunk_np in tqdm(chunks, desc="  tf-bpc", unit="chunk"):
        chunk_batch   = jnp.array(chunk_np[None, :], dtype=jnp.int32)
        frozen_states = jax.lax.stop_gradient(eval_states)
        inputs_c      = chunk_batch[:, :-1]
        targets_c     = chunk_batch[:, 1:]
        logits_c, eval_states = forward_train_chunked(
            base_xlstm, params, inputs_c, frozen_states,
            config.num_heads, config.block_map, config.hira_scale,
        )
        total_eval_loss += float(cross_entropy_loss(logits_c, targets_c))
    train_bpc = math.exp(total_eval_loss / num_chunks) / math.log(2)
    print(f"Train BPC (teacher-forced): {train_bpc:.4f}")

    # generative evaluation: seed = first byte of train
    n_train_pred  = n_real - 1
    train_target  = np.array(tokens[1:], dtype=np.uint8)
    print(f"Generating {n_train_pred} bytes from first train byte "
          f"(byte=0x{int(tokens[0]):02x})...")
    train_gen, train_wrong, train_acc, train_cer, train_first_wrong = _eval_generative(
        base_xlstm, params, config,
        seed_byte=int(tokens[0]), target_np=train_target,
    )
    train_first_str = (f"byte {train_first_wrong}" if train_first_wrong is not None
                       else "none (perfect)")

    print(f"\n{'='*60}")
    print(f"Final train-set evaluation ({n_train_pred} bytes):")
    print(f"  BPC (teacher-forced): {train_bpc:.4f}")
    print(f"  Bytes wrong:          {train_wrong} / {n_train_pred}")
    print(f"  First wrong byte idx: {train_first_str}")
    print(f"  Accuracy:             {train_acc*100:.4f}%")
    print(f"  CER:                  {train_cer*100:.4f}%")
    print(f"{'='*60}")

    eval_log_file.close()
    train_log_file.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
