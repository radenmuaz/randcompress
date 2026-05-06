"""
randcompress v2.1 — HiRA + structured frozen basis + gradient accumulation
               + sequential curriculum learning

Key changes from v2
-------------------
Curriculum learning (sequential memorization):
  Memorization is sequential — the recurrent state builds context left-to-right,
  so earlier bytes must be solid before later bytes can be learned.

  Schedule: each epoch covers a growing prefix of the dataset.
    window = min(curriculum_start_chunks + curriculum_step_chunks * epoch, n_chunks)
  The active window is repeated `curriculum_repeat` times per epoch so early
  bytes receive many gradient updates before new bytes are introduced.
  State is NOT reset between repetitions within an epoch — each pass sees the
  data with the state context built up by the previous pass (richer signal).

  Baked-in detection (curriculum_baked_threshold > 0):
    Tracks per-chunk EMA loss. On repetition passes (pass_idx > 0), chunks whose
    loss has fallen below the threshold are considered "baked" — we still run a
    forward pass to advance state but skip backprop, saving compute without
    losing sequential context.

  curriculum=False → standard TBPTT over all chunks (identical to v2).

Inherited from v2
-----------------
  Structured frozen basis, HiRA with orthonormal A, grad accumulation,
  per-epoch validation, final eval on full train set.
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

PAD_TOKEN = 0
DTYPE     = jnp.float32

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
    vocab_size:    int   = 256
    d_model:       int   = 256
    num_heads:     int   = 8
    d_ff:          int   = 1024
    num_layers:    int   = 4
    max_seq_len:   int   = 512     # also used for multiscale τ_max
    seed:          int   = 0
    conv_kernel:   int   = 4
    block_map:     str   = "mmmm"  # 'm'=mLSTM 's'=sLSTM, len==num_layers
    lora_r:        int   = 8       # HiRA rank r
    batch_size:    int   = 16
    learning_rate: float = 1e-3
    weight_decay:  float = 0.0
    grad_clip_norm:float = 1.0
    grad_accum_k:  int   = 1       # chunks per update; 1 = standard TBPTT
    # ---- curriculum ----
    curriculum:                 bool  = False  # enable sequential curriculum
    curriculum_start_chunks:    int   = 4      # initial window (chunks)
    curriculum_step_chunks:     int   = 4      # chunks added per epoch
    curriculum_repeat:          int   = 4      # full passes over window per epoch
    curriculum_baked_threshold: float = 0.0   # skip backprop if chunk loss < this (0=off)


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

def causal_conv1d(x, w):
    C, K = w.shape
    w    = w.astype(x.dtype)
    x_pad = jnp.pad(x, ((0,0),(K-1,0),(0,0)))
    x_t   = x_pad.transpose(0,2,1)
    w_t   = w[:, None, :]
    out   = jax.lax.conv_general_dilated(
        x_t, w_t, window_strides=(1,), padding='VALID',
        feature_group_count=C, dimension_numbers=('NCH','OIH','NCH'),
    )
    return out.transpose(0,2,1)

def causal_conv1d_step(x_t, conv_buf, w):
    w      = w.astype(x_t.dtype)
    window = jnp.concatenate([conv_buf, x_t[:,None,:]], axis=1)
    return jnp.einsum('bkc,ck->bc', window, w), window[:,1:,:]

def gated_ffn(ffn_up, ffn_down, x):
    h    = jnp.dot(x, ffn_up.T)
    d_ff = h.shape[-1] // 2
    return jnp.dot(jax.nn.gelu(h[...,:d_ff]) * h[...,d_ff:], ffn_down.T)


# ============================================================================
# mLSTM cell
# ============================================================================

def _mlstm_cell_parallel_inner(q, k, v, i_preact, f_preact):
    B, NH, S, DH = q.shape
    log_f  = jax.nn.log_sigmoid(f_preact)
    lf_cs  = jnp.concatenate([
        jnp.zeros((B,NH,1,1), log_f.dtype),
        jnp.cumsum(log_f, axis=2),
    ], axis=2)
    rep    = lf_cs[...,0]
    log_fg = rep[:,:,1:,None] - rep[:,:,None,1:]
    mask   = jnp.tril(jnp.ones((S,S), bool))
    log_fg = jnp.where(mask, log_fg, -jnp.inf)
    log_D  = log_fg + i_preact.transpose(0,1,3,2)
    max_log_D = log_D.max(axis=-1, keepdims=True)
    D      = jnp.exp(log_D - max_log_D)
    k_s    = k / jnp.sqrt(jnp.array(DH, k.dtype))
    qk     = jnp.einsum('bnsd,bntd->bnst', q, k_s)
    C      = qk * D
    norm   = jnp.maximum(jnp.abs(C.sum(axis=-1, keepdims=True)), jnp.exp(-max_log_D))
    h      = jnp.einsum('bnst,bntd->bnsd', C/norm, v)
    return h, (k_s, v, log_D, max_log_D)

def mlstm_cell_parallel(q, k, v, i_preact, f_preact):
    h, _ = _mlstm_cell_parallel_inner(q, k, v, i_preact, f_preact)
    return h

def mlstm_cell_parallel_with_state(q, k, v, i_preact, f_preact):
    h, (k_s, v_, log_D, max_log_D) = _mlstm_cell_parallel_inner(q, k, v, i_preact, f_preact)
    m_T  = max_log_D[:,:,-1,0]
    w_T  = jnp.exp(log_D[:,:,-1,:] - m_T[:,:,None])
    C_T  = jnp.einsum('bns,bnsd,bnse->bnde', w_T, k_s, v_)
    n_T  = jnp.einsum('bns,bnsd->bnd', w_T, k_s)
    return h, (C_T, n_T, m_T)

def mlstm_cell_step(q, k, v, i_preact, f_preact, C_prev, n_prev, m_prev):
    DH     = q.shape[-1]
    log_f  = jax.nn.log_sigmoid(f_preact)
    m_t    = jnp.maximum(log_f + m_prev, i_preact)
    f_t    = jnp.exp(log_f + m_prev - m_t)
    i_t    = jnp.exp(i_preact - m_t)
    k_s    = k / jnp.sqrt(jnp.array(DH, k.dtype))
    kv     = jnp.einsum('bnd,bne->bnde', k_s, v)
    C_t    = f_t[:,:,None,None]*C_prev + i_t[:,:,None,None]*kv
    n_t    = f_t[:,:,None]*n_prev + i_t[:,:,None]*k_s
    h_num  = jnp.einsum('bnd,bnde->bne', q, C_t)
    denom  = jnp.maximum(jnp.abs((q*n_t).sum(axis=-1)), jnp.exp(-m_t))
    return h_num/(denom[:,:,None]+1e-8), C_t, n_t, m_t


# ============================================================================
# sLSTM cell
# ============================================================================

def slstm_cell(raw, sc, sn, sm):
    i_raw, f_raw, z_raw, o_raw = jnp.split(raw, 4, axis=-1)
    log_f = logsigmoid(f_raw)
    m_t   = jnp.maximum(i_raw, log_f + sm)
    i_t   = jnp.minimum(1.0, jnp.exp(i_raw - m_t))
    f_t   = jnp.minimum(1.0, jnp.exp(log_f + sm - m_t))
    o_t   = jax.nn.sigmoid(o_raw)
    c_t   = f_t*sc + i_t*jnp.tanh(z_raw)
    n_t   = f_t*sn + i_t
    y_t   = o_t * c_t / (jnp.abs(n_t) + 1e-8)
    return y_t, c_t, n_t, m_t


# ============================================================================
# Parameter structures
# ============================================================================

class mLSTMParams(NamedTuple):
    W_up:  jnp.ndarray   # [2*inner, d_model]
    W_q:   jnp.ndarray   # [NH, DH, inner]
    W_k:   jnp.ndarray   # [NH, DH, inner]
    W_v:   jnp.ndarray   # [NH, DH, inner]
    W_i:   jnp.ndarray   # [NH, inner]
    W_f:   jnp.ndarray   # [NH, inner]
    b_f:   jnp.ndarray   # [NH]  per-head forget bias (multi-timescale)
    b_i:   jnp.ndarray   # [NH]  per-head input  bias
    skip:  jnp.ndarray   # [inner]
    ln_w:  jnp.ndarray   # [inner]
    ln_b:  jnp.ndarray   # [inner]
    W_down:jnp.ndarray   # [d_model, inner]
    conv_w:jnp.ndarray   # [inner, conv_kernel]

class sLSTMParams(NamedTuple):
    W_i:   jnp.ndarray   # [H, d_model]
    W_f:   jnp.ndarray
    W_z:   jnp.ndarray
    W_o:   jnp.ndarray
    Ry:    jnp.ndarray   # [4H, H]  near-critical recurrent weight
    b:     jnp.ndarray   # [4H]     b[H:2H] = multi-timescale forget bias
    gn_w:  jnp.ndarray   # [H]
    gn_b:  jnp.ndarray
    conv_w:jnp.ndarray   # [d_model, conv_kernel]

class xLSTMBlockParams(NamedTuple):
    norm1_w:  jnp.ndarray
    norm1_b:  jnp.ndarray
    norm2_w:  jnp.ndarray
    norm2_b:  jnp.ndarray
    ffn_up:   jnp.ndarray   # [2*d_ff, d_model]
    ffn_down: jnp.ndarray   # [d_model, d_ff]
    mlstm: Optional[mLSTMParams] = None
    slstm: Optional[sLSTMParams] = None

class xLSTMParams(NamedTuple):
    embedding: jnp.ndarray  # [vocab_size, d_model]
    blocks:    list
    norm_w:    jnp.ndarray
    norm_b:    jnp.ndarray


# ============================================================================
# HiRA structures
# ============================================================================

class HiRAAdapter(NamedTuple):
    A: jnp.ndarray   # [r, d_in]  orthonormal rows (SVD init)
    B: jnp.ndarray   # [d_out, r] zero init

class mLSTMHiRA(NamedTuple):
    W_up:  HiRAAdapter
    W_q:   HiRAAdapter
    W_k:   HiRAAdapter
    W_v:   HiRAAdapter
    W_i:   HiRAAdapter
    W_f:   HiRAAdapter
    W_down:HiRAAdapter

class sLSTMHiRA(NamedTuple):
    W_i: HiRAAdapter
    W_f: HiRAAdapter
    W_z: HiRAAdapter
    W_o: HiRAAdapter

class BlockHiRA(NamedTuple):
    ffn_up:  HiRAAdapter
    ffn_down:HiRAAdapter
    mlstm: Optional[mLSTMHiRA] = None
    slstm: Optional[sLSTMHiRA] = None

class RandCompressParams(NamedTuple):
    output_proj: jnp.ndarray   # [d_model, vocab_size]
    emb_hira:    HiRAAdapter
    blocks:      list


# ============================================================================
# HiRA application  ΔW = W₀ ⊙ (B·A)
# ============================================================================

def apply_hira(base, hira):
    return base + base * jnp.dot(hira.B, hira.A)

def apply_mlstm_hira(p: mLSTMParams, l: mLSTMHiRA) -> mLSTMParams:
    NH, DH, inner = p.W_q.shape
    dm = NH * DH
    def qkv(w, a):
        return apply_hira(w.reshape(dm, inner), a).reshape(NH, DH, inner)
    return mLSTMParams(
        W_up  =apply_hira(p.W_up,  l.W_up),
        W_q   =qkv(p.W_q, l.W_q), W_k=qkv(p.W_k, l.W_k), W_v=qkv(p.W_v, l.W_v),
        W_i   =apply_hira(p.W_i,   l.W_i),
        W_f   =apply_hira(p.W_f,   l.W_f),
        b_f   =p.b_f,   # frozen — structural timescale basis
        b_i   =p.b_i,   # frozen
        skip  =p.skip, ln_w=p.ln_w, ln_b=p.ln_b,
        W_down=apply_hira(p.W_down, l.W_down),
        conv_w=p.conv_w,
    )

def apply_slstm_hira(p: sLSTMParams, l: sLSTMHiRA) -> sLSTMParams:
    return sLSTMParams(
        W_i=apply_hira(p.W_i, l.W_i), W_f=apply_hira(p.W_f, l.W_f),
        W_z=apply_hira(p.W_z, l.W_z), W_o=apply_hira(p.W_o, l.W_o),
        Ry=p.Ry, b=p.b, gn_w=p.gn_w, gn_b=p.gn_b, conv_w=p.conv_w,
    )

def apply_block_hira(base: xLSTMBlockParams, hira: BlockHiRA) -> xLSTMBlockParams:
    return xLSTMBlockParams(
        norm1_w=base.norm1_w, norm1_b=base.norm1_b,
        norm2_w=base.norm2_w, norm2_b=base.norm2_b,
        ffn_up  =apply_hira(base.ffn_up,   hira.ffn_up),
        ffn_down=apply_hira(base.ffn_down, hira.ffn_down),
        mlstm=apply_mlstm_hira(base.mlstm, hira.mlstm) if base.mlstm is not None else None,
        slstm=apply_slstm_hira(base.slstm, hira.slstm) if base.slstm is not None else None,
    )

def apply_xlstm_hira(base: xLSTMParams, rc: RandCompressParams) -> xLSTMParams:
    return xLSTMParams(
        embedding=apply_hira(base.embedding, rc.emb_hira),
        blocks=[apply_block_hira(b, l) for b, l in zip(base.blocks, rc.blocks)],
        norm_w=base.norm_w, norm_b=base.norm_b,
    )


# ============================================================================
# mLSTM layers  (b_f / b_i added to gate preacts)
# ============================================================================

def _mlstm_preacts(p, x_conv_a, x_m):
    q   = jnp.einsum('ndi,bsi->bnsd', p.W_q, x_conv_a)
    k   = jnp.einsum('ndi,bsi->bnsd', p.W_k, x_conv_a)
    v   = jnp.einsum('ndi,bsi->bnsd', p.W_v, x_m)
    i_p = (jnp.einsum('ni,bsi->bns', p.W_i, x_conv_a)
           + p.b_i[None, :, None])[..., None]
    f_p = (jnp.einsum('ni,bsi->bns', p.W_f, x_conv_a)
           + p.b_f[None, :, None])[..., None]
    return q, k, v, i_p, f_p

def _mlstm_preacts_step(p, x_conv_a, x_m):
    q   = jnp.einsum('ndi,bi->bnd', p.W_q, x_conv_a)
    k   = jnp.einsum('ndi,bi->bnd', p.W_k, x_conv_a)
    v   = jnp.einsum('ndi,bi->bnd', p.W_v, x_m)
    i_p = jnp.einsum('ni,bi->bn', p.W_i, x_conv_a) + p.b_i[None, :]
    f_p = jnp.einsum('ni,bi->bn', p.W_f, x_conv_a) + p.b_f[None, :]
    return q, k, v, i_p, f_p

def mlstm_layer_parallel(p: mLSTMParams, x: jnp.ndarray, num_heads: int):
    B, S, _ = x.shape
    inner   = p.W_up.shape[0] // 2
    xz      = jnp.dot(x, p.W_up.T)
    x_m, z  = xz[...,:inner], xz[...,inner:]
    x_conv_a= jax.nn.silu(causal_conv1d(x_m, p.conv_w))
    q,k,v,i_p,f_p = _mlstm_preacts(p, x_conv_a, x_m)
    h       = mlstm_cell_parallel(q, k, v, i_p, f_p)
    h_flat  = h.transpose(0,2,1,3).reshape(B, S, inner)
    h_out   = (h_flat + p.skip*x_conv_a) * jax.nn.silu(z)
    h_norm  = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T)

def mlstm_layer_parallel_with_state(p: mLSTMParams, x: jnp.ndarray, num_heads: int):
    B, S, _ = x.shape
    inner   = p.W_up.shape[0] // 2
    K       = p.conv_w.shape[1]
    xz      = jnp.dot(x, p.W_up.T)
    x_m, z  = xz[...,:inner], xz[...,inner:]
    x_conv_a= jax.nn.silu(causal_conv1d(x_m, p.conv_w))
    q,k,v,i_p,f_p = _mlstm_preacts(p, x_conv_a, x_m)
    h, (C_T,n_T,m_T) = mlstm_cell_parallel_with_state(q, k, v, i_p, f_p)
    h_flat  = h.transpose(0,2,1,3).reshape(B, S, inner)
    h_out   = (h_flat + p.skip*x_conv_a) * jax.nn.silu(z)
    h_norm  = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    conv_buf= x_m[:, -(K-1):, :]
    return jnp.dot(h_norm, p.W_down.T), (C_T, n_T, m_T, conv_buf)

def mlstm_layer_step(p: mLSTMParams, x_t: jnp.ndarray, state, num_heads: int):
    C, n, m, conv_buf = state
    B     = x_t.shape[0]
    inner = p.W_up.shape[0] // 2
    xz    = jnp.dot(x_t, p.W_up.T)
    x_m, z = xz[...,:inner], xz[...,inner:]
    x_conv, new_buf = causal_conv1d_step(x_m, conv_buf, p.conv_w)
    x_conv_a = jax.nn.silu(x_conv)
    q,k,v,i_p,f_p = _mlstm_preacts_step(p, x_conv_a, x_m)
    h, C_t, n_t, m_t = mlstm_cell_step(q, k, v, i_p, f_p, C, n, m)
    h_out = (h.reshape(B, inner) + p.skip*x_conv_a) * jax.nn.silu(z)
    h_norm= multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T), (C_t, n_t, m_t, new_buf)


# ============================================================================
# sLSTM layers
# ============================================================================

def slstm_layer(p: sLSTMParams, x: jnp.ndarray, num_heads: int):
    B = x.shape[0]
    H = p.W_i.shape[0]
    x_conv_a = jax.nn.silu(causal_conv1d(x, p.conv_w))
    gates = jnp.concatenate([
        jnp.dot(x_conv_a, p.W_i.T), jnp.dot(x_conv_a, p.W_f.T),
        jnp.dot(x,        p.W_z.T), jnp.dot(x,        p.W_o.T),
    ], axis=-1)
    init = tuple(jnp.zeros((B, H), x.dtype) for _ in range(4))
    def step(state, gate_t):
        sy, sc, sn, sm = state
        raw = gate_t + jnp.dot(sy, p.Ry.T) + p.b
        y_t, c_t, n_t, m_t = slstm_cell(raw, sc, sn, sm)
        return (y_t, c_t, n_t, m_t), y_t
    _, ys = jax.lax.scan(step, init, gates.transpose(1,0,2))
    return multihead_layer_norm(ys.transpose(1,0,2), p.gn_w, p.gn_b, num_heads)

def slstm_layer_step(p: sLSTMParams, x_t: jnp.ndarray, state, num_heads: int):
    sy, sc, sn, sm, conv_buf = state
    x_conv, new_buf = causal_conv1d_step(x_t, conv_buf, p.conv_w)
    x_conv_a = jax.nn.silu(x_conv)
    gate_t = jnp.concatenate([
        jnp.dot(x_conv_a, p.W_i.T), jnp.dot(x_conv_a, p.W_f.T),
        jnp.dot(x_t,      p.W_z.T), jnp.dot(x_t,      p.W_o.T),
    ], axis=-1)
    raw = gate_t + jnp.dot(sy, p.Ry.T) + p.b
    y_t, c_t, n_t, m_t = slstm_cell(raw, sc, sn, sm)
    return multihead_layer_norm(y_t, p.gn_w, p.gn_b, num_heads), (y_t,c_t,n_t,m_t,new_buf)


# ============================================================================
# Forward passes
# ============================================================================

def forward_train(base_xlstm, rc, tokens, num_heads, block_map):
    xlstm = apply_xlstm_hira(base_xlstm, rc)
    x     = xlstm.embedding[tokens]
    for block, btype in zip(xlstm.blocks, block_map):
        x_n = layer_norm(x, block.norm1_w, block.norm1_b)
        x   = x + (mlstm_layer_parallel(block.mlstm, x_n, num_heads) if btype=='m'
                   else slstm_layer(block.slstm, x_n, num_heads))
        x_n = layer_norm(x, block.norm2_w, block.norm2_b)
        x   = x + gated_ffn(block.ffn_up, block.ffn_down, x_n)
    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj)

def _slstm_final_state(p, x):
    B = x.shape[0]; H = p.W_i.shape[0]; K = p.conv_w.shape[1]
    x_conv_a = jax.nn.silu(causal_conv1d(x, p.conv_w))
    gates = jnp.concatenate([
        jnp.dot(x_conv_a,p.W_i.T), jnp.dot(x_conv_a,p.W_f.T),
        jnp.dot(x,p.W_z.T),        jnp.dot(x,p.W_o.T),
    ], axis=-1)
    init = tuple(jnp.zeros((B,H), x.dtype) for _ in range(4))
    def step(state, gate_t):
        sy,sc,sn,sm = state
        y_t,c_t,n_t,m_t = slstm_cell(gate_t + jnp.dot(sy,p.Ry.T) + p.b, sc,sn,sm)
        return (y_t,c_t,n_t,m_t), None
    (y_T,c_T,n_T,m_T),_ = jax.lax.scan(step, init, gates.transpose(1,0,2))
    return (y_T, c_T, n_T, m_T, x[:, -(K-1):, :])

def init_step_states(config, batch_size):
    B, K   = batch_size, config.conv_kernel
    NH, DH = config.num_heads, config.d_model // config.num_heads
    bf     = DTYPE
    states = []
    for btype in config.block_map:
        if btype == 'm':
            inner = config.d_model
            states.append((
                jnp.zeros((B,NH,DH,DH), bf),
                jnp.zeros((B,NH,DH),    bf),
                jnp.zeros((B,NH),       bf),
                jnp.zeros((B,K-1,inner),bf),
            ))
        else:
            H = config.d_model
            states.append((
                jnp.zeros((B,H),              bf),
                jnp.zeros((B,H),              bf),
                jnp.zeros((B,H),              bf),
                jnp.zeros((B,H),              bf),
                jnp.zeros((B,K-1,config.d_model),bf),
            ))
    return states

def forward_step(base_xlstm, rc, token, states, num_heads, block_map):
    xlstm     = apply_xlstm_hira(base_xlstm, rc)
    x         = xlstm.embedding[token]
    new_states= []
    for block, btype, state in zip(xlstm.blocks, block_map, states):
        x_n = layer_norm(x, block.norm1_w, block.norm1_b)
        if btype == 'm':
            cell_out, ns = mlstm_layer_step(block.mlstm, x_n, state, num_heads)
        else:
            cell_out, ns = slstm_layer_step(block.slstm, x_n, state, num_heads)
        new_states.append(ns)
        x   = x + cell_out
        x_n = layer_norm(x, block.norm2_w, block.norm2_b)
        x   = x + gated_ffn(block.ffn_up, block.ffn_down, x_n)
    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj), new_states

_forward_step_jit = jax.jit(forward_step, static_argnames=["num_heads","block_map"])


# ============================================================================
# Structured init helpers  (numpy, run once at startup)
# ============================================================================

def _bf(x): return jnp.array(x, dtype=DTYPE)

def _ortho(key, n, m):
    """[n, m] matrix with orthonormal rows (n ≤ m gives exact orthonormality)."""
    raw = np.array(jr.normal(key, (max(n,m), min(n,m))))
    Q, _ = np.linalg.qr(raw)           # Q: [max,min], orthonormal columns
    mat  = Q.T                          # [min,max], orthonormal rows
    if n > m:
        extra = np.array(jr.normal(jr.fold_in(key,1),(n-m,m))) / math.sqrt(m)
        mat   = np.concatenate([mat, extra], axis=0)
    return _bf(mat[:n])

def _fir_bank(n_channels, K):
    """[n_channels, K] causal FIR filter bank (DCT-II basis tiled across channels).
    K=4 gives: DC average, fundamental, 2nd, 3rd harmonic — orthonormal filters.
    """
    basis = np.zeros((K, K))
    for k in range(K):
        for n in range(K):
            basis[k, n] = np.cos(np.pi/K * (n + 0.5) * k)
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    basis /= norms
    bank = np.array([basis[c % K] for c in range(n_channels)])
    return _bf(bank)

def _multiscale_forget_bias(n, seq_len):
    """[n] forget gate biases with log-uniform time constants in [1, seq_len].
    τ_h = exp(h/(n-1) * log(seq_len)),  b_h = logit(exp(-1/τ_h)).
    Head 0 → τ≈1 (fast, 1-step memory).  Head n-1 → τ=seq_len (full-window).
    """
    tau = np.exp(np.linspace(0.0, np.log(max(float(seq_len), 2.0)), n))
    f   = np.exp(-1.0 / tau)
    return _bf(np.log(f / (1.0 - f + 1e-9)))   # logit = inverse sigmoid


# ============================================================================
# Frozen xLSTM initialisation  (structured basis)
# ============================================================================

def init_mlstm_params(key, d_model, num_heads, conv_kernel, seq_len) -> mLSTMParams:
    inner = d_model
    DH    = inner // num_heads
    ks    = jr.split(key, 10)

    # W_up [2*inner, d_model]: row-normalised for flat output gain
    W_up_raw = np.array(jr.normal(ks[0], (2*inner, d_model)))
    W_up_raw /= np.linalg.norm(W_up_raw, axis=1, keepdims=True)
    W_up = _bf(W_up_raw / math.sqrt(d_model))

    # W_q, W_k, W_v: orthogonal across the full [NH*DH, inner] space →
    # different heads attend to strictly orthogonal subspaces
    def _qkv(k):
        M = np.array(_ortho(k, inner, inner))     # [inner, inner] orthogonal
        return _bf(M.reshape(num_heads, DH, inner) / math.sqrt(inner))
    W_q = _qkv(ks[1])
    W_k = _qkv(ks[2])
    W_v = _qkv(ks[3])

    # W_i: random with per-head scale spread (some heads reactive, some conservative)
    i_scale = np.linspace(0.5, 2.0, num_heads)[:, None]
    W_i = _bf(np.array(jr.normal(ks[4], (num_heads, inner))) / math.sqrt(inner) * i_scale)

    # W_f: random direction component; actual timescale set by b_f bias
    W_f = _bf(np.array(jr.normal(ks[5], (num_heads, inner))) / math.sqrt(inner))

    # b_f: per-head forget bias spanning τ=1..seq_len (log-uniform)
    b_f = _multiscale_forget_bias(num_heads, seq_len)
    b_i = jnp.zeros((num_heads,), DTYPE)

    # W_down [d_model, inner]: orthonormal rows for diverse readout directions
    W_down = _ortho(ks[6], d_model, inner) / math.sqrt(inner)

    # conv_w [inner, K]: structured FIR filter bank (DCT-II basis)
    conv_w = _fir_bank(inner, conv_kernel)

    # FFN-equivalent skip and layernorm
    skip = jnp.ones((inner,), DTYPE)
    ln_w = jnp.ones((inner,), DTYPE)
    ln_b = jnp.zeros((inner,), DTYPE)

    return mLSTMParams(
        W_up=W_up, W_q=W_q, W_k=W_k, W_v=W_v,
        W_i=W_i, W_f=W_f, b_f=b_f, b_i=b_i,
        skip=skip, ln_w=ln_w, ln_b=ln_b,
        W_down=W_down, conv_w=conv_w,
    )

def init_slstm_params(key, d_model, conv_kernel, seq_len) -> sLSTMParams:
    H  = d_model
    ks = jr.split(key, 7)

    # Input projections: orthonormal rows so all input directions equally represented
    W_i = _ortho(ks[0], H, d_model) / math.sqrt(d_model)
    W_f = _ortho(ks[1], H, d_model) / math.sqrt(d_model)
    W_z = _ortho(ks[2], H, d_model) / math.sqrt(d_model)
    W_o = _ortho(ks[3], H, d_model) / math.sqrt(d_model)

    # Ry [4H, H]: near-critical via SVD (spectral radius ≈ 0.95, ESN principle)
    # All singular values = 0.95 → max memory capacity, guaranteed stability
    raw_R = np.array(jr.normal(ks[4], (4*H, H)))
    U, _, Vt = np.linalg.svd(raw_R, full_matrices=False)  # U:[4H,H], Vt:[H,H]
    Ry = _bf(0.95 * (U @ Vt))

    # b [4H]: forget gate slice b[H:2H] gets multi-timescale init,
    # input/z/o slices stay zero
    f_bias = np.array(_multiscale_forget_bias(H, seq_len))
    b_arr  = np.zeros(4*H)
    b_arr[H:2*H] = f_bias
    b = _bf(b_arr)

    gn_w   = jnp.ones((H,),  DTYPE)
    gn_b   = jnp.zeros((H,), DTYPE)
    conv_w = _fir_bank(d_model, conv_kernel)

    return sLSTMParams(W_i=W_i, W_f=W_f, W_z=W_z, W_o=W_o,
                       Ry=Ry, b=b, gn_w=gn_w, gn_b=gn_b, conv_w=conv_w)

def init_xlstm_block_params(key, d_model, num_heads, d_ff, conv_kernel,
                             seq_len, btype) -> xLSTMBlockParams:
    ks = jr.split(key, 4)
    # FFN: row-normalised projections
    ffn_up_raw = np.array(jr.normal(ks[2], (2*d_ff, d_model)))
    ffn_up_raw /= np.linalg.norm(ffn_up_raw, axis=1, keepdims=True)
    ffn_up   = _bf(ffn_up_raw / math.sqrt(d_model))
    ffn_down = _ortho(ks[3], d_model, d_ff) / math.sqrt(d_ff)
    return xLSTMBlockParams(
        norm1_w=jnp.ones((d_model,),DTYPE), norm1_b=jnp.zeros((d_model,),DTYPE),
        norm2_w=jnp.ones((d_model,),DTYPE), norm2_b=jnp.zeros((d_model,),DTYPE),
        ffn_up=ffn_up, ffn_down=ffn_down,
        mlstm=init_mlstm_params(ks[0], d_model, num_heads, conv_kernel, seq_len)
              if btype=='m' else None,
        slstm=init_slstm_params(ks[1], d_model, conv_kernel, seq_len)
              if btype=='s' else None,
    )

def init_xlstm_params(key, config) -> xLSTMParams:
    ks  = jr.split(key, 2 + config.num_layers)
    emb = _bf(np.array(jr.normal(ks[0], (config.vocab_size, config.d_model))) * 0.01)
    blocks = [
        init_xlstm_block_params(
            ks[2+i], config.d_model, config.num_heads, config.d_ff,
            config.conv_kernel, config.max_seq_len, config.block_map[i]
        )
        for i in range(config.num_layers)
    ]
    return xLSTMParams(
        embedding=emb, blocks=blocks,
        norm_w=jnp.ones((config.d_model,),DTYPE),
        norm_b=jnp.zeros((config.d_model,),DTYPE),
    )


# ============================================================================
# HiRA initialisation  (orthonormal A rows via SVD)
# ============================================================================

def init_hira_adapter(key, d_out, d_in, r) -> HiRAAdapter:
    """A: orthonormal rows (right singular vectors of random matrix).
    Each HiRA direction is independent → B receives diverse gradient signals.
    B=0 ensures ΔW=W₀⊙(B·A)=0 at init.
    """
    raw = np.array(jr.normal(key, (r, d_in)))
    if r <= d_in:
        _, _, Vt = np.linalg.svd(raw, full_matrices=False)  # Vt: [r, d_in]
        A = _bf(Vt / math.sqrt(d_in))
    else:
        A = _bf(raw / math.sqrt(d_in))
    return HiRAAdapter(A=A, B=jnp.zeros((d_out, r), DTYPE))

def init_block_hira(key, config, btype) -> BlockHiRA:
    d, d_ff, r = config.d_model, config.d_ff, config.lora_r
    inner, NH  = d, config.num_heads
    ks = jr.split(key, 12); ki = 0
    if btype == 'm':
        mhira = mLSTMHiRA(
            W_up  =init_hira_adapter(ks[ki],   2*inner, d,     r),
            W_q   =init_hira_adapter(ks[ki+1], inner,   inner, r),
            W_k   =init_hira_adapter(ks[ki+2], inner,   inner, r),
            W_v   =init_hira_adapter(ks[ki+3], inner,   inner, r),
            W_i   =init_hira_adapter(ks[ki+4], NH,      inner, r),
            W_f   =init_hira_adapter(ks[ki+5], NH,      inner, r),
            W_down=init_hira_adapter(ks[ki+6], d,       inner, r),
        ); ki += 7; shira = None
    else:
        shira = sLSTMHiRA(
            W_i=init_hira_adapter(ks[ki],   d, d, r),
            W_f=init_hira_adapter(ks[ki+1], d, d, r),
            W_z=init_hira_adapter(ks[ki+2], d, d, r),
            W_o=init_hira_adapter(ks[ki+3], d, d, r),
        ); ki += 4; mhira = None
    return BlockHiRA(
        ffn_up  =init_hira_adapter(ks[ki],   2*d_ff, d,    r),
        ffn_down=init_hira_adapter(ks[ki+1], d,      d_ff, r),
        mlstm=mhira, slstm=shira,
    )

def init_randcompress_params(key, config) -> RandCompressParams:
    ks = jr.split(key, 2 + config.num_layers)
    return RandCompressParams(
        output_proj=_bf(np.array(jr.normal(ks[0],(config.d_model,config.vocab_size)))*0.01),
        emb_hira   =init_hira_adapter(ks[1], config.vocab_size, config.d_model, config.lora_r),
        blocks     =[init_block_hira(ks[2+i], config, config.block_map[i])
                     for i in range(config.num_layers)],
    )


# ============================================================================
# AdamW
# ============================================================================

class AdamWState(NamedTuple):
    m: any; v: any
    beta1_power: jnp.ndarray
    beta2_power: jnp.ndarray

def adamw_init(params) -> AdamWState:
    z = lambda p: jnp.zeros_like(p, jnp.float32)
    return AdamWState(
        m=jax.tree_util.tree_map(z, params),
        v=jax.tree_util.tree_map(z, params),
        beta1_power=jnp.array(0.9,   jnp.float32),
        beta2_power=jnp.array(0.999, jnp.float32),
    )

def adamw_update(state, grads, params, lr, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0, max_norm=1.0):
    leaves = jax.tree_util.tree_leaves(grads)
    gnorm  = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32)**2) for g in leaves))
    scale  = jnp.minimum(1.0, max_norm / (gnorm + 1e-6))
    grads  = jax.tree_util.tree_map(lambda g: g*scale, grads)
    m, v, b1p, b2p = state
    b1p = b1p * beta1;  b2p = b2p * beta2
    g32 = jax.tree_util.tree_map(lambda g: g.astype(jnp.float32), grads)
    m   = jax.tree_util.tree_map(lambda m_,g: beta1*m_ + (1-beta1)*g,   m, g32)
    v   = jax.tree_util.tree_map(lambda v_,g: beta2*v_ + (1-beta2)*g*g, v, g32)
    bc1, bc2 = 1.0-b1p, 1.0-b2p
    new_p = jax.tree_util.tree_map(
        lambda p,m_,v_: (
            p.astype(jnp.float32)
            - lr*(m_/bc1/(jnp.sqrt(v_/bc2)+eps) + weight_decay*p.astype(jnp.float32))
        ).astype(p.dtype),
        params, m, v,
    )
    return new_p, AdamWState(m=m, v=v, beta1_power=b1p, beta2_power=b2p)


# ============================================================================
# Loss and LR
# ============================================================================

def cross_entropy_loss(logits, targets):
    log_probs   = jax.nn.log_softmax(logits, axis=-1)
    loss_per_tok= -jnp.take_along_axis(log_probs, targets[...,None], axis=-1)[...,0]
    mask        = (targets != PAD_TOKEN).astype(jnp.float32)
    return (loss_per_tok*mask).sum() / (mask.sum()+1e-8)

def cosine_lr(step, total_steps, lr_max, lr_min=1e-4, warmup=100):
    warmup_lr = lr_max * jnp.minimum(step/warmup, 1.0)
    t         = jnp.maximum(step-warmup,0) / jnp.maximum(total_steps-warmup,1)
    cos_lr    = lr_min + 0.5*(lr_max-lr_min)*(1.0+jnp.cos(math.pi*t))
    return jnp.where(step < warmup, warmup_lr, cos_lr)


# ============================================================================
# Persistent TBPTT helpers
# ============================================================================

def causal_conv1d_with_buf(x, w, buf):
    C, K = w.shape
    w    = w.astype(x.dtype)
    x_pad= jnp.concatenate([buf, x], axis=1)
    x_t  = x_pad.transpose(0,2,1)
    w_t  = w[:,None,:]
    out  = jax.lax.conv_general_dilated(
        x_t, w_t, window_strides=(1,), padding='VALID',
        feature_group_count=C, dimension_numbers=('NCH','OIH','NCH'),
    )
    return out.transpose(0,2,1)

def mlstm_cell_parallel_with_init_state(q, k, v, i_preact, f_preact, C0, n0, m0):
    B, NH, S, DH = q.shape
    log_f  = jax.nn.log_sigmoid(f_preact)
    lf_cs  = jnp.concatenate([
        jnp.zeros((B,NH,1,1), log_f.dtype), jnp.cumsum(log_f, axis=2)
    ], axis=2)
    rep    = lf_cs[...,0]
    log_fg = rep[:,:,1:,None] - rep[:,:,None,1:]
    log_fg = jnp.where(jnp.tril(jnp.ones((S,S),bool)), log_fg, -jnp.inf)
    log_D_seq  = log_fg + i_preact.transpose(0,1,3,2)
    log_w_init = m0[:,:,None] + lf_cs[:,:,1:,0]
    max_log_D  = jnp.maximum(log_D_seq.max(axis=-1), log_w_init)[:,:,:,None]
    D_seq  = jnp.exp(log_D_seq - max_log_D)
    d_init = jnp.exp(log_w_init - max_log_D[:,:,:,0])
    k_s    = k / jnp.sqrt(jnp.array(DH, k.dtype))
    qk     = jnp.einsum('bnsd,bntd->bnst', q, k_s)
    h_num  = (jnp.einsum('bnst,bntd->bnsd', qk*D_seq, v)
              + jnp.einsum('bnsd,bnde->bnse', q, C0)*d_init[:,:,:,None])
    n_full = (jnp.einsum('bnst,bntd->bnsd', D_seq, k_s)
              + n0[:,:,None,:]*d_init[:,:,:,None])
    qn     = (q*n_full).sum(axis=-1)
    denom  = jnp.maximum(jnp.abs(qn), jnp.exp(-max_log_D[:,:,:,0]))
    h      = h_num / (denom[:,:,:,None]+1e-8)
    w_T    = D_seq[:,:,-1,:]
    d_T    = d_init[:,:,-1]
    C_T    = jnp.einsum('bns,bnsd,bnse->bnde', w_T, k_s, v) + d_T[:,:,None,None]*C0
    n_T    = n_full[:,:,-1,:]
    m_T    = max_log_D[:,:,-1,0]
    return h, (C_T, n_T, m_T)

def mlstm_layer_parallel_with_init(p, x, num_heads, state):
    C0, n0, m0, conv_buf = state
    B, S, _ = x.shape
    inner   = p.W_up.shape[0]//2
    K       = p.conv_w.shape[1]
    xz      = jnp.dot(x, p.W_up.T)
    x_m, z  = xz[...,:inner], xz[...,inner:]
    x_conv_a= jax.nn.silu(causal_conv1d_with_buf(x_m, p.conv_w, conv_buf))
    q,k,v,i_p,f_p = _mlstm_preacts(p, x_conv_a, x_m)
    h, (C_T,n_T,m_T) = mlstm_cell_parallel_with_init_state(q,k,v,i_p,f_p,C0,n0,m0)
    h_flat  = h.transpose(0,2,1,3).reshape(B,S,inner)
    h_out   = (h_flat + p.skip*x_conv_a)*jax.nn.silu(z)
    h_norm  = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T), (C_T, n_T, m_T, x_m[:,-(K-1):,:])

def slstm_layer_with_init(p, x, num_heads, state):
    y0, c0, n0, m0, conv_buf = state
    K        = p.conv_w.shape[1]
    x_conv_a = jax.nn.silu(causal_conv1d_with_buf(x, p.conv_w, conv_buf))
    gates    = jnp.concatenate([
        jnp.dot(x_conv_a,p.W_i.T), jnp.dot(x_conv_a,p.W_f.T),
        jnp.dot(x,p.W_z.T),        jnp.dot(x,p.W_o.T),
    ], axis=-1)
    def step(state, gate_t):
        sy,sc,sn,sm = state
        y_t,c_t,n_t,m_t = slstm_cell(gate_t+jnp.dot(sy,p.Ry.T)+p.b, sc,sn,sm)
        return (y_t,c_t,n_t,m_t), y_t
    (y_T,c_T,n_T,m_T),ys = jax.lax.scan(step,(y0,c0,n0,m0),gates.transpose(1,0,2))
    out = multihead_layer_norm(ys.transpose(1,0,2), p.gn_w, p.gn_b, num_heads)
    return out, (y_T, c_T, n_T, m_T, x[:,-(K-1):,:])

def forward_train_chunked(base_xlstm, rc, inputs, layer_states, num_heads, block_map):
    xlstm      = apply_xlstm_hira(base_xlstm, rc)
    x          = xlstm.embedding[inputs]
    new_states = []
    for block, btype, state in zip(xlstm.blocks, block_map, layer_states):
        x_n = layer_norm(x, block.norm1_w, block.norm1_b)
        if btype == 'm':
            cell_out, ns = mlstm_layer_parallel_with_init(block.mlstm, x_n, num_heads, state)
        else:
            cell_out, ns = slstm_layer_with_init(block.slstm, x_n, num_heads, state)
        new_states.append(ns)
        x   = x + cell_out
        x_n = layer_norm(x, block.norm2_w, block.norm2_b)
        x   = x + gated_ffn(block.ffn_up, block.ffn_down, x_n)
    x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
    return jnp.dot(x, rc.output_proj), new_states


# ============================================================================
# Gradient accumulation train step
# ============================================================================

@jax.jit(static_argnames=["config"])
def grad_chunk_persistent(base_xlstm, params, chunk_batch, layer_states, config):
    """Loss + gradients for one chunk, no param update (used for grad accumulation)."""
    inputs, targets = chunk_batch[:,:-1], chunk_batch[:,1:]
    def loss_fn(p):
        logits, new_states = forward_train_chunked(
            base_xlstm, p, inputs, layer_states, config.num_heads, config.block_map)
        return cross_entropy_loss(logits, targets), new_states
    (loss, new_states), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, grads, new_states


@jax.jit(static_argnames=["num_heads", "block_map"])
def _advance_states(base_xlstm, params, chunk_batch, layer_states, num_heads, block_map):
    """Forward-only pass to advance recurrent states. No loss, no grad.
    Used for baked chunks that still need state continuity.
    """
    _, new_states = forward_train_chunked(
        base_xlstm, params, chunk_batch[:,:-1], layer_states, num_heads, block_map)
    return new_states


# ============================================================================
# Curriculum schedule
# ============================================================================

def make_epoch_schedule(epoch, num_chunks, config):
    """Return list of (chunk_idx, pass_number) for this epoch.

    curriculum=False: single sequential pass — [(0,0),(1,0),...,(N-1,0)].

    curriculum=True:
      window = min(start + step*epoch, num_chunks)
      Repeat the window `curriculum_repeat` times.
      pass_number tracks which repetition so the training loop can identify
      repeated passes and skip baked chunks on them.

    State flows continuously through the entire schedule (no intra-epoch reset).
    """
    if not config.curriculum:
        return [(i, 0) for i in range(num_chunks)]

    window = min(
        config.curriculum_start_chunks + config.curriculum_step_chunks * epoch,
        num_chunks,
    )
    schedule = []
    for pass_idx in range(config.curriculum_repeat):
        for ci in range(window):
            schedule.append((ci, pass_idx))
    return schedule


# ============================================================================
# Data
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


# ============================================================================
# Evaluation helpers
# ============================================================================

def _eval_generative(base_xlstm, params, config, seed_bytes, target_np):
    """Greedy generation from seed_bytes (hint prefix); compare to target_np.

    The seed bytes are fed through the model to warm up the hidden state.
    The first predicted token after the last seed byte is compared to target_np[0].
    Returns (generated, bytes_wrong, accuracy, cer, first_wrong_idx).
    """
    n      = len(target_np)
    states = init_step_states(config, batch_size=1)

    # warm up state on all seed bytes; keep the last predicted token to start gen
    cur_token = jnp.array([seed_bytes[0]])
    for sb in seed_bytes[1:]:
        _, states = _forward_step_jit(
            base_xlstm, params, cur_token, states, config.num_heads, config.block_map)
        cur_token = jnp.array([sb])

    gen_result = []
    for _ in tqdm(range(n), desc="  gen", unit="tok", leave=False):
        logit, states = _forward_step_jit(
            base_xlstm, params, cur_token, states, config.num_heads, config.block_map)
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


# ============================================================================
# Main
# ============================================================================

def main():
    train_log_path = "train_v2_persistent.log"
    eval_log_path  = "eval_v2_final.log"
    train_log_file = open(train_log_path, "w", buffering=1)
    sys.stdout     = _Tee(sys.__stdout__, train_log_file)

    dataset_path = "datasets/quran-uthmani.txt"
    val_path     = "datasets/surat_al-fatihah.txt"
    tokens     = load_dataset(dataset_path)
    val_tokens = load_dataset(val_path)
    n_real = len(tokens)
    n_val  = len(val_tokens)
    chunk_size = 1024
    print(f"Train: {n_real} bytes   Val: {n_val} bytes   chunk_size={chunk_size}")

    config = Config(
        vocab_size=256, d_model=128, num_heads=8, d_ff=256,
        num_layers=4, max_seq_len=chunk_size, seed=0, conv_kernel=4,
        block_map="msms", lora_r=16, batch_size=1,
        learning_rate=1e-3, weight_decay=0.0, grad_clip_norm=10.0,
        grad_accum_k=4,
        curriculum=True,
        curriculum_start_chunks=4,
        curriculum_step_chunks=4,
        curriculum_repeat=4,
        curriculum_baked_threshold=0.3,
    )

    key = jr.key(config.seed)
    key, k_xlstm, k_rc = jr.split(key, 3)

    print("Initialising frozen xLSTM (structured basis)...")
    base_xlstm = init_xlstm_params(k_xlstm, config)
    frozen     = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(base_xlstm))

    print("Initialising HiRA adapters (orthonormal A)...")
    params     = init_randcompress_params(k_rc, config)
    trainable  = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    total      = frozen + trainable

    print(f"Params  — frozen: {frozen/1e6:.3f}M  trainable: {trainable/1e6:.3f}M  "
          f"total: {total/1e6:.3f}M")
    param_bytes    = trainable * np.dtype(DTYPE).itemsize
    compress_ratio = n_real / param_bytes
    print(f"Ratios  — trainable/total: {trainable/total*100:.1f}%  "
          f"trainable/dataset: {trainable/n_real:.2f} params/byte  "
          f"({n_real/1e6:.3f}M bytes dataset)")
    print(f"Compression — {n_real/1e6:.3f}MB / {param_bytes/1e6:.3f}MB "
          f"= {compress_ratio:.2f}x  "
          f"({'compressed' if compress_ratio>1 else 'over-parameterised'})")

    opt_state = adamw_init(params)

    usable     = (n_real-1)//chunk_size*chunk_size + 1
    raw        = tokens[:usable]
    num_chunks = (len(raw)-1)//chunk_size
    chunks     = [raw[i*chunk_size : i*chunk_size+chunk_size+1] for i in range(num_chunks)]

    # Validation tensors (al-Fatihah fits in one forward pass)
    val_inputs_jnp  = jnp.array(val_tokens[:-1][None,:], dtype=jnp.int32)
    val_targets_jnp = jnp.array(val_tokens[1:][None,:],  dtype=jnp.int32)
    val_target_np   = np.array(val_tokens[1:], dtype=np.uint8)
    n_val_pred      = n_val - 1

    num_epochs  = 20
    K           = config.grad_accum_k

    # Precompute total steps for cosine LR (accounts for varying epoch lengths)
    total_steps = sum(
        len(make_epoch_schedule(e, num_chunks, config))
        for e in range(num_epochs)
    )

    # Per-chunk EMA loss for baked-in detection (inf = never seen)
    chunk_losses = np.full(num_chunks, np.inf)

    global_step = 0
    t_total     = 0.0

    curric_str = (f"curriculum  start={config.curriculum_start_chunks}  "
                  f"step={config.curriculum_step_chunks}  "
                  f"repeat={config.curriculum_repeat}  "
                  f"baked_thr={config.curriculum_baked_threshold}"
                  if config.curriculum else "curriculum=off")
    print(f"\nPersistent TBPTT  epochs={num_epochs}  grad_accum_k={K}  {curric_str}")
    print(f"Total chunks: {num_chunks}   Total steps (scheduled): {total_steps}")
    print("-" * 60)

    for epoch in range(num_epochs):
        schedule = make_epoch_schedule(epoch, num_chunks, config)
        window   = schedule[-1][0] + 1 if schedule else 0   # highest chunk in this epoch

        states     = init_step_states(config, batch_size=1)
        epoch_loss = 0.0
        n_trained  = 0   # chunks that actually ran backprop this epoch
        n_baked    = 0   # chunks skipped due to baked-in threshold
        recent     = []

        pbar = tqdm(total=len(schedule), desc=f"E{epoch+1:04d}", unit="chunk",
                    dynamic_ncols=True, leave=True)

        for block_start in range(0, len(schedule), K):
            block     = schedule[block_start : block_start + K]
            accum_grads = None
            block_loss  = 0.0
            n_block_trained = 0
            t0 = time.perf_counter()

            for (ci, pass_idx) in block:
                lr            = cosine_lr(global_step, total_steps, config.learning_rate)
                chunk_batch   = jnp.array(chunks[ci][None,:], dtype=jnp.int32)
                frozen_states = jax.lax.stop_gradient(states)

                # Baked: chunk is memorised and this is a repetition pass — advance
                # state without backprop so sequential context is preserved.
                is_baked = (
                    config.curriculum_baked_threshold > 0
                    and np.isfinite(chunk_losses[ci])
                    and chunk_losses[ci] < config.curriculum_baked_threshold
                    and pass_idx > 0
                )

                if is_baked:
                    states = _advance_states(
                        base_xlstm, params, chunk_batch, frozen_states,
                        config.num_heads, config.block_map)
                    n_baked    += 1
                    global_step += 1
                    pbar.update(1)
                    continue

                loss, grads, states = grad_chunk_persistent(
                    base_xlstm, params, chunk_batch, frozen_states, config)
                loss_f = float(loss)

                # EMA update for baked detection
                chunk_losses[ci] = (0.9 * chunk_losses[ci] + 0.1 * loss_f
                                    if np.isfinite(chunk_losses[ci]) else loss_f)

                block_loss      += loss_f
                n_block_trained += 1
                n_trained       += 1
                global_step     += 1

                accum_grads = (grads if accum_grads is None
                               else jax.tree_util.tree_map(jnp.add, accum_grads, grads))
                pbar.update(1)

            # Only update if at least one chunk in this block had gradients
            if accum_grads is not None:
                if n_block_trained > 1:
                    accum_grads = jax.tree_util.tree_map(
                        lambda g: g / n_block_trained, accum_grads)
                params, opt_state = adamw_update(
                    opt_state, accum_grads, params,
                    lr=lr, weight_decay=config.weight_decay,
                    max_norm=config.grad_clip_norm)

            t_total += (time.perf_counter() - t0) * 1000

            if n_block_trained > 0:
                avg_block_loss = block_loss / n_block_trained
                epoch_loss    += block_loss
                recent.append(avg_block_loss)
                if len(recent) > 50: recent.pop(0)
                bpc     = math.exp(avg_block_loss) / math.log(2)
                avg_bpc = math.exp(sum(recent)/len(recent)) / math.log(2)
                pbar.set_postfix(loss=f"{avg_block_loss:.4f}", bpc=f"{bpc:.3f}",
                                 avg_bpc=f"{avg_bpc:.3f}",
                                 baked=n_baked,
                                 ms=f"{t_total/max(global_step,1):.1f}")

        pbar.close()
        avg_loss = epoch_loss / max(n_trained, 1)
        bpc      = math.exp(avg_loss) / math.log(2)
        avg_ms   = t_total / max(global_step, 1)
        print(f"  epoch {epoch+1:4d}  loss={avg_loss:.5f}  bpc={bpc:.4f}"
              f"  lr={float(lr):.2e}  window={window}/{num_chunks}"
              f"  trained={n_trained}  baked={n_baked}  avg_ms/chunk={avg_ms:.1f}")

        # ---- validation: surat al-Fatihah ----
        val_logits = forward_train(base_xlstm, params, val_inputs_jnp,
                                   config.num_heads, config.block_map)
        val_loss   = float(cross_entropy_loss(val_logits, val_targets_jnp))
        val_bpc    = math.exp(val_loss) / math.log(2)

        val_gen, val_wrong, val_acc, _, val_first = _eval_generative(
            base_xlstm, params, config,
            seed_bytes=[int(val_tokens[0])], target_np=val_target_np)
        val_first_str = f"byte {val_first}" if val_first is not None else "none (perfect)"
        print(f"  [VAL] bpc={val_bpc:.4f}  acc={val_acc*100:.2f}%  "
              f"wrong={val_wrong}/{n_val_pred}  first_wrong={val_first_str}")
        print(f"  [GEN] {ByteTokenizer.decode(val_gen)}")

        if avg_loss < 0.01:
            print(f"\n  Converged at epoch {epoch+1}")
            break

    # ---- final evaluation on full train set ----
    print("\n" + "="*60)
    print("Final evaluation on training dataset...")
    print("="*60)

    eval_log_file = open(eval_log_path, "w", buffering=1)
    sys.stdout    = _Tee(sys.__stdout__, train_log_file, eval_log_file)

    print("Computing train BPC (teacher-forced, chunked)...")
    eval_states     = init_step_states(config, batch_size=1)
    total_eval_loss = 0.0
    for chunk_np in tqdm(chunks, desc="  tf-bpc", unit="chunk"):
        chunk_batch   = jnp.array(chunk_np[None,:], dtype=jnp.int32)
        frozen_states = jax.lax.stop_gradient(eval_states)
        logits_c, eval_states = forward_train_chunked(
            base_xlstm, params, chunk_batch[:,:-1], frozen_states,
            config.num_heads, config.block_map)
        total_eval_loss += float(cross_entropy_loss(logits_c, chunk_batch[:,1:]))
    train_bpc = math.exp(total_eval_loss/num_chunks) / math.log(2)
    print(f"Train BPC (teacher-forced): {train_bpc:.4f}")

    n_train_pred = n_real - 1
    train_target = np.array(tokens[1:], dtype=np.uint8)
    print(f"Generating {n_train_pred} bytes from first train byte "
          f"(0x{int(tokens[0]):02x})...")
    train_gen, train_wrong, train_acc, train_cer, train_first = _eval_generative(
        base_xlstm, params, config,
        seed_bytes=[int(tokens[0])], target_np=train_target)
    train_first_str = f"byte {train_first}" if train_first is not None else "none (perfect)"

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
