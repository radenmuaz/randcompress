"""
randcompress v5 — greedy sequential overfit

Algorithm
---------
Split dataset into fixed-size segments. For each new segment k:
  SOLO     phase — train on segment k alone until 100% TF-accuracy (or FAIL).
  COMBINED phase — train on segments 0..k together until 100% on all (or FAIL).

Per-segment TF-accuracy and first-fail-byte are reported before/after each phase
to track forgetting. Checkpoint saved after every successful phase.
MAX_ITER_PER_PHASE is the fixed budget; exceeding it declares FAIL and stops.
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
from tqdm import tqdm

PAD_TOKEN = 0
DTYPE     = jnp.float32

SEGMENT_SIZE      = 128   # bytes per segment (= chunk_size → 1 chunk per segment)
MAX_ITER_PER_PHASE = 5000
CHECK_EVERY       = 100   # accuracy eval interval

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
    d_model:       int   = 64
    num_heads:     int   = 4
    d_ff:          int   = 128
    num_layers:    int   = 4
    max_seq_len:   int   = 128
    seed:          int   = 0
    conv_kernel:   int   = 4
    block_map:     str   = "smsm"
    lora_r:        int   = 16
    batch_size:    int   = 1
    learning_rate: float = 5e-3
    weight_decay:  float = 0.0
    grad_clip_norm:float = 100.0
    random_shuffle:bool  = False


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

def gated_ffn(ffn_up, ffn_down, x):
    h    = jnp.dot(x, ffn_up.T)
    d_ff = h.shape[-1] // 2
    return jnp.dot(jax.nn.gelu(h[...,:d_ff]) * h[...,d_ff:], ffn_down.T)


# ============================================================================
# mLSTM cell
# ============================================================================

def mlstm_cell_parallel(q, k, v, i_preact, f_preact):
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
    return h

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
    W_up:  jnp.ndarray
    W_q:   jnp.ndarray
    W_k:   jnp.ndarray
    W_v:   jnp.ndarray
    W_i:   jnp.ndarray
    W_f:   jnp.ndarray
    b_f:   jnp.ndarray
    b_i:   jnp.ndarray
    skip:  jnp.ndarray
    ln_w:  jnp.ndarray
    ln_b:  jnp.ndarray
    W_down:jnp.ndarray
    conv_w:jnp.ndarray

class sLSTMParams(NamedTuple):
    W_i:   jnp.ndarray
    W_f:   jnp.ndarray
    W_z:   jnp.ndarray
    W_o:   jnp.ndarray
    Ry:    jnp.ndarray
    b:     jnp.ndarray
    gn_w:  jnp.ndarray
    gn_b:  jnp.ndarray
    conv_w:jnp.ndarray

class xLSTMBlockParams(NamedTuple):
    norm1_w:  jnp.ndarray
    norm1_b:  jnp.ndarray
    norm2_w:  jnp.ndarray
    norm2_b:  jnp.ndarray
    ffn_up:   jnp.ndarray
    ffn_down: jnp.ndarray
    mlstm: Optional[mLSTMParams] = None
    slstm: Optional[sLSTMParams] = None

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
    output_proj: jnp.ndarray
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
        b_f   =p.b_f, b_i=p.b_i,
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
# mLSTM layers
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

def mlstm_layer_parallel_with_init(p, x, num_heads, state):
    C0, n0, m0, conv_buf = state
    B, S, _ = x.shape
    inner   = p.W_up.shape[0] // 2
    K       = p.conv_w.shape[1]
    xz      = jnp.dot(x, p.W_up.T)
    x_m, z  = xz[...,:inner], xz[...,inner:]
    x_conv_a= jax.nn.silu(causal_conv1d_with_buf(x_m, p.conv_w, conv_buf))
    q,k,v,i_p,f_p = _mlstm_preacts(p, x_conv_a, x_m)
    NH, DH = C0.shape[1], C0.shape[2]
    log_f  = jax.nn.log_sigmoid(f_p)
    lf_cs  = jnp.concatenate([jnp.zeros((B,NH,1,1),log_f.dtype),
                               jnp.cumsum(log_f, axis=2)], axis=2)
    rep    = lf_cs[...,0]
    log_fg = rep[:,:,1:,None] - rep[:,:,None,1:]
    log_fg = jnp.where(jnp.tril(jnp.ones((S,S),bool)), log_fg, -jnp.inf)
    log_D_seq  = log_fg + i_p.transpose(0,1,3,2)
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
    w_T  = D_seq[:,:,-1,:]
    d_T  = d_init[:,:,-1]
    C_T  = jnp.einsum('bns,bnsd,bnse->bnde', w_T, k_s, v) + d_T[:,:,None,None]*C0
    n_T  = n_full[:,:,-1,:]
    m_T  = max_log_D[:,:,-1,0]
    h_flat = h.transpose(0,2,1,3).reshape(B,S,inner)
    h_out  = (h_flat + p.skip*x_conv_a)*jax.nn.silu(z)
    h_norm = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T), (C_T, n_T, m_T, x_m[:,-(K-1):,:])


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

_fwd_step_jit    = jax.jit(forward_step,         static_argnames=["num_heads","block_map"])
_fwd_chunked_jit = jax.jit(forward_train_chunked, static_argnames=["num_heads","block_map"])


# ============================================================================
# Exact BPTT step (scan + remat over all chunks)
# ============================================================================

@jax.jit(static_argnames=["config"])
def exact_bptt_step(base_xlstm, params, all_chunks, config):
    """all_chunks: [num_chunks, chunk_size+1] int32. Gradients through all boundaries."""
    num_heads = config.num_heads
    block_map = config.block_map

    def loss_fn(p):
        init_states = init_step_states(config, batch_size=1)

        @jax.checkpoint
        def scan_body(states, chunk):
            inputs  = chunk[None, :-1]
            targets = chunk[None, 1:]
            logits, new_states = forward_train_chunked(
                base_xlstm, p, inputs, states, num_heads, block_map)
            loss = cross_entropy_loss(logits, targets)
            return new_states, loss

        _, per_chunk_losses = jax.lax.scan(scan_body, init_states, all_chunks)
        return per_chunk_losses.mean()

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads


# ============================================================================
# State initialisation
# ============================================================================

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

def _fir_bank(n_channels, K):
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
    tau = np.exp(np.linspace(0.0, np.log(max(float(seq_len), 2.0)), n))
    f   = np.exp(-1.0 / tau)
    return _bf(np.log(f / (1.0 - f + 1e-9)))


# ============================================================================
# Frozen xLSTM initialisation
# ============================================================================

def init_mlstm_params(key, d_model, num_heads, conv_kernel, seq_len) -> mLSTMParams:
    inner = d_model
    DH    = inner // num_heads
    ks    = jr.split(key, 10)
    W_up_raw = np.array(jr.normal(ks[0], (2*inner, d_model)))
    W_up_raw /= np.linalg.norm(W_up_raw, axis=1, keepdims=True)
    W_up = _bf(W_up_raw / math.sqrt(d_model))
    def _qkv(k):
        M = np.array(_ortho(k, inner, inner))
        return _bf(M.reshape(num_heads, DH, inner) / math.sqrt(inner))
    W_q = _qkv(ks[1]); W_k = _qkv(ks[2]); W_v = _qkv(ks[3])
    i_scale = np.linspace(0.5, 2.0, num_heads)[:, None]
    W_i = _bf(np.array(jr.normal(ks[4], (num_heads, inner))) / math.sqrt(inner) * i_scale)
    W_f = _bf(np.array(jr.normal(ks[5], (num_heads, inner))) / math.sqrt(inner))
    b_f = _multiscale_forget_bias(num_heads, seq_len)
    b_i = jnp.zeros((num_heads,), DTYPE)
    W_down = _ortho(ks[6], d_model, inner) / math.sqrt(inner)
    conv_w = _fir_bank(inner, conv_kernel)
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
    W_i = _ortho(ks[0], H, d_model) / math.sqrt(d_model)
    W_f = _ortho(ks[1], H, d_model) / math.sqrt(d_model)
    W_z = _ortho(ks[2], H, d_model) / math.sqrt(d_model)
    W_o = _ortho(ks[3], H, d_model) / math.sqrt(d_model)
    raw_R = np.array(jr.normal(ks[4], (4*H, H)))
    U, _, Vt = np.linalg.svd(raw_R, full_matrices=False)
    Ry = _bf(0.95 * (U @ Vt))
    f_bias = np.array(_multiscale_forget_bias(H, seq_len))
    b_arr  = np.zeros(4*H); b_arr[H:2*H] = f_bias
    b = _bf(b_arr)
    gn_w   = jnp.ones((H,),  DTYPE)
    gn_b   = jnp.zeros((H,), DTYPE)
    conv_w = _fir_bank(d_model, conv_kernel)
    return sLSTMParams(W_i=W_i, W_f=W_f, W_z=W_z, W_o=W_o,
                       Ry=Ry, b=b, gn_w=gn_w, gn_b=gn_b, conv_w=conv_w)

def init_xlstm_block_params(key, d_model, num_heads, d_ff, conv_kernel,
                             seq_len, btype) -> xLSTMBlockParams:
    ks = jr.split(key, 4)
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
# HiRA initialisation
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
# Loss
# ============================================================================

def cross_entropy_loss(logits, targets):
    log_probs   = jax.nn.log_softmax(logits, axis=-1)
    loss_per_tok= -jnp.take_along_axis(log_probs, targets[...,None], axis=-1)[...,0]
    mask        = (targets != PAD_TOKEN).astype(jnp.float32)
    return (loss_per_tok*mask).sum() / (mask.sum()+1e-8)


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

def split_segments(tokens, segment_size):
    segs = []
    n = len(tokens)
    i = 0
    while i < n:
        segs.append(tokens[i : i + segment_size])
        i += segment_size
    return segs

def make_chunks_array(tokens, chunk_size):
    """Pack tokens into [num_chunks, chunk_size+1] int32 array, zero-padded."""
    n = len(tokens)
    num_chunks = max(1, math.ceil((n - 1) / chunk_size))
    pad_len    = num_chunks * chunk_size + 1 - n
    padded     = np.concatenate([tokens, np.zeros(pad_len, dtype=np.uint8)])
    chunks     = [padded[i*chunk_size : i*chunk_size + chunk_size + 1]
                  for i in range(num_chunks)]
    return jnp.array(np.stack(chunks), dtype=jnp.int32)


# ============================================================================
# Segment evaluation  (stateful: state carries from segment to segment)
# ============================================================================

def eval_segments_stateful(base_xlstm, params, config, seg_list):
    """
    Zero-state TF eval with state flowing across segments.
    preds[k] within a segment predicts seg[k+1].
    Returns list of (accuracy, first_fail_pred_idx) per segment.
    first_fail_pred_idx is 0-based index into the segment's prediction positions;
    the failed byte is at position (first_fail_pred_idx + 1) within the segment.
    """
    chunk_size = config.max_seq_len
    states = init_step_states(config, batch_size=1)
    results = []

    for seg in seg_list:
        n = len(seg)
        if n < 2:
            results.append((1.0, None))
            continue

        num_chunks = max(1, math.ceil((n - 1) / chunk_size))
        pad_len    = num_chunks * chunk_size + 1 - n
        padded     = np.concatenate([seg, np.zeros(pad_len, dtype=np.uint8)])

        all_preds = []
        all_tgts  = []
        for ci in range(num_chunks):
            s     = ci * chunk_size
            chunk = padded[s : s + chunk_size + 1]
            inp   = jnp.array(chunk[None, :-1], dtype=jnp.int32)
            tgt   = jnp.array(chunk[None, 1:],  dtype=jnp.int32)
            logits, states = _fwd_chunked_jit(
                base_xlstm, params, inp, states, config.num_heads, config.block_map)
            all_preds.append(np.array(jnp.argmax(logits[0], axis=-1)))
            all_tgts.append(np.array(tgt[0]))

        preds_np = np.concatenate(all_preds)[:n-1]
        tgts_np  = np.concatenate(all_tgts)[:n-1]
        mask     = tgts_np != PAD_TOKEN
        correct  = (preds_np == tgts_np) & mask
        n_valid  = int(mask.sum())
        acc      = float(correct.sum()) / max(n_valid, 1)
        wrong    = np.where(~correct & mask)[0]
        # wrong[0] is prediction index k → byte (k+1) of segment is wrong
        first_fail = int(wrong[0]) if len(wrong) > 0 else None
        results.append((acc, first_fail))

    return results


# ============================================================================
# Generative evaluation (seed → autoregressive decode, persistent state)
# ============================================================================

def eval_generative(base_xlstm, params, config, all_tokens):
    """
    Seed with all_tokens[0], generate len(all_tokens)-1 bytes autoregressively.
    Returns (generated_np, bytes_wrong, accuracy, first_wrong_byte_idx_or_None).
    """
    target_np  = np.array(all_tokens[1:], dtype=np.uint8)
    n          = len(target_np)
    states     = init_step_states(config, batch_size=1)
    cur_token  = jnp.array([int(all_tokens[0])])
    gen_result = []
    for _ in range(n):
        logit, states = _fwd_step_jit(
            base_xlstm, params, cur_token, states, config.num_heads, config.block_map)
        cur_token = jnp.argmax(logit, axis=-1)
        gen_result.append(int(cur_token[0]))
    generated   = np.array(gen_result, dtype=np.uint8)
    wrong_mask  = generated != target_np
    bytes_wrong = int(np.sum(wrong_mask))
    accuracy    = (n - bytes_wrong) / n
    wrong_idxs  = np.where(wrong_mask)[0]
    first_wrong = int(wrong_idxs[0]) if len(wrong_idxs) > 0 else None
    return generated, bytes_wrong, accuracy, first_wrong


# ============================================================================
# Checkpoint  (single "last" directory, overwritten after each phase)
# ============================================================================

def save_checkpoint(ckpt_dir, params, config, seg_idx, phase):
    import json, pickle
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {**config._asdict(), "seg_idx": seg_idx, "phase": phase}
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(ckpt_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    print(f"  [ckpt] → {ckpt_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    _script = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join("log", _script)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    dataset_path = "datasets/juz1.txt"
    tokens = load_dataset(dataset_path)
    n_total = len(tokens)
    print(f"Dataset: {dataset_path}  ({n_total} bytes)")

    config = Config(
        vocab_size=256, d_model=64, num_heads=4, d_ff=128,
        num_layers=4, max_seq_len=SEGMENT_SIZE, seed=0, conv_kernel=4,
        block_map="smsm", lora_r=16, batch_size=1,
        learning_rate=5e-3, weight_decay=0.0, grad_clip_norm=100.0,
        random_shuffle=False,
    )

    key = jr.key(config.seed)
    key, k_xlstm, k_rc = jr.split(key, 3)

    print("Initialising frozen xLSTM...")
    base_xlstm = init_xlstm_params(k_xlstm, config)
    frozen     = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(base_xlstm))

    print("Initialising HiRA adapters...")
    params    = init_randcompress_params(k_rc, config)
    trainable = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    total          = frozen + trainable
    dtype_bytes    = np.dtype(DTYPE).itemsize
    param_bytes    = trainable * dtype_bytes
    compress_ratio = n_total / param_bytes
    print(f"{'metric':<28}  {'value':>12}")
    print(f"{'-'*28}  {'-'*12}")
    print(f"{'frozen params':<28}  {frozen/1e6:>11.3f}M")
    print(f"{'trainable params':<28}  {trainable/1e6:>11.3f}M")
    print(f"{'total params':<28}  {total/1e6:>11.3f}M")
    print(f"{'trainable / total':<28}  {trainable/total*100:>11.1f}%")
    print(f"{'---':<28}  {'---':>12}")
    print(f"{'dtype':<28}  {jnp.dtype(DTYPE).name:>12}")
    print(f"{'dtype bytes':<28}  {dtype_bytes:>12}")
    print(f"{'trainable param bytes':<28}  {param_bytes/1e6:>11.3f}M")
    print(f"{'dataset bytes':<28}  {n_total/1e6:>11.3f}M")
    print(f"{'---':<28}  {'---':>12}")
    print(f"{'params / dataset byte':<28}  {trainable/n_total:>12.2f}")
    print(f"{'compression ratio':<28}  {compress_ratio:>11.2f}x")
    print(f"{'compressed?':<28}  {'yes' if compress_ratio > 1 else 'no (over-param)':>12}")

    opt_state = adamw_init(params)
    lr        = config.learning_rate

    segs   = split_segments(tokens, SEGMENT_SIZE)
    n_segs = len(segs)
    print(f"Segments: {n_segs} × {SEGMENT_SIZE} bytes  (last={len(segs[-1])} bytes)")
    print(f"MAX_ITER_PER_PHASE={MAX_ITER_PER_PHASE}  CHECK_EVERY={CHECK_EVERY}  lr={lr}")
    print()

    # Per-phase summary table rows
    phase_summary = []

    completed = []  # segments that passed solo+combined
    t_run_start = time.perf_counter()

    def _print_time(seg_elapsed, run_elapsed):
        done = sum(len(s) for s in completed)
        print(f"[time] seg={seg_elapsed:.1f}s  total={run_elapsed:.1f}s  "
              f"bytes={done} ({done/1e6:.3f}M  {done/n_total*100:.1f}% of {n_total})")

    for seg_idx in range(n_segs):
        seg         = segs[seg_idx]
        seg_name    = f"seg{seg_idx+1:03d}"
        byte_lo     = seg_idx * SEGMENT_SIZE
        byte_hi     = byte_lo + len(seg)
        t_seg_start = time.perf_counter()

        print(f"\n{'='*70}")
        print(f"SEGMENT {seg_idx+1}/{n_segs}  bytes [{byte_lo}, {byte_hi})  len={len(seg)}")
        print(f"{'='*70}")

        # ------------------------------------------------------------------ #
        # Forgetting snapshot BEFORE solo phase                               #
        # ------------------------------------------------------------------ #
        if completed:
            pre_accs = eval_segments_stateful(base_xlstm, params, config, completed + [seg])
            print(f"[pre-solo forgetting]")
            for pi, (pa, pfw) in enumerate(pre_accs):
                label = f"NEW seg{seg_idx+1}" if pi == len(completed) else f"seg{pi+1:03d}"
                fw_str = f"pred{pfw}→byte{pfw+1}" if pfw is not None else "ok"
                print(f"  {label}: acc={pa:.1%}  first_fail={fw_str}")

        # ------------------------------------------------------------------ #
        # SOLO phase                                                           #
        # ------------------------------------------------------------------ #
        print(f"\n[SOLO] seg{seg_idx+1}  budget={MAX_ITER_PER_PHASE} iters")
        solo_chunks = make_chunks_array(seg, config.max_seq_len)
        t0          = time.perf_counter()
        solo_iters  = 0
        solo_success = False
        last_accs   = None

        pbar = tqdm(range(1, MAX_ITER_PER_PHASE + 1), desc=f"solo s{seg_idx+1}", unit="it", file=sys.stderr)
        for it in pbar:
            loss, grads = exact_bptt_step(base_xlstm, params, solo_chunks, config)
            params, opt_state = adamw_update(
                opt_state, grads, params, lr=lr,
                weight_decay=config.weight_decay, max_norm=config.grad_clip_norm)
            solo_iters = it
            pbar.set_postfix(loss=f"{float(loss):.5f}")

            if it % CHECK_EVERY == 0 or it == MAX_ITER_PER_PHASE:
                accs = eval_segments_stateful(base_xlstm, params, config, [seg])
                last_accs = accs
                acc, fw = accs[0]
                elapsed = time.perf_counter() - t0
                fw_str = f"pred{fw}→byte{fw+1}" if fw is not None else "ok"
                print(f"  [solo it={it:5d}] loss={float(loss):.5f}  "
                      f"acc={acc:.1%}  first_fail={fw_str}  {elapsed:.1f}s")
                if acc == 1.0:
                    solo_success = True
                    break
        pbar.close()

        elapsed_solo = time.perf_counter() - t0

        if not solo_success:
            final_acc = last_accs[0][0] if last_accs else 0.0
            final_fw  = last_accs[0][1] if last_accs else None
            fw_str = f"pred{final_fw}→byte{final_fw+1}" if final_fw is not None else "ok"
            seg_elapsed = time.perf_counter() - t_seg_start
            run_elapsed = time.perf_counter() - t_run_start
            print(f"\n[FAIL] seg{seg_idx+1} SOLO — could not reach 100% in "
                  f"{MAX_ITER_PER_PHASE} iters  "
                  f"(final acc={final_acc:.1%}  first_fail={fw_str})")
            _print_time(seg_elapsed, run_elapsed)
            phase_summary.append(
                (seg_idx+1, "SOLO", solo_iters, elapsed_solo, False, final_acc))
            break

        print(f"[PASS] seg{seg_idx+1} SOLO — 100% in {solo_iters} iters  {elapsed_solo:.1f}s")
        phase_summary.append((seg_idx+1, "SOLO", solo_iters, elapsed_solo, True, 1.0))

        # ------------------------------------------------------------------ #
        # Forgetting snapshot AFTER solo phase (effect on past segments)      #
        # ------------------------------------------------------------------ #
        if completed:
            post_solo_accs = eval_segments_stateful(
                base_xlstm, params, config, completed + [seg])
            print(f"[post-solo forgetting]")
            for pi, (pa, pfw) in enumerate(post_solo_accs):
                label = f"seg{pi+1:03d}"
                fw_str = f"pred{pfw}→byte{pfw+1}" if pfw is not None else "ok"
                print(f"  {label}: acc={pa:.1%}  first_fail={fw_str}")

        completed.append(seg)

        # Generative eval on completed segments so far
        print(f"  [gen eval] segs 1..{len(completed)} ...")
        gen_tokens = np.concatenate(completed)
        gen_out, gen_wrong, gen_acc, gen_fw = eval_generative(
            base_xlstm, params, config, gen_tokens)
        gen_fw_str = f"byte {gen_fw}" if gen_fw is not None else "none (perfect)"
        print(f"  [gen] acc={gen_acc:.1%}  wrong={gen_wrong}/{len(gen_tokens)-1}  "
              f"first_wrong={gen_fw_str}")
        gen_path = os.path.join(log_dir, f"gen_seg{seg_idx+1:03d}_solo.txt")
        with open(gen_path, "w", encoding="utf-8", errors="replace") as gf:
            gf.write(f"phase:       seg{seg_idx+1} SOLO\n"
                     f"segs:        1..{len(completed)}\n"
                     f"gen_acc:     {gen_acc:.1%}\n"
                     f"wrong:       {gen_wrong}/{len(gen_tokens)-1}\n"
                     f"first_wrong: {gen_fw_str}\n"
                     f"\n--- seed byte ---\n{ByteTokenizer.decode([gen_tokens[0]])}\n"
                     f"\n--- target ---\n{ByteTokenizer.decode(gen_tokens[1:])}\n"
                     f"\n--- generated ---\n{ByteTokenizer.decode(gen_out)}\n")

        save_checkpoint(os.path.join(log_dir, "ckpt_last"), params, config, seg_idx, "solo")

        if seg_idx == 0:
            seg_elapsed = time.perf_counter() - t_seg_start
            run_elapsed = time.perf_counter() - t_run_start
            _print_time(seg_elapsed, run_elapsed)
            continue

        # ------------------------------------------------------------------ #
        # COMBINED phase: train on ALL completed segments together            #
        # ------------------------------------------------------------------ #
        n_done = len(completed)
        print(f"\n[COMBINED] segs 1..{n_done}  budget={MAX_ITER_PER_PHASE} iters")
        combined_tokens = np.concatenate(completed)
        combined_chunks = make_chunks_array(combined_tokens, config.max_seq_len)
        t0              = time.perf_counter()
        comb_iters      = 0
        comb_success    = False
        last_comb_accs  = None

        pbar = tqdm(range(1, MAX_ITER_PER_PHASE + 1), desc=f"comb 1..{n_done}", unit="it", file=sys.stderr)
        for it in pbar:
            loss, grads = exact_bptt_step(base_xlstm, params, combined_chunks, config)
            params, opt_state = adamw_update(
                opt_state, grads, params, lr=lr,
                weight_decay=config.weight_decay, max_norm=config.grad_clip_norm)
            comb_iters = it
            pbar.set_postfix(loss=f"{float(loss):.5f}")

            if it % CHECK_EVERY == 0 or it == MAX_ITER_PER_PHASE:
                accs = eval_segments_stateful(base_xlstm, params, config, completed)
                last_comb_accs = accs
                all_perfect = all(a == 1.0 for a, _ in accs)
                elapsed = time.perf_counter() - t0
                parts = []
                for pi, (pa, pfw) in enumerate(accs):
                    fw_str = f"b{pfw+1}" if pfw is not None else "ok"
                    parts.append(f"s{pi+1}={pa:.0%}({fw_str})")
                print(f"  [comb it={it:5d}] loss={float(loss):.5f}  "
                      f"{' '.join(parts)}  {elapsed:.1f}s")
                if all_perfect:
                    comb_success = True
                    break
        pbar.close()

        elapsed_comb = time.perf_counter() - t0

        if not comb_success:
            seg_elapsed = time.perf_counter() - t_seg_start
            run_elapsed = time.perf_counter() - t_run_start
            print(f"\n[FAIL] COMBINED segs 1..{n_done} — could not reach 100% in "
                  f"{MAX_ITER_PER_PHASE} iters")
            if last_comb_accs:
                for pi, (pa, pfw) in enumerate(last_comb_accs):
                    fw_str = f"pred{pfw}→byte{pfw+1}" if pfw is not None else "ok"
                    print(f"  seg{pi+1:03d}: acc={pa:.1%}  first_fail={fw_str}")
            _print_time(seg_elapsed, run_elapsed)
            phase_summary.append(
                (seg_idx+1, "COMBINED", comb_iters, elapsed_comb, False,
                 min(a for a, _ in last_comb_accs) if last_comb_accs else 0.0))
            break

        mean_acc = sum(a for a, _ in last_comb_accs) / len(last_comb_accs)
        print(f"[PASS] COMBINED segs 1..{n_done} — 100% in {comb_iters} iters  "
              f"{elapsed_comb:.1f}s")
        phase_summary.append(
            (seg_idx+1, "COMBINED", comb_iters, elapsed_comb, True, mean_acc))

        # Generative eval on all completed segments
        print(f"  [gen eval] segs 1..{n_done} ...")
        gen_tokens = np.concatenate(completed)
        gen_out, gen_wrong, gen_acc, gen_fw = eval_generative(
            base_xlstm, params, config, gen_tokens)
        gen_fw_str = f"byte {gen_fw}" if gen_fw is not None else "none (perfect)"
        print(f"  [gen] acc={gen_acc:.1%}  wrong={gen_wrong}/{len(gen_tokens)-1}  "
              f"first_wrong={gen_fw_str}")
        gen_path = os.path.join(log_dir, f"gen_segs001_to_{seg_name}_combined.txt")
        with open(gen_path, "w", encoding="utf-8", errors="replace") as gf:
            gf.write(f"phase:       COMBINED segs 1..{n_done}\n"
                     f"gen_acc:     {gen_acc:.1%}\n"
                     f"wrong:       {gen_wrong}/{len(gen_tokens)-1}\n"
                     f"first_wrong: {gen_fw_str}\n"
                     f"\n--- seed byte ---\n{ByteTokenizer.decode([gen_tokens[0]])}\n"
                     f"\n--- target ---\n{ByteTokenizer.decode(gen_tokens[1:])}\n"
                     f"\n--- generated ---\n{ByteTokenizer.decode(gen_out)}\n")

        save_checkpoint(os.path.join(log_dir, "ckpt_last"), params, config, seg_idx, "combined")
        seg_elapsed = time.perf_counter() - t_seg_start
        run_elapsed = time.perf_counter() - t_run_start
        print(f"[time] seg={seg_elapsed:.1f}s  total={run_elapsed:.1f}s")

    # ------------------------------------------------------------------ #
    # Summary table                                                        #
    # ------------------------------------------------------------------ #
    print(f"\n\n{'='*70}")
    print(f"PHASE SUMMARY")
    print(f"{'='*70}")
    print(f"{'seg':>5}  {'phase':<10}  {'iters':>6}  {'time(s)':>8}  {'pass':>5}  {'acc':>7}")
    print(f"{'-'*5}  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*7}")
    for seg_i, phase, iters, elapsed, passed, acc in phase_summary:
        print(f"{seg_i:>5}  {phase:<10}  {iters:>6}  {elapsed:>8.1f}  "
              f"{'PASS' if passed else 'FAIL':>5}  {acc:>7.1%}")

    print(f"\nLog: {log_path}")
    log_file.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
