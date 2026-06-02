"""
randcompress v4.2 — exact BPTT with gradient rematerialisation

Changes from v4.1
-----------------
Training : exact BPTT — no stop_gradient between chunks.  All sequential
           chunks for the epoch are stacked into one array and processed via
           jax.lax.scan with jax.checkpoint on the scan body.  Gradients
           flow back through every chunk boundary to the very first token.
           Memory cost is O(state_size * num_chunks + activations_one_chunk)
           instead of O(activations * num_chunks) thanks to remat.

Generation: same as v4.1 — persistent state, no reset.

Dataset : surat_al-fatihah.txt   (562 bytes)
chunk_size : 128
n_seed     : 32
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
    max_seq_len:   int   = 512
    seed:          int   = 0
    conv_kernel:   int   = 4
    block_map:     str   = "mmmm"
    lora_r:        int   = 8
    batch_size:    int   = 16
    learning_rate: float = 1e-3
    weight_decay:  float = 0.0
    grad_clip_norm:float = 1.0
    random_shuffle:bool  = True


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
    """Zero-state forward pass (used for training and teacher-forced eval)."""
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

# ---- TBPTT helpers: forward with initial state carry-in ----

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

def mlstm_layer_parallel_with_init(p, x, num_heads, state):
    C0, n0, m0, conv_buf = state
    B, S, _ = x.shape
    inner   = p.W_up.shape[0] // 2
    K       = p.conv_w.shape[1]
    xz      = jnp.dot(x, p.W_up.T)
    x_m, z  = xz[...,:inner], xz[...,inner:]
    x_conv_a= jax.nn.silu(causal_conv1d_with_buf(x_m, p.conv_w, conv_buf))
    q,k,v,i_p,f_p = _mlstm_preacts(p, x_conv_a, x_m)

    # parallel scan with init state
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

@jax.jit(static_argnames=["config"])
def exact_bptt_step(base_xlstm, params, all_chunks, config):
    """Exact BPTT over all chunks via scan + remat.

    all_chunks : [num_chunks, chunk_size+1] int32
    Gradients flow through every chunk boundary.  jax.checkpoint on the scan
    body recomputes per-chunk activations in the backward pass instead of
    storing them, so peak memory is O(state * num_chunks + acts_one_chunk).
    """
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


@jax.jit(static_argnames=["config"])
def val_loss_scan(base_xlstm, params, all_val_chunks, config):
    """Teacher-forced BPC over all val chunks via scan (no gradients)."""
    init_states = init_step_states(config, batch_size=1)

    def scan_body(states, chunk):
        inputs  = chunk[None, :-1]
        targets = chunk[None, 1:]
        logits, new_states = forward_train_chunked(
            base_xlstm, params, inputs, states, config.num_heads, config.block_map)
        loss = cross_entropy_loss(logits, targets)
        return new_states, loss

    _, per_chunk_losses = jax.lax.scan(scan_body, init_states, all_val_chunks)
    return per_chunk_losses.mean()


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
# Batcher
# ============================================================================

def make_batch(tokens, batch_size, chunk_size, rng_key, random_shuffle=True):
    """Return [batch_size, chunk_size+1] int32 array.

    random_shuffle=True  → each row is a random segment from anywhere in tokens.
    random_shuffle=False → sequential non-overlapping chunks (cycled via step index;
                           pass rng_key as the chunk index wrapped in jnp.array).
    """
    n = len(tokens)
    seg_len = chunk_size + 1
    if random_shuffle:
        max_start = n - seg_len
        starts = np.array(jr.randint(rng_key, (batch_size,), 0, max_start + 1))
    else:
        # rng_key doubles as integer step index (0-based) when shuffle is off
        step = int(jax.device_get(rng_key)) if hasattr(rng_key, 'shape') else int(rng_key)
        num_chunks = (n - 1) // chunk_size
        base = (step * batch_size) % num_chunks
        idxs = [(base + i) % num_chunks for i in range(batch_size)]
        starts = np.array([i * chunk_size for i in idxs])
    rows = np.stack([tokens[s : s + seg_len] for s in starts])
    return jnp.array(rows, dtype=jnp.int32)


# ============================================================================
# Training step (stateless)
# ============================================================================

@jax.jit(static_argnames=["config"])
def train_step(base_xlstm, params, batch, config):
    """Single stateless step: zero hidden state, random segment, one update."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    def loss_fn(p):
        logits = forward_train(base_xlstm, p, inputs, config.num_heads, config.block_map)
        return cross_entropy_loss(logits, targets)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads


# ============================================================================
# Checkpoint save / load
# ============================================================================

def save_checkpoint(ckpt_dir, params, config, n_seed, n_output,
                    seed_bytes, warmup_tokens):
    import os, json, pickle
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {**config._asdict(),
            "n_seed": n_seed, "n_output": n_output,
            "warmup_tokens": warmup_tokens}
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(ckpt_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    with open(os.path.join(ckpt_dir, "seed.bin"), "wb") as f:
        f.write(bytes(seed_bytes))
    print(f"Checkpoint saved to {ckpt_dir}/")


def load_checkpoint(ckpt_dir):
    import os, json, pickle
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        meta = json.load(f)
    n_seed        = meta.pop("n_seed")
    n_output      = meta.pop("n_output")
    warmup_tokens = meta.pop("warmup_tokens")
    config        = Config(**meta)
    with open(os.path.join(ckpt_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.array, params)
    with open(os.path.join(ckpt_dir, "seed.bin"), "rb") as f:
        seed_bytes = list(f.read())
    return config, params, seed_bytes, n_seed, n_output, warmup_tokens


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
    """Persistent-state autoregressive generation — no chunk reset, no warmup.

    State flows continuously from the first seed byte through the full output,
    mirroring TBPTT training where state also carries across chunk boundaries.
    """
    n         = len(target_np)
    states    = init_step_states(config, batch_size=1)
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
    import os
    _script = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join("log", _script)
    os.makedirs(log_dir, exist_ok=True)
    train_log_path = f"{log_dir}/train.log"
    eval_log_path  = f"{log_dir}/eval_final.log"
    train_log_file = open(train_log_path, "w", buffering=1)
    sys.stdout     = _Tee(sys.__stdout__, train_log_file)

    dataset_path = "datasets/juz1.txt"
    val_path     = dataset_path
    # val_path     = "datasets/surat_al-fatihah.txt"
    tokens     = load_dataset(dataset_path)
    val_tokens = load_dataset(val_path)
    n_real = len(tokens)
    n_val  = len(val_tokens)
    chunk_size = 128
    n_seed     = 64
    print(f"Train: {n_real} bytes   Val: {n_val} bytes   chunk_size={chunk_size}  n_seed={n_seed}")

    # config = Config(
    #     vocab_size=256, d_model=32, num_heads=4, d_ff=64,
    #     num_layers=4, max_seq_len=chunk_size, seed=0, conv_kernel=4,
    #     block_map="smsm", lora_r=16, batch_size=1,
    #     learning_rate=1e-3, weight_decay=0.0, grad_clip_norm=100.0,
    #     random_shuffle=False,
    # )
    config = Config(
        vocab_size=256, d_model=64, num_heads=4, d_ff=128,
        num_layers=4, max_seq_len=chunk_size, seed=0, conv_kernel=4,
        block_map="smsm", lora_r=16, batch_size=1,
        learning_rate=5e-3,
        weight_decay=0.0, grad_clip_norm=100.0,
        random_shuffle=False,
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

    # Pad dataset to next chunk boundary so every byte is trained on.
    # PAD_TOKEN=0 is already masked from loss (safe for Arabic UTF-8).
    num_chunks  = math.ceil((n_real - 1) / chunk_size)
    pad_len     = num_chunks * chunk_size + 1 - n_real
    tokens_padded = np.concatenate([tokens, np.zeros(pad_len, dtype=np.uint8)])
    eval_chunks = [tokens_padded[i*chunk_size : i*chunk_size + chunk_size + 1]
                   for i in range(num_chunks)]
    # stacked array for exact BPTT scan: [num_chunks, chunk_size+1]
    all_chunks_jnp = jnp.array(
        np.stack(eval_chunks), dtype=jnp.int32)
    print(f"n_real={n_real}  pad={pad_len}  num_chunks={num_chunks}  exact BPTT + remat")

    val_chunks_n = max(1, (n_val - 1) // chunk_size)
    val_eval_chunks = [val_tokens[i*chunk_size : i*chunk_size + chunk_size + 1]
                       for i in range(val_chunks_n)]
    all_val_chunks_jnp = jnp.array(np.stack(val_eval_chunks), dtype=jnp.int32)

    num_epochs  = 10000
    global_step = 0
    t_total     = 0.0
    lr          = config.learning_rate


    print(f"\nExact BPTT + remat  epochs={num_epochs}  log -> {train_log_path}")
    print("-" * 60)

    pbar = tqdm(range(num_epochs), desc="train", unit="epoch")
    for epoch in pbar:
        # lr = cosine_lr(global_step, num_epochs, config.learning_rate)

        t0 = time.perf_counter()
        loss, grads = exact_bptt_step(base_xlstm, params, all_chunks_jnp, config)
        params, opt_state = adamw_update(
            opt_state, grads, params,
            lr=lr, weight_decay=config.weight_decay, max_norm=config.grad_clip_norm)
        elapsed_ms  = (time.perf_counter() - t0) * 1000
        t_total    += elapsed_ms
        global_step += 1

        loss_f = float(loss)
        bpc    = math.exp(loss_f) / math.log(2)
        avg_ms = t_total / global_step
        pbar.set_postfix(loss=f"{loss_f:.5f}", bpc=f"{bpc:.4f}",
                         lr=f"{lr:.2e}", ms=f"{avg_ms:.1f}")

        if epoch % 1000 == 0:
            # teacher-forced BPC on val (scan, JIT compiled)
            val_mean_loss = val_loss_scan(base_xlstm, params, all_val_chunks_jnp, config)
            val_bpc = math.exp(float(val_mean_loss)) / math.log(2)

            val_target_gen = np.array(val_tokens[n_seed:], dtype=np.uint8)
            val_gen, val_wrong, val_acc, _, val_first = _eval_generative(
                base_xlstm, params, config,
                seed_bytes=[int(b) for b in val_tokens[:n_seed]],
                target_np=val_target_gen)
            val_first_str = f"byte {val_first}" if val_first is not None else "none (perfect)"
            print(f"  [VAL] bpc={val_bpc:.4f}  acc={val_acc*100:.2f}%  "
                  f"wrong={val_wrong}/{len(val_target_gen)}  first_wrong={val_first_str}")
            val_seed_text   = ByteTokenizer.decode(val_tokens[:n_seed])
            val_gen_text    = ByteTokenizer.decode(val_gen)
            val_target_text = ByteTokenizer.decode(val_target_gen)
            # print(f"  seed: {val_seed_text!r}")
            # print(f"  gen:  {val_gen_text}")
            val_gen_path = f"{log_dir}/val_gen_epoch{epoch+1:04d}.txt"
            with open(val_gen_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(
                    f"epoch:       {epoch+1}\n"
                    f"bpc:         {val_bpc:.4f}\n"
                    f"acc:         {val_acc*100:.2f}%\n"
                    f"wrong:       {val_wrong}/{len(val_target_gen)}\n"
                    f"first_wrong: {val_first_str}\n"
                    f"\n--- seed ---\n{val_seed_text}\n"
                    f"\n--- target ---\n{val_target_text}\n"
                    f"\n--- generated ---\n{val_gen_text}\n"
                )
            train_target = np.array(tokens[n_seed:], dtype=np.uint8)
            n_train_pred = len(train_target)
            save_checkpoint(os.path.join(log_dir, "last_epoch"), params, config,
                            n_seed=n_seed, n_output=n_train_pred,
                            seed_bytes=[int(b) for b in tokens[:n_seed]],
                            warmup_tokens=0)
            print()

        if val_acc == 1.0:
            print(f"\n  Converged at epoch {epoch+1}")
            break

    # ---- final evaluation ----
    print("\n" + "="*60)
    print("Final evaluation on training dataset...")
    print("="*60)

    eval_log_file = open(eval_log_path, "w", buffering=1)
    sys.stdout    = _Tee(sys.__stdout__, train_log_file, eval_log_file)

    print("Computing train BPC (teacher-forced, persistent state)...")
    eval_states     = init_step_states(config, batch_size=1)
    total_eval_loss = 0.0
    for ci in tqdm(range(num_chunks), desc="  tf-bpc", unit="chunk"):
        seg           = eval_chunks[ci]
        chunk_batch   = jnp.array(seg[None,:], dtype=jnp.int32)
        frozen_es     = jax.lax.stop_gradient(eval_states)
        logits_c, eval_states = forward_train_chunked(
            base_xlstm, params, chunk_batch[:,:-1], frozen_es,
            config.num_heads, config.block_map)
        total_eval_loss += float(cross_entropy_loss(logits_c, chunk_batch[:,1:]))
    train_bpc = math.exp(total_eval_loss / num_chunks) / math.log(2)
    print(f"Train BPC (teacher-forced, persistent): {train_bpc:.4f}")

    train_target = np.array(tokens[n_seed:], dtype=np.uint8)
    n_train_pred = len(train_target)
    print(f"Generating {n_train_pred} bytes seeded with first {n_seed} bytes (persistent state)...")
    train_generated, train_wrong, train_acc, train_cer, train_first = _eval_generative(
        base_xlstm, params, config,
        seed_bytes=[int(b) for b in tokens[:n_seed]], target_np=train_target)
    train_first_str = f"byte {train_first}" if train_first is not None else "none (perfect)"

    train_seed_text = ByteTokenizer.decode(tokens[:n_seed])
    train_gen_text  = ByteTokenizer.decode(train_generated)
    gen_text_path = eval_log_path.replace(".log", "_generated.txt")
    with open(gen_text_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"seed: {train_seed_text}\n---\n{train_gen_text}")
    print(f"  seed: {train_seed_text!r}")
    print(f"  gen:  {train_gen_text}")
    print(f"Generated text saved to {gen_text_path}")

    print(f"\n{'='*60}")
    print(f"Final train-set evaluation ({n_train_pred} bytes):")
    print(f"  BPC (teacher-forced): {train_bpc:.4f}")
    print(f"  Bytes wrong:          {train_wrong} / {n_train_pred}")
    print(f"  First wrong byte idx: {train_first_str}")
    print(f"  Accuracy:             {train_acc*100:.4f}%")
    print(f"  CER:                  {train_cer*100:.4f}%")
    print(f"{'='*60}")

    ckpt_dir = os.path.join(log_dir, "checkpoint")
    save_checkpoint(ckpt_dir, params, config,
                    n_seed=n_seed, n_output=n_train_pred,
                    seed_bytes=[int(b) for b in tokens[:n_seed]],
                    warmup_tokens=0)

    eval_log_file.close()
    train_log_file.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
