"""
randcompress v4.4.2.1 — Power Attention mLSTM + Curriculum Learning

Changes from v4.4.2:
  - mLSTM upgraded to Power Attention (arXiv:2507.04239), power_p=2 by default.
    Parallel mode: attention weights raised to p-th power: qk → qk**p.
    Step/recurrent mode: q and k expanded through symmetric power map φ_p before
    accumulation. State C: [B,NH,DH,DH] → [B,NH,D_sym,DH] where
    D_sym = C(DH+p-1,p) = DH*(DH+1)//2 for p=2.
    mlstm_layer_parallel_with_init updated consistently for init-state handling.
  - Config: new power_p=2 field. Set power_p=1 to recover v4.4.1 behavior.
  - POWER_P module global set from config.power_p in main().
  - Curriculum learning (from v8): dataset split into segments, trained in order.
    SOLO phase per segment (TBPTT on that segment alone until 100% TF-accuracy).
    COMBINED phase (TBPTT on all completed segments until all 100% or budget).
    TBPTT + stop_gradient within each phase unchanged.
    New config field: max_iter_per_phase (replaces max_iters).
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
    d_model:       int   = 64
    num_heads:     int   = 8
    num_layers:    int   = 4      # always overridden to len(block_map)
    segment_size:  int   = 1024
    power_p:       int   = 2      # mLSTM attention kernel degree; 1 = v4.4.1 baseline
    seed:          int   = 0
    block_map:     str   = "smmm"
    stride_map:    str   = "1111"  # one digit per layer; stride-N fires every N tokens
    # block_map:     str   = "mmmm"
    # stride_map:    str   = "1248"  # one digit per layer; stride-N fires every N tokens
    lora_r:        int   = 4
    batch_size:    int   = 1
    # ── optimiser ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-2
    weight_decay:  float = 0.0
    grad_clip_norm:float = 1e2
    sinkgd_l:      int   = 5
    # ── loss ─────────────────────────────────────────────────────────────────
    margin:        float = 1.0
    ce_weight:     float = 0.1
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
    max_iters:     int   = 100000   # kept for compat; curriculum uses max_iter_per_phase
    max_iter_per_phase: int = 100000 # SOLO / COMBINED budget per segment
    check_every:   int   = 100
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
    """Parse stride_map string to tuple of ints, padding with 1s if short."""
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
    qk     = qk ** POWER_P                           # power attention kernel
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
    phi_k  = spow(k_s, POWER_P)                     # [B, NH, D_sym]
    phi_q  = spow(q,   POWER_P)                     # [B, NH, D_sym]
    kv     = jnp.einsum('bnd,bne->bnde', phi_k, v)  # [B, NH, D_sym, DH]
    C_t    = f_t[:,:,None,None]*C_prev + i_t[:,:,None,None]*kv
    n_t    = f_t[:,:,None]*n_prev + i_t[:,:,None]*phi_k
    h_num  = jnp.einsum('bnd,bnde->bne', phi_q, C_t)
    denom  = jnp.maximum(jnp.abs((phi_q*n_t).sum(axis=-1)), jnp.exp(-m_t))
    return h_num/(denom[:,:,None]+1e-8), C_t, n_t, m_t


# ============================================================================
# sLSTM cell (unchanged)
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
# Parameter structures  (conv_w removed; xLSTMBlockParams has no ffn)
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

class sLSTMParams(NamedTuple):
    W_i:   jnp.ndarray
    W_f:   jnp.ndarray
    W_z:   jnp.ndarray
    W_o:   jnp.ndarray
    Ry:    jnp.ndarray
    b:     jnp.ndarray
    gn_w:  jnp.ndarray
    gn_b:  jnp.ndarray

class xLSTMBlockParams(NamedTuple):
    norm_w:  jnp.ndarray
    norm_b:  jnp.ndarray
    mlstm: Optional[mLSTMParams] = None
    slstm: Optional[sLSTMParams] = None

class xLSTMParams(NamedTuple):
    embedding: jnp.ndarray
    blocks:    list
    norm_w:    jnp.ndarray
    norm_b:    jnp.ndarray


# ============================================================================
# HiRA structures  (ffn adapters removed)
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
    )

def apply_slstm_hira(p: sLSTMParams, l: sLSTMHiRA) -> sLSTMParams:
    return sLSTMParams(
        W_i=apply_hira(p.W_i, l.W_i), W_f=apply_hira(p.W_f, l.W_f),
        W_z=apply_hira(p.W_z, l.W_z), W_o=apply_hira(p.W_o, l.W_o),
        Ry=p.Ry, b=p.b, gn_w=p.gn_w, gn_b=p.gn_b,
    )

def apply_block_hira(base: xLSTMBlockParams, hira: BlockHiRA) -> xLSTMBlockParams:
    return xLSTMBlockParams(
        norm_w=base.norm_w, norm_b=base.norm_b,
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
# mLSTM layers  (no conv: x_conv_a = silu(x_m))
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
    xa      = jax.nn.silu(x_m)
    q,k,v,i_p,f_p = _mlstm_preacts(p, xa, x_m)
    h       = mlstm_cell_parallel(q, k, v, i_p, f_p)
    h_flat  = h.transpose(0,2,1,3).reshape(B, S, inner)
    h_out   = (h_flat + p.skip*xa) * jax.nn.silu(z)
    h_norm  = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T)

def mlstm_layer_step(p: mLSTMParams, x_t: jnp.ndarray, state, num_heads: int):
    C, n, m = state
    B     = x_t.shape[0]
    inner = p.W_up.shape[0] // 2
    xz    = jnp.dot(x_t, p.W_up.T)
    x_m, z = xz[...,:inner], xz[...,inner:]
    xa    = jax.nn.silu(x_m)
    q,k,v,i_p,f_p = _mlstm_preacts_step(p, xa, x_m)
    h, C_t, n_t, m_t = mlstm_cell_step(q, k, v, i_p, f_p, C, n, m)
    h_out = (h.reshape(B, inner) + p.skip*xa) * jax.nn.silu(z)
    h_norm= multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T), (C_t, n_t, m_t)

def mlstm_layer_parallel_with_init(p, x, num_heads, state):
    C0, n0, m0 = state
    B, S, _ = x.shape
    inner   = p.W_up.shape[0] // 2
    xz      = jnp.dot(x, p.W_up.T)
    x_m, z  = xz[...,:inner], xz[...,inner:]
    xa      = jax.nn.silu(x_m)
    q,k,v,i_p,f_p = _mlstm_preacts(p, xa, x_m)
    NH  = q.shape[1]
    DH  = q.shape[-1]              # get DH from q, not C0 (C0.shape[2] = D_sym now)
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
    phi_k  = spow(k_s, POWER_P)   # [B, NH, S, D_sym]
    phi_q  = spow(q,   POWER_P)   # [B, NH, S, D_sym]
    qk     = jnp.einsum('bnsd,bntd->bnst', q, k_s) ** POWER_P
    h_num  = (jnp.einsum('bnst,bntd->bnsd', qk*D_seq, v)
              + jnp.einsum('bnsd,bnde->bnse', phi_q, C0)*d_init[:,:,:,None])
    n_full = (jnp.einsum('bnst,bntd->bnsd', D_seq, phi_k)
              + n0[:,:,None,:]*d_init[:,:,:,None])
    qn     = (phi_q*n_full).sum(axis=-1)
    denom  = jnp.maximum(jnp.abs(qn), jnp.exp(-max_log_D[:,:,:,0]))
    h      = h_num / (denom[:,:,:,None]+1e-8)
    w_T  = D_seq[:,:,-1,:]
    d_T  = d_init[:,:,-1]
    C_T  = jnp.einsum('bns,bnsd,bnse->bnde', w_T, phi_k, v) + d_T[:,:,None,None]*C0
    n_T  = n_full[:,:,-1,:]
    m_T  = max_log_D[:,:,-1,0]
    h_flat = h.transpose(0,2,1,3).reshape(B,S,inner)
    h_out  = (h_flat + p.skip*xa)*jax.nn.silu(z)
    h_norm = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T), (C_T, n_T, m_T)


# ============================================================================
# sLSTM layers  (no conv: x_conv_a = silu(x))
# ============================================================================

def slstm_layer(p: sLSTMParams, x: jnp.ndarray, num_heads: int):
    B = x.shape[0]
    H = p.W_i.shape[0]
    xa = jax.nn.silu(x)
    gates = jnp.concatenate([
        jnp.dot(xa, p.W_i.T), jnp.dot(xa, p.W_f.T),
        jnp.dot(x,  p.W_z.T), jnp.dot(x,  p.W_o.T),
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
    sy, sc, sn, sm = state
    xa = jax.nn.silu(x_t)
    gate_t = jnp.concatenate([
        jnp.dot(xa, p.W_i.T), jnp.dot(xa, p.W_f.T),
        jnp.dot(x_t, p.W_z.T), jnp.dot(x_t, p.W_o.T),
    ], axis=-1)
    raw = gate_t + jnp.dot(sy, p.Ry.T) + p.b
    y_t, c_t, n_t, m_t = slstm_cell(raw, sc, sn, sm)
    return multihead_layer_norm(y_t, p.gn_w, p.gn_b, num_heads), (y_t,c_t,n_t,m_t)

def slstm_layer_with_init(p, x, num_heads, state):
    y0, c0, n0, m0 = state
    xa = jax.nn.silu(x)
    gates = jnp.concatenate([
        jnp.dot(xa, p.W_i.T), jnp.dot(xa, p.W_f.T),
        jnp.dot(x,  p.W_z.T), jnp.dot(x,  p.W_o.T),
    ], axis=-1)
    def step(state, gate_t):
        sy,sc,sn,sm = state
        y_t,c_t,n_t,m_t = slstm_cell(gate_t+jnp.dot(sy,p.Ry.T)+p.b, sc,sn,sm)
        return (y_t,c_t,n_t,m_t), y_t
    (y_T,c_T,n_T,m_T),ys = jax.lax.scan(step,(y0,c0,n0,m0),gates.transpose(1,0,2))
    out = multihead_layer_norm(ys.transpose(1,0,2), p.gn_w, p.gn_b, num_heads)
    return out, (y_T, c_T, n_T, m_T)


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
            cell_out_sub = (mlstm_layer_parallel(block.mlstm, x_sub, num_heads)
                            if btype == 'm' else slstm_layer(block.slstm, x_sub, num_heads))
            cell_out = jnp.repeat(cell_out_sub, stride, axis=1)[:, :S, :]
        else:
            cell_out = (mlstm_layer_parallel(block.mlstm, x_n, num_heads)
                        if btype == 'm' else slstm_layer(block.slstm, x_n, num_heads))
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
                cell_out_sub, new_lstm = mlstm_layer_parallel_with_init(
                    block.mlstm, x_sub, num_heads, lstm_state)
            else:
                cell_out_sub, new_lstm = slstm_layer_with_init(
                    block.slstm, x_sub, num_heads, lstm_state)
            cell_out = jnp.repeat(cell_out_sub, stride, axis=1)[:, :S, :]
        else:
            if btype == 'm':
                cell_out, new_lstm = mlstm_layer_parallel_with_init(
                    block.mlstm, x_n, num_heads, lstm_state)
            else:
                cell_out, new_lstm = slstm_layer_with_init(
                    block.slstm, x_n, num_heads, lstm_state)
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
                cell_out, new_lstm = slstm_layer_step(block.slstm, x_n, lstm_state, num_heads)
            new_states.append((new_lstm, cell_out))
        else:
            # fire if at a stride boundary; otherwise hold last output
            def fire(args):
                x_n_, lstm_state_ = args
                if btype == 'm':
                    return mlstm_layer_step(block.mlstm, x_n_, lstm_state_, num_heads)
                else:
                    return slstm_layer_step(block.slstm, x_n_, lstm_state_, num_heads)

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
# Single TBPTT step with carry-in state
# ============================================================================

@functools.partial(jax.jit, static_argnames=["config"])
def grad_chunk_persistent(base_xlstm, params, inputs, targets, layer_states, config):
    oh        = config.output_heads
    ob        = config.output_bits
    num_heads = config.num_heads
    block_map = config.block_map
    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    margin    = config.margin
    ce_weight = config.ce_weight

    def loss_fn(p):
        logits_flat, new_states = forward_train_chunked(
            base_xlstm, p, inputs, layer_states, num_heads, block_map, stride_map)
        logits = _reshape_logits(logits_flat, oh, ob)
        loss = sum(
            training_loss(logits[..., h, :], targets[..., h], margin, ce_weight)
            for h in range(oh)
        ) / oh
        return loss, new_states

    (loss, new_states), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, grads, new_states


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
            inner = config.d_model
            D_sym = _dsym(DH, config.power_p)
            lstm_state = (
                jnp.zeros((B, NH, D_sym, DH), bf),  # C  [D_sym × DH]
                jnp.zeros((B, NH, D_sym),     bf),  # n  [D_sym]
                jnp.zeros((B, NH),            bf),  # m
            )
        else:
            H = config.d_model
            lstm_state = (
                jnp.zeros((B, H), bf),  # y
                jnp.zeros((B, H), bf),  # c
                jnp.zeros((B, H), bf),  # n
                jnp.zeros((B, H), bf),  # m
            )
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
    skip = jnp.ones((inner,), DTYPE)
    ln_w = jnp.ones((inner,), DTYPE)
    ln_b = jnp.zeros((inner,), DTYPE)
    return mLSTMParams(
        W_up=W_up, W_q=W_q, W_k=W_k, W_v=W_v,
        W_i=W_i, W_f=W_f, b_f=b_f, b_i=b_i,
        skip=skip, ln_w=ln_w, ln_b=ln_b,
        W_down=W_down,
    )

def init_slstm_params(key, d_model, seq_len) -> sLSTMParams:
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
    gn_w = jnp.ones((H,),  DTYPE)
    gn_b = jnp.zeros((H,), DTYPE)
    return sLSTMParams(W_i=W_i, W_f=W_f, W_z=W_z, W_o=W_o,
                       Ry=Ry, b=b, gn_w=gn_w, gn_b=gn_b)

def init_xlstm_block_params(key, d_model, num_heads,
                             seq_len, btype) -> xLSTMBlockParams:
    ks = jr.split(key, 2)
    return xLSTMBlockParams(
        norm_w=jnp.ones((d_model,), DTYPE),
        norm_b=jnp.zeros((d_model,), DTYPE),
        mlstm=init_mlstm_params(ks[0], d_model, num_heads, seq_len) if btype=='m' else None,
        slstm=init_slstm_params(ks[1], d_model, seq_len)            if btype=='s' else None,
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
    inner, NH = d, config.num_heads
    ks = jr.split(key, 8); ki = 0
    if btype == 'm':
        mhira = mLSTMHiRA(
            W_up  =init_hira_adapter(ks[ki],   2*inner, d,     r),
            W_q   =init_hira_adapter(ks[ki+1], inner,   inner, r),
            W_k   =init_hira_adapter(ks[ki+2], inner,   inner, r),
            W_v   =init_hira_adapter(ks[ki+3], inner,   inner, r),
            W_i   =init_hira_adapter(ks[ki+4], NH,      inner, r),
            W_f   =init_hira_adapter(ks[ki+5], NH,      inner, r),
            W_down=init_hira_adapter(ks[ki+6], d,       inner, r),
        ); shira = None
    else:
        shira = sLSTMHiRA(
            W_i=init_hira_adapter(ks[ki],   d, d, r),
            W_f=init_hira_adapter(ks[ki+1], d, d, r),
            W_z=init_hira_adapter(ks[ki+2], d, d, r),
            W_o=init_hira_adapter(ks[ki+3], d, d, r),
        ); mhira = None
    return BlockHiRA(mlstm=mhira, slstm=shira)

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

def sinkgd_update(state, grads, params, lr, weight_decay=0.0, max_norm=1.0, L=1):
    return _sinkgd_step(grads, params,
                        jnp.array(lr), jnp.array(weight_decay), jnp.array(max_norm), L), state


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

    # auto-extend stride_map to match block_map length
    stride_map = cfg['stride_map']
    if len(stride_map) < cfg['num_layers']:
        stride_map = stride_map + "1" * (cfg['num_layers'] - len(stride_map))
    cfg['stride_map'] = stride_map[:cfg['num_layers']]

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
    print(f"Curriculum segments: {n_segs} × up to {config.segment_size} bytes")
    print(f"max_iter_per_phase={config.max_iter_per_phase}  check_every={config.check_every}  lr={lr}")
    print(f"Residual budget: {budget_bytes} bytes  ({max_entries} correction entries)")
    print()

    t_start      = time.perf_counter()
    global_iters = 0
    completed    = []     # list of byte segments that passed both phases
    failed_seg   = None

    def _tbptt_phase(phase_name, inputs, targets, eval_segs, p, opt_st):
        """TBPTT curriculum phase. Cycles chunks in order; resets state each epoch.
        Returns (success, iters_done, last_accs, updated_params, updated_opt_state).
        """
        nonlocal global_iters
        n_chunks  = inputs.shape[0]
        chunk_idx = 0
        states    = init_step_states(config, batch_size=1)
        last_accs = None
        success   = False
        iters_done = 0

        pbar = tqdm(range(1, config.max_iter_per_phase + 1), desc=phase_name,
                    unit="it", file=sys.stderr, dynamic_ncols=True)
        for it in pbar:
            if chunk_idx == 0:                          # epoch boundary → reset state
                states = init_step_states(config, batch_size=1)

            inp = inputs [chunk_idx][None, :]
            tgt = targets[chunk_idx][None, :]
            chunk_idx = (chunk_idx + 1) % n_chunks

            frozen_sg = jax.lax.stop_gradient(states)
            loss, grads, states = grad_chunk_persistent(
                base_xlstm, p, inp, tgt, frozen_sg, config)
            p, opt_st = sinkgd_update(
                opt_st, grads, p, lr=lr,
                weight_decay=config.weight_decay, max_norm=config.grad_clip_norm,
                L=config.sinkgd_l)
            global_iters += 1
            iters_done    = it

            bpc = math.exp(float(loss)) / math.log(2)
            pbar.set_postfix(loss=f"{float(loss):.4f}", bpc=f"{bpc:.3f}")

            if it % config.check_every == 0 or it == config.max_iter_per_phase:
                accs      = eval_segments_stateful(base_xlstm, p, config, eval_segs)
                last_accs = accs
                min_acc   = min(a for a, _ in accs)
                fw_str    = next((str(fw) for _, fw in accs if fw is not None), "ok")
                print()
                pbar.set_description(f"{phase_name} acc={min_acc:.1%} fw={fw_str}")
                if all(a == 1.0 for a, _ in accs):
                    success = True
                    pbar.close()
                    break
        else:
            pbar.close()

        return success, iters_done, last_accs, p, opt_st

    for seg_idx, seg in enumerate(segs):
        t_seg   = time.perf_counter()
        byte_lo = seg_idx * config.segment_size
        print(f"\nSEGMENT {seg_idx+1}/{n_segs}  bytes [{byte_lo}, {byte_lo+len(seg)})  len={len(seg)}")

        # ── SOLO phase ────────────────────────────────────────────────────────
        seg_inp, seg_tgt = make_chunks_and_targets(seg, config)
        print(f"[SOLO]     chunks={seg_inp.shape[0]}  budget={config.max_iter_per_phase}")
        solo_ok, solo_iters, solo_accs, params, opt_state = _tbptt_phase(
            f"solo s{seg_idx+1}", seg_inp, seg_tgt, [seg], params, opt_state)

        acc_s  = solo_accs[0][0] if solo_accs else 0.0
        fw_s   = solo_accs[0][1] if solo_accs else None
        fw_str = f"byte {fw_s}" if fw_s is not None else "ok"
        elapsed = time.perf_counter() - t_seg
        if solo_ok:
            print(f"[PASS] solo  seg{seg_idx+1}  acc=100%  iters={solo_iters}  {elapsed:.1f}s")
        else:
            print(f"[FAIL] solo  seg{seg_idx+1}  acc={acc_s:.2%}  first_wrong={fw_str}"
                  f"  iters={solo_iters}  {elapsed:.1f}s")
            failed_seg = seg_idx + 1
            break

        completed.append(seg)

        if len(completed) == 1:
            continue     # no COMBINED phase for the very first segment

        # ── COMBINED phase ────────────────────────────────────────────────────
        combined = np.concatenate(completed)
        comb_inp, comb_tgt = make_chunks_and_targets(combined, config)
        n_done = len(completed)
        print(f"[COMBINED] segs 1..{n_done}  chunks={comb_inp.shape[0]}  budget={config.max_iter_per_phase}")
        comb_ok, comb_iters, comb_accs, params, opt_state = _tbptt_phase(
            f"comb 1..{n_done}", comb_inp, comb_tgt, completed, params, opt_state)

        min_acc = min(a for a, _ in comb_accs) if comb_accs else 0.0
        elapsed = time.perf_counter() - t_seg
        if comb_ok:
            print(f"[PASS] comb  1..{n_done}  acc=100%  iters={comb_iters}  {elapsed:.1f}s")
        else:
            print(f"[FAIL] comb  1..{n_done}  min_acc={min_acc:.2%}"
                  f"  iters={comb_iters}  {elapsed:.1f}s")
            failed_seg = seg_idx + 1
            break

        save_checkpoint(os.path.join(log_dir, "ckpt_last"), params, config, seg_idx, "combined")

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    stop_reason   = (f"FAIL at segment {failed_seg}" if failed_seg is not None
                     else f"all {n_segs} segments mastered")
    print(f"\n[STOP] {stop_reason}  total_iters={global_iters}  elapsed={elapsed_total:.1f}s")

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
