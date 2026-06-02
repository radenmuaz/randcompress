"""
randcompress v8.1

Algorithm
---------
Split dataset into fixed-size segments. For each new segment k:
  SOLO     phase — train on segment k alone until 100% TF-accuracy (or FAIL).
  COMBINED phase — train on segments 0..k together until 100% on all (or FAIL).

Per-segment TF-accuracy and first-fail-byte are reported before/after each phase
to track forgetting. Checkpoint saved after every successful phase.
config.max_iter_per_phase is the fixed budget; exceeding it declares FAIL and stops.
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
    # ── tokenisation (new v8) ─────────────────────────────────────────────────
    # input_bits  → input_vocab  = 2**input_bits  (embedding table)
    # output_bits → output_vocab = 2**output_bits (per-head softmax)
    # output_heads: predictions fired per input step
    # vocab_size is DERIVED = 2**input_bits; stored for checkpoint compat
    #
    # preset table (--preset <name>):
    #   byte         8  8  1   sequence 1×   standard v7
    #   nibble        4  4  1   sequence 2×   16-class per step
    #   binary        1  1  1   sequence 8×   2-class per step
    #   byte2bits     8  1  8   sequence 1×   asymmetric: byte in, 8 binary heads out
    #   byte2nibbles  8  4  2   sequence 1×   asymmetric: byte in, 2 nibble heads out
    #   mtp_binary    1  1  8   sequence 8×   binary seq, predict next 8 bits at once
    input_bits:    int   = 8
    output_bits:   int   = 8
    output_heads:  int   = 1
    vocab_size:    int   = 256    # derived = 2**input_bits; set by _parse_config
    # ── architecture ─────────────────────────────────────────────────────────
    d_model:       int   = 128
    num_heads:     int   = 4
    d_ff:          int   = 256
    num_layers:    int   = 4      # always overridden to len(block_map)
    segment_size:  int   = 1024   # in BYTES (converted to tokens in train loop)
    seed:          int   = 0
    conv_kernel:   int   = 4
    block_map:     str   = "msms"
    lora_r:        int   = 16
    batch_size:    int   = 1
    # ── optimiser ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-2
    weight_decay:  float = 0.0
    grad_clip_norm:float = 1e6
    random_shuffle:bool  = False
    sinkgd_l:      int   = 1
    # ── loss ─────────────────────────────────────────────────────────────────
    margin:        float = 1.0    # 0.0 → pure CE
    ce_weight:     float = 0.1
    # ── PI integral action (v8.1) ─────────────────────────────────────────────
    lambda_i:      float = 0.5    # integral gain; 0.0 → disables PI (uniform weights)
    w_max:         float = 10.0   # max per-position weight (clips integral explosion)
    # ── Lyapunov stability regularizer (v8.1) ────────────────────────────────
    alpha_l:       float = 0.01   # Lyapunov weight; 0.0 → disables stability term
    # ── residual ─────────────────────────────────────────────────────────────
    residual_budget: float = 0.01  # fraction of dataset bytes; 0.0 → disabled
    # residual_budget: float = 0.0  # fraction of dataset bytes; 0.0 → disabled
    # ── misc ─────────────────────────────────────────────────────────────────
    pad_token:     int   = 0      # -1 → masking disabled (use for sub-byte modes)
    dtype:         str   = "float32"
    max_iter_per_phase: int = 50000
    check_every:   int   = 500
    # dataset:       str   = "datasets/juz1.txt"
    dataset:       str   = "datasets/quran-uthmani.txt"

# preset table ──────────────────────────────────────────────────────────────
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

# ── tokenisation helpers ───────────────────────────────────────────────────

def bytes_to_tokens(raw: np.ndarray, bits: int) -> np.ndarray:
    """Split each byte into 8//bits sub-tokens. bits ∈ {8,4,1}."""
    if bits == 8: return raw.astype(np.int32)
    if bits == 4:
        hi = ((raw >> 4) & 0xF).astype(np.int32)
        lo = (raw & 0xF).astype(np.int32)
        return np.stack([hi, lo], axis=-1).ravel()
    # bits == 1
    return np.unpackbits(raw).astype(np.int32)

def tokens_to_bytes(toks: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of bytes_to_tokens."""
    if bits == 8: return toks.astype(np.uint8)
    if bits == 4:
        hi = (toks[0::2].astype(np.uint8) & 0xF) << 4
        lo = toks[1::2].astype(np.uint8) & 0xF
        return hi | lo
    return np.packbits(toks.astype(np.uint8))

def compute_targets_np(tokens: np.ndarray, config) -> np.ndarray:
    """
    Build target array for training from the input-space token sequence.
    Returns shape [N_steps, output_heads] int32.
    For symmetric modes: standard next-token shift.
    For asymmetric (input_bits=8, output_bits<8): decompose next byte per head.
    For MTP (output_heads>1, symmetric): stack next H tokens per step.
    """
    output_mask = (1 << config.output_bits) - 1
    N = len(tokens) - 1  # number of prediction steps (need at least 1 lookahead)

    if config.input_bits == config.output_bits and config.output_heads == 1:
        # standard next-token
        return tokens[1:N+1].reshape(N, 1)

    if config.input_bits == 8 and config.output_bits < 8:
        # asymmetric: tokens are bytes; decompose each next byte into heads
        next_bytes = tokens[1:N+1].astype(np.int32)
        heads = np.stack(
            [(next_bytes >> (h * config.output_bits)) & output_mask
             for h in range(config.output_heads)],
            axis=-1)
        return heads  # [N, output_heads]

    # MTP symmetric: predict next output_heads tokens
    # clip N so indices don't exceed
    N_mtp = len(tokens) - config.output_heads
    heads = np.stack(
        [tokens[1+h : 1+h+N_mtp] for h in range(config.output_heads)],
        axis=-1)
    return heads  # [N_mtp, output_heads]

# ── config validation ──────────────────────────────────────────────────────

_WARNINGS = []

def validate_config(config, n_dataset_bytes: int):
    """Print warnings for bad / expensive config combinations."""
    warns = []
    ib, ob, oh = config.input_bits, config.output_bits, config.output_heads

    # derived
    input_vocab  = 2 ** ib
    output_vocab = 2 ** ob
    tokens_per_byte = 8 // ib
    bits_per_step   = ob * oh

    if input_vocab != config.vocab_size:
        warns.append(f"vocab_size={config.vocab_size} ≠ 2**input_bits={input_vocab}; will be forced to {input_vocab}")
    if ib not in (1, 4, 8):
        warns.append(f"input_bits={ib} not in {{1,4,8}} — unsupported")
    if ob not in (1, 4, 8):
        warns.append(f"output_bits={ob} not in {{1,4,8}} — unsupported")
    if bits_per_step < ib:
        warns.append(f"output_bits*output_heads={bits_per_step} < input_bits={ib}: "
                     f"each step produces fewer bits than it consumes — can't reconstruct bytes")
    if ib < 8 and config.pad_token == 0:
        warns.append(f"input_bits={ib} (sub-byte) but pad_token=0: token 0 is valid — "
                     f"masking will silently drop real data; set pad_token=-1")
    if tokens_per_byte > 1:
        seq_len = n_dataset_bytes * tokens_per_byte
        if seq_len > 500_000:
            warns.append(f"input_bits={ib} → sequence length {seq_len:,} tokens "
                         f"({tokens_per_byte}× dataset) — training will be very slow")
    param_bytes = (config.d_model * config.d_ff * 2 +
                   config.d_model * input_vocab +
                   config.d_model * output_vocab * oh) * 4 * config.num_layers
    if param_bytes > n_dataset_bytes * 10:
        warns.append(f"param estimate ~{param_bytes//1024}KB >> dataset {n_dataset_bytes//1024}KB "
                     f"— heavily over-parameterised, compression impossible at this scale")
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

def _reshape_logits(flat, output_heads, output_bits):
    """flat: [..., output_heads*output_vocab] → [..., output_heads, output_vocab]"""
    output_vocab = 2 ** output_bits
    return flat.reshape(*flat.shape[:-1], output_heads, output_vocab)


# ============================================================================
# Exact BPTT step (scan + remat over all chunks)
# ============================================================================

@jax.jit(static_argnames=["config"])
def exact_bptt_step(base_xlstm, params, all_inputs, all_targets, chunk_weights, config):
    """
    all_inputs:    [num_chunks, chunk_size] int32
    all_targets:   [num_chunks, chunk_size, output_heads] int32  precomputed
    chunk_weights: [num_chunks, chunk_size] float32  PI integral weights
    """
    num_heads = config.num_heads
    block_map = config.block_map
    margin    = config.margin
    ce_weight = config.ce_weight
    alpha_l   = config.alpha_l
    oh        = config.output_heads
    ob        = config.output_bits

    def loss_fn(p):
        init_states = init_step_states(config, batch_size=1)
        xlstm_eff   = apply_xlstm_hira(base_xlstm, p)
        embedding   = xlstm_eff.embedding

        @jax.checkpoint
        def scan_body(states, xs):
            inputs, tgt, weights = xs              # [S], [S, oh], [S]
            logits_flat, new_states = forward_train_chunked(
                base_xlstm, p, inputs[None, :], states, num_heads, block_map)
            logits = _reshape_logits(logits_flat, oh, ob)  # [1, S, oh, ov]
            tgt_b  = tgt[None, :]                          # [1, S, oh]

            L_primary = weighted_training_loss(logits, tgt_b, weights, margin, ce_weight)
            L_stable  = lyapunov_loss_cheap(logits, tgt_b, embedding, config)
            return new_states, L_primary + L_stable

        _, per_chunk_losses = jax.lax.scan(
            scan_body, init_states, (all_inputs, all_targets, chunk_weights))
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
            config.conv_kernel, config.segment_size, config.block_map[i]
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
    # A: orthonormal rows via SVD. B: zero-init (ΔW=0 at start).
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
    out_dim = config.output_heads * (2 ** config.output_bits)  # total output width
    return RandCompressParams(
        output_proj=_bf(np.array(jr.normal(ks[0], (config.d_model, out_dim))) * 0.01),
        emb_hira   =init_hira_adapter(ks[1], config.vocab_size, config.d_model, config.lora_r),
        blocks     =[init_block_hira(ks[2+i], config, config.block_map[i])
                     for i in range(config.num_layers)],
    )


# ============================================================================
# SinkGD  (Scetbon et al., 2025 — arXiv:2502.06742)
# Stateless optimizer: SR-Sinkhorn normalises each gradient matrix then SGD.
#
# SR-Sinkhorn(W, L): for ℓ=1..L alternate
#   X ← √n · Q(X)⁻¹ X      (each row → ℓ₂-norm √n)
#   X ← √m · X R(X)⁻¹      (each col → ℓ₂-norm √m)
# At convergence: ‖X*_{i,:}‖₂ = √n  and  ‖X*_{:,j}‖₂ = √m  ∀ i, j
# ============================================================================

def _sr_sinkhorn_2d(W, L):
    """SR-Sinkhorn on a 2-D float32 matrix W ∈ ℝ^{m×n}.
    Python loop unrolled at JIT-trace time — no fori_loop overhead."""
    m, n = W.shape
    for _ in range(L):
        W = jnp.sqrt(n) * W / (jnp.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
        W = jnp.sqrt(m) * W / (jnp.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    return W

def _sinkgd_normalize(g, L):
    """Apply SR-Sinkhorn to any gradient tensor."""
    g32 = g.astype(jnp.float32)
    if g32.ndim == 0:
        return g32
    if g32.ndim == 1:
        n = g32.shape[0]
        return jnp.sqrt(n) * g32 / (jnp.linalg.norm(g32) + 1e-8)
    m = g32.shape[0]
    n = int(np.prod(g32.shape[1:]))
    return _sr_sinkhorn_2d(g32.reshape(m, n), L).reshape(g32.shape)

def sinkgd_init(params):
    return None   # stateless

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
    """Multiclass hinge loss — directly penalises argmax failures.
    loss_t = max(0, max_{j≠y} logit_j − logit_y + margin)
    Zero once correct token leads by ≥ margin. No gradient when already right.
    Cross-entropy needs BPC→0 for 100% argmax; this only needs margin>0 (higher BPC ok).
    """
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    # mask target class with -inf so max below ignores it
    neg_inf       = jax.nn.one_hot(targets, logits.shape[-1]) * 1e9
    best_wrong    = (logits - neg_inf).max(axis=-1)
    loss_per_tok  = jnp.maximum(0.0, best_wrong - target_logits + margin)
    mask          = (targets != PAD_TOKEN).astype(jnp.float32)
    return (loss_per_tok * mask).sum() / (mask.sum() + 1e-8)

def training_loss(logits, targets, margin=1.0, ce_weight=0.05):
    """Primary: argmax margin.  Small CE term keeps gradients smooth near zero loss.
    margin=0.0  → pure cross-entropy (degenerates to v6 behaviour).
    ce_weight=0.0 → pure margin loss, no CE regularizer.
    """
    if margin == 0.0:
        return cross_entropy_loss(logits, targets)
    ce = cross_entropy_loss(logits, targets)
    mg = argmax_margin_loss(logits, targets, margin)
    return mg + ce_weight * ce

def weighted_training_loss(logits, targets, weights, margin=1.0, ce_weight=0.05):
    """Per-token loss weighted by PI integral importance.
    weights: [S] float32, per input-step importance (≥1.0).
    lambda_i=0.0 → all weights==1.0 → degenerates to training_loss.
    """
    ob = logits.shape[-1]
    oh = logits.shape[-2] if logits.ndim == 4 else 1
    loss_per_head = []
    for h in range(oh):
        lg = logits[..., h, :] if oh > 1 else logits
        tg = targets[..., h]   if oh > 1 else targets
        # per-token margin loss (not averaged yet)
        if margin == 0.0:
            log_probs   = jax.nn.log_softmax(lg, axis=-1)
            per_tok_ce  = -jnp.take_along_axis(log_probs, tg[..., None], axis=-1)[..., 0]
            per_tok     = per_tok_ce
        else:
            tgt_logit   = jnp.take_along_axis(lg, tg[..., None], axis=-1)[..., 0]
            neg_inf_mask= jax.nn.one_hot(tg, lg.shape[-1]) * 1e9
            best_wrong  = (lg - neg_inf_mask).max(axis=-1)
            per_tok_mg  = jnp.maximum(0.0, best_wrong - tgt_logit + margin)
            log_probs   = jax.nn.log_softmax(lg, axis=-1)
            per_tok_ce  = -jnp.take_along_axis(log_probs, tg[..., None], axis=-1)[..., 0]
            per_tok     = per_tok_mg + ce_weight * per_tok_ce
        mask = (tg != PAD_TOKEN).astype(jnp.float32)
        # apply PI weights (broadcast over batch dim if needed)
        w = weights[None, :] if per_tok.ndim == 2 else weights
        loss_per_head.append((per_tok * mask * w).sum() / (mask * w).sum().clip(1e-8))
    return sum(loss_per_head) / oh

def lyapunov_loss_cheap(logits, targets, embedding, config):
    """Embedding-distance proxy for hidden-state divergence.
    Penalises positions where argmax ≠ target by the embedding gap
    ||embed(pred) - embed(target)||².
    Cheap: no second forward pass. Captures first-order cascade risk.
    alpha_l=0.0 → disabled.
    """
    if config.alpha_l == 0.0:
        return jnp.array(0.0)
    oh, ob = config.output_heads, config.output_bits
    out_mask = (1 << ob) - 1
    total = jnp.array(0.0)
    for h in range(oh):
        lg = logits[..., h, :] if oh > 1 else logits   # [B, S, V]
        tg = targets[..., h]   if oh > 1 else targets   # [B, S]
        preds = jax.lax.stop_gradient(jnp.argmax(lg, axis=-1))
        wrong = (preds != tg).astype(jnp.float32)
        # for byte-level: map pred/tgt tokens → embedding indices
        emb_gap = jnp.sum((embedding[preds] - embedding[tg])**2, axis=-1)
        total  += jnp.mean(emb_gap * wrong)
    return config.alpha_l * total / oh

# ── PI integral state (lives outside JIT) ─────────────────────────────────

def make_integral_weights(n_bytes: int) -> np.ndarray:
    return np.ones(n_bytes, dtype=np.float32)

def update_integral(weights, seg_accs, seg_byte_offsets, seg_lengths, config):
    """
    After each eval: add lambda_i to weights from first-fail byte onwards in
    each segment. Clips to w_max. lambda_i=0 → no-op.
    """
    if config.lambda_i == 0.0:
        return weights
    for (acc, fail_at), offset, seg_len in zip(seg_accs, seg_byte_offsets, seg_lengths):
        if fail_at is not None:
            lo = offset + fail_at
            hi = offset + seg_len
            weights[lo:hi] = np.minimum(weights[lo:hi] + config.lambda_i, config.w_max)
    return weights


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

def split_segments(raw_bytes, segment_size):
    """Split raw byte array into segments of segment_size bytes."""
    segs, n, i = [], len(raw_bytes), 0
    while i < n:
        segs.append(raw_bytes[i : i + segment_size])
        i += segment_size
    return segs

def make_chunks_array(raw_bytes, config):
    """
    Tokenise raw_bytes with config.input_bits, then pack into
    [num_chunks, chunk_size+1] int32 array, zero-padded.
    chunk_size = config.segment_size * (8 // config.input_bits).
    """
    toks       = bytes_to_tokens(np.asarray(raw_bytes, dtype=np.uint8), config.input_bits)
    chunk_size = config.segment_size * (8 // config.input_bits)
    n          = len(toks)
    num_chunks = max(1, math.ceil((n - 1) / chunk_size))
    pad_len    = num_chunks * chunk_size + 1 - n
    padded     = np.concatenate([toks, np.zeros(pad_len, dtype=np.int32)])
    chunks     = [padded[i*chunk_size : i*chunk_size + chunk_size + 1]
                  for i in range(num_chunks)]
    return jnp.array(np.stack(chunks), dtype=jnp.int32)


def make_chunks_and_targets(raw_bytes, config):
    """
    Returns:
      all_inputs:  [num_chunks, chunk_size] int32
      all_targets: [num_chunks, chunk_size, output_heads] int32
    """
    toks       = bytes_to_tokens(np.asarray(raw_bytes, dtype=np.uint8), config.input_bits)
    chunk_size = config.segment_size * (8 // config.input_bits)
    n          = len(toks)
    num_chunks = max(1, math.ceil((n - 1) / chunk_size))
    oh         = config.output_heads
    ob         = config.output_bits
    out_mask   = (1 << ob) - 1

    pad_len = max(0, num_chunks * chunk_size + oh - n)
    padded  = np.concatenate([toks, np.zeros(pad_len + oh, dtype=np.int32)])

    all_inputs = np.stack([padded[i*chunk_size : i*chunk_size + chunk_size]
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
# Segment evaluation  (stateful: state carries from segment to segment)
# ============================================================================

def eval_segments_stateful(base_xlstm, params, config, seg_list):
    """
    Zero-state TF eval with state flowing across segments.
    Works in INPUT-TOKEN space. Reports byte-level accuracy.
    Returns list of (accuracy, fail_at_byte_idx) per segment.
    """
    tok_per_byte = 8 // config.input_bits
    chunk_size   = config.segment_size * tok_per_byte  # chunk in token units
    ob, oh       = config.output_bits, config.output_heads
    out_mask     = (1 << ob) - 1
    states       = init_step_states(config, batch_size=1)
    results      = []

    for seg_bytes in seg_list:
        seg   = bytes_to_tokens(np.array(seg_bytes, dtype=np.uint8), config.input_bits)
        n_tok = len(seg)
        n_b   = len(seg_bytes)
        if n_b < 2:
            results.append((1.0, None))
            continue

        num_chunks = max(1, math.ceil((n_tok - 1) / chunk_size))
        pad_len    = num_chunks * chunk_size + 1 - n_tok
        padded     = np.concatenate([seg, np.zeros(pad_len, dtype=np.int32)])

        byte_correct = []
        for ci in range(num_chunks):
            s     = ci * chunk_size
            chunk = padded[s : s + chunk_size + 1]
            inp   = jnp.array(chunk[None, :-1], dtype=jnp.int32)
            logits_flat, states = _fwd_chunked_jit(
                base_xlstm, params, inp, states, config.num_heads, config.block_map)
            logits = _reshape_logits(logits_flat, oh, ob)   # [1, S, oh, ov]
            preds  = np.array(jnp.argmax(logits[0], axis=-1))  # [S, oh]

            # reconstruct predicted bytes from heads
            if config.input_bits == 8 and ob < 8:
                # asymmetric: heads reconstruct next byte
                pred_bytes = np.zeros(preds.shape[0], np.uint8)
                for h in range(oh):
                    pred_bytes |= (preds[:, h].astype(np.uint8) & out_mask) << (h * ob)
                tgt_bytes = padded[s+1 : s+chunk_size+1].astype(np.uint8)
            elif oh == 1:
                # symmetric: tokens → bytes
                pred_toks = preds[:, 0]
                pred_bytes = tokens_to_bytes(pred_toks, config.input_bits)
                tgt_bytes  = tokens_to_bytes(padded[s+1:s+chunk_size+1], config.input_bits)
            else:
                # MTP: each head is a token prediction; use head-0 for accuracy
                pred_toks  = preds[:, 0]
                pred_bytes = tokens_to_bytes(pred_toks, config.input_bits)
                tgt_bytes  = tokens_to_bytes(padded[s+1:s+chunk_size+1], config.input_bits)

            # trim to actual segment length
            valid = min(len(pred_bytes), max(0, n_b - 1 - ci * (chunk_size // tok_per_byte)))
            byte_correct.extend((pred_bytes[:valid] == tgt_bytes[:valid]).tolist())

        byte_correct = byte_correct[:n_b - 1]
        n_valid = len(byte_correct)
        acc     = float(sum(byte_correct)) / max(n_valid, 1)
        fail_at = next((i for i, ok in enumerate(byte_correct) if not ok), None)
        results.append((acc, fail_at))

    return results


# ============================================================================
# Generative evaluation (seed → autoregressive decode, persistent state)
# ============================================================================

def eval_generative(base_xlstm, params, config, all_tokens):
    """
    Autoregressive decode with optional residual collection.

    residual_budget (from config):
      0.0  → no residual stored; pure argmax, cascade allowed.
      >0.0 → collect corrections up to budget bytes of residual storage.
             At each failure: store correction AND feed correct token (state
             correction prevents cascade while budget remains).
             Once budget exhausted: stop correcting, let cascade run.

    Returns (generated_np, bytes_wrong, accuracy, first_wrong, residual_dict).
    residual_dict is empty when residual_budget=0.0.
    """
    all_bytes    = np.array(all_tokens, dtype=np.uint8)  # always byte-level
    target_bytes = all_bytes[1:]
    n            = len(target_bytes)
    budget_bytes = int(config.residual_budget * n)
    bytes_per_entry = 4
    max_entries  = budget_bytes // bytes_per_entry if budget_bytes > 0 else 0

    ob, oh   = config.output_bits, config.output_heads
    out_mask = (1 << ob) - 1
    tok_per_byte = 8 // config.input_bits

    states   = init_step_states(config, batch_size=1)
    residual = {}

    input_toks = bytes_to_tokens(all_bytes, config.input_bits)
    cur_token  = jnp.array([int(input_toks[0])])

    if tok_per_byte > 1:
        # Sub-byte modes (nibble, binary): step at token granularity, reconstruct bytes.
        # No residual correction — operate purely autoregressively.
        pred_subtoks = []
        n_pred = n * tok_per_byte
        for t in tqdm(range(n_pred), desc="gen", unit="tok", leave=False, file=sys.stderr):
            logit_flat, states = _fwd_step_jit(
                base_xlstm, params, cur_token, states, config.num_heads, config.block_map)
            logits = _reshape_logits(logit_flat, oh, ob)   # [1, 1, ov]
            pred_tok  = int(jnp.argmax(logits[0, 0]))
            pred_subtoks.append(pred_tok)
            cur_token = jnp.array([pred_tok])
        gen_bytes = list(tokens_to_bytes(np.array(pred_subtoks, np.int32), config.input_bits))
    else:
        # Byte-level or asymmetric (byte2bits, byte2nibbles): one step per output byte.
        gen_bytes = []
        _pbar = tqdm(range(n), desc="gen", unit="B", unit_scale=True,
                     leave=False, file=sys.stderr)
        for byte_idx in _pbar:
            logit_flat, states = _fwd_step_jit(
                base_xlstm, params, cur_token, states, config.num_heads, config.block_map)
            logits = _reshape_logits(logit_flat, oh, ob)   # [1, oh, ov]

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

            gen_bytes.append(pred_byte)
            _pbar.set_postfix(wrong=len(residual))
        _pbar.close()

    generated   = np.array(gen_bytes, dtype=np.uint8)
    wrong_mask  = generated != target_bytes
    bytes_wrong = int(np.sum(wrong_mask))
    accuracy    = (n - bytes_wrong) / n
    wrong_idxs  = np.where(wrong_mask)[0]
    first_wrong = int(wrong_idxs[0]) if len(wrong_idxs) > 0 else None
    return generated, bytes_wrong, accuracy, first_wrong, residual


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

def _save_residual(ckpt_dir, residual: dict):
    """Save residual corrections. Empty dict → file still written (0 entries = no corrections)."""
    import pickle
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "residual.pkl"), "wb") as f:
        pickle.dump(residual, f)


# ============================================================================
# Main
# ============================================================================

def _parse_config():
    """Build Config from hardcoded defaults, optional --config json, then --key val overrides.
    Returns (config, diff_vs_defaults).
    Usage:
        python script.py [--config cfg.json] [--key val ...]
    """
    import json as _json

    # Field type map for casting CLI strings
    _types = {f: type(v) for f, v in Config()._asdict().items()}

    def _cast(field, s):
        t = _types[field]
        if t is bool:
            return s.lower() in ("1", "true", "yes")
        return t(s)

    # 1. Start from class defaults
    cfg = Config()._asdict()

    argv = sys.argv[1:]

    # 2. --preset <name> applies preset fields (before other overrides)
    if "--preset" in argv:
        idx = argv.index("--preset")
        name = argv[idx + 1]
        if name not in _PRESETS:
            print(f"  [CONFIG WARN] unknown preset '{name}'; known: {list(_PRESETS)}")
        else:
            cfg.update(_PRESETS[name])

    # 3. --config path.json overrides
    if "--config" in argv:
        idx = argv.index("--config")
        with open(argv[idx + 1]) as f:
            for k, v in _json.load(f).items():
                if k in cfg:
                    cfg[k] = v

    # 4. --key val overrides (highest priority)
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and argv[i] not in ("--config", "--preset"):
            field = argv[i][2:]
            if field in cfg and i + 1 < len(argv):
                cfg[field] = _cast(field, argv[i + 1])
                i += 2
                continue
        i += 1

    # derived: block_map → num_layers; input_bits → vocab_size
    cfg['num_layers'] = len(cfg['block_map'])
    cfg['vocab_size'] = 2 ** cfg['input_bits']

    config = Config(**cfg)

    # 4. Diff vs library defaults (Config())
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
    global PAD_TOKEN, DTYPE
    PAD_TOKEN = config.pad_token
    DTYPE     = jnp.dtype(config.dtype)

    # Save full config + diff to log dir
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

    raw_bytes = load_dataset(config.dataset)
    n_total   = len(raw_bytes)

    print(f"\nConfig validation:")
    warns = validate_config(config, n_total)
    if not warns:
        print("  OK")

    # tokenise dataset (sub-byte modes expand sequence; byte mode is a no-op)
    tokens  = bytes_to_tokens(raw_bytes, config.input_bits)
    tok_per_byte = 8 // config.input_bits
    print(f"Tokenisation: input_bits={config.input_bits} output_bits={config.output_bits} "
          f"output_heads={config.output_heads}  "
          f"seq={len(tokens)} tokens ({tok_per_byte}× bytes)")
    print(f"Dataset: {config.dataset}  ({n_total} bytes)")

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

    opt_state = sinkgd_init(params)
    lr        = config.learning_rate

    segs   = split_segments(raw_bytes, config.segment_size)  # byte-level segments
    n_segs = len(segs)
    print(f"Segments: {n_segs} × {config.segment_size} bytes  (last={len(segs[-1])} bytes)")
    print(f"config.max_iter_per_phase={config.max_iter_per_phase}  config.check_every={config.check_every}  lr={lr}")

    # Crossover: first segment where cumulative data bytes > param_bytes
    # (i.e. the first time storing these weights is actually smaller than the data seen so far)
    crossover_byte = param_bytes
    crossover_seg  = math.ceil(param_bytes / config.segment_size)  # 1-based segment index
    crossover_frac = param_bytes / n_total * 100
    if crossover_seg <= n_segs:
        print(f"Compression crossover at seg {crossover_seg}/{n_segs}  "
              f"(byte {crossover_byte:.0f} / {n_total}  =  {crossover_frac:.1f}% of dataset)")
    else:
        print(f"Compression crossover: never within dataset  "
              f"(param_bytes={param_bytes:.0f} > dataset={n_total} bytes)")
    print()

    # Per-phase summary table rows
    phase_summary = []

    # ── PI integral weights (v8.1) ─────────────────────────────────────────
    integral_weights = make_integral_weights(n_total)  # [N_bytes] float32, starts 1.0
    # precompute byte offsets for each segment
    seg_byte_offsets = [i * config.segment_size for i in range(len(segs))]

    completed = []  # segments that passed solo+combined
    t_run_start = time.perf_counter()
    outer_pbar  = tqdm(total=n_total, desc="compress", unit="B", unit_scale=True,
                       unit_divisor=1024, file=sys.stderr, dynamic_ncols=True)

    def _print_time(seg_elapsed, run_elapsed):
        done = sum(len(s) for s in completed)
        print(f"[time] seg={seg_elapsed:.1f}s  total={run_elapsed:.1f}s  "
              f"bytes={done} ({done/1e6:.3f}M  {done/n_total*100:.1f}% of {n_total})")

    for seg_idx in range(n_segs):
        seg         = segs[seg_idx]
        seg_name    = f"seg{seg_idx+1:03d}"
        byte_lo     = seg_idx * config.segment_size
        byte_hi     = byte_lo + len(seg)
        t_seg_start = time.perf_counter()

        # print(f"\n{'='*70}")
        print(f"SEGMENT {seg_idx+1}/{n_segs}  bytes [{byte_lo}, {byte_hi})  len={len(seg)}")
        if seg_idx + 1 == crossover_seg:
            print(f"  *** COMPRESSION CROSSOVER: data trained ({byte_hi}B) now exceeds "
                  f"param storage ({param_bytes:.0f}B) — actually compressing from here ***")
        # print(f"{'='*70}")

        # pre-solo forgetting printed only on FAIL (budget exhausted)

        # ------------------------------------------------------------------ #
        # SOLO phase                                                           #
        # ------------------------------------------------------------------ #
        print(f"[SOLO] seg{seg_idx+1}  budget={config.max_iter_per_phase} iters")
        solo_inputs, solo_targets = make_chunks_and_targets(seg, config)
        tok_per_byte  = 8 // config.input_bits
        chunk_size_tok = config.segment_size * tok_per_byte
        n_solo_chunks  = solo_inputs.shape[0]

        def _solo_weights():
            w = integral_weights[byte_lo:byte_hi]
            w_tok = np.repeat(w, tok_per_byte) if tok_per_byte > 1 else w.copy()
            pad   = n_solo_chunks * chunk_size_tok - len(w_tok)
            return jnp.array(
                np.concatenate([w_tok, np.ones(pad, np.float32)])
                .reshape(n_solo_chunks, chunk_size_tok))

        t0          = time.perf_counter()
        solo_iters  = 0
        solo_success = False
        last_accs   = None

        pbar = tqdm(range(1, config.max_iter_per_phase + 1), desc=f"solo s{seg_idx+1}", unit="it", file=sys.stderr)
        for it in pbar:
            loss, grads = exact_bptt_step(
                base_xlstm, params, solo_inputs, solo_targets, _solo_weights(), config)
            params, opt_state = sinkgd_update(
                opt_state, grads, params, lr=lr,
                weight_decay=config.weight_decay, max_norm=config.grad_clip_norm,
                L=config.sinkgd_l)
            solo_iters = it
            pbar.set_postfix(loss=f"{float(loss):.5f}")

            if it % config.check_every == 0 or it == config.max_iter_per_phase:
                accs = eval_segments_stateful(base_xlstm, params, config, [seg])
                last_accs = accs
                acc, fw = accs[0]
                # ── PI: update integral weights from failure position ──────
                update_integral(integral_weights, accs,
                                [byte_lo], [len(seg)], config)
                fw_str = f"{fw}" if fw is not None else "ok"
                print()
                bpc = math.exp(float(loss)) / math.log(2)
                w_mean = float(integral_weights[byte_lo:byte_hi].mean())
                pbar.set_description(
                    f"acc={acc:.1%} fail={fw_str} bpc={bpc:.4f} w={w_mean:.2f}")
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
                  f"{config.max_iter_per_phase} iters  "
                  f"(final acc={final_acc:.1%}  fail_at={fw_str})")
            if completed:
                forget_accs = eval_segments_stateful(base_xlstm, params, config, completed + [seg])
                print(f"[forgetting at fail]")
                for pi, (pa, pfw) in enumerate(forget_accs):
                    label = f"NEW seg{seg_idx+1}" if pi == len(completed) else f"seg{pi+1:03d}"
                    fw_str2 = f"pred{pfw}→byte{pfw+1}" if pfw is not None else "ok"
                    print(f"  {label}: acc={pa:.1%}  fail_at={fw_str2}")
            _print_time(seg_elapsed, run_elapsed)
            phase_summary.append(
                (seg_idx+1, "SOLO", solo_iters, elapsed_solo, False, final_acc))
            break

        print(f"[PASS] seg{seg_idx+1} SOLO — 100% in {solo_iters} iters  {elapsed_solo:.1f}s")
        phase_summary.append((seg_idx+1, "SOLO", solo_iters, elapsed_solo, True, 1.0))

        # ------------------------------------------------------------------ #
        # Forgetting snapshot AFTER solo phase (effect on past segments)      #
        # ------------------------------------------------------------------ #
        if completed and solo_iters == config.max_iter_per_phase:
            post_solo_accs = eval_segments_stateful(
                base_xlstm, params, config, completed + [seg])
            print(f"[post-solo forgetting]")
            for pi, (pa, pfw) in enumerate(post_solo_accs):
                label = f"seg{pi+1:03d}"
                fw_str = f"pred{pfw}→byte{pfw+1}" if pfw is not None else "ok"
                print(f"  {label}: acc={pa:.1%}  fail_at={fw_str}")

        completed.append(seg)

        # Generative eval + checkpoint (silent on PASS)
        gen_tokens = np.concatenate(completed)
        gen_out, gen_wrong, gen_acc, gen_fw, residual = eval_generative(
            base_xlstm, params, config, gen_tokens)
        gen_fw_str   = f"byte {gen_fw}" if gen_fw is not None else "none (perfect)"
        residual_kb  = len(residual) * 4 / 1024
        gen_path = os.path.join(log_dir, f"gen_seg{seg_idx+1:03d}_solo.txt")
        full_gen = np.concatenate([[gen_tokens[0]], gen_out])
        with open(gen_path, "w", encoding="utf-8", errors="replace") as gf:
            gf.write(f"phase:       seg{seg_idx+1} SOLO\n"
                     f"segs:        1..{len(completed)}\n"
                     f"gen_acc:     {gen_acc:.1%}\n"
                     f"wrong:       {gen_wrong}/{len(gen_tokens)-1}\n"
                     f"first_wrong: {gen_fw_str}\n"
                     f"residual:    {len(residual)} entries  {residual_kb:.2f} KB\n"
                     f"\n--- target ---\n{ByteTokenizer.decode(gen_tokens)}\n"
                     f"\n--- generated ---\n{ByteTokenizer.decode(full_gen)}\n")
        save_checkpoint(os.path.join(log_dir, "ckpt_last"), params, config, seg_idx, "solo")
        _save_residual(os.path.join(log_dir, "ckpt_last"), residual)

        if seg_idx == 0:
            run_elapsed = time.perf_counter() - t_run_start
            outer_pbar.update(len(seg))
            outer_pbar.set_postfix(seg=f"{seg_idx+1}/{n_segs}",
                                   gen=f"{gen_acc:.0%}", t=f"{run_elapsed:.0f}s")
            continue

        # ------------------------------------------------------------------ #
        # COMBINED phase: train on ALL completed segments together            #
        # ------------------------------------------------------------------ #
        n_done = len(completed)
        print(f"[COMBINED] segs 1..{n_done}  budget={config.max_iter_per_phase} iters")
        combined_tokens = np.concatenate(completed)
        combined_inputs, combined_targets = make_chunks_and_targets(combined_tokens, config)
        n_comb_chunks   = combined_inputs.shape[0]
        comb_size_tok   = config.segment_size * (8 // config.input_bits)
        comb_byte_len   = sum(len(s) for s in completed)
        comb_byte_off   = seg_byte_offsets[:n_done]

        def _comb_weights():
            # stitch integral_weights across all completed segments
            w_all = np.concatenate([
                integral_weights[seg_byte_offsets[i]:seg_byte_offsets[i]+len(completed[i])]
                for i in range(n_done)])
            tok_per = 8 // config.input_bits
            w_tok = np.repeat(w_all, tok_per) if tok_per > 1 else w_all.copy()
            pad   = n_comb_chunks * comb_size_tok - len(w_tok)
            return jnp.array(
                np.concatenate([w_tok, np.ones(pad, np.float32)])
                .reshape(n_comb_chunks, comb_size_tok))

        t0              = time.perf_counter()
        comb_iters      = 0
        comb_success    = False
        last_comb_accs  = None

        pbar = tqdm(range(1, config.max_iter_per_phase + 1), desc=f"comb 1..{n_done}", unit="it", file=sys.stderr)
        for it in pbar:
            loss, grads = exact_bptt_step(
                base_xlstm, params, combined_inputs, combined_targets, _comb_weights(), config)
            params, opt_state = sinkgd_update(
                opt_state, grads, params, lr=lr,
                weight_decay=config.weight_decay, max_norm=config.grad_clip_norm,
                L=config.sinkgd_l)
            comb_iters = it
            pbar.set_postfix(loss=f"{float(loss):.5f}")

            if it % config.check_every == 0 or it == config.max_iter_per_phase:
                accs = eval_segments_stateful(base_xlstm, params, config, completed)
                last_comb_accs = accs
                # ── PI: update integral weights for all completed segs ─────
                update_integral(integral_weights, accs,
                                comb_byte_off, [len(s) for s in completed], config)
                all_perfect = all(a == 1.0 for a, _ in accs)
                min_acc = min(a for a, _ in accs)
                fail_at = next((fw for a, fw in accs if a < 1.0 and fw is not None), None)
                fw_str = f"{fail_at}" if fail_at is not None else "ok"
                bpc = math.exp(loss) / math.log(2)
                print()
                pbar.set_description(f"acc={min_acc:.1%} fail_at={fw_str} bpc={bpc:.4f}")
                if it == config.max_iter_per_phase:
                    elapsed = time.perf_counter() - t0
                    parts = []
                    for pi, (pa, pfw) in enumerate(accs):
                        fw_str2 = f"b{pfw+1}" if pfw is not None else "ok"
                        parts.append(f"s{pi+1}={pa:.0%}({fw_str2})")
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
                  f"{config.max_iter_per_phase} iters")
            if last_comb_accs:
                for pi, (pa, pfw) in enumerate(last_comb_accs):
                    fw_str = f"pred{pfw}→byte{pfw+1}" if pfw is not None else "ok"
                    print(f"  seg{pi+1:03d}: acc={pa:.1%}  fail_at={fw_str}")
            _print_time(seg_elapsed, run_elapsed)
            phase_summary.append(
                (seg_idx+1, "COMBINED", comb_iters, elapsed_comb, False,
                 min(a for a, _ in last_comb_accs) if last_comb_accs else 0.0))
            break

        mean_acc = sum(a for a, _ in last_comb_accs) / len(last_comb_accs)
        phase_summary.append(
            (seg_idx+1, "COMBINED", comb_iters, elapsed_comb, True, mean_acc))

        # Generative eval + checkpoint (silent on PASS)
        gen_tokens = np.concatenate(completed)
        gen_out, gen_wrong, gen_acc, gen_fw, residual = eval_generative(
            base_xlstm, params, config, gen_tokens)
        gen_fw_str  = f"byte {gen_fw}" if gen_fw is not None else "none (perfect)"
        residual_kb = len(residual) * 4 / 1024
        gen_path = os.path.join(log_dir, f"gen_segs001_to_{seg_name}_combined.txt")
        full_gen = np.concatenate([[gen_tokens[0]], gen_out])
        with open(gen_path, "w", encoding="utf-8", errors="replace") as gf:
            gf.write(f"phase:       COMBINED segs 1..{n_done}\n"
                     f"gen_acc:     {gen_acc:.1%}\n"
                     f"wrong:       {gen_wrong}/{len(gen_tokens)-1}\n"
                     f"first_wrong: {gen_fw_str}\n"
                     f"residual:    {len(residual)} entries  {residual_kb:.2f} KB\n"
                     f"\n--- target ---\n{ByteTokenizer.decode(gen_tokens)}\n"
                     f"\n--- generated ---\n{ByteTokenizer.decode(full_gen)}\n")
        save_checkpoint(os.path.join(log_dir, "ckpt_last"), params, config, seg_idx, "combined")
        _save_residual(os.path.join(log_dir, "ckpt_last"), residual)
        run_elapsed = time.perf_counter() - t_run_start
        outer_pbar.update(len(seg))
        outer_pbar.set_postfix(seg=f"{seg_idx+1}/{n_segs}",
                               gen=f"{gen_acc:.0%}", t=f"{run_elapsed:.0f}s")

    outer_pbar.close()

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
