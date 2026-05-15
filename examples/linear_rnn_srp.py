"""
randcompress v10 — Fully linear cells: gate-free mLSTM + whitened sRNN
Train loop: simple TBPTT with random state reset probability (SRP).

mLSTM (block type 'm') — linear, no gates:
  - Replace forget/input gates with frozen per-head scalar decay alpha [NH],
    log-uniform τ in [1, seq_len], alpha_h = exp(-1/τ_h).
  - Cell: C_t = alpha[:,None,None]*C_{t-1} + phi_k⊗v  (pure linear accumulation)
  - Output: h = phi_q @ C, then additive skip + multihead LayerNorm + W_down.
  - State: C only [B,NH,D_sym,DH].

sRNN (block type 's') — whitened vanilla RNN:
  - h_t = LayerNorm(W_hh h_{t-1} + W_hx x_t + b)
  - W_hh: near-critical SVD init, spectral radius=0.95.
  - State: h only [B, d_model].

Training (SRP — random state reset):
  Each iteration: with prob state_reset_prob, reset hidden state and jump to a
  random chunk. Otherwise carry state forward from previous chunk (stop_gradient
  at chunk boundary). Matches zero-state eval regime.
"""

import jax
jax.config.update("jax_enable_x64", True)   # uint64 arithmetic for range coder
# Suppress the scatter int64→int32 FutureWarning from jax_enable_x64 interactions.
# Root cause: jnp.argmax returns int64 with x64 enabled; fixed with .astype(jnp.int32)
# in _quantize_cdf. Remaining instances are harmless (correct implicit cast) until
# a future JAX tightens the rules — suppress until then.
import warnings as _w
_w.filterwarnings("ignore", message="scatter inputs have incompatible types",
                  category=FutureWarning)
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
RC_PREC   = 16   # range coder CDF precision; cumfreqs sum to 1 << RC_PREC

class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()   # flush immediately so tail -f log works in real time
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
    stride_map:    str   = "1111"  # one digit per layer; stride-N fires every N tokens
    block_map:     str   = "smmm"
    # stride_map:    str   = "1248"  # one digit per layer; stride-N fires every N tokens
    # block_map:     str   = "smmm"
    lora_r:        int   = 4
    batch_size:    int   = 1
    # ── optimiser ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-2    # SinkGD: step = √(mn)·lr; output_proj is 64×256 → √=128, so lr≈1/128
    weight_decay:  float = 0.0
    grad_clip_norm:float = 1e2
    sinkgd_l:      int   = 5
    # ── loss ─────────────────────────────────────────────────────────────────
    margin:        float = 0.
    ce_weight:     float = 1.0
    # ── gradient accumulation ────────────────────────────────────────────────
    grad_accum_steps: int   = 1     # K chunks per vgrad call; state flows through all K
    # ── hidden state dropout ──────────────────────────────────────────────────
    state_reset_prob: float = 0.0   # prob of zeroing each layer state between chunks
    # ── residual / stop ──────────────────────────────────────────────────────
    residual_budget: float = 0.
    target_bpb:    float = 0.0
    # ── generative eval ──────────────────────────────────────────────────────
    gen_seed_len:  int   = 16   # teacher-forced warm-up bytes before eval starts
    # ── misc ─────────────────────────────────────────────────────────────────
    pad_token:       int   = 0
    dtype:           str   = "float32"
    max_iters:       int   = 100000
    check_every:     int   = 100
    max_eval_tokens: int   = 0     # 0 = all tokens; >0 = limit monitoring eval (fast BPB snapshot)
    remat:           bool  = True  # True = chunk CDF/round-trip to stay in memory (always tests all tokens)
                                   # False = single-shot (fast, may OOM on large files)
    remat_chunk:     int   = 50000 # tokens per remat chunk (ignored when remat=False)
    save_ckpt:       bool  = True  # save full compressed bundle at each checkpoint; False = log only
    dataset:         str   = "datasets/surat_al-fatihah.txt"
    # dataset:       str   = "datasets/juz1.txt"
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

_SPOW_CACHE: dict = {}   # (D, p) → (i_idx, j_idx, scale); computed once per unique (D,p)

def spow(x, p):
    """Symmetric degree-p polynomial feature map.

    p=1: identity.  p=2: all products x_i*x_j (i<=j), off-diagonal * sqrt(2).
    [..., D] → [..., D_sym] where D_sym = C(D+p-1, p).
    Indices and scale cached on first call for each (D, p) pair.
    """
    if p == 1:
        return x
    D = x.shape[-1]
    key = (D, p)
    if key not in _SPOW_CACHE:
        i_idx, j_idx = np.tril_indices(D)
        scale = np.where(i_idx == j_idx, 1.0, np.sqrt(2.0)).astype(np.float32)
        _SPOW_CACHE[key] = (i_idx, j_idx, scale)
    i_idx, j_idx, scale = _SPOW_CACHE[key]
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


@functools.partial(jax.jit, static_argnames=[
    "chunk_size", "num_heads", "block_map", "stride_map", "oh", "ob"])
def _collect_chunk_jit(base_xlstm, params, toks_chunk, init_states,
                        chunk_size, num_heads, block_map, stride_map, oh, ob):
    """Scan forward_step over chunk_size tokens in one JIT call.

    Key optimisations vs calling _fwd_step_jit per token:
      - apply_xlstm_hira called ONCE per chunk (not per token)
      - lax.scan eliminates Python dispatch overhead (O(T) → O(T/S) JIT calls)
      - Uses mlstm_layer_step / srnn_layer_step (same ops as _fwd_step_jit),
        so logits are bit-identical to the sequential decoder — round-trip OK.

    Returns: chunk_logits [chunk_size, V] (head-0), new_states
    """
    xlstm      = apply_xlstm_hira(base_xlstm, params)   # hoisted out of scan
    stride_tup = stride_map                               # static tuple

    def step(states, t_tok):
        t, tok = t_tok
        x = xlstm.embedding[tok][None, :]                # [1, d_model]
        new_states = []
        for block, btype, stride, state in zip(
                xlstm.blocks, block_map, stride_tup, states):
            lstm_state, last_out = state
            x_n = layer_norm(x, block.norm_w, block.norm_b)
            if stride == 1:
                if btype == 'm':
                    cell_out, new_lstm = mlstm_layer_step(
                        block.mlstm, x_n, lstm_state, num_heads)
                else:
                    cell_out, new_lstm = srnn_layer_step(
                        block.srnn, x_n, lstm_state, num_heads)
                new_states.append((new_lstm, cell_out))
            else:
                # stride > 1: fire/hold decided at runtime
                _block, _btype = block, btype          # capture loop vars
                def fire(args):
                    xn, ls = args
                    if _btype == 'm':
                        return mlstm_layer_step(_block.mlstm, xn, ls, num_heads)
                    return srnn_layer_step(_block.srnn, xn, ls, num_heads)
                def hold(args):
                    _, ls = args
                    return last_out, ls
                cell_out, new_lstm = jax.lax.cond(
                    t % jnp.int32(stride) == jnp.int32(0),
                    fire, hold, (x_n, lstm_state))
                new_states.append((new_lstm, cell_out))
            x = x + cell_out
        x = layer_norm(x, xlstm.norm_w, xlstm.norm_b)
        logit_flat = jnp.dot(x, params.output_proj)
        logits     = _reshape_logits(logit_flat, oh, ob)  # [1, oh, V]
        return new_states, logits[0, 0, :]                # carry, [V]

    new_states, chunk_logits = jax.lax.scan(
        step, init_states,
        (jnp.arange(chunk_size, dtype=jnp.int32), toks_chunk))
    return chunk_logits, new_states   # [chunk_size, V], states


def _reshape_logits(flat, output_heads, output_bits):
    output_vocab = 2 ** output_bits
    return flat.reshape(*flat.shape[:-1], output_heads, output_vocab)


# ============================================================================
# Hidden-state dropout  (applied between chunks inside the grad window)
# ============================================================================

def _dropout_states(states, key, prob):
    """Independently zero each layer's full state with probability `prob`.
    Scalar Bernoulli mask per layer — either keep everything or zero everything.
    Gradient flows through kept layers; zeroed layers break the chain (mask=0 → grad=0).
    """
    new_states = []
    for i, (lstm_state, last_out) in enumerate(states):
        mask = jr.bernoulli(jr.fold_in(key, i), 1.0 - prob).astype(DTYPE)
        new_states.append((
            jax.tree_util.tree_map(lambda s: s * mask, lstm_state),
            last_out * mask,
        ))
    return new_states


# ============================================================================
# K-chunk BPTT with state flowing through (no stop_gradient within window)
# ============================================================================

@functools.partial(jax.jit, static_argnames=["config", "K"])
def grad_accum_persistent(base_xlstm, params, inputs_K, targets_K, init_states, rng_key, config, K):
    """
    inputs_K:    [K, 1, S] int32
    targets_K:   [K, 1, S, oh] int32
    init_states: carry-in state (stop_gradient applied by caller at window boundary)
    rng_key:     for hidden-state dropout between chunks

    State flows through all K chunks without stop_gradient.
    Hidden-state dropout (state_reset_prob) applied between chunks (not after last).
    One vgrad call → one optimizer update.
    Returns (avg_loss, grads, final_states).
    """
    oh         = config.output_heads
    ob         = config.output_bits
    num_heads  = config.num_heads
    block_map  = config.block_map
    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    margin     = config.margin
    ce_weight  = config.ce_weight
    srp        = config.state_reset_prob

    def loss_fn(p):
        states     = init_states
        total_loss = jnp.zeros(())
        for k in range(K):
            logits_flat, states = forward_train_chunked(
                base_xlstm, p, inputs_K[k], states, num_heads, block_map, stride_map)
            logits = _reshape_logits(logits_flat, oh, ob)
            chunk_loss = sum(
                training_loss(logits[..., h, :], targets_K[k][..., h], margin, ce_weight)
                for h in range(oh)
            ) / oh
            total_loss = total_loss + chunk_loss
            if k < K - 1 and srp > 0.0:
                states = _dropout_states(states, jr.fold_in(rng_key, k), srp)
        return total_loss / K, states

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
    loss_per_tok= -jnp.take_along_axis(log_probs, jnp.maximum(targets, 0)[...,None], axis=-1)[...,0]
    mask        = (targets >= 0).astype(jnp.float32)   # -1 padding; handles all binary files
    return (loss_per_tok*mask).sum() / (mask.sum()+1e-8)

def argmax_margin_loss(logits, targets, margin=1.0):
    safe_tgt      = jnp.maximum(targets, 0)
    target_logits = jnp.take_along_axis(logits, safe_tgt[..., None], axis=-1)[..., 0]
    neg_inf       = jax.nn.one_hot(safe_tgt, logits.shape[-1]) * 1e9
    best_wrong    = (logits - neg_inf).max(axis=-1)
    loss_per_tok  = jnp.maximum(0.0, best_wrong - target_logits + margin)
    mask          = (targets >= 0).astype(jnp.float32)
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

# ============================================================================
# Range coder  (JAX JIT + lax.scan + lax.while_loop, uint64 arithmetic)
#
# Encode: lax.scan over tokens forward; each step narrows [low,high] then
#   flushes agreed top bytes via while_loop → writes to fixed buffer.
# Decode: lax.scan over positions forward; each step computes slot, binary-
#   searches CDF for symbol, narrows [low,high], refills from buffer via
#   while_loop. AR decoding interleaves model forward + decode_step in scan.
#
# All ops bit-exact across encode/decode given the same quantized CDFs.
# ============================================================================

def _quantize_cdf(logits_1d):
    """[V] float32 → [V+1] uint32 cumulative freqs, sum = 1<<RC_PREC.
    Min freq = 1 (no zero-prob symbol).  Deterministic; safe under jax.vmap.

    Deficit is absorbed into the largest-freq entry only (not element-wise
    maximum), so the sum stays exactly M even in pathological distributions.
    """
    M     = jnp.int32(1 << RC_PREC)
    probs = jax.nn.softmax(logits_1d.astype(jnp.float32))
    freqs = jnp.maximum(jnp.int32(1), jnp.round(probs * M).astype(jnp.int32))
    # absorb rounding error into the largest entry only — keeps sum == M
    deficit  = M - freqs.sum()
    max_idx  = jnp.argmax(freqs).astype(jnp.int32)   # int32: avoid scatter int64 warning
    adjusted = jnp.maximum(jnp.int32(1), freqs[max_idx] + deficit)
    freqs    = freqs.at[max_idx].set(adjusted)
    cumfreqs = jnp.zeros(logits_1d.shape[0] + 1, jnp.uint32)
    cumfreqs = cumfreqs.at[1:].set(jnp.cumsum(freqs).astype(jnp.uint32))
    # Hard-force last entry = M regardless of any rounding residual (defensive)
    cumfreqs = cumfreqs.at[-1].set(jnp.uint32(1 << RC_PREC))
    return cumfreqs   # [V+1], cumfreqs[-1] == M exactly


@functools.partial(jax.jit, static_argnames=["max_buf"])
def rc_encode(symbols, all_cumfreqs, max_buf):
    """
    symbols:      [T] int32        — token ids in [0, V)
    all_cumfreqs: [T, V+1] uint32  — quantized CDF per position
    max_buf:      static int       — output buffer size (bytes); T*3+64 is safe
    Returns: (buf [max_buf] uint8, n_bytes int32)

    Uses uint64 for all multiplications to avoid overflow.
    Agreed top bytes are flushed via while_loop (0–4 iters per symbol).
    Tail: 4 unconditional bytes written after the scan to finalise the state.
    """
    _U32 = jnp.uint32
    _U64 = jnp.uint64
    _M   = _U64(1 << RC_PREC)
    _FF  = _U64(0xFF)
    _4G  = _U64(0xFFFFFFFF)

    def _flush_cond(s):
        low, high, _buf, _pos = s
        return (_U64(low) >> _U64(24)) == (_U64(high) >> _U64(24))

    def _flush_body(s):
        low, high, buf, pos = s
        byte = (low >> _U32(24)).astype(jnp.uint8)
        buf  = buf.at[pos].set(byte)
        low  = _U32((_U64(low)  << _U64(8)) & _4G)
        high = _U32((_U64(high) << _U64(8) | _FF) & _4G)
        return low, high, buf, pos + jnp.int32(1)

    def encode_step(carry, inputs):
        low, high, buf, pos = carry
        sym, cumfreqs = inputs
        rng    = _U64(high) - _U64(low) + _U64(1)
        cum_lo = _U64(cumfreqs[sym])
        cum_hi = _U64(cumfreqs[sym + _U32(1)])
        high   = _U32(_U64(low) + rng * cum_hi // _M - _U64(1))
        low    = _U32(_U64(low) + rng * cum_lo // _M)
        low, high, buf, pos = jax.lax.while_loop(_flush_cond, _flush_body,
                                                   (low, high, buf, pos))
        return (low, high, buf, pos), None

    buf  = jnp.zeros(max_buf, jnp.uint8)
    init = (_U32(0), _U32(0xFFFFFFFF), buf, jnp.int32(0))
    (low, _high, buf, pos), _ = jax.lax.scan(
        encode_step, init, (symbols.astype(jnp.uint32), all_cumfreqs))

    # flush 4 tail bytes to finalise
    for _ in range(4):
        buf = buf.at[pos].set((low >> _U32(24)).astype(jnp.uint8))
        pos = pos + jnp.int32(1)
        low = _U32((_U64(low) << _U64(8)) & _4G)

    return buf, pos


@functools.partial(jax.jit, static_argnames=["n_tokens"])
def rc_decode(buf, all_cumfreqs, n_tokens):
    """
    buf:           [max_buf] uint8 — encoded stream (from rc_encode)
    all_cumfreqs:  [n_tokens, V+1] uint32 — same CDFs used during encode
    n_tokens:      static int
    Returns: [n_tokens] int32 decoded symbols.

    Initialises x by reading first 4 bytes, then lax.scan forward.
    Each step: binary-search CDF for symbol, narrow [low,high], refill via while_loop.
    """
    _U32 = jnp.uint32
    _U64 = jnp.uint64
    _M   = _U64(1 << RC_PREC)
    _FF  = _U64(0xFF)
    _4G  = _U64(0xFFFFFFFF)

    # read initial 4 bytes into x
    x = _U32(0)
    for i in range(4):
        x = _U32((_U64(x) << _U64(8)) | _U64(buf[i].astype(jnp.uint32)))

    def _refill_cond(s):
        low, high, _x, _pos = s
        return (_U64(low) >> _U64(24)) == (_U64(high) >> _U64(24))

    def _refill_body(s):
        low, high, x, pos = s
        low  = _U32((_U64(low)  << _U64(8)) & _4G)
        high = _U32((_U64(high) << _U64(8) | _FF) & _4G)
        x    = _U32((_U64(x)   << _U64(8) | _U64(buf[pos].astype(jnp.uint32))) & _4G)
        return low, high, x, pos + jnp.int32(1)

    def decode_step(carry, cumfreqs):
        low, high, x, pos = carry
        rng  = _U64(high) - _U64(low) + _U64(1)
        # map x into [0, M): slot = ((x-low+1)*M - 1) // rng
        slot = _U32((_U64(x - low + _U32(1)) * _M - _U64(1)) // rng)
        # binary search: last index where cumfreqs[i] <= slot
        V    = cumfreqs.shape[0] - 1
        sym  = jnp.searchsorted(cumfreqs[:V].view(jnp.int32),
                                  slot.view(jnp.int32), side='right') - 1
        sym  = jnp.clip(sym, 0, V - 1).astype(jnp.uint32)
        cum_lo = _U64(cumfreqs[sym])
        cum_hi = _U64(cumfreqs[sym + _U32(1)])
        high   = _U32(_U64(low) + rng * cum_hi // _M - _U64(1))
        low    = _U32(_U64(low) + rng * cum_lo // _M)
        low, high, x, pos = jax.lax.while_loop(_refill_cond, _refill_body,
                                                 (low, high, x, pos))
        return (low, high, x, pos), sym.astype(jnp.int32)

    init   = (_U32(0), _U32(0xFFFFFFFF), x, jnp.int32(4))
    _, syms = jax.lax.scan(decode_step, init, all_cumfreqs)
    return syms   # [n_tokens] int32


def rc_decode_init(buf):
    """Read first 4 bytes of buf into the RC decode carry state."""
    _U32 = jnp.uint32; _U64 = jnp.uint64
    x = _U32(0)
    for i in range(4):
        x = _U32((_U64(x) << _U64(8)) | _U64(buf[i].astype(jnp.uint32)))
    return (_U32(0), _U32(0xFFFFFFFF), x, jnp.int32(4))


@functools.partial(jax.jit, static_argnames=["n_chunk"])
def rc_decode_chunk(buf, chunk_cumfreqs, n_chunk, carry):
    """Decode n_chunk symbols continuing from carry=(low,high,x,pos).

    Used by the remat round-trip verifier to decode in O(chunk × V) memory
    instead of O(T_total × V), making it safe for files of any size.
    Returns (syms [n_chunk] int32, new_carry).
    """
    _U32 = jnp.uint32; _U64 = jnp.uint64
    _M = _U64(1 << RC_PREC); _FF = _U64(0xFF); _4G = _U64(0xFFFFFFFF)

    def _refill_cond(s):
        l, h, _x, _p = s
        return (_U64(l) >> _U64(24)) == (_U64(h) >> _U64(24))

    def _refill_body(s):
        l, h, x, pos = s
        l   = _U32((_U64(l) << _U64(8)) & _4G)
        h   = _U32((_U64(h) << _U64(8) | _FF) & _4G)
        x   = _U32((_U64(x) << _U64(8) | _U64(buf[pos].astype(jnp.uint32))) & _4G)
        return l, h, x, pos + jnp.int32(1)

    def decode_step(carry, cumfreqs):
        low, high, x, pos = carry
        rng  = _U64(high) - _U64(low) + _U64(1)
        slot = _U32((_U64(x - low + _U32(1)) * _M - _U64(1)) // rng)
        V    = cumfreqs.shape[0] - 1
        sym  = jnp.searchsorted(cumfreqs[:V].view(jnp.int32),
                                  slot.view(jnp.int32), side='right') - 1
        sym  = jnp.clip(sym, 0, V - 1).astype(jnp.uint32)
        cum_lo = _U64(cumfreqs[sym])
        cum_hi = _U64(cumfreqs[sym + _U32(1)])
        high   = _U32(_U64(low) + rng * cum_hi // _M - _U64(1))
        low    = _U32(_U64(low) + rng * cum_lo // _M)
        low, high, x, pos = jax.lax.while_loop(_refill_cond, _refill_body,
                                                (low, high, x, pos))
        return (low, high, x, pos), sym.astype(jnp.int32)

    new_carry, syms = jax.lax.scan(decode_step, carry, chunk_cumfreqs)
    return syms, new_carry   # [n_chunk] int32, new carry


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
    """Load any file as raw bytes (binary-safe, no encoding assumed)."""
    with open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8).copy()

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
    # pad with -1: impossible for any uint8/nibble/bit token → safe for binary files
    padded  = np.concatenate([toks, np.full(pad_len + oh, -1, dtype=np.int32)])
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
# Teacher-forced prob collection + compression eval
# ============================================================================

def collect_probs(base_xlstm, params, config, all_inputs, all_targets):
    """Collect teacher-forced logits using _collect_chunk_jit (chunk-level lax.scan).

    Correctness: _collect_chunk_jit inlines mlstm_layer_step / srnn_layer_step
    (the same ops as _fwd_step_jit), so logits are bit-identical to the
    decoder — RC round-trip is guaranteed OK.

    Speed vs per-token _fwd_step_jit:
      - apply_xlstm_hira called O(num_chunks) times instead of O(T)
      - Python dispatch overhead O(T) → O(T / chunk_size)
      - XLA can pipeline across tokens within the scan

    Returns:
      logits_np [T, V]  float32  — head-0 logits at every valid position
      tgts_np   [T]     int32    — head-0 targets
    """
    stride_map = _parse_stride_map(config.stride_map, config.num_layers)
    num_chunks = all_inputs.shape[0]
    chunk_size = all_inputs.shape[1]
    oh, ob     = config.output_heads, config.output_bits
    states     = init_step_states(config, batch_size=1)
    max_tok    = (config.max_eval_tokens if config.max_eval_tokens > 0
                  else num_chunks * chunk_size)
    logits_list, tgts_list = [], []
    processed = 0
    pbar = tqdm(total=max_tok, desc="collect probs", unit="tok",
                leave=False, file=sys.stderr)
    for ci in range(num_chunks):
        if processed >= max_tok:
            break
        this_size = min(chunk_size, max_tok - processed)
        toks = jnp.array(all_inputs[ci, :this_size], jnp.int32)
        chunk_logits, states = _collect_chunk_jit(
            base_xlstm, params, toks, states,
            this_size, config.num_heads, config.block_map,
            stride_map, oh, ob)                             # [this_size, V]
        logits_list.append(np.array(chunk_logits, np.float32))
        tgts_list.append(np.array(all_targets[ci, :this_size, 0], np.int32))
        processed += this_size
        pbar.update(this_size)
    pbar.close()
    return np.concatenate(logits_list), np.concatenate(tgts_list)


def eval_compression(base_xlstm, params, config, all_inputs, all_targets,
                     n_raw_bytes, label=""):
    """
    Measure and verify two lossless compression schemes.  Prints two lines:

      [compression it=100]  0.3s  param=99328B  n_raw=561  wrong=3/561(99.5%)  OK
        CE=0.0281bpb  rc=0.0854bpb(6B)  p+rc=0.006x(99334B)  p+raw=0.006x(99340B)  p+resid=0.006x(99333B)

    Fields: CE = Shannon entropy lower bound | rc = actual range-coded stream |
    p+rc = params+rc total | p+raw = params+argmax-corrections(raw) |
    p+resid = params+argmax-corrections(range-coded).  ratio > 1x = compression.

    How to read each row
    --------------------
    CE lower bound
        Shannon entropy of the data under this model: sum(-log2 p(token_t)) / 8.
        Theoretical minimum bytes the range coder can produce.  The gap between
        this and "range coded" is coding overhead (typically < 1 bit total).

    range coded
        Actual bytes written by rc_encode (pure range coding, no model params).
        Should be within a few bytes of CE lower bound.  "OK" means rc_decode
        recovered the exact original tokens; "ROUNDTRIP FAIL" means a bug.

    params + rc
        Total compressed file = trainable adapter weights + range-coded stream.
        ratio < 1x means the file grew (model is over-parameterised relative
        to the data).  This is the primary compression metric once the model
        is small enough to fit fewer bytes than the raw data.

    argmax acc
        Fraction of tokens where argmax(logits) == target (teacher-forced).
        "wrong=K/T" is the number of positions the model predicts incorrectly.
        100% accuracy → range-coded stream ≈ 0 bytes (model entropy = 0).

    params + resid(raw)
        Alternative scheme: transmit only the wrong positions as raw corrections.
        Each correction = 4 bytes (2-byte position delta + 2-byte token id).
        Useful when wrong count is very small; cheaper than full range coding
        when n_wrong * 4 < rc_stream_bytes.

    params + resid(rc)
        Same as resid(raw) but the correction tokens are range-coded with the
        model's own probability at those positions.  Since the model was wrong,
        p(correct_token) is low → each correction costs more bits than average,
        so resid(rc) ≈ resid(raw) or slightly larger when the model is very
        wrong at those spots.  Becomes useful when n_wrong is moderate and the
        model still assigns non-trivial probability to the correct token.

    BPC  =  (bytes × 8) / n_raw_bytes   (bits per raw input byte)
    ratio = n_raw_bytes / total_bytes    (> 1x means the file actually shrank)
    """
    t0 = time.perf_counter()

    # ── trainable param size ─────────────────────────────────────────────────
    param_bytes = sum(
        np.prod(p.shape) * np.dtype(DTYPE).itemsize
        for p in jax.tree_util.tree_leaves(params))

    # ── collect teacher-forced logits ────────────────────────────────────────
    logits_np, tgts_np = collect_probs(base_xlstm, params, config, all_inputs, all_targets)
    T     = len(tgts_np)
    V     = 1 << config.output_bits

    # strip padding positions (-1 sentinel covers all file types including binary)
    valid = tgts_np >= 0
    vlog  = logits_np[valid]     # [T_v, V]
    vtgt  = tgts_np[valid]       # [T_v]
    T_v   = int(valid.sum())

    # ── CE lower bound ───────────────────────────────────────────────────────
    lp       = jax.nn.log_softmax(jnp.array(vlog), axis=-1)
    tgt_lp   = np.array(lp)[np.arange(T_v), vtgt]
    ce_bits  = float(-np.sum(tgt_lp) / math.log(2))
    ce_bpb   = ce_bits / n_raw_bytes

    # ── argmax accuracy ──────────────────────────────────────────────────────
    preds    = np.argmax(vlog, axis=-1)
    wrong    = preds != vtgt
    n_wrong  = int(wrong.sum())
    acc      = (T_v - n_wrong) / max(T_v, 1)

    # ── quantize CDFs (vmapped, JIT-compiled) ────────────────────────────────
    # Process in batches of CDF_BATCH to avoid OOM on large T_v (1.4 GB for 1.36M tokens)
    CDF_BATCH = 50_000
    cumfreqs_list = []
    for i in range(0, T_v, CDF_BATCH):
        batch = jnp.array(vlog[i:i + CDF_BATCH])
        cumfreqs_list.append(np.array(jax.vmap(_quantize_cdf)(batch), dtype=np.uint32))
    cumfreqs_np = np.concatenate(cumfreqs_list, axis=0)   # [T_v, V+1] uint32, on CPU
    cumfreqs    = jnp.array(cumfreqs_np)                  # back to JAX for rc_encode

    # seed token: first raw input byte (stored separately for decompressor)
    seed_token = int(all_inputs[0][0])

    # ── Scheme A: range-code all tokens ──────────────────────────────────────
    max_buf_a   = T_v * 3 + 64
    buf_a, nb_a = rc_encode(jnp.array(vtgt, jnp.int32), cumfreqs, max_buf_a)
    rc_bytes    = int(nb_a)
    rc_np       = np.array(buf_a[:rc_bytes], dtype=np.uint8)   # trimmed stream

    # verify round-trip on ALL tokens
    # remat=True  → chunked decode: O(remat_chunk × V) memory, always works
    # remat=False → single-shot decode: O(T_v × V) memory, may OOM on large files
    if config.remat:
        chunk = config.remat_chunk
        rc_carry    = rc_decode_init(buf_a)
        decoded_all = []
        for _i in range(0, T_v, chunk):
            _end  = min(_i + chunk, T_v)
            _cf   = jnp.array(cumfreqs_np[_i:_end])
            _syms, rc_carry = rc_decode_chunk(buf_a, _cf, _end - _i, rc_carry)
            decoded_all.append(np.array(_syms))
        decode_ok = bool(np.all(np.concatenate(decoded_all) == vtgt))
    else:
        dec_a     = rc_decode(buf_a, cumfreqs, T_v)
        decode_ok = bool(jnp.all(dec_a == jnp.array(vtgt, jnp.int32)))
    rt_note = ""

    # ── Scheme B: residual raw (argmax + 4-byte corrections) ─────────────────
    resid_raw_bytes = n_wrong * 4

    # ── Scheme B-rc: range-code corrections with model probs ─────────────────
    if n_wrong > 0:
        w_cumfreqs  = cumfreqs[jnp.array(np.where(valid)[0][wrong], jnp.int32)]
        max_buf_b   = n_wrong * 3 + 32
        _, nb_b     = rc_encode(jnp.array(vtgt[wrong], jnp.int32), w_cumfreqs, max_buf_b)
        resid_rc_bytes = int(nb_b)
    else:
        resid_rc_bytes = 0

    elapsed = time.perf_counter() - t0

    # ── totals ────────────────────────────────────────────────────────────────
    tot_A      = param_bytes + rc_bytes
    tot_B      = param_bytes + resid_raw_bytes
    tot_B_rc = param_bytes + resid_rc_bytes

    def _r(total): return n_raw_bytes / total if total > 0 else float('inf')
    def _bpb(bits): return bits / n_raw_bytes

    hdr  = f"[compression{' ' + label if label else ''}]"
    rt   = ('OK' + rt_note) if decode_ok else ('FAIL' + rt_note)
    # T_v == n_raw_bytes - 1 is the "full file" case (seed byte is not a prediction target)
    eval_note = (f"eval all {n_raw_bytes} bytes"
                 if T_v >= n_raw_bytes - 1
                 else f"eval first {T_v} of {n_raw_bytes} bytes")
    print(f"\n{hdr}  {elapsed:.1f}s  param={param_bytes}B  {eval_note}"
          f"  argmax: {T_v-n_wrong}/{T_v} correct ({acc:.1%})  wrong={n_wrong}  {rt}")
    print(f"  CE={ce_bpb:.4f}bpb"
          f"  rc={_bpb(rc_bytes*8):.4f}bpb({rc_bytes}B)"
          f"  p+rc={_r(tot_A):.3f}x({tot_A}B)"
          f"  p+raw={_r(tot_B):.3f}x({tot_B}B)"
          f"  p+resid={_r(tot_B_rc):.3f}x({tot_B_rc}B)")

    return dict(
        ce_bits=ce_bits, ce_bpb=ce_bpb,
        rc_bytes=rc_bytes, rc_bpb=_bpb(rc_bytes * 8),
        rc_np=rc_np,           seed_token=seed_token,
        T_valid=T_v,               n_raw_bytes=n_raw_bytes,
        param_bytes=param_bytes,
        total_rc=tot_A,          ratio_rc=_r(tot_A),
        n_wrong=n_wrong,           argmax_acc=acc,
        total_resid_raw=tot_B,     ratio_resid_raw=_r(tot_B),
        total_resid_rc=tot_B_rc, ratio_resid_rc=_r(tot_B_rc),
        decode_ok=decode_ok,
    )


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

def save_checkpoint_params(ckpt_dir, params, config):
    """Save params + config only — no rc_stream (used for mid-training snapshots).
    Safe to call with a partial eval (max_eval_tokens < n_raw_bytes).
    """
    import json, pickle
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config._asdict(), f, indent=2)


def save_compressed(ckpt_dir, cstats, params, config):
    """Save a complete decompressible bundle.  Only call when cstats covers the
    FULL file (max_eval_tokens=0 or T_valid == n_raw_bytes), otherwise the
    rc_stream would be for a prefix only and the decompressor would recover
    fewer bytes than the original file.

    Bundle layout:
      config.json      — full Config dict (architecture + seed)
      params.pkl       — HiRA adapter weights as numpy arrays
      rc_stream.bin    — range-coded bytes covering ALL T_valid tokens
      meta.json        — {n_raw_bytes, T_valid, seed_token, rc_bytes, …}

    The decompressor reconstructs the frozen base model from the seed
    (deterministic), restores HiRA from params.pkl, reads seed_token as the
    first model input, then AR-decodes T_valid tokens from rc_stream.bin.
    """
    import json, pickle
    # T_valid == n_raw_bytes - 1 is the correct "full file" case:
    # the seed byte (token[0]) is stored separately; T_valid predictions cover bytes [1..end].
    if cstats["T_valid"] < cstats["n_raw_bytes"] - 1:
        raise ValueError(
            f"save_compressed: T_valid={cstats['T_valid']} < "
            f"n_raw_bytes-1={cstats['n_raw_bytes']-1}. "
            "Run eval_compression with max_eval_tokens=0 (full file) before saving.")
    os.makedirs(ckpt_dir, exist_ok=True)
    # range-coded stream
    with open(os.path.join(ckpt_dir, "rc_stream.bin"), "wb") as f:
        f.write(bytes(cstats["rc_np"].tolist()))
    # HiRA params
    with open(os.path.join(ckpt_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    # config
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config._asdict(), f, indent=2)
    # params.pkl actual on-disk size (written just above)
    param_disk_bytes = os.path.getsize(os.path.join(ckpt_dir, "params.pkl"))
    # meta
    meta = dict(
        n_raw_bytes      = int(cstats["n_raw_bytes"]),
        T_valid          = int(cstats["T_valid"]),
        seed_token       = int(cstats["seed_token"]),
        rc_bytes         = int(cstats["rc_bytes"]),
        param_bytes      = int(cstats["param_bytes"]),
        param_disk_bytes = int(param_disk_bytes),
        input_bits       = int(config.input_bits),
        output_bits      = int(config.output_bits),
        output_heads     = int(config.output_heads),
    )
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


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

    all_inputs, all_targets = make_chunks_and_targets(raw_bytes, config)
    num_chunks = all_inputs.shape[0]

    K = config.grad_accum_steps
    print(f"Chunks: {num_chunks} × {all_inputs.shape[1]} tokens")
    print(f"max_iters={config.max_iters}  check_every={config.check_every}  "
          f"grad_accum_steps={K}  state_reset_prob={config.state_reset_prob}  lr={lr}")
    print()

    # ── step-0 baseline: compression with random weights ─────────────────────
    print("Step 0 — baseline compression (random weights):")
    eval_compression(base_xlstm, params, config, all_inputs, all_targets,
                     n_total, label="step=0")

    # ── training loop: K-chunk BPTT, hidden-state dropout, stop_gradient at window boundary ──
    states      = init_step_states(config, batch_size=1)
    chunk_idx   = 0
    stop_reason = None
    rng_key     = jr.key(config.seed + 2)
    t_start     = time.perf_counter()

    pbar = tqdm(range(1, config.max_iters + 1), desc="train", unit="it",
                file=sys.stderr, dynamic_ncols=True)
    for it in pbar:
        # Collect K sequential chunks
        inputs_K  = jnp.stack([all_inputs [( chunk_idx + k) % num_chunks][None, :] for k in range(K)])
        targets_K = jnp.stack([all_targets[(chunk_idx + k) % num_chunks][None, :] for k in range(K)])
        chunk_idx = (chunk_idx + K) % num_chunks

        # stop_gradient only at the window boundary (between optimizer steps)
        boundary_states = jax.lax.stop_gradient(states)
        rng_key, subkey = jr.split(rng_key)
        loss, grads, states = grad_accum_persistent(
            base_xlstm, params, inputs_K, targets_K, boundary_states, subkey, config, K)
        params, opt_state = sinkgd_update(
            opt_state, grads, params, lr=lr,
            weight_decay=config.weight_decay, max_norm=config.grad_clip_norm,
            L=config.sinkgd_l)

        # CE in nats/token → bits/byte: divide by ln2, multiply by tokens_per_byte
        bpb = float(loss) / math.log(2) * (8 // config.input_bits)
        pbar.set_postfix(loss=f"{float(loss):.5f}", bpb=f"{bpb:.4f}")

        if it % config.check_every == 0 or it == config.max_iters:
            elapsed = time.perf_counter() - t_start
            print(f"\n[it={it:6d}] train_bpb={bpb:.4f}  {elapsed:.0f}s")
            # monitoring eval: respects max_eval_tokens (fast prefix snapshot)
            cstats = eval_compression(base_xlstm, params, config, all_inputs, all_targets,
                                      n_total, label=f"it={it}")
            ckpt_it = os.path.join(log_dir, f"ckpt_it{it:06d}")
            if config.save_ckpt:
                # full eval over entire file for a valid compressed bundle
                full_config = config._replace(max_eval_tokens=0)
                full_cstats = eval_compression(base_xlstm, params, full_config,
                                               all_inputs, all_targets, n_total,
                                               label=f"it={it} full")
                save_compressed(ckpt_it, full_cstats, params, config)
            else:
                save_checkpoint_params(ckpt_it, params, config)

            bpb_done = config.target_bpb > 0 and cstats["rc_bpb"] < config.target_bpb
            if bpb_done:
                stop_reason = f"target_bpb (rc_bpb={cstats['rc_bpb']:.4f} < {config.target_bpb})"
                print(f"[STOP] {stop_reason}")
                pbar.close()
                break
    else:
        pbar.close()

    elapsed_total = time.perf_counter() - t_start
    if stop_reason is None:
        stop_reason = f"max_iters ({config.max_iters})"
        print(f"\n[STOP] {stop_reason}  elapsed={elapsed_total:.1f}s")

    # ── final compression eval + checkpoint (full file, no token limit) ──────
    print("\nFinal compression eval (full file)...")
    full_config = config._replace(max_eval_tokens=0)   # override: encode all tokens
    cstats      = eval_compression(base_xlstm, params, full_config, all_inputs, all_targets,
                                   n_total, label="final")
    ckpt_dir = os.path.join(log_dir, "ckpt_last")
    save_compressed(ckpt_dir, cstats, params, config)   # guard raises if T_valid < n_raw

    import json as _js, pickle as _pk
    with open(os.path.join(ckpt_dir, "compression.json"), "w") as f:
        _js.dump({k: (v if isinstance(v, (int, float, bool, str)) else str(v))
                  for k, v in cstats.items()}, f, indent=2)

    print(f"\nLog: {log_path}")
    log_file.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
