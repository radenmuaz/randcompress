"""
randcompress v4.5.1 — Unigram tokenisation + Curriculum Learning (SOLO/COMBINED)

Key additions vs v4.4:
  - UnigramVocab: learned variable-length byte-sequence vocabulary via EM.
  - build_unigram_vocab: EM tokeniser (greedy longest-match E-step, recount M-step).
  - encode_unigram / decode_unigram: argsort/gather tricks for efficient tokenisation.

Argsort trick (encode):
  len_order = np.argsort(-vocab.vocab_lens)   # descending length order
  Scan positions longest-first so a longer match always beats a shorter prefix.

Gather trick (decode, numpy):
  gathered = vocab.vocab_bytes[token_ids]     # [T, max_len] uint8 in one index op
  Then trim each row by vocab_lens[token_ids].

JAX lax.scan tokenizer/detokenizer (encode_unigram_jax / decode_unigram_jax):
  Fixed max_iter (static), noop + PAD=0 once input/output exhausted.
  encode: scan over max_tokens steps; each step gathers window[pos:pos+max_len] then
          broadcast-compares against all V vocab entries; argmax → best token; noop when done.
  decode: scan over T tokens; each step gathers vocab_bytes[token_ids[t]] → writes bytes
          into flat output buffer at write_pos; noop beyond max_bytes.

Training — curriculum learning (from v8):
  Dataset split into token-level segments (each = segment_size tokens = 1 chunk).
  SOLO phase:     TBPTT on one segment until 100% token accuracy or budget exhausted.
  COMBINED phase: TBPTT on all completed segments until all 100% or budget.
  State resets at each epoch boundary; stop_gradient at every chunk boundary.
  New config field: max_iter_per_phase (replaces max_iters as primary budget).

Stopping:
  100% token TF-accuracy per phase (curriculum criterion).
  residual_budget / target_bpc still apply at final generative eval.

BPC correction for variable-length tokens:
  tokens_per_byte = len(all_tokens) / n_bytes
  BPC = CE_loss_nats * tokens_per_byte / log(2)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import NamedTuple, Optional
from collections import Counter
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
# UnigramVocab
# ============================================================================

class UnigramVocab(NamedTuple):
    vocab_bytes:  np.ndarray  # [V, max_len] uint8, 0-padded
    vocab_lens:   np.ndarray  # [V] int32
    vocab_scores: np.ndarray  # [V] float32 log-probs from EM


def build_unigram_vocab(raw_bytes: np.ndarray, vocab_size: int,
                         max_token_len: int = 16, n_em_iters: int = 5) -> UnigramVocab:
    """
    Build a unigram vocabulary via EM on raw_bytes.

    E-step: greedy longest-match tokenisation (argsort trick — scan lengths longest-first).
    M-step: recount token frequencies, prune to vocab_size keeping highest-freq entries.

    Always includes all 256 single bytes for full coverage.
    Final vocab sorted lexicographically so token 0 = b'\\x00' (natural PAD, never in text).

    Returns UnigramVocab with padded arrays.
    """
    data = np.asarray(raw_bytes, dtype=np.uint8)
    N    = len(data)

    # ── seed vocab: all substrings up to max_token_len ───────────────────────
    print(f"Building unigram vocab (vocab_size={vocab_size}, max_token_len={max_token_len}, "
          f"n_em_iters={n_em_iters})")
    print("  Counting substrings...", end=" ", flush=True)
    counts: Counter = Counter()
    # Always seed all 256 bytes
    for b in range(256):
        counts[bytes([b])] = counts.get(bytes([b]), 0) + 1
    # Count longer substrings (up to max_token_len)
    data_bytes = data.tobytes()
    for length in range(2, max_token_len + 1):
        for i in range(N - length + 1):
            tok = data_bytes[i:i + length]
            counts[tok] = counts.get(tok, 0) + 1
    print(f"{len(counts):,} candidates")

    # Prune to vocab_size (keep all 256 bytes + top-freq longer tokens)
    all_single = {bytes([b]): counts.get(bytes([b]), 1) for b in range(256)}
    longer = {k: v for k, v in counts.items() if len(k) > 1}
    n_longer = max(0, vocab_size - 256)
    top_longer = sorted(longer.items(), key=lambda x: -x[1])[:n_longer]
    current_vocab = {**all_single, **dict(top_longer)}

    def _tokenize(vocab_dict):
        """Greedy longest-match tokenization. Argsort trick: build by_len sorted longest-first."""
        by_len = {}
        for tok in vocab_dict:
            L = len(tok)
            if L not in by_len:
                by_len[L] = set()
            by_len[L].add(tok)
        lengths = sorted(by_len.keys(), reverse=True)  # longest first

        tokens = []
        i = 0
        while i < N:
            matched = False
            for L in lengths:
                if i + L > N:
                    continue
                candidate = data_bytes[i:i + L]
                if candidate in by_len[L]:
                    tokens.append(candidate)
                    i += L
                    matched = True
                    break
            if not matched:
                tokens.append(data_bytes[i:i + 1])
                i += 1
        return tokens

    # ── EM loop ───────────────────────────────────────────────────────────────
    for em_iter in range(n_em_iters):
        # E-step: tokenize with current vocab
        token_seq = _tokenize(current_vocab)
        n_tokens  = len(token_seq)
        approx_bpc = (n_tokens / N) / math.log(2) if N > 0 else float('inf')
        print(f"  EM iter {em_iter+1}/{n_em_iters}: {n_tokens:,} tokens  "
              f"~{approx_bpc:.4f} bpc  (vocab={len(current_vocab)})", flush=True)

        # M-step: recount
        new_counts: Counter = Counter()
        for tok in token_seq:
            new_counts[tok] = new_counts.get(tok, 0) + 1
        # Ensure all 256 bytes present
        for b in range(256):
            key = bytes([b])
            if key not in new_counts:
                new_counts[key] = 1

        # Prune to vocab_size
        longer_new  = {k: v for k, v in new_counts.items() if len(k) > 1}
        n_longer    = max(0, vocab_size - 256)
        top_longer  = sorted(longer_new.items(), key=lambda x: -x[1])[:n_longer]
        current_vocab = {**{bytes([b]): new_counts.get(bytes([b]), 1) for b in range(256)},
                         **dict(top_longer)}

    # ── final tokenize for scores ─────────────────────────────────────────────
    token_seq = _tokenize(current_vocab)
    final_counts: Counter = Counter()
    for tok in token_seq:
        final_counts[tok] = final_counts.get(tok, 0) + 1

    total_count = sum(final_counts.values())

    # ── sort vocab lexicographically → token 0 = b'\x00' ─────────────────────
    vocab_list = sorted(current_vocab.keys())  # lex order

    V       = len(vocab_list)
    max_len = max(len(t) for t in vocab_list)

    vocab_bytes  = np.zeros((V, max_len), dtype=np.uint8)
    vocab_lens   = np.zeros(V, dtype=np.int32)
    vocab_scores = np.zeros(V, dtype=np.float32)

    for idx, tok in enumerate(vocab_list):
        L = len(tok)
        vocab_bytes[idx, :L] = list(tok)
        vocab_lens[idx]      = L
        cnt = final_counts.get(tok, 1)
        vocab_scores[idx]    = math.log(cnt / total_count + 1e-10)

    n_tok_final = len(token_seq)
    approx_bpc_final = (n_tok_final / N) / math.log(2) if N > 0 else float('inf')
    print(f"  Final vocab: {V} tokens  max_len={max_len}  "
          f"~{approx_bpc_final:.4f} bpc  token_0={repr(vocab_list[0])}")

    return UnigramVocab(vocab_bytes=vocab_bytes, vocab_lens=vocab_lens,
                        vocab_scores=vocab_scores)


def encode_unigram(raw_bytes: np.ndarray, vocab: UnigramVocab) -> np.ndarray:
    """
    Greedy longest-match tokenization.

    Argsort trick: len_order = np.argsort(-vocab.vocab_lens) gives tokens sorted
    longest-first. We build a by_len dict grouped by length (from longest down)
    for O(max_len) probe per position.

    Returns int32 token ids.
    """
    data = np.asarray(raw_bytes, dtype=np.uint8)
    N    = len(data)
    data_bytes = data.tobytes()

    # Argsort trick: descending length order
    len_order = np.argsort(-vocab.vocab_lens)  # [V] indices, longest first

    # Build by_len: length → dict{bytes_key: token_id}
    by_len: dict = {}
    for idx in len_order:
        L   = int(vocab.vocab_lens[idx])
        key = bytes(vocab.vocab_bytes[idx, :L].tolist())
        if L not in by_len:
            by_len[L] = {}
        by_len[L][key] = int(idx)

    lengths = sorted(by_len.keys(), reverse=True)  # longest first

    token_ids = []
    i = 0
    while i < N:
        matched = False
        for L in lengths:
            if i + L > N:
                continue
            candidate = data_bytes[i:i + L]
            if candidate in by_len[L]:
                token_ids.append(by_len[L][candidate])
                i += L
                matched = True
                break
        if not matched:
            # Fallback: single byte
            b = int(data[i])
            b_key = bytes([b])
            if 1 in by_len and b_key in by_len[1]:
                token_ids.append(by_len[1][b_key])
            else:
                token_ids.append(b)  # last resort
            i += 1

    return np.array(token_ids, dtype=np.int32)


def decode_unigram(token_ids: np.ndarray, vocab: UnigramVocab) -> np.ndarray:
    """
    Decode token ids back to raw bytes.

    Gather trick: `gathered = vocab.vocab_bytes[token_ids]` gathers all byte
    sequences at once as [T, max_len] uint8. Then trim each row by
    vocab_lens[token_ids] to remove padding.

    Returns flat uint8 bytes.
    """
    token_ids = np.asarray(token_ids, dtype=np.int32)
    T         = len(token_ids)
    if T == 0:
        return np.array([], dtype=np.uint8)

    # Gather trick: one index op for all tokens
    gathered = vocab.vocab_bytes[token_ids]    # [T, max_len]
    lens     = vocab.vocab_lens[token_ids]     # [T]

    out = []
    for t in range(T):
        L = int(lens[t])
        out.extend(gathered[t, :L].tolist())

    return np.array(out, dtype=np.uint8)


@functools.partial(jax.jit, static_argnames=['max_tokens'])
def encode_unigram_jax(input_bytes, vocab_bytes_jax, vocab_lens_jax, vocab_scores_jax, max_tokens):
    """
    Greedy longest-match tokenization via lax.scan.

    Argsort/gather trick per scan step:
      1. Gather window: input[pos : pos+max_len] via index arithmetic
      2. Broadcast-compare window against all V vocab entries  (gather trick)
      3. Argmax over (match & fits) weighted by length + score  → best token

    State: (byte_pos, done_flag).
    Noop when done: emit PAD token 0, advance 0.
    Output: token_ids[max_tokens] int32, padded with 0.

    max_tokens: static upper bound (use N = len(input_bytes) for safety).
    """
    N       = input_bytes.shape[0]
    max_len = vocab_bytes_jax.shape[1]
    k_idx   = jnp.arange(max_len)          # [max_len], reused each step
    v_idx   = jnp.arange(vocab_bytes_jax.shape[0])  # [V]
    padded  = jnp.concatenate(
        [input_bytes, jnp.zeros(max_len, jnp.uint8)])  # [N + max_len]

    def step(carry, _):
        pos, done = carry
        safe_pos  = jnp.clip(pos, 0, N - 1)

        # Gather window at current position  — GATHER TRICK
        window = padded[safe_pos + k_idx]               # [max_len]

        # Compare window against all vocab entries
        byte_match = window[None, :] == vocab_bytes_jax # [V, max_len]
        in_tok     = k_idx[None, :] < vocab_lens_jax[:, None]  # [V, max_len]
        fits       = safe_pos + vocab_lens_jax <= N     # [V]
        match_v    = jnp.all(byte_match | ~in_tok, axis=-1) & fits  # [V]

        # Score: longest match wins; tie-break by log-prob
        score   = jnp.where(match_v,
                             vocab_lens_jax.astype(jnp.float32)
                             + vocab_scores_jax * 1e-4,
                             -1.0)
        best_v  = jnp.argmax(score)                     # scalar
        best_l  = vocab_lens_jax[best_v]                # scalar

        # Noop if done: emit PAD=0, advance 0
        emit    = jnp.where(done, jnp.int32(0), best_v)
        advance = jnp.where(done, jnp.int32(0), jnp.maximum(best_l, 1))
        new_pos = pos + advance
        return (new_pos, done | (new_pos >= N)), emit

    _, token_ids = jax.lax.scan(
        step, (jnp.int32(0), jnp.bool_(False)), None, length=max_tokens)
    return token_ids  # [max_tokens] int32, trailing entries are PAD=0


@functools.partial(jax.jit, static_argnames=['max_bytes'])
def decode_unigram_jax(token_ids, vocab_bytes_jax, vocab_lens_jax, max_bytes):
    """
    Token ids → flat byte array via lax.scan.

    Gather trick: at each scan step, vocab_bytes_jax[token_ids[t]] gathers
    the byte sequence for token t in one index op.

    State: (byte_buf[max_bytes], write_pos).
    Noop when token is PAD (id==0) and write_pos >= actual content end.
    Output: (bytes[max_bytes] uint8, n_bytes int32).
    Padded slots remain 0x00.

    max_bytes: static upper bound on output length.
    """
    T       = token_ids.shape[0]
    max_len = vocab_bytes_jax.shape[1]
    k_idx   = jnp.arange(max_len)          # [max_len], reused each step

    def step(carry, t):
        buf, write_pos = carry
        tok_bytes = vocab_bytes_jax[token_ids[t]]   # [max_len] — GATHER TRICK
        tok_len   = vocab_lens_jax[token_ids[t]]    # scalar

        write_idx = write_pos + k_idx               # [max_len]
        valid     = (k_idx < tok_len) & (write_idx < max_bytes)
        clipped   = jnp.clip(write_idx, 0, max_bytes - 1)
        new_buf   = buf.at[clipped].set(
            jnp.where(valid, tok_bytes, buf[clipped]))
        new_pos   = write_pos + tok_len
        return (new_buf, new_pos), None

    buf_init = jnp.zeros(max_bytes, jnp.uint8)
    (buf_final, n_bytes), _ = jax.lax.scan(
        step, (buf_init, jnp.int32(0)), jnp.arange(T))
    return buf_final, n_bytes  # [max_bytes] uint8, int32


# ============================================================================
# Config
# ============================================================================

class Config(NamedTuple):
    vocab_size:       int   = 1024   # unigram vocab size
    max_token_len:    int   = 16     # max bytes per token
    n_em_iters:       int   = 5      # EM iterations for vocab build
    output_heads:     int   = 1      # MTP heads
    d_model:          int   = 128
    num_heads:        int   = 8
    d_ff:             int   = 512
    num_layers:       int   = 8      # always overridden to len(block_map)
    segment_size:     int   = 512    # tokens per chunk (NOT bytes)
    seed:             int   = 0
    conv_kernel:      int   = 4
    block_map:        str   = "m"*8
    lora_r:           int   = 16
    # ── optimiser ────────────────────────────────────────────────────────────
    learning_rate:    float = 1e-2
    weight_decay:     float = 0.0
    grad_clip_norm:   float = 1e6
    sinkgd_l:         int   = 2
    # ── loss ─────────────────────────────────────────────────────────────────
    margin:           float = 0.0    # 0.0 → pure CE
    ce_weight:        float = 1.0
    # ── TBPTT state drop ─────────────────────────────────────────────────────
    state_reset_prob: float = 0.1    # prob of resetting hidden state each step
    # ── residual / stop ──────────────────────────────────────────────────────
    residual_budget:  float = 0.1    # fraction of dataset bytes; stop when errors fit
    target_bpc:       float = 0.0    # also stop when BPC < this; 0.0 = disabled
    # ── misc ─────────────────────────────────────────────────────────────────
    dtype:            str   = "float32"
    max_iters:        int   = 100000   # kept for compat; curriculum uses max_iter_per_phase
    max_iter_per_phase: int = 10000   # SOLO / COMBINED budget per segment
    check_every:      int   = 1000
    dataset:          str   = "datasets/quran-uthmani.txt"

# preset table
_PRESETS = {
    "byte":         dict(vocab_size=256,   max_token_len=1,  output_heads=1,
                         d_model=64,  d_ff=256,  block_map="m"*8),
    "unigram_512":  dict(vocab_size=512,   max_token_len=16, output_heads=1,
                         d_model=128, d_ff=512,  block_map="m"*8),
    "unigram_1k":   dict(vocab_size=1024,  max_token_len=16, output_heads=1,
                         d_model=128, d_ff=512,  block_map="m"*8),
    "unigram_4k":   dict(vocab_size=4096,  max_token_len=16, output_heads=1,
                         d_model=256, d_ff=1024, block_map="m"*8),
    "unigram_16k":  dict(vocab_size=16384, max_token_len=16, output_heads=1,
                         d_model=256, d_ff=1024, block_map="m"*8),
    "mtp2_1k":      dict(vocab_size=1024,  max_token_len=16, output_heads=2,
                         d_model=128, d_ff=512,  block_map="m"*8),
    "mtp4_1k":      dict(vocab_size=1024,  max_token_len=16, output_heads=4,
                         d_model=128, d_ff=512,  block_map="m"*8),
    "mtp2_4k":      dict(vocab_size=4096,  max_token_len=16, output_heads=2,
                         d_model=256, d_ff=1024, block_map="m"*8),
}


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

_fwd_step_jit    = jax.jit(forward_step,          static_argnames=["num_heads","block_map"])
_fwd_chunked_jit = jax.jit(forward_train_chunked, static_argnames=["num_heads","block_map"])

def _reshape_logits(flat, output_heads, vocab_size):
    """flat: [..., output_heads*vocab_size] → [..., output_heads, vocab_size]"""
    return flat.reshape(*flat.shape[:-1], output_heads, vocab_size)


# ============================================================================
# Single TBPTT step with carry-in state
# ============================================================================

@functools.partial(jax.jit, static_argnames=["config"])
def grad_chunk_persistent(base_xlstm, params, inputs, targets, layer_states, config):
    """
    inputs:       [1, S] int32
    targets:      [1, S, output_heads] int32
    layer_states: carry-in hidden state (stop_gradient applied by caller)
    Returns (loss, grads, new_states).
    """
    oh        = config.output_heads
    vs        = config.vocab_size
    num_heads = config.num_heads
    block_map = config.block_map
    margin    = config.margin
    ce_weight = config.ce_weight

    def loss_fn(p):
        logits_flat, new_states = forward_train_chunked(
            base_xlstm, p, inputs, layer_states, num_heads, block_map)
        logits = _reshape_logits(logits_flat, oh, vs)   # [1, S, oh, vs]
        loss = sum(
            training_loss(logits[..., h, :], targets[..., h], margin, ce_weight)
            for h in range(oh)
        ) / oh
        return loss, new_states

    (loss, new_states), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, grads, new_states


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
    out_dim = config.output_heads * config.vocab_size   # total output width
    return RandCompressParams(
        output_proj=_bf(np.array(jr.normal(ks[0], (config.d_model, out_dim))) * 0.01),
        emb_hira   =init_hira_adapter(ks[1], config.vocab_size, config.d_model, config.lora_r),
        blocks     =[init_block_hira(ks[2+i], config, config.block_map[i])
                     for i in range(config.num_layers)],
    )


# ============================================================================
# SinkGD  (Scetbon et al., 2025 — arXiv:2502.06742)
# SR-Sinkhorn normalises each gradient matrix then SGD.
# ============================================================================

def _sr_sinkhorn_2d(W, L):
    """SR-Sinkhorn on a 2-D float32 matrix W ∈ ℝ^{m×n}."""
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
    """Multiclass hinge loss — directly penalises argmax failures."""
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    neg_inf       = jax.nn.one_hot(targets, logits.shape[-1]) * 1e9
    best_wrong    = (logits - neg_inf).max(axis=-1)
    loss_per_tok  = jnp.maximum(0.0, best_wrong - target_logits + margin)
    mask          = (targets != PAD_TOKEN).astype(jnp.float32)
    return (loss_per_tok * mask).sum() / (mask.sum() + 1e-8)

def training_loss(logits, targets, margin=1.0, ce_weight=0.05):
    """margin=0.0 → pure cross-entropy. ce_weight=0.0 → pure margin loss."""
    if margin == 0.0:
        return cross_entropy_loss(logits, targets)
    ce = cross_entropy_loss(logits, targets)
    mg = argmax_margin_loss(logits, targets, margin)
    return mg + ce_weight * ce


# ============================================================================
# Data helpers
# ============================================================================

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().encode('utf-8')
    return np.frombuffer(raw, dtype=np.uint8).copy()


def make_chunks_and_targets(all_tokens: np.ndarray, segment_size: int, output_heads: int):
    """
    Pack pre-tokenized all_tokens [T] int32 into chunked arrays.

    segment_size: tokens per chunk (Config.segment_size in v4.5 means TOKENS).
    output_heads: MTP heads. head h predicts token at offset 1+h.

    Returns:
      all_inputs:  [num_chunks, segment_size] int32
      all_targets: [num_chunks, segment_size, output_heads] int32
    """
    toks       = np.asarray(all_tokens, dtype=np.int32)
    chunk_size = segment_size
    n          = len(toks)
    oh         = output_heads

    # Pad enough for inputs + MTP lookahead
    num_chunks = max(1, math.ceil((n - 1) / chunk_size))
    pad_len    = max(0, num_chunks * chunk_size + oh - n + 1)
    padded     = np.concatenate([toks, np.zeros(pad_len + oh, dtype=np.int32)])

    all_inputs = np.stack([padded[i*chunk_size : i*chunk_size + chunk_size]
                           for i in range(num_chunks)]).astype(np.int32)

    tgt_chunks = []
    for i in range(num_chunks):
        chunk_tgts = np.zeros((chunk_size, oh), dtype=np.int32)
        for h in range(oh):
            chunk_tgts[:, h] = padded[i*chunk_size + 1 + h : i*chunk_size + chunk_size + 1 + h]
        tgt_chunks.append(chunk_tgts)

    all_targets = np.stack(tgt_chunks)   # [num_chunks, chunk_size, oh]
    return jnp.array(all_inputs, dtype=jnp.int32), jnp.array(all_targets, dtype=jnp.int32)


# ============================================================================
# Generative evaluation (token-space, with byte-level accuracy via decode)
# ============================================================================

def eval_generative(base_xlstm, params, config, all_tokens, vocab: UnigramVocab,
                    tokens_per_byte: float):
    """
    Autoregressive decode in TOKEN space. Seed = first token.
    Generate len(all_tokens)-1 more tokens greedily.

    After generation: decode both target and generated sequences to bytes
    using decode_unigram for byte-level accuracy reporting.

    residual_budget: collect (token_idx → correct_token_id) corrections.
    Each entry = 4 bytes storage. Once budget exhausted: stop correcting.

    tokens_per_byte: used for BPC display.

    Returns (generated_tokens_np, bytes_wrong, accuracy, first_wrong_byte, residual_dict).
    """
    toks        = np.asarray(all_tokens, dtype=np.int32)
    n_tokens    = len(toks) - 1   # tokens to predict (skip seed)
    target_bytes_full = decode_unigram(toks, vocab)
    n_bytes     = len(target_bytes_full)
    budget_bytes    = int(config.residual_budget * n_bytes)
    bytes_per_entry = 4
    max_entries     = budget_bytes // bytes_per_entry if budget_bytes > 0 else 0

    oh     = config.output_heads
    states = init_step_states(config, batch_size=1)
    residual: dict = {}

    cur_token = jnp.array([int(toks[0])])
    gen_toks  = []

    _pbar = tqdm(range(n_tokens), desc="gen", unit="tok", unit_scale=True,
                 leave=False, file=sys.stderr)
    for tok_idx in _pbar:
        logit_flat, states = _fwd_step_jit(
            base_xlstm, params, cur_token, states, config.num_heads, config.block_map)
        logits = _reshape_logits(logit_flat, oh, config.vocab_size)  # [1, oh, vs]

        # Use head-0 prediction (head-0 = next token; MTP heads are auxiliary)
        pred_tok    = int(jnp.argmax(logits[0, 0]))
        correct_tok = int(toks[tok_idx + 1])

        if pred_tok != correct_tok and len(residual) < max_entries:
            residual[tok_idx] = correct_tok
            cur_token = jnp.array([correct_tok])
        else:
            cur_token = jnp.array([pred_tok])

        gen_toks.append(pred_tok)
        _pbar.set_postfix(wrong=len(residual))
    _pbar.close()

    # Decode both sequences to bytes for byte-level accuracy
    gen_tokens_np  = np.array(gen_toks, dtype=np.int32)
    target_toks_np = toks[1:]   # target = toks[1:]

    gen_bytes    = decode_unigram(gen_tokens_np,  vocab)
    target_bytes = decode_unigram(target_toks_np, vocab)

    # Align lengths for comparison
    min_len = min(len(gen_bytes), len(target_bytes))
    gen_bytes    = gen_bytes[:min_len]
    target_bytes = target_bytes[:min_len]

    wrong_mask  = gen_bytes != target_bytes
    bytes_wrong = int(np.sum(wrong_mask))
    accuracy    = (min_len - bytes_wrong) / max(min_len, 1)
    wrong_idxs  = np.where(wrong_mask)[0]
    first_wrong = int(wrong_idxs[0]) if len(wrong_idxs) > 0 else None

    return gen_tokens_np, bytes_wrong, accuracy, first_wrong, residual


# ============================================================================
# Curriculum helpers
# ============================================================================

def split_token_segments(all_tokens, segment_size):
    """Split token array into segments of segment_size tokens."""
    toks, segs, i = np.asarray(all_tokens, dtype=np.int32), [], 0
    while i < len(toks):
        segs.append(toks[i : i + segment_size])
        i += segment_size
    return segs

def eval_segments_stateful(base_xlstm, params, config, seg_token_list):
    """Teacher-forced token accuracy, state flowing across segments (zero-init start).
    Returns list of (accuracy, first_wrong_token_idx) per segment.
    """
    states  = init_step_states(config, batch_size=1)
    oh      = config.output_heads
    vs      = config.vocab_size
    results = []

    for seg_toks in seg_token_list:
        seg = np.asarray(seg_toks, dtype=np.int32)
        n   = len(seg)
        if n < 2:
            results.append((1.0, None)); continue

        inp_arr, tgt_arr = make_chunks_and_targets(seg, config.segment_size, oh)

        tok_correct = []
        for ci in range(inp_arr.shape[0]):
            inp = inp_arr[ci][None, :]
            logits_flat, states = _fwd_chunked_jit(
                base_xlstm, params, inp, states, config.num_heads, config.block_map)
            logits  = _reshape_logits(logits_flat, oh, vs)          # [1, S, oh, vs]
            preds   = np.array(jnp.argmax(logits[0, :, 0], axis=-1)) # [S] head-0
            targets = np.array(tgt_arr[ci, :, 0])                    # [S] head-0
            valid   = min(n - 1 - ci * config.segment_size, config.segment_size)
            if valid > 0:
                tok_correct.extend((preds[:valid] == targets[:valid]).tolist())

        total = len(tok_correct)
        acc   = sum(tok_correct) / total if total > 0 else 1.0
        wrong = [i for i, ok in enumerate(tok_correct) if not ok]
        results.append((acc, wrong[0] if wrong else None))

    return results


# ============================================================================
# Checkpoint
# ============================================================================

def save_checkpoint(ckpt_dir, params, config, vocab: UnigramVocab, seg_idx, phase):
    import json, pickle
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = {**config._asdict(), "seg_idx": seg_idx, "phase": phase}
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(ckpt_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    # Save vocab arrays for decompression
    np.save(os.path.join(ckpt_dir, "vocab_bytes.npy"),  vocab.vocab_bytes)
    np.save(os.path.join(ckpt_dir, "vocab_lens.npy"),   vocab.vocab_lens)
    np.save(os.path.join(ckpt_dir, "vocab_scores.npy"), vocab.vocab_scores)

def _save_residual(ckpt_dir, residual: dict):
    """Save residual corrections."""
    import pickle
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "residual.pkl"), "wb") as f:
        pickle.dump(residual, f)


# ============================================================================
# CLI config parser
# ============================================================================

def _parse_config():
    """Build Config from defaults, optional --preset, then --key val overrides.
    Returns (config, diff_vs_defaults).
    Usage:
        python script.py [--preset <name>] [--key val ...]
    """
    import json as _json

    _types = {f: type(v) for f, v in Config()._asdict().items()}

    def _cast(field, s):
        t = _types[field]
        if t is bool:
            return s.lower() in ("1", "true", "yes")
        return t(s)

    cfg  = Config()._asdict()
    argv = sys.argv[1:]

    # --preset <name>
    if "--preset" in argv:
        idx  = argv.index("--preset")
        name = argv[idx + 1]
        if name not in _PRESETS:
            print(f"  [CONFIG WARN] unknown preset '{name}'; known: {list(_PRESETS)}")
        else:
            cfg.update(_PRESETS[name])

    # --config path.json
    if "--config" in argv:
        idx = argv.index("--config")
        with open(argv[idx + 1]) as f:
            for k, v in _json.load(f).items():
                if k in cfg:
                    cfg[k] = v

    # --key val (highest priority)
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and argv[i] not in ("--config", "--preset"):
            field = argv[i][2:]
            if field in cfg and i + 1 < len(argv):
                cfg[field] = _cast(field, argv[i + 1])
                i += 2
                continue
        i += 1

    # derived: block_map → num_layers
    cfg['num_layers'] = len(cfg['block_map'])

    config = Config(**cfg)

    lib_defaults = Config()._asdict()
    diff = {k: {"default": lib_defaults[k], "value": cfg[k]}
            for k in cfg if cfg[k] != lib_defaults[k]}

    return config, diff


# ============================================================================
# Main
# ============================================================================

def main():
    import json as _json
    _script  = os.path.splitext(os.path.basename(__file__))[0]
    log_dir  = os.path.join("log", _script)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)

    config, config_diff = _parse_config()
    global PAD_TOKEN, DTYPE
    PAD_TOKEN = 0   # null byte — always lex-sorted first, never appears in UTF-8 text
    DTYPE     = jnp.dtype(config.dtype)

    # Save full config + diff
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

    # ── load dataset ─────────────────────────────────────────────────────────
    raw_bytes = load_dataset(config.dataset)
    n_bytes   = len(raw_bytes)
    print(f"\nDataset: {config.dataset}  ({n_bytes:,} bytes)")

    # ── build unigram vocab ───────────────────────────────────────────────────
    print()
    vocab = build_unigram_vocab(
        raw_bytes,
        vocab_size    = config.vocab_size,
        max_token_len = config.max_token_len,
        n_em_iters    = config.n_em_iters,
    )
    V       = len(vocab.vocab_lens)
    max_len = int(vocab.vocab_bytes.shape[1])

    # ── tokenize corpus ───────────────────────────────────────────────────────
    print("\nTokenizing corpus...", end=" ", flush=True)
    all_tokens      = encode_unigram(raw_bytes, vocab)
    n_tokens        = len(all_tokens)
    tokens_per_byte = n_tokens / n_bytes
    approx_bpc      = tokens_per_byte / math.log(2)
    print(f"{n_tokens:,} tokens  ({tokens_per_byte:.4f} tok/byte  "
          f"lower-bound BPC ≈ {approx_bpc:.4f})")

    # Verify round-trip
    roundtrip = decode_unigram(all_tokens, vocab)
    if len(roundtrip) != len(raw_bytes) or not np.array_equal(roundtrip, raw_bytes):
        diff_count = int(np.sum(roundtrip[:min(len(roundtrip),len(raw_bytes))] !=
                                raw_bytes[:min(len(roundtrip),len(raw_bytes))]))
        print(f"  [WARN] encode/decode round-trip mismatch: ~{diff_count} bytes differ")
    else:
        print("  encode/decode round-trip: OK")

    # ── print vocab stats ─────────────────────────────────────────────────────
    print(f"\nVocab stats:")
    print(f"  vocab_size={V}  max_token_len={max_len}")
    single_byte = int(np.sum(vocab.vocab_lens == 1))
    multi_byte  = V - single_byte
    print(f"  single-byte tokens: {single_byte}  multi-byte tokens: {multi_byte}")
    # show top-10 most-frequent multi-byte tokens
    multi_idxs = np.where(vocab.vocab_lens > 1)[0]
    if len(multi_idxs) > 0:
        top_k = min(10, len(multi_idxs))
        top_by_score = multi_idxs[np.argsort(-vocab.vocab_scores[multi_idxs])[:top_k]]
        print(f"  top-{top_k} multi-byte tokens:")
        for idx in top_by_score:
            L   = int(vocab.vocab_lens[idx])
            tok = bytes(vocab.vocab_bytes[idx, :L].tolist())
            sc  = float(vocab.vocab_scores[idx])
            try:
                tok_str = tok.decode('utf-8')
            except Exception:
                tok_str = repr(tok)
            print(f"    [{idx:5d}] {tok_str!r:20s}  log_prob={sc:.3f}")

    # ── initialise model ──────────────────────────────────────────────────────
    key = jr.key(config.seed)
    key, k_xlstm, k_rc = jr.split(key, 3)

    print("\nInitialising frozen xLSTM...")
    base_xlstm = init_xlstm_params(k_xlstm, config)
    frozen_count = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(base_xlstm))

    print("Initialising HiRA adapters...")
    params    = init_randcompress_params(k_rc, config)
    trainable = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    total          = frozen_count + trainable
    dtype_bytes    = np.dtype(DTYPE).itemsize
    param_bytes    = trainable * dtype_bytes
    residual_bytes = int(config.residual_budget * n_bytes)
    total_coded    = param_bytes + residual_bytes
    compress_ratio   = n_bytes / param_bytes  if param_bytes > 0 else float('inf')
    compress_ratio_r = n_bytes / total_coded  if total_coded  > 0 else float('inf')

    print(f"{'metric':<32}  {'value':>12}")
    print(f"{'-'*32}  {'-'*12}")
    print(f"{'frozen params':<32}  {frozen_count/1e6:>11.3f}M")
    print(f"{'trainable params':<32}  {trainable/1e6:>11.3f}M")
    print(f"{'total params':<32}  {total/1e6:>11.3f}M")
    print(f"{'trainable / total':<32}  {trainable/total*100:>11.1f}%")
    print(f"{'---':<32}  {'---':>12}")
    print(f"{'dtype':<32}  {jnp.dtype(DTYPE).name:>12}")
    print(f"{'trainable param bytes':<32}  {param_bytes/1e6:>11.3f}M")
    print(f"{'residual budget bytes':<32}  {residual_bytes/1e6:>11.3f}M  "
          f"({config.residual_budget*100:.1f}%)")
    print(f"{'total coded bytes':<32}  {total_coded/1e6:>11.3f}M")
    print(f"{'dataset bytes':<32}  {n_bytes/1e6:>11.3f}M")
    print(f"{'---':<32}  {'---':>12}")
    print(f"{'params / dataset byte':<32}  {trainable/n_bytes:>12.2f}")
    print(f"{'compression ratio (params)':<32}  {compress_ratio:>11.2f}x")
    print(f"{'compression ratio (+ residual)':<32}  {compress_ratio_r:>11.2f}x")
    print(f"{'compressed?':<32}  {'yes' if compress_ratio_r > 1 else 'no (over-param)':>12}")

    opt_state = sinkgd_init(params)
    lr        = config.learning_rate

    bytes_per_entry = 4
    budget_bytes_   = int(config.residual_budget * (n_bytes - 1))
    max_entries     = budget_bytes_ // bytes_per_entry

    segs   = split_token_segments(all_tokens, config.segment_size)
    n_segs = len(segs)
    print(f"\nCurriculum segments: {n_segs} × up to {config.segment_size} tokens")
    print(f"max_iter_per_phase={config.max_iter_per_phase}  check_every={config.check_every}  lr={lr}")
    print(f"tokens_per_byte={tokens_per_byte:.4f}  BPC = loss_nats × {tokens_per_byte/math.log(2):.4f}")
    print(f"Residual budget: {budget_bytes_} bytes  ({max_entries} correction entries)")
    print()

    t_start      = time.perf_counter()
    global_iters = 0
    completed    = []     # list of token segments that passed both phases
    failed_seg   = None
    stop_reason  = None

    def _tbptt_phase(phase_name, inputs, targets, eval_segs, p, opt_st):
        """TBPTT curriculum phase. Cycles chunks in order; resets state each epoch.
        Returns (success, iters_done, last_accs, updated_params, updated_opt_state).
        """
        nonlocal global_iters
        n_chunks   = inputs.shape[0]
        chunk_idx  = 0
        states     = init_step_states(config, batch_size=1)
        last_accs  = None
        success    = False
        iters_done = 0

        pbar = tqdm(range(1, config.max_iter_per_phase + 1), desc=phase_name,
                    unit="it", file=sys.stderr, dynamic_ncols=True)
        for it in pbar:
            if chunk_idx == 0:                          # epoch boundary → reset state
                states = init_step_states(config, batch_size=1)

            inp = inputs [chunk_idx][None, :]
            tgt = targets[chunk_idx][None, :]
            chunk_idx = (chunk_idx + 1) % n_chunks

            frozen_states = jax.lax.stop_gradient(states)
            loss, grads, states = grad_chunk_persistent(
                base_xlstm, p, inp, tgt, frozen_states, config)
            p, opt_st = sinkgd_update(
                opt_st, grads, p, lr=lr,
                weight_decay=config.weight_decay, max_norm=config.grad_clip_norm,
                L=config.sinkgd_l)
            global_iters += 1
            iters_done    = it

            bpc = float(loss) * tokens_per_byte / math.log(2)
            pbar.set_postfix(loss=f"{float(loss):.4f}", bpc=f"{bpc:.3f}")

            if it % config.check_every == 0 or it == config.max_iter_per_phase:
                accs      = eval_segments_stateful(base_xlstm, p, config, eval_segs)
                last_accs = accs
                min_acc   = min(a for a, _ in accs)
                fw_str    = next((str(fw) for _, fw in accs if fw is not None), "ok")
                pbar.set_description(f"{phase_name} acc={min_acc:.1%} fw={fw_str}")
                if all(a == 1.0 for a, _ in accs):
                    success = True
                    pbar.close()
                    break
        else:
            pbar.close()

        return success, iters_done, last_accs, p, opt_st

    for seg_idx, seg in enumerate(segs):
        t_seg    = time.perf_counter()
        tok_lo   = seg_idx * config.segment_size
        print(f"\nSEGMENT {seg_idx+1}/{n_segs}  tokens [{tok_lo}, {tok_lo+len(seg)})  len={len(seg)}")

        # ── SOLO phase ────────────────────────────────────────────────────────
        seg_inp, seg_tgt = make_chunks_and_targets(seg, config.segment_size, config.output_heads)
        print(f"[SOLO]     chunks={seg_inp.shape[0]}  budget={config.max_iter_per_phase}")
        solo_ok, solo_iters, solo_accs, params, opt_state = _tbptt_phase(
            f"solo s{seg_idx+1}", seg_inp, seg_tgt, [seg], params, opt_state)

        acc_s  = solo_accs[0][0] if solo_accs else 0.0
        fw_s   = solo_accs[0][1] if solo_accs else None
        fw_str = f"tok {fw_s}" if fw_s is not None else "ok"
        elapsed = time.perf_counter() - t_seg
        if solo_ok:
            print(f"[PASS] solo  seg{seg_idx+1}  acc=100%  iters={solo_iters}  {elapsed:.1f}s")
        else:
            print(f"[FAIL] solo  seg{seg_idx+1}  acc={acc_s:.2%}  first_wrong={fw_str}"
                  f"  iters={solo_iters}  {elapsed:.1f}s")
            failed_seg  = seg_idx + 1
            stop_reason = f"FAIL at segment {failed_seg}"
            break

        completed.append(seg)

        if len(completed) == 1:
            continue

        # ── COMBINED phase ────────────────────────────────────────────────────
        combined = np.concatenate(completed)
        comb_inp, comb_tgt = make_chunks_and_targets(combined, config.segment_size, config.output_heads)
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
            failed_seg  = seg_idx + 1
            stop_reason = f"FAIL at segment {failed_seg} (combined)"
            break

        save_checkpoint(os.path.join(log_dir, "ckpt_last"), params, config, vocab, seg_idx, "combined")

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    if stop_reason is None:
        stop_reason = f"all {n_segs} segments mastered"
    print(f"\n[STOP] {stop_reason}  total_iters={global_iters}  elapsed={elapsed_total:.1f}s")

    # ── final generative eval + checkpoint ──────────────────────────────────
    print("\nFinal generative eval...")
    gen_toks_out, gen_wrong, gen_acc, gen_fw, residual = eval_generative(
        base_xlstm, params, config, all_tokens, vocab, tokens_per_byte)
    gen_fw_str  = f"byte {gen_fw}" if gen_fw is not None else "none (perfect)"
    residual_kb = len(residual) * bytes_per_entry / 1024
    print(f"  gen_wrong={gen_wrong}  gen_acc={gen_acc:.4%}  first_wrong={gen_fw_str}")
    print(f"  corrections={len(residual)} entries  {residual_kb:.2f} KB  "
          f"budget={max_entries} entries ({budget_bytes_} bytes)")
    print(f"  reconstructable={'YES' if gen_wrong <= max_entries else 'NO'}")

    # Decode generated and target to bytes for writing
    gen_bytes_out    = decode_unigram(gen_toks_out, vocab)
    target_bytes_out = decode_unigram(all_tokens[1:], vocab)

    gen_path = os.path.join(log_dir, "gen_final.txt")
    with open(gen_path, "w", encoding="utf-8", errors="replace") as gf:
        gf.write(f"stop_reason:  {stop_reason}\n"
                 f"gen_wrong:    {gen_wrong}\n"
                 f"gen_acc:      {gen_acc:.4%}\n"
                 f"first_wrong:  {gen_fw_str}\n"
                 f"corrections:  {len(residual)} entries  {residual_kb:.2f} KB\n"
                 f"budget:       {max_entries} entries  {budget_bytes_} bytes\n"
                 f"\n--- target ---\n"
                 f"{bytes(target_bytes_out.tolist()).decode('utf-8', errors='replace')}\n"
                 f"\n--- generated ---\n"
                 f"{bytes(gen_bytes_out.tolist()).decode('utf-8', errors='replace')}\n")

    ckpt_dir = os.path.join(log_dir, "ckpt_last")
    save_checkpoint(ckpt_dir, params, config, vocab, 0, "final")
    _save_residual(ckpt_dir, residual)

    print(f"\nLog: {log_path}")
    log_file.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
