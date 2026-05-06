"""
decompress.py — reconstruct a file from a randcompress checkpoint folder.

Usage:
    uv run python examples/decompress.py <checkpoint_dir> [output_file]

The checkpoint folder must contain:
    config.json   — model config + generation settings
    params.pkl    — trained HiRA adapter weights
    seed.bin      — seed bytes used to prime generation

If output_file is omitted, writes to <checkpoint_dir>/reconstructed.txt.
"""

import sys
import os
import json
import pickle
import math
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm

DTYPE = jnp.float32


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

def causal_conv1d_step(x_t, conv_buf, w):
    w      = w.astype(x_t.dtype)
    window = jnp.concatenate([conv_buf, x_t[:,None,:]], axis=1)
    return jnp.einsum('bkc,ck->bc', window, w), window[:,1:,:]

def gated_ffn(ffn_up, ffn_down, x):
    h    = jnp.dot(x, ffn_up.T)
    d_ff = h.shape[-1] // 2
    return jnp.dot(jax.nn.gelu(h[...,:d_ff]) * h[...,d_ff:], ffn_down.T)


# ============================================================================
# mLSTM cell (step only)
# ============================================================================

def mlstm_cell_step(q, k, v, i_preact, f_preact, C_prev, n_prev, m_prev):
    DH    = q.shape[-1]
    log_f = jax.nn.log_sigmoid(f_preact)
    m_t   = jnp.maximum(log_f + m_prev, i_preact)
    f_t   = jnp.exp(log_f + m_prev - m_t)
    i_t   = jnp.exp(i_preact - m_t)
    k_s   = k / jnp.sqrt(jnp.array(DH, k.dtype))
    kv    = jnp.einsum('bnd,bne->bnde', k_s, v)
    C_t   = f_t[:,:,None,None]*C_prev + i_t[:,:,None,None]*kv
    n_t   = f_t[:,:,None]*n_prev + i_t[:,:,None]*k_s
    h_num = jnp.einsum('bnd,bnde->bne', q, C_t)
    denom = jnp.maximum(jnp.abs((q*n_t).sum(axis=-1)), jnp.exp(-m_t))
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
# HiRA application
# ============================================================================

def apply_hira(base, hira):
    return base + base * jnp.dot(hira.B, hira.A)

def apply_mlstm_hira(p, l):
    NH, DH, inner = p.W_q.shape
    dm = NH * DH
    def qkv(w, a):
        return apply_hira(w.reshape(dm, inner), a).reshape(NH, DH, inner)
    return mLSTMParams(
        W_up=apply_hira(p.W_up, l.W_up),
        W_q=qkv(p.W_q, l.W_q), W_k=qkv(p.W_k, l.W_k), W_v=qkv(p.W_v, l.W_v),
        W_i=apply_hira(p.W_i, l.W_i), W_f=apply_hira(p.W_f, l.W_f),
        b_f=p.b_f, b_i=p.b_i, skip=p.skip, ln_w=p.ln_w, ln_b=p.ln_b,
        W_down=apply_hira(p.W_down, l.W_down), conv_w=p.conv_w,
    )

def apply_slstm_hira(p, l):
    return sLSTMParams(
        W_i=apply_hira(p.W_i, l.W_i), W_f=apply_hira(p.W_f, l.W_f),
        W_z=apply_hira(p.W_z, l.W_z), W_o=apply_hira(p.W_o, l.W_o),
        Ry=p.Ry, b=p.b, gn_w=p.gn_w, gn_b=p.gn_b, conv_w=p.conv_w,
    )

def apply_block_hira(base, hira):
    return xLSTMBlockParams(
        norm1_w=base.norm1_w, norm1_b=base.norm1_b,
        norm2_w=base.norm2_w, norm2_b=base.norm2_b,
        ffn_up=apply_hira(base.ffn_up, hira.ffn_up),
        ffn_down=apply_hira(base.ffn_down, hira.ffn_down),
        mlstm=apply_mlstm_hira(base.mlstm, hira.mlstm) if base.mlstm is not None else None,
        slstm=apply_slstm_hira(base.slstm, hira.slstm) if base.slstm is not None else None,
    )

def apply_xlstm_hira(base, rc):
    return xLSTMParams(
        embedding=apply_hira(base.embedding, rc.emb_hira),
        blocks=[apply_block_hira(b, l) for b, l in zip(base.blocks, rc.blocks)],
        norm_w=base.norm_w, norm_b=base.norm_b,
    )


# ============================================================================
# mLSTM / sLSTM step layers
# ============================================================================

def _mlstm_preacts_step(p, x_conv_a, x_m):
    q   = jnp.einsum('ndi,bi->bnd', p.W_q, x_conv_a)
    k   = jnp.einsum('ndi,bi->bnd', p.W_k, x_conv_a)
    v   = jnp.einsum('ndi,bi->bnd', p.W_v, x_m)
    i_p = jnp.einsum('ni,bi->bn', p.W_i, x_conv_a) + p.b_i[None, :]
    f_p = jnp.einsum('ni,bi->bn', p.W_f, x_conv_a) + p.b_f[None, :]
    return q, k, v, i_p, f_p

def mlstm_layer_step(p, x_t, state, num_heads):
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
    h_norm = multihead_layer_norm(h_out, p.ln_w, p.ln_b, num_heads)
    return jnp.dot(h_norm, p.W_down.T), (C_t, n_t, m_t, new_buf)

def slstm_layer_step(p, x_t, state, num_heads):
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
# Forward step + zero-state init
# ============================================================================

def init_step_states(config, batch_size):
    B, K   = batch_size, config.conv_kernel
    NH, DH = config.num_heads, config.d_model // config.num_heads
    states = []
    for btype in config.block_map:
        if btype == 'm':
            inner = config.d_model
            states.append((
                jnp.zeros((B,NH,DH,DH), DTYPE),
                jnp.zeros((B,NH,DH),    DTYPE),
                jnp.zeros((B,NH),       DTYPE),
                jnp.zeros((B,K-1,inner),DTYPE),
            ))
        else:
            H = config.d_model
            states.append((
                jnp.zeros((B,H),                 DTYPE),
                jnp.zeros((B,H),                 DTYPE),
                jnp.zeros((B,H),                 DTYPE),
                jnp.zeros((B,H),                 DTYPE),
                jnp.zeros((B,K-1,config.d_model),DTYPE),
            ))
    return states

def forward_step(base_xlstm, rc, token, states, num_heads, block_map):
    xlstm      = apply_xlstm_hira(base_xlstm, rc)
    x          = xlstm.embedding[token]
    new_states = []
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

_forward_step_jit = jax.jit(forward_step, static_argnames=["num_heads", "block_map"])


# ============================================================================
# Frozen xLSTM init (must match compressor exactly)
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
    return _bf(np.array([basis[c % K] for c in range(n_channels)]))

def _multiscale_forget_bias(n, seq_len):
    tau = np.exp(np.linspace(0.0, np.log(max(float(seq_len), 2.0)), n))
    f   = np.exp(-1.0 / tau)
    return _bf(np.log(f / (1.0 - f + 1e-9)))

def init_mlstm_params(key, d_model, num_heads, conv_kernel, seq_len):
    inner = d_model; DH = inner // num_heads
    ks = jr.split(key, 10)
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
    return mLSTMParams(W_up=W_up, W_q=W_q, W_k=W_k, W_v=W_v,
                       W_i=W_i, W_f=W_f, b_f=b_f, b_i=b_i,
                       skip=jnp.ones((inner,),DTYPE), ln_w=jnp.ones((inner,),DTYPE),
                       ln_b=jnp.zeros((inner,),DTYPE), W_down=W_down, conv_w=conv_w)

def init_slstm_params(key, d_model, conv_kernel, seq_len):
    H = d_model; ks = jr.split(key, 7)
    W_i = _ortho(ks[0], H, d_model) / math.sqrt(d_model)
    W_f = _ortho(ks[1], H, d_model) / math.sqrt(d_model)
    W_z = _ortho(ks[2], H, d_model) / math.sqrt(d_model)
    W_o = _ortho(ks[3], H, d_model) / math.sqrt(d_model)
    raw_R = np.array(jr.normal(ks[4], (4*H, H)))
    U, _, Vt = np.linalg.svd(raw_R, full_matrices=False)
    Ry = _bf(0.95 * (U @ Vt))
    b_arr = np.zeros(4*H); b_arr[H:2*H] = np.array(_multiscale_forget_bias(H, seq_len))
    return sLSTMParams(W_i=W_i, W_f=W_f, W_z=W_z, W_o=W_o, Ry=Ry, b=_bf(b_arr),
                       gn_w=jnp.ones((H,),DTYPE), gn_b=jnp.zeros((H,),DTYPE),
                       conv_w=_fir_bank(d_model, conv_kernel))

def init_xlstm_block_params(key, d_model, num_heads, d_ff, conv_kernel, seq_len, btype):
    ks = jr.split(key, 4)
    ffn_up_raw = np.array(jr.normal(ks[2], (2*d_ff, d_model)))
    ffn_up_raw /= np.linalg.norm(ffn_up_raw, axis=1, keepdims=True)
    ffn_up   = _bf(ffn_up_raw / math.sqrt(d_model))
    ffn_down = _ortho(ks[3], d_model, d_ff) / math.sqrt(d_ff)
    return xLSTMBlockParams(
        norm1_w=jnp.ones((d_model,),DTYPE), norm1_b=jnp.zeros((d_model,),DTYPE),
        norm2_w=jnp.ones((d_model,),DTYPE), norm2_b=jnp.zeros((d_model,),DTYPE),
        ffn_up=ffn_up, ffn_down=ffn_down,
        mlstm=init_mlstm_params(ks[0], d_model, num_heads, conv_kernel, seq_len) if btype=='m' else None,
        slstm=init_slstm_params(ks[1], d_model, conv_kernel, seq_len) if btype=='s' else None,
    )

def init_xlstm_params(key, config):
    ks  = jr.split(key, 2 + config.num_layers)
    emb = _bf(np.array(jr.normal(ks[0], (config.vocab_size, config.d_model))) * 0.01)
    blocks = [
        init_xlstm_block_params(ks[2+i], config.d_model, config.num_heads, config.d_ff,
                                 config.conv_kernel, config.max_seq_len, config.block_map[i])
        for i in range(config.num_layers)
    ]
    return xLSTMParams(embedding=emb, blocks=blocks,
                       norm_w=jnp.ones((config.d_model,),DTYPE),
                       norm_b=jnp.zeros((config.d_model,),DTYPE))


# ============================================================================
# Checkpoint load
# ============================================================================

def load_checkpoint(ckpt_dir):
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
# Generation
# ============================================================================

def _warmup_state(base_xlstm, params, config, context_tokens):
    states = init_step_states(config, batch_size=1)
    if len(context_tokens) == 0:
        return states, None
    cur_token = jnp.array([context_tokens[0]])
    for t in context_tokens[1:]:
        _, states = _forward_step_jit(
            base_xlstm, params, cur_token, states, config.num_heads, config.block_map)
        cur_token = jnp.array([t])
    return states, cur_token


def generate(base_xlstm, params, config, seed_bytes, n_output,
             chunk_size, warmup_tokens):
    assert warmup_tokens < chunk_size
    history       = list(seed_bytes)
    step_in_chunk = 0

    states, cur_token = _warmup_state(base_xlstm, params, config,
                                      history[:warmup_tokens] if warmup_tokens else [])
    if cur_token is None:
        cur_token = jnp.array([history[0]])

    for sb in history[1:]:
        _, states = _forward_step_jit(
            base_xlstm, params, cur_token, states, config.num_heads, config.block_map)
        step_in_chunk += 1
        if step_in_chunk >= chunk_size:
            ctx = history[max(0, len(history) - warmup_tokens):] if warmup_tokens else []
            states, wt = _warmup_state(base_xlstm, params, config, ctx)
            step_in_chunk = len(ctx)
            if wt is not None:
                cur_token = wt
        cur_token = jnp.array([sb])

    result = []
    for _ in tqdm(range(n_output), desc="generating", unit="byte"):
        if step_in_chunk >= chunk_size:
            ctx = history[max(0, len(history) - warmup_tokens):] if warmup_tokens else []
            states, wt = _warmup_state(base_xlstm, params, config, ctx)
            step_in_chunk = len(ctx)
            if wt is not None:
                cur_token = wt
        logit, states = _forward_step_jit(
            base_xlstm, params, cur_token, states, config.num_heads, config.block_map)
        step_in_chunk += 1
        cur_token = jnp.argmax(logit, axis=-1)
        tok = int(cur_token[0])
        result.append(tok)
        history.append(tok)

    return bytes(result)


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ckpt_dir    = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ckpt_dir, "reconstructed.bin")

    print(f"Loading checkpoint from {ckpt_dir}/")
    config, params, seed_bytes, n_seed, n_output, warmup_tokens = load_checkpoint(ckpt_dir)
    chunk_size = config.max_seq_len

    print(f"Config: d_model={config.d_model}  layers={config.num_layers}  "
          f"block_map={config.block_map}  chunk_size={chunk_size}")
    print(f"Seed: {n_seed} bytes  Output: {n_output} bytes  warmup: {warmup_tokens}")

    print("Rebuilding frozen xLSTM basis (deterministic from seed)...")
    key = jr.key(config.seed)
    _, k_xlstm, _ = jr.split(key, 3)
    base_xlstm = init_xlstm_params(k_xlstm, config)

    seed_text = bytes(seed_bytes).decode("utf-8", errors="replace")
    print(f"Seed text: {seed_text!r}")

    output_bytes = generate(base_xlstm, params, config, seed_bytes, n_output,
                            chunk_size=chunk_size, warmup_tokens=warmup_tokens)

    with open(output_path, "wb") as f:
        f.write(output_bytes)
    print(f"Reconstructed {n_output} bytes -> {output_path}")

    try:
        text = output_bytes.decode("utf-8", errors="replace")
        print(f"\n--- reconstructed text ---\n{text}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
