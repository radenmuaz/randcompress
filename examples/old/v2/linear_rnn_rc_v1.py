"""
linear_rnn_rc_v1 — single-file train / compress / decompress

Modes (--mode):
  train      — overfit HiRA adapters to a file, save bundle
  compress   — encode a file with trained params, write bundle
  decompress — AR decode from bundle, write recovered file

Usage
-----
  uv run python examples/linear_rnn_rc_v1.py --mode train \\
      --dataset datasets/juz1.txt --log_dir log/juz1_v1

  uv run python examples/linear_rnn_rc_v1.py --mode compress \\
      --params  log/juz1_v1/ckpt_last \\
      --input   datasets/juz1.txt \\
      --output  log/juz1_v1/ckpt_last \\
      --scheme  canonical-uint64

  uv run python examples/linear_rnn_rc_v1.py --mode decompress \\
      --bundle  log/juz1_v1/ckpt_last \\
      --output  /tmp/juz1_recovered.bin \\
      --verify  datasets/juz1.txt

RC schemes (--scheme):
  canonical-uint64  JAX JIT, 64-bit state, RC_PREC=16  [default]
  canonical-c       ctypes rc_codec.c, same stream as uint64, fastest
  canonical-uint32  numpy, 32-bit state, RC_PREC=16
  canonical-uint16  numpy, 16-bit state, RC_PREC=8  (lossy CDFs for V=256)

Bundle layout:
  config.json      — full Config dict
  params.pkl       — HiRA adapter weights
  rc_stream.bin    — range-coded stream
  meta.json        — {n_raw_bytes, T_valid, seed_token, rc_bytes, rc_scheme, ...}
"""

import jax
jax.config.update("jax_enable_x64", True)
import warnings as _w
_w.filterwarnings("ignore", message="scatter inputs have incompatible types",
                  category=FutureWarning)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import NamedTuple, Optional
import math, sys, time, os, functools, json, pickle, hashlib, argparse, shutil, ctypes as _ct
from tqdm import tqdm

PAD_TOKEN = 0
DTYPE     = jnp.float32
POWER_P   = 2
RC_PREC   = 16

class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, s):
        for f in self.files: f.write(s); f.flush()
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
    d_model:       int   = 64
    num_heads:     int   = 8
    num_layers:    int   = 4
    segment_size:  int   = 1024
    power_p:       int   = 2
    seed:          int   = 0
    stride_map:    str   = "1111"
    block_map:     str   = "smmm"
    lora_r:        int   = 4
    batch_size:    int   = 1
    learning_rate: float = 1e-2
    weight_decay:  float = 0.0
    grad_clip_norm:float = 1e2
    sinkgd_l:      int   = 5
    margin:        float = 0.
    ce_weight:     float = 1.0
    grad_accum_steps: int   = 1
    state_reset_prob: float = 0.0
    residual_budget: float = 0.
    target_bpb:    float = 0.0
    gen_seed_len:  int   = 16
    pad_token:       int   = 0
    dtype:           str   = "float32"
    max_iters:       int   = 100
    check_every:     int   = 100
    max_eval_tokens: int   = 0
    remat:           bool  = True
    remat_chunk:     int   = 50000
    save_ckpt:       bool  = True
    dataset:       str   = "datasets/quran-uthmani.txt"

_PRESETS = {
    "byte":      dict(input_bits=8, output_bits=8, output_heads=1, vocab_size=256, pad_token=0),
    "nibble":    dict(input_bits=4, output_bits=4, output_heads=1, vocab_size=16,  pad_token=-1),
    "binary":    dict(input_bits=1, output_bits=1, output_heads=1, vocab_size=2,   pad_token=-1),
}

def _parse_stride_map(s, n_layers):
    strides = [int(c) for c in s]
    if len(strides) < n_layers: strides += [1] * (n_layers - len(strides))
    return tuple(strides[:n_layers])

def bytes_to_tokens(raw, bits):
    if bits == 8: return raw.astype(np.int32)
    if bits == 4:
        hi = ((raw >> 4) & 0xF).astype(np.int32)
        lo = (raw & 0xF).astype(np.int32)
        return np.stack([hi, lo], axis=-1).ravel()
    return np.unpackbits(raw).astype(np.int32)

def tokens_to_bytes(toks, bits):
    if bits == 8: return toks.astype(np.uint8)
    if bits == 4:
        hi = (toks[0::2].astype(np.uint8) & 0xF) << 4
        lo = toks[1::2].astype(np.uint8) & 0xF
        return hi | lo
    return np.packbits(toks.astype(np.uint8))

def validate_config(config, n_dataset_bytes):
    warns = []
    ib, ob, oh = config.input_bits, config.output_bits, config.output_heads
    if 2**ib != config.vocab_size: warns.append(f"vocab_size mismatch")
    if ib not in (1,4,8): warns.append(f"input_bits={ib} not in {{1,4,8}}")
    if ib < 8 and config.pad_token == 0: warns.append(f"sub-byte but pad_token=0")
    for w in warns: print(f"  [CONFIG WARN] {w}")
    return warns


# ============================================================================
# Model: mLSTM + sRNN parameter structures
# ============================================================================

class mLSTMParams(NamedTuple):
    W_q:jnp.ndarray; W_k:jnp.ndarray; W_v:jnp.ndarray; alpha:jnp.ndarray
    skip:jnp.ndarray; ln_w:jnp.ndarray; ln_b:jnp.ndarray; W_down:jnp.ndarray

class sRNNParams(NamedTuple):
    W_hx:jnp.ndarray; W_hh:jnp.ndarray; b:jnp.ndarray
    ln_w:jnp.ndarray; ln_b:jnp.ndarray

class xLSTMBlockParams(NamedTuple):
    norm_w:jnp.ndarray; norm_b:jnp.ndarray
    mlstm:Optional[mLSTMParams]=None; srnn:Optional[sRNNParams]=None

class xLSTMParams(NamedTuple):
    embedding:jnp.ndarray; blocks:list; norm_w:jnp.ndarray; norm_b:jnp.ndarray

class HiRAAdapter(NamedTuple):
    A:jnp.ndarray; B:jnp.ndarray

class mLSTMHiRA(NamedTuple):
    W_q:HiRAAdapter; W_k:HiRAAdapter; W_v:HiRAAdapter; W_down:HiRAAdapter

class sRNNHiRA(NamedTuple):
    W_hx:HiRAAdapter; W_hh:HiRAAdapter

class BlockHiRA(NamedTuple):
    mlstm:Optional[mLSTMHiRA]=None; srnn:Optional[sRNNHiRA]=None

class RandCompressParams(NamedTuple):
    output_proj:jnp.ndarray; emb_hira:HiRAAdapter; blocks:list


# ============================================================================
# HiRA application
# ============================================================================

def apply_hira(base, h): return base + base * jnp.dot(h.B, h.A)

def apply_mlstm_hira(p, l):
    NH, DH, d = p.W_q.shape; dm = NH*DH
    def qkv(w, a): return apply_hira(w.reshape(dm,d), a).reshape(NH,DH,d)
    return mLSTMParams(W_q=qkv(p.W_q,l.W_q), W_k=qkv(p.W_k,l.W_k), W_v=qkv(p.W_v,l.W_v),
        alpha=p.alpha, skip=p.skip, ln_w=p.ln_w, ln_b=p.ln_b, W_down=apply_hira(p.W_down,l.W_down))

def apply_srnn_hira(p, l):
    return sRNNParams(W_hx=apply_hira(p.W_hx,l.W_hx), W_hh=apply_hira(p.W_hh,l.W_hh),
        b=p.b, ln_w=p.ln_w, ln_b=p.ln_b)

def apply_block_hira(base, hira):
    return xLSTMBlockParams(norm_w=base.norm_w, norm_b=base.norm_b,
        mlstm=apply_mlstm_hira(base.mlstm,hira.mlstm) if base.mlstm else None,
        srnn =apply_srnn_hira(base.srnn,  hira.srnn)  if base.srnn  else None)

def apply_xlstm_hira(base, rc):
    return xLSTMParams(embedding=apply_hira(base.embedding, rc.emb_hira),
        blocks=[apply_block_hira(b,l) for b,l in zip(base.blocks,rc.blocks)],
        norm_w=base.norm_w, norm_b=base.norm_b)


# ============================================================================
# Power attention helpers
# ============================================================================

def logsigmoid(x): return -jnp.logaddexp(0.0, -x)

def layer_norm(x, w, b, eps=1e-6):
    mean=x.mean(axis=-1,keepdims=True); var=x.var(axis=-1,keepdims=True)
    return (x-mean)/jnp.sqrt(var+eps)*w+b

def multihead_layer_norm(x, w, b, num_heads, eps=1e-6):
    *lead,D=x.shape; x_r=x.reshape(*lead,num_heads,D//num_heads)
    mean=x_r.mean(axis=-1,keepdims=True); var=x_r.var(axis=-1,keepdims=True)
    return ((x_r-mean)/jnp.sqrt(var+eps)).reshape(*lead,D)*w+b

_SPOW_CACHE = {}

def spow(x, p):
    if p==1: return x
    D=x.shape[-1]; key=(D,p)
    if key not in _SPOW_CACHE:
        i_idx,j_idx=np.tril_indices(D)
        scale=np.where(i_idx==j_idx,1.0,np.sqrt(2.0)).astype(np.float32)
        _SPOW_CACHE[key]=(i_idx,j_idx,scale)
    i_idx,j_idx,scale=_SPOW_CACHE[key]
    return x[...,i_idx]*x[...,j_idx]*scale

def _dsym(DH, p):
    if p==1: return DH
    return math.comb(DH+p-1, p)


# ============================================================================
# mLSTM / sRNN layers
# ============================================================================

def mlstm_layer_scan_with_init(p, x, num_heads, state):
    C0=state; B,S,d=x.shape; DH=d//num_heads
    q=jnp.einsum('ndi,bsi->bnsd',p.W_q,x); k=jnp.einsum('ndi,bsi->bnsd',p.W_k,x)
    v=jnp.einsum('ndi,bsi->bnsd',p.W_v,x)
    def step(C, t_in):
        q_t,k_t,v_t=t_in; k_s=k_t/jnp.sqrt(jnp.array(DH,k_t.dtype))
        phi_k=spow(k_s,POWER_P); phi_q=spow(q_t,POWER_P)
        kv=jnp.einsum('bnd,bne->bnde',phi_k,v_t)
        C_new=p.alpha[:,None,None]*C+kv
        h_num=jnp.einsum('bnd,bnde->bne',phi_q,C_new)
        return C_new, h_num
    C_T,hs=jax.lax.scan(step,C0,(q.transpose(2,0,1,3),k.transpose(2,0,1,3),v.transpose(2,0,1,3)))
    h_flat=hs.transpose(1,2,0,3).reshape(B,S,d)
    h_out=h_flat+p.skip*x; h_norm=multihead_layer_norm(h_out,p.ln_w,p.ln_b,num_heads)
    return jnp.dot(h_norm,p.W_down.T), C_T

def mlstm_layer_scan(p, x, num_heads):
    B,S,d=x.shape; DH=d//num_heads; D_sym=_dsym(DH,POWER_P)
    C0=jnp.zeros((B,num_heads,D_sym,DH),x.dtype)
    out,_=mlstm_layer_scan_with_init(p,x,num_heads,C0); return out

def mlstm_layer_step(p, x_t, state, num_heads):
    C=state; B,d=x_t.shape; DH=d//num_heads
    q=jnp.einsum('ndi,bi->bnd',p.W_q,x_t); k=jnp.einsum('ndi,bi->bnd',p.W_k,x_t)
    v=jnp.einsum('ndi,bi->bnd',p.W_v,x_t)
    k_s=k/jnp.sqrt(jnp.array(DH,k.dtype)); phi_k=spow(k_s,POWER_P); phi_q=spow(q,POWER_P)
    kv=jnp.einsum('bnd,bne->bnde',phi_k,v); C_t=p.alpha[:,None,None]*C+kv
    h_num=jnp.einsum('bnd,bnde->bne',phi_q,C_t); h_flat=h_num.reshape(B,d)
    h_out=h_flat+p.skip*x_t; h_norm=multihead_layer_norm(h_out,p.ln_w,p.ln_b,num_heads)
    return jnp.dot(h_norm,p.W_down.T), C_t

def srnn_layer_scan_with_init(p, x, num_heads, state):
    h0=state
    def step(h, x_t):
        pre=jnp.dot(x_t,p.W_hx.T)+jnp.dot(h,p.W_hh.T)+p.b
        h_new=layer_norm(pre,p.ln_w,p.ln_b); return h_new, h_new
    h_T,ys=jax.lax.scan(step,h0,x.transpose(1,0,2)); return ys.transpose(1,0,2), h_T

def srnn_layer_scan(p, x, num_heads):
    B=x.shape[0]; h0=jnp.zeros((B,p.W_hx.shape[0]),x.dtype)
    out,_=srnn_layer_scan_with_init(p,x,num_heads,h0); return out

def srnn_layer_step(p, x_t, state, num_heads):
    h=state; pre=jnp.dot(x_t,p.W_hx.T)+jnp.dot(h,p.W_hh.T)+p.b
    h_t=layer_norm(pre,p.ln_w,p.ln_b); return h_t, h_t


# ============================================================================
# Forward passes
# ============================================================================

def forward_train(base_xlstm, rc, tokens, num_heads, block_map, stride_map):
    xlstm=apply_xlstm_hira(base_xlstm,rc); x=xlstm.embedding[tokens]; S=x.shape[1]
    for block,btype,stride in zip(xlstm.blocks,block_map,stride_map):
        x_n=layer_norm(x,block.norm_w,block.norm_b)
        if stride>1:
            x_sub=x_n[:,::stride,:]
            cell_out_sub=(mlstm_layer_scan(block.mlstm,x_sub,num_heads)
                          if btype=='m' else srnn_layer_scan(block.srnn,x_sub,num_heads))
            cell_out=jnp.repeat(cell_out_sub,stride,axis=1)[:,:S,:]
        else:
            cell_out=(mlstm_layer_scan(block.mlstm,x_n,num_heads)
                      if btype=='m' else srnn_layer_scan(block.srnn,x_n,num_heads))
        x=x+cell_out
    x=layer_norm(x,xlstm.norm_w,xlstm.norm_b); return jnp.dot(x,rc.output_proj)

def forward_train_chunked(base_xlstm, rc, inputs, layer_states, num_heads, block_map, stride_map):
    xlstm=apply_xlstm_hira(base_xlstm,rc); x=xlstm.embedding[inputs]; S=x.shape[1]; new_states=[]
    for block,btype,stride,state in zip(xlstm.blocks,block_map,stride_map,layer_states):
        lstm_state,_last_out=state; x_n=layer_norm(x,block.norm_w,block.norm_b)
        if stride>1:
            x_sub=x_n[:,::stride,:]
            if btype=='m': cell_out_sub,new_lstm=mlstm_layer_scan_with_init(block.mlstm,x_sub,num_heads,lstm_state)
            else:           cell_out_sub,new_lstm=srnn_layer_scan_with_init(block.srnn,x_sub,num_heads,lstm_state)
            cell_out=jnp.repeat(cell_out_sub,stride,axis=1)[:,:S,:]
        else:
            if btype=='m': cell_out,new_lstm=mlstm_layer_scan_with_init(block.mlstm,x_n,num_heads,lstm_state)
            else:           cell_out,new_lstm=srnn_layer_scan_with_init(block.srnn,x_n,num_heads,lstm_state)
        new_states.append((new_lstm, cell_out[:,-1,:]))
        x=x+cell_out
    x=layer_norm(x,xlstm.norm_w,xlstm.norm_b)
    return jnp.dot(x,rc.output_proj), new_states

def forward_step(base_xlstm, rc, token, states, num_heads, block_map, stride_map, t):
    xlstm=apply_xlstm_hira(base_xlstm,rc); x=xlstm.embedding[token]; new_states=[]
    for block,btype,stride,state in zip(xlstm.blocks,block_map,stride_map,states):
        lstm_state,last_out=state; x_n=layer_norm(x,block.norm_w,block.norm_b)
        if stride==1:
            if btype=='m': cell_out,new_lstm=mlstm_layer_step(block.mlstm,x_n,lstm_state,num_heads)
            else:           cell_out,new_lstm=srnn_layer_step(block.srnn,x_n,lstm_state,num_heads)
            new_states.append((new_lstm,cell_out))
        else:
            def fire(args):
                xn,ls=args
                if btype=='m': return mlstm_layer_step(block.mlstm,xn,ls,num_heads)
                return srnn_layer_step(block.srnn,xn,ls,num_heads)
            def hold(args): _,ls=args; return last_out,ls
            cell_out,new_lstm=jax.lax.cond(t%stride==0,fire,hold,(x_n,lstm_state))
            new_states.append((new_lstm,cell_out))
        x=x+cell_out
    x=layer_norm(x,xlstm.norm_w,xlstm.norm_b)
    return jnp.dot(x,rc.output_proj), new_states

_fwd_step_jit    = jax.jit(forward_step,          static_argnames=["num_heads","block_map","stride_map"])
_fwd_chunked_jit = jax.jit(forward_train_chunked,  static_argnames=["num_heads","block_map","stride_map"])

@functools.partial(jax.jit, static_argnames=["chunk_size","num_heads","block_map","stride_map","oh","ob"])
def _collect_chunk_jit(base_xlstm, params, toks_chunk, init_states,
                        chunk_size, num_heads, block_map, stride_map, oh, ob):
    xlstm=apply_xlstm_hira(base_xlstm,params); stride_tup=stride_map
    def step(states, t_tok):
        t,tok=t_tok; x=xlstm.embedding[tok][None,:]; new_states=[]
        for block,btype,stride,state in zip(xlstm.blocks,block_map,stride_tup,states):
            lstm_state,last_out=state; x_n=layer_norm(x,block.norm_w,block.norm_b)
            if stride==1:
                if btype=='m': cell_out,new_lstm=mlstm_layer_step(block.mlstm,x_n,lstm_state,num_heads)
                else:           cell_out,new_lstm=srnn_layer_step(block.srnn,x_n,lstm_state,num_heads)
                new_states.append((new_lstm,cell_out))
            else:
                _block,_btype=block,btype
                def fire(args):
                    xn,ls=args
                    if _btype=='m': return mlstm_layer_step(_block.mlstm,xn,ls,num_heads)
                    return srnn_layer_step(_block.srnn,xn,ls,num_heads)
                def hold(args): _,ls=args; return last_out,ls
                cell_out,new_lstm=jax.lax.cond(t%jnp.int32(stride)==jnp.int32(0),fire,hold,(x_n,lstm_state))
                new_states.append((new_lstm,cell_out))
            x=x+cell_out
        x=layer_norm(x,xlstm.norm_w,xlstm.norm_b); logit_flat=jnp.dot(x,params.output_proj)
        logits=_reshape_logits(logit_flat,oh,ob); return new_states, logits[0,0,:]
    new_states,chunk_logits=jax.lax.scan(step,init_states,
        (jnp.arange(chunk_size,dtype=jnp.int32),toks_chunk))
    return chunk_logits, new_states

def _reshape_logits(flat, output_heads, output_bits):
    return flat.reshape(*flat.shape[:-1], output_heads, 2**output_bits)


# ============================================================================
# Hidden-state dropout
# ============================================================================

def _dropout_states(states, key, prob):
    new_states=[]
    for i,(lstm_state,last_out) in enumerate(states):
        mask=jr.bernoulli(jr.fold_in(key,i),1.0-prob).astype(DTYPE)
        new_states.append((jax.tree_util.tree_map(lambda s:s*mask,lstm_state),last_out*mask))
    return new_states


# ============================================================================
# K-chunk BPTT
# ============================================================================

@functools.partial(jax.jit, static_argnames=["config","K"])
def grad_accum_persistent(base_xlstm, params, inputs_K, targets_K, init_states, rng_key, config, K):
    oh=config.output_heads; ob=config.output_bits; num_heads=config.num_heads
    block_map=config.block_map; stride_map=_parse_stride_map(config.stride_map,config.num_layers)
    margin=config.margin; ce_weight=config.ce_weight; srp=config.state_reset_prob
    def loss_fn(p):
        states=init_states; total_loss=jnp.zeros(())
        for k in range(K):
            logits_flat,states=forward_train_chunked(base_xlstm,p,inputs_K[k],states,num_heads,block_map,stride_map)
            logits=_reshape_logits(logits_flat,oh,ob)
            chunk_loss=sum(training_loss(logits[...,h,:],targets_K[k][...,h],margin,ce_weight)
                           for h in range(oh))/oh
            total_loss=total_loss+chunk_loss
            if k<K-1 and srp>0.0:
                states=_dropout_states(states,jr.fold_in(rng_key,k),srp)
        return total_loss/K, states
    (loss,new_states),grads=jax.value_and_grad(loss_fn,has_aux=True)(params)
    return loss, grads, new_states


# ============================================================================
# State init
# ============================================================================

def init_step_states(config, batch_size):
    B=batch_size; NH=config.num_heads; DH=config.d_model//config.num_heads; bf=DTYPE; states=[]
    for btype in config.block_map:
        if btype=='m':
            D_sym=_dsym(DH,config.power_p); lstm_state=jnp.zeros((B,NH,D_sym,DH),bf)
        else:
            lstm_state=jnp.zeros((B,config.d_model),bf)
        states.append((lstm_state,jnp.zeros((B,config.d_model),bf)))
    return states


# ============================================================================
# Structured init
# ============================================================================

def _bf(x): return jnp.array(x, dtype=DTYPE)

def _ortho(key, n, m):
    raw=np.array(jr.normal(key,(max(n,m),min(n,m)))); Q,_=np.linalg.qr(raw); mat=Q.T
    if n>m:
        extra=np.array(jr.normal(jr.fold_in(key,1),(n-m,m)))/math.sqrt(m)
        mat=np.concatenate([mat,extra],axis=0)
    return _bf(mat[:n])

def init_mlstm_params(key, d_model, num_heads, seq_len):
    d=d_model; DH=d//num_heads; ks=jr.split(key,5)
    def _qkv(k):
        M=np.array(_ortho(k,d,d)); return _bf(M.reshape(num_heads,DH,d)/math.sqrt(d))
    W_q=_qkv(ks[0]); W_k=_qkv(ks[1]); W_v=_qkv(ks[2])
    tau=np.exp(np.linspace(0.,np.log(max(float(seq_len),2.)),num_heads))
    alpha=_bf(np.exp(-1./tau)); W_down=_ortho(ks[3],d,d)/math.sqrt(d)
    return mLSTMParams(W_q=W_q,W_k=W_k,W_v=W_v,alpha=alpha,
        skip=jnp.ones((d,),DTYPE),ln_w=jnp.ones((d,),DTYPE),ln_b=jnp.zeros((d,),DTYPE),W_down=W_down)

def init_srnn_params(key, d_model, seq_len):
    H=d_model; ks=jr.split(key,3); W_hx=_ortho(ks[0],H,H)/math.sqrt(H)
    raw_hh=np.array(jr.normal(ks[1],(H,H))); U,_,Vt=np.linalg.svd(raw_hh,full_matrices=False)
    W_hh=_bf(0.95*(U@Vt))
    return sRNNParams(W_hx=W_hx,W_hh=W_hh,b=jnp.zeros((H,),DTYPE),
        ln_w=jnp.ones((H,),DTYPE),ln_b=jnp.zeros((H,),DTYPE))

def init_xlstm_block_params(key, d_model, num_heads, seq_len, btype):
    ks=jr.split(key,2)
    return xLSTMBlockParams(norm_w=jnp.ones((d_model,),DTYPE),norm_b=jnp.zeros((d_model,),DTYPE),
        mlstm=init_mlstm_params(ks[0],d_model,num_heads,seq_len) if btype=='m' else None,
        srnn =init_srnn_params(ks[1],d_model,seq_len)              if btype=='s' else None)

def init_xlstm_params(key, config):
    ks=jr.split(key,2+config.num_layers)
    emb=_bf(np.array(jr.normal(ks[0],(config.vocab_size,config.d_model)))*0.01)
    blocks=[init_xlstm_block_params(ks[2+i],config.d_model,config.num_heads,config.segment_size,config.block_map[i])
            for i in range(config.num_layers)]
    return xLSTMParams(embedding=emb,blocks=blocks,
        norm_w=jnp.ones((config.d_model,),DTYPE),norm_b=jnp.zeros((config.d_model,),DTYPE))

def init_hira_adapter(key, d_out, d_in, r):
    raw=np.array(jr.normal(key,(r,d_in)))
    if r<=d_in: _,_,Vt=np.linalg.svd(raw,full_matrices=False); A=_bf(Vt/math.sqrt(d_in))
    else: A=_bf(raw/math.sqrt(d_in))
    return HiRAAdapter(A=A,B=jnp.zeros((d_out,r),DTYPE))

def init_block_hira(key, config, btype):
    d,r=config.d_model,config.lora_r; ks=jr.split(key,4); mhira=srnnhira=None
    if btype=='m':
        mhira=mLSTMHiRA(W_q=init_hira_adapter(ks[0],d,d,r),W_k=init_hira_adapter(ks[1],d,d,r),
            W_v=init_hira_adapter(ks[2],d,d,r),W_down=init_hira_adapter(ks[3],d,d,r))
    else:
        srnnhira=sRNNHiRA(W_hx=init_hira_adapter(ks[0],d,d,r),W_hh=init_hira_adapter(ks[1],d,d,r))
    return BlockHiRA(mlstm=mhira,srnn=srnnhira)

def init_randcompress_params(key, config):
    ks=jr.split(key,2+config.num_layers)
    out_dim=config.output_heads*(2**config.output_bits)
    return RandCompressParams(
        output_proj=_bf(np.array(jr.normal(ks[0],(config.d_model,out_dim)))*0.01),
        emb_hira=init_hira_adapter(ks[1],config.vocab_size,config.d_model,config.lora_r),
        blocks=[init_block_hira(ks[2+i],config,config.block_map[i]) for i in range(config.num_layers)])


# ============================================================================
# SinkGD optimizer
# ============================================================================

def _sr_sinkhorn_2d(W, L):
    m,n=W.shape
    for _ in range(L):
        W=jnp.sqrt(n)*W/(jnp.linalg.norm(W,axis=1,keepdims=True)+1e-8)
        W=jnp.sqrt(m)*W/(jnp.linalg.norm(W,axis=0,keepdims=True)+1e-8)
    return W

def _sinkgd_normalize(g, L):
    g32=g.astype(jnp.float32)
    if g32.ndim==0: return g32
    if g32.ndim==1:
        n=g32.shape[0]; return jnp.sqrt(n)*g32/(jnp.linalg.norm(g32)+1e-8)
    m=g32.shape[0]; n=int(np.prod(g32.shape[1:]))
    return _sr_sinkhorn_2d(g32.reshape(m,n),L).reshape(g32.shape)

def sinkgd_init(params): return None

@functools.partial(jax.jit, static_argnums=(5,))
def _sinkgd_step(grads, params, lr, weight_decay, max_norm, L):
    leaves=jax.tree_util.tree_leaves(grads)
    gnorm=jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32)**2) for g in leaves))
    scale=jnp.minimum(1.0,max_norm/(gnorm+1e-6)); grads=jax.tree_util.tree_map(lambda g:g*scale,grads)
    norm_g=jax.tree_util.tree_map(lambda g:_sinkgd_normalize(g,L),grads)
    return jax.tree_util.tree_map(lambda p,g:(p.astype(jnp.float32)-lr*g-lr*weight_decay*p.astype(jnp.float32)).astype(p.dtype),params,norm_g)

def _center_hira_b(params):
    def fn(x):
        if isinstance(x,HiRAAdapter): return HiRAAdapter(A=x.A,B=x.B-x.B.mean(axis=0,keepdims=True))
        return x
    return jax.tree_util.tree_map(fn,params,is_leaf=lambda x:isinstance(x,HiRAAdapter))

def sinkgd_update(state, grads, params, lr, weight_decay=0.0, max_norm=1.0, L=1):
    new_params=_sinkgd_step(grads,params,jnp.array(lr),jnp.array(weight_decay),jnp.array(max_norm),L)
    return _center_hira_b(new_params), state


# ============================================================================
# Loss
# ============================================================================

def cross_entropy_loss(logits, targets):
    log_probs=jax.nn.log_softmax(logits,axis=-1)
    loss_per_tok=-jnp.take_along_axis(log_probs,jnp.maximum(targets,0)[...,None],axis=-1)[...,0]
    mask=(targets>=0).astype(jnp.float32)
    return (loss_per_tok*mask).sum()/(mask.sum()+1e-8)

def argmax_margin_loss(logits, targets, margin=1.0):
    safe_tgt=jnp.maximum(targets,0)
    target_logits=jnp.take_along_axis(logits,safe_tgt[...,None],axis=-1)[...,0]
    neg_inf=jax.nn.one_hot(safe_tgt,logits.shape[-1])*1e9
    best_wrong=(logits-neg_inf).max(axis=-1)
    loss_per_tok=jnp.maximum(0.0,best_wrong-target_logits+margin)
    mask=(targets>=0).astype(jnp.float32)
    return (loss_per_tok*mask).sum()/(mask.sum()+1e-8)

def training_loss(logits, targets, margin=1.0, ce_weight=0.05):
    if margin==0.0: return cross_entropy_loss(logits,targets)
    return argmax_margin_loss(logits,targets,margin)+ce_weight*cross_entropy_loss(logits,targets)


# ============================================================================
# Data helpers
# ============================================================================

def load_dataset(path):
    with open(path,'rb') as f: return np.frombuffer(f.read(),dtype=np.uint8).copy()

def make_chunks_and_targets(raw_bytes, config):
    toks=bytes_to_tokens(np.asarray(raw_bytes,dtype=np.uint8),config.input_bits)
    chunk_size=config.segment_size*(8//config.input_bits); n=len(toks)
    num_chunks=max(1,math.ceil((n-1)/chunk_size)); oh=config.output_heads; ob=config.output_bits
    out_mask=(1<<ob)-1; pad_len=max(0,num_chunks*chunk_size+oh-n)
    padded=np.concatenate([toks,np.full(pad_len+oh,-1,dtype=np.int32)])
    all_inputs=np.stack([padded[i*chunk_size:i*chunk_size+chunk_size] for i in range(num_chunks)]).astype(np.int32)
    tgt_chunks=[]
    for i in range(num_chunks):
        chunk_tgts=np.zeros((chunk_size,oh),dtype=np.int32)
        if config.input_bits==8 and ob<8:
            next_bytes=padded[i*chunk_size+1:i*chunk_size+chunk_size+1]
            for h in range(oh): chunk_tgts[:,h]=(next_bytes>>(h*ob))&out_mask
        else:
            for h in range(oh): chunk_tgts[:,h]=padded[i*chunk_size+1+h:i*chunk_size+chunk_size+1+h]
        tgt_chunks.append(chunk_tgts)
    return jnp.array(all_inputs,dtype=jnp.int32), jnp.array(np.stack(tgt_chunks),dtype=jnp.int32)


# ============================================================================
# CDF quantization
# ============================================================================

def _quantize_cdf(logits_1d):
    M=jnp.int32(1<<RC_PREC); probs=jax.nn.softmax(logits_1d.astype(jnp.float32))
    freqs=jnp.maximum(jnp.int32(1),jnp.round(probs*M).astype(jnp.int32))
    deficit=M-freqs.sum(); max_idx=jnp.argmax(freqs).astype(jnp.int32)
    adjusted=jnp.maximum(jnp.int32(1),freqs[max_idx]+deficit); freqs=freqs.at[max_idx].set(adjusted)
    cumfreqs=jnp.zeros(logits_1d.shape[0]+1,jnp.int32)
    cumfreqs=cumfreqs.at[1:].set(jnp.cumsum(freqs)); cumfreqs=cumfreqs.at[-1].set(M)
    return cumfreqs

def _cdf_ok(cumfreqs):
    M=jnp.int32(1<<RC_PREC)
    return (cumfreqs[-1]==M) & jnp.all(cumfreqs[1:]>cumfreqs[:-1])


# ============================================================================
# Range coders
# ============================================================================

# ── 64-bit JAX JIT (canonical, top-byte agree) ───────────────────────────────

_U64 = jnp.uint64; _U8 = jnp.uint8
_UINT64_MAX = _U64(0xFFFFFFFFFFFFFFFF)
_RC_MAX_BUF_PER_SYM = 3

def _rc_scale(r, f):
    rhi=(r>>_U64(32)).astype(_U64); rlo=(r&_U64(0xFFFFFFFF)).astype(_U64); f64=f.astype(_U64)
    return (rhi*f64<<_U64(16))+(rlo*f64>>_U64(16))

@functools.partial(jax.jit, static_argnames=["T","V"])
def rc_encode_jax(symbols, all_cumfreqs, T, V):
    max_buf=T*_RC_MAX_BUF_PER_SYM+8
    def _flush_once(carry):
        low,high,buf,pos=carry; agreed=(low>>_U64(56))==(high>>_U64(56))
        byte=(low>>_U64(56)).astype(_U8); in_bounds=pos<jnp.int32(max_buf)
        buf=jax.lax.cond(agreed&in_bounds,lambda b:b.at[pos].set(byte),lambda b:b,buf)
        pos=jax.lax.cond(agreed,lambda p:p+jnp.int32(1),lambda p:p,pos)
        low=jax.lax.cond(agreed,lambda l:(l<<_U64(8)).astype(_U64),lambda l:l,low)
        high=jax.lax.cond(agreed,lambda h:((h<<_U64(8))|_U64(0xFF)).astype(_U64),lambda h:h,high)
        return low,high,buf,pos
    def encode_step(carry,inputs):
        low,high,buf,pos=carry; sym,cf=inputs; sym64=sym.astype(_U64)
        cum_lo=cf[sym64].astype(_U64); cum_hi=cf[sym64+_U64(1)].astype(_U64)
        range_=(high-low+_U64(1)).astype(_U64)
        high=(low+_rc_scale(range_,cum_hi)-_U64(1)).astype(_U64)
        low=(low+_rc_scale(range_,cum_lo)).astype(_U64)
        s=(low,high,buf,pos)
        for _ in range(8): s=_flush_once(s)
        return s,None
    buf=jnp.zeros(max_buf,_U8); init=(_U64(0),_UINT64_MAX-_U64(1),buf,jnp.int32(0))
    (low,_high,buf,pos),_=jax.lax.scan(encode_step,init,(symbols.astype(jnp.int32),all_cumfreqs.astype(jnp.int32)))
    for _ in range(8):
        in_bounds=pos<jnp.int32(max_buf)
        buf=jax.lax.cond(in_bounds,lambda b:b.at[pos].set((low>>_U64(56)).astype(_U8)),lambda b:b,buf)
        pos=jax.lax.cond(in_bounds,lambda p:p+jnp.int32(1),lambda p:p,pos)
        low=(low<<_U64(8)).astype(_U64)
    return buf,pos

@functools.partial(jax.jit, static_argnames=["T","V"])
def rc_decode_jax(buf, all_cumfreqs, T, V):
    max_buf=T*_RC_MAX_BUF_PER_SYM+8; code=_U64(0)
    for i in range(8): code=(code<<_U64(8))|buf[i].astype(_U64)
    def _refill_once(state):
        low,high,code_,pos=state; agreed=(low>>_U64(56))==(high>>_U64(56))
        in_bounds=pos<jnp.int32(max_buf)
        byte=jax.lax.cond(in_bounds,lambda p:buf[p].astype(_U64),lambda p:_U64(0),pos)
        low=jax.lax.cond(agreed,lambda l:(l<<_U64(8)).astype(_U64),lambda l:l,low)
        high=jax.lax.cond(agreed,lambda h:((h<<_U64(8))|_U64(0xFF)).astype(_U64),lambda h:h,high)
        code_=jax.lax.cond(agreed,lambda c:((c<<_U64(8))|byte).astype(_U64),lambda c:c,code_)
        pos=jax.lax.cond(agreed&in_bounds,lambda p:p+jnp.int32(1),lambda p:p,pos)
        return low,high,code_,pos
    def decode_step(carry,cf):
        low,high,code_,pos=carry; range_=(high-low+_U64(1)).astype(_U64); cf64=cf.astype(_U64)
        def bsearch_body(state):
            lo,hi=state; mid=(lo+hi+_U64(1))>>_U64(1)
            boundary=low+_rc_scale(range_,cf64[mid])
            lo=jnp.where(boundary<=code_,mid,lo); hi=jnp.where(boundary<=code_,hi,mid-_U64(1))
            return lo,hi
        sym,_=jax.lax.while_loop(lambda s:s[0]<s[1],bsearch_body,(_U64(0),_U64(V-1)))
        cum_lo=cf64[sym]; cum_hi=cf64[sym+_U64(1)]
        high=(low+_rc_scale(range_,cum_hi)-_U64(1)).astype(_U64); low=(low+_rc_scale(range_,cum_lo)).astype(_U64)
        s=(low,high,code_,pos)
        for _ in range(8): s=_refill_once(s)
        return s,sym.astype(jnp.int32)
    init=(_U64(0),_UINT64_MAX-_U64(1),code,jnp.int32(8))
    _,syms=jax.lax.scan(decode_step,init,all_cumfreqs.astype(jnp.int32))
    return syms

# ── ctypes C codec ────────────────────────────────────────────────────────────

_RC_C_SOURCE = r"""
/* Canonical (low, high) range coder. RC_PREC=16, M=65536.
 * 64-bit state; __uint128_t for multiplications that overflow uint64.
 * Emit a byte when top byte of low and high agree.
 * Init: high=UINT64_MAX so range=2^64 (handled by u128 scale).
 */
#include <stdint.h>
#include <stddef.h>
#define M ((uint64_t)(1U << 16))
typedef unsigned __int128 u128;
static inline uint64_t scale(u128 range, uint64_t f) { return (uint64_t)(range * f / M); }

size_t rc_encode(const int32_t *cdfs, const int32_t *symbols,
                 size_t n_sym, int32_t V, uint8_t *out_buf, size_t out_cap) {
    uint64_t low=0, high=UINT64_MAX; size_t pos=0; int32_t stride=V+1;
    for (size_t t=0; t<n_sym; ++t) {
        const int32_t *cf=cdfs+(size_t)t*stride; int32_t sym=symbols[t];
        u128 range=(u128)high-(u128)low+1;
        uint64_t clo=(uint64_t)(uint32_t)cf[sym], chi=(uint64_t)(uint32_t)cf[sym+1];
        high=low+scale(range,chi)-1; low=low+scale(range,clo);
        while ((low>>56)==(high>>56)) {
            if (pos>=out_cap) return (size_t)-1;
            out_buf[pos++]=(uint8_t)(low>>56); low<<=8; high=(high<<8)|0xFFULL;
        }
    }
    for (int i=0;i<8;++i) { if(pos>=out_cap) return (size_t)-1; out_buf[pos++]=(uint8_t)(low>>56); low<<=8; }
    return pos;
}

int rc_decode(const uint8_t *in_buf, size_t in_len,
              const int32_t *cdfs, size_t n_sym, int32_t V, int32_t *out_syms) {
    uint64_t low=0, high=UINT64_MAX, code=0; size_t pos=0; int32_t stride=V+1;
    for (int i=0;i<8;++i) code=(code<<8)|(pos<in_len?in_buf[pos++]:0);
    for (size_t t=0; t<n_sym; ++t) {
        const int32_t *cf=cdfs+(size_t)t*stride;
        u128 range=(u128)high-(u128)low+1;
        int32_t lo=0, hi=V-1;
        while (lo<hi) { int32_t mid=(lo+hi+1)/2;
            if (low+scale(range,(uint64_t)(uint32_t)cf[mid])<=code) lo=mid; else hi=mid-1; }
        out_syms[t]=lo;
        uint64_t clo=(uint64_t)(uint32_t)cf[lo], chi=(uint64_t)(uint32_t)cf[lo+1];
        high=low+scale(range,chi)-1; low=low+scale(range,clo);
        while ((low>>56)==(high>>56)) {
            low<<=8; high=(high<<8)|0xFFULL; code=(code<<8)|(pos<in_len?in_buf[pos++]:0);
        }
    }
    return 0;
}
"""

_RC_CLIB = None

def _get_rc_clib():
    global _RC_CLIB
    if _RC_CLIB is None:
        import tempfile, subprocess
        # write inlined source to a temp file and compile
        with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
            f.write(_RC_C_SOURCE); src_path = f.name
        so_path = src_path.replace('.c', '.so')
        subprocess.check_call(["cc", "-O2", "-shared", "-fPIC", src_path, "-o", so_path])
        lib = _ct.CDLL(so_path)
        lib.rc_encode.restype  = _ct.c_size_t
        lib.rc_encode.argtypes = [_ct.POINTER(_ct.c_int32), _ct.POINTER(_ct.c_int32),
                                   _ct.c_size_t, _ct.c_int32,
                                   _ct.POINTER(_ct.c_uint8), _ct.c_size_t]
        lib.rc_decode.restype  = _ct.c_int
        lib.rc_decode.argtypes = [_ct.POINTER(_ct.c_uint8), _ct.c_size_t,
                                   _ct.POINTER(_ct.c_int32), _ct.c_size_t, _ct.c_int32,
                                   _ct.POINTER(_ct.c_int32)]
        _RC_CLIB = lib
    return _RC_CLIB

def _rc_encode_c_native(symbols_np, cumfreqs_np):
    lib=_get_rc_clib(); T,Vp1=cumfreqs_np.shape; V=Vp1-1
    out_cap=T*3+16; out_buf=np.zeros(out_cap,np.uint8)
    cdfs_c=cumfreqs_np.astype(np.int32).ravel(); syms_c=symbols_np.astype(np.int32)
    n=lib.rc_encode(cdfs_c.ctypes.data_as(_ct.POINTER(_ct.c_int32)),
        syms_c.ctypes.data_as(_ct.POINTER(_ct.c_int32)),
        _ct.c_size_t(T),_ct.c_int32(V),
        out_buf.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),_ct.c_size_t(out_cap))
    if n==_ct.c_size_t(-1).value: raise RuntimeError("rc_encode (C): overflow")
    return out_buf[:n]

def _rc_decode_c_native(in_bytes, cumfreqs_np):
    lib=_get_rc_clib(); T,Vp1=cumfreqs_np.shape; V=Vp1-1
    out_syms=np.zeros(T,np.int32); in_buf=in_bytes.astype(np.uint8)
    cdfs_c=cumfreqs_np.astype(np.int32).ravel()
    ret=lib.rc_decode(in_buf.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),_ct.c_size_t(len(in_buf)),
        cdfs_c.ctypes.data_as(_ct.POINTER(_ct.c_int32)),_ct.c_size_t(T),_ct.c_int32(V),
        out_syms.ctypes.data_as(_ct.POINTER(_ct.c_int32)))
    if ret!=0: raise RuntimeError(f"rc_decode (C): error {ret}")
    return out_syms

# ── numpy 32-bit and 16-bit coders ───────────────────────────────────────────

_RC_SCHEMES = {
    'canonical-uint64': (64,16,np.uint64,None),
    'canonical-uint32': (32,16,np.uint32,np.uint64),
    'canonical-uint16': (16, 8,np.uint16,np.uint32),
}

def _rc_requantize(cumfreqs_np, M_out):
    T,Vp1=cumfreqs_np.shape; V=Vp1-1; out=np.zeros((T,Vp1),np.int32)
    for i in range(T):
        freqs=np.maximum(np.diff(cumfreqs_np[i]).astype(np.float64),0.0); freqs/=freqs.sum()
        f=freqs*M_out; ff=np.floor(f).astype(np.int32); rem=M_out-int(ff.sum())
        ff[np.argsort(f-ff)[::-1][:rem]]+=1
        for j in range(V):
            if ff[j]==0: ff[j]=1; ff[int(ff.argmax())]-=1
        out[i,1:]=np.cumsum(ff); out[i,-1]=M_out
    return out

def _rc_encode_np(symbols_np, cumfreqs_np, BITS, RC_PREC, np_state_t, np_scale_t):
    M=1<<RC_PREC
    if int(cumfreqs_np[0,-1])!=M: cumfreqs_np=_rc_requantize(cumfreqs_np,M)
    T,Vp1=cumfreqs_np.shape; MASK_PY=(1<<BITS)-1; TOP_SH=BITS-8; tail=BITS//8
    def scale(rng,f):
        if np_scale_t is None:
            r=np.uint64(rng);g=np.uint64(f)
            return np.uint64((r>>np.uint64(32))*g<<np.uint64(16)+(r&np.uint64(0xFFFFFFFF))*g>>np.uint64(16))
        return np_state_t(int(np_scale_t(rng)*np_scale_t(f))>>RC_PREC)
    low=np_state_t(0); high=np_state_t(MASK_PY-1); buf=bytearray()
    for t in range(T):
        sym=int(symbols_np[t]); cf=cumfreqs_np[t]; rng=int(high)-int(low)+1
        new_hi=np_state_t(int(low)+int(scale(rng,int(cf[sym+1])))-1)
        new_lo=np_state_t(int(low)+int(scale(rng,int(cf[sym]))))
        if int(new_hi)<int(new_lo): new_hi=new_lo
        high,low=new_hi,new_lo
        for _ in range(tail):
            lt=(int(low)>>TOP_SH)&0xFF; ht=(int(high)>>TOP_SH)&0xFF
            if lt==ht:
                buf.append(lt); low=np_state_t((int(low)<<8)&MASK_PY)
                high=np_state_t(((int(high)<<8)|0xFF)&MASK_PY)
            else: break
    for _ in range(tail): buf.append((int(low)>>TOP_SH)&0xFF); low=np_state_t((int(low)<<8)&MASK_PY)
    return np.array(buf,dtype=np.uint8)

def _rc_decode_np(in_bytes, cumfreqs_np, BITS, RC_PREC, np_state_t, np_scale_t):
    M=1<<RC_PREC
    if int(cumfreqs_np[0,-1])!=M: cumfreqs_np=_rc_requantize(cumfreqs_np,M)
    T,Vp1=cumfreqs_np.shape; V=Vp1-1; MASK_PY=(1<<BITS)-1; TOP_SH=BITS-8; tail=BITS//8
    data=[int(b) for b in in_bytes]+[0]*(tail*4)
    def scale(rng,f):
        if np_scale_t is None:
            r=np.uint64(rng);g=np.uint64(f)
            return np.uint64((r>>np.uint64(32))*g<<np.uint64(16)+(r&np.uint64(0xFFFFFFFF))*g>>np.uint64(16))
        return np_state_t(int(np_scale_t(rng)*np_scale_t(f))>>RC_PREC)
    code=0
    for i in range(tail): code=((code<<8)|data[i])&MASK_PY
    code=np_state_t(code); low=np_state_t(0); high=np_state_t(MASK_PY-1); pos=tail; out=np.empty(T,np.int32)
    for t in range(T):
        cf=cumfreqs_np[t]; rng=int(high)-int(low)+1; lo_s,hi_s=0,V-1
        while lo_s<hi_s:
            mid=(lo_s+hi_s+1)//2
            if int(low)+int(scale(rng,int(cf[mid])))<=int(code): lo_s=mid
            else: hi_s=mid-1
        sym=lo_s
        new_hi=np_state_t(int(low)+int(scale(rng,int(cf[sym+1])))-1)
        new_lo=np_state_t(int(low)+int(scale(rng,int(cf[sym]))))
        if int(new_hi)<int(new_lo): new_hi=new_lo
        high,low=new_hi,new_lo; out[t]=sym
        for _ in range(tail):
            lt=(int(low)>>TOP_SH)&0xFF; ht=(int(high)>>TOP_SH)&0xFF
            if lt==ht:
                low=np_state_t((int(low)<<8)&MASK_PY); high=np_state_t(((int(high)<<8)|0xFF)&MASK_PY)
                code=np_state_t(((int(code)<<8)|data[pos])&MASK_PY); pos+=1
            else: break
    return out

# ── dispatch ──────────────────────────────────────────────────────────────────

def rc_encode(symbols_np, cumfreqs_np, scheme='canonical-uint64'):
    if scheme=='canonical-c': return _rc_encode_c_native(symbols_np,cumfreqs_np)
    if scheme=='canonical-uint64':
        T,Vp1=cumfreqs_np.shape; V=Vp1-1
        buf,n=rc_encode_jax(jnp.array(symbols_np,jnp.int32),jnp.array(cumfreqs_np,jnp.int32),T,V)
        return np.array(buf[:int(n)],dtype=np.uint8)
    BITS,RC_PREC_s,np_state_t,np_scale_t=_RC_SCHEMES[scheme]
    return _rc_encode_np(symbols_np,cumfreqs_np,BITS,RC_PREC_s,np_state_t,np_scale_t)

def rc_decode(in_bytes, cumfreqs_np, scheme='canonical-uint64'):
    if scheme=='canonical-c': return _rc_decode_c_native(in_bytes,cumfreqs_np)
    if scheme=='canonical-uint64':
        T,Vp1=cumfreqs_np.shape; V=Vp1-1
        max_buf=T*_RC_MAX_BUF_PER_SYM+8; padded=np.zeros(max_buf,dtype=np.uint8)
        n=min(len(in_bytes),max_buf); padded[:n]=in_bytes.astype(np.uint8)[:n]
        syms=rc_decode_jax(jnp.array(padded,jnp.uint8),jnp.array(cumfreqs_np,jnp.int32),T,V)
        return np.array(syms,dtype=np.int32)
    BITS,RC_PREC_s,np_state_t,np_scale_t=_RC_SCHEMES[scheme]
    return _rc_decode_np(in_bytes,cumfreqs_np,BITS,RC_PREC_s,np_state_t,np_scale_t)

# ── decompress step decoder (matches canonical-uint64) ───────────────────────

_FULL64 = 0xFFFFFFFFFFFFFFFF
_M_DEC  = 1 << RC_PREC

def _decompress_step_init(rc_raw, scheme='canonical-uint64'):
    """Initialise AR step-decoder state for the given scheme."""
    if scheme == 'canonical-c':
        # rc_codec.c: high=UINT64_MAX (range=2^64, handled in Python unbounded ints)
        rc_low=0; rc_high=_FULL64; rc_code=0; rc_pos=0
        for _ in range(8):
            byte=int(rc_raw[rc_pos]) if rc_pos<len(rc_raw) else 0
            rc_code=(rc_code<<8)|byte; rc_pos+=1
        return rc_low, rc_high, rc_code, rc_pos
    if scheme == 'canonical-uint64':
        # JAX coder: high=UINT64_MAX-1 (range=UINT64_MAX fits in uint64)
        rc_low=0; rc_high=_FULL64-1; rc_code=0; rc_pos=0
        for _ in range(8):
            byte=int(rc_raw[rc_pos]) if rc_pos<len(rc_raw) else 0
            rc_code=(rc_code<<8)|byte; rc_pos+=1
        return rc_low, rc_high, rc_code, rc_pos
    # numpy schemes: (low, high, code, pos) with BITS-wide state
    BITS, RC_PREC_s, np_state_t, _ = _RC_SCHEMES[scheme]
    MASK_PY=(1<<BITS)-1; tail=BITS//8
    code=0; rc_pos=0
    for _ in range(tail):
        byte=int(rc_raw[rc_pos]) if rc_pos<len(rc_raw) else 0
        code=((code<<8)|byte)&MASK_PY; rc_pos+=1
    return 0, MASK_PY-1, code, rc_pos   # low, high, code, pos

def _decompress_step(rc_low, rc_high, rc_code, rc_pos, rc_raw, cf, V, scheme='canonical-uint64'):
    """One AR decode step.  State types must match the scheme used to encode."""
    if scheme in ('canonical-uint64', 'canonical-c'):
        # Both use Python unbounded ints; boundary-exact search = same as rc_codec.c
        rc_range=rc_high-rc_low+1; lo_s,hi_s=0,V-1
        while lo_s<hi_s:
            mid=(lo_s+hi_s+1)//2
            if rc_low+rc_range*int(cf[mid])//_M_DEC<=rc_code: lo_s=mid
            else: hi_s=mid-1
        sym=lo_s; cum_lo=int(cf[sym]); cum_hi=int(cf[sym+1])
        rc_high=rc_low+rc_range*cum_hi//_M_DEC-1; rc_low=rc_low+rc_range*cum_lo//_M_DEC
        while (rc_low>>56)==(rc_high>>56):
            rc_low=rc_low<<8; rc_high=(rc_high<<8)|0xFF
            byte=int(rc_raw[rc_pos]) if rc_pos<len(rc_raw) else 0
            rc_code=(rc_code<<8)|byte; rc_pos+=1
        return sym, rc_low, rc_high, rc_code, rc_pos
    # numpy schemes
    BITS, RC_PREC_s, np_state_t, np_scale_t = _RC_SCHEMES[scheme]
    MASK_PY=(1<<BITS)-1; TOP_SH=BITS-8; tail=BITS//8; M_s=1<<RC_PREC_s
    def scale(rng, f):
        if np_scale_t is None:
            r=np.uint64(rng);g=np.uint64(f)
            return int(np.uint64((r>>np.uint64(32))*g<<np.uint64(16)+(r&np.uint64(0xFFFFFFFF))*g>>np.uint64(16)))
        return int(np_state_t(int(np_scale_t(rng)*np_scale_t(f))>>RC_PREC_s))
    # requantize cf if needed
    if int(cf[-1]) != M_s:
        freqs=np.maximum(np.diff(cf).astype(np.float64),0.0); freqs/=freqs.sum()
        f=freqs*M_s; ff=np.floor(f).astype(np.int32); rem=M_s-int(ff.sum())
        ff[np.argsort(f-ff)[::-1][:rem]]+=1
        for j in range(V):
            if ff[j]==0: ff[j]=1; ff[int(ff.argmax())]-=1
        cf2=np.zeros(V+1,np.int32); cf2[1:]=np.cumsum(ff); cf2[-1]=M_s
    else:
        cf2=cf
    rng=int(rc_high)-int(rc_low)+1; lo_s,hi_s=0,V-1
    while lo_s<hi_s:
        mid=(lo_s+hi_s+1)//2
        if int(rc_low)+scale(rng,int(cf2[mid]))<=int(rc_code): lo_s=mid
        else: hi_s=mid-1
    sym=lo_s
    new_hi=np_state_t(int(rc_low)+scale(rng,int(cf2[sym+1]))-1)
    new_lo=np_state_t(int(rc_low)+scale(rng,int(cf2[sym])))
    if int(new_hi)<int(new_lo): new_hi=new_lo
    rc_high,rc_low=new_hi,new_lo
    for _ in range(tail):
        lt=(int(rc_low)>>TOP_SH)&0xFF; ht=(int(rc_high)>>TOP_SH)&0xFF
        if lt==ht:
            rc_low=np_state_t((int(rc_low)<<8)&MASK_PY)
            rc_high=np_state_t(((int(rc_high)<<8)|0xFF)&MASK_PY)
            byte=int(rc_raw[rc_pos]) if rc_pos<len(rc_raw) else 0
            rc_code=np_state_t(((int(rc_code)<<8)|byte)&MASK_PY); rc_pos+=1
        else: break
    return sym, rc_low, rc_high, rc_code, rc_pos


# ============================================================================
# Compression eval (used during training)
# ============================================================================

def collect_probs(base_xlstm, params, config, all_inputs, all_targets):
    stride_map=_parse_stride_map(config.stride_map,config.num_layers)
    num_chunks=all_inputs.shape[0]; chunk_size=all_inputs.shape[1]
    oh,ob=config.output_heads,config.output_bits
    states=init_step_states(config,batch_size=1)
    max_tok=(config.max_eval_tokens if config.max_eval_tokens>0 else num_chunks*chunk_size)
    logits_list=[]; tgts_list=[]; processed=0
    pbar=tqdm(total=max_tok,desc="collect probs",unit="tok",leave=False,file=sys.stderr)
    for ci in range(num_chunks):
        if processed>=max_tok: break
        this_size=min(chunk_size,max_tok-processed)
        for pos in range(this_size):
            tok=jnp.array([int(all_inputs[ci,pos])],jnp.int32)
            lf,states=_collect_chunk_jit(base_xlstm,params,tok,states,1,config.num_heads,config.block_map,stride_map,oh,ob)
            logits_list.append(np.array(lf[0],np.float32)); tgts_list.append(int(all_targets[ci,pos,0]))
            processed+=1; pbar.update(1)
            if processed>=max_tok: break
    pbar.close()
    return np.stack(logits_list), np.array(tgts_list,np.int32)

def eval_compression(base_xlstm, params, config, all_inputs, all_targets, n_raw_bytes,
                     label="", scheme='canonical-uint64'):
    t0=time.perf_counter()
    param_bytes=sum(np.prod(p.shape)*np.dtype(DTYPE).itemsize for p in jax.tree_util.tree_leaves(params))
    logits_np,tgts_np=collect_probs(base_xlstm,params,config,all_inputs,all_targets)
    T=len(tgts_np); V=1<<config.output_bits
    valid=tgts_np>=0; vlog=logits_np[valid]; vtgt=tgts_np[valid]; T_v=int(valid.sum())
    lp=jax.nn.log_softmax(jnp.array(vlog),axis=-1)
    tgt_lp=np.array(lp)[np.arange(T_v),vtgt]
    ce_bits=float(-np.sum(tgt_lp)/math.log(2)); ce_bpb=ce_bits/n_raw_bytes
    preds=np.argmax(vlog,axis=-1); n_wrong=int(np.sum(preds!=vtgt)); acc=(T_v-n_wrong)/max(T_v,1)
    _qcdf_jit=jax.jit(_quantize_cdf); _cdf_ok_jit=jax.jit(_cdf_ok)
    cumfreqs_np=np.empty((T_v,V+1),dtype=np.int32)
    pbar_cdf=tqdm(total=T_v,desc="quantize CDFs",unit="tok",leave=False,file=sys.stderr)
    for i in range(T_v):
        cf=_qcdf_jit(jnp.array(vlog[i])); cumfreqs_np[i]=np.array(cf,np.int32); pbar_cdf.update(1)
    pbar_cdf.close()
    print(f"  encode ({scheme})...",end=" ",flush=True)
    rc_np=rc_encode(vtgt,cumfreqs_np,scheme=scheme); rc_bytes=len(rc_np); print(f"{rc_bytes}B")
    print(f"  verify...",end=" ",flush=True)
    decoded=rc_decode(rc_np,cumfreqs_np,scheme=scheme)
    decode_ok=bool(np.all(decoded==vtgt)); print("OK" if decode_ok else "FAIL")
    elapsed=time.perf_counter()-t0; tot_rc=param_bytes+rc_bytes
    def _r(total): return n_raw_bytes/total if total>0 else float('inf')
    def _bpb(bits): return bits/n_raw_bytes
    hdr=f"[compression{' '+label if label else ''}]"; rt='OK' if decode_ok else 'FAIL'
    eval_note=(f"eval all {n_raw_bytes} bytes" if T_v>=n_raw_bytes-1 else f"eval first {T_v} of {n_raw_bytes} bytes")
    print(f"\n{hdr}  {elapsed:.1f}s  param={param_bytes}B  {eval_note}"
          f"  argmax: {T_v-n_wrong}/{T_v} correct ({acc:.1%})  {rt}")
    print(f"  CE={ce_bpb:.4f}bpb  rc={_bpb(rc_bytes*8):.4f}bpb({rc_bytes}B)  p+rc={_r(tot_rc):.3f}x({tot_rc}B)")
    return dict(ce_bits=ce_bits,ce_bpb=ce_bpb,rc_bytes=rc_bytes,rc_bpb=_bpb(rc_bytes*8),
        rc_np=rc_np,seed_token=int(all_inputs[0][0]),T_valid=T_v,n_raw_bytes=n_raw_bytes,
        param_bytes=param_bytes,total_rc=tot_rc,ratio_rc=_r(tot_rc),n_wrong=n_wrong,argmax_acc=acc,decode_ok=decode_ok)


# ============================================================================
# Checkpoint / bundle I/O
# ============================================================================

class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module=='__main__': return globals().get(name) or getattr(sys.modules[__name__],name,None) or super().find_class(module,name)
        return super().find_class(module,name)

def _save_bundle(ckpt_dir, cstats, params, config, scheme='canonical-uint64'):
    if cstats["T_valid"]<cstats["n_raw_bytes"]-1:
        raise ValueError(f"save_bundle: T_valid={cstats['T_valid']} < n_raw-1")
    os.makedirs(ckpt_dir,exist_ok=True)
    with open(os.path.join(ckpt_dir,"rc_stream.bin"),"wb") as f: f.write(bytes(cstats["rc_np"].tolist()))
    with open(os.path.join(ckpt_dir,"params.pkl"),"wb") as f: pickle.dump(jax.tree_util.tree_map(np.array,params),f)
    with open(os.path.join(ckpt_dir,"config.json"),"w") as f: json.dump(config._asdict(),f,indent=2)
    param_disk=os.path.getsize(os.path.join(ckpt_dir,"params.pkl"))
    sha256=hashlib.sha256(cstats["rc_np"]).hexdigest()
    meta=dict(n_raw_bytes=int(cstats["n_raw_bytes"]),T_valid=int(cstats["T_valid"]),
        seed_token=int(cstats["seed_token"]),rc_bytes=int(cstats["rc_bytes"]),
        rc_stream_sha256=sha256,rc_scheme=scheme,param_bytes=int(cstats["param_bytes"]),
        param_disk_bytes=int(param_disk),input_bits=int(config.input_bits),
        output_bits=int(config.output_bits),output_heads=int(config.output_heads))
    with open(os.path.join(ckpt_dir,"meta.json"),"w") as f: json.dump(meta,f,indent=2)

def _save_params(ckpt_dir, params, config):
    os.makedirs(ckpt_dir,exist_ok=True)
    with open(os.path.join(ckpt_dir,"params.pkl"),"wb") as f: pickle.dump(jax.tree_util.tree_map(np.array,params),f)
    with open(os.path.join(ckpt_dir,"config.json"),"w") as f: json.dump(config._asdict(),f,indent=2)

def _load_bundle(ckpt_dir):
    with open(os.path.join(ckpt_dir,"meta.json")) as f: meta=json.load(f)
    with open(os.path.join(ckpt_dir,"config.json")) as f: cfg_dict=json.load(f)
    with open(os.path.join(ckpt_dir,"params.pkl"),"rb") as f: params_np=_Unpickler(f).load()
    with open(os.path.join(ckpt_dir,"rc_stream.bin"),"rb") as f: rc_raw=np.frombuffer(f.read(),dtype=np.uint8).copy()
    if "rc_stream_sha256" in meta:
        actual=hashlib.sha256(rc_raw.tobytes()).hexdigest()
        if actual!=meta["rc_stream_sha256"]: raise RuntimeError(f"rc_stream.bin corrupt: sha256 mismatch")
    valid_keys=set(Config._fields)
    config=Config(**{k:v for k,v in cfg_dict.items() if k in valid_keys})
    return meta, config, params_np, rc_raw

def cosine_lr(step, total_steps, lr_max, lr_min=1e-4, warmup=100):
    warmup_lr=lr_max*jnp.minimum(step/warmup,1.0)
    t=jnp.maximum(step-warmup,0)/jnp.maximum(total_steps-warmup,1)
    cos_lr=lr_min+0.5*(lr_max-lr_min)*(1.0+jnp.cos(math.pi*t))
    return jnp.where(step<warmup,warmup_lr,cos_lr)


# ============================================================================
# Mode: train
# ============================================================================

def _parse_train_args(argv):
    _types={f:type(v) for f,v in Config()._asdict().items()}
    def _cast(field,s):
        t=_types[field]
        if t is bool: return s.lower() in ("1","true","yes")
        return t(s)
    cfg=Config()._asdict()
    if "--preset" in argv:
        i=argv.index("--preset"); name=argv[i+1]
        if name in _PRESETS: cfg.update(_PRESETS[name])
    i=0
    while i<len(argv):
        if argv[i].startswith("--"):
            field=argv[i][2:]
            if field in cfg and i+1<len(argv): cfg[field]=_cast(field,argv[i+1]); i+=2; continue
        i+=1
    cfg['num_layers']=len(cfg['block_map']); cfg['vocab_size']=2**cfg['input_bits']
    sm=cfg['stride_map']
    if len(sm)<cfg['num_layers']: sm=sm+"1"*(cfg['num_layers']-len(sm))
    cfg['stride_map']=sm[:cfg['num_layers']]
    config=Config(**cfg); lib_defaults=Config()._asdict()
    diff={k:{"default":lib_defaults[k],"value":cfg[k]} for k in cfg if cfg[k]!=lib_defaults[k]}
    return config, diff

def cmd_train(argv, log_dir, scheme):
    config, config_diff = _parse_train_args(argv)
    global PAD_TOKEN, DTYPE, POWER_P
    PAD_TOKEN=config.pad_token; DTYPE=jnp.dtype(config.dtype); POWER_P=config.power_p
    os.makedirs(log_dir,exist_ok=True)
    log_path=os.path.join(log_dir,"train.log"); log_file=open(log_path,"w",buffering=1)
    sys.stdout=_Tee(sys.__stdout__,log_file)
    with open(os.path.join(log_dir,"config.json"),"w") as f: json.dump(config._asdict(),f,indent=2)
    with open(os.path.join(log_dir,"config_diff.json"),"w") as f: json.dump(config_diff,f,indent=2)
    if config_diff:
        print("Config overrides:"); [print(f"  {k}: {v['default']} → {v['value']}") for k,v in config_diff.items()]
    else: print("Config: all defaults")
    stride_map=_parse_stride_map(config.stride_map,config.num_layers); print(f"stride_map: {stride_map}")
    raw_bytes=load_dataset(config.dataset); n_total=len(raw_bytes)
    print(f"\nConfig validation:"); warns=validate_config(config,n_total)
    if not warns: print("  OK")
    tokens=bytes_to_tokens(raw_bytes,config.input_bits); tok_per_byte=8//config.input_bits
    print(f"Dataset: {config.dataset}  ({n_total} bytes)  {len(tokens)} tokens")
    key=jr.key(config.seed); key,k_xlstm,k_rc=jr.split(key,3)
    print("Initialising frozen xLSTM..."); base_xlstm=init_xlstm_params(k_xlstm,config)
    frozen_n=sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(base_xlstm))
    print("Initialising HiRA adapters..."); params=init_randcompress_params(k_rc,config)
    trainable=sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    total=frozen_n+trainable; dtype_bytes=np.dtype(DTYPE).itemsize; param_bytes=trainable*dtype_bytes
    print(f"  frozen={frozen_n/1e6:.3f}M  trainable={trainable/1e6:.3f}M  param_bytes={param_bytes/1e3:.1f}kB")
    opt_state=sinkgd_init(params); lr=config.learning_rate
    all_inputs,all_targets=make_chunks_and_targets(raw_bytes,config); num_chunks=all_inputs.shape[0]
    K=config.grad_accum_steps
    print(f"Chunks: {num_chunks} × {all_inputs.shape[1]} tokens")
    print(f"max_iters={config.max_iters}  check_every={config.check_every}  lr={lr}  scheme={scheme}\n")
    print("Step 0 — baseline:")
    eval_compression(base_xlstm,params,config,all_inputs,all_targets,n_total,label="step=0",scheme=scheme)
    states=init_step_states(config,batch_size=1); chunk_idx=0; rng_key=jr.key(config.seed+2)
    t_start=time.perf_counter()
    pbar=tqdm(range(1,config.max_iters+1),desc="train",unit="it",file=sys.stderr,dynamic_ncols=True)
    for it in pbar:
        inputs_K=jnp.stack([all_inputs[(chunk_idx+k)%num_chunks][None,:] for k in range(K)])
        targets_K=jnp.stack([all_targets[(chunk_idx+k)%num_chunks][None,:] for k in range(K)])
        chunk_idx=(chunk_idx+K)%num_chunks
        boundary_states=jax.lax.stop_gradient(states); rng_key,subkey=jr.split(rng_key)
        loss,grads,states=grad_accum_persistent(base_xlstm,params,inputs_K,targets_K,boundary_states,subkey,config,K)
        params,opt_state=sinkgd_update(opt_state,grads,params,lr=lr,weight_decay=config.weight_decay,max_norm=config.grad_clip_norm,L=config.sinkgd_l)
        bpb=float(loss)/math.log(2)*(8//config.input_bits)
        pbar.set_postfix(loss=f"{float(loss):.5f}",bpb=f"{bpb:.4f}")
        if it%config.check_every==0 or it==config.max_iters:
            elapsed=time.perf_counter()-t_start; print(f"\n[it={it:6d}] train_bpb={bpb:.4f}  {elapsed:.0f}s")
            cstats=eval_compression(base_xlstm,params,config,all_inputs,all_targets,n_total,label=f"it={it}",scheme=scheme)
            ckpt_it=os.path.join(log_dir,f"ckpt_it{it:06d}")
            if config.save_ckpt:
                full_config=config._replace(max_eval_tokens=0)
                full_cstats=eval_compression(base_xlstm,params,full_config,all_inputs,all_targets,n_total,label=f"it={it} full",scheme=scheme)
                _save_bundle(ckpt_it,full_cstats,params,config,scheme=scheme)
            else: _save_params(ckpt_it,params,config)
            if config.target_bpb>0 and cstats["rc_bpb"]<config.target_bpb:
                print(f"[STOP] target_bpb reached"); pbar.close(); break
    else: pbar.close()
    elapsed_total=time.perf_counter()-t_start; print(f"\n[STOP] max_iters  elapsed={elapsed_total:.1f}s")
    print("\nFinal compression eval (full file)...")
    full_config=config._replace(max_eval_tokens=0)
    cstats=eval_compression(base_xlstm,params,full_config,all_inputs,all_targets,n_total,label="final",scheme=scheme)
    ckpt_dir=os.path.join(log_dir,"ckpt_last"); _save_bundle(ckpt_dir,cstats,params,config,scheme=scheme)
    with open(os.path.join(ckpt_dir,"compression.json"),"w") as f:
        json.dump({k:(v if isinstance(v,(int,float,bool,str)) else str(v)) for k,v in cstats.items()},f,indent=2)
    print(f"\nLog: {log_path}"); log_file.close(); sys.stdout=sys.__stdout__


# ============================================================================
# Mode: compress
# ============================================================================

def cmd_compress(params_dir, input_path, output_dir, scheme):
    t_total=time.perf_counter()
    print(f"Params: {params_dir}\nInput:  {input_path}\nOutput: {output_dir}\nScheme: {scheme}")
    with open(os.path.join(params_dir,"config.json")) as f: cfg_dict=json.load(f)
    with open(os.path.join(params_dir,"params.pkl"),"rb") as f: params_np=_Unpickler(f).load()
    config=Config(**{k:v for k,v in cfg_dict.items() if k in set(Config._fields)})
    global DTYPE, POWER_P; DTYPE=jnp.dtype(config.dtype); POWER_P=config.power_p
    key=jr.key(config.seed); _,k_xlstm,_=jr.split(key,3)
    base_xlstm=init_xlstm_params(k_xlstm,config); params=jax.tree_util.tree_map(jnp.array,params_np)
    raw_bytes=load_dataset(input_path); n_raw=len(raw_bytes)
    print(f"Input:  {n_raw} bytes")
    all_inputs,all_targets=make_chunks_and_targets(raw_bytes,config)
    stride_map=_parse_stride_map(config.stride_map,config.num_layers)
    oh,ob=config.output_heads,config.output_bits; V=1<<ob
    t0=time.perf_counter(); states=init_step_states(config,batch_size=1)
    logits_list=[]; tgts_list=[]
    num_chunks=all_inputs.shape[0]; chunk_size=all_inputs.shape[1]
    pbar=tqdm(total=num_chunks*chunk_size,desc="collect logits",unit="tok",file=sys.stderr)
    for ci in range(num_chunks):
        for pos in range(chunk_size):
            tok=jnp.array([int(all_inputs[ci,pos])],jnp.int32)
            lf,states=_collect_chunk_jit(base_xlstm,params,tok,states,1,config.num_heads,config.block_map,stride_map,oh,ob)
            logits_list.append(np.array(lf[0],np.float32)); tgts_list.append(int(all_targets[ci,pos,0])); pbar.update(1)
    pbar.close(); t_logits=time.perf_counter()-t0
    logits_np=np.stack(logits_list); tgts_np=np.array(tgts_list,np.int32)
    valid=tgts_np>=0; vlog,vtgt=logits_np[valid],tgts_np[valid]; T_v=int(valid.sum())
    lp=jax.nn.log_softmax(jnp.array(vlog),axis=-1); tgt_lp=np.array(lp)[np.arange(T_v),vtgt]
    ce_bits=float(-np.sum(tgt_lp)/np.log(2)); ce_bpb=ce_bits/n_raw
    n_wrong=int(np.sum(np.argmax(vlog,axis=-1)!=vtgt)); acc=(T_v-n_wrong)/max(T_v,1)
    _qcdf_jit=jax.jit(_quantize_cdf); cumfreqs_np=np.empty((T_v,V+1),dtype=np.int32)
    t0=time.perf_counter()
    pbar_cdf=tqdm(total=T_v,desc="quantize CDFs",unit="tok",leave=False,file=sys.stderr)
    for i in range(T_v):
        cumfreqs_np[i]=np.array(_qcdf_jit(jnp.array(vlog[i])),np.int32); pbar_cdf.update(1)
    pbar_cdf.close(); t_cdf=time.perf_counter()-t0
    t0=time.perf_counter()
    print(f"  encode ({scheme})...",end=" ",flush=True)
    rc_np=rc_encode(vtgt,cumfreqs_np,scheme=scheme); rc_bytes=len(rc_np)
    t_enc=time.perf_counter()-t0; print(f"{rc_bytes}B  {t_enc:.2f}s  ({rc_bytes/t_enc/1e3:.1f} kB/s)")
    t0=time.perf_counter(); print(f"  verify...",end=" ",flush=True)
    decoded=rc_decode(rc_np,cumfreqs_np,scheme=scheme); decode_ok=bool(np.all(decoded==vtgt))
    t_dec=time.perf_counter()-t0; print(f"{'OK' if decode_ok else 'FAIL'}  {t_dec:.2f}s")
    if not decode_ok: raise RuntimeError("RC round-trip failed")
    os.makedirs(output_dir,exist_ok=True)
    with open(os.path.join(output_dir,"rc_stream.bin"),"wb") as f: f.write(bytes(rc_np.tolist()))
    for fname in ("params.pkl","config.json"):
        src=os.path.join(params_dir,fname); dst=os.path.join(output_dir,fname)
        if os.path.abspath(src)!=os.path.abspath(dst): shutil.copy2(src,dst)
    seed_token=int(all_inputs[0][0]); param_bytes=sum(np.prod(p.shape)*np.dtype(DTYPE).itemsize for p in jax.tree_util.tree_leaves(params))
    meta=dict(n_raw_bytes=int(n_raw),T_valid=int(T_v),seed_token=int(seed_token),rc_bytes=int(rc_bytes),
        rc_stream_sha256=hashlib.sha256(rc_np.tobytes()).hexdigest(),rc_scheme=scheme,
        param_bytes=int(param_bytes),param_disk_bytes=os.path.getsize(os.path.join(output_dir,"params.pkl")),
        input_bits=int(config.input_bits),output_bits=int(config.output_bits),output_heads=int(config.output_heads))
    with open(os.path.join(output_dir,"meta.json"),"w") as f: json.dump(meta,f,indent=2)
    t_wall=time.perf_counter()-t_total; tot_bytes=param_bytes+rc_bytes; ratio=n_raw/tot_bytes if tot_bytes>0 else float('inf')
    print(f"\n[compress]  {t_wall:.1f}s  (logits={t_logits:.1f}s  cdfs={t_cdf:.1f}s  enc={t_enc:.2f}s  dec={t_dec:.2f}s)")
    print(f"  argmax: {T_v-n_wrong}/{T_v} ({acc:.1%})  CE={ce_bpb:.4f}bpb  rc={rc_bytes*8/n_raw:.4f}bpb({rc_bytes}B)  p+rc={ratio:.3f}x")
    print(f"Bundle: {output_dir}")


# ============================================================================
# Mode: decompress
# ============================================================================

def cmd_decompress(bundle_dir, output_path, verify_path=None):
    meta,config,params_np,rc_raw=_load_bundle(bundle_dir)
    n_raw_bytes=meta["n_raw_bytes"]; T_valid=meta["T_valid"]
    seed_token=meta["seed_token"]; rc_bytes=meta["rc_bytes"]
    scheme=meta.get("rc_scheme","canonical-uint64")
    print(f"Bundle: {bundle_dir}\n  n_raw={n_raw_bytes}  T_valid={T_valid}  seed={seed_token}  rc_bytes={rc_bytes}  scheme={scheme}")
    global DTYPE, POWER_P; DTYPE=jnp.dtype(config.dtype); POWER_P=config.power_p
    key=jr.key(config.seed); _,k_xlstm,_=jr.split(key,3)
    print("Reconstructing base model..."); base_xlstm=init_xlstm_params(k_xlstm,config)
    print("Restoring params..."); params=jax.tree_util.tree_map(jnp.array,params_np)
    stride_map=_parse_stride_map(config.stride_map,config.num_layers)
    oh,ob=config.output_heads,config.output_bits; V=1<<ob
    _qcdf_jit=jax.jit(_quantize_cdf)
    rc_low,rc_high,rc_code,rc_pos=_decompress_step_init(rc_raw,scheme=scheme)
    model_states=init_step_states(config,batch_size=1); decoded=[seed_token]
    cur_tok=jnp.array([seed_token],jnp.int32)
    print(f"AR decoding (scheme={scheme})...")
    pbar=tqdm(total=T_valid,desc="decode",unit="tok",file=sys.stderr)
    for t in range(T_valid):
        chunk_logits,model_states=_collect_chunk_jit(base_xlstm,params,cur_tok,model_states,
            1,config.num_heads,config.block_map,stride_map,oh,ob)
        cf=np.array(_qcdf_jit(chunk_logits[0]),dtype=np.int32)
        sym,rc_low,rc_high,rc_code,rc_pos=_decompress_step(rc_low,rc_high,rc_code,rc_pos,rc_raw,cf,V,scheme=scheme)
        decoded.append(sym); cur_tok=jnp.array([sym],jnp.int32); pbar.update(1)
    pbar.close()
    raw_out=tokens_to_bytes(np.array(decoded,dtype=np.int32),config.input_bits)[:n_raw_bytes]
    print(f"Decoded {len(raw_out)} bytes  (expected {n_raw_bytes})")
    with open(output_path,"wb") as f: f.write(bytes(raw_out.tolist()))
    print(f"Written: {output_path}")
    if verify_path is not None:
        with open(verify_path,"rb") as f: original=np.frombuffer(f.read(),dtype=np.uint8)
        min_len=min(len(raw_out),len(original)); wrong=int(np.sum(raw_out[:min_len]!=original[:min_len]))
        if wrong==0 and len(raw_out)==len(original): print("Verification: PERFECT MATCH")
        else:
            wrongs=np.where(raw_out[:min_len]!=original[:min_len])[0]
            print(f"Verification: {wrong} bytes wrong")
            if len(wrongs): print(f"  first wrong byte at position {int(wrongs[0])}")
    return raw_out


# ============================================================================
# CLI
# ============================================================================

def main():
    parser=argparse.ArgumentParser(description="linear_rnn_rc_v1: train / compress / decompress")
    parser.add_argument("--mode", required=True, choices=["train","compress","decompress"],
                        help="Operation mode")
    parser.add_argument("--log_dir",  default=None, help="[train] log directory")
    parser.add_argument("--dataset",  default=None, help="[train] dataset file")
    parser.add_argument("--scheme",   default="canonical-uint64",
                        choices=["canonical-uint64","canonical-c","canonical-uint32","canonical-uint16"],
                        help="RC scheme (default: canonical-uint64)")
    parser.add_argument("--params",  default=None, help="[compress] source bundle dir")
    parser.add_argument("--input",   default=None, help="[compress] input file")
    parser.add_argument("--output",  default=None, help="[compress] output bundle dir")
    parser.add_argument("--bundle",  default=None, help="[decompress] bundle dir")
    parser.add_argument("--verify",  default=None, help="[decompress] original file for verification")

    # parse known args; remaining go to _parse_train_args for Config fields
    args, rest = parser.parse_known_args()

    if args.mode == "train":
        log_dir = args.log_dir or os.path.join("log", os.path.splitext(os.path.basename(__file__))[0])
        train_argv = rest
        if args.dataset:
            train_argv = ["--dataset", args.dataset] + train_argv
        cmd_train(train_argv, log_dir, scheme=args.scheme)

    elif args.mode == "compress":
        if not args.params: parser.error("--params required for compress")
        if not args.input:  parser.error("--input required for compress")
        if not args.output: parser.error("--output required for compress")
        cmd_compress(args.params, args.input, args.output, scheme=args.scheme)

    elif args.mode == "decompress":
        if not args.bundle: parser.error("--bundle required for decompress")
        if not args.output: parser.error("--output required for decompress")
        cmd_decompress(args.bundle, args.output, verify_path=args.verify)


if __name__ == "__main__":
    main()
