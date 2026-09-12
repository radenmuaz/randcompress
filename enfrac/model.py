"""enfrac (JAX/Equinox port of overfitter_peft): byte-level FractalAR
(arxiv.org/html/2502.17437v2), same recursive architecture as enfrac_zero/model.py, but every
Linear defaults to a HiRA-adapted frozen-base layer instead of a plain trainable linear -- see
randcompress's (archived, see archive/randcompress/models/hira.py) original HiRA design this is
ported from (W = W0 + W0*(B@A), only B trainable, W0/A regenerated deterministically from
cfg.seed, never saved).

Fork rationale (unchanged from the PyTorch overfitter_peft/ this was ported from): trades the
baseline's "small model, everything trained directly" tradeoff for "arbitrarily large frozen
base, tiny trainable adapter" -- d_model/mlp_mult/byte_embed_dim become nearly free to grow
(O(d) compute, O(r) storage per layer) instead of directly taxing the compressed bundle size.
use_hira=True is the default (set use_hira=False on ModelConfig to fall back to plain linears,
matching enfrac_zero/'s architecture exactly, modulo the JAX/Equinox backend).

IMPORTANT determinism contract: HiraLinear's frozen (W0, A) are drawn from a single KeySeq (a
jax.random.PRNGKey threaded via jax.random.split, mirroring PyTorch's torch.Generator) passed
through construction in a FIXED order (ByteFractalGen -> Trunk -> Block -> Attn/SwiGLU, plus
patch_in/cond_proj/head). Reconstructing a bundle from (config, seed) alone requires this exact
construction order to never change -- reordering module construction between save and load
silently regenerates a DIFFERENT frozen base.

Trainable/frozen split: rather than PyTorch buffers vs. Parameters, every array leaf here is a
plain field on an eqx.Module (equinox pytrees don't distinguish trainable-ness structurally).
`trainable_filter()` below builds the boolean partition mask by leaf *field name* instead: only
leaves named "B" (HiRA's trainable factor), "weight"/"bias" (plain linears, RMSNorm) and
"root_cond" are trainable; everything else (byte_embed's "table", HiRA's "W0"/"A") is frozen.
train.py partitions the model with this mask before every gradient step.

IMPORTANT for compress.py/decompress.py: collect_logits() and generate() MUST stay the same code
path (collect_logits literally calls generate() with a teacher-forcing symbol_fn) -- floating-
point matmuls aren't perfectly associative, so a batched recompute and a step-by-step recompute
of the "same" math can disagree in the last bit, which is enough to desync range coding.
forward()/__call__() (the fast batched path) is fine for training loss -- never for the CDFs
that get range-coded.

LEVEL 0 IS A REAL, FULL-SEQUENCE CAUSAL AR TRANSFORMER, INDEXING CONVENTION, AND THE TWO
patch_in SCHEMES: identical to enfrac_zero/model.py's module docstring (n_levels =
len(patch_len_list); patch_len_list[l] is level l's OWN patch size for every level including 0;
level 0 has no parent -- root_cond is fed directly, no cond_proj[0]; cfg.patch_in_scheme is
"mean_pool" (default, params/compute independent of patch_len) or "linear" (exact but scales
with patch_len -- only tractable for small patches). Read that docstring for the full rationale;
it is not duplicated here except where HiRA changes something.

An EARLIER version of this file carried a single fixed-size state vector (cond_out -> next
chunk's cond_in) between per-chunk calls -- that was an RNN-style bottleneck and was WRONG (see
enfrac_zero/model.py's docstring for why). There is no chunk carry anymore: level 0's sequence
IS the whole file's level-0 timesteps, attended in one causally-masked, fully parallel pass
(remat_time/remat_depth control jax.checkpoint granularity only, exact gradients either way).
This applies identically whether use_hira is True or False -- the recursion structure is shared;
only the Linear type differs.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .codec import quantize_cdf

ROPE_PRESETS = {"llama2": 10000.0, "llama3": 500000.0, "qwen3": 1000000.0}
_MEAN_POOL_TARGET_ELEMENTS = 2_097_152   # rows*P target per _chunked_mean_pool scan step -- see its docstring


class KeySeq:
    """Mutable-state PRNGKey sequence, mirroring torch.Generator's sequential-draw semantics:
    each .next() call advances internal state and returns a fresh subkey."""
    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)

    def next(self) -> jax.Array:
        self.key, sub = jax.random.split(self.key)
        return sub


def rope_cos_sin_for_positions(position_ids: jax.Array, head_dim: int, base: float):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = position_ids.astype(jnp.float32)[:, None] * inv_freq[None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    cos, sin = cos[None, None], sin[None, None]
    return x * cos + rotate_half(x) * sin


def causal_mask(seq_len: int) -> jax.Array:
    pos = jnp.arange(seq_len)
    allow = pos[None, :] <= pos[:, None]
    return allow[None, None]


def sdpa(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array) -> jax.Array:
    d = q.shape[-1]
    scores = jnp.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(d)
    scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhts,bhsd->bhtd", attn, v)


def flash_attn_causal(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """TPU-native flash attention (jax.experimental.pallas.ops.tpu.flash_attention) -- same
    causal scaled-dot-product-attention MATH as sdpa() (mathematically equivalent, not
    bit-identical, same order-of-1e-6 floating-point summation-order difference as every other
    chunked-vs-monolithic comparison in this codebase), but never materializes the full
    [B,H,T,T] score matrix: compute is still O(T^2) (same FLOPs as sdpa -- flash attention is a
    memory optimization, not a compute one), but PEAK MEMORY is O(T) via blockwise
    online-softmax, computed with a Pallas TPU kernel rather than plain XLA ops.

    q/k/v: [B, H, T, head_dim] -- same layout Attn.__call__ already builds for sdpa(). Always
    causal (this codebase's attention is always causal -- see causal_mask()), so no separate mask
    argument like sdpa() takes.

    HARD CONSTRAINT, not a tuning knob: the underlying kernel requires block_k to be a multiple
    of 128 (verified: block_k=32 raises `NotImplementedError`), so T should be >=128 and ideally a
    clean multiple of 128 -- this is why ModelConfig.use_flash_attn_list is meant for levels with
    a BUMPED seq_len (512/1024/2048/4096/8192 -- see docs/enwik9_scaling_calcs.md's shallow-config
    proposals), not the small seq_len=8 levels 1+ commonly use by default. TPU-ONLY: this kernel
    requires real TPU hardware to execute (confirmed: raises `Only interpret mode is supported on
    CPU backend` when traced on a CPU backend) -- there is no local/CPU test path for the actual
    numerics, unlike every other change in this codebase this session, which were all verified
    locally before deployment. Test on a real TPU host before trusting a config that enables this."""
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
    d = q.shape[-1]
    return flash_attention(q, k, v, causal=True, sm_scale=1.0 / math.sqrt(d))


def init_hira_A(d_out: int, d_in: int, r: int, key: jax.Array) -> jax.Array:
    """A: [r, d_in] orthonormal rows (right singular vectors of a random matrix) -- frozen.
    dtype=float32 forced explicitly (NOT ambient-default) -- see HiraLinear's docstring for why:
    this must reproduce the exact same values regardless of whether the caller has
    jax_enable_x64 on, since it's never saved and always regenerated from `key` alone."""
    raw = jax.random.normal(key, (r, d_in), dtype=jnp.float32)
    if r <= d_in:
        _, _, Vt = jnp.linalg.svd(raw, full_matrices=False)
        A = Vt[:r] / math.sqrt(d_in)
    else:
        A = raw / math.sqrt(d_in)
    return A


class HiraLinear(eqx.Module):
    """W = W0 + W0*(B@A) (HiRA) -- W0 [d_out,d_in] and A [r,d_in] frozen (drawn from `keyseq` at
    construction, never saved -- see module docstring's determinism contract), only B [d_out,r]
    is trainable (zero-init, so the layer starts as an identity pass-through of W0).

    W0/A's random init is explicitly forced to dtype=float32, NOT the ambient/default float type
    -- jax.random with a given PRNGKey produces COMPLETELY DIFFERENT values (not just different
    precision) depending on whether jax_enable_x64 is on when it's called, since generating a
    float64 sample consumes different underlying random bits than a float32 one. Since W0/A are
    frozen and NEVER saved (regenerated from `keyseq`/cfg.seed alone every load), constructing
    the model under jax_enable_x64=True without this fix would silently produce a DIFFERENT
    frozen base than what training actually used -- verified empirically this session: byte_acc
    collapsed from ~76% to ~25% when collect_logits_fp64's fp64 model cast was combined with an
    ambient-dtype-dependent frozen init (make_byte_embedding() had the same bug -- see there).
    Any later cast_dtype(jnp.float64) is fine and intended -- that's a value-preserving upcast of
    the CORRECT float32 values, not a re-randomization."""
    W0: jax.Array
    A: jax.Array
    B: jax.Array

    def __init__(self, d_in: int, d_out: int, r: int, keyseq: KeySeq):
        orthogonal = jax.nn.initializers.orthogonal(scale=1.0)
        self.W0 = orthogonal(keyseq.next(), (d_out, d_in), dtype=jnp.float32) / math.sqrt(d_in)
        self.A = init_hira_A(d_out, d_in, r, keyseq.next())
        self.B = jnp.zeros((d_out, r), dtype=jnp.float32)

    def __call__(self, x: jax.Array) -> jax.Array:
        W = self.W0 + self.W0 * (self.B @ self.A)
        return x @ W.T


class PlainLinear(eqx.Module):
    """Plain trainable linear (no bias), used when cfg.use_hira=False."""
    weight: jax.Array

    def __init__(self, d_in: int, d_out: int, keyseq: KeySeq):
        limit = 1.0 / math.sqrt(d_in)
        self.weight = jax.random.uniform(keyseq.next(), (d_out, d_in), minval=-limit, maxval=limit)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.weight.T


def make_linear(d_in: int, d_out: int, cfg: "ModelConfig", keyseq: KeySeq):
    if cfg.use_hira:
        return HiraLinear(d_in, d_out, cfg.hira_r, keyseq)
    return PlainLinear(d_in, d_out, keyseq)


class PatchInLinear(eqx.Module):
    """"linear" patch_in scheme -- see module docstring. proj: make_linear(patch_len*Ebyte -> D).
    Params/compute scale with patch_len; only use for small patch_len."""
    proj: eqx.Module
    patch_len: int = eqx.field(static=True)
    byte_embed_dim: int = eqx.field(static=True)

    def __init__(self, patch_len: int, byte_embed_dim: int, d_out: int, cfg: "ModelConfig", keyseq: KeySeq):
        self.patch_len = patch_len
        self.byte_embed_dim = byte_embed_dim
        self.proj = make_linear(patch_len * byte_embed_dim, d_out, cfg, keyseq)

    def __call__(self, emb: jax.Array) -> jax.Array:
        flat = emb.reshape(*emb.shape[:-2], self.patch_len * self.byte_embed_dim)
        return self.proj(flat)


class PatchInMeanPool(eqx.Module):
    """"mean_pool" patch_in scheme (default) -- see module docstring. Mean-pools byte embeddings
    across the patch_len axis, then make_linear(Ebyte -> D), independent of patch_len."""
    proj: eqx.Module

    def __init__(self, byte_embed_dim: int, d_out: int, cfg: "ModelConfig", keyseq: KeySeq):
        self.proj = make_linear(byte_embed_dim, d_out, cfg, keyseq)

    def __call__(self, emb: jax.Array) -> jax.Array:
        pooled = emb.mean(axis=-2)
        return self.proj(pooled)


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.weight = jnp.ones(d_model)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x * self.weight


class Attn(eqx.Module):
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    wq: eqx.Module
    wk: eqx.Module
    wv: eqx.Module
    out: eqx.Module

    def __init__(self, d_model: int, n_heads: int, cfg: "ModelConfig", keyseq: KeySeq):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = make_linear(d_model, d_model, cfg, keyseq)
        self.wk = make_linear(d_model, d_model, cfg, keyseq)
        self.wv = make_linear(d_model, d_model, cfg, keyseq)
        self.out = make_linear(d_model, d_model, cfg, keyseq)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, mask: jax.Array,
                 use_flash: bool = False) -> jax.Array:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        q = self.wq(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = flash_attn_causal(q, k, v) if use_flash else sdpa(q, k, v, mask)
        return self.out(y.transpose(0, 2, 1, 3).reshape(B, T, D))

    def step(self, x: jax.Array, cos_t: jax.Array, sin_t: jax.Array, t: jax.Array,
              k_cache: jax.Array, v_cache: jax.Array):
        """One-token incremental step for a FIXED-SIZE, preallocated KV cache -- see
        enfrac_zero/model.py's Attn.step docstring (identical logic, HiRA-agnostic: wq/wk/wv/out
        are whichever Linear cfg.use_hira selects, called the same way either way)."""
        B, T, D = x.shape   # T == 1
        H, hd = self.n_heads, self.head_dim
        t = t.astype(jnp.int32)   # normalize regardless of caller/x64 mode -- dynamic_update_slice
                                    # requires ALL its index args to share one dtype, and bare
                                    # python 0 literals promote to int64 under jax_enable_x64
        zero = jnp.zeros((), dtype=t.dtype)
        q = self.wq(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k_t = self.wk(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v_t = self.wv(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        q, k_t = apply_rope(q, cos_t, sin_t), apply_rope(k_t, cos_t, sin_t)
        k_cache = jax.lax.dynamic_update_slice(k_cache, k_t, (zero, zero, t, zero))
        v_cache = jax.lax.dynamic_update_slice(v_cache, v_t, (zero, zero, t, zero))
        max_len = k_cache.shape[2]
        valid = jnp.arange(max_len) <= t                              # [max_len] -- causal mask
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k_cache) / math.sqrt(hd)   # [1,H,1,max_len]
        scores = jnp.where(valid[None, None, None, :], scores, jnp.finfo(scores.dtype).min)
        attn = jax.nn.softmax(scores, axis=-1)
        y = jnp.einsum("bhts,bhsd->bhtd", attn, v_cache)
        out = self.out(y.transpose(0, 2, 1, 3).reshape(B, T, D))
        return out, k_cache, v_cache


class SwiGLU(eqx.Module):
    gate: eqx.Module
    up: eqx.Module
    down: eqx.Module

    def __init__(self, d_model: int, mlp_mult: int, cfg: "ModelConfig", keyseq: KeySeq):
        hidden = mlp_mult * d_model
        self.gate = make_linear(d_model, hidden, cfg, keyseq)
        self.up = make_linear(d_model, hidden, cfg, keyseq)
        self.down = make_linear(hidden, d_model, cfg, keyseq)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


class Block(eqx.Module):
    ln1: RMSNorm
    attn: Attn
    ln2: RMSNorm
    mlp: SwiGLU

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, cfg: "ModelConfig", keyseq: KeySeq):
        self.ln1 = RMSNorm(d_model)
        self.attn = Attn(d_model, n_heads, cfg, keyseq)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult, cfg, keyseq)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, mask: jax.Array,
                 use_flash: bool = False) -> jax.Array:
        x = x + self.attn(self.ln1(x), cos, sin, mask, use_flash)
        x = x + self.mlp(self.ln2(x))
        return x

    def step(self, x: jax.Array, cos_t: jax.Array, sin_t: jax.Array, t: jax.Array,
              k_cache: jax.Array, v_cache: jax.Array):
        attn_out, k_cache, v_cache = self.attn.step(self.ln1(x), cos_t, sin_t, t, k_cache, v_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, k_cache, v_cache


def _remat_group_size(spec: str, total: int) -> int:
    """'0.5' -> fraction of total (rounded, >=1); '512' -> exact count, capped at total."""
    n = max(1, round(float(spec) * total)) if "." in spec else int(spec)
    return min(max(1, n), total)


@eqx.filter_jit
def _trunk_forward_jit(trunk: "Trunk", x: jax.Array, rope_base: float) -> jax.Array:
    """Module-level jit wrapper for Trunk.__call__, used by _gen_children's recompute-from-scratch
    loop (levels 1+) -- see enfrac_zero/model.py's identical function for the full rationale
    (bounded shapes T=1..seq_len mean jax.jit's shape-keyed cache compiles each once and reuses
    it for every subsequent call; measured ~30x faster per call than bare eager at juz1 scale).
    HiRA-agnostic -- trunk's Linear type doesn't affect this."""
    return trunk(x, rope_base)


@eqx.filter_jit
def _trunk_step_jit(trunk: "Trunk", x_t: jax.Array, t_arr: jax.Array, rope_base: float, kv_cache: list):
    """Module-level jit wrapper for Trunk.step, used by generate()'s level-0 decode loop -- see
    enfrac_zero/model.py's identical function for the full rationale: THIS WAS A REAL BUG, found
    and fixed this session -- generate() used to define `step_fn` as a LOCAL closure decorated
    with @eqx.filter_jit INSIDE its own body, which only avoided recompiling WITHIN one
    generate() call (the fixed-size KV cache keeps shapes constant across steps t=0..n_timesteps)
    but did nothing for the cost of calling generate() MULTIPLE times (e.g. once per file chunk
    in compress.py's/decompress.py's per-chunk loop) -- every fresh call created a brand new
    Python closure object, and jax.jit's/eqx.filter_jit's compiled-executable cache keys in part
    on the wrapped callable's own identity, so every call was a guaranteed cache miss regardless
    of shape. Moving step_fn to module level (here), the same fix _trunk_forward_jit already
    applied to _gen_children, makes generate() compile once per process per distinct
    (n_timesteps, kv_cache shape), not once per call -- measured ~26x speedup on a repeated-call
    micro-benchmark. HiRA-agnostic -- trunk's Linear type doesn't affect this."""
    return trunk.step(x_t, t_arr, rope_base, kv_cache)


class Trunk(eqx.Module):
    """One level's transformer: a Block stack + final norm. Instantiated once per level by
    default; aliased (literally the same instance) across levels when share_trunk=True."""
    head_dim: int = eqx.field(static=True)
    blocks: list
    ln_f: RMSNorm

    def __init__(self, d_model: int, n_heads: int, n_layers: int, mlp_mult: int,
                 cfg: "ModelConfig", keyseq: KeySeq):
        self.head_dim = d_model // n_heads
        self.blocks = [Block(d_model, n_heads, mlp_mult, cfg, keyseq) for _ in range(n_layers)]
        self.ln_f = RMSNorm(d_model)

    def __call__(self, x: jax.Array, rope_base: float, use_flash: bool = False) -> jax.Array:
        T = x.shape[1]
        pos = jnp.arange(T)
        cos, sin = rope_cos_sin_for_positions(pos, self.head_dim, rope_base)
        mask = causal_mask(T)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask, use_flash)
        return self.ln_f(x)

    def init_kv_cache(self, max_len: int, batch: int = 1):
        """Preallocated, FIXED-SIZE (max_len = that level's seq_len, or n_timesteps for level 0)
        zero KV cache, one (k, v) pair per block -- see enfrac_zero/model.py's
        Trunk.init_kv_cache docstring (identical, HiRA-agnostic). `batch` is the number of
        independent parallel local sequences sharing this cache's shape -- 1 for level 0, or Bcur
        (all parents at that level) for levels 1+. dtype is pinned to self.ln_f.weight's own dtype
        (not left to jnp.zeros' ambient default) -- REAL BUG found this session: under
        jax_enable_x64=True (compress.py's dtype="float64" default) with a float32 model
        (collect_logits_batched's chunk_batch_size>1 path, which never casts to float64), a bare
        jnp.zeros(...) here defaults to float64 while k_t/v_t computed from the float32 weights
        stay float32, and jax.lax.dynamic_update_slice in Attn.step then hard-crashes on the dtype
        mismatch (not a silent corruption, but still a real crash found on real TPU hardware)."""
        kv_dtype = self.ln_f.weight.dtype
        return [
            (jnp.zeros((batch, blk.attn.n_heads, max_len, blk.attn.head_dim), dtype=kv_dtype),
             jnp.zeros((batch, blk.attn.n_heads, max_len, blk.attn.head_dim), dtype=kv_dtype))
            for blk in self.blocks
        ]

    def step(self, x: jax.Array, t: jax.Array, rope_base: float, kv_cache: list):
        """Real incremental KV-cache decoding for ONE new level-0 timestep -- see
        enfrac_zero/model.py's Trunk.step docstring for the full rationale (O(t) per step,
        O(n_timesteps^2) total, vs the old O(n_timesteps^3) recompute-from-scratch). t must be a
        traced jnp scalar (not a python int) so the caller's single jit trace is reused."""
        cos_t, sin_t = rope_cos_sin_for_positions(jnp.asarray(t)[None], self.head_dim, rope_base)
        new_cache = []
        for blk, (k_c, v_c) in zip(self.blocks, kv_cache):
            x, k_c, v_c = blk.step(x, cos_t, sin_t, t, k_c, v_c)
            new_cache.append((k_c, v_c))
        return self.ln_f(x), new_cache

    def forward_remat(self, x: jax.Array, rope_base: float, remat_time: str, remat_depth: str,
                       use_flash: bool = False) -> jax.Array:
        """Level 0's full-sequence forward -- IDENTICAL math to __call__ (real causal attention
        over the whole sequence, no truncation, no detach), just computed under jax.checkpoint
        for memory. See enfrac_zero/model.py's Trunk.forward_remat -- identical logic, HiRA-
        agnostic (operates only on the Block list). use_flash: see flash_attn_causal's docstring
        -- only sensible once T is bumped well past level 0's default (e.g. >=512, see
        docs/enwik9_scaling_calcs.md)."""
        T = x.shape[1]
        pos = jnp.arange(T)
        cos, sin = rope_cos_sin_for_positions(pos, self.head_dim, rope_base)
        mask = causal_mask(T)
        depth_group = _remat_group_size(remat_depth, len(self.blocks))
        time_group = _remat_group_size(remat_time, T)

        def run_all_layers(x):
            for start in range(0, len(self.blocks), depth_group):
                group = self.blocks[start:start + depth_group]
                def run_group(x, group=group):
                    for blk in group:
                        x = blk(x, cos, sin, mask, use_flash)
                    return x
                x = jax.checkpoint(run_group)(x) if depth_group < len(self.blocks) else run_group(x)
            return x

        x = jax.checkpoint(run_all_layers)(x) if time_group < T else run_all_layers(x)
        return self.ln_f(x)


@dataclass(frozen=True)
class ModelConfig:
    patch_len_list: tuple = (1024, 128, 16, 1)   # patch_len_list[l] = level l's OWN patch size
                                                    # (all levels, including 0); last elem must be 1
    d_model_list: tuple = (64, 64, 64, 64)        # length = n_levels = len(patch_len_list)
    n_layers_list: tuple = (4, 4, 4, 4)
    n_heads_list: tuple = (4, 4, 4, 4)
    mlp_mult_list: tuple = (2, 2, 2, 2)
    byte_embed_dim: int = 256                     # decoupled from any level's d_model
    rope_preset: str = "qwen3"
    share_trunk: bool = False                     # opt-in alias; requires matching dims
    patch_in_scheme: str | tuple = "mean_pool"     # "mean_pool" (default) or "linear" -- see
                                                    # module docstring. Either a single string
                                                    # (broadcast to every level) or a tuple of
                                                    # length n_levels, one scheme per level (e.g.
                                                    # ("mean_pool","mean_pool","mean_pool","linear",
                                                    # "linear","mean_pool") to use the exact
                                                    # "linear" scheme only at small-patch levels
                                                    # where it's cheap -- see PatchInLinear's
                                                    # docstring: cost is patch_len*byte_embed_dim*
                                                    # d_model, only tractable for small patch_len).
                                                    # patch_len==1 levels are unaffected either way
                                                    # (always a plain Linear, no patch_in scheme).
    use_hira: bool = True                          # main-package default; False = plain linear
    hira_r: int = 4                                 # HiRA rank -- only used when use_hira=True
    seed: int = 0                                    # drives frozen W0/A -- must match at load time
    use_flash_attn_list: tuple | None = None        # per-level bool, length n_levels; None ->
                                                    # all False. See flash_attn_causal's docstring:
                                                    # only sensible for a level whose seq_len is
                                                    # >=128 (ideally a clean multiple of 128) --
                                                    # levels 1+'s default seq_len=8 gets no benefit
                                                    # and isn't even supported by the kernel.
                                                    # TRAINING-forward only (__call__/_train_recurse
                                                    # via ByteFractalGen); decode (_gen_children/
                                                    # generate()) never uses this, which is fine --
                                                    # __call__ has no bit-exactness requirement
                                                    # against generate() (see module docstring).


def make_byte_embedding(dim: int) -> jax.Array:
    """Fixed, MAXIMALLY separated per-byte representation -- exact one-hot (mutually orthogonal)
    when dim>=256; fixed random unit vectors otherwise. Never trained, never saved -- regenerated
    from a hardcoded seed on every load (see module docstring) -- so dtype=float32 is forced
    explicitly here, NOT left to the ambient default: jax.random.normal with a given PRNGKey
    produces COMPLETELY DIFFERENT values (not just different precision) depending on whether
    jax_enable_x64 is on, since generating a float64 sample consumes different random bits than a
    float32 one. Without this, constructing the model under jax_enable_x64=True (e.g. for
    collect_logits_fp64) would silently give every byte a DIFFERENT embedding than what training
    actually used -- verified empirically this session (byte_acc collapsed ~76% -> ~25%). A later
    cast_dtype(jnp.float64) is fine -- that upcasts these CORRECT float32 values, it doesn't
    re-randomize them."""
    if dim >= 256:
        w = jnp.zeros((256, dim), dtype=jnp.float32)
        w = w.at[:, :256].set(jnp.eye(256, dtype=jnp.float32))
    else:
        g = jax.random.PRNGKey(0)
        w = jax.random.normal(g, (256, dim), dtype=jnp.float32)
        w = w / jnp.linalg.norm(w, axis=-1, keepdims=True)
    return w


class FrozenEmbedding(eqx.Module):
    table: jax.Array   # named "table", NOT "weight" -- excluded from trainable_filter on purpose

    def __call__(self, idx: jax.Array) -> jax.Array:
        return self.table[idx]


class ByteFractalGen(eqx.Module):
    cfg: "ModelConfig" = eqx.field(static=True)
    n_levels: int = eqx.field(static=True)
    seq_lens: tuple = eqx.field(static=True)
    patch_in_scheme: tuple = eqx.field(static=True)   # resolved per-level ("mean_pool"|"linear"),
                                                        # always length n_levels regardless of
                                                        # whether cfg.patch_in_scheme was a single
                                                        # str or a tuple -- embed_patch's memory-
                                                        # safety check indexes this PER LEVEL, never
                                                        # cfg.patch_in_scheme directly (comparing a
                                                        # tuple to the literal "mean_pool" is always
                                                        # False, which would silently disable the
                                                        # chunked mean-pool OOM guard at every level).
    use_flash_attn: tuple = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)
    byte_embed: FrozenEmbedding
    trunks: list
    patch_in: list
    cond_proj: list
    head: eqx.Module
    root_cond: jax.Array

    def __init__(self, cfg: ModelConfig):
        assert cfg.patch_len_list[-1] == 1, "last patch_len must be 1 (byte-atomic terminal level)"
        for a, b in zip(cfg.patch_len_list, cfg.patch_len_list[1:]):
            assert a % b == 0, f"patch_len_list must divide evenly level to level, got {a} -> {b}"
        self.cfg = cfg
        self.n_levels = len(cfg.patch_len_list)
        self.patch_in_scheme = (cfg.patch_in_scheme,) * self.n_levels \
            if isinstance(cfg.patch_in_scheme, str) else tuple(cfg.patch_in_scheme)
        assert len(self.patch_in_scheme) == self.n_levels, \
            f"patch_in_scheme tuple must have length n_levels={self.n_levels}, got {len(self.patch_in_scheme)}"
        assert all(s in ("mean_pool", "linear") for s in self.patch_in_scheme), \
            f"patch_in_scheme entries must be 'mean_pool' or 'linear', got {self.patch_in_scheme!r}"
        self.seq_lens = (0,) + tuple(
            cfg.patch_len_list[l - 1] // cfg.patch_len_list[l] for l in range(1, self.n_levels))
        self.use_flash_attn = cfg.use_flash_attn_list if cfg.use_flash_attn_list is not None \
            else (False,) * self.n_levels
        assert len(self.use_flash_attn) == self.n_levels, \
            f"use_flash_attn_list must have length n_levels={self.n_levels}, got {len(self.use_flash_attn)}"
        for name, lst in [("d_model_list", cfg.d_model_list), ("n_layers_list", cfg.n_layers_list),
                          ("n_heads_list", cfg.n_heads_list), ("mlp_mult_list", cfg.mlp_mult_list)]:
            assert len(lst) == self.n_levels, f"{name} must have length n_levels={self.n_levels}, got {len(lst)}"
        for l, (d, h) in enumerate(zip(cfg.d_model_list, cfg.n_heads_list)):
            assert d % h == 0, f"level {l}: d_model={d} not divisible by n_heads={h}"
            assert (d // h) % 2 == 0, (
                f"level {l}: head_dim={d // h} (d_model={d}/n_heads={h}) must be even -- "
                f"RoPE's rotate_half splits it in two")
        self.rope_base = ROPE_PRESETS[cfg.rope_preset]

        # Single KeySeq threaded through ALL frozen-weight construction below, in a FIXED order
        # (see module docstring's determinism contract) -- this IS the "seed" that lets a HiRA
        # bundle reconstruct its frozen base from cfg.seed alone.
        keyseq = KeySeq(cfg.seed)

        # Frozen, maximally-separated byte representation -- fixed data, never trained.
        # Independent of `keyseq`/HiRA -- frozen-by-construction, not part of the PEFT mechanism.
        self.byte_embed = FrozenEmbedding(make_byte_embedding(cfg.byte_embed_dim))

        if cfg.share_trunk:
            d0, l0, h0, m0 = cfg.d_model_list[0], cfg.n_layers_list[0], cfg.n_heads_list[0], cfg.mlp_mult_list[0]
            for l in range(self.n_levels):
                assert (cfg.d_model_list[l], cfg.n_layers_list[l], cfg.n_heads_list[l], cfg.mlp_mult_list[l]) \
                    == (d0, l0, h0, m0), \
                    (f"share_trunk=True requires identical dims at every level; level {l} "
                     f"({cfg.d_model_list[l]},{cfg.n_layers_list[l]},{cfg.n_heads_list[l]},{cfg.mlp_mult_list[l]}) "
                     f"!= level 0 ({d0},{l0},{h0},{m0})")
            # A single Trunk instance, called n_levels times from __call__/generate (see
            # trunk_at()) -- NOT `[shared] * n_levels` stored in the list. JAX pytrees flatten by
            # structural position, not object identity, so a list with the same Trunk repeated
            # would produce n_levels INDEPENDENT leaf copies (unlike PyTorch's ModuleList, where
            # object-identity aliasing makes autograd accumulate one shared .grad); those copies
            # would drift apart after the very first optimizer step. Keeping exactly one Trunk
            # object and reusing it inside a single forward call keeps it a single pytree leaf,
            # so gradients from all n_levels uses correctly accumulate onto that one leaf.
            self.trunks = [Trunk(d0, h0, l0, m0, cfg, keyseq)]
        else:
            self.trunks = [
                Trunk(cfg.d_model_list[l], cfg.n_heads_list[l], cfg.n_layers_list[l], cfg.mlp_mult_list[l], cfg, keyseq)
                for l in range(self.n_levels)
            ]

        # patch_in[l] embeds level l's OWN patch_len_list[l] bytes -> d_model_list[l]. Every
        # level uses the same construction, uniformly -- see module docstring for the two schemes.
        # FIXED order (l=0..n_levels-1), part of the determinism contract.
        self.patch_in = []
        for l in range(self.n_levels):
            P = cfg.patch_len_list[l]
            d_out = cfg.d_model_list[l]
            if P == 1:
                self.patch_in.append(make_linear(cfg.byte_embed_dim, d_out, cfg, keyseq))
            elif self.patch_in_scheme[l] == "linear":
                self.patch_in.append(PatchInLinear(P, cfg.byte_embed_dim, d_out, cfg, keyseq))
            else:
                self.patch_in.append(PatchInMeanPool(cfg.byte_embed_dim, d_out, cfg, keyseq))

        # cond_proj[l]: projects the PARENT's d_model_list[l-1] -> this level's d_model_list[l],
        # for l>=1 only -- level 0 has no parent (root_cond fed directly, unprojected). FIXED
        # order (l=1..n_levels-1), part of the determinism contract.
        self.cond_proj = [None] + [
            make_linear(cfg.d_model_list[l - 1], cfg.d_model_list[l], cfg, keyseq)
            for l in range(1, self.n_levels)
        ]

        self.head = make_linear(cfg.d_model_list[-1], 256, cfg, keyseq)   # terminal level only
        self.root_cond = 0.02 * jax.random.normal(keyseq.next(), (1, cfg.d_model_list[0]))

    def trunk_at(self, level: int) -> "Trunk":
        return self.trunks[0] if self.cfg.share_trunk else self.trunks[level]

    def embed_patch(self, level: int, byte_patch: jax.Array) -> jax.Array:
        """byte_patch: [..., patch_len_list[level]] int32 -> [..., d_model_list[level]]. The one
        place patch_in[level] is ever called from -- keeps __call__/generate/_gen_children using
        the exact same embedding math (needed for collect_logits()/generate() bit-exactness).
        mean_pool is routed through _chunked_mean_pool instead of a plain
        `self.byte_embed(byte_patch)` whenever the TOTAL element count (leading rows * P) is
        large -- see that method's docstring for why: the naive call materializes a
        [..., P, byte_embed_dim] intermediate BEFORE pooling, which can overflow TPU HBM even
        though the mean-pool scheme's whole point is params/compute independent of P. This isn't
        only a large-P problem: the LEADING row count also matters, since embed_patch is called
        from inside _train_recurse's scan body with up to micro_batch*seq_len rows at once --
        verified a [65536, 32768, 256] intermediate (level 2's own P=32768, with 65536 rows
        flowing in from a chunked parent level) would have requested ~550GB, an order of
        magnitude worse than the large-P case this was first written for."""
        P = self.cfg.patch_len_list[level]
        if P == 1:
            emb = self.byte_embed(byte_patch[..., 0])          # [..., Ebyte]
            return self.patch_in[level](emb)
        if self.patch_in_scheme[level] == "mean_pool":
            leading = 1
            for s in byte_patch.shape[:-1]:
                leading *= s
            if leading * P > _MEAN_POOL_TARGET_ELEMENTS:
                pooled = self._chunked_mean_pool(byte_patch, P)   # [..., Ebyte]
            else:
                pooled = self.byte_embed(byte_patch).mean(axis=-2)   # [..., Ebyte] -- direct, small enough
            return self.patch_in[level].proj(pooled)
        emb = self.byte_embed(byte_patch)                       # [..., P, Ebyte]
        return self.patch_in[level](emb)

    def _chunked_mean_pool(self, byte_patch: jax.Array, P: int) -> jax.Array:
        """Computes byte_embed(byte_patch).mean(axis=-2) without ever materializing the full
        [rows, P, byte_embed_dim] intermediate -- see embed_patch's docstring for why that matters
        (both large P AND large row counts can make it huge; a first version of this method only
        bounded P, which left a much larger row-count-driven blowup -- ~550GB at level 2 -- in
        place; found and fixed before it was ever exercised for real). Chunks over the FLATTENED
        ROW axis only (never over P): each chunk of `n_chunk` rows, computed and pooled via
        jax.lax.scan, keeps the per-step intermediate to roughly n_chunk*P*byte_embed_dim
        elements, and n_chunk is chosen (`_MEAN_POOL_TARGET_ELEMENTS // P`) so that PRODUCT stays
        around a fixed, small target regardless of P -- i.e. the chunk shrinks automatically for
        large P, and grows (up to the row count) for small P. Each row's own mean is independent
        of every other row, so padding rows (added so the row count divides evenly) can just be
        computed as normal (garbage in, garbage out) and sliced away at the end -- no valid-mask
        needed, unlike _train_recurse's chunking, since there's no cross-row accumulation here."""
        leading_shape = byte_patch.shape[:-1]
        flat = byte_patch.reshape(-1, P)             # [N, P]
        N = flat.shape[0]
        n_chunk = max(1, _MEAN_POOL_TARGET_ELEMENTS // P)

        if N <= n_chunk:
            return self.byte_embed(flat).mean(axis=1).reshape(*leading_shape, -1)

        n_groups = -(-N // n_chunk)   # ceil
        pad = n_groups * n_chunk - N
        if pad:
            flat = jnp.concatenate([flat, jnp.zeros((pad, P), flat.dtype)], axis=0)
        flat_r = flat.reshape(n_groups, n_chunk, P)

        def body(carry, x_chunk):
            e = self.byte_embed(x_chunk)              # [n_chunk, P, Ebyte]
            return carry, e.mean(axis=1)                # [n_chunk, Ebyte]

        _, pooled_chunks = jax.lax.scan(body, None, flat_r)   # [n_groups, n_chunk, Ebyte]
        pooled = pooled_chunks.reshape(n_groups * n_chunk, -1)[:N]
        return pooled.reshape(*leading_shape, -1)

    def _train_recurse(self, level: int, cond: jax.Array, bytes_flat: jax.Array, valid: jax.Array,
                        remat_depth: str, micro_batch: int, use_scan: bool = True):
        """jax.lax.scan-chunked replacement for the old direct "compute this whole level's Bcur-
        batched forward in one shot" loop body. Necessary because at real corpus scale (enwik8/9)
        the row count at deep levels -- level l's Bcur == total patch count of patch_len_list[l]-
        sized patches in the whole file, which reaches the TOTAL RAW BYTE COUNT by the terminal
        level -- can be in the hundreds of millions to billions. Materializing that many rows at
        once overflows TPU HBM: verified empirically that a plain `cur_bytes.reshape(-1, seq_len,
        child_len)` on a 125,042,688-row int32 array alone requested 64GB against a v4-8's 34GB
        HBM (TPU pads the size-8 minor dim up to a 128-tile, a 16x blowup on top of the ~4GB raw
        data). Fixed by processing `cond`/`bytes_flat` in `micro_batch`-sized chunks via
        jax.lax.scan -- a REAL loop, compiled ONCE regardless of chunk count (a Python-unrolled
        loop instead would blow up compile time/graph size at these row counts, since the leaf
        level alone can have hundreds of thousands of chunks). Each chunk's own trunk call uses
        plain Trunk.__call__ (NOT forward_remat/jax.checkpoint) -- combining jax.checkpoint with
        this nested jax.lax.scan structure was tried and hit a real XLA compiler crash on TPU
        (`windowing_util.cc: VerifyCanonicalBounds` / `RET_CHECK ... CouldLeS32` / SIGABRT),
        confirmed via a controlled A/B (disabling just the checkpoint call made the crash
        disappear, replaced by an unrelated, separately-fixed OOM). remat_depth is threaded
        through the recursion only because __call__'s own signature already carries it for level
        0's forward_remat call; it does nothing at levels 1+ for now. seq_len is small (8) at
        every level, so per-chunk activation memory here is bounded by chunk_size (<= micro_batch)
        regardless -- checkpointing wasn't needed for memory at these levels, only chunking was.
        HiRA-agnostic: operates only on cond_proj/embed_patch/trunk_at/head, whose Linear type
        doesn't affect this.

        Non-terminal levels recurse into level+1 IMMEDIATELY for each chunk (depth-first), rather
        than assembling this level's full [Bcur*seq_len, D] output array first -- for the deepest
        transitions that output array is itself hundreds of GB, too large to exist even
        momentarily regardless of how the compute that produced it was chunked. This bounds peak
        memory by micro_batch at every level simultaneously, not by any level's total row count.

        cond: [B, D_{level-1}] parent conditioning (pre cond_proj[level]) -- for level==1 this is
        level 0's own output. bytes_flat: [B, patch_len_list[level-1]] this batch's own bytes at
        the PARENT level's patch granularity (split into this level's (seq_len, child_len)
        structure per chunk). valid: [B] bool -- False marks rows that are zero-padding (added so
        B is a multiple of micro_batch), excluded from every returned total. Returns
        (ce_nats_sum, position_count, correct_sum), all jnp scalars, aggregated over valid rows
        across the WHOLE subtree from `level` down to the terminal level."""
        seq_len = self.seq_lens[level]
        child_len = self.cfg.patch_len_list[level]
        B = cond.shape[0]
        D_in = cond.shape[1]
        parent_len = bytes_flat.shape[1]

        # chunk_size == B (no padding, one chunk) whenever B already fits in one micro_batch --
        # only pad up to a micro_batch-multiple when B actually EXCEEDS it. Using a fixed
        # chunk_size=micro_batch unconditionally (padding small B up to it) would inflate every
        # level's row count by (micro_batch/B)x, and since each level's B multiplies by seq_len
        # on recursion, that inflation compounds through the WHOLE remaining subtree -- wasting
        # that same factor of compute at every deeper level too, not just this one.
        if B <= micro_batch:
            chunk_size = B
            n_chunks = 1
        else:
            chunk_size = micro_batch
            n_chunks = -(-B // micro_batch)   # ceil
        pad = n_chunks * chunk_size - B
        if pad:
            cond = jnp.concatenate([cond, jnp.zeros((pad, D_in), cond.dtype)], axis=0)
            bytes_flat = jnp.concatenate([bytes_flat, jnp.zeros((pad, parent_len), bytes_flat.dtype)], axis=0)
            valid = jnp.concatenate([valid, jnp.zeros((pad,), dtype=jnp.bool_)], axis=0)

        cond_r = cond.reshape(n_chunks, chunk_size, D_in)
        bytes_r = bytes_flat.reshape(n_chunks, chunk_size, parent_len)
        valid_r = valid.reshape(n_chunks, chunk_size)

        def body(carry, xs):
            ce_acc, pos_acc, correct_acc = carry
            cond_chunk, bytes_chunk_flat, valid_chunk = xs
            bytes_chunk = bytes_chunk_flat.reshape(chunk_size, seq_len, child_len)
            cond_l = self.cond_proj[level](cond_chunk)                        # [m, D]
            proj = self.embed_patch(level, bytes_chunk)                        # [m, seq_len, D]

            if child_len == 1:
                shifted = jnp.concatenate([cond_l[:, None, :], proj[:, :-1]], axis=1)
                h = self.trunk_at(level).forward_remat(shifted, self.rope_base, "1.0", remat_depth,
                                                         self.use_flash_attn[level])
                logits = self.head(h)                                          # [m, seq_len, 256]
                targets = bytes_chunk[..., 0]                                   # [m, seq_len]
                logp = jax.nn.log_softmax(logits, axis=-1)
                nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
                row_valid = valid_chunk[:, None].astype(nll.dtype)              # [m, 1]
                ce_acc = ce_acc + (nll * row_valid).sum()
                pos_acc = pos_acc + row_valid.sum() * seq_len
                correct_acc = correct_acc + ((logits.argmax(-1) == targets) * row_valid).sum()
                return (ce_acc, pos_acc, correct_acc), None
            else:
                seq = jnp.concatenate([cond_l[:, None, :], proj], axis=1)
                h = self.trunk_at(level).forward_remat(seq, self.rope_base, "1.0", remat_depth,
                                                         self.use_flash_attn[level])
                cond_next = h[:, :-1, :].reshape(chunk_size * seq_len, -1)
                bytes_next = bytes_chunk.reshape(chunk_size * seq_len, child_len)
                valid_next = jnp.repeat(valid_chunk, seq_len)
                sub_ce, sub_pos, sub_correct = self._train_recurse(
                    level + 1, cond_next, bytes_next, valid_next, remat_depth, micro_batch, use_scan)
                return (ce_acc + sub_ce, pos_acc + sub_pos, correct_acc + sub_correct), None

        init = (jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))
        if use_scan:
            (ce_total, pos_total, correct_total), _ = jax.lax.scan(body, init, (cond_r, bytes_r, valid_r))
        else:
            # ABLATION (temporary, diagnostic only) -- see enfrac_zero/model.py's identical twin
            # for the full rationale: tests whether jax.lax.scan itself (not chunk sizes) is what
            # triggers XLA to fuse the nested per-level structure into one giant HLO.
            carry = init
            for i in range(n_chunks):
                carry, _ = body(carry, (cond_r[i], bytes_r[i], valid_r[i]))
            ce_total, pos_total, correct_total = carry
        return ce_total, pos_total, correct_total

    def _train_flat(self, cond: jax.Array, bytes_flat: jax.Array, valid: jax.Array,
                     remat_depth: str, micro_batch: int,
                     level_ckpt: bool = os.environ.get("LEVEL_CKPT") == "1"):
        """FLAT (non-recursive, non-nested-scan) replacement for _train_recurse -- see
        enfrac_zero/model.py's identical twin for the full rationale (diagnostic for the
        scan-fusion OOM that persisted regardless of micro_batch: 8192->1024->128 changed peak
        HBM by <7%). Walks levels 1..n_levels-1 in a plain Python for-loop (only ~6 levels, not a
        chunk-level unroll) with each level's own micro_batch-chunked jax.lax.scan called at the
        TOP level, never nested inside another scan's traced body."""
        ce_total = jnp.zeros(())
        pos_total = jnp.zeros(())
        correct_total = jnp.zeros(())
        for level in range(1, self.n_levels):
            seq_len = self.seq_lens[level]
            child_len = self.cfg.patch_len_list[level]
            terminal = (child_len == 1)
            B = cond.shape[0]
            D_in = cond.shape[1]
            parent_len = bytes_flat.shape[1]

            if B <= micro_batch:
                chunk_size, n_chunks = B, 1
            else:
                chunk_size, n_chunks = micro_batch, -(-B // micro_batch)
            pad = n_chunks * chunk_size - B
            if pad:
                cond = jnp.concatenate([cond, jnp.zeros((pad, D_in), cond.dtype)], axis=0)
                bytes_flat = jnp.concatenate([bytes_flat, jnp.zeros((pad, parent_len), bytes_flat.dtype)], axis=0)
                valid = jnp.concatenate([valid, jnp.zeros((pad,), dtype=jnp.bool_)], axis=0)

            cond_r = cond.reshape(n_chunks, chunk_size, D_in)
            bytes_r = bytes_flat.reshape(n_chunks, chunk_size, parent_len)
            valid_r = valid.reshape(n_chunks, chunk_size)

            def body(carry, xs, level=level, chunk_size=chunk_size, seq_len=seq_len,
                      child_len=child_len, terminal=terminal):
                ce_acc, pos_acc, correct_acc = carry
                cond_chunk, bytes_chunk_flat, valid_chunk = xs
                bytes_chunk = bytes_chunk_flat.reshape(chunk_size, seq_len, child_len)
                cond_l = self.cond_proj[level](cond_chunk)
                proj = self.embed_patch(level, bytes_chunk)
                if terminal:
                    shifted = jnp.concatenate([cond_l[:, None, :], proj[:, :-1]], axis=1)
                    h = self.trunk_at(level).forward_remat(shifted, self.rope_base, "1.0", remat_depth,
                                                             self.use_flash_attn[level])
                    logits = self.head(h)
                    targets = bytes_chunk[..., 0]
                    logp = jax.nn.log_softmax(logits, axis=-1)
                    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
                    row_valid = valid_chunk[:, None].astype(nll.dtype)
                    ce_acc = ce_acc + (nll * row_valid).sum()
                    pos_acc = pos_acc + row_valid.sum() * seq_len
                    correct_acc = correct_acc + ((logits.argmax(-1) == targets) * row_valid).sum()
                    return (ce_acc, pos_acc, correct_acc), None
                else:
                    seq = jnp.concatenate([cond_l[:, None, :], proj], axis=1)
                    h = self.trunk_at(level).forward_remat(seq, self.rope_base, "1.0", remat_depth,
                                                             self.use_flash_attn[level])
                    cond_next = h[:, :-1, :].reshape(chunk_size * seq_len, -1)
                    bytes_next = bytes_chunk.reshape(chunk_size * seq_len, child_len)
                    valid_next = jnp.repeat(valid_chunk, seq_len)
                    return (ce_acc, pos_acc, correct_acc), (cond_next, bytes_next, valid_next)

            init = (jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))

            def run_level(cond_r, bytes_r, valid_r):
                return jax.lax.scan(body, init, (cond_r, bytes_r, valid_r))

            if level_ckpt:
                # Checkpoint this level's ENTIRE scan -- see enfrac_zero/model.py's identical twin
                # for the full rationale (safe now that _train_flat is non-nested, unlike the old
                # recursive structure where checkpoint+nested-scan crashed the XLA compiler).
                (ce_l, pos_l, correct_l), ys = jax.checkpoint(run_level)(cond_r, bytes_r, valid_r)
            else:
                (ce_l, pos_l, correct_l), ys = run_level(cond_r, bytes_r, valid_r)
            ce_total, pos_total, correct_total = ce_total + ce_l, pos_total + pos_l, correct_total + correct_l
            if terminal:
                break
            cond_chunks, bytes_chunks, valid_chunks = ys
            cond = cond_chunks.reshape(n_chunks * chunk_size * seq_len, -1)
            bytes_flat = bytes_chunks.reshape(n_chunks * chunk_size * seq_len, child_len)
            valid = valid_chunks.reshape(n_chunks * chunk_size * seq_len)
        return ce_total, pos_total, correct_total

    def __call__(self, byte_seq: jax.Array, remat_time: str = "1.0", remat_depth: str = "1.0",
                 micro_batch: int = 8192,
                 use_scan: bool = os.environ.get("DISABLE_SCAN") != "1",
                 flat_scan: bool = os.environ.get("FLAT_SCAN") != "0") -> tuple[jax.Array, dict]:
        """byte_seq: [B, n_timesteps, patch_len_list[0]] int32 -- B independent FILE CHUNKS
        (weight-shared, never attending across each other), each a training window's own
        level-0 sequence (n_timesteps = that many patch_len_list[0]-byte patches -- see
        config.py's TrainConfig docstring). B=1 (the whole file as one chunk) is the long-standing
        special case -- see train.py's make_file_chunks/TrainConfig's file_chunk_bytes docstring
        for the general B>1 case (real minibatch training over chunks of one file, not the whole
        file every step). Level 0 is a real causal AR transformer over ALL n_timesteps at once PER
        CHUNK -- teacher-forced, fully parallel, one causally-masked attention computation per
        chunk (see module docstring: no recurrence, no state carry; chunks are independent along
        the batch axis, ordinary batched attention already keeps them from attending to each
        other -- no extra masking needed). remat_time/remat_depth control jax.checkpoint
        granularity for level 0's forward (see Trunk.forward_remat) -- exact gradients either way,
        just a memory/recompute tradeoff. Levels 1+ recurse via _train_recurse (see its
        docstring): LOCAL attention within one parent patch's own children, processed in
        micro_batch-sized jax.lax.scan chunks (not one Bcur-sized batch -- Bcur reaches
        B*n_raw_bytes-per-chunk by the terminal level, which can overflow TPU HBM at real corpus
        scale) with remat_depth checkpointing applied at every level, not just level 0. Returned
        loss/metrics are AGGREGATE over the whole batch B (sum of nats / sum of positions) -- call
        with B=1 (a single chunk) to get that one chunk's own bpb, which is exactly what
        train.py's per-epoch full-pass evaluation does to report per-chunk numbers."""
        B, n_timesteps, P0 = byte_seq.shape

        # -- Level 0: full-sequence causal attention per chunk, teacher-forced, root_cond fed
        # unprojected and broadcast across the B independent chunks --
        proj0 = self.embed_patch(0, byte_seq)                                          # [B, n_timesteps, D0]
        root = jnp.broadcast_to(self.root_cond, (B, 1, self.root_cond.shape[-1]))      # [B, 1, D0]
        shifted0 = jnp.concatenate([root, proj0[:, :-1]], axis=1)                       # [B, n_timesteps, D0]
        h0 = self.trunk_at(0).forward_remat(shifted0, self.rope_base, remat_time, remat_depth,
                                             self.use_flash_attn[0])                     # [B, n_timesteps, D0]
        cond = h0.reshape(B * n_timesteps, -1)                                          # flatten (B,T)->rows for levels 1+

        # -- Levels 1+: LOCAL to one parent patch's children, chunked over B*n_timesteps parents --
        byte_seq_flat = byte_seq.reshape(B * n_timesteps, P0)
        valid0 = jnp.ones((B * n_timesteps,), dtype=jnp.bool_)
        if flat_scan:
            total_ce_nats, total_positions, total_correct = self._train_flat(
                cond, byte_seq_flat, valid0, remat_depth, micro_batch)
        else:
            total_ce_nats, total_positions, total_correct = self._train_recurse(
                1, cond, byte_seq_flat, valid0, remat_depth, micro_batch, use_scan)

        mean_loss = total_ce_nats / total_positions
        metrics = {
            "loss": mean_loss, "bpb": mean_loss / math.log(2),
            "byte_acc": total_correct / total_positions,
            "ce_nats": total_ce_nats, "positions": total_positions, "correct": total_correct,
        }
        return mean_loss, metrics

    def _gen_children(self, level: int, cond: jax.Array, symbol_fn) -> list[list[int]]:
        """LOCAL recursion for levels 1+: generate this level's seq_len sub-patches given the
        incoming (parent) condition, recursing into level+1 for each. Deliberately NOT using
        Trunk.step's KV-cache here (unlike level 0, see generate()) -- measured empirically (both
        locally and on real TPU runs) to be a WASH-TO-REGRESSION at this scale, not a win: unlike
        level 0 (where n_timesteps can be hundreds-to-thousands, so avoiding O(T^3) recompute
        matters enormously), seq_len here is small and FIXED (currently 8 everywhere) -- a
        KV-cache step still pays for attention over the full preallocated max_len every call
        (masked, not shrunk), so total cost stays the same O(seq_len^2) order as plain
        recompute-from-scratch, while adding real per-step overhead (dynamic_update_slice,
        masking) that isn't offset by any actual FLOP savings. Simple recompute (concatenate the
        real, actually-short-so-far sequence and rerun the trunk) is at least as fast and simpler.
        Bounded, small attention -- no remat/KV-cache machinery needed here, only level 0 spans a
        long, growing sequence.

        ITERATIVE, not Python-recursive: an explicit stack of frames stands in for the call stack
        (each frame = one in-progress "call" to this method, at some level, paused mid-loop while
        its child subtree is being generated). Produces the EXACT SAME sequence of
        _trunk_forward_jit/symbol_fn calls, in the EXACT SAME order, as the old recursive version
        -- this is a pure control-flow rewrite, not a behavior change (verified bit-exact against
        the prior recursive implementation). Done specifically so this can later be driven by
        jax.lax.scan/fori_loop/while_loop (which need flat, statically-bounded iteration, not
        Python call-stack recursion of varying depth) instead of a bare Python loop -- not yet
        done here, this step only removes the recursion as a blocker for that."""
        def make_frame(lvl: int, c: jax.Array) -> dict:
            b = c.shape[0]
            cond_l = self.cond_proj[lvl](c)                     # [b, D]
            return {
                "level": lvl,
                "seq_len": self.seq_lens[lvl],
                "child_len": self.cfg.patch_len_list[lvl],
                "seq_in": [cond_l[:, None, :]],                  # [b, 1, D], grows each step
                "i": 0,
                "out": [[] for _ in range(b)],
                "B": b,
            }

        stack = [make_frame(level, cond)]
        child_result: list[list[int]] | None = None   # set when a child frame just finished

        while stack:
            frame = stack[-1]

            if child_result is not None:
                # Resuming a frame that just recursed: fold the child's output in, advance i.
                for b in range(frame["B"]):
                    frame["out"][b].extend(child_result[b])
                if frame["i"] < frame["seq_len"] - 1:
                    child_t = jnp.array(child_result, dtype=jnp.int32)   # [B, child_len]
                    frame["seq_in"].append(self.embed_patch(frame["level"], child_t)[:, None, :])
                frame["i"] += 1
                child_result = None
                if frame["i"] >= frame["seq_len"]:
                    stack.pop()
                    child_result = frame["out"]
                    continue

            i = frame["i"]
            x = jnp.concatenate(frame["seq_in"], axis=1)
            h = _trunk_forward_jit(self.trunk_at(frame["level"]), x, self.rope_base)

            if frame["child_len"] == 1:
                logits = self.head(h[:, -1, :])                  # [B, 256]
                syms = symbol_fn(logits)                         # list[int], len B
                for b in range(frame["B"]):
                    frame["out"][b].append(syms[b])
                if i < frame["seq_len"] - 1:
                    sym_t = jnp.array(syms, dtype=jnp.int32)[:, None]   # [B, 1]
                    frame["seq_in"].append(self.embed_patch(frame["level"], sym_t)[:, None, :])
                frame["i"] += 1
                if frame["i"] >= frame["seq_len"]:
                    stack.pop()
                    child_result = frame["out"]
            else:
                cond_i = h[:, -1, :]                              # [B, D] CONTINUOUS, passed to child unchanged
                stack.append(make_frame(frame["level"] + 1, cond_i))

        return child_result

    def generate(self, n_timesteps: int, symbol_fn, batch_size: int = 1) -> list[list[int]]:
        """Generate n_timesteps level-0 patches (patch_len_list[0] bytes each) autoregressively
        -- level 0's causal attention genuinely spans every already-decoded timestep (see module
        docstring), not a fixed-size carried state.

        REAL KV-CACHE: see enfrac_zero/model.py's generate() docstring -- identical approach
        (Trunk.step with a preallocated fixed-size cache via _trunk_step_jit, MODULE level --
        see its docstring for a real bug found and fixed this session: a local closure here
        compiled once per generate() CALL, not once per process), HiRA-agnostic.

        batch_size>1: generate batch_size INDEPENDENT chains in LOCKSTEP (e.g. independent file
        chunks) -- see enfrac_zero/model.py's generate() docstring for the full rationale
        (_gen_children is already batch-generic, needed no changes; the dominant per-timestep
        cost is now paid once per timestep for the whole batch instead of once per chain).
        symbol_fn receives logits [batch_size, 256], returns batch_size symbols.

        Returns a list of batch_size flat byte-lists (each n_timesteps * patch_len_list[0] long,
        the last timestep possibly containing padding -- trimmed by the caller via n_raw_bytes)."""
        trunk0 = self.trunk_at(0)
        kv_cache = trunk0.init_kv_cache(n_timesteps, batch=batch_size)
        rope_base = self.rope_base

        x_t = jnp.broadcast_to(self.root_cond, (batch_size, 1, self.root_cond.shape[-1]))   # [B, 1, D0]
        all_bytes: list[list[int]] = [[] for _ in range(batch_size)]

        for t in tqdm(range(n_timesteps), desc="generate (level-0 timesteps)", unit="timestep"):
            h_t, kv_cache = _trunk_step_jit(trunk0, x_t, jnp.asarray(t, dtype=jnp.int32), rope_base, kv_cache)
            cond_t = h_t[:, -1, :]            # [B, D0] -- condition for THIS timestep's children

            child_bytes = self._gen_children(1, cond_t, symbol_fn)   # list of B lists, each len P0
            for b in range(batch_size):
                all_bytes[b].extend(child_bytes[b])

            if t < n_timesteps - 1:
                patch_batch = jnp.array(child_bytes, dtype=jnp.int32)          # [B, P0]
                proj_next = self.embed_patch(0, patch_batch)                    # [B, D0]
                x_t = proj_next[:, None, :]   # [B, 1, D0]
        return all_bytes

    def collect_logits(self, byte_seq: jax.Array) -> tuple[jax.Array, jax.Array]:
        """byte_seq: [n_timesteps, patch_len_list[0]]. Teacher-forced, routed through the SAME
        step-by-step generate() decompress.py uses (bit-exactness -- see module docstring).
        Returns (symbols[n_timesteps*P0], logits[n_timesteps*P0, 256])."""
        n_timesteps = byte_seq.shape[0]
        true_bytes = [int(v) for row in jax.device_get(byte_seq) for v in row]
        flat_symbols, flat_logits = [], []
        counter = [0]

        def symbol_fn(logits_batch):
            flat_logits.append(logits_batch[0])
            sym = true_bytes[counter[0]]
            flat_symbols.append(sym)
            counter[0] += 1
            return [sym]

        self.generate(n_timesteps, symbol_fn)
        return jnp.array(flat_symbols, dtype=jnp.int32), jnp.stack(flat_logits)

    def collect_logits_batched(self, byte_seq: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Batched-across-CHUNKS sibling of collect_logits() -- see enfrac_zero/model.py's
        identical method for the full rationale (why this is NOT the kind of batching
        collect_logits()'s own docstring warns is unsafe -- the sequential timestep loop inside
        generate() is untouched, only the independent-chunk batch axis grows). HiRA-agnostic.
        byte_seq: [B, n_timesteps, patch_len_list[0]]. Returns (symbols[B, n_timesteps*P0],
        logits[B, n_timesteps*P0, 256])."""
        B, n_timesteps, P0 = byte_seq.shape
        true_bytes = np.asarray(jax.device_get(byte_seq)).reshape(B, n_timesteps * P0)   # [B, T*P0]
        flat_logits_per_chain = [[] for _ in range(B)]
        pos_counter = [0]

        def symbol_fn(logits_batch):   # [B, 256]
            p = pos_counter[0]
            for b in range(B):
                flat_logits_per_chain[b].append(logits_batch[b])
            syms = [int(true_bytes[b, p]) for b in range(B)]
            pos_counter[0] += 1
            return syms

        self.generate(n_timesteps, symbol_fn, batch_size=B)
        logits = jnp.stack([jnp.stack(flat_logits_per_chain[b]) for b in range(B)])   # [B, T*P0, 256]
        symbols = jnp.asarray(true_bytes, dtype=jnp.int32)                              # [B, T*P0]
        return symbols, logits

    def cast_dtype(self, dtype) -> "ByteFractalGen":
        """Returns a copy of this model with every inexact (float) array leaf cast to `dtype`
        (e.g. jnp.float64) -- integers/static fields untouched. Used by collect_logits_fp64() and
        by decompress.py (which MUST cast to the SAME dtype recorded in meta.json at compress
        time -- dtype changes the actual computed logits, so it's a correctness invariant like
        device/batch_size, not a free-to-vary knob). Safe to call regardless of use_hira -- the
        frozen W0/A (HiRA) and byte_embed table are only VALUE-correct if they were constructed
        with dtype=float32 explicitly forced (see make_byte_embedding()/HiraLinear's docstrings)
        BEFORE this cast; this method only upcasts/downcasts precision, it never re-randomizes."""
        return jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, self)

    def _level0_cond_fast(self, byte_seq: jax.Array, max_step: int | None = None) -> jax.Array:
        """ENCODE-ONLY fast level-0 pass -- see enfrac_zero/model.py's identical method for the
        full docstring (chunked jax.lax.scan over Trunk.step; max_step=1 is the finest chunking,
        NOT a "provably safe" guarantee by itself -- see collect_logits_fp64()'s docstring for
        why float64 is what actually made this safe, verified empirically)."""
        n_timesteps = byte_seq.shape[0]
        step = n_timesteps if max_step is None else max(1, min(max_step, n_timesteps))
        trunk0 = self.trunk_at(0)
        rope_base = self.rope_base
        proj0 = self.embed_patch(0, byte_seq)                                 # [n_timesteps, D0]
        shifted0 = jnp.concatenate([self.root_cond, proj0[:-1]], axis=0)       # [n_timesteps, D0]
        kv_cache = trunk0.init_kv_cache(n_timesteps, batch=1)

        def scan_fn(kv_cache, elem):
            x_t, t = elem
            h_t, kv_cache = trunk0.step(x_t, t, rope_base, kv_cache)
            return kv_cache, h_t[:, -1, :]                                     # [1, D0]

        cond_chunks = []
        for start in range(0, n_timesteps, step):
            end = min(start + step, n_timesteps)
            xs = shifted0[start:end, None, None, :]                            # [chunk, 1, 1, D0]
            ts = jnp.arange(start, end, dtype=jnp.int32)
            kv_cache, cond_chunk = jax.lax.scan(scan_fn, kv_cache, (xs, ts))
            cond_chunks.append(cond_chunk[:, 0, :])
        return jnp.concatenate(cond_chunks, axis=0)                            # [n_timesteps, D0]

    def collect_logits_fp64(self, byte_seq: jax.Array, verify_prefix: int = 8) -> tuple[jax.Array, jax.Array]:
        """FAST encode-only path, made SAFE by running in float64 instead of float32 -- see
        enfrac_zero/model.py's identical method for the full rationale (batching ALL n_timesteps
        through _gen_children at once, verified empirically to diverge from collect_logits() by
        ~1e-6 in float32 -- enough to flip ~1.6-2.3% of quantized CDF bins -- but only ~1e-14 in
        float64, empirically 0 mismatches on both random-init and real trained models).

        THE CALLER MUST cast this model to float64 first (self.cast_dtype(jnp.float64)) AND that
        model must have been constructed with jax_enable_x64 OFF or its frozen W0/A/byte_embed
        built with explicit dtype=float32 (see HiraLinear's and make_byte_embedding()'s
        docstrings) -- otherwise the frozen base itself silently differs from what training used,
        which collapses accuracy independently of anything this method does (verified: byte_acc
        76% -> 25% from exactly this bug, now fixed at the init call sites, not here).

        Returns (symbols[n_timesteps*P0], logits[n_timesteps*P0, 256]), same flat order as
        collect_logits()."""
        n_timesteps = byte_seq.shape[0]
        P0 = self.cfg.patch_len_list[0]

        cond_all = self._level0_cond_fast(byte_seq, max_step=1)          # [n_timesteps, D0]
        true_bytes = np.asarray(jax.device_get(byte_seq))                 # [n_timesteps, P0]

        flat_logits_per_chain = [[] for _ in range(n_timesteps)]
        pos_counter = [0]

        def symbol_fn(logits_batch):
            p = pos_counter[0]
            for b in range(n_timesteps):
                flat_logits_per_chain[b].append(logits_batch[b])
            syms = [int(true_bytes[b, p]) for b in range(n_timesteps)]
            pos_counter[0] += 1
            return syms

        self._gen_children(1, cond_all, symbol_fn)

        logits = jnp.stack([jnp.stack(flat_logits_per_chain[b]) for b in range(n_timesteps)])
        logits = logits.reshape(n_timesteps * P0, 256)
        symbols = byte_seq.reshape(-1)

        k = min(verify_prefix, n_timesteps)
        _, ref_logits = self.collect_logits(byte_seq[:k])
        fast_prefix = np.asarray(logits[: k * P0], dtype=np.float64)
        ref_prefix = np.asarray(ref_logits, dtype=np.float64)
        for i in range(fast_prefix.shape[0]):
            if not np.array_equal(quantize_cdf(fast_prefix[i]), quantize_cdf(ref_prefix[i])):
                raise RuntimeError(
                    f"collect_logits_fp64: fast/reference CDF mismatch at prefix position {i} -- "
                    f"the fp64 fast path is not safe for this model/data; fall back to "
                    f"collect_logits() instead."
                )

        return symbols, logits


TRAINABLE_LEAF_NAMES = {"B", "weight", "bias", "root_cond"}


def trainable_filter(model: ByteFractalGen):
    """Pytree of bools matching `model`'s structure -- True at every trainable array leaf (HiRA's
    B, plain linears' weight, RMSNorm's weight, root_cond), False everywhere else (frozen W0/A,
    byte_embed's table, and all non-array/static fields). Use with eqx.partition before every
    gradient step: `eqx.partition(model, trainable_filter(model))`."""
    def mark(path, leaf):
        if not eqx.is_inexact_array(leaf):
            return False
        key = path[-1]
        name = getattr(key, "name", None)
        return name in TRAINABLE_LEAF_NAMES
    return jax.tree_util.tree_map_with_path(mark, model)
