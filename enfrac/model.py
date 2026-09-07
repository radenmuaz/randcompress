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
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .codec import quantize_cdf

ROPE_PRESETS = {"llama2": 10000.0, "llama3": 500000.0, "qwen3": 1000000.0}


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

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, mask: jax.Array) -> jax.Array:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        q = self.wq(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = sdpa(q, k, v, mask)
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

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, mask: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln1(x), cos, sin, mask)
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

    def __call__(self, x: jax.Array, rope_base: float) -> jax.Array:
        T = x.shape[1]
        pos = jnp.arange(T)
        cos, sin = rope_cos_sin_for_positions(pos, self.head_dim, rope_base)
        mask = causal_mask(T)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        return self.ln_f(x)

    def init_kv_cache(self, max_len: int, batch: int = 1):
        """Preallocated, FIXED-SIZE (max_len = that level's seq_len, or n_timesteps for level 0)
        zero KV cache, one (k, v) pair per block -- see enfrac_zero/model.py's
        Trunk.init_kv_cache docstring (identical, HiRA-agnostic). `batch` is the number of
        independent parallel local sequences sharing this cache's shape -- 1 for level 0, or Bcur
        (all parents at that level) for levels 1+."""
        return [
            (jnp.zeros((batch, blk.attn.n_heads, max_len, blk.attn.head_dim)),
             jnp.zeros((batch, blk.attn.n_heads, max_len, blk.attn.head_dim)))
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

    def forward_remat(self, x: jax.Array, rope_base: float, remat_time: str, remat_depth: str) -> jax.Array:
        """Level 0's full-sequence forward -- IDENTICAL math to __call__ (real causal attention
        over the whole sequence, no truncation, no detach), just computed under jax.checkpoint
        for memory. See enfrac_zero/model.py's Trunk.forward_remat -- identical logic, HiRA-
        agnostic (operates only on the Block list)."""
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
                        x = blk(x, cos, sin, mask)
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
    patch_in_scheme: str = "mean_pool"             # "mean_pool" (default) or "linear" -- see
                                                    # module docstring; applied uniformly to every
                                                    # level (patch_len==1 levels are unaffected)
    use_hira: bool = True                          # main-package default; False = plain linear
    hira_r: int = 4                                 # HiRA rank -- only used when use_hira=True
    seed: int = 0                                    # drives frozen W0/A -- must match at load time


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
        assert cfg.patch_in_scheme in ("mean_pool", "linear"), \
            f"patch_in_scheme must be 'mean_pool' or 'linear', got {cfg.patch_in_scheme!r}"
        self.cfg = cfg
        self.n_levels = len(cfg.patch_len_list)
        self.seq_lens = (0,) + tuple(
            cfg.patch_len_list[l - 1] // cfg.patch_len_list[l] for l in range(1, self.n_levels))
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
            elif cfg.patch_in_scheme == "linear":
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
        the exact same embedding math (needed for collect_logits()/generate() bit-exactness)."""
        P = self.cfg.patch_len_list[level]
        if P == 1:
            emb = self.byte_embed(byte_patch[..., 0])          # [..., Ebyte]
        else:
            emb = self.byte_embed(byte_patch)                   # [..., P, Ebyte]
        return self.patch_in[level](emb)

    def __call__(self, byte_seq: jax.Array, remat_time: str = "1.0", remat_depth: str = "1.0"
                 ) -> tuple[jax.Array, dict]:
        """byte_seq: [n_timesteps, patch_len_list[0]] int32 -- the WHOLE training window's
        level-0 sequence (n_timesteps = that many patch_len_list[0]-byte patches -- see
        config.py's TrainConfig docstring). Level 0 is a real causal AR transformer over ALL
        n_timesteps at once -- teacher-forced, fully parallel, one causally-masked attention
        computation (see module docstring: no recurrence, no state carry). remat_time/remat_depth
        control jax.checkpoint granularity for level 0's forward only -- exact gradients either
        way, just a memory/recompute tradeoff. Levels 1+ recurse exactly as before: LOCAL
        attention within one parent patch's own children, batched independently across all
        n_timesteps parents."""
        n_timesteps = byte_seq.shape[0]

        total_ce_nats = jnp.zeros(())
        total_positions = 0
        total_correct = jnp.zeros(())

        # -- Level 0: full-sequence causal attention, teacher-forced, root_cond fed unprojected --
        proj0 = self.embed_patch(0, byte_seq)                                        # [n_timesteps, D0]
        shifted0 = jnp.concatenate([self.root_cond, proj0[:-1]], axis=0)[None, :, :]  # [1, n_timesteps, D0]
        h0 = self.trunk_at(0).forward_remat(shifted0, self.rope_base, remat_time, remat_depth)
        cond = h0[0]                                                                   # [n_timesteps, D0]

        cur_bytes = byte_seq   # [n_timesteps, P0] -- level-0 timesteps' own bytes, recursed into levels 1+

        # -- Levels 1+: LOCAL to one parent patch's children, batched over n_timesteps parents --
        for l in range(1, self.n_levels):
            seq_len = self.seq_lens[l]
            child_len = self.cfg.patch_len_list[l]
            cur_bytes = cur_bytes.reshape(-1, seq_len, child_len)
            Bcur = cur_bytes.shape[0]
            D = self.cfg.d_model_list[l]

            cond_l = self.cond_proj[l](cond)
            proj = self.embed_patch(l, cur_bytes)                                  # [Bcur, seq_len, D]

            if child_len == 1:
                shifted = jnp.concatenate([cond_l[:, None, :], proj[:, :-1]], axis=1)
                h = self.trunk_at(l)(shifted, self.rope_base)
                logits = self.head(h)                                           # [Bcur, seq_len, 256]
                targets = cur_bytes[..., 0]
                logp = jax.nn.log_softmax(logits, axis=-1)
                nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
                total_ce_nats = total_ce_nats + nll.sum()
                total_positions += targets.size
                total_correct = total_correct + (logits.argmax(-1) == targets).sum()
            else:
                seq = jnp.concatenate([cond_l[:, None, :], proj], axis=1)
                h = self.trunk_at(l)(seq, self.rope_base)
                cond_next = h[:, :-1, :]                     # CONTINUOUS -- no softmax at non-terminal levels
                cur_bytes = cur_bytes.reshape(Bcur * seq_len, child_len)
                cond = cond_next.reshape(Bcur * seq_len, D)

        mean_loss = total_ce_nats / total_positions
        metrics = {
            "loss": mean_loss, "bpb": mean_loss / math.log(2),
            "byte_acc": total_correct / total_positions,
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
        long, growing sequence."""
        B = cond.shape[0]
        seq_len = self.seq_lens[level]
        child_len = self.cfg.patch_len_list[level]
        cond_l = self.cond_proj[level](cond)                  # [B, D]
        out = [[] for _ in range(B)]
        seq_in = [cond_l[:, None, :]]                          # [B, 1, D]
        if child_len == 1:
            for i in range(seq_len):
                x = jnp.concatenate(seq_in, axis=1)
                h = _trunk_forward_jit(self.trunk_at(level), x, self.rope_base)
                logits = self.head(h[:, -1, :])                 # [B, 256]
                syms = symbol_fn(logits)                        # list[int], len B
                for b in range(B):
                    out[b].append(syms[b])
                if i < seq_len - 1:
                    sym_t = jnp.array(syms, dtype=jnp.int32)[:, None]   # [B, 1]
                    seq_in.append(self.embed_patch(level, sym_t)[:, None, :])
        else:
            for i in range(seq_len):
                x = jnp.concatenate(seq_in, axis=1)
                h = _trunk_forward_jit(self.trunk_at(level), x, self.rope_base)
                cond_i = h[:, -1, :]                            # [B, D] CONTINUOUS, passed to child unchanged
                child_bytes = self._gen_children(level + 1, cond_i, symbol_fn)   # list of B lists, len child_len
                for b in range(B):
                    out[b].extend(child_bytes[b])
                if i < seq_len - 1:
                    child_t = jnp.array(child_bytes, dtype=jnp.int32)   # [B, child_len]
                    seq_in.append(self.embed_patch(level, child_t)[:, None, :])
        return out

    def generate(self, n_timesteps: int, symbol_fn) -> list[int]:
        """Generate n_timesteps level-0 patches (patch_len_list[0] bytes each) autoregressively
        -- level 0's causal attention genuinely spans every already-decoded timestep (see module
        docstring), not a fixed-size carried state.

        REAL KV-CACHE: see enfrac_zero/model.py's generate() docstring -- identical approach
        (Trunk.step with a preallocated fixed-size cache, JIT-compiled once and reused for every
        t), HiRA-agnostic.

        Returns the flat list of all decoded bytes (n_timesteps * patch_len_list[0], the last
        timestep possibly containing padding -- trimmed by the caller via n_raw_bytes)."""
        trunk0 = self.trunk_at(0)
        kv_cache = trunk0.init_kv_cache(n_timesteps)
        rope_base = self.rope_base

        @eqx.filter_jit
        def step_fn(x_t, t_arr, kv_cache):
            return trunk0.step(x_t, t_arr, rope_base, kv_cache)

        x_t = self.root_cond[None, :, :]      # [1, 1, D0]
        all_bytes: list[int] = []

        for t in range(n_timesteps):
            h_t, kv_cache = step_fn(x_t, jnp.asarray(t, dtype=jnp.int32), kv_cache)
            cond_t = h_t[:, -1, :]            # [1, D0] -- condition for THIS timestep's children

            child_bytes = self._gen_children(1, cond_t, symbol_fn)   # list of 1 list, len P0
            patch_bytes = child_bytes[0]
            all_bytes.extend(patch_bytes)

            if t < n_timesteps - 1:
                proj_next = self.embed_patch(0, jnp.asarray(patch_bytes, dtype=jnp.int32)[None, :])   # [1, D0]
                x_t = proj_next[:, None, :]   # [1, 1, D0]
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
