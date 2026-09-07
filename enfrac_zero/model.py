"""enfrac.baseline (JAX/Equinox port of overfitter/model.py): byte-level FractalAR
(arxiv.org/html/2502.17437v2), independent per-level weights by default -- the plain, no-adapter
architecture. Every linear here is a genuinely trainable PlainLinear (no HiRA frozen-base split)
-- unlike enfrac/model.py (the PEFT/main package), there's no determinism contract to preserve:
the whole model (all weights) is saved in the checkpoint, nothing is regenerated from a seed.

Pure math (RoPE, causal masking, scaled-dot-product attention, RMSNorm, the frozen byte
embedding table) is identical regardless of HiRA and is imported from enfrac.model rather than
duplicated -- only the linear layer and the top-level config/model classes differ.

LEVEL 0 IS A REAL, FULL-SEQUENCE CAUSAL AR TRANSFORMER -- this is what makes the model see the
whole file, not just one patch_len_list[0]-byte window. The reference FractalGen processes one
WHOLE image per fractal tree -- an image already fits inside one tree's top level. A file doesn't
fit inside one tree the same way, but the fix is NOT to restart level 0 at every "chunk" -- it's
to let level 0's own sequence BE the whole file's top-level patches, all of them, in one causal
attention computation (n_timesteps = ceil(n_raw_bytes / patch_len_list[0]); "timestep" means one
patch_len_list[0]-byte patch, not one byte -- see config.py's TrainConfig docstring). Timestep t
can attend, via ordinary causal self-attention, all the way back to every earlier timestep --
training is teacher-forced and FULLY PARALLEL across all n_timesteps at once (one causally-masked
attention call, exactly like training any GPT-style transformer on a long sequence). There is no
recurrence, no fixed-size state threaded between "chunks" -- an EARLIER version of this file did
exactly that (a single carried vector, cond_out -> next cond_in) and it was wrong: that's an
RNN-style fixed-size bottleneck, which caps what the model can ever represent about everything
before position t into one vector, no matter how much training. Real full attention has no such
cap. remat_time/remat_depth (see Trunk.forward_remat) make long sequences memory-feasible via
jax.checkpoint -- recompute during backward instead of storing, exact gradients, no truncation,
no detach anywhere (contrast with TBPTT's stop_gradient, which this deliberately does not use).

INDEXING CONVENTION (n_levels = len(patch_len_list), NOT len(patch_len_list)-1): level l's own
patch/token size is patch_len_list[l] directly, for every level including 0 -- there is no
implicit "level 0 = whole file, patch_len_list describes only levels 1+" offset. Level 0 has no
parent (root_cond, shape [1, d_model_list[0]], is fed directly as the first position of its
shifted sequence -- there is no cond_proj[0]); levels l>=1 have a FIXED local sequence length
seq_lens[l] = patch_len_list[l-1] // patch_len_list[l], one parent patch's own children, and DO
have cond_proj[l] (projects the parent's d_model_list[l-1] -> this level's d_model_list[l]). The
terminal level (patch_len_list[l]==1) is always the last level, n_levels-1, by construction (the
divisibility chain forces it). This generalizes to any n_levels >= 2.

patch_in[l] embeds level l's own patch_len_list[l] bytes -> d_model_list[l], uniformly for every
level (including 0) via embed_patch()/PATCH_IN SCHEMES below -- one of TWO interchangeable
mechanisms, chosen by cfg.patch_in_scheme and applied to every level the same way:

  - "linear": embed every byte via the frozen byte table, flatten all patch_len_list[l] byte
    embeddings, one dense PlainLinear(patch_len_list[l]*byte_embed_dim -> d_model_list[l]). Exact
    (no information loss -- byte order and content both fully visible to the projection), but its
    parameter count and flattened width scale linearly with patch_len_list[l] -- fine for small
    patches (the levels 1+ case, where patch_len_list[l] is usually <=64), but for a LARGE
    patch_len_list[0] (e.g. 262144 for a deep enwik8 config) this blows up to millions of input
    features and a correspondingly huge weight matrix. See PatchInLinear.
  - "mean_pool" (default): embed every byte via the frozen byte table, MEAN-POOL across the
    patch_len_list[l] positions to one byte_embed_dim-sized summary vector, then a small
    PlainLinear(byte_embed_dim -> d_model_list[l]) -- parameter count and compute are INDEPENDENT
    of patch_len_list[l]. This loses within-patch byte order in that summary token (it's a coarse
    "what's roughly in this patch" signal for level 0's/this level's own long-range attention
    token) -- but the actual bytes are never lost: they're still processed exactly, in full
    (with real order-sensitive attention), by the recursive levels *underneath* this one. Use
    this whenever patch_len_list[l] is too large for "linear" to be tractable -- which in
    practice means: always, unless a specific level's patch is small enough that exact
    (non-pooled) embedding is worth the extra params. See PatchInMeanPool.

Levels 1+ are otherwise unchanged: LOCAL causal attention within one level-0 timestep's own
children, batched independently across all n_timesteps parents (no cross-parent causality needed
there, so no remat machinery needed there either -- see _gen_children()).

Generation (compress.py/decompress.py) is autoregressive at level 0 too, by necessity -- see
generate()'s docstring for the current recompute-from-scratch approach and its known KV-cache
follow-up.

IMPORTANT for compress.py/decompress.py: collect_logits() and generate() MUST stay the same
code path -- see enfrac/model.py's module docstring for why (bit-exactness for range coding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from enfrac.codec import quantize_cdf
from enfrac.model import (
    ROPE_PRESETS,
    FrozenEmbedding,
    RMSNorm,
    apply_rope,
    causal_mask,
    make_byte_embedding,
    rope_cos_sin_for_positions,
    sdpa,
)


class PlainLinear(eqx.Module):
    weight: jax.Array

    def __init__(self, d_in: int, d_out: int, key: jax.Array):
        limit = 1.0 / math.sqrt(d_in)
        self.weight = jax.random.uniform(key, (d_out, d_in), minval=-limit, maxval=limit)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.weight.T


class PatchInLinear(eqx.Module):
    """"linear" patch_in scheme -- see module docstring. proj: PlainLinear(patch_len*Ebyte -> D).
    Params/compute scale with patch_len; only use for small patch_len."""
    proj: PlainLinear
    patch_len: int = eqx.field(static=True)
    byte_embed_dim: int = eqx.field(static=True)

    def __init__(self, patch_len: int, byte_embed_dim: int, d_out: int, key: jax.Array):
        self.patch_len = patch_len
        self.byte_embed_dim = byte_embed_dim
        self.proj = PlainLinear(patch_len * byte_embed_dim, d_out, key)

    def __call__(self, emb: jax.Array) -> jax.Array:
        # emb: [..., patch_len, Ebyte] -> flatten last two axes -> [..., patch_len*Ebyte]
        flat = emb.reshape(*emb.shape[:-2], self.patch_len * self.byte_embed_dim)
        return self.proj(flat)


class PatchInMeanPool(eqx.Module):
    """"mean_pool" patch_in scheme (default) -- see module docstring. Mean-pools byte embeddings
    across the patch_len axis (order lost in this summary token, but not in the underlying bytes
    -- see docstring), then a small PlainLinear(Ebyte -> D) independent of patch_len."""
    proj: PlainLinear

    def __init__(self, byte_embed_dim: int, d_out: int, key: jax.Array):
        self.proj = PlainLinear(byte_embed_dim, d_out, key)

    def __call__(self, emb: jax.Array) -> jax.Array:
        # emb: [..., patch_len, Ebyte] -> mean over patch_len -> [..., Ebyte]
        pooled = emb.mean(axis=-2)
        return self.proj(pooled)


class Attn(eqx.Module):
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    wq: PlainLinear
    wk: PlainLinear
    wv: PlainLinear
    out: PlainLinear

    def __init__(self, d_model: int, n_heads: int, key: jax.Array):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.wq = PlainLinear(d_model, d_model, k1)
        self.wk = PlainLinear(d_model, d_model, k2)
        self.wv = PlainLinear(d_model, d_model, k3)
        self.out = PlainLinear(d_model, d_model, k4)

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
        """One-token incremental step for a FIXED-SIZE, preallocated KV cache -- k_cache/v_cache:
        [1, H, max_len, hd] (max_len = n_timesteps, filled with zeros past position t). Writes
        this step's (rope-rotated) k/v into slot t via dynamic_update_slice (keeps array shapes
        CONSTANT across steps, so a single jax.jit trace of Trunk.step is reused for every t --
        the whole point of a real KV cache, vs recomputing full attention from scratch each step).
        x: [1, 1, D] (this step's single new token). t: traced scalar int32 (NOT a python int --
        passing a python int here would make jit re-trace/recompile at every step)."""
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
    gate: PlainLinear
    up: PlainLinear
    down: PlainLinear

    def __init__(self, d_model: int, mlp_mult: int, key: jax.Array):
        hidden = mlp_mult * d_model
        k1, k2, k3 = jax.random.split(key, 3)
        self.gate = PlainLinear(d_model, hidden, k1)
        self.up = PlainLinear(d_model, hidden, k2)
        self.down = PlainLinear(hidden, d_model, k3)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


class Block(eqx.Module):
    ln1: RMSNorm
    attn: Attn
    ln2: RMSNorm
    mlp: SwiGLU

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.ln1 = RMSNorm(d_model)
        self.attn = Attn(d_model, n_heads, k1)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult, k2)

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
    loop (levels 1+). The shapes there are BOUNDED (T=1..seq_len, seq_len is small and fixed --
    currently 8 everywhere), so jax.jit's shape-keyed compilation cache means each distinct T
    compiles ONCE (the first time it's seen, across the whole process) and every subsequent call
    with that same T -- of which there are millions over a real file -- just executes the cached
    executable instead of re-tracing/dispatching from scratch. Measured empirically: ~30x faster
    per call than the bare eager `trunk(x, rope_base)` at juz1-scale (d_model=48), even accounting
    for the varying T=1..8 pattern _gen_children actually produces (8 distinct cache entries, not
    1). This mirrors what generate()'s level-0 step_fn already does for the SAME reason -- the
    fix that was missing at levels 1+ (see _gen_children's docstring: adding a KV-cache there was
    tried and reverted as a regression, because the ALGORITHM wasn't the bottleneck, the eager
    DISPATCH overhead was; jit-caching the existing recompute-from-scratch shapes fixes that
    directly, matching how the reference PyTorch FractalGen implementation's own smallest/deepest
    generator (PixelLoss) also just recomputes a tiny sequence from scratch every step -- it
    doesn't use a KV-cache there either, it's just running on a framework/runtime where per-call
    dispatch is cheap by default (PyTorch eager + CUDA async kernel launch); this is JAX's way of
    getting the same property without switching frameworks: NOT bit-identical to plain eager
    (~1e-6 magnitude difference, same as any jit-vs-eager comparison in this codebase), but
    self-consistent -- both encode (collect_logits_fp64/collect_logits) and decode (generate())
    call _gen_children, so they see the SAME jit-wrapped computation, keeping range-coding
    correctness intact."""
    return trunk(x, rope_base)


class Trunk(eqx.Module):
    """One level's transformer: a Block stack + final norm. Instantiated once per level by
    default; shared_trunk uses a single instance called n_levels times (see trunk_at())."""
    head_dim: int = eqx.field(static=True)
    blocks: list
    ln_f: RMSNorm

    def __init__(self, d_model: int, n_heads: int, n_layers: int, mlp_mult: int, key: jax.Array):
        self.head_dim = d_model // n_heads
        keys = jax.random.split(key, n_layers)
        self.blocks = [Block(d_model, n_heads, mlp_mult, keys[i]) for i in range(n_layers)]
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
        zero KV cache, one (k, v) pair per block -- shape stays constant across every step() call,
        which is what lets a single jax.jit trace of step() be reused for a whole generation run
        instead of recompiling at every t (see step()'s docstring). `batch` is the number of
        INDEPENDENT parallel chains sharing this cache's shape -- 1 for level 0 (one file), or
        Bcur (all parents at that level, all their local caches are independent of each other but
        processed together) for levels 1+."""
        return [
            (jnp.zeros((batch, blk.attn.n_heads, max_len, blk.attn.head_dim)),
             jnp.zeros((batch, blk.attn.n_heads, max_len, blk.attn.head_dim)))
            for blk in self.blocks
        ]

    def step(self, x: jax.Array, t: jax.Array, rope_base: float, kv_cache: list):
        """Real incremental KV-cache decoding for ONE new level-0 timestep -- O(t) work at step
        t (attends the single new query against the t already-cached keys), O(n_timesteps^2)
        total over a full generation run, replacing the old recompute-from-scratch approach's
        O(n_timesteps^3) (which re-ran full causal attention over the whole growing sequence at
        EVERY step). x: [1, 1, D] (this step's single new token). t: traced scalar int32 (pass a
        jnp.array, not a python int, so the caller's jax.jit trace is reused across steps instead
        of retracing/recompiling every single one -- see generate())."""
        cos_t, sin_t = rope_cos_sin_for_positions(jnp.asarray(t)[None], self.head_dim, rope_base)
        new_cache = []
        for blk, (k_c, v_c) in zip(self.blocks, kv_cache):
            x, k_c, v_c = blk.step(x, cos_t, sin_t, t, k_c, v_c)
            new_cache.append((k_c, v_c))
        return self.ln_f(x), new_cache

    def forward_remat(self, x: jax.Array, rope_base: float, remat_time: str, remat_depth: str) -> jax.Array:
        """Level 0's full-sequence forward -- IDENTICAL math to __call__ (real causal attention
        over the whole sequence, no truncation, no detach), just computed under jax.checkpoint
        for memory. remat_depth groups consecutive layers under one checkpoint each (standard
        activation checkpointing). remat_time!="1.0" wraps the WHOLE layer stack (all depth
        groups) in one additional outer jax.checkpoint -- see class docstring in config.py's
        TrainConfig for why true sub-sequence-only chunking isn't implemented here."""
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
    share_trunk: bool = False                      # opt-in alias; requires matching dims
    patch_in_scheme: str = "mean_pool"             # "mean_pool" (default) or "linear" -- see
                                                    # module docstring; applied uniformly to every
                                                    # level (patch_len==1 levels are unaffected --
                                                    # both schemes reduce to the same plain
                                                    # byte-embedding projection there)
    seed: int = 0                                    # drives (trainable) init only -- fully saved


class ByteFractalGen(eqx.Module):
    cfg: "ModelConfig" = eqx.field(static=True)
    n_levels: int = eqx.field(static=True)
    seq_lens: tuple = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)
    byte_embed: FrozenEmbedding
    trunks: list
    patch_in: list
    cond_proj: list
    head: PlainLinear
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

        key = jax.random.PRNGKey(cfg.seed)
        key, k_trunks, k_patch_in, k_cond_proj, k_head, k_root = jax.random.split(key, 6)

        # Frozen, maximally-separated byte representation -- fixed data, never trained.
        self.byte_embed = FrozenEmbedding(make_byte_embedding(cfg.byte_embed_dim))

        if cfg.share_trunk:
            d0, l0, h0, m0 = cfg.d_model_list[0], cfg.n_layers_list[0], cfg.n_heads_list[0], cfg.mlp_mult_list[0]
            for l in range(self.n_levels):
                assert (cfg.d_model_list[l], cfg.n_layers_list[l], cfg.n_heads_list[l], cfg.mlp_mult_list[l]) \
                    == (d0, l0, h0, m0), \
                    (f"share_trunk=True requires identical dims at every level; level {l} "
                     f"({cfg.d_model_list[l]},{cfg.n_layers_list[l]},{cfg.n_heads_list[l]},{cfg.mlp_mult_list[l]}) "
                     f"!= level 0 ({d0},{l0},{h0},{m0})")
            # See enfrac/model.py's ByteFractalGen for why this is a single Trunk object called
            # n_levels times (trunk_at()), not `[shared] * n_levels` stored in the list -- JAX
            # pytree lists don't alias by identity the way PyTorch's ModuleList does.
            self.trunks = [Trunk(d0, h0, l0, m0, k_trunks)]
        else:
            level_keys = jax.random.split(k_trunks, self.n_levels)
            self.trunks = [
                Trunk(cfg.d_model_list[l], cfg.n_heads_list[l], cfg.n_layers_list[l], cfg.mlp_mult_list[l], level_keys[l])
                for l in range(self.n_levels)
            ]

        # patch_in[l] embeds level l's OWN patch_len_list[l] bytes -> d_model_list[l]. Every
        # level uses the same construction, uniformly -- see module docstring for the two schemes.
        patch_in_keys = jax.random.split(k_patch_in, self.n_levels)
        self.patch_in = []
        for l in range(self.n_levels):
            P = cfg.patch_len_list[l]
            d_out = cfg.d_model_list[l]
            if P == 1:
                # Both schemes reduce to the same thing when there's only one byte -- just embed
                # it directly, no flatten/pool needed.
                self.patch_in.append(PlainLinear(cfg.byte_embed_dim, d_out, patch_in_keys[l]))
            elif cfg.patch_in_scheme == "linear":
                self.patch_in.append(PatchInLinear(P, cfg.byte_embed_dim, d_out, patch_in_keys[l]))
            else:
                self.patch_in.append(PatchInMeanPool(cfg.byte_embed_dim, d_out, patch_in_keys[l]))

        # cond_proj[l] projects the PARENT's d_model_list[l-1] -> this level's d_model_list[l],
        # for l>=1 only -- level 0 has no parent (root_cond is fed directly, unprojected).
        cond_proj_keys = jax.random.split(k_cond_proj, max(self.n_levels - 1, 1))
        self.cond_proj = [None] + [
            PlainLinear(cfg.d_model_list[l - 1], cfg.d_model_list[l], cond_proj_keys[l - 1])
            for l in range(1, self.n_levels)
        ]

        self.head = PlainLinear(cfg.d_model_list[-1], 256, k_head)   # terminal level only
        self.root_cond = 0.02 * jax.random.normal(k_root, (1, cfg.d_model_list[0]))

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
        level-0 sequence (n_timesteps = that many patch_len_list[0]-byte patches; see config.py's
        TrainConfig docstring for the patch-vs-byte distinction). Level 0 is a real causal AR
        transformer over ALL n_timesteps at once -- teacher-forced, fully parallel, one
        causally-masked attention computation (see module docstring: no recurrence, no state
        carry). remat_time/remat_depth control jax.checkpoint granularity for level 0's forward
        only (see Trunk.forward_remat) -- exact gradients either way, just a memory/recompute
        tradeoff. Levels 1+ recurse exactly as before: LOCAL attention within one parent patch's
        own children, batched independently across all n_timesteps parents (no cross-parent
        causality needed there, so no remat machinery needed there either)."""
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
        """Generate n_timesteps level-0 patches (patch_len_list[0] bytes each -- n_timesteps
        patches, NOT n_timesteps bytes) autoregressively -- the whole point being that level 0's
        causal attention genuinely spans every already-decoded timestep (see module/class
        docstring), not a fixed-size carried state.

        REAL KV-CACHE: level 0 uses Trunk.step() with a preallocated, fixed-size cache (see
        init_kv_cache/step docstrings) -- each new timestep is O(t) (attends 1 query against t
        cached keys) instead of recomputing full causal attention over the whole growing
        sequence from scratch (O(t^2) per step, O(n_timesteps^3) total, the old approach). The
        step function is JIT-compiled ONCE (cache shape is constant across steps -- only the
        traced position `t` changes) and reused for the entire run.

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
        """DEFAULT, CORRECT implementation -- the one compress.py calls. byte_seq:
        [n_timesteps, patch_len_list[0]]. Teacher-forced by routing through generate() with a
        symbol_fn that just plays back the known true bytes instead of decoding them from a range
        coder -- this reuses the EXACT SAME Python-loop-driven Trunk.step calls generate() makes
        at decode time.

        Every "make this faster" idea tried this session (jax.lax.scan fusion across steps;
        batching ALL n_timesteps through _gen_children at once via B=n_timesteps instead of
        n_timesteps separate B=1 calls) was verified UNSAFE using quantize_cdf comparisons on a
        REALISTICALLY SIZED model (juz1-scale: d_model=48, 3 layers, 4 heads) -- both changed
        floating-point results relative to this method by ~1e-6, enough to flip quantized CDF
        bins at up to ~1.6% of positions. Critically, both had ALSO passed an initial check on a
        tiny toy model (3 levels, d_model=16) with EXACT (0.0 diff) match -- that match does not
        generalize; toy-scale verification is not sufficient evidence of safety here, real model
        scale is required. Since decode (generate()) can never use scan or larger batches (it
        doesn't know future bytes, so it's forced to go one real timestep at a time), any
        divergence here would silently desync the range coder -- wrong bytes with no error
        raised. This method is the only one currently proven safe; see collect_logits_fast() and
        _level0_cond_fast()/_level_logits_fast() for the (NOT SAFE, kept for reference only)
        experiments.
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
        time -- see collect_logits_fp64()'s docstring: dtype changes the actual computed logits,
        so it's a correctness invariant like device/batch_size, not a free-to-vary knob)."""
        return jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, self)

    def collect_logits_fp64(self, byte_seq: jax.Array, verify_prefix: int = 8) -> tuple[jax.Array, jax.Array]:
        """FAST encode-only path, made SAFE by running in float64 instead of float32.

        The batched-across-timesteps approach (_gen_children called ONCE with B=n_timesteps
        instead of n_timesteps separate B=1 calls -- see _gen_children's own docstring) is
        mathematically identical to collect_logits()'s per-chain loop, but in float32 the two
        differ by ~1e-6 due to XLA choosing different matmul/attention lowering for different
        batch sizes -- enough to flip ~1.6% of quantized CDF bins on a realistic model (verified
        empirically this session). In float64, that same divergence drops to ~1e-14 (matches the
        ~1e9x jump in mantissa precision, 23 bits -> 52 bits) -- verified EMPIRICALLY to produce
        ZERO CDF mismatches (0/44,544 positions on a real trained juz1-scale model, 0/10,240 on a
        random-init one). Not a mathematical proof that mismatches are IMPOSSIBLE (floating point
        is fundamentally non-associative regardless of precision), just empirically far below the
        1/65536 CDF quantization granularity -- hence verify_prefix: before trusting the fast
        result for the whole file, this method ALSO runs the slow, definitely-safe
        collect_logits() reference on just the first `verify_prefix` timesteps and checks their
        quantized CDFs match exactly; if not, raises rather than silently risking a desync
        (cheap: constant cost, independent of file size).

        THE CALLER MUST cast this model to float64 first (self.cast_dtype(jnp.float64)) -- this
        method does not do it implicitly, so the SAME cast model object is available for
        decompress.py to mirror (dtype is a correctness invariant, see cast_dtype()'s docstring
        and compress.py/decompress.py's meta.json handling).

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

    def _level0_cond_fast(self, byte_seq: jax.Array, max_step: int | None = None) -> jax.Array:
        """ENCODE-ONLY fast level-0 pass: byte_seq is entirely known upfront (teacher forcing),
        so unlike generate() there is no host callback to wait on -- every step's input is
        already computable, which is exactly what jax.lax.scan needs. Uses the SAME Trunk.step
        (KV-cache) per-position FORMULA as generate() -- but a multi-step scan is NOT guaranteed
        bit-identical to N eager calls (XLA can fuse/lower a scanned loop differently -- see
        collect_logits_fast()'s docstring for the empirically-measured mismatch this caused).

        max_step bounds how many timesteps get fused into one lax.scan call: the sequence is
        split into ceil(n_timesteps/max_step) chunks, each its own scan, continuing the SAME
        kv_cache across chunk boundaries (chunking doesn't change the cache contents or the
        causal math, only how many steps XLA compiles together at once). max_step=1 makes every
        chunk a length-1 scan -- nothing to fuse across, so it degenerates to exactly the same
        per-step computation as an eager Trunk.step call, which is why it's the PROVABLY SAFE
        special case (see find_safe_max_step()). max_step=None (default) uses one chunk spanning
        the whole sequence -- the original (fast, unverified) behavior.

        Returns cond: [n_timesteps, D0] (same semantics as generate()'s per-step cond_t, stacked)."""
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

    def _level_logits_fast(self, level: int, cond: jax.Array, cur_bytes: jax.Array, max_step: int | None = None):
        """ENCODE-ONLY fast pass for levels 1+: cur_bytes ([Bcur, seq_len, child_len]) is fully
        known, so -- exactly like _level0_cond_fast -- the whole local sequence can be driven by
        jax.lax.scan over the SAME Trunk.step KV-cache math _gen_children uses, batched across
        ALL Bcur parents in one scan (Attn.step is already batch-generic, no vmap needed): each
        scan iteration processes every parent's i-th child simultaneously. This replaces
        (Bcur * seq_len) individually-dispatched eager calls with ceil(seq_len/max_step) jitted
        scan chunks, independent of Bcur -- the actual fix for "seqlen huge with nesting": the
        old recompute-from-scratch design (and even a naive KV-cache-in-a-python-loop) still cost
        one dispatch per (parent, position) pair, i.e. proportional to total file bytes; this
        costs a small constant number of jitted scan chunks no matter how large Bcur (and
        therefore the file) is. See _level0_cond_fast()'s docstring for what max_step controls
        and why max_step=1 is the provably-safe special case.

        Returns (logits, targets) if child_len==1 (terminal), else cond_next [Bcur, seq_len, D]."""
        Bcur = cond.shape[0]
        seq_len = self.seq_lens[level]
        step = seq_len if max_step is None else max(1, min(max_step, seq_len))
        child_len = self.cfg.patch_len_list[level]
        trunk = self.trunk_at(level)
        rope_base = self.rope_base
        cond_l = self.cond_proj[level](cond)                                   # [Bcur, D]

        proj = self.embed_patch(level, cur_bytes)                               # [Bcur, seq_len, D]
        shifted = jnp.concatenate([cond_l[:, None, :], proj[:, :-1]], axis=1)    # [Bcur, seq_len, D]
        kv_cache = trunk.init_kv_cache(seq_len, batch=Bcur)

        def scan_fn(kv_cache, elem):
            x_t, t = elem
            h_t, kv_cache = trunk.step(x_t, t, rope_base, kv_cache)
            return kv_cache, h_t[:, -1, :]                                       # [Bcur, D]

        h_chunks = []
        for start in range(0, seq_len, step):
            end = min(start + step, seq_len)
            xs = shifted[:, start:end, :].transpose(1, 0, 2)[:, :, None, :]       # [chunk, Bcur, 1, D]
            ts = jnp.arange(start, end, dtype=jnp.int32)
            kv_cache, h_chunk = jax.lax.scan(scan_fn, kv_cache, (xs, ts))          # [chunk, Bcur, D]
            h_chunks.append(h_chunk)
        h = jnp.concatenate(h_chunks, axis=0).transpose(1, 0, 2)                    # [Bcur, seq_len, D]

        if child_len == 1:
            logits = self.head(h)                                                  # [Bcur, seq_len, 256]
            return logits, cur_bytes[..., 0]
        return h   # cond_next, CONTINUOUS -- no softmax at non-terminal levels

    def collect_logits_fast(self, byte_seq: jax.Array, max_step: int | None = None) -> tuple[jax.Array, jax.Array]:
        """NOT SAFE TO RANGE-CODE WITH AT max_step=None (or any unverified value) -- experimental,
        NOT called by compress.py by default. Use find_safe_max_step() to discover a max_step
        value proven (on a calibration sample) to match collect_logits()'s CDFs exactly before
        trusting this for a real bundle; max_step=1 is safe by construction (see
        _level0_cond_fast()'s docstring) but gives no speedup over collect_logits().

        byte_seq is entirely known upfront (this IS teacher forcing), so in principle there's no
        reason to drive generate()'s Python-loop + host-callback machinery byte by byte (that
        machinery exists for DECODE, where the next byte genuinely isn't known until the range
        coder produces it). This method reuses the exact same Trunk.step per-position FORMULA as
        generate()/_gen_children, driven by jax.lax.scan in chunks of at most max_step steps
        (jitted, batched across all parents at each level) instead of eager per-byte dispatch --
        but "same formula" is NOT "same floating-point result" once max_step > 1: empirically,
        quantize_cdf(these logits) differed from quantize_cdf(collect_logits()'s logits) at ~1%
        of positions with max_step=None on an undertrained model (max abs logit diff ~2e-6,
        enough to flip a quantization bin at some positions) -- XLA compiles/fuses a multi-step
        scan differently than N separately-dispatched calls to the identical function, even
        though the math is the same. Since decode (generate()) can NEVER use a multi-step scan
        (the next byte depends on host-side range-coder feedback, a real data dependency scan
        can't trace through), any CDF mismatch here would silently desync decode -- wrong bytes
        from that point on, no error raised.

        byte_seq: [n_timesteps, patch_len_list[0]]. Returns (symbols[n_timesteps*P0],
        logits[n_timesteps*P0, 256]), in the same flat file-byte order collect_logits() produces
        (same row-major reshape structure as __call__'s teacher-forced training pass)."""
        cond = self._level0_cond_fast(byte_seq, max_step)                        # [n_timesteps, D0]
        cur_bytes = byte_seq

        for l in range(1, self.n_levels):
            seq_len = self.seq_lens[l]
            child_len = self.cfg.patch_len_list[l]
            cur_bytes_r = cur_bytes.reshape(-1, seq_len, child_len)
            Bcur = cur_bytes_r.shape[0]

            if child_len == 1:
                logits, targets = self._level_logits_fast(l, cond, cur_bytes_r, max_step)
                return targets.reshape(-1), logits.reshape(-1, 256)

            cond_next = self._level_logits_fast(l, cond, cur_bytes_r, max_step)   # [Bcur, seq_len, D]
            D = cond_next.shape[-1]
            cur_bytes = cur_bytes_r.reshape(Bcur * seq_len, child_len)
            cond = cond_next.reshape(Bcur * seq_len, D)

        raise AssertionError("no terminal (patch_len==1) level found past level 0")

    def find_safe_max_step(self, byte_seq_calib: jax.Array, start: int | None = None,
                            min_step: int = 1) -> int:
        """Exponential-backoff calibration: find the largest max_step for which
        collect_logits_fast()'s quantized CDFs exactly match collect_logits()'s (the reference,
        safe-by-construction implementation) on byte_seq_calib -- run this ONCE on a small
        representative sample (e.g. a prefix of the real file) to pick a max_step to trust for
        the full, much larger file's real collect_logits_fast() call, without paying the
        reference's full cost again. Starts at `start` (default: byte_seq_calib's own
        n_timesteps, i.e. try one giant chunk first) and halves on any mismatch down to
        `min_step` (default 1, which never needs testing -- it's safe by construction, see
        _level0_cond_fast()'s docstring -- so this never returns less than that, and always
        terminates)."""
        from enfrac.codec import quantize_cdf
        import numpy as np
        n_timesteps = byte_seq_calib.shape[0]
        step = n_timesteps if start is None else max(min_step, min(start, n_timesteps))
        if step <= min_step:
            return min_step

        _, ref_logits = self.collect_logits(byte_seq_calib)
        ref_np = np.asarray(ref_logits, dtype=np.float32)
        ref_cdfs = [quantize_cdf(ref_np[i]) for i in range(ref_np.shape[0])]

        while step > min_step:
            _, fast_logits = self.collect_logits_fast(byte_seq_calib, max_step=step)
            fast_np = np.asarray(fast_logits, dtype=np.float32)
            ok = all(np.array_equal(quantize_cdf(fast_np[i]), ref_cdfs[i]) for i in range(fast_np.shape[0]))
            if ok:
                return step
            step = max(min_step, step // 2)
        return min_step


TRAINABLE_LEAF_NAMES = {"weight", "bias", "root_cond"}


def trainable_filter(model: ByteFractalGen):
    """Pytree of bools matching `model`'s structure -- True at every trainable array leaf
    (all PlainLinear.weight, RMSNorm.weight, root_cond), False at byte_embed's frozen table."""
    def mark(path, leaf):
        if not eqx.is_inexact_array(leaf):
            return False
        key = path[-1]
        name = getattr(key, "name", None)
        return name in TRAINABLE_LEAF_NAMES
    return jax.tree_util.tree_map_with_path(mark, model)
