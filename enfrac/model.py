"""enfrac (JAX/Equinox port of overfitter_peft): byte-level FractalAR
(arxiv.org/html/2502.17437v2), same recursive architecture as enfrac/baseline/model.py, but every
Linear defaults to a HiRA-adapted frozen-base layer instead of a plain trainable linear -- see
randcompress's (archived, see archive/randcompress/models/hira.py) original HiRA design this is
ported from (W = W0 + W0*(B@A), only B trainable, W0/A regenerated deterministically from
cfg.seed, never saved).

Fork rationale (unchanged from the PyTorch overfitter_peft/ this was ported from): trades the
baseline's "small model, everything trained directly" tradeoff for "arbitrarily large frozen
base, tiny trainable adapter" -- d_model/mlp_mult/byte_embed_dim become nearly free to grow
(O(d) compute, O(r) storage per layer) instead of directly taxing the compressed bundle size.
use_hira=True is the default (set use_hira=False on ModelConfig to fall back to plain linears,
matching enfrac/baseline/'s architecture exactly, modulo the JAX/Equinox backend).

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
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp

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
    """A: [r, d_in] orthonormal rows (right singular vectors of a random matrix) -- frozen."""
    raw = jax.random.normal(key, (r, d_in))
    if r <= d_in:
        _, _, Vt = jnp.linalg.svd(raw, full_matrices=False)
        A = Vt[:r] / math.sqrt(d_in)
    else:
        A = raw / math.sqrt(d_in)
    return A


class HiraLinear(eqx.Module):
    """W = W0 + W0*(B@A) (HiRA) -- W0 [d_out,d_in] and A [r,d_in] frozen (drawn from `keyseq` at
    construction, never saved -- see module docstring's determinism contract), only B [d_out,r]
    is trainable (zero-init, so the layer starts as an identity pass-through of W0)."""
    W0: jax.Array
    A: jax.Array
    B: jax.Array

    def __init__(self, d_in: int, d_out: int, r: int, keyseq: KeySeq):
        orthogonal = jax.nn.initializers.orthogonal(scale=1.0)
        self.W0 = orthogonal(keyseq.next(), (d_out, d_in)) / math.sqrt(d_in)
        self.A = init_hira_A(d_out, d_in, r, keyseq.next())
        self.B = jnp.zeros((d_out, r))

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


@dataclass(frozen=True)
class ModelConfig:
    patch_len_list: tuple = (1024, 128, 16, 1)   # last element must be 1 (byte-atomic)
    d_model_list: tuple = (64, 64, 64)           # length = n_levels = len(patch_len_list)-1
    n_layers_list: tuple = (4, 4, 4)
    n_heads_list: tuple = (4, 4, 4)
    mlp_mult_list: tuple = (2, 2, 2)
    byte_embed_dim: int = 256                     # decoupled from any level's d_model
    rope_preset: str = "qwen3"
    share_trunk: bool = False                     # opt-in alias; requires matching dims
    use_hira: bool = True                          # main-package default; False = plain linear
    hira_r: int = 4                                 # HiRA rank -- only used when use_hira=True
    seed: int = 0                                    # drives frozen W0/A -- must match at load time


def make_byte_embedding(dim: int) -> jax.Array:
    """Fixed, MAXIMALLY separated per-byte representation -- exact one-hot (mutually orthogonal)
    when dim>=256; fixed random unit vectors otherwise. Never trained -- see module docstring."""
    if dim >= 256:
        w = jnp.zeros((256, dim))
        w = w.at[:, :256].set(jnp.eye(256))
    else:
        g = jax.random.PRNGKey(0)
        w = jax.random.normal(g, (256, dim))
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
        self.cfg = cfg
        self.n_levels = len(cfg.patch_len_list) - 1
        self.seq_lens = tuple(cfg.patch_len_list[l] // cfg.patch_len_list[l + 1] for l in range(self.n_levels))
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

        # patch_in[l]: flatten this level's child bytes (via the shared byte_embed) and project
        # into level l's own working dimension. Uniform formula for intermediate AND terminal
        # levels (child_len=1 there, just a per-token projection, not really a "patch").
        self.patch_in = [
            make_linear(cfg.patch_len_list[l + 1] * cfg.byte_embed_dim, cfg.d_model_list[l], cfg, keyseq)
            for l in range(self.n_levels)
        ]
        # cond_proj[l]: adapt the incoming condition (root_cond for l=0, else parent's d_model)
        # into level l's own working dimension. Always a real learned adapter.
        self.cond_proj = [
            make_linear(cfg.d_model_list[0] if l == 0 else cfg.d_model_list[l - 1], cfg.d_model_list[l], cfg, keyseq)
            for l in range(self.n_levels)
        ]

        self.head = make_linear(cfg.d_model_list[-1], 256, cfg, keyseq)   # terminal level only
        self.root_cond = 0.02 * jax.random.normal(keyseq.next(), (1, cfg.d_model_list[0]))

    def trunk_at(self, level: int) -> "Trunk":
        return self.trunks[0] if self.cfg.share_trunk else self.trunks[level]

    def __call__(self, byte_chunk: jax.Array) -> tuple[jax.Array, dict]:
        """byte_chunk: [B, patch_len_list[0]] int32. Fast batched path -- training loss only,
        never for range-coding CDFs (see collect_logits)."""
        B = byte_chunk.shape[0]
        cond = jnp.broadcast_to(self.root_cond, (B, self.root_cond.shape[-1]))
        cur_bytes = byte_chunk
        total_ce_nats = jnp.zeros(())
        total_positions = 0
        total_correct = jnp.zeros(())

        for l in range(self.n_levels):
            seq_len = self.seq_lens[l]
            child_len = self.cfg.patch_len_list[l + 1]
            cur_bytes = cur_bytes.reshape(-1, seq_len, child_len)
            Bcur = cur_bytes.shape[0]
            D = self.cfg.d_model_list[l]

            cond_l = self.cond_proj[l](cond)
            patch_bytes_emb = self.byte_embed(cur_bytes)                        # [Bcur, seq_len, child_len, Ebyte]
            flat = patch_bytes_emb.reshape(Bcur, seq_len, child_len * self.cfg.byte_embed_dim)
            proj = self.patch_in[l](flat)                                       # [Bcur, seq_len, D]

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

    def generate(self, symbol_fn, batch_size: int = 1) -> list[list[int]]:
        """Generate batch_size independent patch_len_list[0]-byte chunks IN PARALLEL. See
        overfitter_peft/model.py's docstring -- ported verbatim, only the tensor backend differs.

        symbol_fn(logits[batch_size, 256]) -> list[int] of length batch_size."""

        def gen_patch(level: int, cond: jax.Array) -> list[list[int]]:
            B = cond.shape[0]
            seq_len = self.seq_lens[level]
            child_len = self.cfg.patch_len_list[level + 1]
            cond_l = self.cond_proj[level](cond)                  # [B, D]
            out = [[] for _ in range(B)]
            seq_in = [cond_l[:, None, :]]                          # [B, 1, D]
            if child_len == 1:
                for i in range(seq_len):
                    x = jnp.concatenate(seq_in, axis=1)
                    h = self.trunk_at(level)(x, self.rope_base)
                    logits = self.head(h[:, -1, :])                 # [B, 256]
                    syms = symbol_fn(logits)                        # list[int], len B
                    for b in range(B):
                        out[b].append(syms[b])
                    if i < seq_len - 1:
                        sym_t = jnp.array(syms, dtype=jnp.int32)[:, None]   # [B, 1]
                        emb = self.byte_embed(sym_t)                 # [B, 1, Ebyte]
                        seq_in.append(self.patch_in[level](emb))
                return out
            else:
                for i in range(seq_len):
                    x = jnp.concatenate(seq_in, axis=1)
                    h = self.trunk_at(level)(x, self.rope_base)
                    cond_i = h[:, -1, :]                            # [B, D] CONTINUOUS, passed to child unchanged
                    child_bytes = gen_patch(level + 1, cond_i)      # list of B lists, len child_len
                    for b in range(B):
                        out[b].extend(child_bytes[b])
                    if i < seq_len - 1:
                        child_t = jnp.array(child_bytes, dtype=jnp.int32)   # [B, child_len]
                        child_emb = self.byte_embed(child_t).reshape(B, 1, child_len * self.cfg.byte_embed_dim)
                        seq_in.append(self.patch_in[level](child_emb))
                return out

        root = jnp.broadcast_to(self.root_cond, (batch_size, self.root_cond.shape[-1]))
        return gen_patch(0, root)

    def collect_logits(self, byte_chunks: jax.Array) -> tuple[jax.Array, jax.Array]:
        """byte_chunks: [B, P0]. Teacher-forced, routed through the SAME batched step-by-step
        generate() decompress.py uses (bit-exactness -- see module docstring). Returns FLAT
        (symbols[B*P0], logits[B*P0, 256]) in the exact INTERLEAVED order generate()'s internal
        symbol_fn calls happen in."""
        B = byte_chunks.shape[0]
        true_bytes = [[int(v) for v in row] for row in jax.device_get(byte_chunks)]
        flat_symbols, flat_logits = [], []
        counters = [0] * B

        def symbol_fn(logits_batch):
            syms = []
            for b in range(B):
                flat_logits.append(logits_batch[b])
                sym = true_bytes[b][counters[b]]
                flat_symbols.append(sym)
                syms.append(sym)
                counters[b] += 1
            return syms

        self.generate(symbol_fn, batch_size=B)
        return jnp.array(flat_symbols, dtype=jnp.int32), jnp.stack(flat_logits)


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
