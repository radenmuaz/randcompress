"""enfrac.baseline (JAX/Equinox port of overfitter/model.py): byte-level FractalAR
(arxiv.org/html/2502.17437v2), independent per-level weights by default -- the plain, no-adapter
architecture. Every linear here is a genuinely trainable PlainLinear (no HiRA frozen-base split)
-- unlike enfrac/model.py (the PEFT/main package), there's no determinism contract to preserve:
the whole model (all weights) is saved in the checkpoint, nothing is regenerated from a seed.

Pure math (RoPE, causal masking, scaled-dot-product attention, RMSNorm, the frozen byte
embedding table) is identical regardless of HiRA and is imported from enfrac.model rather than
duplicated -- only the linear layer and the top-level config/model classes differ.

IMPORTANT for compress.py/decompress.py: collect_logits() and generate() MUST stay the same
code path -- see enfrac/model.py's module docstring for why (bit-exactness for range coding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

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


@dataclass(frozen=True)
class ModelConfig:
    patch_len_list: tuple = (1024, 128, 16, 1)   # last element must be 1 (byte-atomic)
    d_model_list: tuple = (64, 64, 64)           # length = n_levels = len(patch_len_list)-1
    n_layers_list: tuple = (4, 4, 4)
    n_heads_list: tuple = (4, 4, 4)
    mlp_mult_list: tuple = (2, 2, 2)
    byte_embed_dim: int = 256                     # decoupled from any level's d_model
    rope_preset: str = "qwen3"
    share_trunk: bool = False                      # opt-in alias; requires matching dims
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

        patch_in_keys = jax.random.split(k_patch_in, self.n_levels)
        self.patch_in = [
            PlainLinear(cfg.patch_len_list[l + 1] * cfg.byte_embed_dim, cfg.d_model_list[l], patch_in_keys[l])
            for l in range(self.n_levels)
        ]
        cond_proj_keys = jax.random.split(k_cond_proj, self.n_levels)
        self.cond_proj = [
            PlainLinear(cfg.d_model_list[0] if l == 0 else cfg.d_model_list[l - 1], cfg.d_model_list[l], cond_proj_keys[l])
            for l in range(self.n_levels)
        ]

        self.head = PlainLinear(cfg.d_model_list[-1], 256, k_head)   # terminal level only
        self.root_cond = 0.02 * jax.random.normal(k_root, (1, cfg.d_model_list[0]))

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
        """See enfrac/model.py's ByteFractalGen.generate() -- identical recursion, plain linears."""

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
        """See enfrac/model.py's ByteFractalGen.collect_logits() -- identical contract."""
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
