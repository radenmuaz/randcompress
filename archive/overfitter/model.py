"""ByteFractalGen: byte-level FractalAR (arxiv.org/html/2502.17437v2), independent per-level
weights by default -- see randcompress's fractalgen/ folder for the reference authors'
implementation this was ported from.

A generative model built by recursively invoking a generator on progressively smaller
sub-patches: a length-L sequence costs O(L) total attention, not O(L^2) for one flat pass,
because every level only attends over a short seq_len (patch_len_list[l] // patch_len_list[l+1]).
Training is teacher-forced and fully vectorized: each level's sub-patches become the BATCH
dimension for the next level's call (forward()/generate() below), so the whole multi-level
recursion is a Python loop over levels, not over samples or positions.

Adaptations for 1D discrete bytes instead of 2D continuous-pixel image patches:
  - "patchify" splits a byte sequence into equal sub-chunks, not a 2D image into square tiles.
  - The reference feeds RAW PIXEL FLOATS directly into every level's patchify step -- no
    embedding at all, a pixel value is already a usable vector component. Bytes are discrete, so
    SOME fixed mapping to vectors is unavoidable -- byte_embed is that mapping: ONE global,
    FROZEN, maximally-separated (one-hot when byte_embed_dim>=256) table, decoupled from any
    level's own d_model. It is NOT trainable and never will be: every level but the terminal one
    outputs a CONTINUOUS hidden vector (no softmax, see forward()/generate()), exactly like the
    reference's non-terminal levels -- making the input-side representation trainable would blur
    a distinction the real architecture keeps clean (fixed data representation vs. learned
    processing). Only the terminal level's `head` turns a hidden vector into a byte distribution.
  - Per-level weights are INDEPENDENT by default (own d_model/n_layers/n_heads/mlp_mult, own
    trunk), matching the reference (which tapers capacity level to level, e.g.
    num_blocks_list=(24,6,3,1)). `share_trunk=True` is an explicit opt-in that requires every
    level's dims to match and literally aliases one trunk module across levels.
  - Since adjacent levels can have different d_model, a `cond_proj[l]` linear adapter maps a
    parent's condition vector into the child level's own working dimension.

IMPORTANT for compress.py/decompress.py: collect_logits() and generate() MUST stay the same
code path (collect_logits literally calls generate() with a teacher-forcing symbol_fn) --
floating-point matmuls aren't perfectly associative, so a batched recompute and a step-by-step
recompute of the "same" math can disagree in the last bit, which is enough to desync range
coding. forward() (the fast batched path) is fine for training loss -- never for the CDFs that
get range-coded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ROPE_PRESETS = {"llama2": 10000.0, "llama3": 500000.0, "qwen3": 1000000.0}


def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos[None, None], sin[None, None]
    return x * cos + rotate_half(x) * sin


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device)
    allow = pos.view(1, -1) <= pos.view(-1, 1)
    return allow.view(1, 1, seq_len, seq_len)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class Attn(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * d_model
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Attn(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.mlp(self.ln2(x))
        return x


class Trunk(nn.Module):
    """One level's transformer: a Block stack + final norm. Instantiated once per level by
    default; aliased (literally the same instance) across levels when share_trunk=True."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int, mlp_mult: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_mult) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, rope_base: float) -> torch.Tensor:
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        cos, sin = rope_cos_sin_for_positions(pos, self.head_dim, rope_base, x.device)
        mask = causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin, mask)
        return self.ln_f(x)


@dataclass
class FractalConfig:
    patch_len_list: tuple[int, ...] = (1024, 128, 16, 1)   # last element must be 1 (byte-atomic)
    d_model_list: tuple[int, ...] = (64, 64, 64)           # length = n_levels = len(patch_len_list)-1
    n_layers_list: tuple[int, ...] = (4, 4, 4)
    n_heads_list: tuple[int, ...] = (4, 4, 4)
    mlp_mult_list: tuple[int, ...] = (2, 2, 2)
    byte_embed_dim: int = 256                              # decoupled from any level's d_model
    rope_preset: str = "qwen3"
    share_trunk: bool = False                              # opt-in alias; requires matching dims


def make_byte_embedding(dim: int) -> torch.Tensor:
    """Fixed, MAXIMALLY separated per-byte representation -- exact one-hot (mutually orthogonal,
    the theoretical max pairwise distance for equal-norm vectors) when dim>=256. For dim<256 you
    can't have 256 mutually orthogonal vectors, so this falls back to fixed random unit vectors
    (an approximation, not the true max-separation packing) -- not meant to carry meaning by
    itself either way; the trainable trunk does the actual representation learning."""
    if dim >= 256:
        w = torch.zeros(256, dim)
        w[:, :256] = torch.eye(256)
    else:
        g = torch.Generator().manual_seed(0)
        w = torch.randn(256, dim, generator=g)
        w = w / w.norm(dim=-1, keepdim=True)
    return w


class ByteFractalGen(nn.Module):
    def __init__(self, cfg: FractalConfig):
        super().__init__()
        assert cfg.patch_len_list[-1] == 1, "last patch_len must be 1 (byte-atomic terminal level)"
        for a, b in zip(cfg.patch_len_list, cfg.patch_len_list[1:]):
            assert a % b == 0, f"patch_len_list must divide evenly level to level, got {a} -> {b}"
        self.cfg = cfg
        self.n_levels = len(cfg.patch_len_list) - 1
        self.seq_lens = [cfg.patch_len_list[l] // cfg.patch_len_list[l + 1] for l in range(self.n_levels)]
        for name, lst in [("d_model_list", cfg.d_model_list), ("n_layers_list", cfg.n_layers_list),
                          ("n_heads_list", cfg.n_heads_list), ("mlp_mult_list", cfg.mlp_mult_list)]:
            assert len(lst) == self.n_levels, f"{name} must have length n_levels={self.n_levels}, got {len(lst)}"
        for l, (d, h) in enumerate(zip(cfg.d_model_list, cfg.n_heads_list)):
            assert d % h == 0, f"level {l}: d_model={d} not divisible by n_heads={h}"
            assert (d // h) % 2 == 0, (
                f"level {l}: head_dim={d//h} (d_model={d}/n_heads={h}) must be even -- "
                f"RoPE's rotate_half splits it in two")
        self.rope_base = ROPE_PRESETS[cfg.rope_preset]

        # Frozen, maximally-separated byte representation -- fixed data, never trained (see
        # module docstring for why this must stay frozen).
        self.byte_embed = nn.Embedding(256, cfg.byte_embed_dim)
        self.byte_embed.weight.data.copy_(make_byte_embedding(cfg.byte_embed_dim))
        self.byte_embed.weight.requires_grad_(False)

        if cfg.share_trunk:
            d0, l0, h0, m0 = cfg.d_model_list[0], cfg.n_layers_list[0], cfg.n_heads_list[0], cfg.mlp_mult_list[0]
            for l in range(self.n_levels):
                assert (cfg.d_model_list[l], cfg.n_layers_list[l], cfg.n_heads_list[l], cfg.mlp_mult_list[l]) \
                    == (d0, l0, h0, m0), \
                    (f"share_trunk=True requires identical dims at every level; level {l} "
                     f"({cfg.d_model_list[l]},{cfg.n_layers_list[l]},{cfg.n_heads_list[l]},{cfg.mlp_mult_list[l]}) "
                     f"!= level 0 ({d0},{l0},{h0},{m0})")
            shared = Trunk(d0, h0, l0, m0)
            self.trunks = nn.ModuleList([shared] * self.n_levels)
        else:
            self.trunks = nn.ModuleList([
                Trunk(cfg.d_model_list[l], cfg.n_heads_list[l], cfg.n_layers_list[l], cfg.mlp_mult_list[l])
                for l in range(self.n_levels)
            ])

        # patch_in[l]: flatten this level's child bytes (via the shared byte_embed) and project
        # into level l's own working dimension. Uniform formula for intermediate AND terminal
        # levels (child_len=1 there, just a per-token projection, not really a "patch").
        self.patch_in = nn.ModuleList([
            nn.Linear(cfg.patch_len_list[l + 1] * cfg.byte_embed_dim, cfg.d_model_list[l], bias=False)
            for l in range(self.n_levels)
        ])
        # cond_proj[l]: adapt the incoming condition (root_cond for l=0, else parent's d_model)
        # into level l's own working dimension. Always a real learned adapter, even when dims
        # happen to match, matching the reference's own cond_emb (never skipped/Identity there).
        self.cond_proj = nn.ModuleList([
            nn.Linear(cfg.d_model_list[0] if l == 0 else cfg.d_model_list[l - 1], cfg.d_model_list[l], bias=False)
            for l in range(self.n_levels)
        ])

        self.head = nn.Linear(cfg.d_model_list[-1], 256, bias=False)   # terminal level only
        self.root_cond = nn.Parameter(torch.zeros(1, cfg.d_model_list[0]))
        nn.init.normal_(self.root_cond, std=0.02)

    def forward(self, byte_chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """byte_chunk: [B, patch_len_list[0]] int64. Fast batched path -- training loss only,
        never for range-coding CDFs (see collect_logits)."""
        B = byte_chunk.shape[0]
        cond = self.root_cond.expand(B, -1)
        cur_bytes = byte_chunk
        total_ce_nats = cur_bytes.new_zeros((), dtype=torch.float32)
        total_positions = 0
        n_correct = 0

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
                # Terminal level: teacher-forced AR. proj already IS this level's byte embedding
                # (child_len=1), so shift it right by one and prepend the condition, matching the
                # non-terminal branch's cat-then-trunk pattern but consuming proj as the "target"
                # embedding sequence rather than prepending it whole.
                shifted = torch.cat([cond_l.unsqueeze(1), proj[:, :-1]], dim=1)
                h = self.trunks[l](shifted, self.rope_base)
                logits = self.head(h)                                           # [Bcur, seq_len, 256]
                targets = cur_bytes.squeeze(-1)
                loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="sum")
                total_ce_nats = total_ce_nats + loss
                total_positions += targets.numel()
                n_correct += (logits.argmax(-1) == targets).sum().item()
            else:
                seq = torch.cat([cond_l.unsqueeze(1), proj], dim=1)
                h = self.trunks[l](seq, self.rope_base)
                cond_next = h[:, :-1, :]                     # CONTINUOUS -- no softmax at non-terminal levels
                cur_bytes = cur_bytes.reshape(Bcur * seq_len, child_len)
                cond = cond_next.reshape(Bcur * seq_len, D)

        mean_loss = total_ce_nats / total_positions
        metrics = {
            "loss": mean_loss, "bpb": mean_loss / math.log(2),
            "byte_acc": torch.tensor(n_correct / total_positions),
        }
        return mean_loss, metrics

    @torch.no_grad()
    def generate(self, symbol_fn, batch_size: int = 1) -> list[list[int]]:
        """Generate batch_size independent patch_len_list[0]-byte chunks IN PARALLEL. Chunks
        share nothing (root_cond is a fixed constant, no cross-chunk state), so every trunk call
        below batches all `batch_size` chunks together -- the expensive part (neural forward
        passes) is parallel across chunks even though byte-by-byte decoding within one chunk
        stays inherently sequential (each position's condition depends on the ACTUAL content of
        prior positions, at every level, not just the terminal one).

        symbol_fn(logits[batch_size, 256]) -> list[int] of length batch_size: one symbol per
        batch row -- argmax/sampling for pure generation, or a range-decoder step per row for
        decompress.py (each row can use its own independent RC stream/state). batch_size=1
        (default) reproduces the original single-chunk behavior exactly."""

        def gen_patch(level: int, cond: torch.Tensor) -> list[list[int]]:
            B = cond.shape[0]
            seq_len = self.seq_lens[level]
            child_len = self.cfg.patch_len_list[level + 1]
            cond_l = self.cond_proj[level](cond)                  # [B, D]
            out = [[] for _ in range(B)]
            seq_in = [cond_l.unsqueeze(1)]                         # [B, 1, D]
            if child_len == 1:
                for i in range(seq_len):
                    x = torch.cat(seq_in, dim=1)
                    h = self.trunks[level](x, self.rope_base)
                    logits = self.head(h[:, -1, :])                 # [B, 256]
                    syms = symbol_fn(logits)                        # list[int], len B
                    for b in range(B):
                        out[b].append(syms[b])
                    if i < seq_len - 1:
                        sym_t = torch.tensor(syms, device=cond.device).unsqueeze(1)   # [B, 1]
                        emb = self.byte_embed(sym_t)                 # [B, 1, Ebyte]
                        seq_in.append(self.patch_in[level](emb))
                return out
            else:
                for i in range(seq_len):
                    x = torch.cat(seq_in, dim=1)
                    h = self.trunks[level](x, self.rope_base)
                    cond_i = h[:, -1, :]                            # [B, D] CONTINUOUS, passed to child unchanged
                    child_bytes = gen_patch(level + 1, cond_i)      # list of B lists, len child_len
                    for b in range(B):
                        out[b].extend(child_bytes[b])
                    if i < seq_len - 1:
                        child_t = torch.tensor(child_bytes, dtype=torch.long, device=cond.device)  # [B, child_len]
                        child_emb = self.byte_embed(child_t).reshape(B, 1, child_len * self.cfg.byte_embed_dim)
                        seq_in.append(self.patch_in[level](child_emb))
                return out

        root = self.root_cond.expand(batch_size, -1)
        return gen_patch(0, root)

    @torch.no_grad()
    def collect_logits(self, byte_chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """byte_chunks: [B, P0]. Teacher-forced, routed through the SAME batched step-by-step
        generate() decompress.py uses (bit-exactness -- see module docstring). Returns FLAT
        (symbols[B*P0], logits[B*P0, 256]) in the exact INTERLEAVED order generate()'s internal
        symbol_fn calls happen in: all B rows' step-i symbol before any row's step-i+1. This is
        the order compress.py must feed to rc_encode -- decompress.py's generate() call with the
        SAME batch_size will walk the SAME rc_stream in this SAME order. Batch composition
        (which chunks, what size, what order) must match exactly between compress and decompress:
        batched matmuls aren't bit-identical to differently-batched ones (float non-associativity),
        so a batch-size mismatch between the two sides would desync the range coder."""
        B = byte_chunks.shape[0]
        true_bytes = byte_chunks.tolist()
        flat_symbols, flat_logits = [], []
        counters = [0] * B

        def symbol_fn(logits_batch):
            syms = []
            for b in range(B):
                flat_logits.append(logits_batch[b].clone())
                sym = true_bytes[b][counters[b]]
                flat_symbols.append(sym)
                syms.append(sym)
                counters[b] += 1
            return syms

        self.generate(symbol_fn, batch_size=B)
        return torch.tensor(flat_symbols, dtype=torch.long), torch.stack(flat_logits)
