"""Byte-level FractalGen, single shared trunk -- ported from the reference image FractalGen
(https://arxiv.org/html/2502.17437v2, models/fractalgen.py + models/ar.py in this same folder,
the original authors' repo checked out here for reference).

Core idea unchanged: a generative model built by recursively invoking a generator on
progressively smaller sub-patches, so a length-L sequence costs O(L) total attention (each level
only attends over a short seq_len), not O(L^2) for one flat pass. Training is teacher-forced and
fully vectorized via the same trick the reference implementation uses: each level's sub-patches
become the BATCH dimension for the next level's call, so the whole multi-level recursion is one
forward pass with a Python loop over levels (not over samples or positions).

Adaptations for 1D bytes instead of 2D continuous-pixel image patches:
  - "patchify" splits a byte sequence into equal sub-chunks, not a 2D image into square tiles.
  - patch content is byte EMBEDDINGS (discrete, vocab=256), not raw pixel floats -- so the
    terminal level is a plain softmax cross-entropy head, no PixelLoss (that module exists in
    the reference solely to model continuous RGB values; bytes are already discrete symbols).
  - byte_embed can be frozen at a fixed one-hot (dim>=256) or fixed random-unit-vector (dim<256)
    init -- "as long as all bytes get a distinguishable representation," the LM's own trainable
    layers do the actual representation learning, not the embedding table itself.
  - "shared all layers": ONE transformer Block stack (from overfitter/summformer.py, reused
    as-is) is called at every level and every recursive step, unlike the reference, which gives
    each level (and each of AR's own internal blocks) independent weights. Per-level patch-in
    projections stay level-specific (patch sizes differ level to level, so their input shapes
    differ) -- everything else (attention/MLP trunk, output head, condition projection) is one
    shared instance.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from overfitter.summformer import Block, RMSNorm, causal_mask, rope_cos_sin_for_positions, ROPE_PRESETS


@dataclass
class FractalConfig:
    patch_len_list: tuple[int, ...] = (1024, 128, 16, 1)   # last element must be 1 (byte-atomic)
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None
    mlp_mult: int = 2
    qk_norm: bool = True
    rope_preset: str = "qwen3"
    freeze_byte_embed: bool = True   # one-hot (d_model>=256) or fixed random unit vectors


def make_byte_embedding(d_model: int) -> torch.Tensor:
    """Fixed, distinguishable per-byte representation -- one-hot if it fits, else a fixed
    (seeded) set of unit vectors. Not meant to carry meaning by itself; the trunk learns that."""
    if d_model >= 256:
        w = torch.zeros(256, d_model)
        w[:, :256] = torch.eye(256)
    else:
        g = torch.Generator().manual_seed(0)
        w = torch.randn(256, d_model, generator=g)
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
        D = cfg.d_model
        self.head_dim = D // cfg.n_heads
        self.rope_base = ROPE_PRESETS[cfg.rope_preset]
        assert D % cfg.n_heads == 0

        self.byte_embed = nn.Embedding(256, D)
        self.byte_embed.weight.data.copy_(make_byte_embedding(D))
        self.byte_embed.weight.requires_grad_(not cfg.freeze_byte_embed)

        # shared trunk -- ONE Block stack, called at every level and every recursive step
        self.trunk = nn.ModuleList(
            [Block(D, cfg.n_heads, cfg.mlp_mult, cfg.n_kv_heads, cfg.qk_norm) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(D)

        # level-specific patch-in projections (shapes differ: child_len bytes -> D), except the
        # terminal level (child_len=1) which needs none -- byte_embed alone is its per-token input
        self.patch_in = nn.ModuleList([
            nn.Linear(cfg.patch_len_list[l + 1] * D, D, bias=False)
            for l in range(self.n_levels) if cfg.patch_len_list[l + 1] > 1
        ])
        self.head = nn.Linear(D, 256, bias=False)   # shared terminal byte-prediction head
        self.root_cond = nn.Parameter(torch.zeros(1, D))   # unconditional "root" -- one file, no classes
        nn.init.normal_(self.root_cond, std=0.02)

    def _run_trunk(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        cos, sin = rope_cos_sin_for_positions(pos, self.head_dim, self.rope_base, x.device)
        mask = causal_mask(pos, pos, None)
        for blk in self.trunk:
            x = blk(x, cos, sin, mask)
        return self.ln_f(x)

    def forward(self, byte_chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """byte_chunk: [B, patch_len_list[0]] int64. Returns (mean CE loss, metrics)."""
        B = byte_chunk.shape[0]
        cond = self.root_cond.expand(B, -1)
        cur_bytes = byte_chunk
        total_ce_nats = cur_bytes.new_zeros((), dtype=torch.float32)
        total_positions = 0
        n_correct = 0
        patch_in_idx = 0

        for l in range(self.n_levels):
            seq_len = self.seq_lens[l]
            child_len = self.cfg.patch_len_list[l + 1]
            cur_bytes = cur_bytes.reshape(-1, seq_len, child_len)
            Bcur = cur_bytes.shape[0]

            if child_len == 1:
                # Terminal level: teacher-forced byte-level AR over this seq_len-byte atomic patch.
                targets = cur_bytes.squeeze(-1)                              # [Bcur, seq_len]
                tok_emb = self.byte_embed(targets)                           # [Bcur, seq_len, D]
                shifted = torch.cat([cond.unsqueeze(1), tok_emb[:, :-1]], dim=1)
                h = self._run_trunk(shifted)                                 # [Bcur, seq_len, D]
                logits = self.head(h)                                        # [Bcur, seq_len, 256]
                loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="sum")
                total_ce_nats = total_ce_nats + loss
                total_positions += targets.numel()
                n_correct += (logits.argmax(-1) == targets).sum().item()
            else:
                patch_bytes_emb = self.byte_embed(cur_bytes)                 # [Bcur, seq_len, child_len, D]
                flat = patch_bytes_emb.reshape(Bcur, seq_len, child_len * self.cfg.d_model)
                proj = self.patch_in[patch_in_idx](flat)                     # [Bcur, seq_len, D]
                patch_in_idx += 1
                seq = torch.cat([cond.unsqueeze(1), proj], dim=1)            # [Bcur, seq_len+1, D]
                h = self._run_trunk(seq)
                cond_next = h[:, :-1, :]                                     # causal: position i conditions child i

                cur_bytes = cur_bytes.reshape(Bcur * seq_len, child_len)
                cond = cond_next.reshape(Bcur * seq_len, self.cfg.d_model)

        mean_loss = total_ce_nats / total_positions
        import math
        metrics = {
            "loss": mean_loss, "bpb": mean_loss / math.log(2),
            "byte_acc": torch.tensor(n_correct / total_positions),
        }
        return mean_loss, metrics

    @torch.no_grad()
    def collect_logits(self, byte_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced, but deliberately routed through the SAME step-by-step generate()
        used by decompress.py -- NOT the fast batched forward()/collect path. Floating-point
        matmuls aren't perfectly associative, so a batched computation and an incremental
        recomputation of the "same" math can disagree in the last bit; for range coding that's
        enough to desync the decoder (a 1-unit difference in a quantized integer CDF).
        Routing both sides through one shared function eliminates the possibility by
        construction, exactly like overfitter's _make_incremental_stepper is shared by
        compress.py and decompress.py there. Slower than a batched pass, but only used for
        compression stats/CDFs, not the training hot loop (forward() stays batched)."""
        assert byte_chunk.shape[0] == 1
        true_bytes = byte_chunk[0].tolist()
        logits_out = []
        counter = [0]

        def symbol_fn(logits_row):
            logits_out.append(logits_row.clone())
            sym = true_bytes[counter[0]]
            counter[0] += 1
            return sym

        self.generate(symbol_fn)
        return byte_chunk[0].clone(), torch.stack(logits_out)

    @torch.no_grad()
    def generate(self, symbol_fn) -> list[int]:
        """Generate one full patch_len_list[0]-byte chunk top-down. symbol_fn(logits[256]) -> int
        picks the byte given this position's logits -- plug in argmax/sampling for pure
        generation, or a range-decoder step for decompress.py. No state carried across chunks
        (root_cond is a fixed constant), so this is called once per chunk, independently."""

        def gen_patch(level: int, cond: torch.Tensor) -> list[int]:
            seq_len = self.seq_lens[level]
            child_len = self.cfg.patch_len_list[level + 1]
            if child_len == 1:
                out = []
                seq_in = [cond.view(1, 1, -1)]
                for i in range(seq_len):
                    x = torch.cat(seq_in, dim=1)
                    h = self._run_trunk(x)
                    logits = self.head(h[:, -1, :]).squeeze(0)   # [256]
                    sym = symbol_fn(logits)
                    out.append(sym)
                    if i < seq_len - 1:
                        seq_in.append(self.byte_embed(
                            torch.tensor([[sym]], device=cond.device)))
                return out
            else:
                level_patch_idx = sum(
                    1 for ll in range(level) if self.cfg.patch_len_list[ll + 1] > 1)
                out = []
                seq_in = [cond.view(1, 1, -1)]
                for i in range(seq_len):
                    x = torch.cat(seq_in, dim=1)
                    h = self._run_trunk(x)
                    cond_i = h[:, -1, :]                          # [1, D]
                    child_bytes = gen_patch(level + 1, cond_i)
                    out.extend(child_bytes)
                    if i < seq_len - 1:
                        child_t = torch.tensor([child_bytes], dtype=torch.long, device=cond.device)
                        child_emb = self.byte_embed(child_t).reshape(1, 1, child_len * self.cfg.d_model)
                        seq_in.append(self.patch_in[level_patch_idx](child_emb))
                return out

        return gen_patch(0, self.root_cond)
