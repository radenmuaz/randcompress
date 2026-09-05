import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ----------------------------------------------------------------------------
# small shared utilities (copied/trimmed from qcute_zero.py)
# ----------------------------------------------------------------------------

def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Logger:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.text_path = run_dir / "run.log"
        self.json_path = run_dir / "run.jsonl"
        self.text_f = open(self.text_path, "a")
        self.json_f = open(self.json_path, "a")
        self.start_time = time.time()

    def __call__(self, msg: str, **record) -> None:
        elapsed_s = int(time.time() - self.start_time)
        elapsed_hms = format_hms(elapsed_s)
        line = f"[{elapsed_hms}] {msg}"
        tqdm.write(line)
        self.text_f.write(line + "\n")
        self.text_f.flush()
        json_record = {"elapsed_s": elapsed_s, "elapsed_hms": elapsed_hms, **({} if record else {"msg": msg}), **record}
        self.json_f.write(json.dumps(json_record) + "\n")
        self.json_f.flush()


class Checkpointer:
    def __init__(self, run_dir: Path, save_every_n_evals: int = 1, minimize: bool = True):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = run_dir / "best.pt"
        self.last_path = run_dir / "last.pt"
        self.save_every_n_evals = max(1, save_every_n_evals)
        self.minimize = minimize
        self.best_metric = float("inf") if minimize else float("-inf")
        self._eval_count = 0

    def is_better(self, metric: float) -> bool:
        if not math.isfinite(metric) or metric <= 0:
            return False
        return metric < self.best_metric if self.minimize else metric > self.best_metric

    def step(self, state: dict, metric: float) -> None:
        self._eval_count += 1
        if self.is_better(metric):
            self.best_metric = metric
            torch.save(state, self.best_path)
        if self._eval_count % self.save_every_n_evals == 0:
            torch.save(state, self.last_path)


def unpack_words(data: bytes, bits: int) -> list:
    if bits == 8:
        return list(data)
    words = []
    mask = (1 << bits) - 1
    for byte in data:
        for shift in range(8 - bits, -1, -bits):
            words += [(byte >> shift) & mask]
    return words


def load_enwik8(path: Path, bits: int, n_bytes: int | None = None) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read(n_bytes) if n_bytes else f.read()
    return torch.tensor(unpack_words(data, bits), dtype=torch.long)


def split_train_val(data: torch.Tensor, val_frac: float) -> tuple:
    n_val = max(1, int(len(data) * val_frac))
    return data[:-n_val], data[-n_val:]


def sample_context(data: torch.Tensor, batch_size: int, context_len: int, device: str) -> torch.Tensor:
    n = max(1, len(data) - context_len)
    starts = torch.randint(0, n, (batch_size,))
    return torch.stack([data[s:s + context_len] for s in starts]).to(device)


def lr_at(step: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak


def load_config_module(path: Path) -> dict:
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------
# RoPE + attention primitives (unchanged from qcute_zero.py)
# ----------------------------------------------------------------------------

ROPE_PRESETS = {"llama2": 10000.0, "llama3": 500000.0, "qwen3": 1000000.0}  # theta only, no
                                                                              # Llama3.1 NTK-by-parts scaling


def rope_cos_sin_for_positions(position_ids: torch.Tensor, head_dim: int, base: float, device: torch.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    return x * cos + rotate_half(x) * sin


def sdpa_with_sink(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Mandatory zero-value/zero-key sink -- see qcute_zero.py's own docstring for the full
    rationale (every query row keeps >=1 valid key; a sink-only row is a provably clean zero)."""
    B, H, T, hd = q.shape
    sink_k = k.new_zeros(B, H, 1, hd)
    sink_v = v.new_zeros(B, H, 1, hd)
    k2 = torch.cat([sink_k, k], dim=2)
    v2 = torch.cat([sink_v, v], dim=2)
    sink_col = attn_mask.new_ones(attn_mask.shape[:-1] + (1,))
    mask2 = torch.cat([sink_col, attn_mask], dim=-1)
    return F.scaled_dot_product_attention(q, k2, v2, attn_mask=mask2)


def causal_mask(query_pos: torch.Tensor, key_pos: torch.Tensor, window: int | None) -> torch.Tensor:
    allow = key_pos.view(1, -1) <= query_pos.view(-1, 1)
    if window is not None:
        allow = allow & ((query_pos.view(-1, 1) - key_pos.view(1, -1)) < window)
    return allow.view(1, 1, *allow.shape)


def _fmt_bytes(t: torch.Tensor) -> str:
    return bytes(int(b) & 0xFF for b in t.tolist()).decode("latin-1", errors="replace")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class Attn(nn.Module):
    """GQA (n_kv_heads < n_heads repeats each KV head across n_heads//n_kv_heads query heads) +
    optional Qwen3-style QK-norm (per-head RMSNorm on Q/K, applied before RoPE) + optional
    decoupled head_dim (Qwen3-style: some of its smaller variants set head_dim independently of
    d_model//n_heads -- None here is a no-op, deriving head_dim the old way)."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int | None = None, qk_norm: bool = True,
                 head_dim: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.d_model = d_model
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.attn_dim = n_heads * self.head_dim
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(self.attn_dim, d_model, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, Hkv, T, hd = x.shape
        return x[:, :, None].expand(B, Hkv, self.n_rep, T, hd).reshape(B, Hkv * self.n_rep, T, hd)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Hkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, Hkv, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, Hkv, hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        y = sdpa_with_sink(q, k, v, attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, self.attn_dim))

    def forward_incremental(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                             cache, window: int | None):
        B, Tn, D = x_new.shape
        H, Hkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.wq(x_new).view(B, Tn, H, hd).transpose(1, 2)
        k = self.wk(x_new).view(B, Tn, Hkv, hd).transpose(1, 2)
        v = self.wv(x_new).view(B, Tn, Hkv, hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos_new, sin_new), apply_rope(k, cos_new, sin_new)
        if cache is None:
            k_all, v_all, S_prev = k, v, 0
        else:
            k_prev, v_prev = cache
            k_all, v_all = torch.cat([k_prev, k], dim=2), torch.cat([v_prev, v], dim=2)
            S_prev = k_prev.shape[2]
        S = k_all.shape[2]
        new_pos = torch.arange(S_prev, S_prev + Tn, device=x_new.device)
        key_pos = torch.arange(S, device=x_new.device)
        mask = causal_mask(new_pos, key_pos, window)
        y = sdpa_with_sink(q, self._repeat_kv(k_all), self._repeat_kv(v_all), mask)
        out = self.out(y.transpose(1, 2).reshape(B, Tn, self.attn_dim))
        if window is not None and S > window:
            k_all, v_all = k_all[:, :, -window:], v_all[:, :, -window:]
        return out, (k_all, v_all)

    def forward_cross(self, x_q: torch.Tensor, x_kv: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor,
                       cos_k: torch.Tensor, sin_k: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x_q.shape
        _, S, _ = x_kv.shape
        H, Hkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.wq(x_q).view(B, T, H, hd).transpose(1, 2)
        k = self.wk(x_kv).view(B, S, Hkv, hd).transpose(1, 2)
        v = self.wv(x_kv).view(B, S, Hkv, hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        y = sdpa_with_sink(q, k, v, attn_mask)
        return self.out(y.transpose(1, 2).reshape(B, T, self.attn_dim))


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
    """Self-attention + MLP. Shared (same weights) across the byte-level pass, every fuse stage's
    own hierarchical-summarization pass, and the post-cross-attn refinement pass -- this IS the
    "single LM" the design hinges on, same as qcute_zero."""
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, n_kv_heads: int | None = None, qk_norm: bool = True,
                 head_dim: int | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Attn(d_model, n_heads, n_kv_heads, qk_norm, head_dim)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_mult)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_incremental(self, x_new: torch.Tensor, cos_new: torch.Tensor, sin_new: torch.Tensor,
                             cache, window: int | None):
        attn_out, new_cache = self.attn.forward_incremental(self.ln1(x_new), cos_new, sin_new, cache, window)
        x_new = x_new + attn_out
        x_new = x_new + self.mlp(self.ln2(x_new))
        return x_new, new_cache


class FuseStage(nn.Module):
    """Cross-attention + MLP, one instance per periodic-fusion stage, own weights throughout.
    Unchanged from qcute_zero -- agnostic to how the code_kv sequence was produced (quantized or,
    here, plain continuous pooled hidden states)."""
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, n_layers: int,
                 n_kv_heads: int | None = None, qk_norm: bool = True, head_dim: int | None = None):
        super().__init__()
        self.ln1 = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.attn = nn.ModuleList([Attn(d_model, n_heads, n_kv_heads, qk_norm, head_dim) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.mlp = nn.ModuleList([SwiGLU(d_model, mlp_mult) for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, code_kv: torch.Tensor, cos_q, sin_q, cos_k, sin_k,
                attn_mask: torch.Tensor) -> torch.Tensor:
        for l in range(len(self.attn)):
            xn = self.ln1[l](x)
            coden = self.ln1[l](code_kv)
            x = x + self.attn[l].forward_cross(xn, coden, cos_q, sin_q, cos_k, sin_k, attn_mask)
            x = x + self.mlp[l](self.ln2[l](x))
        return x

    def readout(self, x: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        return F.linear(self.ln_out(x), embed_weight)


def resolve_fuse_window(w, n_fuse: int) -> tuple:
    if isinstance(w, (tuple, list)):
        assert len(w) == n_fuse
        return tuple(w)
    return (w,) * n_fuse


# ----------------------------------------------------------------------------
# Config + model
# ----------------------------------------------------------------------------

@dataclass
class Config:
    Ks: tuple[int, ...] = (32, 32, 1)       # same semantics as qcute_zero: cumulative periods
    d_model: int = 256
    n_layers: int = 4                        # shared "block regular", reused for every level
    fuse_n_layers: int | None = None         # defaults to n_layers if unset
    n_heads: int = 4
    n_kv_heads: int | None = None            # None = max(1, n_heads//4) (Llama3/Qwen3-style GQA-by-default); set == n_heads for plain MHA
    head_dim: int | None = None              # None = d_model // n_heads (no-op, Llama3/big-Qwen3 style); set to
                                              # decouple from d_model/n_heads (small-Qwen3 style, e.g. head_dim=128)
    qk_norm: bool = True                    # Qwen3-style per-head RMSNorm on Q/K before RoPE
    mlp_mult: int = 4
    rope_base: float = 10000.0
    rope_preset: str | None = "qwen3"           # "llama2"/"llama3"/"qwen3" overrides rope_base (see ROPE_PRESETS)
    context_len: int = 256
    attn_window: int | None = None
    fuse_window: int | tuple | None = None   # per-fuse-stage cross-attn window, in BYTES
    input_preset: int = 8                    # byte alphabet bits -- vocab = 2**input_preset
    mtp_heads: int = 1                       # extra byte-ahead heads reading the final post-
                                              # cascade hidden state (1 = disabled)
    mtp_weight: float = 1.0
    weight_tie: bool = False                 # True: head.weight literally refs embed.weight
    share_lm: bool = False                   # True ties every level to the same Block stack
    share_fuse: bool = False                 # True ties every fuse stage to the same FuseStage


class SummTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.rope_preset is not None:
            cfg.rope_base = ROPE_PRESETS[cfg.rope_preset]
        self.cfg = cfg
        D = cfg.d_model
        self.head_dim = cfg.head_dim if cfg.head_dim is not None else D // cfg.n_heads
        V = 2 ** cfg.input_preset
        self.vocab = V
        self.n_fuse = len(cfg.Ks) - 1
        assert D % cfg.n_heads == 0

        self.embed = nn.Embedding(V, D)
        nn.init.normal_(self.embed.weight, std=0.02)

        # per-level LM stacks: level 0 = byte pass + post-cross-attn refinement pass, level s+1 =
        # fuse stage s's own hierarchical-summarization pass over its pooled (continuous) sequence.
        n_lms = self.n_fuse + 1
        if cfg.share_lm:
            first = nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult, cfg.n_kv_heads, cfg.qk_norm, cfg.head_dim) for _ in range(cfg.n_layers)])
            self.lms = nn.ModuleList([first] * n_lms)
        else:
            self.lms = nn.ModuleList(
                [nn.ModuleList([Block(D, cfg.n_heads, cfg.mlp_mult, cfg.n_kv_heads, cfg.qk_norm, cfg.head_dim) for _ in range(cfg.n_layers)])
                 for _ in range(n_lms)])
        if cfg.share_lm:
            first_ln = RMSNorm(D)
            self.ln_fs = nn.ModuleList([first_ln] * n_lms)
        else:
            self.ln_fs = nn.ModuleList([RMSNorm(D) for _ in range(n_lms)])

        self.head = nn.Linear(D, V, bias=False)
        if cfg.weight_tie:
            self.head.weight = self.embed.weight
        else:
            nn.init.normal_(self.head.weight, std=0.02)

        fuse_layers = cfg.fuse_n_layers if cfg.fuse_n_layers is not None else cfg.n_layers
        if cfg.share_fuse:
            first_fs = FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers, cfg.n_kv_heads, cfg.qk_norm, cfg.head_dim)
            self.fuse_stages = nn.ModuleList([first_fs] * self.n_fuse)
        else:
            self.fuse_stages = nn.ModuleList(
                [FuseStage(D, cfg.n_heads, cfg.mlp_mult, fuse_layers, cfg.n_kv_heads, cfg.qk_norm, cfg.head_dim) for _ in range(self.n_fuse)])
        self.fuse_windows = resolve_fuse_window(cfg.fuse_window, self.n_fuse)

        self.extra_heads = nn.ModuleList(
            [nn.Linear(D, V, bias=False) for _ in range(max(0, cfg.mtp_heads - 1))])

    def _run_blocks(self, level: int, x: torch.Tensor, cos, sin, attn_mask) -> torch.Tensor:
        for block in self.lms[level]:
            x = block(x, cos, sin, attn_mask)
        return self.ln_fs[level](x)

    def _cascade(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Runs the full hierarchical summarize+fuse cascade, full recompute. Returns the final
        byte-level query stream x_cross (B, L, D), post every active fuse stage's cross-attention
        + refinement pass. A position before any stage's first boundary just carries the plain
        byte-level h through untouched (no cross-attention has happened yet for it)."""
        cfg = self.cfg
        B, L = byte_ids.shape
        D = cfg.d_model
        hd = self.head_dim
        device = byte_ids.device

        byte_pos = torch.arange(L, device=device)
        cos_b, sin_b = rope_cos_sin_for_positions(byte_pos, hd, cfg.rope_base, device)
        byte_mask = causal_mask(byte_pos, byte_pos, cfg.attn_window)
        x0 = self.embed(byte_ids)
        h = self._run_blocks(0, x0, cos_b, sin_b, byte_mask)

        cur_h = h                # source hidden states to pool this stage's summary from
        x_cross = h              # running byte-level query stream, refined by each fuse stage
        cum_K = 1
        for s in range(self.n_fuse):
            K_s = cfg.Ks[s]
            cum_K *= K_s
            cur_len = cur_h.shape[1]
            n_blocks = cur_len // K_s
            if n_blocks < 1:
                break

            # pooling: last-of-block hidden state, no quantizer -- this continuous vector IS the
            # level's summary/"code".
            code_h = cur_h[:, K_s - 1::K_s, :][:, :n_blocks, :]

            # hierarchical-summarization pass: SAME shared blocks, causal self-attention over the
            # short pooled sequence (length n_blocks = cur_len // K_s).
            code_local_pos = torch.arange(n_blocks, device=device)
            cos_c, sin_c = rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base, device)
            code_mask = causal_mask(code_local_pos, code_local_pos, None)
            h_code = self._run_blocks(s + 1, code_h, cos_c, sin_c, code_mask)

            # cross-attn: byte-level query stream attends into h_code, causal on the CUMULATIVE
            # (absolute-byte) boundary -- same non-circularity argument as qcute_zero.
            code_pos_abs = (torch.arange(n_blocks, device=device) + 1) * cum_K - 1
            window_s = self.fuse_windows[s]
            fuse_mask = causal_mask(byte_pos, code_pos_abs, window_s)
            cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device)

            x_cross = self.fuse_stages[s](x_cross, h_code, cos_b, sin_b, cos_k, sin_k, fuse_mask)
            x_cross = self._run_blocks(0, x_cross, cos_b, sin_b, byte_mask)   # refinement pass
            cur_h = h_code

        return x_cross

    def forward(self, byte_ids: torch.Tensor) -> tuple:
        cfg = self.cfg
        V = self.vocab
        L = byte_ids.shape[1]
        x_cross = self._cascade(byte_ids)

        logits = self.head(x_cross[:, :-1, :])
        loss = F.cross_entropy(logits.reshape(-1, V), byte_ids[:, 1:].reshape(-1))
        acc = (logits.argmax(-1) == byte_ids[:, 1:]).float().mean()

        mtp_losses, mtp_accs = [], []
        for i, head_i in enumerate(self.extra_heads):
            k = i + 2
            if L <= k:
                continue
            logits_i = head_i(x_cross[:, :-k, :])
            targets_i = byte_ids[:, k:]
            mtp_losses.append(F.cross_entropy(logits_i.reshape(-1, V), targets_i.reshape(-1)))
            mtp_accs.append((logits_i.argmax(-1) == targets_i).float().mean())

        total_loss = loss
        if mtp_losses:
            total_loss = total_loss + cfg.mtp_weight * torch.stack(mtp_losses).mean()

        metrics = {
            "loss": total_loss, "final_loss": loss, "bpb": loss / math.log(2), "byte_acc": acc,
            **{f"mtp{i+2}_loss": l for i, l in enumerate(mtp_losses)},
            **{f"mtp{i+2}_acc": a for i, a in enumerate(mtp_accs)},
        }
        return total_loss, metrics

    @torch.no_grad()
    def _forward_next_byte_logits(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Full recompute over the whole sequence so far, returns logits for the NEXT byte."""
        x_cross = self._cascade(byte_ids)
        return self.head(x_cross[:, -1, :])

    @torch.no_grad()
    def generate_no_cache(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
        """Byte-by-byte, full recompute each step -- correctness reference for generate_kv_cache."""
        was_training = self.training
        self.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        all_bytes = prompt_bytes
        for _ in range(n_new_bytes):
            logits = self._forward_next_byte_logits(all_bytes)
            next_byte = logits.argmax(-1, keepdim=True)
            all_bytes = torch.cat([all_bytes, next_byte], dim=1)
        if was_training:
            self.train()
        return all_bytes[0]

    def _make_incremental_stepper(self, Bsz: int, device_t: torch.device):
        """Factory for the real incremental-KV-cache stepper, ported from qcute_zero.py's own
        (see that file for the full rationale/history of the backlog logic) with the quantizer's
        extract_greedy step simply removed -- the pooled hidden state IS the code, no discretization
        needed, so a new code boundary just recomputes this stage's short pooled sequence fresh."""
        cfg = self.cfg
        D = cfg.d_model
        hd = self.head_dim

        byte_caches = [None] * cfg.n_layers
        refine_caches = [[None] * cfg.n_layers for _ in range(self.n_fuse)]
        h_hist = None
        stage_h_hist = [torch.zeros(Bsz, 0, D, device=device_t) for _ in range(self.n_fuse)]
        x_in_backlog = [None] * self.n_fuse
        cum_Ks = []
        cum = 1
        for K_s in cfg.Ks[:self.n_fuse]:
            cum *= K_s
            cum_Ks.append(cum)

        def step(byte_chunk: torch.Tensor, start_pos: int) -> torch.Tensor:
            nonlocal h_hist
            Tn = byte_chunk.shape[1]
            pos = torch.arange(start_pos, start_pos + Tn, device=device_t)
            cos_b, sin_b = rope_cos_sin_for_positions(pos, hd, cfg.rope_base, device_t)
            h_new = self.embed(byte_chunk)
            for l, block in enumerate(self.lms[0]):
                h_new, byte_caches[l] = block.forward_incremental(h_new, cos_b, sin_b, byte_caches[l], cfg.attn_window)
            h_new = self.ln_fs[0](h_new)
            h_hist = h_new if h_hist is None else torch.cat([h_hist, h_new], dim=1)

            x_in = h_new
            cur_h_hist = h_hist
            logits_full = self.head(x_in)   # fallback if n_fuse==0 or no stage active yet
            for s in range(self.n_fuse):
                K_s = cfg.Ks[s]
                n_blocks = cur_h_hist.shape[1] // K_s
                if n_blocks > stage_h_hist[s].shape[1]:
                    code_h = cur_h_hist[:, K_s - 1::K_s, :][:, :n_blocks, :]
                    code_local_pos = torch.arange(n_blocks, device=device_t)
                    cos_c, sin_c = rope_cos_sin_for_positions(code_local_pos, hd, cfg.rope_base, device_t)
                    code_mask = causal_mask(code_local_pos, code_local_pos, None)
                    stage_h_hist[s] = self._run_blocks(s + 1, code_h, cos_c, sin_c, code_mask)
                h_code = stage_h_hist[s]
                n_blocks_now = h_code.shape[1]

                if n_blocks_now < 1:
                    x_in_backlog[s] = x_in if x_in_backlog[s] is None else torch.cat([x_in_backlog[s], x_in], dim=1)
                    break

                code_pos_abs = (torch.arange(n_blocks_now, device=device_t) + 1) * cum_Ks[s] - 1
                window_s = self.fuse_windows[s]
                cos_k, sin_k = rope_cos_sin_for_positions(code_pos_abs, hd, cfg.rope_base, device_t)

                if refine_caches[s][0] is None:
                    x_q = x_in if x_in_backlog[s] is None else torch.cat([x_in_backlog[s], x_in], dim=1)
                    x_in_backlog[s] = None
                else:
                    x_q = x_in
                q_len = x_q.shape[1]
                q_start = (start_pos + Tn) - q_len
                q_pos = torch.arange(q_start, q_start + q_len, device=device_t)
                cos_q, sin_q = rope_cos_sin_for_positions(q_pos, hd, cfg.rope_base, device_t)
                fuse_mask = causal_mask(q_pos, code_pos_abs, window_s)

                x_cross = self.fuse_stages[s](x_q, h_code, cos_q, sin_q, cos_k, sin_k, fuse_mask)
                for l, block in enumerate(self.lms[0]):
                    x_cross, refine_caches[s][l] = block.forward_incremental(
                        x_cross, cos_q, sin_q, refine_caches[s][l], cfg.attn_window)
                x_cross = self.ln_fs[0](x_cross)
                logits_full = self.fuse_stages[s].readout(x_cross, self.head.weight)
                x_in = x_cross
                cur_h_hist = h_code
            return logits_full

        return step

    @torch.no_grad()
    def generate_kv_cache(self, prompt_bytes: torch.Tensor, n_new_bytes: int, device: str) -> torch.Tensor:
        """Real incremental KV cache -- O(1) new attention work per new byte, vs generate_no_cache's
        full O(L) recompute. Produces the exact same argmax trajectory (see
        check_kv_cache_consistency)."""
        was_training = self.training
        self.eval()
        prompt_bytes = prompt_bytes.to(device)
        if prompt_bytes.dim() == 1:
            prompt_bytes = prompt_bytes.unsqueeze(0)
        step = self._make_incremental_stepper(prompt_bytes.shape[0], torch.device(device))

        all_bytes = prompt_bytes
        logits_all = step(all_bytes, 0)          # prime the caches with the whole prompt
        next_logits = logits_all[:, -1, :]
        for _ in range(n_new_bytes):
            next_byte = next_logits.argmax(-1, keepdim=True)
            all_bytes = torch.cat([all_bytes, next_byte], dim=1)
            logits_all = step(next_byte, all_bytes.shape[1] - 1)
            next_logits = logits_all[:, -1, :]

        if was_training:
            self.train()
        return all_bytes[0]

    @torch.no_grad()
    def check_kv_cache_consistency(self, val_data: torch.Tensor, device: str,
                                    n_checks: int = 3, prompt_len: int = 8, n_new_bytes: int = 24) -> dict:
        """Diagnostic: generate_no_cache vs generate_kv_cache MUST produce bit-exact identical
        greedy trajectories. Checks n_checks random prompts sampled from val_data at varying
        lengths (short prompts specifically exercise the "stage not yet active" backlog path).
        Returns {"match_rate": float, "n_checks": int} -- should always be 1.0."""
        was_training = self.training
        self.eval()
        n_match = 0
        for i in range(n_checks):
            pl = max(1, prompt_len - i * (prompt_len // max(1, n_checks)))
            start = torch.randint(0, max(1, val_data.shape[0] - pl - n_new_bytes), (1,)).item()
            prompt = val_data[start:start + pl].to(device)
            out_full = self.generate_no_cache(prompt, n_new_bytes, device)
            out_cache = self.generate_kv_cache(prompt, n_new_bytes, device)
            if torch.equal(out_full, out_cache):
                n_match += 1
        if was_training:
            self.train()
        return {"match_rate": n_match / n_checks, "n_checks": n_checks}


# ----------------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------------

def eval_model(model, data: torch.Tensor, batch_size: int, n_batches: int, device: str) -> dict:
    model.eval()
    totals: dict = {}
    with torch.no_grad():
        for _ in range(n_batches):
            ctx = sample_context(data, batch_size, model.cfg.context_len, device)
            _, metrics = model(ctx)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v.item()
    model.train()
    return {k: v / n_batches for k, v in totals.items()}


def train(model, train_data, val_data, args, log, run_name: str, device: str) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    checkpointer = Checkpointer(args.logs_dir / run_name, args.save_every_n_evals, minimize=True)
    model.train()
    pbar = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
    for step in pbar:
        lr = lr_at(step, args.warmup_steps, args.lr_peak)
        for g in opt.param_groups:
            g["lr"] = lr
        ctx = sample_context(train_data, args.batch_size, model.cfg.context_len, device)
        loss, metrics = model(ctx)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0:
            scalars = {k: v.item() for k, v in metrics.items()}
            log(f"{pbar}", step=step, lr=lr, **scalars)

        if step % args.eval_every == 0 or step == args.steps:
            val = eval_model(model, val_data, args.batch_size, args.eval_batches, device)
            val_str = "  ".join(f"val_{k}={v:.4f}" for k, v in val.items())
            checkpointer.step({"model": model.state_dict(), "cfg": asdict(model.cfg), "step": step}, val["loss"])
            log(f"{pbar}  {val_str}  best_val_loss={checkpointer.best_metric:.4f}",
                step=step, **{f"val_{k}": v for k, v in val.items()}, best_val_loss=checkpointer.best_metric)

    if args.check_kv_cache:
        result = model.check_kv_cache_consistency(val_data, device)
        log(f"check_kv_cache_consistency: match_rate={result['match_rate']:.3f} (n_checks={result['n_checks']})",
            **result)


def build_argparser(description: str) -> tuple:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description=description, parents=[pre])
    p.add_argument("--Ks", default=(32, 32, 1))
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--fuse_n_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_kv_heads", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--qk_norm", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--mlp_mult", type=int, default=4)
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--rope_preset", type=str, default="qwen3", choices=list(ROPE_PRESETS))
    p.add_argument("--context_len", type=int, default=256)
    p.add_argument("--attn_window", default=None)
    p.add_argument("--fuse_window", default=None)
    p.add_argument("--input_preset", type=int, default=8)
    p.add_argument("--mtp_heads", type=int, default=1)
    p.add_argument("--mtp_weight", type=float, default=1.0)
    p.add_argument("--weight_tie", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--share_lm", type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--share_fuse", type=lambda x: x.lower() != "false", default=False)

    p.add_argument("--data", type=Path, default=Path("datasets/enwik8_1M.gz"))
    p.add_argument("--n_bytes", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_peak", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_batches", type=int, default=5)
    p.add_argument("--save_every_n_evals", type=int, default=1)
    p.add_argument("--check_kv_cache", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--logs_dir", type=Path, default=Path("logs"))
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)

    if pre_args.config:
        p.set_defaults(**{k: v for k, v in load_config_module(pre_args.config).items() if k in {a.dest for a in p._actions}})
    args = p.parse_args()
    if isinstance(args.Ks, str):
        args.Ks = tuple(int(x) for x in args.Ks.split(","))
    else:
        args.Ks = tuple(args.Ks)
    return args, pre_args


def config_from_args(args) -> Config:
    return Config(
        Ks=args.Ks, d_model=args.d_model, n_layers=args.n_layers, fuse_n_layers=args.fuse_n_layers,
        n_heads=args.n_heads, n_kv_heads=args.n_kv_heads, qk_norm=args.qk_norm, head_dim=args.head_dim,
        mlp_mult=args.mlp_mult, rope_base=args.rope_base, rope_preset=args.rope_preset, context_len=args.context_len,
        attn_window=args.attn_window, fuse_window=args.fuse_window, input_preset=args.input_preset,
        mtp_heads=args.mtp_heads, mtp_weight=args.mtp_weight, weight_tie=args.weight_tie,
        share_lm=args.share_lm, share_fuse=args.share_fuse,
    )


def main() -> None:
    args, pre_args = build_argparser("summformer: hierarchical-summarization transformer (qcute_zero, simplified)")
    torch.manual_seed(args.seed)
    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = config_from_args(args)
    model = SummTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    run_name = args.run_name or (pre_args.config.stem if pre_args.config else f"summformer_{int(time.time())}")
    log = Logger(args.logs_dir / run_name)
    print(f"run_name={run_name}  logging to {log.text_path} -- tail -f {log.text_path}")
    log(f"Ks={cfg.Ks} n_fuse={model.n_fuse} d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"context_len={cfg.context_len} params={n_params/1e6:.3f}M device={device}")

    data = load_enwik8(args.data, cfg.input_preset, args.n_bytes)
    train_data, val_data = split_train_val(data, args.val_frac)
    train(model, train_data, val_data, args, log, run_name, device)


if __name__ == "__main__":
    main()
