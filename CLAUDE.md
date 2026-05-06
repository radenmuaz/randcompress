# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What this project is

Neural compression via memorization: freeze a randomly-initialized xLSTM, then train only HiRA adapters to overfit a single file. The trained adapter weights *are* the compressed representation. Decompression reconstructs the file autoregressively from a zero hidden state seeded with the first byte.

**Active file**: `examples/randcompress_v2.py`. v1 is kept for reference. The `randcompress/` package and other `examples/` scripts are earlier experiments.

## Running

```bash
uv sync
uv run python examples/randcompress_v2.py
```

Logs: `train_v2_persistent.log` (training + validation each epoch), `eval_v2_final.log` (post-training evaluation on full train set).

Datasets:
```
datasets/quran-uthmani.txt      # training
datasets/surat_al-fatihah.txt   # validation (al-Fatihah, ~300 bytes, fits in one forward pass)
```

No test suite. `pytest` is a dev dep but unused.

## Architecture (`randcompress_v2.py`)

### Compression scheme

- **Base xLSTM** (`xLSTMParams`): structured-randomly initialized, fully frozen. Seed + structured init rules fully determine it — no need to store weights.
- **HiRA adapters** (`RandCompressParams`): only trainable parameters. Update rule: `W = W₀ + W₀ ⊙ (B·A)`. B zero-init (ΔW=0 at start), A orthonormal rows (SVD init). Adapters on: embedding, all mLSTM/sLSTM weight matrices, FFN up/down.
- **output_proj**: directly trainable (no HiRA), maps `d_model → vocab_size`.

**Why HiRA over LoRA**: `ΔW = W₀ ⊙ (B·A)` decomposes as `Σₖ diag(bₖ) W₀ diag(aₖ)`, which has `rank = rank(W₀)` even for rank-1 B·A. Same parameter count as LoRA but full-rank updates.

### xLSTM blocks

Each block: `LayerNorm → [mLSTM or sLSTM] → residual → LayerNorm → gated FFN → residual`. Layer type set by `block_map` (e.g. `"msms"`).

**mLSTM** — matrix memory `C ∈ [B, NH, DH, DH]`. Training uses parallel attention-like computation over the chunk. Eval steps recurrently. Has per-head gate biases `b_f [NH]` and `b_i [NH]` for multi-timescale control.

**sLSTM** — scalar states `(y, c, n, m) ∈ [B, H]`. Always `jax.lax.scan` (training and eval identical). Has explicit recurrent connection `Ry`.

### Structured frozen initialization (v2)

The frozen basis is initialized to maximize diversity and signal richness:

| component | init strategy | theory |
|---|---|---|
| `W_q, W_k, W_v` | orthonormal rows (QR), reshaped to `[NH, DH, inner]` | each head attends to an orthogonal subspace (observability) |
| `b_f [NH]` (mLSTM forget bias) | `logit(exp(-1/τ))`, τ log-uniform in `[1, max_seq_len]` | multi-timescale memory, head 0 = fast, head NH-1 = full-window |
| `b[H:2H]` (sLSTM forget bias) | same log-uniform spread over H units | multi-timescale for sLSTM |
| `Ry` (sLSTM recurrent) | SVD → all singular values = 0.95 | near-critical (ESN principle): max memory capacity, guaranteed stability |
| `conv_w` | DCT-II basis tiled across channels | structured FIR filter bank: DC, fundamental, 2nd, 3rd harmonic |
| `W_up, W_down, ffn_*` | row-normalised or orthonormal rows | flat singular value spectrum, no dead input directions |

Helper functions: `_ortho(key, n, m)`, `_fir_bank(n_channels, K)`, `_multiscale_forget_bias(n, seq_len)`.

### HiRA initialization (v2)

`init_hira_adapter`: A rows are right singular vectors of a random matrix (via SVD). This gives exact row orthonormality so each HiRA direction produces an independent gradient signal to B from step one. B remains zero.

### Training: Persistent TBPTT with gradient accumulation

File split into non-overlapping chunks of `chunk_size` tokens. State carried across chunks via `stop_gradient`. State reset to zeros at epoch start (matching eval).

**Gradient accumulation** (`grad_accum_k=K`):
- K chunks processed with params fixed (zero intra-block staleness)
- Gradients summed and averaged, one param update per block
- K=1 is the base case (standard per-chunk update)
- Compared to standard TBPTT (K=1): intra-block staleness goes from 0→K-1 updates to 0; inter-block staleness is K updates (same as before)

Key functions:
- `forward_train_chunked` — one chunk forward from initial layer state
- `grad_chunk_persistent` — gradients only, no update (used for accumulation)
- `adamw_update` — hand-written AdamW with grad clipping before moment updates

### Validation (per epoch)

After each epoch, on `surat_al-fatihah.txt`:
1. **Teacher-forced BPC**: single `forward_train` call (file fits in one chunk)
2. **Generative**: seed = first byte, zero states, greedy decode. Reports accuracy, bytes wrong, first wrong byte index, decoded text.

### Final evaluation (post-training)

On full training dataset:
1. **Teacher-forced BPC**: chunked stateful forward (same as training)
2. **Generative**: seed = `tokens[0]`, generate `n_real - 1` bytes, compare to `tokens[1:]`. Reports bytes wrong, first wrong byte index, accuracy, CER.

Logged to `eval_v2_final.log`.

## Key design decisions

- **Goal is overfitting**: weight_decay=0, no dropout. Everything pushes toward lower train loss.
- **Eval always uses zero hidden state**: training carry-over states are never used at eval time; eval always starts fresh from token[0] with zero states.
- **`PAD_TOKEN = 0`** masked from loss — safe because dataset is Arabic UTF-8 (never produces 0x00).
- **Cosine LR with warmup**: `global_step` increments per chunk, so the LR schedule is identical at K=1 and K>1 (same total decay curve, just fewer actual updates).
- **No optax**: AdamW hand-written for full control and no dependency.
- **DTYPE = float32** throughout (bfloat16 disabled; swap in DTYPE line if needed).

## Staleness analysis (TBPTT)

The carry-over hidden state passed to chunk t+1 was computed with params from chunk t. Staleness = how many gradient updates old the state is.

| strategy | intra-block staleness | inter-block staleness |
|---|---|---|
| standard TBPTT (K=1) | 0 (trivially) | 1 update per chunk |
| grad accumulation (K>1) | **0** (params frozen within block) | K updates at block boundary |
| K-chunk EM (double-pass) | 0→K-1 (grows) | ~0 (last step fresh) |

xLSTM's forget gates attenuate stale state in ~50-100 tokens, so inter-block staleness is self-correcting. Intra-block staleness is the more persistent problem — grad accumulation eliminates it.
