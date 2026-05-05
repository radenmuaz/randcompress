# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Neural compression via memorization: freeze a randomly-initialized xLSTM, then train only HiRA adapters to overfit a single file. The trained adapter weights *are* the compressed representation. Decompression reconstructs the file autoregressively from a zero hidden state seeded with the first byte.

The active development file is `examples/randcompress_v1.py`. The `randcompress/` package and other `examples/` scripts are earlier experiments.

## Running

```bash
# install (uv preferred)
uv sync

# train (edits config inside main())
uv run python examples/randcompress_v1.py

# dataset lives at
datasets/quran-uthmani.txt      # training
datasets/surat_al-fatihah.txt   # validation (al-Fatihah, first surah)
```

No test suite exists. `pytest` is listed as a dev dep but unused.

## Architecture (`randcompress_v1.py`)

### Compression scheme
- **Base xLSTM** (`xLSTMParams`): randomly initialized, fully frozen. Never updated.
- **HiRA adapters** (`RandCompressParams`): the only trainable parameters. Applied as `W_eff = W₀ + W₀ ⊙ (B·A)`. B is zero-initialized (delta=0 at start), A is random. Adapters wrap: embedding, all mLSTM/sLSTM weight matrices, FFN up/down projections, output head.
- **output_proj**: directly trainable (not HiRA-wrapped), maps `d_model → vocab_size`.

### xLSTM blocks
Each block: `LayerNorm → [mLSTM or sLSTM] → residual → LayerNorm → gated FFN → residual`. Block type per layer set by `block_map` string (e.g. `"msms"`).

**mLSTM**: matrix memory state `C ∈ [B, NH, DH, DH]`. During training uses parallel attention-like computation over the full chunk (no within-chunk recurrence). During eval steps recurrently. Train/eval compute different things within a chunk.

**sLSTM**: scalar state `(y, c, n, m) ∈ [B, H]` each. Always runs as `jax.lax.scan` in both training and eval — train and eval are identical computationally. Has recurrent connection `Ry` so previous output explicitly gates the next step.

### Training: Persistent TBPTT
File is split into non-overlapping chunks of `chunk_size` tokens. Within each epoch, chunks are processed in order with recurrent state carried across chunk boundaries via `jax.lax.stop_gradient` (truncated BPTT). State is **reset to zeros at the start of each epoch**. This matches eval, which also uses zero initial state.

Key functions:
- `forward_train_chunked` — forward through one chunk given an initial layer state
- `grad_chunk_persistent` — computes gradients without applying them (used for grad accumulation)
- `train_chunk_persistent` — computes gradients and applies update in one call

### Eval / decompression
`_eval_generative`: seeds with first byte of file, zero states, steps token-by-token with `forward_step` (recurrent), greedily reconstructs. Metric is byte accuracy and CER vs original file.

### Optimizer
Hand-written AdamW in `adamw_update`. No optax. Gradient clipping applied before moment updates.

## Key design constraints

- **Eval always uses zero hidden state**: do not make the model depend on non-zero init states at training time in ways that won't exist at eval.
- **Goal is overfitting**: weight decay = 0, no dropout, no regularization. Everything should push toward lower training loss.
- **Fixed LR** (not cosine decay): equal learning pressure on all chunks across all epochs; Adam's second moment handles per-parameter scaling.
- `PAD_TOKEN = 0` (null byte) is masked out of the loss — safe because dataset is Arabic UTF-8 which never produces 0x00.
