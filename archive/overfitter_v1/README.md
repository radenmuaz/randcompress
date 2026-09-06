# overfitter

Neural compression via memorization, using `SummTransformer` (`summformer.py`): a
hierarchical-summarization transformer with windowed byte-level attention plus a
cascade of pooled-KV "summary" stages for long-range context. Unlike `randcompress/`
(frozen random base + trained HiRA/LoRA adapters), here the **whole model is trained**
to overfit a single file. The trained weights *are* the compressed representation;
range coding removes the remaining redundancy the model's predictions didn't capture.

## Pipeline

```
train.py       overfit the whole file (single causal sequence, plain AdamW) -> model.pt
compress.py    teacher-forced logits -> quantized CDFs -> range-coded bundle
decompress.py  autoregressive range-decode, mirroring compress.py exactly -> original file
```

## Usage

```bash
# 1. Train: overfit a model to one file
uv run python -m overfitter.train \
    --dataset datasets/surat_al-fatihah.txt \
    --log_dir runs/overfitter/fatihah \
    --Ks 32,32 --d_model 128 --n_layers 4 --n_heads 4 \
    --mtp_heads 4 --steps 3000 --check_every 100

# 2. Compress: model + file -> bundle (model weights + range-coded residual)
uv run python -m overfitter.compress \
    --ckpt runs/overfitter/fatihah \
    --input datasets/surat_al-fatihah.txt \
    --output runs/overfitter/fatihah_compressed

# 3. Decompress: bundle -> reconstructed file (with optional verification)
uv run python -m overfitter.decompress \
    --bundle runs/overfitter/fatihah_compressed \
    --output /tmp/recovered.txt \
    --verify datasets/surat_al-fatihah.txt
```

`train.py` stops early once it hits 100% byte accuracy, then runs
`model.check_kv_cache_consistency(...)` as a sanity check — this must report
`match_rate=1.000`, confirming the incremental KV-cache path that `compress`/`decompress`
rely on is bit-exact with full recompute (see note below).

## Key flags (`--Ks`, `--mtp_heads`, ...)

All of `summformer.Config` is exposed via CLI flags in `config.py` (`--Ks`, `--d_model`,
`--n_layers`, `--n_heads`, `--attn_window`, `--fuse_window`, `--mtp_heads`, ...) — see
that file for the full list and defaults. Vocab is fixed at 256 (byte-level only).

- **`--Ks`**: comma-separated cumulative pooling periods for the summarization cascade,
  e.g. `32,32` pools bytes into 32-byte summaries, then those into 32×32-byte summaries.
- **`--mtp_heads`**: multi-token-prediction heads (`k`). At each block start the model
  predicts `k` bytes ahead from one hidden state (main head + `k-1` extra heads), so
  compress/decompress advance the KV cache in blocks of `k` real bytes instead of one
  byte at a time — fewer incremental forward calls per file. `k=1` is plain
  next-byte AR.

## Bundle format

```
ckpt_dir/            (from train.py)
  model.pt           full model state_dict
  config.json         {"model": ModelConfig fields, "train": TrainConfig fields}

bundle_dir/           (from compress.py — copy of ckpt_dir plus:)
  rc_stream.bin       range-coded byte stream
  meta.json           {n_raw_bytes, seed_byte, rc_bytes, rc_stream_sha256, param_bytes}
```

The first raw byte of the file is stored uncompressed (`seed_byte`) as the seed for
autoregressive decoding — everything else is range-coded against the model's
predictions.

## Implementation notes

- `codec.py` / `rc_codec.c` — the range coder itself, unchanged and model-agnostic
  (shared conceptually with `randcompress/`, see its own docs).
- `compress.py`/`decompress.py` use `SummTransformer._make_incremental_stepper`
  directly (the real KV-cache path, not `_cascade`'s O(L²) full recompute) — see the
  `step()` docstring/comment in `summformer.py` for a fix applied there: the
  incremental path used to apply an extra, never-trained `ln_out` RMSNorm
  (`FuseStage.readout`) that training's `forward()` never applies, which would have
  made compression quality diverge from the trained model. It now reuses `self.head`
  directly, matching training exactly at every position.
