# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Neural compression via memorization: freeze a randomly-initialized model, then train only adapters (HiRA or LoRA) to overfit a single file. The adapter weights *are* the compressed representation. Decompression reconstructs the file autoregressively using range coding.

**Active package**: `randcompress/` (PyTorch). Old JAX experiments are in `examples_old/`.

## Running

```bash
uv sync
uv run python examples/train_msrnn.py                     # default: datasets/juz1.txt
uv run python examples/train_msrnn.py \
    --dataset datasets/surat_al-fatihah.txt \
    --max_iter_per_phase 2000 --check_every 100
uv run python examples/train_msrnn_baseline.py            # plain LoRA + AdamW ablation

# Compress + decompress round-trip
uv run python examples/compress.py \
    --ckpt runs/train_msrnn/ckpt_last \
    --input datasets/surat_al-fatihah.txt \
    --output runs/compressed
uv run python examples/decompress.py \
    --bundle runs/compressed \
    --output /tmp/recovered.bin \
    --verify datasets/surat_al-fatihah.txt
```

Logs land in `--log_dir` (default `runs/<script_name>/`). `train.log` captures stdout + stderr (including tqdm) with eager flushing — safe to `tail -f`.

Datasets:
```
datasets/juz1.txt               # default (~44 KB)
datasets/surat_al-fatihah.txt   # tiny (~562 B), fast iteration
datasets/quran-uthmani.txt      # full quran (~1.4 MB)
```

No test suite.

## Package structure (`randcompress/`)

```
randcompress/
  config.py        ModelConfig + TrainConfig dataclasses; parse_configs() argparse CLI
  tokenizer.py     bytes_to_tokens, tokens_to_bytes, ByteTokenizer, load_bytes
  codec.py         ctypes binding to rc_codec.c: quantize_cdf, rc_encode, rc_decode
  train.py         CurriculumTrainer, loss fns (CE + AGD), SinkGD + AdamW + SGD
  compress.py      encode(): teacher-forced logits → CDF → rc_encode → bundle
  decompress.py    decode(): rc_decode AR loop → tokens → bytes
  checkpoint.py    save/load checkpoint and full bundle (config+params+rc_stream+meta)
  models/
    base.py        RandCompressModel ABC
    hira.py        apply_hira, apply_lora, apply_adapter, make_adapter_params, center_b
    msrnn.py       MsRNN — linear mLSTM + whitened sRNN (active model)
    __init__.py    MODEL_REGISTRY, get_model(mcfg, tcfg)

rc_codec.c         C range coder (compiled to /tmp/*.so on first use)
examples/          Clean usage scripts (one per setup)
examples_old/      Old JAX monolithic scripts (reference only)
```

## Architecture: MsRNN (`models/msrnn.py`)

### Compression scheme

- **Frozen base** (`init_frozen`): structured-randomly initialized, never updated. Seed + init rules fully determine it — not stored.
- **Adapters** (`init_adapters`): only trainable parameters.
  - **HiRA** (`use_hira=True`, default): `W = W₀ + W₀ ⊙ (B·A)` — full-rank update from rank-r factors. B zero-init, A orthonormal rows.
  - **Baseline** (`use_hira=False`): `W = W₀ + B·A` — plain LoRA delta, same rank r, same param count.
- **output_proj**: always a direct trainable parameter, no adapter wrapper.

### Cell types (set by `block_map`, one char per layer)

**`m` — linear mLSTM** (no gates, no sigmoid/tanh/exp at runtime):
- State: `C ∈ [B, NH, D_sym, DH]`
- Update: `C_t = α[:,None,None] * C_{t-1} + phi_k ⊗ v_t`, output `h = phi_q @ C`
- `phi` = symmetric degree-`power_p` feature map (`_spow`). p=2 default → `D_sym = C(DH+1,2)`
- Per-head decay `α[NH]` log-uniform in `[1, segment_size]`, frozen
- Scan: pre-projects Q/K/V for full sequence, then Python loop over time (einsum ops batched)

**`s` — whitened sRNN**:
- State: `h ∈ [B, d_model]`
- Update: `h_t = LayerNorm(W_hh h_{t-1} + W_hx x_t + b)`
- `W_hh` near-critical SVD init (σ=0.95), `W_hx` orthonormal
- Scan: pre-projects `x @ W_hx.T` for full sequence, then Python loop for recurrent part

### Structured frozen init

| component | strategy |
|---|---|
| `W_q, W_k, W_v` (mLSTM) | orthonormal rows, `[NH, DH, d]` |
| `alpha [NH]` (mLSTM) | `exp(-1/τ)`, τ log-uniform `[1, seq_len]` |
| `W_hh` (sRNN) | SVD init, all singular values = 0.95 |
| `W_hx` (sRNN) | orthonormal |
| `W_down` (mLSTM) | orthonormal rows / d |
| `embedding` | N(0, 0.01) |

### HiRA B-centering

After each SinkGD step, `center_b(adapters)` zero-centers columns of every `.B` tensor in-place. This maintains zero-mean activations, which is required for SinkGD's Sinkhorn normalization to be exact.

### Stride map

`stride_map` (one digit per layer) controls temporal subsampling. A layer with stride N fires every N tokens; its output is `repeat_interleave`-ed back to full resolution.

## Training (`train.py`)

### Loss functions

- `cross_entropy_loss(logits, targets, pad_token)` — standard masked CE
- `agd_loss(logits, targets, pad_token, detach_weights)` — Arithmetic Gradient Descent: weights each position by `1/I_t` where `I_t = ∏_{s≤t} P(y_s)`. Up-weights hard early tokens. **Slower to overfit** (~5-9× vs CE) — CE is better for this memorization task.
- `training_loss(logits, targets, tcfg)` — dispatches via `tcfg.use_agd_loss`

### Optimizers (`--optimizer` flag)

1. **`sinkgd`** (default): `sinkgd_step()` — global grad clip, then Sinkhorn row/col normalization per tensor (`_sinkhorn_2d`, `L` iterations), then `p -= lr * norm_g`. B-centering applied after. No momentum state.
2. **`adamw`**: `torch.optim.AdamW`
3. **`sgd`**: `torch.optim.SGD` with momentum

### Curriculum TBPTT (`CurriculumTrainer.run`)

File split into `segment_size`-byte segments. Per segment:
1. **SOLO**: train on this segment until 100% argmax accuracy or `max_iter_per_phase` iters
2. **COMBINED**: train on all segments seen so far

State resets to zeros at each epoch boundary. State carried across chunks via `.detach()` (TBPTT). Eval (`eval_segments_stateful`) uses teacher-forced chunked forward + argmax accuracy.

## Config (`config.py`)

Two dataclasses, all fields CLI-overridable as `--field value`:

**`ModelConfig`**: `model`, `d_model`, `num_heads`, `num_layers`, `block_map`, `stride_map`, `lora_r`, `use_hira`, `rope_scale` (transformer), `power_p`, `seed`

**`TrainConfig`**: `dataset`, `log_dir`, `input_bits`, `output_bits`, `output_heads`, `vocab_size`, `pad_token`, `segment_size`, `max_iter_per_phase`, `check_every`, `learning_rate`, `weight_decay`, `grad_clip_norm`, `optimizer`, `sgd_momentum`, `sinkgd_l`, `use_agd_loss`, `agd_detach_weights`, `gen_seed_len`, `residual_budget`, `dtype`

`--no_hira` sets `use_hira=False`.

## Range codec (`codec.py` + `rc_codec.c`)

- `rc_codec.c` compiled on first import to `/tmp/rc_codec_<sha1>.so` via ctypes
- `quantize_cdf(logits_1d)` → `(V+1,)` int32 cumfreqs summing to 65536
- `rc_encode(symbols, cdfs)` → `bytes`; `rc_decode(stream, cdfs)` → `(T,)` int32
- CDF layout: `cdfs[t, 0]=0`, `cdfs[t, V]=65536`, strictly increasing

## Bundle format (compress/decompress)

```
ckpt_dir/
  config.json      ModelConfig + TrainConfig fields
  params.pkl       {name: np.ndarray} adapter weights
  rc_stream.bin    range-coded byte stream
  meta.json        {n_raw_bytes, seed_token, rc_bytes, rc_stream_sha256, param_bytes, ...}
```

Decompression: load bundle → reconstruct frozen from seed → AR decode with RC state machine → `tokens_to_bytes`.

## Key design decisions

- **Goal is overfitting**: `weight_decay=0`, no dropout. Everything pushes toward lower train loss.
- **Eval always resets state to zero**: carry-over training states never used at eval time.
- **`pad_token=0`** masked from loss — safe for Arabic UTF-8 (no 0x00 in dataset).
- **PyTorch only**: no JAX, no optax. SinkGD is hand-written.
- **`float32` default** — swap via `--dtype bfloat16` if needed.
