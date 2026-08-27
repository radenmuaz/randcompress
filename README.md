# randcompress

Neural compression via memorisation: freeze a randomly-initialised model, train
only adapters (HiRA or LoRA) to overfit a single file. The adapter weights *are*
the compressed representation. Decompression reconstructs the file
autoregressively using range coding.

**Backend**: PyTorch (CPU / GPU / MPS). JAX experiments are in `examples_old/`.

## Quick start

```bash
uv sync
```

**Compress a file in three steps:**

```bash
# 1. Train — overfit the model to your file (562-byte example, ~2 min on CPU)
uv run python examples/train_msrnn.py \
    --dataset datasets/surat_al-fatihah.txt \
    --max_iter_per_phase 2000 --check_every 200 \
    --log_dir runs/fatihah

# 2. Compress — encode file with trained weights → bundle
uv run python examples/compress.py \
    --ckpt runs/fatihah/ckpt_last \
    --input datasets/surat_al-fatihah.txt \
    --output runs/fatihah_bundle

# 3. Decompress — reconstruct original from bundle alone
uv run python examples/decompress.py \
    --bundle runs/fatihah_bundle \
    --output recovered.bin \
    --verify datasets/surat_al-fatihah.txt
# → Verification: PERFECT MATCH ✓
```

`tail -f runs/fatihah/train.log` streams live training progress (stdout +
tqdm both logged).

## Detailed usage

### Training

```bash
# Default: MsRNN (smmm), d=64, HiRA r=4, SinkGD, juz1.txt (~44 KB)
uv run python examples/train_msrnn.py

# Larger model for bigger files
uv run python examples/train_msrnn.py \
    --dataset datasets/quran-uthmani.txt \
    --d_model 128 --block_map smmmm --lora_r 8 \
    --max_iter_per_phase 5000 --log_dir runs/quran_large

# LoRA baseline (same param count as HiRA, no Hadamard)
uv run python examples/train_msrnn_baseline.py \
    --dataset datasets/surat_al-fatihah.txt

# Transformer, standard context
uv run python examples/train_transformer.py \
    --dataset datasets/juz1.txt --num_layers 4 --d_model 64

# Transformer, ~1M token file (dilated window + sinks, 32K effective context)
uv run python examples/train_transformer_1m.py \
    --dataset datasets/quran-uthmani.txt \
    --max_iter_per_phase 500 --log_dir runs/quran_transformer

# TTT-Linear RNN
uv run python examples/train_ttt.py --dataset datasets/juz1.txt

# Gated DeltaNet
uv run python examples/train_deltanet.py --dataset datasets/juz1.txt

# AdamW instead of SinkGD
uv run python examples/train_msrnn.py --optimizer adamw --learning_rate 3e-4

# AGD loss (arithmetic gradient descent — slower to overfit than CE)
uv run python examples/train_msrnn.py --use_agd_loss true
```

### Compress

```bash
# Standard compress (precomputes adapter weights for speed)
uv run python examples/compress.py \
    --ckpt runs/fatihah/ckpt_last \
    --input datasets/surat_al-fatihah.txt \
    --output runs/fatihah_bundle

# Low-memory: recompute adapters per step (slower, same output)
uv run python examples/compress.py \
    --ckpt runs/fatihah/ckpt_last \
    --input datasets/surat_al-fatihah.txt \
    --output runs/fatihah_bundle \
    --precompute_weights false
```

The compress step prints:
```
  encode... 928B  0.00s
  verify... OK
[compress]  0.7s  (logits=0.6s  cdfs=0.0s  enc=0.00s  dec=0.00s)
  argmax: 561/1024 (54.8%)  CE=1.234bpb  rc=1.318bpb (928B)  p+rc=0.45x
```
`verify... OK` confirms the RC round-trip is lossless before writing the bundle.

### Decompress

```bash
# Decompress and verify against original
uv run python examples/decompress.py \
    --bundle runs/fatihah_bundle \
    --output recovered.bin \
    --verify datasets/surat_al-fatihah.txt

# Decompress without verification (bundle alone is sufficient)
uv run python examples/decompress.py \
    --bundle runs/fatihah_bundle \
    --output recovered.bin
```

### Transformer long-context flags

```bash
# Dilated window: attend every 4th past position, 2048 slots → 32K effective context
uv run python examples/train_transformer.py \
    --kv_window 2048 --attn_dilation 4 \
    --n_sinks_zero 1 --n_sinks_train 1 \
    --rope_scale 4.0

# TPU-friendly strided backend (contiguous slice, no gather)
uv run python examples/train_transformer.py \
    --attn_backend strided --attn_dilation 4

# Multi-phase strided: 2 evenly-spaced dilation grids per layer
# (denser non-uniform coverage, still O(kv_window) — no gather, no mask)
uv run python examples/train_transformer.py \
    --attn_backend strided --attn_dilation 8 --n_phases 2 --kv_window 1024
```

## Model backends

Four frozen-base + adapter models, all sharing the same training loop,
compress/decompress pipeline, and codec.

| key | architecture | state | frozen weights | adapter targets |
|:----|:-------------|:------|:---------------|:----------------|
| `msrnn` | linear mLSTM + whitened sRNN | `C [NH,D_sym,DH]`, `h [d]` | W_q/k/v, W_down, W_hx/hh | same + embedding |
| `transformer` | causal attn (YaRN RoPE) + SwiGLU | KV cache (windowed) | W_q/k/v/o, W_gate/up/down | same + embedding |
| `ttt` | TTT-Linear RNN | `W [d,d]` per layer | W_q/k/v, eta | same + embedding |
| `deltanet` | Gated DeltaNet (vector gate) | `C [NH,DH,DH]` | W_q/k/v/β/o | same + embedding |

Select with `--model msrnn|transformer|ttt|deltanet`.

## Adapter modes

| mode | update rule | flag |
|:-----|:------------|:-----|
| **HiRA** (default) | `W = W₀ + W₀ ⊙ (B·A)` — full-rank update from rank-r factors | *(default)* |
| **LoRA baseline** | `W = W₀ + B·A` — standard low-rank delta, same param count | `--no_hira` |

B is zero-init (ΔW=0 at start), A has orthonormal rows (SVD init).
`output_proj` is always a direct trainable parameter (no adapter).

## Example scripts

```bash
# MsRNN — HiRA + SinkGD (recommended for most files)
uv run python examples/train_msrnn.py

# MsRNN — plain LoRA + AdamW (ablation baseline)
uv run python examples/train_msrnn_baseline.py

# Transformer (default config)
uv run python examples/train_transformer.py

# Transformer tuned for ~1M token files (dilated window, attention sinks)
uv run python examples/train_transformer_1m.py

# TTT-Linear RNN
uv run python examples/train_ttt.py

# Gated DeltaNet
uv run python examples/train_deltanet.py

# Compress trained checkpoint → bundle
uv run python examples/compress.py --ckpt <ckpt_dir> --input <file> --output <bundle_dir>

# Decompress bundle → file
uv run python examples/decompress.py --bundle <bundle_dir> --output <out> [--verify <original>]
```

All training flags can be overridden with `--field value`. Logs go to
`--log_dir` (default `runs/<script_name>/`) with eager tqdm flushing — safe
to `tail -f runs/.../train.log`.

## Compress / decompress

The pipeline is **lossless by construction**: teacher-forced logits → quantised
CDFs → range-encode true symbols → write bundle. Decompressor reconstructs the
same CDFs autoregressively (each decoded symbol equals the true token since RC
is lossless) → range-decode → original bytes.

```
bundle/
  config.json      model + train config
  params.pkl       adapter weights {name: np.ndarray}
  rc_stream.bin    range-coded byte stream (verified SHA-256)
  meta.json        n_raw_bytes, seed_token, rc_bytes, param_bytes, …
  compression.json CE bpb, RC bpb, ratio, argmax accuracy
```

`--precompute_weights true` (default): apply HiRA once before the collect loop
— faster, uses one extra copy of model weights in memory. Set to `false` for
low-memory devices (recomputes adapters per step, same CDFs).

## Optimisers

| flag | algorithm | notes |
|:-----|:----------|:------|
| `sinkgd` (default) | Sinkhorn GD | grad clip → row/col Sinkhorn normalise → update; HiRA B-centering post-step |
| `adamw` | AdamW | `torch.optim.AdamW` |
| `sgd` | SGD + momentum | `torch.optim.SGD`, `--sgd_momentum 0.9` |

## Training: curriculum TBPTT

File split into `--segment_size`-byte segments. Per segment:

1. **SOLO** — train on this segment until 100% argmax accuracy or
   `--max_iter_per_phase` iters
2. **COMBINED** — train on all completed segments until all pass 100%

State resets to zeros at each epoch boundary. State carried across chunks via
`.detach()` (TBPTT). Eval uses teacher-forced chunked forward + argmax accuracy.

## Transformer: long-context options

The transformer supports optional sliding window + dilation for files with
millions of tokens.

**Effective context** stacked over L layers:
```
per layer:  1 + (kv_window - 1) × attn_dilation
stacked L:  1 + L × (kv_window - 1) × attn_dilation
```

**Attention sinks** — always present in the KV, never evicted by the window:
- `--n_sinks_zero N` — constant-zero K/V (score = exp(0) for any query; no parameters)
- `--n_sinks_train N` — learned K/V (query-dependent attraction; parameters in frozen model)

Both types may be combined. They ensure no train/test mismatch: the window only evicts
content tokens, sinks are always at the front of every attention call.

**Attention backends:**

| `--attn_backend` | reads | cost | best for |
|:----------------|:------|:-----|:---------|
| `gather` (default) | scattered | O(W) | CPU / GPU |
| `strided` | contiguous slice | O(W) | TPU (systolic array) |

`strided` reshapes the cache as `[T//d, d, NH, DH]` and slices phase `t % d` —
zero non-contiguous reads. Requires no masking.

`--n_phases N` (strided only): attend to N evenly-spaced phases per layer,
giving non-uniform coverage. Each phase is one contiguous slice; total slots
remain `kv_window`. N=1 is standard dilation; N=4 with d=8 covers 4 independent
grids → denser coverage without gather or mask.

**1M-token recommended defaults** (`examples/train_transformer_1m.py`):

```
kv_window=2048  attn_dilation=4  rope_scale=4.0
n_sinks_zero=1  n_sinks_train=1  attn_backend=gather
→ effective context ≈ 32K tokens  KV memory ≈ 16 MB (d=64, 4 layers)
```

## Key config fields

**Model** (`--field value`):

| field | default | meaning |
|:------|--------:|:--------|
| `model` | `msrnn` | `msrnn` \| `transformer` \| `ttt` \| `deltanet` |
| `d_model` | 64 | hidden dimension |
| `num_heads` | 8 | attention heads (keep DH = d/H = 8 or 16) |
| `num_layers` | 4 | layers (msrnn: overridden by `len(block_map)`) |
| `block_map` | `smmm` | msrnn layer types: `s`=sRNN, `m`=mLSTM |
| `lora_r` | 4 | adapter rank |
| `use_hira` | `true` | HiRA (`true`) or plain LoRA (`false`) |
| `kv_window` | 0 | transformer: 0=full cache, N=sliding window |
| `attn_dilation` | 1 | transformer: attend every d-th position |
| `attn_backend` | `gather` | `gather` or `strided` (TPU) |
| `n_phases` | 1 | strided: phases per layer |
| `n_sinks_zero` | 1 | transformer: constant-zero KV sinks |
| `n_sinks_train` | 1 | transformer: trainable KV sinks |
| `rope_scale` | 1.0 | transformer: YaRN frequency scale |
| `power_p` | 2 | msrnn: mLSTM symmetric power-map degree |

**Training**:

| field | default | meaning |
|:------|--------:|:--------|
| `dataset` | `datasets/juz1.txt` | file to compress |
| `segment_size` | 1024 | bytes per curriculum segment |
| `max_iter_per_phase` | 10000 | SOLO / COMBINED iteration budget |
| `check_every` | 100 | argmax eval interval |
| `learning_rate` | 1e-2 | |
| `optimizer` | `sinkgd` | `sinkgd` \| `adamw` \| `sgd` |
| `sinkgd_l` | 5 | Sinkhorn iterations |
| `use_agd_loss` | `false` | CE loss (`false`) or AGD loss (`true`) |
| `precompute_weights` | `true` | apply adapters once before collect loop |
| `log_dir` | `""` | overrides `runs/<script_name>` |

## Dataset

Quran text used for benchmarking:
<https://data.mendeley.com/datasets/9yvrzxktmr/2>

```
datasets/surat_al-fatihah.txt    # 562 B  — fast iteration / smoke tests
datasets/juz1.txt                # 44 KB  — default training target
datasets/quran-uthmani.txt       # 1.4 MB — full benchmark
```
