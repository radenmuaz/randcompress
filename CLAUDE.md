# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Neural compression via memorization: freeze a randomly-initialized model, then train only adapters (HiRA or LoRA) to overfit a single file. The adapter weights *are* the compressed representation. Decompression reconstructs the file autoregressively using range coding.

**Active package**: `randcompress/` (PyTorch, frozen-base + HiRA/LoRA adapters). Old JAX experiments are in `examples_old/`.

**Second active package**: `overfitter/` — no frozen base / no adapters, no random-seed reconstruction. The whole model is trained directly to overfit one file. **`overfitter/` is now ByteFractalGen (FractalAR)**, a byte-level port of [FractalGen](https://arxiv.org/html/2502.17437v2) — a recursive generator invoked on progressively smaller sub-patches, giving `O(L)` total attention instead of `O(L²)`. See `## Package: overfitter/` below and `overfitter/README.md`.

**Superseded (KIV, moot)**: the original `overfitter/` design — `SummTransformer`, a hierarchical-summarization transformer (windowed byte-level attention + pooled-KV "summary" stages) — has been archived to `overfitter_v1/` and is no longer maintained or developed. The `## Package: overfitter/` and "Reference run" sections below still describe `SummTransformer`/`overfitter_v1` details (Ks gotchas, share_lm ablations, etc.) — kept for historical reference only, not applicable to the current `overfitter/` (FractalAR) package.

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

**`logs/datasets/`** — larger/external datasets, git-ignored (not checked in, unlike `datasets/` above). Currently holds `enwik8` (100,000,000 B, gunzipped from the committed-nowhere `enwik8.gz`). `logs/` in general is the git-ignored, in-repo location for both experiment logs (`logs/overfitter/...`, see below) and these larger datasets — use it instead of `/tmp` so results/data persist across sessions and `tail -f` works normally, but never `git add` anything under it.

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

## Package: `overfitter/` (SummTransformer, full-model overfit)

No frozen base, no HiRA/LoRA adapters, no seed-derived reconstruction — the whole model
(`SummTransformer` in `overfitter/summformer.py`, a hierarchical-summarization transformer:
windowed byte-level attention + a cascade of pooled-KV "summary" stages for long range) trains
directly to overfit one file. Full details, architecture notes, and the bundle format are in
`overfitter/README.md` — read that first for anything overfitter-related.

### Running

```bash
uv run python -m overfitter.train \
    --dataset datasets/juz1.txt --log_dir runs/overfitter/juz1 \
    --Ks 16,16,16 --d_model 24 --n_layers 8 --n_heads 4 --n_kv_heads 4 \
    --attn_window 64 --share_lm true --tbptt_chunk_size 512 --epochs 8
uv run python -m overfitter.compress --ckpt runs/overfitter/juz1 --input datasets/juz1.txt --output runs/overfitter/juz1_compressed
uv run python -m overfitter.decompress --bundle runs/overfitter/juz1_compressed --output /tmp/recovered.txt --verify datasets/juz1.txt
```

`train.py` writes `<log_dir>/train.log` with eager flush (mirrors `randcompress/train.py`'s own `_Tee`) — **`tail -f <log_dir>/train.log`** to watch a run live, including tqdm's progress bar. Don't rely on tailing a background shell's own captured stdout if it's piped through something like `| tail -N` — that buffers until the process exits; the log file is the thing that streams.

### `Ks` gotchas (read before picking values)

- `n_fuse = len(cfg.Ks)` — every element is a real pooling stage, **no trailing placeholder** (unlike this repo's other `Ks=(32,32,1)`-style convention elsewhere — overfitter's `Config.Ks` was deliberately changed to not need one).
- **A stage whose cumulative product (`Ks[0]*Ks[1]*...`) approaches or exceeds the training file length never fires**, and its query backlog in `_make_incremental_stepper` grows unboundedly for the entire run — this caused a real MPS OOM (7GB+ on a ~340K-param model) during development. Rule of thumb: `Ks[0] ≈ sqrt(L / attn_window)` balances the byte-level pass (linear cost) against stage-0's self-attention (quadratic in block count).
- **Check before training**: `uv run python -m overfitter.analyze --Ks ... --d_model ... --n_layers ... --attn_window ... --dataset <file> --chunk_size ... --epochs ...` prints KV-cache memory, per-epoch FLOPs, and explicitly warns if any stage is starved *or* backlog-spikes (see next point) — no training run required.
- **Deep `Ks` (many stages) vs `tbptt_chunk_size` — a second, distinct pathology from starvation**: even a stage that eventually fires can still crash the run if its `cum_K` exceeds `tbptt_chunk_size`. Queries "backlog" (`x_in_backlog` in `_make_incremental_stepper`) for up to `~cum_K` positions while waiting for that stage to fire, then the *entire* backlog gets pushed through cross-attention and the refinement pass's byte-level attention **in one call** — and `attn_window` only bounds the *persistent* cache between calls, not that single call's own `Tn × (Tn+window)` attention matrix. Real repro: `Ks=2` × 14 stages (deepest `cum_K=16,384`), `tbptt_chunk_size=512`, `attn_window=4` → MPS OOM at ~7.5GB from one `[1,4,16384,16389]` score tensor. Fix: keep every stage's `cum_K ≤ tbptt_chunk_size` (or grow `chunk_size`) — `analyze.py` now checks and warns about this specifically (separate from the starvation warning).

### Key implementation facts (see `overfitter/README.md` for more)

- Stage-level summary self-attention is real incremental KV-caching (not recompute-from-scratch) — this was a real cubic-cost bug, fixed; verify any future change to `_make_incremental_stepper` against `model.check_kv_cache_consistency(...)`, which must stay `match_rate=1.0`.
- `compress.py` prints both a theoretical `CE bpb` (from the model's raw softmax, cheap) and the confirmed `rc bpb` (from the actual range-coded stream, verified via a round-trip decode before printing) — the RC number is always the one that matters, CE is just a sanity check.
- `share_lm=True` (tie every level's Block stack together) is the strongest lever for both size and quality at small scale — an ablation on `juz1.txt` found it alone beat off/off and off/on/on on both loss and accuracy; `share_fuse=True` alone hurt quality noticeably. Don't assume this generalizes without re-ablating for a new config regime.
- Ablatable knobs beyond `share_lm`/`share_fuse`: `pos_scheme` (`rope`|`none`, NoPE), `mlp_type` (`swiglu`|`mlp`|`none`), `use_bias`, `norm_type` (`rmsnorm`|`layernorm`), `qk_norm`, and `optimizer` (`adamw`|`sinkgd_lm`|`sinkgd_all`, see below) — added specifically to explore "fast overfitting" architecture variants (generalization is *not* a goal here, see the brainstorm note this section is a summary of).
- **`optimizer` (`TrainConfig`, default `sinkgd_lm`)**: SinkGD (`randcompress/train.py`'s hand-written Sinkhorn-normalized GD, ported to `overfitter/train.py`) for attention/MLP matrix weights; AdamW carved out for `embed`/`head`/`extra_heads` since Sinkhorn row/col normalization assumes matrix structure a per-symbol embedding/readout table doesn't have as naturally (byte frequencies are Zipfian, not balanced). `sinkgd_all` matches what `randcompress/train.py` actually does today (SinkGD uniformly, no embed/head special-casing) — useful as an explicit ablation arm, not assumed better or worse a priori. `adamw` is the uniform baseline.
- **Experiment workflow: one job at a time, not parallel background batches.** Earlier ablations in this project ran 4 configs concurrently in one shell loop — fine for quick small-file checks, but for anything running long enough to matter (full `juz1.txt`-scale, `Ks`-deep configs) run them sequentially, one at a time, waiting for each to finish before starting the next. Keeps MPS memory pressure predictable and keeps each run's log unambiguous to `tail -f`.

## Reference run: full quran-uthmani.txt (1,359,946 B) — old JAX xLSTM/msrnn

`log_old/quran/` (`examples_old/old/randcompress_v4.2_quran.py`) is a **verified successful**
whole-file train→compress→decode round trip on the full Quran file, no chunk/segment split —
one continuous sequence scanned front-to-back, `max_iters=100` (100 full passes over the file),
JAX `remat=True, remat_chunk=50000` (gradient checkpointing) for memory. Final result
(`log_old/quran/ckpt_last/compression.json`, `decode_ok: true`, `T_valid=1359945` matches the
full file):

```
CE=1.8620 bpb   rc=1.8155 bpb (308,631 B)   ratio=3.334x (407,959 B)   argmax_acc=56.7%
```

~20 min for the 100 training passes, ~10–16 min per full encode/decode eval (that step does a
Python per-token loop). Beats raw `store` but not gzip (5.09x on the same file, see
`compression_results/benchmark_results.json`) — a real but not dramatically strong ratio; useful
as a baseline/sanity target, not a SOTA claim.

**Why it could train on the whole file directly**: msrnn/xLSTM is a *linear RNN* — state
recurrence is `O(L)`, so gradient checkpointing alone bounds memory. This does **not** carry
over to attention-based models (e.g. `overfitter/summformer.py`): attention is `O(L²)`, and
`torch`'s masked SDPA still materializes the full `L×L` score matrix even under a windowed mask
(no true sliding-window kernel), so checkpointing doesn't save it — that needs real bounded
attention (windowed KV cache, TBPTT-chunked training) instead. See `overfitter/README.md`.

Current PyTorch codebase's own attempts at the full file, for comparison:
- `runs/quran_compressed`/`quran_compressed2`: `ratio=1.49x, ce_bpb=6.5, argmax_acc=36%` — weak/undertrained.
- `runs/quran_msrnn/train.log`: segment-curriculum trainer (1024-byte segments) never finished —
  log stops partway through segment 1 of 1329.
