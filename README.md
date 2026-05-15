# randcompress

Neural compression via memorisation: freeze a randomly-initialised linear RNN,
train only HiRA adapters to overfit a single file. The adapter weights *are* the
compressed representation. Decompression reconstructs the file autoregressively
using range coding.

## Quick start

```bash
uv sync
uv run python examples/linear_rnn_srp.py          # surat al-Fatihah (562 B)
uv run python examples/linear_rnn_srp_quran.py    # quran-uthmani (~1.4 MB)
```

Decompress a saved checkpoint:

```bash
uv run python examples/linear_rnn_srp_decompress.py \
    --bundle log/linear_rnn_srp/ckpt_last \
    --output recovered.bin \
    --verify datasets/surat_al-fatihah.txt
```

## Architecture

- **Frozen base**: randomly-initialised sRNN + linear mLSTM (gate-free, power-2
  attention kernel). Seed + structured init fully determines weights — no storage.
- **HiRA adapters** (trainable): `W = W₀ + W₀ ⊙ (B·A)`. B zero-init, A
  orthonormal rows. Full-rank update with same parameter count as LoRA.
- **Range coding**: teacher-forced logits → quantised CDFs → arithmetic encode.
  Decoder runs the model autoregressively to reproduce CDFs at each position.

## Scaling guide

Benchmarked on Apple M-series CPU (Metal JAX backend). Content assumed to be
UTF-8 text or source code; binary data yields 5–8 bpb regardless of model size.

| File size | Config | Params | Param % | Train¹ | Fwd pass² | BPB³ | Ratio⁴ |
|----------:|:-------|-------:|--------:|-------:|----------:|-----:|-------:|
| 100 KB | d32-sm-r2 | 37 KB | 38 % | ~1 min | < 1 s | ~2.2 | ~1.5× |
| 500 KB | d64-sm-r4 | 81 KB | 17 % | ~8 min | 4 s | ~1.5 | ~2.8× |
| **1.4 MB** | **d64-sm-r4** | **81 KB** | **6 %** | **~17 min** | **12 s** | **~1.2** | **~4.7×** |
| 5 MB | d128-smm-r8 | 220 KB | 4.5 % | ~2 hr | 3 min | ~1.2 | ~5× |
| 10 MB | d128-smm-r8 | 220 KB | 2.3 % | ~3 hr | 6 min | ~1.2 | ~5.8× |
| 50 MB | d256-smmm-r16 | 736 KB | 1.5 % | ~2 days † | 2 hr † | ~1.0 | ~7× |
| 100 MB | d256-smmm-r16 | 736 KB | 0.8 % | ~5 days † | 4 hr † | ~1.0 | ~7.5× |
| 500 MB | d512-smmm-r32 | 2.4 MB | 0.5 % | impractical† | 3 days † | ~0.9 | ~8.5× |

**1.4 MB row is measured** (quran-uthmani.txt, 20 k iters). All others estimated.  
† **GPU required** — Metal/CUDA ≈ 20–50× faster; divide times by ~30.

**Footnotes**

¹ **Train** — time to reach target BPB. Epochs: 80 / 40 / 30 / 15 / 10 / 10 / 10 / 5
  (smaller files need more passes to overfit). Speed model: cost ∝ d² (sRNN
  dominates), giving ≈ 160 k / 40 k / 10 k / 2.5 k / 625 tok/s for d = 32 / 64 /
  128 / 256 / 512.

² **Fwd pass** — one `collect_probs` run over the full file, done once per
  checkpoint. Same d² speed scaling.

³ **BPB** (bits per byte) after training. Binary or high-entropy content stays near
  8 bpb regardless of model size.

⁴ **Ratio** = file / (params + range-coded stream), where stream ≈ file × BPB / 8.
  Minimum useful file size is where params < 10 % of file (≈ 800 KB for the
  smallest config). Below that, params alone approach the file size and compression
  is not meaningful.

**`num_heads` recommendation**: keep DH = d\_model / num\_heads = 16 fixed by
scaling H with d (H = 2 / 4 / 8 / 16 / 32 for d = 32 / 64 / 128 / 256 / 512).
This holds D\_sym = 136 constant and prevents the power-2 feature map from
blowing up at large d.

## Config fields (key)

| Field | Default | Meaning |
|:------|--------:|:--------|
| `d_model` | 64 | Hidden dimension |
| `num_heads` | 8 | Attention heads (keep DH = d/H = 16) |
| `block_map` | `"smmm"` | Layer types: `s`=sRNN, `m`=mLSTM |
| `lora_r` | 4 | HiRA adapter rank |
| `segment_size` | 1024 | Tokens per TBPTT chunk |
| `grad_accum_steps` | 1 | Chunks per gradient update (K-step BPTT window) |
| `sinkgd_l` | 5 | Sinkhorn iterations in SinkGD optimiser |
| `max_eval_tokens` | 0 | 0 = eval full file; N = eval first N tokens (fast monitoring) |
| `remat` | True | True = chunked range-code verify (O(chunk×V) memory); False = single-shot |
| `remat_chunk` | 50000 | Tokens per remat chunk |
| `save_ckpt` | True | Save full compressed bundle at each checkpoint |
| `check_every` | 100 | Eval interval (iters) |

## Dataset

Quran text used for benchmarking:
<https://data.mendeley.com/datasets/9yvrzxktmr/2>
