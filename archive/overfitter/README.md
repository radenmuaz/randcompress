# overfitter — ByteFractalGen (FractalAR)

Neural compression via memorization, using a byte-level port of **FractalGen**
(arxiv.org/html/2502.17437v2) — a generative model built by recursively invoking a generator on
progressively smaller sub-patches. The whole model is trained directly to overfit one file; the
trained weights *are* the compressed representation, and range coding removes the remaining
redundancy the model's predictions didn't capture.

The reference authors' original implementation (2D images, continuous RGB pixels) is checked out
for reference at `../fractalgen/`. `overfitter_v1/` is the previous (non-fractal, attention +
hierarchical-summarization) architecture this package replaced — kept for reference, not
maintained.

## Why "fractal"

A length-`L` byte sequence costs `O(L)` total attention, not `O(L²)` for one flat pass, because
every level only attends over a short `seq_len` (`patch_len_list[l] // patch_len_list[l+1]`).
Concretely for `patch_len_list=(1024,128,16,1)`: level 0 attends over 8 positions, level 1 over 8,
level 2 (terminal) over 16 — never over the full 1024-byte sequence at once. Training is
teacher-forced and fully vectorized: each level's sub-patches become the **batch** dimension for
the next level's call, so the whole multi-level recursion is a Python loop over levels (3, here),
not over samples or positions.

Adapted from the original for 1D discrete bytes instead of 2D continuous pixels — see the
docstring at the top of `model.py` for the full rationale (frozen one-hot byte representation,
independent-per-level weights by default, only the terminal level ever produces a softmax).

## Running

### Sanity checks (overparam — params vastly exceed the file being memorized)

These use whatever `d_model`/`n_layers` is convenient and end up wildly overparameterized
relative to the tiny file (e.g. juz1: ~1M params for a 44KB file) — they exist to prove the
pipeline is *correct* (lossless round trip), not to demonstrate real compression. Every run
below produces a bundle *larger* than the original file (`params` alone dwarf it) — see the
"real run" section further down for a config that actually compresses.

```bash
# Tiny sanity check (3 levels, 8-byte chunks)
uv run python -m overfitter.train \
    --dataset datasets/surat_al-fatihah.txt --log_dir logs/overfitter/fatihah \
    --patch_len_list 8,4,2,1 --d_model 32 --n_layers 2 --n_heads 4 --mlp_mult 2 \
    --steps 1500 --log_every 300 --lr 5e-3
uv run python -m overfitter.compress --ckpt logs/overfitter/fatihah \
    --input datasets/surat_al-fatihah.txt --output logs/overfitter/fatihah_compressed --batch_size 8
uv run python -m overfitter.decompress --bundle logs/overfitter/fatihah_compressed \
    --output /tmp/recovered.txt --verify datasets/surat_al-fatihah.txt

# juz1.txt, ~44KB, overparam (~1M params, independent per-level trunks, default byte_embed_dim=256)
uv run python -m overfitter.train \
    --dataset datasets/juz1.txt --log_dir logs/overfitter/juz1 \
    --patch_len_list 1024,128,16,1 --d_model 64 --n_layers 4 --n_heads 4 --mlp_mult 2 \
    --steps 3000 --log_every 300 --lr 3e-3
uv run python -m overfitter.compress --ckpt logs/overfitter/juz1 \
    --input datasets/juz1.txt --output logs/overfitter/juz1_compressed --batch_size 32
uv run python -m overfitter.decompress --bundle logs/overfitter/juz1_compressed \
    --output /tmp/recovered.txt --verify datasets/juz1.txt
```

| | fatihah (562B, overparam) | juz1 (44,443B, overparam) |
|---|---|---|
| passes/chunk | ~21 | ~68 |
| train time | 22.5-25.4s (1500 steps) | 93.7s (3000 steps, 59.9 kB/s) |
| train acc / bpb | 50.0% / 1.802 | 96.4% / 0.142 |
| compress: argmax / CE bpb / rc bpb | 60.4% / 1.674 / 1.779 | 96.0% / 0.172 / 0.179 |
| model / rc / total bytes | 784,384 / 125 / 784,509 | 11,852,800 / 995 / 11,853,795 |
| ACTUAL compression ratio | 0.001x (EXPANDED) | 0.004x (EXPANDED) |
| compress / decompress speed | 7.72 / 8.63 kB/s (batch=8) | 1.45 / 1.50 kB/s (batch=1) |
| decompress verification | PERFECT MATCH | PERFECT MATCH |

### Real run — actual (>1x) compression on the full quran-uthmani.txt (1,359,946B)

Getting `total_bytes < n_raw_bytes` needs the model itself small enough that `param_bytes` has
headroom under the file size, *and* still fits well enough that `rc_bytes` doesn't eat that
headroom back up — shrinking the model helps the first but hurts the second, so this took real
tuning (see the `share_trunk`/`byte_embed_dim` notes below — `patch_in`'s size scales with
`byte_embed_dim`, not `d_model`, and dominated the earlier "just shrink d_model" attempts).

```bash
uv run python -m overfitter.train \
    --dataset datasets/quran-uthmani.txt --log_dir logs/overfitter/quran_real \
    --patch_len_list 1024,128,16,1 --d_model 24 --n_layers 2 --n_heads 2 --mlp_mult 2 \
    --byte_embed_dim 64 --share_trunk true \
    --steps 40000 --log_every 1.0 --lr 3e-3
uv run python -m overfitter.compress --ckpt logs/overfitter/quran_real \
    --input datasets/quran-uthmani.txt --output logs/overfitter/quran_real_compressed --batch_size 32
uv run python -m overfitter.decompress --bundle logs/overfitter/quran_real_compressed \
    --output /tmp/recovered.txt --verify datasets/quran-uthmani.txt
```

| | quran, overparam attempt (999,232 params) | **quran, real run (258,640 params)** |
|---|---|---|
| config | `d_model=24 n_layers=2 n_heads=4`, independent trunks, `byte_embed_dim=256` | `d_model=24 n_layers=2 n_heads=2 share_trunk=true byte_embed_dim=64` |
| passes/chunk | ~30 | ~31 |
| train time | 684.2s (40,000 steps, 59.9 kB/s) | 551.1s (40,000 steps, 74.3 kB/s) |
| train acc / bpb | 76.8% / 0.960 | 74.5% / 0.988 |
| compress: argmax / CE bpb / rc bpb | 73.0% / 1.060 / 1.065 | 72.0% / 1.102 / 1.105 |
| model / rc / total bytes | 3,996,928 / 181,004 / 4,177,932 | **1,034,560 / 187,756 / 1,222,316** |
| **ACTUAL compression ratio** | 0.326x (EXPANDED) | **1.1126x (COMPRESSED)** |
| compress / decompress speed | 23.93 / 27.96 kB/s (batch=32) | 25.27 / 31.03 kB/s (batch=32) |
| decompress verification | PERFECT MATCH | **PERFECT MATCH** |

`train.py` writes `<log_dir>/train.log` with eager flush — `tail -f <log_dir>/train.log` to watch
a run live, including tqdm's progress bar and per-epoch theoretical-bpb estimates (see
`--log_every` below).

### Ablation: `share_trunk` (same config otherwise, full quran-uthmani.txt, 40,000 steps)

| | `share_trunk=true` (real run, above) | `share_trunk=false` |
|---|---|---|
| trainable params (bytes) | 242,256 (969,024 B) | 265,536 (1,062,144 B) |
| train time | 543.9s (73.5 steps/s) | 661.8s (60.4 steps/s) |
| train acc / bpb | 74.51% / 0.988 | **75.20% / 0.971** |
| compress: argmax / CE bpb / rc bpb | 72.0% / 1.1023 / 1.1045 | 72.4% / 1.0871 / 1.0898 |
| model / rc / total bytes | 1,034,560 / 187,756 / 1,222,316 | 1,127,680 / 185,265 / 1,312,945 |
| **ACTUAL compression ratio** | **1.1126x (COMPRESSED)** | 1.0358x (COMPRESSED) |
| decompress verification | PERFECT MATCH | PERFECT MATCH |

Sharing trains *faster* but not to a *better* bpb — the unshared run actually reaches slightly
lower bpb (0.971 vs 0.988) given the same step budget. But `share_trunk` only aliases the trunk
blocks, ~9% of total params (`patch_in` dominates — see `--byte_embed_dim` below), and near the
compression crossover point that ~9% (93KB here) is worth more than the small residual-bpb
difference: 1.11x vs 1.04x. Don't assume this generalizes to configs far from the crossover, where
the bpb difference would dominate instead.

## Key config knobs (`overfitter/model.py`'s `FractalConfig`)

- **`patch_len_list`**: `(1024, 128, 16, 1)` — must end in `1` (byte-atomic terminal level) and
  divide evenly level to level. `n_levels = len(patch_len_list) - 1`.
- **`d_model` / `n_layers` / `n_heads` / `mlp_mult`** (CLI, broadcast uniformly to every level):
  each fractal level gets its **own independent trunk** by default (own attention/MLP weights),
  matching the reference paper's own design (it tapers capacity level to level, e.g.
  `num_blocks_list=(24,6,3,1)` for images). Construct a `FractalConfig` directly (not via the CLI)
  to give levels genuinely different sizes — the CLI's scalar flags are a convenience for the
  common "same size everywhere" case.
- **`--batch_size`** (compress/decompress): chunks are fully independent (no cross-chunk state —
  `root_cond` is a fixed constant), so `batch_size` of them run through the shared trunk in one
  batched forward call per recursion step. This is real parallelism, not just a batching
  convenience: FractalAR's whole appeal for decompression is that unlike a single long
  autoregressive stream, independent chunks can be generated in lockstep. **compress.py and
  decompress.py must use the exact same `batch_size`** — it's saved in `meta.json` and
  decompress.py reads it back automatically. Batched and differently-batched matmuls are not
  bit-identical (floating-point non-associativity), which is fatal for range coding if the two
  sides disagree, so never override `batch_size` by hand on the decode side.
- **`--share_trunk`** (CLI flag, default `false`): opt-in — when `true`, every level must have
  identical `d_model`/`n_layers`/`n_heads`/`mlp_mult` (checked, with a clear error if not — also
  `head_dim = d_model/n_heads` must come out **even**, RoPE's `rotate_half` splits it in two), and
  literally the same trunk module is reused at every level (matches this package's predecessor's
  `share_lm=True`, which was the strongest lever for both size and quality there).
- **`--byte_embed_dim`** (default `256`): the byte representation is **one global, frozen** table
  (exact one-hot when `>=256`, fixed unit vectors otherwise), decoupled from any level's own
  `d_model`. It is never trained — see `model.py`'s module docstring for why (every level but the
  terminal one outputs a *continuous* hidden vector, exactly like the reference feeding raw pixel
  floats with no embedding at all; making this trainable would blur that distinction).
  **This is the real lever for param count, not `d_model`**: each level's `patch_in` projection is
  `Linear(child_len * byte_embed_dim, d_model)` — at the default `byte_embed_dim=256` and
  `child_len=128` (level 0 of `patch_len_list=(1024,128,16,1)`), that's `128*256=32,768` input
  features *before* `d_model` even enters the picture, dwarfing the trunk itself at small
  `d_model`. Shrinking `d_model` alone barely moves total params until `byte_embed_dim` also comes
  down (see the "real run" table above: dropping `256→64` was what actually got the model under
  the file size, not the `d_model`/`share_trunk` changes alongside it).
- **`--log_every`**: an int (`100`) means print current step's `bpb`/`acc`/`loss` every N steps;
  a float (`0.5`) means every that fraction of an epoch (`n_chunks` steps = one epoch), rounded to
  the nearest step count — `1.0` is one full epoch, not to be confused with the literal int `1`
  (every single step — a real, different, much noisier setting; passing `1` prints a warning since
  it's almost never intended). The tqdm bar's postfix (`loss`/`acc`/`bpb`) always updates every
  step regardless of `log_every` — it's free, no extra forward pass, just formatting the metrics
  already computed for that step.
  Separately, at every actual epoch boundary (`step % n_chunks == 0`, unconditional, not gated by
  `log_every`), `train.py` also prints a **theoretical, RC-free compression estimate** straight
  from that epoch's training loss (`[epoch ~N]  CE~X.XXXXbpb (theoretical)  est size
  ~YB (rc) + ZB (params) = TB  est ratio ~R.RRRRx`) — no `rc_encode()` call, just the CE bits
  already computed during the forward pass, scaled to the full file. This tracked the real,
  post-`compress.py` numbers closely in testing (within ~2.5% on bpb, ~0.1% on ratio) — cheap
  enough to watch every epoch without waiting for a real compress run.

## Correctness discipline: `collect_logits()` and `generate()` must stay one code path

`compress.py`'s CDF collection (`collect_logits`) is implemented as a thin wrapper that calls
`generate()` itself with a teacher-forcing symbol callback, rather than a separate fast batched
computation. This is deliberate: floating-point matmuls aren't perfectly associative, so a batched
training-style computation and the recursive step-by-step generation computation can disagree in
the last bit even though they're mathematically equivalent — enough to desync range coding (a
1-unit difference in a quantized integer CDF corrupts everything after it). Any future change to
either function needs to preserve this — verify with a quick round-trip test (train → compress →
decompress → byte-for-byte compare), not just a loss/logit sanity check.

## Bundle format

```
ckpt_dir/            (from train.py)
  model.pt            full model state_dict
  config.json          FractalConfig fields
  meta.json            {n_raw_bytes}

bundle_dir/           (from compress.py -- copy of ckpt_dir plus:)
  rc_stream.bin        range-coded byte stream
  meta.json            {n_raw_bytes, rc_bytes, param_bytes, total_bytes, ratio,
                         T_valid, n_wrong, argmax_acc, ce_bpb, batch_size}
```

`compress.py` prints an explicit, labeled breakdown at the end — model size, rc-coded residual,
their sum, the original file size, and the actual ratio with a `[COMPRESSED]`/`[EXPANDED]`
verdict (not just a compact `p+rc=` shorthand), and `total_bytes`/`ratio` are saved in
`meta.json` too (mirroring `randcompress`'s own `compression.json` convention).

Padding: the last chunk is zero-padded up to `patch_len_list[0]` if the file length isn't a
multiple of it. Padding bytes are compressed too (deterministic, the model learns to predict
zeros there) and simply trimmed back out of the final decompressed output using `n_raw_bytes`.
