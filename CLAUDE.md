# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Neural compression via memorization: train a model to overfit a single file; the trained
weights *are* the compressed representation, and range coding removes the redundancy the
model's own predictions didn't capture. Decompression reconstructs the file autoregressively.

**Active codebase**: `enfrac/` and `enfrac_zero/` — a JAX + [Equinox](https://github.com/patrick-kidger/equinox)
+ [optax](https://github.com/google-deepmind/optax) port of a byte-level
[FractalGen](https://arxiv.org/html/2502.17437v2) (ByteFractalGen/FractalAR): a generative model
built by recursively invoking a generator on progressively smaller sub-patches, giving `O(L)`
total attention instead of `O(L²)`. Current focus: multi-device TPU training and scaling up to
bigger files (enwik8, eventually enwik9) — not architecture research.

**Everything else is archived** (`archive/`) and not maintained: the original PyTorch
`randcompress/` package (MsRNN + HiRA/LoRA adapters), `overfitter/` (SummTransformer,
hierarchical-summarization transformer), `overfitter_peft/` and `overfitter_v1/`, old JAX
experiments (`examples_old/`), and the reference FractalGen checkout (`fractalgen/`). These were
each in turn superseded by what came after — `enfrac`/`enfrac_zero` are direct JAX ports of
`overfitter_peft`/`overfitter` respectively, kept at feature parity with them at fork time. Do
not develop against anything under `archive/`; consult it only for historical context.

## Two variants

| package | architecture | trainable | seed/config drives |
|:--------|:-------------|:----------|:--------------------|
| `enfrac` | HiRA-adapted frozen base: `W = W₀ + W₀⊙(B·A)` per linear | only `B` (+ `root_cond`) | frozen `(W₀, A)` reconstruction |
| `enfrac_zero` | plain linears, no frozen base | everything | nothing special (full checkpoint saved) |

Both share the same recursive architecture (RoPE attention + SwiGLU trunks, frozen
maximally-separated byte embedding, `patch_in`/`cond_proj`/`head` projections), `codec.py` (a
ctypes-bound C range coder — **compress/decompress logic stays C**, unchanged across both
variants and across the JAX port itself), `tokenizer.py`, `config.py`, and the generic
multi-device `train()` loop in `enfrac/train.py` (which `enfrac_zero/train.py` calls directly).
Only the model class (`ModelConfig`/`ByteFractalGen`) and each package's thin CLI differ.

See `enfrac/README.md` for the full architecture writeup (recursion, `patch_len_list`,
`share_trunk`, `byte_embed_dim` tradeoffs, the HiRA determinism contract) — this file covers
operational/config/correctness concerns, not the model internals.

## Running

```bash
uv sync

# Train -> compress -> decompress, HiRA/PEFT variant
uv run python -m enfrac.train --config enfrac/configs/fatihah.py --log_dir logs/enfrac/fatihah
uv run python -m enfrac.compress --config enfrac/configs/fatihah.py \
    --ckpt logs/enfrac/fatihah --input quran_data/surat_al-fatihah.txt --output logs/enfrac/fatihah_compressed
uv run python -m enfrac.decompress --bundle logs/enfrac/fatihah_compressed \
    --output /tmp/recovered.txt --verify quran_data/surat_al-fatihah.txt

# Same shape for the no-HiRA baseline
uv run python -m enfrac_zero.train --config enfrac_zero/configs/fatihah.py --log_dir logs/enfrac_zero/fatihah
uv run python -m enfrac_zero.compress --config enfrac_zero/configs/fatihah.py \
    --ckpt logs/enfrac_zero/fatihah --input quran_data/surat_al-fatihah.txt --output logs/enfrac_zero/fatihah_compressed
uv run python -m enfrac_zero.decompress --bundle logs/enfrac_zero/fatihah_compressed \
    --output /tmp/recovered.txt --verify quran_data/surat_al-fatihah.txt
```

`train.py` writes `<log_dir>/train.log` with eager flush — `tail -f <log_dir>/train.log` to
watch a run live, including tqdm's progress bar.

No test suite.

## Config system

Three layers, later wins: dataclass defaults < `--config <file.py>` < individual `--<field>` CLI
flags. A config file is a plain `.py` module, `exec`'d and searched for `model`/`model_config`,
`train`/`train_config`, and (for `compress.py`) `compress` attributes (each a dict or a
dataclass instance). Unknown field names raise immediately (typo protection).

```python
# my_config.py
model = dict(patch_len_list=(1024, 128, 16, 1), d_model_list=(24, 24, 24), hira_r=8, seed=0)
train = dict(steps=20000, lr=3e-3, log_every="0.5", per_device_batch=16)
compress = dict(batch_size=2048)   # read by compress.py's own --config flag
```

`enfrac/configs/` and `enfrac_zero/configs/` are separate directories (not shared) because the
two packages' `ModelConfig` dataclasses aren't identical (`enfrac`'s has `use_hira`/`hira_r`,
`enfrac_zero`'s doesn't) — the configs otherwise mirror each other dataset-for-dataset so results
are comparable. Sample configs, smallest to largest:

| config | file size | purpose |
|:-------|:----------|:--------|
| `fatihah` | 562 B | sanity check (lossless round trip), not real compression |
| `juz1` | ~44 KB | sanity check |
| `quran_uthmani` | ~1.4 MB | real compression demo (both variants exceed 1x) |
| `enwik8` | 100 MB | TPU-scale; see its docstring for the param-count sizing search and epoch-count reasoning (20 epochs, not the ~100 originally floated — quran_uthmani's real-compression run only needed ~0.28 epoch-equivalents) |

## Multi-device (TPU) training

`enfrac.train.train()` auto-detects `jax.local_device_count()`. With >1 device it switches from
the single-chunk-per-step `eqx.filter_jit` path to a `jax.pmap`'d data-parallel step: each device
gets `per_device_batch` chunks (a `TrainConfig` field, default 1), computes its own gradient,
`jax.lax.pmean` averages across devices, every replica applies the identical update. Verified
(not just assumed) by simulating 4 CPU devices locally
(`XLA_FLAGS=--xla_force_host_platform_device_count=4`) and diffing every device's trainable
leaves after 50 steps: zero drift. `enfrac_zero.train` inherits this for free (calls the same
`train()`). See `enfrac/README.md`'s "Multi-device" section for the JAX-vs-PyTorch `share_trunk`
gotcha this uncovered along the way (`[shared] * n` in a JAX pytree list does NOT alias like
PyTorch's `ModuleList` does — see `trunk_at()` in `model.py`).

**Installing JAX for a real TPU host**: this repo's `pyproject.toml` pins plain `jax`/`jaxlib`
(CPU wheels). On a TPU VM, install the TPU extra instead:
`uv pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`.

## Correctness invariants for compress.py/decompress.py

Two things must match exactly between compress and decompress, and **both are enforced in code,
not just documented** — neither has a CLI override on the decode side, both are read back from
the bundle's `meta.json` automatically:

1. **`batch_size`** — chunks are grouped into batches for `collect_logits()`/`generate()`;
   batched matmuls aren't bit-identical to differently-batched ones (floating-point
   non-associativity), which desyncs the range coder.
2. **`device`** (`cpu`/`tpu`/`gpu`) — same reasoning, one level up: TPU/GPU/CPU backends compute
   matmuls differently (TPU commonly uses reduced precision), so compressing on one backend and
   decompressing on another desyncs the coder just as fatally as a batch_size mismatch. **This
   was a real bug**, not a hypothetical: recompressing bundles on TPU then decompressing them on
   CPU (before `device` was added to `meta.json`) produced ~85-99% wrong bytes, not merely a
   worse ratio — silent, severe corruption, not a graceful degradation. Both scripts default
   `--device cpu` (host-dispatch-latency-bound recursion, not FLOP-bound — a TPU host's own
   many-core CPU is the better fit than routing through the accelerator), and `load_bundle()` in
   both `decompress.py`s sets `jax.config.update("jax_platform_name", ...)` from `meta.json`
   *before* the model is reconstructed (must happen before any other jax call).

If you ever add a new "must match" parameter to this pipeline, follow the same pattern: save it
into `meta.json` at compress time, read it back automatically at decompress time, don't expose a
decode-side flag that could silently disagree with what was actually used to encode.

## Operational lesson: TPU jobs need `tmux`, not backgrounded SSH

Queued-resource TPU VMs (`gcloud compute tpus queued-resources ...`) drop idle SSH connections
aggressively — a long-running job launched as a bare `ssh ... &` dies silently the moment the
connection drops, with nothing surfaced until you notice the output stopped hours ago. Always:

1. `tmux new-session -d -s NAME` — a bare persistent shell, **no command attached**.
2. `tmux send-keys -t NAME 'actual command here' Enter` — sent as if typed interactively.
3. (optional extra safety) `tmux set-option -t NAME remain-on-exit on`.
4. Check on it later with `tmux capture-pane -t NAME -p -S -N` (last N lines) — works whether the
   job is still running, finished, or crashed, since the pane never closes.

**Never** `tmux new-session -d -s NAME 'command'` — that ties the pane's lifetime to the command,
so it closes the instant the command exits or crashes, and there's nothing left to inspect
afterward. See `TPU.md`/`TPU_WORKFLOW.md` for the actual provisioning/SSH/rsync commands used for
this project's TPUs (`raden-tpu` project, `us-central2-b` zone, v4-8 nodes).

Also expect the local SSH *control master* (multiplexed connection) to die independently of the
remote tmux session — `ssh -O check -o ControlPath=... host` reporting "no such file" doesn't
mean the remote job died, only that the local multiplexed socket did. Re-issue the SSH command
fresh (a new connection reconnects fine); the tmux session on the far end is unaffected.

## Datasets

```
quran_data/                     # small, git-tracked
  surat_al-fatihah.txt            # 562 B
  juz1.txt                        # ~44 KB
  quran-uthmani.txt               # ~1.4 MB (also *_nl.txt / *-numbered.txt variants)

datasets/                       # larger, git-ignored
  enwik8                           # 100,000,000 B
```

`uv run python -m enfrac.download_data [--which enwik8,enwik9] [--out_dir datasets]` fetches
from the Large Text Compression Benchmark (mattmahoney.net), streams to disk with a progress
bar, verifies the extracted size against the known byte count, and cleans up the `.zip`.

`logs/` (git-ignored) holds experiment logs, checkpoints (`logs/<package>/<name>/`), and
compressed bundles (`logs/<package>/<name>_compressed/`) — use it instead of `/tmp` so results
persist across sessions.

## Key design decisions

- **Goal is overfitting/memorization, not generalization**: no dropout, `weight_decay=0.0` on
  the optimizer.
- **Perfect argmax decode is not the goal.** Range coding always encodes/decodes losslessly
  regardless of prediction quality — imperfect predictions inflate `rc_bytes` (worse ratio), they
  never break correctness. Only the two invariants above (`batch_size`, `device`) can actually
  break correctness.
- **`float32` throughout** (JAX's default) — no `bfloat16`/mixed precision in this codebase yet.
- **Bit-exactness discipline**: `collect_logits()` and `generate()` must stay the same code path
  (the former literally calls the latter with a teacher-forcing callback) — a batched training
  loss computation and the recursive step-by-step generation computation can disagree in the
  last bit even though mathematically equivalent, which is fatal for range coding. `forward()`/
  `__call__()` (the fast batched path) is fine for training loss, never for CDFs that get
  range-coded. See `model.py`'s module docstring.
