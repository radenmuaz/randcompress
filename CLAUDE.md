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
total attention instead of `O(L²)`. Current focus: scaling up to bigger files (enwik8, enwik9,
both have real configs now) on single-device TPU — not architecture research.

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
| `enwik9` | 1 GB | TPU-scale, one level deeper than `enwik8` (8 levels, `patch_len_list[0]=2,097,152`) to keep level-0's own sequence short (477 timesteps) at 10x the byte count. Sized via `jax.eval_shape` param counts: `enfrac` ~977.7 MB total / ~95.8 MB trainable (HiRA, `hira_r=165`, `d_model=1216`, `n_layers=8`), `enfrac_zero` ~100.0 MB total (`d_model=448`, `n_layers=7`). **Only trains correctly with the levels-1+ memory-chunking fixes below** — the un-chunked code OOMs hard at this scale (levels 1+ and level 0's own mean-pool patch embedding both reach total-file-byte-count row counts by the deepest levels). |

## Single-device training (no pmap currently)

`enfrac.train.train()` is a single `eqx.filter_jit` call — **there is no `jax.pmap`/multi-device
sharding in the current code** (verified by grep: zero references to `pmap`/`per_device_batch`/
`local_device_count` anywhere in `enfrac/`/`enfrac_zero/`). An earlier version of this file (and
of `enfrac/README.md`) documented an auto-detecting `jax.pmap`'d data-parallel path — that
description is now **stale/inaccurate**; whatever code implemented it is gone. On a multi-chip
host (e.g. a v4-8 with 4 `TpuDevice`s visible via `jax.local_devices()`), training currently uses
only the one device JAX places arrays on by default — the other chips sit idle. This is a real,
known limitation, not a design choice — worth revisiting (real `pmap`/sharding across devices) if
compile time or throughput becomes the binding constraint at enwik8/9 scale, but not attempted
this session. See `enfrac/README.md`'s "JAX-specific correctness note" section for the
JAX-vs-PyTorch `share_trunk` gotcha found along the way (`[shared] * n` in a JAX pytree list does
NOT alias like PyTorch's `ModuleList` does — see `trunk_at()` in `model.py`), which is unrelated
to the pmap question and still accurate.

**Installing JAX for a real TPU host**: this repo's `pyproject.toml` pins plain `jax`/`jaxlib`
(CPU wheels). On a TPU VM, install the TPU extra instead:
`uv pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`
— **do this before the first real training run**, not just before compress/decompress: training
silently falls back to CPU with only a warning (`jax._src.xla_bridge: ... Falling back to cpu`)
printed, not an error, and a ~245M-param model training on CPU is catastrophically slower than on
TPU, not merely somewhat slower. This was a real bug hit this session: two TPU hosts ran a
sizeable chunk of a training attempt on CPU before the missing `jax[tpu]` install was noticed.

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

**Never redirect a launched command's output purely to a file** (`cmd > log.txt 2>&1`) if you also
want to check on it via `tmux capture-pane` — a plain redirect sends stdout/stderr straight to the
file and NONE of it reaches the pane's own screen buffer, so `capture-pane` shows an empty/stale
pane even though the job is alive and actively printing (e.g. tqdm progress) to the file. This
looks exactly like a hang from `capture-pane`'s perspective but isn't one — confirmed this session:
a real training job's tqdm bar was updating fine in the log file the whole time, while
`tmux capture-pane` on the same session showed nothing new. Use `tee` instead so both work:
`cmd 2>&1 | tee log.txt` — then `tmux capture-pane` AND `tail -f log.txt` both show live output.

Also expect the local SSH *control master* (multiplexed connection) to die independently of the
remote tmux session — `ssh -O check -o ControlPath=... host` reporting "no such file" doesn't
mean the remote job died, only that the local multiplexed socket did. Re-issue the SSH command
fresh (a new connection reconnects fine); the tmux session on the far end is unaffected.

**After launching (or relaunching) a TPU training run, always give the user the direct SSH +
tmux-attach one-liner for it** — `gcloud compute tpus tpu-vm ssh` is slow (re-resolves/auths every
call); a direct `ssh -i ~/.ssh/google_compute_engine muaz@<external IP> -t "tmux attach -t
<session>"` is instant and is what the user actually wants to run themselves. The `-i
~/.ssh/google_compute_engine` key is REQUIRED -- plain `ssh muaz@<ip>` fails, since that's not the
user's default key. This key is auto-generated and its public half auto-pushed to the instance's
metadata the first time `gcloud compute tpus tpu-vm ssh`/`queued-resources ssh` connects to that
node (see TPU_WORKFLOW.md) -- direct ssh only works AFTER that first gcloud-wrapped connection has
happened at least once per node. Get the external IP with `gcloud compute tpus tpu-vm describe
<node> --zone=... --project=... --format="value(networkEndpoints[0].accessConfig.externalIp)"`
once per node (IPs are stable across a node's lifetime unless it's reimaged/recreated). Give it
plain, one line per node, no table.

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
  never break correctness. `batch_size`/`device`/`dtype` (see below) are the invariants that can
  actually break correctness.
- **`dtype` is now a THIRD must-match invariant, alongside `batch_size`/`device`.** `compress.py`
  defaults to `dtype="float64"` (enables `jax_enable_x64`) because the fast encode path
  (`collect_logits_fp64`, batches every level-0 timestep's local recursion through one call
  instead of one Python loop per timestep) is only numerically safe in float64 — verified
  empirically that the same batching in float32 flips ~1.6-2.3% of quantized CDF bins relative to
  the per-timestep-sequential reference. `dtype` is recorded in `meta.json` and read back by
  `decompress.py` automatically, same pattern as `batch_size`/`device`. A REAL correctness bug
  was found and fixed here too: under `jax_enable_x64`, `jax.random.normal`/`orthogonal` produce
  **completely different values** from the same `PRNGKey` (not just different precision) — this
  silently corrupted the frozen byte-embedding table (and would have corrupted `enfrac`'s frozen
  HiRA `W0`/`A` too) whenever the model was constructed under x64 mode, collapsing accuracy from
  ~76% to ~25%. Fixed by forcing `dtype=jnp.float32` explicitly at every frozen/seed-derived
  random init site (`make_byte_embedding`, `HiraLinear.__init__`, `init_hira_A`) — any NEW frozen
  (never-checkpointed) random init added to this codebase must do the same, or it will silently
  regenerate wrong under x64.
- **Bit-exactness discipline**: `collect_logits()` and `generate()` must stay the same code path
  (the former literally calls the latter with a teacher-forcing callback) — a batched training
  loss computation and the recursive step-by-step generation computation can disagree in the
  last bit even though mathematically equivalent, which is fatal for range coding. `forward()`/
  `__call__()` (the fast batched path) is fine for training loss, never for CDFs that get
  range-coded. See `model.py`'s module docstring.

## Training-time memory chunking at real corpus scale (enwik8/9)

`ByteFractalGen.__call__` (training loss) is a single `eqx.filter_jit` call that -- by design --
processes the WHOLE file's level-0 sequence and every level's full recursive batch in one shot
(no TBPTT, no gradient truncation, no chunk-carried state; see "Bit-exactness discipline" above
and `model.py`'s module docstring for why that discipline exists). At quran_uthmani/juz1/fatihah
scale this was never a problem. At enwik8/9 scale it hit THREE distinct, real OOMs, found and
fixed in sequence this session (each confirmed via an actual TPU crash message, not predicted):

1. **Levels 1+'s batch dimension (`Bcur`) reaches the total raw byte count by the terminal
   level.** The old code computed each level's entire `[Bcur, seq_len, D]` batch in one plain
   call; at enwik9 scale `Bcur` reaches 125M+ rows by the last couple of levels. A single
   `cur_bytes.reshape(-1, seq_len, child_len)` on a 125,042,688-row array alone requested **64GB**
   against a v4-8's 34GB HBM (TPU pads the size-8 minor dim to a 128-tile, a 16x blowup on top of
   the ~4GB of actual data). **Fixed**: `ByteFractalGen._train_recurse` replaces the old
   level-by-level loop with a `jax.lax.scan`-chunked recursive walk — processes `micro_batch`
   (`TrainConfig` field, default 8192) rows at a time, and for non-terminal levels recurses into
   level+1 IMMEDIATELY per chunk (depth-first) rather than ever assembling a full level's output
   array (which for the deepest transitions would itself be hundreds of GB). Verified
   loss/gradient-identical (to ~1e-6/1e-7, pure floating-point summation-order noise, not a
   correctness gap) against the old unchunked code at small scale.
2. **Combining `jax.checkpoint` with that nested `jax.lax.scan` structure crashes the XLA
   compiler**, not a memory error: `windowing_util.cc: Non-OK-status: VerifyCanonicalBounds` /
   `RET_CHECK failure ... CouldLeS32(0, bound)` / SIGABRT. Confirmed via a controlled A/B
   (disabling just the `jax.checkpoint` call inside `_train_recurse` made the crash disappear,
   replaced by an unrelated, separately-fixed OOM below) — a real XLA/TPU compiler limitation, not
   a bug in this codebase's logic. **Fixed**: levels 1+ use plain `Trunk.__call__` inside the scan
   (not `forward_remat`/`jax.checkpoint`) — `remat_depth` still threads through `_train_recurse`'s
   signature (inherited from `__call__`) but does nothing at levels 1+ currently. This isn't a
   real loss: `seq_len` is only 8 at every level 1+, so per-chunk activation memory there was
   already bounded by `micro_batch` regardless of checkpointing — level 0 (whose sequence is long)
   still uses `forward_remat`/`jax.checkpoint` via `remat_time`/`remat_depth` as before.
3. **`embed_patch`'s mean-pool scheme materializes `byte_embed(byte_patch)` — shape
   `[rows, P, byte_embed_dim]` — BEFORE pooling**, defeating the entire point of "mean_pool is
   params/compute independent of `P`" (only true for params, never was for memory). This bites at
   TWO independent axes, not just one: large `P` (level 0's own `patch_len_list[0]=2,097,152`
   alone would need ~1TB for `[477, 2097152, 256]`), AND large row counts flowing in from
   `_train_recurse`'s scan body (up to `micro_batch * seq_len` = 65,536 rows) — level 2's own
   `P=32,768` combined with 65,536 incoming rows would need **~550GB**, an order of magnitude
   worse than the large-`P` case alone and found only after fix #1 was already deployed and
   fix #3's first (P-only-chunked) attempt was checked against real per-level shapes. **Fixed**:
   `ByteFractalGen._chunked_mean_pool` chunks over the FLATTENED ROW axis only (never `P`), with
   an adaptive chunk size `_MEAN_POOL_TARGET_ELEMENTS // P` (target ≈2.1M elements per
   `jax.lax.scan` step) so the `rows_per_chunk * P` product stays roughly constant regardless of
   which level it's called from. Verified exact-match (diff 0.0, since each row's mean is
   independent — no cross-row accumulation, so padding rows can just be sliced away afterward
   rather than needing a valid-mask) against the naive unchunked mean at small scale.

**Takeaway for any future change to `__call__`/`_train_recurse`/`embed_patch`**: at enwik8/9
scale, EVERY tensor whose leading dimension is "some level's total patch count" or "the file's
total byte count" is a potential multi-GB-to-TB blowup if computed in one shot — chunk it via
`jax.lax.scan` (never a bare Python loop, which would unroll to hundreds of thousands of duplicate
HLO ops and blow up compile time/graph size instead of memory) before assuming a change is safe
at real corpus scale, and re-verify against `/tmp` or a scratch script's small-model A/B (matching
this session's testing pattern) before deploying to a real multi-hour TPU run.

## KIV / urgent TODO: decompress is Python/framework-overhead-bound, not compute-bound

`decompress.py` is unavoidably sequential (see `model.py`'s module docstring: level-0 timestep
`t+1` needs timestep `t`'s *entire* subtree actually decoded first, and every byte within one
subtree needs its causal predecessor's *actual* decoded value — not fixable by batching, since
the real bytes genuinely don't exist yet at decode time). That sequential *shape* is inherent.
But profiled with `cProfile` on a representative decode-like call (juz1-scale model, after JIT
warmup) and found the wall-clock cost itself is NOT dominated by that sequential model compute —
it's dominated by pure Python/framework bookkeeping:

```
equinox _module.py:__getattribute__     34.8%  (2.17M calls) -- equinox's pytree/module machinery
jax tree_util.py:tree_leaves            26.8%  -- pytree flattening before every jit call
jax pjit cache_miss / compile           29.0%  -- 184 recompile events DESPITE warmup (see below)
```

At least ~60% of total wall-clock time is pure Python overhead, not FLOPs, not XLA execution --
confirmed by direct profiling, not inferred. Two concrete, actionable findings:

1. **`generate()`'s level-0 `step_fn` is a closure defined fresh inside `generate()` every call**
   -- so JAX's compiled-executable cache (keyed partly on the wrapped function object's identity)
   does NOT survive across separate `generate()` invocations, only within one call's own loop.
   Every fresh call to `generate()`/`collect_logits()` re-triggers a real compile for level 0's
   step function. `_gen_children`'s `_trunk_forward_jit` is module-level so it doesn't have this
   specific problem, but level 0's step function does.
2. **Equinox's `__getattribute__` override and JAX's `tree_leaves` flattening are called millions
   of times** for what should be a handful of tiny matmuls (`d_model` ~48-96, `seq_len<=8` at
   levels 1+) -- every `_trunk_forward_jit`/`step_fn` call re-flattens the whole `Trunk` pytree
   (all its weight arrays) to pass as a traced jit argument, which is pure overhead unrelated to
   the actual compute size.

Not yet attempted this session (ran out of turn budget): (a) fix the step_fn closure-recompile
issue (hoist it out of `generate()`, or restructure so its cache persists across calls); (b)
reduce per-call pytree overhead (e.g. close over `trunk`'s weights as a static capture instead of
passing the whole module as a traced argument each call, mirroring how `generate()`'s step_fn
already closes over `trunk0`/`rope_base` rather than accepting them as arguments). Both are
compatible with the existing bit-exactness/correctness invariants (they change dispatch
mechanics, not the computation itself) and should be tackled before any further architecture-
level optimization attempt (e.g. speculative decoding, discussed as the most promising next
architectural lever but a bigger, riskier undertaking than these two fixes).
