# enfrac — JAX/Equinox port of overfitter_peft / overfitter

Byte-level FractalAR (ByteFractalGen), ported from the PyTorch `overfitter_peft/` (HiRA/PEFT
frozen-base adapters) and `overfitter/` (plain, no-adapter baseline) packages to
JAX + [Equinox](https://github.com/patrick-kidger/equinox) + [optax](https://github.com/google-deepmind/optax).
See `overfitter/README.md` and `overfitter_peft/README.md` for the architecture writeup
(recursion, `patch_len_list`, `share_trunk`, `byte_embed_dim` tradeoffs) — unchanged here, only
the tensor backend differs.

## Layout

- `model.py`, `train.py`, `compress.py`, `decompress.py`, `checkpoint.py` — the **main** package,
  ported from `overfitter_peft/`: HiRA-adapted frozen-base linears by default (`use_hira=True`
  on `ModelConfig`; set `use_hira=False` to fall back to plain linears with the same file layout).
- `../enfrac_zero/` (sibling package) — ported from `overfitter/`: the plain, no-adapter
  architecture (its own `ModelConfig` has no `use_hira`/`hira_r`). Imports pure math (RoPE,
  attention, RMSNorm, the frozen byte-embedding table) and the generic `train()` loop/
  `codec.py`/`tokenizer.py` from `enfrac` outright — only the model class and CLI differ.
- `codec.py`, `tokenizer.py` — unchanged from the PyTorch packages (pure numpy/ctypes, no
  backend dependency). **Compress/decompress stay C**: `codec.py` still ctypes-binds
  `rc_codec.c`; only the orchestration in `compress.py`/`decompress.py` moved from torch to jax.
- `config.py` — config system shared by both packages (see below).
- `configs/` — sample dataset configs for **this** (HiRA/PEFT) package: fatihah/juz1/quran_uthmani,
  each with a `model` dict (including explicit `use_hira`/`hira_r`/`seed`), a `train` dict, and a
  `compress` dict. `../enfrac_zero/configs/` holds the no-HiRA counterparts (same
  architecture dims and dataset/step settings, minus `use_hira`/`hira_r` since
  `enfrac_zero.model.ModelConfig` has no such fields) -- the two directories are separate
  because the two packages' `ModelConfig` dataclasses aren't identical, not because the configs
  are meant to diverge.

## Config system

Three layers, later wins: dataclass defaults < `--config <file.py>` < individual `--<field>`
CLI flags. A config file is a plain `.py` module; it's `exec`'d and searched for `model`/
`model_config` and `train`/`train_config` attributes (each a dict or a dataclass instance):

```python
# my_config.py
model = dict(patch_len_list=(1024, 128, 16, 1), d_model_list=(24, 24, 24), hira_r=8, seed=0)
train = dict(steps=20000, lr=3e-3, log_every="0.5")
compress = dict(batch_size=32)   # read by compress.py's own --config flag
```

```bash
uv run python -m enfrac.train --config enfrac/configs/juz1.py \
    --log_dir logs/enfrac/juz1 --lr 1e-3   # --lr overrides the file's lr
uv run python -m enfrac.compress --config enfrac/configs/juz1.py \
    --ckpt logs/enfrac/juz1 --input quran_data/juz1.txt --output logs/enfrac/juz1_compressed
```

Unknown field names in a config file raise immediately (typo protection) rather than being
silently ignored. `enfrac_zero.train`/`.compress` take the same `--config` flag against
their own (HiRA-free) `ModelConfig` -- point them at `enfrac_zero/configs/*.py` instead.

## Single-device training (no pmap currently)

**Correction, this no longer describes the current code**: an earlier version of this section
documented an auto-detecting `jax.pmap`'d data-parallel training path (`per_device_batch`,
`jax.lax.pmean` gradient averaging across devices). That code is gone -- `enfrac.train.train()` is
now a single `eqx.filter_jit` call with no `jax.pmap`/sharding anywhere (verified by grep: zero
references to `pmap`/`per_device_batch`/`local_device_count` in `enfrac/`/`enfrac_zero/`). On a
multi-chip host (e.g. a v4-8 with 4 `TpuDevice`s visible via `jax.local_devices()`), training
currently runs on only the one device JAX places arrays on by default -- the other chips are idle.
See `CLAUDE.md`'s "Single-device training" section for the same note plus a related real bug
(training silently falling back to CPU when `jax[tpu]` isn't installed, with only a warning, not
an error) and "Training-time memory chunking at real corpus scale" for how large single-device
training runs (enwik8/9) are kept within one TPU chip's HBM despite processing the whole file's
recursion in one `eqx.filter_jit` call.

Installing JAX for an actual TPU host (this repo's `pyproject.toml` pins plain `jax`/`jaxlib`,
the CPU wheels) needs the TPU extra instead:
`uv pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`
-- see https://docs.jax.dev/en/latest/installation.html for the current install command, which
changes across JAX releases. **Install this before the first training run, not just before
compress/decompress** -- see the CLAUDE.md note above for why.

**compress.py/decompress.py stay single-device on purpose.** Their cost is dominated by many
small host<->device dispatches from `generate()`'s python-level autoregressive recursion (see
`model.py`'s determinism-contract docstring for why the recursion can't be batched away --
bit-exactness with the training-time computation is required for range coding), not raw FLOPs,
so sharding them would need a much larger rewrite (sharding *inside* the recursion, across
levels with different batch sizes) for comparatively little payoff at the model sizes this
package targets.

## JAX-specific correctness note: `share_trunk`

PyTorch's `nn.ModuleList([shared] * n)` works because Python object identity makes every list
entry literally the same `nn.Module`, so autograd accumulates one shared `.grad`. JAX pytrees
flatten by structural position, not identity — the analogous `[shared] * n` would silently
produce `n` *independent* leaf copies that drift apart after the first optimizer step. Both
`enfrac/model.py` and `enfrac_zero/model.py` instead store exactly one `Trunk` and call it
`n_levels` times per forward pass (`trunk_at()`), which keeps it a single pytree leaf and makes
sharing actually work.
