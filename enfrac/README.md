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

## Multi-device (TPU pod slice / multi-GPU) training

`enfrac.train.train()` auto-detects `jax.local_device_count()` and switches from the default
single-chunk-per-step path to a `jax.pmap`'d data-parallel step when more than one device is
visible -- no flag needed, and `enfrac_zero.train` inherits this for free since it calls the
same `train()`. Each step then consumes `n_devices * per_device_batch` chunks (`per_device_batch`
is a `TrainConfig` field, default `1`): every device computes its own gradient on its shard,
`jax.lax.pmean` averages them across devices, and every replica applies the identical averaged
update -- the standard "replicate params, average grads" recipe. Verified locally by simulating
4 CPU devices (`XLA_FLAGS=--xla_force_host_platform_device_count=4`) and diffing every device's
trainable leaves after 50 steps: **zero drift**, confirming replicas stay bit-identical rather
than silently diverging.

```bash
# on a TPU host, e.g. a v4-8 (4 chips visible to jax.local_devices()):
uv run python -m enfrac.train --config enfrac/configs/quran_uthmani.py \
    --log_dir logs/enfrac/quran_uthmani --per_device_batch 2   # 4 devices * 2 = batch of 8/step
```

Installing JAX for an actual TPU host (this repo's `pyproject.toml` pins plain `jax`/`jaxlib`,
the CPU wheels) needs the TPU extra instead: `pip install -U "jax[tpu]"` (or the pinned version
matching this repo's `jax==0.11.1`) -- see https://docs.jax.dev/en/latest/installation.html for
the current install command, which changes across JAX releases.

**compress.py/decompress.py stay single-device on purpose.** Their cost is dominated by many
small host<->device dispatches from `generate()`'s python-level autoregressive recursion (see
`model.py`'s determinism-contract docstring for why the recursion can't be batched away --
bit-exactness with the training-time computation is required for range coding), not raw FLOPs,
so `pmap`-ing them would need a much larger rewrite (sharding *inside* the recursion, across
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
