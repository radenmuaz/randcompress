# randcompress

Neural compression via memorization: train a model to overfit one file; the trained weights
*are* the compressed representation, and range coding removes the redundancy the model's own
predictions didn't capture. Decompression reconstructs the file autoregressively.

**Active codebase**: `enfrac/` and `enfrac_zero/` (JAX + [Equinox](https://github.com/patrick-kidger/equinox)
+ [optax](https://github.com/google-deepmind/optax)), a byte-level port of
[FractalGen](https://arxiv.org/html/2502.17437v2) — a generative model built by recursively
invoking a generator on progressively smaller sub-patches (`O(L)` total attention, not `O(L²)`).
Current focus: multi-device TPU training and scaling up to bigger files (enwik8, eventually
enwik9) — see `enfrac/README.md` for the full architecture writeup, config system, and
multi-device design, and `TPU.md`/`TPU_WORKFLOW.md` for the TPU workflow.

Everything else (the old PyTorch `randcompress/` MsRNN+HiRA/LoRA package, `overfitter/`
SummTransformer, `overfitter_peft/`, `overfitter_v1/`, old JAX experiments) is superseded and
archived under `archive/` — kept for reference only, not maintained.

## Two variants

| package | architecture | trainable | use case |
|:--------|:-------------|:----------|:---------|
| `enfrac` | HiRA-adapted frozen base (`W = W₀ + W₀⊙(B·A)`) | only `B` (+ root_cond) | large nominal model size, small compressed bundle |
| `enfrac_zero` | plain linears, no frozen base | everything | baseline comparison, no PEFT overhead |

Both share the same recursive FractalAR architecture, `codec.py` (ctypes-bound C range coder,
unchanged across variants — compress/decompress logic stays C), config system, and multi-device
training loop; only the model class and `ModelConfig` differ. See `enfrac/README.md`.

## Quick start

```bash
uv sync

# Train -> compress -> decompress on a small sanity-check file
uv run python -m enfrac.train --config enfrac/configs/fatihah.py --log_dir logs/enfrac/fatihah
uv run python -m enfrac.compress --config enfrac/configs/fatihah.py \
    --ckpt logs/enfrac/fatihah --input quran_data/surat_al-fatihah.txt --output logs/enfrac/fatihah_compressed
uv run python -m enfrac.decompress --bundle logs/enfrac/fatihah_compressed \
    --output /tmp/recovered.txt --verify quran_data/surat_al-fatihah.txt
# -> Verification: PERFECT MATCH

# Same for the no-HiRA baseline
uv run python -m enfrac_zero.train --config enfrac_zero/configs/fatihah.py --log_dir logs/enfrac_zero/fatihah
```

Sample configs (`enfrac/configs/*.py`, `enfrac_zero/configs/*.py`): `fatihah` (562B, sanity
check), `juz1` (~44KB, sanity check), `quran_uthmani` (~1.4MB, real compression demo), `enwik8`
(100MB, TPU-scale — see its docstring for sizing/epoch-count reasoning). A config file is a
plain `.py` module with `model`/`train`/`compress` dicts; `--config file.py --field value`
layers CLI overrides on top. See `enfrac/README.md`'s "Config system" section.

## Multi-device / TPU

`enfrac.train.train()` auto-detects `jax.local_device_count()` and switches to a `jax.pmap`'d
data-parallel step when more than one device is visible — no flag needed. `compress.py`/
`decompress.py` default to `--device cpu` instead (host-dispatch-latency-bound, not FLOP-bound;
a TPU host's own many-core CPU is the better fit). See `enfrac/README.md`'s "Multi-device"
section for the full rationale and verification, and `TPU.md`/`TPU_WORKFLOW.md` for
provisioning/SSH/rsync commands.

**Two hard-won correctness rules, both enforced in code (not just documented):**
- `batch_size` and `--device` **must** match between compress and decompress — both are saved
  into the bundle's `meta.json` and read back automatically (no CLI override exists for either
  on the decode side) because differently-batched *or* differently-backended (TPU vs CPU vs GPU)
  matmuls aren't bit-identical, which desyncs the range coder. This was a real bug: recompressing
  on TPU then decompressing on CPU (before the fix) produced bundles that decoded to ~85-99%
  wrong bytes, not just a worse ratio.
- **Always launch long-running TPU jobs inside a persistent `tmux` session** (`tmux new-session
  -d -s NAME` with no command, then `tmux send-keys -t NAME '...' Enter` — never
  `tmux new-session -d -s NAME 'command'`, which closes the pane the moment the command exits or
  crashes, losing all visibility). These queued-resource TPU VMs drop idle SSH connections
  aggressively; a job running under a bare backgrounded SSH session dies silently with the
  connection, with no error surfaced until you notice hours later.

## Datasets

```
quran_data/                     # small, git-tracked
  surat_al-fatihah.txt            # 562 B
  juz1.txt                        # ~44 KB
  quran-uthmani.txt               # ~1.4 MB

datasets/                       # larger, git-ignored (see enfrac/download_data.py)
  enwik8                          # 100,000,000 B
```

`uv run python -m enfrac.download_data` fetches enwik8/enwik9 from the Large Text Compression
Benchmark and unzips them into `datasets/`.

## Repo layout

```
enfrac/              main (HiRA/PEFT) package -- model.py, train.py, compress.py, decompress.py,
                      checkpoint.py, config.py, codec.py, tokenizer.py, configs/, download_data.py
enfrac_zero/          baseline (no-HiRA) counterpart -- same shape, imports shared pieces from enfrac
rc_codec.c            C range coder (ctypes-bound, compiled to /tmp/*.so on first use)
quran_data/           small git-tracked datasets
datasets/             larger git-ignored datasets (enwik8, ...)
logs/                 git-ignored experiment logs and checkpoints/bundles
TPU.md, TPU_WORKFLOW.md   TPU provisioning and SSH/rsync/tmux workflow notes
archive/              superseded packages (randcompress/, overfitter/, overfitter_peft/,
                      overfitter_v1/, old JAX experiments) -- reference only, not maintained
```

See `CLAUDE.md` for the full guide (architecture, config system, correctness invariants,
operational lessons) and `enfrac/README.md` for the architecture deep-dive.
