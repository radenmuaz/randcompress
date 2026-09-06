"""enwik8 (100,000,000 B) -- the first config in this repo sized for a real memorization-
compression attempt at "big" scale rather than a sanity check (contrast with fatihah/juz1's
overparam sanity checks and quran_uthmani's modest real-compression demo).

## Sizing (see enfrac/model.py's trainable_filter() -- this isn't a formula, just constructing
ByteFractalGen at candidate dims and counting actual param bytes until close):
  - Target: ~100MB total (frozen HiRA base + trainable adapter). Hit with d_model=368, n_layers=4,
    n_heads=8, mlp_mult=4, byte_embed_dim=20, share_trunk=True -> 99.57MB total.
  - Target: ~40MB trainable. hira_r=360 (very high rank relative to d_model=368 -- NOT a typical
    low-rank PEFT setup; chosen specifically to push the trainable *fraction* up as far as this
    architecture allows) gets trainable to 31.12MB, short of 40MB: patch_in's frozen W0
    (shape child_len*byte_embed_dim -> d_model) still dominates total param count even at
    byte_embed_dim=20, and HiRA's trainable fraction per layer is bounded by ~r/d_in, which caps
    out once r approaches d_model itself -- r can't meaningfully exceed d_model without the
    "low-rank" premise breaking down further than it already has here. 31MB is the closest
    achievable to 40MB without an even more extreme (and increasingly meaningless) r.
  - Perfect argmax decode isn't the goal here -- range coding always encodes losslessly
    regardless of prediction quality, just at a higher bit-rate when accuracy is lower (rc_bytes
    inflate, correctness doesn't break). See enfrac_zero/configs/enwik8.py for the directly-
    40MB-trainable baseline counterpart (no frozen base, so no total-vs-trainable tension).

  uv run python -m enfrac.train --config enfrac/configs/enwik8.py --log_dir logs/enfrac/enwik8
  uv run python -m enfrac.compress --config enfrac/configs/enwik8.py \
      --ckpt logs/enfrac/enwik8 --input datasets/enwik8 --output logs/enfrac/enwik8_compressed
  uv run python -m enfrac.decompress --bundle logs/enfrac/enwik8_compressed \
      --output /tmp/enwik8_out --verify datasets/enwik8

## Epoch count: 20, not the ~100 originally floated

enfrac/configs/quran_uthmani.py's real-compression run only used ~0.28 epoch-equivalents (6000
steps / 21,250 chunks, single-device) and still hit 2.09x compression -- this codebase's
per-chunk teacher-forced loss converges fast relative to a full epoch because every chunk is an
independent, fully-supervised 1024-byte memorization target, not a generalization problem. 20
epochs over enwik8's ~97,657 chunks gives several orders of magnitude more gradient signal than
that quran_uthmani run needed, while keeping wall time bounded. `steps` below is computed FROM
n_epochs (not hardcoded) so changing per_device_batch or running on a different device count
doesn't silently change how much of the data actually gets seen -- see train()'s multi-device
auto-detection in enfrac/train.py.

## batch_size=2048 for compress/decompress

Both now default to --device cpu (host-dispatch-latency-bound, not FLOP-bound -- see
compress.py's --device help). Wall time there scales with n_raw_bytes/batch_size regardless of
patch_len_list (see quran_uthmani.py's docstring for the derivation) -- 2048 keeps total
sequential host-side steps to ~48,800 for the full 100MB file, feasible on a many-core host.
"""
import math

N_CHUNKS = math.ceil(100_000_000 / 1024)   # 97,657 (patch_len_list[0]=1024)
N_EPOCHS = 20
PER_DEVICE_BATCH = 16   # x4 (TPU v4-8) devices = batch_sz=64/step; train() auto-detects device count
BATCH_SZ = 4 * PER_DEVICE_BATCH
STEPS = math.ceil(N_EPOCHS * N_CHUNKS / BATCH_SZ)

model = dict(
    patch_len_list=(1024, 128, 16, 1),
    d_model_list=(368, 368, 368),
    n_layers_list=(4, 4, 4),
    n_heads_list=(8, 8, 8),
    mlp_mult_list=(4, 4, 4),
    byte_embed_dim=20,
    share_trunk=True,
    use_hira=True,
    hira_r=360,
    seed=0,
)

train = dict(
    dataset="datasets/enwik8",
    steps=STEPS,
    lr=3e-3,
    warmup_steps=500,
    log_every="0.1",
    per_device_batch=PER_DEVICE_BATCH,
)

compress = dict(batch_size=2048)
