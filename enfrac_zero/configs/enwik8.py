"""enwik8 (100,000,000 B) -- directly-init baseline counterpart to enfrac/configs/enwik8.py.

No frozen base / HiRA split here, so "40MB" is just the model's total (=trainable) size
directly, no sizing tension to navigate: d_model=272, n_layers=4, n_heads=8, mlp_mult=4,
byte_embed_dim=128, share_trunk=True -> 40.44MB total (see enfrac/model.py's counterpart search
-- same method, just no hira_r knob here). Compare against enfrac's 99.57MB-total/31.12MB-
trainable HiRA config at the identical epoch/batch budget below.

  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik8.py --log_dir logs/enfrac_zero/enwik8
  uv run python -m enfrac_zero.compress --config enfrac_zero/configs/enwik8.py \
      --ckpt logs/enfrac_zero/enwik8 --input datasets/enwik8 --output logs/enfrac_zero/enwik8_compressed
  uv run python -m enfrac_zero.decompress --bundle logs/enfrac_zero/enwik8_compressed \
      --output /tmp/enwik8_zero_out --verify datasets/enwik8

See enfrac/configs/enwik8.py's docstring for the epoch-count (20) and batch_size (2048)
reasoning -- identical here since both are properties of the dataset/chunking and the
compress/decompress dispatch-latency bottleneck, not of which model variant is running.
"""
import math

N_CHUNKS = math.ceil(100_000_000 / 1024)
N_EPOCHS = 20
PER_DEVICE_BATCH = 16
BATCH_SZ = 4 * PER_DEVICE_BATCH
STEPS = math.ceil(N_EPOCHS * N_CHUNKS / BATCH_SZ)

model = dict(
    patch_len_list=(1024, 128, 16, 1),
    d_model_list=(272, 272, 272),
    n_layers_list=(4, 4, 4),
    n_heads_list=(8, 8, 8),
    mlp_mult_list=(4, 4, 4),
    byte_embed_dim=128,
    share_trunk=True,
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
