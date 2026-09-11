"""enwik9 (1,000,000,000 B), FILE-CHUNKED, no-HiRA baseline, 128 KiB (2^17) file chunks -- SAME
model shape as enwik9_chunked_1mb_r4.py (d_model=448, n_layers=7, patch_len_list=(1024,256,64,16,
4,1), ratio-4 downsample) -- only file_chunk_bytes shrinks from 1 MiB to 128 KiB, so
chunk_n_timesteps drops from 1024 to 128 and n_file_chunks grows from 954 to 7630.

Mirrors enfrac/configs/enwik9_chunked_128kb_r4.py (the HiRA counterpart, confirmed working on
real TPU hardware at chunk_batch_size=8/pmap, 100% duty cycle across all 4 chips, ~21-25GB/chip).
chunk_batch_size=8 here is a direct carryover from that run as a safe starting point -- this
model is much smaller (24.4M total params vs HiRA's 302.7M), so there is likely headroom to go
higher once this is confirmed stable (same incremental-doubling approach used for the HiRA runs).

n_file_chunks = ceil(1,000,000,000 / 131,072) = 7630. n_epochs=60 (matches enwik9_chunked_1mb_r4.py's
tripled epoch count -- goal is heavy overfitting, not early stopping).

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_chunked_128kb_r4.py --log_dir logs/enfrac_zero/enwik9_chunked_128kb_r4
"""

model = dict(
    patch_len_list=(1024, 256, 64, 16, 4, 1),
    d_model_list=(448,) * 6,
    n_layers_list=(7,) * 6,
    n_heads_list=(8,) * 6,
    mlp_mult_list=(4,) * 6,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9_chunked_128kb_r4",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=60,
    file_chunk_bytes=131_072,
    chunk_batch_size=32,
    micro_batch=1024,
    seed=0,
)
