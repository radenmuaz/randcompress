"""enwik9 (1,000,000,000 B), FILE-CHUNKED, HiRA counterpart to
enfrac_zero/configs/enwik9_chunked_1mb_r4.py -- SAME 1 MiB file chunks, patch0=1024 with
downsample rate 4 (patch_len_list=(1024,256,64,16,4,1), 6 levels), chunk_n_timesteps=1024,
patch_in_scheme="mean_pool" everywhere. Only d_model/n_layers/hira differ (HiRA, everything else
frozen except B). No scan-machinery changes (plain, unmodified `_train_recurse`).

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9_chunked_1mb_r4.py --log_dir logs/enfrac/enwik9_chunked_1mb_r4
"""

model = dict(
    patch_len_list=(1024, 256, 64, 16, 4, 1),
    d_model_list=(1216,) * 6,
    n_layers_list=(8,) * 6,
    n_heads_list=(8,) * 6,
    mlp_mult_list=(4,) * 6,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    use_hira=True,
    hira_r=165,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac/enwik9_chunked_1mb_r4",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    file_chunk_bytes=1_048_576,
    chunk_batch_size=1,
    micro_batch=1024,
    seed=0,
)
