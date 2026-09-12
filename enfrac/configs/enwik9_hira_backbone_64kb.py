"""enwik9 (1,000,000,000 B), HiRA using the SAME backbone shape as enfrac_zero's tpu5/tpu6
baseline (d_model=448, n_layers=7, n_heads=8, mlp_mult=4, byte_embed_dim=256,
patch_len_list=(1024,256,64,16,4,1), share_trunk=True) -- but now that backbone is FROZEN
(random, regenerated from seed, never stored) and only a HiRA adapter (~20% of total params) is
trainable, instead of enfrac_zero's fully-trainable version of the identical shape. Sibling of
enwik9_hira_backbone_128kb.py -- same model/hira_r, smaller file chunk + bigger batch.

hira_r=164: total=37,729,792 params, trainable=7,543,296 (19.99% of total) -- see
enwik9_hira_backbone_128kb.py's docstring for the search.

64 KiB (2^16) file chunks -- n_file_chunks = ceil(1,000,000,000 / 65,536) = 15259,
chunk_n_timesteps=64. chunk_batch_size=64 (per_device=16) -- this backbone is much smaller than
the earlier d_model=1408 HiRA configs (37.7M vs 302.7M total params) so a larger batch than
those configs' bs=16 should fit; verify on real hardware and back off if it OOMs.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9_hira_backbone_64kb.py --log_dir logs/enfrac/enwik9_hira_backbone_64kb
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
    use_hira=True,
    hira_r=164,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac/enwik9_hira_backbone_64kb",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=60,
    file_chunk_bytes=65_536,
    chunk_batch_size=64,
    micro_batch=1024,
    seed=0,
)
