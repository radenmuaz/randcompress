"""enwik9 (1,000,000,000 B), HiRA using the SAME backbone shape as enfrac_zero's tpu5/tpu6
baseline (d_model=448, n_layers=7, n_heads=8, mlp_mult=4, byte_embed_dim=256,
patch_len_list=(1024,256,64,16,4,1), share_trunk=True) -- but now that backbone is FROZEN
(random, regenerated from seed, never stored) and only a HiRA adapter (~20% of total params) is
trainable, instead of enfrac_zero's fully-trainable version of the identical shape. Meant as a
direct comparison: same architecture/capacity, HiRA-adapted vs fully-trained, same file-chunking.

hira_r=164 found via real ByteFractalGen construction (not eval_shape -- see CLAUDE.md's note on
eval_shape's trainable-count quirk): total=37,729,792 params, trainable=7,543,296 (19.99% of
total) -- the closest whole hira_r to exactly 20% trainable in a search over hira_r=32..256.

128 KiB (2^17) file chunks -- n_file_chunks = ceil(1,000,000,000 / 131,072) = 7630,
chunk_n_timesteps=128. chunk_batch_size=32 (per_device=8) -- this backbone is much smaller
(37.7M total params vs the earlier d_model=1408 HiRA configs' 302.7M) so a larger batch than
those configs' bs=8 should fit; verify on real hardware and back off if it OOMs.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9_hira_backbone_128kb.py --log_dir logs/enfrac/enwik9_hira_backbone_128kb
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
    log_dir="logs/enfrac/enwik9_hira_backbone_128kb",
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
