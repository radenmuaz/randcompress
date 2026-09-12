"""Sibling of enwik9_longleaf_1mb.py -- SAME model (patch_len_list=(4096,2048,1024,1),
d_model=640, n_layers=3, share_trunk=True, flash attention at the terminal/leaf level only --
see that file's docstring for the long-context-on-leaves / shallow-network rationale and the
87.1MB real param count), only file_chunk_bytes/chunk_batch_size differ.

128 KiB (2^17) file chunks -- n_file_chunks = ceil(1,000,000,000 / 131,072) = 7630,
chunk_n_timesteps (level 0) = 32.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_longleaf_128kb.py --log_dir logs/enfrac_zero/enwik9_longleaf_128kb
"""

model = dict(
    patch_len_list=(4096, 2048, 1024, 1),
    d_model_list=(640,) * 4,
    n_layers_list=(3,) * 4,
    n_heads_list=(8,) * 4,
    mlp_mult_list=(4,) * 4,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    use_flash_attn_list=(False, False, False, True),
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9_longleaf_128kb",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=60,
    file_chunk_bytes=131_072,
    chunk_batch_size=4,
    micro_batch=1024,
    seed=0,
)
