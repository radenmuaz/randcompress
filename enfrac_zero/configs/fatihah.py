"""Tiny sanity config (562B file) -- proves the pipeline (train -> compress -> decompress,
lossless round trip), not real compression: params vastly exceed the file.

  uv run python -m enfrac_zero.train --config enfrac_zero/configs/fatihah.py --log_dir logs/enfrac_zero/fatihah

See enfrac/configs/fatihah.py for the HiRA counterpart. patch_len_list=(8,4,1) -> n_levels=3,
level-0 n_timesteps = ceil(562/8) = 71 (see model.py's module docstring for why "timestep" means
one 8-byte patch, not one byte).
"""

model = dict(
    patch_len_list=(8, 4, 1),
    d_model_list=(32, 32, 32),
    n_layers_list=(2, 2, 2),
    n_heads_list=(4, 4, 4),
    mlp_mult_list=(2, 2, 2),
    byte_embed_dim=64,
    patch_in_scheme="mean_pool",
    seed=0,
)

train = dict(
    dataset="quran_data/surat_al-fatihah.txt",
    log_dir="logs/enfrac_zero/fatihah",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="1.0",
    n_epochs=400,
    seed=0,
)
