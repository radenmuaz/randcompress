"""juz1.txt (~44KB) sanity/pipeline config -- still overparameterized, not tuned for a real
compression ratio (see enfrac/configs/quran_uthmani.py for that flavor of config).

  uv run python -m enfrac.train --config enfrac/configs/juz1.py --log_dir logs/enfrac/juz1

See enfrac_zero/configs/juz1.py for the no-HiRA baseline counterpart. patch_len_list=(512,64,8,1)
-> n_levels=4, level-0 n_timesteps = ceil(44443/512) = 87.
"""

model = dict(
    patch_len_list=(512, 64, 8, 1),
    d_model_list=(48, 48, 48, 48),
    n_layers_list=(3, 3, 3, 3),
    n_heads_list=(4, 4, 4, 4),
    mlp_mult_list=(2, 2, 2, 2),
    byte_embed_dim=128,
    patch_in_scheme="mean_pool",
    use_hira=True,
    hira_r=8,
    seed=0,
)

train = dict(
    dataset="quran_data/juz1.txt",
    log_dir="logs/enfrac/juz1",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="1.0",
    n_epochs=400,
    seed=0,
)
