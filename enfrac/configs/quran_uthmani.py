"""quran-uthmani.txt (~1.4MB) -- the real-compression demo config (see README/CLAUDE.md).

  uv run python -m enfrac.train --config enfrac/configs/quran_uthmani.py --log_dir logs/enfrac/quran_uthmani

See enfrac_zero/configs/quran_uthmani.py for the no-HiRA baseline counterpart.
patch_len_list=(4096,512,64,8,1) -> n_levels=5, level-0 n_timesteps = ceil(1,359,946/4096) = 333.
"""

model = dict(
    patch_len_list=(4096, 512, 64, 8, 1),
    d_model_list=(64, 64, 64, 64, 64),
    n_layers_list=(4, 4, 4, 4, 4),
    n_heads_list=(4, 4, 4, 4, 4),
    mlp_mult_list=(2, 2, 2, 2, 2),
    byte_embed_dim=128,
    patch_in_scheme="mean_pool",
    use_hira=True,
    hira_r=8,
    seed=0,
)

train = dict(
    dataset="quran_data/quran-uthmani.txt",
    log_dir="logs/enfrac/quran_uthmani",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="1.0",
    n_epochs=20,
    seed=0,
)
