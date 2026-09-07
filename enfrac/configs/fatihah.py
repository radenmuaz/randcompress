"""Tiny sanity config (562B file) -- proves the pipeline (train -> compress -> decompress,
lossless round trip), not real compression: params (mostly frozen W0/A, never saved) vastly
exceed the file; only the trainable HiRA B factors + root_cond count toward the bundle.

  uv run python -m enfrac.train --config enfrac/configs/fatihah.py --log_dir logs/enfrac/fatihah

See enfrac_zero/configs/fatihah.py for the no-HiRA baseline counterpart. patch_len_list=(8,4,1)
-> n_levels=3, level-0 n_timesteps = ceil(562/8) = 71.
"""

model = dict(
    patch_len_list=(8, 4, 1),
    d_model_list=(32, 32, 32),
    n_layers_list=(2, 2, 2),
    n_heads_list=(4, 4, 4),
    mlp_mult_list=(2, 2, 2),
    byte_embed_dim=64,
    patch_in_scheme="mean_pool",
    use_hira=True,
    hira_r=4,
    seed=0,
)

train = dict(
    dataset="quran_data/surat_al-fatihah.txt",
    log_dir="logs/enfrac/fatihah",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="1.0",
    n_epochs=800,
    seed=0,
)
