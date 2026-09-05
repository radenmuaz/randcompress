"""juz1.txt (~44KB) sanity/pipeline config -- still overparameterized, not tuned for a real
compression ratio (see enfrac/configs/quran_uthmani.py for that flavor of config).

  uv run python -m enfrac.train --config enfrac/configs/juz1.py --log_dir logs/enfrac/juz1

See enfrac_zero/configs/juz1.py for the no-HiRA counterpart. use_hira/hira_r/seed spelled
out explicitly (see fatihah.py's docstring for why) -- same values as ModelConfig's defaults.
"""

model = dict(
    patch_len_list=(64, 16, 4, 1),
    d_model_list=(48, 48, 48),
    n_layers_list=(3, 3, 3),
    n_heads_list=(4, 4, 4),
    mlp_mult_list=(2, 2, 2),
    byte_embed_dim=128,
    use_hira=True,
    hira_r=4,
    seed=0,
)

train = dict(
    dataset="quran_data/juz1.txt",
    steps=4000,
    lr=3e-3,
    warmup_steps=100,
    log_every="500",
)

compress = dict(batch_size=32)
