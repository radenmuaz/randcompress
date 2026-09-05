"""Tiny sanity config (562B file) -- proves the pipeline (train -> compress -> decompress,
lossless round trip), not real compression: params vastly exceed the file.

  uv run python -m enfrac.train --config enfrac/configs/fatihah.py --log_dir logs/enfrac/fatihah

See enfrac_baseline/configs/fatihah.py for the no-HiRA counterpart (same architecture dims,
no use_hira/hira_r since enfrac_baseline.model.ModelConfig doesn't have those fields).

use_hira/hira_r/seed are spelled out explicitly here rather than left to ModelConfig's
defaults (use_hira=True, hira_r=4, seed=0 -- same values, but explicit so the config file is
the actual source of truth, not an implicit dataclass default someone has to go read the code
to discover). Both are saved into every checkpoint's config.json regardless (see
checkpoint.py's save_model(): json.dump(asdict(model.cfg), ...) covers every ModelConfig
field), but a config file that already states them is easier to diff/compare across runs.
"""

model = dict(
    patch_len_list=(8, 4, 1),
    d_model_list=(32, 32),
    n_layers_list=(2, 2),
    n_heads_list=(4, 4),
    mlp_mult_list=(2, 2),
    byte_embed_dim=64,
    use_hira=True,
    hira_r=4,
    seed=0,
)

train = dict(
    dataset="quran_data/surat_al-fatihah.txt",
    steps=2000,
    lr=5e-3,
    warmup_steps=50,
    log_every="200",
)

compress = dict(batch_size=8)
