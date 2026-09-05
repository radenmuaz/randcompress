"""Tiny sanity config (562B file) -- proves the pipeline (train -> compress -> decompress,
lossless round trip), not real compression: params vastly exceed the file.

  uv run python -m enfrac_zero.train --config enfrac_zero/configs/fatihah.py --log_dir logs/enfrac_zero/fatihah

See enfrac/configs/fatihah.py for the HiRA counterpart (same architecture dims, plus
use_hira/hira_r since enfrac.model.ModelConfig has those fields and this one doesn't -- the
baseline is a plain, no-adapter model, every weight here is directly trainable).

`seed` is spelled out explicitly rather than left to ModelConfig's default (seed=0 -- same
value, but explicit so the config file is the actual source of truth). It's saved into every
checkpoint's config.json regardless (checkpoint.py's save_model(): json.dump(asdict(model.cfg),
...) covers every ModelConfig field), but stating it here makes it easier to diff across runs.
"""

model = dict(
    patch_len_list=(8, 4, 1),
    d_model_list=(32, 32),
    n_layers_list=(2, 2),
    n_heads_list=(4, 4),
    mlp_mult_list=(2, 2),
    byte_embed_dim=64,
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
