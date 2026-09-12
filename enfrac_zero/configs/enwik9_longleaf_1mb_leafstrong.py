"""Sibling of enwik9_longleaf_1mb.py -- SAME patch_len_list=(4096,2048,1024,1) (see that file's
docstring for the long-context-on-leaves rationale), but capacity is now DELIBERATELY SKEWED
instead of uniform: level 3 (the terminal/LEAF, byte-prediction level, whose local seq_len=1024
is the long-context leaf this whole design is built around) gets the BIG trunk (d_model=512,
n_layers=3); levels 0-2 (root + middle) share a small trunk shape (d_model=256, n_layers=2).

share_trunk=False (REQUIRED -- share_trunk=True demands identical dims at every level, which
skewed capacity by definition breaks). This means each level now has its OWN independent weights
-- a genuinely different model from enwik9_longleaf_1mb.py's single-shared-trunk baseline, not
just a resize.

Real construction: total=19,668,480 (78.7MB), trainable=19,602,944 -- under the 100MB budget,
roughly comparable to (slightly under) the baseline's 87.1MB.

Hypothesis: does concentrating capacity at the LEAF (where the actual byte-level prediction
happens, with the longest local context) improve bpb relative to the uniform baseline, or relative
to the sibling root-strong config (enwik9_longleaf_1mb_rootstrong.py)?

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_longleaf_1mb_leafstrong.py --log_dir logs/enfrac_zero/enwik9_longleaf_1mb_leafstrong
"""

model = dict(
    patch_len_list=(4096, 2048, 1024, 1),
    d_model_list=(256, 256, 256, 512),
    n_layers_list=(2, 2, 2, 3),
    n_heads_list=(8, 8, 8, 8),
    mlp_mult_list=(4, 4, 4, 4),
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=False,
    use_flash_attn_list=(False, False, False, True),
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9_longleaf_1mb_leafstrong",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=60,
    file_chunk_bytes=1_048_576,
    chunk_batch_size=4,
    micro_batch=32,
    seed=0,
)
