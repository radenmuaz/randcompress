"""Sibling of enwik9_longleaf_1mb.py -- CONTROLLED ABLATION, not a capacity skew: SAME uniform
capacity philosophy (every level gets the same trunk shape), but share_trunk=False instead of
True, with d_model/n_layers chosen so the TOTAL param count stays close to the baseline's
87.1MB rather than ballooning ~4x (one trunk reused 4x vs four independent trunks of the same
size -- naively reusing d_model=640/n_layers=3 unshared measured 323.1MB on real construction).

d_model=336, n_layers=3 (uniform across all 4 levels, share_trunk=False): real construction
total=22,520,080 (90.1MB) -- within ~3% of the baseline's 21,779,456 (87.1MB), so any bpb
difference between this and enwik9_longleaf_1mb.py isolates the effect of WEIGHT SHARING itself
(same total capacity, same architecture, only independent-vs-shared trunk weights differ), not a
capacity difference.

Motivation: the baseline (share_trunk=True) trains one trunk that's reused, unchanged, at every
recursion level -- same weights asked to do 4 structurally different jobs (root's long-range
attention, two middle levels, and the leaf's long local context) at once. This tests whether that
forced sharing is itself hurting bpb/compression, independent of total capacity.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_longleaf_1mb_unshared.py --log_dir logs/enfrac_zero/enwik9_longleaf_1mb_unshared
"""

model = dict(
    patch_len_list=(4096, 2048, 1024, 1),
    d_model_list=(336,) * 4,
    n_layers_list=(3,) * 4,
    n_heads_list=(8,) * 4,
    mlp_mult_list=(4,) * 4,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=False,
    use_flash_attn_list=(False, False, False, True),
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9_longleaf_1mb_unshared",
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
