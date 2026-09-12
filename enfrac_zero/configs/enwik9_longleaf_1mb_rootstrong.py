"""Sibling of enwik9_longleaf_1mb_leafstrong.py -- SAME skewed-capacity idea, INVERTED: level 0
(the ROOT, level-0's own long-range attention over the whole file chunk's timesteps) gets the big
trunk (d_model=512, n_layers=3) instead; levels 1-3 (middle + leaf) share the small trunk shape
(d_model=256, n_layers=2). share_trunk=False (required, see leafstrong's docstring).

Real construction: total=19,603,200 (78.4MB), trainable=19,537,664 -- essentially the same total
size as enwik9_longleaf_1mb_leafstrong.py (78.7MB), so any bpb difference between the two isolates
WHERE capacity helps, not how much capacity there is.

Hypothesis: does concentrating capacity at the ROOT (long-range structure across the whole file
chunk) improve bpb relative to the leaf-strong sibling, or the uniform baseline
(enwik9_longleaf_1mb.py)?

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_longleaf_1mb_rootstrong.py --log_dir logs/enfrac_zero/enwik9_longleaf_1mb_rootstrong
"""

model = dict(
    patch_len_list=(4096, 2048, 1024, 1),
    d_model_list=(512, 256, 256, 256),
    n_layers_list=(3, 2, 2, 2),
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
    log_dir="logs/enfrac_zero/enwik9_longleaf_1mb_rootstrong",
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
