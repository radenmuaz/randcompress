"""enwik9 (1,000,000,000 B), sibling of enwik9_hira_2xfrozen_128kb.py -- SAME model/hira_r
(d_model=496, n_layers=7, hira_r=478: frozen=48,491,136 ~= 2x enfrac_zero's total, trainable=
24,313,280 ~= enfrac_zero's trainable -- see that file's docstring for the search and hypothesis
being tested), only file_chunk_bytes/chunk_batch_size differ.

64 KiB (2^16) file chunks -- chunk_batch_size=64 (per_device=16), same as
enwik9_hira_backbone_64kb.py used successfully (~26GB/22GB per chip @ 37.7M total params) --
this model is ~1.9x bigger (72.8M total) so watch for OOM and back off (32, then 16) if needed.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9_hira_2xfrozen_64kb.py --log_dir logs/enfrac/enwik9_hira_2xfrozen_64kb
"""

model = dict(
    patch_len_list=(1024, 256, 64, 16, 4, 1),
    d_model_list=(496,) * 6,
    n_layers_list=(7,) * 6,
    n_heads_list=(8,) * 6,
    mlp_mult_list=(4,) * 6,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    use_hira=True,
    hira_r=478,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac/enwik9_hira_2xfrozen_64kb",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=60,
    file_chunk_bytes=65_536,
    chunk_batch_size=64,
    micro_batch=1024,
    seed=0,
)
