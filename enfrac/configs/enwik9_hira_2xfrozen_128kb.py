"""enwik9 (1,000,000,000 B), HiRA hypothesis test: does a bigger RANDOM (frozen, never
pretrained) backbone help bpb at all, when the TRAINABLE budget is held roughly equal to
enfrac_zero's fully-trained baseline (tpu5/tpu6, d_model=448/n_layers=7, 24,292,352 trainable)?

Sized via real ByteFractalGen construction (not eval_shape -- see CLAUDE.md's note on
eval_shape's trainable-count quirk), searching (d_model, hira_r) jointly since hira_r affects
BOTH frozen (A) and trainable (B) simultaneously:

  d_model=496, n_layers=7, hira_r=478:
    frozen    = 48,491,136  (target: 2x enfrac_zero's 24,357,888 total = 48,715,776, -0.5%)
    trainable = 24,313,280  (target: enfrac_zero's 24,292,352 trainable,            +0.09%)
    total     = 72,804,416

If bpb ends up similar to or worse than enfrac_zero's 1mb_r4/128kb_r4 runs (which start at
bpb~3.1-3.2, epoch1) despite 2x the frozen capacity, that's evidence the RANDOM backbone itself
isn't adding useful representational capacity (only the trainable budget matters) -- consistent
with the earlier enwik9_hira_backbone_* runs, which found a ~20%-trainable HiRA over a SAME-SIZE
random backbone converges far worse (bpb=7.56 @ epoch1) than full training at equal total size.
This run controls for trainable budget instead, isolating the frozen-backbone-size question.

128 KiB (2^17) file chunks -- chunk_batch_size=32 (per_device=8), same as
enwik9_hira_backbone_128kb.py used successfully (~26GB/22GB per chip @ 37.7M total params) --
this model is ~1.9x bigger (72.8M total) so watch for OOM and back off (16, then 8) if needed.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9_hira_2xfrozen_128kb.py --log_dir logs/enfrac/enwik9_hira_2xfrozen_128kb
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
    log_dir="logs/enfrac/enwik9_hira_2xfrozen_128kb",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=60,
    file_chunk_bytes=131_072,
    chunk_batch_size=32,
    micro_batch=1024,
    seed=0,
)
