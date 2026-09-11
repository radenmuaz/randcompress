"""enwik9 (1,000,000,000 B) -- TPU-scale HiRA config, ~1GB total model / ~100MB HiRA (10%) trainable.
Sized via jax.eval_shape param counts (not guessed): d_model_list=1216 * 8 levels, n_layers=8,
n_heads=8, mlp_mult=4, hira_r=165 -> total 244,420,288 params (977.7MB fp32), trainable (HiRA B)
23,940,288 params (95.8MB, 9.8% of total).

patch_len_list=(2097152,262144,32768,4096,512,64,8,1) -> n_levels=8 (one level deeper than
enwik8's 7-level config: enwik9 is 10x enwik8's byte count, so the fractal needs one more
8x-ratio level to keep level-0's own sequence length in the same ballpark as enwik8's).
level-0 n_timesteps = ceil(1,000,000,000/2,097,152) = 477 (vs enwik8's 382 -- same order of
magnitude, so level-0 attention stays cheap). patch_in_scheme="mean_pool" is required at this
patch_len_list[0] scale for the same reason as enwik8 (see enfrac_zero/configs/enwik9.py).

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9.py --log_dir logs/enfrac/enwik9
  uv run python -m enfrac.compress --ckpt logs/enfrac/enwik9 --input datasets/enwik9 --output logs/enfrac/enwik9_compressed
  uv run python -m enfrac.decompress --bundle logs/enfrac/enwik9_compressed --output /tmp/enwik9_out --verify datasets/enwik9
"""

model = dict(
    patch_len_list=(2097152, 262144, 32768, 4096, 512, 64, 8, 1),
    d_model_list=(1216,) * 8,
    n_layers_list=(8,) * 8,
    n_heads_list=(8,) * 8,
    mlp_mult_list=(4,) * 8,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    use_hira=True,
    hira_r=165,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac/enwik9",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    seed=0,
)
