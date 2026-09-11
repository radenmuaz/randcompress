"""enwik9 (1,000,000,000 B) -- directly-init baseline counterpart to enfrac/configs/enwik9.py,
sized to ~100MB / 25M params total (fully trainable, no HiRA). Sized via jax.eval_shape param
counts (not guessed): d_model_list=448 * 8 levels, n_layers=7, n_heads=8, mlp_mult=4 -> total
24,988,672 params (100.0MB fp32).

patch_len_list=(2097152,262144,32768,4096,512,64,8,1) -> n_levels=8, one level deeper than
enwik8's 7-level config (enwik9 is 10x enwik8's byte count, so the fractal needs one more
8x-ratio level to keep level-0's own attention sequence in the same ballpark). level-0
n_timesteps = ceil(1,000,000,000/2,097,152) = 477. patch_in_scheme="mean_pool" is required at
this patch_len_list[0] scale for the same reason as enwik8's config: the "linear" scheme's
flatten+dense patch_in[0] would need an intractable ~2,097,152*byte_embed_dim input.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9.py --log_dir logs/enfrac_zero/enwik9
  uv run python -m enfrac_zero.compress --ckpt logs/enfrac_zero/enwik9 --input datasets/enwik9 --output logs/enfrac_zero/enwik9_compressed
  uv run python -m enfrac_zero.decompress --bundle logs/enfrac_zero/enwik9_compressed --output /tmp/enwik9_zero_out --verify datasets/enwik9
"""

model = dict(
    patch_len_list=(2097152, 262144, 32768, 4096, 512, 64, 8, 1),
    d_model_list=(448,) * 8,
    n_layers_list=(7,) * 8,
    n_heads_list=(8,) * 8,
    mlp_mult_list=(4,) * 8,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    seed=0,
)
