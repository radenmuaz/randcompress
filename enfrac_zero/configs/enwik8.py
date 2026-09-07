"""enwik8 (100,000,000 B) -- directly-init baseline counterpart to enfrac/configs/enwik8.py.
TPU-scale; not intended for a quick local run.

patch_len_list=(262144,32768,4096,512,64,8,1) -> n_levels=7, level-0 n_timesteps =
ceil(100,000,000/262144) = 382 (uniform seq_len=8 at every level transition). A big
patch_len_list[0] keeps level-0's own attention sequence short (382, not tens of thousands) --
see model.py's module docstring for why patch_in_scheme="mean_pool" (params/compute independent
of patch size) is required here: patch_len_list[0]=262144 would make the "linear"
scheme's patch_in[0] an intractable ~262144*byte_embed_dim -> d_model_list[0] dense layer.

  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik8.py --log_dir logs/enfrac_zero/enwik8
  uv run python -m enfrac_zero.compress --ckpt logs/enfrac_zero/enwik8 --input datasets/enwik8 --output logs/enfrac_zero/enwik8_compressed
  uv run python -m enfrac_zero.decompress --bundle logs/enfrac_zero/enwik8_compressed --output /tmp/enwik8_zero_out --verify datasets/enwik8
"""

model = dict(
    patch_len_list=(262144, 32768, 4096, 512, 64, 8, 1),
    d_model_list=(96, 96, 96, 96, 96, 96, 96),
    n_layers_list=(6, 6, 6, 6, 6, 6, 6),
    n_heads_list=(6, 6, 6, 6, 6, 6, 6),
    mlp_mult_list=(3, 3, 3, 3, 3, 3, 3),
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    seed=0,
)

train = dict(
    dataset="datasets/enwik8",
    log_dir="logs/enfrac_zero/enwik8",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    seed=0,
)
