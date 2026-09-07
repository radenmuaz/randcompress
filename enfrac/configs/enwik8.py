"""enwik8 (100,000,000 B) -- TPU-scale HiRA config; not intended for a quick local run.

patch_len_list=(262144,32768,4096,512,64,8,1) -> n_levels=7, level-0 n_timesteps =
ceil(100,000,000/262144) = 382. See enfrac_zero/configs/enwik8.py's docstring for why
patch_in_scheme="mean_pool" is required at this patch_len_list[0] scale (the "linear" scheme's
flatten+dense patch_in[0] would be intractable: ~262144*byte_embed_dim input features).

  uv run python -m enfrac.train --config enfrac/configs/enwik8.py --log_dir logs/enfrac/enwik8
  uv run python -m enfrac.compress --ckpt logs/enfrac/enwik8 --input datasets/enwik8 --output logs/enfrac/enwik8_compressed
  uv run python -m enfrac.decompress --bundle logs/enfrac/enwik8_compressed --output /tmp/enwik8_out --verify datasets/enwik8
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
    use_hira=True,
    hira_r=32,
    seed=0,
)

train = dict(
    dataset="datasets/enwik8",
    log_dir="logs/enfrac/enwik8",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    seed=0,
)
