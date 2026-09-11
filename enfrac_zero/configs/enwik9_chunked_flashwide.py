"""enwik9 (1,000,000,000 B), FILE-CHUNKED -- "flash-wide" candidate: the MINIMUM possible level
count (2: level 0 + one terminal level) at the same 10 MiB file chunk, with patch_len_list=(1024,1)
chosen SPECIFICALLY so BOTH levels get a genuinely long sequence (level0: chunk_n_timesteps=10240,
level1/terminal: seq_len=1024) instead of the short seq_len=8-32 levels 1+ had in every earlier
6-level/4-level candidate. Both 10240 and 1024 are exact multiples of 128
(10,485,760/1024=10240.0, 1024/1=1024), so flash_attention's pallas TPU kernel is usable at BOTH
levels without any padding trick (unlike the earlier flashpad experiment, where only level 0 could
ever satisfy the >=128/divisible-by-128 requirement -- levels 1+'s seq_len=8 there was an
architectural constant, unfixable by chunk-size choices).

Rationale for going this shallow: extensive real-TPU testing on 6-level and 4-level (D2)
candidates never got the flat-scan + per-level-checkpoint HBM requirement below ~120.9G (still
~3.9x over the 30.75G v4-8 budget), with diminishing returns per fix (26%->11%->14%) -- see
enwik9_chunked_d2.py's docstring. Flash attention was ALSO tested directly on those (level-0-only,
since levels 1+ there had seq_len=8/9, unusable regardless of padding) and had ZERO measurable
effect (140.72G vs 140.71G) BECAUSE the score matrices at those short sequences are only a few MB
-- flash's whole benefit (avoiding O(T^2) score-matrix materialization) only matters when T is
actually large. This config is the first candidate where that's true at EVERY level, so flash
attention becomes a real, relevant lever rather than a no-op.

Sizing (real jax.eval_shape): d_model=448, n_layers=7 (UNCHANGED from every earlier candidate, for
direct comparability) -> total 23,096,320 params (92.4MB fp32), close to the ~100MB target -- only
2 levels needs almost no extra params beyond the shared trunk itself.

UNTESTED AT THIS SCALE.

  FLAT_SCAN=1 LEVEL_CKPT=1 uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_chunked_flashwide.py --log_dir logs/enfrac_zero/enwik9_chunked_flashwide
"""

model = dict(
    patch_len_list=(1024, 1),
    d_model_list=(448,) * 2,
    n_layers_list=(7,) * 2,
    n_heads_list=(8,) * 2,
    mlp_mult_list=(4,) * 2,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    use_flash_attn_list=(True, True),
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9_chunked_flashwide",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    file_chunk_bytes=10_485_760,
    chunk_batch_size=1,
    micro_batch=1024,
    seed=0,
)
