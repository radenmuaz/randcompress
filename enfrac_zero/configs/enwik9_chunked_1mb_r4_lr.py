"""enwik9 (1,000,000,000 B), FILE-CHUNKED -- SAME shape/sizing as enwik9_chunked_1mb_r4.py (that
config is confirmed working and converging well on real TPU hardware: bpb 5.7->1.84 by epoch 16/20,
~62% byte accuracy, ~3.06x estimated compression ratio, ~5.05s/step steady state via pmap across
4 chips). This variant ONLY changes lr: 3e-3 -> 1e-2 (~3.3x more aggressive) to test whether a
higher learning rate converges faster (fewer epochs to reach a given bpb) without destabilizing
training (grad_clip=1.0 already provides some protection against divergence).

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_chunked_1mb_r4_lr.py --log_dir logs/enfrac_zero/enwik9_chunked_1mb_r4_lr
"""

model = dict(
    patch_len_list=(1024, 256, 64, 16, 4, 1),
    d_model_list=(448,) * 6,
    n_layers_list=(7,) * 6,
    n_heads_list=(8,) * 6,
    mlp_mult_list=(4,) * 6,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9_chunked_1mb_r4_lr",
    lr=1e-2,   # bumped from 3e-3 -- more aggressive, testing faster convergence
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    file_chunk_bytes=1_048_576,
    chunk_batch_size=4,
    micro_batch=1024,
    seed=0,
)
