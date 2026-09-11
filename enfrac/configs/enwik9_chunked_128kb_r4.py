"""enwik9 (1,000,000,000 B), FILE-CHUNKED, HiRA, 128 KiB (2^17) file chunks -- smaller still than
enwik9_chunked_512kb_r4.py's 512 KiB (2^19), same model shape (d_model=1408, hira_r=128,
patch_len_list=(1024,256,64,16,4,1)) -- only file_chunk_bytes shrinks further, so
chunk_n_timesteps drops to 128 (from 512).

Same sizing as enwik9_chunked_512kb_r4.py: frozen=1,127.8MB > file's own 1,000,000,000 bytes,
trainable=83.1MB (under 100MB) -- real jax.eval_shape construction.

n_file_chunks = ceil(1,000,000,000 / 131,072) = 7630.

Run alongside enwik9_chunked_512kb_r4.py (different file_chunk_bytes, same everything else) to
find the largest file_chunk_bytes that still fits the 30.75G v4-8 HBM budget for this bigger
d_model=1408 model -- if 512kb OOMs, try this 128kb variant; keep halving further if needed.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9_chunked_128kb_r4.py --log_dir logs/enfrac/enwik9_chunked_128kb_r4
"""

model = dict(
    patch_len_list=(1024, 256, 64, 16, 4, 1),
    d_model_list=(1408,) * 6,
    n_layers_list=(8,) * 6,
    n_heads_list=(8,) * 6,
    mlp_mult_list=(4,) * 6,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    use_hira=True,
    hira_r=128,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac/enwik9_chunked_128kb_r4",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    file_chunk_bytes=131_072,
    chunk_batch_size=8,
    micro_batch=1024,
    seed=0,
)
