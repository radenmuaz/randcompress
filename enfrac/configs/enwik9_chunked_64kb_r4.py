"""enwik9 (1,000,000,000 B), FILE-CHUNKED, HiRA, 64 KiB (2^16) file chunks -- smaller than
enwik9_chunked_128kb_r4.py's 128 KiB (2^17). SAME model shape (d_model=1408, hira_r=128,
patch_len_list=(1024,256,64,16,4,1)) -- only file_chunk_bytes shrinks further, so
chunk_n_timesteps drops to 64 (from 128).

Motivation: enwik9_chunked_128kb_r4.py fits and trains at chunk_batch_size=1 (single device),
but is impractically slow (steps_per_epoch=7630, ~2.5-4s/step even single-device -> ~100+ hours
for 20 epochs). This config's smaller chunk is being tried SPECIFICALLY with pmap
(chunk_batch_size=4, one chunk/device across all 4 TPU chips) to see if it both (a) fits in HBM
at this smaller size (512kb and the pmap-4x version of 128kb needed checking) and (b) gives a
practical per-epoch wall-clock via the same ~4x pmap speedup seen on enfrac_zero's 1mb_r4 config.

Same sizing as enwik9_chunked_128kb_r4.py / enwik9_chunked_512kb_r4.py: frozen=1,127.8MB > file's
own 1,000,000,000 bytes, trainable=83.1MB (under 100MB) -- real jax.eval_shape construction.

n_file_chunks = ceil(1,000,000,000 / 65,536) = 15259.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9_chunked_64kb_r4.py --log_dir logs/enfrac/enwik9_chunked_64kb_r4 --chunk_batch_size 4
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
    log_dir="logs/enfrac/enwik9_chunked_64kb_r4",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    file_chunk_bytes=65_536,
    chunk_batch_size=16,
    micro_batch=1024,
    seed=0,
)
