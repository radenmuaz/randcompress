"""enwik9 (1,000,000,000 B), FILE-CHUNKED, HiRA, 512 KiB (2^19) file chunks -- smaller than
enwik9_chunked_1mb_r4.py's 1 MiB, testing whether the OOM found there (56.18G required at
chunk_batch_size=4/pmap, d_model=1216) is fixed by shrinking the chunk the same way it was for
enfrac_zero's 1mb_r4 config. SAME patch_len_list=(1024,256,64,16,4,1) (ratio-4 downsample,
patch0=1024) -- only file_chunk_bytes shrinks, so chunk_n_timesteps drops from 1024 to 512.

d_model BUMPED from 1216 to 1408, hira_r LOWERED from 165 to 128 (real construction, not
estimated) so the FROZEN base exceeds the raw file's own 1,000,000,000 bytes while trainable
stays under 100MB: frozen=1,127.8MB > file size, trainable=83.1MB. The point: HiRA's frozen (W0,
A) is regenerated from `seed` at load time, never stored -- so the compressed bundle only pays
for the 83.1MB trainable B (+ range-coded residual), regardless of how much bigger the frozen
base gets than the data itself.

n_file_chunks = ceil(1,000,000,000 / 524,288) = 1908.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9_chunked_512kb_r4.py --log_dir logs/enfrac/enwik9_chunked_512kb_r4
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
    log_dir="logs/enfrac/enwik9_chunked_512kb_r4",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    file_chunk_bytes=524_288,
    chunk_batch_size=1,
    micro_batch=1024,
    seed=0,
)
