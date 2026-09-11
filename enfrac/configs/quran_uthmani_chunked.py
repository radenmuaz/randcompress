"""quran-uthmani.txt (~1.4MB), FILE-CHUNKED starter test (HiRA variant) -- see
enfrac_zero/configs/quran_uthmani_chunked.py for the full rationale (identical architecture/
chunking, only HiRA vs plain linears differ). file_chunk_bytes=12288 (3x patch_len_list[0]=4096)
-> n_file_chunks = ceil(1,359,946/12288) = 111, chunk_batch_size=16.

  uv run python -m enfrac.train --config enfrac/configs/quran_uthmani_chunked.py --log_dir logs/enfrac/quran_uthmani_chunked
  uv run python -m enfrac.compress --config enfrac/configs/quran_uthmani_chunked.py --ckpt logs/enfrac/quran_uthmani_chunked --input quran_data/quran-uthmani.txt --output logs/enfrac/quran_uthmani_chunked_compressed
  uv run python -m enfrac.decompress --bundle logs/enfrac/quran_uthmani_chunked_compressed --output /tmp/quran_chunked_out.txt --verify quran_data/quran-uthmani.txt
"""

model = dict(
    patch_len_list=(4096, 512, 64, 8, 1),
    d_model_list=(64, 64, 64, 64, 64),
    n_layers_list=(4, 4, 4, 4, 4),
    n_heads_list=(4, 4, 4, 4, 4),
    mlp_mult_list=(2, 2, 2, 2, 2),
    byte_embed_dim=128,
    patch_in_scheme="mean_pool",
    use_hira=True,
    hira_r=8,
    seed=0,
)

train = dict(
    dataset="quran_data/quran-uthmani.txt",
    log_dir="logs/enfrac/quran_uthmani_chunked",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="1.0",
    n_epochs=20,
    file_chunk_bytes=12288,
    chunk_batch_size=16,
    seed=0,
)
