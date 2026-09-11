"""quran-uthmani.txt (~1.4MB), FILE-CHUNKED starter test -- same model/architecture as
quran_uthmani.py, but the file is split into independent, weight-shared file chunks (see
train.py's make_file_chunks / ByteFractalGen.__call__'s docstring / CLAUDE.md's "file chunking"
section) instead of one long level-0 sequence over the whole file. This is the "starter" test for
that feature: file_chunk_bytes=12288 (3x patch_len_list[0]=4096, ~12KB, closest clean multiple to
the requested "~10k") -> n_file_chunks = ceil(1,359,946/12288) = 111, each chunk's own level-0
sequence is just 3 timesteps (vs 333 for the whole-file version) -- chunk_batch_size=16 samples
16 of those 111 chunks (WITH replacement) per training step.

  uv run python -m enfrac_zero.train --config enfrac_zero/configs/quran_uthmani_chunked.py --log_dir logs/enfrac_zero/quran_uthmani_chunked
  uv run python -m enfrac_zero.compress --config enfrac_zero/configs/quran_uthmani_chunked.py --ckpt logs/enfrac_zero/quran_uthmani_chunked --input quran_data/quran-uthmani.txt --output logs/enfrac_zero/quran_uthmani_chunked_compressed
  uv run python -m enfrac_zero.decompress --bundle logs/enfrac_zero/quran_uthmani_chunked_compressed --output /tmp/quran_chunked_out.txt --verify quran_data/quran-uthmani.txt

See enfrac/configs/quran_uthmani_chunked.py for the HiRA counterpart.
"""

model = dict(
    patch_len_list=(4096, 512, 64, 8, 1),
    d_model_list=(64, 64, 64, 64, 64),
    n_layers_list=(4, 4, 4, 4, 4),
    n_heads_list=(4, 4, 4, 4, 4),
    mlp_mult_list=(2, 2, 2, 2, 2),
    byte_embed_dim=128,
    patch_in_scheme="mean_pool",
    seed=0,
)

train = dict(
    dataset="quran_data/quran-uthmani.txt",
    log_dir="logs/enfrac_zero/quran_uthmani_chunked",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="1.0",
    n_epochs=20,
    file_chunk_bytes=12288,
    chunk_batch_size=16,
    seed=0,
)
