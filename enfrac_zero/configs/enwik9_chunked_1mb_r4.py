"""enwik9 (1,000,000,000 B), FILE-CHUNKED, 1 MiB file chunks, patch0=1024 with downsample rate 4
(patch_len_list=(1024,256,64,16,4,1), 6 levels) -- chunk_n_timesteps = 1,048,576/1024 = 1024
exactly (1 MiB chunk = 1024 patches of 1024 bytes, i.e. "roughly 1024x1024").

patch_in_scheme="mean_pool" for every level (settled after weighing "linear" -- would need a
per-level tuple since linear's cost, patch_len*byte_embed_dim*d_model, is 469.8MB at level 0 alone
(patch=1024), only affordable at levels 2-5; kept simple with plain mean_pool everywhere instead).

Ratio-4 (not the earlier configs' ratio-8/32) keeps each level's own local seq_len at just 4,
smaller than every prior candidate's local seq_len -- deliberately chosen (alongside the smaller
1 MiB file chunk) to keep every level's row count within a SINGLE micro_batch=1024 jax.lax.scan
chunk wherever possible, per this session's real-TPU finding that crossing into a 2nd scan chunk
at any level adds a large, disproportionate ~11.5GB (see enwik9_chunked_1mb.py's docstring for
the full diagnostic). NOT the D2 shape (patch0=32768, ratio 32) -- a fresh design.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_chunked_1mb_r4.py --log_dir logs/enfrac_zero/enwik9_chunked_1mb_r4
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
    log_dir="logs/enfrac_zero/enwik9_chunked_1mb_r4",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=60,   # tripled from 20 -- the 20-epoch run's bpb was still improving epoch-over-epoch
                   # at the end (no plateau), and the goal here is heavy overfitting, not stopping
                   # early. Old checkpoints don't carry opt_state/rng, so this reruns from scratch
                   # (see .checkpoint.has_resumable_checkpoint) into a fresh log_dir; NEW runs from
                   # here on save full resumable state every epoch via on_epoch_end.
    file_chunk_bytes=1_048_576,
    chunk_batch_size=4,   # one file chunk per device, data-parallel across all 4 TPU chips via
                          # jax.pmap (see train.py) -- verified on real TPU hardware: stable,
                          # ~22-26GB HBM/chip (well under the 30.75GB budget), ~4x epoch speedup
                          # over chunk_batch_size=1 (single-device) at the same per-step wall time.
    micro_batch=1024,
    seed=0,
)
