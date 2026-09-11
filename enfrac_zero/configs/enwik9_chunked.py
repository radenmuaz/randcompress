"""enwik9 (1,000,000,000 B), FILE-CHUNKED -- directly-init baseline counterpart to
enfrac/configs/enwik9_chunked.py (see that file's docstring for the full rationale: why file
chunking replaces the old whole-file enwik9.py config, the patch0=32768/6-level sizing choice,
why patch_in_scheme stays "mean_pool" everywhere, and why training's chunk_batch_size starts
conservative). This file only differs in d_model/n_layers/hira (no HiRA, everything trainable).

Sized via real jax.eval_shape param counts: d_model_list=448 (x6 levels, trimmed from the old
8-level config's *8), n_layers=7, n_heads=8, mlp_mult=4 -> total 24,357,888 params (97.4MB fp32),
essentially the same budget as the old 8-level config's 100.0MB.

file_chunk_bytes=10,485,760 (10 MiB) -> n_file_chunks=96, patch_len_list=(32768,4096,512,64,8,1)
-> chunk_n_timesteps=320 (same magnitude as the old whole-file config's 477 -- decode wall-clock
scales directly with this number, not with chunk_batch_size, see decompress.py's docstring).

UNTESTED AT THIS SCALE -- verify on TPU (ideally a few chunks/epochs first) before a full run.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_chunked.py --log_dir logs/enfrac_zero/enwik9_chunked
  uv run python -m enfrac_zero.compress --ckpt logs/enfrac_zero/enwik9_chunked --input datasets/enwik9 --output logs/enfrac_zero/enwik9_chunked_compressed --chunk_batch_size 16
  uv run python -m enfrac_zero.decompress --bundle logs/enfrac_zero/enwik9_chunked_compressed --output /tmp/enwik9_zero_chunked_out --verify datasets/enwik9

See enfrac/configs/enwik9_chunked.py for the HiRA counterpart.
"""

model = dict(
    patch_len_list=(32768, 4096, 512, 64, 8, 1),
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
    log_dir="logs/enfrac_zero/enwik9_chunked",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    file_chunk_bytes=10_485_760,
    chunk_batch_size=1,   # bumped down from 2 -- 2 OOM'd on real TPU (328.23G requested vs
                          # 30.75G HBM) despite being far smaller than the old whole-file config's
                          # blowup; see enwik9_chunked.py's docstring for the chunk_batch_size
                          # tradeoff this confirms in practice, not just in theory. Even
                          # chunk_batch_size=1 still OOM'd (226.11G) -- the drop from cbs=2's 328G
                          # was only ~31%, not ~50%, showing chunk_batch_size isn't the dominant
                          # cost here (consistent with CLAUDE.md's note that remat_depth is INERT
                          # for levels 1+ in the current _train_recurse -- only level 0 uses it --
                          # so micro_batch is the one remaining real lever for levels 1+ memory).
    micro_batch=1024,     # bumped down from the 8192 default -- see chunk_batch_size's comment;
                          # testing whether shrinking _train_recurse's own scan-chunk size (not
                          # yet tried this session for the file-chunked config) actually reduces
                          # peak HBM the way it's supposed to, or whether XLA is fusing across scan
                          # boundaries regardless (the same unresolved failure mode as the old
                          # whole-file config, see docs/enwik9_scaling_calcs.md).
    seed=0,
)
