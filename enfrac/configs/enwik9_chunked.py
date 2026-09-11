"""enwik9 (1,000,000,000 B), FILE-CHUNKED (candidate B from the chunked-sizing discussion) --
successor to enfrac/configs/enwik9.py, which trains level 0 as ONE long sequence over the whole
file (patch_len_list[0]=2,097,152, 477 level-0 timesteps, 8 levels total). That config never
finished a real training run: even with the levels-1+ memory-chunking fixes (see CLAUDE.md's
"Training-time memory chunking" section), it still hit an unresolved XLA scan-fusion compiler
crash at the deepest levels (self-similar consecutive-level shapes, `rows` reaching 125M+ by the
terminal level -- see docs/enwik9_scaling_calcs.md's "Problem Solving" notes).

This config instead splits the file into independent, weight-shared 10 MiB chunks (see
train.py's make_file_chunks / CLAUDE.md's "file chunking" section) -- each chunk gets its OWN
short level-0 sequence, so patch_len_list[0] no longer needs to be huge just to keep the WHOLE
FILE's timestep count low. Sized via real jax.eval_shape param counts + arithmetic (not guessed,
see docs/enwik9_scaling_calcs.md's "file-chunked candidates" section for the full A/B/C
comparison this was picked from):

  file_chunk_bytes=10,485,760 (10 MiB) -> n_file_chunks = ceil(1,000,000,000/10,485,760) = 96
  (last chunk padded by ~6.6MB, warn_padding). patch_len_list=(32768,4096,512,64,8,1) -- ONE
  LEVEL SHALLOWER than the old whole-file config (6 vs 8) -- chunk_n_timesteps =
  10,485,760/32768 = 320, matching the old config's whole-file 477 timesteps in order of
  magnitude (this matters a lot: generate()'s level-0 decode loop is sequential PER TIMESTEP,
  batched only across chunks never across timesteps, so chunk_n_timesteps directly multiplies
  total decode wall-clock -- see decompress.py's module docstring / CLAUDE.md's KIV section).
  d_model/n_layers/n_heads/mlp_mult/hira_r kept identical to the old config's per-level values
  (just trimmed to 6 elements, share_trunk=True makes trainable param count ~level-count-
  invariant anyway) -> total 239,552,064 params (958.2MB fp32), essentially the same budget as
  the old 8-level config's 977.7MB.

  patch_in_scheme="mean_pool" for ALL levels (not "linear" anywhere): the "linear" scheme's cost
  is patch_len*byte_embed_dim*d_model -- at level 0 (patch_len=32768) that's 10.2B params (40.8GB
  fp32), ~43x this whole model's entire budget, in ONE projection. Even level 2 (patch_len=512)
  would cost 637.5MB alone. Only patch_len<=64 levels are cheap enough for "linear" to make sense
  here (patch_in_scheme also accepts a per-level tuple now, e.g. ("mean_pool",)*4+("linear","linear"),
  see ModelConfig's docstring -- not used in this config on purpose, plain mean_pool everywhere).

  chunk_batch_size=2 for TRAINING (train.py's per-step minibatch sampling) -- deliberately
  conservative, NOT the quran_uthmani_chunked example's cbs=16: training's chunk_batch_size
  multiplies row counts at EVERY deep level via _train_recurse (unlike compress/decompress's
  chunk_batch_size, which only batches one timestep's children at a time and is safe to set much
  higher -- already proven on real TPU hardware this session). At this config's scale,
  chunk_batch_size=16 would push the deepest level to ~168M rows/step -- comparable to the exact
  magnitude that triggered the old config's unresolved scan-fusion crash. chunk_batch_size=1 (10.5M
  rows at the deepest level, one whole chunk) is the safest possible starting point; =2 is a small
  step up. Raise cautiously, watching for the same compiler crash signature (RET_CHECK /
  windowing_util.cc / SIGABRT, not an OOM) before trusting a larger value.

  remat_depth="0.5" kept enabled (the crash-mitigation fallback from the original enwik9 attempt)
  as a precaution -- the shallower depth (6 vs 8 levels) and much smaller per-chunk row counts
  (10.5M vs the old config's whole-file 125M+ at the deepest level, even at chunk_batch_size=1)
  may avoid the scan-fusion crash outright, but that failure was never fully root-caused, so this
  keeps the known mitigation in place rather than assuming it's unnecessary.

UNTESTED AT THIS SCALE: unlike the file-chunking/chunk-batching FEATURES themselves (both
validated end-to-end on real TPU hardware this session, at quran_uthmani scale), this specific
config has not yet been run -- verify on TPU (ideally starting with just a few chunks / few
epochs) before committing to a full 20-epoch, 96-chunk run.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac.train --config enfrac/configs/enwik9_chunked.py --log_dir logs/enfrac/enwik9_chunked
  uv run python -m enfrac.compress --ckpt logs/enfrac/enwik9_chunked --input datasets/enwik9 --output logs/enfrac/enwik9_chunked_compressed --chunk_batch_size 16
  uv run python -m enfrac.decompress --bundle logs/enfrac/enwik9_chunked_compressed --output /tmp/enwik9_chunked_out --verify datasets/enwik9
"""

model = dict(
    patch_len_list=(32768, 4096, 512, 64, 8, 1),
    d_model_list=(1216,) * 6,
    n_layers_list=(8,) * 6,
    n_heads_list=(8,) * 6,
    mlp_mult_list=(4,) * 6,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    use_hira=True,
    hira_r=165,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac/enwik9_chunked",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    file_chunk_bytes=10_485_760,
    chunk_batch_size=1,   # bumped down from 2 -- this config OOM'd on real TPU at
                          # chunk_batch_size=2 (824.05G requested vs 30.75G HBM); see this file's
                          # docstring for the chunk_batch_size tradeoff this confirms in practice.
    micro_batch=1024,     # bumped down from the 8192 default -- see chunk_batch_size's comment;
                          # enfrac_zero's identical-shape config still OOM'd at chunk_batch_size=1
                          # (226.11G, only ~31% less than cbs=2's 328G -- not the expected ~50%,
                          # showing chunk_batch_size isn't the dominant cost; remat_depth is INERT
                          # for levels 1+ in the current _train_recurse, so micro_batch is the
                          # remaining real lever -- testing whether it actually bounds peak HBM.
    seed=0,
)
