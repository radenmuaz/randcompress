"""enwik9 (1,000,000,000 B), FILE-CHUNKED, "D2" candidate: SHALLOWER (4 levels, down from
enwik9_chunked.py's 6) at the SAME 10 MiB file chunk and d_model/n_layers, to cut scan-nesting
depth further -- see the shallow-but-wide candidate discussion this session. Real motivating
data: extensive TPU testing on the 6-level config never got the per-step HLO temp-memory
requirement below ~140.7G (best combo found: flat-scan `_train_flat` + per-level
`jax.checkpoint` + remat_depth=0.5), still ~4.6x over the 30.75G v4-8 HBM budget, while
`micro_batch` (chunk-size knob) was shown to have almost NO effect (8192->1024->128 changed
peak HBM by <7% total) -- strong evidence the cost is driven by the STATIC nested-scan/level
STRUCTURE itself, not by how much data flows through each step. Fewer levels = fewer per-level
scan transitions = smaller HLO, is the untested next lever.

patch_len_list=(32768,1024,32,1) -- ratio 32 per level (32768=32^3), chosen so
chunk_n_timesteps stays at 320 (same order of magnitude as enwik9_chunked.py's 320/the original
whole-file config's 477 -- proven-tractable decode-step count; decode wall-clock scales directly
with this number, not chunk_batch_size, see decompress.py's docstring).

Sizing (real jax.eval_shape, not estimated): d_model=448 (UNCHANGED from enwik9_chunked.py --
computed that at only 4 levels this already lands at 94.9MB, almost the entire ~100MB target on
its own; widening further overshoots fast, e.g. d_model=640 -> 192MB, roughly quadratic since
the shared trunk dominates the budget -- so there isn't meaningful room to go WIDER while
staying in budget, unlike what "shallow but wide" originally hoped; the real win here is fewer
scan-nesting levels, not a wider trunk) -> total 23,727,104 params (94.9MB fp32).

Requires model.py's _train_flat / FLAT_SCAN=1 env var (and LEVEL_CKPT=1 for the best-found
combo) -- see enwik9_chunked.py and CLAUDE.md for the full flat-scan/level-checkpoint rationale.
UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  FLAT_SCAN=1 LEVEL_CKPT=1 uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_chunked_d2.py --log_dir logs/enfrac_zero/enwik9_chunked_d2
"""

model = dict(
    patch_len_list=(32768, 1024, 32, 1),
    d_model_list=(448,) * 4,
    n_layers_list=(7,) * 4,
    n_heads_list=(8,) * 4,
    mlp_mult_list=(4,) * 4,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9_chunked_d2",
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
