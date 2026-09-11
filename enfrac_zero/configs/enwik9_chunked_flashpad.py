"""enwik9 (1,000,000,000 B), FILE-CHUNKED -- FLASH-ATTENTION test variant of enwik9_chunked.py.
Same 6-level shapes/sizing as enwik9_chunked.py, but file_chunk_bytes bumped from 10,485,760 to
12,582,912 (384 * patch_len_list[0]=32768, a multiple of 128) SPECIFICALLY so chunk_n_timesteps
lands on 384 (flash_attention's pallas TPU kernel requires kv_seq_len divisible by
block_k_major=128 -- verified on real TPU this session: chunk_n_timesteps=320 raised
`ValueError: kv_seq_len=320 should be divisible by block_k_major=128`). No new masking/padding
logic needed: since a 12 MiB chunk is carved from real consecutive file bytes, every chunk except
the (already-handled, zero-padded-with-warning) last one is filled with REAL data across all 384
timesteps -- this is just a chunk-SIZE choice, not padding-within-a-chunk.

use_flash_attn_list=(True,False,False,False,False,False) -- level 0 ONLY. Levels 1+ have
seq_len=8, far below flash's practical minimum (block_k must be >=128, see flash_attn_causal's
docstring) -- not fixable by padding chunk size, that's an inherent per-level architectural
constant (seq_len = patch_len_list[l-1] // patch_len_list[l], fixed by the model's ratio-8
recursion, not by file_chunk_bytes).

PURPOSE: diagnostic only, testing whether flash attention meaningfully reduces the ~140.7G HBM
requirement found with the best combo so far (flat-scan + per-level checkpoint + remat_depth=0.5,
see enwik9_chunked.py's docstring) -- prior size analysis suggested attention score matrices are
only a few MB at this scale (T=320-384, tiny relative to the ~140G problem), so this is expected
to show little effect, but worth confirming empirically rather than assuming.

  FLAT_SCAN=1 LEVEL_CKPT=1 uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_chunked_flashpad.py --log_dir logs/enfrac_zero/enwik9_chunked_flashpad --micro_batch 1024
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
    use_flash_attn_list=(True, False, False, False, False, False),
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9_chunked_flashpad",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=20,
    file_chunk_bytes=12_582_912,
    chunk_batch_size=1,
    micro_batch=1024,
    seed=0,
)
