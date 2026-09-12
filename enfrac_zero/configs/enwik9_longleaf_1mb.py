"""enwik9 (1,000,000,000 B), DELIBERATE long-context-on-LEAVES / shallow-network design --
inverted from every prior enfrac_zero config in this repo, which used SMALL local seq_len at
every level 1+ (typically 4 or 8) and put the "long context" only at level 0.

patch_len_list=(4096, 2048, 1024, 1) -- only 4 levels (SHALLOW: fewer recursion depths than the
6-level configs elsewhere), with seq_lens = (0, 2, 2, 1024): level 1/2 have a tiny local seq_len
of 2 each, but the TERMINAL (leaf, byte-level) attention at level 3 has local seq_len=1024 -- a
genuinely long local context at the byte-prediction leaf, unlike prior designs' seq_len=4-8
leaves. This tests whether giving the leaf level itself a long attention window (instead of only
level 0) changes what the model can capture.

use_flash_attn_list=(False,False,False,True): flash attention ONLY at the terminal level, whose
seq_len=1024 is well past enfrac.model.flash_attn_causal's usual >=128 benefit threshold; levels
0-2 have short sequences (level0's n_timesteps depends on file_chunk_bytes, see below; levels 1-2
are fixed at seq_len=2) where flash gives no benefit and isn't worth the extra kernel complexity.

Sized via real ByteFractalGen construction (not eval_shape): d_model=640, n_layers=3 (SHALLOW),
n_heads=8, mlp_mult=4, share_trunk=True -- total=21,779,456 params (87.1MB), under the 100MB
budget with margin. n_layers=3 chosen specifically to keep the network shallow while d_model=640
gives it width instead -- the opposite depth/width tradeoff from the deeper-narrower 6-level/7-
layer configs used elsewhere.

1 MiB (2^20) file chunks -- n_file_chunks = ceil(1,000,000,000 / 1,048,576) = 954,
chunk_n_timesteps (level 0) = 256. Sibling configs: enwik9_longleaf_512kb.py (chunk_n_timesteps=
128), enwik9_longleaf_128kb.py (32), enwik9_longleaf_64kb.py (16) -- same model, only
file_chunk_bytes/chunk_batch_size differ.

micro_batch=32 (down from the usual default 1024): at chunk_n_timesteps=256, one file chunk
alone produces Bcur=1024 rows entering level 3's own seq_len=1024 local attention (256 * seq_len[1]=2
* seq_len[2]=2). Measured on real TPU hardware, successive halvings: 1024->86.89G, 256->33.11G,
128->28.18G, 64->25.87G (only 25.30G free -- just 0.57G short, diminishing returns per halving,
consistent with a fixed level-0-driven memory floor that micro_batch chunking (which only affects
levels 1+) can't shrink further). micro_batch=32 is the next halving; verify on real hardware.

UNTESTED AT THIS SCALE.

  uv run python -m enfrac.download_data --which enwik9 --out_dir datasets
  uv run python -m enfrac_zero.train --config enfrac_zero/configs/enwik9_longleaf_1mb.py --log_dir logs/enfrac_zero/enwik9_longleaf_1mb
"""

model = dict(
    patch_len_list=(4096, 2048, 1024, 1),
    d_model_list=(640,) * 4,
    n_layers_list=(3,) * 4,
    n_heads_list=(8,) * 4,
    mlp_mult_list=(4,) * 4,
    byte_embed_dim=256,
    patch_in_scheme="mean_pool",
    share_trunk=True,
    use_flash_attn_list=(False, False, False, True),
    seed=0,
)

train = dict(
    dataset="datasets/enwik9",
    log_dir="logs/enfrac_zero/enwik9_longleaf_1mb",
    lr=3e-3,
    grad_clip=1.0,
    remat_time="1.0",
    remat_depth="0.5",
    n_epochs=60,
    file_chunk_bytes=1_048_576,
    chunk_batch_size=4,
    micro_batch=32,
    seed=0,
)
