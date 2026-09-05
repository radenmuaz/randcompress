"""Full quran-uthmani.txt (~1.36MB) config -- deliberately small/fast model (sanity-scale, like
overfitter/README.md's "sanity checks" section, not its longer "real run" one): proves the
pipeline handles the full file end to end within a reasonable session, not a tuned compression
ratio (it happens to actually compress at this scale -- see logs/enfrac_baseline/quran_uthmani_compressed/meta.json
from the last run -- but that wasn't optimized for). share_trunk=True keeps param count down and
keeps model.py construction (one Trunk regardless of n_levels) cheap.

  uv run python -m enfrac_baseline.train --config enfrac_baseline/configs/quran_uthmani.py --log_dir logs/enfrac_baseline/quran_uthmani

See enfrac/configs/quran_uthmani.py for the HiRA counterpart. `seed` spelled out explicitly (see
fatihah.py's docstring for why) -- same value as ModelConfig's default.

compress.py/decompress.py's per-byte python recursion (generate()/collect_logits(), see
enfrac/model.py's determinism-contract docstring for why it can't be batched away -- the same
bit-exactness discipline applies here even though this model has no frozen base to reconstruct)
means wall time scales with n_raw_bytes / batch_size regardless of patch_len_list --
batch_size=128 here to keep that under a few minutes on CPU.
"""

model = dict(
    patch_len_list=(64, 16, 4, 1),
    d_model_list=(32, 32, 32),
    n_layers_list=(2, 2, 2),
    n_heads_list=(4, 4, 4),
    mlp_mult_list=(2, 2, 2),
    byte_embed_dim=64,
    share_trunk=True,
    seed=0,
)

train = dict(
    dataset="quran_data/quran-uthmani.txt",
    steps=6000,
    lr=3e-3,
    warmup_steps=200,
    log_every="1000",
)

compress = dict(batch_size=128)
