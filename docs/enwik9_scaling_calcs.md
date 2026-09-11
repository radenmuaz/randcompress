# enwik9 scaling calculations (raw outputs, for reference)

Working notes from sizing/feasibility discussion for `enfrac`/`enfrac_zero` at enwik9 scale.
All numbers computed directly (scripts run against real param counts via `jax.eval_shape` or
plain arithmetic against the actual configs), not estimated by hand.

## Model param sizing (enwik9 configs, both packages)

Sized via `jax.eval_shape` on the real `ModelConfig`/`ByteFractalGen` construction.

| package | d_model | n_layers | n_heads | mlp_mult | hira_r | total params | total (fp32) | trainable params | trainable (fp32) |
|---|---|---|---|---|---|---|---|---|---|
| enfrac (HiRA) | 1216 | 8 | 8 | 4 | 165 | 244,420,288 | 977.7 MB | 23,940,288 | 95.8 MB |
| enfrac_zero | 448 | 7 | 8 | 4 | n/a | 24,988,672 | 100.0 MB | 24,923,136 | 99.7 MB |

`patch_len_list=(2097152, 262144, 32768, 4096, 512, 64, 8, 1)`, 8 levels, `n_timesteps=477` for a
1,000,000,000-byte file. `seq_lens=(0,8,8,8,8,8,8,8)`.

## Per-level-0-patch cumulative bytes

1 patch = 2,097,152 raw bytes = 2.097 MB. Cumulative every 128 patches (of 477 total):

| patches | cumulative bytes | % of file |
|---|---|---|
| 128 | 268,435,456 | 26.8% |
| 256 | 536,870,912 | 53.7% |
| 384 | 805,306,368 | 80.5% |
| 477 (end) | 1,000,341,504 | 100.0% |

## Per-level memory table (chunked/fixed vs the bugs found along the way)

Computed for the real enwik9 `patch_len_list`, both configs. `rows(Bcur)` = total patch count at
that level across the whole file (`timesteps_per_level`). `KV/row/layer` = `2*n_heads*head_dim*4`
bytes. "unchunked" = naive one-shot computation (what crashed); "chunked (actual)" = current fix
(`micro_batch=8192` row-chunking via `jax.lax.scan`).

### enfrac (HiRA): d_model=1216, n_layers=8, n_heads=8, head_dim=152

| lvl | patch_len | rows (Bcur) | attn_seq | KV/row/layer | KV full (unchunked) | KV chunked (8192) | mean_pool unchunked | mean_pool chunked (final fix) |
|---|---|---|---|---|---|---|---|---|
| 0 | 2,097,152 | 477 | 477 | 9,728 B | 37.1 MB | 37.1 MB | 1024.35 GB | ~2.1 GB (target) |
| 1 | 262,144 | 3,816 | 9 | 700,416 B | 2.67 GB | 2.67 GB | 1024.35 GB | ~2.1 GB |
| 2 | 32,768 | 30,528 | 9 | 700,416 B | 21.38 GB | 5.74 GB | 1024.35 GB | ~2.1 GB |
| 3 | 4,096 | 244,224 | 9 | 700,416 B | 171.06 GB | 5.74 GB | 1024.35 GB | ~2.1 GB |
| 4 | 512 | 1,953,792 | 9 | 700,416 B | 1368.47 GB | 5.74 GB | 1024.35 GB | ~2.1 GB |
| 5 | 64 | 15,630,336 | 9 | 700,416 B | 10947.74 GB | 5.74 GB | 1024.35 GB | ~2.1 GB |
| 6 | 8 | 125,042,688 | 9 | 700,416 B | 87581.90 GB | 5.74 GB | 1024.35 GB | ~2.1 GB |
| 7 (terminal) | 1 | 1,000,341,504 | 8 | 622,592 B | 622804.62 GB | 5.10 GB | n/a | n/a |

### enfrac_zero: d_model=448, n_layers=7, n_heads=8, head_dim=56

| lvl | patch_len | rows (Bcur) | attn_seq | KV/row/layer | KV full (unchunked) | KV chunked (8192) |
|---|---|---|---|---|---|---|
| 0 | 2,097,152 | 477 | 477 | 3,584 B | 12.0 MB | 12.0 MB |
| 1 | 262,144 | 3,816 | 9 | 225,792 B | 861.6 MB | 861.6 MB |
| 2 | 32,768 | 30,528 | 9 | 225,792 B | 6.89 GB | 1.85 GB |
| 3 | 4,096 | 244,224 | 9 | 225,792 B | 55.14 GB | 1.85 GB |
| 4 | 512 | 1,953,792 | 9 | 225,792 B | 441.15 GB | 1.85 GB |
| 5 | 64 | 15,630,336 | 9 | 225,792 B | 3529.20 GB | 1.85 GB |
| 6 | 8 | 125,042,688 | 9 | 225,792 B | 28233.64 GB | 1.85 GB |
| 7 (terminal) | 1 | 1,000,341,504 | 8 | 200,704 B | 200772.54 GB | 1.64 GB |

**Note on the "chunked" columns**: these are the INTENDED bound if `jax.lax.scan` nesting executed
independently per level. In practice XLA fused several consecutive self-similar levels (4/5/6,
all `B=65536` in the steady state) into one batched custom-call, producing real crashes far above
this table's numbers (`f32[4,8,8,8,8,8192,8,4864]`, ~20.9TB) — see "Fourth crash" below. The
mean-pool "unchunked" column is CONSTANT across levels because `rows[l] * patch_len[l]` always
equals the total leaf byte count (1,000,341,504) by construction.

## Level-0 seqlen (n_timesteps) scaling table

`n_timesteps = ceil(1,000,000,000 / patch_len_list[0])`. Memory computed for enfrac (d=1216,
n_layers=8, n_heads=8, mlp_mult=4), level-0 forward only (no remat), fp32.

| level-0 patch_len | n_timesteps | KV cache | attn scores | forward total |
|---|---|---|---|---|
| 2²¹ = 2,097,152 (current) | 477 | 37.1 MB | 58.2 MB | 0.21 GB |
| 2²⁰ = 1,048,576 | 954 | 74.2 MB | 233.0 MB | 0.53 GB |
| 2¹⁹ = 524,288 | 1,908 | 148.5 MB | 932.0 MB | 1.53 GB |
| 2¹⁸ = 262,144 (~4096 target) | 3,815 | 296.9 MB | 3.73 GB | 4.91 GB |
| 2¹⁷ = 131,072 (~8192 target) | 7,630 | 593.8 MB | 14.90 GB | 17.28 GB |

**Verdict**: up to ~3,815 timesteps (2¹⁸ patch_len) is comfortably cheap either way (<5GB). At
~7,630 timesteps (2¹⁷), attn_scores (14.9GB) starts eating real HBM budget — first row in this
range that actually warrants remat_time/remat_depth to stay safe, though still far below the
multi-TB levels-1+ blowups that were the actual problem this session.

## Shallow-but-long config proposals (levels 1+, keeping patch_len[0]=2,097,152)

Since `product(seq_lens[1:]) = patch_len_list[0] = 2,097,152 = 2²¹`, valid power-of-2
factorizations using the 512–8192 range (2⁹–2¹³):

| # | levels | seq_lens | patch_len_list | notes |
|---|---|---|---|---|
| A | 3 | 4096, 64, 8 | `(2097152, 512, 8, 1)` | closest to current 8-level/seq=8 design, just 3 levels shallower; bumps only level 1 |
| B | 2 | 4096, 512 | `(2097152, 512, 1)` | shallowest 2-transition option; larger seqlen at the low-row-count level (minimizes total attention pairs) |
| C | 2 | 2048, 1024 | `(2097152, 1024, 1)` | more balanced 2-level split than B |
| D | 3 | 8192, 16, 16 | `(2097152, 256, 16, 1)` | uses 8192 at level 1 (only 477 rows there), rest stays tiny/cheap |

Design rule: put the **larger** seqlen at the level with **fewer rows** (shallow, e.g. level 1
with only 477 parents) and smaller seqlen deeper (where row count is already large) — minimizes
total `rows * seqlen^2` attention pairs for a fixed total product.

**Caveat**: seqlen ≥ 512 reintroduces a real per-chunk attention-score blowup with the *naive*
`sdpa` (score matrix `[chunk_size, n_heads, seqlen, seqlen]`) unless flash attention (no full
score-matrix materialization) is used at those levels — this is why flash attention was wired up
next (see model.py's `flash_attn_causal` / `ModelConfig.use_flash_attn_list`).

## Level-0 alternate seqlen targets → levels-1+ factorization

| n_timesteps target | patch_len[0] | seq_lens (levels 1+) | patch_len_list |
|---|---|---|---|
| 954 (2²⁰) | 1,048,576 | 2048, 512 | `(1048576, 512, 1)` |
| 954 (2²⁰) | 1,048,576 | 1024, 1024 | `(1048576, 1024, 1)` |
| 1,908 (2¹⁹) | 524,288 | 1024, 512 | `(524288, 512, 1)` |

## The four distinct bugs found/fixed this session (training-time memory, enwik9 scale)

1. **Levels 1+ batch dimension (`Bcur`) reaches total raw byte count** at the terminal level — a
   plain `cur_bytes.reshape(-1, seq_len, child_len)` on a 125,042,688-row array requested 64GB
   against 34GB HBM (TPU pads the size-8 minor dim to a 128-tile, 16x blowup on ~4GB raw data).
   Fixed: `_train_recurse`, a `jax.lax.scan`-chunked recursive walk (`micro_batch=8192` default).
2. **`jax.checkpoint` + nested `jax.lax.scan` crashes the XLA compiler**: `windowing_util.cc:
   VerifyCanonicalBounds` / `RET_CHECK ... CouldLeS32` / SIGABRT. Confirmed via A/B (disabling
   checkpoint made it disappear). Fixed: levels 1+ use plain `Trunk.__call__`, no remat there
   (not needed — `seq_len` was only 8, already memory-bounded by the row-chunking above).
3. **`embed_patch`'s mean-pool materializes `[rows, P, byte_embed_dim]` before pooling** — bites
   at BOTH large `P` (level 0's own 2,097,152) and large row counts flowing in from
   `_train_recurse`'s scan body (up to `micro_batch*seq_len`=65,536). Level 2's own P=32,768
   combined with 65,536 rows would need ~550GB. Fixed: `_chunked_mean_pool`, chunks over the
   flattened ROW axis with an adaptive `n_chunk = _MEAN_POOL_TARGET_ELEMENTS // P` (~2.1M target
   elements/step, independent of level).
4. **XLA fuses multiple nested `jax.lax.scan` levels into one giant batched custom-call** instead
   of executing them as bounded sequential loops — found AFTER fixes 1-3 were deployed and
   verified locally. Crash shapes: `f32[4,8,8,8,8,8192,8,4864]` (enfrac, ~20.9TB),
   `f32[4,8,8,8,8,8192,8,1792]` (enfrac_zero, ~7.7TB) — trailing dim matches each model's MLP
   hidden size (`mlp_mult*d_model`). Likely trigger: levels 4/5/6 share an IDENTICAL shape in the
   micro_batch=8192 steady state (`B=65536` → chunk into 8×8192 → recurse with 65536 again), which
   XLA's compiler may be vectorizing across. **Status at time of writing: unresolved** — jax was
   upgraded 0.6.2→0.11.1 (new libtpu 0.0.17→0.0.46.1) as the first fix attempt (also relevant to
   bug #2, which was on the OLD jax version); re-test pending.

## Flash attention (wired, opt-in per level)

Added `flash_attn_causal()` in `enfrac/model.py` (TPU-native, via
`jax.experimental.pallas.ops.tpu.flash_attention`) and threaded `use_flash: bool` through
`Attn`/`Block`/`Trunk.__call__`/`Trunk.forward_remat` in both `enfrac/model.py` and
`enfrac_zero/model.py`. Controlled per-level by `ModelConfig.use_flash_attn_list` (tuple of bool,
length `n_levels`; `None` -> all `False`, fully backward compatible -- verified byte-identical to
the pre-change code with the default). TRAINING-forward only (`__call__`/`_train_recurse`) --
decode (`_gen_children`/`generate()`) is untouched, which is fine since `__call__` has no
bit-exactness requirement against `generate()` (see module docstring).

**Hard constraint**: the pallas kernel requires `block_k` to be a multiple of 128, so only levels
with `seq_len >= 128` (ideally a clean multiple of 128) should set this flag -- default levels
1+'s `seq_len=8` isn't supported at all. This is exactly the shallow-but-long config proposals
above (seq_len 512-8192).

**Not locally testable**: the pallas TPU kernel raises `Only interpret mode is supported on CPU
backend` when traced on CPU -- no way to verify its actual numerics without a real TPU host,
unlike every other change made this session (all of which were verified locally first). Test on
TPU before trusting a config that enables `use_flash_attn_list`.

## JAX/environment state

- Upgraded `jax`/`jaxlib` 0.6.2 → 0.11.1 (pyproject.toml floor bumped, `requires-python` bumped
  3.10→3.12, `.python-version` pinned to 3.12, `uv.lock` regenerated).
- `libtpu` on both TPU hosts: 0.0.17 → 0.0.46.1 (reinstalled via
  `uv pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`
  after each `uv sync`, since the TPU extra isn't in `pyproject.toml`'s base dependency list).
- Real bug hit along the way: training silently fell back to CPU on both TPU hosts because
  `jax[tpu]` was never installed there for TRAINING (only compress/decompress had used it before,
  and those default to `--device cpu` on purpose) — only a warning printed
  (`jax._src.xla_bridge: ... Falling back to cpu`), not an error.
