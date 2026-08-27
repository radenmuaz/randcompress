# MTP + Hierarchical Decode Plan

## 1. MTP — Multi-Token Prediction

### Distribution

AR factorizes:
$$p(x_1,\ldots,x_T) = \prod_{t=0}^{T-1} p(x_{t+1} \mid x_{\leq t})$$

MTP-k groups tokens into blocks of size k. Block b covers positions bk+1…bk+k.
The model at block-start (after seeing x_{≤bk}) outputs k distributions:

$$\tilde{p}_j(x_{bk+j+1} \mid x_{\leq bk}), \quad j = 0,\ldots,k-1$$

MTP joint:
$$\tilde{p}(x_1,\ldots,x_T) = \prod_{b=0}^{T/k-1} \prod_{j=0}^{k-1} \tilde{p}_j(x_{bk+j+1} \mid x_{\leq bk})$$

**Valid for range coding**: at the moment of coding position bk+j+1, its CDF depends
only on x_1,…,x_{bk} — all of which were coded in earlier blocks. ✓

**What head j learns**: CE training with target shifted by j+1 drives head j toward
the true marginal p(x_{t+j+1} | x_{≤t}) = ∑_{x_{t+1}..x_{t+j}} p(x_{t+1..t+j+1} | x_{≤t}).

**Overhead vs AR**: per block b, positions j=1..k-1 each pay
I(x_{bk+1:bk+j} ; x_{bk+j+1} | x_{≤bk}) extra bits — the mutual information that
within-block predecessors carry about the target, which MTP ignores.
j=0 is identical to AR (no overhead).

**k=1 degenerates to AR**: step_mtp = step, no scan_states needed, joint = AR joint. ✓

### State bookkeeping (encoder and decoder identical)

Block b, current token x_{bk}:

```
step_mtp(x_{bk}, state_{bk})  →  state_{bk+1},  logits[0..k-1]
   logits[j]  = CDF for x_{bk+j+1}
encode/decode x_{bk+1} using logits[0]
encode/decode x_{bk+2} using logits[1]
...
encode/decode x_{bk+k} using logits[k-1]
scan_states([x_{bk+1}..x_{bk+k-1}], state_{bk+1})  →  state_{bk+k}
next input: x_{bk+k}
```

Decoder has x_1..x_{bk} available (decoded in prior blocks) → same state_{bk} → same CDFs. ✓

### File-by-file changes

#### `randcompress/config.py`
- Add `mtp_k: int = 1` to `ModelConfig`

#### `randcompress/models/msrnn.py`
Refactor into shared helpers, then add MTP methods:

- `_layer_scan(frozen, adapters, x, states)` — extracted from `forward`
- `_layer_step(frozen, adapters, x, state, t)` — extracted from `step`
- `_layer_scan_eff(ew, x, states)` — extracted from `step_eff` scan path
- `_layer_step_eff(ew, x, state, t)` — extracted from `step_eff`

New public methods:
- `step_mtp(frozen, adapters, token, state, t)` → `([B, k, oh, ov], new_state)` — 1 RNN step + k projections
- `scan_states(frozen, adapters, tokens, states, t_start)` → `new_states` — n RNN steps, no output
- `step_mtp_eff(ew, token, state, t)` — precomputed-weight version
- `scan_states_eff(ew, tokens, states, t_start)` — precomputed-weight version

Updated:
- `init_adapters` — add `output_proj_1..output_proj_{k-1}` (direct params, same as output_proj)
- `forward` — use `_layer_scan`, return `[B, S, k, oh, ov]`
- `step` — calls `step_mtp`, returns head 0: `[B, oh, ov]`
- `step_eff` — calls `step_mtp_eff`, returns head 0
- `precompute_eff_weights` — include all k output_proj tensors

#### `randcompress/train.py`
- `make_chunks(raw_bytes, tcfg, mtp_k=1)` — targets shape `[NC, S, k, oh]`;
  `targets[c, s, j, h] = padded[c*S + s + j + 1 + h_offset]`;
  pad_len extended by `mtp_k - 1` extra tokens
- `compute_loss_and_grads` — `loss = mean over j in 0..k-1, h in 0..oh-1 of CE(logits[:,:,j,h,:], tgt[:,:,j,h])`
- `eval_segments_stateful` — use `logits[0, :, 0, :, :].argmax` (head 0 only for accuracy)

#### `randcompress/compress.py`
Replace inner `for pos in range(chunk_size)` loop:

```python
all_tokens = all_inputs.ravel()   # [T_total]
for t in range(0, T_total, k):
    tok = tensor([all_tokens[t]])
    logits_k, states = step_mtp_fn(tok, states, t)        # [1, k, oh, V]
    for j in range(min(k, T_total - t)):
        logits_list.append(logits_k[0, j].cpu().numpy())  # [oh, V]
    if k > 1:
        adv = tensor([all_tokens[t+1 : min(t+k, T_total)]])
        if adv.numel() > 0:
            states = scan_fn(adv, states, t + 1)
    pbar.update(min(k, T_total - t))
```

Rest of encode (quantize CDFs, RC encode, verify, bundle) unchanged.

#### `randcompress/decompress.py`
Replace `for t in range(T_valid)` loop:

```python
for t in range(0, T_valid, k):
    logits_k, states = model.step_mtp(frozen, adapters, cur_tok, states, t)
    block_toks = []
    for j in range(min(k, T_valid - t)):
        syms = decode_one_position(logits_k[0, j], ...)  # per head
        block_toks.append(sym_head0)
    if len(block_toks) > 1:
        adv = tensor([block_toks[:-1]])
        states = model.scan_states(frozen, adapters, adv, states, t + 1)
    cur_tok = tensor([block_toks[-1]])
```

---

## 2. Hierarchical Decode — Parallel Chunks

### Key insight

Context passes to children through **weights**, not state. Every chunk decodes from **zero state**.
This breaks the sequential dependency between chunks → full parallel decode.

### Training tree (depth d, width n)

```
depth 0 — root:     1 model,  full file,  1 epoch (not overfitting)
depth 1 — branches: n clones, each 1/n of file, from zero state, overfit
depth 2 — leaves:   n² clones, each 1/n² of file, from zero state, overfit
...
depth d — leaves:   n^d clones
```

Each child inherits parent's adapter weights as init, then finetunes on its chunk.
Root and intermediates discarded. Bundle contains only n^d leaf param sets.

### Why smaller models compensate

Each leaf overfits 1/n^d of the file. A model with P/n^d params may suffice.
Total bundle: n^d × (P/n^d) = P — same as monolithic. Decode latency: T/n^d
(all leaves in parallel). Cold-start warm-up ∝ d_model, shrinks with smaller model.

### Config additions
```python
tree_depth: int = 1   # 1 = flat (current); d > 1 = hierarchical
tree_width: int = 1   # branching factor; n^d leaves total
```

### Bundle format
```
bundle/
  config.json        model config (shared frozen seed)
  meta.json          {tree_depth, tree_width, chunk_size, ...}
  leaves/
    0/ params.pkl    bytes [0, S)
    1/ params.pkl    bytes [S, 2S)
    ...
  rc_stream.bin      per-leaf streams concatenated (or separate files)
```

### Decode
- Load all n^d leaf param dicts
- Spawn one process per leaf, each: `init_frozen(seed)` → zero state → AR decode its chunk
- Concatenate results in order

### Composition with MTP
MTP reduces iterations within each leaf chunk (T/n^d → T/n^d/k). Hierarchical
reduces sequential dependency across chunks. Both enabled simultaneously.

---

## Implementation order

1. MTP (this PR): config + msrnn + train + compress + decompress
2. Hierarchical (next): new train_hierarchical.py + decompress_hierarchical.py + bundle format
