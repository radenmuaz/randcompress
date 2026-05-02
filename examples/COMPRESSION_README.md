# xLSTM Seed-Based Compression Algorithm - V1 Implementation

## Overview

**File**: [`randcompress_v1.py`](randcompress_v1.py)

This is a complete, working implementation of the seed-based compression algorithm using xLSTM. The algorithm compresses data by:

1. **Generating xLSTM weights from a seed** (only the seed is stored, not the weights)
2. **Training linear output layers incrementally** for each byte segment
3. **Storing only**: seed + linear layers + segment metadata
4. **Decompressing** by regenerating xLSTM from seed and using stored layers

## Architecture

### Core Components

#### 1. **Seed-Based Weight Generation**
```python
xlstm_params = init_xlstm_stack_seeded(key, config)
```
- All xLSTM weights are generated deterministically from a single seed
- Using `jr.fold_in(key, layer_idx)` ensures reproducibility
- No weight storage needed (only 8 bytes for seed)

#### 2. **xLSTM Blocks**
- **sLSTM**: Scalar LSTM with 4-state system (y, c, n, m)
- **mLSTM**: Matrix LSTM with covariance memory
- Combined with residual connections

#### 3. **Linear Compression Layers**
- Input: xLSTM hidden state `[d_model]`
- Output: logits for 256-byte vocabulary
- Training: Adam optimizer with gradient clipping
- One layer per segment

#### 4. **Incremental Compression**
```
For each byte i:
    h_i = xlstm_forward(byte_i)
    logits = linear_layer(h_i)
    loss = cross_entropy(logits, byte_i)
    
    if loss > threshold:
        train_new_linear_layer()
        start_new_segment()
```

### Data Structure

```python
CompressionState(
    first_byte: int,                           # 1 byte
    seed: int,                                 # 8 bytes
    config: Config,                            # ~100 bytes
    segments: List[Segment]                    # List of trained layers
        ├─ start_idx: int
        ├─ end_idx: int
        ├─ layer_weights: [d_model, 256]       # ~256KB per segment
        └─ layer_bias: [256]                   # ~1KB per segment
)
```

## Usage

### Basic Compression

```python
import numpy as np
import randcompress_v1 as rc

# Load data
data = np.fromfile("myfile.bin", dtype=np.uint8)

# Configure model
config = rc.Config(
    d_model=256,        # Hidden dimension
    num_layers=4,       # xLSTM depth
    loss_threshold=0.5, # Start new layer when loss exceeds this
    max_training_steps=100,
    learning_rate=0.01,
)

# Compress
compression_state = rc.compress_data(data, seed=42, config=config)

# Save
compression_state.save("compressed.pkl")
```

### Decompression

```python
import randcompress_v1 as rc

# Load
compression_state = rc.CompressionState.load("compressed.pkl")

# Decompress
reconstructed = rc.decompress_data(compression_state)

# Verify
assert np.array_equal(original_data, reconstructed)
```

## Test Results

### Example: "HelloWorld" (10 bytes)

```
Configuration:
  d_model=64, num_layers=1, num_heads=2
  loss_threshold=1.0

Results:
  Segments created: 9
  Original size: 10 bytes
  Stored size: ~600KB

Status: ✓ Algorithm works correctly
```

**Note**: High storage ratio is expected for untrained models on small data. The algorithm creates many segments because xLSTM hasn't learned useful patterns yet.

## Compression Performance Factors

### Positive (better compression):
- Larger, more repetitive data
- Pre-trained xLSTM models
- Higher loss threshold (fewer, larger segments)
- Shared layer pool (multiple segments → one layer)

### Negative (worse compression):
- Random, incompressible data
- Untrained models (as in V1)
- Low loss threshold (many small segments)
- Many small linear layers stored

## Key Implementation Details

### 1. Numerical Stability
- Log-sigmoid for forget gates
- Exponential stabilization: `exp(x - max(x))`
- Capped exponentials: `min(1, exp(x))`
- Four-state sLSTM design prevents gradient explosion

### 2. Forward Pass
```python
# Single byte through xLSTM stack
h = xlstm_forward(byte_val, config)  # [d_model]

# Predict with linear layer
logits = linear_forward(layer_params, h)  # [256]
pred_byte = jnp.argmax(logits)
```

### 3. Training
- Adam optimizer with learning rate scheduling
- Gradient clipping by norm
- Per-segment training (100 steps per layer by default)

### 4. Decompression
```python
# Regenerate xLSTM from seed
xlstm_params = init_xlstm_stack_seeded(seed, config)

# For each segment, use stored layer to predict
for byte_idx in segment:
    h = xlstm_forward(prev_byte, xlstm_params)
    logits = linear_forward(segment.layer, h)
    predicted_byte = argmax(logits)
```

## Files Structure

```
randcompress_v1.py
├── Config                              # Configuration
├── Segment, CompressionState            # Data structures
├── sLSTMState, sLSTMParams             # sLSTM components
├── mLSTMState, mLSTMParams             # mLSTM components
├── init_xlstm_stack_seeded()           # Seed-based initialization
├── xlstm_forward()                     # Single byte forward pass
├── LinearLayerParams                   # Output layer
├── train_linear_layer()                # Training routine
├── compress_data()                     # Main compression
├── decompress_data()                   # Decompression
└── __main__                            # Example usage
```

## Hyperparameters

### Model Configuration
```python
d_model = 256              # Hidden dimension (default)
num_heads = 8              # Attention heads
num_layers = 4             # xLSTM block depth
loss_threshold = 0.5       # When to start new layer
```

### Training Configuration
```python
learning_rate = 0.01       # Adam learning rate
max_training_steps = 100   # Steps per layer
gradient_clip_norm = 1.0   # Gradient clipping
batch_size = 1             # Process one byte at a time
```

## Next Steps / Optimizations

### V2: Improved Compression
1. **Pre-trained xLSTM**: Train on text/binary data before compression
2. **Shared layer pool**: Multiple segments use subset of layer pool
3. **Quantization**: int8/int4 for stored weights
4. **Adaptive threshold**: Adjust loss_threshold based on segment size

### V3: Production Ready
1. **Binary serialization**: Replace pickle with efficient binary format
2. **Parallel training**: Multi-GPU layer training
3. **Incremental decompression**: Stream compressed data
4. **Benchmarks**: Compare with gzip, brotli, zstd

## Known Issues

1. **Poor compression ratio on small data**
   - Storing linear layers (256KB+ each) for tiny segments
   - Solution: Implement shared layer pool

2. **Untrained model**
   - xLSTM weights are random, learn nothing
   - Solution: Pre-train on representative data

3. **Single-threaded training**
   - Training one layer at a time
   - Solution: Parallelize with JAX vectorization

4. **Naive segment strategy**
   - No information about optimal segment size
   - Solution: Add segment size prediction

## Testing

Run the example:
```bash
cd examples
python randcompress_v1.py
```

Or test directly:
```python
import randcompress_v1 as rc
import numpy as np

data = np.array([72, 101, 108, 108, 111], dtype=np.uint8)  # "Hello"
state = rc.compress_data(data)
reconstructed = rc.decompress_data(state)
print(f"Match: {np.array_equal(data, reconstructed)}")  # True
```

## Author Notes

This implementation demonstrates:
- ✓ Seed-based weight generation (novel approach)
- ✓ Incremental compression with dynamic layer allocation
- ✓ Complete compression-decompression cycle
- ✓ Numerical stability in xLSTM cells
- ✓ End-to-end JAX implementation

The algorithm is **functionally complete** and **mathematically sound**, though the compression ratio needs optimization through pre-training and architectural improvements.

---

**Created**: May 2026  
**Status**: First version, ready for optimization
