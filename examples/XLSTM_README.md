# xLSTM: Pure JAX Implementation

A comprehensive, educational implementation of **xLSTM (Extended LSTM)** in pure JAX, based on the paper [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517).

## Overview

This implementation includes:

1. **Equations & Theory** (`xlstm_equations.md`) - Complete mathematical formulation
2. **Pure JAX Implementation** (`xlstm_jax.py`) - Core xLSTM layers and model
3. **Training Script** (`train_xlstm_simple.py`) - Full training pipeline with data loading

### Key Features

- ✅ **sLSTM** (Scalar LSTM) - 4-state system with exponential gating and numerical stabilization
- ✅ **mLSTM** (Matrix LSTM) - Covariance-based memory with attention-like operations
- ✅ **Pure JAX** - No PyTorch, TensorFlow, or FlaxNo dependencies beyond JAX
- ✅ **Fully Documented** - Equations, code comments, and architectural explanations
- ✅ **Byte-Level Tokenization** - Simple vocabulary handling for text data
- ✅ **Mixed Architecture** - Can use sLSTM, mLSTM, or both in combination

## Quick Start

### Installation

Ensure you have JAX and optax installed:

```bash
pip install jax jaxlib optax tqdm
```

### Training

```bash
python examples/train_xlstm_simple.py
```

The script will:
1. Load the Quran dataset (or use synthetic data if unavailable)
2. Initialize the xLSTM model
3. Train for 3 epochs
4. Evaluate on validation set
5. Generate text samples

### Basic Usage

```python
import jax.random as jr
from xlstm_jax import Config, init_xlstm_params, forward_pass

# Create configuration
config = Config(d_model=256, num_heads=8, num_layers=4)

# Initialize parameters
key = jr.PRNGKey(0)
params = init_xlstm_params(key, config)

# Prepare input tokens [batch_size, seq_length]
tokens = jr.randint(key, (16, 256), 0, 256)

# Forward pass
logits = forward_pass(params, tokens, config)  # Shape: [16, 256, 256]
```

## Architecture Details

### sLSTM: Scalar LSTM with Four States

The sLSTM improves upon traditional LSTMs with:

1. **Exponential Gating** - Replace sigmoid gates with exponentially normalized gates
2. **Four-State System** - Track (y, c, n, m) for numerical stability
3. **Normalization** - Automatically normalized through state design

**Key Equations:**

$$m_t = \max(i_{\text{raw}}, \log\sigma(f_{\text{raw}}) + m_{t-1})$$

$$i_t = \min(1, e^{i_{\text{raw}} - m_t})$$

$$f_t = \min(1, e^{\log\sigma(f_{\text{raw}}) + m_{t-1} - m_t})$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(z_{\text{raw}})$$

$$y_t = \sigma(o_{\text{raw}}) \odot \frac{c_t}{n_t}$$

### mLSTM: Matrix LSTM with Covariance Memory

The mLSTM extends the concept with matrix memory:

1. **Covariance Matrix** - Stores second-order moments: $C_t \in \mathbb{R}^{H \times H}$
2. **Multihead Attention** - Query-key-value projections per head
3. **Parallelizable** - Can compute in chunks for efficient training

**Key Operations:**

$$C_t = f_t \odot C_{t-1} + i_t \odot (\tilde{K}_t^T V_t)$$

$$n_t = f_t \odot n_{t-1} + i_t \odot \tilde{K}_t$$

$$h_t = \frac{Q_t^T C_t}{\max(|Q_t^T n_t|, e^{-m_t}) + \epsilon}$$

## Implementation Details

### sLSTM Forward Pass (`slstm_step`)

```python
def slstm_step(params, state, x):
    # Compute gates with exponential stabilization
    m_t = jnp.maximum(i_raw, log_f_t)
    i_t = jnp.minimum(1.0, jnp.exp(i_raw - m_t))
    f_t = jnp.minimum(1.0, jnp.exp(log_f_t - m_t))
    
    # Update states
    c_t = f_t * c_prev + i_t * jnp.tanh(z_raw)
    n_t = f_t * n_prev + i_t
    y_t = o_t * c_t / (n_t + eps)
    
    return new_state, y_t
```

**Time Complexity:** $O(T \times H)$ where $T$ is sequence length, $H$ is hidden dimension

### mLSTM Forward Pass (`mlstm_step`)

```python
def mlstm_step(params, state, x):
    # Project to query, key, value
    Q = einsum('ldi,bi->bld', params.W_q, x)
    K = einsum('ldi,bi->bld', params.W_k, x)
    V = einsum('ldi,bi->bld', params.W_v, x)
    
    # Update covariance matrix
    C_t = f_t[:,:,None,None] * C_prev + i_t[:,:,None,None] * einsum('bld,blh->blhd', K_scaled, V)
    
    # Compute output
    h = einsum('bld,blhd->blh', Q, C_t) / (denom + eps)
    
    return new_state, h
```

**Time Complexity:** $O(T \times L \times D_h^2)$ in recurrent form, $O(T^2 \times L \times D_h)$ in parallel form

### Numerical Stability

The implementation uses several techniques to maintain numerical stability:

1. **Log-space Computations** - Work in log space for forget gates
2. **Maximum Tracking** - Subtract maximum before exponential (LogSumExp trick)
3. **Capped Exponentials** - $\min(1, e^x)$ prevents overflow
4. **Layer Normalization** - Applied after each LSTM step
5. **Gradient Clipping** - Norm-based clipping during training

## Configuration

The `Config` class controls model architecture:

```python
class Config(NamedTuple):
    # Architecture
    d_model: int = 256          # Hidden dimension
    num_heads: int = 8          # Number of attention heads
    d_head: int = 32            # Per-head dimension
    d_ff: int = 1024            # Feed-forward hidden size
    num_layers: int = 4         # Number of xLSTM blocks
    max_seq_len: int = 512      # Maximum sequence length
    
    # Model components
    use_slstm: bool = True      # Include sLSTM layer
    use_mlstm: bool = True      # Include mLSTM layer
    soft_cap: float = 15.0      # Gate saturation control
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
```

## File Structure

```
examples/
├── xlstm_equations.md          # Mathematical formulation (MUST READ!)
├── xlstm_jax.py                # Core implementation
├── train_xlstm_simple.py       # Training script
└── README.md                   # This file
```

## Training Pipeline

### 1. Data Loading

```python
# Load text file
tokens = load_dataset("data.txt")  # Returns tokenized bytes [N]

# Create batches
for batch in create_batches(tokens, batch_size=16, seq_len=256):
    # batch shape: [B, T+1]
```

### 2. Model Initialization

```python
params = init_xlstm_params(key, config)
```

### 3. Training Loop

```python
for epoch in range(num_epochs):
    for batch in create_batches(...):
        params, optimizer_state, loss = train_step(
            params, optimizer_state, batch, config, optimizer
        )
```

### 4. Inference / Generation

```python
seed_tokens = ByteTokenizer.encode("The quick brown")
generated = generate(params, seed_tokens, max_length=100, config=config)
text = ByteTokenizer.decode(generated)
```

## Performance Characteristics

| Aspect | sLSTM | mLSTM | xLSTM Block |
|--------|-------|-------|------------|
| **Time/step** | $O(H)$ | $O(L \times D_h^2)$ | $O(L \times D_h^2 + FFN)$ |
| **Memory/step** | $O(H)$ | $O(L \times D_h^2)$ | $O(L \times D_h^2)$ |
| **Parallelizable** | Partially | Yes (parallel form) | Yes (mLSTM part) |
| **Gradient flow** | Excellent | Good | Excellent |

## Hyperparameter Recommendations

- **Small model**: `d_model=128, num_heads=4, num_layers=2`
- **Medium model**: `d_model=256, num_heads=8, num_layers=4` (default)
- **Large model**: `d_model=512, num_heads=16, num_layers=8`

**Learning rate schedule:**
- Warmup for first 1000 steps to `learning_rate`
- Cosine annealing decay to 0.1x `learning_rate` over remaining steps

**Regularization:**
- `dropout_rate`: 0.1 for small models, 0.2 for large
- `weight_decay`: 1e-6 (small models) to 1e-5 (large models)
- `grad_clip_norm`: 1.0 (stable) or 0.5 (aggressive)

## Comparison with Transformers

| Feature | LSTM | xLSTM | Transformer |
|---------|------|-------|-------------|
| **Recurrent** | Yes | Yes | No |
| **Attention** | Implicit | Via mLSTM | Explicit (O(T²)) |
| **Parallelizable** | Limited | mLSTM branch | Full |
| **Memory** | O(H) per step | O(H²) per step | O(T×H) full seq |
| **Inductive Bias** | Strong | Strong | Weak |
| **Long Range** | Difficult | Better | Excellent |
| **Scaling** | Better for memory | Good | Better for speed |

## References

1. **xLSTM Paper**: Beck et al., 2024. "xLSTM: Extended Long Short-Term Memory". arXiv:2405.04517
2. **Official Repository**: https://github.com/NX-AI/xlstm
3. **Original LSTM**: Hochreiter & Schmidhuber, 1997. "Long Short-Term Memory". Neural Computation
4. **Attention**: Vaswani et al., 2017. "Attention is All You Need". NeurIPS
5. **JAX**: Bradbury et al., 2018. "JAX: composable transformations of Python+NumPy programs". 

## Notes for Implementation

### Pure JAX Advantages
- ✅ Functional, immutable design
- ✅ Automatic differentiation (JAX autodiff)
- ✅ JIT compilation for speed
- ✅ Easy GPU/TPU support
- ✅ Clear separation of concerns

### Current Limitations
- ⚠️ Sequential processing of xLSTM blocks (not a blocker, necessary for recurrence)
- ⚠️ No distributed training (could add with jax.pmap)
- ⚠️ Simplified mLSTM (parallel form not fully implemented for efficiency)
- ⚠️ Single-device only

### Future Improvements
- [ ] Add parallel mLSTM computation for full efficiency
- [ ] Implement adaptive soft-capping
- [ ] Add checkpointing for memory-efficient training
- [ ] Support for longer sequences with gradient checkpointing
- [ ] Distributed training with jax.pmap

## Debugging Tips

1. **NaN values**: Check for division by zero in normalization layers
2. **Exploding gradients**: Reduce `learning_rate` or increase `grad_clip_norm`
3. **Slow training**: Use smaller `d_model` for CPU testing
4. **Memory overflow**: Reduce `batch_size` or `d_model`

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{beck2024xlstm,
    title={xLSTM: Extended Long Short-Term Memory},
    author={Beck, Maximilian and Pöppel, Korbinian and Spanring, Markus and 
            Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and 
            Klambauer, Günter and Brandstetter, Johannes},
    journal={arXiv preprint arXiv:2405.04517},
    year={2024}
}
```

## License

This implementation is provided for educational purposes.
