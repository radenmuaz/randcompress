"""
xLSTM Seed-Based Compression Algorithm - First Version

Core Concepts:
- Weights generated from seed (only seed stored)
- Incremental compression: add linear layer per segment when needed
- Stores: seed + linear layers + segment metadata
- Decompression: regenerate xLSTM from seed, use stored layers to predict bytes
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import NamedTuple, List, Tuple, Optional
import optax
import pickle
from pathlib import Path
import struct
import math


# ============================================================================
# Configuration and Data Structures
# ============================================================================

class Config(NamedTuple):
    """Configuration for xLSTM compression model."""
    vocab_size: int = 256  # Byte vocabulary
    d_model: int = 256  # Hidden dimension
    num_heads: int = 8
    d_head: int = 32
    d_ff: int = 1024
    num_layers: int = 4
    max_seq_len: int = 512
    
    # Compression specific
    use_slstm: bool = True
    use_mlstm: bool = True
    soft_cap: float = 15.0
    
    # Compression hyperparameters
    loss_threshold: float = 0.5  # Start new layer when loss exceeds this
    max_training_steps: int = 100  # Training steps per layer
    learning_rate: float = 0.01
    gradient_clip_norm: float = 1.0
    batch_size: int = 1  # Process one byte at a time


class Segment(NamedTuple):
    """Single compression segment with its own linear layer."""
    start_idx: int
    end_idx: int
    layer_weights: jnp.ndarray  # [d_model, 256]
    layer_bias: jnp.ndarray     # [256]


class CompressionState(NamedTuple):
    """Complete compression state for serialization."""
    first_byte: int
    seed: int
    config: Config
    segments: List[Segment]
    
    def to_dict(self):
        """Convert to serializable dictionary."""
        return {
            'first_byte': int(self.first_byte),
            'seed': int(self.seed),
            'config': self.config,
            'segments': [
                {
                    'start_idx': int(seg.start_idx),
                    'end_idx': int(seg.end_idx),
                    'layer_weights': np.array(seg.layer_weights),
                    'layer_bias': np.array(seg.layer_bias),
                }
                for seg in self.segments
            ],
        }
    
    @staticmethod
    def from_dict(data):
        """Restore from serialized dictionary."""
        segments = [
            Segment(
                start_idx=seg['start_idx'],
                end_idx=seg['end_idx'],
                layer_weights=jnp.array(seg['layer_weights']),
                layer_bias=jnp.array(seg['layer_bias']),
            )
            for seg in data['segments']
        ]
        return CompressionState(
            first_byte=data['first_byte'],
            seed=data['seed'],
            config=data['config'],
            segments=segments,
        )
    
    def save(self, path):
        """Save compression state to file."""
        data = self.to_dict()
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved compression state to {path}")
    
    @staticmethod
    def load(path):
        """Load compression state from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return CompressionState.from_dict(data)


# ============================================================================
# Utility Functions
# ============================================================================

def logsigmoid(x):
    """Numerically stable log-sigmoid."""
    return -jnp.logaddexp(0, -x)


def layer_norm(x, weight=None, bias=None, eps=1e-6):
    """Layer normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + eps)
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    return normalized


# ============================================================================
# sLSTM Cell (Simplified)
# ============================================================================

class sLSTMState(NamedTuple):
    y: jnp.ndarray  # Output [d_model]
    c: jnp.ndarray  # Cell state [d_model]
    n: jnp.ndarray  # Normalization [d_model]
    m: jnp.ndarray  # Max for stability [d_model]


class sLSTMParams(NamedTuple):
    Wx: jnp.ndarray  # [4*d_model, vocab_size]
    Ry: jnp.ndarray  # [4*d_model, d_model]
    b: jnp.ndarray   # [4*d_model]


def init_slstm_params_seeded(key, vocab_size, d_model):
    """Initialize sLSTM params with seed."""
    key_wx, key_ry = jr.split(key)
    
    wx_scale = 1.0 / math.sqrt(vocab_size)
    ry_scale = 1.0 / math.sqrt(d_model)
    
    Wx = jr.normal(key_wx, (4 * d_model, vocab_size)) * wx_scale
    Ry = jr.normal(key_ry, (4 * d_model, d_model)) * ry_scale
    b = jnp.zeros((4 * d_model,))
    
    return sLSTMParams(Wx=Wx, Ry=Ry, b=b)


def init_slstm_state(d_model):
    """Initialize sLSTM state (no batch dimension)."""
    return sLSTMState(
        y=jnp.zeros(d_model),
        c=jnp.zeros(d_model),
        n=jnp.zeros(d_model),
        m=jnp.zeros(d_model),
    )


def slstm_step(params: sLSTMParams, state: sLSTMState, x: int, d_model: int) -> Tuple[sLSTMState, jnp.ndarray]:
    """Single sLSTM step.
    
    Args:
        params: sLSTM parameters
        state: Previous state
        x: Input byte (scalar int)
        d_model: Model dimension
    
    Returns:
        new_state: Updated state
        output: Hidden state [d_model]
    """
    y_prev, c_prev, n_prev, m_prev = state
    
    # One-hot encode input byte
    x_onehot = jax.nn.one_hot(x, params.Wx.shape[1])  # [vocab_size]
    
    # Compute pre-activations
    z = jnp.dot(x_onehot, params.Wx.T) + jnp.dot(y_prev, params.Ry.T) + params.b  # [4*d_model]
    
    # Split into gates
    i_raw, f_raw, z_raw, o_raw = jnp.split(z, 4)  # Each [d_model]
    
    # Exponential stabilization
    log_f_t = logsigmoid(f_raw) + m_prev
    m_t = jnp.maximum(i_raw, log_f_t)
    
    i_t = jnp.minimum(1.0, jnp.exp(i_raw - m_t))
    f_t = jnp.minimum(1.0, jnp.exp(log_f_t - m_t))
    o_t = jax.nn.sigmoid(o_raw)
    
    # Cell update
    c_t = f_t * c_prev + i_t * jnp.tanh(z_raw)
    n_t = f_t * n_prev + i_t
    
    # Output
    eps = 1e-8
    y_t = o_t * c_t / (n_t + eps)
    
    new_state = sLSTMState(y=y_t, c=c_t, n=n_t, m=m_t)
    return new_state, y_t


# ============================================================================
# mLSTM Cell (Simplified)
# ============================================================================

class mLSTMState(NamedTuple):
    C: jnp.ndarray  # Covariance [num_heads, d_head, d_head]
    n: jnp.ndarray  # Normalization [num_heads, d_head]
    m: jnp.ndarray  # Max [num_heads]


class mLSTMParams(NamedTuple):
    W_q: jnp.ndarray
    W_k: jnp.ndarray
    W_v: jnp.ndarray
    W_i: jnp.ndarray
    W_f: jnp.ndarray
    W_out: jnp.ndarray


def init_mlstm_params_seeded(key, vocab_size, d_model, num_heads):
    """Initialize mLSTM params with seed."""
    d_h = d_model // num_heads
    keys = jr.split(key, 6)
    
    scale_in = 1.0 / math.sqrt(vocab_size)
    scale_out = 1.0 / math.sqrt(d_model)
    
    return mLSTMParams(
        W_q=jr.normal(keys[0], (num_heads, d_h, vocab_size)) * scale_in,
        W_k=jr.normal(keys[1], (num_heads, d_h, vocab_size)) * scale_in,
        W_v=jr.normal(keys[2], (num_heads, d_h, vocab_size)) * scale_in,
        W_i=jr.normal(keys[3], (num_heads, vocab_size)) * scale_in,
        W_f=jr.normal(keys[4], (num_heads, vocab_size)) * scale_in,
        W_out=jr.normal(keys[5], (d_model, num_heads * d_h)) * scale_out,
    )


def init_mlstm_state(d_model, num_heads):
    """Initialize mLSTM state."""
    d_h = d_model // num_heads
    return mLSTMState(
        C=jnp.zeros((num_heads, d_h, d_h)),
        n=jnp.zeros((num_heads, d_h)),
        m=jnp.zeros(num_heads),
    )


def mlstm_step(
    params: mLSTMParams,
    state: mLSTMState,
    x: int,
    d_model: int,
    num_heads: int,
) -> Tuple[mLSTMState, jnp.ndarray]:
    """Single mLSTM step."""
    d_h = d_model // num_heads
    x_onehot = jax.nn.one_hot(x, params.W_q.shape[2])  # [vocab_size]
    
    # Q, K, V
    Q = jnp.einsum('ldi,i->ld', params.W_q, x_onehot)
    K = jnp.einsum('ldi,i->ld', params.W_k, x_onehot)
    V = jnp.einsum('ldi,i->ld', params.W_v, x_onehot)
    
    # Gates
    i_raw = jnp.dot(params.W_i, x_onehot)
    f_raw = jnp.dot(params.W_f, x_onehot)
    
    log_f = logsigmoid(f_raw)
    m_t = jnp.maximum(log_f + state.m, i_raw)
    
    f_t = jnp.minimum(1.0, jnp.exp(log_f + state.m - m_t))
    i_t = jnp.minimum(1.0, jnp.exp(i_raw - m_t))
    
    # Update covariance
    K_scaled = K / jnp.sqrt(d_h)
    C_t = f_t[:, jnp.newaxis, jnp.newaxis] * state.C + i_t[:, jnp.newaxis, jnp.newaxis] * jnp.einsum('ld,lh->ldh', K_scaled, V)
    n_t = f_t[:, jnp.newaxis] * state.n + i_t[:, jnp.newaxis] * K_scaled
    
    # Output
    h_num = jnp.einsum('ld,ldh->lh', Q, C_t)
    qn = jnp.sum(Q * n_t, axis=1)
    h_denom = jnp.maximum(jnp.abs(qn), jnp.exp(-m_t))
    h = h_num / (h_denom[:, jnp.newaxis] + 1e-8)
    
    h_flat = h.reshape(d_model)
    output = jnp.dot(h_flat, params.W_out.T)
    
    new_state = mLSTMState(C=C_t, n=n_t, m=m_t)
    return new_state, output


# ============================================================================
# xLSTM Block
# ============================================================================

class xLSTMBlockParams(NamedTuple):
    slstm: Optional[sLSTMParams] = None
    mlstm: Optional[mLSTMParams] = None


class xLSTMStackParams(NamedTuple):
    """Stack of xLSTM blocks (generated from seed)."""
    blocks: List[xLSTMBlockParams]


def init_xlstm_stack_seeded(key, config: Config) -> xLSTMStackParams:
    """Initialize xLSTM stack from seed."""
    blocks = []
    
    for i in range(config.num_layers):
        block_key = jr.fold_in(key, i)
        keys = jr.split(block_key, 2)
        
        slstm = None
        mlstm = None
        
        if config.use_slstm:
            slstm = init_slstm_params_seeded(keys[0], config.vocab_size, config.d_model)
        
        if config.use_mlstm:
            mlstm = init_mlstm_params_seeded(keys[1], config.vocab_size, config.d_model, config.num_heads)
        
        blocks.append(xLSTMBlockParams(slstm=slstm, mlstm=mlstm))
    
    return xLSTMStackParams(blocks=blocks)


def xlstm_forward(
    params: xLSTMStackParams,
    byte_val: int,
    config: Config,
) -> jnp.ndarray:
    """Forward pass through xLSTM for single byte.
    
    Returns hidden state [d_model] from processing the byte.
    """
    d_model = config.d_model
    num_heads = config.num_heads
    
    # Initialize states
    slstm_states = []
    mlstm_states = []
    
    if config.use_slstm:
        for _ in range(config.num_layers):
            slstm_states.append(init_slstm_state(d_model))
    
    if config.use_mlstm:
        for _ in range(config.num_layers):
            mlstm_states.append(init_mlstm_state(d_model, num_heads))
    
    # Encode byte
    x = jax.nn.one_hot(byte_val, config.vocab_size)  # [vocab_size]
    
    # Process through layers
    for layer_idx, block in enumerate(params.blocks):
        slstm_out = x if not config.use_slstm else None
        mlstm_out = x if not config.use_mlstm else None
        
        if config.use_slstm:
            new_state, slstm_out = slstm_step(block.slstm, slstm_states[layer_idx], byte_val, d_model)
            slstm_states[layer_idx] = new_state
        
        if config.use_mlstm:
            new_state, mlstm_out = mlstm_step(block.mlstm, mlstm_states[layer_idx], byte_val, d_model, num_heads)
            mlstm_states[layer_idx] = new_state
        
        # Combine outputs (residual-style)
        x = slstm_out + mlstm_out if (config.use_slstm and config.use_mlstm) else (slstm_out or mlstm_out)
    
    return x


def xlstm_sequence_forward(
    params: xLSTMStackParams,
    bytes_seq: np.ndarray,
    config: Config,
) -> List[jnp.ndarray]:
    """Forward pass through sequence of bytes.
    
    Returns list of hidden states, one per byte.
    """
    hidden_states = []
    for byte_val in bytes_seq:
        h = xlstm_forward(params, int(byte_val), config)
        hidden_states.append(h)
    return hidden_states


# ============================================================================
# Linear Layer Training
# ============================================================================

class LinearLayerParams(NamedTuple):
    weight: jnp.ndarray  # [d_model, 256]
    bias: jnp.ndarray    # [256]


def init_linear_layer(key, d_model):
    """Initialize linear layer parameters."""
    scale = 1.0 / math.sqrt(d_model)
    weight = jr.normal(key, (d_model, 256)) * scale
    bias = jnp.zeros(256)
    return LinearLayerParams(weight=weight, bias=bias)


def linear_forward(params: LinearLayerParams, h: jnp.ndarray) -> jnp.ndarray:
    """Forward pass through linear layer.
    
    Args:
        h: Hidden state [d_model]
    
    Returns:
        logits: [256] (vocabulary logits)
    """
    return jnp.dot(h, params.weight) + params.bias


def compute_loss(logits: jnp.ndarray, target_byte: int) -> jnp.ndarray:
    """Cross-entropy loss for single byte."""
    target_onehot = jax.nn.one_hot(target_byte, 256)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(target_onehot * log_probs)
    return loss


def train_linear_layer(
    params: LinearLayerParams,
    hidden_states: List[jnp.ndarray],
    target_bytes: np.ndarray,
    config: Config,
) -> Tuple[LinearLayerParams, float]:
    """Train linear layer for segment.
    
    Args:
        hidden_states: List of hidden states [d_model] for each byte
        target_bytes: Target byte values
        config: Configuration
    
    Returns:
        trained_params: Updated parameters
        final_loss: Final training loss
    """
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    
    def loss_fn(p):
        total_loss = 0.0
        for h, target in zip(hidden_states, target_bytes):
            logits = linear_forward(p, h)
            total_loss = total_loss + compute_loss(logits, target)
        return total_loss / len(hidden_states)
    
    # Training loop
    for step in range(config.max_training_steps):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # Gradient clipping
        grads_flat = jax.tree_util.tree_leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads_flat))
        scale = jnp.minimum(1.0, config.gradient_clip_norm / (grad_norm + 1e-8))
        grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        
        # Update
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if step % 20 == 0:
            print(f"  Train step {step}: loss = {loss:.6f}")
    
    return params, float(loss)


# ============================================================================
# Main Compression Algorithm
# ============================================================================

def compress_data(
    data: np.ndarray,
    seed: int = 42,
    config: Optional[Config] = None,
) -> CompressionState:
    """Compress byte data using xLSTM.
    
    Args:
        data: Byte array to compress [n]
        seed: Random seed for xLSTM weight generation
        config: Configuration
    
    Returns:
        CompressionState with seed and learned layers
    """
    if config is None:
        config = Config()
    
    print(f"Starting compression of {len(data)} bytes...")
    print(f"Config: d_model={config.d_model}, num_layers={config.num_layers}")
    print(f"Seed: {seed}")
    
    # Generate xLSTM params from seed
    key = jr.PRNGKey(seed)
    xlstm_params = init_xlstm_stack_seeded(key, config)
    print("xLSTM model initialized with seed")
    
    segments = []
    current_segment_start = 0
    current_layer = None
    current_hidden_states = []
    current_targets = []
    
    # Main compression loop
    for byte_idx in range(len(data)):
        byte_val = data[byte_idx]
        
        # Get hidden state for this byte
        h = xlstm_forward(xlstm_params, int(byte_val), config)
        
        if current_layer is None:
            # Initialize first layer
            layer_key = jr.fold_in(key, byte_idx)
            current_layer = init_linear_layer(layer_key, config.d_model)
        
        # Try to predict with current layer
        logits = linear_forward(current_layer, h)
        pred_loss = float(compute_loss(logits, int(byte_val)))
        
        current_hidden_states.append(h)
        current_targets.append(byte_val)
        
        # Check if we should start a new layer
        if pred_loss > config.loss_threshold and byte_idx > 0:
            # Train current layer
            print(f"\nSegment {len(segments)}: bytes {current_segment_start}-{byte_idx - 1}")
            print(f"  Pre-training loss: {pred_loss:.6f}")
            
            trained_layer, final_loss = train_linear_layer(
                current_layer,
                current_hidden_states,
                current_targets,
                config,
            )
            
            print(f"  Final training loss: {final_loss:.6f}")
            
            # Save segment
            segment = Segment(
                start_idx=current_segment_start,
                end_idx=byte_idx - 1,
                layer_weights=trained_layer.weight,
                layer_bias=trained_layer.bias,
            )
            segments.append(segment)
            
            # Start new segment
            current_segment_start = byte_idx
            current_layer = None
            current_hidden_states = []
            current_targets = []
            print(f"  Starting new segment at byte {byte_idx}")
        
        if byte_idx % 100 == 0 and byte_idx > 0:
            print(f"Processed {byte_idx} bytes, {len(segments)} segments so far")
    
    # Finalize last segment
    if current_layer is not None and len(current_targets) > 0:
        print(f"\nFinal segment {len(segments)}: bytes {current_segment_start}-{len(data) - 1}")
        trained_layer, final_loss = train_linear_layer(
            current_layer,
            current_hidden_states,
            current_targets,
            config,
        )
        print(f"  Final training loss: {final_loss:.6f}")
        
        segment = Segment(
            start_idx=current_segment_start,
            end_idx=len(data) - 1,
            layer_weights=trained_layer.weight,
            layer_bias=trained_layer.bias,
        )
        segments.append(segment)
    
    # Create compression state
    compression_state = CompressionState(
        first_byte=int(data[0]),
        seed=seed,
        config=config,
        segments=segments,
    )
    
    print(f"\nCompression complete!")
    print(f"Total segments: {len(segments)}")
    
    # Compute compression ratio
    original_size = len(data)
    seed_size = 8  # uint64
    config_size = 100  # Approximate
    segment_size = len(segments) * (
        8  # start_idx + end_idx (2x4 bytes)
        + (config.d_model * 256 * 4)  # weights (float32)
        + (256 * 4)  # bias (float32)
    )
    total_stored = seed_size + config_size + segment_size
    ratio = (total_stored / original_size) * 100
    
    print(f"Original size: {original_size} bytes")
    print(f"Stored size: {total_stored} bytes (~{ratio:.1f}%)")
    
    return compression_state


# ============================================================================
# Decompression
# ============================================================================

def decompress_data(compression_state: CompressionState) -> np.ndarray:
    """Decompress data from compression state.
    
    Args:
        compression_state: Compression state with seed and layers
    
    Returns:
        Decompressed byte array
    """
    print("Starting decompression...")
    
    config = compression_state.config
    seed = compression_state.seed
    
    # Regenerate xLSTM from seed
    key = jr.PRNGKey(seed)
    xlstm_params = init_xlstm_stack_seeded(key, config)
    
    # Reconstruct bytes
    reconstructed = []
    segment_idx = 0
    current_segment = compression_state.segments[segment_idx]
    
    for byte_idx in range(current_segment.end_idx + 1):
        # Get current segment
        if segment_idx < len(compression_state.segments):
            current_segment = compression_state.segments[segment_idx]
            if byte_idx > current_segment.end_idx and segment_idx + 1 < len(compression_state.segments):
                segment_idx += 1
                current_segment = compression_state.segments[segment_idx]
        
        # Get hidden state
        if len(reconstructed) == 0:
            # First byte is stored
            reconstructed.append(compression_state.first_byte)
            continue
        
        byte_val = reconstructed[-1]
        h = xlstm_forward(xlstm_params, int(byte_val), config)
        
        # Use stored layer to predict
        layer_params = LinearLayerParams(
            weight=current_segment.layer_weights,
            bias=current_segment.layer_bias,
        )
        logits = linear_forward(layer_params, h)
        pred_byte = jnp.argmax(logits)
        reconstructed.append(int(pred_byte))
    
    reconstructed = np.array(reconstructed, dtype=np.uint8)
    print(f"Decompression complete: {len(reconstructed)} bytes")
    
    return reconstructed


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Load sample data
    data_path = Path("../datasets/quran-uthmani-numbered.txt")
    if data_path.exists():
        with open(data_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        # Use first 1000 bytes for testing
        data = data[:1000]
    else:
        # Synthetic data for testing
        print("Dataset not found, using synthetic data...")
        data = np.random.randint(0, 256, size=500, dtype=np.uint8)
    
    print(f"Data shape: {data.shape}")
    print(f"First 20 bytes: {data[:20]}")
    
    # Compress
    config = Config(
        d_model=128,
        num_layers=2,
        num_heads=4,
        loss_threshold=0.3,
        max_training_steps=50,
    )
    
    compression_state = compress_data(data, seed=42, config=config)
    
    # Save
    compression_state.save("compression_state.pkl")
    
    # Load and decompress
    loaded_state = CompressionState.load("compression_state.pkl")
    reconstructed = decompress_data(loaded_state)
    
    # Verify
    print(f"\nOriginal size: {len(data)}")
    print(f"Reconstructed size: {len(reconstructed)}")
    print(f"Match: {np.allclose(data, reconstructed[:len(data)])}")
