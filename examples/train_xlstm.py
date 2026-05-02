"""
xLSTM Training Script - Pure JAX
Based on xlstm_equations.md and xlstm_jax.py
Trains xLSTM on text data with byte tokenization.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import Tuple
import time
from tqdm import tqdm
from pathlib import Path
import optax

"""
Pure JAX implementation of xLSTM (Extended LSTM).
Includes sLSTM, mLSTM, and combined xLSTM blocks.
Based on: https://arxiv.org/abs/2405.04517
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import Dict, Tuple, NamedTuple, Optional
from functools import partial
import math


# ============================================================================
# Configuration
# ============================================================================

class Config(NamedTuple):
    """Configuration for xLSTM model."""
    # Model dimensions
    vocab_size: int = 256  # For byte tokenization
    d_model: int = 256
    num_heads: int = 8
    d_head: int = 32  # Per-head dimension
    d_ff: int = 1024
    num_layers: int = 4
    max_seq_len: int = 512
    
    # xLSTM specific
    # use_slstm: bool = True
    # use_mlstm: bool = True
    soft_cap: float = 15.0
    
    # Regularization
    dropout_rate: float = 0.1
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
    
    # Inference
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9


# ============================================================================
# Activation Functions and Utilities
# ============================================================================

def logsigmoid(x):
    """Numerically stable log-sigmoid: log(1 / (1 + exp(-x)))."""
    return -jnp.logaddexp(0, -x)


def soft_cap(x, cap_value):
    """Soft capping using tanh: cap * tanh(x / cap)."""
    if cap_value is None:
        return x
    return cap_value * jnp.tanh(x / cap_value)


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
# sLSTM: Scalar LSTM with Four States
# ============================================================================

class sLSTMState(NamedTuple):
    """sLSTM state: (y, c, n, m)."""
    y: jnp.ndarray  # Output hidden state [B, H]
    c: jnp.ndarray  # Cell state / numerator [B, H]
    n: jnp.ndarray  # Normalization denominator [B, H]
    m: jnp.ndarray  # Stability max value [B, H]


class sLSTMParams(NamedTuple):
    """sLSTM parameters."""
    Wx: jnp.ndarray  # Input to hidden [4*H, d_in]
    Ry: jnp.ndarray  # Recurrent [4*H, H]
    b: jnp.ndarray   # Bias [4*H]
    layer_norm_weight: jnp.ndarray  # Layer norm scale [H]
    layer_norm_bias: jnp.ndarray    # Layer norm bias [H]


def init_slstm_params(key, d_in, d_hidden, scale=0.01):
    """Initialize sLSTM parameters."""
    key_wx, key_ry, key_ln = jr.split(key, 3)
    
    # Xavier-like initialization
    wx_scale = 1.0 / math.sqrt(d_in)
    ry_scale = 1.0 / math.sqrt(d_hidden)
    
    Wx = jr.normal(key_wx, (4 * d_hidden, d_in)) * wx_scale
    Ry = jr.normal(key_ry, (4 * d_hidden, d_hidden)) * ry_scale
    b = jnp.zeros((4 * d_hidden,))
    
    ln_weight = jnp.ones((d_hidden,))
    ln_bias = jnp.zeros((d_hidden,))
    
    return sLSTMParams(Wx, Ry, b, ln_weight, ln_bias)


def init_slstm_state(batch_size, d_hidden):
    """Initialize sLSTM state."""
    return sLSTMState(
        y=jnp.zeros((batch_size, d_hidden)),
        c=jnp.zeros((batch_size, d_hidden)),
        n=jnp.zeros((batch_size, d_hidden)),
        m=jnp.zeros((batch_size, d_hidden)),
    )


def slstm_step(params: sLSTMParams, state: sLSTMState, x: jnp.ndarray) -> Tuple[sLSTMState, jnp.ndarray]:
    """Single sLSTM step.
    
    Implements the forward pass described in xlstm_equations.md Section 2.
    
    Args:
        params: sLSTM parameters
        state: Previous state (y, c, n, m)
        x: Input [B, d_in]
    
    Returns:
        new_state: Updated sLSTM state
        output: Output y_t [B, H]
    """
    y_prev, c_prev, n_prev, m_prev = state
    H = params.Wx.shape[0] // 4
    
    # Input processing: z = Wx * x + Ry * y + b
    z = jnp.dot(x, params.Wx.T) + jnp.dot(y_prev, params.Ry.T) + params.b  # [B, 4*H]
    
    # Split into four gate components
    i_raw, f_raw, z_raw, o_raw = jnp.split(z, 4, axis=1)  # Each [B, H]
    
    # ========== Gate computation with exponential stabilization ==========
    # Compute log(f_t) = log_sigmoid(f_raw) + m_prev
    log_f_t = logsigmoid(f_raw) + m_prev  # [B, H]
    
    # Compute new maximum for numerical stability
    # m_t = max(i_raw, log_f_t) where log_f_t = log_sigmoid(f_raw) + m_prev
    m_t = jnp.maximum(i_raw, log_f_t)  # [B, H]
    
    # Gate values with exponential stabilization
    # i_t = min(1, exp(i_raw - m_t))
    i_t = jnp.minimum(1.0, jnp.exp(i_raw - m_t))  # [B, H]
    
    # f_t = min(1, exp(log_f_t - m_t))
    f_t = jnp.minimum(1.0, jnp.exp(log_f_t - m_t))  # [B, H]
    
    # Output gate: o_t = sigmoid(o_raw)
    o_t = jax.nn.sigmoid(o_raw)  # [B, H]
    
    # ========== Cell state update ==========
    c_t = f_t * c_prev + i_t * jnp.tanh(z_raw)  # [B, H]
    
    # ========== Normalization tracking ==========
    n_t = f_t * n_prev + i_t  # [B, H]
    
    # ========== Output ==========
    # y_t = o_t * (c_t / n_t)
    # Add small epsilon for numerical stability
    epsilon = 1e-8
    y_t = o_t * c_t / (n_t + epsilon)  # [B, H]
    
    # Apply layer normalization
    y_t_norm = layer_norm(y_t, params.layer_norm_weight, params.layer_norm_bias)
    
    # Create new state
    new_state = sLSTMState(y=y_t_norm, c=c_t, n=n_t, m=m_t)
    
    return new_state, y_t_norm


# ============================================================================
# mLSTM: Matrix LSTM with Covariance Memory
# ============================================================================

class mLSTMState(NamedTuple):
    """mLSTM state."""
    C: jnp.ndarray      # Covariance matrix [B, L, D_h, D_h]
    n: jnp.ndarray      # Normalization vector [B, L, D_h]
    m: jnp.ndarray      # Stability constant [B, L]


class mLSTMParams(NamedTuple):
    """mLSTM parameters."""
    # Query, Key, Value projections
    W_q: jnp.ndarray
    W_k: jnp.ndarray
    W_v: jnp.ndarray
    
    # Gate projections (input and forget)
    W_i: jnp.ndarray
    W_f: jnp.ndarray
    
    # Output projection
    W_out: jnp.ndarray
    
    # Layer norm
    ln_weight: jnp.ndarray
    ln_bias: jnp.ndarray


def init_mlstm_params(key, d_in, d_model, num_heads):
    """Initialize mLSTM parameters."""
    d_h = d_model // num_heads
    
    keys = jr.split(key, 8)
    
    scale_in = 1.0 / math.sqrt(d_in)
    scale_out = 1.0 / math.sqrt(d_model)
    scale_gate = 1.0 / math.sqrt(d_in)
    
    return mLSTMParams(
        W_q=jr.normal(keys[0], (num_heads, d_h, d_in)) * scale_in,
        W_k=jr.normal(keys[1], (num_heads, d_h, d_in)) * scale_in,
        W_v=jr.normal(keys[2], (num_heads, d_h, d_in)) * scale_in,
        W_i=jr.normal(keys[3], (num_heads, d_in)) * scale_gate,
        W_f=jr.normal(keys[4], (num_heads, d_in)) * scale_gate,
        W_out=jr.normal(keys[5], (d_model, num_heads * d_h)) * scale_out,
        ln_weight=jnp.ones((d_model,)),
        ln_bias=jnp.zeros((d_model,)),
    )


def init_mlstm_state(batch_size, num_heads, d_h):
    """Initialize mLSTM state."""
    return mLSTMState(
        C=jnp.zeros((batch_size, num_heads, d_h, d_h)),
        n=jnp.zeros((batch_size, num_heads, d_h)),
        m=jnp.zeros((batch_size, num_heads)),
    )


def mlstm_step(
    params: mLSTMParams,
    state: mLSTMState,
    x: jnp.ndarray,
) -> Tuple[mLSTMState, jnp.ndarray]:
    """Single mLSTM step (recurrent form).
    
    Implements the recurrent (single-step) form described in xlstm_equations.md Section 3.
    
    Args:
        params: mLSTM parameters
        state: Previous state (C, n, m)
        x: Input [B, d_in]
    
    Returns:
        new_state: Updated mLSTM state
        output: Output [B, d_model]
    """
    B = x.shape[0]
    C_prev, n_prev, m_prev = state
    
    num_heads = params.W_q.shape[0]
    d_h = params.W_q.shape[1]
    d_model = d_h * num_heads
    
    # ========== Compute Q, K, V ==========
    # Q: [B, L, D_h]
    Q = jnp.einsum('ldi,bi->bld', params.W_q, x)
    K = jnp.einsum('ldi,bi->bld', params.W_k, x)
    V = jnp.einsum('ldi,bi->bld', params.W_v, x)
    
    # ========== Gate pre-activations ==========
    # i_raw, f_raw: [B, L]
    i_raw = jnp.einsum('ld,bd->bl', params.W_i, x)  # [B, L]
    f_raw = jnp.einsum('ld,bd->bl', params.W_f, x)  # [B, L]
    
    # ========== Exponential gate stabilization ==========
    log_f = logsigmoid(f_raw)  # [B, L]
    
    # m_t = max(log_f + m_prev, i_raw)
    m_t = jnp.maximum(log_f + m_prev, i_raw)  # [B, L]
    
    # Gate values
    f_t = jnp.minimum(1.0, jnp.exp(log_f + m_prev - m_t))  # [B, L]
    i_t = jnp.minimum(1.0, jnp.exp(i_raw - m_t))  # [B, L]
    
    # ========== Scale keys ==========
    K_scaled = K / jnp.sqrt(d_h)  # [B, L, D_h]
    
    # ========== Update C (covariance matrix) ==========
    # C_t = f_t * C_prev + i_t * (K_scaled^T @ V)
    # Shape: [B, L, D_h, D_h]
    f_t_expanded = f_t[:, :, jnp.newaxis, jnp.newaxis]  # [B, L, 1, 1]
    i_t_expanded = i_t[:, :, jnp.newaxis, jnp.newaxis]  # [B, L, 1, 1]
    
    kv_product = jnp.einsum('bld,blh->blhd', K_scaled, V)  # [B, L, D_h, D_h]
    C_t = f_t_expanded * C_prev + i_t_expanded * kv_product
    
    # ========== Update n (normalization vector) ==========
    # n_t = f_t * n_prev + i_t * K_scaled
    n_t = f_t[:, :, jnp.newaxis] * n_prev + i_t[:, :, jnp.newaxis] * K_scaled  # [B, L, D_h]
    
    # ========== Compute output ==========
    # h_num = Q^T @ C_t: [B, L, D_h]
    h_num = jnp.einsum('bld,blhd->blh', Q, C_t)
    
    # h_denom = max(|Q^T @ n_t|, exp(-m_t))
    qn_prod = jnp.sum(Q * n_t, axis=2)  # [B, L]
    max_val = jnp.exp(-m_t)  # [B, L]
    h_denom = jnp.maximum(jnp.abs(qn_prod), max_val)  # [B, L]
    
    # h = h_num / h_denom
    epsilon = 1e-8
    h = h_num / (h_denom[:, :, jnp.newaxis] + epsilon)  # [B, L, D_h]
    
    # ========== Project output ==========
    # Reshape [B, L, D_h] -> [B, L*D_h]
    h_flat = h.reshape((B, d_model))
    
    # Output projection
    output = jnp.dot(h_flat, params.W_out.T)
    
    # Apply layer normalization
    output = layer_norm(output, params.ln_weight, params.ln_bias)
    
    # Create new state
    new_state = mLSTMState(C=C_t, n=n_t, m=m_t)
    
    return new_state, output


# ============================================================================
# Full xLSTM Block and Model
# ============================================================================

class xLSTMBlockParams(NamedTuple):
    """Single xLSTM block."""
    slstm: Optional[sLSTMParams] = None
    mlstm: Optional[mLSTMParams] = None
    ff_w1: Optional[jnp.ndarray] = None
    ff_b1: Optional[jnp.ndarray] = None
    ff_w2: Optional[jnp.ndarray] = None
    ff_b2: Optional[jnp.ndarray] = None


class xLSTMParams(NamedTuple):
    """Full xLSTM model parameters."""
    embedding: jnp.ndarray  # [vocab_size, d_model]
    blocks: list  # List of xLSTMBlockParams
    output_proj: jnp.ndarray  # [d_model, vocab_size]


def init_xlstm_params(key, config: Config) -> xLSTMParams:
    """Initialize all xLSTM parameters."""
    keys = jr.split(key, 10)
    
    d_in = config.d_model
    d_h = config.d_model // config.num_heads
    
    # Embedding
    embedding = jr.normal(keys[0], (config.vocab_size, config.d_model)) * 0.01
    
    # Initialize blocks
    blocks = []
    for i in range(config.num_layers):
        block_keys = jr.split(keys[1 + i], 6)
        
        slstm_params = None
        mlstm_params = None
        
        # if config.use_slstm:
        slstm_params = init_slstm_params(block_keys[0], d_in, d_in)
        # if config.use_mlstm:
        mlstm_params = init_mlstm_params(block_keys[1], d_in, config.d_model, config.num_heads)
        
        # FFN parameters
        scale_ff1 = 1.0 / math.sqrt(d_in)
        scale_ff2 = 1.0 / math.sqrt(config.d_ff)
        
        ff_w1 = jr.normal(block_keys[2], (config.d_ff, d_in)) * scale_ff1
        ff_b1 = jnp.zeros((config.d_ff,))
        ff_w2 = jr.normal(block_keys[3], (d_in, config.d_ff)) * scale_ff2
        ff_b2 = jnp.zeros((d_in,))
        
        block = xLSTMBlockParams(
            slstm=slstm_params,
            mlstm=mlstm_params,
            ff_w1=ff_w1,
            ff_b1=ff_b1,
            ff_w2=ff_w2,
            ff_b2=ff_b2,
        )
        blocks.append(block)
    
    # Output projection
    output_proj = jr.normal(keys[-1], (config.d_model, config.vocab_size)) * 0.01
    
    return xLSTMParams(embedding=embedding, blocks=blocks, output_proj=output_proj)


# ============================================================================
# Training
# ============================================================================

def cross_entropy_loss(logits, targets, ignore_index=-1):
    """Compute cross-entropy loss."""
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    
    # One-hot encode targets
    num_classes = logits.shape[-1]
    targets_one_hot = jax.nn.one_hot(targets_flat, num_classes)
    
    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    
    # Compute loss per element
    loss_per_element = -jnp.sum(targets_one_hot * log_probs, axis=1)
    
    # Mask out ignore_index
    if ignore_index >= 0:
        mask = targets_flat != ignore_index
        loss_per_element = loss_per_element * mask
        loss = jnp.sum(loss_per_element) / jnp.maximum(jnp.sum(mask), 1)
    else:
        loss = jnp.mean(loss_per_element)
    
    return loss


def forward_pass(params: xLSTMParams, tokens: jnp.ndarray, config: Config, key=None):
    """Forward pass through xLSTM model.
    
    Args:
        params: Model parameters
        tokens: Input tokens [B, T]
        config: Configuration
        key: PRNG key (for dropout)
    
    Returns:
        logits: [B, T, vocab_size]
    """
    B, T = tokens.shape
    
    # Embedding lookup
    x = params.embedding[tokens]  # [B, T, d_model]
    
    # Process through blocks
    for block_idx, block in enumerate(params.blocks):
        # Initialize states
        slstm_state = init_slstm_state(B, config.d_model)# if config.use_slstm else None
        mlstm_state = init_mlstm_state(B, config.num_heads, config.d_model // config.num_heads)# if config.use_mlstm else None
        
        # Process sequence
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # [B, d_model]
            
            # sLSTM path
            slstm_state, slstm_out = slstm_step(block.slstm, slstm_state, x_t)
            mlstm_state, mlstm_out = mlstm_step(block.mlstm, mlstm_state, x_t)
            
            # Combine paths: simple concatenation then projection
            combined = slstm_out + mlstm_out  # Residual-like combination
            
            # FFN
            hidden = jnp.dot(combined, block.ff_w1.T) + block.ff_b1
            hidden = jax.nn.gelu(hidden)
            ffn_out = jnp.dot(hidden, block.ff_w2.T) + block.ff_b2
            
            # Residual connection
            x_t_out = x_t + combined + ffn_out
            
            outputs.append(x_t_out)
        
        x = jnp.stack(outputs, axis=1)  # [B, T, d_model]
    
    # Output projection to vocabulary
    logits = jnp.dot(x, params.output_proj)  # [B, T, vocab_size]
    
    return logits


@jax.jit
def compute_loss(params: xLSTMParams, tokens: jnp.ndarray, config: Config):
    """Compute language modeling loss."""
    # Use all but last token as input, all but first as target
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    
    logits = forward_pass(params, inputs, config)
    loss = cross_entropy_loss(logits, targets)
    
    return loss


def update_step(params, optimizer_state, tokens, config, optimizer):
    """Single training step."""
    loss, grads = jax.value_and_grad(lambda p: compute_loss(p, tokens, config))(params)
    
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    
    return params, optimizer_state, loss


# ============================================================================
# Helper functions
# ============================================================================

def generate(
    params: xLSTMParams,
    seed_tokens: jnp.ndarray,
    max_length: int,
    config: Config,
    key,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    """Generate text from seed."""
    generated = list(seed_tokens.flatten())
    
    for _ in range(max_length - len(generated)):
        # Get last config.max_seq_len tokens
        context = jnp.array([generated[-config.max_seq_len:]], dtype=jnp.int32)
        
        # Forward pass
        logits = forward_pass(params, context, config, key)
        
        # Get last token logits
        next_logits = logits[0, -1, :] / temperature
        
        # Top-k sampling
        if top_k is not None:
            indices = jnp.argsort(next_logits)[-top_k:]
            next_logits = jnp.where(
                jnp.arange(config.vocab_size) >= indices[0],
                next_logits,
                -jnp.inf,
            )
        
        # Sample
        key, subkey = jr.split(key)
        next_token = jr.categorical(subkey, next_logits)
        generated.append(int(next_token))
    
    return jnp.array(generated, dtype=jnp.int32)


# ============================================================================
# Data Loading and Tokenization
# ============================================================================

class ByteTokenizer:
    """Simple byte-level tokenizer."""
    vocab_size = 256
    
    @staticmethod
    def encode(text: str) -> np.ndarray:
        """Encode text to bytes."""
        if isinstance(text, str):
            text = text.encode('utf-8')
        return np.frombuffer(text, dtype=np.uint8)
    
    @staticmethod
    def decode(tokens: np.ndarray) -> str:
        """Decode bytes to text."""
        return bytes(tokens.astype(np.uint8)).decode('utf-8', errors='replace')


def load_dataset(path: str) -> np.ndarray:
    """Load and tokenize dataset."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = ByteTokenizer.encode(text)
    return tokens


def create_batches(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True,
    key=None,
):
    """Create batches of sequences.
    
    Args:
        tokens: 1D array of token IDs
        batch_size: Batch size
        seq_len: Sequence length
        shuffle: Whether to shuffle batches
        key: PRNG key for shuffling
    
    Yields:
        Batch of shape [batch_size, seq_len]
    """
    num_sequences = (len(tokens) - 1) // seq_len
    
    # Create starting indices
    indices = np.arange(num_sequences) * seq_len
    
    if shuffle and key is not None:
        key, subkey = jr.split(key)
        indices = jr.permutation(subkey, indices)
    
    # Create batches
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch = []
        
        for idx in batch_indices:
            seq = tokens[idx : idx + seq_len + 1]  # +1 for target
            if len(seq) == seq_len + 1:
                batch.append(seq)
        
        if batch:
            batch = np.stack(batch)
            yield jnp.array(batch, dtype=jnp.int32)


# ============================================================================
# Training Loop
# ============================================================================

def create_optimizer(learning_rate: float, weight_decay: float):
    """Create optimizer (AdamW)."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )


def train_step(
    params: xLSTMParams,
    optimizer_state,
    batch: jnp.ndarray,
    config: Config,
    optimizer,
):
    """Single training step.
    
    Args:
        params: Model parameters
        optimizer_state: Optimizer state
        batch: Batch of shape [B, T+1]
        config: Configuration
        optimizer: Optax optimizer
    
    Returns:
        Updated params, optimizer_state, and loss
    """
    # Split into input and target
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    def loss_fn(p):
        logits = forward_pass(p, inputs, config)
        return cross_entropy_loss(logits, targets)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    
    return params, optimizer_state, loss


@jax.jit
def eval_step(params: xLSTMParams, batch: jnp.ndarray, config: Config):
    """Evaluation step."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    logits = forward_pass(params, inputs, config)
    loss = cross_entropy_loss(logits, targets)
    
    return loss


def main():
    """Main training function."""
    # ========== Configuration ==========
    config = Config(
        vocab_size=256,
        d_model=256,
        num_heads=8,
        d_head=32,
        d_ff=1024,
        num_layers=4,
        max_seq_len=256,
        # use_slstm=True,
        # use_mlstm=True,
        soft_cap=15.0,
        dropout_rate=0.1,
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-6,
        grad_clip_norm=1.0,
    )
    
    # ========== Data Loading ==========
    print("Loading dataset...")
    dataset_path = "datasets/quran-uthmani.txt"
    tokens = load_dataset(dataset_path)
    print(f"Loaded {len(tokens)} tokens")
    # if not Path(dataset_path).exists():
    #     print(f"Dataset not found at {dataset_path}")
    #     print("Using synthetic data for demonstration...")
    #     # Create synthetic data
    #     tokens = np.random.randint(0, config.vocab_size, (10000,))
    # else:
    #     tokens = load_dataset(dataset_path)
    #     print(f"Loaded {len(tokens)} tokens")
    
    # Split into train/val
    train_ratio = 0.9
    split = int(len(tokens) * train_ratio)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    
    print(f"Training tokens: {len(train_tokens)}, Validation tokens: {len(val_tokens)}")
    
    # ========== Model Initialization ==========
    print("Initializing model...")
    key = jr.PRNGKey(42)
    key, subkey = jr.split(key)
    
    params = init_xlstm_params(subkey, config)
    
    # Count parameters
    num_params = sum(
        np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)
    )
    print(f"Model parameters: {num_params:,}")
    
    # ========== Optimizer Setup ==========
    optimizer = create_optimizer(config.learning_rate, config.weight_decay)
    optimizer_state = optimizer.init(params)
    
    # ========== Training Loop ==========
    num_epochs = 3
    steps_per_epoch = (len(train_tokens) - 1) // config.max_seq_len // config.batch_size
    
    global_step = 0
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # ========== Training ==========
        key, subkey = jr.split(key)
        train_losses = []
        
        with tqdm(total=steps_per_epoch, desc="Training") as pbar:
            for batch in create_batches(
                train_tokens,
                config.batch_size,
                config.max_seq_len,
                shuffle=True,
                key=subkey,
            ):
                if batch.shape[0] < config.batch_size:
                    continue
                
                params, optimizer_state, loss = train_step(
                    params, optimizer_state, batch, config, optimizer
                )
                
                train_losses.append(float(loss))
                global_step += 1
                
                pbar.update(1)
                pbar.set_postfix({"loss": f"{np.mean(train_losses[-10:]):.4f}"})
        
        mean_train_loss = np.mean(train_losses)
        print(f"Mean training loss: {mean_train_loss:.4f}")
        
        # ========== Validation ==========
        print("Evaluating on validation set...")
        val_losses = []
        
        for batch in create_batches(
            val_tokens,
            config.batch_size,
            config.max_seq_len,
            shuffle=False,
        ):
            if batch.shape[0] < config.batch_size:
                continue
            
            val_loss = eval_step(params, batch, config)
            val_losses.append(float(val_loss))
        
        mean_val_loss = np.mean(val_losses)
        print(f"Mean validation loss: {mean_val_loss:.4f}")
        print(f"Perplexity: {np.exp(mean_val_loss):.2f}")
        
        # Save if best
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print(f"✓ New best validation loss!")
        
        print("-" * 60)
    
    # ========== Generation Example ==========
    print("\nGeneration example:")
    print("-" * 60)
    
    seed = "The quick brown"
    seed_tokens = ByteTokenizer.encode(seed)
    
    print(f"Seed: {seed}")
    print(f"Generating... (this may be slow on CPU)")
    
    key, subkey = jr.split(key)
    generated = generate(
        params,
        jnp.array(seed_tokens),
        max_length=100,
        config=config,
        key=subkey,
        temperature=0.8,
        top_k=40,
    )
    
    generated_text = ByteTokenizer.decode(np.array(generated))
    print(f"Generated:\n{generated_text}\n")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
