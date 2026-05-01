"""
xLSTM: Extended Long Short-Term Memory in pure JAX without Flax.
Single file with byte/bit/hex tokenization, training, and inference.

Based on: xLSTM: Extended Long Short-Term Memory (Beck et al., 2024)
https://arxiv.org/abs/2405.04517

Features:
- sLSTM (Scalar LSTM) with exponential gating
- mLSTM (Matrix LSTM) with covariance-based updates
- Alternating sLSTM/mLSTM layers
- Pre-normalization with residual connections
- Feedforward networks with GELU activation
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import Dict, Tuple, NamedTuple, Optional
from enum import Enum
import time
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class Config(NamedTuple):
    """Configuration for xLSTM model."""
    # Model dimensions
    vocab_size: int = 256
    d_model: int = 256
    d_ff: int = 1024
    num_heads: int = 8
    num_layers: int = 4
    max_seq_len: int = 512
    
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


class TokenizerType(Enum):
    """Tokenizer types."""
    BYTE = "byte"
    BIT = "bit"
    HEX = "hex"


# ============================================================================
# Tokenizer
# ============================================================================

class Tokenizer:
    """Text tokenizer supporting byte, bit, and hex bases."""
    
    def __init__(self, tokenizer_type: TokenizerType):
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == TokenizerType.BYTE:
            self.vocab_size = 256
        elif tokenizer_type == TokenizerType.BIT:
            self.vocab_size = 2
        elif tokenizer_type == TokenizerType.HEX:
            self.vocab_size = 16
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to token array."""
        if isinstance(text, str):
            text = text.encode('utf-8')
        
        if self.tokenizer_type == TokenizerType.BYTE:
            return np.frombuffer(text, dtype=np.uint8)
        elif self.tokenizer_type == TokenizerType.BIT:
            bytes_array = np.frombuffer(text, dtype=np.uint8)
            bits = []
            for byte in bytes_array:
                for i in range(7, -1, -1):
                    bits.append((byte >> i) & 1)
            return np.array(bits, dtype=np.uint8)
        elif self.tokenizer_type == TokenizerType.HEX:
            bytes_array = np.frombuffer(text, dtype=np.uint8)
            hexes = []
            for byte in bytes_array:
                hexes.append(byte >> 4)
                hexes.append(byte & 0x0F)
            return np.array(hexes, dtype=np.uint8)
    
    def decode(self, tokens: np.ndarray) -> str:
        """Decode token array to text."""
        if self.tokenizer_type == TokenizerType.BYTE:
            return bytes(tokens.astype(np.uint8)).decode('utf-8', errors='replace')
        elif self.tokenizer_type == TokenizerType.BIT:
            bits = tokens.astype(int)
            bytes_list = []
            for i in range(0, len(bits), 8):
                byte = 0
                for j in range(8):
                    if i + j < len(bits):
                        byte = (byte << 1) | bits[i + j]
                    else:
                        byte = byte << 1
                bytes_list.append(byte)
            return bytes(bytes_list).decode('utf-8', errors='replace')
        elif self.tokenizer_type == TokenizerType.HEX:
            hexes = tokens.astype(int)
            bytes_list = []
            for i in range(0, len(hexes), 2):
                if i + 1 < len(hexes):
                    byte = (hexes[i] << 4) | hexes[i + 1]
                    bytes_list.append(byte)
            return bytes(bytes_list).decode('utf-8', errors='replace')
    
    def tokenize_file(self, filepath: str) -> np.ndarray:
        """Tokenize a text file."""
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        return self.encode(text)


# ============================================================================
# Parameter Initialization
# ============================================================================

def init_dense_params(rng, in_features, out_features):
    """Initialize dense layer parameters."""
    rng, subkey = jr.split(rng)
    scale = jnp.sqrt(2.0 / in_features)
    w = jr.normal(subkey, (in_features, out_features)) * scale
    b = jnp.zeros(out_features)
    return w, b, rng


def init_embedding_params(rng, vocab_size, d_model):
    """Initialize embedding parameters."""
    rng, subkey = jr.split(rng)
    scale = jnp.sqrt(1.0 / d_model)
    embed = jr.normal(subkey, (vocab_size, d_model)) * scale
    return embed, rng


def init_layer_norm_params(d_model):
    """Initialize layer norm parameters."""
    return jnp.ones(d_model), jnp.zeros(d_model)


def init_model_params(rng, config: Config):
    """Initialize all model parameters."""
    params = {}
    
    # Embedding
    params['embedding'], rng = init_embedding_params(rng, config.vocab_size, config.d_model)
    
    # xLSTM layers (alternating sLSTM and mLSTM)
    params['layers'] = []
    for layer_idx in range(config.num_layers):
        layer_params = {}
        
        # Determine layer type: sLSTM at even indices, mLSTM at odd
        is_slstm = (layer_idx % 2 == 0)
        
        if is_slstm:
            # sLSTM parameters
            layer_params['w_i'], layer_params['b_i'], rng = init_dense_params(rng, config.d_model, config.d_model)
            layer_params['w_f'], layer_params['b_f'], rng = init_dense_params(rng, config.d_model, config.d_model)
            layer_params['w_g'], layer_params['b_g'], rng = init_dense_params(rng, config.d_model, config.d_model)
            layer_params['exp_bias_i'] = jnp.zeros(config.d_model)
            layer_params['exp_bias_f'] = jnp.zeros(config.d_model)
        else:
            # mLSTM parameters
            layer_params['w_i'], layer_params['b_i'], rng = init_dense_params(rng, config.d_model, config.d_model)
            layer_params['w_f'], layer_params['b_f'], rng = init_dense_params(rng, config.d_model, config.d_model)
            layer_params['w_q'], layer_params['b_q'], rng = init_dense_params(rng, config.d_model, config.d_model)
            layer_params['w_k'], layer_params['b_k'], rng = init_dense_params(rng, config.d_model, config.d_model)
            layer_params['w_v'], layer_params['b_v'], rng = init_dense_params(rng, config.d_model, config.d_model)
            layer_params['exp_bias_i'] = jnp.zeros(config.d_model)
            layer_params['exp_bias_f'] = jnp.zeros(config.d_model)
        
        # Feedforward network
        layer_params['ff_w1'], layer_params['ff_b1'], rng = init_dense_params(rng, config.d_model, config.d_ff)
        layer_params['ff_w2'], layer_params['ff_b2'], rng = init_dense_params(rng, config.d_ff, config.d_model)
        
        # Layer norms
        layer_params['ln1_scale'], layer_params['ln1_bias'] = init_layer_norm_params(config.d_model)
        layer_params['ln2_scale'], layer_params['ln2_bias'] = init_layer_norm_params(config.d_model)
        
        params['layers'].append(layer_params)
    
    # Final layer norm
    params['ln_final_scale'], params['ln_final_bias'] = init_layer_norm_params(config.d_model)
    
    # Output projection
    params['output_w'], params['output_b'], rng = init_dense_params(rng, config.d_model, config.vocab_size)
    
    return params, rng


# ============================================================================
# Activation Functions
# ============================================================================

def layer_norm(x, scale, bias, eps=1e-6):
    """RMSNorm - Root Mean Square Layer Normalization."""
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * scale + bias


def dense(x, w, b):
    """Dense layer."""
    return jnp.dot(x, w) + b


def dropout(x, rng, rate=0.1, training=True):
    """Dropout."""
    if not training or rate == 0.0:
        return x, rng
    keep_prob = 1.0 - rate
    rng, subkey = jr.split(rng)
    mask = jr.bernoulli(subkey, keep_prob, x.shape)
    return x * mask / keep_prob, rng


def exp_gate(x, bias, eps=1e-6):
    """Exponential gating: exp(x + bias) / ||exp(x + bias)||"""
    exp_x = jnp.exp(x + bias)
    norm = jnp.sqrt(jnp.sum(exp_x**2, axis=-1, keepdims=True) + eps)
    return exp_x / norm


# ============================================================================
# xLSTM Cells
# ============================================================================

def slstm_cell(x, h_prev, c_prev, params):
    """Scalar LSTM cell with exponential gating.
    
    Equations:
        i_t = exgate(W_i x_t + b_i + bias_i)
        f_t = exgate(W_f x_t + b_f + bias_f)
        g_t = tanh(W_g x_t + b_g)
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = tanh(c_t)
    """
    # Input and forget gates with exponential gating
    i = exp_gate(dense(x, params['w_i'], params['b_i']), params['exp_bias_i'])
    f = exp_gate(dense(x, params['w_f'], params['b_f']), params['exp_bias_f'])
    
    # Cell candidate
    g = jnp.tanh(dense(x, params['w_g'], params['b_g']))
    
    # Cell state update
    c = f * c_prev + i * g
    
    # Hidden state
    h = jnp.tanh(c)
    
    return h, c


def mlstm_cell(x, h_prev, m_prev, params, d_model):
    """Matrix LSTM cell with covariance update.
    
    Equations:
        i_t = exgate(W_i x_t + b_i)
        f_t = exgate(W_f x_t + b_f)
        Q_t = W_q x_t + b_q
        K_t = W_k x_t + b_k
        V_t = W_v x_t + b_v
        M_t = f_t ⊙ M_{t-1} + i_t ⊙ (V_t ⊗ K_t^T)
        h_t = tanh(M_t x_t)
    """
    batch_size = x.shape[0]
    
    # Input and forget gates (per-dimension gates)
    i = exp_gate(dense(x, params['w_i'], params['b_i']), params['exp_bias_i'])  # (batch, d_model)
    f = exp_gate(dense(x, params['w_f'], params['b_f']), params['exp_bias_f'])  # (batch, d_model)
    
    # Query, Key, Value projections
    q = dense(x, params['w_q'], params['b_q'])  # (batch, d_model)
    k = dense(x, params['w_k'], params['b_k'])  # (batch, d_model)
    v = dense(x, params['w_v'], params['b_v'])  # (batch, d_model)
    
    # Memory update: outer product V ⊗ K^T
    m_update = jnp.einsum('bi,bj->bij', v, k) / (d_model + 1e-6)  # (batch, d_model, d_model)
    
    # Update matrix memory with per-dimension gates
    m = f[:, :, None] * m_prev + i[:, :, None] * m_update  # (batch, d_model, d_model)
    
    # Apply memory to input: for each dimension, apply its memory matrix
    h_candidate = jnp.einsum('bij,bj->bi', m, x)  # (batch, d_model)
    h = jnp.tanh(h_candidate)
    
    return h, m


def xlstm_layer(x, state_prev, layer_params, config: Config, rng, training=True):
    """Single xLSTM layer (sLSTM or mLSTM + FFN with residuals)."""
    batch_size, seq_len, d_model = x.shape
    
    # Infer layer type from parameters: if 'w_g' exists, it's sLSTM
    is_slstm = 'w_g' in layer_params
    
    # Pre-norm
    x_norm = layer_norm(x, layer_params['ln1_scale'], layer_params['ln1_bias'])
    
    # Initialize LSTM state
    if state_prev is None:
        h = jnp.zeros((batch_size, d_model))
        if is_slstm:
            c = jnp.zeros((batch_size, d_model))
            state = (h, c)
        else:
            m = jnp.zeros((batch_size, d_model, d_model))
            state = (h, m)
    else:
        state = state_prev
        h = state[0]
        if is_slstm:
            c = state[1]
        else:
            m = state[1]
    
    # Process sequence through LSTM
    h_out_list = []
    for t in range(seq_len):
        x_t = x_norm[:, t, :]  # Shape should be (batch_size, d_model)
        if is_slstm:
            h, c = slstm_cell(x_t, h, c, layer_params)
            state = (h, c)
        else:
            h, m = mlstm_cell(x_t, h, m, layer_params, d_model)
            state = (h, m)
        # DEBUG: Print shape on first iteration
        if t == 0:
            print(f"DEBUG: h.shape = {h.shape}, expected (batch_size={batch_size}, d_model={d_model})")
        h_out_list.append(h)
    
    # Convert list to array and reshape from (seq_len, batch_size, d_model) to (batch_size, seq_len, d_model)
    lstm_out = jnp.array(h_out_list)  # Shape: (seq_len, batch_size, d_model)
    print(f"DEBUG: lstm_out.shape after jnp.array = {lstm_out.shape}")
    lstm_out = lstm_out.transpose(1, 0, 2)  # Shape: (batch_size, seq_len, d_model)
    
    # Dropout and residual
    lstm_out, rng = dropout(lstm_out, rng, config.dropout_rate, training)
    x = x + lstm_out
    
    # Feedforward with pre-norm
    x_norm = layer_norm(x, layer_params['ln2_scale'], layer_params['ln2_bias'])
    ff = dense(x_norm, layer_params['ff_w1'], layer_params['ff_b1'])
    ff = jax.nn.gelu(ff)
    ff, rng = dropout(ff, rng, config.dropout_rate, training)
    ff = dense(ff, layer_params['ff_w2'], layer_params['ff_b2'])
    ff, rng = dropout(ff, rng, config.dropout_rate, training)
    
    # Residual connection
    x = x + ff
    
    return x, state, rng


# ============================================================================
# Forward Pass
# ============================================================================

def model_forward(input_ids, params, config: Config, rng, training=True):
    """Forward pass through xLSTM."""
    B, T = input_ids.shape
    
    # Embedding
    x = params['embedding'][input_ids]
    
    # Initialize LSTM states
    states = [None] * config.num_layers
    
    # Forward through layers
    for layer_idx, layer_params in enumerate(params['layers']):
        x, states[layer_idx], rng = xlstm_layer(x, states[layer_idx], layer_params, config, rng, training)
    
    # Final layer norm
    x = layer_norm(x, params['ln_final_scale'], params['ln_final_bias'])
    
    # Output projection
    logits = dense(x, params['output_w'], params['output_b'])
    
    return logits, rng


# ============================================================================
# Loss and Optimization
# ============================================================================

def compute_loss(params, input_ids, config: Config, rng):
    """Cross-entropy loss."""
    logits, _ = model_forward(input_ids, params, config, rng, training=True)
    
    # Shift for next token prediction
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    
    B, T = targets.shape
    logits_flat = logits.reshape(B * T, config.vocab_size)
    targets_flat = targets.reshape(B * T)
    
    # One-hot encode
    targets_onehot = jax.nn.one_hot(targets_flat, config.vocab_size)
    
    # Cross-entropy
    log_softmax = jax.nn.log_softmax(logits_flat)
    loss = -jnp.sum(targets_onehot * log_softmax) / (B * T)
    
    return loss


def update_step(params, input_ids, config: Config, rng, learning_rate):
    """Single training step."""
    rng, subkey = jr.split(rng)
    
    # Compute gradients
    loss, grads = jax.value_and_grad(compute_loss)(params, input_ids, config, subkey)
    
    # Gradient clipping
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    grad_scale = jnp.minimum(1.0, config.grad_clip_norm / (grad_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * grad_scale, grads)
    
    # SGD update
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    
    return params, loss, rng


# ============================================================================
# Training
# ============================================================================

def train_model(tokens, config: Config, num_epochs: int = 3, seq_len: int = 256):
    """Train the model."""
    print(f"Training xLSTM on {len(tokens)} tokens")
    
    rng = jr.PRNGKey(42)
    params, rng = init_model_params(rng, config)
    print(f"Model initialized with {config.num_layers} layers (sLSTM/mLSTM alternating)")
    
    num_tokens = len(tokens) - 1
    num_batches = num_tokens // (config.batch_size * seq_len)
    
    if num_batches == 0:
        seq_len = max(16, min(seq_len, len(tokens) // (config.batch_size * 2)))
        num_batches = max(1, num_tokens // (config.batch_size * seq_len))
    
    print(f"Batch size: {config.batch_size}, Seq len: {seq_len}, Num batches: {num_batches}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx in pbar:
            start_idx = batch_idx * config.batch_size * (seq_len + 1)
            end_idx = start_idx + config.batch_size * (seq_len + 1)
            
            if end_idx > len(tokens):
                break
            
            batch_data = tokens[start_idx:end_idx]
            batch_data = batch_data.reshape(config.batch_size, seq_len + 1)
            batch_input = batch_data[:, :-1]
            batch_input = jnp.array(batch_input, dtype=jnp.int32)
            
            params, loss, rng = update_step(params, batch_input, config, rng, config.learning_rate)
            epoch_loss += loss
            pbar.set_postfix({"loss": f"{float(loss):.4f}"})
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    return params, rng


# ============================================================================
# Inference
# ============================================================================

def sample_next_token(logits, config: Config, rng):
    """Sample next token."""
    logits = logits[0, -1, :]
    logits = logits / config.temperature
    probs = jax.nn.softmax(logits)
    
    # Top-p sampling
    sorted_probs = jnp.sort(probs)[::-1]
    cumsum_probs = jnp.cumsum(sorted_probs)
    cutoff_idx = jnp.searchsorted(cumsum_probs, config.top_p)
    cutoff_idx = jnp.minimum(cutoff_idx, len(probs) - 1)
    cutoff_logit = sorted_probs[cutoff_idx]
    
    probs = jnp.where(probs >= cutoff_logit, probs, 0.0)
    probs = probs / (jnp.sum(probs) + 1e-10)
    
    # Top-k
    top_k_probs, top_k_indices = jax.lax.top_k(probs, min(config.top_k, len(probs)))
    top_k_probs = top_k_probs / jnp.sum(top_k_probs)
    
    rng, subkey = jr.split(rng)
    token = jr.choice(subkey, top_k_indices, p=top_k_probs)
    
    return int(token), rng


def generate(params, initial_tokens, config: Config, max_tokens: int = 256, rng=None):
    """Generate text."""
    if rng is None:
        rng = jr.PRNGKey(0)
    
    generated = list(initial_tokens)
    
    for i in tqdm(range(max_tokens), desc="Generating"):
        current_seq = generated[-config.max_seq_len:]
        current_tokens = jnp.array([current_seq], dtype=jnp.int32)
        
        logits, rng = model_forward(current_tokens, params, config, rng, training=False)
        token, rng = sample_next_token(logits, config, rng)
        generated.append(token)
    
    return np.array(generated, dtype=np.uint8)


# ============================================================================
# Main
# ============================================================================

def main():
    """Main script."""
    config = Config(
        vocab_size=256,
        d_model=256,
        d_ff=1024,
        num_heads=8,
        num_layers=4,
        batch_size=8,
        learning_rate=1e-4,
        max_seq_len=256
    )
    
    tokenizer = Tokenizer(TokenizerType.BYTE)
    
    filepath = "datasets/quran-uthmani.txt"
    try:
        tokens = tokenizer.tokenize_file(filepath)
        print(f"✓ Loaded {len(tokens)} tokens from {filepath}")
    except FileNotFoundError:
        print(f"✗ File {filepath} not found")
        return
    
    print("\n=== Training xLSTM (sLSTM + mLSTM) ===")
    params, rng = train_model(tokens, config, num_epochs=2, seq_len=128)
    
    print("\n=== Inference ===")
    
    initial_text = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
    
    try:
        initial_tokens = tokenizer.encode(initial_text)
        print(f"Initial text: {initial_text}")
        print(f"Initial tokens: {list(initial_tokens[:20])}")
    except Exception as e:
        print(f"Error: {e}")
        initial_tokens = np.array([0, 1, 2, 3], dtype=np.uint8)
    
    generated_tokens = generate(params, initial_tokens, config, max_tokens=512, rng=rng)
    
    try:
        generated_text = tokenizer.decode(generated_tokens)
        print(f"\n✓ Generated text (first 300 chars):")
        print(generated_text[:300])
    except Exception as e:
        print(f"× Error decoding: {e}")


if __name__ == "__main__":
    main()
