"""
Causal Transformer in pure JAX without Flax.
Single file with byte/bit/hex tokenization, training, and inference.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from typing import Dict, Tuple, NamedTuple
from enum import Enum
import time
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

class Config(NamedTuple):
    """Configuration for causal transformer model."""
    # Model dimensions
    vocab_size: int = 256  # For byte tokenization
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
    BYTE = "byte"       # 0-255
    BIT = "bit"         # 0-1
    HEX = "hex"         # 0-15


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
# Model Parameters
# ============================================================================

def init_dense_params(rng, in_features, out_features):
    """Initialize dense layer parameters."""
    rng, subkey = jr.split(rng)
    # He initialization
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
    
    # Transformer layers
    params['layers'] = []
    for layer_idx in range(config.num_layers):
        layer_params = {}
        
        # Multi-head attention
        head_dim = config.d_model // config.num_heads
        
        # Q, K, V projections
        layer_params['attn_q_w'], layer_params['attn_q_b'], rng = init_dense_params(
            rng, config.d_model, config.d_model
        )
        layer_params['attn_k_w'], layer_params['attn_k_b'], rng = init_dense_params(
            rng, config.d_model, config.d_model
        )
        layer_params['attn_v_w'], layer_params['attn_v_b'], rng = init_dense_params(
            rng, config.d_model, config.d_model
        )
        
        # Output projection
        layer_params['attn_out_w'], layer_params['attn_out_b'], rng = init_dense_params(
            rng, config.d_model, config.d_model
        )
        
        # Layer norm 1
        layer_params['ln1_scale'], layer_params['ln1_bias'] = init_layer_norm_params(config.d_model)
        
        # Feed-forward
        layer_params['ff_w1'], layer_params['ff_b1'], rng = init_dense_params(
            rng, config.d_model, config.d_ff
        )
        layer_params['ff_w2'], layer_params['ff_b2'], rng = init_dense_params(
            rng, config.d_ff, config.d_model
        )
        
        # Layer norm 2
        layer_params['ln2_scale'], layer_params['ln2_bias'] = init_layer_norm_params(config.d_model)
        
        params['layers'].append(layer_params)
    
    # Final layer norm
    params['ln_final_scale'], params['ln_final_bias'] = init_layer_norm_params(config.d_model)
    
    # Output projection
    params['output_w'], params['output_b'], rng = init_dense_params(
        rng, config.d_model, config.vocab_size
    )
    
    return params, rng


# ============================================================================
# Layer Operations
# ============================================================================

def layer_norm(x, scale, bias, eps=1e-6):
    """RMSNorm - Root Mean Square Layer Normalization."""
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    x_norm = x / rms
    return x_norm * scale + bias


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


def multi_head_attention(x, params, config: Config, mask, rng, training=True):
    """Multi-head self-attention."""
    B, T, D = x.shape
    num_heads = config.num_heads
    head_dim = D // num_heads
    
    # Project to Q, K, V
    Q = dense(x, params['attn_q_w'], params['attn_q_b'])
    K = dense(x, params['attn_k_w'], params['attn_k_b'])
    V = dense(x, params['attn_v_w'], params['attn_v_b'])
    
    # Reshape for multi-head
    Q = Q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    # Scaled dot-product attention
    scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
    
    # Apply causal mask
    scores = jnp.where(mask[None, None, :, :], scores, -1e10)
    
    # Softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Dropout
    attn_weights, rng = dropout(attn_weights, rng, config.dropout_rate, training)
    
    # Apply to values
    attn_out = jnp.matmul(attn_weights, V)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)
    
    # Output projection
    attn_out = dense(attn_out, params['attn_out_w'], params['attn_out_b'])
    attn_out, rng = dropout(attn_out, rng, config.dropout_rate, training)
    
    return attn_out, rng


def feed_forward(x, params, config: Config, rng, training=True):
    """Feed-forward network."""
    x = dense(x, params['ff_w1'], params['ff_b1'])
    x = jax.nn.gelu(x)
    x, rng = dropout(x, rng, config.dropout_rate, training)
    x = dense(x, params['ff_w2'], params['ff_b2'])
    x, rng = dropout(x, rng, config.dropout_rate, training)
    return x, rng


def transformer_block(x, block_params, config: Config, mask, rng, training=True):
    """Transformer encoder block with pre-norm."""
    # Pre-norm attention
    x_norm = layer_norm(x, block_params['ln1_scale'], block_params['ln1_bias'])
    attn_out, rng = multi_head_attention(x_norm, block_params, config, mask, rng, training)
    x = x + attn_out
    
    # Pre-norm feed-forward
    x_norm = layer_norm(x, block_params['ln2_scale'], block_params['ln2_bias'])
    ff_out, rng = feed_forward(x_norm, block_params, config, rng, training)
    x = x + ff_out
    
    return x, rng


# ============================================================================
# Forward Pass
# ============================================================================

def model_forward(input_ids, params, config: Config, rng, training=True):
    """Forward pass through the transformer."""
    B, T = input_ids.shape
    
    # Create causal mask
    causal_mask = jnp.tril(jnp.ones((T, T))) == 1
    
    # Embedding
    x = params['embedding'][input_ids]
    
    # Apply transformer blocks
    for layer_params in params['layers']:
        x, rng = transformer_block(x, layer_params, config, causal_mask, rng, training)
    
    # Final layer norm
    x = layer_norm(x, params['ln_final_scale'], params['ln_final_bias'])
    
    # Output projection
    logits = dense(x, params['output_w'], params['output_b'])
    
    return logits, rng


# ============================================================================
# Loss and Optimization
# ============================================================================

def compute_loss(params, input_ids, config: Config, rng):
    """Compute cross-entropy loss."""
    logits, _ = model_forward(input_ids, params, config, rng, training=True)
    
    # Shift for next token prediction
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    
    # Cross-entropy loss
    B, T = targets.shape
    logits_flat = logits.reshape(B * T, config.vocab_size)
    targets_flat = targets.reshape(B * T)
    
    # One-hot encode targets
    targets_onehot = jax.nn.one_hot(targets_flat, config.vocab_size)
    
    # Compute cross-entropy
    log_softmax = jax.nn.log_softmax(logits_flat)
    loss = -jnp.sum(targets_onehot * log_softmax) / (B * T)
    
    return loss


@jax.jit(static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def _jit_update_step(params, input_ids, config_d_model, config_vocab_size, config_batch_size, config_d_ff, config_num_heads, config_num_layers, config_dropout_rate, learning_rate, rng):
    """JIT-compiled update step."""
    # Reconstruct config from scalars for JIT compatibility
    config = Config(
        vocab_size=config_vocab_size,
        d_model=config_d_model,
        d_ff=config_d_ff,
        num_heads=config_num_heads,
        num_layers=config_num_layers,
        batch_size=config_batch_size,
        dropout_rate=config_dropout_rate,
        learning_rate=learning_rate
    )
    
    rng, subkey = jr.split(rng)
    
    # Compute gradients
    loss, grads = jax.value_and_grad(compute_loss)(params, input_ids, config, subkey)
    
    # Gradient clipping
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
    grad_scale = jnp.minimum(1.0, config.grad_clip_norm / (grad_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * grad_scale, grads)
    
    # Update parameters with simple SGD
    def update_param(p, g):
        return p - learning_rate * g
    
    params = jax.tree_util.tree_map(update_param, params, grads)
    
    return params, loss, rng


def update_step(params, input_ids, config: Config, rng, learning_rate):
    """Single training step."""
    return _jit_update_step(
        params, input_ids,
        config.d_model, config.vocab_size, config.batch_size,
        config.d_ff, config.num_heads, config.num_layers,
        config.dropout_rate, learning_rate, rng
    )


# ============================================================================
# Training
# ============================================================================

def train_model(tokens, config: Config, num_epochs: int = 3, seq_len: int = 256):
    """Train the model."""
    print(f"Training on {len(tokens)} tokens")
    
    # Initialize
    rng = jr.PRNGKey(42)
    params, rng = init_model_params(rng, config)
    print(f"Model initialized with {config.num_layers} layers")
    
    # Prepare data: create batches
    num_tokens = len(tokens) - 1
    num_batches = num_tokens // (config.batch_size * seq_len)
    
    if num_batches == 0:
        print("Not enough data for batching, using smaller seq_len")
        seq_len = max(16, min(seq_len, len(tokens) // (config.batch_size * 2)))
        num_batches = max(1, num_tokens // (config.batch_size * seq_len))
    
    print(f"Batch size: {config.batch_size}, Seq len: {seq_len}, Num batches: {num_batches}")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx in pbar:
            start_idx = batch_idx * config.batch_size * (seq_len + 1)
            end_idx = start_idx + config.batch_size * (seq_len + 1)
            
            if end_idx > len(tokens):
                break
            
            # Create batch
            batch_data = tokens[start_idx:end_idx]
            batch_data = batch_data.reshape(config.batch_size, seq_len + 1)
            batch_input = batch_data[:, :-1]
            
            batch_input = jnp.array(batch_input, dtype=jnp.int32)
            
            # Update
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
    """Sample next token using temperature, top-k, and top-p."""
    # Take last position
    logits = logits[0, -1, :]
    
    # Apply temperature
    logits = logits / config.temperature
    
    # Compute probabilities
    probs = jax.nn.softmax(logits)
    
    # Top-p (nucleus) sampling
    sorted_probs = jnp.sort(probs)[::-1]
    cumsum_probs = jnp.cumsum(sorted_probs)
    sorted_indices = jnp.argsort(probs)[::-1]
    
    cutoff_idx = jnp.searchsorted(cumsum_probs, config.top_p)
    cutoff_idx = jnp.minimum(cutoff_idx, len(probs) - 1)
    cutoff_logit = sorted_probs[cutoff_idx]
    
    probs = jnp.where(probs >= cutoff_logit, probs, 0.0)
    probs = probs / (jnp.sum(probs) + 1e-10)
    
    # Top-k
    top_k_probs, top_k_indices = jax.lax.top_k(probs, min(config.top_k, len(probs)))
    top_k_probs = top_k_probs / jnp.sum(top_k_probs)
    
    # Sample
    rng, subkey = jr.split(rng)
    token = jr.choice(subkey, top_k_indices, p=top_k_probs)
    
    return int(token), rng


def generate(params, initial_tokens, config: Config, max_tokens: int = 256, rng=None):
    """Generate text autoregressively."""
    if rng is None:
        rng = jr.PRNGKey(0)
    
    generated = list(initial_tokens)
    
    for i in tqdm(range(max_tokens), desc="Generating"):
        # Keep only last max_seq_len tokens
        current_seq = generated[-config.max_seq_len:]
        current_tokens = jnp.array([current_seq], dtype=jnp.int32)
        
        # Forward pass
        logits, rng = model_forward(current_tokens, params, config, rng, training=False)
        
        # Sample next token
        token, rng = sample_next_token(logits, config, rng)
        generated.append(token)
    
    return np.array(generated, dtype=np.uint8)


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training and inference script."""
    # Configuration
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
    
    # Tokenizer
    tokenizer = Tokenizer(TokenizerType.BYTE)
    
    # Load data
    filepath = "datasets/quran-uthmani.txt"
    try:
        tokens = tokenizer.tokenize_file(filepath)
        print(f"✓ Loaded {len(tokens)} tokens from {filepath}")
    except FileNotFoundError:
        print(f"✗ File {filepath} not found")
        return
    
    # Train
    print("\n=== Training ===")
    params, rng = train_model(tokens, config, num_epochs=2, seq_len=128)
    
    # Generate
    print("\n=== Inference ===")
    
    # Start with some initial text
    initial_text = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
    
    try:
        initial_tokens = tokenizer.encode(initial_text)
        print(f"Initial text: {initial_text}")
        print(f"Initial tokens: {list(initial_tokens)}")
    except Exception as e:
        print(f"Error encoding initial text: {e}")
        initial_tokens = np.array([0, 1, 2, 3], dtype=np.uint8)
    
    # Generate
    generated_tokens = generate(params, initial_tokens, config, max_tokens=512, rng=rng)
    
    # Decode
    try:
        generated_text = tokenizer.decode(generated_tokens)
        print(f"\n✓ Generated text (first 300 chars):")
        print(generated_text[:300])
    except Exception as e:
        print(f"× Error decoding: {e}")
        print(f"Generated tokens: {generated_tokens[:50]}")


if __name__ == "__main__":
    main()
