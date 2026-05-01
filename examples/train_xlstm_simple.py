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

# Import xLSTM components
from xlstm_jax import (
    Config,
    xLSTMParams,
    init_xlstm_params,
    forward_pass,
    cross_entropy_loss,
    generate,
)


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
    params = jax.tree_map(lambda p, u: p + u, params, updates)
    
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
        use_slstm=True,
        use_mlstm=True,
        soft_cap=15.0,
        dropout_rate=0.1,
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-6,
        grad_clip_norm=1.0,
    )
    
    # ========== Data Loading ==========
    print("Loading dataset...")
    dataset_path = "../datasets/quran-uthmani.txt"
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found at {dataset_path}")
        print("Using synthetic data for demonstration...")
        # Create synthetic data
        tokens = np.random.randint(0, config.vocab_size, (10000,))
    else:
        tokens = load_dataset(dataset_path)
        print(f"Loaded {len(tokens)} tokens")
    
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
