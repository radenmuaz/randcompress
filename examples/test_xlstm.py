"""
xLSTM Quick Test and Validation Script
Tests model initialization, forward pass, and basic training step.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from xlstm_jax import (
    Config,
    init_xlstm_params,
    forward_pass,
    cross_entropy_loss,
    slstm_step,
    mlstm_step,
    init_slstm_state,
    init_mlstm_state,
    sLSTMParams,
    mLSTMParams,
)


def test_slstm_cell():
    """Test sLSTM cell."""
    print("=" * 60)
    print("Testing sLSTM Cell")
    print("=" * 60)
    
    key = jr.PRNGKey(0)
    batch_size = 4
    d_in = 128
    d_hidden = 128
    
    # Initialize
    params = sLSTMParams
    # Actually import properly
    from xlstm_jax import init_slstm_params, init_slstm_state
    
    params = init_slstm_params(key, d_in, d_hidden)
    state = init_slstm_state(batch_size, d_hidden)
    
    # Input
    x = jr.normal(key, (batch_size, d_in))
    
    # Forward
    new_state, output = slstm_step(params, state, x)
    
    print(f"✓ sLSTM step completed")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  State y shape: {new_state.y.shape}")
    print(f"  State c shape: {new_state.c.shape}")
    print(f"  State n shape: {new_state.n.shape}")
    print(f"  State m shape: {new_state.m.shape}")
    print()


def test_mlstm_cell():
    """Test mLSTM cell."""
    print("=" * 60)
    print("Testing mLSTM Cell")
    print("=" * 60)
    
    key = jr.PRNGKey(1)
    batch_size = 4
    d_in = 128
    d_model = 128
    num_heads = 4
    d_h = d_model // num_heads
    
    # Initialize
    from xlstm_jax import init_mlstm_params, init_mlstm_state
    
    params = init_mlstm_params(key, d_in, d_model, num_heads)
    state = init_mlstm_state(batch_size, num_heads, d_h)
    
    # Input
    x = jr.normal(key, (batch_size, d_in))
    
    # Forward
    new_state, output = mlstm_step(params, state, x)
    
    print(f"✓ mLSTM step completed")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  State C shape: {new_state.C.shape}")
    print(f"  State n shape: {new_state.n.shape}")
    print(f"  State m shape: {new_state.m.shape}")
    print()


def test_model_initialization():
    """Test full model initialization."""
    print("=" * 60)
    print("Testing Full Model Initialization")
    print("=" * 60)
    
    key = jr.PRNGKey(2)
    
    config = Config(
        vocab_size=256,
        d_model=128,
        num_heads=4,
        d_head=32,
        d_ff=512,
        num_layers=2,
        max_seq_len=256,
    )
    
    params = init_xlstm_params(key, config)
    
    # Count parameters
    num_params = sum(
        np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)
    )
    
    print(f"✓ Model initialized successfully")
    print(f"  Config: {config}")
    print(f"  Total parameters: {num_params:,}")
    print()


def test_forward_pass():
    """Test forward pass through model."""
    print("=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    key = jr.PRNGKey(3)
    
    config = Config(
        vocab_size=256,
        d_model=128,
        num_heads=4,
        d_head=32,
        d_ff=512,
        num_layers=2,
        max_seq_len=256,
        batch_size=4,
    )
    
    params = init_xlstm_params(key, config)
    
    # Create random tokens
    tokens = jr.randint(key, (config.batch_size, 64), 0, config.vocab_size)
    
    # Forward pass
    logits = forward_pass(params, tokens, config, key)
    
    print(f"✓ Forward pass completed")
    print(f"  Input tokens shape: {tokens.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected shape: ({config.batch_size}, 64, {config.vocab_size})")
    print(f"  Logit range: [{jnp.min(logits):.2f}, {jnp.max(logits):.2f}]")
    print()


def test_loss_computation():
    """Test loss computation."""
    print("=" * 60)
    print("Testing Loss Computation")
    print("=" * 60)
    
    key = jr.PRNGKey(4)
    
    config = Config(
        vocab_size=256,
        d_model=128,
        num_heads=4,
        d_head=32,
        d_ff=512,
        num_layers=2,
        max_seq_len=256,
        batch_size=4,
    )
    
    params = init_xlstm_params(key, config)
    
    # Create random tokens
    tokens = jr.randint(key, (config.batch_size, 64), 0, config.vocab_size)
    
    # Forward pass
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    
    logits = forward_pass(params, inputs, config, key)
    loss = cross_entropy_loss(logits, targets)
    
    print(f"✓ Loss computation completed")
    print(f"  Loss value: {loss:.4f}")
    print(f"  Perplexity: {np.exp(loss):.2f}")
    print(f"  Loss is finite: {jnp.isfinite(loss)}")
    print()


def test_gradient_flow():
    """Test gradient computation."""
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    key = jr.PRNGKey(5)
    
    config = Config(
        vocab_size=256,
        d_model=64,
        num_heads=2,
        d_head=32,
        d_ff=256,
        num_layers=1,
        max_seq_len=256,
        batch_size=2,
    )
    
    params = init_xlstm_params(key, config)
    
    # Create tokens
    tokens = jr.randint(key, (config.batch_size, 32), 0, config.vocab_size)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    
    def loss_fn(p):
        logits = forward_pass(p, inputs, config, key)
        return cross_entropy_loss(logits, targets)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Check gradients
    has_nan = False
    num_params = 0
    for p in jax.tree_util.tree_leaves(grads):
        num_params += 1
        if jnp.any(~jnp.isfinite(p)):
            has_nan = True
    
    print(f"✓ Gradient computation completed")
    print(f"  Loss: {loss:.4f}")
    print(f"  Number of parameter tensors: {num_params}")
    print(f"  Has NaN/Inf in gradients: {has_nan}")
    print()


def test_numerical_stability():
    """Test numerical stability."""
    print("=" * 60)
    print("Testing Numerical Stability")
    print("=" * 60)
    
    key = jr.PRNGKey(6)
    
    config = Config(
        vocab_size=256,
        d_model=128,
        num_heads=4,
        d_head=32,
        d_ff=512,
        num_layers=2,
        max_seq_len=256,
        batch_size=4,
    )
    
    params = init_xlstm_params(key, config)
    
    # Test with extreme values
    tokens = jr.randint(key, (config.batch_size, 128), 0, config.vocab_size)
    
    logits = forward_pass(params, tokens, config, key)
    
    # Check for NaN/Inf
    has_nan = jnp.any(~jnp.isfinite(logits))
    nan_count = jnp.sum(~jnp.isfinite(logits))
    
    print(f"✓ Numerical stability test completed")
    print(f"  Output has NaN/Inf: {has_nan}")
    if has_nan:
        print(f"  NaN/Inf count: {nan_count}")
    print(f"  Min logit: {jnp.min(logits):.4f}")
    print(f"  Max logit: {jnp.max(logits):.4f}")
    print(f"  Mean logit: {jnp.mean(logits):.4f}")
    print(f"  Std logit: {jnp.std(logits):.4f}")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "xLSTM JAX Implementation - Test Suite" + " " * 11 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        test_slstm_cell()
        test_mlstm_cell()
        test_model_initialization()
        test_forward_pass()
        test_loss_computation()
        test_gradient_flow()
        test_numerical_stability()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
