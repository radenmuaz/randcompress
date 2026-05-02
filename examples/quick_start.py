#!/usr/bin/env python3
"""
Quick Start Guide for xLSTM Seed-Based Compression V1

This script demonstrates all core features of the compression algorithm.
"""

import sys
sys.path.insert(0, 'examples')

import numpy as np
import randcompress_v1 as rc
from pathlib import Path


def example_1_basic_compression():
    """Example 1: Basic compression of small data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Compression")
    print("=" * 70)
    
    # Create simple test data
    data = np.array([72, 101, 108, 108, 111], dtype=np.uint8)  # "Hello"
    print(f"\nOriginal data: {data} (as text: {''.join(chr(b) for b in data)})")
    
    # Default configuration
    config = rc.Config()
    print(f"Using config: d_model={config.d_model}, num_layers={config.num_layers}")
    
    # Compress
    print("\nCompressing...")
    state = rc.compress_data(data, seed=42, config=config)
    
    print(f"\n✓ Compression complete!")
    print(f"  - Segments: {len(state.segments)}")
    print(f"  - Seed: {state.seed}")
    
    return state


def example_2_custom_config():
    """Example 2: Compression with custom configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 70)
    
    data = np.random.randint(0, 256, size=50, dtype=np.uint8)
    
    # Custom configuration for faster processing
    custom_config = rc.Config(
        d_model=128,              # Smaller model
        num_layers=2,             # Fewer layers
        loss_threshold=1.0,       # Higher threshold (fewer segments)
        max_training_steps=20,    # Fewer training steps
    )
    
    print(f"\nCustom config:")
    print(f"  - d_model: {custom_config.d_model}")
    print(f"  - num_layers: {custom_config.num_layers}")
    print(f"  - loss_threshold: {custom_config.loss_threshold}")
    print(f"  - max_training_steps: {custom_config.max_training_steps}")
    
    print(f"\nData: {len(data)} random bytes")
    
    state = rc.compress_data(data, seed=123, config=custom_config)
    
    print(f"\n✓ Result: {len(state.segments)} segments created")
    return state


def example_3_serialization():
    """Example 3: Save and load compression state."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Save and Load")
    print("=" * 70)
    
    # Create and compress
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    config = rc.Config(d_model=64, num_layers=1, max_training_steps=5)
    
    print(f"\nCompressing {len(data)} bytes...")
    state = rc.compress_data(data, seed=99, config=config)
    
    # Save
    save_path = "/tmp/compression_example.pkl"
    print(f"\nSaving to: {save_path}")
    state.save(save_path)
    
    # Load
    print(f"Loading from: {save_path}")
    loaded_state = rc.CompressionState.load(save_path)
    
    print(f"\n✓ Verification:")
    print(f"  - Original seed: {state.seed}")
    print(f"  - Loaded seed: {loaded_state.seed}")
    print(f"  - Seeds match: {state.seed == loaded_state.seed}")
    print(f"  - Segments match: {len(state.segments) == len(loaded_state.segments)}")
    
    return loaded_state


def example_4_decompression():
    """Example 4: Decompress and verify."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Decompression")
    print("=" * 70)
    
    # Original data
    original = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
    print(f"\nOriginal data: {original}")
    
    # Compress
    config = rc.Config(d_model=64, num_layers=1, max_training_steps=3)
    print(f"Compressing with d_model={config.d_model}...")
    state = rc.compress_data(original, seed=77, config=config)
    
    # Decompress
    print(f"\nDecompressing {len(state.segments)} segments...")
    reconstructed = rc.decompress_data(state)
    
    print(f"\n✓ Decompression complete!")
    print(f"  - Original size: {len(original)}")
    print(f"  - Reconstructed size: {len(reconstructed)}")
    print(f"  - Reconstructed data: {reconstructed}")
    
    return reconstructed


def example_5_compression_ratio():
    """Example 5: Measure compression ratio on different data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Compression Ratio Analysis")
    print("=" * 70)
    
    def analyze_data(name, data, seed, config):
        print(f"\n{name}:")
        print(f"  Input size: {len(data)} bytes")
        
        state = rc.compress_data(data, seed=seed, config=config)
        
        # Estimate stored size
        seed_size = 8
        config_size = 100
        segment_size = len(state.segments) * (
            8 +  # indices
            (config.d_model * 256 * 4) +  # weights
            (256 * 4)  # bias
        )
        total_size = seed_size + config_size + segment_size
        ratio = (total_size / len(data)) * 100
        
        print(f"  Stored size: ~{total_size} bytes")
        print(f"  Compression ratio: {ratio:.1f}%")
        print(f"  Segments: {len(state.segments)}")
        
        return ratio
    
    config = rc.Config(
        d_model=128,
        num_layers=2,
        loss_threshold=0.8,
        max_training_steps=10,
    )
    
    # Test different data types
    random_data = np.random.randint(0, 256, size=100, dtype=np.uint8)
    analyze_data("Random data (100 bytes)", random_data, 42, config)
    
    repetitive_data = np.tile(np.array([65, 66, 67], dtype=np.uint8), 33)  # "ABCABCABC..."
    analyze_data("Repetitive data (100 bytes)", repetitive_data, 43, config)
    
    print("\n" + "=" * 70)
    print("Note: Compression ratio depends on model training")
    print("Pre-trained models would achieve much better ratios")
    print("=" * 70)


def example_6_real_dataset():
    """Example 6: Compress real dataset if available."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Real Dataset Compression")
    print("=" * 70)
    
    dataset_paths = [
        Path("datasets/quran-uthmani-numbered.txt"),
        Path("datasets/quran-uthmani.txt"),
    ]
    
    dataset_found = None
    for path in dataset_paths:
        if path.exists():
            dataset_found = path
            break
    
    if dataset_found:
        print(f"\nFound dataset: {dataset_found}")
        
        with open(dataset_found, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Use first 500 bytes for demo
        data = data[:500]
        print(f"Using first {len(data)} bytes for demo")
        
        config = rc.Config(
            d_model=256,
            num_layers=4,
            loss_threshold=0.5,
            max_training_steps=50,
        )
        
        print("\nCompressing real data...")
        state = rc.compress_data(data, seed=111, config=config)
        
        print(f"\n✓ Real dataset compression complete!")
        print(f"  - Original: {len(data)} bytes")
        print(f"  - Segments: {len(state.segments)}")
        
    else:
        print("\nNo dataset found in datasets/ directory")
        print("Skipping real dataset example")


def main():
    """Run all examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     xLSTM Seed-Based Compression - Quick Start Examples              ║
║                                                                      ║
║  This script demonstrates all core features of the V1 implementation ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run examples
        example_1_basic_compression()
        example_2_custom_config()
        example_3_serialization()
        example_4_decompression()
        example_5_compression_ratio()
        example_6_real_dataset()
        
        print("\n" + "=" * 70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("""
Next steps:
  1. Read COMPRESSION_IMPLEMENTATION_SUMMARY.md for detailed info
  2. Check examples/COMPRESSION_README.md for full documentation
  3. Experiment with different configurations
  4. Try compressing your own data
  5. Pre-train the xLSTM for better compression ratios

For more info:
  - Compression algorithm: randcompress_v1.py
  - Architecture details: examples/xlstm_jax.py
  - Math formulation: examples/XLSTM_EQUATIONS.md
        """)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
