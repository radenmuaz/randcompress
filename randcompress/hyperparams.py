"""Hyperparameter configurations for causal transformer."""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for causal transformer model."""
    
    # Model dimensions
    vocab_size: int = 256  # For byte tokenization
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 4096
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 4000
    grad_clip_norm: float = 1.0
    
    # Inference
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    def update(self, **kwargs):
        """Update config with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
