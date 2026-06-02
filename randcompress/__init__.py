"""randcompress — neural compression via memorization."""

__version__ = "0.1.0"

from .config import ModelConfig, TrainConfig, parse_configs
from .models import get_model, MODEL_REGISTRY
from .train import CurriculumTrainer
from .compress import encode
from .decompress import decode
from .checkpoint import save_checkpoint, load_checkpoint, save_bundle, load_bundle

__all__ = [
    "ModelConfig", "TrainConfig", "parse_configs",
    "get_model", "MODEL_REGISTRY",
    "CurriculumTrainer",
    "encode", "decode",
    "save_checkpoint", "load_checkpoint", "save_bundle", "load_bundle",
]
