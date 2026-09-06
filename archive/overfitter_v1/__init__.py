"""overfitter — overfit a single file with SummTransformer, compress/decompress via range coding."""

__version__ = "0.1.0"

from .config import TrainConfig, parse_configs
from .summformer import Config as ModelConfig
from .summformer import SummTransformer
from .train import train_overfit
from .compress import encode
from .decompress import decode
from .checkpoint import save_model, load_model, save_bundle, load_bundle

__all__ = [
    "ModelConfig", "TrainConfig", "parse_configs",
    "SummTransformer",
    "train_overfit",
    "encode", "decode",
    "save_model", "load_model", "save_bundle", "load_bundle",
]
