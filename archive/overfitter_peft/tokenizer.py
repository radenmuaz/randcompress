"""Byte-only I/O — overfitter always uses vocab=256, no bit-packing."""
import numpy as np


def load_bytes(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8).copy()
