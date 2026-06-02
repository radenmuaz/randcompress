import numpy as np


def bytes_to_tokens(raw: np.ndarray, bits: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.uint8)
    if bits == 8:
        return raw.astype(np.int32)
    if bits == 4:
        hi = ((raw >> 4) & 0xF).astype(np.int32)
        lo = (raw & 0xF).astype(np.int32)
        return np.stack([hi, lo], axis=-1).ravel()
    if bits == 1:
        return np.unpackbits(raw).astype(np.int32)
    raise ValueError(f"bits must be 1, 4, or 8; got {bits}")


def tokens_to_bytes(toks: np.ndarray, bits: int) -> np.ndarray:
    toks = np.asarray(toks)
    if bits == 8:
        return toks.astype(np.uint8)
    if bits == 4:
        hi = (toks[0::2].astype(np.uint8) & 0xF) << 4
        lo = toks[1::2].astype(np.uint8) & 0xF
        return hi | lo
    if bits == 1:
        return np.packbits(toks.astype(np.uint8))
    raise ValueError(f"bits must be 1, 4, or 8; got {bits}")


class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text) -> np.ndarray:
        if isinstance(text, str):
            text = text.encode("utf-8")
        return np.frombuffer(text, dtype=np.uint8).copy()

    @staticmethod
    def decode(tokens) -> str:
        return bytes(np.asarray(tokens, dtype=np.uint8)).decode("utf-8", errors="replace")


def load_bytes(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8).copy()
