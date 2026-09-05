"""
Range codec — ctypes binding to rc_codec.c.

Compiles rc_codec.c once per process (cached in /tmp). Provides:
  quantize_cdf(logits_1d, M) -> (V+1,) int32  cumulative freq array
  rc_encode(symbols, cdfs)   -> bytes
  rc_decode(stream, cdfs)    -> (T,) int32
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile

import numpy as np

RC_PREC = 16
RC_M    = 1 << RC_PREC   # 65536

# ── Locate rc_codec.c ─────────────────────────────────────────────────────────

def _find_rc_codec_c() -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "rc_codec.c"),
        os.path.join(os.path.dirname(__file__), "rc_codec.c"),
        "rc_codec.c",
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "rc_codec.c not found. Expected at repo root or next to codec.py."
    )


# ── Compile + cache ───────────────────────────────────────────────────────────

_lib_cache: ctypes.CDLL | None = None

def _get_rc_clib() -> ctypes.CDLL:
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache

    src = _find_rc_codec_c()
    with open(src, "rb") as f:
        h = hashlib.sha1(f.read()).hexdigest()[:12]

    so_path = os.path.join(tempfile.gettempdir(), f"rc_codec_{h}.so")
    if not os.path.exists(so_path):
        subprocess.check_call(
            ["cc", "-O2", "-shared", "-fPIC", src, "-o", so_path],
            stderr=subprocess.DEVNULL,
        )

    lib = ctypes.CDLL(so_path)

    c_int32_p  = ctypes.POINTER(ctypes.c_int32)
    c_uint8_p  = ctypes.POINTER(ctypes.c_uint8)

    lib.rc_encode.restype  = ctypes.c_size_t
    lib.rc_encode.argtypes = [
        c_int32_p, c_int32_p, ctypes.c_size_t, ctypes.c_int32,
        c_uint8_p, ctypes.c_size_t,
    ]
    lib.rc_decode.restype  = ctypes.c_int
    lib.rc_decode.argtypes = [
        c_uint8_p, ctypes.c_size_t,
        c_int32_p, ctypes.c_size_t, ctypes.c_int32,
        c_int32_p,
    ]

    _lib_cache = lib
    return lib


# ── CDF helpers ───────────────────────────────────────────────────────────────

def quantize_cdf(logits_1d: np.ndarray) -> np.ndarray:
    """Softmax → integer CDF summing to RC_M=65536. Returns (V+1,) int32."""
    logits = logits_1d.astype(np.float64)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    freqs = np.maximum(1, np.round(probs * RC_M).astype(np.int64))
    deficit = RC_M - int(freqs.sum())
    freqs[int(np.argmax(freqs))] += deficit

    cumfreqs = np.zeros(len(freqs) + 1, dtype=np.int32)
    cumfreqs[1:] = np.cumsum(freqs).astype(np.int32)
    cumfreqs[-1] = RC_M
    return cumfreqs


# ── Encode / Decode ───────────────────────────────────────────────────────────

def rc_encode(symbols: np.ndarray, cdfs: np.ndarray) -> bytes:
    """
    symbols: (T,) int32   — token indices
    cdfs:    (T, V+1) int32 — cumulative freq arrays from quantize_cdf
    Returns: bytes (the compressed stream)
    """
    lib = _get_rc_clib()
    T, Vp1 = cdfs.shape
    V = Vp1 - 1

    syms   = np.ascontiguousarray(symbols, dtype=np.int32)
    cdfs_c = np.ascontiguousarray(cdfs, dtype=np.int32)

    out_cap = T * 3 + 64
    out_buf = np.zeros(out_cap, dtype=np.uint8)

    n = lib.rc_encode(
        cdfs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        syms.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(T),
        ctypes.c_int32(V),
        out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(out_cap),
    )
    if n == ctypes.c_size_t(-1).value:
        raise RuntimeError("rc_encode: output buffer overflow")
    return bytes(out_buf[:n])


def rc_decode(stream: bytes, cdfs: np.ndarray) -> np.ndarray:
    """
    stream: bytes
    cdfs:   (T, V+1) int32
    Returns: (T,) int32 decoded symbols
    """
    lib = _get_rc_clib()
    T, Vp1 = cdfs.shape
    V = Vp1 - 1

    in_buf  = np.frombuffer(stream, dtype=np.uint8)
    cdfs_c  = np.ascontiguousarray(cdfs, dtype=np.int32)
    out_sym = np.zeros(T, dtype=np.int32)

    ret = lib.rc_decode(
        in_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(len(in_buf)),
        cdfs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(T),
        ctypes.c_int32(V),
        out_sym.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if ret != 0:
        raise RuntimeError(f"rc_decode failed with code {ret}")
    return out_sym
