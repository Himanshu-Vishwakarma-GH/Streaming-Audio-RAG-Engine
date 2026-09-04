"""
simd_wrapper.py - High-Performance Python ctypes Wrapper for simd_engine.dll
PS-003: Speculative-Decoded Sub-250ms Streaming Audio RAG Engine

Exposes:
1. AVX2 SIMD Vector Search:
   - simd_init(num_vectors, dim)
   - simd_load_vectors(numpy_array)
   - simd_search_top1(query_vector) -> (best_idx, score)
   - simd_dot_product(vec_a, vec_b) -> float
   - simd_cleanup()
2. Lock-Free SPSC Circular Audio Ring Buffer:
   - ring_buffer_init(capacity_frames)
   - ring_buffer_push_frame(samples_320_int16) -> bool
   - ring_buffer_push(samples_int16_array) -> int (frames)
   - ring_buffer_push_bytes(pcm_bytes) -> int (frames)
   - ring_buffer_pop_frame() -> Optional[np.ndarray] (320 int16)
   - ring_buffer_pop(num_samples) -> np.ndarray
   - ring_buffer_pop_bytes(max_bytes) -> bytes
   - ring_buffer_available_read() -> int
   - ring_buffer_available_write() -> int
   - ring_buffer_clear()
   - ring_buffer_cleanup()
"""

import os
import sys
import ctypes
from ctypes import (
    c_int,
    c_float,
    c_size_t,
    c_uint8,
    c_int16,
    c_void_p,
    POINTER,
    byref
)
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np

# Audio constants
FRAME_DURATION_MS = 20
SAMPLE_RATE_HZ = 16000
FRAME_SAMPLES = int(SAMPLE_RATE_HZ * (FRAME_DURATION_MS / 1000.0))  # 320 samples
FRAME_BYTES = FRAME_SAMPLES * 2  # 640 bytes (16-bit PCM = 2 bytes)
DEFAULT_BUFFER_CAPACITY_FRAMES = 1024  # ~20.48s buffer

# Determine DLL path
def find_simd_dll() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "simd_engine.dll",
        Path.cwd() / "simd_engine.dll",
        Path(__file__).resolve().parent.parent / "simd_engine.dll",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"simd_engine.dll not found in candidate paths: {[str(c) for c in candidates]}"
    )


class SIMDLibrary:
    """Singleton-style loader and ctypes binder for simd_engine.dll."""

    _instance: Optional["SIMDLibrary"] = None

    def __init__(self, dll_path: Optional[Union[str, Path]] = None):
        if dll_path is None:
            dll_path = find_simd_dll()
        else:
            dll_path = Path(dll_path)
            if not dll_path.exists():
                raise FileNotFoundError(f"DLL not found: {dll_path}")

        self.dll_path = dll_path
        # Load DLL using ctypes
        self.lib = ctypes.CDLL(str(dll_path))
        self._bind_functions()

    @classmethod
    def get_instance(cls, dll_path: Optional[Union[str, Path]] = None) -> "SIMDLibrary":
        if cls._instance is None:
            cls._instance = cls(dll_path)
        return cls._instance

    def _bind_functions(self):
        # 1. avx2_dot_product / simd_dot_product
        self.lib.avx2_dot_product.argtypes = [
            POINTER(c_float),
            POINTER(c_float),
            c_int,
        ]
        self.lib.avx2_dot_product.restype = c_float

        if hasattr(self.lib, "simd_dot_product"):
            self.lib.simd_dot_product.argtypes = [
                POINTER(c_float),
                POINTER(c_float),
                c_int,
            ]
            self.lib.simd_dot_product.restype = c_float

        # 2. simd_init
        self.lib.simd_init.argtypes = [c_int, c_int]
        self.lib.simd_init.restype = c_int

        # 3. simd_load_vectors
        self.lib.simd_load_vectors.argtypes = [
            POINTER(c_float),
            c_int,
            c_int,
        ]
        self.lib.simd_load_vectors.restype = c_int

        # 4. simd_cleanup
        self.lib.simd_cleanup.argtypes = []
        self.lib.simd_cleanup.restype = c_int

        # 5. simd_get_num_vectors
        self.lib.simd_get_num_vectors.argtypes = []
        self.lib.simd_get_num_vectors.restype = c_int

        # 6. simd_get_dim
        self.lib.simd_get_dim.argtypes = []
        self.lib.simd_get_dim.restype = c_int

        # 7. simd_search_top1
        self.lib.simd_search_top1.argtypes = [
            POINTER(c_float),
            POINTER(c_int),
            POINTER(c_float),
        ]
        self.lib.simd_search_top1.restype = c_int

        # 8. Ring buffer global functions
        self.lib.ring_buffer_init.argtypes = [c_size_t]
        self.lib.ring_buffer_init.restype = c_int

        self.lib.ring_buffer_push_frame.argtypes = [POINTER(c_int16)]
        self.lib.ring_buffer_push_frame.restype = c_int

        self.lib.ring_buffer_push.argtypes = [POINTER(c_int16), c_size_t]
        self.lib.ring_buffer_push.restype = c_int

        self.lib.ring_buffer_push_bytes.argtypes = [POINTER(c_uint8), c_size_t]
        self.lib.ring_buffer_push_bytes.restype = c_int

        self.lib.ring_buffer_pop_frame.argtypes = [POINTER(c_int16)]
        self.lib.ring_buffer_pop_frame.restype = c_int

        self.lib.ring_buffer_pop.argtypes = [POINTER(c_int16), c_size_t]
        self.lib.ring_buffer_pop.restype = c_int

        self.lib.ring_buffer_pop_bytes.argtypes = [POINTER(c_uint8), c_size_t]
        self.lib.ring_buffer_pop_bytes.restype = c_int

        self.lib.ring_buffer_available_read.argtypes = []
        self.lib.ring_buffer_available_read.restype = c_size_t

        self.lib.ring_buffer_available_write.argtypes = []
        self.lib.ring_buffer_available_write.restype = c_size_t

        self.lib.ring_buffer_clear.argtypes = []
        self.lib.ring_buffer_clear.restype = None

        self.lib.ring_buffer_cleanup.argtypes = []
        self.lib.ring_buffer_cleanup.restype = None


class SIMDVectorEngine:
    """Pythonic object interface to the C++ AVX2 SIMD Vector Search Engine."""

    def __init__(self, dll_path: Optional[Union[str, Path]] = None):
        self._lib = SIMDLibrary.get_instance(dll_path).lib

    def init(self, num_vectors: int, dim: int) -> int:
        """Initializes aligned storage for num_vectors of dimensionality dim."""
        return self._lib.simd_init(int(num_vectors), int(dim))

    def load_vectors(self, vectors: np.ndarray) -> int:
        """Loads a 2D numpy float32 matrix into SIMD cache-aligned memory."""
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)

        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array (N, dim), got shape {vectors.shape}")

        num_vectors, dim = vectors.shape
        ptr = vectors.ctypes.data_as(POINTER(c_float))
        rc = self._lib.simd_load_vectors(ptr, int(num_vectors), int(dim))
        if rc != 0:
            raise RuntimeError(f"simd_load_vectors failed with error code {rc}")
        return rc

    def search_top1(self, query: np.ndarray) -> Tuple[int, float]:
        """
        Executes flat AVX2 FMA dot-product search against all indexed vectors.
        Returns:
            (best_chunk_index, similarity_score)
        """
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if not query.flags["C_CONTIGUOUS"]:
            query = np.ascontiguousarray(query)

        query = query.flatten()
        expected_dim = self.get_dim()
        if expected_dim > 0 and len(query) != expected_dim:
            raise ValueError(f"Query dim {len(query)} does not match index dim {expected_dim}")

        best_idx = c_int(-1)
        best_score = c_float(-1.0)
        ptr = query.ctypes.data_as(POINTER(c_float))

        rc = self._lib.simd_search_top1(ptr, byref(best_idx), byref(best_score))
        if rc != 0:
            raise RuntimeError(f"simd_search_top1 failed with error code {rc}")

        return int(best_idx.value), float(best_score.value)

    def dot_product(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculates AVX2 FMA dot product of two float32 vectors."""
        a = np.ascontiguousarray(vec_a, dtype=np.float32).flatten()
        b = np.ascontiguousarray(vec_b, dtype=np.float32).flatten()
        if len(a) != len(b):
            raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")

        pa = a.ctypes.data_as(POINTER(c_float))
        pb = b.ctypes.data_as(POINTER(c_float))
        return float(self._lib.avx2_dot_product(pa, pb, len(a)))

    def get_num_vectors(self) -> int:
        return int(self._lib.simd_get_num_vectors())

    def get_dim(self) -> int:
        return int(self._lib.simd_get_dim())

    def cleanup(self) -> int:
        return int(self._lib.simd_cleanup())


class AudioRingBuffer:
    """Pythonic interface to the C++ Lock-Free SPSC Circular Audio Ring Buffer."""

    def __init__(self, capacity_frames: int = DEFAULT_BUFFER_CAPACITY_FRAMES, dll_path: Optional[Union[str, Path]] = None):
        self._lib = SIMDLibrary.get_instance(dll_path).lib
        rc = self._lib.ring_buffer_init(capacity_frames)
        if rc != 0:
            raise RuntimeError(f"Failed to initialize ring buffer with capacity {capacity_frames}")
        self.capacity_frames = capacity_frames

    def push_frame(self, frame_samples: Union[np.ndarray, bytes]) -> bool:
        """Pushes exactly 1 20ms frame (320 int16 samples / 640 bytes)."""
        if isinstance(frame_samples, (bytes, bytearray)):
            if len(frame_samples) != FRAME_BYTES:
                raise ValueError(f"Expected {FRAME_BYTES} bytes, got {len(frame_samples)}")
            arr = np.frombuffer(frame_samples, dtype=np.int16)
        else:
            arr = np.ascontiguousarray(frame_samples, dtype=np.int16).flatten()
            if len(arr) != FRAME_SAMPLES:
                raise ValueError(f"Expected {FRAME_SAMPLES} samples, got {len(arr)}")

        ptr = arr.ctypes.data_as(POINTER(c_int16))
        return bool(self._lib.ring_buffer_push_frame(ptr))

    def push(self, samples: np.ndarray) -> int:
        """Pushes arbitrary number of int16 samples in 320-sample frame blocks."""
        arr = np.ascontiguousarray(samples, dtype=np.int16).flatten()
        ptr = arr.ctypes.data_as(POINTER(c_int16))
        return int(self._lib.ring_buffer_push(ptr, len(arr)))

    def push_bytes(self, pcm_bytes: bytes) -> int:
        """Pushes raw PCM bytes in 640-byte frame blocks."""
        byte_arr = (c_uint8 * len(pcm_bytes)).from_buffer_copy(pcm_bytes)
        return int(self._lib.ring_buffer_push_bytes(byte_arr, len(pcm_bytes)))

    def pop_frame(self) -> Optional[np.ndarray]:
        """Pops 1 20ms frame (320 samples). Returns None if buffer is empty."""
        out_buf = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        ptr = out_buf.ctypes.data_as(POINTER(c_int16))
        success = self._lib.ring_buffer_pop_frame(ptr)
        if success:
            return out_buf
        return None

    def pop_frames(self, num_frames: int = 1) -> np.ndarray:
        """Pops up to num_frames (each 320 samples). Returns (N, 320) or flat array."""
        total_samples = num_frames * FRAME_SAMPLES
        out_buf = np.zeros(total_samples, dtype=np.int16)
        ptr = out_buf.ctypes.data_as(POINTER(c_int16))
        popped_frames = self._lib.ring_buffer_pop(ptr, total_samples)
        if popped_frames == 0:
            return np.empty(0, dtype=np.int16)
        return out_buf[:popped_frames * FRAME_SAMPLES]

    def pop_bytes(self, max_bytes: int = FRAME_BYTES) -> bytes:
        """Pops raw PCM bytes (aligned to 640-byte frames)."""
        buf = (c_uint8 * max_bytes)()
        popped_frames = self._lib.ring_buffer_pop_bytes(buf, max_bytes)
        if popped_frames == 0:
            return b""
        total_popped_bytes = popped_frames * FRAME_BYTES
        return bytes(buf)[:total_popped_bytes]

    def available_read(self) -> int:
        """Returns number of 20ms frames available for consumer read."""
        return int(self._lib.ring_buffer_available_read())

    def available_write(self) -> int:
        """Returns number of 20ms frames available for producer write."""
        return int(self._lib.ring_buffer_available_write())

    def clear(self) -> None:
        self._lib.ring_buffer_clear()

    def cleanup(self) -> None:
        self._lib.ring_buffer_cleanup()


# Module-level convenience functions directly wrapping C-ABI calls
_simd_lib_instance = None

def _get_lib():
    global _simd_lib_instance
    if _simd_lib_instance is None:
        _simd_lib_instance = SIMDLibrary.get_instance()
    return _simd_lib_instance.lib

def simd_init(num_vectors: int, dim: int) -> int:
    return _get_lib().simd_init(int(num_vectors), int(dim))

def simd_load_vectors(vectors: np.ndarray) -> int:
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    num_vectors, dim = arr.shape
    ptr = arr.ctypes.data_as(POINTER(c_float))
    return _get_lib().simd_load_vectors(ptr, num_vectors, dim)

def simd_search_top1(query_vec: np.ndarray) -> Tuple[int, float]:
    arr = np.ascontiguousarray(query_vec, dtype=np.float32).flatten()
    best_idx = c_int(-1)
    best_score = c_float(-1.0)
    ptr = arr.ctypes.data_as(POINTER(c_float))
    rc = _get_lib().simd_search_top1(ptr, byref(best_idx), byref(best_score))
    if rc != 0:
        raise RuntimeError(f"simd_search_top1 failed: code {rc}")
    return int(best_idx.value), float(best_score.value)

def simd_cleanup() -> int:
    return _get_lib().simd_cleanup()

def ring_buffer_init(capacity_frames: int = DEFAULT_BUFFER_CAPACITY_FRAMES) -> int:
    return _get_lib().ring_buffer_init(capacity_frames)

def ring_buffer_push_frame(samples: np.ndarray) -> int:
    arr = np.ascontiguousarray(samples, dtype=np.int16).flatten()
    ptr = arr.ctypes.data_as(POINTER(c_int16))
    return _get_lib().ring_buffer_push_frame(ptr)

def ring_buffer_pop_frame(out_samples: np.ndarray) -> int:
    arr = np.ascontiguousarray(out_samples, dtype=np.int16).flatten()
    ptr = arr.ctypes.data_as(POINTER(c_int16))
    return _get_lib().ring_buffer_pop_frame(ptr)
