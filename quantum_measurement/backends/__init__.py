from __future__ import annotations

import importlib
import warnings

from .base import Backend
from .numpy_backend import NumPyBackend
from .cupy_backend import CuPyBackend


def is_cupy_available() -> bool:
    try:
        cp = importlib.import_module("cupy")

        _ = cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def get_backend(device: str = "cpu", seed: int | None = None) -> Backend:
    device_lower = device.lower()
    if device_lower in {"cpu", "numpy"}:
        return NumPyBackend(seed=seed)
    if device_lower in {"gpu", "cuda", "cupy"}:
        if is_cupy_available():
            return CuPyBackend(seed=seed)
        warnings.warn("GPU backend requested but CuPy/CUDA unavailable; falling back to NumPy backend.", RuntimeWarning)
        return NumPyBackend(seed=seed)
    raise ValueError(f"Unsupported device '{device}'. Use 'cpu' or 'gpu'.")


__all__ = [
    "Backend",
    "CuPyBackend",
    "NumPyBackend",
    "get_backend",
    "is_cupy_available",
]
