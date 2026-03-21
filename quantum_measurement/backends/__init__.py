from __future__ import annotations

import importlib

from .base import Backend
from .multi_cpu_backend import MultiCpuBackend, MultiCpuBackendConfig
from .numpy_backend import NumPyBackend


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
            from .cupy_backend import CuPyBackend

            return CuPyBackend(seed=seed)
        raise RuntimeError("GPU backend requested but CuPy/CUDA unavailable.")
    raise ValueError(f"Unsupported device '{device}'. Use 'cpu' or 'gpu'.")


__all__ = [
    "Backend",
    "MultiCpuBackend",
    "MultiCpuBackendConfig",
    "NumPyBackend",
    "get_backend",
    "is_cupy_available",
]
