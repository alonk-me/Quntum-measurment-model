from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .base import Backend

cp = importlib.import_module("cupy") if importlib.util.find_spec("cupy") else None


class CuPyBackend(Backend):
    name = "cupy"
    is_gpu = True

    def __init__(self) -> None:
        if cp is None:
            raise RuntimeError("CuPy is not available. Install cupy-cuda12x (or matching CUDA build).")

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        return cp.array(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Any:
        return cp.zeros(shape, dtype=dtype)

    def diag(self, a: Any) -> Any:
        return cp.diag(a)

    def hstack(self, arrays: list[Any] | tuple[Any, ...]) -> Any:
        return cp.hstack(arrays)

    def vstack(self, arrays: list[Any] | tuple[Any, ...]) -> Any:
        return cp.vstack(arrays)

    def matmul(self, a: Any, b: Any) -> Any:
        return cp.matmul(a, b)

    def conj(self, a: Any) -> Any:
        return cp.conj(a)

    def transpose(self, a: Any) -> Any:
        return cp.transpose(a)

    def real(self, a: Any) -> Any:
        return cp.real(a)

    def clip(self, a: Any, a_min: float, a_max: float) -> Any:
        return cp.clip(a, a_min, a_max)

    def copy(self, a: Any) -> Any:
        return cp.array(a, copy=True)

    def asnumpy(self, a: Any) -> np.ndarray:
        if cp is None:
            return np.asarray(a)
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
        return np.asarray(a)
