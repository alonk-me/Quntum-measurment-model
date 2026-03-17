from __future__ import annotations

from typing import Any

import numpy as np

from .base import Backend


class NumPyBackend(Backend):
    name = "numpy"
    is_gpu = False

    def array(self, data: Any, dtype: Any | None = None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def diag(self, a: Any) -> np.ndarray:
        return np.diag(a)

    def hstack(self, arrays: list[Any] | tuple[Any, ...]) -> np.ndarray:
        return np.hstack(arrays)

    def vstack(self, arrays: list[Any] | tuple[Any, ...]) -> np.ndarray:
        return np.vstack(arrays)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return np.matmul(a, b)

    def conj(self, a: Any) -> np.ndarray:
        return np.conj(a)

    def transpose(self, a: Any) -> np.ndarray:
        return np.transpose(a)

    def real(self, a: Any) -> np.ndarray:
        return np.real(a)

    def clip(self, a: Any, a_min: float, a_max: float) -> np.ndarray:
        return np.clip(a, a_min, a_max)

    def copy(self, a: Any) -> np.ndarray:
        return np.array(a, copy=True)

    def asnumpy(self, a: Any) -> np.ndarray:
        return np.asarray(a)
