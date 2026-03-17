from __future__ import annotations

from typing import Any

import numpy as np

from .base import Backend


class NumPyBackend(Backend):
    name = "numpy"
    is_gpu = False

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._workspace: dict[tuple[str, tuple[int, ...], np.dtype[Any]], np.ndarray] = {}

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

    def seed(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def random(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.rng.random(shape)

    def standard_normal(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.rng.standard_normal(shape)

    def choice_pm1(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)

    def get_workspace(self, key: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        np_dtype = np.dtype(dtype)
        cache_key = (key, tuple(shape), np_dtype)
        arr = self._workspace.get(cache_key)
        if arr is None:
            arr = np.empty(shape, dtype=np_dtype)
            self._workspace[cache_key] = arr
        arr.fill(0)
        return arr

    def clear_workspace(self) -> None:
        self._workspace.clear()

    def memory_pool_stats(self) -> dict[str, int]:
        return {"used_bytes": 0, "total_bytes": 0, "workspace_entries": len(self._workspace)}

    def batched_commutator_update(self, G_batch: Any, h: Any, dt: float) -> np.ndarray:
        comm = self.matmul(G_batch, h) - self.matmul(h, G_batch)
        return G_batch + (-2.0j * dt) * comm

    def symmetrize_clip_diag_inplace(self, G_batch: Any) -> np.ndarray:
        G_batch[...] = 0.5 * (G_batch + np.conj(np.swapaxes(G_batch, -1, -2)))
        dim = int(G_batch.shape[-1])
        diag_idx = np.arange(dim)
        diag = np.real(G_batch[..., diag_idx, diag_idx])
        G_batch[..., diag_idx, diag_idx] = np.clip(diag, 0.0, 1.0) + 0.0j
        return G_batch
