from __future__ import annotations

import importlib
from typing import Any
import warnings

import numpy as np

from .base import Backend

cp = importlib.import_module("cupy") if importlib.util.find_spec("cupy") else None


class CuPyBackend(Backend):
    name = "cupy"
    is_gpu = True

    def __init__(self, seed: int | None = None) -> None:
        if cp is None:
            raise RuntimeError("CuPy is not available. Install cupy-cuda12x (or matching CUDA build).")
        self.rng = cp.random.default_rng(seed)
        self._mempool = cp.get_default_memory_pool()
        self._pinned_mempool = cp.get_default_pinned_memory_pool()
        self._workspace: dict[tuple[str, tuple[int, ...], Any], Any] = {}
        self._stable_warned = False
        # Fused elementwise kernel over flattened batch tensors:
        # (symmetrize + diagonal clipping) in one GPU kernel launch.
        self._sym_clip_kernel = cp.ElementwiseKernel(
            "raw T x, int32 dim",
            "T y",
            """
            const int d = dim;
            const size_t idx = i;
            const int col = (int)(idx % d);
            const int row = (int)((idx / d) % d);
            const size_t batch = idx / ((size_t)d * (size_t)d);
            const size_t pair = batch * ((size_t)d * (size_t)d) + (size_t)col * (size_t)d + (size_t)row;

            T val = (x[idx] + conj(x[pair])) * (T)0.5;
            if (row == col) {
                double re = real(val);
                if (re < 0.0) re = 0.0;
                if (re > 1.0) re = 1.0;
                val = T(re, 0.0);
            }
            y = val;
            """,
            name="symmetrize_clip_diag_fused",
        )

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

    def seed(self, seed: int | None = None) -> None:
        if cp is None:
            raise RuntimeError("CuPy is not available. Cannot reseed CuPy RNG.")
        self.rng = cp.random.default_rng(seed)

    def random(self, shape: tuple[int, ...]) -> Any:
        return self.rng.random(shape)

    def standard_normal(self, shape: tuple[int, ...]) -> Any:
        return self.rng.standard_normal(shape)

    def choice_pm1(self, shape: tuple[int, ...]) -> Any:
        # Draw from {-1, +1} directly on device using Generator.random,
        # which is available across CuPy RNG Generator versions.
        pm = (self.rng.random(shape) < 0.5).astype(cp.int8)
        return (pm * 2 - 1).astype(cp.int8)

    def get_workspace(self, key: str, shape: tuple[int, ...], dtype: Any) -> Any:
        if cp is None:
            raise RuntimeError("CuPy is not available. Cannot allocate GPU workspace.")
        cp_dtype = cp.dtype(dtype)
        cache_key = (key, tuple(shape), cp_dtype)
        arr = self._workspace.get(cache_key)
        if arr is None:
            arr = cp.empty(shape, dtype=cp_dtype)
            self._workspace[cache_key] = arr
        arr.fill(0)
        return arr

    def clear_workspace(self) -> None:
        self._workspace.clear()

    def memory_pool_stats(self) -> dict[str, int]:
        return {
            "used_bytes": int(self._mempool.used_bytes()),
            "total_bytes": int(self._mempool.total_bytes()),
            "pinned_free_blocks": int(self._pinned_mempool.n_free_blocks()),
            "workspace_entries": len(self._workspace),
        }

    def batched_commutator_update(
        self,
        G_batch: Any,
        h: Any,
        dt: float,
        *,
        use_stable_integrator: bool = False,
        precomputed_u: Any | None = None,
        warn_on_ignored_stable: bool = False,
    ) -> Any:
        if cp is None:
            raise RuntimeError("CuPy is not available. Cannot run GPU commutator update.")
        if use_stable_integrator and warn_on_ignored_stable and not self._stable_warned:
            warnings.warn(
                "Stable integrator is currently CPU-only; CuPy backend will continue with Euler path.",
                RuntimeWarning,
            )
            self._stable_warned = True
        gh = self.get_workspace("gh", tuple(G_batch.shape), G_batch.dtype)
        hg = self.get_workspace("hg", tuple(G_batch.shape), G_batch.dtype)
        cp.matmul(G_batch, h, out=gh)
        cp.matmul(h, G_batch, out=hg)
        gh -= hg
        gh *= (-2.0j * dt)
        gh += G_batch
        return gh

    def symmetrize_clip_diag_inplace(self, G_batch: Any) -> Any:
        if cp is None:
            raise RuntimeError("CuPy is not available. Cannot run GPU in-place symmetrization.")
        if isinstance(G_batch, cp.ndarray) and G_batch.ndim == 3 and G_batch.dtype.kind == "c":
            dim = int(G_batch.shape[-1])
            flat = G_batch.reshape(-1)
            out = self._sym_clip_kernel(flat, np.int32(dim), size=flat.size)
            G_batch[...] = out.reshape(G_batch.shape)
            return G_batch

        # Fallback path for non-standard shapes/dtypes.
        G_batch[...] = 0.5 * (G_batch + cp.conj(cp.swapaxes(G_batch, -1, -2)))
        dim = int(G_batch.shape[-1])
        diag_idx = cp.arange(dim)
        diag = cp.real(G_batch[..., diag_idx, diag_idx])
        G_batch[..., diag_idx, diag_idx] = cp.clip(diag, 0.0, 1.0) + 0.0j
        return G_batch
