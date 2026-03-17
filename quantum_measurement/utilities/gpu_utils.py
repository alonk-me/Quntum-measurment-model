from __future__ import annotations

import importlib
from typing import Any


def check_gpu_available() -> bool:
    try:
        cp = importlib.import_module("cupy")

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_gpu_memory_info(device_id: int = 0) -> dict[str, Any]:
    """Return basic VRAM information for the selected device.

    Raises RuntimeError if CuPy/CUDA is unavailable.
    """
    try:
        cp = importlib.import_module("cupy")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("CuPy is required for GPU memory inspection.") from exc

    try:
        with cp.cuda.Device(device_id):
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception as exc:
        raise RuntimeError(f"Unable to query CUDA memory for device {device_id}.") from exc

    used_bytes = total_bytes - free_bytes
    return {
        "device_id": device_id,
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "used_bytes": int(used_bytes),
        "total_gb": total_bytes / (1024**3),
        "free_gb": free_bytes / (1024**3),
        "used_gb": used_bytes / (1024**3),
    }


def estimate_trajectory_batch_size(
    L: int,
    max_vram_gb: float | None = None,
    usage_fraction: float = 0.60,
) -> int:
    """Estimate a conservative trajectory batch size.

    Uses a simple model based on dominant (2L x 2L) complex matrices.
    For L=256, this gives roughly 4 MB per trajectory.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    if not (0.0 < usage_fraction <= 1.0):
        raise ValueError("usage_fraction must be in (0, 1]")

    if max_vram_gb is None:
        if not check_gpu_available():
            return 1
        max_vram_gb = float(get_gpu_memory_info()["total_gb"])

    # Heuristic: approximately 2 * (2L)^2 complex128 matrices resident, plus overhead.
    # complex128 is 16 bytes. Overhead factor widens safety margin.
    bytes_per_traj = int(2.0 * (2 * L) * (2 * L) * 16 * 2.0)
    bytes_per_traj = max(bytes_per_traj, 1)

    budget_bytes = int(max_vram_gb * (1024**3) * usage_fraction)
    batch_size = budget_bytes // bytes_per_traj
    return max(1, int(batch_size))
