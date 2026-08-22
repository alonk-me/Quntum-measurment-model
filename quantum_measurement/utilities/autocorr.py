"""Autocorrelation helpers for z(t) analysis.

Provides NumPy-based autocorrelation routines for single-trajectory and
ensemble computations. Returns absolute (unnormalized) autocorrelations
C(k) = mean_i z[i] * z[i+k] for k=0..N-1 where N = len(z).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def autocorr_single(z: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Compute autocorrelation for a single 1-D trajectory.

    Parameters
    ----------
    z : np.ndarray
        1-D array of length N (z values at times 0..N-1 or 0..N_steps).
    max_lag : Optional[int]
        Maximum lag to compute (inclusive). If None, compute up to N-1.

    Returns
    -------
    np.ndarray
        1-D array of length (max_lag + 1) containing C(k) = mean_i z[i] * z[i+k].
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError("z must be a 1-D array")

    n_points = z.shape[0]
    if n_points == 0:
        return np.empty(0, dtype=float)

    if max_lag is None:
        max_lag = n_points - 1
    max_lag = min(int(max_lag), n_points - 1)
    if max_lag < 0:
        return np.empty(0, dtype=float)

    c = np.empty(max_lag + 1, dtype=float)
    for lag in range(max_lag + 1):
        left = z[: n_points - lag]
        right = z[lag:]
        c[lag] = float(np.mean(left * right)) if left.size else 0.0
    return c


def autocorr_ensemble(z_trajs: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """Compute ensemble-averaged autocorrelation for multiple trajectories.

    Parameters
    ----------
    z_trajs : np.ndarray
        Array of shape (n_traj, N) or (n_traj, N, 1) with z values.
    max_lag : Optional[int]
        Maximum lag to compute. If None, uses N-1.

    Returns
    -------
    np.ndarray
        Ensemble-averaged autocorrelation array of length (max_lag + 1).
    """
    z = np.asarray(z_trajs)
    if z.ndim == 3 and z.shape[2] == 1:
        z = z[:, :, 0]
    if z.ndim != 2:
        raise ValueError("z_trajs must be 2-D (n_traj, N) or 3-D with last dim 1")

    n_traj, n_points = z.shape
    if n_traj == 0:
        return np.empty(0, dtype=float)

    if max_lag is None:
        max_lag = n_points - 1
    max_lag = min(int(max_lag), n_points - 1)
    if max_lag < 0:
        return np.empty(0, dtype=float)

    c_sum = np.zeros(max_lag + 1, dtype=float)
    for idx in range(n_traj):
        c_sum += autocorr_single(z[idx], max_lag=max_lag)
    return c_sum / float(n_traj)
