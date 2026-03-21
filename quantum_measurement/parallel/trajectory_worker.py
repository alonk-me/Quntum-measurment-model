"""Pure-NumPy trajectory worker for multi-CPU free-fermion simulation.

Priority mapping:
- Accuracy: enforces complex128 end-to-end and deterministic SeedSequence child streams.
- Overflow: applies condition checks, Tikhonov regularization, projection guards, and safe entropy.
- Speed: vectorized NumPy algebra in inner updates; returns only compact trajectory outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
from scipy.special import xlogy


EPS_LOG = 1e-15
DENOM_EPS = 1e-14
COND_THRESHOLD = 1e12
REG_EPS = 1e-10
TRACE_DRIFT_TOL = 1e-12


@dataclass(frozen=True)
class TrajectoryTask:
    traj_id: int
    child_seed: int


@dataclass
class TrajectoryConfig:
    L: int
    J: float
    gamma: float
    dt: float
    n_steps: int
    closed_boundary: bool
    n_particles: float | None = None


def build_hamiltonian(L: int, J: float, closed_boundary: bool) -> np.ndarray:
    h11 = np.zeros((L, L), dtype=np.complex128)
    h12 = np.zeros((L, L), dtype=np.complex128)
    h21 = np.zeros((L, L), dtype=np.complex128)
    h22 = np.zeros((L, L), dtype=np.complex128)

    for i in range(L - 1):
        h11[i, i + 1] = -J
        h12[i, i + 1] = -J
        h21[i, i + 1] = +J
        h22[i, i + 1] = +J

    if closed_boundary and L > 1:
        h11[L - 1, 0] = -J
        h12[L - 1, 0] = -J
        h21[L - 1, 0] = +J
        h22[L - 1, 0] = +J

    top = np.hstack((h11, h12))
    bottom = np.hstack((h21, h22))
    return np.vstack((top, bottom)).astype(np.complex128, copy=False)


def _entropy_from_eigvals(eigvals: np.ndarray) -> float:
    vals = np.clip(np.real(eigvals), EPS_LOG, 1.0 - EPS_LOG)
    return float(-np.sum(xlogy(vals, vals) + xlogy(1.0 - vals, 1.0 - vals)))


def _apply_projection_multi_site(
    C: np.ndarray,
    site_indices: np.ndarray,
    traj_id: int,
    step: int,
    L: int,
    gamma: float,
    logger: Any,
) -> np.ndarray:
    """Apply batched rank-1 projection updates for all measured sites.

    Vectorized form:
        C <- C - sum_k outer(C v_k, v_k^H C) / denom_k
    where each v_k is a canonical basis vector for measured site k.
    """
    denom_all = np.real(C[site_indices, site_indices]).astype(np.float64, copy=False)
    valid = (denom_all >= DENOM_EPS) & (denom_all <= 1.0 - DENOM_EPS)

    if logger is not None:
        skipped = np.where(~valid)[0]
        for pos in skipped:
            site_idx = int(site_indices[pos])
            denom = float(denom_all[pos])
            from quantum_measurement.numerics.overflow_log import OverflowEvent

            logger.emit(
                OverflowEvent(
                    event_type="projection_skip",
                    traj_id=traj_id,
                    step=step,
                    L=L,
                    gamma=gamma,
                    message="Skipped projection due to deterministic-site denominator",
                    payload={"site_idx": site_idx, "denom": denom},
                )
            )

    if not np.any(valid):
        return C

    valid_sites = site_indices[valid]
    denom = denom_all[valid].astype(np.float64, copy=False)

    # V selects measured sites from canonical basis, shape (dim, m).
    V = np.eye(C.shape[0], dtype=np.complex128)[:, valid_sites]
    CV = C @ V
    VC = V.conj().T @ C
    C = C - (CV * (1.0 / denom)[None, :]) @ VC
    return C


def _apply_projection_site(
    C: np.ndarray,
    site_idx: int,
    traj_id: int,
    step: int,
    L: int,
    gamma: float,
    logger: Any,
) -> np.ndarray:
    """Backward-compatible shim, delegates to multi-site projection."""
    return _apply_projection_multi_site(
        C=C,
        site_indices=np.array([site_idx], dtype=np.int64),
        traj_id=traj_id,
        step=step,
        L=L,
        gamma=gamma,
        logger=logger,
    )


def _stabilize_C(
    C: np.ndarray,
    n_particles: float,
    traj_id: int,
    step: int,
    L: int,
    gamma: float,
    logger: Any,
) -> np.ndarray:
    """Hermitize, clip eigenvalues to [0,1], and conditionally renormalize trace."""
    C = 0.5 * (C + C.conj().T)

    cond_number = np.linalg.cond(C)
    if math.isfinite(cond_number) and cond_number > COND_THRESHOLD:
        C = C + REG_EPS * np.eye(C.shape[0], dtype=np.complex128)
        if logger is not None:
            from quantum_measurement.numerics.overflow_log import OverflowEvent

            logger.emit(
                OverflowEvent(
                    event_type="cond_regularization",
                    traj_id=traj_id,
                    step=step,
                    L=L,
                    gamma=gamma,
                    message="Applied Tikhonov regularization due to high condition number",
                    payload={"cond": float(cond_number), "eps": REG_EPS},
                )
            )

    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(np.real(eigvals), 0.0, 1.0)
    C = (eigvecs @ np.diag(eigvals) @ eigvecs.conj().T).astype(np.complex128, copy=False)

    tr = float(np.real(np.trace(C)))
    if abs(tr - n_particles) > TRACE_DRIFT_TOL and abs(tr) > 0.0:
        C *= n_particles / tr
        if logger is not None:
            from quantum_measurement.numerics.overflow_log import OverflowEvent

            logger.emit(
                OverflowEvent(
                    event_type="trace_renormalization",
                    traj_id=traj_id,
                    step=step,
                    L=L,
                    gamma=gamma,
                    message="Applied conditional trace renormalization",
                    payload={"trace_before": tr, "target": n_particles},
                )
            )

    return C


def run_single_trajectory(
    task: TrajectoryTask,
    config: TrajectoryConfig,
    logger: Any = None,
) -> tuple[int, np.ndarray]:
    """Run one deterministic trajectory and return (traj_id, S_A_array)."""
    _ = np.random.default_rng(task.child_seed)
    L = int(config.L)
    dim = 2 * L
    h = build_hamiltonian(L, float(config.J), bool(config.closed_boundary))

    C = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(L, dim):
        C[i, i] = 1.0 + 0.0j

    n_particles = float(config.n_particles) if config.n_particles is not None else float(np.real(np.trace(C)))

    sigma_diag = np.concatenate([np.ones(L), -np.ones(L)]).astype(np.float64, copy=False)
    sigma_z = np.diag(sigma_diag).astype(np.complex128, copy=False)

    S_A = np.zeros(config.n_steps + 1, dtype=np.float64)
    S_A[0] = _entropy_from_eigvals(np.linalg.eigvalsh(C[:L, :L]))

    for step in range(config.n_steps):
        comm = C @ h - h @ C
        C = C + (-2.0j * config.dt) * comm

        term1 = (C @ sigma_z) @ C
        term2 = 0.5 * (sigma_z @ C + C @ sigma_z)
        C = C + config.dt * config.gamma * (term1 - term2)

        # Batched multi-site projection for all measured sites in one vectorized update.
        all_sites = np.arange(L, dtype=np.int64)
        C = _apply_projection_multi_site(C, all_sites, task.traj_id, step, L, config.gamma, logger)

        C = _stabilize_C(C, n_particles, task.traj_id, step, L, config.gamma, logger)

        eigvals_A = np.linalg.eigvalsh(C[:L, :L])
        S_A[step + 1] = _entropy_from_eigvals(eigvals_A)

    return task.traj_id, S_A


def run_trajectory_chunk(
    chunk_tasks: list[TrajectoryTask],
    config: TrajectoryConfig,
    log_path: str | None = None,
) -> tuple[list[tuple[int, np.ndarray]], dict[str, int]]:
    """Run a chunk of trajectories and return compact results + overflow counters."""
    logger = None
    if log_path is not None:
        from quantum_measurement.numerics.overflow_log import OverflowLogger

        logger = OverflowLogger(log_path)

    out: list[tuple[int, np.ndarray]] = []
    for task in chunk_tasks:
        out.append(run_single_trajectory(task, config, logger=logger))

    counts: dict[str, int] = logger.counts() if logger is not None else {}
    return out, counts
