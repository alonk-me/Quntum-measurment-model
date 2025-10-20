"""
jw_H_XX_euler
=============

Sanity check utilities for a two‑qubit system with the XX Hamiltonian

    H = J (sigma_x^(1) sigma_x^(2)),

using an explicit Euler time stepping scheme that avoids diagonalization.
The state update is

    psi_{n+1} = psi_n + (-i H psi_n) dt.

For sufficiently small time steps this reproduces the exact dynamics to
good accuracy and provides a simple baseline for testing.
"""

from __future__ import annotations

import numpy as np
from multiprocessing import Pool
from typing import Iterable, List, Tuple


def _pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return sigma_x, sigma_y, sigma_z


def _build_xx_hamiltonian(J: float) -> np.ndarray:
    sigma_x, _, _ = _pauli_matrices()
    H = J * (np.kron(sigma_x, sigma_x))
    return H


def _map_initial_state(initial: str) -> np.ndarray:
    if len(initial) != 2 or any(c not in ("0", "1") for c in initial):
        raise ValueError("Initial state must be a two‑character string of '0' and '1'.")
    index = int(initial, 2)
    psi0 = np.zeros(4, dtype=complex)
    psi0[index] = 1.0 + 0.0j
    return psi0


def wavefunction_magnetization(
    J: float,
    T: float,
    steps: int,
    initial_state: str = "01",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute <sigma_z^(1)>(t) via explicit Euler for XX H (no diagonalization)."""
    H = _build_xx_hamiltonian(J)
    psi = _map_initial_state(initial_state)
    _, _, sigma_z = _pauli_matrices()
    sz1 = np.kron(sigma_z, np.eye(2, dtype=complex))
    dt = T / steps
    times = np.linspace(0.0, T, steps + 1)
    magnetization = np.empty(steps + 1, dtype=float)
    magnetization[0] = float(np.real(np.conj(psi).T @ (sz1 @ psi)))
    iH = -1.0j * H
    for n in range(steps):
        # Explicit Euler step: psi <- psi + (iH psi) dt, where iH = -i H
        psi = psi + dt * (iH @ psi)
        # Optional re-normalization to control drift
        norm = np.linalg.norm(psi)
        if norm == 0.0:
            raise FloatingPointError("Wavefunction norm vanished; consider reducing dt.")
        psi = psi / norm
        magnetization[n + 1] = float(np.real(np.conj(psi).T @ (sz1 @ psi)))
    return times, magnetization


def _wavefunction_job(args: Tuple[float, float, int, str]) -> Tuple[float, np.ndarray, np.ndarray]:
    J, T, steps, init_state = args
    times, magnet = wavefunction_magnetization(J, T, steps, init_state)
    return J, times, magnet


def simulate_wavefunction_parallel(
    params: Iterable[Tuple[float, float, int, str]],
    processes: int | None = None,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    param_list = list(params)
    results: List[Tuple[float, np.ndarray, np.ndarray]] = []
    with Pool(processes=processes) as pool:
        for res in pool.map(_wavefunction_job, param_list):
            results.append(res)
    return results


def corr_magnetization(
    J: float,
    T: float,
    steps: int,
    L: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute <sigma_z^(1)>(t) via the free‑fermion correlation matrix.

    Provided for comparison with the explicit Euler wavefunction update.
    """
    if L != 2:
        raise NotImplementedError("corr_magnetization currently implemented for L=2 only.")
    from free_fermion.matrix_commutator_solver import compute_time_series

    times, values = compute_time_series(L=L, J=J, T=T, steps=steps)
    magnetization = np.real(values.astype(complex))
    return times, magnetization
