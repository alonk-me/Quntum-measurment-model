"""
jw_H_XX
=======

Sanity check utilities for a two‑qubit system with the XX Hamiltonian

    H = J (sigma_x^(1) sigma_x^(2)).

This module mirrors the API of jw_H_XY for convenience, but the
Hamiltonian contains only the X⊗X term. Two approaches are provided:
direct wavefunction evolution via diagonalization, and a simple
parallel runner for evaluating multiple parameter sets.

Note: A JW correlation‑matrix variant matching this H is not provided
here; corr_magnetization raises NotImplementedError.
"""

from __future__ import annotations

import numpy as np
from multiprocessing import Pool
from typing import Iterable, List, Tuple
from matrix_commutator_solver import compute_time_series

def _pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return sigma_x, sigma_y, sigma_z


def _build_xx_hamiltonian(J: float) -> np.ndarray:
    """Construct the 4x4 XX Hamiltonian: H = J (X⊗X)."""
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
    """Compute <sigma_z^(1)>(t) via wavefunction evolution for XX H (diag)."""
    H = _build_xx_hamiltonian(J)
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0 = _map_initial_state(initial_state)
    # sigma_z on first qubit
    _, _, sigma_z = _pauli_matrices()
    sz1 = np.kron(sigma_z, np.eye(2, dtype=complex))
    times = np.linspace(0.0, T, steps + 1)
    magnetization = np.empty(steps + 1, dtype=float)
    for idx, t in enumerate(times):
        phase = np.exp(-1.0j * eigvals * t)
        U = eigvecs @ (phase[:, None] * eigvecs.conj().T)
        psi_t = U @ psi0
        magnetization[idx] = float(np.real(np.conj(psi_t).T @ (sz1 @ psi_t)))
    return times, magnetization


def corr_magnetization(
    J: float,
    T: float,
    steps: int,
    L: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute <sigma_z^(1)>(t) via the free‑fermion correlation matrix.

    This wraps free_fermion.matrix_commutator_solver.compute_time_series(L,J,T,steps)
    and returns the real part of the observable series, which corresponds to
    <sigma_z^(1)>(t) according to the documentation.
    """
    if L != 2:
        raise NotImplementedError("Direct qubit chain simulation not implemented for L != 2.")

    times, values = compute_time_series(L=L, J=J, T=T, steps=steps)
    # The solver returns 1 + 2*G[0,0], which maps to <sigma_z1>
    magnetization = np.real(values.astype(complex))
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
