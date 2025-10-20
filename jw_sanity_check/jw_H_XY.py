"""
jw_H_XY
========

Helper functions to perform a sanity check of the Jordan–Wigner (JW)
mapping for a two‑qubit system with an XY coupling Hamiltonian

    H = J (sigma_x^(1) sigma_x^(2) + sigma_y^(1) sigma_y^(2)).

The goal is to verify that the expectation value of sigma_z on the first
qubit obtained from a direct spin‑based wavefunction simulation agrees
with the corresponding quantity computed from the free‑fermion
correlation matrix derived via the JW transformation.

This module is adapted from the previous jw_sanity_check.py, but the
module name now explicitly indicates the XY Hamiltonian.
"""

from __future__ import annotations

import numpy as np
from multiprocessing import Pool
from typing import Iterable, List, Tuple


def _pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three Pauli matrices (2x2 complex arrays)."""
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return sigma_x, sigma_y, sigma_z


def _build_xy_hamiltonian(J: float) -> np.ndarray:
    """Construct the 4x4 XY Hamiltonian for two qubits: H = J (X⊗X + Y⊗Y)."""
    sigma_x, sigma_y, _ = _pauli_matrices()
    H = J * (np.kron(sigma_x, sigma_x) + np.kron(sigma_y, sigma_y))
    return H


def _map_initial_state(initial: str) -> np.ndarray:
    """Map a two‑qubit computational basis label (e.g., '01') to a 4-vector."""
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
    """Compute <sigma_z^(1)>(t) via direct wavefunction evolution for XY H."""
    H = _build_xy_hamiltonian(J)
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


def _build_h(L: int, J: float) -> np.ndarray:
    """Construct the 2L x 2L single‑particle h matrix used for JW correlation EOM."""
    h11 = np.zeros((L, L), dtype=complex)
    h12 = np.zeros((L, L), dtype=complex)
    h21 = np.zeros((L, L), dtype=complex)
    h22 = np.zeros((L, L), dtype=complex)
    for i in range(L - 1):
        h11[i, i + 1] = -J
        h12[i, i + 1] = -J
        h22[i, i + 1] = J
        h21[i, i + 1] = J
    top = np.hstack((h11, h12))
    bottom = np.hstack((h21, h22))
    h = np.vstack((top, bottom))
    return h


def _initial_G(L: int, occ: Iterable[int]) -> np.ndarray:
    """Initial (2L x 2L) correlation matrix with diagonal occupations occ."""
    G0 = np.zeros((2 * L, 2 * L), dtype=complex)
    occ_list = list(occ)
    if len(occ_list) != L:
        raise ValueError(f"Occupation list must have length {L}.")
    for j, n in enumerate(occ_list):
        G0[j, j] = 1.0 if n else 0.0
    return G0


def corr_magnetization(
    J: float,
    T: float,
    steps: int,
    L: int = 2,
    occ: Iterable[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute <sigma_z^(1)>(t) via JW correlation matrix for XY H (L=2 only)."""
    if L != 2:
        raise NotImplementedError("corr_magnetization is currently implemented only for L=2.")
    if occ is None:
        occ = [0, 1]
    h = _build_h(L, J)
    G = _initial_G(L, occ)
    dt = T / steps
    times = np.linspace(0.0, T, steps + 1)
    magnetization = np.empty(steps + 1, dtype=float)
    magnetization[0] = float(1.0 - 8.0 * G[0, 0].real)
    prefactor = -2.0j
    for n in range(steps):
        comm = G @ h - h @ G
        G = G + dt * prefactor * comm
        G = 0.5 * (G + G.conj().T)
        magnetization[n + 1] = float(1.0 - 8.0 * G[0, 0].real)
    return times, magnetization


def _wavefunction_job(args: Tuple[float, float, int, str]) -> Tuple[float, np.ndarray, np.ndarray]:
    J, T, steps, init_state = args
    times, magnet = wavefunction_magnetization(J, T, steps, init_state)
    return J, times, magnet


def simulate_wavefunction_parallel(
    params: Iterable[Tuple[float, float, int, str]],
    processes: int | None = None,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Run multiple wavefunction simulations in parallel for XY H."""
    param_list = list(params)
    results: List[Tuple[float, np.ndarray, np.ndarray]] = []
    with Pool(processes=processes) as pool:
        for res in pool.map(_wavefunction_job, param_list):
            results.append(res)
    return results
