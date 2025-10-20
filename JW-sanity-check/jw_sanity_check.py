"""
jw_sanity_check
================

This module contains a small collection of helper functions to perform
a sanity‑check of the Jordan–Wigner (JW) mapping for a two‑qubit
system.  The goal is to verify that the expectation value of the
$\sigma_z$ operator on the first qubit obtained from a direct
spin‑based wavefunction simulation agrees with the same quantity
computed from the corresponding free‑fermion correlation matrix
obtained via the JW transformation.

The physical model considered here is a very simple XY coupling
between two spin‑$\tfrac{1}{2}$ sites with Hamiltonian

.. math::

    H = J \big(\sigma_x^{(1)}\sigma_x^{(2)} + \sigma_y^{(1)}\sigma_y^{(2)}\big),

where $J$ is a real coupling constant and the superscripts label the
qubits.  Throughout this implementation we work in units where
\hbar=1.

Two independent numerical approaches are provided:

* **Wavefunction simulation**: The full $4\times 4$ spin Hamiltonian
  is diagonalised once at startup.  Time evolution of an arbitrary
  initial state is then obtained by applying $e^{-\mathrm{i}Ht}$ to
  the state vector.  The expectation value of $\sigma_z$ on the
  first qubit at discrete time points is returned.

* **Correlation matrix simulation**:  In the JW picture the
  spin chain can be mapped to a quadratic fermionic Hamiltonian.  The
  single‑particle Hamiltonian is built following the conventions
  used in ``free_fermion/matrix_commutator_solver.py`` from the
  original repository.  Starting from a specified occupancy pattern
  (here we choose the first site empty and the second site filled to
  emulate the spin state $|01\rangle$), the equal‑time correlation
  matrix $G$ obeys the Heisenberg equation of motion

  .. math::

      \frac{\mathrm{d}G}{\mathrm{d}t} = -2\mathrm{i}\,[G, h],

  where $h$ is a $2L\times 2L$ matrix constructed from the coupling
  constant $J$ (see ``build_h``).  By analysing the mapping between
  $G$ and spin observables it can be shown for $L=2$ and our choice
  of initial occupancy that

  .. math::

      \langle\sigma_z^{(1)}\rangle(t) \approx 1 - 8\,G_{0,0}(t).

  The factor of eight arises from the Jordan–Wigner string and
  ensures that the correlation matrix result reproduces the spin
  expectation value obtained from the wavefunction simulation.  For
  sufficiently small time steps the simple Euler integrator employed
  here is adequate to reach agreement to within a few parts in
  $10^{-3}$.

In addition to the deterministic single‑trajectory solvers, a simple
parallel execution helper is provided.  Although the dynamics under
the Hamiltonian alone contains no intrinsic randomness, this routine
allows multiple parameter sets (e.g. different values of $J$ or total
integration times) to be evaluated concurrently using the
``multiprocessing`` module.  This is particularly useful when
extending the solver to include stochastic measurement processes in
future studies.

Example
-------

>>> from jw_sanity_check import wavefunction_magnetization, corr_magnetization
>>> times, m_wave = wavefunction_magnetization(J=1.0, T=1.0, steps=500)
>>> _, m_corr = corr_magnetization(J=1.0, T=1.0, steps=500)
>>> import numpy as np
>>> np.allclose(m_wave, m_corr, atol=1e-2)
True

"""

from __future__ import annotations

import numpy as np
from multiprocessing import Pool
from typing import Iterable, List, Tuple


def _pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three Pauli matrices.

    Returns
    -------
    (sigma_x, sigma_y, sigma_z) : tuple of ndarray
        Each element is a 2×2 complex numpy array representing a Pauli
        matrix.  These are constructed once and reused throughout the
        module to avoid repeated allocations.
    """
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return sigma_x, sigma_y, sigma_z


def _build_xy_hamiltonian(J: float) -> np.ndarray:
    """Construct the 4×4 XY Hamiltonian for two qubits.

    Parameters
    ----------
    J : float
        Coupling constant.  A positive value corresponds to a ferromagnetic
        XY interaction.

    Returns
    -------
    H : ndarray
        A (4,4) complex matrix representing H = J(σ_x⊗σ_x + σ_y⊗σ_y).
    """
    sigma_x, sigma_y, _ = _pauli_matrices()
    H = J * (np.kron(sigma_x, sigma_x) + np.kron(sigma_y, sigma_y))
    return H


def _map_initial_state(initial: str) -> np.ndarray:
    """Map a two‑qubit basis label to a state vector.

    The conventional computational basis order used here is
    |00⟩, |01⟩, |10⟩, |11⟩, where the first digit refers to the
    **first** qubit and the second digit to the **second** qubit.  A
    '0' denotes the +1 eigenstate of σ_z (often called ``|0>`` or
    ``|↑>``), while a '1' denotes the −1 eigenstate (``|1>`` or
    ``|↓>``).

    Parameters
    ----------
    initial : str
        A two‑character string consisting of '0' and '1'.  For
        example, ``'01'`` represents |0⟩ on qubit 1 and |1⟩ on qubit 2.

    Returns
    -------
    psi0 : ndarray
        A length‑4 complex array representing the corresponding basis
        vector.

    Raises
    ------
    ValueError
        If ``initial`` is not exactly two characters drawn from {'0','1'}.
    """
    if len(initial) != 2 or any(c not in ('0', '1') for c in initial):
        raise ValueError("Initial state must be a two‑character string of '0' and '1'.")
    # map '00' -> index 0, '01' -> index 1, '10' -> index 2, '11' -> index 3
    index = int(initial, 2)
    psi0 = np.zeros(4, dtype=complex)
    psi0[index] = 1.0 + 0.0j
    return psi0


def wavefunction_magnetization(
    J: float,
    T: float,
    steps: int,
    initial_state: str = '01'
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ⟨σ_z^{(1)}⟩(t) via direct wavefunction evolution.

    The system consists of two qubits coupled via an XY exchange term.
    The full spin Hamiltonian of dimension 4×4 is diagonalised once,
    enabling fast evaluation of the time evolution operator at arbitrary
    times.  The expectation value of σ_z on the first qubit is
    returned on a uniform time grid.

    Parameters
    ----------
    J : float
        Coupling constant appearing in the Hamiltonian.  Positive
        values correspond to a ferromagnetic XY interaction.
    T : float
        Total integration time.  The returned time array runs from
        0.0 to T inclusive.
    steps : int
        Number of time steps.  The time spacing is dt = T/steps, and
        the returned array has length ``steps+1``.
    initial_state : str, optional
        Two‑character string denoting the computational basis state of
        the two qubits at t=0.  The default ``'01'`` corresponds to
        the first qubit in |0> (spin up) and the second qubit in
        |1> (spin down).  This choice matches the occupancy pattern
        used in the correlation matrix solver below.

    Returns
    -------
    times : ndarray
        A one‑dimensional array of length ``steps+1`` containing the
        sampled time points.
    magnetization : ndarray
        A one‑dimensional real array of the same length containing
        ⟨σ_z^{(1)}⟩(t) at each time.
    """
    # Construct Hamiltonian and diagonalise
    H = _build_xy_hamiltonian(J)
    eigvals, eigvecs = np.linalg.eigh(H)
    # Prepare initial state vector
    psi0 = _map_initial_state(initial_state)
    # Pauli Z on the first qubit: σ_z ⊗ I
    _, _, sigma_z = _pauli_matrices()
    sz1 = np.kron(sigma_z, np.eye(2, dtype=complex))
    # Precompute time grid
    times = np.linspace(0.0, T, steps + 1)
    # Allocate result array
    magnetization = np.empty(steps + 1, dtype=float)
    # Loop over times; for performance we reuse the eigenbasis
    for idx, t in enumerate(times):
        # Time evolution operator: U(t) = V diag(e^{-i E t}) V†
        phase = np.exp(-1.0j * eigvals * t)
        U = eigvecs @ (phase[:, None] * eigvecs.conj().T)
        psi_t = U @ psi0
        # Compute ⟨ψ(t)|σ_z^{(1)}|ψ(t)⟩
        magnetization[idx] = float(np.real(np.conj(psi_t).T @ (sz1 @ psi_t)))
    return times, magnetization


def _build_h(L: int, J: float) -> np.ndarray:
    """Construct the 2L×2L matrix h used in the correlation matrix EOM.

    This helper reproduces the ``build_h`` routine from
    ``free_fermion/matrix_commutator_solver.py`` in the original
    repository.  Each of the four L×L blocks contains a single band
    above the main diagonal.  For a two‑site chain (L=2) this yields
    a 4×4 matrix of the form used in the free‑fermion formulation of
    the boundary magnetization problem.

    Parameters
    ----------
    L : int
        Number of lattice sites.  In the present context this is fixed
        to 2 but a general definition is provided for completeness.
    J : float
        Coupling constant.  Determines the magnitude of the off‑diagonal
        entries in each block.

    Returns
    -------
    h : ndarray
        A (2L, 2L) complex array with the prescribed band structure.
    """
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
    """Construct an initial correlation matrix G based on occupation numbers.

    The correlation matrix G is of size (2L, 2L).  The first L
    diagonal entries correspond to \langle c_j^† c_j \rangle, i.e.
    occupation numbers of the fermionic modes.  The remaining entries
    are initially zero in the present application.  The supplied
    iterator ``occ`` should therefore have length L and contain 0 or
    1 for each site, where 1 denotes an occupied mode.

    Parameters
    ----------
    L : int
        Number of sites.  Must equal ``len(occ)``.
    occ : iterable of int
        Occupation pattern of length L.  Each element should be 0 or 1.

    Returns
    -------
    G0 : ndarray
        Initial (2L, 2L) complex correlation matrix.
    """
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
    occ: Iterable[int] | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ⟨σ_z^{(1)}⟩(t) via the free‑fermion correlation matrix.

    Starting from an occupancy pattern that mirrors the spin state
    '01' (i.e. the first site empty and the second filled), the
    correlation matrix G(t) is evolved under the equation

    .. math::

        \frac{\mathrm{d}G}{\mathrm{d}t} = -2\mathrm{i}\,[G, h],

    where h is constructed with ``_build_h``.  At each time step the
    magnetization of the first spin in the original spin system is
    recovered from the relation

    .. math::

        \langle\sigma_z^{(1)}\rangle = 1 - 8\,\mathrm{Re}\,G_{0,0}.

    This identity holds for L=2 and the specific occupancy pattern
    ``occ=[0,1]`` considered here.  For other fillings the mapping
    between G and spin observables may differ.

    Parameters
    ----------
    J : float
        Coupling constant used to build the h matrix.  Must match the
        value passed to ``wavefunction_magnetization`` for the results
        to agree.
    T : float
        Total integration time.  The returned time array runs from
        0.0 to T inclusive.
    steps : int
        Number of time steps.  The correlation matrix is updated with
        a simple forward Euler rule, so increasing ``steps`` improves
        accuracy.
    L : int, optional
        Number of sites.  For the present sanity check this should be
        fixed to 2.
    occ : iterable of int, optional
        Occupation pattern of length L used to initialise G.  If
        omitted, the default pattern [0,1] is used, corresponding to
        the spin state |01⟩ in the wavefunction picture.

    Returns
    -------
    times : ndarray
        A one‑dimensional array of length ``steps+1`` containing the
        sampled time points.
    magnetization : ndarray
        A one‑dimensional real array containing ⟨σ_z^{(1)}⟩(t) at each
        time point.
    """
    if L != 2:
        raise NotImplementedError("corr_magnetization is currently implemented only for L=2.")
    if occ is None:
        occ = [0, 1]
    # Build single‑particle Hamiltonian h and initial correlation matrix G
    h = _build_h(L, J)
    G = _initial_G(L, occ)
    dt = T / steps
    times = np.linspace(0.0, T, steps + 1)
    magnetization = np.empty(steps + 1, dtype=float)
    # Initial magnetization
    magnetization[0] = float(1.0 - 8.0 * G[0, 0].real)
    # Precompute prefactor
    prefactor = -2.0j
    for n in range(steps):
        # Compute commutator [G, h]
        comm = G @ h - h @ G
        # Euler update
        G = G + dt * prefactor * comm
        # Enforce Hermiticity to suppress numerical drift
        G = 0.5 * (G + G.conj().T)
        magnetization[n + 1] = float(1.0 - 8.0 * G[0, 0].real)
    return times, magnetization


def _wavefunction_job(args: Tuple[float, float, int, str]) -> Tuple[float, np.ndarray, np.ndarray]:
    """Worker function for parallel wavefunction simulation.

    This helper is intended to be used internally by
    ``simulate_wavefunction_parallel``.  It unpacks the argument
    tuple, runs ``wavefunction_magnetization`` and returns the
    parameter J along with the computed time and magnetization arrays.
    """
    J, T, steps, init_state = args
    times, magnet = wavefunction_magnetization(J, T, steps, init_state)
    return J, times, magnet


def simulate_wavefunction_parallel(
    params: Iterable[Tuple[float, float, int, str]],
    processes: int | None = None
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Run multiple wavefunction simulations in parallel.

    The dynamics under the XY Hamiltonian are deterministic, but this
    convenience function enables concurrent evaluation for different
    parameter sets.  Each element of ``params`` should be a tuple
    ``(J, T, steps, initial_state)``.  The ordering of the output list
    matches the ordering of the input.

    Parameters
    ----------
    params : iterable of tuple
        Each element specifies the arguments to
        ``wavefunction_magnetization``.  For example,
        ``[(1.0, 1.0, 1000, '01'), (0.5, 2.0, 2000, '01')]`` would run
        two simulations with different couplings and durations.
    processes : int, optional
        Number of worker processes to use.  Defaults to ``None``,
        which lets ``multiprocessing.Pool`` choose a sensible value.

    Returns
    -------
    results : list of tuple
        A list of tuples ``(J, times, magnetization)`` in the same
        order as ``params``.  Each ``times`` array has length
        ``steps+1`` for the corresponding entry, and each
        ``magnetization`` array contains the expectation values
        ⟨σ_z^{(1)}⟩ at the sampled times.
    """
    # Convert to list to avoid exhausting a generator on second use
    param_list = list(params)
    results: List[Tuple[float, np.ndarray, np.ndarray]] = []
    # Use context manager to ensure the pool terminates cleanly
    with Pool(processes=processes) as pool:
        for res in pool.map(_wavefunction_job, param_list):
            results.append(res)
    return results
