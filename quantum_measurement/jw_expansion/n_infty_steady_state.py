r"""Correlation matrix verification for n_∞(g) calculations.

This module provides tools to verify that the momentum-space formulas
in ``n_infty.py`` correctly arise from diagonalizing the steady-state
correlation matrix for a monitored free-fermion chain.

The module can validate steady-state predictions using both analytical
methods and numerical simulations from ``non_hermitian_hat.py``.

The key verification is that eigenvalues of the steady-state correlation
matrix C match the momentum-space dispersion Δ(k,g) used in the
finite-size sums.

Theory
------
For a monitored free-fermion chain with hopping J and monitoring rate γ,
the steady-state correlation matrix C satisfies a modified Lyapunov equation.
In momentum space, the eigenvalues are determined by:

    Δ(k, g) = sqrt(1 - g² - 2ig cos k),  where g = γ/(4J)

This module constructs C explicitly in real space and compares its
eigenvalues with the momentum-space prediction.

Functions
---------
build_bdg_hamiltonian :
    Construct Bogoliubov-de Gennes Hamiltonian for given boundary conditions
momentum_space_eigenvalues :
    Compute eigenvalues directly from Δ(k,g) formula
steady_state_correlation_matrix :
    Solve for steady-state C from Lyapunov equation
get_correlation_eigenvalues :
    Extract and sort eigenvalues from correlation matrix
compare_eigenvalues :
    Compare two sets of eigenvalues with detailed metrics
extract_steady_state_from_simulation :
    Run NonHermitianHatSimulator to equilibrium and extract final correlation matrix
validate_steady_state_with_simulation :
    Compare simulated steady-state with analytical momentum-space predictions
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Dict, Tuple
from n_infty import delta, sign_im, term_value


def build_bdg_hamiltonian(
    L: int,
    J: float = 1.0,
    boundary: Literal["apbc", "pbc"] = "apbc"
) -> np.ndarray:
    """Construct the Bogoliubov-de Gennes Hamiltonian for free fermions.

    The BdG Hamiltonian has 2L×2L structure with four L×L blocks.
    For nearest-neighbor hopping with coupling J, the blocks follow
    the pattern from the free-fermion correlation matrix formalism.

    Parameters
    ----------
    L : int
        Number of sites in the chain. Must be at least 2.
    J : float, optional
        Hopping coupling constant. Default is 1.0.
    boundary : {'apbc', 'pbc'}, optional
        Boundary condition:
        - 'apbc': Anti-periodic (twist by π)
        - 'pbc': Periodic (no twist)
        Default is 'apbc'.

    Returns
    -------
    h : np.ndarray
        Complex array of shape (2L, 2L) containing the BdG Hamiltonian.

    Examples
    --------
    >>> h = build_bdg_hamiltonian(L=3, J=1.0, boundary='apbc')
    >>> h.shape
    (6, 6)
    """
    if L < 2:
        raise ValueError("L must be at least 2")
    
    h11 = np.zeros((L, L), dtype=complex)
    h12 = np.zeros((L, L), dtype=complex)
    h21 = np.zeros((L, L), dtype=complex)
    h22 = np.zeros((L, L), dtype=complex)

    # Fill nearest-neighbor couplings
    for i in range(L - 1):
        h11[i, i + 1] = -J
        h12[i, i + 1] = -J
        h22[i, i + 1] = +J
        h21[i, i + 1] = +J

    # Boundary coupling between last and first site
    if L > 1:
        phase = -1.0 if boundary == "apbc" else 1.0
        h11[L - 1, 0] = -J * phase
        h12[L - 1, 0] = -J * phase
        h22[L - 1, 0] = +J * phase
        h21[L - 1, 0] = +J * phase

    # Assemble full BdG matrix
    top = np.hstack((h11, h12))
    bottom = np.hstack((h21, h22))
    h = np.vstack((top, bottom))
    
    return h


def momentum_space_eigenvalues(
    L: int,
    J: float,
    gamma: float,
    boundary: Literal["apbc", "pbc"] = "apbc"
) -> Tuple[np.ndarray, float]:
    """Compute steady-state mode contributions from finite-size formula.

    This function calculates the contribution to n_∞ from each momentum
    mode using the finite-size formula from n_infty_theory.md Eq. (70-82).

    Parameters
    ----------
    L : int
        Number of sites. Must be odd for proper momentum grid.
    J : float
        Hopping coupling constant.
    gamma : float
        Monitoring rate.
    boundary : {'apbc', 'pbc'}, optional
        Boundary condition. Default is 'apbc'.

    Returns
    -------
    mode_contributions : np.ndarray
        Array containing term_value(k, g) for each momentum mode.
    n_infinity : float
        The occupation n_∞ = (1/L) Σ term_value(k).

    Notes
    -----
    The finite-size formula is:
        term(k,g) = (2 sin²k) / [sin²k + |cos k - ig - sign_im(Δ)·Δ|²]
    This differs from the thermodynamic-limit formula n(k) = 1/2 - (1/2g)|Im Δ|.
    
    For finite L, we sum term_value over the discrete momentum grid and 
    normalize by L (not by the number of modes).
    """
    if L % 2 == 0:
        raise ValueError("L must be odd for proper momentum grid")
    
    g = gamma / (4.0 * J)
    
    # Build momentum grid
    if boundary == "apbc":
        n_max = (L - 3) // 2
        k_values = np.array([(2.0 * np.pi / L) * (n + 0.5) 
                              for n in range(n_max + 1)])
    else:  # pbc
        n_max = (L - 1) // 2
        k_values = np.array([(2.0 * np.pi / L) * n 
                              for n in range(1, n_max + 1)])
    
    occupations = []
    
    for k in k_values:
        # Use the finite-size formula from n_infty_theory.md Eq. (70-82)
        # This is the term_value formula, not the simpler integral formula
        occupation = term_value(k, g)
        occupations.append(occupation)
    
    occupations = np.array(occupations)
    
    # Compute n_∞ using the same normalization as sum_apbc/sum_pbc
    n_infinity = np.sum(occupations) / L
    
    return occupations, n_infinity


def steady_state_correlation_matrix(
    L: int,
    J: float,
    gamma: float,
    boundary: Literal["apbc", "pbc"] = "apbc",
    method: Literal["momentum", "lyapunov"] = "momentum"
) -> np.ndarray:
    """Compute the steady-state correlation matrix C.

    Solves for the correlation matrix that satisfies the steady-state
    condition for a monitored free-fermion chain.

    Parameters
    ----------
    L : int
        Number of sites.
    J : float
        Hopping coupling constant.
    gamma : float
        Monitoring rate.
    boundary : {'apbc', 'pbc'}, optional
        Boundary condition. Default is 'apbc'.
    method : {'momentum', 'lyapunov'}, optional
        Solution method:
        - 'momentum': Fourier transform from momentum space (faster)
        - 'lyapunov': Direct solution of Lyapunov equation (more general)
        Default is 'momentum'.

    Returns
    -------
    C : np.ndarray
        Hermitian matrix of shape (2L, 2L) containing the steady-state
        correlation matrix.

    Notes
    -----
    The steady-state satisfies: -2i[C, h_eff] - γC = 0
    where h_eff is the BdG Hamiltonian and γ is the monitoring rate.
    """
    if method == "momentum":
        return _steady_state_momentum_method(L, J, gamma, boundary)
    elif method == "lyapunov":
        return _steady_state_lyapunov_method(L, J, gamma, boundary)
    else:
        raise ValueError(f"Unknown method: {method}")


def _steady_state_momentum_method(
    L: int,
    J: float,
    gamma: float,
    boundary: Literal["apbc", "pbc"]
) -> np.ndarray:
    """Compute steady-state C via momentum-space eigendecomposition."""
    # Build BdG Hamiltonian
    h = build_bdg_hamiltonian(L, J, boundary)
    
    # Add imaginary potential: h_eff = h - i*gamma/2 * I
    h_eff = h - 1j * (gamma / 2.0) * np.eye(2 * L)
    
    # Diagonalize h_eff
    eigenvalues, eigenvectors = np.linalg.eig(h_eff)
    
    # Compute steady-state occupations
    # For each eigenvalue λ of h_eff, the steady-state occupation is
    # determined by the balance between coherent and dissipative dynamics
    
    # Sort by real part of eigenvalue
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Build correlation matrix from occupied states
    # The states with negative imaginary part of eigenvalue decay
    # Steady state corresponds to projecting onto the stable subspace
    
    C = np.zeros((2 * L, 2 * L), dtype=complex)
    
    # For monitored system, compute occupation from eigenvalue structure
    for i in range(2 * L):
        lam = eigenvalues[i]
        vec = eigenvectors[:, i]
        
        # Occupation based on the imaginary part
        # States with Im(λ) < 0 decay, those with Im(λ) = -γ/2 are steady
        if np.abs(np.imag(lam) + gamma / 2.0) < 1e-10:
            # This is a steady eigenmode
            occupation = _compute_occupation_from_eigenvalue(lam, gamma, J)
            C += occupation * np.outer(vec, vec.conj())
    
    # Ensure Hermiticity
    C = 0.5 * (C + C.conj().T)
    
    return C


def _compute_occupation_from_eigenvalue(
    lam: complex,
    gamma: float,
    J: float
) -> float:
    """Compute steady-state occupation for an eigenmode."""
    # For the monitored system, occupation is determined by
    # the balance between coherent evolution and dissipation
    # This is derived from the quantum jump formalism
    
    # Real part of eigenvalue determines the effective energy
    # Imaginary part (should be -γ/2 for steady state) gives decay
    
    # Simplified formula: occupation decreases with energy
    # For a more accurate implementation, use the full formula from theory
    
    # Placeholder: equal distribution for now
    # TODO: Implement exact formula from n_infty_theory.md
    return 0.5


def _steady_state_lyapunov_method(
    L: int,
    J: float,
    gamma: float,
    boundary: Literal["apbc", "pbc"]
) -> np.ndarray:
    """Compute steady-state C via direct Lyapunov equation solution."""
    # This is a more complex method - implement if momentum method fails
    raise NotImplementedError(
        "Lyapunov method not yet implemented. Use method='momentum'."
    )


def get_correlation_eigenvalues(C: np.ndarray) -> np.ndarray:
    """Extract sorted eigenvalues from correlation matrix.

    Parameters
    ----------
    C : np.ndarray
        Hermitian correlation matrix of shape (2L, 2L).

    Returns
    -------
    eigenvalues : np.ndarray
        Sorted array of real eigenvalues.

    Notes
    -----
    Uses Hermitian eigenvalue solver for numerical stability.
    Eigenvalues are sorted in ascending order.
    """
    eigenvalues = np.linalg.eigvalsh(C)
    return np.sort(eigenvalues)


def compare_eigenvalues(
    eigs1: np.ndarray,
    eigs2: np.ndarray,
    tolerance: float = 1e-8,
    label1: str = "Method 1",
    label2: str = "Method 2"
) -> Dict[str, any]:
    """Compare two sets of eigenvalues with detailed metrics.

    Parameters
    ----------
    eigs1 : np.ndarray
        First set of eigenvalues (sorted).
    eigs2 : np.ndarray
        Second set of eigenvalues (sorted).
    tolerance : float, optional
        Absolute tolerance for considering values equal. Default is 1e-8.
    label1 : str, optional
        Label for first method. Default is "Method 1".
    label2 : str, optional
        Label for second method. Default is "Method 2".

    Returns
    -------
    comparison : dict
        Dictionary containing:
        - 'max_abs_diff': Maximum absolute difference
        - 'mean_abs_diff': Mean absolute difference
        - 'rms_diff': RMS difference
        - 'all_close': Boolean indicating if all values are within tolerance
        - 'num_matched': Number of eigenvalue pairs within tolerance
        - 'matched_pairs': List of (eig1, eig2, diff) tuples

    Examples
    --------
    >>> eigs1 = np.array([0.1, 0.5, 0.9])
    >>> eigs2 = np.array([0.1 + 1e-10, 0.5 + 1e-10, 0.9 + 1e-10])
    >>> result = compare_eigenvalues(eigs1, eigs2)
    >>> result['all_close']
    True
    """
    if len(eigs1) != len(eigs2):
        raise ValueError(
            f"Eigenvalue arrays have different lengths: {len(eigs1)} vs {len(eigs2)}"
        )
    
    eigs1_sorted = np.sort(eigs1)
    eigs2_sorted = np.sort(eigs2)
    
    diff = np.abs(eigs1_sorted - eigs2_sorted)
    
    max_abs_diff = np.max(diff)
    mean_abs_diff = np.mean(diff)
    rms_diff = np.sqrt(np.mean(diff**2))
    all_close = np.allclose(eigs1_sorted, eigs2_sorted, atol=tolerance, rtol=0)
    num_matched = np.sum(diff < tolerance)
    
    matched_pairs = [
        (e1, e2, d) 
        for e1, e2, d in zip(eigs1_sorted, eigs2_sorted, diff)
    ]
    
    return {
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'rms_diff': rms_diff,
        'all_close': all_close,
        'num_matched': num_matched,
        'total': len(eigs1),
        'matched_pairs': matched_pairs,
        'label1': label1,
        'label2': label2
    }


def verify_n_infinity_from_occupations(
    k_occupations: np.ndarray,
    L: int,
    g: float,
    J: float,
    boundary: Literal["apbc", "pbc"] = "apbc"
) -> Dict[str, float]:
    """Verify n_∞ by comparing momentum-space occupations with sum formula.

    Parameters
    ----------
    k_occupations : np.ndarray
        Occupation n(k) for each momentum mode.
    L : int
        Number of sites.
    g : float
        Dimensionless measurement strength.
    J : float
        Hopping coupling.
    boundary : {'apbc', 'pbc'}, optional
        Boundary condition.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'n_from_occupations': Average of momentum occupations
        - 'n_from_sum_formula': Result from sum_apbc or sum_pbc
        - 'difference': Absolute difference
        - 'matches': Boolean indicating if they match within tolerance
    """
    from n_infty import sum_apbc, sum_pbc
    
    n_from_occ = np.mean(k_occupations)
    
    if boundary == "apbc":
        n_from_sum = sum_apbc(g, L)
    else:
        n_from_sum = sum_pbc(g, L)
    
    diff = abs(n_from_occ - n_from_sum)
    matches = diff < 1e-10
    
    return {
        'n_from_occupations': n_from_occ,
        'n_from_sum_formula': n_from_sum,
        'difference': diff,
        'matches': matches
    }


def extract_steady_state_from_simulation(
    L: int,
    J: float,
    gamma: float,
    dt: float = 0.0001,
    N_steps: int = 50000,
    boundary: Literal["pbc", "open"] = "open"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run NonHermitianHatSimulator to equilibrium and extract final correlation matrix.

    This function provides a numerical steady-state solution by evolving
    the system from the vacuum state until it reaches equilibrium.

    Parameters
    ----------
    L : int
        Number of sites in the chain.
    J : float
        Hopping coupling constant.
    gamma : float
        Monitoring rate.
    dt : float, optional
        Time step for Euler integration. Default is 0.0001.
    N_steps : int, optional
        Number of time steps. Total evolution time is dt * N_steps.
        Default is 50000 (total time = 5.0 for default dt).
    boundary : {'pbc', 'open'}, optional
        Boundary condition. Note: 'apbc' not supported by simulator.
        Default is 'open'.

    Returns
    -------
    G_final : np.ndarray
        Steady-state correlation matrix of shape (2L, 2L).
    n_traj : np.ndarray
        Array of shape (N_steps+1, L) recording occupations over time.
    Q_total : float
        Total entropy production accumulated during evolution.

    Examples
    --------
    >>> G, n_traj, Q = extract_steady_state_from_simulation(
    ...     L=5, J=1.0, gamma=0.4, N_steps=10000
    ... )
    >>> G.shape
    (10, 10)
    
    Notes
    -----
    The simulator evolves from vacuum state |↓↓...↓⟩ to steady state.
    Ensure N_steps is large enough for convergence. Check that the
    occupations n_traj[-1] have stabilized.
    """
    from non_hermitian_hat import NonHermitianHatSimulator
    
    closed = (boundary == "pbc")
    sim = NonHermitianHatSimulator(
        L=L,
        J=J,
        gamma=gamma,
        dt=dt,
        N_steps=N_steps,
        closed_boundary=closed
    )
    
    Q_total, n_traj, G_final = sim.simulate_trajectory()
    
    return G_final, n_traj, Q_total


def validate_steady_state_with_simulation(
    L: int,
    J: float,
    gamma: float,
    boundary: Literal["pbc", "open"] = "open",
    dt: float = 0.0001,
    N_steps: int = 50000,
    tolerance: float = 1e-6
) -> Dict[str, any]:
    """Compare simulated steady-state with analytical momentum-space predictions.

    This function validates the analytical steady-state formulas by comparing
    them with numerical results from the NonHermitianHatSimulator.

    Parameters
    ----------
    L : int
        Number of sites. Must be odd for proper momentum grid.
    J : float
        Hopping coupling constant.
    gamma : float
        Monitoring rate.
    boundary : {'pbc', 'open'}, optional
        Boundary condition. Note: Analytical code uses 'apbc'/'pbc', but
        simulator only supports 'pbc'/'open'. Default is 'open'.
    dt : float, optional
        Time step for simulation. Default is 0.0001.
    N_steps : int, optional
        Number of simulation steps. Default is 50000.
    tolerance : float, optional
        Tolerance for eigenvalue comparison. Default is 1e-6.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'G_simulated': Final correlation matrix from simulation (2L×2L)
        - 'eigs_simulated': Eigenvalues from simulation
        - 'mode_contributions': Momentum-space mode contributions
        - 'n_simulated': Final occupation from simulation
        - 'n_analytical': Occupation from momentum-space sum
        - 'occupation_diff': Absolute difference in occupations
        - 'convergence_check': Max change in last 10% of trajectory
        - 'Q_total': Total entropy production from simulation
        - 'T_total': Total evolution time
        - 'final_occupations': Site occupations at final time

    Examples
    --------
    >>> results = validate_steady_state_with_simulation(
    ...     L=5, J=1.0, gamma=0.4, N_steps=20000
    ... )
    >>> print(f"Occupation difference: {results['occupation_diff']:.2e}")
    >>> print(f"Converged: {results['convergence_check'] < 1e-4}")

    Notes
    -----
    For accurate results:
    - Use odd L for proper momentum grid
    - Ensure N_steps is large enough for convergence
    - Check 'convergence_check' to verify steady state was reached
    - Boundary condition mapping: 'open' ≈ 'apbc', 'pbc' = 'pbc'
    """
    from n_infty import sum_apbc, sum_pbc
    
    # Get simulated steady state
    G_sim, n_traj, Q_total = extract_steady_state_from_simulation(
        L, J, gamma, dt, N_steps, boundary
    )
    
    # Extract simulated eigenvalues and occupations
    eigs_sim = get_correlation_eigenvalues(G_sim)
    n_sim = np.real(np.diag(G_sim)[:L]).mean()
    
    # Get analytical predictions from momentum space
    g = gamma / (4.0 * J)
    boundary_analytical = "pbc" if boundary == "pbc" else "apbc"
    
    try:
        mode_contributions, n_ana = momentum_space_eigenvalues(
            L, J, gamma, boundary=boundary_analytical
        )
    except ValueError as e:
        # If L is even, fall back to direct sum formula
        if boundary == "pbc":
            n_ana = sum_pbc(g, L)
        else:
            n_ana = sum_apbc(g, L)
        mode_contributions = None
    
    # Check convergence by looking at changes in final 10% of trajectory
    tail_start = int(0.9 * N_steps)
    convergence_check = np.max(np.abs(
        n_traj[-1, :] - n_traj[tail_start, :]
    ))
    
    return {
        'G_simulated': G_sim,
        'eigs_simulated': eigs_sim,
        'mode_contributions': mode_contributions,
        'n_simulated': n_sim,
        'n_analytical': n_ana,
        'occupation_diff': abs(n_sim - n_ana),
        'convergence_check': convergence_check,
        'Q_total': Q_total,
        'T_total': dt * N_steps,
        'final_occupations': n_traj[-1, :],
        'occupation_trajectory': n_traj
    }
