"""
correlation_matrix_susceptibility.py
=====================================

This module provides functions to compute susceptibility from free-fermion
correlation matrices using the Jordan-Wigner transformation.

For the Jordan-Wigner mapping:
    σᵢᶻ = 2c†ᵢcᵢ - 1

The correlation matrix element G[i,i] = ⟨c†ᵢcᵢ⟩ gives the occupation number,
so:
    ⟨σᵢᶻ⟩ = 2*Re(G[i,i]) - 1

Two-point spin correlations can be computed from:
    ⟨σᵢᶻσⱼᶻ⟩ = 4*Re(G[i,i]*G[j,j]) - 2*Re(G[i,i]) - 2*Re(G[j,j]) + 1
              + 4*|G[i,j]|² (for i≠j)

Functions
---------
extract_spin_correlations : Extract ⟨σᵢᶻ⟩ from correlation matrix G
susceptibility_from_correlation_matrix : Compute susceptibility from G(t) time series
compute_G_correlation_function : Two-time correlation of G matrix elements
"""

import numpy as np
from typing import Tuple, List, Optional


def extract_spin_correlations(G: np.ndarray) -> np.ndarray:
    """Extract spin expectation values ⟨σᵢᶻ⟩ from correlation matrix.
    
    Uses the Jordan-Wigner mapping:
        σᵢᶻ = 2c†ᵢcᵢ - 1
        ⟨σᵢᶻ⟩ = 2*Re(G[i,i]) - 1
    
    Parameters
    ----------
    G : np.ndarray
        Correlation matrix of shape (2L, 2L) where L is the number of qubits.
        The first L diagonal elements correspond to the L qubit sites.
    
    Returns
    -------
    np.ndarray
        Array of ⟨σᵢᶻ⟩ values for each site, shape (L,)
    
    Examples
    --------
    >>> G = np.diag([1.0, 1.0, 0.0, 0.0])  # |↑↑⟩ state for L=2
    >>> z_values = extract_spin_correlations(G)
    >>> print(z_values)  # [1.0, 1.0]
    """
    # For a 2L×2L matrix, the first L elements are the qubit sites
    L = G.shape[0] // 2
    
    z_values = np.zeros(L)
    for i in range(L):
        z_values[i] = 2.0 * np.real(G[i, i]) - 1.0
    
    return z_values


def extract_two_point_spin_correlation(
    G: np.ndarray,
    i: int,
    j: int,
) -> float:
    """Extract two-point spin correlation ⟨σᵢᶻσⱼᶻ⟩ from correlation matrix.
    
    For i = j (single site):
        ⟨σᵢᶻσᵢᶻ⟩ = ⟨(σᵢᶻ)²⟩ = 1
    
    For i ≠ j:
        ⟨σᵢᶻσⱼᶻ⟩ = (2⟨c†ᵢcᵢ⟩ - 1)(2⟨c†ⱼcⱼ⟩ - 1)
                   = 4*Re(G[i,i])*Re(G[j,j]) - 2*Re(G[i,i]) - 2*Re(G[j,j]) + 1
    
    Parameters
    ----------
    G : np.ndarray
        Correlation matrix (2L, 2L)
    i : int
        First site index (0 to L-1)
    j : int
        Second site index (0 to L-1)
    
    Returns
    -------
    float
        Two-point correlation ⟨σᵢᶻσⱼᶻ⟩
    
    Notes
    -----
    This is a mean-field approximation valid when fermions are uncorrelated.
    For exact correlations with off-diagonal G elements, more complex formulas
    involving Wick's theorem are needed.
    """
    if i == j:
        return 1.0
    
    # Mean-field approximation: ⟨σᵢᶻσⱼᶻ⟩ ≈ ⟨σᵢᶻ⟩⟨σⱼᶻ⟩
    # This is exact when off-diagonal elements of G are zero
    G_ii = np.real(G[i, i])
    G_jj = np.real(G[j, j])
    
    z_i = 2.0 * G_ii - 1.0
    z_j = 2.0 * G_jj - 1.0
    
    correlation = z_i * z_j
    
    # Include correction from off-diagonal elements if significant
    if np.abs(G[i, j]) > 1e-10:
        # More accurate formula including off-diagonal terms
        # ⟨σᵢᶻσⱼᶻ⟩ = 1 - 2G_ii - 2G_jj + 4*G_ii*G_jj + 4*|G_ij|²
        correlation_exact = (
            1.0 
            - 2.0 * G_ii 
            - 2.0 * G_jj 
            + 4.0 * G_ii * G_jj 
            + 4.0 * np.abs(G[i, j])**2
        )
        return correlation_exact
    
    return correlation


def susceptibility_from_correlation_matrix(
    times: np.ndarray,
    G_trajectory: List[np.ndarray],
    site_i: Optional[int] = None,
    site_j: Optional[int] = None,
) -> float:
    """Compute susceptibility from correlation matrix trajectory.
    
    Extracts spin expectations ⟨σᵢᶻ(t)⟩ from each G(t) and computes the
    static susceptibility by integrating the correlation function.
    
    Parameters
    ----------
    times : np.ndarray
        Time points, shape (n_times,)
    G_trajectory : List[np.ndarray]
        List of correlation matrices G(t), each of shape (2L, 2L)
    site_i : int, optional
        First site index. If None, compute total susceptibility (sum over all sites)
    site_j : int, optional
        Second site index. If None, use site_i (autocorrelation)
    
    Returns
    -------
    float
        Susceptibility χ_ij or total χ if sites not specified
    
    Examples
    --------
    >>> times = np.linspace(0, 10, 100)
    >>> G_traj = [np.eye(4, dtype=complex) for _ in range(100)]
    >>> chi = susceptibility_from_correlation_matrix(times, G_traj, site_i=0)
    """
    from quantum_measurement.susceptibility.static_susceptibility import (
        compute_static_susceptibility
    )
    
    n_times = len(G_trajectory)
    L = G_trajectory[0].shape[0] // 2
    
    if site_i is None:
        # Compute total susceptibility: sum over all site pairs
        total_chi = 0.0
        for i in range(L):
            for j in range(L):
                # Extract z trajectories
                z_i_traj = np.array([extract_spin_correlations(G)[i] for G in G_trajectory])
                z_j_traj = np.array([extract_spin_correlations(G)[j] for G in G_trajectory])
                
                # Add to total
                total_chi += compute_static_susceptibility(times, z_i_traj, z_j_traj)
        
        return total_chi
    
    else:
        if site_j is None:
            site_j = site_i
        
        # Extract z trajectories for specified sites
        z_i_traj = np.array([extract_spin_correlations(G)[site_i] for G in G_trajectory])
        z_j_traj = np.array([extract_spin_correlations(G)[site_j] for G in G_trajectory])
        
        # Compute susceptibility
        chi = compute_static_susceptibility(times, z_i_traj, z_j_traj)
        
        return chi


def susceptibility_matrix_from_G_trajectory(
    times: np.ndarray,
    G_trajectory: List[np.ndarray],
) -> np.ndarray:
    """Compute full susceptibility matrix from correlation matrix trajectory.
    
    Parameters
    ----------
    times : np.ndarray
        Time points, shape (n_times,)
    G_trajectory : List[np.ndarray]
        List of correlation matrices G(t)
    
    Returns
    -------
    np.ndarray
        Susceptibility matrix χ[i,j], shape (L, L)
    """
    L = G_trajectory[0].shape[0] // 2
    chi_matrix = np.zeros((L, L))
    
    for i in range(L):
        for j in range(L):
            chi_matrix[i, j] = susceptibility_from_correlation_matrix(
                times, G_trajectory, site_i=i, site_j=j
            )
    
    return chi_matrix


def compute_G_correlation_function(
    G_trajectory: List[np.ndarray],
    i: int,
    j: int,
) -> np.ndarray:
    """Compute correlation function of G matrix elements.
    
    Computes C(t) = ⟨G[i,j](t) G[i,j](0)⟩ - ⟨G[i,j]⟩²
    
    This can be used to study the dynamics of the correlation matrix itself.
    
    Parameters
    ----------
    G_trajectory : List[np.ndarray]
        Trajectory of correlation matrices
    i : int
        Row index
    j : int
        Column index
    
    Returns
    -------
    np.ndarray
        Correlation function of G[i,j]
    """
    n_times = len(G_trajectory)
    G_values = np.array([G[i, j] for G in G_trajectory])
    
    mean_G = np.mean(G_values)
    
    # Autocorrelation
    correlation = np.zeros(n_times, dtype=complex)
    for t in range(n_times):
        correlation[t] = np.mean(G_values[t:] * np.conj(G_values[:n_times-t]))
    
    connected = correlation - np.abs(mean_G)**2
    
    return connected
