"""
static_susceptibility.py
========================

This module provides functions to calculate static (zero-frequency) magnetic
susceptibility for quantum systems. The static susceptibility χ measures the
linear response of magnetization to an applied field:

    χ = ∂⟨M⟩/∂h |_{h=0}

For quantum spin systems, this is related to equal-time correlation functions:

    χ = ∫ dt [⟨σᵢᶻ(t)σⱼᶻ(0)⟩ - ⟨σᵢᶻ(t)⟩⟨σⱼᶻ(0)⟩]

Functions
---------
compute_connected_correlation : Compute connected correlation function
compute_static_susceptibility : Compute integrated static susceptibility
compute_single_site_susceptibility : Single-site susceptibility calculation
"""

import numpy as np
from typing import Tuple, Optional


def compute_connected_correlation(
    sigma_i: np.ndarray,
    sigma_j: np.ndarray,
) -> np.ndarray:
    """Compute the connected (cumulant) correlation function.
    
    The connected correlation function is defined as:
        C_ij(t) = ⟨σᵢ(t)σⱼ(0)⟩ - ⟨σᵢ(t)⟩⟨σⱼ(0)⟩
    
    This removes the contribution from independent fluctuations and measures
    genuine correlations between the two observables.
    
    Parameters
    ----------
    sigma_i : np.ndarray
        Time series of expectation values ⟨σᵢ(t)⟩ of shape (n_times,) or
        trajectory ensemble of shape (n_trajectories, n_times)
    sigma_j : np.ndarray
        Time series of expectation values ⟨σⱼ(t)⟩, same shape as sigma_i
        For equal-time correlation, use sigma_j = sigma_i
    
    Returns
    -------
    np.ndarray
        Connected correlation function. For ensemble data, returns average
        over trajectories.
    
    Notes
    -----
    For a single trajectory: C(t) = ⟨σᵢ(t)⟩⟨σⱼ(0)⟩ - ⟨σᵢ⟩⟨σⱼ⟩
    For ensemble: C(t) = ⟨σᵢ(t)σⱼ(0)⟩_ensemble - ⟨σᵢ⟩⟨σⱼ⟩
    """
    if sigma_i.ndim == 1:
        # Single trajectory case
        mean_i = np.mean(sigma_i)
        mean_j = np.mean(sigma_j)
        
        # Compute correlation by shifting sigma_j
        n_times = len(sigma_i)
        correlation = np.zeros(n_times)
        for t in range(n_times):
            correlation[t] = np.mean(sigma_i[t:] * sigma_j[:n_times-t])
        
        connected = correlation - mean_i * mean_j
        
    else:
        # Ensemble case: (n_trajectories, n_times)
        mean_i = np.mean(sigma_i)
        mean_j = np.mean(sigma_j)
        
        # Compute correlation over ensemble
        n_traj, n_times = sigma_i.shape
        correlation = np.zeros(n_times)
        for t in range(n_times):
            # Average over both time shifts and trajectories
            for traj in range(n_traj):
                correlation[t] += np.mean(sigma_i[traj, t:] * sigma_j[traj, :n_times-t])
        correlation /= n_traj
        
        connected = correlation - mean_i * mean_j
    
    return connected


def compute_static_susceptibility(
    times: np.ndarray,
    sigma_i: np.ndarray,
    sigma_j: Optional[np.ndarray] = None,
) -> float:
    """Compute static susceptibility from time-series data.
    
    The static susceptibility is obtained by integrating the connected
    correlation function over time:
    
        χ_ij = ∫₀^T dt [⟨σᵢ(t)σⱼ(0)⟩ - ⟨σᵢ⟩⟨σⱼ⟩]
    
    Parameters
    ----------
    times : np.ndarray
        Time points array of shape (n_times,)
    sigma_i : np.ndarray
        Time series of ⟨σᵢᶻ(t)⟩ values, shape (n_times,) or (n_trajectories, n_times)
    sigma_j : np.ndarray, optional
        Time series of ⟨σⱼᶻ(t)⟩ values. If None, use sigma_i (autocorrelation)
    
    Returns
    -------
    float
        Static susceptibility χ_ij
    
    Examples
    --------
    >>> times = np.linspace(0, 10, 1000)
    >>> z_trajectory = np.random.randn(1000)  # Example trajectory
    >>> chi = compute_static_susceptibility(times, z_trajectory)
    """
    if sigma_j is None:
        sigma_j = sigma_i
    
    # Compute connected correlation function
    connected_corr = compute_connected_correlation(sigma_i, sigma_j)
    
    # Integrate using trapezoidal rule
    # Handle both 1D and 2D cases
    if sigma_i.ndim == 1:
        chi = np.trapz(connected_corr, times)
    else:
        # Use average times for ensemble
        chi = np.trapz(connected_corr, times)
    
    return chi


def compute_single_site_susceptibility(
    times: np.ndarray,
    z_values: np.ndarray,
) -> float:
    """Compute single-site magnetic susceptibility.
    
    For a single qubit, the susceptibility is:
        χ = ∫ dt [⟨σᶻ(t)σᶻ(0)⟩ - ⟨σᶻ⟩²]
    
    Parameters
    ----------
    times : np.ndarray
        Time points, shape (n_times,)
    z_values : np.ndarray
        Trajectory of ⟨σᶻ(t)⟩ values, shape (n_times,) or (n_trajectories, n_times)
    
    Returns
    -------
    float
        Single-site susceptibility χ
    
    Examples
    --------
    >>> from quantum_measurement.susceptibility import compute_single_site_susceptibility
    >>> times = np.linspace(0, 5, 500)
    >>> z = np.cos(2*times)  # Example oscillating magnetization
    >>> chi = compute_single_site_susceptibility(times, z)
    """
    return compute_static_susceptibility(times, z_values, z_values)


def compute_two_point_susceptibility(
    times: np.ndarray,
    z_i: np.ndarray,
    z_j: np.ndarray,
) -> float:
    """Compute two-point susceptibility between sites i and j.
    
    The two-point susceptibility measures correlations between different sites:
        χ_ij = ∫ dt [⟨σᵢᶻ(t)σⱼᶻ(0)⟩ - ⟨σᵢᶻ⟩⟨σⱼᶻ⟩]
    
    Parameters
    ----------
    times : np.ndarray
        Time points, shape (n_times,)
    z_i : np.ndarray
        Trajectory of ⟨σᵢᶻ(t)⟩ at site i
    z_j : np.ndarray
        Trajectory of ⟨σⱼᶻ(t)⟩ at site j
    
    Returns
    -------
    float
        Two-point susceptibility χ_ij
    """
    return compute_static_susceptibility(times, z_i, z_j)


def compute_susceptibility_matrix(
    times: np.ndarray,
    z_trajectories: np.ndarray,
) -> np.ndarray:
    """Compute the full susceptibility matrix for a multi-site system.
    
    For an L-site system, compute the L×L susceptibility matrix where:
        χ[i,j] = ∫ dt [⟨σᵢᶻ(t)σⱼᶻ(0)⟩ - ⟨σᵢᶻ⟩⟨σⱼᶻ⟩]
    
    Parameters
    ----------
    times : np.ndarray
        Time points, shape (n_times,)
    z_trajectories : np.ndarray
        Array of ⟨σᶻ⟩ trajectories for all sites
        Shape: (L, n_times) for single trajectory
               (n_trajectories, L, n_times) for ensemble
    
    Returns
    -------
    np.ndarray
        Susceptibility matrix of shape (L, L)
    
    Examples
    --------
    >>> times = np.linspace(0, 10, 1000)
    >>> L = 4  # 4 sites
    >>> z_traj = np.random.randn(L, 1000)
    >>> chi_matrix = compute_susceptibility_matrix(times, z_traj)
    >>> print(chi_matrix.shape)  # (4, 4)
    """
    if z_trajectories.ndim == 2:
        # Single trajectory: (L, n_times)
        L = z_trajectories.shape[0]
        chi_matrix = np.zeros((L, L))
        
        for i in range(L):
            for j in range(L):
                chi_matrix[i, j] = compute_two_point_susceptibility(
                    times, z_trajectories[i], z_trajectories[j]
                )
    
    elif z_trajectories.ndim == 3:
        # Ensemble: (n_trajectories, L, n_times)
        n_traj, L, n_times = z_trajectories.shape
        chi_matrix = np.zeros((L, L))
        
        # Average susceptibility over ensemble
        for traj_idx in range(n_traj):
            for i in range(L):
                for j in range(L):
                    chi_matrix[i, j] += compute_two_point_susceptibility(
                        times, z_trajectories[traj_idx, i], z_trajectories[traj_idx, j]
                    )
        chi_matrix /= n_traj
    
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {z_trajectories.shape}")
    
    return chi_matrix
