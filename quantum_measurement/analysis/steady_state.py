"""Steady-state occupation computation for monitored free-fermion chains.

This module provides a high-level API for computing the steady-state occupation
n_∞(γ, L) using adaptive convergence detection. It extracts and enhances the
core simulation logic from run_ninf_scan.py to enable reuse in susceptibility
calculations and other analyses.

Functions
---------
compute_n_inf :
    Compute steady-state occupation for given (γ, L) with convergence diagnostics
get_adaptive_max_steps :
    Determine max simulation steps based on system parameters
estimate_memory_gb :
    Estimate memory requirements for a simulation

Examples
--------
>>> result = compute_n_inf(gamma=4.0, L=17)
>>> print(f"n_inf = {result['n_infinity']:.6f}")
>>> print(f"Converged: {result['diagnostics']['converged']}")
"""

from __future__ import annotations

import numpy as np
import time
import gc
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

# Import quantum measurement modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from quantum_measurement.jw_expansion.non_hermitian_hat import NonHermitianHatSimulator

# Default convergence parameters
DEFAULT_TOLERANCE = 1e-4
DEFAULT_WINDOW_SIZE = 1000
DEFAULT_DT = 0.001


def get_adaptive_max_steps(
    L: int,
    gamma: float,
    dt: float = DEFAULT_DT
) -> int:
    """Calculate max_steps based on system size and measurement strength.
    
    The relaxation time scales as τ ~ 2/γ. Larger systems need more steps
    due to longer correlation times. This function uses a logarithmic
    correction to avoid excessive scaling while ensuring convergence.
    
    Parameters
    ----------
    L : int
        System size (number of sites)
    gamma : float
        Measurement rate
    dt : float, optional
        Time step for integration. Default is 0.001.
        
    Returns
    -------
    int
        Maximum number of steps to allow
        
    Notes
    -----
    The estimate is based on:
    - Base time: 20 relaxation times (20/γ)
    - Size correction: logarithmic factor 1 + 0.2*log(L/9)
    - Bounds: minimum 30k steps, maximum 500k steps
    """
    # Base estimate: 20 relaxation times
    base_time = 20.0 / gamma  # Relaxation time in J^-1 units
    base_steps = int(base_time / dt)
    
    # Size correction (logarithmic to avoid explosion)
    size_factor = 1.0 + 0.2 * np.log(max(L / 9.0, 1.0))
    
    # Apply correction and bound
    max_steps = int(base_steps * size_factor)
    
    # Safety bounds
    max_steps = max(30000, min(max_steps, 500000))
    
    return max_steps


def estimate_memory_gb(
    L: int,
    gamma: float = 0.1,
    dt: float = DEFAULT_DT
) -> float:
    """Estimate peak memory usage for time evolution of size L.
    
    The correlation matrix is 2L×2L complex128, and we store trajectory
    of length ~max_steps. Main memory costs:
    - G matrix: (2L)² × 16 bytes
    - n_traj: max_steps × L × 8 bytes
    - Working memory: ~3× matrix size for operations
    
    Parameters
    ----------
    L : int
        Chain length
    gamma : float, optional
        Measurement rate (used for adaptive max_steps estimate). Default is 0.1.
    dt : float, optional
        Time step. Default is 0.001.
        
    Returns
    -------
    float
        Estimated peak memory in GB
    """
    matrix_size_gb = (2 * L)**2 * 16 / (1024**3)  # Complex128 = 16 bytes
    max_steps_estimate = get_adaptive_max_steps(L, gamma, dt)
    trajectory_size_gb = max_steps_estimate * L * 8 / (1024**3)  # Float64 = 8 bytes
    working_memory_gb = matrix_size_gb * 3  # Temporary arrays in evolution
    
    total_gb = matrix_size_gb + trajectory_size_gb + working_memory_gb
    return total_gb


def compute_n_inf(
    gamma: float,
    L: int,
    J: float = 1.0,
    dt: float = DEFAULT_DT,
    tolerance: float = DEFAULT_TOLERANCE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    max_steps: Optional[int] = None,
    return_trajectory: bool = False,
    return_G_final: bool = False
) -> Dict:
    """Compute steady-state occupation n_∞(γ, L) with convergence diagnostics.
    
    This function runs a non-Hermitian time evolution starting from the vacuum
    state until the system reaches steady state. It uses adaptive convergence
    detection and returns comprehensive diagnostics.
    
    Parameters
    ----------
    gamma : float
        Measurement rate (monitoring strength)
    L : int
        System size (number of sites)
    J : float, optional
        Hopping coupling constant. Default is 1.0.
    dt : float, optional
        Time step for Euler integration. Default is 0.001.
    tolerance : float, optional
        Convergence threshold for occupation change. Default is 1e-4.
    window_size : int, optional
        Window size for convergence detection. Default is 1000.
    max_steps : int, optional
        Maximum number of time steps. If None, determined adaptively.
    return_trajectory : bool, optional
        If True, include full trajectory in output. Default is False.
    return_G_final : bool, optional
        If True, include final correlation matrix. Default is False.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'n_infinity' : float
            Steady-state occupation (average over final window)
        - 'diagnostics' : dict
            Convergence diagnostics:
            - 'converged' : bool - Whether convergence was achieved
            - 'steps' : int - Total steps taken
            - 'convergence_step' : int - Step where convergence occurred
            - 't_sat' : float - Saturation time (convergence_step * dt)
            - 'final_norm' : float - Final correlation matrix trace
            - 'final_hermiticity_error' : float - Max |G - G†|
            - 'max_steps_allocated' : int - Maximum steps allowed
            - 'runtime_sec' : float - Wall-clock time
        - 'n_traj' : ndarray (if return_trajectory=True)
            Trajectory of occupations, shape (steps+1, L)
        - 'G_final' : ndarray (if return_G_final=True)
            Final correlation matrix, shape (2L, 2L)
            
    Examples
    --------
    >>> # Basic usage
    >>> result = compute_n_inf(gamma=4.0, L=17)
    >>> n_inf = result['n_infinity']
    >>> converged = result['diagnostics']['converged']
    
    >>> # With full diagnostics
    >>> result = compute_n_inf(gamma=4.0, L=17, return_trajectory=True)
    >>> t_sat = result['diagnostics']['t_sat']
    >>> n_traj = result['n_traj']
    
    Notes
    -----
    The simulation uses periodic boundary conditions (closed_boundary=True)
    to match the analytical formulas in n_infty.py (sum_pbc).
    """
    start_time = time.time()
    
    # Determine max_steps adaptively if not provided
    if max_steps is None:
        max_steps = get_adaptive_max_steps(L, gamma, dt)
    
    # Initialize simulator with periodic BC
    sim = NonHermitianHatSimulator(
        L=L,
        J=J,
        gamma=gamma,
        dt=dt,
        N_steps=max_steps,
        closed_boundary=True  # Periodic BC for sum_pbc comparison
    )
    
    # Run simulation
    Q_total, n_traj, G_final = sim.simulate_trajectory(return_G_final=True)
    
    # Post-hoc convergence detection
    converged = False
    convergence_step = max_steps
    
    for step in range(window_size, len(n_traj)):
        n_recent = n_traj[step - window_size//2:step, :].mean(axis=0)
        n_previous = n_traj[step - window_size:step - window_size//2, :].mean(axis=0)
        max_diff = np.max(np.abs(n_recent - n_previous))
        
        if max_diff < tolerance:
            converged = True
            convergence_step = step
            break
    
    # Extract steady-state occupation from final window
    n_infinity = n_traj[-window_size:, :].mean()
    
    # Calculate additional diagnostics
    t_sat = convergence_step * dt
    final_norm = np.trace(G_final).real
    final_hermiticity_error = np.max(np.abs(G_final - G_final.conj().T))
    runtime = time.time() - start_time
    
    # Build diagnostics dict
    diagnostics = {
        'converged': bool(converged),
        'steps': len(n_traj) - 1,
        'convergence_step': int(convergence_step),
        't_sat': float(t_sat),
        'final_norm': float(final_norm),
        'final_hermiticity_error': float(final_hermiticity_error),
        'max_steps_allocated': int(max_steps),
        'runtime_sec': float(runtime)
    }
    
    # Build result dictionary
    result = {
        'n_infinity': float(n_infinity),
        'diagnostics': diagnostics
    }
    
    # Add optional returns
    if return_trajectory:
        result['n_traj'] = n_traj
    if return_G_final:
        result['G_final'] = G_final
    
    # Clean up
    gc.collect()
    
    return result


def compute_n_inf_with_error(
    gamma: float,
    L: int,
    **kwargs
) -> Tuple[float, float]:
    """Compute n_infinity with error estimate.
    
    This is a convenience wrapper that returns (n_infinity, error_estimate)
    where the error is estimated from the fluctuations in the final window.
    
    Parameters
    ----------
    gamma : float
        Measurement rate
    L : int
        System size
    **kwargs
        Additional arguments passed to compute_n_inf
        
    Returns
    -------
    n_infinity : float
        Steady-state occupation
    error : float
        Standard error estimate from final window
    """
    # Force return of trajectory to compute error
    kwargs['return_trajectory'] = True
    result = compute_n_inf(gamma, L, **kwargs)
    
    # Get window size
    window_size = kwargs.get('window_size', DEFAULT_WINDOW_SIZE)
    
    # Compute error from final window
    n_traj = result['n_traj']
    final_window = n_traj[-window_size:, :]
    n_infinity = result['n_infinity']
    error = np.std(final_window.mean(axis=1)) / np.sqrt(window_size)
    
    return n_infinity, error
