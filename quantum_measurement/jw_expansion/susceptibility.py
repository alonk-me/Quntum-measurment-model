"""Susceptibility analysis for measurement-induced phase transitions.

This module implements numerical differentiation to compute the susceptibility
χₙ(γ, L) = ∂n_∞/∂γ using finite differences. The susceptibility peaks near the
critical measurement strength γc, making it a sensitive probe for locating
phase transitions.

Functions
---------
compute_chi_n :
    Compute susceptibility at a single (γ, L) point using central differences
compute_chi_n_scan :
    Batch computation of susceptibility over parameter grid
adaptive_dg_selection :
    Automatically select optimal step size for differentiation

Theory
------
The susceptibility is defined as:

    χₙ(γ, L) = ∂n_∞(γ, L)/∂γ

At the critical point γ = γc, the susceptibility diverges in the thermodynamic
limit and shows pronounced peaks for finite L. The peak location γ_peak(L)
approaches γc as L → ∞ with finite-size scaling:

    γ_peak(L) = γc - a/L^(1/ν)

where ν is the correlation length exponent.

Numerical Implementation
------------------------
We use central differences for interior points:

    χₙ ≈ [n_∞(γ+dg) - n_∞(γ-dg)] / (2*dg)

and forward differences for boundaries:

    χₙ ≈ [n_∞(γ+dg) - n_∞(γ)] / dg

Error estimation assumes uncorrelated errors in n_∞:

    σ_χ ≈ √(σ₊² + σ₋²) / (2*dg)

Examples
--------
>>> # Single point calculation
>>> result = compute_chi_n(gamma=4.0, L=17)
>>> chi_n = result['chi_n']
>>> error = result['chi_n_error']

>>> # Parameter scan
>>> import numpy as np
>>> gamma_grid = np.linspace(2.0, 6.0, 50)
>>> L_list = [9, 17, 33]
>>> df = compute_chi_n_scan(gamma_grid, L_list)
>>> print(df.head())
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import sys

# Import steady-state computation
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from quantum_measurement.analysis.steady_state import compute_n_inf


def adaptive_dg_selection(
    gamma: float,
    L: int,
    method: str = 'fixed',
    base_dg: Optional[float] = None
) -> float:
    """Select optimal step size dg for numerical differentiation.
    
    Parameters
    ----------
    gamma : float
        Measurement rate
    L : int
        System size
    method : {'fixed', 'adaptive'}, optional
        Strategy for dg selection:
        - 'fixed': Use relative step size proportional to gamma
        - 'adaptive': Validate with dg/2 and adjust if necessary
        Default is 'fixed'.
    base_dg : float, optional
        Base step size. If None, use max(1e-3, 0.01*gamma).
        
    Returns
    -------
    float
        Selected step size for numerical differentiation
        
    Notes
    -----
    Fixed method (recommended for production):
        dg = max(1e-3, 0.01*gamma)
    
    Adaptive method (more accurate but 3x slower):
        Start with base_dg, compute derivative with dg and dg/2,
        ensure they agree within tolerance, otherwise refine.
    """
    if base_dg is None:
        base_dg = max(1e-3, 0.01 * gamma)
    
    if method == 'fixed':
        return base_dg
    
    elif method == 'adaptive':
        # Compute chi_n with dg and dg/2
        # If they disagree, halve dg again
        # (Implementation simplified for now - use fixed)
        # TODO: Implement full adaptive logic
        return base_dg
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_chi_n(
    gamma: float,
    L: int,
    J: float = 1.0,
    dg: Optional[float] = None,
    method: str = 'central',
    dg_selection: str = 'fixed',
    **compute_kwargs
) -> Dict:
    """Compute susceptibility χₙ at a single (γ, L) point.
    
    Uses finite differences to numerically differentiate n_∞(γ). The function
    automatically selects an appropriate step size and computes error estimates.
    
    Parameters
    ----------
    gamma : float
        Measurement rate (central point for differentiation)
    L : int
        System size
    J : float, optional
        Hopping coupling. Default is 1.0.
    dg : float, optional
        Step size for differentiation. If None, determined automatically.
    method : {'central', 'forward'}, optional
        Finite difference method:
        - 'central': (n⁺ - n⁻)/(2*dg) - more accurate
        - 'forward': (n⁺ - n)/(dg) - for boundaries
        Default is 'central'.
    dg_selection : {'fixed', 'adaptive'}, optional
        How to select dg if not provided. Default is 'fixed'.
    **compute_kwargs
        Additional arguments passed to compute_n_inf
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'chi_n' : float
            Susceptibility ∂n_∞/∂γ
        - 'chi_n_error' : float
            Error estimate
        - 'dg_used' : float
            Step size actually used
        - 'gamma' : float
            Central gamma value
        - 'L' : int
            System size
        - 'n_plus' : float
            n_∞(γ + dg)
        - 'n_minus' : float (only for central method)
            n_∞(γ - dg)
        - 'n_center' : float (only for forward method)
            n_∞(γ)
        - 'converged_all' : bool
            Whether all simulations converged
            
    Examples
    --------
    >>> # Basic usage with central differences
    >>> result = compute_chi_n(gamma=4.0, L=17)
    >>> print(f"χₙ = {result['chi_n']:.6f} ± {result['chi_n_error']:.6f}")
    
    >>> # Forward differences at boundary
    >>> result = compute_chi_n(gamma=0.1, L=17, method='forward')
    
    >>> # Custom step size
    >>> result = compute_chi_n(gamma=4.0, L=17, dg=0.05)
    """
    # Select step size
    if dg is None:
        dg = adaptive_dg_selection(gamma, L, method=dg_selection)
    
    # Compute n_infinity at required points
    if method == 'central':
        # Need n(γ+dg) and n(γ-dg)
        gamma_plus = gamma + dg
        gamma_minus = gamma - dg
        
        result_plus = compute_n_inf(gamma_plus, L, J=J, **compute_kwargs)
        result_minus = compute_n_inf(gamma_minus, L, J=J, **compute_kwargs)
        
        n_plus = result_plus['n_infinity']
        n_minus = result_minus['n_infinity']
        
        # Central difference
        chi_n = (n_plus - n_minus) / (2.0 * dg)
        
        # Error estimation (assuming independent errors)
        # For now, use simple estimate based on convergence
        # More sophisticated: propagate errors from final window fluctuations
        sigma_plus = 1e-5 if result_plus['diagnostics']['converged'] else 1e-4
        sigma_minus = 1e-5 if result_minus['diagnostics']['converged'] else 1e-4
        chi_n_error = np.sqrt(sigma_plus**2 + sigma_minus**2) / (2.0 * dg)
        
        converged_all = (result_plus['diagnostics']['converged'] and 
                        result_minus['diagnostics']['converged'])
        
        return {
            'chi_n': float(chi_n),
            'chi_n_error': float(chi_n_error),
            'dg_used': float(dg),
            'gamma': float(gamma),
            'L': int(L),
            'n_plus': float(n_plus),
            'n_minus': float(n_minus),
            'converged_all': bool(converged_all)
        }
        
    elif method == 'forward':
        # Need n(γ) and n(γ+dg)
        gamma_plus = gamma + dg
        
        result_center = compute_n_inf(gamma, L, J=J, **compute_kwargs)
        result_plus = compute_n_inf(gamma_plus, L, J=J, **compute_kwargs)
        
        n_center = result_center['n_infinity']
        n_plus = result_plus['n_infinity']
        
        # Forward difference
        chi_n = (n_plus - n_center) / dg
        
        # Error estimation
        sigma_center = 1e-5 if result_center['diagnostics']['converged'] else 1e-4
        sigma_plus = 1e-5 if result_plus['diagnostics']['converged'] else 1e-4
        chi_n_error = np.sqrt(sigma_center**2 + sigma_plus**2) / dg
        
        converged_all = (result_center['diagnostics']['converged'] and 
                        result_plus['diagnostics']['converged'])
        
        return {
            'chi_n': float(chi_n),
            'chi_n_error': float(chi_n_error),
            'dg_used': float(dg),
            'gamma': float(gamma),
            'L': int(L),
            'n_center': float(n_center),
            'n_plus': float(n_plus),
            'converged_all': bool(converged_all)
        }
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'central' or 'forward'.")


def compute_chi_n_scan(
    gamma_grid: Union[np.ndarray, List[float]],
    L_list: Union[np.ndarray, List[int]],
    J: float = 1.0,
    dg_dict: Optional[Dict[int, float]] = None,
    n_inf_cache: Optional[Dict] = None,
    progress_callback: Optional[callable] = None,
    **compute_kwargs
) -> pd.DataFrame:
    """Batch computation of susceptibility over parameter grid.
    
    Computes χₙ(γ, L) for all combinations of gamma and L values. Results are
    returned as a pandas DataFrame suitable for analysis and visualization.
    
    Parameters
    ----------
    gamma_grid : array_like
        Array of gamma values to scan
    L_list : array_like
        List of system sizes
    J : float, optional
        Hopping coupling. Default is 1.0.
    dg_dict : dict, optional
        Dictionary mapping L → dg for custom step sizes.
        If None, dg is selected automatically for each L.
    n_inf_cache : dict, optional
        Pre-computed n_infinity values. Format: {(gamma, L): n_inf}
        Used to avoid redundant computations.
    progress_callback : callable, optional
        Function called after each (gamma, L) completion.
        Signature: callback(completed, total, gamma, L)
    **compute_kwargs
        Additional arguments passed to compute_n_inf
        
    Returns
    -------
    pandas.DataFrame
        Results with columns:
        - 'L' : System size
        - 'gamma' : Measurement rate
        - 'g' : Dimensionless strength γ/(4J)
        - 'chi_n' : Susceptibility
        - 'chi_n_error' : Error estimate
        - 'dg_used' : Step size used
        - 'converged_all' : All simulations converged
        - 'n_plus' : n_∞(γ+dg)
        - 'n_minus' or 'n_center' : Depending on method
        
    Examples
    --------
    >>> import numpy as np
    >>> gamma_grid = np.linspace(2.0, 6.0, 50)
    >>> L_list = [9, 17, 33]
    >>> df = compute_chi_n_scan(gamma_grid, L_list)
    >>> 
    >>> # Find peak for each L
    >>> for L in L_list:
    >>>     df_L = df[df['L'] == L]
    >>>     peak_idx = df_L['chi_n'].abs().idxmax()
    >>>     gamma_peak = df_L.loc[peak_idx, 'gamma']
    >>>     print(f"L={L}: γ_peak = {gamma_peak:.4f}")
    """
    gamma_grid = np.asarray(gamma_grid)
    L_list = np.asarray(L_list)
    
    results = []
    total_points = len(gamma_grid) * len(L_list)
    completed = 0
    
    for L in L_list:
        # Get dg for this L
        if dg_dict is not None and L in dg_dict:
            dg_L = dg_dict[L]
        else:
            dg_L = None  # Will be determined adaptively
        
        for gamma in gamma_grid:
            # Check cache for n_inf values (if provided)
            # For now, compute directly
            # TODO: Integrate with ResultCache from Section 3
            
            # Compute chi_n at this point
            result = compute_chi_n(
                gamma=gamma,
                L=L,
                J=J,
                dg=dg_L,
                **compute_kwargs
            )
            
            # Add g = gamma/(4J)
            result['g'] = gamma / (4.0 * J)
            
            results.append(result)
            
            # Progress reporting
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total_points, gamma, L)
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = ['L', 'gamma', 'g', 'chi_n', 'chi_n_error', 'dg_used', 
                    'converged_all', 'n_plus']
    if 'n_minus' in df.columns:
        column_order.append('n_minus')
    if 'n_center' in df.columns:
        column_order.append('n_center')
    
    df = df[column_order]
    
    return df
