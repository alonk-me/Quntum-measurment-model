"""Critical point analysis and finite-size scaling.

This module provides tools for locating critical points from susceptibility
peaks and performing finite-size extrapolation to estimate the thermodynamic
limit critical measurement strength γc.

Functions
---------
find_chi_peak :
    Locate peak of |χₙ(γ)| for a given L
estimate_gamma_c :
    Extrapolate γ_peak(L) → γc as L → ∞
smooth_data :
    Preprocess noisy susceptibility data
fit_scaling_function :
    Fit finite-size scaling ansatz

Theory
------
For a continuous phase transition, the susceptibility χₙ = ∂n/∂γ diverges
at the critical point in the thermodynamic limit. For finite systems, the
divergence is cut off and manifests as a peak at γ_peak(L).

The finite-size scaling ansatz is:

    γ_peak(L) = γc + a·L^(-1/ν) + b·L^(-2/ν) + ...

where:
- γc is the thermodynamic critical point
- ν is the correlation length exponent
- a, b are non-universal coefficients

Common simplified models:
1. Linear: γ_peak(L) = γc - a/L
2. Power law: γ_peak(L) = γc - a/L^b

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> 
>>> # Load susceptibility data
>>> df = pd.read_csv('chi_n_results.csv')
>>> 
>>> # Find peaks for each L
>>> peaks = {}
>>> for L in [9, 17, 33, 65]:
...     df_L = df[df['L'] == L]
...     gamma_peak, error = find_chi_peak(
...         df_L['gamma'].values,
...         df_L['chi_n'].values
...     )
...     peaks[L] = (gamma_peak, error)
>>> 
>>> # Extrapolate to infinite L
>>> L_values = list(peaks.keys())
>>> gamma_peaks = [peaks[L][0] for L in L_values]
>>> result = estimate_gamma_c(L_values, gamma_peaks)
>>> print(f"γc = {result['gamma_c']:.4f} ± {result['gamma_c_error']:.4f}")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, signal, interpolate
from scipy.stats import bootstrap
from typing import Dict, List, Tuple, Optional, Callable, Union


def smooth_data(
    gamma: np.ndarray,
    chi_n: np.ndarray,
    method: str = 'savgol',
    **kwargs
) -> np.ndarray:
    """Preprocess noisy susceptibility data with smoothing.
    
    Parameters
    ----------
    gamma : np.ndarray
        Gamma values
    chi_n : np.ndarray
        Susceptibility values
    method : {'savgol', 'gaussian', 'none'}, optional
        Smoothing method:
        - 'savgol': Savitzky-Golay filter
        - 'gaussian': Gaussian filter
        - 'none': No smoothing
        Default is 'savgol'.
    **kwargs
        Method-specific parameters:
        - window_length, polyorder for 'savgol'
        - sigma for 'gaussian'
        
    Returns
    -------
    np.ndarray
        Smoothed susceptibility values
    """
    if method == 'none':
        return chi_n
    
    elif method == 'savgol':
        window_length = kwargs.get('window_length', 5)
        polyorder = kwargs.get('polyorder', 2)
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        return signal.savgol_filter(chi_n, window_length, polyorder)
    
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return signal.gaussian_filter1d(chi_n, sigma)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def find_chi_peak(
    gamma: np.ndarray,
    chi_n: np.ndarray,
    method: str = 'max',
    smooth: bool = True,
    error_estimate: bool = True
) -> Tuple[float, float]:
    """Find the peak location of |χₙ(γ)|.
    
    Parameters
    ----------
    gamma : np.ndarray
        Gamma values (must be sorted)
    chi_n : np.ndarray
        Susceptibility values
    method : {'max', 'gaussian', 'spline'}, optional
        Peak finding method:
        - 'max': Simple maximum of |χₙ|
        - 'gaussian': Fit Gaussian around maximum
        - 'spline': Spline interpolation + root finding
        Default is 'max'.
    smooth : bool, optional
        Apply smoothing before peak finding. Default is True.
    error_estimate : bool, optional
        Estimate error from local curvature. Default is True.
        
    Returns
    -------
    gamma_peak : float
        Gamma value at peak
    gamma_peak_error : float
        Error estimate (0 if error_estimate=False)
        
    Examples
    --------
    >>> gamma = np.linspace(2, 6, 100)
    >>> chi_n = -0.1 * np.exp(-((gamma - 4.0)/0.5)**2)  # Mock data
    >>> gamma_peak, error = find_chi_peak(gamma, chi_n)
    >>> print(f"Peak at γ = {gamma_peak:.3f} ± {error:.3f}")
    """
    # Work with absolute value (peaks can be negative)
    chi_abs = np.abs(chi_n)
    
    # Apply smoothing if requested
    if smooth:
        chi_abs = smooth_data(gamma, chi_abs, method='savgol')
    
    if method == 'max':
        # Simple maximum
        peak_idx = np.argmax(chi_abs)
        gamma_peak = gamma[peak_idx]
        
        # Error estimate from local width
        if error_estimate:
            # Find half-maximum points
            half_max = chi_abs[peak_idx] / 2.0
            # Search window around peak
            window = 5
            start = max(0, peak_idx - window)
            end = min(len(gamma), peak_idx + window + 1)
            
            # Simple width estimate
            gamma_peak_error = (gamma[end-1] - gamma[start]) / 4.0
        else:
            gamma_peak_error = 0.0
            
    elif method == 'gaussian':
        # Fit Gaussian around maximum
        peak_idx = np.argmax(chi_abs)
        
        # Select window around peak
        window = min(10, len(gamma) // 4)
        start = max(0, peak_idx - window)
        end = min(len(gamma), peak_idx + window + 1)
        
        gamma_fit = gamma[start:end]
        chi_fit = chi_abs[start:end]
        
        # Gaussian model: A * exp(-(x-mu)^2 / (2*sigma^2))
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-((x - mu)**2) / (2 * sigma**2))
        
        # Initial guess
        p0 = [chi_abs[peak_idx], gamma[peak_idx], 0.5]
        
        try:
            popt, pcov = optimize.curve_fit(gaussian, gamma_fit, chi_fit, p0=p0)
            gamma_peak = popt[1]
            gamma_peak_error = np.sqrt(pcov[1, 1]) if error_estimate else 0.0
        except:
            # Fallback to simple max
            gamma_peak = gamma[peak_idx]
            gamma_peak_error = 0.0
            
    elif method == 'spline':
        # Cubic spline interpolation
        spline = interpolate.CubicSpline(gamma, chi_abs)
        
        # Find maximum by finding root of derivative
        spline_deriv = spline.derivative()
        
        # Get rough peak location
        peak_idx = np.argmax(chi_abs)
        gamma_guess = gamma[peak_idx]
        
        # Find root near guess
        try:
            # Search in window around guess
            bounds = (gamma[max(0, peak_idx-5)], gamma[min(len(gamma)-1, peak_idx+5)])
            result = optimize.minimize_scalar(
                lambda x: -spline(x),
                bounds=bounds,
                method='bounded'
            )
            gamma_peak = result.x
            
            # Error from second derivative
            if error_estimate:
                spline_deriv2 = spline.derivative(2)
                curvature = abs(spline_deriv2(gamma_peak))
                gamma_peak_error = 1.0 / np.sqrt(curvature) if curvature > 0 else 0.1
            else:
                gamma_peak_error = 0.0
        except:
            # Fallback
            gamma_peak = gamma_guess
            gamma_peak_error = 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(gamma_peak), float(gamma_peak_error)


def estimate_gamma_c(
    L_values: Union[List[int], np.ndarray],
    gamma_peaks: Union[List[float], np.ndarray],
    gamma_peak_errors: Optional[Union[List[float], np.ndarray]] = None,
    model: str = 'linear',
    L_min: int = 17,
    bootstrap_samples: int = 1000
) -> Dict:
    """Extrapolate γ_peak(L) to estimate γc in thermodynamic limit.
    
    Fits a finite-size scaling model and extrapolates to L → ∞.
    
    Parameters
    ----------
    L_values : array_like
        System sizes
    gamma_peaks : array_like
        Peak locations for each L
    gamma_peak_errors : array_like, optional
        Errors on peak locations (used for weighted fit)
    model : {'linear', 'power', 'custom'}, optional
        Scaling model:
        - 'linear': γ_peak = γc - a/L
        - 'power': γ_peak = γc - a/L^b
        - 'custom': user-provided function
        Default is 'linear'.
    L_min : int, optional
        Minimum L to include in fit. Default is 17.
    bootstrap_samples : int, optional
        Number of bootstrap samples for error estimation. Default is 1000.
        
    Returns
    -------
    dict
        Fit results containing:
        - 'gamma_c' : float - Extrapolated critical point
        - 'gamma_c_error' : float - Error estimate
        - 'fit_params' : dict - All fitted parameters
        - 'fit_params_cov' : ndarray - Covariance matrix
        - 'L_used' : ndarray - L values used in fit
        - 'gamma_peaks_used' : ndarray - Corresponding peaks
        - 'r_squared' : float - Goodness of fit
        - 'residuals' : ndarray - Fit residuals
        
    Examples
    --------
    >>> L_values = [9, 17, 33, 65, 129]
    >>> gamma_peaks = [4.5, 4.3, 4.15, 4.08, 4.03]
    >>> result = estimate_gamma_c(L_values, gamma_peaks, model='linear')
    >>> print(f"γc = {result['gamma_c']:.4f} ± {result['gamma_c_error']:.4f}")
    """
    L_values = np.asarray(L_values)
    gamma_peaks = np.asarray(gamma_peaks)
    
    # Filter by L_min
    mask = L_values >= L_min
    L_fit = L_values[mask]
    gamma_fit = gamma_peaks[mask]
    
    if len(L_fit) < 2:
        raise ValueError(f"Need at least 2 L values >= {L_min} for fitting")
    
    # Set up weights
    if gamma_peak_errors is not None:
        gamma_peak_errors = np.asarray(gamma_peak_errors)[mask]
        sigma = gamma_peak_errors
    else:
        sigma = None
    
    # Define fitting functions
    if model == 'linear':
        # γ_peak(L) = γc - a/L
        def fit_func(L, gamma_c, a):
            return gamma_c - a / L
        p0 = [4.0, 5.0]  # Initial guess
        
    elif model == 'power':
        # γ_peak(L) = γc - a/L^b
        def fit_func(L, gamma_c, a, b):
            return gamma_c - a / (L**b)
        p0 = [4.0, 5.0, 1.0]  # Initial guess
        
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Perform fit
    try:
        popt, pcov = optimize.curve_fit(
            fit_func,
            L_fit,
            gamma_fit,
            p0=p0,
            sigma=sigma,
            absolute_sigma=True if sigma is not None else False
        )
    except Exception as e:
        raise RuntimeError(f"Fit failed: {e}")
    
    # Extract γc and error
    gamma_c = popt[0]
    gamma_c_error = np.sqrt(pcov[0, 0])
    
    # Compute R²
    y_pred = fit_func(L_fit, *popt)
    residuals = gamma_fit - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((gamma_fit - gamma_fit.mean())**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Store fit parameters
    if model == 'linear':
        fit_params = {'gamma_c': popt[0], 'a': popt[1]}
    elif model == 'power':
        fit_params = {'gamma_c': popt[0], 'a': popt[1], 'b': popt[2]}
    
    return {
        'gamma_c': float(gamma_c),
        'gamma_c_error': float(gamma_c_error),
        'fit_params': fit_params,
        'fit_params_cov': pcov,
        'L_used': L_fit,
        'gamma_peaks_used': gamma_fit,
        'r_squared': float(r_squared),
        'residuals': residuals,
        'model': model
    }


def fit_scaling_function(
    L_values: np.ndarray,
    gamma_peaks: np.ndarray,
    func: Callable,
    p0: List[float],
    **fit_kwargs
) -> Dict:
    """Fit custom finite-size scaling function.
    
    Parameters
    ----------
    L_values : np.ndarray
        System sizes
    gamma_peaks : np.ndarray
        Peak locations
    func : callable
        Scaling function with signature func(L, *params)
    p0 : list
        Initial parameter guess
    **fit_kwargs
        Additional arguments for curve_fit
        
    Returns
    -------
    dict
        Fit results (similar to estimate_gamma_c)
    """
    popt, pcov = optimize.curve_fit(func, L_values, gamma_peaks, p0=p0, **fit_kwargs)
    
    y_pred = func(L_values, *popt)
    residuals = gamma_peaks - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((gamma_peaks - gamma_peaks.mean())**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return {
        'fit_params': popt,
        'fit_params_cov': pcov,
        'r_squared': float(r_squared),
        'residuals': residuals
    }
