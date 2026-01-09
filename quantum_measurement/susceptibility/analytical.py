"""
analytical.py
=============

Analytical susceptibility solutions for validation of numerical calculations.

This module provides analytical formulas for magnetic susceptibility in simple
quantum systems that can be used to validate numerical simulations.

Functions
---------
single_qubit_susceptibility : Analytical χ for single qubit in thermal equilibrium
transverse_ising_susceptibility : Analytical χ for transverse-field Ising model
two_qubit_xx_susceptibility : Analytical χ for two-qubit XX model
"""

import numpy as np
from typing import Optional


def single_qubit_susceptibility(
    temperature: float,
    field: float = 0.0,
) -> float:
    """Analytical susceptibility for a single qubit in thermal equilibrium.
    
    For a single spin-1/2 in a magnetic field h at temperature T:
        H = -h σᶻ
        Z = 2 cosh(h/T)
        ⟨σᶻ⟩ = tanh(h/T)
        χ = ∂⟨σᶻ⟩/∂h = sech²(h/T) / T
    
    At zero field (h=0):
        χ = 1 / T
    
    Parameters
    ----------
    temperature : float
        Temperature T (in energy units)
    field : float, optional
        Applied magnetic field h (default: 0)
    
    Returns
    -------
    float
        Magnetic susceptibility χ
    
    Notes
    -----
    This is the equilibrium susceptibility. For quantum measurement scenarios
    with continuous monitoring, the effective susceptibility may differ due to
    measurement backaction.
    
    Examples
    --------
    >>> chi = single_qubit_susceptibility(temperature=1.0)
    >>> print(f"χ(T=1, h=0) = {chi}")  # 1.0
    >>> chi_high_T = single_qubit_susceptibility(temperature=10.0)
    >>> print(f"χ(T=10) = {chi_high_T}")  # 0.1
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    if field == 0.0:
        # Zero-field susceptibility
        return 1.0 / temperature
    else:
        # Finite-field susceptibility
        beta = 1.0 / temperature
        chi = beta / np.cosh(beta * field)**2
        return chi


def transverse_ising_susceptibility_high_T(
    temperature: float,
    J: float = 1.0,
) -> float:
    """High-temperature susceptibility for transverse-field Ising model.
    
    For the Hamiltonian H = -J Σᵢ σᵢˣσᵢ₊₁ˣ - h Σᵢ σᵢᶻ
    In the high-temperature limit (T >> J), the susceptibility is:
        χ ≈ 1/T + J²/(2T³) + O(J⁴/T⁵)
    
    Parameters
    ----------
    temperature : float
        Temperature T
    J : float
        Coupling strength
    
    Returns
    -------
    float
        Longitudinal susceptibility χᶻᶻ
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    chi = 1.0 / temperature + (J**2) / (2 * temperature**3)
    return chi


def two_qubit_xx_analytical_correlation(
    t: float,
    J: float,
) -> np.ndarray:
    """Analytical time evolution of correlations for two-qubit XX model.
    
    For H = J σ₁ˣσ₂ˣ with initial state |↑↑⟩:
        ⟨σ₁ᶻ(t)⟩ = ⟨σ₂ᶻ(t)⟩ = cos(2Jt)
        ⟨σ₁ᶻ(t)σ₂ᶻ(t)⟩ = cos²(2Jt)
    
    Parameters
    ----------
    t : float or np.ndarray
        Time or array of times
    J : float
        Coupling strength
    
    Returns
    -------
    np.ndarray
        Array [⟨σ₁ᶻ⟩, ⟨σ₂ᶻ⟩, ⟨σ₁ᶻσ₂ᶻ⟩] at time t
    """
    z1 = np.cos(2 * J * t)
    z2 = np.cos(2 * J * t)
    z1z2 = np.cos(2 * J * t)**2
    
    return np.array([z1, z2, z1z2])


def zero_temperature_susceptibility(
    omega: np.ndarray,
    J: float = 1.0,
) -> np.ndarray:
    """Zero-temperature dynamic susceptibility for free fermions.
    
    For a free-fermion system at T=0, the dynamic susceptibility has
    a simple form related to the density of states.
    
    This is a simplified model for demonstration.
    
    Parameters
    ----------
    omega : np.ndarray
        Frequency array
    J : float
        Energy scale
    
    Returns
    -------
    np.ndarray
        χ''(ω) - imaginary part of dynamic susceptibility
    """
    # Simple Lorentzian for demonstration
    gamma = 0.1 * J  # Broadening
    chi_imag = gamma / (omega**2 + gamma**2)
    
    return chi_imag


def analytical_connected_correlation_decay(
    t: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Analytical exponential decay of connected correlation for damped system.
    
    For a measurement-induced damped system:
        C(t) = C(0) * exp(-γt)
    
    where γ is the measurement-induced dephasing rate.
    
    Parameters
    ----------
    t : np.ndarray
        Time array
    gamma : float
        Dephasing/measurement rate
    
    Returns
    -------
    np.ndarray
        Connected correlation function C(t)
    
    Notes
    -----
    This applies to systems with continuous weak measurement where the
    correlation function decays exponentially due to measurement backaction.
    """
    return np.exp(-gamma * t)


def analytical_susceptibility_from_decay_rate(
    gamma: float,
) -> float:
    """Analytical susceptibility for exponentially decaying correlations.
    
    For C(t) = C(0) * exp(-γt):
        χ = ∫₀^∞ C(t) dt = C(0) / γ
    
    Assuming C(0) = 1 (normalized correlation at t=0):
        χ = 1 / γ
    
    Parameters
    ----------
    gamma : float
        Decay rate
    
    Returns
    -------
    float
        Static susceptibility χ
    
    Examples
    --------
    >>> gamma = 0.5  # Strong measurement
    >>> chi = analytical_susceptibility_from_decay_rate(gamma)
    >>> print(f"χ = {chi}")  # 2.0
    """
    if gamma <= 0:
        raise ValueError("Decay rate must be positive")
    
    return 1.0 / gamma


def validate_numerical_susceptibility(
    numerical_chi: float,
    analytical_chi: float,
    tolerance: float = 0.01,
) -> bool:
    """Validate numerical susceptibility against analytical result.
    
    Parameters
    ----------
    numerical_chi : float
        Numerically computed susceptibility
    analytical_chi : float
        Analytical/expected susceptibility
    tolerance : float
        Relative error tolerance (default: 1%)
    
    Returns
    -------
    bool
        True if agreement is within tolerance
    
    Examples
    --------
    >>> num_chi = 0.995
    >>> ana_chi = 1.0
    >>> is_valid = validate_numerical_susceptibility(num_chi, ana_chi, tolerance=0.01)
    >>> print(is_valid)  # True (0.5% error < 1%)
    """
    if analytical_chi == 0:
        return abs(numerical_chi) < tolerance
    
    relative_error = abs(numerical_chi - analytical_chi) / abs(analytical_chi)
    
    return relative_error < tolerance
