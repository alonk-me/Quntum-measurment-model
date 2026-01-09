"""
Susceptibility Module
=====================

This module provides functions to calculate magnetic susceptibility for quantum
measurement simulations. Susceptibility measures the linear response of a 
quantum system to external perturbations.

Submodules
----------
static_susceptibility : Static (zero-frequency) susceptibility calculations
correlation_matrix_susceptibility : Susceptibility from free-fermion correlation matrices
analytical : Analytical solutions for validation

Functions
---------
compute_static_susceptibility : Compute static susceptibility from correlations
compute_connected_correlation : Compute connected correlation functions
"""

from quantum_measurement.susceptibility.static_susceptibility import (
    compute_static_susceptibility,
    compute_connected_correlation,
    compute_single_site_susceptibility,
    compute_susceptibility_matrix,
)

from quantum_measurement.susceptibility.correlation_matrix_susceptibility import (
    susceptibility_from_correlation_matrix,
    extract_spin_correlations,
)

__all__ = [
    'compute_static_susceptibility',
    'compute_connected_correlation',
    'compute_single_site_susceptibility',
    'compute_susceptibility_matrix',
    'susceptibility_from_correlation_matrix',
    'extract_spin_correlations',
]
