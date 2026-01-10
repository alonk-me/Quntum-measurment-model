"""Analysis modules for quantum measurement simulations.

This package provides analysis tools for computing physical observables
and analyzing critical behavior in monitored quantum systems.

Modules
-------
steady_state :
    Compute steady-state occupations n_∞(γ,L) with adaptive convergence
critical_point :
    Find critical points and perform finite-size scaling analysis
"""

from .steady_state import compute_n_inf, compute_n_inf_with_error

__all__ = [
    'compute_n_inf',
    'compute_n_inf_with_error',
]
