"""
jw_expansion
============

Jordan-Wigner expansion simulation for multi-qubit systems with both
Hamiltonian dynamics and measurement noise using correlation matrix formalism.

Simulators:
- TwoQubitCorrelationSimulator: Specialized 2-qubit implementation
- MultiQubitCorrelationSimulator: Generalized L-qubit implementation

n_infty Tools:
Tools for computing the steady-state occupation n_∞(g).
* The monitoring strength is parameterised by ``g = γ/(4J)``.  All
  functions expect a real, non‑negative ``g``.  Small positive
  numbers (e.g. ``g=1e-6``) are recommended instead of ``g=0`` to
  avoid division by zero in the integral.
* ``L`` must be an odd integer for the APBC sums.  For PBC sums the
  only requirement is that ``L`` be at least 3 and odd, so that the
  momentum grid contains the correct number of modes.
"""

from .two_qubit_correlation_simulator import TwoQubitCorrelationSimulator
from .multi_qubit_correlation_simulator import MultiQubitCorrelationSimulator
from .n_infty import (
    delta,
    sign_im,
    term_value,
    sum_apbc,
    sum_pbc,
    integral_expr,
    small_g_limit,
    large_g_limit,
)

__all__ = [
    'TwoQubitCorrelationSimulator',
    'MultiQubitCorrelationSimulator',
    'delta',
    'sign_im',
    'term_value',
    'sum_apbc',
    'sum_pbc',
    'integral_expr',
    'small_g_limit',
    'large_g_limit',
]
