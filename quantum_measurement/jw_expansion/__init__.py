"""
jw_expansion
============

Jordan-Wigner expansion simulation for multi-qubit systems with both
Hamiltonian dynamics and measurement noise using correlation matrix formalism.

Simulators:
- TwoQubitCorrelationSimulator: Specialized 2-qubit implementation
- MultiQubitCorrelationSimulator: Generalized L-qubit implementation
"""

from .two_qubit_correlation_simulator import TwoQubitCorrelationSimulator
from .l_qubit_correlation_simulator import LQubitCorrelationSimulator

__all__ = ['TwoQubitCorrelationSimulator', 'LQubitCorrelationSimulator']
