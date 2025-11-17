"""
jw_expansion
============

Jordan-Wigner expansion simulation for 2-qubit systems with both
Hamiltonian dynamics and measurement noise using correlation matrix formalism.
"""

from .two_qubit_correlation_simulator import TwoQubitCorrelationSimulator

__all__ = ['TwoQubitCorrelationSimulator']
