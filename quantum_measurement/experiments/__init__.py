"""Experiments module for quantum measurement simulations.

This module provides high-level experiment runners that wrap the underlying
simulators (Kraus operators, SSE) and return structured results suitable for
analysis and plotting.

Available experiment runners
----------------------------
- :func:`run_krauss_experiment` – Kraus operator Monte-Carlo experiment.
- :func:`run_sse_experiment` – Stochastic Schrödinger Equation experiment.
"""

from quantum_measurement.experiments.krauss_experiment import (
    KraussExperimentConfig,
    KraussExperimentResult,
    run_krauss_experiment,
)
from quantum_measurement.experiments.sse_experiment import (
    SSEExperimentConfig,
    SSEExperimentResult,
    run_sse_experiment,
)

__all__ = [
    "KraussExperimentConfig",
    "KraussExperimentResult",
    "run_krauss_experiment",
    "SSEExperimentConfig",
    "SSEExperimentResult",
    "run_sse_experiment",
]
