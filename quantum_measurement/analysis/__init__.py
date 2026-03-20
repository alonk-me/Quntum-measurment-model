"""Analysis module for quantum measurement simulations.

This module consolidates the analysis tools that were previously scattered
across individual scripts into a coherent, importable library.

Available tools
---------------
- :func:`fit_arrow_of_time` – fit Q-distribution to the Arrow-of-Time PDF.
- :func:`compute_statistics` – compute summary statistics for a Q-distribution.
- :func:`plot_Q_distribution` – plot Q histogram with optional theoretical overlay.
- :func:`plot_mean_Q_vs_theta` – plot ⟨Q⟩ as a function of θ.
"""

from quantum_measurement.analysis.distribution_analysis import (
    QStatistics,
    compute_statistics,
    fit_arrow_of_time,
    plot_Q_distribution,
    plot_mean_Q_vs_theta,
)

__all__ = [
    "QStatistics",
    "compute_statistics",
    "fit_arrow_of_time",
    "plot_Q_distribution",
    "plot_mean_Q_vs_theta",
]
