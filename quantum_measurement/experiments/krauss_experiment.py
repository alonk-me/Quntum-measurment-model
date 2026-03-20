"""Kraus-operator Monte-Carlo experiment runner.

This module wraps :mod:`quantum_measurement.krauss_operators.krauss_operators_simulation`
in a clean, self-contained experiment interface. A single call to
:func:`run_krauss_experiment` runs an ensemble of trajectories and returns a
:class:`KraussExperimentResult` that bundles the raw data together with the
experiment configuration, making it easy to persist, inspect, and feed into
the analysis module.

Typical usage::

    from quantum_measurement.experiments import run_krauss_experiment, KraussExperimentConfig

    cfg = KraussExperimentConfig(num_traj=1000, N=20_000, epsilon=0.01)
    result = run_krauss_experiment(cfg)
    print(f"mean Q = {result.mean_Q:.4f}, theta = {result.theta:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KraussExperimentConfig:
    """Configuration for a Kraus-operator experiment.

    Attributes
    ----------
    num_traj : int
        Number of independent Monte-Carlo trajectories.
    N : int
        Number of discrete measurement steps per trajectory.
    epsilon : float
        Measurement strength parameter (``a = 1 + epsilon``).
    omega_dt : float
        Angular frequency of coherent rotation about the y-axis multiplied
        by the time step.  Set to ``0.0`` (default) for pure measurement
        dynamics (no Hamiltonian).
    seed : int or None
        Seed for the random number generator.  ``None`` gives a non-
        reproducible run.
    """

    num_traj: int = 500
    N: int = 10_000
    epsilon: float = 0.01
    omega_dt: float = 0.0
    seed: Optional[int] = None

    @property
    def theta(self) -> float:
        """Theoretical dimensionless parameter θ = N · ε²."""
        return self.N * self.epsilon ** 2


@dataclass
class KraussExperimentResult:
    """Results of a Kraus-operator experiment.

    Attributes
    ----------
    config : KraussExperimentConfig
        The configuration used to produce these results.
    Q_values : numpy.ndarray
        Array of entropy-production values, one per trajectory.
    mean_Q : float
        Sample mean of the entropy production.
    std_Q : float
        Sample standard deviation of the entropy production.
    theta : float
        Theoretical dimensionless parameter θ = N · ε².
    """

    config: KraussExperimentConfig
    Q_values: np.ndarray
    mean_Q: float
    std_Q: float
    theta: float


def run_krauss_experiment(
    config: KraussExperimentConfig,
    progress: bool = False,
) -> KraussExperimentResult:
    """Run a Kraus-operator Monte-Carlo experiment.

    Parameters
    ----------
    config : KraussExperimentConfig
        Experiment parameters.
    progress : bool
        If *True* and *tqdm* is available, show a progress bar during the
        trajectory loop.

    Returns
    -------
    KraussExperimentResult
        Bundled results containing raw Q values and summary statistics.
    """
    from quantum_measurement.krauss_operators.krauss_operators_simulation import (
        simulate_Q_distribution,
    )

    logger.info(
        "Starting Kraus experiment: num_traj=%d, N=%d, epsilon=%.4f, theta=%.4f",
        config.num_traj,
        config.N,
        config.epsilon,
        config.theta,
    )

    Q_values = simulate_Q_distribution(
        num_traj=config.num_traj,
        N=config.N,
        epsilon=config.epsilon,
        omega_dt=config.omega_dt,
        seed=config.seed,
    )

    mean_Q = float(np.mean(Q_values))
    std_Q = float(np.std(Q_values))

    logger.info("Kraus experiment done: mean_Q=%.4f, std_Q=%.4f", mean_Q, std_Q)

    return KraussExperimentResult(
        config=config,
        Q_values=Q_values,
        mean_Q=mean_Q,
        std_Q=std_Q,
        theta=config.theta,
    )
