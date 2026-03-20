"""Stochastic Schrödinger Equation (SSE) experiment runner.

This module wraps :class:`quantum_measurement.sse_simulation.sse.SSEWavefunctionSimulator`
in a clean, self-contained experiment interface.  A single call to
:func:`run_sse_experiment` simulates an ensemble of trajectories and returns an
:class:`SSEExperimentResult` containing raw trajectories and summary statistics.

Typical usage::

    from quantum_measurement.experiments import run_sse_experiment, SSEExperimentConfig

    cfg = SSEExperimentConfig(n_trajectories=200, epsilon=0.1, N_steps=500)
    result = run_sse_experiment(cfg)
    print(f"mean Q = {result.mean_Q:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SSEExperimentConfig:
    """Configuration for an SSE experiment.

    Attributes
    ----------
    n_trajectories : int
        Number of independent SSE trajectories to simulate.
    epsilon : float
        Measurement strength parameter ε.
    N_steps : int
        Number of discrete time steps per trajectory.
    J : float
        Hamiltonian coupling parameter (default ``0.0``, no Hamiltonian).
    initial_state : str
        Name of the initial quantum state preset.  Supported values:
        ``'bloch_equator'``, ``'up'``, ``'down'``, ``'plus_y'``,
        ``'minus_y'``, ``'custom'``.
    seed : int or None
        Seed for the random number generator.
    """

    n_trajectories: int = 200
    epsilon: float = 0.1
    N_steps: int = 200
    J: float = 0.0
    initial_state: str = "bloch_equator"
    seed: Optional[int] = None

    @property
    def theta(self) -> float:
        """Theoretical dimensionless parameter θ = N_steps · ε²."""
        return self.N_steps * self.epsilon ** 2


@dataclass
class SSEExperimentResult:
    """Results of an SSE experiment.

    Attributes
    ----------
    config : SSEExperimentConfig
        The configuration used to produce these results.
    Q_values : numpy.ndarray, shape (n_trajectories,)
        Entropy-production value for each trajectory.
    z_trajectories : numpy.ndarray, shape (n_trajectories, N_steps + 1)
        Bloch-z expectation values at every time step.
    measurement_results : numpy.ndarray, shape (n_trajectories, N_steps)
        Measurement outcome ξ ∈ {+1, −1} at every step.
    mean_Q : float
        Sample mean of the entropy production.
    std_Q : float
        Sample standard deviation of the entropy production.
    theoretical_mean_Q : float
        Theoretical mean ⟨Q⟩ ≈ (3/2) · N · ε² for equatorial initial state.
    theoretical_var_Q : float
        Theoretical variance Var(Q) ≈ 2 · N · ε² for equatorial initial state.
    """

    config: SSEExperimentConfig
    Q_values: np.ndarray
    z_trajectories: np.ndarray
    measurement_results: np.ndarray
    mean_Q: float
    std_Q: float
    theoretical_mean_Q: float
    theoretical_var_Q: float


def run_sse_experiment(
    config: SSEExperimentConfig,
    progress: bool = False,
) -> SSEExperimentResult:
    """Run an SSE ensemble experiment.

    Parameters
    ----------
    config : SSEExperimentConfig
        Experiment parameters.
    progress : bool
        If *True* and *tqdm* is available, show a progress bar.

    Returns
    -------
    SSEExperimentResult
        Bundled results containing raw trajectories and summary statistics.
    """
    from quantum_measurement.sse_simulation.sse import SSEWavefunctionSimulator

    rng = np.random.default_rng(config.seed)

    logger.info(
        "Starting SSE experiment: n_traj=%d, N_steps=%d, epsilon=%.4f, J=%.4f",
        config.n_trajectories,
        config.N_steps,
        config.epsilon,
        config.J,
    )

    sim = SSEWavefunctionSimulator(
        epsilon=config.epsilon,
        N_steps=config.N_steps,
        J=config.J,
        initial_state=config.initial_state,
        rng=rng,
    )

    Q_values, z_trajectories, measurement_results = sim.simulate_ensemble(
        config.n_trajectories,
        progress=progress,
    )

    mean_Q = float(np.mean(Q_values))
    std_Q = float(np.std(Q_values))
    th_mean, th_var = sim.theoretical_mean_variance()

    logger.info(
        "SSE experiment done: mean_Q=%.4f (theory=%.4f), std_Q=%.4f",
        mean_Q,
        th_mean,
        std_Q,
    )

    return SSEExperimentResult(
        config=config,
        Q_values=Q_values,
        z_trajectories=z_trajectories,
        measurement_results=measurement_results,
        mean_Q=mean_Q,
        std_Q=std_Q,
        theoretical_mean_Q=th_mean,
        theoretical_var_Q=th_var,
    )
