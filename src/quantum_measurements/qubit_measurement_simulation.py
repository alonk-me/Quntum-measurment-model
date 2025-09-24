"""High-level interface for the single-qubit SSE simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .entropy import compute_entropy_production
from .sse_qubit import TrajectoryConfig, run_ensemble, run_trajectory
from .theory import dressel_eq14_pdf
from .visualisation import plot_histogram_with_theory


@dataclass(slots=True)
class EnsembleResult:
    """Results of an ensemble simulation of the single-qubit SSE."""

    epsilon: float
    steps: int
    dt: float
    Q_values: np.ndarray
    deterministic: np.ndarray
    stochastic: np.ndarray

    @property
    def theta(self) -> float:
        """Return the Dressel parameter θ = 2 N ε."""

        return 2.0 * self.steps * self.epsilon


def simulate_trajectory(steps: int, epsilon: float, dt: float = 1.0, seed: Optional[int] = None):
    """Return trajectory data and entropy production for a single run."""

    config = TrajectoryConfig(steps=steps, epsilon=epsilon, dt=dt, seed=seed)
    trajectory = run_trajectory(config)
    ep = compute_entropy_production(trajectory, epsilon)
    return trajectory, ep


def simulate_ensemble(
    num_trajectories: int,
    steps: int,
    epsilon: float,
    dt: float = 1.0,
    seed: Optional[int] = None,
) -> EnsembleResult:
    """Simulate an ensemble and collect entropy production statistics."""

    config = TrajectoryConfig(steps=steps, epsilon=epsilon, dt=dt, seed=seed)
    trajectories = run_ensemble(config, num_trajectories)
    entropy_values = [compute_entropy_production(traj, epsilon) for traj in trajectories]
    Q = np.fromiter((ep.total for ep in entropy_values), dtype=float, count=num_trajectories)
    deterministic = np.fromiter((ep.deterministic for ep in entropy_values), dtype=float, count=num_trajectories)
    stochastic = np.fromiter((ep.stochastic for ep in entropy_values), dtype=float, count=num_trajectories)
    return EnsembleResult(epsilon, steps, dt, Q, deterministic, stochastic)


def save_ensemble(result: EnsembleResult, filename: str | Path) -> None:
    """Serialise ensemble data to ``filename`` using :func:`numpy.savez`."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        epsilon=result.epsilon,
        steps=result.steps,
        dt=result.dt,
        Q_values=result.Q_values,
        deterministic=result.deterministic,
        stochastic=result.stochastic,
        theta=result.theta,
    )


__all__ = [
    "EnsembleResult",
    "dressel_eq14_pdf",
    "plot_histogram_with_theory",
    "save_ensemble",
    "simulate_ensemble",
    "simulate_trajectory",
]

