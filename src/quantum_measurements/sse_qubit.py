r"""Stochastic Schrödinger equation for a single qubit.

This module implements the Stratonovich interpretation of the stochastic
Schrödinger equation (SSE) for a continuously monitored qubit following the
formalism of Turkeshi *et al.*.  The measurement backaction is modelled with
the symmetric Kraus operators described in the accompanying SRS.  These
operators generate the same stochastic increments as the Stratonovich form of
the SSE for a measurement of :math:`\sigma_z`.

The module exposes utilities to integrate single trajectories, collect
ensembles and expose data for entropy production calculations.  The entropy
production itself is computed in :mod:`quantum_measurements.entropy` to keep
the solver reusable when additional observables are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def _normalise(state: np.ndarray) -> np.ndarray:
    """Return the normalised copy of a state vector."""

    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("The state vector norm vanished during normalisation.")
    return state / norm


def expectation_sigma_z(state: np.ndarray) -> float:
    r"""Compute :math:`\langle\sigma_z\rangle` for ``state``."""

    return float(np.vdot(state, SIGMA_Z @ state).real)


def symmetric_kraus_operators(epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the symmetric Kraus operators for measurement strength ``epsilon``.

    Parameters
    ----------
    epsilon:
        Dimensionless measurement strength.  The Stratonovich limit is
        recovered for :math:`\epsilon \ll 1`.  The value must satisfy
        :math:`|\epsilon| < 1` so that the square-roots remain real.
    """

    if not (-1.0 < epsilon < 1.0):
        raise ValueError("epsilon must satisfy |epsilon| < 1 for symmetric Kraus operators.")

    factor = 1.0 / np.sqrt(2.0)
    plus = factor * np.diag([np.sqrt(1.0 + epsilon), np.sqrt(1.0 - epsilon)])
    minus = factor * np.diag([np.sqrt(1.0 - epsilon), np.sqrt(1.0 + epsilon)])
    return plus.astype(complex), minus.astype(complex)


@dataclass(slots=True)
class TrajectoryConfig:
    """Configuration describing a single SSE trajectory."""

    steps: int
    epsilon: float
    dt: float = 1.0
    initial_state: Optional[np.ndarray] = None
    seed: Optional[int] = None


@dataclass(slots=True)
class TrajectoryData:
    """Container holding the data produced by :func:`run_trajectory`."""

    times: np.ndarray
    states: np.ndarray
    xi: np.ndarray
    z_before: np.ndarray
    z_after: np.ndarray
    z_mid: np.ndarray


def _initial_state(config: TrajectoryConfig) -> np.ndarray:
    if config.initial_state is None:
        return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    if config.initial_state.shape != (2,):
        raise ValueError("initial_state must be a length-2 vector in the computational basis.")
    return _normalise(config.initial_state.astype(complex))


def stratonovich_step(
    state: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float]:
    r"""Apply one symmetric Kraus step to ``state``.

    Parameters
    ----------
    state:
        The normalised qubit state prior to the measurement.
    epsilon:
        Measurement strength (dimensionless).  Must satisfy ``|epsilon| < 1``.
    rng:
        Source of randomness used to sample the measurement record.

    Returns
    -------
    (state', xi, z_after):
        * ``state'`` – the post-measurement state,
        * ``xi`` – measurement record taking values :math:`\pm 1`,
        * ``z_after`` – :math:`\langle\sigma_z\rangle` evaluated for the
          updated state.
    """

    M_plus, M_minus = symmetric_kraus_operators(epsilon)
    amp_plus = M_plus @ state
    amp_minus = M_minus @ state
    prob_plus = np.vdot(amp_plus, amp_plus).real
    prob_minus = np.vdot(amp_minus, amp_minus).real

    # numerical guard to ensure probabilities sum to 1
    norm = prob_plus + prob_minus
    prob_plus /= norm
    prob_minus /= norm

    sample = rng.random()
    if sample < prob_plus:
        xi = 1
        new_state = _normalise(amp_plus)
    else:
        xi = -1
        new_state = _normalise(amp_minus)

    return new_state, xi, expectation_sigma_z(new_state)


def run_trajectory(config: TrajectoryConfig) -> TrajectoryData:
    """Simulate a single trajectory according to ``config``."""

    rng = np.random.default_rng(config.seed)
    state = _initial_state(config)
    epsilon = config.epsilon

    times = np.linspace(0.0, config.dt * config.steps, config.steps + 1)
    states = np.empty((config.steps + 1, 2), dtype=complex)
    states[0] = state

    xi = np.empty(config.steps, dtype=int)
    z_before = np.empty(config.steps, dtype=float)
    z_after = np.empty(config.steps, dtype=float)
    z_mid = np.empty(config.steps, dtype=float)

    for step in range(config.steps):
        z_before[step] = expectation_sigma_z(state)
        state, xi_value, z_post = stratonovich_step(state, epsilon, rng)
        xi[step] = xi_value
        z_after[step] = z_post
        z_mid[step] = 0.5 * (z_before[step] + z_post)
        states[step + 1] = state

    return TrajectoryData(times, states, xi, z_before, z_after, z_mid)


def run_ensemble(
    config: TrajectoryConfig,
    num_trajectories: int,
) -> list[TrajectoryData]:
    """Simulate ``num_trajectories`` independent trajectories."""

    trajectories: list[TrajectoryData] = []
    base_seed = config.seed
    for index in range(num_trajectories):
        seed = None if base_seed is None else base_seed + index
        traj_config = TrajectoryConfig(
            steps=config.steps,
            epsilon=config.epsilon,
            dt=config.dt,
            initial_state=config.initial_state,
            seed=seed,
        )
        trajectories.append(run_trajectory(traj_config))
    return trajectories

