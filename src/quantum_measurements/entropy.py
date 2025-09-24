"""Entropy production utilities for single-qubit SSE trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .sse_qubit import TrajectoryData


@dataclass(slots=True)
class EntropyProduction:
    """Entropy production contributions for a single trajectory."""

    deterministic: float
    stochastic: float

    @property
    def total(self) -> float:
        return self.deterministic + self.stochastic


def compute_entropy_production(trajectory: TrajectoryData, epsilon: float) -> EntropyProduction:
    """Compute the entropy production according to Eq. (2) of the SRS."""

    z_mid = trajectory.z_mid
    z_after = trajectory.z_after
    xi = trajectory.xi.astype(float)

    det = 2.0 * epsilon * epsilon * float(np.sum(z_after * z_mid))
    sto = 2.0 * epsilon * float(np.sum(xi * z_mid))
    return EntropyProduction(det, sto)


def compute_entropy_productions(
    trajectories: Iterable[TrajectoryData],
    epsilon: float,
) -> np.ndarray:
    """Compute entropy production for an ensemble of trajectories."""

    totals = [compute_entropy_production(traj, epsilon).total for traj in trajectories]
    return np.asarray(totals, dtype=float)

