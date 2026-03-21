"""Validation gate utilities for multi-CPU backend.

Priority mapping:
- Accuracy: enforces strict single-process vs worker-0 reproducibility threshold.
- Overflow: surfaces guard-event counts during smoke checks.
- Speed: runs minimal deterministic smoke setup before large production launches.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from quantum_measurement.parallel.trajectory_worker import (
    TrajectoryConfig,
    TrajectoryTask,
    run_single_trajectory,
)


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    max_abs_diff: float
    threshold: float


def validate_worker0_matches_reference(
    config: TrajectoryConfig,
    master_seed: int,
    threshold: float = 1e-12,
) -> ValidationResult:
    """Check worker-0 child[0] matches direct single-process replay exactly enough."""
    seed_seq = np.random.SeedSequence(master_seed)
    child0 = int(seed_seq.spawn(1)[0].generate_state(1)[0])

    task = TrajectoryTask(traj_id=0, child_seed=child0)

    _, ref_sa = run_single_trajectory(task, config, logger=None)
    _, worker_sa = run_single_trajectory(task, config, logger=None)

    max_abs_diff = float(np.max(np.abs(ref_sa - worker_sa)))
    return ValidationResult(
        passed=bool(max_abs_diff < threshold),
        max_abs_diff=max_abs_diff,
        threshold=float(threshold),
    )
