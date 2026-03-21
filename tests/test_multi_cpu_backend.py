from __future__ import annotations

import numpy as np

from quantum_measurement.backends.multi_cpu_backend import MultiCpuBackend, MultiCpuBackendConfig
from quantum_measurement.parallel.trajectory_worker import (
    TrajectoryConfig,
    TrajectoryTask,
    _apply_projection_multi_site,
    run_single_trajectory,
)
from quantum_measurement.backends.multi_cpu_backend import _available_worker_cores
from quantum_measurement.parallel.validation import validate_worker0_matches_reference


def test_validation_gate_worker0_matches_reference() -> None:
    config = TrajectoryConfig(
        L=3,
        J=1.0,
        gamma=0.2,
        dt=0.01,
        n_steps=5,
        closed_boundary=False,
    )
    result = validate_worker0_matches_reference(config=config, master_seed=123, threshold=1e-12)
    assert result.passed
    assert result.max_abs_diff < result.threshold


def test_worker_deterministic_for_same_child_seed() -> None:
    config = TrajectoryConfig(
        L=3,
        J=1.0,
        gamma=0.1,
        dt=0.01,
        n_steps=4,
        closed_boundary=True,
    )
    task = TrajectoryTask(traj_id=7, child_seed=999)
    _, s1 = run_single_trajectory(task, config)
    _, s2 = run_single_trajectory(task, config)
    assert np.array_equal(s1, s2)


def test_multi_cpu_backend_returns_compact_payload() -> None:
    backend = MultiCpuBackend(MultiCpuBackendConfig(max_workers=2, master_seed=7))
    config = TrajectoryConfig(
        L=3,
        J=1.0,
        gamma=0.15,
        dt=0.01,
        n_steps=3,
        closed_boundary=False,
    )

    results, counts = backend.run_trajectories(total_trajectories=2, config=config)
    assert len(results) == 2
    assert all(isinstance(item[0], int) for item in results)
    assert all(isinstance(item[1], np.ndarray) for item in results)
    assert all(item[1].shape == (config.n_steps + 1,) for item in results)
    assert isinstance(counts, dict)


def test_projection_multi_site_vectorized_update_shapes() -> None:
    L = 3
    dim = 2 * L
    C = np.eye(dim, dtype=np.complex128)
    C = C * 0.5
    updated = _apply_projection_multi_site(
        C=C,
        site_indices=np.arange(L, dtype=np.int64),
        traj_id=0,
        step=0,
        L=L,
        gamma=0.1,
        logger=None,
    )
    assert updated.shape == C.shape
    assert np.all(np.isfinite(np.real(updated)))


def test_available_worker_cores_respects_reserve() -> None:
    all_cores = _available_worker_cores(0)
    reserved = _available_worker_cores(1)
    assert len(all_cores) >= 1
    assert len(reserved) >= 1
    assert len(reserved) <= len(all_cores)
