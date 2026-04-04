from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "reproduce_turkeshi_fig2.py"
_SPEC = importlib.util.spec_from_file_location("reproduce_turkeshi_fig2", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

get_time_params = _MODULE.get_time_params
apply_profile_overrides = _MODULE.apply_profile_overrides
resolve_subsystem_sites = _MODULE.resolve_subsystem_sites
parse_float_list = _MODULE.parse_float_list
parse_int_list = _MODULE.parse_int_list
run_point = _MODULE.run_point
PointResult = _MODULE.PointResult
checkpoint_file_for = _MODULE.checkpoint_file_for
partial_checkpoint_file_for = _MODULE.partial_checkpoint_file_for
load_point_checkpoint = _MODULE.load_point_checkpoint
save_point_checkpoint = _MODULE.save_point_checkpoint
save_point_partial_checkpoint = _MODULE.save_point_partial_checkpoint


def test_parse_list_helpers() -> None:
    assert parse_int_list(" 9, 17,33 ") == [9, 17, 33]
    assert parse_float_list("1,2, 3.5") == [1.0, 2.0, 3.5]


def test_get_time_params_positive_outputs() -> None:
    T, dt, n_steps = get_time_params(
        gamma=2.0,
        t_multiplier=60.0,
        t_min=100.0,
        dt_ratio=5e-3,
        dt_max=1e-3,
    )
    assert T > 0.0
    assert dt > 0.0
    assert n_steps > 0
    assert abs(n_steps * dt - T) < max(1e-6, dt)


def test_profile_override_diagnostic_fast() -> None:
    t_multiplier, t_min, dt_ratio, dt_max = apply_profile_overrides(
        "diagnostic-fast",
        t_multiplier=60.0,
        t_min=100.0,
        dt_ratio=5e-3,
        dt_max=1e-3,
    )
    assert t_multiplier == 1.0
    assert t_min == 5.0
    assert dt_ratio == 5e-3
    assert dt_max == 1e-3


def test_resolve_subsystem_sites_defaults_and_clamps() -> None:
    assert resolve_subsystem_sites(8, None, 0.25) == 2
    assert resolve_subsystem_sites(8, 20, 0.25) == 8
    assert resolve_subsystem_sites(8, 0, 0.25) == 1


def test_run_point_serial_tiny() -> None:
    result = run_point(
        L=8,
        gamma=2.0,
        J=1.0,
        n_trajectories=3,
        closed_boundary=False,
        t_multiplier=1.0,
        t_min=0.05,
        dt_ratio=1e-1,
        dt_max=0.02,
        subsystem_sites=2,
        subsystem_start=0,
        mode="serial",
        workers=1,
        master_seed=123,
        checkpoint_dir=None,
        partial_checkpoint_every_chunks=1,
    )

    assert result.L == 8
    assert np.isclose(result.gamma, 2.0)
    assert result.n_steps + 1 == result.t_grid.shape[0]
    assert result.s_mean.shape == result.t_grid.shape
    assert result.s_std.shape == result.t_grid.shape
    assert np.all(np.isfinite(result.s_mean))
    assert np.all(np.isfinite(result.s_std))


def test_checkpoint_roundtrip(tmp_path) -> None:
    result = PointResult(
        L=8,
        gamma=2.0,
        dt=0.01,
        n_steps=3,
        t_grid=np.array([0.0, 0.01, 0.02, 0.03]),
        s_mean=np.array([0.0, 0.1, 0.2, 0.3]),
        s_std=np.array([0.0, 0.01, 0.02, 0.03]),
    )
    ckpt_dir = tmp_path / "ckpt"
    save_point_checkpoint(
        ckpt_dir,
        result,
        n_trajectories=4,
        mode="serial",
        closed_boundary=False,
        J=1.0,
        subsystem_sites=2,
        subsystem_start=0,
    )

    ckpt_file = checkpoint_file_for(ckpt_dir, L=8, gamma=2.0)
    loaded = load_point_checkpoint(
        ckpt_file,
        expected_n_trajectories=4,
        expected_mode="serial",
        expected_closed_boundary=False,
        expected_J=1.0,
        expected_subsystem_sites=2,
        expected_subsystem_start=0,
    )

    assert loaded.L == result.L
    assert np.isclose(loaded.gamma, result.gamma)
    assert np.isclose(loaded.dt, result.dt)
    assert loaded.n_steps == result.n_steps
    assert np.allclose(loaded.t_grid, result.t_grid)
    assert np.allclose(loaded.s_mean, result.s_mean)
    assert np.allclose(loaded.s_std, result.s_std)


def test_checkpoint_mismatch_rejected(tmp_path) -> None:
    result = PointResult(
        L=8,
        gamma=2.0,
        dt=0.01,
        n_steps=1,
        t_grid=np.array([0.0, 0.01]),
        s_mean=np.array([0.0, 0.1]),
        s_std=np.array([0.0, 0.01]),
    )
    ckpt_dir = tmp_path / "ckpt"
    save_point_checkpoint(
        ckpt_dir,
        result,
        n_trajectories=4,
        mode="serial",
        closed_boundary=False,
        J=1.0,
        subsystem_sites=2,
        subsystem_start=0,
    )
    ckpt_file = checkpoint_file_for(ckpt_dir, L=8, gamma=2.0)

    try:
        load_point_checkpoint(
            ckpt_file,
            expected_n_trajectories=5,
            expected_mode="serial",
            expected_closed_boundary=False,
            expected_J=1.0,
            expected_subsystem_sites=2,
            expected_subsystem_start=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_partial_checkpoint_write(tmp_path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    t_grid = np.array([0.0, 0.1])
    s_mean = np.array([0.0, 0.2])
    s_std = np.array([0.0, 0.01])
    out = save_point_partial_checkpoint(
        ckpt_dir,
        L=8,
        gamma=2.0,
        dt=0.1,
        n_steps=1,
        t_grid=t_grid,
        s_mean_partial=s_mean,
        s_std_partial=s_std,
        n_completed=1,
        n_total=4,
    )
    assert out == partial_checkpoint_file_for(ckpt_dir, 8, 2.0)
    assert out.exists()
