import sys
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_z2_scan  # noqa: E402


class _DummyBackend:
    def seed(self, _seed):
        return None


class _DummySimNaN:
    def __init__(self, **_kwargs):
        self.backend = _DummyBackend()

    def simulate_z2_mean_ensemble(self, **_kwargs):
        return float("nan")


class _DummySimRangeViolation:
    def __init__(self, **_kwargs):
        self.backend = _DummyBackend()

    def simulate_z2_mean_ensemble(self, **_kwargs):
        return 1.2


def _call_point(nan_mode: str):
    return run_z2_scan._run_single_point_with_config(
        L=9,
        gamma=1.0,
        backend_device="cpu",
        rng=np.random.default_rng(123),
        n_trajectories_per_point=1,
        batch_size_per_point=None,
        compute_uncertainty=False,
        use_stable_integrator=True,
        enable_stability_monitor=False,
        t_multiplier=2.0,
        t_min=2.0,
        dt_ratio=0.01,
        dt_max=0.001,
        nan_mode=nan_mode,
    )


def test_finish_full_sweep_flags_nan(monkeypatch):
    monkeypatch.setattr(run_z2_scan, "LQubitCorrelationSimulator", _DummySimNaN)

    row = _call_point("finish_full_sweep")
    assert row["nan_detected"] is True
    assert row["point_status"] == "nan"
    assert np.isnan(row["z2_mean"])


def test_fail_on_nan_raises(monkeypatch):
    monkeypatch.setattr(run_z2_scan, "LQubitCorrelationSimulator", _DummySimNaN)

    with pytest.raises(ValueError, match="NaN detected"):
        _call_point("fail_on_nan")


def test_invalid_nan_mode_raises(monkeypatch):
    monkeypatch.setattr(run_z2_scan, "LQubitCorrelationSimulator", _DummySimNaN)

    with pytest.raises(ValueError, match="Invalid nan_mode"):
        _call_point("bad_mode")


def test_range_violation_flagged(monkeypatch):
    monkeypatch.setattr(run_z2_scan, "LQubitCorrelationSimulator", _DummySimRangeViolation)

    row = _call_point("finish_full_sweep")
    assert row["nan_detected"] is False
    assert row["range_violation"] is True
    assert row["point_status"] == "range_violation"


def test_phase_adaptive_schedule_caps_high_gamma_steps():
    sched = run_z2_scan.build_time_schedule(
        gamma=100.0,
        t_multiplier=60.0,
        t_min=100.0,
        dt_ratio=5e-3,
        dt_max=1e-3,
        enable_phase_adaptation=True,
        n_steps_min=1000,
        n_steps_max=20_000,
        dt_min=1e-6,
    )
    assert sched["adaptive_schedule"] is True
    assert sched["N_steps"] <= 20_000
    assert sched["dt"] <= 1e-3
    assert np.sqrt(100.0 * sched["dt"]) <= 0.5 + 1e-12


def test_phase_adaptation_requires_stable_integrator(monkeypatch):
    monkeypatch.setattr(run_z2_scan, "LQubitCorrelationSimulator", _DummySimRangeViolation)

    with pytest.raises(ValueError, match="requires --use-stable-integrator"):
        run_z2_scan._run_single_point_with_config(
            L=9,
            gamma=1.0,
            backend_device="cpu",
            rng=np.random.default_rng(123),
            n_trajectories_per_point=1,
            batch_size_per_point=None,
            compute_uncertainty=False,
            use_stable_integrator=False,
            enable_stability_monitor=False,
            t_multiplier=2.0,
            t_min=2.0,
            dt_ratio=0.01,
            dt_max=0.001,
            nan_mode="fail_on_nan",
            enable_phase_adaptation=True,
        )


def test_resume_metadata_mismatch_blocks(tmp_path):
    csv_path = tmp_path / "z2.csv"
    csv_path.write_text("L,gamma\n", encoding="utf-8")

    meta_a = run_z2_scan._build_run_metadata(
        use_stable_integrator=True,
        enable_phase_adaptation=True,
        nan_mode="fail_on_nan",
        schedule_config={"t_multiplier": 60.0, "n_steps_max": 500000},
    )
    run_z2_scan._ensure_resume_metadata(csv_file=csv_path, resume=False, metadata=meta_a)

    meta_b = run_z2_scan._build_run_metadata(
        use_stable_integrator=True,
        enable_phase_adaptation=True,
        nan_mode="fail_on_nan",
        schedule_config={"t_multiplier": 30.0, "n_steps_max": 500000},
    )

    with pytest.raises(ValueError, match="configuration mismatch"):
        run_z2_scan._ensure_resume_metadata(csv_file=csv_path, resume=True, metadata=meta_b)


def test_resume_metadata_missing_blocks(tmp_path):
    csv_path = tmp_path / "z2.csv"
    csv_path.write_text("L,gamma\n", encoding="utf-8")

    meta = run_z2_scan._build_run_metadata(
        use_stable_integrator=True,
        enable_phase_adaptation=False,
        nan_mode="fail_on_nan",
        schedule_config={"t_multiplier": 60.0},
    )

    with pytest.raises(ValueError, match="metadata file missing"):
        run_z2_scan._ensure_resume_metadata(csv_file=csv_path, resume=True, metadata=meta)


def test_build_and_load_phase_map_artifact(tmp_path):
    csv_path = tmp_path / "calibration.csv"
    # Synthetic transition centered near g=1.
    g_vals = np.logspace(-1, 1, 25)
    z2_plus_one = 1.1 + 0.7 * (1.0 / (1.0 + np.exp(-(np.log10(g_vals) - 0.0) / 0.15)))
    df = pd.DataFrame(
        {
            "L": [9] * len(g_vals),
            "gamma": 4.0 * g_vals,
            "g": g_vals,
            "z2_plus_one": z2_plus_one,
        }
    )
    df.to_csv(csv_path, index=False)

    out_path = tmp_path / "phase_map.json"
    built = run_z2_scan.build_phase_map_artifact(
        calibration_csv=csv_path,
        output_path=out_path,
        slope_threshold_rel=0.30,
        min_points=9,
    )
    assert built == out_path
    phase_map = run_z2_scan.load_phase_map_artifact(out_path)
    assert phase_map["phase_map_version"] == run_z2_scan.PHASE_MAP_VERSION
    assert phase_map["critical_g_min"] > 0.0
    assert phase_map["critical_g_max"] > phase_map["critical_g_min"]


def test_static_phase_map_lookup_overrides_inline_bounds():
    phase_map = {
        "phase_map_version": 1,
        "method": "test",
        "critical_g_min": 0.5,
        "critical_g_max": 0.6,
    }
    # gamma=4.0 => g=1.0 should be "after" with this phase map.
    sched = run_z2_scan.build_time_schedule(
        gamma=4.0,
        t_multiplier=60.0,
        t_min=100.0,
        dt_ratio=5e-3,
        dt_max=1e-3,
        enable_phase_adaptation=True,
        phase_critical_g_min=0.95,
        phase_critical_g_max=1.05,
        phase_map=phase_map,
    )
    assert sched["phase_label"] == "after"
    assert sched["phase_source"] == "phase_map"


def test_noncritical_step_cap_prevents_inflation():
    # gamma=8.0 => g=2.0 (after phase, non-critical)
    _, _, n_base, _ = run_z2_scan._baseline_time_schedule(
        gamma=8.0,
        t_multiplier=60.0,
        t_min=100.0,
        dt_ratio=5e-3,
        dt_max=1e-3,
    )
    sched = run_z2_scan.build_time_schedule(
        gamma=8.0,
        t_multiplier=60.0,
        t_min=100.0,
        dt_ratio=5e-3,
        dt_max=1e-3,
        enable_phase_adaptation=True,
        phase_steps_mult_after=2.0,
        phase_time_mult_after=1.3,
        noncritical_max_step_ratio=1.0,
        critical_max_step_ratio=1.3,
    )
    assert sched["phase_label"] == "after"
    assert sched["N_steps"] <= n_base


def test_critical_step_cap_limits_uplift():
    # gamma=4.0 => g=1.0 (critical)
    _, _, n_base, _ = run_z2_scan._baseline_time_schedule(
        gamma=4.0,
        t_multiplier=60.0,
        t_min=100.0,
        dt_ratio=5e-3,
        dt_max=1e-3,
    )
    sched = run_z2_scan.build_time_schedule(
        gamma=4.0,
        t_multiplier=60.0,
        t_min=100.0,
        dt_ratio=5e-3,
        dt_max=1e-3,
        enable_phase_adaptation=True,
        phase_steps_mult_critical=3.0,
        phase_time_mult_critical=1.5,
        noncritical_max_step_ratio=1.0,
        critical_max_step_ratio=1.1,
    )
    assert sched["phase_label"] == "critical"
    assert sched["N_steps"] <= int(np.ceil(1.1 * n_base))


def test_apply_adaptive_profile_next_pass_sets_expected_overrides():
    class _Args:
        adaptive_profile = run_z2_scan.ADAPTIVE_PROFILE_NEXT_PASS_V1
        enable_phase_adaptation = True
        t_multiplier = 60.0
        t_min = 100.0
        n_steps_min = 5000
        phase_steps_mult_before = 0.75
        phase_steps_mult_critical = 1.30
        phase_steps_mult_after = 0.60
        phase_time_mult_before = 1.00
        phase_time_mult_critical = 1.10
        phase_time_mult_after = 0.85
        noncritical_max_step_ratio = 1.00
        critical_max_step_ratio = 1.15

    args = _Args()
    run_z2_scan._apply_adaptive_profile(args)

    assert args.t_multiplier == 0.95
    assert args.t_min == 4.0
    assert args.n_steps_min == 3500
    assert args.phase_steps_mult_before == 0.70
    assert args.phase_steps_mult_critical == 1.12
    assert args.phase_steps_mult_after == 0.60
    assert args.phase_time_mult_before == 0.82
    assert args.phase_time_mult_critical == 1.00
    assert args.phase_time_mult_after == 0.82
    assert args.noncritical_max_step_ratio == 0.90
    assert args.critical_max_step_ratio == 1.08


def test_apply_adaptive_profile_ignored_without_adaptation():
    class _Args:
        adaptive_profile = run_z2_scan.ADAPTIVE_PROFILE_NEXT_PASS_V1
        enable_phase_adaptation = False
        t_multiplier = 60.0
        t_min = 100.0

    args = _Args()
    run_z2_scan._apply_adaptive_profile(args)

    assert args.t_multiplier == 60.0
    assert args.t_min == 100.0
