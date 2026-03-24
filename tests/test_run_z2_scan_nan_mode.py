import sys
from pathlib import Path

import numpy as np
import pytest

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
