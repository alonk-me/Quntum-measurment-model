from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest

from quantum_measurement.backends.multi_cpu_backend import MultiCpuBackend, MultiCpuBackendConfig


def _sim_factory_nan(L, gamma, backend_device, rng):
    _ = (backend_device, rng)
    return {
        "L": int(L),
        "gamma": float(gamma),
        "z2_mean": float("nan"),
        "z2_plus_one": float("nan"),
        "runtime_sec": 0.01,
    }


def _sim_factory_ok(L, gamma, backend_device, rng):
    _ = (backend_device, rng)
    return {
        "L": int(L),
        "gamma": float(gamma),
        "z2_mean": 1.0,
        "z2_plus_one": 2.0,
        "runtime_sec": 0.01,
    }


def _sim_factory_range_violation(L, gamma, backend_device, rng):
    _ = (L, gamma, backend_device, rng)
    return {
        "z2_mean": 1.15,
        "z2_plus_one": 2.15,
        "runtime_sec": 0.01,
    }


def _sim_factory_nan_then_slow(L, gamma, backend_device, rng):
    _ = (L, backend_device, rng)
    if float(gamma) < 0.5:
        return {
            "gamma": float(gamma),
            "z2_mean": float("nan"),
            "z2_plus_one": float("nan"),
            "runtime_sec": 0.01,
        }

    # Simulate a long-running sibling worker that should be terminated early.
    time.sleep(8.0)
    return {
        "gamma": float(gamma),
        "z2_mean": 1.0,
        "z2_plus_one": 2.0,
        "runtime_sec": 8.0,
    }


def test_fail_on_nan_mode_raises(tmp_path: Path) -> None:
    log_path = tmp_path / "events_fail.jsonl"
    backend = MultiCpuBackend(
        MultiCpuBackendConfig(
            max_workers=1,
            reserve_cores=0,
            master_seed=7,
            nan_mode="fail_on_nan",
            log_path=log_path,
        )
    )

    with pytest.raises(ValueError, match="NaN point row detected"):
        backend.run_sweep(
            L_values=[9],
            gamma_grid=[1.0],
            simulator_factory=_sim_factory_nan,
            backend_device="cpu",
            output_csv=None,
            resume=False,
        )

    text = log_path.read_text(encoding="utf-8")
    assert "nan_point_row" in text


def test_finish_full_sweep_mode_continues_and_marks_row(tmp_path: Path) -> None:
    log_path = tmp_path / "events_finish.jsonl"
    csv_path = tmp_path / "out.csv"
    backend = MultiCpuBackend(
        MultiCpuBackendConfig(
            max_workers=1,
            reserve_cores=0,
            master_seed=11,
            nan_mode="finish_full_sweep",
            log_path=log_path,
        )
    )

    rows = backend.run_sweep(
        L_values=[9],
        gamma_grid=[1.0],
        simulator_factory=_sim_factory_nan,
        backend_device="cpu",
        output_csv=csv_path,
        resume=False,
        csv_header=["L", "gamma", "z2_mean", "z2_plus_one", "runtime_sec", "nan_detected", "point_status"],
    )

    assert len(rows) == 1
    assert rows[0]["nan_detected"] is True
    assert rows[0]["point_status"] == "nan"
    assert csv_path.exists()

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(evt.get("event_type") == "nan_point_row" for evt in events)


def test_invalid_nan_mode_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid nan_mode"):
        _ = MultiCpuBackend(MultiCpuBackendConfig(nan_mode="bad_mode"))


def test_ok_row_default_status(tmp_path: Path) -> None:
    backend = MultiCpuBackend(
        MultiCpuBackendConfig(
            max_workers=1,
            reserve_cores=0,
            master_seed=5,
            nan_mode="finish_full_sweep",
        )
    )
    rows = backend.run_sweep(
        L_values=[9],
        gamma_grid=[1.0],
        simulator_factory=_sim_factory_ok,
        backend_device="cpu",
        output_csv=None,
        resume=False,
    )
    assert len(rows) == 1
    assert rows[0]["nan_detected"] is False
    assert rows[0]["point_status"] == "ok"


def test_fail_on_nan_aborts_without_waiting_for_slow_worker(tmp_path: Path) -> None:
    backend = MultiCpuBackend(
        MultiCpuBackendConfig(
            max_workers=2,
            reserve_cores=0,
            master_seed=13,
            nan_mode="fail_on_nan",
            log_path=tmp_path / "events_fast_abort.jsonl",
        )
    )

    start = time.monotonic()
    with pytest.raises(ValueError, match="NaN point row detected"):
        backend.run_sweep(
            L_values=[9],
            gamma_grid=[0.1, 1.0],
            simulator_factory=_sim_factory_nan_then_slow,
            backend_device="cpu",
            output_csv=None,
            resume=False,
        )
    elapsed = time.monotonic() - start

    # If fail-fast shutdown works, we should not wait for the 8s sibling task.
    assert elapsed < 6.0


def test_range_violation_is_flagged_and_logged(tmp_path: Path) -> None:
    log_path = tmp_path / "events_range.jsonl"
    backend = MultiCpuBackend(
        MultiCpuBackendConfig(
            max_workers=1,
            reserve_cores=0,
            master_seed=17,
            nan_mode="finish_full_sweep",
            log_path=log_path,
        )
    )

    rows = backend.run_sweep(
        L_values=[9],
        gamma_grid=[1.0],
        simulator_factory=_sim_factory_range_violation,
        backend_device="cpu",
        output_csv=None,
        resume=False,
    )

    assert len(rows) == 1
    assert rows[0]["nan_detected"] is False
    assert rows[0]["range_violation"] is True
    assert rows[0]["point_status"] == "range_violation"

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(evt.get("event_type") == "range_violation_point_row" for evt in events)
