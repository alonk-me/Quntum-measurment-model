from __future__ import annotations

from pathlib import Path
import sqlite3

from quantum_measurement.experiments import ExperimentStore


def test_experiment_store_defaults_and_run_lifecycle(tmp_path: Path) -> None:
    db_path = tmp_path / "results" / "experiments.sqlite3"
    store = ExperimentStore(db_path)

    store.register_experiment_type("unit_test_exp", "unit test")
    run_id = store.create_run(
        experiment_type="unit_test_exp",
        status="running",
        config={"a": 1},
        resume_enabled=True,
        requested_cores=8,
        actual_cores=4,
        executor_kind="parameter_sweep",
        backend_device="cpu",
        csv_path=str(tmp_path / "out.csv"),
        event_log_path=None,
        raw_series_enabled=True,
    )

    store.upsert_point(
        run_id,
        {
            "L": 9,
            "gamma": 1.0,
            "g": 0.25,
            "z2_mean": 0.2,
            "z2_plus_one": 1.2,
            "nan_detected": False,
            "range_violation": False,
            "point_status": "ok",
            "runtime_sec": 0.1,
        },
    )
    store.log_event(run_id, "test_event", "test event message", payload={"x": 1})
    store.record_monitor_session(
        run_id,
        {
            "script_path": "/tmp/monitor.py",
            "output_path": "/tmp/live_progress.png",
            "pid": 123,
            "status": "terminated",
            "duration_sec": 2.5,
            "error_count": 0,
            "nan_count": 0,
            "range_violation_count": 0,
            "instability_count": 0,
        },
    )
    store.finalize_run(run_id, status="completed", error_message=None)

    conn = sqlite3.connect(db_path)
    try:
        run_row = conn.execute("SELECT status, requested_cores, actual_cores FROM runs WHERE id=?", (run_id,)).fetchone()
        assert run_row is not None
        assert run_row[0] == "completed"
        assert run_row[1] == 8
        assert run_row[2] == 4

        point_count = conn.execute("SELECT COUNT(*) FROM run_points WHERE run_id=?", (run_id,)).fetchone()[0]
        event_count = conn.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
        monitor_count = conn.execute("SELECT COUNT(*) FROM monitor_sessions WHERE run_id=?", (run_id,)).fetchone()[0]

        assert point_count == 1
        assert event_count >= 1
        assert monitor_count == 1
    finally:
        conn.close()


def test_resolve_db_path_explicit(tmp_path: Path) -> None:
    db = ExperimentStore.resolve_db_path(tmp_path / "x.db")
    assert db == (tmp_path / "x.db").resolve()
