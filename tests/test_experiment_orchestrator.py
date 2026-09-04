from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from quantum_measurement.experiments import DEFAULT_REGISTRY, ExperimentRunOrchestrator, ExperimentStore


def test_orchestrator_ingests_csv_and_event_log(tmp_path: Path) -> None:
    db_path = tmp_path / "exp.db"
    store = ExperimentStore(db_path)
    orch = ExperimentRunOrchestrator(store, DEFAULT_REGISTRY.get("z2_scan"))

    csv_path = tmp_path / "z2.csv"
    csv_path.write_text(
        "L,gamma,g,z2_mean,z2_plus_one,nan_detected,range_violation,point_status,runtime_sec\n"
        "9,1.0,0.25,0.2,1.2,false,false,ok,0.5\n"
        "17,2.0,0.5,nan,nan,true,false,nan,0.8\n",
        encoding="utf-8",
    )

    event_log = tmp_path / "events.jsonl"
    event_log.write_text(
        json.dumps({"event_type": "sweep_point_completed", "message": "done", "payload": {"L": 9}}) + "\n",
        encoding="utf-8",
    )

    ctx = orch.start_run(
        config={"x": 1},
        resume_enabled=True,
        requested_cores=4,
        actual_cores=4,
        executor_kind="parameter_sweep",
        backend_device="cpu",
        csv_path=str(csv_path),
        event_log_path=str(event_log),
        raw_series_enabled=True,
        enable_monitor=False,
    )

    orch.finish_run(
        run_id=ctx.run_id,
        status="completed",
        csv_path=csv_path,
        event_log_path=event_log,
        error_message=None,
    )

    conn = sqlite3.connect(db_path)
    try:
        run_status = conn.execute("SELECT status FROM runs WHERE id=?", (ctx.run_id,)).fetchone()[0]
        point_count = conn.execute("SELECT COUNT(*) FROM run_points WHERE run_id=?", (ctx.run_id,)).fetchone()[0]
        event_count = conn.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (ctx.run_id,)).fetchone()[0]

        assert run_status == "completed"
        assert point_count == 2
        assert event_count >= 3  # run_start + csv_ingested + run_finished + jsonl events
    finally:
        conn.close()
