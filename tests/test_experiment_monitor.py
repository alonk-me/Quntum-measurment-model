from __future__ import annotations

from pathlib import Path

from quantum_measurement.experiments.monitor import MonitorSession, MonitorSessionConfig, summarize_numerical_health_from_csv


def test_monitor_session_stops_and_cleans_visual(tmp_path: Path) -> None:
    monitor_script = tmp_path / "dummy_monitor.py"
    monitor_script.write_text(
        "import argparse, time\n"
        "p=argparse.ArgumentParser(); p.add_argument('--csv'); p.add_argument('--interval', type=int, default=1); p.parse_args();\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    csv_path = tmp_path / "dummy.csv"
    csv_path.write_text("L,gamma\n", encoding="utf-8")

    output_path = tmp_path / "live_progress.png"
    output_path.write_text("placeholder", encoding="utf-8")

    session = MonitorSession(
        MonitorSessionConfig(
            script_path=str(monitor_script),
            output_path=str(output_path),
            csv_path=str(csv_path),
            interval_seconds=1,
            enabled=True,
        )
    )

    session.start()
    meta = session.stop(cleanup_visual=True)

    assert meta["status"] in {"terminated", "exited"}
    assert meta["duration_sec"] is None or meta["duration_sec"] >= 0.0
    assert not output_path.exists()


def test_summarize_numerical_health_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "points.csv"
    csv_path.write_text(
        "L,gamma,point_status,nan_detected,range_violation\n"
        "9,1.0,ok,false,false\n"
        "9,2.0,nan,true,false\n"
        "9,3.0,range_violation,false,true\n",
        encoding="utf-8",
    )
    stats = summarize_numerical_health_from_csv(csv_path)
    assert stats["error_count"] == 2
    assert stats["nan_count"] == 1
    assert stats["range_violation_count"] == 1
    assert stats["instability_count"] == 2
