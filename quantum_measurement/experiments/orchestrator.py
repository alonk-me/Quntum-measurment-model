from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .definitions import ExperimentDefinition
from .monitor import MonitorSession, MonitorSessionConfig, summarize_numerical_health_from_csv
from .store import ExperimentStore


@dataclass
class RunContext:
    run_id: int
    experiment_name: str
    csv_path: Path | None


class ExperimentRunOrchestrator:
    """Coordinates run registration, monitor lifecycle, and DB ingestion."""

    def __init__(self, store: ExperimentStore, definition: ExperimentDefinition) -> None:
        self.store = store
        self.definition = definition
        self.monitor: MonitorSession | None = None

    def start_run(
        self,
        *,
        config: dict[str, Any],
        resume_enabled: bool,
        requested_cores: int | None,
        actual_cores: int | None,
        executor_kind: str | None,
        backend_device: str | None,
        csv_path: str | None,
        event_log_path: str | None,
        raw_series_enabled: bool,
        enable_monitor: bool = True,
        monitor_interval_seconds: int | None = None,
    ) -> RunContext:
        self.store.register_experiment_type(self.definition.name, self.definition.description)
        run_id = self.store.create_run(
            experiment_type=self.definition.name,
            status="running",
            config=config,
            resume_enabled=resume_enabled,
            requested_cores=requested_cores,
            actual_cores=actual_cores,
            executor_kind=executor_kind,
            backend_device=backend_device,
            csv_path=str(Path(csv_path).resolve()) if csv_path else None,
            event_log_path=str(Path(event_log_path).resolve()) if event_log_path else None,
            raw_series_enabled=raw_series_enabled,
        )
        self.store.log_event(run_id, "run_started", "Experiment run started.", payload={"config": config})

        if enable_monitor and self.definition.monitor_profile is not None and csv_path is not None:
            interval = (
                int(monitor_interval_seconds)
                if monitor_interval_seconds is not None
                else int(self.definition.monitor_profile.interval_seconds)
            )
            self.monitor = MonitorSession(
                MonitorSessionConfig(
                    script_path=self.definition.monitor_profile.script_path,
                    output_path=self.definition.monitor_profile.output_path,
                    csv_path=csv_path,
                    interval_seconds=interval,
                    enabled=True,
                )
            )
            self.monitor.start()
            self.store.log_event(
                run_id,
                "monitor_started",
                "Run monitor started.",
                payload={
                    "script": self.definition.monitor_profile.script_path,
                    "interval": interval,
                },
            )

        return RunContext(run_id=run_id, experiment_name=self.definition.name, csv_path=Path(csv_path).resolve() if csv_path else None)

    def ingest_csv_points(self, run_id: int, csv_path: str | Path | None) -> int:
        if csv_path is None:
            return 0
        path = Path(csv_path)
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.store.upsert_point(run_id, row)
                count += 1
        self.store.log_event(run_id, "csv_ingested", "CSV point rows ingested into DB.", payload={"rows": count})
        self.store.add_artifact(run_id, "raw_csv", path, metadata={"rows": count, "keep_all": True})
        return count

    def ingest_jsonl_events(self, run_id: int, event_log_path: str | Path | None) -> int:
        if event_log_path is None:
            return 0
        path = Path(event_log_path)
        if not path.exists():
            return 0

        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                self.store.log_event(
                    run_id,
                    event_type=str(rec.get("event_type", "event_log")),
                    message=str(rec.get("message", "event")),
                    payload=rec.get("payload") if isinstance(rec.get("payload"), dict) else {"raw": rec},
                    ts=str(rec.get("ts")) if rec.get("ts") is not None else None,
                )
                count += 1

        self.store.add_artifact(run_id, "event_log", path, metadata={"rows": count})
        return count

    def finish_run(
        self,
        *,
        run_id: int,
        status: str,
        csv_path: str | Path | None,
        event_log_path: str | Path | None,
        error_message: str | None = None,
    ) -> None:
        monitor_meta = None
        if self.monitor is not None:
            monitor_meta = self.monitor.stop(cleanup_visual=True)
            health = summarize_numerical_health_from_csv(csv_path) if csv_path is not None else {}
            monitor_meta.update(health)
            self.store.record_monitor_session(run_id, monitor_meta)
            self.store.log_event(run_id, "monitor_stopped", "Run monitor stopped.", payload=monitor_meta)

        points = self.ingest_csv_points(run_id, csv_path)
        events = self.ingest_jsonl_events(run_id, event_log_path)

        self.store.log_event(
            run_id,
            "run_finished",
            "Experiment run finalized.",
            payload={"status": status, "points_ingested": points, "events_ingested": events},
        )
        self.store.finalize_run(run_id, status=status, error_message=error_message)
