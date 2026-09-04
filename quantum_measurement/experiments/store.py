from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_DB_PATH = Path("results") / "experiments.sqlite3"


class ExperimentStore:
    """SQLite-backed store for experiment runs and aggregated outputs."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = self.resolve_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @staticmethod
    def resolve_db_path(db_path: str | Path | None = None) -> Path:
        if db_path is not None:
            return Path(db_path).expanduser().resolve()
        env_path = os.getenv("QUANTUM_EXPERIMENT_DB_PATH")
        if env_path:
            return Path(env_path).expanduser().resolve()
        return (Path.cwd() / DEFAULT_DB_PATH).resolve()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiment_types (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    resume_enabled INTEGER NOT NULL DEFAULT 1,
                    config_json TEXT NOT NULL,
                    requested_cores INTEGER,
                    actual_cores INTEGER,
                    executor_kind TEXT,
                    backend_device TEXT,
                    csv_path TEXT,
                    event_log_path TEXT,
                    raw_series_enabled INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    FOREIGN KEY(experiment_type) REFERENCES experiment_types(name)
                );

                CREATE TABLE IF NOT EXISTS run_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    point_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    L INTEGER,
                    gamma REAL,
                    g REAL,
                    z2_mean REAL,
                    z2_plus_one REAL,
                    n_inf_sim REAL,
                    n_inf_exact REAL,
                    abs_error REAL,
                    rel_error REAL,
                    converged INTEGER,
                    steps INTEGER,
                    runtime_sec REAL,
                    nan_detected INTEGER,
                    range_violation INTEGER,
                    point_status TEXT,
                    payload_json TEXT NOT NULL,
                    UNIQUE(run_id, point_key),
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    ts TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS monitor_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    script_path TEXT NOT NULL,
                    output_path TEXT,
                    pid INTEGER,
                    started_at TEXT,
                    ended_at TEXT,
                    duration_sec REAL,
                    status TEXT,
                    error_count INTEGER,
                    nan_count INTEGER,
                    range_violation_count INTEGER,
                    instability_count INTEGER,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_runs_type_time ON runs(experiment_type, started_at);
                CREATE INDEX IF NOT EXISTS idx_run_points_lookup ON run_points(run_id, L, gamma);
                CREATE INDEX IF NOT EXISTS idx_events_lookup ON events(run_id, ts, event_type);
                """
            )

    def register_experiment_type(self, name: str, description: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO experiment_types(name, description, created_at)
                VALUES(?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET description=excluded.description
                """,
                (name, description, _utcnow_iso()),
            )

    def create_run(
        self,
        *,
        experiment_type: str,
        status: str,
        config: dict[str, Any],
        resume_enabled: bool,
        requested_cores: int | None,
        actual_cores: int | None,
        executor_kind: str | None,
        backend_device: str | None,
        csv_path: str | None,
        event_log_path: str | None,
        raw_series_enabled: bool,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs(
                    experiment_type, started_at, status, resume_enabled, config_json,
                    requested_cores, actual_cores, executor_kind, backend_device,
                    csv_path, event_log_path, raw_series_enabled
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_type,
                    _utcnow_iso(),
                    status,
                    int(bool(resume_enabled)),
                    json.dumps(config, sort_keys=True),
                    requested_cores,
                    actual_cores,
                    executor_kind,
                    backend_device,
                    csv_path,
                    event_log_path,
                    int(bool(raw_series_enabled)),
                ),
            )
            return int(cur.lastrowid)

    def finalize_run(self, run_id: int, status: str, error_message: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status=?, ended_at=?, error_message=?
                WHERE id=?
                """,
                (status, _utcnow_iso(), error_message, run_id),
            )

    def log_event(
        self,
        run_id: int,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
        ts: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO events(run_id, ts, event_type, message, payload_json)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    ts or _utcnow_iso(),
                    event_type,
                    message,
                    json.dumps(payload or {}, sort_keys=True),
                ),
            )

    def upsert_point(self, run_id: int, row: dict[str, Any]) -> None:
        L = _coerce_int(row.get("L"))
        gamma = _coerce_float(row.get("gamma"))
        point_key = f"{L}:{gamma}" if (L is not None and gamma is not None) else json.dumps(
            {"L": row.get("L"), "gamma": row.get("gamma")}, sort_keys=True
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO run_points(
                    run_id, point_key, created_at, L, gamma, g,
                    z2_mean, z2_plus_one, n_inf_sim, n_inf_exact,
                    abs_error, rel_error, converged, steps, runtime_sec,
                    nan_detected, range_violation, point_status, payload_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, point_key) DO UPDATE SET
                    g=excluded.g,
                    z2_mean=excluded.z2_mean,
                    z2_plus_one=excluded.z2_plus_one,
                    n_inf_sim=excluded.n_inf_sim,
                    n_inf_exact=excluded.n_inf_exact,
                    abs_error=excluded.abs_error,
                    rel_error=excluded.rel_error,
                    converged=excluded.converged,
                    steps=excluded.steps,
                    runtime_sec=excluded.runtime_sec,
                    nan_detected=excluded.nan_detected,
                    range_violation=excluded.range_violation,
                    point_status=excluded.point_status,
                    payload_json=excluded.payload_json
                """,
                (
                    run_id,
                    point_key,
                    _utcnow_iso(),
                    L,
                    gamma,
                    _coerce_float(row.get("g")),
                    _coerce_float(row.get("z2_mean")),
                    _coerce_float(row.get("z2_plus_one")),
                    _coerce_float(row.get("n_inf_sim")),
                    _coerce_float(row.get("n_inf_exact")),
                    _coerce_float(row.get("abs_error")),
                    _coerce_float(row.get("rel_error")),
                    _coerce_boolint(row.get("converged")),
                    _coerce_int(row.get("steps")),
                    _coerce_float(row.get("runtime_sec")),
                    _coerce_boolint(row.get("nan_detected")),
                    _coerce_boolint(row.get("range_violation")),
                    _coerce_str(row.get("point_status")),
                    json.dumps(row, sort_keys=True, default=str),
                ),
            )

    def record_monitor_session(self, run_id: int, session: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO monitor_sessions(
                    run_id, script_path, output_path, pid, started_at, ended_at,
                    duration_sec, status, error_count, nan_count,
                    range_violation_count, instability_count, metadata_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _coerce_str(session.get("script_path")) or "",
                    _coerce_str(session.get("output_path")),
                    _coerce_int(session.get("pid")),
                    _coerce_str(session.get("started_at")),
                    _coerce_str(session.get("ended_at")),
                    _coerce_float(session.get("duration_sec")),
                    _coerce_str(session.get("status")),
                    _coerce_int(session.get("error_count")),
                    _coerce_int(session.get("nan_count")),
                    _coerce_int(session.get("range_violation_count")),
                    _coerce_int(session.get("instability_count")),
                    json.dumps(session, sort_keys=True, default=str),
                ),
            )

    def add_artifact(self, run_id: int, artifact_type: str, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts(run_id, artifact_type, path, created_at, metadata_json)
                VALUES(?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    artifact_type,
                    str(Path(path).resolve()),
                    _utcnow_iso(),
                    json.dumps(metadata or {}, sort_keys=True, default=str),
                ),
            )


def _coerce_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None


def _coerce_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _coerce_boolint(v: Any) -> int | None:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(bool(v))
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    return None


def _coerce_str(v: Any) -> str | None:
    if v is None:
        return None
    return str(v)
