from __future__ import annotations

import sqlite3
from pathlib import Path


def list_recent_runs(db_path: str | Path, limit: int = 20) -> list[dict]:
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, experiment_type, status, started_at, ended_at,
                   requested_cores, actual_cores, csv_path
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def load_run_points(db_path: str | Path, run_id: int) -> list[dict]:
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT L, gamma, g, z2_mean, z2_plus_one, n_inf_sim, n_inf_exact,
                   abs_error, rel_error, converged, steps, runtime_sec,
                   nan_detected, range_violation, point_status
            FROM run_points
            WHERE run_id = ?
            ORDER BY L, gamma
            """,
            (int(run_id),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
