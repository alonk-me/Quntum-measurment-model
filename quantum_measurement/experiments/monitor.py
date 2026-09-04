from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import signal
import subprocess
import sys
import time


@dataclass
class MonitorSessionConfig:
    script_path: str
    output_path: str
    csv_path: str
    interval_seconds: int = 30
    enabled: bool = True


class MonitorSession:
    """Process lifecycle manager for per-run visual monitors."""

    def __init__(self, config: MonitorSessionConfig) -> None:
        self.config = config
        self.process: subprocess.Popen | None = None
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None

    def start(self) -> None:
        if not self.config.enabled:
            return
        if self.process is not None and self.process.poll() is None:
            return

        script = Path(self.config.script_path).resolve()
        cmd = [
            sys.executable,
            str(script),
            "--csv",
            str(Path(self.config.csv_path).resolve()),
            "--interval",
            str(int(self.config.interval_seconds)),
        ]
        self.started_at = datetime.now(timezone.utc)
        self.process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

    def stop(self, *, cleanup_visual: bool = True) -> dict:
        self.ended_at = datetime.now(timezone.utc)
        status = "disabled"
        pid = None

        if self.process is not None:
            pid = self.process.pid
            if self.process.poll() is None:
                status = "terminated"
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    else:
                        self.process.terminate()
                except Exception:
                    pass

                try:
                    self.process.wait(timeout=5)
                except Exception:
                    try:
                        if os.name != "nt":
                            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                        else:
                            self.process.kill()
                    except Exception:
                        pass
            else:
                status = "exited"

        if cleanup_visual:
            output_path = Path(self.config.output_path)
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass

        duration = None
        if self.started_at is not None and self.ended_at is not None:
            duration = max(0.0, (self.ended_at - self.started_at).total_seconds())

        return {
            "script_path": str(Path(self.config.script_path).resolve()),
            "output_path": str(Path(self.config.output_path).resolve()),
            "pid": pid,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_sec": duration,
            "status": status,
        }


def summarize_numerical_health_from_csv(csv_path: str | Path) -> dict:
    """Extract lightweight post-run numerical health stats from CSV."""
    import csv

    path = Path(csv_path)
    if not path.exists():
        return {
            "error_count": 0,
            "nan_count": 0,
            "range_violation_count": 0,
            "instability_count": 0,
        }

    nan_count = 0
    range_count = 0
    error_count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            point_status = str(row.get("point_status", "")).strip().lower()
            if point_status in {"nan", "range_violation", "error"}:
                error_count += 1
            if str(row.get("nan_detected", "")).strip().lower() in {"1", "true", "yes"}:
                nan_count += 1
            if str(row.get("range_violation", "")).strip().lower() in {"1", "true", "yes"}:
                range_count += 1

    return {
        "error_count": int(error_count),
        "nan_count": int(nan_count),
        "range_violation_count": int(range_count),
        "instability_count": int(nan_count + range_count),
    }
