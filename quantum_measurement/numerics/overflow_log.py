"""Structured overflow and numerical-guard logging for multi-CPU trajectories.

Priority mapping:
- Accuracy: records every mitigation (regularization, clipping, renormalization) for auditability.
- Overflow: centralizes condition-number and projection-skip warnings to prevent silent corruption.
- Speed: uses compact JSON-line logging and in-memory counters with low overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import threading
import time


@dataclass
class OverflowEvent:
    event_type: str
    traj_id: int
    step: int
    L: int
    gamma: float
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        rec = {
            "ts": time.time(),
            "event_type": self.event_type,
            "traj_id": self.traj_id,
            "step": self.step,
            "L": self.L,
            "gamma": self.gamma,
            "message": self.message,
            "payload": self.payload,
        }
        return json.dumps(rec, separators=(",", ":"))


class OverflowLogger:
    """Thread-safe structured logger and counters for numerical guard events."""

    def __init__(self, log_path: Path | str | None = None) -> None:
        self.log_path = Path(log_path) if log_path is not None else None
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: OverflowEvent) -> None:
        with self._lock:
            self._counts[event.event_type] = self._counts.get(event.event_type, 0) + 1
            if self.log_path is not None:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(event.to_json_line())
                    f.write("\n")

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def merge_counts(self, partial: dict[str, int]) -> None:
        with self._lock:
            for key, value in partial.items():
                self._counts[key] = self._counts.get(key, 0) + int(value)
