from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CorePolicy:
    """Core allocation defaults for an experiment."""

    default_requested: int | None = None
    allow_override: bool = True


@dataclass(frozen=True)
class MonitorProfile:
    """Monitor defaults for an experiment type."""

    script_path: str
    output_path: str
    interval_seconds: int = 30


@dataclass(frozen=True)
class PlotProfile:
    """Plotting defaults for an experiment type."""

    mode: str
    notes: str = ""


@dataclass(frozen=True)
class ExperimentDefinition:
    """Declarative experiment registration record."""

    name: str
    description: str
    aggregate_table_hint: str
    raw_series_default: bool = False
    core_policy: CorePolicy = field(default_factory=CorePolicy)
    monitor_profile: MonitorProfile | None = None
    plot_profile: PlotProfile | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
