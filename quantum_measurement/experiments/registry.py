from __future__ import annotations

from pathlib import Path

from .definitions import CorePolicy, ExperimentDefinition, MonitorProfile, PlotProfile


class ExperimentRegistry:
    def __init__(self) -> None:
        self._defs: dict[str, ExperimentDefinition] = {}

    def register(self, definition: ExperimentDefinition) -> None:
        self._defs[definition.name] = definition

    def get(self, name: str) -> ExperimentDefinition:
        if name not in self._defs:
            raise KeyError(f"Unknown experiment type: {name}")
        return self._defs[name]

    def names(self) -> list[str]:
        return sorted(self._defs.keys())


DEFAULT_REGISTRY = ExperimentRegistry()


def _register_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    DEFAULT_REGISTRY.register(
        ExperimentDefinition(
            name="z2_scan",
            description="1+<z^2> parameter sweep",
            aggregate_table_hint="run_points",
            raw_series_default=False,
            core_policy=CorePolicy(default_requested=38, allow_override=True),
            monitor_profile=MonitorProfile(
                script_path=str(repo_root / "scripts" / "plot_z2_progress.py"),
                output_path=str(repo_root / "results" / "z2_scan" / "live_progress.png"),
                interval_seconds=30,
            ),
            plot_profile=PlotProfile(mode="z2_default"),
            tags=("sweep", "z2"),
        )
    )

    DEFAULT_REGISTRY.register(
        ExperimentDefinition(
            name="ninf_scan",
            description="n_inf(gamma) parameter sweep",
            aggregate_table_hint="run_points",
            raw_series_default=False,
            core_policy=CorePolicy(default_requested=38, allow_override=True),
            monitor_profile=MonitorProfile(
                script_path=str(repo_root / "scripts" / "plot_progress.py"),
                output_path=str(repo_root / "results" / "ninf_scan" / "live_progress.png"),
                interval_seconds=30,
            ),
            plot_profile=PlotProfile(mode="ninf_default"),
            tags=("sweep", "ninf"),
        )
    )


_register_defaults()
