"""Multi-CPU FP64 backend orchestration using ProcessPoolExecutor.

Priority mapping:
- Accuracy: deterministic SeedSequence child assignment and ordered result reduction.
- Overflow: integrates structured overflow event counters and strict validation gate hooks.
- Speed: uses forkserver process context, chunked trajectory dispatch, and compact return payloads.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import csv
import os
import multiprocessing as mp
import warnings

import numpy as np
import psutil

from quantum_measurement.numerics.overflow_log import OverflowLogger
from quantum_measurement.parallel.trajectory_worker import (
    TrajectoryConfig,
    TrajectoryTask,
    run_trajectory_chunk,
)


def _init_worker(
    worker_cores: list[int],
    shared_slot_counter: Any,
    shared_slot_lock: Any,
    mkl_threads: int,
) -> None:
    os.environ.setdefault("MKL_NUM_THREADS", str(mkl_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(mkl_threads))

    with shared_slot_lock:
        slot = int(shared_slot_counter.value)
        shared_slot_counter.value += 1

    worker_core = int(worker_cores[slot]) if slot < len(worker_cores) else None

    if worker_core is not None:
        try:
            os.sched_setaffinity(0, {int(worker_core)})
        except Exception:
            try:
                p = psutil.Process()
                p.cpu_affinity([int(worker_core)])
            except Exception:
                warnings.warn(
                    f"Failed to pin worker slot {slot} to core {worker_core}; continuing without affinity.",
                    RuntimeWarning,
                )


def _available_worker_cores(reserve_cores: int) -> list[int]:
    try:
        available = sorted(int(c) for c in os.sched_getaffinity(0))
    except Exception:
        available = list(range(os.cpu_count() or 1))

    if reserve_cores <= 0:
        return available
    if len(available) <= reserve_cores:
        return available
    return available[reserve_cores:]


@dataclass(frozen=True)
class MultiCpuBackendConfig:
    max_workers: int = 38
    reserve_cores: int = 4
    mkl_threads_per_worker: int = 1
    master_seed: int = 42
    log_path: Path | str | None = None


class MultiCpuBackend:
    """Drop-in sweep executor with additional trajectory-chunk execution APIs."""

    def __init__(self, config: MultiCpuBackendConfig | None = None) -> None:
        self.config = config if config is not None else MultiCpuBackendConfig()
        self.overflow_logger = OverflowLogger(self.config.log_path)

    def run_sweep(
        self,
        L_values: Iterable[int],
        gamma_grid: Iterable[float],
        simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
        backend_device: str = "cpu",
        output_csv: Path | str | None = None,
        resume: bool = True,
        csv_header: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """API-compatible sweep execution to preserve current simulator interface."""
        csv_path = Path(output_csv) if output_csv is not None else None
        completed: set[tuple[int, float]] = set()
        if csv_path is not None and resume and csv_path.exists():
            completed = self._load_completed_pairs(csv_path)

        tasks: list[tuple[int, float]] = []
        for L in L_values:
            for gamma in gamma_grid:
                key = (int(L), float(gamma))
                if key in completed:
                    continue
                tasks.append(key)

        if not tasks:
            return []

        seed_seq = np.random.SeedSequence(self.config.master_seed)
        child = seed_seq.spawn(len(tasks))
        seeds = [int(s.generate_state(1)[0]) for s in child]

        ctx = mp.get_context("forkserver")
        available_cores = _available_worker_cores(self.config.reserve_cores)
        if not available_cores:
            available_cores = _available_worker_cores(0)
        n_workers = max(1, min(self.config.max_workers, len(tasks), len(available_cores)))
        worker_cores = available_cores[:n_workers]
        slot_counter = ctx.Value("i", 0)
        slot_lock = ctx.Lock()

        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(worker_cores, slot_counter, slot_lock, self.config.mkl_threads_per_worker),
        ) as pool:
            futures = []
            for idx, (L, gamma) in enumerate(tasks):
                futures.append(
                    pool.submit(
                        _run_point,
                        int(L),
                        float(gamma),
                        backend_device,
                        int(seeds[idx]),
                        simulator_factory,
                    )
                )

            for fut in futures:
                row = fut.result()
                self._validate_row(row)
                if csv_path is not None:
                    self._append_result_row(csv_path, row, csv_header)
                results.append(row)

        return results

    def run_trajectories(
        self,
        total_trajectories: int,
        config: TrajectoryConfig,
    ) -> tuple[list[tuple[int, np.ndarray]], dict[str, int]]:
        """Execute trajectory chunks across workers and return compact payload only."""
        if total_trajectories < 1:
            return [], {}

        seed_seq = np.random.SeedSequence(self.config.master_seed)
        child = seed_seq.spawn(total_trajectories)
        tasks = [
            TrajectoryTask(traj_id=i, child_seed=int(child[i].generate_state(1)[0]))
            for i in range(total_trajectories)
        ]

        available_cores = _available_worker_cores(self.config.reserve_cores)
        if not available_cores:
            available_cores = _available_worker_cores(0)
        n_workers = max(1, min(self.config.max_workers, total_trajectories, len(available_cores)))
        worker_cores = available_cores[:n_workers]
        chunk_size = (total_trajectories + n_workers - 1) // n_workers
        chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

        ctx = mp.get_context("forkserver")
        slot_counter = ctx.Value("i", 0)
        slot_lock = ctx.Lock()

        all_results: list[tuple[int, np.ndarray]] = []
        merged_counts: dict[str, int] = {}

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(worker_cores, slot_counter, slot_lock, self.config.mkl_threads_per_worker),
        ) as pool:
            futures = [
                pool.submit(
                    run_trajectory_chunk,
                    chunk,
                    config,
                    str(self.config.log_path) if self.config.log_path is not None else None,
                )
                for chunk in chunks
            ]

            for fut in futures:
                chunk_out, counts = fut.result()
                all_results.extend(chunk_out)
                for key, value in counts.items():
                    merged_counts[key] = merged_counts.get(key, 0) + int(value)

        all_results.sort(key=lambda x: x[0])
        self.overflow_logger.merge_counts(merged_counts)
        return all_results, merged_counts

    def _validate_row(self, row: dict[str, Any]) -> None:
        if "L" not in row or "gamma" not in row:
            raise ValueError("Each result row must include 'L' and 'gamma'.")

    def _load_completed_pairs(self, csv_path: Path) -> set[tuple[int, float]]:
        completed: set[tuple[int, float]] = set()
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                completed.add((int(rec["L"]), float(rec["gamma"])))
        return completed

    def _append_result_row(self, csv_path: Path, row: dict[str, Any], csv_header: list[str] | None) -> None:
        header = csv_header if csv_header is not None else list(row.keys())
        self._ensure_csv_header(csv_path, header)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row.get(key) for key in header])

    def _ensure_csv_header(self, csv_path: Path, header: list[str]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def _run_point(
    L: int,
    gamma: float,
    backend_device: str,
    seed: int,
    simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    row = simulator_factory(L, gamma, backend_device, rng)
    row.setdefault("L", L)
    row.setdefault("gamma", gamma)
    return row
