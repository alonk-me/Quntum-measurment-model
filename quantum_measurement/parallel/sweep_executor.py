from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import csv
import importlib
import multiprocessing as mp
import os
import warnings

import numpy as np


@dataclass(frozen=True)
class SweepTask:
    index: int
    L: int
    gamma: float


def _detect_gpu_count() -> int:
    if not importlib.util.find_spec("cupy"):
        return 0
    try:
        cp = importlib.import_module("cupy")
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _is_gpu_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda_error_out_of_memory" in msg or "cudamemoryerror" in msg


def _execute_task(
    task: SweepTask,
    simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
    backend_device: str,
    seed: int,
    gpu_device_id: int | None,
) -> dict[str, Any]:
    if backend_device == "gpu" and gpu_device_id is not None and importlib.util.find_spec("cupy"):
        cp = importlib.import_module("cupy")
        cp.cuda.Device(gpu_device_id).use()

    rng = np.random.default_rng(seed)
    result = simulator_factory(task.L, task.gamma, backend_device, rng)
    if not isinstance(result, dict):
        raise TypeError("simulator_factory must return dict[str, Any]")

    result.setdefault("L", task.L)
    result.setdefault("gamma", task.gamma)
    return result


class ParameterSweepExecutor:
    def __init__(
        self,
        parallel_backend: str = "sequential",
        n_workers: int | None = None,
        verbose: bool = True,
        base_seed: int = 42,
        continue_on_error: bool = True,
    ) -> None:
        if parallel_backend not in {"sequential", "multiprocessing", "ray"}:
            raise ValueError("parallel_backend must be one of {'sequential', 'multiprocessing', 'ray'}")
        self.parallel_backend = parallel_backend
        self.n_workers = None if n_workers is None else max(1, int(n_workers))
        self.verbose = verbose
        self.base_seed = int(base_seed)
        self.continue_on_error = continue_on_error

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
        if self.parallel_backend == "ray":
            raise NotImplementedError("Ray backend is deferred in this phase; use sequential or multiprocessing.")

        csv_path = Path(output_csv) if output_csv is not None else None
        completed: set[tuple[int, float]] = set()
        if csv_path is not None and resume and csv_path.exists():
            completed = self._load_completed_pairs(csv_path)
            if self.verbose:
                print(f"Resuming sweep from {csv_path}; completed points: {len(completed)}")

        tasks = self._build_tasks(L_values, gamma_grid, completed)
        if self.verbose:
            print(f"Pending tasks: {len(tasks)}")

        if not tasks:
            return []

        seeds = self._build_task_seeds(len(tasks))

        if self.parallel_backend == "sequential":
            return self._run_sequential(tasks, simulator_factory, backend_device, seeds, csv_path, csv_header)

        return self._run_multiprocessing(tasks, simulator_factory, backend_device, seeds, csv_path, csv_header)

    def _effective_cpu_workers(self, tasks: list[SweepTask]) -> int:
        requested = self.n_workers if self.n_workers is not None else (os.cpu_count() or 1)
        requested = max(1, int(requested))

        # L-aware cap to avoid RAM pressure at large matrix sizes.
        max_L = max((task.L for task in tasks), default=1)
        if max_L >= 129:
            requested = max(1, requested // 2)
        elif max_L >= 65:
            requested = max(1, (requested * 3) // 4)

        return min(requested, len(tasks))

    def _retry_on_gpu_oom(
        self,
        task: SweepTask,
        simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
        seed: int,
        exc: Exception,
    ) -> dict[str, Any] | None:
        if not _is_gpu_oom_error(exc):
            return None
        if self.verbose:
            warnings.warn(
                f"GPU OOM at L={task.L}, gamma={task.gamma}; retrying this task on CPU.",
                RuntimeWarning,
            )
        return _execute_task(task, simulator_factory, "cpu", seed, None)

    def _build_tasks(
        self,
        L_values: Iterable[int],
        gamma_grid: Iterable[float],
        completed: set[tuple[int, float]],
    ) -> list[SweepTask]:
        tasks: list[SweepTask] = []
        idx = 0
        for L in L_values:
            for gamma in gamma_grid:
                key = (int(L), float(gamma))
                if key in completed:
                    continue
                tasks.append(SweepTask(index=idx, L=int(L), gamma=float(gamma)))
                idx += 1
        return tasks

    def _build_task_seeds(self, n_tasks: int) -> list[int]:
        seed_seq = np.random.SeedSequence(self.base_seed)
        child_seqs = seed_seq.spawn(n_tasks)
        return [int(child.generate_state(1)[0]) for child in child_seqs]

    def _run_sequential(
        self,
        tasks: list[SweepTask],
        simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
        backend_device: str,
        seeds: list[int],
        csv_path: Path | None,
        csv_header: list[str] | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        gpu_count = _detect_gpu_count() if backend_device == "gpu" else 0

        for idx, task in enumerate(tasks):
            gpu_device_id = (idx % gpu_count) if gpu_count > 0 else None
            try:
                row = _execute_task(task, simulator_factory, backend_device, seeds[idx], gpu_device_id)
                self._validate_row(row)
                if csv_path is not None:
                    self._append_result_row(csv_path, row, csv_header)
                results.append(row)
            except Exception as exc:
                if backend_device == "gpu":
                    retry_row = self._retry_on_gpu_oom(task, simulator_factory, seeds[idx], exc)
                    if retry_row is not None:
                        self._validate_row(retry_row)
                        if csv_path is not None:
                            self._append_result_row(csv_path, retry_row, csv_header)
                        results.append(retry_row)
                        continue
                if self.continue_on_error:
                    warnings.warn(f"Task failed for L={task.L}, gamma={task.gamma}: {exc}", RuntimeWarning)
                else:
                    raise
        return results

    def _run_multiprocessing(
        self,
        tasks: list[SweepTask],
        simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
        backend_device: str,
        seeds: list[int],
        csv_path: Path | None,
        csv_header: list[str] | None,
    ) -> list[dict[str, Any]]:
        gpu_count = _detect_gpu_count() if backend_device == "gpu" else 0
        n_workers = self.n_workers if self.n_workers is not None else 1

        if backend_device == "gpu" and gpu_count > 0:
            if n_workers > gpu_count:
                warnings.warn(
                    f"Requested n_workers={n_workers} but only {gpu_count} GPU(s) detected; capping workers.",
                    RuntimeWarning,
                )
            n_workers = min(n_workers, gpu_count)
        elif backend_device == "cpu":
            n_workers = self._effective_cpu_workers(tasks)

        if n_workers < 1:
            n_workers = 1

        results: list[dict[str, Any]] = []
        with mp.Pool(processes=n_workers) as pool:
            async_jobs: list[tuple[SweepTask, int, Any]] = []
            for idx, task in enumerate(tasks):
                gpu_device_id = (idx % gpu_count) if (backend_device == "gpu" and gpu_count > 0) else None
                job = pool.apply_async(
                    _execute_task,
                    (task, simulator_factory, backend_device, seeds[idx], gpu_device_id),
                )
                async_jobs.append((task, seeds[idx], job))

            for task, seed, job in async_jobs:
                try:
                    row = job.get()
                    self._validate_row(row)
                    if csv_path is not None:
                        self._append_result_row(csv_path, row, csv_header)
                    results.append(row)
                except Exception as exc:
                    if backend_device == "gpu":
                        retry_row = self._retry_on_gpu_oom(task, simulator_factory, seed, exc)
                        if retry_row is not None:
                            self._validate_row(retry_row)
                            if csv_path is not None:
                                self._append_result_row(csv_path, retry_row, csv_header)
                            results.append(retry_row)
                            continue
                    if self.continue_on_error:
                        warnings.warn(f"Task failed for L={task.L}, gamma={task.gamma}: {exc}", RuntimeWarning)
                    else:
                        raise
        return results

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
