"""Multi-CPU FP64 backend orchestration using ProcessPoolExecutor.

Priority mapping:
- Accuracy: deterministic SeedSequence child assignment and ordered result reduction.
- Overflow: integrates structured overflow event counters and strict validation gate hooks.
- Speed: uses forkserver process context, chunked trajectory dispatch, and compact return payloads.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import csv
import os
import multiprocessing as mp
import sys
import time
import warnings

import numpy as np
import psutil

from quantum_measurement.numerics.overflow_log import OverflowLogger
from quantum_measurement.numerics.overflow_log import OverflowEvent
from quantum_measurement.parallel.trajectory_worker import (
    TrajectoryConfig,
    TrajectoryTask,
    run_trajectory_chunk,
)


NAN_MODES = ("fail_on_nan", "finish_full_sweep")


def _init_worker(
    worker_cores: list[int],
    shared_slot_counter: Any,
    shared_slot_lock: Any,
    mkl_threads: int,
    repo_root: str | None = None,
) -> None:
    # Ensure workers import in-repo sources first, not a stale site-packages install.
    if repo_root:
        root = os.path.abspath(repo_root)
        if root not in sys.path:
            sys.path.insert(0, root)

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
    nan_mode: str = "fail_on_nan"


class MultiCpuBackend:
    """Drop-in sweep executor with additional trajectory-chunk execution APIs."""

    def __init__(self, config: MultiCpuBackendConfig | None = None) -> None:
        self.config = config if config is not None else MultiCpuBackendConfig()
        if self.config.nan_mode not in NAN_MODES:
            raise ValueError(f"Invalid nan_mode={self.config.nan_mode}. Expected one of {NAN_MODES}.")
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
        repo_root = str(Path(__file__).resolve().parents[2])

        results_by_idx: list[dict[str, Any] | None] = [None] * len(tasks)
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                worker_cores,
                slot_counter,
                slot_lock,
                self.config.mkl_threads_per_worker,
                repo_root,
            ),
        )
        fatal_exc: Exception | None = None
        fatal_context: dict[str, Any] | None = None
        try:
            futures: list[Any] = []
            future_to_idx: dict[Any, int] = {}
            for idx, (L, gamma) in enumerate(tasks):
                self._emit_sweep_event(
                    event_type="sweep_point_submitted",
                    L=int(L),
                    gamma=float(gamma),
                    message="Sweep point submitted to worker pool.",
                    payload={
                        "idx": int(idx),
                        "seed": int(seeds[idx]),
                        "backend_device": str(backend_device),
                    },
                )
                fut = pool.submit(
                    _run_point,
                    int(L),
                    float(gamma),
                    backend_device,
                    int(seeds[idx]),
                    simulator_factory,
                )
                futures.append(fut)
                future_to_idx[fut] = idx

            for fut in as_completed(futures):
                idx = future_to_idx[fut]
                L, gamma = tasks[idx]
                seed = int(seeds[idx])
                try:
                    row = fut.result()
                except Exception as exc:
                    self._emit_sweep_event(
                        event_type="sweep_worker_exception",
                        L=int(L),
                        gamma=float(gamma),
                        message="Worker task raised an exception.",
                        payload={
                            "idx": int(idx),
                            "seed": int(seed),
                            "backend_device": str(backend_device),
                            "exception": repr(exc),
                        },
                    )
                    fatal_context = {
                        "idx": int(idx),
                        "L": int(L),
                        "gamma": float(gamma),
                        "seed": int(seed),
                    }
                    fatal_exc = RuntimeError(
                        "Sweep worker failed "
                        f"(idx={idx}, L={L}, gamma={gamma}, seed={seed}, backend={backend_device})"
                    )
                    fatal_exc.__cause__ = exc
                    break

                try:
                    nan_detected, range_violation = self._validate_row(row)
                except Exception as exc:
                    self._emit_sweep_event(
                        event_type="invalid_result_row",
                        L=int(L),
                        gamma=float(gamma),
                        message="Result row validation failed.",
                        payload={
                            "idx": int(idx),
                            "seed": int(seed),
                            "exception": repr(exc),
                            "row_keys": sorted(list(row.keys())) if isinstance(row, dict) else [],
                        },
                    )
                    fatal_context = {
                        "idx": int(idx),
                        "L": int(L),
                        "gamma": float(gamma),
                        "seed": int(seed),
                    }
                    fatal_exc = exc
                    break

                if nan_detected:
                    self._emit_sweep_event(
                        event_type="nan_point_row",
                        L=int(L),
                        gamma=float(gamma),
                        message="Non-finite point result detected in sweep row.",
                        payload={
                            "idx": int(idx),
                            "seed": int(seed),
                            "nan_mode": self.config.nan_mode,
                            "point_status": row.get("point_status"),
                        },
                    )
                    if self.config.nan_mode == "fail_on_nan":
                        fatal_context = {
                            "idx": int(idx),
                            "L": int(L),
                            "gamma": float(gamma),
                            "seed": int(seed),
                        }
                        fatal_exc = ValueError(
                            f"NaN point row detected (idx={idx}, L={L}, gamma={gamma}, seed={seed})"
                        )
                        break

                if range_violation:
                    self._emit_sweep_event(
                        event_type="range_violation_point_row",
                        L=int(L),
                        gamma=float(gamma),
                        message="Out-of-range z2_mean detected in sweep row.",
                        payload={
                            "idx": int(idx),
                            "seed": int(seed),
                            "z2_mean": row.get("z2_mean"),
                            "point_status": row.get("point_status"),
                        },
                    )

                if csv_path is not None:
                    self._append_result_row(csv_path, row, csv_header)
                self._emit_sweep_event(
                    event_type="sweep_point_completed",
                    L=int(L),
                    gamma=float(gamma),
                    message="Sweep point completed and accepted.",
                    payload={
                        "idx": int(idx),
                        "seed": int(seed),
                        "nan_detected": bool(row.get("nan_detected", False)),
                        "point_status": row.get("point_status", "ok"),
                        "runtime_sec": row.get("runtime_sec"),
                        "z2_mean": row.get("z2_mean"),
                        "phase_label": row.get("phase_label"),
                        "adaptive_schedule": row.get("adaptive_schedule"),
                    },
                )
                results_by_idx[idx] = row
        finally:
            if fatal_exc is not None:
                self._shutdown_executor_fast(pool)
            else:
                pool.shutdown(wait=True)

        if fatal_exc is not None:
            if fatal_context is not None:
                self._emit_sweep_event(
                    event_type="sweep_fail_fast_abort",
                    L=int(fatal_context["L"]),
                    gamma=float(fatal_context["gamma"]),
                    message="Sweep aborted and worker pool was force-stopped due to fatal condition.",
                    payload={
                        "idx": int(fatal_context["idx"]),
                        "seed": int(fatal_context["seed"]),
                        "nan_mode": self.config.nan_mode,
                        "exception": repr(fatal_exc),
                    },
                )
            raise fatal_exc

        return [row for row in results_by_idx if row is not None]

    def _shutdown_executor_fast(self, pool: ProcessPoolExecutor, grace_seconds: float = 3.0) -> None:
        """Best-effort fail-fast shutdown for in-flight workers after fatal conditions."""
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            return

        processes = list(getattr(pool, "_processes", {}).values())
        if not processes:
            return

        deadline = time.time() + float(max(0.1, grace_seconds))
        for proc in processes:
            if proc is not None and proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    continue

        for proc in processes:
            if proc is None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                proc.join(timeout=remaining)
            except Exception:
                continue

        for proc in processes:
            if proc is not None and proc.is_alive():
                try:
                    proc.kill()
                except Exception:
                    continue

    def run_trajectories(
        self,
        total_trajectories: int,
        config: TrajectoryConfig,
        progress_callback: Callable[[list[tuple[int, np.ndarray]], int, int], None] | None = None,
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
        # Use multiple waves of chunks so fast workers can keep pulling work.
        target_chunks = max(n_workers * 4, n_workers)
        chunk_size = max(1, total_trajectories // target_chunks)
        chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

        ctx = mp.get_context("forkserver")
        slot_counter = ctx.Value("i", 0)
        slot_lock = ctx.Lock()
        repo_root = str(Path(__file__).resolve().parents[2])

        all_results: list[tuple[int, np.ndarray]] = []
        merged_counts: dict[str, int] = {}

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                worker_cores,
                slot_counter,
                slot_lock,
                self.config.mkl_threads_per_worker,
                repo_root,
            ),
        ) as pool:
            future_to_chunk_len = {}
            futures = []
            for chunk in chunks:
                fut = pool.submit(
                    run_trajectory_chunk,
                    chunk,
                    config,
                    str(self.config.log_path) if self.config.log_path is not None else None,
                )
                futures.append(fut)
                future_to_chunk_len[fut] = len(chunk)

            completed_trajectories = 0
            for fut in as_completed(futures):
                chunk_out, counts = fut.result()
                all_results.extend(chunk_out)
                completed_trajectories += int(future_to_chunk_len.get(fut, len(chunk_out)))
                if progress_callback is not None:
                    progress_callback(chunk_out, completed_trajectories, total_trajectories)
                for key, value in counts.items():
                    merged_counts[key] = merged_counts.get(key, 0) + int(value)

        all_results.sort(key=lambda x: x[0])
        self.overflow_logger.merge_counts(merged_counts)
        return all_results, merged_counts

    def _validate_row(self, row: dict[str, Any]) -> tuple[bool, bool]:
        if "L" not in row or "gamma" not in row:
            raise ValueError("Each result row must include 'L' and 'gamma'.")

        if "z2_mean" not in row:
            raise ValueError("Each result row must include 'z2_mean'.")

        if "nan_detected" in row:
            nan_detected = bool(row["nan_detected"])
        else:
            z2_mean = float(row["z2_mean"])
            nan_detected = not np.isfinite(z2_mean)
            row["nan_detected"] = bool(nan_detected)

        if "range_violation" in row:
            range_violation = bool(row["range_violation"])
        else:
            z2_mean = float(row["z2_mean"])
            range_violation = bool(
                np.isfinite(z2_mean) and ((z2_mean < -1e-9) or (z2_mean > (1.0 + 1e-6)))
            )
            row["range_violation"] = bool(range_violation)

        if "point_status" not in row:
            if nan_detected:
                row["point_status"] = "nan"
            elif range_violation:
                row["point_status"] = "range_violation"
            else:
                row["point_status"] = "ok"
        return bool(nan_detected), bool(range_violation)

    def _emit_sweep_event(
        self,
        event_type: str,
        L: int,
        gamma: float,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = OverflowEvent(
            event_type=event_type,
            traj_id=-1,
            step=-1,
            L=int(L),
            gamma=float(gamma),
            message=message,
            payload=payload or {},
        )
        self.overflow_logger.emit(event)

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
