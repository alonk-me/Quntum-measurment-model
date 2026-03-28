"""
Hybrid CPU/GPU executor for parameter sweeps with deterministic queue and checkpoint support.

Orchestrates parameter sweep via:
- Shared deterministic work queue
- CPU worker threads for small systems  
- GPU worker thread for large systems
- Result writer thread for atomic CSV appends
- Optional memmap checkpointing for restartability
"""

from __future__ import annotations

import csv
import logging
import queue
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from quantum_measurement.backends import get_backend
from .work_queue import DeterministicWorkQueue, WorkItem, RouteHint
from .memmap_checkpoint import MemmapCheckpointWriter

logger = logging.getLogger(__name__)


@dataclass
class HybridExecutorConfig:
    """Configuration for hybrid CPU/GPU executor."""
    
    n_cpu_workers: int = 4
    """Number of CPU worker threads."""
    
    n_gpu_workers: int = 1
    """Number of GPU worker threads (typically 1 due to device context constraints)."""
    
    n_aware_threshold: int = 500
    """Complexity threshold for CPU/GPU routing (qubits/size estimate)."""
    
    gpu_streams: int = 2
    """Number of GPU streams (Phase 3 feature, for now unused)."""
    
    checkpoint_every_steps: int = 50
    """Save intermediate state every N evolution steps."""
    
    enable_memmap: bool = True
    """Enable memory-mapped checkpointing for restartability."""
    
    route_policy: str = "n_aware"
    """Routing policy: 'n_aware' or 'gpu_exclusive'."""
    
    precision_policy: str = "fp64"
    """Precision strategy: 'fp64' or 'mixed' (Phase 4 feature)."""
    
    base_seed: int = 42
    """Base seed for reproducible RNG seeding."""
    
    queue_maxsize: int = 128
    """Max size for work queue (for backpressure)."""
    
    verbose: bool = True
    """Enable verbose logging."""
    
    continue_on_error: bool = True
    """Continue sweep on task error (vs fail-fast)."""
    
    checkpoint_dir: Path | str | None = None
    """Directory for checkpoint files; if None, checkpointing disabled."""


class HybridExecutor:
    """
    Hybrid CPU/GPU executor for deterministic, restartable parameter sweeps.
    
    Maintains a shared deterministic work queue, routes tasks via N-aware policy,
    and provides fault-tolerant execution with optional memmap checkpointing.
    """
    
    def __init__(self, config: HybridExecutorConfig) -> None:
        """
        Initialize hybrid executor.
        
        Args:
            config: HybridExecutorConfig with execution parameters.
        """
        self.config = config
        self._lock = threading.RLock()
        
        # Result accumulation
        self.results: list[dict[str, Any]] = []
        self.errors: list[tuple[str, Exception]] = []
        
        # Checkpointing
        self.checkpoint_writer: MemmapCheckpointWriter | None = None
        if config.enable_memmap and config.checkpoint_dir is not None:
            self.checkpoint_writer = MemmapCheckpointWriter(
                checkpoint_dir=config.checkpoint_dir,
                checkpoint_interval=config.checkpoint_every_steps,
                base_seed=config.base_seed,
            )
    
    def run_sweep(
        self,
        L_values: Iterable[int],
        gamma_grid: Iterable[float],
        simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
        backend_device: str = "hybrid",
        output_csv: Path | str | None = None,
        resume: bool = True,
        csv_header: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run parameter sweep using hybrid CPU/GPU orchestration.
        
        Args:
            L_values: System sizes to sweep.
            gamma_grid: Loss rates to sweep.
            simulator_factory: Callable that creates simulator and runs trajectory.
            backend_device: Device target ('cpu', 'gpu', or 'hybrid' for auto-routing).
            output_csv: Path to output CSV file.
            resume: Whether to skip completed (L, gamma) pairs from existing CSV.
            csv_header: Column names for CSV output.
        
        Returns:
            List of result dicts, one per completed task.
        """
        # Validate GPU availability for GPU-routed tasks
        if backend_device in ("gpu", "hybrid"):
            try:
                _ = get_backend("gpu")
            except RuntimeError as e:
                if backend_device == "gpu":
                    raise RuntimeError(f"GPU backend requested but unavailable: {e}") from e
                # For hybrid, fallback to CPU is acceptable in Phase 1
                if self.config.verbose:
                    logger.warning(f"GPU unavailable in hybrid mode; falling back to CPU: {e}")
        
        # Load previous progress if resuming
        completed_pairs: set[tuple[int, float]] = set()
        if resume and output_csv is not None:
            csv_path = Path(output_csv)
            if csv_path.exists():
                completed_pairs = self._load_completed_pairs(csv_path)
                if self.config.verbose:
                    logger.info(f"Resuming from CSV; {len(completed_pairs)} completed pairs")
        
        # Load checkpoint index if available
        if self.checkpoint_writer is not None and resume:
            checkpoint_pairs = self.checkpoint_writer.get_completed_pairs()
            completed_pairs.update(checkpoint_pairs)
            if self.config.verbose:
                logger.info(f"Loaded {len(checkpoint_pairs)} pairs from checkpoint")
        
        # Build deterministic work queue
        work_queue = DeterministicWorkQueue(
            L_values=L_values,
            gamma_grid=gamma_grid,
            base_seed=self.config.base_seed,
            n_aware_threshold=self.config.n_aware_threshold,
            completed_pairs=completed_pairs,
        )
        
        n_pending = len(work_queue)
        if self.config.verbose:
            logger.info(f"Pending tasks: {n_pending}")
        
        if n_pending == 0:
            return []
        
        # Prepare CSV writer
        csv_path = Path(output_csv) if output_csv is not None else None
        
        # Reset result accumulators
        self.results = []
        self.errors = []
        
        # Start worker threads
        result_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        work_item_queue: queue.Queue[WorkItem | None] = queue.Queue(maxsize=self.config.queue_maxsize)
        
        stop_event = threading.Event()
        
        # CPU workers
        cpu_threads = []
        for i in range(self.config.n_cpu_workers):
            t = threading.Thread(
                target=self._cpu_worker,
                args=(i, work_item_queue, result_queue, simulator_factory, backend_device, stop_event),
                daemon=False,
            )
            t.start()
            cpu_threads.append(t)
        
        # GPU worker
        gpu_thread = None
        if backend_device in ("gpu", "hybrid") and self.config.n_gpu_workers > 0:
            gpu_thread = threading.Thread(
                target=self._gpu_worker,
                args=(work_item_queue, result_queue, simulator_factory, backend_device, stop_event),
                daemon=False,
            )
            gpu_thread.start()
        
        # Result writer thread
        writer_thread = threading.Thread(
            target=self._result_writer,
            args=(result_queue, csv_path, csv_header, stop_event),
            daemon=False,
        )
        writer_thread.start()
        
        # Queue producer thread
        producer_thread = threading.Thread(
            target=self._queue_producer,
            args=(work_queue, work_item_queue, n_pending, stop_event),
            daemon=False,
        )
        producer_thread.start()
        
        # Wait for all work to complete
        producer_thread.join()
        work_item_queue.join()
        
        # Signal workers to stop
        stop_event.set()
        
        # Send sentinel values
        for _ in range(self.config.n_cpu_workers + (1 if gpu_thread else 0)):
            work_item_queue.put(None)
        
        # Wait for result writer to finish
        result_queue.put(None)
        writer_thread.join()
        
        # Wait for all worker threads
        for t in cpu_threads:
            t.join(timeout=10)
        if gpu_thread is not None:
            gpu_thread.join(timeout=10)
        
        if self.config.verbose:
            logger.info(
                f"Sweep complete: {len(self.results)} completed, {len(self.errors)} errors"
            )
        
        return self.results
    
    def _queue_producer(
        self,
        work_queue: DeterministicWorkQueue,
        out_queue: queue.Queue[WorkItem | None],
        n_total: int,
        stop_event: threading.Event,
    ) -> None:
        """Producer thread: feeds work_queue items to worker threads."""
        try:
            count = 0
            while not stop_event.is_set() and count < n_total:
                item = work_queue.get_next()
                if item is None:
                    break
                out_queue.put(item)
                count += 1
        except Exception as e:
            logger.error(f"Producer thread error: {e}")
            stop_event.set()
    
    def _cpu_worker(
        self,
        worker_id: int,
        work_queue: queue.Queue[WorkItem | None],
        result_queue: queue.Queue[dict[str, Any] | None],
        simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
        backend_device: str,
        stop_event: threading.Event,
    ) -> None:
        """CPU worker thread: executes CPU-routed tasks."""
        while not stop_event.is_set():
            try:
                item = work_queue.get(timeout=1)
                if item is None:
                    break
                
                # Only process CPU-routed tasks
                if item.route_hint != RouteHint.CPU:
                    work_queue.put(item)  # Re-queue for GPU
                    work_queue.task_done()
                    continue
                
                result = self._execute_task_item(
                    item,
                    simulator_factory,
                    backend_device="cpu",
                    worker_type="cpu",
                    worker_id=worker_id,
                )
                
                result_queue.put(result)
                work_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"CPU worker {worker_id} error: {e}")
                with self._lock:
                    self.errors.append((f"cpu_worker_{worker_id}", e))
                work_queue.task_done()
    
    def _gpu_worker(
        self,
        work_queue: queue.Queue[WorkItem | None],
        result_queue: queue.Queue[dict[str, Any] | None],
        simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
        backend_device: str,
        stop_event: threading.Event,
    ) -> None:
        """GPU worker thread: executes GPU-routed tasks."""
        while not stop_event.is_set():
            try:
                item = work_queue.get(timeout=1)
                if item is None:
                    break
                
                # Process GPU-routed items + leftover CPU items
                result = self._execute_task_item(
                    item,
                    simulator_factory,
                    backend_device="gpu",
                    worker_type="gpu",
                    worker_id=0,
                )
                
                result_queue.put(result)
                work_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"GPU worker error: {e}")
                with self._lock:
                    self.errors.append(("gpu_worker", e))
                work_queue.task_done()
    
    def _execute_task_item(
        self,
        item: WorkItem,
        simulator_factory: Callable[[int, float, str, np.random.Generator], dict[str, Any]],
        backend_device: str,
        worker_type: str,
        worker_id: int,
    ) -> dict[str, Any]:
        """Execute a single task item and return result dict."""
        start_time = time.time()
        
        try:
            # Create seeded RNG
            rng = np.random.default_rng(item.seed)
            
            # Call simulator factory
            result = simulator_factory(
                item.L,
                item.gamma,
                backend_device,
                rng,
            )
            
            if not isinstance(result, dict):
                raise TypeError("simulator_factory must return dict[str, Any]")
            
            # Ensure required fields
            result.setdefault("L", item.L)
            result.setdefault("gamma", item.gamma)
            result.setdefault("route", item.route_hint.value)
            result.setdefault("worker_type", worker_type)
            result.setdefault("worker_id", worker_id)
            result.setdefault("seed", item.seed)
            
            elapsed = time.time() - start_time
            result.setdefault("runtime_sec", elapsed)
            
            # Save checkpoint if enabled
            if self.checkpoint_writer is not None:
                checkpoint_meta = {
                    "worker_type": worker_type,
                    "runtime_sec": elapsed,
                }
                # Note: for Phase 1, we don't checkpoint intermediate G matrix
                # That comes in Phase 5 when we track step-level progress
                self.checkpoint_writer.save_task_checkpoint(
                    item.L,
                    item.gamma,
                    G_matrix=np.zeros((item.L, item.L), dtype=np.complex128),  # Placeholder
                    step=0,
                    task_metadata=checkpoint_meta,
                )
            
            with self._lock:
                self.results.append(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Task error for L={item.L}, gamma={item.gamma}: {e}")
            with self._lock:
                self.errors.append((f"L{item.L}_g{item.gamma:.4f}", e))
            
            if self.config.continue_on_error:
                # Return partial result with error marker
                return {
                    "L": item.L,
                    "gamma": item.gamma,
                    "error": str(e),
                    "route": item.route_hint.value,
                    "worker_type": worker_type,
                }
            else:
                raise
    
    def _result_writer(
        self,
        result_queue: queue.Queue[dict[str, Any] | None],
        csv_path: Path | None,
        csv_header: list[str] | None,
        stop_event: threading.Event,
    ) -> None:
        """Result writer thread: atomic CSV appends."""
        if csv_path is None:
            # No CSV output; drain queue for cleanups
            while True:
                try:
                    item = result_queue.get(timeout=1)
                    if item is None:
                        break
                except queue.Empty:
                    continue
        else:
            csv_path = Path(csv_path)
            while True:
                try:
                    item = result_queue.get(timeout=1)
                    if item is None:
                        break
                    
                    self._append_result_row(csv_path, item, csv_header)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Result writer error: {e}")
    
    def _append_result_row(
        self,
        csv_path: Path,
        row: dict[str, Any],
        csv_header: list[str] | None,
    ) -> None:
        """Thread-safe CSV row append."""
        header = csv_header if csv_header is not None else list(row.keys())
        
        # Ensure header written
        self._ensure_csv_header(csv_path, header)
        
        # Atomic append with lock
        with self._lock:
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([row.get(key) for key in header])
    
    def _ensure_csv_header(self, csv_path: Path, header: list[str]) -> None:
        """Ensure CSV file has header row."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists() and csv_path.stat().st_size > 0:
            return
        
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    @staticmethod
    def _load_completed_pairs(csv_path: Path) -> set[tuple[int, float]]:
        """Load set of completed (L, gamma) pairs from CSV."""
        completed: set[tuple[int, float]] = set()
        if not csv_path.exists():
            return completed
        
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for rec in reader:
                    if rec.get("L") and rec.get("gamma"):
                        completed.add((int(rec["L"]), float(rec["gamma"])))
        except Exception as e:
            logger.warning(f"Failed to load completed pairs from CSV: {e}")
        
        return completed
