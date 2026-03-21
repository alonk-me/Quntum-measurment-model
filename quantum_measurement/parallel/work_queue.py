"""
Work queue module for hybrid CPU/GPU executor.

Provides deterministic task ordering, route hints, and seed generation
via SeedSequence.spawn() for reproducible RNG assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np


class RouteHint(Enum):
    """Route hints for N-aware task routing."""
    CPU = "cpu"
    GPU = "gpu"
    AUTODETECT = "autodetect"


@dataclass(frozen=True)
class WorkItem:
    """A single task in the work queue with metadata for routing and seeding."""
    
    index: int
    """Sequential task index for deterministic ordering."""
    
    L: int
    """System size (number of qubits)."""
    
    gamma: float
    """Loss rate parameter."""
    
    seed: int
    """Seeded RNG value for this task's trajectory ensemble."""
    
    route_hint: RouteHint
    """Suggested compute route: CPU, GPU, or AUTODETECT."""
    
    complexity_score: float
    """Complexity estimate (e.g., L for system size correlation)."""
    
    @property
    def task_key(self) -> tuple[int, float]:
        """Resume identity key for this task: (L, gamma)."""
        return (self.L, self.gamma)


class DeterministicWorkQueue:
    """
    Deterministic work queue with stable seed assignment.
    
    Maintains deterministic FIFO order and reproducible RNG seeding
    via SeedSequence.spawn(). Supports resume by skipping completed
    (L, gamma) pairs.
    """
    
    def __init__(
        self,
        L_values: Iterable[int],
        gamma_grid: Iterable[float],
        base_seed: int = 42,
        n_aware_threshold: int = 500,
        completed_pairs: set[tuple[int, float]] | None = None,
    ) -> None:
        """
        Initialize deterministic work queue.
        
        Args:
            L_values: System sizes to sweep.
            gamma_grid: Loss rates to sweep.
            base_seed: Base for SeedSequence.
            n_aware_threshold: Threshold for CPU/GPU routing (N=threshold ≈ crossover).
            completed_pairs: Set of (L, gamma) pairs to skip (for resume).
        """
        self.base_seed = base_seed
        self.n_aware_threshold = n_aware_threshold
        self.completed_pairs = completed_pairs or set()
        
        # Build deterministic task list
        self._items: list[WorkItem] = []
        idx = 0
        L_values_list = list(L_values)
        gamma_grid_list = list(gamma_grid)
        
        # Pre-generate all seeds upfront for reproducibility
        total_tasks = len(L_values_list) * len(gamma_grid_list)
        seed_seq = np.random.SeedSequence(base_seed)
        child_seqs = seed_seq.spawn(total_tasks)
        seed_idx = 0
        
        for L in L_values_list:
            for gamma in gamma_grid_list:
                task_key = (int(L), float(gamma))
                if task_key in self.completed_pairs:
                    seed_idx += 1
                    continue
                
                seed = int(child_seqs[seed_idx].generate_state(1)[0])
                
                # N-aware routing: CPU for small systems, GPU for large
                complexity = float(L)
                if complexity <= self.n_aware_threshold:
                    route = RouteHint.CPU
                else:
                    route = RouteHint.GPU
                
                item = WorkItem(
                    index=idx,
                    L=int(L),
                    gamma=float(gamma),
                    seed=seed,
                    route_hint=route,
                    complexity_score=complexity,
                )
                self._items.append(item)
                idx += 1
                seed_idx += 1
        
        self._position = 0
    
    def __len__(self) -> int:
        """Return number of remaining tasks."""
        return len(self._items) - self._position
    
    def __iter__(self):
        """Iterate over remaining work items in order."""
        while self._position < len(self._items):
            yield self._items[self._position]
            self._position += 1
    
    def get_next(self) -> WorkItem | None:
        """Get next work item or None if queue exhausted."""
        if self._position < len(self._items):
            item = self._items[self._position]
            self._position += 1
            return item
        return None
    
    def peek(self) -> WorkItem | None:
        """Peek at next item without consuming it."""
        if self._position < len(self._items):
            return self._items[self._position]
        return None
    
    def reset(self) -> None:
        """Reset position to start (for debugging/testing)."""
        self._position = 0
    
    def items_by_route(self) -> tuple[list[WorkItem], list[WorkItem]]:
        """Partition remaining items by route hint. Returns (cpu_items, gpu_items)."""
        cpu_items = []
        gpu_items = []
        for item in self._items[self._position:]:
            if item.route_hint == RouteHint.CPU:
                cpu_items.append(item)
            else:  # GPU or AUTODETECT
                gpu_items.append(item)
        return cpu_items, gpu_items
