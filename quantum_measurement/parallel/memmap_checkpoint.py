"""
Memory-mapped checkpoint module for restartable hybrid executor.

Provides crash-safe checkpoint save/load for trajectory intermediate states
and enables resume without recomputing completed tasks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint session."""
    
    base_seed: int
    """Base seed for reproducibility."""
    
    checkpoint_interval: int
    """Steps between checkpoint saves."""
    
    schema_version: int
    """Checkpoint schema version for migration."""
    
    timestamp: float
    """UNIX timestamp when checkpoint was created."""
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckpointMetadata:
        """Deserialize from dict."""
        return cls(
            base_seed=int(d["base_seed"]),
            checkpoint_interval=int(d["checkpoint_interval"]),
            schema_version=int(d.get("schema_version", 1)),
            timestamp=float(d.get("timestamp", 0.0)),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)


class MemmapCheckpointWriter:
    """
    Writes and manages checkpoints for trajectory intermediate states.
    
    Creates memory-mapped array files for correlation matrices and maintains
    a crash-safe index of completed tasks.
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(
        self,
        checkpoint_dir: Path | str,
        checkpoint_interval: int = 50,
        base_seed: int = 42,
    ) -> None:
        """
        Initialize checkpoint writer.
        
        Args:
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_interval: Save state every N steps.
            base_seed: Base seed for reproducibility.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = int(checkpoint_interval)
        self.base_seed = int(base_seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize index
        self.index_path = self.checkpoint_dir / "checkpoint_index.json"
        self.completed_tasks: dict[tuple[int, float], dict[str, Any]] = self._load_index()
    
    def _load_index(self) -> dict[tuple[int, float], dict[str, Any]]:
        """Load completed task index from disk (crash-safe)."""
        if not self.index_path.exists():
            return {}
        
        try:
            with self.index_path.open("r") as f:
                raw = json.load(f)
            
            # Convert string keys back to (L, gamma) tuples
            completed = {}
            for key_str, metadata in raw.items():
                L, gamma = json.loads(key_str)
                completed[(L, float(gamma))] = metadata
            
            logger.info(f"Loaded {len(completed)} completed tasks from checkpoint index")
            return completed
        except Exception as e:
            logger.warning(f"Failed to load checkpoint index: {e}; starting fresh")
            return {}
    
    def _save_index(self) -> None:
        """Atomically save task index to disk (crash-safe via temp + rename)."""
        temp_path = self.index_path.with_suffix(".tmp")
        
        # Convert (L, gamma) tuples to JSON-serializable string keys
        raw = {
            json.dumps([L, gamma], separators=(',', ':')): metadata
            for (L, gamma), metadata in self.completed_tasks.items()
        }
        
        with temp_path.open("w") as f:
            json.dump(raw, f, indent=2)
        
        temp_path.replace(self.index_path)
    
    def save_task_checkpoint(
        self,
        L: int,
        gamma: float,
        G_matrix: np.ndarray,
        step: int,
        task_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save task checkpoint for correlation matrix G.
        
        Args:
            L: System size.
            gamma: Loss rate.
            G_matrix: Correlation matrix (L x L complex128).
            step: Evolution step number.
            task_metadata: Optional metadata (timing, numerical info).
        """
        task_key = (int(L), float(gamma))
        
        # Create task-specific directory
        task_dir = self.checkpoint_dir / f"L{L}_g{gamma:.4f}"
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save G matrix to memmap file
        G_path = task_dir / "G_matrix.npy"
        np.save(G_path, G_matrix)
        
        # Save metadata with step and timing
        metadata = {
            "L": L,
            "gamma": gamma,
            "step": step,
            "G_shape": list(G_matrix.shape),
            "G_dtype": str(G_matrix.dtype),
            "checkpoint_interval": self.checkpoint_interval,
        }
        if task_metadata:
            metadata.update(task_metadata)
        
        # Update in-memory index
        self.completed_tasks[task_key] = metadata
        
        # Atomically flush index to disk
        self._save_index()
        
        logger.debug(f"Saved checkpoint for L={L}, gamma={gamma} at step {step}")
    
    def load_task_checkpoint(
        self,
        L: int,
        gamma: float,
    ) -> tuple[np.ndarray, int, dict[str, Any]] | None:
        """
        Load task checkpoint for correlation matrix.
        
        Args:
            L: System size.
            gamma: Loss rate.
        
        Returns:
            Tuple of (G_matrix, step, metadata) if checkpoint exists, else None.
        """
        task_key = (int(L), float(gamma))
        
        if task_key not in self.completed_tasks:
            return None
        
        task_dir = self.checkpoint_dir / f"L{L}_g{gamma:.4f}"
        G_path = task_dir / "G_matrix.npy"
        
        if not G_path.exists():
            logger.warning(f"Checkpoint metadata exists but G_matrix missing for L={L}, gamma={gamma}")
            return None
        
        try:
            G_matrix = np.load(G_path)
            metadata = self.completed_tasks[task_key]
            step = metadata.get("step", 0)
            
            logger.debug(f"Loaded checkpoint for L={L}, gamma={gamma} from step {step}")
            return G_matrix, step, metadata
        except Exception as e:
            logger.error(f"Failed to load checkpoint for L={L}, gamma={gamma}: {e}")
            return None
    
    def get_completed_pairs(self) -> set[tuple[int, float]]:
        """Return set of completed (L, gamma) pairs."""
        return set(self.completed_tasks.keys())
    
    def cleanup_stale_checkpoints(self, completed_tasks: set[tuple[int, float]]) -> None:
        """
        Clean up checkpoint directories for tasks no longer in sweep.
        
        Args:
            completed_tasks: Set of (L, gamma) pairs that should be kept.
        """
        for (L, gamma) in list(self.completed_tasks.keys()):
            if (L, gamma) not in completed_tasks:
                task_dir = self.checkpoint_dir / f"L{L}_g{gamma:.4f}"
                if task_dir.exists():
                    import shutil
                    shutil.rmtree(task_dir)
                    del self.completed_tasks[(L, gamma)]
        
        self._save_index()
