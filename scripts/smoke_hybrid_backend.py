#!/usr/bin/env python
"""
Smoke test for hybrid CPU/GPU executor with minimal grid.

Validates:
1. Queue determinism (consistent task ordering)
2. GPU stream cycling (if GPU available)
3. Memmap checkpoint creation and integrity
4. Resume without recomputation
5. Numerical sanity (finite z2_plus_one, metadata recorded)
6. CSV compatibility with existing analysis tools
"""

import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from quantum_measurement.parallel import HybridExecutor, HybridExecutorConfig, DeterministicWorkQueue
from quantum_measurement.jw_expansion import LQubitCorrelationSimulator
from quantum_measurement.backends import get_backend


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_minimal_test_grid():
    """Create minimal test grid: L in {3, 9}, gamma in {0.4, 4.0}."""
    L_values = [3, 9]
    gamma_grid = [0.4, 4.0]
    return L_values, gamma_grid


def create_simulator_factory(backend_device: str = "hybrid") -> Callable:
    """Create a simulator factory that runs z2 ensemble."""
    
    J_GLOBAL = 1.0
    T_MULTIPLIER = 100.0
    T_MIN = 100.0
    DT_RATIO = 1e-2
    DT_MAX = 1e-3
    
    def factory(
        L: int,
        gamma: float,
        device: str,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Run z2 ensemble simulation and return results."""
        
        # Translate gamma to time and epsilon parameters
        g = gamma / (4 * J_GLOBAL)
        tau = 1.0 / gamma
        T = max(tau * T_MULTIPLIER, T_MIN)
        dt = min(tau * DT_RATIO, DT_MAX)
        N_steps = int(round(T / dt))
        epsilon = float(np.sqrt(gamma * dt))
        
        # Create seeded RNG for this point
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        point_rng = np.random.default_rng(seed)
        
        # Create simulator
        backend = get_backend(device if device != "hybrid" else "cpu")
        
        sim = LQubitCorrelationSimulator(
            L=L,
            J=J_GLOBAL,
            epsilon=epsilon,
            N_steps=N_steps,
            T=T,
            closed_boundary=True,
            device=device if device != "hybrid" else "cpu",
            rng=point_rng,
        )
        
        # For GPU backend, seed the backend RNG if available
        if device == "gpu" and hasattr(sim, "backend") and hasattr(sim.backend, "seed"):
            sim.backend.seed(seed)
        ) -> Dict[str, Any]:
            """Run z2 ensemble simulation and return results (smoke test = minimal simulation)."""
        
            # For smoke test, use minimal evolution time to speed up testing
            # In production, this would be based on gamma
            J_GLOBAL = 1.0
            T = 0.1  # Very short evolution time
            dt = 0.01
            N_steps = int(round(T / dt))  # Just 10 steps
            epsilon = 0.1  # Moderate measurement strength
        
            # Create seeded RNG for this point
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            point_rng = np.random.default_rng(seed)
        
            # Create simulator
            sim = LQubitCorrelationSimulator(
                L=L,
                J=J_GLOBAL,
                epsilon=epsilon,
                N_steps=N_steps,
                T=T,
                closed_boundary=True,
                device=device if device != "hybrid" else "cpu",
                rng=point_rng,
            )
        
            # Quick trajectory ensemble (smoke test uses minimal trajectories)
            n_traj = 1  # Minimal trajectories for speed
            batch_size = 1
        
            z2_result = sim.simulate_z2_mean_ensemble(
                n_trajectories=n_traj,
                batch_size=batch_size,
                return_std_err=True,
            )
        
            z2_mean, z2_std, z2_stderr = z2_result
        
            return {
                "L": L,
                "gamma": gamma,
                "z2_plus_one": float(z2_mean),
                "z2_std": float(z2_std) if z2_std is not None else 0.0,
                "z2_stderr": float(z2_stderr) if z2_stderr is not None else 0.0,
                "n_trajectories": n_traj,
                "batch_size": batch_size,
            }
    items1 = []
    items2 = []
    
    for item in queue1:
        items1.append((item.L, item.gamma, item.seed, item.route_hint.value))
    
    for item in queue2:
        items2.append((item.L, item.gamma, item.seed, item.route_hint.value))
    
    assert items1 == items2, f"Queue order mismatch: {items1} != {items2}"
    assert len(items1) == 4, f"Expected 4 tasks, got {len(items1)}"
    assert len(set((L, g) for L, g, _, _ in items1)) == 4, "Duplicate (L, gamma) pairs"
    
    logger.info(f"✓ Queue determinism: {len(items1)} tasks in consistent order")
    return True


def test_memmap_checkpoint():
    """Test 2: Checkpoint writer creates, saves, and loads state."""
    logger.info("=== Test 2: Memmap Checkpoint ===")
    
    from quantum_measurement.parallel import MemmapCheckpointWriter
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = MemmapCheckpointWriter(tmpdir, checkpoint_interval=50, base_seed=42)
        
        # Save a task checkpoint
        G_test = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        ckpt.save_task_checkpoint(
            L=3,
            gamma=0.4,
            G_matrix=G_test,
            step=100,
            task_metadata={"test_key": "test_value"},
        )
        
        # Verify it was saved
        assert (3, 0.4) in ckpt.completed_tasks, "Task not recorded in index"
        
        # Load it back
        result = ckpt.load_task_checkpoint(3, 0.4)
        assert result is not None, "Failed to load checkpoint"
        G_loaded, step, metadata = result
        assert step == 100, f"Step mismatch: {step} != 100"
        assert metadata.get("test_key") == "test_value", "Metadata not preserved"
        assert np.allclose(G_loaded, G_test), "G matrix not preserved"
        
        logger.info(f"✓ Memmap checkpoint: saved/loaded G matrix and metadata")
    
    return True


def test_hybrid_executor_run():
    """Test 3: HybridExecutor runs smoke grid and produces CSV."""
    logger.info("=== Test 3: Hybrid Executor Run ===")
    
    L_values, gamma_grid = create_minimal_test_grid()
    factory = create_simulator_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = Path(tmpdir) / "smoke_results.csv"
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        
        config = HybridExecutorConfig(
            n_cpu_workers=2,
            n_gpu_workers=0,  # Smoke test on CPU only
            checkpoint_every_steps=5,
            enable_memmap=True,
            checkpoint_dir=checkpoint_dir,
            verbose=True,
        )
        
        executor = HybridExecutor(config)
        results = executor.run_sweep(
            L_values=L_values,
            gamma_grid=gamma_grid,
            simulator_factory=factory,
            backend_device="cpu",
            output_csv=output_csv,
            resume=True,
            csv_header=["L", "gamma", "z2_plus_one", "z2_std", "z2_stderr", "n_trajectories", "batch_size", "route", "worker_type", "runtime_sec", "seed"],
        )
        
        # Check results
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        for result in results:
            assert "z2_plus_one" in result, "Missing z2_plus_one"
            assert np.isfinite(result["z2_plus_one"]), f"Non-finite z2_plus_one: {result['z2_plus_one']}"
            assert "n_trajectories" in result, "Missing n_trajectories"
            assert "batch_size" in result, "Missing batch_size"
        
        # Check CSV
        assert output_csv.exists(), "CSV file not created"
        
        with output_csv.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 4, f"CSV has {len(rows)} rows, expected 4"
        
        logger.info(f"✓ Hybrid executor: processed {len(results)} tasks, wrote CSV")
    
    return True


def test_resume_without_recompute():
    """Test 4: Resume skips completed tasks without recomputing."""
    logger.info("=== Test 4: Resume Without Recompute ===")
    
    L_values, gamma_grid = create_minimal_test_grid()
    factory = create_simulator_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = Path(tmpdir) / "resume_results.csv"
        checkpoint_dir = Path(tmpdir) / "checkpoints_resume"
        
        config = HybridExecutorConfig(
            n_cpu_workers=1,
            n_gpu_workers=0,
            checkpoint_every_steps=5,
            enable_memmap=True,
            checkpoint_dir=checkpoint_dir,
            verbose=True,
        )
        
        # Run first sweep
        executor = HybridExecutor(config)
        results1 = executor.run_sweep(
            L_values=L_values,
            gamma_grid=gamma_grid,
            simulator_factory=factory,
            backend_device="cpu",
            output_csv=output_csv,
            resume=False,  # Fresh start
            csv_header=["L", "gamma", "z2_plus_one", "z2_std", "z2_stderr", "n_trajectories", "batch_size", "route", "worker_type", "runtime_sec", "seed"],
        )
        
        # Count rows in first CSV
        with output_csv.open("r") as f:
            first_rows = len(list(csv.DictReader(f)))
        
        logger.info(f"  First sweep: {first_rows} rows written")
        
        # Run second sweep with resume=True (should skip all completed)
        executor2 = HybridExecutor(config)
        results2 = executor2.run_sweep(
            L_values=L_values,
            gamma_grid=gamma_grid,
            simulator_factory=factory,
            backend_device="cpu",
            output_csv=output_csv,
            resume=True,  # Resume
            csv_header=["L", "gamma", "z2_plus_one", "z2_std", "z2_stderr", "n_trajectories", "batch_size", "route", "worker_type", "runtime_sec", "seed"],
        )
        
        # Count rows after resume (should still be same)
        with output_csv.open("r") as f:
            second_rows = len(list(csv.DictReader(f)))
        
        logger.info(f"  Resume sweep: {second_rows} rows total")
        
        assert second_rows == first_rows, f"Duplicate rows: {second_rows} != {first_rows}"
        assert len(results2) == 0, f"Expected 0 new results on resume, got {len(results2)}"
        
        logger.info(f"✓ Resume integrity: no duplicate (L, gamma) pairs")
    
    return True


def test_csv_compatibility():
    """Test 5: CSV output compatible with existing analysis pipeline."""
    logger.info("=== Test 5: CSV Compatibility ===")
    
    L_values, gamma_grid = create_minimal_test_grid()
    factory = create_simulator_factory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = Path(tmpdir) / "compat_results.csv"
        
        config = HybridExecutorConfig(
            n_cpu_workers=1,
            n_gpu_workers=0,
            verbose=False,
        )
        
        executor = HybridExecutor(config)
        executor.run_sweep(
            L_values=L_values,
            gamma_grid=gamma_grid,
            simulator_factory=factory,
            backend_device="cpu",
            output_csv=output_csv,
            csv_header=["L", "gamma", "z2_plus_one", "z2_std", "n_trajectories"],
        )
        
        # Try loading with pandas-like workflow (test without importing pandas itself)
        with output_csv.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check essential columns
                assert "L" in row, "Missing L column"
                assert "gamma" in row, "Missing gamma column"
                assert "z2_plus_one" in row, "Missing z2_plus_one column"
                
                # Validate types
                L_val = int(row["L"])
                gamma_val = float(row["gamma"])
                z2_val = float(row["z2_plus_one"])
                
                assert L_val in [3, 9], f"Unexpected L: {L_val}"
                assert gamma_val in [0.4, 4.0], f"Unexpected gamma: {gamma_val}"
                assert np.isfinite(z2_val), f"Non-finite z2_plus_one: {z2_val}"
        
        logger.info(f"✓ CSV compatibility: schema loads and parses correctly")
    
    return True


def run_all_smoke_tests():
    """Run all smoke tests in sequence."""
    logger.info("\n" + "="*60)
    logger.info("HYBRID BACKEND SMOKE TEST")
    logger.info("="*60 + "\n")
    
    tests = [
        ("Queue Determinism", test_queue_determinism),
        ("Memmap Checkpoint", test_memmap_checkpoint),
        ("Hybrid Executor", test_hybrid_executor_run),
        ("Resume Integrity", test_resume_without_recompute),
        ("CSV Compatibility", test_csv_compatibility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, "PASS"))
            logger.info("")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED: {e}")
            results.append((test_name, "FAIL"))
            import traceback
            traceback.print_exc()
            logger.info("")
    
    # Summary
    logger.info("="*60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        logger.info(f"{symbol} {test_name}: {status}")
    
    n_passed = sum(1 for _, status in results if status == "PASS")
    n_total = len(results)
    
    logger.info(f"\n{n_passed}/{n_total} tests passed")
    
    return n_passed == n_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for hybrid executor")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s: %(message)s")
    
    success = run_all_smoke_tests()
    sys.exit(0 if success else 1)
