#!/usr/bin/env python
"""
Test script for parallel SSE simulation.

This script validates that:
1. The pure function version produces the same statistical results as the original
2. Parallel execution produces correct results
3. Performance improvement is achieved with parallel execution
"""

import numpy as np
import time
from sse_wavefunction_simulation import SSEWavefunctionSimulator, _simulate_trajectory_pure


def test_pure_function_vs_method():
    """Test that pure function matches the method version."""
    print("Test 1: Pure function vs. method")
    print("-" * 50)
    
    sim = SSEWavefunctionSimulator(epsilon=0.1, N_steps=100, J=0.0)
    
    # Run multiple times to check statistical consistency
    n_samples = 100
    Q_method = []
    Q_pure = []
    
    for i in range(n_samples):
        # Method version (uses pure function internally now)
        Q1, _, _ = sim.simulate_trajectory()
        Q_method.append(Q1)
        
        # Direct pure function call
        config = sim._get_config_dict()
        Q2, _, _ = _simulate_trajectory_pure(config, sim.psi_initial, seed=None)
        Q_pure.append(Q2)
    
    mean_method = np.mean(Q_method)
    mean_pure = np.mean(Q_pure)
    std_method = np.std(Q_method)
    std_pure = np.std(Q_pure)
    
    print(f"Method version: mean={mean_method:.3f}, std={std_method:.3f}")
    print(f"Pure function:  mean={mean_pure:.3f}, std={std_pure:.3f}")
    
    # They should have similar statistics (not identical since random)
    assert abs(mean_method - mean_pure) < 0.5, "Means differ too much"
    assert abs(std_method - std_pure) < 0.5, "Stds differ too much"
    print("✓ Pure function produces consistent results\n")


def test_parallel_correctness():
    """Test that parallel execution produces correct statistics."""
    print("Test 2: Parallel correctness")
    print("-" * 50)
    
    sim = SSEWavefunctionSimulator(epsilon=0.1, N_steps=100, J=0.0)
    n_traj = 500
    
    # Sequential
    Q_seq, _, _ = sim.simulate_ensemble(n_traj, progress=False)
    
    # Parallel
    Q_par, _, _ = sim.simulate_ensemble_parallel(n_traj, max_workers=2, progress=False)
    
    mean_seq = np.mean(Q_seq)
    mean_par = np.mean(Q_par)
    var_seq = np.var(Q_seq)
    var_par = np.var(Q_par)
    
    theoretical_mean, theoretical_var = sim.theoretical_mean_variance()
    
    print(f"Sequential: mean={mean_seq:.3f}, var={var_seq:.3f}")
    print(f"Parallel:   mean={mean_par:.3f}, var={var_par:.3f}")
    print(f"Theory:     mean={theoretical_mean:.3f}, var={theoretical_var:.3f}")
    
    # Both should be close to theory
    assert abs(mean_seq - theoretical_mean) < 0.5, "Sequential mean off"
    assert abs(mean_par - theoretical_mean) < 0.5, "Parallel mean off"
    assert abs(var_seq - theoretical_var) < 0.5, "Sequential var off"
    assert abs(var_par - theoretical_var) < 0.5, "Parallel var off"
    print("✓ Parallel execution produces correct statistics\n")


def test_performance():
    """Test that parallel execution provides speedup."""
    print("Test 3: Performance comparison")
    print("-" * 50)
    
    sim = SSEWavefunctionSimulator(epsilon=0.1, N_steps=100, J=0.0)
    n_traj = 200
    
    # Sequential
    start = time.time()
    Q_seq, _, _ = sim.simulate_ensemble(n_traj, progress=False)
    seq_time = time.time() - start
    
    # Parallel with 2 workers
    start = time.time()
    Q_par, _, _ = sim.simulate_ensemble_parallel(n_traj, max_workers=2, progress=False)
    par_time = time.time() - start
    
    speedup = seq_time / par_time
    
    print(f"Sequential: {seq_time:.3f}s ({n_traj/seq_time:.1f} traj/s)")
    print(f"Parallel:   {par_time:.3f}s ({n_traj/par_time:.1f} traj/s)")
    print(f"Speedup:    {speedup:.2f}x")
    
    # Expect some speedup (even if not perfect 2x due to overhead)
    assert speedup > 1.2, f"Insufficient speedup: {speedup:.2f}x"
    print("✓ Parallel execution provides speedup\n")


def test_reproducibility():
    """Test that using seeds produces reproducible results."""
    print("Test 4: Reproducibility with seeds")
    print("-" * 50)
    
    sim = SSEWavefunctionSimulator(epsilon=0.1, N_steps=100, J=0.0)
    config = sim._get_config_dict()
    psi = sim.psi_initial
    
    # Run with same seed twice
    Q1, z1, m1 = _simulate_trajectory_pure(config, psi, seed=42)
    Q2, z2, m2 = _simulate_trajectory_pure(config, psi, seed=42)
    
    # Should be identical
    assert np.allclose(Q1, Q2), "Q values differ"
    assert np.allclose(z1, z2), "Trajectories differ"
    assert np.array_equal(m1, m2), "Measurements differ"
    
    print(f"Trajectory 1: Q={Q1:.6f}")
    print(f"Trajectory 2: Q={Q2:.6f}")
    print("✓ Seeded results are reproducible\n")


if __name__ == "__main__":
    print("=" * 50)
    print("SSE Parallel Simulation Tests")
    print("=" * 50)
    print()
    
    test_pure_function_vs_method()
    test_parallel_correctness()
    test_performance()
    test_reproducibility()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
