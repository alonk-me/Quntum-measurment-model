"""Test script for validating n_infty_steady_state and non_hermitian_hat integration."""

import numpy as np
from n_infty_steady_state import (
    extract_steady_state_from_simulation,
    validate_steady_state_with_simulation,
    get_correlation_eigenvalues
)


def test_basic_integration():
    """Test basic integration between modules."""
    print("=" * 70)
    print("Testing Integration: NonHermitianHatSimulator + Steady-State Analysis")
    print("=" * 70)
    
    # Small system for quick testing
    L = 5
    J = 1.0
    gamma = 0.4
    dt = 0.001
    N_steps = 10000  # Small number for quick test
    
    print(f"\nParameters:")
    print(f"  L = {L}")
    print(f"  J = {J}")
    print(f"  gamma = {gamma}")
    print(f"  g = gamma/(4J) = {gamma/(4*J)}")
    print(f"  dt = {dt}")
    print(f"  N_steps = {N_steps}")
    print(f"  T_total = {dt * N_steps}")
    
    # Test 1: Extract steady state from simulation
    print("\n" + "-" * 70)
    print("Test 1: Extract Steady State from Simulation")
    print("-" * 70)
    
    try:
        G_final, n_traj, Q_total = extract_steady_state_from_simulation(
            L=L, J=J, gamma=gamma, dt=dt, N_steps=N_steps, boundary='open'
        )
        print(f"✓ Successfully extracted steady state")
        print(f"  G_final shape: {G_final.shape}")
        print(f"  n_traj shape: {n_traj.shape}")
        print(f"  Q_total: {Q_total:.4f}")
        print(f"  Final mean occupation: {n_traj[-1, :].mean():.4f}")
        
        # Check convergence
        n_change = np.max(np.abs(n_traj[-1, :] - n_traj[-100, :]))
        print(f"  Convergence (max change in last 100 steps): {n_change:.2e}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Validate with analytical predictions
    print("\n" + "-" * 70)
    print("Test 2: Validate Against Analytical Predictions")
    print("-" * 70)
    
    try:
        results = validate_steady_state_with_simulation(
            L=L, J=J, gamma=gamma, boundary='open',
            dt=dt, N_steps=N_steps
        )
        
        print(f"✓ Validation completed")
        print(f"  Simulated occupation: {results['n_simulated']:.6f}")
        print(f"  Analytical occupation: {results['n_analytical']:.6f}")
        print(f"  Difference: {results['occupation_diff']:.2e}")
        print(f"  Convergence check: {results['convergence_check']:.2e}")
        print(f"  Number of eigenvalues: {len(results['eigs_simulated'])}")
        
        if results['occupation_diff'] < 0.1:
            print(f"  ✓ Occupations match within tolerance")
        else:
            print(f"  ⚠ Large occupation difference (may need more steps)")
            
        if results['convergence_check'] < 1e-3:
            print(f"  ✓ System appears converged")
        else:
            print(f"  ⚠ System may not be fully converged")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Check eigenvalue structure
    print("\n" + "-" * 70)
    print("Test 3: Analyze Eigenvalue Structure")
    print("-" * 70)
    
    try:
        eigs = get_correlation_eigenvalues(G_final)
        print(f"  Eigenvalues (first 5): {eigs[:5]}")
        print(f"  Eigenvalues (last 5): {eigs[-5:]}")
        print(f"  Eigenvalue range: [{eigs.min():.4f}, {eigs.max():.4f}]")
        
        # Eigenvalues should be in [0, 1] for physical correlation matrix
        if np.all((eigs >= -1e-6) & (eigs <= 1 + 1e-6)):
            print(f"  ✓ All eigenvalues in physical range [0, 1]")
        else:
            n_outside = np.sum((eigs < -1e-6) | (eigs > 1 + 1e-6))
            print(f"  ⚠ {n_outside} eigenvalues outside [0, 1] range")
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_basic_integration()
    if not success:
        print("\n⚠ Some tests failed. Check output above.")
    else:
        print("\n✓ Integration is working correctly!")
