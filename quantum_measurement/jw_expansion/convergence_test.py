"""Test convergence and compare with analytical predictions."""

import numpy as np
import matplotlib.pyplot as plt
from n_infty_steady_state import validate_steady_state_with_simulation
from n_infty import sum_apbc


def test_convergence():
    """Test that simulation converges to analytical steady state."""
    L = 5
    J = 1.0
    gamma = 0.4
    g = gamma / (4.0 * J)
    
    print("=" * 70)
    print("Convergence Test: Simulation vs Analytical Steady State")
    print("=" * 70)
    print(f"Parameters: L={L}, J={J}, gamma={gamma}, g={g}")
    print()
    
    # Test with different numbers of steps
    step_counts = [5000, 10000, 20000, 50000]
    
    results_list = []
    
    for N_steps in step_counts:
        print(f"Running with N_steps = {N_steps}...")
        results = validate_steady_state_with_simulation(
            L=L, J=J, gamma=gamma, boundary='open',
            dt=0.0001, N_steps=N_steps
        )
        results_list.append(results)
        
        print(f"  T_total = {results['T_total']:.1f}")
        print(f"  n_simulated = {results['n_simulated']:.6f}")
        print(f"  n_analytical = {results['n_analytical']:.6f}")
        print(f"  Difference = {results['occupation_diff']:.2e}")
        print(f"  Convergence check = {results['convergence_check']:.2e}")
        print()
    
    # Compare convergence
    print("-" * 70)
    print("Convergence Summary:")
    print("-" * 70)
    print(f"{'N_steps':<10} {'T_total':<10} {'n_sim':<12} {'|n_sim-n_ana|':<15} {'Converged?':<12}")
    print("-" * 70)
    
    for N_steps, res in zip(step_counts, results_list):
        converged = "✓" if res['convergence_check'] < 1e-4 else "⚠"
        print(f"{N_steps:<10} {res['T_total']:<10.1f} {res['n_simulated']:<12.6f} "
              f"{res['occupation_diff']:<15.2e} {converged:<12}")
    
    print()
    print(f"Analytical prediction: n_∞ = {results_list[0]['n_analytical']:.6f}")
    
    # Show that simulation approaches analytical value
    diffs = [res['occupation_diff'] for res in results_list]
    if diffs[-1] < diffs[0]:
        print("✓ Difference decreases with more steps (good!)")
    else:
        print("⚠ Difference not decreasing - may need different parameters")
    
    return results_list


def plot_occupation_trajectory():
    """Plot how occupation evolves to steady state."""
    L = 5
    J = 1.0
    gamma = 0.4
    g = gamma / (4.0 * J)
    
    print("\n" + "=" * 70)
    print("Plotting Occupation Trajectory")
    print("=" * 70)
    
    results = validate_steady_state_with_simulation(
        L=L, J=J, gamma=gamma, boundary='open',
        dt=0.0001, N_steps=50000
    )
    
    n_traj = results['occupation_trajectory']
    T_total = results['T_total']
    n_analytical = results['n_analytical']
    
    # Time array
    time = np.linspace(0, T_total, len(n_traj))
    
    # Mean occupation over all sites
    n_mean = n_traj.mean(axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, n_mean, 'b-', linewidth=2, label='Simulation')
    plt.axhline(y=n_analytical, color='r', linestyle='--', linewidth=2,
                label=f'Analytical n_∞ = {n_analytical:.4f}')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Mean Occupation ⟨n⟩', fontsize=12)
    plt.title(f'Approach to Steady State (L={L}, γ={gamma}, g={g:.2f})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    filename = 'occupation_trajectory.png'
    plt.savefig(filename, dpi=150)
    print(f"✓ Saved plot to {filename}")
    
    # Also plot individual sites
    plt.figure(figsize=(10, 6))
    for i in range(L):
        plt.plot(time, n_traj[:, i], alpha=0.7, label=f'Site {i}')
    plt.axhline(y=n_analytical, color='r', linestyle='--', linewidth=2,
                label=f'Analytical n_∞')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Occupation n_i', fontsize=12)
    plt.title(f'Site Occupations (L={L}, γ={gamma}, g={g:.2f})', fontsize=14)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = 'site_occupations.png'
    plt.savefig(filename, dpi=150)
    print(f"✓ Saved plot to {filename}")
    
    plt.show()


if __name__ == "__main__":
    # Run convergence test
    results = test_convergence()
    
    # Optionally plot trajectories (comment out if no display)
    try:
        plot_occupation_trajectory()
    except Exception as e:
        print(f"\nSkipping plots (no display or matplotlib issue): {e}")
