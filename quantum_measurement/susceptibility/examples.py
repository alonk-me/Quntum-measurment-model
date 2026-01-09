"""
Susceptibility Validation Examples
===================================

Simple examples demonstrating susceptibility calculations and validation
against analytical results.
"""

import numpy as np
from quantum_measurement.jw_expansion import (
    TwoQubitCorrelationSimulator,
    LQubitCorrelationSimulator
)
from quantum_measurement.susceptibility import (
    compute_static_susceptibility,
    compute_connected_correlation,
    compute_susceptibility_matrix
)
from quantum_measurement.susceptibility.analytical import (
    analytical_susceptibility_from_decay_rate,
    validate_numerical_susceptibility
)


def example_two_qubit_susceptibility():
    """Example: Compute susceptibility for two-qubit system."""
    print("="*60)
    print("Example 1: Two-Qubit Susceptibility")
    print("="*60)
    
    # Create simulator
    sim = TwoQubitCorrelationSimulator(
        J=1.0,
        epsilon=0.1,
        N_steps=1000,
        T=10.0
    )
    
    print(f"\nParameters:")
    print(f"  J = {sim.J}")
    print(f"  ε = {sim.epsilon}")
    print(f"  N_steps = {sim.N_steps}")
    print(f"  T = {sim.T}")
    
    # Run trajectory
    Q, z_trajectory, _ = sim.simulate_trajectory()
    print(f"\nEntropy production: Q = {Q:.3f}")
    
    # Compute susceptibilities
    chi_00 = sim.compute_susceptibility(z_trajectory, site_i=0, site_j=0)
    chi_11 = sim.compute_susceptibility(z_trajectory, site_i=1, site_j=1)
    chi_01 = sim.compute_susceptibility(z_trajectory, site_i=0, site_j=1)
    
    print(f"\nSusceptibilities:")
    print(f"  χ₀₀ = {chi_00:.4f}")
    print(f"  χ₁₁ = {chi_11:.4f}")
    print(f"  χ₀₁ = {chi_01:.4f}")
    
    return sim, z_trajectory


def example_ensemble_averaging():
    """Example: Ensemble averaging of susceptibility."""
    print("\n" + "="*60)
    print("Example 2: Ensemble Averaging")
    print("="*60)
    
    sim = TwoQubitCorrelationSimulator(
        J=1.0,
        epsilon=0.1,
        N_steps=500,
        T=5.0
    )
    
    n_traj = 20
    print(f"\nSimulating {n_traj} trajectories...")
    
    Q_values, z_trajectories, _ = sim.simulate_ensemble(n_traj)
    
    # Compute susceptibility for each trajectory
    chi_values = []
    for i in range(n_traj):
        chi = sim.compute_susceptibility(z_trajectories[i], site_i=0)
        chi_values.append(chi)
    
    chi_values = np.array(chi_values)
    mean_chi = np.mean(chi_values)
    std_chi = np.std(chi_values)
    
    print(f"\nResults:")
    print(f"  ⟨Q⟩ = {np.mean(Q_values):.3f} ± {np.std(Q_values):.3f}")
    print(f"  ⟨χ⟩ = {mean_chi:.4f} ± {std_chi:.4f}")
    
    return mean_chi, std_chi


def example_l_qubit_chain():
    """Example: Susceptibility matrix for L-qubit chain."""
    print("\n" + "="*60)
    print("Example 3: L-Qubit Chain Susceptibility Matrix")
    print("="*60)
    
    L = 4
    sim = LQubitCorrelationSimulator(
        L=L,
        J=1.0,
        epsilon=0.1,
        N_steps=500,
        T=5.0,
        closed_boundary=False
    )
    
    print(f"\nL = {L} qubit chain (open boundaries)")
    print(f"J = {sim.J}, ε = {sim.epsilon}")
    
    # Run trajectory
    Q, z_trajectory, _ = sim.simulate_trajectory()
    print(f"\nEntropy production: Q = {Q:.3f}")
    
    # Compute susceptibility matrix
    chi_matrix = sim.compute_susceptibility_matrix(z_trajectory)
    
    print(f"\nSusceptibility matrix ({L}×{L}):")
    for i in range(L):
        row_str = "  "
        for j in range(L):
            row_str += f"{chi_matrix[i,j]:7.4f} "
        print(row_str)
    
    print(f"\nDiagonal elements (single-site susceptibilities):")
    for i in range(L):
        print(f"  χ_{i}{i} = {chi_matrix[i,i]:.4f}")
    
    return chi_matrix


def example_spatial_correlations():
    """Example: Spatial dependence of susceptibility."""
    print("\n" + "="*60)
    print("Example 4: Spatial Correlations")
    print("="*60)
    
    L = 6
    sim = LQubitCorrelationSimulator(
        L=L,
        J=1.0,
        epsilon=0.1,
        N_steps=500,
        T=5.0,
        closed_boundary=False
    )
    
    print(f"\nL = {L} qubit chain")
    
    # Run trajectory
    Q, z_trajectory, _ = sim.simulate_trajectory()
    
    # Compute χ(r) from site 0
    print(f"\nSusceptibility vs distance from site 0:")
    for r in range(L):
        chi_r = sim.compute_susceptibility(z_trajectory, site_i=0, site_j=r)
        print(f"  r = {r}: χ(r) = {chi_r:.4f}")


def example_measurement_strength_dependence():
    """Example: Dependence on measurement strength."""
    print("\n" + "="*60)
    print("Example 5: Measurement Strength Dependence")
    print("="*60)
    
    epsilon_values = [0.05, 0.1, 0.15, 0.2]
    
    print(f"\nComputing χ for different ε values:")
    print(f"{'ε':>6s}  {'χ (numerical)':>15s}  {'χ (analytical)':>15s}  {'Ratio':>8s}")
    print("-" * 60)
    
    for eps in epsilon_values:
        sim = TwoQubitCorrelationSimulator(
            J=1.0,
            epsilon=eps,
            N_steps=500,
            T=5.0
        )
        
        Q, z_traj, _ = sim.simulate_trajectory()
        chi_num = sim.compute_susceptibility(z_traj, site_i=0)
        
        # Analytical approximation: χ ∝ 1/γ where γ ∝ ε²
        chi_ana = analytical_susceptibility_from_decay_rate(eps**2)
        
        ratio = chi_num / chi_ana if chi_ana != 0 else 0
        
        print(f"{eps:6.2f}  {chi_num:15.4f}  {chi_ana:15.4f}  {ratio:8.3f}")


def example_validation():
    """Example: Validation against analytical results."""
    print("\n" + "="*60)
    print("Example 6: Validation Against Analytical Results")
    print("="*60)
    
    # For a simple decay model, we expect χ = 1/γ
    gamma = 0.1  # Decay rate
    analytical_chi = analytical_susceptibility_from_decay_rate(gamma)
    
    print(f"\nAnalytical susceptibility (decay rate γ={gamma}): χ = {analytical_chi:.4f}")
    
    # Generate synthetic data with exponential decay
    times = np.linspace(0, 10, 1000)
    correlation = np.exp(-gamma * times)
    
    # Add some noise
    np.random.seed(42)
    correlation += 0.01 * np.random.randn(len(times))
    
    # Integrate to get numerical susceptibility
    numerical_chi = np.trapezoid(correlation, times)
    
    print(f"Numerical susceptibility (from integration): χ = {numerical_chi:.4f}")
    
    # Validate
    is_valid = validate_numerical_susceptibility(
        numerical_chi, 
        analytical_chi, 
        tolerance=0.05  # 5% tolerance
    )
    
    print(f"\nValidation: {'PASS ✓' if is_valid else 'FAIL ✗'}")
    print(f"Relative error: {abs(numerical_chi - analytical_chi) / analytical_chi * 100:.2f}%")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("SUSCEPTIBILITY MODULE - VALIDATION EXAMPLES")
    print("="*60)
    
    # Run examples
    example_two_qubit_susceptibility()
    example_ensemble_averaging()
    example_l_qubit_chain()
    example_spatial_correlations()
    example_measurement_strength_dependence()
    example_validation()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
