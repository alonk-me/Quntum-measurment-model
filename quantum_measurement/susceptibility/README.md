# Susceptibility Module

This module provides tools for calculating magnetic susceptibility in quantum measurement simulations using the correlation matrix formalism.

## Overview

Magnetic susceptibility (χ) measures the linear response of a quantum system to external perturbations. For quantum spin systems under continuous measurement, susceptibility quantifies how spin correlations respond to applied fields and is related to the integrated correlation function:

```
χ_ij = ∫₀^T dt [⟨σᵢᶻ(t)σⱼᶻ(0)⟩ - ⟨σᵢᶻ(t)⟩⟨σⱼᶻ(0)⟩]
```

## Modules

### `static_susceptibility.py`

Computes static (zero-frequency) susceptibility from time-series data.

**Key Functions:**
- `compute_connected_correlation(sigma_i, sigma_j)` - Computes connected correlation function
- `compute_static_susceptibility(times, sigma_i, sigma_j)` - Integrates correlation to get χ
- `compute_single_site_susceptibility(times, z_values)` - Single-site susceptibility
- `compute_susceptibility_matrix(times, z_trajectories)` - Full L×L susceptibility matrix

### `correlation_matrix_susceptibility.py`

Extracts susceptibility from free-fermion correlation matrices using the Jordan-Wigner mapping.

**Key Functions:**
- `extract_spin_correlations(G)` - Extract ⟨σᵢᶻ⟩ from correlation matrix G
- `susceptibility_from_correlation_matrix(times, G_trajectory, site_i, site_j)` - Compute χ from G(t)
- `susceptibility_matrix_from_G_trajectory(times, G_trajectory)` - Full susceptibility matrix

### `analytical.py`

Analytical solutions for validation.

**Key Functions:**
- `single_qubit_susceptibility(temperature, field)` - Analytical χ for thermal equilibrium
- `analytical_susceptibility_from_decay_rate(gamma)` - χ from exponential decay
- `validate_numerical_susceptibility(numerical_chi, analytical_chi)` - Validation helper

## Integration with Simulators

### Two-Qubit Simulator

```python
from quantum_measurement.jw_expansion import TwoQubitCorrelationSimulator

# Create simulator
sim = TwoQubitCorrelationSimulator(J=1.0, epsilon=0.1, N_steps=1000, T=10.0)

# Run trajectory
Q, z_trajectory, xi_trajectory = sim.simulate_trajectory()

# Compute susceptibility for site 0
chi_00 = sim.compute_susceptibility(z_trajectory, site_i=0)
print(f"χ₀₀ = {chi_00:.4f}")

# Compute cross-susceptibility between sites 0 and 1
chi_01 = sim.compute_susceptibility(z_trajectory, site_i=0, site_j=1)
print(f"χ₀₁ = {chi_01:.4f}")

# Ensemble average
Q_values, z_trajectories, _ = sim.simulate_ensemble(n_trajectories=100)
mean_chi, std_chi = sim.compute_susceptibility_ensemble(z_trajectories, site_i=0)
print(f"⟨χ⟩ = {mean_chi:.4f} ± {std_chi:.4f}")
```

### L-Qubit Simulator

```python
from quantum_measurement.jw_expansion import LQubitCorrelationSimulator
import numpy as np

# Create 4-site chain
sim = LQubitCorrelationSimulator(L=4, J=1.0, epsilon=0.1, N_steps=1000, T=10.0)

# Run trajectory
Q, z_trajectory, xi_trajectory = sim.simulate_trajectory()

# Single-site susceptibility
chi_0 = sim.compute_susceptibility(z_trajectory, site_i=0)

# Two-site susceptibility
chi_02 = sim.compute_susceptibility(z_trajectory, site_i=0, site_j=2)

# Full susceptibility matrix
chi_matrix = sim.compute_susceptibility_matrix(z_trajectory)
print(f"Susceptibility matrix:\n{chi_matrix}")
print(f"Diagonal elements: {np.diag(chi_matrix)}")
```

## Direct Usage of Susceptibility Functions

You can also use the susceptibility functions directly:

```python
from quantum_measurement.susceptibility import (
    compute_static_susceptibility,
    compute_connected_correlation,
    compute_susceptibility_matrix
)
import numpy as np

# Generate example data
times = np.linspace(0, 10, 1000)
z_trajectory = np.random.randn(1000)  # Example trajectory

# Compute susceptibility
chi = compute_static_susceptibility(times, z_trajectory)
print(f"χ = {chi:.4f}")

# For multiple sites
L = 4
z_trajectories = np.random.randn(L, 1000)
chi_matrix = compute_susceptibility_matrix(times, z_trajectories)
```

## Physical Interpretation

1. **Positive χ**: Indicates tendency for parallel spin alignment (ferromagnetic correlations)
2. **Negative χ**: Indicates tendency for anti-parallel alignment (antiferromagnetic correlations)
3. **Magnitude**: Larger |χ| means stronger correlations and slower decay of correlation function

## Connection to Measurement Physics

Under continuous weak measurement:
- Measurement strength ε controls decoherence rate
- Susceptibility relates to measurement-induced correlations
- χ ∝ 1/γ where γ is the effective dephasing rate
- For strong measurement: χ → 0 (rapid correlation decay)
- For weak measurement: χ larger (slower correlation decay)

## Validation

Compare numerical results with analytical predictions:

```python
from quantum_measurement.susceptibility.analytical import (
    analytical_susceptibility_from_decay_rate,
    validate_numerical_susceptibility
)

# For a system with known decay rate
gamma = 0.5
analytical_chi = analytical_susceptibility_from_decay_rate(gamma)

# Compare with numerical calculation
numerical_chi = compute_static_susceptibility(times, z_trajectory)
is_valid = validate_numerical_susceptibility(numerical_chi, analytical_chi, tolerance=0.05)
print(f"Validation: {'PASS' if is_valid else 'FAIL'}")
```

## Advanced Features

### Ensemble Averaging

For better statistics, compute susceptibility over multiple trajectories:

```python
# Simulate ensemble
n_traj = 100
Q_values, z_trajectories, _ = sim.simulate_ensemble(n_traj)

# Average susceptibility
chi_values = []
for i in range(n_traj):
    chi = sim.compute_susceptibility(z_trajectories[i], site_i=0)
    chi_values.append(chi)

mean_chi = np.mean(chi_values)
std_chi = np.std(chi_values)
print(f"⟨χ⟩ = {mean_chi:.4f} ± {std_chi:.4f}")
```

### Spatial Dependence

Study how susceptibility depends on spatial separation:

```python
L = 10
sim = LQubitCorrelationSimulator(L=L, J=1.0, epsilon=0.1, N_steps=1000, T=10.0)
Q, z_traj, _ = sim.simulate_trajectory()

# Compute χ(r) vs distance r
for r in range(L):
    chi_r = sim.compute_susceptibility(z_traj, site_i=0, site_j=r)
    print(f"χ(r={r}) = {chi_r:.4f}")
```

## Notes

- Susceptibility calculations require integration over time, so longer simulation times (larger T) give more accurate results
- Numerical integration uses the trapezoidal rule
- For oscillatory correlation functions, ensure T is large enough to capture multiple periods
- Connected correlations (with mean subtracted) are used to isolate genuine correlations

## References

- Landau & Lifshitz, "Statistical Physics" - Theory of susceptibility
- Quantum trajectory theory literature for measurement-induced effects
- Jordan-Wigner transformation for free-fermion representation
