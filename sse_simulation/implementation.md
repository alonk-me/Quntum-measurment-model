# SSE Wavefunction Simulation Implementation

## Overview

This document explains the implementation of the **wavefunction-based Stochastic Schrödinger Equation (SSE) simulator** with discrete measurements, following the theoretical framework from `EP_Production.md`. The simulator evolves quantum states using the discrete SSE and calculates entropy production using the exact discrete formula.

## Theoretical Foundation

### Discrete SSE Evolution

The simulator implements the discrete version of the SSE where the quantum state evolves as:

```
|ψ_{i+1}⟩ = |ψ_i⟩ + |δψ⟩
```

where the state increment is:

```
|δψ⟩ = ξ_i ε (σ_z - ⟨σ_z⟩)/2 |ψ⟩ - (ε²/2) (σ_z - ⟨σ_z⟩)²/4 |ψ⟩
```

This includes:
- **First-order term**: `ξ_i ε (σ_z - ⟨σ_z⟩)/2 |ψ⟩` (measurement backaction)
- **Second-order term**: `-(ε²/2) (σ_z - ⟨σ_z⟩)²/4 |ψ⟩` (measurement-induced decoherence)

### Measurement Process

At each time step:
1. **Calculate expectation value**: `z = ⟨ψ|σ_z|ψ⟩`
2. **Determine measurement probabilities**: 
   - `P(+1) = (1 + εz)/2`
   - `P(-1) = (1 - εz)/2`
3. **Sample measurement outcome**: `ξ_i ∈ {+1, -1}` with probabilities above
4. **Update wavefunction** using the SSE increment
5. **Normalize** to maintain unit norm

### Discrete Entropy Production

The entropy production uses the **exact discrete Stratonovich formula** from EP_Production.md:

```
Q = 2ε Σ_{i=1}^N r_i (z_{i-1} + z_i)/2
```

where:
- `r_i = ξ_i` is the measurement result at step i
- `z_{i-1}` is the z-expectation before the measurement  
- `z_i` is the z-expectation after the measurement
- The `(z_{i-1} + z_i)/2` term implements Stratonovich discretization

### Optional Hamiltonian Evolution

When `J ≠ 0`, the simulator includes coherent evolution under Hamiltonian `H = J σ_x`:

```
|ψ⟩ → exp(-i J σ_x dt) |ψ⟩
```

This is applied before each measurement step using the analytical matrix exponential:
```
exp(-i θ σ_x) = cos(θ)I - i sin(θ)σ_x
```

## Implementation Details

### Core Classes

#### `SSEWavefunctionSimulator`

The main simulator class with configurable parameters:

**Parameters:**
- `epsilon` (float): Measurement strength ε
- `N_steps` (int): Number of discrete measurement steps  
- `J` (float): Hamiltonian coupling parameter (default: 0)
- `initial_state` (str): Initial quantum state specification
- `theta`, `phi` (float): Custom initial state angles
- `rng` (Generator): NumPy random number generator

**Key Methods:**

##### `_prepare_initial_state()`
Sets up the initial quantum state based on the `initial_state` parameter:

- `'bloch_equator'`: `|+x⟩ = (|0⟩ + |1⟩)/√2` (default)
- `'up'`: `|0⟩`
- `'down'`: `|1⟩` 
- `'plus_y'`: `|+y⟩ = (|0⟩ + i|1⟩)/√2`
- `'minus_y'`: `|-y⟩ = (|0⟩ - i|1⟩)/√2`
- `'custom'`: General Bloch sphere state using θ, φ angles

##### `_measurement_update(psi)`
Applies a single discrete measurement step:

1. **Calculate z expectation**: `z = Re(⟨ψ|σ_z|ψ⟩)`
2. **Compute probabilities**: `P(±1) = (1 ± εz)/2`  
3. **Sample outcome**: `ξ ∈ {+1, -1}`
4. **Apply SSE update**: 
   ```python
   sigma_z_minus_z = σ_z - z * I
   first_order = ξ * ε * 0.5 * (σ_z - z) @ ψ
   second_order = -0.5 * ε² * 0.25 * (σ_z - z)² @ ψ
   ψ_new = ψ + first_order + second_order
   ```
5. **Normalize**: `ψ_new = ψ_new / ||ψ_new||`

##### `simulate_trajectory()`
Simulates a single stochastic trajectory:

```python
for i in range(N_steps):
    # Optional Hamiltonian evolution
    if J != 0:
        ψ = apply_hamiltonian_evolution(ψ, dt)
    
    # Measurement step  
    ψ, ξ, z_before = measurement_update(ψ)
    z_after = expectation_value_z(ψ)
    
    # Accumulate entropy production
    Q += 2ε * ξ * (z_before + z_after) / 2
```

Returns: `(Q, z_trajectory, measurement_results)`

##### `simulate_ensemble(n_trajectories)`
Runs multiple independent trajectories for statistical analysis.

Returns: `(Q_values, z_trajectories, measurement_results)`

### Theoretical Predictions

#### `theoretical_mean_variance()`
Calculates expected mean and variance for comparison with simulations.

For initial state on Bloch equator (`z_initial = 0`):
- **Mean**: `⟨Q⟩ = (3/2) * N * ε²`
- **Variance**: `Var(Q) = 2 * N * ε²`

These formulas connect to the Dressel et al. framework via `T/τ = N * ε²`.

## Usage Examples

### Basic Single Trajectory
```python
from sse_wavefunction_simulation import SSEWavefunctionSimulator

# Create simulator
sim = SSEWavefunctionSimulator(
    epsilon=0.1,
    N_steps=100,
    initial_state='bloch_equator'
)

# Simulate single trajectory
Q, z_trajectory, measurements = sim.simulate_trajectory()
```

### Ensemble Statistics
```python
# Simulate ensemble
n_trajectories = 1000
Q_values, z_trajectories, _ = sim.simulate_ensemble(n_trajectories)

# Compare with theory
theoretical_mean, theoretical_var = sim.theoretical_mean_variance()
print(f"Observed: ⟨Q⟩ = {np.mean(Q_values):.3f}")
print(f"Theory:   ⟨Q⟩ = {theoretical_mean:.3f}")
```

### Different Initial States
```python
# Test different initial conditions
states = ['up', 'down', 'bloch_equator', 'plus_y']

for state in states:
    sim = SSEWavefunctionSimulator(epsilon=0.1, initial_state=state)
    Q, z_traj, _ = sim.simulate_trajectory()
    print(f"{state}: Q = {Q:.3f}, z_initial = {z_traj[0]:.3f}")
```

### Hamiltonian Coupling
```python
# Include coherent evolution
sim = SSEWavefunctionSimulator(
    epsilon=0.1,
    N_steps=100,
    J=1.0,  # Hamiltonian coupling
    initial_state='bloch_equator'
)

Q_values, z_trajectories, _ = sim.simulate_ensemble(500)
```

## Key Implementation Features

### 1. **Exact Discrete Formula**
- Uses the precise Stratonovich discretization from EP_Production.md
- No continuous approximations or Itô vs Stratonovich ambiguities
- Direct implementation of `Q = 2ε Σ r_i (z_{i-1} + z_i)/2`

### 2. **Wavefunction Evolution**
- Evolves full quantum state `|ψ⟩ ∈ ℂ²` rather than just Bloch vector
- Maintains quantum coherences and superposition
- Proper normalization after each measurement

### 3. **Configurable Initial States**
- Supports common quantum states (computational basis, Bloch equator, etc.)
- Custom states via spherical coordinates (θ, φ)
- Easy extension to arbitrary initial conditions

### 4. **Optional Hamiltonian Dynamics**
- Clean separation of measurement and coherent evolution
- Analytical matrix exponential for σ_x Hamiltonian
- Time-ordered evolution: Hamiltonian → Measurement

### 5. **Robust Numerics**
- Careful normalization to prevent numerical drift
- Proper complex arithmetic for quantum amplitudes
- Vectorized operations for computational efficiency

## Validation and Testing

The implementation includes comprehensive validation:

1. **Statistical Agreement**: Ensemble averages match theoretical predictions
2. **Initial State Dependence**: Different initial states produce expected Q distributions  
3. **Hamiltonian Effects**: J ≠ 0 modifies dynamics as expected
4. **Discretization Convergence**: Results converge with increasing N_steps
5. **Conservation Laws**: Probability conservation maintained

## Connection to Theory

This implementation directly realizes the discrete measurement protocol from EP_Production.md:

- **Measurement strength ε**: Controls the information gain per measurement
- **Discrete time steps**: No continuous limit approximations
- **Stratonovich midpoint rule**: Exact implementation of `(z_{i-1} + z_i)/2`
- **Entropy production**: Measures arrow-of-time via information-theoretic quantity

The simulator provides a computational laboratory for studying quantum measurement thermodynamics in the discrete regime where analytical solutions are challenging.

## Performance Notes

- **Trajectory simulation**: O(N_steps) per trajectory
- **Ensemble simulation**: Embarrassingly parallel over trajectories  
- **Memory usage**: O(N_steps) per trajectory for storing results
- **Numerical precision**: Complex128 arithmetic maintains quantum coherences

For large ensembles, consider using multiprocessing or implementing the trajectory loop in compiled languages (Numba, Cython) for speed improvements.