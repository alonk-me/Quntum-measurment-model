# Quantum Measurement Simulations

A comprehensive Monte Carlo simulation framework for studying entropy production and arrow of time in single-qubit weak measurements. This project implements both continuous and discrete measurement protocols for quantum trajectories.

## Overview

This project provides two complementary approaches to quantum measurement simulation:

1. **Stochastic Schrödinger Equation (SSE)** - Continuous measurement protocol using stochastic differential equations
2. **Kraus Operators** - Discrete measurement protocol using probabilistic quantum operations

Both implementations explore the **arrow of time** in quantum measurements by computing entropy production functionals and comparing with analytical predictions from the literature.

## Project Structure

```
EP-simulation/
├── sse_simulation/          # Continuous measurement (SSE)
│   ├── sse.py              # Main SSE simulator class
│   ├── demo.ipynb          # SSE demonstration notebook
│   └── run_simulation.py   # Command-line interface
├── krauss_operators/        # Discrete measurement (Kraus)
│   ├── krauss_operators_simulation.py  # Main simulation functions
│   ├── datatypes.py        # Data structures for states and results
│   └── krauss_operators_notebook.ipynb # Kraus demonstration notebook
├── utilities/              # Shared utilities and analysis tools
├── .venv/                  # Virtual environment
├── pyproject.toml          # Project configuration and dependencies
└── README.md              # This file
```

## Features

### Stochastic Schrödinger Equation (SSE) Simulator
- **Continuous monitoring** of single qubits under measurement
- Configurable measurement operators (σ_z, projectors, custom)
- **Discrete vs continuous protocols** - Recently updated to support discrete ±1 outcomes
- Entropy production calculation: Q = 2ε² Σ z_i⟨z⟩ + 2ε Σ ξ_i⟨z⟩
- Comparison with analytical Arrow of Time distributions

### Kraus Operators Simulator  
- **Discrete measurement steps** using Born rule probabilities
- **Coherent evolution** with configurable Hamiltonian (ω·σ_y rotation)
- **State-dependent probabilities**: P±1 = 0.5(1 ± εz_ψ)
- **Validated quantum states** with automatic normalization
- **Parameter relationships**: Enforces θ = T/τ, ω = 4π/τ constraints

### Common Features
- **Flexible initial states** - Support for arbitrary superposition states
- **Ensemble simulations** - Generate large datasets for statistical analysis
- **Visualization tools** - Built-in plotting and comparison with theory
- **Parameter validation** - Ensures physical consistency
- **Reproducible results** - Seed control for random number generation

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment support

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd EP-simulation
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   
   # Optional: Install progress bar support
   pip install -e .[progress]
   ```

## Quick Start

### SSE Simulation Example

```python
from sse_simulation.sse import SingleQubitSSE, sigma_z
import numpy as np

# Create simulator with discrete measurement protocol
gamma = 5     # measurement strength  
dt = 0.001    # time step
T = 1.0       # total time

sim = SingleQubitSSE(gamma=gamma, dt=dt, T=T, meas_op=sigma_z())

# Initial state |+⟩ = (|0⟩ + |1⟩)/√2
psi0 = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)

# Run single trajectory
z_trajectory, Q_value = sim.run_trajectory(psi0, compute_entropy=True)

print(f"Entropy production Q = {Q_value:.3f}")
```

### Kraus Operators Example

```
from krauss_operators.krauss_operators_simulation import run_trajectory, simulate_Q_distribution
from krauss_operators.datatypes import InitialState
import numpy as np

# Parameters following physical relationships
N = 1000              # number of measurements
epsilon = 0.01        # measurement strength
theta = N * epsilon**2  # θ = T/τ relationship
omega_dt = 4*np.pi * epsilon**2  # ω·dt from physical constraints

# Custom initial state
initial_state = InitialState.from_unnormalized(alpha=2.0, beta=1.0)

# Single trajectory
result = run_trajectory(N, epsilon, omega_dt, initial_state)
print(f"Entropy production Q = {result.Q:.3f}")

# Ensemble simulation
Q_values = simulate_Q_distribution(
    num_traj=1000, N=N, epsilon=epsilon, 
    omega_dt=omega_dt, initial_state=initial_state
)
print(f"Mean Q = {np.mean(Q_values):.3f}")
```

## Key Concepts

### Entropy Production
Both simulators compute the **entropy production functional Q**, which quantifies the arrow of time in quantum measurements:

- **Q > 0**: Forward time evolution (more probable)
- **Q < 0**: Reverse time evolution (less probable) 
- **Q = 0**: Time-symmetric evolution

### Parameter Relationships
The discrete protocol enforces important physical relationships:
- **θ = T/τ** (dimensionless time)
- **ω = 4π/τ** (rotation frequency)
- **T·ω ≥ 2π** (minimum rotation constraint)
- **ε = √(γ·dt)** (discrete step size)

### State Validation
The `InitialState` class ensures quantum states are properly normalized:
```python
# Strict normalization required
state = InitialState(alpha=0.6, beta=0.8)  # |α|² + |β|² = 1

# Or auto-normalize from any amplitudes  
state = InitialState.from_unnormalized(alpha=3.0, beta=4.0)
```

## Notebooks

Interactive Jupyter notebooks demonstrate the simulators:

- **`sse_simulation/demo.ipynb`** - SSE simulation with discrete measurements
- **`krauss_operators/krauss_operators_notebook.ipynb`** - Kraus operators with coherent evolution

## Development

### Running Tests
```bash
# Test SSE functionality
python -c "from sse_simulation.sse import SingleQubitSSE; print('SSE import OK')"

# Test Kraus functionality  
python -c "from krauss_operators.datatypes import InitialState; print('Kraus import OK')"
```

### Code Structure
- **`sse.py`** - Main SSE simulator class with discrete measurement protocol
- **`krauss_operators_simulation.py`** - Discrete measurement functions and analysis
- **`datatypes.py`** - Validated quantum state representations
- **Notebooks** - Interactive demonstrations and parameter exploration

## Physics Background

This project implements quantum trajectory theory for single-qubit measurements, exploring:

- **Continuous vs Discrete Monitoring** - Different approaches to quantum measurement
- **Arrow of Time** - Statistical irreversibility in quantum evolution  
- **Measurement Backaction** - How measurements affect quantum states
- **Entropy Production** - Quantifying the "cost" of measurement information

### References
- Dressel et al., "Arrow of Time for Continuous Quantum Measurement"
- Turkeshi et al., "Measurement-Induced Entanglement Transitions"

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License

## Acknowledgments

- Implementation based on quantum trajectory theory literature
- Arrow of Time analytical distributions from Dressel et al.
- Kraus operator formalism following standard quantum measurement theory

---

**Note**: This is a research project for exploring quantum measurement theory. Results should be validated against analytical predictions and literature values.