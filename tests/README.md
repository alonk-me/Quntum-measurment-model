# Tests for EP-simulation

This directory contains automated tests for the EP-simulation project.

## Test Structure

```
tests/
├── __init__.py
├── test_non_hermitian_simulators.py    # Unit tests for simulator classes
└── test_steady_state_integration.py     # Integration tests for analytical/numerical validation
```

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_non_hermitian_simulators.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run specific test class or function:
```bash
pytest tests/test_non_hermitian_simulators.py::TestNonHermitianHatSimulator::test_initialization
```

## Test Files

### `test_non_hermitian_simulators.py`
Unit tests for all three NonHermitian simulator variants:
- **NonHermitianHatSimulator** (base class)
- **NonHermitianSpinSimulator** (magnetization-based)
- **NonHermitianAdjustedSimulator** (adjusted entropy)

Tests cover:
- Backward compatibility (`return_G_final=False`)
- New signature with G_final (`return_G_final=True`)
- Hermiticity of correlation matrices
- Physical eigenvalue ranges
- Consistency across simulator variants
- Boundary conditions (open/periodic)

**Runtime**: ~0.4 seconds (uses minimal parameters for speed)

### `test_steady_state_integration.py`
Integration tests validating the connection between:
- Analytical steady-state formulas (`n_infty_steady_state.py`)
- Numerical simulations (`non_hermitian_hat.py`)

Tests cover:
- Steady-state extraction from simulations
- Eigenvalue computation and structure
- Validation function output structure
- Physical constraints (occupations in [0,1], positive entropy)
- Parameter variations (L, gamma)

**Runtime**: ~0.8 seconds (uses short simulations: L≤5, N_steps≤1000)

## Configuration

Tests are configured via `pytest.ini` in the project root:
- Test discovery pattern: `test_*.py`
- Verbose output enabled by default
- Short traceback format

## Backward Compatibility

All simulator classes support both old and new return signatures:

**Old (backward compatible)**:
```python
Q, n_traj = sim.simulate_trajectory(return_G_final=False)
```

**New (with correlation matrix)**:
```python
Q, n_traj, G_final = sim.simulate_trajectory(return_G_final=True)
```

Default behavior:
- `NonHermitianHatSimulator`: `return_G_final=True` (new behavior)
- `NonHermitianSpinSimulator`: `return_G_final=False` (backward compatible)
- `NonHermitianAdjustedSimulator`: `return_G_final=False` (backward compatible)

## Test Performance

All tests use minimal parameters for fast execution:
- System size: L = 2-5
- Time steps: N_steps = 100-1000
- Total test suite runtime: < 1 second

For production simulations with accurate results, use larger values as shown in the notebooks.

## Adding New Tests

1. Create new test file following naming convention: `test_*.py`
2. Import required modules from `quantum_measurement/jw_expansion/`
3. Use fixtures for common parameters (see existing tests for examples)
4. Keep tests fast by using minimal simulation parameters
5. Run tests to verify they pass: `pytest tests/your_new_test.py -v`

## Dependencies

Tests require:
- `pytest` (installed in project virtual environment)
- `numpy`
- All modules from `quantum_measurement/jw_expansion/`

Install pytest:
```bash
pip install pytest
```
