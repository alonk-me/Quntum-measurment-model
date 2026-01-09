# Work Breakdown Structure: Susceptibility Implementation

## Overview
This WBS outlines the implementation of magnetic susceptibility calculations for the quantum measurement simulation framework. Susceptibility (χ) measures the linear response of a quantum system to external perturbations and is a key observable in quantum many-body physics.

## 1. Core Susceptibility Module

### 1.1 Static Susceptibility
- **File**: `quantum_measurement/susceptibility/static_susceptibility.py`
- **Functionality**:
  - Compute static (zero-frequency) magnetic susceptibility: χ = ∂⟨M⟩/∂h
  - Calculate connected correlation functions: ⟨σᵢᶻσⱼᶻ⟩ - ⟨σᵢᶻ⟩⟨σⱼᶻ⟩
  - Integrate over equal-time correlations
  - Support both single-site and multi-site susceptibilities

### 1.2 Dynamic Susceptibility
- **File**: `quantum_measurement/susceptibility/dynamic_susceptibility.py`
- **Functionality**:
  - Compute time-dependent susceptibility: χ(t) 
  - Calculate two-time correlation functions: ⟨σᵢᶻ(t)σⱼᶻ(0)⟩
  - Support for retarded, advanced, and time-ordered response functions
  - Fourier transform utilities for frequency-domain susceptibility χ(ω)

### 1.3 Susceptibility from Correlation Matrix
- **File**: `quantum_measurement/susceptibility/correlation_matrix_susceptibility.py`
- **Functionality**:
  - Extract susceptibility from free-fermion correlation matrix G(t)
  - Use Jordan-Wigner mapping: σᵢᶻ = 2c†ᵢcᵢ - 1
  - Compute χᵢⱼ(t) from G matrix elements
  - Integration with existing correlation simulators

## 2. Integration with Existing Simulators

### 2.1 Free Fermion Module Integration
- **Files to modify**:
  - `quantum_measurement/free_fermion/matrix_commutator_solver.py`
  - New: `quantum_measurement/free_fermion/susceptibility_calculator.py`
- **Changes**:
  - Add method to compute susceptibility from time-evolved G(t)
  - Store time-series data for correlation function calculations
  - Add susceptibility analysis to validation tools

### 2.2 Two-Qubit Simulator Integration
- **File to modify**: `quantum_measurement/jw_expansion/two_qubit_correlation_simulator.py`
- **Changes**:
  - Add `compute_susceptibility()` method to TwoQubitCorrelationSimulator class
  - Track two-time correlations during trajectory evolution
  - Return susceptibility alongside entropy production Q

### 2.3 L-Qubit Simulator Integration
- **File to modify**: `quantum_measurement/jw_expansion/l_qubit_correlation_simulator.py`
- **Changes**:
  - Add `compute_susceptibility()` method to LQubitCorrelationSimulator class
  - Support spatial dependence: χᵢⱼ between sites i and j
  - Handle both open and closed boundary conditions

## 3. Analytical Comparison and Validation

### 3.1 Analytical Solutions
- **File**: `quantum_measurement/susceptibility/analytical.py`
- **Functionality**:
  - Analytical susceptibility for single qubit: χ = ∂tanh(h/T)/∂h
  - Analytical solutions for two-qubit XX model
  - Known results for transverse-field Ising model
  - Reference implementations for validation

### 3.2 Statistical Comparison Tools
- **File**: `quantum_measurement/susceptibility/validation.py`
- **Functionality**:
  - Extend statistical_model_fit.py framework
  - Compare numerical susceptibility with analytical predictions
  - Chi-squared tests, correlation coefficients
  - Visualization tools for susceptibility plots

## 4. Documentation and Examples

### 4.1 Module Documentation
- **File**: `quantum_measurement/susceptibility/README.md`
- **Content**:
  - Theory background on susceptibility
  - Mathematical definitions and formulas
  - Connection to correlation functions
  - API documentation for all functions

### 4.2 Jupyter Notebook Examples
- **File**: `quantum_measurement/susceptibility/susceptibility_demo.ipynb`
- **Content**:
  - Single-qubit susceptibility calculation
  - Two-qubit correlation and susceptibility
  - L-qubit chain susceptibility
  - Comparison with analytical results
  - Visualization examples

### 4.3 Usage Examples
- **File**: `quantum_measurement/susceptibility/examples.py`
- **Content**:
  - Command-line examples
  - Simple use cases
  - Integration examples with existing simulators

## 5. Testing

### 5.1 Unit Tests
- **File**: `tests/test_susceptibility.py` (if tests directory exists)
- **Tests**:
  - Test static susceptibility calculations
  - Test dynamic susceptibility and Fourier transforms
  - Test integration with correlation matrix
  - Validate against known analytical results

### 5.2 Integration Tests
- **Tests**:
  - End-to-end workflow tests
  - Consistency with existing entropy production calculations
  - Numerical stability tests

## 6. Dependencies

### 6.1 Required Libraries
- numpy: Array operations and linear algebra
- scipy: Fourier transforms, integration
- matplotlib: Visualization
- All existing dependencies from pyproject.toml

### 6.2 Optional Dependencies
- scipy.fft: Fast Fourier transform for χ(ω)
- scipy.integrate: Numerical integration for static susceptibility

## Implementation Priority

1. **Phase 1** (Core): 
   - Create susceptibility module structure
   - Implement static susceptibility calculation
   - Add basic correlation matrix integration

2. **Phase 2** (Integration):
   - Integrate with two-qubit simulator
   - Integrate with L-qubit simulator
   - Add analytical comparison tools

3. **Phase 3** (Advanced):
   - Implement dynamic susceptibility
   - Add frequency-domain analysis
   - Complete documentation and examples

## Success Criteria

- [ ] Susceptibility module created with proper structure
- [ ] Static susceptibility calculation working
- [ ] Integration with at least one correlation simulator
- [ ] Validation against analytical results (< 1% error for simple cases)
- [ ] Documentation and usage examples completed
- [ ] All tests passing
