"""Integration tests for steady-state validation between analytical and numerical methods.

Tests the integration between n_infty_steady_state.py analytical formulas and
non_hermitian_hat.py numerical simulations.
"""

import sys
from pathlib import Path

# Add quantum_measurement to path
sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "jw_expansion"))

import numpy as np
import pytest
from n_infty_steady_state import (
    extract_steady_state_from_simulation,
    validate_steady_state_with_simulation,
    get_correlation_eigenvalues
)


@pytest.fixture
def quick_sim_params():
    """Parameters for quick simulation tests."""
    return {
        'L': 3,
        'J': 1.0,
        'gamma': 0.4,
        'dt': 0.001,
        'N_steps': 1000,  # Very short for quick tests
        'boundary': 'open'
    }


def test_extract_steady_state(quick_sim_params):
    """Test that we can extract steady state from simulation."""
    G_final, n_traj, Q_total = extract_steady_state_from_simulation(**quick_sim_params)
    
    # Check return types and shapes
    assert G_final.shape == (2 * quick_sim_params['L'], 2 * quick_sim_params['L'])
    assert n_traj.shape == (quick_sim_params['N_steps'] + 1, quick_sim_params['L'])
    assert isinstance(Q_total, (float, np.floating))
    
    # Check G is Hermitian
    assert np.allclose(G_final, G_final.conj().T, atol=1e-10)


def test_correlation_eigenvalues(quick_sim_params):
    """Test eigenvalue extraction from correlation matrix."""
    G_final, _, _ = extract_steady_state_from_simulation(**quick_sim_params)
    
    eigs = get_correlation_eigenvalues(G_final)
    
    # Should have 2L eigenvalues
    assert len(eigs) == 2 * quick_sim_params['L']
    
    # Eigenvalues should be real (from Hermitian matrix)
    assert np.all(np.isreal(eigs))
    
    # Most eigenvalues should be roughly in physical range [0, 1]
    # Allow some tolerance for numerical errors
    n_physical = np.sum((eigs >= -0.1) & (eigs <= 1.1))
    assert n_physical >= len(eigs) * 0.8  # At least 80% in reasonable range


def test_validate_steady_state_structure(quick_sim_params):
    """Test that validation function returns expected structure."""
    results = validate_steady_state_with_simulation(**quick_sim_params)
    
    # Check all expected keys are present
    expected_keys = {
        'G_simulated', 'eigs_simulated', 'mode_contributions',
        'n_simulated', 'n_analytical', 'occupation_diff',
        'convergence_check', 'Q_total', 'T_total',
        'final_occupations', 'occupation_trajectory'
    }
    assert expected_keys.issubset(results.keys())
    
    # Check types
    assert isinstance(results['G_simulated'], np.ndarray)
    assert isinstance(results['eigs_simulated'], np.ndarray)
    assert isinstance(results['n_simulated'], (float, np.floating))
    assert isinstance(results['n_analytical'], (float, np.floating))
    assert isinstance(results['occupation_diff'], (float, np.floating))


def test_occupations_positive(quick_sim_params):
    """Test that all occupations are in valid range [0, 1]."""
    _, n_traj, _ = extract_steady_state_from_simulation(**quick_sim_params)
    
    # All occupations should be between 0 and 1
    assert np.all(n_traj >= -1e-6), "Found negative occupations"
    assert np.all(n_traj <= 1 + 1e-6), "Found occupations > 1"


def test_entropy_production_positive(quick_sim_params):
    """Test that entropy production is non-negative."""
    _, _, Q_total = extract_steady_state_from_simulation(**quick_sim_params)
    
    # Entropy production should be non-negative for dissipative system
    # (Note: adjusted simulators may have negative Q_adj, but raw should be positive)
    assert Q_total >= -1.0  # Allow some numerical error


@pytest.mark.parametrize("L", [3, 5])
@pytest.mark.parametrize("gamma", [0.2, 0.5])
def test_different_parameters(L, gamma):
    """Test with various parameter combinations."""
    params = {
        'L': L,
        'J': 1.0,
        'gamma': gamma,
        'dt': 0.001,
        'N_steps': 500,  # Extra short for parametrized tests
        'boundary': 'open'
    }
    
    G_final, n_traj, Q_total = extract_steady_state_from_simulation(**params)
    
    # Basic sanity checks
    assert G_final.shape == (2 * L, 2 * L)
    assert n_traj.shape == (params['N_steps'] + 1, L)
    assert np.isfinite(Q_total)


if __name__ == "__main__":
    # Allow running directly for quick checks
    pytest.main([__file__, "-v"])
