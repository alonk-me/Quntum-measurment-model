"""Unit tests for NonHermitian simulator classes.

Tests all three simulator variants:
- NonHermitianHatSimulator (base class)
- NonHermitianSpinSimulator (magnetization-based)
- NonHermitianAdjustedSimulator (adjusted entropy)
"""

import sys
from pathlib import Path

# Add quantum_measurement to path
sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "jw_expansion"))

import numpy as np
import pytest
from non_hermitian_hat import NonHermitianHatSimulator
from non_hermitian_spin import NonHermitianSpinSimulator
from non_hermitian_adjusted import NonHermitianAdjustedSimulator


@pytest.fixture
def minimal_params():
    """Minimal parameters for very quick tests."""
    return {
        'L': 2,
        'J': 1.0,
        'gamma': 0.5,
        'dt': 0.01,
        'N_steps': 100,
        'closed_boundary': False
    }


class TestNonHermitianHatSimulator:
    """Tests for base NonHermitianHatSimulator class."""
    
    def test_initialization(self, minimal_params):
        """Test simulator initializes correctly."""
        sim = NonHermitianHatSimulator(**minimal_params)
        
        assert sim.L == minimal_params['L']
        assert sim.J == minimal_params['J']
        assert sim.gamma == minimal_params['gamma']
        assert sim.T_total == minimal_params['dt'] * minimal_params['N_steps']
    
    def test_backward_compatibility_mode(self, minimal_params):
        """Test that return_G_final=False works (backward compatibility)."""
        sim = NonHermitianHatSimulator(**minimal_params)
        result = sim.simulate_trajectory(return_G_final=False)
        
        # Should return only 2 values
        assert len(result) == 2
        Q_total, n_traj = result
        
        assert isinstance(Q_total, (float, np.floating))
        assert n_traj.shape == (minimal_params['N_steps'] + 1, minimal_params['L'])
    
    def test_new_mode_with_G_final(self, minimal_params):
        """Test that return_G_final=True returns 3 values."""
        sim = NonHermitianHatSimulator(**minimal_params)
        result = sim.simulate_trajectory(return_G_final=True)
        
        # Should return 3 values
        assert len(result) == 3
        Q_total, n_traj, G_final = result
        
        assert isinstance(Q_total, (float, np.floating))
        assert n_traj.shape == (minimal_params['N_steps'] + 1, minimal_params['L'])
        assert G_final.shape == (2 * minimal_params['L'], 2 * minimal_params['L'])
    
    def test_G_final_is_hermitian(self, minimal_params):
        """Test that final correlation matrix is Hermitian."""
        sim = NonHermitianHatSimulator(**minimal_params)
        _, _, G_final = sim.simulate_trajectory(return_G_final=True)
        
        assert np.allclose(G_final, G_final.conj().T, atol=1e-10)
    
    def test_G_final_eigenvalues_mostly_physical(self, minimal_params):
        """Test that G_final has mostly physical eigenvalues."""
        sim = NonHermitianHatSimulator(**minimal_params)
        _, _, G_final = sim.simulate_trajectory(return_G_final=True)
        
        eigs = np.linalg.eigvalsh(G_final)
        
        # Most eigenvalues should be in rough range [0, 1]
        # Allow tolerance for numerical integration errors
        n_reasonable = np.sum((eigs >= -0.2) & (eigs <= 1.2))
        assert n_reasonable >= len(eigs) * 0.7  # At least 70%


class TestNonHermitianSpinSimulator:
    """Tests for NonHermitianSpinSimulator class."""
    
    def test_backward_compatibility(self, minimal_params):
        """Test spin simulator backward compatibility."""
        sim = NonHermitianSpinSimulator(**minimal_params)
        result = sim.simulate_trajectory(return_G_final=False)
        
        assert len(result) == 2
        Q, z_traj = result
        
        # Magnetization should be in range [-1, 1]
        assert np.all(z_traj >= -1.1), "Magnetization below -1"
        assert np.all(z_traj <= 1.1), "Magnetization above 1"
    
    def test_with_G_final(self, minimal_params):
        """Test spin simulator can return G_final."""
        sim = NonHermitianSpinSimulator(**minimal_params)
        Q, z_traj, G_final = sim.simulate_trajectory(return_G_final=True)
        
        assert G_final.shape == (2 * minimal_params['L'], 2 * minimal_params['L'])
        assert np.allclose(G_final, G_final.conj().T, atol=1e-10)
    
    def test_magnetization_from_occupation(self, minimal_params):
        """Test that z = 2n - 1 relation holds."""
        sim = NonHermitianSpinSimulator(**minimal_params)
        Q, z_traj, G_final = sim.simulate_trajectory(return_G_final=True)
        
        # Extract occupations from G_final
        n_final = np.real(np.diag(G_final)[:minimal_params['L']])
        
        # Extract final magnetizations
        z_final = z_traj[-1, :]
        
        # Check z = 2n - 1
        z_from_n = 2 * n_final - 1
        assert np.allclose(z_final, z_from_n, atol=1e-10)


class TestNonHermitianAdjustedSimulator:
    """Tests for NonHermitianAdjustedSimulator class."""
    
    def test_backward_compatibility(self, minimal_params):
        """Test adjusted simulator backward compatibility."""
        sim = NonHermitianAdjustedSimulator(**minimal_params)
        result = sim.simulate_trajectory(return_G_final=False)
        
        assert len(result) == 2
        Q_adj, n_traj = result
        
        # Adjusted Q can be negative
        assert isinstance(Q_adj, (float, np.floating))
    
    def test_with_G_final(self, minimal_params):
        """Test adjusted simulator can return G_final."""
        sim = NonHermitianAdjustedSimulator(**minimal_params)
        Q_adj, n_traj, G_final = sim.simulate_trajectory(return_G_final=True)
        
        assert G_final.shape == (2 * minimal_params['L'], 2 * minimal_params['L'])
        assert np.allclose(G_final, G_final.conj().T, atol=1e-10)
    
    def test_entropy_adjustment(self, minimal_params):
        """Test that adjustment formula is applied correctly."""
        # Create base simulator
        sim_base = NonHermitianHatSimulator(**minimal_params)
        Q_base, n_traj_base, G_base = sim_base.simulate_trajectory(return_G_final=True)
        
        # Create adjusted simulator with same params
        sim_adj = NonHermitianAdjustedSimulator(**minimal_params)
        Q_adj, n_traj_adj, G_adj = sim_adj.simulate_trajectory(return_G_final=True)
        
        # Check adjustment formula: Q_adj = gamma * L * T_total - Q_base
        expected_Q_adj = (minimal_params['gamma'] * minimal_params['L'] * 
                          minimal_params['dt'] * minimal_params['N_steps'] - Q_base)
        
        assert np.isclose(Q_adj, expected_Q_adj, rtol=1e-10)
        
        # Trajectories should be identical (same evolution)
        assert np.allclose(n_traj_base, n_traj_adj, atol=1e-10)
        assert np.allclose(G_base, G_adj, atol=1e-10)


class TestConsistencyAcrossSimulators:
    """Test consistency between different simulator variants."""
    
    def test_same_G_final_for_all_variants(self, minimal_params):
        """All three simulators should produce same G_final."""
        sim_hat = NonHermitianHatSimulator(**minimal_params)
        sim_spin = NonHermitianSpinSimulator(**minimal_params)
        sim_adj = NonHermitianAdjustedSimulator(**minimal_params)
        
        _, _, G_hat = sim_hat.simulate_trajectory(return_G_final=True)
        _, _, G_spin = sim_spin.simulate_trajectory(return_G_final=True)
        _, _, G_adj = sim_adj.simulate_trajectory(return_G_final=True)
        
        # All should produce same final correlation matrix
        assert np.allclose(G_hat, G_spin, atol=1e-10)
        assert np.allclose(G_hat, G_adj, atol=1e-10)


@pytest.mark.parametrize("boundary", [False, True])
def test_boundary_conditions(boundary, minimal_params):
    """Test both open and periodic boundary conditions."""
    params = minimal_params.copy()
    params['closed_boundary'] = boundary
    
    sim = NonHermitianHatSimulator(**params)
    Q, n_traj, G = sim.simulate_trajectory(return_G_final=True)
    
    # Should complete without errors
    assert G.shape == (2 * params['L'], 2 * params['L'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
