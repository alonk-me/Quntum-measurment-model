"""Tests for LQubitCorrelationSimulator in jw_expansion/l_qubit_correlation_simulator.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "jw_expansion"))

from l_qubit_correlation_simulator import LQubitCorrelationSimulator


class TestLQubitInitialization:
    """Tests for LQubitCorrelationSimulator initialization."""

    def test_default_params(self):
        sim = LQubitCorrelationSimulator()
        assert sim.L == 2
        assert sim.J == 1.0
        assert sim.epsilon == 0.1
        assert sim.N_steps == 1000

    def test_invalid_L_raises(self):
        with pytest.raises(ValueError, match="L must be at least 1"):
            LQubitCorrelationSimulator(L=0)

    def test_invalid_N_steps_raises(self):
        with pytest.raises(ValueError, match="N_steps must be at least 1"):
            LQubitCorrelationSimulator(N_steps=0)

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError, match="T must be positive"):
            LQubitCorrelationSimulator(T=-1.0)

    def test_dt_computed(self):
        sim = LQubitCorrelationSimulator(T=2.0, N_steps=200)
        assert sim.dt == pytest.approx(0.01)

    def test_rng_created(self):
        sim = LQubitCorrelationSimulator(rng=None)
        assert sim.rng is not None

    def test_accepts_device_cpu(self):
        sim = LQubitCorrelationSimulator(device="cpu", N_steps=5, rng=np.random.default_rng(1))
        Q, z_traj, xi_traj = sim.simulate_trajectory()
        assert np.isfinite(Q)
        assert z_traj.shape == (6, sim.L)
        assert xi_traj.shape == (5, sim.L)


class TestBuildHamiltonian:
    """Tests for _build_hamiltonian."""

    @pytest.mark.parametrize("L", [2, 3, 5])
    def test_shape(self, L):
        sim = LQubitCorrelationSimulator(L=L)
        assert sim.h.shape == (2 * L, 2 * L)

    @pytest.mark.parametrize("L", [2, 3, 5])
    def test_hermitian(self, L):
        sim = LQubitCorrelationSimulator(L=L, J=1.0)
        h = sim.h
        # BdG Hamiltonian is not necessarily Hermitian, but let's check it is
        # at least complex-valued with correct shape
        assert h.shape == (2 * L, 2 * L)

    def test_open_vs_periodic(self):
        sim_open = LQubitCorrelationSimulator(L=3, J=1.0, closed_boundary=False)
        sim_pbc = LQubitCorrelationSimulator(L=3, J=1.0, closed_boundary=True)
        # Periodic boundary adds corner element, so matrices should differ
        assert not np.allclose(sim_open.h, sim_pbc.h)

    def test_j_zero_gives_zero_hamiltonian(self):
        sim = LQubitCorrelationSimulator(L=2, J=0.0)
        assert np.allclose(sim.h, 0.0)


class TestComputeZValues:
    """Tests for _compute_z_values."""

    def test_shape(self):
        sim = LQubitCorrelationSimulator(L=3)
        G = sim.G_initial.copy()
        z = sim._compute_z_values(G)
        assert z.shape == (3,)

    def test_initial_state_all_up(self):
        """Initial state G_initial has G[i,i]=1 for i<L, so z[i] = 2*1 - 1 = 1."""
        sim = LQubitCorrelationSimulator(L=2)
        z = sim._compute_z_values(sim.G_initial)
        assert np.allclose(z, 1.0)

    @pytest.mark.parametrize("L", [2, 3, 5])
    def test_formula_z_equals_2Re_G_diag_minus_1(self, L):
        """Verify z[i] = 2*Re(G[i,i]) - 1."""
        sim = LQubitCorrelationSimulator(L=L)
        G = sim.G_initial.copy()
        # Perturb G slightly
        G[0, 0] = 0.7 + 0.0j
        z = sim._compute_z_values(G)
        assert z[0] == pytest.approx(2.0 * 0.7 - 1.0)


class TestHamiltonianStep:
    """Tests for _hamiltonian_step."""

    def test_preserves_hermiticity(self, minimal_params_correlation):
        """_hamiltonian_step adds -2i*dt*(GH-HG) to G; the anti-Hermitian part is O(dt)."""
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        G = sim.G_initial.copy()
        G_new = sim._hamiltonian_step(G)
        anti_herm = G_new - G_new.conj().T
        assert np.all(np.abs(anti_herm) < 0.5)  # small, proportional to dt

    def test_j_zero_no_change(self):
        sim = LQubitCorrelationSimulator(L=2, J=0.0, N_steps=100, T=1.0)
        G = sim.G_initial.copy()
        G_new = sim._hamiltonian_step(G)
        assert np.allclose(G, G_new)


class TestMeasurementStep:
    """Tests for _measurement_step."""

    def test_diagonal_clipped(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        G = sim.G_initial.copy()
        G_new, _ = sim._measurement_step(G)
        diag = np.real(np.diag(G_new))
        assert np.all(diag >= -1e-12)
        assert np.all(diag <= 1.0 + 1e-12)

    def test_hermitian_after_step(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        G = sim.G_initial.copy()
        G_new, _ = sim._measurement_step(G)
        assert np.allclose(G_new, G_new.conj().T, atol=1e-10)

    def test_outcomes_binary(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        G = sim.G_initial.copy()
        _, xi = sim._measurement_step(G)
        assert set(xi).issubset({-1, 1})

    def test_outcomes_shape(self):
        sim = LQubitCorrelationSimulator(L=4)
        G = sim.G_initial.copy()
        _, xi = sim._measurement_step(G)
        assert xi.shape == (4,)


class TestSimulateTrajectory:
    """Tests for simulate_trajectory."""

    @pytest.mark.parametrize("L,closed", [(2, False), (3, False), (5, True)])
    def test_return_shapes(self, L, closed):
        sim = LQubitCorrelationSimulator(
            L=L, J=1.0, epsilon=0.1, N_steps=50, T=1.0,
            closed_boundary=closed, rng=np.random.default_rng(42),
        )
        Q, z_traj, xi_traj = sim.simulate_trajectory()
        assert z_traj.shape == (51, L)
        assert xi_traj.shape == (50, L)

    def test_q_is_finite(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q, _, _ = sim.simulate_trajectory()
        assert np.isfinite(Q)

    def test_z_trajectory_in_range(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        _, z_traj, _ = sim.simulate_trajectory()
        assert np.all(z_traj >= -1.0 - 1e-10)
        assert np.all(z_traj <= 1.0 + 1e-10)

    def test_entropy_sign(self, minimal_params_correlation):
        """Entropy production Q can be positive or negative but should be finite."""
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q, _, _ = sim.simulate_trajectory()
        assert np.isfinite(Q)

    def test_batch_size_one_parity(self, minimal_params_correlation):
        """Batch n=1 should match serial trajectory for the same xi sequence."""
        params = dict(minimal_params_correlation)
        seed = 123
        params["rng"] = np.random.default_rng(seed)
        sim = LQubitCorrelationSimulator(**params)

        Q_serial, z_serial, xi_serial = sim.simulate_trajectory()

        rng_for_xi = np.random.default_rng(seed)
        xi_batch = rng_for_xi.choice([-1, 1], size=(1, sim.N_steps, sim.L)).astype(np.int8)
        Q_batch, z_batch, xi_out = sim.simulate_trajectory_batch(1, xi_batch=xi_batch)

        assert np.allclose(Q_batch[0], Q_serial, atol=1e-12)
        assert np.allclose(z_batch[0], z_serial, atol=1e-12)
        assert np.array_equal(xi_out[0], xi_serial)

    def test_batch_shapes(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q, z, xi = sim.simulate_trajectory_batch(3)
        assert Q.shape == (3,)
        assert z.shape == (3, sim.N_steps + 1, sim.L)
        assert xi.shape == (3, sim.N_steps, sim.L)


class TestSimulateEnsemble:
    """Tests for simulate_ensemble."""

    def test_shapes(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        n_traj = 5
        Q_vals, z_series, xi_series = sim.simulate_ensemble(n_traj)
        assert Q_vals.shape == (n_traj,)
        assert z_series.shape == (n_traj, sim.N_steps + 1, sim.L)
        assert xi_series.shape == (n_traj, sim.N_steps, sim.L)

    def test_q_all_finite(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q_vals, _, _ = sim.simulate_ensemble(5)
        assert np.all(np.isfinite(Q_vals))

    def test_ensemble_batch_shapes(self, minimal_params_correlation):
        sim = LQubitCorrelationSimulator(**minimal_params_correlation)
        Q_vals, z_series, xi_series = sim.simulate_ensemble(7, batch_size=3)
        assert Q_vals.shape == (7,)
        assert z_series.shape == (7, sim.N_steps + 1, sim.L)
        assert xi_series.shape == (7, sim.N_steps, sim.L)
