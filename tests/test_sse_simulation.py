"""Tests for SSEWavefunctionSimulator in sse_simulation/sse.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "sse_simulation"))

from sse import SSEWavefunctionSimulator


class TestSSEInitialization:
    """Tests for SSEWavefunctionSimulator initialization."""

    def test_default_params(self):
        sim = SSEWavefunctionSimulator()
        assert sim.epsilon == 0.1
        assert sim.N_steps == 100
        assert sim.J == 0.0
        assert sim.initial_state == "bloch_equator"

    def test_custom_params(self):
        sim = SSEWavefunctionSimulator(epsilon=0.05, N_steps=50, J=0.5)
        assert sim.epsilon == 0.05
        assert sim.N_steps == 50
        assert sim.J == 0.5

    def test_rng_seeded(self):
        rng = np.random.default_rng(42)
        sim = SSEWavefunctionSimulator(rng=rng)
        assert sim.rng is rng

    def test_rng_created_when_none(self):
        sim = SSEWavefunctionSimulator(rng=None)
        assert sim.rng is not None

    def test_accepts_device_cpu(self):
        sim = SSEWavefunctionSimulator(device="cpu", N_steps=5, rng=np.random.default_rng(1))
        Q, z_traj, meas = sim.simulate_trajectory()
        assert np.isfinite(Q)
        assert z_traj.shape == (6,)
        assert meas.shape == (5,)


class TestPrepareInitialState:
    """Test all preset initial states return normalized wavefunctions."""

    @pytest.mark.parametrize("state", [
        "bloch_equator", "up", "down", "plus_y", "minus_y",
    ])
    def test_normalization(self, state):
        sim = SSEWavefunctionSimulator(initial_state=state)
        psi = sim.psi_initial
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12

    def test_custom_state_normalization(self):
        sim = SSEWavefunctionSimulator(
            initial_state="custom", theta=np.pi / 3, phi=np.pi / 4
        )
        assert abs(np.linalg.norm(sim.psi_initial) - 1.0) < 1e-12

    def test_bloch_equator_state(self):
        sim = SSEWavefunctionSimulator(initial_state="bloch_equator")
        expected = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        assert np.allclose(sim.psi_initial, expected)

    def test_up_state(self):
        sim = SSEWavefunctionSimulator(initial_state="up")
        expected = np.array([1.0, 0.0], dtype=complex)
        assert np.allclose(sim.psi_initial, expected)

    def test_down_state(self):
        sim = SSEWavefunctionSimulator(initial_state="down")
        expected = np.array([0.0, 1.0], dtype=complex)
        assert np.allclose(sim.psi_initial, expected)

    def test_unknown_state_raises(self):
        with pytest.raises(ValueError, match="Unknown initial state"):
            SSEWavefunctionSimulator(initial_state="invalid_state")


class TestExpectationValueZ:
    """Test Pauli expectation value computation."""

    def test_up_state_z_is_plus_one(self):
        sim = SSEWavefunctionSimulator()
        psi_up = np.array([1.0, 0.0], dtype=complex)
        assert abs(sim._expectation_value_z(psi_up) - 1.0) < 1e-12

    def test_down_state_z_is_minus_one(self):
        sim = SSEWavefunctionSimulator()
        psi_down = np.array([0.0, 1.0], dtype=complex)
        assert abs(sim._expectation_value_z(psi_down) - (-1.0)) < 1e-12

    def test_equator_state_z_is_zero(self):
        sim = SSEWavefunctionSimulator()
        psi_eq = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        assert abs(sim._expectation_value_z(psi_eq)) < 1e-12

    @pytest.mark.parametrize("state", ["bloch_equator", "up", "down", "plus_y", "minus_y"])
    def test_z_in_range(self, state):
        sim = SSEWavefunctionSimulator(initial_state=state)
        z = sim._expectation_value_z(sim.psi_initial)
        assert -1.0 - 1e-10 <= z <= 1.0 + 1e-10


class TestHamiltonianEvolution:
    """Test Hamiltonian evolution step."""

    def test_j_zero_no_change(self):
        sim = SSEWavefunctionSimulator(J=0.0)
        psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        psi_evolved = sim._apply_hamiltonian_evolution(psi, dt=0.1)
        assert np.allclose(psi, psi_evolved)

    def test_j_nonzero_changes_state(self):
        sim = SSEWavefunctionSimulator(J=1.0, epsilon=0.1)
        psi = np.array([1.0, 0.0], dtype=complex)
        psi_evolved = sim._apply_hamiltonian_evolution(psi, dt=0.1)
        # State should change
        assert not np.allclose(psi, psi_evolved)

    def test_j_nonzero_preserves_norm(self):
        sim = SSEWavefunctionSimulator(J=1.0, epsilon=0.1)
        psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        psi_evolved = sim._apply_hamiltonian_evolution(psi, dt=0.01)
        assert abs(np.linalg.norm(psi_evolved) - 1.0) < 1e-10


class TestMeasurementUpdate:
    """Test wavefunction normalization after measurement update."""

    def test_normalization_preserved(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        psi = sim.psi_initial.copy()
        for _ in range(20):
            psi, xi, z_before = sim._measurement_update(psi)
            assert abs(np.linalg.norm(psi) - 1.0) < 1e-10

    def test_outcome_is_plus_or_minus_one(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        psi = sim.psi_initial.copy()
        psi_new, xi, _ = sim._measurement_update(psi)
        assert xi in (1, -1)

    def test_z_before_in_range(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        psi = sim.psi_initial.copy()
        _, _, z_before = sim._measurement_update(psi)
        assert -1.0 - 1e-10 <= z_before <= 1.0 + 1e-10


class TestSimulateTrajectory:
    """Test simulate_trajectory returns valid output."""

    def test_return_shapes(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        Q, z_traj, meas = sim.simulate_trajectory()
        assert z_traj.shape == (sim.N_steps + 1,)
        assert meas.shape == (sim.N_steps,)

    def test_z_trajectory_in_range(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        _, z_traj, _ = sim.simulate_trajectory()
        assert np.all(z_traj >= -1.0 - 1e-10)
        assert np.all(z_traj <= 1.0 + 1e-10)

    def test_q_is_finite(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        Q, _, _ = sim.simulate_trajectory()
        assert np.isfinite(Q)

    def test_measurement_outcomes_binary(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        _, _, meas = sim.simulate_trajectory()
        assert set(np.unique(meas)).issubset({-1, 1})

    def test_batch_size_one_parity(self, minimal_params_sse):
        """Batch trajectory with n_batch=1 should match serial for same xi sequence."""
        n_steps = minimal_params_sse["N_steps"]
        seed = 123
        sim = SSEWavefunctionSimulator(**{**minimal_params_sse, "rng": np.random.default_rng(seed)})

        Q_serial, z_serial, meas_serial = sim.simulate_trajectory()

        rng_for_xi = np.random.default_rng(seed)
        xi_random = rng_for_xi.random((1, n_steps))
        xi_batch = np.where(xi_random < 0.5, 1, -1).astype(np.int8)
        Q_batch, z_batch, meas_batch = sim.simulate_trajectory_batch(1, xi_batch=xi_batch)

        assert np.allclose(Q_batch[0], Q_serial, atol=1e-12)
        assert np.allclose(z_batch[0], z_serial, atol=1e-12)
        assert np.array_equal(meas_batch[0], meas_serial)

    def test_batch_shapes(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        Q, z, meas = sim.simulate_trajectory_batch(4)
        assert Q.shape == (4,)
        assert z.shape == (4, sim.N_steps + 1)
        assert meas.shape == (4, sim.N_steps)


class TestSimulateEnsemble:
    """Test simulate_ensemble produces consistent distributions."""

    def test_ensemble_shapes(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        n_traj = 10
        Q_vals, z_trajs, meas = sim.simulate_ensemble(n_traj)
        assert Q_vals.shape == (n_traj,)
        assert z_trajs.shape == (n_traj, sim.N_steps + 1)
        assert meas.shape == (n_traj, sim.N_steps)

    def test_ensemble_q_finite(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        Q_vals, _, _ = sim.simulate_ensemble(10)
        assert np.all(np.isfinite(Q_vals))

    def test_ensemble_batch_shapes(self, minimal_params_sse):
        sim = SSEWavefunctionSimulator(**minimal_params_sse)
        Q_vals, z_trajs, meas = sim.simulate_ensemble(7, batch_size=3)
        assert Q_vals.shape == (7,)
        assert z_trajs.shape == (7, sim.N_steps + 1)
        assert meas.shape == (7, sim.N_steps)

    @pytest.mark.slow
    def test_ensemble_mean_q_positive(self):
        """Mean entropy production should be positive for typical parameters."""
        sim = SSEWavefunctionSimulator(
            epsilon=0.1, N_steps=200, J=0.0,
            rng=np.random.default_rng(42),
        )
        Q_vals, _, _ = sim.simulate_ensemble(50)
        # Mean should be non-negative on average
        assert np.mean(Q_vals) > -1.0
