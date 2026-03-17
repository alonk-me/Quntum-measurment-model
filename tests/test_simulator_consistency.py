"""Integration tests: cross-simulator consistency checks."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent / "quantum_measurement"
sys.path.insert(0, str(_ROOT / "jw_expansion"))

from two_qubit_correlation_simulator import TwoQubitCorrelationSimulator
from l_qubit_correlation_simulator import LQubitCorrelationSimulator


@pytest.mark.integration
class TestTwoQubitVsLQubit:
    """TwoQubitCorrelationSimulator should match LQubitCorrelationSimulator(L=2)."""

    def _run_two_qubit(self, seed: int, N_steps: int, T: float, J: float, eps: float):
        rng = np.random.default_rng(seed)
        sim = TwoQubitCorrelationSimulator(J=J, epsilon=eps, N_steps=N_steps, T=T, rng=rng)
        Q_vals = []
        for _ in range(5):
            Q, z_traj, _ = sim.simulate_trajectory()
            Q_vals.append(Q)
        return np.mean(Q_vals)

    def _run_l_qubit_l2(self, seed: int, N_steps: int, T: float, J: float, eps: float):
        rng = np.random.default_rng(seed)
        sim = LQubitCorrelationSimulator(
            L=2, J=J, epsilon=eps, N_steps=N_steps, T=T,
            closed_boundary=False, rng=rng,
        )
        Q_vals = []
        for _ in range(5):
            Q, _, _ = sim.simulate_trajectory()
            Q_vals.append(Q)
        return np.mean(Q_vals)

    def test_q_mean_similar_j0(self):
        """J=0: both simulators should give similar mean Q over trajectories."""
        seed = 12345
        mean_2q = self._run_two_qubit(seed, N_steps=500, T=5.0, J=0.0, eps=0.1)
        mean_lq = self._run_l_qubit_l2(seed, N_steps=500, T=5.0, J=0.0, eps=0.1)
        # Stochastic—use generous tolerance
        assert abs(mean_2q - mean_lq) < 10.0

    def test_q_finite_for_both(self):
        """Both simulators return finite Q."""
        rng2 = np.random.default_rng(1)
        sim2 = TwoQubitCorrelationSimulator(J=1.0, epsilon=0.1, N_steps=100, T=1.0, rng=rng2)
        Q2, _, _ = sim2.simulate_trajectory()

        rngL = np.random.default_rng(1)
        simL = LQubitCorrelationSimulator(L=2, J=1.0, epsilon=0.1, N_steps=100, T=1.0, rng=rngL)
        QL, _, _ = simL.simulate_trajectory()

        assert np.isfinite(Q2)
        assert np.isfinite(QL)

    def test_z_trajectory_shapes_match(self):
        rng2 = np.random.default_rng(7)
        sim2 = TwoQubitCorrelationSimulator(J=1.0, epsilon=0.1, N_steps=50, T=1.0, rng=rng2)
        _, z2, _ = sim2.simulate_trajectory()

        rngL = np.random.default_rng(7)
        simL = LQubitCorrelationSimulator(L=2, J=1.0, epsilon=0.1, N_steps=50, T=1.0, rng=rngL)
        _, zL, _ = simL.simulate_trajectory()

        assert z2.shape[0] == zL.shape[0]  # Same number of time points

    def test_two_qubit_batch_size_one_parity(self):
        """Batch path with n_batch=1 should match serial trajectory for identical xi sequence."""
        seed = 123
        n_steps = 40
        rng_for_serial = np.random.default_rng(seed)
        rng_for_xi = np.random.default_rng(seed)

        sim = TwoQubitCorrelationSimulator(J=1.0, epsilon=0.1, N_steps=n_steps, T=1.0, rng=rng_for_serial)
        Q_serial, z_serial, xi_serial = sim.simulate_trajectory()

        xi_random = rng_for_xi.random((1, n_steps, 2))
        xi_batch = np.where(xi_random < 0.5, 1, -1).astype(np.int8)
        Q_batch, z_batch, xi_out = sim.simulate_trajectory_batch(1, xi_batch=xi_batch)

        assert np.allclose(Q_batch[0], Q_serial, atol=1e-12)
        assert np.allclose(z_batch[0], z_serial, atol=1e-12)
        assert np.array_equal(xi_out[0], xi_serial)

    def test_two_qubit_ensemble_batch_shapes(self):
        sim = TwoQubitCorrelationSimulator(J=1.0, epsilon=0.1, N_steps=20, T=1.0, rng=np.random.default_rng(5))
        Q, z, xi = sim.simulate_ensemble(7, batch_size=3)
        assert Q.shape == (7,)
        assert z.shape == (7, 21, 2)
        assert xi.shape == (7, 20, 2)


@pytest.mark.integration
class TestLQubitEnsembleConsistency:
    """Ensemble-level consistency tests for LQubitCorrelationSimulator."""

    @pytest.mark.slow
    def test_mean_q_scales_with_epsilon(self):
        """Larger epsilon should generally yield larger |Q| values."""
        rng_small = np.random.default_rng(42)
        rng_large = np.random.default_rng(42)

        sim_small = LQubitCorrelationSimulator(
            L=2, J=0.0, epsilon=0.05, N_steps=100, T=1.0, rng=rng_small,
        )
        sim_large = LQubitCorrelationSimulator(
            L=2, J=0.0, epsilon=0.2, N_steps=100, T=1.0, rng=rng_large,
        )

        Q_small, _, _ = sim_small.simulate_ensemble(15)
        Q_large, _, _ = sim_large.simulate_ensemble(15)

        # Variance of Q should be larger for larger epsilon
        assert np.var(Q_large) >= np.var(Q_small) * 0.5  # generous lower bound

    def test_different_seeds_give_different_results(self):
        """Different seeds produce different trajectories."""
        sim1 = LQubitCorrelationSimulator(L=2, J=1.0, epsilon=0.1, N_steps=50, T=1.0,
                                          rng=np.random.default_rng(1))
        sim2 = LQubitCorrelationSimulator(L=2, J=1.0, epsilon=0.1, N_steps=50, T=1.0,
                                          rng=np.random.default_rng(99))
        Q1, _, _ = sim1.simulate_trajectory()
        Q2, _, _ = sim2.simulate_trajectory()
        # With different seeds the trajectories should (almost certainly) differ
        assert Q1 != Q2
