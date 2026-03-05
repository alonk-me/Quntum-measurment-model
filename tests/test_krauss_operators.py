"""Tests for Kraus operator simulation in krauss_operators/."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "krauss_operators"))

from datatypes import InitialState, TrajectoryResult
from krauss_operators_simulation import (
    run_trajectory,
    simulate_Q_distribution,
    eq14_pdf,
    fit_eq14,
)


class TestInitialState:
    """Tests for the InitialState dataclass."""

    def test_normalized_state_accepted(self):
        state = InitialState(alpha=1.0 / np.sqrt(2), beta=1.0 / np.sqrt(2))
        assert state.alpha == pytest.approx(1.0 / np.sqrt(2))

    def test_unnormalized_raises(self):
        with pytest.raises(ValueError, match="not normalized"):
            InitialState(alpha=1.0, beta=1.0)

    def test_zero_state_raises(self):
        # alpha=0, beta=0 should also fail (zero norm)
        with pytest.raises(ValueError):
            InitialState(alpha=0.0, beta=0.0)

    def test_from_unnormalized_normalizes(self):
        state = InitialState.from_unnormalized(3.0, 4.0)
        a, b = state.normalized()
        assert abs(a**2 + b**2 - 1.0) < 1e-12

    def test_from_unnormalized_zero_raises(self):
        with pytest.raises(ValueError, match="zero state"):
            InitialState.from_unnormalized(0.0, 0.0)

    def test_normalized_returns_original_amplitudes(self):
        a = 0.6
        b = 0.8
        state = InitialState(alpha=a, beta=b)
        out_a, out_b = state.normalized()
        assert out_a == pytest.approx(a)
        assert out_b == pytest.approx(b)


class TestRunTrajectory:
    """Tests for run_trajectory function."""

    def test_returns_trajectory_result(self):
        rng = np.random.default_rng(42)
        result = run_trajectory(N=10, epsilon=0.1, rng=rng)
        assert isinstance(result, TrajectoryResult)

    def test_outcome_lengths(self):
        rng = np.random.default_rng(0)
        N = 20
        result = run_trajectory(N=N, epsilon=0.05, rng=rng)
        assert len(result.outcomes) == N
        assert len(result.z_averages) == N

    def test_outcomes_binary(self):
        rng = np.random.default_rng(7)
        result = run_trajectory(N=50, epsilon=0.05, rng=rng)
        assert set(result.outcomes).issubset({-1, 1})

    def test_state_norm_preserved(self):
        """Born probabilities should sum to 1 (implicit norm check)."""
        rng = np.random.default_rng(1)
        # All z_averages should be in [-1, 1]
        result = run_trajectory(N=100, epsilon=0.1, rng=rng)
        z_avgs = np.array(result.z_averages)
        assert np.all(z_avgs >= -1.0 - 1e-10)
        assert np.all(z_avgs <= 1.0 + 1e-10)

    def test_z_before_in_range(self):
        rng = np.random.default_rng(2)
        result = run_trajectory(N=50, epsilon=0.1, rng=rng)
        assert np.all(np.array(result.zs_before) >= -1.0 - 1e-10)
        assert np.all(np.array(result.zs_before) <= 1.0 + 1e-10)

    def test_z_after_in_range(self):
        rng = np.random.default_rng(3)
        result = run_trajectory(N=50, epsilon=0.1, rng=rng)
        assert np.all(np.array(result.zs_after) >= -1.0 - 1e-10)
        assert np.all(np.array(result.zs_after) <= 1.0 + 1e-10)

    def test_q_is_finite(self):
        rng = np.random.default_rng(4)
        result = run_trajectory(N=100, epsilon=0.1, rng=rng)
        assert np.isfinite(result.Q)

    def test_with_initial_state(self):
        rng = np.random.default_rng(5)
        state = InitialState.from_unnormalized(1.0, 0.0)
        result = run_trajectory(N=20, epsilon=0.05, initial_state=state, rng=rng)
        assert len(result.outcomes) == 20


class TestSimulateQDistribution:
    """Tests for simulate_Q_distribution."""

    def test_returns_array_of_correct_length(self):
        Q_vals = simulate_Q_distribution(num_traj=20, N=100, epsilon=0.1, seed=42)
        assert isinstance(Q_vals, np.ndarray)
        assert len(Q_vals) == 20

    def test_all_finite(self):
        Q_vals = simulate_Q_distribution(num_traj=10, N=50, epsilon=0.1, seed=0)
        assert np.all(np.isfinite(Q_vals))

    def test_reproducible_with_seed(self):
        Q1 = simulate_Q_distribution(num_traj=5, N=30, epsilon=0.1, seed=99)
        Q2 = simulate_Q_distribution(num_traj=5, N=30, epsilon=0.1, seed=99)
        assert np.allclose(Q1, Q2)


class TestEq14Pdf:
    """Tests for eq14_pdf probability density function."""

    def test_nonnegative(self):
        x = np.linspace(0.01, 5.0, 100)
        y = eq14_pdf(x, theta=1.0)
        assert np.all(y >= 0)

    def test_zero_theta_returns_zeros(self):
        x = np.array([1.0, 2.0, 3.0])
        y = eq14_pdf(x, theta=0.0)
        assert np.all(y == 0)

    def test_negative_theta_returns_zeros(self):
        x = np.array([1.0, 2.0])
        y = eq14_pdf(x, theta=-1.0)
        assert np.all(y == 0)


class TestFitEq14:
    """Tests for fit_eq14 parameter estimation."""

    def test_returns_two_floats(self):
        Q_vals = simulate_Q_distribution(num_traj=200, N=100, epsilon=0.1, seed=42)
        theta_hat, theta_err = fit_eq14(Q_vals)
        assert isinstance(theta_hat, float)
        assert isinstance(theta_err, float)

    def test_theta_positive(self):
        Q_vals = simulate_Q_distribution(num_traj=200, N=100, epsilon=0.1, seed=7)
        theta_hat, _ = fit_eq14(Q_vals)
        assert theta_hat > 0
