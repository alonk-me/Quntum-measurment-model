"""Tests for the experiments module (Kraus and SSE experiment runners)."""

import numpy as np
import pytest

from quantum_measurement.experiments import (
    KraussExperimentConfig,
    KraussExperimentResult,
    run_krauss_experiment,
    SSEExperimentConfig,
    SSEExperimentResult,
    run_sse_experiment,
)


# ---------------------------------------------------------------------------
# KraussExperimentConfig
# ---------------------------------------------------------------------------


class TestKraussExperimentConfig:
    def test_default_theta(self):
        cfg = KraussExperimentConfig(N=10_000, epsilon=0.01)
        assert cfg.theta == pytest.approx(1.0)

    def test_theta_computation(self):
        cfg = KraussExperimentConfig(N=200, epsilon=0.1)
        assert cfg.theta == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# run_krauss_experiment
# ---------------------------------------------------------------------------


class TestRunKraussExperiment:
    @pytest.fixture
    def small_cfg(self):
        return KraussExperimentConfig(num_traj=30, N=200, epsilon=0.1, seed=42)

    def test_returns_result_type(self, small_cfg):
        result = run_krauss_experiment(small_cfg)
        assert isinstance(result, KraussExperimentResult)

    def test_q_values_shape(self, small_cfg):
        result = run_krauss_experiment(small_cfg)
        assert result.Q_values.shape == (small_cfg.num_traj,)

    def test_q_values_finite(self, small_cfg):
        result = run_krauss_experiment(small_cfg)
        assert np.all(np.isfinite(result.Q_values))

    def test_mean_q_positive(self, small_cfg):
        result = run_krauss_experiment(small_cfg)
        assert result.mean_Q > 0

    def test_std_q_positive(self, small_cfg):
        result = run_krauss_experiment(small_cfg)
        assert result.std_Q >= 0

    def test_theta_matches_config(self, small_cfg):
        result = run_krauss_experiment(small_cfg)
        assert result.theta == pytest.approx(small_cfg.theta)

    def test_config_stored_on_result(self, small_cfg):
        result = run_krauss_experiment(small_cfg)
        assert result.config is small_cfg

    def test_reproducible_with_seed(self):
        cfg = KraussExperimentConfig(num_traj=20, N=100, epsilon=0.1, seed=7)
        r1 = run_krauss_experiment(cfg)
        r2 = run_krauss_experiment(cfg)
        assert np.allclose(r1.Q_values, r2.Q_values)


# ---------------------------------------------------------------------------
# SSEExperimentConfig
# ---------------------------------------------------------------------------


class TestSSEExperimentConfig:
    def test_default_theta(self):
        cfg = SSEExperimentConfig(N_steps=100, epsilon=0.1)
        assert cfg.theta == pytest.approx(1.0)

    def test_theta_computation(self):
        cfg = SSEExperimentConfig(N_steps=500, epsilon=0.2)
        assert cfg.theta == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# run_sse_experiment
# ---------------------------------------------------------------------------


class TestRunSSEExperiment:
    @pytest.fixture
    def small_cfg(self):
        return SSEExperimentConfig(n_trajectories=20, epsilon=0.1, N_steps=50, seed=42)

    def test_returns_result_type(self, small_cfg):
        result = run_sse_experiment(small_cfg)
        assert isinstance(result, SSEExperimentResult)

    def test_q_values_shape(self, small_cfg):
        result = run_sse_experiment(small_cfg)
        assert result.Q_values.shape == (small_cfg.n_trajectories,)

    def test_z_trajectories_shape(self, small_cfg):
        result = run_sse_experiment(small_cfg)
        assert result.z_trajectories.shape == (small_cfg.n_trajectories, small_cfg.N_steps + 1)

    def test_measurement_results_shape(self, small_cfg):
        result = run_sse_experiment(small_cfg)
        assert result.measurement_results.shape == (small_cfg.n_trajectories, small_cfg.N_steps)

    def test_q_values_finite(self, small_cfg):
        result = run_sse_experiment(small_cfg)
        assert np.all(np.isfinite(result.Q_values))

    def test_std_q_nonnegative(self, small_cfg):
        result = run_sse_experiment(small_cfg)
        assert result.std_Q >= 0

    def test_theoretical_mean_positive(self, small_cfg):
        result = run_sse_experiment(small_cfg)
        assert result.theoretical_mean_Q > 0

    def test_config_stored_on_result(self, small_cfg):
        result = run_sse_experiment(small_cfg)
        assert result.config is small_cfg

    def test_reproducible_with_seed(self):
        cfg = SSEExperimentConfig(n_trajectories=10, epsilon=0.1, N_steps=30, seed=99)
        r1 = run_sse_experiment(cfg)
        r2 = run_sse_experiment(cfg)
        assert np.allclose(r1.Q_values, r2.Q_values)
