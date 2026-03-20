"""Tests for the analysis module (distribution analysis tools)."""

import numpy as np
import pytest

from quantum_measurement.analysis import (
    QStatistics,
    compute_statistics,
    fit_arrow_of_time,
    plot_Q_distribution,
    plot_mean_Q_vs_theta,
)


# ---------------------------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    def test_returns_qstatistics(self):
        stats = compute_statistics([1.0, 2.0, 3.0])
        assert isinstance(stats, QStatistics)

    def test_mean_correct(self):
        stats = compute_statistics([1.0, 2.0, 3.0])
        assert stats.mean == pytest.approx(2.0)

    def test_std_correct(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_statistics(data)
        assert stats.std == pytest.approx(float(np.std(data, ddof=1)))

    def test_median_correct(self):
        data = [3.0, 1.0, 2.0]
        stats = compute_statistics(data)
        assert stats.median == pytest.approx(2.0)

    def test_n_samples(self):
        data = list(range(10))
        stats = compute_statistics(data)
        assert stats.n_samples == 10

    def test_single_element(self):
        stats = compute_statistics([42.0])
        assert stats.mean == pytest.approx(42.0)
        assert stats.std == 0.0

    def test_accepts_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        stats = compute_statistics(arr)
        assert stats.mean == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# fit_arrow_of_time
# ---------------------------------------------------------------------------


class TestFitArrowOfTime:
    def _generate_q_values(self, num_traj=300, N=500, epsilon=0.1, seed=42):
        from quantum_measurement.experiments import KraussExperimentConfig, run_krauss_experiment
        cfg = KraussExperimentConfig(num_traj=num_traj, N=N, epsilon=epsilon, seed=seed)
        result = run_krauss_experiment(cfg)
        return result.Q_values

    def test_returns_two_floats(self):
        Q_vals = self._generate_q_values()
        theta_hat, theta_err = fit_arrow_of_time(Q_vals)
        assert isinstance(theta_hat, float)
        assert isinstance(theta_err, float)

    def test_theta_positive(self):
        Q_vals = self._generate_q_values()
        theta_hat, _ = fit_arrow_of_time(Q_vals)
        assert theta_hat > 0

    def test_accepts_list(self):
        Q_vals = self._generate_q_values()
        theta_hat, _ = fit_arrow_of_time(list(Q_vals))
        assert theta_hat > 0


# ---------------------------------------------------------------------------
# plot_Q_distribution
# ---------------------------------------------------------------------------


class TestPlotQDistribution:
    def test_returns_axes(self):
        import matplotlib.pyplot as plt
        Q_vals = np.random.default_rng(0).normal(2.0, 1.0, 100)
        ax = plot_Q_distribution(Q_vals)
        assert ax is not None
        plt.close("all")

    def test_with_theta_hat(self):
        import matplotlib.pyplot as plt
        Q_vals = np.abs(np.random.default_rng(1).normal(1.0, 0.5, 100))
        ax = plot_Q_distribution(Q_vals, theta_hat=1.0)
        assert ax is not None
        plt.close("all")

    def test_saves_file(self, tmp_path):
        Q_vals = np.random.default_rng(2).normal(2.0, 1.0, 100)
        out = str(tmp_path / "dist.png")
        plot_Q_distribution(Q_vals, filename=out)
        import os
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# plot_mean_Q_vs_theta
# ---------------------------------------------------------------------------


class TestPlotMeanQVsTheta:
    def test_returns_axes(self):
        import matplotlib.pyplot as plt
        thetas = [0.5, 1.0, 2.0, 4.0]
        means = [0.6, 1.1, 2.1, 3.9]
        ax = plot_mean_Q_vs_theta(thetas, means)
        assert ax is not None
        plt.close("all")

    def test_saves_file(self, tmp_path):
        out = str(tmp_path / "mean_q.png")
        plot_mean_Q_vs_theta([1.0, 2.0], [1.1, 2.0], filename=out)
        import os
        assert os.path.exists(out)
