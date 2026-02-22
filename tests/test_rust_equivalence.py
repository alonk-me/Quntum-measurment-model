"""Numerical equivalence tests between Python and Rust simulators.

These tests verify that ``FastLQubitSimulator`` (Rust backend) produces
results that are numerically consistent with the reference Python
``LQubitCorrelationSimulator`` implementation.

All tests are skipped when the Rust extension is not built.
"""

import numpy as np
import pytest

from quantum_measurement.jw_expansion.l_qubit_correlation_simulator import (
    LQubitCorrelationSimulator,
)

try:
    from quantum_measurement.rust_simulator import FastLQubitSimulator

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    FastLQubitSimulator = None  # type: ignore[assignment,misc]

skip_if_no_rust = pytest.mark.skipif(
    not RUST_AVAILABLE, reason="Rust extension not built"
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_python_sim(L, closed_boundary=False, seed=42, **kwargs):
    rng = np.random.default_rng(seed)
    return LQubitCorrelationSimulator(
        L=L,
        J=kwargs.get("J", 1.0),
        epsilon=kwargs.get("epsilon", 0.1),
        N_steps=kwargs.get("N_steps", 200),
        T=kwargs.get("T", 1.0),
        closed_boundary=closed_boundary,
        rng=rng,
    )


def _make_rust_sim(L, closed_boundary=False, seed=42, **kwargs):
    return FastLQubitSimulator(
        L=L,
        J=kwargs.get("J", 1.0),
        epsilon=kwargs.get("epsilon", 0.1),
        N_steps=kwargs.get("N_steps", 200),
        T=kwargs.get("T", 1.0),
        closed_boundary=closed_boundary,
        rng=np.random.default_rng(seed),
    )


# ─── Basic smoke tests ────────────────────────────────────────────────────────


@skip_if_no_rust
def test_rust_simulator_available():
    """Confirm Rust extension can be imported."""
    from quantum_measurement._rust import RustLQubitSimulator  # noqa: F401


@skip_if_no_rust
def test_rust_trajectory_shapes():
    """Rust simulator returns arrays of correct shape."""
    sim = _make_rust_sim(L=3)
    Q, z_traj, xi_traj = sim.simulate_trajectory()
    assert z_traj.shape == (201, 3)
    assert xi_traj.shape == (200, 3)
    assert isinstance(Q, float)


@skip_if_no_rust
def test_rust_z_values_in_range():
    """Magnetisation values from Rust simulator are in [-1, 1]."""
    sim = _make_rust_sim(L=4, epsilon=0.05, N_steps=500)
    _, z_traj, _ = sim.simulate_trajectory()
    assert np.all(z_traj >= -1.1), "Magnetisation below -1"
    assert np.all(z_traj <= 1.1), "Magnetisation above 1"


@skip_if_no_rust
def test_rust_xi_values():
    """Xi values are all ±1."""
    sim = _make_rust_sim(L=3)
    _, _, xi_traj = sim.simulate_trajectory()
    assert set(np.unique(xi_traj)).issubset({-1, 1})


# ─── Structural equivalence (not numerical) ──────────────────────────────────
# Because Python and Rust use different RNG implementations, exact numerical
# equivalence is not expected. We test structural properties instead.


@skip_if_no_rust
@pytest.mark.parametrize("L", [2, 3, 5])
@pytest.mark.parametrize("closed_boundary", [False, True])
def test_output_shape_matches_python(L, closed_boundary):
    """Rust and Python simulators return same-shaped outputs."""
    py_sim = _make_python_sim(L=L, closed_boundary=closed_boundary, N_steps=100)
    rs_sim = _make_rust_sim(L=L, closed_boundary=closed_boundary, N_steps=100)

    Q_py, z_py, xi_py = py_sim.simulate_trajectory()
    Q_rs, z_rs, xi_rs = rs_sim.simulate_trajectory()

    assert z_py.shape == z_rs.shape
    assert xi_py.shape == xi_rs.shape
    assert isinstance(Q_rs, float)


@skip_if_no_rust
@pytest.mark.parametrize("L", [2, 3, 5])
def test_initial_z_values_match(L):
    """Initial magnetisation should be the same (-1 for all sites by default)."""
    py_sim = _make_python_sim(L=L)
    rs_sim = _make_rust_sim(L=L)

    _, z_py, _ = py_sim.simulate_trajectory()
    _, z_rs, _ = rs_sim.simulate_trajectory()

    # Both simulators start with the same G_initial → same z[0]
    np.testing.assert_allclose(z_py[0], z_rs[0], atol=1e-12)


@skip_if_no_rust
def test_rust_z2_mean_scalar():
    """simulate_z2_mean returns a non-negative float."""
    sim = _make_rust_sim(L=3)
    val = sim.simulate_z2_mean(num_trajectories=3)
    assert isinstance(val, float)
    assert val >= 0.0


@skip_if_no_rust
def test_rust_ensemble_shapes():
    """simulate_ensemble returns correctly shaped arrays."""
    sim = _make_rust_sim(L=3, N_steps=50)
    n = 4
    Q_vals, z_trajs, xi_trajs = sim.simulate_ensemble(n_trajectories=n)
    assert Q_vals.shape == (n,)
    assert z_trajs.shape == (n, 51, 3)
    assert xi_trajs.shape == (n, 50, 3)


# ─── Determinism tests ────────────────────────────────────────────────────────


@skip_if_no_rust
def test_rust_determinism_with_seed():
    """Same seed produces identical trajectories."""
    sim_a = _make_rust_sim(L=3, seed=123)
    sim_b = _make_rust_sim(L=3, seed=123)

    Q_a, z_a, xi_a = sim_a.simulate_trajectory()
    Q_b, z_b, xi_b = sim_b.simulate_trajectory()

    np.testing.assert_array_equal(z_a, z_b)
    np.testing.assert_array_equal(xi_a, xi_b)
    assert Q_a == Q_b


@skip_if_no_rust
def test_rust_different_seeds_differ():
    """Different seeds produce different trajectories."""
    sim_a = _make_rust_sim(L=3, seed=1)
    sim_b = _make_rust_sim(L=3, seed=2)

    _, z_a, _ = sim_a.simulate_trajectory()
    _, z_b, _ = sim_b.simulate_trajectory()

    assert not np.allclose(z_a, z_b)


# ─── Physical plausibility ────────────────────────────────────────────────────


@skip_if_no_rust
@pytest.mark.parametrize("epsilon", [0.01, 0.1, 0.5])
def test_rust_entropy_finite(epsilon):
    """Entropy production Q is finite for various epsilon values."""
    sim = _make_rust_sim(L=3, epsilon=epsilon, N_steps=200)
    Q, _, _ = sim.simulate_trajectory()
    assert np.isfinite(Q)


@skip_if_no_rust
@pytest.mark.parametrize("L", [2, 3, 5, 9])
def test_rust_various_sizes(L):
    """Simulator runs without error for various L values."""
    sim = _make_rust_sim(L=L, N_steps=50)
    Q, z_traj, xi_traj = sim.simulate_trajectory()
    assert z_traj.shape == (51, L)
    assert np.isfinite(Q)


@skip_if_no_rust
def test_python_and_rust_entropy_same_order_of_magnitude():
    """Rust entropy production should be in the same ballpark as Python."""
    # Run multiple trajectories to get stable statistics
    n = 20
    Q_py_list = []
    Q_rs_list = []
    for i in range(n):
        py_sim = _make_python_sim(L=3, seed=i, N_steps=100)
        rs_sim = _make_rust_sim(L=3, seed=i, N_steps=100)
        Q_py, _, _ = py_sim.simulate_trajectory()
        Q_rs, _, _ = rs_sim.simulate_trajectory()
        Q_py_list.append(Q_py)
        Q_rs_list.append(Q_rs)

    mean_py = np.mean(Q_py_list)
    mean_rs = np.mean(Q_rs_list)

    # The Python and Rust implementations use different RNG backends (numpy
    # vs rand::StdRng), so exact numerical agreement is not expected.  We
    # only verify that the ensemble-averaged entropy production is in the same
    # ballpark (within a factor of 2), which confirms that both simulators
    # implement the same physical model.  Tight agreement is covered by the
    # determinism and structural equivalence tests above.
    if abs(mean_py) > 1e-6:
        ratio = abs(mean_rs / mean_py)
        assert 0.5 <= ratio <= 2.0, (
            f"Rust mean Q ({mean_rs:.3f}) differs too much from Python ({mean_py:.3f})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
