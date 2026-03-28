"""Focused regression tests for stable-integrator projector-snap defaults."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "jw_expansion"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from l_qubit_correlation_simulator import LQubitCorrelationSimulator
from run_z2_scan import get_time_params


def _make_stable_sim(L: int, gamma: float, seed: int, enforce: bool = False) -> LQubitCorrelationSimulator:
    _, dt, _, _ = get_time_params(gamma)
    n_steps = 500
    epsilon = float(np.sqrt(gamma * dt))
    T = float(dt * n_steps)
    sim = LQubitCorrelationSimulator(
        L=L,
        J=1.0,
        epsilon=epsilon,
        N_steps=n_steps,
        T=T,
        closed_boundary=True,
        device="cpu",
        use_stable_integrator=True,
        stable_projector_enforce=enforce,
        rng=np.random.default_rng(seed),
    )
    if hasattr(sim.backend, "seed"):
        sim.backend.seed(seed)
    return sim


def test_stable_mode_enforce_default_is_false():
    sim = LQubitCorrelationSimulator(
        L=8,
        J=1.0,
        epsilon=0.05,
        N_steps=20,
        T=0.02,
        closed_boundary=True,
        device="cpu",
        use_stable_integrator=True,
        rng=np.random.default_rng(1),
    )
    assert sim.stable_projector_enforce is False


def test_eigenvalues_in_interior_stable_mode():
    sim = _make_stable_sim(L=16, gamma=2.0, seed=42, enforce=False)
    G = sim.backend.copy(sim.G_initial)

    for _ in range(500):
        G, _ = sim._stable_step_single(G)

    g_np = np.asarray(sim.backend.asnumpy(G), dtype=complex)
    eigvals = np.linalg.eigvalsh(0.5 * (g_np + g_np.conj().T))
    assert np.sum((eigvals > 1e-3) & (eigvals < 1.0 - 1e-3)) >= 1


@pytest.mark.parametrize("L", [8, 16, 32])
@pytest.mark.parametrize("gamma", [0.5, 2.0, 5.0])
def test_z2_non_degenerate_stable_mode(L, gamma):
    z2_vals = []
    for seed in [11, 12, 13, 14]:
        sim = _make_stable_sim(L=L, gamma=gamma, seed=seed, enforce=False)
        z2_vals.append(float(sim.simulate_z2_mean()))

    z2_vals = np.asarray(z2_vals, dtype=float)
    assert np.all(np.isfinite(z2_vals))
    assert float(np.std(z2_vals, ddof=1)) > 0.0


def test_warn_on_snap_with_stable_integrator():
    with pytest.warns(UserWarning, match="projector snap is incompatible with stable integrator"):
        _ = LQubitCorrelationSimulator(
            L=8,
            J=1.0,
            epsilon=0.05,
            N_steps=20,
            T=0.02,
            closed_boundary=True,
            device="cpu",
            use_stable_integrator=True,
            stable_projector_enforce=True,
            rng=np.random.default_rng(2),
        )


def test_bdg_enforcement_disabled_by_default(monkeypatch):
    sim = LQubitCorrelationSimulator(
        L=8,
        J=1.0,
        epsilon=0.05,
        N_steps=20,
        T=0.02,
        closed_boundary=True,
        device="cpu",
        use_stable_integrator=True,
        rng=np.random.default_rng(3),
    )

    calls = {"count": 0}
    original = sim._enforce_bdg

    def _counting(G):
        calls["count"] += 1
        return original(G)

    monkeypatch.setattr(sim, "_enforce_bdg", _counting)
    _ = sim._maybe_enforce_bdg(sim.backend.copy(sim.G_initial))
    assert calls["count"] == 0
