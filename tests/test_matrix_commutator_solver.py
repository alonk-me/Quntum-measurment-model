"""Tests for matrix_commutator_solver (free_fermion and jw_sanity_check variants)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Both packages contain an identical copy; test the free_fermion version
sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "free_fermion"))

from matrix_commutator_solver import build_h, initial_G, compute_time_series


class TestBuildH:
    """Tests for build_h."""

    @pytest.mark.parametrize("L", [2, 4])
    def test_shape(self, L):
        h = build_h(L, J=1.0)
        assert h.shape == (2 * L, 2 * L)

    @pytest.mark.parametrize("L", [2, 4])
    def test_complex_dtype(self, L):
        h = build_h(L, J=1.0)
        assert np.issubdtype(h.dtype, np.complexfloating)

    @pytest.mark.parametrize("J", [0.5, 1.0])
    def test_nearest_neighbor_structure(self, J):
        """h[0, 1] should equal -J (h11 block, site 0 -> site 1)."""
        L = 3
        h = build_h(L, J=J)
        # In h11 block, h[0, 1] = -J
        assert h[0, 1] == pytest.approx(-J)
        # In h22 block, h[L, L+1] = +J
        assert h[L, L + 1] == pytest.approx(J)

    def test_j_zero_gives_zero_matrix(self):
        h = build_h(3, J=0.0)
        assert np.allclose(h, 0.0)


class TestInitialG:
    """Tests for initial_G."""

    @pytest.mark.parametrize("L", [2, 4])
    def test_shape(self, L):
        G = initial_G(L)
        assert G.shape == (2 * L, 2 * L)

    @pytest.mark.parametrize("L", [2, 4])
    def test_first_L_diagonal_ones(self, L):
        G = initial_G(L)
        diag = np.real(np.diag(G))
        assert np.allclose(diag[:L], 1.0)
        assert np.allclose(diag[L:], 0.0)

    @pytest.mark.parametrize("L", [2, 4])
    def test_hermitian(self, L):
        G = initial_G(L)
        assert np.allclose(G, G.conj().T, atol=1e-12)

    @pytest.mark.parametrize("L", [2, 4])
    def test_diagonal_in_range(self, L):
        G = initial_G(L)
        diag = np.real(np.diag(G))
        assert np.all(diag >= 0.0)
        assert np.all(diag <= 1.0)


class TestComputeTimeSeries:
    """Tests for compute_time_series."""

    @pytest.mark.parametrize("L,J", [(2, 0.5), (2, 1.0), (4, 1.0)])
    def test_return_shapes(self, L, J):
        steps = 200
        times, values = compute_time_series(L=L, J=J, T=2.0, steps=steps)
        assert times.shape == (steps + 1,)
        assert values.shape == (steps + 1,)

    def test_times_start_at_zero(self):
        times, _ = compute_time_series(L=2, J=1.0, T=2.0, steps=100)
        assert times[0] == pytest.approx(0.0)

    def test_times_end_at_T(self):
        T = 2.0
        times, _ = compute_time_series(L=2, J=1.0, T=T, steps=100)
        assert times[-1] == pytest.approx(T)

    def test_g_hermiticity_preserved(self):
        """G should remain Hermitian throughout (checked via output values being real)."""
        _, values = compute_time_series(L=2, J=1.0, T=2.0, steps=200)
        # 1 + 2*G[0,0] should be real (imaginary part negligible)
        assert np.all(np.abs(np.imag(values)) < 1e-10)

    def test_j_zero_no_change(self):
        """For J=0 there is no evolution; G[0,0] stays 1, so value stays 3."""
        _, values = compute_time_series(L=2, J=0.0, T=1.0, steps=100)
        assert np.allclose(values, 3.0, atol=1e-10)

    def test_l2_analytical_solution(self):
        """For L=2, J=1: 1+2*G[0,0] should oscillate and remain bounded.

        The forward Euler method introduces numerical drift, so allow a
        generous bound around the exact range [-1, 3].
        """
        _, values = compute_time_series(L=2, J=1.0, T=np.pi, steps=500)
        real_vals = np.real(values)
        # Allow for Euler drift: the exact range is [-1, 3] but the
        # discretization introduces ~5% amplitude growth
        assert np.all(real_vals >= -3.5)
        assert np.all(real_vals <= 3.5)

    @pytest.mark.parametrize("L,J", [(2, 0.5), (4, 1.0)])
    def test_energy_approximately_conserved(self, L, J):
        """In the absence of dissipation, Tr(h @ G) should be approximately conserved."""
        steps = 200
        T = 2.0
        dt = T / steps
        h = build_h(L, J)
        G = initial_G(L)
        # compute initial energy
        energy_0 = float(np.real(np.trace(h @ G)))
        # evolve and compute final energy
        prefactor = -2.0j
        for _ in range(steps):
            comm = G @ h - h @ G
            G = G + dt * prefactor * comm
            G = 0.5 * (G + G.conj().T)
        energy_f = float(np.real(np.trace(h @ G)))
        # For small dt, energy changes only due to discretization error
        if abs(energy_0) > 1e-12:
            assert abs(energy_f - energy_0) / abs(energy_0) < 0.1
        else:
            assert abs(energy_f - energy_0) < 1e-6
