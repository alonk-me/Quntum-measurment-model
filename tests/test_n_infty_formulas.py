"""Tests for analytical steady-state formulas in jw_expansion/n_infty.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "jw_expansion"))

from n_infty import (
    delta,
    sign_im,
    term_value,
    sum_apbc,
    sum_pbc,
    integral_expr,
    small_g_limit,
    large_g_limit,
)


class TestDelta:
    """Tests for delta(k, g) = sqrt(1 - g^2 - 2i*g*cos(k))."""

    def test_returns_complex(self):
        d = delta(np.pi / 2, 0.5)
        assert np.iscomplex(d) or isinstance(d, (complex, np.complexfloating))

    def test_small_g_near_unity(self):
        # For g ~ 0, delta ~ sqrt(1 - 2i*g*cos(k)) ≈ 1 - i*g*cos(k)
        k = np.pi / 4
        g = 1e-4
        d = delta(k, g)
        # Real part of delta should be close to 1 for very small g
        assert abs(np.real(d) - 1.0) < 0.01

    def test_array_input(self):
        k_arr = np.linspace(0, np.pi, 10)
        d = delta(k_arr, 1.0)
        assert d.shape == (10,)

    def test_imaginary_part_sign(self):
        # For g > 0, k in (0, pi): cos(k) can be positive or negative
        # Im(delta) = Im(sqrt(1 - g^2 - 2ig*cos(k)))
        g = 0.5
        d_k0 = delta(0.01, g)  # cos(k) ~ 1, so Im part should be negative
        assert np.imag(d_k0) < 0


class TestSignIm:
    """Tests for sign_im helper."""

    def test_positive_imaginary(self):
        assert sign_im(1.0 + 2.0j) == 1

    def test_negative_imaginary(self):
        assert sign_im(1.0 - 2.0j) == -1

    def test_zero_imaginary(self):
        assert sign_im(3.0 + 0.0j) == 0

    def test_array_input(self):
        z = np.array([1 + 2j, 1 - 2j, 3 + 0j])
        result = sign_im(z)
        assert list(result) == [1, -1, 0]


class TestTermValue:
    """Tests for term_value summand."""

    def test_returns_float_in_range(self):
        for g in [0.1, 1.0, 5.0]:
            for k in [np.pi / 6, np.pi / 4, np.pi / 3]:
                tv = term_value(k, g)
                assert 0.0 <= tv <= 2.0, f"term_value({k}, {g}) = {tv} out of range"

    def test_nonnegative(self):
        g = 1.0
        k_vals = np.linspace(0.01, np.pi - 0.01, 20)
        for k in k_vals:
            assert term_value(k, g) >= 0.0


class TestSumApbc:
    """Tests for sum_apbc finite-size sum."""

    def test_odd_L_required(self):
        with pytest.raises(ValueError):
            sum_apbc(1.0, L=4)

    def test_L_at_least_3(self):
        with pytest.raises(ValueError):
            sum_apbc(1.0, L=1)

    def test_result_in_physical_range(self):
        for g in [0.1, 1.0, 5.0]:
            for L in [3, 5, 7]:
                n = sum_apbc(g, L)
                assert 0.0 <= n <= 0.5, f"sum_apbc({g}, {L}) = {n} out of range"

    def test_converges_to_integral(self):
        """Finite-size sum should approach integral as L grows."""
        g = 1.0
        n_exact = integral_expr(g)
        n_L51 = sum_apbc(g, L=51)
        assert abs(n_L51 - n_exact) < 0.05


class TestSumPbc:
    """Tests for sum_pbc finite-size sum."""

    def test_odd_L_required(self):
        with pytest.raises(ValueError):
            sum_pbc(1.0, L=4)

    def test_result_in_physical_range(self):
        for g in [0.1, 1.0, 5.0]:
            for L in [3, 5, 9]:
                n = sum_pbc(g, L)
                assert 0.0 <= n <= 0.5, f"sum_pbc({g}, {L}) = {n} out of range"

    def test_converges_to_integral(self):
        g = 1.0
        n_exact = integral_expr(g)
        n_L51 = sum_pbc(g, L=51)
        assert abs(n_L51 - n_exact) < 0.05


class TestIntegralExpr:
    """Tests for integral_expr thermodynamic limit."""

    def test_small_g_limit(self):
        """integral_expr should approach 0.5 - 1/pi for small g."""
        n = integral_expr(g=1e-3)
        expected = small_g_limit()
        assert abs(n - expected) < 1e-2

    def test_large_g_limit(self):
        """integral_expr should approach 1/(8g^2) for large g."""
        g = 20.0
        n = integral_expr(g=g)
        expected = large_g_limit(g)
        assert abs(n - expected) / expected < 0.05

    def test_zero_g_raises(self):
        with pytest.raises(ValueError):
            integral_expr(0.0)

    def test_negative_g_raises(self):
        with pytest.raises(ValueError):
            integral_expr(-1.0)

    def test_result_in_range(self):
        for g in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            n = integral_expr(g)
            assert 0.0 <= n <= 0.5


class TestSmallGLimit:
    """Tests for small_g_limit."""

    def test_expected_value(self):
        expected = 0.5 - 1.0 / np.pi
        assert abs(small_g_limit() - expected) < 1e-12

    def test_approximately_0182(self):
        assert abs(small_g_limit() - 0.18169) < 1e-4


class TestLargeGLimit:
    """Tests for large_g_limit."""

    def test_formula(self):
        g = 5.0
        assert abs(large_g_limit(g) - 1.0 / (8.0 * g**2)) < 1e-15

    def test_decreases_with_g(self):
        assert large_g_limit(10.0) < large_g_limit(5.0)

    def test_positive(self):
        assert large_g_limit(3.0) > 0.0


class TestOccupationsPhysicalRange:
    """Test that occupations stay in [0, 1] for physical parameters."""

    @pytest.mark.parametrize("g", [0.1, 0.5, 1.0, 5.0, 10.0])
    @pytest.mark.parametrize("L", [3, 7, 11])
    def test_apbc_physical(self, g, L):
        n = sum_apbc(g, L)
        assert 0.0 <= n <= 0.5

    @pytest.mark.parametrize("g", [0.1, 0.5, 1.0, 5.0, 10.0])
    @pytest.mark.parametrize("L", [3, 7, 11])
    def test_pbc_physical(self, g, L):
        n = sum_pbc(g, L)
        assert 0.0 <= n <= 0.5
