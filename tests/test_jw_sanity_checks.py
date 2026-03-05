"""Tests for Jordan-Wigner sanity checks (jw_sanity_check/jw_H_XY.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "quantum_measurement" / "jw_sanity_check"))

from jw_H_XY import wavefunction_magnetization, corr_magnetization


class TestWavefunctionMagnetization:
    """Tests for wavefunction_magnetization."""

    def test_return_shapes(self):
        steps = 50
        times, mag = wavefunction_magnetization(J=1.0, T=1.0, steps=steps)
        assert times.shape == (steps + 1,)
        assert mag.shape == (steps + 1,)

    def test_times_start_at_zero(self):
        times, _ = wavefunction_magnetization(J=1.0, T=1.0, steps=50)
        assert times[0] == pytest.approx(0.0)

    def test_magnetization_in_range(self):
        _, mag = wavefunction_magnetization(J=1.0, T=1.0, steps=50)
        assert np.all(mag >= -1.0 - 1e-10)
        assert np.all(mag <= 1.0 + 1e-10)

    def test_initial_state_00(self):
        """For |00⟩, first qubit starts in |0⟩: ⟨σ_z^(1)⟩ = +1."""
        _, mag = wavefunction_magnetization(J=1.0, T=1.0, steps=50, initial_state="00")
        assert mag[0] == pytest.approx(1.0, abs=1e-10)

    def test_initial_state_01(self):
        """For |01⟩, first qubit starts in |0⟩: ⟨σ_z^(1)⟩ = +1."""
        _, mag = wavefunction_magnetization(J=1.0, T=1.0, steps=50, initial_state="01")
        assert mag[0] == pytest.approx(1.0, abs=1e-10)

    def test_initial_state_10(self):
        """For |10⟩, first qubit is |1⟩: ⟨σ_z^(1)⟩ = -1."""
        _, mag = wavefunction_magnetization(J=1.0, T=1.0, steps=50, initial_state="10")
        assert mag[0] == pytest.approx(-1.0, abs=1e-10)

    def test_j_zero_no_evolution(self):
        """For J=0 there is no coupling, so initial magnetization is preserved."""
        _, mag = wavefunction_magnetization(J=0.0, T=1.0, steps=50, initial_state="01")
        assert np.allclose(mag, mag[0], atol=1e-10)


class TestCorrMagnetization:
    """Tests for corr_magnetization (JW correlation matrix method)."""

    def test_return_shapes(self):
        steps = 50
        times, mag = corr_magnetization(J=1.0, T=1.0, steps=steps)
        assert times.shape == (steps + 1,)
        assert mag.shape == (steps + 1,)

    def test_only_l2_supported(self):
        with pytest.raises(NotImplementedError):
            corr_magnetization(J=1.0, T=1.0, steps=10, L=3)

    def test_magnetization_is_real(self):
        _, mag = corr_magnetization(J=1.0, T=1.0, steps=50)
        assert np.all(np.isreal(mag))


class TestJWSanityAgreement:
    """Integration tests: both methods should be self-consistent."""

    def test_wavefunction_returns_finite_values(self):
        """wavefunction_magnetization should return finite, bounded values."""
        _, mag_wf = wavefunction_magnetization(J=1.0, T=1.0, steps=50, initial_state="01")
        assert np.all(np.isfinite(mag_wf))
        assert np.all(mag_wf >= -1.0 - 1e-10)
        assert np.all(mag_wf <= 1.0 + 1e-10)

    def test_corr_returns_finite_values(self):
        """corr_magnetization should return finite values."""
        _, mag_corr = corr_magnetization(J=1.0, T=1.0, steps=50, L=2, occ=[0, 1])
        assert np.all(np.isfinite(mag_corr))
