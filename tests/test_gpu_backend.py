"""Tests for backend abstraction and GPU helper behavior."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent / "quantum_measurement"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "jw_expansion"))

from quantum_measurement.backends import get_backend, is_cupy_available
from quantum_measurement.jw_expansion.two_qubit_correlation_simulator import TwoQubitCorrelationSimulator


class TestBackendFactory:
    def test_cpu_backend_selected(self):
        backend = get_backend("cpu")
        assert backend.name == "numpy"
        assert backend.is_gpu is False

    def test_gpu_falls_back_or_selects_cupy(self):
        backend = get_backend("gpu")
        if is_cupy_available():
            assert backend.name == "cupy"
            assert backend.is_gpu is True
        else:
            assert backend.name == "numpy"
            assert backend.is_gpu is False


class TestBackendMathParity:
    def test_numpy_backend_core_ops(self):
        backend = get_backend("cpu")
        a = backend.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
        b = backend.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

        out = backend.matmul(a, b)
        np_out = np.array([[2.0, 1.0], [4.0, 3.0]], dtype=complex)
        assert np.allclose(backend.asnumpy(out), np_out)

    @pytest.mark.gpu
    @pytest.mark.skipif(not is_cupy_available(), reason="CuPy/CUDA unavailable")
    def test_cupy_matches_numpy_for_matmul(self):
        b_np = get_backend("cpu")
        b_gpu = get_backend("gpu")

        a = np.array([[1.0, -0.5], [2.0, 3.0]], dtype=complex)
        c = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=complex)

        out_np = b_np.matmul(b_np.array(a), b_np.array(c))
        out_gpu = b_gpu.matmul(b_gpu.array(a), b_gpu.array(c))

        assert np.allclose(b_gpu.asnumpy(out_gpu), b_np.asnumpy(out_np), atol=1e-12)


class TestSimulatorDeviceIntegration:
    def test_two_qubit_accepts_device_cpu(self):
        sim = TwoQubitCorrelationSimulator(N_steps=5, device="cpu", rng=np.random.default_rng(1))
        Q, z, xi = sim.simulate_trajectory()
        assert np.isfinite(Q)
        assert z.shape == (6, 2)
        assert xi.shape == (5, 2)

    @pytest.mark.gpu
    @pytest.mark.skipif(not is_cupy_available(), reason="CuPy/CUDA unavailable")
    def test_two_qubit_runs_on_gpu(self):
        sim = TwoQubitCorrelationSimulator(N_steps=5, device="gpu", rng=np.random.default_rng(1))
        Q, z, xi = sim.simulate_trajectory()
        assert np.isfinite(Q)
        assert z.shape == (6, 2)
        assert xi.shape == (5, 2)
