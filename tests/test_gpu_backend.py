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
from quantum_measurement.jw_expansion.l_qubit_correlation_simulator import LQubitCorrelationSimulator
from quantum_measurement.jw_expansion.non_hermitian_hat import NonHermitianHatSimulator
from quantum_measurement.sse_simulation.sse import SSEWavefunctionSimulator


class TestBackendFactory:
    def test_cpu_backend_selected(self):
        backend = get_backend("cpu")
        assert backend.name == "numpy"
        assert backend.is_gpu is False

    def test_cpu_backend_seed_reproducible(self):
        b1 = get_backend("cpu", seed=123)
        b2 = get_backend("cpu", seed=123)
        x1 = b1.asnumpy(b1.standard_normal((4,)))
        x2 = b2.asnumpy(b2.standard_normal((4,)))
        assert np.allclose(x1, x2)

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


class TestBackendRng:
    def test_choice_pm1_domain_cpu(self):
        backend = get_backend("cpu", seed=7)
        xi = backend.asnumpy(backend.choice_pm1((100,)))
        assert set(np.unique(xi)).issubset({-1, 1})

    def test_reseed_cpu_changes_sequence_deterministically(self):
        backend = get_backend("cpu", seed=99)
        first = backend.asnumpy(backend.random((5,)))
        backend.seed(99)
        second = backend.asnumpy(backend.random((5,)))
        assert np.allclose(first, second)

    @pytest.mark.gpu
    @pytest.mark.skipif(not is_cupy_available(), reason="CuPy/CUDA unavailable")
    def test_gpu_backend_seed_reproducible(self):
        b1 = get_backend("gpu", seed=321)
        b2 = get_backend("gpu", seed=321)
        x1 = b1.asnumpy(b1.standard_normal((4,)))
        x2 = b2.asnumpy(b2.standard_normal((4,)))
        assert np.allclose(x1, x2)

    @pytest.mark.gpu
    @pytest.mark.skipif(not is_cupy_available(), reason="CuPy/CUDA unavailable")
    def test_choice_pm1_domain_gpu(self):
        backend = get_backend("gpu", seed=11)
        xi = backend.asnumpy(backend.choice_pm1((100,)))
        assert set(np.unique(xi)).issubset({-1, 1})


class TestBackendOptimizationHooks:
    def test_cpu_backend_optimization_hooks_smoke(self):
        backend = get_backend("cpu", seed=1)
        w = backend.get_workspace("cpu_test", (2, 2), complex)
        assert w.shape == (2, 2)

        G = backend.array(np.eye(2, dtype=complex))[None, :, :]
        h = backend.array(np.array([[0, 1], [1, 0]], dtype=complex))
        out = backend.batched_commutator_update(G, h, dt=0.01)
        assert out.shape == (1, 2, 2)

        sym = backend.symmetrize_clip_diag_inplace(out.copy())
        assert sym.shape == out.shape

        stats = backend.memory_pool_stats()
        assert "workspace_entries" in stats

    @pytest.mark.gpu
    @pytest.mark.skipif(not is_cupy_available(), reason="CuPy/CUDA unavailable")
    def test_gpu_workspace_reuse_and_pool_stats(self):
        backend = get_backend("gpu", seed=1)
        w1 = backend.get_workspace("gpu_test", (4, 4), complex)
        w2 = backend.get_workspace("gpu_test", (4, 4), complex)

        # Same key/shape should reuse cached workspace allocation.
        assert id(w1) == id(w2)

        stats = backend.memory_pool_stats()
        assert "used_bytes" in stats
        assert "total_bytes" in stats
        assert stats["workspace_entries"] >= 1


class TestSimulatorDeviceIntegration:
    def test_two_qubit_accepts_device_cpu(self):
        sim = TwoQubitCorrelationSimulator(N_steps=5, device="cpu", rng=np.random.default_rng(1))
        Q, z, xi = sim.simulate_trajectory()
        assert np.isfinite(Q)
        assert z.shape == (6, 2)
        assert xi.shape == (5, 2)

    def test_two_qubit_batch_cpu_shapes(self):
        sim = TwoQubitCorrelationSimulator(N_steps=5, device="cpu", rng=np.random.default_rng(1))
        Q, z, xi = sim.simulate_trajectory_batch(3)
        assert Q.shape == (3,)
        assert z.shape == (3, 6, 2)
        assert xi.shape == (3, 5, 2)

    @pytest.mark.gpu
    @pytest.mark.skipif(not is_cupy_available(), reason="CuPy/CUDA unavailable")
    def test_two_qubit_runs_on_gpu(self):
        sim = TwoQubitCorrelationSimulator(N_steps=5, device="gpu", rng=np.random.default_rng(1))
        Q, z, xi = sim.simulate_trajectory()
        assert np.isfinite(Q)
        assert z.shape == (6, 2)
        assert xi.shape == (5, 2)

    @pytest.mark.gpu
    @pytest.mark.skipif(not is_cupy_available(), reason="CuPy/CUDA unavailable")
    def test_two_qubit_batch_gpu_shapes(self):
        sim = TwoQubitCorrelationSimulator(N_steps=5, device="gpu", rng=np.random.default_rng(1))
        Q, z, xi = sim.simulate_trajectory_batch(3)
        assert Q.shape == (3,)
        assert z.shape == (3, 6, 2)
        assert xi.shape == (3, 5, 2)


@pytest.mark.gpu
@pytest.mark.skipif(not is_cupy_available(), reason="CuPy/CUDA unavailable")
class TestGpuBatchParity:
    def test_two_qubit_gpu_cpu_batch_parity(self):
        n_steps = 8
        n_batch = 3
        xi = np.random.default_rng(123).choice([-1, 1], size=(n_batch, n_steps, 2)).astype(np.int8)

        sim_cpu = TwoQubitCorrelationSimulator(N_steps=n_steps, device="cpu")
        sim_gpu = TwoQubitCorrelationSimulator(N_steps=n_steps, device="gpu")

        q_cpu, z_cpu, xi_cpu = sim_cpu.simulate_trajectory_batch(n_batch, xi_batch=xi)
        q_gpu, z_gpu, xi_gpu = sim_gpu.simulate_trajectory_batch(n_batch, xi_batch=xi)

        assert np.allclose(q_gpu, q_cpu, atol=1e-10)
        assert np.allclose(z_gpu, z_cpu, atol=1e-10)
        assert np.array_equal(xi_gpu, xi_cpu)

    def test_sse_gpu_cpu_batch_parity(self):
        n_steps = 12
        n_batch = 4
        xi = np.random.default_rng(321).choice([-1, 1], size=(n_batch, n_steps)).astype(np.int8)

        sim_cpu = SSEWavefunctionSimulator(N_steps=n_steps, device="cpu")
        sim_gpu = SSEWavefunctionSimulator(N_steps=n_steps, device="gpu")

        q_cpu, z_cpu, xi_cpu = sim_cpu.simulate_trajectory_batch(n_batch, xi_batch=xi)
        q_gpu, z_gpu, xi_gpu = sim_gpu.simulate_trajectory_batch(n_batch, xi_batch=xi)

        assert np.allclose(q_gpu, q_cpu, atol=1e-10)
        assert np.allclose(z_gpu, z_cpu, atol=1e-10)
        assert np.array_equal(xi_gpu, xi_cpu)

    def test_lqubit_gpu_cpu_batch_parity(self):
        n_steps = 8
        n_batch = 2
        L = 3
        xi = np.random.default_rng(999).choice([-1, 1], size=(n_batch, n_steps, L)).astype(np.int8)

        sim_cpu = LQubitCorrelationSimulator(L=L, N_steps=n_steps, device="cpu")
        sim_gpu = LQubitCorrelationSimulator(L=L, N_steps=n_steps, device="gpu")

        q_cpu, z_cpu, xi_cpu = sim_cpu.simulate_trajectory_batch(n_batch, xi_batch=xi)
        q_gpu, z_gpu, xi_gpu = sim_gpu.simulate_trajectory_batch(n_batch, xi_batch=xi)

        assert np.allclose(q_gpu, q_cpu, atol=1e-10)
        assert np.allclose(z_gpu, z_cpu, atol=1e-10)
        assert np.array_equal(xi_gpu, xi_cpu)

    def test_nonhermitian_gpu_cpu_batch_parity(self):
        n_batch = 2
        sim_cpu = NonHermitianHatSimulator(L=3, N_steps=20, dt=0.01, gamma=0.3, device="cpu")
        sim_gpu = NonHermitianHatSimulator(L=3, N_steps=20, dt=0.01, gamma=0.3, device="gpu")

        q_cpu, n_cpu, g_cpu = sim_cpu.simulate_trajectory_batch(n_batch, return_G_final=True)
        q_gpu, n_gpu, g_gpu = sim_gpu.simulate_trajectory_batch(n_batch, return_G_final=True)

        assert np.allclose(q_gpu, q_cpu, atol=1e-10)
        assert np.allclose(n_gpu, n_cpu, atol=1e-10)
        assert np.allclose(g_gpu, g_cpu, atol=1e-10)
