import csv

import numpy as np

from quantum_measurement.parallel import ParameterSweepExecutor
from quantum_measurement.parallel.sweep_executor import SweepTask


def _dummy_factory(L, gamma, backend_device, rng):
    return {
        "L": int(L),
        "gamma": float(gamma),
        "device": backend_device,
        "value": float(rng.random() + L + gamma),
    }


def _oom_then_cpu_factory(L, gamma, backend_device, rng):
    del rng
    if backend_device == "gpu":
        raise RuntimeError("CUDA out of memory")
    return {
        "L": int(L),
        "gamma": float(gamma),
        "device": backend_device,
        "value": float(L + gamma),
    }


class TestSweepExecutor:
    def test_sequential_writes_and_resume_skips_completed(self, tmp_path):
        csv_file = tmp_path / "sweep.csv"
        L_values = [2, 3]
        gamma_grid = [0.5, 1.0]

        executor = ParameterSweepExecutor(
            parallel_backend="sequential",
            n_workers=1,
            base_seed=123,
            verbose=False,
        )

        first = executor.run_sweep(
            L_values=L_values,
            gamma_grid=gamma_grid,
            simulator_factory=_dummy_factory,
            backend_device="cpu",
            output_csv=csv_file,
            resume=True,
            csv_header=["L", "gamma", "device", "value"],
        )
        assert len(first) == 4

        second = executor.run_sweep(
            L_values=L_values,
            gamma_grid=gamma_grid,
            simulator_factory=_dummy_factory,
            backend_device="cpu",
            output_csv=csv_file,
            resume=True,
            csv_header=["L", "gamma", "device", "value"],
        )
        assert len(second) == 0

        with csv_file.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4

    def test_multiprocessing_completes_full_grid(self, tmp_path):
        csv_file = tmp_path / "sweep_mp.csv"
        L_values = [2, 3]
        gamma_grid = [0.2, 0.4, 0.8]

        executor = ParameterSweepExecutor(
            parallel_backend="multiprocessing",
            n_workers=2,
            base_seed=999,
            verbose=False,
        )

        out = executor.run_sweep(
            L_values=L_values,
            gamma_grid=gamma_grid,
            simulator_factory=_dummy_factory,
            backend_device="cpu",
            output_csv=csv_file,
            resume=False,
            csv_header=["L", "gamma", "device", "value"],
        )

        assert len(out) == len(L_values) * len(gamma_grid)
        pairs = {(int(r["L"]), float(r["gamma"])) for r in out}
        assert len(pairs) == len(out)

        with csv_file.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(out)

    def test_seed_reproducibility_without_csv(self):
        executor_a = ParameterSweepExecutor(
            parallel_backend="sequential",
            n_workers=1,
            base_seed=17,
            verbose=False,
        )
        executor_b = ParameterSweepExecutor(
            parallel_backend="sequential",
            n_workers=1,
            base_seed=17,
            verbose=False,
        )

        a = executor_a.run_sweep(
            L_values=[2, 3],
            gamma_grid=[0.1, 0.2],
            simulator_factory=_dummy_factory,
            backend_device="cpu",
            output_csv=None,
            resume=False,
        )
        b = executor_b.run_sweep(
            L_values=[2, 3],
            gamma_grid=[0.1, 0.2],
            simulator_factory=_dummy_factory,
            backend_device="cpu",
            output_csv=None,
            resume=False,
        )

        a_vals = np.array([r["value"] for r in sorted(a, key=lambda x: (x["L"], x["gamma"]))])
        b_vals = np.array([r["value"] for r in sorted(b, key=lambda x: (x["L"], x["gamma"]))])
        assert np.allclose(a_vals, b_vals)

    def test_gpu_oom_fallback_retries_on_cpu(self):
        executor = ParameterSweepExecutor(
            parallel_backend="sequential",
            n_workers=1,
            base_seed=1,
            verbose=False,
        )
        out = executor.run_sweep(
            L_values=[9],
            gamma_grid=[1.0],
            simulator_factory=_oom_then_cpu_factory,
            backend_device="gpu",
            output_csv=None,
            resume=False,
        )
        assert len(out) == 1
        assert out[0]["device"] == "cpu"

    def test_cpu_worker_policy_l_aware_limits(self):
        executor = ParameterSweepExecutor(
            parallel_backend="multiprocessing",
            n_workers=None,
            base_seed=1,
            verbose=False,
        )
        small_tasks = [SweepTask(index=0, L=9, gamma=0.1), SweepTask(index=1, L=17, gamma=0.2)]
        large_tasks = [SweepTask(index=0, L=129, gamma=0.1)]
        small_workers = executor._effective_cpu_workers(small_tasks)
        large_workers = executor._effective_cpu_workers(large_tasks)
        assert small_workers >= large_workers
        assert large_workers >= 1
