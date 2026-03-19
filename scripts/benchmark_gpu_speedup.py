"""Benchmark trajectories/sec versus batch size.

This script benchmarks the new batched ensemble execution path for selected
simulators and reports throughput (trajectories per second) for CPU and GPU.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_measurement.backends import is_cupy_available
from quantum_measurement.jw_expansion.l_qubit_correlation_simulator import LQubitCorrelationSimulator
from quantum_measurement.jw_expansion.two_qubit_correlation_simulator import TwoQubitCorrelationSimulator
from quantum_measurement.sse_simulation.sse import SSEWavefunctionSimulator


def build_simulator(name: str, device: str, n_steps: int, L: int, seed: int):
    if name == "two_qubit":
        return TwoQubitCorrelationSimulator(
            J=1.0,
            epsilon=0.1,
            N_steps=n_steps,
            T=1.0,
            device=device,
            rng=np.random.default_rng(seed),
        )
    if name == "sse":
        return SSEWavefunctionSimulator(
            epsilon=0.1,
            N_steps=n_steps,
            J=0.0,
            device=device,
            rng=np.random.default_rng(seed),
        )
    if name == "l_qubit":
        return LQubitCorrelationSimulator(
            L=L,
            J=1.0,
            epsilon=0.1,
            N_steps=n_steps,
            T=1.0,
            device=device,
            rng=np.random.default_rng(seed),
        )
    raise ValueError(f"Unsupported simulator: {name}")


def benchmark_once(sim, n_trajectories: int, batch_size: int) -> float:
    start = time.perf_counter()
    sim.simulate_ensemble(n_trajectories=n_trajectories, progress=False, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    return n_trajectories / max(elapsed, 1e-12)


def run_benchmark(
    simulator_name: str,
    n_steps: int,
    n_trajectories: int,
    batch_sizes: list[int],
    L: int,
    repeats: int,
    csv_path: Path | None,
) -> None:
    devices = ["cpu"]
    if is_cupy_available():
        devices.append("gpu")

    rows: list[dict[str, float | int | str]] = []

    for device in devices:
        for batch_size in batch_sizes:
            # Warmup run to reduce first-run overhead noise.
            sim = build_simulator(simulator_name, device, n_steps, L, seed=42)
            _ = benchmark_once(sim, min(8, n_trajectories), min(batch_size, 8))

            throughputs = []
            for rep in range(repeats):
                sim = build_simulator(simulator_name, device, n_steps, L, seed=42 + rep)
                tps = benchmark_once(sim, n_trajectories, batch_size)
                throughputs.append(tps)

            mean_tps = float(np.mean(throughputs))
            std_tps = float(np.std(throughputs))
            rows.append(
                {
                    "simulator": simulator_name,
                    "device": device,
                    "batch_size": batch_size,
                    "n_trajectories": n_trajectories,
                    "n_steps": n_steps,
                    "throughput_traj_per_sec": mean_tps,
                    "std_traj_per_sec": std_tps,
                }
            )

    # Console report
    print("\nBenchmark results (trajectories/sec):")
    print("simulator | device | batch_size | throughput | std")
    print("-" * 62)
    for row in rows:
        print(
            f"{row['simulator']:>9} | {row['device']:>4} | {row['batch_size']:>10} | "
            f"{row['throughput_traj_per_sec']:>10.2f} | {row['std_traj_per_sec']:>6.2f}"
        )

    # Optional CSV export
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "simulator",
                    "device",
                    "batch_size",
                    "n_trajectories",
                    "n_steps",
                    "throughput_traj_per_sec",
                    "std_traj_per_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trajectories/sec vs batch size")
    parser.add_argument("--simulator", choices=["two_qubit", "sse", "l_qubit"], default="two_qubit")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--n-trajectories", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--L", type=int, default=8, help="Used only for l_qubit simulator")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--csv", type=Path, default=None, help="Optional output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        simulator_name=args.simulator,
        n_steps=args.n_steps,
        n_trajectories=args.n_trajectories,
        batch_sizes=args.batch_sizes,
        L=args.L,
        repeats=args.repeats,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
