#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from quantum_measurement.backends import is_cupy_available
from quantum_measurement.utilities.gpu_utils import estimate_trajectory_batch_size
from benchmark_gpu_speedup import benchmark_once, build_simulator


def classify_region(gamma: float) -> str:
    g = gamma / 4.0
    if g <= 0.2:
        return "weak"
    if g <= 1.2:
        return "critical"
    return "strong"


def load_completed(csv_path: Path) -> set[tuple[int, int, int, int, float, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    completed: set[tuple[int, int, int, int, float, str]] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(
                (
                    int(row["L"]),
                    int(row["n_steps"]),
                    int(row["n_trajectories"]),
                    int(row["batch_size"]),
                    float(row["gamma"]),
                    str(row["device"]),
                )
            )
    return completed


def append_row(csv_path: Path, row: dict[str, object], header: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-run GPU benchmark campaign with resume support")
    parser.add_argument("--simulator", choices=["l_qubit", "two_qubit", "sse"], default="l_qubit")
    parser.add_argument("--L-values", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--n-steps-values", type=int, nargs="+", default=[10000, 30000, 100000])
    parser.add_argument("--n-trajectories-values", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 96, 128, 192])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.4, 4.0, 8.0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=float, default=1.0, help="Simulator total-time used to derive epsilon from gamma")
    parser.add_argument("--usage-fraction", type=float, default=0.60)
    parser.add_argument("--max-vram-gb", type=float, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.csv or (Path("results/test_scan") / f"long_gpu_benchmark_{stamp}.csv")

    devices = ["cpu"]
    if is_cupy_available():
        devices.append("gpu")

    header = [
        "timestamp",
        "simulator",
        "device",
        "L",
        "n_steps",
        "n_trajectories",
        "batch_size",
        "gamma",
        "g",
        "region",
        "T",
        "epsilon",
        "repeats",
        "throughput_traj_per_sec",
        "std_traj_per_sec",
        "campaign_runtime_sec",
    ]

    completed = set()
    if not args.no_resume:
        completed = load_completed(csv_path)

    combos = list(
        itertools.product(
            args.L_values,
            args.n_steps_values,
            args.n_trajectories_values,
            args.gammas,
            devices,
        )
    )

    print(f"Running long benchmark campaign -> {csv_path}")
    print(f"Total high-level tuples (without batch axis): {len(combos)}")
    print(f"Resume enabled: {not args.no_resume}, completed rows loaded: {len(completed)}")

    t0_campaign = time.perf_counter()
    n_done = 0
    n_skipped = 0

    for L, n_steps, n_trajectories, gamma, device in combos:
        epsilon = float(np.sqrt(max(gamma * args.T / max(n_steps, 1), 0.0)))
        region = classify_region(gamma)

        max_batch_gpu = None
        if device == "gpu":
            max_batch_gpu = estimate_trajectory_batch_size(
                L=L,
                max_vram_gb=args.max_vram_gb,
                usage_fraction=args.usage_fraction,
            )

        for batch_size in args.batch_sizes:
            if device == "gpu" and max_batch_gpu is not None and batch_size > max_batch_gpu:
                print(
                    f"SKIP (VRAM guard) L={L} n_steps={n_steps} n_traj={n_trajectories} "
                    f"batch={batch_size} > max_batch_gpu={max_batch_gpu}"
                )
                n_skipped += 1
                continue

            key = (int(L), int(n_steps), int(n_trajectories), int(batch_size), float(gamma), str(device))
            if key in completed:
                n_skipped += 1
                continue

            # Warmup
            sim = build_simulator(
                args.simulator,
                device,
                n_steps,
                L,
                seed=args.seed,
                epsilon=epsilon,
                T=args.T,
            )
            _ = benchmark_once(sim, min(16, n_trajectories), min(batch_size, 16))

            throughputs: list[float] = []
            for rep in range(args.repeats):
                sim = build_simulator(
                    args.simulator,
                    device,
                    n_steps,
                    L,
                    seed=args.seed + rep,
                    epsilon=epsilon,
                    T=args.T,
                )
                throughputs.append(benchmark_once(sim, n_trajectories, batch_size))

            row = {
                "timestamp": datetime.now().isoformat(),
                "simulator": args.simulator,
                "device": device,
                "L": int(L),
                "n_steps": int(n_steps),
                "n_trajectories": int(n_trajectories),
                "batch_size": int(batch_size),
                "gamma": float(gamma),
                "g": float(gamma / 4.0),
                "region": region,
                "T": float(args.T),
                "epsilon": float(epsilon),
                "repeats": int(args.repeats),
                "throughput_traj_per_sec": float(np.mean(throughputs)),
                "std_traj_per_sec": float(np.std(throughputs)),
                "campaign_runtime_sec": float(time.perf_counter() - t0_campaign),
            }
            append_row(csv_path, row, header)
            n_done += 1
            print(
                f"DONE device={device} L={L} n_steps={n_steps} n_traj={n_trajectories} "
                f"batch={batch_size} gamma={gamma:.3g} tps={row['throughput_traj_per_sec']:.3f}"
            )

    elapsed = time.perf_counter() - t0_campaign
    print("\nCampaign complete")
    print(f"  csv: {csv_path}")
    print(f"  rows written: {n_done}")
    print(f"  rows skipped: {n_skipped}")
    print(f"  elapsed_sec: {elapsed:.2f}")


if __name__ == "__main__":
    main()
