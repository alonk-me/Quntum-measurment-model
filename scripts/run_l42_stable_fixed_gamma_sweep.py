#!/usr/bin/env python3
"""Run L=42 stable-integrator gamma sweep with regression guards.

This runner is intentionally narrow and reproducible for the first
production L=42 stable-path campaign.
"""

import csv
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantum_measurement.jw_expansion.l_qubit_correlation_simulator import LQubitCorrelationSimulator
from run_z2_scan import get_time_params


L_VALUE = 42
N_WORKERS = 16
N_TRAJECTORIES = 16
BASE_SEED = 42
USE_STABLE_INTEGRATOR = True
STABLE_PROJECTOR_ENFORCE = False

# Prescan-comparable grid within [0.1, 10.0].
GAMMA_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

OUTPUT_ROOT = Path("results/z2_scan/l42_stable_fixed_gamma_sweep")
OUTPUT_DIR = OUTPUT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "z2_sweep_L42.csv"

CSV_HEADER = [
    "timestamp",
    "L",
    "gamma",
    "g",
    "dt",
    "epsilon",
    "z2_mean",
    "z2_std",
    "n_finite",
    "n_total",
    "any_exact_one",
    "runtime_sec",
    "n_workers",
    "n_trajectories",
    "seed_base",
    "use_stable_integrator",
    "stable_projector_enforce",
    "z2_plus_one",
    "status",
]


def _trajectory_task(payload):
    gamma, seed = payload
    _, dt, _, _ = get_time_params(gamma)
    epsilon = float(np.sqrt(gamma * dt))
    sim = LQubitCorrelationSimulator(
        L=L_VALUE,
        J=1.0,
        epsilon=epsilon,
        N_steps=int(round(max(1.0 / gamma * 60.0, 100.0) / dt)),
        T=float(max(1.0 / gamma * 60.0, 100.0)),
        closed_boundary=True,
        device="cpu",
        use_stable_integrator=USE_STABLE_INTEGRATOR,
        stable_projector_enforce=STABLE_PROJECTOR_ENFORCE,
        rng=np.random.default_rng(seed),
    )
    z2 = float(sim.simulate_z2_mean())
    return z2


def _seeds_for_gamma(gamma_index):
    rng = np.random.default_rng(BASE_SEED + gamma_index)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=N_TRAJECTORIES)
    return [int(s) for s in seeds]


def _load_completed(csv_path):
    completed = set()
    if not csv_path.exists():
        return completed
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(float(row["gamma"]))
    return completed


def _append_row(csv_path, row):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main():
    completed = _load_completed(CSV_PATH)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"CSV: {CSV_PATH}")
    print(f"Gamma grid: {GAMMA_GRID}")
    print(f"Config: L={L_VALUE} workers={N_WORKERS} n_traj={N_TRAJECTORIES} stable={USE_STABLE_INTEGRATOR} snap={STABLE_PROJECTOR_ENFORCE}")

    with mp.Pool(processes=N_WORKERS) as pool:
        for idx, gamma in enumerate(GAMMA_GRID):
            if gamma in completed:
                print(f"Skip completed gamma={gamma}")
                continue

            _, dt, _, _ = get_time_params(gamma)
            epsilon = float(np.sqrt(gamma * dt))
            seeds = _seeds_for_gamma(idx)
            start = time.time()
            values = pool.map(_trajectory_task, [(gamma, seed) for seed in seeds])
            runtime = float(time.time() - start)

            arr = np.asarray(values, dtype=float)
            finite_mask = np.isfinite(arr)
            n_finite = int(np.sum(finite_mask))
            n_total = int(arr.size)
            finite_vals = arr[finite_mask]

            z2_mean = float(np.mean(finite_vals)) if n_finite > 0 else float("nan")
            z2_std = float(np.std(finite_vals, ddof=1)) if n_finite >= 2 else float("nan")
            any_exact_one = bool(np.any(arr == 1.0))

            regression = bool(any_exact_one or (n_finite < n_total) or (z2_std == 0.0))
            status = "REGRESSION_STOP" if regression else "OK"

            row = {
                "timestamp": datetime.now().isoformat(),
                "L": int(L_VALUE),
                "gamma": float(gamma),
                "g": float(gamma / 4.0),
                "dt": float(dt),
                "epsilon": float(epsilon),
                "z2_mean": float(z2_mean),
                "z2_std": float(z2_std),
                "n_finite": n_finite,
                "n_total": n_total,
                "any_exact_one": any_exact_one,
                "runtime_sec": runtime,
                "n_workers": int(N_WORKERS),
                "n_trajectories": int(N_TRAJECTORIES),
                "seed_base": int(BASE_SEED),
                "use_stable_integrator": bool(USE_STABLE_INTEGRATOR),
                "stable_projector_enforce": bool(STABLE_PROJECTOR_ENFORCE),
                "z2_plus_one": float(1.0 + z2_mean) if np.isfinite(z2_mean) else float("nan"),
                "status": status,
            }
            _append_row(CSV_PATH, row)

            print(
                f"gamma={gamma:.6g} dt={dt:.3e} eps={epsilon:.6f} "
                f"mean={z2_mean:.12f} std={z2_std:.12f} finite={n_finite}/{n_total} any_exact_one={any_exact_one}"
            )

            if regression:
                print("Regression signal detected; stopping sweep immediately.")
                raise SystemExit(2)

    print(f"Complete: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
