"""Benchmark Rust vs Python simulator performance.

Usage::

    python scripts/benchmark_rust_vs_python.py

Outputs a markdown table saved to docs/RUST_PERFORMANCE.md.
"""

import time
import sys
from pathlib import Path

import numpy as np

# Make sure we can import from the repo
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from quantum_measurement.jw_expansion.l_qubit_correlation_simulator import (
    LQubitCorrelationSimulator,
)

try:
    from quantum_measurement.rust_simulator import FastLQubitSimulator

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("WARNING: Rust extension not available. Build with `maturin develop --release`.")


def _time_fn(fn, runs=3):
    """Return median wall-clock time of *fn()* over *runs* repetitions."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def benchmark_single(L, n_steps, runs=3):
    """Benchmark single trajectory for given L and n_steps."""
    common = dict(J=1.0, epsilon=0.1, T=1.0, N_steps=n_steps, closed_boundary=False)

    py_sim = LQubitCorrelationSimulator(L=L, **common, rng=np.random.default_rng(42))
    t_py = _time_fn(py_sim.simulate_trajectory, runs=runs)

    if RUST_AVAILABLE:
        rs_sim = FastLQubitSimulator(L=L, **common, rng=np.random.default_rng(42))
        t_rs = _time_fn(rs_sim.simulate_trajectory, runs=runs)
        speedup = t_py / t_rs if t_rs > 0 else float("inf")
    else:
        t_rs = float("nan")
        speedup = float("nan")

    return t_py, t_rs, speedup


def benchmark_ensemble(L, n_steps, n_traj=10, runs=3):
    """Benchmark ensemble simulation."""
    common = dict(J=1.0, epsilon=0.1, T=1.0, N_steps=n_steps, closed_boundary=False)

    py_sim = LQubitCorrelationSimulator(L=L, **common, rng=np.random.default_rng(42))
    t_py = _time_fn(lambda: py_sim.simulate_ensemble(n_traj), runs=runs)

    if RUST_AVAILABLE:
        rs_sim = FastLQubitSimulator(L=L, **common, rng=np.random.default_rng(42))
        t_rs = _time_fn(lambda: rs_sim.simulate_ensemble(n_traj), runs=runs)
        speedup = t_py / t_rs if t_rs > 0 else float("inf")
    else:
        t_rs = float("nan")
        speedup = float("nan")

    return t_py, t_rs, speedup


def main():
    L_values = [9, 17, 33]
    n_steps_values = [1000, 10_000]
    runs = 3

    print("=" * 70)
    print("Rust vs Python Benchmark")
    print("=" * 70)

    rows_single = []
    rows_ensemble = []

    # Single trajectory benchmarks
    print("\n--- Single trajectory ---")
    for L in L_values:
        for n_steps in n_steps_values:
            print(f"  L={L:3d}, N_steps={n_steps:6d} ... ", end="", flush=True)
            t_py, t_rs, sp = benchmark_single(L, n_steps, runs=runs)
            rows_single.append((L, n_steps, t_py, t_rs, sp))
            rust_str = f"{t_rs:.3f}s" if RUST_AVAILABLE else "N/A"
            sp_str = f"{sp:.1f}x" if RUST_AVAILABLE else "N/A"
            print(f"Python {t_py:.3f}s | Rust {rust_str} | Speedup {sp_str}")

    # Ensemble benchmarks (10 trajectories)
    print("\n--- Ensemble (10 trajectories) ---")
    for L in L_values:
        for n_steps in n_steps_values:
            print(f"  L={L:3d}, N_steps={n_steps:6d} ... ", end="", flush=True)
            t_py, t_rs, sp = benchmark_ensemble(L, n_steps, n_traj=10, runs=runs)
            rows_ensemble.append((L, n_steps, t_py, t_rs, sp))
            rust_str = f"{t_rs:.3f}s" if RUST_AVAILABLE else "N/A"
            sp_str = f"{sp:.1f}x" if RUST_AVAILABLE else "N/A"
            print(f"Python {t_py:.3f}s | Rust {rust_str} | Speedup {sp_str}")

    # Build markdown table
    lines = [
        "# Rust Extension Performance",
        "",
        "Benchmarks run with 3-repetition median wall-clock time.",
        f"Platform: Python 3.x, Rust release build.",
        "",
        "## Single Trajectory",
        "",
        "| L | N_steps | Python (s) | Rust (s) | Speedup |",
        "|---|---------|-----------|---------|---------|",
    ]
    for L, n, t_py, t_rs, sp in rows_single:
        rust_s = f"{t_rs:.4f}" if RUST_AVAILABLE else "N/A"
        sp_s = f"{sp:.1f}×" if RUST_AVAILABLE else "N/A"
        lines.append(f"| {L} | {n} | {t_py:.4f} | {rust_s} | {sp_s} |")

    lines += [
        "",
        "## Ensemble (10 trajectories)",
        "",
        "| L | N_steps | Python (s) | Rust (s) | Speedup |",
        "|---|---------|-----------|---------|---------|",
    ]
    for L, n, t_py, t_rs, sp in rows_ensemble:
        rust_s = f"{t_rs:.4f}" if RUST_AVAILABLE else "N/A"
        sp_s = f"{sp:.1f}×" if RUST_AVAILABLE else "N/A"
        lines.append(f"| {L} | {n} | {t_py:.4f} | {rust_s} | {sp_s} |")

    md = "\n".join(lines) + "\n"

    # Save to docs/
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    out_path = docs_dir / "RUST_PERFORMANCE.md"
    out_path.write_text(md)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
