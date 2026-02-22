#!/usr/bin/env python3
"""
Production Script: 1+<z^2> vs log(gamma/4) parameter sweep

This script runs a long parameter sweep for the L-qubit correlation simulator
and records the mean of z^2 over time and sites. Results are written
incrementally to CSV for safe resume and live monitoring.

Usage:
    python scripts/run_z2_scan.py

    # Resume an existing run
    python scripts/run_z2_scan.py --csv results/z2_scan/z2_data_20260214_123000.csv
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure local imports work when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))
from quantum_measurement.jw_expansion.l_qubit_correlation_simulator import (
    LQubitCorrelationSimulator,
)

# ============================================================================
# Configuration Parameters
# ============================================================================

J_GLOBAL = 1.0

# Gamma grid
GAMMA_GLOBAL_MIN = 1e-1
GAMMA_GLOBAL_MAX = 1e2
N_GAMMA_GLOBAL = 80

G_CRITICAL_CENTER = 1.0
G_CRITICAL_WIDTH = 0.4
N_G_CRITICAL = 120

# Time-step construction
T_MULTIPLIER = 100.0  # Increased from 20 to reduce noise near phase transition
T_MIN = 100.0  # Increased from 10 for better convergence
DT_RATIO = 1e-2
DT_MAX = 1e-3
N_STEPS_WARNING = 1e8

# L values (existing list capped at 128, no fallback)
L_VALUES_BASE = [9, 17, 33, 65, 129, 256]
L_MAX = 129

# Output paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "z2_scan"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_CSV = RESULTS_DIR / f"z2_data_{TIMESTAMP}.csv"

CSV_HEADER = [
    "timestamp",
    "L",
    "gamma",
    "g",
    "z2_mean",
    "z2_plus_one",
    "tau",
    "T",
    "dt",
    "N_steps",
    "epsilon",
    "runtime_sec",
]


# ============================================================================
# Grid and parameter helpers
# ============================================================================

def construct_gamma_grid():
    """Build combined gamma grid: global log-spaced + critical region."""
    gamma_global = np.logspace(
        np.log10(GAMMA_GLOBAL_MIN), np.log10(GAMMA_GLOBAL_MAX), N_GAMMA_GLOBAL
    )
    g_critical = np.linspace(
        G_CRITICAL_CENTER - G_CRITICAL_WIDTH,
        G_CRITICAL_CENTER + G_CRITICAL_WIDTH,
        N_G_CRITICAL,
    )
    gamma_critical = g_critical * 4 * J_GLOBAL

    gamma_combined = np.unique(np.concatenate([gamma_global, gamma_critical]))
    gamma_combined = gamma_combined[gamma_combined > 0]
    return gamma_combined


def get_time_params(gamma):
    """Return (T, dt, N_steps, tau) based on gamma."""
    tau = 1.0 / gamma
    T = max(tau * T_MULTIPLIER, T_MIN)
    dt = min(tau * DT_RATIO, DT_MAX)
    N_steps = int(round(T / dt))
    return T, dt, N_steps, tau


def resolve_L_values():
    """Filter L list to requested cap."""
    return [L for L in L_VALUES_BASE if L <= L_MAX]


# ============================================================================
# CSV management
# ============================================================================

def initialize_csv(csv_file):
    """Create CSV file with header."""
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
    print(f"Initialized CSV: {csv_file}")


def append_to_csv(csv_file, data_dict):
    """Append single row to CSV."""
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        row = [data_dict[key] for key in CSV_HEADER]
        writer.writerow(row)


def load_existing_results(csv_file):
    """Load existing CSV to avoid recomputing."""
    if not csv_file.exists():
        return set()

    completed = set()
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            L = int(row["L"])
            gamma = float(row["gamma"])
            completed.add((L, gamma))

    return completed


# ============================================================================
# Main sweep
# ============================================================================

def run_parameter_sweep(csv_file):
    print("=" * 80)
    print("1+<z^2> PARAMETER SWEEP")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV file: {csv_file}")
    print()

    L_values = resolve_L_values()
    gamma_grid = construct_gamma_grid()
    g_grid = gamma_grid / (4 * J_GLOBAL)

    print("Parameter grid:")
    print(f"  L values: {L_values}")
    print(f"  gamma range: [{gamma_grid.min():.3e}, {gamma_grid.max():.3e}]")
    print(f"  g range: [{g_grid.min():.3e}, {g_grid.max():.3e}]")
    print(f"  N_gamma points: {len(gamma_grid)}")
    print(f"  Total runs: {len(L_values) * len(gamma_grid)}")
    print()

    if not csv_file.exists():
        initialize_csv(csv_file)
        completed = set()
    else:
        completed = load_existing_results(csv_file)
        print(f"Resuming: {len(completed)} runs already completed")
        print()

    total_runs = len(L_values) * len(gamma_grid)
    run_count = 0

    for L in L_values:
        print(f"\n{'='*80}")
        print(f"SYSTEM SIZE: L = {L}")
        print(f"{'='*80}\n")

        for gamma in gamma_grid:
            run_count += 1

            if (L, gamma) in completed:
                continue

            g = gamma / (4 * J_GLOBAL)
            T, dt, N_steps, tau = get_time_params(gamma)
            epsilon = float(np.sqrt(gamma * dt))

            progress_pct = 100 * run_count / total_runs
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Run "
                f"{run_count}/{total_runs} ({progress_pct:.1f}%)"
            )
            print(f"  L={L:3d}, gamma={gamma:8.4f}, g={g:8.4f}")
            print(f"  T={T:.3e}, dt={dt:.3e}, N_steps={N_steps}, eps={epsilon:.3e}")

            if N_steps >= N_STEPS_WARNING:
                print(f"  WARNING: N_steps={N_steps} exceeds {N_STEPS_WARNING:.0e}")

            start_time = time.time()

            try:
                sim = LQubitCorrelationSimulator(
                    L=L,
                    J=J_GLOBAL,
                    epsilon=epsilon,
                    N_steps=N_steps,
                    T=T,
                    closed_boundary=True,
                )
                z2_mean = float(sim.simulate_z2_mean())
                z2_plus_one = 1.0 + z2_mean
                runtime = time.time() - start_time

                data_dict = {
                    "timestamp": datetime.now().isoformat(),
                    "L": L,
                    "gamma": float(gamma),
                    "g": float(g),
                    "z2_mean": z2_mean,
                    "z2_plus_one": z2_plus_one,
                    "tau": tau,
                    "T": T,
                    "dt": dt,
                    "N_steps": int(N_steps),
                    "epsilon": epsilon,
                    "runtime_sec": runtime,
                }
                append_to_csv(csv_file, data_dict)

                print(f"  z2_mean={z2_mean:.6f}, 1+z2={z2_plus_one:.6f}")
                print(f"  Time: {runtime:.1f}s")
                print("  SAVED to CSV")

            except Exception as exc:
                print(f"  ERROR: {exc}")
                import traceback

                traceback.print_exc()
                print("  Skipping this point...")

            print()

    print("=" * 80)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {csv_file}")
    print()


# ============================================================================
# Post-processing: Generate Plots
# ============================================================================

def generate_verification_plots(csv_file):
    """Generate plots from completed z2 scan data."""
    
    print("\nGenerating verification plots...")
    
    if not csv_file.exists():
        print("  ✗ No CSV file found, skipping plots")
        return
    
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["g", "z2_plus_one"])
    print(f"  ✓ Loaded {len(df)} data points from {csv_file.name}")
    
    if len(df) == 0:
        print("  ✗ No valid data points, skipping plots")
        return
    
    L_unique = sorted(df["L"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_unique)))
    
    # Plot 1: Log scale on g (gamma/4)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, L in enumerate(L_unique):
        df_L = df[df["L"] == L].sort_values("g")
        ax.plot(df_L["g"], df_L["z2_plus_one"], marker="o", markersize=4, 
                linewidth=1.2, color=colors[idx], label=f"L={L} ({len(df_L)} pts)")
    
    ax.axhline(1.25, color="orange", linestyle="--", linewidth=1.5, 
               label="1.25", alpha=0.7)
    ax.axhline(2.0, color="green", linestyle="--", linewidth=1.5, 
               label="2.0", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("g = γ/(4J)", fontweight="bold")
    ax.set_ylabel("1+<z²>", fontweight="bold")
    ax.set_title("1+<z²> vs γ/4 (log scale)", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    
    plot_file = csv_file.parent / f"{csv_file.stem}_logscale.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_file.name}")
    plt.close()
    
    # Plot 2: Linear x-axis with log10(gamma/4)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, L in enumerate(L_unique):
        df_L = df[df["L"] == L].sort_values("g")
        log_g = np.log10(df_L["g"].values)
        ax.plot(log_g, df_L["z2_plus_one"], marker="o", markersize=4, 
                linewidth=1.2, color=colors[idx], label=f"L={L} ({len(df_L)} pts)")
    
    ax.axhline(1.25, color="orange", linestyle="--", linewidth=1.5, 
               label="1.25", alpha=0.7)
    ax.axhline(2.0, color="green", linestyle="--", linewidth=1.5, 
               label="2.0", alpha=0.7)
    ax.set_xlabel("log₁₀(γ/4)", fontweight="bold")
    ax.set_ylabel("1+<z²>", fontweight="bold")
    ax.set_title("1+<z²> vs log₁₀(γ/4)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    
    plot_file = csv_file.parent / f"{csv_file.stem}_log10.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_file.name}")
    plt.close()
    
    # Plot 3: Runtime analysis
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for idx, L in enumerate(L_unique):
        df_L = df[df["L"] == L].sort_values("gamma")
        ax.scatter(df_L["gamma"], df_L["runtime_sec"], s=30, color=colors[idx],
                   alpha=0.6, label=f"L={L}")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("γ", fontweight="bold")
    ax.set_ylabel("Runtime (s)", fontweight="bold")
    ax.set_title("Runtime vs γ", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    
    plot_file = csv_file.parent / f"{csv_file.stem}_runtime.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_file.name}")
    plt.close()
    
    print("  ✓ All plots generated")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Long-run scan for 1+<z^2> vs log(gamma/4)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (resume if exists; create if missing)",
    )

    args = parser.parse_args()
    csv_file = Path(args.csv) if args.csv else DEFAULT_CSV

    run_parameter_sweep(csv_file)
    generate_verification_plots(csv_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
        sys.exit(1)
