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
import functools
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
from quantum_measurement.parallel import ParameterSweepExecutor
from quantum_measurement.backends import MultiCpuBackend, MultiCpuBackendConfig

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
T_MULTIPLIER = 60.0
T_MIN = 100.0  # Increased from 10 for better convergence
DT_RATIO = 5e-3
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
    "n_trajectories",
    "batch_size",
    "z2_std",
    "z2_stderr",
    "tau",
    "T",
    "dt",
    "N_steps",
    "epsilon",
    "runtime_sec",
    "nan_detected",
    "range_violation",
    "point_status",
]

NAN_MODES = ("fail_on_nan", "finish_full_sweep")


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


def get_time_params_with_config(gamma, t_multiplier, t_min, dt_ratio, dt_max):
    """Return (T, dt, N_steps, tau) using explicit runtime parameters."""
    tau = 1.0 / gamma
    T = max(tau * float(t_multiplier), float(t_min))
    dt = min(tau * float(dt_ratio), float(dt_max))
    N_steps = int(round(T / dt))
    return T, dt, N_steps, tau


def validate_time_grid(gamma_grid, t_multiplier, t_min, dt_ratio, dt_max):
    """Validate effective time-grid values before launching workers."""
    dt_vals = []
    epsilon_vals = []
    n_steps_vals = []
    invalid_reasons = []

    for gamma in gamma_grid:
        T, dt, n_steps, _ = get_time_params_with_config(
            gamma,
            t_multiplier=t_multiplier,
            t_min=t_min,
            dt_ratio=dt_ratio,
            dt_max=dt_max,
        )
        epsilon = float(np.sqrt(gamma * dt))
        dt_vals.append(float(dt))
        epsilon_vals.append(float(epsilon))
        n_steps_vals.append(int(n_steps))

        if not np.isfinite(dt) or dt <= 0.0 or dt > 1e-2:
            invalid_reasons.append(f"gamma={gamma}: invalid dt={dt}")
        if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 0.5:
            invalid_reasons.append(f"gamma={gamma}: invalid epsilon={epsilon}")
        if not np.isfinite(T) or T <= 0.0 or n_steps <= 0:
            invalid_reasons.append(f"gamma={gamma}: invalid T/N_steps (T={T}, N_steps={n_steps})")

    print(
        "Time-grid preflight: "
        f"dt in [{min(dt_vals):.3e}, {max(dt_vals):.3e}], "
        f"epsilon in [{min(epsilon_vals):.3e}, {max(epsilon_vals):.3e}], "
        f"N_steps in [{min(n_steps_vals)}, {max(n_steps_vals)}]"
    )

    if invalid_reasons:
        preview = "; ".join(invalid_reasons[:6])
        if len(invalid_reasons) > 6:
            preview += f"; ... and {len(invalid_reasons) - 6} more"
        raise ValueError(
            "Time-grid preflight failed. Adjust --t-multiplier/--t-min/--dt-ratio/--dt-max. "
            + preview
        )


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

def _run_single_point_with_config(
    L,
    gamma,
    backend_device,
    rng,
    n_trajectories_per_point,
    batch_size_per_point,
    compute_uncertainty,
    use_stable_integrator=False,
    enable_stability_monitor=False,
    t_multiplier=T_MULTIPLIER,
    t_min=T_MIN,
    dt_ratio=DT_RATIO,
    dt_max=DT_MAX,
    nan_mode="fail_on_nan",
):
    if nan_mode not in NAN_MODES:
        raise ValueError(f"Invalid nan_mode={nan_mode}. Expected one of {NAN_MODES}.")

    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    g = gamma / (4 * J_GLOBAL)
    T, dt, N_steps, tau = get_time_params_with_config(
        gamma,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
    )
    epsilon = float(np.sqrt(gamma * dt))
    if not np.isfinite(dt) or dt <= 0.0 or dt > 1e-2:
        raise ValueError(f"Invalid dt={dt} for L={L}, gamma={gamma}")
    if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 0.5:
        raise ValueError(f"Invalid epsilon={epsilon} for L={L}, gamma={gamma}")

    start_time = time.time()
    sim = LQubitCorrelationSimulator(
        L=L,
        J=J_GLOBAL,
        epsilon=epsilon,
        N_steps=N_steps,
        T=T,
        closed_boundary=True,
        device=backend_device,
        use_stable_integrator=bool(use_stable_integrator),
        enable_stability_monitor=bool(enable_stability_monitor),
        rng=np.random.default_rng(seed),
    )

    # Ensure backend RNG is reproducible for batched device-side sampling.
    if hasattr(sim.backend, "seed"):
        sim.backend.seed(seed)

    batch_size_used = int(batch_size_per_point) if batch_size_per_point is not None else None
    result = sim.simulate_z2_mean_ensemble(
        n_trajectories=int(n_trajectories_per_point),
        batch_size=batch_size_used,
        return_std_err=bool(compute_uncertainty),
    )

    if compute_uncertainty:
        z2_mean, z2_std, z2_stderr = result
    else:
        z2_mean = float(result)
        z2_std = float("nan")
        z2_stderr = float("nan")

    z2_plus_one = 1.0 + z2_mean
    nan_detected = not np.isfinite(z2_mean)
    range_violation = bool(
        np.isfinite(z2_mean) and ((z2_mean < -1e-9) or (z2_mean > (1.0 + 1e-6)))
    )
    if nan_detected:
        point_status = "nan"
    elif range_violation:
        point_status = "range_violation"
    else:
        point_status = "ok"
    if nan_detected and nan_mode == "fail_on_nan":
        raise ValueError(
            f"NaN detected for point L={L}, gamma={gamma}, seed={seed}, mode={nan_mode}."
        )

    runtime = time.time() - start_time

    return {
        "timestamp": datetime.now().isoformat(),
        "L": int(L),
        "gamma": float(gamma),
        "g": float(g),
        "z2_mean": float(z2_mean),
        "z2_plus_one": float(z2_plus_one),
        "n_trajectories": int(n_trajectories_per_point),
        "batch_size": batch_size_used,
        "z2_std": float(z2_std),
        "z2_stderr": float(z2_stderr),
        "tau": float(tau),
        "T": float(T),
        "dt": float(dt),
        "N_steps": int(N_steps),
        "epsilon": epsilon,
        "runtime_sec": float(runtime),
        "nan_detected": bool(nan_detected),
        "range_violation": bool(range_violation),
        "point_status": point_status,
    }


def _build_point_runner(
    n_trajectories_per_point,
    batch_size_per_point,
    compute_uncertainty,
    use_stable_integrator=False,
    enable_stability_monitor=False,
    t_multiplier=T_MULTIPLIER,
    t_min=T_MIN,
    dt_ratio=DT_RATIO,
    dt_max=DT_MAX,
    nan_mode="fail_on_nan",
):
    return functools.partial(
        _run_single_point_with_config,
        n_trajectories_per_point=n_trajectories_per_point,
        batch_size_per_point=batch_size_per_point,
        compute_uncertainty=compute_uncertainty,
        use_stable_integrator=use_stable_integrator,
        enable_stability_monitor=enable_stability_monitor,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
        nan_mode=nan_mode,
    )


def run_parameter_sweep(
    csv_file,
    backend_device="cpu",
    parallel_backend="sequential",
    executor_kind="parameter_sweep",
    n_workers=None,
    base_seed=42,
    resume=True,
    l_values_override=None,
    gamma_grid_override=None,
    n_trajectories_per_point=1,
    batch_size_per_point=None,
    compute_uncertainty=False,
    use_stable_integrator=False,
    enable_stability_monitor=False,
    t_multiplier=T_MULTIPLIER,
    t_min=T_MIN,
    dt_ratio=DT_RATIO,
    dt_max=DT_MAX,
    nan_mode="fail_on_nan",
    event_log_path=None,
):
    if nan_mode not in NAN_MODES:
        raise ValueError(f"Invalid nan_mode={nan_mode}. Expected one of {NAN_MODES}.")

    print("=" * 80)
    print("1+<z^2> PARAMETER SWEEP")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CSV file: {csv_file}")
    print()

    L_values = sorted({int(v) for v in l_values_override}) if l_values_override is not None else resolve_L_values()
    gamma_grid = (
        np.array(sorted({float(v) for v in gamma_grid_override}), dtype=float)
        if gamma_grid_override is not None
        else construct_gamma_grid()
    )
    g_grid = gamma_grid / (4 * J_GLOBAL)

    print("Parameter grid:")
    print(f"  L values: {L_values}")
    print(f"  gamma range: [{gamma_grid.min():.3e}, {gamma_grid.max():.3e}]")
    print(f"  g range: [{g_grid.min():.3e}, {g_grid.max():.3e}]")
    print(f"  N_gamma points: {len(gamma_grid)}")
    print(f"  Total runs: {len(L_values) * len(gamma_grid)}")
    print(f"  trajectories per point: {int(n_trajectories_per_point)}")
    print(f"  batch size per point: {batch_size_per_point if batch_size_per_point is not None else 'auto'}")
    print(f"  compute uncertainty: {bool(compute_uncertainty)}")
    print(f"  use stable integrator: {bool(use_stable_integrator)}")
    print(f"  enable stability monitor: {bool(enable_stability_monitor)}")
    print(f"  nan mode: {nan_mode}")
    print(
        "  time-grid: "
        f"t_multiplier={float(t_multiplier):.6g}, "
        f"t_min={float(t_min):.6g}, "
        f"dt_ratio={float(dt_ratio):.6g}, "
        f"dt_max={float(dt_max):.6g}"
    )
    validate_time_grid(
        gamma_grid,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
    )
    print()

    if resume and csv_file.exists():
        completed = load_existing_results(csv_file)
        print(f"Resuming: {len(completed)} runs already completed")
        print()

    if executor_kind == "multi_cpu":
        effective_event_log = (
            Path(event_log_path)
            if event_log_path is not None
            else csv_file.with_name(f"{csv_file.stem}_events.jsonl")
        )
        cfg = MultiCpuBackendConfig(
            max_workers=(n_workers if n_workers is not None else 38),
            master_seed=base_seed,
            nan_mode=nan_mode,
            log_path=effective_event_log,
        )
        executor = MultiCpuBackend(config=cfg)
        print(f"Executor: multi_cpu (workers={cfg.max_workers}, reserve_cores={cfg.reserve_cores})")
        print(f"Event log: {effective_event_log}")
    else:
        executor = ParameterSweepExecutor(
            parallel_backend=parallel_backend,
            n_workers=n_workers,
            base_seed=base_seed,
            verbose=True,
            continue_on_error=True,
        )
        print(f"Executor: parameter_sweep ({parallel_backend})")

    point_runner = _build_point_runner(
        n_trajectories_per_point=n_trajectories_per_point,
        batch_size_per_point=batch_size_per_point,
        compute_uncertainty=compute_uncertainty,
        use_stable_integrator=use_stable_integrator,
        enable_stability_monitor=enable_stability_monitor,
        t_multiplier=t_multiplier,
        t_min=t_min,
        dt_ratio=dt_ratio,
        dt_max=dt_max,
        nan_mode=nan_mode,
    )

    try:
        _ = executor.run_sweep(
            L_values=L_values,
            gamma_grid=gamma_grid,
            simulator_factory=point_runner,
            backend_device=backend_device,
            output_csv=csv_file,
            resume=resume,
            csv_header=CSV_HEADER,
        )
    except Exception as exc:
        print("=" * 80)
        print("SWEEP FAILED")
        print("=" * 80)
        print(f"Failure type: {type(exc).__name__}")
        print(f"Failure detail: {exc}")
        print(f"CSV target: {csv_file}")
        if executor_kind == "multi_cpu":
            print(f"Event log: {cfg.log_path}")
            print(f"Event counts: {executor.overflow_logger.counts()}")
        raise

    print("=" * 80)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {csv_file}")
    if executor_kind == "multi_cpu":
        print(f"Event counts: {executor.overflow_logger.counts()}")
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
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--parallel-backend",
        choices=["sequential", "multiprocessing"],
        default="sequential",
    )
    parser.add_argument(
        "--executor",
        choices=["parameter_sweep", "multi_cpu"],
        default="parameter_sweep",
        help="Executor implementation to use. 'multi_cpu' enables strict core-affinity backend.",
    )
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--l-values", nargs="+", type=int, default=None)
    parser.add_argument("--gamma-values", nargs="+", type=float, default=None)
    parser.add_argument("--n-trajectories-per-point", type=int, default=1)
    parser.add_argument("--batch-size-per-point", type=int, default=None)
    parser.add_argument("--compute-uncertainty", action="store_true")
    parser.add_argument("--use-stable-integrator", action="store_true")
    parser.add_argument("--enable-stability-monitor", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--t-multiplier", type=float, default=T_MULTIPLIER)
    parser.add_argument("--t-min", type=float, default=T_MIN)
    parser.add_argument("--dt-ratio", type=float, default=DT_RATIO)
    parser.add_argument("--dt-max", type=float, default=DT_MAX)
    parser.add_argument(
        "--event-log",
        type=str,
        default=None,
        help="Path to JSONL event log for multi_cpu executor (default: <csv_stem>_events.jsonl).",
    )
    parser.add_argument(
        "--nan-mode",
        choices=NAN_MODES,
        default="fail_on_nan",
        help="NaN policy: fail on first NaN point or finish full sweep with NaNs flagged.",
    )

    args = parser.parse_args()
    if args.t_multiplier <= 0 or args.t_min <= 0 or args.dt_ratio <= 0 or args.dt_max <= 0:
        raise ValueError("Time-grid overrides must be positive: --t-multiplier, --t-min, --dt-ratio, --dt-max")

    csv_file = Path(args.csv) if args.csv else DEFAULT_CSV

    run_parameter_sweep(
        csv_file,
        backend_device=args.device,
        parallel_backend=args.parallel_backend,
        executor_kind=args.executor,
        n_workers=args.n_workers,
        base_seed=args.base_seed,
        resume=(not args.no_resume),
        l_values_override=args.l_values,
        gamma_grid_override=args.gamma_values,
        n_trajectories_per_point=args.n_trajectories_per_point,
        batch_size_per_point=args.batch_size_per_point,
        compute_uncertainty=args.compute_uncertainty,
        use_stable_integrator=args.use_stable_integrator,
        enable_stability_monitor=args.enable_stability_monitor,
        t_multiplier=args.t_multiplier,
        t_min=args.t_min,
        dt_ratio=args.dt_ratio,
        dt_max=args.dt_max,
        nan_mode=args.nan_mode,
        event_log_path=args.event_log,
    )
    if not args.skip_plots:
        generate_verification_plots(csv_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
        sys.exit(1)
