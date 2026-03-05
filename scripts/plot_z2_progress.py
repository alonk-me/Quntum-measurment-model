#!/usr/bin/env python3
"""
Live Progress Monitor for 1+<z^2> Parameter Sweep

This script monitors the CSV output from run_z2_scan.py and generates a
live plot for remote monitoring.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "results" / "z2_scan"
REFRESH_INTERVAL = 30
PLOT_FILE = RESULTS_DIR / "live_progress.png"

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 9


# ============================================================================
# Helpers
# ============================================================================

def find_latest_csv(results_dir):
    csv_files = list(results_dir.glob("z2_data_*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def load_csv_safe(csv_file):
    try:
        return pd.read_csv(csv_file)
    except Exception:
        return None


def format_time_delta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}hr"


# ============================================================================
# Plotting
# ============================================================================

def generate_progress_plot(df, csv_file):
    if df is None or len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Waiting for data...", ha="center", va="center", fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
        plt.close()
        return

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Main plot
    ax1 = fig.add_subplot(gs[:, 0])

    df = df.sort_values(["L", "gamma"])
    L_unique = sorted(df["L"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_unique)))

    for idx, L in enumerate(L_unique):
        df_L = df[df["L"] == L]
        log_g = np.log10(df_L["g"].values)
        y = df_L["z2_plus_one"].values
        ax1.plot(log_g, y, marker="o", markersize=3, linewidth=1.2, color=colors[idx],
                 label=f"L={L} ({len(df_L)})")

    ax1.axhline(1.25, color="orange", linestyle="--", linewidth=1.0, label="1.25")
    ax1.axhline(2.0, color="green", linestyle="--", linewidth=1.0, label="2.0")
    ax1.set_xlabel("log₁₀(γ/4)", fontweight="bold")
    ax1.set_ylabel("1+<z²>", fontweight="bold")
    ax1.set_title("Live Progress: 1+<z²> vs log₁₀(γ/4)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8, ncol=2)

    # Runtime distribution
    ax2 = fig.add_subplot(gs[0, 1])
    runtimes = df["runtime_sec"].values
    ax2.hist(runtimes, bins=30, edgecolor="black", linewidth=0.5, alpha=0.7)
    ax2.set_xlabel("Runtime (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Runtime Distribution", fontweight="bold", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Status panel
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    latest = df.iloc[-1]
    status_text = f"""
LIVE STATUS
{'='*22}
File: {csv_file.name}
Points: {len(df)}

Latest Point:
  Time: {latest['timestamp']}
  L: {int(latest['L'])}
  gamma: {latest['gamma']:.4f}
  g: {latest['g']:.4f}
  1+z2: {latest['z2_plus_one']:.6f}

Runtime:
  Total: {format_time_delta(df['runtime_sec'].sum())}
  Mean: {format_time_delta(df['runtime_sec'].mean())}
"""
    ax3.text(
        0.05,
        0.95,
        status_text,
        transform=ax3.transAxes,
        fontsize=8,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    fig.suptitle(
        "1+<z^2> Parameter Sweep — Live Progress Monitor\n"
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=12,
        fontweight="bold",
    )

    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Monitor loop
# ============================================================================

def monitor_loop(csv_file=None, interval=REFRESH_INTERVAL):
    print("=" * 80)
    print("LIVE PROGRESS MONITOR FOR 1+<z^2> SWEEP")
    print("=" * 80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Refresh interval: {interval}s")
    print(f"Output plot: {PLOT_FILE}")
    print()

    if csv_file is None:
        print("Auto-detecting latest CSV file...")
    else:
        print(f"Monitoring: {csv_file}")

    print("\nPress Ctrl+C to stop monitoring\n")
    print("=" * 80)

    iteration = 0
    last_n_points = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            if csv_file is None:
                current_csv = find_latest_csv(RESULTS_DIR)
                if current_csv is None:
                    print(f"[{timestamp}] No CSV files found yet, waiting...")
                    time.sleep(interval)
                    continue
            else:
                current_csv = Path(csv_file)
                if not current_csv.exists():
                    print(f"[{timestamp}] CSV file not found: {current_csv}")
                    time.sleep(interval)
                    continue

            df = load_csv_safe(current_csv)
            if df is None:
                print(f"[{timestamp}] Error reading CSV (may be mid-write), retrying...")
                time.sleep(5)
                continue

            n_points = len(df)
            new_points = n_points - last_n_points

            generate_progress_plot(df, current_csv)

            status_symbol = "⟳" if new_points > 0 else "⏸"
            latest = df.iloc[-1]
            print(
                f"[{timestamp}] {status_symbol} Iteration {iteration}: "
                f"{n_points} total points (+{new_points} new), "
                f"latest: L={int(latest['L'])} gamma={latest['gamma']:.3f}"
            )

            last_n_points = n_points
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user (Ctrl+C)")
        print(f"Final plot saved to: {PLOT_FILE}")
        return


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Live progress monitor for 1+<z^2> parameter sweep",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (auto-detect if not specified)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=REFRESH_INTERVAL,
        help=f"Refresh interval in seconds (default: {REFRESH_INTERVAL})",
    )

    args = parser.parse_args()
    monitor_loop(csv_file=args.csv, interval=args.interval)


if __name__ == "__main__":
    main()
