#!/usr/bin/env python3
"""
Live Progress Monitor for n_∞(γ) Parameter Sweep

This script continuously monitors the CSV output from run_ninf_scan.py and
generates real-time progress plots. Designed to run in a parallel tmux pane
for live visualization during long remote runs.

Features:
- Auto-detects latest CSV file in results/ninf_scan/
- Updates plots every 30 seconds
- Shows completion progress, current status, runtime estimates
- Lightweight: minimal memory/CPU overhead
- Graceful handling of file updates and truncation

Usage:
    # In a separate tmux pane
    python scripts/plot_progress.py
    
    # Or specify CSV file explicitly
    python scripts/plot_progress.py --csv results/ninf_scan/ninf_data_20260106.csv
    
    # Adjust refresh rate
    python scripts/plot_progress.py --interval 60

Author: Auto-generated for Quantum Measurement Model verification
Date: 2026-01-06
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import time
import pandas as pd
import argparse
from datetime import datetime
import sys

# Configure matplotlib
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 9

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent.parent / "results" / "ninf_scan"
REFRESH_INTERVAL = 30  # seconds
PLOT_FILE = RESULTS_DIR / "live_progress.png"

# ============================================================================
# Helper Functions
# ============================================================================

def find_latest_csv(results_dir):
    """Find most recently modified CSV file in results directory."""
    csv_files = list(results_dir.glob("ninf_data_*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=lambda p: p.stat().st_mtime)

def load_csv_safe(csv_file):
    """Load CSV with error handling for incomplete writes."""
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        # CSV might be mid-write, return None
        return None

def format_time_delta(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}hr"

# ============================================================================
# Progress Analysis
# ============================================================================

def analyze_progress(df):
    """
    Extract progress metrics from dataframe.
    
    Returns
    -------
    dict with progress statistics
    """
    if df is None or len(df) == 0:
        return None
    
    # Basic counts
    n_total = len(df)
    n_converged = df['converged'].sum()
    
    # L distribution
    L_counts = df['L'].value_counts().to_dict()
    
    # Errors
    max_abs_error = df['abs_error'].max()
    mean_abs_error = df['abs_error'].mean()
    max_rel_error = df['rel_error'].max()
    
    # Runtime
    total_runtime = df['runtime_sec'].sum()
    mean_runtime = df['runtime_sec'].mean()
    
    # Latest entry
    if 'timestamp' in df.columns:
        try:
            latest_time = pd.to_datetime(df['timestamp'].iloc[-1])
            latest_str = latest_time.strftime('%H:%M:%S')
        except:
            latest_str = "N/A"
    else:
        latest_str = "N/A"
    
    latest_L = df['L'].iloc[-1]
    latest_gamma = df['gamma'].iloc[-1]
    latest_g = df['g'].iloc[-1]
    
    return {
        'n_total': n_total,
        'n_converged': n_converged,
        'convergence_rate': n_converged / n_total if n_total > 0 else 0,
        'L_counts': L_counts,
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'max_rel_error': max_rel_error,
        'total_runtime': total_runtime,
        'mean_runtime': mean_runtime,
        'latest_time': latest_str,
        'latest_L': latest_L,
        'latest_gamma': latest_gamma,
        'latest_g': latest_g,
    }

# ============================================================================
# Plot Generation
# ============================================================================

def generate_progress_plot(df, stats, csv_file):
    """Generate comprehensive progress visualization."""
    
    if df is None or len(df) == 0:
        # Empty plot with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Waiting for data...", 
                ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Panel 1: Current n_∞(g) with all data points
    # ========================================================================
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Get analytical reference
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from quantum_measurement.jw_expansion.n_infty import integral_expr, sum_pbc
    
    g_analytical = np.logspace(-3, 1, 200)
    n_analytical = np.array([integral_expr(g) for g in g_analytical])
    ax1.plot(g_analytical, n_analytical, 'k-', linewidth=2, 
             label='L→∞ (exact)', zorder=10)
    
    # Plot data by L
    L_unique = sorted(df['L'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_unique)))
    
    for idx, L in enumerate(L_unique):
        df_L = df[df['L'] == L]
        n_count = len(df_L)
        
        # Analytical curve for this L
        n_ana_L = np.array([sum_pbc(g, L) for g in g_analytical])
        ax1.plot(g_analytical, n_ana_L, '--', color=colors[idx], 
                 linewidth=1, alpha=0.3)
        
        # Simulation points
        ax1.scatter(df_L['g'], df_L['n_inf_sim'], s=20, color=colors[idx],
                    marker='o', edgecolor='black', linewidth=0.3, zorder=20,
                    label=f'L={L} ({n_count} pts)')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('g = γ/(4J)', fontweight='bold')
    ax1.set_ylabel('n_∞(g)', fontweight='bold')
    ax1.set_title('Live Progress: n_∞(g) Simulation vs Analytical', fontweight='bold', fontsize=11)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(0, 0.5)
    
    # ========================================================================
    # Panel 2: Error distribution
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Histogram of absolute errors
    errors = df['abs_error'].values
    ax2.hist(errors, bins=30, edgecolor='black', linewidth=0.5, alpha=0.7)
    ax2.axvline(stats['mean_abs_error'], color='red', linestyle='--', 
                linewidth=1.5, label=f'Mean: {stats["mean_abs_error"]:.2e}')
    ax2.set_xlabel('|n_∞^sim - n_∞^exact|', fontsize=9)
    ax2.set_ylabel('Count', fontsize=9)
    ax2.set_title('Error Distribution', fontweight='bold', fontsize=10)
    ax2.legend(fontsize=7)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # ========================================================================
    # Panel 3: Convergence statistics
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 2])
    
    converged = df['converged'].sum()
    not_converged = len(df) - converged
    
    wedges, texts, autotexts = ax3.pie(
        [converged, not_converged],
        labels=['Converged', 'Max Steps'],
        autopct='%1.1f%%',
        colors=['lightgreen', 'lightcoral'],
        startangle=90
    )
    ax3.set_title('Convergence Rate', fontweight='bold', fontsize=10)
    
    # ========================================================================
    # Panel 4: Runtime analysis
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Runtime vs L
    for L in L_unique:
        df_L = df[df['L'] == L]
        runtimes = df_L['runtime_sec'].values
        ax4.scatter([L] * len(runtimes), runtimes, alpha=0.5, s=10)
    
    # Box plot overlay
    runtime_by_L = [df[df['L'] == L]['runtime_sec'].values for L in L_unique]
    bp = ax4.boxplot(runtime_by_L, positions=L_unique, widths=np.array(L_unique)*0.3,
                      patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.5)
    
    ax4.set_xlabel('L', fontweight='bold')
    ax4.set_ylabel('Runtime (s)', fontweight='bold')
    ax4.set_title('Runtime vs System Size', fontweight='bold', fontsize=10)
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # Panel 5: Progress by L
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    L_list = sorted(stats['L_counts'].keys())
    counts = [stats['L_counts'][L] for L in L_list]
    
    bars = ax5.bar(range(len(L_list)), counts, tick_label=L_list, 
                    color=colors[:len(L_list)], edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('L', fontweight='bold')
    ax5.set_ylabel('Completed Runs', fontweight='bold')
    ax5.set_title('Progress by System Size', fontweight='bold', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(count)}', ha='center', va='bottom', fontsize=8)
    
    # ========================================================================
    # Panel 6: Status text summary
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    status_text = f"""
LIVE STATUS MONITOR
{'='*30}

File: {csv_file.name}

Total Runs: {stats['n_total']}
Converged: {stats['n_converged']} ({stats['convergence_rate']*100:.1f}%)

Latest Point:
  Time: {stats['latest_time']}
  L = {stats['latest_L']}
  γ = {stats['latest_gamma']:.4f}
  g = {stats['latest_g']:.4f}

Errors:
  Max Abs: {stats['max_abs_error']:.2e}
  Mean Abs: {stats['mean_abs_error']:.2e}
  Max Rel: {stats['max_rel_error']*100:.3f}%

Runtime:
  Total: {format_time_delta(stats['total_runtime'])}
  Mean: {format_time_delta(stats['mean_runtime'])}

L Distribution:
{chr(10).join(f'  L={L}: {count} runs' for L, count in sorted(stats['L_counts'].items()))}
"""
    
    ax6.text(0.05, 0.95, status_text, transform=ax6.transAxes,
             fontsize=8, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ========================================================================
    # Main title with timestamp
    # ========================================================================
    fig.suptitle(f'n_∞(γ) Parameter Sweep — Live Progress Monitor\n'
                 f'Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 fontsize=13, fontweight='bold')
    
    # Save
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Main Monitor Loop
# ============================================================================

def monitor_loop(csv_file=None, interval=REFRESH_INTERVAL):
    """Main monitoring loop."""
    
    print("="*80)
    print("LIVE PROGRESS MONITOR FOR n_∞(γ) PARAMETER SWEEP")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Refresh interval: {interval}s")
    print(f"Output plot: {PLOT_FILE}")
    print()
    
    if csv_file is None:
        print("Auto-detecting latest CSV file...")
    else:
        print(f"Monitoring: {csv_file}")
    
    print("\nPress Ctrl+C to stop monitoring\n")
    print("="*80)
    
    iteration = 0
    last_n_points = 0
    
    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Find or use CSV
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
            
            # Load data
            df = load_csv_safe(current_csv)
            if df is None:
                print(f"[{timestamp}] Error reading CSV (may be mid-write), retrying...")
                time.sleep(5)
                continue
            
            n_points = len(df)
            new_points = n_points - last_n_points
            
            # Analyze
            stats = analyze_progress(df)
            if stats is None:
                print(f"[{timestamp}] Empty dataframe, waiting...")
                time.sleep(interval)
                continue
            
            # Generate plot
            generate_progress_plot(df, stats, current_csv)
            
            # Log update
            status_symbol = "⟳" if new_points > 0 else "⏸"
            print(f"[{timestamp}] {status_symbol} Iteration {iteration}: "
                  f"{n_points} total points (+{new_points} new), "
                  f"converged: {stats['n_converged']}/{n_points}, "
                  f"latest: L={stats['latest_L']} γ={stats['latest_gamma']:.3f}")
            
            last_n_points = n_points
            
            # Wait
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠ Monitoring stopped by user (Ctrl+C)")
        print(f"Final plot saved to: {PLOT_FILE}")
        sys.exit(0)

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Live progress monitor for n_∞(γ) parameter sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect latest CSV
  python scripts/plot_progress.py
  
  # Monitor specific file
  python scripts/plot_progress.py --csv results/ninf_scan/ninf_data_20260106.csv
  
  # Faster updates
  python scripts/plot_progress.py --interval 15
        """
    )
    
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV file (auto-detect if not specified)')
    parser.add_argument('--interval', type=int, default=REFRESH_INTERVAL,
                        help=f'Refresh interval in seconds (default: {REFRESH_INTERVAL})')
    
    args = parser.parse_args()
    
    # Run monitor
    monitor_loop(csv_file=args.csv, interval=args.interval)
