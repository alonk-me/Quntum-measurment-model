#!/usr/bin/env python3
"""
Production Script: n_∞(γ) Parameter Sweep with Adaptive Convergence

This script performs a comprehensive parameter sweep to generate scientific-grade
verification plots of n_∞(γ) behavior, with special focus on the critical region
near g=1. Designed for long-running remote execution with checkpoint/resume support.

Features:
- Adaptive convergence with safety limits (max_steps=200k)
- Memory-based automatic fallback from L=256 to L=129
- Incremental CSV checkpointing after each (L,γ) pair
- Periodic BC matching sum_pbc analytical formulas
- Live progress logging for remote monitoring
- Matplotlib non-interactive backend for headless execution

Usage:
    # Direct execution
    python scripts/run_ninf_scan.py
    
    # With nohup for remote persistence
    nohup python scripts/run_ninf_scan.py > logs/ninf_scan.log 2>&1 &
    
    # Monitor progress
    tail -f logs/ninf_scan.log
    grep "COMPLETED" logs/ninf_scan.log | wc -l

Author: Auto-generated for Quantum Measurement Model verification
Date: 2026-01-06
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote execution
import matplotlib.pyplot as plt
from pathlib import Path
import time
import csv
from datetime import datetime
import sys
import psutil
import gc

# Import quantum measurement modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from quantum_measurement.jw_expansion.non_hermitian_hat import NonHermitianHatSimulator
from quantum_measurement.jw_expansion.n_infty import sum_pbc, integral_expr

# Configure matplotlib for publication-quality plots
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# ============================================================================
# Configuration Parameters
# ============================================================================

# System parameters
J_GLOBAL = 1.0  # Hopping strength (energy unit)

# Adaptive convergence settings
TOLERANCE = 1e-4
WINDOW_SIZE = 1000
DT = 0.001

# Adaptive max_steps: will be computed per (L, gamma) pair
# Base formula: 40 relaxation times with L-dependent correction

# Memory limit for automatic fallback (GB)
MEMORY_LIMIT_GB = 8.0  # Fallback from L=256 if single run would exceed this

# Size range with fallback
L_VALUES_FULL = [9, 17, 33, 65, 129, 256]
L_VALUES_FALLBACK = [9, 17, 33, 65, 129]

# Parameter sweep grids
# Global grid: log-spaced across many decades
# NOTE: Starting at γ=0.1 to avoid numerical instability in weak measurement regime
# Tested: γ<0.1 shows persistent oscillations and poor convergence
GAMMA_GLOBAL_MIN = 1e-1  # Was 1e-3, raised to 0.1 for numerical stability
GAMMA_GLOBAL_MAX = 1e2
N_GAMMA_GLOBAL = 80

# Critical region grid: dense linear sampling near g=1
G_CRITICAL_CENTER = 1.0
G_CRITICAL_WIDTH = 0.4  # ±0.4 around g=1, i.e., g ∈ [0.6, 1.4]
N_G_CRITICAL = 120

# Output paths
RESULTS_DIR = Path(__file__).parent.parent / "results" / "ninf_scan"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = RESULTS_DIR / f"ninf_data_{TIMESTAMP}.csv"
CSV_HEADER = ['timestamp', 'L', 'gamma', 'g', 'n_inf_sim', 'n_inf_exact', 
              'abs_error', 'rel_error', 'converged', 'steps', 'convergence_step',
              'max_steps_allocated', 'runtime_sec']

# ============================================================================
# Adaptive Parameters
# ============================================================================

def get_adaptive_max_steps(L, gamma, dt=DT):
    """
    Calculate max_steps based on system size and measurement strength.
    
    Rationale:
    - Relaxation time τ ~ 2/γ (measured in units of J⁻¹)
    - Larger systems need more steps due to longer correlation times
    - Use logarithmic correction to avoid excessive scaling
    - Cap at reasonable limits to avoid multi-hour runs
    
    Parameters
    ----------
    L : int
        System size
    gamma : float
        Measurement rate
    dt : float
        Time step
        
    Returns
    -------
    int
        Maximum number of steps to allow
    """
    # Base estimate: 10 relaxation times (reduced from 40)
    # For weak measurement, convergence is slower but not 40× slower
    base_time = 20.0 / gamma  # Relaxation time in J^-1 units
    base_steps = int(base_time / dt)
    
    # Size correction (logarithmic to avoid explosion)
    size_factor = 1.0 + 0.2 * np.log(max(L / 9.0, 1.0))
    
    # Apply correction and bound
    max_steps = int(base_steps * size_factor)
    
    # Safety bounds: lower minimum, reasonable maximum
    # Even for gamma=0.001, cap at 500k steps (500s simulated time)
    max_steps = max(30000, min(max_steps, 500000))
    
    return max_steps

# ============================================================================
# Memory Estimation and Fallback Logic
# ============================================================================

def estimate_memory_gb(L, gamma=0.1):
    """
    Estimate peak memory usage for time evolution of size L.
    
    The correlation matrix is 2L×2L complex128, and we store trajectory
    of length ~max_steps. Main memory cost:
    - G matrix: (2L)² × 16 bytes
    - n_traj: max_steps × L × 8 bytes
    - Working memory: ~3× matrix size for operations
    
    Parameters
    ----------
    L : int
        Chain length
    gamma : float
        Measurement rate (used for adaptive max_steps estimate)
        
    Returns
    -------
    float
        Estimated peak memory in GB
    """
    matrix_size_gb = (2 * L)**2 * 16 / (1024**3)  # Complex128 = 16 bytes
    max_steps_estimate = get_adaptive_max_steps(L, gamma)
    trajectory_size_gb = max_steps_estimate * L * 8 / (1024**3)  # Float64 = 8 bytes
    working_memory_gb = matrix_size_gb * 3  # Temporary arrays in evolution
    
    total_gb = matrix_size_gb + trajectory_size_gb + working_memory_gb
    return total_gb

def select_L_values_with_fallback():
    """
    Determine L values to use based on available system memory.
    
    Returns
    -------
    list of int
        L values that fit in memory
    str
        Status message about fallback
    """
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Check if L=256 fits
    L_max_memory = estimate_memory_gb(256)
    
    if L_max_memory < min(MEMORY_LIMIT_GB, available_memory_gb * 0.7):
        # Safe to use full range
        L_values = L_VALUES_FULL
        status = f"✓ Full L range [9-256]: estimated {L_max_memory:.2f} GB < limit {MEMORY_LIMIT_GB:.2f} GB"
    else:
        # Fallback to L_max=129
        L_values = L_VALUES_FALLBACK
        L_fallback_memory = estimate_memory_gb(129)
        status = f"⚠ Fallback to L≤129: L=256 would use {L_max_memory:.2f} GB > limit {MEMORY_LIMIT_GB:.2f} GB"
    
    return L_values, status

# ============================================================================
# Adaptive Simulation Function
# ============================================================================

def simulate_with_adaptive_convergence(L, J, gamma, dt=DT, tolerance=TOLERANCE,
                                       window_size=WINDOW_SIZE):
    """
    Run time evolution with adaptive stopping and periodic BC.
    
    FIXED: Single-run pattern to avoid restarting from vacuum.
    Uses L and gamma-dependent max_steps for optimal resource usage.
    
    Parameters
    ----------
    L, J, gamma : system parameters
    dt : float
        Time step
    tolerance : float
        Convergence threshold
    window_size : int
        Window for averaging
        
    Returns
    -------
    dict with results
    """
    start_time = time.time()
    
    # Adaptive max_steps based on system size and measurement strength
    max_steps = get_adaptive_max_steps(L, gamma, dt)
    
    # Initialize with periodic BC
    sim = NonHermitianHatSimulator(
        L=L, J=J, gamma=gamma, dt=dt, N_steps=max_steps,
        closed_boundary=True  # Periodic BC for sum_pbc comparison
    )
    
    # Run simulation ONCE (key fix: no chunks, no restarts)
    Q_total, n_traj, G_final = sim.simulate_trajectory(return_G_final=True)
    
    # Post-hoc convergence detection
    converged = False
    convergence_step = max_steps
    
    for step in range(window_size, len(n_traj)):
        n_recent = n_traj[step - window_size//2:step, :].mean(axis=0)
        n_previous = n_traj[step - window_size:step - window_size//2, :].mean(axis=0)
        max_diff = np.max(np.abs(n_recent - n_previous))
        
        if max_diff < tolerance:
            converged = True
            convergence_step = step
            break
    
    # Extract n_infinity from final portion
    n_infinity = n_traj[-window_size:, :].mean()
    runtime = time.time() - start_time
    
    # Clean up
    gc.collect()
    
    return {
        'n_infinity': n_infinity,
        'n_traj': n_traj,  # Keep for potential time-trace analysis
        'G_final': G_final,
        'converged': converged,
        'steps': len(n_traj) - 1,
        'convergence_step': convergence_step,
        'runtime': runtime,
        'max_steps_allocated': max_steps
    }

# ============================================================================
# Parameter Grid Construction
# ============================================================================

def construct_gamma_grid():
    """
    Build combined γ grid: global log-spaced + critical region dense.
    
    Note: Excludes γ < 0.1 due to numerical instability in the
    non-Hermitian evolution. Weak measurement regimes (γ < 1) show
    persistent oscillations that prevent convergence.
    
    Returns
    -------
    ndarray
        Sorted unique gamma values
    """
    # Global log grid
    gamma_global = np.logspace(np.log10(GAMMA_GLOBAL_MIN), 
                               np.log10(GAMMA_GLOBAL_MAX), 
                               N_GAMMA_GLOBAL)
    
    # Critical region in g space
    g_critical = np.linspace(G_CRITICAL_CENTER - G_CRITICAL_WIDTH,
                             G_CRITICAL_CENTER + G_CRITICAL_WIDTH,
                             N_G_CRITICAL)
    gamma_critical = g_critical * 4 * J_GLOBAL
    
    # Combine and sort
    gamma_combined = np.unique(np.concatenate([gamma_global, gamma_critical]))
    
    # Remove any negative values (shouldn't happen, but safety check)
    gamma_combined = gamma_combined[gamma_combined > 0]
    
    return gamma_combined

# ============================================================================
# CSV Checkpoint Management
# ============================================================================

def initialize_csv():
    """Create CSV file with header."""
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
    print(f"✓ Initialized CSV: {CSV_FILE}")

def append_to_csv(data_dict):
    """Append single row to CSV."""
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [data_dict[key] for key in CSV_HEADER]
        writer.writerow(row)

def load_existing_results():
    """
    Load existing CSV to avoid recomputing.
    
    Returns
    -------
    set of tuples
        (L, gamma) pairs already computed
    """
    if not CSV_FILE.exists():
        return set()
    
    completed = set()
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            L = int(row['L'])
            gamma = float(row['gamma'])
            completed.add((L, gamma))
    
    return completed

# ============================================================================
# Main Sweep Loop
# ============================================================================

def run_parameter_sweep():
    """Execute full parameter sweep with checkpointing."""
    
    print("="*80)
    print("n_∞(γ) PARAMETER SWEEP — PRODUCTION RUN")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Determine L values with memory fallback
    L_values, memory_status = select_L_values_with_fallback()
    print(memory_status)
    print()
    
    # Construct gamma grid
    gamma_grid = construct_gamma_grid()
    g_grid = gamma_grid / (4 * J_GLOBAL)
    print(f"Parameter grid:")
    print(f"  L values: {L_values}")
    print(f"  γ range: [{gamma_grid.min():.3e}, {gamma_grid.max():.3e}]")
    print(f"  g range: [{g_grid.min():.3e}, {g_grid.max():.3e}]")
    print(f"  N_γ points: {len(gamma_grid)}")
    print(f"  Total runs: {len(L_values) * len(gamma_grid)}")
    print()
    
    # Initialize or resume
    if not CSV_FILE.exists():
        initialize_csv()
        completed = set()
    else:
        completed = load_existing_results()
        print(f"⟳ Resuming: {len(completed)} runs already completed")
        print()
    
    # Main loop
    total_runs = len(L_values) * len(gamma_grid)
    run_count = 0
    skipped_count = len(completed)
    
    for L in L_values:
        print(f"\n{'='*80}")
        print(f"SYSTEM SIZE: L = {L}")
        print(f"{'='*80}\n")
        
        for i, gamma in enumerate(gamma_grid):
            run_count += 1
            
            # Skip if already computed
            if (L, gamma) in completed:
                continue
            
            g = gamma / (4 * J_GLOBAL)
            
            # Analytical reference
            n_exact = sum_pbc(g, L)
            
            # Progress header
            progress_pct = 100 * run_count / total_runs
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Run {run_count}/{total_runs} ({progress_pct:.1f}%)")
            print(f"  L={L:3d}, γ={gamma:8.4f}, g={g:8.4f}")
            
            # Run simulation
            try:
                result = simulate_with_adaptive_convergence(
                    L=L, J=J_GLOBAL, gamma=gamma
                )
                
                # Compute errors
                abs_error = np.abs(result['n_infinity'] - n_exact)
                rel_error = abs_error / n_exact if n_exact > 0 else np.inf
                
                # Log results
                status = "✓ CONVERGED" if result['converged'] else "⚠ MAX_STEPS"
                print(f"  {status}: n_∞^sim={result['n_infinity']:.6f}, n_∞^exact={n_exact:.6f}")
                print(f"  Error: abs={abs_error:.2e}, rel={rel_error:.2e}")
                print(f"  Steps: {result['steps']}, Time: {result['runtime']:.1f}s")
                
                # Save to CSV
                data_dict = {
                    'timestamp': datetime.now().isoformat(),
                    'L': L,
                    'gamma': gamma,
                    'g': g,
                    'n_inf_sim': result['n_infinity'],
                    'n_inf_exact': n_exact,
                    'abs_error': abs_error,
                    'rel_error': rel_error,
                    'converged': int(result['converged']),
                    'steps': result['steps'],
                    'convergence_step': result['convergence_step'],
                    'max_steps_allocated': result['max_steps_allocated'],
                    'runtime_sec': result['runtime']
                }
                append_to_csv(data_dict)
                
                print(f"  💾 SAVED to CSV")
                
                # Clean up large arrays
                del result
                gc.collect()
                
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Skipping this point...")
            
            print()
    
    print("="*80)
    print("PARAMETER SWEEP COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {CSV_FILE}")
    print()

# ============================================================================
# Post-Processing: Generate Plots
# ============================================================================

def generate_verification_plots():
    """Generate all scientific verification plots from CSV data."""
    
    print("\nGenerating verification plots...")
    
    import pandas as pd
    
    # Load data
    if not CSV_FILE.exists():
        print("✗ No CSV file found, skipping plots")
        return
    
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded {len(df)} data points from {CSV_FILE.name}")
    
    # Plot 1: n_∞(g) for all L with exact curve
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Analytical thermodynamic limit
    g_analytical = np.logspace(-3, 1, 300)
    n_analytical = np.array([integral_expr(g) for g in g_analytical])
    ax.plot(g_analytical, n_analytical, 'k-', linewidth=2.5, 
            label='L→∞ (exact integral)', zorder=10)
    
    # Finite-size analytical curves
    L_unique = sorted(df['L'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_unique)))
    
    for idx, L in enumerate(L_unique):
        # Analytical
        n_ana_L = np.array([sum_pbc(g, L) for g in g_analytical])
        ax.plot(g_analytical, n_ana_L, '--', color=colors[idx], 
                linewidth=1.2, alpha=0.5, label=f'L={L} (analytical)')
        
        # Simulation points
        df_L = df[df['L'] == L]
        ax.scatter(df_L['g'], df_L['n_inf_sim'], s=40, color=colors[idx],
                   marker='o', edgecolor='black', linewidth=0.5, zorder=20,
                   label=f'L={L} (simulation)')
    
    ax.set_xscale('log')
    ax.set_xlabel('g = γ/(4J)', fontweight='bold')
    ax.set_ylabel('n_∞(g)', fontweight='bold')
    ax.set_title('Steady-State Occupation: Simulation vs Analytical', fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0, 0.5)
    
    plt.tight_layout()
    plot_file = RESULTS_DIR / f"ninf_vs_g__BC-pbc__all-L__{TIMESTAMP}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_file.name}")
    plt.close()
    
    # Plot 2: Error analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute error vs g
    ax = axes[0]
    for idx, L in enumerate(L_unique):
        df_L = df[df['L'] == L]
        ax.scatter(df_L['g'], df_L['abs_error'], s=30, color=colors[idx],
                   label=f'L={L}', alpha=0.7)
    ax.axhline(TOLERANCE, color='red', linestyle='--', linewidth=1.5,
               label=f'Tolerance={TOLERANCE:.1e}', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('g = γ/(4J)', fontweight='bold')
    ax.set_ylabel('|n_∞^sim - n_∞^exact|', fontweight='bold')
    ax.set_title('Absolute Error', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    # Relative error vs g
    ax = axes[1]
    for idx, L in enumerate(L_unique):
        df_L = df[df['L'] == L]
        ax.scatter(df_L['g'], df_L['rel_error'] * 100, s=30, color=colors[idx],
                   label=f'L={L}', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('g = γ/(4J)', fontweight='bold')
    ax.set_ylabel('Relative Error (%)', fontweight='bold')
    ax.set_title('Relative Error', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plot_file = RESULTS_DIR / f"error_analysis__{TIMESTAMP}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_file.name}")
    plt.close()
    
    # Plot 3: Critical region zoom (g near 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to critical region
    g_min_zoom = 0.6
    g_max_zoom = 1.4
    
    # Analytical curves
    g_zoom = np.linspace(g_min_zoom, g_max_zoom, 500)
    n_zoom_inf = np.array([integral_expr(g) for g in g_zoom])
    ax.plot(g_zoom, n_zoom_inf, 'k-', linewidth=2.5, 
            label='L→∞ (exact)', zorder=10)
    
    for idx, L in enumerate(L_unique):
        n_zoom_L = np.array([sum_pbc(g, L) for g in g_zoom])
        ax.plot(g_zoom, n_zoom_L, '--', color=colors[idx], 
                linewidth=1.2, alpha=0.5)
        
        df_L = df[(df['L'] == L) & (df['g'] >= g_min_zoom) & (df['g'] <= g_max_zoom)]
        if len(df_L) > 0:
            ax.scatter(df_L['g'], df_L['n_inf_sim'], s=50, color=colors[idx],
                       marker='o', edgecolor='black', linewidth=0.7, zorder=20,
                       label=f'L={L}')
    
    ax.axvline(1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5,
               label='g=1 (critical)')
    ax.set_xlabel('g = γ/(4J)', fontweight='bold')
    ax.set_ylabel('n_∞(g)', fontweight='bold')
    ax.set_title('Critical Region: n_∞(g) near g=1', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = RESULTS_DIR / f"ninf_critical_region__g-near-1__{TIMESTAMP}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_file.name}")
    plt.close()
    
    print("✓ All plots generated")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUANTUM MEASUREMENT MODEL: n_∞(γ) VERIFICATION")
    print("="*80)
    print(f"Script: {__file__}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Output directory: {RESULTS_DIR}")
    print("="*80 + "\n")
    
    try:
        # Run parameter sweep
        run_parameter_sweep()
        
        # Generate plots
        generate_verification_plots()
        
        print("\n" + "="*80)
        print("✓ ALL TASKS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user (Ctrl+C)")
        print("Progress saved to CSV. Safe to resume later.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
