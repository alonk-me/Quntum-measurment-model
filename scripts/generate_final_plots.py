#!/usr/bin/env python3
"""
Enhanced Analysis: n_∞(g) with First and Second Derivatives

Generates three publication-quality plots:
1. n_∞(g) simulation vs analytical (all L + exact)
2. First derivative dn_∞/dg per L (smoothed, σ=0.02)
3. Second derivative d²n_∞/dg² per L (smoothed, σ=0.03)

Filters unconverged runs and applies Gaussian smoothing to reduce noise.

Usage:
    python scripts/generate_final_plots.py
    python scripts/generate_final_plots.py --csv results/ninf_scan/ninf_data_TIMESTAMP.csv

Author: Enhanced analysis for Quantum Measurement Model
Date: 2026-01-14
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
import sys
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from quantum_measurement.jw_expansion.n_infty import sum_pbc, integral_expr

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

RESULTS_DIR = Path(__file__).parent.parent / "results" / "ninf_scan"

# ============================================================================
# Smoothing and Derivative Utilities
# ============================================================================

def smooth_data_gaussian(g_vals, n_vals, sigma_g=0.02):
    """
    Apply Gaussian smoothing in g-space.
    
    Parameters
    ----------
    g_vals : ndarray
        g values (must be sorted)
    n_vals : ndarray
        n_∞ values
    sigma_g : float
        Smoothing width in g units
        
    Returns
    -------
    n_smooth : ndarray
        Smoothed n_∞ values
    """
    # Convert sigma from g-space to index-space
    # Estimate local spacing
    avg_dg = np.median(np.diff(g_vals))
    sigma_idx = sigma_g / avg_dg
    
    n_smooth = gaussian_filter1d(n_vals, sigma=sigma_idx)
    return n_smooth


def compute_first_derivative(g_vals, n_vals, sigma_g=0.02):
    """
    Compute dn/dg using central differences with Gaussian smoothing.
    
    Parameters
    ----------
    g_vals : ndarray
        g values (must be sorted)
    n_vals : ndarray
        n_∞ values
    sigma_g : float
        Smoothing width in g units
        
    Returns
    -------
    g_deriv, dn_dg : ndarrays
    """
    # Smooth first
    n_smooth = smooth_data_gaussian(g_vals, n_vals, sigma_g)
    
    # Central differences (2nd order accurate)
    dn_dg = np.zeros_like(g_vals)
    dn_dg[1:-1] = (n_smooth[2:] - n_smooth[:-2]) / (g_vals[2:] - g_vals[:-2])
    
    # Forward/backward at edges
    dn_dg[0] = (n_smooth[1] - n_smooth[0]) / (g_vals[1] - g_vals[0])
    dn_dg[-1] = (n_smooth[-1] - n_smooth[-2]) / (g_vals[-1] - g_vals[-2])
    
    return g_vals, dn_dg


def compute_second_derivative(g_vals, n_vals, sigma_g=0.03):
    """
    Compute d²n/dg² using central differences with Gaussian smoothing.
    
    Parameters
    ----------
    g_vals : ndarray
        g values (must be sorted)
    n_vals : ndarray
        n_∞ values
    sigma_g : float
        Smoothing width in g units (typically larger than first derivative)
        
    Returns
    -------
    g_deriv2, d2n_dg2 : ndarrays
    """
    # Smooth with larger sigma for second derivative
    n_smooth = smooth_data_gaussian(g_vals, n_vals, sigma_g)
    
    # Second derivative using central differences
    # d²f/dx² ≈ (f[i+1] - 2*f[i] + f[i-1]) / h²
    d2n_dg2 = np.zeros_like(g_vals)
    
    for i in range(1, len(g_vals) - 1):
        h_plus = g_vals[i+1] - g_vals[i]
        h_minus = g_vals[i] - g_vals[i-1]
        h_avg = (h_plus + h_minus) / 2.0
        
        d2n_dg2[i] = (n_smooth[i+1] - 2*n_smooth[i] + n_smooth[i-1]) / h_avg**2
    
    # Use forward/backward at edges
    d2n_dg2[0] = d2n_dg2[1]
    d2n_dg2[-1] = d2n_dg2[-2]
    
    return g_vals, d2n_dg2


# ============================================================================
# Plot 1: n_∞(g) Simulation vs Analytical
# ============================================================================

def generate_ninf_vs_g_plot(df, output_file):
    """
    Generate n_∞(g) plot with all L values + analytical curves.
    Style matches live_progress.png.
    """
    print("\n" + "="*80)
    print("PLOT 1: n_∞(g) Simulation vs Analytical")
    print("="*80)
    
    # Filter to gamma >= 0.1
    df_filtered = df[df['gamma'] >= 0.1].copy()
    print(f"Data points with γ≥0.1: {len(df_filtered)}")
    
    # Filter unconverged runs
    df_good = df_filtered[df_filtered['converged'] == 1].copy()
    print(f"Converged runs: {len(df_good)} / {len(df_filtered)} ({100*len(df_good)/len(df_filtered):.1f}%)")
    
    if len(df_good) == 0:
        print("⚠️  No converged data available!")
        return
    
    # Get unique L values
    L_values = sorted(df_good['L'].unique())
    print(f"L values: {L_values}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Analytical curves
    g_grid = np.logspace(-2, 1.5, 500)
    
    # Thermodynamic limit
    n_exact = np.array([integral_expr(g) for g in g_grid])
    ax.plot(g_grid, n_exact, 'k-', linewidth=2.5, label='L→∞ (exact)', zorder=10)
    
    # Finite-L analytical curves
    colors_L = plt.cm.viridis(np.linspace(0, 0.8, len(L_values)))
    
    for i, L in enumerate(L_values):
        n_L = np.array([sum_pbc(g, L) for g in g_grid])
        ax.plot(g_grid, n_L, '--', color=colors_L[i], linewidth=1.5, 
                alpha=0.6, label=f'L={L} (analytical)')
    
    # Simulation points
    for i, L in enumerate(L_values):
        df_L = df_good[df_good['L'] == L]
        ax.scatter(df_L['g'], df_L['n_inf_sim'], s=60, marker='o', 
                  edgecolor='black', linewidth=1.2, color=colors_L[i],
                  alpha=0.9, zorder=20, label=f'L={L} (simulation)')
    
    ax.set_xscale('log')
    ax.set_xlabel('g = γ/(4J)', fontsize=12, fontweight='bold')
    ax.set_ylabel('n_∞(g)', fontsize=12, fontweight='bold')
    ax.set_title('Steady-State Occupation: Simulation vs Analytical', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================================
# Plot 2: First Derivative dn_∞/dg
# ============================================================================

def generate_first_derivative_plot(df, output_file):
    """
    Generate dn_∞/dg plot for each L separately.
    Uses σ_smooth = 0.02 in g-space.
    """
    print("\n" + "="*80)
    print("PLOT 2: First Derivative dn_∞/dg (σ=0.02)")
    print("="*80)
    
    # Filter data
    df_filtered = df[df['gamma'] >= 0.1].copy()
    df_good = df_filtered[df_filtered['converged'] == 1].copy()
    
    if len(df_good) == 0:
        print("⚠️  No converged data available!")
        return
    
    L_values = sorted(df_good['L'].unique())
    print(f"Computing derivatives for L: {L_values}")
    
    # Create subplots (one per L)
    n_L = len(L_values)
    fig, axes = plt.subplots(n_L, 1, figsize=(12, 4*n_L), sharex=True)
    if n_L == 1:
        axes = [axes]
    
    colors_L = plt.cm.viridis(np.linspace(0, 0.8, n_L))
    
    for i, (L, ax) in enumerate(zip(L_values, axes)):
        df_L = df_good[df_good['L'] == L].copy()
        df_L = df_L.sort_values('g')
        
        g_vals = df_L['g'].values
        n_vals = df_L['n_inf_sim'].values
        
        print(f"  L={L}: {len(g_vals)} points")
        
        if len(g_vals) < 5:
            print(f"    ⚠️  Too few points, skipping")
            continue
        
        # Compute derivative with σ=0.02
        g_deriv, dn_dg = compute_first_derivative(g_vals, n_vals, sigma_g=0.02)
        
        # Plot
        ax.plot(g_deriv, dn_dg, '-', color=colors_L[i], linewidth=2, 
                label=f'L={L}', alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5,
                   label='g=1')
        
        ax.set_ylabel('dn_∞/dg', fontsize=11, fontweight='bold')
        ax.set_title(f'First Derivative (L={L}, σ_smooth=0.02)', 
                     fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('g = γ/(4J)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================================
# Plot 3: Second Derivative d²n_∞/dg²
# ============================================================================

def generate_second_derivative_plot(df, output_file):
    """
    Generate d²n_∞/dg² plot for each L separately.
    Uses σ_smooth = 0.03 in g-space.
    """
    print("\n" + "="*80)
    print("PLOT 3: Second Derivative d²n_∞/dg² (σ=0.03)")
    print("="*80)
    
    # Filter data
    df_filtered = df[df['gamma'] >= 0.1].copy()
    df_good = df_filtered[df_filtered['converged'] == 1].copy()
    
    if len(df_good) == 0:
        print("⚠️  No converged data available!")
        return
    
    L_values = sorted(df_good['L'].unique())
    print(f"Computing second derivatives for L: {L_values}")
    
    # Create subplots (one per L)
    n_L = len(L_values)
    fig, axes = plt.subplots(n_L, 1, figsize=(12, 4*n_L), sharex=True)
    if n_L == 1:
        axes = [axes]
    
    colors_L = plt.cm.viridis(np.linspace(0, 0.8, n_L))
    
    for i, (L, ax) in enumerate(zip(L_values, axes)):
        df_L = df_good[df_good['L'] == L].copy()
        df_L = df_L.sort_values('g')
        
        g_vals = df_L['g'].values
        n_vals = df_L['n_inf_sim'].values
        
        print(f"  L={L}: {len(g_vals)} points")
        
        if len(g_vals) < 5:
            print(f"    ⚠️  Too few points, skipping")
            continue
        
        # Compute second derivative with σ=0.03
        g_deriv2, d2n_dg2 = compute_second_derivative(g_vals, n_vals, sigma_g=0.03)
        
        # Plot
        ax.plot(g_deriv2, d2n_dg2, '-', color=colors_L[i], linewidth=2,
                label=f'L={L}', alpha=0.8)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5,
                   label='g=1')
        
        ax.set_ylabel('d²n_∞/dg²', fontsize=11, fontweight='bold')
        ax.set_title(f'Second Derivative (L={L}, σ_smooth=0.03)', 
                     fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('g = γ/(4J)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate enhanced n_∞(g) analysis plots')
    parser.add_argument('--csv', type=str, default=None,
                       help='Path to CSV data file (auto-detects latest if not provided)')
    args = parser.parse_args()
    
    # Find CSV file
    if args.csv:
        csv_file = Path(args.csv)
    else:
        # Auto-detect latest
        csv_files = list(RESULTS_DIR.glob('ninf_data_*.csv'))
        if not csv_files:
            print("❌ No CSV files found in results/ninf_scan/")
            return
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    if not csv_file.exists():
        print(f"❌ File not found: {csv_file}")
        return
    
    print("="*80)
    print("ENHANCED n_∞(g) ANALYSIS")
    print("="*80)
    print(f"Input CSV: {csv_file}")
    print(f"Output directory: {RESULTS_DIR}")
    print()
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Total data points: {len(df)}")
    print(f"Gamma range: [{df['gamma'].min():.3f}, {df['gamma'].max():.3f}]")
    print(f"g range: [{df['g'].min():.3f}, {df['g'].max():.3f}]")
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate plots
    output_ninf = RESULTS_DIR / f"ninf_vs_g__enhanced__{timestamp}.png"
    output_deriv1 = RESULTS_DIR / f"first_derivative__per_L__{timestamp}.png"
    output_deriv2 = RESULTS_DIR / f"second_derivative__per_L__{timestamp}.png"
    
    generate_ninf_vs_g_plot(df, output_ninf)
    generate_first_derivative_plot(df, output_deriv1)
    generate_second_derivative_plot(df, output_deriv2)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {output_ninf.name}")
    print(f"  2. {output_deriv1.name}")
    print(f"  3. {output_deriv2.name}")
    print()


if __name__ == '__main__':
    main()
