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
from scipy.signal import savgol_filter
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

# Critical-region plotting defaults (match legacy derivative_analysis)
CRITICAL_G_MIN = 0.6
CRITICAL_G_MAX = 1.4
MIN_POINTS_FIRST = 5
MIN_POINTS_SECOND = 7

SMOOTHING_PRESETS = {
    "aggressive": {
        "use_resample": True,
        "resample_points": 400,
        "sigma_idx_first": 1.0,
        "sigma_idx_second": 1.5,
        "apply_savgol": True,
        "savgol_window": 21,
        "savgol_poly": 3,
        "max_first_alert": 2.0,
        "max_second_alert": 6.0,
    },
    "moderate": {
        "use_resample": True,
        "resample_points": 260,
        "sigma_idx_first": 0.6,
        "sigma_idx_second": 1.0,
        "apply_savgol": True,
        "savgol_window": 15,
        "savgol_poly": 3,
        "max_first_alert": 3.0,
        "max_second_alert": 8.0,
    },
    "raw": {
        "use_resample": False,
        "resample_points": None,
        "sigma_idx_first": 0.3,
        "sigma_idx_second": 0.6,
        "apply_savgol": False,
        "savgol_window": None,
        "savgol_poly": None,
        "max_first_alert": 4.0,
        "max_second_alert": 10.0,
    },
}

# ============================================================================
# Smoothing and Derivative Utilities
# ============================================================================

def smooth_data_gaussian(g_vals, n_vals, sigma_g=0.02, mode='g', savgol_config=None):
    """
    Apply Gaussian smoothing in g-space.
    
    Parameters
    ----------
    g_vals : ndarray
        g values (must be sorted)
    n_vals : ndarray
        n_∞ values
    sigma_g : float
        Smoothing width (interpreted per `mode`)
    mode : str
        'g'  -> sigma_g specified in g units (converted to index space)
        'index' -> sigma_g already represents Gaussian sigma in index units
        
    Returns
    -------
    n_smooth : ndarray
        Smoothed n_∞ values
    """
    if len(n_vals) < 2:
        return n_vals

    if mode not in {'g', 'index'}:
        raise ValueError(f"Unknown smoothing mode: {mode}")

    if mode == 'g':
        diffs = np.diff(g_vals)
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) == 0:
            avg_dg = 1.0
        else:
            avg_dg = np.median(positive_diffs)
        avg_dg = max(avg_dg, 1e-9)
        sigma_idx = sigma_g / avg_dg
    else:
        sigma_idx = sigma_g

    n_smooth = gaussian_filter1d(n_vals, sigma=sigma_idx)

    if savgol_config:
        window = savgol_config.get('window', 11)
        poly = savgol_config.get('poly', 3)
        # Ensure odd window <= len(n_vals)
        window = max(poly + 2, window)
        if window % 2 == 0:
            window += 1
        if window > len(n_vals):
            window = len(n_vals) if len(n_vals) % 2 == 1 else len(n_vals) - 1
        if window >= poly + 2 and window >= 3:
            try:
                n_smooth = savgol_filter(n_smooth, window_length=window,
                                         polyorder=poly, mode='interp')
            except ValueError:
                pass
    return n_smooth


def compute_first_derivative(g_vals, n_vals, sigma_g=0.02, smooth_mode='g',
                             savgol_config=None):
    """
    Compute dn/dg using central differences with Gaussian smoothing.
    
    Parameters
    ----------
    g_vals : ndarray
        g values (must be sorted)
    n_vals : ndarray
        n_∞ values
    sigma_g : float
        Smoothing width interpreted per `smooth_mode`
    smooth_mode : str
        Passed to smooth_data_gaussian (default 'g')
        
    Returns
    -------
    g_deriv, dn_dg : ndarrays
    """
    # Smooth first
    n_smooth = smooth_data_gaussian(g_vals, n_vals, sigma_g, mode=smooth_mode,
                                    savgol_config=savgol_config)
    
    # Central differences (2nd order accurate)
    dn_dg = np.zeros_like(g_vals)
    dn_dg[1:-1] = (n_smooth[2:] - n_smooth[:-2]) / (g_vals[2:] - g_vals[:-2])
    
    # Forward/backward at edges
    dn_dg[0] = (n_smooth[1] - n_smooth[0]) / (g_vals[1] - g_vals[0])
    dn_dg[-1] = (n_smooth[-1] - n_smooth[-2]) / (g_vals[-1] - g_vals[-2])
    
    return g_vals, dn_dg


def compute_second_derivative(g_vals, n_vals, sigma_g=0.03, smooth_mode='g',
                              savgol_config=None):
    """
    Compute d²n/dg² using central differences with Gaussian smoothing.
    
    Parameters
    ----------
    g_vals : ndarray
        g values (must be sorted)
    n_vals : ndarray
        n_∞ values
    sigma_g : float
        Smoothing width interpreted per `smooth_mode`
    smooth_mode : str
        Passed to smooth_data_gaussian (default 'g')
        
    Returns
    -------
    g_deriv2, d2n_dg2 : ndarrays
    """
    # Smooth with larger sigma for second derivative
    n_smooth = smooth_data_gaussian(g_vals, n_vals, sigma_g, mode=smooth_mode,
                                    savgol_config=savgol_config)
    
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


def get_savgol_config(preset_cfg):
    if not preset_cfg.get('apply_savgol'):
        return None
    return {
        'window': preset_cfg.get('savgol_window', 11),
        'poly': preset_cfg.get('savgol_poly', 3)
    }


def prepare_critical_series(g_vals, n_vals, preset_cfg,
                             g_min=CRITICAL_G_MIN, g_max=CRITICAL_G_MAX):
    """Return (g, n) arrays ready for differentiation based on preset."""
    if len(g_vals) < 2:
        return None

    order = np.argsort(g_vals)
    g_sorted = g_vals[order]
    n_sorted = n_vals[order]

    mask = (g_sorted >= g_min) & (g_sorted <= g_max)
    g_sorted = g_sorted[mask]
    n_sorted = n_sorted[mask]

    if len(g_sorted) < 2:
        return None

    if preset_cfg['use_resample']:
        if g_sorted[0] > g_min or g_sorted[-1] < g_max:
            return None
        num_points = preset_cfg['resample_points']
        g_target = np.linspace(g_min, g_max, num_points)
        n_interp = np.interp(g_target, g_sorted, n_sorted)
        return g_target, n_interp
    else:
        return g_sorted, n_sorted


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

def generate_first_derivative_plot(df, output_file, preset_cfg, preset_name):
    """Reproduce legacy-style derivative analysis with configurable smoothing."""
    print("\n" + "="*80)
    print(f"PLOT 2: First Derivative dn_∞/dg (preset={preset_name})")
    print("="*80)

    df_filtered = df[df['gamma'] >= 0.1].copy()
    df_good = df_filtered[df_filtered['converged'] == 1].copy()

    if len(df_good) == 0:
        print("⚠️  No converged data available!")
        return

    L_values = sorted(df_good['L'].unique())
    print(f"Computing derivatives for L: {L_values}")

    g_zoom = np.linspace(CRITICAL_G_MIN, CRITICAL_G_MAX, 800)
    n_exact_zoom = np.array([integral_expr(g) for g in g_zoom])
    sigma_first = preset_cfg['sigma_idx_first']
    savgol_cfg = get_savgol_config(preset_cfg)
    g_deriv_exact, dn_exact = compute_first_derivative(
        g_zoom, n_exact_zoom, sigma_g=sigma_first, smooth_mode='index',
        savgol_config=savgol_cfg)

    colors_L = plt.cm.viridis(np.linspace(0, 0.85, len(L_values)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax_occ, ax_deriv = axes

    # ------------------------------------------------------------------
    # Panel 1: n_∞(g) in critical region
    # ------------------------------------------------------------------
    ax_occ.plot(g_zoom, n_exact_zoom, 'k-', linewidth=2.5, label='L→∞ (exact)')

    for color, L in zip(colors_L, L_values):
        n_analytical_L = np.array([sum_pbc(g, L) for g in g_zoom])
        ax_occ.plot(g_zoom, n_analytical_L, '--', color=color, linewidth=1.2,
                    alpha=0.6)

        df_L = df_good[(df_good['L'] == L) &
                       (df_good['g'] >= CRITICAL_G_MIN) &
                       (df_good['g'] <= CRITICAL_G_MAX)].copy()

        print(f"  L={L}: {len(df_L)} points in critical window")

        if len(df_L) > 0:
            ax_occ.scatter(df_L['g'], df_L['n_inf_sim'], s=40, marker='o',
                           edgecolor='black', linewidth=0.6, color=color,
                           alpha=0.9, label=f'L={L} (simulation)')

    ax_occ.axvline(1.0, color='red', linestyle=':', linewidth=1.8, alpha=0.5,
                   label='g=1')
    ax_occ.set_ylabel('n_∞(g)', fontweight='bold', fontsize=12)
    ax_occ.set_title('Critical Region Steady-State Occupation',
                     fontweight='bold', fontsize=13)
    ax_occ.set_xlim(CRITICAL_G_MIN, CRITICAL_G_MAX)
    ax_occ.grid(True, alpha=0.3)
    ax_occ.legend(loc='best', fontsize=9, ncol=2)

    # ------------------------------------------------------------------
    # Panel 2: dn_∞/dg (legacy style)
    # ------------------------------------------------------------------
    ax_deriv.plot(g_deriv_exact, dn_exact, 'k-', linewidth=2.3,
                  label='L→∞ (numerical)')

    for color, L in zip(colors_L, L_values):
        df_L = df_good[(df_good['L'] == L) &
                       (df_good['g'] >= CRITICAL_G_MIN) &
                       (df_good['g'] <= CRITICAL_G_MAX)].copy()
        df_L = df_L.sort_values('g')

        if len(df_L) < MIN_POINTS_FIRST:
            print(f"    ⚠️  L={L} skipped (need ≥{MIN_POINTS_FIRST} points)")
            continue

        g_vals = df_L['g'].values
        n_vals = df_L['n_inf_sim'].values
        resampled = prepare_critical_series(g_vals, n_vals, preset_cfg)
        if resampled is None:
            print(f"    ⚠️  L={L} lacks full critical coverage, skipping")
            continue

        g_series, n_series = resampled
        g_deriv, dn_dg = compute_first_derivative(
            g_series, n_series, sigma_g=sigma_first,
            smooth_mode='index', savgol_config=savgol_cfg)

        max_abs = float(np.max(np.abs(dn_dg))) if len(dn_dg) else 0.0
        if max_abs > preset_cfg['max_first_alert']:
            print(f"    ⚠️  L={L} peak |dn/dg|={max_abs:.2f} (check smoothness)")

        ax_deriv.plot(g_deriv, dn_dg, '--', color=color, linewidth=1.6,
                      alpha=0.85, label=f'L={L}')

    ax_deriv.axvline(1.0, color='red', linestyle=':', linewidth=1.8, alpha=0.5)
    ax_deriv.axhline(0, color='gray', linestyle='-', linewidth=0.7, alpha=0.5)
    ax_deriv.set_xlabel('g = γ/(4J)', fontweight='bold', fontsize=12)
    ax_deriv.set_ylabel('dn_∞/dg', fontweight='bold', fontsize=12)
    ax_deriv.set_title(f'Derivative Sensitivity (preset={preset_name})',
                       fontweight='bold', fontsize=13)
    ax_deriv.set_xlim(CRITICAL_G_MIN, CRITICAL_G_MAX)
    ax_deriv.grid(True, alpha=0.3)
    ax_deriv.legend(loc='best', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================================
# Plot 3: Second Derivative d²n_∞/dg²
# ============================================================================

def generate_second_derivative_plot(df, output_file, preset_cfg, preset_name):
    """Critical-region visualization of dn/dg (context) and d²n/dg²."""
    print("\n" + "="*80)
    print(f"PLOT 3: Second Derivative d²n_∞/dg² (preset={preset_name})")
    print("="*80)

    df_filtered = df[df['gamma'] >= 0.1].copy()
    df_good = df_filtered[df_filtered['converged'] == 1].copy()

    if len(df_good) == 0:
        print("⚠️  No converged data available!")
        return

    L_values = sorted(df_good['L'].unique())
    print(f"Computing higher derivatives for L: {L_values}")

    g_zoom = np.linspace(CRITICAL_G_MIN, CRITICAL_G_MAX, 800)
    n_exact_zoom = np.array([integral_expr(g) for g in g_zoom])
    sigma_first = preset_cfg['sigma_idx_first']
    sigma_second = preset_cfg['sigma_idx_second']
    savgol_cfg = get_savgol_config(preset_cfg)
    g_deriv_exact, dn_exact = compute_first_derivative(
        g_zoom, n_exact_zoom, sigma_g=sigma_first, smooth_mode='index',
        savgol_config=savgol_cfg)
    g_d2_exact, d2_exact = compute_second_derivative(
        g_zoom, n_exact_zoom, sigma_g=sigma_second, smooth_mode='index',
        savgol_config=savgol_cfg)

    colors_L = plt.cm.viridis(np.linspace(0, 0.85, len(L_values)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax_first, ax_second = axes

    # Precompute per-L derivatives to reuse between panels
    first_deriv_data = {}
    second_deriv_data = {}

    for color, L in zip(colors_L, L_values):
        df_L = df_good[(df_good['L'] == L) &
                       (df_good['g'] >= CRITICAL_G_MIN) &
                       (df_good['g'] <= CRITICAL_G_MAX)].copy()
        df_L = df_L.sort_values('g')

        if len(df_L) < MIN_POINTS_FIRST:
            print(f"    ⚠️  L={L} skipped (need ≥{MIN_POINTS_FIRST} points)")
            continue

        g_vals = df_L['g'].values
        n_vals = df_L['n_inf_sim'].values
        resampled = prepare_critical_series(g_vals, n_vals, preset_cfg)
        if resampled is None:
            print(f"    ⚠️  L={L} lacks full critical coverage, skipping")
            continue

        g_series, n_series = resampled
        first_deriv_data[L] = compute_first_derivative(
            g_series, n_series, sigma_g=sigma_first,
            smooth_mode='index', savgol_config=savgol_cfg)

        if len(df_L) >= MIN_POINTS_SECOND:
            second_deriv_data[L] = compute_second_derivative(
                g_series, n_series, sigma_g=sigma_second,
                smooth_mode='index', savgol_config=savgol_cfg)
        else:
            print(f"    ⚠️  L={L} lacks dense data for second derivative")

    # ------------------------------------------------------------------
    # Panel 1: dn_∞/dg (context)
    # ------------------------------------------------------------------
    ax_first.plot(g_deriv_exact, dn_exact, 'k-', linewidth=2.3,
                  label='L→∞ (numerical)')

    for color, L in zip(colors_L, L_values):
        if L not in first_deriv_data:
            continue
        g_deriv, dn_dg = first_deriv_data[L]
        max_abs = float(np.max(np.abs(dn_dg))) if len(dn_dg) else 0.0
        if max_abs > preset_cfg['max_first_alert']:
            print(f"    ⚠️  L={L} peak |dn/dg|={max_abs:.2f} (context panel)")
        ax_first.plot(g_deriv, dn_dg, '--', color=color, linewidth=1.6,
                      alpha=0.85, label=f'L={L}')

    ax_first.axvline(1.0, color='red', linestyle=':', linewidth=1.8, alpha=0.5)
    ax_first.axhline(0, color='gray', linestyle='-', linewidth=0.7, alpha=0.5)
    ax_first.set_ylabel('dn_∞/dg', fontweight='bold', fontsize=12)
    ax_first.set_title(f'Derivative Context (preset={preset_name})',
                       fontweight='bold', fontsize=13)
    ax_first.set_xlim(CRITICAL_G_MIN, CRITICAL_G_MAX)
    ax_first.grid(True, alpha=0.3)
    ax_first.legend(loc='best', fontsize=9, ncol=2)

    # ------------------------------------------------------------------
    # Panel 2: d²n_∞/dg²
    # ------------------------------------------------------------------
    ax_second.plot(g_d2_exact, d2_exact, 'k-', linewidth=2.3,
                   label='L→∞ (numerical)')

    for color, L in zip(colors_L, L_values):
        if L not in second_deriv_data:
            continue
        g_vals, d2_vals = second_deriv_data[L]
        max_abs = float(np.max(np.abs(d2_vals))) if len(d2_vals) else 0.0
        if max_abs > preset_cfg['max_second_alert']:
            print(f"    ⚠️  L={L} peak |d²n/dg²|={max_abs:.2f}")
        ax_second.plot(g_vals, d2_vals, '--', color=color, linewidth=1.6,
                       alpha=0.85, label=f'L={L}')

    ax_second.axvline(1.0, color='red', linestyle=':', linewidth=1.8, alpha=0.5)
    ax_second.axhline(0, color='gray', linestyle='-', linewidth=0.7, alpha=0.5)
    ax_second.set_xlabel('g = γ/(4J)', fontweight='bold', fontsize=12)
    ax_second.set_ylabel('d²n_∞/dg²', fontweight='bold', fontsize=12)
    ax_second.set_title(f'Second Derivative (preset={preset_name})',
                        fontweight='bold', fontsize=13)
    ax_second.set_xlim(CRITICAL_G_MIN, CRITICAL_G_MAX)
    ax_second.grid(True, alpha=0.3)
    ax_second.legend(loc='best', fontsize=9, ncol=2)

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
    parser.add_argument('--smooth-mode', type=str, default='moderate',
                        choices=list(SMOOTHING_PRESETS.keys()),
                        help='Choose smoothing preset for derivative plots (default: moderate)')
    parser.add_argument('--extra-smooth-mode', action='append', default=[],
                        choices=list(SMOOTHING_PRESETS.keys()),
                        help='Generate additional derivative plots using this preset (can repeat)')
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

    generated_files = []

    # n_inf plot (preset independent)
    output_ninf = RESULTS_DIR / f"ninf_vs_g__enhanced__{timestamp}.png"
    generate_ninf_vs_g_plot(df, output_ninf)
    generated_files.append(output_ninf.name)

    # Determine presets to render
    preset_order = []
    for preset in [args.smooth_mode] + args.extra_smooth_mode:
        if preset not in preset_order:
            preset_order.append(preset)

    for preset in preset_order:
        cfg = SMOOTHING_PRESETS[preset]
        suffix = f"{preset}__{timestamp}"
        output_deriv1 = RESULTS_DIR / f"first_derivative__{suffix}.png"
        output_deriv2 = RESULTS_DIR / f"second_derivative__{suffix}.png"

        generate_first_derivative_plot(df, output_deriv1, cfg, preset)
        generate_second_derivative_plot(df, output_deriv2, cfg, preset)

        generated_files.append(output_deriv1.name)
        generated_files.append(output_deriv2.name)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    for idx, name in enumerate(generated_files, start=1):
        print(f"  {idx}. {name}")
    print()


if __name__ == '__main__':
    main()
