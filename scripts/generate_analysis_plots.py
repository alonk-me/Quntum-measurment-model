#!/usr/bin/env python3
"""
Post-Processing: Derivative Analysis and Time-Trace Showcase

Generate specialized plots focusing on:
1. dn_∞/dg near critical point g=1
2. Representative time-trace gallery with decaying oscillations

Run after main parameter sweep completes to produce publication-grade figures.

Usage:
    python scripts/generate_analysis_plots.py
    python scripts/generate_analysis_plots.py --csv results/ninf_scan/ninf_data_20260106.csv

Author: Auto-generated for Quantum Measurement Model verification
Date: 2026-01-06
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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from quantum_measurement.jw_expansion.non_hermitian_hat import NonHermitianHatSimulator
from quantum_measurement.jw_expansion.n_infty import sum_pbc, integral_expr
from scipy.optimize import curve_fit

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

RESULTS_DIR = Path(__file__).parent.parent / "results" / "ninf_scan"

# ============================================================================
# Derivative Analysis Near g=1
# ============================================================================

def compute_derivative_numerical(g_vals, n_vals, method='central', sigma=1.0):
    """
    Compute dn/dg using numerical differentiation.
    
    Parameters
    ----------
    g_vals : ndarray
        g values (must be sorted)
    n_vals : ndarray
        n_∞ values
    method : str
        'central', 'forward', or 'savgol'
    sigma : float
        Gaussian smoothing width (for noise reduction)
        
    Returns
    -------
    g_deriv, dn_dg : ndarrays
    """
    # Smooth first to reduce noise
    n_smooth = gaussian_filter1d(n_vals, sigma=sigma)
    
    if method == 'central':
        # Central differences (2nd order accurate)
        dn_dg = np.zeros_like(g_vals)
        dn_dg[1:-1] = (n_smooth[2:] - n_smooth[:-2]) / (g_vals[2:] - g_vals[:-2])
        # Forward/backward at edges
        dn_dg[0] = (n_smooth[1] - n_smooth[0]) / (g_vals[1] - g_vals[0])
        dn_dg[-1] = (n_smooth[-1] - n_smooth[-2]) / (g_vals[-1] - g_vals[-2])
        return g_vals, dn_dg
    
    elif method == 'forward':
        # Forward differences
        dn_dg = np.diff(n_smooth) / np.diff(g_vals)
        g_deriv = g_vals[:-1]
        return g_deriv, dn_dg
    
    else:
        raise ValueError(f"Unknown method: {method}")

def generate_derivative_plot(csv_file, output_dir):
    """Generate dn_∞/dg analysis plot near g=1."""
    
    print("\n" + "="*80)
    print("DERIVATIVE ANALYSIS: dn_∞/dg near g=1")
    print("="*80)
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} data points from {csv_file.name}")
    
    # Focus on critical region
    g_min, g_max = 0.6, 1.4
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Get analytical reference
    g_analytical = np.linspace(g_min, g_max, 1000)
    n_analytical = np.array([integral_expr(g) for g in g_analytical])
    
    # Analytical derivative via numerical diff on fine grid
    g_ana_deriv, dn_ana = compute_derivative_numerical(g_analytical, n_analytical, 
                                                        sigma=0.5)
    
    # ========================================================================
    # Panel 1: n_∞(g) in critical region
    # ========================================================================
    ax = axes[0]
    
    # Analytical
    ax.plot(g_analytical, n_analytical, 'k-', linewidth=2.5, 
            label='L→∞ (exact)', zorder=10)
    
    # Each L
    L_unique = sorted(df['L'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_unique)))
    
    for idx, L in enumerate(L_unique):
        # Analytical for this L
        n_ana_L = np.array([sum_pbc(g, L) for g in g_analytical])
        ax.plot(g_analytical, n_ana_L, '--', color=colors[idx], 
                linewidth=1, alpha=0.4)
        
        # Simulation data in region
        df_L = df[(df['L'] == L) & (df['g'] >= g_min) & (df['g'] <= g_max)]
        if len(df_L) > 0:
            ax.scatter(df_L['g'], df_L['n_inf_sim'], s=40, color=colors[idx],
                       marker='o', edgecolor='black', linewidth=0.5, zorder=20,
                       label=f'L={L}')
    
    ax.axvline(1.0, color='red', linestyle=':', linewidth=2, alpha=0.5,
               label='g = 1')
    ax.set_ylabel('n_∞(g)', fontweight='bold', fontsize=12)
    ax.set_title('Critical Region: Steady-State Occupation', 
                 fontweight='bold', fontsize=13)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Panel 2: dn_∞/dg
    # ========================================================================
    ax = axes[1]
    
    # Analytical derivative
    ax.plot(g_ana_deriv, dn_ana, 'k-', linewidth=2.5, 
            label='L→∞ (numerical deriv)', zorder=10)
    
    # Derivatives for each L
    for idx, L in enumerate(L_unique):
        df_L = df[(df['L'] == L) & (df['g'] >= g_min) & (df['g'] <= g_max)]
        
        if len(df_L) >= 5:  # Need enough points for derivative
            # Sort by g
            df_L_sorted = df_L.sort_values('g')
            g_L = df_L_sorted['g'].values
            n_L = df_L_sorted['n_inf_sim'].values
            
            # Compute derivative
            g_deriv, dn_dg = compute_derivative_numerical(g_L, n_L, sigma=1.0)
            
            ax.plot(g_deriv, dn_dg, '--', color=colors[idx], 
                    linewidth=1.5, alpha=0.7, label=f'L={L}')
    
    ax.axvline(1.0, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('g = γ/(4J)', fontweight='bold', fontsize=12)
    ax.set_ylabel('dn_∞/dg', fontweight='bold', fontsize=12)
    ax.set_title('Derivative: Sensitivity to Measurement Strength', 
                 fontweight='bold', fontsize=13)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = csv_file.stem.split('_')[-1] if '_' in csv_file.stem else 'latest'
    output_file = output_dir / f"derivative_analysis__g-near-1__{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file.name}")
    plt.close()
    
    # Find and report maximum derivative location
    idx_max = np.argmax(np.abs(dn_ana))
    g_max_deriv = g_ana_deriv[idx_max]
    dn_max_deriv = dn_ana[idx_max]
    
    print(f"\nAnalytical derivative analysis:")
    print(f"  Maximum |dn_∞/dg| at g = {g_max_deriv:.4f}")
    print(f"  Value: dn_∞/dg = {dn_max_deriv:.6f}")
    print(f"  Distance from g=1: {abs(g_max_deriv - 1.0):.4f}")

# ============================================================================
# Time-Trace Gallery
# ============================================================================

def exponential_decay_oscillation(t, A, gamma_eff, omega, phi, n_inf):
    """Model: n(t) = n_inf + A * exp(-gamma_eff*t) * cos(omega*t + phi)"""
    return n_inf + A * np.exp(-gamma_eff * t) * np.cos(omega * t + phi)

def generate_time_trace_gallery(output_dir):
    """Generate gallery of time traces for representative γ values."""
    
    print("\n" + "="*80)
    print("TIME-TRACE GALLERY: Decaying Oscillations")
    print("="*80)
    
    # Parameters
    L = 33
    J = 1.0
    gamma_values = [1e-3, 0.1, 0.3, 1.0, 3.0, 30.0]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, gamma in enumerate(gamma_values):
        print(f"\nSimulating γ = {gamma:.3f}...")
        
        g = gamma / (4 * J)
        ax = axes[idx]
        
        # Run simulation
        sim = NonHermitianHatSimulator(
            L=L, J=J, gamma=gamma, dt=0.001, N_steps=50000,
            closed_boundary=True
        )
        
        Q, n_traj, G = sim.simulate_trajectory(return_G_final=True)
        
        # Time axis
        dt = 0.001
        times = np.arange(len(n_traj)) * dt
        
        # Spatially averaged occupation
        n_avg = n_traj.mean(axis=1)
        
        # Analytical steady state
        n_inf_exact = sum_pbc(g, L)
        
        # Plot trajectory
        ax.plot(times, n_avg, linewidth=0.8, alpha=0.8, color='C0',
                label='⟨n(t)⟩ simulation')
        
        # Steady state line
        ax.axhline(n_inf_exact, color='red', linestyle='--', linewidth=1.5,
                   label=f'n_∞ = {n_inf_exact:.4f}', alpha=0.7)
        
        # Try to fit exponential decay (if we have oscillations)
        if gamma < 10.0 and len(times) > 1000:
            try:
                # Fit to latter half to avoid transients
                fit_start = len(times) // 4
                fit_end = min(len(times), fit_start + 10000)
                
                t_fit = times[fit_start:fit_end]
                n_fit = n_avg[fit_start:fit_end]
                
                # Initial guess
                A_guess = (n_fit.max() - n_fit.min()) / 2
                gamma_guess = gamma / 2
                omega_guess = 2 * J
                
                p0 = [A_guess, gamma_guess, omega_guess, 0, n_inf_exact]
                
                popt, _ = curve_fit(exponential_decay_oscillation, t_fit, n_fit, 
                                     p0=p0, maxfev=5000)
                
                A_fit, gamma_fit, omega_fit, phi_fit, n_inf_fit = popt
                
                # Plot fit
                n_fit_curve = exponential_decay_oscillation(t_fit, *popt)
                ax.plot(t_fit, n_fit_curve, 'g--', linewidth=1.2, alpha=0.6,
                        label=f'Fit: Γ={gamma_fit:.3f}')
                
                print(f"  Fit: Γ_eff = {gamma_fit:.4f} (theory: γ/2 = {gamma/2:.4f})")
                print(f"       ω = {omega_fit:.4f} (theory: 2J = {2*J:.4f})")
                
            except Exception as e:
                print(f"  Fit failed: {e}")
        
        # Formatting
        ax.set_xlabel('Time (J⁻¹)', fontsize=10)
        ax.set_ylabel('⟨n⟩', fontsize=10)
        ax.set_title(f'γ = {gamma:.3g} (g = {g:.3g})', fontweight='bold', fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Adjust x-range to show oscillations
        if gamma <= 1.0:
            ax.set_xlim(0, min(30, times[-1]))
        else:
            ax.set_xlim(0, min(5, times[-1]))
    
    plt.suptitle(f'Time-Trace Gallery: Decaying Oscillations (L={L})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f"time_trace_gallery__L-{L:04d}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file.name}")
    plt.close()

# ============================================================================
# Entry Point
# ============================================================================

def main(csv_file=None):
    """Main analysis routine."""
    
    print("="*80)
    print("POST-PROCESSING: DERIVATIVE ANALYSIS & TIME-TRACE SHOWCASE")
    print("="*80)
    
    # Find CSV
    if csv_file is None:
        csv_files = sorted(RESULTS_DIR.glob("ninf_data_*.csv"))
        if not csv_files:
            print("✗ No CSV files found in results/ninf_scan/")
            return
        csv_file = csv_files[-1]  # Most recent
        print(f"Auto-detected: {csv_file.name}")
    else:
        csv_file = Path(csv_file)
    
    if not csv_file.exists():
        print(f"✗ CSV file not found: {csv_file}")
        return
    
    # Generate derivative analysis
    generate_derivative_plot(csv_file, RESULTS_DIR)
    
    # Generate time-trace gallery
    generate_time_trace_gallery(RESULTS_DIR)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS PLOTS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate derivative and time-trace analysis')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV file (auto-detect if not specified)')
    args = parser.parse_args()
    
    main(csv_file=args.csv)
