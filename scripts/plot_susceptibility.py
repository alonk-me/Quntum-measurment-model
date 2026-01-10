#!/usr/bin/env python3
"""Visualization script for susceptibility analysis results.

This script generates publication-quality plots from susceptibility scan data:
1. χₙ(γ) curves for all system sizes
2. Critical region zoom
3. Finite-size scaling plot (γ_peak vs 1/L)
4. Error analysis plots

Usage:
    python scripts/plot_susceptibility.py results/susceptibility_scan/chi_n_results.csv
    
    # With options
    python scripts/plot_susceptibility.py chi_n_results.csv --output-dir plots/ --dpi 300

Output:
    - chi_n_vs_gamma.png: Main susceptibility curves
    - chi_n_critical_zoom.png: Zoomed view of critical region
    - finite_size_scaling.png: Extrapolation plot
    - error_analysis.png: Error distributions

Author: Quantum Measurement Model
Date: 2026-01-10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_measurement.analysis.critical_point import find_chi_peak


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate susceptibility analysis plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('csv_file', type=Path,
                        help='Input CSV file with susceptibility data')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for plots (default: same as csv_file)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Plot resolution (DPI)')
    parser.add_argument('--gamma-c', type=float, default=None,
                        help='Critical gamma for reference line')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                        help='Output format')
    
    return parser.parse_args()


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6


def plot_chi_n_curves(df, output_file, gamma_c=None, dpi=300):
    """Plot χₙ(γ) for all L values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Susceptibility data
    output_file : Path
        Output file path
    gamma_c : float, optional
        Critical gamma for reference line
    dpi : int
        Resolution
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique L values
    L_values = sorted(df['L'].unique())
    
    # Color map
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_values)))
    
    # Plot each L
    for L, color in zip(L_values, colors):
        df_L = df[df['L'] == L].sort_values('gamma')
        ax.plot(df_L['gamma'], df_L['chi_n'], 
                marker='o', markersize=3, label=f'L={L}',
                color=color, alpha=0.8)
    
    # Add gamma_c reference if provided
    if gamma_c is not None:
        ax.axvline(gamma_c, color='red', linestyle='--', linewidth=2,
                   label=f'γc ≈ {gamma_c:.3f}', alpha=0.7)
    
    ax.set_xlabel('Measurement rate γ')
    ax.set_ylabel('Susceptibility χₙ = ∂n∞/∂γ')
    ax.set_title('Susceptibility vs Measurement Strength')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_file}")


def plot_critical_zoom(df, output_file, gamma_c=None, dpi=300):
    """Plot zoomed view of critical region.
    
    Parameters
    ----------
    df : pd.DataFrame
        Susceptibility data
    output_file : Path
        Output file path
    gamma_c : float, optional
        Critical gamma (used to set zoom range)
    dpi : int
        Resolution
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine zoom range
    if gamma_c is not None:
        gamma_min = gamma_c - 1.0
        gamma_max = gamma_c + 1.0
    else:
        # Use middle 50% of gamma range
        gamma_all = df['gamma'].values
        gamma_min = np.percentile(gamma_all, 25)
        gamma_max = np.percentile(gamma_all, 75)
    
    # Filter data
    df_zoom = df[(df['gamma'] >= gamma_min) & (df['gamma'] <= gamma_max)]
    
    # Get unique L values
    L_values = sorted(df_zoom['L'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_values)))
    
    # Plot each L
    for L, color in zip(L_values, colors):
        df_L = df_zoom[df_zoom['L'] == L].sort_values('gamma')
        
        # Plot with error bars if available
        if 'chi_n_error' in df_L.columns:
            ax.errorbar(df_L['gamma'], df_L['chi_n'],
                       yerr=df_L['chi_n_error'],
                       marker='o', markersize=4, label=f'L={L}',
                       color=color, alpha=0.8, capsize=3)
        else:
            ax.plot(df_L['gamma'], df_L['chi_n'],
                   marker='o', markersize=4, label=f'L={L}',
                   color=color, alpha=0.8)
    
    # Add gamma_c reference
    if gamma_c is not None:
        ax.axvline(gamma_c, color='red', linestyle='--', linewidth=2,
                   label=f'γc ≈ {gamma_c:.3f}', alpha=0.7)
    
    ax.set_xlabel('Measurement rate γ')
    ax.set_ylabel('Susceptibility χₙ')
    ax.set_title('Critical Region (Zoomed)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_file}")


def plot_finite_size_scaling(df, output_file, gamma_c=None, dpi=300):
    """Plot γ_peak vs 1/L with extrapolation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Susceptibility data
    output_file : Path
        Output file path
    gamma_c : float, optional
        Estimated critical gamma
    dpi : int
        Resolution
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Find peaks for each L
    L_values = sorted(df['L'].unique())
    gamma_peaks = []
    errors = []
    
    for L in L_values:
        df_L = df[df['L'] == L].sort_values('gamma')
        gamma = df_L['gamma'].values
        chi_n = df_L['chi_n'].values
        
        gamma_peak, error = find_chi_peak(gamma, chi_n, 
                                         method='max',
                                         error_estimate=True)
        gamma_peaks.append(gamma_peak)
        errors.append(error)
    
    L_values = np.array(L_values)
    gamma_peaks = np.array(gamma_peaks)
    errors = np.array(errors)
    
    # Plot data
    ax.errorbar(1/L_values, gamma_peaks, yerr=errors,
                marker='o', markersize=8, linestyle='none',
                color='blue', capsize=5, label='Peaks')
    
    # Add gamma_c horizontal line
    if gamma_c is not None:
        ax.axhline(gamma_c, color='red', linestyle='--', linewidth=2,
                   label=f'γc ≈ {gamma_c:.3f}', alpha=0.7)
        
        # Try to fit and plot extrapolation
        try:
            from quantum_measurement.analysis.critical_point import estimate_gamma_c as fit_gamma_c
            
            # Filter L >= 17 for fit
            mask = L_values >= 17
            result = fit_gamma_c(L_values[mask], gamma_peaks[mask],
                               gamma_peak_errors=errors[mask],
                               model='linear', L_min=17)
            
            # Plot fit line
            L_fit_plot = np.linspace(L_values[mask].min(), L_values[mask].max(), 100)
            gamma_fit = result['fit_params']['gamma_c'] - \
                       result['fit_params']['a'] / L_fit_plot
            ax.plot(1/L_fit_plot, gamma_fit, 'r-', alpha=0.5,
                   label=f"Fit: γc = {result['gamma_c']:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not fit extrapolation: {e}")
    
    ax.set_xlabel('1/L')
    ax.set_ylabel('γ_peak')
    ax.set_title('Finite-Size Scaling')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to start from 0
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_file}")


def plot_error_analysis(df, output_file, dpi=300):
    """Plot error distributions and convergence statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Susceptibility data
    output_file : Path
        Output file path
    dpi : int
        Resolution
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Error vs gamma
    L_values = sorted(df['L'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_values)))
    
    for L, color in zip(L_values, colors):
        df_L = df[df['L'] == L].sort_values('gamma')
        if 'chi_n_error' in df_L.columns:
            ax1.semilogy(df_L['gamma'], df_L['chi_n_error'],
                        marker='o', markersize=3, label=f'L={L}',
                        color=color, alpha=0.7)
    
    ax1.set_xlabel('γ')
    ax1.set_ylabel('χₙ error')
    ax1.set_title('Error vs Gamma')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convergence rate
    if 'converged_all' in df.columns:
        convergence_rate = []
        for L in L_values:
            df_L = df[df['L'] == L]
            rate = df_L['converged_all'].mean() * 100
            convergence_rate.append(rate)
        
        ax2.bar(range(len(L_values)), convergence_rate,
                tick_label=[f'L={L}' for L in L_values],
                color='steelblue', alpha=0.7)
        ax2.set_ylabel('Convergence rate (%)')
        ax2.set_title('Convergence Success Rate')
        ax2.set_ylim([0, 105])
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Chi_n magnitude vs L
    for L, color in zip(L_values, colors):
        df_L = df[df['L'] == L].sort_values('gamma')
        ax3.plot(df_L['gamma'], np.abs(df_L['chi_n']),
                marker='o', markersize=3, label=f'L={L}',
                color=color, alpha=0.7)
    
    ax3.set_xlabel('γ')
    ax3.set_ylabel('|χₙ|')
    ax3.set_title('Susceptibility Magnitude')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Step size distribution
    if 'dg_used' in df.columns:
        dg_values = df['dg_used'].values
        ax4.hist(dg_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('dg (step size)')
        ax4.set_ylabel('Count')
        ax4.set_title('Step Size Distribution')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {output_file}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set up plot style
    setup_plot_style()
    
    # Load data
    print(f"Loading data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    print(f"  Loaded {len(df)} data points")
    print(f"  System sizes: {sorted(df['L'].unique())}")
    
    # Set output directory
    if args.output_dir is None:
        output_dir = args.csv_file.parent
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Output directory: {output_dir}")
    
    # Load gamma_c if available
    gamma_c = args.gamma_c
    gamma_c_file = args.csv_file.with_name(args.csv_file.stem + '_gamma_c.json')
    if gamma_c is None and gamma_c_file.exists():
        with open(gamma_c_file, 'r') as f:
            data = json.load(f)
            gamma_c = data.get('gamma_c')
            print(f"  Loaded γc = {gamma_c:.6f} from {gamma_c_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Main chi_n curves
    output_file = output_dir / f"chi_n_vs_gamma.{args.format}"
    plot_chi_n_curves(df, output_file, gamma_c=gamma_c, dpi=args.dpi)
    
    # 2. Critical region zoom
    output_file = output_dir / f"chi_n_critical_zoom.{args.format}"
    plot_critical_zoom(df, output_file, gamma_c=gamma_c, dpi=args.dpi)
    
    # 3. Finite-size scaling
    output_file = output_dir / f"finite_size_scaling.{args.format}"
    plot_finite_size_scaling(df, output_file, gamma_c=gamma_c, dpi=args.dpi)
    
    # 4. Error analysis
    output_file = output_dir / f"error_analysis.{args.format}"
    plot_error_analysis(df, output_file, dpi=args.dpi)
    
    print("\n✓ All plots generated successfully!")


if __name__ == '__main__':
    main()
