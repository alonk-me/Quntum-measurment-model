#!/usr/bin/env python3
"""Estimate critical point γc from susceptibility peak analysis.

This script analyzes susceptibility scan results to:
1. Find peak locations γ_peak(L) for each system size
2. Perform finite-size extrapolation to estimate γc
3. Generate diagnostic plots
4. Save results to JSON

Usage:
    python scripts/estimate_gamma_c.py results/susceptibility_scan/chi_n_results.csv
    
    # With options
    python scripts/estimate_gamma_c.py chi_n_results.csv --L-min 17 --model power --output gamma_c_report.json

Output:
    - JSON file with γc estimate and fit parameters
    - Terminal output with summary
    - Optional diagnostic plots

Author: Quantum Measurement Model
Date: 2026-01-10
"""

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_measurement.analysis.critical_point import find_chi_peak, estimate_gamma_c


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Estimate γc from susceptibility peaks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('csv_file', type=Path,
                        help='Input CSV file with susceptibility data')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output JSON file (default: csv_file_gamma_c.json)')
    parser.add_argument('--L-min', type=int, default=17,
                        help='Minimum L for extrapolation')
    parser.add_argument('--model', choices=['linear', 'power'], default='linear',
                        help='Finite-size scaling model')
    parser.add_argument('--peak-method', choices=['max', 'gaussian', 'spline'],
                        default='max',
                        help='Peak finding method')
    parser.add_argument('--plot', action='store_true',
                        help='Generate diagnostic plots')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    print(f"  Loaded {len(df)} data points")
    
    # Get unique L values
    L_values = sorted(df['L'].unique())
    print(f"  System sizes: {L_values}")
    
    # Find peaks for each L
    print("\nFinding susceptibility peaks...")
    peaks = {}
    
    for L in L_values:
        df_L = df[df['L'] == L].sort_values('gamma')
        gamma = df_L['gamma'].values
        chi_n = df_L['chi_n'].values
        
        gamma_peak, error = find_chi_peak(
            gamma,
            chi_n,
            method=args.peak_method,
            smooth=True,
            error_estimate=True
        )
        
        peaks[L] = {
            'gamma_peak': gamma_peak,
            'error': error
        }
        
        print(f"  L={L:3d}: γ_peak = {gamma_peak:.6f} ± {error:.6f}")
    
    # Prepare data for extrapolation
    L_fit = np.array([L for L in L_values if L >= args.L_min])
    gamma_peaks = np.array([peaks[L]['gamma_peak'] for L in L_fit])
    gamma_errors = np.array([peaks[L]['error'] for L in L_fit])
    
    print(f"\nUsing L >= {args.L_min} for extrapolation:")
    print(f"  L values: {L_fit.tolist()}")
    
    # Perform extrapolation
    print(f"\nExtrapolating with {args.model} model...")
    result = estimate_gamma_c(
        L_fit,
        gamma_peaks,
        gamma_peak_errors=gamma_errors,
        model=args.model,
        L_min=args.L_min
    )
    
    # Print results
    print("\n" + "="*70)
    print("CRITICAL POINT ESTIMATE")
    print("="*70)
    print(f"γc = {result['gamma_c']:.6f} ± {result['gamma_c_error']:.6f}")
    print(f"gc = {result['gamma_c']/4.0:.6f} ± {result['gamma_c_error']/4.0:.6f}")
    print(f"\nFit quality (R²): {result['r_squared']:.6f}")
    print(f"Model: {args.model}")
    print("\nFit parameters:")
    for key, value in result['fit_params'].items():
        if key in result['fit_params_cov']:
            idx = list(result['fit_params'].keys()).index(key)
            error = np.sqrt(result['fit_params_cov'][idx, idx])
            print(f"  {key} = {value:.6f} ± {error:.6f}")
        else:
            print(f"  {key} = {value:.6f}")
    
    print(f"\nResiduals:")
    for L, res in zip(result['L_used'], result['residuals']):
        print(f"  L={L:3d}: {res:+.6f}")
    
    # Prepare output
    output_data = {
        'gamma_c': float(result['gamma_c']),
        'gamma_c_error': float(result['gamma_c_error']),
        'gc': float(result['gamma_c'] / 4.0),
        'gc_error': float(result['gamma_c_error'] / 4.0),
        'model': args.model,
        'peak_method': args.peak_method,
        'L_min': args.L_min,
        'fit_params': {k: float(v) for k, v in result['fit_params'].items()},
        'r_squared': float(result['r_squared']),
        'peaks': {
            int(L): {
                'gamma_peak': float(peaks[L]['gamma_peak']),
                'error': float(peaks[L]['error'])
            }
            for L in L_values
        },
        'L_used_for_fit': L_fit.tolist(),
        'residuals': {
            int(L): float(res)
            for L, res in zip(result['L_used'], result['residuals'])
        }
    }
    
    # Save to JSON
    if args.output is None:
        output_file = args.csv_file.with_name(
            args.csv_file.stem + '_gamma_c.json'
        )
    else:
        output_file = args.output
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved results to {output_file}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating diagnostic plots...")
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: gamma_peak vs 1/L
            L_plot = np.array(L_values)
            gamma_plot = np.array([peaks[L]['gamma_peak'] for L in L_plot])
            
            ax1.scatter(1/L_plot, gamma_plot, s=100, marker='o', label='Data')
            ax1.axhline(result['gamma_c'], color='r', linestyle='--', 
                       label=f'γc = {result["gamma_c"]:.4f}')
            
            # Plot fit line
            if args.model == 'linear':
                L_fit_plot = np.linspace(L_fit.min(), L_fit.max(), 100)
                gamma_fit_plot = result['fit_params']['gamma_c'] - \
                                result['fit_params']['a'] / L_fit_plot
                ax1.plot(1/L_fit_plot, gamma_fit_plot, 'r-', alpha=0.5, label='Fit')
            
            ax1.set_xlabel('1/L')
            ax1.set_ylabel('γ_peak')
            ax1.set_title('Finite-Size Scaling')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Residuals
            ax2.scatter(result['L_used'], result['residuals'], s=100, marker='o')
            ax2.axhline(0, color='r', linestyle='--')
            ax2.set_xlabel('L')
            ax2.set_ylabel('Residual')
            ax2.set_title('Fit Residuals')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_file.with_suffix('.png')
            plt.savefig(plot_file, dpi=300)
            print(f"✓ Saved plot to {plot_file}")
            
        except ImportError:
            print("  Warning: matplotlib not available, skipping plots")


if __name__ == '__main__':
    main()
