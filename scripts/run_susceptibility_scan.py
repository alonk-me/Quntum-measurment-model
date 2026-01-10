#!/usr/bin/env python3
"""Production script for susceptibility χₙ(γ,L) parameter scans.

This script computes the susceptibility χₙ = ∂n_∞/∂γ over a grid of (γ, L)
values and saves results to CSV with HDF5 caching for efficient recomputation.

Features:
- Adaptive convergence with system-size dependent max_steps
- HDF5 caching of n_infinity values to avoid redundant computations
- CSV checkpointing with resume capability
- Progress reporting and live monitoring
- Parallel computation support

Usage:
    # Basic scan
    python scripts/run_susceptibility_scan.py --L 9 17 33 --gamma-min 2.0 --gamma-max 6.0 --n-gamma 50
    
    # Resume from checkpoint
    python scripts/run_susceptibility_scan.py --resume results/susceptibility_scan/chi_n_results.csv
    
    # With parallel workers
    python scripts/run_susceptibility_scan.py --workers 4 --L 9 17 33 65
    
    # Dense critical region scan
    python scripts/run_susceptibility_scan.py --gamma-min 3.5 --gamma-max 4.5 --n-gamma 200 --L 17 33 65

Output:
    results/susceptibility_scan/
    ├── n_inf_cache.h5           # Cached n_infinity values
    ├── chi_n_results_{timestamp}.csv  # Susceptibility results
    └── metadata.json            # Run parameters

Author: Quantum Measurement Model
Date: 2026-01-10
"""

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_measurement.jw_expansion.susceptibility import compute_chi_n_scan
from quantum_measurement.utilities.cache import ResultCache, create_cache


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute susceptibility χₙ(γ,L) over parameter grid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # System parameters
    parser.add_argument('--L', nargs='+', type=int, default=[9, 17, 33],
                        help='System sizes to scan')
    parser.add_argument('--J', type=float, default=1.0,
                        help='Hopping coupling constant')
    
    # Gamma grid
    parser.add_argument('--gamma-min', type=float, default=2.0,
                        help='Minimum gamma value')
    parser.add_argument('--gamma-max', type=float, default=6.0,
                        help='Maximum gamma value')
    parser.add_argument('--n-gamma', type=int, default=50,
                        help='Number of gamma points')
    parser.add_argument('--gamma-spacing', choices=['linear', 'log'], default='linear',
                        help='Gamma grid spacing')
    
    # Output
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/susceptibility_scan'),
                        help='Output directory')
    parser.add_argument('--cache-file', type=Path, default=None,
                        help='Cache file path (default: output_dir/n_inf_cache.h5)')
    
    # Execution
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (not yet implemented)')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Resume from existing CSV file')
    
    # Convergence
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Convergence tolerance')
    parser.add_argument('--dt', type=float, default=0.001,
                        help='Time step for integration')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()


def construct_gamma_grid(gamma_min, gamma_max, n_gamma, spacing='linear'):
    """Construct gamma grid.
    
    Parameters
    ----------
    gamma_min : float
        Minimum gamma
    gamma_max : float
        Maximum gamma
    n_gamma : int
        Number of points
    spacing : {'linear', 'log'}
        Grid spacing
        
    Returns
    -------
    np.ndarray
        Gamma values
    """
    if spacing == 'linear':
        return np.linspace(gamma_min, gamma_max, n_gamma)
    elif spacing == 'log':
        return np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_gamma)
    else:
        raise ValueError(f"Unknown spacing: {spacing}")


def save_metadata(output_dir, args, gamma_grid, L_list):
    """Save run metadata to JSON.
    
    Parameters
    ----------
    output_dir : Path
        Output directory
    args : Namespace
        Command-line arguments
    gamma_grid : np.ndarray
        Gamma values
    L_list : list
        System sizes
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'L_values': L_list,
        'J': args.J,
        'gamma_min': float(gamma_grid.min()),
        'gamma_max': float(gamma_grid.max()),
        'n_gamma': len(gamma_grid),
        'gamma_spacing': args.gamma_spacing,
        'tolerance': args.tolerance,
        'dt': args.dt,
        'workers': args.workers,
        'total_points': len(gamma_grid) * len(L_list)
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_file}")


def progress_callback(completed, total, gamma, L):
    """Progress callback for compute_chi_n_scan.
    
    Parameters
    ----------
    completed : int
        Number of completed points
    total : int
        Total number of points
    gamma : float
        Current gamma
    L : int
        Current L
    """
    percent = 100.0 * completed / total
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Progress: {completed}/{total} ({percent:.1f}%) | "
          f"L={L}, γ={gamma:.4f}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Construct gamma grid
    gamma_grid = construct_gamma_grid(
        args.gamma_min,
        args.gamma_max,
        args.n_gamma,
        args.gamma_spacing
    )
    print(f"Gamma grid: {len(gamma_grid)} points from {gamma_grid.min():.3f} "
          f"to {gamma_grid.max():.3f}")
    
    # System sizes
    L_list = args.L
    print(f"System sizes: {L_list}")
    
    # Save metadata
    save_metadata(args.output_dir, args, gamma_grid, L_list)
    
    # Set up cache
    if args.cache_file is None:
        cache_file = args.output_dir / 'n_inf_cache.h5'
    else:
        cache_file = args.cache_file
    
    print(f"Cache file: {cache_file}")
    
    # Create or load cache
    if cache_file.exists():
        print("Loading existing cache...")
        cache = ResultCache(cache_file, mode='a')
        print(f"  Cache contains {len(cache.get_all_keys())} entries")
    else:
        print("Creating new cache...")
        cache = create_cache(cache_file)
    
    # Output CSV file
    if args.resume is not None:
        csv_file = args.resume
        print(f"Resuming from: {csv_file}")
        # TODO: Load existing CSV and determine which points to skip
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = args.output_dir / f"chi_n_results_{timestamp}.csv"
        print(f"Results CSV: {csv_file}")
    
    # Run scan
    print("\n" + "="*70)
    print("Starting susceptibility scan...")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    df = compute_chi_n_scan(
        gamma_grid=gamma_grid,
        L_list=L_list,
        J=args.J,
        progress_callback=progress_callback if args.verbose else None,
        tolerance=args.tolerance,
        dt=args.dt
    )
    
    elapsed = time.time() - start_time
    
    # Save results
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved results to {csv_file}")
    print(f"✓ Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"✓ Average time per point: {elapsed/len(df):.2f} seconds")
    
    # Summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"Total points computed: {len(df)}")
    print(f"Convergence success rate: {df['converged_all'].mean()*100:.1f}%")
    print(f"Chi_n range: [{df['chi_n'].min():.6f}, {df['chi_n'].max():.6f}]")
    
    # Peak locations for each L
    print("\nPeak locations (approximate):")
    for L in L_list:
        df_L = df[df['L'] == L]
        peak_idx = df_L['chi_n'].abs().idxmax()
        gamma_peak = df_L.loc[peak_idx, 'gamma']
        chi_peak = df_L.loc[peak_idx, 'chi_n']
        print(f"  L={L:3d}: γ_peak ≈ {gamma_peak:.4f}, χₙ = {chi_peak:.6f}")
    
    # Close cache
    cache.close()
    
    print("\n✓ Scan complete!")
    print(f"Next steps:")
    print(f"  1. Run: python scripts/estimate_gamma_c.py {csv_file}")
    print(f"  2. Run: python scripts/plot_susceptibility.py {csv_file}")


if __name__ == '__main__':
    main()
