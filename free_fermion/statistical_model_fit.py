"""
statistical_model_fit.py
========================

This module provides statistical analysis functions for comparing numerical
results with analytical models. It includes visualization and statistical
metrics for model validation.

Functions
---------
compare_with_analytical : Compare numerical simulation with analytical solution
print_summary_statistics : Print formatted statistical comparison table
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Tuple


def compare_with_analytical(
        times: np.ndarray,
        values: np.ndarray,
        J: float, L: int,
        show_plot: bool = True,
        Residuals_plot: bool = False
        ) -> Dict[str, Any]:
    """Compare numerical results with analytical solution cos(2Jt).
    
    Parameters
    ----------
    times : np.ndarray
        Time points array
    values : np.ndarray
        Numerical simulation values (can be complex, real part will be used)
    J : float
        Coupling constant for analytical solution
    L : int
        Chain length for labeling
    show_plot : bool, optional
        Whether to display comparison plots, by default True
    Residuals_plot : bool, optional
        Whether to display residuals plot, by default False
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing statistical metrics:
        - correlation: Pearson correlation coefficient
        - p_value: P-value for correlation significance
        - r_squared: Coefficient of determination
        - mse: Mean squared error
        - chi2: Chi-squared statistic
        - chi2_reduced: Reduced chi-squared statistic
    """
    # Analytical solution
    analytical = 2 + np.cos(2 * J * times)
    
    # Ensure we're working with real values
    numerical_real = values.real if np.iscomplexobj(values) else values
    
    # Statistical metrics
    # Chi-squared test
    chi2 = np.sum((numerical_real - analytical)**2 / np.var(analytical))
    chi2_reduced = chi2 / (len(times) - 1)  # degrees of freedom = N - 1
    
    # Correlation coefficient
    correlation, p_value = stats.pearsonr(numerical_real, analytical)
    
    # Mean squared error
    mse = np.mean((numerical_real - analytical)**2)
    
    # R-squared
    ss_res = np.sum((numerical_real - analytical)**2)
    ss_tot = np.sum((analytical - np.mean(analytical))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    if show_plot:
        plt.figure(figsize=(12, 5))
        
        # Left plot: Overlay comparison
        plt.subplot(1, 2, 1)
        plt.plot(times, numerical_real, 'b--', label=f'Numerical (L={L})', linewidth=2)
        plt.plot(times, analytical, 'r-', label=r'Analytical: $\cos(2Jt)$', linewidth=2, alpha=0.8)
        plt.xlabel('Time $t$')
        plt.ylabel(r'$1 + 2\, G_{0,0}(t)$ / $\langle\sigma_1^z(t)\rangle$')
        plt.title(f'Numerical vs Analytical Comparison (L={L})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if Residuals_plot:
            # Right plot: Residuals
            plt.subplot(1, 2, 2)
            residuals = numerical_real - analytical
            plt.plot(times, residuals, 'g-', linewidth=1)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Time $t$')
            plt.ylabel('Residuals (Numerical - Analytical)')
            plt.title(f'Residuals (L={L})')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Print statistical metrics
    print(f"\n=== Statistical Comparison for L={L} ===")
    print(f"Correlation coefficient: {correlation:.6f}")
    print(f"P-value: {p_value:.2e}")
    print(f"R-squared: {r_squared:.6f}")
    print(f"Mean Squared Error: {mse:.2e}")
    print(f"Chi-squared: {chi2:.2f}")
    print(f"Reduced Chi-squared: {chi2_reduced:.4f}")
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'r_squared': r_squared,
        'mse': mse,
        'chi2': chi2,
        'chi2_reduced': chi2_reduced
    }


def print_summary_statistics(stats_dict: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted table comparing statistical metrics across different systems.
    
    Parameters
    ----------
    stats_dict : Dict[str, Dict[str, Any]]
        Dictionary where keys are system labels (e.g., 'L=2', 'L=3') and values
        are statistics dictionaries returned by compare_with_analytical()
    """
    print("\n" + "="*60)
    print("SUMMARY OF STATISTICAL ANALYSIS")
    print("="*60)
    
    # Get system labels
    systems = list(stats_dict.keys())
    
    # Header
    header = f"{'Metric':<20}"
    for system in systems:
        header += f" {system:<15}"
    print(header)
    print("-" * len(header))
    
    # Metrics to display
    metrics = [
        ('R-squared', 'r_squared', '.6f'),
        ('Correlation', 'correlation', '.6f'),
        ('P-value', 'p_value', '.2e'),
        ('MSE', 'mse', '.2e'),
        ('Chi-squared', 'chi2', '.2f'),
        ('Reduced Chi-sq', 'chi2_reduced', '.4f')
    ]
    
    # Print each metric row
    for metric_name, metric_key, format_spec in metrics:
        row = f"{metric_name:<20}"
        for system in systems:
            value = stats_dict[system][metric_key]
            row += f" {value:<15{format_spec}}"
        print(row)
    
    print("="*60)


def create_summary_plot(times_dict: Dict[str, np.ndarray], 
                       values_dict: Dict[str, np.ndarray], 
                       stats_dict: Dict[str, Dict[str, Any]], 
                       J: float) -> None:
    """Create a summary comparison plot for multiple systems.
    
    Parameters
    ----------
    times_dict : Dict[str, np.ndarray]
        Dictionary mapping system labels to time arrays
    values_dict : Dict[str, np.ndarray]
        Dictionary mapping system labels to numerical values
    stats_dict : Dict[str, Dict[str, Any]]
        Dictionary mapping system labels to statistics
    J : float
        Coupling constant for analytical solution
    """
    systems = list(times_dict.keys())
    n_systems = len(systems)
    
    plt.figure(figsize=(7 * n_systems, 6))
    
    for i, system in enumerate(systems):
        plt.subplot(1, n_systems, i + 1)
        
        times = times_dict[system]
        values = values_dict[system]
        stats = stats_dict[system]
        
        # Ensure real values
        numerical_real = values.real if np.iscomplexobj(values) else values
        analytical = np.cos(2 * J * times)
        
        plt.plot(times, numerical_real, 'b-', label=f'Numerical ({system})', linewidth=2)
        plt.plot(times, analytical, 'r--', label=r'Analytical: $\cos(2Jt)$', linewidth=2, alpha=0.8)
        plt.xlabel('Time $t$')
        plt.ylabel(r'$\langle\sigma_1^z(t)\rangle$')
        plt.title(f'{system}: Numerical vs Analytical\n$R^2 = {stats["r_squared"]:.6f}$, $\\chi^2_{{red}} = {stats["chi2_reduced"]:.4f}$')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.show()