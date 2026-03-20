"""Distribution analysis tools for quantum measurement entropy production.

This module consolidates the analysis routines that were previously spread
across ``scripts/generate_analysis_plots.py`` and the Kraus operators module
into a single, importable library.

Key functionality
-----------------
* :func:`compute_statistics` – descriptive statistics for a Q array.
* :func:`fit_arrow_of_time` – least-squares fit of Q distribution to the
  Arrow-of-Time density (Eq. 14 of Dressel *et al.*).
* :func:`plot_Q_distribution` – histogram of Q with optional theoretical overlay.
* :func:`plot_mean_Q_vs_theta` – scatter plot of ⟨Q⟩ vs θ with the identity line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QStatistics:
    """Descriptive statistics for an entropy-production distribution.

    Attributes
    ----------
    mean : float
        Sample mean of Q.
    std : float
        Sample standard deviation of Q.
    median : float
        Sample median of Q.
    skewness : float
        Sample skewness of Q.
    n_samples : int
        Number of samples.
    """

    mean: float
    std: float
    median: float
    skewness: float
    n_samples: int


# ---------------------------------------------------------------------------
# Arrow-of-Time probability density (Eq. 14, Dressel et al.)
# ---------------------------------------------------------------------------


def _arrow_of_time_pdf(x: np.ndarray, theta: float) -> np.ndarray:
    """Evaluate the Arrow-of-Time probability density for given θ.

    .. math::

       p(x;\\theta) = \\frac{e^{x/2}}{\\sqrt{\\pi\\theta}\\,(e^x-1)^{1/2}}
                     \\exp\\!\\left[-\\frac{\\theta}{2}
                     -\\frac{(\\operatorname{arccosh}(e^{x/2}))^2}{2\\theta}\\right]

    Parameters
    ----------
    x : numpy.ndarray
        Points at which to evaluate the density.  Should satisfy ``x > 0``.
    theta : float
        Dimensionless parameter θ = T/τ.  Must be positive.

    Returns
    -------
    numpy.ndarray
        Probability density values.  Returns zeros for θ ≤ 0 or x ≤ 0.
    """
    theta = float(theta)
    if theta <= 0:
        return np.zeros_like(x, dtype=float)

    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    valid = x > 0
    xv = x[valid]

    prefactor = np.exp(xv / 2.0) / (np.sqrt(np.pi * theta) * np.sqrt(np.exp(xv) - 1.0))
    exponent = -theta / 2.0 - (np.arccosh(np.exp(xv / 2.0))) ** 2 / (2.0 * theta)
    out[valid] = prefactor * np.exp(exponent)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_statistics(Q_values: Sequence[float]) -> QStatistics:
    """Compute descriptive statistics for an array of entropy-production values.

    Parameters
    ----------
    Q_values : array-like of float
        Entropy-production samples.

    Returns
    -------
    QStatistics
        Descriptive statistics dataclass.
    """
    arr = np.asarray(Q_values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))
    if std > 0 and n > 2:
        skewness = float(np.mean(((arr - mean) / std) ** 3))
    else:
        skewness = 0.0
    return QStatistics(mean=mean, std=std, median=median, skewness=skewness, n_samples=n)


def fit_arrow_of_time(
    Q_values: Sequence[float],
    bins: int = 60,
) -> Tuple[float, float]:
    """Fit an empirical Q distribution to the Arrow-of-Time density.

    Constructs a normalised histogram of ``Q_values`` and performs a
    non-linear least-squares fit to the Arrow-of-Time probability density
    function to estimate the dimensionless parameter θ.

    Parameters
    ----------
    Q_values : array-like of float
        Entropy-production samples.
    bins : int
        Number of histogram bins used for the fit.

    Returns
    -------
    (theta_hat, theta_err) : (float, float)
        Best-fit θ and its one-sigma uncertainty from the covariance matrix.
        ``theta_err`` is ``nan`` if the covariance cannot be estimated.
    """
    arr = np.asarray(Q_values, dtype=float)

    hist, edges = np.histogram(arr, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask = hist > 0
    xdata = centers[mask]
    ydata = hist[mask]

    theta0 = max(float(np.mean(arr)) / 2.0, 0.1)

    try:
        params, cov = curve_fit(
            _arrow_of_time_pdf,
            xdata,
            ydata,
            p0=[theta0],
            bounds=(1e-6, np.inf),
            maxfev=5000,
        )
        theta_hat = float(params[0])
        theta_err = float(np.sqrt(np.diag(cov))[0]) if cov.size == 1 else float("nan")
    except RuntimeError as exc:
        logger.warning("Arrow-of-Time fit failed: %s", exc)
        theta_hat = theta0
        theta_err = float("nan")

    logger.debug("fit_arrow_of_time: theta_hat=%.4f, theta_err=%.4f", theta_hat, theta_err)
    return theta_hat, theta_err


def plot_Q_distribution(
    Q_values: Sequence[float],
    theta_hat: Optional[float] = None,
    title: str = "Entropy production distribution",
    filename: Optional[str] = None,
    bins: int = 60,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Plot a histogram of Q values with an optional Arrow-of-Time overlay.

    Parameters
    ----------
    Q_values : array-like of float
        Entropy-production samples.
    theta_hat : float or None
        If provided, overlay the fitted Arrow-of-Time density.
    title : str
        Plot title.
    filename : str or None
        If given, save the figure to this path.
    bins : int
        Number of histogram bins.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.  If *None*, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    arr = np.asarray(Q_values, dtype=float)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(arr, bins=bins, density=True, alpha=0.5, edgecolor="black", label="Empirical")

    if theta_hat is not None and theta_hat > 0:
        x_min = max(arr.min(), 1e-6)
        x_max = arr.max()
        x = np.linspace(x_min, x_max, 400)
        y = _arrow_of_time_pdf(x, theta_hat)
        ax.plot(x, y, linewidth=2, label=f"Arrow-of-Time fit (θ={theta_hat:.3f})")

    ax.set_xlabel("Entropy production Q")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend()

    if created_fig:
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            logger.info("Saved Q distribution plot to %s", filename)
        plt.close()

    return ax


def plot_mean_Q_vs_theta(
    theta_values: Sequence[float],
    mean_Q_values: Sequence[float],
    title: str = "⟨Q⟩ vs θ",
    filename: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Plot the mean entropy production as a function of θ.

    A diagonal reference line ⟨Q⟩ = θ is drawn to guide the eye.

    Parameters
    ----------
    theta_values : array-like of float
        Dimensionless parameter θ = N · ε² for each run.
    mean_Q_values : array-like of float
        Corresponding empirical mean entropy productions.
    title : str
        Plot title.
    filename : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.  If *None*, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    thetas = np.asarray(theta_values, dtype=float)
    means = np.asarray(mean_Q_values, dtype=float)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(thetas, means, label="Simulated ⟨Q⟩", zorder=5)

    ref = np.linspace(0, thetas.max() * 1.05, 200)
    ax.plot(ref, ref, "k--", linewidth=1, label="⟨Q⟩ = θ")

    ax.set_xlabel("θ = N · ε²")
    ax.set_ylabel("⟨Q⟩")
    ax.set_title(title)
    ax.legend()

    if created_fig:
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            logger.info("Saved mean Q vs theta plot to %s", filename)
        plt.close()

    return ax
