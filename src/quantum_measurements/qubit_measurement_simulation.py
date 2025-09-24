r"""
qubit_measurement_simulation.py
--------------------------------

This module implements a Monte–Carlo simulation of a single qubit
subjected to repeated weak measurements.  There is no unitary
evolution (the Hamiltonian is set to zero), so the system evolves
purely due to the backaction of the measurement.  The model and
notation follow the discussion in Appendix A of Turkeshi et al.
("Measurement‑Induced Entanglement Transitions in the Quantum Ising
Chain") and the Arrow of Time paper, which analyse continuous
measurements and stochastic Schrödinger equations.  Here we work in discrete
time with two diagonal Kraus operators to model the weak measurement.

Measurement model
~~~~~~~~~~~~~~~~~

We define a small positive parameter :math:`\epsilon\ll1` and set
``a = 1 + epsilon``.  The Kraus operators are

.. math::

   M_0 = \frac{1}{\sqrt{1 + a^2}} \begin{pmatrix}1 & 0\\0 & a\end{pmatrix},
   \quad
   M_1 = \frac{1}{\sqrt{1 + a^2}} \begin{pmatrix}a & 0\\0 & 1\end{pmatrix},

which satisfy :math:`M_0^\dagger M_0 + M_1^\dagger M_1 = I`.  In
each measurement step the qubit wavefunction :math:`|\psi\rangle =
(\alpha,\beta)` is acted on by either :math:`M_0` or :math:`M_1`.
The probability of applying :math:`M_0` is given by Born’s rule

.. math::

   P_0 = \|M_0 |\psi\rangle\|^2 = \frac{(a\alpha)^2 + \beta^2}{1 + a^2},

and the probability of applying :math:`M_1` is :math:`P_1=1-P_0`
.Conditional on the outcome, the state is updated and
normalised:

.. math::

   |\psi'\rangle = \frac{M_0|\psi\rangle}{\|M_0|\psi\rangle\|} =
   \frac{(a\alpha,\;\beta)}{\sqrt{(a\alpha)^2 + \beta^2}}\quad \text{(if outcome 0)},

   |\psi'\rangle = \frac{M_1|\psi\rangle}{\|M_1|\psi\rangle\|} =
   \frac{(\alpha,\;a\beta)}{\sqrt{\alpha^2 + (a\beta)^2}}\quad \text{(if outcome 1)}.

We encode the measurement outcome as a random variable
:math:`\xi\in\{+1,-1\}`.  When the outcome is ``0`` (application of
:math:`M_0`) we assign :math:`\xi=+1`; when the outcome is ``1``
(application of :math:`M_1`) we assign :math:`\xi=-1`.  This sign
convention ensures that the entropy production defined below reflects
the direction of the state change, consistent with the Arrow of Time
analysis.

Entropy production
~~~~~~~~~~~~~~~~~~

For a pure state :math:`(\alpha,\beta)` the expectation value of the
Pauli operator :math:`\sigma_z` is :math:`\langle\sigma_z\rangle =
|\alpha|^2 - |\beta|^2`.  During the ``i``th measurement the
expectation values before and after the measurement are denoted
:math:`z_{i}^{\text{before}}` and :math:`z_{i}^{\text{after}}` respectively.  We define
their average as

.. math::

   z_i = \frac{z_{i}^{\text{before}} + z_{i}^{\text{after}}}{2}.

The entropy production for a trajectory of ``N`` measurements is

.. math::

   Q = 2 \sum_{i=0}^{N-1} \xi_i \, z_i.

This quantity is analogous to the logarithm of the retrodictive ratio
:math:`\ln(R)` studied in the Arrow of Time paper, which has a
probability density function given by equation (14) in that article.

Functions provided
------------------

This module exposes the following functions:

* :func:`run_trajectory`: simulate a single measurement trajectory
  returning the measurement outcomes, z‑values, and entropy
  production.
* :func:`simulate_Q_distribution`: run many independent trajectories
  and collect their entropy productions.
* :func:`eq14_pdf`: evaluate the probability density from Eq.
  (14) of the Arrow of Time paper for a given parameter \(\theta\).
* :func:`fit_eq14`: fit the empirical Q distribution to Eq. (14)
  using nonlinear least squares to estimate \(\theta = T/\tau\).
* :func:`plot_Q_fit`: plot the histogram of Q values along with the
  fitted curve for visual comparison.

The simulation results can be used to test the Arrow of Time
distribution and explore how the parameter \(\theta\) depends on the
measurement strength and the number of steps.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
import matplotlib.pyplot as plt
from math import exp, sqrt, cosh, pi
from scipy.optimize import curve_fit

try:  # pragma: no cover - fallback when tqdm is unavailable
    from tqdm import tqdm
except ImportError:  # pragma: no cover - lightweight fallback
    def tqdm(iterable, *args, **kwargs):
        """Fallback progress iterator when :mod:`tqdm` is missing."""

        return iterable

@dataclass
class TrajectoryResult:
    r"""Data class storing the results of a single measurement trajectory.

    Attributes
    ----------
    outcomes : list of int
        Sequence of measurement outcomes :math:`\xi_i \in \{+1,-1\}`.
    z_averages : list of float
        Average :math:`\sigma_z` expectation values for each step.
    Q : float
        Computed entropy production for the trajectory.
    zs_before : list of float
        Expectation values of :math:`\sigma_z` immediately before each
        measurement.
    zs_after : list of float
        Expectation values of :math:`\sigma_z` immediately after each
        measurement.
    """

    outcomes: List[int]
    z_averages: List[float]
    Q: float
    zs_before: List[float]
    zs_after: List[float]


def run_trajectory(N: int, epsilon: float, rng: Optional[np.random.Generator] = None) -> TrajectoryResult:
    r"""Simulate a single measurement trajectory.

    Parameters
    ----------
    N : int
        Number of measurement steps.
    epsilon : float
        Measurement strength parameter.  ``a = 1 + epsilon``.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.  If ``None`` a new
        generator is created.

    Returns
    -------
    TrajectoryResult
        Dataclass containing outcomes, z‑averages and entropy production.
    """
    if rng is None:
        rng = np.random.default_rng()

    a = 1.0 + epsilon
    a2 = a * a

    # initial state on equator of Bloch sphere: |psi_0> = (1/sqrt{2}, 1/sqrt{2})
    alpha = 1.0 / np.sqrt(2.0)
    beta = 1.0 / np.sqrt(2.0)

    outcomes: List[int] = []
    z_avgs: List[float] = []
    zs_before: List[float] = []
    zs_after: List[float] = []

    for _ in range(N):
        # compute expectation before measurement
        z_before = alpha * alpha - beta * beta

        # Born probabilities using analytic formulas
        num0 = (a * alpha) * (a * alpha) + beta * beta
        num1 = alpha * alpha + (a * beta) * (a * beta)
        denom = 1.0 + a2
        P0 = num0 / denom

        # draw random outcome: xi = +1 for M0, -1 for M1
        r = rng.uniform()
        if r < P0:
            xi = +1
            # apply M0: (a*alpha, beta)/norm
            new_alpha = a * alpha
            new_beta = beta
            norm = np.sqrt(new_alpha * new_alpha + new_beta * new_beta)
            new_alpha /= norm
            new_beta /= norm
        else:
            xi = -1
            # apply M1: (alpha, a*beta)/norm
            new_alpha = alpha
            new_beta = a * beta
            norm = np.sqrt(new_alpha * new_alpha + new_beta * new_beta)
            new_alpha /= norm
            new_beta /= norm

        # expectation after measurement
        z_after = new_alpha * new_alpha - new_beta * new_beta

        # average expectation used for entropy production
        z_avg = 0.5 * (z_before + z_after)

        outcomes.append(xi)
        z_avgs.append(float(z_avg))
        zs_before.append(float(z_before))
        zs_after.append(float(z_after))

        # update state
        alpha, beta = new_alpha, new_beta

    # compute entropy production
    Q = 2.0 * epsilon * float(np.sum(np.array(outcomes, dtype=float) * np.array(z_avgs)))


    return TrajectoryResult(outcomes, z_avgs, Q, zs_before, zs_after)


def simulate_Q_distribution(num_traj: int, N: int, epsilon: float, seed: Optional[int] = None) -> np.ndarray:
    r"""Simulate many trajectories and return array of entropy production values.

    Parameters
    ----------
    num_traj : int
        Number of independent trajectories.
    N : int
        Number of measurement steps per trajectory.
    epsilon : float
        Measurement strength parameter.
    seed : int, optional
        Seed for the random generator to allow reproducibility.

    Returns
    -------
    numpy.ndarray
        Array of length ``num_traj`` containing the entropy productions.
    """
    rng = np.random.default_rng(seed)
    Q_values = np.empty(num_traj, dtype=float)
    for i in tqdm(range(num_traj)):

        res = run_trajectory(N, epsilon, rng)
        Q_values[i] = res.Q
    return Q_values


def eq14_pdf(x: np.ndarray, theta: float) -> np.ndarray:
    r"""Evaluate the probability density function from Eq. (14) of the Arrow of Time paper.

    The formula reads

    .. math::

       p(x;\theta) = \frac{\exp\left[-\frac{(x-\theta)^2}{4\theta}\right]}{\sqrt{4\pi\theta}\,\cosh(x/2)},

    where :math:`\theta=T/\tau` is a dimensionless parameter related to the
    total monitoring time and the measurement timescale.  This function
    returns the density evaluated at ``x`` given ``theta``.

    Parameters
    ----------
    x : numpy.ndarray
        Points at which to evaluate the density.
    theta : float
        The parameter :math:`\theta`.  Must be positive.

    Returns
    -------
    numpy.ndarray
        Array of probability densities at each point in ``x``.
    """
    theta = float(theta)
    # avoid division by zero
    if theta <= 0:
        return np.zeros_like(x)
    # compute pdf
    prefactor1 = 1.0 / np.sqrt(2.0 * pi * theta)
    prefactor2 = np.exp(x)/np.sqrt(np.exp(x)-1)
    exponent = -theta/2.0 - (np.arccosh(np.exp(x/2.0)))**2/(theta*2.0)
    return prefactor1 * prefactor2 * np.exp(exponent)


def fit_eq14(Q_values: Sequence[float]) -> Tuple[float, float]:
    r"""Fit the empirical distribution of Q to the Arrow of Time density and estimate theta.

    This function constructs a histogram of ``Q_values`` and performs a
    nonlinear least–squares fit of the histogram to the analytical
    expression :func:`eq14_pdf`.  It returns the best–fit value of
    ``theta`` and its estimated standard error from the covariance
    matrix.

    Parameters
    ----------
    Q_values : sequence of float
        Empirically measured entropy productions.

    Returns
    -------
    (float, float)
        Tuple ``(theta_hat, theta_error)`` giving the estimated
        parameter and its one–sigma uncertainty.
    """
    # create histogram data
    hist, bin_edges = np.histogram(Q_values, bins=60, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # remove zero bins to avoid fitting log zeros
    mask = hist > 0
    xdata = bin_centers[mask]
    ydata = hist[mask]

    # initial guess for theta: mean/2 (based on approximate scaling)
    mean_Q = np.mean(Q_values)
    theta0 = max(mean_Q / 2.0, 0.1)

    def model(x, theta):
        return eq14_pdf(x, theta)

    params, cov = curve_fit(model, xdata, ydata, p0=[theta0], bounds=(1e-6, np.inf))
    theta_hat = params[0]
    theta_err = np.sqrt(np.diag(cov))[0] if cov.size == 1 else float('nan')
    return theta_hat, theta_err


def plot_Q_fit(Q_values: Sequence[float], theta_hat: float, filename: str) -> None:
    r"""Plot the empirical distribution of Q alongside the fitted Arrow of Time density.

    Parameters
    ----------
    Q_values : sequence of float
        Entropy production values from the simulation.
    theta_hat : float
        Fitted parameter :math:`\theta` obtained from :func:`fit_eq14`.
    filename : str
        Name of the file to save the plot (PNG).
    """
    # plot histogram
    plt.figure(figsize=(8, 4))
    hist, bins, _ = plt.hist(Q_values, bins=60, density=True, alpha=0.5, edgecolor='black', label='Empirical')
    # overlay fitted pdf
    x = np.linspace(min(Q_values), max(Q_values), 400)
    y = eq14_pdf(x, theta_hat)
    plt.plot(x, y, label=f'Eq.14 fit (θ={theta_hat:.2f})', linewidth=2)
    plt.xlabel('Entropy production Q')
    plt.ylabel('Probability density')
    plt.title('Entropy production distribution and Eq.14 fit')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def calculate_theta(num_measurments: int, epsilon: int|float) -> float:
  r"""Calculate the theta value.

    Parameters
    ----------
    num_measurments : int
        Number of measurement steps per trajectory.
    epsilon : int|float
        Measurement strength parameter.

    Returns
    -------
    float
        The theta value.
    """
  return num_measurments*epsilon**2


def average_Q_vs_theta(num_traj: int, num_measurments:list[int],epsilon:list[int|float] , seed: Optional[int] = None):
  r"""Simulate many Q and return array of mean Q and theta.

    Parameters
    ----------
    num_traj : int
        Number of independent trajectories.
     num_measurments: list[int]
        list of number of measurement steps per trajectory.
    epsilon : list[int|float]
        Measurement strength parameter
    seed : int, optional
        Seed for the random generator to allow reproducibility.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray) : mean_Q, theta
        Array of length ``num_traj`` containing the entropy productions.
    """
  if len(num_measurments) != len(epsilon):
    raise ValueError("num_measurments and epsilon must have the same length")
  length_of_lists= len(num_measurments)
  theta_list = []
  mean_Q_array = []
  for i in range(length_of_lists):
    Q_values = simulate_Q_distribution(num_traj, num_measurments[i], epsilon[i], seed)
    mean_Q = np.mean(np.array(Q_values))
    mean_Q_array.append(mean_Q)
    theta = calculate_theta(num_measurments[i], epsilon[i])
    theta_list.append(theta)
  return mean_Q_array, theta_list

