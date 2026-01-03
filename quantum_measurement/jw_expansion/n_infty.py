"""Numerical routines for steady‑state occupation n_∞(g).

This module implements the finite‑size correlation–matrix sums and
thermodynamic‑limit integral used to compute the steady‑state fermionic
occupation ``n_∞`` as a function of the dimensionless measurement
strength ``g``.  These formulas are extracted from a physics
discussion about monitored free–fermion chains.  The key expressions
are reproduced below for completeness.

Definitions
-----------

We define a dimensionless measurement strength

.. math::

    g = \frac{\gamma}{4J},

where ``γ`` is the physical measurement rate and ``J`` is the hopping
coupling.  In the thermodynamic limit ``L\to\infty`` the occupation is
given by an integral over momentum ``k∈[0,π]``:

.. math::

    n_∞(g) = \tfrac{1}{2} - \tfrac{1}{2} \int_0^π \frac{dk}{\pi}
      \frac{1}{g} \bigl|\operatorname{Im} \sqrt{1 - g^2 - 2 i g \cos k}\bigr|.

For finite odd ``L`` the occupation can be computed from the
single–particle correlation matrix.  For anti‑periodic boundary
conditions (APBC) the allowed momenta are

.. math::

    k_n = \frac{2\pi}{L}\Bigl(n + \tfrac{1}{2}\Bigr), \quad n=0,\dots,\frac{L-3}{2},

while for periodic boundary conditions (PBC) they are

.. math::

    k_n = \frac{2\pi}{L}\,n, \quad n=1,\dots,\frac{L-1}{2}.

In both cases one defines

.. math::

    \Delta(k,g) = \sqrt{1 - g^2 - 2 i g \cos k}

with the complex square root taken on its principal branch.  The
sign function

.. math::

    \operatorname{sign\_im}(\Delta) = \operatorname{sign}\bigl(\operatorname{Im}\,\Delta\bigr)

ensures continuity across branch cuts.  The APBC and PBC sums then
read

.. math::

    S_{\rm APBC}(g;L) = \frac{1}{L}\sum_{n=0}^{(L-3)/2}
      \frac{2 \sin^2 k_n}{\sin^2 k_n + \bigl|\cos k_n - i g -\operatorname{sign\_im}(\Delta)\,\Delta\bigr|^2},

.. math::

    S_{\rm PBC}(g;L) = \frac{1}{L}\sum_{n=1}^{(L-1)/2}
      \frac{2 \sin^2 k_n}{\sin^2 k_n + \bigl|\cos k_n - i g -\operatorname{sign\_im}(\Delta)\,\Delta\bigr|^2}.

Both sums converge to ``n_∞(g)`` as ``L`` grows.  For small ``g`` one
finds the limiting value

.. math::

    n_∞(0) = \frac{1}{2} - \frac{1}{\pi} \approx 0.18169,

while for large ``g`` the occupation decays as

.. math::

    n_∞(g) \sim \frac{1}{8 g^2}.

Examples
~~~~~~~~

>>> n0 = integral_expr(1e-3)
>>> abs(n0 - (0.5 - 1/np.pi)) < 1e-3
True

>>> n_large = integral_expr(10.0)
>>> abs(n_large * (10.0**2) - 1/8) < 1e-2
True

These functions can be used in scripts or notebooks to reproduce the
figures and tables described in the assignment.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


def delta(k: np.ndarray | float, g: float) -> np.ndarray:
    """Compute the complex square root Δ(k,g) = sqrt(1 - g^2 - 2 i g cos k).

    The principal branch of the square root is used, as implemented by
    ``numpy.sqrt`` for complex arguments.

    Parameters
    ----------
    k : array_like
        Momentum value(s) in radians.
    g : float
        Measurement strength.  Must be non‑negative.  Small positive
        ``g`` (e.g. 1e-6) should be used instead of zero to avoid
        division by zero in subsequent expressions.

    Returns
    -------
    complex or ndarray of complex
        The value(s) of Δ(k,g).
    """
    k_arr = np.asarray(k, dtype=float)
    # compute inside using complex arithmetic
    return np.sqrt(1.0 - g**2 - 2j * g * np.cos(k_arr))


def sign_im(z: np.ndarray | complex) -> np.ndarray:
    """Return the sign of the imaginary part of ``z``.

    For zero imaginary part the sign is zero.  This helper is used to
    choose the correct branch of the correlation matrix denominators.

    Parameters
    ----------
    z : complex or array_like of complex
        Input complex number(s).

    Returns
    -------
    int or ndarray of ints
        +1 if ``Im(z) > 0``, -1 if ``Im(z) < 0`` and 0 if ``Im(z) == 0``.
    """
    im = np.imag(z)
    return np.sign(im)


def term_value(k: float, g: float) -> float:
    """Compute the summand appearing in the APBC/PBC sums.

    The common term is

        (2 * sin^2(k)) / [ sin^2(k) + | cos(k) - i g - sign_im(Δ) * Δ |^2 ]

    where Δ = sqrt(1 - g^2 - 2 i g cos(k)).

    Parameters
    ----------
    k : float
        Momentum in radians.
    g : float
        Measurement strength (non‑negative).

    Returns
    -------
    float
        Value of the summand for this ``k`` and ``g``.
    """
    sin_k = np.sin(k)
    cos_k = np.cos(k)
    # compute Δ and its sign
    d = delta(k, g)
    s = sign_im(d)
    denom = sin_k**2 + np.abs(cos_k - 1j * g - s * d) ** 2
    return (2.0 * sin_k**2) / denom


def sum_apbc(g: float, L: int) -> float:
    """Evaluate the APBC correlation–matrix sum for a given ``g`` and odd ``L``.

    This function computes

    ``(1/L) ∑_{n=0}^{(L-3)/2} term_value(k_n, g)``

    where ``k_n = 2π/L * (n + 0.5)``.

    Parameters
    ----------
    g : float
        Measurement strength.
    L : int
        System size.  Must be odd and ≥ 3.

    Returns
    -------
    float
        The finite‑size occupation for APBC.
    """
    if L % 2 == 0 or L < 3:
        raise ValueError("L must be an odd integer ≥ 3 for APBC.")
    n_max = (L - 3) // 2
    total = 0.0
    for n in range(n_max + 1):
        k = (2.0 * np.pi / L) * (n + 0.5)
        total += term_value(k, g)
    return total / L


def sum_pbc(g: float, L: int) -> float:
    """Evaluate the PBC correlation–matrix sum for a given ``g`` and odd ``L``.

    This function computes

    ``(1/L) ∑_{n=1}^{(L-1)/2} term_value(k_n, g)``

    where ``k_n = 2π n / L``.

    Parameters
    ----------
    g : float
        Measurement strength.
    L : int
        System size.  Must be odd and ≥ 3.

    Returns
    -------
    float
        The finite‑size occupation for PBC.
    """
    if L % 2 == 0 or L < 3:
        raise ValueError("L must be an odd integer ≥ 3 for PBC.")
    n_max = (L - 1) // 2
    total = 0.0
    for n in range(1, n_max + 1):
        k = (2.0 * np.pi / L) * n
        total += term_value(k, g)
    return total / L


def integral_expr(g: float, Nk: int = 3000) -> float:
    """Evaluate the thermodynamic–limit integral for ``n_∞(g)``.

    The integral expression is

    ``n_∞(g) = 1/2 - (1/2) * (1/π) ∫_0^π |Im[ Δ(k,g) ]|/g dk``

    where ``Δ(k,g) = sqrt(1 - g^2 - 2 i g cos(k))``.  Numerical
    integration is performed using the trapezoidal rule over a linear
    grid of ``Nk`` points.

    Parameters
    ----------
    g : float
        Measurement strength.  A small positive ``g`` should be used
        instead of zero.
    Nk : int, optional
        Number of k–points to use in the trapezoidal integration.  The
        default yields reasonable accuracy; larger values improve
        precision but increase computation time.

    Returns
    -------
    float
        Value of the integral expression ``n_∞(g)``.
    """
    if g <= 0:
        raise ValueError("g must be strictly positive for the integral expression.")
    k_vals = np.linspace(0.0, np.pi, Nk)
    d = delta(k_vals, g)
    integrand = np.abs(np.imag(d)) / g
    integral = np.trapz(integrand, k_vals) / np.pi
    return 0.5 - 0.5 * integral


def small_g_limit() -> float:
    """Return the analytic small‑g limit of ``n_∞``.

    This constant is ``1/2 - 1/π``, as derived in the accompanying
    theory.  It provides a useful reference when comparing finite‑g
    results.

    Returns
    -------
    float
        The value ``0.5 - 1/π ≈ 0.18169``.
    """
    return 0.5 - 1.0 / np.pi


def large_g_limit(g: float) -> float:
    """Return the leading large‑g approximation for ``n_∞(g)``.

    At strong measurement the occupation decays as ``1/(8 g^2)``.  This
    helper implements that formula for a given ``g``.  It is valid
    only for ``g`` sufficiently large (``g >> 1``).

    Parameters
    ----------
    g : float
        Measurement strength.

    Returns
    -------
    float
        Approximate value ``1/(8 g^2)``.
    """
    return 1.0 / (8.0 * g * g)