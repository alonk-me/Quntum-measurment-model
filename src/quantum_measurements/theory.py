"""Analytical results used to interpret the simulations."""

from __future__ import annotations

import numpy as np


def dressel_eq14_pdf(x: np.ndarray, theta: float) -> np.ndarray:
    """Probability density from Eq. (14) of Dressel *et al.*"""

    theta = float(theta)
    x = np.asarray(x, dtype=float)
    if theta <= 0.0:
        return np.zeros_like(x)
    normalisation = np.sqrt(4.0 * np.pi * theta)
    exponent = -((x - theta) ** 2) / (4.0 * theta)
    return np.exp(exponent) / (normalisation * np.cosh(x / 2.0))

