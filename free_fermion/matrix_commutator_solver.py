"""
matrix_commutator_solver.py
============================

This module provides a simple numerical solver for the matrix differential
equation

.. math::

    \frac{\mathrm{d}}{\mathrm{d}t} G(t) = -2\mathrm{i}\,[G(t), h],

where ``G`` is a ``2L×2L`` complex matrix, ``h`` is a fixed
``2L×2L`` matrix composed of four ``L×L`` blocks, and ``[G, h] = Gh - hG``
is the commutator.  The initial condition at ``t=0`` is a diagonal matrix
with ``L`` ones followed by ``L`` zeros.

The code below constructs ``h`` as prescribed and integrates the equation
using a simple forward Euler method.  For small time‐steps the method
reproduces the exact solution to good accuracy.  A parametric function
``compute_time_series`` returns both the time grid and the quantity
``1 + 2 * G[0, 0]`` as a function of time, which for the Ising model
corresponds to the expectation value of the boundary magnetization in
the Jordan–Wigner fermionic representation.

Example
-------

The following snippet computes and prints the first few values of
``1 + 2*G[0, 0]`` for a two‑site chain with ``J=1``:

>>> from matrix_commutator_solver import compute_time_series
>>> times, values = compute_time_series(L=2, J=1.0, T=1.0, steps=100)
>>> print(values[:5])

The module can be imported from a Jupyter notebook or used as a script.
When run as a script it will compute the series for ``L=2`` and ``L=3``
with default parameters and print a short summary.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def build_h(L: int, J: float) -> np.ndarray:
    """Construct the 2L×2L matrix ``h``.

    The Hamiltonian block structure is defined by four ``L×L`` blocks:

    - ``h11`` and ``h12`` contain a single non‑zero band directly above
      the main diagonal with value ``-J``.
    - ``h22`` and ``h21`` contain a single non‑zero band directly above
      the main diagonal with value ``+J``.

    Parameters
    ----------
    L: int
        Size of the chain.  The resulting matrix will be of shape
        ``(2*L, 2*L)``.
    J: float
        Coupling constant.  Determines the magnitude of the off–diagonal
        entries in each block.

    Returns
    -------
    h: numpy.ndarray
        Complex array of shape ``(2*L, 2*L)`` containing the constructed
        block matrix.
    """
    # Build individual blocks
    h11 = np.zeros((L, L), dtype=np.complex128)
    h12 = np.zeros((L, L), dtype=np.complex128)
    h21 = np.zeros((L, L), dtype=np.complex128)
    h22 = np.zeros((L, L), dtype=np.complex128)

    # Fill the bands: positions (i, i+1)
    for i in range(L - 1):
        # Blocks h11 and h12 have entries -J above the main diagonal
        h11[i, i + 1] = -J
        h12[i, i + 1] = -J
        # Blocks h22 and h21 have entries +J above the main diagonal
        h22[i, i + 1] = J
        h21[i, i + 1] = J

    # Assemble the full block matrix
    top = np.hstack((h11, h12))
    bottom = np.hstack((h21, h22))
    h = np.vstack((top, bottom))
    return h


def initial_G(L: int) -> np.ndarray:
    """Construct the initial matrix ``G``.

    ``G`` is a ``2L×2L`` diagonal matrix with the first ``L`` diagonal
    entries equal to ``1`` and the last ``L`` equal to ``0``.

    Parameters
    ----------
    L: int
        Half the dimension of the matrix.

    Returns
    -------
    G0: numpy.ndarray
        The initial condition matrix of shape ``(2*L, 2*L)``.
    """
    G0 = np.zeros((2 * L, 2 * L), dtype=np.complex128)
    # First L entries are ones
    for i in range(L):
        G0[i, i] = 1.0
    return G0


def compute_time_series(L: int, J: float, T: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ``1 + 2*G[0, 0]`` as a function of time.

    The integration uses a forward Euler discretization.  For sufficiently
    small time steps the results approximate the exact solution to
    acceptable accuracy for qualitative comparisons.

    Parameters
    ----------
    L: int
        Size of the chain.  Determines the dimension of ``G`` and ``h``.
    J: float
        Coupling constant.
    T: float
        Final integration time.  The integration starts at zero.
    steps: int
        Number of time steps for the Euler method.  Larger values give
        better accuracy at the cost of increased computation.

    Returns
    -------
    times: numpy.ndarray
        Array of shape ``(steps + 1,)`` containing the time points.
    values: numpy.ndarray
        Array of shape ``(steps + 1,)`` containing ``1 + 2*G[0, 0]`` at
        each time point.
    """
    dt = T / steps
    h = build_h(L, J)
    G = initial_G(L)
    # Time grid including t=0
    times = np.linspace(0.0, T, steps + 1)
    values = np.empty(steps + 1, dtype=np.complex128)
    # Record initial value
    values[0] = 1.0 + 2.0 * G[0, 0]

    # Precompute for efficiency: -2i
    prefactor = -2.0j
    for n in range(steps):
        # Compute the commutator derivative
        comm = G @ h - h @ G
        # Euler update
        G = G + dt * prefactor * comm
        # Symmetrize to counteract numerical drift and maintain Hermiticity
        G = 0.5 * (G + G.conj().T)
        # Store the observable
        values[n + 1] = 1.0 + 2.0 * G[0, 0]
    return times, values


def _demo() -> None:
    """Run a simple demonstration when executed as a script."""
    print("Demonstration of matrix_commutator_solver.")
    for L in (2, 3):
        J = 1.0
        T = 10.0
        steps = 2000
        t, val = compute_time_series(L=L, J=J, T=T, steps=steps)
        # Only print the last value as a sanity check
        print(f"L={L}: final value 1+2*G[0,0] ≈ {val[-1].real:.5f} (at t={T})")


if __name__ == "__main__":
    _demo()