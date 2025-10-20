"""
Data types for quantum measurement simulation.

This module contains dataclasses used by the Kraus operators simulation
for representing quantum states and trajectory results.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class InitialState:
    r"""Data class for specifying the initial quantum state.

    Attributes
    ----------
    alpha : float
        Amplitude of the |0⟩ component. Must satisfy |alpha| ≤ 1.
    beta : float
        Amplitude of the |1⟩ component. Must satisfy |beta| ≤ 1.
    
    Note
    ----
    The amplitudes must be normalized such that |alpha|² + |beta|² = 1.
    If they are not normalized, a ValueError will be raised.
    """
    alpha: float
    beta: float

    def __post_init__(self):
        """Validate the initial state parameters."""
        # Check that individual amplitudes are not too large
        if abs(self.alpha) > 1.0:
            raise ValueError(f"Amplitude |alpha| = {abs(self.alpha):.3f} must be ≤ 1.0")
        if abs(self.beta) > 1.0:
            raise ValueError(f"Amplitude |beta| = {abs(self.beta):.3f} must be ≤ 1.0")
        
        # Check normalization
        norm_squared = self.alpha**2 + self.beta**2
        if abs(norm_squared - 1.0) > 1e-10:
            raise ValueError(f"State is not normalized: |alpha|² + |beta|² = {norm_squared:.6f} ≠ 1.0")
        
        # Check for zero norm (shouldn't happen with above checks, but just in case)
        if norm_squared < 1e-15:
            raise ValueError("Initial state cannot have zero norm")

    def normalized(self) -> Tuple[float, float]:
        """Return the amplitudes (already normalized by construction)."""
        return self.alpha, self.beta

    @classmethod
    def from_unnormalized(cls, alpha: float, beta: float) -> 'InitialState':
        """Create a normalized InitialState from unnormalized amplitudes.
        
        Parameters
        ----------
        alpha : float
            Unnormalized amplitude of the |0⟩ component.
        beta : float
            Unnormalized amplitude of the |1⟩ component.
            
        Returns
        -------
        InitialState
            A new InitialState with normalized amplitudes.
            
        Raises
        ------
        ValueError
            If both alpha and beta are zero (undefined normalization).
        """
        norm = np.sqrt(alpha**2 + beta**2)
        if norm == 0:
            raise ValueError("Cannot normalize zero state: both alpha and beta are zero")
        return cls(alpha / norm, beta / norm)


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