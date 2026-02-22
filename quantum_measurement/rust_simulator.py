"""Python wrapper for the Rust-accelerated LQubitCorrelationSimulator.

This module provides ``FastLQubitSimulator``, a drop-in replacement for
``LQubitCorrelationSimulator`` that delegates the heavy computation to a
compiled Rust extension (``quantum_measurement._rust``).  If the Rust
extension is not available the module raises ``ImportError`` so that the
caller can fall back to the pure-Python implementation.

Example
-------
::

    from quantum_measurement.rust_simulator import FastLQubitSimulator

    sim = FastLQubitSimulator(L=4, J=1.0, epsilon=0.1, N_steps=500, T=1.0)
    Q, z_traj, xi_traj = sim.simulate_trajectory()
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from quantum_measurement._rust import RustLQubitSimulator  # raises ImportError if not built


class FastLQubitSimulator:
    """Rust-accelerated simulator with the same API as LQubitCorrelationSimulator.

    Parameters
    ----------
    L : int
        Number of qubits (sites).
    J : float
        Nearest-neighbour coupling constant.
    epsilon : float
        Measurement strength.
    N_steps : int
        Number of discrete time steps.
    T : float
        Total evolution time.
    closed_boundary : bool, optional
        Periodic boundary conditions when ``True``.
    rng : numpy.random.Generator or None, optional
        If provided and it has a ``integers`` method the seed is extracted
        for reproducibility; otherwise an entropy seed is used.
    """

    def __init__(
        self,
        L: int = 2,
        J: float = 1.0,
        epsilon: float = 0.1,
        N_steps: int = 1000,
        T: float = 1.0,
        closed_boundary: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        seed: Optional[int] = None
        if rng is not None:
            # Extract a reproducible seed from the numpy Generator
            seed = int(rng.integers(0, 2**63))

        self._sim = RustLQubitSimulator(
            l=L,
            j=J,
            epsilon=epsilon,
            n_steps=N_steps,
            t=T,
            closed_boundary=closed_boundary,
            seed=seed,
        )
        self.L = L
        self.J = J
        self.epsilon = epsilon
        self.N_steps = N_steps
        self.T = T
        self.closed_boundary = closed_boundary
        self.dt = T / N_steps

    # ------------------------------------------------------------------
    # Core simulation methods
    # ------------------------------------------------------------------

    def simulate_trajectory(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Simulate a single trajectory using the Rust kernel.

        Returns
        -------
        Q : float
            Entropy production.
        z_trajectory : numpy.ndarray, shape (N_steps+1, L)
            Magnetisation per site at each step.
        xi_trajectory : numpy.ndarray, shape (N_steps, L)
            Measurement outcomes (±1) at each step.
        """
        return self._sim.simulate_trajectory()

    def simulate_z2_mean(self, num_trajectories: int = 1) -> float:
        """Return mean ⟨z²⟩ over ``num_trajectories`` parallel trajectories.

        Parameters
        ----------
        num_trajectories : int
            Number of independent trajectories (run in parallel via Rayon).

        Returns
        -------
        float
            Mean of z² over all sites, time steps, and trajectories.
        """
        return self._sim.simulate_z2_mean(num_trajectories)

    def simulate_ensemble(
        self, n_trajectories: int, progress: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate an ensemble of trajectories in parallel (Rayon).

        Parameters
        ----------
        n_trajectories : int
            Number of independent trajectories.
        progress : bool, optional
            Ignored (kept for API compatibility).

        Returns
        -------
        Q_values : numpy.ndarray, shape (n_trajectories,)
        z_trajectories : numpy.ndarray, shape (n_trajectories, N_steps+1, L)
        xi_trajectories : numpy.ndarray, shape (n_trajectories, N_steps, L)
        """
        q_list, z_list, xi_list = self._sim.simulate_ensemble(n_trajectories)
        Q_values = np.array(q_list, dtype=float)
        z_trajectories = np.stack(z_list, axis=0)
        xi_trajectories = np.stack(xi_list, axis=0)
        return Q_values, z_trajectories, xi_trajectories

    # ------------------------------------------------------------------
    # Convenience methods for parameter sweeps
    # ------------------------------------------------------------------

    def simulate_z2_vs_gamma(
        self,
        gamma_list: list,
        num_traj: int = 10,
    ) -> np.ndarray:
        """Sweep over a list of gamma values and compute ⟨z²⟩ for each.

        For each ``gamma`` value a fresh simulator is created with the
        same ``L``, ``J``, ``N_steps``, ``T``, and ``closed_boundary``
        settings but with ``epsilon = gamma``.

        Parameters
        ----------
        gamma_list : list of float
            Values of the measurement strength to sweep over.
        num_traj : int
            Number of trajectories to average per gamma value.

        Returns
        -------
        z2_values : numpy.ndarray, shape (len(gamma_list),)
            Mean ⟨z²⟩ for each gamma.
        """
        results = []
        for gamma in gamma_list:
            sim = RustLQubitSimulator(
                l=self.L,
                j=self.J,
                epsilon=gamma,
                n_steps=self.N_steps,
                t=self.T,
                closed_boundary=self.closed_boundary,
                seed=None,
            )
            results.append(sim.simulate_z2_mean(num_traj))
        return np.array(results)
