"""
Non-Hermitian evolution for monitored free fermions (spin operator version).
===========================================================================

This module complements ``non_hermitian_hat.py`` by providing a simulator
which reports observables in terms of the spin magnetisation ``σ^z`` rather
than the occupation ``n``.  The underlying dynamics are identical:

* A chain of ``L`` sites evolves under coherent ``XX`` hopping of strength
  ``J``.
* Continuous measurement of the local **number** operators ``n_j``
  introduces an imaginary potential ``-i γ/2 Σ n_j`` leading to exponential
  decay of fermion amplitudes at rate ``γ/2``.

Internally the evolution of the correlation matrix ``G`` and the entropy
production are identical to those in ``NonHermitianHatSimulator``.  The
only difference is in how observables are reported and how the entropy
contribution is expressed.  The magnetisation on site ``j`` is related to
the occupation by ``σ^z_j = 2 n_j - 1``.  Consequently the instantaneous
entropy production rate can be written as

.. math::

    \\frac{\\mathrm{d}Q}{\\mathrm{d}t}
      = \\gamma \\sum_{j=0}^{L-1} \\bigl(1 - \\langle n_j \\rangle\\bigr)
      = \\frac{\\gamma}{2} \\sum_{j=0}^{L-1}
        \\bigl(1 - \\langle σ^z_j \\rangle\\bigr),

which makes explicit the normalisation difference between number and spin
operators.

The class defined here, ``NonHermitianSpinSimulator``, follows the same
interface as ``NonHermitianHatSimulator`` but returns arrays of
magnetisation expectation values instead of occupations.  The total
evolution time is ``T_total = dt * N_steps``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

from non_hermitian_hat import NonHermitianHatSimulator


@dataclass
class NonHermitianSpinSimulator(NonHermitianHatSimulator):
    """Simulate non‑Hermitian evolution and report spin observables.

    This subclass reuses all functionality from
    ``NonHermitianHatSimulator`` but interprets the diagonal elements of
    the covariance matrix as occupations ``⟨n_i⟩`` and converts them to
    magnetisation ``⟨σ^z_i⟩ = 2⟨n_i⟩ - 1``.  The entropy production is
    accumulated with the same formula ``Q̇ = γ Σ (1 - n_i)`` which can
    equivalently be expressed in terms of ``σ^z_i``.
    """

    def simulate_trajectory(self) -> Tuple[float, np.ndarray]:
        """Run a single trajectory and return entropy and magnetisation.

        Returns
        -------
        Q_total : float
            Total entropy production.
        z_traj : numpy.ndarray
            Array of shape ``(N_steps+1, L)`` containing ``⟨σ^z⟩`` on each
            site at each time step.
        """
        G = self.G_initial.copy()
        z_traj = np.zeros((self.N_steps + 1, self.L), dtype=float)
        # initial magnetisation: z = 2n - 1, n=0 for vacuum → z=-1
        n_initial = np.real(np.diag(G)[: self.L])
        z_traj[0] = 2.0 * n_initial - 1.0
        Q = 0.0
        for step in range(self.N_steps):
            # record occupations before step
            n_before = np.real(np.diag(G)[: self.L])
            # Hamiltonian evolution
            G = self._hamiltonian_step(G)
            # Imaginary potential
            G = self._nonhermitian_step(G)
            # Symmetrise and clip diagonal
            G = 0.5 * (G + G.conj().T)
            diag = np.diag(G).copy()
            diag_clipped = np.clip(np.real(diag), 0.0, 1.0)
            for i in range(2 * self.L):
                G[i, i] = diag_clipped[i] + 0.0j
            # occupations after step
            n_after = np.real(np.diag(G)[: self.L])
            # magnetisation after step
            z_traj[step + 1] = 2.0 * n_after - 1.0
            # Stratonovich average of occupations
            n_avg = 0.5 * (n_before + n_after)
            # entropy increment expressed in terms of occupations
            Q += self.gamma * self.dt * np.sum(1.0 - n_avg)
        return Q, z_traj
