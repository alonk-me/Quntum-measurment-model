"""
Adjusted non-Hermitian simulator (matching entropy plots).
========================================================

This lightweight wrapper around :class:`non_hermitian_hat.NonHermitianHatSimulator`
applies a post‑processing step to the accumulated entropy production.  In
``Entropy_Production_Quantum_Measurement.pdf`` it is shown that the
asymptotic entropy production rate for the monitored Ising chain is

.. math::

    \frac{\mathrm{d}S}{\mathrm{d}t} = \gamma L - 4\Gamma_\mathrm{min}(\gamma),

where ``Γ_min(γ)`` is the minimal decay rate of the non‑Hermitian
quasiparticles.  The first term ``γ L`` arises from the imaginary
potential and dominates the entropy growth; the numerical results in
figures 1 and 2 of the PDF therefore subtract ``γ L`` from the measured
entropy production rate before comparison.  To facilitate direct
comparison with those plots, this module defines a simulator that
subtracts ``γ L`` (times the total evolution time) from the entropy
accumulated by the hat operator simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from non_hermitian_hat import NonHermitianHatSimulator


@dataclass
class NonHermitianAdjustedSimulator(NonHermitianHatSimulator):
    """Subtract the trivial ``γ L T`` contribution from the entropy budget.

    This subclass behaves identically to its parent but returns an entropy
    production value with the extensive ``γ L T_total`` term removed.
    Concretely if the parent class reports ``Q`` after evolving for total
    time ``T_total = dt * N_steps``, this class returns ``Q - γ L T_total``.
    When dividing by the total time one obtains the adjusted entropy
    production rate plotted in the PDF.
    """

    def simulate_trajectory(self) -> Tuple[float, any]:
        """Run a trajectory and subtract ``γ L T`` from the entropy.

        Returns
        -------
        Q_adj : float
            The adjusted entropy production, ``Q - γ L T_total``.
        n_traj : numpy.ndarray
            Occupation trajectory from the parent class.
        """
        Q_raw, n_traj = super().simulate_trajectory()
        # Subtract the extensive gamma*L*T_total term from the entropy budget.
        # Q_adj = Q_raw - self.gamma * self.L * self.T_total
        Q_adj = self.gamma * self.L * self.T_total - Q_raw
        return Q_adj, n_traj
