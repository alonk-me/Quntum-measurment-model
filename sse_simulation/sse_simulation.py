"""
sse_simulation
================

This module implements a simple simulator for the stochastic Schrödinger
equation (SSE) describing a continuously monitored qubit undergoing
Rabi oscillations and homodyne detection of its σ_z observable.

The dynamics implemented here follow the Stratonovich form of the
quantum trajectory equations derived in Dressel et al.,
“Arrow of Time for Continuous Quantum Measurement” (Phys. Rev. Lett.
119, 220507 (2017)).  In that work the authors show that when a qubit
is continuously measured in the σ_z basis and simultaneously driven
around the Bloch sphere by a Hamiltonian proportional to σ_y, the
evolution of the Bloch vector components (x, y, z) can be written as

    ˙x = −Ω z − (x z/τ) r,
    ˙y = −(y z/τ) r,
    ˙z = Ω x + [(1 − z²)/τ] r,

where Ω is the Rabi frequency, τ is the characteristic measurement
time (inverse of the measurement strength), and r(t) is the (rescaled)
measurement record.  The measurement record is given by

    r(t) = z(t) + √τ ξ(t),

with ξ(t) representing a white‑noise process satisfying
⟨ξ(t) ξ(t′)⟩ = δ(t − t′).  These equations assume efficient detection and
Markovian (memoryless) evolution【32376578886034†L170-L196】.  Because the
noise enters multiplicatively through r(t), the above equations are
written in the Stratonovich sense.  To integrate them numerically we
work with finite time steps dt and directly sample the rescaled
measurement increment

    dI = r(t) dt = z(t) dt + √τ dW,

where dW is an increment of the Wiener process drawn from a normal
distribution with mean zero and variance dt.  The stochastic updates
for the Bloch coordinates then become

    x ← x + [−Ω z] dt − (x z/τ) dI
    y ← y − (y z/τ) dI
    z ← z + [Ω x] dt + [(1 − z²)/τ] dI.

At the same time one can accumulate the arrow‑of‑time log‑likelihood
ratio defined by Dressel et al. as

    ln R = (2/τ) ∫₀ᵀ dt r(t) z(t),

which distinguishes forward from backward evolution【32376578886034†L260-L281】.
In discrete time this integral can be approximated as

    ln R ← ln R + (2/τ) z dI,

since r(t) dt ≡ dI.

The simulator implemented below generates ensembles of such
stochastic trajectories starting from an initial pure state on the
Bloch sphere (by default x = 1, y = 0, z = 0).  It returns the
log‑likelihood ratio for each trajectory, which can be compared
with theoretical results or the distributions shown in Fig. 2 of
Dressel et al.【32376578886034†L299-L312】.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SSESimulator:
    """Simulator for stochastic Schrödinger equation of a monitored qubit.

    Parameters
    ----------
    tau : float
        Characteristic measurement time τ.  Smaller values correspond
        to stronger measurements.
    Omega : float
        Rabi frequency Ω of the qubit drive.  The Rabi period is
        2π/Ω.  In Dressel et al. the choice 2π/Ω = 0.5 τ is used to
        ensure that the qubit undergoes several coherent oscillations
        within the measurement time【32376578886034†L299-L312】.
    dt : float
        Integration time step.  This should be much smaller than both
        τ and the Rabi period.  A default of 1e‑3·τ works well for
        demonstrating qualitative behaviour.
    x0, y0, z0 : float
        Initial Bloch vector components.  Default is (1, 0, 0), i.e.
        a +x eigenstate.
    rng : np.random.Generator | None
        Optional NumPy random number generator.  If not provided a
        default generator will be created.
    """

    tau: float = 1.0
    Omega: float = 4.0 * np.pi
    dt: float = 1e-3
    x0: float = 1.0
    y0: float = 0.0
    z0: float = 0.0
    rng: np.random.Generator | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()

    def simulate_trajectory(self, T: float) -> float:
        """Simulate a single stochastic trajectory up to time T.

        The simulation integrates the Stratonovich SDEs for the Bloch
        vector and returns the arrow‑of‑time log‑likelihood ratio
        `ln_R` for that trajectory.

        Parameters
        ----------
        T : float
            Final time (in the same units as τ).  Note that the
            simulation uses a fixed step size dt, so the number of
            integration steps is round(T/dt).

        Returns
        -------
        float
            The accumulated log‑likelihood ratio ln R for the
            trajectory.
        """
        # initial Bloch coordinates
        x = self.x0
        y = self.y0
        z = self.z0
        ln_R = 0.0

        # number of steps
        n_steps = int(np.round(T / self.dt))
        # amplitude for measurement noise increment: sqrt(tau) * sqrt(dt)
        sqrt_tau = np.sqrt(self.tau)
        sqrt_dt = np.sqrt(self.dt)

        for _ in range(n_steps):
            # sample Wiener increment
            dW = self.rng.normal(loc=0.0, scale=sqrt_dt)
            # measurement increment dI = z dt + sqrt(tau) dW
            dI = z * self.dt + sqrt_tau * dW
            # update Bloch vector according to Stratonovich form
            # deterministic drift
            dx_det = -self.Omega * z * self.dt
            dy_det = 0.0
            dz_det = self.Omega * x * self.dt
            # multiplicative part
            dx_stoch = -(x * z / self.tau) * dI
            dy_stoch = -(y * z / self.tau) * dI
            dz_stoch = ((1.0 - z * z) / self.tau) * dI

            x += dx_det + dx_stoch
            y += dy_det + dy_stoch
            z += dz_det + dz_stoch

            # renormalize to avoid numerical drift outside Bloch sphere
            # Without renormalization the integration can accumulate
            # small errors that take the state slightly off the sphere.
            # Here we simply clamp the length to <= 1.
            r2 = x * x + y * y + z * z
            if r2 > 1.0:
                norm = np.sqrt(r2)
                x /= norm
                y /= norm
                z /= norm

            # accumulate log‑likelihood ratio ln R += (2/τ) * z * dI
            ln_R += (2.0 / self.tau) * z * dI

        return ln_R

    def simulate_ensemble(self, n_traj: int, T: float, progress: bool = False) -> np.ndarray:
        """Simulate an ensemble of trajectories.

        Parameters
        ----------
        n_traj : int
            Number of independent trajectories to simulate.
        T : float
            Final time for each trajectory.
        progress : bool, optional
            If True, prints a simple progress indicator every 10 %.  This
            can be useful for long runs.  Defaults to False.

        Returns
        -------
        np.ndarray
            Array of shape (n_traj,) containing the log‑likelihood
            ratios ln R of the individual trajectories.
        """
        results = np.empty(n_traj, dtype=float)
        # simple progress printing
        step = max(1, n_traj // 10)
        for i in range(n_traj):
            if progress and i % step == 0:
                print(f"Simulating trajectory {i+1}/{n_traj}...")
            results[i] = self.simulate_trajectory(T)
        return results


def compute_histogram(data: np.ndarray, bins: int = 50, density: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute a histogram of the data.

    Parameters
    ----------
    data : np.ndarray
        Input samples.
    bins : int
        Number of histogram bins.
    density : bool
        If True the histogram is normalized to form a probability density.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple ``(bin_centers, counts)`` containing the centres of the
        bins and the corresponding (normalized) counts.
    """
    counts, edges = np.histogram(data, bins=bins, density=density)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, counts


if __name__ == "__main__":
    # Example usage: run a short simulation and print summary statistics
    sim = SSESimulator(tau=1.0, Omega=4.0 * np.pi, dt=1e-3)
    n = 1000
    T = 0.2
    samples = sim.simulate_ensemble(n, T, progress=True)
    print(f"Mean ln_R = {np.mean(samples):.3f}, Var ln_R = {np.var(samples):.3f}")