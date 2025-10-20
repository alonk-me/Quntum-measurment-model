"""
Module implementing a stochastic Schrödinger equation (SSE) simulator
for continuously monitored qubit systems.

This module provides a sim        where dξ = ε * xi with ε = sqrt(γ * dt) and xi = ±1 is a discrete
        measurement outcome. After each update the state is renormalised.

        Parameters
        ----------
        psi : np.ndarray
            Current state vector (2 complex components).

        xi : int
            Discrete measurement outcome, either +1 or -1.at integrates a single-qubit state
under the Itô-form stochastic Schrödinger equation for a diffusive
measurement of the σ_z operator.  The update rule is adapted from the
continuous measurement formalism of Jacobs and Steck and Turkeshi et al.

The simulator is written with extensibility in mind: it accepts a
user‑defined Hamiltonian and measurement operators, although the
default behaviour corresponds to measuring the projector n = (σ_z + I)/2
with no Hamiltonian dynamics.  Future extensions to multiple spins can
build upon this framework by supplying higher‑dimensional state vectors
and appropriate measurement operators.

In addition to propagating the state, the simulator can compute a
trajectory‑dependent “entropy production” functional Q based on the
log‑likelihood ratio between forward and backward measurement records.
For a single qubit monitored along σ_z this functional reduces to

    Q = \frac{2}{\tau} \int_0^T z(t)\,dW_t,

where z(t) = ⟨σ_z⟩_t is the instantaneous Bloch‑sphere z–coordinate and
W_t is a Wiener process.  This expression is derived from Eq. (4) of
Dressel et al., *“Arrow of Time for Continuous Quantum Measurement”*
and matches Eq. (14) of that work when appropriate scalings are
chosen.  We approximate the integral by a sum over discrete time
steps.

The entropy functional Q should not be confused with the von Neumann
entropy of the state; for a pure state the latter is zero at all
times.  Rather, Q quantifies the statistical irreversibility of the
measurement record and plays the role of an entropy production along
individual quantum trajectories.

Example usage:

>>> import numpy as np
>>> from sse_simulation.sse import SingleQubitSSE
>>> sim = SingleQubitSSE(gamma=0.5, dt=1e-3, T=1.0)
>>> psi0 = np.array([1.0+0j, 0.0+0j])  # |0⟩
>>> z_traj, Q = sim.run_trajectory(psi0)
>>> print(f"Final z: {z_traj[-1]:.3f}, Q: {Q:.3f}")

"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Tuple


def sigma_x() -> np.ndarray:
    """Return the Pauli‑X matrix."""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def sigma_y() -> np.ndarray:
    """Return the Pauli‑Y matrix."""
    return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


def sigma_z() -> np.ndarray:
    """Return the Pauli‑Z matrix."""
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


class SingleQubitSSE:
    """Simulator for a single qubit subject to continuous measurement.

    Parameters
    ----------
    gamma : float
        Dimensionless measurement strength.  Larger values correspond
        to stronger monitoring of the measured operator.  In the
        notation of Ref. Jacobs & Steck and Turkeshi et al., the
        measurement time constant is τ = 1/γ.

    dt : float
        Integration timestep.  The dynamics are simulated via the
        Euler–Maruyama method, so dt should be small compared to all
        dynamical timescales.

    T : float
        Total simulation time.  The number of timesteps is
        N_steps = int(T/dt).

    H : np.ndarray, optional
        Hermitian 2×2 matrix representing the Hamiltonian.  If None
        (default) the Hamiltonian is taken to be zero.

    meas_op : np.ndarray, optional
        Hermitian 2×2 operator to measure.  The default is the
        projector n = (σ_z + I)/2, which produces monitoring of the
        spin along the z axis.  Note that the simulator does not
        normalise meas_op; any Hermitian operator is allowed.  The
        expectation value ⟨meas_op⟩ is used in the update formula.
    """

    def __init__(
        self,
        gamma: float,
        dt: float,
        T: float,
        H: np.ndarray | None = None,
        meas_op: np.ndarray | None = None,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if T <= 0:
            raise ValueError("T must be positive")

        self.gamma = gamma
        self.dt = dt
        self.T = T
        self.N_steps = int(np.ceil(T / dt))
        self.tau = 1.0 / gamma  # measurement time constant
        self.H = H if H is not None else np.zeros((2, 2), dtype=complex)
        # Default measurement operator is the projector n = (σ_z + I)/2
        if meas_op is None:
            self.meas_op = 0.5 * (sigma_z() + np.eye(2, dtype=complex))
        else:
            # ensure Hermitian
            if not np.allclose(meas_op, meas_op.conj().T):
                raise ValueError("Measurement operator must be Hermitian")
            self.meas_op = meas_op.copy().astype(complex)

    def _measurement_update(
        self,
        psi: np.ndarray,
        xi: int,
    ) -> np.ndarray:
        """Apply a single Euler–Maruyama step of the SSE to the state.

        The update equation follows Eq. (24) of the Jacobs & Steck
        formulation (Quantum state diffusion) and Eq. (30) of Turkeshi
        et al.  For a Hermitian measurement operator M we have

        d|ψ⟩ = [ -iH dt + (M - ⟨M⟩) dξ - (γ/2) (M - ⟨M⟩)^2 dt ] |ψ⟩,

        where dξ = √γ dW and dW is a standard Wiener increment with
        mean zero and variance dt.  After each update the state is
        renormalised.

        Parameters
        ----------
        psi : np.ndarray
            Current state vector (2 complex components).

        dW : float
            Wiener increment for this time step.  Should satisfy
            `E[dW] = 0` and `Var[dW] = dt`.

        Returns
        -------
        psi_new : np.ndarray
            Updated and renormalised state vector.
        """
        # expectation of measurement operator
        expect = np.vdot(psi, self.meas_op @ psi).real
        # measurement operator minus its expectation
        delta_M = self.meas_op - expect * np.eye(2, dtype=complex)
        # epsilon parameter
        epsilon = np.sqrt(self.gamma * self.dt)
        # noise term dξ = ε * xi
        dxi = epsilon * xi
        # deterministic part: -i H dt - (γ/2)(M - ⟨M⟩)^2 dt
        drift = (-1j * (self.H @ psi)) * self.dt
        drift += -(self.gamma / 2.0) * (delta_M @ (delta_M @ psi)) * self.dt
        # stochastic part: (M - ⟨M⟩) dξ
        diffusion = (delta_M @ psi) * dxi
        psi_new = psi + drift + diffusion
        # renormalise
        psi_new /= np.linalg.norm(psi_new)
        return psi_new

    def _bloch_z(self, psi: np.ndarray) -> float:
        """Compute the Bloch z‑coordinate z = ⟨σ_z⟩ for the given state."""
        return float(np.vdot(psi, sigma_z() @ psi).real)

    def _generate_discrete_outcome(self, psi: np.ndarray) -> int:
        """Generate discrete measurement outcome xi = ±1 with state-dependent probabilities.
        
        The probability of getting xi = +1 is p_+1 = 0.5 * (1 + ε * z_psi)
        where z_psi = ⟨σ_z⟩ and ε = sqrt(γ * dt).
        
        Parameters
        ----------
        psi : np.ndarray
            Current state vector.
            
        Returns
        -------
        xi : int
            Discrete measurement outcome, either +1 or -1.
        """
        epsilon = np.sqrt(self.gamma * self.dt)
        z_psi = self._bloch_z(psi)
        
        # Probability of getting +1
        p_plus = 0.5 * (1.0 + epsilon * z_psi)
        
        # Ensure probability is in valid range [0, 1]
        p_plus = np.clip(p_plus, 0.0, 1.0)
        
        # Generate random outcome
        if np.random.random() < p_plus:
            return 1
        else:
            return -1

    def run_trajectory(
        self,
        psi0: np.ndarray,
        compute_entropy: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Propagate a single trajectory and optionally compute its Q.

        Parameters
        ----------
        psi0 : np.ndarray
            Normalised initial state vector.  Should have shape (2,).

        compute_entropy : bool, default=True
            If True, accumulate the trajectory‑dependent entropy
            production Q based on Eq. (4) of Dressel et al.  If False,
            Q = 0.0 is returned.

        Returns
        -------
        z_traj : np.ndarray
            Array of length `N_steps + 1` containing the Bloch z‑coordinate
            at each time point (including the initial state).

        Q : float
            Accumulated entropy production along the trajectory.
        """
        psi = psi0.astype(complex) / np.linalg.norm(psi0)
        z_traj = np.empty(self.N_steps + 1, dtype=float)
        z_traj[0] = self._bloch_z(psi)
        
        # Store measurement outcomes for discrete Q calculation
        xi_outcomes = []
        epsilon = np.sqrt(self.gamma * self.dt)
        Q = 0.0
        
        for k in range(self.N_steps):
            # generate discrete measurement outcome xi = ±1
            xi = self._generate_discrete_outcome(psi)
            xi_outcomes.append(xi)
            
            # update state
            psi = self._measurement_update(psi, xi)
            
            # record z
            z = self._bloch_z(psi)
            z_traj[k + 1] = z
            
        if compute_entropy:
            # Compute Q using discrete formula: 
            # Q = 2*ε² * Σ(z_i * (z_{i-1} + z_i)/2) + 2*ε * Σ(ξ_i * (z_{i-1} + z_i)/2)
            for i in range(self.N_steps):
                z_avg = (z_traj[i] + z_traj[i + 1]) / 2.0
                Q += 2.0 * epsilon**2 * z_traj[i] * z_avg
                Q += 2.0 * epsilon * xi_outcomes[i] * z_avg
                
        return z_traj, Q

    def run_ensemble(
        self,
        psi0: np.ndarray,
        n_trajs: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run an ensemble of trajectories and collect Q values.

        Parameters
        ----------
        psi0 : np.ndarray
            Initial state for all trajectories.

        n_trajs : int
            Number of trajectories to simulate.

        Returns
        -------
        z_trajs : np.ndarray
            Array of shape (n_trajs, N_steps + 1) containing Bloch z
            coordinates of all trajectories.

        Q_vals : np.ndarray
            Array of length n_trajs with the Q value for each trajectory.
        """
        z_trajs = np.empty((n_trajs, self.N_steps + 1), dtype=float)
        Q_vals = np.empty(n_trajs, dtype=float)
        for idx in range(n_trajs):
            z_traj, Q = self.run_trajectory(psi0, compute_entropy=True)
            z_trajs[idx] = z_traj
            Q_vals[idx] = Q
        return z_trajs, Q_vals


def eq14_distribution(x: np.ndarray, gamma: float, T: float) -> np.ndarray:
    """Evaluate the analytical distribution from Eq. (14).

    Eq. (14) of Dressel et al. gives the probability density of
    x = ln R for a monitored qubit with no Hamiltonian drive (zi = 0).
    The expression reads

        P(x) = (r τ)/(2π T) * e^x / √(e^x − 1)
                * exp[− T/(2τ) − τ/(2T) (arcosh(e^x / 2))^2],

    where τ = 1/γ is the measurement time constant.  The prefactor r
    accounts for the readout scaling and is set to unity in this
    implementation.  The output is not normalised; the distribution
    should be renormalised numerically if used for comparisons.

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the density.

    gamma : float
        Measurement strength γ.

    T : float
        Duration of the measurement run.

    Returns
    -------
    P : np.ndarray
        Unnormalised probability density evaluated at x.
    """
    tau = 1.0 / gamma
    # Avoid invalid values for e^x / 2 < 1 by clipping
    ex = np.exp(x)
    # Where ex/2 < 1, arcosh is undefined; mask those points to zero
    mask = ex >= 2.0
    P = np.zeros_like(x, dtype=float)
    # Compute distribution only where defined
    valid_ex = ex[mask]
    # argument for arcosh
    arg = valid_ex / 2.0
    # arcosh function defined as acosh(x) = ln(x + √(x^2 - 1))
    acosh = np.log(arg + np.sqrt(arg * arg - 1.0))
    numerator = valid_ex
    denom = np.sqrt(valid_ex - 1.0)
    prefactor = (tau) / (2.0 * np.pi * T)
    exponent = np.exp(-T / (2.0 * tau) - (tau / (2.0 * T)) * (acosh**2))
    P_valid = prefactor * numerator / denom * exponent
    P[mask] = P_valid
    return P
