r"""
LQubitCorrelationSimulator
==========================

This module generalizes the ``TwoQubitCorrelationSimulator`` from the
`quantum_measurement` package to an arbitrary number of qubits ``L`` and
supports both open and closed boundary conditions.  The simulation uses
the correlation‑matrix formalism of free fermions to model continuous
measurement on each qubit together with Hamiltonian evolution generated
by a nearest‑neighbour ``XX`` coupling of strength ``J``.  The evolution
equation for the ``2L×2L`` correlation matrix ``G`` reads

.. math::

    \dot G = -2\mathrm{i}
    \,[G, h] + \varepsilon\bigl(G\,\hat\xi + \hat\xi\,G - 2G\,\hat\xi\,G\bigr)
    
    \qquad{} - \varepsilon^2\bigl(G - \operatorname{diag}(G)\bigr),

where ``h`` is the Bogoliubov–de Gennes Hamiltonian of the free‑fermion
chain, ``\hat\xi`` is a diagonal matrix encoding stochastic measurement
outcomes, ``\varepsilon`` is the measurement strength and ``[\cdot,\cdot]``
denotes the commutator.  A short derivation of this equation and its
connection to the Jordan–Wigner transformation can be found in the
``free_fermion`` notes in the original repository.  The class defined
below exposes methods for simulating single trajectories and ensembles
of trajectories, as well as computing the entropy production ``Q`` in
each trajectory.  When ``J = 0`` the Hamiltonian is trivial and the
measurement processes on different sites remain independent.

Example
-------

The following snippet runs a single trajectory for a four‑site open
chain and prints the final magnetization (\langle\sigma^z\rangle) on
each site::

    from l_qubit_correlation_simulator import LQubitCorrelationSimulator

    sim = LQubitCorrelationSimulator(L=4, J=0.0, epsilon=0.1, N_steps=500, T=1.0,
                                     closed_boundary=False)
    Q, z_traj, xi_traj = sim.simulate_trajectory()
    print("Final z", z_traj[-1])

The entropic cost of the measurement, stored in ``Q``, is accumulated
according to the Stratonovich discretisation described in the EP
production notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class LQubitCorrelationSimulator:
    r"""Free‑fermion correlation matrix simulator for an L‑site chain.

    Parameters
    ----------
    L : int
        Number of qubits (sites) in the chain.  The correlation matrix
        will be of dimension ``2*L × 2*L``.
    J : float
        Coupling constant for the nearest‑neighbour ``XX`` Hamiltonian.
    epsilon : float
        Measurement strength; controls the magnitude of both the
        stochastic and deterministic measurement terms.
    N_steps : int
        Number of discrete time steps used in the simulation.  The
        integration time step is ``dt = T / N_steps``.
    T : float
        Total evolution time of each trajectory.
    closed_boundary : bool, optional
        If ``True`` the chain has periodic (closed) boundary conditions
        and the last site is coupled back to the first.  If ``False``
        the chain is open.
    rng : numpy.random.Generator or None, optional
        Random number generator used for sampling measurement outcomes.
        When ``None`` a new default RNG is constructed.
    """

    L: int = 2
    J: float = 1.0
    epsilon: float = 0.1
    N_steps: int = 1000
    T: float = 1.0
    closed_boundary: bool = False
    rng: Optional[np.random.Generator] = field(default=None, repr=False)

    # Derived quantities initialised after construction
    dt: float = field(init=False)
    h: np.ndarray = field(init=False, repr=False)
    G_initial: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate input
        if self.L < 1:
            raise ValueError("L must be at least 1")
        if self.N_steps < 1:
            raise ValueError("N_steps must be at least 1")
        if self.T <= 0.0:
            raise ValueError("T must be positive")

        # Initialise RNG
        if self.rng is None:
            self.rng = np.random.default_rng()

        # Time step
        self.dt = self.T / self.N_steps

        # Build the BdG Hamiltonian
        self.h = self._build_hamiltonian()

        # Construct initial correlation matrix: ones on the first L diagonal
        # entries (occupied modes) and zeros on the last L (empty modes).
        self.G_initial = np.zeros((2 * self.L, 2 * self.L), dtype=complex)
        for i in range(self.L):
            self.G_initial[i, i] = 1.0 + 0.0j

    def _build_hamiltonian(self) -> np.ndarray:
        r"""Construct the 2L×2L Bogoliubov–de Gennes Hamiltonian ``h``.

        The matrix ``h`` is built from four L×L blocks ``h11``, ``h12``,
        ``h21`` and ``h22``.  Following the pattern from the original
        two‑site implementation and the ``matrix_commutator_solver``
        provided in the same repository, we populate a single
        super‑diagonal with ±``J`` and optionally wrap around for closed
        boundary conditions.  In particular,

        * ``h11`` and ``h12`` receive ``-J`` on the super‑diagonal,
        * ``h21`` and ``h22`` receive ``+J`` on the super‑diagonal.

        For a closed chain an additional entry is placed in the corner
        connecting site ``L-1`` to site ``0`` in each block.

        Returns
        -------
        h : numpy.ndarray
            Complex array of shape ``(2*L, 2*L)`` containing the
            Hamiltonian.
        """
        L = self.L
        J = self.J
        h11 = np.zeros((L, L), dtype=complex)
        h12 = np.zeros((L, L), dtype=complex)
        h21 = np.zeros((L, L), dtype=complex)
        h22 = np.zeros((L, L), dtype=complex)

        # Fill off‑diagonal entries corresponding to nearest neighbours
        for i in range(L - 1):
            h11[i, i + 1] = -J
            h12[i, i + 1] = -J
            h22[i, i + 1] = +J
            h21[i, i + 1] = +J

        # Periodic coupling between last and first sites
        if self.closed_boundary and L > 1:
            i = L - 1
            j = 0
            h11[i, j] = -J
            h12[i, j] = -J
            h22[i, j] = +J
            h21[i, j] = +J

        # Assemble full BdG matrix
        top = np.hstack((h11, h12))
        bottom = np.hstack((h21, h22))
        h = np.vstack((top, bottom))
        return h

    def _compute_z_values(self, G: np.ndarray) -> np.ndarray:
        r"""Compute the expectation values ⟨σ^z_i⟩ for each site.

        In the Jordan–Wigner mapping the magnetisation on site ``i`` is

        .. math::

            \langle\sigma^z_i\rangle = 2\,\mathrm{Re}\,G_{ii} - 1,

        where ``i`` refers to the annihilation index ``0 ≤ i < L``.  The
        returned array has shape ``(L,)``.
        """
        z = 2.0 * np.real(np.diag(G)[: self.L]) - 1.0
        return z.astype(float)

    def _hamiltonian_step(self, G: np.ndarray) -> np.ndarray:
        r"""Apply a single time step of unitary evolution under ``h``.

        The Hamiltonian contribution to the equation of motion is

        .. math::

            \dot G = -2\mathrm{i}\,[G, h],

        so over a small time step ``dt`` we perform an explicit Euler
        update ``G ← G + dt * (-2i) * (G@h - h@G)``.  This simple
        discretisation suffices for small ``dt`` and is consistent
        with the implementation used in the original two‑site code.
        """
        commutator = G @ self.h - self.h @ G
        dG = -2.0j * self.dt * commutator
        return G + dG

    def _measurement_step(self, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Apply a measurement step with stochastic outcomes on each site.

        Measurement back‑action on the correlation matrix is modelled by
        the operator ``\hat\xi = \operatorname{diag}(\xi_1, …, \xi_L,
        -\xi_1, …, -\xi_L)`` where each ``\xi_i = ±1`` is sampled
        independently with equal probability.  The evolution rule is

        .. math::

            G ← G + ε\bigl(G\,\hat\xi + \hat\xi\,G - 2G\,\hat\xi\,G\bigr)
                - ε^2\bigl(G - \operatorname{diag}(G)\bigr).

        After the update the matrix is symmetrised and its diagonal
        entries are clipped to the interval ``[0, 1]`` to maintain a
        physically valid covariance matrix.

        Returns
        -------
        G_new : numpy.ndarray
            Updated correlation matrix.
        xi : numpy.ndarray
            Array of shape ``(L,)`` containing the sampled ±1 outcomes.
        """
        # Draw ±1 measurement outcomes independently for each site
        xi = self.rng.choice([-1, 1], size=self.L)

        # Construct xi_hat as a diagonal matrix with +xi on the particle
        # sector and -xi on the hole sector
        xi_hat_diag = np.concatenate((xi, -xi)).astype(complex)
        xi_hat = np.diag(xi_hat_diag)

        # Stochastic term proportional to ε
        stochastic = self.epsilon * (G @ xi_hat + xi_hat @ G - 2.0 * G @ xi_hat @ G)
        # Deterministic damping term proportional to ε²
        G_diag = np.diag(np.diag(G))
        damping = - (self.epsilon ** 2) * (G - G_diag)

        G_new = G + stochastic + damping

        # Symmetrise to counteract numerical drift and enforce Hermiticity
        G_new = 0.5 * (G_new + G_new.conj().T)

        # Clip diagonal entries (occupation probabilities) into [0,1]
        diag = np.diag(G_new).copy()
        diag_clipped = np.clip(np.real(diag), 0.0, 1.0)
        for i in range(2 * self.L):
            G_new[i, i] = diag_clipped[i] + 0.0j

        return G_new, xi

    def simulate_trajectory(self) -> Tuple[float, np.ndarray, np.ndarray]:
        r"""Simulate a single measurement trajectory.

        At each time step the correlation matrix ``G`` is first updated
        under Hamiltonian evolution and then under measurement back‑action.
        The magnetisation ``z_i = \langle\sigma^z_i\rangle`` is recorded
        before any update (for ``i=0`` this corresponds to the leftmost
        qubit) and after each combined step.  Entropy production ``Q``
        is accumulated according to the Stratonovich discretisation used
        in the original two‑site simulator:

        .. math::

            Q ← Q + \sum_i \bigl[2ε^2 z_{\mathrm{before},i}
            \frac{z_{\mathrm{before},i} + z_{\mathrm{after},i}}{2}
            + 2ε\,\xi_i\frac{z_{\mathrm{before},i} + z_{\mathrm{after},i}}{2}\bigr].

        Returns
        -------
        Q : float
            Total entropy production accrued along the trajectory.
        z_trajectory : numpy.ndarray
            Array of shape ``(N_steps+1, L)`` containing the magnetisation
            on each site at each time step, including the initial state.
        xi_trajectory : numpy.ndarray
            Array of shape ``(N_steps, L)`` containing the ±1 measurement
            outcomes at each time step.
        """
        G = self.G_initial.copy()
        z_traj = np.zeros((self.N_steps + 1, self.L), dtype=float)
        xi_traj = np.zeros((self.N_steps, self.L), dtype=int)

        # Record initial magnetisation
        z_traj[0] = self._compute_z_values(G)
        Q = 0.0

        for step in range(self.N_steps):
            z_before = z_traj[step]

            # Hamiltonian evolution
            G = self._hamiltonian_step(G)

            # Measurement evolution
            G, xi = self._measurement_step(G)
            xi_traj[step] = xi

            # Compute magnetisation after the step
            z_after = self._compute_z_values(G)
            z_traj[step + 1] = z_after

            # Update entropy production using Stratonovich discretisation
            # Note: loops over each site allow for vectorisation in the future
            for i in range(self.L):
                avg_z = 0.5 * (z_before[i] + z_after[i])
                Q += 2.0 * (self.epsilon ** 2) * z_before[i] * avg_z
                Q += 2.0 * self.epsilon * xi[i] * avg_z

        return Q, z_traj, xi_traj

    def simulate_ensemble(self, n_trajectories: int, progress: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate an ensemble of measurement trajectories.

        Parameters
        ----------
        n_trajectories : int
            Number of independent trajectories to simulate.
        progress : bool, optional
            If ``True`` and `tqdm` is installed, displays a progress bar.

        Returns
        -------
        Q_values : numpy.ndarray
            Array of shape ``(n_trajectories,)`` containing the entropy
            production of each trajectory.
        z_trajectories : numpy.ndarray
            Array of shape ``(n_trajectories, N_steps+1, L)`` containing
            magnetisation series for each trajectory.
        xi_trajectories : numpy.ndarray
            Array of shape ``(n_trajectories, N_steps, L)`` containing
            the measurement outcomes for each trajectory.
        """
        Q_values = np.zeros(n_trajectories, dtype=float)
        z_series = np.zeros((n_trajectories, self.N_steps + 1, self.L), dtype=float)
        xi_series = np.zeros((n_trajectories, self.N_steps, self.L), dtype=int)

        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(n_trajectories), desc="Simulating trajectories")
            except ImportError:
                iterator = range(n_trajectories)
        else:
            iterator = range(n_trajectories)

        for idx in iterator:
            Q, z_traj, xi_traj = self.simulate_trajectory()
            Q_values[idx] = Q
            z_series[idx] = z_traj
            xi_series[idx] = xi_traj

        return Q_values, z_series, xi_series

    def theoretical_prediction(self, A: Optional[list[float]] = None) -> float:
        r"""Estimate the theoretical mean entropy production ⟨Q⟩.

        In the strong‑coupling limit (``|J| → ∞``) the measurement dynamics
        dominates and one can derive an approximate expression for
        ``⟨Q⟩`` based on the average ``A = ⟨z_i^2⟩`` over time and
        sites.  For a two‑site chain the original code assumes
        ``A ≈ 1/2``.  This method generalises the formula to ``L`` sites as

        .. math::

            \langle Q\rangle ≈ 2 T/\tau \sum_i (A_i + 1),

        where ``T/τ = N_steps × ε²`` and we assume the same ``A`` on each
        site.  If ``A`` is omitted a default value of ``1/2`` is used.

        Parameters
        ----------
        A : list[float], optional
            list of average of ``z_i^2`` across time and sites.  Defaults
            to ``0.5`` if not provided.

        Returns
        -------
        float
            Approximate theoretical mean entropy production.
        """
        T_over_tau = self.N_steps * (self.epsilon ** 2)
        if A is None:
            A = 0.5
            # Sum over all sites: each contributes (A + 1)
            return T_over_tau * self.L * (A + 1.0)
        else:
            A_sum = np.sum(A + 1.0)
            return T_over_tau * A_sum
    
    def compute_susceptibility(
        self,
        z_trajectory: np.ndarray,
        site_i: int = 0,
        site_j: Optional[int] = None,
    ) -> float:
        """Compute susceptibility between sites i and j from a z trajectory.
        
        Parameters
        ----------
        z_trajectory : np.ndarray
            Trajectory of shape (N_steps+1, L) containing ⟨σᶻ⟩ for all sites
        site_i : int
            First site index (0 to L-1)
        site_j : int, optional
            Second site index. If None, use site_i (autocorrelation)
        
        Returns
        -------
        float
            Static susceptibility χ_ij
        
        Examples
        --------
        >>> sim = LQubitCorrelationSimulator(L=4)
        >>> Q, z_traj, _ = sim.simulate_trajectory()
        >>> chi = sim.compute_susceptibility(z_traj, site_i=0, site_j=1)
        """
        from quantum_measurement.susceptibility import compute_static_susceptibility
        
        if site_j is None:
            site_j = site_i
        
        times = np.linspace(0, self.T, self.N_steps + 1)
        z_i = z_trajectory[:, site_i]
        z_j = z_trajectory[:, site_j]
        
        chi = compute_static_susceptibility(times, z_i, z_j)
        return chi
    
    def compute_susceptibility_matrix(
        self,
        z_trajectory: np.ndarray,
    ) -> np.ndarray:
        """Compute full L×L susceptibility matrix from a trajectory.
        
        Parameters
        ----------
        z_trajectory : np.ndarray
            Trajectory of shape (N_steps+1, L)
        
        Returns
        -------
        np.ndarray
            Susceptibility matrix χ[i,j] of shape (L, L)
        
        Examples
        --------
        >>> sim = LQubitCorrelationSimulator(L=4)
        >>> Q, z_traj, _ = sim.simulate_trajectory()
        >>> chi_matrix = sim.compute_susceptibility_matrix(z_traj)
        """
        from quantum_measurement.susceptibility import compute_susceptibility_matrix
        
        times = np.linspace(0, self.T, self.N_steps + 1)
        # Transpose to get shape (L, N_steps+1)
        z_traj_transposed = z_trajectory.T
        
        chi_matrix = compute_susceptibility_matrix(times, z_traj_transposed)
        return chi_matrix
    
    def compute_susceptibility_ensemble(
        self,
        z_trajectories: np.ndarray,
        site_i: int = 0,
        site_j: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute average susceptibility over an ensemble of trajectories.
        
        Parameters
        ----------
        z_trajectories : np.ndarray
            Ensemble of trajectories, shape (n_trajectories, N_steps+1, L)
        site_i : int
            First site index (0 to L-1)
        site_j : int, optional
            Second site index. If None, use site_i
        
        Returns
        -------
        Tuple[float, float]
            (mean_chi, std_chi) - mean and standard deviation of susceptibility
        
        Examples
        --------
        >>> sim = LQubitCorrelationSimulator(L=4)
        >>> Q_vals, z_trajs, _ = sim.simulate_ensemble(100)
        >>> mean_chi, std_chi = sim.compute_susceptibility_ensemble(z_trajs, 0, 1)
        """
        n_traj = z_trajectories.shape[0]
        chi_values = np.zeros(n_traj)
        
        for i in range(n_traj):
            chi_values[i] = self.compute_susceptibility(
                z_trajectories[i], site_i, site_j
            )
        
        return np.mean(chi_values), np.std(chi_values)

if __name__ == "__main__":  # pragma: no cover
    # Simple test run when executed directly
    print("LQubitCorrelationSimulator demonstration")
    sim = LQubitCorrelationSimulator(L=3, J=1.0, epsilon=0.1, N_steps=100, T=1.0, closed_boundary=True)
    Q, z_traj, xi_traj = sim.simulate_trajectory()
    print(f"Final magnetisation: {z_traj[-1]}")
    print(f"Total entropy production Q: {Q:.3f}")
