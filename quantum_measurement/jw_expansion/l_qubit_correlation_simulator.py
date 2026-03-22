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
from typing import Any, Optional, Tuple
import logging
import warnings
import numpy as np
from scipy.linalg import expm

from quantum_measurement.backends import Backend, get_backend
from quantum_measurement.utilities.gpu_utils import estimate_trajectory_batch_size


_LOGGER = logging.getLogger(__name__)
_TAU_Y_CACHE: dict[int, np.ndarray] = {}


@dataclass
class InstabilityMonitor:
    projector_threshold: float = 1e-10
    hermitian_threshold: float = 1e-13
    bdg_threshold: float = 1e-10
    projector: list[float] = field(default_factory=list)
    hermitian: list[float] = field(default_factory=list)
    bdg: list[float] = field(default_factory=list)

    def append(self, projector: float, hermitian: float, bdg: float) -> None:
        self.projector.append(float(projector))
        self.hermitian.append(float(hermitian))
        self.bdg.append(float(bdg))

    def snapshot(self) -> dict[str, np.ndarray]:
        return {
            "projector": np.asarray(self.projector, dtype=float),
            "hermitian": np.asarray(self.hermitian, dtype=float),
            "bdg": np.asarray(self.bdg, dtype=float),
        }


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
    device: str = "cpu"
    use_stable_integrator: bool = False
    enable_stability_monitor: bool = False
    stable_projector_enforce: bool = True
    backend: Backend | None = field(default=None, repr=False)
    rng: Optional[np.random.Generator] = field(default=None, repr=False)

    # Derived quantities initialised after construction
    dt: float = field(init=False)
    h: Any = field(init=False, repr=False)
    G_initial: Any = field(init=False, repr=False)
    _tau_y: Any = field(init=False, repr=False)
    _identity_2L: Any = field(init=False, repr=False)
    _u_ham: Any | None = field(init=False, default=None, repr=False)
    _stable_mode_active: bool = field(init=False, default=False, repr=False)
    _stability_monitor: InstabilityMonitor | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = get_backend(self.device)

        # Validate input
        if self.L < 1:
            raise ValueError("L must be at least 1")
        if self.N_steps < 1:
            raise ValueError("N_steps must be at least 1")
        if self.T <= 0.0:
            raise ValueError("T must be positive")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")

        # Initialise RNG
        if self.rng is None:
            self.rng = np.random.default_rng()

        # Time step
        self.dt = self.T / self.N_steps
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        # Build the BdG Hamiltonian
        self.h = self._build_hamiltonian()
        self._tau_y = self.backend.array(self._build_tau_y(self.L), dtype=complex)
        self._identity_2L = self.backend.array(np.eye(2 * self.L, dtype=complex), dtype=complex)

        if self.use_stable_integrator and self.backend.is_gpu:
            warnings.warn(
                "Stable integrator is currently CPU-only; falling back to Euler path on GPU backend.",
                RuntimeWarning,
            )
        self._stable_mode_active = bool(self.use_stable_integrator and (not self.backend.is_gpu))
        if self._stable_mode_active:
            h_np = np.asarray(self.backend.asnumpy(self.h), dtype=complex)
            u_left = expm(2.0j * h_np * self.dt)
            u_right = expm(-2.0j * h_np * self.dt)
            self._u_ham = (u_left, u_right)

        if self.enable_stability_monitor:
            self._stability_monitor = InstabilityMonitor()

        # Construct initial correlation matrix: ones on the first L diagonal
        # entries (occupied modes) and zeros on the last L (empty modes).
        self.G_initial = self.backend.zeros((2 * self.L, 2 * self.L), dtype=complex)
        for i in range(self.L):
            self.G_initial[i, i] = 1.0 + 0.0j

    @staticmethod
    def _build_tau_y(L: int) -> np.ndarray:
        cached = _TAU_Y_CACHE.get(L)
        if cached is not None:
            return cached

        eye = np.eye(L, dtype=complex)
        zero = np.zeros((L, L), dtype=complex)
        tau_y = np.block(
            [
                [zero, -1.0j * eye],
                [1.0j * eye, zero],
            ]
        )
        _TAU_Y_CACHE[L] = tau_y
        return tau_y

    def _build_hamiltonian(self) -> Any:
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
        h11 = self.backend.zeros((L, L), dtype=complex)
        h12 = self.backend.zeros((L, L), dtype=complex)
        h21 = self.backend.zeros((L, L), dtype=complex)
        h22 = self.backend.zeros((L, L), dtype=complex)

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
        top = self.backend.hstack((h11, h12))
        bottom = self.backend.hstack((h21, h22))
        h = self.backend.vstack((top, bottom))
        return h

    def _compute_z_values(self, G: Any) -> np.ndarray:
        r"""Compute the expectation values ⟨σ^z_i⟩ for each site.

        In the Jordan–Wigner mapping the magnetisation on site ``i`` is

        .. math::

            \langle\sigma^z_i\rangle = 2\,\mathrm{Re}\,G_{ii} - 1,

        where ``i`` refers to the annihilation index ``0 ≤ i < L``.  The
        returned array has shape ``(L,)``.
        """
        diag_real = self.backend.asnumpy(self.backend.real(self.backend.diag(G)))
        z = 2.0 * diag_real[: self.L] - 1.0
        return z.astype(float)

    def _enforce_bdg(self, G: Any) -> Any:
        G_ph = self._identity_2L - self.backend.matmul(
            self.backend.matmul(self._tau_y, self.backend.conj(G)),
            self._tau_y,
        )
        return 0.5 * (G + G_ph)

    def _enforce_bdg_batch(self, G_batch: Any) -> Any:
        G_ph = self._identity_2L - self.backend.matmul(
            self.backend.matmul(self._tau_y, self.backend.conj(G_batch)),
            self._tau_y,
        )
        return 0.5 * (G_batch + G_ph)

    def _enforce_hermiticity(self, G: Any) -> Any:
        return 0.5 * (G + self.backend.conj(self.backend.transpose(G)))

    def _enforce_hermiticity_batch(self, G_batch: Any) -> Any:
        return 0.5 * (G_batch + self.backend.conj(np.swapaxes(G_batch, -1, -2)))

    def _enforce_projector(self, G: Any) -> Any:
        g_np = np.asarray(self.backend.asnumpy(G), dtype=complex)
        g_herm = 0.5 * (g_np + g_np.conj().T)
        eigvals, eigvecs = np.linalg.eigh(g_herm)
        snapped = (eigvals >= 0.5).astype(float)
        g_proj = (eigvecs * snapped) @ eigvecs.conj().T
        return self.backend.array(g_proj, dtype=complex)

    def _enforce_projector_batch(self, G_batch: Any) -> Any:
        g_np = np.asarray(self.backend.asnumpy(G_batch), dtype=complex)
        out = np.empty_like(g_np)
        for idx in range(g_np.shape[0]):
            g_herm = 0.5 * (g_np[idx] + g_np[idx].conj().T)
            eigvals, eigvecs = np.linalg.eigh(g_herm)
            snapped = (eigvals >= 0.5).astype(float)
            out[idx] = (eigvecs * snapped) @ eigvecs.conj().T
        return self.backend.array(out, dtype=complex)

    def _exact_hamiltonian_step(self, G: Any) -> Any:
        if self._u_ham is None:
            raise RuntimeError("Stable integrator requested without precomputed unitary")
        if not isinstance(self._u_ham, tuple) or len(self._u_ham) != 2:
            raise RuntimeError("Stable integrator precomputation is malformed")
        U_left = self.backend.array(self._u_ham[0], dtype=complex)
        U_right = self.backend.array(self._u_ham[1], dtype=complex)
        return self.backend.matmul(self.backend.matmul(U_left, G), U_right)

    def _exact_hamiltonian_step_batch(self, G_batch: Any) -> Any:
        if self._u_ham is None:
            raise RuntimeError("Stable integrator requested without precomputed unitary")
        return self.backend.batched_commutator_update(
            G_batch,
            self.h,
            self.dt,
            use_stable_integrator=True,
            precomputed_u=self._u_ham,
        )

    def _update_stability_monitor(self, G: Any) -> None:
        if self._stability_monitor is None:
            return

        g_np = np.asarray(self.backend.asnumpy(G), dtype=complex)
        if not np.all(np.isfinite(g_np)):
            raise RuntimeError("InstabilityMonitor detected NaN/Inf in correlation matrix")

        tau_y = np.asarray(self.backend.asnumpy(self._tau_y), dtype=complex)
        identity = np.eye(2 * self.L, dtype=complex)
        projector = np.linalg.norm(g_np @ g_np - g_np, ord="fro")
        hermitian = np.linalg.norm(g_np - g_np.conj().T, ord="fro")
        bdg = np.linalg.norm(tau_y @ g_np.conj() @ tau_y + g_np - identity, ord="fro")

        if not np.isfinite(projector) or not np.isfinite(hermitian) or not np.isfinite(bdg):
            raise RuntimeError("InstabilityMonitor detected NaN/Inf in residuals")

        self._stability_monitor.append(projector, hermitian, bdg)

        if projector > self._stability_monitor.projector_threshold:
            _LOGGER.warning("Projector residual exceeded threshold: %.3e", projector)
        if hermitian > self._stability_monitor.hermitian_threshold:
            _LOGGER.warning("Hermitian residual exceeded threshold: %.3e", hermitian)
        if bdg > self._stability_monitor.bdg_threshold:
            _LOGGER.warning("BdG residual exceeded threshold: %.3e", bdg)

    def get_stability_monitor_data(self) -> dict[str, np.ndarray]:
        if self._stability_monitor is None:
            return {"projector": np.array([], dtype=float), "hermitian": np.array([], dtype=float), "bdg": np.array([], dtype=float)}
        return self._stability_monitor.snapshot()

    def _hamiltonian_step(self, G: Any) -> Any:
        r"""Apply a single time step of unitary evolution under ``h``.

        The Hamiltonian contribution to the equation of motion is

        .. math::

            \dot G = -2\mathrm{i}\,[G, h],

        so over a small time step ``dt`` we perform an explicit Euler
        update ``G ← G + dt * (-2i) * (G@h - h@G)``.  This simple
        discretisation suffices for small ``dt`` and is consistent
        with the implementation used in the original two‑site code.
        """
        if self._stable_mode_active:
            return self._exact_hamiltonian_step(G)

        commutator = self.backend.matmul(G, self.h) - self.backend.matmul(self.h, G)
        dG = -2.0j * self.dt * commutator
        return G + dG

    def _measurement_step(self, G: Any, apply_projection: bool = True) -> Tuple[Any, np.ndarray]:
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
        xi_hat = self.backend.array(self.backend.diag(xi_hat_diag), dtype=complex)

        # Stochastic term proportional to ε
        stochastic = self.epsilon * (
            self.backend.matmul(G, xi_hat)
            + self.backend.matmul(xi_hat, G)
            - 2.0 * self.backend.matmul(self.backend.matmul(G, xi_hat), G)
        )
        # Deterministic damping term proportional to ε²
        G_diag = self.backend.diag(self.backend.diag(G))
        damping = - (self.epsilon ** 2) * (G - G_diag)

        G_new = G + stochastic + damping

        if apply_projection:
            # Symmetrise to counteract numerical drift and enforce Hermiticity
            G_new = 0.5 * (G_new + self.backend.conj(self.backend.transpose(G_new)))

            # Clip diagonal entries (occupation probabilities) into [0,1]
            diag = self.backend.real(self.backend.diag(G_new))
            diag_clipped = self.backend.asnumpy(self.backend.clip(diag, 0.0, 1.0))
            for i in range(2 * self.L):
                G_new[i, i] = float(diag_clipped[i]) + 0.0j

        return G_new, xi

    def _compute_z_values_batch(self, G_batch: Any) -> np.ndarray:
        """Compute z values for a batch of correlation matrices.

        Parameters
        ----------
        G_batch : backend array
            Shape (n_batch, 2L, 2L)
        """
        diag = self.backend.real(G_batch[:, range(self.L), range(self.L)])
        z = 2.0 * self.backend.asnumpy(diag) - 1.0
        return z.astype(float)

    def _hamiltonian_step_batch(self, G_batch: Any) -> Any:
        """Apply Hamiltonian step to a batch of trajectories."""
        if self._stable_mode_active:
            return self._exact_hamiltonian_step_batch(G_batch)

        return self.backend.batched_commutator_update(G_batch, self.h, self.dt)

    def _measurement_step_batch(self, G_batch: Any, xi_step: Any, apply_projection: bool = True) -> Any:
        """Apply measurement evolution to a batch for one time step."""
        n_batch = int(G_batch.shape[0])
        dim = 2 * self.L

        xi_hat = self.backend.get_workspace("lq_xi_hat", (n_batch, dim, dim), complex)
        for site in range(self.L):
            xi_hat[:, site, site] = xi_step[:, site]
            xi_hat[:, site + self.L, site + self.L] = -xi_step[:, site]

        stochastic = self.epsilon * (
            self.backend.matmul(G_batch, xi_hat)
            + self.backend.matmul(xi_hat, G_batch)
            - 2.0 * self.backend.matmul(self.backend.matmul(G_batch, xi_hat), G_batch)
        )

        damping = - (self.epsilon ** 2) * G_batch

        G_new = G_batch + stochastic + damping
        for idx in range(dim):
            G_new[:, idx, idx] += (self.epsilon ** 2) * G_batch[:, idx, idx]

        if apply_projection:
            G_new = self.backend.symmetrize_clip_diag_inplace(G_new)

        return G_new

    def _stable_step_single(self, G: Any) -> Tuple[Any, np.ndarray]:
        G = self._hamiltonian_step(G)
        G = self._enforce_bdg(G)
        G, xi = self._measurement_step(G, apply_projection=False)
        G = self._enforce_bdg(G)
        G = self._enforce_hermiticity(G)
        if self.stable_projector_enforce:
            G = self._enforce_projector(G)
            G = self._enforce_bdg(G)
            G = self._enforce_hermiticity(G)
        return G, xi

    def _stable_step_batch(self, G_batch: Any, xi_step: Any) -> Any:
        G_batch = self._hamiltonian_step_batch(G_batch)
        G_batch = self._enforce_bdg_batch(G_batch)
        G_batch = self._measurement_step_batch(G_batch, xi_step, apply_projection=False)
        G_batch = self._enforce_bdg_batch(G_batch)
        G_batch = self._enforce_hermiticity_batch(G_batch)
        if self.stable_projector_enforce:
            G_batch = self._enforce_projector_batch(G_batch)
            G_batch = self._enforce_bdg_batch(G_batch)
            G_batch = self._enforce_hermiticity_batch(G_batch)
        return G_batch

    def simulate_trajectory_batch(
        self,
        n_batch: int,
        xi_batch: Any | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate a batch of measurement trajectories in parallel.

        Parameters
        ----------
        n_batch : int
            Number of trajectories to simulate in this batch.
        xi_batch : backend/NumPy array | None
            Optional pre-generated ±1 measurement outcomes with shape
            (n_batch, N_steps, L).
        """
        if n_batch < 1:
            raise ValueError("n_batch must be at least 1")

        if xi_batch is None:
            xi_batch_dev = self.backend.choice_pm1((n_batch, self.N_steps, self.L))
        else:
            xi_batch_dev = self.backend.array(xi_batch)

        G_batch = self.backend.zeros((n_batch, 2 * self.L, 2 * self.L), dtype=complex)
        for idx in range(n_batch):
            G_batch[idx] = self.G_initial

        z_traj = np.zeros((n_batch, self.N_steps + 1, self.L), dtype=float)
        xi_traj = np.zeros((n_batch, self.N_steps, self.L), dtype=int)
        Q = np.zeros(n_batch, dtype=float)

        z_traj[:, 0, :] = self._compute_z_values_batch(G_batch)

        for step in range(self.N_steps):
            z_before = z_traj[:, step, :]
            xi_step = xi_batch_dev[:, step, :]
            if self._stable_mode_active:
                G_batch = self._stable_step_batch(G_batch, xi_step)
            else:
                G_batch = self._hamiltonian_step_batch(G_batch)
                G_batch = self._measurement_step_batch(G_batch, xi_step)
            xi_step_np = self.backend.asnumpy(xi_step).astype(int)
            xi_traj[:, step, :] = xi_step_np

            z_after = self._compute_z_values_batch(G_batch)
            if self._stability_monitor is not None:
                self._update_stability_monitor(G_batch[0])
            z_traj[:, step + 1, :] = z_after

            avg_z = 0.5 * (z_before + z_after)
            Q += np.sum(2.0 * (self.epsilon ** 2) * z_before * avg_z, axis=1)
            Q += np.sum(2.0 * self.epsilon * xi_step_np * avg_z, axis=1)

        return Q, z_traj, xi_traj

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
        G = self.backend.copy(self.G_initial)
        z_traj = np.zeros((self.N_steps + 1, self.L), dtype=float)
        xi_traj = np.zeros((self.N_steps, self.L), dtype=int)

        # Record initial magnetisation
        z_traj[0] = self._compute_z_values(G)
        Q = 0.0

        for step in range(self.N_steps):
            z_before = z_traj[step]
            if self._stable_mode_active:
                G, xi = self._stable_step_single(G)
            else:
                # Hamiltonian evolution
                G = self._hamiltonian_step(G)

                # Measurement evolution
                G, xi = self._measurement_step(G)
            xi_traj[step] = xi
            self._update_stability_monitor(G)

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

    def simulate_z2_mean(self) -> float:
        r"""Simulate a single trajectory and return mean z^2 over time and sites.

        This method avoids storing the full trajectory to reduce memory usage
        for long runs. The returned value is equivalent to
        ``np.average(z_traj**2)`` from ``simulate_trajectory``.

        Returns
        -------
        float
            Mean of z^2 over all sites and time steps.
        """
        G = self.backend.copy(self.G_initial)

        z = self._compute_z_values(G)
        sum_z2 = float(np.sum(z ** 2))

        for step in range(self.N_steps):
            if self._stable_mode_active:
                G, _ = self._stable_step_single(G)
            else:
                G = self._hamiltonian_step(G)
                G, _ = self._measurement_step(G)
            self._update_stability_monitor(G)
            z = self._compute_z_values(G)
            if not np.all(np.isfinite(z)):
                return float("nan")
            sum_z2 += float(np.sum(z ** 2))

        total_samples = (self.N_steps + 1) * self.L
        return sum_z2 / total_samples

    def simulate_z2_mean_ensemble(
        self,
        n_trajectories: int,
        batch_size: int | None = None,
        return_std_err: bool = False,
    ) -> float | tuple[float, float, float]:
        r"""Compute mean z^2 over an ensemble without storing full trajectories.

        This method is designed for long parameter sweeps where retaining full
        trajectory tensors is unnecessary. It uses the batched trajectory update
        path and backend RNG (`choice_pm1`), enabling fully device-side random
        generation for GPU runs.

        Parameters
        ----------
        n_trajectories : int
            Number of independent trajectories to average.
        batch_size : int | None, optional
            Number of trajectories per batch. If ``None``, GPU uses
            ``estimate_trajectory_batch_size(L)`` and CPU defaults to 1.
        return_std_err : bool, optional
            If ``True``, also return the sample standard deviation and standard
            error of per-trajectory z^2 means.

        Returns
        -------
        float | tuple[float, float, float]
            Mean z^2 over trajectories. If ``return_std_err`` is true, returns
            ``(mean, std, stderr)``.
        """
        if n_trajectories < 1:
            raise ValueError("n_trajectories must be at least 1")

        if batch_size is None:
            if self.backend.is_gpu:
                batch_size = min(n_trajectories, estimate_trajectory_batch_size(self.L))
            else:
                batch_size = 1
        batch_size = max(1, int(batch_size))

        samples_per_traj = (self.N_steps + 1) * self.L
        mean_sum = 0.0
        mean_sq_sum = 0.0
        counted = 0

        for start_idx in range(0, n_trajectories, batch_size):
            end_idx = min(start_idx + batch_size, n_trajectories)
            current_batch = end_idx - start_idx

            G_batch = self.backend.zeros((current_batch, 2 * self.L, 2 * self.L), dtype=complex)
            for idx in range(current_batch):
                G_batch[idx] = self.G_initial

            z_batch = self._compute_z_values_batch(G_batch)
            if not np.all(np.isfinite(z_batch)):
                if return_std_err:
                    return float("nan"), float("nan"), float("nan")
                return float("nan")
            z2_sum = np.sum(z_batch ** 2, axis=1)

            for step in range(self.N_steps):
                xi_step = self.backend.choice_pm1((current_batch, self.L))
                if self._stable_mode_active:
                    G_batch = self._stable_step_batch(G_batch, xi_step)
                else:
                    G_batch = self._hamiltonian_step_batch(G_batch)
                    G_batch = self._measurement_step_batch(G_batch, xi_step)
                if self._stability_monitor is not None:
                    self._update_stability_monitor(G_batch[0])
                z_batch = self._compute_z_values_batch(G_batch)
                if not np.all(np.isfinite(z_batch)):
                    if return_std_err:
                        return float("nan"), float("nan"), float("nan")
                    return float("nan")
                z2_sum += np.sum(z_batch ** 2, axis=1)

            traj_means = z2_sum / samples_per_traj
            mean_sum += float(np.sum(traj_means))
            counted += current_batch

            if return_std_err:
                mean_sq_sum += float(np.sum(traj_means ** 2))

        mean_z2 = mean_sum / max(counted, 1)
        if not return_std_err:
            return mean_z2

        var = max((mean_sq_sum / max(counted, 1)) - (mean_z2 ** 2), 0.0)
        std = float(np.sqrt(var))
        stderr = std / float(np.sqrt(max(counted, 1)))
        return mean_z2, std, stderr

    def simulate_ensemble(
        self,
        n_trajectories: int,
        progress: bool = False,
        batch_size: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        if batch_size is None:
            if self.backend.is_gpu:
                batch_size = min(n_trajectories, estimate_trajectory_batch_size(self.L))
            else:
                batch_size = 1
        batch_size = max(1, int(batch_size))

        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, n_trajectories, batch_size), desc="Simulating trajectories")
            except ImportError:
                iterator = range(0, n_trajectories, batch_size)
        else:
            iterator = range(0, n_trajectories, batch_size)

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n_trajectories)
            current_batch = end_idx - start_idx
            Q_batch, z_batch, xi_batch = self.simulate_trajectory_batch(current_batch)
            Q_values[start_idx:end_idx] = Q_batch
            z_series[start_idx:end_idx] = z_batch
            xi_series[start_idx:end_idx] = xi_batch

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

if __name__ == "__main__":  # pragma: no cover
    # Simple test run when executed directly
    print("LQubitCorrelationSimulator demonstration")
    sim = LQubitCorrelationSimulator(L=3, J=1.0, epsilon=0.1, N_steps=100, T=1.0, closed_boundary=True)
    Q, z_traj, xi_traj = sim.simulate_trajectory()
    print(f"Final magnetisation: {z_traj[-1]}")
    print(f"Total entropy production Q: {Q:.3f}")
