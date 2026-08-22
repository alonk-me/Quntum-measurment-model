"""Single-qubit tangent-state validation for the average arrow of time.

The routines in this module implement the same weak-measurement map used by
``notebooks/aot_single_qubit_density_susceptibility_v2.ipynb``.  The main
difference is that the primitive Rademacher noise is passed explicitly and an
ensemble of trajectories is propagated in parallel.  Explicit noise is
important for a paired finite-difference validation: the trajectories at
``theta``, ``theta + h``, and ``theta - h`` must see the same noise path.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigs


I2 = np.eye(2, dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


@dataclass(frozen=True)
class EnsembleObservables:
    """Per-trajectory observables for one value of the measurement strength."""

    q: np.ndarray
    chi_q: np.ndarray
    Q: np.ndarray
    chi_Q: np.ndarray
    final_z: np.ndarray
    max_tangent_norm: np.ndarray
    max_norm_error: float
    max_gauge_error: float


@dataclass(frozen=True)
class PairedDerivativeResult:
    """Tangent and paired central-difference estimates at one step size."""

    h: float
    tangent_q: np.ndarray
    finite_difference_q: np.ndarray
    tangent_Q: np.ndarray
    finite_difference_Q: np.ndarray

    @property
    def q_difference(self) -> np.ndarray:
        return self.tangent_q - self.finite_difference_q

    @property
    def Q_difference(self) -> np.ndarray:
        return self.tangent_Q - self.finite_difference_Q


@dataclass(frozen=True)
class DresselNoDriveReference:
    """Analytical no-drive AoT mean and logarithmic susceptibility."""

    s: float
    mean_Q: float
    chi_Q: float


def dressel_no_drive_reference(
    s: float,
    *,
    quadrature_order: int = 128,
) -> DresselNoDriveReference:
    """Return the exact Dressel no-drive reference for an equatorial qubit.

    With ``s = T / tau``, the integrated measurement signal can be sampled as
    ``Gamma = s + sqrt(s) Z`` for a standard normal ``Z`` when evaluating the
    even function ``Q = 2 log(cosh(Gamma))``.  Differentiating this Gaussian
    expectation gives the susceptibility with respect to ``ln(s)`` (and hence
    ``ln(gamma)`` at fixed observation time).
    """

    if s <= 0:
        raise ValueError("s must be positive")
    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least 8")

    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    gamma_record = s + np.sqrt(2.0 * s) * nodes
    normalized_weights = weights / np.sqrt(np.pi)
    log_cosh = np.logaddexp(gamma_record, -gamma_record) - np.log(2.0)
    tanh_gamma = np.tanh(gamma_record)
    sech_squared = 1.0 - tanh_gamma * tanh_gamma

    mean_Q = float(np.dot(normalized_weights, 2.0 * log_cosh))
    chi_Q = float(
        s * np.dot(normalized_weights, 2.0 * tanh_gamma + sech_squared)
    )
    return DresselNoDriveReference(s=float(s), mean_Q=mean_Q, chi_Q=chi_Q)


def dressel_no_drive_pdf(Q: np.ndarray | float, s: float) -> np.ndarray | float:
    """Evaluate Dressel et al. Eq. 14 for ``Q >= 0``.

    The density has an integrable inverse-square-root singularity at zero.
    Negative values are outside its support and return zero.
    """

    if s <= 0:
        raise ValueError("s must be positive")
    Q_array = np.asarray(Q, dtype=float)
    density = np.zeros_like(Q_array)
    positive = Q_array > 0.0
    if np.any(positive):
        x = Q_array[positive]
        with np.errstate(invalid="ignore", divide="ignore"):
            # arccosh(exp(x/2)) written without constructing exp(x/2).
            inverse = 0.5 * x + np.log1p(np.sqrt(-np.expm1(-x)))
            log_expm1 = np.empty_like(x)
            moderate = x < 50.0
            log_expm1[moderate] = np.log(np.expm1(x[moderate]))
            log_expm1[~moderate] = x[~moderate] + np.log1p(
                -np.exp(-x[~moderate])
            )
            log_density = (
                -0.5 * np.log(2.0 * np.pi * s)
                + x
                - 0.5 * log_expm1
                - 0.5 * s
                - inverse * inverse / (2.0 * s)
            )
            density[positive] = np.exp(log_density)
    density[Q_array == 0.0] = np.inf
    if Q_array.ndim == 0:
        return float(density)
    return density


def dressel_Q_from_final_z(
    final_z: np.ndarray | float,
    *,
    clip: float = np.finfo(float).tiny,
) -> np.ndarray | float:
    """Compute ``Q = -log(1-z_T^2)`` from the final no-drive state."""

    if clip <= 0:
        raise ValueError("clip must be positive")
    z_array = np.asarray(final_z, dtype=float)
    one_minus_z_squared = np.maximum(1.0 - z_array * z_array, clip)
    Q = -np.log(one_minus_z_squared)
    if z_array.ndim == 0:
        return float(Q)
    return Q


def exact_unitary(J: float, dt: float) -> np.ndarray:
    """Return ``exp(-1j * J * sigma_y * dt)`` by Hermitian diagonalization."""

    hamiltonian = J * SIGMA_Y
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * eigenvalues * dt)
    return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T


def rademacher_noise(
    n_trajectories: int,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    """Generate the parameter-independent primitive noise used by the map."""

    if n_trajectories <= 0 or n_steps <= 0:
        raise ValueError("n_trajectories and n_steps must be positive")
    rng = np.random.default_rng(seed)
    return rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=(n_trajectories, n_steps),
    )


def bloch_angle_map(
    phi: np.ndarray,
    gamma: float,
    xi: float,
    *,
    J: float = 1.0,
    dt: float = 0.005,
) -> np.ndarray:
    """Apply one notebook step to real states on the Bloch circle.

    The notebook's initial state, Hamiltonian, and measurement update preserve
    real state vectors.  Writing ``psi=(cos(phi/2), sin(phi/2))`` gives
    ``z=cos(phi)``.  This map supplies an independent one-dimensional
    transition-operator reference for the stochastic trajectory simulation.
    """

    if gamma <= 0 or J < 0 or dt <= 0:
        raise ValueError("gamma and dt must be positive, and J must be nonnegative")
    if xi not in (-1.0, 1.0):
        raise ValueError("xi must be -1 or +1")
    phi = np.asarray(phi, dtype=float)
    z = np.cos(phi)
    epsilon = np.sqrt(gamma * dt)
    a_zero = 0.5 * (1.0 - z)
    a_one = -0.5 * (1.0 + z)
    measurement_zero = (
        1.0 + xi * epsilon * a_zero - 0.5 * epsilon**2 * a_zero**2
    )
    measurement_one = (
        1.0 + xi * epsilon * a_one - 0.5 * epsilon**2 * a_one**2
    )
    half_angle = np.arctan2(
        measurement_one * np.sin(0.5 * phi),
        measurement_zero * np.cos(0.5 * phi),
    )
    return np.mod(2.0 * half_angle + 2.0 * J * dt, 2.0 * np.pi)


def stationary_transition_operator(
    g: float,
    *,
    n_grid: int = 4096,
    J: float = 1.0,
    dt: float = 0.005,
) -> tuple[csr_matrix, np.ndarray]:
    """Build a mass-preserving Ulam operator for the discrete one-qubit map."""

    if g <= 0 or J <= 0 or dt <= 0:
        raise ValueError("g, J, and dt must be positive")
    if n_grid < 32:
        raise ValueError("n_grid must be at least 32")

    phi = 2.0 * np.pi * np.arange(n_grid) / n_grid
    spacing = 2.0 * np.pi / n_grid
    source = np.arange(n_grid)
    rows = []
    columns = []
    values = []
    gamma = 4.0 * J * g
    for xi in (-1.0, 1.0):
        mapped = bloch_angle_map(phi, gamma, xi, J=J, dt=dt)
        grid_coordinate = mapped / spacing
        floor_coordinate = np.floor(grid_coordinate)
        lower = floor_coordinate.astype(int) % n_grid
        fraction = grid_coordinate - floor_coordinate
        upper = (lower + 1) % n_grid
        rows.extend((lower, upper))
        columns.extend((source, source))
        values.extend((0.5 * (1.0 - fraction), 0.5 * fraction))

    operator = coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(n_grid, n_grid),
    ).tocsr()
    return operator, phi


def stationary_aot_density(
    g: float,
    *,
    n_grid: int = 4096,
    J: float = 1.0,
    dt: float = 0.005,
    return_distribution: bool = False,
) -> float | tuple[float, np.ndarray, np.ndarray]:
    """Compute stationary ``q=<1+z^2>`` without trajectory sampling.

    The invariant density is the unit eigenvector of the one-step transfer
    operator.  Requesting several eigenpairs and selecting the value nearest
    one avoids accidentally selecting a slowly decaying complex mode.
    """

    operator, phi = stationary_transition_operator(
        g,
        n_grid=n_grid,
        J=J,
        dt=dt,
    )
    eigenvalues, eigenvectors = eigs(
        operator,
        k=3,
        which="LR",
        v0=np.ones(n_grid),
        tol=1.0e-12,
        maxiter=500_000,
    )
    stationary_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    eigenvalue = eigenvalues[stationary_index]
    if abs(eigenvalue - 1.0) > 1.0e-9:
        raise RuntimeError(f"stationary eigenvalue did not converge: {eigenvalue}")
    vector = eigenvectors[:, stationary_index]
    if np.max(np.abs(np.imag(vector))) > 1.0e-9:
        raise RuntimeError("stationary eigenvector has a significant imaginary part")
    probability = np.real(vector)
    probability *= np.sign(np.sum(probability))
    negative_tolerance = 1.0e-10 * np.max(np.abs(probability))
    if np.min(probability) < -negative_tolerance:
        raise RuntimeError("stationary eigenvector is not nonnegative")
    probability = np.maximum(probability, 0.0)
    probability /= np.sum(probability)
    residual = np.linalg.norm(operator @ probability - probability, ord=1)
    if residual > 1.0e-8:
        raise RuntimeError(f"stationary distribution residual is too large: {residual}")

    q = float(np.dot(probability, 1.0 + np.cos(phi) ** 2))
    if return_distribution:
        return q, probability, phi
    return q


def stationary_aot_susceptibility(
    g: float,
    *,
    h: float = 0.01,
    n_grid: int = 4096,
    J: float = 1.0,
    dt: float = 0.005,
) -> float:
    """Differentiate the deterministic stationary reference in ``ln(g)``."""

    if h <= 0:
        raise ValueError("h must be positive")
    q_plus = stationary_aot_density(
        g * np.exp(h), n_grid=n_grid, J=J, dt=dt
    )
    q_minus = stationary_aot_density(
        g * np.exp(-h), n_grid=n_grid, J=J, dt=dt
    )
    return float((q_plus - q_minus) / (2.0 * h))


def _validate_noise(noise: np.ndarray) -> np.ndarray:
    noise = np.asarray(noise)
    if noise.ndim == 1:
        noise = noise[np.newaxis, :]
    if noise.ndim != 2 or noise.shape[1] == 0:
        raise ValueError("noise must have shape (n_trajectories, n_steps)")
    if not np.all((noise == -1) | (noise == 1)):
        raise ValueError("noise entries must all be -1 or +1")
    return noise


def _initial_states(n_trajectories: int, psi0: Optional[np.ndarray]) -> np.ndarray:
    if psi0 is None:
        state = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    else:
        state = np.asarray(psi0, dtype=complex)
        if state.shape != (2,):
            raise ValueError("psi0 must have shape (2,)")
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("psi0 must be nonzero")
        state = state / norm
    return np.broadcast_to(state, (n_trajectories, 2)).copy()


def simulate_ensemble(
    gamma: float,
    noise: np.ndarray,
    *,
    J: float = 1.0,
    dt: float = 0.005,
    burn_in: int = 0,
    psi0: Optional[np.ndarray] = None,
    propagate_tangent: bool = True,
    store_history: bool = False,
) -> tuple[EnsembleObservables, Optional[dict[str, np.ndarray]]]:
    """Propagate a paired-noise ensemble using the notebook measurement map.

    ``z`` and ``u`` are sampled immediately before each measurement update, as
    in the notebook.  The tangent is initialized to zero because ``psi0`` is
    parameter independent.  The differentiation variable is ``ln(gamma)``.
    This is also ``ln(g)`` plus a constant when a fixed nonzero Hamiltonian
    scale ``J`` is present.
    """

    if gamma <= 0 or J < 0 or dt <= 0:
        raise ValueError("gamma and dt must be positive, and J must be nonnegative")
    noise = _validate_noise(noise)
    n_trajectories, n_steps = noise.shape
    if not 0 <= burn_in < n_steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps")

    psi = _initial_states(n_trajectories, psi0)
    eta = np.zeros_like(psi)
    unitary = exact_unitary(J, dt)
    epsilon = np.sqrt(gamma * dt)
    epsilon_squared = gamma * dt

    q_sum = np.zeros(n_trajectories)
    chi_q_sum = np.zeros(n_trajectories)
    max_tangent_norm = np.zeros(n_trajectories)
    max_norm_error = 0.0
    max_gauge_error = 0.0

    history: Optional[dict[str, np.ndarray]] = None
    if store_history:
        history = {
            "z": np.empty((n_trajectories, n_steps)),
            "u": np.empty((n_trajectories, n_steps)),
        }

    for step in range(n_steps):
        z = np.abs(psi[:, 0]) ** 2 - np.abs(psi[:, 1]) ** 2
        if propagate_tangent:
            u = 2.0 * np.real(
                np.conj(eta[:, 0]) * psi[:, 0]
                - np.conj(eta[:, 1]) * psi[:, 1]
            )
        else:
            u = np.zeros(n_trajectories)

        if history is not None:
            history["z"][:, step] = z
            history["u"][:, step] = u
        if step >= burn_in:
            q_sum += 1.0 + z * z
            if propagate_tangent:
                chi_q_sum += 2.0 * z * u

        # A=(sigma_z-zI)/2 is diagonal for one qubit.  Keeping its two
        # diagonal entries explicitly makes the ensemble update inexpensive.
        a = np.column_stack((0.5 * (1.0 - z), -0.5 * (1.0 + z)))
        xi = noise[:, step].astype(float)[:, np.newaxis]
        measurement = (
            1.0
            + xi * epsilon * a
            - 0.5 * epsilon_squared * a * a
        )

        v = measurement * psi
        if propagate_tangent:
            a_theta = -0.5 * u[:, np.newaxis]
            measurement_theta = (
                xi * 0.5 * epsilon * a
                + xi * epsilon * a_theta
                - 0.5 * epsilon_squared * a * a
                - 0.5 * epsilon_squared * (2.0 * a * a_theta)
            )
            w = measurement * eta + measurement_theta * psi

        norm = np.linalg.norm(v, axis=1)
        psi = v / norm[:, np.newaxis]
        if propagate_tangent:
            projection = np.real(np.sum(np.conj(psi) * w, axis=1))
            eta = (w - psi * projection[:, np.newaxis]) / norm[:, np.newaxis]

        psi = psi @ unitary.T
        if propagate_tangent:
            eta = eta @ unitary.T
            # Remove accumulated roundoff in the real normalization gauge.
            projection = np.real(np.sum(np.conj(psi) * eta, axis=1))
            eta -= psi * projection[:, np.newaxis]
            max_tangent_norm = np.maximum(
                max_tangent_norm,
                np.linalg.norm(eta, axis=1),
            )
            max_gauge_error = max(
                max_gauge_error,
                float(np.max(np.abs(np.real(np.sum(np.conj(psi) * eta, axis=1))))),
            )
        max_norm_error = max(
            max_norm_error,
            float(np.max(np.abs(np.sum(np.abs(psi) ** 2, axis=1) - 1.0))),
        )

    measured_steps = n_steps - burn_in
    q = q_sum / measured_steps
    chi_q = chi_q_sum / measured_steps
    Q = epsilon_squared * q_sum
    chi_Q = Q + epsilon_squared * chi_q_sum
    observables = EnsembleObservables(
        q=q,
        chi_q=chi_q,
        Q=Q,
        chi_Q=chi_Q,
        final_z=np.abs(psi[:, 0]) ** 2 - np.abs(psi[:, 1]) ** 2,
        max_tangent_norm=max_tangent_norm,
        max_norm_error=max_norm_error,
        max_gauge_error=max_gauge_error,
    )
    return observables, history


def paired_derivative_validation(
    g: float,
    h: float,
    noise: np.ndarray,
    *,
    J: float = 1.0,
    dt: float = 0.005,
    burn_in: int = 0,
    central: Optional[EnsembleObservables] = None,
) -> tuple[PairedDerivativeResult, EnsembleObservables]:
    """Compare tangent derivatives to a same-noise central difference."""

    if g <= 0 or J <= 0:
        raise ValueError("g and J must be positive")
    gamma = 4.0 * J * g
    return paired_log_gamma_derivative_validation(
        gamma,
        h,
        noise,
        J=J,
        dt=dt,
        burn_in=burn_in,
        central=central,
    )


def paired_log_gamma_derivative_validation(
    gamma: float,
    h: float,
    noise: np.ndarray,
    *,
    J: float = 0.0,
    dt: float = 0.005,
    burn_in: int = 0,
    central: Optional[EnsembleObservables] = None,
) -> tuple[PairedDerivativeResult, EnsembleObservables]:
    """Compare tangent and paired derivatives with respect to ``ln(gamma)``."""

    if gamma <= 0 or h <= 0:
        raise ValueError("gamma and h must be positive")
    if J < 0:
        raise ValueError("J must be nonnegative")
    if central is None:
        central, _ = simulate_ensemble(
            gamma,
            noise,
            J=J,
            dt=dt,
            burn_in=burn_in,
        )

    plus, _ = simulate_ensemble(
        gamma * np.exp(h),
        noise,
        J=J,
        dt=dt,
        burn_in=burn_in,
        propagate_tangent=False,
    )
    minus, _ = simulate_ensemble(
        gamma * np.exp(-h),
        noise,
        J=J,
        dt=dt,
        burn_in=burn_in,
        propagate_tangent=False,
    )
    result = PairedDerivativeResult(
        h=h,
        tangent_q=central.chi_q,
        finite_difference_q=(plus.q - minus.q) / (2.0 * h),
        tangent_Q=central.chi_Q,
        finite_difference_Q=(plus.Q - minus.Q) / (2.0 * h),
    )
    return result, central


def derivative_step_sweep(
    g: float,
    h_values: Iterable[float],
    noise: np.ndarray,
    *,
    J: float = 1.0,
    dt: float = 0.005,
    burn_in: int = 0,
) -> tuple[list[PairedDerivativeResult], EnsembleObservables]:
    """Run a central-difference step-size sweep with one shared tangent run."""

    gamma = 4.0 * J * g
    central, _ = simulate_ensemble(
        gamma,
        noise,
        J=J,
        dt=dt,
        burn_in=burn_in,
    )
    results = []
    for h in h_values:
        result, _ = paired_derivative_validation(
            g,
            float(h),
            noise,
            J=J,
            dt=dt,
            burn_in=burn_in,
            central=central,
        )
        results.append(result)
    return results, central


def mean_and_sem(values: np.ndarray) -> tuple[float, float]:
    """Return the sample mean and standard error across trajectories."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty one-dimensional array")
    sem = 0.0 if values.size == 1 else float(np.std(values, ddof=1) / np.sqrt(values.size))
    return float(np.mean(values)), sem


def paired_summary(result: PairedDerivativeResult) -> dict[str, float]:
    """Summarize accuracy and sampling uncertainty for ``chi_q``."""

    tangent_mean, tangent_sem = mean_and_sem(result.tangent_q)
    finite_difference_mean, finite_difference_sem = mean_and_sem(
        result.finite_difference_q
    )
    difference_mean, difference_sem = mean_and_sem(result.q_difference)
    return {
        "h": result.h,
        "tangent_mean": tangent_mean,
        "tangent_sem": tangent_sem,
        "finite_difference_mean": finite_difference_mean,
        "finite_difference_sem": finite_difference_sem,
        "difference_mean": difference_mean,
        "difference_sem": difference_sem,
        "difference_rms": float(np.sqrt(np.mean(result.q_difference**2))),
        "difference_max_abs": float(np.max(np.abs(result.q_difference))),
        "tangent_std": float(np.std(result.tangent_q, ddof=1)),
        "finite_difference_std": float(
            np.std(result.finite_difference_q, ddof=1)
        ),
    }


def timed_simulation(*args, **kwargs) -> tuple[EnsembleObservables, float]:
    """Run ``simulate_ensemble`` and return wall-clock seconds."""

    start = perf_counter()
    result, _ = simulate_ensemble(*args, **kwargs)
    return result, perf_counter() - start
