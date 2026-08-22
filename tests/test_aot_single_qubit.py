import numpy as np
from scipy.integrate import quad

from quantum_measurement.aot_single_qubit import (
    derivative_step_sweep,
    dressel_Q_from_final_z,
    dressel_no_drive_pdf,
    dressel_no_drive_reference,
    paired_derivative_validation,
    paired_log_gamma_derivative_validation,
    rademacher_noise,
    simulate_ensemble,
    stationary_aot_density,
    stationary_aot_susceptibility,
    stationary_transition_operator,
)


def test_tangent_matches_pathwise_central_difference():
    noise = rademacher_noise(n_trajectories=8, n_steps=500, seed=187)
    result, central = paired_derivative_validation(
        g=1.0,
        h=1.0e-5,
        noise=noise,
        dt=0.005,
        burn_in=100,
    )

    np.testing.assert_allclose(
        result.tangent_q,
        result.finite_difference_q,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        result.tangent_Q,
        result.finite_difference_Q,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert central.max_norm_error < 2.0e-14
    assert central.max_gauge_error < 2.0e-14


def test_central_difference_has_second_order_convergence():
    noise = rademacher_noise(n_trajectories=4, n_steps=300, seed=911)
    results, _ = derivative_step_sweep(
        g=0.7,
        h_values=[4.0e-3, 2.0e-3, 1.0e-3],
        noise=noise,
        dt=0.005,
        burn_in=50,
    )
    errors = np.array(
        [np.sqrt(np.mean(result.q_difference**2)) for result in results]
    )

    # Halving h should reduce a central difference's O(h^2) error by about 4.
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


def test_full_derivative_contains_explicit_prefactor_term():
    gamma = 4.0
    dt = 0.005
    noise = rademacher_noise(n_trajectories=6, n_steps=400, seed=72)
    observables, _ = simulate_ensemble(
        gamma,
        noise,
        dt=dt,
        burn_in=80,
    )
    measured_steps = noise.shape[1] - 80

    expected = observables.Q + gamma * dt * measured_steps * observables.chi_q
    np.testing.assert_allclose(observables.chi_Q, expected, rtol=1.0e-14, atol=1.0e-14)


def test_explicit_noise_reproduces_identical_runs():
    noise = rademacher_noise(n_trajectories=3, n_steps=100, seed=12)
    first, first_history = simulate_ensemble(
        2.0,
        noise,
        store_history=True,
    )
    second, second_history = simulate_ensemble(
        2.0,
        noise.copy(),
        store_history=True,
    )

    np.testing.assert_array_equal(first.q, second.q)
    np.testing.assert_array_equal(first.chi_q, second.chi_q)
    np.testing.assert_array_equal(first_history["z"], second_history["z"])
    np.testing.assert_array_equal(first_history["u"], second_history["u"])


def test_stationary_transition_operator_preserves_probability():
    operator, _ = stationary_transition_operator(g=1.0, n_grid=256)
    np.testing.assert_allclose(
        np.asarray(operator.sum(axis=0)).ravel(),
        1.0,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.min(operator.data) >= 0.0


def test_deterministic_stationary_reference():
    q = stationary_aot_density(g=1.0, n_grid=2048)
    chi_q = stationary_aot_susceptibility(
        g=1.0,
        h=0.01,
        n_grid=2048,
    )

    np.testing.assert_allclose(q, 1.66530, rtol=0.0, atol=3.0e-5)
    np.testing.assert_allclose(chi_q, 0.18200, rtol=0.0, atol=3.0e-4)


def test_zero_drive_tangent_matches_pathwise_central_difference():
    noise = rademacher_noise(n_trajectories=8, n_steps=500, seed=220507)
    result, central = paired_log_gamma_derivative_validation(
        gamma=1.0,
        h=1.0e-5,
        noise=noise,
        J=0.0,
        dt=1.0 / noise.shape[1],
    )

    np.testing.assert_allclose(
        result.tangent_Q,
        result.finite_difference_Q,
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    assert central.max_norm_error < 2.0e-14
    assert central.max_gauge_error < 2.0e-14
    np.testing.assert_array_less(np.abs(central.final_z), 1.0)


def test_zero_drive_explicit_noise_is_reproducible():
    noise = rademacher_noise(n_trajectories=5, n_steps=200, seed=14)
    first, _ = simulate_ensemble(0.7, noise, J=0.0, dt=0.005)
    second, _ = simulate_ensemble(0.7, noise.copy(), J=0.0, dt=0.005)

    np.testing.assert_array_equal(first.Q, second.Q)
    np.testing.assert_array_equal(first.chi_Q, second.chi_Q)
    np.testing.assert_array_equal(first.final_z, second.final_z)
    assert np.all(dressel_Q_from_final_z(first.final_z) >= 0.0)


def test_dressel_reference_susceptibility_matches_log_central_difference():
    s = 1.3
    h = 1.0e-5
    reference = dressel_no_drive_reference(s)
    plus = dressel_no_drive_reference(s * np.exp(h)).mean_Q
    minus = dressel_no_drive_reference(s * np.exp(-h)).mean_Q
    numerical = (plus - minus) / (2.0 * h)

    np.testing.assert_allclose(reference.chi_Q, numerical, rtol=2.0e-9)


def test_dressel_pdf_normalization_and_mean_match_quadrature():
    for s in (0.2, 1.0, 4.0):
        reference = dressel_no_drive_reference(s)
        normalization = quad(
            lambda Q: dressel_no_drive_pdf(Q, s),
            0.0,
            np.inf,
            epsabs=1.0e-10,
        )[0]
        mean_Q = quad(
            lambda Q: Q * dressel_no_drive_pdf(Q, s),
            0.0,
            np.inf,
            epsabs=1.0e-10,
        )[0]

        np.testing.assert_allclose(normalization, 1.0, rtol=0.0, atol=3.0e-10)
        np.testing.assert_allclose(mean_Q, reference.mean_Q, rtol=3.0e-9)

    tail_density = dressel_no_drive_pdf(np.array([-1.0, 100.0, 10_000.0]), 1.0)
    assert tail_density[0] == 0.0
    assert np.all(np.isfinite(tail_density))
    assert np.all(tail_density >= 0.0)
