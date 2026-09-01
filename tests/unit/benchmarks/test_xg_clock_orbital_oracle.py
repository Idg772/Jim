import numpy as np

from benchmarks.xg.oracles.clock_orbital_oracle import (
    adaptive_stationary_time,
    fisher_projected_bias_sigma,
    noise_weighted_inner_product,
    profile_waveform_residual,
    project_orbital_delay,
    remove_affine_orbital_motion,
    response_impact_delta_log_l,
    time_to_coalescence_from_stationary_time,
)


def test_adaptive_phase_derivative_recovers_injected_stationary_time():
    frequencies = np.asarray([5.0, 20.0, 100.0])
    constant_time = -100.0
    curvature = 1.0e-4

    def carrier(frequency):
        amplitude = 1.0 + 2.0e-4 * frequency
        phase = (
            -2.0 * np.pi * (constant_time * frequency + curvature * frequency**3 / 3.0)
        )
        return amplitude * np.exp(1j * phase)

    result = adaptive_stationary_time(
        carrier,
        frequencies,
        initial_step_hz=0.2,
        absolute_tolerance_s=2.0e-8,
        relative_tolerance=1.0e-11,
    )
    expected = constant_time + curvature * frequencies**2

    np.testing.assert_allclose(result.stationary_time_s, expected, atol=2.0e-8)
    assert np.all(result.estimated_abs_error_s <= 3.0e-8)
    assert np.all(result.final_step_hz < 0.2)
    assert np.all(result.refinement_count >= 4)


def test_time_to_coalescence_is_nonnegative_and_freezes_at_cutoff():
    stationary_time = np.asarray([-8.0, -0.2, 0.3, -0.4])
    frequency = np.asarray([10.0, 100.0, 500.0, 1_000.0])

    clock = time_to_coalescence_from_stationary_time(
        stationary_time,
        coalescence_time_s=0.0,
        frequency_hz=frequency,
        cutoff_frequency_hz=800.0,
    )

    np.testing.assert_array_equal(clock, np.asarray([8.0, 0.2, 0.0, 0.0]))
    assert np.all(clock >= 0.0)


def test_response_impact_normalizes_reference_to_declared_snr():
    reference = np.asarray([1.0 + 0.0j, 0.0 + 2.0j])
    candidate = reference + np.asarray([0.1j, -0.2 + 0.0j])
    weights = np.asarray([2.0, 0.5])

    result = response_impact_delta_log_l(
        reference,
        candidate,
        weights,
        target_snr=20.0,
    )

    reference_norm = noise_weighted_inner_product(reference, reference, weights)
    scale = 20.0 / np.sqrt(reference_norm)
    expected = (
        0.5
        * scale**2
        * noise_weighted_inner_product(
            candidate - reference,
            candidate - reference,
            weights,
        )
    )
    np.testing.assert_allclose(result.delta_log_l, expected, rtol=1.0e-14)
    np.testing.assert_allclose(result.amplitude_scale, scale, rtol=1.0e-14)


def test_affine_orbit_is_removed_exactly_and_curvature_is_retained():
    reference_time = 1_300_000_000.0
    elapsed = np.asarray([-4_000.0, -300.0, 0.0, 700.0, 4_000.0])
    time = reference_time + elapsed
    origin = np.asarray([120.0, -330.0, 42.0])
    velocity = np.asarray([8.0e-5, -2.0e-5, 1.0e-5])
    acceleration = np.asarray([2.0e-11, -5.0e-12, 8.0e-12])
    position = (
        origin
        + elapsed[:, None] * velocity
        + 0.5 * elapsed[:, None] ** 2 * acceleration
    )
    velocity_samples = velocity + elapsed[:, None] * acceleration

    residual = remove_affine_orbital_motion(
        time,
        position,
        velocity_samples,
        reference_index=2,
    )
    expected = 0.5 * elapsed[:, None] ** 2 * acceleration

    np.testing.assert_allclose(residual, expected, rtol=0.0, atol=3.0e-14)

    affine_position = origin + elapsed[:, None] * velocity
    affine_residual = remove_affine_orbital_motion(
        time,
        affine_position,
        velocity,
        reference_index=2,
    )
    np.testing.assert_allclose(affine_residual, 0.0, rtol=0.0, atol=3.0e-14)

    direction = np.asarray([0.0, 0.6, 0.8])
    np.testing.assert_allclose(
        project_orbital_delay(residual, direction),
        expected @ direction,
        rtol=0.0,
        atol=3.0e-14,
    )


def test_fisher_profile_recovers_tangent_bias_and_projected_residual():
    weights = np.asarray([1.0, 2.0, 0.5, 1.5])
    reference = np.asarray([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex128)
    tangents = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0j, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    injected_bias = np.asarray([0.3, -0.4])
    perpendicular = np.asarray([0.0, 0.0, 0.2 + 0.1j, -0.1j])
    difference = injected_bias @ tangents + perpendicular
    target_snr = np.sqrt(noise_weighted_inner_product(reference, reference, weights))

    result = profile_waveform_residual(
        reference,
        difference,
        tangents,
        weights,
        parameter_names=("mass", "time"),
        target_snr=target_snr,
    )

    recovered_bias = np.asarray(
        [result.parameter_bias["mass"], result.parameter_bias["time"]]
    )
    expected_fisher = np.diag([1.0, 2.0])
    expected_covariance = np.diag([1.0, 0.5])
    expected_profiled = 0.5 * noise_weighted_inner_product(
        perpendicular,
        perpendicular,
        weights,
    )

    np.testing.assert_allclose(recovered_bias, injected_bias, atol=1.0e-15)
    np.testing.assert_allclose(result.fisher_matrix, expected_fisher, atol=1.0e-15)
    np.testing.assert_allclose(
        result.covariance_matrix,
        expected_covariance,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(result.projected_residual, perpendicular, atol=1.0e-15)
    np.testing.assert_allclose(result.profiled_delta_log_l, expected_profiled)
    np.testing.assert_allclose(result.parameter_sigma["mass"], 1.0)
    np.testing.assert_allclose(result.parameter_sigma["time"], np.sqrt(0.5))
    np.testing.assert_allclose(result.parameter_bias_sigma["mass"], 0.3)
    np.testing.assert_allclose(
        result.parameter_bias_sigma["time"],
        0.4 / np.sqrt(0.5),
    )
    assert result.rank == 2
    np.testing.assert_allclose(result.condition_number, 2.0)


def test_fisher_bias_projection_ignores_unidentifiable_motion():
    fisher = np.diag([4.0, 1.0, 0.0])
    shift = np.asarray([0.25, -0.5, 1.0e9])

    projected = fisher_projected_bias_sigma(fisher, shift)

    np.testing.assert_allclose(projected, np.asarray([0.5, 0.5]))
