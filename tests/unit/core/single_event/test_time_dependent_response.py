import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.constants import DAYSID_SI
from jimgw.core.single_event.detector import get_H1
from jimgw.core.single_event.time_dependent_response import (
    SIDEREAL_ANGULAR_FREQUENCY,
    SIDEREAL_COLLOCATION_PHASES,
    SiderealHarmonics,
    collocate_antenna_harmonics,
    collocate_delay_harmonics,
    combine_sidereal_sidebands,
    delay_from_geocenter,
    detector_time_phasor,
    emission_gmst,
    emission_time,
    evaluate_sidereal_harmonics,
    linear_gmst,
    long_wavelength_antenna_patterns,
    project_dynamic_long_wavelength,
    project_static_long_wavelength,
    sidereal_harmonics_from_collocation,
    time_to_coalescence_2pn,
)


def test_2pn_clock_matches_fixed_independent_values():
    frequency = jnp.asarray([5.0, 20.0, 100.0])
    expected = np.asarray([6836.797457517856, 171.0659871432222, 2.3523147867479364])

    result = time_to_coalescence_2pn(frequency, 1.4, 1.3, 0.12, -0.08, mode=2)

    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-14, atol=0.0)
    assert np.all(np.asarray(result) > 0.0)
    assert np.all(np.diff(np.asarray(result)) < 0.0)


def test_2pn_clock_uses_absolute_mode_number_and_frequency_rescaling():
    frequency = jnp.asarray([8.0, 21.0, 64.0])
    tau_three = time_to_coalescence_2pn(frequency, 9.0, 7.0, mode=3)
    tau_minus_three = time_to_coalescence_2pn(frequency, 9.0, 7.0, mode=-3)
    tau_two_rescaled = time_to_coalescence_2pn(2.0 * frequency / 3.0, 9.0, 7.0, mode=2)

    np.testing.assert_array_equal(np.asarray(tau_three), np.asarray(tau_minus_three))
    np.testing.assert_allclose(
        np.asarray(tau_three), np.asarray(tau_two_rescaled), rtol=1e-14
    )
    with pytest.raises(ValueError, match="nonzero"):
        time_to_coalescence_2pn(frequency, 9.0, 7.0, mode=0)


def test_2pn_clock_is_jittable_and_differentiable():
    clock = jax.jit(
        lambda frequency, mass: time_to_coalescence_2pn(
            frequency, mass, 1.2, 0.05, -0.03
        )
    )
    result = clock(jnp.asarray([5.0, 30.0]), jnp.asarray(1.4))
    derivative = jax.grad(lambda mass: time_to_coalescence_2pn(10.0, mass, 1.2))(
        jnp.asarray(1.4)
    )

    assert np.all(np.isfinite(np.asarray(result)))
    assert np.isfinite(float(derivative))


def test_positive_tau_moves_time_and_unwrapped_gmst_earlier():
    trigger_time = 1_500_000_000.0
    t_c = 0.02
    tau = jnp.asarray([0.0, DAYSID_SI / 4.0])

    times = emission_time(trigger_time, t_c, tau)
    angles = emission_gmst(0.0, t_c, tau, wrap=False)

    np.testing.assert_allclose(
        np.asarray(times - trigger_time), np.asarray(t_c - tau), atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(angles), SIDEREAL_ANGULAR_FREQUENCY * np.asarray(t_c - tau)
    )
    assert angles[1] < angles[0]


def test_linear_gmst_wraps_without_using_absolute_times():
    reference = 2.345
    elapsed = jnp.asarray([-2.0 * DAYSID_SI, 0.0, 3.0 * DAYSID_SI])
    wrapped = linear_gmst(reference, elapsed)
    unwrapped = linear_gmst(reference, elapsed, wrap=False)

    np.testing.assert_allclose(np.asarray(wrapped), reference, atol=3e-15)
    np.testing.assert_allclose(
        np.asarray(jnp.mod(unwrapped, 2.0 * jnp.pi)), np.asarray(wrapped), atol=3e-15
    )


def test_vector_response_matches_scalar_detector_methods():
    detector = get_H1()
    ra = 1.37
    dec = -0.48
    psi = 0.29
    gmst = jnp.linspace(-0.2, 2.0 * jnp.pi + 0.3, 17)

    actual_patterns = jax.jit(long_wavelength_antenna_patterns)(
        detector.tensor, ra, dec, psi, gmst
    )
    actual_delay = jax.jit(delay_from_geocenter)(detector.vertex, ra, dec, gmst)
    expected_patterns = jax.vmap(
        lambda angle: detector.antenna_pattern(ra, dec, psi, angle)
    )(gmst)
    expected_delay = jax.vmap(
        lambda angle: detector.delay_from_geocenter(ra, dec, angle)
    )(gmst)

    np.testing.assert_allclose(
        np.asarray(actual_patterns.plus), np.asarray(expected_patterns["p"]), atol=2e-15
    )
    np.testing.assert_allclose(
        np.asarray(actual_patterns.cross),
        np.asarray(expected_patterns["c"]),
        atol=2e-15,
    )
    np.testing.assert_allclose(
        np.asarray(actual_delay), np.asarray(expected_delay), atol=2e-17
    )


@pytest.mark.parametrize("dec", [-jnp.pi / 2.0, jnp.pi / 2.0])
def test_vector_response_is_finite_at_poles_and_under_wrap(dec):
    detector = get_H1()
    gmst = jnp.asarray([-2.0 * jnp.pi, 0.0, 2.0 * jnp.pi])

    pattern = long_wavelength_antenna_patterns(detector.tensor, 0.3, dec, 0.8, gmst)
    periodic = long_wavelength_antenna_patterns(
        detector.tensor, 0.3 + 2.0 * jnp.pi, dec, 0.8 + jnp.pi, gmst
    )
    gradient = jax.grad(
        lambda right_ascension: (
            long_wavelength_antenna_patterns(
                detector.tensor, right_ascension, dec, 0.8, 0.4
            ).plus
        )
    )(jnp.asarray(0.3))

    assert np.all(np.isfinite(np.asarray(pattern.plus)))
    assert np.all(np.isfinite(np.asarray(pattern.cross)))
    assert np.isfinite(float(gradient))
    np.testing.assert_allclose(np.asarray(pattern.plus), np.asarray(periodic.plus))
    np.testing.assert_allclose(np.asarray(pattern.cross), np.asarray(periodic.cross))


def test_five_collocations_reconstruct_antenna_over_sidereal_day():
    detector = get_H1()
    ra = 5.91
    dec = 1.12
    psi = 0.77
    gmst_reference = 6.1
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, 1025)

    harmonics = collocate_antenna_harmonics(
        detector.tensor, ra, dec, psi, gmst_reference
    )
    reconstructed_plus = evaluate_sidereal_harmonics(harmonics.plus, phase)
    reconstructed_cross = evaluate_sidereal_harmonics(harmonics.cross, phase)
    direct = long_wavelength_antenna_patterns(
        detector.tensor, ra, dec, psi, gmst_reference + phase
    )

    np.testing.assert_allclose(
        np.asarray(reconstructed_plus), np.asarray(direct.plus), atol=2e-12
    )
    np.testing.assert_allclose(
        np.asarray(reconstructed_cross), np.asarray(direct.cross), atol=2e-12
    )


def test_batched_collocations_reconstruct_batched_phase_grids():
    detector = get_H1()
    ra = jnp.asarray([0.2, 2.3, 5.8])
    dec = jnp.asarray([-0.8, 0.0, 1.1])
    psi = jnp.asarray([0.1, 0.7, 1.3])
    gmst_reference = jnp.asarray([0.3, 2.0, 6.0])
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, 129)

    harmonics = collocate_antenna_harmonics(
        detector.tensor, ra, dec, psi, gmst_reference
    )
    reconstructed = evaluate_sidereal_harmonics(harmonics.plus, phase[None, :])
    direct = long_wavelength_antenna_patterns(
        detector.tensor,
        ra[:, None],
        dec[:, None],
        psi[:, None],
        gmst_reference[:, None] + phase,
    ).plus

    assert reconstructed.shape == (3, 129)
    np.testing.assert_allclose(
        np.asarray(reconstructed), np.asarray(direct), atol=2e-12
    )


def test_collocation_formula_recovers_known_coefficients():
    expected = SiderealHarmonics(0.3, -0.2, 0.1, 0.07, -0.04)
    values = evaluate_sidereal_harmonics(expected, SIDEREAL_COLLOCATION_PHASES)

    actual = sidereal_harmonics_from_collocation(values)

    for actual_coefficient, expected_coefficient in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_coefficient, expected_coefficient, atol=2e-16)


def test_raw_delay_is_first_harmonic_but_its_phasor_is_not_five_harmonic():
    detector = get_H1()
    ra = 1.81
    dec = -0.24
    gmst_reference = 0.61
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, 1025)

    harmonics = collocate_delay_harmonics(detector.vertex, ra, dec, gmst_reference)
    reconstructed = evaluate_sidereal_harmonics(harmonics, phase)
    direct = delay_from_geocenter(detector.vertex, ra, dec, gmst_reference + phase)
    np.testing.assert_allclose(
        np.asarray(reconstructed), np.asarray(direct), atol=2e-15
    )
    assert abs(float(harmonics.cos_two)) < 2e-15
    assert abs(float(harmonics.sin_two)) < 2e-15

    phasor_at_collocations = detector_time_phasor(
        jnp.full((5,), 300.0),
        delay_from_geocenter(
            detector.vertex,
            ra,
            dec,
            gmst_reference + SIDEREAL_COLLOCATION_PHASES,
        ),
    )
    phasor_harmonics = sidereal_harmonics_from_collocation(phasor_at_collocations)
    phasor_reconstruction = evaluate_sidereal_harmonics(phasor_harmonics, phase)
    phasor_direct = detector_time_phasor(jnp.full(phase.shape, 300.0), direct)
    assert np.max(np.abs(np.asarray(phasor_reconstruction - phasor_direct))) > 0.1


def test_sideband_combination_uses_documented_shift_and_fourier_sign():
    harmonics = SiderealHarmonics(0.7, -0.2, 0.13, 0.05, -0.09)
    rng = np.random.default_rng(5104)
    shifted = rng.normal(size=(11, 5)) + 1j * rng.normal(size=(11, 5))

    actual = combine_sidereal_sidebands(harmonics, jnp.asarray(shifted))
    h_zero, h_plus_one, h_minus_one, h_plus_two, h_minus_two = shifted.T
    expected = (
        0.7 * h_zero
        - 0.1 * (h_plus_one + h_minus_one)
        + 0.065j * (h_plus_one - h_minus_one)
        + 0.025 * (h_plus_two + h_minus_two)
        - 0.045j * (h_plus_two - h_minus_two)
    )

    np.testing.assert_allclose(np.asarray(actual), expected, atol=3e-16)


def test_static_projection_matches_current_detector_path():
    detector = get_H1()
    start_time = 1_126_259_458.0
    detector.data.start_time = start_time
    trigger_time = 1_126_259_462.0
    params = {
        "ra": 1.375,
        "dec": -1.2108,
        "psi": 0.3,
        "gmst": 2.1,
        "trigger_time": trigger_time,
        "t_c": 0.013,
    }
    rng = np.random.default_rng(23)
    frequency = jnp.linspace(5.0, 1024.0, 513)
    h_sky = {
        "p": jnp.asarray(rng.normal(size=513) + 1j * rng.normal(size=513)),
        "c": jnp.asarray(rng.normal(size=513) + 1j * rng.normal(size=513)),
    }

    actual = project_static_long_wavelength(
        frequency,
        h_sky["p"],
        h_sky["c"],
        detector.tensor,
        detector.vertex,
        params["ra"],
        params["dec"],
        params["psi"],
        params["gmst"],
        trigger_time,
        start_time,
        params["t_c"],
    )
    expected = detector.fd_response(frequency, h_sky, params)

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=2e-13, atol=2e-14
    )


def test_dynamic_projection_changes_antenna_and_delay_at_emission_time():
    detector = get_H1()
    frequency = jnp.asarray([5.0, 8.0, 13.0, 21.0, 55.0, 144.0])
    tau = jnp.asarray([7200.0, 3900.0, 1800.0, 700.0, 70.0, 3.0])
    h_plus = jnp.asarray([1.0 + 0.2j] * len(frequency))
    h_cross = jnp.asarray([-0.1 + 0.4j] * len(frequency))
    ra, dec, psi = 4.7, -0.32, 1.1
    gmst_at_trigger = 0.08
    trigger_time, start_time, t_c = 1_500_000_000.0, 1_499_992_000.0, 0.017

    actual = jax.jit(project_dynamic_long_wavelength)(
        frequency,
        h_plus,
        h_cross,
        tau,
        detector.tensor,
        detector.vertex,
        ra,
        dec,
        psi,
        gmst_at_trigger,
        trigger_time,
        start_time,
        t_c,
    )
    gmst = emission_gmst(gmst_at_trigger, t_c, tau)
    patterns = jax.vmap(lambda angle: detector.antenna_pattern(ra, dec, psi, angle))(
        gmst
    )
    delay = jax.vmap(lambda angle: detector.delay_from_geocenter(ra, dec, angle))(gmst)
    geocentric_time = trigger_time - start_time + t_c
    expected = (patterns["p"] * h_plus + patterns["c"] * h_cross) * jnp.exp(
        -2j * jnp.pi * frequency * (geocentric_time + delay)
    )

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=3e-12, atol=3e-12
    )


def test_static_time_shift_has_negative_fourier_phase_sign():
    detector = get_H1()
    frequency = jnp.asarray([7.0, 31.0, 97.0])
    h_plus = jnp.ones(3, dtype=jnp.complex128)
    h_cross = jnp.zeros(3, dtype=jnp.complex128)
    common = (
        frequency,
        h_plus,
        h_cross,
        detector.tensor,
        detector.vertex,
        1.2,
        -0.4,
        0.1,
        2.3,
        100.0,
        96.0,
    )
    delta_t = 0.004

    reference = project_static_long_wavelength(*common, 0.0)
    shifted = project_static_long_wavelength(*common, delta_t)

    np.testing.assert_allclose(
        np.asarray(shifted / reference),
        np.asarray(jnp.exp(-2j * jnp.pi * frequency * delta_t)),
        rtol=3e-13,
        atol=3e-13,
    )
