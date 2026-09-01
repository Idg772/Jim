import numpy as np

from benchmarks.xg.oracles.response_oracle import (
    C_SI,
    DetectorGeometry,
    finite_arm_transfer,
    frequency_domain_response,
    segmented_round_trip_response,
    segmented_round_trip_response_function,
)


def _geometry(*, vertex=True) -> DetectorGeometry:
    return DetectorGeometry(
        vertex_m=np.asarray([6.0e6, -7.0e5, 1.4e6]) if vertex else np.zeros(3),
        x_arm=np.asarray([1.0, 0.0, 0.0]),
        y_arm=np.asarray([0.0, 1.0, 0.0]),
        arm_length_m=40_000.0,
    )


def test_independent_finite_arm_transfer_has_published_low_frequency_slope():
    frequency = np.asarray([0.0, 1.0e-4])
    mu = 0.37
    transfer = finite_arm_transfer(frequency, mu, 40_000.0)
    numerical_slope = (transfer[1] - transfer[0]) / frequency[1]
    expected_slope = -1j * np.pi * (40_000.0 / C_SI) * (2.0 - mu)

    np.testing.assert_allclose(transfer[0], 1.0, atol=1e-15)
    np.testing.assert_allclose(numerical_slope, expected_slope, rtol=2e-7)


def test_segmented_time_domain_arm_response_converges_to_frequency_formula():
    sample_rate = 262_144.0
    duration = 0.2
    time = np.arange(round(sample_rate * duration), dtype=np.float64) / sample_rate
    frequency = 1_500.0
    carrier = np.exp(2j * np.pi * frequency * time)
    geometry = _geometry(vertex=False)
    kwargs = {
        "geometry": geometry,
        "gmst_at_zero": 0.0,
        "ra": 0.8,
        "dec": -0.3,
        "psi": 0.4,
    }

    time_domain = segmented_round_trip_response(
        time,
        carrier,
        np.zeros_like(carrier),
        arm_subsegments=64,
        **kwargs,
    )
    response_plus, _ = frequency_domain_response(
        frequency,
        np.zeros(1),
        **kwargs,
    )
    # Exclude interpolation boundaries and the tiny sidereal drift over the
    # comparison interval; the midpoint response is the adiabatic reference.
    middle = slice(round(0.05 * sample_rate), round(0.15 * sample_rate))
    ratio = np.mean(time_domain[middle] / carrier[middle])
    midpoint_response, _ = frequency_domain_response(
        frequency,
        np.asarray([0.1]),
        **kwargs,
    )

    assert np.isfinite(response_plus[0])
    np.testing.assert_allclose(ratio, midpoint_response[0], rtol=2e-4, atol=2e-5)


def test_frequency_response_is_periodic_after_one_sidereal_day():
    kwargs = {
        "geometry": _geometry(),
        "gmst_at_zero": 1.1,
        "ra": 2.4,
        "dec": 0.2,
        "psi": 0.7,
    }
    first = frequency_domain_response(700.0, 0.0, **kwargs)
    second = frequency_domain_response(700.0, 86_164.09053083288, **kwargs)

    np.testing.assert_allclose(first[0], second[0], atol=2e-13)
    np.testing.assert_allclose(first[1], second[1], atol=2e-13)


def test_callable_round_trip_response_avoids_source_interpolation_error():
    time = np.linspace(0.0, 0.1, 2049)
    frequency = 1_200.0
    carrier = lambda query: np.exp(2j * np.pi * frequency * query)
    kwargs = {
        "geometry": _geometry(vertex=False),
        "gmst_at_zero": 0.4,
        "ra": 1.1,
        "dec": 0.2,
        "psi": 0.7,
    }

    time_domain = segmented_round_trip_response_function(
        time,
        carrier,
        lambda query: np.zeros_like(query, dtype=np.complex128),
        arm_subsegments=128,
        **kwargs,
    )
    predicted, _ = frequency_domain_response(
        frequency,
        np.asarray([0.05]),
        **kwargs,
    )
    ratio = np.mean(time_domain[512:-512] / carrier(time[512:-512]))

    np.testing.assert_allclose(ratio, predicted[0], rtol=3e-5, atol=3e-6)
