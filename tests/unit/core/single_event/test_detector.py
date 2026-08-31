from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.constants import C_SI, EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
from jimgw.core.single_event.data import PowerSpectrum
from jimgw.core.single_event.detector import (
    GroundBased2G,
    finite_arm_transfer,
    get_CE,
    get_ET,
    get_H1,
    get_L1,
    get_V1,
)
from jimgw.core.single_event.time_dependent_response import emission_gmst
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from tests.utils import assert_all_in_range

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

GPS_TIME = 1126259462.0
DURATION = 4.0
F_MIN, F_MAX = 20.0, 1024.0
SAMPLING_FREQUENCY = F_MAX * 2

# Likelihood-space (fully expanded) parameters used as the reference injection.
REFERENCE_PARAMS = {
    "M_c": 28.0,
    "eta": 0.24,
    "s1_x": 0.3,
    "s1_y": 0.2,
    "s1_z": 0.1,
    "s2_x": -0.1,
    "s2_y": 0.2,
    "s2_z": -0.3,
    "d_L": 440.0,
    "phase_c": 0.0,
    "iota": 0.0,
    "ra": 1.5,
    "dec": 0.5,
    "psi": 0.3,
    "t_c": 0.0,
}


def make_detector():
    det = get_H1()
    psd = PowerSpectrum.from_file(str(FIXTURES_DIR / "GW150914_psd_H1.npz"))
    det.set_psd(psd)
    return det


def inject_reference(det, trigger_time=GPS_TIME, **overrides):
    """Inject the reference signal (zero noise) into *det*."""
    params = {**REFERENCE_PARAMS, **overrides}
    det.inject_signal(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        trigger_time=trigger_time,
        waveform_model=RippleIMRPhenomD(f_ref=20.0),
        parameters=params,
        f_min=F_MIN,
        f_max=F_MAX,
        zero_noise=True,
    )


# ---------------------------------------------------------------------------
# inject_signal tests
# ---------------------------------------------------------------------------


class TestInjectSignal:
    """Tests for inject_signal: core behavior and the transform pipeline."""

    # ------------------------------------------------------------------
    # Core behavior
    # ------------------------------------------------------------------

    def test_zero_noise_creates_data(self):
        """Data object is populated after a zero-noise injection."""
        det = make_detector()
        inject_reference(det)

        assert det.data is not None
        assert len(det.data.td) == int(DURATION * SAMPLING_FREQUENCY)
        assert det.data.start_time == GPS_TIME - DURATION + 2.0

    def test_zero_noise_signal_nonzero_in_band(self):
        """Injected signal is non-zero inside the frequency band."""
        det = make_detector()
        inject_reference(det)

        assert jnp.any(jnp.abs(det.sliced_fd_data) > 0)

    def test_zero_noise_frequency_bounds_respected(self):
        """Sliced frequencies lie within the requested band."""
        det = make_detector()
        inject_reference(det)

        assert_all_in_range(det.sliced_frequencies, F_MIN, F_MAX)

    def test_noisy_injection_differs_from_zero_noise(self):
        """Adding noise produces data that differs from the zero-noise case."""
        det_clean = make_detector()
        inject_reference(det_clean)

        det_noisy = make_detector()
        params = dict(REFERENCE_PARAMS)
        det_noisy.inject_signal(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            trigger_time=GPS_TIME,
            waveform_model=RippleIMRPhenomD(f_ref=20.0),
            parameters=params,
            f_min=F_MIN,
            f_max=F_MAX,
            zero_noise=False,
            rng_key=jax.random.key(42),
        )

        assert not jnp.allclose(
            det_clean.sliced_fd_data,
            det_noisy.sliced_fd_data,
            rtol=1e-05,
            atol=1e-23,
        )


# ---------------------------------------------------------------------------
# ET geometry tests
# ---------------------------------------------------------------------------


class TestET:
    """Tests for get_ET(): geometric consistency of the triangular ET configuration."""

    ET_ARM_LENGTH_M = 1e4  # 10 km

    def setup_method(self):
        self.ifos = get_ET()

    def test_returns_three_detectors(self):
        """get_ET returns exactly three GroundBased2G instances."""
        assert len(self.ifos) == 3

    def test_detector_names(self):
        """Sub-detectors are named ET1, ET2, ET3 in order."""
        assert [ifo.name for ifo in self.ifos] == ["ET1", "ET2", "ET3"]

    def test_arm_opening_angle_is_60_degrees(self):
        """Each sub-detector has 60° (π/3) between its x and y arms."""
        for ifo in self.ifos:
            delta = ifo.yarm_azimuth - ifo.xarm_azimuth
            assert abs(delta - np.pi / 3) < 1e-10, (
                f"{ifo.name}: arm opening angle is {np.degrees(delta):.4f}°, expected 60°"
            )

    def test_arms_rotated_240_degrees_between_detectors(self):
        """Consecutive sub-detectors have arm azimuths rotated by 240° (4π/3 rad)."""
        rotation = (4 / 3) * np.pi
        for i in range(2):
            dx = self.ifos[i + 1].xarm_azimuth - self.ifos[i].xarm_azimuth
            dy = self.ifos[i + 1].yarm_azimuth - self.ifos[i].yarm_azimuth
            assert abs(dx - rotation) < 1e-10, (
                f"ET{i + 1}→ET{i + 2} xarm rotation: {dx:.6f} rad, expected {rotation:.6f} rad"
            )
            assert abs(dy - rotation) < 1e-10, (
                f"ET{i + 1}→ET{i + 2} yarm rotation: {dy:.6f} rad, expected {rotation:.6f} rad"
            )

    def test_vertex_separations_match_arm_length(self):
        """
        Haversine distance between every pair of ET vertex positions should
        equal the arm length (10 km) to within 50 m.

        This checks both the propagation formula and that the triangle closes,
        following the approach used in bilby's TriangularInterferometerTest.
        """
        # Use the same WGS-84 radius get_ET uses: computed at ET1's latitude
        # (the initial latitude, before any vertex propagation).
        _a = EARTH_SEMI_MAJOR_AXIS / 1e3
        _b = EARTH_SEMI_MINOR_AXIS / 1e3
        lat0 = float(self.ifos[0].latitude)
        R = (
            _a * _b / np.sqrt(_a**2 * np.sin(lat0) ** 2 + _b**2 * np.cos(lat0) ** 2)
        ) * 1e3
        for ifo_a, ifo_b in combinations(self.ifos, 2):
            lat1 = float(ifo_a.latitude)
            lon1 = float(ifo_a.longitude)
            lat2 = float(ifo_b.latitude)
            lon2 = float(ifo_b.longitude)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            assert abs(dist - self.ET_ARM_LENGTH_M) < 50.0, (
                f"{ifo_a.name}↔{ifo_b.name}: {dist:.0f} m "
                f"(expected ~{self.ET_ARM_LENGTH_M:.0f} m ± 50 m)"
            )


# ---------------------------------------------------------------------------
# Finite-arm response tests
# ---------------------------------------------------------------------------


def test_builtin_detectors_have_explicit_arm_length_metadata():
    expected_lengths = {
        "H1": 4_000.0,
        "L1": 4_000.0,
        "V1": 3_000.0,
        "CE": 40_000.0,
    }
    detectors = [get_H1(), get_L1(), get_V1(), get_CE()]

    assert {det.name: det.arm_length_m for det in detectors} == expected_lengths
    assert all(not det.finite_arm_response for det in detectors)
    assert all(det.arm_length_m == 10_000.0 for det in get_ET())


@pytest.mark.parametrize("arm_length_m", [0.0, -1.0, np.nan, np.inf])
def test_invalid_arm_length_metadata_is_rejected(arm_length_m):
    with pytest.raises(ValueError, match="arm_length_m must be finite and positive"):
        GroundBased2G("invalid", arm_length_m=arm_length_m)


def test_enabling_finite_arm_response_without_metadata_fails_closed():
    with pytest.raises(ValueError, match="requires arm_length_m metadata"):
        GroundBased2G("missing", finite_arm_response=True)

    detector = GroundBased2G("missing")
    with pytest.raises(ValueError, match="requires arm_length_m metadata"):
        detector.frequency_dependent_antenna_pattern(1.0, 0.2, 0.3, 2.0, 100.0)


def test_finite_arm_transfer_matches_reference_formula_and_zero_frequency_limit():
    frequency = np.array([0.0, 37.0, 513.0, 3_000.0])
    direction_cosine = np.array([-0.9, -0.25, 0.4, 0.95])
    arm_length_m = 40_000.0
    x = frequency * arm_length_m / C_SI
    reference = 0.5 * (
        np.exp(-1j * np.pi * x * (1.0 + direction_cosine))
        * np.sinc(x * (1.0 - direction_cosine))
        + np.exp(1j * np.pi * x * (1.0 - direction_cosine))
        * np.sinc(x * (1.0 + direction_cosine))
    )

    result = finite_arm_transfer(frequency, direction_cosine, arm_length_m)

    np.testing.assert_allclose(result, reference, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(
        finite_arm_transfer(jnp.zeros(3), direction_cosine[:3], arm_length_m),
        jnp.ones(3, dtype=jnp.complex128),
    )


def test_frequency_dependent_antenna_pattern_has_long_wavelength_limit():
    detector = get_CE()
    sky = (1.2, -0.4, 0.7, 2.1)
    static = detector.antenna_pattern(*sky)
    finite = detector.frequency_dependent_antenna_pattern(*sky, frequency=0.0)

    assert finite.keys() == static.keys()
    for mode in static:
        np.testing.assert_allclose(finite[mode], static[mode], rtol=1e-13, atol=1e-14)


def test_frequency_dependent_antenna_pattern_is_vector_safe_and_jittable():
    detector = get_CE()
    frequency = jnp.array([20.0, 300.0, 1_000.0, 3_000.0])
    gmst = jnp.array([0.3, 0.7, 1.4, 2.2])

    vector_result = detector.frequency_dependent_antenna_pattern(
        1.2, -0.4, 0.7, gmst, frequency
    )
    scalar_results = [
        detector.frequency_dependent_antenna_pattern(
            1.2, -0.4, 0.7, gmst_i, frequency_i
        )
        for gmst_i, frequency_i in zip(gmst, frequency)
    ]
    jitted = jax.jit(
        lambda frequencies, gmsts: detector.frequency_dependent_antenna_pattern(
            1.2, -0.4, 0.7, gmsts, frequencies
        )
    )(frequency, gmst)

    for mode in vector_result:
        scalar_mode = jnp.stack([result[mode] for result in scalar_results])
        np.testing.assert_allclose(
            vector_result[mode], scalar_mode, rtol=1e-13, atol=1e-14
        )
        np.testing.assert_allclose(jitted[mode], vector_result[mode], rtol=1e-13)


def test_fd_response_can_opt_in_per_call_or_from_detector_configuration():
    detector = get_CE()
    frequency = jnp.array([20.0, 1_000.0, 3_000.0])
    h_sky = {
        "p": jnp.ones(3, dtype=jnp.complex128),
        "c": 1j * jnp.ones(3, dtype=jnp.complex128),
    }
    params = {
        "ra": 1.2,
        "dec": -0.4,
        "psi": 0.7,
        "gmst": 2.1,
        "trigger_time": 0.0,
        "t_c": 0.0,
    }

    default = detector.fd_response(frequency, h_sky, params)
    explicit_static = detector.fd_response(
        frequency, h_sky, params, finite_arm=False
    )
    explicit_finite = detector.fd_response(
        frequency, h_sky, params, finite_arm=True
    )
    detector.finite_arm_response = True
    configured_finite = detector.fd_response(frequency, h_sky, params)

    np.testing.assert_array_equal(default, explicit_static)
    np.testing.assert_allclose(configured_finite, explicit_finite, rtol=1e-13)
    assert not np.allclose(default, explicit_finite, rtol=1e-6, atol=1e-8)


def test_time_dependent_response_requires_and_consumes_emission_clock():
    detector = get_CE()
    detector.time_dependent_response = True
    frequency = jnp.array([5.0, 20.0, 100.0])
    h_sky = {
        "p": jnp.ones(3, dtype=jnp.complex128),
        "c": 1j * jnp.ones(3, dtype=jnp.complex128),
    }
    params = {
        "ra": 1.2,
        "dec": -0.4,
        "psi": 0.7,
        "gmst": 2.1,
        "trigger_time": 100.0,
        "t_c": 0.02,
    }

    with pytest.raises(ValueError, match="requires a waveform with a __tau__"):
        detector.fd_response(frequency, h_sky, params)

    tau = jnp.array([7_000.0, 300.0, 2.0])
    result = detector.fd_response(
        frequency,
        {**h_sky, "__tau__": tau},
        params,
    )
    gmst = emission_gmst(params["gmst"], params["t_c"], tau)
    m, n, omega = detector._wave_frame(
        params["ra"], params["dec"], params["psi"], gmst
    )
    expected_patterns = {
        polarization.name: jnp.einsum(
            "ij,ij...->...",
            detector.tensor,
            polarization.tensor_from_basis(m, n),
        )
        for polarization in detector.polarization_mode
    }
    delay = -jnp.einsum("i...,i->...", omega, detector.vertex) / C_SI
    projected = expected_patterns["p"] * h_sky["p"]
    projected += expected_patterns["c"] * h_sky["c"]
    expected = projected * jnp.exp(
        -2j
        * jnp.pi
        * frequency
        * (params["trigger_time"] - detector.start_time + params["t_c"] + delay)
    )

    np.testing.assert_allclose(result, expected, rtol=3e-12, atol=3e-12)


def test_timed_waveform_cannot_be_projected_with_static_response():
    detector = get_H1()
    h_sky = {
        "p": jnp.ones(1, dtype=jnp.complex128),
        "c": jnp.zeros(1, dtype=jnp.complex128),
        "__tau__": jnp.ones(1),
    }
    params = {
        "ra": 1.2,
        "dec": -0.4,
        "psi": 0.7,
        "gmst": 2.1,
        "trigger_time": 0.0,
        "t_c": 0.0,
    }

    with pytest.raises(ValueError, match="requires time_dependent_response=True"):
        detector.fd_response(jnp.array([20.0]), h_sky, params)


def test_fd_response_matches_complex_exp_reference():
    det = make_detector()
    inject_reference(det)
    params = {
        "ra": 1.375,
        "dec": -1.2108,
        "psi": 0.3,
        "gmst": 2.1,
        "trigger_time": GPS_TIME,
        "t_c": 0.013,
    }
    antenna = det.antenna_pattern(
        params["ra"], params["dec"], params["psi"], params["gmst"]
    )
    rng = np.random.default_rng(5)
    n = 513
    frequency = jnp.linspace(F_MIN, F_MAX, n)
    h_sky = {
        key: jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))
        for key in antenna
    }

    result = det.fd_response(frequency, h_sky, params)

    # Pre-change reference: complex-exponential phasor applied to the same
    # antenna-projected strain.
    time_shift = det.delay_from_geocenter(params["ra"], params["dec"], params["gmst"])
    time_shift += params["trigger_time"] - det.start_time + params["t_c"]
    projected = sum(h_sky[key] * antenna[key] for key in h_sky)
    reference = projected * jnp.exp(-2j * jnp.pi * frequency * time_shift)

    assert result.dtype == reference.dtype
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(reference), rtol=1e-13, atol=1e-14
    )
