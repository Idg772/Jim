"""Independent native sums share the noisy summary stream, including its tail."""

import copy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import i0e

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_CE_A, get_ET_Sardinia
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.likelihood import (
    _XG_QUALIFICATION_PLAN_AUTHORITY,
    HeterodynedTransientLikelihoodFD,
    _QualificationXGPlan,
)
from jimgw.core.single_event.native_probe import NativeProbeBank
from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2

jax.config.update("jax_enable_x64", True)
GPS = 1_300_000_000.0
PARAMETERS = {
    "M_c": 1.1802650981093186,
    "eta": 0.2499511169561633,
    "s1_z": -0.014,
    "s2_z": -0.023,
    "lambda_1": 483.0,
    "lambda_2": 771.0,
    "d_L": 20.0,
    "iota": 2.016,
    "ra": 6.05,
    "dec": 0.173,
    "psi": 1.68,
    "phase_c": 0.63,
    "t_c": 0.035,
}
EDGES = np.array([2.0, 3.1, 5.1, 6.9, 8.0])
ANCHORS = np.linspace(-0.2, 0.2, 5)


def _direct_native_sums(network, point, *, phase):
    """Dense native sums for these small fixtures, independent of the stream.

    Evaluate the source once on the full union, so its first two frequencies
    fix Ripple's cutoff convention naturally. Project only each detector's
    supported band, with its independent complex-exponential delay path. The
    ordinary rectangular 4 df rule includes every endpoint; no heterodyne
    moments, padding, chunk reducers or production probe helpers are used.
    """
    detectors, waveform = network
    parameters = {
        **point,
        "trigger_time": GPS,
        "gmst": greenwich_mean_sidereal_time(GPS),
    }
    if phase:
        parameters["phase_c"] = 0.0
    frequency = np.unique(
        np.concatenate([np.asarray(d.sliced_frequencies) for d in detectors])
    )
    sky = waveform(jnp.asarray(frequency), parameters)
    channels = {}
    z, q = 0j, 0.0
    for detector in detectors:
        f = np.asarray(detector.sliced_frequencies)
        indices = np.searchsorted(frequency, f)
        np.testing.assert_array_equal(frequency[indices], f)
        band_sky = jax.tree.map(lambda value, indices=indices: value[indices], sky)
        h = np.asarray(
            detector.fd_response(jnp.asarray(f), band_sky, parameters, optimize=False)
        )
        data, psd = np.asarray(detector.sliced_fd_data), np.asarray(detector.sliced_psd)
        assert np.all(np.isfinite(h))
        weight = 4 * (f[1] - f[0])
        overlap = weight * np.sum(np.conj(h) * data / psd)
        norm = weight * np.sum((h.real**2 + h.imag**2) / psd)
        channels[detector.name] = {
            "native_samples": len(f),
            "complex_overlap": {"real": overlap.real, "imag": overlap.imag},
            "waveform_norm": norm,
        }
        z += overlap
        q += norm
    match = np.log(i0e(abs(z))) + abs(z) if phase else z.real
    return {"log_likelihood": match - q / 2, "channels": channels}


@pytest.fixture(scope="module")
def network():
    rng = np.random.default_rng(431)
    detectors = [get_CE_A(), *get_ET_Sardinia()]
    for index, detector in enumerate(detectors):
        data = Data.from_host_fd(
            np.asarray((rng.normal(size=33) + 1j * rng.normal(size=33)) * 1e-21),
            delta_t=1 / 16,
            start_time=GPS - 131070 + index / 8,
        )
        detector.set_data(data)
        detector.set_psd(
            PowerSpectrum(
                1e-42 * (1 + data.frequencies / (10 + index)), data.frequencies
            )
        )
        detector.time_dependent_response = True
        detector.finite_arm_response = True
        detector.configure_orbital_motion_response(
            enabled=True,
            reference_time=GPS,
            validity_s=(-131072.125, 0.125),
            acceleration_over_c=(2e-11, -1e-12, -4e-13),
            jerk_over_c=(1.4e-20, 3.8e-18, 1.7e-18),
        )
        detector.set_frequency_bounds(5 if detector.name == "CE" else 2, 8)
    return detectors, DominantModeTimeCachedWaveform(
        RippleIMRPhenomD_NRTidalv2(f_ref=20)
    )


def construct(network, points, *, phase=True, edges=EDGES, chunk_size=4):
    detectors, waveform = network
    digest = HeterodynedTransientLikelihoodFD._bin_edges_sha256(
        edges,
        interpolation_order=2,
        phasor_moment_order=2,
        phasor_time_anchors=ANCHORS,
        reference_projection="carrier",
    )
    return HeterodynedTransientLikelihoodFD(
        detectors,
        waveform,
        trigger_time=GPS,
        f_min={d.name: 5.0 if d.name == "CE" else 2.0 for d in detectors},
        f_max=float(edges[-1]),
        n_bins=len(edges) - 1,
        frequency_bin_edges=edges,
        reference_parameters=PARAMETERS,
        interpolation_order=2,
        phasor_moment_order=2,
        phasor_time_anchors=ANCHORS,
        reference_projection="carrier",
        reference_chunk_size=chunk_size,
        summary_backend="jax",
        phase_marginalization=phase,
        native_probe_parameters=points,
        xg_plan=_QualificationXGPlan(digest, _XG_QUALIFICATION_PLAN_AUTHORITY),
    )


class _CountedArray(np.ndarray):
    def __array_finalize__(self, source):
        self.reads = getattr(source, "reads", [])

    def __getitem__(self, index):
        if isinstance(index, slice):
            self.reads.append((index.start, index.stop, index.step))
        return super().__getitem__(index)


@pytest.mark.parametrize("phase", [False, True])
def test_noisy_mixed_band_native_parity_and_one_read_per_chunk(
    network, monkeypatch, phase
):
    network = copy.deepcopy(network)
    points = [
        PARAMETERS,
        {**PARAMETERS, "M_c": 1.18, "ra": 1.3, "t_c": -0.015},
        {**PARAMETERS, "d_L": 25.0, "phase_c": 1.2},
    ]
    original = HeterodynedTransientLikelihoodFD._compute_reference_coefficients_jax
    reads = {}

    def counted(self, detector, *args, **kwargs):
        detector._sliced_fd_data = np.asarray(detector.sliced_fd_data).view(
            _CountedArray
        )
        detector._sliced_fd_data.reads = []
        result = original(self, detector, *args, **kwargs)
        reads[detector.name] = list(detector.sliced_fd_data.reads)
        return result

    monkeypatch.setattr(
        HeterodynedTransientLikelihoodFD, "_compute_reference_coefficients_jax", counted
    )
    likelihood = construct(network, points, phase=phase)
    assert likelihood.native_probe_results is not None
    assert len(likelihood.native_probe_results) == len(points)
    for detector in network[0]:
        size = len(detector.sliced_frequencies)
        assert size % 4 == 1
        assert reads[detector.name] == [
            (start, min(start + 4, size), None) for start in range(0, size, 4)
        ]
    for point, result in zip(points, likelihood.native_probe_results, strict=True):
        expected = _direct_native_sums(network, point, phase=phase)
        assert result["native_prefix"] == [2.0, 2.25]
        assert result["detector_phasor_optimized"] is False
        assert result["qualification"] is False
        np.testing.assert_allclose(
            result["log_likelihood"], expected["log_likelihood"], rtol=0, atol=2e-8
        )
        for name, channel in result["channels"].items():
            assert (
                channel["native_samples"]
                == expected["channels"][name]["native_samples"]
            )
            assert channel["frequency_max"] == 8.0
            for key in ("real", "imag"):
                np.testing.assert_allclose(
                    channel["complex_overlap"][key],
                    expected["channels"][name]["complex_overlap"][key],
                    rtol=0,
                    atol=2e-8,
                )
            np.testing.assert_allclose(
                channel["waveform_norm"],
                expected["channels"][name]["waveform_norm"],
                rtol=0,
                atol=2e-8,
            )


def test_default_has_no_fused_probe_bank(network):
    controls = copy.deepcopy(network)
    likelihood = construct(controls, None)
    assert likelihood.native_probe_results is None
    fused = construct(controls, [PARAMETERS])
    for detector in controls[0]:
        name = detector.name
        np.testing.assert_array_equal(
            likelihood.summary_data[name], fused.summary_data[name]
        )
        np.testing.assert_array_equal(
            likelihood.phasor_data_moments[name], fused.phasor_data_moments[name]
        )


@pytest.mark.parametrize(
    "points", [[], [PARAMETERS] * 10, [{**PARAMETERS, "M_c": np.nan}]]
)
def test_invalid_probe_bank_fails_before_native_stream(network, monkeypatch, points):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid bank reached native stream")

    monkeypatch.setattr(
        HeterodynedTransientLikelihoodFD,
        "_compute_reference_coefficients_jax",
        forbidden,
    )
    with pytest.raises(ValueError, match="native probe"):
        construct(copy.deepcopy(network), points)


def test_invalid_probe_waveform_cannot_be_hidden_by_valid_reference(network):
    with pytest.raises(ValueError, match="native probe"):
        construct(copy.deepcopy(network), [{**PARAMETERS, "M_c": 0.7}])


def test_native_probe_shape_guard_and_invalid_stream_samples(network):
    detectors, waveform = copy.deepcopy(network)
    detector = detectors[1]
    likelihood = SimpleNamespace(
        detectors=[detector],
        trigger_time=GPS,
        phase_marginalization=True,
        time_marginalization=False,
        distance_marginalization=False,
        fixed_parameters={},
    )
    original = detector._sliced_fd_data
    detector._sliced_fd_data = original[:, None]
    with pytest.raises(ValueError, match="one-dimensional"):
        NativeProbeBank(likelihood, waveform, [PARAMETERS])
    detector._sliced_fd_data = original
    bank = NativeProbeBank(likelihood, waveform, [PARAMETERS])
    device = jax.devices()[0]
    reducer = bank.make_reducer(detector, device)
    f, d, psd = (
        jnp.asarray(array[-4:])
        for array in (
            detector.sliced_frequencies,
            detector.sliced_fd_data,
            detector.sliced_psd,
        )
    )
    for bad_psd in (0.0, -1.0, np.nan, np.inf):
        state = reducer(
            f,
            d,
            psd.at[-1].set(bad_psd),
            len(original) - 4,
            4,
            bank.initial_state(device),
        )
        with pytest.raises(ValueError, match="invalid native probe"):
            bank.channel_result(detector, state, chunks=1, chunk_size=4)
    for bad_f, bad_d in ((f.at[-1].add(0.01), d), (f, d.at[-1].set(jnp.nan))):
        state = reducer(
            bad_f, bad_d, psd, len(original) - 4, 4, bank.initial_state(device)
        )
        with pytest.raises(ValueError, match="invalid native probe"):
            bank.channel_result(detector, state, chunks=1, chunk_size=4)


def test_padded_tail_preserves_the_detectors_own_orbital_validity(network):
    detectors, waveform = copy.deepcopy(network)
    detector = detectors[0]
    detector.configure_orbital_motion_response(
        enabled=True,
        reference_time=GPS,
        validity_s=(-20000.0, 0.125),
        acceleration_over_c=(2e-11, -1e-12, -4e-13),
        jerk_over_c=(1.4e-20, 3.8e-18, 1.7e-18),
    )
    likelihood = SimpleNamespace(
        detectors=detectors,
        trigger_time=GPS,
        phase_marginalization=False,
        time_marginalization=False,
        distance_marginalization=False,
        fixed_parameters={},
    )
    bank = NativeProbeBank(likelihood, waveform, [PARAMETERS])
    np.testing.assert_array_equal(bank.prefix, [2.0, 2.25])
    f, data, psd = (
        jnp.asarray(array[-1:])
        for array in (
            detector.sliced_frequencies,
            detector.sliced_fd_data,
            detector.sliced_psd,
        )
    )
    # The independent single-sample response is valid at 8 Hz. Its CE orbital
    # contract deliberately excludes the 2 Hz clock belonging to the ET band.
    sky = waveform(jnp.concatenate((jnp.asarray(bank.prefix), f)), bank.parameters[0])
    sky = jax.tree.map(lambda value: value[2:], sky)
    h = detector.fd_response(f, sky, bank.parameters[0], optimize=False)
    assert bool(jnp.all(jnp.isfinite(h)))
    weight = 4 * bank.bands[detector.name]["df"]
    overlap = weight * jnp.sum(jnp.conj(h) * data / psd)
    norm = weight * jnp.sum((h.real**2 + h.imag**2) / psd)
    expected = np.asarray([overlap.real, overlap.imag, norm])
    device = jax.devices()[0]
    state = bank.make_reducer(detector, device)(
        jnp.pad(f, (0, 3), constant_values=jnp.nan),
        jnp.pad(data, (0, 3), constant_values=jnp.nan),
        jnp.pad(psd, (0, 3), constant_values=jnp.nan),
        len(detector.sliced_frequencies) - 1,
        1,
        bank.initial_state(device),
    )
    actual = bank.channel_result(detector, state, chunks=1, chunk_size=4)
    np.testing.assert_allclose(actual["sums"][0], expected, rtol=0, atol=2e-8)


@pytest.fixture(scope="module")
def full_band_network():
    # Coarse native quadrature is solely an arithmetic test, not a resolved
    # 21-hour signal. Keep the full physical clock, epoch and finite arms.
    frequency = np.arange(2049, dtype=np.float64)
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD_NRTidalv2(f_ref=20))
    params = {
        **PARAMETERS,
        "trigger_time": GPS,
        "gmst": greenwich_mean_sidereal_time(GPS),
    }
    sky = waveform(frequency[2:], params)
    rng = np.random.default_rng(38261)
    detectors = [get_CE_A(), *get_ET_Sardinia()]
    for index, detector in enumerate(detectors):
        psd = (1 + index / 4) * 1e-45 * (1 + (30 / np.maximum(frequency, 1)) ** 4)
        noise = np.sqrt(psd / 4) * (
            rng.normal(size=len(frequency)) + 1j * rng.normal(size=len(frequency))
        )
        data = Data.from_host_fd(
            noise, delta_t=1 / 4096, start_time=GPS - 131070 + index / 8
        )
        detector.set_data(data)
        detector.set_psd(PowerSpectrum(psd, data.frequencies))
        detector.time_dependent_response = True
        detector.finite_arm_response = True
        detector.configure_orbital_motion_response(
            enabled=True,
            reference_time=GPS,
            validity_s=(-131072.125, 0.125),
            acceleration_over_c=(2e-11, -1e-12, -4e-13),
            jerk_over_c=(1.4e-20, 3.8e-18, 1.7e-18),
        )
        data.fd[2:] += np.asarray(
            detector.fd_response(frequency[2:], sky, params, optimize=False)
        )
        detector.set_frequency_bounds(5 if detector.name == "CE" else 2, 2048)
    return detectors, waveform


@pytest.mark.parametrize("phase", [False, True])
def test_full_2_to_2048_hz_bright_source_against_stock_native_oracle(
    full_band_network, phase, record_property
):
    edges = np.array([2.0, 5.1, 17.0, 64.0, 256.0, 1024.0, 2048.0])
    points = [
        PARAMETERS,
        {**PARAMETERS, "d_L": 1.0, "iota": 0.0, "ra": 1.3, "t_c": -0.01},
        {**PARAMETERS, "M_c": 200.0, "lambda_1": 0.0, "lambda_2": 0.0, "d_L": 2000.0},
    ]
    likelihood = construct(
        full_band_network, points, phase=phase, edges=edges, chunk_size=256
    )
    errors = []
    for point, actual in zip(points, likelihood.native_probe_results, strict=True):
        expected = _direct_native_sums(full_band_network, point, phase=phase)
        errors.append(abs(actual["log_likelihood"] - expected["log_likelihood"]))
        np.testing.assert_allclose(
            actual["log_likelihood"], expected["log_likelihood"], rtol=0, atol=0.001
        )
        for name, channel in actual["channels"].items():
            assert (
                channel["native_samples"]
                == expected["channels"][name]["native_samples"]
            )
            assert channel["frequency_max"] == 2048.0
            assert channel["chunks"] == (channel["native_samples"] + 255) // 256
    record_property("maximum_native_log_likelihood_difference_nats", max(errors))
