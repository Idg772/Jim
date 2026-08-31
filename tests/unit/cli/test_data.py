"""Tests for detector-response configuration during data construction."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.cli import _data
from jimgw.cli._config import FileDataConfig, InjectionDataConfig
from jimgw.cli._likelihood import (
    _validate_xg_detector_inputs,
    detector_metadata_sha256,
)
from jimgw.cli._transforms import to_likelihood_space
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import GroundBased2G, get_H1, get_L1
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time
from jimgw.core.single_event.waveform import RippleIMRPhenomD

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures"


def _file_config() -> FileDataConfig:
    return FileDataConfig(
        detectors=["H1"],
        trigger_time=1_126_259_462.4,
        strain_files={"H1": "unused-strain.npz"},
        psd_files={"H1": "unused-psd.npz"},
    )


def test_response_flags_are_set_before_data_or_injection_loading(monkeypatch):
    observed = {}

    def capture_flags(ifos, config, **kwargs):
        del config, kwargs
        observed["dynamic"] = ifos[0].time_dependent_response
        observed["finite"] = ifos[0].finite_arm_response
        ifos[0].set_data(Data(jnp.zeros(8), 0.25))
        ifos[0].set_psd(PowerSpectrum(jnp.ones(5), jnp.arange(5, dtype=float) * 0.5))

    monkeypatch.setattr(_data, "_load_files", capture_flags)
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD(f_ref=20.0))

    detectors = _data.build_data(
        _file_config(),
        f_min=0.5,
        f_max=2.0,
        waveform=waveform,
        time_dependent_response=True,
        finite_arm_response=True,
    )

    assert observed == {"dynamic": True, "finite": True}
    assert detectors[0].time_dependent_response is True
    assert detectors[0].finite_arm_response is True


def test_detector_metadata_receipt_changes_with_physical_geometry():
    detector = get_H1()
    baseline = detector_metadata_sha256([detector])
    detector.arm_length_m = float(detector.arm_length_m) + 1.0

    assert detector_metadata_sha256([detector]) != baseline


def test_xg_detector_input_provenance_and_precision_are_realized():
    detector = get_H1()
    detector.set_data(Data(jnp.zeros(8, dtype=jnp.float64), 0.25))
    detector.set_psd(
        PowerSpectrum(
            jnp.ones(5, dtype=jnp.float64),
            jnp.arange(5, dtype=jnp.float64) * 0.5,
        )
    )
    inputs = {"strain:H1": "a" * 64, "psd:H1": "b" * 64}
    detector.input_provenance_sha256 = tuple(sorted(inputs.items()))

    _validate_xg_detector_inputs([detector], inputs)

    detector.set_data(detector.data)
    with pytest.raises(ValueError, match="not built from the qualified"):
        _validate_xg_detector_inputs([detector], inputs)


def test_xg_detector_input_precision_rejects_float32():
    detector = get_H1()
    detector.set_data(Data(jnp.zeros(8, dtype=jnp.float32), 0.25))
    detector.set_psd(
        PowerSpectrum(
            jnp.ones(5, dtype=jnp.float32),
            jnp.arange(5, dtype=jnp.float32) * 0.5,
        )
    )
    inputs = {"strain:H1": "a" * 64, "psd:H1": "b" * 64}
    detector.input_provenance_sha256 = tuple(sorted(inputs.items()))

    with pytest.raises(TypeError, match="requires float64"):
        _validate_xg_detector_inputs([detector], inputs)


def test_dynamic_data_construction_rejects_untimed_waveform(monkeypatch):
    monkeypatch.setattr(
        _data,
        "_load_files",
        lambda ifos, config, **kwargs: None,
    )

    with pytest.raises(ValueError, match="requires a waveform emission-time cache"):
        _data.build_data(
            _file_config(),
            f_min=5.0,
            f_max=2_048.0,
            waveform=RippleIMRPhenomD(f_ref=20.0),
            time_dependent_response=True,
        )


def test_finite_arm_data_construction_requires_detector_metadata(monkeypatch):
    monkeypatch.setattr(
        _data,
        "get_detector_preset",
        lambda: {"H1": GroundBased2G("H1")},
    )
    monkeypatch.setattr(
        _data,
        "_load_files",
        lambda ifos, config, **kwargs: None,
    )

    with pytest.raises(ValueError, match="requires arm-length metadata"):
        _data.build_data(
            _file_config(),
            f_min=5.0,
            f_max=2_048.0,
            waveform=RippleIMRPhenomD(f_ref=20.0),
            finite_arm_response=True,
        )


def test_dynamic_finite_arm_injection_matches_recovery_projection():
    trigger_time = 1_126_259_462.4
    injection_parameters = {
        "M_c": 30.0,
        "q": 0.95,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "d_L": 400.0,
        "phase_c": 0.0,
        "t_c": 0.0,
        "iota": 0.4,
        "ra": 1.375,
        "dec": -1.2108,
        "psi": 0.2,
    }
    config = InjectionDataConfig(
        detectors=["H1"],
        trigger_time=trigger_time,
        duration=4.0,
        sampling_frequency=2048.0,
        injection_parameters=injection_parameters,
        zero_noise=True,
        psd_files={"H1": FIXTURES_DIR / "GW150914_psd_H1.npz"},
        waveform_chunk_size=31,
    )
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD(f_ref=20.0))

    detectors = _data.build_data(
        config,
        f_min=20.0,
        f_max=512.0,
        waveform=waveform,
        time_frame="geocentric",
        time_dependent_response=True,
        finite_arm_response=True,
    )
    detector = detectors[0]
    likelihood_parameters = to_likelihood_space(
        injection_parameters,
        waveform_f_ref=float(waveform.f_ref),
        trigger_time=trigger_time,
        ifos=detectors,
        time_frame="geocentric",
    )
    likelihood_parameters["trigger_time"] = trigger_time
    likelihood_parameters["gmst"] = float(greenwich_mean_sidereal_time(trigger_time))
    expected = detector.fd_response(
        detector.sliced_frequencies,
        waveform(detector.sliced_frequencies, likelihood_parameters),
        likelihood_parameters,
    )

    np.testing.assert_allclose(
        detector.sliced_fd_data,
        expected,
        rtol=2e-12,
        atol=2e-12,
    )


def test_xg_injection_rejects_unversioned_builtin_sensitivity():
    config = InjectionDataConfig(
        detectors=["H1"],
        trigger_time=1_126_259_462.4,
        duration=4.0,
        sampling_frequency=2048.0,
        injection_parameters={
            "M_c": 30.0,
            "q": 0.95,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "d_L": 400.0,
            "phase_c": 0.0,
            "t_c": 0.0,
            "iota": 0.4,
            "ra": 1.375,
            "dec": -1.2108,
            "psi": 0.2,
        },
        zero_noise=True,
    )
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD(f_ref=20.0))

    with pytest.raises(ValueError, match="built-in.*not qualified"):
        _data.build_data(
            config,
            f_min=20.0,
            f_max=512.0,
            waveform=waveform,
            time_frame="geocentric",
            time_dependent_response=True,
        )


def test_psd_validation_rejects_nonpositive_in_band_values():
    psd = PowerSpectrum(
        jnp.asarray([1.0, 1.0, 0.0, 1.0]),
        jnp.asarray([0.0, 10.0, 20.0, 30.0]),
    )

    with pytest.raises(ValueError, match="strictly positive"):
        _data._validate_psd_values(psd, 10.0, 30.0, "test PSD")


def test_injection_noise_keys_are_seeded_and_detector_distinct(monkeypatch):
    config = InjectionDataConfig(
        detectors=["H1", "L1"],
        trigger_time=1_126_259_462.4,
        duration=4.0,
        sampling_frequency=2048.0,
        injection_parameters={
            "M_c": 30.0,
            "q": 0.95,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "d_L": 400.0,
            "phase_c": 0.0,
            "t_c": 0.0,
            "iota": 0.4,
            "ra": 1.375,
            "dec": -1.2108,
            "psi": 0.2,
        },
        psd_files={
            "H1": FIXTURES_DIR / "GW150914_psd_H1.npz",
            "L1": FIXTURES_DIR / "GW150914_psd_L1.npz",
        },
    )
    waveform = RippleIMRPhenomD(f_ref=20.0)
    observed_keys = []

    def capture_key(self, **kwargs):
        del self
        observed_keys.append(np.asarray(jax.random.key_data(kwargs["rng_key"])))

    monkeypatch.setattr(GroundBased2G, "inject_signal", capture_key)

    for _ in range(2):
        _data._load_injection(
            [get_H1(), get_L1()],
            config,
            waveform,
            f_min=20.0,
            f_max=512.0,
            time_frame="geocentric",
            seed=17,
            require_configured_psd=True,
        )

    np.testing.assert_array_equal(observed_keys[0], observed_keys[2])
    np.testing.assert_array_equal(observed_keys[1], observed_keys[3])
    assert not np.array_equal(observed_keys[0], observed_keys[1])
