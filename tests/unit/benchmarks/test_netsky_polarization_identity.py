from __future__ import annotations

import copy

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign.run_injection import (
    _analysis_components,
    _transient_likelihood_kwargs,
)

_DURATION_SECONDS = 1.0
_SAMPLING_FREQUENCY_HZ = 256.0
_F_MAX_HZ = 0.5 * _SAMPLING_FREQUENCY_HZ
_LIKELIHOOD_F_MAX_HZ = _F_MAX_HZ - 1.0 / _DURATION_SECONDS


def _campaign_config() -> dict[str, object]:
    """Return the production NETSKY likelihood contract on a compact grid."""

    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config.update(
        {
            "blocking_scheme": common.NETSKY_SCHEME,
            "duration_seconds": _DURATION_SECONDS,
            "sampling_frequency_hz": _SAMPLING_FREQUENCY_HZ,
            "f_max_hz": _F_MAX_HZ,
            "likelihood_f_max_hz": _LIKELIHOOD_F_MAX_HZ,
            "phase_marginalization": True,
            "time_marginalization": {
                "tc_range_seconds": [-0.03, 0.03],
                "upsample_factor": 1,
            },
            "distance_marginalization": False,
        }
    )
    config["prior"]["t_c"]["range_seconds"] = [-0.03, 0.03]
    return config


def _physical_points() -> tuple[dict[str, float], ...]:
    intrinsic = {
        "M_c": 1.95,
        "q": 0.82,
        "s1_mag": 0.021,
        "s1_theta": 0.73,
        "s1_phi": 0.61,
        "s2_mag": 0.017,
        "s2_theta": 2.12,
        "s2_phi": 4.37,
        "lambda_1": 650.0,
        "lambda_2": 910.0,
    }
    return (
        {
            **intrinsic,
            "iota": 0.46,
            "ra": 0.41,
            "dec": -0.62,
            "psi": 0.13,
            "d_L": 54.0,
        },
        {
            **intrinsic,
            "iota": 1.31,
            "ra": 2.27,
            "dec": 0.24,
            "psi": 1.17,
            "d_L": 96.0,
        },
        {
            **intrinsic,
            "iota": 2.43,
            "ra": 5.08,
            "dec": 0.79,
            "psi": 2.81,
            "d_L": 139.0,
        },
    )


def _transform_forward(transforms: list[object], values: dict[str, object]):
    transformed = values
    for transform in transforms:
        transformed = transform.forward(transformed)
    return transformed


def _transform_backward(transforms: list[object], values: dict[str, object]):
    transformed = values
    for transform in reversed(transforms):
        transformed = transform.backward(transformed)
    return transformed


@pytest.fixture(scope="module")
def netsky_likelihood():
    import jax.numpy as jnp

    from benchmarks.device_parallel_nss.paper_model import (
        RippleIMRPhenomPv2NRTidalv2,
    )
    from jimgw.core.single_event.data import PowerSpectrum
    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
    from jimgw.core.single_event.likelihood import TransientLikelihoodFD

    config = _campaign_config()
    ifos = [get_H1(), get_L1(), get_V1()]
    frequencies = jnp.fft.rfftfreq(
        int(_DURATION_SECONDS * _SAMPLING_FREQUENCY_HZ),
        1.0 / _SAMPLING_FREQUENCY_HZ,
    )
    for ifo, scale in zip(ifos, (1.0, 1.25, 2.0), strict=True):
        ifo.set_psd(
            PowerSpectrum(
                scale * 1.0e-47 * jnp.ones_like(frequencies),
                frequencies,
                name=f"{ifo.name}_compact_campaign_psd",
            )
        )

    components = _analysis_components(config, jnp, ifos)
    injection = _physical_points()[1]
    injection = _transform_forward(
        components["likelihood_transforms"],
        {**injection, "phase_c": 0.71, "t_c": 0.0},
    )
    for ifo in ifos:
        ifo.inject_signal(
            duration=_DURATION_SECONDS,
            sampling_frequency=_SAMPLING_FREQUENCY_HZ,
            trigger_time=float(config["trigger_time_gps"]),
            waveform_model=components["waveform"],
            parameters=injection,
            f_min=float(config["f_min_hz"]),
            f_max=_F_MAX_HZ,
            start_time=float(config["trigger_time_gps"]) - 0.5 * _DURATION_SECONDS,
            zero_noise=True,
        )

    likelihood = TransientLikelihoodFD(
        ifos,
        waveform=components["waveform"],
        **_transient_likelihood_kwargs(config, components),
    )
    assert type(components["waveform"]) is RippleIMRPhenomPv2NRTidalv2
    return config, components, likelihood


def test_campaign_waveform_obeys_full_period_polarization_identity(
    netsky_likelihood,
) -> None:
    config, components, likelihood = netsky_likelihood

    assert config["waveform"] == "IMRPhenomPv2_NRTidalv2"
    assert config["carrier_time_anchor"] == "imrphenomd"
    assert config["prior"]["phase_c"] == {
        "distribution": "uniform",
        "range_radians": [0.0, "2pi"],
    }
    assert config["prior"]["psi"] == {
        "distribution": "uniform",
        "range_radians": [0.0, "pi"],
    }
    assert likelihood.phase_marginalization is True
    assert likelihood.time_marginalization is True
    assert likelihood.detector_names == ["H1", "L1", "V1"]

    expected_sampling_names = {name for block in common.NETSKY_BLOCKS for name in block}
    cache_parameters = _transform_forward(
        components["likelihood_transforms"], _physical_points()[0]
    )
    shared_cache = likelihood.generate_waveform(cache_parameters)
    for physical in _physical_points():
        sampled = _transform_forward(components["sample_transforms"], physical)
        assert set(sampled) == expected_sampling_names
        assert -1.0 < float(sampled["cos_zenith"]) < 1.0
        assert -1.0 < float(sampled["cos_iota"]) < 1.0
        assert 0.0 <= float(sampled["azimuth"]) < 2.0 * np.pi
        assert 0.0 <= float(sampled["psi"]) < np.pi
        assert np.isfinite(float(sampled["log_d_hat"]))

        shifted_sampled = dict(sampled)
        shifted_sampled["psi"] = np.mod(
            float(sampled["psi"]) + 0.5 * np.pi,
            np.pi,
        )
        shifted_physical = _transform_backward(
            components["sample_transforms"], shifted_sampled
        )
        assert float(shifted_physical["d_L"]) == pytest.approx(
            physical["d_L"], rel=2.0e-12, abs=2.0e-10
        )
        polarization_image = {
            **physical,
            "psi": float(shifted_sampled["psi"]),
        }

        base_parameters = _transform_forward(
            components["likelihood_transforms"], physical
        )
        shifted_parameters = _transform_forward(
            components["likelihood_transforms"], polarization_image
        )

        direct = np.asarray(
            [
                likelihood.evaluate(base_parameters),
                likelihood.evaluate(shifted_parameters),
            ],
            dtype=float,
        )
        cached = np.asarray(
            [
                likelihood.evaluate_from_waveform(base_parameters, shared_cache),
                likelihood.evaluate_from_waveform(shifted_parameters, shared_cache),
            ],
            dtype=float,
        )

        assert np.isfinite(direct).all()
        assert np.isfinite(cached).all()
        np.testing.assert_allclose(direct[1], direct[0], rtol=2.0e-13, atol=2.0e-10)
        np.testing.assert_allclose(cached[1], cached[0], rtol=2.0e-13, atol=2.0e-10)
        np.testing.assert_allclose(cached, direct, rtol=2.0e-12, atol=2.0e-10)
