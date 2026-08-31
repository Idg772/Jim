from collections.abc import Mapping
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Complex, Float
from ripplegw.interfaces import FrequencyDomainWaveform

from jimgw.core.single_event.dominant_mode import (
    TIME_TO_COALESCENCE_KEY,
    DominantModeTimeCachedWaveform,
)
from jimgw.core.single_event.waveform import (
    RippleIMRPhenomD,
    RippleIMRPhenomHM,
    RippleIMRPhenomPv2,
    RippleTaylorF2,
)
from jimgw.typing import FloatLike

TAYLOR_PARAMS = {
    "M_c": 1.2,
    "eta": 0.24,
    "s1_z": 0.02,
    "s2_z": -0.01,
    "lambda_1": 300.0,
    "lambda_2": 400.0,
    "d_L": 100.0,
    "phase_c": 0.1,
    "iota": 0.4,
}


class _CustomCacheWaveform(FrequencyDomainWaveform):
    f_ref = 20.0
    waveform_metadata: ClassVar[dict[str, object]] = {
        "domain": "FD",
        "is_precessing": False,
        "source_type": "cbc",
    }

    def __init__(self):
        self.build_calls = 0
        self.reconstruction_calls = 0

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("M_c", "eta", "s1_z", "s2_z", "d_L", "iota")

    @property
    def cacheable_parameter_names(self) -> frozenset[str]:
        return frozenset(("d_L", "iota"))

    def __call__(
        self,
        frequency: Float[Array, " n"],
        params: Mapping[str, FloatLike],
    ) -> dict[str, Complex[Array, " n"]]:
        carrier = (params["M_c"] + 1j * params["eta"]) * frequency
        return {
            "p": carrier * (1.0 + jnp.cos(params["iota"]) ** 2) / params["d_L"],
            "c": -2j * carrier * jnp.cos(params["iota"]) / params["d_L"],
        }

    def build_waveform_cache(self, frequency, params):
        self.build_calls += 1
        return {"carrier": (params["M_c"] + 1j * params["eta"]) * frequency}

    def waveform_from_cache(self, frequency, params, cache):
        del frequency
        self.reconstruction_calls += 1
        carrier = cache["carrier"]
        return {
            "p": carrier * (1.0 + jnp.cos(params["iota"]) ** 2) / params["d_L"],
            "c": -2j * carrier * jnp.cos(params["iota"]) / params["d_L"],
        }


def test_direct_adapter_output_contains_polarizations_and_positive_tau():
    waveform = DominantModeTimeCachedWaveform(RippleTaylorF2(f_ref=20.0))
    frequency = jnp.asarray([5.0, 10.0, 20.0, 100.0])

    output = waveform(frequency, TAYLOR_PARAMS)

    assert set(output) == {"p", "c", TIME_TO_COALESCENCE_KEY}
    assert output["p"].shape == frequency.shape
    assert output["c"].shape == frequency.shape
    assert waveform.time_dependent_response is True
    assert waveform.f_ref == 20.0
    assert np.all(np.asarray(output[TIME_TO_COALESCENCE_KEY]) > 0.0)
    assert np.all(np.diff(np.asarray(output[TIME_TO_COALESCENCE_KEY])) < 0.0)


def test_unit_distance_cache_scales_only_physical_polarizations():
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD(f_ref=20.0))
    frequency = jnp.linspace(20.0, 200.0, 64)
    params = {
        key: value
        for key, value in TAYLOR_PARAMS.items()
        if key not in {"lambda_1", "lambda_2"}
    }
    cache = waveform.build_waveform_cache(frequency, params)
    first = waveform.waveform_from_cache(frequency, params, cache)
    distant_params = {**params, "d_L": 250.0}
    distant = waveform.waveform_from_cache(frequency, distant_params, cache)

    np.testing.assert_allclose(
        np.asarray(distant["p"]), np.asarray(first["p"]) * 100.0 / 250.0
    )
    np.testing.assert_allclose(
        np.asarray(distant["c"]), np.asarray(first["c"]) * 100.0 / 250.0
    )
    assert distant[TIME_TO_COALESCENCE_KEY] is cache[TIME_TO_COALESCENCE_KEY]
    np.testing.assert_array_equal(
        np.asarray(distant[TIME_TO_COALESCENCE_KEY]),
        np.asarray(first[TIME_TO_COALESCENCE_KEY]),
    )


def test_adapter_delegates_custom_iota_and_distance_reconstruction():
    source = _CustomCacheWaveform()
    waveform = DominantModeTimeCachedWaveform(source)
    frequency = jnp.linspace(5.0, 80.0, 32)
    params = {
        "M_c": 1.21,
        "eta": 0.247,
        "s1_z": 0.03,
        "s2_z": -0.02,
        "d_L": 70.0,
        "iota": 0.2,
    }
    cache = waveform.build_waveform_cache(frequency, params)
    changed = {**params, "d_L": 190.0, "iota": 1.2}

    reconstructed = waveform.waveform_from_cache(frequency, changed, cache)
    direct = waveform(frequency, changed)

    assert source.build_calls == 1
    assert source.reconstruction_calls == 1
    assert waveform.cacheable_parameter_names == frozenset(("d_L", "iota"))
    assert waveform.emission_time_parameter_names == frozenset(
        ("M_c", "eta", "s1_z", "s2_z")
    )
    np.testing.assert_allclose(np.asarray(reconstructed["p"]), np.asarray(direct["p"]))
    np.testing.assert_allclose(np.asarray(reconstructed["c"]), np.asarray(direct["c"]))
    assert reconstructed[TIME_TO_COALESCENCE_KEY] is cache[TIME_TO_COALESCENCE_KEY]


def test_cache_reconstruction_is_jittable_and_keeps_tau_unchanged():
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD(f_ref=20.0))
    frequency = jnp.linspace(20.0, 100.0, 16)
    params = {
        key: value
        for key, value in TAYLOR_PARAMS.items()
        if key not in {"lambda_1", "lambda_2"}
    }
    cache = waveform.build_waveform_cache(frequency, params)

    reconstructed = jax.jit(waveform.waveform_from_cache)(frequency, params, cache)

    np.testing.assert_array_equal(
        np.asarray(reconstructed[TIME_TO_COALESCENCE_KEY]),
        np.asarray(cache[TIME_TO_COALESCENCE_KEY]),
    )


@pytest.mark.parametrize(
    "source",
    [
        RippleIMRPhenomHM(f_ref=20.0),
        RippleIMRPhenomPv2(f_ref=20.0),
    ],
)
def test_adapter_rejects_known_combined_higher_mode_and_precessing_sources(source):
    with pytest.raises(ValueError, match="does not expose"):
        DominantModeTimeCachedWaveform(source)


def test_adapter_requires_timing_and_distance_parameters():
    class IncompleteWaveform(FrequencyDomainWaveform):
        f_ref = 20.0

        @property
        def parameter_names(self):
            return ("M_c", "eta")

        def __call__(self, frequency, params):
            del params
            return {"p": frequency + 0j, "c": frequency * 0j}

    with pytest.raises(ValueError, match="missing"):
        DominantModeTimeCachedWaveform(IncompleteWaveform())


def test_adapter_rejects_reserved_or_non_tensor_output_leaves():
    class InvalidOutputWaveform(_CustomCacheWaveform):
        def __call__(self, frequency, params):
            output = super().__call__(frequency, params)
            output[TIME_TO_COALESCENCE_KEY] = jnp.ones_like(frequency)
            return output

    waveform = DominantModeTimeCachedWaveform(InvalidOutputWaveform())

    with pytest.raises(ValueError, match="exactly the p/c"):
        waveform(
            jnp.asarray([20.0]),
            {
                "M_c": 1.2,
                "eta": 0.24,
                "s1_z": 0.0,
                "s2_z": 0.0,
                "d_L": 100.0,
                "iota": 0.3,
            },
        )
