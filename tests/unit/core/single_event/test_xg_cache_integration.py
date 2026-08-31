"""End-to-end source-cache tests for the rotating detector response."""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomD

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"


def _cache_digest(cache) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(cache):
        array = np.asarray(jax.device_get(leaf))
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("finite_arm_response", [False, True])
def test_rotating_response_reuses_only_intrinsic_source_cache(finite_arm_response):
    detector = get_H1()
    detector.set_data(
        Data.from_file(str(FIXTURES_DIR / "GW150914_strain_H1.npz"))
    )
    detector.set_psd(
        PowerSpectrum.from_file(str(FIXTURES_DIR / "GW150914_psd_H1.npz"))
    )
    detector.time_dependent_response = True
    detector.finite_arm_response = finite_arm_response
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD(f_ref=20.0))
    likelihood = TransientLikelihoodFD(
        detectors=[detector],
        waveform=waveform,
        f_min=20.0,
        f_max=128.0,
        trigger_time=1_126_259_462.4,
    )
    params = {
        "M_c": 30.0,
        "eta": 0.249,
        "s1_z": 0.01,
        "s2_z": -0.02,
        "d_L": 400.0,
        "phase_c": 0.0,
        "t_c": 0.0,
        "iota": 0.4,
        "ra": 1.375,
        "dec": -1.2108,
        "psi": 0.2,
    }

    cache = likelihood.generate_waveform(params)
    initial_digest = _cache_digest(cache)
    proposals = (
        {**params, "ra": 5.8, "dec": 0.7, "psi": 1.1},
        {**params, "t_c": 0.04},
        {**params, "d_L": 900.0},
    )

    for proposal in proposals:
        cached = likelihood.evaluate_from_waveform(proposal, cache)
        rebuilt = likelihood.evaluate(proposal)
        np.testing.assert_allclose(cached, rebuilt, rtol=1e-11, atol=1e-11)
        assert _cache_digest(cache) == initial_digest

    cached_gradient = jax.grad(
        lambda tc: likelihood.evaluate_from_waveform(
            {**params, "t_c": tc}, cache
        )
    )(jnp.asarray(0.01))
    rebuilt_gradient = jax.grad(
        lambda tc: likelihood.evaluate({**params, "t_c": tc})
    )(jnp.asarray(0.01))
    np.testing.assert_allclose(
        cached_gradient,
        rebuilt_gradient,
        rtol=2e-10,
        atol=2e-10,
    )
