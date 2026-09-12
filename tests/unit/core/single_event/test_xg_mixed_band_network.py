"""Small analytic-PSD CE-A/ET integration, not a physical-duration recovery test.

The 0.25 Hz FD quadrature deliberately does not resolve a 77747 s signal in
time. The waveform clock and detector response do retain that long emission
time. This checks masks and coherent likelihood algebra, not quadrature
convergence, orbital accuracy, or the production 131072 s heterodyne plan.
Rotation and finite arms are retained; orbital motion is covered separately by
the full-band network tests with explicit coefficients. CE contributes only
at frequencies supported by its analytic test PSD.
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import i0e

from jimgw.cli._config import PipelineConfig
from jimgw.cli._data import build_data
from jimgw.cli._prior import build_prior
from jimgw.cli.xg_qualification import (
    bind_xg_qualification_candidate,
    build_xg_qualification_candidate,
    plan_xg_qualification_bin_edges,
)
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.likelihood import (
    TransientLikelihoodFD,
)
from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from tests.xg_fixtures import network_config

jax.config.update("jax_enable_x64", True)

LOW = {"CE": 5.0, "ET1": 2.0, "ET2": 2.0, "ET3": 2.0}
GPS = 1_300_000_000.0
DF = 0.25
FREQUENCIES = np.arange(2.0, 64.0 + DF, DF)


def _reference():
    q = 0.9724191074091124
    return {
        "M_c": 1.1802650981093186,
        "eta": q / (1.0 + q) ** 2,
        "s1_z": -0.014169601016451747,
        "s2_z": -0.02350583516601907,
        "lambda_1": 483.1557044220429,
        "lambda_2": 771.0436351634714,
        "d_L": 200.0,
        "iota": 2.016083447361016,
        "ra": 6.052874807646849,
        "dec": 0.17257877754217157,
        "psi": 1.6782976747256861,
        "phase_c": 0.0,
        "t_c": 0.03512602655783986,
        "trigger_time": GPS,
        "gmst": greenwich_mean_sidereal_time(GPS),
    }


@pytest.fixture(scope="module")
def network(tmp_path_factory):
    raw = network_config()
    raw["data"].update(duration=4.0, sampling_frequency=128.0, zero_noise=True)
    raw["data"]["injection_parameters"].update(d_L=200.0, phase_c=0.0)
    psd_dir = tmp_path_factory.mktemp("mixed-band-psds")
    for name, support_start in (("CE", 5.0), ("ET", 1.0)):
        frequency = np.geomspace(support_start, 128.0, 257)
        psd = 1e-45 * (1 + (10 / frequency) ** 4 + (frequency / 100) ** 2)
        path = psd_dir / f"{name}.txt"
        np.savetxt(path, np.column_stack((frequency, psd)))
        raw["data"]["psd_files"][name] = str(path)
    lc = raw["likelihood"]
    lc["f_max"] = 64.0
    edges = np.r_[np.arange(2.0, 8.0, 0.25), np.arange(8.0, 65.0)]
    lc["heterodyne"].update(
        n_bins=len(edges) - 1,
        frequency_bin_edges=edges.tolist(),
        phasor_time_anchors=[-0.2, 0.0, 0.2],
        reference_parameters={"type": "provided", "values": _reference()},
    )
    raw["output"]["dir"] = str(tmp_path_factory.mktemp("mixed-band-output"))
    cfg = PipelineConfig.model_validate(raw, context={"prepare_xg_qualification": True})
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD_NRTidalv2(f_ref=20.0))
    reference = _reference()
    sky = waveform(jnp.asarray(FREQUENCIES), reference)
    detectors = build_data(
        cfg.data,
        f_min=LOW,
        f_max=64.0,
        waveform=waveform,
        time_frame=cfg.sampling.time_frame,
        time_dependent_response=True,
        finite_arm_response=True,
        seed=cfg.seed,
        input_provenance_sha256=cfg.xg_input_files_sha256(),
    )
    inputs = {}
    for detector in detectors:
        native = np.asarray(detector.data.frequencies)
        full_mask = (native >= 2.0) & (native <= 64.0)
        np.testing.assert_array_equal(native[full_mask], FREQUENCIES)
        # Retain independent full-band input arrays for the oracle. It must
        # not obtain its masks or data from likelihood.sliced_* properties.
        inputs[detector.name] = {
            "data": np.asarray(detector.data.fd)[full_mask].copy(),
            "psd": np.asarray(detector.psd.values)[full_mask].copy(),
        }
    return detectors, waveform, reference, inputs, sky, raw


@pytest.fixture(scope="module", params=[False, True], ids=["phase-fixed", "phase-marg"])
def likelihoods(network, request):
    detectors, waveform, _, _, _, raw = network
    raw = copy.deepcopy(raw)
    raw["likelihood"]["phase_marginalization"] = request.param
    if not request.param:
        raw["prior"]["phase_c"] = {"type": "uniform", "min": 0.0, "max": 2.0 * np.pi}
    cfg = PipelineConfig.model_validate(raw, context={"prepare_xg_qualification": True})
    shared = {
        "detectors": detectors,
        "waveform": waveform,
        "f_min": LOW,
        "f_max": 64.0,
        "trigger_time": GPS,
        "phase_marginalization": request.param,
    }
    dense = TransientLikelihoodFD(**shared)
    digest = plan_xg_qualification_bin_edges(cfg, detectors, waveform)
    binding = bind_xg_qualification_candidate(cfg, digest)
    heterodyne = build_xg_qualification_candidate(
        binding, cfg, detectors, waveform, build_prior(cfg.prior), []
    )
    assert cfg.verified_xg_manifest is None
    return dense, heterodyne, request.param


def _manual_terms(network, parameters, *, minimum=2.0):
    detectors, waveform, _, inputs, _, _ = network
    sky = waveform(jnp.asarray(FREQUENCIES), parameters)
    z, q = [], []
    for detector in detectors:
        mask = FREQUENCIES >= max(LOW[detector.name], minimum)
        projected = np.asarray(
            detector.fd_response(jnp.asarray(FREQUENCIES), sky, parameters)
        )[mask]
        data = inputs[detector.name]["data"][mask]
        psd = inputs[detector.name]["psd"][mask]
        z.append(4.0 * DF * np.sum(np.conj(projected) * data / psd))
        q.append(4.0 * DF * np.sum(np.abs(projected) ** 2 / psd))
    return np.asarray(z), np.asarray(q)


def _log_i0(value):
    return np.log(i0e(value)) + np.abs(value)


def _manual_likelihood(z, q, marginalize):
    match = _log_i0(np.abs(z.sum())) if marginalize else z.sum().real
    return match - 0.5 * q.sum()


def test_detector_specific_bands_and_actual_two_hz_clock(network, likelihoods):
    detectors, _, reference, inputs, sky, _ = network
    dense, heterodyne, _ = likelihoods
    np.testing.assert_array_equal(dense.frequencies, FREQUENCIES)
    np.testing.assert_array_equal(heterodyne.frequencies, FREQUENCIES)
    assert not heterodyne.identical_frequency_grids
    for detector, mask in zip(detectors, dense.frequency_masks, strict=True):
        expected = FREQUENCIES >= LOW[detector.name]
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(
            detector.sliced_frequencies, FREQUENCIES[expected]
        )
        assert np.all(np.isfinite(detector.sliced_psd))
    assert np.all(inputs["CE"]["data"][FREQUENCIES < 5.0] == 0.0)
    tau = np.asarray(sky["__tau__"])
    assert 77_000.0 < tau[0] < 78_000.0
    assert np.all(np.diff(tau) < 0)
    offsets = reference["t_c"] - tau
    assert offsets.min() > -131072.125 and offsets.max() < 0.125


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"t_c": _reference()["t_c"] + 0.0002},
        {"M_c": _reference()["M_c"] + 1e-8},
        {
            "ra": _reference()["ra"] + 0.001,
            "dec": _reference()["dec"] - 0.001,
            "d_L": 200.2,
        },
    ],
    ids=["injection", "time", "mass", "sky-distance"],
)
def test_dense_and_carrier_heterodyne_match_coherent_oracle(
    network, likelihoods, changes
):
    dense, heterodyne, marginalize = likelihoods
    parameters = {**network[2], **changes}
    z, q = _manual_terms(network, parameters)
    expected = _manual_likelihood(z, q, marginalize)
    assert np.isfinite(expected)
    np.testing.assert_allclose(dense.evaluate(parameters), expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        heterodyne.evaluate(parameters), expected, rtol=0, atol=2e-5
    )


def test_et_low_band_and_single_coherent_phase_are_observable(network, likelihoods):
    dense, _, marginalize = likelihoods
    z, q = _manual_terms(network, network[2])
    z_high, q_high = _manual_terms(network, network[2], minimum=5.0)
    # Compiled injection and the eager oracle can round the long waveform
    # phase differently. Bound the actual noise-weighted waveform error,
    # whose half is the zero-noise log-likelihood loss, rather than requiring
    # an exactly real overlap. This is a numerical budget for this tiny fixture.
    detectors, _, reference, inputs, sky, _ = network
    mismatch = []
    for detector in detectors:
        mask = FREQUENCIES >= LOW[detector.name]
        projected = np.asarray(
            detector.fd_response(jnp.asarray(FREQUENCIES), sky, reference)
        )[mask]
        data = inputs[detector.name]["data"][mask]
        psd = inputs[detector.name]["psd"][mask]
        assert np.all(np.isfinite(projected)) and np.all(np.isfinite(data))
        mismatch.append(4.0 * DF * np.sum(np.abs(data - projected) ** 2 / psd))
    mismatch = np.asarray(mismatch)
    assert mismatch.sum() <= 1e-14
    overlap_bound = np.sqrt(q * mismatch) + 64.0 * np.finfo(np.float64).eps * q
    assert np.all(np.abs(z - q) <= overlap_bound)
    np.testing.assert_allclose(z.real, q, rtol=2e-14, atol=1e-9)
    assert q[0] == q_high[0]  # CE never obtains fictitious low-frequency power.
    assert np.all(q[1:] - q_high[1:] > 0.01)
    full = _manual_likelihood(z, q, marginalize)
    without_et_low = _manual_likelihood(z_high, q_high, marginalize)
    assert abs(full - without_et_low) > 0.01
    np.testing.assert_allclose(dense.evaluate(network[2]), full, rtol=0, atol=1e-8)
    if marginalize:
        independently_marginalized = np.sum(_log_i0(np.abs(z))) - 0.5 * q.sum()
        assert abs(full - independently_marginalized) > 1.0
