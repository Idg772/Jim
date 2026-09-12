"""Default factored evaluator versus preserved native reference algebra.

The small quadrature is not a resolved 21-hour injection. Its 2 Hz waveform
clock, 131072 s response epoch, actual CE-A/ET geometry, detector-specific
bands, independent noisy samples and all K8/M16/21-anchor moments are retained.
"""

import copy
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_CE_A, get_ET_Sardinia
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.heterodyne_extrinsics import evaluate_extrinsic_summary
from jimgw.core.single_event.likelihood import (
    _XG_QUALIFICATION_PLAN_AUTHORITY,
    HeterodynedTransientLikelihoodFD,
    _QualificationXGPlan,
)
from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.xg_evaluation import (
    XG_EVALUATION_REVISION,
    FastXGEvaluator,
    configure_xg_evaluation,
    make_dresser,
)
from jimgw.core.single_event.xg_waveform import XGBasisCachedWaveform
from jimgw.samplers.config import BlackJAXSwiGConfig

jax.config.update("jax_enable_x64", True)
GPS = 1_300_000_000.0
P = {
    "M_c": 1.1802650981093186,
    "eta": 0.2499511169561633,
    "s1_z": -0.014169601016451747,
    "s2_z": -0.02350583516601907,
    "lambda_1": 483.1557044220429,
    "lambda_2": 771.0436351634714,
    "d_L": 20.0,
    "iota": 2.016083447361016,
    "ra": 6.052874807646849,
    "dec": 0.17257877754217157,
    "psi": 1.6782976747256861,
    "phase_c": 0.0,
    "t_c": 0.03512602655783986,
}


@pytest.fixture(scope="module")
def network():
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD_NRTidalv2(f_ref=20.0))
    p = {**P, "trigger_time": GPS, "gmst": greenwich_mean_sidereal_time(GPS)}
    frequencies = np.arange(513) / 4.0
    positive = jnp.asarray(frequencies[8:])
    sky = waveform(positive, p)
    assert 77_000 < float(sky["__tau__"][0]) < 78_000
    rng = np.random.default_rng(7131)
    detectors = [get_CE_A(), *get_ET_Sardinia()]
    for index, detector in enumerate(detectors):
        buffer = np.zeros(len(frequencies), dtype=np.complex128)
        data = Data.from_host_fd(
            buffer, delta_t=1 / 256, start_time=GPS - 131070 + index / 8
        )
        detector.set_data(data)
        psd = 1e-45 * (1 + (30 / np.maximum(frequencies, 1)) ** 4)
        detector.set_psd(PowerSpectrum(psd, data.frequencies))
        detector.time_dependent_response = True
        detector.finite_arm_response = True
        detector.configure_orbital_motion_response(
            enabled=True,
            reference_time=GPS,
            validity_s=(-131072.125, 0.125),
            acceleration_over_c=(
                2.002108458580428e-11,
                -1.019481229165329e-12,
                -4.473979670828952e-13,
            ),
            jerk_over_c=(
                1.38632199910925e-20,
                3.84800286572565e-18,
                1.686090305174408e-18,
            ),
        )
        buffer[8:] = np.asarray(detector.fd_response(positive, sky, p))
        buffer += np.sqrt(psd / (4 * 0.25)) * (
            rng.normal(size=len(buffer)) + 1j * rng.normal(size=len(buffer))
        )
        detector.set_frequency_bounds(5.0 if detector.name == "CE" else 2.0, 128.0)
    return detectors, waveform


@pytest.fixture(scope="module", params=[False, True], ids=["fixed-phase", "phase-marg"])
def likelihood_pair(network, request):
    detectors, waveform = network
    edges = np.sort(np.r_[np.geomspace(2.0, 128.0, 33), 5.0])
    anchors = np.linspace(-0.2, 0.2, 21)
    digest = HeterodynedTransientLikelihoodFD._bin_edges_sha256(
        edges,
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=anchors,
        reference_projection="carrier",
    )
    options = {
        "detectors": detectors,
        "waveform": waveform,
        "f_min": {d.name: 5.0 if d.name == "CE" else 2.0 for d in detectors},
        "f_max": 128.0,
        "trigger_time": GPS,
        "n_bins": len(edges) - 1,
        "reference_parameters": P,
        "interpolation_order": 8,
        "phasor_moment_order": 16,
        "phasor_time_anchors": anchors,
        "reference_projection": "carrier",
        "frequency_bin_edges": edges,
        "reference_chunk_size": 128,
        "phase_marginalization": request.param,
        "xg_plan": _QualificationXGPlan(digest, _XG_QUALIFICATION_PLAN_AUTHORITY),
    }
    baseline = HeterodynedTransientLikelihoodFD(
        **options, xg_evaluation_mode="baseline"
    )
    fast = HeterodynedTransientLikelihoodFD(**options)
    return baseline, fast


def test_default_keeps_native_moments_and_uses_factored_evaluator(likelihood_pair):
    baseline, fast = likelihood_pair
    assert baseline.evaluation_diagnostics["implementation"] == "baseline"
    assert fast.evaluation_diagnostics["implementation"] == XG_EVALUATION_REVISION
    assert type(fast.waveform) is XGBasisCachedWaveform
    assert fast._reference_waveform is baseline.waveform
    assert fast._baseline_waveform is baseline.waveform
    for field in (
        "summary_data",
        "summary_moments",
        "phasor_data_moments",
        "waveform_node_ref",
    ):
        for original, candidate in zip(
            jax.tree.leaves(getattr(baseline, field)),
            jax.tree.leaves(getattr(fast, field)),
            strict=True,
        ):
            np.testing.assert_array_equal(original, candidate)
    assert np.any(np.asarray(fast.summary_moments["CE"][1][0]) == 0)
    assert fast.construction_diagnostics["evaluation"] == fast.evaluation_diagnostics
    assert (
        fast.evaluation_diagnostics["network_kernel"]["implementation"]
        == "batched-anchored-network-v1"
    )


def test_direct_cached_and_scalar_modes_match_baseline(likelihood_pair):
    baseline, fast = likelihood_pair
    points = [
        P,
        {**P, "M_c": P["M_c"] + 1e-8, "t_c": P["t_c"] + 2e-5},
        {**P, "ra": P["ra"] + 0.002, "dec": P["dec"] - 0.002},
        {**P, "psi": 0.4, "iota": 0.71, "d_L": 71.0},
        {**P, "eta": 0.25, "lambda_1": 510.0},
    ]
    if not baseline.phase_marginalization:
        points.append({**P, "phase_c": 0.4})
    bank = jax.tree.map(lambda *values: jnp.asarray(values), *points)
    control = np.asarray(jax.jit(jax.vmap(baseline.evaluate))(bank))
    actual = np.asarray(jax.jit(jax.vmap(fast.evaluate))(bank))
    # Absolute likelihood differences matter at large SNR; a relative test of
    # the large common logL offset would hide a scientifically meaningful error.
    np.testing.assert_allclose(actual, control, rtol=0, atol=5e-4)
    caches = jax.jit(jax.vmap(fast.generate_waveform))(bank)
    recovered = jax.jit(jax.vmap(fast.evaluate_from_waveform))(bank, caches)
    np.testing.assert_allclose(recovered, actual, rtol=0, atol=1e-7)
    summaries = jax.jit(jax.vmap(fast.build_extrinsic_summary))(bank, caches)
    scalar = jax.jit(
        jax.vmap(lambda p, summary: evaluate_extrinsic_summary(fast, p, summary))
    )(bank, summaries)
    np.testing.assert_allclose(scalar, actual, rtol=0, atol=1e-7)
    stale = jax.tree.map(lambda value: value[0], summaries)
    assert np.isnan(
        evaluate_extrinsic_summary(fast, {**P, "t_c": P["t_c"] + 1e-5}, stale)
    )
    assert np.isnan(fast.evaluate({**P, "t_c": 1.0}))


def test_jim_uses_selected_summary_hook(likelihood_pair):
    _, likelihood = likelihood_pair
    if not likelihood.phase_marginalization:
        return
    sampled = {key: value for key, value in P.items() if key != "phase_c"}
    widths = {"M_c": 1e-7, "eta": 1e-6, "t_c": 1e-3, "s1_z": 0.001, "s2_z": 0.001}
    prior = CombinePrior(
        [
            UniformPrior(
                value - widths.get(key, 0.001),
                value + widths.get(key, 0.001),
                parameter_names=[key],
            )
            for key, value in sampled.items()
        ]
    )
    intrinsic_sky = [key for key in sampled if key not in {"psi", "iota", "d_L"}]
    config = BlackJAXSwiGConfig(
        blocks=[intrinsic_sky, ["psi"], ["iota", "d_L"]],
        scheduler="fsm",
        scalar_extrinsic_cache=True,
        n_live=16,
        n_delete_frac=0.25,
    )
    jim = Jim(likelihood, prior, sampler_config=config)
    position = jnp.asarray([sampled[key] for key in jim.sampling_parameter_names])
    cache = jim._sampler_backend_kwargs["build_cache"](position)
    expected = likelihood.build_extrinsic_summary(P, cache)
    actual = jim._sampler_backend_kwargs["prepare_hit_summary"](position, cache)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)


def test_unsupported_custom_adapter_and_response_fall_back(likelihood_pair):
    baseline, _ = likelihood_pair

    class CustomClock(DominantModeTimeCachedWaveform):
        pass

    candidate = copy.copy(baseline)
    candidate._baseline_waveform = CustomClock(baseline.waveform.source)
    configure_xg_evaluation(candidate, baseline.waveform)
    assert candidate._xg_fast_evaluator is None
    assert candidate.waveform is candidate._baseline_waveform
    assert "unsupported waveform" in candidate.evaluation_diagnostics["fallback_reason"]
    candidate = copy.copy(baseline)
    candidate.detectors = copy.deepcopy(baseline.detectors)
    candidate.detectors[0].finite_arm_response = False
    configure_xg_evaluation(candidate, baseline.waveform)
    assert candidate._xg_fast_evaluator is None
    with pytest.raises(ValueError, match="xg_evaluation_mode"):
        configure_xg_evaluation(candidate, baseline.waveform, "unknown")


@pytest.mark.parametrize("hook", ["_source_projection", "delay_from_geocenter"])
def test_custom_detector_geometry_hooks_keep_baseline(likelihood_pair, hook):
    baseline, _ = likelihood_pair
    candidate = copy.copy(baseline)
    candidate.detectors = copy.deepcopy(baseline.detectors)
    detector = candidate.detectors[0]
    original = getattr(detector, hook)
    setattr(detector, hook, lambda *args: original(*args))
    configure_xg_evaluation(candidate, baseline.waveform)
    assert candidate._xg_fast_evaluator is None
    assert (
        "standard time-dependent" in candidate.evaluation_diagnostics["fallback_reason"]
    )


@pytest.mark.parametrize("override", ["instance", "subclass"])
def test_custom_rigid_shift_keeps_hook_in_per_channel_evaluator(
    likelihood_pair, override
):
    baseline, _ = likelihood_pair
    candidate = copy.copy(baseline)

    class CustomShift(HeterodynedTransientLikelihoodFD):
        def _rigid_time_shift(self, detector, params):
            return jnp.asarray(1.0)

    if override == "instance":
        candidate._rigid_time_shift = lambda detector, params: jnp.asarray(1.0)
    else:
        candidate.__class__ = CustomShift
    configure_xg_evaluation(candidate, baseline.waveform)
    assert type(candidate._xg_fast_evaluator) is FastXGEvaluator
    assert (
        candidate.evaluation_diagnostics["network_fallback_reason"]
        == "custom rigid-time-shift hook"
    )
    # A hook outside the anchor bank must still invalidate this proposal.
    # The network kernel's standard geometric shift would accept it instead.
    assert np.isnan(jax.jit(candidate.evaluate)(P))


def test_rebinned_fast_evaluator_rebuilds_callbacks_and_preserves_prefix(
    likelihood_pair,
):
    from jimgw.core.single_event.heterodyne_rebin import rebin_likelihood

    baseline, fine = likelihood_pair
    # Keep each other edge, the CE band edge, and the inclusive final endpoint.
    edges = np.unique(np.r_[np.asarray(fine.freq_grid_edges)[::2], 5.0, 128.0])
    digest = fine._bin_edges_sha256(
        edges,
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=fine.phasor_time_anchors,
        reference_projection="carrier",
    )
    plan = _QualificationXGPlan(digest, _XG_QUALIFICATION_PLAN_AUTHORITY)
    coarse = rebin_likelihood(fine, edges, xg_plan=plan)
    control = rebin_likelihood(baseline, edges, xg_plan=plan)
    assert coarse._xg_fast_evaluator.likelihood is coarse
    assert fine._xg_fast_evaluator.likelihood is fine
    assert coarse.waveform is not fine.waveform
    np.testing.assert_array_equal(
        coarse.waveform.frequency_prefix, fine.freq_grid_node_flat[:2]
    )
    direct = coarse.evaluate(P)
    np.testing.assert_allclose(direct, control.evaluate(P), rtol=0, atol=5e-4)
    cache = coarse.generate_waveform(P)
    np.testing.assert_allclose(
        coarse.evaluate_from_waveform(P, cache), direct, rtol=0, atol=1e-7
    )
    summary = coarse.build_extrinsic_summary(P, cache)
    np.testing.assert_allclose(
        evaluate_extrinsic_summary(coarse, P, summary), direct, rtol=0, atol=1e-7
    )

    # Recreate the selected coarse grid exactly as a saved configuration does.
    # Its native moment pass still uses each detector's native frequency prefix.
    # The saved node prefix governs only the node reference and proposals.
    for mode, rebinned in (("auto", coarse), ("baseline", control)):
        restored = HeterodynedTransientLikelihoodFD(
            detectors=fine.detectors,
            waveform=fine._baseline_waveform,
            reference_waveform=fine._reference_waveform,
            f_min={d.name: 5.0 if d.name == "CE" else 2.0 for d in fine.detectors},
            f_max=128.0,
            trigger_time=GPS,
            n_bins=len(edges) - 1,
            reference_parameters=P,
            interpolation_order=8,
            phasor_moment_order=16,
            phasor_time_anchors=fine.phasor_time_anchors,
            reference_projection="carrier",
            frequency_bin_edges=edges,
            reference_chunk_size=128,
            phase_marginalization=fine.phase_marginalization,
            xg_plan=plan,
            xg_evaluation_mode=mode,
            node_frequency_prefix=list(map(float, fine.freq_grid_node_flat[:2])),
        )
        assert restored.node_frequency_prefix == tuple(
            map(float, fine.freq_grid_node_flat[:2])
        )
        for detector in fine.detectors:
            np.testing.assert_array_equal(
                restored.waveform_node_ref[detector.name],
                rebinned.waveform_node_ref[detector.name],
            )
        for point in (
            P,
            {**P, "M_c": 200.0, "lambda_1": 0.0, "lambda_2": 0.0, "d_L": 2000.0},
        ):
            reference_cache = rebinned.generate_waveform(point)
            restored_cache = restored.generate_waveform(point)
            for a, b in zip(
                jax.tree.leaves(restored_cache),
                jax.tree.leaves(reference_cache),
                strict=True,
            ):
                np.testing.assert_array_equal(a, b)
            expected = rebinned.evaluate_from_waveform(point, reference_cache)
            actual = restored.evaluate_from_waveform(point, restored_cache)
            assert np.isfinite(actual)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-7)


@pytest.mark.parametrize(
    "prefix",
    [
        [2.0],
        [2.0, 2.0],
        [3.0, 2.0],
        [0.0, 1.0],
        [-2.0, 1.0],
        [2.0, np.inf],
        [np.nan, 3.0],
        [[2.0, 3.0]],
        "invalid",
    ],
)
def test_invalid_node_frequency_prefix_rejected_before_construction(prefix):
    with pytest.raises(ValueError, match="node_frequency_prefix"):
        HeterodynedTransientLikelihoodFD(
            detectors=[], waveform=None, node_frequency_prefix=prefix
        )


@pytest.mark.parametrize("eta", [0.25, 2 / 9])
def test_fixed_basis_waveform_preserves_stock_and_prefix_cutoff(eta):
    source = RippleIMRPhenomD_NRTidalv2(f_ref=20.0)
    stock = DominantModeTimeCachedWaveform(source)
    prefix = jnp.array([2.0, 2.125])
    grid = jnp.geomspace(2.0, 2048.0, 129)
    fast = XGBasisCachedWaveform(source, frequency_prefix=prefix)
    # High mass puts the source's cutoff inside the grid, exercising both
    # waveform amplitude and phase cutoff conventions under a changed spacing.
    for mass in (P["M_c"], 30.0):
        p = {**P, "eta": eta, "M_c": mass}
        expected = jax.tree.map(lambda value: value[2:], stock(jnp.r_[prefix, grid], p))
        actual = fast(grid, p)
        for mode in ("p", "c"):
            scale = float(jnp.max(abs(expected[mode])))
            np.testing.assert_allclose(
                actual[mode], expected[mode], rtol=2e-8, atol=scale * 1e-10
            )
        np.testing.assert_allclose(
            actual["__tau__"], expected["__tau__"], rtol=2e-14, atol=1e-9
        )


def test_waveform_version_guard_rejects_unvalidated_backend(monkeypatch):
    from jimgw.core.single_event import xg_waveform

    source = RippleIMRPhenomD_NRTidalv2()
    monkeypatch.setattr(xg_waveform, "version", lambda name: "0.4.0")
    assert not xg_waveform.supports_source(source)
    with pytest.raises(ValueError, match="Ripple 0.3.0"):
        XGBasisCachedWaveform(source)


@pytest.mark.parametrize("moment_order", [1, 3, 16])
def test_even_odd_dresser_retains_every_finite_polynomial_term(moment_order):
    rng = np.random.default_rng(121)
    order, count = 3, 7
    anchors = np.array([-0.2, 0.0, 0.2])
    data = rng.normal(size=(3, order + moment_order + 1, count)) + 1j * rng.normal(
        size=(3, order + moment_order + 1, count)
    )
    centres, widths = np.linspace(2, 100, count), np.linspace(0.01, 2, count)
    likelihood = SimpleNamespace(
        interpolation_order=order,
        phasor_moment_order=moment_order,
        phasor_time_anchors=anchors,
        phasor_data_moments={"ET1": jnp.asarray(data)},
        freq_grid_centres=jnp.asarray(centres),
        freq_grid_half_widths=jnp.asarray(widths),
        summary_moments={"ET1": (None, jnp.ones((7, count)))},
        _rigid_time_shift=lambda detector, params: params["dt"],
    )
    dress = make_dresser(likelihood)
    detector = SimpleNamespace(name="ET1")
    dt = 0.031
    expected = np.stack(
        [
            sum(
                data[1, k + m] * (2j * np.pi * widths * dt) ** m / math.factorial(m)
                for m in range(moment_order + 1)
            )
            for k in range(order + 1)
        ]
    ) * np.exp(2j * np.pi * centres * dt)
    actual, _ = jax.jit(lambda shift: dress(detector, {"dt": shift}))(jnp.asarray(dt))
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=5e-14)
    outside, _ = dress(detector, {"dt": 0.201})
    assert np.all(np.isnan(outside))
