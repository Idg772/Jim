"""Batched network algebra on noisy, physical-geometry 2--2048 Hz inputs.

The bounded 2 Hz native quadrature does not resolve a 21-hour injection. It
retains the actual long waveform clock, CE-A/Sardinia geometry, finite arms,
orbital response, K8/M16 and all 21 time anchors. Deterministic positive
analytic spectra are identical locally and in CI. These tests compare
evaluators of the same moments, not heterodyne accuracy or design sensitivity.
"""

import copy
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_CE_A, get_ET_Sardinia
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.heterodyne_extrinsics import evaluate_extrinsic_summary
from jimgw.core.single_event.heterodyne_rebin import rebin_likelihood
from jimgw.core.single_event.likelihood import (
    _XG_QUALIFICATION_PLAN_AUTHORITY,
    HeterodynedTransientLikelihoodFD,
    _QualificationXGPlan,
)
from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.xg_evaluation import (
    FastXGEvaluator,
    configure_xg_evaluation,
)
from jimgw.core.single_event.xg_network_evaluation import BatchedXGEvaluator

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[4]
GPS = 1_300_000_000.0
DF = 2.0
ANCHORS = np.linspace(-0.2, 0.2, 21)
PARAMETERS = {
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


def _plan(edges):
    digest = HeterodynedTransientLikelihoodFD._bin_edges_sha256(
        edges,
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=ANCHORS,
        reference_projection="carrier",
    )
    return _QualificationXGPlan(digest, _XG_QUALIFICATION_PLAN_AUTHORITY)


def _options(detectors, waveform, edges, phase_marginalization):
    return {
        "detectors": detectors,
        "waveform": waveform,
        "f_min": {d.name: 5.0 if d.name == "CE" else 2.0 for d in detectors},
        "f_max": 2048.0,
        "trigger_time": GPS,
        "n_bins": len(edges) - 1,
        "reference_parameters": PARAMETERS,
        "interpolation_order": 8,
        "phasor_moment_order": 16,
        "phasor_time_anchors": ANCHORS,
        "reference_projection": "carrier",
        "frequency_bin_edges": edges,
        "reference_chunk_size": 256,
        "phase_marginalization": phase_marginalization,
        "xg_plan": _plan(edges),
    }


def _read_psd_tables(root):
    """Return analytic spectra; keep the root argument for shared test callers."""
    tables = {}
    for name in ("CE", "ET"):
        # Smooth numerical inputs, not a model of a published sensitivity curve.
        frequency = np.geomspace(5.0 if name == "CE" else 1.0, 4096.0, 257)
        if name == "CE":
            psd = 2e-50 * (1 + (12 / frequency) ** 8 + (frequency / 700) ** 2)
        else:
            psd = 8e-50 * (1 + (10 / frequency) ** 10 + (frequency / 500) ** 2)
        tables[name] = np.column_stack((frequency, psd))
    return tables


@pytest.fixture(scope="module")
def native_network():
    return _make_native_network(_read_psd_tables(ROOT))


def _make_native_network(tables):
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD_NRTidalv2(f_ref=20.0))
    frequencies = np.arange(1025) * DF
    positive = jnp.asarray(frequencies[1:])
    p = {**PARAMETERS, "trigger_time": GPS, "gmst": greenwich_mean_sidereal_time(GPS)}
    sky = waveform(positive, p)
    assert 77_000 < float(sky["__tau__"][0]) < 78_000
    rng = np.random.default_rng(901731)
    detectors = [get_CE_A(), *get_ET_Sardinia()]
    for index, detector in enumerate(detectors):
        table = tables["CE" if detector.name == "CE" else "ET"]
        psd = np.exp(
            np.interp(
                np.log(np.maximum(frequencies, table[0, 0])),
                np.log(table[:, 0]),
                np.log(table[:, 1]),
            )
        )
        buffer = np.zeros(frequencies.size, dtype=np.complex128)
        data = Data.from_host_fd(
            buffer, delta_t=1 / 4096, start_time=GPS - 131070 + index / 8
        )
        detector.set_data(data)
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
        buffer[1:] = np.asarray(detector.fd_response(positive, sky, p))
        buffer += np.sqrt(psd / (4 * DF)) * (
            rng.normal(size=buffer.size) + 1j * rng.normal(size=buffer.size)
        )
        detector.set_frequency_bounds(5.0 if detector.name == "CE" else 2.0, 2048.0)
    return detectors, waveform


@pytest.fixture(scope="module", params=[False, True], ids=["fixed-phase", "phase-marg"])
def likelihood_pair(native_network, request):
    detectors, waveform = native_network
    # No edge at 5 Hz: the 4--8 Hz bin must retain its partial CE contribution.
    edges = np.r_[2.0, 4.0, np.geomspace(8.0, 2048.0, 41)]
    likelihood = HeterodynedTransientLikelihoodFD(
        **_options(detectors, waveform, edges, request.param)
    )
    old = copy.copy(likelihood)
    old._xg_fast_evaluator = FastXGEvaluator(old, waveform)
    new = _batched_clone(likelihood)
    return old, new


def _batched_clone(likelihood, **flags):
    clone = copy.copy(likelihood)
    clone._xg_fast_evaluator = BatchedXGEvaluator(
        clone, clone._reference_waveform, **flags
    )
    return clone


def _bank(points):
    return jax.tree.map(lambda *values: jnp.asarray(values), *points)


def _points():
    return [
        PARAMETERS,
        {
            **PARAMETERS,
            "M_c": 1.18,
            "eta": 2 / 9,
            "s1_z": -0.05,
            "s2_z": 0.05,
            "lambda_1": 0.0,
            "lambda_2": 1000.0,
        },
        {
            **PARAMETERS,
            "M_c": 1.1807,
            "eta": 0.25,
            "s1_z": 0.05,
            "s2_z": -0.05,
            "lambda_1": 1000.0,
            "lambda_2": 0.0,
        },
        {**PARAMETERS, "ra": 0.0, "dec": -np.pi / 2, "t_c": -0.1},
        {**PARAMETERS, "ra": 2 * np.pi, "dec": np.pi / 2, "t_c": 0.075},
        {**PARAMETERS, "psi": 0.0, "iota": 0.0, "d_L": 1.0},
        {**PARAMETERS, "psi": np.pi, "iota": np.pi, "d_L": 1.0},
        {**PARAMETERS, "psi": 0.4, "iota": np.pi / 2, "d_L": 1000.0, "phase_c": 0.7},
    ]


def test_analytic_psds_preserve_bright_network(tmp_path, record_property):
    # The historical path argument cannot change the numerical test inputs.
    tables = _read_psd_tables(tmp_path)
    repeated = _read_psd_tables(ROOT)
    for name, table in tables.items():
        np.testing.assert_array_equal(table, repeated[name])
        assert np.all(np.isfinite(table)) and np.all(table > 0)
    detectors, waveform = _make_native_network(tables)
    edges = np.r_[2.0, 4.0, np.geomspace(8.0, 2048.0, 41)]
    likelihood = HeterodynedTransientLikelihoodFD(
        **_options(detectors, waveform, edges, True)
    )
    old = copy.copy(likelihood)
    old._xg_fast_evaluator = FastXGEvaluator(old, waveform)
    new = _batched_clone(likelihood)
    bank = _bank([PARAMETERS, {**PARAMETERS, "d_L": 1.0, "iota": 0.0, "psi": 0.0}])
    expected = jax.jit(jax.vmap(old.evaluate))(bank)
    actual = jax.jit(jax.vmap(new.evaluate))(bank)
    assert np.all(np.isfinite(actual)) and float(abs(expected[1])) > 1e8
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)
    caches = jax.jit(jax.vmap(new.generate_waveform))(bank)
    cached = jax.jit(jax.vmap(new.evaluate_from_waveform))(bank, caches)
    np.testing.assert_allclose(cached, expected, rtol=0, atol=1e-3)
    summaries = jax.jit(jax.vmap(new.build_extrinsic_summary))(bank, caches)
    scalar = jax.jit(jax.vmap(lambda p, s: evaluate_extrinsic_summary(new, p, s)))(
        bank, summaries
    )
    np.testing.assert_allclose(scalar, expected, rtol=0, atol=1e-3)
    record_property(
        "psd_scope", "deterministic analytic fallback; physical CE-A/ET geometry"
    )
    record_property("maximum_direct_error_nats", float(np.max(abs(actual - expected))))
    record_property("maximum_summary_error_nats", float(np.max(abs(scalar - expected))))


def test_full_band_direct_source_cache_and_scalar_parity(
    likelihood_pair, record_property
):
    old, new = likelihood_pair
    bank = _bank(_points())
    expected = np.asarray(jax.jit(jax.vmap(old.evaluate))(bank))
    actual = np.asarray(jax.jit(jax.vmap(new.evaluate))(bank))
    assert np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))
    record_property("maximum_direct_error_nats", float(np.max(abs(actual - expected))))
    # A fixed absolute bound stays meaningful for the high-SNR distance=1 cases.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)
    caches = jax.jit(jax.vmap(new.generate_waveform))(bank)
    cached = jax.jit(jax.vmap(new.evaluate_from_waveform))(bank, caches)
    np.testing.assert_allclose(cached, actual, rtol=0, atol=1e-3)
    old_summaries = jax.jit(jax.vmap(old.build_extrinsic_summary))(bank, caches)
    summaries = jax.jit(jax.vmap(new.build_extrinsic_summary))(bank, caches)
    scalar = jax.jit(jax.vmap(lambda p, s: evaluate_extrinsic_summary(new, p, s)))(
        bank, summaries
    )
    old_scalar = jax.jit(jax.vmap(lambda p, s: evaluate_extrinsic_summary(old, p, s)))(
        bank, old_summaries
    )
    record_property("maximum_cached_error_nats", float(np.max(abs(cached - actual))))
    record_property(
        "maximum_summary_error_nats", float(np.max(abs(scalar - old_scalar)))
    )
    record_property(
        "maximum_summary_direct_error_nats", float(np.max(abs(scalar - actual)))
    )
    np.testing.assert_allclose(scalar, old_scalar, rtol=0, atol=1e-3)
    np.testing.assert_allclose(scalar, actual, rtol=0, atol=1e-3)


@pytest.mark.parametrize("phase_marginalization", [False, True])
@pytest.mark.parametrize("channels", [1, 4])
def test_chebyshev_all_evaluation_routes_preserve_noisy_likelihood(
    native_network, phase_marginalization, channels, record_property
):
    detectors, waveform = native_network
    detectors = detectors[:channels]
    # On this bounded 2-Hz native lattice the first CE sample above5Hz is6Hz.
    edges = np.r_[
        ([6.0] if channels == 1 else [2.0, 4.0]), np.geomspace(8.0, 2048.0, 41)
    ]
    options = _options(detectors, waveform, edges, phase_marginalization)
    digest = HeterodynedTransientLikelihoodFD._bin_edges_sha256(
        edges,
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=ANCHORS,
        reference_projection="carrier",
        phasor_approximation="chebyshev",
    )
    options.update(
        phasor_approximation="chebyshev",
        xg_evaluation_mode="baseline",
        xg_plan=_QualificationXGPlan(digest, _XG_QUALIFICATION_PLAN_AUTHORITY),
    )
    baseline = HeterodynedTransientLikelihoodFD(**options)
    automatic = copy.copy(baseline)
    configure_xg_evaluation(automatic, waveform)
    expected_type = FastXGEvaluator if channels == 1 else BatchedXGEvaluator
    assert isinstance(automatic._xg_fast_evaluator, expected_type)
    assert (
        automatic.evaluation_diagnostics["phasor_polynomial"]["approximation"]
        == "chebyshev"
    )
    serial = copy.copy(automatic)
    serial._xg_fast_evaluator = FastXGEvaluator(serial, waveform)
    batched = copy.copy(automatic)
    batched._xg_fast_evaluator = BatchedXGEvaluator(batched, waveform)
    # The preserved baseline divides two phases with ~1e9-radian arguments.
    # Cancel the common data epoch before that division in this independent
    # comparison; the native moments and physical ratio are unchanged.
    stable = copy.copy(baseline)
    stable.detectors = [copy.copy(d) for d in detectors]
    reference = waveform(baseline.freq_grid_node_flat, baseline.reference_parameters)
    stable.waveform_node_ref = {}
    for detector, original in zip(stable.detectors, detectors, strict=True):
        detector.fd_response = functools.partial(
            original.fd_response, include_data_epoch=False
        )
        stable.waveform_node_ref[detector.name] = detector.fd_response(
            stable.freq_grid_node_flat,
            reference,
            stable.reference_parameters,
            apply_antenna=False,
        ).reshape(9, stable.n_bins)
    bank = _bank(
        [
            _points()[1],
            {**PARAMETERS, "t_c": PARAMETERS["t_c"] + 0.01},
            _points()[3],
            _points()[5],
        ]
    )
    original_expected = jax.jit(jax.vmap(baseline.evaluate))(bank)
    expected = jax.jit(jax.vmap(stable.evaluate))(bank)
    assert np.all(np.isfinite(expected))
    maxima = []
    direct_values = []
    for candidate in (baseline, stable, serial, batched):
        cache = jax.jit(jax.vmap(candidate.generate_waveform))(bank)
        direct = jax.jit(jax.vmap(candidate.evaluate))(bank)
        cached = jax.jit(jax.vmap(candidate.evaluate_from_waveform))(bank, cache)
        summary = jax.jit(jax.vmap(candidate.build_extrinsic_summary))(bank, cache)
        scalar = jax.jit(
            jax.vmap(
                lambda p, s, likelihood=candidate: evaluate_extrinsic_summary(
                    likelihood, p, s
                )
            )
        )(bank, summary)
        target = original_expected if candidate is baseline else expected
        for route, actual in zip(
            ("direct", "cached", "scalar"), (direct, cached, scalar), strict=True
        ):
            np.testing.assert_allclose(
                actual,
                target,
                rtol=0,
                atol=1e-3,
                err_msg=f"{type(candidate._xg_fast_evaluator).__name__}:{route}",
            )
            maxima.append(float(np.max(abs(actual - target))))
        direct_values.append(direct)
    # Keep the original baseline in the regression: adding the new polynomial
    # must not materially increase its preexisting absolute-epoch roundoff.
    taylor_baseline = copy.copy(baseline)
    taylor_baseline.phasor_approximation = "taylor"
    taylor_fast = copy.copy(serial)
    taylor_fast.phasor_approximation = "taylor"
    taylor_fast._xg_fast_evaluator = FastXGEvaluator(taylor_fast, waveform)
    old_difference = jax.jit(jax.vmap(taylor_fast.evaluate))(bank) - jax.jit(
        jax.vmap(taylor_baseline.evaluate)
    )(bank)
    new_difference = direct_values[2] - original_expected
    np.testing.assert_allclose(new_difference, old_difference, rtol=0, atol=1e-5)
    record_property("maximum_chebyshev_route_error_nats", max(maxima))
    record_property(
        "maximum_phasor_change_in_epoch_disagreement_nats",
        float(np.max(abs(new_difference - old_difference))),
    )


def test_cache_and_summary_reuse_preserve_only_supported_changes(likelihood_pair):
    old, new = likelihood_pair
    cache = new.generate_waveform(PARAMETERS)
    summary = new.build_extrinsic_summary(PARAMETERS, cache)
    extrinsic = {**PARAMETERS, "psi": 0.31, "iota": 0.61, "d_L": 1.0}
    expected = old.evaluate(extrinsic)
    np.testing.assert_allclose(
        new.evaluate_from_waveform(extrinsic, cache), expected, rtol=0, atol=1e-3
    )
    np.testing.assert_allclose(
        evaluate_extrinsic_summary(new, extrinsic, summary), expected, rtol=0, atol=1e-3
    )
    moved = {**PARAMETERS, "ra": 1.7, "dec": -0.6, "t_c": -0.03}
    np.testing.assert_allclose(
        new.evaluate_from_waveform(moved, cache), old.evaluate(moved), rtol=0, atol=1e-3
    )
    assert np.isnan(evaluate_extrinsic_summary(new, moved, summary))
    shifts = [
        float(new._rigid_time_shift(d, new._prepare_parameters(moved)))
        for d in new.detectors
    ]
    assert np.ptp(shifts[1:]) > 1e-6


@pytest.mark.parametrize("norm_method", ["gram", "triangular"])
@pytest.mark.parametrize("reuse_opposite_arms", [False, True])
def test_arm_sharing_and_norm_ablations_preserve_likelihood(
    likelihood_pair, norm_method, reuse_opposite_arms
):
    old, new = likelihood_pair
    candidate = _batched_clone(
        new, norm_method=norm_method, reuse_opposite_arms=reuse_opposite_arms
    )
    points = _bank([_points()[1], _points()[5], _points()[7]])
    actual = jax.jit(jax.vmap(candidate.evaluate))(points)
    expected = jax.jit(jax.vmap(old.evaluate))(points)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)


def test_exact_zero_bin_trimming_retains_partial_detector_band(likelihood_pair):
    old, new = likelihood_pair
    evaluator = new._xg_fast_evaluator
    retained = {}
    for group in evaluator.groups:
        for index in np.asarray(group.detector_indices):
            retained[int(index)] = np.asarray(group.bin_indices)
    for index, detector in enumerate(new.detectors):
        a = np.asarray(new.phasor_data_moments[detector.name])
        b = np.asarray(new.summary_moments[detector.name][1])
        expected = np.flatnonzero(np.any(a != 0, axis=(0, 1)) | np.any(b != 0, axis=0))
        np.testing.assert_array_equal(retained[index], expected)
    assert 0 not in retained[0]  # CE has no native samples in 2--4 Hz.
    assert 1 in retained[0]  # Its 6 Hz sample lies in the partial 4--8 Hz bin.
    assert 0 in retained[1]  # ET still contributes at 2 Hz.
    full = _batched_clone(new, trim_zero_bins=False)
    for group in full._xg_fast_evaluator.groups:
        np.testing.assert_array_equal(group.bin_indices, np.arange(new.n_bins))
    bank = _bank([PARAMETERS, _points()[5]])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(new.evaluate))(bank),
        jax.jit(jax.vmap(full.evaluate))(bank),
        rtol=0,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(full.evaluate))(bank),
        jax.jit(jax.vmap(old.evaluate))(bank),
        rtol=0,
        atol=1e-3,
    )


@pytest.mark.parametrize("moment_kind", ["anchored-data", "norm"])
def test_trimming_does_not_threshold_tiny_nonzero_moments(likelihood_pair, moment_kind):
    _, original = likelihood_pair
    changed = copy.copy(original)
    if moment_kind == "anchored-data":
        changed.phasor_data_moments = dict(original.phasor_data_moments)
        data = np.asarray(original.phasor_data_moments["CE"]).copy()
        data[-1, -1, 0] = 1e-300j
        changed.phasor_data_moments["CE"] = jnp.asarray(data)
    else:
        changed.summary_moments = dict(original.summary_moments)
        a, b = original.summary_moments["CE"]
        norm = np.asarray(b).copy()
        norm[-1, 0] = 1e-300
        changed.summary_moments["CE"] = (a, jnp.asarray(norm))
    changed = _batched_clone(changed)
    ce_group = next(
        group
        for group in changed._xg_fast_evaluator.groups
        if 0 in np.asarray(group.detector_indices)
    )
    assert 0 in np.asarray(ce_group.bin_indices)


def test_anchor_boundaries_and_orbit_validity_propagate(likelihood_pair):
    old, new = likelihood_pair
    # Widen only the orbit time interval to isolate both anchor endpoints.
    widened = copy.copy(old)
    widened.detectors = copy.deepcopy(old.detectors)
    for detector in widened.detectors:
        detector.orbital_validity_s = (-131072.125, 1.0)
    widened._xg_fast_evaluator = FastXGEvaluator(widened, old._reference_waveform)
    batched = _batched_clone(widened)
    shifts = [-0.2, -0.2 + 1e-12, 0.01 - 1e-12, 0.01 + 1e-12, 0.2 - 1e-12, 0.2]
    bank = _bank([{**PARAMETERS, "t_c": PARAMETERS["t_c"] + dt} for dt in shifts])
    expected = jax.jit(jax.vmap(widened.evaluate))(bank)
    actual = jax.jit(jax.vmap(batched.evaluate))(bank)
    assert np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)
    for dt in (-0.2 - 1e-10, 0.2 + 1e-10):
        assert np.isnan(batched.evaluate({**PARAMETERS, "t_c": PARAMETERS["t_c"] + dt}))
    invalid = {**PARAMETERS, "M_c": 0.7}  # 2 Hz clock precedes orbital validity.
    assert np.isnan(old.evaluate(invalid))
    assert np.isnan(new.evaluate(invalid))
    cache = new.generate_waveform(invalid)
    assert np.isnan(new.evaluate_from_waveform(invalid, cache))
    summary = new.build_extrinsic_summary(invalid, cache)
    assert np.isnan(evaluate_extrinsic_summary(new, invalid, summary))


def test_zero_weight_detector_still_enforces_full_orbital_contract(likelihood_pair):
    old, _ = likelihood_pair
    changed = copy.copy(old)
    changed.detectors = copy.deepcopy(old.detectors)
    # The reference is valid; a longer proposed source clock invalidates only
    # CE. Its entire zero-weight bank must not suppress that contract check.
    changed.detectors[0].orbital_validity_s = (-90_000.0, 0.125)
    changed.phasor_data_moments = dict(old.phasor_data_moments)
    changed.phasor_data_moments["CE"] = jnp.zeros_like(old.phasor_data_moments["CE"])
    changed.summary_moments = dict(old.summary_moments)
    changed.summary_moments["CE"] = tuple(
        jnp.zeros_like(value) for value in old.summary_moments["CE"]
    )
    changed._xg_fast_evaluator = FastXGEvaluator(changed, old._reference_waveform)
    candidate = _batched_clone(changed)
    assert all(
        0 not in np.asarray(group.detector_indices)
        for group in candidate._xg_fast_evaluator.groups
    )
    invalid = {**PARAMETERS, "M_c": 1.0}
    assert np.isfinite(candidate.evaluate(PARAMETERS))
    assert np.isnan(changed.evaluate(invalid))
    assert np.isnan(candidate.evaluate(invalid))


def test_rebin_and_restoration_rebind_selected_evaluator(likelihood_pair):
    old, fine = likelihood_pair
    edges = np.unique(np.r_[np.asarray(fine.freq_grid_edges)[::2], 2048.0])
    coarse = rebin_likelihood(fine, edges, xg_plan=_plan(edges))
    assert isinstance(coarse._xg_fast_evaluator, BatchedXGEvaluator)
    assert coarse._xg_fast_evaluator.likelihood is coarse
    assert fine._xg_fast_evaluator.likelihood is fine
    assert coarse.waveform is not fine.waveform
    np.testing.assert_array_equal(
        coarse.waveform.frequency_prefix, fine.freq_grid_node_flat[:2]
    )
    control = copy.copy(coarse)
    control._xg_fast_evaluator = FastXGEvaluator(control, old._reference_waveform)
    restored = HeterodynedTransientLikelihoodFD(
        **_options(
            fine.detectors, fine._baseline_waveform, edges, fine.phase_marginalization
        ),
        node_frequency_prefix=list(map(float, fine.freq_grid_node_flat[:2])),
    )
    assert isinstance(restored._xg_fast_evaluator, BatchedXGEvaluator)
    assert restored._xg_fast_evaluator.likelihood is restored
    for candidate in (coarse, restored):
        bank = _bank([PARAMETERS, {**PARAMETERS, "d_L": 1.0, "iota": 0.4}])
        actual = jax.jit(jax.vmap(candidate.evaluate))(bank)
        np.testing.assert_allclose(
            actual, jax.jit(jax.vmap(control.evaluate))(bank), rtol=0, atol=1e-3
        )
        caches = jax.jit(jax.vmap(candidate.generate_waveform))(bank)
        np.testing.assert_allclose(
            jax.jit(jax.vmap(candidate.evaluate_from_waveform))(bank, caches),
            actual,
            rtol=0,
            atol=1e-3,
        )
        summaries = jax.jit(jax.vmap(candidate.build_extrinsic_summary))(bank, caches)
        scalar = jax.jit(
            jax.vmap(lambda p, s, lk=candidate: evaluate_extrinsic_summary(lk, p, s))
        )(bank, summaries)
        np.testing.assert_allclose(scalar, actual, rtol=0, atol=1e-3)
