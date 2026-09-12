"""Nested moment coarsening against independent noisy native sums."""

from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event.heterodyne_rebin import coarsen_moments, rebin_likelihood
from jimgw.core.single_event.likelihood import (
    _XG_PLAN_AUTHORITY,
    _XG_QUALIFICATION_PLAN_AUTHORITY,
    HeterodynedTransientLikelihoodFD,
    _QualificationXGPlan,
    _VerifiedXGPlan,
)

jax.config.update("jax_enable_x64", True)

FINE = np.array([2.0, 2.125, 2.7, 3.0, 5.0, 5.125, 7.75, 8.0])
COARSE = FINE[[0, 3, 4, 7]]
ANCHORS = np.array([-0.2, -0.03, 0.0, 0.17, 0.2])


def direct(f, weights, edges, degree):
    """Independent boolean-bin explicit-power oracle, including final edge."""
    out = np.empty(
        (*weights.shape[:-1], degree + 1, len(edges) - 1), dtype=weights.dtype
    )
    for b, (lo, hi) in enumerate(pairwise(edges)):
        mask = (f >= lo) & ((f <= hi) if b == len(edges) - 2 else (f < hi))
        u = (2 * f[mask] - lo - hi) / (hi - lo)
        for k in range(degree + 1):
            out[..., k, b] = np.sum(weights[..., mask] * u**k, axis=-1)
    return out


@pytest.mark.parametrize("degree", [0, 6, 16, 24])
def test_absolute_anchor_moments_match_direct_noisy_sums(degree):
    rng = np.random.default_rng(391)
    f = np.sort(
        np.r_[np.arange(2.0, 8.0, 0.25), FINE, np.nextafter(FINE[1:-1], -np.inf), 8.0]
    )
    data = rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))
    reference = (1 + 0.1 * f) * np.exp(0.013j * f**2)
    psd = 1 + rng.uniform(size=len(f))
    product = data * reference.conj() / psd
    anchored = np.exp(2j * np.pi * ANCHORS[:, None] * f) * product
    weights = np.concatenate((product[None, :], anchored))
    fine = direct(f, weights, FINE, degree)
    expected = direct(f, weights, COARSE, degree)
    actual = coarsen_moments(FINE, COARSE, fine)
    # Absolute L1-scaled tolerance remains meaningful for cancelled noise sums.
    budget = 128 * np.finfo(float).eps * np.sum(np.abs(product))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=budget)
    norm = abs(reference) ** 2 / psd
    np.testing.assert_allclose(
        coarsen_moments(FINE, COARSE, direct(f, norm, FINE, degree)),
        direct(f, norm, COARSE, degree),
        rtol=2e-14,
        atol=2e-13,
    )


def test_empty_bins_singleton_tail_and_zero_detector_low_band():
    f = np.array([5.0, 5.5, 6.0, 8.0])
    weights = np.array([1 + 2j, -2 + 3j, 0.2j, 19 - 7j])
    fine = direct(f, weights, FINE, 8)
    actual = coarsen_moments(FINE, COARSE, fine)
    np.testing.assert_array_equal(actual[:, :2], 0)
    np.testing.assert_allclose(
        actual, direct(f, weights, COARSE, 8), rtol=0, atol=2e-14
    )
    assert actual[0].sum() == weights.sum()


def test_multistep_and_identity_coarsening():
    rng = np.random.default_rng(81)
    fine = rng.normal(size=(2, 3, 25, len(FINE) - 1))
    middle = coarsen_moments(FINE, COARSE, fine)
    np.testing.assert_allclose(
        coarsen_moments(COARSE, COARSE[[0, -1]], middle),
        coarsen_moments(FINE, COARSE[[0, -1]], fine),
        rtol=0,
        atol=2e-14,
    )
    unchanged = coarsen_moments(FINE, FINE, fine)
    np.testing.assert_array_equal(unchanged, fine)
    assert not np.shares_memory(unchanged, fine)


@pytest.mark.parametrize(
    "edges",
    [
        [2.0, 4.0, 8.0],
        [2.125, 8.0],
        [2.0, 7.75],
        [2.0, 5.0, 5.0, 8.0],
        [2.0, np.nan, 8.0],
    ],
)
def test_invalid_coarse_edges_are_rejected(edges):
    with pytest.raises(ValueError):
        coarsen_moments(FINE, edges, np.zeros((3, len(FINE) - 1)))


@pytest.mark.parametrize(
    "values", [np.zeros(7), np.zeros((0, 7)), np.zeros((3, 8)), np.full((3, 7), np.nan)]
)
def test_invalid_moment_banks_are_rejected(values):
    with pytest.raises(ValueError):
        coarsen_moments(FINE, COARSE, values)


class NoNativeAccess:
    def __getattribute__(self, name):
        raise AssertionError(f"native array was accessed: {name}")


class ToyWaveform:
    def __init__(self, spacing_sensitive=False):
        self.calls = []
        self.spacing_sensitive = spacing_sensitive
        self.frequency_grid_independent = not spacing_sensitive

    def __call__(self, f, p):
        self.calls.append(np.asarray(f).copy())
        scale = 1 + f[1] - f[0] if self.spacing_sensitive else 1.0
        h = scale * p["amplitude"] * (1 + 0.01 * f) * jnp.exp(0.013j * f**2)
        return {"p": h, "c": -0.4j * h}


class ToyDetector:
    time_dependent_response = False
    finite_arm_response = False

    def __init__(self, name, gain):
        self.name, self.gain = name, gain
        self.sliced_frequencies = self.sliced_fd_data = self.sliced_psd = (
            NoNativeAccess()
        )

    def fd_response(self, f, sky, p, **options):
        return self.gain * sky["p"] * jnp.exp(-2j * jnp.pi * f * p["t_c"])


def tiny_likelihood(*, spacing_sensitive=False):
    # Intentionally avoid the constructor: cloning must use only retained small
    # moments/references, so all native data access is guarded by sentinels.
    fine = object.__new__(HeterodynedTransientLikelihoodFD)
    fine.interpolation_order, fine.phasor_moment_order = 3, 4
    fine.phasor_time_anchors = tuple(ANCHORS)
    fine.time_marginalization = fine.phase_marginalization = False
    fine.zero_noise_summary = None
    fine.reference_projection = "projected"
    fine.waveform = fine._reference_waveform = ToyWaveform(spacing_sensitive)
    fine.detectors = [ToyDetector("CE", 1 + 0.1j), ToyDetector("ET1", 0.7 - 0.2j)]
    fine.reference_parameters = {"amplitude": 1.3, "t_c": 0.01}
    fine.frequencies = NoNativeAccess()
    fine._phasor_reference_t_c = 0.01
    fine._phasor_reference_delay = {"CE": 0.0, "ET1": 0.0}
    fine._set_frequency_arrays(jnp.asarray(FINE))
    fine.bin_edges_sha256 = fine._bin_edges_sha256(
        FINE, interpolation_order=3, phasor_moment_order=4, phasor_time_anchors=ANCHORS
    )
    fine.summary_moments, fine.summary_data, fine.phasor_data_moments = {}, {}, {}
    fine.waveform_low_ref, fine.waveform_high_ref, fine.waveform_node_ref = {}, {}, {}
    for d in fine.detectors:
        f = np.arange(2.0, 8.125, 0.125)
        if d.name == "CE":
            f = f[f >= 5.0]
        ref = np.asarray(
            d.fd_response(
                jnp.asarray(f),
                fine.waveform(jnp.asarray(f), fine.reference_parameters),
                fine.reference_parameters,
            )
        )
        w = (np.cos(f * 3) + 1j * np.sin(f * 7)) * ref.conj()
        a, b = direct(f, w, FINE, 7), direct(f, abs(ref) ** 2, FINE, 6)
        fine.summary_moments[d.name] = (jnp.asarray(a), jnp.asarray(b))
        fine.summary_data[d.name] = jnp.asarray(np.concatenate((a, b)))
        fine.phasor_data_moments[d.name] = jnp.asarray(
            direct(f, w[None, :] * np.exp(2j * np.pi * ANCHORS[:, None] * f), FINE, 7)
        )
        for grid, dest in (
            (fine.freq_grid_low, fine.waveform_low_ref),
            (fine.freq_grid_high, fine.waveform_high_ref),
            (fine.freq_grid_node_flat, fine.waveform_node_ref),
        ):
            values = d.fd_response(
                grid,
                fine.waveform(grid, fine.reference_parameters),
                fine.reference_parameters,
            )
            dest[d.name] = (
                values.reshape(4, fine.n_bins)
                if dest is fine.waveform_node_ref
                else values
            )
    return fine


def test_clone_preserves_masks_anchors_phasor_references_without_native_access():
    fine = tiny_likelihood()
    old_data = {
        key: np.asarray(value).copy() for key, value in fine.summary_data.items()
    }
    coarse = rebin_likelihood(fine, COARSE)
    assert coarse.n_bins == 3 and fine.n_bins == 7
    assert coarse.bin_edges_sha256 != fine.bin_edges_sha256
    assert coarse.phasor_time_anchors == fine.phasor_time_anchors
    assert coarse._phasor_reference_delay == fine._phasor_reference_delay
    assert coarse._phasor_reference_t_c == fine._phasor_reference_t_c
    assert coarse.detectors is fine.detectors
    assert coarse.rebin_diagnostics["native_samples_read"] == 0
    assert coarse.rebin_diagnostics["qualification"] is False
    for name, original in old_data.items():
        np.testing.assert_array_equal(fine.summary_data[name], original)
        a, b = coarse.summary_moments[name]
        np.testing.assert_array_equal(coarse.summary_data[name], np.concatenate((a, b)))
    np.testing.assert_array_equal(coarse.summary_moments["CE"][0][:, :2], 0)
    np.testing.assert_array_equal(coarse.phasor_data_moments["CE"][:, :, :2], 0)
    np.testing.assert_array_equal(
        coarse.waveform_low_ref["ET1"],
        np.asarray(fine.waveform_low_ref["ET1"])[[0, 3, 4]],
    )
    np.testing.assert_array_equal(
        coarse.waveform_high_ref["ET1"],
        np.asarray(fine.waveform_high_ref["ET1"])[[2, 3, 6]],
    )
    assert np.isfinite(float(coarse._evaluate({"amplitude": 1.31, "t_c": 0.025})))


def test_clone_reference_preserves_fine_spacing_and_repeated_rebin_prefix():
    fine = tiny_likelihood()
    coarse = rebin_likelihood(fine, COARSE)
    call = fine.waveform.calls[-1]
    np.testing.assert_array_equal(call[:2], fine.freq_grid_node_flat[:2])
    np.testing.assert_array_equal(call[2:], coarse.freq_grid_node_flat)
    assert call[1] - call[0] != float(
        coarse.freq_grid_node_flat[1] - coarse.freq_grid_node_flat[0]
    )
    coarser = rebin_likelihood(coarse, COARSE[[0, -1]])
    np.testing.assert_array_equal(
        coarser._xg_node_frequency_prefix, fine.freq_grid_node_flat[:2]
    )
    assert coarser.node_frequency_prefix == tuple(
        float(value) for value in fine.freq_grid_node_flat[:2]
    )


def test_unknown_spacing_sensitive_source_is_rejected_before_clone_evaluation():
    fine = tiny_likelihood(spacing_sensitive=True)
    calls_before = len(fine.waveform.calls)
    with pytest.raises(ValueError, match="frequency_grid_independent"):
        rebin_likelihood(fine, COARSE)
    assert len(fine.waveform.calls) == calls_before


def test_xg_clone_requires_new_matching_qualification_capability():
    fine = tiny_likelihood()
    fine.detectors[0].finite_arm_response = True
    digest = fine._bin_edges_sha256(
        COARSE,
        interpolation_order=3,
        phasor_moment_order=4,
        phasor_time_anchors=ANCHORS,
    )
    with pytest.raises(ValueError, match="qualification plan"):
        rebin_likelihood(fine, COARSE)
    with pytest.raises(ValueError, match="qualification plan"):
        rebin_likelihood(
            fine, COARSE, xg_plan=_VerifiedXGPlan(digest, _XG_PLAN_AUTHORITY)
        )
    with pytest.raises(ValueError, match="qualification plan"):
        rebin_likelihood(
            fine,
            COARSE,
            xg_plan=_QualificationXGPlan(
                fine.bin_edges_sha256, _XG_QUALIFICATION_PLAN_AUTHORITY
            ),
        )
    cloned = rebin_likelihood(
        fine,
        COARSE,
        xg_plan=_QualificationXGPlan(digest, _XG_QUALIFICATION_PLAN_AUTHORITY),
    )
    assert cloned.bin_edges_sha256 == digest


def test_clone_rejects_stale_research_closures_and_malformed_anchor_banks():
    fine = tiny_likelihood()
    fine._polynomial_likelihood = lambda *args: 0.0
    with pytest.raises(ValueError, match="instance-bound"):
        rebin_likelihood(fine, COARSE)
    del fine._polynomial_likelihood
    fine.phasor_data_moments["CE"] = fine.phasor_data_moments["CE"][:-1]
    with pytest.raises(ValueError, match="anchors and degrees"):
        rebin_likelihood(fine, COARSE)
