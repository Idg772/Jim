"""Degree-K per-bin ratio interpolation for the heterodyned likelihood.

The classic relative-binning scheme represents the waveform ratio inside each
bin by a straight line through the two edges.  A time shift multiplies the
ratio by ``exp(-2 pi i f dt)``, whose within-bin curvature is second order in
the per-bin phase, so the linear scheme's error grows as (bins)^-2 and cannot
reach XG budgets at prior corners.  ``interpolation_order=K`` fits a degree-K
polynomial through K+1 Lobatto nodes per bin and contracts it with the
matching higher summary moments.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    TransientLikelihoodFD,
)
from jimgw.core.single_event.waveform import RippleIMRPhenomD

jax.config.update("jax_enable_x64", True)

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"
GPS = 1126259462.4
F_MIN = 20.0
F_MAX = 1024.0
N_BINS = 128
TIME_SHIFT_S = 6.0e-3  # ~0.3 rad of phasor per bin at 128 bins over the band


@pytest.fixture
def setup():
    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        ifo.set_data(
            Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
        )
        ifo.set_psd(
            PowerSpectrum.from_file(str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz"))
        )
    return ifos, RippleIMRPhenomD(f_ref=20.0)


def reference_params() -> dict[str, float]:
    return {
        "M_c": 30.0,
        "eta": 0.249,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "d_L": 400.0,
        "phase_c": 0.0,
        "t_c": 0.0,
        "iota": 0.0,
        "ra": 1.375,
        "dec": -1.2108,
        "psi": 0.0,
    }


def _heterodyne(setup, **overrides) -> HeterodynedTransientLikelihoodFD:
    ifos, waveform = setup
    kwargs = {
        "detectors": ifos,
        "waveform": waveform,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "trigger_time": GPS,
        "n_bins": N_BINS,
        "reference_parameters": reference_params(),
    }
    kwargs.update(overrides)
    return HeterodynedTransientLikelihoodFD(**kwargs)


def _dense(setup, **overrides) -> TransientLikelihoodFD:
    ifos, waveform = setup
    kwargs = {
        "detectors": ifos,
        "waveform": waveform,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "trigger_time": GPS,
    }
    kwargs.update(overrides)
    return TransientLikelihoodFD(**kwargs)


def _relative_log_l(likelihood, params) -> float:
    """Log-likelihood difference from the reference point (constants cancel)."""

    return float(likelihood.evaluate(params)) - float(
        likelihood.evaluate(reference_params())
    )


@pytest.mark.parametrize("phase_marginalization", [False, True])
def test_cubic_ratio_tracks_dense_likelihood_where_linear_interpolation_fails(
    setup, phase_marginalization
) -> None:
    shifted = {**reference_params(), "t_c": TIME_SHIFT_S}
    dense = _dense(setup, phase_marginalization=phase_marginalization)
    linear = _heterodyne(setup, phase_marginalization=phase_marginalization)
    cubic = _heterodyne(
        setup, phase_marginalization=phase_marginalization, interpolation_order=3
    )

    truth = _relative_log_l(dense, shifted)
    linear_error = abs(_relative_log_l(linear, shifted) - truth)
    cubic_error = abs(_relative_log_l(cubic, shifted) - truth)

    assert linear_error > 0.5, linear_error  # the defect this feature removes
    assert cubic_error < 0.02 * linear_error, (cubic_error, linear_error)
    assert cubic_error < 0.05, cubic_error


def test_default_order_is_one_and_unchanged(setup) -> None:
    implicit = _heterodyne(setup)
    explicit = _heterodyne(setup, interpolation_order=1)
    shifted = {**reference_params(), "t_c": 1.0e-3}

    assert implicit.interpolation_order == 1
    assert explicit.interpolation_order == 1
    for detector in implicit.detectors:
        assert implicit.summary_data[detector.name].shape == (4, implicit.n_bins)
    assert float(explicit.evaluate(shifted)) == float(implicit.evaluate(shifted))
    assert explicit.bin_edges_sha256 == implicit.bin_edges_sha256


def test_cubic_cache_evaluation_matches_direct_evaluation(setup) -> None:
    cubic = _heterodyne(setup, interpolation_order=3)
    shifted = {**reference_params(), "t_c": TIME_SHIFT_S, "d_L": 250.0}

    direct = float(cubic.evaluate(shifted))
    cached = float(
        cubic.evaluate_from_waveform(shifted, cubic.generate_waveform(shifted))
    )

    np.testing.assert_allclose(cached, direct, rtol=1.0e-12, atol=0.0)


def test_higher_order_binds_the_bin_plan_digest_without_moving_edges(setup) -> None:
    linear = _heterodyne(setup)
    cubic = _heterodyne(setup, interpolation_order=3)

    np.testing.assert_array_equal(
        np.asarray(cubic.freq_grid_edges), np.asarray(linear.freq_grid_edges)
    )
    assert cubic.bin_edges_sha256 != linear.bin_edges_sha256
    edges = linear.freq_grid_edges
    assert (
        HeterodynedTransientLikelihoodFD._bin_edges_sha256(edges)
        == HeterodynedTransientLikelihoodFD._bin_edges_sha256(
            edges, interpolation_order=1
        )
        == linear.bin_edges_sha256
    )
    assert (
        HeterodynedTransientLikelihoodFD._bin_edges_sha256(edges, interpolation_order=3)
        == cubic.bin_edges_sha256
    )


@pytest.mark.parametrize("order", [0, -1, 1.5, True, 9])
def test_invalid_interpolation_order_is_rejected(setup, order) -> None:
    with pytest.raises(ValueError, match="interpolation_order"):
        _heterodyne(setup, interpolation_order=order)


def test_time_marginalization_fails_closed_above_first_order(setup) -> None:
    with pytest.raises(ValueError, match="interpolation_order"):
        _heterodyne(
            setup,
            interpolation_order=3,
            time_marginalization={"tc_range": (-0.03, 0.03)},
        )


# --- analytic t_c phasor in the summary moments -------------------------------
#
# The rigid time-shift phasor exp(-2 pi i f dt) is known in closed form, so it
# need not be interpolated at all: strip it from the per-bin ratio (which then
# stays smooth) and dress the data moments with its Taylor expansion inside
# each bin, ``A_k(dt) = exp(2 pi i f_c dt) sum_m (i theta_b)^m / m! A_{k+m}``.
# The bin count is then set by the waveform, not by the phasor winding.

LARGE_TIME_SHIFT_S = 0.05  # ~2.5 rad of phasor per bin at 128 bins over the band


def test_phasor_moments_track_dense_likelihood_where_cubic_alone_fails(setup) -> None:
    shifted = {**reference_params(), "t_c": LARGE_TIME_SHIFT_S}
    dense = _dense(setup, phase_marginalization=True)
    cubic = _heterodyne(setup, phase_marginalization=True, interpolation_order=3)
    dressed = _heterodyne(
        setup,
        phase_marginalization=True,
        interpolation_order=3,
        phasor_moment_order=8,
    )

    truth = _relative_log_l(dense, shifted)
    cubic_error = abs(_relative_log_l(cubic, shifted) - truth)
    dressed_error = abs(_relative_log_l(dressed, shifted) - truth)

    assert cubic_error > 0.5, cubic_error  # the defect this feature removes
    assert dressed_error < 0.02 * cubic_error, (dressed_error, cubic_error)
    assert dressed_error < 0.05, dressed_error


def test_phasor_moments_reduce_to_plain_cubic_at_the_reference_time(setup) -> None:
    params = {**reference_params(), "M_c": 30.2, "d_L": 380.0}
    cubic = _heterodyne(setup, interpolation_order=3)
    dressed = _heterodyne(setup, interpolation_order=3, phasor_moment_order=6)
    assert float(dressed.evaluate(params)) == pytest.approx(
        float(cubic.evaluate(params)), rel=0.0, abs=1e-9
    )


def test_phasor_moment_cache_evaluation_matches_direct_evaluation(setup) -> None:
    dressed = _heterodyne(setup, interpolation_order=3, phasor_moment_order=8)
    params = {**reference_params(), "t_c": LARGE_TIME_SHIFT_S, "d_L": 350.0}
    cache = dressed.generate_waveform(params)
    assert float(dressed.evaluate_from_waveform(params, cache)) == pytest.approx(
        float(dressed.evaluate(params)), rel=0.0, abs=1e-8
    )


def test_phasor_moment_order_binds_the_bin_plan_digest(setup) -> None:
    cubic = _heterodyne(setup, interpolation_order=3)
    dressed = _heterodyne(setup, interpolation_order=3, phasor_moment_order=8)
    np.testing.assert_array_equal(
        np.asarray(cubic.freq_grid_edges), np.asarray(dressed.freq_grid_edges)
    )
    assert dressed.bin_edges_sha256 != cubic.bin_edges_sha256
    assert (
        dressed.bin_edges_sha256
        == HeterodynedTransientLikelihoodFD._bin_edges_sha256(
            dressed.freq_grid_edges, interpolation_order=3, phasor_moment_order=8
        )
    )


@pytest.mark.parametrize("order", [-1, 1.5, True, 17])
def test_invalid_phasor_moment_order_is_rejected(setup, order) -> None:
    with pytest.raises(ValueError, match="phasor_moment_order"):
        _heterodyne(setup, interpolation_order=3, phasor_moment_order=order)


def test_phasor_moments_require_polynomial_ratio(setup) -> None:
    with pytest.raises(ValueError, match="phasor_moment_order"):
        _heterodyne(setup, interpolation_order=1, phasor_moment_order=4)


def test_phasor_moment_order_zero_is_the_unmodified_polynomial_path_at_nonzero_shift(
    setup,
) -> None:
    """M = 0 must fit the *full* ratio (phasor included) and contract with the
    plain moments, i.e. the pre-phasor algorithm, also away from the reference
    time and sky.  Reimplemented here from the class's public data so the
    check does not depend on the branch inside ``_polynomial_likelihood``."""

    import jax.numpy as jnp

    from jimgw.core.utils import log_i0

    params = {
        **reference_params(),
        "t_c": TIME_SHIFT_S,
        "ra": 1.6,
        "dec": -1.0,
        "d_L": 380.0,
    }
    cubic = _heterodyne(setup, phase_marginalization=True, interpolation_order=3)
    assert cubic.phasor_moment_order == 0
    order, n_bins = cubic.interpolation_order, cubic.n_bins
    nodes = cubic.freq_grid_node_flat
    prepared = cubic._prepare_parameters(params)
    polarizations = cubic.waveform(nodes, prepared)
    z = 0.0j
    hh = 0.0
    for detector in cubic.detectors:
        projected = detector.fd_response(nodes, polarizations, prepared)
        ratio = (
            jnp.reshape(projected, (order + 1, n_bins))
            / cubic.waveform_node_ref[detector.name]
        )
        c = cubic._vandermonde_inverse @ ratio
        a, b = cubic.summary_moments[detector.name]
        assert a.shape[0] == order + 1  # no extra moment rows at M = 0
        z += jnp.sum(jnp.conj(c) * a)
        for k in range(order + 1):
            for m in range(order + 1):
                hh += jnp.sum(c[k] * jnp.conj(c[m]) * b[k + m]).real
    expected = float(log_i0(jnp.abs(z)) - 0.5 * hh)
    assert float(cubic.evaluate(params)) == pytest.approx(expected, rel=0.0, abs=1e-9)


def test_time_anchor_bank_tracks_dense_at_large_shifts_and_boundaries(setup):
    anchors = np.linspace(-0.15, 0.15, 31)
    likelihood = _heterodyne(
        setup,
        n_bins=32,
        interpolation_order=4,
        phasor_moment_order=16,
        phasor_time_anchors=anchors,
        phase_marginalization=True,
    )
    dense = _dense(setup, phase_marginalization=True)
    for tc in (-0.12, -0.055 - 1e-8, -0.055 + 1e-8, 0.12):
        params = {**reference_params(), "t_c": tc}
        error = _relative_log_l(likelihood, params) - _relative_log_l(dense, params)
        assert abs(error) < 0.02, (tc, error)
        cached = likelihood.evaluate_from_waveform(
            params, likelihood.generate_waveform(params)
        )
        np.testing.assert_allclose(cached, likelihood.evaluate(params), rtol=1e-12)
    assert np.isnan(float(likelihood.evaluate({**reference_params(), "t_c": 0.3})))


def test_time_anchor_bank_is_bound_to_qualification_digest(setup):
    a = _heterodyne(
        setup,
        interpolation_order=4,
        phasor_moment_order=8,
        phasor_time_anchors=[-0.1, 0.0, 0.1],
    )
    b = _heterodyne(
        setup,
        interpolation_order=4,
        phasor_moment_order=8,
        phasor_time_anchors=[-0.1, 0.01, 0.1],
    )
    assert a.bin_edges_sha256 != b.bin_edges_sha256


@pytest.mark.parametrize("anchors", [[], [0], [0, 0], [1, 0], [0, np.nan]])
def test_invalid_time_anchor_banks_are_rejected(setup, anchors):
    with pytest.raises(ValueError, match="phasor_time_anchors"):
        _heterodyne(
            setup,
            interpolation_order=4,
            phasor_moment_order=8,
            phasor_time_anchors=anchors,
        )


@pytest.mark.parametrize("phase_marginalization", [False, True])
def test_extrinsic_summary_matches_likelihood_and_rejects_stale_state(
    setup, phase_marginalization
):
    from jimgw.core.single_event.heterodyne_extrinsics import (
        build_extrinsic_summary,
        evaluate_extrinsic_summary,
    )

    likelihood = _heterodyne(
        setup,
        interpolation_order=4,
        phasor_moment_order=8,
        phasor_time_anchors=[-0.1, 0.0, 0.1],
        phase_marginalization=phase_marginalization,
    )
    fixed = {**reference_params(), "iota": 1.4, "t_c": 0.03}
    cache = build_extrinsic_summary(likelihood, fixed)
    for iota, psi, distance in [
        (0.0, 0.0, 200.0),
        (1.57, 0.7, 400.0),
        (3.14, 2.8, 900.0),
    ]:
        p = {**fixed, "iota": iota, "psi": psi, "d_L": distance}
        np.testing.assert_allclose(
            evaluate_extrinsic_summary(likelihood, p, cache),
            likelihood.evaluate(p),
            rtol=1e-11,
            atol=1e-9,
        )
    stale = {**fixed, "ra": fixed["ra"] + 0.01}
    assert np.isnan(float(evaluate_extrinsic_summary(likelihood, stale, cache)))


def test_carrier_reference_survives_a_projected_antenna_null(setup):
    from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time

    detector = setup[0][0]
    ref = {**reference_params(), "iota": np.pi / 2}
    antenna = detector.antenna_pattern(
        ref["ra"], ref["dec"], 0.0, greenwich_mean_sidereal_time(GPS)
    )
    ref["psi"] = float(0.5 * np.arctan2(-antenna["p"], antenna["c"]))
    options = {
        "reference_parameters": ref,
        "interpolation_order": 4,
        "phasor_moment_order": 8,
        "phase_marginalization": True,
    }
    with pytest.raises(ValueError, match="reference"):
        _heterodyne(setup, **options)
    likelihood = _heterodyne(setup, **options, reference_projection="carrier")
    dense = _dense(setup, phase_marginalization=True)
    for psi in (ref["psi"], ref["psi"] + 0.3):
        p = {**ref, "psi": psi, "t_c": 0.002}
        error = _relative_log_l(likelihood, p) - _relative_log_l(dense, p)
        assert abs(error) < 0.02, error
