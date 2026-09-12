"""Native noisy sums: tile boundaries, partial chunks and moment contractions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event.heterodyne_moments import polynomial_moments
from jimgw.core.single_event.heterodyne_summary_tiles import (
    NativeGridTilePlanner,
    compiled_summary_tiles,
    plan_from_native_grid,
)

jax.config.update("jax_enable_x64", True)


def test_reusable_planner_reuses_boundaries_and_matches_one_shot(monkeypatch):
    from jimgw.core.single_event import heterodyne_summary_tiles as module

    edges = np.array([2.0, 2.001, 2.005, 2.3, 3.0])
    chunks = [(200, 56), (256, 32), (288, 96), (384, 1), (385, 100)]
    expected = [
        plan_from_native_grid(edges, first, 128, count, tile_size=17)
        for first, count in chunks
    ]
    planner = NativeGridTilePlanner(edges, 128, tile_size=17)

    def no_scalar_recomputation(*args, **kwargs):
        raise AssertionError("Scalar boundary conversion must happen only once")

    monkeypatch.setattr(module, "_native_boundary", no_scalar_recomputation)
    for (first, count), want in zip(chunks, expected, strict=True):
        got = planner.plan(first, count)
        for key in ("starts", "lengths", "bin_ids"):
            np.testing.assert_array_equal(getattr(got, key), getattr(want, key))
        assert got.active_tiles == want.active_tiles


def _reduce(f, d, s, h, edges, anchors, plan, data_order=24, norm_order=16):
    return tuple(
        np.asarray(value)
        for value in compiled_summary_tiles(
            *map(
                jnp.asarray,
                (f, d, s, h, edges, anchors, plan.starts, plan.lengths, plan.bin_ids),
            ),
            tile_size=plan.tile_size,
            data_order=data_order,
            norm_order=norm_order,
        )
    )


@pytest.mark.parametrize("duration", [128.0, 131072.0, 107.3])
def test_native_plan_preserves_exact_edge_searchsorted(duration):
    first, count = 219, 2049
    f = (first + np.arange(count)) / duration
    edges = np.array(
        [
            f[0] - 1 / duration,
            np.nextafter(f[1], -np.inf),
            f[2],
            np.nextafter(f[3], np.inf),
            f[900],
            np.nextafter(f[901], -np.inf),
            f[-1],
        ]
    )
    plan = plan_from_native_grid(edges, first, duration, count, tile_size=31)
    seen = np.full(count, -1)
    for start, length, bin_id in zip(
        plan.starts, plan.lengths, plan.bin_ids, strict=True
    ):
        assert np.all(seen[start : start + length] == -1)
        seen[start : start + length] = bin_id
    expected = np.searchsorted(edges, f, side="right") - 1
    expected[f == edges[-1]] = len(edges) - 2
    np.testing.assert_array_equal(seen, expected)
    assert plan.covered_samples == count


@pytest.mark.parametrize("anchors", [np.empty(0), np.linspace(-0.2, 0.2, 21)])
def test_fp64_gemm_moments_match_native_complex_noise(anchors):
    rng = np.random.default_rng(194)
    first, count, duration = 254, 3073, 128.0
    f = (first + np.arange(count)) / duration
    edges = np.array([2.0, 2.001, 2.008, 2.02, 3.0, 5.0, f[-1]])
    d = 1e-22 * (rng.normal(size=count) + 1j * rng.normal(size=count))
    h = 3e-23 * np.exp(-0.013j * f**2) / f
    psd = 1e-46 * (1 + f**0.3)
    # Unsupported frequencies below 2 Hz must be excluded without NaN leakage.
    psd[f < edges[0]] = np.nan
    d[f < edges[0]] = np.nan
    plan = plan_from_native_grid(edges, first, duration, count, tile_size=128)
    a, b, bank = _reduce(f, d, psd, h, edges, anchors, plan)
    want_a, want_b = polynomial_moments(f, d, psd, h, edges, 24, 16)
    np.testing.assert_allclose(a, want_a, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(b, want_b, rtol=2e-12, atol=2e-11)
    for i, anchor in enumerate(anchors):
        expected, _ = polynomial_moments(
            f, d * np.exp(2j * np.pi * f * anchor), psd, h, edges, 24, 0
        )
        np.testing.assert_allclose(bank[i], expected, rtol=2e-12, atol=2e-11)
    assert bank.shape == (len(anchors), 25, len(edges) - 1)
    # Independently reconstruct a complex degree-8 overlap from the summaries.
    coefficients = rng.normal(size=(9, len(edges) - 1)) + 1j * rng.normal(
        size=(9, len(edges) - 1)
    )
    keep = (f >= edges[0]) & (f <= edges[-1])
    fi = f[keep]
    idx = np.minimum(np.searchsorted(edges, fi, side="right") - 1, len(edges) - 2)
    u = (2 * fi - edges[idx] - edges[idx + 1]) / (edges[idx + 1] - edges[idx])
    ratio = sum(coefficients[k, idx] * u**k for k in range(9))
    np.testing.assert_allclose(
        np.sum(a[:9] * coefficients.conj()),
        np.sum(d[keep] * (h[keep] * ratio).conj() / psd[keep]),
        rtol=3e-12,
        atol=1e-10,
    )


def test_single_valid_endpoint_tail_ignores_invalid_padding():
    edges = np.array([2.0, 5.0, 8.0])
    plan = plan_from_native_grid(edges, 1024, 128, 1, tile_size=16)
    f = np.r_[8.0, np.full(31, 9.0)]
    d = np.r_[2 + 3j, np.full(31, np.nan + 1j * np.nan)]
    h = np.r_[1 - 2j, np.full(31, np.nan + 1j * np.nan)]
    psd = np.r_[4.0, np.full(31, np.nan)]
    a, b, bank = _reduce(f, d, psd, h, edges, [0.0, 0.1], plan, 3, 4)
    np.testing.assert_array_equal(a[:, 0], 0)
    np.testing.assert_allclose(a[:, 1], (2 + 3j) * (1 + 2j) / 4)
    np.testing.assert_allclose(b[:, 1], 1.25)
    np.testing.assert_allclose(bank[0], a)
    np.testing.assert_allclose(bank[1, :, 1], a[:, 1] * np.exp(2j * np.pi * 8 * 0.1))


def test_streamed_chunks_cover_all_samples_once():
    rng = np.random.default_rng(751)
    first, count, duration = 2048, 1025, 1024.0
    f = (first + np.arange(count)) / duration
    edges = np.array([2.0, 2.3, 2.7, 3.0])
    d = rng.normal(size=count) + 1j * rng.normal(size=count)
    h = np.exp(-1j * f**2)
    s = np.ones(count)
    totals = None
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        plan = plan_from_native_grid(
            edges, first + start, duration, stop - start, tile_size=32
        )
        outputs = _reduce(
            f[start:stop],
            d[start:stop],
            s[start:stop],
            h[start:stop],
            edges,
            [-0.2, 0.0, 0.2],
            plan,
            4,
            8,
        )
        totals = (
            outputs
            if totals is None
            else tuple(a + b for a, b in zip(totals, outputs, strict=True))
        )
    a, b = polynomial_moments(f, d, s, h, edges, 4, 8)
    np.testing.assert_allclose(totals[0], a, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(totals[1], b, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("bad_psd", [0.0, -1.0, np.nan, np.inf])
def test_bad_in_band_psd_propagates_failure_and_empty_support_is_zero(bad_psd):
    f = np.arange(64, 96) / 32
    edges = np.array([2.0, 2.5, 3.0])
    plan = plan_from_native_grid(edges, 64, 32, len(f), tile_size=16)
    psd = np.ones(len(f))
    psd[17] = bad_psd
    a, b, _ = _reduce(f, np.ones(len(f)), psd, np.ones(len(f)), edges, [], plan, 0, 0)
    assert np.isnan(a[:, 1]).all() and np.isnan(b[:, 1]).all()
    empty = plan_from_native_grid([5.0, 6.0], 64, 32, len(f), tile_size=16)
    output = _reduce(
        f, np.full(len(f), np.nan), psd, np.ones(len(f)), [5.0, 6.0], [], empty, 0, 0
    )
    for values in output:
        np.testing.assert_array_equal(values, 0)


def test_cancellation_across_tiles_uses_absolute_l1_roundoff_budget():
    first, count, duration = 256, 4096, 128.0
    f = (first + np.arange(count)) / duration
    edges = [f[0], f[-1]]
    rng = np.random.default_rng(975)
    d = np.tile([1e8 + 1e8j, -1e8 - 1e8j], count // 2)
    d += 1e-7 * (rng.normal(size=count) + 1j * rng.normal(size=count))
    plan = plan_from_native_grid(edges, first, duration, count, tile_size=128)
    a, _, bank = _reduce(f, d, np.ones(count), np.ones(count), edges, [0.0], plan, 4, 0)
    # Long-double accumulation is independent of the GEMM/segment hierarchy.
    # A0 nearly cancels, so a relative tolerance against its tiny result is
    # inappropriate. This is an empirical roundoff budget, not a certificate.
    u = (2 * f - edges[0] - edges[1]) / (edges[1] - edges[0])
    for degree in range(5):
        terms = d.astype(np.clongdouble) * u.astype(np.longdouble) ** degree
        expected = np.sum(terms, dtype=np.clongdouble)
        budget = 64 * np.finfo(float).eps * float(np.sum(np.abs(terms)))
        assert abs(a[degree, 0] - expected) <= budget
        # The zero-anchor and unanchored moments are mathematically equal,
        # but independent GPU reduction columns can round differently. Check
        # both against the same independent oracle and unchanged L1 budget;
        # requiring bitwise agreement would test summation order, not accuracy.
        assert abs(bank[0, degree, 0] - expected) <= budget
