from itertools import pairwise

import numpy as np
import pytest

from jimgw.core.single_event.heterodyne_quadrature import zero_noise_moments


def test_adaptive_zero_noise_quadrature_matches_independent_dense_overlaps():
    edges = np.array([2.0, 3.0, 10.0, 30.0])
    signal = lambda f: np.exp(0.4j * f**2) / f
    reference = lambda f: np.exp(0.4j * f**2 - 0.01j * f) / f
    psd = lambda f: 1 + f / 100
    result = zero_noise_moments(
        signal,
        reference,
        psd,
        edges,
        4,
        8,
        anchors=[-0.1, 0.0, 0.1],
        atol=1e-9,
        rtol=1e-10,
    )
    # Independent midpoint integration, not another call to the quadrature.
    for b, (lo, hi) in enumerate(pairwise(edges)):
        f = lo + (np.arange(100000) + 0.5) * (hi - lo) / 100000
        u = (2 * f - hi - lo) / (hi - lo)
        for a, dt in enumerate([-0.1, 0.0, 0.1]):
            for k in [0, 4, 12]:
                truth = (
                    4
                    * (hi - lo)
                    * np.mean(
                        signal(f)
                        * reference(f).conj()
                        / psd(f)
                        * np.exp(2j * np.pi * f * dt)
                        * u**k
                    )
                )
                np.testing.assert_allclose(
                    result.data[a, k, b], truth, atol=2e-9, rtol=1e-8
                )
        norm = 4 * (hi - lo) * np.mean(abs(reference(f)) ** 2 / psd(f))
        np.testing.assert_allclose(result.norm[0, b], norm, rtol=1e-9)
    assert result.evaluations < 10000


def test_quadrature_fails_when_error_budget_cannot_be_met():
    with pytest.raises(RuntimeError, match="budget"):
        zero_noise_moments(
            lambda f: np.exp(100j * f**2),
            lambda f: np.ones_like(f),
            lambda f: np.ones_like(f),
            [2.0, 30.0],
            4,
            8,
            max_evaluations=48,
            atol=1e-14,
            rtol=1e-14,
        )


def test_discrete_quadrature_preserves_fourier_endpoints_and_empty_bins():
    from jimgw.core.single_event.heterodyne_moments import polynomial_moments

    f = np.arange(20, 4001) * 0.1
    edges = np.array([2.0, 2.01, 2.02, 3.53, 13.0, 400.0])
    h = lambda x: np.exp(0.01j * x) / (1 + x)
    d = lambda x: h(x) * np.exp(0.005j * x)
    s = lambda x: np.ones_like(x)
    result = zero_noise_moments(
        d, h, s, edges, 4, 8, native_delta_f=0.1, atol=1e-11, rtol=1e-11
    )
    a, b = polynomial_moments(f, d(f), s(f), h(f), edges, 12, 8)
    np.testing.assert_allclose(result.data[0], 0.4 * a, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result.norm, 0.4 * b, rtol=1e-9, atol=1e-10)
    assert result.evaluations < len(f)


def test_detector_specific_frequency_bounds_exclude_other_network_samples():
    result = zero_noise_moments(
        np.ones_like,
        np.ones_like,
        np.ones_like,
        [2.0, 3.0, 10.0, 20.0],
        2,
        2,
        native_delta_f=0.25,
        native_frequency_bounds=(4.0, 12.0),
        breakpoints=[4.12, 5.0, 7.8, 11.0],
    )
    # First bin is empty; second includes 4 <= f < 10, last includes f=12.
    np.testing.assert_allclose(result.norm[0], [0, 24, 9], atol=1e-12)


def test_native_grid_endpoints_survive_float_division_roundoff():
    df = 0.1
    lo, hi = 3 * df, 13 * df
    result = zero_noise_moments(
        np.ones_like,
        np.ones_like,
        np.ones_like,
        [lo, hi],
        2,
        2,
        native_delta_f=df,
        native_frequency_bounds=(lo, hi),
    )
    np.testing.assert_allclose(result.norm[0, 0], 4 * df * 11, atol=1e-13)


@pytest.mark.parametrize("count,size", [(1, 16), (11, 16), (100, 16), (10000, 32)])
def test_discrete_gauss_rule_matches_independent_polynomial_sums(count, size):
    from jimgw.core.single_event.heterodyne_quadrature import discrete_gauss_rule

    x, w = discrete_gauss_rule(count, size)
    native = np.linspace(-1, 1, count) if count > 1 else np.array([0.0])
    for degree in range(2 * size):
        np.testing.assert_allclose(w @ x**degree, np.mean(native**degree), atol=2e-14)
