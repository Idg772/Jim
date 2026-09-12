"""Independent discrete-overlap checks for compressed summary construction."""

import numpy as np

from jimgw.core.single_event.heterodyne_moments import polynomial_moments


def test_moments_reproduce_complex_polynomial_overlaps_with_noise():
    rng = np.random.default_rng(91)
    f = np.linspace(2.0, 64.0, 4001)
    edges = np.array([2.0, 2.2, 7.0, 31.0, 64.0])
    reference = np.exp(-0.013j * f**2) / f
    data = reference + 0.1 * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
    psd = 0.1 + f**0.3
    a, b = polynomial_moments(f, data, psd, reference, edges, 5, 6)
    index = np.minimum(np.searchsorted(edges, f, side="right") - 1, len(edges) - 2)
    u = (2 * f - edges[index] - edges[index + 1]) / (edges[index + 1] - edges[index])
    c = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    ratio = sum(c[k, index] * u**k for k in range(4))
    h = reference * ratio
    overlap = np.sum(np.conj(c) * a[:4])
    norm = sum(
        np.sum(c[k] * c[j].conj() * b[k + j]) for k in range(4) for j in range(4)
    )
    np.testing.assert_allclose(
        overlap, np.sum(data * h.conj() / psd), rtol=1e-13, atol=1e-12
    )
    np.testing.assert_allclose(norm, np.sum(abs(h) ** 2 / psd), rtol=1e-13, atol=1e-12)
    assert a.shape == (6, 4)
    assert b.shape == (7, 4)
