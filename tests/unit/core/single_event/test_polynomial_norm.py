"""The polynomial likelihood must reproduce a dense positive-measure integral."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import i0e

from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD


@pytest.mark.parametrize("order", [2, 4, 8])
@pytest.mark.parametrize("phase_marginalization", [False, True])
def test_polynomial_contraction_matches_dense_complex_signal(
    order, phase_marginalization
):
    rng = np.random.default_rng(8217 + order)
    bins = 5
    u = np.linspace(-1.0, 1.0, 101)
    weights = rng.uniform(0.1, 1.0, (len(u), bins))
    data = rng.normal(size=weights.shape) + 1j * rng.normal(size=weights.shape)
    coefficients = rng.normal(size=(order + 1, bins)) + 1j * rng.normal(
        size=(order + 1, bins)
    )
    ratio = np.polynomial.polynomial.polyval(u, coefficients).T
    overlap = np.sum(data * np.conj(ratio) * weights)
    norm = np.sum(abs(ratio) ** 2 * weights)
    expected = (
        np.log(i0e(abs(overlap))) + abs(overlap)
        if phase_marginalization
        else overlap.real
    ) - 0.5 * norm
    nodes = np.cos(np.pi * np.arange(order + 1) / order)
    values = np.polynomial.polynomial.polyval(nodes, coefficients).T
    a = np.stack(
        [np.sum(data * weights * u[:, None] ** k, axis=0) for k in range(order + 1)]
    )
    b = np.stack(
        [np.sum(weights * u[:, None] ** k, axis=0) for k in range(2 * order + 1)]
    )
    lk = object.__new__(HeterodynedTransientLikelihoodFD)
    lk.detectors = [
        SimpleNamespace(name="test", fd_response=lambda f, pols, p: pols["p"])
    ]
    lk.interpolation_order = order
    lk.phasor_moment_order = 0
    lk.phase_marginalization = phase_marginalization
    lk.n_bins = bins
    lk.freq_grid_node_flat = jnp.arange(values.size, dtype=jnp.float64)
    lk.waveform_node_ref = {"test": jnp.ones(values.shape)}
    lk.summary_moments = {"test": (jnp.asarray(a), jnp.asarray(b))}
    lk._vandermonde_inverse = jnp.asarray(
        np.linalg.inv(np.vander(nodes, N=order + 1, increasing=True))
    )
    result = jax.jit(lambda pols: lk._polynomial_likelihood({}, pols))(
        {"p": jnp.asarray(values.ravel())}
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-9)
