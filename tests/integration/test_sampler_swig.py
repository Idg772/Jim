"""Integration test: BlackJAX Nested Slice within Gibbs (SwiG) sampler end-to-end with a 2-D Gaussian."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

pytestmark = pytest.mark.integration

blackjax = pytest.importorskip("blackjax")

from jimgw.samplers.config import BlackJAXSwiGConfig
from tests.integration._helpers import make_gaussian_swig_jim


@pytest.fixture(scope="module")
def swig_jim():
    cfg = BlackJAXSwiGConfig(blocks=[["x"], ["y"]], n_live=50, termination_dlogz=0.5)
    jim = make_gaussian_swig_jim(cfg)
    jim.sample()
    return jim


def test_swig_get_samples_shape(swig_jim):
    samples = swig_jim.get_samples()
    assert set(samples.keys()) == {"x", "y", "log_likelihood"}
    n = samples["x"].shape[0]
    assert n > 0
    assert samples["y"].shape == (n,)
    assert samples["log_likelihood"].shape == (n,)


def test_swig_inherits_direct_weighted_sample_path(swig_jim):
    samples = swig_jim.get_weighted_samples()
    assert set(samples) == {
        "x",
        "y",
        "log_likelihood",
        "log_likelihood_birth",
        "log_weights",
    }

    n_nested = len(swig_jim.sampler._nested_samples)
    assert all(value.shape == (n_nested,) for value in samples.values())
    assert samples["log_weights"].dtype == np.float64
    assert logsumexp(samples["log_weights"]) == pytest.approx(0.0, abs=1e-12)

    expected_log_likelihood = np.asarray(swig_jim.sampler._nested_samples["logL"])
    np.testing.assert_array_equal(samples["log_likelihood"], expected_log_likelihood)
    np.testing.assert_array_equal(
        samples["log_likelihood_birth"],
        np.asarray(swig_jim.sampler._nested_samples["logL_birth"]),
    )


def test_swig_posterior_mean_near_half(swig_jim):
    samples = swig_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.1
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.1


def test_swig_log_evidence_finite(swig_jim):
    diag = swig_jim.sampler.get_diagnostics()
    log_z = diag["log_Z"]
    assert np.isfinite(log_z)


def test_swig_does_not_store_cache_on_live_particles(swig_jim):
    assert not hasattr(swig_jim.sampler._final_state.particles, "cache")
