"""Integration test: BlackJAX nested slice sampler end-to-end with a 2-D Gaussian."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import logsumexp

pytestmark = pytest.mark.integration

blackjax = pytest.importorskip("blackjax")

from jimgw.samplers.config import BlackJAXNSSConfig
from tests.integration._helpers import make_gaussian_jim


@pytest.fixture(scope="module")
def nss_jim():
    cfg = BlackJAXNSSConfig(n_live=50, termination_dlogz=0.5)
    jim = make_gaussian_jim(cfg)
    jim.sample()
    return jim


def test_nss_get_samples_shape(nss_jim):
    samples = nss_jim.get_samples()
    assert set(samples.keys()) == {"x", "y", "log_likelihood"}
    n = samples["x"].shape[0]
    assert n > 0
    assert samples["y"].shape == (n,)
    assert samples["log_likelihood"].shape == (n,)


def test_nss_get_weighted_samples_returns_original_nested_points(nss_jim):
    samples = nss_jim.get_weighted_samples()
    assert set(samples) == {
        "x",
        "y",
        "log_likelihood",
        "log_likelihood_birth",
        "log_weights",
    }

    n_nested = len(nss_jim.sampler._nested_samples)
    assert samples["x"].shape == (n_nested,)
    assert samples["y"].shape == (n_nested,)
    assert samples["log_likelihood"].shape == (n_nested,)
    assert samples["log_likelihood_birth"].shape == (n_nested,)
    assert samples["log_weights"].shape == (n_nested,)
    assert samples["log_weights"].dtype == np.float64
    assert logsumexp(samples["log_weights"]) == pytest.approx(0.0, abs=1e-12)

    raw_log_weights = np.asarray(
        nss_jim.sampler._nested_samples.logw(), dtype=np.float64
    )
    expected = raw_log_weights - logsumexp(raw_log_weights)
    np.testing.assert_array_equal(samples["log_weights"], expected)
    np.testing.assert_array_equal(
        samples["log_likelihood_birth"],
        np.asarray(nss_jim.sampler._nested_samples["logL_birth"]),
    )


def test_nss_posterior_mean_near_half(nss_jim):
    samples = nss_jim.get_samples()
    assert abs(float(np.mean(samples["x"])) - 0.5) < 0.1
    assert abs(float(np.mean(samples["y"])) - 0.5) < 0.1


def test_nss_log_evidence_finite(nss_jim):
    diag = nss_jim.sampler.get_diagnostics()
    log_z = diag["log_Z"]
    assert math.isfinite(log_z)
