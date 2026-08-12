"""Tests for public sampler diagnostic helpers."""

import numpy as np
import pytest

from jimgw.samplers import insertion_index_diagnostic


def test_insertion_index_diagnostic_excludes_initial_live_points() -> None:
    log_likelihood = np.arange(1.0, 7.0)
    log_likelihood_birth = np.asarray(
        [-np.inf, 1.0, 2.0, -np.inf, 3.0, -np.inf]
    )

    result = insertion_index_diagnostic(
        log_likelihood,
        log_likelihood_birth,
        n_live=3,
    )

    assert result == {
        "method": "discrete-uniform-kolmogorov-smirnov",
        "n_live": 3,
        "sample_size": 3,
        "statistic": pytest.approx(1.0 / 3.0),
        "p_value": pytest.approx(0.8927783372501085),
        "index_min": 0,
        "index_max": 1,
        "index_mean": pytest.approx(1.0 / 3.0),
        "expected_index_mean": 1.0,
    }
