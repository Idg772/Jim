import itertools

import numpy as np
import pytest

from jimgw.core.single_event.heterodyne_binning import allocate_ratio_bins


def test_error_budget_refines_high_curvature_and_matches_independent_overlap():
    ratio = lambda f: np.stack([np.exp(0.003j * f**3), np.exp(-0.002j * f**3)])
    result = allocate_ratio_bins(
        ratio,
        np.ones_like,
        np.ones_like,
        [2.0, 10.0, 20.0, 30.0],
        order=4,
        error_budget=1e-4,
        max_bins=512,
    )
    assert max(result.estimated_error) < 1e-4
    edges = result.edges
    assert np.count_nonzero(edges > 20) > np.count_nonzero(edges < 10)
    for case in range(2):
        approximate, exact = 0j, 0j
        nodes = -np.cos(np.arange(5) * np.pi / 4)
        for lo, hi in itertools.pairwise(edges):
            f = lo + (np.arange(2000) + 0.5) * (hi - lo) / 2000
            u = (2 * f - lo - hi) / (hi - lo)
            values = ratio((lo + hi) / 2 + (hi - lo) / 2 * nodes)[case]
            coefficients = np.polynomial.polynomial.polyfit(nodes, values, 4)
            approximate += (hi - lo) * np.mean(
                np.polynomial.polynomial.polyval(u, coefficients)
            )
            exact += (hi - lo) * np.mean(ratio(f)[case])
        assert abs(approximate - exact) < 1e-4


def test_allocator_fails_instead_of_returning_an_underresolved_grid():
    with pytest.raises(RuntimeError, match="bin budget"):
        allocate_ratio_bins(
            lambda f: np.exp(10j * f)[None, :],
            np.ones_like,
            np.ones_like,
            [2.0, 10.0],
            order=4,
            error_budget=1e-12,
            max_bins=2,
        )
