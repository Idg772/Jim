import numpy as np
import pytest

from jimgw.core.single_event.heterodyne_selection import (
    legendre_lobatto_nodes,
    select_frozen_grid,
)
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD


@pytest.mark.parametrize("order", range(1, 9))
def test_selection_nodes_match_actual_likelihood(order):
    np.testing.assert_array_equal(
        legendre_lobatto_nodes(order),
        HeterodynedTransientLikelihoodFD._lobatto_nodes(order),
    )


def test_shared_oracle_bank_selects_grid_that_passes_independent_dense_overlap():
    calls = {"training": [], "verification": []}

    def bank(name, phase):
        def evaluate(f):
            calls[name].append(f.copy())
            return np.exp(phase * 1j * f**3)[None, :]

        return evaluate

    result = select_frozen_grid(
        np.linspace(2, 30, 257),
        bank("training", 0.0003),
        bank("verification", -0.0002),
        orders=(4,),
        error_budget=1e-4,
        max_bins=256,
    )
    assert len(calls["training"]) == len(calls["verification"]) == 1
    np.testing.assert_array_equal(calls["training"][0], calls["verification"][0])
    np.testing.assert_array_equal(
        result.frequencies[result.plan.node_indices], result.plan.nodes
    )
    assert result.plan.n_bins < 256
    assert result.cost_metric == "analytical_work_proxy"
    assert np.max(result.verification_error) < 1e-4
    assert not result.plan.edges.flags.writeable
    nodes = legendre_lobatto_nodes(result.plan.order)
    approximate, exact = 0j, 0j
    for lo, hi in zip(result.plan.edges[:-1], result.plan.edges[1:], strict=True):
        # A separate, substantially denser midpoint integration checks the
        # selected interpolation instead of reproducing the engine's checks.
        f = lo + (np.arange(2000) + 0.5) * (hi - lo) / 2000
        u = (2 * f - lo - hi) / (hi - lo)
        values = np.exp(-0.0002j * ((lo + hi) / 2 + (hi - lo) / 2 * nodes) ** 3)
        coefficients = np.polynomial.polynomial.polyfit(
            nodes, values, result.plan.order
        )
        approximate += (hi - lo) * np.mean(
            np.polynomial.polynomial.polyval(u, coefficients)
        )
        exact += (hi - lo) * np.mean(np.exp(-0.0002j * f**3))
    assert abs(approximate - exact) < 1e-4


def test_shared_reference_oracle_preserves_detector_axis_and_explicit_cost_order():
    banks = {}
    observed = []

    def ratios(name):
        def evaluate(f):
            values = np.stack([np.ones_like(f), np.exp(0.1j * f)])
            banks[name] = np.stack([values, values * 2])
            return banks[name]

        return evaluate

    def score(plan, values, name):
        observed.append((name, plan.n_bins))
        assert values.shape == (2, 2, plan.n_bins, 5)
        np.testing.assert_array_equal(values, banks[name][..., plan.node_indices])
        assert plan.reference_node_indices.shape == (16, 5)
        assert plan.check_indices.ndim == 1
        return np.array([0.001, -0.002]) if plan.n_bins >= 4 else np.ones(2)

    result = select_frozen_grid(
        np.linspace(0, 1, 17),
        ratios("training"),
        ratios("verification"),
        orders=(4,),
        bin_counts=(2, 4, 8),
        max_bins=16,
        error_budget=0.01,
        likelihood_error=score,
        cost=lambda p: {2: 0.0, 4: 3.0, 8: 1.0, 16: 10.0}[p.n_bins],
    )
    assert result.plan.n_bins == 8
    assert observed == [("training", 2), ("training", 8), ("verification", 8)]
    assert result.training_residual is None
    assert result.verification_residual is None
    assert result.error_metric == "shared_reference_likelihood"
    assert result.cost_metric == "provided_cost"
    np.testing.assert_array_equal(result.verification_error, [0.001, 0.002])


def test_frozen_winner_failure_does_not_retune_against_verification_bank():
    calls = []

    def score(plan, values, name):
        calls.append((name, plan.n_bins))
        return np.array([0.0 if name == "training" else 1.0])

    with pytest.raises(RuntimeError, match="failed independent verification"):
        select_frozen_grid(
            np.linspace(0, 1, 9),
            lambda f: np.ones((1, len(f))),
            lambda f: np.ones((1, len(f))),
            orders=(2,),
            max_bins=8,
            error_budget=0.01,
            likelihood_error=score,
        )
    assert calls == [("training", 1), ("verification", 1)]


def test_validated_timing_skips_invalid_candidates_and_can_prefer_more_nodes():
    timed, scored = [], []

    def score(plan, values, name):
        scored.append((name, plan.n_bins))
        return np.asarray([1.0 if plan.n_bins == 1 else 0.0])

    def time_valid(plan):
        timed.append(plan.n_bins)
        return {2: 0.002, 4: 0.001, 8: 0.003}[plan.n_bins]

    result = select_frozen_grid(
        np.linspace(0, 1, 9),
        lambda f: np.ones((1, len(f))),
        lambda f: np.ones((1, len(f))),
        orders=(2,),
        max_bins=8,
        error_budget=0.01,
        likelihood_error=score,
        validated_cost=time_valid,
    )
    assert timed == [2, 4, 8]
    assert result.plan.n_bins == 4
    assert result.plan.cost == 0.001
    assert result.cost_metric == "validated_cost"
    assert scored == [
        ("training", 1),
        ("training", 2),
        ("training", 4),
        ("training", 8),
        ("verification", 4),
    ]
    assert "validated_cost" not in result.candidate_diagnostics[0]
    assert result.candidate_diagnostics[2]["validated_cost"] == 0.001
    assert (
        result.candidate_diagnostics[2]["cost"]
        > result.candidate_diagnostics[1]["cost"]
    )


def test_validated_cost_rejects_nonfinite_timing_before_verification():
    def holdout(f):
        pytest.fail("invalid timing must fail before verification")

    with pytest.raises(ValueError, match="validated candidate cost"):
        select_frozen_grid(
            [0, 1],
            lambda f: np.ones((1, len(f))),
            holdout,
            orders=(2,),
            max_bins=1,
            error_budget=0.01,
            likelihood_error=lambda p, v, name: np.asarray([0.0]),
            validated_cost=lambda p: np.nan,
        )


def test_budget_exhaustion_does_not_evaluate_holdout():
    def holdout(f):
        pytest.fail("a failed search must not consume the verification bank")

    with pytest.raises(RuntimeError, match="bin budget exhausted"):
        select_frozen_grid(
            [0, 0.5, 1],
            lambda f: np.exp(100j * f)[None],
            holdout,
            orders=(2,),
            max_bins=2,
            error_budget=1e-8,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_bins": 0}, "max_bins"),
        ({"max_bins": True}, "max_bins"),
        ({"orders": ()}, "at least one"),
        ({"orders": (2.5,)}, "order"),
        ({"orders": (9,)}, "order"),
        ({"bin_counts": (17,)}, "bin_counts"),
        ({"error_budget": np.nan}, "error_budget"),
        ({"cost": lambda p: np.inf}, "cost"),
        ({"data_weight": lambda f: -np.ones_like(f)}, "nonnegative"),
        ({"likelihood_error": lambda p, v, name: [np.nan]}, "finite"),
    ],
)
def test_invalid_oracles_and_search_controls_fail_closed(kwargs, match):
    options = {"orders": (2,), "max_bins": 16, "error_budget": 0.01}
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        select_frozen_grid(
            np.linspace(0, 1, 17),
            lambda f: np.ones((1, len(f))),
            lambda f: np.ones((1, len(f))),
            **options,
        )
