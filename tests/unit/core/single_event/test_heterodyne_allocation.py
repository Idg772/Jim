"""Scientific and contract checks for the bounded, training-only allocator."""

import itertools

import numpy as np
import pytest
from scipy.special import i0e

from jimgw.core.single_event.heterodyne_allocation import (
    allocate_partitions,
    build_interval_catalogue,
)
from jimgw.core.single_event.heterodyne_selection import (
    FrozenGridPlan,
    legendre_lobatto_nodes,
    select_frozen_grid,
)


def test_catalogue_preserves_shifted_incumbent_and_exact_shared_node_bank():
    edges = np.geomspace(2, 1000, 33)
    incumbent = (0, 8, 16, 24, 32)
    shifted = (0, 4, 8, 20, 32)
    cat = build_interval_catalogue(edges, incumbent, extra_partitions=[shifted])
    again = build_interval_catalogue(
        edges, incumbent, extra_partitions=[shifted, shifted]
    )
    assert cat.digest == again.digest
    assert cat.explicit_partitions == (incumbent, shifted)
    np.testing.assert_array_equal(
        cat.intervals[cat.fine_interval_indices],
        np.column_stack((np.arange(32), np.arange(1, 33))),
    )
    for parent, children in enumerate(cat.children):
        if children[0] >= 0:
            assert np.all(children < parent)
            left, right = cat.intervals[children]
            assert left[1] == right[0]
            np.testing.assert_array_equal(cat.intervals[parent], [left[0], right[1]])
    bank = cat.sparse_nodes()
    np.testing.assert_array_equal(
        bank.frequencies[bank.interval_node_indices], cat.interval_nodes()
    )
    np.testing.assert_array_equal(
        bank.frequencies[bank.reference_node_indices], FrozenGridPlan(edges, 8, 0).nodes
    )
    assert bank.check_indices.shape == (32 * 23,)
    assert not bank.frequencies.flags.writeable
    assert not cat.intervals.flags.writeable
    altered = edges.copy()
    altered[4] = np.nextafter(altered[4], np.inf)
    assert (
        build_interval_catalogue(altered, incumbent, extra_partitions=[shifted]).digest
        != cat.digest
    )


def test_realistic_catalogue_is_bounded_and_keeps_full_fine_reference():
    edges = np.geomspace(2, 2048, 2049)
    cat = build_interval_catalogue(
        edges,
        np.arange(257) * 8,
        extra_partitions=[np.arange(167) * 2048 // 166, np.arange(2049)],
    )
    assert cat.n_intervals < 4096
    assert len(cat.fine_interval_indices) == 2048
    assert {len(p) - 1 for p in cat.explicit_partitions} == {166, 256, 2048}
    with pytest.raises(ValueError, match="interval budget"):
        build_interval_catalogue(edges, np.arange(257) * 8, max_intervals=100)
    with pytest.raises(ValueError, match="frequency budget"):
        cat.sparse_nodes(max_frequencies=10)


def _polynomial_stats(cat):
    """Independent high-order quadrature supplies an actual complex fit error."""
    x, w = np.polynomial.legendre.leggauss(128)
    inverse = np.linalg.inv(
        np.polynomial.polynomial.polyvander(legendre_lobatto_nodes(8), 8)
    )

    def waveform(f):
        return 1 + 0.5 * np.exp(-(((f - 0.12) / 0.025) ** 2)) * np.exp(80j * f)

    delta_z = np.zeros((2, cat.n_intervals), dtype=complex)
    delta_q = np.zeros((2, cat.n_intervals))
    exact_z = np.zeros(cat.n_intervals, dtype=complex)
    exact_q = np.zeros(cat.n_intervals)
    for t, (i, j) in enumerate(cat.intervals):
        lo, hi = cat.reference_edges[[i, j]]
        f, weight = (lo + hi) / 2 + (hi - lo) / 2 * x, (hi - lo) / 2 * w
        truth = waveform(f)
        data = 2 + np.sin(7 * f)
        c = inverse @ waveform(cat.interval_nodes()[t])
        fit = np.polynomial.polynomial.polyval(x, c)
        exact_z[t] = np.sum(weight * truth.conj() * data)
        exact_q[t] = np.sum(weight * abs(truth) ** 2)
        delta_z[1, t] = np.sum(weight * (fit - truth).conj() * data)
        delta_q[1, t] = np.sum(weight * (abs(fit) ** 2 - abs(truth) ** 2))
    return (
        delta_z,
        delta_q,
        exact_z[cat.fine_interval_indices].sum(),
        exact_q[cat.fine_interval_indices].sum(),
    )


def test_nonuniform_k8_partition_improves_actual_coherent_likelihood_at_same_cost():
    cat = build_interval_catalogue(np.linspace(0, 1, 17), (0, 4, 8, 12, 16))
    z, q, reference_z, reference_q = _polynomial_stats(cat)
    proposals = allocate_partitions(
        cat,
        z,
        q,
        np.ones(cat.n_intervals),
        error_budget=1e-4,
        multipliers=(),
        budget_counts=(4,),
        max_local_edits=0,
    )

    def logl(overlap, norm):
        amplitude = abs(overlap)
        return np.log(i0e(amplitude)) + amplitude - norm / 2

    ref = logl(reference_z, reference_q)
    errors = {
        p.edge_indices: abs(
            logl(reference_z + p.delta_z[1], reference_q + p.delta_q[1]) - ref
        )
        for p in proposals
    }
    incumbent_error = errors[cat.explicit_partitions[0]]
    best = min(proposals, key=lambda p: errors[p.edge_indices])
    assert best.n_bins == 4
    assert best.edge_indices != cat.explicit_partitions[0]
    assert errors[best.edge_indices] < incumbent_error / 10
    assert best.additive_bound < next(
        p.additive_bound for p in proposals if p.origin == "incumbent"
    )


def test_leaf_budget_recovers_a_tradeoff_unsupported_by_multiplier_pruning():
    cat = build_interval_catalogue(np.arange(5.0), np.arange(5), refinement_depth=0)
    z = np.zeros((2, cat.n_intervals), dtype=complex)
    for t, (lo, hi) in enumerate(cat.intervals):
        z[1, t] = {1: 0.0, 2: 4.5, 4: 10.0}[hi - lo]
    kwargs = {"error_budget": 9.0, "max_local_edits": 0}
    pure = allocate_partitions(
        cat,
        z,
        np.zeros_like(z.real),
        np.ones(cat.n_intervals),
        budget_counts=(),
        **kwargs,
    )
    assert 2 not in {p.n_bins for p in pure}
    enriched = allocate_partitions(
        cat,
        z,
        np.zeros_like(z.real),
        np.ones(cat.n_intervals),
        budget_counts=(2,),
        **kwargs,
    )
    two = next(p for p in enriched if p.n_bins == 2)
    assert two.additive_bound == 9
    assert two.origin == "leaf_budget"


def test_budget_dp_matches_exhaustive_small_tree_and_honors_forest_roots():
    cat = build_interval_catalogue(
        np.arange(9.0), np.arange(9), root_edge_indices=(0, 4, 8), refinement_depth=0
    )
    rng = np.random.default_rng(2)
    error = rng.uniform(0, 2, cat.n_intervals)
    cost = rng.uniform(1, 3, cat.n_intervals)
    z = np.stack((np.zeros_like(error), error)).astype(complex)

    def tree(t):
        result = [(t,)]
        left, right = cat.children[t]
        if left >= 0:
            result += [a + b for a, b in itertools.product(tree(left), tree(right))]
        return result

    exhaustive = [a + b for a, b in itertools.product(*(tree(t) for t in cat.roots))]
    result = allocate_partitions(
        cat,
        z,
        np.zeros_like(z.real),
        cost,
        error_budget=1,
        multipliers=(),
        budget_counts=range(2, 9),
        max_local_edits=32,
    )
    for count in range(2, 9):
        expected = min(
            (sum(error[list(ids)]), sum(cost[list(ids)]))
            for ids in exhaustive
            if len(ids) == count
        )
        actual = min((p.additive_bound, p.cost) for p in result if p.n_bins == count)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert all(4 in p.edge_indices for p in result)


def test_bounds_cover_centered_phase_marginalized_and_fixed_phase_errors():
    cat = build_interval_catalogue(np.arange(5.0), (0, 2, 4))
    rng = np.random.default_rng(19)
    z = rng.normal(size=(5, cat.n_intervals)) + 1j * rng.normal(
        size=(5, cat.n_intervals)
    )
    q = rng.normal(size=z.shape)
    ref_z = np.array([0j, 1 + 2j, -8j, 20, 0j])
    result = allocate_partitions(cat, z, q, np.ones(cat.n_intervals), error_budget=0.1)
    for p in result:
        overlap = abs(ref_z + p.delta_z)
        old = abs(ref_z)
        marginal_error = (
            np.log(i0e(overlap)) + overlap - np.log(i0e(old)) - old - p.delta_q / 2
        )
        fixed_error = p.delta_z.real - p.delta_q / 2
        for error in (marginal_error, fixed_error):
            assert np.all(abs(error - error[0]) <= p.case_bounds + 1e-12)
            assert np.max(abs(error - error[0])) <= p.additive_bound + 1e-12


def test_explicit_candidates_survive_tight_bound_and_output_cap_deterministically():
    cat = build_interval_catalogue(
        np.arange(17.0), (0, 4, 8, 12, 16), extra_partitions=[(0, 3, 8, 16)]
    )
    z = np.ones((2, cat.n_intervals), dtype=complex)
    args = (cat, z, np.zeros_like(z.real), np.ones(cat.n_intervals))
    one = allocate_partitions(*args, error_budget=1e-12, max_candidates=3)
    two = allocate_partitions(*args, error_budget=1e-12, max_candidates=3)
    assert len(one) == 3
    assert {p.edge_indices for p in one}.issuperset(cat.explicit_partitions)
    assert [p.digest for p in one] == [p.digest for p in two]
    assert all(p.additive_bound > 1e-12 for p in one)
    with pytest.raises(ValueError, match="discard explicit"):
        allocate_partitions(*args, error_budget=1, max_candidates=1)


def test_single_fine_bin_and_all_zero_error_are_supported():
    cat = build_interval_catalogue([2.0, 5.0], [0, 1])
    result = allocate_partitions(
        cat, np.zeros((1, 1), complex), np.zeros((1, 1)), [0], error_budget=0.01
    )
    assert len(result) == 1
    assert result[0].edge_indices == (0, 1)
    assert result[0].additive_bound == 0


@pytest.mark.parametrize("partition", [(0.0, 1.0, 4.0), (0, 0, 4), (1, 4), (0, 5)])
def test_rejects_nonexact_or_incomplete_index_partitions(partition):
    with pytest.raises(ValueError, match="fine-edge indices"):
        build_interval_catalogue(np.arange(5.0), partition)


def test_explicit_nonuniform_selector_uses_one_shared_bank_then_verifies_only_frozen_winner():
    calls, scored = [], []
    explicit = ((0, 2, 4, 8), (0, 4, 6, 8))

    def bank(name):
        def call(f):
            calls.append((name, f.copy()))
            return np.ones((1, len(f)), dtype=complex)

        return call

    def score(plan, values, name):
        scored.append((name, tuple(plan.edges)))
        assert values.shape == (1, 3, 9)
        return np.array([0.0 if name == "training" else 1.0])

    with pytest.raises(RuntimeError, match="not used to retune"):
        select_frozen_grid(
            np.arange(9.0),
            bank("training"),
            bank("verification"),
            edge_index_candidates=explicit,
            max_bins=3,
            error_budget=0.01,
            likelihood_error=score,
            validated_cost=lambda p: float(p.edges[1]),
        )
    assert [name for name, _ in calls] == ["training", "verification"]
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    assert scored == [
        ("training", explicit[0]),
        ("training", explicit[1]),
        ("verification", explicit[0]),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"edge_index_candidates": ()},
        {"edge_index_candidates": ((0, 3, 9),)},
        {"edge_index_candidates": ((0.0, 3.0, 8.0),)},
        {"edge_index_candidates": ((0, 3, 8),), "bin_counts": (2,)},
    ],
)
def test_explicit_selector_rejects_invalid_family_before_bank_evaluation(kwargs):
    def never(_):
        pytest.fail("invalid candidate family must not evaluate a bank")

    with pytest.raises(ValueError, match="edge_index_candidates"):
        select_frozen_grid(np.arange(9.0), never, never, error_budget=0.01, **kwargs)
