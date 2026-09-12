"""Bounded K8 partition proposals from reusable interval statistics.

This module does no waveform evaluation, compilation, likelihood qualification,
or holdout selection. Tree solves are exact only for their additive surrogate
and declared tree. The caller must score complete coherent likelihoods, choose
the training winner, and only then consume independent verification data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .heterodyne_selection import _edges, _readonly, legendre_lobatto_nodes


def _positive_integer(value, name, *, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _partition(values, n_fine, name):
    values = np.asarray(values)
    if (
        values.ndim != 1
        or values.dtype.kind not in "iu"
        or len(values) < 2
        or values[0] != 0
        or values[-1] != n_fine
        or np.any(values[1:] <= values[:-1])
    ):
        raise ValueError(
            f"{name} must be increasing integer fine-edge indices spanning 0..{n_fine}"
        )
    return tuple(int(v) for v in values)


def _digest(*arrays):
    result = hashlib.sha256(b"jimgw-k8-interval-allocation-v1\0")
    for array in arrays:
        array = np.asarray(array)
        dtype = "<f8" if array.dtype.kind == "f" else "<i8"
        result.update(np.asarray(array.shape, dtype="<i8").tobytes())
        result.update(np.asarray(array, dtype=dtype).tobytes())
    return result.hexdigest()


@dataclass(frozen=True)
class SparseAllocationNodes:
    """One bank layout; index arrays gather exact, unrounded fit frequencies."""

    frequencies: np.ndarray
    interval_node_indices: np.ndarray
    reference_node_indices: np.ndarray
    check_indices: np.ndarray


@dataclass(frozen=True)
class IntervalCatalogue:
    """Exact fine-edge intervals in child-before-parent order.

    ``roots`` cover the domain and define the searchable forest. Singletons and
    explicit comparator intervals need not be reachable from these roots.
    A terminal interval can span multiple fine bins: its moments can be formed
    directly from ``fine_interval_indices[start:stop]``. Other tree intervals
    have two children whose affine moment translations can be cached bottom-up.
    """

    reference_edges: np.ndarray
    intervals: np.ndarray
    children: np.ndarray
    roots: tuple[int, ...]
    fine_interval_indices: np.ndarray
    explicit_partitions: tuple[tuple[int, ...], ...]
    digest: str

    @property
    def n_intervals(self):
        return len(self.intervals)

    @property
    def order(self):
        return 8

    def interval_nodes(self):
        edges = self.reference_edges[self.intervals]
        return (edges[:, 0] + edges[:, 1])[:, None] / 2 + (
            (edges[:, 1] - edges[:, 0])[:, None] / 2 * legendre_lobatto_nodes(8)
        )

    def sparse_nodes(self, *, max_frequencies=1_000_000):
        """Union all fit nodes, fine Lobatto nodes, and independent Gauss23 checks.

        No approximate frequency deduplication is performed. Fine Lobatto nodes
        are already present through the singleton intervals. The caller invokes
        each parameter bank once on ``frequencies``; waveform prefix/gauge and
        physical-support handling remain the caller's responsibility.
        """
        max_frequencies = _positive_integer(max_frequencies, "max_frequencies")
        nodes = self.interval_nodes()
        checks, _ = np.polynomial.legendre.leggauss(23)
        edges = self.reference_edges
        check_nodes = (edges[:-1] + edges[1:])[:, None] / 2 + (
            np.diff(edges)[:, None] / 2 * checks
        )
        frequencies, inverse = np.unique(
            np.concatenate((nodes.ravel(), check_nodes.ravel())), return_inverse=True
        )
        if len(frequencies) > max_frequencies:
            raise ValueError("allocation sparse frequency budget exceeded")
        node_indices = inverse[: nodes.size].reshape(nodes.shape)
        return SparseAllocationNodes(
            _readonly(frequencies),
            _readonly(node_indices),
            _readonly(node_indices[self.fine_interval_indices]),
            _readonly(inverse[nodes.size :]),
        )


def build_interval_catalogue(
    reference_edges,
    incumbent_edge_indices,
    *,
    extra_partitions=(),
    refinement_depth=1,
    root_edge_indices=None,
    max_intervals=8192,
):
    """Merge incumbent bins and split within them to a bounded depth.

    The incumbent is always reachable by tree pruning. Extra partitions are
    retained as explicit comparators even if they use shifted merges outside
    that tree. Roots can enforce mandatory boundaries from the incumbent.
    All fine singleton intervals are included for independent reference fits
    and moment aggregation, without extending the searchable refinement depth.
    """
    edges = _edges(reference_edges)
    n_fine = len(edges) - 1
    incumbent = _partition(incumbent_edge_indices, n_fine, "incumbent")
    depth = _positive_integer(refinement_depth, "refinement_depth", minimum=0)
    limit = _positive_integer(max_intervals, "max_intervals")
    roots_edges = _partition(
        (0, n_fine) if root_edge_indices is None else root_edge_indices,
        n_fine,
        "roots",
    )
    if not set(roots_edges).issubset(incumbent):
        raise ValueError("root boundaries must be incumbent boundaries")
    extras = sorted(
        {_partition(values, n_fine, "extra partition") for values in extra_partitions}
        - {incumbent},
        key=lambda p: (len(p), p),
    )
    if any(not set(roots_edges).issubset(p) for p in extras):
        raise ValueError("explicit partitions must preserve mandatory root boundaries")
    explicit = (incumbent, *extras)
    tree = {}

    def insert(pair, children=None):
        if pair not in tree:
            tree[pair] = children
            if len(tree) > limit:
                raise ValueError("allocation interval budget exceeded")
        elif children is not None:
            tree[pair] = children

    def split(lo, hi, remaining):
        pair = (lo, hi)
        if remaining and hi - lo > 1:
            mid = (lo + hi) // 2
            children = ((lo, mid), (mid, hi))
            insert(pair, children)
            split(lo, mid, remaining - 1)
            split(mid, hi, remaining - 1)
        else:
            insert(pair)

    def merge(first, last):
        pair = (incumbent[first], incumbent[last])
        if last - first == 1:
            split(*pair, depth)
        else:
            mid = (first + last) // 2
            children = (
                (incumbent[first], incumbent[mid]),
                (incumbent[mid], incumbent[last]),
            )
            insert(pair, children)
            merge(first, mid)
            merge(mid, last)

    positions = {value: index for index, value in enumerate(incumbent)}
    for lo, hi in pairwise(roots_edges):
        merge(positions[lo], positions[hi])
    for index in range(n_fine):
        insert((index, index + 1))
    for partition in explicit:
        for pair in pairwise(partition):
            insert(pair)
    pairs = sorted(tree, key=lambda p: (p[1] - p[0], p[0]))
    lookup = {pair: index for index, pair in enumerate(pairs)}
    children = np.full((len(pairs), 2), -1, dtype=np.int64)
    for index, pair in enumerate(pairs):
        if tree[pair] is not None:
            children[index] = [lookup[p] for p in tree[pair]]
    roots = tuple(lookup[p] for p in pairwise(roots_edges))
    fine_indices = np.asarray([lookup[(i, i + 1)] for i in range(n_fine)])
    digest = _digest(edges, pairs, children, roots, *explicit)
    return IntervalCatalogue(
        edges,
        _readonly(np.asarray(pairs, dtype=np.int64)),
        _readonly(children),
        roots,
        _readonly(fine_indices),
        explicit,
        digest,
    )


@dataclass(frozen=True)
class AllocatedPartition:
    """A training proposal, without a likelihood-accuracy certificate."""

    edge_indices: tuple[int, ...]
    interval_indices: tuple[int, ...]
    edges: np.ndarray
    cost: float
    additive_bound: float
    case_bounds: np.ndarray
    delta_z: np.ndarray
    delta_q: np.ndarray
    origin: str
    digest: str

    @property
    def n_bins(self):
        return len(self.interval_indices)


def _statistics(catalogue, delta_z, delta_q, interval_cost, reference_case):
    z = np.asarray(delta_z, dtype=np.complex128)
    q_input = np.asarray(delta_q)
    if np.iscomplexobj(q_input):
        raise ValueError("delta_q must be real")
    q = np.asarray(q_input, dtype=np.float64)
    cost = np.asarray(interval_cost, dtype=np.float64)
    if (
        z.ndim != 2
        or z.shape[0] == 0
        or z.shape[1] != catalogue.n_intervals
        or q.shape != z.shape
        or cost.shape != (catalogue.n_intervals,)
        or np.any(~np.isfinite(z))
        or np.any(~np.isfinite(q))
        or np.any(~np.isfinite(cost))
        or np.any(cost < 0)
    ):
        raise ValueError(
            "finite delta_z/delta_q must have shape (cases, intervals), "
            "with nonnegative finite interval_cost of shape (intervals,)"
        )
    ref = _positive_integer(reference_case, "reference_case", minimum=0)
    if ref >= len(z):
        raise ValueError("reference_case is outside the parameter bank")
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.abs(z) + 0.5 * np.abs(q)
        bound = raw + raw[ref]
    if np.any(~np.isfinite(bound)):
        raise ValueError("interval error bound overflowed")
    return z, q, cost, bound, np.max(bound, axis=0)


def _tree_partition(catalogue, error, cost, multiplier):
    values = np.zeros((catalogue.n_intervals, 2))
    split = np.zeros(catalogue.n_intervals, dtype=bool)
    for t, (left, right) in enumerate(catalogue.children):
        if np.isinf(multiplier):
            leaf = (error[t], cost[t])
        else:
            leaf = (cost[t] + multiplier * error[t], error[t])
        chosen = leaf
        if left >= 0:
            branch = tuple(values[left] + values[right])
            if branch < leaf:
                chosen = branch
                split[t] = True
        values[t] = chosen
    result, stack = [], list(reversed(catalogue.roots))
    while stack:
        t = stack.pop()
        if split[t]:
            stack.extend(reversed(catalogue.children[t].tolist()))
        else:
            result.append(t)
    return tuple(result)


def _combine_budgets(left, right, cap):
    """Minimize additive error at each exact leaf count, breaking ties on cost."""
    count = min(cap, len(left[0]) + len(right[0]) - 2)
    err = np.full(count + 1, np.inf)
    cost = np.full(count + 1, np.inf)
    choice = np.full(count + 1, -1, dtype=np.int64)
    for k_left in range(1, len(left[0])):
        n_right = min(len(right[0]) - 1, count - k_left)
        if n_right < 1 or not np.isfinite(left[0][k_left]):
            continue
        targets = np.arange(1, n_right + 1) + k_left
        errors = left[0][k_left] + right[0][1 : n_right + 1]
        costs = left[1][k_left] + right[1][1 : n_right + 1]
        better = (errors < err[targets]) | (
            (errors == err[targets]) & (costs < cost[targets])
        )
        selected = targets[better]
        err[selected], cost[selected] = errors[better], costs[better]
        choice[selected] = k_left
    return err, cost, choice


def _budget_partitions(catalogue, error, cost, counts, cap):
    reachable = set()
    stack = list(catalogue.roots)
    while stack:
        t = stack.pop()
        reachable.add(t)
        if catalogue.children[t, 0] >= 0:
            stack.extend(catalogue.children[t].tolist())
    tables = {}
    for t in sorted(reachable):
        left, right = catalogue.children[t]
        if left < 0:
            table = (
                np.asarray([np.inf, error[t]]),
                np.asarray([np.inf, cost[t]]),
                np.asarray([-1, -1]),
            )
        else:
            table = _combine_budgets(tables[left], tables[right], cap)
            table[0][1], table[1][1], table[2][1] = error[t], cost[t], -1
        tables[t] = table
    forest = [tables[catalogue.roots[0]]]
    for root in catalogue.roots[1:]:
        forest.append(_combine_budgets(forest[-1], tables[root], cap))

    def restore_node(t, count):
        if count == 1:
            return [t]
        left, right = catalogue.children[t]
        k_left = int(tables[t][2][count])
        return restore_node(left, k_left) + restore_node(right, count - k_left)

    result = []
    for count in counts:
        if count >= len(forest[-1][0]) or not np.isfinite(forest[-1][0][count]):
            continue
        pieces, remaining = [], count
        for stage in range(len(catalogue.roots) - 1, 0, -1):
            k_left = int(forest[stage][2][remaining])
            pieces.append(restore_node(catalogue.roots[stage], remaining - k_left))
            remaining = k_left
        pieces.append(restore_node(catalogue.roots[0], remaining))
        result.append((count, tuple(t for p in reversed(pieces) for t in p)))
    return result


def allocate_partitions(
    catalogue,
    delta_z,
    delta_q,
    interval_cost,
    *,
    error_budget,
    reference_case=0,
    max_candidates=64,
    multipliers=None,
    budget_counts=None,
    max_budget_bins=512,
    max_local_edits=32,
):
    """Propose bounded layouts using only precomputed training statistics.

    Inputs ``delta_z`` and ``delta_q`` have shape (cases, intervals), relative
    to the SAME fine composed operator. The scalar bound sums the per-interval
    maximum of |delta_z| + |delta_q|/2 plus the reference-case contribution.
    Bounds do not certify the native reference or unsampled parameters.

    Multiplier pruning optimizes cost + lambda*bound on the declared tree.
    An exact-leaf-count DP also proposes unsupported discrete tradeoffs; it
    minimizes bound with cost as tie-breaker, not a constrained GPU objective.
    One-edit alternatives can cross tree boundaries if their intervals are in
    the catalogue. Explicit incumbents are retained even when their bound fails.

    ``max_candidates`` bounds returned training proposals, NOT compiled
    finalists. The caller can score all complete likelihoods and compile only
    its three most promising training-valid layouts. No verification input is
    accepted here, and a failed bound is never treated as a likelihood failure.
    """
    maximum = _positive_integer(max_candidates, "max_candidates")
    cap = _positive_integer(max_budget_bins, "max_budget_bins")
    edit_limit = _positive_integer(max_local_edits, "max_local_edits", minimum=0)
    if len(catalogue.explicit_partitions) > maximum:
        raise ValueError("max_candidates cannot discard explicit incumbent partitions")
    if not np.isfinite(error_budget) or error_budget <= 0:
        raise ValueError("error_budget must be positive and finite")
    z, q, cost, case_bound, error = _statistics(
        catalogue, delta_z, delta_q, interval_cost, reference_case
    )
    lookup = {tuple(pair): t for t, pair in enumerate(catalogue.intervals)}
    proposals = {}

    def add(ids, origin):
        ids = tuple(int(t) for t in ids)
        pairs = catalogue.intervals[list(ids)]
        edges = tuple(int(v) for v in (*pairs[:, 0], pairs[-1, 1]))
        if edges in proposals:
            return
        proposals[edges] = AllocatedPartition(
            edges,
            ids,
            _readonly(catalogue.reference_edges[list(edges)]),
            float(np.sum(cost[list(ids)])),
            float(np.sum(error[list(ids)])),
            _readonly(np.sum(case_bound[:, ids], axis=1)),
            _readonly(np.sum(z[:, ids], axis=1)),
            _readonly(np.sum(q[:, ids], axis=1)),
            origin,
            _digest(catalogue.reference_edges, edges, [8]),
        )

    for i, edges in enumerate(catalogue.explicit_partitions):
        add(
            tuple(lookup[p] for p in pairwise(edges)),
            "incumbent" if i == 0 else "explicit",
        )
    if multipliers is None:
        positive = error > 0
        scale = (
            float(np.median(cost[positive]) / np.median(error[positive]))
            if np.any(positive)
            else 1.0
        )
        scale = min(max(scale, 1e-100), 1e100)
        multipliers = (0.0, *(scale * np.logspace(-6, 6, 25)), np.inf)
    else:
        multipliers = tuple(multipliers)
        if len(multipliers) > 64:
            raise ValueError("at most 64 multipliers are supported")
        if any(np.isnan(v) or v < 0 for v in multipliers):
            raise ValueError("multipliers must be nonnegative and not NaN")
    for multiplier in multipliers:
        add(_tree_partition(catalogue, error, cost, multiplier), "multiplier")
    if budget_counts is None:
        counts = set(
            np.linspace(1, min(cap, len(catalogue.reference_edges) - 1), 32).astype(int)
        )
        counts.update(
            len(p) - 1 for p in catalogue.explicit_partitions if len(p) - 1 <= cap
        )
    else:
        counts = {_positive_integer(v, "budget count") for v in budget_counts}
        if len(counts) > 64 or any(v > cap for v in counts):
            raise ValueError(
                "at most 64 budget counts within max_budget_bins are supported"
            )
    if counts:
        for _, ids in _budget_partitions(
            catalogue, error, cost, sorted(counts), max(counts)
        ):
            add(ids, "leaf_budget")

    # Bound the local stage before materializing full case-by-partition sums.
    seeds = list(proposals.values())[: min(3, len(proposals))]
    alternatives = {}
    for seed in seeds:
        ids = seed.interval_indices
        for i, t in enumerate(ids):
            left, right = catalogue.children[t]
            if left >= 0:
                new = ids[:i] + (int(left), int(right)) + ids[i + 1 :]
                alternatives[new] = (
                    seed.additive_bound - error[t] + error[left] + error[right],
                    seed.cost - cost[t] + cost[left] + cost[right],
                )
            if i + 1 < len(ids):
                pair = (
                    int(catalogue.intervals[t, 0]),
                    int(catalogue.intervals[ids[i + 1], 1]),
                )
                merged = lookup.get(pair)
                if merged is not None:
                    # The merged interval must not cross a mandatory root edge.
                    if any(
                        pair[0] < catalogue.intervals[r, 0] < pair[1]
                        for r in catalogue.roots[1:]
                    ):
                        continue
                    new = ids[:i] + (merged,) + ids[i + 2 :]
                    alternatives[new] = (
                        seed.additive_bound
                        - error[t]
                        - error[ids[i + 1]]
                        + error[merged],
                        seed.cost - cost[t] - cost[ids[i + 1]] + cost[merged],
                    )
    ranked_edits = sorted(alternatives, key=lambda p: (*alternatives[p], p))
    for ids in ranked_edits[:edit_limit]:
        add(ids, "local_edit")

    reserved = [proposals[p] for p in catalogue.explicit_partitions]
    remaining = [
        p
        for p in proposals.values()
        if p.edge_indices not in catalogue.explicit_partitions
    ]
    # Keep cheap, tight-bound, and near-feasible choices; a loose bound must not
    # silently remove all potentially useful cancellation-valid proposals.
    rankings = [
        sorted(
            remaining,
            key=lambda p: (
                p.additive_bound > error_budget,
                p.cost,
                p.additive_bound,
                p.edge_indices,
            ),
        ),
        sorted(remaining, key=lambda p: (p.additive_bound, p.cost, p.edge_indices)),
        sorted(remaining, key=lambda p: (p.cost, p.additive_bound, p.edge_indices)),
    ]
    selected = {p.edge_indices: p for p in reserved}
    for choices in zip(*rankings, strict=True):
        for proposal in choices:
            if len(selected) >= maximum:
                break
            selected.setdefault(proposal.edge_indices, proposal)
    return tuple(
        sorted(
            selected.values(), key=lambda p: (p.cost, p.additive_bound, p.edge_indices)
        )
    )
