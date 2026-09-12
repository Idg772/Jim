"""Native-moment adapter for bounded nonuniform pre-sampling allocation.

The interval search performs no waveform calls, native reads or compilation.
Every trial reuses signed interval statistics. Only the training shortlist is
timed, and independent verification is consumed after freezing one winner.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from itertools import pairwise

import numpy as np

from jimgw.core.single_event.heterodyne_allocation import (
    allocate_partitions,
    build_interval_catalogue,
)
from jimgw.core.single_event.heterodyne_selection import (
    FrozenGridPlan,
    FrozenGridSelection,
)


def catalogue_moments(catalogue, values):
    """Affine native-moment aggregation, batched by interval span.

    Tree parents reuse their two children. Terminal/comparator intervals use
    their exact native fine subintervals. No global-frequency power basis or
    native data is introduced. Leading detector/anchor axes are preserved.
    """
    values = np.asarray(values)
    edges = catalogue.reference_edges
    intervals = catalogue.intervals
    if values.ndim < 2 or values.shape[-1] != len(edges) - 1:
        raise ValueError("catalogue moments require [..., degree, fine_bin]")
    if not np.issubdtype(values.dtype, np.number) or values.shape[-2] < 1:
        raise ValueError("catalogue moments require a nonempty numeric degree axis")
    if not np.all(np.isfinite(values)):
        raise ValueError("catalogue moments must be finite")
    values = values.astype(np.complex128 if np.iscomplexobj(values) else np.float64)
    output = np.empty((*values.shape[:-1], len(intervals)), dtype=values.dtype)
    output[..., catalogue.fine_interval_indices] = values
    centres = np.mean(edges[intervals], axis=1)
    halves = np.diff(edges[intervals], axis=1)[:, 0] / 2
    spans = intervals[:, 1] - intervals[:, 0]
    degree = values.shape[-2]
    for span in np.unique(spans[spans > 1]):
        parents = np.flatnonzero(spans == span)
        sources = []
        starts = []
        parent_links = []
        for parent in parents:
            starts.append(len(sources))
            children = catalogue.children[parent]
            if children[0] >= 0:
                selected = children
            else:
                lo, hi = intervals[parent]
                selected = catalogue.fine_interval_indices[lo:hi]
            sources.extend(selected)
            parent_links.extend([parent] * len(selected))
        sources, parent_links = np.asarray(sources), np.asarray(parent_links)
        alpha = (centres[sources] - centres[parent_links]) / halves[parent_links]
        beta = halves[sources] / halves[parent_links]
        transform = np.zeros((len(sources), degree, degree))
        for k in range(degree):
            for j in range(k + 1):
                transform[:, k, j] = math.comb(k, j) * alpha ** (k - j) * beta**j
        translated = np.einsum(
            "...jf,fkj->...kf", output[..., sources], transform, optimize=True
        )
        output[..., parents] = np.add.reduceat(translated, starts, axis=-1)
    if not np.all(np.isfinite(output)):
        raise ValueError("nonfinite translated catalogue moments")
    return output


def _frozen_plan(catalogue, nodes, edge_indices, proxy=0.0):
    plan = FrozenGridPlan(
        catalogue.reference_edges[np.asarray(edge_indices)], 8, float(proxy)
    )
    indices = np.searchsorted(nodes.frequencies, plan.nodes)
    if np.any(indices >= len(nodes.frequencies)) or not np.array_equal(
        nodes.frequencies[indices], plan.nodes
    ):
        raise ValueError("candidate nodes are absent from the frozen ratio bank")
    return replace(
        plan,
        node_indices=indices,
        reference_node_indices=nodes.reference_node_indices,
        check_indices=nodes.check_indices,
    )


def select_adaptive_network_grid(fine, bank, oracle, timer, settings):
    """Return selection and diagnostics; holdout is never used for ranking."""
    if fine.interpolation_order != 8:
        raise ValueError(
            "adaptive network allocation currently requires interpolation_order=8"
        )
    started = time.perf_counter()
    n_fine = fine.n_bins
    n_base = min(settings.allocation_base_bins, n_fine)
    base = np.arange(n_base + 1, dtype=np.int64) * n_fine // n_base
    counts = sorted(set([n for n in settings.candidate_bins if n <= n_fine] + [n_fine]))
    explicit = [np.arange(n + 1, dtype=np.int64) * n_fine // n for n in counts]
    explicit.extend(
        np.asarray(p, dtype=np.int64) for p in settings.candidate_edge_indices
    )
    catalogue = build_interval_catalogue(
        fine.freq_grid_edges,
        base,
        extra_partitions=explicit,
        refinement_depth=settings.allocation_refinement_depth,
        max_intervals=settings.allocation_max_intervals,
    )
    nodes = catalogue.sparse_nodes()
    catalogue_seconds = time.perf_counter() - started
    ratios = bank(nodes.frequencies, "training")
    reference_plan = _frozen_plan(catalogue, nodes, np.arange(n_fine + 1))
    # Establish independent Gauss/Lobatto convergence before searching.
    oracle(reference_plan, ratios[..., nodes.reference_node_indices], "training")
    stage = time.perf_counter()
    stored = []
    active = []
    for detector in fine.detectors:
        a = catalogue_moments(catalogue, fine.phasor_data_moments[detector.name])
        b = catalogue_moments(catalogue, fine.summary_moments[detector.name][1])
        stored.append((a, b))
        active.append(np.any(a != 0, axis=(0, 1)) | np.any(b != 0, axis=0))
    moment_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    left, right = catalogue.reference_edges[catalogue.intervals].T
    centres, half = (left + right) / 2, (right - left) / 2
    coefficients = np.einsum(
        "ij,...bj->...bi",
        oracle.inverse,
        ratios[..., nodes.interval_node_indices],
        optimize=True,
    )
    z, norm = oracle.interval_statistics(
        centres, half, coefficients, stored, "training"
    )
    del coefficients, stored
    fine_indices = catalogue.fine_interval_indices
    # Interval deltas use a common fine Lobatto operator. The final gate below
    # uses the independent fine Gauss reconstruction retained by the oracle.
    fine_z, fine_norm = z[:, fine_indices], norm[:, fine_indices]
    prefix_z = np.pad(np.cumsum(fine_z, axis=1), ((0, 0), (1, 0)))
    prefix_norm = np.pad(np.cumsum(fine_norm, axis=1), ((0, 0), (1, 0)))
    lo, hi = catalogue.intervals.T
    delta_z = z - (prefix_z[:, hi] - prefix_z[:, lo])
    delta_norm = norm - (prefix_norm[:, hi] - prefix_norm[:, lo])
    delta_z[:, fine_indices] = 0
    delta_norm[:, fine_indices] = 0
    active_count = np.sum(active, axis=0)
    # Shared source + active response, fit, norm and phasor work. A cost proxy;
    # the actual sparse evaluators decide between the few training finalists.
    cost = 9.0 + active_count * (9.0 + 81.0 + 9.0 * (fine.phasor_moment_order + 1))
    statistics_seconds = time.perf_counter() - stage
    stage = time.perf_counter()
    proposals = allocate_partitions(
        catalogue,
        delta_z,
        delta_norm,
        cost,
        error_budget=settings.tolerance * 0.75,
        max_candidates=settings.allocation_max_proposals,
    )
    search_seconds = time.perf_counter() - stage
    lookup = {tuple(pair): i for i, pair in enumerate(catalogue.intervals)}
    reference = oracle.references["training"]
    records, passing = [], []
    # API proposals carry their fine-edge indices, never rounded frequencies.
    for proposal in proposals:
        edges = proposal.edge_indices
        indices = np.asarray([lookup[(a, b)] for a, b in pairwise(edges)])
        values = oracle.log_likelihood(
            z[:, indices].sum(axis=1), norm[:, indices].sum(axis=1)
        )
        errors = abs((values - values[0]) - (reference - reference[0]))
        plan = _frozen_plan(catalogue, nodes, edges, np.sum(cost[indices]))
        passed = bool(np.max(errors) <= settings.tolerance * 0.75)
        record = {
            "plan_sha256": timer.plan_key(plan),
            "n_bins": plan.n_bins,
            "order": 8,
            "frequency_bin_edges": plan.edges.tolist(),
            "active_detector_entries": int(9 * np.sum(active_count[indices])),
            "source_entries": 9 * plan.n_bins,
            "maximum_training_error_nats": float(np.max(errors)),
            "training_passed": passed,
            "proxy_cost": plan.cost,
            "additive_bound_nats": float(
                np.sum(
                    np.max(
                        abs(delta_z[:, indices])
                        + 0.5 * abs(delta_norm[:, indices])
                        + abs(delta_z[0, indices])
                        + 0.5 * abs(delta_norm[0, indices]),
                        axis=0,
                    )
                )
            ),
        }
        records.append(record)
        if passed:
            passing.append((plan, errors, record))
    if not passing:
        raise RuntimeError("no adaptive partition passed the training likelihood gate")
    passing.sort(key=lambda value: (value[0].cost, float(np.max(value[1]))))
    shortlist = []
    shapes = set()
    for item in passing:
        shape = (item[0].n_bins, item[2]["active_detector_entries"])
        if shape in shapes:
            continue
        shapes.add(shape)
        shortlist.append(item)
        if len(shortlist) == settings.max_timed_candidates:
            break
    timed = []
    for candidate, errors, record in shortlist:
        measured_cost = float(timer(candidate))
        if not np.isfinite(measured_cost) or measured_cost < 0:
            raise ValueError("measured candidate cost must be finite and nonnegative")
        timed.append((measured_cost, candidate, errors, record))
    measured, plan, training_error, winner_record = min(timed, key=lambda item: item[0])
    plan = replace(plan, cost=float(measured))
    # The independent bank is read only after selection and cannot retune it.
    verification = bank(nodes.frequencies, "verification")
    verification_error = np.abs(
        oracle(plan, verification[..., plan.node_indices], "verification")
    )
    if np.max(verification_error) > settings.tolerance * 0.75:
        raise RuntimeError(
            "frozen grid failed independent verification: "
            f"{np.max(verification_error):.6g} nats"
        )
    selection = FrozenGridSelection(
        plan=plan,
        training_error=training_error,
        verification_error=verification_error,
        training_residual=None,
        verification_residual=None,
        candidate_diagnostics=tuple(records),
        sparse_frequency_count=len(nodes.frequencies),
        error_metric="native-moment-centered-log-likelihood",
        frequencies=nodes.frequencies,
        cost_metric="measured callbacks among bounded training-valid nonuniform finalists",
    )
    return selection, {
        "catalogue_sha256": catalogue.digest,
        "intervals": catalogue.n_intervals,
        "catalogue_seconds": catalogue_seconds,
        "moment_translation_seconds": moment_seconds,
        "interval_statistics_seconds": statistics_seconds,
        "tree_search_seconds": search_seconds,
        "proposals": len(proposals),
        "training_valid_proposals": len(passing),
        "compiled_finalists": len(shortlist),
        "selected_layout": winner_record,
        "search_scope": "bounded K8 tree and explicit comparator layouts; additive surrogate proposals, complete coherent likelihood gate",
        "wall_seconds": time.perf_counter() - started,
    }
