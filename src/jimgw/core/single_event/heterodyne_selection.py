"""Select a fixed heterodyne grid from one shared sparse ratio bank.

This is an offline empirical selection engine, not a whole-prior certificate.
It never reads native data, constructs a likelihood, or compiles an evaluator.
Callers supply independent training and verification parameter banks. A native
moment oracle can reuse one fine summary bank for every candidate, translating
its moments exactly onto each candidate's coarser edges.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def legendre_lobatto_nodes(order: int) -> np.ndarray:
    """Use the production evaluator's degree-K Legendre-Lobatto node rule."""
    if (
        isinstance(order, bool)
        or not isinstance(order, (int, np.integer))
        or not 1 <= order <= 8
    ):
        raise ValueError("order must be an integer between 1 and 8")
    if order == 1:
        return np.asarray([-1.0, 1.0])
    legendre = np.zeros(order + 1)
    legendre[order] = 1.0
    interior = np.polynomial.legendre.legroots(np.polynomial.legendre.legder(legendre))
    return np.concatenate(([-1.0], np.sort(np.real(interior)), [1.0]))


def _edges(values):
    result = np.array(values, dtype=float, copy=True)
    if (
        result.ndim != 1
        or len(result) < 2
        or np.any(~np.isfinite(result))
        or np.any(np.diff(result) <= 0)
    ):
        raise ValueError("reference edges must be finite and strictly increasing")
    result.flags.writeable = False
    return result


def _readonly(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FrozenGridPlan:
    """A candidate whose boundaries are a subset of the reference boundaries."""

    edges: np.ndarray
    order: int
    cost: float
    node_indices: np.ndarray | None = None
    check_indices: np.ndarray | None = None
    reference_node_indices: np.ndarray | None = None

    @property
    def n_bins(self):
        return len(self.edges) - 1

    @property
    def nodes(self):
        centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        half_widths = 0.5 * np.diff(self.edges)
        return centres[:, None] + half_widths[:, None] * legendre_lobatto_nodes(
            self.order
        )


@dataclass(frozen=True)
class FrozenGridSelection:
    plan: FrozenGridPlan
    training_error: np.ndarray
    verification_error: np.ndarray
    training_residual: np.ndarray | None
    verification_residual: np.ndarray | None
    candidate_diagnostics: tuple[dict, ...]
    sparse_frequency_count: int
    error_metric: str
    frequencies: np.ndarray
    cost_metric: str


def _ratio_bank(oracle, frequencies, name):
    values = np.asarray(oracle(frequencies))
    if (
        values.ndim < 2
        or any(size == 0 for size in values.shape[:-1])
        or values.shape[-1] != len(frequencies)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError(f"{name} ratio must return finite (cases, ..., frequencies)")
    return values


def _weights(oracle, frequencies, name):
    values = np.asarray(oracle(frequencies))
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} weights must be finite and nonnegative")
    return values


def _errors(values, n_cases):
    errors = np.asarray(values, dtype=float)
    if errors.shape != (n_cases,) or np.any(~np.isfinite(errors)):
        raise ValueError("likelihood error oracle must return finite (cases,) errors")
    return np.abs(errors)


def select_frozen_grid(
    reference_edges,
    training_ratio,
    verification_ratio,
    data_weight=None,
    norm_weight=None,
    *,
    orders=(8,),
    bin_counts=None,
    edge_index_candidates=None,
    max_bins=4096,
    error_budget,
    likelihood_error=None,
    cost=None,
    validated_cost=None,
):
    """Choose the cheapest training-valid grid, then verify it independently.

    ``training_ratio(f)`` and ``verification_ratio(f)`` return complex arrays
    shaped ``(cases, ..., frequencies)`` with the rigid time phasor removed;
    optional axes can represent detectors. Each
    oracle is called once, on the SAME sorted union of candidate fit nodes and
    independent Gauss check nodes on the supplied fine/reference panels. Thus
    candidate search does not repeat waveform calls or native-data passes.
    The verification oracle is called only after freezing a training winner;
    failure raises instead of selecting against the held-out bank.

    Candidate edges subsample reference boundaries evenly by panel index.
    Supply counts explicitly to control the bounded search. By default counts
    double from one through ``min(max_bins, reference_panel_count)``, including
    that final count; explicit counts also include the final count. The return
    is cheapest among these supplied candidates,
    not an assertion of global minimum GPU latency.

    ``edge_index_candidates`` instead supplies explicit nonuniform partitions
    as integer indices into ``reference_edges``. Each must span the complete
    reference domain and respect ``max_bins``. It replaces the count family;
    no extra candidate is inserted. It cannot be combined with ``bin_counts``.
    The shared fit/check bank and freeze-before-verification contract are the
    same for either family.

    Without ``likelihood_error``, the empirical gate is the fixed-node integral
    of ``data_weight*abs(e) + norm_weight*(abs(r)*abs(e)+abs(e)**2/2)``.
    Weights exclude quadrature factors; defaults are one. They must broadcast
    to both banks' ``(cases, ..., check_frequencies)`` shape. Contributions from
    any extra axes are summed per case. Gauss integration,
    phasor truncation, fine-reference accuracy, and unsampled cases are not
    certified by this estimate.

    Optionally ``likelihood_error(plan, node_ratios, bank_name)`` returns signed
    per-case log-likelihood errors against a shared precomputed reference.
    ``node_ratios`` has shape ``(cases, ..., bins, order+1)``; ``bank_name`` is
    ``training`` or ``verification``. This oracle becomes the acceptance gate
    and weighted residuals remain diagnostics when weights were supplied;
    otherwise residual estimation is omitted. It must not regenerate native
    data or compile candidate likelihoods. All detector contractions, phase
    marginalization and the error reference convention belong to this oracle.

    ``cost(plan)`` may provide a measured or modeled nonnegative work cost. It
    runs after loading the training bank, so a cached ratio oracle can expose
    those values to a timing callback without further waveform evaluation.
    The default proxy counts node work and polynomial coefficient-pair work.
    If ``validated_cost(plan)`` is supplied, every candidate is screened on
    training errors first and this callback runs only for passing candidates.
    Search then chooses the minimum validated cost across all passing grids,
    before independently verifying that one winner. The callback may compile
    and time an actual sparse evaluator; its returned cost must exclude setup
    and compilation. Record that separate overhead in the caller's diagnostics.
    Plans expose ``node_indices`` (bins, order+1), ``reference_node_indices``
    (reference_bins, order+1) and one shared vector ``check_indices`` into the
    sparse bank. The selection returns the bank's sorted ``frequencies``.
    These allow a cached oracle to reuse shared values without further calls.
    Exact likelihood checks should establish the winning grid before sampling;
    this function never adapts a running likelihood.
    """
    edges = _edges(reference_edges)
    if (
        isinstance(max_bins, bool)
        or not isinstance(max_bins, (int, np.integer))
        or max_bins < 1
    ):
        raise ValueError("max_bins must be a positive integer")
    if not np.isfinite(error_budget) or error_budget <= 0:
        raise ValueError("error_budget must be finite and positive")
    orders = tuple(orders)
    if not orders:
        raise ValueError("at least one interpolation order is required")
    for order in orders:
        legendre_lobatto_nodes(order)
    orders = tuple(sorted(set(orders)))
    limit = min(max_bins, len(edges) - 1)
    if edge_index_candidates is not None:
        if bin_counts is not None:
            raise ValueError("edge_index_candidates cannot be combined with bin_counts")
        partitions = []
        for candidate in edge_index_candidates:
            indices = np.asarray(candidate)
            if (
                indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or not 2 <= len(indices) <= limit + 1
                or indices[0] != 0
                or indices[-1] != len(edges) - 1
                or np.any(indices[1:] <= indices[:-1])
            ):
                raise ValueError(
                    "edge_index_candidates must span the reference with increasing "
                    "integer indices and respect max_bins"
                )
            partitions.append(tuple(int(v) for v in indices))
        if not partitions:
            raise ValueError("edge_index_candidates must not be empty")
        partitions = sorted(set(partitions), key=lambda p: (len(p), p))
        counts = [len(p) - 1 for p in partitions]
    elif bin_counts is None:
        counts = [1]
        while counts[-1] < limit:
            counts.append(min(2 * counts[-1], limit))
    else:
        counts = list(bin_counts)
    if not counts or any(
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or not 1 <= n <= limit
        for n in counts
    ):
        raise ValueError("bin_counts must be positive and within max_bins/reference")
    if edge_index_candidates is None:
        counts.append(limit)
        partitions = [
            tuple(np.arange(count + 1, dtype=np.int64) * (len(edges) - 1) // count)
            for count in sorted(set(counts))
        ]

    candidates = []
    for indices in partitions:
        count = len(indices) - 1
        selected_edges = _readonly(edges[list(indices)])
        for order in orders:
            proxy = float(count * ((order + 1) + (order + 1) ** 2))
            plan = FrozenGridPlan(selected_edges, int(order), proxy)
            candidates.append(plan)

    # Checks are fixed by the reference grid, independent of candidate fits.
    checks, gauss_weights = np.polynomial.legendre.leggauss(2 * max(orders) + 7)
    check_f = (
        0.5 * (edges[:-1] + edges[1:])[:, None] + 0.5 * np.diff(edges)[:, None] * checks
    ).ravel()
    quadrature = (0.5 * np.diff(edges)[:, None] * gauss_weights).ravel()
    reference_nodes = {
        order: FrozenGridPlan(edges, order, 0.0).nodes for order in orders
    }
    frequencies = _readonly(
        np.unique(
            np.concatenate(
                [
                    check_f,
                    *(p.nodes.ravel() for p in candidates),
                    *(nodes.ravel() for nodes in reference_nodes.values()),
                ]
            )
        )
    )
    check_indices = np.searchsorted(frequencies, check_f)
    candidates = [
        replace(
            plan,
            node_indices=_readonly(np.searchsorted(frequencies, plan.nodes)),
            check_indices=_readonly(check_indices),
            reference_node_indices=_readonly(
                np.searchsorted(frequencies, reference_nodes[plan.order])
            ),
        )
        for plan in candidates
    ]
    estimate_residual = (
        likelihood_error is None or data_weight is not None or norm_weight is not None
    )
    dw = nw = None
    if estimate_residual:
        dw = _weights(data_weight or np.ones_like, check_f, "data")
        nw = _weights(norm_weight or np.ones_like, check_f, "norm")
    training = _ratio_bank(training_ratio, frequencies, "training")
    if cost is not None:
        priced = []
        for plan in candidates:
            measured = float(cost(plan))
            if not np.isfinite(measured) or measured < 0:
                raise ValueError("candidate cost must be finite and nonnegative")
            priced.append(replace(plan, cost=measured))
        candidates = priced
    candidates.sort(key=lambda p: (p.cost, p.n_bins, p.order))

    def measure(plan, bank, bank_name):
        values = bank[..., plan.node_indices]
        if not estimate_residual:
            return _errors(
                likelihood_error(plan, values, bank_name), bank.shape[0]
            ), None
        inverse = np.linalg.inv(
            np.polynomial.polynomial.polyvander(
                legendre_lobatto_nodes(plan.order), plan.order
            )
        )
        coefficients = np.einsum("ij,...bj->...bi", inverse, values)
        bins = np.searchsorted(plan.edges, check_f, side="right") - 1
        u = (2 * check_f - plan.edges[bins] - plan.edges[bins + 1]) / (
            plan.edges[bins + 1] - plan.edges[bins]
        )
        predictions = np.einsum(
            "...fk,fk->...f",
            coefficients[..., bins, :],
            np.polynomial.polynomial.polyvander(u, plan.order),
        )
        truth = bank[..., check_indices]
        residual = np.abs(predictions - truth)
        integrand = dw * residual + nw * (np.abs(truth) * residual + residual**2 / 2)
        estimate = np.sum(integrand * quadrature, axis=-1)
        if estimate.ndim > 1:
            estimate = estimate.sum(axis=tuple(range(1, estimate.ndim)))
        if np.any(~np.isfinite(estimate)):
            raise ValueError("interpolation residual estimate overflowed")
        errors = (
            estimate
            if likelihood_error is None
            else _errors(likelihood_error(plan, values, bank_name), bank.shape[0])
        )
        return errors, estimate

    diagnostics = []
    winner = None
    for plan in candidates:
        errors, residual = measure(plan, training, "training")
        passed = bool(np.max(errors) <= error_budget)
        diagnostic = {
            "n_bins": plan.n_bins,
            "edge_indices": np.searchsorted(edges, plan.edges).tolist(),
            "order": plan.order,
            "cost": plan.cost,
            "maximum_training_error": float(np.max(errors)),
            "maximum_training_residual": (
                None if residual is None else float(np.max(residual))
            ),
            "passed": passed,
        }
        diagnostics.append(diagnostic)
        if passed:
            if validated_cost is None:
                winner = plan, errors, residual
                break
            measured = float(validated_cost(plan))
            if not np.isfinite(measured) or measured < 0:
                raise ValueError(
                    "validated candidate cost must be finite and nonnegative"
                )
            diagnostic["validated_cost"] = measured
            measured_plan = replace(plan, cost=measured)
            if winner is None or (measured, plan.n_bins, plan.order) < (
                winner[0].cost,
                winner[0].n_bins,
                winner[0].order,
            ):
                winner = measured_plan, errors, residual
    if winner is None:
        raise RuntimeError(
            f"bin budget exhausted: no candidate through {max(counts)} bins "
            f"meets empirical training error target {error_budget:.6g}"
        )
    plan, training_error, training_residual = winner
    verification = _ratio_bank(verification_ratio, frequencies, "verification")
    verification_error, verification_residual = measure(
        plan, verification, "verification"
    )
    if np.max(verification_error) > error_budget:
        raise RuntimeError(
            "frozen grid failed independent verification: "
            f"error {np.max(verification_error):.6g}, target {error_budget:.6g}; "
            "the verification bank was not used to retune the grid"
        )
    return FrozenGridSelection(
        plan,
        _readonly(training_error),
        _readonly(verification_error),
        None if training_residual is None else _readonly(training_residual),
        None if verification_residual is None else _readonly(verification_residual),
        tuple(diagnostics),
        len(frequencies),
        "shared_reference_likelihood"
        if likelihood_error
        else "weighted_ratio_residual",
        frequencies,
        "validated_cost"
        if validated_cost
        else ("provided_cost" if cost else "analytical_work_proxy"),
    )
