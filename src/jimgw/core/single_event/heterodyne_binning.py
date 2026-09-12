"""Offline likelihood-error-aware allocation of a frozen polynomial bin grid.

The estimate is empirical on supplied training ratios and check nodes; it is
not a rigorous whole-prior error bound. Qualify the resulting fixed grid on
independent cases before sampling. No adaptation takes place inside a chain.
"""

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RatioBinPlan:
    edges: np.ndarray
    estimated_error: np.ndarray
    evaluations: int


def allocate_ratio_bins(
    ratio,
    data_weight,
    norm_weight,
    initial_edges,
    *,
    order,
    error_budget,
    max_bins=4096,
):
    """Greedily split bins with the largest estimated likelihood contribution.

    ``ratio(f)`` returns (cases, frequencies) after removing the rigid time
    phasor. Weights are 4*abs(data*reference)/PSD and 4*abs(reference)**2/PSD;
    they may broadcast over cases. A check-node residual e estimates
    integral(data_weight*abs(e) + norm_weight*(abs(r)*abs(e)+abs(e)**2/2)).
    This controls complex overlap and norm errors conservatively at those
    nodes, including a phase-marginalized likelihood's Lipschitz response.
    Taylor phasor truncation and summary quadrature require separate checks.
    """
    edges = np.asarray(initial_edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("initial edges must be finite and strictly increasing")
    if isinstance(order, bool) or not isinstance(order, int) or not 2 <= order <= 8:
        raise ValueError("order must be an integer between 2 and 8")
    if not np.isfinite(error_budget) or error_budget <= 0:
        raise ValueError("error_budget must be finite and positive")
    if (
        isinstance(max_bins, bool)
        or not isinstance(max_bins, int)
        or max_bins < len(edges) - 1
    ):
        raise ValueError("max_bins must cover the initial grid")
    nodes = -np.cos(np.arange(order + 1) * np.pi / order)
    checks, weights = np.polynomial.legendre.leggauss(2 * order + 7)
    inverse = np.linalg.inv(np.polynomial.polynomial.polyvander(nodes, order))
    evaluate_polynomial = np.polynomial.polynomial.polyvander(checks, order)
    evaluations = 0

    def measure(panels):
        nonlocal evaluations
        bounds = np.asarray(panels)
        lo, hi = bounds[:, 0], bounds[:, 1]
        coordinates = np.concatenate((nodes, checks))
        f = (lo[:, None] + hi[:, None]) / 2 + (hi - lo)[:, None] / 2 * coordinates
        values = np.asarray(ratio(f.ravel()))
        if values.ndim != 2 or values.shape[1] != f.size:
            raise ValueError("ratio must return (cases, frequencies)")
        values = values.reshape(values.shape[0], *f.shape)
        predicted = np.einsum(
            "nk,km,cbm->cbn", evaluate_polynomial, inverse, values[:, :, : order + 1]
        )
        truth = values[:, :, order + 1 :]
        check_f = f[:, order + 1 :]
        dw = np.broadcast_to(
            np.asarray(data_weight(check_f.ravel())), (values.shape[0], check_f.size)
        ).reshape(truth.shape)
        nw = np.broadcast_to(
            np.asarray(norm_weight(check_f.ravel())), (values.shape[0], check_f.size)
        ).reshape(truth.shape)
        if (
            np.any(~np.isfinite(values))
            or np.any(~np.isfinite(dw))
            or np.any(~np.isfinite(nw))
            or np.any(dw < 0)
            or np.any(nw < 0)
        ):
            raise ValueError("ratios and nonnegative weights must be finite")
        residual = abs(predicted - truth)
        error = dw * residual + nw * (abs(truth) * residual + 0.5 * residual**2)
        if np.any(~np.isfinite(error)):
            raise ValueError("likelihood error estimate overflowed")
        evaluations += f.size
        return np.sum(error * weights, axis=-1) * (hi - lo) / 2

    panels = list(itertools.pairwise(edges))
    errors = measure(panels)
    while np.max(errors.sum(axis=1)) > error_budget:
        if len(panels) >= max_bins:
            raise RuntimeError(
                f"bin budget exhausted before the training error target: "
                f"estimated {np.max(errors.sum(axis=1)):.6g}, target {error_budget:.6g}"
            )
        worst = int(np.argmax(errors.sum(axis=1)))
        count = min(8, max_bins - len(panels), len(panels))
        selected = np.argsort(errors[worst])[-count:]
        keep = np.ones(len(panels), dtype=bool)
        keep[selected] = False
        children = []
        for i in selected:
            lo, hi = panels[i]
            mid = (lo + hi) / 2
            if not lo < mid < hi:
                raise RuntimeError("bin refinement reached floating-point resolution")
            children.extend(((lo, mid), (mid, hi)))
        panels = [
            p for p, retain in zip(panels, keep, strict=True) if retain
        ] + children
        errors = np.concatenate((errors[:, keep], measure(children)), axis=1)
    order_by_frequency = np.argsort([p[0] for p in panels])
    frozen_edges = np.array([panels[i][0] for i in order_by_frequency] + [edges[-1]])
    return RatioBinPlan(frozen_edges, errors.sum(axis=1), evaluations)
