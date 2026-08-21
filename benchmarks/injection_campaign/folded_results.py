"""CPU-only result helpers for deterministic quotient-fold expansion."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp

_FOLD_GROUP_ORDER = 8
_NORMALIZATION_TOLERANCE = 1.0e-10


def unfolded_posterior_telemetry(
    input_log_weights: np.ndarray,
    base_log_priors: np.ndarray,
    log_branch_probabilities: np.ndarray,
) -> dict[str, Any]:
    """Summarize a deterministic eight-image posterior expansion.

    The nested-point weights and each row of conditional branch probabilities
    must already be normalized. Prior support is measured independently of the
    likelihood from the base-prior table retained by the unfolding kernel.
    """

    input_log_weights = np.asarray(input_log_weights, dtype=np.float64)
    base_log_priors = np.asarray(base_log_priors, dtype=np.float64)
    log_branch_probabilities = np.asarray(log_branch_probabilities, dtype=np.float64)
    if input_log_weights.ndim != 1:
        raise ValueError("input_log_weights must be one-dimensional")
    if (
        base_log_priors.ndim != 2
        or log_branch_probabilities.ndim != 2
        or base_log_priors.shape != log_branch_probabilities.shape
        or base_log_priors.shape[0] != input_log_weights.size
    ):
        raise ValueError("image tables must have one row per input log weight")
    if base_log_priors.shape[1] != _FOLD_GROUP_ORDER:
        raise ValueError("quotient-fold telemetry requires exactly eight images")
    if input_log_weights.size < 1:
        raise ValueError("quotient-fold telemetry requires at least one point")
    if (
        np.any(np.isnan(input_log_weights))
        or np.any(np.isposinf(input_log_weights))
        or not np.any(np.isfinite(input_log_weights))
    ):
        raise ValueError("input log weights are invalid")
    for name, values in (
        ("base log priors", base_log_priors),
        ("branch log probabilities", log_branch_probabilities),
    ):
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise ValueError(f"{name} are invalid")

    input_log_normalizer = float(logsumexp(input_log_weights))
    if not np.isclose(
        input_log_normalizer,
        0.0,
        rtol=0.0,
        atol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError("input log weights are not normalized")
    branch_log_normalizers = logsumexp(log_branch_probabilities, axis=1)
    if not np.allclose(
        branch_log_normalizers,
        0.0,
        rtol=0.0,
        atol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError("branch probabilities are not row-normalized")

    point_weights = np.exp(input_log_weights)
    branch_probabilities = np.exp(log_branch_probabilities)
    sector_masses = np.sum(point_weights[:, None] * branch_probabilities, axis=0)
    entropy_terms = np.zeros_like(branch_probabilities)
    positive = branch_probabilities > 0.0
    entropy_terms[positive] = (
        -branch_probabilities[positive] * (log_branch_probabilities[positive])
    )
    normalized_entropy = float(
        np.sum(point_weights * np.sum(entropy_terms, axis=1))
        / np.log(float(_FOLD_GROUP_ORDER))
    )
    return {
        "group_order": _FOLD_GROUP_ORDER,
        "folded_points": int(input_log_weights.size),
        "normalized_conditional_image_entropy": normalized_entropy,
        "image_sector_posterior_masses": [float(value) for value in sector_masses],
        "expected_nonidentity_mass": float(1.0 - sector_masses[0]),
        "zero_support_image_fraction": float(np.mean(~np.isfinite(base_log_priors))),
    }
