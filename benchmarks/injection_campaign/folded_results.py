"""CPU-only result helpers for deterministic quotient-fold expansion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

_FOLD_GROUP_ORDER = 8
_NORMALIZATION_TOLERANCE = 1.0e-10
_GAP_QUANTILES = (0.05, 0.5, 0.95)


@dataclass(frozen=True)
class UnfoldedWeightedPosterior:
    """Aligned physical posterior and its paired folded NS diagnostics."""

    samples: dict[str, np.ndarray]
    true_log_likelihood: np.ndarray
    log_weights: np.ndarray
    folded_log_likelihood: np.ndarray
    folded_log_likelihood_birth: np.ndarray
    insertion_diagnostic: dict[str, Any]
    telemetry: dict[str, Any]


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Return deterministic left-continuous weighted gap quantiles."""

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        raise ValueError("likelihood-gap quantiles require positive finite weight")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    thresholds = np.asarray(_GAP_QUANTILES) * cumulative[-1]
    indices = np.searchsorted(cumulative, thresholds, side="left")
    labels = ("p05", "p50", "p95")
    return {
        label: float(values[min(int(index), values.size - 1)])
        for label, index in zip(labels, indices, strict=True)
    }


def unfolded_posterior_telemetry(
    input_log_weights: np.ndarray,
    base_log_priors: np.ndarray,
    log_branch_probabilities: np.ndarray,
    true_log_likelihoods: np.ndarray | None = None,
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
    telemetry: dict[str, Any] = {
        "group_order": _FOLD_GROUP_ORDER,
        "folded_points": int(input_log_weights.size),
        "normalized_conditional_image_entropy": normalized_entropy,
        "image_sector_posterior_masses": [float(value) for value in sector_masses],
        "expected_nonidentity_mass": float(1.0 - sector_masses[0]),
        "zero_support_image_fraction": float(np.mean(~np.isfinite(base_log_priors))),
    }
    if true_log_likelihoods is not None:
        true_log_likelihoods = np.asarray(true_log_likelihoods, dtype=np.float64)
        if true_log_likelihoods.shape != base_log_priors.shape:
            raise ValueError("true image likelihoods must align with the image table")
        if np.any(np.isnan(true_log_likelihoods)) or np.any(
            np.isposinf(true_log_likelihoods)
        ):
            raise ValueError("true image likelihoods are invalid")
        supported = np.isfinite(base_log_priors) & np.isfinite(true_log_likelihoods)
        if np.any(~np.any(supported, axis=1)):
            raise ValueError("every folded point must have a supported image likelihood")
        supported_minima = np.min(
            np.where(supported, true_log_likelihoods, np.inf), axis=1
        )
        supported_maxima = np.max(
            np.where(supported, true_log_likelihoods, -np.inf), axis=1
        )
        spans = supported_maxima - supported_minima

        identity_gap_values: list[float] = []
        identity_gap_weights: list[float] = []
        for row, point_weight in enumerate(point_weights):
            if not supported[row, 0]:
                continue
            nonidentity_supported = supported[row, 1:]
            count = int(np.sum(nonidentity_supported))
            if count == 0:
                continue
            gaps = np.abs(
                true_log_likelihoods[row, 1:][nonidentity_supported]
                - true_log_likelihoods[row, 0]
            )
            identity_gap_values.extend(float(value) for value in gaps)
            identity_gap_weights.extend([float(point_weight) / count] * count)
        gap_summary: dict[str, Any] = {
            "within_orbit_span_weighted_quantiles": _weighted_quantiles(
                spans, point_weights
            ),
            "identity_absolute_gap_weighted_quantiles": None,
        }
        if identity_gap_values:
            gap_summary["identity_absolute_gap_weighted_quantiles"] = (
                _weighted_quantiles(
                    np.asarray(identity_gap_values),
                    np.asarray(identity_gap_weights),
                )
            )
        telemetry["supported_image_log_likelihood_gaps"] = gap_summary
    return telemetry


def extract_unfolded_weighted_posterior(
    jim: Any,
    *,
    n_live: int,
    batch_size: int | None = None,
    insertion_diagnostic_fn: Callable[..., dict[str, Any]] | None = None,
) -> UnfoldedWeightedPosterior:
    """Diagnose, unfold, then reverse-transform a folded weighted collection."""

    if type(n_live) is not int or n_live < 1:
        raise ValueError("n_live must be a positive integer")

    raw = jim.get_weighted_samples(space="sampling")
    required = {
        "samples",
        "log_likelihood",
        "log_likelihood_birth",
        "log_weights",
    }
    if not isinstance(raw, dict) or not required.issubset(raw):
        raise ValueError("folded weighted output is missing required fields")

    positions = np.asarray(raw["samples"])
    log_weights = np.asarray(raw["log_weights"], dtype=np.float64)
    folded_log_likelihood = np.asarray(raw["log_likelihood"])
    folded_log_likelihood_birth = np.asarray(raw["log_likelihood_birth"])
    if positions.ndim != 2:
        raise ValueError("folded sampling positions must be a two-dimensional array")
    n_points = positions.shape[0]
    if any(
        values.shape != (n_points,)
        for values in (
            log_weights,
            folded_log_likelihood,
            folded_log_likelihood_birth,
        )
    ):
        raise ValueError("folded weighted arrays are not aligned")

    if insertion_diagnostic_fn is None:
        from jimgw.samplers.diagnostics import insertion_index_diagnostic

        insertion_diagnostic_fn = insertion_index_diagnostic
    insertion_diagnostic = insertion_diagnostic_fn(
        folded_log_likelihood,
        folded_log_likelihood_birth,
        n_live=n_live,
    )
    if not isinstance(insertion_diagnostic, dict):
        raise TypeError("insertion diagnostic must return a dictionary")

    unfolded = jim.unfold_weighted_samples(
        positions,
        log_weights,
        batch_size=batch_size,
    )
    retained_input_weights = log_weights[~np.isneginf(log_weights)]
    n_retained = retained_input_weights.size
    expected_images = n_retained * _FOLD_GROUP_ORDER
    unfolded_positions = np.asarray(unfolded.positions)
    split_log_weights = np.asarray(unfolded.log_weights, dtype=np.float64)
    true_log_likelihoods = np.asarray(unfolded.true_log_likelihoods)
    base_log_priors = np.asarray(unfolded.base_log_priors, dtype=np.float64)
    log_branch_probabilities = np.asarray(
        unfolded.log_branch_probabilities, dtype=np.float64
    )
    if unfolded_positions.ndim != 2 or unfolded_positions.shape[0] != expected_images:
        raise ValueError("unfolded sampling positions have an invalid shape")
    if any(
        values.shape != (expected_images,)
        for values in (
            split_log_weights,
            true_log_likelihoods,
            base_log_priors,
            log_branch_probabilities,
        )
    ):
        raise ValueError("unfolded image arrays are not aligned")

    telemetry = unfolded_posterior_telemetry(
        retained_input_weights,
        base_log_priors.reshape(n_retained, _FOLD_GROUP_ORDER),
        log_branch_probabilities.reshape(n_retained, _FOLD_GROUP_ORDER),
        true_log_likelihoods.reshape(n_retained, _FOLD_GROUP_ORDER),
    )
    if np.any(np.isnan(split_log_weights)) or np.any(np.isposinf(split_log_weights)):
        raise ValueError("unfolded log weights are invalid")
    positive_mass = np.isfinite(split_log_weights)
    if not np.any(positive_mass):
        raise ValueError("unfolded posterior has no positive-mass image branches")
    output_log_weights = split_log_weights[positive_mass]
    if not np.isclose(
        float(logsumexp(output_log_weights)),
        0.0,
        rtol=0.0,
        atol=_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError("unfolded posterior log weights are not normalized")
    output_likelihoods = true_log_likelihoods[positive_mass]
    if not np.all(np.isfinite(output_likelihoods)):
        raise ValueError("positive-mass image likelihoods must be finite")

    samples = jim.samples_to_prior_space(unfolded_positions[positive_mass])
    if not isinstance(samples, dict) or not samples:
        raise ValueError("reverse transforms returned no physical sample fields")
    sample_count = int(output_log_weights.size)
    if any(np.asarray(values).shape != (sample_count,) for values in samples.values()):
        raise ValueError("reverse-transformed physical samples are not aligned")
    return UnfoldedWeightedPosterior(
        samples={name: np.asarray(values) for name, values in samples.items()},
        true_log_likelihood=output_likelihoods,
        log_weights=output_log_weights,
        folded_log_likelihood=folded_log_likelihood,
        folded_log_likelihood_birth=folded_log_likelihood_birth,
        insertion_diagnostic=insertion_diagnostic,
        telemetry=telemetry,
    )
