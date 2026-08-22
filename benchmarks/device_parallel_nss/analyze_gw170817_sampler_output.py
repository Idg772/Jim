"""Analyze three weighted GW170817 nested-sampling output artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks
from scipy.special import logsumexp
from scipy.stats import gaussian_kde, kstest, kstwo

from benchmarks.injection_campaign.common import (
    FOLDED_TARGET_SEMANTICS,
    POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS,
    UNFOLDED_POSTERIOR_WEIGHTING,
)
from benchmarks.injection_campaign.common import (
    NETSKY_BLOCKS as CAMPAIGN_NETSKY_BLOCKS,
)
from benchmarks.injection_campaign.common import (
    NETSKY_BRIDGE_BLOCKS as CAMPAIGN_NETSKY_BRIDGE_BLOCKS,
)

PARAMETERS = ("M_c", "q", "lambda_1", "lambda_2", "d_L", "iota")
NESTED_FIELDS = {
    "log_likelihood",
    "log_likelihood_birth",
    "log_weights",
}
WEIGHTING = "normalized nested-sampling log weights"
PAPER_BLOCKING_SCHEME = "paper"
ALL_SLOW_BLOCKING_SCHEME = "all-slow"
NETSKY_BLOCKING_SCHEME = "netsky"
FOLDED_TARGET_SPACE = "folded sampling-space target"
FOLDED_TARGET_WEIGHTING = "not applicable: folded nested-sampling contours"
PAPER_BLOCKS = (
    ("M_c", "q", "lambda_1", "lambda_2"),
    ("s1_mag", "s1_theta", "s1_phi"),
    ("s2_mag", "s2_theta", "s2_phi"),
    ("iota",),
    ("zenith", "azimuth"),
    ("psi",),
    ("d_L",),
)
ALL_SLOW_BLOCKS = (
    (
        "M_c",
        "q",
        "lambda_1",
        "lambda_2",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "iota",
    ),
    ("zenith", "azimuth"),
    ("psi",),
    ("d_L",),
)
NETSKY_BLOCKS = tuple(tuple(block) for block in CAMPAIGN_NETSKY_BLOCKS)
NETSKY_BRIDGE_BLOCKS = tuple(tuple(block) for block in CAMPAIGN_NETSKY_BRIDGE_BLOCKS)


@dataclass(frozen=True)
class InsertionRankDiagnostic:
    sample_size: int
    global_statistic: float
    global_p_value: float
    worst_window_p_value: float
    worst_window_start: int
    worst_window_stop: int
    window_size: int
    live_min: int
    live_max: int


def _aligned_1d_arrays(
    values: dict[str, np.ndarray],
    *,
    required: set[str],
) -> dict[str, np.ndarray]:
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError("artifact is missing fields: " + ", ".join(missing))
    arrays = {name: np.asarray(value) for name, value in values.items()}
    invalid = {
        name: list(value.shape) for name, value in arrays.items() if value.ndim != 1
    }
    if invalid:
        raise ValueError(f"artifact fields must be one-dimensional: {invalid}")
    lengths = {name: len(value) for name, value in arrays.items()}
    if not lengths or len(set(lengths.values())) != 1:
        raise ValueError(f"artifact fields have inconsistent lengths: {lengths}")
    return arrays


def insertion_rank_uniforms(
    log_likelihood: np.ndarray,
    log_likelihood_birth: np.ndarray,
    *,
    random_seed: int = 170817,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cohort-held-out insertion ranks mapped continuously to [0, 1]."""

    death = np.asarray(log_likelihood, dtype=np.float64)
    birth = np.asarray(log_likelihood_birth, dtype=np.float64)
    if death.ndim != 1 or birth.ndim != 1 or death.shape != birth.shape:
        raise ValueError("birth and death likelihoods must be aligned 1D arrays")
    if not np.all(np.isfinite(death)):
        raise ValueError("death likelihoods must be finite")
    if np.any(np.isnan(birth)) or np.any(np.isposinf(birth)):
        raise ValueError("birth likelihoods must not contain NaN or +inf")

    replacement = np.isfinite(birth)
    if not np.any(replacement):
        raise ValueError("artifact contains no finite-birth replacement points")
    if np.any(death[replacement] <= birth[replacement]):
        raise ValueError(
            "every replacement death contour must exceed its birth contour"
        )

    rng = np.random.default_rng(random_seed)
    ranks: list[np.ndarray] = []
    live_sizes: list[np.ndarray] = []
    for contour in np.unique(birth[replacement]):
        # Batched replacements share a birth contour and therefore have no
        # intrinsic within-cohort order. Shuffle ties so a sliding window does
        # not inherit the artifact's death-likelihood sort order.
        cohort = rng.permutation(np.flatnonzero(birth == contour))
        live = np.sort(death[(birth < contour) & (contour < death)])
        if live.size == 0:
            raise ValueError(f"birth contour {contour} has an empty held-out live set")
        left = np.searchsorted(live, death[cohort], side="left")
        right = np.searchsorted(live, death[cohort], side="right")
        # The new point and any equal-death live points share an exchangeable
        # rank interval. Jitter makes the resulting KS test continuous.
        jittered = left + rng.random(cohort.size) * (right - left + 1)
        ranks.append(jittered / (live.size + 1))
        live_sizes.append(np.full(cohort.size, live.size + 1, dtype=np.int64))
    return np.concatenate(ranks), np.concatenate(live_sizes)


def _worst_window_ks(ranks: np.ndarray, window_size: int) -> tuple[float, int, int]:
    size = min(window_size, len(ranks))
    if size < 2:
        raise ValueError("at least two insertion ranks are required")
    if size == len(ranks):
        return float(kstest(ranks, "uniform").pvalue), 0, size

    windows = np.lib.stride_tricks.sliding_window_view(ranks, size)
    lower = np.arange(size, dtype=np.float64) / size
    upper = np.arange(1, size + 1, dtype=np.float64) / size
    minimum_p = 1.0
    minimum_index = 0
    for offset in range(0, len(windows), 512):
        ordered = np.sort(windows[offset : offset + 512], axis=1)
        statistic = np.maximum(
            np.max(upper - ordered, axis=1),
            np.max(ordered - lower, axis=1),
        )
        p_values = np.asarray(kstwo.sf(statistic, size), dtype=np.float64)
        local_index = int(np.argmin(p_values))
        if p_values[local_index] < minimum_p:
            minimum_p = float(p_values[local_index])
            minimum_index = offset + local_index
    return minimum_p, minimum_index, minimum_index + size


def insertion_rank_diagnostic(
    log_likelihood: np.ndarray,
    log_likelihood_birth: np.ndarray,
    *,
    window_size: int = 1000,
    random_seed: int = 170817,
) -> InsertionRankDiagnostic:
    ranks, live_sizes = insertion_rank_uniforms(
        log_likelihood,
        log_likelihood_birth,
        random_seed=random_seed,
    )
    global_result = kstest(ranks, "uniform")
    worst_p, start, stop = _worst_window_ks(ranks, window_size)
    return InsertionRankDiagnostic(
        sample_size=len(ranks),
        global_statistic=float(global_result.statistic),
        global_p_value=float(global_result.pvalue),
        worst_window_p_value=worst_p,
        worst_window_start=start,
        worst_window_stop=stop,
        window_size=stop - start,
        live_min=int(live_sizes.min()),
        live_max=int(live_sizes.max()),
    )


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    ordered_weights = ordered_weights / ordered_weights.sum()
    centers = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    return np.interp(
        probabilities,
        centers,
        ordered_values,
        left=ordered_values[0],
        right=ordered_values[-1],
    )


def _quantile_draws(
    values: np.ndarray,
    log_weight_draws: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    log_weights = np.asarray(log_weight_draws, dtype=np.float64)[order]
    weights = np.exp(log_weights - logsumexp(log_weights, axis=0))
    quantiles = np.empty((weights.shape[1], len(probabilities)), dtype=np.float64)
    for draw in range(weights.shape[1]):
        centers = np.cumsum(weights[:, draw]) - 0.5 * weights[:, draw]
        quantiles[draw] = np.interp(
            probabilities,
            centers,
            ordered_values,
            left=ordered_values[0],
            right=ordered_values[-1],
        )
    return quantiles


def weighted_kde_modes(
    values: np.ndarray,
    log_weights: np.ndarray,
    *,
    grid_size: int = 4096,
    relative_height: float = 0.05,
) -> tuple[int, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.exp(np.asarray(log_weights, dtype=np.float64) - logsumexp(log_weights))
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        return 1, [lower]
    grid = np.linspace(lower, upper, grid_size)
    density = np.asarray(gaussian_kde(values, weights=weights)(grid))
    threshold = relative_height * float(density.max())
    indexes = list(find_peaks(density, height=threshold)[0])
    if density[0] >= threshold and density[0] > density[1]:
        indexes.insert(0, 0)
    if density[-1] >= threshold and density[-1] > density[-2]:
        indexes.append(grid_size - 1)
    return len(indexes), [float(grid[index]) for index in indexes]


def _load_npz(path: Path, *, required: set[str]) -> dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as artifact:
        return _aligned_1d_arrays(
            {name: np.asarray(artifact[name]) for name in artifact.files},
            required=required,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_summary(
    path: Path,
    *,
    seed: int,
    bootstrap_samples: int,
    window_size: int,
    random_seed: int,
) -> dict[str, Any]:
    arrays = _load_npz(path, required=NESTED_FIELDS | set(PARAMETERS))
    death = np.asarray(arrays["log_likelihood"], dtype=np.float64)
    birth = np.asarray(arrays["log_likelihood_birth"], dtype=np.float64)
    stored_log_weights = np.asarray(arrays["log_weights"], dtype=np.float64)
    if not np.all(np.isfinite(stored_log_weights)):
        raise ValueError(f"{path}: log_weights must be finite")
    if not np.isclose(logsumexp(stored_log_weights), 0.0, atol=1e-8):
        raise ValueError(f"{path}: log_weights are not normalized")

    order = np.argsort(death, kind="stable")
    death = death[order]
    birth = birth[order]
    parameter_fields = [name for name in arrays if name not in NESTED_FIELDS]
    matrix = np.column_stack(
        [np.asarray(arrays[name])[order] for name in parameter_fields]
    )

    from anesthetic.samples import NestedSamples

    nested = NestedSamples(
        matrix,
        columns=parameter_fields,
        logL=death,
        logL_birth=birth,
        logzero=np.nan,
        dtype=np.float64,
    )
    central_log_weights = np.array(nested.logw(), dtype=np.float64, copy=True)
    central_log_weights -= logsumexp(central_log_weights)
    if not np.allclose(central_log_weights, stored_log_weights[order], atol=1e-9):
        raise ValueError(f"{path}: stored weights disagree with reconstructed topology")

    random_state = np.random.get_state()
    np.random.seed(random_seed)
    try:
        log_weight_draws = np.asarray(nested.logw(bootstrap_samples), dtype=np.float64)
    finally:
        np.random.set_state(random_state)
    log_z_draws = logsumexp(log_weight_draws, axis=0)
    weights = np.exp(central_log_weights)
    probabilities = np.asarray([0.05, 0.5, 0.95])
    parameters: dict[str, dict[str, Any]] = {}
    for name in PARAMETERS:
        values = np.asarray(nested[name], dtype=np.float64)
        estimate = _weighted_quantiles(values, weights, probabilities)
        draws = _quantile_draws(values, log_weight_draws, probabilities)
        parameters[name] = {
            "lower": float(estimate[0]),
            "median": float(estimate[1]),
            "upper": float(estimate[2]),
            "lower_error": float(np.std(draws[:, 0])),
            "median_error": float(np.std(draws[:, 1])),
            "upper_error": float(np.std(draws[:, 2])),
        }
    mode_count, mode_locations = weighted_kde_modes(
        np.asarray(nested["q"], dtype=np.float64), central_log_weights
    )
    return {
        "seed": seed,
        "path": str(path.resolve()),
        "sha256": _sha256(path.resolve()),
        "insertion": insertion_rank_diagnostic(
            death,
            birth,
            window_size=window_size,
            random_seed=random_seed,
        ),
        "log_z": float(nested.logZ()),
        "log_z_error": float(np.std(log_z_draws)),
        "ess": float(nested.neff()),
        "parameters": parameters,
        "q_mode_count": mode_count,
        "q_mode_locations": mode_locations,
    }


def _ridge_occupancy(
    distance: np.ndarray,
    iota: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    distance = np.asarray(distance, dtype=np.float64)
    iota = np.asarray(iota, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        distance.ndim != 1
        or iota.shape != distance.shape
        or weights.shape != distance.shape
        or not np.all(np.isfinite(distance))
        or not np.all(np.isfinite(iota))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or weights.sum() <= 0.0
        or np.any(distance <= 0.0)
        or np.any(iota < 0.0)
        or np.any(iota > np.pi)
    ):
        raise ValueError("ridge occupancy inputs must be aligned finite vectors")
    weights = weights / weights.sum()
    nonnegative = np.cos(iota) >= 0.0
    probabilities = np.asarray([0.05, 0.5, 0.95])

    def branch(mask: np.ndarray) -> dict[str, Any]:
        mass = float(weights[mask].sum())
        quantiles = None
        if mass > 0.0:
            values = _weighted_quantiles(
                distance[mask],
                weights[mask] / mass,
                probabilities,
            )
            quantiles = {
                name: float(value)
                for name, value in zip(("p05", "p50", "p95"), values, strict=True)
            }
        return {"posterior_mass": mass, "d_L_quantiles": quantiles}

    return {
        "criterion": "cos(iota) >= 0 versus cos(iota) < 0",
        "cos_iota_nonnegative": branch(nonnegative),
        "cos_iota_negative": branch(~nonnegative),
    }


def _netsky_seed_summary(
    posterior_path: Path,
    folded_path: Path,
    report: dict[str, Any],
    *,
    seed: int,
    window_size: int,
    random_seed: int,
) -> dict[str, Any]:
    posterior = _load_npz(
        posterior_path,
        required={*PARAMETERS, "log_likelihood", "log_weights"},
    )
    if "log_likelihood_birth" in posterior:
        raise ValueError(
            f"{posterior_path}: unfolded physical posterior must not contain birth "
            "likelihoods"
        )
    log_weights = np.asarray(posterior["log_weights"], dtype=np.float64)
    true_log_likelihood = np.asarray(posterior["log_likelihood"], dtype=np.float64)
    if not np.all(np.isfinite(log_weights)) or not np.all(
        np.isfinite(true_log_likelihood)
    ):
        raise ValueError(
            f"{posterior_path}: posterior weights and likelihoods must be finite"
        )
    if not np.isclose(logsumexp(log_weights), 0.0, atol=1e-8):
        raise ValueError(f"{posterior_path}: log_weights are not normalized")
    weights = np.exp(log_weights)

    folded = _load_npz(
        folded_path,
        required={"log_likelihood", "log_likelihood_birth"},
    )
    death = np.asarray(folded["log_likelihood"], dtype=np.float64)
    birth = np.asarray(folded["log_likelihood_birth"], dtype=np.float64)
    results = report.get("results")
    timing = report.get("timing_seconds")
    if not isinstance(results, dict) or not isinstance(timing, dict):
        raise TypeError("netsky report is missing result or timing data")
    n_live = report.get("config", {}).get("n_live")
    if type(n_live) is not int or n_live < 1:
        raise ValueError("netsky report has an invalid live-point count")

    from jimgw.samplers.diagnostics import insertion_index_diagnostic

    recomputed_insertion = insertion_index_diagnostic(death, birth, n_live=n_live)
    if results.get("insertion_index_diagnostic") != recomputed_insertion:
        raise ValueError(
            "netsky stored insertion-index diagnostic disagrees with the folded "
            "artifact"
        )

    log_z = float(results.get("log_Z", np.nan))
    log_z_error = float(results.get("log_Z_error", np.nan))
    reported_effective_size = float(
        results.get("posterior_weight_effective_size", np.nan)
    )
    computed_effective_size = float(1.0 / np.sum(weights**2))
    postprocess_seconds = float(timing.get("fold_unfold_postprocessing", np.nan))
    fold_telemetry = _validate_fold_telemetry(results.get("quotient_fold"))
    if (
        not np.isfinite(log_z)
        or not np.isfinite(log_z_error)
        or log_z_error <= 0.0
        or not np.isfinite(reported_effective_size)
        or reported_effective_size < 1.0 - 1e-8
        or not np.isclose(
            reported_effective_size,
            computed_effective_size,
            rtol=1e-10,
            atol=1e-10,
        )
        or results.get("posterior_weight_effective_size_semantics")
        != POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
        or not np.isfinite(postprocess_seconds)
        or postprocess_seconds <= 0.0
    ):
        raise ValueError("netsky report has invalid evidence, weight, or timing data")

    probabilities = np.asarray([0.05, 0.5, 0.95])
    parameters = {}
    for name in PARAMETERS:
        values = np.asarray(posterior[name], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{posterior_path}: {name} must be finite")
        estimate = _weighted_quantiles(values, weights, probabilities)
        parameters[name] = {
            key: float(value)
            for key, value in zip(("lower", "median", "upper"), estimate, strict=True)
        }
    mode_count, mode_locations = weighted_kde_modes(posterior["q"], log_weights)
    return {
        "seed": seed,
        "path": str(posterior_path.resolve()),
        "sha256": _sha256(posterior_path.resolve()),
        "folded_path": str(folded_path.resolve()),
        "folded_sha256": _sha256(folded_path.resolve()),
        "insertion": insertion_rank_diagnostic(
            death,
            birth,
            window_size=window_size,
            random_seed=random_seed,
        ),
        "log_z": log_z,
        "log_z_error": log_z_error,
        "ess": reported_effective_size,
        "posterior_weight_effective_size": reported_effective_size,
        "parameters": parameters,
        "q_mode_count": mode_count,
        "q_mode_locations": mode_locations,
        "ridge_occupancy": _ridge_occupancy(
            posterior["d_L"], posterior["iota"], weights
        ),
        "fold_telemetry": fold_telemetry,
        "fold_unfold_postprocessing_seconds": postprocess_seconds,
    }


def _descriptive_distribution(
    values: list[tuple[int, float | None]],
) -> dict[str, Any]:
    numeric = np.asarray(
        [value for _, value in values if value is not None], dtype=np.float64
    )
    if numeric.size and not np.all(np.isfinite(numeric)):
        raise ValueError("cross-seed descriptive values must be finite")
    return {
        "by_seed": [
            {"seed": seed, "value": None if value is None else float(value)}
            for seed, value in values
        ],
        "available_seed_count": int(numeric.size),
        "minimum": float(np.min(numeric)) if numeric.size else None,
        "median": float(np.median(numeric)) if numeric.size else None,
        "maximum": float(np.max(numeric)) if numeric.size else None,
        "range": float(np.ptp(numeric)) if numeric.size else None,
    }


def _netsky_cross_seed_stability(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(summaries) < 2:
        raise ValueError("netsky cross-seed stability requires at least two seeds")
    ordered = sorted(summaries, key=lambda item: int(item["seed"]))
    seeds = [int(item["seed"]) for item in ordered]
    if len(set(seeds)) != len(seeds):
        raise ValueError("netsky cross-seed stability requires unique seeds")

    scalar_getters = {
        "normalized_conditional_image_entropy": lambda item: item["fold_telemetry"][
            "normalized_conditional_image_entropy"
        ],
        "expected_nonidentity_mass": lambda item: item["fold_telemetry"][
            "expected_nonidentity_mass"
        ],
        "zero_support_image_fraction": lambda item: item["fold_telemetry"][
            "zero_support_image_fraction"
        ],
        "fold_unfold_postprocessing_seconds": lambda item: item[
            "fold_unfold_postprocessing_seconds"
        ],
        "posterior_weight_effective_size": lambda item: item[
            "posterior_weight_effective_size"
        ],
    }
    metrics = {
        name: _descriptive_distribution(
            [
                (seed, float(getter(item)))
                for seed, item in zip(seeds, ordered, strict=True)
            ]
        )
        for name, getter in scalar_getters.items()
    }
    sector_masses = [
        _descriptive_distribution(
            [
                (
                    seed,
                    float(
                        item["fold_telemetry"]["image_sector_posterior_masses"][sector]
                    ),
                )
                for seed, item in zip(seeds, ordered, strict=True)
            ]
        )
        for sector in range(8)
    ]

    ridge = {}
    for branch_name in ("cos_iota_nonnegative", "cos_iota_negative"):
        branch = {
            "posterior_mass": _descriptive_distribution(
                [
                    (
                        seed,
                        float(item["ridge_occupancy"][branch_name]["posterior_mass"]),
                    )
                    for seed, item in zip(seeds, ordered, strict=True)
                ]
            )
        }
        branch["d_L_quantiles"] = {
            quantile: _descriptive_distribution(
                [
                    (
                        seed,
                        (
                            None
                            if item["ridge_occupancy"][branch_name]["d_L_quantiles"]
                            is None
                            else float(
                                item["ridge_occupancy"][branch_name]["d_L_quantiles"][
                                    quantile
                                ]
                            )
                        ),
                    )
                    for seed, item in zip(seeds, ordered, strict=True)
                ]
            )
            for quantile in ("p05", "p50", "p95")
        }
        ridge[branch_name] = branch

    gap_distributions = {}
    for family in (
        "within_orbit_span_weighted_quantiles",
        "identity_absolute_gap_weighted_quantiles",
    ):
        gap_distributions[family] = {
            quantile: _descriptive_distribution(
                [
                    (
                        seed,
                        (
                            None
                            if item["fold_telemetry"][
                                "supported_image_log_likelihood_gaps"
                            ][family]
                            is None
                            else float(
                                item["fold_telemetry"][
                                    "supported_image_log_likelihood_gaps"
                                ][family][quantile]
                            )
                        ),
                    )
                    for seed, item in zip(seeds, ordered, strict=True)
                ]
            )
            for quantile in ("p05", "p50", "p95")
        }

    return {
        "role": "descriptive_cross_seed_stability",
        "used_as_acceptance_gate": False,
        "metrics": metrics,
        "image_sector_posterior_masses": sector_masses,
        "ridge_occupancy": ridge,
        "likelihood_gap_quantiles": gap_distributions,
    }


def posterior_q_summary(path: Path) -> dict[str, Any]:
    arrays = _load_npz(path, required={"q", "log_weights"})
    log_weights = np.asarray(arrays["log_weights"], dtype=np.float64)
    if not np.all(np.isfinite(log_weights)):
        raise ValueError(f"{path}: log_weights must be finite")
    weights = np.exp(log_weights - logsumexp(log_weights))
    quantiles = _weighted_quantiles(
        np.asarray(arrays["q"], dtype=np.float64),
        weights,
        np.asarray([0.05, 0.5, 0.95]),
    )
    count, locations = weighted_kde_modes(arrays["q"], log_weights)
    return {
        "lower": float(quantiles[0]),
        "median": float(quantiles[1]),
        "upper": float(quantiles[2]),
        "mode_count": count,
        "mode_locations": locations,
    }


def _z_score(
    first: float, second: float, first_error: float, second_error: float
) -> float:
    denominator = float(np.hypot(first_error, second_error))
    if denominator == 0:
        return 0.0 if first == second else float("inf")
    return abs(first - second) / denominator


def _cross_seed(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    log_z_candidates = [
        (
            _z_score(
                first["log_z"],
                second["log_z"],
                first["log_z_error"],
                second["log_z_error"],
            ),
            (first["seed"], second["seed"]),
        )
        for first, second in combinations(summaries, 2)
    ]
    parameter_z: dict[str, dict[str, Any]] = {}
    for name in PARAMETERS:
        candidates = []
        for first, second in combinations(summaries, 2):
            first_value = first["parameters"][name]
            second_value = second["parameters"][name]
            candidates.append(
                (
                    _z_score(
                        first_value["median"],
                        second_value["median"],
                        first_value["median_error"],
                        second_value["median_error"],
                    ),
                    (first["seed"], second["seed"]),
                )
            )
        value, pair = max(candidates)
        parameter_z[name] = {"value": value, "seeds": pair}
    maximum_parameter = max(PARAMETERS, key=lambda name: parameter_z[name]["value"])
    log_z_value, log_z_pair = max(log_z_candidates)
    mode_counts = [summary["q_mode_count"] for summary in summaries]
    return {
        "max_log_z": {"value": log_z_value, "seeds": log_z_pair},
        "parameter_z": parameter_z,
        "max_median": {
            "parameter": maximum_parameter,
            **parameter_z[maximum_parameter],
        },
        "mode_counts": mode_counts,
        "same_mode_count": len(set(mode_counts)) == 1,
    }


def _estimate_cell(value: dict[str, float]) -> str:
    return (
        f"{value['median']:.6g} ± {value['median_error']:.2g} "
        f"[{value['lower']:.6g} ± {value['lower_error']:.2g}, "
        f"{value['upper']:.6g} ± {value['upper_error']:.2g}]"
    )


def _distribution_cell(distribution: dict[str, Any]) -> str:
    if distribution["available_seed_count"] == 0:
        return "not available"
    return (
        f"min={distribution['minimum']:.4g}; "
        f"median={distribution['median']:.4g}; "
        f"max={distribution['maximum']:.4g}; "
        f"range={distribution['range']:.4g}"
    )


def _ridge_cell(branch: dict[str, Any]) -> str:
    quantiles = branch["d_L_quantiles"]
    if quantiles is None:
        return f"mass={branch['posterior_mass']:.4g}; d_L unavailable"
    return (
        f"mass={branch['posterior_mass']:.4g}; d_L "
        f"[{quantiles['p05']:.4g}, {quantiles['p50']:.4g}, "
        f"{quantiles['p95']:.4g}] Mpc"
    )


def _gap_cell(quantiles: dict[str, float] | None) -> str:
    if quantiles is None:
        return "not available"
    return f"[{quantiles['p05']:.4g}, {quantiles['p50']:.4g}, {quantiles['p95']:.4g}]"


def _render_netsky_report(
    summaries: list[dict[str, Any]],
    *,
    bundle_sha256: str | None,
    blocks: tuple[tuple[str, ...], ...] | None,
    stability: dict[str, Any],
) -> str:
    headers = ["Metric", *[f"Seed {item['seed']}" for item in summaries], "Cross-seed"]
    rows: list[list[str]] = []
    minimum_global_p = min(item["insertion"].global_p_value for item in summaries)
    minimum_window_p = min(item["insertion"].worst_window_p_value for item in summaries)
    rows.append(
        [
            "Folded-target insertion-rank KS p (global)",
            *[f"{item['insertion'].global_p_value:.4g}" for item in summaries],
            f"min={minimum_global_p:.4g}",
        ]
    )
    rows.append(
        [
            "Worst raw 1000-rank-window KS p",
            *[
                f"{item['insertion'].worst_window_p_value:.4g} "
                f"({item['insertion'].worst_window_start}:"
                f"{item['insertion'].worst_window_stop})"
                for item in summaries
            ],
            f"min={minimum_window_p:.4g} (descriptive)",
        ]
    )
    log_z_candidates = [
        (
            _z_score(
                first["log_z"],
                second["log_z"],
                first["log_z_error"],
                second["log_z_error"],
            ),
            (first["seed"], second["seed"]),
        )
        for first, second in combinations(summaries, 2)
    ]
    log_z_score, log_z_pair = max(log_z_candidates)
    rows.append(
        [
            "Folded-target log Z ± sampler error",
            *[f"{item['log_z']:.6f} ± {item['log_z_error']:.3g}" for item in summaries],
            f"max z={log_z_score:.3g} (seeds {log_z_pair[0]}/{log_z_pair[1]})",
        ]
    )
    rows.append(
        [
            "Quadrature-weight concentration",
            *[f"{item['ess']:.1f}" for item in summaries],
            _distribution_cell(stability["metrics"]["posterior_weight_effective_size"]),
        ]
    )
    for name in PARAMETERS:
        rows.append(
            [
                f"{name} physical median [5%, 95%]",
                *[
                    f"{item['parameters'][name]['median']:.6g} "
                    f"[{item['parameters'][name]['lower']:.6g}, "
                    f"{item['parameters'][name]['upper']:.6g}]"
                    for item in summaries
                ],
                "physical posterior; descriptive",
            ]
        )
    mode_counts = [item["q_mode_count"] for item in summaries]
    rows.append(
        [
            "Unfolded physical q KDE mode count (>5% peak)",
            *[
                f"{item['q_mode_count']} at "
                + ", ".join(f"{value:.4g}" for value in item["q_mode_locations"])
                for item in summaries
            ],
            f"{mode_counts}; same count={'yes' if len(set(mode_counts)) == 1 else 'no'}",
        ]
    )
    for branch_name, label in (
        ("cos_iota_nonnegative", "cos(iota) >= 0"),
        ("cos_iota_negative", "cos(iota) < 0"),
    ):
        branch_stability = stability["ridge_occupancy"][branch_name]
        rows.append(
            [
                f"d_L–iota ridge {label}",
                *[
                    _ridge_cell(item["ridge_occupancy"][branch_name])
                    for item in summaries
                ],
                "mass " + _distribution_cell(branch_stability["posterior_mass"]),
            ]
        )
    rows.append(
        [
            "Image-sector posterior masses (0–7)",
            *[
                "["
                + ", ".join(
                    f"{value:.4g}"
                    for value in item["fold_telemetry"]["image_sector_posterior_masses"]
                )
                + "]"
                for item in summaries
            ],
            "; ".join(
                f"{sector}:{distribution['minimum']:.3g}–{distribution['maximum']:.3g}"
                for sector, distribution in enumerate(
                    stability["image_sector_posterior_masses"]
                )
            ),
        ]
    )
    for key, label in (
        (
            "normalized_conditional_image_entropy",
            "Normalized conditional image entropy",
        ),
        ("expected_nonidentity_mass", "Expected nonidentity image mass"),
        ("zero_support_image_fraction", "Zero-support image fraction"),
    ):
        rows.append(
            [
                label,
                *[f"{item['fold_telemetry'][key]:.6g}" for item in summaries],
                _distribution_cell(stability["metrics"][key]),
            ]
        )
    for gap_key, label in (
        (
            "within_orbit_span_weighted_quantiles",
            "Supported-image log-likelihood span [p05, p50, p95]",
        ),
        (
            "identity_absolute_gap_weighted_quantiles",
            "Identity-image absolute log-likelihood gap [p05, p50, p95]",
        ),
    ):
        rows.append(
            [
                label,
                *[
                    _gap_cell(
                        item["fold_telemetry"]["supported_image_log_likelihood_gaps"][
                            gap_key
                        ]
                    )
                    for item in summaries
                ],
                "; ".join(
                    f"{quantile} {_distribution_cell(distribution)}"
                    for quantile, distribution in stability["likelihood_gap_quantiles"][
                        gap_key
                    ].items()
                ),
            ]
        )
    rows.append(
        [
            "Fold/unfold postprocessing seconds",
            *[
                f"{item['fold_unfold_postprocessing_seconds']:.6g}"
                for item in summaries
            ],
            _distribution_cell(
                stability["metrics"]["fold_unfold_postprocessing_seconds"]
            ),
        ]
    )

    lines = ["# GW170817 NETSKY sampler-output analysis", ""]
    if bundle_sha256 is not None:
        lines.extend([f"Frozen bundle SHA-256: `{bundle_sha256}`.", ""])
    if blocks is not None:
        lines.extend(
            [
                "Blocking scheme: `netsky`.",
                "",
                "Resolved blocks: `"
                + json.dumps([list(block) for block in blocks], separators=(",", ":"))
                + "`.",
                "",
            ]
        )
    lines.extend(
        [
            (
                "Insertion ranks and log Z use the folded-target death/birth "
                "topology. Parameter estimates, q modes, and d_L–iota ridge "
                "occupancy use the unfolded physical posterior with true image "
                "likelihoods and normalized split quadrature weights."
            ),
            "",
            (
                "Image-sector masses, conditional entropy, support loss, "
                "likelihood gaps, quadrature-weight concentration, ridge "
                "occupancy, and postprocessing time are descriptive only; no "
                "stability threshold is applied."
            ),
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _render_report(
    summaries: list[dict[str, Any]],
    *,
    bundle_sha256: str | None,
    blocking_scheme: str | None,
    blocks: tuple[tuple[str, ...], ...] | None,
    sanity: tuple[dict[str, Any], dict[str, Any]] | None,
    netsky_stability: dict[str, Any] | None = None,
) -> str:
    if netsky_stability is not None:
        if blocking_scheme != NETSKY_BLOCKING_SCHEME:
            raise ValueError("netsky stability requires the netsky blocking scheme")
        return _render_netsky_report(
            summaries,
            bundle_sha256=bundle_sha256,
            blocks=blocks,
            stability=netsky_stability,
        )
    if blocking_scheme == NETSKY_BLOCKING_SCHEME:
        raise ValueError("netsky report rendering requires stability data")
    cross = _cross_seed(summaries)
    minimum_global_p = min(item["insertion"].global_p_value for item in summaries)
    minimum_window_p = min(item["insertion"].worst_window_p_value for item in summaries)
    ks_healthy = minimum_global_p >= 0.01 and minimum_window_p >= 0.01
    log_z_healthy = cross["max_log_z"]["value"] <= 3.0
    median_healthy = cross["max_median"]["value"] <= 1.0
    modes_healthy = cross["same_mode_count"]
    healthy = ks_healthy and log_z_healthy and median_healthy and modes_healthy

    headers = ["Metric", *[f"Seed {item['seed']}" for item in summaries], "Cross-seed"]
    rows: list[list[str]] = []
    rows.append(
        [
            "Insertion-rank KS p (global)",
            *[f"{item['insertion'].global_p_value:.4g}" for item in summaries],
            f"min={minimum_global_p:.4g} ({'PASS' if minimum_global_p >= 0.01 else 'FAIL'})",
        ]
    )
    rows.append(
        [
            "Worst raw 1000-rank-window KS p",
            *[
                f"{item['insertion'].worst_window_p_value:.4g} "
                f"({item['insertion'].worst_window_start}:"
                f"{item['insertion'].worst_window_stop})"
                for item in summaries
            ],
            f"min={minimum_window_p:.4g} ({'PASS' if minimum_window_p >= 0.01 else 'FAIL'})",
        ]
    )
    log_pair = cross["max_log_z"]["seeds"]
    rows.append(
        [
            "log Z ± sampler error",
            *[f"{item['log_z']:.6f} ± {item['log_z_error']:.3g}" for item in summaries],
            (
                f"max z={cross['max_log_z']['value']:.3g} "
                f"(seeds {log_pair[0]}/{log_pair[1]}; "
                f"{'PASS' if log_z_healthy else 'FAIL'})"
            ),
        ]
    )
    rows.append(["Entropy ESS", *[f"{item['ess']:.1f}" for item in summaries], "—"])
    for name in PARAMETERS:
        result = cross["parameter_z"][name]
        pair = result["seeds"]
        rows.append(
            [
                f"{name} median ± err [5%, 95%]",
                *[_estimate_cell(item["parameters"][name]) for item in summaries],
                (
                    f"max median z={result['value']:.3g} "
                    f"(seeds {pair[0]}/{pair[1]}; "
                    f"{'PASS' if result['value'] <= 1.0 else 'FAIL'})"
                ),
            ]
        )
    rows.append(
        [
            "q KDE mode count (>5% peak)",
            *[
                f"{item['q_mode_count']} at "
                + ", ".join(f"{value:.4g}" for value in item["q_mode_locations"])
                for item in summaries
            ],
            f"{cross['mode_counts']} ({'PASS' if modes_healthy else 'FAIL'})",
        ]
    )
    maximum = cross["max_median"]
    maximum_pair = maximum["seeds"]
    rows.append(
        [
            "Overall mode-death census",
            *["—" for _ in summaries],
            (
                f"{'HEALTHY' if healthy else 'FAIL'}; max median "
                f"z={maximum['value']:.3g} for {maximum['parameter']} "
                f"(seeds {maximum_pair[0]}/{maximum_pair[1]})"
            ),
        ]
    )

    lines = ["# GW170817 sampler-output analysis", ""]
    if bundle_sha256 is not None:
        lines.extend([f"Frozen bundle SHA-256: `{bundle_sha256}`.", ""])
    if blocking_scheme is not None and blocks is not None:
        lines.extend(
            [
                f"Blocking scheme: `{blocking_scheme}`.",
                "",
                "Resolved blocks: `"
                + json.dumps([list(block) for block in blocks], separators=(",", ":"))
                + "`.",
                "",
            ]
        )
    lines.extend(
        [
            (
                "Held-out insertion ranks use the strict "
                "`birth < contour < death` live set, uniform tie jitter, and "
                "birth-contour order with deterministic random ordering inside "
                "equal-birth cohorts. The health gates are KS p ≥ 0.01, "
                "log-Z z ≤ 3, median-shift z ≤ 1, and equal q mode counts."
            ),
            "",
            (
                "The worst-window values are raw minima over overlapping windows "
                "without a look-elsewhere correction; this preserves the plan's "
                "literal gate, but they are not familywise-calibrated p-values. "
                "The reported maximum median z is likewise uncorrected across "
                "parameters and seed pairs. "
                "The log-Z, median-shift, and mode-count gates do not depend on "
                "that scan."
            ),
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    if sanity is not None:
        damaged, reference = sanity
        lines.extend(
            [
                "",
                (
                    "Sanity check (not part of the health gate): known-damaged "
                    f"injection-000 has q={damaged['median']:.4g} "
                    f"[{damaged['lower']:.4g}, {damaged['upper']:.4g}] and "
                    f"{damaged['mode_count']} modes at {damaged['mode_locations']}, "
                    f"versus q={reference['median']:.4g} "
                    f"[{reference['lower']:.4g}, {reference['upper']:.4g}] and "
                    f"{reference['mode_count']} mode(s) for healthy injection-003 "
                    f"at {reference['mode_locations']}."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nested", nargs=3, type=Path, metavar="NESTED_NPZ")
    parser.add_argument("--run-reports", nargs=3, type=Path, metavar="REPORT_JSON")
    parser.add_argument(
        "--folded-nested",
        nargs=3,
        type=Path,
        metavar="FOLDED_NPZ",
        help=(
            "For NETSKY, the three separate folded-target death/birth artifacts "
            "aligned with the physical posterior inputs"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=_positive_int, default=100)
    parser.add_argument("--window-size", type=_positive_int, default=1000)
    parser.add_argument("--random-seed", type=int, default=170817)
    parser.add_argument("--sanity-damaged", type=Path)
    parser.add_argument("--sanity-healthy", type=Path)
    args = parser.parse_args(argv)
    if (args.sanity_damaged is None) != (args.sanity_healthy is None):
        parser.error("--sanity-damaged and --sanity-healthy must be used together")
    if args.folded_nested is not None and args.run_reports is None:
        parser.error("--folded-nested requires --run-reports")
    return args


def _resolve_folded_paths(
    report_paths: list[Path],
    reports: list[dict[str, Any]],
    *,
    explicit_paths: list[Path] | None,
) -> list[Path] | None:
    if explicit_paths is not None:
        if len(explicit_paths) != len(report_paths):
            raise ValueError("folded diagnostic overrides must align with run reports")
        return [path.expanduser().resolve() for path in explicit_paths]
    schemes = {
        report.get("config", {}).get("blocking_scheme", PAPER_BLOCKING_SCHEME)
        for report in reports
    }
    if schemes != {NETSKY_BLOCKING_SCHEME}:
        return None

    resolved = []
    for report_path, report in zip(report_paths, reports, strict=True):
        recorded = (
            report.get("results", {}).get("folded_nested_diagnostics", {}).get("path")
        )
        if not isinstance(recorded, str) or not recorded:
            raise ValueError("netsky report has no folded diagnostic artifact path")
        candidate = Path(recorded).expanduser()
        report_directory = report_path.expanduser().resolve().parent
        if candidate.is_absolute() and not candidate.is_file():
            relocated_candidates = (
                report_directory / candidate.name,
                report_directory / candidate.parent.name / candidate.name,
            )
            candidate = next(
                (path for path in relocated_candidates if path.is_file()),
                candidate,
            )
        elif not candidate.is_absolute():
            candidate = report_directory / candidate
        resolved.append(candidate.resolve())
    return resolved


def _validate_fold_telemetry(telemetry: Any) -> dict[str, Any]:
    if not isinstance(telemetry, dict):
        raise TypeError("netsky quotient-fold telemetry must be an object")
    if telemetry.get("group_order") != 8:
        raise ValueError("netsky quotient-fold telemetry requires group order 8")
    folded_points = telemetry.get("folded_points")
    if type(folded_points) is not int or folded_points < 1:
        raise ValueError("netsky folded-point count must be a positive integer")

    entropy = float(telemetry.get("normalized_conditional_image_entropy", np.nan))
    zero_support = float(telemetry.get("zero_support_image_fraction", np.nan))
    masses = np.asarray(
        telemetry.get("image_sector_posterior_masses", ()), dtype=np.float64
    )
    expected_nonidentity = float(telemetry.get("expected_nonidentity_mass", np.nan))
    if (
        not np.isfinite(entropy)
        or not 0.0 <= entropy <= 1.0
        or not np.isfinite(zero_support)
        or not 0.0 <= zero_support <= 1.0
        or masses.shape != (8,)
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0.0)
        or not np.isclose(masses.sum(), 1.0, rtol=0.0, atol=1e-8)
        or not np.isfinite(expected_nonidentity)
        or not np.isclose(
            expected_nonidentity,
            1.0 - masses[0],
            rtol=0.0,
            atol=1e-8,
        )
    ):
        raise ValueError("netsky quotient-fold mass telemetry is invalid")

    gaps = telemetry.get("supported_image_log_likelihood_gaps")
    if not isinstance(gaps, dict):
        raise TypeError("netsky quotient-fold likelihood-gap telemetry is missing")
    for name, allow_none in (
        ("within_orbit_span_weighted_quantiles", False),
        ("identity_absolute_gap_weighted_quantiles", True),
    ):
        value = gaps.get(name)
        if value is None and allow_none:
            continue
        if not isinstance(value, dict) or set(value) != {"p05", "p50", "p95"}:
            raise ValueError("netsky likelihood-gap quantiles are invalid")
        quantiles = np.asarray(
            [value["p05"], value["p50"], value["p95"]], dtype=np.float64
        )
        if (
            not np.all(np.isfinite(quantiles))
            or np.any(quantiles < 0.0)
            or np.any(np.diff(quantiles) < 0.0)
        ):
            raise ValueError("netsky likelihood-gap quantiles are invalid")
    return dict(telemetry)


def _validate_projection_accounting(
    accounting: Any,
    *,
    folded_points: int,
) -> None:
    if not isinstance(accounting, dict):
        raise TypeError("netsky fold projection accounting is missing")
    integer_fields = (
        "images_per_folded_target_callback",
        "sampler_folded_target_callbacks",
        "sampler_true_image_projections",
        "retained_folded_points_unfolded",
        "unfold_true_image_projections",
        "total_true_image_projections",
    )
    if any(
        type(accounting.get(name)) is not int or accounting[name] < 0
        for name in integer_fields
    ):
        raise ValueError("netsky fold projection accounting is invalid")
    images = accounting["images_per_folded_target_callback"]
    callbacks = accounting["sampler_folded_target_callbacks"]
    retained = accounting["retained_folded_points_unfolded"]
    sampler_projections = accounting["sampler_true_image_projections"]
    unfold_projections = accounting["unfold_true_image_projections"]
    total = accounting["total_true_image_projections"]
    if (
        images != 8
        or retained != folded_points
        or sampler_projections != images * callbacks
        or unfold_projections != images * retained
        or total != sampler_projections + unfold_projections
        or accounting.get("sampler_callback_counter")
        != "results.n_likelihood_evaluations_physical"
    ):
        raise ValueError("netsky fold projection accounting is inconsistent")


def _reports(
    paths: list[Path],
    nested_paths: list[Path],
    *,
    folded_paths: list[Path] | None = None,
) -> tuple[list[int], str, str, tuple[tuple[str, ...], ...]]:
    reports = [json.loads(path.read_text()) for path in paths]
    seeds = [int(report["config"]["seed"]) for report in reports]
    if sorted(seeds) != [0, 1, 2]:
        raise ValueError(f"expected report seeds 0, 1, 2; got {seeds}")
    data_hashes = {report["data"]["sha256"] for report in reports}
    if len(data_hashes) != 1:
        raise ValueError(f"run reports use different frozen bundles: {data_hashes}")
    blocking_schemes = {
        report["config"].get("blocking_scheme", PAPER_BLOCKING_SCHEME)
        for report in reports
    }
    if len(blocking_schemes) != 1:
        raise ValueError(
            f"run reports use different blocking schemes: {blocking_schemes}"
        )
    blocking_scheme = blocking_schemes.pop()
    reported_blocks = {
        tuple(tuple(name for name in block) for block in report["config"]["blocks"])
        for report in reports
    }
    if len(reported_blocks) != 1:
        raise ValueError("run reports use different resolved blocks")
    blocks = reported_blocks.pop()
    expected_blocks = {
        PAPER_BLOCKING_SCHEME: PAPER_BLOCKS,
        ALL_SLOW_BLOCKING_SCHEME: ALL_SLOW_BLOCKS,
        NETSKY_BLOCKING_SCHEME: NETSKY_BLOCKS,
    }.get(blocking_scheme)
    if expected_blocks is None:
        raise ValueError(f"unknown blocking scheme in run reports: {blocking_scheme!r}")
    if blocks != expected_blocks:
        raise ValueError(
            f"run reports record the wrong blocks for {blocking_scheme}: {blocks}"
        )
    if blocking_scheme == NETSKY_BLOCKING_SCHEME:
        if folded_paths is None or len(folded_paths) != len(paths):
            raise ValueError(
                "netsky reports require one separate folded diagnostic artifact "
                "per physical posterior"
            )
    elif folded_paths is not None:
        raise ValueError("separate folded diagnostic artifacts require netsky reports")

    for index, (report, nested_path) in enumerate(
        zip(reports, nested_paths, strict=True)
    ):
        config = report["config"]
        if (
            config.get("workload") != "paper-15d"
            or config.get("sampled_dimensions") != 15
            or config.get("phase_marginalization") is not True
            or config.get("distance_marginalization") is not False
        ):
            raise ValueError("run report is not the strict paper-15d configuration")
        results = report["results"]
        artifact = results.get(
            "posterior_artifact"
            if blocking_scheme == NETSKY_BLOCKING_SCHEME
            else "nested_artifact"
        )
        expected_weighting = (
            UNFOLDED_POSTERIOR_WEIGHTING
            if blocking_scheme == NETSKY_BLOCKING_SCHEME
            else WEIGHTING
        )
        if artifact is None or artifact.get("weighting") != expected_weighting:
            if blocking_scheme == NETSKY_BLOCKING_SCHEME:
                raise ValueError("netsky run report has no unfolded physical artifact")
            raise ValueError("run report has no weighted nested artifact")
        if artifact.get("sha256") != _sha256(nested_path.resolve()):
            raise ValueError(f"nested artifact hash mismatch for {nested_path}")

        if blocking_scheme != NETSKY_BLOCKING_SCHEME:
            continue
        assert folded_paths is not None
        if results.get("nested_artifact") is not None:
            raise ValueError("netsky report must not expose a legacy nested artifact")
        if tuple(tuple(block) for block in config.get("bridge_blocks", ())) != (
            NETSKY_BRIDGE_BLOCKS
        ):
            raise ValueError("netsky run report records the wrong bridge blocks")
        if (
            config.get("periodic_wrapped_covariance") is not True
            or config.get("num_gibbs_sweeps") != 2
        ):
            raise ValueError("netsky run report has an incomplete sampler config")
        fold_config = config.get("fold_symmetry")
        if (
            not isinstance(fold_config, dict)
            or fold_config.get("cos_iota") != "cos_iota"
            or fold_config.get("azimuth") != "azimuth"
            or fold_config.get("psi") != "psi"
            or not np.isfinite(fold_config.get("azimuth_reflection_center", np.nan))
        ):
            raise ValueError("netsky run report has an invalid fold config")

        posterior_count = artifact.get("count")
        required_posterior_fields = {
            *PARAMETERS,
            "log_likelihood",
            "log_weights",
        }
        if (
            artifact.get("space") != "prior"
            or artifact.get("schema_version") != 2
            or artifact.get("weight_effective_size_semantics")
            != POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
            or type(posterior_count) is not int
            or posterior_count < 1
            or not required_posterior_fields.issubset(artifact.get("fields", ()))
            or "log_likelihood_birth" in artifact.get("fields", ())
        ):
            raise ValueError("netsky weighted artifact is not a physical posterior")

        folded_path = folded_paths[index]
        folded_artifact = results.get("folded_nested_diagnostics")
        folded_count = (
            folded_artifact.get("count") if isinstance(folded_artifact, dict) else None
        )
        if (
            not isinstance(folded_artifact, dict)
            or folded_artifact.get("sha256") != _sha256(folded_path.resolve())
            or folded_artifact.get("space") != FOLDED_TARGET_SPACE
            or folded_artifact.get("semantics") != FOLDED_TARGET_SEMANTICS
            or folded_artifact.get("weighting") != FOLDED_TARGET_WEIGHTING
            or type(folded_count) is not int
            or folded_count < 1
            or set(folded_artifact.get("fields", ()))
            != {"log_likelihood", "log_likelihood_birth"}
        ):
            raise ValueError("netsky folded diagnostic artifact is invalid")

        effective_size = float(results.get("posterior_weight_effective_size", np.nan))
        log_z = float(results.get("log_Z", np.nan))
        log_z_error = float(results.get("log_Z_error", np.nan))
        postprocess_seconds = float(
            report.get("timing_seconds", {}).get("fold_unfold_postprocessing", np.nan)
        )
        if (
            results.get("posterior_weight_effective_size_semantics")
            != POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
            or not np.isfinite(effective_size)
            or effective_size < 1.0 - 1e-8
            or effective_size > posterior_count + 1e-8
            or not np.isfinite(log_z)
            or not np.isfinite(log_z_error)
            or log_z_error <= 0.0
            or not np.isfinite(postprocess_seconds)
            or postprocess_seconds <= 0.0
        ):
            raise ValueError(
                "netsky report has invalid evidence, weight, or timing data"
            )
        stored_insertion = results.get("insertion_index_diagnostic")
        if (
            not isinstance(stored_insertion, dict)
            or stored_insertion.get("method") != "discrete-uniform-kolmogorov-smirnov"
            or stored_insertion.get("n_live") != config.get("n_live")
        ):
            raise ValueError("netsky insertion-index diagnostic is invalid")
        quotient_fold = _validate_fold_telemetry(results.get("quotient_fold"))
        if (
            quotient_fold.get("completed_config") != fold_config
            or not isinstance(quotient_fold.get("model_limitation"), str)
            or not quotient_fold["model_limitation"].strip()
            or type(quotient_fold.get("batch_size")) is not int
            or quotient_fold["batch_size"] < 1
        ):
            raise ValueError("netsky completed fold report is invalid")
        if quotient_fold["folded_points"] > folded_count:
            raise ValueError(
                "netsky retained folded-point count exceeds its diagnostic artifact"
            )
        minimum_posterior_count = int(quotient_fold["folded_points"])
        maximum_posterior_count = (
            int(quotient_fold["group_order"]) * minimum_posterior_count
        )
        if not minimum_posterior_count <= posterior_count <= maximum_posterior_count:
            raise ValueError(
                "netsky posterior count is outside the quotient-orbit expansion bounds"
            )
        _validate_projection_accounting(
            quotient_fold.get("projection_accounting"),
            folded_points=minimum_posterior_count,
        )
    return seeds, data_hashes.pop(), blocking_scheme, blocks


def _atomic_write(path: Path, text: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            dir=resolved.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, resolved)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    nested_paths = [path.expanduser().resolve() for path in args.nested]
    folded_paths: list[Path] | None = None
    run_reports: list[dict[str, Any]] | None = None
    if args.run_reports is None:
        seeds = []
        for index, path in enumerate(nested_paths):
            match = re.search(r"seed(\d+)", path.stem)
            seeds.append(int(match.group(1)) if match else index)
        bundle_sha256 = None
        blocking_scheme = None
        blocks = None
    else:
        run_reports = [json.loads(path.read_text()) for path in args.run_reports]
        folded_paths = _resolve_folded_paths(
            args.run_reports,
            run_reports,
            explicit_paths=getattr(args, "folded_nested", None),
        )
        seeds, bundle_sha256, blocking_scheme, blocks = _reports(
            args.run_reports,
            nested_paths,
            folded_paths=folded_paths,
        )

    if blocking_scheme == NETSKY_BLOCKING_SCHEME:
        if folded_paths is None or run_reports is None:
            raise ValueError(
                "netsky analysis requires run reports and separate folded artifacts"
            )
        summaries = [
            _netsky_seed_summary(
                posterior_path,
                folded_path,
                run_report,
                seed=seed,
                window_size=args.window_size,
                random_seed=args.random_seed + seed,
            )
            for posterior_path, folded_path, run_report, seed in zip(
                nested_paths,
                folded_paths,
                run_reports,
                seeds,
                strict=True,
            )
        ]
    else:
        summaries = [
            _seed_summary(
                path,
                seed=seed,
                bootstrap_samples=args.bootstrap_samples,
                window_size=args.window_size,
                random_seed=args.random_seed + seed,
            )
            for path, seed in zip(nested_paths, seeds, strict=True)
        ]
    summaries.sort(key=lambda item: item["seed"])
    sanity = (
        (
            posterior_q_summary(args.sanity_damaged),
            posterior_q_summary(args.sanity_healthy),
        )
        if args.sanity_damaged is not None
        else None
    )
    netsky_stability = (
        _netsky_cross_seed_stability(summaries)
        if blocking_scheme == NETSKY_BLOCKING_SCHEME
        else None
    )
    report = _render_report(
        summaries,
        bundle_sha256=bundle_sha256,
        blocking_scheme=blocking_scheme,
        blocks=blocks,
        sanity=sanity,
        netsky_stability=netsky_stability,
    )
    _atomic_write(args.output, report)
    print(report, end="")


if __name__ == "__main__":
    main()
