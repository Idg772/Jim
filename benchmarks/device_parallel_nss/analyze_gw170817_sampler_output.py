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

PARAMETERS = ("M_c", "q", "lambda_1", "lambda_2", "d_L", "iota")
NESTED_FIELDS = {
    "log_likelihood",
    "log_likelihood_birth",
    "log_weights",
}
WEIGHTING = "normalized nested-sampling log weights"


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


def _render_report(
    summaries: list[dict[str, Any]],
    *,
    bundle_sha256: str | None,
    sanity: tuple[dict[str, Any], dict[str, Any]] | None,
) -> str:
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=_positive_int, default=100)
    parser.add_argument("--window-size", type=_positive_int, default=1000)
    parser.add_argument("--random-seed", type=int, default=170817)
    parser.add_argument("--sanity-damaged", type=Path)
    parser.add_argument("--sanity-healthy", type=Path)
    args = parser.parse_args(argv)
    if (args.sanity_damaged is None) != (args.sanity_healthy is None):
        parser.error("--sanity-damaged and --sanity-healthy must be used together")
    return args


def _reports(paths: list[Path], nested_paths: list[Path]) -> tuple[list[int], str]:
    reports = [json.loads(path.read_text()) for path in paths]
    seeds = [int(report["config"]["seed"]) for report in reports]
    if sorted(seeds) != [0, 1, 2]:
        raise ValueError(f"expected report seeds 0, 1, 2; got {seeds}")
    data_hashes = {report["data"]["sha256"] for report in reports}
    if len(data_hashes) != 1:
        raise ValueError(f"run reports use different frozen bundles: {data_hashes}")
    for report, nested_path in zip(reports, nested_paths, strict=True):
        config = report["config"]
        if (
            config.get("workload") != "paper-15d"
            or config.get("sampled_dimensions") != 15
            or config.get("phase_marginalization") is not True
            or config.get("distance_marginalization") is not False
        ):
            raise ValueError("run report is not the strict paper-15d configuration")
        artifact = report["results"].get("nested_artifact")
        if artifact is None or artifact.get("weighting") != WEIGHTING:
            raise ValueError("run report has no weighted nested artifact")
        if artifact.get("sha256") != _sha256(nested_path.resolve()):
            raise ValueError(f"nested artifact hash mismatch for {nested_path}")
    return seeds, data_hashes.pop()


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
    if args.run_reports is None:
        seeds = []
        for index, path in enumerate(nested_paths):
            match = re.search(r"seed(\d+)", path.stem)
            seeds.append(int(match.group(1)) if match else index)
        bundle_sha256 = None
    else:
        seeds, bundle_sha256 = _reports(args.run_reports, nested_paths)

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
    report = _render_report(
        summaries,
        bundle_sha256=bundle_sha256,
        sanity=sanity,
    )
    _atomic_write(args.output, report)
    print(report, end="")


if __name__ == "__main__":
    main()
