"""Validate and evaluate a matched-replicate blocking diagnostic campaign."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.injection_campaign.common import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_manifest,
    publication_eligible,
    read_catalogue,
    result_dir,
)
from benchmarks.injection_campaign.merge_staged_results import (
    ValidatedResult,
    _collect_results,
    _validate_result,
)
from benchmarks.injection_campaign.prepare_blocking_diagnostic import (
    SCHEMES,
    SEED_DERIVATION_NAMESPACE,
    SOURCE_CARRIER_TIME_ANCHOR,
    SOURCE_RANK_PARAMETERS,
    blocks_for_scheme,
    sampler_seed_for_replicate,
)

REPORT_SCHEMA_VERSION = 1
REPORT_NAME = "blocking-diagnostic-report.json"
DEFAULT_MINIMUM_TWO_SIDED_RANK_TAIL = 1.0e-8
DEFAULT_MAXIMUM_ABSOLUTE_Z = 8.0
REPORTED_PARAMETERS = SOURCE_RANK_PARAMETERS
NON_Q_GATE_PARAMETERS = tuple(
    parameter for parameter in REPORTED_PARAMETERS if parameter != "q"
)
RANK_ENDPOINT_ROUNDOFF_TOLERANCE = 1.0e-12
_SOURCE_SELECTION_RULE = (
    "replicated completed rows from a frozen corrected-anchor D=4 M=1 FSM campaign"
)
_TIMING_SELECTION = (
    "fixed corrected-anchor pathologies with matched sampler replicates; "
    "targeted blocking diagnostic only"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--source-campaign",
        type=Path,
        default=None,
        help=(
            "Frozen corrected-anchor source campaign. Defaults to the named "
            "sibling recorded by the diagnostic manifest."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Summarize completed replicates while explicitly marking the run partial.",
    )
    parser.add_argument(
        "--minimum-two-sided-rank-tail",
        type=float,
        default=DEFAULT_MINIMUM_TWO_SIDED_RANK_TAIL,
    )
    parser.add_argument(
        "--maximum-absolute-z",
        type=float,
        default=DEFAULT_MAXIMUM_ABSOLUTE_Z,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"JSON output path. Defaults to CAMPAIGN/diagnostic/{REPORT_NAME}.",
    )
    return parser.parse_args(argv)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _exact_int(value: object, *, field: str, nonnegative: bool = True) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an exact integer")
    if nonnegative and value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be finite and numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be finite and numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite and numeric")
    return result


def _rank(value: object, *, field: str) -> float:
    rank = _finite_float(value, field=field)
    if (
        rank < -RANK_ENDPOINT_ROUNDOFF_TOLERANCE
        or rank > 1.0 + RANK_ENDPOINT_ROUNDOFF_TOLERANCE
    ):
        raise ValueError(f"{field} must lie in [0, 1]")
    return min(1.0, max(0.0, rank))


def two_sided_rank_tail(rank: float) -> float:
    """Return ``2 * min(rank, 1-rank)`` without scientific clipping."""

    value = _rank(rank, field="rank")
    return 2.0 * min(value, 1.0 - value)


def _normalized_weights(log_weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    logs = np.asarray(log_weights, dtype=float)
    if logs.ndim != 1 or logs.size == 0:
        raise ValueError("log_weights must be a non-empty one-dimensional array")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("log_weights must not contain NaN or +inf")
    finite = np.isfinite(logs)
    if not np.any(finite):
        raise ValueError("log_weights must contain at least one finite value")
    maximum = float(np.max(logs[finite]))
    weights = np.exp(logs - maximum)
    normalizer = float(np.sum(weights))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("log_weights cannot be normalized")
    return weights / normalizer


def _weighted_empirical_median(
    values: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def weighted_posterior_statistics(
    samples: np.ndarray[Any, Any],
    truth: float,
    log_weights: np.ndarray[Any, Any],
    *,
    period: float | None = None,
) -> dict[str, float | str | None]:
    """Return direct weighted rank, median, spread, and truth-minus-median z.

    For right ascension, callers pass ``period=2*pi``.  Samples are then
    unwrapped into the shortest interval around the truth before computing the
    median and spread, avoiding a false gross-mode failure at the 0/2pi seam.
    The reported rank remains the campaign's ordinary prior-coordinate rank.
    """

    values = np.asarray(samples, dtype=float)
    truth_value = _finite_float(truth, field="truth")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")
    weights = _normalized_weights(log_weights)
    if weights.shape != values.shape:
        raise ValueError("samples and log_weights must have the same shape")

    rank = float(np.sum(weights[values < truth_value]))
    if period is None:
        location_values = values
        median = _weighted_empirical_median(location_values, weights)
        mean = float(np.sum(weights * location_values))
        median_delta = truth_value - median
        presentation_median = median
        presentation_mean = mean
        handling = "linear"
        gross_mode_rank = rank
        gross_mode_rank_handling = "prior-coordinate"
    else:
        period_value = _finite_float(period, field="period")
        if period_value <= 0.0:
            raise ValueError("period must be positive")
        offsets = (
            (values - truth_value + 0.5 * period_value) % period_value
        ) - 0.5 * period_value
        location_values = offsets
        median_offset = _weighted_empirical_median(offsets, weights)
        mean_offset = float(np.sum(weights * offsets))
        median_delta = -median_offset
        presentation_median = float((truth_value + median_offset) % period_value)
        presentation_mean = float((truth_value + mean_offset) % period_value)
        mean = mean_offset
        handling = "shortest-offset-about-truth"
        gross_mode_rank = float(np.sum(weights[offsets < 0.0]))
        gross_mode_rank_handling = "shortest-offset-about-truth"

    variance = float(np.sum(weights * (location_values - mean) ** 2))
    standard_deviation = math.sqrt(max(0.0, variance))
    if standard_deviation == 0.0:
        raise ValueError("weighted posterior population standard deviation is zero")
    return {
        "truth": truth_value,
        "rank": min(1.0, max(0.0, rank)),
        "two_sided_rank_tail": two_sided_rank_tail(rank),
        "gross_mode_rank": min(1.0, max(0.0, gross_mode_rank)),
        "gross_mode_two_sided_rank_tail": two_sided_rank_tail(gross_mode_rank),
        "gross_mode_rank_handling": gross_mode_rank_handling,
        "weighted_median": presentation_median,
        "weighted_mean": presentation_mean,
        "weighted_population_std": standard_deviation,
        "z_weighted_median_std": median_delta / standard_deviation,
        "effective_sample_size": float(1.0 / np.sum(weights**2)),
        "period": period,
        "location_handling": handling,
    }


def _source_campaign_path(
    campaign_dir: Path,
    provenance: Mapping[str, Any],
    explicit: Path | None,
) -> Path:
    source_name = provenance.get("source_campaign")
    if (
        not isinstance(source_name, str)
        or not source_name
        or Path(source_name).name != source_name
    ):
        raise ValueError("source campaign name must be a plain directory name")
    source = (
        explicit.expanduser().resolve()
        if explicit is not None
        else (campaign_dir.parent / source_name).resolve()
    )
    if source.name != source_name:
        raise ValueError(
            "explicit source campaign directory does not match manifest provenance"
        )
    if not source.is_dir():
        raise ValueError(f"source campaign does not exist: {source}")
    return source


def _expected_changed_variables(
    source_blocks: list[list[str]],
    diagnostic_blocks: list[list[str]],
    *,
    scheme: str,
    source_time_sampling_frame: object,
) -> dict[str, Any]:
    changed = {
        "blocks": {"source": source_blocks, "diagnostic": diagnostic_blocks},
        "catalogue_sampler_seeds": {
            "source": "one completed sampler seed per selected source row",
            "diagnostic": (
                "replicate 0 preserves the source seed; replicates >=1 use "
                "the frozen scheme-independent SHA-256 derivation"
            ),
        },
        "n_devices": {"source": 4, "diagnostic": 4},
        "num_gibbs_sweeps": {"source": 1, "diagnostic": 1},
        "sampler_scheduler": {"source": "fsm", "diagnostic": "fsm"},
        "carrier_time_anchor": {
            "source": SOURCE_CARRIER_TIME_ANCHOR,
            "diagnostic": SOURCE_CARRIER_TIME_ANCHOR,
        },
    }
    if scheme == "detector-time-fast-extrinsic":
        changed["time_sampling_frame"] = {
            "source": source_time_sampling_frame,
            "diagnostic": "H1",
        }
    return changed


def _validate_manifest_pair(
    manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    campaign_dir: Path,
    source_campaign: Path,
) -> tuple[str, list[int], int]:
    source_config = _mapping(source_manifest.get("config"), field="source config")
    if (
        source_config.get("carrier_time_anchor") != SOURCE_CARRIER_TIME_ANCHOR
        or source_config.get("n_devices") != 4
        or source_config.get("num_gibbs_sweeps") != 1
        or source_config.get("sampler_scheduler", "fsm") != "fsm"
    ):
        raise ValueError(
            "source campaign is not a corrected-anchor D=4, M=1 FSM campaign"
        )
    diagnostic = _mapping(
        manifest.get("implementation_diagnostic"),
        field="manifest.implementation_diagnostic",
    )
    scheme = diagnostic.get("blocking_scheme")
    if (
        diagnostic.get("implementation_label") != "candidate"
        or diagnostic.get("diagnostic_kind") != "blocking-scheme"
        or scheme not in SCHEMES
        or diagnostic.get("source_campaign_config_sha256")
        != source_manifest.get("config_sha256")
    ):
        raise ValueError("blocking implementation provenance is invalid")
    scheme = str(scheme)

    selection = _mapping(manifest.get("selection"), field="manifest.selection")
    source_ids_raw = selection.get("source_injection_ids")
    if (
        selection.get("rule") != _SOURCE_SELECTION_RULE
        or selection.get("ordering") != "sampler-replicate-major then source-order"
        or not isinstance(source_ids_raw, list)
        or not source_ids_raw
        or any(type(value) is not int or value < 0 for value in source_ids_raw)
        or len(source_ids_raw) != len(set(source_ids_raw))
    ):
        raise ValueError("blocking diagnostic selection is invalid")
    source_ids = list(source_ids_raw)
    replicates = _exact_int(
        selection.get("sampler_replicates_per_source"),
        field="selection.sampler_replicates_per_source",
    )
    if replicates < 1:
        raise ValueError("blocking diagnostic must have at least one replicate")
    expected_cases = len(source_ids) * replicates
    if (
        selection.get("start_inclusive") != 0
        or selection.get("stop_exclusive") != expected_cases
        or _exact_int(manifest.get("n_injections"), field="manifest.n_injections")
        != expected_cases
        or _exact_int(manifest.get("catalogue_size"), field="manifest.catalogue_size")
        != expected_cases
    ):
        raise ValueError("blocking diagnostic case count is inconsistent")

    source_blocks = [list(block) for block in source_config.get("blocks", [])]
    diagnostic_blocks = blocks_for_scheme(source_blocks, scheme)
    config = _mapping(manifest.get("config"), field="manifest.config")
    expected_config = copy.deepcopy(dict(source_config))
    expected_config.update(
        {
            "campaign": f"blocking-{scheme}-d4-m1-matched-replicate-diagnostic",
            "paper_configuration": f"Sharded {scheme} blocking diagnostic",
            "carrier_time_anchor": SOURCE_CARRIER_TIME_ANCHOR,
            "sampler_scheduler": "fsm",
            "n_devices": 4,
            "num_gibbs_sweeps": 1,
            "blocks": diagnostic_blocks,
        }
    )
    if scheme == "detector-time-fast-extrinsic":
        expected_config["time_sampling_frame"] = "H1"
    expected_timing = _mapping(
        expected_config.get("timing"), field="source config.timing"
    )
    expected_config["timing"] = copy.deepcopy(dict(expected_timing))
    expected_config["timing"]["selected_events"] = _TIMING_SELECTION
    if dict(config) != expected_config:
        raise ValueError(
            "blocking configuration changes more than labels, blocks, and the "
            "targeted timing annotation"
        )
    changed = _mapping(
        diagnostic.get("changed_variables"),
        field="implementation_diagnostic.changed_variables",
    )
    if dict(changed) != _expected_changed_variables(
        source_blocks,
        diagnostic_blocks,
        scheme=scheme,
        source_time_sampling_frame=source_config.get(
            "time_sampling_frame", "geocentric"
        ),
    ):
        raise ValueError("blocking diagnostic changed_variables are inconsistent")

    scope = _mapping(
        manifest.get("reproduction_scope"), field="manifest.reproduction_scope"
    )
    if (
        publication_eligible(manifest)
        or scope.get("iid_prior_predictive_catalogue") is not False
        or scope.get("pp_calibration_eligible") is not False
    ):
        raise ValueError("blocking diagnostic must be non-IID and nonpublication")

    source_catalogue = _mapping(
        source_manifest.get("catalogue"), field="source manifest.catalogue"
    )
    source_psd = _mapping(source_manifest.get("psd"), field="source manifest.psd")
    expected_hashes = {
        "source_manifest_sha256": file_sha256(source_campaign / "manifest.json"),
        "source_config_sha256": source_manifest.get("config_sha256"),
        "source_catalogue_sha256": source_catalogue.get("sha256"),
        "source_psd_metadata_sha256": canonical_sha256(source_psd),
    }
    for field, expected in expected_hashes.items():
        if provenance.get(field) != expected:
            raise ValueError(f"blocking provenance {field} mismatch")
    if provenance.get("kind") != "corrected-anchor-blocking-diagnostic":
        raise ValueError("blocking provenance kind is invalid")
    seed_derivation = _mapping(
        provenance.get("seed_derivation"), field="catalogue provenance.seed_derivation"
    )
    if seed_derivation != {
        "namespace": SEED_DERIVATION_NAMESPACE,
        "algorithm": "SHA-256",
        "uint32_bytes": "digest[0:4] interpreted big-endian",
        "digest_fields": [
            "namespace",
            "source_catalogue_sha256",
            "source_injection_id",
            "source_sampler_seed",
            "sampler_replicate",
        ],
        "blocking_scheme_in_digest": False,
        "replicate_zero": "preserve source sampler seed",
    }:
        raise ValueError("blocking sampler-seed derivation provenance is invalid")
    if file_sha256(campaign_dir / "manifest.json") == provenance.get(
        "source_manifest_sha256"
    ):
        raise ValueError("source and diagnostic manifests unexpectedly have one hash")
    return scheme, source_ids, replicates


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be an object: {path}")
    return value


def _validate_mapping(
    manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    *,
    campaign_dir: Path,
    source_campaign: Path,
    catalogue: Sequence[Mapping[str, Any]],
    source_catalogue: Sequence[Mapping[str, Any]],
    source_ids: Sequence[int],
    replicates: int,
) -> list[dict[str, Any]]:
    provenance = _mapping(
        _mapping(manifest.get("catalogue"), field="manifest.catalogue").get(
            "provenance"
        ),
        field="manifest.catalogue.provenance",
    )
    raw_mapping = provenance.get("mapping")
    expected_cases = len(source_ids) * replicates
    if not isinstance(raw_mapping, list) or len(raw_mapping) != expected_cases:
        raise ValueError("blocking provenance mapping has the wrong length")
    source_catalogue_sha256 = str(
        _mapping(source_manifest.get("catalogue"), field="source manifest.catalogue")[
            "sha256"
        ]
    )

    source_results: dict[int, tuple[ValidatedResult, dict[str, Any]]] = {}
    for source_id in source_ids:
        if not 0 <= source_id < len(source_catalogue):
            raise ValueError("source mapping injection ID is outside its catalogue")
        directory = result_dir(source_campaign, source_id)
        validated = _validate_result(
            directory, source_id, source_manifest, source_catalogue
        )
        source_results[source_id] = (
            validated,
            _load_json(directory / "summary.json", description="source summary"),
        )

    source_rank_maps: dict[int, dict[str, float]] = {}
    for source_id in source_ids:
        _, source_summary = source_results[source_id]
        source_summary_ranks = _mapping(
            source_summary.get("ranks"), field=f"source ranks for {source_id}"
        )
        source_rank_maps[source_id] = {
            parameter: _rank(
                source_summary_ranks.get(parameter),
                field=f"source ranks.{parameter} for {source_id}",
            )
            for parameter in REPORTED_PARAMETERS
        }

    normalized: list[dict[str, Any]] = []
    expected_id = 0
    for sampler_replicate in range(replicates):
        for source_id in source_ids:
            source_row = source_catalogue[source_id]
            validated, _ = source_results[source_id]
            expected_ranks = source_rank_maps[source_id]
            entry = _mapping(raw_mapping[expected_id], field=f"mapping {expected_id}")
            if (
                _exact_int(entry.get("diagnostic_id"), field="mapping.diagnostic_id")
                != expected_id
                or _exact_int(
                    entry.get("source_injection_id"),
                    field="mapping.source_injection_id",
                )
                != source_id
                or _exact_int(
                    entry.get("sampler_replicate"),
                    field="mapping.sampler_replicate",
                )
                != sampler_replicate
            ):
                raise ValueError(
                    "blocking mapping is not in sampler-replicate-major order"
                )
            expected_seed = sampler_seed_for_replicate(
                source_catalogue_sha256=source_catalogue_sha256,
                source_injection_id=source_id,
                source_sampler_seed=int(source_row["sampler_seed"]),
                sampler_replicate=sampler_replicate,
            )
            row = catalogue[expected_id]
            for name, source_value in source_row.items():
                expected_value = (
                    expected_id
                    if name == "injection_id"
                    else expected_seed
                    if name == "sampler_seed"
                    else source_value
                )
                if row[name] != expected_value:
                    raise ValueError(
                        f"diagnostic catalogue row {expected_id} differs from its "
                        f"source in field {name}"
                    )
            if (
                entry.get("noise_seed") != source_row["noise_seed"]
                or entry.get("source_sampler_seed") != source_row["sampler_seed"]
                or entry.get("sampler_seed") != expected_seed
            ):
                raise ValueError(f"blocking mapping seed mismatch for {expected_id}")
            recorded_ranks = _mapping(
                entry.get("source_ranks"), field=f"mapping {expected_id}.source_ranks"
            )
            if set(recorded_ranks) != set(REPORTED_PARAMETERS) or any(
                _rank(
                    recorded_ranks[parameter],
                    field=f"mapping {expected_id}.source_ranks.{parameter}",
                )
                != expected_ranks[parameter]
                for parameter in REPORTED_PARAMETERS
            ):
                raise ValueError(f"recorded source ranks mismatch for {expected_id}")
            expected_source_hashes = {
                "source_summary_sha256": file_sha256(
                    validated.directory / "summary.json"
                ),
                "source_posterior_sha256": validated.posterior_sha256,
                "source_result_fingerprint": validated.fingerprint,
                "source_result_files_sha256": validated.file_hashes,
            }
            for field, expected in expected_source_hashes.items():
                if entry.get(field) != expected:
                    raise ValueError(
                        f"blocking mapping {field} mismatch for {expected_id}"
                    )
            normalized.append(
                {
                    "diagnostic_id": expected_id,
                    "source_injection_id": source_id,
                    "sampler_replicate": sampler_replicate,
                    "noise_seed": int(source_row["noise_seed"]),
                    "source_sampler_seed": int(source_row["sampler_seed"]),
                    "sampler_seed": expected_seed,
                    "source_ranks": expected_ranks,
                    **expected_source_hashes,
                }
            )
            expected_id += 1
    return normalized


def _load_case_metrics(
    result: ValidatedResult,
    catalogue_row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    *,
    minimum_two_sided_rank_tail: float,
    maximum_absolute_z: float,
    campaign_dir: Path,
) -> dict[str, Any]:
    summary_path = result.directory / "summary.json"
    posterior_path = result.directory / "posterior.npz"
    summary = _load_json(summary_path, description="result summary")
    summary_ranks = _mapping(summary.get("ranks"), field=f"ranks in {summary_path}")
    with np.load(posterior_path, allow_pickle=False) as archive:
        log_weights = np.asarray(archive["log_weights"], dtype=float)
        arrays = {
            parameter: np.asarray(archive[parameter], dtype=float)
            for parameter in REPORTED_PARAMETERS
        }

    parameters: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for parameter in REPORTED_PARAMETERS:
        statistics = weighted_posterior_statistics(
            arrays[parameter],
            float(catalogue_row[parameter]),
            log_weights,
            period=2.0 * math.pi if parameter == "ra" else None,
        )
        reported_rank = _rank(
            summary_ranks.get(parameter), field=f"ranks.{parameter} in {summary_path}"
        )
        if not math.isclose(
            float(statistics["rank"]), reported_rank, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"weighted rank recomputation mismatch for {parameter}: {summary_path}"
            )
        tail_pass = (
            float(statistics["gross_mode_two_sided_rank_tail"])
            >= minimum_two_sided_rank_tail
        )
        z_pass = abs(float(statistics["z_weighted_median_std"])) <= maximum_absolute_z
        eligible = parameter in NON_Q_GATE_PARAMETERS
        parameter_pass: bool | None = tail_pass and z_pass if eligible else None
        gate = {
            "eligible": eligible,
            "minimum_two_sided_rank_tail_pass": tail_pass if eligible else None,
            "maximum_absolute_z_pass": z_pass if eligible else None,
            "pass": parameter_pass,
            "exemption": (
                None
                if eligible
                else "q is reported but exempt from the blocking gross-mode gate"
            ),
        }
        if eligible and not tail_pass:
            failures.append(
                {
                    "parameter": parameter,
                    "criterion": "minimum-two-sided-rank-tail",
                    "value": statistics["gross_mode_two_sided_rank_tail"],
                    "threshold": minimum_two_sided_rank_tail,
                }
            )
        if eligible and not z_pass:
            failures.append(
                {
                    "parameter": parameter,
                    "criterion": "maximum-absolute-z",
                    "value": abs(float(statistics["z_weighted_median_std"])),
                    "threshold": maximum_absolute_z,
                }
            )
        parameters[parameter] = {**statistics, "gross_mode_gate": gate}

    return {
        "diagnostic_id": mapping["diagnostic_id"],
        "source_injection_id": mapping["source_injection_id"],
        "sampler_replicate": mapping["sampler_replicate"],
        "noise_seed": mapping["noise_seed"],
        "source_sampler_seed": mapping["source_sampler_seed"],
        "sampler_seed": mapping["sampler_seed"],
        "source_ranks": mapping["source_ranks"],
        "parameters": parameters,
        "gross_mode_gate": {
            "pass": not failures,
            "failures": failures,
            "q_exempt": True,
        },
        "inputs": {
            "summary": summary_path.relative_to(campaign_dir).as_posix(),
            "summary_sha256": file_sha256(summary_path),
            "posterior": posterior_path.relative_to(campaign_dir).as_posix(),
            "posterior_sha256": result.posterior_sha256,
            "result_fingerprint": result.fingerprint,
            "posterior_samples": result.posterior_samples,
        },
    }


def _linear_span(values: Sequence[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _circular_span(values: Sequence[float], period: float) -> float:
    if len(values) < 2:
        return 0.0
    return max(
        min(abs(first - second), period - abs(first - second))
        for first in values
        for second in values
    )


def _cross_replicate_spread(
    cases: Sequence[Mapping[str, Any]],
    *,
    source_ids: Sequence[int],
    replicates: int,
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    all_rank_spans: list[float] = []
    all_z_spans: list[float] = []
    all_median_spans: list[float] = []
    for source_id in source_ids:
        source_cases = sorted(
            (case for case in cases if int(case["source_injection_id"]) == source_id),
            key=lambda case: int(case["sampler_replicate"]),
        )
        parameter_spreads: dict[str, Any] = {}
        for parameter in REPORTED_PARAMETERS:
            statistics = [case["parameters"][parameter] for case in source_cases]
            ranks = [float(value["rank"]) for value in statistics]
            tails = [float(value["two_sided_rank_tail"]) for value in statistics]
            medians = [float(value["weighted_median"]) for value in statistics]
            zs = [float(value["z_weighted_median_std"]) for value in statistics]
            rank_span = _linear_span(ranks)
            z_span = _linear_span(zs)
            median_span = (
                _circular_span(medians, 2.0 * math.pi)
                if parameter == "ra"
                else _linear_span(medians)
            )
            all_rank_spans.append(rank_span)
            all_z_spans.append(z_span)
            all_median_spans.append(median_span)
            parameter_spreads[parameter] = {
                "completed_replicates": len(statistics),
                "rank_span": rank_span,
                "two_sided_rank_tail_span": _linear_span(tails),
                "minimum_two_sided_rank_tail": min(tails) if tails else None,
                "weighted_median_span": median_span,
                "z_span": z_span,
                "z_population_std_across_replicates": (
                    float(np.std(zs)) if zs else None
                ),
                "maximum_absolute_z": max(map(abs, zs)) if zs else None,
            }
        groups.append(
            {
                "source_injection_id": source_id,
                "expected_replicates": replicates,
                "completed_replicates": len(source_cases),
                "complete": len(source_cases) == replicates,
                "completed_sampler_replicates": [
                    int(case["sampler_replicate"]) for case in source_cases
                ],
                "parameters": parameter_spreads,
            }
        )
    return {
        "definition": (
            "Descriptive within-source spread across matched sampler-seed "
            "replicates; no population-calibration interpretation."
        ),
        "groups": groups,
        "complete_source_groups": sum(group["complete"] for group in groups),
        "requested_source_groups": len(source_ids),
        "maximum_rank_span": max(all_rank_spans, default=0.0),
        "maximum_weighted_median_span": max(all_median_spans, default=0.0),
        "maximum_z_span": max(all_z_spans, default=0.0),
    }


def _parameter_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for parameter in REPORTED_PARAMETERS:
        values = [case["parameters"][parameter] for case in cases]
        ranks = [float(value["rank"]) for value in values]
        tails = [float(value["gross_mode_two_sided_rank_tail"]) for value in values]
        medians = [float(value["weighted_median"]) for value in values]
        zs = [float(value["z_weighted_median_std"]) for value in values]
        summary[parameter] = {
            "completed_cases": len(values),
            "rank_minimum": min(ranks),
            "rank_median": float(np.median(ranks)),
            "rank_maximum": max(ranks),
            "minimum_gross_mode_two_sided_rank_tail": min(tails),
            "weighted_medians": medians,
            "z_median": float(np.median(zs)),
            "maximum_absolute_z": max(map(abs, zs)),
            "gross_mode_gate_eligible": parameter in NON_Q_GATE_PARAMETERS,
        }
    return summary


def _validate_thresholds(
    minimum_two_sided_rank_tail: float, maximum_absolute_z: float
) -> tuple[float, float]:
    minimum_tail = _finite_float(
        minimum_two_sided_rank_tail, field="minimum_two_sided_rank_tail"
    )
    maximum_z = _finite_float(maximum_absolute_z, field="maximum_absolute_z")
    if not 0.0 <= minimum_tail <= 1.0:
        raise ValueError("minimum_two_sided_rank_tail must lie in [0, 1]")
    if maximum_z < 0.0:
        raise ValueError("maximum_absolute_z must be non-negative")
    return minimum_tail, maximum_z


def evaluate_blocking_diagnostic(
    campaign_dir: Path,
    *,
    source_campaign: Path | None = None,
    allow_partial: bool = False,
    minimum_two_sided_rank_tail: float = DEFAULT_MINIMUM_TWO_SIDED_RANK_TAIL,
    maximum_absolute_z: float = DEFAULT_MAXIMUM_ABSOLUTE_Z,
) -> dict[str, Any]:
    """Return a validated gross-mode report for one blocking scheme."""

    minimum_tail, maximum_z = _validate_thresholds(
        minimum_two_sided_rank_tail, maximum_absolute_z
    )
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    catalogue_metadata = _mapping(manifest.get("catalogue"), field="manifest.catalogue")
    provenance = _mapping(
        catalogue_metadata.get("provenance"), field="manifest.catalogue.provenance"
    )
    source_campaign = _source_campaign_path(campaign_dir, provenance, source_campaign)
    source_manifest = load_manifest(source_campaign)
    scheme, source_ids, replicates = _validate_manifest_pair(
        manifest,
        source_manifest,
        provenance,
        campaign_dir=campaign_dir,
        source_campaign=source_campaign,
    )

    catalogue = read_catalogue(campaign_dir / str(catalogue_metadata["path"]))
    source_catalogue_metadata = _mapping(
        source_manifest.get("catalogue"), field="source manifest.catalogue"
    )
    source_catalogue = read_catalogue(
        source_campaign / str(source_catalogue_metadata["path"])
    )
    expected_cases = len(source_ids) * replicates
    if len(catalogue) != expected_cases:
        raise ValueError("blocking diagnostic catalogue has the wrong row count")
    mapping = _validate_mapping(
        manifest,
        source_manifest,
        campaign_dir=campaign_dir,
        source_campaign=source_campaign,
        catalogue=catalogue,
        source_catalogue=source_catalogue,
        source_ids=source_ids,
        replicates=replicates,
    )

    results, incomplete_ids_raw = _collect_results(campaign_dir, manifest, catalogue)
    completed_ids = [result.injection_id for result in results]
    incomplete_ids = list(incomplete_ids_raw)
    if completed_ids != list(range(len(completed_ids))):
        raise ValueError(
            "completed blocking results must form a contiguous staged prefix"
        )
    missing_ids = [
        diagnostic_id
        for diagnostic_id in range(expected_cases)
        if diagnostic_id not in completed_ids
    ]
    if missing_ids and not allow_partial:
        raise ValueError(
            "blocking diagnostic requires every matched replicate to have a "
            "complete validated posterior result"
        )
    if not results:
        raise ValueError("blocking diagnostic has no complete validated results")

    result_by_id = {result.injection_id: result for result in results}
    cases = [
        _load_case_metrics(
            result_by_id[diagnostic_id],
            catalogue[diagnostic_id],
            mapping[diagnostic_id],
            minimum_two_sided_rank_tail=minimum_tail,
            maximum_absolute_z=maximum_z,
            campaign_dir=campaign_dir,
        )
        for diagnostic_id in completed_ids
    ]
    failed_ids = [
        int(case["diagnostic_id"])
        for case in cases
        if not case["gross_mode_gate"]["pass"]
    ]
    complete = not missing_ids
    gate_pass = complete and not failed_ids
    if failed_ids:
        decision = "gross-mode-detected"
    elif complete:
        decision = "no-gross-mode-detected"
    else:
        decision = "partial-no-gross-mode-detected"

    config = _mapping(manifest.get("config"), field="manifest.config")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "campaign": config.get("campaign"),
        "config_sha256": manifest["config_sha256"],
        "blocking_scheme": scheme,
        "status": "complete" if complete else "partial",
        "decision": decision,
        "selection": {
            "source_injection_ids": source_ids,
            "sampler_replicates_per_source": replicates,
            "requested_cases": expected_cases,
            "completed_cases": len(completed_ids),
            "complete": complete,
            "completed_diagnostic_ids": completed_ids,
            "missing_diagnostic_ids": missing_ids,
            "incomplete_artifact_diagnostic_ids": incomplete_ids,
        },
        "gross_mode_gate": {
            "pass": gate_pass,
            "completed_cases_pass": not failed_ids,
            "failed_diagnostic_ids": failed_ids,
            "parameters": list(NON_Q_GATE_PARAMETERS),
            "q_exempt": True,
            "thresholds": {
                "minimum_two_sided_rank_tail_inclusive": minimum_tail,
                "maximum_absolute_z_inclusive": maximum_z,
            },
            "interpretation": (
                "This is an explicit non-q gross-mode screen on selected rows. "
                "It is not a P-P calibration test or calibration claim."
            ),
        },
        "methodology": {
            "rank": "sum normalized nested-sampling weight for samples < truth",
            "two_sided_rank_tail": "2 * min(rank, 1 - rank), without clipping",
            "gross_mode_rank": (
                "prior-coordinate rank for linear parameters; for ra, the "
                "weighted fraction of shortest signed offsets below zero"
            ),
            "gross_mode_two_sided_rank_tail": (
                "2 * min(gross_mode_rank, 1 - gross_mode_rank), without clipping"
            ),
            "weighted_median": (
                "first ordered value with cumulative normalized weight >= 0.5"
            ),
            "z_weighted_median_std": (
                "(truth - weighted_median) / weighted_population_std"
            ),
            "ra_location": (
                "samples unwrapped by shortest 2pi offset about truth for median, "
                "spread, z, and the seam-invariant gross-mode rank; the reported "
                "P-P rank remains in the original prior coordinate"
            ),
            "posterior_weighting": "original nested-sampling log weights",
            "resampling": False,
            "pp_calibration_claim": False,
        },
        "parameter_summary": _parameter_summary(cases),
        "cross_replicate_spread": _cross_replicate_spread(
            cases, source_ids=source_ids, replicates=replicates
        ),
        "provenance": {
            "diagnostic_manifest": "manifest.json",
            "diagnostic_manifest_sha256": file_sha256(campaign_dir / "manifest.json"),
            "diagnostic_catalogue": str(catalogue_metadata["path"]),
            "diagnostic_catalogue_sha256": file_sha256(
                campaign_dir / str(catalogue_metadata["path"])
            ),
            "source_campaign": provenance["source_campaign"],
            "source_manifest_sha256": provenance["source_manifest_sha256"],
            "source_config_sha256": provenance["source_config_sha256"],
            "source_catalogue_sha256": provenance["source_catalogue_sha256"],
            "ordered_mapping": mapping,
        },
        "cases": cases,
    }


def write_blocking_diagnostic_report(
    campaign_dir: Path, report: Mapping[str, Any], *, output: Path | None = None
) -> Path:
    """Atomically write the JSON diagnostic product."""

    path = (
        output.expanduser().resolve()
        if output is not None
        else campaign_dir.expanduser().resolve() / "diagnostic" / REPORT_NAME
    )
    atomic_write_json(path, report)
    return path


def _print_report(report: Mapping[str, Any]) -> None:
    for case in report["cases"]:
        fields = []
        for parameter in REPORTED_PARAMETERS:
            metrics = case["parameters"][parameter]
            fields.append(
                f"{parameter}:rank={metrics['rank']:.6g},"
                f"z={metrics['z_weighted_median_std']:+.3f}"
            )
        print(
            f"diagnostic {case['diagnostic_id']:02d} "
            f"(source {case['source_injection_id']:03d}, "
            f"replicate {case['sampler_replicate']}): "
            + "; ".join(fields)
            + f"; gross-mode-pass={case['gross_mode_gate']['pass']}"
        )
    print(
        f"Decision: {report['decision']} "
        f"({report['selection']['completed_cases']}/"
        f"{report['selection']['requested_cases']} complete)."
    )
    print(report["gross_mode_gate"]["interpretation"])


def run_evaluation(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.expanduser().resolve()
    report = evaluate_blocking_diagnostic(
        campaign_dir,
        source_campaign=args.source_campaign,
        allow_partial=bool(args.allow_partial),
        minimum_two_sided_rank_tail=args.minimum_two_sided_rank_tail,
        maximum_absolute_z=args.maximum_absolute_z,
    )
    report_path = write_blocking_diagnostic_report(
        campaign_dir, report, output=args.report
    )
    _print_report(report)
    print(f"Report: {report_path}")
    return 0 if report["decision"] == "no-gross-mode-detected" else 1


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_evaluation(_parse_args(argv)))


if __name__ == "__main__":
    main()
