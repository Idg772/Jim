"""Compare five frozen FSM pathologies with pinned paper-baseline recoveries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.injection_campaign.common import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    load_manifest,
    read_catalogue,
    result_dir,
)
from benchmarks.injection_campaign.merge_staged_results import (
    _collect_results,
    _validate_result,
)
from benchmarks.injection_campaign.prepare_baseline_diagnostic import (
    PAPER_BASELINE_REVISION,
    PAPER_BASELINE_TREE_SHA256,
    PAPER_CONFIGURATION,
)

EXPECTED_CASES = 5
RANK_PARAMETERS = ("q", "ra", "t_c")
REPORT_SCHEMA_VERSION = 1
REPORT_DIRECTORY = "diagnostic"
REPORT_JSON = "paper-baseline-rank-comparison.json"
REPORT_CSV = "paper-baseline-rank-comparison.csv"
LOG_ZERO_FLOOR = sys.float_info.min
RANK_ENDPOINT_ROUNDOFF_TOLERANCE = 1.0e-12
ATTRIBUTION_CAVEAT = (
    "This targeted comparison is directional evidence, not a single-variable "
    "kernel isolation: the paper baseline uses D=1 and M=3 while the source FSM "
    "campaign uses D=4 and M=1, so implementation, device topology, paper "
    "configuration, and Gibbs-sweep count all change together."
)
CSV_FIELDS = (
    "diagnostic_id",
    "source_injection_id",
    "parameter",
    "source_rank",
    "baseline_rank",
    "source_two_sided_tail_probability",
    "baseline_two_sided_tail_probability",
    "source_tail_severity_log10",
    "baseline_tail_severity_log10",
    "tail_severity_delta_baseline_minus_source",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--source-campaign",
        type=Path,
        default=None,
        help=(
            "Frozen FSM source campaign. Defaults to the named sibling recorded "
            "in the diagnostic manifest."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Evaluate every completed recovery after an explicitly aborted run. "
            "Missing cases are recorded and excluded from aggregates; the default "
            "requires all five."
        ),
    )
    return parser.parse_args(argv)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _exact_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an exact integer")
    return value


def _rank(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        rank = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if (
        not math.isfinite(rank)
        or rank < -RANK_ENDPOINT_ROUNDOFF_TOLERANCE
        or rank > 1.0 + RANK_ENDPOINT_ROUNDOFF_TOLERANCE
    ):
        raise ValueError(f"{field} must be finite and lie in [0, 1]")
    # Weighted reductions can overshoot an exact endpoint by a few ulps. This
    # only normalizes representational roundoff; material excursions are errors.
    return min(1.0, max(0.0, rank))


def tail_metrics(rank: float) -> dict[str, float | bool]:
    """Return an exact two-sided tail probability and log-only severity.

    Neither the rank nor its tail probability is clipped. Only the input to the
    logarithm is floored, which keeps exact endpoint ranks representable without
    producing an infinite value in JSON or CSV.
    """

    value = _rank(rank, field="rank")
    tail_probability = 2.0 * min(value, 1.0 - value)
    log_input = max(tail_probability, LOG_ZERO_FLOOR)
    severity = -math.log10(log_input)
    if severity == 0.0:
        severity = 0.0
    return {
        "rank": value,
        "two_sided_tail_probability": tail_probability,
        "tail_severity_log10": severity,
        "log_input_floored": tail_probability == 0.0,
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
        raise ValueError("manifest source campaign name must be a plain directory name")
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


def _validate_manifest_pair(
    manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    campaign_dir: Path,
    source_campaign: Path,
) -> None:
    for label, value in (
        ("n_injections", manifest.get("n_injections")),
        ("catalogue_size", manifest.get("catalogue_size")),
    ):
        if _exact_int(value, field=f"manifest.{label}") != EXPECTED_CASES:
            raise ValueError(f"baseline diagnostic must contain {EXPECTED_CASES} cases")

    selection = _mapping(manifest.get("selection"), field="manifest.selection")
    source_ids = selection.get("source_injection_ids")
    if (
        selection.get("rule") != "ranked pathological rows from a frozen FSM campaign"
        or selection.get("start_inclusive") != 0
        or selection.get("stop_exclusive") != EXPECTED_CASES
        or not isinstance(source_ids, list)
        or len(source_ids) != EXPECTED_CASES
        or any(type(value) is not int for value in source_ids)
        or len(set(source_ids)) != EXPECTED_CASES
    ):
        raise ValueError("baseline diagnostic selection is invalid or unordered")

    config = _mapping(manifest.get("config"), field="manifest.config")
    source_config = _mapping(source_manifest.get("config"), field="source config")
    if (
        config.get("paper_configuration") != PAPER_CONFIGURATION
        or config.get("n_devices") != 1
        or config.get("num_gibbs_sweeps") != 3
        or "sampler_scheduler" in config
    ):
        raise ValueError("baseline campaign is not the paper High-Res D=1/M=3 setup")
    if (
        "sampler_scheduler" in source_config
        or source_config.get("n_devices") != 4
        or source_config.get("num_gibbs_sweeps") != 1
    ):
        raise ValueError("source campaign is not the FSM D=4/M=1 setup")

    diagnostic = _mapping(
        manifest.get("baseline_diagnostic"), field="manifest.baseline_diagnostic"
    )
    if (
        diagnostic.get("implementation_label") != "paper-baseline"
        or diagnostic.get("implementation_revision") != PAPER_BASELINE_REVISION
        or diagnostic.get("implementation_tree_sha256") != PAPER_BASELINE_TREE_SHA256
        or diagnostic.get("paper_configuration") != PAPER_CONFIGURATION
        or diagnostic.get("paper_timing_available") is not False
        or diagnostic.get("source_campaign_config_sha256")
        != source_manifest.get("config_sha256")
    ):
        raise ValueError("baseline implementation or source provenance is invalid")

    changed = _mapping(
        diagnostic.get("changed_variables"),
        field="manifest.baseline_diagnostic.changed_variables",
    )
    expected_changes = {
        "paper_configuration": {
            "source": source_config.get("paper_configuration"),
            "diagnostic": PAPER_CONFIGURATION,
        },
        "n_devices": {"source": 4, "diagnostic": 1},
        "num_gibbs_sweeps": {"source": 1, "diagnostic": 3},
    }
    for field, expected in expected_changes.items():
        if changed.get(field) != expected:
            raise ValueError(f"baseline changed_variables.{field} is inconsistent")
    if changed.get("implementation") != "pinned paper baseline":
        raise ValueError("baseline implementation change is not recorded")

    scope = _mapping(
        manifest.get("reproduction_scope"), field="manifest.reproduction_scope"
    )
    if (
        scope.get("iid_prior_predictive_catalogue") is not False
        or scope.get("pp_calibration_eligible") is not False
    ):
        raise ValueError(
            "targeted diagnostic must be marked non-IID and nonpublication"
        )

    expected_source_hashes = {
        "source_manifest_sha256": file_sha256(source_campaign / "manifest.json"),
        "source_config_sha256": source_manifest.get("config_sha256"),
        "source_catalogue_sha256": source_manifest.get("catalogue", {}).get("sha256"),
    }
    for field, expected in expected_source_hashes.items():
        if provenance.get(field) != expected:
            raise ValueError(f"diagnostic provenance {field} mismatch")
    if diagnostic.get("source_campaign_config_sha256") != provenance.get(
        "source_config_sha256"
    ):
        raise ValueError("baseline and catalogue source config hashes disagree")
    if file_sha256(campaign_dir / "manifest.json") == provenance.get(
        "source_manifest_sha256"
    ):
        raise ValueError("diagnostic and source manifests unexpectedly have one hash")


def _validate_mapping(
    manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    *,
    campaign_dir: Path,
    source_campaign: Path,
    catalogue: Sequence[Mapping[str, Any]],
    source_catalogue: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    catalogue_metadata = _mapping(manifest["catalogue"], field="manifest.catalogue")
    provenance = _mapping(
        catalogue_metadata.get("provenance"), field="manifest.catalogue.provenance"
    )
    raw_mapping = provenance.get("mapping")
    if not isinstance(raw_mapping, list) or len(raw_mapping) != EXPECTED_CASES:
        raise ValueError(
            f"source mapping must contain exactly {EXPECTED_CASES} entries"
        )
    source_ids = _mapping(manifest["selection"], field="manifest.selection")[
        "source_injection_ids"
    ]

    normalized: list[dict[str, Any]] = []
    for expected_id, raw_entry in enumerate(raw_mapping):
        entry = _mapping(raw_entry, field=f"source mapping {expected_id}")
        diagnostic_id = _exact_int(
            entry.get("diagnostic_id"),
            field=f"source mapping {expected_id}.diagnostic_id",
        )
        source_id = _exact_int(
            entry.get("source_injection_id"),
            field=f"source mapping {expected_id}.source_injection_id",
        )
        if diagnostic_id != expected_id or source_id != source_ids[expected_id]:
            raise ValueError("source mapping is not in diagnostic selection order")
        if not 0 <= source_id < len(source_catalogue):
            raise ValueError("source mapping injection ID is outside its catalogue")

        diagnostic_row = catalogue[diagnostic_id]
        source_row = source_catalogue[source_id]
        for name in diagnostic_row:
            expected = diagnostic_id if name == "injection_id" else source_row[name]
            if diagnostic_row[name] != expected:
                raise ValueError(
                    f"diagnostic catalogue row {diagnostic_id} differs from source "
                    f"in field {name}"
                )
        if (
            entry.get("noise_seed") != source_row["noise_seed"]
            or entry.get("sampler_seed") != source_row["sampler_seed"]
        ):
            raise ValueError(f"source mapping {expected_id} seed mismatch")

        source_directory = result_dir(source_campaign, source_id)
        _validate_result(
            source_directory,
            source_id,
            source_manifest,
            source_catalogue,
        )
        source_summary_path = source_directory / "summary.json"
        expected_summary_hash = entry.get("source_summary_sha256")
        if (
            not isinstance(expected_summary_hash, str)
            or file_sha256(source_summary_path) != expected_summary_hash
        ):
            raise ValueError(
                f"source summary hash mismatch for diagnostic {expected_id}"
            )
        try:
            source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid source summary: {source_summary_path}"
            ) from error
        summary_ranks = _mapping(
            source_summary.get("ranks"), field=f"source ranks {source_id}"
        )
        recorded_ranks = _mapping(
            entry.get("source_ranks"), field=f"recorded source ranks {source_id}"
        )
        if set(recorded_ranks) != set(RANK_PARAMETERS):
            raise ValueError(f"recorded source ranks are incomplete for {source_id}")
        normalized_ranks: dict[str, float] = {}
        for parameter in RANK_PARAMETERS:
            summary_rank = _rank(
                summary_ranks.get(parameter),
                field=f"source summary ranks.{parameter} for {source_id}",
            )
            recorded_rank = _rank(
                recorded_ranks.get(parameter),
                field=f"recorded source ranks.{parameter} for {source_id}",
            )
            if recorded_rank != summary_rank:
                raise ValueError(
                    f"recorded source rank mismatch for {parameter} in {source_id}"
                )
            normalized_ranks[parameter] = summary_rank
        normalized.append(
            {
                "diagnostic_id": diagnostic_id,
                "source_injection_id": source_id,
                "source_ranks": normalized_ranks,
                "source_summary_sha256": expected_summary_hash,
                "noise_seed": source_row["noise_seed"],
                "sampler_seed": source_row["sampler_seed"],
            }
        )
    return normalized


def _validated_baseline_ranks(
    campaign_dir: Path,
    manifest: Mapping[str, Any],
    catalogue: Sequence[Mapping[str, Any]],
    *,
    allow_partial: bool,
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[str, Any]],
    list[int],
    list[int],
    list[int],
]:
    results, incomplete_ids = _collect_results(campaign_dir, manifest, catalogue)
    result_ids = [result.injection_id for result in results]
    missing_ids = [
        injection_id
        for injection_id in range(EXPECTED_CASES)
        if injection_id not in result_ids
    ]
    if missing_ids and not allow_partial:
        raise ValueError(
            f"baseline diagnostic requires all {EXPECTED_CASES} complete validated "
            "posterior results"
        )
    if not result_ids:
        raise ValueError(
            "partial baseline diagnostic requires at least one complete validated "
            "posterior result"
        )

    ranks_by_id: dict[int, dict[str, float]] = {}
    inputs_by_id: dict[int, dict[str, Any]] = {}
    for result in results:
        summary_path = result.directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ranks = _mapping(summary.get("ranks"), field=f"ranks in {summary_path}")
        ranks_by_id[result.injection_id] = {
            parameter: _rank(
                ranks.get(parameter),
                field=f"ranks.{parameter} in {summary_path}",
            )
            for parameter in RANK_PARAMETERS
        }
        inputs_by_id[result.injection_id] = {
            "summary": summary_path.relative_to(campaign_dir).as_posix(),
            "summary_sha256": file_sha256(summary_path),
            "posterior": (result.directory / "posterior.npz")
            .relative_to(campaign_dir)
            .as_posix(),
            "posterior_sha256": result.posterior_sha256,
            "posterior_samples": result.posterior_samples,
        }
    return (
        ranks_by_id,
        inputs_by_id,
        result_ids,
        missing_ids,
        list(incomplete_ids),
    )


def _direction(delta: float) -> str:
    if delta < 0.0:
        return "less-tail-pathological"
    if delta > 0.0:
        return "more-tail-pathological"
    return "unchanged"


def evaluate_baseline_diagnostic(
    campaign_dir: Path,
    *,
    source_campaign: Path | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate and compare completed targeted source/baseline rank triplets."""

    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    catalogue_metadata = _mapping(manifest.get("catalogue"), field="manifest.catalogue")
    provenance = _mapping(
        catalogue_metadata.get("provenance"), field="manifest.catalogue.provenance"
    )
    source_campaign = _source_campaign_path(campaign_dir, provenance, source_campaign)
    source_manifest = load_manifest(source_campaign)
    _validate_manifest_pair(
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
    if len(catalogue) != EXPECTED_CASES:
        raise ValueError(f"diagnostic catalogue must contain {EXPECTED_CASES} rows")
    mapping = _validate_mapping(
        manifest,
        source_manifest,
        campaign_dir=campaign_dir,
        source_campaign=source_campaign,
        catalogue=catalogue,
        source_catalogue=source_catalogue,
    )
    (
        baseline_ranks,
        baseline_inputs,
        completed_ids,
        missing_ids,
        incomplete_artifact_ids,
    ) = _validated_baseline_ranks(
        campaign_dir,
        manifest,
        catalogue,
        allow_partial=allow_partial,
    )

    cases: list[dict[str, Any]] = []
    source_total = 0.0
    baseline_total = 0.0
    improved_comparisons = 0
    for entry in mapping:
        diagnostic_id = entry["diagnostic_id"]
        if diagnostic_id not in baseline_ranks:
            continue
        comparisons: dict[str, Any] = {}
        case_source_total = 0.0
        case_baseline_total = 0.0
        for parameter in RANK_PARAMETERS:
            source_metrics = tail_metrics(entry["source_ranks"][parameter])
            baseline_metrics = tail_metrics(baseline_ranks[diagnostic_id][parameter])
            delta = float(baseline_metrics["tail_severity_log10"]) - float(
                source_metrics["tail_severity_log10"]
            )
            case_source_total += float(source_metrics["tail_severity_log10"])
            case_baseline_total += float(baseline_metrics["tail_severity_log10"])
            if delta < 0.0:
                improved_comparisons += 1
            comparisons[parameter] = {
                "source": source_metrics,
                "baseline": baseline_metrics,
                "tail_severity_delta_baseline_minus_source": delta,
                "direction": _direction(delta),
            }
        case_delta = case_baseline_total - case_source_total
        source_total += case_source_total
        baseline_total += case_baseline_total
        cases.append(
            {
                "diagnostic_id": diagnostic_id,
                "source_injection_id": entry["source_injection_id"],
                "noise_seed": entry["noise_seed"],
                "sampler_seed": entry["sampler_seed"],
                "source_summary_sha256": entry["source_summary_sha256"],
                "comparisons": comparisons,
                "combined_tail_severity": {
                    "source": case_source_total,
                    "baseline": case_baseline_total,
                    "delta_baseline_minus_source": case_delta,
                    "direction": _direction(case_delta),
                },
                "baseline_inputs": baseline_inputs[diagnostic_id],
            }
        )

    total_delta = baseline_total - source_total
    source_config = _mapping(source_manifest["config"], field="source config")
    baseline_config = _mapping(manifest["config"], field="baseline config")
    direction = _direction(total_delta)
    complete = not missing_ids
    status = "complete" if complete else "partial-aborted"
    source_id_by_diagnostic_id = {
        entry["diagnostic_id"]: entry["source_injection_id"] for entry in mapping
    }
    missing_cases = [
        {
            "diagnostic_id": diagnostic_id,
            "source_injection_id": source_id_by_diagnostic_id[diagnostic_id],
            "status": (
                "incomplete-artifacts"
                if diagnostic_id in incomplete_artifact_ids
                else "not-run"
            ),
        }
        for diagnostic_id in missing_ids
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "campaign": baseline_config.get("campaign"),
        "config_sha256": manifest["config_sha256"],
        "status": status,
        "selection": {
            "requested_cases": EXPECTED_CASES,
            "completed_cases": len(completed_ids),
            "missing_cases": len(missing_ids),
            "complete": complete,
            "aborted": not complete,
            "requested_diagnostic_ids": list(range(EXPECTED_CASES)),
            "requested_source_injection_ids": [
                entry["source_injection_id"] for entry in mapping
            ],
            "completed_diagnostic_ids": completed_ids,
            "missing_diagnostic_ids": missing_ids,
            "completed_source_injection_ids": [
                source_id_by_diagnostic_id[diagnostic_id]
                for diagnostic_id in completed_ids
            ],
            "missing_source_injection_ids": [
                source_id_by_diagnostic_id[diagnostic_id]
                for diagnostic_id in missing_ids
            ],
            "incomplete_artifact_diagnostic_ids": incomplete_artifact_ids,
            "rank_parameters": list(RANK_PARAMETERS),
        },
        "methodology": {
            "rank": "sum normalized nested-sampling weight for samples < truth",
            "two_sided_tail_probability": "2 * min(rank, 1 - rank)",
            "tail_severity_log10": "-log10(two_sided_tail_probability)",
            "rank_or_tail_clipping": False,
            "rank_endpoint_roundoff_normalization": {
                "absolute_tolerance": RANK_ENDPOINT_ROUNDOFF_TOLERANCE,
                "rule": (
                    "normalize only ranks within tolerance of 0 or 1; reject "
                    "material excursions"
                ),
                "scientific_tail_clipping": False,
            },
            "log_zero_floor": LOG_ZERO_FLOOR,
            "log_zero_floor_applies_only_to_logarithm": True,
            "posterior_weighting": "original normalized nested-sampling log weights",
            "resampling": False,
        },
        "aggregate": {
            "source_combined_tail_severity": source_total,
            "baseline_combined_tail_severity": baseline_total,
            "tail_severity_delta_baseline_minus_source": total_delta,
            "direction": direction,
            "less_severe_parameter_comparisons": improved_comparisons,
            "parameter_comparisons": len(completed_ids) * len(RANK_PARAMETERS),
            "completed_cases_only": True,
        },
        "attribution": {
            "assessment": (
                f"paper-baseline-{direction}"
                if complete
                else f"partial-aborted-paper-baseline-{direction}"
            ),
            "strength": (
                "directional-only" if complete else "partial-directional-only"
            ),
            "caveat": ATTRIBUTION_CAVEAT,
            "partial_run_caveat": (
                None
                if complete
                else (
                    f"The campaign was explicitly evaluated after abort with "
                    f"{len(completed_ids)} of {EXPECTED_CASES} cases complete; "
                    "missing cases are reported and excluded from all aggregates."
                )
            ),
            "confounded_changes": {
                "implementation": {
                    "source": "FSM implementation",
                    "baseline": "pinned paper baseline",
                },
                "paper_configuration": {
                    "source": source_config.get("paper_configuration"),
                    "baseline": baseline_config.get("paper_configuration"),
                },
                "n_devices_D": {
                    "source": source_config.get("n_devices"),
                    "baseline": baseline_config.get("n_devices"),
                },
                "num_gibbs_sweeps_M": {
                    "source": source_config.get("num_gibbs_sweeps"),
                    "baseline": baseline_config.get("num_gibbs_sweeps"),
                },
            },
        },
        "provenance": {
            "diagnostic_manifest": "manifest.json",
            "diagnostic_manifest_sha256": file_sha256(campaign_dir / "manifest.json"),
            "source_campaign": provenance["source_campaign"],
            "source_manifest_sha256": provenance["source_manifest_sha256"],
            "source_config_sha256": provenance["source_config_sha256"],
            "source_catalogue_sha256": provenance["source_catalogue_sha256"],
            "ordered_mapping": mapping,
        },
        "cases": cases,
        "missing_cases": missing_cases,
    }


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report["cases"]:
        for parameter in RANK_PARAMETERS:
            comparison = case["comparisons"][parameter]
            source = comparison["source"]
            baseline = comparison["baseline"]
            rows.append(
                {
                    "diagnostic_id": case["diagnostic_id"],
                    "source_injection_id": case["source_injection_id"],
                    "parameter": parameter,
                    "source_rank": source["rank"],
                    "baseline_rank": baseline["rank"],
                    "source_two_sided_tail_probability": source[
                        "two_sided_tail_probability"
                    ],
                    "baseline_two_sided_tail_probability": baseline[
                        "two_sided_tail_probability"
                    ],
                    "source_tail_severity_log10": source["tail_severity_log10"],
                    "baseline_tail_severity_log10": baseline["tail_severity_log10"],
                    "tail_severity_delta_baseline_minus_source": comparison[
                        "tail_severity_delta_baseline_minus_source"
                    ],
                }
            )
    return rows


def write_baseline_diagnostic_report(
    campaign_dir: Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Atomically write deterministic JSON and CSV products under diagnostic/."""

    output_dir = campaign_dir.expanduser().resolve() / REPORT_DIRECTORY
    json_path = output_dir / REPORT_JSON
    csv_path = output_dir / REPORT_CSV
    atomic_write_json(json_path, report)
    atomic_write_csv(csv_path, _csv_rows(report), CSV_FIELDS)
    return json_path, csv_path


def run_evaluation(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.expanduser().resolve()
    report = evaluate_baseline_diagnostic(
        campaign_dir,
        source_campaign=args.source_campaign,
        allow_partial=bool(args.allow_partial),
    )
    json_path, csv_path = write_baseline_diagnostic_report(campaign_dir, report)
    aggregate = report["aggregate"]
    selection = report["selection"]
    print(
        f"Paper baseline {report['status']} direction: "
        f"{aggregate['direction']} "
        "(tail-severity delta baseline-source="
        f"{aggregate['tail_severity_delta_baseline_minus_source']:+.6g})"
    )
    print(
        f"Completed {selection['completed_cases']}/{selection['requested_cases']}; "
        f"missing diagnostic IDs={selection['missing_diagnostic_ids']}"
    )
    print(ATTRIBUTION_CAVEAT)
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_evaluation(_parse_args(argv)))


if __name__ == "__main__":
    main()
