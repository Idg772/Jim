"""Evaluate the four-case Appendix-A time/distance nuisance swap."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.injection_campaign.common import (
    MARGINALIZED_PARAMETERS,
    PARAMETERS,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    load_manifest,
    posterior_rank,
    read_catalogue,
    result_dir,
)
from benchmarks.injection_campaign.prepare_time_marginalization_diagnostic import (
    CAMPAIGN_NAME,
    COMMON_PARAMETERS,
    DEFAULT_SOURCE_IDS,
    DIAGNOSTIC_MARGINALIZED_PARAMETERS,
    DIAGNOSTIC_PARAMETERS,
    PAPER_CONFIGURATION,
    RECOVERY_LIKELIHOOD_F_MAX_HZ,
    TIME_MARGINALIZATION,
    TIMING_SELECTED_EVENTS,
)

EXPECTED_CASES = 4
REPORT_SCHEMA_VERSION = 1
REPORT_DIRECTORY = "diagnostic"
REPORT_JSON = "appendix-a-time-marginalization-rank-comparison.json"
REPORT_CSV = "appendix-a-time-marginalization-rank-comparison.csv"
LOG_ZERO_FLOOR = sys.float_info.min
RANK_ENDPOINT_ROUNDOFF_TOLERANCE = 1.0e-12
RANK_RECOMPUTE_TOLERANCE = 1.0e-12
LOG_WEIGHT_NORMALIZATION_TOLERANCE = 1.0e-10
TRUTH_PARAMETERS = (*PARAMETERS, *MARGINALIZED_PARAMETERS)
PHASE_GAUGE_PARAMETERS = ("s1_phi", "s2_phi")
CSV_FIELDS = (
    "diagnostic_id",
    "source_injection_id",
    "comparison_kind",
    "parameter",
    "source_rank",
    "diagnostic_rank",
    "source_tail_severity_log10",
    "diagnostic_tail_severity_log10",
    "tail_severity_delta_diagnostic_minus_source",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--source-campaign",
        type=Path,
        default=None,
        help=(
            "Frozen D=4/M=1 source campaign. Defaults to the sibling recorded "
            "in the diagnostic manifest."
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
    return min(1.0, max(0.0, rank))


def tail_metrics(rank: float) -> dict[str, float | bool]:
    """Return the unclipped two-sided tail probability and log severity."""

    value = _rank(rank, field="rank")
    probability = 2.0 * min(value, 1.0 - value)
    severity = -math.log10(max(probability, LOG_ZERO_FLOOR))
    if severity == 0.0:
        severity = 0.0
    return {
        "rank": value,
        "two_sided_tail_probability": probability,
        "tail_severity_log10": severity,
        "log_input_floored": probability == 0.0,
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


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    campaign_dir: Path,
    source_campaign: Path,
) -> None:
    if (
        _exact_int(manifest.get("n_injections"), field="manifest.n_injections")
        != EXPECTED_CASES
        or _exact_int(
            manifest.get("catalogue_size"), field="manifest.catalogue_size"
        )
        != EXPECTED_CASES
    ):
        raise ValueError(f"diagnostic must contain exactly {EXPECTED_CASES} cases")

    selection = _mapping(manifest.get("selection"), field="manifest.selection")
    if (
        selection.get("source_injection_ids") != list(DEFAULT_SOURCE_IDS)
        or selection.get("start_inclusive") != 0
        or selection.get("stop_exclusive") != EXPECTED_CASES
    ):
        raise ValueError("diagnostic sentinel selection is invalid or unordered")

    source_config = _mapping(source_manifest.get("config"), field="source config")
    config = _mapping(manifest.get("config"), field="diagnostic config")
    if (
        source_config.get("n_devices") != 4
        or source_config.get("num_gibbs_sweeps") != 1
        or source_config.get("sampler_scheduler", "fsm") != "fsm"
        or source_config.get("time_marginalization") is not False
        or not isinstance(source_config.get("distance_marginalization"), Mapping)
        or source_config.get("blocks", [])[-1:] != [["t_c"]]
    ):
        raise ValueError("source is not the expected Sharded D=4/M=1 campaign")
    if (
        config.get("campaign") != CAMPAIGN_NAME
        or config.get("paper_configuration") != PAPER_CONFIGURATION
        or config.get("n_devices") != 4
        or config.get("num_gibbs_sweeps") != 1
        or config.get("sampler_scheduler", "fsm") != "fsm"
        or config.get("phase_marginalization") is not True
        or config.get("time_marginalization") != TIME_MARGINALIZATION
        or config.get("distance_marginalization") is not False
        or config.get("likelihood_f_max_hz") != RECOVERY_LIKELIHOOD_F_MAX_HZ
        or config.get("f_max_hz") != source_config.get("f_max_hz")
        or config.get("blocks", [])[-1:] != [["d_L"]]
    ):
        raise ValueError("diagnostic Appendix-A configuration is invalid")
    expected_config = copy.deepcopy(dict(source_config))
    expected_config.update(
        {
            "campaign": CAMPAIGN_NAME,
            "paper_configuration": PAPER_CONFIGURATION,
            "n_devices": 4,
            "num_gibbs_sweeps": 1,
            "phase_marginalization": True,
            "time_marginalization": copy.deepcopy(TIME_MARGINALIZATION),
            "distance_marginalization": False,
            "likelihood_f_max_hz": RECOVERY_LIKELIHOOD_F_MAX_HZ,
        }
    )
    expected_config["blocks"] = copy.deepcopy(source_config["blocks"])
    expected_config["blocks"][-1] = ["d_L"]
    expected_config["timing"]["selected_events"] = TIMING_SELECTED_EVENTS
    if dict(config) != expected_config:
        raise ValueError(
            "diagnostic config changes more than the recorded nuisance treatment, "
            "labels, and recovery Nyquist exclusion"
        )

    scope = _mapping(
        manifest.get("reproduction_scope"), field="manifest.reproduction_scope"
    )
    if (
        scope.get("iid_prior_predictive_catalogue") is not False
        or scope.get("pp_calibration_eligible") is not False
    ):
        raise ValueError("targeted diagnostic must be non-IID and nonpublication")
    appendix = _mapping(
        manifest.get("appendix_a_diagnostic"),
        field="manifest.appendix_a_diagnostic",
    )
    if (
        appendix.get("sampled_parameters") != list(DIAGNOSTIC_PARAMETERS)
        or appendix.get("marginalized_parameters")
        != list(DIAGNOSTIC_MARGINALIZED_PARAMETERS)
        or appendix.get("common_rank_parameters") != list(COMMON_PARAMETERS)
        or appendix.get("population_calibration_claim_permitted") is not False
    ):
        raise ValueError("Appendix-A parameter inventory is invalid")

    expected_hashes = {
        "source_manifest_sha256": file_sha256(source_campaign / "manifest.json"),
        "source_config_sha256": source_manifest.get("config_sha256"),
        "source_catalogue_sha256": source_manifest.get("catalogue", {}).get(
            "sha256"
        ),
    }
    for field, expected in expected_hashes.items():
        if provenance.get(field) != expected:
            raise ValueError(f"diagnostic provenance {field} mismatch")
    if file_sha256(campaign_dir / "manifest.json") == provenance.get(
        "source_manifest_sha256"
    ):
        raise ValueError("diagnostic and source manifests unexpectedly match")


def _validate_mapping(
    manifest: Mapping[str, Any],
    *,
    campaign_dir: Path,
    source_campaign: Path,
    diagnostic_catalogue: Sequence[Mapping[str, Any]],
    source_catalogue: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    catalogue_metadata = _mapping(manifest["catalogue"], field="manifest.catalogue")
    provenance = _mapping(
        catalogue_metadata.get("provenance"), field="manifest.catalogue.provenance"
    )
    raw_mapping = provenance.get("mapping")
    if not isinstance(raw_mapping, list) or len(raw_mapping) != EXPECTED_CASES:
        raise ValueError("diagnostic source mapping is incomplete")

    normalized: list[dict[str, Any]] = []
    for diagnostic_id, (source_id, raw_entry) in enumerate(
        zip(DEFAULT_SOURCE_IDS, raw_mapping, strict=True)
    ):
        entry = _mapping(raw_entry, field=f"source mapping {diagnostic_id}")
        if (
            _exact_int(
                entry.get("diagnostic_id"),
                field=f"source mapping {diagnostic_id}.diagnostic_id",
            )
            != diagnostic_id
            or _exact_int(
                entry.get("source_injection_id"),
                field=f"source mapping {diagnostic_id}.source_injection_id",
            )
            != source_id
        ):
            raise ValueError("source mapping is not in sentinel order")
        source_row = source_catalogue[source_id]
        diagnostic_row = diagnostic_catalogue[diagnostic_id]
        for name in diagnostic_row:
            expected = diagnostic_id if name == "injection_id" else source_row[name]
            if diagnostic_row[name] != expected:
                raise ValueError(
                    f"diagnostic row {diagnostic_id} differs from source in {name}"
                )
        if (
            entry.get("noise_seed") != source_row["noise_seed"]
            or entry.get("sampler_seed") != source_row["sampler_seed"]
        ):
            raise ValueError(f"source mapping seed mismatch for {diagnostic_id}")

        source_directory = result_dir(source_campaign, source_id)
        summary_path = source_directory / "summary.json"
        posterior_path = source_directory / "posterior.npz"
        if (
            entry.get("source_summary_sha256") != file_sha256(summary_path)
            or entry.get("source_posterior_sha256") != file_sha256(posterior_path)
        ):
            raise ValueError(f"source result hash mismatch for {source_id}")
        recorded_ranks = _mapping(
            entry.get("source_ranks"), field=f"recorded source ranks {source_id}"
        )
        if set(recorded_ranks) != set(PARAMETERS):
            raise ValueError(f"recorded source ranks are incomplete for {source_id}")
        normalized.append(
            {
                "diagnostic_id": diagnostic_id,
                "source_injection_id": source_id,
                "noise_seed": source_row["noise_seed"],
                "sampler_seed": source_row["sampler_seed"],
                "source_summary_sha256": entry["source_summary_sha256"],
                "source_posterior_sha256": entry["source_posterior_sha256"],
            }
        )
    return normalized


def _load_weighted_ranks(
    campaign: Path,
    manifest: Mapping[str, Any],
    truth: Mapping[str, Any],
    injection_id: int,
    parameters: tuple[str, ...],
    *,
    diagnostic: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    directory = result_dir(campaign, injection_id)
    summary_path = directory / "summary.json"
    posterior_path = directory / "posterior.npz"
    if not summary_path.is_file() or not posterior_path.is_file():
        raise ValueError(f"result is incomplete: {directory}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"result summary is invalid: {summary_path}") from error
    label = f"{'diagnostic' if diagnostic else 'source'} result {injection_id}"
    if (
        summary.get("config_sha256") != manifest.get("config_sha256")
        or summary.get("injection_id") != injection_id
        or summary.get("seeds")
        != {"noise": truth["noise_seed"], "sampler": truth["sampler_seed"]}
    ):
        raise ValueError(f"{label}: summary provenance mismatch")
    summary_truth = summary.get("truth")
    if not isinstance(summary_truth, Mapping) or set(summary_truth) != set(
        TRUTH_PARAMETERS
    ):
        raise ValueError(f"{label}: truth inventory is invalid")
    for name in TRUTH_PARAMETERS:
        if float(summary_truth[name]) != float(truth[name]):
            raise ValueError(f"{label}: truth mismatch for {name}")

    if diagnostic:
        pin = _mapping(
            manifest.get("implementation_diagnostic"),
            field="manifest.implementation_diagnostic",
        )
        implementation = _mapping(
            summary.get("implementation"), field=f"{label}.implementation"
        )
        if (
            implementation.get("label") != pin.get("implementation_label")
            or implementation.get("revision")
            != pin.get("implementation_revision")
            or implementation.get("tree_sha256")
            != pin.get("implementation_tree_sha256")
        ):
            raise ValueError(f"{label}: implementation provenance mismatch")
        if summary.get("parameter_treatment") != {
            "sampled": list(DIAGNOSTIC_PARAMETERS),
            "marginalized": list(DIAGNOSTIC_MARGINALIZED_PARAMETERS),
        }:
            raise ValueError(f"{label}: parameter treatment inventory is invalid")
        devices = _mapping(summary.get("devices"), field=f"{label}.devices")
        if (
            devices.get("backend") != "gpu"
            or devices.get("requested_count") != 4
            or devices.get("local_count") != 4
            or summary.get("simulated_cpu") is not False
        ):
            raise ValueError(f"{label}: invalid D=4 GPU inventory")

    posterior = _mapping(summary.get("posterior"), field=f"{label}.posterior")
    if (
        posterior.get("sha256") != file_sha256(posterior_path)
        or posterior.get("path") != "posterior.npz"
    ):
        raise ValueError(f"{label}: posterior metadata mismatch")
    expected_fields = (*parameters, "log_likelihood", "log_weights")
    fields = posterior.get("fields")
    if (
        not isinstance(fields, list)
        or len(fields) != len(set(fields))
        or set(fields) != set(expected_fields)
    ):
        raise ValueError(f"{label}: posterior field inventory is invalid")
    try:
        with np.load(posterior_path, allow_pickle=False) as archive:
            if archive.files != fields:
                raise ValueError(f"{label}: NPZ fields differ from summary")
            arrays = {name: np.asarray(archive[name]) for name in fields}
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        if str(error).startswith(f"{label}:"):
            raise
        raise ValueError(f"{label}: unreadable posterior: {error}") from error

    sample_count = _exact_int(
        summary.get("posterior_samples"), field=f"{label}.posterior_samples"
    )
    if sample_count < 1 or any(
        values.shape != (sample_count,) for values in arrays.values()
    ):
        raise ValueError(f"{label}: posterior array shapes are invalid")
    for name in (*parameters, "log_likelihood"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{label}: {name} contains non-finite values")
    log_weights = arrays["log_weights"]
    if (
        np.any(np.isnan(log_weights))
        or np.any(np.isposinf(log_weights))
        or not np.any(np.isfinite(log_weights))
    ):
        raise ValueError(f"{label}: log_weights are invalid")
    finite = log_weights[np.isfinite(log_weights)]
    maximum = float(np.max(finite))
    log_normalizer = maximum + math.log(float(np.sum(np.exp(finite - maximum))))
    if not math.isclose(
        log_normalizer,
        0.0,
        rel_tol=0.0,
        abs_tol=LOG_WEIGHT_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(f"{label}: log_weights are not normalized")

    stored_ranks = summary.get("ranks")
    if not isinstance(stored_ranks, Mapping) or set(stored_ranks) != set(parameters):
        raise ValueError(f"{label}: rank inventory is invalid")
    ranks: dict[str, float] = {}
    phase_gauge_corrections: dict[str, dict[str, float]] = {}
    for name in parameters:
        stored = _rank(stored_ranks[name], field=f"{label}.ranks.{name}")
        catalogue_truth = float(truth[name])
        recomputed_catalogue_rank = posterior_rank(
            arrays[name], catalogue_truth, arrays["log_weights"]
        )
        if not math.isclose(
            stored,
            recomputed_catalogue_rank,
            rel_tol=0.0,
            abs_tol=RANK_RECOMPUTE_TOLERANCE,
        ):
            raise ValueError(f"{label}: weighted rank mismatch for {name}")
        if name in PHASE_GAUGE_PARAMETERS:
            gauge_truth = float(
                np.mod(catalogue_truth + float(truth["phase_c"]), math.tau)
            )
            corrected_rank = posterior_rank(
                arrays[name], gauge_truth, arrays["log_weights"]
            )
            phase_gauge_corrections[name] = {
                "catalogue_truth_alpha": catalogue_truth,
                "phase_c_truth": float(truth["phase_c"]),
                "sampled_truth_beta": gauge_truth,
                "stored_catalogue_truth_rank": stored,
                "gauge_corrected_rank": corrected_rank,
                "gauge_minus_catalogue_rank": (
                    corrected_rank - recomputed_catalogue_rank
                ),
            }
            ranks[name] = corrected_rank
        else:
            ranks[name] = recomputed_catalogue_rank
    return ranks, {
        "summary": summary_path.relative_to(campaign).as_posix(),
        "summary_sha256": file_sha256(summary_path),
        "posterior": posterior_path.relative_to(campaign).as_posix(),
        "posterior_sha256": file_sha256(posterior_path),
        "posterior_samples": sample_count,
        "phase_gauge_corrections": phase_gauge_corrections,
    }


def _direction(delta: float) -> str:
    if delta < 0.0:
        return "less-tail-pathological"
    if delta > 0.0:
        return "more-tail-pathological"
    return "unchanged"


def evaluate_time_marginalization_diagnostic(
    campaign_dir: Path,
    *,
    source_campaign: Path | None = None,
) -> dict[str, Any]:
    """Validate and compare all four paired weighted-rank results."""

    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    catalogue_metadata = _mapping(manifest.get("catalogue"), field="manifest.catalogue")
    provenance = _mapping(
        catalogue_metadata.get("provenance"), field="manifest.catalogue.provenance"
    )
    source_campaign = _source_campaign_path(
        campaign_dir, provenance, source_campaign
    )
    source_manifest = load_manifest(source_campaign)
    _validate_manifest_contract(
        manifest,
        source_manifest,
        provenance,
        campaign_dir=campaign_dir,
        source_campaign=source_campaign,
    )
    diagnostic_catalogue = read_catalogue(
        campaign_dir / str(catalogue_metadata["path"])
    )
    source_catalogue_metadata = _mapping(
        source_manifest.get("catalogue"), field="source manifest.catalogue"
    )
    source_catalogue = read_catalogue(
        source_campaign / str(source_catalogue_metadata["path"])
    )
    mapping = _validate_mapping(
        manifest,
        campaign_dir=campaign_dir,
        source_campaign=source_campaign,
        diagnostic_catalogue=diagnostic_catalogue,
        source_catalogue=source_catalogue,
    )

    cases: list[dict[str, Any]] = []
    source_total = 0.0
    diagnostic_total = 0.0
    less_severe = 0
    for entry in mapping:
        diagnostic_id = entry["diagnostic_id"]
        source_id = entry["source_injection_id"]
        source_ranks, source_inputs = _load_weighted_ranks(
            source_campaign,
            source_manifest,
            source_catalogue[source_id],
            source_id,
            PARAMETERS,
            diagnostic=False,
        )
        diagnostic_ranks, diagnostic_inputs = _load_weighted_ranks(
            campaign_dir,
            manifest,
            diagnostic_catalogue[diagnostic_id],
            diagnostic_id,
            DIAGNOSTIC_PARAMETERS,
            diagnostic=True,
        )
        comparisons: dict[str, Any] = {}
        case_source = 0.0
        case_diagnostic = 0.0
        for parameter in COMMON_PARAMETERS:
            source_metrics = tail_metrics(source_ranks[parameter])
            diagnostic_metrics = tail_metrics(diagnostic_ranks[parameter])
            delta = float(diagnostic_metrics["tail_severity_log10"]) - float(
                source_metrics["tail_severity_log10"]
            )
            case_source += float(source_metrics["tail_severity_log10"])
            case_diagnostic += float(diagnostic_metrics["tail_severity_log10"])
            if delta < 0.0:
                less_severe += 1
            comparisons[parameter] = {
                "source": source_metrics,
                "diagnostic": diagnostic_metrics,
                "tail_severity_delta_diagnostic_minus_source": delta,
                "direction": _direction(delta),
                "truth_coordinate": (
                    "beta_i = (catalogue alpha_i + phase_c) mod 2 pi"
                    if parameter in PHASE_GAUGE_PARAMETERS
                    else "catalogue coordinate"
                ),
            }
        case_delta = case_diagnostic - case_source
        source_total += case_source
        diagnostic_total += case_diagnostic
        cases.append(
            {
                **entry,
                "common_parameter_comparisons": comparisons,
                "combined_common_tail_severity": {
                    "source": case_source,
                    "diagnostic": case_diagnostic,
                    "delta_diagnostic_minus_source": case_delta,
                    "direction": _direction(case_delta),
                },
                "sampled_nuisance_context": {
                    "source_t_c": tail_metrics(source_ranks["t_c"]),
                    "diagnostic_d_L": tail_metrics(diagnostic_ranks["d_L"]),
                    "cross_parameter_comparison_permitted": False,
                },
                "source_inputs": source_inputs,
                "diagnostic_inputs": diagnostic_inputs,
            }
        )

    total_delta = diagnostic_total - source_total
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "campaign": manifest["config"]["campaign"],
        "config_sha256": manifest["config_sha256"],
        "status": "complete",
        "selection": {
            "case_count": EXPECTED_CASES,
            "diagnostic_ids": list(range(EXPECTED_CASES)),
            "source_injection_ids": list(DEFAULT_SOURCE_IDS),
            "common_rank_parameters": list(COMMON_PARAMETERS),
            "diagnostic_sampled_distance_parameter": "d_L",
            "source_sampled_time_parameter": "t_c",
        },
        "methodology": {
            "rank": "sum normalized nested-sampling weight for samples < truth",
            "posterior_weighting": "original normalized nested-sampling log weights",
            "resampling": False,
            "two_sided_tail_probability": "2 * min(rank, 1 - rank)",
            "tail_severity_log10": "-log10(two_sided_tail_probability)",
            "paired_common_parameter_comparisons": len(COMMON_PARAMETERS),
            "population_calibration_tests_performed": [],
            "population_calibration_claim_permitted": False,
            "interpretation": (
                "Four hand-selected pathologies provide directional causal "
                "evidence only. The source t_c and diagnostic d_L ranks are "
                "reported separately and are not compared as like parameters."
            ),
        },
        "phase_gauge": {
            "corrected_parameters": list(PHASE_GAUGE_PARAMETERS),
            "catalogue_coordinates": "alpha_i = catalogue spin azimuth",
            "sampled_coordinates": "beta_i = (alpha_i + phase_c) mod 2 pi",
            "applied_to": ["source", "diagnostic"],
            "stored_raw_alpha_ranks_used_for_comparison": False,
            "stored_raw_alpha_ranks_recomputed_for_artifact_validation": True,
            "coordinate_transform_absolute_jacobian": 1.0,
        },
        "aggregate_common_tail_severity": {
            "source": source_total,
            "diagnostic": diagnostic_total,
            "delta_diagnostic_minus_source": total_delta,
            "direction": _direction(total_delta),
            "less_severe_comparisons": less_severe,
            "parameter_comparisons": EXPECTED_CASES * len(COMMON_PARAMETERS),
        },
        "sampled_distance_ranks": [
            {
                "diagnostic_id": case["diagnostic_id"],
                "source_injection_id": case["source_injection_id"],
                **case["sampled_nuisance_context"]["diagnostic_d_L"],
            }
            for case in cases
        ],
        "provenance": {
            "diagnostic_manifest": "manifest.json",
            "diagnostic_manifest_sha256": file_sha256(
                campaign_dir / "manifest.json"
            ),
            "source_campaign": provenance["source_campaign"],
            "source_manifest_sha256": provenance["source_manifest_sha256"],
            "source_config_sha256": provenance["source_config_sha256"],
            "source_catalogue_sha256": provenance["source_catalogue_sha256"],
            "ordered_mapping": mapping,
        },
        "cases": cases,
    }


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report["cases"]:
        base = {
            "diagnostic_id": case["diagnostic_id"],
            "source_injection_id": case["source_injection_id"],
        }
        for parameter in COMMON_PARAMETERS:
            comparison = case["common_parameter_comparisons"][parameter]
            rows.append(
                {
                    **base,
                    "comparison_kind": "paired-common-parameter",
                    "parameter": parameter,
                    "source_rank": comparison["source"]["rank"],
                    "diagnostic_rank": comparison["diagnostic"]["rank"],
                    "source_tail_severity_log10": comparison["source"][
                        "tail_severity_log10"
                    ],
                    "diagnostic_tail_severity_log10": comparison["diagnostic"][
                        "tail_severity_log10"
                    ],
                    "tail_severity_delta_diagnostic_minus_source": comparison[
                        "tail_severity_delta_diagnostic_minus_source"
                    ],
                }
            )
        context = case["sampled_nuisance_context"]
        rows.extend(
            [
                {
                    **base,
                    "comparison_kind": "source-sampled-nuisance-context",
                    "parameter": "t_c",
                    "source_rank": context["source_t_c"]["rank"],
                    "diagnostic_rank": "",
                    "source_tail_severity_log10": context["source_t_c"][
                        "tail_severity_log10"
                    ],
                    "diagnostic_tail_severity_log10": "",
                    "tail_severity_delta_diagnostic_minus_source": "",
                },
                {
                    **base,
                    "comparison_kind": "diagnostic-sampled-nuisance-context",
                    "parameter": "d_L",
                    "source_rank": "",
                    "diagnostic_rank": context["diagnostic_d_L"]["rank"],
                    "source_tail_severity_log10": "",
                    "diagnostic_tail_severity_log10": context["diagnostic_d_L"][
                        "tail_severity_log10"
                    ],
                    "tail_severity_delta_diagnostic_minus_source": "",
                },
            ]
        )
    return rows


def write_time_marginalization_diagnostic_report(
    campaign_dir: Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    directory = campaign_dir.expanduser().resolve() / REPORT_DIRECTORY
    json_path = directory / REPORT_JSON
    csv_path = directory / REPORT_CSV
    atomic_write_json(json_path, report)
    atomic_write_csv(csv_path, _csv_rows(report), CSV_FIELDS)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = evaluate_time_marginalization_diagnostic(
        args.campaign_dir,
        source_campaign=args.source_campaign,
    )
    json_path, csv_path = write_time_marginalization_diagnostic_report(
        args.campaign_dir, report
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
