"""Audit spin-azimuth ranks in a phase-marginalized D=4, M=1 campaign.

The precessing waveform uses a phase gauge in which the sampled azimuths are

    beta_i = (alpha_i + phase_c) mod 2 pi,

where ``alpha_i`` is the catalogue spin azimuth.  This command validates the
saved campaign artifacts, reproduces the originally recorded ranks, and then
computes ranks against ``beta_i``.  It is deliberately read-only with respect
to the campaign; reports are written to a caller-selected output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from benchmarks.injection_campaign.common import (
    MARGINALIZED_PARAMETERS,
    PARAMETERS,
    SCHEMA_VERSION,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    load_manifest,
    posterior_rank,
    read_catalogue,
    result_dir,
)

REPORT_SCHEMA_VERSION = 1
REPORT_JSON = "phase-gauge-audit.json"
REPORT_CSV = "phase-gauge-ranks.csv"
SPIN_AZIMUTHS = ("s1_phi", "s2_phi")
RANK_REPRODUCTION_TOLERANCE = 1.0e-12
RANK_ENDPOINT_ROUNDOFF_TOLERANCE = 1.0e-12
CSV_FIELDS = (
    "injection_id",
    "parameter",
    "catalogue_truth",
    "phase_c_truth",
    "gauge_truth",
    "stored_rank",
    "recomputed_catalogue_rank",
    "stored_rank_abs_error",
    "gauge_corrected_rank",
    "gauge_minus_catalogue_rank",
    "posterior_samples",
    "summary_sha256",
    "posterior_sha256",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for the JSON and CSV audit reports.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Audit completed recoveries and record missing IDs. By default every "
            "manifest recovery must be present."
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


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _rank(value: object, *, field: str) -> float:
    rank = _finite_float(value, field=field)
    if (
        rank < -RANK_ENDPOINT_ROUNDOFF_TOLERANCE
        or rank > 1.0 + RANK_ENDPOINT_ROUNDOFF_TOLERANCE
    ):
        raise ValueError(f"{field} must lie in [0, 1]")
    return min(1.0, max(0.0, rank))


def _campaign_path(campaign_dir: Path, relative: object, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{field} must be relative to the campaign")
    resolved = (campaign_dir / candidate).resolve()
    try:
        resolved.relative_to(campaign_dir)
    except ValueError as error:
        raise ValueError(f"{field} escapes the campaign directory") from error
    return resolved


def _validate_campaign_configuration(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    config = _mapping(manifest.get("config"), field="manifest.config")
    if config.get("phase_marginalization") is not True:
        raise ValueError("phase-gauge audit requires phase marginalization")
    if config.get("paper_configuration") != "Sharded":
        raise ValueError("phase-gauge audit requires the paper Sharded configuration")
    if _exact_int(config.get("n_devices"), field="config.n_devices") != 4:
        raise ValueError("phase-gauge audit requires D=4")
    if _exact_int(config.get("num_gibbs_sweeps"), field="config.num_gibbs_sweeps") != 1:
        raise ValueError("phase-gauge audit requires M=1")
    return config


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    catalogue_row: Mapping[str, Any],
    injection_id: int,
) -> None:
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"injection {injection_id}: unsupported summary schema")
    if summary.get("campaign") != manifest["config"]["campaign"]:
        raise ValueError(f"injection {injection_id}: campaign name mismatch")
    if summary.get("config_sha256") != manifest.get("config_sha256"):
        raise ValueError(f"injection {injection_id}: config hash mismatch")
    if (
        _exact_int(summary.get("injection_id"), field="summary.injection_id")
        != injection_id
    ):
        raise ValueError(f"injection {injection_id}: summary ID mismatch")

    truth = _mapping(summary.get("truth"), field="summary.truth")
    for parameter in (*PARAMETERS, *MARGINALIZED_PARAMETERS):
        actual = _finite_float(truth.get(parameter), field=f"summary.truth.{parameter}")
        expected = float(catalogue_row[parameter])
        if actual != expected:
            raise ValueError(
                f"injection {injection_id}: truth mismatch for {parameter}"
            )

    seeds = _mapping(summary.get("seeds"), field="summary.seeds")
    for summary_name, catalogue_name in (
        ("noise", "noise_seed"),
        ("sampler", "sampler_seed"),
    ):
        actual = _exact_int(
            seeds.get(summary_name), field=f"summary.seeds.{summary_name}"
        )
        if actual != catalogue_row[catalogue_name]:
            raise ValueError(f"injection {injection_id}: {summary_name} seed mismatch")

    method = _mapping(summary.get("rank_method"), field="summary.rank_method")
    if method != {
        "comparison": "sample < truth",
        "resampled": False,
        "weighting": "original nested-sampling weights",
    }:
        raise ValueError(
            f"injection {injection_id}: rank method is not the paper method"
        )


def _validate_angle(value: float, *, field: str) -> None:
    if value < 0.0 or value >= math.tau:
        raise ValueError(f"{field} must lie in [0, 2 pi)")


def _uniformity_metrics(ranks: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(ranks, dtype=float)
    result = stats.kstest(values, "uniform", method="exact")
    return {
        "n": int(values.size),
        "mean_rank": float(np.mean(values)),
        "minimum_rank": float(np.min(values)),
        "maximum_rank": float(np.max(values)),
        "ks_uniform": {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "alternative": "two-sided",
            "method": "exact one-sample Kolmogorov-Smirnov",
        },
    }


def audit_phase_gauge(
    campaign_dir: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate saved artifacts and return a gauge-corrected rank report.

    This function performs no writes. Missing recoveries are errors unless
    ``allow_partial`` is set; malformed or inconsistent artifacts are always
    errors.
    """

    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    config = _validate_campaign_configuration(manifest)

    n_injections = _exact_int(
        manifest.get("n_injections"), field="manifest.n_injections"
    )
    catalogue_size = _exact_int(
        manifest.get("catalogue_size"), field="manifest.catalogue_size"
    )
    if n_injections < 1:
        raise ValueError("manifest.n_injections must be positive")

    catalogue_metadata = _mapping(manifest.get("catalogue"), field="manifest.catalogue")
    catalogue_path = _campaign_path(
        campaign_dir,
        catalogue_metadata.get("path"),
        field="manifest.catalogue.path",
    )
    catalogue = read_catalogue(catalogue_path)
    if catalogue_size != len(catalogue) or n_injections > catalogue_size:
        raise ValueError("manifest catalogue sizes are inconsistent")

    completed_ids: list[int] = []
    missing_ids: list[int] = []
    cases: list[dict[str, Any]] = []
    rank_errors = {parameter: [] for parameter in SPIN_AZIMUTHS}
    original_ranks = {parameter: [] for parameter in SPIN_AZIMUTHS}
    corrected_ranks = {parameter: [] for parameter in SPIN_AZIMUTHS}
    shifts = {parameter: [] for parameter in SPIN_AZIMUTHS}

    for injection_id in range(n_injections):
        directory = result_dir(campaign_dir, injection_id)
        summary_path = directory / "summary.json"
        default_posterior_path = directory / "posterior.npz"
        if not summary_path.is_file() or not default_posterior_path.is_file():
            missing_ids.append(injection_id)
            continue

        try:
            summary_value = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"injection {injection_id}: invalid summary {summary_path}: {error}"
            ) from error
        summary = _mapping(summary_value, field="summary")
        row = catalogue[injection_id]
        _validate_summary(
            summary,
            manifest=manifest,
            catalogue_row=row,
            injection_id=injection_id,
        )

        posterior_metadata = _mapping(
            summary.get("posterior"), field="summary.posterior"
        )
        posterior_path = _campaign_path(
            directory,
            posterior_metadata.get("path"),
            field="summary.posterior.path",
        )
        if posterior_path != default_posterior_path.resolve():
            raise ValueError(
                f"injection {injection_id}: posterior path must be posterior.npz"
            )
        expected_posterior_hash = posterior_metadata.get("sha256")
        if not isinstance(expected_posterior_hash, str):
            raise TypeError(f"injection {injection_id}: posterior hash must be a string")
        posterior_hash = file_sha256(posterior_path)
        if posterior_hash != expected_posterior_hash:
            raise ValueError(f"injection {injection_id}: posterior hash mismatch")
        posterior_bytes = _exact_int(
            posterior_metadata.get("bytes"), field="summary.posterior.bytes"
        )
        if posterior_bytes != posterior_path.stat().st_size:
            raise ValueError(f"injection {injection_id}: posterior byte count mismatch")
        if posterior_metadata.get("space") != "prior":
            raise ValueError(
                f"injection {injection_id}: posterior is not in prior space"
            )
        if (
            posterior_metadata.get("weighting")
            != "normalized nested-sampling log weights"
        ):
            raise ValueError(f"injection {injection_id}: posterior weighting mismatch")

        with np.load(posterior_path, allow_pickle=False) as posterior:
            required_fields = {*SPIN_AZIMUTHS, "log_weights"}
            if not required_fields.issubset(posterior.files):
                raise ValueError(
                    f"injection {injection_id}: posterior lacks phase-audit fields"
                )
            metadata_fields = posterior_metadata.get("fields")
            if (
                not isinstance(metadata_fields, list)
                or any(not isinstance(field, str) for field in metadata_fields)
                or set(metadata_fields) != set(posterior.files)
            ):
                raise ValueError(
                    f"injection {injection_id}: posterior field metadata mismatch"
                )
            samples = {
                parameter: np.asarray(posterior[parameter])
                for parameter in SPIN_AZIMUTHS
            }
            log_weights = np.asarray(posterior["log_weights"])

        sample_count = _exact_int(
            summary.get("posterior_samples"), field="summary.posterior_samples"
        )
        if sample_count < 1:
            raise ValueError(f"injection {injection_id}: posterior is empty")
        if log_weights.shape != (sample_count,):
            raise ValueError(
                f"injection {injection_id}: log_weights shape is inconsistent"
            )

        phase_truth = float(row["phase_c"])
        _validate_angle(phase_truth, field=f"catalogue[{injection_id}].phase_c")
        stored_ranks = _mapping(summary.get("ranks"), field="summary.ranks")
        case_parameters: dict[str, Any] = {}
        summary_hash = file_sha256(summary_path)

        for parameter in SPIN_AZIMUTHS:
            values = samples[parameter]
            if values.shape != (sample_count,):
                raise ValueError(
                    f"injection {injection_id}: {parameter} shape is inconsistent"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"injection {injection_id}: {parameter} contains non-finite samples"
                )
            if np.any(values < 0.0) or np.any(values >= math.tau):
                raise ValueError(
                    f"injection {injection_id}: {parameter} samples leave [0, 2 pi)"
                )

            catalogue_truth = float(row[parameter])
            _validate_angle(
                catalogue_truth, field=f"catalogue[{injection_id}].{parameter}"
            )
            gauge_truth = float(np.mod(catalogue_truth + phase_truth, math.tau))
            stored_rank = _rank(
                stored_ranks.get(parameter), field=f"summary.ranks.{parameter}"
            )
            recomputed_rank = posterior_rank(values, catalogue_truth, log_weights)
            rank_error = abs(stored_rank - recomputed_rank)
            if rank_error > RANK_REPRODUCTION_TOLERANCE:
                raise ValueError(
                    f"injection {injection_id}: stored {parameter} rank does not "
                    f"reproduce (absolute error {rank_error:.3g})"
                )
            corrected_rank = posterior_rank(values, gauge_truth, log_weights)
            shift = corrected_rank - recomputed_rank

            rank_errors[parameter].append(rank_error)
            original_ranks[parameter].append(recomputed_rank)
            corrected_ranks[parameter].append(corrected_rank)
            shifts[parameter].append(shift)
            case_parameters[parameter] = {
                "catalogue_truth": catalogue_truth,
                "gauge_truth": gauge_truth,
                "stored_rank": stored_rank,
                "recomputed_catalogue_rank": recomputed_rank,
                "stored_rank_abs_error": rank_error,
                "gauge_corrected_rank": corrected_rank,
                "gauge_minus_catalogue_rank": shift,
            }

        completed_ids.append(injection_id)
        cases.append(
            {
                "injection_id": injection_id,
                "phase_c_truth": phase_truth,
                "posterior_samples": sample_count,
                "summary_sha256": summary_hash,
                "posterior_sha256": posterior_hash,
                "parameters": case_parameters,
            }
        )

    if missing_ids and not allow_partial:
        rendered = ", ".join(str(value) for value in missing_ids)
        raise ValueError(f"missing campaign recoveries: {rendered}")
    if not completed_ids:
        raise ValueError("no completed recoveries are available to audit")

    aggregates: dict[str, Any] = {}
    max_rank_error: dict[str, float] = {}
    for parameter in SPIN_AZIMUTHS:
        absolute_shifts = np.abs(np.asarray(shifts[parameter], dtype=float))
        max_rank_error[parameter] = float(max(rank_errors[parameter]))
        aggregates[parameter] = {
            "catalogue_truth_ranks": _uniformity_metrics(original_ranks[parameter]),
            "gauge_corrected_ranks": _uniformity_metrics(corrected_ranks[parameter]),
            "rank_shift": {
                "mean_signed": float(np.mean(shifts[parameter])),
                "median_absolute": float(np.median(absolute_shifts)),
                "maximum_absolute": float(np.max(absolute_shifts)),
            },
        }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "phase-gauge-rank-audit",
        "status": "partial" if missing_ids else "complete",
        "campaign": {
            "name": config.get("campaign"),
            "config_sha256": manifest.get("config_sha256"),
            "manifest_schema_version": manifest.get("schema_version"),
            "requested_recoveries": n_injections,
            "catalogue_size": catalogue_size,
            "paper_configuration": config.get("paper_configuration"),
            "device_shards_D": config.get("n_devices"),
            "gibbs_sweeps_M": config.get("num_gibbs_sweeps"),
            "n_live": config.get("n_live"),
            "n_delete": config.get("n_delete"),
            "phase_marginalization": config.get("phase_marginalization"),
            "frequency_resolution_hz": {
                "sampling_frequency": config.get("sampling_frequency_hz"),
                "minimum": config.get("f_min_hz"),
                "maximum": config.get("f_max_hz"),
            },
        },
        "selection": {
            "requested_ids": list(range(n_injections)),
            "completed_ids": completed_ids,
            "missing_ids": missing_ids,
            "allow_partial": allow_partial,
        },
        "phase_gauge": {
            "catalogue_coordinates": "alpha_i = catalogue spin azimuth",
            "sampled_coordinates": "beta_i = (alpha_i + phase_c) mod 2 pi",
            "waveform_identity": (
                "h(alpha_1, alpha_2, phase_c) = exp(+2 i phase_c) h(beta_1, beta_2, 0)"
            ),
            "coordinate_transform_absolute_jacobian": 1.0,
            "marginalization_identity": (
                "uniform phase_c integration reduces the complex overlap to "
                "I0(abs(z)) in beta coordinates"
            ),
        },
        "methodology": {
            "gauge_truth": "(catalogue spin azimuth + catalogue phase_c) mod 2 pi",
            "comparison": "posterior sample < truth on the fixed [0, 2 pi) branch",
            "weighting": "original nested-sampling log weights",
            "resampling": False,
            "source_artifacts_mutated": False,
        },
        "verification": {
            "stored_catalogue_truth_ranks_reproduced": True,
            "absolute_tolerance": RANK_REPRODUCTION_TOLERANCE,
            "maximum_absolute_error_by_parameter": max_rank_error,
            "posterior_hashes_verified": True,
            "summary_truths_and_seeds_match_catalogue": True,
        },
        "aggregate": aggregates,
        "provenance": {
            "manifest_path": "manifest.json",
            "manifest_sha256": file_sha256(campaign_dir / "manifest.json"),
            "catalogue_path": str(catalogue_path.relative_to(campaign_dir)),
            "catalogue_sha256": file_sha256(catalogue_path),
        },
        "caveats": [
            (
                "The saved s1_phi and s2_phi arrays are beta-gauge coordinates; "
                "they are not posterior samples of the original catalogue alpha_i."
            ),
            (
                "Ranks for angular variables use a fixed [0, 2 pi) branch cut. "
                "They are calibrated under the matching uniform prior but are "
                "not invariant to moving that cut."
            ),
            (
                "Recovering physical alpha_i samples requires the conditional "
                "common phase at each retained point; the saved marginalized "
                "scalar likelihood is insufficient for that reconstruction."
            ),
            (
                "This coordinate correction directly concerns only s1_phi and "
                "s2_phi and cannot explain rank pathology in non-spin parameters."
            ),
        ],
        "cases": cases,
    }


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise TypeError("report.cases must be a list")
    for case_value in cases:
        case = _mapping(case_value, field="report.cases[]")
        parameters = _mapping(case.get("parameters"), field="case.parameters")
        for parameter in SPIN_AZIMUTHS:
            values = _mapping(
                parameters.get(parameter), field=f"case.parameters.{parameter}"
            )
            rows.append(
                {
                    "injection_id": case["injection_id"],
                    "parameter": parameter,
                    "catalogue_truth": values["catalogue_truth"],
                    "phase_c_truth": case["phase_c_truth"],
                    "gauge_truth": values["gauge_truth"],
                    "stored_rank": values["stored_rank"],
                    "recomputed_catalogue_rank": values["recomputed_catalogue_rank"],
                    "stored_rank_abs_error": values["stored_rank_abs_error"],
                    "gauge_corrected_rank": values["gauge_corrected_rank"],
                    "gauge_minus_catalogue_rank": values["gauge_minus_catalogue_rank"],
                    "posterior_samples": case["posterior_samples"],
                    "summary_sha256": case["summary_sha256"],
                    "posterior_sha256": case["posterior_sha256"],
                }
            )
    return rows


def write_phase_gauge_audit(
    output_dir: Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write a deterministic JSON report and its flattened CSV companion."""

    output_dir = output_dir.expanduser().resolve()
    json_path = output_dir / REPORT_JSON
    csv_path = output_dir / REPORT_CSV
    atomic_write_json(json_path, report)
    atomic_write_csv(csv_path, _csv_rows(report), CSV_FIELDS)
    return json_path, csv_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_phase_gauge(args.campaign_dir, allow_partial=args.allow_partial)
    json_path, csv_path = write_phase_gauge_audit(args.output_dir, report)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
