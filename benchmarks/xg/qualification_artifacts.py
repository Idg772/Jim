"""Fail-closed publication of independently generated XG evidence artifacts.

This module is deliberately an artifact writer, not a qualification generator.
Callers must supply measured scalar metrics, explicit pass/fail outcomes, and
detailed diagnostic payloads.  A typed receipt is published last and therefore
acts as the commit marker for the other immutable files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from base64 import b64encode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from jimgw.cli._config import (
    XGCaseDataset,
    XGCaseRecord,
    XGClockValidationReceipt,
    XGCompressionValidationReceipt,
    XGIndependentResponseReceipt,
    XGOrbitalValidationReceipt,
    XGQualificationKind,
    XGRawResults,
    XGReceiptEvidence,
    XGValidationCorpusReceipt,
)


class QualificationArtifactError(ValueError):
    """Raised before a successful qualification receipt can be published."""


@dataclass(frozen=True)
class QualificationCaseOutcome:
    """Measured output and detailed diagnostics for one predeclared case."""

    case_id: str
    passed: bool
    metrics: Mapping[str, float]
    diagnostics: Any


@dataclass(frozen=True)
class PublishedQualificationArtifact:
    """Paths and digests for one atomically committed qualification receipt."""

    receipt: XGReceiptEvidence
    receipt_path: Path
    receipt_sha256: str
    case_dataset_path: Path
    case_dataset_sha256: str
    raw_results_path: Path
    raw_results_sha256: str
    diagnostics_path: Path
    diagnostics_sha256: str
    generator_source_path: Path
    generator_source_sha256: str


@dataclass(frozen=True)
class _ArtifactSpec:
    stem: str
    receipt_filename: str
    receipt_model: type[XGReceiptEvidence]
    case_roles: frozenset[str]


_ARTIFACT_SPECS: dict[XGQualificationKind, _ArtifactSpec] = {
    "xg-validation-corpus": _ArtifactSpec(
        stem="validation-corpus",
        receipt_filename="validation-corpus.json",
        receipt_model=XGValidationCorpusReceipt,
        case_roles=frozenset(("coverage",)),
    ),
    "xg-clock-validation": _ArtifactSpec(
        stem="clock-validation",
        receipt_filename="clock-validation.json",
        receipt_model=XGClockValidationReceipt,
        case_roles=frozenset(("phase-derivative", "near-merger")),
    ),
    "xg-independent-response": _ArtifactSpec(
        stem="independent-response",
        receipt_filename="independent-response.json",
        receipt_model=XGIndependentResponseReceipt,
        case_roles=frozenset(("response",)),
    ),
    "xg-orbital-validation": _ArtifactSpec(
        stem="orbital-validation",
        receipt_filename="orbital-validation.json",
        receipt_model=XGOrbitalValidationReceipt,
        case_roles=frozenset(("orbital",)),
    ),
    "xg-compression-validation": _ArtifactSpec(
        stem="compression-validation",
        receipt_filename="compression-validation.json",
        receipt_model=XGCompressionValidationReceipt,
        case_roles=frozenset(("compression",)),
    ),
}

_CONTROLLED_RECEIPT_FIELDS = frozenset(
    (
        "schema_version",
        "artifact_kind",
        "case_count",
        "case_dataset_file",
        "case_dataset_sha256",
        "raw_results_file",
        "raw_results_sha256",
        "generator_source_file",
        "generator_source_sha256",
    )
)

_AGGREGATE_METRICS: dict[XGQualificationKind, tuple[tuple[str, str, str], ...]] = {
    "xg-validation-corpus": (),
    "xg-clock-validation": (
        ("abs_timing_error_s", "max_abs_timing_error_s", "max"),
        ("component_delta_log_l", "max_component_delta_log_l", "max"),
    ),
    "xg-independent-response": (
        ("numerical_delta_log_l", "max_numerical_delta_log_l", "max"),
        ("component_delta_log_l", "max_component_delta_log_l", "max"),
        ("combined_delta_log_l", "max_combined_delta_log_l", "max"),
    ),
    "xg-orbital-validation": (
        ("profiled_delta_log_l", "max_profiled_delta_log_l", "max"),
        ("projected_bias_sigma", "max_projected_bias_sigma", "max"),
    ),
    "xg-compression-validation": (
        ("component_delta_log_l", "max_component_delta_log_l", "max"),
        ("combined_delta_log_l", "max_combined_delta_log_l", "max"),
    ),
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise QualificationArtifactError(
            "qualification diagnostics and evidence must be finite JSON data"
        ) from error
    return (serialized + "\n").encode("ascii")


def _model_bytes(model: BaseModel) -> bytes:
    return _canonical_json_bytes(
        model.model_dump(mode="json", exclude_none=True),
    )


def _source_filename(stem: str, source: Path | Sequence[Path]) -> str:
    if not isinstance(source, Path):
        return f"{stem}-generator-bundle.json"
    suffix = source.suffix
    if not suffix or any(
        character not in ".-_abcdefghijklmnopqrstuvwxyz0123456789"
        for character in suffix.lower()
    ):
        suffix = ".txt"
    return f"{stem}-generator{suffix}"


def _generator_source_bytes(source: Path | Sequence[Path]) -> bytes:
    """Read one source or build a canonical authenticated multi-source bundle."""

    sources = (source,) if isinstance(source, Path) else tuple(source)
    if not sources:
        raise QualificationArtifactError("generator source bundle must not be empty")
    records = []
    seen_names = set()
    for path in sources:
        if not isinstance(path, Path):
            raise QualificationArtifactError("generator sources must be paths")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise QualificationArtifactError(
                f"cannot read generator source {path}"
            ) from error
        if not content:
            raise QualificationArtifactError("generator source must not be empty")
        name = path.name
        if name in seen_names:
            raise QualificationArtifactError(
                f"generator source names must be unique: {name}"
            )
        seen_names.add(name)
        records.append(
            {
                "content_base64": b64encode(content).decode("ascii"),
                "name": name,
                "sha256": _sha256(content),
            }
        )
    if len(records) == 1:
        return sources[0].read_bytes()
    return _canonical_json_bytes(
        {
            "artifact_kind": "xg-generator-source-bundle",
            "schema_version": 1,
            "sources": sorted(records, key=lambda record: record["name"]),
        }
    )


def _validated_cases(
    qualification_kind: XGQualificationKind,
    cases: Sequence[XGCaseRecord | Mapping[str, Any]],
) -> list[XGCaseRecord]:
    try:
        validated = [XGCaseRecord.model_validate(case) for case in cases]
    except ValueError as error:
        raise QualificationArtifactError("invalid XG qualification case") from error
    if not validated:
        raise QualificationArtifactError("qualification requires at least one case")
    case_ids = [case.case_id for case in validated]
    if len(case_ids) != len(set(case_ids)):
        raise QualificationArtifactError("qualification case IDs must be unique")
    expected_roles = _ARTIFACT_SPECS[qualification_kind].case_roles
    actual_roles = frozenset(case.parameters.case_role for case in validated)
    if actual_roles != expected_roles:
        raise QualificationArtifactError(
            f"case roles {sorted(actual_roles)} do not match {qualification_kind}"
        )
    return sorted(validated, key=lambda case: case.case_id)


def _validated_outcomes(
    cases: Sequence[XGCaseRecord],
    outcomes: Sequence[QualificationCaseOutcome],
) -> list[QualificationCaseOutcome]:
    if not outcomes:
        raise QualificationArtifactError("qualification requires measured outcomes")
    outcome_ids = [outcome.case_id for outcome in outcomes]
    if len(outcome_ids) != len(set(outcome_ids)):
        raise QualificationArtifactError("qualification outcome IDs must be unique")
    case_ids = {case.case_id for case in cases}
    if set(outcome_ids) != case_ids:
        raise QualificationArtifactError(
            "qualification outcome IDs must exactly match the case dataset"
        )
    failed = sorted(outcome.case_id for outcome in outcomes if not outcome.passed)
    if failed:
        raise QualificationArtifactError(f"qualification cases did not pass: {failed}")
    return sorted(outcomes, key=lambda outcome: outcome.case_id)


def _diagnostics_bytes(
    qualification_kind: XGQualificationKind,
    cases: Sequence[XGCaseRecord],
    outcomes: Sequence[QualificationCaseOutcome],
    generator_source_sha256: str,
) -> bytes:
    outcome_by_id = {outcome.case_id: outcome for outcome in outcomes}
    return _canonical_json_bytes(
        {
            "artifact_kind": "xg-qualification-diagnostics",
            "generator_source_sha256": generator_source_sha256,
            "qualification_kind": qualification_kind,
            "records": [
                {
                    "case": case.model_dump(mode="json"),
                    "diagnostics": outcome_by_id[case.case_id].diagnostics,
                    "metrics": dict(outcome_by_id[case.case_id].metrics),
                    "passed": outcome_by_id[case.case_id].passed,
                }
                for case in cases
            ],
            "schema_version": 1,
        }
    )


def _bind_diagnostics(
    cases: Sequence[XGCaseRecord],
    diagnostics_filename: str,
    diagnostics_sha256: str,
) -> list[XGCaseRecord]:
    bound_cases = []
    for case in cases:
        payload = case.model_dump(mode="python")
        parameters = payload["parameters"]
        assert isinstance(parameters, dict)
        for key, value in (
            ("diagnostics_file", diagnostics_filename),
            ("diagnostics_sha256", diagnostics_sha256),
        ):
            existing = parameters.get(key)
            if existing is not None and existing != value:
                raise QualificationArtifactError(
                    f"case {case.case_id!r} supplies a conflicting {key}"
                )
            parameters[key] = value
        bound_cases.append(XGCaseRecord.model_validate(payload))
    return bound_cases


def _build_raw_results(
    qualification_kind: XGQualificationKind,
    outcomes: Sequence[QualificationCaseOutcome],
) -> XGRawResults:
    try:
        return XGRawResults.model_validate(
            {
                "schema_version": 1,
                "artifact_kind": "xg-raw-results",
                "qualification_kind": qualification_kind,
                "results": [
                    {
                        "case_id": outcome.case_id,
                        "passed": outcome.passed,
                        "metrics": dict(outcome.metrics),
                    }
                    for outcome in outcomes
                ],
            }
        )
    except ValueError as error:
        raise QualificationArtifactError(
            "invalid scalar XG qualification results"
        ) from error


def _aggregate_metric(raw_results: XGRawResults, metric: str, reduction: str) -> float:
    try:
        values = [result.metrics[metric] for result in raw_results.results]
    except KeyError as error:
        raise QualificationArtifactError(
            f"every raw result must report {metric!r}"
        ) from error
    return max(values) if reduction == "max" else min(values)


def _validate_receipt_aggregates(
    qualification_kind: XGQualificationKind,
    receipt: XGReceiptEvidence,
    cases: Sequence[XGCaseRecord],
    raw_results: XGRawResults,
) -> None:
    for metric, receipt_field, reduction in _AGGREGATE_METRICS[qualification_kind]:
        measured = _aggregate_metric(raw_results, metric, reduction)
        declared = cast(float, getattr(receipt, receipt_field))
        if not math.isclose(
            measured,
            declared,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise QualificationArtifactError(
                f"receipt field {receipt_field!r} does not match raw {reduction}"
            )

    if isinstance(receipt, XGValidationCorpusReceipt):
        detectors = {name for case in cases for name in case.parameters.detectors}
        f_min = min(case.parameters.f_min for case in cases)
        f_max = max(case.parameters.f_max for case in cases)
        max_snr = max(case.parameters.network_snr for case in cases)
        epoch_count = len({case.parameters.sidereal_epoch_index for case in cases})
        if detectors != set(receipt.detectors):
            raise QualificationArtifactError(
                "validation-corpus detectors do not match raw cases"
            )
        if not math.isclose(f_min, receipt.f_min) or not math.isclose(
            f_max, receipt.f_max
        ):
            raise QualificationArtifactError(
                "validation-corpus frequency band does not match raw cases"
            )
        if not math.isclose(max_snr, receipt.max_network_snr):
            raise QualificationArtifactError(
                "validation-corpus maximum SNR does not match raw cases"
            )
        if epoch_count != receipt.sidereal_epoch_count:
            raise QualificationArtifactError(
                "validation-corpus epoch count does not match raw cases"
            )
        if not any(case.parameters.detector_null for case in cases):
            raise QualificationArtifactError("validation corpus omits detector nulls")
        if not any(case.parameters.prior_extreme for case in cases):
            raise QualificationArtifactError("validation corpus omits prior extremes")

    if isinstance(receipt, XGClockValidationReceipt):
        phase_count = sum(
            case.parameters.case_role == "phase-derivative" for case in cases
        )
        merger_count = sum(case.parameters.case_role == "near-merger" for case in cases)
        if phase_count != receipt.phase_derivative_case_count:
            raise QualificationArtifactError(
                "phase-derivative case count does not match the receipt"
            )
        if merger_count != receipt.near_merger_case_count:
            raise QualificationArtifactError(
                "near-merger case count does not match the receipt"
            )

    if isinstance(receipt, XGCompressionValidationReceipt):
        optional_metrics = (
            (
                "frozen_response_delta_log_l",
                receipt.max_frozen_response_delta_log_l,
                "max",
            ),
            ("timing_sigma_s", receipt.timing_sigma_s, "min"),
        )
        for metric, declared, reduction in optional_metrics:
            reported = [metric in result.metrics for result in raw_results.results]
            if any(reported) and not all(reported):
                raise QualificationArtifactError(
                    f"optional metric {metric!r} must be reported for every case"
                )
            if declared is None and any(reported):
                raise QualificationArtifactError(
                    f"raw results report undeclared optional metric {metric!r}"
                )
            if declared is not None:
                measured = _aggregate_metric(raw_results, metric, reduction)
                if not math.isclose(
                    measured,
                    declared,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ):
                    raise QualificationArtifactError(
                        f"receipt field for {metric!r} does not match raw {reduction}"
                    )


def _write_staged(path: Path, content: bytes) -> None:
    with path.open("xb") as output_file:
        output_file.write(content)
        output_file.flush()
        os.fsync(output_file.fileno())


def _publish_files(
    bundle_dir: Path,
    files: Sequence[tuple[str, bytes]],
    *,
    receipt_filename: str,
) -> None:
    """Publish complete files without replacement, with the receipt last."""

    bundle_dir.mkdir(parents=True, exist_ok=True)
    targets = {filename: bundle_dir / filename for filename, _ in files}
    existing = sorted(str(path) for path in targets.values() if path.exists())
    if existing:
        raise QualificationArtifactError(
            f"qualification artifacts already exist: {existing}"
        )

    staging_dir = Path(tempfile.mkdtemp(prefix=".xg-artifact-", dir=bundle_dir))
    published: list[Path] = []
    try:
        for filename, content in files:
            _write_staged(staging_dir / filename, content)
        ordered_filenames = [
            filename for filename, _ in files if filename != receipt_filename
        ] + [receipt_filename]
        for filename in ordered_filenames:
            target = targets[filename]
            os.link(staging_dir / filename, target, follow_symlinks=False)
            published.append(target)
        directory_fd = os.open(bundle_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def publish_qualification_artifact(
    bundle_dir: Path,
    *,
    qualification_kind: XGQualificationKind,
    cases: Sequence[XGCaseRecord | Mapping[str, Any]],
    outcomes: Sequence[QualificationCaseOutcome],
    summary_fields: Mapping[str, Any],
    generator_source: Path | Sequence[Path],
) -> PublishedQualificationArtifact:
    """Validate and publish one real XG qualification artifact set.

    ``summary_fields`` must contain the receipt-specific measured values and
    explicit pass flags.  Envelope fields, evidence paths, counts, and hashes
    are controlled by this writer.  No files are published unless all cases
    pass and the typed receipt agrees with the raw scalar results.
    """

    spec = _ARTIFACT_SPECS[qualification_kind]
    controlled = sorted(_CONTROLLED_RECEIPT_FIELDS & set(summary_fields))
    if controlled:
        raise QualificationArtifactError(
            f"summary_fields may not override writer fields: {controlled}"
        )

    validated_cases = _validated_cases(qualification_kind, cases)
    validated_outcomes = _validated_outcomes(validated_cases, outcomes)
    source_bytes = _generator_source_bytes(generator_source)
    source_sha256 = _sha256(source_bytes)

    generator_filename = _source_filename(spec.stem, generator_source)
    diagnostics_filename = f"{spec.stem}-diagnostics.json"
    diagnostics_bytes = _diagnostics_bytes(
        qualification_kind,
        validated_cases,
        validated_outcomes,
        source_sha256,
    )
    diagnostics_sha256 = _sha256(diagnostics_bytes)
    bound_cases = _bind_diagnostics(
        validated_cases,
        diagnostics_filename,
        diagnostics_sha256,
    )
    case_dataset = XGCaseDataset(
        schema_version=1,
        artifact_kind="xg-case-dataset",
        qualification_kind=qualification_kind,
        cases=bound_cases,
    )
    raw_results = _build_raw_results(qualification_kind, validated_outcomes)

    case_filename = f"{spec.stem}-cases.json"
    raw_filename = f"{spec.stem}-results.json"
    case_bytes = _model_bytes(case_dataset)
    raw_bytes = _model_bytes(raw_results)
    case_sha256 = _sha256(case_bytes)
    raw_sha256 = _sha256(raw_bytes)

    receipt_payload = {
        **dict(summary_fields),
        "schema_version": 1,
        "artifact_kind": qualification_kind,
        "case_count": len(bound_cases),
        "case_dataset_file": case_filename,
        "case_dataset_sha256": case_sha256,
        "raw_results_file": raw_filename,
        "raw_results_sha256": raw_sha256,
        "generator_source_file": generator_filename,
        "generator_source_sha256": source_sha256,
    }
    if qualification_kind in (
        "xg-clock-validation",
        "xg-independent-response",
    ):
        supplied_implementation = receipt_payload.get("implementation_sha256")
        if supplied_implementation not in (None, source_sha256):
            raise QualificationArtifactError(
                "independent implementation SHA-256 must match the generator source"
            )
        receipt_payload["implementation_sha256"] = source_sha256
    try:
        receipt = spec.receipt_model.model_validate(receipt_payload)
    except ValueError as error:
        raise QualificationArtifactError(
            f"invalid typed receipt for {qualification_kind}"
        ) from error
    _validate_receipt_aggregates(
        qualification_kind,
        receipt,
        bound_cases,
        raw_results,
    )
    receipt_bytes = _model_bytes(receipt)
    try:
        spec.receipt_model.model_validate_json(receipt_bytes)
    except ValueError as error:  # pragma: no cover - defensive serialization guard
        raise QualificationArtifactError(
            "serialized receipt failed validation"
        ) from error

    files = (
        (diagnostics_filename, diagnostics_bytes),
        (case_filename, case_bytes),
        (raw_filename, raw_bytes),
        (generator_filename, source_bytes),
        (spec.receipt_filename, receipt_bytes),
    )
    _publish_files(
        bundle_dir,
        files,
        receipt_filename=spec.receipt_filename,
    )

    return PublishedQualificationArtifact(
        receipt=receipt,
        receipt_path=bundle_dir / spec.receipt_filename,
        receipt_sha256=_sha256(receipt_bytes),
        case_dataset_path=bundle_dir / case_filename,
        case_dataset_sha256=case_sha256,
        raw_results_path=bundle_dir / raw_filename,
        raw_results_sha256=raw_sha256,
        diagnostics_path=bundle_dir / diagnostics_filename,
        diagnostics_sha256=diagnostics_sha256,
        generator_source_path=bundle_dir / generator_filename,
        generator_source_sha256=source_sha256,
    )


__all__ = [
    "PublishedQualificationArtifact",
    "QualificationArtifactError",
    "QualificationCaseOutcome",
    "publish_qualification_artifact",
]
