"""Compose a complete P-P campaign from predeclared event-wise blockings."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.injection_campaign import common, plot_pp
from benchmarks.injection_campaign import merge_staged_results as merge_module

COMPOSITION_SCHEMA_VERSION = 1
SCHEDULE_FIELDS = ("injection_id", "component")
_COMPONENT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_PRESENTATION_CONFIG_FIELDS = frozenset(("campaign", "paper_configuration", "blocks"))
_IMPLEMENTATION_PIN_FIELDS = (
    "implementation_label",
    "implementation_revision",
    "implementation_tree_sha256",
    "sampler_scheduler",
)
_IMPLEMENTATION_IDENTITY_FIELDS = (
    "implementation_label",
    "implementation_revision",
    "sampler_scheduler",
)


@dataclass(frozen=True)
class ScheduleEntry:
    """One immutable event-to-component assignment."""

    injection_id: int
    component: str


@dataclass(frozen=True)
class ComponentSource:
    """One frozen campaign contributing scheduled results."""

    label: str
    path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    catalogue: list[dict[str, Any]]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        metavar="LABEL=CAMPAIGN_DIR",
        help="Frozen component campaign; repeat for every schedule label.",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        required=True,
        help="CSV with exactly injection_id,component and one row for every ID 0..99.",
    )
    return parser.parse_args(argv)


def parse_components(values: Sequence[str]) -> dict[str, Path]:
    """Parse unique ``LABEL=PATH`` component declarations."""

    components: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not _COMPONENT_NAME_RE.fullmatch(label) or not raw_path:
            raise ValueError(f"invalid component declaration: {value!r}")
        if label in components:
            raise ValueError(f"duplicate component label: {label}")
        components[label] = Path(raw_path).expanduser().resolve()
    if len(components) < 2:
        raise ValueError("blocking composition requires at least two components")
    return components


def read_schedule(
    path: Path,
    *,
    component_names: set[str],
) -> tuple[ScheduleEntry, ...]:
    """Read a complete, non-overlapping leading-100 event schedule."""

    try:
        stream = path.expanduser().resolve().open(newline="", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read composition schedule {path}: {error}") from error
    assignments: dict[int, ScheduleEntry] = {}
    with stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(SCHEDULE_FIELDS):
            raise ValueError(
                "composition schedule header must be exactly "
                + ",".join(SCHEDULE_FIELDS)
            )
        for line_number, row in enumerate(reader, start=2):
            raw_id = row.get("injection_id", "")
            component = row.get("component", "")
            try:
                injection_id = int(raw_id)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid injection ID on schedule line {line_number}"
                ) from error
            if str(injection_id) != raw_id or injection_id < 0:
                raise ValueError(f"invalid injection ID on schedule line {line_number}")
            if injection_id in assignments:
                raise ValueError(f"overlapping injection ID {injection_id}")
            if component not in component_names:
                raise ValueError(
                    f"schedule line {line_number} uses unknown component {component!r}"
                )
            assignments[injection_id] = ScheduleEntry(injection_id, component)

    expected = set(range(common.PAPER_PP_RECOVERIES))
    missing = sorted(expected - assignments.keys())
    extra = sorted(assignments.keys() - expected)
    if missing:
        raise ValueError(
            "schedule is missing injection IDs: " + ", ".join(map(str, missing))
        )
    if extra:
        raise ValueError(
            "schedule contains out-of-range injection IDs: "
            + ", ".join(map(str, extra))
        )
    return tuple(assignments[injection_id] for injection_id in sorted(assignments))


def _implementation_pin(manifest: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    pins = [
        value
        for value in (
            manifest.get("implementation_diagnostic"),
            manifest.get("baseline_diagnostic"),
        )
        if isinstance(value, Mapping)
    ]
    if len(pins) != 1:
        raise ValueError(f"component {label!r} has no unambiguous implementation pin")
    pin = pins[0]
    result = {field: pin.get(field) for field in _IMPLEMENTATION_PIN_FIELDS}
    if any(value is None for value in result.values()):
        raise ValueError(f"component {label!r} has an incomplete implementation pin")
    return result


def load_component(label: str, path: Path) -> ComponentSource:
    """Load one frozen component and validate its immutable campaign inputs."""

    if not path.is_dir():
        raise ValueError(f"component {label!r} campaign does not exist: {path}")
    manifest = common.load_manifest(path)
    common.require_publication_eligible(
        manifest,
        product=f"blocking composition component {label!r}",
    )
    eligible, reasons = plot_pp._remediation_eligibility(manifest, is_complete=True)
    if not eligible:
        raise ValueError(
            f"component {label!r} is not remediation-eligible: " + "; ".join(reasons)
        )
    catalogue = common.read_catalogue(path / manifest["catalogue"]["path"])
    if len(catalogue) != int(manifest.get("catalogue_size", -1)):
        raise ValueError(f"component {label!r} catalogue size is inconsistent")
    return ComponentSource(
        label=label,
        path=path,
        manifest=manifest,
        manifest_sha256=common.file_sha256(path / "manifest.json"),
        catalogue=catalogue,
    )


def _scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: copy.deepcopy(value)
        for field, value in config.items()
        if field not in _PRESENTATION_CONFIG_FIELDS
    }


def validate_component_compatibility(
    components: Mapping[str, ComponentSource],
) -> str:
    """Require identical experiments apart from declared block partitions."""

    if len(components) < 2:
        raise ValueError("blocking composition requires at least two components")
    reference_label = min(components)
    reference = components[reference_label]
    reference_manifest = reference.manifest
    reference_fields = {
        field: reference_manifest.get(field)
        for field in (
            "schema_version",
            "n_injections",
            "catalogue_size",
            "master_seed",
            "selection",
        )
    }
    reference_config = _scientific_config(reference_manifest["config"])
    reference_pin = _implementation_pin(
        reference_manifest,
        label=reference_label,
    )
    for label in sorted(components):
        component = components[label]
        manifest = component.manifest
        actual_fields = {field: manifest.get(field) for field in reference_fields}
        if actual_fields != reference_fields:
            raise ValueError(f"component {label!r} campaign selection mismatch")
        if component.catalogue != reference.catalogue:
            raise ValueError(f"component {label!r} catalogue truth/seed mismatch")
        if manifest.get("catalogue") != reference_manifest.get("catalogue"):
            raise ValueError(f"component {label!r} catalogue provenance mismatch")
        if manifest.get("psd") != reference_manifest.get("psd"):
            raise ValueError(f"component {label!r} PSD provenance mismatch")
        if _scientific_config(manifest["config"]) != reference_config:
            raise ValueError(f"component {label!r} scientific configuration mismatch")
        actual_pin = _implementation_pin(manifest, label=label)
        for field in _IMPLEMENTATION_IDENTITY_FIELDS:
            if actual_pin[field] != reference_pin[field]:
                raise ValueError(f"component {label!r} implementation {field} mismatch")
    return reference_label


def _validate_scheduled_results(
    components: Mapping[str, ComponentSource],
    schedule: Sequence[ScheduleEntry],
) -> dict[int, merge_module.ValidatedResult]:
    """Validate all file presence before opening any scheduled scientific payload."""

    missing: list[str] = []
    for entry in schedule:
        directory = common.result_dir(
            components[entry.component].path, entry.injection_id
        )
        if (
            not (directory / "summary.json").is_file()
            or not (directory / "posterior.npz").is_file()
        ):
            missing.append(f"{entry.injection_id}:{entry.component}")
    if missing:
        raise ValueError("scheduled results are incomplete: " + ", ".join(missing))

    validated: dict[int, merge_module.ValidatedResult] = {}
    for entry in schedule:
        component = components[entry.component]
        validated[entry.injection_id] = merge_module._validate_result(
            common.result_dir(component.path, entry.injection_id),
            entry.injection_id,
            component.manifest,
            component.catalogue,
        )
    return validated


def _rank_rows(
    components: Mapping[str, ComponentSource],
    schedule: Sequence[ScheduleEntry],
    validated: Mapping[int, merge_module.ValidatedResult],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for entry in schedule:
        component = components[entry.component]
        directory = common.result_dir(component.path, entry.injection_id)
        expected_result = validated[entry.injection_id]
        fingerprint, file_hashes = merge_module._result_fingerprint(directory)
        if (
            fingerprint != expected_result.fingerprint
            or file_hashes != expected_result.file_hashes
        ):
            raise RuntimeError(
                f"scheduled result {entry.injection_id:03d} changed after validation"
            )
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        truth = {
            name: float(component.catalogue[entry.injection_id][name])
            for name in (*common.PARAMETERS, *common.MARGINALIZED_PARAMETERS)
        }
        ranks, corrected_legacy = plot_pp._recomputed_ranks(
            summary,
            truth,
            directory / summary["posterior"]["path"],
            phase_marginalization=bool(
                component.manifest["config"]["phase_marginalization"]
            ),
        )
        rows.append(
            {
                "injection_id": entry.injection_id,
                **ranks,
                "_legacy_phase_gauge_corrected_parameters": (
                    ["s1_phi", "s2_phi"] if corrected_legacy else []
                ),
            }
        )
        fingerprint, file_hashes = merge_module._result_fingerprint(directory)
        if (
            fingerprint != expected_result.fingerprint
            or file_hashes != expected_result.file_hashes
        ):
            raise RuntimeError(
                f"scheduled result {entry.injection_id:03d} changed while reading ranks"
            )
        hashes[
            f"{entry.component}/results/injection-{entry.injection_id:03d}/summary.json"
        ] = expected_result.file_hashes["summary.json"]
    return rows, hashes


def _revalidate_sources(
    components: Mapping[str, ComponentSource],
    schedule: Sequence[ScheduleEntry],
    validated: Mapping[int, merge_module.ValidatedResult],
) -> None:
    """Recheck immutable inputs and results immediately before publication."""

    for label, component in components.items():
        manifest = common.load_manifest(component.path)
        if (
            manifest != component.manifest
            or common.file_sha256(component.path / "manifest.json")
            != component.manifest_sha256
        ):
            raise RuntimeError(f"component {label!r} changed during composition")
    for entry in schedule:
        component = components[entry.component]
        current = merge_module._validate_result(
            common.result_dir(component.path, entry.injection_id),
            entry.injection_id,
            component.manifest,
            component.catalogue,
        )
        expected = validated[entry.injection_id]
        if (
            current.fingerprint != expected.fingerprint
            or current.file_hashes != expected.file_hashes
        ):
            raise RuntimeError(
                f"scheduled result {entry.injection_id:03d} changed during composition"
            )


def _component_record(component: ComponentSource) -> dict[str, Any]:
    return {
        "campaign_dir": str(component.path),
        "manifest_sha256": component.manifest_sha256,
        "config_sha256": component.manifest["config_sha256"],
        "catalogue_sha256": component.manifest["catalogue"]["sha256"],
        "blocks": component.manifest["config"]["blocks"],
        "implementation": _implementation_pin(
            component.manifest, label=component.label
        ),
    }


def _compatibility_record(
    components: Mapping[str, ComponentSource],
    *,
    reference_label: str,
) -> dict[str, Any]:
    tree_hashes = {
        label: _implementation_pin(component.manifest, label=label)[
            "implementation_tree_sha256"
        ]
        for label, component in sorted(components.items())
    }
    homogeneous_tree = len(set(tree_hashes.values())) == 1
    if homogeneous_tree:
        attribution = "All component results share one homogeneous implementation tree."
    else:
        attribution = (
            "This composition combines independently validated component execution "
            "trees and cannot be attributed to one homogeneous implementation."
        )
    return {
        "reference_component": reference_label,
        "allowed_config_differences": sorted(_PRESENTATION_CONFIG_FIELDS),
        "allowed_execution_provenance_differences": ["implementation_tree_sha256"],
        "implementation_tree_sha256_by_component": tree_hashes,
        "implementation_tree_homogeneous": homogeneous_tree,
        "implementation_attribution": attribution,
        "required_identical": (
            "catalogue rows/truths/noise seeds/sampler seeds, PSDs, selection, all "
            "scientific configuration, implementation label, Git revision, and "
            "sampler scheduler"
        ),
    }


def _normalised_schedule_rows(
    schedule: Sequence[ScheduleEntry],
) -> list[dict[str, Any]]:
    return [
        {"injection_id": entry.injection_id, "component": entry.component}
        for entry in schedule
    ]


def compose_blocking_results(
    output_dir: Path,
    component_paths: Mapping[str, Path],
    schedule_path: Path,
) -> dict[str, Any]:
    """Validate and atomically publish one complete event-wise composition."""

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ValueError(f"composition output already exists: {output_dir}")
    components = {
        label: load_component(label, path)
        for label, path in sorted(component_paths.items())
    }
    schedule = read_schedule(schedule_path, component_names=set(components))
    reference_label = validate_component_compatibility(components)
    validated = _validate_scheduled_results(components, schedule)
    rows, input_hashes = _rank_rows(components, schedule, validated)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        normalised_schedule = _normalised_schedule_rows(schedule)
        common.atomic_write_csv(
            staging / "schedule.csv",
            normalised_schedule,
            SCHEDULE_FIELDS,
        )
        composition: dict[str, Any] = {
            "schema_version": COMPOSITION_SCHEMA_VERSION,
            "kind": "event-wise-blocking-composition",
            "n_injections": common.PAPER_PP_RECOVERIES,
            "selection": {
                "rule": "explicit deterministic ID-to-component schedule",
                "start_inclusive": 0,
                "stop_exclusive": common.PAPER_PP_RECOVERIES,
            },
            "schedule": {
                "path": "schedule.csv",
                "sha256": common.file_sha256(staging / "schedule.csv"),
                "assignments": normalised_schedule,
            },
            "components": {
                label: _component_record(component)
                for label, component in sorted(components.items())
            },
            "result_sources": [
                {
                    "injection_id": entry.injection_id,
                    "component": entry.component,
                    "fingerprint": validated[entry.injection_id].fingerprint,
                    "summary_sha256": input_hashes[
                        f"{entry.component}/results/"
                        f"injection-{entry.injection_id:03d}/summary.json"
                    ],
                    "posterior_sha256": validated[entry.injection_id].posterior_sha256,
                }
                for entry in schedule
            ],
            "compatibility": _compatibility_record(
                components,
                reference_label=reference_label,
            ),
        }
        composition["composition_sha256"] = common.canonical_sha256(composition)
        common.atomic_write_json(staging / "composition.json", composition)

        reference = components[reference_label]
        report_manifest = {
            "n_injections": common.PAPER_PP_RECOVERIES,
            "config": copy.deepcopy(reference.manifest["config"]),
            "config_sha256": composition["composition_sha256"],
            "reproduction_scope": {"pp_calibration_eligible": True},
        }
        report_manifest["config"].update(
            {
                "campaign": "event-wise-blocking-composition",
                "paper_configuration": "Sharded M=1 (event-wise blocking composition)",
            }
        )
        report = plot_pp.aggregate_rank_rows(
            staging,
            manifest=report_manifest,
            rows=rows,
            selected_ids=list(range(common.PAPER_PP_RECOVERIES)),
            input_summary_sha256=input_hashes,
            report_extensions={
                "blocking_composition": {
                    "manifest": "composition.json",
                    "manifest_sha256": common.file_sha256(staging / "composition.json"),
                    "composition_sha256": composition["composition_sha256"],
                    "schedule": "schedule.csv",
                    "schedule_sha256": composition["schedule"]["sha256"],
                }
            },
            print_report=False,
        )
        _revalidate_sources(components, schedule, validated)
        if output_dir.exists():
            raise RuntimeError(
                f"composition output appeared during validation: {output_dir}"
            )
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    components = parse_components(args.component)
    report = compose_blocking_results(args.output_dir, components, args.schedule)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "composition_sha256": report["blocking_composition"][
                    "composition_sha256"
                ],
                "remediation_assessment": report["remediation_assessment"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
