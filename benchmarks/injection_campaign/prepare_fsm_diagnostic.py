"""Freeze exact FSM rows for a pinned D=4, M=3 candidate diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.injection_campaign.common import (
    CATALOGUE_FIELDS,
    SCHEMA_VERSION,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_manifest,
    read_catalogue,
    refresh_status,
)

DEFAULT_SOURCE_IDS = (97, 12, 41, 32)
CAMPAIGN_NAME = "fsm-sharded-d4-m3-pathology-diagnostic"
PAPER_CONFIGURATION = "Sharded-M3 diagnostic"
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _source_ids(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from error
    if (
        not parsed
        or any(item < 0 for item in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise argparse.ArgumentTypeError(
            "source IDs must be a non-empty unique non-negative sequence"
        )
    return parsed


def _revision(value: str) -> str:
    if _REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase 40-character Git SHA")
    return value


def _tree_sha256(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_campaign", type=Path)
    parser.add_argument("output_campaign", type=Path)
    parser.add_argument("--implementation-revision", type=_revision, required=True)
    parser.add_argument(
        "--implementation-tree-sha256", type=_tree_sha256, required=True
    )
    parser.add_argument(
        "--source-ids",
        type=_source_ids,
        default=DEFAULT_SOURCE_IDS,
        help="Comma-separated source injection IDs in diagnostic order.",
    )
    return parser.parse_args(argv)


def _validate_implementation_pin(revision: str, tree_sha256: str) -> None:
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("implementation_revision must be a lowercase Git SHA")
    if _SHA256_RE.fullmatch(tree_sha256) is None:
        raise ValueError("implementation_tree_sha256 must be a lowercase SHA-256")


def prepare_fsm_diagnostic(
    source_campaign: Path,
    output_campaign: Path,
    *,
    implementation_revision: str,
    implementation_tree_sha256: str,
    source_ids: Sequence[int] = DEFAULT_SOURCE_IDS,
) -> dict[str, Any]:
    """Copy exact immutable inputs and select/reindex four source rows."""

    _validate_implementation_pin(
        implementation_revision,
        implementation_tree_sha256,
    )
    source_campaign = source_campaign.expanduser().resolve()
    output_campaign = output_campaign.expanduser().resolve()
    if output_campaign.exists():
        raise FileExistsError(f"diagnostic campaign already exists: {output_campaign}")

    selected_ids = tuple(int(value) for value in source_ids)
    if not selected_ids or any(value < 0 for value in selected_ids):
        raise ValueError("source_ids must be a non-empty non-negative sequence")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("source_ids must be unique")

    source_manifest = load_manifest(source_campaign)
    source_config = source_manifest["config"]
    if (
        source_config.get("n_devices") != 4
        or source_config.get("num_gibbs_sweeps") != 1
        or source_config.get("sampler_scheduler", "fsm") != "fsm"
    ):
        raise ValueError("source campaign is not the expected FSM D=4, M=1 campaign")
    source_catalogue = read_catalogue(
        source_campaign / source_manifest["catalogue"]["path"]
    )
    if any(value >= len(source_catalogue) for value in selected_ids):
        raise ValueError("source injection ID is outside the frozen catalogue")

    output_campaign.mkdir(parents=True)
    try:
        shutil.copytree(source_campaign / "inputs", output_campaign / "inputs")
        selected_rows: list[dict[str, Any]] = []
        mapping: list[dict[str, Any]] = []
        for diagnostic_id, source_id in enumerate(selected_ids):
            row = copy.deepcopy(source_catalogue[source_id])
            row["injection_id"] = diagnostic_id
            selected_rows.append(row)

            source_summary = (
                source_campaign
                / "results"
                / f"injection-{source_id:03d}"
                / "summary.json"
            )
            if not source_summary.is_file():
                raise ValueError(f"source result is missing: {source_summary}")
            summary = json.loads(source_summary.read_text(encoding="utf-8"))
            ranks = summary.get("ranks")
            if not isinstance(ranks, dict):
                raise TypeError(f"source result has no ranks: {source_summary}")
            mapping.append(
                {
                    "diagnostic_id": diagnostic_id,
                    "source_injection_id": source_id,
                    "source_ranks": {
                        name: float(ranks[name]) for name in ("q", "ra", "t_c")
                    },
                    "source_summary_sha256": file_sha256(source_summary),
                    "noise_seed": row["noise_seed"],
                    "sampler_seed": row["sampler_seed"],
                }
            )

        catalogue_path = output_campaign / "catalogue.csv"
        atomic_write_csv(catalogue_path, selected_rows, CATALOGUE_FIELDS)

        config = copy.deepcopy(source_config)
        config.update(
            {
                "campaign": CAMPAIGN_NAME,
                "paper_configuration": PAPER_CONFIGURATION,
                "sampler_scheduler": "fsm",
                "n_devices": 4,
                "num_gibbs_sweeps": 3,
            }
        )
        config["timing"]["selected_events"] = (
            "four targeted FSM pathologies at D=4, M=3; not a Figure 3 population"
        )

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "master_seed": None,
            "n_injections": len(selected_rows),
            "catalogue_size": len(selected_rows),
            "selection": {
                "rule": "exact completed pathological rows from a frozen FSM campaign",
                "start_inclusive": 0,
                "stop_exclusive": len(selected_rows),
                "source_injection_ids": list(selected_ids),
            },
            "config": config,
            "catalogue": {
                "path": "catalogue.csv",
                "sha256": file_sha256(catalogue_path),
                "bytes": catalogue_path.stat().st_size,
                "generator": "exact selected and reindexed frozen source rows",
                "provenance": {
                    "source_campaign": source_campaign.name,
                    "source_manifest_sha256": file_sha256(
                        source_campaign / "manifest.json"
                    ),
                    "source_config_sha256": source_manifest["config_sha256"],
                    "source_catalogue_sha256": source_manifest["catalogue"]["sha256"],
                    "mapping": mapping,
                },
            },
            "psd": copy.deepcopy(source_manifest["psd"]),
            "reproduction_scope": {
                "methodology": "paired FSM Gibbs-sweep diagnostic",
                "iid_prior_predictive_catalogue": False,
                "pp_calibration_eligible": False,
                "statement": (
                    "Exact selected rows from the completed FSM P-P campaign. "
                    "This targeted diagnostic cannot be published as an independent "
                    "P-P test."
                ),
            },
            "implementation_diagnostic": {
                "implementation_label": "candidate",
                "implementation_revision": implementation_revision,
                "implementation_tree_sha256": implementation_tree_sha256,
                "sampler_scheduler": "fsm",
                "configuration_semantics": (
                    "Current FSM candidate with D=4 and M=3; relative to the source "
                    "FSM D=4, M=1 campaign, only M and descriptive labels change."
                ),
                "source_campaign_config_sha256": source_manifest["config_sha256"],
                "changed_variables": {
                    "n_devices": {"source": 4, "diagnostic": 4},
                    "num_gibbs_sweeps": {"source": 1, "diagnostic": 3},
                    "sampler_scheduler": {"source": "fsm", "diagnostic": "fsm"},
                },
            },
            "storage_policy": copy.deepcopy(source_manifest["storage_policy"]),
        }
        manifest["config_sha256"] = canonical_sha256(manifest)
        atomic_write_json(output_campaign / "manifest.json", manifest)
        refresh_status(output_campaign, len(selected_rows))
        load_manifest(output_campaign)
        return manifest
    except BaseException:
        shutil.rmtree(output_campaign, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = prepare_fsm_diagnostic(
        args.source_campaign,
        args.output_campaign,
        implementation_revision=args.implementation_revision,
        implementation_tree_sha256=args.implementation_tree_sha256,
        source_ids=args.source_ids,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
