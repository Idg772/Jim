"""Freeze exact pathological rows for a D=4, M=1 carrier-anchor diagnostic."""

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
SOURCE_TIME_ANCHOR = "nrtidal-merger"
DIAGNOSTIC_TIME_ANCHOR = "imrphenomd"
TIME_ANCHORS = (SOURCE_TIME_ANCHOR, DIAGNOSTIC_TIME_ANCHOR)
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
    parser.add_argument(
        "--carrier-time-anchor",
        choices=TIME_ANCHORS,
        default=DIAGNOSTIC_TIME_ANCHOR,
        help="Carrier time convention used consistently for injection and recovery.",
    )
    return parser.parse_args(argv)


def _validate_implementation_pin(revision: str, tree_sha256: str) -> None:
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("implementation_revision must be a lowercase Git SHA")
    if _SHA256_RE.fullmatch(tree_sha256) is None:
        raise ValueError("implementation_tree_sha256 must be a lowercase SHA-256")


def prepare_time_anchor_diagnostic(
    source_campaign: Path,
    output_campaign: Path,
    *,
    implementation_revision: str,
    implementation_tree_sha256: str,
    source_ids: Sequence[int] = DEFAULT_SOURCE_IDS,
    carrier_time_anchor: str = DIAGNOSTIC_TIME_ANCHOR,
) -> dict[str, Any]:
    """Copy immutable inputs and select rows while changing only the time anchor."""

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
    if carrier_time_anchor not in TIME_ANCHORS:
        raise ValueError(
            f"unknown carrier_time_anchor {carrier_time_anchor!r}; "
            f"expected one of {list(TIME_ANCHORS)}"
        )

    source_manifest = load_manifest(source_campaign)
    source_config = source_manifest["config"]
    if (
        source_config.get("n_devices") != 4
        or source_config.get("num_gibbs_sweeps") != 1
        or source_config.get("sampler_scheduler", "fsm") != "fsm"
        or source_config.get("carrier_time_anchor", SOURCE_TIME_ANCHOR)
        != SOURCE_TIME_ANCHOR
    ):
        raise ValueError(
            "source campaign is not the expected D=4, M=1 NRTidal-anchor campaign"
        )
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

            source_result = (
                source_campaign / "results" / f"injection-{source_id:03d}"
            )
            source_summary = source_result / "summary.json"
            source_posterior = source_result / "posterior.npz"
            if not source_summary.is_file() or not source_posterior.is_file():
                raise ValueError(f"source result is incomplete: {source_result}")
            summary = json.loads(source_summary.read_text(encoding="utf-8"))
            ranks = summary.get("ranks")
            if not isinstance(ranks, dict):
                raise TypeError(f"source result has no ranks: {source_summary}")
            if summary.get("posterior", {}).get("sha256") != file_sha256(
                source_posterior
            ):
                raise ValueError(f"source posterior hash mismatch: {source_posterior}")
            mapping.append(
                {
                    "diagnostic_id": diagnostic_id,
                    "source_injection_id": source_id,
                    "source_ranks": {
                        name: float(ranks[name])
                        for name in ("q", "ra", "dec", "t_c")
                    },
                    "source_summary_sha256": file_sha256(source_summary),
                    "source_posterior_sha256": file_sha256(source_posterior),
                    "noise_seed": row["noise_seed"],
                    "sampler_seed": row["sampler_seed"],
                }
            )

        catalogue_path = output_campaign / "catalogue.csv"
        atomic_write_csv(catalogue_path, selected_rows, CATALOGUE_FIELDS)

        is_author_anchor = carrier_time_anchor == DIAGNOSTIC_TIME_ANCHOR
        anchor_label = "coauthor-anchor" if is_author_anchor else "local-anchor-control"
        campaign_name = f"{anchor_label}-d4-m1-pathology-diagnostic"
        paper_configuration = f"Sharded {anchor_label} diagnostic"
        config = copy.deepcopy(source_config)
        config.update(
            {
                "campaign": campaign_name,
                "paper_configuration": paper_configuration,
                "carrier_time_anchor": carrier_time_anchor,
            }
        )
        config["timing"]["selected_events"] = (
            "four fixed q/t_c/sky pathologies at D=4, M=1; targeted diagnostic"
        )

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "master_seed": None,
            "n_injections": len(selected_rows),
            "catalogue_size": len(selected_rows),
            "selection": {
                "rule": "exact completed pathological rows from a frozen M1 campaign",
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
                "methodology": "paired carrier-time-anchor diagnostic",
                "iid_prior_predictive_catalogue": False,
                "pp_calibration_eligible": False,
                "statement": (
                    "Exact selected rows from the completed D=4, M=1 campaign. "
                    "The coauthor anchor is a public-source proxy for unreleased "
                    "paper code, and this targeted diagnostic is not a P-P test."
                ),
            },
            "implementation_diagnostic": {
                "implementation_label": "candidate",
                "implementation_revision": implementation_revision,
                "implementation_tree_sha256": implementation_tree_sha256,
                "sampler_scheduler": "fsm",
                "configuration_semantics": (
                    f"Injection and recovery both use the {carrier_time_anchor} carrier "
                    "anchor. D=4, M=1, blocks, priors, catalogue rows, noise seeds, "
                    "sampler seeds, PSDs, and all other scientific settings are "
                    "copied from the source campaign."
                ),
                "source_campaign_config_sha256": source_manifest["config_sha256"],
                "changed_variables": {
                    "carrier_time_anchor": {
                        "source": SOURCE_TIME_ANCHOR,
                        "diagnostic": carrier_time_anchor,
                    }
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
    manifest = prepare_time_anchor_diagnostic(
        args.source_campaign,
        args.output_campaign,
        implementation_revision=args.implementation_revision,
        implementation_tree_sha256=args.implementation_tree_sha256,
        source_ids=args.source_ids,
        carrier_time_anchor=args.carrier_time_anchor,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
