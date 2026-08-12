"""Freeze exact FSM rows for pinned baseline code at paper High-Res settings."""

from __future__ import annotations

import argparse
import copy
import json
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

PAPER_BASELINE_REVISION = "86335bdb1e7ef6191937dd17b2ca53edbb1d899f"
PAPER_BASELINE_TREE_SHA256 = (
    "09085b4d427cfbbb9b379228b2b5d6cb781042687c64f0207740d1af3354c141"
)
PAPER_CONFIGURATION = "High-Res"
DEFAULT_SOURCE_IDS = (97, 12, 40, 41, 32)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_campaign", type=Path)
    parser.add_argument("output_campaign", type=Path)
    parser.add_argument(
        "--source-ids",
        type=_source_ids,
        default=DEFAULT_SOURCE_IDS,
        help="Comma-separated source injection IDs in diagnostic order.",
    )
    parser.add_argument("--n-devices", type=_positive_int, default=1)
    parser.add_argument("--num-gibbs-sweeps", type=_positive_int, default=3)
    return parser.parse_args(argv)


def prepare_baseline_diagnostic(
    source_campaign: Path,
    output_campaign: Path,
    *,
    source_ids: Sequence[int] = DEFAULT_SOURCE_IDS,
    n_devices: int = 1,
    num_gibbs_sweeps: int = 3,
) -> dict[str, Any]:
    """Copy exact immutable inputs and select/reindex source catalogue rows."""

    source_campaign = source_campaign.expanduser().resolve()
    output_campaign = output_campaign.expanduser().resolve()
    if output_campaign.exists():
        raise FileExistsError(f"diagnostic campaign already exists: {output_campaign}")
    if (n_devices, num_gibbs_sweeps) != (1, 3):
        raise ValueError(
            "this paper High-Res diagnostic is fixed to n_devices=1 and "
            "num_gibbs_sweeps=3"
        )
    selected_ids = tuple(int(value) for value in source_ids)
    if not selected_ids or any(value < 0 for value in selected_ids):
        raise ValueError("source_ids must be a non-empty non-negative sequence")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("source_ids must be unique")

    source_manifest = load_manifest(source_campaign)
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
                / ("summary.json")
            )
            source_ranks: dict[str, float] | None = None
            source_summary_sha256: str | None = None
            if source_summary.is_file():
                payload = json.loads(source_summary.read_text(encoding="utf-8"))
                ranks = payload.get("ranks")
                if isinstance(ranks, dict):
                    source_ranks = {
                        name: float(ranks[name]) for name in ("q", "ra", "t_c")
                    }
                source_summary_sha256 = file_sha256(source_summary)
            mapping.append(
                {
                    "diagnostic_id": diagnostic_id,
                    "source_injection_id": source_id,
                    "source_ranks": source_ranks,
                    "source_summary_sha256": source_summary_sha256,
                    "noise_seed": row["noise_seed"],
                    "sampler_seed": row["sampler_seed"],
                }
            )

        catalogue_path = output_campaign / "catalogue.csv"
        atomic_write_csv(catalogue_path, selected_rows, CATALOGUE_FIELDS)
        config = copy.deepcopy(source_manifest["config"])
        config.pop("sampler_scheduler", None)
        config.update(
            {
                "campaign": ("paper-baseline-code-high-res-d1-m3-pathology-diagnostic"),
                "paper_configuration": PAPER_CONFIGURATION,
                "n_devices": n_devices,
                "num_gibbs_sweeps": num_gibbs_sweeps,
            }
        )
        config["timing"]["selected_events"] = (
            "targeted paper-baseline pathology diagnostic; not a Figure 3 population"
        )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "master_seed": None,
            "n_injections": len(selected_rows),
            "catalogue_size": len(selected_rows),
            "selection": {
                "rule": "ranked pathological rows from a frozen FSM campaign",
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
                "methodology": "targeted differential against arXiv:2607.28265v1",
                "iid_prior_predictive_catalogue": False,
                "pp_calibration_eligible": False,
                "statement": (
                    "Exact selected rows from the completed FSM P-P campaign. "
                    "This diagnostic cannot be published as an independent P-P test."
                ),
            },
            "baseline_diagnostic": {
                "implementation_label": "paper-baseline",
                "implementation_revision": PAPER_BASELINE_REVISION,
                "implementation_tree_sha256": PAPER_BASELINE_TREE_SHA256,
                "paper_configuration": PAPER_CONFIGURATION,
                "configuration_semantics": (
                    "Table II High-Res settings (D=1, M=3) executed with the "
                    "pinned paper-baseline implementation"
                ),
                "paper_timing_available": False,
                "source_campaign_config_sha256": source_manifest["config_sha256"],
                "changed_variables": {
                    "implementation": "pinned paper baseline",
                    "paper_configuration": {
                        "source": source_manifest["config"]["paper_configuration"],
                        "diagnostic": PAPER_CONFIGURATION,
                    },
                    "n_devices": {
                        "source": source_manifest["config"]["n_devices"],
                        "diagnostic": n_devices,
                    },
                    "num_gibbs_sweeps": {
                        "source": source_manifest["config"]["num_gibbs_sweeps"],
                        "diagnostic": num_gibbs_sweeps,
                    },
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
    manifest = prepare_baseline_diagnostic(
        args.source_campaign,
        args.output_campaign,
        source_ids=args.source_ids,
        n_devices=args.n_devices,
        num_gibbs_sweeps=args.num_gibbs_sweeps,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
