"""Rebuild the attribution chain on a timing-stratified frozen subset.

The first eight cells retain the cumulative single-toggle configurations that
were reviewed in the 20260813 attribution campaign.  A ninth cell uses the
corrected-anchor production configuration as an endpoint reference.  Every
cell receives the same exact source rows and immutable PSD inputs.
"""

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
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_manifest,
    read_catalogue,
    refresh_status,
)

TEMPLATE_CELLS = (
    "00-legacy",
    "01-stepping-cache",
    "02-replicated-topology",
    "03-fsm-scheduler",
    "04-shared-frequency-grid",
    "05-real-angle-phasor",
    "06-real-inner-product",
    "07-cholesky-factor",
)
CELL_NAMES = (*TEMPLATE_CELLS, "08-production")
SELECTION_RULE = (
    "timing-stratified decile representatives of the corrected-anchor campaign"
)
PIN_FIELDS = (
    "implementation_label",
    "implementation_revision",
    "implementation_tree_sha256",
)
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _selected_rows(
    source_campaign: Path,
    source_manifest: dict[str, Any],
    subset: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = read_catalogue(source_campaign / source_manifest["catalogue"]["path"])
    if any(source_id < 0 or source_id >= len(source_rows) for source_id in subset):
        raise ValueError("source injection ID is outside the frozen catalogue")

    selected_rows: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for diagnostic_id, source_id in enumerate(subset):
        row = copy.deepcopy(source_rows[source_id])
        row["injection_id"] = diagnostic_id
        selected_rows.append(row)

        source_summary = (
            source_campaign / "results" / f"injection-{source_id:03d}" / "summary.json"
        )
        mapping.append(
            {
                "diagnostic_id": diagnostic_id,
                "source_injection_id": source_id,
                "noise_seed": row["noise_seed"],
                "sampler_seed": row["sampler_seed"],
                "source_summary_sha256": (
                    file_sha256(source_summary) if source_summary.is_file() else None
                ),
            }
        )
    return selected_rows, mapping


def _rewrite_manifest(
    template_manifest: dict[str, Any],
    *,
    source_campaign: Path,
    source_manifest: dict[str, Any],
    catalogue_path: Path,
    mapping: list[dict[str, Any]],
    subset: Sequence[int],
    campaign_name: str,
) -> dict[str, Any]:
    manifest = copy.deepcopy(template_manifest)
    manifest.pop("config_sha256", None)
    manifest["created_at_utc"] = datetime.now(UTC).isoformat()
    manifest["n_injections"] = len(subset)
    manifest["catalogue_size"] = len(subset)
    manifest["selection"] = {
        "rule": SELECTION_RULE,
        "source_injection_ids": list(subset),
        "start_inclusive": 0,
        "stop_exclusive": len(subset),
    }
    manifest["config"]["campaign"] = campaign_name
    manifest["catalogue"] = {
        "path": "catalogue.csv",
        "sha256": file_sha256(catalogue_path),
        "bytes": catalogue_path.stat().st_size,
        "generator": "exact selected and reindexed frozen source rows",
        "provenance": {
            "source_campaign": source_campaign.name,
            "source_manifest_sha256": file_sha256(source_campaign / "manifest.json"),
            "source_config_sha256": source_manifest["config_sha256"],
            "source_catalogue_sha256": source_manifest["catalogue"]["sha256"],
            "mapping": copy.deepcopy(mapping),
        },
    }
    manifest["psd"] = copy.deepcopy(source_manifest["psd"])
    manifest["config_sha256"] = canonical_sha256(manifest)
    return manifest


def prepare_cells(
    template_root: Path,
    source_campaign: Path,
    subset: list[int],
    output_root: Path,
    *,
    implementation_revision: str | None = None,
    implementation_tree_sha256: str | None = None,
) -> list[Path]:
    """Create eight cumulative attribution cells and one production reference."""

    template_root = template_root.expanduser().resolve()
    source_campaign = source_campaign.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    selected_ids = [int(source_id) for source_id in subset]
    if len(selected_ids) != 10 or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("subset must contain exactly 10 unique source injection IDs")

    source_manifest = load_manifest(source_campaign)
    template_manifests = {
        cell: load_manifest(template_root / cell) for cell in TEMPLATE_CELLS
    }
    template_pin = {
        name: template_manifests[TEMPLATE_CELLS[0]]["implementation_diagnostic"][name]
        for name in PIN_FIELDS
    }
    for cell, manifest in template_manifests.items():
        candidate = manifest.get("implementation_diagnostic", {})
        if {name: candidate.get(name) for name in PIN_FIELDS} != template_pin:
            raise ValueError(f"attribution template has a different pin: {cell}")
    if (implementation_revision is None) != (implementation_tree_sha256 is None):
        raise ValueError(
            "implementation_revision and implementation_tree_sha256 must be "
            "provided together"
        )
    if implementation_revision is None:
        reviewed_pin = template_pin
    else:
        assert implementation_tree_sha256 is not None
        if _REVISION_RE.fullmatch(implementation_revision) is None:
            raise ValueError("implementation_revision must be a lowercase Git SHA")
        if _SHA256_RE.fullmatch(implementation_tree_sha256) is None:
            raise ValueError(
                "implementation_tree_sha256 must be a lowercase SHA-256 digest"
            )
        reviewed_pin = {
            "implementation_label": "candidate",
            "implementation_revision": implementation_revision,
            "implementation_tree_sha256": implementation_tree_sha256,
        }
    selected_rows, mapping = _selected_rows(
        source_campaign, source_manifest, selected_ids
    )

    output_dirs = [output_root / cell for cell in CELL_NAMES]
    existing = [directory for directory in output_dirs if directory.exists()]
    if existing:
        raise FileExistsError(f"output campaign already exists: {existing[0]}")

    output_root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for cell, output_dir in zip(CELL_NAMES, output_dirs, strict=True):
            output_dir.mkdir()
            created.append(output_dir)
            shutil.copytree(source_campaign / "inputs", output_dir / "inputs")

            catalogue_path = output_dir / "catalogue.csv"
            atomic_write_csv(catalogue_path, selected_rows, CATALOGUE_FIELDS)

            if cell in template_manifests:
                template_manifest = copy.deepcopy(template_manifests[cell])
            else:
                template_manifest = copy.deepcopy(source_manifest)
                production_config = template_manifest["config"]
                production_config.pop("sampler_ablation_variant", None)
                production_config["sampler_scheduler"] = "fsm"
                production_config["likelihood_implementation"] = "optimized"
                production_config["likelihood_optimization_axes"] = {
                    "shared_frequency_grid": True,
                    "detector_phasor": True,
                    "real_inner_product": True,
                }
            diagnostic = template_manifest["implementation_diagnostic"]
            diagnostic.update(reviewed_pin)
            diagnostic.update(
                {
                    "attribution_name": cell,
                    "attribution_stage": int(cell[:2]),
                    "sampler_scheduler": template_manifest["config"][
                        "sampler_scheduler"
                    ],
                    "likelihood_implementation": template_manifest["config"][
                        "likelihood_implementation"
                    ],
                    "likelihood_optimization_axes": copy.deepcopy(
                        template_manifest["config"]["likelihood_optimization_axes"]
                    ),
                }
            )
            ablation_variant = template_manifest["config"].get(
                "sampler_ablation_variant"
            )
            if ablation_variant is None:
                diagnostic.pop("sampler_ablation_variant", None)
            else:
                diagnostic["sampler_ablation_variant"] = ablation_variant
            controls = diagnostic.setdefault("factorial_controls", {})
            controls.update(
                {
                    "cumulative_stage_order": list(CELL_NAMES),
                    "n_devices": template_manifest["config"]["n_devices"],
                    "num_gibbs_sweeps": template_manifest["config"]["num_gibbs_sweeps"],
                    "source_injection_ids": list(selected_ids),
                }
            )
            manifest = _rewrite_manifest(
                template_manifest,
                source_campaign=source_campaign,
                source_manifest=source_manifest,
                catalogue_path=catalogue_path,
                mapping=mapping,
                subset=selected_ids,
                campaign_name=f"pp-residual-attribution-{cell}",
            )
            atomic_write_json(output_dir / "manifest.json", manifest)
            refresh_status(output_dir, len(selected_ids))
            load_manifest(output_dir)
    except BaseException:
        for directory in reversed(created):
            shutil.rmtree(directory, ignore_errors=True)
        raise
    return output_dirs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_root", type=Path)
    parser.add_argument("source_campaign", type=Path)
    parser.add_argument("subset_json", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--implementation-revision")
    parser.add_argument("--implementation-tree-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    subset = json.loads(args.subset_json.read_text(encoding="utf-8"))["injection_ids"]
    directories = prepare_cells(
        args.template_root,
        args.source_campaign,
        subset,
        args.output_root,
        implementation_revision=args.implementation_revision,
        implementation_tree_sha256=args.implementation_tree_sha256,
    )
    for directory in directories:
        print(directory)


if __name__ == "__main__":
    main()
