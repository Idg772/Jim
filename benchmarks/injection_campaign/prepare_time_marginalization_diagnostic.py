"""Freeze four D=4/M=1 sentinels for the Appendix-A nuisance swap."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.injection_campaign.common import (
    CATALOGUE_FIELDS,
    PARAMETERS,
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
CAMPAIGN_NAME = "appendix-a-time-marginalized-d4-m1-pathology-diagnostic"
PAPER_CONFIGURATION = "Sharded time-marginalized diagnostic"
TIME_MARGINALIZATION = {
    "tc_range_seconds": [-0.1, 0.1],
    "upsample_factor": 32,
}
RECOVERY_LIKELIHOOD_F_MAX_HZ = 2048.0 - 1.0 / 128.0
TIMING_SELECTED_EVENTS = (
    "four targeted Appendix-A nuisance-swap sentinels at D=4, M=1; "
    "not a Figure 3 population"
)
COMMON_PARAMETERS = tuple(name for name in PARAMETERS if name != "t_c")
DIAGNOSTIC_PARAMETERS = (*COMMON_PARAMETERS, "d_L")
DIAGNOSTIC_MARGINALIZED_PARAMETERS = ("phase_c", "t_c")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


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
    return parser.parse_args(argv)


def _validate_implementation_pin(revision: str, tree_sha256: str) -> None:
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("implementation_revision must be a lowercase Git SHA")
    if _SHA256_RE.fullmatch(tree_sha256) is None:
        raise ValueError("implementation_tree_sha256 must be a lowercase SHA-256")


def _source_result_metadata(
    source_campaign: Path,
    source_manifest: dict[str, Any],
    source_id: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    directory = source_campaign / "results" / f"injection-{source_id:03d}"
    summary_path = directory / "summary.json"
    posterior_path = directory / "posterior.npz"
    if not summary_path.is_file() or not posterior_path.is_file():
        raise ValueError(f"source result is incomplete: {directory}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"source summary is invalid: {summary_path}") from error
    if (
        summary.get("config_sha256") != source_manifest["config_sha256"]
        or summary.get("injection_id") != source_id
        or summary.get("seeds")
        != {"noise": row["noise_seed"], "sampler": row["sampler_seed"]}
    ):
        raise ValueError(f"source result provenance mismatch: {summary_path}")
    posterior = summary.get("posterior")
    if (
        not isinstance(posterior, dict)
        or posterior.get("sha256") != file_sha256(posterior_path)
    ):
        raise ValueError(f"source posterior hash mismatch: {posterior_path}")
    ranks = summary.get("ranks")
    if not isinstance(ranks, dict) or any(name not in ranks for name in PARAMETERS):
        raise ValueError(f"source result has incomplete ranks: {summary_path}")
    return {
        "source_summary_sha256": file_sha256(summary_path),
        "source_posterior_sha256": file_sha256(posterior_path),
        "source_ranks": {name: float(ranks[name]) for name in PARAMETERS},
    }


def prepare_time_marginalization_diagnostic(
    source_campaign: Path,
    output_campaign: Path,
    *,
    implementation_revision: str,
    implementation_tree_sha256: str,
) -> dict[str, Any]:
    """Copy exact sentinel inputs and swap sampled ``t_c`` for sampled ``d_L``."""

    _validate_implementation_pin(
        implementation_revision,
        implementation_tree_sha256,
    )
    source_campaign = source_campaign.expanduser().resolve()
    output_campaign = output_campaign.expanduser().resolve()
    if output_campaign.exists():
        raise FileExistsError(f"diagnostic campaign already exists: {output_campaign}")

    selected_ids = DEFAULT_SOURCE_IDS

    source_manifest = load_manifest(source_campaign)
    source_config = source_manifest["config"]
    if (
        source_config.get("n_devices") != 4
        or source_config.get("num_gibbs_sweeps") != 1
        or source_config.get("sampler_scheduler", "fsm") != "fsm"
        or source_config.get("time_marginalization") is not False
        or not isinstance(source_config.get("distance_marginalization"), dict)
        or source_config.get("blocks", [])[-1:] != [["t_c"]]
    ):
        raise ValueError(
            "source campaign is not the expected distance-marginalized FSM D=4, M=1 campaign"
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
            source_row = source_catalogue[source_id]
            row = copy.deepcopy(source_row)
            row["injection_id"] = diagnostic_id
            selected_rows.append(row)
            result_metadata = _source_result_metadata(
                source_campaign,
                source_manifest,
                source_id,
                source_row,
            )
            mapping.append(
                {
                    "diagnostic_id": diagnostic_id,
                    "source_injection_id": source_id,
                    "noise_seed": row["noise_seed"],
                    "sampler_seed": row["sampler_seed"],
                    **result_metadata,
                }
            )

        catalogue_path = output_campaign / "catalogue.csv"
        atomic_write_csv(catalogue_path, selected_rows, CATALOGUE_FIELDS)

        config = copy.deepcopy(source_config)
        config.update(
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
        config["blocks"] = copy.deepcopy(source_config["blocks"])
        config["blocks"][-1] = ["d_L"]
        config["timing"]["selected_events"] = TIMING_SELECTED_EVENTS

        changed_variables = {
            "n_devices": {"source": 4, "diagnostic": 4},
            "num_gibbs_sweeps": {"source": 1, "diagnostic": 1},
            "sampled_nuisance": {"source": "t_c", "diagnostic": "d_L"},
            "marginalized_nuisance": {"source": "d_L", "diagnostic": "t_c"},
            "time_marginalization": {
                "source": False,
                "diagnostic": copy.deepcopy(TIME_MARGINALIZATION),
            },
            "distance_marginalization": {
                "source": copy.deepcopy(source_config["distance_marginalization"]),
                "diagnostic": False,
            },
            "final_block": {"source": ["t_c"], "diagnostic": ["d_L"]},
            "injection_f_max_hz": {
                "source": source_config["f_max_hz"],
                "diagnostic": source_config["f_max_hz"],
            },
            "recovery_likelihood_f_max_hz": {
                "source": source_config["f_max_hz"],
                "diagnostic": RECOVERY_LIKELIHOOD_F_MAX_HZ,
                "reason": (
                    "exclude the Nyquist bin from the time-marginalization FFT; "
                    "the injected signal band is unchanged"
                ),
            },
        }
        implementation_diagnostic = {
            "implementation_label": "candidate",
            "implementation_revision": implementation_revision,
            "implementation_tree_sha256": implementation_tree_sha256,
            "sampler_scheduler": "fsm",
            "configuration_semantics": (
                "Appendix-A nuisance-treatment diagnostic with D=4, M=1. "
                "Relative to the source, t_c is marginalized on a 32x grid and "
                "d_L is sampled; the recovery-only Nyquist endpoint is excluded."
            ),
            "source_campaign_config_sha256": source_manifest["config_sha256"],
            "changed_variables": changed_variables,
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "master_seed": None,
            "n_injections": len(selected_rows),
            "catalogue_size": len(selected_rows),
            "selection": {
                "rule": (
                    "exact completed pathological rows from the frozen Sharded "
                    "D=4, M=1 campaign"
                ),
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
                "methodology": "targeted Appendix-A nuisance-treatment differential",
                "iid_prior_predictive_catalogue": False,
                "pp_calibration_eligible": False,
                "statement": (
                    "These four selected source pathologies provide directional "
                    "evidence only and cannot be published as an independent P-P test."
                ),
            },
            "implementation_diagnostic": implementation_diagnostic,
            "appendix_a_diagnostic": {
                "sampled_parameters": list(DIAGNOSTIC_PARAMETERS),
                "marginalized_parameters": list(
                    DIAGNOSTIC_MARGINALIZED_PARAMETERS
                ),
                "common_rank_parameters": list(COMMON_PARAMETERS),
                "source_sampled_nuisance": "t_c",
                "diagnostic_sampled_nuisance": "d_L",
                "population_calibration_claim_permitted": False,
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
    manifest = prepare_time_marginalization_diagnostic(
        args.source_campaign,
        args.output_campaign,
        implementation_revision=args.implementation_revision,
        implementation_tree_sha256=args.implementation_tree_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
