"""Prepare matched-replicate D=4, M=1 blocking-scheme diagnostics.

Each invocation prepares one blocking scheme.  Preparing the schemes
separately is intentional: a campaign manifest has one sampler configuration,
while the sampler seeds below are derived without the scheme name so that all
campaigns remain paired.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.injection_campaign.common import (
    CATALOGUE_FIELDS,
    NETSKY_BLOCKS,
    NETSKY_SCHEME,
    SCHEMA_VERSION,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_manifest,
    read_catalogue,
    refresh_status,
    result_dir,
)
from benchmarks.injection_campaign.merge_staged_results import _validate_result

DEFAULT_SOURCE_IDS = (10, 19, 40)
DEFAULT_SAMPLER_REPLICATES = 3
SOURCE_CARRIER_TIME_ANCHOR = "imrphenomd"
SOURCE_RANK_PARAMETERS = ("ra", "dec", "t_c", "iota", "q")
SCHEMES = (
    "sky-time",
    "paper",
    "mass-time",
    "all-intrinsic",
    "all-slow",
    "all-slow-time",
    "fast-extrinsic",
    "full-extrinsic",
    "detector-time-fast-extrinsic",
)
BLOCKING_SCHEMES = SCHEMES
DEFAULT_SCHEME = SCHEMES[0]
SEED_DERIVATION_NAMESPACE = "jim-blocking-diagnostic-sampler-seed-v1"

_SKY_BLOCK = ("zenith", "azimuth")
_MASS_BLOCK = ("M_c", "q", "lambda_1", "lambda_2")
_S1_BLOCK = ("s1_mag", "s1_theta", "s1_phi")
_S2_BLOCK = ("s2_mag", "s2_theta", "s2_phi")
_IOTA_BLOCK = ("iota",)
_PSI_BLOCK = ("psi",)
_TIME_BLOCK = ("t_c",)
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


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
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
    parser.add_argument("--scheme", choices=SCHEMES, default=DEFAULT_SCHEME)
    parser.add_argument(
        "--sampler-replicates",
        type=_positive_int,
        default=DEFAULT_SAMPLER_REPLICATES,
        help="Number of matched sampler-seed replicates for each source row.",
    )
    parser.add_argument(
        "--source-ids",
        type=_source_ids,
        default=DEFAULT_SOURCE_IDS,
        help="Comma-separated source injection IDs in diagnostic order.",
    )
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


def _normalise_blocks(value: object) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("source campaign has no valid sampler blocks")
    blocks: list[list[str]] = []
    for index, raw in enumerate(value):
        if (
            not isinstance(raw, list)
            or not raw
            or not all(isinstance(name, str) and name for name in raw)
        ):
            raise ValueError(f"source sampler block {index} is invalid")
        blocks.append(list(raw))
    flattened = [name for block in blocks for name in block]
    if len(flattened) != len(set(flattened)):
        raise ValueError("source sampler blocks contain duplicate parameters")
    return blocks


def blocks_for_scheme(source_blocks: object, scheme: str) -> list[list[str]]:
    """Return the requested primary blocks for a diagnostic or remediation arm."""

    accepted_schemes = (*SCHEMES, NETSKY_SCHEME)
    if scheme not in accepted_schemes:
        raise ValueError(
            f"unknown blocking scheme {scheme!r}; expected {list(accepted_schemes)}"
        )
    blocks = _normalise_blocks(source_blocks)
    if scheme == NETSKY_SCHEME:
        return copy.deepcopy(NETSKY_BLOCKS)

    tuples = [tuple(block) for block in blocks]
    for required in (_SKY_BLOCK, _PSI_BLOCK, _TIME_BLOCK):
        if tuples.count(required) != 1:
            raise ValueError(
                "source campaign must contain separate [zenith, azimuth], [psi], "
                "and [t_c] paper blocks"
            )
    if scheme == "paper":
        return blocks

    if scheme in (
        "mass-time",
        "all-intrinsic",
        "all-slow",
        "all-slow-time",
    ):
        required_blocks = (_MASS_BLOCK, _S1_BLOCK, _S2_BLOCK, _IOTA_BLOCK)
        if any(tuples.count(required) != 1 for required in required_blocks):
            raise ValueError(
                "intrinsic blocking diagnostics require the four separate paper "
                "slow blocks"
            )
        if scheme == "mass-time":
            merged_members = {_MASS_BLOCK, _TIME_BLOCK}
            merged = [*_MASS_BLOCK, *_TIME_BLOCK]
        else:
            intrinsic_blocks = (_MASS_BLOCK, _S1_BLOCK, _S2_BLOCK)
            merged_members = set(intrinsic_blocks)
            merged = [name for block in intrinsic_blocks for name in block]
            if scheme in ("all-slow", "all-slow-time"):
                merged_members.add(_IOTA_BLOCK)
                merged.extend(_IOTA_BLOCK)
            if scheme == "all-slow-time":
                merged_members.add(_TIME_BLOCK)
                merged.extend(_TIME_BLOCK)
        merge_index = tuples.index(_MASS_BLOCK)
        return [
            merged if index == merge_index else block
            for index, block in enumerate(blocks)
            if index == merge_index or tuple(block) not in merged_members
        ]

    sky_index = tuples.index(_SKY_BLOCK)
    if scheme == "sky-time":
        merged_members = {_SKY_BLOCK, _TIME_BLOCK}
        merged = [*_SKY_BLOCK, *_TIME_BLOCK]
    elif scheme in ("fast-extrinsic", "detector-time-fast-extrinsic"):
        merged_members = {_SKY_BLOCK, _PSI_BLOCK, _TIME_BLOCK}
        time_name = "t_det" if scheme == "detector-time-fast-extrinsic" else "t_c"
        merged = [*_SKY_BLOCK, *_PSI_BLOCK, time_name]
    else:
        if tuples.count(_IOTA_BLOCK) != 1:
            raise ValueError("full-extrinsic requires a separate [iota] paper block")
        merged_members = {_IOTA_BLOCK, _SKY_BLOCK, _PSI_BLOCK, _TIME_BLOCK}
        merged = [*_IOTA_BLOCK, *_SKY_BLOCK, *_PSI_BLOCK, *_TIME_BLOCK]
    result: list[list[str]] = []
    for index, block in enumerate(blocks):
        block_tuple = tuple(block)
        if index == sky_index:
            result.append(merged)
        elif block_tuple not in merged_members:
            result.append(block)
    return result


def sampler_seed_for_replicate(
    *,
    source_catalogue_sha256: str,
    source_injection_id: int,
    source_sampler_seed: int,
    sampler_replicate: int,
) -> int:
    """Return a scheme-independent, stable uint32 sampler seed.

    Replicate zero is the exact completed source recovery.  Later replicates
    use the first four bytes of a namespaced SHA-256 digest in network byte
    order.  The blocking scheme is deliberately absent from the digest input.
    """

    if _SHA256_RE.fullmatch(source_catalogue_sha256) is None:
        raise ValueError("source_catalogue_sha256 must be a lowercase SHA-256")
    for field, value in (
        ("source_injection_id", source_injection_id),
        ("source_sampler_seed", source_sampler_seed),
        ("sampler_replicate", sampler_replicate),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} must be a non-negative exact integer")
    if source_sampler_seed >= 2**32:
        raise ValueError("source_sampler_seed must fit in uint32")
    if sampler_replicate == 0:
        return source_sampler_seed
    material = "\0".join(
        (
            SEED_DERIVATION_NAMESPACE,
            source_catalogue_sha256,
            str(source_injection_id),
            str(source_sampler_seed),
            str(sampler_replicate),
        )
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _load_source_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid source result summary {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"source result summary must be an object: {path}")
    return value


def prepare_blocking_diagnostic(
    source_campaign: Path,
    output_campaign: Path,
    *,
    implementation_revision: str,
    implementation_tree_sha256: str,
    scheme: str = DEFAULT_SCHEME,
    source_ids: Sequence[int] = DEFAULT_SOURCE_IDS,
    sampler_replicates: int = DEFAULT_SAMPLER_REPLICATES,
) -> dict[str, Any]:
    """Freeze paired source rows for one D=4, M=1 blocking scheme."""

    _validate_implementation_pin(implementation_revision, implementation_tree_sha256)
    if scheme not in SCHEMES:
        raise ValueError(
            f"unknown blocking scheme {scheme!r}; expected {list(SCHEMES)}"
        )
    if type(sampler_replicates) is not int or sampler_replicates < 1:
        raise ValueError("sampler_replicates must be a positive exact integer")
    selected_ids = tuple(int(value) for value in source_ids)
    if not selected_ids or any(value < 0 for value in selected_ids):
        raise ValueError("source_ids must be a non-empty non-negative sequence")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("source_ids must be unique")

    source_campaign = source_campaign.expanduser().resolve()
    output_campaign = output_campaign.expanduser().resolve()
    if output_campaign.exists():
        raise FileExistsError(f"diagnostic campaign already exists: {output_campaign}")

    source_manifest = load_manifest(source_campaign)
    source_config = _mapping(source_manifest.get("config"), field="source config")
    if (
        source_config.get("carrier_time_anchor") != SOURCE_CARRIER_TIME_ANCHOR
        or source_config.get("n_devices") != 4
        or source_config.get("num_gibbs_sweeps") != 1
        or source_config.get("sampler_scheduler", "fsm") != "fsm"
    ):
        raise ValueError(
            "source campaign is not a corrected-anchor D=4, M=1 FSM campaign"
        )
    source_blocks = _normalise_blocks(source_config.get("blocks"))
    diagnostic_blocks = blocks_for_scheme(source_blocks, scheme)
    source_catalogue_metadata = _mapping(
        source_manifest.get("catalogue"), field="source manifest.catalogue"
    )
    source_catalogue_sha256 = source_catalogue_metadata.get("sha256")
    if not isinstance(source_catalogue_sha256, str):
        raise TypeError("source catalogue has no SHA-256")
    source_catalogue = read_catalogue(
        source_campaign / str(source_catalogue_metadata["path"])
    )
    if any(value >= len(source_catalogue) for value in selected_ids):
        raise ValueError("source injection ID is outside the frozen catalogue")

    source_results: dict[int, dict[str, Any]] = {}
    for source_id in selected_ids:
        directory = result_dir(source_campaign, source_id)
        validated = _validate_result(
            directory,
            source_id,
            source_manifest,
            source_catalogue,
        )
        summary_path = directory / "summary.json"
        summary = _load_source_summary(summary_path)
        raw_ranks = _mapping(
            summary.get("ranks"), field=f"source ranks for injection {source_id}"
        )
        source_results[source_id] = {
            "ranks": {
                parameter: float(raw_ranks[parameter])
                for parameter in SOURCE_RANK_PARAMETERS
            },
            "summary_sha256": file_sha256(summary_path),
            "posterior_sha256": validated.posterior_sha256,
            "fingerprint": validated.fingerprint,
            "files_sha256": validated.file_hashes,
        }

    output_campaign.mkdir(parents=True)
    try:
        shutil.copytree(source_campaign / "inputs", output_campaign / "inputs")
        selected_rows: list[dict[str, Any]] = []
        provenance_mapping: list[dict[str, Any]] = []
        diagnostic_id = 0
        used_seeds: dict[int, set[int]] = {
            source_id: set() for source_id in selected_ids
        }
        for sampler_replicate in range(sampler_replicates):
            for source_id in selected_ids:
                source_row = source_catalogue[source_id]
                source_result = source_results[source_id]
                sampler_seed = sampler_seed_for_replicate(
                    source_catalogue_sha256=source_catalogue_sha256,
                    source_injection_id=source_id,
                    source_sampler_seed=int(source_row["sampler_seed"]),
                    sampler_replicate=sampler_replicate,
                )
                if sampler_seed in used_seeds[source_id]:
                    raise ValueError(
                        f"derived sampler-seed collision for source {source_id}"
                    )
                used_seeds[source_id].add(sampler_seed)
                row = copy.deepcopy(source_row)
                row["injection_id"] = diagnostic_id
                row["sampler_seed"] = sampler_seed
                selected_rows.append(row)
                provenance_mapping.append(
                    {
                        "diagnostic_id": diagnostic_id,
                        "source_injection_id": source_id,
                        "sampler_replicate": sampler_replicate,
                        "noise_seed": row["noise_seed"],
                        "source_sampler_seed": source_row["sampler_seed"],
                        "sampler_seed": sampler_seed,
                        "source_ranks": source_result["ranks"],
                        "source_summary_sha256": source_result["summary_sha256"],
                        "source_posterior_sha256": source_result["posterior_sha256"],
                        "source_result_fingerprint": source_result["fingerprint"],
                        "source_result_files_sha256": source_result["files_sha256"],
                    }
                )
                diagnostic_id += 1

        catalogue_path = output_campaign / "catalogue.csv"
        atomic_write_csv(catalogue_path, selected_rows, CATALOGUE_FIELDS)

        config = copy.deepcopy(dict(source_config))
        config.update(
            {
                "campaign": f"blocking-{scheme}-d4-m1-matched-replicate-diagnostic",
                "paper_configuration": f"Sharded {scheme} blocking diagnostic",
                "carrier_time_anchor": SOURCE_CARRIER_TIME_ANCHOR,
                "sampler_scheduler": "fsm",
                "n_devices": 4,
                "num_gibbs_sweeps": 1,
                "blocks": diagnostic_blocks,
            }
        )
        if scheme == "detector-time-fast-extrinsic":
            config["time_sampling_frame"] = "H1"
        timing = _mapping(config.get("timing"), field="source config.timing")
        config["timing"] = copy.deepcopy(dict(timing))
        config["timing"]["selected_events"] = (
            "fixed corrected-anchor pathologies with matched sampler "
            "replicates; targeted blocking diagnostic only"
        )

        source_manifest_path = source_campaign / "manifest.json"
        source_psd = _mapping(source_manifest.get("psd"), field="source manifest.psd")
        changed_variables = {
            "blocks": {
                "source": source_blocks,
                "diagnostic": diagnostic_blocks,
            },
            "catalogue_sampler_seeds": {
                "source": "one completed sampler seed per selected source row",
                "diagnostic": (
                    "replicate 0 preserves the source seed; replicates >=1 use "
                    "the frozen scheme-independent SHA-256 derivation"
                ),
            },
            "n_devices": {"source": 4, "diagnostic": 4},
            "num_gibbs_sweeps": {"source": 1, "diagnostic": 1},
            "sampler_scheduler": {"source": "fsm", "diagnostic": "fsm"},
            "carrier_time_anchor": {
                "source": SOURCE_CARRIER_TIME_ANCHOR,
                "diagnostic": SOURCE_CARRIER_TIME_ANCHOR,
            },
        }
        if scheme == "detector-time-fast-extrinsic":
            changed_variables["time_sampling_frame"] = {
                "source": source_config.get("time_sampling_frame", "geocentric"),
                "diagnostic": "H1",
            }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "master_seed": None,
            "n_injections": len(selected_rows),
            "catalogue_size": len(selected_rows),
            "selection": {
                "rule": (
                    "replicated completed rows from a frozen corrected-anchor "
                    "D=4 M=1 FSM campaign"
                ),
                "start_inclusive": 0,
                "stop_exclusive": len(selected_rows),
                "source_injection_ids": list(selected_ids),
                "sampler_replicates_per_source": sampler_replicates,
                "ordering": "sampler-replicate-major then source-order",
            },
            "config": config,
            "catalogue": {
                "path": "catalogue.csv",
                "sha256": file_sha256(catalogue_path),
                "bytes": catalogue_path.stat().st_size,
                "generator": (
                    "exact reindexed source rows with matched sampler-seed replicates"
                ),
                "provenance": {
                    "kind": "corrected-anchor-blocking-diagnostic",
                    "source_campaign": source_campaign.name,
                    "source_manifest_sha256": file_sha256(source_manifest_path),
                    "source_config_sha256": source_manifest["config_sha256"],
                    "source_catalogue_sha256": source_catalogue_sha256,
                    "source_psd_metadata_sha256": canonical_sha256(source_psd),
                    "seed_derivation": {
                        "namespace": SEED_DERIVATION_NAMESPACE,
                        "algorithm": "SHA-256",
                        "uint32_bytes": "digest[0:4] interpreted big-endian",
                        "digest_fields": [
                            "namespace",
                            "source_catalogue_sha256",
                            "source_injection_id",
                            "source_sampler_seed",
                            "sampler_replicate",
                        ],
                        "blocking_scheme_in_digest": False,
                        "replicate_zero": "preserve source sampler seed",
                    },
                    "mapping": provenance_mapping,
                },
            },
            "psd": copy.deepcopy(dict(source_psd)),
            "reproduction_scope": {
                "methodology": "matched-replicate blocking-scheme diagnostic",
                "iid_prior_predictive_catalogue": False,
                "pp_calibration_eligible": False,
                "statement": (
                    "Selected completed source rows are repeated under matched "
                    "sampler seeds. This targeted experiment can diagnose gross "
                    "modes but is not a P-P calibration population."
                ),
            },
            "implementation_diagnostic": {
                "implementation_label": "candidate",
                "implementation_revision": implementation_revision,
                "implementation_tree_sha256": implementation_tree_sha256,
                "diagnostic_kind": "blocking-scheme",
                "blocking_scheme": scheme,
                "source_campaign_config_sha256": source_manifest["config_sha256"],
                "configuration_semantics": (
                    "The corrected IMRPhenomD carrier anchor, D=4, M=1, FSM "
                    "scheduler, truths, noise realizations, PSDs, and all scientific "
                    "settings are frozen. Only the named proposal-block merge, "
                    "optional detector-arrival-time sampling coordinate, and "
                    "matched sampler-replicate seed are varied."
                ),
                "changed_variables": changed_variables,
            },
            "storage_policy": copy.deepcopy(source_manifest.get("storage_policy", {})),
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
    manifest = prepare_blocking_diagnostic(
        args.source_campaign,
        args.output_campaign,
        implementation_revision=args.implementation_revision,
        implementation_tree_sha256=args.implementation_tree_sha256,
        scheme=args.scheme,
        source_ids=args.source_ids,
        sampler_replicates=args.sampler_replicates,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
