"""Freeze a paired blocking-remediation campaign.

Legacy schemes retain the complete leading-100 M=1 source and change only its
block partition and labels. The netsky arm instead consumes one complete batch
from the frozen D=4/M=2 fast-ridge P-P baseline and records its additional
bridge, covariance, and quotient-fold configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.injection_campaign.common import (
    NETSKY_BRIDGE_BLOCKS,
    NETSKY_SCHEME,
    PAPER_PP_RECOVERIES,
    PARAMETERS,
    SCHEMA_VERSION,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_manifest,
    publication_eligible,
    read_catalogue,
    refresh_status,
    result_dir,
)
from benchmarks.injection_campaign.merge_staged_results import _validate_result
from benchmarks.injection_campaign.prepare_blocking_diagnostic import (
    SCHEMES as M1_SCHEMES,
)
from benchmarks.injection_campaign.prepare_blocking_diagnostic import blocks_for_scheme

SCHEMES = (*M1_SCHEMES, NETSKY_SCHEME)
DEFAULT_SCHEME = "all-slow-time"
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BATCH_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_COMPOSED_INDEX_SCHEMA_VERSION = 1
_COMPOSED_INDEX_FIELDS = frozenset(
    {
        "complete_first_attempt",
        "configuration_semantics_identical",
        "entries",
        "schema_version",
        "source_id_count",
        "source_ids",
        "unique_noise_seeds",
        "unique_sampler_seeds",
        "unique_seed_pairs",
        "unique_source_ids",
        "weighted_ranks_recomputed",
    }
)
_COMPOSED_ENTRY_FIELDS = frozenset(
    {
        "batch",
        "config_sha256",
        "local_injection_id",
        "noise_seed",
        "posterior",
        "posterior_sha256",
        "sampler_seed",
        "source_injection_id",
        "summary",
        "summary_sha256",
    }
)

_FAST_RIDGE_SOURCE_BLOCKS = [
    ["M_c", "q", "lambda_1", "lambda_2"],
    ["s1_mag", "s1_theta", "s1_phi"],
    ["s2_mag", "s2_theta", "s2_phi"],
    ["zenith", "azimuth"],
    ["psi"],
    ["cos_iota", "d_hat"],
]
_FAST_RIDGE_TIME_MARGINALIZATION = {
    "tc_range_seconds": [-0.03, 0.03],
    "upsample_factor": 1,
}
_NETSKY_SAMPLING_PARAMETERIZATION = {
    "distance_transform": ("log_d_hat=log(d_L/(M_c^(5/6)*R_net(ra,dec,psi,iota)))"),
    "sky_transform": (
        "(ra,dec) -> detector-frame (cos_zenith,azimuth) at trigger time"
    ),
    "joint_block": ["azimuth", "cos_iota", "psi", "log_d_hat"],
    "physical_output_parameters": ["ra", "dec", "iota", "psi", "d_L"],
    "sampling_space_parameters": [
        "cos_zenith",
        "azimuth",
        "cos_iota",
        "psi",
        "log_d_hat",
    ],
    "match": "Roulet-style network-frame blocking with Jim's R_net coordinate",
    "fold_model_limitation": (
        "aligned/weak-precession proxy: cos_iota is not cos(theta_JN) and the "
        "full precessing phi_JL handedness term is omitted; keep folding "
        "default-off outside the validated low-spin campaign"
    ),
}


@dataclass(frozen=True)
class _ComposedSource:
    """One validated leading population assembled from complete source batches."""

    manifest: dict[str, Any]
    catalogue: list[dict[str, Any]]
    input_campaign: Path
    index_path: Path
    batches: list[dict[str, Any]]


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _repository_relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"composed source {field} must be a path string")
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(
            f"composed source {field} must be a normalized repository-relative path"
        )
    return path


def _find_index_root(source_campaign: Path, paths: list[Path]) -> Path:
    candidates = [
        candidate
        for candidate in (source_campaign, *source_campaign.parents)
        if all((candidate / path).is_file() for path in paths)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "composed source paths do not resolve from one unambiguous repository root"
        )
    return candidates[0]


def _science_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove batch-specific presentation text from a source configuration."""

    science = copy.deepcopy(config)
    science.pop("campaign", None)
    science.pop("paper_configuration", None)
    timing = science.get("timing")
    if isinstance(timing, dict):
        timing.pop("selected_events", None)
    return science


def _is_exact_id_sequence(value: Any, expected: list[int]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(type(item) is int for item in value)
        and value == expected
    )


def _validate_psd_inventory(psd: Any, detectors: Any) -> None:
    if not isinstance(psd, dict):
        raise TypeError("composed source has no valid PSD inventory")
    files = psd.get("files")
    detector_files = psd.get("detector_files")
    if (
        not isinstance(files, dict)
        or not files
        or not isinstance(detector_files, dict)
        or not isinstance(detectors, list)
        or set(detector_files) != set(detectors)
    ):
        raise ValueError("composed source PSD inventory is incomplete")
    for detector, relative in detector_files.items():
        if not isinstance(detector, str) or relative not in files:
            raise ValueError("composed source PSD detector mapping is invalid")
    for relative, metadata in files.items():
        try:
            normalized = _repository_relative_path(relative, field="PSD inventory path")
        except (TypeError, ValueError) as error:
            raise ValueError("composed source PSD inventory path is invalid") from error
        if normalized.as_posix() != relative or not isinstance(metadata, dict):
            raise ValueError("composed source PSD inventory entry is invalid")
        sha256 = metadata.get("sha256")
        if (
            not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
            or type(metadata.get("bytes")) is not int
            or metadata["bytes"] < 1
        ):
            raise ValueError("composed source PSD inventory metadata is invalid")


def _validate_batch_source_mapping(
    manifest: dict[str, Any],
    source_ids: list[int],
    catalogue: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or (
        selection.get("start_inclusive") != 0
        or selection.get("stop_exclusive") != len(catalogue)
    ):
        raise ValueError(f"composed source batch {label!r} selection is invalid")
    provenance = manifest.get("catalogue", {}).get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    declared_id_lists = [
        value
        for value in (
            selection.get("source_injection_ids"),
            provenance.get("source_ids"),
        )
        if value is not None
    ]
    if not declared_id_lists or any(
        not _is_exact_id_sequence(value, source_ids) for value in declared_id_lists
    ):
        raise ValueError(
            f"composed source batch {label!r} source-ID declaration mismatch"
        )
    expected_mapping = [
        {
            "diagnostic_id": local_id,
            "source_injection_id": source_id,
            "noise_seed": catalogue[local_id]["noise_seed"],
            "sampler_seed": catalogue[local_id]["sampler_seed"],
        }
        for local_id, source_id in enumerate(source_ids)
    ]
    for mapping in (selection.get("mapping"), provenance.get("mapping")):
        if mapping is not None and mapping != expected_mapping:
            raise ValueError(
                f"composed source batch {label!r} seed mapping declaration mismatch"
            )


def _load_composed_manifest(
    source_campaign: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the combined catalogue view without requiring local PSD copies."""

    manifest_path = source_campaign / "manifest.json"
    manifest = _load_json_object(manifest_path, label="composed source manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported campaign schema in {manifest_path}")
    stored_hash = manifest.get("config_sha256")
    hash_input = dict(manifest)
    hash_input.pop("config_sha256", None)
    if stored_hash != canonical_sha256(hash_input):
        raise ValueError(f"campaign manifest hash mismatch: {manifest_path}")
    if not publication_eligible(manifest):
        raise ValueError("source campaign is not an iid P-P calibration campaign")

    expected_ids = list(range(PAPER_PP_RECOVERIES))
    if (
        manifest.get("n_injections") != PAPER_PP_RECOVERIES
        or manifest.get("catalogue_size") != PAPER_PP_RECOVERIES
    ):
        raise ValueError(
            "composed source must contain the complete leading paper population"
        )
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or (
        selection.get("start_inclusive") != 0
        or selection.get("stop_exclusive") != PAPER_PP_RECOVERIES
        or not _is_exact_id_sequence(
            selection.get("source_injection_ids"), expected_ids
        )
    ):
        raise ValueError(
            "composed source selection must cover source IDs in exact leading order"
        )
    catalogue_metadata = manifest.get("catalogue")
    if not isinstance(catalogue_metadata, dict):
        raise TypeError("composed source has no valid catalogue metadata")
    catalogue_relative = _repository_relative_path(
        catalogue_metadata.get("path"), field="catalogue path"
    )
    catalogue_path = source_campaign / catalogue_relative
    if file_sha256(catalogue_path) != catalogue_metadata.get("sha256"):
        raise ValueError(f"campaign catalogue hash mismatch: {catalogue_path}")
    if catalogue_metadata.get("bytes") != catalogue_path.stat().st_size:
        raise ValueError(f"campaign catalogue byte count mismatch: {catalogue_path}")
    catalogue = read_catalogue(catalogue_path)
    if len(catalogue) != PAPER_PP_RECOVERIES:
        raise ValueError("composed source catalogue size does not match its manifest")
    return manifest, catalogue


def _validate_composed_index(
    source_campaign: Path,
    manifest: dict[str, Any],
    catalogue: list[dict[str, Any]],
) -> _ComposedSource:
    """Bind every combined index entry to a strict batch result validation."""

    index_path = source_campaign / "index.json"
    index = _load_json_object(index_path, label="composed source index")
    if frozenset(index) != _COMPOSED_INDEX_FIELDS:
        raise ValueError("composed source index fields do not match schema")
    expected_ids = list(range(PAPER_PP_RECOVERIES))
    entries = index.get("entries")
    if not isinstance(entries, list) or len(entries) != PAPER_PP_RECOVERIES:
        raise ValueError("composed source index must contain one entry per source ID")
    if (
        index.get("schema_version") != _COMPOSED_INDEX_SCHEMA_VERSION
        or index.get("complete_first_attempt") != PAPER_PP_RECOVERIES
        or index.get("configuration_semantics_identical") is not True
        or index.get("source_id_count") != PAPER_PP_RECOVERIES
        or not _is_exact_id_sequence(index.get("source_ids"), expected_ids)
        or index.get("unique_source_ids") != PAPER_PP_RECOVERIES
        or index.get("unique_noise_seeds") != PAPER_PP_RECOVERIES
        or index.get("unique_sampler_seeds") != PAPER_PP_RECOVERIES
        or index.get("unique_seed_pairs") != PAPER_PP_RECOVERIES
        or index.get("weighted_ranks_recomputed")
        != PAPER_PP_RECOVERIES * len(PARAMETERS)
    ):
        raise ValueError("composed source index population summary is invalid")

    normalized_entries: list[dict[str, Any]] = []
    indexed_paths: list[Path] = []
    for source_id, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict) or frozenset(raw_entry) != (
            _COMPOSED_ENTRY_FIELDS
        ):
            raise ValueError(f"composed source entry {source_id} fields are invalid")
        entry = copy.deepcopy(raw_entry)
        if (
            type(entry.get("source_injection_id")) is not int
            or entry["source_injection_id"] != source_id
        ):
            raise ValueError(
                "composed source entries must follow exact source-ID order"
            )
        batch = entry.get("batch")
        if not isinstance(batch, str) or _BATCH_LABEL_RE.fullmatch(batch) is None:
            raise ValueError(f"composed source entry {source_id} has invalid batch")
        local_id = entry.get("local_injection_id")
        if type(local_id) is not int or local_id < 0:
            raise ValueError(
                f"composed source entry {source_id} has invalid local injection ID"
            )
        for seed_field in ("noise_seed", "sampler_seed"):
            if type(entry.get(seed_field)) is not int or entry[seed_field] < 0:
                raise ValueError(
                    f"composed source entry {source_id} has invalid {seed_field}"
                )
        for hash_field in (
            "config_sha256",
            "summary_sha256",
            "posterior_sha256",
        ):
            value = entry.get(hash_field)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(
                    f"composed source entry {source_id} has invalid {hash_field}"
                )
        summary_relative = _repository_relative_path(
            entry.get("summary"), field=f"entry {source_id} summary"
        )
        posterior_relative = _repository_relative_path(
            entry.get("posterior"), field=f"entry {source_id} posterior"
        )
        entry["summary_relative"] = summary_relative
        entry["posterior_relative"] = posterior_relative
        indexed_paths.extend((summary_relative, posterior_relative))
        normalized_entries.append(entry)

    repository_root = _find_index_root(source_campaign, indexed_paths)
    source_ids = [entry["source_injection_id"] for entry in normalized_entries]
    noise_seeds = [entry["noise_seed"] for entry in normalized_entries]
    sampler_seeds = [entry["sampler_seed"] for entry in normalized_entries]
    if (
        source_ids != expected_ids
        or len(set(noise_seeds)) != PAPER_PP_RECOVERIES
        or len(set(sampler_seeds)) != PAPER_PP_RECOVERIES
        or len(set(zip(noise_seeds, sampler_seeds, strict=True))) != PAPER_PP_RECOVERIES
    ):
        raise ValueError("composed source index IDs or seed inventory is invalid")

    batch_entries: dict[str, list[dict[str, Any]]] = {}
    batch_roots: dict[str, Path] = {}
    batch_order: list[str] = []
    previous_batch: str | None = None
    for entry in normalized_entries:
        label = entry["batch"]
        summary_path = repository_root / entry.pop("summary_relative")
        posterior_path = repository_root / entry.pop("posterior_relative")
        if (
            summary_path.name != "summary.json"
            or posterior_path.name != "posterior.npz"
        ):
            raise ValueError("composed source entry artifact names are invalid")
        if summary_path.parent != posterior_path.parent:
            raise ValueError("composed source entry artifacts do not share a result")
        batch_root = summary_path.parents[2]
        if summary_path.parent != result_dir(batch_root, entry["local_injection_id"]):
            raise ValueError("composed source entry path does not match its local ID")
        if label != previous_batch:
            if label in batch_entries:
                raise ValueError("composed source batch entries must be contiguous")
            batch_order.append(label)
            batch_entries[label] = []
            previous_batch = label
        if label in batch_roots and batch_roots[label] != batch_root:
            raise ValueError("composed source batch label maps to multiple campaigns")
        if batch_root in batch_roots.values() and label not in batch_roots:
            raise ValueError("composed source campaign maps to multiple batch labels")
        batch_roots[label] = batch_root
        entry["summary_path"] = summary_path
        entry["posterior_path"] = posterior_path
        batch_entries[label].append(entry)

    combined_config = manifest.get("config")
    if not isinstance(combined_config, dict) or not _is_expected_netsky_source(
        combined_config
    ):
        raise ValueError(
            "composed source is not the expected fast-ridge science configuration"
        )
    reference_science = _science_config(combined_config)
    reference_psd = manifest.get("psd")
    _validate_psd_inventory(reference_psd, combined_config.get("detectors"))
    reference_implementation = manifest.get("implementation_diagnostic")
    provenance_batches: list[dict[str, Any]] = []
    input_campaign: Path | None = None
    selection = manifest["selection"]
    expected_batch_selection = []

    for label in batch_order:
        batch_root = batch_roots[label]
        batch_manifest = load_manifest(batch_root)
        if not publication_eligible(batch_manifest):
            raise ValueError(
                f"composed source batch {label!r} is not publication eligible"
            )
        batch_config = batch_manifest.get("config")
        if not isinstance(batch_config, dict) or not _is_expected_netsky_source(
            batch_config
        ):
            raise ValueError(
                f"composed source batch {label!r} is not the expected fast-ridge source"
            )
        if _science_config(batch_config) != reference_science:
            raise ValueError(
                f"composed source batch {label!r} science configuration mismatch"
            )
        if batch_manifest.get("psd") != reference_psd:
            raise ValueError(f"composed source batch {label!r} PSD inventory mismatch")
        if batch_manifest.get("implementation_diagnostic") != reference_implementation:
            raise ValueError(
                f"composed source batch {label!r} implementation provenance mismatch"
            )
        batch_catalogue = read_catalogue(
            batch_root / str(batch_manifest["catalogue"]["path"])
        )
        indexed = batch_entries[label]
        if (
            len(batch_catalogue) != batch_manifest.get("catalogue_size")
            or len(batch_catalogue) != batch_manifest.get("n_injections")
            or [entry["local_injection_id"] for entry in indexed]
            != list(range(len(batch_catalogue)))
        ):
            raise ValueError(
                f"composed source batch {label!r} local result coverage is invalid"
            )

        batch_source_ids: list[int] = []
        for entry in indexed:
            source_id = entry["source_injection_id"]
            local_id = entry["local_injection_id"]
            batch_source_ids.append(source_id)
            batch_row = copy.deepcopy(batch_catalogue[local_id])
            batch_row["injection_id"] = source_id
            if batch_row != catalogue[source_id]:
                raise ValueError(
                    f"composed source entry {source_id} catalogue mapping mismatch"
                )
            combined_row = catalogue[source_id]
            if (
                entry["noise_seed"] != combined_row["noise_seed"]
                or entry["sampler_seed"] != combined_row["sampler_seed"]
            ):
                raise ValueError(
                    f"composed source entry {source_id} seed mapping mismatch"
                )
            if entry["config_sha256"] != batch_manifest["config_sha256"]:
                raise ValueError(
                    f"composed source entry {source_id} config hash mismatch"
                )
            if file_sha256(entry["summary_path"]) != entry["summary_sha256"]:
                raise ValueError(
                    f"composed source entry {source_id} summary hash mismatch"
                )
            if file_sha256(entry["posterior_path"]) != entry["posterior_sha256"]:
                raise ValueError(
                    f"composed source entry {source_id} posterior hash mismatch"
                )
            validated = _validate_result(
                result_dir(batch_root, local_id),
                local_id,
                batch_manifest,
                batch_catalogue,
            )
            if validated.posterior_sha256 != entry["posterior_sha256"]:
                raise ValueError(
                    f"composed source entry {source_id} validated posterior mismatch"
                )

        expected_batch_selection.append(
            {"label": label, "source_injection_ids": batch_source_ids}
        )
        _validate_batch_source_mapping(
            batch_manifest,
            batch_source_ids,
            batch_catalogue,
            label=label,
        )
        provenance_batches.append(
            {
                "label": label,
                "campaign": str(batch_root.relative_to(repository_root)),
                "manifest_sha256": file_sha256(batch_root / "manifest.json"),
                "config_sha256": batch_manifest["config_sha256"],
                "catalogue_sha256": batch_manifest["catalogue"]["sha256"],
                "source_injection_ids": batch_source_ids,
                "result_count": len(indexed),
            }
        )
        if input_campaign is None:
            input_campaign = batch_root

    if selection.get("batches") != expected_batch_selection:
        raise ValueError("composed source manifest batch selection mismatch")
    assert input_campaign is not None
    return _ComposedSource(
        manifest=manifest,
        catalogue=catalogue,
        input_campaign=input_campaign,
        index_path=index_path,
        batches=provenance_batches,
    )


def _load_composed_source(source_campaign: Path) -> _ComposedSource:
    manifest, catalogue = _load_composed_manifest(source_campaign)
    return _validate_composed_index(source_campaign, manifest, catalogue)


def _is_expected_netsky_source(config: dict[str, Any]) -> bool:
    """Return whether config is the frozen D=4/M=2 fast-ridge source arm."""

    prior = config.get("prior")
    return (
        config.get("blocking_scheme") == "fast-ridge"
        and config.get("blocks") == _FAST_RIDGE_SOURCE_BLOCKS
        and config.get("direction_mode") == "covariance"
        and config.get("adaptive_slice_widths") is True
        and config.get("bracket_mode") == "shrink-only"
        and type(config.get("num_de_jumps")) is int
        and config.get("num_de_jumps") == 0
        and config.get("phase_marginalization") is True
        and config.get("time_marginalization") == _FAST_RIDGE_TIME_MARGINALIZATION
        and config.get("distance_marginalization") is False
        and type(config.get("n_devices")) is int
        and config.get("n_devices") == 4
        and type(config.get("num_gibbs_sweeps")) is int
        and config.get("num_gibbs_sweeps") == 2
        and config.get("sampler_scheduler") == "fsm"
        and config.get("width_adaptation_rate") == 0.25
        and config.get("width_target_expansions") == 1.0
        and config.get("width_target_shrinks") == 3.0
        and config.get("detectors") == ["H1", "L1", "V1"]
        and config.get("waveform") == "IMRPhenomPv2_NRTidalv2"
        and isinstance(prior, dict)
        and prior.get("phase_c")
        == {
            "distribution": "uniform",
            "range_radians": [0.0, "2pi"],
        }
        and prior.get("psi")
        == {
            "distribution": "uniform",
            "range_radians": [0.0, "pi"],
        }
        and prior.get("spin_magnitudes")
        == {"distribution": "uniform", "range": [0.0, 0.05]}
    )


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
    parser.add_argument("--implementation-revision", type=_revision, required=True)
    parser.add_argument(
        "--implementation-tree-sha256", type=_tree_sha256, required=True
    )
    return parser.parse_args(argv)


def prepare_remediation_campaign(
    source_campaign: Path,
    output_campaign: Path,
    *,
    scheme: str,
    implementation_revision: str,
    implementation_tree_sha256: str,
) -> dict[str, Any]:
    """Copy an eligible paired source into the requested remediation arm."""

    if _REVISION_RE.fullmatch(implementation_revision) is None:
        raise ValueError("implementation_revision must be a lowercase Git SHA")
    if _SHA256_RE.fullmatch(implementation_tree_sha256) is None:
        raise ValueError("implementation_tree_sha256 must be a lowercase SHA-256")
    if scheme not in SCHEMES:
        raise ValueError(
            f"unknown remediation scheme {scheme!r}; expected {list(SCHEMES)}"
        )
    is_netsky = scheme == NETSKY_SCHEME

    source_campaign = source_campaign.expanduser().resolve()
    output_campaign = output_campaign.expanduser().resolve()
    if output_campaign.exists():
        raise FileExistsError(f"remediation campaign already exists: {output_campaign}")

    composed_source = (
        _load_composed_source(source_campaign)
        if is_netsky and (source_campaign / "index.json").is_file()
        else None
    )
    source_manifest = (
        composed_source.manifest
        if composed_source is not None
        else load_manifest(source_campaign)
    )
    if not publication_eligible(source_manifest):
        raise ValueError("source campaign is not an iid P-P calibration campaign")
    source_recoveries = source_manifest.get("n_injections")
    source_catalogue_size = source_manifest.get("catalogue_size")
    if is_netsky and (
        type(source_recoveries) is not int
        or source_recoveries < 1
        or type(source_catalogue_size) is not int
        or source_catalogue_size != source_recoveries
    ):
        raise ValueError("netsky source campaign must be a complete non-empty batch")
    if not is_netsky and int(source_recoveries or -1) != PAPER_PP_RECOVERIES:
        raise ValueError(
            f"source campaign must select exactly {PAPER_PP_RECOVERIES} recoveries"
        )
    selection = source_manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("start_inclusive") != 0:
        raise ValueError("source campaign must use the leading contiguous selection")
    expected_stop = source_recoveries if is_netsky else PAPER_PP_RECOVERIES
    if selection.get("stop_exclusive") != expected_stop:
        if is_netsky:
            raise ValueError("netsky source selection must cover its complete batch")
        raise ValueError("source campaign selection does not end at recovery 100")

    source_config = source_manifest.get("config")
    if not isinstance(source_config, dict):
        raise TypeError("source campaign has no valid configuration")
    if is_netsky and not _is_expected_netsky_source(source_config):
        raise ValueError(
            "source campaign is not the expected fast-ridge, time-marginalized, "
            "sampled-d_L FSM D=4, M=2 campaign"
        )
    if not is_netsky and (
        source_config.get("n_devices") != 4
        or source_config.get("num_gibbs_sweeps") != 1
        or source_config.get("sampler_scheduler", "fsm") != "fsm"
    ):
        raise ValueError("source campaign is not the expected FSM D=4, M=1 campaign")
    source_blocks = source_config.get("blocks")
    remediation_blocks = blocks_for_scheme(source_blocks, scheme)

    catalogue = (
        composed_source.catalogue
        if composed_source is not None
        else read_catalogue(source_campaign / str(source_manifest["catalogue"]["path"]))
    )
    if len(catalogue) != int(source_manifest.get("catalogue_size", -1)):
        raise ValueError("source catalogue size does not match its manifest")
    selected_recoveries = source_recoveries if is_netsky else PAPER_PP_RECOVERIES
    assert isinstance(selected_recoveries, int)
    if composed_source is None:
        for injection_id in range(selected_recoveries):
            _validate_result(
                result_dir(source_campaign, injection_id),
                injection_id,
                source_manifest,
                catalogue,
            )

    output_campaign.mkdir(parents=True)
    try:
        catalogue_relative = Path(str(source_manifest["catalogue"]["path"]))
        catalogue_destination = output_campaign / catalogue_relative
        catalogue_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_campaign / catalogue_relative, catalogue_destination)
        input_campaign = (
            composed_source.input_campaign
            if composed_source is not None
            else source_campaign
        )
        shutil.copytree(input_campaign / "inputs", output_campaign / "inputs")
        if composed_source is not None:
            shutil.copy2(
                composed_source.index_path,
                output_campaign / "source-index.json",
            )

        config = copy.deepcopy(source_config)
        if is_netsky:
            config["campaign"] = "pp-netsky-m2-paired-remediation"
            config["paper_configuration"] = "D=4/M=2 netsky paired arm"
        else:
            config["campaign"] = f"paper-fig2a-m1-{scheme}-remediation"
            config["paper_configuration"] = f"Sharded M=1 ({scheme} blocks)"
        config["blocks"] = remediation_blocks
        if is_netsky:
            config.update(
                {
                    "adaptive_slice_widths": False,
                    "blocking_scheme": NETSKY_SCHEME,
                    "bridge_blocks": copy.deepcopy(NETSKY_BRIDGE_BLOCKS),
                    "bracket_mode": "stepping-out",
                    "periodic_wrapped_covariance": True,
                    "num_gibbs_sweeps": 2,
                    "fold_symmetry": {},
                    "fold_unfold_batch_size": 1,
                }
            )

        changed_variables: dict[str, Any] = {
            "blocks": {
                "source": copy.deepcopy(source_blocks),
                "remediation": copy.deepcopy(remediation_blocks),
            }
        }
        if is_netsky:
            changed_variables.update(
                {
                    "adaptive_slice_widths": {
                        "source": source_config.get("adaptive_slice_widths"),
                        "remediation": False,
                    },
                    "blocking_scheme": {
                        "source": source_config.get("blocking_scheme"),
                        "remediation": NETSKY_SCHEME,
                    },
                    "bridge_blocks": {
                        "source": copy.deepcopy(source_config.get("bridge_blocks", [])),
                        "remediation": copy.deepcopy(NETSKY_BRIDGE_BLOCKS),
                    },
                    "bracket_mode": {
                        "source": source_config.get("bracket_mode"),
                        "remediation": "stepping-out",
                    },
                    "periodic_wrapped_covariance": {
                        "source": bool(
                            source_config.get("periodic_wrapped_covariance", False)
                        ),
                        "remediation": True,
                    },
                    "fold_symmetry": {
                        "source": copy.deepcopy(source_config.get("fold_symmetry")),
                        "remediation": {},
                    },
                    "fold_unfold_batch_size": {
                        "source": source_config.get("fold_unfold_batch_size"),
                        "remediation": 1,
                    },
                }
            )

        manifest = {
            key: copy.deepcopy(value)
            for key, value in source_manifest.items()
            if key
            not in {"config_sha256", "created_at_utc", "implementation_diagnostic"}
        }
        manifest.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "config": config,
                "catalogue": {
                    **copy.deepcopy(source_manifest["catalogue"]),
                    "sha256": file_sha256(catalogue_destination),
                    "bytes": catalogue_destination.stat().st_size,
                },
                "implementation_diagnostic": {
                    "implementation_label": "candidate",
                    "implementation_revision": implementation_revision,
                    "implementation_tree_sha256": implementation_tree_sha256,
                    "sampler_scheduler": "fsm",
                    "configuration_semantics": (
                        "Paired D=4/M=2 netsky remediation of the complete "
                        "fast-ridge real-PE-matched P-P source batch. Catalogue "
                        "rows, truths, seeds, PSDs, priors, likelihood, live set, "
                        "stopping rule, and nuisance treatment remain frozen; the "
                        "recorded netsky blocking, bridge, fixed-width stepping-out "
                        "covariance, and quotient-fold configuration selects this arm."
                        if is_netsky
                        else (
                            "Paired full Fig. 2a remediation: D=4, M=1, the complete "
                            "iid catalogue and leading-100 selection, truths, seeds, "
                            "PSDs, priors, likelihood, live set, stopping rule, and "
                            "all other scientific settings match the source. Only "
                            "the block partition and presentation labels change."
                        )
                    ),
                    "source_campaign_config_sha256": source_manifest["config_sha256"],
                    "changed_variables": changed_variables,
                },
                "blocking_remediation": {
                    "scheme": scheme,
                    "paired_source_campaign": source_campaign.name,
                    "paired_source_manifest_sha256": file_sha256(
                        source_campaign / "manifest.json"
                    ),
                    "acceptance_test": (
                        "Fisher-combined exact KS p-value over all 15 sampled "
                        "parameters, including q, must exceed 0.05"
                    ),
                },
            }
        )
        if composed_source is not None:
            manifest["blocking_remediation"]["composed_source"] = {
                "index": {
                    "path": "source-index.json",
                    "sha256": file_sha256(output_campaign / "source-index.json"),
                    "entries": selected_recoveries,
                },
                "batches": copy.deepcopy(composed_source.batches),
                "validation": (
                    "Every indexed artifact hash and every complete batch manifest, "
                    "catalogue, result, source-ID/seed mapping, science configuration, "
                    "and common PSD inventory passed strict validation."
                ),
            }
        if is_netsky:
            manifest["sampling_parameterization"] = copy.deepcopy(
                _NETSKY_SAMPLING_PARAMETERIZATION
            )
        manifest["config_sha256"] = canonical_sha256(manifest)
        atomic_write_json(output_campaign / "manifest.json", manifest)
        refresh_status(output_campaign, selected_recoveries)
        load_manifest(output_campaign)
        return manifest
    except BaseException:
        shutil.rmtree(output_campaign, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = prepare_remediation_campaign(
        args.source_campaign,
        args.output_campaign,
        scheme=args.scheme,
        implementation_revision=args.implementation_revision,
        implementation_tree_sha256=args.implementation_tree_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
