"""Safely merge cumulative staged result archives into a frozen campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import numpy as np

from benchmarks.injection_campaign.common import (
    FOLDED_TARGET_SEMANTICS,
    MARGINALIZED_PARAMETERS,
    PARAMETERS,
    POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS,
    UNFOLDED_POSTERIOR_WEIGHTING,
    UNFOLDED_RANK_WEIGHTING,
    file_sha256,
    load_manifest,
    posterior_rank,
    read_catalogue,
    refresh_status,
    status_rows,
)
from benchmarks.injection_campaign.run_injection import (
    _rank_truth_coordinates,
    _sampled_parameters,
)

_RESULT_DIRECTORY_RE = re.compile(r"injection-([0-9]{3,})\Z")
_ATTEMPT_LOG_RE = re.compile(r"attempt-[0-9]+\.(?:failed|success)\.log\Z")
_RANK_METHOD = {
    "comparison": "sample < truth",
    "resampled": False,
    "weighting": "original nested-sampling weights",
}
_POSTERIOR_WEIGHTING = "normalized nested-sampling log weights"
_UNFOLDED_RANK_WEIGHTING = UNFOLDED_RANK_WEIGHTING
_UNFOLDED_RANK_METHOD = {
    "comparison": "sample < truth",
    "resampled": False,
    "weighting": _UNFOLDED_RANK_WEIGHTING,
}
_UNFOLDED_POSTERIOR_WEIGHTING = UNFOLDED_POSTERIOR_WEIGHTING
_FOLDED_TARGET_SEMANTICS = FOLDED_TARGET_SEMANTICS
_POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS = POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
_LOG_WEIGHT_NORMALIZATION_TOLERANCE = 1.0e-10
_RANK_TOLERANCE = 1.0e-12
_INSERTION_DIAGNOSTIC_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class ValidatedResult:
    """One complete result whose scientific payload passed validation."""

    injection_id: int
    directory: Path
    fingerprint: str
    file_hashes: dict[str, str]
    posterior_sha256: str
    posterior_bytes: int
    posterior_samples: int
    post_jit_sampling_seconds: float | None


@dataclass(frozen=True)
class ValidatedArchive:
    """A safely extracted cumulative campaign archive."""

    path: Path
    sha256: str
    root: Path
    results: tuple[ValidatedResult, ...]
    ignored_incomplete_ids: tuple[int, ...]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaign_dir",
        type=Path,
        help="Existing frozen campaign that will receive validated results.",
    )
    parser.add_argument(
        "archives",
        type=Path,
        nargs="+",
        help="One or more cumulative staged campaign tar archives.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the complete preflight and report the plan without writing.",
    )
    return parser.parse_args(argv)


def _normalise_archive_path(name: str) -> PurePosixPath | None:
    if not name or "\0" in name or "\\" in name:
        raise ValueError(f"archive contains an unsafe path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive contains an unsafe path: {name!r}")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    return PurePosixPath(*parts) if parts else None


def _copy_member(stream: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        shutil.copyfileobj(stream, output)


def _archive_stat_signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _validated_tar_members(
    package: tarfile.TarFile,
) -> tuple[dict[PurePosixPath, tarfile.TarInfo], str]:
    members: dict[PurePosixPath, tarfile.TarInfo] = {}
    for member in package.getmembers():
        path = _normalise_archive_path(member.name)
        if path is None:
            if not member.isdir():
                raise ValueError(
                    f"archive root marker is not a directory: {member.name!r}"
                )
            continue
        if not member.isdir() and not member.isfile():
            raise ValueError(
                f"archive contains a forbidden special member: {member.name!r}"
            )
        if path in members:
            raise ValueError(f"archive contains a duplicate path: {path}")
        members[path] = member

    if not members:
        raise ValueError("staged archive is empty")
    roots = {path.parts[0] for path in members}
    if len(roots) != 1:
        raise ValueError(
            "staged archive must contain exactly one root directory; found "
            + ", ".join(sorted(roots))
        )
    root_name = next(iter(roots))
    root_member = members.get(PurePosixPath(root_name))
    if root_member is not None and not root_member.isdir():
        raise ValueError("staged archive root must be a directory")

    for path in members:
        for length in range(1, len(path.parts)):
            parent = PurePosixPath(*path.parts[:length])
            parent_member = members.get(parent)
            if parent_member is not None and not parent_member.isdir():
                raise ValueError(
                    f"archive file is used as a parent directory: {parent}"
                )
    return members, root_name


def _safe_extract_archive(archive: Path, destination: Path) -> tuple[Path, str]:
    """Hash and extract one stable archive inode without trusting tar paths."""

    try:
        archive_stream = archive.open("rb")
    except OSError as error:
        raise ValueError(f"cannot read staged archive {archive}: {error}") from error

    with archive_stream:
        opened_stat = os.fstat(archive_stream.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: archive_stream.read(1024 * 1024), b""):
            digest.update(chunk)
        archive_sha256 = digest.hexdigest()
        archive_stream.seek(0)
        try:
            package = tarfile.open(fileobj=archive_stream, mode="r:*")  # noqa: SIM115
        except (OSError, tarfile.TarError) as error:
            raise ValueError(
                f"cannot read staged archive {archive}: {error}"
            ) from error

        with package:
            members, root_name = _validated_tar_members(package)
            destination.mkdir(parents=True)
            for path, member in members.items():
                target = destination.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                stream = package.extractfile(member)
                if stream is None:
                    raise ValueError(
                        f"cannot read regular archive member: {member.name!r}"
                    )
                with stream:
                    _copy_member(stream, target)

        closed_stat = os.fstat(archive_stream.fileno())
        try:
            path_stat = archive.stat()
        except OSError as error:
            raise ValueError(
                f"staged archive disappeared during preflight: {archive}"
            ) from error
        if _archive_stat_signature(opened_stat) != _archive_stat_signature(
            closed_stat
        ) or _archive_stat_signature(closed_stat) != _archive_stat_signature(path_stat):
            raise ValueError(f"staged archive changed during preflight: {archive}")
    return destination / root_name, archive_sha256


def _campaign_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError(f"campaign has an invalid {label} path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"campaign {label} path escapes its root")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts:
        raise ValueError(f"campaign has an empty {label} path")
    return root.joinpath(*parts)


def _immutable_hashes(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    catalogue_metadata = manifest.get("catalogue")
    psd_metadata = manifest.get("psd")
    if not isinstance(catalogue_metadata, Mapping) or not isinstance(
        psd_metadata, Mapping
    ):
        raise TypeError("campaign manifest has invalid immutable-file metadata")
    psd_files = psd_metadata.get("files")
    if not isinstance(psd_files, Mapping):
        raise TypeError("campaign manifest has no PSD file inventory")

    relative_paths: list[tuple[str, object]] = [
        ("manifest.json", "manifest.json"),
        ("catalogue", catalogue_metadata.get("path")),
        *[(f"PSD {relative}", relative) for relative in psd_files],
    ]
    hashes: dict[str, str] = {}
    for label, relative in relative_paths:
        path = _campaign_file(root, relative, label=label)
        if not path.is_file():
            raise ValueError(f"campaign is missing immutable file: {path}")
        hashes[path.relative_to(root).as_posix()] = file_sha256(path)
    return hashes


def _validate_immutable_campaign(
    root: Path,
    *,
    destination_manifest: Mapping[str, Any],
    destination_hashes: Mapping[str, str],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    expected_manifest_sha = destination_hashes["manifest.json"]
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != expected_manifest_sha
    ):
        raise ValueError(
            f"staged manifest differs from destination campaign: {manifest_path}"
        )
    manifest = load_manifest(root)
    if manifest.get("config_sha256") != destination_manifest.get("config_sha256"):
        raise ValueError(f"staged config differs from destination campaign: {root}")
    hashes = _immutable_hashes(root, manifest)
    if hashes != dict(destination_hashes):
        differing = sorted(
            key
            for key in set(hashes) | set(destination_hashes)
            if hashes.get(key) != destination_hashes.get(key)
        )
        raise ValueError(
            "staged immutable files differ from destination campaign: "
            + ", ".join(differing)
        )
    return manifest


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid result summary {path}: {error}") from error
    if not isinstance(summary, dict):
        raise TypeError(f"result summary is not a JSON object: {path}")
    return summary


def _exact_int(value: object, *, field: str, result_label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{result_label}: {field} must be an exact integer")
    return value


def _finite_float(
    value: object,
    *,
    field: str,
    result_label: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{result_label}: {field} must be finite and numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{result_label}: {field} must be finite and numeric"
        ) from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{result_label}: {field} must be {qualifier}")
    return result


def _result_fingerprint(directory: Path) -> tuple[str, dict[str, str]]:
    file_hashes = {
        path.relative_to(directory).as_posix(): file_sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    digest = hashlib.sha256()
    for relative, sha256 in file_hashes.items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), file_hashes


def _validate_destination_mutable_paths(destination: Path) -> None:
    """Reject mutable campaign paths that could redirect writes outside the root."""

    results_root = destination / "results"
    if results_root.is_symlink() or (
        results_root.exists() and not results_root.is_dir()
    ):
        raise ValueError(
            f"destination results path must be a real directory: {results_root}"
        )

    status_path = destination / "status.csv"
    if status_path.is_symlink() or (status_path.exists() and not status_path.is_file()):
        raise ValueError(
            f"destination status path must be a regular file: {status_path}"
        )


def _regular_result_entries(directory: Path, *, result_label: str) -> list[Path]:
    """Return a flat result directory inventory without following symlinks."""

    entries = list(directory.iterdir())
    non_regular = sorted(
        entry.name for entry in entries if entry.is_symlink() or not entry.is_file()
    )
    if non_regular:
        raise ValueError(
            f"{result_label}: result entries must be regular files: {non_regular}"
        )
    return entries


def _validate_device_inventory(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    result_label: str,
) -> None:
    if summary.get("simulated_cpu") is not False:
        raise ValueError(f"{result_label}: simulated/non-GPU result is not scientific")
    config = manifest.get("config")
    devices = summary.get("devices")
    if not isinstance(config, Mapping) or not isinstance(devices, Mapping):
        raise TypeError(f"{result_label}: missing device inventory")
    requested = _exact_int(
        config.get("n_devices"), field="config.n_devices", result_label=result_label
    )
    if (
        devices.get("backend") != "gpu"
        or devices.get("requested_count") != requested
        or devices.get("local_count") != requested
    ):
        raise ValueError(f"{result_label}: GPU device count does not match campaign")
    inventory = devices.get("devices")
    if (
        not isinstance(inventory, list)
        or len(inventory) != requested
        or any(
            not isinstance(device, Mapping) or device.get("platform") != "gpu"
            for device in inventory
        )
    ):
        raise ValueError(f"{result_label}: invalid GPU device inventory")


def _validate_truth_and_seeds(
    summary: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    result_label: str,
) -> None:
    expected_parameters = (*PARAMETERS, *MARGINALIZED_PARAMETERS)
    summary_truth = summary.get("truth")
    if not isinstance(summary_truth, Mapping) or set(summary_truth) != set(
        expected_parameters
    ):
        raise ValueError(f"{result_label}: truth parameter inventory is invalid")
    for name in expected_parameters:
        value = _finite_float(
            summary_truth[name], field=f"truth.{name}", result_label=result_label
        )
        if value != truth[name]:
            raise ValueError(f"{result_label}: truth mismatch for {name}")
    expected_seeds = {
        "noise": truth["noise_seed"],
        "sampler": truth["sampler_seed"],
    }
    if summary.get("seeds") != expected_seeds:
        raise ValueError(f"{result_label}: seeds do not match the catalogue")


def _quotient_fold_enabled(
    config: Mapping[str, Any],
    *,
    result_label: str,
) -> bool:
    fold = config.get("fold_symmetry")
    if fold is None:
        return False
    if not isinstance(fold, Mapping):
        raise TypeError(f"{result_label}: fold_symmetry config is invalid")
    return True


def _load_posterior_arrays(
    posterior_path: Path,
    posterior_metadata: Mapping[str, Any],
    sampled_parameters: Sequence[str],
    *,
    quotient_fold: bool,
    result_label: str,
) -> dict[str, np.ndarray[Any, Any]]:
    fields = posterior_metadata.get("fields")
    required_fields = set(sampled_parameters) | {"log_likelihood", "log_weights"}
    allowed_fields = (
        (required_fields,)
        if quotient_fold
        else (required_fields, required_fields | {"log_likelihood_birth"})
    )
    if (
        not isinstance(fields, list)
        or len(fields) != len(set(fields))
        or set(fields) not in allowed_fields
    ):
        raise ValueError(f"{result_label}: posterior field inventory is invalid")
    try:
        with np.load(posterior_path, allow_pickle=False) as posterior:
            if posterior.files != fields:
                raise ValueError(
                    f"{result_label}: NPZ fields do not match summary inventory"
                )
            arrays = {name: np.asarray(posterior[name]) for name in fields}
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        if str(error).startswith(f"{result_label}:"):
            raise
        raise ValueError(
            f"{result_label}: unreadable posterior NPZ: {error}"
        ) from error
    return arrays


def _load_folded_nested_diagnostics(
    directory: Path,
    summary: Mapping[str, Any],
    *,
    result_label: str,
) -> dict[str, np.ndarray[Any, Any]]:
    metadata = summary.get("folded_nested_diagnostics")
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{result_label}: missing folded nested-diagnostic metadata")
    if metadata.get("path") != "folded_nested_diagnostics.npz":
        raise ValueError(f"{result_label}: folded nested-diagnostic path is invalid")
    if metadata.get("semantics") != _FOLDED_TARGET_SEMANTICS:
        raise ValueError(
            f"{result_label}: folded nested-diagnostic semantics are invalid"
        )
    fields = metadata.get("fields")
    required_fields = {"log_likelihood", "log_likelihood_birth"}
    if (
        not isinstance(fields, list)
        or len(fields) != len(set(fields))
        or set(fields) != required_fields
    ):
        raise ValueError(
            f"{result_label}: folded nested-diagnostic field inventory is invalid"
        )

    path = directory / "folded_nested_diagnostics.npz"
    if not path.is_file():
        raise ValueError(
            f"{result_label}: folded nested-diagnostic artifact is missing"
        )
    expected_sha256 = metadata.get("sha256")
    if not isinstance(expected_sha256, str) or file_sha256(path) != expected_sha256:
        raise ValueError(f"{result_label}: folded nested-diagnostic hash mismatch")
    expected_bytes = metadata.get("bytes")
    if type(expected_bytes) is not int or expected_bytes != path.stat().st_size:
        raise ValueError(
            f"{result_label}: folded nested-diagnostic byte count mismatch"
        )
    try:
        with np.load(path, allow_pickle=False) as payload:
            if payload.files != fields:
                raise ValueError(
                    f"{result_label}: folded diagnostic NPZ fields do not match "
                    "summary inventory"
                )
            arrays = {name: np.asarray(payload[name]) for name in fields}
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        if str(error).startswith(f"{result_label}:"):
            raise
        raise ValueError(
            f"{result_label}: unreadable folded diagnostic NPZ: {error}"
        ) from error

    death = arrays["log_likelihood"]
    birth = arrays["log_likelihood_birth"]
    if death.ndim != 1 or birth.shape != death.shape or death.size < 1:
        raise ValueError(f"{result_label}: folded likelihood arrays are not aligned")
    for name, values in arrays.items():
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise ValueError(f"{result_label}: folded {name} is not a real array")
    if not np.all(np.isfinite(death)):
        raise ValueError(
            f"{result_label}: folded log_likelihood contains non-finite values"
        )
    if np.any(np.isnan(birth)) or np.any(np.isposinf(birth)):
        raise ValueError(
            f"{result_label}: folded log_likelihood_birth contains NaN or +inf"
        )
    replacement = np.isfinite(birth)
    if not np.any(replacement):
        raise ValueError(
            f"{result_label}: folded log_likelihood_birth has no replacement points"
        )
    if np.any(death[replacement] <= birth[replacement]):
        raise ValueError(
            f"{result_label}: folded replacement likelihood does not exceed its birth"
        )
    return arrays


def _validate_insertion_index_diagnostic(
    arrays: Mapping[str, np.ndarray[Any, Any]],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    require_evidence: bool,
    result_label: str,
) -> None:
    diagnostics = summary.get("diagnostics")
    stored = (
        diagnostics.get("insertion_index") if isinstance(diagnostics, Mapping) else None
    )
    if "log_likelihood_birth" not in arrays:
        if stored is not None:
            raise ValueError(
                f"{result_label}: insertion-index diagnostic requires "
                "log_likelihood_birth"
            )
        if require_evidence:
            raise ValueError(
                f"{result_label}: remediation result requires "
                "log_likelihood_birth and an insertion-index diagnostic"
            )
        return
    if not isinstance(stored, Mapping):
        raise TypeError(f"{result_label}: missing insertion-index diagnostic")

    n_live = _exact_int(
        config.get("n_live"),
        field="config.n_live",
        result_label=result_label,
    )
    from jimgw.samplers.diagnostics import insertion_index_diagnostic

    expected = insertion_index_diagnostic(
        arrays["log_likelihood"],
        arrays["log_likelihood_birth"],
        n_live=n_live,
    )
    if set(stored) != set(expected):
        raise ValueError(
            f"{result_label}: insertion-index diagnostic inventory is invalid"
        )
    for field, expected_value in expected.items():
        actual_value = stored[field]
        matches = False
        if type(expected_value) is int:
            matches = (
                _exact_int(
                    actual_value,
                    field=f"diagnostics.insertion_index.{field}",
                    result_label=result_label,
                )
                == expected_value
            )
        elif isinstance(expected_value, float):
            matches = math.isclose(
                _finite_float(
                    actual_value,
                    field=f"diagnostics.insertion_index.{field}",
                    result_label=result_label,
                ),
                expected_value,
                rel_tol=_INSERTION_DIAGNOSTIC_TOLERANCE,
                abs_tol=_INSERTION_DIAGNOSTIC_TOLERANCE,
            )
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(
                f"{result_label}: insertion-index diagnostic mismatch for {field}"
            )


def _validate_arrays_and_ranks(
    arrays: Mapping[str, np.ndarray[Any, Any]],
    insertion_arrays: Mapping[str, np.ndarray[Any, Any]],
    summary: Mapping[str, Any],
    truth: Mapping[str, Any],
    config: Mapping[str, Any],
    sampled_parameters: Sequence[str],
    *,
    quotient_fold: bool,
    require_insertion_evidence: bool,
    result_label: str,
) -> tuple[int, float]:
    sample_count = _exact_int(
        summary.get("posterior_samples"),
        field="posterior_samples",
        result_label=result_label,
    )
    if sample_count < 1:
        raise ValueError(f"{result_label}: posterior_samples must be positive")
    for name, values in arrays.items():
        if values.shape != (sample_count,):
            raise ValueError(
                f"{result_label}: {name} does not have shape ({sample_count},)"
            )
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise ValueError(f"{result_label}: {name} is not a real numeric array")

    for name in (*sampled_parameters, "log_likelihood"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{result_label}: {name} contains non-finite values")
    if "log_likelihood_birth" in arrays:
        birth = arrays["log_likelihood_birth"]
        if np.any(np.isnan(birth)) or np.any(np.isposinf(birth)):
            raise ValueError(
                f"{result_label}: log_likelihood_birth contains NaN or +inf"
            )
        replacement = np.isfinite(birth)
        if not np.any(replacement):
            raise ValueError(
                f"{result_label}: log_likelihood_birth has no replacement points"
            )
        if np.any(arrays["log_likelihood"][replacement] <= birth[replacement]):
            raise ValueError(
                f"{result_label}: replacement likelihood does not exceed its birth"
            )
    _validate_insertion_index_diagnostic(
        insertion_arrays,
        summary,
        config,
        require_evidence=require_insertion_evidence,
        result_label=result_label,
    )
    log_weights = arrays["log_weights"]
    if (
        np.any(np.isnan(log_weights))
        or np.any(np.isposinf(log_weights))
        or not np.any(np.isfinite(log_weights))
    ):
        raise ValueError(f"{result_label}: log_weights are invalid")
    finite_weights = log_weights[np.isfinite(log_weights)]
    maximum = float(np.max(finite_weights))
    log_normalizer = maximum + math.log(float(np.sum(np.exp(finite_weights - maximum))))
    if not math.isclose(
        log_normalizer,
        0.0,
        rel_tol=0.0,
        abs_tol=_LOG_WEIGHT_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(
            f"{result_label}: log_weights are not normalized; "
            f"logsumexp={log_normalizer}"
        )

    weights = np.exp(log_weights)
    expected_ess = 1.0 / float(np.sum(weights * weights))
    ess_field = (
        "posterior_weight_effective_size"
        if quotient_fold
        else "posterior_effective_sample_size"
    )
    if quotient_fold and "posterior_effective_sample_size" in summary:
        raise ValueError(
            f"{result_label}: unfolded posterior must use {ess_field}, not "
            "posterior_effective_sample_size"
        )
    stored_ess = _finite_float(
        summary.get(ess_field),
        field=ess_field,
        result_label=result_label,
        positive=True,
    )
    if not math.isclose(stored_ess, expected_ess, rel_tol=1.0e-12, abs_tol=1.0e-10):
        raise ValueError(f"{result_label}: {ess_field} mismatch")

    expected_rank_method = _UNFOLDED_RANK_METHOD if quotient_fold else _RANK_METHOD
    if summary.get("rank_method") != expected_rank_method:
        raise ValueError(f"{result_label}: posterior rank method is invalid")
    ranks = summary.get("ranks")
    if not isinstance(ranks, Mapping) or set(ranks) != set(sampled_parameters):
        raise ValueError(f"{result_label}: posterior rank inventory is invalid")
    stored_rank_truth = summary.get("rank_truth")
    if stored_rank_truth is None:
        # Legacy campaign summaries predate explicit phase-gauge coordinates.
        rank_truth = {name: float(truth[name]) for name in sampled_parameters}
    else:
        if not isinstance(stored_rank_truth, Mapping) or set(stored_rank_truth) != set(
            sampled_parameters
        ):
            raise ValueError(f"{result_label}: rank truth inventory is invalid")
        expected_rank_truth = _rank_truth_coordinates(
            truth,
            tuple(sampled_parameters),
            phase_marginalization=bool(config.get("phase_marginalization", False)),
        )
        rank_truth = {}
        for name in sampled_parameters:
            value = _finite_float(
                stored_rank_truth[name],
                field=f"rank_truth.{name}",
                result_label=result_label,
            )
            if not math.isclose(
                value,
                expected_rank_truth[name],
                rel_tol=0.0,
                abs_tol=_RANK_TOLERANCE,
            ):
                raise ValueError(f"{result_label}: rank truth mismatch for {name}")
            rank_truth[name] = value
    for name in sampled_parameters:
        stored_rank = _finite_float(
            ranks[name], field=f"ranks.{name}", result_label=result_label
        )
        expected_rank = posterior_rank(arrays[name], rank_truth[name], log_weights)
        if not math.isclose(
            stored_rank,
            expected_rank,
            rel_tol=0.0,
            abs_tol=_RANK_TOLERANCE,
        ):
            raise ValueError(f"{result_label}: posterior rank mismatch for {name}")
    return sample_count, log_normalizer


def _validate_timing(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    result_label: str,
) -> float | None:
    timing = summary.get("timing_seconds")
    if not isinstance(timing, Mapping):
        raise TypeError(f"{result_label}: missing timing_seconds")
    sample_call = _finite_float(
        timing.get("sample_call"),
        field="timing_seconds.sample_call",
        result_label=result_label,
        positive=True,
    )
    total = _finite_float(
        timing.get("total"),
        field="timing_seconds.total",
        result_label=result_label,
        positive=True,
    )
    if total < sample_call:
        raise ValueError(f"{result_label}: total time is shorter than sample_call")

    paper = timing.get("paper_convention")
    diagnostic = manifest.get("baseline_diagnostic")
    allow_unavailable = (
        isinstance(diagnostic, Mapping)
        and diagnostic.get("implementation_label") == "paper-baseline"
        and diagnostic.get("paper_timing_available") is False
    )
    if not isinstance(paper, Mapping):
        reason = timing.get("paper_convention_unavailable_reason")
        if not allow_unavailable or not isinstance(reason, str) or not reason.strip():
            raise TypeError(f"{result_label}: missing Figure 3 post-JIT timing")
        if timing.get("sample_phases") is not None:
            raise TypeError(
                f"{result_label}: baseline timing must not claim sampler phase data"
            )
        return None

    likelihood_jit = _finite_float(
        paper.get("likelihood_jit_seconds"),
        field="paper_convention.likelihood_jit_seconds",
        result_label=result_label,
        positive=True,
    )
    sampler_jit = _finite_float(
        paper.get("sampler_jit_seconds"),
        field="paper_convention.sampler_jit_seconds",
        result_label=result_label,
        positive=True,
    )
    post_jit = _finite_float(
        paper.get("post_jit_sampling_seconds"),
        field="paper_convention.post_jit_sampling_seconds",
        result_label=result_label,
        positive=True,
    )
    expected = sample_call - likelihood_jit - sampler_jit
    tolerance = max(1.0e-9, abs(sample_call) * 1.0e-9)
    if not math.isclose(post_jit, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{result_label}: inconsistent post-JIT timing arithmetic")
    sample_phases = timing.get("sample_phases")
    if not isinstance(sample_phases, Mapping):
        raise TypeError(f"{result_label}: missing sampler phase timings")
    phase_likelihood = _finite_float(
        sample_phases.get("likelihood_jit"),
        field="sample_phases.likelihood_jit",
        result_label=result_label,
        positive=True,
    )
    phase_sampler = _finite_float(
        sample_phases.get("sampler_kernel_jit"),
        field="sample_phases.sampler_kernel_jit",
        result_label=result_label,
        positive=True,
    )
    if not math.isclose(
        likelihood_jit, phase_likelihood, rel_tol=0.0, abs_tol=1.0e-12
    ) or not math.isclose(sampler_jit, phase_sampler, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"{result_label}: paper timing does not match sampler phase timings"
        )
    return post_jit


def _validate_implementation(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    result_label: str,
) -> None:
    pins = [
        value
        for value in (
            manifest.get("baseline_diagnostic"),
            manifest.get("implementation_diagnostic"),
        )
        if isinstance(value, Mapping)
    ]
    if len(pins) > 1:
        raise ValueError(f"{result_label}: campaign implementation pin is ambiguous")
    if not pins:
        return
    diagnostic = pins[0]
    implementation = summary.get("implementation")
    if not isinstance(implementation, Mapping):
        raise TypeError(f"{result_label}: missing implementation provenance")
    if (
        implementation.get("label") != diagnostic.get("implementation_label")
        or implementation.get("revision") != diagnostic.get("implementation_revision")
        or implementation.get("tree_sha256")
        != diagnostic.get("implementation_tree_sha256")
    ):
        raise ValueError(f"{result_label}: implementation provenance mismatch")
    if not isinstance(implementation.get("jimgw_module"), str) or not isinstance(
        implementation.get("root"), str
    ):
        raise TypeError(f"{result_label}: invalid implementation paths")


def _requires_insertion_evidence(
    manifest: Mapping[str, Any],
    *,
    result_label: str,
) -> bool:
    """Require the production remediation's birth-contour evidence.

    Older source campaigns predate birth-likelihood storage and remain readable.
    A remediation manifest is the explicit format boundary: once the marker is
    present, every result must carry the evidence needed to recompute its
    insertion-index diagnostic.
    """

    if "blocking_remediation" not in manifest:
        return False
    if not isinstance(manifest["blocking_remediation"], Mapping):
        raise TypeError(f"{result_label}: blocking_remediation marker is invalid")
    return True


def _validate_result(
    directory: Path,
    injection_id: int,
    manifest: Mapping[str, Any],
    catalogue: Sequence[Mapping[str, Any]],
) -> ValidatedResult:
    result_label = f"injection {injection_id:03d}"
    entries = _regular_result_entries(directory, result_label=result_label)
    forbidden = sorted(
        entry.name
        for entry in entries
        if (
            entry.name
            not in {
                "summary.json",
                "posterior.npz",
                "folded_nested_diagnostics.npz",
            }
            and _ATTEMPT_LOG_RE.fullmatch(entry.name) is None
        )
    )
    if forbidden:
        raise ValueError(f"{result_label}: forbidden result entries: {forbidden}")

    summary_path = directory / "summary.json"
    posterior_path = directory / "posterior.npz"
    if not summary_path.is_file() or not posterior_path.is_file():
        raise ValueError(f"{result_label}: result is incomplete")
    summary = _load_summary(summary_path)
    if summary.get("schema_version") != manifest.get("schema_version"):
        raise ValueError(f"{result_label}: summary schema does not match campaign")
    if summary.get("config_sha256") != manifest.get("config_sha256"):
        raise ValueError(f"{result_label}: result belongs to a different campaign")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or summary.get("campaign") != config.get(
        "campaign"
    ):
        raise ValueError(f"{result_label}: result campaign name is invalid")
    quotient_fold = _quotient_fold_enabled(config, result_label=result_label)
    has_folded_artifact = (directory / "folded_nested_diagnostics.npz").is_file()
    if not quotient_fold and (
        "folded_nested_diagnostics" in summary or has_folded_artifact
    ):
        raise ValueError(
            f"{result_label}: non-folded result declares folded nested diagnostics"
        )
    if (
        _exact_int(
            summary.get("injection_id"),
            field="injection_id",
            result_label=result_label,
        )
        != injection_id
    ):
        raise ValueError(f"{result_label}: summary has the wrong injection ID")
    _validate_device_inventory(summary, manifest, result_label=result_label)
    _validate_implementation(summary, manifest, result_label=result_label)

    truth = catalogue[injection_id]
    _validate_truth_and_seeds(summary, truth, result_label=result_label)
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise TypeError(f"{result_label}: campaign has no valid configuration")
    sampled_parameters = _sampled_parameters(config)
    posterior_metadata = summary.get("posterior")
    if not isinstance(posterior_metadata, Mapping):
        raise TypeError(f"{result_label}: missing posterior metadata")
    if posterior_metadata.get("path") != "posterior.npz":
        raise ValueError(f"{result_label}: posterior path must be posterior.npz")
    expected_weighting = (
        _UNFOLDED_POSTERIOR_WEIGHTING if quotient_fold else _POSTERIOR_WEIGHTING
    )
    invalid_unfolded_metadata = quotient_fold and (
        posterior_metadata.get("schema_version") != 2
        or posterior_metadata.get("weight_effective_size_semantics")
        != _POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
    )
    if (
        posterior_metadata.get("space") != "prior"
        or posterior_metadata.get("weighting") != expected_weighting
        or invalid_unfolded_metadata
    ):
        raise ValueError(f"{result_label}: posterior semantics are invalid")
    expected_sha256 = posterior_metadata.get("sha256")
    actual_sha256 = file_sha256(posterior_path)
    if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
        raise ValueError(
            f"{result_label}: posterior hash mismatch; "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    posterior_bytes = posterior_path.stat().st_size
    if posterior_metadata.get("bytes") != posterior_bytes:
        raise ValueError(f"{result_label}: posterior byte count mismatch")

    arrays = _load_posterior_arrays(
        posterior_path,
        posterior_metadata,
        sampled_parameters,
        quotient_fold=quotient_fold,
        result_label=result_label,
    )
    insertion_arrays = (
        _load_folded_nested_diagnostics(
            directory,
            summary,
            result_label=result_label,
        )
        if quotient_fold
        else arrays
    )
    require_insertion_evidence = (
        _requires_insertion_evidence(
            manifest,
            result_label=result_label,
        )
        or quotient_fold
    )
    posterior_samples, _ = _validate_arrays_and_ranks(
        arrays,
        insertion_arrays,
        summary,
        truth,
        config,
        sampled_parameters,
        quotient_fold=quotient_fold,
        require_insertion_evidence=require_insertion_evidence,
        result_label=result_label,
    )
    post_jit = _validate_timing(summary, manifest, result_label=result_label)
    fingerprint, file_hashes = _result_fingerprint(directory)
    return ValidatedResult(
        injection_id=injection_id,
        directory=directory,
        fingerprint=fingerprint,
        file_hashes=file_hashes,
        posterior_sha256=actual_sha256,
        posterior_bytes=posterior_bytes,
        posterior_samples=posterior_samples,
        post_jit_sampling_seconds=post_jit,
    )


def _collect_results(
    campaign_root: Path,
    manifest: Mapping[str, Any],
    catalogue: Sequence[Mapping[str, Any]],
) -> tuple[tuple[ValidatedResult, ...], tuple[int, ...]]:
    results_root = campaign_root / "results"
    if not results_root.exists():
        if results_root.is_symlink():
            raise ValueError(
                f"campaign results path must be a real directory: {results_root}"
            )
        return (), ()
    if results_root.is_symlink() or not results_root.is_dir():
        raise ValueError(
            f"campaign results path must be a real directory: {results_root}"
        )
    n_injections = _exact_int(
        manifest.get("n_injections"),
        field="manifest.n_injections",
        result_label="campaign",
    )
    results: list[ValidatedResult] = []
    incomplete_ids: list[int] = []
    for directory in sorted(results_root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"result path must be a real directory: {directory.name}")
        match = _RESULT_DIRECTORY_RE.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"invalid result directory name: {directory.name}")
        injection_id = int(match.group(1))
        if directory.name != f"injection-{injection_id:03d}":
            raise ValueError(f"non-canonical result directory name: {directory.name}")
        if not 0 <= injection_id < n_injections:
            raise ValueError(f"result ID is outside selected range: {injection_id}")
        _regular_result_entries(directory, result_label=f"injection {injection_id:03d}")
        if not (
            (directory / "summary.json").is_file()
            and (directory / "posterior.npz").is_file()
        ):
            incomplete_ids.append(injection_id)
            continue
        results.append(_validate_result(directory, injection_id, manifest, catalogue))
    return tuple(results), tuple(incomplete_ids)


def _same_result(first: ValidatedResult, second: ValidatedResult) -> bool:
    return (
        first.fingerprint == second.fingerprint
        and first.file_hashes == second.file_hashes
    )


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        status: sum(row.get("status") == status for row in rows)
        for status in ("complete", "failed", "pending", "running", "invalid")
    }


def _archive_report(archive: ValidatedArchive) -> dict[str, Any]:
    return {
        "path": str(archive.path),
        "sha256": archive.sha256,
        "root": archive.root.name,
        "complete_ids": [result.injection_id for result in archive.results],
        "ignored_incomplete_ids": list(archive.ignored_incomplete_ids),
        "results": [
            {
                "injection_id": result.injection_id,
                "fingerprint": result.fingerprint,
                "posterior_sha256": result.posterior_sha256,
                "posterior_bytes": result.posterior_bytes,
                "posterior_samples": result.posterior_samples,
                "post_jit_sampling_seconds": result.post_jit_sampling_seconds,
            }
            for result in archive.results
        ],
    }


def _stage_copies(
    destination: Path,
    candidates: Mapping[int, ValidatedResult],
    copy_ids: Sequence[int],
) -> None:
    if not copy_ids:
        return
    results_root = destination / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    if results_root.is_symlink() or not results_root.is_dir():
        raise RuntimeError(
            f"destination results path must be a real directory: {results_root}"
        )
    staging_root = Path(
        tempfile.mkdtemp(prefix=".merge-staged-results-", dir=results_root)
    )
    try:
        for injection_id in copy_ids:
            candidate = candidates[injection_id]
            staged = staging_root / f"injection-{injection_id:03d}"
            shutil.copytree(candidate.directory, staged)
            fingerprint, file_hashes = _result_fingerprint(staged)
            if (
                fingerprint != candidate.fingerprint
                or file_hashes != candidate.file_hashes
            ):
                raise RuntimeError(
                    f"staged copy verification failed for injection {injection_id:03d}"
                )
        appeared = [
            injection_id
            for injection_id in copy_ids
            if (results_root / f"injection-{injection_id:03d}").exists()
        ]
        if appeared:
            rendered = ", ".join(f"{value:03d}" for value in appeared)
            raise RuntimeError(
                "destination changed after preflight; refusing commit for IDs: "
                + rendered
            )
        for injection_id in copy_ids:
            staged = staging_root / f"injection-{injection_id:03d}"
            staged.rename(results_root / staged.name)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def merge_staged_results(
    campaign_dir: Path,
    archives: Sequence[Path],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate all inputs, then merge new result directories and refresh status."""

    destination = campaign_dir.expanduser().resolve()
    if not destination.is_dir():
        raise ValueError(f"destination campaign does not exist: {destination}")
    if not archives:
        raise ValueError("at least one staged archive is required")
    resolved_archives = [path.expanduser().resolve() for path in archives]
    missing = [path for path in resolved_archives if not path.is_file()]
    if missing:
        raise ValueError(f"staged archive does not exist: {missing[0]}")

    destination_manifest = load_manifest(destination)
    _validate_destination_mutable_paths(destination)
    destination_hashes = _immutable_hashes(destination, destination_manifest)
    catalogue_path = _campaign_file(
        destination,
        destination_manifest["catalogue"]["path"],
        label="catalogue",
    )
    catalogue = read_catalogue(catalogue_path)
    n_injections = _exact_int(
        destination_manifest.get("n_injections"),
        field="manifest.n_injections",
        result_label="campaign",
    )
    if n_injections < 1 or len(catalogue) < n_injections:
        raise ValueError("campaign selection is inconsistent with its catalogue")

    existing_status = status_rows(destination, n_injections)
    status_before = _status_counts(existing_status)
    if status_before["running"]:
        raise ValueError("destination campaign has running injections")
    existing_results, _ = _collect_results(destination, destination_manifest, catalogue)
    existing_by_id = {result.injection_id: result for result in existing_results}

    with tempfile.TemporaryDirectory(prefix="jim-staged-results-") as temporary:
        extraction_parent = Path(temporary)
        validated_archives: list[ValidatedArchive] = []
        for index, archive_path in enumerate(resolved_archives):
            extracted_root, archive_sha256 = _safe_extract_archive(
                archive_path, extraction_parent / f"archive-{index:03d}"
            )
            manifest = _validate_immutable_campaign(
                extracted_root,
                destination_manifest=destination_manifest,
                destination_hashes=destination_hashes,
            )
            results, incomplete_ids = _collect_results(
                extracted_root, manifest, catalogue
            )
            validated_archives.append(
                ValidatedArchive(
                    path=archive_path,
                    sha256=archive_sha256,
                    root=extracted_root,
                    results=results,
                    ignored_incomplete_ids=incomplete_ids,
                )
            )

        candidates: dict[int, ValidatedResult] = {}
        duplicate_archive_ids: list[int] = []
        for archive in validated_archives:
            for result in archive.results:
                previous = candidates.get(result.injection_id)
                if previous is None:
                    candidates[result.injection_id] = result
                elif _same_result(previous, result):
                    duplicate_archive_ids.append(result.injection_id)
                else:
                    raise ValueError(
                        "cross-archive collision for injection "
                        f"{result.injection_id:03d}: {previous.fingerprint} != "
                        f"{result.fingerprint}"
                    )
        if not candidates:
            raise ValueError("staged archives contain no complete results")

        copy_ids: list[int] = []
        skipped_existing_ids: list[int] = []
        for injection_id, candidate in sorted(candidates.items()):
            existing = existing_by_id.get(injection_id)
            target = destination / "results" / f"injection-{injection_id:03d}"
            if existing is not None:
                if not _same_result(existing, candidate):
                    raise ValueError(
                        "destination collision for injection "
                        f"{injection_id:03d}: {existing.fingerprint} != "
                        f"{candidate.fingerprint}"
                    )
                skipped_existing_ids.append(injection_id)
            elif target.exists():
                raise ValueError(
                    f"destination has an incomplete collision for injection "
                    f"{injection_id:03d}"
                )
            else:
                copy_ids.append(injection_id)

        if _immutable_hashes(destination, destination_manifest) != destination_hashes:
            raise RuntimeError("destination immutable files changed during preflight")
        _validate_destination_mutable_paths(destination)

        report: dict[str, Any] = {
            "campaign_dir": str(destination),
            "config_sha256": destination_manifest["config_sha256"],
            "dry_run": dry_run,
            "immutable_sha256": destination_hashes,
            "archives": [_archive_report(archive) for archive in validated_archives],
            "candidate_ids": sorted(candidates),
            "copy_ids": copy_ids,
            "skipped_existing_ids": skipped_existing_ids,
            "duplicate_archive_ids": duplicate_archive_ids,
            "existing_complete_ids_before": sorted(existing_by_id),
            "status_before": status_before,
            "status_regenerated": False,
        }
        if dry_run:
            report["projected_complete_ids"] = sorted(
                set(existing_by_id) | set(candidates)
            )
            return report

        _stage_copies(destination, candidates, copy_ids)

    merged_results, _ = _collect_results(destination, destination_manifest, catalogue)
    merged_by_id = {result.injection_id: result for result in merged_results}
    expected_complete_ids = sorted(set(existing_by_id) | set(candidates))
    if sorted(merged_by_id) != expected_complete_ids:
        raise RuntimeError("post-merge complete result set does not match preflight")
    for injection_id, candidate in candidates.items():
        if not _same_result(candidate, merged_by_id[injection_id]):
            raise RuntimeError(
                f"post-merge verification failed for injection {injection_id:03d}"
            )

    _validate_destination_mutable_paths(destination)
    refreshed = refresh_status(destination, n_injections)
    complete_ids = [
        int(row["injection_id"]) for row in refreshed if row.get("status") == "complete"
    ]
    if complete_ids != expected_complete_ids:
        raise RuntimeError("regenerated status does not match validated results")
    report.update(
        {
            "complete_ids_after": complete_ids,
            "status_after": _status_counts(refreshed),
            "status_regenerated": True,
        }
    )
    return report


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        report = merge_staged_results(
            args.campaign_dir, args.archives, dry_run=args.dry_run
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"refusing staged result merge: {error}") from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
