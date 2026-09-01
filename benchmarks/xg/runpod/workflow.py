"""Immutable packaging and artifact verification for the CE XG Runpod job."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
PACKAGE_KIND = "jim-xg-runpod"
RESULT_KIND = "jim-xg-runpod-result"
CAPTURE_DESCRIPTION = (
    "git archive of committed HEAD plus verified CE PSD and DE405 ephemerides"
)
IMAGE = (
    "runpod/pytorch@"
    "sha256:60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f"
)
IMAGE_SHA256 = "60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f"
GPU_ID = "NVIDIA H200"
GPU_COUNT = 4
GUARD_MINUTES = 120
MAX_GPU_COST_USD = 40.0
MIN_BALANCE_RESERVE_USD = 10.0
PYTHON_VERSION = "3.12"
UV_VERSION = "0.11.2"

CONFIG_PATH = PurePosixPath("benchmarks/xg/xg-ce-4096-65536.toml")
QUALIFICATION_ENTRYPOINT = PurePosixPath("benchmarks/xg/run_qualification.py")
PSD_PATH = PurePosixPath("benchmark-results/xg-ce-4096-65536/inputs/CE_psd.txt")
PSD_SHA256 = "a8934610ae6395a86129a70bf913d2b3469a477f872fec18c626b49dc7aa3f49"
EARTH_EPHEMERIS_PATH = PurePosixPath(
    "benchmark-results/xg-ce-4096-65536/inputs/earth00-40-DE405.dat.gz"
)
EARTH_EPHEMERIS_SHA256 = (
    "4995647b2c47617c90804ad0bc814ce42b426f1e5015a90cf939bcdd0c20ea67"
)
SUN_EPHEMERIS_PATH = PurePosixPath(
    "benchmark-results/xg-ce-4096-65536/inputs/sun00-40-DE405.dat.gz"
)
SUN_EPHEMERIS_SHA256 = (
    "0b132dc5a712ebc16661a10cb88409e2577c16723c64f98e9b9b4d265510700f"
)
VERIFIED_INPUT_SHA256 = {
    PSD_PATH: PSD_SHA256,
    EARTH_EPHEMERIS_PATH: EARTH_EPHEMERIS_SHA256,
    SUN_EPHEMERIS_PATH: SUN_EPHEMERIS_SHA256,
}
PACKAGE_MANIFEST_PATH = PurePosixPath(".runpod/package-manifest.json")
RESULT_MANIFEST_PATH = PurePosixPath(".runpod/artifact-manifest.json")
EXIT_STATUS_PATH = PurePosixPath(".runpod/exit-status")
COMPLETE_PATH = PurePosixPath(".runpod/complete")

REQUIRED_TRACKED_PATHS = frozenset(
    {
        PurePosixPath("LICENSE"),
        PurePosixPath("README.md"),
        PurePosixPath("pyproject.toml"),
        PurePosixPath("uv.lock"),
        CONFIG_PATH,
        QUALIFICATION_ENTRYPOINT,
        PurePosixPath("benchmarks/xg/runpod/run_on_pod.sh"),
        PurePosixPath("benchmarks/xg/runpod/provision.sh"),
        PurePosixPath("benchmarks/xg/runpod/upload_and_run.py"),
        PurePosixPath("benchmarks/xg/runpod/workflow.py"),
    }
)
MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_FILE_BYTES = 2 * 1024**3
MAX_ARCHIVE_TOTAL_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class VerifiedArchive:
    """Verified immutable archive metadata."""

    path: Path
    sha256: str
    root: PurePosixPath
    manifest: dict[str, Any]
    exit_status: int | None = None


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _safe_archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or any(not part for part in path.parts)
    ):
        raise ValueError(f"archive contains an unsafe path: {name!r}")
    if ".git" in path.parts:
        raise ValueError(f"archive contains forbidden Git metadata: {name!r}")
    return path


def _checked_members(package: tarfile.TarFile) -> dict[PurePosixPath, tarfile.TarInfo]:
    members: dict[PurePosixPath, tarfile.TarInfo] = {}
    expanded_bytes = 0
    raw_members = package.getmembers()
    if len(raw_members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("archive has too many members")
    for member in raw_members:
        path = _safe_archive_path(member.name)
        if path in members:
            raise ValueError(f"archive contains a duplicate path: {path}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"archive contains a forbidden special member: {path}")
        if not member.isfile() and not member.isdir():
            raise ValueError(f"archive contains an unsupported member: {path}")
        if member.size < 0 or member.size > MAX_ARCHIVE_FILE_BYTES:
            raise ValueError(f"archive member has an unsafe size: {path}")
        expanded_bytes += member.size
        if expanded_bytes > MAX_ARCHIVE_TOTAL_BYTES:
            raise ValueError("archive expands beyond its size limit")
        members[path] = member
    for path in members:
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            ancestor = members.get(parent)
            if ancestor is not None and not ancestor.isdir():
                raise ValueError(
                    f"archive uses a non-directory as a parent path: {parent}"
                )
    return members


def _single_archive_root(paths: Iterable[PurePosixPath]) -> PurePosixPath:
    path_set = set(paths)
    roots = {path.parts[0] for path in path_set}
    if len(roots) != 1:
        raise ValueError("archive must contain exactly one root directory")
    root = PurePosixPath(roots.pop())
    if root not in path_set:
        raise ValueError("archive must contain an explicit root directory")
    return root


def _extract_checked_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:*") as package:
        members = _checked_members(package)
        for path, member in sorted(
            members.items(), key=lambda item: item[0].as_posix()
        ):
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = package.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {path}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)


def _tree_entries(
    root: Path,
    *,
    excluded: frozenset[PurePosixPath] = frozenset(),
) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()
    ):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains a forbidden symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"artifact tree contains a special path: {path}")
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if relative in excluded:
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": digest,
                "bytes": size,
            }
        )
        aggregate.update(relative.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
    return entries, aggregate.hexdigest()


def _entries_from_archive(
    package: tarfile.TarFile,
    regular_members: Mapping[PurePosixPath, tarfile.TarInfo],
    root: PurePosixPath,
    *,
    excluded: frozenset[PurePosixPath],
) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for full_path, member in sorted(
        regular_members.items(), key=lambda item: item[0].as_posix()
    ):
        relative = PurePosixPath(*full_path.parts[1:])
        if relative in excluded:
            continue
        stream = package.extractfile(member)
        if stream is None:
            raise ValueError(f"cannot read archive member: {full_path}")
        digest = hashlib.sha256()
        with stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        hexdigest = digest.hexdigest()
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": hexdigest,
                "bytes": member.size,
            }
        )
        aggregate.update(relative.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(hexdigest))
        aggregate.update(b"\0")
        aggregate.update(str(member.size).encode("ascii"))
        aggregate.update(b"\0")
    return entries, aggregate.hexdigest()


def _write_deterministic_archive(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            output.open("xb") as raw_stream,
            gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_stream,
                mtime=0,
            ) as gzip_stream,
            tarfile.open(
                fileobj=gzip_stream,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as package,
        ):
            paths = [root, *sorted(root.rglob("*"))]
            for path in paths:
                if path.is_symlink():
                    raise ValueError(
                        f"refusing to archive a symlink: {path.relative_to(root)}"
                    )
                relative = path.relative_to(root.parent).as_posix()
                info = tarfile.TarInfo(relative)
                stat = path.stat()
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if path.is_dir():
                    info.mode = 0o755
                    info.type = tarfile.DIRTYPE
                    package.addfile(info)
                elif path.is_file():
                    info.mode = 0o755 if stat.st_mode & 0o111 else 0o644
                    info.size = stat.st_size
                    with path.open("rb") as stream:
                        package.addfile(info, stream)
                else:
                    raise ValueError(f"unsupported package path: {path}")
    except Exception:
        output.unlink(missing_ok=True)
        raise


def _git_output(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_workspace_archive(repository: Path, output: Path) -> VerifiedArchive:
    """Package committed ``HEAD`` plus the independently verified CE PSD."""

    repository = repository.resolve()
    output = output.expanduser().resolve()
    top_level = Path(_git_output(repository, "rev-parse", "--show-toplevel")).resolve()
    if top_level != repository:
        raise ValueError(f"repository must be the Git top level: {top_level}")
    revision = _git_output(repository, "rev-parse", "--verify", "HEAD^{commit}")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Git did not return a full lowercase commit revision")
    for required in sorted(REQUIRED_TRACKED_PATHS, key=str):
        probe = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "-e", f"{revision}:{required}"],
            check=False,
            capture_output=True,
        )
        if probe.returncode != 0:
            raise ValueError(f"required path is not committed at HEAD: {required}")

    verified_sources: dict[PurePosixPath, Path] = {}
    for relative, expected_digest in VERIFIED_INPUT_SHA256.items():
        source = repository.joinpath(*relative.parts)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"verified XG input is missing: {source}")
        actual_digest = sha256_file(source)
        if actual_digest != expected_digest:
            raise ValueError(
                "XG input SHA-256 mismatch: "
                f"path={relative} expected={expected_digest} observed={actual_digest}"
            )
        verified_sources[relative] = source

    committed_at = _git_output(repository, "show", "-s", "--format=%cI", revision)
    with tempfile.TemporaryDirectory(prefix="jim-xg-runpod-package-") as temporary:
        temporary_path = Path(temporary)
        git_archive = temporary_path / "committed.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                f"--output={git_archive}",
                revision,
            ],
            check=True,
        )
        extracted = temporary_path / "git"
        extracted.mkdir()
        _extract_checked_archive(git_archive, extracted)

        archive_root = temporary_path / f"jim-xg-{revision[:12]}"
        archive_root.mkdir()
        for source in extracted.iterdir():
            shutil.move(str(source), archive_root / source.name)
        for relative, source in verified_sources.items():
            destination = archive_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        entries, tree_sha256 = _tree_entries(archive_root)
        entry_by_path = {entry["path"]: entry for entry in entries}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "package": PACKAGE_KIND,
            "capture": CAPTURE_DESCRIPTION,
            "source_revision": revision,
            "source_committed_at": committed_at,
            "git_metadata_included": False,
            "image": IMAGE,
            "image_sha256": IMAGE_SHA256,
            "gpu_id": GPU_ID,
            "gpu_count": GPU_COUNT,
            "guard_minutes": GUARD_MINUTES,
            "python_version": PYTHON_VERSION,
            "uv_version": UV_VERSION,
            "config_path": CONFIG_PATH.as_posix(),
            "config_sha256": entry_by_path[CONFIG_PATH.as_posix()]["sha256"],
            "uv_lock_sha256": entry_by_path["uv.lock"]["sha256"],
            "qualification_entrypoint": QUALIFICATION_ENTRYPOINT.as_posix(),
            "psd_path": PSD_PATH.as_posix(),
            "psd_sha256": PSD_SHA256,
            "verified_inputs": [
                {
                    "path": path.as_posix(),
                    "sha256": digest,
                }
                for path, digest in sorted(
                    VERIFIED_INPUT_SHA256.items(),
                    key=lambda item: item[0].as_posix(),
                )
            ],
            "payload_tree_sha256": tree_sha256,
            "file_count": len(entries),
            "files": entries,
        }
        manifest_path = archive_root.joinpath(*PACKAGE_MANIFEST_PATH.parts)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        _write_deterministic_archive(archive_root, output)

    return verify_workspace_archive(output)


def _load_manifest_member(
    package: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    label: str,
) -> dict[str, Any]:
    stream = package.extractfile(member)
    if stream is None:
        raise ValueError(f"cannot read {label}")
    with stream:
        payload = stream.read()
    if len(payload) > 20_000_000:
        raise ValueError(f"{label} is unreasonably large")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")  # noqa: TRY004
    return value


def _validate_inventory(
    manifest: Mapping[str, Any],
    actual_entries: Sequence[Mapping[str, Any]],
    actual_tree_sha256: str,
) -> None:
    advertised = manifest.get("files")
    if not isinstance(advertised, list) or advertised != list(actual_entries):
        raise ValueError("archive file inventory does not match its manifest")
    if manifest.get("file_count") != len(actual_entries):
        raise ValueError("archive file count does not match its manifest")
    if manifest.get("payload_tree_sha256") != actual_tree_sha256:
        raise ValueError("archive payload tree SHA-256 mismatch")


def _validate_workspace_manifest(
    manifest: Mapping[str, Any],
    *,
    label: str,
) -> str:
    expected_constants = {
        "schema_version": SCHEMA_VERSION,
        "package": PACKAGE_KIND,
        "capture": CAPTURE_DESCRIPTION,
        "git_metadata_included": False,
        "image": IMAGE,
        "image_sha256": IMAGE_SHA256,
        "gpu_id": GPU_ID,
        "gpu_count": GPU_COUNT,
        "guard_minutes": GUARD_MINUTES,
        "python_version": PYTHON_VERSION,
        "uv_version": UV_VERSION,
        "config_path": CONFIG_PATH.as_posix(),
        "qualification_entrypoint": QUALIFICATION_ENTRYPOINT.as_posix(),
        "psd_path": PSD_PATH.as_posix(),
        "psd_sha256": PSD_SHA256,
        "verified_inputs": [
            {
                "path": path.as_posix(),
                "sha256": digest,
            }
            for path, digest in sorted(
                VERIFIED_INPUT_SHA256.items(),
                key=lambda item: item[0].as_posix(),
            )
        ],
    }
    for name, expected in expected_constants.items():
        if manifest.get(name) != expected:
            raise ValueError(f"{label} has invalid {name}")
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"{label} has an invalid source revision")
    for name in (
        "config_sha256",
        "uv_lock_sha256",
        "payload_tree_sha256",
    ):
        digest = manifest.get(name)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{label} has an invalid {name}")
    return revision


def verify_workspace_archive(archive: Path) -> VerifiedArchive:
    """Verify safety, provenance, inventory, and frozen XG launch constants."""

    archive = archive.expanduser().resolve()
    try:
        package = tarfile.open(archive, mode="r:*")  # noqa: SIM115
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"cannot read workspace archive {archive}: {error}") from error
    with package:
        members = _checked_members(package)
        root = _single_archive_root(members)
        regular = {path: member for path, member in members.items() if member.isfile()}
        manifest_member = regular.get(root / PACKAGE_MANIFEST_PATH)
        if manifest_member is None:
            raise ValueError("workspace archive has no package manifest")
        manifest = _load_manifest_member(
            package, manifest_member, label="workspace package manifest"
        )
        _validate_workspace_manifest(manifest, label="workspace manifest")

        actual_entries, tree_sha256 = _entries_from_archive(
            package,
            regular,
            root,
            excluded=frozenset({PACKAGE_MANIFEST_PATH}),
        )
        _validate_inventory(manifest, actual_entries, tree_sha256)
        entries = {entry["path"]: entry for entry in actual_entries}
        required = REQUIRED_TRACKED_PATHS | set(VERIFIED_INPUT_SHA256)
        missing = sorted(
            path.as_posix() for path in required if path.as_posix() not in entries
        )
        if missing:
            raise ValueError(f"workspace archive is missing required files: {missing}")
        for path, expected_digest in VERIFIED_INPUT_SHA256.items():
            if entries[path.as_posix()]["sha256"] != expected_digest:
                raise ValueError(
                    f"workspace archive contains the wrong XG input: {path}"
                )
        if entries[CONFIG_PATH.as_posix()]["sha256"] != manifest.get("config_sha256"):
            raise ValueError("workspace config digest does not match its manifest")
        if entries["uv.lock"]["sha256"] != manifest.get("uv_lock_sha256"):
            raise ValueError("workspace uv.lock digest does not match its manifest")

    return VerifiedArchive(
        path=archive,
        sha256=sha256_file(archive),
        root=root,
        manifest=manifest,
    )


def verify_extracted_workspace(root: Path) -> dict[str, Any]:
    """Verify an extracted package again before installing or executing it."""

    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"workspace root is not a regular directory: {root}")
    manifest_path = root.joinpath(*PACKAGE_MANIFEST_PATH.parts)
    try:
        manifest_value = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load extracted package manifest: {error}") from error
    if not isinstance(manifest_value, dict):
        raise ValueError(  # noqa: TRY004
            "extracted package manifest must be a JSON object"
        )
    manifest: dict[str, Any] = manifest_value
    _validate_workspace_manifest(manifest, label="extracted workspace manifest")
    entries, tree_sha256 = _tree_entries(
        root, excluded=frozenset({PACKAGE_MANIFEST_PATH})
    )
    _validate_inventory(manifest, entries, tree_sha256)
    entry_by_path = {entry["path"]: entry for entry in entries}
    missing = sorted(
        path.as_posix()
        for path in REQUIRED_TRACKED_PATHS | set(VERIFIED_INPUT_SHA256)
        if path.as_posix() not in entry_by_path
    )
    if missing:
        raise ValueError(f"extracted workspace is missing required files: {missing}")
    for path, expected_digest in VERIFIED_INPUT_SHA256.items():
        if entry_by_path[path.as_posix()]["sha256"] != expected_digest:
            raise ValueError(f"extracted workspace contains the wrong XG input: {path}")
    if entry_by_path[CONFIG_PATH.as_posix()]["sha256"] != manifest.get("config_sha256"):
        raise ValueError("extracted workspace config digest mismatch")
    if entry_by_path["uv.lock"]["sha256"] != manifest.get("uv_lock_sha256"):
        raise ValueError("extracted workspace uv.lock digest mismatch")
    return manifest


def write_result_manifest(root: Path, exit_status: int) -> Path:
    """Write the content-addressed result inventory used before remote archival."""

    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"result root is not a regular directory: {root}")
    if isinstance(exit_status, bool) or not 0 <= exit_status <= 255:
        raise ValueError("exit status must be an integer from 0 through 255")
    status_path = root.joinpath(*EXIT_STATUS_PATH.parts)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(f"{exit_status}\n", encoding="ascii")
    complete_path = root.joinpath(*COMPLETE_PATH.parts)
    if exit_status == 0 and (
        not complete_path.is_file()
        or complete_path.is_symlink()
        or complete_path.stat().st_size != 0
    ):
        raise ValueError("successful result has no empty completion marker")
    package_manifest_path = root.joinpath(*PACKAGE_MANIFEST_PATH.parts)
    try:
        package_manifest_value = json.loads(package_manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load embedded workspace manifest: {error}") from error
    if not isinstance(package_manifest_value, dict):
        raise ValueError(  # noqa: TRY004
            "embedded workspace manifest must be a JSON object"
        )
    package_manifest: dict[str, Any] = package_manifest_value
    source_revision = _validate_workspace_manifest(
        package_manifest,
        label="embedded workspace manifest",
    )
    manifest_path = root.joinpath(*RESULT_MANIFEST_PATH.parts)
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"result manifest already exists: {manifest_path}")
    entries, tree_sha256 = _tree_entries(
        root, excluded=frozenset({RESULT_MANIFEST_PATH})
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package": RESULT_KIND,
        "exit_status": exit_status,
        "source_revision": source_revision,
        "workspace_manifest_sha256": sha256_file(package_manifest_path),
        "workspace_payload_tree_sha256": package_manifest["payload_tree_sha256"],
        "payload_tree_sha256": tree_sha256,
        "file_count": len(entries),
        "files": entries,
    }
    candidate = manifest_path.with_suffix(".json.candidate")
    candidate.write_bytes(_canonical_json_bytes(manifest))
    os.replace(candidate, manifest_path)
    return manifest_path


def verify_result_archive(archive: Path) -> VerifiedArchive:
    """Verify a success or failure artifact archive without extracting it."""

    archive = archive.expanduser().resolve()
    try:
        package = tarfile.open(archive, mode="r:*")  # noqa: SIM115
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"cannot read result archive {archive}: {error}") from error
    with package:
        members = _checked_members(package)
        root = _single_archive_root(members)
        regular = {path: member for path, member in members.items() if member.isfile()}
        manifest_member = regular.get(root / RESULT_MANIFEST_PATH)
        status_member = regular.get(root / EXIT_STATUS_PATH)
        package_manifest_member = regular.get(root / PACKAGE_MANIFEST_PATH)
        if (
            manifest_member is None
            or status_member is None
            or package_manifest_member is None
        ):
            raise ValueError(
                "result archive is missing its manifest, exit status, or workspace binding"
            )
        manifest = _load_manifest_member(
            package, manifest_member, label="result artifact manifest"
        )
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("package") != RESULT_KIND
        ):
            raise ValueError("result artifact manifest has an unsupported schema")
        status_stream = package.extractfile(status_member)
        if status_stream is None:
            raise ValueError("cannot read result exit status")
        if status_member.size > 4:
            raise ValueError("result exit status is unreasonably long")
        with status_stream:
            status_bytes = status_stream.read()
        try:
            status_text = status_bytes.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("result exit status is not ASCII") from error
        if re.fullmatch(r"(?:0|[1-9][0-9]{0,2})", status_text) is None:
            raise ValueError("result exit status is invalid")
        exit_status = int(status_text)
        if exit_status > 255 or manifest.get("exit_status") != exit_status:
            raise ValueError("result exit status does not match its manifest")
        complete_member = regular.get(root / COMPLETE_PATH)
        if exit_status == 0 and (complete_member is None or complete_member.size != 0):
            raise ValueError("successful result has no empty completion marker")
        actual_entries, tree_sha256 = _entries_from_archive(
            package,
            regular,
            root,
            excluded=frozenset({RESULT_MANIFEST_PATH}),
        )
        _validate_inventory(manifest, actual_entries, tree_sha256)
        package_manifest = _load_manifest_member(
            package,
            package_manifest_member,
            label="embedded workspace manifest",
        )
        source_revision = _validate_workspace_manifest(
            package_manifest,
            label="embedded workspace manifest",
        )
        actual_by_path = {entry["path"]: entry for entry in actual_entries}
        embedded_entry = actual_by_path[PACKAGE_MANIFEST_PATH.as_posix()]
        if (
            manifest.get("source_revision") != source_revision
            or manifest.get("workspace_manifest_sha256") != embedded_entry["sha256"]
            or manifest.get("workspace_payload_tree_sha256")
            != package_manifest["payload_tree_sha256"]
        ):
            raise ValueError("result archive does not bind its workspace manifest")
    return VerifiedArchive(
        path=archive,
        sha256=sha256_file(archive),
        root=root,
        manifest=manifest,
        exit_status=exit_status,
    )


def _write_result_archive(root: Path, output: Path) -> VerifiedArchive:
    _write_deterministic_archive(root.resolve(), output.expanduser().resolve())
    return verify_result_archive(output)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    package = commands.add_parser("package")
    package.add_argument("--repository", type=Path, default=Path.cwd())
    package.add_argument("--output", type=Path, required=True)

    verify_package = commands.add_parser("verify-package")
    verify_package.add_argument("archive", type=Path)

    verify_extracted = commands.add_parser("verify-extracted")
    verify_extracted.add_argument("root", type=Path)

    result_manifest = commands.add_parser("manifest-result")
    result_manifest.add_argument("root", type=Path)
    result_manifest.add_argument("--exit-status", type=int, required=True)

    archive_result = commands.add_parser("archive-result")
    archive_result.add_argument("root", type=Path)
    archive_result.add_argument("--output", type=Path, required=True)

    verify_result = commands.add_parser("verify-result")
    verify_result.add_argument("archive", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "package":
        verified = build_workspace_archive(args.repository, args.output)
    elif args.command == "verify-package":
        verified = verify_workspace_archive(args.archive)
    elif args.command == "verify-extracted":
        manifest = verify_extracted_workspace(args.root)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    elif args.command == "manifest-result":
        path = write_result_manifest(args.root, args.exit_status)
        print(path)
        return 0
    elif args.command == "archive-result":
        verified = _write_result_archive(args.root, args.output)
    elif args.command == "verify-result":
        verified = verify_result_archive(args.archive)
    else:  # pragma: no cover - argparse enforces the finite command set
        raise AssertionError(args.command)
    print(
        json.dumps(
            {
                "path": str(verified.path),
                "sha256": verified.sha256,
                "root": verified.root.as_posix(),
                "exit_status": verified.exit_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
