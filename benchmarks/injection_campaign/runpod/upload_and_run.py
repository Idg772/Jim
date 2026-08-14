"""Upload a curated workspace, run/resume the campaign, and retrieve results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks.device_parallel_nss.runpod.upload_and_run import (
    _connection_when_ready,
    _normalise_archive_path,
    _remote_sha256,
    _sha256_file,
    _verify_workspace_archive,
)
from benchmarks.injection_campaign.common import (
    CATALOGUE_FIELDS,
    PAPER_CATALOGUE_SIZE,
    PAPER_PP_RECOVERIES,
    SCHEMA_VERSION,
    STATUS_FIELDS,
    canonical_sha256,
)


@dataclass(frozen=True)
class FrozenCampaign:
    """Validated metadata for a transport-only campaign-input archive."""

    root: PurePosixPath
    manifest: dict[str, Any]
    manifest_sha256: str
    catalogue_sha256: str


_FROZEN_TOP_LEVEL_FILES = frozenset({"manifest.json", "catalogue.csv", "status.csv"})
_FORBIDDEN_CAMPAIGN_PARTS = frozenset(
    {"results", "cache", ".cache", ".jax-cache", "__pycache__"}
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _run_label(value: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,31}", value) is None:
        raise argparse.ArgumentTypeError(
            "must contain 1-32 lowercase letters, digits, or hyphens"
        )
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pod_id")
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--campaign-input",
        type=Path,
        default=None,
        help=(
            "Frozen campaign-input tar.gz. It must contain exactly one campaign "
            "root with manifest.json, catalogue.csv, status.csv, and inputs/. "
            "Required for staged --start/--stop runs."
        ),
    )
    parser.add_argument("--download-dir", type=Path, default=Path("campaign-results"))
    parser.add_argument(
        "--n-injections", type=_positive_int, default=PAPER_PP_RECOVERIES
    )
    parser.add_argument(
        "--catalogue-size", type=_positive_int, default=PAPER_CATALOGUE_SIZE
    )
    parser.add_argument("--seed", type=int, default=260728265)
    parser.add_argument("--retry-count", type=_nonnegative_int, default=2)
    parser.add_argument("--start", type=_nonnegative_int, default=0)
    parser.add_argument("--stop", type=_positive_int, default=None)
    parser.add_argument(
        "--injection-id",
        dest="injection_ids",
        action="append",
        type=_nonnegative_int,
        default=None,
        help="Explicit sparse injection selection; repeat to preserve a chosen order.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Request publication plots after a frozen run. Frozen staged runs "
            "disable plotting by default."
        ),
    )
    parser.add_argument(
        "--fresh-processes",
        action="store_true",
        help=(
            "Run each recovery in a fresh Python process. Required for campaigns "
            "whose cells install process-local monkeypatches."
        ),
    )
    parser.add_argument("--timeout", type=_positive_int, default=1200)
    parser.add_argument("--poll-interval", type=_positive_int, default=10)
    parser.add_argument(
        "--run-label",
        type=_run_label,
        default=None,
        help="Isolate multiple frozen runs on one pod and in one download directory.",
    )
    parser.add_argument(
        "--implementation",
        choices=("candidate", "paper-baseline"),
        default="candidate",
        help="Select the packaged Jim implementation used by the remote harness.",
    )
    return parser.parse_args(argv)


def _member_bytes(
    package: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> bytes:
    stream = package.extractfile(member)
    if stream is None:
        raise ValueError(f"cannot read campaign archive member: {member.name!r}")
    return stream.read()


def _parse_csv(
    payload: bytes,
    *,
    label: str,
    expected_fields: tuple[str, ...],
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"campaign {label} is not UTF-8: {error}") from error
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != list(expected_fields):
        raise ValueError(f"campaign {label} has unexpected columns")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"campaign {label} contains a malformed row")
    return rows


def _verify_frozen_campaign_archive(archive: Path) -> FrozenCampaign:
    """Validate a frozen input archive before any bytes reach the pod."""

    try:
        package = tarfile.open(archive, mode="r:*")  # noqa: SIM115
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"cannot read campaign archive {archive}: {error}") from error

    with package:
        members: dict[PurePosixPath, tarfile.TarInfo] = {}
        for member in package.getmembers():
            path = _normalise_archive_path(member.name)
            if path == PurePosixPath("."):
                continue
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(
                    "campaign archive contains a forbidden special member: "
                    f"{member.name!r}"
                )
            if not member.isfile() and not member.isdir():
                raise ValueError(
                    f"campaign archive contains an unsupported member: {member.name!r}"
                )
            if path in members:
                raise ValueError(f"campaign archive contains a duplicate path: {path}")
            members[path] = member

        roots = {path.parts[0] for path in members}
        if len(roots) != 1:
            raise ValueError("campaign archive must contain exactly one root directory")
        root = PurePosixPath(roots.pop())
        root_member = members.get(root)
        if root_member is not None and not root_member.isdir():
            raise ValueError("campaign archive root must be a directory")

        regular_members: dict[PurePosixPath, tarfile.TarInfo] = {}
        for path, member in members.items():
            relative = PurePosixPath(*path.parts[1:])
            if relative == PurePosixPath("."):
                continue
            if any(part in _FORBIDDEN_CAMPAIGN_PARTS for part in relative.parts):
                raise ValueError(
                    f"campaign archive contains forbidden results/cache data: {path}"
                )
            top_level = relative.parts[0]
            allowed = (
                relative.as_posix() in _FROZEN_TOP_LEVEL_FILES or top_level == "inputs"
            )
            if not allowed:
                raise ValueError(
                    f"campaign archive contains data outside the frozen inputs: {path}"
                )
            if top_level in _FROZEN_TOP_LEVEL_FILES and len(relative.parts) != 1:
                raise ValueError(f"campaign archive has an invalid path: {path}")
            if member.isfile():
                regular_members[relative] = member

        required = {PurePosixPath(name) for name in _FROZEN_TOP_LEVEL_FILES}
        missing = sorted(required - set(regular_members))
        if missing:
            raise ValueError(
                "campaign archive is missing required files: "
                + ", ".join(map(str, missing))
            )
        input_files = {path for path in regular_members if path.parts[0] == "inputs"}
        if not input_files:
            raise ValueError("campaign archive contains no frozen input files")

        manifest_bytes = _member_bytes(
            package, regular_members[PurePosixPath("manifest.json")]
        )
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid frozen campaign manifest: {error}") from error
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("frozen campaign manifest has an unsupported schema")
        stored_hash = manifest.get("config_sha256")
        hash_input = dict(manifest)
        hash_input.pop("config_sha256", None)
        if stored_hash != canonical_sha256(hash_input):
            raise ValueError("frozen campaign manifest hash mismatch")

        n_injections = manifest.get("n_injections")
        catalogue_size = manifest.get("catalogue_size")
        if (
            not isinstance(n_injections, int)
            or isinstance(n_injections, bool)
            or n_injections < 1
            or not isinstance(catalogue_size, int)
            or isinstance(catalogue_size, bool)
            or catalogue_size < n_injections
        ):
            raise ValueError("frozen campaign manifest has invalid campaign sizes")

        catalogue_metadata = manifest.get("catalogue")
        if not isinstance(catalogue_metadata, dict):
            raise ValueError(  # noqa: TRY004
                "frozen campaign manifest has no catalogue metadata"
            )
        if catalogue_metadata.get("path") != "catalogue.csv":
            raise ValueError(
                "frozen campaign manifest has an unexpected catalogue path"
            )
        catalogue_bytes = _member_bytes(
            package, regular_members[PurePosixPath("catalogue.csv")]
        )
        catalogue_sha256 = hashlib.sha256(catalogue_bytes).hexdigest()
        if catalogue_metadata.get("sha256") != catalogue_sha256:
            raise ValueError("frozen campaign catalogue hash mismatch")
        if catalogue_metadata.get("bytes") != len(catalogue_bytes):
            raise ValueError("frozen campaign catalogue size mismatch")
        catalogue_rows = _parse_csv(
            catalogue_bytes,
            label="catalogue",
            expected_fields=CATALOGUE_FIELDS,
        )
        try:
            catalogue_ids = [int(row["injection_id"]) for row in catalogue_rows]
        except ValueError as error:
            raise ValueError(
                "campaign catalogue has a non-integer injection ID"
            ) from error
        if len(catalogue_rows) != catalogue_size or catalogue_ids != list(
            range(catalogue_size)
        ):
            raise ValueError(
                "campaign catalogue IDs must be contiguous and match catalogue_size"
            )

        status_bytes = _member_bytes(
            package, regular_members[PurePosixPath("status.csv")]
        )
        status_rows = _parse_csv(
            status_bytes,
            label="status",
            expected_fields=STATUS_FIELDS,
        )
        try:
            status_ids = [int(row["injection_id"]) for row in status_rows]
        except ValueError as error:
            raise ValueError(
                "campaign status has a non-integer injection ID"
            ) from error
        if len(status_rows) != n_injections or status_ids != list(range(n_injections)):
            raise ValueError(
                "campaign status IDs must be contiguous and match n_injections"
            )
        if any(
            row["status"] != "pending"
            or row["attempts"] != "0"
            or any(
                row[field]
                for field in (
                    "runtime_seconds",
                    "posterior_samples",
                    "summary",
                    "posterior",
                    "error",
                )
            )
            for row in status_rows
        ):
            raise ValueError("frozen campaign status must contain only pristine inputs")

        psd_metadata = manifest.get("psd")
        psd_files = (
            psd_metadata.get("files") if isinstance(psd_metadata, dict) else None
        )
        if not isinstance(psd_files, dict):
            raise ValueError(  # noqa: TRY004
                "frozen campaign manifest has no PSD file inventory"
            )
        inventoried_inputs: set[PurePosixPath] = set()
        for relative_value, metadata in psd_files.items():
            if not isinstance(relative_value, str) or not isinstance(metadata, dict):
                raise ValueError(  # noqa: TRY004
                    "frozen campaign manifest has invalid PSD metadata"
                )
            relative = _normalise_archive_path(relative_value)
            if relative == PurePosixPath(".") or relative.parts[0] != "inputs":
                raise ValueError("frozen campaign PSD inventory escapes inputs/")
            member = regular_members.get(relative)
            if member is None:
                raise ValueError(
                    f"frozen campaign is missing inventoried input: {relative}"
                )
            payload = _member_bytes(package, member)
            if metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise ValueError(f"frozen campaign input hash mismatch: {relative}")
            if metadata.get("bytes") != len(payload):
                raise ValueError(f"frozen campaign input size mismatch: {relative}")
            inventoried_inputs.add(relative)
        if inventoried_inputs != input_files:
            raise ValueError("frozen campaign contains un-inventoried input files")

        return FrozenCampaign(
            root=root,
            manifest=manifest,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            catalogue_sha256=catalogue_sha256,
        )


def _build_run_command(
    args: argparse.Namespace,
    remote_output: str,
    *,
    n_injections: int,
    catalogue_size: int,
    frozen: bool,
) -> list[str]:
    command = [
        "bash",
        "benchmarks/injection_campaign/runpod/run_on_pod.sh",
        "--output-dir",
        remote_output,
        "--n-injections",
        str(n_injections),
        "--catalogue-size",
        str(catalogue_size),
        "--seed",
        str(args.seed),
        "--retry-count",
        str(args.retry_count),
        "--implementation",
        getattr(args, "implementation", "candidate"),
    ]
    injection_ids = getattr(args, "injection_ids", None)
    if injection_ids is not None:
        for injection_id in injection_ids:
            command.extend(("--injection-id", str(injection_id)))
    else:
        command.extend(("--start", str(args.start)))
        if args.stop is not None:
            command.extend(("--stop", str(args.stop)))
    if frozen:
        command.append("--require-existing-campaign")
        if not args.plot:
            command.append("--no-plot")
    if getattr(args, "fresh_processes", False):
        command.append("--fresh-processes")
    return command


def _validate_frozen_implementation_pin(
    manifest: dict[str, Any],
    package: dict[str, Any],
    implementation: str,
) -> None:
    """Bind a pinned frozen diagnostic to the packaged implementation tree."""

    baseline = manifest.get("baseline_diagnostic")
    candidate = manifest.get("implementation_diagnostic")
    pins = [value for value in (baseline, candidate) if isinstance(value, dict)]
    if len(pins) > 1:
        raise ValueError("frozen campaign has ambiguous implementation pins")
    if implementation == "paper-baseline":
        if not isinstance(baseline, dict):
            raise ValueError("frozen campaign does not pin the paper baseline")
        pin = baseline
        section_name = "baseline"
    else:
        if isinstance(baseline, dict):
            raise ValueError("paper-baseline campaign cannot run as candidate")
        if not isinstance(candidate, dict):
            return
        pin = candidate
        section_name = "candidate"

    section = package.get(section_name)
    if not isinstance(section, dict):
        raise TypeError(f"workspace package has no {section_name} implementation")
    if (
        pin.get("implementation_label") != implementation
        or pin.get("implementation_revision") != section.get("revision")
        or pin.get("implementation_tree_sha256") != section.get("tree_sha256")
    ):
        raise ValueError(
            f"frozen campaign does not pin the packaged {section_name} implementation"
        )


def _publish_result_receipt(
    temporary: Path,
    destination: Path,
    expected_sha256: str,
) -> bool:
    """Publish a verified download without replacing an existing receipt.

    A hard link gives us an atomic no-clobber operation because both paths live
    in the download directory. If another run already published this name, an
    identical archive is accepted idempotently; any differing file, symlink, or
    non-file destination is refused.

    Returns ``True`` when this call published the receipt and ``False`` when an
    identical receipt was already present.
    """

    try:
        os.link(temporary, destination)
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file():
            raise RuntimeError(
                "refusing to replace an existing non-regular result receipt: "
                f"{destination}"
            ) from None
        existing_sha256 = _sha256_file(destination)
        if existing_sha256 != expected_sha256:
            raise RuntimeError(
                "refusing to overwrite a differing existing result receipt: "
                f"{destination} (existing={existing_sha256}, "
                f"downloaded={expected_sha256})"
            ) from None
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.n_injections > args.catalogue_size:
        raise SystemExit("--n-injections cannot exceed --catalogue-size")
    if args.injection_ids is not None and (args.start != 0 or args.stop is not None):
        raise SystemExit("--injection-id cannot be combined with --start/--stop")
    if args.injection_ids is not None and len(set(args.injection_ids)) != len(
        args.injection_ids
    ):
        raise SystemExit("--injection-id values must be unique")
    if args.campaign_input is None and (
        args.start != 0 or args.stop is not None or args.injection_ids is not None
    ):
        raise SystemExit(
            "staged --start/--stop/--injection-id runs require --campaign-input"
        )
    if args.implementation == "paper-baseline" and args.campaign_input is None:
        raise SystemExit("paper-baseline runs require --campaign-input")
    if re.fullmatch(r"[A-Za-z0-9_-]+", args.pod_id) is None:
        raise SystemExit(f"unexpected pod id: {args.pod_id!r}")
    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"workspace archive does not exist: {archive}")
    try:
        package = _verify_workspace_archive(archive)
    except ValueError as error:
        raise SystemExit(f"refusing unsafe workspace archive: {error}") from error
    campaign_paths = {entry["path"] for entry in package["candidate"]["files"]}
    required = {
        "benchmarks/injection_campaign/run_campaign.py",
        "benchmarks/injection_campaign/runpod/run_on_pod.sh",
    }
    if not required <= campaign_paths:
        raise SystemExit("workspace package predates the injection campaign files")

    frozen_campaign: FrozenCampaign | None = None
    campaign_input: Path | None = None
    campaign_input_sha256: str | None = None
    n_injections = args.n_injections
    catalogue_size = args.catalogue_size
    if args.campaign_input is not None:
        campaign_input = args.campaign_input.expanduser().resolve()
        if not campaign_input.is_file():
            raise SystemExit(
                f"frozen campaign archive does not exist: {campaign_input}"
            )
        try:
            frozen_campaign = _verify_frozen_campaign_archive(campaign_input)
        except ValueError as error:
            raise SystemExit(f"refusing unsafe frozen campaign: {error}") from error
        campaign_input_sha256 = _sha256_file(campaign_input)
        n_injections = int(frozen_campaign.manifest["n_injections"])
        catalogue_size = int(frozen_campaign.manifest["catalogue_size"])
        try:
            _validate_frozen_implementation_pin(
                frozen_campaign.manifest,
                package,
                args.implementation,
            )
        except (TypeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        print(
            "Verified frozen campaign "
            f"root={frozen_campaign.root} "
            f"config={frozen_campaign.manifest['config_sha256']}",
            flush=True,
        )

    effective_stop: int | None = None
    if args.injection_ids is not None:
        invalid = [value for value in args.injection_ids if value >= n_injections]
        if invalid:
            raise SystemExit(
                "injection IDs outside the frozen campaign: "
                + ", ".join(map(str, invalid))
            )
    else:
        effective_stop = n_injections if args.stop is None else args.stop
        if args.start >= effective_stop:
            raise SystemExit(f"empty injection range [{args.start}, {effective_stop})")
        if effective_stop > n_injections:
            raise SystemExit(
                f"--stop {effective_stop} exceeds frozen campaign size {n_injections}"
            )

    destination_dir = args.download_dir.expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_archive_sha256 = _sha256_file(archive)
    remote_root = f"/workspace/Jim-campaign-{args.pod_id}"
    remote_archive = f"/workspace/jim-campaign-workspace-{args.pod_id}.tar.gz"
    remote_workspace_marker = f"{remote_root}.sha256"
    remote_workspace_staging = f"{remote_root}.extracting"
    label_suffix = f"-{args.run_label}" if args.run_label is not None else ""
    remote_output = f"/workspace/jim-injection-campaign-{args.pod_id}{label_suffix}"
    remote_campaign_archive = (
        f"/workspace/jim-campaign-input-{args.pod_id}{label_suffix}.tar.gz"
    )
    remote_campaign_marker = f"{remote_output}.input.sha256"
    remote_campaign_staging = f"{remote_output}.extracting"
    remote_results = f"{remote_output}.tar.gz"

    with tempfile.TemporaryDirectory(
        prefix=f"jim-campaign-known-hosts-{args.pod_id}-"
    ) as temporary:
        known_hosts = Path(temporary) / "known_hosts"
        connection = _connection_when_ready(
            args.pod_id, args.timeout, args.poll_interval, known_hosts
        )
        scp_options = connection.scp_options(known_hosts)
        subprocess.run(
            [
                "scp",
                *scp_options,
                str(archive),
                f"{connection.target}:{remote_archive}",
            ],
            check=True,
        )
        if (
            _remote_sha256(connection, known_hosts, remote_archive)
            != local_archive_sha256
        ):
            raise RuntimeError("uploaded workspace SHA-256 mismatch")

        if campaign_input is not None:
            subprocess.run(
                [
                    "scp",
                    *scp_options,
                    str(campaign_input),
                    f"{connection.target}:{remote_campaign_archive}",
                ],
                check=True,
            )
            if (
                _remote_sha256(connection, known_hosts, remote_campaign_archive)
                != campaign_input_sha256
            ):
                raise RuntimeError("uploaded frozen campaign SHA-256 mismatch")

        quoted_workspace_root = shlex.quote(remote_root)
        quoted_workspace_marker = shlex.quote(remote_workspace_marker)
        quoted_workspace_staging = shlex.quote(remote_workspace_staging)
        workspace_setup = " ".join(
            [
                f"if test -e {quoted_workspace_root}; then",
                f"test -d {quoted_workspace_root} &&",
                "grep -Fqx",
                shlex.quote(local_archive_sha256),
                f"{quoted_workspace_marker};",
                "else",
                f"test ! -e {quoted_workspace_staging} &&",
                f"mkdir -p {quoted_workspace_staging} &&",
                "tar --no-same-owner --no-same-permissions -xzf",
                shlex.quote(remote_archive),
                f"-C {quoted_workspace_staging} &&",
                f"mv {quoted_workspace_staging} {quoted_workspace_root} &&",
                "printf '%s\\n'",
                shlex.quote(local_archive_sha256),
                f"> {quoted_workspace_marker};",
                "fi",
            ]
        )
        setup_parts = [workspace_setup]
        if campaign_input_sha256 is not None:
            quoted_output = shlex.quote(remote_output)
            quoted_campaign_marker = shlex.quote(remote_campaign_marker)
            quoted_campaign_staging = shlex.quote(remote_campaign_staging)
            campaign_setup = " ".join(
                [
                    f"if test -e {quoted_output}; then",
                    f"test -d {quoted_output} &&",
                    "grep -Fqx",
                    shlex.quote(campaign_input_sha256),
                    f"{quoted_campaign_marker} &&",
                    f"test -f {quoted_output}/manifest.json &&",
                    f"test -f {quoted_output}/catalogue.csv &&",
                    f"test -f {quoted_output}/status.csv &&",
                    f"test -d {quoted_output}/inputs;",
                    "else",
                    f"test ! -e {quoted_campaign_staging} &&",
                    f"mkdir -p {quoted_campaign_staging} &&",
                    "tar --no-same-owner --no-same-permissions --strip-components=1 -xzf",
                    shlex.quote(remote_campaign_archive),
                    f"-C {quoted_campaign_staging} &&",
                    f"test -f {quoted_campaign_staging}/manifest.json &&",
                    f"test -f {quoted_campaign_staging}/catalogue.csv &&",
                    f"test -f {quoted_campaign_staging}/status.csv &&",
                    f"test -d {quoted_campaign_staging}/inputs &&",
                    f"mv {quoted_campaign_staging} {quoted_output} &&",
                    "printf '%s\\n'",
                    shlex.quote(campaign_input_sha256),
                    f"> {quoted_campaign_marker};",
                    "fi",
                ]
            )
            setup_parts.append(campaign_setup)
        setup = " && ".join(f"( {part} )" for part in setup_parts)
        subprocess.run(
            [
                "ssh",
                *connection.ssh_options(known_hosts),
                connection.target,
                setup,
            ],
            check=True,
        )
        if frozen_campaign is not None:
            remote_manifest_sha256 = _remote_sha256(
                connection,
                known_hosts,
                f"{remote_output}/manifest.json",
            )
            if remote_manifest_sha256 != frozen_campaign.manifest_sha256:
                raise RuntimeError("remote frozen campaign manifest mismatch")
            remote_catalogue_sha256 = _remote_sha256(
                connection,
                known_hosts,
                f"{remote_output}/catalogue.csv",
            )
            if remote_catalogue_sha256 != frozen_campaign.catalogue_sha256:
                raise RuntimeError("remote frozen campaign catalogue mismatch")
            print(
                "Verified remote frozen campaign "
                f"config={frozen_campaign.manifest['config_sha256']} "
                f"catalogue={remote_catalogue_sha256}",
                flush=True,
            )
        command = shlex.join(
            _build_run_command(
                args,
                remote_output,
                n_injections=n_injections,
                catalogue_size=catalogue_size,
                frozen=frozen_campaign is not None,
            )
        )
        execution = f"cd {shlex.quote(remote_root)} && {command}"
        result = subprocess.run(
            [
                "ssh",
                *connection.ssh_options(known_hosts),
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=20",
                connection.target,
                execution,
            ],
            check=False,
        )
        remote_results_staging = f"{remote_results}.partial"
        output_parent = str(PurePosixPath(remote_output).parent)
        output_name = PurePosixPath(remote_output).name
        archive_results = " && ".join(
            [
                shlex.join(
                    [
                        "tar",
                        "-C",
                        output_parent,
                        f"--exclude={output_name}/.jax-cache",
                        "-czf",
                        remote_results_staging,
                        output_name,
                    ]
                ),
                shlex.join(["mv", remote_results_staging, remote_results]),
            ]
        )
        subprocess.run(
            [
                "ssh",
                *connection.ssh_options(known_hosts),
                connection.target,
                archive_results,
            ],
            check=True,
        )
        remote_digest = _remote_sha256(connection, known_hosts, remote_results)
        if args.injection_ids is not None:
            encoded_ids = ",".join(map(str, args.injection_ids)).encode()
            selection_hash = hashlib.sha256(encoded_ids).hexdigest()[:12]
            range_suffix = f"-ids{len(args.injection_ids)}-{selection_hash}"
        elif args.start == 0 and args.stop is None:
            range_suffix = ""
        else:
            assert effective_stop is not None
            range_suffix = f"-{args.start:03d}-{effective_stop:03d}"
        destination = destination_dir / (
            f"{args.pod_id}{label_suffix}-campaign{range_suffix}.tar.gz"
        )
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=destination.parent,
            delete=False,
        ) as temporary_stream:
            temporary_destination = Path(temporary_stream.name)
        try:
            subprocess.run(
                [
                    "scp",
                    *scp_options,
                    f"{connection.target}:{remote_results}",
                    str(temporary_destination),
                ],
                check=True,
            )
            if _sha256_file(temporary_destination) != remote_digest:
                raise RuntimeError("downloaded result SHA-256 mismatch")
            published = _publish_result_receipt(
                temporary_destination,
                destination,
                remote_digest,
            )
            if not published:
                print(
                    f"Keeping identical existing result receipt: {destination}",
                    flush=True,
                )
        finally:
            temporary_destination.unlink(missing_ok=True)
    print(destination)
    print(
        f"Results are safe locally. Delete the pod: runpodctl pod delete {args.pod_id}"
    )
    if result.returncode != 0:
        raise SystemExit(
            f"campaign exited with status {result.returncode}; the partial results were downloaded"
        )


if __name__ == "__main__":
    main()
