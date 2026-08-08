"""Upload the benchmark workspace, run it over SSH, and download the results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ALIGNED_WORKLOAD = "aligned-11d"
PAPER_WORKLOAD = "paper-15d"
WORKLOAD_CHOICES = (ALIGNED_WORKLOAD, PAPER_WORKLOAD)
PACKAGE_MANIFEST_PATH = PurePosixPath(".runpod/package-manifest.json")


@dataclass(frozen=True)
class SSHConnection:
    target: str
    port: str
    identity: Path

    def ssh_options(
        self,
        known_hosts: Path,
        *,
        accept_new: bool = False,
    ) -> list[str]:
        return [
            "-o",
            "StrictHostKeyChecking=" + ("accept-new" if accept_new else "yes"),
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-p",
            self.port,
            "-i",
            str(self.identity),
        ]

    def scp_options(self, known_hosts: Path) -> list[str]:
        return [
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-P",
            self.port,
            "-i",
            str(self.identity),
        ]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pod_id")
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("benchmark-results/device-parallel-nss"),
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help=(
            "Optional frozen GW170817 NPZ to upload into the result tree before "
            "execution. The pod verifies and reuses this exact bundle instead "
            "of preparing a fresh copy from GWOSC."
        ),
    )
    parser.add_argument(
        "--workload",
        choices=WORKLOAD_CHOICES,
        default=ALIGNED_WORKLOAD,
        help=(
            "Scientific workload to run remotely. aligned-11d preserves the "
            "historical benchmark; paper-15d selects the full paper model."
        ),
    )
    single_run_group = parser.add_mutually_exclusive_group()
    single_run_group.add_argument(
        "--candidate-only",
        action="store_true",
        help=(
            "Run one four-GPU candidate analysis with profiling and posterior "
            "sample and GPU HLO output instead of the comparison and scaling "
            "matrix."
        ),
    )
    parser.add_argument(
        "--no-hlo",
        action="store_true",
        help=(
            "Disable GPU HLO capture for --candidate-only while retaining "
            "profiling, telemetry, per-slice diagnostics, and posterior output."
        ),
    )
    single_run_group.add_argument(
        "--original-sharded-only",
        action="store_true",
        help=(
            "Run one four-GPU analysis against the pinned original paper-style "
            "sharded-live-state revision, with profiling and posterior output."
        ),
    )
    parser.add_argument("--timeout", type=_positive_int, default=900)
    parser.add_argument("--poll-interval", type=_positive_int, default=10)
    args = parser.parse_args()
    if args.no_hlo and not args.candidate_only:
        parser.error("--no-hlo requires --candidate-only")
    if args.data_file is not None and not args.candidate_only:
        parser.error("--data-file requires --candidate-only")
    return args


def _find_ssh_command(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("sshCommand", "ssh_command"):
            command = value.get(key)
            if isinstance(command, str):
                return command
        for child in value.values():
            found = _find_ssh_command(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_ssh_command(child)
            if found is not None:
                return found
    return None


def _parse_connection(output: str) -> SSHConnection:
    payload = json.loads(output)
    raw_command = _find_ssh_command(payload)
    if raw_command is None:
        raise ValueError("runpodctl response did not include sshCommand")
    parts = shlex.split(raw_command)
    if not parts or Path(parts[0]).name != "ssh":
        raise ValueError(f"unexpected SSH command: {raw_command}")

    target: str | None = None
    port: str | None = None
    identity: Path | None = None
    index = 1
    while index < len(parts):
        part = parts[index]
        if part == "-p":
            port = parts[index + 1]
            index += 2
        elif part == "-i":
            identity = Path(parts[index + 1]).expanduser().resolve()
            index += 2
        elif part.startswith("-"):
            index += 1
        else:
            target = part
            index += 1

    if target is None or port is None or identity is None:
        raise ValueError(f"incomplete SSH command: {raw_command}")
    if not identity.is_file():
        raise ValueError(f"SSH identity does not exist: {identity}")
    return SSHConnection(target=target, port=port, identity=identity)


def _connection_when_ready(
    pod_id: str,
    timeout: int,
    poll_interval: int,
    known_hosts: Path,
) -> SSHConnection:
    deadline = time.monotonic() + timeout
    last_error = "SSH information is not ready"
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["runpodctl", "ssh", "info", pod_id],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            try:
                connection = _parse_connection(result.stdout)
            except (json.JSONDecodeError, ValueError) as error:
                last_error = str(error)
            else:
                accept_new = not known_hosts.exists() or known_hosts.stat().st_size == 0
                probe = subprocess.run(
                    [
                        "ssh",
                        *connection.ssh_options(
                            known_hosts,
                            accept_new=accept_new,
                        ),
                        connection.target,
                        "true",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if probe.returncode == 0:
                    return connection
                last_error = probe.stderr.strip() or "SSH probe failed"
        else:
            last_error = result.stderr.strip() or result.stdout.strip()
        print(f"Waiting for SSH: {last_error}", flush=True)
        time.sleep(poll_interval)
    raise TimeoutError(f"SSH was not ready after {timeout}s: {last_error}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_sha256(
    connection: SSHConnection,
    known_hosts: Path,
    path: str,
) -> str:
    result = subprocess.run(
        [
            "ssh",
            *connection.ssh_options(known_hosts),
            connection.target,
            shlex.join(["sha256sum", path]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.split(maxsplit=1)[0] if result.stdout.strip() else ""
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"unexpected remote SHA-256 output: {result.stdout!r}")
    return digest


def _normalise_archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive contains an unsafe path: {name!r}")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts:
        return PurePosixPath(".")
    if ".git" in parts:
        raise ValueError(f"archive contains forbidden Git metadata: {name!r}")
    return PurePosixPath(*parts)


def _verify_workspace_archive(archive: Path) -> dict[str, Any]:
    """Validate package safety, provenance, and the curated file inventory."""

    try:
        package = tarfile.open(archive, mode="r:*")  # noqa: SIM115
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"cannot read workspace archive {archive}: {error}") from error

    with package:
        regular_members: dict[PurePosixPath, tarfile.TarInfo] = {}
        for member in package.getmembers():
            path = _normalise_archive_path(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(
                    f"archive contains a forbidden special member: {member.name!r}"
                )
            if not member.isfile() and not member.isdir():
                raise ValueError(
                    f"archive contains an unsupported member: {member.name!r}"
                )
            if member.isfile():
                if path in regular_members:
                    raise ValueError(f"archive contains a duplicate path: {path}")
                regular_members[path] = member

        manifest_member = regular_members.get(PACKAGE_MANIFEST_PATH)
        if manifest_member is None:
            raise ValueError(f"archive is missing {PACKAGE_MANIFEST_PATH.as_posix()}")
        manifest_stream = package.extractfile(manifest_member)
        if manifest_stream is None:
            raise ValueError("cannot read package manifest")
        try:
            manifest = json.load(manifest_stream)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid package manifest: {error}") from error

        if (
            manifest.get("schema_version") != 1
            or manifest.get("package") != "jim-gw170817-runpod"
        ):
            raise ValueError("archive contains an unsupported package manifest")
        candidate = manifest.get("candidate")
        baseline = manifest.get("baseline")
        provenance = manifest.get("provenance")
        if not all(
            isinstance(value, dict) for value in (candidate, baseline, provenance)
        ):
            raise ValueError("package manifest is missing provenance sections")
        for label, section in (("candidate", candidate), ("baseline", baseline)):
            revision = section.get("revision")
            if (
                not isinstance(revision, str)
                or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            ):
                raise ValueError(f"package manifest has an invalid {label} revision")
        if provenance.get("git_metadata_included") is not False:
            raise ValueError("package manifest does not declare Git metadata excluded")

        baseline_root_value = baseline.get("root")
        if not isinstance(baseline_root_value, str):
            raise ValueError("package manifest has no baseline root")  # noqa: TRY004
        baseline_root = _normalise_archive_path(baseline_root_value)

        expected_files = {PACKAGE_MANIFEST_PATH}
        for label, section, root in (
            ("candidate", candidate, PurePosixPath(".")),
            ("baseline", baseline, baseline_root),
        ):
            files = section.get("files")
            if not isinstance(files, list) or not files:
                raise ValueError(f"package manifest has no {label} file inventory")
            if section.get("file_count") != len(files):
                raise ValueError(f"package manifest has a wrong {label} file count")
            for entry in files:
                if not isinstance(entry, dict):
                    raise ValueError(  # noqa: TRY004
                        f"invalid {label} file inventory entry"
                    )
                relative_value = entry.get("path")
                digest = entry.get("sha256")
                size = entry.get("bytes")
                if not isinstance(relative_value, str):
                    raise ValueError(f"invalid {label} inventory path")  # noqa: TRY004
                relative = _normalise_archive_path(relative_value)
                path = relative if root == PurePosixPath(".") else root / relative
                member = regular_members.get(path)
                if member is None:
                    raise ValueError(f"archive is missing inventoried file: {path}")
                if not isinstance(size, int) or member.size != size:
                    raise ValueError(f"archive size mismatch for {path}")
                if (
                    not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    raise ValueError(f"invalid SHA-256 for {path}")
                stream = package.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read inventoried file: {path}")
                actual_digest = hashlib.sha256(stream.read()).hexdigest()
                if actual_digest != digest:
                    raise ValueError(f"archive SHA-256 mismatch for {path}")
                expected_files.add(path)

        unexpected = sorted(set(regular_members) - expected_files)
        if unexpected:
            raise ValueError(
                "archive contains files outside the curated inventory: "
                + ", ".join(map(str, unexpected[:5]))
            )
        return manifest


def _build_run_command(
    args: argparse.Namespace,
    remote_output: str,
) -> list[str]:
    command = [
        "bash",
        "benchmarks/device_parallel_nss/runpod/run_on_pod.sh",
        "--output-dir",
        remote_output,
        "--workload",
        args.workload,
    ]
    if args.candidate_only:
        command.append("--candidate-only")
        if args.no_hlo:
            command.append("--no-hlo")
    elif args.original_sharded_only:
        command.append("--original-sharded-only")
    return command


def main() -> None:
    args = _parse_args()
    if re.fullmatch(r"[A-Za-z0-9_-]+", args.pod_id) is None:
        raise SystemExit(f"unexpected pod id: {args.pod_id!r}")
    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"archive does not exist: {archive}")
    try:
        package_manifest = _verify_workspace_archive(archive)
    except ValueError as error:
        raise SystemExit(f"refusing unsafe workspace archive: {error}") from error
    print(
        "Verified workspace package "
        f"candidate={package_manifest['candidate']['revision']} "
        f"baseline={package_manifest['baseline']['revision']}",
        flush=True,
    )
    if not args.candidate_only and not args.original_sharded_only:
        raise SystemExit(
            "the minimized git-free package requires --candidate-only or "
            "--original-sharded-only"
        )
    download_dir = args.download_dir.expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    data_file: Path | None = None
    local_data_sha256: str | None = None
    if args.data_file is not None:
        data_file = args.data_file.expanduser().resolve()
        if not data_file.is_file() or data_file.suffix.lower() != ".npz":
            raise SystemExit(f"frozen data bundle does not exist: {data_file}")
        local_data_sha256 = _sha256_file(data_file)
    mode = "candidate" if args.candidate_only else "original-sharded"
    local_archive_sha256 = _sha256_file(archive)
    remote_root = f"/workspace/Jim-{args.pod_id}-{mode}"
    remote_archive = f"/workspace/jim-gw170817-benchmark-{args.pod_id}-{mode}.tar.gz"
    remote_output = f"/workspace/jim-gw170817-results/{args.pod_id}-{mode}"
    remote_results = f"{remote_output}.tar.gz"

    run_command = _build_run_command(args, remote_output)

    with tempfile.TemporaryDirectory(
        prefix=f"jim-runpod-known-hosts-{args.pod_id}-"
    ) as temporary:
        known_hosts = Path(temporary) / "known_hosts"
        connection = _connection_when_ready(
            args.pod_id,
            args.timeout,
            args.poll_interval,
            known_hosts,
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
        remote_archive_sha256 = _remote_sha256(
            connection,
            known_hosts,
            remote_archive,
        )
        if remote_archive_sha256 != local_archive_sha256:
            raise RuntimeError(
                "uploaded workspace SHA-256 mismatch: "
                f"local={local_archive_sha256} remote={remote_archive_sha256}"
            )

        uploaded_data = f"/workspace/gw170817-frozen-{args.pod_id}.npz"
        if data_file is not None:
            subprocess.run(
                [
                    "scp",
                    *scp_options,
                    str(data_file),
                    f"{connection.target}:{uploaded_data}",
                ],
                check=True,
            )
            remote_data_sha256 = _remote_sha256(
                connection,
                known_hosts,
                uploaded_data,
            )
            if remote_data_sha256 != local_data_sha256:
                raise RuntimeError(
                    "uploaded frozen-data SHA-256 mismatch: "
                    f"local={local_data_sha256} remote={remote_data_sha256}"
                )

        setup_steps = [
            "mkdir -p /workspace",
            f"test ! -e {shlex.quote(remote_root)}",
            f"mkdir {shlex.quote(remote_root)}",
            (
                "tar --no-same-owner -xzf "
                f"{shlex.quote(remote_archive)} -C {shlex.quote(remote_root)}"
            ),
        ]
        candidate_data_sha256: str | None = None
        copied_data = f"{remote_output}/data/gw170817.npz"
        if data_file is not None:
            candidate_data_sha256 = local_data_sha256
            setup_steps.extend(
                [
                    f"mkdir -p {shlex.quote(remote_output + '/data')}",
                    f"cp {shlex.quote(uploaded_data)} {shlex.quote(copied_data)}",
                ]
            )
        elif args.original_sharded_only:
            candidate_data = (
                "/workspace/jim-gw170817-results/"
                f"{args.pod_id}-candidate/data/gw170817.npz"
            )
            candidate_data_sha256 = _remote_sha256(
                connection,
                known_hosts,
                candidate_data,
            )
            setup_steps.extend(
                [
                    f"mkdir -p {shlex.quote(remote_output + '/data')}",
                    f"cp {shlex.quote(candidate_data)} {shlex.quote(copied_data)}",
                ]
            )
        subprocess.run(
            [
                "ssh",
                *connection.ssh_options(known_hosts),
                connection.target,
                " && ".join(setup_steps),
            ],
            check=True,
        )
        if candidate_data_sha256 is not None:
            copied_data_sha256 = _remote_sha256(
                connection,
                known_hosts,
                copied_data,
            )
            if copied_data_sha256 != candidate_data_sha256:
                raise RuntimeError(
                    "frozen-data copy SHA-256 mismatch: "
                    f"expected={candidate_data_sha256} copied={copied_data_sha256}"
                )

        execution_command = " && ".join(
            [
                f"cd {shlex.quote(remote_root)}",
                "export UV_HTTP_TIMEOUT=300",
                "export UV_HTTP_RETRIES=10",
                "export UV_CONCURRENT_DOWNLOADS=4",
                shlex.join(run_command),
            ]
        )
        subprocess.run(
            [
                "ssh",
                *connection.ssh_options(known_hosts),
                connection.target,
                execution_command,
            ],
            check=True,
        )

        if local_data_sha256 is not None:
            used_data_sha256 = _remote_sha256(
                connection,
                known_hosts,
                copied_data,
            )
            if used_data_sha256 != local_data_sha256:
                raise RuntimeError(
                    "benchmark mutated or replaced the frozen data bundle: "
                    f"expected={local_data_sha256} used={used_data_sha256}"
                )

        remote_results_sha256 = _remote_sha256(
            connection,
            known_hosts,
            remote_results,
        )
        destination = download_dir / f"{args.pod_id}-{mode}.tar.gz"
        subprocess.run(
            [
                "scp",
                *scp_options,
                f"{connection.target}:{remote_results}",
                str(destination),
            ],
            check=True,
        )
        local_results_sha256 = _sha256_file(destination)
        if local_results_sha256 != remote_results_sha256:
            raise RuntimeError(
                "downloaded result SHA-256 mismatch: "
                f"remote={remote_results_sha256} local={local_results_sha256}"
            )
    print(destination)
    print(
        f"Results are safe locally. Delete the pod: runpodctl pod delete {args.pod_id}"
    )


if __name__ == "__main__":
    main()
