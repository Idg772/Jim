"""Upload one immutable XG package, run it, and retrieve verified artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.xg.runpod.workflow import (
    GPU_COUNT,
    GPU_ID,
    GUARD_MINUTES,
    IMAGE,
    VerifiedArchive,
    sha256_file,
    verify_result_archive,
    verify_workspace_archive,
)

LAUNCH_RECEIPT_KIND = "jim-xg-runpod-launch"
MINIMUM_GUARD_REMAINING = timedelta(minutes=105)


@dataclass(frozen=True)
class SSHConnection:
    """One pinned direct-SSH connection returned by Runpod."""

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


@dataclass(frozen=True)
class LaunchReceipt:
    """Locally verified binding from the fixed launcher to one paid pod."""

    path: Path
    pod_id: str
    terminate_after: datetime
    workspace_archive_sha256: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_receipt", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("benchmark-results/xg-ce-4096-65536/runpod"),
    )
    parser.add_argument("--timeout", type=_positive_int, default=900)
    parser.add_argument("--poll-interval", type=_positive_int, default=10)
    parser.add_argument(
        "--keep-pod",
        action="store_true",
        help="Keep the pod after a verified download; automatic deletion still applies.",
    )
    return parser.parse_args(argv)


def _utc_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"launch receipt has invalid {field}")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(f"launch receipt has invalid {field}") from error
    if parsed.tzinfo != UTC or parsed.microsecond != 0:
        raise ValueError(f"launch receipt has invalid {field}")
    return parsed


def _load_launch_receipt(
    path: Path,
    workspace: VerifiedArchive,
    *,
    now: datetime | None = None,
) -> LaunchReceipt:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024:
        raise ValueError(f"launch receipt is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"launch receipt is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("launch receipt must be a JSON object")  # noqa: TRY004
    expected_keys = {
        "schema_version",
        "kind",
        "pod_id",
        "image",
        "gpu_id",
        "gpu_count",
        "guard_minutes",
        "launch_started_at",
        "terminate_after",
        "workspace_archive_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("launch receipt has an unexpected schema")
    expected_constants = {
        "schema_version": 1,
        "kind": LAUNCH_RECEIPT_KIND,
        "image": IMAGE,
        "gpu_id": GPU_ID,
        "gpu_count": GPU_COUNT,
        "guard_minutes": GUARD_MINUTES,
        "workspace_archive_sha256": workspace.sha256,
    }
    for field, expected in expected_constants.items():
        if payload.get(field) != expected:
            raise ValueError(f"launch receipt has invalid {field}")
    pod_id = payload.get("pod_id")
    if not isinstance(pod_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", pod_id) is None:
        raise ValueError("launch receipt has an invalid pod id")
    launch_started_at = _utc_timestamp(
        payload.get("launch_started_at"), field="launch_started_at"
    )
    terminate_after = _utc_timestamp(
        payload.get("terminate_after"), field="terminate_after"
    )
    if terminate_after - launch_started_at != timedelta(minutes=GUARD_MINUTES):
        raise ValueError("launch receipt does not prove the fixed deletion guard")
    observed_now = now or datetime.now(UTC)
    if observed_now.tzinfo is None:
        raise ValueError("current time must be timezone-aware")
    if terminate_after - observed_now.astimezone(UTC) < MINIMUM_GUARD_REMAINING:
        raise ValueError("too little deletion-guard time remains for the XG workflow")
    return LaunchReceipt(
        path=path,
        pod_id=pod_id,
        terminate_after=terminate_after,
        workspace_archive_sha256=workspace.sha256,
    )


def _validate_pod_metadata(payload: object, pod_id: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError("runpodctl pod details must be a JSON object")  # noqa: TRY004
    machine = payload.get("machine")
    expected = {
        "id": pod_id,
        "imageName": IMAGE,
        "gpuCount": GPU_COUNT,
        "desiredStatus": "RUNNING",
        "runtimeStatus": "running",
        "volumeMountPath": "/workspace",
    }
    mismatches = [
        field for field, value in expected.items() if payload.get(field) != value
    ]
    if not isinstance(machine, dict) or machine.get("secureCloud") is not True:
        mismatches.append("machine.secureCloud")
    if not isinstance(machine, dict) or machine.get("gpuId") != GPU_ID:
        mismatches.append("machine.gpuId")
    ports = payload.get("ports")
    if not isinstance(ports, list) or "22/tcp" not in ports:
        mismatches.append("ports")
    if mismatches:
        raise ValueError(
            "pod does not match the frozen XG launch: " + ", ".join(mismatches)
        )


def _verify_live_pod(pod_id: str) -> None:
    result = subprocess.run(
        [
            "runpodctl",
            "pod",
            "get",
            pod_id,
            "--include-machine",
            "--output",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("runpodctl returned invalid pod metadata") from error
    _validate_pod_metadata(payload, pod_id)


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
        if part in {"-p", "-i"}:
            if index + 1 >= len(parts):
                raise ValueError(f"incomplete SSH option: {part}")
            if part == "-p":
                port = parts[index + 1]
            else:
                identity = Path(parts[index + 1]).expanduser().resolve()
            index += 2
        elif part.startswith("-"):
            index += 1
        else:
            target = part
            index += 1
    if target is None or port is None or identity is None:
        raise ValueError(f"incomplete SSH command: {raw_command}")
    if re.fullmatch(r"[1-9][0-9]{0,4}", port) is None or int(port) > 65535:
        raise ValueError(f"invalid SSH port: {port}")
    if not identity.is_file():
        raise ValueError(f"SSH identity does not exist: {identity}")
    return SSHConnection(target=target, port=port, identity=identity)


def _connection_when_ready(
    pod_id: str,
    *,
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


def _remote_sha256(
    connection: SSHConnection,
    known_hosts: Path,
    remote_path: str,
) -> str:
    result = subprocess.run(
        [
            "ssh",
            *connection.ssh_options(known_hosts),
            connection.target,
            shlex.join(["sha256sum", remote_path]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.split(maxsplit=1)[0] if result.stdout.strip() else ""
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"unexpected remote SHA-256 output: {result.stdout!r}")
    return digest


def _assert_remote_paths_absent(
    connection: SSHConnection,
    known_hosts: Path,
    paths: tuple[str, ...],
) -> None:
    command = " && ".join(f"test ! -e {shlex.quote(path)}" for path in paths)
    result = subprocess.run(
        [
            "ssh",
            *connection.ssh_options(known_hosts),
            connection.target,
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("a remote XG workspace or result path already exists")


def _download_result(
    connection: SSHConnection,
    known_hosts: Path,
    remote_archive: str,
    destination: Path,
) -> VerifiedArchive:
    remote_digest = _remote_sha256(connection, known_hosts, remote_archive)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"refusing to overwrite partial download: {partial}")
    try:
        subprocess.run(
            [
                "scp",
                *connection.scp_options(known_hosts),
                f"{connection.target}:{remote_archive}",
                str(partial),
            ],
            check=True,
        )
        local_digest = sha256_file(partial)
        if local_digest != remote_digest:
            raise RuntimeError(
                "downloaded result SHA-256 mismatch: "
                f"remote={remote_digest} local={local_digest}"
            )
        verified = verify_result_archive(partial)
        os.replace(partial, destination)
        return VerifiedArchive(
            path=destination,
            sha256=verified.sha256,
            root=verified.root,
            manifest=verified.manifest,
            exit_status=verified.exit_status,
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    archive = args.archive.expanduser().resolve()
    try:
        workspace = verify_workspace_archive(archive)
    except ValueError as error:
        raise SystemExit(f"refusing unsafe XG workspace archive: {error}") from error
    try:
        launch = _load_launch_receipt(args.launch_receipt, workspace)
    except ValueError as error:
        raise SystemExit(f"refusing unsafe XG launch receipt: {error}") from error
    pod_id = launch.pod_id

    download_dir = args.download_dir.expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    destination = download_dir / f"{pod_id}-xg-ce-4096-65536.tar.gz"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite local result archive: {destination}")

    remote_parent = f"/workspace/Jim-xg-{pod_id}"
    remote_archive = f"/workspace/jim-xg-workspace-{pod_id}.tar.gz"
    remote_root = f"{remote_parent}/{workspace.root.as_posix()}"
    remote_output = f"/workspace/jim-xg-results/{pod_id}"
    remote_result = f"/workspace/jim-xg-result-{pod_id}.tar.gz"
    remote_partial = f"{remote_result}.partial"

    with tempfile.TemporaryDirectory(
        prefix=f"jim-xg-known-hosts-{pod_id}-"
    ) as temporary:
        known_hosts = Path(temporary) / "known_hosts"
        try:
            _verify_live_pod(pod_id)
        except (subprocess.CalledProcessError, ValueError) as error:
            raise RuntimeError(
                "the paid pod does not match its frozen XG launch receipt"
            ) from error
        connection = _connection_when_ready(
            pod_id,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            known_hosts=known_hosts,
        )
        _assert_remote_paths_absent(
            connection,
            known_hosts,
            (
                remote_parent,
                remote_archive,
                remote_output,
                remote_result,
                remote_partial,
            ),
        )
        subprocess.run(
            [
                "scp",
                *connection.scp_options(known_hosts),
                str(archive),
                f"{connection.target}:{remote_archive}",
            ],
            check=True,
        )
        uploaded_digest = _remote_sha256(connection, known_hosts, remote_archive)
        if uploaded_digest != workspace.sha256:
            raise RuntimeError(
                "uploaded workspace SHA-256 mismatch: "
                f"local={workspace.sha256} remote={uploaded_digest}"
            )
        setup = " && ".join(
            (
                f"mkdir {shlex.quote(remote_parent)}",
                (
                    "tar --no-same-owner -xzf "
                    f"{shlex.quote(remote_archive)} -C {shlex.quote(remote_parent)}"
                ),
                f"test -d {shlex.quote(remote_root)}",
            )
        )
        subprocess.run(
            [
                "ssh",
                *connection.ssh_options(known_hosts),
                connection.target,
                setup,
            ],
            check=True,
        )
        run_command = " && ".join(
            (
                f"cd {shlex.quote(remote_root)}",
                shlex.join(
                    [
                        "bash",
                        "benchmarks/xg/runpod/run_on_pod.sh",
                        "--output-root",
                        remote_output,
                        "--result-archive",
                        remote_result,
                    ]
                ),
            )
        )
        remote_run = subprocess.run(
            [
                "ssh",
                *connection.ssh_options(known_hosts),
                connection.target,
                run_command,
            ],
            check=False,
        )
        try:
            result = _download_result(
                connection,
                known_hosts,
                remote_result,
                destination,
            )
            binding_pairs = (
                (
                    "source revision",
                    result.manifest.get("source_revision"),
                    workspace.manifest.get("source_revision"),
                ),
                (
                    "workspace payload tree",
                    result.manifest.get("workspace_payload_tree_sha256"),
                    workspace.manifest.get("payload_tree_sha256"),
                ),
            )
            mismatches = [
                label
                for label, observed, expected in binding_pairs
                if observed != expected
            ]
            if mismatches:
                raise RuntimeError(
                    "retrieved result does not bind the uploaded workspace: "
                    + ", ".join(mismatches)
                )
        except Exception as error:
            print(
                "Result retrieval or binding verification failed. The 120-minute "
                "automatic deletion guard "
                f"remains active for pod {pod_id}.",
                file=sys.stderr,
            )
            raise RuntimeError(
                "could not retrieve a verified XG result archive"
            ) from error

    if not args.keep_pod:
        subprocess.run(["runpodctl", "pod", "delete", pod_id], check=True)
    else:
        print(
            f"Pod retained by request; automatic deletion remains active: {pod_id}",
            file=sys.stderr,
        )

    print(destination)
    if result.exit_status != remote_run.returncode:
        raise RuntimeError(
            "remote SSH status and archived workflow status differ: "
            f"ssh={remote_run.returncode} archive={result.exit_status}"
        )
    if result.exit_status != 0:
        raise RuntimeError(
            f"XG workflow failed with status {result.exit_status}; artifacts: {destination}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
