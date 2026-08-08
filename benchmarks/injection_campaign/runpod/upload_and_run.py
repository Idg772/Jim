"""Upload a curated workspace, run/resume the campaign, and retrieve results."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from benchmarks.device_parallel_nss.runpod.upload_and_run import (
    _connection_when_ready,
    _remote_sha256,
    _sha256_file,
    _verify_workspace_archive,
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pod_id")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--download-dir", type=Path, default=Path("campaign-results"))
    parser.add_argument("--n-injections", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=260728265)
    parser.add_argument("--retry-count", type=_nonnegative_int, default=2)
    parser.add_argument("--timeout", type=_positive_int, default=1200)
    parser.add_argument("--poll-interval", type=_positive_int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9_-]+", args.pod_id) is None:
        raise SystemExit(f"unexpected pod id: {args.pod_id!r}")
    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"workspace archive does not exist: {archive}")
    try:
        package = _verify_workspace_archive(archive)
    except ValueError as error:
        raise SystemExit(f"refusing unsafe workspace archive: {error}") from error
    campaign_paths = {
        entry["path"] for entry in package["candidate"]["files"]
    }
    required = {
        "benchmarks/injection_campaign/run_campaign.py",
        "benchmarks/injection_campaign/runpod/run_on_pod.sh",
    }
    if not required <= campaign_paths:
        raise SystemExit("workspace package predates the injection campaign files")

    destination_dir = args.download_dir.expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_archive_sha256 = _sha256_file(archive)
    remote_root = f"/workspace/Jim-campaign-{args.pod_id}"
    remote_archive = f"/workspace/jim-campaign-workspace-{args.pod_id}.tar.gz"
    remote_output = f"/workspace/jim-injection-campaign-{args.pod_id}"
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
        if _remote_sha256(connection, known_hosts, remote_archive) != local_archive_sha256:
            raise RuntimeError("uploaded workspace SHA-256 mismatch")
        setup = " && ".join(
            [
                f"test ! -e {shlex.quote(remote_root)}",
                f"mkdir -p {shlex.quote(remote_root)}",
                (
                    "tar --no-same-owner -xzf "
                    f"{shlex.quote(remote_archive)} -C {shlex.quote(remote_root)}"
                ),
            ]
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
        command = shlex.join(
            [
                "bash",
                "benchmarks/injection_campaign/runpod/run_on_pod.sh",
                "--output-dir",
                remote_output,
                "--n-injections",
                str(args.n_injections),
                "--seed",
                str(args.seed),
                "--retry-count",
                str(args.retry_count),
            ]
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
        remote_digest = _remote_sha256(connection, known_hosts, remote_results)
        destination = destination_dir / f"{args.pod_id}-campaign.tar.gz"
        subprocess.run(
            [
                "scp",
                *scp_options,
                f"{connection.target}:{remote_results}",
                str(destination),
            ],
            check=True,
        )
        if _sha256_file(destination) != remote_digest:
            raise RuntimeError("downloaded result SHA-256 mismatch")
    print(destination)
    print(f"Results are safe locally. Delete the pod: runpodctl pod delete {args.pod_id}")
    if result.returncode != 0:
        raise SystemExit(
            f"campaign exited with status {result.returncode}; the partial results were downloaded"
        )


if __name__ == "__main__":
    main()
