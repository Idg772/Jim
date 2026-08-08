"""Run or resume a sequence of isolated FSM/SwiG injection recoveries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.injection_campaign.common import (
    atomic_write_json,
    file_sha256,
    load_manifest,
    refresh_status,
    result_dir,
)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--start", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--stop",
        type=_positive_int,
        default=None,
        help="Exclusive injection ID; defaults to the catalogue length.",
    )
    parser.add_argument(
        "--retry-count",
        type=_nonnegative_int,
        default=1,
        help="Additional attempts for each failed recovery.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-success-logs", action="store_true")
    parser.add_argument("--simulate-cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate P-P CSVs and plots after the selected jobs finish.",
    )
    return parser.parse_args(argv)


def _tail(path: Path, n_lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n_lines:])
    except OSError:
        return ""


def _is_complete(directory: Path, config_sha256: str) -> bool:
    summary_path = directory / "summary.json"
    posterior_path = directory / "posterior.npz"
    if not summary_path.is_file() or not posterior_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    posterior = summary.get("posterior", {})
    return (
        summary.get("config_sha256") == config_sha256
        and isinstance(posterior.get("sha256"), str)
        and file_sha256(posterior_path) == posterior["sha256"]
    )


def run_campaign(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    n_injections = int(manifest["n_injections"])
    stop = n_injections if args.stop is None else min(args.stop, n_injections)
    if args.start >= stop:
        raise SystemExit(f"empty injection range [{args.start}, {stop})")
    selected = list(range(args.start, stop))
    pending = [
        injection_id
        for injection_id in selected
        if not _is_complete(
            result_dir(campaign_dir, injection_id), manifest["config_sha256"]
        )
    ]
    print(
        f"Campaign {campaign_dir}: {len(selected)} selected, "
        f"{len(selected) - len(pending)} already complete, {len(pending)} pending",
        flush=True,
    )
    if args.dry_run:
        print("Pending IDs: " + ", ".join(map(str, pending)))
        return 0

    failures: list[int] = []
    cache_dir = campaign_dir / ".jax-cache"
    for position, injection_id in enumerate(pending, start=1):
        directory = result_dir(campaign_dir, injection_id)
        directory.mkdir(parents=True, exist_ok=True)
        success = False
        for retry_index in range(args.retry_count + 1):
            previous_attempts = len(list(directory.glob("attempt-*.failed.log")))
            attempt = previous_attempts + 1
            log_path = directory / f"attempt-{attempt:02d}.log"
            atomic_write_json(
                directory / "RUNNING",
                {
                    "started_at_utc": datetime.now(UTC).isoformat(),
                    "attempt": attempt,
                },
            )
            command = [
                sys.executable,
                "-m",
                "benchmarks.injection_campaign.run_injection",
                str(campaign_dir),
                str(injection_id),
                "--jax-compilation-cache-dir",
                str(cache_dir),
            ]
            if args.simulate_cpu:
                command.append("--simulate-cpu")
            if args.verbose:
                command.append("--verbose")
            print(
                f"[{position}/{len(pending)}] injection {injection_id:03d}, "
                f"attempt {retry_index + 1}/{args.retry_count + 1}",
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            (directory / "RUNNING").unlink(missing_ok=True)
            if result.returncode == 0 and _is_complete(
                directory, manifest["config_sha256"]
            ):
                success = True
                (directory / "failure.json").unlink(missing_ok=True)
                if args.keep_success_logs:
                    log_path.rename(directory / f"attempt-{attempt:02d}.success.log")
                else:
                    log_path.unlink(missing_ok=True)
                print(f"  completed injection {injection_id:03d}", flush=True)
                break
            failed_log = directory / f"attempt-{attempt:02d}.failed.log"
            log_path.rename(failed_log)
            error = _tail(failed_log)
            atomic_write_json(
                directory / "failure.json",
                {
                    "failed_at_utc": datetime.now(UTC).isoformat(),
                    "attempt": attempt,
                    "returncode": result.returncode,
                    "log": failed_log.name,
                    "error": error,
                },
            )
            print(f"  failed injection {injection_id:03d}; {failed_log}", flush=True)
            refresh_status(campaign_dir, n_injections)
        if not success:
            failures.append(injection_id)
            if args.fail_fast:
                break
        refresh_status(campaign_dir, n_injections)

    rows = refresh_status(campaign_dir, n_injections)
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("complete", "failed", "pending", "running", "invalid")
    }
    print("Status: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    if args.plot:
        completed = counts["complete"]
        if completed == 0:
            print("No completed recoveries; skipping P-P plots", file=sys.stderr)
        else:
            plot_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.injection_campaign.plot_pp",
                    str(campaign_dir),
                ],
                cwd=Path(__file__).resolve().parents[2],
                check=False,
            )
            if plot_result.returncode != 0:
                failures.append(-1)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_campaign(_parse_args(argv)))


if __name__ == "__main__":
    main()
