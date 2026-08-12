"""Run or resume FSM/SwiG recoveries, isolated or in one sequential worker."""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.injection_campaign.common import (
    atomic_write_json,
    load_manifest,
    publication_eligible,
    read_catalogue,
    refresh_status,
    result_dir,
    validate_completed_result,
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
    parser.add_argument(
        "--injection-id",
        dest="injection_ids",
        action="append",
        type=_nonnegative_int,
        default=None,
        help=(
            "Explicit injection ID to run; repeat for a deterministic sparse "
            "selection. Cannot be combined with a non-default start/stop range."
        ),
    )
    parser.add_argument(
        "--long-lived-worker",
        action="store_true",
        help=(
            "Run selected recoveries sequentially in this Python process, sharing "
            "only JAX/runtime infrastructure between otherwise fresh events."
        ),
    )
    parser.add_argument(
        "--jax-cache-diagnostics",
        action="store_true",
        help="Record compact JAX compilation-cache diagnostics per recovery.",
    )
    parser.add_argument("--keep-success-logs", action="store_true")
    parser.add_argument("--simulate-cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate the P-P and Figure 3 timing products after completion.",
    )
    return parser.parse_args(argv)


def _tail(path: Path, n_lines: int = 40) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-n_lines:]
        )
    except OSError:
        return ""


def _is_complete(
    directory: Path,
    config_sha256: str,
    *,
    injection_id: int | None = None,
    catalogue_row: dict[str, Any] | None = None,
) -> bool:
    try:
        validate_completed_result(
            directory,
            config_sha256,
            injection_id=injection_id,
            catalogue_row=catalogue_row,
        )
    except (OSError, TypeError, ValueError):
        return False
    return True


def _selected_injection_ids(args: argparse.Namespace, n_injections: int) -> list[int]:
    explicit = args.injection_ids
    if explicit is not None:
        if args.start != 0 or args.stop is not None:
            raise SystemExit("--injection-id cannot be combined with --start/--stop")
        if len(set(explicit)) != len(explicit):
            raise SystemExit("--injection-id values must be unique")
        invalid = [value for value in explicit if value < 0 or value >= n_injections]
        if invalid:
            raise SystemExit(
                "injection IDs outside the frozen campaign: "
                + ", ".join(map(str, invalid))
            )
        if not explicit:
            raise SystemExit("explicit injection selection is empty")
        return list(explicit)

    stop = n_injections if args.stop is None else min(args.stop, n_injections)
    if args.start >= stop:
        raise SystemExit(f"empty injection range [{args.start}, {stop})")
    return list(range(args.start, stop))


def _worker_injection_args(
    args: argparse.Namespace,
    campaign_dir: Path,
    injection_id: int,
    cache_dir: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        campaign_dir=campaign_dir,
        injection_id=injection_id,
        jax_compilation_cache_dir=cache_dir,
        simulate_cpu=bool(args.simulate_cpu),
        verbose=bool(args.verbose),
        force=False,
        jax_cache_diagnostics=bool(args.jax_cache_diagnostics),
    )


def _run_worker_attempt(
    *,
    args: argparse.Namespace,
    campaign_dir: Path,
    injection_id: int,
    cache_dir: Path,
    runtime: Any,
    log_path: Path,
) -> int:
    from benchmarks.injection_campaign.run_injection import run_injection

    injection_args = _worker_injection_args(args, campaign_dir, injection_id, cache_dir)
    with (
        log_path.open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            summary = run_injection(injection_args, runtime=runtime)
            print(json.dumps(summary, indent=2, sort_keys=True))
        except SystemExit as error:
            traceback.print_exc()
            return error.code if isinstance(error.code, int) else 1
        except Exception:  # noqa: BLE001 - one failed event must remain resumable
            traceback.print_exc()
            return 1
    return 0


def run_campaign(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    n_injections = int(manifest["n_injections"])
    catalogue = read_catalogue(campaign_dir / manifest["catalogue"]["path"])
    selected = _selected_injection_ids(args, n_injections)
    pending = [
        injection_id
        for injection_id in selected
        if not _is_complete(
            result_dir(campaign_dir, injection_id),
            manifest["config_sha256"],
            injection_id=injection_id,
            catalogue_row=catalogue[injection_id],
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
    runtime = None
    if args.long_lived_worker and pending:
        from benchmarks.injection_campaign.run_injection import (
            prepare_injection_runtime,
        )

        runtime = prepare_injection_runtime(
            campaign_dir,
            simulate_cpu=bool(args.simulate_cpu),
            jax_compilation_cache_dir=cache_dir,
            jax_cache_diagnostics=bool(args.jax_cache_diagnostics),
        )
        print(
            "Long-lived worker initialized in "
            f"{runtime.initialization_seconds:.3f}s for IDs "
            + ", ".join(map(str, pending)),
            flush=True,
        )
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
            print(
                f"[{position}/{len(pending)}] injection {injection_id:03d}, "
                f"attempt {retry_index + 1}/{args.retry_count + 1}",
                flush=True,
            )
            if runtime is not None:
                returncode = _run_worker_attempt(
                    args=args,
                    campaign_dir=campaign_dir,
                    injection_id=injection_id,
                    cache_dir=cache_dir,
                    runtime=runtime,
                    log_path=log_path,
                )
            else:
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
                if args.jax_cache_diagnostics:
                    command.append("--jax-cache-diagnostics")
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        cwd=Path(__file__).resolve().parents[2],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                returncode = result.returncode
            (directory / "RUNNING").unlink(missing_ok=True)
            if returncode == 0 and _is_complete(
                directory,
                manifest["config_sha256"],
                injection_id=injection_id,
                catalogue_row=catalogue[injection_id],
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
                    "returncode": returncode,
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
        if not publication_eligible(manifest):
            print(
                "Targeted non-iid stress catalogue: not publishing P-P or "
                "Figure 3 products.",
                file=sys.stderr,
            )
            return 1 if failures else 0
        incomplete_ids = [
            injection_id
            for injection_id in range(n_injections)
            if not _is_complete(
                result_dir(campaign_dir, injection_id),
                manifest["config_sha256"],
                injection_id=injection_id,
                catalogue_row=catalogue[injection_id],
            )
        ]
        if incomplete_ids:
            missing = ", ".join(str(injection_id) for injection_id in incomplete_ids)
            print(
                "Selected leading-ID set is incomplete; not publishing the "
                f"paper P-P or timing results. Incomplete IDs: {missing}",
                file=sys.stderr,
            )
        else:
            for module in (
                "benchmarks.injection_campaign.plot_pp",
                "benchmarks.injection_campaign.plot_timing",
            ):
                plot_result = subprocess.run(
                    [sys.executable, "-m", module, str(campaign_dir)],
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
