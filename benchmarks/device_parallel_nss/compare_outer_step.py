"""Compare the paper-style and replicated NSS executors on one GPU host."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PAPER_BASELINE = "86335bdb1e7ef6191937dd17b2ca53edbb1d899f"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--baseline-ref", default=PAPER_BASELINE)
    parser.add_argument("--candidate-ref", default="HEAD")
    parser.add_argument("--baseline-label", default="paper-baseline")
    parser.add_argument("--candidate-label", default="ours")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device-counts", nargs="+", type=_positive_int, default=[1, 4]
    )
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-live", type=_positive_int, default=512)
    parser.add_argument("--n-delete", type=_positive_int, default=64)
    parser.add_argument("--dims", type=_positive_int, default=15)
    parser.add_argument("--inner-steps-per-dim", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_positive_int, default=20)
    parser.add_argument("--iterations", type=_positive_int, default=1000)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--simulate-cpu",
        action="store_true",
        help="Use logical CPU devices for a local orchestration smoke test.",
    )
    return parser.parse_args()


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def _resolve_revision(repository: Path, revision: str) -> str:
    result = _run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repository,
    )
    return result.stdout.strip()


def _extract_revision(repository: Path, revision: str, destination: Path) -> None:
    destination.mkdir(parents=True)
    archive_path = destination.parent / f"{destination.name}.tar"
    with archive_path.open("wb") as archive_stream:
        subprocess.run(
            ["git", "archive", "--format=tar", revision],
            cwd=repository,
            check=True,
            stdout=archive_stream,
        )

    destination_root = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination_root):
                raise RuntimeError(f"unsafe path in git archive: {member.name}")
        archive.extractall(destination, filter="data")
    archive_path.unlink()


def _capture(command: list[str], cwd: Path) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} not found",
        }
    result = _run([executable, *command[1:]], cwd=cwd, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _hardware_snapshot(repository: Path) -> dict[str, Any]:
    commands = {
        "nvidia_smi_list": ["nvidia-smi", "-L"],
        "nvidia_smi_query": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version,pstate,power.limit",
            "--format=csv,noheader",
        ],
        "nvidia_smi_topology": ["nvidia-smi", "topo", "-m"],
        "nvidia_smi_nvlink": ["nvidia-smi", "nvlink", "--status"],
        "cpu": ["lscpu"],
        "kernel": ["uname", "-a"],
    }
    return {name: _capture(command, repository) for name, command in commands.items()}


def _visible_gpu_count(snapshot: dict[str, Any]) -> int:
    output = snapshot["nvidia_smi_list"]["stdout"]
    return sum(line.startswith("GPU ") for line in output.splitlines())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "run"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[index]


def _summarise_group(reports: list[dict[str, Any]]) -> dict[str, Any]:
    repeat_medians = [
        report["timing"]["warmed_step_ms"]["median"] for report in reports
    ]
    samples = [
        sample
        for report in reports
        for sample in report["timing"]["warmed_step_ms"]["samples"]
    ]
    compile_seconds = [report["timing"]["compile_seconds"] for report in reports]
    collective_counts = {
        operation: sorted(
            {report["collectives"][operation]["count"] for report in reports}
        )
        for operation in ("all-gather", "all-reduce", "collective-permute")
    }
    return {
        "repeats": len(reports),
        "repeat_medians_ms": repeat_medians,
        "median_step_ms": statistics.median(repeat_medians),
        "pooled_step_ms": {
            "samples": len(samples),
            "median": statistics.median(samples),
            "p05": _percentile(samples, 0.05),
            "p95": _percentile(samples, 0.95),
        },
        "median_compile_seconds": statistics.median(compile_seconds),
        "collective_counts": collective_counts,
    }


def _build_summary(
    reports: list[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
    device_counts: list[int],
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        key = (
            report["implementation"]["label"],
            report["workload"]["n_devices"],
        )
        grouped[key].append(report)

    groups = {
        label: {
            str(device_count): _summarise_group(grouped[(label, device_count)])
            for device_count in device_counts
        }
        for label in (baseline_label, candidate_label)
    }
    comparisons: dict[str, Any] = {"candidate_vs_baseline": {}, "scaling": {}}
    for device_count in device_counts:
        baseline_ms = groups[baseline_label][str(device_count)]["median_step_ms"]
        candidate_ms = groups[candidate_label][str(device_count)]["median_step_ms"]
        comparisons["candidate_vs_baseline"][str(device_count)] = {
            "speedup": baseline_ms / candidate_ms,
            "latency_reduction_percent": 100.0 * (1.0 - candidate_ms / baseline_ms),
        }

    if 1 in device_counts:
        for label in (baseline_label, candidate_label):
            one_gpu_ms = groups[label]["1"]["median_step_ms"]
            comparisons["scaling"][label] = {}
            for device_count in device_counts:
                if device_count == 1:
                    continue
                multi_gpu_ms = groups[label][str(device_count)]["median_step_ms"]
                speedup = one_gpu_ms / multi_gpu_ms
                comparisons["scaling"][label][str(device_count)] = {
                    "speedup": speedup,
                    "parallel_efficiency": speedup / device_count,
                }

    return {"groups": groups, "comparisons": comparisons}


def _format_counts(counts: dict[str, list[int]]) -> str:
    return "/".join(
        ",".join(str(count) for count in counts[operation])
        for operation in ("all-gather", "all-reduce", "collective-permute")
    )


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    baseline_label: str,
    candidate_label: str,
    device_counts: list[int],
) -> None:
    lines = [
        "# Device-parallel NSS comparison",
        "",
        (
            "The table reports the median of fresh-process median step times. "
            "Compilation is excluded from step timing."
        ),
        "",
        "| Implementation | GPUs | Repeats | Step median [ms] | P05 [ms] | P95 [ms] | Compile [s] | AG/AR/CP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label in (baseline_label, candidate_label):
        for device_count in device_counts:
            group = summary["groups"][label][str(device_count)]
            pooled = group["pooled_step_ms"]
            lines.append(
                f"| {label} | {device_count} | {group['repeats']} | "
                f"{group['median_step_ms']:.6f} | {pooled['p05']:.6f} | "
                f"{pooled['p95']:.6f} | {group['median_compile_seconds']:.3f} | "
                f"{_format_counts(group['collective_counts'])} |"
            )

    lines.extend(
        [
            "",
            "## Direct implementation comparison",
            "",
            "| GPUs | Candidate speedup over baseline | Candidate latency reduction |",
            "| ---: | ---: | ---: |",
        ]
    )
    direct = summary["comparisons"]["candidate_vs_baseline"]
    for device_count in device_counts:
        comparison = direct[str(device_count)]
        lines.append(
            f"| {device_count} | {comparison['speedup']:.3f}x | "
            f"{comparison['latency_reduction_percent']:.2f}% |"
        )

    scaling = summary["comparisons"]["scaling"]
    if scaling:
        lines.extend(
            [
                "",
                "## Device scaling",
                "",
                "| Implementation | GPUs | Speedup from one GPU | Parallel efficiency |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label in (baseline_label, candidate_label):
            for device_count, measurement in scaling[label].items():
                lines.append(
                    f"| {label} | {device_count} | {measurement['speedup']:.3f}x | "
                    f"{100.0 * measurement['parallel_efficiency']:.2f}% |"
                )
    path.write_text("\n".join(lines) + "\n")


def _validate_report(
    report: dict[str, Any],
    *,
    expected_backend: str,
    expected_devices: int,
    expected_revision: str,
) -> None:
    environment = report["environment"]
    if environment["backend"] != expected_backend:
        raise RuntimeError(
            f"expected JAX backend {expected_backend}, got {environment['backend']}"
        )
    if environment["local_device_count"] != expected_devices:
        raise RuntimeError(
            f"expected {expected_devices} visible devices, got "
            f"{environment['local_device_count']}"
        )
    if environment["git"]["revision"] != expected_revision:
        raise RuntimeError(
            f"expected revision {expected_revision}, got "
            f"{environment['git']['revision']}"
        )


def main() -> None:
    args = _parse_args()
    repository = args.repository.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    device_counts = list(dict.fromkeys(args.device_counts))
    if output_dir.exists():
        raise SystemExit(f"--output-dir already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()

    baseline_revision = _resolve_revision(repository, args.baseline_ref)
    candidate_revision = _resolve_revision(repository, args.candidate_ref)
    if baseline_revision == candidate_revision:
        raise SystemExit("baseline and candidate resolve to the same commit")

    hardware = _hardware_snapshot(repository)
    if not args.simulate_cpu:
        visible_gpus = _visible_gpu_count(hardware)
        required_gpus = max(device_counts)
        if visible_gpus < required_gpus:
            raise SystemExit(
                f"benchmark requires {required_gpus} GPUs, but nvidia-smi found "
                f"{visible_gpus}"
            )

    benchmark_script = Path(__file__).with_name("benchmark_outer_step.py")
    reports: list[dict[str, Any]] = []
    run_order: list[dict[str, Any]] = []
    labels = (args.baseline_label, args.candidate_label)
    revisions = {
        args.baseline_label: baseline_revision,
        args.candidate_label: candidate_revision,
    }

    with tempfile.TemporaryDirectory(prefix="jim-nss-comparison-") as temporary:
        temporary_root = Path(temporary)
        roots = {
            args.baseline_label: temporary_root / "baseline",
            args.candidate_label: temporary_root / "candidate",
        }
        for label in labels:
            _extract_revision(repository, revisions[label], roots[label])

        for repeat in range(args.repeats):
            seed = args.seed_start + repeat
            implementation_order = (
                labels if repeat % 2 == 0 else tuple(reversed(labels))
            )
            for device_count in device_counts:
                for label in implementation_order:
                    slug = _slug(label)
                    stem = f"{slug}-g{device_count}-seed{seed:04d}-repeat{repeat:02d}"
                    command = [
                        sys.executable,
                        str(benchmark_script),
                        "--implementation-root",
                        str(roots[label]),
                        "--implementation-label",
                        label,
                        "--implementation-revision",
                        revisions[label],
                        "--n-live",
                        str(args.n_live),
                        "--n-delete",
                        str(args.n_delete),
                        "--dims",
                        str(args.dims),
                        "--inner-steps-per-dim",
                        str(args.inner_steps_per_dim),
                        "--warmup",
                        str(args.warmup),
                        "--iterations",
                        str(args.iterations),
                        "--dtype",
                        args.dtype,
                        "--n-devices",
                        str(device_count),
                        "--seed",
                        str(seed),
                    ]
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "JAX_ENABLE_COMPILATION_CACHE": "0",
                            "PYTHONHASHSEED": str(seed),
                            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        }
                    )
                    if args.simulate_cpu:
                        command.append("--simulate-cpu")
                        environment["JAX_PLATFORMS"] = "cpu"
                        environment.pop("CUDA_VISIBLE_DEVICES", None)
                        expected_backend = "cpu"
                    else:
                        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                            str(index) for index in range(device_count)
                        )
                        environment["JAX_PLATFORMS"] = "cuda"
                        expected_backend = "gpu"

                    run_order.append(
                        {
                            "label": label,
                            "revision": revisions[label],
                            "devices": device_count,
                            "seed": seed,
                            "command": command,
                        }
                    )
                    print(
                        f"repeat={repeat + 1}/{args.repeats} label={label} "
                        f"devices={device_count} seed={seed}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result = _run(
                        command,
                        cwd=repository,
                        env=environment,
                        check=False,
                    )
                    (raw_dir / f"{stem}.stderr.log").write_text(result.stderr)
                    if result.returncode != 0:
                        (raw_dir / f"{stem}.stdout.log").write_text(result.stdout)
                        raise RuntimeError(
                            f"benchmark failed for {label}, {device_count} devices, "
                            f"seed {seed}; see {stem}.stderr.log"
                        )
                    report = json.loads(result.stdout)
                    _validate_report(
                        report,
                        expected_backend=expected_backend,
                        expected_devices=device_count,
                        expected_revision=revisions[label],
                    )
                    (raw_dir / f"{stem}.json").write_text(
                        json.dumps(report, indent=2, sort_keys=True) + "\n"
                    )
                    reports.append(report)

    dependency_versions = {
        (
            report["environment"]["python"],
            report["environment"]["jax"],
            report["environment"]["jaxlib"],
            report["environment"]["blackjax"],
        )
        for report in reports
    }
    if len(dependency_versions) != 1:
        raise RuntimeError(
            f"dependency versions changed between runs: {dependency_versions}"
        )

    summary = _build_summary(
        reports,
        args.baseline_label,
        args.candidate_label,
        device_counts,
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "repository": str(repository),
        "revisions": revisions,
        "configuration": {
            "device_counts": device_counts,
            "repeats": args.repeats,
            "seed_start": args.seed_start,
            "n_live": args.n_live,
            "n_delete": args.n_delete,
            "dims": args.dims,
            "inner_steps_per_dim": args.inner_steps_per_dim,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "dtype": args.dtype,
            "simulate_cpu": args.simulate_cpu,
        },
        "hardware": hardware,
        "run_order": run_order,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_markdown(
        output_dir / "summary.md",
        summary,
        args.baseline_label,
        args.candidate_label,
        device_counts,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
