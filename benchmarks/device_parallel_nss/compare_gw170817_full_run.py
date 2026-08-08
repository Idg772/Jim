"""Run the paired four-GPU GW170817 baseline/candidate benchmark matrix.

The benchmark harness is deliberately taken from the working tree while each
Jim implementation is imported from an exported Git snapshot.  This keeps an
uncommitted benchmark harness usable without accidentally benchmarking the
working-tree implementation for both sides of the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PAPER_BASELINE = "86335bdb1e7ef6191937dd17b2ca53edbb1d899f"
RUNNER_BENCHMARK = "gw170817-full-swig-4gpu"
DEVICE_COUNT = 4
DEFAULT_SEEDS = (0, 1, 2)
VISIBLE_DEVICES = "0,1,2,3"
ALIGNED_WORKLOAD = "aligned-11d"
PAPER_WORKLOAD = "paper-15d"
WORKLOAD_DIMENSIONS = {
    ALIGNED_WORKLOAD: 11,
    PAPER_WORKLOAD: 15,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--baseline-ref", default=PAPER_BASELINE)
    parser.add_argument("--candidate-ref", default="HEAD")
    parser.add_argument("--baseline-label", default="paper-baseline")
    parser.add_argument("--candidate-label", default="ours")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workload",
        choices=tuple(WORKLOAD_DIMENSIONS),
        default=ALIGNED_WORKLOAD,
        help=(
            "Scientific workload to run. The historical aligned-spin workload "
            "remains the default; paper-15d selects the full precessing-tidal "
            "parameterization."
        ),
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        help=(
            "Existing frozen GW170817 .npz, or a path at which it should be "
            "prepared. The default is OUTPUT_DIR/data/gw170817.npz."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=(
            "Shared seeds. Implementations alternate baseline/candidate order "
            "for successive seeds. Use '--seeds 0' for an orchestration smoke."
        ),
    )
    parser.add_argument(
        "--runner-script",
        type=Path,
        default=Path(__file__).with_name("benchmark_gw170817_full_run.py"),
        help="Current benchmark harness to run against both exported snapshots.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used for every fresh benchmark process.",
    )
    parser.add_argument(
        "--skip-data-prepare",
        action="store_true",
        help="Require --data-file to exist instead of fetching/preparing it.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Root directory for one steady-state JAX trace per run.",
    )
    parser.add_argument("--profile-warmup-steps", type=_nonnegative_int, default=10)
    parser.add_argument("--profile-steps", type=_positive_int, default=15)
    parser.add_argument(
        "--slice-data-dir",
        type=Path,
        default=None,
        help="Root directory for per-chain, per-slice NPZ artifacts.",
    )
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        default=None,
        help="Root directory for nvidia-smi dmon traces during profile windows.",
    )
    parser.add_argument(
        "--jax-compilation-cache-dir",
        type=Path,
        default=None,
        help="Root directory containing isolated persistent caches per implementation.",
    )
    parser.add_argument(
        "--persistent-cache-probe",
        action="store_true",
        help=(
            "After the matrix, run one candidate outer step in a fresh process "
            "against the populated persistent cache."
        ),
    )
    parser.add_argument(
        "--simulate-cpu",
        action="store_true",
        help=(
            "Expose four logical CPU devices for harness smoke testing. Results "
            "from this mode are not GPU benchmark evidence."
        ),
    )
    args = parser.parse_args(argv)
    if args.telemetry_dir is not None and args.profile_dir is None:
        parser.error("--telemetry-dir requires --profile-dir")
    if args.persistent_cache_probe and args.jax_compilation_cache_dir is None:
        parser.error("--persistent-cache-probe requires --jax-compilation-cache-dir")
    return args


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_harness(
    runner_script: Path,
    harness_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Freeze the runner and its benchmark-local waveform implementation."""

    paper_model_script = runner_script.with_name("paper_model.py")
    paper_model_basis_script = runner_script.with_name("paper_model_basis.py")
    if not paper_model_script.is_file():
        raise FileNotFoundError(
            "the GW170817 benchmark runner requires a sibling paper_model.py; "
            f"got {paper_model_script}"
        )
    if not paper_model_basis_script.is_file():
        raise FileNotFoundError(
            "the GW170817 benchmark runner requires a sibling "
            f"paper_model_basis.py; got {paper_model_basis_script}"
        )

    harness_dir.mkdir()
    frozen_runner = harness_dir / runner_script.name
    frozen_paper_model = harness_dir / paper_model_script.name
    frozen_paper_model_basis = harness_dir / paper_model_basis_script.name
    shutil.copy2(runner_script, frozen_runner)
    shutil.copy2(paper_model_script, frozen_paper_model)
    shutil.copy2(paper_model_basis_script, frozen_paper_model_basis)

    return frozen_runner, {
        # Retain the original runner fields for consumers of schema version 1.
        "source": str(runner_script),
        "frozen_copy": str(frozen_runner),
        "sha256": _sha256_file(frozen_runner),
        "companions": [
            {
                "module": "paper_model",
                "source": str(paper_model_script),
                "frozen_copy": str(frozen_paper_model),
                "sha256": _sha256_file(frozen_paper_model),
            },
            {
                "module": "paper_model_basis",
                "source": str(paper_model_basis_script),
                "frozen_copy": str(frozen_paper_model_basis),
                "sha256": _sha256_file(frozen_paper_model_basis),
            },
        ],
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _capture(command: Sequence[str], cwd: Path) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} not found",
        }
    result = _run([executable, *command[1:]], cwd=cwd, check=False)
    return {
        "command": list(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _hardware_snapshot(repository: Path) -> dict[str, Any]:
    commands = {
        "nvidia_smi_list": ["nvidia-smi", "-L"],
        "nvidia_smi_query": [
            "nvidia-smi",
            (
                "--query-gpu=index,name,uuid,pci.bus_id,memory.total,"
                "driver_version,pstate,power.limit"
            ),
            "--format=csv,noheader",
        ],
        "nvidia_smi_topology": ["nvidia-smi", "topo", "-m"],
        "nvidia_smi_nvlink": ["nvidia-smi", "nvlink", "--status"],
        "cpu": ["lscpu"],
        "kernel": ["uname", "-a"],
    }
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "commands": {
            name: _capture(command, repository) for name, command in commands.items()
        },
    }


def _physical_gpu_count(snapshot: Mapping[str, Any]) -> int:
    output = snapshot["commands"]["nvidia_smi_list"]["stdout"]
    return sum(line.startswith("GPU ") for line in output.splitlines())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "run"


def _normalised_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the intended per-repeat dimension from a runner config."""

    normalised = dict(config)
    for seed_dependent_key in ("seed", "sha256", "initial_positions_sha256"):
        normalised.pop(seed_dependent_key, None)
    return normalised


def _normalised_environment(report: Mapping[str, Any]) -> dict[str, Any]:
    """Ignore only the deliberately isolated per-implementation cache path."""

    environment = report.get("environment")
    if not isinstance(environment, Mapping):
        raise TypeError("runner report is missing the 'environment' object")
    normalised = dict(environment)
    normalised.pop("jax_compilation_cache_dir", None)
    return normalised


def _device_count(report: Mapping[str, Any]) -> int:
    devices = report.get("devices")
    if not isinstance(devices, Mapping):
        raise TypeError("runner report is missing the 'devices' object")
    for key in ("local_count", "local_device_count", "count"):
        value = devices.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    entries = devices.get("devices")
    if isinstance(entries, list):
        return len(entries)
    raise RuntimeError("runner report does not state its local device count")


def _backend(report: Mapping[str, Any]) -> str:
    devices = report.get("devices")
    environment = report.get("environment")
    candidates: list[Any] = []
    if isinstance(devices, Mapping):
        candidates.extend([devices.get("backend"), devices.get("platform")])
    if isinstance(environment, Mapping):
        candidates.extend([environment.get("backend"), environment.get("jax_backend")])
    for value in candidates:
        if isinstance(value, str):
            return value
    raise RuntimeError("runner report does not state its JAX backend")


def _data_sha256(report: Mapping[str, Any]) -> str:
    data = report.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("runner report is missing the 'data' object")
    value = data.get("sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError("runner report has no valid data.sha256")
    return value


def _report_config(report: Mapping[str, Any]) -> dict[str, Any]:
    config = report.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("runner report is missing the 'config' object")
    return dict(config)


def _implementation(report: Mapping[str, Any]) -> Mapping[str, Any]:
    implementation = report.get("implementation")
    if not isinstance(implementation, Mapping):
        raise TypeError("runner report is missing the 'implementation' object")
    return implementation


def _validate_report(
    report: Mapping[str, Any],
    *,
    expected_backend: str,
    expected_data_sha256: str,
    expected_implementation_root: Path,
    expected_label: str,
    expected_revision: str,
    expected_seed: int,
    expected_workload: str,
) -> None:
    schema_version = report.get("schema_version")
    if schema_version not in (1, 2):
        raise RuntimeError(
            f"expected runner schema version 1 or 2, got {schema_version}"
        )
    if report.get("benchmark") != RUNNER_BENCHMARK:
        raise RuntimeError(
            f"expected benchmark {RUNNER_BENCHMARK!r}, got {report.get('benchmark')!r}"
        )
    report_device_count = _device_count(report)
    if report_device_count != DEVICE_COUNT:
        raise RuntimeError(
            f"expected exactly {DEVICE_COUNT} visible JAX devices, got "
            f"{report_device_count}"
        )
    devices = report["devices"]
    for key in ("requested_count", "global_count"):
        value = devices.get(key)
        if value != DEVICE_COUNT:
            raise RuntimeError(f"expected devices.{key}={DEVICE_COUNT}, got {value}")
    device_entries = devices.get("devices")
    if not isinstance(device_entries, list) or len(device_entries) != DEVICE_COUNT:
        raise RuntimeError(
            f"expected metadata for exactly {DEVICE_COUNT} devices, got "
            f"{device_entries!r}"
        )
    backend = _backend(report)
    acceptable_backends = {expected_backend}
    if expected_backend == "gpu":
        acceptable_backends.add("cuda")
    if backend not in acceptable_backends:
        raise RuntimeError(f"expected JAX backend {expected_backend}, got {backend}")
    if _data_sha256(report) != expected_data_sha256:
        raise RuntimeError(
            "runner did not use the frozen data artifact: expected "
            f"{expected_data_sha256}, got {_data_sha256(report)}"
        )

    implementation = _implementation(report)
    if implementation.get("label") != expected_label:
        raise RuntimeError(
            f"expected implementation label {expected_label!r}, got "
            f"{implementation.get('label')!r}"
        )
    if implementation.get("revision") != expected_revision:
        raise RuntimeError(
            f"expected implementation revision {expected_revision}, got "
            f"{implementation.get('revision')}"
        )
    module_file = implementation.get("module_file")
    if not isinstance(module_file, str):
        raise TypeError("runner report has no implementation.module_file")
    implementation_source = (expected_implementation_root / "src").resolve()
    if not Path(module_file).resolve().is_relative_to(implementation_source):
        raise RuntimeError(
            "runner imported jimgw outside the requested source snapshot: "
            f"{module_file}"
        )

    config = _report_config(report)
    if config.get("seed") != expected_seed:
        raise RuntimeError(
            f"expected seed {expected_seed}, got config.seed={config.get('seed')}"
        )
    if config.get("workload") != expected_workload:
        raise RuntimeError(
            f"expected config.workload={expected_workload!r}, got "
            f"{config.get('workload')!r}"
        )
    expected_dimensions = WORKLOAD_DIMENSIONS[expected_workload]
    if config.get("sampled_dimensions") != expected_dimensions:
        raise RuntimeError(
            f"expected config.sampled_dimensions={expected_dimensions} for "
            f"workload {expected_workload!r}, got "
            f"{config.get('sampled_dimensions')!r}"
        )
    n_devices = config.get("n_devices")
    if n_devices != DEVICE_COUNT:
        raise RuntimeError(f"expected config.n_devices={DEVICE_COUNT}, got {n_devices}")
    reported_config_sha256 = config.get("sha256")
    hash_input = dict(config)
    hash_input.pop("sha256", None)
    hash_input.pop("initial_positions_sha256", None)
    calculated_config_sha256 = _sha256_json(hash_input)
    if reported_config_sha256 != calculated_config_sha256:
        raise RuntimeError(
            "runner config fingerprint is invalid: expected "
            f"{calculated_config_sha256}, got {reported_config_sha256}"
        )


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"runner report has no numeric {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"runner report has non-finite {name}: {number}")
    return number


def _sample_seconds(report: Mapping[str, Any]) -> float:
    timing = report.get("timing_seconds")
    if not isinstance(timing, Mapping):
        raise TypeError("runner report is missing 'timing_seconds'")
    return _number(timing.get("sample_call"), name="timing_seconds.sample_call")


def _post_jit_sample_seconds(report: Mapping[str, Any]) -> float | None:
    """Post-JIT sampling seconds, preferring the paper-convention field.

    Fallback chain for schema-v1 reports: paper_convention block ->
    sample_call - jit_compile_estimate -> sample_call -> None.
    """

    timing = report.get("timing_seconds")
    if not isinstance(timing, Mapping):
        return None
    paper = timing.get("paper_convention")
    if isinstance(paper, Mapping):
        value = paper.get("post_jit_sampling_seconds")
        if isinstance(value, int | float):
            return float(value)
    sample_call = timing.get("sample_call")
    if not isinstance(sample_call, int | float):
        return None
    jit = timing.get("jit_compile_estimate")
    if isinstance(jit, int | float):
        return float(sample_call) - float(jit)
    return float(sample_call)


def _total_seconds(report: Mapping[str, Any]) -> float:
    timing = report.get("timing_seconds")
    if not isinstance(timing, Mapping):
        raise TypeError("runner report is missing 'timing_seconds'")
    return _number(timing.get("total"), name="timing_seconds.total")


def _result_number(report: Mapping[str, Any], *keys: str) -> float | None:
    results = report.get("results")
    if not isinstance(results, Mapping):
        return None
    for key in keys:
        value = results.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            number = float(value)
            return number if math.isfinite(number) else None
    return None


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    samples = list(values)
    if not samples:
        raise ValueError("cannot summarise an empty collection")
    return {
        "samples": samples,
        "minimum": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "maximum": max(samples),
    }


def _build_summary(
    reports: Sequence[Mapping[str, Any]],
    *,
    baseline_label: str,
    candidate_label: str,
    revisions: Mapping[str, str],
    seeds: Sequence[int],
    data_sha256: str,
    workload: str,
) -> dict[str, Any]:
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for report in reports:
        implementation = _implementation(report)
        label = implementation["label"]
        seed = _report_config(report)["seed"]
        key = (label, seed)
        if key in by_key:
            raise RuntimeError(f"duplicate report for label={label}, seed={seed}")
        by_key[key] = report

    expected_keys = {
        (label, seed) for label in (baseline_label, candidate_label) for seed in seeds
    }
    missing = expected_keys - set(by_key)
    unexpected = set(by_key) - expected_keys
    if missing or unexpected:
        raise RuntimeError(
            f"incomplete report matrix; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    environments = {
        _canonical_json(_normalised_environment(report)) for report in reports
    }
    if len(environments) != 1:
        raise RuntimeError("software environment changed between benchmark runs")
    device_manifests = {_canonical_json(report.get("devices")) for report in reports}
    if len(device_manifests) != 1:
        raise RuntimeError("JAX device metadata changed between benchmark runs")

    normalised_configs = {
        _canonical_json(_normalised_config(_report_config(report)))
        for report in reports
    }
    if len(normalised_configs) != 1:
        raise RuntimeError("benchmark configuration changed between runs")
    frozen_config = json.loads(next(iter(normalised_configs)))

    pairs: list[dict[str, Any]] = []
    for pair_index, seed in enumerate(seeds):
        baseline = by_key[(baseline_label, seed)]
        candidate = by_key[(candidate_label, seed)]
        if _report_config(baseline) != _report_config(candidate):
            raise RuntimeError(
                f"baseline and candidate configs differ for shared seed {seed}"
            )
        baseline_seconds = _sample_seconds(baseline)
        candidate_seconds = _sample_seconds(candidate)
        baseline_post_jit_seconds = _post_jit_sample_seconds(baseline)
        candidate_post_jit_seconds = _post_jit_sample_seconds(candidate)
        speedup = baseline_seconds / candidate_seconds
        order = (
            [baseline_label, candidate_label]
            if pair_index % 2 == 0
            else [candidate_label, baseline_label]
        )
        pairs.append(
            {
                "seed": seed,
                "order": order,
                "baseline_sample_call_seconds": baseline_seconds,
                "candidate_sample_call_seconds": candidate_seconds,
                "baseline_post_jit_sample_seconds": baseline_post_jit_seconds,
                "candidate_post_jit_sample_seconds": candidate_post_jit_seconds,
                "candidate_speedup": speedup,
                "candidate_time_reduction_percent": 100.0
                * (1.0 - candidate_seconds / baseline_seconds),
                "baseline_results": baseline.get("results"),
                "candidate_results": candidate.get("results"),
            }
        )

    groups: dict[str, Any] = {}
    for label in (baseline_label, candidate_label):
        implementation_reports = [by_key[(label, seed)] for seed in seeds]
        post_jit_values = [
            seconds
            for seconds in (
                _post_jit_sample_seconds(report) for report in implementation_reports
            )
            if seconds is not None
        ]
        groups[label] = {
            "revision": revisions[label],
            "seeds": list(seeds),
            "sample_call_seconds": _distribution(
                _sample_seconds(report) for report in implementation_reports
            ),
            "post_jit_sample_seconds": _distribution(post_jit_values),
            "total_seconds": _distribution(
                _total_seconds(report) for report in implementation_reports
            ),
            "results": [report.get("results") for report in implementation_reports],
        }

    baseline_median = groups[baseline_label]["sample_call_seconds"]["median"]
    candidate_median = groups[candidate_label]["sample_call_seconds"]["median"]
    paired_speedups = [pair["candidate_speedup"] for pair in pairs]
    return {
        "schema_version": 1,
        "benchmark": "gw170817-full-run-four-gpu-comparison",
        "workload": workload,
        "created_at": datetime.now(UTC).isoformat(),
        "invariants": {
            "device_count": DEVICE_COUNT,
            "data_sha256": data_sha256,
            "workload": workload,
            "sampled_dimensions": WORKLOAD_DIMENSIONS[workload],
            "config": frozen_config,
            "config_sha256": _sha256_json(frozen_config),
            "shared_seeds": list(seeds),
            "paired_config_equal": True,
            "paired_initial_positions_equal": True,
            "all_data_hashes_equal": True,
            "software_environment_equal": True,
            "device_metadata_equal": True,
        },
        "groups": groups,
        "pairs": pairs,
        "comparison": {
            "candidate_speedup_ratio_of_medians": (baseline_median / candidate_median),
            "candidate_time_reduction_ratio_of_medians_percent": 100.0
            * (1.0 - candidate_median / baseline_median),
            "paired_speedup_geometric_mean": statistics.geometric_mean(paired_speedups),
            "paired_speedup_median": statistics.median(paired_speedups),
        },
    }


def _format_optional(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.6g}{suffix}"


def _write_markdown(
    path: Path,
    summary: Mapping[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
) -> None:
    invariants = summary["invariants"]
    workload = str(summary["workload"])
    lines = [
        f"# Four-GPU GW170817 full-run comparison ({workload})",
        "",
        (
            "This is a paired four-device comparison. Each implementation uses "
            "the same immutable data artifact, sampler configuration, and seed."
        ),
        (
            "The sample-call timing includes lazy JIT compilation because Jim's "
            "public full-run API does not expose a clean compile/run boundary."
        ),
        (
            "Post-JIT timing follows timing_seconds.paper_convention when "
            "available, with schema-v1 reports falling back to sample-call time "
            "minus the legacy sampler-JIT estimate."
        ),
        "",
        "## Frozen evidence",
        "",
        f"- Workload: `{workload}`",
        f"- Sampled dimensions: `{invariants['sampled_dimensions']}`",
        f"- Devices visible to each JAX process: `{invariants['device_count']}`",
        f"- Frozen data SHA-256: `{invariants['data_sha256']}`",
        f"- Frozen config SHA-256: `{invariants['config_sha256']}`",
        f"- Shared seeds: `{', '.join(map(str, invariants['shared_seeds']))}`",
        "",
        "## Aggregate timing",
        "",
        (
            "| Implementation | Revision | Runs | Sample median [s] | "
            "Post-JIT median [s] | Total median [s] |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label in (baseline_label, candidate_label):
        group = summary["groups"][label]
        lines.append(
            f"| {label} | `{group['revision'][:12]}` | {len(group['seeds'])} | "
            f"{group['sample_call_seconds']['median']:.3f} | "
            f"{group['post_jit_sample_seconds']['median']:.3f} | "
            f"{group['total_seconds']['median']:.3f} |"
        )

    comparison = summary["comparison"]
    lines.extend(
        [
            "",
            "## Paired comparison",
            "",
            (
                "Candidate speedup (ratio of medians): "
                f"**{comparison['candidate_speedup_ratio_of_medians']:.3f}x** "
                "("
                f"{comparison['candidate_time_reduction_ratio_of_medians_percent']:.2f}% "
                "less sample-call time)."
            ),
            "",
            (
                "Geometric mean of the paired speedups: "
                f"**{comparison['paired_speedup_geometric_mean']:.3f}x**."
            ),
            "",
            (
                "| Seed | Run order | Baseline [s] | Candidate [s] | "
                "Baseline post-JIT [s] | Candidate post-JIT [s] | "
                "Sample-call speedup | log Z (baseline / candidate) | Likelihood evals "
                "(baseline / candidate) |"
            ),
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for pair in summary["pairs"]:
        baseline_results = pair["baseline_results"] or {}
        candidate_results = pair["candidate_results"] or {}
        baseline_logz = _result_number({"results": baseline_results}, "log_Z", "logz")
        candidate_logz = _result_number({"results": candidate_results}, "log_Z", "logz")
        baseline_evals = _result_number(
            {"results": baseline_results},
            "n_likelihood_evaluations",
            "likelihood_evaluations",
        )
        candidate_evals = _result_number(
            {"results": candidate_results},
            "n_likelihood_evaluations",
            "likelihood_evaluations",
        )
        lines.append(
            f"| {pair['seed']} | {' / '.join(pair['order'])} | "
            f"{pair['baseline_sample_call_seconds']:.3f} | "
            f"{pair['candidate_sample_call_seconds']:.3f} | "
            f"{_format_optional(pair['baseline_post_jit_sample_seconds'])} | "
            f"{_format_optional(pair['candidate_post_jit_sample_seconds'])} | "
            f"{pair['candidate_speedup']:.3f}x | "
            f"{_format_optional(baseline_logz)} / "
            f"{_format_optional(candidate_logz)} | "
            f"{_format_optional(baseline_evals)} / "
            f"{_format_optional(candidate_evals)} |"
        )

    interpretation = (
        "This exercises Jim's full-resolution, aligned-spin tidal GW170817 "
        "SwiG workflow on four devices. It is the historical 11-dimensional "
        "benchmark analogue, not the paper's precessing-tidal model."
        if workload == ALIGNED_WORKLOAD
        else (
            "This exercises the paper's 15-dimensional precessing-tidal "
            "GW170817 model on trigger-centred frozen public-event data, as "
            "stated in the paper. It does not reproduce the paper's separate "
            "synthetic-injection catalogue or unreleased preprocessing."
        )
    )
    lines.extend(["", "## Interpretation boundary", "", interpretation])
    path.write_text("\n".join(lines) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _runner_environment(
    *,
    implementation_root: Path,
    seed: int,
    simulate_cpu: bool,
) -> dict[str, str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    pythonpath = str(implementation_root / "src")
    if existing_pythonpath:
        pythonpath = os.pathsep.join((pythonpath, existing_pythonpath))
    environment.update(
        {
            "JAX_ENABLE_COMPILATION_CACHE": "1",
            # The current RunPod H200/NVSwitch host rejects NCCL's NVLink
            # SHARP multicast allocation. Ordinary NCCL/NVLink collectives
            # remain enabled and are used identically by both revisions.
            "NCCL_NVLS_ENABLE": "0",
            "PYTHONHASHSEED": str(seed),
            "PYTHONPATH": pythonpath,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    if simulate_cpu:
        environment["JAX_PLATFORMS"] = "cpu"
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES
        environment["JAX_PLATFORMS"] = "cuda"
    return environment


def _implementation_runner_command(
    *,
    python: Path,
    runner_script: Path,
    data_file: Path,
    workload: str,
    seed: int,
    implementation_root: Path,
    implementation_label: str,
    implementation_revision: str,
    output: Path,
) -> list[str]:
    """Build the common command used by scientific runs and cache probes."""

    return [
        str(python),
        str(runner_script),
        "--data-file",
        str(data_file),
        "--workload",
        workload,
        "--seed",
        str(seed),
        "--n-devices",
        str(DEVICE_COUNT),
        "--implementation-root",
        str(implementation_root),
        "--implementation-label",
        implementation_label,
        "--implementation-revision",
        implementation_revision,
        "--output",
        str(output),
    ]


def _prepare_data(
    *,
    python: Path,
    runner_script: Path,
    data_file: Path,
    workload: str,
    candidate_root: Path,
    candidate_label: str,
    candidate_revision: str,
    repository: Path,
    raw_dir: Path,
    simulate_cpu: bool,
) -> None:
    output = raw_dir / "data-prepare.json"
    print(
        f"Preparing frozen GW170817 data at {data_file}",
        file=sys.stderr,
        flush=True,
    )
    command = [
        str(python),
        str(runner_script),
        "--prepare-data",
        "--data-file",
        str(data_file),
        "--workload",
        workload,
        "--output",
        str(output),
        "--implementation-root",
        str(candidate_root),
        "--implementation-label",
        candidate_label,
        "--implementation-revision",
        candidate_revision,
    ]
    if simulate_cpu:
        command.append("--simulate-cpu")
    result = _run(
        command,
        cwd=repository,
        env=_runner_environment(
            implementation_root=candidate_root,
            seed=0,
            simulate_cpu=simulate_cpu,
        ),
        check=False,
    )
    (raw_dir / "data-prepare.stdout.log").write_text(result.stdout)
    (raw_dir / "data-prepare.stderr.log").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            "GW170817 data preparation failed; see raw/data-prepare.stderr.log"
        )
    if not data_file.is_file():
        raise RuntimeError(f"data preparation did not create {data_file}")
    print(
        f"Frozen data ready bytes={data_file.stat().st_size} "
        f"sha256={_sha256_file(data_file)}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    repository = args.repository.expanduser().resolve()
    runner_script = args.runner_script.expanduser().resolve()
    # Keep a virtual environment's ``python`` symlink intact. Resolving it can
    # select uv's bare managed interpreter and silently drop the venv's
    # site-packages in the fresh benchmark processes.
    python = args.python.expanduser().absolute()
    output_dir = args.output_dir.expanduser().resolve()
    profile_dir = (
        args.profile_dir.expanduser().resolve()
        if args.profile_dir is not None
        else None
    )
    slice_data_dir = (
        args.slice_data_dir.expanduser().resolve()
        if args.slice_data_dir is not None
        else None
    )
    telemetry_dir = (
        args.telemetry_dir.expanduser().resolve()
        if args.telemetry_dir is not None
        else None
    )
    compilation_cache_root = (
        args.jax_compilation_cache_dir.expanduser().resolve()
        if args.jax_compilation_cache_dir is not None
        else None
    )
    labels = (args.baseline_label, args.candidate_label)
    seeds = list(dict.fromkeys(args.seeds))

    if not repository.joinpath("src", "jimgw").is_dir():
        raise SystemExit(f"--repository must contain src/jimgw; got {repository}")
    if not runner_script.is_file():
        raise SystemExit(f"--runner-script does not exist: {runner_script}")
    if not python.is_file():
        raise SystemExit(f"--python does not exist: {python}")
    if not seeds:
        raise SystemExit("--seeds must not be empty")
    if labels[0] == labels[1] or _slug(labels[0]) == _slug(labels[1]):
        raise SystemExit("baseline and candidate labels must be distinct")
    if output_dir.exists():
        raise SystemExit(f"--output-dir already exists: {output_dir}")
    if compilation_cache_root is not None:
        existing_caches = [
            compilation_cache_root / f"{_slug(label)}-g{DEVICE_COUNT}"
            for label in labels
            if (compilation_cache_root / f"{_slug(label)}-g{DEVICE_COUNT}").exists()
        ]
        if existing_caches:
            raise SystemExit(
                "persistent-cache targets must be fresh for a cold/warm test: "
                + ", ".join(map(str, existing_caches))
            )

    baseline_revision = _resolve_revision(repository, args.baseline_ref)
    candidate_revision = _resolve_revision(repository, args.candidate_ref)
    if baseline_revision == candidate_revision:
        raise SystemExit("baseline and candidate resolve to the same commit")
    revisions = {
        args.baseline_label: baseline_revision,
        args.candidate_label: candidate_revision,
    }

    output_dir.mkdir(parents=True)
    for artifact_dir in (
        profile_dir,
        slice_data_dir,
        telemetry_dir,
        compilation_cache_root,
    ):
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    harness_dir = output_dir / "harness"
    frozen_runner, harness_manifest = _freeze_harness(runner_script, harness_dir)

    data_file = (
        args.data_file.expanduser().resolve()
        if args.data_file is not None
        else output_dir / "data" / "gw170817.npz"
    )
    data_file.parent.mkdir(parents=True, exist_ok=True)

    hardware = _hardware_snapshot(repository)
    _write_json(output_dir / "hardware.json", hardware)
    if not args.simulate_cpu:
        physical_gpus = _physical_gpu_count(hardware)
        if physical_gpus < DEVICE_COUNT:
            raise SystemExit(
                f"benchmark requires at least {DEVICE_COUNT} GPUs, but "
                f"nvidia-smi found {physical_gpus}"
            )

    reports: list[dict[str, Any]] = []
    run_order: list[dict[str, Any]] = []
    cache_probe_summary: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="jim-gw170817-comparison-") as temporary:
        temporary_root = Path(temporary)
        roots = {
            args.baseline_label: temporary_root / "baseline",
            args.candidate_label: temporary_root / "candidate",
        }
        for label in labels:
            _extract_revision(repository, revisions[label], roots[label])

        if not data_file.exists():
            if args.skip_data_prepare:
                raise SystemExit(f"--skip-data-prepare requires {data_file} to exist")
            _prepare_data(
                python=python,
                runner_script=frozen_runner,
                data_file=data_file,
                workload=args.workload,
                candidate_root=roots[args.candidate_label],
                candidate_label=args.candidate_label,
                candidate_revision=candidate_revision,
                repository=repository,
                raw_dir=raw_dir,
                simulate_cpu=args.simulate_cpu,
            )
        elif not data_file.is_file():
            raise SystemExit(f"--data-file is not a regular file: {data_file}")

        frozen_data_sha256 = _sha256_file(data_file)
        if data_file.exists() and not (raw_dir / "data-prepare.json").exists():
            print(
                f"Reusing frozen data bytes={data_file.stat().st_size} "
                f"sha256={frozen_data_sha256}",
                file=sys.stderr,
                flush=True,
            )
        for pair_index, seed in enumerate(seeds):
            implementation_order = (
                labels if pair_index % 2 == 0 else tuple(reversed(labels))
            )
            for order_index, label in enumerate(implementation_order):
                stem = (
                    f"{pair_index:02d}-{order_index:02d}-{_slug(label)}-"
                    f"g{DEVICE_COUNT}-seed{seed}"
                )
                report_path = raw_dir / f"{stem}.json"
                command = _implementation_runner_command(
                    python=python,
                    runner_script=frozen_runner,
                    data_file=data_file,
                    workload=args.workload,
                    seed=seed,
                    implementation_root=roots[label],
                    implementation_label=label,
                    implementation_revision=revisions[label],
                    output=report_path,
                )
                artifacts: dict[str, str] = {}
                if profile_dir is not None:
                    run_profile_dir = profile_dir / stem
                    command.extend(
                        [
                            "--profile-dir",
                            str(run_profile_dir),
                            "--profile-warmup-steps",
                            str(args.profile_warmup_steps),
                            "--profile-steps",
                            str(args.profile_steps),
                        ]
                    )
                    artifacts["profile_dir"] = str(run_profile_dir)
                if slice_data_dir is not None:
                    slice_output = slice_data_dir / f"{stem}.npz"
                    command.extend(["--slice-data-output", str(slice_output)])
                    artifacts["slice_data"] = str(slice_output)
                if telemetry_dir is not None:
                    telemetry_output = telemetry_dir / f"{stem}.dmon"
                    command.extend(["--telemetry-output", str(telemetry_output)])
                    artifacts["telemetry"] = str(telemetry_output)
                if compilation_cache_root is not None:
                    cache_dir = (
                        compilation_cache_root / f"{_slug(label)}-g{DEVICE_COUNT}"
                    )
                    command.extend(["--jax-compilation-cache-dir", str(cache_dir)])
                    artifacts["jax_compilation_cache_dir"] = str(cache_dir)
                if args.simulate_cpu:
                    command.append("--simulate-cpu")
                run_record = {
                    "pair_index": pair_index,
                    "order_index": order_index,
                    "label": label,
                    "revision": revisions[label],
                    "devices": DEVICE_COUNT,
                    "seed": seed,
                    "workload": args.workload,
                    "command": command,
                    "report": str(report_path),
                    "artifacts": artifacts,
                }
                run_order.append(run_record)
                _write_json(output_dir / "run-order.json", run_order)
                print(
                    f"pair={pair_index + 1}/{len(seeds)} "
                    f"order={order_index + 1}/2 label={label} "
                    f"devices={DEVICE_COUNT} seed={seed}",
                    file=sys.stderr,
                    flush=True,
                )
                result = _run(
                    command,
                    cwd=repository,
                    env=_runner_environment(
                        implementation_root=roots[label],
                        seed=seed,
                        simulate_cpu=args.simulate_cpu,
                    ),
                    check=False,
                )
                stdout_path = raw_dir / f"{stem}.stdout.log"
                stderr_path = raw_dir / f"{stem}.stderr.log"
                stdout_path.write_text(result.stdout)
                stderr_path.write_text(result.stderr)
                run_record.update(
                    {
                        "returncode": result.returncode,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                    }
                )
                _write_json(output_dir / "run-order.json", run_order)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"benchmark failed for {label}, seed {seed}; "
                        f"see {stderr_path.name}"
                    )
                if not report_path.is_file():
                    raise RuntimeError(
                        f"runner did not write expected report {report_path}"
                    )
                report = json.loads(report_path.read_text())
                _validate_report(
                    report,
                    expected_backend="cpu" if args.simulate_cpu else "gpu",
                    expected_data_sha256=frozen_data_sha256,
                    expected_implementation_root=roots[label],
                    expected_label=label,
                    expected_revision=revisions[label],
                    expected_seed=seed,
                    expected_workload=args.workload,
                )
                reports.append(report)
                for artifact_name, artifact_path_text in artifacts.items():
                    artifact_path = Path(artifact_path_text)
                    if artifact_name == "jax_compilation_cache_dir":
                        if not artifact_path.is_dir():
                            raise RuntimeError(
                                f"runner did not create cache directory {artifact_path}"
                            )
                    elif artifact_name == "profile_dir":
                        if not artifact_path.is_dir() or not any(
                            artifact_path.rglob("*")
                        ):
                            raise RuntimeError(
                                f"runner did not create profiler trace {artifact_path}"
                            )
                    elif not artifact_path.is_file():
                        raise RuntimeError(
                            f"runner did not write {artifact_name} artifact {artifact_path}"
                        )
                run_record.update(
                    {
                        "sample_call_seconds": _sample_seconds(report),
                        "post_jit_sample_seconds": _post_jit_sample_seconds(report),
                        "total_seconds": _total_seconds(report),
                        "validated": True,
                    }
                )
                _write_json(output_dir / "run-order.json", run_order)
                print(
                    f"completed label={label} devices={DEVICE_COUNT} seed={seed} "
                    f"sample_call_seconds={_sample_seconds(report):.3f} "
                    f"total_seconds={_total_seconds(report):.3f}",
                    file=sys.stderr,
                    flush=True,
                )

        if args.persistent_cache_probe:
            assert compilation_cache_root is not None
            label = args.candidate_label
            seed = seeds[0]
            cache_dir = compilation_cache_root / f"{_slug(label)}-g{DEVICE_COUNT}"
            report_path = raw_dir / "candidate-persistent-cache-probe.json"
            command = _implementation_runner_command(
                python=python,
                runner_script=frozen_runner,
                data_file=data_file,
                workload=args.workload,
                seed=seed,
                implementation_root=roots[label],
                implementation_label=label,
                implementation_revision=revisions[label],
                output=report_path,
            )
            command.extend(
                [
                    "--jax-compilation-cache-dir",
                    str(cache_dir),
                    "--max-outer-steps",
                    "1",
                    "--retain-per-slice-info",
                ]
            )
            if args.simulate_cpu:
                command.append("--simulate-cpu")
            print(
                "running fresh-process candidate persistent-cache probe",
                file=sys.stderr,
                flush=True,
            )
            result = _run(
                command,
                cwd=repository,
                env=_runner_environment(
                    implementation_root=roots[label],
                    seed=seed,
                    simulate_cpu=args.simulate_cpu,
                ),
                check=False,
            )
            stdout_path = raw_dir / "candidate-persistent-cache-probe.stdout.log"
            stderr_path = raw_dir / "candidate-persistent-cache-probe.stderr.log"
            stdout_path.write_text(result.stdout)
            stderr_path.write_text(result.stderr)
            if result.returncode != 0 or not report_path.is_file():
                raise RuntimeError(
                    f"persistent-cache probe failed; see {stderr_path.name}"
                )
            probe_report = json.loads(report_path.read_text())
            _validate_report(
                probe_report,
                expected_backend="cpu" if args.simulate_cpu else "gpu",
                expected_data_sha256=frozen_data_sha256,
                expected_implementation_root=roots[label],
                expected_label=label,
                expected_revision=revisions[label],
                expected_seed=seed,
                expected_workload=args.workload,
            )
            cold_report = next(
                report
                for report in reports
                if _implementation(report).get("label") == label
                and _report_config(report).get("seed") == seed
            )
            cold_first = _number(
                cold_report["timing_seconds"]["outer_step"].get(
                    "first_step_seconds_including_jit"
                ),
                name="cold first outer step",
            )
            warm_first = _number(
                probe_report["timing_seconds"]["outer_step"].get(
                    "first_step_seconds_including_jit"
                ),
                name="warm-cache first outer step",
            )
            cache_probe_summary = {
                "implementation": label,
                "revision": revisions[label],
                "seed": seed,
                "workload": args.workload,
                "cache_dir": str(cache_dir),
                "cold_full_run_report": next(
                    record["report"]
                    for record in run_order
                    if record["label"] == label and record["seed"] == seed
                ),
                "warm_probe_report": str(report_path),
                "cold_first_step_seconds_including_jit": cold_first,
                "warm_cache_first_step_seconds": warm_first,
                "first_step_speedup": cold_first / warm_first,
                "seconds_recovered": cold_first - warm_first,
            }
            _write_json(
                output_dir / "persistent-cache-summary.json", cache_probe_summary
            )

        if _sha256_file(data_file) != frozen_data_sha256:
            raise RuntimeError("frozen data artifact changed during the matrix")

    summary = _build_summary(
        reports,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        revisions=revisions,
        seeds=seeds,
        data_sha256=frozen_data_sha256,
        workload=args.workload,
    )
    manifest = {
        "schema_version": 1,
        "benchmark": "gw170817-full-run-four-gpu-comparison",
        "workload": args.workload,
        "created_at": datetime.now(UTC).isoformat(),
        "repository": str(repository),
        "revisions": revisions,
        "configuration": {
            "workload": args.workload,
            "sampled_dimensions": WORKLOAD_DIMENSIONS[args.workload],
            "device_count": DEVICE_COUNT,
            "cuda_visible_devices": None if args.simulate_cpu else VISIBLE_DEVICES,
            "seeds": seeds,
            "simulate_cpu": args.simulate_cpu,
            "data_file": str(data_file),
            "data_sha256": frozen_data_sha256,
            "profile_dir": str(profile_dir) if profile_dir is not None else None,
            "slice_data_dir": (
                str(slice_data_dir) if slice_data_dir is not None else None
            ),
            "telemetry_dir": (
                str(telemetry_dir) if telemetry_dir is not None else None
            ),
            "jax_compilation_cache_dir": (
                str(compilation_cache_root)
                if compilation_cache_root is not None
                else None
            ),
            "persistent_cache_probe": cache_probe_summary,
        },
        "harness": harness_manifest,
        "hardware_file": str(output_dir / "hardware.json"),
        "run_order": run_order,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "summary.json", summary)
    _write_markdown(
        output_dir / "summary.md",
        summary,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
