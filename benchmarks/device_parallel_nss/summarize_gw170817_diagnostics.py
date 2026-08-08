"""Summarize the GW170817 sharding diagnostic artifact bundle."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

WORKLOAD_DIMENSIONS = {"aligned-11d": 11, "paper-15d": 15}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--microbenchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workload",
        choices=tuple(WORKLOAD_DIMENSIONS),
        default="aligned-11d",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _load_full_reports(
    directory: Path,
    workload: str = "aligned-11d",
) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(directory.glob("*.json")):
        report = _load_json(path)
        if report.get("benchmark") != "gw170817-full-swig-4gpu":
            continue
        if report.get("operation") is not None:
            continue
        actual_workload = report.get("config", {}).get("workload")
        if actual_workload != workload:
            raise RuntimeError(
                f"{path} uses workload {actual_workload!r}, expected {workload!r}"
            )
        expected_dimensions = WORKLOAD_DIMENSIONS[workload]
        actual_dimensions = report.get("config", {}).get("sampled_dimensions")
        if actual_dimensions != expected_dimensions:
            raise RuntimeError(
                f"{path} has {actual_dimensions!r} dimensions, expected "
                f"{expected_dimensions} for {workload}"
            )
        if report.get("results", {}).get("early_stopped_for_cache_probe"):
            continue
        report["_report_path"] = str(path.resolve())
        reports.append(report)
    return reports


def _implementation_label(report: Mapping[str, Any]) -> str:
    value = report.get("implementation", {}).get("label")
    if not isinstance(value, str):
        raise TypeError("report has no implementation label")
    return value


def _outer_timing(report: Mapping[str, Any]) -> dict[str, Any]:
    timing = report.get("timing_seconds", {}).get("outer_step")
    if not isinstance(timing, dict):
        raise TypeError("report has no outer-step timing")
    return timing


def _steady_samples(report: Mapping[str, Any]) -> list[float]:
    timing = _outer_timing(report)
    samples = [float(value) for value in timing["duration_seconds"]]
    profiled = set(timing.get("profile", {}).get("captured_step_indices", []))
    unprofiled = [
        value
        for index, value in enumerate(samples)
        if index > 0 and index not in profiled
    ]
    return unprofiled or samples[1:]


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    samples = list(values)
    if not samples:
        raise ValueError("cannot summarize an empty collection")
    return {
        "count": len(samples),
        "minimum": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "maximum": max(samples),
        "standard_deviation": float(np.std(samples)),
    }


def _slot_cost(counts: np.ndarray, n_devices: int) -> dict[str, int | float]:
    if counts.ndim != 3:
        raise ValueError(f"expected (iterations, chains, slices), got {counts.shape}")
    n_iterations, n_chains, _ = counts.shape
    if n_chains % n_devices:
        raise ValueError("chain count is not divisible by device count")
    lanes = n_chains // n_devices
    device_counts = counts.reshape(n_iterations, n_devices, lanes, counts.shape[-1])
    fsm_slots = int(device_counts.sum())
    lockstep_slots = int(device_counts.max(axis=2).sum() * lanes)
    return {
        "fsm_slots": fsm_slots,
        "lockstep_slots": lockstep_slots,
        "lockstep_to_fsm_ratio": lockstep_slots / fsm_slots,
    }


def _combine_slot_cost(
    expansions: np.ndarray, shrink: np.ndarray, n_devices: int
) -> dict[str, Any]:
    expansion_cost = _slot_cost(expansions, n_devices)
    shrink_cost = _slot_cost(shrink, n_devices)
    fsm = int(expansion_cost["fsm_slots"] + shrink_cost["fsm_slots"])
    lockstep = int(expansion_cost["lockstep_slots"] + shrink_cost["lockstep_slots"])
    return {
        "combined": {
            "fsm_slots": fsm,
            "lockstep_slots": lockstep,
            "lockstep_to_fsm_ratio": lockstep / fsm,
        },
        "expansions": expansion_cost,
        "shrink": shrink_cost,
    }


def _slice_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    metadata = report.get("results", {}).get("per_slice_update_info")
    if not isinstance(metadata, Mapping):
        raise TypeError("paired report has no per-slice artifact metadata")
    path = Path(str(metadata["path"]))
    with np.load(path, allow_pickle=False) as archive:
        expansions = np.asarray(archive["num_expansions"])
        shrink = np.asarray(archive["num_shrink"])
        block_indices = np.asarray(archive["slice_block_index"])
        requires_rebuild = np.asarray(archive["slice_requires_rebuild"])
        parameter_names = np.asarray(archive["slice_block_parameter_names"]).astype(str)
        n_devices = int(np.asarray(archive["n_devices"]).item())
        n_delete = len(np.unique(np.asarray(archive["chain_index"])))

    n_iterations = expansions.shape[0] // n_delete
    expansions = expansions.reshape(n_iterations, n_delete, expansions.shape[-1])
    shrink = shrink.reshape(n_iterations, n_delete, shrink.shape[-1])
    by_cache_class = {}
    for name, mask in (
        ("waveform_rebuild", requires_rebuild),
        ("cache_hit", ~requires_rebuild),
    ):
        by_cache_class[name] = _combine_slot_cost(
            expansions[:, :, mask], shrink[:, :, mask], n_devices
        )
    by_block = {}
    for block_index in np.unique(block_indices):
        mask = block_indices == block_index
        by_block[str(int(block_index))] = {
            "block_parameters": sorted(set(parameter_names[mask].tolist())),
            "requires_waveform_rebuild": bool(requires_rebuild[mask][0]),
            **_combine_slot_cost(expansions[:, :, mask], shrink[:, :, mask], n_devices),
        }
    return {
        "artifact": str(path),
        "shape": list(expansions.shape),
        "slice_block_parameters": parameter_names.tolist(),
        "slice_requires_rebuild": requires_rebuild.tolist(),
        "all_slices": _combine_slot_cost(expansions, shrink, n_devices),
        "by_cache_class": by_cache_class,
        "by_block": by_block,
    }


def _parse_dmon(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "error": "missing"}
    header: list[str] | None = None
    rows: list[list[str]] = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            candidate = line.lstrip("# ").split()
            if "gpu" in candidate and "sm" in candidate and "mem" in candidate:
                header = candidate
            continue
        rows.append(line.split())
    if header is None:
        return {"path": str(path), "samples": len(rows), "error": "header not parsed"}
    parsed: list[dict[str, str]] = []
    for fields in rows:
        if len(fields) == len(header):
            parsed.append(dict(zip(header, fields, strict=True)))
    by_gpu: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample in parsed:
        gpu = sample["gpu"]
        for column in ("sm", "mem", "pwr"):
            try:
                by_gpu[gpu][column].append(float(sample[column]))
            except (KeyError, ValueError):
                pass
    return {
        "path": str(path),
        "samples": len(parsed),
        "note": (
            "nvidia-smi dmon sm is the SM-active utilization proxy and mem is "
            "memory-interface activity; dmon does not expose achieved occupancy."
        ),
        "gpus": {
            gpu: {column: _distribution(values) for column, values in columns.items()}
            for gpu, columns in by_gpu.items()
        },
    }


def _paired_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reports) != 2:
        raise RuntimeError(f"expected two paired reports, found {len(reports)}")
    output = {}
    for report in reports:
        label = _implementation_label(report)
        timing = _outer_timing(report)
        telemetry_path = timing.get("profile", {}).get("telemetry_output")
        output[label] = {
            "report": report["_report_path"],
            "sample_call_seconds": report["timing_seconds"]["sample_call"],
            "first_step_seconds_including_jit": timing[
                "first_step_seconds_including_jit"
            ],
            "steady_unprofiled_seconds": _distribution(_steady_samples(report)),
            "profile": timing["profile"],
            "telemetry": (
                _parse_dmon(Path(telemetry_path)) if telemetry_path else None
            ),
            "per_slice": _slice_summary(report),
            "results": report["results"],
        }
    return output


def _sweep_summary(
    paired: Mapping[str, Any], sweep_reports: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate_label = next(label for label in paired if "baseline" not in label.lower())
    points = {4: paired[candidate_label]}
    for report in sweep_reports:
        if _implementation_label(report) != candidate_label:
            continue
        devices = int(report["config"]["n_devices"])
        points[devices] = {
            "report": report["_report_path"],
            "steady_unprofiled_seconds": _distribution(_steady_samples(report)),
            "results": report["results"],
        }
    if set(points) != {1, 2, 4}:
        raise RuntimeError(f"incomplete candidate sweep: {sorted(points)}")
    one = points[1]["steady_unprofiled_seconds"]["median"]
    scaling = {}
    for devices, point in sorted(points.items()):
        median = point["steady_unprofiled_seconds"]["median"]
        speedup = one / median
        scaling[str(devices)] = {
            "median_outer_step_seconds": median,
            "speedup_from_one_gpu": speedup,
            "parallel_efficiency": speedup / devices,
        }
    return {"candidate_label": candidate_label, "points": points, "scaling": scaling}


def _microbenchmark_summary(
    report: Mapping[str, Any],
    workload: str | None = None,
) -> dict[str, Any]:
    if workload is not None:
        actual = report.get("workload", {}).get("name")
        if actual != workload:
            raise RuntimeError(
                f"likelihood-lane workload is {actual!r}, expected {workload!r}"
            )
        actual_dimensions = report.get("workload", {}).get("sampled_dimensions")
        if actual_dimensions != WORKLOAD_DIMENSIONS[workload]:
            raise RuntimeError(
                "likelihood-lane sampled dimensions do not match the workload"
            )
    points = {}
    for lane_text, point in report["results"].items():
        lanes = int(lane_text)
        points[lane_text] = {
            variant: {
                "median_batch_seconds": point[variant]["median_seconds"],
                "microseconds_per_lane": (
                    point[variant]["median_seconds"] * 1e6 / lanes
                ),
            }
            for variant in ("waveform_rebuild", "cache_hit")
        }
    for variant in ("waveform_rebuild", "cache_hit"):
        base = points["16"][variant]["median_batch_seconds"]
        for point in points.values():
            point[variant]["batch_cost_relative_to_16_lanes"] = (
                point[variant]["median_batch_seconds"] / base
            )
    return {"report": report, "points": points}


def _persistent_cache_summary(path: Path, workload: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = _load_json(path)
    actual = report.get("workload")
    if actual != workload:
        raise RuntimeError(
            f"persistent-cache workload is {actual!r}, expected {workload!r}"
        )
    return report


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        f"# GW170817 sharding diagnostics ({summary['workload']})",
        "",
        "## Paired four-GPU run",
        "",
        "| Implementation | First step incl. JIT [s] | Steady outer step [s] | Lockstep/FSM slots | SM active median range |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for label, point in summary["paired_four_gpu"].items():
        telemetry = point.get("telemetry") or {}
        sm_medians = [
            metrics["sm"]["median"]
            for metrics in telemetry.get("gpus", {}).values()
            if "sm" in metrics
        ]
        sm_text = (
            f"{min(sm_medians):.1f}--{max(sm_medians):.1f}%" if sm_medians else "—"
        )
        lines.append(
            f"| {label} | {point['first_step_seconds_including_jit']:.3f} | "
            f"{point['steady_unprofiled_seconds']['median']:.6f} | "
            f"{point['per_slice']['all_slices']['combined']['lockstep_to_fsm_ratio']:.3f}x | "
            f"{sm_text} |"
        )

    cache = summary.get("persistent_cache")
    if cache:
        lines.extend(
            [
                "",
                "## Persistent compilation cache",
                "",
                (
                    f"Cold first step: "
                    f"`{cache['cold_first_step_seconds_including_jit']:.3f} s`; "
                    f"fresh-process cache hit: "
                    f"`{cache['warm_cache_first_step_seconds']:.3f} s`; "
                    f"recovered `{cache['seconds_recovered']:.3f} s` "
                    f"(`{cache['first_step_speedup']:.2f}x`)."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Candidate device sweep",
            "",
            "| GPUs | Median outer step [s] | Speedup | Parallel efficiency |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for devices, point in summary["candidate_device_sweep"]["scaling"].items():
        lines.append(
            f"| {devices} | {point['median_outer_step_seconds']:.6f} | "
            f"{point['speedup_from_one_gpu']:.3f}x | "
            f"{100 * point['parallel_efficiency']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## One-GPU likelihood lane sweep",
            "",
            "| Lanes | Rebuild batch [ms] | Rebuild vs 16 | Cache-hit batch [ms] | Cache-hit vs 16 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lanes, point in summary["likelihood_lane_sweep"]["points"].items():
        rebuild = point["waveform_rebuild"]
        hit = point["cache_hit"]
        lines.append(
            f"| {lanes} | {1e3 * rebuild['median_batch_seconds']:.6f} | "
            f"{rebuild['batch_cost_relative_to_16_lanes']:.3f}x | "
            f"{1e3 * hit['median_batch_seconds']:.6f} | "
            f"{hit['batch_cost_relative_to_16_lanes']:.3f}x |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    paired_dir = args.paired_dir.expanduser().resolve()
    sweep_dir = args.sweep_dir.expanduser().resolve()
    microbenchmark_path = args.microbenchmark.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paired_reports = _load_full_reports(paired_dir / "raw", args.workload)
    paired = _paired_summary(paired_reports)
    sweep_reports = _load_full_reports(sweep_dir, args.workload)
    cache_summary = _persistent_cache_summary(
        paired_dir / "persistent-cache-summary.json",
        args.workload,
    )
    summary = {
        "schema_version": 1,
        "benchmark": "gw170817-sharding-diagnostic-study",
        "workload": args.workload,
        "paired_four_gpu": paired,
        "persistent_cache": cache_summary,
        "candidate_device_sweep": _sweep_summary(paired, sweep_reports),
        "likelihood_lane_sweep": _microbenchmark_summary(
            _load_json(microbenchmark_path), args.workload
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_markdown(output_dir / "summary.md", summary)
    print(output_dir)


if __name__ == "__main__":
    main()
