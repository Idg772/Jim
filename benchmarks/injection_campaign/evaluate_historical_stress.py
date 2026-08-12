"""Evaluate the staged historical-point FSM/SwiG stress campaign."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.injection_campaign.common import (
    CATALOGUE_FIELDS,
    DEFAULT_CONFIG,
    PARAMETERS,
    atomic_write_json,
    file_sha256,
    load_manifest,
    publication_eligible,
    read_catalogue,
    result_dir,
)
from benchmarks.injection_campaign.prepare_historical_stress import (
    CONTROL_IDS,
    HISTORICAL_IDS,
    HISTORICAL_MASTER_SEED,
    HISTORICAL_SOURCE_SHA256,
    PROBLEM_IDS,
    SAMPLER_SCHEDULERS,
)

EXPECTED_INJECTIONS = len(HISTORICAL_IDS)
REQUIRED_DEVICE_COUNT = 4
REQUIRED_DEVICE_KIND = "NVIDIA H200"
Z_Q_PASS_LIMIT = 2.5
Z_Q_HARD_STOP_LIMIT = 3.0
EXPECTED_TIMING_FORMULA = "sample_call - likelihood_jit - sampler_kernel_jit"
STRESS_CAMPAIGN_NAME = "paper-sharded-swig-fsm-historical-stress"
STRESS_TIMING_SELECTION = (
    "targeted explicit stress catalogue; retained timings are diagnostic "
    "only and are not a Figure 3 population"
)
EXPECTED_MAPPED_CATALOGUE_SHA256 = (
    "872ab00ba6a6356d96aa70f8a3ee79c6aa4007291e3954d2280254083bdc937d"
)
MAPPED_Q_WIDTH_FLOORS = {
    88: 0.002167958970949595,
    82: 0.00261031248173004,
    75: 0.0038163438934647152,
    58: 0.003824499594987996,
}
LEGACY_Q_WIDTH_SCALE = 4.0 / 7.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Evaluate the completed prefix during the staged run. A clean "
            "partial result permits only the rest of this stress run, never "
            "the full P-P campaign."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=("JSON report path. Defaults to CAMPAIGN/historical-stress-report.json."),
    )
    return parser.parse_args(argv)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _q_width_gate(
    historical_id: int,
    weighted_population_std: float,
) -> tuple[float | None, bool]:
    floor = MAPPED_Q_WIDTH_FLOORS.get(historical_id)
    return floor, floor is None or weighted_population_std > floor


def _finite_float(
    value: object,
    *,
    field: str,
    path: Path,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid {field} in {path}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field} in {path}")
    if positive and result <= 0.0:
        raise ValueError(f"non-positive {field} in {path}")
    if nonnegative and result < 0.0:
        raise ValueError(f"negative {field} in {path}")
    return result


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    logs = np.asarray(log_weights, dtype=float)
    if logs.ndim != 1 or logs.size == 0:
        raise ValueError("log_weights must be a non-empty one-dimensional array")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("log_weights must not contain NaN or +inf")
    finite = np.isfinite(logs)
    if not np.any(finite):
        raise ValueError("log_weights must contain at least one finite value")
    weights = np.exp(logs - float(np.max(logs[finite])))
    normalizer = float(np.sum(weights))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("log_weights cannot be normalized")
    return weights / normalizer


def _weighted_rank(
    samples: np.ndarray,
    truth: float,
    weights: np.ndarray,
) -> float:
    return float(np.sum(weights[samples < truth]))


def weighted_q_statistics(
    q_samples: np.ndarray,
    truth_q: float,
    log_weights: np.ndarray,
) -> dict[str, float]:
    """Return directly weighted q location, spread, rank, and z diagnostics.

    The median is the inverse weighted empirical CDF (the first ordered sample
    whose cumulative normalized weight is at least one half). The standard
    deviation is the weighted population standard deviation. This preserves
    the historical gate's ``(truth - median) / std`` sign convention without
    resampling the nested-sampling output.
    """

    values = np.asarray(q_samples, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("q samples must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or not math.isfinite(float(truth_q)):
        raise ValueError("q samples and truth must be finite")
    weights = _normalized_weights(log_weights)
    if weights.shape != values.shape:
        raise ValueError("q samples and log_weights must have the same shape")

    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    median_index = int(np.searchsorted(np.cumsum(ordered_weights), 0.5, side="left"))
    median = float(ordered_values[min(median_index, values.size - 1)])
    mean = float(np.sum(weights * values))
    variance = float(np.sum(weights * (values - mean) ** 2))
    standard_deviation = math.sqrt(max(0.0, variance))
    if standard_deviation == 0.0:
        raise ValueError("weighted q standard deviation is zero")
    truth = float(truth_q)
    return {
        "truth": truth,
        "weighted_median": median,
        "weighted_mean": mean,
        "weighted_population_std": standard_deviation,
        "z_weighted_median_std": (truth - median) / standard_deviation,
        "rank": _weighted_rank(values, truth, weights),
        "effective_sample_size": float(1.0 / np.sum(weights**2)),
    }


def _validate_provenance(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if int(manifest.get("n_injections", -1)) != EXPECTED_INJECTIONS:
        raise ValueError(
            f"historical stress campaign must contain {EXPECTED_INJECTIONS} recoveries"
        )
    if int(manifest.get("catalogue_size", -1)) != EXPECTED_INJECTIONS:
        raise ValueError(
            f"historical stress catalogue must contain {EXPECTED_INJECTIONS} rows"
        )
    selection = _mapping(manifest.get("selection"), field="manifest.selection")
    if (
        selection.get("rule") != "explicit ordered stress catalogue"
        or int(selection.get("start_inclusive", -1)) != 0
        or int(selection.get("stop_exclusive", -1)) != EXPECTED_INJECTIONS
    ):
        raise ValueError("historical stress campaign has an invalid selection")

    catalogue = _mapping(manifest.get("catalogue"), field="manifest.catalogue")
    if catalogue.get("sha256") != EXPECTED_MAPPED_CATALOGUE_SHA256:
        raise ValueError("historical stress catalogue has the wrong mapped SHA-256")
    provenance = _mapping(
        catalogue.get("provenance"), field="manifest.catalogue.provenance"
    )
    if provenance.get("kind") != "historical-prior-quantile-stress":
        raise ValueError("campaign is not the historical prior-quantile stress set")
    source_sha256 = provenance.get("source_catalogue_sha256")
    if source_sha256 != HISTORICAL_SOURCE_SHA256:
        raise ValueError("historical provenance has the wrong source catalogue hash")
    if provenance.get("source_master_seed") != HISTORICAL_MASTER_SEED:
        raise ValueError("historical provenance has the wrong source master seed")
    if manifest.get("master_seed") is not None:
        raise ValueError(
            "explicit historical stress manifest must not claim a generator seed"
        )
    if catalogue.get("generator") != "explicit ordered rows":
        raise ValueError(
            "historical catalogue is not labelled as explicit ordered rows"
        )
    mapping = provenance.get("mapping")
    if not isinstance(mapping, list) or len(mapping) != EXPECTED_INJECTIONS:
        raise ValueError("historical provenance must map exactly ten cases")

    normalized: list[dict[str, Any]] = []
    for expected_id, raw in enumerate(mapping):
        entry = _mapping(raw, field=f"historical mapping {expected_id}")
        try:
            preflight_id = int(entry["preflight_id"])
            historical_id = int(entry["historical_id"])
            cohort = str(entry["cohort"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"historical mapping {expected_id} is incomplete"
            ) from error
        if preflight_id != expected_id:
            raise ValueError("historical preflight IDs must be ordered from zero")
        expected_historical_id = HISTORICAL_IDS[expected_id]
        if historical_id != expected_historical_id:
            raise ValueError(
                "historical mapping does not match the canonical staged order"
            )
        expected_cohort = "problem" if historical_id in PROBLEM_IDS else "control"
        if cohort != expected_cohort:
            raise ValueError(
                f"historical ID {historical_id} has the wrong cohort {cohort!r}"
            )
        normalized.append(
            {
                "preflight_id": preflight_id,
                "historical_id": historical_id,
                "cohort": cohort,
            }
        )
    if set(PROBLEM_IDS) | set(CONTROL_IDS) != set(HISTORICAL_IDS):
        raise RuntimeError(
            "historical cohort constants do not partition the staged IDs"
        )
    return normalized


def _validate_campaign_methodology(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    config = _mapping(manifest.get("config"), field="manifest.config")
    actual_config = dict(config)
    sampler_scheduler = str(actual_config.get("sampler_scheduler", "fsm"))
    if sampler_scheduler not in SAMPLER_SCHEDULERS:
        raise ValueError(
            f"historical stress campaign has an unsupported sampler scheduler: "
            f"{sampler_scheduler!r}"
        )
    actual_config["sampler_scheduler"] = sampler_scheduler
    expected_config = copy.deepcopy(DEFAULT_CONFIG)
    expected_config["campaign"] = SAMPLER_SCHEDULERS[sampler_scheduler]
    expected_config["sampler_scheduler"] = sampler_scheduler
    expected_config["timing"]["selected_events"] = STRESS_TIMING_SELECTION
    if actual_config != expected_config:
        raise ValueError(
            "historical stress configuration differs from the frozen paper "
            "configuration beyond its campaign name and selection annotation"
        )
    if int(config.get("n_devices", -1)) != REQUIRED_DEVICE_COUNT:
        raise ValueError("historical stress configuration must request four devices")
    timing_config = _mapping(config.get("timing"), field="manifest.config.timing")
    if timing_config.get("post_jit_formula") != EXPECTED_TIMING_FORMULA:
        raise ValueError("campaign does not use the Figure 3 post-JIT formula")
    if timing_config.get("excluded_one_off_phases") != [
        "likelihood_jit",
        "sampler_kernel_jit",
    ]:
        raise ValueError("campaign has the wrong Figure 3 excluded JIT phases")
    if publication_eligible(manifest):
        raise ValueError(
            "historical stress campaign must be labelled ineligible for P-P "
            "calibration publication products"
        )
    scope = _mapping(
        manifest.get("reproduction_scope"), field="manifest.reproduction_scope"
    )
    if scope.get("iid_prior_predictive_catalogue") is not False:
        raise ValueError("historical stress catalogue must be labelled non-iid")
    return config


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object: {path}")
    return value


def _validate_truth_and_seeds(
    summary: Mapping[str, Any],
    catalogue_row: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    truth = _mapping(summary.get("truth"), field=f"truth in {path}")
    for name in CATALOGUE_FIELDS[3:]:
        actual = _finite_float(truth.get(name), field=f"truth.{name}", path=path)
        expected = float(catalogue_row[name])
        if actual != expected:
            raise ValueError(f"summary truth.{name} disagrees with catalogue: {path}")
    seeds = _mapping(summary.get("seeds"), field=f"seeds in {path}")
    if int(seeds.get("noise", -1)) != int(catalogue_row["noise_seed"]):
        raise ValueError(f"summary noise seed disagrees with catalogue: {path}")
    if int(seeds.get("sampler", -1)) != int(catalogue_row["sampler_seed"]):
        raise ValueError(f"summary sampler seed disagrees with catalogue: {path}")


def _validate_timing(
    summary: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, float]:
    timing = _mapping(summary.get("timing_seconds"), field=f"timing_seconds in {path}")
    paper = _mapping(
        timing.get("paper_convention"),
        field=f"timing_seconds.paper_convention in {path}",
    )
    phases = _mapping(
        timing.get("sample_phases"),
        field=f"timing_seconds.sample_phases in {path}",
    )
    sample_call = _finite_float(
        timing.get("sample_call"),
        field="timing_seconds.sample_call",
        path=path,
        positive=True,
    )
    likelihood_jit = _finite_float(
        paper.get("likelihood_jit_seconds"),
        field="paper_convention.likelihood_jit_seconds",
        path=path,
        nonnegative=True,
    )
    sampler_jit = _finite_float(
        paper.get("sampler_jit_seconds"),
        field="paper_convention.sampler_jit_seconds",
        path=path,
        nonnegative=True,
    )
    post_jit = _finite_float(
        paper.get("post_jit_sampling_seconds"),
        field="paper_convention.post_jit_sampling_seconds",
        path=path,
        positive=True,
    )
    expected_post_jit = sample_call - likelihood_jit - sampler_jit
    tolerance = max(1e-9, abs(sample_call) * 1e-9)
    if not math.isclose(post_jit, expected_post_jit, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"inconsistent Figure 3 timing arithmetic in {path}")
    for phase_name, expected in (
        ("likelihood_jit", likelihood_jit),
        ("sampler_kernel_jit", sampler_jit),
    ):
        measured = _finite_float(
            phases.get(phase_name),
            field=f"sample_phases.{phase_name}",
            path=path,
            nonnegative=True,
        )
        if not math.isclose(measured, expected, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"paper timing disagrees with sample phase {phase_name} in {path}"
            )
    return {
        "sample_call_seconds": sample_call,
        "likelihood_jit_seconds": likelihood_jit,
        "sampler_kernel_jit_seconds": sampler_jit,
        "post_jit_sampling_seconds": post_jit,
    }


def _validate_devices(
    summary: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    if summary.get("simulated_cpu") is not False:
        raise ValueError(f"historical stress result used CPU simulation: {path}")
    devices = _mapping(summary.get("devices"), field=f"devices in {path}")
    backend = str(devices.get("backend", "")).lower()
    try:
        requested = int(devices["requested_count"])
        local = int(devices["local_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid device counts in {path}") from error
    details = devices.get("devices")
    if not isinstance(details, list):
        raise TypeError(f"devices.devices must be a list in {path}")
    if requested != REQUIRED_DEVICE_COUNT or local != REQUIRED_DEVICE_COUNT:
        raise ValueError(f"result did not use exactly four local devices: {path}")
    if len(details) != REQUIRED_DEVICE_COUNT:
        raise ValueError(f"result does not record exactly four devices: {path}")
    if backend != "gpu":
        raise ValueError(f"result did not use the GPU backend: {path}")
    platforms: list[str] = []
    kinds: list[str] = []
    ids: list[int] = []
    for index, raw in enumerate(details):
        device = _mapping(raw, field=f"devices.devices[{index}] in {path}")
        platform = str(device.get("platform", "")).lower()
        if platform != "gpu":
            raise ValueError(f"result records a non-GPU device: {path}")
        device_kind = str(device.get("device_kind", ""))
        if device_kind != REQUIRED_DEVICE_KIND:
            raise ValueError(
                f"result did not run on four {REQUIRED_DEVICE_KIND} devices: {path}"
            )
        try:
            device_id = int(device["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"result records an invalid device ID: {path}") from error
        platforms.append(platform)
        kinds.append(device_kind)
        ids.append(device_id)
    if len(set(ids)) != REQUIRED_DEVICE_COUNT:
        raise ValueError(f"result device IDs are not unique: {path}")
    return {
        "backend": backend,
        "count": local,
        "platforms": platforms,
        "device_kinds": kinds,
    }


def _load_case(
    campaign_dir: Path,
    *,
    manifest: Mapping[str, Any],
    catalogue_row: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> dict[str, Any] | None:
    injection_id = int(mapping["preflight_id"])
    directory = result_dir(campaign_dir, injection_id)
    summary_path = directory / "summary.json"
    posterior_path = directory / "posterior.npz"
    failure_path = directory / "failure.json"
    running_path = directory / "RUNNING"
    if not summary_path.exists() and not posterior_path.exists():
        if failure_path.exists():
            raise ValueError(
                f"historical stress recovery {injection_id} has a failure record: "
                f"{failure_path}"
            )
        if running_path.exists():
            raise ValueError(
                f"historical stress recovery {injection_id} is still marked running: "
                f"{running_path}"
            )
        return None
    if not summary_path.is_file() or not posterior_path.is_file():
        raise ValueError(f"incomplete result artifact pair in {directory}")
    if failure_path.exists() or running_path.exists():
        raise ValueError(
            f"completed result has stale failure or running state: {directory}"
        )

    summary = _load_json(summary_path, description="result summary")
    if summary.get("schema_version") != 2:
        raise ValueError(
            f"historical stress result must use summary schema 2: {summary_path}"
        )
    if summary.get("config_sha256") != manifest["config_sha256"]:
        raise ValueError(f"result belongs to a different campaign: {summary_path}")
    if summary.get("injection_id") != injection_id:
        raise ValueError(f"result has the wrong injection ID: {summary_path}")
    _validate_truth_and_seeds(summary, catalogue_row, path=summary_path)

    posterior = _mapping(summary.get("posterior"), field=f"posterior in {summary_path}")
    if posterior.get("path") != "posterior.npz":
        raise ValueError(f"unexpected posterior path in {summary_path}")
    expected_sha256 = posterior.get("sha256")
    if not isinstance(expected_sha256, str):
        raise TypeError(f"missing posterior hash in {summary_path}")
    if file_sha256(posterior_path) != expected_sha256:
        raise ValueError(f"posterior hash mismatch: {posterior_path}")
    if int(posterior.get("bytes", -1)) != posterior_path.stat().st_size:
        raise ValueError(f"posterior byte count mismatch: {posterior_path}")
    if posterior.get("weighting") != "normalized nested-sampling log weights":
        raise ValueError(
            f"posterior does not record nested-sampling weights: {summary_path}"
        )

    fields = posterior.get("fields")
    expected_fields = {*PARAMETERS, "log_likelihood", "log_weights"}
    if (
        not isinstance(fields, list)
        or not all(isinstance(field, str) for field in fields)
        or len(fields) != len(set(fields))
        or set(fields) != expected_fields
    ):
        raise ValueError(f"posterior field declaration is invalid: {summary_path}")
    if posterior.get("space") != "prior":
        raise ValueError(
            f"posterior does not record prior-space samples: {summary_path}"
        )
    with np.load(posterior_path, allow_pickle=False) as archive:
        if set(archive.files) != set(fields):
            raise ValueError(
                f"posterior archive fields disagree with its summary: {posterior_path}"
            )
        arrays = {field: np.asarray(archive[field], dtype=float) for field in fields}
    try:
        sample_count = int(summary["posterior_samples"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid posterior sample count: {summary_path}") from error
    if sample_count < 1:
        raise ValueError(f"posterior sample count must be positive: {summary_path}")
    for field, values in arrays.items():
        if values.ndim != 1 or values.shape != (sample_count,):
            raise ValueError(
                f"posterior field {field} is not aligned and one-dimensional: "
                f"{posterior_path}"
            )
        if field != "log_weights" and not np.all(np.isfinite(values)):
            raise ValueError(
                f"posterior field {field} contains non-finite values: {posterior_path}"
            )
    q_samples = arrays["q"]
    t_c_samples = arrays["t_c"]
    log_weights = arrays["log_weights"]
    weights = _normalized_weights(log_weights)
    finite_logs = np.isfinite(log_weights)
    maximum_log_weight = float(np.max(log_weights[finite_logs]))
    log_normalizer = maximum_log_weight + math.log(
        float(np.sum(np.exp(log_weights - maximum_log_weight)))
    )
    if not math.isclose(log_normalizer, 0.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"posterior log_weights are not normalized: {posterior_path}")
    effective_sample_size = float(1.0 / np.sum(weights**2))
    reported_ess = _finite_float(
        summary.get("posterior_effective_sample_size"),
        field="posterior_effective_sample_size",
        path=summary_path,
        positive=True,
    )
    if reported_ess > sample_count + 1e-9 or not math.isclose(
        reported_ess,
        effective_sample_size,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"posterior effective sample size is inconsistent: {summary_path}"
        )

    diagnostics = _mapping(
        summary.get("diagnostics"), field=f"diagnostics in {summary_path}"
    )
    for field in ("n_iterations", "n_likelihood_evaluations"):
        try:
            value = int(diagnostics[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid diagnostics.{field} in {summary_path}"
            ) from error
        if value < 1:
            raise ValueError(f"diagnostics.{field} must be positive in {summary_path}")
    _finite_float(
        diagnostics.get("log_Z"), field="diagnostics.log_Z", path=summary_path
    )
    _finite_float(
        diagnostics.get("log_Z_error"),
        field="diagnostics.log_Z_error",
        path=summary_path,
        nonnegative=True,
    )

    q_statistics = weighted_q_statistics(
        q_samples,
        float(catalogue_row["q"]),
        log_weights,
    )
    t_c_rank = _weighted_rank(t_c_samples, float(catalogue_row["t_c"]), weights)
    ranks = _mapping(summary.get("ranks"), field=f"ranks in {summary_path}")
    rank_method = _mapping(
        summary.get("rank_method"), field=f"rank_method in {summary_path}"
    )
    if dict(rank_method) != {
        "weighting": "original nested-sampling weights",
        "comparison": "sample < truth",
        "resampled": False,
    }:
        raise ValueError(f"unexpected rank methodology in {summary_path}")
    for name, recomputed in (("q", q_statistics["rank"]), ("t_c", t_c_rank)):
        reported = _finite_float(
            ranks.get(name), field=f"ranks.{name}", path=summary_path
        )
        if not 0.0 <= reported <= 1.0:
            raise ValueError(f"rank {name} lies outside [0, 1] in {summary_path}")
        if not math.isclose(reported, recomputed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"reported {name} rank disagrees with posterior: {summary_path}"
            )

    endpoint_parameters = [
        name
        for name, rank in (("q", q_statistics["rank"]), ("t_c", t_c_rank))
        if rank == 0.0 or rank == 1.0
    ]
    absolute_z = abs(q_statistics["z_weighted_median_std"])
    z_gate_pass = absolute_z <= Z_Q_PASS_LIMIT
    hard_stop = absolute_z > Z_Q_HARD_STOP_LIMIT
    historical_id = int(mapping["historical_id"])
    q_width_floor, q_width_gate_pass = _q_width_gate(
        historical_id,
        q_statistics["weighted_population_std"],
    )
    timing = _validate_timing(summary, path=summary_path)
    device_report = _validate_devices(summary, path=summary_path)
    case_pass = z_gate_pass and q_width_gate_pass and not endpoint_parameters
    return {
        **dict(mapping),
        "status": "pass" if case_pass else ("hard-stop" if hard_stop else "fail"),
        "q": q_statistics,
        "t_c_rank": t_c_rank,
        "rank_endpoint_parameters": endpoint_parameters,
        "z_q_gate_pass": z_gate_pass,
        "q_width_mapped_legacy_floor": q_width_floor,
        "q_width_gate_pass": q_width_gate_pass,
        "rank_endpoint_gate_pass": not endpoint_parameters,
        "case_pass": case_pass,
        "hard_stop": hard_stop,
        "timing": timing,
        "devices": device_report,
        "inputs": {
            "summary": str(summary_path.relative_to(campaign_dir)),
            "summary_sha256": file_sha256(summary_path),
            "posterior": str(posterior_path.relative_to(campaign_dir)),
            "posterior_sha256": expected_sha256,
        },
    }


def evaluate_historical_stress(
    campaign_dir: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Evaluate completed stress cases and return a deterministic gate report."""

    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    mapping = _validate_provenance(manifest)
    config = _validate_campaign_methodology(manifest)

    catalogue_path = campaign_dir / str(
        _mapping(manifest["catalogue"], field="manifest.catalogue")["path"]
    )
    catalogue = read_catalogue(catalogue_path)
    if len(catalogue) != EXPECTED_INJECTIONS:
        raise ValueError("historical stress catalogue does not have ten rows")

    cases: list[dict[str, Any]] = []
    missing_ids: list[int] = []
    for entry in mapping:
        injection_id = int(entry["preflight_id"])
        case = _load_case(
            campaign_dir,
            manifest=manifest,
            catalogue_row=catalogue[injection_id],
            mapping=entry,
        )
        if case is None:
            missing_ids.append(injection_id)
        else:
            cases.append(case)
    completed_ids = [int(case["preflight_id"]) for case in cases]
    if completed_ids != list(range(len(completed_ids))):
        raise ValueError(
            "completed historical stress results must form a contiguous staged "
            "prefix from preflight ID zero"
        )
    if missing_ids and not allow_partial:
        missing = ", ".join(map(str, missing_ids))
        raise ValueError(
            "historical stress campaign is incomplete; "
            f"use --allow-partial only for a staged check. Missing IDs: {missing}"
        )
    if not cases:
        raise ValueError("no completed historical stress results were found")

    hard_stop_ids = [int(case["preflight_id"]) for case in cases if case["hard_stop"]]
    failed_ids = [int(case["preflight_id"]) for case in cases if not case["case_pass"]]
    complete = not missing_ids
    if hard_stop_ids:
        decision = "hard-stop"
    elif failed_ids:
        decision = "fail"
    elif complete:
        decision = "pass"
    else:
        decision = "partial-pass"

    provenance = _mapping(
        _mapping(manifest["catalogue"], field="manifest.catalogue").get("provenance"),
        field="manifest.catalogue.provenance",
    )
    return {
        "schema_version": 1,
        "campaign": config.get("campaign"),
        "config_sha256": manifest["config_sha256"],
        "decision": decision,
        "hard_stop": bool(hard_stop_ids),
        "continue_stress_run": decision in {"partial-pass", "pass"},
        "proceed_to_full_pp_campaign": decision == "pass",
        "selection": {
            "requested_cases": EXPECTED_INJECTIONS,
            "completed_cases": len(cases),
            "complete": complete,
            "completed_preflight_ids": completed_ids,
            "missing_preflight_ids": missing_ids,
            "failed_preflight_ids": failed_ids,
            "hard_stop_preflight_ids": hard_stop_ids,
        },
        "thresholds": {
            "absolute_z_q_pass_maximum": Z_Q_PASS_LIMIT,
            "absolute_z_q_hard_stop_strictly_above": Z_Q_HARD_STOP_LIMIT,
            "q_or_t_c_exact_rank_endpoint_allowed": False,
            "required_non_cpu_devices": REQUIRED_DEVICE_COUNT,
            "required_device_kind": REQUIRED_DEVICE_KIND,
            "mapped_q_width_floors_by_historical_id": {
                str(historical_id): floor
                for historical_id, floor in MAPPED_Q_WIDTH_FLOORS.items()
            },
        },
        "methodology": {
            "sampler_scheduler": config.get("sampler_scheduler", "fsm"),
            "z_q_weighted_median_std": (
                "(q_truth - weighted_q_median) / weighted_population_std_q"
            ),
            "weighted_median": (
                "inverse weighted empirical CDF: first ordered q with cumulative "
                "normalized weight >= 0.5"
            ),
            "weights": "original nested-sampling log weights; no resampling",
            "rank": "sum of normalized weights for samples strictly below truth",
            "timing": EXPECTED_TIMING_FORMULA,
            "scientific_use": provenance.get("scientific_use"),
            "anti_narrowing": {
                "rule": (
                    "weighted_population_std_q must exceed the mapped legacy "
                    "width floor for historical IDs 88, 82, 75, and 58"
                ),
                "legacy_to_paper_q_prior_width_scale": LEGACY_Q_WIDTH_SCALE,
                "basis": (
                    "linear q quantile map: paper width 0.5 divided by legacy "
                    "width 0.875 = 4/7"
                ),
            },
            "legacy_runtime_1_35x_gate": {
                "status": "waived-not-comparable",
                "reason": (
                    "The stress run changes the priors, PSD, segment placement, "
                    "sampled-t_c geometry, hardware context, and reports the "
                    "Figure 3 post-JIT convention; unchanged 1.35x comparisons "
                    "to the legacy campaign are not scientifically interpretable."
                ),
            },
        },
        "provenance": {
            "kind": provenance.get("kind"),
            "source_catalogue_sha256": provenance.get("source_catalogue_sha256"),
            "source_master_seed": provenance.get("source_master_seed"),
            "mapping": mapping,
        },
        "inputs": {
            "manifest": "manifest.json",
            "manifest_sha256": file_sha256(campaign_dir / "manifest.json"),
            "catalogue": str(catalogue_path.relative_to(campaign_dir)),
            "catalogue_sha256": file_sha256(catalogue_path),
        },
        "cases": cases,
    }


def _print_report(report: Mapping[str, Any]) -> None:
    for case in report["cases"]:
        q = case["q"]
        endpoints = case["rank_endpoint_parameters"]
        endpoint_text = ",".join(endpoints) if endpoints else "none"
        print(
            f"preflight {case['preflight_id']:02d} "
            f"(historical {case['historical_id']:02d}, {case['cohort']}): "
            f"weighted-median/std z_q={q['z_weighted_median_std']:+.3f}, "
            f"std_q={q['weighted_population_std']:.6g}, "
            f"q_rank={q['rank']:.6f}, t_c_rank={case['t_c_rank']:.6f}, "
            f"endpoints={endpoint_text}, status={case['status']}"
        )
    selection = report["selection"]
    print(
        f"Decision: {report['decision']} "
        f"({selection['completed_cases']}/{selection['requested_cases']} complete; "
        f"proceed_to_full_pp_campaign={report['proceed_to_full_pp_campaign']})"
    )


def run_evaluation(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.expanduser().resolve()
    report = evaluate_historical_stress(
        campaign_dir,
        allow_partial=bool(args.allow_partial),
    )
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else campaign_dir / "historical-stress-report.json"
    )
    atomic_write_json(report_path, report)
    _print_report(report)
    print(f"Report: {report_path}")
    return 0 if report["decision"] in {"partial-pass", "pass"} else 1


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_evaluation(_parse_args(argv)))


if __name__ == "__main__":
    main()
