"""Fail-closed local preflight for the GW170817 full/heterodyne D=4 pair.

This does not run nested sampling.  It reconstructs the two likelihood targets
from the exact frozen data, author-compatible waveform convention and frozen
relative-binning reference, then checks pointwise agreement on deterministic
posterior and likelihood-range probes.  The output is a provenance-bearing JSON
artifact intended to be reviewed before any paid GPU is provisioned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REQUESTED_BINS = 5000
TC_RANGE = (-0.03, 0.03)
TIME_UPSAMPLE_FACTOR = 8
CONVERGENCE_UPSAMPLE_FACTOR = 16
POSTERIOR_PROBES = 48
LIKELIHOOD_RANGE_PROBES = 12
STRESS_PROBES_PER_ARTIFACT = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sampling_to_likelihood(sample: dict[str, float]) -> dict[str, float]:
    q = sample["q"]

    def spin(label: str) -> dict[str, float]:
        magnitude = sample[f"{label}_mag"]
        theta = sample[f"{label}_theta"]
        phi = sample[f"{label}_phi"]
        return {
            f"{label}_x": magnitude * math.sin(theta) * math.cos(phi),
            f"{label}_y": magnitude * math.sin(theta) * math.sin(phi),
            f"{label}_z": magnitude * math.cos(theta),
        }

    return {
        "M_c": sample["M_c"],
        "eta": q / (1.0 + q) ** 2,
        **spin("s1"),
        **spin("s2"),
        "iota": sample["iota"],
        "lambda_1": sample["lambda_1"],
        "lambda_2": sample["lambda_2"],
        "d_L": sample["d_L"],
        "ra": sample["ra"],
        "dec": sample["dec"],
        "psi": sample["psi"],
        "phase_c": 0.0,
        "t_c": 0.0,
    }


def _systematic_indices(log_weights: np.ndarray[Any, Any], count: int) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    targets = (np.arange(count, dtype=np.float64) + 0.5) / count
    return np.searchsorted(cdf, targets, side="left")


def _build_likelihoods(data_file: Path, reference: dict[str, Any]):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from benchmarks.device_parallel_nss import (
        benchmark_gw170817_full_run as benchmark,
    )
    from benchmarks.device_parallel_nss.paper_heterodyne import (
        PaperTimeMarginalizedHeterodynedLikelihoodFD,
    )
    from benchmarks.device_parallel_nss.paper_model import (
        RippleIMRPhenomPv2NRTidalv2,
    )
    from jimgw.core.single_event.data import Data, PowerSpectrum
    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1

    print("preflight: loading frozen bundle", flush=True)
    _, arrays = benchmark._read_bundle(data_file, benchmark.PAPER_WORKLOAD)
    ifos = [get_H1(), get_L1(), get_V1()]
    for detector in ifos:
        print(f"preflight: projecting {detector.name} data", flush=True)
        strain, psd = benchmark._likelihood_inputs_from_bundle(
            detector.name,
            arrays,
            Data=Data,
            PowerSpectrum=PowerSpectrum,
            jnp=jnp,
        )
        detector.set_data(strain)
        detector.set_psd(psd)

    print("preflight: constructing author-compatible waveform", flush=True)
    waveform = RippleIMRPhenomPv2NRTidalv2(
        f_ref=20.0,
        time_anchor="imrphenomd",
    )
    common = {
        "detectors": ifos,
        "waveform": waveform,
        "trigger_time": benchmark.GPS,
        "f_min": benchmark.F_MIN,
        "f_max": benchmark.F_MAX,
        "phase_marginalization": True,
    }
    print("preflight: constructing full likelihood", flush=True)
    full = _construct_full_likelihood_identical_grid(
        **common,
        tc_range=TC_RANGE,
        upsample_factor=TIME_UPSAMPLE_FACTOR,
    )
    print("preflight: constructing convergence-reference full likelihood", flush=True)
    convergence_full = _construct_full_likelihood_identical_grid(
        **common,
        tc_range=TC_RANGE,
        upsample_factor=CONVERGENCE_UPSAMPLE_FACTOR,
    )
    print("preflight: constructing compressed likelihood", flush=True)
    compressed = PaperTimeMarginalizedHeterodynedLikelihoodFD(
        **common,
        n_bins=REQUESTED_BINS,
        reference_parameters=reference["likelihood_parameters"],
        time_marginalization={
            "tc_range": TC_RANGE,
            "upsample_factor": TIME_UPSAMPLE_FACTOR,
        },
    )
    print("preflight: likelihood construction complete", flush=True)
    return jax, full, convergence_full, compressed


def _construct_full_likelihood_identical_grid(
    *,
    detectors: Any,
    waveform: Any,
    trigger_time: float,
    f_min: float,
    f_max: float,
    phase_marginalization: bool,
    tc_range: tuple[float, float],
    upsample_factor: int = 1,
):
    """Construct the stock full evaluator without its costly generic grid union.

    The scientific runner uses the ordinary class on GPU.  This preflight runs
    on a local CPU where the generic ``jnp.isin`` setup for three 259k-point
    grids can exhaust memory.  The frozen H1/L1/V1 inputs are required to have
    byte-identical frequency grids, making the union and all three masks known
    exactly.  Every evaluation and marginalization method remains the stock
    :class:`TransientLikelihoodFD` implementation.
    """

    import jax
    import jax.numpy as jnp

    from jimgw.core.single_event.likelihood import (
        SingleEventLikelihood,
        TransientLikelihoodFD,
    )
    from jimgw.core.single_event.marginalization_config import TimeMargConfig
    from jimgw.core.single_event.time_utils import (
        greenwich_mean_sidereal_time as compute_gmst,
    )

    likelihood = TransientLikelihoodFD.__new__(TransientLikelihoodFD)
    SingleEventLikelihood.__init__(likelihood, detectors, waveform, None)
    likelihood.likelihood_optimizations = True
    likelihood.likelihood_optimization_axes = {
        "shared_frequency_grid": True,
        "detector_phasor": True,
        "real_inner_product": True,
    }
    frequencies = likelihood._set_detector_frequency_bounds(f_min, f_max)
    host = [np.asarray(jax.device_get(value)) for value in frequencies]
    if not all(np.array_equal(host[0], value) for value in host[1:]):
        raise RuntimeError("preflight full evaluator requires identical detector grids")
    likelihood.df = frequencies[0][1] - frequencies[0][0]
    likelihood.frequencies = frequencies[0]
    likelihood.frequency_masks = [
        jnp.ones(len(frequencies[0]), dtype=bool) for _ in detectors
    ]
    likelihood._identical_masks = True
    likelihood.trigger_time = trigger_time
    likelihood.gmst = compute_gmst(trigger_time)
    likelihood.time_marginalization = True
    likelihood.phase_marginalization = phase_marginalization
    likelihood.distance_marginalization = False
    likelihood._init_time_marginalization(
        TimeMargConfig(tc_range=tc_range, upsample_factor=upsample_factor)
    )
    if phase_marginalization:
        likelihood._init_phase_marginalization()
    likelihood._install_time_marg_data_weights()
    return likelihood


def _evaluate(value: Any) -> float:
    return float(np.asarray(value.block_until_ready()))


def run_preflight(
    data_file: Path,
    reference_file: Path,
    posterior_file: Path,
    stress_posterior_files: tuple[Path, ...] = (),
) -> dict[str, Any]:
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    if _sha256(posterior_file) != reference["source_sha256"]:
        raise RuntimeError("posterior probe artifact does not match frozen reference")
    if reference["waveform_contract"] != {
        "f_ref_hz": 20.0,
        "model": "IMRPhenomPv2_NRTidalv2",
        "time_anchor": "imrphenomd",
    }:
        raise RuntimeError("frozen reference has the wrong waveform contract")

    setup_started = time.perf_counter()
    jax, full, convergence_full, compressed = _build_likelihoods(
        data_file,
        reference,
    )
    setup_seconds = time.perf_counter() - setup_started

    direct_full = jax.jit(full.evaluate)
    direct_convergence_full = jax.jit(convergence_full.evaluate)
    direct_compressed = jax.jit(compressed.evaluate)
    reference_parameters = reference["likelihood_parameters"]
    reference_full = _evaluate(direct_full(reference_parameters))
    reference_convergence_full = _evaluate(
        direct_convergence_full(reference_parameters)
    )
    reference_compressed = _evaluate(direct_compressed(reference_parameters))
    cache = compressed.generate_waveform(reference_parameters)
    reference_cached = _evaluate(
        compressed.evaluate_from_waveform(reference_parameters, cache)
    )
    reference_direct = _evaluate(compressed.evaluate(reference_parameters))

    sampling_fields = (
        "M_c",
        "q",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "iota",
        "lambda_1",
        "lambda_2",
        "d_L",
        "ra",
        "dec",
        "psi",
    )
    with np.load(posterior_file, allow_pickle=False) as nested:
        for field in (*sampling_fields, "log_likelihood", "log_weights"):
            if field not in nested.files:
                raise RuntimeError(f"posterior probe artifact lacks {field!r}")
        arrays = {field: np.asarray(nested[field]) for field in sampling_fields}
        stored_log_likelihood = np.asarray(nested["log_likelihood"])
        log_weights = np.asarray(nested["log_weights"])

    posterior_indices = _systematic_indices(log_weights, POSTERIOR_PROBES)
    likelihood_indices = np.unique(
        np.quantile(
            np.arange(len(stored_log_likelihood)),
            np.linspace(0.0, 1.0, LIKELIHOOD_RANGE_PROBES),
        ).astype(int)
    )
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group, indices in (
        ("posterior", posterior_indices),
        ("likelihood_range", likelihood_indices),
    ):
        for raw_index in indices:
            index = int(raw_index)
            if index in seen:
                continue
            seen.add(index)
            sample = {field: float(arrays[field][index]) for field in sampling_fields}
            parameters = _sampling_to_likelihood(sample)
            full_value = _evaluate(direct_full(parameters))
            convergence_full_value = _evaluate(direct_convergence_full(parameters))
            compressed_value = _evaluate(direct_compressed(parameters))
            rows.append(
                {
                    "group": group,
                    "index": index,
                    "stored_full": float(stored_log_likelihood[index]),
                    "recomputed_full": full_value,
                    "convergence_full": convergence_full_value,
                    "full_u8_minus_u16": full_value - convergence_full_value,
                    "compressed": compressed_value,
                    "compressed_minus_full": compressed_value - full_value,
                    "recomputed_minus_stored": (
                        full_value - float(stored_log_likelihood[index])
                    ),
                }
            )

    stress_artifacts: list[dict[str, Any]] = []
    for stress_path in stress_posterior_files:
        with np.load(stress_path, allow_pickle=False) as nested:
            for field in (*sampling_fields, "log_likelihood", "log_weights"):
                if field not in nested.files:
                    raise RuntimeError(f"stress artifact {stress_path} lacks {field!r}")
            stress_arrays = {
                field: np.asarray(nested[field]) for field in sampling_fields
            }
            stress_stored = np.asarray(nested["log_likelihood"])
            stress_weights = np.asarray(nested["log_weights"])
        stress_indices = _systematic_indices(
            stress_weights,
            STRESS_PROBES_PER_ARTIFACT,
        )
        stress_artifacts.append(
            {
                "path": str(stress_path.resolve()),
                "sha256": _sha256(stress_path),
                "probes": len(stress_indices),
            }
        )
        for raw_index in stress_indices:
            index = int(raw_index)
            sample = {
                field: float(stress_arrays[field][index]) for field in sampling_fields
            }
            parameters = _sampling_to_likelihood(sample)
            full_value = _evaluate(direct_full(parameters))
            convergence_full_value = _evaluate(direct_convergence_full(parameters))
            compressed_value = _evaluate(direct_compressed(parameters))
            rows.append(
                {
                    "group": f"stress:{stress_path.name}",
                    "index": index,
                    "stored_u1_full": float(stress_stored[index]),
                    "recomputed_full": full_value,
                    "convergence_full": convergence_full_value,
                    "full_u8_minus_u16": full_value - convergence_full_value,
                    "compressed": compressed_value,
                    "compressed_minus_full": compressed_value - full_value,
                    "u16_full_minus_stored_u1": (
                        full_value - float(stress_stored[index])
                    ),
                }
            )

    posterior_rows = [row for row in rows if row["group"] == "posterior"]
    posterior_delta = np.asarray(
        [row["compressed_minus_full"] for row in posterior_rows]
    )
    posterior_centered = posterior_delta - np.mean(posterior_delta)
    importance = np.exp(posterior_delta - np.max(posterior_delta))
    importance_ess_fraction = float(
        importance.sum() ** 2 / (len(importance) * np.square(importance).sum())
    )
    all_delta = np.asarray([row["compressed_minus_full"] for row in rows])
    full_values = np.asarray([row["recomputed_full"] for row in rows])
    convergence_delta = np.asarray([row["full_u8_minus_u16"] for row in rows])
    reference_level = max(reference_full, float(np.max(full_values)))
    gaps = np.abs(reference_level - full_values)
    beta_mask = gaps >= 0.1
    beta = np.abs(all_delta[beta_mask]) / gaps[beta_mask]

    bin_edges = np.concatenate(
        (
            np.asarray(compressed.freq_grid_low),
            np.asarray(compressed.freq_grid_high)[-1:],
        )
    )
    metrics = {
        "reference_abs_delta": abs(reference_compressed - reference_full),
        "reference_u16_full_minus_source_u1": (
            reference_convergence_full - float(reference["source_log_likelihood"])
        ),
        "reference_u8_minus_u16": reference_full - reference_convergence_full,
        "reference_direct_cache_abs_delta": abs(reference_direct - reference_cached),
        "reference_direct_jit_abs_delta": abs(reference_direct - reference_compressed),
        "posterior_max_abs_delta": float(np.max(np.abs(posterior_delta))),
        "posterior_p99_abs_delta": float(np.quantile(np.abs(posterior_delta), 0.99)),
        "posterior_centered_rms_delta": float(
            np.sqrt(np.mean(np.square(posterior_centered)))
        ),
        "posterior_importance_ess_fraction": importance_ess_fraction,
        "all_probe_max_abs_delta": float(np.max(np.abs(all_delta))),
        "all_probe_p99_abs_delta": float(np.quantile(np.abs(all_delta), 0.99)),
        "all_probe_beta_p99": float(np.quantile(beta, 0.99)),
        "time_convergence_max_abs_u8_minus_u16": float(
            np.max(np.abs(convergence_delta))
        ),
        "time_convergence_p99_abs_u8_minus_u16": float(
            np.quantile(np.abs(convergence_delta), 0.99)
        ),
        "all_values_finite": bool(
            np.isfinite(all_delta).all() and np.isfinite(full_values).all()
        ),
    }
    criteria = {
        "reference_abs_delta_le_0.1": metrics["reference_abs_delta"] <= 0.1,
        "compressed_direct_cache_le_1e-9": (
            metrics["reference_direct_cache_abs_delta"] <= 1e-9
        ),
        "compressed_direct_jit_le_1e-9": (
            metrics["reference_direct_jit_abs_delta"] <= 1e-9
        ),
        "posterior_max_abs_delta_le_0.1": (metrics["posterior_max_abs_delta"] <= 0.1),
        "posterior_p99_abs_delta_le_0.05": (metrics["posterior_p99_abs_delta"] <= 0.05),
        "posterior_centered_rms_delta_le_0.05": (
            metrics["posterior_centered_rms_delta"] <= 0.05
        ),
        "posterior_importance_ess_fraction_ge_0.99": (
            metrics["posterior_importance_ess_fraction"] >= 0.99
        ),
        "all_probe_beta_p99_le_0.01": metrics["all_probe_beta_p99"] <= 0.01,
        "time_convergence_max_abs_u8_minus_u16_le_0.01": (
            metrics["time_convergence_max_abs_u8_minus_u16"] <= 0.01
        ),
        "time_convergence_p99_abs_u8_minus_u16_le_0.005": (
            metrics["time_convergence_p99_abs_u8_minus_u16"] <= 0.005
        ),
        "all_values_finite": metrics["all_values_finite"],
    }
    return {
        "schema_version": 1,
        "purpose": "pre-spend likelihood-only correctness gate",
        "passed": all(criteria.values()),
        "criteria": criteria,
        "metrics": metrics,
        "setup_seconds": setup_seconds,
        "data": {"path": str(data_file.resolve()), "sha256": _sha256(data_file)},
        "reference": {
            "path": str(reference_file.resolve()),
            "sha256": _sha256(reference_file),
            "source_sha256": reference["source_sha256"],
        },
        "compression": {
            "requested_bins": REQUESTED_BINS,
            "realized_bins": int(compressed.n_bins),
            "bin_edges_sha256": _array_sha256(bin_edges),
            "coefficient_builder": compressed.coefficient_builder,
            "time_points": len(compressed.tc_window),
            "time_normalization_count": int(compressed._tc_normalization_count),
            "time_upsample_factor": int(compressed.tc_upsample),
        },
        "waveform": reference["waveform_contract"],
        "stress_artifacts": stress_artifacts,
        "rows": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--posterior-file", type=Path, required=True)
    parser.add_argument(
        "--stress-posterior-file",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    result = run_preflight(
        args.data_file.expanduser().resolve(),
        args.reference_file.expanduser().resolve(),
        args.posterior_file.expanduser().resolve(),
        tuple(path.expanduser().resolve() for path in args.stress_posterior_file),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit("likelihood-pair preflight failed")


if __name__ == "__main__":
    main()
