"""Run one paper-15d FSM/SwiG injection recovery on all visible devices."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.injection_campaign.common import (
    PARAMETERS,
    atomic_savez_compressed,
    atomic_write_json,
    file_sha256,
    load_manifest,
    posterior_rank,
    read_catalogue,
    result_dir,
)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("injection_id", type=_nonnegative_int)
    parser.add_argument(
        "--jax-compilation-cache-dir",
        type=Path,
        default=None,
        help="Persistent cache shared by every recovery in this campaign.",
    )
    parser.add_argument(
        "--simulate-cpu",
        action="store_true",
        help="Expose logical CPU devices for setup validation; not a scientific run.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a completed result for this injection.",
    )
    return parser.parse_args(argv)


def _configure_cpu_simulation(n_devices: int) -> None:
    desired = f"--xla_force_host_platform_device_count={n_devices}"
    tokens = os.environ.get("XLA_FLAGS", "").split()
    configured = [
        token
        for token in tokens
        if token.startswith("--xla_force_host_platform_device_count=")
    ]
    if configured and configured != [desired]:
        raise SystemExit("XLA_FLAGS has a conflicting host device count")
    if not configured:
        os.environ["XLA_FLAGS"] = " ".join([*tokens, desired]).strip()
    os.environ["JAX_PLATFORMS"] = "cpu"


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _device_report(jax: Any, requested: int, simulate_cpu: bool) -> dict[str, Any]:
    devices = jax.local_devices()
    if len(devices) != requested:
        raise SystemExit(
            f"campaign requires exactly {requested} visible devices; JAX sees {len(devices)}"
        )
    if not simulate_cpu and jax.default_backend() != "gpu":
        raise SystemExit(
            f"campaign requires GPUs; JAX selected {jax.default_backend()!r}. "
            "Use --simulate-cpu only for setup validation."
        )
    return {
        "backend": jax.default_backend(),
        "requested_count": requested,
        "local_count": jax.local_device_count(),
        "devices": [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
            }
            for device in devices
        ],
    }


def _injection_parameters(
    truth: dict[str, Any], likelihood_transforms: list[Any]
) -> dict[str, Any]:
    parameters = {name: truth[name] for name in (*PARAMETERS, "phase_c", "t_c")}
    for transform in likelihood_transforms:
        parameters = transform.forward(parameters)
    return parameters


def _time_marginalization_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve time-marginalization settings without changing old manifests."""

    jitter_time = config.get("time_marginalization_jitter_time", False)
    if not isinstance(jitter_time, bool):
        raise TypeError("time_marginalization_jitter_time must be a boolean")
    return {
        "tc_range": tuple(config["time_marginalization_tc_range_seconds"]),
        "upsample_factor": int(config.get("time_marginalization_upsample_factor", 1)),
        "jitter_time": jitter_time,
    }


def _recovery_sampling_components(
    prior: Any,
    periodic: dict[str, tuple[float, float]],
    likelihood: Any,
) -> tuple[Any, dict[str, tuple[float, float]]]:
    """Add the sampled one-cell time shift when jitter is enabled."""

    if not likelihood.jitter_time:
        return prior, dict(periodic)

    from jimgw.core.prior import CombinePrior, UniformPrior

    bounds = tuple(float(value) for value in likelihood.time_jitter_bounds)
    jitter_prior = UniformPrior(*bounds, parameter_names=["time_jitter"])
    recovery_prior = CombinePrior([*prior.base_prior, jitter_prior])
    return recovery_prior, {**periodic, "time_jitter": bounds}


def run_injection(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    n_injections = int(manifest["n_injections"])
    if args.injection_id >= n_injections:
        raise SystemExit(
            f"injection_id {args.injection_id} is outside [0, {n_injections})"
        )
    config = manifest["config"]
    directory = result_dir(campaign_dir, args.injection_id)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    posterior_path = directory / "posterior.npz"
    if summary_path.is_file() and posterior_path.is_file() and not args.force:
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing is not None:
            if existing.get("config_sha256") != manifest["config_sha256"]:
                raise SystemExit(
                    f"completed result has a different campaign hash: {directory}"
                )
            expected_sha256 = existing.get("posterior", {}).get("sha256")
            if (
                isinstance(expected_sha256, str)
                and file_sha256(posterior_path) == expected_sha256
            ):
                return existing

    catalogue = read_catalogue(campaign_dir / manifest["catalogue"]["path"])
    truth = catalogue[args.injection_id]
    n_devices = int(config["n_devices"])
    if args.simulate_cpu:
        _configure_cpu_simulation(n_devices)

    import jax
    import jax.numpy as jnp
    import jaxlib

    jax.config.update("jax_enable_x64", True)
    cache_dir = (
        args.jax_compilation_cache_dir.expanduser().resolve()
        if args.jax_compilation_cache_dir is not None
        else campaign_dir / ".jax-cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    import blackjax

    import jimgw
    from benchmarks.device_parallel_nss.benchmark_gw170817_full_run import (
        PAPER_WORKLOAD,
        _analysis_components,
    )
    from jimgw.core.jim import Jim
    from jimgw.core.single_event.data import PowerSpectrum
    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
    from jimgw.core.single_event.likelihood import TransientLikelihoodFD
    from jimgw.samplers.config import BlackJAXSwiGConfig

    devices = _device_report(jax, n_devices, args.simulate_cpu)
    ifos = [get_H1(), get_L1(), get_V1()]
    psd_files = manifest["psd"]["detector_files"]
    for ifo in ifos:
        ifo.set_psd(PowerSpectrum.from_file(str(campaign_dir / psd_files[ifo.name])))

    components = _analysis_components(PAPER_WORKLOAD, jnp, ifos)
    injection_parameters = _injection_parameters(
        truth, components["likelihood_transforms"]
    )
    injection_started = time.perf_counter()
    noise_key = jax.random.key(truth["noise_seed"])
    start_time = (
        float(config["trigger_time_gps"]) - float(config["duration_seconds"]) / 2.0
    )
    for detector_index, ifo in enumerate(ifos):
        ifo.inject_signal(
            duration=float(config["duration_seconds"]),
            sampling_frequency=float(config["sampling_frequency_hz"]),
            trigger_time=float(config["trigger_time_gps"]),
            waveform_model=components["waveform"],
            parameters=injection_parameters,
            f_min=float(config["f_min_hz"]),
            f_max=float(config["f_max_hz"]),
            start_time=start_time,
            zero_noise=False,
            rng_key=jax.random.fold_in(noise_key, detector_index),
        )
    injection_seconds = time.perf_counter() - injection_started

    setup_started = time.perf_counter()
    likelihood = TransientLikelihoodFD(
        ifos,
        waveform=components["waveform"],
        trigger_time=float(config["trigger_time_gps"]),
        f_min=float(config["f_min_hz"]),
        f_max=float(config["f_max_hz"]),
        phase_marginalization=True,
        time_marginalization=_time_marginalization_config(config),
    )
    recovery_prior, recovery_periodic = _recovery_sampling_components(
        components["prior"], components["periodic"], likelihood
    )
    sampler_config = BlackJAXSwiGConfig(
        blocks=config["blocks"],
        n_live=int(config["n_live"]),
        n_delete_frac=float(config["n_delete_frac"]),
        num_inner_steps_per_dim=int(config["num_inner_steps_per_dim"]),
        num_gibbs_sweeps=int(config["num_gibbs_sweeps"]),
        termination_dlogz=float(config["termination_dlogz"]),
        n_devices=n_devices,
    )
    jim = Jim(
        likelihood,
        recovery_prior,
        sample_transforms=components["sample_transforms"],
        likelihood_transforms=components["likelihood_transforms"],
        periodic=recovery_periodic,
        sampler_config=sampler_config,
        seed=int(truth["sampler_seed"]),
        verbose=args.verbose,
    )
    initial_positions = jim.sample_initial_positions(
        int(config["n_live"]),
        rng_key=jax.random.key(truth["sampler_seed"]),
    )
    setup_seconds = time.perf_counter() - setup_started

    sample_started = time.perf_counter()
    jim.sample(initial_positions)
    sample_seconds = time.perf_counter() - sample_started

    extraction_started = time.perf_counter()
    diagnostics = jim.get_diagnostics()
    samples = {name: np.asarray(values) for name, values in jim.get_samples().items()}
    missing = sorted(set(PARAMETERS) - samples.keys())
    if missing:
        raise RuntimeError("posterior is missing parameters: " + ", ".join(missing))
    counts = {name: int(values.shape[0]) for name, values in samples.items()}
    if len(set(counts.values())) != 1 or any(
        values.ndim != 1 for values in samples.values()
    ):
        raise RuntimeError(f"invalid posterior array shapes: {counts}")
    atomic_savez_compressed(posterior_path, samples)
    ranks = {
        name: posterior_rank(samples[name], float(truth[name])) for name in PARAMETERS
    }
    extraction_seconds = time.perf_counter() - extraction_started

    total_seconds = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "campaign": manifest["config"]["campaign"],
        "config_sha256": manifest["config_sha256"],
        "injection_id": args.injection_id,
        "truth": {name: truth[name] for name in (*PARAMETERS, "phase_c", "t_c")},
        "seeds": {
            "noise": truth["noise_seed"],
            "sampler": truth["sampler_seed"],
        },
        "network": {
            "optimal_snr_by_detector": {
                ifo.name: _safe_float(ifo.optimal_snr) for ifo in ifos
            },
            "matched_filter_snr_by_detector": {
                ifo.name: _safe_float(abs(ifo.match_filtered_snr)) for ifo in ifos
            },
        },
        "ranks": ranks,
        "posterior_samples": next(iter(counts.values())),
        "posterior": {
            "path": "posterior.npz",
            "sha256": file_sha256(posterior_path),
            "bytes": posterior_path.stat().st_size,
            "fields": list(samples),
        },
        "diagnostics": {
            "n_iterations": _safe_int(diagnostics.get("n_iterations")),
            "n_likelihood_evaluations": _safe_int(
                diagnostics.get("n_likelihood_evaluations")
            ),
            "log_Z": _safe_float(diagnostics.get("log_Z")),
            "log_Z_error": _safe_float(diagnostics.get("log_Z_error")),
        },
        "timing_seconds": {
            "total": total_seconds,
            "data_injection": injection_seconds,
            "problem_setup": setup_seconds,
            "sample_call": sample_seconds,
            "result_extraction": extraction_seconds,
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "blackjax": getattr(blackjax, "__version__", None),
            "jimgw": getattr(jimgw, "__version__", None),
            "ripplegw": _package_version("ripplegw"),
            "jax_compilation_cache_dir": str(cache_dir),
        },
        "devices": devices,
        "simulated_cpu": bool(args.simulate_cpu),
    }
    atomic_write_json(summary_path, summary)
    (directory / "failure.json").unlink(missing_ok=True)
    (directory / "RUNNING").unlink(missing_ok=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_injection(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
