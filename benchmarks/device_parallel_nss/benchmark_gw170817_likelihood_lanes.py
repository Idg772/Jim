"""Benchmark real GW170817 likelihood slots across vmap lane counts.

The waveform-rebuild variant includes ``build_cache`` plus evaluation from the
new cache. The cache-hit variant evaluates the same positions against caches
prepared before the timed call. Each point is a single synchronized vmap'd
likelihood slot on one visible device.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from . import benchmark_gw170817_full_run as full
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.device_parallel_nss import benchmark_gw170817_full_run as full


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-root", type=Path, default=None)
    parser.add_argument("--implementation-label", default="candidate")
    parser.add_argument("--implementation-revision", default=None)
    parser.add_argument(
        "--workload",
        choices=full.WORKLOAD_CHOICES,
        default=full.ALIGNED_WORKLOAD,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lanes", nargs="+", type=_positive_int, default=[16, 32, 64, 128]
    )
    parser.add_argument("--warmup", type=_positive_int, default=2)
    parser.add_argument("--repeats", type=_positive_int, default=10)
    parser.add_argument("--simulate-cpu", action="store_true")
    return parser.parse_args(argv)


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _build_jim(args: argparse.Namespace, jax: Any, jnp: Any):
    from jimgw.core.jim import Jim
    from jimgw.core.single_event.data import Data, PowerSpectrum
    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
    from jimgw.core.single_event.likelihood import TransientLikelihoodFD
    from jimgw.samplers.config import BlackJAXSwiGConfig

    _, arrays = full._read_bundle(
        args.data_file.expanduser().resolve(),
        args.workload,
    )
    ifos = [get_H1(), get_L1(), get_V1()]
    for ifo in ifos:
        name = ifo.name
        strain, psd = full._likelihood_inputs_from_bundle(
            name,
            arrays,
            Data=Data,
            PowerSpectrum=PowerSpectrum,
            jnp=jnp,
        )
        ifo.set_data(strain)
        ifo.set_psd(psd)

    components = full._analysis_components(args.workload, jnp, ifos)
    likelihood = TransientLikelihoodFD(
        ifos,
        waveform=components["waveform"],
        trigger_time=full.GPS,
        f_min=full.F_MIN,
        f_max=full.F_MAX,
        phase_marginalization=True,
        time_marginalization={"tc_range": (-0.03, 0.03)},
    )
    return Jim(
        likelihood,
        components["prior"],
        sample_transforms=components["sample_transforms"],
        likelihood_transforms=components["likelihood_transforms"],
        periodic=components["periodic"],
        sampler_config=BlackJAXSwiGConfig(
            blocks=[list(block) for block in components["spec"]["blocks"]],
            n_live=full.N_LIVE,
            n_delete_frac=full.N_DELETE_FRAC,
            num_inner_steps_per_dim=full.NUM_INNER_STEPS_PER_DIM,
            num_gibbs_sweeps=full.NUM_GIBBS_SWEEPS,
            termination_dlogz=full.TERMINATION_DLOGZ,
            n_devices=1,
        ),
        seed=args.seed,
    )


def _measure(
    jax: Any, function: Any, arguments: tuple[Any, ...], warmup: int, repeats: int
):
    compiled = jax.jit(function)
    started = time.perf_counter()
    first = compiled(*arguments)
    jax.block_until_ready(first)
    first_seconds = time.perf_counter() - started
    for _ in range(warmup):
        jax.block_until_ready(compiled(*arguments))

    samples = []
    checksum = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = compiled(*arguments)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - started)
        checksum = float(np.asarray(jax.device_get(result)).sum())
    return {
        "first_call_seconds_including_jit": first_seconds,
        "warmup_calls_after_first": warmup,
        "samples_seconds": samples,
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "maximum_seconds": max(samples),
        "standard_deviation_seconds": float(np.std(samples)),
        "checksum": checksum,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import jaxlib

    jax.config.update("jax_enable_x64", True)
    devices = full._device_metadata(jax, 1)
    if not args.simulate_cpu and devices["backend"] != "gpu":
        raise SystemExit(f"expected one GPU, got backend={devices['backend']}")

    jim = _build_jim(args, jax, jnp)
    lane_counts = list(dict.fromkeys(args.lanes))
    positions = jim.sample_initial_positions(
        max(lane_counts), rng_key=jax.random.key(args.seed)
    )
    build_cache = jim.sampler._build_cache
    likelihood_from_cache = jim.sampler._log_likelihood_from_cache_fn
    results: dict[str, Any] = {}

    for lanes in lane_counts:
        lane_positions = positions[:lanes]

        def rebuild_one(position):
            cache = build_cache(position)
            return likelihood_from_cache(position, cache)

        rebuild_batch = jax.named_call(
            jax.vmap(rebuild_one), name="waveform_rebuild_likelihood_slot"
        )
        rebuild = _measure(
            jax,
            rebuild_batch,
            (lane_positions,),
            args.warmup,
            args.repeats,
        )

        cache_builder = jax.jit(
            jax.named_call(jax.vmap(build_cache), name="prepare_waveform_cache")
        )
        cache_started = time.perf_counter()
        caches = cache_builder(lane_positions)
        jax.block_until_ready(caches)
        cache_prepare_seconds = time.perf_counter() - cache_started

        def hit_one(position, cache):
            return likelihood_from_cache(position, cache)

        cache_hit_batch = jax.named_call(
            jax.vmap(hit_one), name="cache_hit_likelihood_slot"
        )
        cache_hit = _measure(
            jax,
            cache_hit_batch,
            (lane_positions, caches),
            args.warmup,
            args.repeats,
        )
        results[str(lanes)] = {
            "lanes": lanes,
            "waveform_rebuild": rebuild,
            "cache_hit": cache_hit,
            "untimed_cache_prepare_seconds_including_jit": cache_prepare_seconds,
        }
        del caches, cache_builder, rebuild_batch, cache_hit_batch
        gc.collect()

    import jimgw

    inferred_root = Path(jimgw.__file__).resolve().parents[2]
    implementation_root = args.implementation_root or inferred_root
    return {
        "schema_version": 1,
        "benchmark": "gw170817-vmapped-likelihood-lanes",
        "implementation": {
            "label": args.implementation_label,
            "module_file": str(Path(jimgw.__file__).resolve()),
            **full._git_metadata(implementation_root, args.implementation_revision),
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": _package_version("numpy"),
        },
        "devices": devices,
        "data": {
            "path": str(args.data_file.expanduser().resolve()),
            "sha256": full._sha256(args.data_file.expanduser().resolve()),
        },
        "workload": {
            "name": args.workload,
            "seed": args.seed,
            "lanes": lane_counts,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dtype": "float64",
            "waveform": full._workload_spec(args.workload)["waveform"],
            "sampled_dimensions": full._workload_spec(args.workload)[
                "sampled_dimensions"
            ],
            "waveform_rebuild_definition": (
                "vmap(build_cache(position) then log_likelihood_from_cache)"
            ),
            "cache_hit_definition": (
                "vmap(log_likelihood_from_cache(position, prebuilt_cache))"
            ),
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    full._select_implementation(args.implementation_root)
    if args.simulate_cpu:
        full._configure_cpu_simulation(1)
    report = _run(args)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    full._atomic_write_report(args.output, text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
