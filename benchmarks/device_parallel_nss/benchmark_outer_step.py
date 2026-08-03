"""Benchmark Jim's fixed-state device-parallel NSS outer-step seam.

The likelihood is intentionally cheap.  This keeps the benchmark focused on
replacement-chain vectorisation, state placement, and device communication.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-live", "--N", type=_positive_int, default=512)
    parser.add_argument("--n-delete", "--K", type=_positive_int, default=64)
    parser.add_argument("--dims", type=_positive_int, default=15)
    parser.add_argument("--iterations", type=_positive_int, default=100)
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--n-devices", type=_positive_int, default=1)
    parser.add_argument("--inner-steps-per-dim", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--simulate-cpu",
        action="store_true",
        help=(
            "Expose --n-devices logical CPU devices. Omit this on a real "
            "multi-accelerator host."
        ),
    )
    return parser.parse_args()


def _configure_cpu_simulation(n_devices: int) -> None:
    desired = f"--xla_force_host_platform_device_count={n_devices}"
    existing = os.environ.get("XLA_FLAGS", "").split()
    configured = [
        token
        for token in existing
        if token.startswith("--xla_force_host_platform_device_count=")
    ]
    if configured and configured != [desired]:
        raise SystemExit(
            "XLA_FLAGS already contains a conflicting host device count: "
            + " ".join(configured)
        )
    if not configured:
        os.environ["XLA_FLAGS"] = " ".join([*existing, desired]).strip()
    configured_platforms = os.environ.get("JAX_PLATFORMS")
    if configured_platforms and configured_platforms.split(",", 1)[0] != "cpu":
        raise SystemExit(
            "--simulate-cpu conflicts with JAX_PLATFORMS=" + configured_platforms
        )
    os.environ["JAX_PLATFORMS"] = "cpu"


def _normalise_hlo_instruction(line: str) -> str:
    return " ".join(line.strip().split())


def _collective_census(hlo: str) -> dict[str, dict[str, Any]]:
    census: dict[str, dict[str, Any]] = {}
    for operation in ("all-gather", "all-reduce", "collective-permute"):
        pattern = re.compile(rf"(^|[^a-z-]){re.escape(operation)}\(")
        instructions = [
            _normalise_hlo_instruction(line)
            for line in hlo.splitlines()
            if pattern.search(line)
        ]
        census[operation] = {
            "count": len(instructions),
            "instructions": instructions,
        }
    return census


def _describe_placements(jax: Any, tree: Any) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []
    for path, leaf in jax.tree.flatten_with_path(tree)[0]:
        sharding = leaf.sharding
        spec = getattr(sharding, "spec", None)
        leaves.append(
            {
                "path": jax.tree_util.keystr(path) or ".",
                "shape": list(leaf.shape),
                "dtype": str(leaf.dtype),
                "sharding": str(spec if spec is not None else sharding),
                "fully_replicated": bool(sharding.is_fully_replicated),
            }
        )
    return {
        "leaf_count": len(leaves),
        "fully_replicated_leaf_count": sum(leaf["fully_replicated"] for leaf in leaves),
        "sharding_specs": sorted({leaf["sharding"] for leaf in leaves}),
        "leaves": leaves,
    }


def _timing_summary(step_times_ms: list[float]) -> dict[str, Any]:
    ordered = sorted(step_times_ms)

    def percentile(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return ordered[index]

    return {
        "mean": statistics.fmean(step_times_ms),
        "median": statistics.median(step_times_ms),
        "p05": percentile(0.05),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
        "samples": step_times_ms,
    }


def _git_metadata() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        result = subprocess.run(
            ("git", *args),
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "revision": revision,
        "dirty": bool(status) if status is not None else None,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import jaxlib

    jax.config.update("jax_enable_x64", args.dtype == "float64")
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32

    import blackjax
    from blackjax.ns.adaptive import init as adaptive_init
    from blackjax.ns.base import init_state_strategy

    from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler
    from jimgw.samplers.blackjax.sharding import (
        make_live_mesh,
        place_key,
        place_replicated_state,
    )
    from jimgw.samplers.config import BlackJAXNSSConfig

    if args.n_delete >= args.n_live:
        raise SystemExit("--n-delete must be smaller than --n-live")

    n_delete_frac = args.n_delete / args.n_live
    if int(args.n_live * n_delete_frac) != args.n_delete:
        raise SystemExit(
            "--n-delete/--n-live cannot be represented by the sampler's "
            "n_delete_frac calculation"
        )

    def log_prior(position):
        inside = jnp.all((position >= 0.0) & (position <= 1.0))
        return jnp.where(inside, jnp.asarray(0.0, dtype=dtype), -jnp.inf)

    def log_likelihood(position):
        return -20.0 * jnp.sum((position - 0.5) ** 2)

    config = BlackJAXNSSConfig(
        n_live=args.n_live,
        n_delete_frac=n_delete_frac,
        num_inner_steps_per_dim=args.inner_steps_per_dim,
        termination_dlogz=2.0,
        n_devices=args.n_devices,
    )
    sampler = BlackJAXNSSSampler(
        n_dims=args.dims,
        log_prior_fn=log_prior,
        log_likelihood_fn=log_likelihood,
        log_posterior_fn=lambda x: log_prior(x) + log_likelihood(x),
        config=config,
    )
    mesh = make_live_mesh(args.n_devices, args.n_live, args.n_delete)

    setup_started = time.perf_counter()
    positions = jax.random.uniform(
        jax.random.key(args.seed),
        (args.n_live, args.dims),
        dtype=dtype,
    )
    single_init = partial(
        init_state_strategy,
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
    )
    state = adaptive_init(
        positions,
        init_state_fn=lambda pos: jax.lax.map(
            single_init,
            pos,
            batch_size=args.n_delete,
        ),
        update_inner_kernel_params_fn=sampler._update_inner_kernel_params_fn,
    )
    if mesh is not None:
        state = place_replicated_state(state, mesh)

    raw_keys = jax.random.split(
        jax.random.key(args.seed + 1),
        args.warmup + args.iterations,
    )
    keys = [place_key(key, mesh) if mesh is not None else key for key in raw_keys]
    jax.block_until_ready((state, tuple(keys)))
    setup_seconds = time.perf_counter() - setup_started

    step = jax.jit(sampler._build_nested_sampler(args.n_delete, mesh).step)
    lower_started = time.perf_counter()
    lowered = step.lower(keys[0], state)
    lower_seconds = time.perf_counter() - lower_started
    compile_started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started

    output = None
    for key in keys[: args.warmup]:
        output = executable(key, state)
        jax.block_until_ready(output)

    step_times_ms: list[float] = []
    for key in keys[args.warmup :]:
        step_started = time.perf_counter_ns()
        output = executable(key, state)
        jax.block_until_ready(output)
        elapsed_ns = time.perf_counter_ns() - step_started
        step_times_ms.append(elapsed_ns / 1_000_000)

    assert output is not None
    live_state, dead_history = output
    devices = jax.local_devices()
    return {
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "operating_system": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "blackjax": blackjax.__version__,
            "backend": jax.default_backend(),
            "local_device_count": len(devices),
            "local_devices": [
                {
                    "description": str(device),
                    "device_kind": getattr(device, "device_kind", None),
                    "id": device.id,
                    "platform": device.platform,
                    "process_index": device.process_index,
                }
                for device in devices
            ],
            "git": _git_metadata(),
        },
        "workload": {
            "n_live": args.n_live,
            "n_delete": args.n_delete,
            "dims": args.dims,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "n_devices": args.n_devices,
            "inner_steps_per_dim": args.inner_steps_per_dim,
            "dtype": args.dtype,
            "seed": args.seed,
            "fixed_state": True,
            "simulate_cpu": args.simulate_cpu,
        },
        "timing": {
            "setup_seconds": setup_seconds,
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
            "lower_and_compile_seconds": lower_seconds + compile_seconds,
            "warmed_step_ms": _timing_summary(step_times_ms),
        },
        "collectives": _collective_census(executable.as_text()),
        "placement": {
            "live_state": _describe_placements(jax, live_state),
            "dead_history": _describe_placements(jax, dead_history),
        },
    }


def main() -> None:
    args = _parse_args()
    if args.simulate_cpu:
        _configure_cpu_simulation(args.n_devices)
    report = _run(args)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
