"""Run one frozen paper-methodology SwiG injection recovery."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp

from benchmarks.injection_campaign.common import (
    MARGINALIZED_PARAMETERS,
    PARAMETERS,
    atomic_savez_compressed,
    atomic_write_json,
    file_sha256,
    load_manifest,
    posterior_rank,
    read_catalogue,
    result_dir,
    validate_completed_result,
)

_SIMPLIFIED_CONSTANTS_ENV = "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"
_EMBEDDED_CONSTANTS_ENV = "JAX_EMBEDDED_CONSTANTS_MAX_BYTES"
_DEFAULT_EMBEDDED_CONSTANTS_MAX_BYTES = 32
_TRUE_ENV_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})


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
    parser.add_argument(
        "--jax-cache-diagnostics",
        action="store_true",
        help=(
            "Record compact persistent-cache hit/miss counts and cache-directory "
            "inventory in the result summary."
        ),
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


def _configure_jax_constant_environment() -> dict[str, Any]:
    """Configure closed-over array handling before importing JAX.

    Campaign likelihoods close over the detector strain arrays.  Hoisting those
    large arrays as executable arguments keeps their values out of the HLO and
    persistent-cache key while preserving JAX's default 32-byte threshold for
    genuinely small constants.
    """

    jax_was_imported = "jax" in sys.modules
    simplified = os.environ.setdefault(_SIMPLIFIED_CONSTANTS_ENV, "True")
    if simplified.strip().lower() not in _TRUE_ENV_VALUES:
        raise RuntimeError(
            f"{_SIMPLIFIED_CONSTANTS_ENV} must be true for campaign recoveries"
        )
    embedded_raw = os.environ.setdefault(
        _EMBEDDED_CONSTANTS_ENV,
        str(_DEFAULT_EMBEDDED_CONSTANTS_MAX_BYTES),
    )
    try:
        embedded_max_bytes = int(embedded_raw)
    except ValueError as error:
        raise RuntimeError(
            f"{_EMBEDDED_CONSTANTS_ENV} must be a non-negative integer"
        ) from error
    if embedded_max_bytes < 0:
        raise RuntimeError(f"{_EMBEDDED_CONSTANTS_ENV} must be a non-negative integer")
    return {
        "simplified_jaxpr_constants_environment": simplified,
        "embedded_constants_max_bytes_environment": embedded_raw,
        "configured_before_jax_import": not jax_was_imported,
    }


def _read_jax_config(jax: Any, name: str) -> Any:
    value = getattr(jax.config, name, None)
    if value is not None:
        return value
    try:
        return jax.config.read(name)
    except (AttributeError, KeyError):
        return None


def _jax_constant_handling_report(
    jax: Any, preimport_report: Mapping[str, Any]
) -> dict[str, Any]:
    simplified = _read_jax_config(jax, "jax_use_simplified_jaxpr_constants")
    embedded_max_bytes = _read_jax_config(jax, "jax_embedded_constants_max_bytes")
    if simplified is False:
        raise RuntimeError(
            "JAX imported without simplified closed-over constant handling enabled"
        )
    requested_max_bytes = int(
        str(preimport_report["embedded_constants_max_bytes_environment"])
    )
    if (
        embedded_max_bytes is not None
        and int(embedded_max_bytes) != requested_max_bytes
    ):
        raise RuntimeError(
            "JAX embedded-constant threshold does not match the pre-import request"
        )
    return {
        **preimport_report,
        "simplified_jaxpr_constants_effective": (
            bool(simplified) if simplified is not None else None
        ),
        "embedded_constants_max_bytes_effective": (
            int(embedded_max_bytes) if embedded_max_bytes is not None else None
        ),
        "assertion": (
            "effective"
            if simplified is True
            else "configuration-not-exposed-by-this-jax-release"
        ),
    }


def _cache_inventory(cache_dir: Path) -> dict[str, int]:
    files = 0
    total_bytes = 0
    try:
        paths = cache_dir.rglob("*")
        for path in paths:
            try:
                if path.is_file():
                    files += 1
                    total_bytes += path.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return {"files": files, "bytes": total_bytes}


class _JaxCompilerDiagnostics(logging.Handler):
    """Collect bounded, public-summary-safe compiler/cache diagnostics."""

    def __init__(self, cache_dir: Path) -> None:
        super().__init__(level=logging.DEBUG)
        self.cache_dir = cache_dir
        self._counts: dict[str, int] = {}
        self._modules: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        kind: str | None = None
        if "Persistent compilation cache hit" in message:
            kind = "persistent_cache_hits"
        elif "PERSISTENT COMPILATION CACHE MISS" in message:
            kind = "persistent_cache_misses"
        elif "Not writing persistent cache entry" in message:
            kind = "persistent_cache_write_skips"
        elif message.startswith("Compiling "):
            kind = "compilation_requests"
        if kind is None:
            return
        self._counts[kind] = self._counts.get(kind, 0) + 1
        if isinstance(record.args, tuple) and record.args:
            module = record.args[0]
            if isinstance(module, str) and len(self._modules) < 64:
                self._modules.add(module)

    def begin_event(self) -> dict[str, int]:
        self._counts.clear()
        self._modules.clear()
        return _cache_inventory(self.cache_dir)

    def finish_event(self, before: Mapping[str, int]) -> dict[str, Any]:
        after = _cache_inventory(self.cache_dir)
        return {
            "compiler_events": dict(sorted(self._counts.items())),
            "modules": sorted(self._modules),
            "cache_before": dict(before),
            "cache_after": after,
            "cache_delta": {
                "files": after["files"] - int(before["files"]),
                "bytes": after["bytes"] - int(before["bytes"]),
            },
        }


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


@dataclass(frozen=True)
class InjectionRuntime:
    """Process-local infrastructure shared by sequential recoveries.

    Scientific event objects deliberately do not live here: detectors,
    likelihood, Jim, sampler state, and PRNG streams are reconstructed for every
    call to :func:`run_injection`.
    """

    campaign_dir: Path
    manifest: dict[str, Any]
    catalogue: list[dict[str, Any]]
    config: dict[str, Any]
    cache_dir: Path
    simulate_cpu: bool
    jax: Any
    jnp: Any
    jaxlib: Any
    blackjax: Any
    jimgw: Any
    Jim: type[Any]
    PowerSpectrum: type[Any]
    detector_factories: tuple[Any, Any, Any]
    TransientLikelihoodFD: type[Any]
    BlackJAXSwiGConfig: type[Any]
    devices: dict[str, Any]
    constant_handling: dict[str, Any]
    cache_diagnostics: _JaxCompilerDiagnostics | None
    initialization_seconds: float


def prepare_injection_runtime(
    campaign_dir: Path,
    *,
    simulate_cpu: bool,
    jax_compilation_cache_dir: Path | None = None,
    jax_cache_diagnostics: bool = False,
) -> InjectionRuntime:
    """Initialize JAX once for one or more sequential campaign recoveries."""

    initialization_started = time.perf_counter()
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    config = manifest["config"]
    n_devices = int(config["n_devices"])
    if simulate_cpu:
        _configure_cpu_simulation(n_devices)
    preimport_report = _configure_jax_constant_environment()

    import jax
    import jax.numpy as jnp
    import jaxlib

    jax.config.update("jax_enable_x64", True)
    constant_handling = _jax_constant_handling_report(jax, preimport_report)
    cache_dir = (
        jax_compilation_cache_dir.expanduser().resolve()
        if jax_compilation_cache_dir is not None
        else campaign_dir / ".jax-cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    compiler_diagnostics: _JaxCompilerDiagnostics | None = None
    if jax_cache_diagnostics:
        for name in (
            "jax_log_compiles",
            "jax_explain_cache_misses",
            "jax_raise_persistent_cache_errors",
        ):
            try:
                jax.config.update(name, True)
            except AttributeError:
                continue
        compiler_diagnostics = _JaxCompilerDiagnostics(cache_dir)
        compiler_logger = logging.getLogger("jax._src.compiler")
        compiler_logger.setLevel(logging.DEBUG)
        compiler_logger.addHandler(compiler_diagnostics)

    import blackjax

    import jimgw
    from jimgw.core.jim import Jim
    from jimgw.core.single_event.data import PowerSpectrum
    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
    from jimgw.core.single_event.likelihood import TransientLikelihoodFD
    from jimgw.samplers.config import BlackJAXSwiGConfig

    devices = _device_report(jax, n_devices, simulate_cpu)
    catalogue = read_catalogue(campaign_dir / manifest["catalogue"]["path"])
    return InjectionRuntime(
        campaign_dir=campaign_dir,
        manifest=manifest,
        catalogue=catalogue,
        config=config,
        cache_dir=cache_dir,
        simulate_cpu=simulate_cpu,
        jax=jax,
        jnp=jnp,
        jaxlib=jaxlib,
        blackjax=blackjax,
        jimgw=jimgw,
        Jim=Jim,
        PowerSpectrum=PowerSpectrum,
        detector_factories=(get_H1, get_L1, get_V1),
        TransientLikelihoodFD=TransientLikelihoodFD,
        BlackJAXSwiGConfig=BlackJAXSwiGConfig,
        devices=devices,
        constant_handling=constant_handling,
        cache_diagnostics=compiler_diagnostics,
        initialization_seconds=time.perf_counter() - initialization_started,
    )


def _injection_parameters(
    truth: dict[str, Any], likelihood_transforms: list[Any]
) -> dict[str, Any]:
    parameters = {name: truth[name] for name in (*PARAMETERS, *MARGINALIZED_PARAMETERS)}
    for transform in likelihood_transforms:
        parameters = transform.forward(parameters)
    return parameters


def _time_marginalization_settings(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return typed-likelihood settings for a time-marginalized campaign."""

    raw = config.get("time_marginalization", False)
    if raw is False or raw is None:
        return None
    if raw is True:
        return {
            "tc_range": tuple(config["prior"]["t_c"]["range_seconds"]),
            "upsample_factor": 1,
        }
    if not isinstance(raw, Mapping):
        raise TypeError("time_marginalization must be false or a mapping")
    tc_range = raw.get("tc_range_seconds")
    if (
        not isinstance(tc_range, list | tuple)
        or len(tc_range) != 2
        or any(isinstance(value, bool) for value in tc_range)
    ):
        raise ValueError(
            "time_marginalization.tc_range_seconds must contain two numbers"
        )
    upsample_factor = raw.get("upsample_factor", 1)
    if type(upsample_factor) is not int or upsample_factor < 1:
        raise ValueError(
            "time_marginalization.upsample_factor must be an exact positive integer"
        )
    return {
        "tc_range": tuple(float(value) for value in tc_range),
        "upsample_factor": upsample_factor,
    }


def _sampled_parameters(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return direct posterior fields for the configured nuisance treatment."""

    if _time_marginalization_settings(config) is not None:
        return tuple(name for name in PARAMETERS if name != "t_c") + ("d_L",)
    return PARAMETERS


def _marginalized_parameters(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return analytically integrated physical parameters."""

    names: list[str] = []
    if bool(config.get("phase_marginalization", False)):
        names.append("phase_c")
    if _time_marginalization_settings(config) is not None:
        names.append("t_c")
    else:
        names.append("d_L")
    return tuple(names)


def _rank_truth_coordinates(
    truth: Mapping[str, Any],
    sampled_parameters: tuple[str, ...],
    *,
    phase_marginalization: bool,
) -> dict[str, float]:
    """Express catalogue truths in the coordinates retained by the posterior.

    With phase marginalization, the precessing waveform fixes the orbital-phase
    gauge and retains ``beta_i = (s_i_phi + phase_c) mod 2 pi``.  The posterior
    arrays therefore cannot be ranked against the catalogue's ungauged spin
    azimuths directly.  All other sampled coordinates retain their catalogue
    values.
    """

    coordinates = {name: float(truth[name]) for name in sampled_parameters}
    if phase_marginalization:
        phase_c = float(truth["phase_c"])
        for name in ("s1_phi", "s2_phi"):
            if name in coordinates:
                coordinates[name] = float(
                    np.mod(coordinates[name] + phase_c, 2.0 * np.pi)
                )
    return coordinates


def _transient_likelihood_kwargs(
    config: Mapping[str, Any], components: Mapping[str, Any]
) -> dict[str, Any]:
    """Build one source of truth for campaign and preflight likelihood wiring."""

    time_settings = _time_marginalization_settings(config)
    distance_config = config.get("distance_marginalization")
    if time_settings is not None:
        if distance_config not in (False, None):
            raise ValueError(
                "time-marginalized campaigns must disable distance marginalization"
            )
        distance_settings = None
    else:
        if not isinstance(distance_config, Mapping):
            raise TypeError(
                "distance-marginalized campaigns require a configuration mapping"
            )
        distance_settings = {
            "distance_prior": components["distance_prior"],
            "n_dist_points": int(distance_config["n_grid_points"]),
        }
    return {
        "trigger_time": float(config["trigger_time_gps"]),
        "f_min": float(config["f_min_hz"]),
        "f_max": float(config.get("likelihood_f_max_hz", config["f_max_hz"])),
        "phase_marginalization": bool(config["phase_marginalization"]),
        "time_marginalization": time_settings,
        "distance_marginalization": distance_settings,
    }


def _analysis_components(
    config: dict[str, Any], jnp: Any, ifos: list[Any]
) -> dict[str, Any]:
    """Construct the paper injection prior, waveform, and parameter transforms."""

    from benchmarks.device_parallel_nss.paper_model import (
        RippleIMRPhenomPv2NRTidalv2,
    )
    from jimgw.core.prior import (
        CombinePrior,
        CosinePrior,
        PowerLawPrior,
        SinePrior,
        UniformPrior,
        UniformSpherePrior,
    )
    from jimgw.core.single_event.transforms import (
        GeocentricArrivalTimeToDetectorArrivalTimeTransform,
        MassRatioToSymmetricMassRatioTransform,
        SkyFrameToDetectorFrameSkyPositionTransform,
        SphereSpinToCartesianSpinTransform,
    )

    prior_config = config["prior"]
    distance_config = prior_config["d_L"]
    distance_prior = PowerLawPrior(
        *distance_config["range_mpc"],
        float(distance_config["alpha"]),
        parameter_names=["d_L"],
    )
    prior_components = [
        UniformPrior(*prior_config["M_c"]["range"], parameter_names=["M_c"]),
        UniformPrior(*prior_config["q"]["range"], parameter_names=["q"]),
        UniformSpherePrior(
            parameter_names=["s1"],
            max_mag=float(prior_config["spin_magnitudes"]["range"][1]),
        ),
        UniformSpherePrior(
            parameter_names=["s2"],
            max_mag=float(prior_config["spin_magnitudes"]["range"][1]),
        ),
        SinePrior(parameter_names=["iota"]),
        UniformPrior(
            *prior_config["lambda_1"]["range"],
            parameter_names=["lambda_1"],
        ),
        UniformPrior(
            *prior_config["lambda_2"]["range"],
            parameter_names=["lambda_2"],
        ),
        UniformPrior(0.0, 2.0 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
    ]
    if _time_marginalization_settings(config) is not None:
        prior_components.append(distance_prior)
    else:
        prior_components.append(
            UniformPrior(
                *prior_config["t_c"]["range_seconds"],
                parameter_names=["t_c"],
            )
        )
    prior = CombinePrior(prior_components)
    time_sampling_frame = config.get("time_sampling_frame", "geocentric")
    if not isinstance(time_sampling_frame, str) or not time_sampling_frame:
        raise ValueError("time_sampling_frame must be 'geocentric' or a detector name")
    sample_transforms: list[Any] = []
    if time_sampling_frame != "geocentric":
        if _time_marginalization_settings(config) is not None:
            raise ValueError(
                "detector-arrival-time sampling requires sampled geocentric time"
            )
        try:
            time_ifo = next(ifo for ifo in ifos if ifo.name == time_sampling_frame)
        except StopIteration as error:
            raise ValueError(
                f"time_sampling_frame names unknown detector {time_sampling_frame!r}"
            ) from error
        # The time transform must run while physical ra/dec are still present.
        # Reverse posterior transforms then recover ra/dec from detector sky
        # before converting t_det back to physical t_c.
        sample_transforms.append(
            GeocentricArrivalTimeToDetectorArrivalTimeTransform(
                trigger_time=float(config["trigger_time_gps"]),
                ifo=time_ifo,
            )
        )
    sample_transforms.append(
        SkyFrameToDetectorFrameSkyPositionTransform(
            trigger_time=float(config["trigger_time_gps"]),
            ifos=ifos,
        )
    )

    return {
        "waveform": RippleIMRPhenomPv2NRTidalv2(
            f_ref=float(config["waveform_f_ref_hz"]),
            time_anchor=str(config["carrier_time_anchor"]),
        ),
        "prior": prior,
        "distance_prior": distance_prior,
        "sample_transforms": sample_transforms,
        "likelihood_transforms": [
            MassRatioToSymmetricMassRatioTransform,
            SphereSpinToCartesianSpinTransform("s1"),
            SphereSpinToCartesianSpinTransform("s2"),
        ],
        "periodic": {
            "s1_phi": (0.0, 2.0 * float(jnp.pi)),
            "s2_phi": (0.0, 2.0 * float(jnp.pi)),
            "azimuth": (0.0, 2.0 * float(jnp.pi)),
            "psi": (0.0, float(jnp.pi)),
        },
    }


def _paper_convention_timing(
    sample_call_seconds: float,
    sample_phases: object,
) -> dict[str, Any]:
    """Derive the Figure 3 wall time by excluding both one-off JIT phases."""

    if not isinstance(sample_phases, dict):
        raise TypeError("sampler did not report phase timings")
    components: dict[str, float] = {}
    for field in ("likelihood_jit", "sampler_kernel_jit"):
        value = _safe_float(sample_phases.get(field))
        if value is None or value < 0.0:
            raise RuntimeError(f"sampler did not report a valid {field} timing")
        components[field] = value
    post_jit = (
        float(sample_call_seconds)
        - components["likelihood_jit"]
        - components["sampler_kernel_jit"]
    )
    if not np.isfinite(post_jit) or post_jit <= 0.0:
        raise RuntimeError(
            "post-JIT sampling time is non-positive after subtracting compile phases"
        )
    return {
        "paper_reference": "arXiv:2607.28265v1 Figure 3 and Table III",
        "definition": (
            "sample_call - likelihood_jit - sampler_kernel_jit; excludes the "
            "two one-off compilation costs paid once per run"
        ),
        "likelihood_jit_seconds": components["likelihood_jit"],
        "sampler_jit_seconds": components["sampler_kernel_jit"],
        "post_jit_sampling_seconds": post_jit,
    }


def _is_paper_baseline(manifest: dict[str, Any]) -> bool:
    diagnostic = manifest.get("baseline_diagnostic")
    return isinstance(diagnostic, dict) and diagnostic.get("implementation_label") == (
        "paper-baseline"
    )


def _pinned_implementation_diagnostic(
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    baseline = manifest.get("baseline_diagnostic")
    candidate = manifest.get("implementation_diagnostic")
    pins = [value for value in (baseline, candidate) if isinstance(value, dict)]
    if len(pins) > 1:
        raise RuntimeError("campaign contains ambiguous implementation pins")
    return pins[0] if pins else None


def _implementation_report(
    manifest: dict[str, Any], jimgw: Any, config_type: type[Any]
) -> dict[str, Any]:
    """Verify and report the implementation selected by the pod runner."""

    label = os.environ.get("JIM_IMPLEMENTATION_LABEL", "candidate")
    revision = os.environ.get("JIM_IMPLEMENTATION_REVISION")
    tree_sha256 = os.environ.get("JIM_IMPLEMENTATION_TREE_SHA256")
    root_value = os.environ.get("JIM_IMPLEMENTATION_ROOT")
    module_path = Path(jimgw.__file__).resolve()
    diagnostic = _pinned_implementation_diagnostic(manifest)
    if diagnostic is not None:
        expected_label = diagnostic.get("implementation_label")
        expected_revision = diagnostic.get("implementation_revision")
        expected_tree_sha256 = diagnostic.get("implementation_tree_sha256")
        if (
            label != expected_label
            or revision != expected_revision
            or tree_sha256 != expected_tree_sha256
        ):
            raise RuntimeError(
                "campaign requires the pinned implementation: "
                f"expected revision={expected_revision}, tree={expected_tree_sha256}; "
                f"got label={label!r}, revision={revision!r}, tree={tree_sha256!r}"
            )
        if not root_value:
            raise RuntimeError("pinned implementation root is not configured")
        root = Path(root_value).resolve()
        try:
            module_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"jimgw imported from {module_path}, outside pinned root {root}"
            ) from error
        fields = getattr(config_type, "model_fields", {})
        if expected_label == "paper-baseline" and "scheduler" in fields:
            raise RuntimeError(
                "paper-baseline import unexpectedly exposes the candidate scheduler API"
            )
        if expected_label == "candidate" and "scheduler" not in fields:
            raise RuntimeError("candidate import does not expose the FSM scheduler API")
    return {
        "label": label,
        "revision": revision,
        "tree_sha256": tree_sha256,
        "root": str(Path(root_value).resolve()) if root_value else None,
        "jimgw_module": str(module_path),
    }


def _build_sampler_config(config: dict[str, Any], config_type: type[Any]) -> Any:
    """Build SwiG config across the pinned paper and candidate APIs."""

    kwargs: dict[str, Any] = {
        "blocks": config["blocks"],
        "n_live": int(config["n_live"]),
        "n_delete_frac": float(config["n_delete_frac"]),
        "num_inner_steps_per_dim": int(config["num_inner_steps_per_dim"]),
        "num_gibbs_sweeps": int(config["num_gibbs_sweeps"]),
        "termination_dlogz": float(config["termination_dlogz"]),
        "n_devices": int(config["n_devices"]),
    }
    if "scheduler" in getattr(config_type, "model_fields", {}):
        kwargs["scheduler"] = str(config.get("sampler_scheduler", "fsm"))
    return config_type(**kwargs)


def _weighted_samples(jim: Any, jax: Any, jnp: Any) -> dict[str, np.ndarray]:
    """Return direct NS weights, including on the pinned pre-API baseline."""

    getter = getattr(jim, "get_weighted_samples", None)
    if callable(getter):
        return {name: np.asarray(values) for name, values in getter().items()}

    sampler = getattr(jim, "sampler", None)
    nested = getattr(sampler, "_nested_samples", None)
    n_dims = getattr(sampler, "n_dims", None)
    if nested is None or not isinstance(n_dims, int):
        raise RuntimeError(
            "pinned implementation retained no direct nested-sampling collection"
        )
    sample_array = np.asarray(nested.iloc[:, :n_dims])
    log_likelihood = np.asarray(nested["logL"])
    log_likelihood_birth = np.asarray(nested["logL_birth"])
    log_weights = np.asarray(nested.logw(), dtype=np.float64)
    log_weights = log_weights - logsumexp(log_weights)

    named = jax.vmap(jim.add_name)(jnp.asarray(sample_array))
    for transform in reversed(jim.sample_transforms):
        named = jax.vmap(transform.backward)(named)
    result = {name: np.asarray(named[name]) for name in jim.prior_parameter_names}
    result["log_likelihood"] = log_likelihood
    result["log_likelihood_birth"] = log_likelihood_birth
    result["log_weights"] = log_weights
    return result


def run_injection(
    args: argparse.Namespace,
    *,
    runtime: InjectionRuntime | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    shared_runtime = runtime is not None
    if runtime is not None and runtime.campaign_dir != campaign_dir:
        raise ValueError(
            "injection runtime belongs to a different campaign: "
            f"{runtime.campaign_dir} != {campaign_dir}"
        )
    manifest = runtime.manifest if runtime is not None else load_manifest(campaign_dir)
    n_injections = int(manifest["n_injections"])
    if args.injection_id < 0 or args.injection_id >= n_injections:
        raise SystemExit(
            f"injection_id {args.injection_id} is outside [0, {n_injections})"
        )
    config = manifest["config"]
    directory = result_dir(campaign_dir, args.injection_id)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    posterior_path = directory / "posterior.npz"
    catalogue = (
        runtime.catalogue
        if runtime is not None
        else read_catalogue(campaign_dir / manifest["catalogue"]["path"])
    )
    truth = catalogue[args.injection_id]
    if summary_path.is_file() and posterior_path.is_file() and not args.force:
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            if existing.get("config_sha256") != manifest["config_sha256"]:
                raise SystemExit(
                    f"completed result has a different campaign hash: {directory}"
                )
            try:
                return validate_completed_result(
                    directory,
                    manifest["config_sha256"],
                    injection_id=args.injection_id,
                    catalogue_row=truth,
                )
            except (OSError, TypeError, ValueError):
                # A stale/corrupt payload is pending work, just as it is in the
                # campaign runner.  Recompute it rather than returning it.
                pass

    if runtime is None:
        runtime = prepare_injection_runtime(
            campaign_dir,
            simulate_cpu=bool(args.simulate_cpu),
            jax_compilation_cache_dir=args.jax_compilation_cache_dir,
            jax_cache_diagnostics=bool(getattr(args, "jax_cache_diagnostics", False)),
        )
    elif bool(args.simulate_cpu) != runtime.simulate_cpu:
        raise ValueError("injection arguments disagree with the shared runtime backend")

    jax = runtime.jax
    jnp = runtime.jnp
    cache_diagnostics_before = (
        runtime.cache_diagnostics.begin_event()
        if runtime.cache_diagnostics is not None
        else None
    )

    # Every event receives fresh mutable scientific objects.  Only process-level
    # imports, the device client, the catalogue, and cache instrumentation are
    # retained by InjectionRuntime.
    ifos = [factory() for factory in runtime.detector_factories]
    psd_files = manifest["psd"]["detector_files"]
    for ifo in ifos:
        ifo.set_psd(
            runtime.PowerSpectrum.from_file(str(campaign_dir / psd_files[ifo.name]))
        )

    components = _analysis_components(config, jnp, ifos)
    injection_parameters = _injection_parameters(
        truth, components["likelihood_transforms"]
    )
    injection_started = time.perf_counter()
    noise_key = jax.random.key(truth["noise_seed"])
    duration = float(config["duration_seconds"])
    center_offset = float(config["segment_center_offset_seconds"])
    start_offset = float(config["segment_start_offset_seconds"])
    expected_start_offset = center_offset - duration / 2.0
    if not np.isclose(start_offset, expected_start_offset, rtol=0.0, atol=1e-12):
        raise ValueError(
            "segment_start_offset_seconds is inconsistent with the configured "
            "duration and segment center"
        )
    start_time = float(config["trigger_time_gps"]) + start_offset
    for detector_index, ifo in enumerate(ifos):
        ifo.inject_signal(
            duration=duration,
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
    likelihood = runtime.TransientLikelihoodFD(
        ifos,
        waveform=components["waveform"],
        **_transient_likelihood_kwargs(config, components),
    )
    implementation = _implementation_report(
        manifest, runtime.jimgw, runtime.BlackJAXSwiGConfig
    )
    sampler_config = _build_sampler_config(config, runtime.BlackJAXSwiGConfig)
    jim = runtime.Jim(
        likelihood,
        components["prior"],
        sample_transforms=components["sample_transforms"],
        likelihood_transforms=components["likelihood_transforms"],
        periodic=components["periodic"],
        sampler_config=sampler_config,
        seed=int(truth["sampler_seed"]),
        verbose=args.verbose,
    )
    # Jim derives disjoint initial-live-set and sampler streams from the one
    # per-event sampler seed, so no PRNG key is reused across those stages.
    initial_positions = jim.sample_initial_positions(int(config["n_live"]))
    setup_seconds = time.perf_counter() - setup_started

    sample_started = time.perf_counter()
    jim.sample(initial_positions)
    sample_seconds = time.perf_counter() - sample_started

    extraction_started = time.perf_counter()
    diagnostics = jim.get_diagnostics()
    sample_phases = diagnostics.get("sample_phase_seconds")
    timing_unavailable_reason: str | None = None
    try:
        paper_timing = _paper_convention_timing(sample_seconds, sample_phases)
    except (TypeError, RuntimeError):
        if not _is_paper_baseline(manifest):
            raise
        paper_timing = None
        timing_unavailable_reason = (
            "The pinned paper baseline predates split likelihood/sampler JIT "
            "phase instrumentation; no post-JIT timing is inferred."
        )
    weighted = _weighted_samples(jim, jax, jnp)
    log_weights = weighted.pop("log_weights")
    log_likelihood = weighted.pop("log_likelihood")
    log_likelihood_birth = weighted.pop("log_likelihood_birth")
    samples = weighted
    sampled_parameters = _sampled_parameters(config)
    missing = sorted(set(sampled_parameters) - samples.keys())
    if missing:
        raise RuntimeError("posterior is missing parameters: " + ", ".join(missing))
    counts = {name: int(values.shape[0]) for name, values in samples.items()}
    if len(set(counts.values())) != 1 or any(
        values.ndim != 1 for values in samples.values()
    ):
        raise RuntimeError(f"invalid posterior array shapes: {counts}")
    sample_count = next(iter(counts.values()))
    if (
        log_weights.shape != (sample_count,)
        or log_likelihood.shape != (sample_count,)
        or log_likelihood_birth.shape != (sample_count,)
    ):
        raise RuntimeError(
            "weighted posterior metadata does not align with parameter samples"
        )
    from jimgw.samplers.diagnostics import insertion_index_diagnostic

    insertion_diagnostic = insertion_index_diagnostic(
        log_likelihood,
        log_likelihood_birth,
        n_live=int(config["n_live"]),
    )
    atomic_savez_compressed(
        posterior_path,
        {
            **samples,
            "log_likelihood": log_likelihood,
            "log_likelihood_birth": log_likelihood_birth,
            "log_weights": log_weights,
        },
    )
    rank_truth = _rank_truth_coordinates(
        truth,
        sampled_parameters,
        phase_marginalization=bool(config["phase_marginalization"]),
    )
    ranks = {
        name: posterior_rank(samples[name], rank_truth[name], log_weights)
        for name in sampled_parameters
    }
    weights = np.exp(log_weights)
    effective_sample_size = float(1.0 / np.sum(weights**2))
    extraction_seconds = time.perf_counter() - extraction_started

    cache_diagnostics = (
        runtime.cache_diagnostics.finish_event(cache_diagnostics_before)
        if runtime.cache_diagnostics is not None
        and cache_diagnostics_before is not None
        else None
    )
    total_seconds = time.perf_counter() - started
    summary = {
        "schema_version": 2,
        "campaign": manifest["config"]["campaign"],
        "config_sha256": manifest["config_sha256"],
        "injection_id": args.injection_id,
        "truth": {
            name: truth[name] for name in (*PARAMETERS, *MARGINALIZED_PARAMETERS)
        },
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
            "optimal_snr": _safe_float(
                np.sqrt(sum(float(ifo.optimal_snr) ** 2 for ifo in ifos))
            ),
        },
        "ranks": ranks,
        "rank_truth": rank_truth,
        "rank_coordinates": {
            "default": "catalogue physical coordinate",
            "phase_marginalized_spin_azimuths": {
                "parameters": [
                    name for name in ("s1_phi", "s2_phi") if name in sampled_parameters
                ],
                "posterior_coordinate": ("beta_i = (s_i_phi + phase_c) mod 2pi"),
                "truth_transform_applied": bool(config["phase_marginalization"]),
            },
        },
        "parameter_treatment": {
            "sampled": list(sampled_parameters),
            "marginalized": list(_marginalized_parameters(config)),
        },
        "rank_method": {
            "weighting": "original nested-sampling weights",
            "comparison": "sample < truth",
            "resampled": False,
        },
        "posterior_samples": sample_count,
        "posterior_effective_sample_size": effective_sample_size,
        "posterior": {
            "path": "posterior.npz",
            "sha256": file_sha256(posterior_path),
            "bytes": posterior_path.stat().st_size,
            "fields": [
                *samples,
                "log_likelihood",
                "log_likelihood_birth",
                "log_weights",
            ],
            "space": "prior",
            "weighting": "normalized nested-sampling log weights",
        },
        "diagnostics": {
            "n_iterations": _safe_int(diagnostics.get("n_iterations")),
            "n_likelihood_evaluations": _safe_int(
                diagnostics.get("n_likelihood_evaluations")
            ),
            "log_Z": _safe_float(diagnostics.get("log_Z")),
            "log_Z_error": _safe_float(diagnostics.get("log_Z_error")),
            "insertion_index": insertion_diagnostic,
        },
        "timing_seconds": {
            "total": total_seconds,
            "data_injection": injection_seconds,
            "problem_setup": setup_seconds,
            "sample_call": sample_seconds,
            "result_extraction": extraction_seconds,
            "sample_phases": sample_phases,
            "paper_convention": paper_timing,
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": runtime.jaxlib.__version__,
            "blackjax": getattr(runtime.blackjax, "__version__", None),
            "jimgw": getattr(runtime.jimgw, "__version__", None),
            "ripplegw": _package_version("ripplegw"),
            "jax_compilation_cache_dir": str(runtime.cache_dir),
            "jax_constant_handling": runtime.constant_handling,
        },
        "devices": runtime.devices,
        "simulated_cpu": bool(args.simulate_cpu),
        "execution": {
            "mode": "long-lived-worker" if shared_runtime else "one-shot",
            "shared_runtime": shared_runtime,
            "runtime_initialization_seconds": runtime.initialization_seconds,
        },
        "implementation": implementation,
    }
    if cache_diagnostics is not None:
        summary["jax_cache_diagnostics"] = cache_diagnostics
    if timing_unavailable_reason is not None:
        summary["timing_seconds"]["paper_convention_unavailable_reason"] = (
            timing_unavailable_reason
        )
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
