"""Small, CPU-only helpers shared by the injection campaign commands."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2
CAMPAIGN_NAME = "paper-sharded-swig-fsm-pp"
PAPER_CATALOGUE_SIZE = 1000
PAPER_PP_RECOVERIES = 100
UNFOLDED_RANK_WEIGHTING = (
    "deterministically unfolded nested-sampling quadrature weights"
)
UNFOLDED_POSTERIOR_WEIGHTING = "normalized unfolded posterior log weights"
FOLDED_TARGET_SEMANTICS = "folded-target nested-sampling death and birth likelihoods"
POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS = (
    "quadrature-weight concentration diagnostic, not independent-draw ESS"
)
NETSKY_SCHEME = "netsky"

NETSKY_BLOCKS = [
    ["M_c", "q", "lambda_1", "lambda_2"],
    ["s1_mag", "s1_theta", "s1_phi"],
    ["s2_mag", "s2_theta", "s2_phi"],
    ["cos_zenith"],
    ["azimuth", "cos_iota", "psi", "log_d_hat"],
]
NETSKY_BRIDGE_BLOCKS = [["cos_zenith", "azimuth"]]

PARAMETERS = (
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
    "ra",
    "dec",
    "psi",
    "t_c",
)
MARGINALIZED_PARAMETERS = ("phase_c", "d_L")
CATALOGUE_FIELDS = (
    "injection_id",
    "noise_seed",
    "sampler_seed",
    *PARAMETERS,
    *MARGINALIZED_PARAMETERS,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "campaign": CAMPAIGN_NAME,
    "workload": "paper-bns-injection-15d",
    "paper_reference": "arXiv:2607.28265v1",
    "paper_configuration": "Sharded",
    "sampler_scheduler": "fsm",
    "detectors": ["H1", "L1", "V1"],
    "trigger_time_gps": 1187008882.0,
    "duration_seconds": 128.0,
    "segment_center_offset_seconds": -2.0,
    "segment_start_offset_seconds": -66.0,
    "sampling_frequency_hz": 4096.0,
    "f_min_hz": 20.0,
    "f_max_hz": 2048.0,
    "phase_marginalization": True,
    "time_marginalization": False,
    "distance_marginalization": {
        "n_grid_points": 10000,
        "prior": {
            "distribution": "power-law",
            "alpha": 2.0,
            "range_mpc": [30.0, 150.0],
        },
    },
    "waveform": "IMRPhenomPv2_NRTidalv2",
    "waveform_f_ref_hz": 20.0,
    "carrier_time_anchor": "imrphenomd",
    "n_devices": 4,
    "n_live": 512,
    "n_delete": 64,
    "n_delete_frac": 0.125,
    "num_inner_steps_per_dim": 1,
    "num_gibbs_sweeps": 1,
    "termination_log_z_live_minus_dead": -3.0,
    "termination_dlogz": 0.04858735157374206,
    "blocks": [
        ["M_c", "q", "lambda_1", "lambda_2"],
        ["s1_mag", "s1_theta", "s1_phi"],
        ["s2_mag", "s2_theta", "s2_phi"],
        ["iota"],
        ["zenith", "azimuth"],
        ["psi"],
        ["t_c"],
    ],
    "prior": {
        "M_c": {"distribution": "uniform", "range": [1.5, 2.5]},
        "q": {"distribution": "uniform", "range": [0.5, 1.0]},
        "spin_magnitudes": {"distribution": "uniform", "range": [0.0, 0.05]},
        "spin_directions": "isotropic",
        "iota": "isotropic",
        "lambda_1": {"distribution": "uniform", "range": [0.0, 5000.0]},
        "lambda_2": {"distribution": "uniform", "range": [0.0, 5000.0]},
        "d_L": {
            "distribution": "power-law",
            "alpha": 2.0,
            "range_mpc": [30.0, 150.0],
        },
        "sky": "isotropic",
        "psi": {"distribution": "uniform", "range_radians": [0.0, "pi"]},
        "t_c": {"distribution": "uniform", "range_seconds": [-0.1, 0.1]},
        "phase_c": {
            "distribution": "uniform",
            "range_radians": [0.0, "2pi"],
        },
    },
    "psd": {
        "family": "paper-design-sensitivity",
        "interpolation": "linear in PSD, matching Bilby",
        "H1": {"source": "aLIGO_O4_high_asd.txt", "source_quantity": "ASD"},
        "L1": {"source": "aLIGO_O4_high_asd.txt", "source_quantity": "ASD"},
        "V1": {"source": "AdV_psd.txt", "source_quantity": "PSD"},
    },
    "credible_levels": {
        "weighting": "original nested-sampling weights",
        "comparison": "strictly less than truth",
        "resampling": False,
    },
    "timing": {
        "paper_reference": "arXiv:2607.28265v1 Figure 3 and Table III",
        "measurement": "wall time around jim.sample",
        "excluded_one_off_phases": [
            "likelihood_jit",
            "sampler_kernel_jit",
        ],
        "post_jit_formula": "sample_call - likelihood_jit - sampler_kernel_jit",
        "selected_events": "all 100 main-text recoveries; no warm-up discarded",
    },
}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with a stable encoding."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_savez_compressed(path: Path, arrays: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError(f"NPZ output must end in .npz: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=".npz",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _power_law_sample(
    rng: np.random.Generator, low: float, high: float, alpha: float
) -> float:
    u = float(rng.random())
    exponent = alpha + 1.0
    if exponent == 0.0:
        return float(low * (high / low) ** u)
    return float(
        (u * (high**exponent - low**exponent) + low**exponent) ** (1 / exponent)
    )


def _isotropic_polar_angle(rng: np.random.Generator) -> float:
    return float(np.arccos(rng.uniform(-1.0, 1.0)))


def generate_catalogue(n_injections: int, master_seed: int) -> list[dict[str, Any]]:
    """Draw deterministic iid truths from the paper's Table I recovery prior."""

    if n_injections < 1:
        raise ValueError("n_injections must be positive")
    seed_sequences = np.random.SeedSequence(master_seed).spawn(n_injections)
    rows: list[dict[str, Any]] = []
    for injection_id, seed_sequence in enumerate(seed_sequences):
        truth_sequence, noise_sequence, sampler_sequence = seed_sequence.spawn(3)
        rng = np.random.default_rng(truth_sequence)
        row: dict[str, Any] = {
            "injection_id": injection_id,
            "noise_seed": int(noise_sequence.generate_state(1, dtype=np.uint32)[0]),
            "sampler_seed": int(sampler_sequence.generate_state(1, dtype=np.uint32)[0]),
            "M_c": float(rng.uniform(1.5, 2.5)),
            "q": float(rng.uniform(0.5, 1.0)),
            "s1_mag": float(rng.uniform(0.0, 0.05)),
            "s1_theta": _isotropic_polar_angle(rng),
            "s1_phi": float(rng.uniform(0.0, 2.0 * np.pi)),
            "s2_mag": float(rng.uniform(0.0, 0.05)),
            "s2_theta": _isotropic_polar_angle(rng),
            "s2_phi": float(rng.uniform(0.0, 2.0 * np.pi)),
            "iota": _isotropic_polar_angle(rng),
            "lambda_1": float(rng.uniform(0.0, 5000.0)),
            "lambda_2": float(rng.uniform(0.0, 5000.0)),
            "ra": float(rng.uniform(0.0, 2.0 * np.pi)),
            "dec": float(np.arcsin(rng.uniform(-1.0, 1.0))),
            "psi": float(rng.uniform(0.0, np.pi)),
            "t_c": float(rng.uniform(-0.1, 0.1)),
            "phase_c": float(rng.uniform(0.0, 2.0 * np.pi)),
            "d_L": _power_law_sample(rng, 30.0, 150.0, 2.0),
        }
        rows.append(row)
    return rows


def read_catalogue(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CATALOGUE_FIELDS):
            raise ValueError(
                f"catalogue fields do not match schema: {reader.fieldnames}"
            )
        rows = []
        for raw in reader:
            rows.append(
                {
                    "injection_id": int(raw["injection_id"]),
                    "noise_seed": int(raw["noise_seed"]),
                    "sampler_seed": int(raw["sampler_seed"]),
                    **{
                        name: float(raw[name])
                        for name in (*PARAMETERS, *MARGINALIZED_PARAMETERS)
                    },
                }
            )
    if [row["injection_id"] for row in rows] != list(range(len(rows))):
        raise ValueError("catalogue injection_id values must be contiguous from zero")
    return rows


def load_manifest(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.expanduser().resolve()
    path = campaign_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid campaign manifest {path}: {error}") from error
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported campaign schema in {path}")
    stored_hash = manifest.get("config_sha256")
    hash_input = dict(manifest)
    hash_input.pop("config_sha256", None)
    if stored_hash != canonical_sha256(hash_input):
        raise ValueError(f"campaign manifest hash mismatch: {path}")
    catalogue = campaign_dir / manifest["catalogue"]["path"]
    if file_sha256(catalogue) != manifest["catalogue"]["sha256"]:
        raise ValueError(f"campaign catalogue hash mismatch: {catalogue}")
    for relative, expected in manifest["psd"]["files"].items():
        psd_path = campaign_dir / relative
        if file_sha256(psd_path) != expected["sha256"]:
            raise ValueError(f"campaign PSD hash mismatch: {psd_path}")
    return manifest


def publication_eligible(manifest: Mapping[str, Any]) -> bool:
    """Return whether the manifest represents an iid publication population."""

    scope = manifest.get("reproduction_scope")
    return not (
        isinstance(scope, Mapping) and scope.get("pp_calibration_eligible") is False
    )


def require_publication_eligible(manifest: Mapping[str, Any], *, product: str) -> None:
    """Refuse publication-style products for hand-selected stress catalogues."""

    if not publication_eligible(manifest):
        raise ValueError(
            f"{product} is disabled for this targeted, non-iid stress catalogue"
        )


def result_dir(campaign_dir: Path, injection_id: int) -> Path:
    return campaign_dir / "results" / f"injection-{injection_id:03d}"


def _validate_completed_folded_diagnostics(
    directory: Path,
    summary: Mapping[str, Any],
) -> None:
    """Validate an explicitly declared folded-target diagnostic companion."""

    if "folded_nested_diagnostics" not in summary:
        return
    metadata = summary["folded_nested_diagnostics"]
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"completed result folded diagnostic metadata is invalid: {directory}"
        )
    if metadata.get("path") != "folded_nested_diagnostics.npz":
        raise ValueError(
            f"completed result has a different folded diagnostic path: {directory}"
        )
    artifact = directory / "folded_nested_diagnostics.npz"
    if not artifact.is_file():
        raise ValueError(
            f"completed result folded diagnostic artifact is missing: {directory}"
        )
    expected_sha256 = metadata.get("sha256")
    if not isinstance(expected_sha256, str) or file_sha256(artifact) != expected_sha256:
        raise ValueError(
            f"completed result folded diagnostic hash mismatch: {directory}"
        )
    expected_bytes = metadata.get("bytes")
    if (
        type(expected_bytes) is not int
        or expected_bytes < 0
        or artifact.stat().st_size != expected_bytes
    ):
        raise ValueError(
            f"completed result folded diagnostic byte count mismatch: {directory}"
        )


def validate_completed_result(
    directory: Path,
    config_sha256: str,
    *,
    injection_id: int | None = None,
    catalogue_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a completed summary after validating its resume identity.

    This deliberately validates the compact envelope needed to decide whether
    an expensive recovery may be skipped.  Full posterior semantics remain the
    responsibility of publication and staged-result validators.  The optional
    identity arguments preserve the legacy campaign-hash-only call shape while
    allowing campaign runners to bind a result to its frozen catalogue row.
    """

    if (injection_id is None) != (catalogue_row is None):
        raise ValueError(
            "injection_id and catalogue_row must either both be provided or omitted"
        )
    summary_path = directory / "summary.json"
    posterior_path = directory / "posterior.npz"
    if not summary_path.is_file() or not posterior_path.is_file():
        raise ValueError(f"completed result is missing files: {directory}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid completed result summary: {summary_path}") from error
    if not isinstance(summary, dict):
        raise TypeError(f"completed result summary must be an object: {summary_path}")
    if summary.get("config_sha256") != config_sha256:
        raise ValueError(f"completed result has a different campaign hash: {directory}")

    posterior = summary.get("posterior")
    if not isinstance(posterior, Mapping):
        raise TypeError(f"completed result has no posterior metadata: {directory}")
    stored_path = posterior.get("path")
    if stored_path is not None and stored_path != "posterior.npz":
        raise ValueError(
            f"completed result has a different posterior path: {directory}"
        )
    expected_sha256 = posterior.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or file_sha256(posterior_path) != expected_sha256
    ):
        raise ValueError(f"completed result posterior hash mismatch: {directory}")
    expected_bytes = posterior.get("bytes")
    if expected_bytes is not None and (
        type(expected_bytes) is not int
        or expected_bytes != posterior_path.stat().st_size
    ):
        raise ValueError(f"completed result posterior byte count mismatch: {directory}")

    _validate_completed_folded_diagnostics(directory, summary)

    if injection_id is None:
        return summary
    assert catalogue_row is not None
    if type(injection_id) is not int or injection_id < 0:
        raise ValueError("injection_id must be a non-negative exact integer")
    if catalogue_row.get("injection_id") != injection_id:
        raise ValueError("catalogue row does not match the requested injection ID")
    if type(summary.get("injection_id")) is not int or summary["injection_id"] != (
        injection_id
    ):
        raise ValueError(f"completed result has the wrong injection ID: {directory}")

    seeds = summary.get("seeds")
    if (
        not isinstance(seeds, Mapping)
        or type(seeds.get("noise")) is not int
        or type(seeds.get("sampler")) is not int
        or seeds.get("noise") != catalogue_row.get("noise_seed")
        or seeds.get("sampler") != catalogue_row.get("sampler_seed")
    ):
        raise ValueError(
            f"completed result seeds do not match the catalogue: {directory}"
        )
    truth = summary.get("truth")
    if not isinstance(truth, Mapping):
        raise TypeError(f"completed result has no truth metadata: {directory}")
    for name in (*PARAMETERS, *MARGINALIZED_PARAMETERS):
        raw_value = truth.get(name)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(
                f"completed result has invalid truth metadata for {name}: {directory}"
            )
        try:
            stored_value = float(raw_value)
            expected_value = float(catalogue_row[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"completed result has invalid truth metadata for {name}: {directory}"
            ) from error
        if stored_value != expected_value:
            raise ValueError(
                f"completed result truth does not match the catalogue for {name}: "
                f"{directory}"
            )
    return summary


def posterior_rank(
    samples: np.ndarray,
    truth: float,
    log_weights: np.ndarray,
) -> float:
    """Return the paper's directly weighted credible level at a scalar truth."""

    values = np.asarray(samples)
    logs = np.asarray(log_weights)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("posterior samples must be a non-empty one-dimensional array")
    if logs.shape != values.shape:
        raise ValueError("log_weights must have the same shape as posterior samples")
    if not np.all(np.isfinite(values)) or not np.isfinite(truth):
        raise ValueError("posterior samples and truth must be finite")
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("posterior log_weights must not contain NaN or +inf")
    finite = np.isfinite(logs)
    if not np.any(finite):
        raise ValueError("posterior log_weights must include a finite value")
    maximum = float(np.max(logs[finite]))
    weights = np.exp(logs - maximum)
    normalizer = float(np.sum(weights))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("posterior log_weights cannot be normalized")
    rank = float(np.sum(weights[values < truth]) / normalizer)
    # The numerator is a subset of the denominator, so the mathematical value
    # is in [0, 1]. Different reduction shapes can nevertheless overshoot an
    # endpoint by a few ulps (for example, 1.0000000000000002).
    return float(np.clip(rank, 0.0, 1.0))


STATUS_FIELDS = (
    "injection_id",
    "status",
    "attempts",
    "runtime_seconds",
    "posterior_samples",
    "summary",
    "posterior",
    "error",
)


def status_rows(campaign_dir: Path, n_injections: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for injection_id in range(n_injections):
        directory = result_dir(campaign_dir, injection_id)
        summary_path = directory / "summary.json"
        posterior_path = directory / "posterior.npz"
        failure_path = directory / "failure.json"
        failed_attempts = len(list(directory.glob("attempt-*.failed.log")))
        if summary_path.is_file() and posterior_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                expected_sha256 = summary.get("posterior", {}).get("sha256")
                if (
                    not isinstance(expected_sha256, str)
                    or file_sha256(posterior_path) != expected_sha256
                ):
                    status, runtime, count, error = (
                        "invalid",
                        "",
                        "",
                        "posterior hash mismatch",
                    )
                    attempts = failed_attempts + 1
                else:
                    status = "complete"
                    attempts = failed_attempts + 1
                    runtime = summary.get("timing_seconds", {}).get("total", "")
                    count = summary.get("posterior_samples", "")
                    error = ""
            except (OSError, json.JSONDecodeError):
                status, runtime, count, error = "invalid", "", "", "bad summary"
                attempts = failed_attempts
        elif failure_path.is_file():
            status, runtime, count = "failed", "", ""
            attempts = failed_attempts
            try:
                error = json.loads(failure_path.read_text(encoding="utf-8")).get(
                    "error", ""
                )
            except (OSError, json.JSONDecodeError):
                error = "bad failure record"
        elif (directory / "RUNNING").exists():
            status, runtime, count, error = "running", "", "", ""
            attempts = failed_attempts + 1
        else:
            status, runtime, count, error = "pending", "", "", ""
            attempts = failed_attempts
        rows.append(
            {
                "injection_id": injection_id,
                "status": status,
                "attempts": attempts,
                "runtime_seconds": runtime,
                "posterior_samples": count,
                "summary": (
                    str(summary_path.relative_to(campaign_dir))
                    if summary_path.is_file()
                    else ""
                ),
                "posterior": (
                    str(posterior_path.relative_to(campaign_dir))
                    if posterior_path.is_file()
                    else ""
                ),
                "error": error,
            }
        )
    return rows


def refresh_status(campaign_dir: Path, n_injections: int) -> list[dict[str, Any]]:
    rows = status_rows(campaign_dir, n_injections)
    atomic_write_csv(campaign_dir / "status.csv", rows, STATUS_FIELDS)
    return rows
