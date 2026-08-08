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

SCHEMA_VERSION = 1
CAMPAIGN_NAME = "paper-15d-swig-pp"

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
    "d_L",
    "ra",
    "dec",
    "psi",
)
NUISANCE_PARAMETERS = ("phase_c", "t_c")
CATALOGUE_FIELDS = (
    "injection_id",
    "noise_seed",
    "sampler_seed",
    *PARAMETERS,
    *NUISANCE_PARAMETERS,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "campaign": CAMPAIGN_NAME,
    "workload": "paper-15d",
    "detectors": ["H1", "L1", "V1"],
    "trigger_time_gps": 1187008882.43,
    "duration_seconds": 128.0,
    "sampling_frequency_hz": 4096.0,
    "f_min_hz": 20.0,
    "f_max_hz": 2048.0 - 1.0 / 128.0,
    "phase_marginalization": True,
    "time_marginalization_tc_range_seconds": [-0.03, 0.03],
    "time_marginalization_upsample_factor": 32,
    "waveform": "IMRPhenomPv2_NRTidalv2",
    "waveform_f_ref_hz": 20.0,
    "n_devices": 4,
    "n_live": 512,
    "n_delete": 64,
    "n_delete_frac": 0.125,
    "num_inner_steps_per_dim": 1,
    "num_gibbs_sweeps": 1,
    "termination_dlogz": 0.0485873516,
    "blocks": [
        ["M_c", "q", "lambda_1", "lambda_2"],
        ["s1_mag", "s1_theta", "s1_phi"],
        ["s2_mag", "s2_theta", "s2_phi"],
        ["iota"],
        ["zenith", "azimuth"],
        ["psi"],
        ["d_L"],
    ],
    "prior": {
        "M_c": {"distribution": "uniform", "range": [1.18, 1.21]},
        "q": {"distribution": "uniform", "range": [0.125, 1.0]},
        "spin_magnitudes": {"distribution": "uniform", "range": [0.0, 0.05]},
        "spin_directions": "isotropic",
        "iota": "isotropic",
        "lambda_1": {"distribution": "uniform", "range": [0.0, 5000.0]},
        "lambda_2": {"distribution": "uniform", "range": [0.0, 5000.0]},
        "d_L": {
            "distribution": "power-law",
            "alpha": 2.0,
            "range_mpc": [1.0, 75.0],
        },
        "sky": "isotropic",
        "psi": {"distribution": "uniform", "range_radians": [0.0, "pi"]},
    },
    "psd": {
        "family": "design-sensitivity",
        "H1": "aLIGO_ZERO_DET_high_P_psd.txt",
        "L1": "aLIGO_ZERO_DET_high_P_psd.txt",
        "V1": "AdV_psd.txt",
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


_UNIFORM_PRIOR_RANGES: dict[str, tuple[float, float]] = {
    "M_c": (1.18, 1.21),
    "q": (0.125, 1.0),
    "s1_mag": (0.0, 0.05),
    "s2_mag": (0.0, 0.05),
    "s1_phi": (0.0, 2.0 * np.pi),
    "s2_phi": (0.0, 2.0 * np.pi),
    "lambda_1": (0.0, 5000.0),
    "lambda_2": (0.0, 5000.0),
    "ra": (0.0, 2.0 * np.pi),
    "psi": (0.0, np.pi),
    "phase_c": (0.0, 2.0 * np.pi),
    "t_c": (-0.03, 0.03),
}
_SINE_ANGLE_PARAMETERS = ("s1_theta", "s2_theta", "iota")
_D_L_RANGE = (1.0, 75.0)  # PowerLawPrior alpha=2: CDF proportional to d^3


def prior_cdf(name: str, value: float) -> float:
    """Analytic CDF of the campaign prior for one catalogue parameter."""

    if name in _UNIFORM_PRIOR_RANGES:
        low, high = _UNIFORM_PRIOR_RANGES[name]
        return (value - low) / (high - low)
    if name in _SINE_ANGLE_PARAMETERS:
        return 0.5 * (1.0 - np.cos(value))
    if name == "dec":
        return 0.5 * (1.0 + np.sin(value))
    if name == "d_L":
        low, high = _D_L_RANGE
        return (value**3 - low**3) / (high**3 - low**3)
    raise KeyError(f"no analytic prior CDF for parameter {name!r}")


def prior_ppf(name: str, u: float) -> float:
    """Inverse of :func:`prior_cdf`."""

    if name in _UNIFORM_PRIOR_RANGES:
        low, high = _UNIFORM_PRIOR_RANGES[name]
        return low + u * (high - low)
    if name in _SINE_ANGLE_PARAMETERS:
        return float(np.arccos(1.0 - 2.0 * u))
    if name == "dec":
        return float(np.arcsin(2.0 * u - 1.0))
    if name == "d_L":
        low, high = _D_L_RANGE
        return float((u * (high**3 - low**3) + low**3) ** (1.0 / 3.0))
    raise KeyError(f"no analytic prior PPF for parameter {name!r}")


def generate_catalogue(
    n_injections: int, master_seed: int, stratified: bool = True
) -> list[dict[str, Any]]:
    """Draw deterministic truths from exactly the paper-15d recovery prior.

    With ``stratified`` (the default) each parameter's quantiles form an
    independently shuffled Latin hypercube: every marginal hits each of the
    ``n_injections`` prior strata exactly once, so prior-dominated parameters
    cannot inherit truth-draw fluctuations into the P-P test. Truths remain
    individually prior-distributed; the P-P binomial bands become slightly
    conservative for prior-dominated parameters.
    """

    if n_injections < 1:
        raise ValueError("n_injections must be positive")
    root = np.random.SeedSequence(master_seed)
    per_injection = root.spawn(n_injections)
    truth_rng = np.random.default_rng(root.spawn(1)[0])
    names = (*PARAMETERS, *NUISANCE_PARAMETERS)
    if stratified:
        quantiles = np.column_stack(
            [
                (
                    truth_rng.permutation(n_injections)
                    + truth_rng.random(n_injections)
                )
                / n_injections
                for _ in names
            ]
        )
    else:
        quantiles = truth_rng.random((n_injections, len(names)))
    rows: list[dict[str, Any]] = []
    for injection_id, seed_sequence in enumerate(per_injection):
        _, noise_sequence, sampler_sequence = seed_sequence.spawn(3)
        row: dict[str, Any] = {
            "injection_id": injection_id,
            "noise_seed": int(noise_sequence.generate_state(1, dtype=np.uint32)[0]),
            "sampler_seed": int(
                sampler_sequence.generate_state(1, dtype=np.uint32)[0]
            ),
        }
        for column, name in enumerate(names):
            row[name] = prior_ppf(name, float(quantiles[injection_id, column]))
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
                        for name in (*PARAMETERS, *NUISANCE_PARAMETERS)
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


def result_dir(campaign_dir: Path, injection_id: int) -> Path:
    return campaign_dir / "results" / f"injection-{injection_id:03d}"


def posterior_rank(samples: np.ndarray, truth: float) -> float:
    """Return a tie-safe posterior CDF value for a scalar truth."""

    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("posterior samples must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or not np.isfinite(truth):
        raise ValueError("posterior samples and truth must be finite")
    less = np.count_nonzero(values < truth)
    equal = np.count_nonzero(values == truth)
    return float((less + 0.5 * equal) / values.size)


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
