"""Create an immutable truth catalogue and exact design-PSD input bundle."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.injection_campaign.common import (
    CATALOGUE_FIELDS,
    DEFAULT_CONFIG,
    MARGINALIZED_PARAMETERS,
    PAPER_CATALOGUE_SIZE,
    PAPER_PP_RECOVERIES,
    PARAMETERS,
    SCHEMA_VERSION,
    atomic_savez_compressed,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    generate_catalogue,
    refresh_status,
)

_PAPER_FIELD_BOUNDS = {
    "M_c": (1.5, 2.5),
    "q": (0.5, 1.0),
    "s1_mag": (0.0, 0.05),
    "s1_theta": (0.0, np.pi),
    "s1_phi": (0.0, 2.0 * np.pi),
    "s2_mag": (0.0, 0.05),
    "s2_theta": (0.0, np.pi),
    "s2_phi": (0.0, 2.0 * np.pi),
    "iota": (0.0, np.pi),
    "lambda_1": (0.0, 5000.0),
    "lambda_2": (0.0, 5000.0),
    "ra": (0.0, 2.0 * np.pi),
    "dec": (-np.pi / 2.0, np.pi / 2.0),
    "psi": (0.0, np.pi),
    "t_c": (-0.1, 0.1),
    "phase_c": (0.0, 2.0 * np.pi),
    "d_L": (30.0, 150.0),
}


def _exact_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"explicit catalogue {field} must be an exact integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"explicit catalogue {field} must be an exact integer"
        ) from error
    if not np.isfinite(numeric) or numeric != parsed or parsed < 0:
        raise ValueError(f"explicit catalogue {field} must be an exact integer")
    return parsed


def _normalise_explicit_row(
    row: Mapping[str, Any], *, expected_id: int
) -> dict[str, Any]:
    if set(row) != set(CATALOGUE_FIELDS):
        raise ValueError(f"explicit catalogue row {expected_id} has the wrong fields")
    injection_id = _exact_nonnegative_int(row["injection_id"], field="injection_id")
    if injection_id != expected_id:
        raise ValueError(
            "explicit catalogue IDs must be contiguous and ordered from zero"
        )
    normalized: dict[str, Any] = {"injection_id": injection_id}
    for name in ("noise_seed", "sampler_seed"):
        seed = _exact_nonnegative_int(row[name], field=name)
        if seed >= 2**32:
            raise ValueError(f"explicit catalogue {name} must fit in uint32")
        normalized[name] = seed
    for name in (*PARAMETERS, *MARGINALIZED_PARAMETERS):
        try:
            value = float(row[name])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"explicit catalogue {name} must be finite and numeric"
            ) from error
        if not np.isfinite(value):
            raise ValueError(f"explicit catalogue {name} must be finite and numeric")
        low, high = _PAPER_FIELD_BOUNDS[name]
        if not low <= value <= high:
            raise ValueError(
                f"explicit catalogue {name}={value} lies outside [{low}, {high}]"
            )
        normalized[name] = value
    return normalized


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--n-injections",
        type=_positive_int,
        default=PAPER_PP_RECOVERIES,
        help="Number of leading catalogue entries to recover for the P-P test.",
    )
    parser.add_argument(
        "--catalogue-size",
        type=_positive_int,
        default=PAPER_CATALOGUE_SIZE,
        help="Number of iid Table-I truths to freeze; the paper uses 1000.",
    )
    parser.add_argument("--seed", type=int, default=260728265)
    parser.add_argument(
        "--noise-curves-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing Bilby's aLIGO_O4_high_asd.txt and AdV_psd.txt. "
            "By default they are located from the installed bilby package."
        ),
    )
    return parser.parse_args(argv)


def _bilby_noise_curves_dir() -> Path:
    try:
        import bilby.gw.detector
    except ImportError as error:
        raise SystemExit(
            "bilby is needed once to prepare the design PSD bundle. Install the "
            "cross-validation dependency group or pass --noise-curves-dir."
        ) from error
    return Path(bilby.gw.detector.__file__).resolve().parent / "noise_curves"


def _interpolate_design_psd(
    source: Path,
    frequencies: np.ndarray,
    *,
    source_quantity: str,
) -> np.ndarray:
    raw = np.loadtxt(source)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"design PSD must have two columns: {source}")
    source_frequencies, source_values = raw.T
    valid = (
        np.isfinite(source_frequencies)
        & np.isfinite(source_values)
        & (source_frequencies > 0)
        & (source_values > 0)
    )
    source_frequencies = source_frequencies[valid]
    source_values = source_values[valid]
    if source_frequencies.size < 2:
        raise ValueError(f"design PSD has insufficient positive data: {source}")
    if source_quantity == "ASD":
        source_values = source_values**2
    elif source_quantity != "PSD":
        raise ValueError(f"unknown spectral-density quantity: {source_quantity}")
    # Bilby's PowerSpectralDensity interpolates linearly in PSD. Values outside
    # the source grid are irrelevant to the [20, 2048] Hz likelihood band; use
    # endpoint values there so the frozen full rFFT array remains finite.
    values = np.interp(
        frequencies,
        source_frequencies,
        source_values,
        left=source_values[0],
        right=source_values[-1],
    )
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError(f"interpolated design PSD is invalid: {source}")
    return values


def prepare_campaign(
    campaign_dir: Path,
    *,
    n_injections: int,
    catalogue_size: int,
    seed: int,
    noise_curves_dir: Path | None,
    catalogue_rows: Sequence[Mapping[str, Any]] | None = None,
    catalogue_provenance: Mapping[str, Any] | None = None,
    campaign_name: str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest_path = campaign_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"campaign already exists at {campaign_dir}; use a new directory "
            "or run the resumable campaign command"
        )
    if n_injections < 1:
        raise ValueError("n_injections must be positive")
    if catalogue_size < 1:
        raise ValueError("catalogue_size must be positive")
    if n_injections > catalogue_size:
        raise ValueError("n_injections cannot exceed catalogue_size")
    explicit_catalogue = catalogue_rows is not None
    normalized_catalogue: list[dict[str, Any]] | None = None
    if catalogue_rows is not None:
        if len(catalogue_rows) != catalogue_size:
            raise ValueError(
                "explicit catalogue length must equal catalogue_size: "
                f"{len(catalogue_rows)} != {catalogue_size}"
            )
        normalized_catalogue = [
            _normalise_explicit_row(row, expected_id=expected_id)
            for expected_id, row in enumerate(catalogue_rows)
        ]
    campaign_dir.mkdir(parents=True, exist_ok=True)
    catalogue_path = campaign_dir / "catalogue.csv"
    catalogue = (
        generate_catalogue(catalogue_size, seed)
        if normalized_catalogue is None
        else normalized_catalogue
    )
    atomic_write_csv(catalogue_path, catalogue, CATALOGUE_FIELDS)

    curves = (
        noise_curves_dir.expanduser().resolve()
        if noise_curves_dir is not None
        else _bilby_noise_curves_dir()
    )
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_overrides is not None:
        unknown = sorted(set(config_overrides) - set(config))
        if unknown:
            raise ValueError(
                "config overrides contain unknown fields: " + ", ".join(unknown)
            )
        if "campaign" in config_overrides:
            raise ValueError("use campaign_name instead of overriding campaign")
        config.update(copy.deepcopy(dict(config_overrides)))
    if campaign_name is not None:
        if not campaign_name.strip():
            raise ValueError("campaign_name must not be empty")
        config["campaign"] = campaign_name
    if explicit_catalogue:
        config["timing"]["selected_events"] = (
            "targeted explicit stress catalogue; retained timings are diagnostic "
            "only and are not a Figure 3 population"
        )
    frequencies = np.fft.rfftfreq(
        round(config["duration_seconds"] * config["sampling_frequency_hz"]),
        d=1.0 / config["sampling_frequency_hz"],
    )
    psd_files: dict[str, dict[str, object]] = {}
    sources = {
        "inputs/psd/aLIGO-O4-high.npz": (
            curves / "aLIGO_O4_high_asd.txt",
            "ASD",
        ),
        "inputs/psd/AdV-design.npz": (curves / "AdV_psd.txt", "PSD"),
    }
    for relative, (source, source_quantity) in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"missing design PSD curve: {source}")
        destination = campaign_dir / relative
        atomic_savez_compressed(
            destination,
            {
                "frequencies": frequencies,
                "values": _interpolate_design_psd(
                    source,
                    frequencies,
                    source_quantity=source_quantity,
                ),
                "source_name": np.asarray(source.name),
                "source_sha256": np.asarray(file_sha256(source)),
                "source_quantity": np.asarray(source_quantity),
            },
        )
        psd_files[relative] = {
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
            "source_name": source.name,
            "source_sha256": file_sha256(source),
            "source_quantity": source_quantity,
        }

    catalogue_metadata: dict[str, object] = {
        "path": "catalogue.csv",
        "sha256": file_sha256(catalogue_path),
        "bytes": catalogue_path.stat().st_size,
        "generator": (
            "explicit ordered rows"
            if explicit_catalogue
            else "deterministic iid Table-I draws"
        ),
    }
    if catalogue_provenance is not None:
        catalogue_metadata["provenance"] = copy.deepcopy(dict(catalogue_provenance))
    reproduction_scope = (
        {
            "methodology": "arXiv:2607.28265v1 Sections III-V",
            "paper_catalogue_available": False,
            "paper_seeds_available": False,
            "iid_prior_predictive_catalogue": False,
            "pp_calibration_eligible": False,
            "statement": (
                "Uses the published Sharded recovery methodology with an "
                "explicit targeted catalogue. This selected stress set is not "
                "an iid prior-predictive sample and must not be used for a P-P "
                "calibration claim."
            ),
        }
        if explicit_catalogue
        else {
            "methodology": "arXiv:2607.28265v1 Sections III-V",
            "paper_catalogue_available": False,
            "paper_seeds_available": False,
            "iid_prior_predictive_catalogue": True,
            "pp_calibration_eligible": True,
            "statement": (
                "Matches the published Sharded methodology with a new "
                "deterministic iid catalogue; it cannot reproduce unpublished "
                "event-level ranks or p-values."
            ),
        }
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "master_seed": None if explicit_catalogue else seed,
        "n_injections": n_injections,
        "catalogue_size": catalogue_size,
        "selection": {
            "rule": (
                "explicit ordered stress catalogue"
                if explicit_catalogue
                else "first catalogue entries"
            ),
            "start_inclusive": 0,
            "stop_exclusive": n_injections,
        },
        "config": config,
        "catalogue": catalogue_metadata,
        "psd": {
            "detector_files": {
                "H1": "inputs/psd/aLIGO-O4-high.npz",
                "L1": "inputs/psd/aLIGO-O4-high.npz",
                "V1": "inputs/psd/AdV-design.npz",
            },
            "files": psd_files,
        },
        "reproduction_scope": reproduction_scope,
        "storage_policy": {
            "keep": [
                "manifest.json",
                "catalogue.csv",
                "status.csv",
                "inputs/psd/*.npz",
                "results/*/summary.json",
                "results/*/posterior.npz",
                "failed attempt logs",
                "pp/*.csv",
                "pp/*.json",
                "pp/*.png",
                "timing/*.csv",
                "timing/*.json",
                "timing/*.png",
            ],
            "discard": [
                "successful run logs",
                "JAX profiles",
                "HLO dumps",
                "per-slice traces",
                "uncompressed strain arrays",
            ],
        },
    }
    manifest["config_sha256"] = canonical_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    refresh_status(campaign_dir, n_injections)
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = prepare_campaign(
        args.campaign_dir,
        n_injections=args.n_injections,
        catalogue_size=args.catalogue_size,
        seed=args.seed,
        noise_curves_dir=args.noise_curves_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
