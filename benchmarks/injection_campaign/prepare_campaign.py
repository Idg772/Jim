"""Create an immutable truth catalogue and exact design-PSD input bundle."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchmarks.injection_campaign.common import (
    CATALOGUE_FIELDS,
    DEFAULT_CONFIG,
    SCHEMA_VERSION,
    atomic_savez_compressed,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    generate_catalogue,
    refresh_status,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--n-injections", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=260728265)
    parser.add_argument(
        "--noise-curves-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing Bilby's aLIGO_ZERO_DET_high_P_psd.txt and "
            "AdV_psd.txt. By default they are located from the installed bilby package."
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


def _interpolate_design_psd(source: Path, frequencies: np.ndarray) -> np.ndarray:
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
    values = np.exp(
        np.interp(
            frequencies,
            source_frequencies,
            np.log(source_values),
            left=np.log(source_values[0]),
            right=np.log(source_values[-1]),
        )
    )
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError(f"interpolated design PSD is invalid: {source}")
    return values


def prepare_campaign(
    campaign_dir: Path,
    *,
    n_injections: int,
    seed: int,
    noise_curves_dir: Path | None,
) -> dict[str, object]:
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest_path = campaign_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"campaign already exists at {campaign_dir}; use a new directory "
            "or run the resumable campaign command"
        )
    campaign_dir.mkdir(parents=True, exist_ok=True)
    catalogue_path = campaign_dir / "catalogue.csv"
    catalogue = generate_catalogue(n_injections, seed)
    atomic_write_csv(catalogue_path, catalogue, CATALOGUE_FIELDS)

    curves = (
        noise_curves_dir.expanduser().resolve()
        if noise_curves_dir is not None
        else _bilby_noise_curves_dir()
    )
    config = copy.deepcopy(DEFAULT_CONFIG)
    frequencies = np.fft.rfftfreq(
        round(config["duration_seconds"] * config["sampling_frequency_hz"]),
        d=1.0 / config["sampling_frequency_hz"],
    )
    psd_files: dict[str, dict[str, object]] = {}
    sources = {
        "inputs/psd/aLIGO-design.npz": curves
        / "aLIGO_ZERO_DET_high_P_psd.txt",
        "inputs/psd/AdV-design.npz": curves / "AdV_psd.txt",
    }
    for relative, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"missing design PSD curve: {source}")
        destination = campaign_dir / relative
        atomic_savez_compressed(
            destination,
            {
                "frequencies": frequencies,
                "values": _interpolate_design_psd(source, frequencies),
                "source_name": np.asarray(source.name),
                "source_sha256": np.asarray(file_sha256(source)),
            },
        )
        psd_files[relative] = {
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
            "source_name": source.name,
            "source_sha256": file_sha256(source),
        }

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "master_seed": seed,
        "n_injections": n_injections,
        "config": config,
        "catalogue": {
            "path": "catalogue.csv",
            "sha256": file_sha256(catalogue_path),
            "bytes": catalogue_path.stat().st_size,
        },
        "psd": {
            "detector_files": {
                "H1": "inputs/psd/aLIGO-design.npz",
                "L1": "inputs/psd/aLIGO-design.npz",
                "V1": "inputs/psd/AdV-design.npz",
            },
            "files": psd_files,
        },
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
                "pp/*.png",
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
        seed=args.seed,
        noise_curves_dir=args.noise_curves_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
