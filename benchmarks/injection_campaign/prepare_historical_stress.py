"""Prepare a paper-prior stress campaign from the historical mode-loss cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.injection_campaign.common import CATALOGUE_FIELDS, file_sha256
from benchmarks.injection_campaign.prepare_campaign import prepare_campaign

HISTORICAL_SOURCE_SHA256 = (
    "cc109cd6e6d733aa1a71f9007c4e3d95f72ae5da20630685aaacdd4c830204d8"
)
HISTORICAL_MASTER_SEED = 2026080801
HISTORICAL_IDS = (0, 82, 88, 75, 23, 58, 15, 81, 36, 96)
PROBLEM_IDS = (82, 88, 75, 23, 58, 15)
CONTROL_IDS = (0, 81, 36, 96)
LEGACY_CATALOGUE_FIELDS = (
    "injection_id",
    "noise_seed",
    "sampler_seed",
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
    "phase_c",
    "t_c",
)
DEFAULT_SOURCE_CATALOGUE = (
    Path(__file__).resolve().parents[2]
    / "campaign-results/fsm-swig-100-jitter-20260808/catalogue.csv"
)
SAMPLER_SCHEDULERS = {
    "fsm": "paper-sharded-swig-fsm-historical-stress",
    "pre-fsm-lockstep": "paper-sharded-swig-pre-fsm-lockstep-historical-stress",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--source-catalogue",
        type=Path,
        default=DEFAULT_SOURCE_CATALOGUE,
    )
    parser.add_argument("--noise-curves-dir", type=Path, default=None)
    parser.add_argument(
        "--sampler-scheduler",
        choices=tuple(SAMPLER_SCHEDULERS),
        default="fsm",
    )
    return parser.parse_args(argv)


def _uniform_quantile_map(
    value: float,
    *,
    source: tuple[float, float],
    target: tuple[float, float],
) -> float:
    source_low, source_high = source
    target_low, target_high = target
    quantile = (value - source_low) / (source_high - source_low)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"value {value} lies outside source range {source}")
    return float(target_low + quantile * (target_high - target_low))


def _power_law_quantile_map(
    value: float,
    *,
    source: tuple[float, float],
    target: tuple[float, float],
    alpha: float,
) -> float:
    exponent = alpha + 1.0
    source_low, source_high = source
    target_low, target_high = target
    quantile = (value**exponent - source_low**exponent) / (
        source_high**exponent - source_low**exponent
    )
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"value {value} lies outside source range {source}")
    return float(
        (
            target_low**exponent
            + quantile * (target_high**exponent - target_low**exponent)
        )
        ** (1.0 / exponent)
    )


def map_historical_row(
    source_row: dict[str, str],
    *,
    preflight_id: int,
) -> dict[str, Any]:
    """Preserve prior quantiles while moving a legacy truth into paper support."""

    row: dict[str, Any] = {
        "injection_id": preflight_id,
        "noise_seed": int(source_row["noise_seed"]),
        "sampler_seed": int(source_row["sampler_seed"]),
    }
    for name in CATALOGUE_FIELDS[3:]:
        row[name] = float(source_row[name])
    row["M_c"] = _uniform_quantile_map(
        float(source_row["M_c"]), source=(1.18, 1.21), target=(1.5, 2.5)
    )
    row["q"] = _uniform_quantile_map(
        float(source_row["q"]), source=(0.125, 1.0), target=(0.5, 1.0)
    )
    row["t_c"] = _uniform_quantile_map(
        float(source_row["t_c"]), source=(-0.03, 0.03), target=(-0.1, 0.1)
    )
    row["d_L"] = _power_law_quantile_map(
        float(source_row["d_L"]),
        source=(1.0, 75.0),
        target=(30.0, 150.0),
        alpha=2.0,
    )
    return row


def prepare_historical_stress(
    campaign_dir: Path,
    *,
    source_catalogue: Path,
    noise_curves_dir: Path | None,
    sampler_scheduler: str = "fsm",
) -> dict[str, object]:
    try:
        campaign_name = SAMPLER_SCHEDULERS[sampler_scheduler]
    except KeyError as error:
        raise ValueError(
            f"unsupported sampler scheduler: {sampler_scheduler!r}"
        ) from error
    source_catalogue = source_catalogue.expanduser().resolve()
    source_sha256 = file_sha256(source_catalogue)
    if source_sha256 != HISTORICAL_SOURCE_SHA256:
        raise ValueError(
            "historical catalogue SHA-256 mismatch: "
            f"{source_sha256} != {HISTORICAL_SOURCE_SHA256}"
        )
    with source_catalogue.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(LEGACY_CATALOGUE_FIELDS):
            raise ValueError(
                f"historical catalogue has the wrong fields: {reader.fieldnames}"
            )
        source_rows = list(reader)
    source_ids = [int(row["injection_id"]) for row in source_rows]
    if source_ids != list(range(100)):
        raise ValueError("historical catalogue IDs must be exactly 0 through 99")
    indexed = dict(zip(source_ids, source_rows, strict=True))
    missing = sorted(set(HISTORICAL_IDS) - indexed.keys())
    if missing:
        raise ValueError(f"historical catalogue is missing IDs: {missing}")
    rows = [
        map_historical_row(indexed[source_id], preflight_id=preflight_id)
        for preflight_id, source_id in enumerate(HISTORICAL_IDS)
    ]
    mapping = [
        {
            "preflight_id": preflight_id,
            "historical_id": source_id,
            "cohort": "problem" if source_id in PROBLEM_IDS else "control",
        }
        for preflight_id, source_id in enumerate(HISTORICAL_IDS)
    ]
    return prepare_campaign(
        campaign_dir,
        n_injections=len(rows),
        catalogue_size=len(rows),
        seed=HISTORICAL_MASTER_SEED,
        noise_curves_dir=noise_curves_dir,
        catalogue_rows=rows,
        campaign_name=campaign_name,
        config_overrides={"sampler_scheduler": sampler_scheduler},
        catalogue_provenance={
            "kind": "historical-prior-quantile-stress",
            "scientific_use": "targeted sampler regression; not a P-P calibration set",
            "source_catalogue_sha256": source_sha256,
            "source_master_seed": HISTORICAL_MASTER_SEED,
            "mapping": mapping,
            "quantile_mapping": {
                "M_c": {"source": [1.18, 1.21], "target": [1.5, 2.5]},
                "q": {"source": [0.125, 1.0], "target": [0.5, 1.0]},
                "t_c": {"source": [-0.03, 0.03], "target": [-0.1, 0.1]},
                "d_L": {
                    "source": [1.0, 75.0],
                    "target": [30.0, 150.0],
                    "distribution": "power-law alpha=2",
                },
            },
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = prepare_historical_stress(
        args.campaign_dir,
        source_catalogue=args.source_catalogue,
        noise_curves_dir=args.noise_curves_dir,
        sampler_scheduler=args.sampler_scheduler,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
