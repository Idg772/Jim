"""Analyze the paired P-P residual-attribution cell chain.

Every reported speed factor is calculated from event-matched ratios.  This
keeps event difficulty out of the comparison and leaves pod hardware as a
shared nuisance factor within the chain.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

CELL_NAMES = (
    "00-legacy",
    "01-stepping-cache",
    "02-replicated-topology",
    "03-fsm-scheduler",
    "04-shared-frequency-grid",
    "05-real-angle-phasor",
    "06-real-inner-product",
    "07-cholesky-factor",
    "08-production",
)
METRICS = {
    "paper_convention": (
        "timing_seconds",
        "paper_convention",
        "post_jit_sampling_seconds",
    ),
    "ns_loop": ("timing_seconds", "sample_phases", "ns_loop"),
}
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_RESAMPLES = 1000
OBSERVED_ENDPOINT_RATIO = 6.851098
CARRIER_ANCHOR_FACTOR = 1.054


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _factor_statistics(ratios: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in ratios]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("paired timing ratios must be finite and positive")
    geometric_mean = statistics.geometric_mean(values)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(
        statistics.geometric_mean(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "geomean": geometric_mean,
        "bootstrap_95_percent_interval": [
            _percentile(estimates, 0.025),
            _percentile(estimates, 0.975),
        ],
        "paired_ratios": values,
        "n_events": len(values),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def _read_metric(summary_path: Path, path: Sequence[str]) -> float:
    try:
        value: Any = json.loads(summary_path.read_text(encoding="utf-8"))
        for part in path:
            value = value[part]
        result = float(value)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid timing metric in {summary_path}") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"timing metric must be finite and positive: {summary_path}")
    return result


def _summary_paths(cell_dir: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for summary_path in sorted(cell_dir.glob("results/injection-*/summary.json")):
        try:
            injection_id = int(summary_path.parent.name.removeprefix("injection-"))
        except ValueError as error:
            raise ValueError(
                f"invalid result directory name: {summary_path.parent}"
            ) from error
        if injection_id in paths:
            raise ValueError(f"duplicate injection {injection_id} in {cell_dir}")
        paths[injection_id] = summary_path
    if not paths:
        raise ValueError(f"no completed summaries found in {cell_dir}")
    return paths


def _metric_analysis(
    paths_by_cell: dict[str, dict[int, Path]],
    event_ids: Sequence[int],
    metric_path: Sequence[str],
) -> dict[str, Any]:
    timings = {
        cell: {
            event_id: _read_metric(paths[event_id], metric_path)
            for event_id in event_ids
        }
        for cell, paths in paths_by_cell.items()
    }
    chain_factors: dict[str, dict[str, Any]] = {}
    for previous, current in pairwise(CELL_NAMES):
        chain_factors[current] = _factor_statistics(
            [
                timings[previous][event_id] / timings[current][event_id]
                for event_id in event_ids
            ]
        )

    full_chain = _factor_statistics(
        [
            timings[CELL_NAMES[0]][event_id] / timings[CELL_NAMES[-1]][event_id]
            for event_id in event_ids
        ]
    )
    fidelity = dict(chain_factors["08-production"])
    fidelity["within_five_percent"] = abs(fidelity["geomean"] - 1.0) <= 0.05
    fidelity["caveat"] = (
        None
        if fidelity["within_five_percent"]
        else (
            "Cell 07 does not reproduce production within 5%; quote all chain "
            "factors with this emulation-fidelity caveat."
        )
    )
    low, high = full_chain["bootstrap_95_percent_interval"]
    external = OBSERVED_ENDPOINT_RATIO / (full_chain["geomean"] * CARRIER_ANCHOR_FACTOR)
    return {
        "metric_path": ".".join(metric_path),
        "chain_factors": chain_factors,
        "F_impl": full_chain,
        "emulation_fidelity": fidelity,
        "external_residual": {
            "geomean": external,
            "bootstrap_95_percent_interval": [
                OBSERVED_ENDPOINT_RATIO / (high * CARRIER_ANCHOR_FACTOR),
                OBSERVED_ENDPOINT_RATIO / (low * CARRIER_ANCHOR_FACTOR),
            ],
            "formula": "observed_endpoint_ratio / (F_impl * carrier_anchor_factor)",
        },
    }


def analyze(root: Path) -> dict[str, Any]:
    """Return paired chain factors for all completed events under ``root``."""

    root = root.expanduser().resolve()
    paths_by_cell = {cell: _summary_paths(root / cell) for cell in CELL_NAMES}
    event_ids = sorted(paths_by_cell[CELL_NAMES[0]])
    for cell, paths in paths_by_cell.items():
        if sorted(paths) != event_ids:
            raise ValueError(f"cell {cell} does not contain the same event IDs")

    try:
        subset = json.loads((root / "subset.json").read_text(encoding="utf-8"))
        mapping_factor = float(subset["mapping_factor"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid subset metadata in {root / 'subset.json'}"
        ) from error
    if not math.isfinite(mapping_factor) or mapping_factor <= 0.0:
        raise ValueError("subset mapping factor must be finite and positive")

    result: dict[str, Any] = {
        "schema_version": 1,
        "event_ids": event_ids,
        "n_events": len(event_ids),
        "mapping_factor": mapping_factor,
        "observed_endpoint_ratio": OBSERVED_ENDPOINT_RATIO,
        "carrier_anchor_factor": CARRIER_ANCHOR_FACTOR,
        "anchor_note": (
            "All cells share the current carrier anchor, so its 1.054x factor is "
            "multiplied back into the paper-to-current residual split."
        ),
    }
    for metric_name, metric_path in METRICS.items():
        result[metric_name] = _metric_analysis(paths_by_cell, event_ids, metric_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
