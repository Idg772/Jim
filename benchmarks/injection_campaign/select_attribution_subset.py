"""Pick a timing-stratified event subset whose median maps to the population.

Sorting the 100 corrected-anchor events by paper-convention post-JIT time and
taking ranks 4, 14, ..., 94 (0-indexed decile representatives) gives a
10-event subset whose median estimates the population median. Every
attribution cell then runs this same subset, so all cross-cell factors are
paired and the absolute anchor carries a recorded subset->population mapping.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PAPER_FIGURE3 = {"min": 258.0, "median": 306.0, "max": 383.0}


def _event_times(campaign_dir: Path) -> list[tuple[int, float]]:
    rows = []
    for summary_path in sorted(campaign_dir.glob("results/injection-*/summary.json")):
        summary = json.loads(summary_path.read_text())
        seconds = summary["timing_seconds"]["paper_convention"][
            "post_jit_sampling_seconds"
        ]
        rows.append((int(summary["injection_id"]), float(seconds)))
    if len(rows) != 100:
        raise ValueError(f"expected 100 completed events, found {len(rows)}")
    return rows


def select_subset(campaign_dir: Path, n: int = 10) -> dict:
    rows = sorted(_event_times(campaign_dir), key=lambda item: item[1])
    stride = len(rows) // n
    picks = [rows[(stride - 1) // 2 + index * stride] for index in range(n)]
    times = [seconds for _, seconds in rows]
    subset_times = [seconds for _, seconds in picks]
    population_median = statistics.median(times)
    subset_median = statistics.median(subset_times)
    return {
        "injection_ids": sorted(injection_id for injection_id, _ in picks),
        "subset_median_seconds": subset_median,
        "population_median_seconds": population_median,
        "mapping_factor": population_median / subset_median,
        "shape_comparison": {
            "paper": PAPER_FIGURE3,
            "ours": {
                "min": min(times),
                "median": population_median,
                "max": max(times),
            },
            "spread_max_over_min": {
                "paper": PAPER_FIGURE3["max"] / PAPER_FIGURE3["min"],
                "ours": max(times) / min(times),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    result = select_subset(args.campaign_dir, n=args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["shape_comparison"], indent=2))


if __name__ == "__main__":
    main()
