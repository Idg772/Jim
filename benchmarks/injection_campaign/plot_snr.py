"""Render a paper Figure 7-style network-SNR histogram for a campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from benchmarks.injection_campaign.common import (
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    load_manifest,
    require_publication_eligible,
    result_dir,
)
from benchmarks.injection_campaign.plot_pp import _selected_injection_ids

plt.switch_backend("Agg")


SNR_FIELDS = (
    "injection_id",
    "network_optimal_snr",
    "H1_optimal_snr",
    "L1_optimal_snr",
    "V1_optimal_snr",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Plot the available evaluated recoveries when the selected set is "
            "incomplete. The output is labelled partial-exploratory."
        ),
    )
    return parser.parse_args(argv)


def _positive_finite(value: object, *, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid {field} in {path}") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"non-positive or non-finite {field} in {path}")
    return result


def load_snr_rows(
    campaign_dir: Path,
    *,
    allow_partial: bool = False,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    """Load exact optimal network SNRs from the selected recovery summaries."""

    manifest = load_manifest(campaign_dir)
    require_publication_eligible(manifest, product="Figure 7 network-SNR summary")
    selected_ids = _selected_injection_ids(manifest)
    rows: list[dict[str, float | int]] = []
    missing_ids: list[int] = []
    for injection_id in selected_ids:
        path = result_dir(campaign_dir, injection_id) / "summary.json"
        if not path.is_file():
            missing_ids.append(injection_id)
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid result summary {path}: {error}") from error
        if summary.get("config_sha256") != manifest["config_sha256"]:
            raise ValueError(f"result belongs to a different campaign: {path}")
        if summary.get("injection_id") != injection_id:
            raise ValueError(f"result has the wrong injection ID: {path}")
        network = summary.get("network")
        if not isinstance(network, dict):
            raise TypeError(f"missing or invalid network diagnostics in {path}")
        by_detector = network.get("optimal_snr_by_detector")
        if not isinstance(by_detector, dict):
            raise TypeError(f"missing per-detector optimal SNRs in {path}")
        rows.append(
            {
                "injection_id": injection_id,
                "network_optimal_snr": _positive_finite(
                    network.get("optimal_snr"), field="network.optimal_snr", path=path
                ),
                **{
                    f"{detector}_optimal_snr": _positive_finite(
                        by_detector.get(detector),
                        field=f"network.optimal_snr_by_detector.{detector}",
                        path=path,
                    )
                    for detector in ("H1", "L1", "V1")
                },
            }
        )
    if missing_ids and not allow_partial:
        missing = ", ".join(str(injection_id) for injection_id in missing_ids)
        raise ValueError(
            "incomplete selected leading-ID set for Figure 7: "
            f"found {len(rows)}/{len(selected_ids)} SNRs; missing IDs: {missing}"
        )
    if not rows:
        raise ValueError("no evaluated network SNRs were found")
    return manifest, rows


def aggregate_and_plot_snr(
    campaign_dir: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Create the Figure 7-style histogram, audit CSV, and JSON report."""

    campaign_dir = campaign_dir.expanduser().resolve()
    manifest, rows = load_snr_rows(campaign_dir, allow_partial=allow_partial)
    output_dir = campaign_dir / "snr"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "network-snr.csv"
    atomic_write_csv(csv_path, rows, SNR_FIELDS)

    snr = np.asarray([row["network_optimal_snr"] for row in rows], dtype=float)
    median = float(np.median(snr))
    mean = float(np.mean(snr))
    minimum = float(np.min(snr))
    maximum = float(np.max(snr))
    selected_count = int(manifest["n_injections"])
    catalogue_size = int(manifest.get("catalogue_size", selected_count))
    is_complete = len(rows) == selected_count

    # Figure 7 spans 0--140 in 4-SNR bins. Preserve that frame unless this
    # campaign contains a louder event, in which case expand by whole 20s.
    upper = max(140.0, 20.0 * math.ceil(maximum / 20.0))
    bins = np.arange(0.0, upper + 4.0, 4.0)
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.6,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    ):
        figure, axis = plt.subplots(figsize=(3.4, 2.36))
        axis.hist(
            snr,
            bins=bins,
            histtype="stepfilled",
            color="#1f77b4",
            edgecolor="none",
            alpha=0.25,
        )
        counts, _ = np.histogram(snr, bins=bins)
        axis.hist(
            snr,
            bins=bins,
            histtype="step",
            color="#1f77b4",
            linewidth=1.1,
        )
        axis.axvline(
            median,
            color="black",
            linestyle=":",
            linewidth=1.0,
            label=f"Median = {median:.0f}",
        )
        axis.set(
            xlim=(0.0, upper),
            xlabel="Network SNR",
            ylabel="Count",
            xticks=np.arange(0.0, upper, 25.0),
        )
        peak = int(np.max(counts))
        tick_step = 5 if peak <= 25 else 10 if peak <= 60 else 25 if peak <= 120 else 50
        axis.set_yticks(np.arange(0, peak + 1, tick_step))
        axis.legend(loc="upper right", frameon=False, fontsize=8, handlelength=2.2)
        figure.tight_layout(pad=0.35)
        png_path = output_dir / "figure-7-equivalent.png"
        figure.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
        plt.close(figure)

    report_path = output_dir / "report.json"
    report = {
        "schema_version": 1,
        "paper_reference": "arXiv:2607.28265v1 Figure 7",
        "config_sha256": manifest["config_sha256"],
        "diagnostic_status": "complete" if is_complete else "partial-exploratory",
        "population": "selected recoveries with evaluated optimal network SNR",
        "evaluated_injections": len(rows),
        "selected_injections": selected_count,
        "frozen_catalogue_size": catalogue_size,
        "full_catalogue_evaluated": len(rows) == catalogue_size,
        "summary": {
            "median": median,
            "mean": mean,
            "minimum": minimum,
            "maximum": maximum,
        },
        "histogram": {
            "bin_width": 4.0,
            "lower_edge": 0.0,
            "upper_edge": upper,
        },
        "input_summary_sha256": {
            str(
                (
                    result_dir(campaign_dir, int(row["injection_id"])) / "summary.json"
                ).relative_to(campaign_dir)
            ): file_sha256(
                result_dir(campaign_dir, int(row["injection_id"])) / "summary.json"
            )
            for row in rows
        },
        "artifact_sha256": {
            str(path.relative_to(campaign_dir)): file_sha256(path)
            for path in (csv_path, png_path)
        },
        "outputs": [
            str(csv_path.relative_to(campaign_dir)),
            str(png_path.relative_to(campaign_dir)),
            str(report_path.relative_to(campaign_dir)),
        ],
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    aggregate_and_plot_snr(args.campaign_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
