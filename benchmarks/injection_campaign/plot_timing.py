"""Render a Figure 3-style timing summary for an injection campaign."""

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
    load_manifest,
    result_dir,
)

plt.switch_backend("Agg")


SEGMENT_SECONDS = 128.0
SUMMARY_FIELDS = (
    "measurement",
    "n_injections",
    "median_seconds",
    "min_seconds",
    "max_seconds",
    "below_segment_count",
    "below_segment_fraction",
    "segment_seconds",
    "timing_definition",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        help="Optional publication PDF path; PNG and CSV stay in CAMPAIGN/timing.",
    )
    return parser.parse_args(argv)


def _finite_seconds(value: object, *, field: str, path: Path) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid {field} in {path}") from error
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"non-positive or non-finite {field} in {path}")
    return seconds


def load_timings(campaign_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load validated sampler-call and end-to-end times from completed runs."""

    manifest = load_manifest(campaign_dir)
    sample_call: list[float] = []
    end_to_end: list[float] = []
    for injection_id in range(int(manifest["n_injections"])):
        path = result_dir(campaign_dir, injection_id) / "summary.json"
        if not path.is_file():
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid result summary {path}: {error}") from error
        if summary.get("config_sha256") != manifest["config_sha256"]:
            raise ValueError(f"result belongs to a different campaign: {path}")
        timing = summary.get("timing_seconds", {})
        sample_call.append(
            _finite_seconds(timing.get("sample_call"), field="sample_call", path=path)
        )
        end_to_end.append(
            _finite_seconds(timing.get("total"), field="total", path=path)
        )
    if not sample_call:
        raise ValueError("no completed injection timing summaries were found")
    return manifest, {
        "Sampler call*": np.asarray(sample_call, dtype=float),
        "End-to-end": np.asarray(end_to_end, dtype=float),
    }


def _summarize(
    label: str, values: np.ndarray, *, timing_definition: str
) -> dict[str, object]:
    return {
        "measurement": label.rstrip("*"),
        "n_injections": int(values.size),
        "median_seconds": float(np.median(values)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
        "below_segment_count": int(np.count_nonzero(values < SEGMENT_SECONDS)),
        "below_segment_fraction": float(np.mean(values < SEGMENT_SECONDS)),
        "segment_seconds": SEGMENT_SECONDS,
        "timing_definition": timing_definition,
    }


def _draw_table(ax: Any, summaries: list[dict[str, object]]) -> None:
    ax.axis("off")
    columns = (
        "Measurement",
        "Median [s]",
        "Min [s]",
        "Max [s]",
        "<128 s",
    )
    cells = [
        [
            summary["measurement"],
            f'{float(summary["median_seconds"]):.1f}',
            f'{float(summary["min_seconds"]):.1f}',
            f'{float(summary["max_seconds"]):.1f}',
            (
                f'{int(summary["below_segment_count"])}/'
                f'{int(summary["n_injections"])}'
            ),
        ]
        for summary in summaries
    ]
    table = ax.table(
        cellText=cells,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        colWidths=(0.28, 0.19, 0.16, 0.16, 0.21),
        bbox=(0.0, 0.36, 1.0, 0.60),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("0.30")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.visible_edges = "TB"
            cell.get_text().set_weight("bold")
        elif row == len(cells):
            cell.visible_edges = "B"
        else:
            cell.visible_edges = ""
        if column == 0:
            cell.get_text().set_ha("left")
    ax.text(
        0.0,
        0.0,
        (
            "* Recorded wall time inside jim.sample; the campaign did not "
            "separately instrument one-off JIT phases.\n"
            "  End-to-end includes imports, data injection, setup, sampling, "
            "and result extraction."
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="0.25",
    )


def aggregate_and_plot_timing(
    campaign_dir: Path, *, pdf_output: Path | None = None
) -> dict[str, Any]:
    """Create the timing histogram, compact CSV, and optional PDF."""

    campaign_dir = campaign_dir.expanduser().resolve()
    manifest, timings = load_timings(campaign_dir)
    output_dir = campaign_dir / "timing"
    output_dir.mkdir(parents=True, exist_ok=True)

    definitions = {
        "Sampler call*": "wall time measured directly around jim.sample",
        "End-to-end": "run_injection process wall time",
    }
    summaries = [
        _summarize(label, values, timing_definition=definitions[label])
        for label, values in timings.items()
    ]
    csv_path = output_dir / "figure-3-summary.csv"
    atomic_write_csv(csv_path, summaries, SUMMARY_FIELDS)

    all_values = np.concatenate(tuple(timings.values()))
    lower = 5.0 * math.floor((float(np.min(all_values)) - 5.0) / 5.0)
    upper = 5.0 * math.ceil((float(np.max(all_values)) + 5.0) / 5.0)
    bins = np.linspace(lower, upper, 21)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
        }
    ):
        figure = plt.figure(figsize=(7.4, 6.2))
        grid = figure.add_gridspec(
            2,
            1,
            height_ratios=(3.35, 1.45),
            left=0.10,
            right=0.98,
            bottom=0.08,
            top=0.80,
            hspace=0.35,
        )
        axis = figure.add_subplot(grid[0])
        table_axis = figure.add_subplot(grid[1])
        colors = ("#2678b2", "#f28e2b")
        for (label, values), color in zip(timings.items(), colors, strict=True):
            axis.hist(
                values,
                bins=bins,
                histtype="stepfilled",
                alpha=0.24,
                color=color,
                edgecolor=color,
                linewidth=1.6,
                label=label,
            )
        axis.axvline(
            SEGMENT_SECONDS,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="Segment = 128 s",
        )
        axis.set(
            xlim=(lower, upper),
            xlabel="Wall time [s]",
            ylabel="Count",
        )
        axis.grid(axis="y", alpha=0.18)
        figure.suptitle(
            "Per-injection timing - FSM/SwiG, 4 x NVIDIA H200 "
            f"(N = {len(next(iter(timings.values())))})",
            y=0.965,
            fontsize=13,
        )
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=3,
            frameon=False,
            fontsize=9.2,
        )
        _draw_table(table_axis, summaries)

        png_path = output_dir / "figure-3-equivalent.png"
        figure.savefig(png_path, dpi=220, bbox_inches="tight")
        if pdf_output is not None:
            pdf_output = pdf_output.expanduser().resolve()
            pdf_output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                pdf_output,
                bbox_inches="tight",
                metadata={
                    "Title": "FSM/SwiG injection-campaign timing",
                    "Subject": "Figure 3-style timing distribution and summary table",
                },
            )
        plt.close(figure)

    report = {
        "config_sha256": manifest["config_sha256"],
        "completed_injections": len(next(iter(timings.values()))),
        "requested_injections": manifest["n_injections"],
        "outputs": [
            str(csv_path.relative_to(campaign_dir)),
            str(png_path.relative_to(campaign_dir)),
        ],
        "pdf_output": str(pdf_output) if pdf_output is not None else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    aggregate_and_plot_timing(args.campaign_dir, pdf_output=args.pdf_output)


if __name__ == "__main__":
    main()
