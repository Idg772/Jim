"""Render the paper's Figure 3 post-JIT timing summary for Sharded runs."""

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
    load_manifest,
    require_publication_eligible,
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
PER_INJECTION_FIELDS = (
    "injection_id",
    "post_jit_sampling_seconds",
    "likelihood_jit_seconds",
    "sampler_jit_seconds",
    "sample_call_seconds",
    "end_to_end_seconds",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        help="Optional publication PDF path; PNG and CSV stay in CAMPAIGN/timing.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Generate an explicitly labelled exploratory timing plot before all "
            "selected recoveries finish. The paper-equivalent default is strict."
        ),
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


def load_timings(
    campaign_dir: Path,
    *,
    allow_partial: bool = False,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    """Load Figure 3 post-JIT timings and their auditable components."""

    manifest = load_manifest(campaign_dir)
    require_publication_eligible(manifest, product="Figure 3 timing summary")
    n_injections = int(manifest["n_injections"])
    rows: list[dict[str, float | int]] = []
    missing_ids: list[int] = []
    for injection_id in range(n_injections):
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
        timing = summary.get("timing_seconds")
        if not isinstance(timing, dict):
            raise TypeError(f"missing or invalid timing_seconds in {path}")
        paper = timing.get("paper_convention")
        if not isinstance(paper, dict):
            raise TypeError(
                f"missing Figure 3 post-JIT timing in {path}; rerun this injection"
            )
        sample_call = _finite_seconds(
            timing.get("sample_call"), field="sample_call", path=path
        )
        likelihood_jit = _finite_seconds(
            paper.get("likelihood_jit_seconds"),
            field="paper_convention.likelihood_jit_seconds",
            path=path,
        )
        sampler_jit = _finite_seconds(
            paper.get("sampler_jit_seconds"),
            field="paper_convention.sampler_jit_seconds",
            path=path,
        )
        post_jit = _finite_seconds(
            paper.get("post_jit_sampling_seconds"),
            field="paper_convention.post_jit_sampling_seconds",
            path=path,
        )
        expected = sample_call - likelihood_jit - sampler_jit
        tolerance = max(1e-9, abs(sample_call) * 1e-9)
        if not math.isclose(post_jit, expected, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"inconsistent Figure 3 timing arithmetic in {path}")
        rows.append(
            {
                "injection_id": injection_id,
                "post_jit_sampling_seconds": post_jit,
                "likelihood_jit_seconds": likelihood_jit,
                "sampler_jit_seconds": sampler_jit,
                "sample_call_seconds": sample_call,
                "end_to_end_seconds": _finite_seconds(
                    timing.get("total"), field="total", path=path
                ),
            }
        )
    if missing_ids and not allow_partial:
        missing = ", ".join(str(injection_id) for injection_id in missing_ids)
        raise ValueError(
            "incomplete selected leading-ID set for Figure 3: "
            f"found {len(rows)}/{n_injections} timings; missing IDs: {missing}"
        )
    if not rows:
        raise ValueError("no completed injection timing summaries were found")
    return manifest, rows


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
            f"{float(summary['median_seconds']):.1f}",
            f"{float(summary['min_seconds']):.1f}",
            f"{float(summary['max_seconds']):.1f}",
            (f"{int(summary['below_segment_count'])}/{int(summary['n_injections'])}"),
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
            "Figure 3 convention: wall time inside jim.sample minus the "
            "measured likelihood and sampler-kernel JIT phases.\n"
            "Raw sample-call, JIT, and end-to-end values are retained in "
            "per-injection-timing.csv."
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="0.25",
    )


def aggregate_and_plot_timing(
    campaign_dir: Path,
    *,
    pdf_output: Path | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Create a Figure 3 post-JIT histogram, audit CSV, and optional PDF."""

    campaign_dir = campaign_dir.expanduser().resolve()
    manifest, rows = load_timings(campaign_dir, allow_partial=allow_partial)
    output_dir = campaign_dir / "timing"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_injection_path = output_dir / "per-injection-timing.csv"
    atomic_write_csv(per_injection_path, rows, PER_INJECTION_FIELDS)

    post_jit = np.asarray(
        [row["post_jit_sampling_seconds"] for row in rows], dtype=float
    )
    timing_definition = (
        "jim.sample wall time minus measured one-off likelihood and "
        "sampler-kernel JIT compilation"
    )
    summaries = [_summarize("Sharded", post_jit, timing_definition=timing_definition)]
    csv_path = output_dir / "figure-3-summary.csv"
    atomic_write_csv(csv_path, summaries, SUMMARY_FIELDS)

    data_min = float(np.min(post_jit))
    data_max = float(np.max(post_jit))
    data_span = max(data_max - data_min, 1.0)
    histogram_lower = max(0.0, data_min - 0.04 * data_span)
    histogram_upper = data_max + 0.04 * data_span
    bins = np.linspace(histogram_lower, histogram_upper, 21)

    plotted_min = min(histogram_lower, SEGMENT_SECONDS)
    plotted_max = max(histogram_upper, SEGMENT_SECONDS)
    plotted_span = max(plotted_max - plotted_min, 1.0)
    lower = max(0.0, plotted_min - 0.05 * plotted_span)
    upper = plotted_max + 0.05 * plotted_span
    is_complete = len(rows) == int(manifest["n_injections"])
    result_kind = "complete" if is_complete else "partial exploratory"
    configuration = str(
        manifest.get("config", {}).get("paper_configuration", "Sharded")
    )

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
        color = "#2678b2"
        axis.hist(
            post_jit,
            bins=bins,
            histtype="stepfilled",
            alpha=0.30,
            color=color,
            edgecolor=color,
            linewidth=1.6,
            label=configuration,
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
            xlabel="Post-JIT sampling time [s]",
            ylabel="Count",
        )
        axis.grid(axis="y", alpha=0.18)
        figure.suptitle(
            f"{configuration} FSM/SwiG Figure 3 timing "
            f"({result_kind}, N = {len(rows)})",
            y=0.965,
            fontsize=13,
        )
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=2,
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
                    "Subject": (
                        "Figure 3 post-JIT timing distribution and summary table"
                    ),
                },
            )
        plt.close(figure)

    def _distribution(field: str) -> dict[str, float]:
        values = np.asarray([row[field] for row in rows], dtype=float)
        return {
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    report_path = output_dir / "report.json"
    report = {
        "schema_version": 1,
        "paper_reference": "arXiv:2607.28265v1 Figure 3 and Table III",
        "paper_configuration": configuration,
        "config_sha256": manifest["config_sha256"],
        "diagnostic_status": "complete" if is_complete else "partial-exploratory",
        "completed_injections": len(rows),
        "requested_injections": manifest["n_injections"],
        "timing_definition": timing_definition,
        "x_scale": "linear",
        "histogram_bin_edges_seconds": [float(edge) for edge in bins],
        "figure_3_summary": summaries[0],
        "audit_distributions_seconds": {
            field: _distribution(field)
            for field in (
                "likelihood_jit_seconds",
                "sampler_jit_seconds",
                "sample_call_seconds",
                "end_to_end_seconds",
            )
        },
        "outputs": [
            str(per_injection_path.relative_to(campaign_dir)),
            str(csv_path.relative_to(campaign_dir)),
            str(png_path.relative_to(campaign_dir)),
            str(report_path.relative_to(campaign_dir)),
        ],
        "pdf_output": str(pdf_output) if pdf_output is not None else None,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    aggregate_and_plot_timing(
        args.campaign_dir,
        pdf_output=args.pdf_output,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
