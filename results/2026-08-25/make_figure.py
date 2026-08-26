"""Build the timing table and P-P + timing figure for the U=8 N=100 campaign."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

PARAMETER_ORDER = (
    "M_c",
    "q",
    "s1_mag",
    "s1_theta",
    "s1_phi",
    "s2_mag",
    "s2_theta",
    "s2_phi",
    "iota",
    "psi",
    "ra",
    "dec",
    "lambda_1",
    "lambda_2",
    "d_L",
)
PARAMETER_LABELS = {
    "M_c": r"$\mathcal{M}_c$",
    "q": r"$q$",
    "s1_mag": r"$|\mathbf{s}_1|$",
    "s1_theta": r"$\theta_1$",
    "s1_phi": r"$\phi_1$",
    "s2_mag": r"$|\mathbf{s}_2|$",
    "s2_theta": r"$\theta_2$",
    "s2_phi": r"$\phi_2$",
    "iota": r"$\iota$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
    "d_L": r"$d_L$",
}


def load_ranks() -> dict[str, np.ndarray]:
    with open(HERE / "pp" / "ranks.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 rank rows, found {len(rows)}")
    return {
        name: np.asarray([float(row[name]) for row in rows]) for name in PARAMETER_ORDER
    }


def load_report() -> tuple[dict[str, float], float]:
    report = json.loads((HERE / "pp" / "report.json").read_text())
    ks = {
        entry["parameter"]: float(entry["ks_pvalue"])
        for entry in report["per_parameter"]
    }
    if set(ks) != set(PARAMETER_ORDER):
        raise RuntimeError("report parameters do not match the campaign parameters")
    return ks, float(report["combined_test"]["pvalue"])


def extract_timing() -> list[dict[str, float | int | str]]:
    index = json.loads((HERE / "index.json").read_text())
    entries = index["entries"]
    if len(entries) != 100:
        raise RuntimeError(f"expected 100 index entries, found {len(entries)}")
    rows: list[dict[str, float | int | str]] = []
    for entry in entries:
        summary_path = PROJECT / entry["summary"]
        summary = json.loads(summary_path.read_text())
        if summary["injection_id"] != entry["local_injection_id"]:
            raise RuntimeError(f"injection id mismatch for {summary_path}")
        timing = summary["timing_seconds"]
        rows.append(
            {
                "source_injection_id": entry["source_injection_id"],
                "batch": entry["batch"],
                "post_jit_sampling_seconds": timing["paper_convention"][
                    "post_jit_sampling_seconds"
                ],
                "sample_call_seconds": timing["sample_call"],
                "total_seconds": timing["total"],
            }
        )
    source_ids = [row["source_injection_id"] for row in rows]
    if sorted(source_ids) != list(range(100)):
        raise RuntimeError("source injection ids are not exactly 0..99")
    return rows


def write_timing_csv(rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "source_injection_id",
        "batch",
        "post_jit_sampling_seconds",
        "sample_call_seconds",
        "total_seconds",
    ]
    with open(HERE / "timing.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _confidence_band(
    n: int, x: np.ndarray, sigma: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact central Gaussian-sigma pointwise binomial interval for the ECDF."""
    tail = 0.5 * math.erfc(sigma / math.sqrt(2.0))
    return binom.ppf(tail, n, x) / n, binom.ppf(1.0 - tail, n, x) / n


def _draw_expected(ax: plt.Axes, n: int) -> None:
    x = np.linspace(0.0, 1.0, 301)
    for sigma, color in ((3, "0.93"), (2, "0.84"), (1, "0.73")):
        lower, upper = _confidence_band(n, x, sigma)
        ax.fill_between(x, lower, upper, color=color, label="_nolegend_", zorder=0)
    ax.plot(x, x, color="black", linewidth=0.9, linestyle="-", label="_nolegend_")


def _ecdf(ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(ranks)
    x = np.concatenate(([0.0], ordered, [1.0]))
    y = np.concatenate(([0.0], np.arange(1, ranks.size + 1) / ranks.size, [1.0]))
    return x, y


def draw_figure(
    rank_arrays: dict[str, np.ndarray],
    ks_pvalues: dict[str, float],
    fisher_pvalue: float,
    post_jit_seconds: np.ndarray,
) -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    config = manifest["config"]
    n = rank_arrays[PARAMETER_ORDER[0]].size

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    ):
        figure = plt.figure(figsize=(9.6, 3.9))
        pp_axis = figure.add_axes((0.07, 0.16, 0.335, 0.79))
        hist_axis = figure.add_axes((0.635, 0.16, 0.335, 0.79))

        _draw_expected(pp_axis, n)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for index, name in enumerate(PARAMETER_ORDER):
            x, y = _ecdf(rank_arrays[name])
            pp_axis.step(
                x,
                y,
                where="post",
                color=colors[index % len(colors)],
                linestyle="-" if index < 10 else "--",
                linewidth=1.0,
                label=f"{PARAMETER_LABELS[name]} ({ks_pvalues[name]:.2f})",
            )
        pp_axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="C.I.",
            ylabel="Fraction of truths in C.I.",
            xticks=np.linspace(0.0, 1.0, 6),
            yticks=np.linspace(0.0, 1.0, 6),
        )
        pp_axis.set_aspect("equal", adjustable="box")
        pp_axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.015, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=8.3,
            handlelength=2.1,
            handletextpad=0.55,
            labelspacing=0.18,
        )
        pp_axis.text(
            0.965,
            0.06,
            (
                "fast-ridge\n"
                "\n"
                r"$\phi_c$, $t_c$ marg., ZoomFFT"
                "\n"
                f"$U = {config['time_marginalization']['upsample_factor']}$, "
                f"$M = {config['num_gibbs_sweeps']}$\n"
                f"$D = {config['n_devices']}$\n"
                f"$N = {n}$\n"
                f"$p = {fisher_pvalue:.2f}$"
            ),
            ha="right",
            va="bottom",
            fontsize=8.6,
            linespacing=1.12,
            transform=pp_axis.transAxes,
        )

        edges = np.histogram_bin_edges(post_jit_seconds, bins=15)
        hist_axis.hist(
            post_jit_seconds,
            bins=edges,
            color="#4878cf",
            edgecolor="white",
            linewidth=0.6,
        )
        median = float(np.median(post_jit_seconds))
        hist_axis.axvline(median, color="black", linewidth=1.0, linestyle="--")
        hist_axis.text(
            median,
            hist_axis.get_ylim()[1] * 0.97,
            f"  median {median:.1f} s",
            ha="left",
            va="top",
            fontsize=9.5,
        )
        hist_axis.set(
            xlabel="Post-JIT sampling time [s]",
            ylabel="Injections",
        )
        hist_axis.margins(x=0.02)
        hist_axis.spines["top"].set_visible(False)
        hist_axis.spines["right"].set_visible(False)

        for suffix in ("png", "pdf"):
            figure.savefig(
                HERE / f"pp-timing.{suffix}",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.03,
                facecolor="white",
            )
        plt.close(figure)


def main() -> None:
    rank_arrays = load_ranks()
    ks_pvalues, fisher_pvalue = load_report()
    timing_rows = extract_timing()
    write_timing_csv(timing_rows)
    post_jit = np.asarray(
        [row["post_jit_sampling_seconds"] for row in timing_rows], dtype=float
    )
    draw_figure(rank_arrays, ks_pvalues, fisher_pvalue, post_jit)
    print(
        f"n={post_jit.size} post-JIT sampling seconds: "
        f"median={np.median(post_jit):.1f} min={post_jit.min():.1f} "
        f"max={post_jit.max():.1f} mean={post_jit.mean():.1f}"
    )
    print(f"Fisher-combined p = {fisher_pvalue:.4f}")


if __name__ == "__main__":
    main()
