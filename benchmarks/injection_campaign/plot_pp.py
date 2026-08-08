"""Aggregate completed posterior ranks and render P-P calibration plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom, kstest

from benchmarks.injection_campaign.common import (
    PARAMETERS,
    atomic_write_csv,
    file_sha256,
    load_manifest,
    prior_cdf,
    read_catalogue,
    result_dir,
)

plt.switch_backend("Agg")


RANK_FIELDS = ("injection_id", *PARAMETERS)
SUMMARY_FIELDS = (
    "parameter",
    "n_injections",
    "mean_rank",
    "cdf_at_50",
    "cdf_at_90",
    "ks_statistic",
    "ks_pvalue",
    "truth_ks_statistic",
    "truth_ks_pvalue",
    "residual_mean",
    "residual_stderr",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    return parser.parse_args(argv)


def load_rank_rows(campaign_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(campaign_dir)
    rows: list[dict[str, Any]] = []
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
        posterior_path = path.parent / summary.get("posterior", {}).get(
            "path", "posterior.npz"
        )
        expected_sha256 = summary.get("posterior", {}).get("sha256")
        if (
            not posterior_path.is_file()
            or not isinstance(expected_sha256, str)
            or file_sha256(posterior_path) != expected_sha256
        ):
            raise ValueError(f"result posterior is missing or corrupt: {path.parent}")
        ranks = summary.get("ranks", {})
        row = {"injection_id": injection_id}
        for name in PARAMETERS:
            value = float(ranks[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"rank outside [0, 1] for {name}: {path}")
            row[name] = value
        rows.append(row)
    if not rows:
        raise ValueError("no completed injection summaries were found")
    return manifest, rows


def _confidence_band(n: int, x: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    tail = (1.0 - confidence) / 2.0
    lower = binom.ppf(tail, n, x) / n
    upper = binom.ppf(1.0 - tail, n, x) / n
    return lower, upper


def _draw_expected(ax: Any, n: int) -> None:
    x = np.linspace(0.0, 1.0, 301)
    lower95, upper95 = _confidence_band(n, x, 0.95)
    lower68, upper68 = _confidence_band(n, x, 0.68)
    ax.fill_between(x, lower95, upper95, color="0.88", label="95% band")
    ax.fill_between(x, lower68, upper68, color="0.73", label="68% band")
    ax.plot(x, x, color="black", linewidth=1.1, linestyle="--", label="Expected")


def _ecdf(ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(ranks)
    x = np.concatenate(([0.0], ordered, [1.0]))
    y = np.concatenate(([0.0], np.arange(1, ranks.size + 1) / ranks.size, [1.0]))
    return x, y


def aggregate_and_plot(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest, rows = load_rank_rows(campaign_dir)
    output_dir = campaign_dir / "pp"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "ranks.csv", rows, RANK_FIELDS)
    n = len(rows)
    rank_arrays = {
        name: np.asarray([row[name] for row in rows], dtype=float)
        for name in PARAMETERS
    }
    catalogue = read_catalogue(campaign_dir / manifest["catalogue"]["path"])
    truth_by_id = {row["injection_id"]: row for row in catalogue}
    truth_quantiles = {
        name: np.asarray(
            [prior_cdf(name, truth_by_id[row["injection_id"]][name]) for row in rows],
            dtype=float,
        )
        for name in PARAMETERS
    }
    summaries = []
    for name, ranks in rank_arrays.items():
        ks = kstest(ranks, "uniform")
        residual = ranks - truth_quantiles[name]
        truth_ks = kstest(truth_quantiles[name], "uniform")
        summaries.append(
            {
                "parameter": name,
                "n_injections": n,
                "mean_rank": float(np.mean(ranks)),
                "cdf_at_50": float(np.mean(ranks <= 0.5)),
                "cdf_at_90": float(np.mean(ranks <= 0.9)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "truth_ks_statistic": float(truth_ks.statistic),
                "truth_ks_pvalue": float(truth_ks.pvalue),
                "residual_mean": float(np.mean(residual)),
                "residual_stderr": float(
                    np.std(residual, ddof=1) / np.sqrt(residual.size)
                ),
            }
        )
    atomic_write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)

    combined, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    _draw_expected(ax, n)
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, len(PARAMETERS)))
    for color, name in zip(colors, PARAMETERS, strict=True):
        x, y = _ecdf(rank_arrays[name])
        ax.step(x, y, where="post", color=color, alpha=0.72, linewidth=1.2, label=name)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Credible level",
        ylabel="Fraction of injections",
        title=f"FSM/SwiG P–P calibration ({n} recoveries)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[3:], labels[3:], ncol=3, fontsize=8, loc="upper left")
    combined.savefig(output_dir / "pp-combined.png", dpi=180)
    plt.close(combined)

    figure, axes = plt.subplots(4, 4, figsize=(13.0, 12.5), constrained_layout=True)
    for axis, name, color in zip(axes.flat, PARAMETERS, colors, strict=False):
        _draw_expected(axis, n)
        x, y = _ecdf(rank_arrays[name])
        axis.step(x, y, where="post", color=color, linewidth=1.45)
        axis.set(xlim=(0, 1), ylim=(0, 1), title=name)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    axes.flat[-1].axis("off")
    figure.supxlabel("Credible level")
    figure.supylabel("Fraction of injections")
    figure.suptitle(f"FSM/SwiG parameter P–P plots ({n} recoveries)")
    figure.savefig(output_dir / "pp-grid.png", dpi=170)
    plt.close(figure)

    report = {
        "config_sha256": manifest["config_sha256"],
        "completed_injections": n,
        "requested_injections": manifest["n_injections"],
        "outputs": [
            "pp/ranks.csv",
            "pp/summary.csv",
            "pp/pp-combined.png",
            "pp/pp-grid.png",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> None:
    aggregate_and_plot(_parse_args(argv).campaign_dir)


if __name__ == "__main__":
    main()
