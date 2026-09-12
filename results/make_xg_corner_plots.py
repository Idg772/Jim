"""Create weighted corner plots from the retained XG run results."""

import argparse
import hashlib
import json
import os
import tempfile
import tomllib
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jim-corner-mpl")
)

import matplotlib

matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PARAMETERS = (
    ("M_c", r"$\mathcal{M}_c\ [M_\odot]$"),
    ("q", r"$q$"),
    ("s1_z", r"$\chi_{1z}$"),
    ("s2_z", r"$\chi_{2z}$"),
    ("lambda_1", r"$\Lambda_1$"),
    ("lambda_2", r"$\Lambda_2$"),
    ("d_L", r"$d_L\ [\mathrm{Mpc}]$"),
    ("iota", r"$\iota\ [\mathrm{rad}]$"),
    ("ra", r"$\alpha\ [\mathrm{rad}]$"),
    ("dec", r"$\delta\ [\mathrm{rad}]$"),
    ("psi", r"$\psi\ [\mathrm{rad}]$"),
    ("t_c", r"$t_c\ [\mathrm{s}]$"),
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plot_run(folder, run):
    run_folder = folder / run["id"]
    source = run_folder / "nested/samples.npz"
    config_path = run_folder / "config.final.toml"
    config = tomllib.loads(config_path.read_text())
    injection = config["data"]["injection_parameters"]
    truth = np.array([injection[key] for key, _ in PARAMETERS])
    with np.load(source, allow_pickle=False) as archive:
        values = np.column_stack([archive[key] for key, _ in PARAMETERS])
        log_weights = archive["log_weights"]
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError(f"Invalid values or weights in {source}")

    # Display the central 99.9% of each marginal and always include the injection.
    ranges = []
    for column, injected in zip(values.T, truth, strict=True):
        order = np.argsort(column)
        cumulative = np.cumsum(weights[order])
        low, high = np.interp((0.0005, 0.9995), cumulative, column[order])
        low, high = min(low, injected), max(high, injected)
        padding = 0.05 * (high - low)
        ranges.append((float(low - padding), float(high + padding)))

    color = "#276b91" if len(config["data"]["detectors"]) > 1 else "#267c70"
    figure = corner.corner(
        values,
        weights=weights,
        labels=[label for _, label in PARAMETERS],
        range=ranges,
        bins=48,
        smooth=0.8,
        smooth1d=0.8,
        levels=(0.5, 0.9),
        color=color,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        max_n_ticks=3,
        label_kwargs={"fontsize": 12},
        hist_kwargs={"linewidth": 1.3},
        contour_kwargs={"linewidths": 0.9},
    )
    axes = np.asarray(figure.axes).reshape(len(PARAMETERS), len(PARAMETERS))
    for row in range(len(PARAMETERS)):
        for col in range(row + 1):
            axis = axes[row, col]
            axis.axvline(truth[col], color="black", linestyle="-", linewidth=1.1)
            if row != col:
                axis.axhline(truth[row], color="black", linestyle="-", linewidth=1.1)
            axis.tick_params(labelsize=8)
    figure.set_size_inches(21, 21)
    figure.text(0.62, 0.93, run["id"], ha="center", fontsize=20)
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=color,
                linewidth=2,
                label="Posterior: 50% and 90% contours",
            ),
            Line2D(
                [],
                [],
                color="black",
                linestyle="-",
                linewidth=1.3,
                label="Injected value",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.62, 0.91),
        frameon=False,
        fontsize=13,
    )
    figure.text(
        0.62,
        0.85,
        "All weighted nested points; phase marginalized",
        ha="center",
        fontsize=12,
    )
    outputs = []
    for extension in ("png", "pdf"):
        path = run_folder / f"corner.{extension}"
        figure.savefig(path, dpi=160, facecolor="white")
        outputs.append({"path": str(path.relative_to(folder)), "sha256": sha256(path)})
    plt.close(figure)
    print(f"Created {run['id']}/corner.png and corner.pdf", flush=True)
    return {
        "run": run["id"],
        "source": str(source.relative_to(folder)),
        "source_sha256": sha256(source),
        "configuration_sha256": sha256(config_path),
        "points": len(weights),
        "injection": {key: injection[key] for key, _ in PARAMETERS},
        "ranges": dict(zip((key for key, _ in PARAMETERS), ranges, strict=True)),
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", choices=("2026-09-09", "2026-09-11"))
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / args.date
    index = json.loads((folder / "index.json").read_text())
    plots = [
        plot_run(folder, run)
        for run in index["runs"]
        if (folder / run["id"] / "nested/samples.npz").exists()
    ]
    (folder / "corner-plots.json").write_text(
        json.dumps(
            {
                "method": "Weighted histograms of all saved nested points; no resampling.",
                "weights": "exp(log_weights - max(log_weights)), normalized to sum to one",
                "contour_probability": [0.5, 0.9],
                "bins": 48,
                "gaussian_smoothing_bins": 0.8,
                "range": "Central 99.9% per marginal, expanded to include the injection, with 5% padding.",
                "periodic_coordinates": "Original physical angles; no branch folding or selection.",
                "injection_lines": "Solid black vertical and horizontal lines; values from each run configuration.",
                "plots": plots,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
