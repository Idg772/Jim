# /// script
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "scipy",
#   "anesthetic"
# ]
# ///

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Keep Matplotlib's cache out of a potentially read-only home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "jim-corner-mpl")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from anesthetic import Samples
from anesthetic.plot import kde_plot_1d
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
STEM = "candidate-paper-15d-fast-ridge-adaptive-shrink-only-m2-covariance-g4"

POOLED_COLOR = "#E68624"
SEED_COLOR = "#2878B5"
SEED_LINESTYLES = ("-", "--", ":")
CONTOUR_PROBABILITY = 0.90
KDE_GRID_POINTS_1D = 256
KDE_GRID_POINTS_2D = 600
NESTED_COMPRESSION = "entropy"
RNG_SEED = 1_187_008_882


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    physical_bounds: tuple[float, float]


PARAMETERS = (
    Parameter("M_c", r"$\mathcal{M}_c\,[M_\odot]$", (1.18, 1.21)),
    Parameter("q", r"$q$", (0.125, 1.0)),
    Parameter("s1_mag", r"$|\mathbf{s}_1|$", (0.0, 0.05)),
    Parameter("s1_theta", r"$\theta_1$", (0.0, np.pi)),
    Parameter("s1_phi", r"$\phi_1$", (0.0, 2.0 * np.pi)),
    Parameter("s2_mag", r"$|\mathbf{s}_2|$", (0.0, 0.05)),
    Parameter("s2_theta", r"$\theta_2$", (0.0, np.pi)),
    Parameter("s2_phi", r"$\phi_2$", (0.0, 2.0 * np.pi)),
    Parameter("iota", r"$\iota$", (0.0, np.pi)),
    Parameter("d_L", r"$d_L\,[\mathrm{Mpc}]$", (1.0, 75.0)),
    Parameter("psi", r"$\psi$", (0.0, np.pi)),
    Parameter("ra", r"$\alpha$", (0.0, 2.0 * np.pi)),
    Parameter("dec", r"$\delta$", (-0.5 * np.pi, 0.5 * np.pi)),
    Parameter("lambda_1", r"$\Lambda_1$", (0.0, 5000.0)),
    Parameter("lambda_2", r"$\Lambda_2$", (0.0, 5000.0)),
)
PARAMETER_KEYS = tuple(parameter.key for parameter in PARAMETERS)
PARAMETER_LABELS = tuple(parameter.label for parameter in PARAMETERS)

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ("STIX Two Text", "Times New Roman", "DejaVu Serif"),
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "axes.labelsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
}


@dataclass
class Posterior:
    name: str
    values: dict[str, np.ndarray]
    weights: np.ndarray
    samples: Samples
    report: dict[str, Any] | None = None


def stable_logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def make_samples(
    values: dict[str, np.ndarray], weights: np.ndarray, label: str
) -> Samples:
    matrix = np.column_stack([values[key] for key in PARAMETER_KEYS])
    samples = Samples(matrix, columns=PARAMETER_KEYS, weights=weights, label=label)
    samples.set_labels(PARAMETER_LABELS, inplace=True)
    return samples


def load_seed(seed: int) -> Posterior:
    seed_dir = HERE / f"seed-{seed}"
    nested_path = seed_dir / "nested" / f"{STEM}-seed{seed}.npz"
    report_path = seed_dir / f"{STEM}-seed{seed}.json"
    with np.load(nested_path, allow_pickle=False) as archive:
        missing = sorted({*PARAMETER_KEYS, "log_weights"}.difference(archive.files))
        if missing:
            raise ValueError(f"{nested_path}: missing fields: {', '.join(missing)}")
        values = {
            key: np.asarray(archive[key], dtype=np.float64) for key in PARAMETER_KEYS
        }
        log_weights = np.asarray(archive["log_weights"], dtype=np.float64)

    normalizer = stable_logsumexp(log_weights)
    weights = np.exp(log_weights - normalizer)
    weights /= weights.sum()
    for parameter in PARAMETERS:
        low, high = parameter.physical_bounds
        value = values[parameter.key]
        if np.min(value) < low - 1e-9 or np.max(value) > high + 1e-9:
            raise ValueError(
                f"{nested_path}: {parameter.key} falls outside [{low}, {high}]"
            )

    report = json.loads(report_path.read_text())
    config = report["config"]
    if config["num_gibbs_sweeps"] != 2:
        raise ValueError(f"{report_path}: expected num_gibbs_sweeps=2")
    if config.get("num_de_jumps", 0) != 0:
        raise ValueError(f"{report_path}: expected zero DE jumps")
    return Posterior(
        name=f"seed {seed}",
        values=values,
        weights=weights,
        samples=make_samples(values, weights, f"seed {seed}"),
        report=report,
    )


def pool_seeds(posteriors: tuple[Posterior, ...]) -> Posterior:
    # Each independently normalized seed contributes equal total mass.
    values = {
        key: np.concatenate([posterior.values[key] for posterior in posteriors])
        for key in PARAMETER_KEYS
    }
    weights = np.concatenate(
        [posterior.weights / len(posteriors) for posterior in posteriors]
    )
    weights /= weights.sum()
    return Posterior(
        name="combined",
        values=values,
        weights=weights,
        samples=make_samples(values, weights, "combined"),
    )


def mass_threshold(density: np.ndarray, probability: float) -> float:
    flat = density.ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cumulative /= cumulative[-1]
    index = min(int(np.searchsorted(cumulative, probability)), len(order) - 1)
    return float(flat[order[index]])


def sky_density(posterior: Posterior) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Wrap right ascension directly onto Matplotlib's [-pi, pi] longitude.
    longitude = (posterior.values["ra"] + np.pi) % (2.0 * np.pi) - np.pi
    latitude = posterior.values["dec"]
    density, longitude_edges, latitude_edges = np.histogram2d(
        longitude,
        latitude,
        bins=(96, 48),
        range=((-np.pi, np.pi), (-0.5 * np.pi, 0.5 * np.pi)),
        weights=posterior.weights,
    )
    density = gaussian_filter(density.T, sigma=(1.15, 1.15), mode=("nearest", "wrap"))
    longitude_centers = 0.5 * (longitude_edges[:-1] + longitude_edges[1:])
    latitude_centers = 0.5 * (latitude_edges[:-1] + latitude_edges[1:])
    return longitude_centers, latitude_centers, density


def add_chirp_mass_inset(
    figure: plt.Figure,
    individual: tuple[Posterior, ...],
    combined: Posterior | None = None,
) -> None:
    axis = figure.add_axes((0.635, 0.735, 0.275, 0.062))
    if combined is None:
        posterior = individual[0]
        kde_plot_1d(
            axis,
            posterior.values["M_c"],
            weights=posterior.weights,
            color=SEED_COLOR,
            facecolor=SEED_COLOR,
            levels=[CONTOUR_PROBABILITY],
            q=0,
            nplot_1d=KDE_GRID_POINTS_1D,
            bw_scale=0.9,
            linewidth=0.8,
            alpha=0.25,
        )
    else:
        kde_plot_1d(
            axis,
            combined.values["M_c"],
            weights=combined.weights,
            color=POOLED_COLOR,
            facecolor=POOLED_COLOR,
            levels=[CONTOUR_PROBABILITY],
            q=0,
            nplot_1d=KDE_GRID_POINTS_1D,
            bw_scale=0.9,
            alpha=0.38,
            linewidth=0.8,
        )
        for posterior, linestyle in zip(
            individual, SEED_LINESTYLES, strict=True
        ):
            kde_plot_1d(
                axis,
                posterior.values["M_c"],
                weights=posterior.weights,
                color=SEED_COLOR,
                facecolor=False,
                q=0,
                nplot_1d=KDE_GRID_POINTS_1D,
                bw_scale=0.9,
                linewidth=0.65,
                linestyle=linestyle,
            )
    axis.set_yticks([])
    axis.set_xlabel(r"$\mathcal{M}_c\,[M_\odot]$", fontsize=7)
    axis.set_title("Chirp mass", fontsize=7, pad=2)
    axis.tick_params(axis="x", labelsize=6, length=2, pad=1)
    for spine in axis.spines.values():
        spine.set_linewidth(0.55)


def add_sky_inset(
    figure: plt.Figure,
    individual: tuple[Posterior, ...],
    combined: Posterior | None = None,
) -> None:
    axis = figure.add_axes((0.635, 0.525, 0.275, 0.145), projection="mollweide")
    if combined is None:
        contours = ((individual[0], SEED_COLOR, True, 0.7, "-", 1),)
    else:
        contours = (
            (combined, POOLED_COLOR, True, 0.7, "-", 1),
            *(
                (seed, SEED_COLOR, False, 0.6, linestyle, 2)
                for seed, linestyle in zip(
                    individual, SEED_LINESTYLES, strict=True
                )
            ),
        )
    for posterior, color, filled, linewidth, linestyle, zorder in contours:
        longitude, latitude, density = sky_density(posterior)
        x, y = np.meshgrid(longitude, latitude)
        threshold = mass_threshold(density, CONTOUR_PROBABILITY)
        maximum = float(np.max(density))
        if filled:
            axis.contourf(
                x,
                y,
                density,
                levels=(threshold, np.nextafter(maximum, np.inf)),
                colors=(color,),
                alpha=0.38,
                zorder=zorder,
            )
        axis.contour(
            x,
            y,
            density,
            levels=(threshold,),
            colors=(color,),
            linewidths=linewidth,
            linestyles=linestyle,
            zorder=zorder + 0.1,
        )
    axis.grid(True, color="#D2D2D2", linewidth=0.4)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_title(r"sky $(\alpha,\delta)$, full sky", fontsize=7, pad=3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.55)


def style_corner_axes(axes: Any) -> None:
    for y_key, row in axes.iterrows():
        for x_key, axis in row.items():
            if axis is None:
                continue
            if x_key == y_key:
                axis.twin.set_yticks([])
            axis.xaxis.set_major_locator(mticker.MaxNLocator(2, min_n_ticks=2))
            axis.yaxis.set_major_locator(mticker.MaxNLocator(2, min_n_ticks=2))
            axis.tick_params(axis="both", labelsize=5.1, length=2, pad=1)
            axis.xaxis.label.set_size(7.5)
            axis.yaxis.label.set_size(7.5)
            for spine in axis.spines.values():
                spine.set_linewidth(0.55)
            if x_key == "M_c":
                formatter = mticker.ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                axis.xaxis.set_major_formatter(formatter)


def render_corner(
    individual: tuple[Posterior, ...],
    output_path: Path,
    combined: Posterior | None = None,
) -> None:
    np.random.seed(RNG_SEED)
    common_kind = {"diagonal": "kde_1d", "lower": "kde_2d"}
    base = combined if combined is not None else individual[0]
    base_color = POOLED_COLOR if combined is not None else SEED_COLOR
    base_alpha = 0.38 if combined is not None else 0.25
    axes = base.samples[list(PARAMETER_KEYS)].plot_2d(
        axes=list(PARAMETER_KEYS),
        kind=common_kind,
        label=base.name,
        diagonal_kwargs={
            "color": base_color,
            "facecolor": base_color,
            "levels": [CONTOUR_PROBABILITY],
            "ncompress": NESTED_COMPRESSION,
            "nplot_1d": KDE_GRID_POINTS_1D,
            "bw_scale": 0.9,
            "alpha": base_alpha,
            "linewidth": 0.75,
        },
        lower_kwargs={
            "color": base_color,
            "facecolor": base_color,
            "edgecolor": base_color,
            "levels": [CONTOUR_PROBABILITY],
            "ncompress": NESTED_COMPRESSION,
            "nplot_2d": KDE_GRID_POINTS_2D,
            "bw_scale": 1.0,
            "alpha": base_alpha,
            "linewidths": 0.65,
        },
    )
    overlays = (
        zip(individual, SEED_LINESTYLES, strict=True)
        if combined is not None
        else ()
    )
    for posterior, linestyle in overlays:
        posterior.samples[list(PARAMETER_KEYS)].plot_2d(
            axes=axes,
            kind=common_kind,
            label=posterior.name,
            diagonal_kwargs={
                "color": SEED_COLOR,
                "facecolor": False,
                "ncompress": NESTED_COMPRESSION,
                "nplot_1d": KDE_GRID_POINTS_1D,
                "bw_scale": 0.9,
                "linewidth": 0.65,
                "linestyle": linestyle,
                "alpha": 0.95,
            },
            lower_kwargs={
                "color": SEED_COLOR,
                "facecolor": None,
                "edgecolor": SEED_COLOR,
                "levels": [CONTOUR_PROBABILITY],
                "ncompress": NESTED_COMPRESSION,
                "nplot_2d": KDE_GRID_POINTS_2D,
                "bw_scale": 1.0,
                "linewidths": 0.6,
                "linestyles": linestyle,
                "alpha": 0.95,
            },
        )

    figure = axes.iloc[0, 0].figure
    figure.set_size_inches(12.0, 12.0)
    figure.subplots_adjust(
        left=0.065,
        bottom=0.065,
        right=0.975,
        top=0.94,
        wspace=0.045,
        hspace=0.045,
    )
    style_corner_axes(axes)
    add_chirp_mass_inset(figure, individual, combined)
    add_sky_inset(figure, individual, combined)

    figure.text(
        0.772,
        0.925,
        "GW170817",
        ha="center",
        va="top",
        fontsize=14,
    )
    handles = [
        Line2D(
            (0, 1),
            (0, 0),
            color=SEED_COLOR,
            linewidth=0.8,
            linestyle=linestyle,
            label=posterior.name,
        )
        for posterior, linestyle in zip(
            individual,
            SEED_LINESTYLES[: len(individual)],
            strict=True,
        )
    ]
    if combined is not None:
        handles.append(
            Line2D(
                (0, 1),
                (0, 0),
                color=POOLED_COLOR,
                alpha=0.58,
                linewidth=4.2,
                label="combined",
            )
        )
    figure.legend(
        handles=handles,
        title="M=2",
        loc="upper center",
        bbox_to_anchor=(0.772, 0.892),
        frameon=False,
        fontsize=8.5,
        title_fontsize=9.5,
        handlelength=2.2,
        borderaxespad=0,
    )
    figure.savefig(
        output_path,
        dpi=220,
        facecolor="white",
        metadata={
            "Title": "GW170817 15D posterior",
            "Description": (
                "M=2, zero-DE individual seeds and equal-seed pooled posterior; "
                "Figure-4 styling"
            ),
        },
    )
    plt.close(figure)


def main() -> None:
    posteriors = tuple(load_seed(seed) for seed in SEEDS)
    combined = pool_seeds(posteriors)

    for seed, posterior in zip(SEEDS, posteriors, strict=True):
        results = posterior.report["results"]
        ess = 1.0 / np.sum(posterior.weights**2)
        print(
            f"seed {seed}: n_dead={posterior.weights.size} ESS={ess:.0f} "
            f"log_Z={results['log_Z']:.2f} +/- {results['log_Z_error']:.2f} "
            f"n_iterations={results['n_iterations']}"
        )
        render_corner(
            (posterior,),
            HERE / f"corner-seed{seed}.png",
        )

    render_corner(
        posteriors,
        HERE / "corner-overlay.png",
        combined=combined,
    )
    print("wrote corner-seed0/1/2.png and corner-overlay.png")


if __name__ == "__main__":
    with plt.rc_context(PAPER_STYLE):
        main()
