"""Plot the retained timing records. Run from any directory."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    with (folder / "timing.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    labels = [f"{row['run']}\n4 × {row['gpu']}" for row in rows]
    x = np.arange(len(rows))
    loop = np.array([float(row["sampling_loop_seconds"]) for row in rows])
    finalise = np.array([float(row["finalise_seconds"]) for row in rows])
    sample = [float(row["sample_call_seconds"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(x, loop, label="Sampling loop", color="#326b9c")
    axes[0].bar(x, finalise, bottom=loop, label="Final processing", color="#e0a348")
    axes[0].set_title("Sampling time with measured compilation excluded")
    axes[0].legend(frameon=False)
    for pos, total in zip(x, loop + finalise):
        axes[0].annotate(f"{total:.2f}", (pos, total), xytext=(0, 4),
                         textcoords="offset points", ha="center")
    axes[1].bar(x, sample, color="#69747c", label="Observed sample call")
    axes[1].set_title("Observed sample call, including first-use compilation")
    for pos, total in zip(x, sample):
        axes[1].annotate(f"{total:.2f}", (pos, total), xytext=(0, 4),
                         textcoords="offset points", ha="center")
    for ax in axes:
        ax.set_ylabel("Seconds")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[1].set_xticks(x, labels, fontsize=8)
    fig.text(0.5, 0.01, "These intervals do not measure a complete warm-worker pipeline.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"timing.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
