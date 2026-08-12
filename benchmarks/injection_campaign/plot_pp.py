"""Aggregate completed posterior ranks and render P-P calibration plots."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom, combine_pvalues, kstest

from benchmarks.injection_campaign.common import (
    DEFAULT_CONFIG,
    MARGINALIZED_PARAMETERS,
    PAPER_PP_RECOVERIES,
    PARAMETERS,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    load_manifest,
    posterior_rank,
    read_catalogue,
    require_publication_eligible,
    result_dir,
)
from benchmarks.injection_campaign.run_injection import _rank_truth_coordinates

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
)
SIGMA_LEVELS = (1, 2, 3)
REMEDIATION_ALPHA = 0.05
REMEDIATION_EXCLUDED_PARAMETERS: tuple[str, ...] = ()
REMEDIATION_ALLOWED_CONFIG_VARIATIONS = frozenset(
    ("campaign", "paper_configuration", "blocks", "timing")
)
REMEDIATION_REQUIRED_CONFIG = {
    field: copy.deepcopy(value)
    for field, value in DEFAULT_CONFIG.items()
    if field not in REMEDIATION_ALLOWED_CONFIG_VARIATIONS
}
REMEDIATION_BLOCK_PARAMETERS = frozenset(
    (set(PARAMETERS) - {"ra", "dec"}) | {"zenith", "azimuth"}
)
RANK_RECOMPUTATION_TOLERANCE = 1.0e-12
PAPER_PARAMETER_ORDER = (
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
    "t_c",
)
PAPER_PARAMETER_LABELS = {
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
    "t_c": r"$t_c$",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Generate explicitly labelled exploratory plots from the completed "
            "subset. By default every selected leading injection is required."
        ),
    )
    return parser.parse_args(argv)


def _selected_injection_ids(manifest: dict[str, Any]) -> list[int]:
    """Return and validate the manifest's paper-style leading-ID selection."""

    n_injections = int(manifest["n_injections"])
    if n_injections < 1:
        raise ValueError("campaign must select at least one injection")
    selection = manifest.get("selection")
    if selection is not None:
        try:
            start = int(selection["start_inclusive"])
            stop = int(selection["stop_exclusive"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "campaign manifest has an invalid injection selection"
            ) from error
        if start != 0 or stop != n_injections:
            raise ValueError(
                "P-P diagnostics require the selected leading injection IDs "
                f"[0, {n_injections}); manifest declares [{start}, {stop})"
            )
    return list(range(n_injections))


def _remediation_eligibility(
    manifest: dict[str, Any], *, is_complete: bool
) -> tuple[bool, list[str]]:
    """Require the full corrected Fig. 2a-style M=1 protocol."""

    reasons: list[str] = []
    if not is_complete:
        reasons.append("selected recovery set is incomplete")
    if manifest.get("n_injections") != PAPER_PP_RECOVERIES:
        reasons.append(f"requires exactly {PAPER_PP_RECOVERIES} selected recoveries")
    config = manifest.get("config")
    if not isinstance(config, dict):
        reasons.append("manifest configuration is missing or invalid")
    else:
        expected_fields = set(DEFAULT_CONFIG)
        actual_fields = set(config)
        missing_fields = sorted(expected_fields - actual_fields)
        extra_fields = sorted(actual_fields - expected_fields)
        if missing_fields:
            reasons.append(f"config is missing fields {missing_fields}")
        if extra_fields:
            reasons.append(f"config has unsupported fields {extra_fields}")
        for field, expected in REMEDIATION_REQUIRED_CONFIG.items():
            if config.get(field) != expected:
                reasons.append(f"config.{field} must equal {expected!r}")
        blocks = config.get("blocks")
        flattened_blocks: list[str] = []
        if (
            not isinstance(blocks, list)
            or not blocks
            or any(
                not isinstance(block, list)
                or not block
                or not all(isinstance(name, str) and name for name in block)
                for block in blocks
            )
        ):
            reasons.append("config.blocks must be a non-empty list of named blocks")
        else:
            flattened_blocks = [name for block in blocks for name in block]
            if len(flattened_blocks) != len(set(flattened_blocks)):
                reasons.append("config.blocks must not repeat parameters")
            if set(flattened_blocks) != REMEDIATION_BLOCK_PARAMETERS:
                reasons.append(
                    "config.blocks must cover exactly the 15 sampling-space parameters"
                )
    return not reasons, reasons


def _validated_summary_truth(
    summary: dict[str, Any],
    catalogue_row: dict[str, Any],
    *,
    summary_path: Path,
) -> dict[str, float]:
    """Bind a result summary's physical truths and seeds to the frozen catalogue."""

    truth = summary.get("truth")
    if not isinstance(truth, dict):
        raise TypeError(f"result has no valid truth inventory: {summary_path}")
    expected_names = {*PARAMETERS, *MARGINALIZED_PARAMETERS}
    if set(truth) != expected_names:
        raise ValueError(f"result truth inventory is invalid: {summary_path}")
    normalized: dict[str, float] = {}
    for name in (*PARAMETERS, *MARGINALIZED_PARAMETERS):
        try:
            actual = float(truth[name])
            expected = float(catalogue_row[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid truth for {name}: {summary_path}") from error
        if not math.isfinite(actual) or actual != expected:
            raise ValueError(f"truth mismatch for {name}: {summary_path}")
        normalized[name] = actual

    expected_seeds = {
        "noise": catalogue_row["noise_seed"],
        "sampler": catalogue_row["sampler_seed"],
    }
    if summary.get("seeds") != expected_seeds:
        raise ValueError(f"result seeds do not match the catalogue: {summary_path}")
    return normalized


def _recomputed_ranks(
    summary: dict[str, Any],
    truth: dict[str, float],
    posterior_path: Path,
    *,
    phase_marginalization: bool,
) -> tuple[dict[str, float], bool]:
    """Recompute every rank and validate both legacy and current summaries."""

    expected_rank_truth = _rank_truth_coordinates(
        truth,
        PARAMETERS,
        phase_marginalization=phase_marginalization,
    )

    stored_rank_truth = summary.get("rank_truth")
    legacy = stored_rank_truth is None
    if not legacy:
        if not isinstance(stored_rank_truth, dict) or set(stored_rank_truth) != set(
            PARAMETERS
        ):
            raise ValueError(f"invalid rank truth inventory: {posterior_path}")
        for name, expected in expected_rank_truth.items():
            try:
                actual = float(stored_rank_truth[name])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid rank truth for {name}: {posterior_path}"
                ) from error
            if not math.isfinite(actual) or not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=RANK_RECOMPUTATION_TOLERANCE,
            ):
                raise ValueError(f"rank truth mismatch for {name}: {posterior_path}")

    ranks = summary.get("ranks")
    if not isinstance(ranks, dict) or set(ranks) != set(PARAMETERS):
        raise ValueError(f"result has no valid rank inventory: {posterior_path}")
    stored_truth = (
        {name: truth[name] for name in PARAMETERS} if legacy else expected_rank_truth
    )
    try:
        with np.load(posterior_path, allow_pickle=False) as posterior:
            log_weights = np.asarray(posterior["log_weights"])
            recomputed = {
                name: posterior_rank(
                    np.asarray(posterior[name]),
                    expected_rank_truth[name],
                    log_weights,
                )
                for name in PARAMETERS
            }
            expected_stored_ranks = {
                name: posterior_rank(
                    np.asarray(posterior[name]), stored_truth[name], log_weights
                )
                for name in PARAMETERS
            }
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"cannot derive posterior ranks: {posterior_path}") from error
    for name, expected in expected_stored_ranks.items():
        try:
            actual = float(ranks[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid stored rank for {name}: {posterior_path}"
            ) from error
        if not math.isfinite(actual) or not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=RANK_RECOMPUTATION_TOLERANCE,
        ):
            raise ValueError(f"stored rank mismatch for {name}: {posterior_path}")
    return recomputed, legacy and phase_marginalization


def load_rank_rows(
    campaign_dir: Path,
    *,
    allow_partial: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_manifest(campaign_dir)
    require_publication_eligible(manifest, product="P-P calibration")
    selected_ids = _selected_injection_ids(manifest)
    catalogue = read_catalogue(campaign_dir / manifest["catalogue"]["path"])
    if len(catalogue) < len(selected_ids):
        raise ValueError("campaign catalogue is shorter than its selected recovery set")
    rows: list[dict[str, Any]] = []
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
        truth = _validated_summary_truth(
            summary,
            catalogue[injection_id],
            summary_path=path,
        )
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
        ranks, corrected_legacy = _recomputed_ranks(
            summary,
            truth,
            posterior_path,
            phase_marginalization=bool(
                manifest.get("config", {}).get("phase_marginalization", False)
            ),
        )
        row = {"injection_id": injection_id}
        for name in PARAMETERS:
            try:
                value = float(ranks[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"missing or invalid rank for {name}: {path}"
                ) from error
            tolerance = 16.0 * np.finfo(float).eps
            if not -tolerance <= value <= 1.0 + tolerance:
                raise ValueError(f"rank outside [0, 1] for {name}: {path}")
            row[name] = float(np.clip(value, 0.0, 1.0))
        row["_legacy_phase_gauge_corrected_parameters"] = (
            ["s1_phi", "s2_phi"] if corrected_legacy else []
        )
        rows.append(row)
    if missing_ids and not allow_partial:
        missing = ", ".join(str(injection_id) for injection_id in missing_ids)
        raise ValueError(
            "incomplete selected leading-ID set: "
            f"found {len(rows)}/{len(selected_ids)} recoveries; missing IDs: {missing}"
        )
    if not rows:
        raise ValueError("no completed injection summaries were found")
    return manifest, rows


def _confidence_band(
    n: int,
    x: np.ndarray,
    sigma: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an exact central Gaussian-sigma binomial interval for the ECDF."""

    tail = 0.5 * math.erfc(sigma / math.sqrt(2.0))
    lower = binom.ppf(tail, n, x) / n
    upper = binom.ppf(1.0 - tail, n, x) / n
    return lower, upper


def _draw_expected(
    ax: Any,
    n: int,
    *,
    diagonal_linestyle: str = "--",
) -> None:
    x = np.linspace(0.0, 1.0, 301)
    for sigma, color in ((3, "0.93"), (2, "0.84"), (1, "0.73")):
        lower, upper = _confidence_band(n, x, sigma)
        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            label="_nolegend_",
            zorder=0,
        )
    ax.plot(
        x,
        x,
        color="black",
        linewidth=0.9,
        linestyle=diagonal_linestyle,
        label="_nolegend_",
    )


def _ecdf(ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(ranks)
    x = np.concatenate(([0.0], ordered, [1.0]))
    y = np.concatenate(([0.0], np.arange(1, ranks.size + 1) / ranks.size, [1.0]))
    return x, y


def _format_pvalue(value: float) -> str:
    return f"{value:.3g}"


def _draw_paper_style_combined(
    output_path: Path,
    *,
    rank_arrays: dict[str, np.ndarray],
    ks_pvalues: dict[str, float],
    fisher_pvalue: float,
    configuration: str,
    n: int,
) -> None:
    """Render the full P-P panel in the visual style of paper Figures 2/5/6."""

    if set(PAPER_PARAMETER_ORDER) != set(PARAMETERS):
        raise RuntimeError("paper plot order does not match the campaign parameters")

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
        figure = plt.figure(figsize=(5.8, 3.9))
        axis = figure.add_axes((0.13, 0.16, 0.53, 0.79))
        _draw_expected(axis, n, diagonal_linestyle="-")

        paper_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for index, name in enumerate(PAPER_PARAMETER_ORDER):
            x, y = _ecdf(rank_arrays[name])
            axis.step(
                x,
                y,
                where="post",
                color=paper_colors[index % len(paper_colors)],
                linestyle="-" if index < 10 else "--",
                linewidth=1.0,
                label=(f"{PAPER_PARAMETER_LABELS[name]} ({ks_pvalues[name]:.2f})"),
            )

        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="C.I.",
            ylabel="Fraction of truths in C.I.",
            xticks=np.linspace(0.0, 1.0, 6),
            yticks=np.linspace(0.0, 1.0, 6),
        )
        axis.set_aspect("equal", adjustable="box")
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.015, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=8.3,
            handlelength=2.1,
            handletextpad=0.55,
            labelspacing=0.18,
        )
        axis.text(
            0.965,
            0.10,
            (
                f"{configuration}\n"
                "$d_L$ marg.\n"
                "$M = 1$\n"
                "$D = 4$\n"
                f"$N = {n}$\n"
                f"$p = {fisher_pvalue:.2f}$"
            ),
            ha="right",
            va="bottom",
            fontsize=9.5,
            linespacing=1.12,
            transform=axis.transAxes,
        )
        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
        plt.close(figure)


def _json_number(value: float) -> float | str:
    """Represent the rare infinite Fisher statistic without invalid JSON."""

    value = float(value)
    return value if math.isfinite(value) else ("Infinity" if value > 0 else "-Infinity")


def aggregate_rank_rows(
    output_root: Path,
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    selected_ids: list[int],
    input_summary_sha256: dict[str, str],
    allow_partial: bool = False,
    report_extensions: dict[str, Any] | None = None,
    print_report: bool = True,
) -> dict[str, Any]:
    """Render and report already-validated rank rows."""

    output_root = output_root.expanduser().resolve()
    if not rows:
        raise ValueError("no rank rows were provided")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected injection IDs must be unique")
    included_ids = [int(row["injection_id"]) for row in rows]
    if len(included_ids) != len(set(included_ids)):
        raise ValueError("rank rows contain duplicate injection IDs")
    unexpected_ids = sorted(set(included_ids) - set(selected_ids))
    if unexpected_ids:
        raise ValueError(f"rank rows contain unselected IDs: {unexpected_ids}")
    missing_ids = sorted(set(selected_ids) - set(included_ids))
    if missing_ids and not allow_partial:
        raise ValueError(f"rank rows are incomplete; missing IDs: {missing_ids}")
    if len(input_summary_sha256) != len(rows) or any(
        not isinstance(value, str) or len(value) != 64
        for value in input_summary_sha256.values()
    ):
        raise ValueError("input summary hash inventory is invalid")
    is_complete = not missing_ids
    output_dir = output_root / "pp"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(
        output_dir / "ranks.csv",
        [{name: row[name] for name in RANK_FIELDS} for row in rows],
        RANK_FIELDS,
    )
    n = len(rows)
    rank_arrays = {
        name: np.asarray([row[name] for row in rows], dtype=float)
        for name in PARAMETERS
    }
    summaries: list[dict[str, Any]] = []
    ks_pvalues: dict[str, float] = {}
    for name, ranks in rank_arrays.items():
        ks = kstest(
            ranks,
            "uniform",
            alternative="two-sided",
            method="exact",
        )
        ks_pvalues[name] = float(ks.pvalue)
        summaries.append(
            {
                "parameter": name,
                "n_injections": n,
                "mean_rank": float(np.mean(ranks)),
                "cdf_at_50": float(np.mean(ranks <= 0.5)),
                "cdf_at_90": float(np.mean(ranks <= 0.9)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    atomic_write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    fisher = combine_pvalues(
        [ks_pvalues[name] for name in PARAMETERS],
        method="fisher",
    )
    fisher_pvalue = float(fisher.pvalue)
    remediation_parameters = tuple(
        name for name in PARAMETERS if name not in REMEDIATION_EXCLUDED_PARAMETERS
    )
    remediation_fisher = combine_pvalues(
        [ks_pvalues[name] for name in remediation_parameters],
        method="fisher",
    )
    remediation_pvalue = float(remediation_fisher.pvalue)
    remediation_eligible, remediation_ineligibility_reasons = _remediation_eligibility(
        manifest, is_complete=is_complete
    )
    result_kind = "complete" if is_complete else "partial exploratory"
    configuration = str(
        manifest.get("config", {}).get("paper_configuration", "Sharded")
    )

    colors = plt.cm.turbo(np.linspace(0.02, 0.98, len(PARAMETERS)))
    _draw_paper_style_combined(
        output_dir / "pp-combined.png",
        rank_arrays=rank_arrays,
        ks_pvalues=ks_pvalues,
        fisher_pvalue=fisher_pvalue,
        configuration=configuration,
        n=n,
    )

    figure, axes = plt.subplots(4, 4, figsize=(13.0, 12.5), constrained_layout=True)
    for axis, name, color in zip(axes.flat, PARAMETERS, colors, strict=False):
        _draw_expected(axis, n)
        x, y = _ecdf(rank_arrays[name])
        axis.step(x, y, where="post", color=color, linewidth=1.45)
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            title=f"{name} (KS p={_format_pvalue(ks_pvalues[name])})",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    axes.flat[-1].axis("off")
    figure.supxlabel("C.I.")
    figure.supylabel("Fraction of truths in C.I.")
    figure.suptitle(
        f"{configuration} FSM/SwiG parameter P–P plots "
        f"({result_kind}, N={n}; Fisher p={_format_pvalue(fisher_pvalue)})"
    )
    figure.savefig(output_dir / "pp-grid.png", dpi=170)
    plt.close(figure)

    output_paths = (
        output_dir / "ranks.csv",
        output_dir / "summary.csv",
        output_dir / "pp-combined.png",
        output_dir / "pp-grid.png",
    )
    confidence_bands = []
    for sigma in SIGMA_LEVELS:
        tail = 0.5 * math.erfc(sigma / math.sqrt(2.0))
        confidence_bands.append(
            {
                "sigma": sigma,
                "central_probability": 1.0 - 2.0 * tail,
                "lower_tail_probability": tail,
                "upper_tail_probability": tail,
                "lower_count_quantile": tail,
                "upper_count_quantile": 1.0 - tail,
                "distribution": "Binomial(N, credible_level)",
                "interval": "equal-tail exact discrete quantiles",
            }
        )
    report = {
        "schema_version": 1,
        "paper_reference": manifest.get("config", {}).get(
            "paper_reference", "arXiv:2607.28265v1"
        ),
        "paper_configuration": configuration,
        "config_sha256": manifest["config_sha256"],
        "diagnostic_status": "complete" if is_complete else "partial-exploratory",
        "completed_injections": n,
        "requested_injections": manifest["n_injections"],
        "selection": {
            "rule": "leading contiguous injection IDs",
            "expected_injection_ids": selected_ids,
            "included_injection_ids": included_ids,
            "missing_injection_ids": missing_ids,
            "complete": is_complete,
            "allow_partial": allow_partial,
        },
        "methodology": {
            "credible_levels": manifest.get("config", {}).get("credible_levels"),
            "parameters": list(PARAMETERS),
            "n_parameters": len(PARAMETERS),
            "per_parameter_test": {
                "name": "one-sample Kolmogorov-Smirnov",
                "null_distribution": "Uniform(0, 1)",
                "alternative": "two-sided",
                "pvalue_method": "exact",
            },
            "combined_test": {
                "name": "Fisher's method",
                "number_of_pvalues": len(PARAMETERS),
                "degrees_of_freedom": 2 * len(PARAMETERS),
                "statistic": _json_number(fisher.statistic),
                "pvalue": fisher_pvalue,
            },
            "rank_coordinates": {
                "default": "catalogue physical coordinate",
                "phase_marginalized_spin_azimuths": (
                    "beta_i = (s_i_phi + phase_c) mod 2pi"
                ),
                "legacy_summaries_corrected_from_weighted_posteriors": sum(
                    bool(row["_legacy_phase_gauge_corrected_parameters"])
                    for row in rows
                ),
            },
            "confidence_bands": confidence_bands,
        },
        "remediation_assessment": {
            "criterion": (
                "Fisher-combined exact KS p-value > 0.05 over all 15 sampled "
                "parameters, including q"
            ),
            "alpha": REMEDIATION_ALPHA,
            "excluded_parameters": list(REMEDIATION_EXCLUDED_PARAMETERS),
            "exclusion_reason": None,
            "parameters": list(remediation_parameters),
            "combined_test": {
                "name": "Fisher's method",
                "number_of_pvalues": len(remediation_parameters),
                "degrees_of_freedom": 2 * len(remediation_parameters),
                "statistic": _json_number(remediation_fisher.statistic),
                "pvalue": remediation_pvalue,
            },
            "required_recoveries": PAPER_PP_RECOVERIES,
            "required_config": REMEDIATION_REQUIRED_CONFIG,
            "allowed_config_variations": sorted(REMEDIATION_ALLOWED_CONFIG_VARIATIONS),
            "blocking_requirement": (
                "any non-overlapping block partition covering exactly the 15 "
                "sampling-space parameters"
            ),
            "eligible": remediation_eligible,
            "ineligibility_reasons": remediation_ineligibility_reasons,
            "passes": (
                remediation_pvalue > REMEDIATION_ALPHA if remediation_eligible else None
            ),
        },
        "per_parameter": summaries,
        "input_summary_sha256": dict(input_summary_sha256),
        "artifact_sha256": {
            str(path.relative_to(output_root)): file_sha256(path)
            for path in output_paths
        },
        "outputs": [
            "pp/ranks.csv",
            "pp/summary.csv",
            "pp/pp-combined.png",
            "pp/pp-grid.png",
            "pp/report.json",
        ],
    }
    extensions = report_extensions or {}
    overlap = sorted(set(report) & set(extensions))
    if overlap:
        raise ValueError(f"report extensions replace standard fields: {overlap}")
    report.update(copy.deepcopy(extensions))
    atomic_write_json(output_dir / "report.json", report)
    if print_report:
        print(json.dumps(report, indent=2, sort_keys=True))
    return report


def aggregate_and_plot(
    campaign_dir: Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest, rows = load_rank_rows(campaign_dir, allow_partial=allow_partial)
    selected_ids = _selected_injection_ids(manifest)
    included_ids = [int(row["injection_id"]) for row in rows]
    input_summary_sha256 = {
        str(
            (result_dir(campaign_dir, injection_id) / "summary.json").relative_to(
                campaign_dir
            )
        ): file_sha256(result_dir(campaign_dir, injection_id) / "summary.json")
        for injection_id in included_ids
    }
    return aggregate_rank_rows(
        campaign_dir,
        manifest=manifest,
        rows=rows,
        selected_ids=selected_ids,
        input_summary_sha256=input_summary_sha256,
        allow_partial=allow_partial,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    aggregate_and_plot(args.campaign_dir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
