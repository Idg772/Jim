"""Tests for the GW170817 weighted sampler-output analysis."""

import heapq
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.device_parallel_nss import analyze_gw170817_sampler_output as analysis


def _synthetic_nested_run(
    *,
    random_seed: int,
    replacement_scale: float,
    n_live: int = 64,
    n_replacements: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    death = list(rng.exponential(size=n_live))
    birth = [-np.inf] * n_live
    active = [(value, index) for index, value in enumerate(death)]
    heapq.heapify(active)
    for _ in range(n_replacements):
        contour, _ = heapq.heappop(active)
        replacement_death = contour + replacement_scale * rng.exponential()
        index = len(death)
        death.append(replacement_death)
        birth.append(contour)
        heapq.heappush(active, (replacement_death, index))
    return np.asarray(death), np.asarray(birth)


def test_insertion_rank_ks_accepts_a_perfect_nested_run() -> None:
    death, birth = _synthetic_nested_run(random_seed=12, replacement_scale=1.0)

    result = analysis.insertion_rank_diagnostic(
        death,
        birth,
        window_size=1000,
        random_seed=34,
    )

    assert result.global_p_value > 0.01
    assert result.worst_window_p_value > 0.01


def test_insertion_rank_ks_rejects_a_biased_nested_run() -> None:
    death, birth = _synthetic_nested_run(random_seed=12, replacement_scale=0.01)

    result = analysis.insertion_rank_diagnostic(
        death,
        birth,
        window_size=1000,
        random_seed=34,
    )

    assert result.global_p_value < 1e-20
    assert result.worst_window_p_value < 1e-20


def _write_report_triplet(
    root: Path,
    *,
    schemes: list[str],
    blocks: list[list[str]],
) -> tuple[list[Path], list[Path]]:
    nested_paths: list[Path] = []
    report_paths: list[Path] = []
    for seed, scheme in enumerate(schemes):
        nested_path = root / f"nested-seed{seed}.npz"
        nested_path.write_bytes(f"nested-{seed}".encode())
        report_path = root / f"report-seed{seed}.json"
        report_path.write_text(
            json.dumps(
                {
                    "config": {
                        "seed": seed,
                        "workload": "paper-15d",
                        "sampled_dimensions": 15,
                        "phase_marginalization": True,
                        "distance_marginalization": False,
                        "blocking_scheme": scheme,
                        "blocks": blocks,
                    },
                    "data": {"sha256": "frozen-bundle"},
                    "results": {
                        "nested_artifact": {
                            "weighting": analysis.WEIGHTING,
                            "sha256": analysis._sha256(nested_path),
                        }
                    },
                }
            )
        )
        nested_paths.append(nested_path)
        report_paths.append(report_path)
    return report_paths, nested_paths


def test_reports_reject_mixed_blocking_schemes(tmp_path: Path) -> None:
    report_paths, nested_paths = _write_report_triplet(
        tmp_path,
        schemes=["all-slow", "all-slow", "paper"],
        blocks=[list(block) for block in analysis.ALL_SLOW_BLOCKS],
    )

    with pytest.raises(ValueError, match="different blocking schemes"):
        analysis._reports(report_paths, nested_paths)


def test_reports_reject_wrong_all_slow_blocks(tmp_path: Path) -> None:
    report_paths, nested_paths = _write_report_triplet(
        tmp_path,
        schemes=["all-slow"] * 3,
        blocks=[list(block) for block in analysis.PAPER_BLOCKS],
    )

    with pytest.raises(ValueError, match="wrong blocks for all-slow"):
        analysis._reports(report_paths, nested_paths)


def _write_netsky_report_triplet(
    tmp_path: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    report_paths, posterior_paths = _write_report_triplet(
        tmp_path,
        schemes=["netsky"] * 3,
        blocks=[list(block) for block in analysis.NETSKY_BLOCKS],
    )
    folded_paths = []
    for seed, (report_path, posterior_path) in enumerate(
        zip(report_paths, posterior_paths, strict=True)
    ):
        folded_path = tmp_path / f"folded-seed{seed}.npz"
        folded_path.write_bytes(f"folded-{seed}".encode())
        report = json.loads(report_path.read_text())
        report["config"].update(
            {
                "bridge_blocks": [
                    list(block) for block in analysis.NETSKY_BRIDGE_BLOCKS
                ],
                "periodic_wrapped_covariance": True,
                "num_gibbs_sweeps": 2,
                "n_live": 512,
                "fold_symmetry": {
                    "cos_iota": "cos_iota",
                    "azimuth": "azimuth",
                    "psi": "psi",
                    "azimuth_reflection_center": 0.25,
                },
            }
        )
        posterior_artifact = report["results"].pop("nested_artifact")
        posterior_artifact.update(
            {
                "path": str(posterior_path),
                "space": "prior",
                "weighting": analysis.UNFOLDED_POSTERIOR_WEIGHTING,
                "schema_version": 2,
                "count": 800,
                "weight_effective_size_semantics": (
                    analysis.POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
                ),
                "fields": [
                    *analysis.PARAMETERS,
                    "log_likelihood",
                    "log_weights",
                ],
            }
        )
        report["results"]["posterior_artifact"] = posterior_artifact
        report["results"].update(
            {
                "folded_nested_diagnostics": {
                    "path": str(folded_path),
                    "sha256": analysis._sha256(folded_path),
                    "space": analysis.FOLDED_TARGET_SPACE,
                    "semantics": analysis.FOLDED_TARGET_SEMANTICS,
                    "weighting": analysis.FOLDED_TARGET_WEIGHTING,
                    "count": 100,
                    "fields": ["log_likelihood", "log_likelihood_birth"],
                },
                "posterior_weight_effective_size": 120.0,
                "posterior_weight_effective_size_semantics": (
                    analysis.POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
                ),
                "log_Z": -42.0 + seed,
                "log_Z_error": 0.2,
                "insertion_index_diagnostic": {
                    "method": "discrete-uniform-kolmogorov-smirnov",
                    "n_live": 512,
                    "sample_size": 100,
                    "statistic": 0.05,
                    "p_value": 0.5,
                },
                "quotient_fold": {
                    "completed_config": report["config"]["fold_symmetry"],
                    "model_limitation": "low-spin test fixture",
                    "batch_size": 1,
                    "group_order": 8,
                    "folded_points": 100,
                    "normalized_conditional_image_entropy": 0.8,
                    "image_sector_posterior_masses": [0.3, *([0.1] * 7)],
                    "expected_nonidentity_mass": 0.7,
                    "zero_support_image_fraction": 0.1,
                    "supported_image_log_likelihood_gaps": {
                        "within_orbit_span_weighted_quantiles": {
                            "p05": 0.1,
                            "p50": 0.5,
                            "p95": 1.0,
                        },
                        "identity_absolute_gap_weighted_quantiles": {
                            "p05": 0.05,
                            "p50": 0.25,
                            "p95": 0.75,
                        },
                    },
                    "projection_accounting": {
                        "images_per_folded_target_callback": 8,
                        "sampler_folded_target_callbacks": 200,
                        "sampler_true_image_projections": 1600,
                        "retained_folded_points_unfolded": 100,
                        "unfold_true_image_projections": 800,
                        "total_true_image_projections": 2400,
                        "sampler_callback_counter": (
                            "results.n_likelihood_evaluations_physical"
                        ),
                    },
                },
            }
        )
        report["timing_seconds"] = {"fold_unfold_postprocessing": 0.5}
        report_path.write_text(json.dumps(report))
        folded_paths.append(folded_path)
    return report_paths, posterior_paths, folded_paths


def test_reports_accept_netsky_with_separate_physical_and_folded_artifacts(
    tmp_path: Path,
) -> None:
    report_paths, posterior_paths, folded_paths = _write_netsky_report_triplet(tmp_path)

    seeds, bundle_hash, scheme, blocks = analysis._reports(
        report_paths,
        posterior_paths,
        folded_paths=folded_paths,
    )

    assert seeds == [0, 1, 2]
    assert bundle_hash == "frozen-bundle"
    assert scheme == analysis.NETSKY_BLOCKING_SCHEME
    assert blocks == analysis.NETSKY_BLOCKS


@pytest.mark.parametrize("posterior_count", [99, 801])
def test_reports_reject_netsky_posterior_outside_orbit_expansion_bounds(
    tmp_path: Path,
    posterior_count: int,
) -> None:
    report_paths, posterior_paths, folded_paths = _write_netsky_report_triplet(tmp_path)
    report = json.loads(report_paths[1].read_text())
    report["results"]["posterior_artifact"]["count"] = posterior_count
    report["results"]["posterior_weight_effective_size"] = min(
        50.0, float(posterior_count)
    )
    report_paths[1].write_text(json.dumps(report))

    with pytest.raises(ValueError, match="posterior count is outside"):
        analysis._reports(
            report_paths,
            posterior_paths,
            folded_paths=folded_paths,
        )


def test_netsky_seed_summary_separates_physical_and_folded_artifacts(
    tmp_path: Path,
) -> None:
    from jimgw.samplers.diagnostics import insertion_index_diagnostic

    posterior_path = tmp_path / "posterior.npz"
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])
    np.savez(
        posterior_path,
        M_c=np.asarray([1.19, 1.20, 1.21, 1.22]),
        q=np.asarray([0.7, 0.75, 0.9, 0.95]),
        lambda_1=np.asarray([100.0, 200.0, 300.0, 400.0]),
        lambda_2=np.asarray([500.0, 600.0, 700.0, 800.0]),
        d_L=np.asarray([35.0, 45.0, 80.0, 100.0]),
        iota=np.asarray([0.2, 0.4, 2.7, 2.9]),
        log_likelihood=np.full(4, 999.0),
        log_weights=np.log(probabilities),
    )
    death, birth = _synthetic_nested_run(
        random_seed=12,
        replacement_scale=1.0,
        n_live=8,
        n_replacements=40,
    )
    folded_path = tmp_path / "folded.npz"
    np.savez(
        folded_path,
        log_likelihood=death,
        log_likelihood_birth=birth,
    )
    report = {
        "config": {"n_live": 8},
        "results": {
            "log_Z": -123.5,
            "log_Z_error": 0.25,
            "posterior_weight_effective_size": 1.0 / np.sum(probabilities**2),
            "posterior_weight_effective_size_semantics": (
                analysis.POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
            ),
            "insertion_index_diagnostic": insertion_index_diagnostic(
                death,
                birth,
                n_live=8,
            ),
            "quotient_fold": {
                "group_order": 8,
                "folded_points": len(death),
                "normalized_conditional_image_entropy": 0.75,
                "image_sector_posterior_masses": [0.3, *([0.1] * 7)],
                "expected_nonidentity_mass": 0.7,
                "zero_support_image_fraction": 0.125,
                "supported_image_log_likelihood_gaps": {
                    "within_orbit_span_weighted_quantiles": {
                        "p05": 0.1,
                        "p50": 0.5,
                        "p95": 1.5,
                    },
                    "identity_absolute_gap_weighted_quantiles": None,
                },
            },
        },
        "timing_seconds": {"fold_unfold_postprocessing": 0.75},
    }

    summary = analysis._netsky_seed_summary(
        posterior_path,
        folded_path,
        report,
        seed=7,
        window_size=20,
        random_seed=31,
    )

    assert summary["log_z"] == pytest.approx(-123.5)
    assert summary["insertion"].sample_size == 40
    assert summary["parameters"]["q"] == {
        "lower": pytest.approx(0.7),
        "median": pytest.approx(0.9071428571428571),
        "upper": pytest.approx(0.95),
    }
    assert summary["ridge_occupancy"]["cos_iota_nonnegative"][
        "posterior_mass"
    ] == pytest.approx(0.3)
    assert summary["ridge_occupancy"]["cos_iota_negative"][
        "posterior_mass"
    ] == pytest.approx(0.7)
    assert summary["fold_telemetry"]["expected_nonidentity_mass"] == pytest.approx(0.7)


def test_netsky_cross_seed_stability_is_descriptive_without_a_gate() -> None:
    summaries = []
    for seed, offset in enumerate((0.0, 0.1, 0.2)):
        summaries.append(
            {
                "seed": seed,
                "posterior_weight_effective_size": 100.0 + seed,
                "fold_unfold_postprocessing_seconds": 0.5 + offset,
                "ridge_occupancy": {
                    "cos_iota_nonnegative": {
                        "posterior_mass": 0.4 + offset,
                        "d_L_quantiles": {"p05": 30.0, "p50": 40.0, "p95": 50.0},
                    },
                    "cos_iota_negative": {
                        "posterior_mass": 0.6 - offset,
                        "d_L_quantiles": {"p05": 60.0, "p50": 70.0, "p95": 80.0},
                    },
                },
                "fold_telemetry": {
                    "normalized_conditional_image_entropy": 0.7 + offset,
                    "image_sector_posterior_masses": [
                        0.3 + offset,
                        *([(0.7 - offset) / 7.0] * 7),
                    ],
                    "expected_nonidentity_mass": 0.7 - offset,
                    "zero_support_image_fraction": 0.1 + offset,
                    "supported_image_log_likelihood_gaps": {
                        "within_orbit_span_weighted_quantiles": {
                            "p05": 0.1,
                            "p50": 0.5 + offset,
                            "p95": 1.0,
                        },
                        "identity_absolute_gap_weighted_quantiles": None,
                    },
                },
            }
        )

    stability = analysis._netsky_cross_seed_stability(summaries)

    assert stability["role"] == "descriptive_cross_seed_stability"
    assert stability["used_as_acceptance_gate"] is False
    assert "passed" not in stability
    assert stability["metrics"]["normalized_conditional_image_entropy"][
        "range"
    ] == pytest.approx(0.2)
    assert len(stability["image_sector_posterior_masses"]) == 8
    assert stability["ridge_occupancy"]["cos_iota_nonnegative"]["posterior_mass"][
        "range"
    ] == pytest.approx(0.2)
    assert stability["likelihood_gap_quantiles"][
        "within_orbit_span_weighted_quantiles"
    ]["p50"]["range"] == pytest.approx(0.2)


def test_render_report_labels_netsky_physical_and_folded_diagnostics() -> None:
    summaries = []
    for seed, offset in enumerate((0.0, 0.05, 0.1)):
        summaries.append(
            {
                "seed": seed,
                "insertion": analysis.InsertionRankDiagnostic(
                    sample_size=100,
                    global_statistic=0.1,
                    global_p_value=0.5,
                    worst_window_p_value=0.2,
                    worst_window_start=0,
                    worst_window_stop=100,
                    window_size=100,
                    live_min=512,
                    live_max=512,
                ),
                "log_z": -42.0 + offset,
                "log_z_error": 0.2,
                "ess": 100.0 + seed,
                "posterior_weight_effective_size": 100.0 + seed,
                "parameters": {
                    name: {
                        "lower": 1.0 + offset,
                        "median": 2.0 + offset,
                        "upper": 3.0 + offset,
                    }
                    for name in analysis.PARAMETERS
                },
                "q_mode_count": 2,
                "q_mode_locations": [0.75, 0.9],
                "ridge_occupancy": {
                    "cos_iota_nonnegative": {
                        "posterior_mass": 0.4 + offset,
                        "d_L_quantiles": {"p05": 30.0, "p50": 40.0, "p95": 50.0},
                    },
                    "cos_iota_negative": {
                        "posterior_mass": 0.6 - offset,
                        "d_L_quantiles": {"p05": 60.0, "p50": 70.0, "p95": 80.0},
                    },
                },
                "fold_telemetry": {
                    "normalized_conditional_image_entropy": 0.7 + offset,
                    "image_sector_posterior_masses": [
                        0.3 + offset,
                        *([(0.7 - offset) / 7.0] * 7),
                    ],
                    "expected_nonidentity_mass": 0.7 - offset,
                    "zero_support_image_fraction": 0.1 + offset,
                    "supported_image_log_likelihood_gaps": {
                        "within_orbit_span_weighted_quantiles": {
                            "p05": 0.1,
                            "p50": 0.5,
                            "p95": 1.0,
                        },
                        "identity_absolute_gap_weighted_quantiles": None,
                    },
                },
                "fold_unfold_postprocessing_seconds": 0.5 + offset,
            }
        )
    stability = analysis._netsky_cross_seed_stability(summaries)

    report = analysis._render_report(
        summaries,
        bundle_sha256="bundle",
        blocking_scheme=analysis.NETSKY_BLOCKING_SCHEME,
        blocks=analysis.NETSKY_BLOCKS,
        sanity=None,
        netsky_stability=stability,
    )

    assert "Folded-target insertion-rank KS p (global)" in report
    assert "Unfolded physical q KDE mode count" in report
    assert "d_L–iota ridge cos(iota) >= 0" in report
    assert "Image-sector posterior masses (0–7)" in report
    assert "Normalized conditional image entropy" in report
    assert "Zero-support image fraction" in report
    assert "Supported-image log-likelihood span" in report
    assert "Quadrature-weight concentration" in report
    assert "Fold/unfold postprocessing seconds" in report
    assert "descriptive only; no stability threshold" in report


def test_cli_and_metadata_resolve_separate_netsky_folded_artifacts(
    tmp_path: Path,
) -> None:
    posterior = [tmp_path / f"posterior-seed{seed}.npz" for seed in range(3)]
    folded = [tmp_path / f"folded-seed{seed}.npz" for seed in range(3)]
    report_paths = [tmp_path / f"seed-{seed}" / "report.json" for seed in range(3)]
    args = analysis._parse_args(
        [
            *(str(path) for path in posterior),
            "--run-reports",
            *(str(path) for path in report_paths),
            "--folded-nested",
            *(str(path) for path in folded),
            "--output",
            str(tmp_path / "analysis.md"),
        ]
    )
    assert args.folded_nested == folded

    reports = []
    expected = []
    for seed, report_path in enumerate(report_paths):
        report_path.parent.mkdir()
        relative = Path(f"posterior-seed{seed}-folded-nested-diagnostics.npz")
        reports.append(
            {
                "config": {"blocking_scheme": analysis.NETSKY_BLOCKING_SCHEME},
                "results": {"folded_nested_diagnostics": {"path": str(relative)}},
            }
        )
        expected.append((report_path.parent / relative).resolve())

    assert (
        analysis._resolve_folded_paths(
            report_paths,
            reports,
            explicit_paths=None,
        )
        == expected
    )


def test_main_routes_netsky_physical_and_folded_artifacts_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior = [tmp_path / f"posterior-seed{seed}.npz" for seed in range(3)]
    folded = [tmp_path / f"folded-seed{seed}.npz" for seed in range(3)]
    reports = [tmp_path / f"report-seed{seed}.json" for seed in range(3)]
    for path in reports:
        path.write_text("{}")
    output = tmp_path / "analysis.md"
    monkeypatch.setattr(
        analysis,
        "_parse_args",
        lambda argv: SimpleNamespace(
            nested=posterior,
            folded_nested=folded,
            run_reports=reports,
            output=output,
            bootstrap_samples=10,
            window_size=20,
            random_seed=30,
            sanity_damaged=None,
            sanity_healthy=None,
        ),
    )
    report_calls = []

    def fake_reports(paths, nested_paths, *, folded_paths=None):
        report_calls.append((paths, nested_paths, folded_paths))
        return (
            [0, 1, 2],
            "bundle",
            analysis.NETSKY_BLOCKING_SCHEME,
            analysis.NETSKY_BLOCKS,
        )

    monkeypatch.setattr(analysis, "_reports", fake_reports)
    summary_calls = []

    def fake_netsky_summary(posterior_path, folded_path, report, **kwargs):
        summary_calls.append((posterior_path, folded_path, report, kwargs))
        return {"seed": kwargs["seed"]}

    monkeypatch.setattr(analysis, "_netsky_seed_summary", fake_netsky_summary)
    monkeypatch.setattr(
        analysis,
        "_seed_summary",
        lambda *args, **kwargs: pytest.fail("legacy summary path was used"),
    )
    monkeypatch.setattr(
        analysis,
        "_netsky_cross_seed_stability",
        lambda summaries: {"role": "descriptive_cross_seed_stability"},
    )
    rendered = {}

    def fake_render(summaries, **kwargs):
        rendered.update(kwargs)
        return "NETSKY analysis\n"

    monkeypatch.setattr(analysis, "_render_report", fake_render)

    analysis.main([])

    assert report_calls == [(reports, posterior, folded)]
    assert [(call[0], call[1]) for call in summary_calls] == list(
        zip(posterior, folded, strict=True)
    )
    assert rendered["netsky_stability"] == {"role": "descriptive_cross_seed_stability"}
    assert output.read_text() == "NETSKY analysis\n"
