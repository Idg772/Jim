from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign import evaluate_blocking_diagnostic as evaluator
from benchmarks.injection_campaign.prepare_blocking_diagnostic import (
    DEFAULT_SAMPLER_REPLICATES,
    DEFAULT_SCHEME,
    DEFAULT_SOURCE_IDS,
    SCHEMES,
    SOURCE_CARRIER_TIME_ANCHOR,
    prepare_blocking_diagnostic,
    sampler_seed_for_replicate,
)

SOURCE_IDS = (2, 0)
REVISION = "a" * 40
TREE_SHA256 = "b" * 64


def _write_result(
    campaign: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    *,
    endpoint_parameters: tuple[str, ...] = (),
    high_z_parameters: tuple[str, ...] = (),
) -> None:
    injection_id = int(row["injection_id"])
    directory = common.result_dir(campaign, injection_id)
    directory.mkdir(parents=True, exist_ok=True)
    sample_count = 8
    centered_offsets = np.asarray([-4, -3, -2, -1, 1, 2, 3, 4], dtype=float)
    endpoint_offsets = np.arange(1, sample_count + 1, dtype=float)
    weights = (
        np.asarray([1.0e-6, *([(1.0 - 1.0e-6) / 7.0] * 7)])
        if high_z_parameters
        else np.full(sample_count, 1.0 / sample_count)
    )
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for parameter in common.PARAMETERS:
        if parameter in high_z_parameters:
            arrays[parameter] = float(row[parameter]) + np.asarray(
                [-1.0e-3, *(1.0e-3 + np.arange(7) * 1.0e-9)]
            )
        else:
            offsets = (
                endpoint_offsets
                if parameter in endpoint_parameters
                else centered_offsets
            )
            arrays[parameter] = float(row[parameter]) + offsets * 1.0e-6
    arrays["log_likelihood"] = np.linspace(-10.0, -8.0, sample_count)
    arrays["log_weights"] = np.log(weights)
    posterior_path = directory / "posterior.npz"
    common.atomic_savez_compressed(posterior_path, arrays)

    summary: dict[str, Any] = {
        "schema_version": common.SCHEMA_VERSION,
        "campaign": manifest["config"]["campaign"],
        "config_sha256": manifest["config_sha256"],
        "injection_id": injection_id,
        "truth": {
            name: row[name]
            for name in (*common.PARAMETERS, *common.MARGINALIZED_PARAMETERS)
        },
        "seeds": {"noise": row["noise_seed"], "sampler": row["sampler_seed"]},
        "ranks": {
            parameter: common.posterior_rank(
                arrays[parameter],
                float(row[parameter]),
                arrays["log_weights"],
            )
            for parameter in common.PARAMETERS
        },
        "rank_method": {
            "comparison": "sample < truth",
            "resampled": False,
            "weighting": "original nested-sampling weights",
        },
        "posterior_samples": sample_count,
        "posterior_effective_sample_size": float(1.0 / np.sum(weights**2)),
        "posterior": {
            "path": "posterior.npz",
            "sha256": common.file_sha256(posterior_path),
            "bytes": posterior_path.stat().st_size,
            "fields": list(arrays),
            "space": "prior",
            "weighting": "normalized nested-sampling log weights",
        },
        "diagnostics": {
            "n_iterations": 100,
            "n_likelihood_evaluations": 1000,
            "log_Z": -12.0,
            "log_Z_error": 0.2,
        },
        "timing_seconds": {
            "sample_call": 10.0,
            "total": 15.0,
            "sample_phases": {
                "likelihood_jit": 2.0,
                "sampler_kernel_jit": 3.0,
            },
            "paper_convention": {
                "likelihood_jit_seconds": 2.0,
                "sampler_jit_seconds": 3.0,
                "post_jit_sampling_seconds": 5.0,
            },
        },
        "devices": {
            "backend": "gpu",
            "requested_count": 4,
            "local_count": 4,
            "devices": [{"id": device_id, "platform": "gpu"} for device_id in range(4)],
        },
        "simulated_cpu": False,
    }
    implementation = manifest.get("implementation_diagnostic")
    if isinstance(implementation, dict):
        summary["implementation"] = {
            "label": implementation["implementation_label"],
            "revision": implementation["implementation_revision"],
            "tree_sha256": implementation["implementation_tree_sha256"],
            "root": "/workspace/candidate",
            "jimgw_module": "/workspace/candidate/src/jimgw/__init__.py",
        }
    common.atomic_write_json(directory / "summary.json", summary)


def _source_campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "corrected-anchor-source"
    campaign.mkdir(parents=True)
    rows = common.generate_catalogue(3, 8128)
    catalogue_path = campaign / "catalogue.csv"
    common.atomic_write_csv(catalogue_path, rows, common.CATALOGUE_FIELDS)
    psd_path = campaign / "inputs/psd/design.npz"
    psd_path.parent.mkdir(parents=True)
    psd_path.write_bytes(b"fixed PSD fixture\n")

    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config.update(
        {
            "campaign": "corrected-anchor-source",
            "paper_configuration": "Sharded",
            "carrier_time_anchor": SOURCE_CARRIER_TIME_ANCHOR,
            "sampler_scheduler": "fsm",
            "n_devices": 4,
            "num_gibbs_sweeps": 1,
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": common.SCHEMA_VERSION,
        "created_at_utc": "2026-08-11T00:00:00+00:00",
        "master_seed": 8128,
        "n_injections": len(rows),
        "catalogue_size": len(rows),
        "selection": {"start_inclusive": 0, "stop_exclusive": len(rows)},
        "config": config,
        "catalogue": {
            "path": "catalogue.csv",
            "sha256": common.file_sha256(catalogue_path),
            "bytes": catalogue_path.stat().st_size,
        },
        "psd": {
            "files": {
                "inputs/psd/design.npz": {
                    "sha256": common.file_sha256(psd_path),
                    "bytes": psd_path.stat().st_size,
                }
            }
        },
        "reproduction_scope": {
            "iid_prior_predictive_catalogue": True,
            "pp_calibration_eligible": True,
        },
        "storage_policy": {"fixture": True},
    }
    manifest["config_sha256"] = common.canonical_sha256(manifest)
    common.atomic_write_json(campaign / "manifest.json", manifest)
    for row in rows:
        _write_result(campaign, manifest, row)
    return campaign


def _prepared_campaign(
    tmp_path: Path, *, scheme: str = "sky-time"
) -> tuple[Path, Path, dict[str, Any]]:
    source = _source_campaign(tmp_path)
    campaign = tmp_path / f"diagnostic-{scheme}"
    manifest = prepare_blocking_diagnostic(
        source,
        campaign,
        implementation_revision=REVISION,
        implementation_tree_sha256=TREE_SHA256,
        scheme=scheme,
        source_ids=SOURCE_IDS,
        sampler_replicates=3,
    )
    return source, campaign, manifest


def _complete_campaign(
    tmp_path: Path, *, scheme: str = "sky-time"
) -> tuple[Path, Path, dict[str, Any]]:
    source, campaign, manifest = _prepared_campaign(tmp_path, scheme=scheme)
    for row in common.read_catalogue(campaign / "catalogue.csv"):
        _write_result(campaign, manifest, row, endpoint_parameters=("q",))
    return source, campaign, manifest


def _rewrite_manifest(campaign: Path, mutation: Any) -> None:
    path = campaign / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest.pop("config_sha256")
    manifest["config_sha256"] = common.canonical_sha256(manifest)
    common.atomic_write_json(path, manifest)


def test_preparer_builds_m1_schemes_with_identical_replicate_seeds(
    tmp_path: Path,
) -> None:
    source = _source_campaign(tmp_path)
    source_rows = common.read_catalogue(source / "catalogue.csv")
    expected_blocks = {
        "paper": common.DEFAULT_CONFIG["blocks"],
        "mass-time": [
            ["M_c", "q", "lambda_1", "lambda_2", "t_c"],
            *common.DEFAULT_CONFIG["blocks"][1:6],
        ],
        "all-intrinsic": [
            [
                "M_c",
                "q",
                "lambda_1",
                "lambda_2",
                "s1_mag",
                "s1_theta",
                "s1_phi",
                "s2_mag",
                "s2_theta",
                "s2_phi",
            ],
            *common.DEFAULT_CONFIG["blocks"][3:],
        ],
        "all-slow": [
            [
                "M_c",
                "q",
                "lambda_1",
                "lambda_2",
                "s1_mag",
                "s1_theta",
                "s1_phi",
                "s2_mag",
                "s2_theta",
                "s2_phi",
                "iota",
            ],
            *common.DEFAULT_CONFIG["blocks"][4:],
        ],
        "all-slow-time": [
            [
                "M_c",
                "q",
                "lambda_1",
                "lambda_2",
                "s1_mag",
                "s1_theta",
                "s1_phi",
                "s2_mag",
                "s2_theta",
                "s2_phi",
                "iota",
                "t_c",
            ],
            *common.DEFAULT_CONFIG["blocks"][4:6],
        ],
        "sky-time": [
            *common.DEFAULT_CONFIG["blocks"][:4],
            ["zenith", "azimuth", "t_c"],
            ["psi"],
        ],
        "fast-extrinsic": [
            *common.DEFAULT_CONFIG["blocks"][:4],
            ["zenith", "azimuth", "psi", "t_c"],
        ],
        "full-extrinsic": [
            *common.DEFAULT_CONFIG["blocks"][:3],
            ["iota", "zenith", "azimuth", "psi", "t_c"],
        ],
        "detector-time-fast-extrinsic": [
            *common.DEFAULT_CONFIG["blocks"][:4],
            ["zenith", "azimuth", "psi", "t_det"],
        ],
    }
    seeds_by_scheme: dict[str, list[int]] = {}

    for scheme in SCHEMES:
        campaign = tmp_path / scheme
        manifest = prepare_blocking_diagnostic(
            source,
            campaign,
            implementation_revision=REVISION,
            implementation_tree_sha256=TREE_SHA256,
            scheme=scheme,
            source_ids=SOURCE_IDS,
            sampler_replicates=3,
        )
        rows = common.read_catalogue(campaign / "catalogue.csv")
        mapping = manifest["catalogue"]["provenance"]["mapping"]
        seeds_by_scheme[scheme] = [row["sampler_seed"] for row in rows]

        assert manifest["n_injections"] == 6
        assert manifest["config"]["blocks"] == expected_blocks[scheme]
        assert manifest["config"]["n_devices"] == 4
        assert manifest["config"]["num_gibbs_sweeps"] == 1
        assert manifest["config"]["sampler_scheduler"] == "fsm"
        assert manifest["config"]["carrier_time_anchor"] == "imrphenomd"
        assert manifest["config"].get("time_sampling_frame", "geocentric") == (
            "H1" if scheme == "detector-time-fast-extrinsic" else "geocentric"
        )
        assert manifest["reproduction_scope"]["pp_calibration_eligible"] is False
        assert [entry["diagnostic_id"] for entry in mapping] == list(range(6))
        assert [entry["source_injection_id"] for entry in mapping] == [
            2,
            0,
            2,
            0,
            2,
            0,
        ]
        assert [entry["sampler_replicate"] for entry in mapping] == [0, 0, 1, 1, 2, 2]
        assert rows[0]["sampler_seed"] == source_rows[2]["sampler_seed"]
        assert rows[1]["sampler_seed"] == source_rows[0]["sampler_seed"]
        assert len({rows[index]["sampler_seed"] for index in (0, 2, 4)}) == 3
        assert len({rows[index]["sampler_seed"] for index in (1, 3, 5)}) == 3
        common.load_manifest(campaign)

    assert seeds_by_scheme["paper"] == seeds_by_scheme["sky-time"]
    assert seeds_by_scheme["paper"] == seeds_by_scheme["mass-time"]
    assert seeds_by_scheme["paper"] == seeds_by_scheme["all-intrinsic"]
    assert seeds_by_scheme["paper"] == seeds_by_scheme["all-slow"]
    assert seeds_by_scheme["paper"] == seeds_by_scheme["all-slow-time"]
    assert seeds_by_scheme["paper"] == seeds_by_scheme["fast-extrinsic"]
    assert seeds_by_scheme["paper"] == seeds_by_scheme["full-extrinsic"]
    assert seeds_by_scheme["paper"] == seeds_by_scheme["detector-time-fast-extrinsic"]
    assert DEFAULT_SOURCE_IDS == (10, 19, 40)
    assert DEFAULT_SAMPLER_REPLICATES == 3
    assert DEFAULT_SCHEME == "sky-time"
    assert SCHEMES[0] == "sky-time"


def test_seed_derivation_is_stable_uint32_and_preserves_replicate_zero() -> None:
    catalogue_hash = "c" * 64
    source_seed = 123456
    first = [
        sampler_seed_for_replicate(
            source_catalogue_sha256=catalogue_hash,
            source_injection_id=40,
            source_sampler_seed=source_seed,
            sampler_replicate=replicate,
        )
        for replicate in range(3)
    ]
    second = [
        sampler_seed_for_replicate(
            source_catalogue_sha256=catalogue_hash,
            source_injection_id=40,
            source_sampler_seed=source_seed,
            sampler_replicate=replicate,
        )
        for replicate in range(3)
    ]

    assert first == second
    assert first[0] == source_seed
    assert len(set(first)) == 3
    assert all(0 <= seed < 2**32 for seed in first)


def test_evaluator_reports_weighted_metrics_spread_and_q_exemption(
    tmp_path: Path,
) -> None:
    source, campaign, _ = _complete_campaign(tmp_path)

    report = evaluator.evaluate_blocking_diagnostic(campaign, source_campaign=source)
    output = evaluator.write_blocking_diagnostic_report(campaign, report)
    first = output.read_bytes()
    evaluator.write_blocking_diagnostic_report(campaign, report)

    assert output.read_bytes() == first
    assert json.loads(first) == report
    assert report["status"] == "complete"
    assert report["decision"] == "no-gross-mode-detected"
    assert report["gross_mode_gate"]["pass"] is True
    assert report["gross_mode_gate"]["parameters"] == ["ra", "dec", "t_c", "iota"]
    assert report["gross_mode_gate"]["q_exempt"] is True
    assert report["methodology"]["pp_calibration_claim"] is False
    assert report["parameter_summary"]["q"]["rank_minimum"] == 0.0
    assert report["parameter_summary"]["q"]["gross_mode_gate_eligible"] is False
    assert report["cross_replicate_spread"]["requested_source_groups"] == 2
    assert report["cross_replicate_spread"]["complete_source_groups"] == 2
    assert len(report["cross_replicate_spread"]["groups"]) == 2
    assert report["cases"][0]["parameters"]["q"]["gross_mode_gate"] == {
        "eligible": False,
        "minimum_two_sided_rank_tail_pass": None,
        "maximum_absolute_z_pass": None,
        "pass": None,
        "exemption": "q is reported but exempt from the blocking gross-mode gate",
    }
    for parameter in ("ra", "dec", "t_c", "iota", "q"):
        assert "weighted_median" in report["cases"][0]["parameters"][parameter]
        assert "z_weighted_median_std" in report["cases"][0]["parameters"][parameter]


def test_ra_gross_mode_rank_is_invariant_at_periodic_seam() -> None:
    truth = 1.0e-4
    samples = np.asarray(
        [
            2.0 * math.pi - 4.0e-4,
            2.0 * math.pi - 3.0e-4,
            2.0 * math.pi - 2.0e-4,
            2.0 * math.pi - 1.0e-4,
            3.0e-4,
            4.0e-4,
            5.0e-4,
            6.0e-4,
        ]
    )
    statistics = evaluator.weighted_posterior_statistics(
        samples,
        truth,
        np.log(np.full(samples.size, 1.0 / samples.size)),
        period=2.0 * math.pi,
    )

    assert statistics["rank"] == 0.0
    assert statistics["two_sided_rank_tail"] == 0.0
    assert statistics["gross_mode_rank"] == 0.5
    assert statistics["gross_mode_two_sided_rank_tail"] == 1.0
    assert statistics["gross_mode_rank_handling"] == "shortest-offset-about-truth"


def test_non_q_gate_detects_rank_endpoint_and_thresholds_are_configurable(
    tmp_path: Path,
) -> None:
    source, campaign, manifest = _complete_campaign(tmp_path)
    row = common.read_catalogue(campaign / "catalogue.csv")[0]
    _write_result(
        campaign,
        manifest,
        row,
        endpoint_parameters=("q", "ra"),
    )

    failed = evaluator.evaluate_blocking_diagnostic(campaign, source_campaign=source)
    relaxed = evaluator.evaluate_blocking_diagnostic(
        campaign,
        source_campaign=source,
        minimum_two_sided_rank_tail=0.0,
        maximum_absolute_z=8.0,
    )

    assert failed["decision"] == "gross-mode-detected"
    assert failed["gross_mode_gate"]["failed_diagnostic_ids"] == [0]
    assert (
        failed["cases"][0]["parameters"]["ra"]["gross_mode_gate"][
            "minimum_two_sided_rank_tail_pass"
        ]
        is False
    )
    assert relaxed["decision"] == "no-gross-mode-detected"
    assert relaxed["gross_mode_gate"]["pass"] is True

    _write_result(
        campaign,
        manifest,
        row,
        endpoint_parameters=("q",),
        high_z_parameters=("dec",),
    )
    high_z = evaluator.evaluate_blocking_diagnostic(campaign, source_campaign=source)
    high_z_dec = high_z["cases"][0]["parameters"]["dec"]

    assert high_z_dec["gross_mode_gate"]["minimum_two_sided_rank_tail_pass"] is True
    assert high_z_dec["gross_mode_gate"]["maximum_absolute_z_pass"] is False
    assert abs(high_z_dec["z_weighted_median_std"]) > 8.0


def test_allow_partial_marks_missing_replicates_and_never_passes_complete_gate(
    tmp_path: Path,
) -> None:
    source, campaign, _ = _complete_campaign(tmp_path)
    missing = common.result_dir(campaign, 5)
    for path in missing.iterdir():
        path.unlink()
    missing.rmdir()

    with pytest.raises(ValueError, match="requires every matched replicate"):
        evaluator.evaluate_blocking_diagnostic(campaign, source_campaign=source)
    report = evaluator.evaluate_blocking_diagnostic(
        campaign, source_campaign=source, allow_partial=True
    )

    assert report["status"] == "partial"
    assert report["decision"] == "partial-no-gross-mode-detected"
    assert report["selection"]["missing_diagnostic_ids"] == [5]
    assert report["gross_mode_gate"]["completed_cases_pass"] is True
    assert report["gross_mode_gate"]["pass"] is False
    assert report["cross_replicate_spread"]["complete_source_groups"] == 1

    exit_code = evaluator.run_evaluation(
        argparse.Namespace(
            campaign_dir=campaign,
            source_campaign=source,
            allow_partial=True,
            minimum_two_sided_rank_tail=(evaluator.DEFAULT_MINIMUM_TWO_SIDED_RANK_TAIL),
            maximum_absolute_z=evaluator.DEFAULT_MAXIMUM_ABSOLUTE_Z,
            report=tmp_path / "partial-report.json",
        )
    )
    assert exit_code == 1


def test_evaluator_rejects_source_result_and_manifest_provenance_tampering(
    tmp_path: Path,
) -> None:
    source, campaign, _ = _complete_campaign(tmp_path)
    source_summary_path = common.result_dir(source, SOURCE_IDS[0]) / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_summary["harmless_but_hash_changing_field"] = True
    common.atomic_write_json(source_summary_path, source_summary)

    with pytest.raises(ValueError, match="source_summary_sha256 mismatch"):
        evaluator.evaluate_blocking_diagnostic(campaign, source_campaign=source)

    source, campaign, _ = _complete_campaign(tmp_path / "fresh")

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["implementation_diagnostic"]["changed_variables"]["num_gibbs_sweeps"][
            "diagnostic"
        ] = 2

    _rewrite_manifest(campaign, mutate)
    with pytest.raises(ValueError, match="changed_variables are inconsistent"):
        evaluator.evaluate_blocking_diagnostic(campaign, source_campaign=source)


def test_preparer_rejects_non_corrected_anchor_source(tmp_path: Path) -> None:
    source = _source_campaign(tmp_path)

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["config"]["carrier_time_anchor"] = "nrtidal-merger"

    _rewrite_manifest(source, mutate)
    with pytest.raises(ValueError, match="not a corrected-anchor D=4, M=1 FSM"):
        prepare_blocking_diagnostic(
            source,
            tmp_path / "invalid",
            implementation_revision=REVISION,
            implementation_tree_sha256=TREE_SHA256,
            source_ids=SOURCE_IDS,
        )
