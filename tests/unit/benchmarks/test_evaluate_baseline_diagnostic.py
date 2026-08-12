from __future__ import annotations

import copy
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign import evaluate_baseline_diagnostic as evaluator
from benchmarks.injection_campaign.prepare_baseline_diagnostic import (
    PAPER_BASELINE_REVISION,
    PAPER_BASELINE_TREE_SHA256,
    prepare_baseline_diagnostic,
)

SOURCE_IDS = (4, 1, 3, 0, 2)


def _write_result(
    campaign: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    *,
    device_count: int,
    selected_below_counts: dict[str, int],
    baseline: bool,
) -> None:
    injection_id = int(row["injection_id"])
    directory = common.result_dir(campaign, injection_id)
    directory.mkdir(parents=True)
    sample_count = 4
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for parameter in common.PARAMETERS:
        below_count = selected_below_counts.get(parameter, 2)
        offsets = np.concatenate(
            (
                -np.arange(below_count, 0, -1, dtype=float),
                np.arange(1, sample_count - below_count + 1, dtype=float),
            )
        )
        arrays[parameter] = float(row[parameter]) + offsets * 1.0e-6
    arrays["log_likelihood"] = np.asarray([-4.0, -3.0, -2.0, -1.0])
    arrays["log_weights"] = np.log(np.full(sample_count, 0.25))
    posterior_path = directory / "posterior.npz"
    common.atomic_savez_compressed(posterior_path, arrays)

    ranks = {
        parameter: common.posterior_rank(
            arrays[parameter],
            float(row[parameter]),
            arrays["log_weights"],
        )
        for parameter in common.PARAMETERS
    }
    timing: dict[str, Any] = {
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
    }
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
        "ranks": ranks,
        "rank_method": {
            "comparison": "sample < truth",
            "resampled": False,
            "weighting": "original nested-sampling weights",
        },
        "posterior_samples": sample_count,
        "posterior_effective_sample_size": 4.0,
        "posterior": {
            "path": "posterior.npz",
            "sha256": common.file_sha256(posterior_path),
            "bytes": posterior_path.stat().st_size,
            "fields": list(arrays),
            "space": "prior",
            "weighting": "normalized nested-sampling log weights",
        },
        "timing_seconds": timing,
        "devices": {
            "backend": "gpu",
            "requested_count": device_count,
            "local_count": device_count,
            "devices": [
                {"id": device_id, "platform": "gpu"}
                for device_id in range(device_count)
            ],
        },
        "simulated_cpu": False,
    }
    if baseline:
        timing.update(
            {
                "sample_phases": None,
                "paper_convention": None,
                "paper_convention_unavailable_reason": (
                    "Pinned baseline has no split JIT phase instrumentation."
                ),
            }
        )
        summary["implementation"] = {
            "label": "paper-baseline",
            "revision": PAPER_BASELINE_REVISION,
            "tree_sha256": PAPER_BASELINE_TREE_SHA256,
            "root": "/workspace/paper-baseline",
            "jimgw_module": "/workspace/paper-baseline/src/jimgw/__init__.py",
        }
    common.atomic_write_json(directory / "summary.json", summary)


def _source_campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "source-fsm"
    campaign.mkdir()
    rows = common.generate_catalogue(5, 8128)
    catalogue_path = campaign / "catalogue.csv"
    common.atomic_write_csv(catalogue_path, rows, common.CATALOGUE_FIELDS)
    psd_path = campaign / "inputs/psd/design.npz"
    psd_path.parent.mkdir(parents=True)
    psd_path.write_bytes(b"fixed PSD fixture\n")

    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config.update(
        {
            "campaign": "source-fsm",
            "paper_configuration": "Sharded",
            "n_devices": 4,
            "num_gibbs_sweeps": 1,
        }
    )
    config.pop("sampler_scheduler")
    manifest: dict[str, Any] = {
        "schema_version": common.SCHEMA_VERSION,
        "created_at_utc": "2026-08-10T00:00:00+00:00",
        "master_seed": 8128,
        "n_injections": 5,
        "catalogue_size": 5,
        "selection": {"start_inclusive": 0, "stop_exclusive": 5},
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

    source_below = (
        {"q": 0, "ra": 1, "t_c": 4},
        {"q": 0, "ra": 2, "t_c": 1},
        {"q": 1, "ra": 4, "t_c": 0},
        {"q": 4, "ra": 0, "t_c": 2},
        {"q": 1, "ra": 3, "t_c": 0},
    )
    for row, below_counts in zip(rows, source_below, strict=True):
        _write_result(
            campaign,
            manifest,
            row,
            device_count=4,
            selected_below_counts=below_counts,
            baseline=False,
        )
    return campaign


def _completed_diagnostic(tmp_path: Path) -> tuple[Path, Path]:
    source = _source_campaign(tmp_path)
    campaign = tmp_path / "baseline-diagnostic"
    manifest = prepare_baseline_diagnostic(
        source,
        campaign,
        source_ids=SOURCE_IDS,
        n_devices=1,
        num_gibbs_sweeps=3,
    )
    rows = common.read_catalogue(campaign / "catalogue.csv")
    for row in rows:
        _write_result(
            campaign,
            manifest,
            row,
            device_count=1,
            selected_below_counts={"q": 2, "ra": 2, "t_c": 2},
            baseline=True,
        )
    return source, campaign


def _rewrite_manifest(campaign: Path, mutation: Any) -> None:
    path = campaign / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest.pop("config_sha256")
    manifest["config_sha256"] = common.canonical_sha256(manifest)
    common.atomic_write_json(path, manifest)


def test_tail_metrics_preserves_endpoint_tail_and_only_floors_logarithm() -> None:
    endpoint = evaluator.tail_metrics(0.0)
    roundoff_endpoint = evaluator.tail_metrics(1.0 + 2.0e-16)
    centered = evaluator.tail_metrics(0.5)

    assert endpoint["rank"] == 0.0
    assert endpoint["two_sided_tail_probability"] == 0.0
    assert endpoint["log_input_floored"] is True
    assert np.isfinite(endpoint["tail_severity_log10"])
    assert roundoff_endpoint["rank"] == 1.0
    assert roundoff_endpoint["two_sided_tail_probability"] == 0.0
    assert roundoff_endpoint["log_input_floored"] is True
    assert centered == {
        "rank": 0.5,
        "two_sided_tail_probability": 1.0,
        "tail_severity_log10": 0.0,
        "log_input_floored": False,
    }


def test_evaluator_validates_and_writes_deterministic_json_and_csv(
    tmp_path: Path,
) -> None:
    source, campaign = _completed_diagnostic(tmp_path)

    report = evaluator.evaluate_baseline_diagnostic(campaign, source_campaign=source)
    json_path, csv_path = evaluator.write_baseline_diagnostic_report(campaign, report)
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()
    evaluator.write_baseline_diagnostic_report(campaign, report)

    assert json_path.parent == campaign / "diagnostic"
    assert csv_path.parent == campaign / "diagnostic"
    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv
    assert json.loads(first_json) == report
    assert report["status"] == "complete"
    assert report["selection"]["requested_source_injection_ids"] == list(SOURCE_IDS)
    assert report["selection"]["completed_diagnostic_ids"] == [0, 1, 2, 3, 4]
    assert report["selection"]["missing_diagnostic_ids"] == []
    assert report["selection"]["aborted"] is False
    assert report["aggregate"]["direction"] == "less-tail-pathological"
    assert report["aggregate"]["parameter_comparisons"] == 15
    assert report["attribution"]["strength"] == "directional-only"
    assert report["attribution"]["confounded_changes"]["n_devices_D"] == {
        "source": 4,
        "baseline": 1,
    }
    assert report["attribution"]["confounded_changes"]["num_gibbs_sweeps_M"] == {
        "source": 1,
        "baseline": 3,
    }
    assert "D=1 and M=3" in report["attribution"]["caveat"]
    assert "D=4 and M=1" in report["attribution"]["caveat"]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15
    assert [row["parameter"] for row in rows[:3]] == ["q", "ra", "t_c"]
    assert rows[0]["diagnostic_id"] == "0"
    assert rows[0]["source_injection_id"] == str(SOURCE_IDS[0])


@pytest.mark.parametrize("mutation", ["order", "summary-hash", "source-rank"])
def test_evaluator_rejects_unfaithful_source_mapping(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, campaign = _completed_diagnostic(tmp_path)

    def mutate(manifest: dict[str, Any]) -> None:
        mapping = manifest["catalogue"]["provenance"]["mapping"]
        if mutation == "order":
            mapping[0], mapping[1] = mapping[1], mapping[0]
        elif mutation == "summary-hash":
            mapping[0]["source_summary_sha256"] = "0" * 64
        else:
            mapping[0]["source_ranks"]["q"] = 0.125

    _rewrite_manifest(campaign, mutate)
    expected = {
        "order": "source mapping is not in diagnostic selection order",
        "summary-hash": "source summary hash mismatch",
        "source-rank": "recorded source rank mismatch",
    }[mutation]
    with pytest.raises(ValueError, match=expected):
        evaluator.evaluate_baseline_diagnostic(campaign, source_campaign=source)


def test_evaluator_requires_all_five_complete_results(tmp_path: Path) -> None:
    source, campaign = _completed_diagnostic(tmp_path)
    shutil.rmtree(common.result_dir(campaign, 4))

    with pytest.raises(ValueError, match="all 5 complete validated posterior results"):
        evaluator.evaluate_baseline_diagnostic(campaign, source_campaign=source)


def test_allow_partial_records_aborted_selection_and_aggregates_completed_cases(
    tmp_path: Path,
) -> None:
    source, campaign = _completed_diagnostic(tmp_path)
    shutil.rmtree(common.result_dir(campaign, 2))

    report = evaluator.evaluate_baseline_diagnostic(
        campaign,
        source_campaign=source,
        allow_partial=True,
    )
    _, csv_path = evaluator.write_baseline_diagnostic_report(campaign, report)

    assert report["status"] == "partial-aborted"
    assert report["selection"] == {
        "requested_cases": 5,
        "completed_cases": 4,
        "missing_cases": 1,
        "complete": False,
        "aborted": True,
        "requested_diagnostic_ids": [0, 1, 2, 3, 4],
        "requested_source_injection_ids": list(SOURCE_IDS),
        "completed_diagnostic_ids": [0, 1, 3, 4],
        "missing_diagnostic_ids": [2],
        "completed_source_injection_ids": [4, 1, 0, 2],
        "missing_source_injection_ids": [3],
        "incomplete_artifact_diagnostic_ids": [],
        "rank_parameters": ["q", "ra", "t_c"],
    }
    assert [case["diagnostic_id"] for case in report["cases"]] == [0, 1, 3, 4]
    assert report["missing_cases"] == [
        {"diagnostic_id": 2, "source_injection_id": 3, "status": "not-run"}
    ]
    assert report["aggregate"]["parameter_comparisons"] == 12
    assert report["aggregate"]["completed_cases_only"] is True
    assert report["attribution"]["strength"] == "partial-directional-only"
    assert report["attribution"]["assessment"].startswith("partial-aborted-")
    assert "4 of 5 cases complete" in report["attribution"]["partial_run_caveat"]
    assert "D=1 and M=3" in report["attribution"]["caveat"]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 12


def test_allow_partial_requires_at_least_one_complete_result(tmp_path: Path) -> None:
    source, campaign = _completed_diagnostic(tmp_path)
    shutil.rmtree(campaign / "results")

    with pytest.raises(ValueError, match="requires at least one complete validated"):
        evaluator.evaluate_baseline_diagnostic(
            campaign,
            source_campaign=source,
            allow_partial=True,
        )


def test_allow_partial_still_revalidates_every_completed_result(
    tmp_path: Path,
) -> None:
    source, campaign = _completed_diagnostic(tmp_path)
    shutil.rmtree(common.result_dir(campaign, 2))
    summary_path = common.result_dir(campaign, 1) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ranks"]["ra"] = 0.75
    common.atomic_write_json(summary_path, summary)

    with pytest.raises(ValueError, match="posterior rank mismatch for ra"):
        evaluator.evaluate_baseline_diagnostic(
            campaign,
            source_campaign=source,
            allow_partial=True,
        )


def test_evaluator_revalidates_weighted_baseline_ranks(tmp_path: Path) -> None:
    source, campaign = _completed_diagnostic(tmp_path)
    summary_path = common.result_dir(campaign, 2) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ranks"]["q"] = 0.75
    common.atomic_write_json(summary_path, summary)

    with pytest.raises(ValueError, match="posterior rank mismatch for q"):
        evaluator.evaluate_baseline_diagnostic(campaign, source_campaign=source)


def test_evaluator_enforces_d1_m3_vs_d4_m1_caveat_provenance(
    tmp_path: Path,
) -> None:
    source, campaign = _completed_diagnostic(tmp_path)

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["baseline_diagnostic"]["changed_variables"]["num_gibbs_sweeps"] = {
            "source": 1,
            "diagnostic": 2,
        }

    _rewrite_manifest(campaign, mutate)
    with pytest.raises(ValueError, match="changed_variables.num_gibbs_sweeps"):
        evaluator.evaluate_baseline_diagnostic(campaign, source_campaign=source)
