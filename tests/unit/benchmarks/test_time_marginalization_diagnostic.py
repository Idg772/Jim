from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign import (
    evaluate_time_marginalization_diagnostic as evaluator,
)
from benchmarks.injection_campaign import (
    prepare_time_marginalization_diagnostic as preparer,
)

REVISION = "a" * 40
TREE_SHA256 = "b" * 64


def _write_weighted_result(
    campaign: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    parameters: tuple[str, ...],
    *,
    below_counts: dict[str, int],
    diagnostic: bool,
) -> None:
    injection_id = int(row["injection_id"])
    directory = common.result_dir(campaign, injection_id)
    directory.mkdir(parents=True)
    sample_count = 4
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for parameter in parameters:
        below = below_counts.get(parameter, 2)
        offsets = np.concatenate(
            (
                -np.arange(below, 0, -1, dtype=float),
                np.arange(1, sample_count - below + 1, dtype=float),
            )
        )
        arrays[parameter] = float(row[parameter]) + offsets * 1.0e-8
    arrays["log_likelihood"] = np.asarray([-4.0, -3.0, -2.0, -1.0])
    arrays["log_weights"] = np.log(np.full(sample_count, 0.25))
    posterior_path = directory / "posterior.npz"
    common.atomic_savez_compressed(posterior_path, arrays)
    ranks = {
        parameter: common.posterior_rank(
            arrays[parameter], float(row[parameter]), arrays["log_weights"]
        )
        for parameter in parameters
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
            "weighting": "original nested-sampling weights",
            "comparison": "sample < truth",
            "resampled": False,
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
        "simulated_cpu": False,
    }
    if diagnostic:
        summary["parameter_treatment"] = {
            "sampled": list(preparer.DIAGNOSTIC_PARAMETERS),
            "marginalized": list(preparer.DIAGNOSTIC_MARGINALIZED_PARAMETERS),
        }
        summary["implementation"] = {
            "label": "candidate",
            "revision": REVISION,
            "tree_sha256": TREE_SHA256,
            "root": "/workspace/candidate",
            "jimgw_module": "/workspace/candidate/src/jimgw/__init__.py",
        }
        summary["devices"] = {
            "backend": "gpu",
            "requested_count": 4,
            "local_count": 4,
            "devices": [
                {"id": device_id, "platform": "gpu"} for device_id in range(4)
            ],
        }
    common.atomic_write_json(directory / "summary.json", summary)


def _source_campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "paper-sharded-pp"
    campaign.mkdir()
    rows = common.generate_catalogue(100, 701)
    catalogue_path = campaign / "catalogue.csv"
    common.atomic_write_csv(catalogue_path, rows, common.CATALOGUE_FIELDS)
    psd_path = campaign / "inputs/psd/design.npz"
    psd_path.parent.mkdir(parents=True)
    psd_path.write_bytes(b"immutable design PSD fixture\n")
    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config.update(
        {
            "campaign": "paper-sharded-pp",
            "paper_configuration": "Sharded",
            "n_devices": 4,
            "num_gibbs_sweeps": 1,
            "sampler_scheduler": "fsm",
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": common.SCHEMA_VERSION,
        "created_at_utc": "2026-08-10T00:00:00+00:00",
        "master_seed": 701,
        "n_injections": 100,
        "catalogue_size": 100,
        "selection": {"start_inclusive": 0, "stop_exclusive": 100},
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
    for source_id in preparer.DEFAULT_SOURCE_IDS:
        row = rows[source_id]
        _write_weighted_result(
            campaign,
            manifest,
            row,
            common.PARAMETERS,
            below_counts={"q": source_id % 5, "ra": (source_id + 1) % 5},
            diagnostic=False,
        )
    return campaign


def _prepared_diagnostic(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source = _source_campaign(tmp_path)
    campaign = tmp_path / "appendix-a-diagnostic"
    manifest = preparer.prepare_time_marginalization_diagnostic(
        source,
        campaign,
        implementation_revision=REVISION,
        implementation_tree_sha256=TREE_SHA256,
    )
    return source, campaign, manifest


def _complete_diagnostic(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source, campaign, manifest = _prepared_diagnostic(tmp_path)
    rows = common.read_catalogue(campaign / "catalogue.csv")
    for diagnostic_id, row in enumerate(rows):
        _write_weighted_result(
            campaign,
            manifest,
            row,
            preparer.DIAGNOSTIC_PARAMETERS,
            below_counts={
                "q": 2,
                "ra": 2,
                "d_L": diagnostic_id,
            },
            diagnostic=True,
        )
    return source, campaign, manifest


def test_preparer_freezes_exact_rows_and_only_changes_nuisance_treatment(
    tmp_path: Path,
) -> None:
    source, campaign, manifest = _prepared_diagnostic(tmp_path)
    source_manifest = common.load_manifest(source)
    source_rows = common.read_catalogue(source / "catalogue.csv")
    rows = common.read_catalogue(campaign / "catalogue.csv")

    assert manifest["selection"]["source_injection_ids"] == [97, 12, 41, 32]
    assert len(rows) == 4
    for diagnostic_id, source_id in enumerate(preparer.DEFAULT_SOURCE_IDS):
        assert rows[diagnostic_id] == {
            **source_rows[source_id],
            "injection_id": diagnostic_id,
        }
    assert (campaign / "inputs/psd/design.npz").read_bytes() == (
        source / "inputs/psd/design.npz"
    ).read_bytes()

    config = manifest["config"]
    source_config = source_manifest["config"]
    assert config["n_devices"] == source_config["n_devices"] == 4
    assert config["num_gibbs_sweeps"] == source_config["num_gibbs_sweeps"] == 1
    assert config["f_max_hz"] == source_config["f_max_hz"] == 2048.0
    assert config["likelihood_f_max_hz"] == 2048.0 - 1.0 / 128.0
    assert config["time_marginalization"] == {
        "tc_range_seconds": [-0.1, 0.1],
        "upsample_factor": 32,
    }
    assert config["distance_marginalization"] is False
    assert config["blocks"][:-1] == source_config["blocks"][:-1]
    assert config["blocks"][-1] == ["d_L"]
    assert source_config["blocks"][-1] == ["t_c"]

    changed = manifest["implementation_diagnostic"]["changed_variables"]
    assert changed["sampled_nuisance"] == {
        "source": "t_c",
        "diagnostic": "d_L",
    }
    assert changed["injection_f_max_hz"] == {
        "source": 2048.0,
        "diagnostic": 2048.0,
    }
    assert "Nyquist" in changed["recovery_likelihood_f_max_hz"]["reason"]
    assert manifest["reproduction_scope"]["pp_calibration_eligible"] is False
    assert (
        manifest["appendix_a_diagnostic"][
            "population_calibration_claim_permitted"
        ]
        is False
    )


def test_evaluator_recomputes_14_paired_weighted_ranks_and_reports_dl_separately(
    tmp_path: Path,
) -> None:
    source, campaign, _ = _complete_diagnostic(tmp_path)

    report = evaluator.evaluate_time_marginalization_diagnostic(
        campaign, source_campaign=source
    )
    json_path, csv_path = evaluator.write_time_marginalization_diagnostic_report(
        campaign, report
    )

    assert report["status"] == "complete"
    assert report["selection"]["common_rank_parameters"] == list(
        preparer.COMMON_PARAMETERS
    )
    assert len(preparer.COMMON_PARAMETERS) == 14
    assert report["aggregate_common_tail_severity"]["parameter_comparisons"] == 56
    assert report["methodology"]["population_calibration_tests_performed"] == []
    assert report["methodology"]["population_calibration_claim_permitted"] is False
    assert report["phase_gauge"] == {
        "corrected_parameters": ["s1_phi", "s2_phi"],
        "catalogue_coordinates": "alpha_i = catalogue spin azimuth",
        "sampled_coordinates": "beta_i = (alpha_i + phase_c) mod 2 pi",
        "applied_to": ["source", "diagnostic"],
        "stored_raw_alpha_ranks_used_for_comparison": False,
        "stored_raw_alpha_ranks_recomputed_for_artifact_validation": True,
        "coordinate_transform_absolute_jacobian": 1.0,
    }
    assert len(report["sampled_distance_ranks"]) == 4
    for case in report["cases"]:
        assert set(case["common_parameter_comparisons"]) == set(
            preparer.COMMON_PARAMETERS
        )
        context = case["sampled_nuisance_context"]
        assert context["cross_parameter_comparison_permitted"] is False
        assert 0.0 <= context["diagnostic_d_L"]["rank"] <= 1.0

    first = report["cases"][0]
    source_correction = first["source_inputs"]["phase_gauge_corrections"]["s1_phi"]
    diagnostic_correction = first["diagnostic_inputs"][
        "phase_gauge_corrections"
    ]["s1_phi"]
    expected_beta = np.mod(
        source_correction["catalogue_truth_alpha"]
        + source_correction["phase_c_truth"],
        2.0 * np.pi,
    )
    assert source_correction["sampled_truth_beta"] == pytest.approx(expected_beta)
    assert diagnostic_correction["sampled_truth_beta"] == pytest.approx(expected_beta)
    assert first["common_parameter_comparisons"]["s1_phi"]["source"][
        "rank"
    ] == pytest.approx(source_correction["gauge_corrected_rank"])
    assert first["common_parameter_comparisons"]["s1_phi"]["diagnostic"][
        "rank"
    ] == pytest.approx(diagnostic_correction["gauge_corrected_rank"])
    assert source_correction["gauge_corrected_rank"] != pytest.approx(
        source_correction["stored_catalogue_truth_rank"]
    )

    serialized = json.dumps(report).lower()
    assert '"ks"' not in serialized
    assert "fisher" not in serialized
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()
    evaluator.write_time_marginalization_diagnostic_report(campaign, report)
    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * (14 + 2)
    distance_rows = [row for row in rows if row["parameter"] == "d_L"]
    assert len(distance_rows) == 4
    assert all(row["source_rank"] == "" for row in distance_rows)


def test_evaluator_rejects_a_stored_rank_not_derived_from_direct_weights(
    tmp_path: Path,
) -> None:
    source, campaign, _ = _complete_diagnostic(tmp_path)
    summary_path = common.result_dir(campaign, 0) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ranks"]["d_L"] = 0.987654321
    common.atomic_write_json(summary_path, summary)

    with pytest.raises(ValueError, match="weighted rank mismatch for d_L"):
        evaluator.evaluate_time_marginalization_diagnostic(
            campaign, source_campaign=source
        )


def test_workspace_package_lists_new_diagnostic_modules_and_test() -> None:
    repository = Path(__file__).resolve().parents[3]
    script = (
        repository / "benchmarks/device_parallel_nss/runpod/package_workspace.sh"
    ).read_text(encoding="utf-8")
    for relative in (
        "benchmarks/injection_campaign/prepare_time_marginalization_diagnostic.py",
        "benchmarks/injection_campaign/evaluate_time_marginalization_diagnostic.py",
        "tests/unit/benchmarks/test_time_marginalization_diagnostic.py",
    ):
        assert f'"{relative}"' in script
