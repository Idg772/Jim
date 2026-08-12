from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign import (
    prepare_remediation_campaign as remediation,
)

REVISION = "a" * 40
TREE_SHA256 = "b" * 64


def _write_valid_result(
    campaign: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
) -> None:
    injection_id = int(row["injection_id"])
    directory = common.result_dir(campaign, injection_id)
    directory.mkdir(parents=True)
    offsets = np.asarray([-3.0, -1.0, 1.0, 3.0]) * 1.0e-6
    samples = {
        name: np.asarray(float(row[name]) + offsets) for name in common.PARAMETERS
    }
    log_weights = np.log(np.full(4, 0.25))
    arrays = {
        **samples,
        "log_likelihood": np.asarray([-4.0, -3.0, -2.0, -1.0]),
        "log_weights": log_weights,
    }
    posterior_path = directory / "posterior.npz"
    common.atomic_savez_compressed(posterior_path, arrays)
    common.atomic_write_json(
        directory / "summary.json",
        {
            "schema_version": common.SCHEMA_VERSION,
            "campaign": manifest["config"]["campaign"],
            "config_sha256": manifest["config_sha256"],
            "injection_id": injection_id,
            "truth": {
                name: row[name]
                for name in (*common.PARAMETERS, *common.MARGINALIZED_PARAMETERS)
            },
            "seeds": {
                "noise": row["noise_seed"],
                "sampler": row["sampler_seed"],
            },
            "ranks": {
                name: common.posterior_rank(
                    samples[name], float(row[name]), log_weights
                )
                for name in common.PARAMETERS
            },
            "rank_method": {
                "comparison": "sample < truth",
                "resampled": False,
                "weighting": "original nested-sampling weights",
            },
            "posterior_samples": len(log_weights),
            "posterior_effective_sample_size": 4.0,
            "posterior": {
                "path": "posterior.npz",
                "sha256": common.file_sha256(posterior_path),
                "bytes": posterior_path.stat().st_size,
                "fields": list(arrays),
                "space": "prior",
                "weighting": "normalized nested-sampling log weights",
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
                "devices": [
                    {"id": device_id, "platform": "gpu"} for device_id in range(4)
                ],
            },
            "simulated_cpu": False,
        },
    )


def _source_campaign(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    campaign = tmp_path / "source"
    campaign.mkdir()
    rows = common.generate_catalogue(2, 4815)
    catalogue_path = campaign / "catalogue.csv"
    common.atomic_write_csv(catalogue_path, rows, common.CATALOGUE_FIELDS)
    psd_path = campaign / "inputs/psd/design.npz"
    psd_path.parent.mkdir(parents=True)
    psd_path.write_bytes(b"fixed test PSD\n")
    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config["campaign"] = "paper-fig2a-source"
    manifest: dict[str, Any] = {
        "schema_version": common.SCHEMA_VERSION,
        "created_at_utc": "2026-08-11T00:00:00+00:00",
        "master_seed": 4815,
        "n_injections": len(rows),
        "catalogue_size": len(rows),
        "selection": {
            "rule": "first catalogue entries",
            "start_inclusive": 0,
            "stop_exclusive": len(rows),
        },
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
        _write_valid_result(campaign, manifest, row)
    common.refresh_status(campaign, len(rows))
    return campaign, manifest


def test_preparer_freezes_a_paired_all_slow_time_m1_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remediation, "PAPER_PP_RECOVERIES", 2)
    source, source_manifest = _source_campaign(tmp_path)
    output = tmp_path / "remediation"

    manifest = remediation.prepare_remediation_campaign(
        source,
        output,
        scheme="all-slow-time",
        implementation_revision=REVISION,
        implementation_tree_sha256=TREE_SHA256,
    )

    assert common.load_manifest(output) == manifest
    changed_config_fields = {
        key
        for key in source_manifest["config"]
        if source_manifest["config"][key] != manifest["config"][key]
    }
    assert changed_config_fields == {"campaign", "paper_configuration", "blocks"}
    assert manifest["config"]["n_devices"] == 4
    assert manifest["config"]["num_gibbs_sweeps"] == 1
    assert manifest["config"]["blocks"] == [
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
        ["zenith", "azimuth"],
        ["psi"],
    ]
    pin = manifest["implementation_diagnostic"]
    assert pin["implementation_revision"] == REVISION
    assert pin["implementation_tree_sha256"] == TREE_SHA256
    assert pin["source_campaign_config_sha256"] == source_manifest["config_sha256"]
    assert manifest["blocking_remediation"]["acceptance_test"] == (
        "Fisher-combined exact KS p-value over all 15 sampled parameters, "
        "including q, must exceed 0.05"
    )
    assert (output / "catalogue.csv").read_bytes() == (
        source / "catalogue.csv"
    ).read_bytes()
    assert (output / "inputs/psd/design.npz").read_bytes() == (
        source / "inputs/psd/design.npz"
    ).read_bytes()
    assert not (output / "results").exists()
    assert [row["status"] for row in common.status_rows(output, 2)] == [
        "pending",
        "pending",
    ]


def test_preparer_rejects_a_source_result_that_only_weak_status_accepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remediation, "PAPER_PP_RECOVERIES", 2)
    source, _ = _source_campaign(tmp_path)
    summary_path = common.result_dir(source, 0) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["injection_id"] = 1
    common.atomic_write_json(summary_path, summary)
    assert common.status_rows(source, 2)[0]["status"] == "complete"
    output = tmp_path / "remediation"

    with pytest.raises(ValueError, match="wrong injection ID"):
        remediation.prepare_remediation_campaign(
            source,
            output,
            scheme="all-slow-time",
            implementation_revision=REVISION,
            implementation_tree_sha256=TREE_SHA256,
        )

    assert not output.exists()
