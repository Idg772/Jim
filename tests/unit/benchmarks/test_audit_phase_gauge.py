from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.injection_campaign import audit_phase_gauge as audit
from benchmarks.injection_campaign import common


def _write_result(
    campaign: Path,
    manifest: dict[str, Any],
    row: dict[str, Any],
    samples: dict[str, np.ndarray[Any, Any]],
) -> None:
    injection_id = int(row["injection_id"])
    directory = common.result_dir(campaign, injection_id)
    directory.mkdir(parents=True)
    arrays = {
        **samples,
        "log_weights": np.log(np.full(4, 0.25)),
    }
    posterior_path = directory / "posterior.npz"
    common.atomic_savez_compressed(posterior_path, arrays)
    ranks = {
        parameter: common.posterior_rank(
            arrays[parameter],
            float(row[parameter]),
            arrays["log_weights"],
        )
        for parameter in audit.SPIN_AZIMUTHS
    }
    summary = {
        "schema_version": common.SCHEMA_VERSION,
        "campaign": manifest["config"]["campaign"],
        "config_sha256": manifest["config_sha256"],
        "injection_id": injection_id,
        "truth": {
            parameter: row[parameter]
            for parameter in (*common.PARAMETERS, *common.MARGINALIZED_PARAMETERS)
        },
        "seeds": {
            "noise": row["noise_seed"],
            "sampler": row["sampler_seed"],
        },
        "ranks": ranks,
        "rank_method": {
            "comparison": "sample < truth",
            "resampled": False,
            "weighting": "original nested-sampling weights",
        },
        "posterior_samples": 4,
        "posterior": {
            "path": "posterior.npz",
            "sha256": common.file_sha256(posterior_path),
            "bytes": posterior_path.stat().st_size,
            "fields": list(arrays),
            "space": "prior",
            "weighting": "normalized nested-sampling log weights",
        },
    }
    common.atomic_write_json(directory / "summary.json", summary)


def _campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    rows = common.generate_catalogue(2, 8128)
    rows[0].update({"phase_c": 0.5, "s1_phi": 0.2, "s2_phi": 1.0})
    rows[1].update({"phase_c": 1.0, "s1_phi": 5.8, "s2_phi": 5.7})

    catalogue_path = campaign / "catalogue.csv"
    common.atomic_write_csv(catalogue_path, rows, common.CATALOGUE_FIELDS)
    psd_path = campaign / "inputs/psd/design.npz"
    psd_path.parent.mkdir(parents=True)
    psd_path.write_bytes(b"phase-gauge audit PSD fixture\n")

    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config.update(
        {
            "campaign": "phase-gauge-audit-fixture",
            "paper_configuration": "Sharded",
            "n_devices": 4,
            "num_gibbs_sweeps": 1,
            "phase_marginalization": True,
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": common.SCHEMA_VERSION,
        "created_at_utc": "2026-08-10T00:00:00+00:00",
        "master_seed": 8128,
        "n_injections": 2,
        "catalogue_size": 2,
        "selection": {"start_inclusive": 0, "stop_exclusive": 2},
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
    }
    manifest["config_sha256"] = common.canonical_sha256(manifest)
    common.atomic_write_json(campaign / "manifest.json", manifest)

    _write_result(
        campaign,
        manifest,
        rows[0],
        {
            "s1_phi": np.asarray([0.1, 0.3, 0.6, 0.8]),
            "s2_phi": np.asarray([0.5, 1.2, 1.4, 1.8]),
        },
    )
    _write_result(
        campaign,
        manifest,
        rows[1],
        {
            "s1_phi": np.asarray([0.1, 0.4, 0.6, 6.0]),
            "s2_phi": np.asarray([0.1, 0.3, 0.5, 5.8]),
        },
    )
    return campaign


def _source_hashes(campaign: Path) -> dict[Path, str]:
    paths = [campaign / "manifest.json", campaign / "catalogue.csv"]
    paths.extend(sorted((campaign / "results").glob("injection-*/*")))
    return {path: common.file_sha256(path) for path in paths}


def _rewrite_summary(campaign: Path, injection_id: int, mutation: Any) -> None:
    path = common.result_dir(campaign, injection_id) / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    mutation(summary)
    common.atomic_write_json(path, summary)


def _rewrite_manifest(campaign: Path, mutation: Any) -> None:
    path = campaign / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutation(manifest)
    manifest.pop("config_sha256")
    manifest["config_sha256"] = common.canonical_sha256(manifest)
    common.atomic_write_json(path, manifest)


def test_audit_recomputes_gauge_ranks_and_writes_deterministic_reports(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    hashes_before = _source_hashes(campaign)

    report = audit.audit_phase_gauge(campaign)

    assert report["status"] == "complete"
    assert report["campaign"]["device_shards_D"] == 4
    assert report["campaign"]["gibbs_sweeps_M"] == 1
    assert report["verification"]["stored_catalogue_truth_ranks_reproduced"] is True
    assert report["phase_gauge"]["coordinate_transform_absolute_jacobian"] == 1.0
    assert [case["injection_id"] for case in report["cases"]] == [0, 1]

    first = report["cases"][0]["parameters"]
    assert first["s1_phi"]["stored_rank"] == pytest.approx(0.25)
    assert first["s1_phi"]["gauge_truth"] == pytest.approx(0.7)
    assert first["s1_phi"]["gauge_corrected_rank"] == pytest.approx(0.75)
    assert first["s2_phi"]["gauge_corrected_rank"] == pytest.approx(0.75)

    wrapped = report["cases"][1]["parameters"]
    assert wrapped["s1_phi"]["gauge_truth"] == pytest.approx(
        np.mod(5.8 + 1.0, 2.0 * np.pi)
    )
    assert wrapped["s1_phi"]["stored_rank"] == pytest.approx(0.75)
    assert wrapped["s1_phi"]["gauge_corrected_rank"] == pytest.approx(0.5)

    output = tmp_path / "audit-output"
    json_path, csv_path = audit.write_phase_gauge_audit(output, report)
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()
    audit.write_phase_gauge_audit(output, report)
    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["injection_id"], row["parameter"]) for row in rows] == [
        ("0", "s1_phi"),
        ("0", "s2_phi"),
        ("1", "s1_phi"),
        ("1", "s2_phi"),
    ]
    assert _source_hashes(campaign) == hashes_before


def test_audit_rejects_a_stored_rank_that_cannot_be_reproduced(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    _rewrite_summary(
        campaign,
        0,
        lambda summary: summary["ranks"].__setitem__("s1_phi", 0.5),
    )

    with pytest.raises(ValueError, match="stored s1_phi rank does not reproduce"):
        audit.audit_phase_gauge(campaign)


def test_audit_rejects_a_posterior_hash_mismatch(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    _rewrite_summary(
        campaign,
        0,
        lambda summary: summary["posterior"].__setitem__("sha256", "0" * 64),
    )

    with pytest.raises(ValueError, match="posterior hash mismatch"):
        audit.audit_phase_gauge(campaign)


def test_partial_audit_is_explicit_and_records_missing_ids(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    (common.result_dir(campaign, 1) / "summary.json").unlink()

    with pytest.raises(ValueError, match="missing campaign recoveries: 1"):
        audit.audit_phase_gauge(campaign)

    report = audit.audit_phase_gauge(campaign, allow_partial=True)
    assert report["status"] == "partial"
    assert report["selection"] == {
        "requested_ids": [0, 1],
        "completed_ids": [0],
        "missing_ids": [1],
        "allow_partial": True,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("n_devices", 1, "requires D=4"),
        ("num_gibbs_sweeps", 3, "requires M=1"),
        ("phase_marginalization", False, "requires phase marginalization"),
    ),
)
def test_audit_requires_the_phase_marginalized_d4_m1_configuration(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    campaign = _campaign(tmp_path)
    _rewrite_manifest(
        campaign,
        lambda manifest: manifest["config"].__setitem__(field, value),
    )

    with pytest.raises(ValueError, match=message):
        audit.audit_phase_gauge(campaign)
