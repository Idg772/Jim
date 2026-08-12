from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign import merge_staged_results as merge_module
from jimgw.samplers.diagnostics import insertion_index_diagnostic


def _minimal_campaign(
    directory: Path,
    *,
    n_injections: int = 2,
    master_seed: int = 1234,
) -> Path:
    directory.mkdir()
    rows = common.generate_catalogue(n_injections, master_seed)
    catalogue_path = directory / "catalogue.csv"
    common.atomic_write_csv(catalogue_path, rows, common.CATALOGUE_FIELDS)

    psd_path = directory / "inputs/psd/test-design.npz"
    psd_path.parent.mkdir(parents=True)
    psd_path.write_bytes(b"immutable test PSD\n")
    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config["campaign"] = "merge-staged-results-test"
    manifest = {
        "schema_version": common.SCHEMA_VERSION,
        "n_injections": n_injections,
        "catalogue_size": n_injections,
        "config": config,
        "catalogue": {
            "path": "catalogue.csv",
            "sha256": common.file_sha256(catalogue_path),
            "bytes": catalogue_path.stat().st_size,
        },
        "psd": {
            "files": {
                "inputs/psd/test-design.npz": {
                    "sha256": common.file_sha256(psd_path),
                    "bytes": psd_path.stat().st_size,
                }
            }
        },
    }
    manifest["config_sha256"] = common.canonical_sha256(manifest)
    common.atomic_write_json(directory / "manifest.json", manifest)
    common.refresh_status(directory, n_injections)
    return directory


def _minimal_baseline_campaign(directory: Path) -> Path:
    campaign = _minimal_campaign(directory, n_injections=1)
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"].update(
        {
            "campaign": "paper-baseline-code-high-res-d1-m3-pathology-diagnostic",
            "paper_configuration": "High-Res",
            "n_devices": 1,
            "num_gibbs_sweeps": 3,
        }
    )
    manifest["config"].pop("sampler_scheduler", None)
    manifest["baseline_diagnostic"] = {
        "implementation_label": "paper-baseline",
        "implementation_revision": "86335bdb1e7ef6191937dd17b2ca53edbb1d899f",
        "implementation_tree_sha256": (
            "09085b4d427cfbbb9b379228b2b5d6cb781042687c64f0207740d1af3354c141"
        ),
        "paper_configuration": "High-Res",
        "paper_timing_available": False,
    }
    manifest.pop("config_sha256")
    manifest["config_sha256"] = common.canonical_sha256(manifest)
    common.atomic_write_json(manifest_path, manifest)
    return campaign


def _mark_blocking_remediation(campaign: Path) -> None:
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["blocking_remediation"] = {"scheme": "all-slow-time"}
    manifest.pop("config_sha256")
    manifest["config_sha256"] = common.canonical_sha256(manifest)
    common.atomic_write_json(manifest_path, manifest)


def test_candidate_implementation_pin_is_enforced() -> None:
    revision = "a" * 40
    tree_sha256 = "b" * 64
    manifest = {
        "implementation_diagnostic": {
            "implementation_label": "candidate",
            "implementation_revision": revision,
            "implementation_tree_sha256": tree_sha256,
        }
    }
    summary = {
        "implementation": {
            "label": "candidate",
            "revision": revision,
            "tree_sha256": tree_sha256,
            "root": "/workspace/candidate",
            "jimgw_module": "/workspace/candidate/src/jimgw/__init__.py",
        }
    }

    merge_module._validate_implementation(
        summary,
        manifest,
        result_label="candidate result",
    )

    summary["implementation"]["tree_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="implementation provenance mismatch"):
        merge_module._validate_implementation(
            summary,
            manifest,
            result_label="candidate result",
        )


def _write_valid_result(
    campaign: Path,
    injection_id: int,
    *,
    variant: float = 0.0,
    log_likelihood_birth: np.ndarray | None = None,
    include_insertion_diagnostic: bool = True,
) -> Path:
    manifest = common.load_manifest(campaign)
    truth = common.read_catalogue(campaign / "catalogue.csv")[injection_id]
    directory = common.result_dir(campaign, injection_id)
    directory.mkdir(parents=True)

    offsets = np.asarray([-0.3, -0.1, 0.2, 0.5]) + variant
    samples = {
        name: np.asarray(float(truth[name]) + offsets) for name in common.PARAMETERS
    }
    log_weights = np.log(np.asarray([0.1, 0.2, 0.3, 0.4]))
    arrays = {
        **samples,
        "log_likelihood": np.asarray([-4.0, -3.0, -2.0, -1.0]),
        "log_weights": log_weights,
    }
    if log_likelihood_birth is not None:
        arrays["log_likelihood_birth"] = np.asarray(log_likelihood_birth)
    posterior_path = directory / "posterior.npz"
    common.atomic_savez_compressed(posterior_path, arrays)
    ranks = {
        name: common.posterior_rank(values, float(truth[name]), log_weights)
        for name, values in samples.items()
    }
    summary = {
        "schema_version": common.SCHEMA_VERSION,
        "campaign": manifest["config"]["campaign"],
        "config_sha256": manifest["config_sha256"],
        "injection_id": injection_id,
        "truth": {
            name: truth[name]
            for name in (*common.PARAMETERS, *common.MARGINALIZED_PARAMETERS)
        },
        "seeds": {
            "noise": truth["noise_seed"],
            "sampler": truth["sampler_seed"],
        },
        "ranks": ranks,
        "rank_method": {
            "weighting": "original nested-sampling weights",
            "comparison": "sample < truth",
            "resampled": False,
        },
        "posterior_samples": len(log_weights),
        "posterior_effective_sample_size": float(
            1.0 / np.sum(np.exp(log_weights) ** 2)
        ),
        "posterior": {
            "path": "posterior.npz",
            "sha256": common.file_sha256(posterior_path),
            "bytes": posterior_path.stat().st_size,
            "fields": list(arrays),
            "space": "prior",
            "weighting": "normalized nested-sampling log weights",
        },
        "timing_seconds": {
            "sample_call": 100.0 + injection_id,
            "total": 150.0 + injection_id,
            "sample_phases": {
                "likelihood_jit": 10.0,
                "sampler_kernel_jit": 20.0,
            },
            "paper_convention": {
                "likelihood_jit_seconds": 10.0,
                "sampler_jit_seconds": 20.0,
                "post_jit_sampling_seconds": 70.0 + injection_id,
            },
        },
        "devices": {
            "backend": "gpu",
            "requested_count": 4,
            "local_count": 4,
            "devices": [{"id": index, "platform": "gpu"} for index in range(4)],
        },
        "simulated_cpu": False,
    }
    if log_likelihood_birth is not None and include_insertion_diagnostic:
        summary["diagnostics"] = {
            "insertion_index": insertion_index_diagnostic(
                arrays["log_likelihood"],
                arrays["log_likelihood_birth"],
                n_live=int(manifest["config"]["n_live"]),
            )
        }
    common.atomic_write_json(directory / "summary.json", summary)
    return directory


def _write_valid_baseline_result(campaign: Path, injection_id: int) -> Path:
    directory = _write_valid_result(campaign, injection_id)
    manifest = common.load_manifest(campaign)
    diagnostic = manifest["baseline_diagnostic"]
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["timing_seconds"].update(
        {
            "sample_phases": None,
            "paper_convention": None,
            "paper_convention_unavailable_reason": (
                "The pinned paper baseline predates split likelihood/sampler JIT "
                "phase instrumentation; no post-JIT timing is inferred."
            ),
        }
    )
    summary["implementation"] = {
        "label": diagnostic["implementation_label"],
        "revision": diagnostic["implementation_revision"],
        "tree_sha256": diagnostic["implementation_tree_sha256"],
        "root": "/workspace/paper-baseline",
        "jimgw_module": "/workspace/paper-baseline/src/jimgw/__init__.py",
    }
    summary["devices"] = {
        "backend": "gpu",
        "requested_count": 1,
        "local_count": 1,
        "devices": [{"id": 0, "platform": "gpu"}],
    }
    common.atomic_write_json(summary_path, summary)
    return directory


def _stage_archive(
    tmp_path: Path,
    campaign: Path,
    name: str,
    results: dict[int, float],
    *,
    incomplete_ids: tuple[int, ...] = (),
    birth_likelihoods: dict[int, np.ndarray] | None = None,
    include_insertion_diagnostics: bool = True,
) -> Path:
    stage = tmp_path / name
    stage.mkdir()
    for path in campaign.iterdir():
        if path.is_dir():
            shutil.copytree(path, stage / path.name)
        else:
            (stage / path.name).write_bytes(path.read_bytes())
    for injection_id, variant in results.items():
        _write_valid_result(
            stage,
            injection_id,
            variant=variant,
            log_likelihood_birth=(birth_likelihoods or {}).get(injection_id),
            include_insertion_diagnostic=include_insertion_diagnostics,
        )
    for injection_id in incomplete_ids:
        directory = common.result_dir(stage, injection_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "failure.json").write_text('{"fixture": true}\n')

    archive = tmp_path / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        package.add(stage, arcname=stage.name)
    return archive


def _stage_baseline_archive(tmp_path: Path, campaign: Path, name: str) -> Path:
    stage = tmp_path / name
    stage.mkdir()
    for path in campaign.iterdir():
        if path.is_dir():
            shutil.copytree(path, stage / path.name)
        else:
            (stage / path.name).write_bytes(path.read_bytes())
    _write_valid_baseline_result(stage, 0)
    archive = tmp_path / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        package.add(stage, arcname=stage.name)
    return archive


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(common.file_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _rewrite_summary(directory: Path, mutation: str) -> None:
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    if mutation == "rank":
        summary["ranks"]["M_c"] = 0.987654321
    elif mutation == "timing":
        summary["timing_seconds"]["paper_convention"]["post_jit_sampling_seconds"] += (
            1.0
        )
    elif mutation == "seed":
        summary["seeds"]["sampler"] += 1
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    common.atomic_write_json(summary_path, summary)


def test_cli_merges_disjoint_archives_regenerates_status_and_is_idempotent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign")
    first = _stage_archive(
        tmp_path,
        campaign,
        "stage-a",
        {0: 0.0},
        incomplete_ids=(1,),
    )
    second = _stage_archive(tmp_path, campaign, "stage-b", {1: 0.0})
    conflicting = _stage_archive(tmp_path, campaign, "stage-conflict", {0: 0.25})
    before = _tree_sha256(campaign)

    merge_module.main([str(campaign), str(first), str(first), str(second), "--dry-run"])
    dry_report = json.loads(capsys.readouterr().out)

    assert dry_report["candidate_ids"] == [0, 1]
    assert dry_report["copy_ids"] == [0, 1]
    assert dry_report["duplicate_archive_ids"] == [0]
    assert dry_report["archives"][0]["ignored_incomplete_ids"] == [1]
    assert dry_report["status_regenerated"] is False
    assert _tree_sha256(campaign) == before

    report = merge_module.merge_staged_results(campaign, [first, second])

    assert report["copy_ids"] == [0, 1]
    assert report["complete_ids_after"] == [0, 1]
    assert report["status_after"] == {
        "complete": 2,
        "failed": 0,
        "pending": 0,
        "running": 0,
        "invalid": 0,
    }
    assert report["status_regenerated"] is True

    repeated = merge_module.merge_staged_results(campaign, [first, second])

    assert repeated["copy_ids"] == []
    assert repeated["skipped_existing_ids"] == [0, 1]
    assert repeated["complete_ids_after"] == [0, 1]

    completed_tree = _tree_sha256(campaign)
    with pytest.raises(ValueError, match="destination collision for injection 000"):
        merge_module.merge_staged_results(campaign, [conflicting])
    assert _tree_sha256(campaign) == completed_tree


def test_merge_accepts_and_preserves_optional_birth_likelihoods(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=1)
    expected_birth = np.asarray([-np.inf, -4.0, -3.0, -2.0])
    archive = _stage_archive(
        tmp_path,
        campaign,
        "stage-with-birth-likelihoods",
        {0: 0.0},
        birth_likelihoods={0: expected_birth},
    )

    report = merge_module.merge_staged_results(campaign, [archive])

    assert report["copy_ids"] == [0]
    result = common.result_dir(campaign, 0)
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert "log_likelihood_birth" in summary["posterior"]["fields"]
    with np.load(result / "posterior.npz", allow_pickle=False) as posterior:
        np.testing.assert_array_equal(posterior["log_likelihood_birth"], expected_birth)


@pytest.mark.parametrize(
    "field",
    (
        "method",
        "n_live",
        "sample_size",
        "statistic",
        "p_value",
        "index_min",
        "index_max",
        "index_mean",
        "expected_index_mean",
    ),
)
def test_tampered_insertion_index_summary_is_rejected_before_merge(
    tmp_path: Path,
    field: str,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=1)
    archive = _stage_archive(
        tmp_path,
        campaign,
        "stage-with-tampered-insertion-diagnostic",
        {0: 0.0},
        birth_likelihoods={0: np.asarray([-np.inf, -4.0, -3.0, -2.0])},
    )
    stage = tmp_path / "stage-with-tampered-insertion-diagnostic"
    summary_path = stage / "results/injection-000/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stored = summary["diagnostics"]["insertion_index"]
    value = stored[field]
    stored[field] = f"{value}-tampered" if isinstance(value, str) else value + 1
    common.atomic_write_json(summary_path, summary)
    with tarfile.open(archive, "w:gz") as package:
        package.add(stage, arcname=stage.name)
    before = _tree_sha256(campaign)

    with pytest.raises(
        ValueError,
        match=rf"insertion-index diagnostic mismatch for {field}",
    ):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before
    assert not (campaign / "results").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing insertion-index diagnostic"),
        ("extra-field", "insertion-index diagnostic inventory is invalid"),
    ],
)
def test_birth_likelihoods_require_a_complete_insertion_index_summary(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=1)
    archive = _stage_archive(
        tmp_path,
        campaign,
        "stage-with-invalid-insertion-inventory",
        {0: 0.0},
        birth_likelihoods={0: np.asarray([-np.inf, -4.0, -3.0, -2.0])},
    )
    stage = tmp_path / "stage-with-invalid-insertion-inventory"
    summary_path = stage / "results/injection-000/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        summary["diagnostics"].pop("insertion_index")
    else:
        summary["diagnostics"]["insertion_index"]["unexpected"] = 0
    common.atomic_write_json(summary_path, summary)
    with tarfile.open(archive, "w:gz") as package:
        package.add(stage, arcname=stage.name)
    before = _tree_sha256(campaign)

    with pytest.raises((TypeError, ValueError), match=message):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before
    assert not (campaign / "results").exists()


def test_legacy_result_without_birth_likelihoods_or_insertion_summary_still_merges(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=1)
    archive = _stage_archive(tmp_path, campaign, "legacy-stage", {0: 0.0})

    report = merge_module.merge_staged_results(campaign, [archive])

    assert report["complete_ids_after"] == [0]
    summary = json.loads(
        (common.result_dir(campaign, 0) / "summary.json").read_text(encoding="utf-8")
    )
    assert "log_likelihood_birth" not in summary["posterior"]["fields"]
    assert "diagnostics" not in summary


def test_remediation_result_without_birth_likelihoods_is_rejected_before_merge(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=1)
    _mark_blocking_remediation(campaign)
    archive = _stage_archive(tmp_path, campaign, "remediation-stage", {0: 0.0})
    before = _tree_sha256(campaign)

    with pytest.raises(
        ValueError,
        match=(
            "remediation result requires log_likelihood_birth and an "
            "insertion-index diagnostic"
        ),
    ):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before
    assert not (campaign / "results").exists()


def test_blocking_remediation_marker_is_the_insertion_evidence_boundary() -> None:
    assert not merge_module._requires_insertion_evidence(
        {"implementation_diagnostic": {"implementation_label": "candidate"}},
        result_label="legacy candidate",
    )
    assert merge_module._requires_insertion_evidence(
        {"blocking_remediation": {"scheme": "all-slow-time"}},
        result_label="remediation candidate",
    )
    with pytest.raises(TypeError, match="blocking_remediation marker is invalid"):
        merge_module._requires_insertion_evidence(
            {"blocking_remediation": "all-slow-time"},
            result_label="malformed candidate",
        )


def test_insertion_index_summary_without_birth_likelihoods_is_rejected(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=1)
    archive = _stage_archive(tmp_path, campaign, "inconsistent-stage", {0: 0.0})
    stage = tmp_path / "inconsistent-stage"
    summary_path = stage / "results/injection-000/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["diagnostics"] = {"insertion_index": {}}
    common.atomic_write_json(summary_path, summary)
    with tarfile.open(archive, "w:gz") as package:
        package.add(stage, arcname=stage.name)

    with pytest.raises(
        ValueError,
        match="insertion-index diagnostic requires log_likelihood_birth",
    ):
        merge_module.merge_staged_results(campaign, [archive])

    assert not (campaign / "results").exists()


@pytest.mark.parametrize(
    ("birth_likelihoods", "message"),
    [
        (
            np.asarray([-np.inf, np.nan, -3.0, -2.0]),
            r"log_likelihood_birth contains NaN or \+inf",
        ),
        (
            np.asarray([-np.inf, np.inf, -3.0, -2.0]),
            r"log_likelihood_birth contains NaN or \+inf",
        ),
        (
            np.full(4, -np.inf),
            "log_likelihood_birth has no replacement points",
        ),
        (
            np.asarray([-np.inf, -3.0, -3.0, -2.0]),
            "replacement likelihood does not exceed its birth",
        ),
    ],
    ids=("nan", "positive-infinity", "no-replacements", "invalid-ordering"),
)
def test_invalid_birth_likelihoods_are_rejected_before_merge(
    tmp_path: Path,
    birth_likelihoods: np.ndarray,
    message: str,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=1)
    archive = _stage_archive(
        tmp_path,
        campaign,
        "stage-with-invalid-birth-likelihoods",
        {0: 0.0},
        birth_likelihoods={0: birth_likelihoods},
        include_insertion_diagnostics=False,
    )
    before = _tree_sha256(campaign)

    with pytest.raises(ValueError, match=message):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before
    assert not (campaign / "results").exists()


def test_baseline_result_with_explicitly_unavailable_jit_timing_merges(
    tmp_path: Path,
) -> None:
    campaign = _minimal_baseline_campaign(tmp_path / "campaign")
    archive = _stage_baseline_archive(tmp_path, campaign, "baseline-stage")

    report = merge_module.merge_staged_results(campaign, [archive])

    assert report["copy_ids"] == [0]
    assert report["complete_ids_after"] == [0]
    assert report["archives"][0]["results"][0]["post_jit_sampling_seconds"] is None
    summary = json.loads(
        (campaign / "results/injection-000/summary.json").read_text(encoding="utf-8")
    )
    assert summary["timing_seconds"]["paper_convention"] is None
    assert summary["timing_seconds"]["sample_phases"] is None
    assert summary["timing_seconds"]["paper_convention_unavailable_reason"]
    assert summary["implementation"] == {
        "label": "paper-baseline",
        "revision": "86335bdb1e7ef6191937dd17b2ca53edbb1d899f",
        "tree_sha256": (
            "09085b4d427cfbbb9b379228b2b5d6cb781042687c64f0207740d1af3354c141"
        ),
        "root": "/workspace/paper-baseline",
        "jimgw_module": "/workspace/paper-baseline/src/jimgw/__init__.py",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-reason", "missing Figure 3 post-JIT timing"),
        ("claimed-phases", "baseline timing must not claim sampler phase data"),
    ],
)
def test_baseline_unavailable_timing_contract_is_enforced_before_merge(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    campaign = _minimal_baseline_campaign(tmp_path / "campaign")
    archive = _stage_baseline_archive(tmp_path, campaign, "baseline-stage")
    stage = tmp_path / "baseline-stage"
    summary_path = stage / "results/injection-000/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "missing-reason":
        summary["timing_seconds"].pop("paper_convention_unavailable_reason")
    else:
        summary["timing_seconds"]["sample_phases"] = {
            "likelihood_jit": 10.0,
            "sampler_kernel_jit": 20.0,
        }
    common.atomic_write_json(summary_path, summary)
    with tarfile.open(archive, "w:gz") as package:
        package.add(stage, arcname=stage.name)
    before = _tree_sha256(campaign)

    with pytest.raises((TypeError, ValueError), match=message):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before
    assert not (campaign / "results").exists()


def test_baseline_implementation_tree_mismatch_is_rejected_before_merge(
    tmp_path: Path,
) -> None:
    campaign = _minimal_baseline_campaign(tmp_path / "campaign")
    archive = _stage_baseline_archive(tmp_path, campaign, "baseline-stage")
    stage = tmp_path / "baseline-stage"
    summary_path = stage / "results/injection-000/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["implementation"]["tree_sha256"] = "0" * 64
    common.atomic_write_json(summary_path, summary)
    with tarfile.open(archive, "w:gz") as package:
        package.add(stage, arcname=stage.name)
    before = _tree_sha256(campaign)

    with pytest.raises(ValueError, match="implementation provenance mismatch"):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before
    assert not (campaign / "results").exists()


def test_cross_archive_conflict_is_rejected_before_any_result_is_copied(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign")
    first = _stage_archive(tmp_path, campaign, "stage-a", {0: 0.0, 1: 0.0})
    conflicting = _stage_archive(tmp_path, campaign, "stage-b", {0: 0.25})
    before = _tree_sha256(campaign)

    with pytest.raises(ValueError, match="cross-archive collision for injection 000"):
        merge_module.merge_staged_results(campaign, [first, conflicting])

    assert _tree_sha256(campaign) == before
    assert not (campaign / "results").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("posterior", "posterior hash mismatch"),
        ("rank", "posterior rank mismatch for M_c"),
        ("timing", "inconsistent post-JIT timing arithmetic"),
        ("seed", "seeds do not match the catalogue"),
    ],
)
def test_corrupt_scientific_result_is_rejected_without_writes(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign")
    stage = tmp_path / "corrupt-stage"
    stage.mkdir()
    for path in campaign.iterdir():
        if path.is_dir():
            shutil.copytree(path, stage / path.name)
        else:
            (stage / path.name).write_bytes(path.read_bytes())
    result = _write_valid_result(stage, 0)
    if mutation == "posterior":
        with (result / "posterior.npz").open("ab") as stream:
            stream.write(b"corrupt")
    else:
        _rewrite_summary(result, mutation)
    archive = tmp_path / "corrupt-stage.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        package.add(stage, arcname=stage.name)
    before = _tree_sha256(campaign)

    with pytest.raises(ValueError, match=message):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before
    assert not (campaign / "results").exists()


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("traversal", "unsafe path"),
        ("symlink", "forbidden special member"),
        ("appledouble", "exactly one root directory"),
    ],
)
def test_unsafe_tar_members_are_rejected_without_writes(
    tmp_path: Path,
    kind: str,
    message: str,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign")
    archive = tmp_path / f"unsafe-{kind}.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        root = tarfile.TarInfo("stage/")
        root.type = tarfile.DIRTYPE
        package.addfile(root)
        if kind == "traversal":
            payload = b"escape"
            member = tarfile.TarInfo("stage/../../escape")
            member.size = len(payload)
            package.addfile(member, io.BytesIO(payload))
        elif kind == "symlink":
            member = tarfile.TarInfo("stage/posterior-link")
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            package.addfile(member)
        else:
            second_root = tarfile.TarInfo("._stage/")
            second_root.type = tarfile.DIRTYPE
            package.addfile(second_root)
    before = _tree_sha256(campaign)

    with pytest.raises(ValueError, match=message):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before
    assert not (tmp_path / "escape").exists()


def test_symlinked_destination_results_cannot_redirect_copies(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=1)
    archive = _stage_archive(tmp_path, campaign, "stage", {0: 0.0})
    outside = tmp_path / "outside-results"
    outside.mkdir()
    (campaign / "results").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="results path must be a real directory"):
        merge_module.merge_staged_results(campaign, [archive])

    assert list(outside.iterdir()) == []


def test_symlinked_destination_status_cannot_overwrite_external_file(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=1)
    archive = _stage_archive(tmp_path, campaign, "stage", {0: 0.0})
    victim = tmp_path / "victim.txt"
    victim.write_text("keep this content\n")
    (campaign / "status.csv").unlink()
    (campaign / "status.csv").symlink_to(victim)

    with pytest.raises(ValueError, match="status path must be a regular file"):
        merge_module.merge_staged_results(campaign, [archive])

    assert victim.read_text() == "keep this content\n"


def test_symlinked_existing_result_directory_is_rejected(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=2)
    archive = _stage_archive(tmp_path, campaign, "stage", {1: 0.0})
    result = _write_valid_result(campaign, 0)
    external_result = tmp_path / "external-result"
    result.rename(external_result)
    result.symlink_to(external_result, target_is_directory=True)

    with pytest.raises(ValueError, match="result path must be a real directory"):
        merge_module.merge_staged_results(campaign, [archive])

    assert not common.result_dir(campaign, 1).exists()
    assert (external_result / "summary.json").is_file()


def test_symlinked_existing_result_file_is_rejected(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", n_injections=2)
    archive = _stage_archive(tmp_path, campaign, "stage", {1: 0.0})
    result = _write_valid_result(campaign, 0)
    posterior = result / "posterior.npz"
    external_posterior = tmp_path / "external-posterior.npz"
    shutil.copy2(posterior, external_posterior)
    expected_sha256 = common.file_sha256(external_posterior)
    posterior.unlink()
    posterior.symlink_to(external_posterior)

    with pytest.raises(ValueError, match="result entries must be regular files"):
        merge_module.merge_staged_results(campaign, [archive])

    assert common.file_sha256(external_posterior) == expected_sha256
    assert not common.result_dir(campaign, 1).exists()


def test_archive_atomic_replacement_during_preflight_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign")
    archive = _stage_archive(tmp_path, campaign, "stage-original", {0: 0.0})
    replacement = _stage_archive(tmp_path, campaign, "stage-replacement", {1: 0.0})
    original_tar_open = tarfile.open
    replaced = False

    def replacing_tar_open(*args: object, **kwargs: object) -> tarfile.TarFile:
        nonlocal replaced
        package = original_tar_open(*args, **kwargs)
        if not replaced:
            replacement.replace(archive)
            replaced = True
        return package

    monkeypatch.setattr(merge_module.tarfile, "open", replacing_tar_open)
    before = _tree_sha256(campaign)

    with pytest.raises(ValueError, match="archive changed during preflight"):
        merge_module.merge_staged_results(campaign, [archive])

    assert replaced is True
    assert _tree_sha256(campaign) == before


def test_staged_campaign_must_match_all_immutable_destination_inputs(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign", master_seed=1234)
    other = _minimal_campaign(tmp_path / "other", master_seed=5678)
    archive = _stage_archive(tmp_path, other, "other-stage", {0: 0.0})
    before = _tree_sha256(campaign)

    with pytest.raises(ValueError, match="staged manifest differs from destination"):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("catalogue.csv", "catalogue hash mismatch"),
        ("inputs/psd/test-design.npz", "PSD hash mismatch"),
    ],
)
def test_staged_immutable_payload_hashes_are_verified(
    tmp_path: Path,
    relative: str,
    message: str,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign")
    archive = _stage_archive(tmp_path, campaign, "tampered-stage", {0: 0.0})
    stage = tmp_path / "tampered-stage"
    with (stage / relative).open("ab") as stream:
        stream.write(b"tampered")
    with tarfile.open(archive, "w:gz") as package:
        package.add(stage, arcname=stage.name)
    before = _tree_sha256(campaign)

    with pytest.raises(ValueError, match=message):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before


def test_incomplete_destination_collision_is_never_overwritten(
    tmp_path: Path,
) -> None:
    campaign = _minimal_campaign(tmp_path / "campaign")
    archive = _stage_archive(tmp_path, campaign, "stage", {0: 0.0})
    incomplete = common.result_dir(campaign, 0)
    incomplete.mkdir(parents=True)
    failed_log = incomplete / "attempt-01.failed.log"
    failed_log.write_text("existing local attempt\n")
    before = _tree_sha256(campaign)

    with pytest.raises(ValueError, match="incomplete collision for injection 000"):
        merge_module.merge_staged_results(campaign, [archive])

    assert _tree_sha256(campaign) == before
    assert failed_log.read_text() == "existing local attempt\n"
