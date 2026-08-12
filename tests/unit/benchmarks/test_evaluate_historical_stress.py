import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.injection_campaign import evaluate_historical_stress as stress
from benchmarks.injection_campaign.common import (
    CATALOGUE_FIELDS,
    DEFAULT_CONFIG,
    PARAMETERS,
    SCHEMA_VERSION,
    atomic_savez_compressed,
    atomic_write_csv,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    generate_catalogue,
    posterior_rank,
    result_dir,
)
from benchmarks.injection_campaign.prepare_campaign import prepare_campaign


def _test_catalogue() -> list[dict[str, Any]]:
    catalogue = generate_catalogue(stress.EXPECTED_INJECTIONS, 17)
    for row in catalogue:
        row["q"] = 0.75
        row["t_c"] = 0.0
    return catalogue


def _historical_mapping() -> list[dict[str, Any]]:
    return [
        {
            "preflight_id": preflight_id,
            "historical_id": historical_id,
            "cohort": ("problem" if historical_id in stress.PROBLEM_IDS else "control"),
        }
        for preflight_id, historical_id in enumerate(stress.HISTORICAL_IDS)
    ]


def _historical_provenance() -> dict[str, Any]:
    return {
        "kind": "historical-prior-quantile-stress",
        "scientific_use": "targeted sampler regression; not a P-P set",
        "source_catalogue_sha256": stress.HISTORICAL_SOURCE_SHA256,
        "source_master_seed": stress.HISTORICAL_MASTER_SEED,
        "mapping": _historical_mapping(),
    }


@pytest.fixture(autouse=True)
def _pin_synthetic_mapped_catalogue_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "synthetic-mapped-catalogue.csv"
    atomic_write_csv(path, _test_catalogue(), CATALOGUE_FIELDS)
    monkeypatch.setattr(
        stress,
        "EXPECTED_MAPPED_CATALOGUE_SHA256",
        file_sha256(path),
    )


def _write_result(
    campaign_dir: Path,
    manifest: dict[str, Any],
    catalogue_row: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> None:
    overrides = {} if overrides is None else overrides
    injection_id = int(catalogue_row["injection_id"])
    directory = result_dir(campaign_dir, injection_id)
    directory.mkdir(parents=True)
    q = np.asarray(overrides.get("q", [0.70, 0.75, 0.80]), dtype=float)
    t_c = np.asarray(overrides.get("t_c", [-0.01, 0.0, 0.01]), dtype=float)
    weights = np.asarray(overrides.get("weights", [0.25, 0.5, 0.25]), dtype=float)
    weights /= np.sum(weights)
    log_weights = np.log(weights)
    arrays = {name: np.full(q.size, float(catalogue_row[name])) for name in PARAMETERS}
    arrays["q"] = q
    arrays["t_c"] = t_c
    if "bad_parameter" in overrides:
        arrays[str(overrides["bad_parameter"])] = np.asarray([1.0, np.nan, 2.0])
    arrays["log_likelihood"] = np.linspace(-10.0, -8.0, q.size)
    arrays["log_weights"] = log_weights
    posterior_path = directory / "posterior.npz"
    atomic_savez_compressed(posterior_path, arrays)

    timing = {
        "sample_call": 10.0,
        "sample_phases": {
            "likelihood_jit": 2.0,
            "sampler_kernel_jit": 3.0,
        },
        "paper_convention": {
            "likelihood_jit_seconds": 2.0,
            "sampler_jit_seconds": 3.0,
            "post_jit_sampling_seconds": overrides.get("post_jit", 5.0),
        },
    }
    platform = str(overrides.get("platform", "gpu"))
    device_kind = str(overrides.get("device_kind", stress.REQUIRED_DEVICE_KIND))
    devices = {
        "backend": platform,
        "requested_count": 4,
        "local_count": 4,
        "devices": [
            {
                "id": device_id,
                "platform": platform,
                "device_kind": device_kind,
                "process_index": 0,
            }
            for device_id in range(4)
        ],
    }
    summary = {
        "schema_version": 2,
        "campaign": manifest["config"]["campaign"],
        "config_sha256": manifest["config_sha256"],
        "injection_id": injection_id,
        "truth": {name: catalogue_row[name] for name in CATALOGUE_FIELDS[3:]},
        "seeds": {
            "noise": catalogue_row["noise_seed"],
            "sampler": catalogue_row["sampler_seed"],
        },
        "ranks": {
            "q": posterior_rank(q, float(catalogue_row["q"]), log_weights),
            "t_c": posterior_rank(t_c, float(catalogue_row["t_c"]), log_weights),
        },
        "rank_method": {
            "weighting": "original nested-sampling weights",
            "comparison": "sample < truth",
            "resampled": False,
        },
        "posterior_samples": q.size,
        "posterior_effective_sample_size": float(1.0 / np.sum(weights**2)),
        "posterior": {
            "path": "posterior.npz",
            "sha256": file_sha256(posterior_path),
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
        "timing_seconds": timing,
        "devices": devices,
        "simulated_cpu": platform == "cpu",
    }
    atomic_write_json(directory / "summary.json", summary)


def _write_campaign(
    tmp_path: Path,
    *,
    completed_ids: tuple[int, ...],
    overrides: dict[int, dict[str, Any]] | None = None,
    sampler_scheduler: str = "fsm",
) -> Path:
    campaign_dir = tmp_path / "historical-stress"
    campaign_dir.mkdir()
    catalogue = _test_catalogue()
    catalogue_path = campaign_dir / "catalogue.csv"
    atomic_write_csv(catalogue_path, catalogue, CATALOGUE_FIELDS)

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["campaign"] = stress.SAMPLER_SCHEDULERS[sampler_scheduler]
    config["sampler_scheduler"] = sampler_scheduler
    config["timing"]["selected_events"] = stress.STRESS_TIMING_SELECTION
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": "2026-08-09T00:00:00+00:00",
        "master_seed": None,
        "n_injections": stress.EXPECTED_INJECTIONS,
        "catalogue_size": stress.EXPECTED_INJECTIONS,
        "selection": {
            "rule": "explicit ordered stress catalogue",
            "start_inclusive": 0,
            "stop_exclusive": stress.EXPECTED_INJECTIONS,
        },
        "config": config,
        "catalogue": {
            "path": "catalogue.csv",
            "sha256": file_sha256(catalogue_path),
            "bytes": catalogue_path.stat().st_size,
            "generator": "explicit ordered rows",
            "provenance": _historical_provenance(),
        },
        "psd": {"detector_files": {}, "files": {}},
        "reproduction_scope": {
            "methodology": "arXiv:2607.28265v1 Sections III-V",
            "paper_catalogue_available": False,
            "paper_seeds_available": False,
            "iid_prior_predictive_catalogue": False,
            "pp_calibration_eligible": False,
            "statement": "targeted non-iid stress catalogue",
        },
    }
    manifest["config_sha256"] = canonical_sha256(manifest)
    atomic_write_json(campaign_dir / "manifest.json", manifest)

    overrides = {} if overrides is None else overrides
    for injection_id in completed_ids:
        _write_result(
            campaign_dir,
            manifest,
            catalogue[injection_id],
            overrides=overrides.get(injection_id),
        )
    return campaign_dir


def test_weighted_q_statistics_uses_original_nested_weights() -> None:
    statistics = stress.weighted_q_statistics(
        np.asarray([0.0, 1.0, 2.0]),
        1.8,
        np.log(np.asarray([0.2, 0.5, 0.3])),
    )

    assert statistics["weighted_median"] == 1.0
    assert statistics["weighted_mean"] == pytest.approx(1.1)
    assert statistics["weighted_population_std"] == pytest.approx(0.7)
    assert statistics["z_weighted_median_std"] == pytest.approx(8.0 / 7.0)
    assert statistics["rank"] == pytest.approx(0.7)
    assert statistics["effective_sample_size"] == pytest.approx(1.0 / 0.38)


def test_clean_partial_result_only_permits_continuing_stress_run(
    tmp_path: Path,
) -> None:
    campaign_dir = _write_campaign(tmp_path, completed_ids=(0, 1))

    report = stress.evaluate_historical_stress(campaign_dir, allow_partial=True)

    assert report["decision"] == "partial-pass"
    assert report["continue_stress_run"] is True
    assert report["proceed_to_full_pp_campaign"] is False
    assert report["selection"]["completed_preflight_ids"] == [0, 1]
    assert report["selection"]["missing_preflight_ids"] == list(range(2, 10))
    assert all(case["devices"]["count"] == 4 for case in report["cases"])
    assert all(
        case["timing"]["post_jit_sampling_seconds"] == 5.0 for case in report["cases"]
    )


def test_pre_fsm_lockstep_sharded_campaign_uses_the_same_gate(tmp_path: Path) -> None:
    campaign_dir = _write_campaign(
        tmp_path,
        completed_ids=(0, 1),
        sampler_scheduler="pre-fsm-lockstep",
    )

    report = stress.evaluate_historical_stress(campaign_dir, allow_partial=True)

    assert report["decision"] == "partial-pass"
    assert report["methodology"]["sampler_scheduler"] == "pre-fsm-lockstep"
    assert report["thresholds"]["required_non_cpu_devices"] == 4


def test_evaluator_accepts_manifest_created_by_explicit_catalogue_preparer(
    tmp_path: Path,
) -> None:
    curves = tmp_path / "curves"
    curves.mkdir()
    frequencies = np.asarray([1.0, 4096.0])
    np.savetxt(
        curves / "aLIGO_O4_high_asd.txt",
        np.column_stack((frequencies, [1e-23, 1e-23])),
    )
    np.savetxt(
        curves / "AdV_psd.txt",
        np.column_stack((frequencies, [1e-46, 1e-46])),
    )
    catalogue = _test_catalogue()
    campaign_dir = tmp_path / "prepared-stress"
    manifest = prepare_campaign(
        campaign_dir,
        n_injections=stress.EXPECTED_INJECTIONS,
        catalogue_size=stress.EXPECTED_INJECTIONS,
        seed=17,
        noise_curves_dir=curves,
        catalogue_rows=catalogue,
        catalogue_provenance=_historical_provenance(),
        campaign_name=stress.STRESS_CAMPAIGN_NAME,
    )
    _write_result(campaign_dir, manifest, catalogue[0])

    report = stress.evaluate_historical_stress(campaign_dir, allow_partial=True)

    assert manifest["selection"]["rule"] == "explicit ordered stress catalogue"
    assert manifest["reproduction_scope"]["pp_calibration_eligible"] is False
    assert report["decision"] == "partial-pass"


def test_all_ten_clean_results_permit_full_pp_campaign(tmp_path: Path) -> None:
    campaign_dir = _write_campaign(tmp_path, completed_ids=tuple(range(10)))

    report = stress.evaluate_historical_stress(campaign_dir)

    assert report["decision"] == "pass"
    assert report["continue_stress_run"] is True
    assert report["proceed_to_full_pp_campaign"] is True
    assert report["selection"]["complete"] is True


def test_z_q_above_three_is_an_immediate_hard_stop(tmp_path: Path) -> None:
    campaign_dir = _write_campaign(
        tmp_path,
        completed_ids=(0, 1),
        overrides={1: {"q": [0.70, 0.71, 0.72], "weights": [1, 1, 1]}},
    )

    report = stress.evaluate_historical_stress(campaign_dir, allow_partial=True)

    assert report["decision"] == "hard-stop"
    assert report["hard_stop"] is True
    assert report["continue_stress_run"] is False
    assert report["selection"]["hard_stop_preflight_ids"] == [1]
    assert report["cases"][1]["q"]["z_weighted_median_std"] > 3.0
    assert report["cases"][1]["rank_endpoint_parameters"] == ["q"]


def test_exact_t_c_rank_endpoint_blocks_even_when_z_q_passes(tmp_path: Path) -> None:
    campaign_dir = _write_campaign(
        tmp_path,
        completed_ids=(0,),
        overrides={0: {"t_c": [-0.03, -0.02, -0.01]}},
    )

    report = stress.evaluate_historical_stress(campaign_dir, allow_partial=True)

    assert report["decision"] == "fail"
    assert report["hard_stop"] is False
    assert report["continue_stress_run"] is False
    assert report["cases"][0]["z_q_gate_pass"] is True
    assert report["cases"][0]["rank_endpoint_parameters"] == ["t_c"]


def test_mapped_legacy_q_width_floor_blocks_artificial_narrowing(
    tmp_path: Path,
) -> None:
    campaign_dir = _write_campaign(
        tmp_path,
        completed_ids=(0, 1),
        overrides={1: {"q": [0.749, 0.75, 0.751]}},
    )

    report = stress.evaluate_historical_stress(campaign_dir, allow_partial=True)

    case = report["cases"][1]
    assert case["historical_id"] == 82
    assert case["q"]["z_weighted_median_std"] == 0.0
    assert case["q_width_mapped_legacy_floor"] == 0.00261031248173004
    assert case["q_width_gate_pass"] is False
    assert report["decision"] == "fail"


def test_mapped_q_width_gate_is_strict_at_exact_audited_boundary() -> None:
    floor = stress.MAPPED_Q_WIDTH_FLOORS[82]

    assert stress._q_width_gate(82, floor) == (floor, False)
    assert stress._q_width_gate(82, np.nextafter(floor, np.inf)) == (floor, True)


def test_completed_partial_results_must_form_a_contiguous_prefix(
    tmp_path: Path,
) -> None:
    campaign_dir = _write_campaign(tmp_path, completed_ids=(0, 2))

    with pytest.raises(ValueError, match="contiguous staged prefix"):
        stress.evaluate_historical_stress(campaign_dir, allow_partial=True)


@pytest.mark.parametrize("state_file", ["failure.json", "RUNNING"])
def test_failed_or_running_recovery_is_never_treated_as_pending(
    tmp_path: Path,
    state_file: str,
) -> None:
    campaign_dir = _write_campaign(tmp_path, completed_ids=(0,))
    directory = result_dir(campaign_dir, 1)
    directory.mkdir(parents=True)
    atomic_write_json(directory / state_file, {"state": state_file})

    with pytest.raises(ValueError, match="failure record|marked running"):
        stress.evaluate_historical_stress(campaign_dir, allow_partial=True)


def test_non_q_posterior_corruption_cannot_pass_the_gate(tmp_path: Path) -> None:
    campaign_dir = _write_campaign(
        tmp_path,
        completed_ids=(0,),
        overrides={0: {"bad_parameter": "M_c"}},
    )

    with pytest.raises(ValueError, match="M_c contains non-finite"):
        stress.evaluate_historical_stress(campaign_dir, allow_partial=True)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"post_jit": 6.0}, "timing arithmetic"),
        ({"platform": "cpu"}, "CPU simulation"),
        ({"device_kind": "NVIDIA A100"}, "NVIDIA H200"),
    ],
)
def test_invalid_methodology_artifact_is_rejected(
    tmp_path: Path,
    override: dict[str, Any],
    message: str,
) -> None:
    campaign_dir = _write_campaign(
        tmp_path,
        completed_ids=(0,),
        overrides={0: override},
    )

    with pytest.raises(ValueError, match=message):
        stress.evaluate_historical_stress(campaign_dir, allow_partial=True)


def test_partial_results_require_explicit_staged_mode(tmp_path: Path) -> None:
    campaign_dir = _write_campaign(tmp_path, completed_ids=(0, 1))

    with pytest.raises(ValueError, match="--allow-partial"):
        stress.evaluate_historical_stress(campaign_dir)
