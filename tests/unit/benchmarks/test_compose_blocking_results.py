from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign import compose_blocking_results as compose_module
from benchmarks.injection_campaign.compose_blocking_results import (
    ComponentSource,
    read_schedule,
    validate_component_compatibility,
)


def _write_schedule(path: Path, rows: list[tuple[int, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("injection_id", "component"))
        writer.writerows(rows)
    return path


def _component(label: str, path: Path) -> ComponentSource:
    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config["campaign"] = f"component-{label}"
    config["paper_configuration"] = f"Component {label}"
    manifest = {
        "schema_version": common.SCHEMA_VERSION,
        "n_injections": common.PAPER_PP_RECOVERIES,
        "catalogue_size": common.PAPER_CATALOGUE_SIZE,
        "master_seed": 260728265,
        "selection": {
            "rule": "first catalogue entries",
            "start_inclusive": 0,
            "stop_exclusive": common.PAPER_PP_RECOVERIES,
        },
        "config": config,
        "config_sha256": label * 64,
        "catalogue": {"path": "catalogue.csv", "sha256": "a" * 64},
        "psd": {"files": {"inputs/psd/design.npz": {"sha256": "b" * 64, "bytes": 10}}},
        "implementation_diagnostic": {
            "implementation_label": "candidate",
            "implementation_revision": "c" * 40,
            "implementation_tree_sha256": "d" * 64,
            "sampler_scheduler": "fsm",
        },
    }
    catalogue = [
        {
            "injection_id": injection_id,
            "noise_seed": injection_id + 1000,
            "sampler_seed": injection_id + 2000,
        }
        for injection_id in range(common.PAPER_CATALOGUE_SIZE)
    ]
    return ComponentSource(
        label=label,
        path=path,
        manifest=manifest,
        manifest_sha256="e" * 64,
        catalogue=catalogue,
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [(injection_id, "primary") for injection_id in range(100)]
            + [(99, "fallback")],
            "overlapping injection ID 99",
        ),
        (
            [(injection_id, "primary") for injection_id in range(99)],
            "schedule is missing injection IDs: 99",
        ),
    ],
    ids=("overlap", "gap"),
)
def test_schedule_requires_each_leading_id_exactly_once(
    tmp_path: Path,
    rows: list[tuple[int, str]],
    message: str,
) -> None:
    schedule_path = _write_schedule(tmp_path / "schedule.csv", rows)

    with pytest.raises(ValueError, match=message):
        read_schedule(schedule_path, component_names={"primary", "fallback"})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("catalogue", "catalogue truth/seed mismatch"),
        ("psd", "PSD provenance mismatch"),
        ("config", "scientific configuration mismatch"),
    ],
)
def test_components_must_match_outside_declared_blocking_variations(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    primary = _component("primary", tmp_path / "primary")
    fallback = _component("fallback", tmp_path / "fallback")
    if mutation == "catalogue":
        fallback.catalogue[0]["noise_seed"] += 1
    elif mutation == "psd":
        fallback.manifest["psd"]["files"]["inputs/psd/design.npz"]["sha256"] = "0" * 64
    elif mutation == "config":
        fallback.manifest["config"]["waveform"] = "different-waveform"

    with pytest.raises(ValueError, match=message):
        validate_component_compatibility({"primary": primary, "fallback": fallback})


def test_components_may_differ_only_in_blocks_and_presentation_labels(
    tmp_path: Path,
) -> None:
    primary = _component("primary", tmp_path / "primary")
    fallback = _component("fallback", tmp_path / "fallback")
    fallback.manifest["config"]["blocks"] = [
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

    reference = validate_component_compatibility(
        {"primary": primary, "fallback": fallback}
    )

    assert reference == "fallback"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("implementation_label", "different-candidate"),
        ("implementation_revision", "0" * 40),
        ("sampler_scheduler", "different-scheduler"),
    ],
)
def test_implementation_identity_mismatch_aborts_before_results_ranks_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    component_paths = {
        "primary": tmp_path / "primary",
        "fallback": tmp_path / "fallback",
    }
    components = {
        label: _component(label, path) for label, path in component_paths.items()
    }
    components["primary"].manifest["implementation_diagnostic"][field] = value
    schedule_path = _write_schedule(
        tmp_path / "schedule.csv",
        [
            (injection_id, "primary" if injection_id < 92 else "fallback")
            for injection_id in range(common.PAPER_PP_RECOVERIES)
        ],
    )
    monkeypatch.setattr(
        compose_module,
        "load_component",
        lambda label, _path: components[label],
    )
    result_validation_called = False
    rank_loader_called = False

    def forbidden_result_validation(*_args: object, **_kwargs: object) -> object:
        nonlocal result_validation_called
        result_validation_called = True
        raise AssertionError("results must not validate after incompatible provenance")

    def forbidden_rank_loader(*_args: object, **_kwargs: object) -> object:
        nonlocal rank_loader_called
        rank_loader_called = True
        raise AssertionError("ranks must not load after incompatible provenance")

    monkeypatch.setattr(
        compose_module,
        "_validate_scheduled_results",
        forbidden_result_validation,
    )
    monkeypatch.setattr(compose_module, "_rank_rows", forbidden_rank_loader)
    output = tmp_path / "composition"

    with pytest.raises(ValueError, match=rf"implementation {field} mismatch"):
        compose_module.compose_blocking_results(
            output,
            component_paths,
            schedule_path,
        )

    assert result_validation_called is False
    assert rank_loader_called is False
    assert not output.exists()


def test_strong_result_validation_failure_aborts_before_rank_loading_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component_paths = {
        "primary": tmp_path / "primary",
        "fallback": tmp_path / "fallback",
    }
    components = {
        label: _component(label, path) for label, path in component_paths.items()
    }
    schedule_rows = [
        (injection_id, "primary" if injection_id < 92 else "fallback")
        for injection_id in range(common.PAPER_PP_RECOVERIES)
    ]
    schedule_path = _write_schedule(tmp_path / "schedule.csv", schedule_rows)
    for injection_id, label in schedule_rows:
        directory = common.result_dir(component_paths[label], injection_id)
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text("{}", encoding="utf-8")
        (directory / "posterior.npz").write_bytes(b"tampered")

    monkeypatch.setattr(
        compose_module,
        "load_component",
        lambda label, _path: components[label],
    )
    monkeypatch.setattr(
        compose_module,
        "validate_component_compatibility",
        lambda _components: "primary",
    )
    rank_loader_called = False

    def fail_closed_validator(
        _directory: Path,
        injection_id: int,
        *_args: object,
    ) -> object:
        if injection_id == 42:
            raise ValueError("posterior hash mismatch")
        return object()

    def forbidden_rank_loader(*_args: object, **_kwargs: object) -> object:
        nonlocal rank_loader_called
        rank_loader_called = True
        raise AssertionError("ranks must not load before all results validate")

    monkeypatch.setattr(
        compose_module.merge_module,
        "_validate_result",
        fail_closed_validator,
    )
    monkeypatch.setattr(compose_module, "_rank_rows", forbidden_rank_loader)
    output = tmp_path / "composition"

    with pytest.raises(ValueError, match="posterior hash mismatch"):
        compose_module.compose_blocking_results(
            output,
            component_paths,
            schedule_path,
        )

    assert rank_loader_called is False
    assert not output.exists()


def test_complete_composition_records_schedule_and_keeps_q_in_strict_all15_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component_paths = {
        "primary": tmp_path / "primary",
        "fallback": tmp_path / "fallback",
    }
    components = {
        label: _component(label, path) for label, path in component_paths.items()
    }
    components["fallback"].manifest["implementation_diagnostic"][
        "implementation_tree_sha256"
    ] = "0" * 64
    schedule_rows = [
        (injection_id, "primary" if injection_id < 92 else "fallback")
        for injection_id in range(common.PAPER_PP_RECOVERIES)
    ]
    schedule_path = _write_schedule(tmp_path / "schedule.csv", schedule_rows)
    monkeypatch.setattr(
        compose_module,
        "load_component",
        lambda label, _path: components[label],
    )
    validated = {
        injection_id: compose_module.merge_module.ValidatedResult(
            injection_id=injection_id,
            directory=tmp_path,
            fingerprint=f"{injection_id:064x}",
            file_hashes={},
            posterior_sha256=f"{injection_id + 100:064x}",
            posterior_bytes=1,
            posterior_samples=2,
            post_jit_sampling_seconds=1.0,
        )
        for injection_id in range(common.PAPER_PP_RECOVERIES)
    }
    rank_rows = [
        {
            "injection_id": injection_id,
            **{
                name: (injection_id + 0.5) / common.PAPER_PP_RECOVERIES
                for name in common.PARAMETERS
            },
            "_legacy_phase_gauge_corrected_parameters": [],
        }
        for injection_id in range(common.PAPER_PP_RECOVERIES)
    ]
    input_hashes = {
        f"{label}/results/injection-{injection_id:03d}/summary.json": "f" * 64
        for injection_id, label in schedule_rows
    }
    monkeypatch.setattr(
        compose_module,
        "_validate_scheduled_results",
        lambda *_args: validated,
    )
    monkeypatch.setattr(
        compose_module,
        "_rank_rows",
        lambda *_args: (rank_rows, input_hashes),
    )
    monkeypatch.setattr(compose_module, "_revalidate_sources", lambda *_args: None)
    output = tmp_path / "composition"

    report = compose_module.compose_blocking_results(
        output,
        component_paths,
        schedule_path,
    )

    assessment = report["remediation_assessment"]
    assert assessment["eligible"] is True
    assert assessment["parameters"] == list(common.PARAMETERS)
    assert assessment["excluded_parameters"] == []
    assert "q" in assessment["parameters"]
    assert assessment["combined_test"]["number_of_pvalues"] == 15
    assert assessment["combined_test"]["degrees_of_freedom"] == 30
    composition = json.loads((output / "composition.json").read_text(encoding="utf-8"))
    assert composition["schedule"]["assignments"] == [
        {"injection_id": injection_id, "component": label}
        for injection_id, label in schedule_rows
    ]
    assert (
        composition["components"]["fallback"]["implementation"][
            "implementation_tree_sha256"
        ]
        == "0" * 64
    )
    assert (
        composition["components"]["primary"]["implementation"][
            "implementation_tree_sha256"
        ]
        == "d" * 64
    )
    compatibility = composition["compatibility"]
    assert compatibility["allowed_execution_provenance_differences"] == [
        "implementation_tree_sha256"
    ]
    assert compatibility["implementation_tree_sha256_by_component"] == {
        "fallback": "0" * 64,
        "primary": "d" * 64,
    }
    assert compatibility["implementation_tree_homogeneous"] is False
    assert compatibility["implementation_attribution"] == (
        "This composition combines independently validated component execution trees "
        "and cannot be attributed to one homogeneous implementation."
    )
    assert len(composition["result_sources"]) == common.PAPER_PP_RECOVERIES
    assert (output / "pp/report.json").is_file()
    assert (output / "pp/pp-combined.png").is_file()
