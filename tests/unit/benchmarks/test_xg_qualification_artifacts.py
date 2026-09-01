from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.xg.qualification_artifacts import (
    QualificationArtifactError,
    QualificationCaseOutcome,
    publish_qualification_artifact,
)
from jimgw.cli._config import (
    XGCaseDataset,
    XGClockValidationReceipt,
    XGCompressionValidationReceipt,
    XGIndependentResponseReceipt,
    XGOrbitalValidationReceipt,
    XGRawResults,
    XGValidationCorpusReceipt,
)


def _case(case_id: str, role: str, index: int = 0) -> dict:
    return {
        "case_id": case_id,
        "parameters": {
            "detectors": ["CE"],
            "f_min": 5.0,
            "f_max": 2048.0,
            "network_snr": 2090.0,
            "sidereal_epoch_index": index,
            "ra": 0.1 * index,
            "dec": 0.0,
            "psi": 0.2,
            "duration_s": 8192.0,
            "prior_extreme": index == 0,
            "detector_null": index == 1,
            "case_role": role,
            "M_c": 1.18,
        },
    }


def _outcome(case_id: str, metrics: dict[str, float], *, passed: bool = True):
    return QualificationCaseOutcome(
        case_id=case_id,
        passed=passed,
        metrics=metrics,
        diagnostics={"frequency_hz": [5.0, 20.0], "residual": [0.0, 1.0e-6]},
    )


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "oracle.py"
    path.write_text("# independent oracle\nVALUE = 1\n")
    return path


@pytest.mark.parametrize(
    ("kind", "cases", "outcomes", "summary", "receipt_type", "receipt_name"),
    (
        (
            "xg-validation-corpus",
            [_case("coverage-0", "coverage", 0), _case("coverage-1", "coverage", 1)],
            [
                _outcome("coverage-0", {"coverage": 1.0}),
                _outcome("coverage-1", {"coverage": 1.0}),
            ],
            {
                "generator": "frozen-corpus-design",
                "detectors": ["CE"],
                "f_min": 5.0,
                "f_max": 2048.0,
                "max_network_snr": 2090.0,
                "sidereal_epoch_count": 2,
                "includes_detector_nulls": True,
                "includes_prior_extremes": True,
                "passed": True,
            },
            XGValidationCorpusReceipt,
            "validation-corpus.json",
        ),
        (
            "xg-clock-validation",
            [
                _case("clock-0", "phase-derivative", 0),
                _case("clock-1", "near-merger", 1),
            ],
            [
                _outcome(
                    "clock-0",
                    {"abs_timing_error_s": 1.0e-6, "component_delta_log_l": 0.003},
                ),
                _outcome(
                    "clock-1",
                    {"abs_timing_error_s": 2.0e-6, "component_delta_log_l": 0.004},
                ),
            ],
            {
                "implementation_name": "independent-phase-derivative",
                "phase_derivative_case_count": 1,
                "near_merger_case_count": 1,
                "max_abs_timing_error_s": 2.0e-6,
                "timing_error_budget_s": 3.0e-6,
                "max_component_delta_log_l": 0.004,
                "post_cutoff_nonnegative": True,
                "post_cutoff_monotonic": True,
                "passed": True,
            },
            XGClockValidationReceipt,
            "clock-validation.json",
        ),
        (
            "xg-independent-response",
            [_case("response-0", "response")],
            [
                _outcome(
                    "response-0",
                    {
                        "numerical_delta_log_l": 0.001,
                        "component_delta_log_l": 0.004,
                        "combined_delta_log_l": 0.02,
                    },
                )
            ],
            {
                "oracle_kind": "retarded-worldline-round-trip",
                "implementation_name": "independent-time-domain",
                "includes_dynamic_delay": True,
                "includes_finite_arm": True,
                "sample_rate_converged": True,
                "interpolation_converged": True,
                "max_numerical_delta_log_l": 0.001,
                "max_component_delta_log_l": 0.004,
                "max_combined_delta_log_l": 0.02,
                "passed": True,
            },
            XGIndependentResponseReceipt,
            "independent-response.json",
        ),
        (
            "xg-orbital-validation",
            [_case("orbital-0", "orbital")],
            [
                _outcome(
                    "orbital-0",
                    {"profiled_delta_log_l": 0.003, "projected_bias_sigma": 0.05},
                )
            ],
            {
                "full_ephemeris": True,
                "geocentric_detector_frame_convention": True,
                "constant_delay_removed": True,
                "constant_velocity_removed": True,
                "full_parameter_profiled": True,
                "max_profiled_delta_log_l": 0.003,
                "max_projected_bias_sigma": 0.05,
                "projected_bias_budget_sigma": 0.1,
                "passed": True,
            },
            XGOrbitalValidationReceipt,
            "orbital-validation.json",
        ),
        (
            "xg-compression-validation",
            [_case("compression-0", "compression")],
            [
                _outcome(
                    "compression-0",
                    {"component_delta_log_l": 0.005, "combined_delta_log_l": 0.02},
                )
            ],
            {
                "bin_edges_sha256": "b" * 64,
                "dense_standard_likelihood_oracle": True,
                "direct_time_quadrature_converged": True,
                "max_component_delta_log_l": 0.005,
                "max_combined_delta_log_l": 0.02,
                "passed": True,
            },
            XGCompressionValidationReceipt,
            "compression-validation.json",
        ),
    ),
)
def test_publishes_each_typed_receipt_with_bound_diagnostics(
    tmp_path: Path,
    kind: str,
    cases: list[dict],
    outcomes: list[QualificationCaseOutcome],
    summary: dict,
    receipt_type: type,
    receipt_name: str,
) -> None:
    bundle = tmp_path / "bundle"
    published = publish_qualification_artifact(
        bundle,
        qualification_kind=kind,
        cases=cases,
        outcomes=outcomes,
        summary_fields=summary,
        generator_source=_source(tmp_path),
    )

    assert published.receipt_path.name == receipt_name
    assert isinstance(published.receipt, receipt_type)
    assert published.receipt_path.read_bytes().endswith(b"\n")
    assert (
        published.receipt_sha256
        == hashlib.sha256(published.receipt_path.read_bytes()).hexdigest()
    )
    case_dataset = XGCaseDataset.model_validate_json(
        published.case_dataset_path.read_bytes()
    )
    raw_results = XGRawResults.model_validate_json(
        published.raw_results_path.read_bytes()
    )
    assert [case.case_id for case in case_dataset.cases] == sorted(
        case["case_id"] for case in cases
    )
    assert [result.case_id for result in raw_results.results] == sorted(
        outcome.case_id for outcome in outcomes
    )
    diagnostics = json.loads(published.diagnostics_path.read_bytes())
    assert diagnostics["artifact_kind"] == "xg-qualification-diagnostics"
    assert diagnostics["records"][0]["diagnostics"]["frequency_hz"] == [5.0, 20.0]
    for case in case_dataset.cases:
        payload = case.parameters.model_dump()
        assert payload["diagnostics_file"] == published.diagnostics_path.name
        assert payload["diagnostics_sha256"] == published.diagnostics_sha256
    assert (
        published.generator_source_sha256
        == hashlib.sha256(published.generator_source_path.read_bytes()).hexdigest()
    )
    if isinstance(
        published.receipt,
        (XGClockValidationReceipt, XGIndependentResponseReceipt),
    ):
        assert (
            published.receipt.implementation_sha256 == published.generator_source_sha256
        )


def test_output_is_canonical_across_input_order_and_directories(tmp_path: Path) -> None:
    cases = [_case("coverage-b", "coverage", 1), _case("coverage-a", "coverage", 0)]
    outcomes = [
        _outcome("coverage-b", {"coverage": 1.0}),
        _outcome("coverage-a", {"coverage": 1.0}),
    ]
    summary = {
        "generator": "frozen-corpus-design",
        "detectors": ["CE"],
        "f_min": 5.0,
        "f_max": 2048.0,
        "max_network_snr": 2090.0,
        "sidereal_epoch_count": 2,
        "includes_detector_nulls": True,
        "includes_prior_extremes": True,
        "passed": True,
    }
    source = _source(tmp_path)
    first = publish_qualification_artifact(
        tmp_path / "first",
        qualification_kind="xg-validation-corpus",
        cases=cases,
        outcomes=outcomes,
        summary_fields=summary,
        generator_source=source,
    )
    second = publish_qualification_artifact(
        tmp_path / "second",
        qualification_kind="xg-validation-corpus",
        cases=list(reversed(cases)),
        outcomes=list(reversed(outcomes)),
        summary_fields=summary,
        generator_source=source,
    )

    assert first.receipt_path.read_bytes() == second.receipt_path.read_bytes()
    assert first.case_dataset_path.read_bytes() == second.case_dataset_path.read_bytes()
    assert first.raw_results_path.read_bytes() == second.raw_results_path.read_bytes()
    assert first.diagnostics_path.read_bytes() == second.diagnostics_path.read_bytes()


def test_generator_bundle_authenticates_every_source_file(tmp_path: Path) -> None:
    first_source = _source(tmp_path)
    second_source = tmp_path / "campaign.py"
    second_source.write_text("# campaign logic\nVALUE = 2\n")
    published = publish_qualification_artifact(
        tmp_path / "bundle",
        qualification_kind="xg-validation-corpus",
        cases=[_case("coverage-0", "coverage", 0), _case("coverage-1", "coverage", 1)],
        outcomes=[
            _outcome("coverage-0", {"coverage": 1.0}),
            _outcome("coverage-1", {"coverage": 1.0}),
        ],
        summary_fields={
            "generator": "frozen-corpus-design",
            "detectors": ["CE"],
            "f_min": 5.0,
            "f_max": 2048.0,
            "max_network_snr": 2090.0,
            "sidereal_epoch_count": 2,
            "includes_detector_nulls": True,
            "includes_prior_extremes": True,
            "passed": True,
        },
        generator_source=(first_source, second_source),
    )

    bundle = json.loads(published.generator_source_path.read_bytes())
    assert bundle["artifact_kind"] == "xg-generator-source-bundle"
    assert [source["name"] for source in bundle["sources"]] == [
        "campaign.py",
        "oracle.py",
    ]
    assert {source["name"]: source["sha256"] for source in bundle["sources"]} == {
        first_source.name: hashlib.sha256(first_source.read_bytes()).hexdigest(),
        second_source.name: hashlib.sha256(second_source.read_bytes()).hexdigest(),
    }


def test_failed_case_publishes_nothing(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    with pytest.raises(QualificationArtifactError, match="did not pass"):
        publish_qualification_artifact(
            bundle,
            qualification_kind="xg-compression-validation",
            cases=[_case("compression-0", "compression")],
            outcomes=[
                _outcome(
                    "compression-0",
                    {"component_delta_log_l": 0.02, "combined_delta_log_l": 0.06},
                    passed=False,
                )
            ],
            summary_fields={},
            generator_source=_source(tmp_path),
        )
    assert not bundle.exists()


def test_receipt_aggregate_must_equal_raw_measurements(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    with pytest.raises(QualificationArtifactError, match="does not match raw max"):
        publish_qualification_artifact(
            bundle,
            qualification_kind="xg-compression-validation",
            cases=[_case("compression-0", "compression")],
            outcomes=[
                _outcome(
                    "compression-0",
                    {"component_delta_log_l": 0.005, "combined_delta_log_l": 0.02},
                )
            ],
            summary_fields={
                "bin_edges_sha256": "b" * 64,
                "dense_standard_likelihood_oracle": True,
                "direct_time_quadrature_converged": True,
                "max_component_delta_log_l": 0.004,
                "max_combined_delta_log_l": 0.02,
                "passed": True,
            },
            generator_source=_source(tmp_path),
        )
    assert not bundle.exists()


def test_nonfinite_diagnostics_fail_before_publication(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    outcome = QualificationCaseOutcome(
        case_id="compression-0",
        passed=True,
        metrics={"component_delta_log_l": 0.005, "combined_delta_log_l": 0.02},
        diagnostics={"bad": float("nan")},
    )
    with pytest.raises(QualificationArtifactError, match="finite JSON"):
        publish_qualification_artifact(
            bundle,
            qualification_kind="xg-compression-validation",
            cases=[_case("compression-0", "compression")],
            outcomes=[outcome],
            summary_fields={
                "bin_edges_sha256": "b" * 64,
                "dense_standard_likelihood_oracle": True,
                "direct_time_quadrature_converged": True,
                "max_component_delta_log_l": 0.005,
                "max_combined_delta_log_l": 0.02,
                "passed": True,
            },
            generator_source=_source(tmp_path),
        )
    assert not bundle.exists()


def test_existing_artifacts_are_never_replaced(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    receipt_path = bundle / "compression-validation.json"
    receipt_path.write_text("existing\n")
    with pytest.raises(QualificationArtifactError, match="already exist"):
        publish_qualification_artifact(
            bundle,
            qualification_kind="xg-compression-validation",
            cases=[_case("compression-0", "compression")],
            outcomes=[
                _outcome(
                    "compression-0",
                    {"component_delta_log_l": 0.005, "combined_delta_log_l": 0.02},
                )
            ],
            summary_fields={
                "bin_edges_sha256": "b" * 64,
                "dense_standard_likelihood_oracle": True,
                "direct_time_quadrature_converged": True,
                "max_component_delta_log_l": 0.005,
                "max_combined_delta_log_l": 0.02,
                "passed": True,
            },
            generator_source=_source(tmp_path),
        )
    assert receipt_path.read_text() == "existing\n"
