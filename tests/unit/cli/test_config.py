"""Unit tests for CLI config schema (TOML round-trips, validation, prior parsing)."""

import hashlib
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jimgw.cli._config import (
    CLIOptimizerRefParams,
    CosineSpec,
    FileDataConfig,
    GaussianSpec,
    GWOSCDataConfig,
    InjectionDataConfig,
    LikelihoodConfig,
    PipelineConfig,
    PowerLawSpec,
    PriorConfig,
    RayleighSpec,
    SineSpec,
    UniformSpec,
    WaveformConfig,
    XGOrbitalValidationReceipt,
    xg_implementation_sha256,
    xg_runtime_environment_sha256,
    xg_source_revision,
)

_MINIMAL_RAW = {
    "data": {
        "type": "gwosc",
        "detectors": ["H1", "L1"],
        "trigger_time": 1126259462.4,
        "duration": 4.0,
        "psd_duration": 1024.0,
    },
    "waveform": {"approximant": "IMRPhenomXAS"},
    "prior": {
        "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
        "q": {"type": "uniform", "min": 0.125, "max": 1.0},
    },
    "likelihood": {"f_min": 20.0, "f_max": 1024.0},
    "sampler": {"type": "flowmc"},
    "output": {"dir": "tests/tmp/test"},
}


def _xg_file_raw() -> dict:
    return {
        **_MINIMAL_RAW,
        "data": {
            "type": "file",
            "detectors": ["H1", "L1"],
            "trigger_time": 1126259462.4,
            "strain_files": {
                "H1": "tests/fixtures/GW150914_strain_H1.npz",
                "L1": "tests/fixtures/GW150914_strain_L1.npz",
            },
            "psd_files": {
                "H1": "tests/fixtures/GW150914_psd_H1.npz",
                "L1": "tests/fixtures/GW150914_psd_L1.npz",
            },
        },
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "time_dependent_response": True,
            "finite_arm_response": True,
            "phase_marginalization": True,
            "time_marginalization": {
                "tc_range": [-0.03, 0.03],
                "upsample_factor": 64,
                "phasor_block_size": 32,
                "freeze_response": True,
                "timing_sigma_s": 4.0e-5,
                "samples_per_timing_sigma": 4,
                "normalization": "window",
            },
            "heterodyne": {
                "n_bins": 50_000,
                "reference_chunk_size": 131_072,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }


def _add_valid_xg_manifest(tmp_path, raw: dict) -> Path:
    preflight = PipelineConfig.model_validate(
        raw,
        context={"prepare_xg_qualification": True},
    )

    def evidence(
        stem: str,
        qualification_kind: str,
        case_count: int,
        metrics: dict[str, float],
    ) -> dict[str, str]:
        case_ids = [f"{stem}-{index:03d}" for index in range(case_count)]

        def case_role(index: int) -> str:
            if qualification_kind == "xg-validation-corpus":
                return "coverage"
            if qualification_kind == "xg-clock-validation":
                return "phase-derivative" if index < 12 else "near-merger"
            return {
                "xg-independent-response": "response",
                "xg-orbital-validation": "orbital",
                "xg-compression-validation": "compression",
            }[qualification_kind]

        case_dataset = {
            "schema_version": 1,
            "artifact_kind": "xg-case-dataset",
            "qualification_kind": qualification_kind,
            "cases": [
                {
                    "case_id": case_id,
                    "parameters": {
                        "detectors": ["H1", "L1"],
                        "f_min": 5.0,
                        "f_max": 2048.0,
                        "network_snr": 2000.0,
                        "sidereal_epoch_index": index % 4,
                        "ra": float(index) * 0.1,
                        "dec": 0.0,
                        "psi": 0.0,
                        "duration_s": 7200.0,
                        "prior_extreme": index == 0,
                        "detector_null": index == 1,
                        "case_role": case_role(index),
                        "case_index": index,
                    },
                }
                for index, case_id in enumerate(case_ids)
            ],
        }
        raw_results = {
            "schema_version": 1,
            "artifact_kind": "xg-raw-results",
            "qualification_kind": qualification_kind,
            "results": [
                {"case_id": case_id, "passed": True, "metrics": metrics}
                for case_id in case_ids
            ],
        }
        files = {
            "case_dataset": (
                f"{stem}-cases.json",
                json.dumps(case_dataset, sort_keys=True).encode(),
            ),
            "raw_results": (
                f"{stem}-results.json",
                json.dumps(raw_results, sort_keys=True).encode(),
            ),
            "generator_source": (
                f"{stem}-generator.py",
                f"# frozen {stem} generator\n".encode(),
            ),
        }
        fields = {}
        for field_name, (filename, content) in files.items():
            (tmp_path / filename).write_bytes(content)
            fields[f"{field_name}_file"] = filename
            fields[f"{field_name}_sha256"] = hashlib.sha256(content).hexdigest()
        return fields

    artifact_payloads = {
        "validation-corpus.json": {
            **evidence(
                "validation-corpus",
                "xg-validation-corpus",
                12,
                {"coverage": 1.0},
            ),
            "schema_version": 1,
            "artifact_kind": "xg-validation-corpus",
            "generator": "independent-corpus-builder",
            "case_count": 12,
            "detectors": ["H1", "L1"],
            "f_min": 5.0,
            "f_max": 2048.0,
            "max_network_snr": 2000.0,
            "sidereal_epoch_count": 4,
            "includes_detector_nulls": True,
            "includes_prior_extremes": True,
            "passed": True,
        },
        "clock-validation.json": {
            **evidence(
                "clock-validation",
                "xg-clock-validation",
                18,
                {
                    "abs_timing_error_s": 1.0e-6,
                    "component_delta_log_l": 0.003,
                },
            ),
            "schema_version": 1,
            "artifact_kind": "xg-clock-validation",
            "implementation_name": "independent-phase-derivative",
            "implementation_sha256": "1" * 64,
            "case_count": 18,
            "phase_derivative_case_count": 12,
            "near_merger_case_count": 6,
            "max_abs_timing_error_s": 1.0e-6,
            "timing_error_budget_s": 2.0e-6,
            "max_component_delta_log_l": 0.003,
            "post_cutoff_nonnegative": True,
            "post_cutoff_monotonic": True,
            "passed": True,
        },
        "independent-response.json": {
            **evidence(
                "independent-response",
                "xg-independent-response",
                8,
                {
                    "numerical_delta_log_l": 0.001,
                    "component_delta_log_l": 0.004,
                    "combined_delta_log_l": 0.015,
                },
            ),
            "schema_version": 1,
            "artifact_kind": "xg-independent-response",
            "oracle_kind": "retarded-worldline-round-trip",
            "implementation_name": "independent-time-domain",
            "implementation_sha256": "2" * 64,
            "case_count": 8,
            "includes_dynamic_delay": True,
            "includes_finite_arm": True,
            "sample_rate_converged": True,
            "interpolation_converged": True,
            "max_numerical_delta_log_l": 0.001,
            "max_component_delta_log_l": 0.004,
            "max_combined_delta_log_l": 0.015,
            "passed": True,
        },
        "orbital-validation.json": {
            **evidence(
                "orbital-validation",
                "xg-orbital-validation",
                8,
                {
                    "profiled_delta_log_l": 0.003,
                    "projected_bias_sigma": 0.05,
                },
            ),
            "schema_version": 1,
            "artifact_kind": "xg-orbital-validation",
            "case_count": 8,
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
        "compression-validation.json": {
            **evidence(
                "compression-validation",
                "xg-compression-validation",
                12,
                {
                    "component_delta_log_l": 0.005,
                    "combined_delta_log_l": 0.02,
                    "frozen_response_delta_log_l": 0.004,
                    "timing_sigma_s": 4.0e-5,
                },
            ),
            "schema_version": 1,
            "artifact_kind": "xg-compression-validation",
            "case_count": 12,
            "bin_edges_sha256": "b" * 64,
            "dense_standard_likelihood_oracle": True,
            "direct_time_quadrature_converged": True,
            "max_component_delta_log_l": 0.005,
            "max_combined_delta_log_l": 0.02,
            "max_frozen_response_delta_log_l": 0.004,
            "timing_sigma_s": 4.0e-5,
            "passed": True,
        },
    }
    artifact_payloads["clock-validation.json"]["implementation_sha256"] = (
        artifact_payloads["clock-validation.json"]["generator_source_sha256"]
    )
    artifact_payloads["independent-response.json"]["implementation_sha256"] = (
        artifact_payloads["independent-response.json"]["generator_source_sha256"]
    )
    artifact_hashes = {}
    for filename, payload in artifact_payloads.items():
        content = json.dumps(payload, sort_keys=True).encode()
        artifact_path = tmp_path / filename
        artifact_path.write_bytes(content)
        artifact_hashes[filename] = hashlib.sha256(content).hexdigest()

    manifest_path = tmp_path / "xg-qualification.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_revision": xg_source_revision(),
                "implementation_sha256": xg_implementation_sha256(),
                "runtime_environment_sha256": xg_runtime_environment_sha256(),
                "analysis_contract_sha256": preflight.xg_analysis_contract_sha256(),
                "input_files_sha256": preflight.xg_input_files_sha256(),
                "detector_metadata_sha256": "d" * 64,
                "bin_edges_sha256": "b" * 64,
                "validation_corpus_file": "validation-corpus.json",
                "validation_corpus_sha256": artifact_hashes["validation-corpus.json"],
                "clock_validation_file": "clock-validation.json",
                "clock_validation_sha256": artifact_hashes["clock-validation.json"],
                "independent_response_file": "independent-response.json",
                "independent_response_sha256": artifact_hashes[
                    "independent-response.json"
                ],
                "orbital_validation_file": "orbital-validation.json",
                "orbital_validation_sha256": artifact_hashes["orbital-validation.json"],
                "compression_validation_file": "compression-validation.json",
                "compression_validation_sha256": artifact_hashes[
                    "compression-validation.json"
                ],
                "waveform_approximant": "IMRPhenomXAS",
                "detectors": ["H1", "L1"],
                "f_min": 5.0,
                "f_max": 2048.0,
                "n_bins": 50_000,
                "time_dependent_response": True,
                "finite_arm_response": True,
                "max_network_snr": 2000.0,
                "max_component_delta_log_l": 0.005,
                "max_combined_delta_log_l": 0.02,
                "max_frozen_response_delta_log_l": 0.004,
                "timing_sigma_s": 4.0e-5,
                "float_precision": "float64",
            }
        )
    )
    heterodyne = raw["likelihood"]["heterodyne"]
    heterodyne["qualification_manifest"] = str(manifest_path)
    heterodyne["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    return manifest_path


def test_pipeline_config_minimal():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert isinstance(cfg.data, GWOSCDataConfig)
    assert cfg.data.detectors == ["H1", "L1"]
    assert cfg.waveform.approximant == "IMRPhenomXAS"
    assert cfg.waveform.f_ref == 20.0  # default
    assert cfg.seed == 0  # default


def test_pipeline_config_file_data():
    raw = {
        **_MINIMAL_RAW,
        "data": {
            "type": "file",
            "detectors": ["H1"],
            "trigger_time": 1126259462.4,
            "strain_files": {"H1": "tests/fixtures/GW150914_strain_H1.npz"},
            "psd_files": {"H1": "tests/fixtures/GW150914_psd_H1.npz"},
        },
    }
    cfg = PipelineConfig.model_validate(raw)
    assert isinstance(cfg.data, FileDataConfig)


def test_et_file_data_requires_distinct_strain_and_allows_shared_psd():
    cfg = FileDataConfig(
        detectors=["ET"],
        trigger_time=1126259462.4,
        strain_files={
            "ET1": "et1.npz",
            "ET2": "et2.npz",
            "ET3": "et3.npz",
        },
        psd_files={"ET": "et-psd.npz"},
    )
    assert set(cfg.strain_files) == {"ET1", "ET2", "ET3"}

    with pytest.raises(ValidationError, match="ET2"):
        FileDataConfig(
            detectors=["ET"],
            trigger_time=1126259462.4,
            strain_files={"ET1": "et1.npz", "ET3": "et3.npz"},
            psd_files={"ET": "et-psd.npz"},
        )


def test_pipeline_config_injection_data():
    raw = {
        **_MINIMAL_RAW,
        "data": {
            "type": "injection",
            "detectors": ["H1"],
            "trigger_time": 1126259462.4,
            "duration": 4.0,
            "sampling_frequency": 2048.0,
            "injection_parameters": {
                "M_c": 28.3,
                "q": 0.85,
                "s1_z": 0.0,
                "s2_z": 0.0,
                "iota": 0.4,
                "d_L": 440.0,
                "t_c": 0.0,
                "phase_c": 0.0,
                "psi": 0.0,
                "ra": 1.375,
                "dec": -1.21,
            },
        },
    }
    cfg = PipelineConfig.model_validate(raw)
    assert isinstance(cfg.data, InjectionDataConfig)
    assert cfg.data.zero_noise is False  # default
    assert cfg.data.waveform_chunk_size == 262_144


def test_injection_waveform_chunk_size_rejects_boolean():
    raw = {
        **_MINIMAL_RAW,
        "data": {
            "type": "injection",
            "detectors": ["H1"],
            "trigger_time": 1126259462.4,
            "duration": 4.0,
            "sampling_frequency": 2048.0,
            "injection_parameters": {"M_c": 28.3},
            "waveform_chunk_size": True,
        },
    }

    with pytest.raises(ValidationError, match="waveform_chunk_size"):
        PipelineConfig.model_validate(raw)


def test_prior_spec_uniform():
    cfg = PriorConfig.model_validate(
        {"M_c": {"type": "uniform", "min": 10.0, "max": 80.0}}
    )
    spec = cfg.root["M_c"]
    assert isinstance(spec, UniformSpec)
    assert spec.min == 10.0
    assert spec.max == 80.0


def test_prior_spec_rayleigh():
    cfg = PriorConfig.model_validate({"sigma": {"type": "rayleigh", "scale": 15.0}})
    spec = cfg.root["sigma"]
    assert isinstance(spec, RayleighSpec)
    assert spec.scale == 15.0


def test_prior_spec_gaussian():
    cfg = PriorConfig.model_validate(
        {"x": {"type": "gaussian", "loc": 2.0, "scale": 0.5}}
    )
    spec = cfg.root["x"]
    assert isinstance(spec, GaussianSpec)
    assert spec.loc == 2.0
    assert spec.scale == 0.5


def test_prior_spec_sine():
    cfg = PriorConfig.model_validate({"iota": {"type": "sine"}})
    assert isinstance(cfg.root["iota"], SineSpec)


def test_prior_spec_cosine():
    cfg = PriorConfig.model_validate({"dec": {"type": "cosine"}})
    assert isinstance(cfg.root["dec"], CosineSpec)


def test_prior_spec_power_law():
    cfg = PriorConfig.model_validate(
        {"d_L": {"type": "power_law", "min": 1.0, "max": 2000.0, "alpha": 2.0}}
    )
    spec = cfg.root["d_L"]
    assert isinstance(spec, PowerLawSpec)
    assert spec.min == 1.0
    assert spec.max == 2000.0
    assert spec.alpha == 2.0


def test_prior_insertion_order_preserved():
    params = ["d_L", "M_c", "q", "iota", "dec"]
    raw_prior = {
        "d_L": {"type": "power_law", "min": 1.0, "max": 2000.0, "alpha": 2.0},
        "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
        "q": {"type": "uniform", "min": 0.125, "max": 1.0},
        "iota": {"type": "sine"},
        "dec": {"type": "cosine"},
    }
    cfg = PriorConfig.model_validate(raw_prior)
    assert list(cfg.root.keys()) == params


def test_unknown_approximant_rejected():
    raw = {**_MINIMAL_RAW, "waveform": {"approximant": "NonExistent"}}
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)


def test_extra_fields_rejected():
    raw = {**_MINIMAL_RAW, "unknown_section": {"foo": 1}}
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)


def test_dump_resolved_round_trip():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    dumped = cfg.model_dump(mode="json")
    # Strip inactive FlowMC kernel sub-configs (mirrors _output.py logic)
    if dumped.get("sampler", {}).get("type") == "flowmc":
        active = dumped["sampler"]["local_kernel"].lower()
        for kernel in ("mala", "hmc", "grw"):
            if kernel != active:
                dumped["sampler"].pop(kernel, None)
    cfg2 = PipelineConfig.model_validate(dumped)
    assert cfg.waveform.approximant == cfg2.waveform.approximant
    assert cfg.seed == cfg2.seed


def test_file_config_from_toml():
    with open("tests/fixtures/GW150914_test.toml", "rb") as f:
        raw = tomllib.load(f)
    cfg = PipelineConfig.model_validate(raw)
    assert isinstance(cfg.data, FileDataConfig)
    assert cfg.data.trigger_time == 1126259462.4
    assert len(cfg.prior.root) == 11
    assert cfg.sampler.type == "flowmc"


def test_sampling_config_defaults():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert cfg.sampling.time_frame == "detector"
    assert cfg.sampling.sky_frame == "detector"
    assert cfg.sampling.inclination_coordinate == "iota"


def test_likelihood_config_values():
    cfg = PipelineConfig.model_validate(_MINIMAL_RAW)
    assert cfg.likelihood.f_min == 20.0
    assert cfg.likelihood.f_max == 1024.0
    assert cfg.likelihood.time_dependent_response is False
    assert cfg.likelihood.finite_arm_response is False
    assert cfg.likelihood.phase_marginalization is False
    assert cfg.likelihood.time_marginalization is None
    assert cfg.likelihood.distance_marginalization is None


@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomHM", "IMRPhenomPv2", "IMRPhenomXHM", "IMRPhenomXP", "IMRPhenomXPHM"],
)
def test_phase_marginalization_rejects_nonquadrupole_waveforms(approximant):
    raw = {
        **_MINIMAL_RAW,
        "waveform": {"approximant": approximant},
        "likelihood": {
            "f_min": 20.0,
            "f_max": 1024.0,
            "phase_marginalization": True,
        },
    }

    with pytest.raises(ValidationError, match="not analytically valid"):
        PipelineConfig.model_validate(raw)


@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomHM", "IMRPhenomPv2", "IMRPhenomXHM", "IMRPhenomXP", "SineGaussian"],
)
def test_dynamic_response_rejects_unsupported_waveform_scope(approximant):
    raw = {
        **_MINIMAL_RAW,
        "waveform": {"approximant": approximant},
        "likelihood": {
            "f_min": 5.0,
            "f_max": 1024.0,
            "time_dependent_response": True,
        },
    }

    with pytest.raises(ValidationError, match="supported dominant-mode CBC"):
        PipelineConfig.model_validate(raw)


def test_finite_arm_response_rejects_sine_gaussian_scope():
    raw = {
        **_MINIMAL_RAW,
        "waveform": {"approximant": "SineGaussian"},
        "likelihood": {
            "f_min": 5.0,
            "f_max": 1024.0,
            "finite_arm_response": True,
        },
    }

    with pytest.raises(ValidationError, match="frequency-domain CBC"):
        PipelineConfig.model_validate(raw)


def test_dense_dynamic_truth_path_does_not_require_compression_receipt():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 1024.0,
            "time_dependent_response": True,
        },
    }

    cfg = PipelineConfig.model_validate(raw)
    assert cfg.verified_xg_manifest is None


def test_heterodyne_accepts_direct_sum_time_marginalization(tmp_path):
    raw = _xg_file_raw()
    _add_valid_xg_manifest(tmp_path, raw)

    cfg = PipelineConfig.model_validate(raw)
    assert cfg.likelihood.time_marginalization is not None
    assert cfg.likelihood.time_dependent_response is True
    assert cfg.likelihood.finite_arm_response is True
    assert cfg.likelihood.time_marginalization.upsample_factor == 64
    assert cfg.likelihood.time_marginalization.phasor_block_size == 32
    assert cfg.likelihood.time_marginalization.freeze_response is True
    assert cfg.likelihood.time_marginalization.timing_sigma_s == 4.0e-5
    assert cfg.likelihood.time_marginalization.normalization == "window"
    assert cfg.likelihood.heterodyne is not None
    assert cfg.likelihood.heterodyne.reference_chunk_size == 131_072
    assert cfg.verified_xg_manifest is not None


def test_xg_manifest_assembler_builds_a_verifiable_bundle(tmp_path):
    from jimgw.cli.xg_qualification import (
        _manifest_bytes,
        build_xg_qualification_manifest,
    )

    raw = _xg_file_raw()
    _add_valid_xg_manifest(tmp_path, raw)
    heterodyne = raw["likelihood"]["heterodyne"]
    heterodyne.pop("qualification_manifest")
    heterodyne.pop("qualification_manifest_sha256")
    preflight = PipelineConfig.model_validate(
        raw,
        context={"prepare_xg_qualification": True},
    )
    manifest = build_xg_qualification_manifest(preflight, tmp_path)
    manifest_path = tmp_path / "assembled-xg-qualification.json"
    manifest_bytes = _manifest_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    heterodyne["qualification_manifest"] = str(manifest_path)
    heterodyne["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    verified = PipelineConfig.model_validate(raw)
    assert verified.verified_xg_manifest == manifest


def test_xg_manifest_is_bound_to_normalized_run_contract(tmp_path):
    raw = _xg_file_raw()
    _add_valid_xg_manifest(tmp_path, raw)
    raw["waveform"] = {"approximant": "IMRPhenomXAS", "f_ref": 30.0}

    with pytest.raises(ValidationError, match="analysis_contract_sha256"):
        PipelineConfig.model_validate(raw)


def test_xg_builder_rejects_config_mutated_after_receipt_verification(tmp_path):
    from jimgw.cli._likelihood import build_likelihood

    raw = _xg_file_raw()
    _add_valid_xg_manifest(tmp_path, raw)
    cfg = PipelineConfig.model_validate(raw)
    cfg.data.trigger_time += 1.0

    with pytest.raises(ValueError, match="changed after XG qualification"):
        build_likelihood(
            cfg,
            [],
            object(),  # type: ignore[arg-type]
            prior=None,  # type: ignore[arg-type]
            likelihood_transforms=[],
        )


def test_xg_plan_rechecks_manifest_bytes_after_validation(tmp_path):
    raw = _xg_file_raw()
    manifest_path = _add_valid_xg_manifest(tmp_path, raw)
    cfg = PipelineConfig.model_validate(raw)
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest changed after validation"):
        cfg._issue_verified_xg_plan()


def test_xg_plan_rechecks_evidence_bytes_after_validation(tmp_path):
    raw = _xg_file_raw()
    manifest_path = _add_valid_xg_manifest(tmp_path, raw)
    cfg = PipelineConfig.model_validate(raw)
    manifest = json.loads(manifest_path.read_text())
    receipt_path = tmp_path / manifest["clock_validation_file"]
    receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifacts changed after validation"):
        cfg._issue_verified_xg_plan()


def test_xg_manifest_property_is_a_defensive_copy(tmp_path):
    raw = _xg_file_raw()
    _add_valid_xg_manifest(tmp_path, raw)
    cfg = PipelineConfig.model_validate(raw)
    exposed = cfg.verified_xg_manifest
    assert exposed is not None
    exposed.input_files_sha256["unverified"] = "0" * 64

    fresh = cfg.verified_xg_manifest
    assert fresh is not None
    assert "unverified" not in fresh.input_files_sha256
    cfg._issue_verified_xg_plan()


def test_xg_runtime_hash_observes_jax_precision(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(config=SimpleNamespace(jax_enable_x64=False)),
    )
    float32_digest = xg_runtime_environment_sha256()
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(config=SimpleNamespace(jax_enable_x64=True)),
    )

    assert xg_runtime_environment_sha256() != float32_digest


def test_xg_builder_rejects_response_disabled_after_receipt_verification(tmp_path):
    from jimgw.cli._likelihood import build_likelihood

    raw = _xg_file_raw()
    _add_valid_xg_manifest(tmp_path, raw)
    cfg = PipelineConfig.model_validate(raw)
    cfg.likelihood.time_dependent_response = False
    cfg.likelihood.finite_arm_response = False

    with pytest.raises(ValueError, match="mode changed after XG qualification"):
        build_likelihood(
            cfg,
            [],
            object(),  # type: ignore[arg-type]
            prior=None,  # type: ignore[arg-type]
            likelihood_transforms=[],
        )


def test_xg_manifest_is_bound_to_input_file_contents(tmp_path):
    raw = _xg_file_raw()
    psd_copy = tmp_path / "H1-psd.npz"
    psd_copy.write_bytes(Path("tests/fixtures/GW150914_psd_H1.npz").read_bytes())
    raw["data"]["psd_files"]["H1"] = str(psd_copy)
    _add_valid_xg_manifest(tmp_path, raw)
    psd_copy.write_bytes(psd_copy.read_bytes() + b"changed")

    with pytest.raises(ValidationError, match="input_files_sha256"):
        PipelineConfig.model_validate(raw)


def test_xg_manifest_rejects_unstructured_artifact_receipt(tmp_path):
    raw = _xg_file_raw()
    manifest_path = _add_valid_xg_manifest(tmp_path, raw)
    clock_path = tmp_path / "clock-validation.json"
    clock_path.write_text("{}")
    manifest = json.loads(manifest_path.read_text())
    manifest["clock_validation_sha256"] = hashlib.sha256(
        clock_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    raw["likelihood"]["heterodyne"]["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValidationError, match="invalid XG qualification artifact"):
        PipelineConfig.model_validate(raw)


def test_xg_manifest_rejects_raw_case_count_mismatch(tmp_path):
    raw = _xg_file_raw()
    manifest_path = _add_valid_xg_manifest(tmp_path, raw)
    manifest = json.loads(manifest_path.read_text())
    receipt_path = tmp_path / manifest["clock_validation_file"]
    receipt = json.loads(receipt_path.read_text())
    cases_path = tmp_path / receipt["case_dataset_file"]
    cases = json.loads(cases_path.read_text())
    cases["cases"] = cases["cases"][:1]
    cases_path.write_text(json.dumps(cases))
    receipt["case_dataset_sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    manifest["clock_validation_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    raw["likelihood"]["heterodyne"]["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValidationError, match="case_count does not match"):
        PipelineConfig.model_validate(raw)


def test_xg_manifest_rejects_coerced_raw_result(tmp_path):
    raw = _xg_file_raw()
    manifest_path = _add_valid_xg_manifest(tmp_path, raw)
    manifest = json.loads(manifest_path.read_text())
    receipt_path = tmp_path / manifest["clock_validation_file"]
    receipt = json.loads(receipt_path.read_text())
    results_path = tmp_path / receipt["raw_results_file"]
    results = json.loads(results_path.read_text())
    results["results"][0]["passed"] = 1
    results["results"][0]["metrics"]["component_delta_log_l"] = False
    results_path.write_text(json.dumps(results))
    receipt["raw_results_sha256"] = hashlib.sha256(
        results_path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    manifest["clock_validation_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    raw["likelihood"]["heterodyne"]["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValidationError, match="invalid raw evidence"):
        PipelineConfig.model_validate(raw)


def test_xg_manifest_derives_corpus_coverage_from_raw_cases(tmp_path):
    raw = _xg_file_raw()
    manifest_path = _add_valid_xg_manifest(tmp_path, raw)
    manifest = json.loads(manifest_path.read_text())
    receipt_path = tmp_path / manifest["validation_corpus_file"]
    receipt = json.loads(receipt_path.read_text())
    cases_path = tmp_path / receipt["case_dataset_file"]
    cases = json.loads(cases_path.read_text())
    for case in cases["cases"]:
        case["parameters"]["detector_null"] = False
    cases_path.write_text(json.dumps(cases))
    receipt["case_dataset_sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    manifest["validation_corpus_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    raw["likelihood"]["heterodyne"]["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValidationError, match="raw_detector_nulls"):
        PipelineConfig.model_validate(raw)


def test_xg_manifest_requires_each_evidence_family_to_cover_run_envelope(tmp_path):
    raw = _xg_file_raw()
    manifest_path = _add_valid_xg_manifest(tmp_path, raw)
    manifest = json.loads(manifest_path.read_text())
    receipt_path = tmp_path / manifest["clock_validation_file"]
    receipt = json.loads(receipt_path.read_text())
    cases_path = tmp_path / receipt["case_dataset_file"]
    cases = json.loads(cases_path.read_text())
    for case in cases["cases"]:
        case["parameters"]["f_min"] = 20.0
    cases_path.write_text(json.dumps(cases))
    receipt["case_dataset_sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    manifest["clock_validation_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    raw["likelihood"]["heterodyne"]["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValidationError, match="clock_validation.raw_frequency_band"):
        PipelineConfig.model_validate(raw)


def test_xg_manifest_binds_independent_implementation_to_generator(tmp_path):
    raw = _xg_file_raw()
    manifest_path = _add_valid_xg_manifest(tmp_path, raw)
    manifest = json.loads(manifest_path.read_text())
    receipt_path = tmp_path / manifest["independent_response_file"]
    receipt = json.loads(receipt_path.read_text())
    receipt["implementation_sha256"] = "f" * 64
    receipt_path.write_text(json.dumps(receipt))
    manifest["independent_response_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    raw["likelihood"]["heterodyne"]["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValidationError, match="generator_source"):
        PipelineConfig.model_validate(raw)


def test_dynamic_time_marginalization_requires_frozen_response_evidence(tmp_path):
    raw = _xg_file_raw()
    manifest_path = _add_valid_xg_manifest(tmp_path, raw)
    manifest = json.loads(manifest_path.read_text())
    receipt_path = tmp_path / manifest["compression_validation_file"]
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("max_frozen_response_delta_log_l")
    receipt_path.write_text(json.dumps(receipt))
    manifest["compression_validation_sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    raw["likelihood"]["heterodyne"]["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    with pytest.raises(
        ValidationError,
        match="compression_validation.max_frozen_response_delta_log_l",
    ):
        PipelineConfig.model_validate(raw)


def test_xg_orbital_receipt_rejects_relaxed_bias_budget():
    with pytest.raises(ValidationError, match="projected_bias_budget_sigma"):
        XGOrbitalValidationReceipt.model_validate(
            {
                "case_dataset_file": "orbital-cases.json",
                "case_dataset_sha256": "a" * 64,
                "raw_results_file": "orbital-results.json",
                "raw_results_sha256": "b" * 64,
                "generator_source_file": "orbital-generator.py",
                "generator_source_sha256": "c" * 64,
                "schema_version": 1,
                "artifact_kind": "xg-orbital-validation",
                "case_count": 8,
                "full_ephemeris": True,
                "geocentric_detector_frame_convention": True,
                "constant_delay_removed": True,
                "constant_velocity_removed": True,
                "full_parameter_profiled": True,
                "max_profiled_delta_log_l": 0.003,
                "max_projected_bias_sigma": 9.0,
                "projected_bias_budget_sigma": 10.0,
                "passed": True,
            }
        )


def test_xg_builder_requires_64_bit_jax_precision(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import jimgw.cli._likelihood as likelihood_builder

    raw = _xg_file_raw()
    _add_valid_xg_manifest(tmp_path, raw)
    cfg = PipelineConfig.model_validate(raw)
    monkeypatch.setattr(
        likelihood_builder,
        "jax",
        SimpleNamespace(config=SimpleNamespace(jax_enable_x64=False)),
    )

    with pytest.raises(RuntimeError, match="64-bit precision"):
        likelihood_builder.build_likelihood(
            cfg,
            [],
            object(),  # type: ignore[arg-type]
            prior=None,  # type: ignore[arg-type]
            likelihood_transforms=[],
        )


def test_xg_manifest_builder_requires_64_bit_jax_precision(tmp_path, monkeypatch):
    from jimgw.cli.xg_qualification import build_xg_qualification_manifest

    raw = _xg_file_raw()
    _add_valid_xg_manifest(tmp_path, raw)
    heterodyne = raw["likelihood"]["heterodyne"]
    heterodyne.pop("qualification_manifest")
    heterodyne.pop("qualification_manifest_sha256")
    preflight = PipelineConfig.model_validate(
        raw,
        context={"prepare_xg_qualification": True},
    )
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(config=SimpleNamespace(jax_enable_x64=False)),
    )

    with pytest.raises(ValueError, match="64-bit precision"):
        build_xg_qualification_manifest(preflight, tmp_path)


def test_xg_builder_accepts_realized_configured_waveform_type(tmp_path):
    from jimgw.cli._likelihood import build_likelihood
    from jimgw.cli._waveform import build_waveform
    from jimgw.core.single_event.dominant_mode import (
        DominantModeTimeCachedWaveform,
    )

    raw = _xg_file_raw()
    _add_valid_xg_manifest(tmp_path, raw)
    cfg = PipelineConfig.model_validate(raw)
    waveform = DominantModeTimeCachedWaveform(build_waveform(cfg.waveform))

    with pytest.raises(ValueError, match="constructed network"):
        build_likelihood(
            cfg,
            [],
            waveform,
            prior=None,  # type: ignore[arg-type]
            likelihood_transforms=[],
        )


def test_xg_preflight_rejects_live_gwosc_substrate():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 1024.0,
            "finite_arm_response": True,
            "heterodyne": {
                "n_bins": 1024,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="immutable local data"):
        PipelineConfig.model_validate(
            raw,
            context={"prepare_xg_qualification": True},
        )


def test_time_marginalized_heterodyne_rejects_optimizer_reference():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "time_marginalization": {"tc_range": [-0.03, 0.03]},
            "heterodyne": {"n_bins": 1024},
        },
    }

    with pytest.raises(ValidationError, match="requires fixed reference_parameters"):
        PipelineConfig.model_validate(raw)


@pytest.mark.parametrize(
    "response_flag", ["time_dependent_response", "finite_arm_response"]
)
def test_xg_heterodyne_rejects_optimizer_reference_without_time_marginalization(
    response_flag,
):
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            response_flag: True,
            "heterodyne": {"n_bins": 1024},
        },
    }

    with pytest.raises(ValidationError, match="requires fixed reference_parameters"):
        PipelineConfig.model_validate(raw)


def test_xg_heterodyne_requires_explicit_prequalified_bin_count():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "finite_arm_response": True,
            "heterodyne": {
                "epsilon": 0.5,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="explicit, prequalified n_bins"):
        PipelineConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("heterodyne", "message"),
    [
        ({"n_bins": 64, "epsilon": 0.5}, "at most one"),
        ({"epsilon": float("inf")}, "epsilon must be finite"),
        ({"n_bins": 4_194_304}, "less than or equal"),
        ({"n_bins": True}, "valid integer"),
        ({"epsilon": True}, "valid number"),
    ],
)
def test_heterodyne_rejects_unbuildable_bin_plan(heterodyne, message):
    with pytest.raises(ValidationError, match=message):
        LikelihoodConfig.model_validate(
            {
                "f_min": 20.0,
                "f_max": 1024.0,
                "heterodyne": heterodyne,
            }
        )


def test_xg_heterodyne_requires_hashed_qualification_manifest():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "finite_arm_response": True,
            "heterodyne": {
                "n_bins": 50_000,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="qualification_manifest"):
        PipelineConfig.model_validate(raw)


def test_standalone_likelihood_config_cannot_forge_verified_receipt():
    cfg = LikelihoodConfig.model_validate(
        {
            "f_min": 5.0,
            "f_max": 2048.0,
            "finite_arm_response": True,
            "heterodyne": {
                "n_bins": 50_000,
                "qualification_manifest": "claimed.json",
                "qualification_manifest_sha256": "a" * 64,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        }
    )

    assert not hasattr(cfg, "verified_xg_manifest")


def test_xg_heterodyne_rejects_changed_qualification_manifest(tmp_path):
    manifest_path = tmp_path / "xg-qualification.json"
    manifest_path.write_text("{}")
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "finite_arm_response": True,
            "heterodyne": {
                "n_bins": 50_000,
                "qualification_manifest": str(manifest_path),
                "qualification_manifest_sha256": "0" * 64,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="SHA-256"):
        PipelineConfig.model_validate(raw)


def test_xg_response_rejects_unqualified_multiband_path():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "time_dependent_response": True,
            "multiband": {},
        },
    }

    with pytest.raises(ValidationError, match="not qualified with the multiband"):
        PipelineConfig.model_validate(raw)


def test_dynamic_direct_sum_requires_frozen_response_acknowledgement():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "time_dependent_response": True,
            "time_marginalization": {"tc_range": [-0.03, 0.03]},
            "heterodyne": {
                "n_bins": 1024,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="freeze_response = true"):
        PipelineConfig.model_validate(raw)


def test_dynamic_time_marginalization_requires_timing_width():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "time_dependent_response": True,
            "time_marginalization": {
                "tc_range": [-0.03, 0.03],
                "freeze_response": True,
                "normalization": "window",
            },
            "heterodyne": {
                "n_bins": 1024,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="timing_sigma_s"):
        PipelineConfig.model_validate(raw)


def test_dynamic_time_marginalization_requires_window_normalization():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "time_dependent_response": True,
            "time_marginalization": {
                "tc_range": [-0.03, 0.03],
                "freeze_response": True,
                "timing_sigma_s": 4.0e-5,
            },
            "heterodyne": {
                "n_bins": 1024,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="normalization = 'window'"):
        PipelineConfig.model_validate(raw)


def test_dynamic_frozen_response_requires_zero_inside_time_prior():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "time_dependent_response": True,
            "time_marginalization": {
                "tc_range": [0.01, 0.03],
                "freeze_response": True,
                "timing_sigma_s": 4.0e-5,
                "normalization": "window",
            },
            "heterodyne": {
                "n_bins": 1024,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="must lie inside"):
        PipelineConfig.model_validate(raw)


def test_dynamic_response_rejects_dense_time_marginalization():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 5.0,
            "f_max": 2048.0,
            "time_dependent_response": True,
            "time_marginalization": {
                "tc_range": [-0.03, 0.03],
                "timing_sigma_s": 4.0e-5,
            },
        },
    }

    with pytest.raises(ValidationError, match="cannot use dense FFT"):
        PipelineConfig.model_validate(raw)


@pytest.mark.parametrize(
    "option",
    [
        {"freeze_response": True},
        {"phasor_block_size": 32},
        {"timing_sigma_s": 1.0e-4},
        {"samples_per_timing_sigma": 8},
        {"normalization": "window"},
    ],
)
def test_dense_time_marginalization_rejects_heterodyne_only_options(option):
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 20.0,
            "f_max": 1024.0,
            "time_marginalization": {
                "tc_range": [-0.03, 0.03],
                **option,
            },
        },
    }

    with pytest.raises(ValidationError, match="heterodyne-only"):
        PipelineConfig.model_validate(raw)


@pytest.mark.parametrize("sampled_time", ["t_c", "t_det"])
def test_time_marginalization_rejects_sampled_time_parameter(sampled_time):
    raw = {
        **_MINIMAL_RAW,
        "prior": {
            **_MINIMAL_RAW["prior"],
            sampled_time: {"type": "uniform", "min": -0.03, "max": 0.03},
        },
        "likelihood": {
            "f_min": 20.0,
            "f_max": 1024.0,
            "time_marginalization": {"tc_range": [-0.03, 0.03]},
        },
    }

    with pytest.raises(ValidationError, match="removes sampled coalescence time"):
        PipelineConfig.model_validate(raw)


def test_time_marginalization_rejects_fixed_time_parameter():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 20.0,
            "f_max": 1024.0,
            "fixed_parameters": {"t_c": 0.0},
            "time_marginalization": {"tc_range": [-0.03, 0.03]},
        },
    }

    with pytest.raises(ValidationError, match="fixed_parameters.t_c"):
        PipelineConfig.model_validate(raw)


def test_time_marginalization_rejects_nonfinite_timing_width():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 20.0,
            "f_max": 1024.0,
            "time_marginalization": {
                "tc_range": [-0.03, 0.03],
                "timing_sigma_s": float("inf"),
            },
            "heterodyne": {
                "n_bins": 1024,
                "reference_parameters": {
                    "type": "provided",
                    "values": {"M_c": 1.2, "eta": 0.249},
                },
            },
        },
    }

    with pytest.raises(ValidationError, match="timing_sigma_s must be finite"):
        PipelineConfig.model_validate(raw)


def test_heterodyne_reference_chunk_size_rejects_boolean():
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 20.0,
            "f_max": 1024.0,
            "heterodyne": {"reference_chunk_size": True},
        },
    }

    with pytest.raises(ValidationError, match="reference_chunk_size"):
        PipelineConfig.model_validate(raw)


@pytest.mark.parametrize("tc_range", [[0.1, -0.1], [0.0, float("inf")]])
def test_time_marginalization_rejects_invalid_range(tc_range):
    raw = {
        **_MINIMAL_RAW,
        "likelihood": {
            "f_min": 20.0,
            "f_max": 1024.0,
            "time_marginalization": {"tc_range": tc_range},
        },
    }

    with pytest.raises(ValidationError, match="finite increasing bounds"):
        PipelineConfig.model_validate(raw)


def test_optimizer_ref_params_target_defaults_to_none():
    cfg = CLIOptimizerRefParams.model_validate({})
    assert cfg.popsize == 500
    assert cfg.n_steps == 1000
    assert cfg.target is None


def test_optimizer_ref_params_target_parses():
    cfg = CLIOptimizerRefParams.model_validate({"target": -1234.5})
    assert cfg.target == -1234.5


def test_waveform_config_f_ref_default():
    cfg = WaveformConfig.model_validate({"approximant": "IMRPhenomD"})
    assert cfg.f_ref == 20.0


def test_file_data_config_missing_strain_rejected():
    with pytest.raises(ValidationError, match="strain_files missing"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "data": {
                    "type": "file",
                    "detectors": ["H1", "L1"],
                    "trigger_time": 1126259462.4,
                    "strain_files": {"H1": "tests/fixtures/GW150914_strain_H1.npz"},
                    "psd_files": {
                        "H1": "tests/fixtures/GW150914_psd_H1.npz",
                        "L1": "tests/fixtures/GW150914_psd_L1.npz",
                    },
                },
            }
        )


def test_file_data_config_missing_psd_rejected():
    with pytest.raises(ValidationError, match="psd_files missing"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "data": {
                    "type": "file",
                    "detectors": ["H1", "L1"],
                    "trigger_time": 1126259462.4,
                    "strain_files": {
                        "H1": "tests/fixtures/GW150914_strain_H1.npz",
                        "L1": "tests/fixtures/GW150914_strain_L1.npz",
                    },
                    "psd_files": {"H1": "tests/fixtures/GW150914_psd_H1.npz"},
                },
            }
        )


# ---------------------------------------------------------------------------
# Prior spec validators
# ---------------------------------------------------------------------------


def test_uniform_spec_inverted_bounds_rejected():
    with pytest.raises(ValidationError, match="min < max"):
        PriorConfig.model_validate({"x": {"type": "uniform", "min": 5.0, "max": 1.0}})


def test_uniform_spec_equal_bounds_rejected():
    with pytest.raises(ValidationError, match="min < max"):
        PriorConfig.model_validate({"x": {"type": "uniform", "min": 3.0, "max": 3.0}})


def test_power_law_spec_inverted_bounds_rejected():
    with pytest.raises(ValidationError, match="min < max"):
        PriorConfig.model_validate(
            {"d_L": {"type": "power_law", "min": 2000.0, "max": 1.0, "alpha": 2.0}}
        )


def test_gaussian_spec_zero_scale_rejected():
    with pytest.raises(ValidationError, match="scale > 0"):
        PriorConfig.model_validate(
            {"x": {"type": "gaussian", "loc": 0.0, "scale": 0.0}}
        )


def test_gaussian_spec_negative_scale_rejected():
    with pytest.raises(ValidationError, match="scale > 0"):
        PriorConfig.model_validate(
            {"x": {"type": "gaussian", "loc": 0.0, "scale": -1.0}}
        )


def test_rayleigh_spec_zero_scale_rejected():
    with pytest.raises(ValidationError, match="scale > 0"):
        PriorConfig.model_validate({"sigma": {"type": "rayleigh", "scale": 0.0}})


def test_rayleigh_spec_negative_scale_rejected():
    with pytest.raises(ValidationError, match="scale > 0"):
        PriorConfig.model_validate({"sigma": {"type": "rayleigh", "scale": -0.5}})


# ---------------------------------------------------------------------------
# Detector list validators
# ---------------------------------------------------------------------------


def test_duplicate_detectors_rejected():
    with pytest.raises(ValidationError, match="Duplicate"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "data": {**_MINIMAL_RAW["data"], "detectors": ["H1", "H1"]},
            }
        )


# ---------------------------------------------------------------------------
# Sky parametrization completeness
# ---------------------------------------------------------------------------


def test_incomplete_equatorial_sky_rejected():
    with pytest.raises(ValidationError, match="both 'ra' and 'dec'"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "prior": {
                    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
                    "ra": {"type": "uniform", "min": 0.0, "max": 6.283},
                    # dec missing
                },
            }
        )


def test_incomplete_detector_sky_rejected():
    with pytest.raises(ValidationError, match="both 'azimuth' and 'zenith'"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "prior": {
                    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
                    "azimuth": {"type": "uniform", "min": 0.0, "max": 6.283},
                    # zenith missing
                },
                "sampling": {"sky_frame": "detector"},
            }
        )


# ---------------------------------------------------------------------------
# NS AW sampler prior constraints
# ---------------------------------------------------------------------------


def test_ns_aw_non_uniform_t_c_rejected():
    with pytest.raises(ValidationError, match="uniform"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "prior": {
                    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
                    "t_c": {"type": "gaussian", "loc": 0.0, "scale": 0.05},
                },
                "sampler": {"type": "blackjax-ns-aw"},
            }
        )


def test_ns_aw_non_uniform_t_det_rejected():
    with pytest.raises(ValidationError, match="uniform"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "prior": {
                    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
                    "t_det": {"type": "gaussian", "loc": 0.0, "scale": 0.05},
                },
                "sampler": {"type": "blackjax-ns-aw"},
            }
        )


# ---------------------------------------------------------------------------
# SwiG sampler config
# ---------------------------------------------------------------------------


def test_swig_sampler_parses_from_toml():
    from jimgw.samplers.config import BlackJAXSwiGConfig

    cfg = PipelineConfig.model_validate(
        {
            **_MINIMAL_RAW,
            "sampler": {
                "type": "blackjax-swig",
                "blocks": [["M_c"], ["q"]],
                "n_live": 4,
                "n_delete_frac": 0.5,
            },
        }
    )
    assert isinstance(cfg.sampler, BlackJAXSwiGConfig)
    assert cfg.sampler.blocks == [["M_c"], ["q"]]


def test_swig_sampler_requires_blocks():
    with pytest.raises(ValidationError, match="blocks"):
        PipelineConfig.model_validate(
            {
                **_MINIMAL_RAW,
                "sampler": {"type": "blackjax-swig"},
            }
        )
