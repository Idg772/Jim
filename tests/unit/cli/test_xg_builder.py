"""End-to-end regression for the qualified XG CLI builder path."""

import hashlib
import json
import math
from pathlib import Path

import jax.numpy as jnp

from jimgw.cli._config import PipelineConfig
from jimgw.cli._data import build_data
from jimgw.cli._likelihood import build_likelihood, detector_metadata_sha256
from jimgw.cli._prior import build_prior
from jimgw.cli._transforms import infer_likelihood_transforms
from jimgw.cli._waveform import build_waveform
from jimgw.cli.xg_qualification import build_xg_qualification_manifest
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures"

REFERENCE_PARAMETERS = {
    "M_c": 30.0,
    "eta": 0.24,
    "s1_z": 0.01,
    "s2_z": -0.02,
    "d_L": 400.0,
    "phase_c": 0.1,
    "iota": 0.4,
    "ra": 1.375,
    "dec": -1.2108,
    "psi": 0.2,
    "t_c": 0.0,
}


def _raw_xg_config(tmp_path: Path) -> dict:
    return {
        "seed": 17,
        "data": {
            "type": "file",
            "detectors": ["H1"],
            "trigger_time": 1_126_259_462.4,
            "strain_files": {"H1": str(FIXTURES_DIR / "GW150914_strain_H1.npz")},
            "psd_files": {"H1": str(FIXTURES_DIR / "GW150914_psd_H1.npz")},
        },
        "waveform": {"approximant": "IMRPhenomD", "f_ref": 20.0},
        "prior": {
            "M_c": {"type": "uniform", "min": 25.0, "max": 35.0},
            "eta": {"type": "uniform", "min": 0.20, "max": 0.25},
            "s1_z": {"type": "uniform", "min": -0.1, "max": 0.1},
            "s2_z": {"type": "uniform", "min": -0.1, "max": 0.1},
            "d_L": {"type": "uniform", "min": 100.0, "max": 1_000.0},
            "phase_c": {"type": "uniform", "min": 0.0, "max": 2.0 * math.pi},
            "iota": {"type": "sine"},
            "ra": {"type": "uniform", "min": 0.0, "max": 2.0 * math.pi},
            "dec": {"type": "cosine"},
            "psi": {"type": "uniform", "min": 0.0, "max": math.pi},
            "t_c": {"type": "uniform", "min": -0.05, "max": 0.05},
        },
        "sampling": {"time_frame": "geocentric", "sky_frame": "geocentric"},
        "likelihood": {
            "f_min": 20.0,
            "f_max": 128.0,
            "time_dependent_response": True,
            "finite_arm_response": True,
            "heterodyne": {
                "n_bins": 16,
                "reference_parameters": {
                    "type": "provided",
                    "values": REFERENCE_PARAMETERS,
                },
            },
        },
        "sampler": {"type": "flowmc"},
        "output": {"dir": str(tmp_path / "output")},
    }


def _write_json(path: Path, payload: dict) -> str:
    content = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _evidence(
    bundle_dir: Path,
    stem: str,
    qualification_kind: str,
    case_roles: list[str],
    metrics: dict[str, float],
) -> dict[str, str]:
    case_count = len(case_roles)
    case_ids = [f"{stem}-{index:03d}" for index in range(case_count)]
    case_file = f"{stem}-cases.json"
    result_file = f"{stem}-results.json"
    generator_file = f"{stem}-generator.py"
    case_digest = _write_json(
        bundle_dir / case_file,
        {
            "schema_version": 1,
            "artifact_kind": "xg-case-dataset",
            "qualification_kind": qualification_kind,
            "cases": [
                {
                    "case_id": case_id,
                    "parameters": {
                        "detectors": ["H1"],
                        "f_min": 20.0,
                        "f_max": 128.0,
                        "network_snr": 1_000.0,
                        "sidereal_epoch_index": index,
                        "ra": REFERENCE_PARAMETERS["ra"],
                        "dec": REFERENCE_PARAMETERS["dec"],
                        "psi": REFERENCE_PARAMETERS["psi"],
                        "duration_s": 4.0,
                        "prior_extreme": qualification_kind == "xg-validation-corpus"
                        and index == 0,
                        "detector_null": qualification_kind == "xg-validation-corpus"
                        and index == 1,
                        "case_role": case_roles[index],
                    },
                }
                for index, case_id in enumerate(case_ids)
            ],
        },
    )
    result_digest = _write_json(
        bundle_dir / result_file,
        {
            "schema_version": 1,
            "artifact_kind": "xg-raw-results",
            "qualification_kind": qualification_kind,
            "results": [
                {"case_id": case_id, "passed": True, "metrics": metrics}
                for case_id in case_ids
            ],
        },
    )
    generator_content = b"# independent frozen test generator\n"
    (bundle_dir / generator_file).write_bytes(generator_content)
    return {
        "case_dataset_file": case_file,
        "case_dataset_sha256": case_digest,
        "raw_results_file": result_file,
        "raw_results_sha256": result_digest,
        "generator_source_file": generator_file,
        "generator_source_sha256": hashlib.sha256(generator_content).hexdigest(),
    }


def _write_qualification_receipts(bundle_dir: Path, bin_edges_sha256: str) -> None:
    receipts = {
        "validation-corpus.json": {
            **_evidence(
                bundle_dir,
                "validation-corpus",
                "xg-validation-corpus",
                ["coverage", "coverage"],
                {"coverage": 1.0},
            ),
            "schema_version": 1,
            "artifact_kind": "xg-validation-corpus",
            "generator": "independent-test-corpus",
            "case_count": 2,
            "detectors": ["H1"],
            "f_min": 20.0,
            "f_max": 128.0,
            "max_network_snr": 1_000.0,
            "sidereal_epoch_count": 2,
            "includes_detector_nulls": True,
            "includes_prior_extremes": True,
            "passed": True,
        },
        "clock-validation.json": {
            **_evidence(
                bundle_dir,
                "clock-validation",
                "xg-clock-validation",
                ["phase-derivative", "near-merger"],
                {
                    "abs_timing_error_s": 1.0e-6,
                    "component_delta_log_l": 0.003,
                },
            ),
            "schema_version": 1,
            "artifact_kind": "xg-clock-validation",
            "implementation_name": "independent-phase-derivative",
            "implementation_sha256": "1" * 64,
            "case_count": 2,
            "phase_derivative_case_count": 1,
            "near_merger_case_count": 1,
            "max_abs_timing_error_s": 1.0e-6,
            "timing_error_budget_s": 2.0e-6,
            "max_component_delta_log_l": 0.003,
            "post_cutoff_nonnegative": True,
            "post_cutoff_monotonic": True,
            "passed": True,
        },
        "independent-response.json": {
            **_evidence(
                bundle_dir,
                "independent-response",
                "xg-independent-response",
                ["response"],
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
            "case_count": 1,
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
            **_evidence(
                bundle_dir,
                "orbital-validation",
                "xg-orbital-validation",
                ["orbital"],
                {
                    "profiled_delta_log_l": 0.003,
                    "projected_bias_sigma": 0.05,
                },
            ),
            "schema_version": 1,
            "artifact_kind": "xg-orbital-validation",
            "case_count": 1,
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
            **_evidence(
                bundle_dir,
                "compression-validation",
                "xg-compression-validation",
                ["compression"],
                {
                    "component_delta_log_l": 0.005,
                    "combined_delta_log_l": 0.02,
                },
            ),
            "schema_version": 1,
            "artifact_kind": "xg-compression-validation",
            "case_count": 1,
            "bin_edges_sha256": bin_edges_sha256,
            "dense_standard_likelihood_oracle": True,
            "direct_time_quadrature_converged": True,
            "max_component_delta_log_l": 0.005,
            "max_combined_delta_log_l": 0.02,
            "passed": True,
        },
    }
    receipts["clock-validation.json"]["implementation_sha256"] = receipts[
        "clock-validation.json"
    ]["generator_source_sha256"]
    receipts["independent-response.json"]["implementation_sha256"] = receipts[
        "independent-response.json"
    ]["generator_source_sha256"]
    for filename, receipt in receipts.items():
        _write_json(bundle_dir / filename, receipt)


def test_validated_xg_cli_builds_real_heterodyned_likelihood(tmp_path: Path) -> None:
    raw = _raw_xg_config(tmp_path)
    preflight = PipelineConfig.model_validate(
        raw,
        context={"prepare_xg_qualification": True},
    )

    source_waveform = build_waveform(preflight.waveform)
    baseline_ifos = build_data(
        preflight.data,
        f_min=preflight.likelihood.f_min,
        f_max=preflight.likelihood.f_max,
        waveform=source_waveform,
    )
    baseline = HeterodynedTransientLikelihoodFD(
        detectors=baseline_ifos,
        waveform=source_waveform,
        f_min=preflight.likelihood.f_min,
        f_max=preflight.likelihood.f_max,
        trigger_time=preflight.data.trigger_time,
        n_bins=preflight.likelihood.heterodyne.n_bins,
        reference_parameters=REFERENCE_PARAMETERS,
    )

    _write_qualification_receipts(tmp_path, baseline.bin_edges_sha256)
    manifest = build_xg_qualification_manifest(preflight, tmp_path)
    manifest_path = tmp_path / "xg-qualification.json"
    manifest_bytes = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    heterodyne_raw = raw["likelihood"]["heterodyne"]
    heterodyne_raw["qualification_manifest"] = str(manifest_path)
    heterodyne_raw["qualification_manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()

    cfg = PipelineConfig.model_validate(raw)
    waveform = DominantModeTimeCachedWaveform(build_waveform(cfg.waveform))
    ifos = build_data(
        cfg.data,
        f_min=cfg.likelihood.f_min,
        f_max=cfg.likelihood.f_max,
        waveform=waveform,
        time_frame=cfg.sampling.time_frame,
        time_dependent_response=cfg.likelihood.time_dependent_response,
        finite_arm_response=cfg.likelihood.finite_arm_response,
        seed=cfg.seed,
        input_provenance_sha256=cfg.verified_xg_manifest.input_files_sha256,
    )
    prior = build_prior(cfg.prior)
    likelihood_transforms = infer_likelihood_transforms(
        frozenset(prior.parameter_names),
        cfg.data.trigger_time,
        ifos,
        cfg.sampling,
        cfg.waveform.f_ref,
        phase_marginalization=cfg.likelihood.phase_marginalization,
    )
    likelihood = build_likelihood(
        cfg,
        ifos,
        waveform,
        prior,
        likelihood_transforms,
    )

    assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)
    assert cfg.verified_xg_manifest is not None
    assert (
        cfg.verified_xg_manifest.detector_metadata_sha256
        == detector_metadata_sha256(ifos)
    )
    assert likelihood.bin_edges_sha256 == baseline.bin_edges_sha256
    assert bool(jnp.isfinite(likelihood.evaluate(REFERENCE_PARAMETERS)))
