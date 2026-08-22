from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks.device_parallel_nss import run_gw170817_likelihood_pair as pair


@pytest.fixture(autouse=True)
def _restore_pair_runtime_globals() -> Iterator[None]:
    """Keep pair-runtime configuration from leaking into other test modules."""

    from benchmarks.device_parallel_nss import benchmark_gw170817_full_run as benchmark

    pair_state = {
        name: copy.deepcopy(getattr(pair, name))
        for name in (
            "BLOCKING_SCHEME",
            "DIRECTION_MODE",
            "N_DEVICES",
            "NUM_GIBBS_SWEEPS",
            "PAPER_BLOCKS",
            "FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK",
        )
    }
    benchmark_state = {
        name: copy.deepcopy(getattr(benchmark, name))
        for name in (
            "NUM_GIBBS_SWEEPS",
            "FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK",
            "PAPER_FAST_RIDGE_INTRINSIC_PERIODIC_MH_LIMITATIONS",
        )
    }
    yield
    for name, value in pair_state.items():
        setattr(pair, name, value)
    for name, value in benchmark_state.items():
        setattr(benchmark, name, value)


def _time_grid_metadata() -> dict[str, Any]:
    normalization = int(
        pair.DURATION_SECONDS
        * pair.TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ
        / 2.0
        * pair.TIME_UPSAMPLE_FACTOR
    )
    step = pair.DURATION_SECONDS / normalization
    first = math.floor(pair.TC_RANGE_SECONDS[0] / step) + 1
    last = math.ceil(pair.TC_RANGE_SECONDS[1] / step) - 1
    window = np.asarray(step * np.arange(first, last + 1), dtype="<f8")
    return {
        "tc_range_seconds": list(pair.TC_RANGE_SECONDS),
        "upsample_factor": pair.TIME_UPSAMPLE_FACTOR,
        "window_point_count": len(window),
        "window_sha256": hashlib.sha256(window.tobytes()).hexdigest(),
        "normalization_point_count": normalization,
        "resolution_seconds": step,
        "limitation": "corrective sub-grid",
    }


def _likelihood_metadata(arm: pair.LikelihoodArm, reference_sha: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": arm.kind,
        "class": (
            "benchmarks.device_parallel_nss.paper_heterodyne."
            "PaperTimeMarginalizedHeterodynedLikelihoodFD"
            if arm.compressed
            else "jimgw.core.single_event.likelihood.TransientLikelihoodFD"
        ),
        "phase_marginalization": "analytic network log-I0",
        "time_marginalization": pair.TIME_MARGINALIZATION_DESCRIPTION,
        "time_grid": _time_grid_metadata(),
        "distance_marginalization": False,
        "reference_json_sha256": reference_sha,
        "reference_used": arm.compressed,
    }
    if arm.compressed:
        value.update(
            {
                "compression": "relative-binning linear bin-edge expansion",
                "bins": {
                    "requested_bins": pair.N_BINS_REQUESTED,
                    "realized_bins": pair.N_BINS_REQUESTED,
                    "bin_edges_count": pair.N_BINS_REQUESTED + 1,
                    "bin_edges_dtype": "float64-little-endian",
                    "bin_edges_sha256": "b" * 64,
                    "first_edge_hz": 20.0,
                    "last_edge_hz": 2047.0,
                    "reference_denominator_min_abs_by_detector": {
                        "H1": 0.1,
                        "L1": 0.1,
                        "V1": 0.1,
                    },
                },
                "coefficient_builder": {
                    "qualified_name": (
                        "benchmarks.device_parallel_nss.paper_heterodyne."
                        "PaperTimeMarginalizedHeterodynedLikelihoodFD."
                        "_compute_coefficients"
                    ),
                    "algorithm": "numpy-segmented-v1",
                    "source_sha256": "c" * 64,
                },
            }
        )
    else:
        value.update({"compression": None, "bins": None, "coefficient_builder": None})
    return value


def _weighted(path: Path, *, q_shift: float = 0.0) -> None:
    count = pair.N_LIVE + pair.PREFIX_STEPS * pair.N_DELETE
    arrays = {
        name: np.linspace(0.1, 0.2, count, dtype=np.float64)
        for name in pair.POSITION_FIELDS
    }
    deaths = np.linspace(1.0, 2.0 + q_shift, count)
    births = deaths - 0.1
    births[: pair.N_LIVE] = -np.inf
    arrays.update(
        {
            "q": np.linspace(0.5, 0.7 + q_shift, count),
            "d_L": np.linspace(20.0, 30.0, count),
            "log_likelihood": deaths,
            "log_likelihood_birth": births,
            "log_weights": np.full(count, -math.log(count)),
        }
    )
    np.savez(path, **arrays)


def _report(
    arm: pair.LikelihoodArm,
    weighted: Path,
    *,
    data_sha: str = "d" * 64,
    reference_sha: str = "r" * 64,
    source_sha: str = "s" * 64,
) -> dict[str, Any]:
    config = {
        "seed": 2,
        "workload": pair.WORKLOAD,
        "blocking_scheme": pair.BLOCKING_SCHEME,
        "waveform": pair.WAVEFORM,
        "waveform_f_ref_hz": pair.WAVEFORM_F_REF_HZ,
        "carrier_time_anchor": pair.CARRIER_TIME_ANCHOR,
        "sampled_dimensions": 15,
        "blocks": [list(block) for block in pair.PAPER_BLOCKS],
        "priors": {
            "M_c": {"range": [1.18, 1.21]},
            "q": {"range": [0.125, 1.0]},
            "d_L": {"range_mpc": [1.0, 75.0]},
        },
        "phase_marginalization": True,
        "time_marginalization_tc_range_seconds": list(pair.TC_RANGE_SECONDS),
        "time_marginalization_fft_sample_rate_hz": (
            pair.TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ
        ),
        "distance_marginalization": False,
        "n_devices": pair.N_DEVICES,
        "n_live": pair.N_LIVE,
        "n_delete": pair.N_DELETE,
        "num_inner_steps_per_dim": pair.NUM_INNER_STEPS_PER_DIM,
        "num_gibbs_sweeps": pair.NUM_GIBBS_SWEEPS,
        "termination_dlogz": pair.TERMINATION_DLOGZ,
        "dtype": "float64",
        "paper_notation": {
            "D_devices": pair.N_DEVICES,
            "M_gibbs_sweeps": pair.NUM_GIBBS_SWEEPS,
        },
        "initial_positions_sha256": "initial",
        "likelihood": _likelihood_metadata(arm, reference_sha),
    }
    if pair.DIRECTION_MODE != "covariance":
        config["direction_mode"] = pair.DIRECTION_MODE
    if pair.BLOCKING_SCHEME == pair.FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME:
        config["num_slice_steps_by_block"] = list(
            pair.FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE
        )
    hybrid_scheme = pair.BLOCKING_SCHEME in pair.PERIODIC_MH_BLOCKING_SCHEMES
    complementary_de_scheme = (
        pair.BLOCKING_SCHEME in pair.COMPLEMENTARY_DE_BLOCK_BY_SCHEME
    )
    if hybrid_scheme:
        config["block_kernel_modes"] = list(
            pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
        )
        config["fixed_work"] = dict(
            pair.COMPLEMENTARY_DE_FIXED_WORK_BY_SCHEME[pair.BLOCKING_SCHEME]
            if complementary_de_scheme
            else pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK
        )
    if complementary_de_scheme:
        complementary_de_block = pair.COMPLEMENTARY_DE_BLOCK_BY_SCHEME[
            pair.BLOCKING_SCHEME
        ]
        config["complementary_de_jump_block"] = {
            "parameters": list(pair.FAST_RIDGE_INTRINSIC_BLOCKS[0]),
            "attempts": complementary_de_block["attempts"],
        }
    config["sha256"] = pair._config_sha256(config)
    artifact_sha = pair._sha256(weighted)
    with np.load(weighted, allow_pickle=False) as archive:
        artifact_count = len(archive[archive.files[0]])
        artifact_fields = list(archive.files)
        artifact_dtypes = {name: str(archive[name].dtype) for name in archive.files}
    likelihood_evaluations = (
        (5000 if arm.compressed else 5200)
        if complementary_de_scheme
        else (1000 if arm.compressed else 1200)
    )
    h4_results: dict[str, Any] = {}
    if hybrid_scheme:
        replacements = pair.PREFIX_STEPS * pair.N_DELETE
        attempts_history = np.ones((replacements, 3), dtype=np.int32)
        acceptances_history = np.zeros_like(attempts_history)
        acceptances_history[::10, :] = 1
        acceptances = int(acceptances_history.sum())
        h4_results = {
            "n_likelihood_evaluations_physical": (
                likelihood_evaluations + 2 * 12 * replacements
            ),
            "n_slice_updates": 12 * replacements,
            "n_periodic_uniform_independence_attempts": 3 * replacements,
            "n_periodic_uniform_independence_acceptances": acceptances,
            "periodic_uniform_independence_acceptance_rate": (
                acceptances / (3 * replacements)
            ),
            "n_likelihood_evaluations_periodic_uniform_independence": (
                3 * replacements
            ),
            "n_likelihood_evaluations_periodic_uniform_independence_waveform_rebuild": (
                2 * replacements
            ),
            "n_likelihood_evaluations_periodic_uniform_independence_cache_hit": (
                replacements
            ),
            "periodic_uniform_independence_blocks": [
                {
                    "parameters": [name],
                    "requires_waveform_rebuild": rebuild,
                    "attempts_per_replacement": 1,
                    "n_attempts": replacements,
                    "n_acceptances": int(acceptances_history[:, index].sum()),
                    "acceptance_rate": float(
                        acceptances_history[:, index].sum() / replacements
                    ),
                }
                for index, (name, rebuild) in enumerate(
                    (("s1_phi", True), ("s2_phi", True), ("psi", False))
                )
            ],
            "periodic_uniform_independence_attempts_by_block_history": (
                attempts_history.tolist()
            ),
            "periodic_uniform_independence_acceptances_by_block_history": (
                acceptances_history.tolist()
            ),
        }
    h5_results: dict[str, Any] = {}
    if complementary_de_scheme:
        replacements = pair.PREFIX_STEPS * pair.N_DELETE
        attempts_per_replacement = pair.COMPLEMENTARY_DE_BLOCK_BY_SCHEME[
            pair.BLOCKING_SCHEME
        ]["attempts"]
        total_attempts = replacements * attempts_per_replacement
        acceptances_by_attempt = np.zeros(
            (replacements, attempts_per_replacement), dtype=np.int32
        )
        acceptances_by_attempt[:, 0] = 1
        donor_indices = np.broadcast_to(
            np.asarray([0, 1], dtype=np.int32),
            (replacements, attempts_per_replacement, 2),
        ).copy()
        violations_by_attempt = np.zeros_like(acceptances_by_attempt)
        sampling_names = [
            "M_c",
            "q",
            "s1_mag",
            "s1_theta",
            "s1_phi",
            "s2_mag",
            "s2_theta",
            "s2_phi",
            "cos_iota",
            "lambda_1",
            "lambda_2",
            "d_hat",
            "zenith",
            "azimuth",
            "psi",
        ]
        positions_before = np.zeros(
            (replacements, attempts_per_replacement, len(sampling_names)),
            dtype=np.float64,
        )
        positions_before[..., sampling_names.index("q")] = 0.7
        positions_before[..., sampling_names.index("s1_mag")] = 0.02
        positions_before[..., sampling_names.index("s1_theta")] = 0.4
        positions_before[..., sampling_names.index("s2_mag")] = 0.03
        positions_before[..., sampling_names.index("s2_theta")] = 0.7
        proposals = positions_before.copy()
        proposals[..., sampling_names.index("q")] = 0.65
        total_acceptances = int(acceptances_by_attempt.sum())
        h5_results = {
            "n_likelihood_evaluations_complementary_de": total_attempts,
            "n_likelihood_evaluations_complementary_de_waveform_rebuild": (
                total_attempts
            ),
            "n_likelihood_evaluations_complementary_de_cache_hit": 0,
            "n_complementary_de_attempts": total_attempts,
            "n_complementary_de_acceptances": total_acceptances,
            "n_complementary_de_donor_policy_violations": 0,
            "complementary_de_acceptance_rate": (total_acceptances / total_attempts),
            "complementary_de_attempts_history": np.full(
                replacements, attempts_per_replacement, dtype=np.int32
            ).tolist(),
            "complementary_de_acceptances_history": (
                acceptances_by_attempt.sum(axis=1).tolist()
            ),
            "complementary_de_attempts_by_block_history": np.full(
                (replacements, 1), attempts_per_replacement, dtype=np.int32
            ).tolist(),
            "complementary_de_acceptances_by_block_history": (
                acceptances_by_attempt.sum(axis=1, keepdims=True).tolist()
            ),
            "complementary_de_donor_policy_violations_history": np.zeros(
                replacements, dtype=np.int32
            ).tolist(),
            "complementary_de_acceptances_by_attempt_history": (
                acceptances_by_attempt.tolist()
            ),
            "complementary_de_donor_indices_by_attempt_history": (
                donor_indices.tolist()
            ),
            "complementary_de_donor_policy_violations_by_attempt_history": (
                violations_by_attempt.tolist()
            ),
            "complementary_de_complement_size_history": np.full(
                replacements, 447, dtype=np.int32
            ).tolist(),
            "complementary_de_parent_index_history": np.full(
                replacements, 2, dtype=np.int32
            ).tolist(),
            "complementary_de_position_before_by_attempt_history": (
                positions_before.tolist()
            ),
            "complementary_de_proposal_position_by_attempt_history": (
                proposals.tolist()
            ),
            "complementary_de_sampling_parameter_names": sampling_names,
            "complementary_de_blocks": [
                {
                    "parameters": list(pair.FAST_RIDGE_INTRINSIC_BLOCKS[0]),
                    "requires_waveform_rebuild": True,
                    "attempts_per_replacement": attempts_per_replacement,
                    "n_attempts": total_attempts,
                    "n_acceptances": total_acceptances,
                    "acceptance_rate": total_acceptances / total_attempts,
                    "gamma": 1.0,
                    "placement": "after-target-block",
                    "n_donor_policy_violations": 0,
                }
            ],
        }
    return {
        "schema_version": 2,
        "benchmark": f"gw170817-full-swig-{pair.N_DEVICES}gpu",
        "timing_only": False,
        "ablation": {
            "name": pair.FSM_VARIANT,
            "topology": "replicated-live",
            "interval": "cached-stepping-out",
            "scheduler": "fsm",
            "direction_parameter": "covariance",
        },
        "implementation": {
            "label": arm.label,
            "module_file": "/same/jimgw/__init__.py",
            "repository": "/same",
            "revision": "abc",
            "dirty": False,
        },
        "environment": {"jax": "same", "jaxlib": "same"},
        "devices": {
            "requested_count": pair.N_DEVICES,
            "local_count": pair.N_DEVICES,
        },
        "data": {
            "path": "/same/data.npz",
            "sha256": data_sha,
            "manifest": {
                "workload": pair.WORKLOAD,
                "detectors": list(pair.DETECTORS),
                "duration_seconds": pair.DURATION_SECONDS,
                "f_min_hz": pair.F_MIN_HZ,
                "nominal_f_max_hz": pair.NOMINAL_F_MAX_HZ,
                "likelihood_f_max_hz": pair.LIKELIHOOD_F_MAX_HZ,
                "analysis_strain_product": "LOSC_CLN_16_V1",
                "gwosc_sample_rate_hz": 16384,
            },
        },
        "config": config,
        "timing_seconds": {
            "outer_step": {"outer_steps": pair.PREFIX_STEPS},
        },
        "results": {
            "n_iterations": pair.PREFIX_STEPS,
            "n_likelihood_evaluations": likelihood_evaluations,
            "n_likelihood_evaluations_de_jumps": 0,
            "n_de_jump_attempts": 0,
            "n_de_jump_acceptances": 0,
            "n_likelihood_evaluations_de_jumps_waveform_rebuild": None,
            "n_likelihood_evaluations_de_jumps_cache_hit": None,
            **h4_results,
            **h5_results,
            "log_Z": 10.0 if arm.compressed else 20.0,
            "log_Z_error": 0.5 if arm.compressed else 0.6,
            "early_stopped_for_cache_probe": True,
            "nested_artifact": {
                "path": str(weighted),
                "sha256": artifact_sha,
                "bytes": weighted.stat().st_size,
                "format": "npz",
                "space": "prior",
                "weighting": "normalized nested-sampling log weights",
                "count": artifact_count,
                "fields": artifact_fields,
                "dtypes": artifact_dtypes,
            },
        },
        "paired_likelihood_referee": {
            "experiment": (f"D={pair.N_DEVICES} full-vs-heterodyne5000 likelihood A/B"),
            "workload": pair.WORKLOAD,
            "blocking_scheme": pair.BLOCKING_SCHEME,
            "arm_label": arm.label,
            "likelihood_kind": arm.kind,
            "n_devices": pair.N_DEVICES,
            "mode": "prefix",
            "prefix_steps": pair.PREFIX_STEPS,
            "source_tree_sha256": source_sha,
            "data_sha256": data_sha,
            "reference_json_sha256": reference_sha,
            "reference_source_path": "source.npz",
            "reference_source_sha256": "a" * 64,
            "reference_source_log_likelihood": 532.5,
            "initial_positions_rng_key_sha256": "i" * 64,
            "sampler_rng_key_sha256": "k" * 64,
            "proposal": {
                "direction_parameter": "covariance",
                "direction_mode": pair.DIRECTION_MODE,
                **(
                    {
                        "num_slice_steps_by_block": list(
                            pair.FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE
                        )
                    }
                    if pair.BLOCKING_SCHEME
                    == pair.FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME
                    else {}
                ),
                **(
                    {
                        "block_kernel_modes": list(
                            pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
                        ),
                        "fixed_work": dict(
                            pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK
                        ),
                    }
                    if hybrid_scheme
                    else {}
                ),
                **(
                    {
                        "complementary_de_jump_block": {
                            "parameters": list(pair.FAST_RIDGE_INTRINSIC_BLOCKS[0]),
                            "attempts": pair.COMPLEMENTARY_DE_BLOCK_BY_SCHEME[
                                pair.BLOCKING_SCHEME
                            ]["attempts"],
                        },
                        "fixed_work": dict(
                            pair.COMPLEMENTARY_DE_FIXED_WORK_BY_SCHEME[
                                pair.BLOCKING_SCHEME
                            ]
                        ),
                    }
                    if complementary_de_scheme
                    else {}
                ),
                "num_de_jumps": 0,
                "de_jump_blocks": [],
            },
            "paper_notation": {
                "D_devices": pair.N_DEVICES,
                "m_live_points": pair.N_LIVE,
                "k_deleted_points": pair.N_DELETE,
                "M_gibbs_sweeps": pair.NUM_GIBBS_SWEEPS,
            },
            "waveform": {
                "model": pair.WAVEFORM,
                "f_ref_hz": pair.WAVEFORM_F_REF_HZ,
                "time_anchor": pair.CARRIER_TIME_ANCHOR,
            },
        },
        "simulated_cpu": True,
    }


def _netsky_posterior(path: Path) -> None:
    count = 2 * (pair.N_LIVE + pair.PREFIX_STEPS * pair.N_DELETE)
    arrays = {
        name: np.linspace(0.1, 0.2, count, dtype=np.float64)
        for name in pair.POSITION_FIELDS
    }
    arrays.update(
        {
            "q": np.linspace(0.5, 0.7, count),
            "d_L": np.linspace(20.0, 30.0, count),
            "log_likelihood": np.linspace(100.0, 110.0, count),
            "log_weights": np.full(count, -math.log(count)),
        }
    )
    np.savez(path, **arrays)


def _netsky_folded_diagnostics(path: Path) -> None:
    count = pair.N_LIVE + pair.PREFIX_STEPS * pair.N_DELETE
    death = np.linspace(1.0, 2.0, count)
    birth = death - 0.1
    birth[: pair.N_LIVE] = -np.inf
    np.savez(
        path,
        log_likelihood=death,
        log_likelihood_birth=birth,
    )


def _artifact_metadata(
    path: Path,
    *,
    space: str,
    weighting: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        fields = list(archive.files)
        count = len(archive[fields[0]])
        dtypes = {name: str(archive[name].dtype) for name in fields}
    return {
        "path": str(path.resolve()),
        "sha256": pair._sha256(path),
        "bytes": path.stat().st_size,
        "format": "npz",
        "space": space,
        "weighting": weighting,
        "count": count,
        "fields": fields,
        "dtypes": dtypes,
    }


def _netsky_report(
    arm: pair.LikelihoodArm,
    posterior: Path,
    folded: Path,
) -> dict[str, Any]:
    from jimgw.samplers.diagnostics import insertion_index_diagnostic

    report = _report(arm, posterior)
    fold_config = {
        "cos_iota": "cos_iota",
        "azimuth": "azimuth",
        "psi": "psi",
        "azimuth_reflection_center": 1.25,
    }
    fixed_work = {
        "total_updates": 32,
        "total_slice_updates": 32,
        "waveform_rebuild_slice_updates": 20,
        "cache_hit_slice_updates": 12,
        "primary_slice_updates": 30,
        "bridge_slice_updates": 2,
    }
    report["config"].update(
        {
            "bridge_blocks": [list(block) for block in pair.NETSKY_BRIDGE_BLOCKS],
            "fold_symmetry": fold_config,
            "periodic_wrapped_covariance": True,
            "fixed_work": fixed_work,
        }
    )
    report["config"]["sha256"] = pair._config_sha256(report["config"])
    report["paired_likelihood_referee"]["proposal"].update(
        {
            "bridge_blocks": [list(block) for block in pair.NETSKY_BRIDGE_BLOCKS],
            "periodic_wrapped_covariance": True,
            "fixed_work": fixed_work,
        }
    )
    posterior_metadata = _artifact_metadata(
        posterior,
        space="prior",
        weighting=pair.UNFOLDED_POSTERIOR_WEIGHTING,
    )
    posterior_metadata.update(
        {
            "schema_version": 2,
            "weight_effective_size_semantics": (
                pair.POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
            ),
        }
    )
    folded_metadata = _artifact_metadata(
        folded,
        space="folded sampling-space target",
        weighting="not applicable: folded nested-sampling contours",
    )
    folded_metadata["semantics"] = pair.FOLDED_TARGET_SEMANTICS
    with np.load(posterior, allow_pickle=False) as archive:
        log_weights = np.asarray(archive["log_weights"])
    with np.load(folded, allow_pickle=False) as archive:
        folded_death = np.asarray(archive["log_likelihood"])
        folded_birth = np.asarray(archive["log_likelihood_birth"])
    posterior_ess = float(1.0 / np.sum(np.exp(log_weights) ** 2))
    folded_count = len(folded_death)
    retained_folded_count = folded_count - 1
    physical_callbacks = 2000 if arm.compressed else 2200
    model_limitation = "weak-precession quotient-fold model limitation"
    report["timing_seconds"].update(
        {
            "result_extraction": 0.5,
            "fold_unfold_postprocessing": 0.2,
        }
    )
    report["results"].update(
        {
            "n_likelihood_evaluations_physical": physical_callbacks,
            "posterior_samples": posterior_metadata["count"],
            "posterior_artifact": posterior_metadata,
            "folded_nested_diagnostics": folded_metadata,
            "insertion_index_diagnostic": insertion_index_diagnostic(
                folded_death,
                folded_birth,
                n_live=pair.N_LIVE,
            ),
            "posterior_weight_effective_size": posterior_ess,
            "posterior_weight_effective_size_semantics": (
                pair.POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
            ),
            "quotient_fold": {
                "completed_config": fold_config,
                "model_limitation": model_limitation,
                "batch_size": 1,
                "group_order": 8,
                "folded_points": retained_folded_count,
                "normalized_conditional_image_entropy": 0.75,
                "image_sector_posterior_masses": [0.3, *([0.1] * 7)],
                "expected_nonidentity_mass": 0.7,
                "zero_support_image_fraction": 0.125,
                "supported_image_log_likelihood_gaps": {
                    "within_orbit_span_weighted_quantiles": {
                        "p05": 0.1,
                        "p50": 0.2,
                        "p95": 0.3,
                    },
                    "identity_absolute_gap_weighted_quantiles": {
                        "p05": 0.05,
                        "p50": 0.15,
                        "p95": 0.25,
                    },
                },
                "projection_accounting": {
                    "images_per_folded_target_callback": 8,
                    "sampler_folded_target_callbacks": physical_callbacks,
                    "sampler_true_image_projections": 8 * physical_callbacks,
                    "retained_folded_points_unfolded": retained_folded_count,
                    "unfold_true_image_projections": 8 * retained_folded_count,
                    "total_true_image_projections": (
                        8 * physical_callbacks + 8 * retained_folded_count
                    ),
                    "sampler_callback_counter": (
                        "results.n_likelihood_evaluations_physical"
                    ),
                },
            },
        }
    )
    report["results"].pop("nested_artifact")
    report["limitations"] = [model_limitation]
    return report


def test_cli_defaults_to_adversarial_seed2_d4_fsm_u8_prefix(
    tmp_path: Path,
) -> None:
    args = pair._parse_args(
        ["--data-file", str(tmp_path / "data.npz"), "--prefix", str(tmp_path / "x")]
    )

    assert args.seed == 2
    assert args.mode == "prefix"
    assert args.n_devices == 4
    assert args.gpu_devices == "0,1,2,3"
    assert pair.N_DEVICES == 4
    assert pair.FULL.label == "jim-paper-15d-d4-fsm-full"
    assert pair.HETERODYNE.label == "jim-paper-15d-d4-fsm-heterodyne5000"
    assert pair._benchmark_name() == "gw170817-full-swig-4gpu"
    assert pair._pair_name() == "jim-paper-15d-d4-fsm-likelihood-pair"
    assert pair.TIME_UPSAMPLE_FACTOR == 8
    assert pair.FSM_VARIANT == "replicated-cached-fsm-cov"
    assert pair.ARMS == (pair.HETERODYNE, pair.FULL)


def test_netsky_cli_defaults_to_two_gibbs_sweeps_and_canonical_blocks(
    tmp_path: Path,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "netsky"),
            "--blocking-scheme",
            "netsky",
        ]
    )

    assert args.num_gibbs_sweeps == 2
    assert pair.BLOCKS_BY_SCHEME[pair.NETSKY_SCHEME] == pair.NETSKY_BLOCKS
    assert [len(block) for block in pair.NETSKY_BLOCKS] == [4, 3, 3, 1, 4]


@pytest.mark.parametrize(
    "extra_args",
    (
        ["--num-gibbs-sweeps", "1"],
        ["--num-gibbs-sweeps", "3"],
        ["--direction-mode", "covariance-basis-8d"],
    ),
)
def test_netsky_cli_rejects_noncanonical_sampler_work(
    tmp_path: Path,
    extra_args: list[str],
) -> None:
    with pytest.raises(SystemExit):
        pair._parse_args(
            [
                "--data-file",
                str(tmp_path / "data.npz"),
                "--prefix",
                str(tmp_path / "netsky"),
                "--blocking-scheme",
                "netsky",
                *extra_args,
            ]
        )


@pytest.mark.parametrize("value", ["0", "0,1,2", "0,1,2,2", "0,1,2,x", "-1,0,1,2"])
def test_cli_rejects_any_gpu_mask_other_than_four_distinct_indices(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        pair._parse_args(
            [
                "--data-file",
                str(tmp_path / "data.npz"),
                "--prefix",
                str(tmp_path / "x"),
                "--gpu-devices",
                value,
            ]
        )


def test_cli_accepts_fast_ridge_on_one_economical_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge",
            "--n-devices",
            "1",
            "--implementation-revision",
            "tree-sha256-test",
        ]
    )

    assert args.blocking_scheme == "fast-ridge"
    assert args.n_devices == 1
    assert args.gpu_devices == "0"
    monkeypatch.setattr(pair, "N_DEVICES", pair.N_DEVICES)
    monkeypatch.setattr(pair, "BLOCKING_SCHEME", pair.BLOCKING_SCHEME)
    monkeypatch.setattr(pair, "PAPER_BLOCKS", pair.PAPER_BLOCKS)
    pair._configure_runtime(args)
    assert pair.FULL.label == "jim-paper-15d-d1-fsm-full"
    assert pair.HETERODYNE.label == "jim-paper-15d-d1-fsm-heterodyne5000"
    full_paths = pair._artifact_paths(args.prefix, args.mode, pair.FULL)
    assert ".jim-paper-15d-d1-fsm-full." in full_paths["report"].name
    reference, reference_sha = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)
    metadata = pair._pairing_metadata(
        args,
        pair.FULL,
        {"sha256": "s" * 64},
        "d" * 64,
        reference,
        reference_sha,
    )
    assert metadata["experiment"] == "D=1 full-vs-heterodyne5000 likelihood A/B"
    assert metadata["arm_label"] == "jim-paper-15d-d1-fsm-full"
    assert metadata["n_devices"] == 1
    assert pair._benchmark_name() == "gw170817-full-swig-1gpu"
    assert pair._pair_name() == "jim-paper-15d-d1-fsm-likelihood-pair"

    compressed_weighted = tmp_path / "d1-compressed.npz"
    full_weighted = tmp_path / "d1-full.npz"
    _weighted(compressed_weighted)
    _weighted(full_weighted)
    reports = {
        pair.HETERODYNE.kind: _report(pair.HETERODYNE, compressed_weighted),
        pair.FULL.kind: _report(pair.FULL, full_weighted),
    }
    comparison = pair.compare_pair(
        reports,
        {
            pair.HETERODYNE.kind: compressed_weighted,
            pair.FULL.kind: full_weighted,
        },
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )
    assert comparison["strict_pass"] is True
    assert "D=1 FSM" in comparison["criterion"]
    command = pair._cell_command(args, pair.FULL)
    assert command[command.index("--blocking-scheme") + 1] == "fast-ridge"
    assert command[command.index("--n-devices") + 1] == "1"


def test_cli_accepts_fixed_budget_intrinsic_fast_ridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic",
            "--n-devices",
            "1",
            "--implementation-revision",
            "tree-sha256-test",
        ]
    )

    monkeypatch.setattr(pair, "N_DEVICES", pair.N_DEVICES)
    monkeypatch.setattr(pair, "BLOCKING_SCHEME", pair.BLOCKING_SCHEME)
    monkeypatch.setattr(pair, "PAPER_BLOCKS", pair.PAPER_BLOCKS)
    pair._configure_runtime(args)

    assert args.blocking_scheme == "fast-ridge-intrinsic"
    assert pair.PAPER_BLOCKS == pair.FAST_RIDGE_INTRINSIC_BLOCKS
    assert [len(block) for block in pair.PAPER_BLOCKS] == [8, 1, 1, 2, 1, 2]
    assert sum(map(len, pair.PAPER_BLOCKS)) == 15
    command = pair._cell_command(args, pair.HETERODYNE)
    assert command[command.index("--blocking-scheme") + 1] == ("fast-ridge-intrinsic")
    assert command[command.index("--implementation-revision") + 1] == (
        "tree-sha256-test"
    )


def test_cli_pins_five_step_intrinsic_covariance_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic-5step",
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)

    reference, reference_sha = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)
    metadata = pair._pairing_metadata(
        args,
        pair.HETERODYNE,
        {"sha256": "s" * 64},
        "d" * 64,
        reference,
        reference_sha,
    )

    assert pair.BLOCKING_SCHEME == "fast-ridge-intrinsic-5step"
    assert pair.PAPER_BLOCKS == pair.FAST_RIDGE_INTRINSIC_BLOCKS
    assert metadata["proposal"]["direction_mode"] == "covariance"
    assert metadata["proposal"]["num_slice_steps_by_block"] == [5, 1, 1, 2, 1, 2]
    with pytest.raises(SystemExit):
        pair._parse_args(
            [
                "--data-file",
                str(tmp_path / "data.npz"),
                "--prefix",
                str(tmp_path / "bad"),
                "--blocking-scheme",
                "fast-ridge-intrinsic-5step",
                "--direction-mode",
                "covariance-basis-8d",
            ]
        )


def test_cli_pins_periodic_mh_hybrid_work_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic-periodic-mh",
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)

    reference, reference_sha = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)
    metadata = pair._pairing_metadata(
        args,
        pair.HETERODYNE,
        {"sha256": "s" * 64},
        "d" * 64,
        reference,
        reference_sha,
    )

    assert pair.BLOCKING_SCHEME == "fast-ridge-intrinsic-periodic-mh"
    assert pair.PAPER_BLOCKS == pair.FAST_RIDGE_INTRINSIC_BLOCKS
    assert metadata["proposal"]["direction_mode"] == "covariance"
    assert metadata["proposal"]["block_kernel_modes"] == [
        "slice",
        "periodic-uniform-independence",
        "periodic-uniform-independence",
        "slice",
        "periodic-uniform-independence",
        "slice",
    ]
    assert "num_slice_steps_by_block" not in metadata["proposal"]
    assert metadata["proposal"]["fixed_work"] == {
        "total_updates": 15,
        "total_slice_updates": 12,
        "waveform_rebuild_slice_updates": 8,
        "cache_hit_slice_updates": 4,
        "periodic_independence_attempts": 3,
        "waveform_rebuild_periodic_independence_attempts": 2,
        "cache_hit_periodic_independence_attempts": 1,
        "cache_segments": 2,
    }
    with pytest.raises(SystemExit):
        pair._parse_args(
            [
                "--data-file",
                str(tmp_path / "data.npz"),
                "--prefix",
                str(tmp_path / "bad"),
                "--blocking-scheme",
                "fast-ridge-intrinsic-periodic-mh",
                "--direction-mode",
                "covariance-basis-8d",
            ]
        )


def test_cli_pins_periodic_mh_cde4_pair_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheme = "fast-ridge-intrinsic-periodic-mh-cde4"
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            scheme,
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)

    reference, reference_sha = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)
    metadata = pair._pairing_metadata(
        args,
        pair.HETERODYNE,
        {"sha256": "s" * 64},
        "d" * 64,
        reference,
        reference_sha,
    )

    assert pair.BLOCKING_SCHEME == scheme
    assert pair.PAPER_BLOCKS == pair.FAST_RIDGE_INTRINSIC_BLOCKS
    assert metadata["proposal"] == {
        "direction_parameter": "covariance",
        "direction_mode": "covariance",
        "block_kernel_modes": list(pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES),
        "complementary_de_jump_block": {
            "parameters": list(pair.FAST_RIDGE_INTRINSIC_BLOCKS[0]),
            "attempts": 4,
        },
        "complementary_de_policy": dict(
            pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_POLICY
        ),
        "fixed_work": dict(pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_FIXED_WORK),
        "num_de_jumps": 0,
        "de_jump_blocks": [],
    }
    command = pair._cell_command(args, pair.HETERODYNE)
    assert command[command.index("--blocking-scheme") + 1] == scheme
    with pytest.raises(SystemExit):
        pair._parse_args(
            [
                "--data-file",
                str(tmp_path / "data.npz"),
                "--prefix",
                str(tmp_path / "bad-d4"),
                "--blocking-scheme",
                scheme,
                "--n-devices",
                "4",
            ]
        )


def test_cli_pins_periodic_mh_cde8_pair_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheme = "fast-ridge-intrinsic-periodic-mh-cde8"
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            scheme,
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)

    reference, reference_sha = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)
    metadata = pair._pairing_metadata(
        args,
        pair.HETERODYNE,
        {"sha256": "s" * 64},
        "d" * 64,
        reference,
        reference_sha,
    )

    assert pair.BLOCKING_SCHEME == scheme
    assert pair.PAPER_BLOCKS == pair.FAST_RIDGE_INTRINSIC_BLOCKS
    assert metadata["proposal"] == {
        "direction_parameter": "covariance",
        "direction_mode": "covariance",
        "block_kernel_modes": list(pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES),
        "complementary_de_jump_block": {
            "parameters": list(pair.FAST_RIDGE_INTRINSIC_BLOCKS[0]),
            "attempts": 8,
        },
        "complementary_de_policy": dict(
            pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_POLICY
        ),
        "fixed_work": dict(pair.FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_FIXED_WORK),
        "num_de_jumps": 0,
        "de_jump_blocks": [],
    }
    command = pair._cell_command(args, pair.HETERODYNE)
    assert command[command.index("--blocking-scheme") + 1] == scheme


def test_pair_validation_pins_periodic_mh_counts_in_both_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic-periodic-mh",
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)
    full_weighted = tmp_path / "full.npz"
    compressed_weighted = tmp_path / "compressed.npz"
    _weighted(full_weighted)
    _weighted(compressed_weighted)
    reports = {
        pair.FULL.kind: _report(pair.FULL, full_weighted),
        pair.HETERODYNE.kind: _report(pair.HETERODYNE, compressed_weighted),
    }
    weighted = {
        pair.FULL.kind: full_weighted,
        pair.HETERODYNE.kind: compressed_weighted,
    }

    comparison = pair.compare_pair(
        reports,
        weighted,
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )
    assert comparison["strict_pass"] is True

    reports[pair.HETERODYNE.kind]["results"][
        "n_periodic_uniform_independence_attempts"
    ] -= 1
    comparison = pair.compare_pair(
        reports,
        weighted,
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )
    assert comparison["strict_pass"] is False
    errors = comparison["arm_validation"][pair.HETERODYNE.kind]["report_errors"]
    assert any("periodic_uniform_independence_attempts" in error for error in errors)


def test_pair_validation_pins_cde4_histories_and_donor_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic-periodic-mh-cde4",
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)
    full_weighted = tmp_path / "full.npz"
    compressed_weighted = tmp_path / "compressed.npz"
    _weighted(full_weighted)
    _weighted(compressed_weighted)
    reports = {
        pair.FULL.kind: _report(pair.FULL, full_weighted),
        pair.HETERODYNE.kind: _report(pair.HETERODYNE, compressed_weighted),
    }
    weighted = {
        pair.FULL.kind: full_weighted,
        pair.HETERODYNE.kind: compressed_weighted,
    }

    comparison = pair.compare_pair(
        reports,
        weighted,
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )
    assert comparison["strict_pass"] is True

    reports[pair.HETERODYNE.kind]["results"][
        "complementary_de_donor_policy_violations_by_attempt_history"
    ][0][0] = 1
    comparison = pair.compare_pair(
        reports,
        weighted,
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )
    assert comparison["strict_pass"] is False
    errors = comparison["arm_validation"][pair.HETERODYNE.kind]["report_errors"]
    assert any("donor-violation histories" in error for error in errors)


def test_pair_validation_pins_cde8_histories_and_donor_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic-periodic-mh-cde8",
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)
    full_weighted = tmp_path / "full.npz"
    compressed_weighted = tmp_path / "compressed.npz"
    _weighted(full_weighted)
    _weighted(compressed_weighted)
    reports = {
        pair.FULL.kind: _report(pair.FULL, full_weighted),
        pair.HETERODYNE.kind: _report(pair.HETERODYNE, compressed_weighted),
    }
    weighted = {
        pair.FULL.kind: full_weighted,
        pair.HETERODYNE.kind: compressed_weighted,
    }

    comparison = pair.compare_pair(
        reports,
        weighted,
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )
    assert comparison["strict_pass"] is True
    for report in reports.values():
        history = report["results"]["complementary_de_acceptances_by_attempt_history"]
        assert np.asarray(history).shape == (pair.PREFIX_STEPS * pair.N_DELETE, 8)

    reports[pair.HETERODYNE.kind]["results"][
        "complementary_de_donor_policy_violations_by_attempt_history"
    ][0][7] = 1
    comparison = pair.compare_pair(
        reports,
        weighted,
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )
    assert comparison["strict_pass"] is False
    errors = comparison["arm_validation"][pair.HETERODYNE.kind]["report_errors"]
    assert any("donor-violation histories" in error for error in errors)


def test_pair_validation_rejects_noncanonical_cde8_attempt_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic-periodic-mh-cde8",
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)
    full_weighted = tmp_path / "full.npz"
    compressed_weighted = tmp_path / "compressed.npz"
    _weighted(full_weighted)
    _weighted(compressed_weighted)
    reports = {
        pair.FULL.kind: _report(pair.FULL, full_weighted),
        pair.HETERODYNE.kind: _report(pair.HETERODYNE, compressed_weighted),
    }
    history = reports[pair.FULL.kind]["results"][
        "complementary_de_acceptances_by_attempt_history"
    ]
    reports[pair.FULL.kind]["results"][
        "complementary_de_acceptances_by_attempt_history"
    ] = [[row] for row in history]
    violations = reports[pair.FULL.kind]["results"][
        "complementary_de_donor_policy_violations_by_attempt_history"
    ]
    reports[pair.FULL.kind]["results"][
        "complementary_de_donor_policy_violations_by_attempt_history"
    ] = [[row] for row in violations]

    comparison = pair.compare_pair(
        reports,
        {pair.FULL.kind: full_weighted, pair.HETERODYNE.kind: compressed_weighted},
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )

    assert comparison["strict_pass"] is False
    errors = comparison["arm_validation"][pair.FULL.kind]["report_errors"]
    assert "results.complementary_de per-attempt histories are invalid" in errors


def test_pair_validation_reproduces_v4_noncanonical_d1_cde_history_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce the verified v4 artifact's exact fail-closed predicates."""

    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic-periodic-mh-cde4",
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)

    full_weighted = tmp_path / "full.npz"
    compressed_weighted = tmp_path / "compressed.npz"
    _weighted(full_weighted)
    _weighted(compressed_weighted)
    reports = {
        pair.FULL.kind: _report(pair.FULL, full_weighted),
        pair.HETERODYNE.kind: _report(pair.HETERODYNE, compressed_weighted),
    }
    weighted = {
        pair.FULL.kind: full_weighted,
        pair.HETERODYNE.kind: compressed_weighted,
    }

    scalar_fields = (
        "complementary_de_attempts_history",
        "complementary_de_acceptances_history",
        "complementary_de_donor_policy_violations_history",
        "complementary_de_complement_size_history",
    )
    structured_fields = (
        "complementary_de_attempts_by_block_history",
        "complementary_de_acceptances_by_block_history",
        "complementary_de_acceptances_by_attempt_history",
        "complementary_de_donor_indices_by_attempt_history",
        "complementary_de_donor_policy_violations_by_attempt_history",
        "complementary_de_position_before_by_attempt_history",
        "complementary_de_proposal_position_by_attempt_history",
    )
    for report in reports.values():
        results = report["results"]
        for field in scalar_fields:
            results[field] = np.asarray(results[field])[:, None].tolist()
        for field in structured_fields:
            results[field] = np.asarray(results[field])[:, None, ...].tolist()

    full_results = reports[pair.FULL.kind]["results"]
    assert np.asarray(full_results["complementary_de_attempts_history"]).shape == (
        pair.PREFIX_STEPS * pair.N_DELETE,
        1,
    )
    assert np.asarray(full_results["complementary_de_parent_index_history"]).shape == (
        pair.PREFIX_STEPS * pair.N_DELETE,
    )

    comparison = pair.compare_pair(
        reports,
        weighted,
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )

    assert comparison["strict_pass"] is False
    for arm in pair.ARMS:
        assert comparison["arm_validation"][arm.kind]["report_errors"] == [
            "results.complementary_de scalar histories are invalid",
            "results.complementary_de donor histories are invalid",
        ]


def test_cli_pins_covariance_basis_direction_for_intrinsic_screen_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic",
            "--direction-mode",
            "covariance-basis-8d",
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)

    command = pair._cell_command(args, pair.HETERODYNE)
    reference, reference_sha = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)
    metadata = pair._pairing_metadata(
        args,
        pair.HETERODYNE,
        {"sha256": "s" * 64},
        "d" * 64,
        reference,
        reference_sha,
    )

    assert pair.DIRECTION_MODE == "covariance-basis-8d"
    assert command[command.index("--direction-mode") + 1] == "covariance-basis-8d"
    assert metadata["proposal"]["direction_mode"] == "covariance-basis-8d"
    assert pair._pair_name().endswith("-covariance-basis-8d")


def test_cli_rejects_covariance_basis_without_intrinsic_block(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        pair._parse_args(
            [
                "--data-file",
                str(tmp_path / "data.npz"),
                "--prefix",
                str(tmp_path / "x"),
                "--direction-mode",
                "covariance-basis-8d",
            ]
        )


def test_pair_validation_pins_covariance_basis_in_both_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--blocking-scheme",
            "fast-ridge-intrinsic",
            "--direction-mode",
            "covariance-basis-8d",
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "DIRECTION_MODE", "N_DEVICES", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)
    full_weighted = tmp_path / "full.npz"
    compressed_weighted = tmp_path / "compressed.npz"
    _weighted(full_weighted)
    _weighted(compressed_weighted)
    reports = {
        pair.FULL.kind: _report(pair.FULL, full_weighted),
        pair.HETERODYNE.kind: _report(pair.HETERODYNE, compressed_weighted),
    }

    comparison = pair.compare_pair(
        reports,
        {
            pair.FULL.kind: full_weighted,
            pair.HETERODYNE.kind: compressed_weighted,
        },
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )
    assert comparison["strict_pass"] is True

    reports[pair.HETERODYNE.kind]["paired_likelihood_referee"]["proposal"][
        "direction_mode"
    ] = "covariance"
    comparison = pair.compare_pair(
        reports,
        {
            pair.FULL.kind: full_weighted,
            pair.HETERODYNE.kind: compressed_weighted,
        },
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )
    assert comparison["strict_pass"] is False


def test_artifact_paths_and_child_commands_are_arm_specific(tmp_path: Path) -> None:
    args = pair._parse_args(
        ["--data-file", str(tmp_path / "data.npz"), "--prefix", str(tmp_path / "x")]
    )

    compressed = pair._artifact_paths(args.prefix, "prefix", pair.HETERODYNE)
    full = pair._artifact_paths(args.prefix, "prefix", pair.FULL)
    assert compressed["report"] != full["report"]
    assert "heterodyne5000" in compressed["weighted"].name
    assert "fsm-full" in full["weighted"].name
    compressed_command = pair._cell_command(args, pair.HETERODYNE)
    full_command = pair._cell_command(args, pair.FULL)
    assert compressed_command[-1] == pair.HETERODYNE.kind
    assert full_command[-1] == pair.FULL.kind
    assert "--sampler-seed" not in compressed_command
    assert "--sampler-seed" not in full_command


def test_netsky_artifact_paths_separate_posterior_and_folded_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pair, "BLOCKING_SCHEME", pair.NETSKY_SCHEME)

    paths = pair._artifact_paths(tmp_path / "netsky", "prefix", pair.FULL)

    assert set(paths) == {"report", "weighted", "folded"}
    assert paths["weighted"].name.endswith(".weighted.npz")
    assert paths["folded"].name.endswith(".folded-nested-diagnostics.npz")
    assert paths["weighted"] != paths["folded"]


def test_netsky_cell_config_threads_both_scientific_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pair, "BLOCKING_SCHEME", pair.NETSKY_SCHEME)
    paths = pair._artifact_paths(tmp_path / "netsky", "prefix", pair.FULL)
    benchmark_args = SimpleNamespace(
        timing_only=True,
        nested_output=None,
        folded_nested_output=None,
        samples_output=tmp_path / "samples.npz",
        max_outer_steps=None,
    )

    pair._configure_cell_artifacts(benchmark_args, paths, mode="prefix")

    assert benchmark_args.timing_only is False
    assert benchmark_args.nested_output == paths["weighted"]
    assert benchmark_args.folded_nested_output == paths["folded"]
    assert benchmark_args.samples_output is None
    assert benchmark_args.max_outer_steps == pair.PREFIX_STEPS


def test_pair_cli_routes_an_independent_sampler_seed_to_both_arms(
    tmp_path: Path,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "x"),
            "--seed",
            "5",
            "--sampler-seed",
            "1000005",
        ]
    )

    for arm in (pair.FULL, pair.HETERODYNE):
        command = pair._cell_command(args, arm)
        assert command[command.index("--seed") + 1] == "5"
        assert command[command.index("--sampler-seed") + 1] == "1000005"

    legacy = pair._key_hashes(5)
    repeated = pair._key_hashes(5, sampler_seed=1_000_005)
    assert (
        repeated["initial_positions_rng_key_sha256"]
        == (legacy["initial_positions_rng_key_sha256"])
    )
    assert repeated["sampler_rng_key_sha256"] != legacy["sampler_rng_key_sha256"]

    reference, reference_sha = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)
    metadata = pair._pairing_metadata(
        args,
        pair.FULL,
        {"sha256": "s" * 64},
        "d" * 64,
        reference,
        reference_sha,
    )
    assert metadata["trajectory_seed_override"] == {
        "initial_positions_seed": 5,
        "sampler_seed": 1_000_005,
    }
    assert (
        metadata["initial_positions_rng_key_sha256"]
        == (repeated["initial_positions_rng_key_sha256"])
    )
    assert metadata["sampler_rng_key_sha256"] == (repeated["sampler_rng_key_sha256"])

    legacy_args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "legacy"),
            "--seed",
            "5",
        ]
    )
    legacy_metadata = pair._pairing_metadata(
        legacy_args,
        pair.FULL,
        {"sha256": "s" * 64},
        "d" * 64,
        reference,
        reference_sha,
    )
    assert "trajectory_seed_override" not in legacy_metadata


def test_frozen_reference_is_internally_consistent_and_anchor_pinned() -> None:
    reference, digest = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)

    assert len(digest) == 64
    assert reference["waveform_contract"] == {
        "model": pair.WAVEFORM,
        "f_ref_hz": pair.WAVEFORM_F_REF_HZ,
        "time_anchor": pair.CARRIER_TIME_ANCHOR,
    }


def test_reference_rejects_anchor_or_transform_corruption(tmp_path: Path) -> None:
    reference = json.loads(pair.DEFAULT_REFERENCE_FILE.read_text())
    reference["waveform_contract"]["time_anchor"] = "nrtidal-merger"
    wrong_anchor = tmp_path / "wrong-anchor.json"
    wrong_anchor.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="waveform contract"):
        pair._load_reference(wrong_anchor)

    reference = json.loads(pair.DEFAULT_REFERENCE_FILE.read_text())
    reference["likelihood_parameters"]["eta"] += 0.01
    wrong_eta = tmp_path / "wrong-eta.json"
    wrong_eta.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="q and eta"):
        pair._load_reference(wrong_eta)


def test_bin_metadata_requires_all_5000_realized_and_nonzero_reference() -> None:
    edges = np.linspace(20.0, 2047.0, pair.N_BINS_REQUESTED + 1)
    likelihood = SimpleNamespace(
        requested_n_bins=pair.N_BINS_REQUESTED,
        n_bins=pair.N_BINS_REQUESTED,
        freq_grid_low=edges[:-1],
        freq_grid_high=edges[1:],
        waveform_low_ref={
            name: np.ones(pair.N_BINS_REQUESTED) for name in pair.DETECTORS
        },
        waveform_high_ref={
            name: np.ones(pair.N_BINS_REQUESTED) for name in pair.DETECTORS
        },
    )

    metadata = pair._bin_metadata(likelihood)
    assert metadata["requested_bins"] == 5000
    assert metadata["realized_bins"] == 5000
    assert metadata["bin_edges_count"] == 5001

    likelihood.n_bins = 4999
    with pytest.raises(RuntimeError, match="exactly 5000"):
        pair._bin_metadata(likelihood)


def test_full_time_metadata_pins_exact_u8_window_count_and_hash() -> None:
    from jimgw.core.single_event.likelihood import (
        _build_time_marginalization_fine_window,
    )

    coarse_count = int(
        pair.DURATION_SECONDS * pair.TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ / 2.0
    )
    candidates, mask, _ = _build_time_marginalization_fine_window(
        coarse_count,
        pair.TIME_UPSAMPLE_FACTOR,
        pair.DURATION_SECONDS,
        pair.TC_RANGE_SECONDS,
    )
    likelihood = SimpleNamespace(
        tc_array=np.empty(coarse_count),
        _tc_fine_candidate_indices=candidates,
        _tc_fine_mask=mask,
        tc_upsample=pair.TIME_UPSAMPLE_FACTOR,
        tc_range=pair.TC_RANGE_SECONDS,
    )

    metadata = pair._time_metadata(likelihood, compressed=False)

    assert metadata["upsample_factor"] == 8
    assert metadata["window_point_count"] == 983
    assert metadata["normalization_point_count"] == coarse_count * 8
    assert metadata["window_sha256"] == _time_grid_metadata()["window_sha256"]


def test_probe_bank_uses_weighted_systematic_points_and_top_logl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.npz"
    count = 100
    arrays = {name: np.linspace(0.1, 0.2, count) for name in pair.POSITION_FIELDS}
    arrays["q"] = np.linspace(0.2, 0.9, count)
    arrays["d_L"] = np.linspace(10.0, 40.0, count)
    arrays["log_likelihood"] = np.linspace(400.0, 500.0, count)
    arrays["log_likelihood_birth"] = np.linspace(399.0, 499.0, count)
    arrays["log_weights"] = np.full(count, -math.log(count))
    np.savez(path, **arrays)

    points, manifests, top_names = pair._load_probe_bank([path])

    assert len(points) == pair.PROBE_SYSTEMATIC_POINTS_PER_FILE + 1
    assert manifests[0]["stored_top_q"] == pytest.approx(0.9)
    assert len(top_names) == 1
    top = dict(points)[next(iter(top_names))]
    assert top["eta"] == pytest.approx(0.9 / 1.9**2)
    assert top["phase_c"] == 0.0
    assert top["t_c"] == 0.0


def test_foreign_cache_ridge_bank_changes_only_iota_and_distance() -> None:
    reference, _ = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)

    parent, targets = pair._foreign_cache_ridge_points(reference)

    assert [name for name, _ in targets] == [
        "foreign-cache-face-on",
        "foreign-cache-edge-on",
        "foreign-cache-face-away",
    ]
    for _, target in targets:
        changed = {
            name for name in parent if not math.isclose(parent[name], target[name])
        }
        assert changed == {"iota", "d_L"}


def test_foreign_cache_evaluator_reuses_one_parent_and_check_fails_closed() -> None:
    class FakeJax:
        @staticmethod
        def jit(function: Any) -> Any:
            return function

        @staticmethod
        def device_get(value: Any) -> Any:
            return value

    class FakeLikelihood:
        def __init__(self) -> None:
            self.generated: list[dict[str, float]] = []

        def generate_waveform(self, params: dict[str, float]) -> dict[str, float]:
            self.generated.append(dict(params))
            return {"parent_iota": params["iota"], "parent_d_L": params["d_L"]}

        @staticmethod
        def evaluate(params: dict[str, float]) -> float:
            return params["iota"] + 0.1 * params["d_L"]

        def evaluate_from_waveform(
            self,
            params: dict[str, float],
            cache: dict[str, float],
        ) -> float:
            assert cache == {"parent_iota": 1.0, "parent_d_L": 20.0}
            return self.evaluate(params)

    parent = {"iota": 1.0, "d_L": 20.0}
    targets = [
        ("near", {"iota": 0.1, "d_L": 15.0}),
        ("far", {"iota": 3.0, "d_L": 70.0}),
    ]
    likelihood = FakeLikelihood()

    values = pair._evaluate_foreign_cache_paths(
        FakeJax,
        likelihood,
        parent,
        targets,
    )

    assert likelihood.generated == [parent]
    assert set(values["near"]) == {
        "direct",
        "foreign_cache",
        "jit_direct",
        "jit_foreign_cache",
    }
    check = pair._foreign_cache_parity_check(values, values)
    assert check["passed"] is True
    assert check["observed_max_abs_delta"] == 0.0

    bad_values = {name: dict(paths) for name, paths in values.items()}
    bad_values["far"]["jit_foreign_cache"] += 2.0 * pair.PARITY_ATOL
    failed = pair._foreign_cache_parity_check(values, bad_values)
    assert failed["passed"] is False
    assert failed["observed_max_abs_delta"] == pytest.approx(2.0 * pair.PARITY_ATOL)
    assert failed["per_likelihood_and_target"][pair.HETERODYNE.kind]["far"][
        "max_abs_delta"
    ] == pytest.approx(2.0 * pair.PARITY_ATOL)


def test_u8_u16_gate_covers_every_local_and_probe_point() -> None:
    points = [
        ("reference", {}),
        ("intrinsic-small", {}),
        ("probe-00-systematic-00", {}),
        ("probe-00-top-logL", {}),
    ]
    full = {name: {"direct": 500.0 + index} for index, (name, _) in enumerate(points)}
    u16 = {name: paths["direct"] + 0.001 for name, paths in full.items()}
    local_names = {"reference", "intrinsic-small"}

    passed = pair._u8_u16_point_bank_check(
        points,
        full,
        u16,
        local_names=local_names,
    )

    assert passed["passed"] is True
    assert passed["point_count"] == 4
    assert passed["local_point_count"] == 2
    assert passed["independent_probe_point_count"] == 2
    assert set(passed["per_point"]) == {name for name, _ in points}
    assert passed["per_point"]["probe-00-systematic-00"]["source"] == (
        "independent-probe"
    )
    assert "every selected coordinate" in passed["note"]

    u16["probe-00-systematic-00"] = (
        full["probe-00-systematic-00"]["direct"] + 2.0 * pair.MAX_U8_U16_ABS_LOGL_DELTA
    )
    failed = pair._u8_u16_point_bank_check(
        points,
        full,
        u16,
        local_names=local_names,
    )
    assert failed["passed"] is False
    assert failed["observed_max_abs_delta"] == pytest.approx(
        2.0 * pair.MAX_U8_U16_ABS_LOGL_DELTA
    )

    with pytest.raises(ValueError, match="complete point bank"):
        pair._u8_u16_point_bank_check(
            points,
            full,
            {name: value for name, value in u16.items() if name != "probe-00-top-logL"},
            local_names=local_names,
        )


def _q_time_grid_regression_inputs() -> tuple[
    list[tuple[str, dict[str, float]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, float],
]:
    reference, _ = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)
    points = pair._q_time_grid_points(reference)
    drops = (824.0, 500.0, 300.0, 201.0, 200.0, 150.0, 100.0, 50.0, 10.0, 0.5)

    def paths(value: float) -> dict[str, float]:
        return {
            "direct": value,
            "cache": value,
            "jit_direct": value,
            "jit_cache": value,
        }

    full: dict[str, dict[str, float]] = {}
    compressed: dict[str, dict[str, float]] = {}
    u16: dict[str, float] = {}
    for index, ((name, _), drop) in enumerate(zip(points, drops, strict=True)):
        full_value = 1000.0 - drop
        compression_delta = (
            0.5 * pair.MAX_RELATIVE_BINNING_BETA * drop
            if drop >= pair.MIN_BETA_LOGL_DROP
            else 0.5 * pair.MAX_REFERENCE_ABS_LOGL_DELTA
        )
        full[name] = paths(full_value)
        compressed[name] = paths(full_value + compression_delta)
        u16[name] = full_value + (0.19 if index == 0 else 0.001)
    return points, full, compressed, u16


def test_complete_q_time_grid_reports_all_points_and_gates_relevant_u8_u16() -> None:
    points, full, compressed, u16 = _q_time_grid_regression_inputs()

    assert len(points) == len(pair.Q_TIME_GRID_VALUES)
    for q, (name, params) in zip(pair.Q_TIME_GRID_VALUES, points, strict=True):
        assert name == f"q-time-grid-{q:.3f}"
        assert params["eta"] == pytest.approx(q / (1.0 + q) ** 2)
        assert params["t_c"] == 0.0

    summary = pair._q_time_grid_regression_check(
        points,
        full,
        compressed,
        u16,
        bank_max_log_likelihood=1000.0,
    )
    checks = summary["checks"]

    assert checks["q_time_grid_direct_cache_jit_parity"]["passed"] is True
    assert checks["q_time_grid_compression_accuracy"]["passed"] is True
    convergence = checks["q_time_grid_u8_u16_relevant_convergence"]
    assert convergence["passed"] is True
    assert convergence["maximum_full_u8_log_likelihood_drop"] == 200.0
    assert convergence["observed_max_abs_delta_relevant"] == pytest.approx(0.001)
    assert convergence["observed_max_abs_delta_all_points"] == pytest.approx(0.19)
    assert convergence["excluded_point_names"] == [name for name, _ in points[:4]]
    assert list(summary["per_point"]) == [name for name, _ in points]
    first = summary["per_point"][points[0][0]]
    assert first["within_u8_u16_relevance_window"] is False
    assert first["u16_minus_full_u8"] == pytest.approx(0.19)
    assert "post-failure" in summary["note"]


def test_q_time_grid_u8_u16_gate_fails_relevant_error_and_empty_window() -> None:
    points, full, compressed, u16 = _q_time_grid_regression_inputs()
    relevant_name = points[4][0]
    u16[relevant_name] = (
        full[relevant_name]["direct"] + 2.0 * pair.MAX_U8_U16_ABS_LOGL_DELTA
    )

    failed = pair._q_time_grid_regression_check(
        points,
        full,
        compressed,
        u16,
        bank_max_log_likelihood=1000.0,
    )
    convergence = failed["checks"]["q_time_grid_u8_u16_relevant_convergence"]
    assert convergence["passed"] is False
    assert convergence["observed_max_abs_delta_relevant"] == pytest.approx(
        2.0 * pair.MAX_U8_U16_ABS_LOGL_DELTA
    )

    empty = pair._q_time_grid_regression_check(
        points,
        full,
        compressed,
        u16,
        bank_max_log_likelihood=2000.0,
    )["checks"]["q_time_grid_u8_u16_relevant_convergence"]
    assert empty["passed"] is False
    assert empty["relevant_point_count"] == 0
    assert empty["observed_max_abs_delta_relevant"] is None


@pytest.mark.parametrize("failure_kind", ["parity", "beta", "near-max"])
def test_q_time_grid_cache_and_compression_checks_fail_closed(
    failure_kind: str,
) -> None:
    points, full, compressed, u16 = _q_time_grid_regression_inputs()
    if failure_kind == "parity":
        full[points[0][0]]["jit_cache"] += 2.0 * pair.PARITY_ATOL
        expected_check = "q_time_grid_direct_cache_jit_parity"
    elif failure_kind == "beta":
        name = points[1][0]
        full_value = full[name]["direct"]
        compressed[name] = {
            path: full_value + 0.02 * 500.0 for path in compressed[name]
        }
        expected_check = "q_time_grid_compression_accuracy"
    else:
        name = points[-1][0]
        full_value = full[name]["direct"]
        compressed[name] = {
            path: full_value + 2.0 * pair.MAX_REFERENCE_ABS_LOGL_DELTA
            for path in compressed[name]
        }
        expected_check = "q_time_grid_compression_accuracy"

    checks = pair._q_time_grid_regression_check(
        points,
        full,
        compressed,
        u16,
        bank_max_log_likelihood=1000.0,
    )["checks"]

    assert checks[expected_check]["passed"] is False


def test_waveform_cache_dependency_contract_keeps_q_and_time_slow() -> None:
    waveform = SimpleNamespace(parameter_names=("M_c", "eta", "d_L", "iota"))
    full = SimpleNamespace(
        waveform=waveform,
        waveform_cacheable_parameter_names=frozenset(("d_L", "iota")),
    )
    compressed = copy.copy(full)

    passed = pair._waveform_cache_dependency_check(full, compressed)

    assert passed["passed"] is True
    assert passed["per_likelihood"][pair.FULL.kind]["eta_is_cacheable"] is False
    assert passed["per_likelihood"][pair.HETERODYNE.kind]["t_c_is_cacheable"] is False

    compressed.waveform_cacheable_parameter_names = frozenset(("d_L", "iota", "eta"))
    failed = pair._waveform_cache_dependency_check(full, compressed)
    assert failed["passed"] is False
    assert failed["per_likelihood"][pair.HETERODYNE.kind]["eta_is_cacheable"] is True


class _NameTransform:
    """Minimal transform surface exercised by production dependency inference."""

    def __init__(self, before: tuple[str, ...], after: tuple[str, ...]) -> None:
        self.name_mapping = (before, after)

    def propagate_name(self, names: tuple[str, ...]) -> tuple[str, ...]:
        remaining = [name for name in names if name not in self.name_mapping[0]]
        return (*remaining, *self.name_mapping[1])


def _fixed_work_components() -> dict[str, Any]:
    return {
        "spec": {"blocks": pair.FAST_RIDGE_INTRINSIC_BLOCKS},
        "prior": SimpleNamespace(parameter_names=pair.POSITION_FIELDS),
        "sample_transforms": [
            _NameTransform(("d_L",), ("d_hat",)),
            _NameTransform(("iota",), ("cos_iota",)),
            _NameTransform(("ra", "dec"), ("zenith", "azimuth")),
        ],
        "likelihood_transforms": [
            _NameTransform(("q",), ("eta",)),
            _NameTransform(
                ("s1_mag", "s1_theta", "s1_phi"),
                ("s1_x", "s1_y", "s1_z"),
            ),
            _NameTransform(
                ("s2_mag", "s2_theta", "s2_phi"),
                ("s2_x", "s2_y", "s2_z"),
            ),
        ],
    }


def _fixed_work_likelihood() -> SimpleNamespace:
    waveform = SimpleNamespace(
        parameter_names=(
            "M_c",
            "eta",
            "s1_x",
            "s1_y",
            "s1_z",
            "s2_x",
            "s2_y",
            "s2_z",
            "lambda_1",
            "lambda_2",
            "d_L",
            "phase_c",
            "iota",
        )
    )
    return SimpleNamespace(
        waveform=waveform,
        waveform_cacheable_parameter_names=frozenset(("d_L", "iota")),
        fixed_parameters={},
        time_marginalization=True,
    )


def test_fixed_work_cache_schedule_uses_production_dependency_inference() -> None:
    full = _fixed_work_likelihood()
    compressed = copy.copy(full)

    check = pair._fixed_work_cache_schedule_check(
        full,
        compressed,
        _fixed_work_components(),
        blocking_scheme="fast-ridge-intrinsic",
    )

    assert check["passed"] is True
    assert check["observed"]["block_sizes"] == [8, 1, 1, 2, 1, 2]
    assert check["observed"]["rebuild_required_by_block"] == {
        pair.FULL.kind: [True, True, True, False, False, False],
        pair.HETERODYNE.kind: [True, True, True, False, False, False],
    }
    assert check["observed"]["total_slice_updates"] == 15
    assert check["observed"]["waveform_rebuild_slice_updates"] == 10
    assert check["observed"]["cache_hit_slice_updates"] == 5
    assert check["observed"]["num_gibbs_sweeps"] == 1
    assert check["observed"]["num_inner_steps_per_dim"] == 1
    for arm in (pair.FULL.kind, pair.HETERODYNE.kind):
        dependency = check["observed"]["per_likelihood"][arm]
        assert dependency["eta_is_waveform_parameter"] is True
        assert dependency["eta_is_cacheable"] is False
        assert dependency["q_requires_waveform_rebuild"] is True
        assert dependency["q_block_requires_waveform_rebuild"] is True
        assert dependency["t_c_is_sampling_parameter"] is False
        assert dependency["t_c_is_marginalized"] is True


def test_netsky_fixed_work_includes_two_cache_hit_bridge_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pair, "NUM_GIBBS_SWEEPS", 2)
    components = _fixed_work_components()
    components["spec"]["blocks"] = pair.NETSKY_BLOCKS
    components["sample_transforms"].extend(
        [
            _NameTransform(("zenith",), ("cos_zenith",)),
            _NameTransform(("d_hat",), ("log_d_hat",)),
        ]
    )
    components["bridge_blocks"] = pair.NETSKY_BRIDGE_BLOCKS

    check = pair._fixed_work_cache_schedule_check(
        _fixed_work_likelihood(),
        _fixed_work_likelihood(),
        components,
        blocking_scheme=pair.NETSKY_SCHEME,
    )

    assert check["passed"] is True
    observed = check["observed"]
    assert observed["bridge_blocks"] == pair.NETSKY_BRIDGE_BLOCKS
    assert observed["total_updates"] == 32
    assert observed["total_slice_updates"] == 32
    assert observed["waveform_rebuild_slice_updates"] == 20
    assert observed["cache_hit_slice_updates"] == 12
    assert observed["bridge_slice_updates"] == 2


def test_fixed_work_cache_schedule_validates_covariance_basis_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pair, "DIRECTION_MODE", "covariance-basis-8d")

    check = pair._fixed_work_cache_schedule_check(
        _fixed_work_likelihood(),
        _fixed_work_likelihood(),
        _fixed_work_components(),
        blocking_scheme="fast-ridge-intrinsic",
    )

    assert check["passed"] is True
    assert check["expected"]["direction_mode"] == "covariance-basis-8d"
    assert check["observed"]["direction_mode"] == "covariance-basis-8d"


def test_fixed_work_cache_schedule_pins_five_step_intrinsic_budget() -> None:
    check = pair._fixed_work_cache_schedule_check(
        _fixed_work_likelihood(),
        _fixed_work_likelihood(),
        _fixed_work_components(),
        blocking_scheme="fast-ridge-intrinsic-5step",
    )

    assert check["passed"] is True
    assert check["observed"]["num_slice_steps_by_block"] == [5, 1, 1, 2, 1, 2]
    assert check["observed"]["resolved_slice_steps_by_block"] == [5, 1, 1, 2, 1, 2]
    assert check["observed"]["total_slice_updates"] == 12
    assert check["observed"]["waveform_rebuild_slice_updates"] == 7
    assert check["observed"]["cache_hit_slice_updates"] == 5


def test_fixed_work_cache_schedule_pins_periodic_mh_hybrid_budget() -> None:
    check = pair._fixed_work_cache_schedule_check(
        _fixed_work_likelihood(),
        _fixed_work_likelihood(),
        _fixed_work_components(),
        blocking_scheme="fast-ridge-intrinsic-periodic-mh",
    )

    assert check["passed"] is True
    assert check["observed"]["num_slice_steps_by_block"] is None
    assert check["observed"]["block_kernel_modes"] == [
        "slice",
        "periodic-uniform-independence",
        "periodic-uniform-independence",
        "slice",
        "periodic-uniform-independence",
        "slice",
    ]
    assert check["observed"]["total_updates"] == 15
    assert check["observed"]["total_slice_updates"] == 12
    assert check["observed"]["waveform_rebuild_slice_updates"] == 8
    assert check["observed"]["cache_hit_slice_updates"] == 4
    assert check["observed"]["periodic_independence_attempts"] == 3
    assert check["observed"]["waveform_rebuild_periodic_independence_attempts"] == 2
    assert check["observed"]["cache_hit_periodic_independence_attempts"] == 1
    assert check["observed"]["cache_segments"] == 2


def test_fixed_work_cache_schedule_pins_periodic_mh_cde4_budget() -> None:
    check = pair._fixed_work_cache_schedule_check(
        _fixed_work_likelihood(),
        _fixed_work_likelihood(),
        _fixed_work_components(),
        blocking_scheme="fast-ridge-intrinsic-periodic-mh-cde4",
    )

    assert check["passed"] is True
    assert check["observed"]["total_updates"] == 19
    assert check["observed"]["total_slice_updates"] == 12
    assert check["observed"]["periodic_independence_attempts"] == 3
    assert check["observed"]["complementary_de_attempts"] == 4
    assert check["observed"]["waveform_rebuild_complementary_de_attempts"] == 4
    assert check["observed"]["cache_hit_complementary_de_attempts"] == 0
    assert check["observed"]["complementary_de_jump_block"] == {
        "parameters": list(pair.FAST_RIDGE_INTRINSIC_BLOCKS[0]),
        "attempts": 4,
    }
    assert check["observed"]["complementary_de_gamma"] == 1.0
    assert check["observed"]["complementary_de_insert_after_block"] == 0
    assert check["observed"]["cache_segments"] == 2


def test_fixed_work_cache_schedule_pins_periodic_mh_cde8_budget() -> None:
    check = pair._fixed_work_cache_schedule_check(
        _fixed_work_likelihood(),
        _fixed_work_likelihood(),
        _fixed_work_components(),
        blocking_scheme="fast-ridge-intrinsic-periodic-mh-cde8",
    )

    assert check["passed"] is True
    assert check["observed"]["total_updates"] == 23
    assert check["observed"]["total_slice_updates"] == 12
    assert check["observed"]["periodic_independence_attempts"] == 3
    assert check["observed"]["complementary_de_attempts"] == 8
    assert check["observed"]["waveform_rebuild_complementary_de_attempts"] == 8
    assert check["observed"]["cache_hit_complementary_de_attempts"] == 0
    assert check["observed"]["complementary_de_jump_block"] == {
        "parameters": list(pair.FAST_RIDGE_INTRINSIC_BLOCKS[0]),
        "attempts": 8,
    }
    assert check["observed"]["complementary_de_gamma"] == 1.0
    assert check["observed"]["complementary_de_insert_after_block"] == 0
    assert check["observed"]["cache_segments"] == 2


def test_periodic_move_preflight_changes_angles_and_pins_cache_paths() -> None:
    reference = {
        "likelihood_parameters": {
            "M_c": 1.19,
            "eta": 0.24,
            "s1_x": 0.01,
            "s1_y": 0.02,
            "s1_z": 0.03,
            "s2_x": -0.02,
            "s2_y": 0.01,
            "s2_z": -0.01,
            "iota": 0.5,
            "lambda_1": 100.0,
            "lambda_2": 200.0,
            "d_L": 40.0,
            "ra": 1.0,
            "dec": 0.2,
            "psi": 0.3,
            "t_c": 0.0,
            "phase_c": 0.0,
        }
    }

    parent, own_cache_points, foreign_cache_points = pair._periodic_move_points(
        reference
    )

    assert [name for name, _ in own_cache_points] == [
        "periodic-move-s1_phi",
        "periodic-move-s2_phi",
        "periodic-move-psi",
    ]
    assert [name for name, _ in foreign_cache_points] == ["periodic-move-psi"]
    points = dict(own_cache_points)
    assert points["periodic-move-s1_phi"]["s1_z"] == parent["s1_z"]
    assert points["periodic-move-s1_phi"]["s1_x"] != parent["s1_x"]
    assert points["periodic-move-s2_phi"]["s2_y"] != parent["s2_y"]
    assert points["periodic-move-psi"]["psi"] != parent["psi"]
    assert foreign_cache_points[0][1] == points["periodic-move-psi"]

    exact_paths = {
        name: {"direct": 1.0, "cache": 1.0, "jit_direct": 1.0, "jit_cache": 1.0}
        for name, _ in own_cache_points
    }
    exact_foreign = {
        "periodic-move-psi": {
            "direct": 1.0,
            "foreign_cache": 1.0,
            "jit_direct": 1.0,
            "jit_foreign_cache": 1.0,
        }
    }
    check = pair._periodic_move_cache_parity_check(
        exact_paths,
        exact_paths,
        exact_foreign,
        exact_foreign,
    )
    assert check["passed"] is True
    assert check["expected_cache_policy"] == {
        "s1_phi": "waveform-rebuild",
        "s2_phi": "waveform-rebuild",
        "psi": "cache-hit",
    }

    drifting = copy.deepcopy(exact_paths)
    drifting["periodic-move-s1_phi"]["jit_cache"] += 2.0e-6
    check = pair._periodic_move_cache_parity_check(
        exact_paths,
        drifting,
        exact_foreign,
        exact_foreign,
    )
    assert check["passed"] is False


def test_netsky_preflight_proves_polarization_identity_on_both_likelihoods() -> None:
    reference = {
        "likelihood_parameters": {
            "M_c": 1.19,
            "eta": 0.24,
            "psi": 0.3,
        }
    }

    parent, points = pair._netsky_polarization_identity_points(reference)

    assert [name for name, _ in points] == [
        "polarization-base",
        "polarization-half-period",
    ]
    assert points[0][1] == parent
    assert points[1][1]["psi"] == pytest.approx(
        (parent["psi"] + 0.5 * math.pi) % math.pi
    )

    exact = {
        name: {
            "direct": 12.0,
            "foreign_cache": 12.0,
            "jit_direct": 12.0,
            "jit_foreign_cache": 12.0,
        }
        for name, _ in points
    }
    check = pair._netsky_polarization_identity_check(exact, exact)
    assert check["passed"] is True
    assert set(check["per_likelihood"]) == {pair.FULL.kind, pair.HETERODYNE.kind}

    drifting = copy.deepcopy(exact)
    drifting["polarization-half-period"]["jit_foreign_cache"] += 2.0e-6
    check = pair._netsky_polarization_identity_check(exact, drifting)
    assert check["passed"] is False


def test_source_manifest_hashes_netsky_transitive_dependencies() -> None:
    manifest = pair._source_manifest(pair._repository())
    paths = {item["path"] for item in manifest["files"]}

    assert {
        "benchmarks/injection_campaign/common.py",
        "benchmarks/injection_campaign/folded_results.py",
        "benchmarks/injection_campaign/run_injection.py",
    } <= paths


def test_cde4_preflight_builds_gamma_one_proposal_and_pins_cache_paths() -> None:
    reference, _ = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)

    point, contract = pair._complementary_de_proposal_point(reference)

    parent = contract["parent"]
    donor_a = contract["donor_a"]
    donor_b = contract["donor_b"]
    proposal = contract["proposal"]
    for name in pair.FAST_RIDGE_INTRINSIC_BLOCKS[0]:
        assert proposal[name] == pytest.approx(
            parent[name] + donor_a[name] - donor_b[name]
        )
    for candidate in (donor_a, donor_b, proposal):
        assert candidate["d_L"] == pytest.approx(
            parent["d_L"] * (candidate["M_c"] / parent["M_c"]) ** (5.0 / 6.0)
        )
        assert candidate["d_L"] != parent["d_L"]
    assert contract["fixed_sampling_coordinates"] == ["d_hat"]
    assert "M_c_new/M_c_parent" in contract["physical_distance_relation"]
    assert point == pair._sampling_mapping_to_likelihood(proposal)
    assert contract["policy"] == {
        "attempts_per_replacement": 4,
        "gamma": 1.0,
        "placement": "after-target-block",
        "donor_pair_ordered": True,
        "donors_redrawn_each_attempt": True,
        "donor_source": "strict-live-survivors-pre-batch-snapshot",
        "exclude_original_parent_index": True,
        "exclude_batch_dead_points": True,
        "exclude_evolving_endpoints": True,
        "exclude_simultaneous_newborns": True,
        "contour_acceptance": "strict-greater-than",
        "cache_policy": "waveform-rebuild-transactional-rollback",
        "prior_ratio": "complete-transformed-space-prior",
    }

    exact = {
        "direct": 1.0,
        "cache": 1.0,
        "jit_direct": 1.0,
        "jit_cache": 1.0,
    }
    check = pair._complementary_de_proposal_cache_parity_check(
        {"complementary-de-proposal": exact},
        {"complementary-de-proposal": exact},
        contract,
    )
    assert check["passed"] is True
    drifting = copy.deepcopy(exact)
    drifting["jit_cache"] += 2.0e-6
    check = pair._complementary_de_proposal_cache_parity_check(
        {"complementary-de-proposal": exact},
        {"complementary-de-proposal": drifting},
        contract,
    )
    assert check["passed"] is False


@pytest.mark.parametrize(
    ("scheme", "expected"),
    (
        ("fast-ridge-intrinsic", (False, False)),
        ("fast-ridge-intrinsic-5step", (False, False)),
        ("fast-ridge-intrinsic-periodic-mh", (True, False)),
        ("fast-ridge-intrinsic-periodic-mh-cde4", (True, True)),
        ("fast-ridge-intrinsic-periodic-mh-cde8", (True, True)),
    ),
)
def test_preflight_probe_families_include_h4_periodic_matrix_for_h5(
    scheme: str,
    expected: tuple[bool, bool],
) -> None:
    assert pair._preflight_probe_families(scheme) == expected


@pytest.mark.parametrize(
    ("blocking_scheme", "block_sizes"),
    (
        ("paper", [4, 3, 3, 1, 2, 1, 1]),
        ("fast-ridge", [4, 3, 3, 2, 1, 2]),
        ("fast-ridge-intrinsic", [8, 1, 1, 2, 1, 2]),
    ),
)
def test_fixed_work_cache_schedule_supports_frozen_pair_schemes(
    blocking_scheme: str,
    block_sizes: list[int],
) -> None:
    components = _fixed_work_components()
    components["spec"]["blocks"] = pair.BLOCKS_BY_SCHEME[blocking_scheme]
    if blocking_scheme == "paper":
        components["sample_transforms"] = components["sample_transforms"][2:]

    check = pair._fixed_work_cache_schedule_check(
        _fixed_work_likelihood(),
        _fixed_work_likelihood(),
        components,
        blocking_scheme=blocking_scheme,
    )

    assert check["passed"] is True
    assert check["observed"]["block_sizes"] == block_sizes
    assert check["observed"]["total_slice_updates"] == 15
    assert check["observed"]["waveform_rebuild_slice_updates"] == 10
    assert check["observed"]["cache_hit_slice_updates"] == 5


@pytest.mark.parametrize(
    "failure_kind",
    ("wrong-blocks", "eta-cacheable", "extra-gibbs-sweep", "time-sampled"),
)
def test_fixed_work_cache_schedule_fails_closed_on_mismatch(
    failure_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _fixed_work_components()
    full = _fixed_work_likelihood()
    compressed = copy.copy(full)
    if failure_kind == "wrong-blocks":
        components["spec"]["blocks"] = pair.FAST_RIDGE_BLOCKS
    elif failure_kind == "eta-cacheable":
        compressed.waveform_cacheable_parameter_names = frozenset(
            ("d_L", "iota", "eta")
        )
    elif failure_kind == "extra-gibbs-sweep":
        monkeypatch.setattr(pair, "NUM_GIBBS_SWEEPS", 2)
    else:
        compressed.time_marginalization = False

    check = pair._fixed_work_cache_schedule_check(
        full,
        compressed,
        components,
        blocking_scheme="fast-ridge-intrinsic",
    )

    assert check["passed"] is False


def test_likelihood_factory_forces_correct_waveform_u16_and_5000_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jimgw.core.single_event.likelihood as likelihood_module
    from benchmarks.device_parallel_nss import paper_heterodyne, paper_model

    captured: dict[str, Any] = {}

    class FakeWaveform:
        def __init__(self, *, f_ref: float, time_anchor: str) -> None:
            self.f_ref = f_ref
            self.time_anchor = time_anchor

    class FakeFull:
        pass

    class FakeCompressed:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.phase_marginalization = True
            self.time_marginalization = True
            self.distance_marginalization = False

    monkeypatch.setattr(paper_model, "RippleIMRPhenomPv2NRTidalv2", FakeWaveform)
    monkeypatch.setattr(
        paper_heterodyne,
        "PaperTimeMarginalizedHeterodynedLikelihoodFD",
        FakeCompressed,
    )
    monkeypatch.setattr(likelihood_module, "TransientLikelihoodFD", FakeFull)
    monkeypatch.setattr(
        pair,
        "_likelihood_metadata",
        lambda arm, likelihood, reference_sha: {"kind": arm.kind},
    )

    class FakeBenchmark:
        GPS = 1.0
        F_MIN = pair.F_MIN_HZ
        F_MAX = pair.LIKELIHOOD_F_MAX_HZ

        @staticmethod
        def _analysis_components(
            workload: str,
            jnp: Any,
            ifos: list[Any],
            *,
            blocking_scheme: str,
        ) -> dict[str, Any]:
            del workload, jnp, ifos, blocking_scheme
            return {"waveform": object(), "preserved": True}

        @staticmethod
        def _config_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            return {"workload": pair.WORKLOAD, "blocking_scheme": pair.BLOCKING_SCHEME}

    benchmark = FakeBenchmark()
    reference, reference_sha = pair._load_reference(pair.DEFAULT_REFERENCE_FILE)
    verify = pair._install_scientific_target(
        benchmark,
        pair.HETERODYNE,
        reference,
        reference_sha,
    )
    components = benchmark._analysis_components(
        pair.WORKLOAD,
        object(),
        [],
        blocking_scheme=pair.BLOCKING_SCHEME,
    )
    likelihood_module.TransientLikelihoodFD(
        [],
        waveform=components["waveform"],
        trigger_time=benchmark.GPS,
        f_min=benchmark.F_MIN,
        f_max=benchmark.F_MAX,
        phase_marginalization=True,
        time_marginalization={"tc_range": pair.TC_RANGE_SECONDS},
    )
    config = benchmark._config_report()
    verify()

    assert captured["time_marginalization"] == {
        "tc_range": pair.TC_RANGE_SECONDS,
        "upsample_factor": pair.TIME_UPSAMPLE_FACTOR,
    }
    assert captured["n_bins"] == pair.N_BINS_REQUESTED
    assert captured["reference_parameters"] == reference["likelihood_parameters"]
    assert config["carrier_time_anchor"] == pair.CARRIER_TIME_ANCHOR
    assert config["likelihood"] == {"kind": pair.HETERODYNE.kind}


def test_pair_comparison_allows_different_scientific_outputs(tmp_path: Path) -> None:
    compressed_path = tmp_path / "compressed.npz"
    full_path = tmp_path / "full.npz"
    _weighted(compressed_path, q_shift=0.01)
    _weighted(full_path, q_shift=0.0)
    reports = {
        pair.HETERODYNE.kind: _report(pair.HETERODYNE, compressed_path),
        pair.FULL.kind: _report(pair.FULL, full_path),
    }

    comparison = pair.compare_pair(
        reports,
        {
            pair.HETERODYNE.kind: compressed_path,
            pair.FULL.kind: full_path,
        },
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )

    assert comparison["strict_pass"] is True
    assert (
        reports[pair.HETERODYNE.kind]["results"]["log_Z"]
        != reports[pair.FULL.kind]["results"]["log_Z"]
    )


def test_netsky_pair_validation_separates_posterior_and_folded_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = pair._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--prefix",
            str(tmp_path / "netsky"),
            "--blocking-scheme",
            pair.NETSKY_SCHEME,
            "--n-devices",
            "1",
        ]
    )
    for name in ("BLOCKING_SCHEME", "N_DEVICES", "NUM_GIBBS_SWEEPS", "PAPER_BLOCKS"):
        monkeypatch.setattr(pair, name, getattr(pair, name))
    pair._configure_runtime(args)
    reports: dict[str, dict[str, Any]] = {}
    posterior_paths: dict[str, Path] = {}
    folded_paths: dict[str, Path] = {}
    for arm in pair.ARMS:
        posterior = tmp_path / f"{arm.kind}.weighted.npz"
        folded = tmp_path / f"{arm.kind}.folded.npz"
        _netsky_posterior(posterior)
        _netsky_folded_diagnostics(folded)
        posterior_paths[arm.kind] = posterior
        folded_paths[arm.kind] = folded
        reports[arm.kind] = _netsky_report(arm, posterior, folded)

    comparison = pair.compare_pair(
        reports,
        posterior_paths,
        folded_paths=folded_paths,
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )

    assert comparison["strict_pass"] is True
    for arm in pair.ARMS:
        validation = comparison["arm_validation"][arm.kind]
        assert validation["weighted_artifact"]["count"] == 2 * (
            pair.N_LIVE + pair.PREFIX_STEPS * pair.N_DELETE
        )
        assert validation["folded_nested_diagnostics"]["count"] == (
            pair.N_LIVE + pair.PREFIX_STEPS * pair.N_DELETE
        )


def test_netsky_posterior_rejects_folded_birth_likelihood(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pair, "BLOCKING_SCHEME", pair.NETSKY_SCHEME)
    posterior = tmp_path / "posterior.npz"
    folded = tmp_path / "folded.npz"
    _netsky_posterior(posterior)
    _netsky_folded_diagnostics(folded)
    report = _netsky_report(pair.FULL, posterior, folded)
    with np.load(posterior, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["log_likelihood_birth"] = arrays["log_likelihood"] - 1.0
    np.savez(posterior, **arrays)
    report["results"]["posterior_artifact"] = _artifact_metadata(
        posterior,
        space="prior",
        weighting=pair.UNFOLDED_POSTERIOR_WEIGHTING,
    )

    validation = pair._validate_weighted_artifact(posterior, report)

    assert validation["passed"] is False
    assert any(
        "must not contain log_likelihood_birth" in error
        for error in validation["errors"]
    )


def test_netsky_folded_validator_recomputes_rows_and_insertion_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pair, "BLOCKING_SCHEME", pair.NETSKY_SCHEME)
    posterior = tmp_path / "posterior.npz"
    folded = tmp_path / "folded.npz"
    _netsky_posterior(posterior)
    _netsky_folded_diagnostics(folded)
    report = _netsky_report(pair.FULL, posterior, folded)
    report["results"]["n_iterations"] += 1
    report["results"]["insertion_index_diagnostic"]["p_value"] = 0.5

    validation = pair._validate_folded_nested_diagnostics(folded, report)

    assert validation["passed"] is False
    assert any(
        "n_live + n_iterations*n_delete" in error for error in validation["errors"]
    )
    assert any(
        "insertion-index diagnostic mismatch" in error for error in validation["errors"]
    )


def test_netsky_report_rejects_config_telemetry_projection_and_timing_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pair, "BLOCKING_SCHEME", pair.NETSKY_SCHEME)
    posterior = tmp_path / "posterior.npz"
    folded = tmp_path / "folded.npz"
    _netsky_posterior(posterior)
    _netsky_folded_diagnostics(folded)
    report = _netsky_report(pair.FULL, posterior, folded)
    del report["config"]["fold_symmetry"]["azimuth_reflection_center"]
    quotient = report["results"]["quotient_fold"]
    quotient["expected_nonidentity_mass"] = 0.6
    quotient["projection_accounting"]["total_true_image_projections"] -= 1
    invalid_expanded_count = 8 * quotient["folded_points"] + 1
    report["results"]["posterior_samples"] = invalid_expanded_count
    report["results"]["posterior_artifact"]["count"] = invalid_expanded_count
    report["timing_seconds"]["fold_unfold_postprocessing"] = 0.6

    errors = pair._validate_netsky_report_semantics(report)

    assert any("completed fold configuration" in error for error in errors)
    assert any("non-identity mass" in error for error in errors)
    assert any("projection_accounting drifted" in error for error in errors)
    assert any("expanded posterior count" in error for error in errors)
    assert any("does not contain fold/unfold time" in error for error in errors)


def test_pair_comparison_fails_on_initial_positions_or_u1(tmp_path: Path) -> None:
    compressed_path = tmp_path / "compressed.npz"
    full_path = tmp_path / "full.npz"
    _weighted(compressed_path)
    _weighted(full_path)
    reports = {
        pair.HETERODYNE.kind: _report(pair.HETERODYNE, compressed_path),
        pair.FULL.kind: _report(pair.FULL, full_path),
    }
    reports[pair.HETERODYNE.kind]["config"]["initial_positions_sha256"] = "changed"
    reports[pair.HETERODYNE.kind]["config"]["likelihood"]["time_grid"][
        "upsample_factor"
    ] = 1
    reports[pair.HETERODYNE.kind]["config"]["sha256"] = pair._config_sha256(
        reports[pair.HETERODYNE.kind]["config"]
    )

    comparison = pair.compare_pair(
        reports,
        {
            pair.HETERODYNE.kind: compressed_path,
            pair.FULL.kind: full_path,
        },
        mode="prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )

    assert comparison["strict_pass"] is False
    errors = comparison["arm_validation"][pair.HETERODYNE.kind]["report_errors"]
    assert any("upsample" in error for error in errors)
    setup_check = next(
        check
        for check in comparison["pairing_checks"]
        if check["name"] == "scientific_config_except_likelihood"
    )
    assert setup_check["passed"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("log_Z", math.nan, "log_Z is not finite"),
        ("log_Z_error", 0.0, "log_Z_error is not positive"),
        ("log_Z_error", math.inf, "log_Z_error is not positive"),
        ("n_iterations", 0, "n_iterations is not a positive"),
        (
            "n_likelihood_evaluations",
            0,
            "n_likelihood_evaluations is not a positive",
        ),
    ],
)
def test_arm_report_rejects_invalid_evidence_and_work_counters(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    weighted = tmp_path / "weighted.npz"
    _weighted(weighted)
    report = _report(pair.HETERODYNE, weighted)
    report["results"][field] = value

    errors = pair._validate_arm_report(
        report,
        pair.HETERODYNE,
        "prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )

    assert any(message in error for error in errors)


def test_arm_report_couples_iterations_evaluations_and_nested_count(
    tmp_path: Path,
) -> None:
    weighted = tmp_path / "weighted.npz"
    _weighted(weighted)
    report = _report(pair.FULL, weighted)
    report["results"]["nested_artifact"]["count"] += 1
    report["results"]["n_likelihood_evaluations"] = 1

    errors = pair._validate_arm_report(
        report,
        pair.FULL,
        "prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )

    assert any("n_live + n_iterations*n_delete" in error for error in errors)
    assert any("below nested point count" in error for error in errors)


def test_arm_report_couples_likelihood_class_and_reference_hash(
    tmp_path: Path,
) -> None:
    weighted = tmp_path / "weighted.npz"
    _weighted(weighted)
    report = _report(pair.HETERODYNE, weighted)
    report["config"]["likelihood"]["class"] = "wrong.Class"
    report["config"]["likelihood"]["reference_json_sha256"] = "x" * 64
    report["config"]["sha256"] = pair._config_sha256(report["config"])

    errors = pair._validate_arm_report(
        report,
        pair.HETERODYNE,
        "prefix",
        data_sha256="d" * 64,
        reference_sha256="r" * 64,
        source_sha256="s" * 64,
        simulate_cpu=True,
    )

    assert any("config.likelihood.class" in error for error in errors)
    assert any("config.likelihood.reference_json_sha256" in error for error in errors)


@pytest.mark.parametrize("birth_failure", ["equal", "missing_initial"])
def test_weighted_artifact_rejects_invalid_birth_death_contract(
    tmp_path: Path,
    birth_failure: str,
) -> None:
    path = tmp_path / "bad-birth.npz"
    _weighted(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if birth_failure == "equal":
        index = pair.N_LIVE
        arrays["log_likelihood_birth"][index] = arrays["log_likelihood"][index]
    else:
        arrays["log_likelihood_birth"][0] = arrays["log_likelihood"][0] - 1.0
    np.savez(path, **arrays)
    report = _report(pair.FULL, path)

    validation = pair._validate_weighted_artifact(path, report)

    assert validation["passed"] is False
    expected = "strictly below" if birth_failure == "equal" else "exactly n_live"
    assert any(expected in error for error in validation["errors"])


def test_weighted_artifact_rejects_report_metadata_drift(tmp_path: Path) -> None:
    path = tmp_path / "weighted.npz"
    _weighted(path)
    report = _report(pair.FULL, path)
    report["results"]["nested_artifact"]["count"] += 1
    report["results"]["nested_artifact"]["fields"] = ["wrong"]
    report["results"]["nested_artifact"]["dtypes"] = {"wrong": "float64"}

    validation = pair._validate_weighted_artifact(path, report)

    assert validation["passed"] is False
    assert any("report count" in error for error in validation["errors"])
    assert any("report fields" in error for error in validation["errors"])
    assert any("report dtypes" in error for error in validation["errors"])


def test_weighted_artifact_rejects_unnormalized_weights(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    _weighted(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["log_weights"] = np.full(
        len(arrays["log_weights"]),
        -math.log(len(arrays["log_weights"])) + math.log(2.0),
    )
    np.savez(path, **arrays)
    report = {"results": {"nested_artifact": {"sha256": pair._sha256(path)}}}

    validation = pair._validate_weighted_artifact(path, report)

    assert validation["passed"] is False
    assert any("not normalized" in error for error in validation["errors"])
