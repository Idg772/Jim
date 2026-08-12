"""Evaluate the fixed D=4/M=1 IMRPhenomD carrier-anchor diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.injection_campaign.common import (
    MARGINALIZED_PARAMETERS,
    PARAMETERS,
    atomic_write_csv,
    atomic_write_json,
    file_sha256,
    load_manifest,
    posterior_rank,
    read_catalogue,
    result_dir,
)
from benchmarks.injection_campaign.prepare_time_anchor_diagnostic import (
    DEFAULT_SOURCE_IDS,
    DIAGNOSTIC_TIME_ANCHOR,
    SOURCE_TIME_ANCHOR,
)

EXPECTED_CASES = 4
RANK_PARAMETERS = ("q", "ra", "dec", "t_c")
TRUTH_PARAMETERS = (*PARAMETERS, *MARGINALIZED_PARAMETERS)
REPORT_SCHEMA_VERSION = 1
REPORT_DIRECTORY = "diagnostic"
REPORT_JSON = "time-anchor-diagnostic.json"
REPORT_CSV = "time-anchor-diagnostic.csv"
RANK_ROUNDOFF_TOLERANCE = 1.0e-12
RANK_RECOMPUTE_TOLERANCE = 1.0e-12
LOG_WEIGHT_NORMALIZATION_TOLERANCE = 1.0e-10
EXTREME_TAIL_PROBABILITY = 1.0e-3
CENTRAL_RANK_LOW = 0.05
CENTRAL_RANK_HIGH = 0.95
MAX_EXTREME_CELLS = 1
MIN_CENTRAL_CELLS = 10
MAX_ENDPOINT_CELLS = 0
MAX_AUTHOR_LEVERAGE = 0.25
MAX_ANCHOR_SD_RATIO = 0.1
ANCHOR_BATCH_SIZE = 2048
FROZEN_LOCAL_LEVERAGE = {
    97: 1.4025,
    12: 1.2028,
    41: 1.7647,
    32: 1.9261,
}
FROZEN_LOCAL_LEVERAGE_TOLERANCE = 5.0e-4

CSV_FIELDS = (
    "diagnostic_id",
    "source_injection_id",
    "source_q_rank",
    "control_q_rank",
    "author_q_rank",
    "source_ra_rank",
    "control_ra_rank",
    "author_ra_rank",
    "source_dec_rank",
    "control_dec_rank",
    "author_dec_rank",
    "source_t_c_rank",
    "control_t_c_rank",
    "author_t_c_rank",
    "source_corr_t_c_delta_anchor",
    "control_corr_t_c_delta_anchor",
    "author_corr_t_c_delta_anchor",
    "source_applied_anchor_leverage",
    "control_applied_anchor_leverage",
    "author_applied_anchor_leverage",
    "current_code_leverage_reduction_fraction",
    "leverage_reduction_fraction",
    "author_to_local_anchor_sd_ratio_on_author_posterior",
    "author_leverage_gate_passed",
    "anchor_sd_ratio_gate_passed",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--source-campaign",
        type=Path,
        default=None,
        help="Archived local-anchor D=4/M=1 source campaign.",
    )
    parser.add_argument(
        "--local-control-campaign",
        type=Path,
        default=None,
        help=(
            "Optional current-code local-anchor control. It is provenance-checked "
            "and reported, but the preregistered comparison remains archived "
            "source versus author anchor."
        ),
    )
    return parser.parse_args(argv)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _exact_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an exact integer")
    return value


def _normalized_rank(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        rank = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(rank):
        raise ValueError(f"{field} must be finite")
    if rank < -RANK_ROUNDOFF_TOLERANCE or rank > 1.0 + RANK_ROUNDOFF_TOLERANCE:
        raise ValueError(f"{field} lies outside [0, 1] beyond roundoff")
    if rank < 0.0:
        return 0.0
    if rank > 1.0:
        return 1.0
    return rank


def _rank_metrics(rank: float) -> dict[str, Any]:
    rank = _normalized_rank(rank, field="rank")
    tail = 2.0 * min(rank, 1.0 - rank)
    return {
        "rank": rank,
        "two_sided_tail_probability": tail,
        "extreme_tail": tail < EXTREME_TAIL_PROBABILITY,
        "central_90": CENTRAL_RANK_LOW <= rank <= CENTRAL_RANK_HIGH,
        "exact_endpoint": rank == 0.0 or rank == 1.0,
    }


def _normalized_weights(log_weights: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label}: log_weights must be non-empty and 1D")
    if (
        np.any(np.isnan(values))
        or np.any(np.isposinf(values))
        or not np.any(np.isfinite(values))
    ):
        raise ValueError(f"{label}: invalid log_weights")
    maximum = float(np.max(values[np.isfinite(values)]))
    unnormalized = np.exp(values - maximum)
    normalizer = float(np.sum(unnormalized))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError(f"{label}: log_weights cannot be normalized")
    log_normalizer = maximum + math.log(normalizer)
    if not math.isclose(
        log_normalizer,
        0.0,
        rel_tol=0.0,
        abs_tol=LOG_WEIGHT_NORMALIZATION_TOLERANCE,
    ):
        raise ValueError(f"{label}: stored log_weights are not normalized")
    return unnormalized / normalizer


def _weighted_moments(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    label: str,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.shape != weights.shape:
        raise ValueError(f"{label}: weighted arrays are misaligned")
    mean_x = float(np.sum(weights * x))
    mean_y = float(np.sum(weights * y))
    centered_x = x - mean_x
    centered_y = y - mean_y
    variance_x = float(np.sum(weights * centered_x * centered_x))
    variance_y = float(np.sum(weights * centered_y * centered_y))
    covariance = float(np.sum(weights * centered_x * centered_y))
    if variance_x <= 0.0 or variance_y <= 0.0:
        raise ValueError(f"{label}: weighted variance is not positive")
    correlation = covariance / math.sqrt(variance_x * variance_y)
    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "variance_x": variance_x,
        "variance_y": variance_y,
        "sd_x": math.sqrt(variance_x),
        "sd_y": math.sqrt(variance_y),
        "covariance": covariance,
        "correlation": float(np.clip(correlation, -1.0, 1.0)),
    }


def _weighted_residual(
    values: np.ndarray,
    delays: np.ndarray,
    weights: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    delays = np.asarray(delays, dtype=np.float64)
    if delays.shape != (values.size, 3) or weights.shape != values.shape:
        raise ValueError(f"{label}: residualization arrays are misaligned")
    delay_mean = np.sum(weights[:, None] * delays, axis=0)
    centered = delays - delay_mean
    delay_sd = np.sqrt(np.sum(weights[:, None] * centered * centered, axis=0))
    if np.any(delay_sd <= 0.0) or not np.all(np.isfinite(delay_sd)):
        raise ValueError(f"{label}: detector-delay variance is not positive")
    design = np.column_stack([np.ones(values.size), centered / delay_sd])
    root_weight = np.sqrt(weights)
    weighted_design = design * root_weight[:, None]
    weighted_values = values * root_weight
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted_design,
        weighted_values,
        rcond=1.0e-12,
    )
    if rank < 2:
        raise ValueError(f"{label}: detector-delay design is degenerate")
    residual = values - design @ coefficients
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else math.inf
    )
    if not np.all(np.isfinite(residual)) or not math.isfinite(condition):
        raise ValueError(f"{label}: residualization is non-finite")
    return residual, {
        "design_columns": ["intercept", "delay_H1", "delay_L1", "delay_V1"],
        "numerical_rank": int(rank),
        "condition_number": condition,
        "coefficients_on_standardized_delays": coefficients.tolist(),
        "delay_weighted_means_seconds": delay_mean.tolist(),
        "delay_weighted_sds_seconds": delay_sd.tolist(),
    }


def _partial_geometry(
    t_c: np.ndarray,
    anchor: np.ndarray,
    delays: np.ndarray,
    weights: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    residual_t, t_fit = _weighted_residual(
        t_c, delays, weights, label=f"{label} t_c"
    )
    residual_anchor, anchor_fit = _weighted_residual(
        anchor, delays, weights, label=f"{label} anchor"
    )
    moments = _weighted_moments(
        residual_t,
        residual_anchor,
        weights,
        label=f"{label} residual geometry",
    )
    leverage = abs(moments["covariance"] / moments["variance_x"])
    return {
        "leverage_abs_cov_anchor_tc_over_var_tc": leverage,
        "sd_ratio_anchor_over_t_c": moments["sd_y"] / moments["sd_x"],
        "rho": moments["correlation"],
        "partial_r_squared": moments["correlation"] ** 2,
        "residual_t_c_sd_seconds": moments["sd_x"],
        "residual_anchor_sd_seconds": moments["sd_y"],
        "t_c_delay_fit": t_fit,
        "anchor_delay_fit": anchor_fit,
    }


def _source_campaign_path(
    campaign_dir: Path,
    provenance: Mapping[str, Any],
    explicit: Path | None,
) -> Path:
    source_name = provenance.get("source_campaign")
    if (
        not isinstance(source_name, str)
        or not source_name
        or Path(source_name).name != source_name
    ):
        raise ValueError("source campaign name must be a plain directory name")
    source = (
        explicit.expanduser().resolve()
        if explicit is not None
        else (campaign_dir.parent / source_name).resolve()
    )
    if source.name != source_name:
        raise ValueError("explicit source campaign disagrees with provenance")
    if not source.is_dir():
        raise ValueError(f"source campaign does not exist: {source}")
    return source


def _scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    for key in ("campaign", "paper_configuration", "carrier_time_anchor"):
        normalized.pop(key, None)
    timing = normalized.get("timing")
    if isinstance(timing, dict):
        timing.pop("selected_events", None)
    return normalized


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    *,
    campaign_dir: Path,
    source_campaign: Path,
) -> Mapping[str, Any]:
    if (
        _exact_int(manifest.get("n_injections"), field="manifest.n_injections")
        != EXPECTED_CASES
        or _exact_int(
            manifest.get("catalogue_size"), field="manifest.catalogue_size"
        )
        != EXPECTED_CASES
    ):
        raise ValueError(f"diagnostic must contain exactly {EXPECTED_CASES} cases")
    selection = _mapping(manifest.get("selection"), field="manifest.selection")
    if (
        selection.get("source_injection_ids") != list(DEFAULT_SOURCE_IDS)
        or selection.get("start_inclusive") != 0
        or selection.get("stop_exclusive") != EXPECTED_CASES
    ):
        raise ValueError("diagnostic selection is not the frozen sentinel order")

    source_config = _mapping(source_manifest.get("config"), field="source config")
    config = _mapping(manifest.get("config"), field="diagnostic config")
    for label, candidate in (("source", source_config), ("diagnostic", config)):
        if (
            candidate.get("n_devices") != 4
            or candidate.get("num_gibbs_sweeps") != 1
            or candidate.get("sampler_scheduler", "fsm") != "fsm"
            or candidate.get("time_marginalization") is not False
            or not isinstance(candidate.get("distance_marginalization"), Mapping)
            or candidate.get("blocks", [])[-1:] != [["t_c"]]
        ):
            raise ValueError(f"{label} is not the expected sampled-t_c D=4/M=1 run")
    if source_config.get("carrier_time_anchor", SOURCE_TIME_ANCHOR) != SOURCE_TIME_ANCHOR:
        raise ValueError("source does not use the local NRTidal-merger anchor")
    if config.get("carrier_time_anchor") != DIAGNOSTIC_TIME_ANCHOR:
        raise ValueError("diagnostic does not use the IMRPhenomD anchor")
    if config.get("campaign") != "coauthor-anchor-d4-m1-pathology-diagnostic":
        raise ValueError("diagnostic campaign label is invalid")
    if config.get("paper_configuration") != "Sharded coauthor-anchor diagnostic":
        raise ValueError("diagnostic paper-configuration label is invalid")
    if _scientific_config(config) != _scientific_config(source_config):
        raise ValueError(
            "diagnostic scientific configuration changes more than the time anchor"
        )
    if manifest.get("psd") != source_manifest.get("psd"):
        raise ValueError("diagnostic PSD inventory differs from source")
    if manifest.get("storage_policy") != source_manifest.get("storage_policy"):
        raise ValueError("diagnostic storage policy differs from source")

    scope = _mapping(
        manifest.get("reproduction_scope"), field="manifest.reproduction_scope"
    )
    if (
        scope.get("iid_prior_predictive_catalogue") is not False
        or scope.get("pp_calibration_eligible") is not False
    ):
        raise ValueError("targeted diagnostic must be non-IID and nonpublication")
    pin = _mapping(
        manifest.get("implementation_diagnostic"),
        field="manifest.implementation_diagnostic",
    )
    changed = _mapping(pin.get("changed_variables"), field="changed_variables")
    if dict(changed) != {
        "carrier_time_anchor": {
            "source": SOURCE_TIME_ANCHOR,
            "diagnostic": DIAGNOSTIC_TIME_ANCHOR,
        }
    }:
        raise ValueError("implementation diagnostic records the wrong changed variable")
    if (
        pin.get("implementation_label") != "candidate"
        or pin.get("sampler_scheduler") != "fsm"
        or pin.get("source_campaign_config_sha256")
        != source_manifest.get("config_sha256")
    ):
        raise ValueError("implementation diagnostic pin is invalid")

    catalogue = _mapping(manifest.get("catalogue"), field="manifest.catalogue")
    provenance = _mapping(
        catalogue.get("provenance"), field="manifest.catalogue.provenance"
    )
    expected_hashes = {
        "source_manifest_sha256": file_sha256(source_campaign / "manifest.json"),
        "source_config_sha256": source_manifest.get("config_sha256"),
        "source_catalogue_sha256": _mapping(
            source_manifest.get("catalogue"), field="source catalogue"
        ).get("sha256"),
    }
    for field, expected in expected_hashes.items():
        if provenance.get(field) != expected:
            raise ValueError(f"diagnostic provenance {field} mismatch")
    if file_sha256(campaign_dir / "manifest.json") == provenance.get(
        "source_manifest_sha256"
    ):
        raise ValueError("source and diagnostic manifests unexpectedly match")
    return provenance


def _validate_mapping(
    manifest: Mapping[str, Any],
    *,
    source_campaign: Path,
    diagnostic_catalogue: Sequence[Mapping[str, Any]],
    source_catalogue: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    catalogue = _mapping(manifest.get("catalogue"), field="manifest.catalogue")
    provenance = _mapping(catalogue.get("provenance"), field="catalogue.provenance")
    raw_mapping = provenance.get("mapping")
    if not isinstance(raw_mapping, list) or len(raw_mapping) != EXPECTED_CASES:
        raise ValueError("diagnostic source mapping is incomplete")
    normalized: list[dict[str, Any]] = []
    for diagnostic_id, (source_id, raw) in enumerate(
        zip(DEFAULT_SOURCE_IDS, raw_mapping, strict=True)
    ):
        entry = _mapping(raw, field=f"mapping[{diagnostic_id}]")
        if (
            entry.get("diagnostic_id") != diagnostic_id
            or entry.get("source_injection_id") != source_id
        ):
            raise ValueError("diagnostic mapping is not in frozen sentinel order")
        source_row = source_catalogue[source_id]
        diagnostic_row = diagnostic_catalogue[diagnostic_id]
        for name, actual in diagnostic_row.items():
            expected = diagnostic_id if name == "injection_id" else source_row[name]
            if actual != expected:
                raise ValueError(
                    f"diagnostic row {diagnostic_id} differs from source in {name}"
                )
        if (
            entry.get("noise_seed") != source_row["noise_seed"]
            or entry.get("sampler_seed") != source_row["sampler_seed"]
        ):
            raise ValueError(f"mapping seed mismatch for diagnostic {diagnostic_id}")
        source_dir = result_dir(source_campaign, source_id)
        summary_path = source_dir / "summary.json"
        posterior_path = source_dir / "posterior.npz"
        if (
            entry.get("source_summary_sha256") != file_sha256(summary_path)
            or entry.get("source_posterior_sha256") != file_sha256(posterior_path)
        ):
            raise ValueError(f"mapping source-result hash mismatch for {source_id}")
        recorded = _mapping(
            entry.get("source_ranks"), field=f"mapping source ranks {source_id}"
        )
        if set(recorded) != set(RANK_PARAMETERS):
            raise ValueError(f"mapping source ranks are incomplete for {source_id}")
        normalized.append(
            {
                "diagnostic_id": diagnostic_id,
                "source_injection_id": source_id,
                "noise_seed": source_row["noise_seed"],
                "sampler_seed": source_row["sampler_seed"],
                "source_summary_sha256": entry["source_summary_sha256"],
                "source_posterior_sha256": entry["source_posterior_sha256"],
                "recorded_source_ranks": {
                    name: _normalized_rank(
                        recorded[name], field=f"mapping source rank {source_id}.{name}"
                    )
                    for name in RANK_PARAMETERS
                },
            }
        )
    return normalized


def _load_result(
    campaign: Path,
    manifest: Mapping[str, Any],
    truth: Mapping[str, Any],
    injection_id: int,
    *,
    diagnostic: bool,
) -> dict[str, Any]:
    directory = result_dir(campaign, injection_id)
    summary_path = directory / "summary.json"
    posterior_path = directory / "posterior.npz"
    if not summary_path.is_file() or not posterior_path.is_file():
        raise ValueError(f"result is incomplete: {directory}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid summary {summary_path}: {error}") from error
    label = f"{'author' if diagnostic else 'source'} result {injection_id}"
    if (
        summary.get("schema_version") != 2
        or summary.get("config_sha256") != manifest.get("config_sha256")
        or summary.get("injection_id") != injection_id
        or summary.get("seeds")
        != {"noise": truth["noise_seed"], "sampler": truth["sampler_seed"]}
    ):
        raise ValueError(f"{label}: summary provenance mismatch")
    summary_truth = _mapping(summary.get("truth"), field=f"{label}.truth")
    if set(summary_truth) != set(TRUTH_PARAMETERS):
        raise ValueError(f"{label}: truth inventory is invalid")
    for name in TRUTH_PARAMETERS:
        if float(summary_truth[name]) != float(truth[name]):
            raise ValueError(f"{label}: truth mismatch for {name}")

    if diagnostic:
        pin = _mapping(
            manifest.get("implementation_diagnostic"),
            field="manifest.implementation_diagnostic",
        )
        implementation = _mapping(
            summary.get("implementation"), field=f"{label}.implementation"
        )
        if (
            implementation.get("label") != pin.get("implementation_label")
            or implementation.get("revision") != pin.get("implementation_revision")
            or implementation.get("tree_sha256")
            != pin.get("implementation_tree_sha256")
        ):
            raise ValueError(f"{label}: implementation pin mismatch")
        devices = _mapping(summary.get("devices"), field=f"{label}.devices")
        if (
            summary.get("simulated_cpu") is not False
            or devices.get("backend") != "gpu"
            or devices.get("requested_count") != 4
            or devices.get("local_count") != 4
        ):
            raise ValueError(f"{label}: result is not a real D=4 GPU run")

    posterior = _mapping(summary.get("posterior"), field=f"{label}.posterior")
    if (
        posterior.get("path") != "posterior.npz"
        or posterior.get("sha256") != file_sha256(posterior_path)
        or posterior.get("bytes") != posterior_path.stat().st_size
        or posterior.get("space") != "prior"
        or posterior.get("weighting")
        != "normalized nested-sampling log weights"
    ):
        raise ValueError(f"{label}: posterior provenance mismatch")
    fields = posterior.get("fields")
    expected_fields = {*PARAMETERS, "log_likelihood", "log_weights"}
    if (
        not isinstance(fields, list)
        or len(fields) != len(set(fields))
        or set(fields) != expected_fields
    ):
        raise ValueError(f"{label}: posterior field inventory is invalid")
    try:
        with np.load(posterior_path, allow_pickle=False) as archive:
            if set(archive.files) != set(fields):
                raise ValueError(f"{label}: NPZ field inventory mismatch")
            arrays = {
                name: np.asarray(archive[name], dtype=np.float64) for name in fields
            }
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as error:
        if str(error).startswith(f"{label}:"):
            raise
        raise ValueError(f"{label}: unreadable posterior: {error}") from error
    sample_count = _exact_int(
        summary.get("posterior_samples"), field=f"{label}.posterior_samples"
    )
    if sample_count < 1 or any(
        values.shape != (sample_count,) for values in arrays.values()
    ):
        raise ValueError(f"{label}: posterior arrays are misaligned")
    for name in (*PARAMETERS, "log_likelihood"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{label}: {name} contains non-finite values")
    weights = _normalized_weights(arrays["log_weights"], label=label)

    stored_ranks = _mapping(summary.get("ranks"), field=f"{label}.ranks")
    if set(stored_ranks) != set(PARAMETERS):
        raise ValueError(f"{label}: rank inventory is invalid")
    ranks: dict[str, float] = {}
    for name in RANK_PARAMETERS:
        stored = _normalized_rank(stored_ranks[name], field=f"{label}.ranks.{name}")
        recomputed = _normalized_rank(
            posterior_rank(
                arrays[name], float(truth[name]), arrays["log_weights"]
            ),
            field=f"{label}.recomputed.{name}",
        )
        if not math.isclose(
            stored,
            recomputed,
            rel_tol=0.0,
            abs_tol=RANK_RECOMPUTE_TOLERANCE,
        ):
            raise ValueError(f"{label}: weighted rank mismatch for {name}")
        ranks[name] = recomputed
    return {
        "arrays": arrays,
        "weights": weights,
        "ranks": ranks,
        "artifacts": {
            "summary": summary_path.relative_to(campaign).as_posix(),
            "summary_sha256": file_sha256(summary_path),
            "posterior": posterior_path.relative_to(campaign).as_posix(),
            "posterior_sha256": file_sha256(posterior_path),
            "posterior_samples": sample_count,
            "posterior_effective_sample_size": float(1.0 / np.sum(weights**2)),
        },
    }


@cache
def _compiled_anchor_kernel(f_ref: float):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ripplegw.constants import MTSUN
    from ripplegw.conversions import Mc_eta_to_ms
    from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
        _get_merger_frequency,
    )
    from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_utils import get_coeffs
    from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomPv2_utils import (
        convert_spins,
        phP_get_transition_frequencies,
    )

    from benchmarks.device_parallel_nss.paper_model import _phenomd_peak_time_shift
    from benchmarks.device_parallel_nss.paper_model_basis import (
        REQUIRED_SIXTH_EXPONENTS,
        FrequencyPowerBasis,
        phase_of_basis,
        phase_with_qm_correction_basis,
    )

    def one(
        chirp_mass,
        q,
        s1_mag,
        s1_theta,
        s1_phi,
        s2_mag,
        s2_theta,
        s2_phi,
        iota,
        lambda_1,
        lambda_2,
    ):
        eta = q / (1.0 + q) ** 2
        heavy_mass, light_mass = Mc_eta_to_ms(jnp.array([chirp_mass, eta]))
        heavy_spin = (
            s1_mag * jnp.sin(s1_theta) * jnp.cos(s1_phi),
            s1_mag * jnp.sin(s1_theta) * jnp.sin(s1_phi),
            s1_mag * jnp.cos(s1_theta),
        )
        light_spin = (
            s2_mag * jnp.sin(s2_theta) * jnp.cos(s2_phi),
            s2_mag * jnp.sin(s2_theta) * jnp.sin(s2_phi),
            s2_mag * jnp.cos(s2_theta),
        )
        chi_light_l, chi_heavy_l, chi_p, *_ = convert_spins(
            light_mass,
            heavy_mass,
            f_ref,
            0.0,
            iota,
            *light_spin,
            *heavy_spin,
        )
        bbh_intrinsic = jnp.array(
            [heavy_mass, light_mass, chi_heavy_l, chi_light_l]
        )
        tidal_intrinsic = jnp.array(
            [
                heavy_mass,
                light_mass,
                chi_heavy_l,
                chi_light_l,
                lambda_1,
                lambda_2,
            ]
        )
        coefficients = get_coeffs(bbh_intrinsic)
        transitions = phP_get_transition_frequencies(
            bbh_intrinsic,
            coefficients[5],
            coefficients[6],
            chi_p,
        )
        mass_seconds = (heavy_mass + light_mass) * MTSUN

        def carrier_phase(frequency):
            basis = FrequencyPowerBasis.build(
                frequency, REQUIRED_SIXTH_EXPONENTS
            )
            bbh_phase = phase_with_qm_correction_basis(
                basis,
                mass_seconds,
                bbh_intrinsic,
                tidal_intrinsic,
                coefficients,
                transitions,
            )
            return phase_of_basis(
                basis,
                mass_seconds,
                tidal_intrinsic,
                bbh_phase,
            )

        merger_frequency = _get_merger_frequency(tidal_intrinsic)
        local_shift = jax.grad(carrier_phase)(merger_frequency) / (2.0 * jnp.pi)
        author_shift = _phenomd_peak_time_shift(
            mass_seconds, bbh_intrinsic, coefficients
        )
        return local_shift, author_shift

    return jax.jit(jax.vmap(one))


def _anchor_shifts(
    arrays: Mapping[str, np.ndarray],
    *,
    f_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    import jax

    jax.config.update("jax_enable_x64", True)
    names = (
        "M_c",
        "q",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "iota",
        "lambda_1",
        "lambda_2",
    )
    sample_count = arrays["q"].size
    kernel = _compiled_anchor_kernel(float(f_ref))
    local_chunks: list[np.ndarray] = []
    author_chunks: list[np.ndarray] = []
    for start in range(0, sample_count, ANCHOR_BATCH_SIZE):
        stop = min(sample_count, start + ANCHOR_BATCH_SIZE)
        size = stop - start
        arguments: list[np.ndarray] = []
        for name in names:
            values = np.asarray(arrays[name][start:stop], dtype=np.float64)
            if size < ANCHOR_BATCH_SIZE:
                values = np.pad(values, (0, ANCHOR_BATCH_SIZE - size), mode="edge")
            arguments.append(values)
        local, author = kernel(*arguments)
        local_chunks.append(np.asarray(jax.device_get(local))[:size])
        author_chunks.append(np.asarray(jax.device_get(author))[:size])
    local_values = np.concatenate(local_chunks)
    author_values = np.concatenate(author_chunks)
    if (
        local_values.shape != (sample_count,)
        or author_values.shape != (sample_count,)
        or not np.all(np.isfinite(local_values))
        or not np.all(np.isfinite(author_values))
    ):
        raise ValueError("computed carrier-anchor shifts are invalid")
    return local_values, author_values


def _detector_delays(
    arrays: Mapping[str, np.ndarray],
    *,
    trigger_time: float,
) -> np.ndarray:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
    from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time

    gmst = greenwich_mean_sidereal_time(float(trigger_time))
    ra = jnp.asarray(arrays["ra"])
    dec = jnp.asarray(arrays["dec"])
    values = [
        np.asarray(detector.delay_from_geocenter(ra, dec, gmst), dtype=np.float64)
        for detector in (get_H1(), get_L1(), get_V1())
    ]
    delays = np.column_stack(values)
    if delays.shape != (ra.size, 3) or not np.all(np.isfinite(delays)):
        raise ValueError("detector-delay matrix is invalid")
    return delays


def _geometry_for_result(
    result: Mapping[str, Any],
    *,
    f_ref: float,
    trigger_time: float,
    applied_anchor: str,
    label: str,
) -> dict[str, Any]:
    arrays = _mapping(result.get("arrays"), field=f"{label}.arrays")
    weights = np.asarray(result["weights"], dtype=np.float64)
    local, author = _anchor_shifts(arrays, f_ref=f_ref)
    delta = local - author
    t_c = np.asarray(arrays["t_c"], dtype=np.float64)
    delays = _detector_delays(arrays, trigger_time=trigger_time)
    delta_moments = _weighted_moments(
        t_c, delta, weights, label=f"{label} t_c/delta_anchor"
    )
    local_moments = _weighted_moments(
        t_c, local, weights, label=f"{label} t_c/local_anchor"
    )
    author_moments = _weighted_moments(
        t_c, author, weights, label=f"{label} t_c/author_anchor"
    )
    if applied_anchor == SOURCE_TIME_ANCHOR:
        applied = local
    elif applied_anchor == DIAGNOSTIC_TIME_ANCHOR:
        applied = author
    else:
        raise ValueError(f"unknown applied anchor {applied_anchor!r}")
    partial = _partial_geometry(
        t_c,
        applied,
        delays,
        weights,
        label=f"{label} applied anchor",
    )
    return {
        "applied_anchor": applied_anchor,
        "corr_t_c_delta_anchor": delta_moments["correlation"],
        "corr_t_c_local_anchor": local_moments["correlation"],
        "corr_t_c_imrphenomd_anchor": author_moments["correlation"],
        "local_anchor_weighted_sd_seconds": local_moments["sd_y"],
        "imrphenomd_anchor_weighted_sd_seconds": author_moments["sd_y"],
        "imrphenomd_to_local_anchor_sd_ratio": (
            author_moments["sd_y"] / local_moments["sd_y"]
        ),
        "delta_anchor_weighted_sd_seconds": delta_moments["sd_y"],
        "partial_geometry_after_detector_delay_residualization": partial,
    }


def _rank_grid(cases: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for case in cases:
        for parameter in RANK_PARAMETERS:
            metrics = _rank_metrics(case["rank_comparisons"][parameter][side])
            cells.append(
                {
                    "diagnostic_id": case["diagnostic_id"],
                    "source_injection_id": case["source_injection_id"],
                    "parameter": parameter,
                    **metrics,
                }
            )
    tails = [float(cell["two_sided_tail_probability"]) for cell in cells]
    return {
        "cell_count": len(cells),
        "extreme_tail_count": sum(bool(cell["extreme_tail"]) for cell in cells),
        "central_90_count": sum(bool(cell["central_90"]) for cell in cells),
        "exact_endpoint_count": sum(bool(cell["exact_endpoint"]) for cell in cells),
        "median_two_sided_tail_probability": float(np.median(tails)),
        "cells": cells,
    }


def _validate_optional_control(
    control_campaign: Path | None,
    *,
    source_campaign: Path,
    source_manifest: Mapping[str, Any],
    source_catalogue: Sequence[Mapping[str, Any]],
    author_manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    if control_campaign is None:
        return None
    campaign = control_campaign.expanduser().resolve()
    manifest = load_manifest(campaign)
    config = _mapping(manifest.get("config"), field="local control config")
    source_config = _mapping(source_manifest.get("config"), field="source config")
    author_config = _mapping(author_manifest.get("config"), field="author config")
    if (
        config.get("carrier_time_anchor") != SOURCE_TIME_ANCHOR
        or config.get("campaign") != "local-anchor-control-d4-m1-pathology-diagnostic"
        or config.get("paper_configuration")
        != "Sharded local-anchor-control diagnostic"
        or _scientific_config(config) != _scientific_config(source_config)
        or _scientific_config(config) != _scientific_config(author_config)
        or manifest.get("n_injections") != EXPECTED_CASES
        or manifest.get("catalogue_size") != EXPECTED_CASES
    ):
        raise ValueError("optional local control has an invalid configuration")
    selection = _mapping(manifest.get("selection"), field="local control selection")
    if (
        selection.get("source_injection_ids") != list(DEFAULT_SOURCE_IDS)
        or selection.get("start_inclusive") != 0
        or selection.get("stop_exclusive") != EXPECTED_CASES
    ):
        raise ValueError("optional local control has the wrong sentinel selection")
    if (
        manifest.get("psd") != source_manifest.get("psd")
        or manifest.get("storage_policy") != source_manifest.get("storage_policy")
    ):
        raise ValueError("optional local control inputs differ from source")
    scope = _mapping(
        manifest.get("reproduction_scope"),
        field="local control reproduction_scope",
    )
    if (
        scope.get("iid_prior_predictive_catalogue") is not False
        or scope.get("pp_calibration_eligible") is not False
    ):
        raise ValueError("optional local control is not marked as targeted")

    control_pin = _mapping(
        manifest.get("implementation_diagnostic"),
        field="local control implementation_diagnostic",
    )
    author_pin = _mapping(
        author_manifest.get("implementation_diagnostic"),
        field="author implementation_diagnostic",
    )
    for field in (
        "implementation_label",
        "implementation_revision",
        "implementation_tree_sha256",
        "sampler_scheduler",
    ):
        if control_pin.get(field) != author_pin.get(field):
            raise ValueError(
                f"control and author implementation pins differ in {field}"
            )
    if control_pin.get("source_campaign_config_sha256") != source_manifest.get(
        "config_sha256"
    ):
        raise ValueError("optional local control references the wrong source config")
    changed = _mapping(
        control_pin.get("changed_variables"),
        field="local control changed_variables",
    )
    if dict(changed) != {
        "carrier_time_anchor": {
            "source": SOURCE_TIME_ANCHOR,
            "diagnostic": SOURCE_TIME_ANCHOR,
        }
    }:
        raise ValueError("optional local control records the wrong anchor semantics")

    catalogue_metadata = _mapping(manifest.get("catalogue"), field="control catalogue")
    provenance = _mapping(
        catalogue_metadata.get("provenance"), field="control catalogue.provenance"
    )
    source_catalogue_metadata = _mapping(
        source_manifest.get("catalogue"), field="source catalogue"
    )
    expected_provenance = {
        "source_campaign": source_campaign.name,
        "source_manifest_sha256": file_sha256(source_campaign / "manifest.json"),
        "source_config_sha256": source_manifest.get("config_sha256"),
        "source_catalogue_sha256": source_catalogue_metadata.get("sha256"),
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(f"optional local control provenance {field} mismatch")
    catalogue = read_catalogue(campaign / str(catalogue_metadata["path"]))
    mapping = _validate_mapping(
        manifest,
        source_campaign=source_campaign,
        diagnostic_catalogue=catalogue,
        source_catalogue=source_catalogue,
    )
    return {
        "path": campaign,
        "manifest": manifest,
        "catalogue": catalogue,
        "mapping": mapping,
        "metadata": {
            "campaign": config["campaign"],
            "manifest_sha256": file_sha256(campaign / "manifest.json"),
            "config_sha256": manifest["config_sha256"],
            "implementation_revision": control_pin["implementation_revision"],
            "implementation_tree_sha256": control_pin[
                "implementation_tree_sha256"
            ],
            "validated": True,
            "used_for_preregistered_gates": False,
        },
    }


def evaluate_time_anchor_diagnostic(
    campaign_dir: Path,
    *,
    source_campaign: Path | None = None,
    local_control_campaign: Path | None = None,
) -> dict[str, Any]:
    """Validate artifacts, recompute ranks, and apply the frozen A/B gates."""

    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    catalogue_metadata = _mapping(manifest.get("catalogue"), field="manifest.catalogue")
    provenance = _mapping(
        catalogue_metadata.get("provenance"), field="manifest.catalogue.provenance"
    )
    source_campaign = _source_campaign_path(
        campaign_dir, provenance, source_campaign
    )
    source_manifest = load_manifest(source_campaign)
    _validate_manifest_contract(
        manifest,
        source_manifest,
        campaign_dir=campaign_dir,
        source_campaign=source_campaign,
    )
    diagnostic_catalogue = read_catalogue(
        campaign_dir / str(catalogue_metadata["path"])
    )
    source_catalogue_metadata = _mapping(
        source_manifest.get("catalogue"), field="source manifest.catalogue"
    )
    source_catalogue = read_catalogue(
        source_campaign / str(source_catalogue_metadata["path"])
    )
    mapping = _validate_mapping(
        manifest,
        source_campaign=source_campaign,
        diagnostic_catalogue=diagnostic_catalogue,
        source_catalogue=source_catalogue,
    )
    control_context = _validate_optional_control(
        local_control_campaign,
        source_campaign=source_campaign,
        source_manifest=source_manifest,
        source_catalogue=source_catalogue,
        author_manifest=manifest,
    )

    source_config = _mapping(source_manifest.get("config"), field="source config")
    diagnostic_config = _mapping(manifest.get("config"), field="diagnostic config")
    f_ref = float(diagnostic_config["waveform_f_ref_hz"])
    trigger_time = float(diagnostic_config["trigger_time_gps"])
    control_config = (
        _mapping(
            _mapping(control_context["manifest"], field="control manifest").get(
                "config"
            ),
            field="control config",
        )
        if control_context is not None
        else None
    )
    cases: list[dict[str, Any]] = []
    for entry in mapping:
        diagnostic_id = entry["diagnostic_id"]
        source_id = entry["source_injection_id"]
        source_result = _load_result(
            source_campaign,
            source_manifest,
            source_catalogue[source_id],
            source_id,
            diagnostic=False,
        )
        author_result = _load_result(
            campaign_dir,
            manifest,
            diagnostic_catalogue[diagnostic_id],
            diagnostic_id,
            diagnostic=True,
        )
        control_result: dict[str, Any] | None = None
        control_entry: Mapping[str, Any] | None = None
        if control_context is not None:
            control_manifest = _mapping(
                control_context["manifest"], field="control manifest"
            )
            control_catalogue = control_context["catalogue"]
            control_mapping = control_context["mapping"]
            control_entry = _mapping(
                control_mapping[diagnostic_id],
                field=f"control mapping {diagnostic_id}",
            )
            if control_entry.get("source_injection_id") != source_id:
                raise ValueError("control and author sentinel mappings differ")
            control_result = _load_result(
                control_context["path"],
                control_manifest,
                control_catalogue[diagnostic_id],
                diagnostic_id,
                diagnostic=True,
            )
        for parameter in RANK_PARAMETERS:
            if not math.isclose(
                source_result["ranks"][parameter],
                entry["recorded_source_ranks"][parameter],
                rel_tol=0.0,
                abs_tol=RANK_RECOMPUTE_TOLERANCE,
            ):
                raise ValueError(
                    f"mapping source rank mismatch for {source_id}.{parameter}"
                )
            if (
                control_result is not None
                and control_entry is not None
                and not math.isclose(
                    source_result["ranks"][parameter],
                    control_entry["recorded_source_ranks"][parameter],
                    rel_tol=0.0,
                    abs_tol=RANK_RECOMPUTE_TOLERANCE,
                )
            ):
                raise ValueError(
                    f"control mapping source rank mismatch for "
                    f"{source_id}.{parameter}"
                )
        source_geometry = _geometry_for_result(
            source_result,
            f_ref=float(source_config["waveform_f_ref_hz"]),
            trigger_time=float(source_config["trigger_time_gps"]),
            applied_anchor=SOURCE_TIME_ANCHOR,
            label=f"source {source_id}",
        )
        author_geometry = _geometry_for_result(
            author_result,
            f_ref=f_ref,
            trigger_time=trigger_time,
            applied_anchor=DIAGNOSTIC_TIME_ANCHOR,
            label=f"author {diagnostic_id}",
        )
        control_geometry: dict[str, Any] | None = None
        if control_result is not None:
            if control_config is None:
                raise RuntimeError("control result exists without control configuration")
            control_geometry = _geometry_for_result(
                control_result,
                f_ref=float(control_config["waveform_f_ref_hz"]),
                trigger_time=float(control_config["trigger_time_gps"]),
                applied_anchor=SOURCE_TIME_ANCHOR,
                label=f"control {diagnostic_id}",
            )
        source_leverage = source_geometry[
            "partial_geometry_after_detector_delay_residualization"
        ]["leverage_abs_cov_anchor_tc_over_var_tc"]
        author_leverage = author_geometry[
            "partial_geometry_after_detector_delay_residualization"
        ]["leverage_abs_cov_anchor_tc_over_var_tc"]
        control_leverage = (
            control_geometry[
                "partial_geometry_after_detector_delay_residualization"
            ]["leverage_abs_cov_anchor_tc_over_var_tc"]
            if control_geometry is not None
            else None
        )
        if control_leverage is not None and control_leverage <= 0.0:
            raise ValueError(
                f"control {diagnostic_id} applied-anchor leverage is not positive"
            )
        frozen = FROZEN_LOCAL_LEVERAGE[source_id]
        if not math.isclose(
            source_leverage,
            frozen,
            rel_tol=0.0,
            abs_tol=FROZEN_LOCAL_LEVERAGE_TOLERANCE,
        ):
            raise ValueError(
                f"source {source_id} leverage {source_leverage:.8g} disagrees "
                f"with frozen reference {frozen:.8g}"
            )
        sd_ratio = author_geometry["imrphenomd_to_local_anchor_sd_ratio"]
        rank_comparisons = {
            parameter: {
                "source": source_result["ranks"][parameter],
                **(
                    {"control": control_result["ranks"][parameter]}
                    if control_result is not None
                    else {}
                ),
                "author": author_result["ranks"][parameter],
            }
            for parameter in RANK_PARAMETERS
        }
        cases.append(
            {
                **{key: value for key, value in entry.items() if key != "recorded_source_ranks"},
                "rank_comparisons": rank_comparisons,
                "source_geometry": source_geometry,
                "control_geometry": control_geometry,
                "author_geometry": author_geometry,
                "current_code_control_vs_author": (
                    {
                        "control_applied_anchor_leverage": control_leverage,
                        "author_applied_anchor_leverage": author_leverage,
                        "leverage_reduction_fraction": (
                            1.0 - author_leverage / control_leverage
                        ),
                        "control_corr_t_c_delta_anchor": control_geometry[
                            "corr_t_c_delta_anchor"
                        ],
                        "author_corr_t_c_delta_anchor": author_geometry[
                            "corr_t_c_delta_anchor"
                        ],
                        "same_candidate_implementation": True,
                        "used_for_preregistered_gates": False,
                    }
                    if control_geometry is not None and control_leverage is not None
                    else None
                ),
                "paired_geometry": {
                    "source_applied_anchor_leverage": source_leverage,
                    "author_applied_anchor_leverage": author_leverage,
                    "leverage_reduction_fraction": 1.0 - author_leverage / source_leverage,
                    "author_to_local_anchor_sd_ratio_on_author_posterior": sd_ratio,
                    "author_leverage_gate_passed": author_leverage <= MAX_AUTHOR_LEVERAGE,
                    "anchor_sd_ratio_gate_passed": sd_ratio <= MAX_ANCHOR_SD_RATIO,
                    "frozen_source_leverage": frozen,
                    "source_minus_frozen_leverage": source_leverage - frozen,
                },
                "source_inputs": source_result["artifacts"],
                "control_inputs": (
                    control_result["artifacts"]
                    if control_result is not None
                    else None
                ),
                "author_inputs": author_result["artifacts"],
            }
        )

    source_grid = _rank_grid(cases, "source")
    control_grid = (
        _rank_grid(cases, "control") if control_context is not None else None
    )
    author_grid = _rank_grid(cases, "author")
    leverage_pass = all(
        case["paired_geometry"]["author_leverage_gate_passed"] for case in cases
    )
    manipulation_pass = all(
        case["paired_geometry"]["anchor_sd_ratio_gate_passed"] for case in cases
    )
    rank_pass = (
        author_grid["extreme_tail_count"] <= MAX_EXTREME_CELLS
        and author_grid["central_90_count"] >= MIN_CENTRAL_CELLS
        and author_grid["exact_endpoint_count"] == MAX_ENDPOINT_CELLS
    )
    weak_reduction_count = sum(
        case["paired_geometry"]["leverage_reduction_fraction"] < 0.25
        for case in cases
    )
    if manipulation_pass and leverage_pass and rank_pass:
        decision = "major-cause-confirmed"
    elif (
        manipulation_pass
        and author_grid["extreme_tail_count"] >= 6
        and weak_reduction_count >= 3
    ):
        decision = "major-cause-refuted"
    else:
        decision = "mixed-inconclusive"

    optional_control_report: dict[str, Any] | None = None
    if control_context is not None:
        if control_grid is None:
            raise RuntimeError("control context exists without a control rank grid")
        control_leverages = [
            case["current_code_control_vs_author"][
                "control_applied_anchor_leverage"
            ]
            for case in cases
        ]
        current_reductions = [
            case["current_code_control_vs_author"][
                "leverage_reduction_fraction"
            ]
            for case in cases
        ]
        optional_control_report = {
            **control_context["metadata"],
            "rank_grid": control_grid,
            "median_control_applied_anchor_leverage": float(
                np.median(control_leverages)
            ),
            "median_control_to_author_leverage_reduction_fraction": float(
                np.median(current_reductions)
            ),
            "comparison": (
                "Current-code local and IMRPhenomD runs share an exact candidate "
                "implementation pin; this descriptive comparison is not used by "
                "the preregistered gates."
            ),
        }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "decision": decision,
        "campaign": diagnostic_config["campaign"],
        "config_sha256": manifest["config_sha256"],
        "selection": {
            "diagnostic_ids": list(range(EXPECTED_CASES)),
            "source_injection_ids": list(DEFAULT_SOURCE_IDS),
            "rank_parameters": list(RANK_PARAMETERS),
            "rank_cell_count": EXPECTED_CASES * len(RANK_PARAMETERS),
        },
        "methodology": {
            "rank": "sum original normalized nested-sampling weight for samples < truth",
            "rank_roundoff_normalization": "only values within 1e-12 of [0,1] are clamped",
            "resampling": False,
            "extreme_tail": "2 * min(rank, 1-rank) < 1e-3",
            "central_90": "0.05 <= rank <= 0.95",
            "delta_anchor": "nrtidal-merger shift minus imrphenomd shift",
            "corr_t_c_delta_anchor": "weighted Pearson correlation; descriptive, not gated",
            "applied_anchor_geometry": (
                "weighted residuals after WLS on [1, delay_H1, delay_L1, delay_V1]"
            ),
            "leverage": "abs(Cov_w(residual_t_c,residual_anchor)/Var_w(residual_t_c))",
            "partial_r_squared": "squared weighted residual correlation",
            "population_calibration_claim_permitted": False,
        },
        "rank_grids": {
            "source": source_grid,
            "control": control_grid,
            "author": author_grid,
        },
        "preregistered_gates": {
            "manipulation": {
                "criterion": "SD(author anchor)/SD(local anchor) <= 0.1 on every author posterior",
                "threshold": MAX_ANCHOR_SD_RATIO,
                "passed": manipulation_pass,
            },
            "applied_anchor_leverage": {
                "criterion": "sky-delay-residualized author leverage <= 0.25 in every event",
                "threshold": MAX_AUTHOR_LEVERAGE,
                "passed": leverage_pass,
            },
            "rank_geometry": {
                "criteria": {
                    "extreme_tail_count_max": MAX_EXTREME_CELLS,
                    "central_90_count_min": MIN_CENTRAL_CELLS,
                    "exact_endpoint_count": MAX_ENDPOINT_CELLS,
                },
                "passed": rank_pass,
            },
            "all_decisive_gates_passed": manipulation_pass and leverage_pass and rank_pass,
            "refutation_rule": {
                "criterion": "author extreme cells >= 6 and leverage reduction <25% in >=3/4",
                "weak_leverage_reduction_event_count": weak_reduction_count,
                "triggered": (
                    manipulation_pass
                    and author_grid["extreme_tail_count"] >= 6
                    and weak_reduction_count >= 3
                ),
            },
        },
        "descriptive_aggregates": {
            "median_source_applied_anchor_leverage": float(
                np.median(
                    [
                        case["paired_geometry"]["source_applied_anchor_leverage"]
                        for case in cases
                    ]
                )
            ),
            "median_author_applied_anchor_leverage": float(
                np.median(
                    [
                        case["paired_geometry"]["author_applied_anchor_leverage"]
                        for case in cases
                    ]
                )
            ),
            "median_leverage_reduction_fraction": float(
                np.median(
                    [
                        case["paired_geometry"]["leverage_reduction_fraction"]
                        for case in cases
                    ]
                )
            ),
            "median_author_to_local_anchor_sd_ratio": float(
                np.median(
                    [
                        case["paired_geometry"][
                            "author_to_local_anchor_sd_ratio_on_author_posterior"
                        ]
                        for case in cases
                    ]
                )
            ),
        },
        "optional_local_control": optional_control_report,
        "provenance": {
            "diagnostic_manifest": "manifest.json",
            "diagnostic_manifest_sha256": file_sha256(campaign_dir / "manifest.json"),
            "source_campaign": provenance["source_campaign"],
            "source_manifest_sha256": provenance["source_manifest_sha256"],
            "source_config_sha256": provenance["source_config_sha256"],
            "source_catalogue_sha256": provenance["source_catalogue_sha256"],
            "ordered_mapping": mapping,
        },
        "cases": cases,
    }


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report["cases"]:
        ranks = case["rank_comparisons"]
        paired = case["paired_geometry"]
        control_geometry = case["control_geometry"]
        control_comparison = case["current_code_control_vs_author"]
        rows.append(
            {
                "diagnostic_id": case["diagnostic_id"],
                "source_injection_id": case["source_injection_id"],
                "source_q_rank": ranks["q"]["source"],
                "control_q_rank": ranks["q"].get("control", ""),
                "author_q_rank": ranks["q"]["author"],
                "source_ra_rank": ranks["ra"]["source"],
                "control_ra_rank": ranks["ra"].get("control", ""),
                "author_ra_rank": ranks["ra"]["author"],
                "source_dec_rank": ranks["dec"]["source"],
                "control_dec_rank": ranks["dec"].get("control", ""),
                "author_dec_rank": ranks["dec"]["author"],
                "source_t_c_rank": ranks["t_c"]["source"],
                "control_t_c_rank": ranks["t_c"].get("control", ""),
                "author_t_c_rank": ranks["t_c"]["author"],
                "source_corr_t_c_delta_anchor": case["source_geometry"][
                    "corr_t_c_delta_anchor"
                ],
                "control_corr_t_c_delta_anchor": (
                    control_geometry["corr_t_c_delta_anchor"]
                    if control_geometry is not None
                    else ""
                ),
                "author_corr_t_c_delta_anchor": case["author_geometry"][
                    "corr_t_c_delta_anchor"
                ],
                "source_applied_anchor_leverage": paired[
                    "source_applied_anchor_leverage"
                ],
                "control_applied_anchor_leverage": (
                    control_comparison["control_applied_anchor_leverage"]
                    if control_comparison is not None
                    else ""
                ),
                "author_applied_anchor_leverage": paired[
                    "author_applied_anchor_leverage"
                ],
                "current_code_leverage_reduction_fraction": (
                    control_comparison["leverage_reduction_fraction"]
                    if control_comparison is not None
                    else ""
                ),
                "leverage_reduction_fraction": paired["leverage_reduction_fraction"],
                "author_to_local_anchor_sd_ratio_on_author_posterior": paired[
                    "author_to_local_anchor_sd_ratio_on_author_posterior"
                ],
                "author_leverage_gate_passed": paired[
                    "author_leverage_gate_passed"
                ],
                "anchor_sd_ratio_gate_passed": paired[
                    "anchor_sd_ratio_gate_passed"
                ],
            }
        )
    return rows


def write_time_anchor_diagnostic_report(
    campaign_dir: Path,
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    directory = campaign_dir.expanduser().resolve() / REPORT_DIRECTORY
    json_path = directory / REPORT_JSON
    csv_path = directory / REPORT_CSV
    atomic_write_json(json_path, report)
    atomic_write_csv(csv_path, _csv_rows(report), CSV_FIELDS)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = evaluate_time_anchor_diagnostic(
        args.campaign_dir,
        source_campaign=args.source_campaign,
        local_control_campaign=args.local_control_campaign,
    )
    json_path, csv_path = write_time_anchor_diagnostic_report(
        args.campaign_dir, report
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
