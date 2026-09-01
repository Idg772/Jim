import hashlib
import json
import logging
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from ripplegw.interfaces import Waveform

from jimgw.cli._config import (
    CLIInjectionRefParams,
    CLIOptimizerRefParams,
    CLIProvidedRefParams,
    DataConfig,
    InjectionDataConfig,
    LikelihoodConfig,
    PipelineConfig,
)
from jimgw.cli._prior import build_prior
from jimgw.cli._transforms import to_likelihood_space
from jimgw.core.constants import EARTH_RADIUS_LIGHT_S
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.detector import GroundBased2G
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
    TransientLikelihoodFD,
)
from jimgw.core.single_event.marginalization_config import (
    DistanceMargConfig,
    HeterodyneTimeMargConfig,
    PhaseMargConfig,
    TimeMargConfig,
)
from jimgw.core.transforms import NtoMTransform

logger = logging.getLogger(__name__)


def detector_metadata_sha256(ifos: list[GroundBased2G]) -> str:
    """Hash detector site, arm, and response metadata in network order."""

    payload = [
        {
            "name": ifo.name,
            "latitude": ifo.latitude,
            "longitude": ifo.longitude,
            "elevation": ifo.elevation,
            "xarm_azimuth": ifo.xarm_azimuth,
            "yarm_azimuth": ifo.yarm_azimuth,
            "xarm_tilt": ifo.xarm_tilt,
            "yarm_tilt": ifo.yarm_tilt,
            "arm_length_m": ifo.arm_length_m,
            "orbital_motion_response": ifo.orbital_motion_response,
            "orbital_reference_time": ifo.orbital_reference_time,
            "orbital_validity_s": ifo.orbital_validity_s,
            "orbital_acceleration_over_c": ifo.orbital_acceleration_over_c,
            "orbital_jerk_over_c": ifo.orbital_jerk_over_c,
        }
        for ifo in ifos
    ]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_xg_detector_inputs(
    ifos: list[GroundBased2G],
    input_files_sha256: dict[str, str],
) -> None:
    """Check realized detector provenance and precision without copying arrays."""

    expected_provenance = tuple(sorted(input_files_sha256.items()))
    for ifo in ifos:
        if ifo.input_provenance_sha256 != expected_provenance:
            raise ValueError(
                f"{ifo.name} data were not built from the qualified XG inputs"
            )
        realized_arrays = {
            "strain frequencies": (ifo.data.frequencies, np.dtype(np.float64)),
            "strain": (ifo.data.fd, np.dtype(np.complex128)),
            "PSD frequencies": (ifo.psd.frequencies, np.dtype(np.float64)),
            "PSD": (ifo.psd.values, np.dtype(np.float64)),
        }
        for label, (array, expected_dtype) in realized_arrays.items():
            if np.dtype(array.dtype) != expected_dtype:
                raise TypeError(
                    f"{ifo.name} {label} uses {array.dtype}, but the qualified "
                    f"XG path requires {expected_dtype}"
                )


def build_likelihood(
    pipeline_cfg: PipelineConfig,
    ifos: list[GroundBased2G],
    waveform: Waveform,
    prior: CombinePrior,
    likelihood_transforms: list[NtoMTransform],
) -> Union[
    TransientLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
]:
    """Build a likelihood from one validated pipeline contract.

    Uses ``HeterodynedTransientLikelihoodFD`` when the heterodyne config is set,
    otherwise falls back to ``TransientLikelihoodFD``.  ``prior`` and
    ``likelihood_transforms`` are required for the heterodyne case (the optimizer
    needs them to find reference parameters).

    The data config is only used when ``reference_parameters.type =
    "injection"`` — it must be an ``InjectionDataConfig`` in that case.
    """
    cfg: LikelihoodConfig = pipeline_cfg.likelihood
    trigger_time = pipeline_cfg.data.trigger_time
    waveform_f_ref = pipeline_cfg.waveform.f_ref
    time_frame = pipeline_cfg.sampling.time_frame
    data_cfg: DataConfig = pipeline_cfg.data
    verified_xg_manifest = pipeline_cfg.verified_xg_manifest
    uses_xg_response = (
        cfg.time_dependent_response
        or cfg.finite_arm_response
        or cfg.orbital_motion_response
    )
    uses_xg_compression = uses_xg_response and cfg.heterodyne is not None

    if uses_xg_response and not jax.config.jax_enable_x64:
        raise RuntimeError("XG likelihood construction requires JAX 64-bit precision")
    if uses_xg_compression:
        if verified_xg_manifest is None:
            raise ValueError(
                "XG likelihood construction requires a qualification receipt "
                "verified against the complete pipeline config"
            )
        if (
            pipeline_cfg.xg_analysis_contract_sha256()
            != verified_xg_manifest.analysis_contract_sha256
        ):
            raise ValueError(
                "the pipeline config changed after XG qualification verification"
            )
        if (
            pipeline_cfg.xg_input_files_sha256()
            != verified_xg_manifest.input_files_sha256
        ):
            raise ValueError(
                "the XG input files changed after qualification verification"
            )
        source_waveform = (
            waveform.source
            if isinstance(waveform, DominantModeTimeCachedWaveform)
            else waveform
        )
        expected_class_name = pipeline_cfg.waveform.approximant
        if type(source_waveform).__name__ != expected_class_name:
            raise TypeError(
                "the realized waveform does not match the qualified approximant"
            )
        realized_f_ref = getattr(waveform, "f_ref", None)
        if realized_f_ref is None or float(realized_f_ref) != waveform_f_ref:
            raise ValueError(
                "the realized waveform reference frequency does not match the "
                "qualified pipeline config"
            )
    elif verified_xg_manifest is not None:
        raise ValueError(
            "the pipeline response or compression mode changed after XG "
            "qualification verification"
        )

    if cfg.time_dependent_response and not isinstance(
        waveform,
        DominantModeTimeCachedWaveform,
    ):
        raise TypeError(
            "time_dependent_response requires DominantModeTimeCachedWaveform"
        )
    for ifo in ifos:
        if cfg.finite_arm_response and ifo.arm_length_m is None:
            raise ValueError(
                f"finite_arm_response requires arm_length_m metadata for {ifo.name}"
            )
        ifo.time_dependent_response = cfg.time_dependent_response
        ifo.finite_arm_response = cfg.finite_arm_response
        ifo.configure_orbital_motion_response(
            enabled=cfg.orbital_motion_response,
            reference_time=cfg.orbital_reference_time,
            validity_s=cfg.orbital_validity_s,
            acceleration_over_c=cfg.orbital_acceleration_over_c,
            jerk_over_c=cfg.orbital_jerk_over_c,
        )

    phase_marg = None
    if cfg.phase_marginalization:
        phase_marg = PhaseMargConfig()

    fixed_params = cfg.fixed_parameters if cfg.fixed_parameters else None

    if cfg.heterodyne is not None:
        if verified_xg_manifest is not None:
            metadata_digest = detector_metadata_sha256(ifos)
            if metadata_digest != verified_xg_manifest.detector_metadata_sha256:
                raise ValueError(
                    "XG qualification detector metadata does not match the "
                    "constructed network"
                )
            _validate_xg_detector_inputs(
                ifos,
                verified_xg_manifest.input_files_sha256,
            )
        heterodyne_time_marg = None
        if cfg.time_marginalization is not None:
            heterodyne_time_marg = HeterodyneTimeMargConfig(
                tc_range=cfg.time_marginalization.tc_range,
                upsample_factor=cfg.time_marginalization.upsample_factor,
                phasor_block_size=cfg.time_marginalization.phasor_block_size,
                freeze_response=cfg.time_marginalization.freeze_response,
                timing_sigma_s=cfg.time_marginalization.timing_sigma_s,
                samples_per_timing_sigma=(
                    cfg.time_marginalization.samples_per_timing_sigma
                ),
                normalization=cfg.time_marginalization.normalization,
            )
        ref_cfg = cfg.heterodyne.reference_parameters
        reference_params: Optional[dict] = None
        optimizer_popsize = 500
        optimizer_n_steps = 1000
        optimizer_target: Optional[float] = None

        if isinstance(ref_cfg, CLIOptimizerRefParams):
            optimizer_popsize = ref_cfg.popsize
            optimizer_n_steps = ref_cfg.n_steps
            optimizer_target = ref_cfg.target
            # Phase-marginalised heterodyned likelihood with the optimizer: the
            # optimizer needs phase_c in the prior to search over it, but the
            # user should not have to (and must not) include it themselves since
            # it is a marginalised parameter.  Add a default Uniform(0, 2π)
            # component here; the caller's `prior` (without phase_c) is still
            # passed to Jim.__init__ unchanged.
            if cfg.phase_marginalization and "phase_c" not in prior.parameter_names:
                prior = CombinePrior(
                    list(prior.base_prior)
                    + [UniformPrior(0.0, 2 * jnp.pi, ["phase_c"])]
                )
                logger.info(
                    "Added Uniform(0, 2π) prior on phase_c for optimizer reference parameter search"
                )
        elif isinstance(ref_cfg, CLIProvidedRefParams):
            reference_params = ref_cfg.values
        elif isinstance(ref_cfg, CLIInjectionRefParams):
            if not isinstance(data_cfg, InjectionDataConfig):
                raise TypeError(
                    "Heterodyne reference_parameters.type='injection' requires "
                    "data.type='injection'."
                )
            reference_params = to_likelihood_space(
                data_cfg.injection_parameters,
                waveform_f_ref=waveform_f_ref,
                trigger_time=trigger_time,
                ifos=ifos,
                time_frame=time_frame,
            )
            logger.info(
                "Using injection parameters as heterodyne reference: %s",
                reference_params,
            )

        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            fixed_parameters=fixed_params,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            trigger_time=trigger_time,
            n_bins=cfg.heterodyne.n_bins,
            epsilon=cfg.heterodyne.epsilon,
            optimizer_popsize=optimizer_popsize,
            optimizer_n_steps=optimizer_n_steps,
            optimizer_target=optimizer_target,
            prior=prior,
            likelihood_transforms=likelihood_transforms,
            phase_marginalization=phase_marg,
            time_marginalization=heterodyne_time_marg,
            reference_parameters=reference_params,
            reference_chunk_size=cfg.heterodyne.reference_chunk_size,
            xg_plan=(
                pipeline_cfg._issue_verified_xg_plan()
                if verified_xg_manifest is not None
                else None
            ),
        )
        logger.info(
            "Built heterodyne likelihood: f_min=%.1f, f_max=%.1f, n_bins=%d",
            cfg.f_min,
            cfg.f_max,
            likelihood.n_bins,
        )
        return likelihood

    if cfg.multiband is not None:
        mb = cfg.multiband

        # MultibandedTransientLikelihoodFD._infer_time_offsets only searches for
        # t_c. When the NS AW sampler is used it renames t_c → t_det in the built
        # prior (same relative-offset bounds). Detect that here and compute
        # time_offset / delta_f_end explicitly so auto-inference works correctly.
        mb_time_offset = mb.time_offset
        mb_delta_f_end = mb.delta_f_end
        if (mb_time_offset is None or mb_delta_f_end is None) and (
            "t_c" not in prior.parameter_names and "t_det" in prior.parameter_names
        ):
            tdet_comp = next(
                (p for p in prior.base_prior if "t_det" in p.parameter_names),
                None,
            )
            tdet_bounds = tdet_comp.get_bounds() if tdet_comp is not None else None
            if tdet_bounds is not None:
                xmin, xmax = tdet_bounds
                ref_ifo = (
                    ifos[0]
                    if time_frame == "detector"
                    else next(ifo for ifo in ifos if ifo.name == time_frame)
                )
                t_end = (
                    float(ref_ifo.data.start_time)
                    + float(ref_ifo.data.duration)
                    - trigger_time
                )
                s = EARTH_RADIUS_LIGHT_S
                if mb_time_offset is None:
                    mb_time_offset = t_end - xmin + s
                    logger.info(
                        "time_offset inferred from t_det prior: %.4f s", mb_time_offset
                    )
                if mb_delta_f_end is None:
                    denom = t_end - xmax - s
                    if denom <= 0:
                        raise ValueError(
                            f"Cannot infer delta_f_end from t_det prior: "
                            f"t_end - xmax - s = {t_end:.4f} - {xmax:.4f} - {s:.6f} = {denom:.6f} <= 0. "
                            "Check that the t_det prior upper bound is well within the data segment."
                        )
                    mb_delta_f_end = 100.0 / denom
                    logger.info(
                        "delta_f_end inferred from t_det prior: %.4f Hz", mb_delta_f_end
                    )

        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            fixed_parameters=fixed_params,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            trigger_time=trigger_time,
            highest_mode=mb.highest_mode,
            accuracy_factor=mb.accuracy_factor,
            prior=prior,
            reference_chirp_mass=mb.reference_chirp_mass,
            time_offset=mb_time_offset,
            delta_f_end=mb_delta_f_end,
            max_banding_frequency=mb.max_banding_frequency,
            min_banding_duration=mb.min_banding_duration,
        )
        logger.info(
            "Built multiband likelihood: f_min=%.1f, f_max=%.1f, "
            "reference_chirp_mass=%.4f M_sun",
            cfg.f_min,
            cfg.f_max,
            likelihood.reference_chirp_mass,
        )
        return likelihood

    time_marg = None
    if cfg.time_marginalization is not None:
        time_marg = TimeMargConfig(
            tc_range=cfg.time_marginalization.tc_range,
            upsample_factor=cfg.time_marginalization.upsample_factor,
        )

    dist_marg = None
    if cfg.distance_marginalization is not None:
        dist_combined = build_prior(cfg.distance_marginalization.distance_prior)
        dist_marg = DistanceMargConfig(
            distance_prior=dist_combined.base_prior[0],
            n_dist_points=cfg.distance_marginalization.n_dist_points,
            ref_dist=cfg.distance_marginalization.ref_dist,
        )

    likelihood = TransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        fixed_parameters=fixed_params,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        trigger_time=trigger_time,
        phase_marginalization=phase_marg,
        time_marginalization=time_marg,
        distance_marginalization=dist_marg,
    )
    logger.info(
        "Built likelihood: f_min=%.1f, f_max=%.1f, trigger_time=%.3f",
        cfg.f_min,
        cfg.f_max,
        trigger_time,
    )
    return likelihood
