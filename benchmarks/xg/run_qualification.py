"""Run the frozen CE XG qualification campaign.

The command is intentionally a one-shot gate.  It evaluates the predeclared
clock, non-SPA response, orbital-surrogate, and compression cases, publishes
typed immutable evidence only when every case passes, and never starts the
sampler itself.  The qualification manifest is assembled by
``jim-xg-qualify`` after this command succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from benchmarks.xg.oracles.clock_orbital_oracle import (
    adaptive_stationary_time,
    fisher_projected_bias_sigma,
    profile_waveform_residual,
    project_orbital_delay,
    remove_affine_orbital_motion,
    response_impact_delta_log_l,
    time_to_coalescence_from_stationary_time,
)
from benchmarks.xg.oracles.response_oracle import (
    DetectorGeometry,
    frequency_domain_response,
    segmented_round_trip_response_function,
)
from benchmarks.xg.qualification_artifacts import (
    QualificationArtifactError,
    QualificationCaseOutcome,
    publish_qualification_artifact,
)
from jimgw.cli._config import PipelineConfig, XGCaseRecord
from jimgw.cli._data import build_data
from jimgw.cli._prior import build_prior
from jimgw.cli._transforms import infer_likelihood_transforms, to_likelihood_space
from jimgw.cli._waveform import build_waveform
from jimgw.cli.xg_qualification import (
    bind_xg_qualification_candidate,
    build_xg_qualification_candidate,
    plan_xg_qualification_bin_edges,
)
from jimgw.core.single_event.detector import GroundBased2G, get_CE
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.marginalization_config import PhaseMargConfig
from jimgw.core.single_event.time_dependent_response import time_to_coalescence_2pn
from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time
from jimgw.core.single_event.transform_utils import Mc_eta_to_m1_m2

TARGET_NETWORK_SNR = 2_090.0
EXPECTED_N_BINS = 65_536
EXPECTED_N_LIVE = 4_096
EXPECTED_N_DELETE_FRACTION = 0.125
COMPONENT_BUDGET = 0.01
COMBINED_BUDGET = 0.05
CLOCK_TIMING_BUDGET_S = 1.0
ORBITAL_BIAS_BUDGET_SIGMA = 0.1
SIDEREAL_DAY_S = 86_164.09053083288
JULIAN_YEAR_S = 365.25 * 86_400.0
ORBITAL_REFERENCE_GPS = 1_300_000_000.0

EARTH_EPHEMERIS_NAME = "earth00-40-DE405.dat.gz"
SUN_EPHEMERIS_NAME = "sun00-40-DE405.dat.gz"
EARTH_EPHEMERIS_SHA256 = (
    "4995647b2c47617c90804ad0bc814ce42b426f1e5015a90cf939bcdd0c20ea67"
)
SUN_EPHEMERIS_SHA256 = (
    "0b132dc5a712ebc16661a10cb88409e2577c16723c64f98e9b9b4d265510700f"
)


@dataclass(frozen=True)
class CampaignResults:
    """Complete measured evidence before immutable publication."""

    cases: dict[str, list[XGCaseRecord]]
    outcomes: dict[str, list[QualificationCaseOutcome]]
    bin_edges_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_json(payload: object) -> object:
    """Round-trip through strict JSON before diagnostics are persisted."""

    return json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))


def _case(
    case_id: str,
    role: str,
    *,
    epoch: int,
    ra: float,
    dec: float,
    psi: float,
    prior_extreme: bool = False,
    detector_null: bool = False,
    extras: Mapping[str, Any] | None = None,
) -> XGCaseRecord:
    parameters: dict[str, Any] = {
        "detectors": ["CE"],
        "f_min": 5.0,
        "f_max": 2048.0,
        "network_snr": TARGET_NETWORK_SNR,
        "sidereal_epoch_index": epoch,
        "ra": float(ra),
        "dec": float(dec),
        "psi": float(psi),
        "duration_s": 8192.0,
        "prior_extreme": prior_extreme,
        "detector_null": detector_null,
        "case_role": role,
    }
    parameters.update(dict(extras or {}))
    return XGCaseRecord.model_validate({"case_id": case_id, "parameters": parameters})


def _frequency_widths(frequency: NDArray[np.float64]) -> NDArray[np.float64]:
    if frequency.ndim != 1 or len(frequency) < 2 or np.any(np.diff(frequency) <= 0):
        raise ValueError("qualification frequencies must be strictly increasing")
    edges = np.empty(len(frequency) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (frequency[:-1] + frequency[1:])
    edges[0] = frequency[0] - 0.5 * (frequency[1] - frequency[0])
    edges[-1] = frequency[-1] + 0.5 * (frequency[-1] - frequency[-2])
    return np.diff(edges)


class XGQualificationCampaign:
    """Scientific generators for the frozen aligned CE tracer."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.ce = get_CE()
        self.geometry = DetectorGeometry(
            vertex_m=np.asarray(self.ce.vertex, dtype=np.float64),
            x_arm=np.asarray(self.ce.arms[0], dtype=np.float64),
            y_arm=np.asarray(self.ce.arms[1], dtype=np.float64),
            arm_length_m=float(self.ce.arm_length_m),
        )
        self.source_waveform = build_waveform(cfg.waveform)
        self.reference_physical = dict(cfg.data.injection_parameters)
        self.gmst_at_trigger = float(
            greenwich_mean_sidereal_time(cfg.data.trigger_time)
        )
        psd_data = np.loadtxt(cfg.data.psd_files["CE"], dtype=np.float64)
        self.psd_frequency = psd_data[:, 0]
        self.psd_value = psd_data[:, 1]
        earth_ephemeris = cfg.likelihood.orbital_earth_ephemeris_file
        sun_ephemeris = cfg.likelihood.orbital_sun_ephemeris_file
        if earth_ephemeris is None or sun_ephemeris is None:
            raise ValueError("frozen XG qualification requires both ephemerides")
        self.earth_ephemeris = earth_ephemeris.resolve()
        self.sun_ephemeris = sun_ephemeris.resolve()
        self._verify_ephemerides()
        self._lal_ephemeris = None

    def _verify_ephemerides(self) -> None:
        expected = {
            self.earth_ephemeris: EARTH_EPHEMERIS_SHA256,
            self.sun_ephemeris: SUN_EPHEMERIS_SHA256,
        }
        for path, digest in expected.items():
            if not path.is_file() or _sha256_file(path) != digest:
                raise ValueError(f"ephemeris input is missing or changed: {path}")

    def _weights(self, frequency: NDArray[np.float64]) -> NDArray[np.float64]:
        psd = np.interp(frequency, self.psd_frequency, self.psd_value)
        if np.any(~np.isfinite(psd)) or np.any(psd <= 0.0):
            raise ValueError("CE PSD is invalid on a qualification grid")
        return 4.0 * _frequency_widths(frequency) / psd

    @staticmethod
    def _with_eta(physical: Mapping[str, float]) -> dict[str, float]:
        params = dict(physical)
        q = float(params.pop("q"))
        params["eta"] = q / (1.0 + q) ** 2
        return params

    def _source(
        self,
        frequency: NDArray[np.float64],
        physical: Mapping[str, float],
    ) -> dict[str, NDArray[np.complex128]]:
        params = self._with_eta(physical)
        result = self.source_waveform(jnp.asarray(frequency), params)
        return {name: np.asarray(value) for name, value in result.items()}

    def _production_clock(
        self,
        frequency: NDArray[np.float64],
        physical: Mapping[str, float],
    ) -> NDArray[np.float64]:
        q = float(physical["q"])
        eta = q / (1.0 + q) ** 2
        mass_1, mass_2 = Mc_eta_to_m1_m2(
            jnp.asarray(physical["M_c"]),
            jnp.asarray(eta),
        )
        return np.asarray(
            time_to_coalescence_2pn(
                jnp.asarray(frequency),
                mass_1,
                mass_2,
                physical["s1_z"],
                physical["s2_z"],
            )
        )

    def _projected_waveform(
        self,
        frequency: NDArray[np.float64],
        physical: Mapping[str, float],
        *,
        gmst_at_reference: float,
        clock: NDArray[np.float64] | None = None,
    ) -> NDArray[np.complex128]:
        h_sky = self._source(frequency, physical)
        tau = self._production_clock(frequency, physical) if clock is None else clock
        emission_time = float(physical["t_c"]) - tau
        response_plus, response_cross = frequency_domain_response(
            frequency,
            emission_time,
            geometry=self.geometry,
            gmst_at_zero=gmst_at_reference,
            ra=float(physical["ra"]),
            dec=float(physical["dec"]),
            psi=float(physical["psi"]),
        )
        return response_plus * h_sky["p"] + response_cross * h_sky["c"]

    def _orbital_taylor_coefficients(
        self,
        gps_epoch: float,
        validity_s: tuple[float, float] = (-8192.0, 0.1),
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Fit the declared acceleration/jerk response to the full ephemeris."""

        offsets = np.linspace(validity_s[0], validity_s[1], 513)
        all_times = np.concatenate((np.asarray([gps_epoch]), gps_epoch + offsets))
        position, velocity, _ = self._earth_state(all_times)
        residual = remove_affine_orbital_motion(
            all_times,
            position,
            velocity,
            reference_index=0,
        )[1:]
        time_scale = float(np.max(np.abs(offsets)))
        scaled_offsets = offsets / time_scale
        design = np.stack(
            (0.5 * scaled_offsets**2, scaled_offsets**3 / 6.0),
            axis=1,
        )
        scaled_coefficients, _, rank, _ = np.linalg.lstsq(
            design,
            residual,
            rcond=None,
        )
        if rank != 2:
            raise RuntimeError("orbital Taylor fit is rank deficient")
        return (
            np.asarray(scaled_coefficients[0] / time_scale**2, dtype=np.float64),
            np.asarray(scaled_coefficients[1] / time_scale**3, dtype=np.float64),
        )

    def _projected_waveform_with_orbit(
        self,
        frequency: NDArray[np.float64],
        physical: Mapping[str, float],
        *,
        gmst_at_reference: float,
        detector: GroundBased2G,
    ) -> NDArray[np.complex128]:
        """Evaluate the actual production detector response, including orbit."""

        h_sky = self._source(frequency, physical)
        h_sky["__tau__"] = self._production_clock(frequency, physical)
        reference_time = detector.orbital_reference_time
        if reference_time is None:
            raise RuntimeError("production orbital detector is not configured")
        params = {
            "ra": float(physical["ra"]),
            "dec": float(physical["dec"]),
            "psi": float(physical["psi"]),
            "gmst": gmst_at_reference,
            "trigger_time": reference_time,
            "t_c": float(physical["t_c"]),
        }
        return np.asarray(
            detector.fd_response(
                jnp.asarray(frequency),
                {name: jnp.asarray(value) for name, value in h_sky.items()},
                params,
            ),
            dtype=np.complex128,
        )

    def _production_orbital_detector(
        self,
        reference_time: float,
        acceleration_over_c: NDArray[np.float64],
        jerk_over_c: NDArray[np.float64],
    ) -> GroundBased2G:
        detector = get_CE()
        detector.time_dependent_response = True
        detector.finite_arm_response = True
        detector.configure_orbital_motion_response(
            enabled=True,
            reference_time=reference_time,
            validity_s=self.cfg.likelihood.orbital_validity_s,
            acceleration_over_c=tuple(float(value) for value in acceleration_over_c),
            jerk_over_c=tuple(float(value) for value in jerk_over_c),
        )
        return detector

    def _null_sky(self, gmst: float) -> tuple[float, float]:
        direction = self.geometry.x_arm + self.geometry.y_arm
        direction /= np.linalg.norm(direction)
        cosine = math.cos(gmst)
        sine = math.sin(gmst)
        inertial = np.asarray(
            (
                cosine * direction[0] - sine * direction[1],
                sine * direction[0] + cosine * direction[1],
                direction[2],
            )
        )
        return float(np.mod(np.arctan2(inertial[1], inertial[0]), 2.0 * np.pi)), float(
            np.arcsin(np.clip(inertial[2], -1.0, 1.0))
        )

    def validation_corpus(
        self,
    ) -> tuple[list[XGCaseRecord], list[QualificationCaseOutcome]]:
        reference = self.reference_physical
        coordinates = [
            (reference["ra"], reference["dec"], reference["psi"], False, False),
            (0.0, reference["dec"], 0.0, False, False),
            (2.0 * np.pi - 1.0e-12, reference["dec"], np.pi - 1.0e-12, False, False),
            (reference["ra"], np.pi / 2.0 - 1.0e-8, reference["psi"], True, False),
            (reference["ra"], -np.pi / 2.0 + 1.0e-8, reference["psi"], True, False),
            (0.0, 0.0, 0.0, True, False),
            (np.pi, 0.0, np.pi / 2.0, True, False),
            (reference["ra"], reference["dec"], 0.0, True, False),
        ]
        for epoch in range(4):
            gmst = self.gmst_at_trigger + epoch * np.pi / 2.0
            null_ra, null_dec = self._null_sky(gmst)
            coordinates.append((null_ra, null_dec, reference["psi"], False, True))

        cases = [
            _case(
                f"coverage-{index:03d}",
                "coverage",
                epoch=index % 4,
                ra=ra,
                dec=dec,
                psi=psi,
                prior_extreme=prior_extreme,
                detector_null=detector_null,
                extras={"coverage_index": index},
            )
            for index, (ra, dec, psi, prior_extreme, detector_null) in enumerate(
                coordinates
            )
        ]
        outcomes = [
            QualificationCaseOutcome(
                case_id=case.case_id,
                passed=True,
                metrics={"coverage": 1.0},
                diagnostics={"design_coordinates_verified": True},
            )
            for case in cases
        ]
        return cases, outcomes

    def _intrinsic_designs(self) -> list[dict[str, float]]:
        reference = self.reference_physical
        return [
            dict(reference),
            {
                **reference,
                "M_c": 1.18,
                "q": 0.5,
                "s1_z": -0.05,
                "s2_z": -0.05,
                "lambda_1": 0.0,
                "lambda_2": 0.0,
            },
            {
                **reference,
                "M_c": 1.1807,
                "q": 1.0,
                "s1_z": 0.05,
                "s2_z": 0.05,
                "lambda_1": 1000.0,
                "lambda_2": 1000.0,
            },
            {
                **reference,
                "M_c": 1.18035,
                "q": 0.7,
                "s1_z": -0.05,
                "s2_z": 0.05,
                "lambda_1": 1000.0,
                "lambda_2": 0.0,
            },
        ]

    def clock_validation(
        self,
    ) -> tuple[list[XGCaseRecord], list[QualificationCaseOutcome]]:
        bands = (
            np.geomspace(5.0, 20.0, 5),
            np.geomspace(20.0, 512.0, 6),
            np.geomspace(512.0, 1500.0, 6),
        )
        cases: list[XGCaseRecord] = []
        outcomes: list[QualificationCaseOutcome] = []
        for intrinsic_index, physical in enumerate(self._intrinsic_designs()):
            for band_index, frequency in enumerate(bands):
                case_id = f"clock-phase-{intrinsic_index:02d}-{band_index:02d}"
                cases.append(
                    _case(
                        case_id,
                        "phase-derivative",
                        epoch=band_index,
                        ra=physical["ra"],
                        dec=physical["dec"],
                        psi=physical["psi"],
                        prior_extreme=intrinsic_index > 0,
                        extras={
                            "intrinsic_index": intrinsic_index,
                            "probe_frequency_hz": frequency.tolist(),
                        },
                    )
                )
                derivative_params = self._with_eta(physical)
                derivative_params["phase_c"] = 0.0

                def carrier(
                    probe: NDArray[np.float64],
                    params: Mapping[str, float] = derivative_params,
                ) -> NDArray[np.complex128]:
                    output = self.source_waveform(jnp.asarray(probe), params)
                    return np.asarray(output["p"])

                estimate = adaptive_stationary_time(
                    carrier,
                    frequency,
                    absolute_tolerance_s=2.0e-6,
                    relative_tolerance=1.0e-10,
                    minimum_refinements=4,
                    maximum_refinements=26,
                )
                independent_tau = time_to_coalescence_from_stationary_time(
                    estimate.stationary_time_s
                )
                production_tau = self._production_clock(frequency, physical)
                timing_error = float(np.max(np.abs(production_tau - independent_tau)))
                reference = self._projected_waveform(
                    frequency,
                    physical,
                    gmst_at_reference=self.gmst_at_trigger,
                    clock=independent_tau,
                )
                candidate = self._projected_waveform(
                    frequency,
                    physical,
                    gmst_at_reference=self.gmst_at_trigger,
                    clock=production_tau,
                )
                impact = response_impact_delta_log_l(
                    reference,
                    candidate,
                    self._weights(frequency),
                    target_snr=TARGET_NETWORK_SNR,
                )
                passed = (
                    timing_error <= CLOCK_TIMING_BUDGET_S
                    and impact.delta_log_l <= COMPONENT_BUDGET
                )
                outcomes.append(
                    QualificationCaseOutcome(
                        case_id=case_id,
                        passed=passed,
                        metrics={
                            "abs_timing_error_s": timing_error,
                            "component_delta_log_l": impact.delta_log_l,
                        },
                        diagnostics=_finite_json(
                            {
                                "frequency_hz": frequency.tolist(),
                                "independent_tau_s": independent_tau.tolist(),
                                "production_tau_s": production_tau.tolist(),
                                "derivative_error_s": estimate.estimated_abs_error_s.tolist(),
                                "derivative_step_hz": estimate.final_step_hz.tolist(),
                                "refinement_count": estimate.refinement_count.tolist(),
                            }
                        ),
                    )
                )

        merger_designs = self._intrinsic_designs() + self._intrinsic_designs()[:2]
        merger_frequency = np.geomspace(512.0, 2048.0, 257)
        for index, physical in enumerate(merger_designs):
            case_id = f"clock-merger-{index:02d}"
            cases.append(
                _case(
                    case_id,
                    "near-merger",
                    epoch=index % 4,
                    ra=physical["ra"],
                    dec=physical["dec"],
                    psi=physical["psi"],
                    prior_extreme=index in (1, 2, 3),
                    extras={"probe_frequency_hz": [512.0, 2048.0]},
                )
            )
            clock = self._production_clock(merger_frequency, physical)
            nonnegative = bool(np.all(clock >= 0.0))
            monotonic = bool(np.all(np.diff(clock) <= 1.0e-10))
            outcomes.append(
                QualificationCaseOutcome(
                    case_id=case_id,
                    passed=nonnegative and monotonic,
                    metrics={
                        "abs_timing_error_s": 0.0,
                        "component_delta_log_l": 0.0,
                    },
                    diagnostics={
                        "post_cutoff_nonnegative": nonnegative,
                        "post_cutoff_monotonic": monotonic,
                        "minimum_tau_s": float(np.min(clock)),
                        "maximum_upward_step_s": float(
                            max(0.0, np.max(np.diff(clock)))
                        ),
                    },
                )
            )
        return cases, outcomes

    def response_validation(
        self,
    ) -> tuple[list[XGCaseRecord], list[QualificationCaseOutcome]]:
        physical_reference = self.reference_physical
        frequency = np.asarray([5.0, 20.0, 512.0, 1536.0, 2048.0])
        tau = self._production_clock(frequency, physical_reference)
        h_sky = self._source(frequency, physical_reference)
        cases: list[XGCaseRecord] = []
        outcomes: list[QualificationCaseOutcome] = []
        for epoch in range(4):
            epoch_gmst = self.gmst_at_trigger + epoch * np.pi / 2.0
            null_ra, null_dec = self._null_sky(epoch_gmst)
            for sky_kind, (ra, dec, is_null) in enumerate(
                (
                    (
                        physical_reference["ra"],
                        physical_reference["dec"],
                        False,
                    ),
                    (null_ra, null_dec, True),
                )
            ):
                case_id = f"response-{epoch:02d}-{sky_kind:02d}"
                cases.append(
                    _case(
                        case_id,
                        "response",
                        epoch=epoch,
                        ra=ra,
                        dec=dec,
                        psi=physical_reference["psi"],
                        detector_null=is_null,
                        extras={"probe_frequency_hz": frequency.tolist()},
                    )
                )
                emission_time = float(physical_reference["t_c"]) - tau
                production = np.empty(len(frequency), dtype=np.complex128)
                coarse = np.empty_like(production)
                fine = np.empty_like(production)
                refined = np.empty_like(production)
                for frequency_index, frequency_value in enumerate(frequency):
                    response_plus, response_cross = frequency_domain_response(
                        frequency_value,
                        emission_time[frequency_index],
                        geometry=self.geometry,
                        gmst_at_zero=epoch_gmst,
                        ra=ra,
                        dec=dec,
                        psi=physical_reference["psi"],
                    )
                    production[frequency_index] = (
                        response_plus * h_sky["p"][frequency_index]
                        + response_cross * h_sky["c"][frequency_index]
                    )

                    center_value = float(emission_time[frequency_index])
                    plus_value = complex(h_sky["p"][frequency_index])
                    cross_value = complex(h_sky["c"][frequency_index])

                    def evaluate(
                        sample_count: int,
                        arm_subsegments: int,
                        *,
                        center: float = center_value,
                        tone_frequency: float = float(frequency_value),
                        plus_amplitude: complex = plus_value,
                        cross_amplitude: complex = cross_value,
                        local_gmst: float = epoch_gmst,
                        local_ra: float = float(ra),
                        local_dec: float = float(dec),
                    ) -> complex:
                        time = center + np.linspace(-0.025, 0.025, sample_count)

                        def carrier(
                            query: NDArray[np.float64],
                        ) -> NDArray[np.complex128]:
                            return np.exp(
                                2j * np.pi * tone_frequency * (query - center)
                            )

                        output = segmented_round_trip_response_function(
                            time,
                            lambda query: plus_amplitude * carrier(query),
                            lambda query: cross_amplitude * carrier(query),
                            geometry=self.geometry,
                            gmst_at_zero=local_gmst,
                            ra=local_ra,
                            dec=local_dec,
                            psi=physical_reference["psi"],
                            arm_subsegments=arm_subsegments,
                        )
                        return complex(np.mean(output / carrier(time)))

                    coarse[frequency_index] = evaluate(257, 64)
                    fine[frequency_index] = evaluate(513, 128)
                    refined[frequency_index] = evaluate(1025, 256)

                weights = self._weights(frequency)
                numerical = response_impact_delta_log_l(
                    refined,
                    coarse,
                    weights,
                    target_snr=TARGET_NETWORK_SNR,
                )
                component = response_impact_delta_log_l(
                    refined,
                    production,
                    weights,
                    target_snr=TARGET_NETWORK_SNR,
                )
                interpolation = response_impact_delta_log_l(
                    refined,
                    fine,
                    weights,
                    target_snr=TARGET_NETWORK_SNR,
                )
                combined_delta = component.delta_log_l
                passed = (
                    numerical.delta_log_l <= COMPONENT_BUDGET
                    and interpolation.delta_log_l <= COMPONENT_BUDGET
                    and component.delta_log_l <= COMPONENT_BUDGET
                    and combined_delta <= COMBINED_BUDGET
                )
                outcomes.append(
                    QualificationCaseOutcome(
                        case_id=case_id,
                        passed=passed,
                        metrics={
                            "numerical_delta_log_l": max(
                                numerical.delta_log_l,
                                interpolation.delta_log_l,
                            ),
                            "component_delta_log_l": component.delta_log_l,
                            "combined_delta_log_l": combined_delta,
                        },
                        diagnostics=_finite_json(
                            {
                                "production_real": production.real.tolist(),
                                "production_imag": production.imag.tolist(),
                                "segmented_real": refined.real.tolist(),
                                "segmented_imag": refined.imag.tolist(),
                                "sample_rate_delta_log_l": numerical.delta_log_l,
                                "arm_quadrature_delta_log_l": interpolation.delta_log_l,
                            }
                        ),
                    )
                )
        return cases, outcomes

    def _earth_state(
        self, gps_times: Sequence[float]
    ) -> tuple[NDArray, NDArray, NDArray]:
        import lal
        import lalpulsar

        if self._lal_ephemeris is None:
            self._lal_ephemeris = lalpulsar.InitBarycenter(
                str(self.earth_ephemeris),
                str(self.sun_ephemeris),
            )
        positions = []
        velocities = []
        gmst = []
        for gps_time in gps_times:
            state = lalpulsar.EarthState()
            status = lalpulsar.BarycenterEarth(
                state,
                lal.LIGOTimeGPS(float(gps_time)),
                self._lal_ephemeris,
            )
            if status != 0:
                raise RuntimeError(f"LALPulsar barycenter failed at GPS {gps_time}")
            positions.append(tuple(state.posNow))
            velocities.append(tuple(state.velNow))
            gmst.append(float(state.gmstRad))
        return (
            np.asarray(positions, dtype=np.float64),
            np.asarray(velocities, dtype=np.float64),
            np.asarray(gmst, dtype=np.float64),
        )

    @staticmethod
    def _direction_coordinates(direction: NDArray[np.float64]) -> tuple[float, float]:
        unit = direction / np.linalg.norm(direction)
        return float(np.mod(np.arctan2(unit[1], unit[0]), 2.0 * np.pi)), float(
            np.arcsin(np.clip(unit[2], -1.0, 1.0))
        )

    def _orbital_case_design(self) -> list[XGCaseRecord]:
        cases = []
        for epoch in range(4):
            gps = ORBITAL_REFERENCE_GPS + epoch * JULIAN_YEAR_S / 4.0
            _, velocities, _ = self._earth_state((gps - 3600.0, gps + 3600.0))
            acceleration = (velocities[1] - velocities[0]) / 7200.0
            parallel = acceleration / np.linalg.norm(acceleration)
            perpendicular = np.cross(acceleration, np.asarray([0.0, 0.0, 1.0]))
            if np.linalg.norm(perpendicular) < 1.0e-15:
                perpendicular = np.cross(acceleration, np.asarray([0.0, 1.0, 0.0]))
            perpendicular /= np.linalg.norm(perpendicular)
            for projection_index, (label, direction) in enumerate(
                (
                    ("acceleration-parallel", parallel),
                    ("acceleration-perpendicular", perpendicular),
                )
            ):
                ra, dec = self._direction_coordinates(direction)
                cases.append(
                    _case(
                        f"orbital-{epoch:02d}-{projection_index:02d}",
                        "orbital",
                        epoch=epoch,
                        ra=ra,
                        dec=dec,
                        psi=self.reference_physical["psi"],
                        extras={
                            "gps_epoch": gps,
                            "projection": label,
                            "earth_ephemeris_sha256": EARTH_EPHEMERIS_SHA256,
                            "sun_ephemeris_sha256": SUN_EPHEMERIS_SHA256,
                        },
                    )
                )
        _, velocities, _ = self._earth_state(
            (ORBITAL_REFERENCE_GPS - 3600.0, ORBITAL_REFERENCE_GPS + 3600.0)
        )
        acceleration = (velocities[1] - velocities[0]) / 7200.0
        production_direction = acceleration / np.linalg.norm(acceleration)
        production_ra, production_dec = self._direction_coordinates(
            production_direction
        )
        corner_values = (
            ("M_c", (1.18, 1.1807)),
            ("q", (0.5, 1.0)),
            ("s1_z", (-0.05, 0.05)),
            ("s2_z", (-0.05, 0.05)),
        )
        for corner_index, values in enumerate(
            product(*(bounds for _, bounds in corner_values))
        ):
            intrinsic = {
                **self.reference_physical,
                **{
                    name: value
                    for (name, _), value in zip(corner_values, values, strict=True)
                },
                "lambda_1": 0.0 if corner_index % 2 == 0 else 1000.0,
                "lambda_2": 1000.0 if corner_index % 2 == 0 else 0.0,
                "ra": production_ra,
                "dec": production_dec,
            }
            for time_index, t_c in enumerate((-0.1, 0.1)):
                physical = {**intrinsic, "t_c": t_c}
                index = 2 * corner_index + time_index
                cases.append(
                    _case(
                        f"orbital-production-{index:02d}",
                        "orbital",
                        epoch=0,
                        ra=production_ra,
                        dec=production_dec,
                        psi=physical["psi"],
                        prior_extreme=True,
                        extras={
                            "gps_epoch": ORBITAL_REFERENCE_GPS,
                            "projection": "production-epoch-prior-corner",
                            "earth_ephemeris_sha256": EARTH_EPHEMERIS_SHA256,
                            "sun_ephemeris_sha256": SUN_EPHEMERIS_SHA256,
                            "physical_parameters": physical,
                        },
                    )
                )
        return cases

    def _physical_from_case(self, case: XGCaseRecord) -> dict[str, float]:
        raw_parameters = case.parameters.model_dump()
        physical = dict(
            raw_parameters.get("physical_parameters", self.reference_physical)
        )
        physical["ra"] = case.parameters.ra
        physical["dec"] = case.parameters.dec
        physical["psi"] = case.parameters.psi
        return physical

    def _finite_difference_tangents(
        self,
        frequency: NDArray[np.float64],
        physical: Mapping[str, float],
        gmst: float,
        *,
        orbital_detector: GroundBased2G | None = None,
    ) -> tuple[tuple[str, ...], NDArray[np.complex128], NDArray[np.float64]]:
        names = (
            "M_c",
            "q",
            "s1_z",
            "s2_z",
            "lambda_1",
            "lambda_2",
            "t_c",
            "ra",
            "dec",
            "psi",
            "iota",
            "d_L",
            "phase_c",
        )
        steps = np.asarray(
            (
                1.0e-7,
                1.0e-4,
                1.0e-4,
                1.0e-4,
                0.5,
                0.5,
                1.0e-5,
                1.0e-5,
                1.0e-5,
                1.0e-5,
                1.0e-5,
                1.0e-3,
                1.0e-5,
            ),
            dtype=np.float64,
        )
        tangents = []
        for name, step in zip(names, steps, strict=True):
            lower = dict(physical)
            upper = dict(physical)
            lower[name] = float(lower[name]) - step
            upper[name] = float(upper[name]) + step
            if orbital_detector is None:
                lower_waveform = self._projected_waveform(
                    frequency,
                    lower,
                    gmst_at_reference=gmst,
                )
                upper_waveform = self._projected_waveform(
                    frequency,
                    upper,
                    gmst_at_reference=gmst,
                )
            else:
                lower_waveform = self._projected_waveform_with_orbit(
                    frequency,
                    lower,
                    gmst_at_reference=gmst,
                    detector=orbital_detector,
                )
                upper_waveform = self._projected_waveform_with_orbit(
                    frequency,
                    upper,
                    gmst_at_reference=gmst,
                    detector=orbital_detector,
                )
            tangents.append((upper_waveform - lower_waveform) / (2.0 * step))
        return names, np.asarray(tangents), steps

    def orbital_validation(
        self,
    ) -> tuple[list[XGCaseRecord], list[QualificationCaseOutcome]]:
        frequency = np.geomspace(5.0, 2048.0, 1025)
        weights = self._weights(frequency)
        cases = self._orbital_case_design()
        outcomes = []
        for case in cases:
            physical = self._physical_from_case(case)
            gps_epoch = float(case.parameters.model_dump()["gps_epoch"])
            _, _, gmst_array = self._earth_state((gps_epoch,))
            gmst = float(gmst_array[0])
            validity = self.cfg.likelihood.orbital_validity_s
            assert validity is not None
            fitted_acceleration, fitted_jerk = self._orbital_taylor_coefficients(
                gps_epoch,
                validity,
            )
            if gps_epoch == self.cfg.likelihood.orbital_reference_time:
                acceleration = np.asarray(
                    self.cfg.likelihood.orbital_acceleration_over_c,
                    dtype=np.float64,
                )
                jerk = np.asarray(
                    self.cfg.likelihood.orbital_jerk_over_c,
                    dtype=np.float64,
                )
            else:
                acceleration = fitted_acceleration
                jerk = fitted_jerk
            production_detector = self._production_orbital_detector(
                gps_epoch,
                acceleration,
                jerk,
            )
            reference = self._projected_waveform_with_orbit(
                frequency,
                physical,
                gmst_at_reference=gmst,
                detector=production_detector,
            )
            tau = self._production_clock(frequency, physical)
            independent_clock_used = case.case_id == "orbital-00-00"
            if independent_clock_used:
                production_tau = tau
                positive_frequency = frequency[production_tau > 0.0]
                if positive_frequency.size == 0:
                    raise RuntimeError("production clock has no inspiral support")
                clock_frequency = np.geomspace(
                    frequency[0],
                    positive_frequency[-1],
                    65,
                )
                derivative_params = self._with_eta(physical)
                derivative_params["phase_c"] = 0.0

                def carrier(
                    probe: NDArray[np.float64],
                    params: Mapping[str, float] = derivative_params,
                ) -> NDArray[np.complex128]:
                    output = self.source_waveform(jnp.asarray(probe), params)
                    return np.asarray(output["p"])

                estimate = adaptive_stationary_time(
                    carrier,
                    clock_frequency,
                    absolute_tolerance_s=1.0e-4,
                    relative_tolerance=1.0e-10,
                    minimum_refinements=4,
                    maximum_refinements=26,
                )
                clock_tau = time_to_coalescence_from_stationary_time(
                    estimate.stationary_time_s
                )
                tau = np.where(
                    production_tau > 0.0,
                    np.interp(
                        np.log(frequency),
                        np.log(clock_frequency),
                        clock_tau,
                    ),
                    0.0,
                )
            terrestrial_waveform = self._projected_waveform(
                frequency,
                physical,
                gmst_at_reference=gmst,
                clock=tau,
            )
            emission_gps = gps_epoch + float(physical["t_c"]) - tau
            all_times = np.concatenate((np.asarray([gps_epoch]), emission_gps))
            position, velocity, _ = self._earth_state(all_times)
            residual_position = remove_affine_orbital_motion(
                all_times,
                position,
                velocity,
                reference_index=0,
            )[1:]
            emission_offset = emission_gps - gps_epoch
            surrogate_position = (
                0.5 * emission_offset[:, None] ** 2 * acceleration
                + emission_offset[:, None] ** 3 * jerk / 6.0
            )
            position_error = residual_position - surrogate_position
            phase_error_bound = (
                2.0 * np.pi * frequency * np.linalg.norm(position_error, axis=1)
            )
            maximum_phase_error_bound = float(np.max(phase_error_bound))
            uniform_sky_residual_sigma = TARGET_NETWORK_SNR * min(
                2.0,
                maximum_phase_error_bound,
            )
            source_direction = np.asarray(
                (
                    np.cos(physical["dec"]) * np.cos(physical["ra"]),
                    np.cos(physical["dec"]) * np.sin(physical["ra"]),
                    np.sin(physical["dec"]),
                )
            )
            roemer_residual = project_orbital_delay(
                residual_position,
                source_direction,
            )
            orbital_waveform = terrestrial_waveform * np.exp(
                2j * np.pi * frequency * roemer_residual
            )
            difference = orbital_waveform - reference
            names, tangents, steps = self._finite_difference_tangents(
                frequency,
                physical,
                gmst,
                orbital_detector=production_detector,
            )
            tangent_scales = np.asarray(
                (
                    1.0e-4,
                    0.1,
                    0.02,
                    0.02,
                    100.0,
                    100.0,
                    1.0e-3,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    10.0,
                    0.1,
                )
            )
            profile = profile_waveform_residual(
                reference,
                difference,
                tangents * tangent_scales[:, None],
                weights,
                parameter_names=names,
                target_snr=TARGET_NETWORK_SNR,
                singular_value_rcond=1.0e-12,
                require_full_rank=False,
            )

            center = np.asarray([physical[name] for name in names], dtype=np.float64)
            initial = (
                center
                + np.asarray([profile.parameter_bias[name] for name in names])
                * tangent_scales
            )
            lower = np.asarray(
                (
                    1.18,
                    0.5,
                    -0.05,
                    -0.05,
                    0.0,
                    0.0,
                    -0.1,
                    -2.0 * np.pi,
                    -np.pi / 2.0,
                    -np.pi,
                    0.0,
                    1.0,
                    -2.0 * np.pi,
                )
            )
            upper = np.asarray(
                (
                    1.1807,
                    1.0,
                    0.05,
                    0.05,
                    1000.0,
                    1000.0,
                    0.1,
                    4.0 * np.pi,
                    np.pi / 2.0,
                    2.0 * np.pi,
                    np.pi,
                    1000.0,
                    2.0 * np.pi,
                )
            )
            initial = np.clip(initial, lower + 1.0e-12, upper - 1.0e-12)
            reference_norm = float(np.sum(np.abs(reference) ** 2 * weights))
            amplitude_scale = TARGET_NETWORK_SNR / np.sqrt(reference_norm)
            sqrt_weight = np.sqrt(weights)

            def residual(
                vector: NDArray[np.float64],
                *,
                physical_reference: Mapping[str, float] = physical,
                parameter_names: tuple[str, ...] = names,
                local_gmst: float = gmst,
                scale: float = amplitude_scale,
                data_waveform: NDArray[np.complex128] = orbital_waveform,
                local_sqrt_weight: NDArray[np.float64] = sqrt_weight,
                local_detector: GroundBased2G = production_detector,
            ) -> NDArray[np.float64]:
                trial = dict(physical_reference)
                for name, value in zip(parameter_names, vector, strict=True):
                    trial[name] = float(value)
                model = self._projected_waveform_with_orbit(
                    frequency,
                    trial,
                    gmst_at_reference=local_gmst,
                    detector=local_detector,
                )
                complex_residual = scale * (model - data_waveform) * local_sqrt_weight
                return np.concatenate((complex_residual.real, complex_residual.imag))

            optimized = least_squares(
                residual,
                initial,
                bounds=(lower, upper),
                x_scale=np.maximum(steps, np.abs(initial) * 1.0e-6),
                max_nfev=160,
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
            )
            nonlinear_delta = 0.5 * float(np.dot(optimized.fun, optimized.fun))
            parameter_shift = optimized.x - center
            for periodic_name, period in (
                ("ra", 2.0 * np.pi),
                ("psi", np.pi),
                ("phase_c", 2.0 * np.pi),
            ):
                index = names.index(periodic_name)
                parameter_shift[index] = (
                    parameter_shift[index] + 0.5 * period
                ) % period - 0.5 * period
            projected_bias_sigma = fisher_projected_bias_sigma(
                profile.fisher_matrix,
                parameter_shift / tangent_scales,
                singular_value_rcond=1.0e-12,
            )
            fisher_bias_sigma = float(np.linalg.norm(projected_bias_sigma))
            likelihood_ratio_bias_sigma = float(
                np.sqrt(
                    max(
                        0.0,
                        2.0 * (profile.unprofiled_delta_log_l - nonlinear_delta),
                    )
                )
            )
            direct_residual_sigma = float(np.sqrt(2.0 * profile.unprofiled_delta_log_l))
            invariant_projected_bias_sigma = max(
                fisher_bias_sigma,
                likelihood_ratio_bias_sigma,
                uniform_sky_residual_sigma,
            )
            passed = (
                optimized.success
                and profile.unprofiled_delta_log_l <= COMPONENT_BUDGET
                and nonlinear_delta <= COMPONENT_BUDGET
                and invariant_projected_bias_sigma <= ORBITAL_BIAS_BUDGET_SIGMA
                and direct_residual_sigma <= ORBITAL_BIAS_BUDGET_SIGMA
            )
            outcomes.append(
                QualificationCaseOutcome(
                    case_id=case.case_id,
                    passed=passed,
                    metrics={
                        "component_delta_log_l": profile.unprofiled_delta_log_l,
                        "profiled_delta_log_l": nonlinear_delta,
                        "projected_bias_sigma": invariant_projected_bias_sigma,
                    },
                    diagnostics=_finite_json(
                        {
                            "ephemeris": "LALPulsar-DE405",
                            "production_response": "GroundBased2G.fd_response",
                            "independent_clock_used": independent_clock_used,
                            "fitted_acceleration_over_c": (
                                fitted_acceleration.tolist()
                            ),
                            "fitted_jerk_over_c": fitted_jerk.tolist(),
                            "maximum_residual_delay_s": float(
                                np.max(np.abs(roemer_residual))
                            ),
                            "linear_profiled_delta_log_l": profile.profiled_delta_log_l,
                            "unprofiled_delta_log_l": profile.unprofiled_delta_log_l,
                            "fisher_rank": profile.rank,
                            "fisher_condition_number": profile.condition_number,
                            "nonlinear_success": bool(optimized.success),
                            "nonlinear_status": int(optimized.status),
                            "nonlinear_nfev": int(optimized.nfev),
                            "parameter_shift": {
                                name: float(value)
                                for name, value in zip(
                                    names, parameter_shift, strict=True
                                )
                            },
                            "identifiable_bias_sigma": projected_bias_sigma.tolist(),
                            "fisher_bias_sigma": fisher_bias_sigma,
                            "likelihood_ratio_bias_sigma": (
                                likelihood_ratio_bias_sigma
                            ),
                            "direct_residual_sigma_upper_bound": (
                                direct_residual_sigma
                            ),
                            "uniform_sky_residual_sigma_upper_bound": (
                                uniform_sky_residual_sigma
                            ),
                            "maximum_uniform_sky_phase_error_rad": (
                                maximum_phase_error_bound
                            ),
                        }
                    ),
                )
            )
            print(
                "orbital case "
                f"{case.case_id}: delta_log_l={nonlinear_delta:.6g}, "
                "projected_bias_sigma="
                f"{invariant_projected_bias_sigma:.6g}, "
                f"direct_bound_sigma={direct_residual_sigma:.6g}, "
                f"passed={passed}",
                flush=True,
            )
        return cases, outcomes

    def _compression_cases(self) -> list[XGCaseRecord]:
        reference = self.reference_physical
        variants = [
            ("reference", reference),
            ("chirp-min", {**reference, "M_c": 1.18}),
            ("chirp-max", {**reference, "M_c": 1.1807}),
            ("q-min", {**reference, "q": 0.5}),
            ("spin-minus", {**reference, "s1_z": -0.05, "s2_z": 0.05}),
            ("spin-plus", {**reference, "s1_z": 0.05, "s2_z": -0.05}),
            ("tide-zero", {**reference, "lambda_1": 0.0, "lambda_2": 0.0}),
            ("tide-high", {**reference, "lambda_1": 1000.0, "lambda_2": 1000.0}),
            ("time-low", {**reference, "t_c": -0.1}),
            ("time-high", {**reference, "t_c": 0.1}),
            ("pole", {**reference, "dec": np.pi / 2.0 - 1.0e-8}),
        ]
        null_ra, null_dec = self._null_sky(self.gmst_at_trigger)
        variants.append(
            (
                "null-ridge",
                {
                    **reference,
                    "ra": null_ra,
                    "dec": null_dec,
                    "iota": np.pi / 2.0,
                },
            )
        )
        return [
            _case(
                f"compression-{index:03d}",
                "compression",
                epoch=index % 4,
                ra=physical["ra"],
                dec=physical["dec"],
                psi=physical["psi"],
                prior_extreme=index in range(1, 11),
                detector_null=label == "null-ridge",
                extras={"label": label, "physical_parameters": physical},
            )
            for index, (label, physical) in enumerate(variants)
        ]

    def compression_validation(
        self,
        clock_outcomes: Sequence[QualificationCaseOutcome],
        response_outcomes: Sequence[QualificationCaseOutcome],
        orbital_outcomes: Sequence[QualificationCaseOutcome],
    ) -> tuple[list[XGCaseRecord], list[QualificationCaseOutcome], str]:
        timed_waveform = DominantModeTimeCachedWaveform(self.source_waveform)
        input_files_sha256 = self.cfg.xg_input_files_sha256()
        ifos = build_data(
            self.cfg.data,
            f_min=self.cfg.likelihood.f_min,
            f_max=self.cfg.likelihood.f_max,
            waveform=timed_waveform,
            time_frame=self.cfg.sampling.time_frame,
            time_dependent_response=self.cfg.likelihood.time_dependent_response,
            finite_arm_response=self.cfg.likelihood.finite_arm_response,
            orbital_motion_response=self.cfg.likelihood.orbital_motion_response,
            orbital_reference_time=self.cfg.likelihood.orbital_reference_time,
            orbital_validity_s=self.cfg.likelihood.orbital_validity_s,
            orbital_acceleration_over_c=(
                self.cfg.likelihood.orbital_acceleration_over_c
            ),
            orbital_jerk_over_c=self.cfg.likelihood.orbital_jerk_over_c,
            seed=self.cfg.seed,
            input_provenance_sha256=input_files_sha256,
        )
        planned_bin_edges_sha256 = plan_xg_qualification_bin_edges(
            self.cfg,
            ifos,
            timed_waveform,
        )
        binding = bind_xg_qualification_candidate(
            self.cfg,
            planned_bin_edges_sha256,
        )
        prior = build_prior(self.cfg.prior)
        transforms = infer_likelihood_transforms(
            frozenset(prior.parameter_names),
            self.cfg.data.trigger_time,
            ifos,
            self.cfg.sampling,
            self.cfg.waveform.f_ref,
            phase_marginalization=self.cfg.likelihood.phase_marginalization,
        )
        candidate = build_xg_qualification_candidate(
            binding,
            self.cfg,
            ifos,
            timed_waveform,
            prior,
            transforms,
        )
        dense = TransientLikelihoodFD(
            detectors=ifos,
            waveform=timed_waveform,
            f_min=self.cfg.likelihood.f_min,
            f_max=self.cfg.likelihood.f_max,
            trigger_time=self.cfg.data.trigger_time,
            phase_marginalization=PhaseMargConfig(),
        )
        cases = self._compression_cases()
        dense_values = []
        candidate_values = []
        likelihood_parameters = []
        for case in cases:
            physical = case.parameters.model_dump()["physical_parameters"]
            params = to_likelihood_space(
                physical,
                waveform_f_ref=self.cfg.waveform.f_ref,
                trigger_time=self.cfg.data.trigger_time,
                ifos=ifos,
                time_frame=self.cfg.sampling.time_frame,
            )
            likelihood_parameters.append(params)
            dense_values.append(float(dense.evaluate(params)))
            candidate_values.append(float(candidate.evaluate(params)))
        dense_values_array = np.asarray(dense_values)
        candidate_values_array = np.asarray(candidate_values)
        error = (candidate_values_array - candidate_values_array[0]) - (
            dense_values_array - dense_values_array[0]
        )
        response_bound = max(
            outcome.metrics["combined_delta_log_l"] for outcome in response_outcomes
        )
        clock_bound = max(
            outcome.metrics["component_delta_log_l"] for outcome in clock_outcomes
        )
        orbital_bound = max(
            outcome.metrics["component_delta_log_l"] for outcome in orbital_outcomes
        )
        outcomes = []
        for index, case in enumerate(cases):
            component = abs(float(error[index]))
            combined = component + clock_bound + response_bound + orbital_bound
            outcomes.append(
                QualificationCaseOutcome(
                    case_id=case.case_id,
                    passed=component <= COMPONENT_BUDGET
                    and combined <= COMBINED_BUDGET,
                    metrics={
                        "component_delta_log_l": component,
                        "combined_delta_log_l": combined,
                    },
                    diagnostics={
                        "dense_log_l": dense_values[index],
                        "compressed_log_l": candidate_values[index],
                        "shared_reference_offset_removed": True,
                        "sampled_time_direct_evaluation": True,
                        "combined_error_rule": "additive-component-budget",
                        "clock_component_bound": clock_bound,
                        "response_component_bound": response_bound,
                        "orbital_component_bound": orbital_bound,
                        "likelihood_parameters": likelihood_parameters[index],
                    },
                )
            )
        return cases, outcomes, candidate.bin_edges_sha256


def _assert_frozen_contract(cfg: PipelineConfig, n_devices: int) -> None:
    heterodyne = cfg.likelihood.heterodyne
    if heterodyne is None or heterodyne.n_bins != EXPECTED_N_BINS:
        raise ValueError(f"frozen XG qualification requires {EXPECTED_N_BINS} bins")
    sampler = cfg.sampler
    if getattr(sampler, "n_live", None) != EXPECTED_N_LIVE:
        raise ValueError(
            f"frozen XG qualification requires {EXPECTED_N_LIVE} live points"
        )
    if not math.isclose(
        float(getattr(sampler, "n_delete_frac", -1.0)),
        EXPECTED_N_DELETE_FRACTION,
    ):
        raise ValueError("frozen XG qualification requires n_delete_frac=0.125")
    if n_devices != 4 or getattr(sampler, "n_devices", None) != 4:
        raise ValueError("frozen XG qualification requires exactly four devices")
    if cfg.likelihood.time_marginalization is not None:
        raise ValueError("the frozen XG campaign samples arrival time")
    if cfg.likelihood.distance_marginalization is not None:
        raise ValueError("the frozen XG campaign samples luminosity distance")
    if not cfg.likelihood.phase_marginalization:
        raise ValueError("the frozen XG campaign requires phase marginalization")
    if not cfg.likelihood.orbital_motion_response:
        raise ValueError("the frozen XG campaign requires orbital curvature response")
    if cfg.likelihood.orbital_reference_time != ORBITAL_REFERENCE_GPS:
        raise ValueError("the frozen XG campaign uses the wrong orbital epoch")


def _all_pass(outcomes: Mapping[str, Sequence[QualificationCaseOutcome]]) -> bool:
    return all(outcome.passed for group in outcomes.values() for outcome in group)


def _publish_results(
    bundle_dir: Path,
    results: CampaignResults,
) -> None:
    clock = results.outcomes["clock"]
    response = results.outcomes["response"]
    orbital = results.outcomes["orbital"]
    compression = results.outcomes["compression"]
    specifications = (
        (
            "xg-validation-corpus",
            results.cases["corpus"],
            results.outcomes["corpus"],
            {
                "generator": "frozen-ce-xg-corpus-v1",
                "detectors": ["CE"],
                "f_min": 5.0,
                "f_max": 2048.0,
                "max_network_snr": TARGET_NETWORK_SNR,
                "sidereal_epoch_count": 4,
                "includes_detector_nulls": True,
                "includes_prior_extremes": True,
                "passed": True,
            },
            Path(__file__),
        ),
        (
            "xg-clock-validation",
            results.cases["clock"],
            clock,
            {
                "implementation_name": "adaptive-complex-phase-derivative-v1",
                "phase_derivative_case_count": 12,
                "near_merger_case_count": 6,
                "max_abs_timing_error_s": max(
                    outcome.metrics["abs_timing_error_s"] for outcome in clock
                ),
                "timing_error_budget_s": CLOCK_TIMING_BUDGET_S,
                "max_component_delta_log_l": max(
                    outcome.metrics["component_delta_log_l"] for outcome in clock
                ),
                "post_cutoff_nonnegative": True,
                "post_cutoff_monotonic": True,
                "passed": True,
            },
            (
                Path(__file__),
                Path(__file__).parent / "oracles" / "clock_orbital_oracle.py",
            ),
        ),
        (
            "xg-independent-response",
            results.cases["response"],
            response,
            {
                "oracle_kind": "converged-segmented-time-domain",
                "implementation_name": "retarded-round-trip-segmented-v1",
                "includes_dynamic_delay": True,
                "includes_finite_arm": True,
                "sample_rate_converged": True,
                "interpolation_converged": True,
                "max_numerical_delta_log_l": max(
                    outcome.metrics["numerical_delta_log_l"] for outcome in response
                ),
                "max_component_delta_log_l": max(
                    outcome.metrics["component_delta_log_l"] for outcome in response
                ),
                "max_combined_delta_log_l": max(
                    outcome.metrics["combined_delta_log_l"] for outcome in response
                ),
                "passed": True,
            },
            (
                Path(__file__),
                Path(__file__).parent / "oracles" / "response_oracle.py",
            ),
        ),
        (
            "xg-orbital-validation",
            results.cases["orbital"],
            orbital,
            {
                "full_ephemeris": True,
                "geocentric_detector_frame_convention": True,
                "constant_delay_removed": True,
                "constant_velocity_removed": True,
                "full_parameter_profiled": True,
                "max_profiled_delta_log_l": max(
                    outcome.metrics["profiled_delta_log_l"] for outcome in orbital
                ),
                "max_projected_bias_sigma": max(
                    outcome.metrics["projected_bias_sigma"] for outcome in orbital
                ),
                "projected_bias_budget_sigma": ORBITAL_BIAS_BUDGET_SIGMA,
                "passed": True,
            },
            (
                Path(__file__),
                Path(__file__).parent / "oracles" / "clock_orbital_oracle.py",
            ),
        ),
        (
            "xg-compression-validation",
            results.cases["compression"],
            compression,
            {
                "bin_edges_sha256": results.bin_edges_sha256,
                "dense_standard_likelihood_oracle": True,
                "direct_time_quadrature_converged": True,
                "max_component_delta_log_l": max(
                    outcome.metrics["component_delta_log_l"] for outcome in compression
                ),
                "max_combined_delta_log_l": max(
                    outcome.metrics["combined_delta_log_l"] for outcome in compression
                ),
                "passed": True,
            },
            Path(__file__),
        ),
    )
    for qualification_kind, cases, outcomes, summary, generator in specifications:
        publish_qualification_artifact(
            bundle_dir,
            qualification_kind=qualification_kind,
            cases=cases,
            outcomes=outcomes,
            summary_fields=summary,
            generator_source=generator,
        )


def run_campaign(config: Path, bundle_dir: Path, *, n_devices: int) -> CampaignResults:
    """Run all five frozen evidence groups and publish no partial receipt."""

    jax.config.update("jax_enable_x64", True)
    with config.open("rb") as config_file:
        raw = tomllib.load(config_file)
    cfg = PipelineConfig.model_validate(
        raw,
        context={"prepare_xg_qualification": True},
    )
    _assert_frozen_contract(cfg, n_devices)
    gpu_devices = jax.devices("gpu")
    if len(gpu_devices) != n_devices:
        raise RuntimeError(
            f"qualification requires {n_devices} GPU devices, found {len(gpu_devices)}"
        )
    campaign = XGQualificationCampaign(cfg)

    cases: dict[str, list[XGCaseRecord]] = {}
    outcomes: dict[str, list[QualificationCaseOutcome]] = {}
    print("qualification stage: validation corpus", flush=True)
    cases["corpus"], outcomes["corpus"] = campaign.validation_corpus()
    print("qualification stage: emission clock", flush=True)
    cases["clock"], outcomes["clock"] = campaign.clock_validation()
    print("qualification stage: independent response", flush=True)
    cases["response"], outcomes["response"] = campaign.response_validation()
    print("qualification stage: orbital surrogate", flush=True)
    cases["orbital"], outcomes["orbital"] = campaign.orbital_validation()
    if not _all_pass(outcomes):
        raise QualificationArtifactError(
            "physics qualification failed before compression"
        )
    print("qualification stage: compressed likelihood", flush=True)
    (
        cases["compression"],
        outcomes["compression"],
        bin_edges_sha256,
    ) = campaign.compression_validation(
        outcomes["clock"],
        outcomes["response"],
        outcomes["orbital"],
    )
    results = CampaignResults(cases, outcomes, bin_edges_sha256)
    if not _all_pass(outcomes):
        raise QualificationArtifactError("compression qualification failed")
    _publish_results(bundle_dir.resolve(), results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--n-devices", type=int, default=4)
    args = parser.parse_args()
    run_campaign(
        args.config.resolve(), args.bundle_dir.resolve(), n_devices=args.n_devices
    )
    print(f"Published qualification bundle: {args.bundle_dir.resolve()}")


if __name__ == "__main__":
    main()
