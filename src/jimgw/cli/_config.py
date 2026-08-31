"""Pydantic config models for the jim-run CLI pipeline.

All models are JAX-free so that ``jim-run --help`` starts in milliseconds.
Heavy imports (JAX, equinox, ripplegw) are deferred to the builder functions
called only after config validation.

Design intent: users specify *what* (prior bounds, waveform, sampler settings).
The CLI figures out *how* (transforms, parameter conversions, consistency checks).
"""

import hashlib
import json
import math
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union, cast

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    PrivateAttr,
    RootModel,
    ValidationInfo,
    field_validator,
    model_validator,
)

from jimgw.cli._utils import (
    CARTESIAN_SPIN_PARAMS,
    DETECTOR_SKY_PARAMS,
    EQUATORIAL_SKY_PARAMS,
    J_FRAME_SPIN_PARAMS,
    SUPPORTED_DETECTORS,
)

# SamplerConfig is safe to import here — samplers/config.py only uses numpy.
from jimgw.samplers.config import SamplerConfig

# ---------------------------------------------------------------------------
# Data section
# ---------------------------------------------------------------------------


def _expanded_detector_names(name: str) -> tuple[str, ...]:
    return ("ET1", "ET2", "ET3") if name == "ET" else (name,)


def _detector_file_key(
    name: str,
    values: dict[str, Any],
    *,
    allow_family: bool = True,
) -> Optional[str]:
    """Return the configured key used by one concrete detector."""

    if name in values:
        return name
    if allow_family and name.startswith("ET") and "ET" in values:
        return "ET"
    return None


class _DataBase(BaseModel):
    model_config = {"extra": "forbid"}
    detectors: list[str]
    trigger_time: float

    @field_validator("detectors")
    @classmethod
    def _check_detectors(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("data.detectors must be a non-empty list")
        unknown = [d for d in v if d not in SUPPORTED_DETECTORS]
        if unknown:
            raise ValueError(
                f"Unknown detector name(s): {unknown}. "
                f"Supported: {sorted(SUPPORTED_DETECTORS)}"
            )
        if len(v) != len(set(v)):
            duplicates = [d for d in set(v) if v.count(d) > 1]
            raise ValueError(f"Duplicate detector name(s): {duplicates}")
        return v


class GWOSCDataConfig(_DataBase):
    """Fetch strain and PSD from GWOSC."""

    type: Literal["gwosc"] = "gwosc"
    duration: float = Field(gt=0.0)
    post_trigger_duration: float = 2.0
    psd_duration: float = Field(gt=0.0)


class InjectionDataConfig(_DataBase):
    """Synthetic injection into design-sensitivity noise."""

    type: Literal["injection"] = "injection"
    duration: float = Field(gt=0.0)
    sampling_frequency: float = Field(gt=0.0)
    injection_parameters: dict[str, float]
    zero_noise: bool = False
    psd_files: dict[str, Path] = Field(default_factory=dict)
    psd_is_asd: dict[str, bool] = Field(default_factory=dict)
    waveform_chunk_size: int = Field(default=262_144, ge=1, strict=True)


class FileDataConfig(_DataBase):
    """Load pre-saved strain and PSD from local files (useful for CI/offline use).

    Supported strain formats: ``.npz``, ``.gwf`` / ``.gwf.gz``, ``.hdf5`` / ``.h5``,
    ``.csv``.  PSD files must be ``.npz`` archives.

    For frame (``.gwf``) and HDF5 files a channel name is required.  Provide
    ``strain_channels`` to map detector names to channel strings; if omitted,
    common LIGO/Virgo preset channel names are tried automatically for GWF files.
    """

    type: Literal["file"] = "file"
    strain_files: dict[str, Path]  # detector_name -> strain file path
    psd_files: dict[str, Path]  # detector_name -> .npz with 'values', 'frequencies'
    strain_channels: dict[str, str] = Field(
        default_factory=dict
    )  # detector_name -> channel (e.g. "H1:GDS-CALIB_STRAIN")
    psd_is_asd: dict[str, bool] = Field(
        default_factory=dict
    )  # detector_name -> True when the psd_file contains ASD values (Hz^{-1/2})

    @model_validator(mode="after")
    def _check_all_detectors_have_files(self) -> "FileDataConfig":
        missing_strain = [
            detector
            for requested in self.detectors
            for detector in _expanded_detector_names(requested)
            if _detector_file_key(
                detector,
                self.strain_files,
                allow_family=False,
            )
            is None
        ]
        missing_psd = [
            detector
            for requested in self.detectors
            for detector in _expanded_detector_names(requested)
            if _detector_file_key(detector, self.psd_files) is None
        ]
        if missing_strain:
            raise ValueError(f"strain_files missing for: {missing_strain}")
        if missing_psd:
            raise ValueError(f"psd_files missing for: {missing_psd}")
        return self


DataConfig = Annotated[
    Union[GWOSCDataConfig, InjectionDataConfig, FileDataConfig],
    Discriminator("type"),
]


def xg_implementation_sha256() -> str:
    """Hash the installed Jim Python source used by the XG likelihood path."""

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = source_path.relative_to(package_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def xg_runtime_environment_sha256() -> str:
    """Hash runtime versions that affect waveform and compression numerics."""

    import jax

    package_versions = {}
    for package_name in ("jax", "jaxlib", "numpy", "scipy", "ripplegw", "equinox"):
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = "missing"
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions,
        "float_precision": "float64" if jax.config.jax_enable_x64 else "float32",
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def xg_source_revision() -> str:
    """Return the installed Jim distribution revision recorded in receipts."""

    try:
        return f"jimgw-{version('jimgw')}"
    except PackageNotFoundError:
        return "jimgw-uninstalled"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _xg_input_files_sha256(data: DataConfig) -> dict[str, str]:
    """Hash every local strain and sensitivity input used by an XG run."""

    if isinstance(data, GWOSCDataConfig):
        raise ValueError(  # noqa: TRY004 - valid config type, unsupported substrate
            "XG likelihoods require immutable local data and sensitivity files; "
            "GWOSC downloads are not a qualified XG substrate"
        )

    files: dict[str, Path] = {}
    for requested in data.detectors:
        for detector in _expanded_detector_names(requested):
            if isinstance(data, FileDataConfig):
                strain_key = _detector_file_key(
                    detector,
                    data.strain_files,
                    allow_family=False,
                )
                if strain_key is None:
                    raise ValueError(f"XG strain file is missing for {detector}")
                files[f"strain:{detector}"] = data.strain_files[strain_key]

            psd_key = _detector_file_key(detector, data.psd_files)
            if psd_key is None:
                raise ValueError(
                    f"XG sensitivity file is missing for {detector}; built-in or "
                    "downloaded sensitivity curves are not qualified"
                )
            files[f"psd:{detector}"] = data.psd_files[psd_key]

    try:
        return {name: _file_sha256(path) for name, path in sorted(files.items())}
    except OSError as error:
        raise ValueError(f"cannot hash XG input file: {error.filename}") from error


# ---------------------------------------------------------------------------
# Waveform section
# ---------------------------------------------------------------------------

Approximant = Literal[
    "TaylorF2",
    "IMRPhenomD",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomHM",
    "IMRPhenomPv2",
    "IMRPhenomXAS",
    "IMRPhenomXAS_NRTidalv3",
    "IMRPhenomXHM",
    "IMRPhenomXP",
    "IMRPhenomXPHM",
    "SineGaussian",
]

_NONQUADRUPOLE_PHASE_APPROXIMANTS = frozenset(
    ("IMRPhenomHM", "IMRPhenomPv2", "IMRPhenomXHM", "IMRPhenomXP", "IMRPhenomXPHM")
)
_DYNAMIC_RESPONSE_APPROXIMANTS = frozenset(
    (
        "TaylorF2",
        "IMRPhenomD",
        "IMRPhenomD_NRTidalv2",
        "IMRPhenomXAS",
        "IMRPhenomXAS_NRTidalv3",
    )
)


class WaveformConfig(BaseModel):
    model_config = {"extra": "forbid"}
    approximant: Approximant
    f_ref: float = Field(default=20.0, gt=0.0)


# ---------------------------------------------------------------------------
# Prior section
#
# Parameter names are the dict keys; each value is an inline table with a
# `type` discriminator plus type-specific bounds.
#
# Example TOML:
#   [prior]
#   M_c  = { type = "uniform",   min = 10.0, max = 80.0 }
#   iota = { type = "sine" }
#   d_L  = { type = "power_law", min = 1.0, max = 2000.0, alpha = 2.0 }
# ---------------------------------------------------------------------------


class UniformSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["uniform"] = "uniform"
    min: float
    max: float

    @model_validator(mode="after")
    def _check_bounds(self) -> "UniformSpec":
        if self.min >= self.max:
            raise ValueError(
                f"uniform prior requires min < max, got min={self.min}, max={self.max}"
            )
        return self


class GaussianSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["gaussian"] = "gaussian"
    loc: float
    scale: float

    @model_validator(mode="after")
    def _check_scale(self) -> "GaussianSpec":
        if self.scale <= 0:
            raise ValueError(
                f"gaussian prior requires scale > 0, got scale={self.scale}"
            )
        return self


class SineSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["sine"] = "sine"


class CosineSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["cosine"] = "cosine"


class PowerLawSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["power_law"] = "power_law"
    min: float
    max: float
    alpha: float

    @model_validator(mode="after")
    def _check_bounds(self) -> "PowerLawSpec":
        if self.min >= self.max:
            raise ValueError(
                f"power_law prior requires min < max, got min={self.min}, max={self.max}"
            )
        return self


class RayleighSpec(BaseModel):
    model_config = {"extra": "forbid"}
    type: Literal["rayleigh"] = "rayleigh"
    scale: float

    @model_validator(mode="after")
    def _check_scale(self) -> "RayleighSpec":
        if self.scale <= 0:
            raise ValueError(
                f"rayleigh prior requires scale > 0, got scale={self.scale}"
            )
        return self


class UniformSphereSpec(BaseModel):
    """Maps to UniformSpherePrior — generates three parameters: {name}_mag/theta/phi."""

    model_config = {"extra": "forbid"}
    type: Literal["uniform_sphere"] = "uniform_sphere"


PriorSpec = Annotated[
    Union[
        UniformSpec,
        GaussianSpec,
        SineSpec,
        CosineSpec,
        PowerLawSpec,
        RayleighSpec,
        UniformSphereSpec,
    ],
    Discriminator("type"),
]


class PriorConfig(RootModel[dict[str, PriorSpec]]):
    """Ordered dict of parameter_name → prior spec.

    Insertion order is preserved (Python 3.7+, TOML spec) and determines
    the parameter ordering passed to CombinePrior.
    """


# ---------------------------------------------------------------------------
# Sampling space section (optional)
# ---------------------------------------------------------------------------


class SamplingConfig(BaseModel):
    """Controls the coordinate system the sampler explores.

    Only relevant when the prior parametrization differs from the preferred
    sampling space. The CLI auto-infers transforms for every other case.
    """

    model_config = {"extra": "forbid"}

    time_frame: str = "detector"
    """Detector name to sample arrival time in (e.g. "H1"). Special values:
    - ``"detector"`` (default): use the first entry in data.detectors.
    - ``"geocentric"``: sample t_c directly without a time sample transform.
    Only used when ``t_c`` is in the prior."""

    sky_frame: Literal["detector", "geocentric"] = "detector"
    """Sampling space for sky position.
    - ``"detector"``: sample in azimuth/zenith (default, better mixing).
    - ``"geocentric"``: sample directly in ra/dec.
    Only used when ``ra``/``dec`` are in the prior."""


# ---------------------------------------------------------------------------
# Likelihood section
# ---------------------------------------------------------------------------
# CLI-level marg configs: structurally equivalent to the library versions but
# JAX-free. Converted to the real configs in the likelihood builder (Stage 6).


class CLIPhaseMargConfig(BaseModel):
    model_config = {"extra": "forbid"}


class CLITimeMargConfig(BaseModel):
    model_config = {"extra": "forbid"}
    tc_range: tuple[float, float] = (-0.1, 0.1)
    upsample_factor: int = Field(default=1, ge=1)
    phasor_block_size: int = Field(default=64, ge=1)
    freeze_response: bool = False
    timing_sigma_s: Optional[float] = Field(default=None, gt=0.0)
    samples_per_timing_sigma: int = Field(default=4, ge=2)
    normalization: Literal["full_grid", "window"] = "full_grid"

    @model_validator(mode="after")
    def _validate_tc_range(self) -> "CLITimeMargConfig":
        lower, upper = self.tc_range
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("tc_range must contain finite increasing bounds")
        if self.timing_sigma_s is not None and not math.isfinite(self.timing_sigma_s):
            raise ValueError("timing_sigma_s must be finite")
        return self


class CLIDistanceMargConfig(BaseModel):
    """Distance marginalization config.

    ``distance_prior`` is a nested prior dict (same syntax as the top-level
    ``[prior]`` section) that is built into a ``Prior`` object by the builder.
    """

    model_config = {"extra": "forbid"}
    distance_prior: PriorConfig
    n_dist_points: int = Field(default=10000, ge=2)
    ref_dist: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check_single_distance_param(self) -> "CLIDistanceMargConfig":
        if len(self.distance_prior.root) != 1:
            raise ValueError(
                "distance_marginalization.distance_prior must contain exactly one parameter"
            )
        return self


class CLIOptimizerRefParams(BaseModel):
    """Find reference parameters automatically via CMA-ES (default)."""

    model_config = {"extra": "forbid"}
    type: Literal["optimizer"] = "optimizer"
    popsize: int = Field(default=500, ge=1)
    n_steps: int = Field(default=1000, ge=1)
    target: Optional[float] = None


class CLIProvidedRefParams(BaseModel):
    """Explicit likelihood-space reference parameters; skips CMA-ES."""

    model_config = {"extra": "forbid"}
    type: Literal["provided"] = "provided"
    values: dict[str, float]


class CLIInjectionRefParams(BaseModel):
    """Use injection parameters (converted to likelihood space) as reference.

    Only valid for injection runs (``data.type = "injection"``).
    """

    model_config = {"extra": "forbid"}
    type: Literal["injection"] = "injection"


HeterodynedRefParams = Annotated[
    Union[CLIOptimizerRefParams, CLIProvidedRefParams, CLIInjectionRefParams],
    Discriminator("type"),
]


_MAX_HETERODYNE_BIN_EDGES = 4_194_304


class CLIHeterodynedConfig(BaseModel):
    """Enable the relative-binning (heterodyne) likelihood.

    When present, ``HeterodynedTransientLikelihoodFD`` is used instead of
    ``TransientLikelihoodFD``.  The ``reference_parameters`` sub-section
    selects how reference parameters are obtained:

    - ``type = "optimizer"`` (default): CMA-ES search using the prior.
    - ``type = "provided"``: explicit likelihood-space values (skips CMA-ES).
    - ``type = "injection"``: use ``data.injection_parameters`` (injection
      runs only).

    Binning is controlled by at most one of ``epsilon`` or ``n_bins``
    (mutually exclusive).  When neither is set, ``epsilon=0.5`` (rad per
    bin) is used as the default.
    """

    model_config = {"extra": "forbid"}
    n_bins: Optional[int] = Field(
        default=None,
        ge=1,
        le=_MAX_HETERODYNE_BIN_EDGES - 1,
        strict=True,
    )
    epsilon: Optional[float] = Field(default=None, gt=0.0, strict=True)
    reference_chunk_size: int = Field(default=262_144, ge=1, strict=True)
    qualification_manifest: Optional[Path] = None
    qualification_manifest_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reference_parameters: HeterodynedRefParams = Field(
        default_factory=CLIOptimizerRefParams
    )

    @model_validator(mode="after")
    def _validate_binning_plan(self) -> "CLIHeterodynedConfig":
        if self.n_bins is not None and self.epsilon is not None:
            raise ValueError("Specify at most one of n_bins and epsilon")
        if self.epsilon is not None and not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite")
        return self


class CLIMultibandedConfig(BaseModel):
    """Enable the multi-banded likelihood.

    When present, ``MultibandedTransientLikelihoodFD`` is used instead of
    ``TransientLikelihoodFD``.

    ``reference_chirp_mass``, ``time_offset``, and ``delta_f_end`` are all
    optional: when omitted they are inferred automatically from the prior
    (``M_c`` minimum and ``t_c`` range respectively).  You only need to set
    them explicitly to override the inferred values.
    """

    model_config = {"extra": "forbid"}
    reference_chirp_mass: Optional[float] = Field(default=None, gt=0.0)
    highest_mode: int = Field(default=2, ge=1)
    accuracy_factor: float = Field(default=5.0, gt=0.0)
    time_offset: Optional[float] = Field(default=None, ge=0.0)
    delta_f_end: Optional[float] = Field(default=None, gt=0.0)
    max_banding_frequency: Optional[float] = Field(default=None, gt=0.0)
    min_banding_duration: float = Field(default=0.0, ge=0.0)


XGQualificationKind = Literal[
    "xg-validation-corpus",
    "xg-clock-validation",
    "xg-independent-response",
    "xg-orbital-validation",
    "xg-compression-validation",
]


XGCaseRole = Literal[
    "coverage",
    "phase-derivative",
    "near-merger",
    "response",
    "orbital",
    "compression",
]


class XGCaseParameters(BaseModel):
    """Scientific design coordinates required for auditable XG coverage."""

    model_config = {"extra": "allow", "strict": True}
    detectors: list[str] = Field(min_length=1)
    f_min: float = Field(gt=0.0)
    f_max: float = Field(gt=0.0)
    network_snr: float = Field(gt=0.0)
    sidereal_epoch_index: int = Field(ge=0, strict=True)
    ra: float
    dec: float
    psi: float
    duration_s: float = Field(gt=0.0)
    prior_extreme: bool
    detector_null: bool
    case_role: XGCaseRole

    @model_validator(mode="after")
    def _validate_design_coordinates(self) -> "XGCaseParameters":
        numeric_values = (
            self.f_min,
            self.f_max,
            self.network_snr,
            self.ra,
            self.dec,
            self.psi,
            self.duration_s,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("XG case coordinates must be finite")
        if self.f_min >= self.f_max:
            raise ValueError("XG cases require f_min < f_max")
        if any(not detector for detector in self.detectors):
            raise ValueError("XG case detector names must be non-empty")
        return self


class XGCaseRecord(BaseModel):
    """One immutable design point in a qualification corpus."""

    model_config = {"extra": "forbid", "strict": True}
    case_id: str = Field(min_length=1)
    parameters: XGCaseParameters


class XGCaseDataset(BaseModel):
    """Typed case dataset referenced by a qualification summary."""

    model_config = {"extra": "forbid", "strict": True}
    schema_version: Literal[1]
    artifact_kind: Literal["xg-case-dataset"]
    qualification_kind: XGQualificationKind
    cases: list[XGCaseRecord] = Field(min_length=1)


class XGResultRecord(BaseModel):
    """One successful raw result with auditable scalar metrics."""

    model_config = {"extra": "forbid", "strict": True}
    case_id: str = Field(min_length=1)
    passed: Literal[True]
    metrics: dict[str, float] = Field(min_length=1)

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, metrics: dict[str, float]) -> dict[str, float]:
        invalid = [
            name
            for name, value in metrics.items()
            if not math.isfinite(value) or value < 0.0
        ]
        if invalid:
            raise ValueError(f"raw result metrics are invalid for {sorted(invalid)}")
        return metrics


class XGRawResults(BaseModel):
    """Typed per-case outputs referenced by a qualification summary."""

    model_config = {"extra": "forbid", "strict": True}
    schema_version: Literal[1]
    artifact_kind: Literal["xg-raw-results"]
    qualification_kind: XGQualificationKind
    results: list[XGResultRecord] = Field(min_length=1)


class XGReceiptEvidence(BaseModel):
    """Immutable raw evidence bound to one qualification summary."""

    model_config = {"extra": "forbid", "strict": True}
    case_count: int = Field(ge=1, strict=True)
    case_dataset_file: str = Field(min_length=1)
    case_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_results_file: str = Field(min_length=1)
    raw_results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_source_file: str = Field(min_length=1)
    generator_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def evidence_kind(self) -> XGQualificationKind:
        """Return the concrete receipt discriminator from the subclass."""

        return cast(XGQualificationKind, self.model_dump()["artifact_kind"])


class XGValidationCorpusReceipt(XGReceiptEvidence):
    """Coverage receipt for the frozen deterministic validation corpus."""

    model_config = {"extra": "forbid", "strict": True}
    schema_version: Literal[1]
    artifact_kind: Literal["xg-validation-corpus"]
    generator: str = Field(min_length=1)
    case_count: int = Field(ge=1, strict=True)
    detectors: list[str] = Field(min_length=1)
    f_min: float = Field(gt=0.0)
    f_max: float = Field(gt=0.0)
    max_network_snr: float = Field(gt=0.0)
    sidereal_epoch_count: int = Field(ge=2, strict=True)
    includes_detector_nulls: Literal[True]
    includes_prior_extremes: Literal[True]
    passed: Literal[True]

    @model_validator(mode="after")
    def _validate_corpus_bounds(self) -> "XGValidationCorpusReceipt":
        values = (self.f_min, self.f_max, self.max_network_snr)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("validation corpus bounds must be finite")
        if self.f_min >= self.f_max:
            raise ValueError("validation corpus requires f_min < f_max")
        return self


class XGClockValidationReceipt(XGReceiptEvidence):
    """Independent phase-derivative and near-merger clock receipt."""

    model_config = {"extra": "forbid", "strict": True}
    schema_version: Literal[1]
    artifact_kind: Literal["xg-clock-validation"]
    implementation_name: str = Field(min_length=1)
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase_derivative_case_count: int = Field(ge=1, strict=True)
    near_merger_case_count: int = Field(ge=1, strict=True)
    max_abs_timing_error_s: float = Field(ge=0.0)
    timing_error_budget_s: float = Field(gt=0.0)
    max_component_delta_log_l: float = Field(ge=0.0, le=0.01)
    post_cutoff_nonnegative: Literal[True]
    post_cutoff_monotonic: Literal[True]
    passed: Literal[True]

    @model_validator(mode="after")
    def _validate_clock_budget(self) -> "XGClockValidationReceipt":
        values = (
            self.max_abs_timing_error_s,
            self.timing_error_budget_s,
            self.max_component_delta_log_l,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("clock validation metrics must be finite")
        if self.max_abs_timing_error_s > self.timing_error_budget_s:
            raise ValueError("clock validation exceeds its timing error budget")
        if self.case_count != (
            self.phase_derivative_case_count + self.near_merger_case_count
        ):
            raise ValueError(
                "clock case_count must equal phase-derivative plus near-merger cases"
            )
        return self


class XGIndependentResponseReceipt(XGReceiptEvidence):
    """Non-SPA, independently implemented detector-response receipt."""

    model_config = {"extra": "forbid", "strict": True}
    schema_version: Literal[1]
    artifact_kind: Literal["xg-independent-response"]
    oracle_kind: Literal[
        "retarded-worldline-round-trip",
        "converged-segmented-time-domain",
    ]
    implementation_name: str = Field(min_length=1)
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=1, strict=True)
    includes_dynamic_delay: bool
    includes_finite_arm: bool
    sample_rate_converged: Literal[True]
    interpolation_converged: Literal[True]
    max_numerical_delta_log_l: float = Field(ge=0.0, le=0.01)
    max_component_delta_log_l: float = Field(ge=0.0, le=0.01)
    max_combined_delta_log_l: float = Field(ge=0.0, le=0.05)
    passed: Literal[True]

    @model_validator(mode="after")
    def _validate_response_metrics(self) -> "XGIndependentResponseReceipt":
        values = (
            self.max_numerical_delta_log_l,
            self.max_component_delta_log_l,
            self.max_combined_delta_log_l,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("independent-response metrics must be finite")
        return self


class XGOrbitalValidationReceipt(XGReceiptEvidence):
    """Profiled full-ephemeris orbital-omission receipt."""

    model_config = {"extra": "forbid", "strict": True}
    schema_version: Literal[1]
    artifact_kind: Literal["xg-orbital-validation"]
    case_count: int = Field(ge=1, strict=True)
    full_ephemeris: Literal[True]
    geocentric_detector_frame_convention: Literal[True]
    constant_delay_removed: Literal[True]
    constant_velocity_removed: Literal[True]
    full_parameter_profiled: Literal[True]
    max_profiled_delta_log_l: float = Field(ge=0.0, le=0.01)
    max_projected_bias_sigma: float = Field(ge=0.0)
    projected_bias_budget_sigma: float = Field(gt=0.0, le=0.1)
    passed: Literal[True]

    @model_validator(mode="after")
    def _validate_bias_budget(self) -> "XGOrbitalValidationReceipt":
        values = (self.max_projected_bias_sigma, self.projected_bias_budget_sigma)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("orbital validation metrics must be finite")
        if self.max_projected_bias_sigma > self.projected_bias_budget_sigma:
            raise ValueError("orbital validation exceeds its projected-bias budget")
        return self


class XGCompressionValidationReceipt(XGReceiptEvidence):
    """Dense-oracle receipt for one immutable compressed bin/time plan."""

    model_config = {"extra": "forbid", "strict": True}
    schema_version: Literal[1]
    artifact_kind: Literal["xg-compression-validation"]
    case_count: int = Field(ge=1, strict=True)
    bin_edges_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dense_standard_likelihood_oracle: Literal[True]
    direct_time_quadrature_converged: Literal[True]
    max_component_delta_log_l: float = Field(ge=0.0, le=0.01)
    max_combined_delta_log_l: float = Field(ge=0.0, le=0.05)
    max_frozen_response_delta_log_l: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.01,
    )
    timing_sigma_s: Optional[float] = Field(default=None, gt=0.0)
    passed: Literal[True]

    @model_validator(mode="after")
    def _validate_compression_metrics(self) -> "XGCompressionValidationReceipt":
        values = (
            self.max_component_delta_log_l,
            self.max_combined_delta_log_l,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("compression validation metrics must be finite")
        optional_values = (
            self.max_frozen_response_delta_log_l,
            self.timing_sigma_s,
        )
        if any(
            value is not None and not math.isfinite(value) for value in optional_values
        ):
            raise ValueError("optional compression validation metrics must be finite")
        return self


class XGQualificationManifest(BaseModel):
    """Immutable evidence record for a compressed XG likelihood plan."""

    model_config = {"extra": "forbid", "frozen": True, "strict": True}
    schema_version: Literal[1]
    source_revision: str = Field(min_length=7)
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_files_sha256: dict[str, str] = Field(min_length=1)
    detector_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bin_edges_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_corpus_file: str = Field(min_length=1)
    validation_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clock_validation_file: str = Field(min_length=1)
    clock_validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    independent_response_file: str = Field(min_length=1)
    independent_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    orbital_validation_file: str = Field(min_length=1)
    orbital_validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compression_validation_file: str = Field(min_length=1)
    compression_validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    waveform_approximant: Approximant
    detectors: list[str] = Field(min_length=1)
    f_min: float = Field(gt=0.0)
    f_max: float = Field(gt=0.0)
    n_bins: int = Field(ge=1, strict=True)
    time_dependent_response: bool
    finite_arm_response: bool
    max_network_snr: float = Field(gt=0.0)
    max_component_delta_log_l: float = Field(ge=0.0, le=0.01)
    max_combined_delta_log_l: float = Field(ge=0.0, le=0.05)
    max_frozen_response_delta_log_l: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.01,
    )
    timing_sigma_s: Optional[float] = Field(default=None, gt=0.0)
    float_precision: Literal["float64"] = "float64"

    @model_validator(mode="after")
    def _validate_finite_values(self) -> "XGQualificationManifest":
        numeric_values = (
            self.f_min,
            self.f_max,
            self.max_network_snr,
            self.max_component_delta_log_l,
            self.max_combined_delta_log_l,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("qualification manifest values must be finite")
        if self.f_min >= self.f_max:
            raise ValueError("qualification manifest requires f_min < f_max")
        if self.max_frozen_response_delta_log_l is not None and not math.isfinite(
            self.max_frozen_response_delta_log_l
        ):
            raise ValueError("max_frozen_response_delta_log_l must be finite")
        if self.timing_sigma_s is not None and not math.isfinite(self.timing_sigma_s):
            raise ValueError("qualification timing_sigma_s must be finite")
        invalid_input_digests = [
            name
            for name, digest in self.input_files_sha256.items()
            if len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ]
        if invalid_input_digests:
            raise ValueError(
                "qualification input file hashes are invalid for "
                f"{sorted(invalid_input_digests)}"
            )
        return self


class LikelihoodConfig(BaseModel):
    model_config = {"extra": "forbid"}
    f_min: float = Field(gt=0.0)
    f_max: float = Field(gt=0.0)
    fixed_parameters: dict[str, float] = Field(default_factory=dict)
    time_dependent_response: bool = False
    finite_arm_response: bool = False
    phase_marginalization: bool = False
    time_marginalization: Optional[CLITimeMargConfig] = None
    distance_marginalization: Optional[CLIDistanceMargConfig] = None
    heterodyne: Optional[CLIHeterodynedConfig] = None
    multiband: Optional[CLIMultibandedConfig] = None

    @model_validator(mode="after")
    def _validate_marginalization_conflicts(self) -> "LikelihoodConfig":
        if self.heterodyne is not None and self.multiband is not None:
            raise ValueError("heterodyne and multiband cannot both be set")
        if self.heterodyne is not None and self.distance_marginalization is not None:
            raise ValueError(
                "distance_marginalization cannot be used with heterodyne likelihood"
            )
        if (
            self.heterodyne is not None
            and (
                self.time_marginalization is not None
                or self.time_dependent_response
                or self.finite_arm_response
            )
            and isinstance(
                self.heterodyne.reference_parameters,
                CLIOptimizerRefParams,
            )
        ):
            raise ValueError(
                "XG or time-marginalized heterodyne likelihood requires fixed "
                "reference_parameters of type 'provided' or 'injection'"
            )
        if (
            (self.time_dependent_response or self.finite_arm_response)
            and self.heterodyne is not None
            and self.heterodyne.n_bins is None
        ):
            raise ValueError(
                "XG heterodyne likelihood requires an explicit, prequalified n_bins"
            )
        if (
            self.time_dependent_response
            and self.heterodyne is not None
            and self.time_marginalization is not None
            and not self.time_marginalization.freeze_response
        ):
            raise ValueError(
                "time-dependent response with direct-sum time marginalization "
                "requires time_marginalization.freeze_response = true"
            )
        if (
            self.time_dependent_response
            and self.heterodyne is not None
            and self.time_marginalization is not None
            and not (
                self.time_marginalization.tc_range[0]
                < 0.0
                < self.time_marginalization.tc_range[1]
            )
        ):
            raise ValueError(
                "frozen time-dependent response is evaluated at t_c = 0, which "
                "must lie inside time_marginalization.tc_range"
            )
        if (
            self.time_dependent_response
            and self.time_marginalization is not None
            and self.heterodyne is None
        ):
            raise ValueError(
                "time-dependent response cannot use dense FFT time "
                "marginalization; sample t_c or select heterodyne"
            )
        if (
            self.time_dependent_response
            and self.heterodyne is not None
            and self.time_marginalization is not None
            and self.time_marginalization.timing_sigma_s is None
        ):
            raise ValueError(
                "time-dependent response with time marginalization requires "
                "time_marginalization.timing_sigma_s from the frozen network "
                "timing-information envelope"
            )
        if (
            self.time_dependent_response
            and self.heterodyne is not None
            and self.time_marginalization is not None
            and self.time_marginalization.normalization != "window"
        ):
            raise ValueError(
                "time-dependent response requires time_marginalization."
                "normalization = 'window' so tc_range is the explicit "
                "uniform-prior support"
            )
        if self.heterodyne is None and self.time_marginalization is not None:
            heterodyne_only_options = []
            if self.time_marginalization.freeze_response:
                heterodyne_only_options.append("freeze_response")
            if self.time_marginalization.phasor_block_size != 64:
                heterodyne_only_options.append("phasor_block_size")
            if self.time_marginalization.timing_sigma_s is not None:
                heterodyne_only_options.append("timing_sigma_s")
            if self.time_marginalization.samples_per_timing_sigma != 4:
                heterodyne_only_options.append("samples_per_timing_sigma")
            if self.time_marginalization.normalization != "full_grid":
                heterodyne_only_options.append("normalization")
            if heterodyne_only_options:
                raise ValueError(
                    "heterodyne-only time_marginalization options require the "
                    "heterodyne likelihood: " + ", ".join(heterodyne_only_options)
                )
        if self.multiband is not None:
            if self.time_dependent_response or self.finite_arm_response:
                raise ValueError(
                    "XG detector responses are not qualified with the multiband "
                    "likelihood; use dense explicit t_c or a qualified heterodyne path"
                )
            if self.time_marginalization is not None:
                raise ValueError(
                    "time_marginalization cannot be used with multiband likelihood"
                )
            if self.distance_marginalization is not None:
                raise ValueError(
                    "distance_marginalization cannot be used with multiband likelihood"
                )
            if self.phase_marginalization:
                raise ValueError(
                    "phase_marginalization cannot be used with multiband likelihood"
                )
        return self


# ---------------------------------------------------------------------------
# Output section
# ---------------------------------------------------------------------------


class OutputConfig(BaseModel):
    model_config = {"extra": "forbid"}
    dir: Path
    save_corner: bool = False
    n_samples: int = Field(
        default=10000, ge=0, description="Number of posterior samples to save. 0 = all."
    )
    overwrite: bool = False
    corner_parameters: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    model_config = {"extra": "forbid"}

    _verified_xg_manifest: Optional[XGQualificationManifest] = PrivateAttr(default=None)
    _verified_xg_manifest_path: Optional[Path] = PrivateAttr(default=None)
    _verified_xg_manifest_sha256: Optional[str] = PrivateAttr(default=None)
    _verified_xg_artifact_hashes: dict[Path, str] = PrivateAttr(default_factory=dict)

    seed: int = 0
    data: DataConfig
    waveform: WaveformConfig
    prior: PriorConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    likelihood: LikelihoodConfig
    sampler: SamplerConfig
    output: OutputConfig

    @property
    def verified_xg_manifest(self) -> Optional[XGQualificationManifest]:
        """Return evidence verified against this complete pipeline contract."""

        if self._verified_xg_manifest is None:
            return None
        return self._verified_xg_manifest.model_copy(deep=True)

    def _issue_verified_xg_plan(self) -> Any:
        """Issue the core capability only from a still-matching pipeline."""

        manifest = self._verified_xg_manifest
        manifest_path = self._verified_xg_manifest_path
        manifest_digest = self._verified_xg_manifest_sha256
        if manifest is None or manifest_path is None or manifest_digest is None:
            raise ValueError("this pipeline has no verified XG qualification")
        try:
            current_manifest_bytes = manifest_path.read_bytes()
            current_manifest = XGQualificationManifest.model_validate_json(
                current_manifest_bytes
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                "the verified XG manifest is no longer readable"
            ) from error
        if (
            hashlib.sha256(current_manifest_bytes).hexdigest() != manifest_digest
            or current_manifest != manifest
        ):
            raise ValueError("the verified XG manifest changed after validation")
        changed_artifacts = []
        for artifact_path, expected_digest in self._verified_xg_artifact_hashes.items():
            try:
                actual_digest = _file_sha256(artifact_path)
            except OSError:
                changed_artifacts.append(str(artifact_path))
                continue
            if actual_digest != expected_digest:
                changed_artifacts.append(str(artifact_path))
        if changed_artifacts:
            raise ValueError(
                "XG qualification artifacts changed after validation: "
                f"{sorted(changed_artifacts)}"
            )
        if self.xg_analysis_contract_sha256() != manifest.analysis_contract_sha256:
            raise ValueError("the pipeline changed after XG qualification")
        if self.xg_input_files_sha256() != manifest.input_files_sha256:
            raise ValueError("the XG inputs changed after qualification")

        from jimgw.core.single_event.likelihood import (
            _XG_PLAN_AUTHORITY,
            _VerifiedXGPlan,
        )

        return _VerifiedXGPlan(manifest.bin_edges_sha256, _XG_PLAN_AUTHORITY)

    def xg_analysis_contract_sha256(self) -> str:
        """Hash the normalized scientific and execution plan for qualification."""

        likelihood = self.likelihood.model_dump(mode="json")
        heterodyne = likelihood.get("heterodyne")
        if heterodyne is not None:
            heterodyne.pop("qualification_manifest", None)
            heterodyne.pop("qualification_manifest_sha256", None)
        payload = {
            "contract_schema_version": 1,
            "float_precision": "float64",
            "seed": self.seed,
            "data": self.data.model_dump(mode="json"),
            "waveform": self.waveform.model_dump(mode="json"),
            "prior": self.prior.model_dump(mode="json"),
            "sampling": self.sampling.model_dump(mode="json"),
            "likelihood": likelihood,
            "sampler": self.sampler.model_dump(mode="json"),
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def xg_input_files_sha256(self) -> dict[str, str]:
        """Return hashes for the immutable local inputs in this run."""

        return _xg_input_files_sha256(self.data)

    @model_validator(mode="after")
    def _validate_time_marginalization_parameters(self) -> "PipelineConfig":
        if self.likelihood.time_marginalization is None:
            return self
        sampled_time = {"t_c", "t_det"} & set(self.prior.root)
        if sampled_time:
            raise ValueError(
                "time_marginalization removes sampled coalescence time; remove "
                f"{sorted(sampled_time)} from [prior] and use tc_range as its prior"
            )
        if "t_c" in self.likelihood.fixed_parameters:
            raise ValueError(
                "time_marginalization cannot be combined with fixed_parameters.t_c"
            )
        return self

    @model_validator(mode="after")
    def _validate_phase_marginalization_waveform(self) -> "PipelineConfig":
        if (
            self.likelihood.phase_marginalization
            and self.waveform.approximant in _NONQUADRUPOLE_PHASE_APPROXIMANTS
        ):
            raise ValueError(
                "phase_marginalization is not analytically valid for the selected "
                f"waveform approximant {self.waveform.approximant!r}"
            )
        return self

    @model_validator(mode="after")
    def _validate_xg_waveform_scope(self) -> "PipelineConfig":
        approximant = self.waveform.approximant
        if (
            self.likelihood.time_dependent_response
            and approximant not in _DYNAMIC_RESPONSE_APPROXIMANTS
        ):
            raise ValueError(
                "time-dependent response requires a supported dominant-mode CBC "
                "approximant"
            )
        if self.likelihood.finite_arm_response and approximant == "SineGaussian":
            raise ValueError(
                "finite-arm XG response requires a frequency-domain CBC approximant"
            )
        return self

    @model_validator(mode="after")
    def _validate_xg_qualification_manifest(
        self,
        info: ValidationInfo,
    ) -> "PipelineConfig":
        heterodyne = self.likelihood.heterodyne
        if heterodyne is None:
            return self
        uses_xg_compression = (
            self.likelihood.time_dependent_response
            or self.likelihood.finite_arm_response
        )
        manifest_path = heterodyne.qualification_manifest
        expected_digest = heterodyne.qualification_manifest_sha256
        if not uses_xg_compression:
            if manifest_path is not None or expected_digest is not None:
                raise ValueError(
                    "qualification_manifest is only valid for an XG heterodyne path"
                )
            return self

        if info.context and info.context.get("prepare_xg_qualification", False):
            self.xg_input_files_sha256()
            return self
        if manifest_path is None or expected_digest is None:
            raise ValueError(
                "XG heterodyne likelihood requires qualification_manifest and "
                "qualification_manifest_sha256 from the frozen deterministic "
                "likelihood campaign"
            )
        manifest_path = manifest_path.resolve()
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"cannot read XG qualification manifest {manifest_path}"
            ) from error
        actual_digest = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                "XG qualification manifest SHA-256 does not match the configured digest"
            )
        try:
            manifest = XGQualificationManifest.model_validate(
                json.loads(manifest_bytes)
            )
        except ValueError as error:
            raise ValueError("invalid XG qualification manifest") from error
        input_file_hashes = self.xg_input_files_sha256()

        verified_artifact_hashes: dict[Path, str] = {}
        artifact_specs = (
            (
                "validation_corpus",
                manifest.validation_corpus_file,
                manifest.validation_corpus_sha256,
                XGValidationCorpusReceipt,
            ),
            (
                "clock_validation",
                manifest.clock_validation_file,
                manifest.clock_validation_sha256,
                XGClockValidationReceipt,
            ),
            (
                "independent_response",
                manifest.independent_response_file,
                manifest.independent_response_sha256,
                XGIndependentResponseReceipt,
            ),
            (
                "orbital_validation",
                manifest.orbital_validation_file,
                manifest.orbital_validation_sha256,
                XGOrbitalValidationReceipt,
            ),
            (
                "compression_validation",
                manifest.compression_validation_file,
                manifest.compression_validation_sha256,
                XGCompressionValidationReceipt,
            ),
        )

        def verify_artifact_file(
            artifact_name: str,
            artifact_digest: str,
        ) -> Path:
            relative_path = Path(artifact_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    "XG qualification artifact paths must stay inside the "
                    "manifest directory"
                )
            artifact_path = manifest_path.parent / relative_path
            try:
                actual_artifact_digest = _file_sha256(artifact_path)
            except OSError as error:
                raise ValueError(
                    f"cannot read XG qualification artifact {artifact_path}"
                ) from error
            if actual_artifact_digest != artifact_digest:
                raise ValueError(
                    "XG qualification artifact SHA-256 does not match for "
                    f"{artifact_name}"
                )
            verified_artifact_hashes[artifact_path.resolve()] = artifact_digest
            return artifact_path

        parsed_receipts: dict[str, BaseModel] = {}
        for (
            receipt_name,
            artifact_name,
            artifact_digest,
            receipt_model,
        ) in artifact_specs:
            artifact_path = verify_artifact_file(artifact_name, artifact_digest)
            try:
                parsed_receipts[receipt_name] = receipt_model.model_validate_json(
                    artifact_path.read_bytes()
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"invalid XG qualification artifact {artifact_name}"
                ) from error

        corpus_receipt = parsed_receipts["validation_corpus"]
        clock_receipt = parsed_receipts["clock_validation"]
        response_receipt = parsed_receipts["independent_response"]
        orbital_receipt = parsed_receipts["orbital_validation"]
        compression_receipt = parsed_receipts["compression_validation"]
        assert isinstance(corpus_receipt, XGValidationCorpusReceipt)
        assert isinstance(clock_receipt, XGClockValidationReceipt)
        assert isinstance(response_receipt, XGIndependentResponseReceipt)
        assert isinstance(orbital_receipt, XGOrbitalValidationReceipt)
        assert isinstance(compression_receipt, XGCompressionValidationReceipt)

        case_datasets_by_receipt: dict[str, XGCaseDataset] = {}
        raw_results_by_receipt: dict[str, XGRawResults] = {}
        for receipt_name, receipt in parsed_receipts.items():
            assert isinstance(receipt, XGReceiptEvidence)
            case_dataset_path = verify_artifact_file(
                receipt.case_dataset_file,
                receipt.case_dataset_sha256,
            )
            raw_results_path = verify_artifact_file(
                receipt.raw_results_file,
                receipt.raw_results_sha256,
            )
            verify_artifact_file(
                receipt.generator_source_file,
                receipt.generator_source_sha256,
            )
            try:
                case_dataset = XGCaseDataset.model_validate_json(
                    case_dataset_path.read_bytes()
                )
                raw_results = XGRawResults.model_validate_json(
                    raw_results_path.read_bytes()
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"invalid raw evidence for {receipt.evidence_kind}"
                ) from error
            if (
                case_dataset.qualification_kind != receipt.evidence_kind
                or raw_results.qualification_kind != receipt.evidence_kind
            ):
                raise ValueError(
                    f"raw evidence kind does not match {receipt.evidence_kind}"
                )
            case_ids = [case.case_id for case in case_dataset.cases]
            result_ids = [result.case_id for result in raw_results.results]
            if len(case_ids) != len(set(case_ids)):
                raise ValueError(
                    f"case dataset has duplicate IDs for {receipt.evidence_kind}"
                )
            if len(result_ids) != len(set(result_ids)):
                raise ValueError(
                    f"raw results have duplicate IDs for {receipt.evidence_kind}"
                )
            if len(case_ids) != receipt.case_count:
                raise ValueError(
                    f"case_count does not match raw evidence for {receipt.evidence_kind}"
                )
            if set(result_ids) != set(case_ids):
                raise ValueError(
                    f"raw result case IDs do not match {receipt.evidence_kind}"
                )
            case_datasets_by_receipt[receipt_name] = case_dataset
            raw_results_by_receipt[receipt_name] = raw_results

        expected_case_roles: dict[str, set[XGCaseRole]] = {
            "validation_corpus": {"coverage"},
            "clock_validation": {"phase-derivative", "near-merger"},
            "independent_response": {"response"},
            "orbital_validation": {"orbital"},
            "compression_validation": {"compression"},
        }
        for receipt_name, expected_roles in expected_case_roles.items():
            actual_roles = {
                case.parameters.case_role
                for case in case_datasets_by_receipt[receipt_name].cases
            }
            if actual_roles != expected_roles:
                raise ValueError(
                    f"raw case roles do not match qualification kind for {receipt_name}"
                )

        clock_cases = case_datasets_by_receipt["clock_validation"].cases
        phase_derivative_count = sum(
            case.parameters.case_role == "phase-derivative" for case in clock_cases
        )
        near_merger_count = sum(
            case.parameters.case_role == "near-merger" for case in clock_cases
        )
        if (
            phase_derivative_count != clock_receipt.phase_derivative_case_count
            or near_merger_count != clock_receipt.near_merger_case_count
        ):
            raise ValueError("clock case roles do not match receipt case counts")

        def aggregate_metric(
            receipt_name: str,
            metric_name: str,
            *,
            reduction: str = "max",
        ) -> float:
            try:
                values = [
                    result.metrics[metric_name]
                    for result in raw_results_by_receipt[receipt_name].results
                ]
            except KeyError as error:
                raise ValueError(
                    f"raw results for {receipt_name} omit {metric_name}"
                ) from error
            return max(values) if reduction == "max" else min(values)

        raw_aggregate_checks = (
            (
                "clock_validation",
                "abs_timing_error_s",
                clock_receipt.max_abs_timing_error_s,
                "max",
            ),
            (
                "clock_validation",
                "component_delta_log_l",
                clock_receipt.max_component_delta_log_l,
                "max",
            ),
            (
                "independent_response",
                "numerical_delta_log_l",
                response_receipt.max_numerical_delta_log_l,
                "max",
            ),
            (
                "independent_response",
                "component_delta_log_l",
                response_receipt.max_component_delta_log_l,
                "max",
            ),
            (
                "independent_response",
                "combined_delta_log_l",
                response_receipt.max_combined_delta_log_l,
                "max",
            ),
            (
                "orbital_validation",
                "profiled_delta_log_l",
                orbital_receipt.max_profiled_delta_log_l,
                "max",
            ),
            (
                "orbital_validation",
                "projected_bias_sigma",
                orbital_receipt.max_projected_bias_sigma,
                "max",
            ),
            (
                "compression_validation",
                "component_delta_log_l",
                compression_receipt.max_component_delta_log_l,
                "max",
            ),
            (
                "compression_validation",
                "combined_delta_log_l",
                compression_receipt.max_combined_delta_log_l,
                "max",
            ),
        )
        raw_aggregate_mismatches = [
            f"{receipt_name}.{metric_name}"
            for receipt_name, metric_name, expected, reduction in raw_aggregate_checks
            if not math.isclose(
                aggregate_metric(
                    receipt_name,
                    metric_name,
                    reduction=reduction,
                ),
                expected,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ]
        if compression_receipt.max_frozen_response_delta_log_l is not None and not (
            math.isclose(
                aggregate_metric(
                    "compression_validation",
                    "frozen_response_delta_log_l",
                ),
                compression_receipt.max_frozen_response_delta_log_l,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raw_aggregate_mismatches.append(
                "compression_validation.frozen_response_delta_log_l"
            )
        if compression_receipt.timing_sigma_s is not None and not math.isclose(
            aggregate_metric(
                "compression_validation",
                "timing_sigma_s",
                reduction="min",
            ),
            compression_receipt.timing_sigma_s,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raw_aggregate_mismatches.append("compression_validation.timing_sigma_s")

        for receipt_name, case_dataset in case_datasets_by_receipt.items():
            cases = case_dataset.cases
            case_detectors = {
                detector for case in cases for detector in case.parameters.detectors
            }
            case_f_min = min(case.parameters.f_min for case in cases)
            case_f_max = max(case.parameters.f_max for case in cases)
            case_max_network_snr = max(case.parameters.network_snr for case in cases)
            if case_detectors != set(manifest.detectors):
                raw_aggregate_mismatches.append(f"{receipt_name}.raw_detectors")
            if not math.isclose(case_f_min, manifest.f_min) or not math.isclose(
                case_f_max,
                manifest.f_max,
            ):
                raw_aggregate_mismatches.append(f"{receipt_name}.raw_frequency_band")
            if not math.isclose(
                case_max_network_snr,
                manifest.max_network_snr,
            ):
                raw_aggregate_mismatches.append(f"{receipt_name}.raw_max_network_snr")

        corpus_cases = case_datasets_by_receipt["validation_corpus"].cases
        corpus_detectors = {
            detector for case in corpus_cases for detector in case.parameters.detectors
        }
        corpus_f_min = min(case.parameters.f_min for case in corpus_cases)
        corpus_f_max = max(case.parameters.f_max for case in corpus_cases)
        corpus_max_network_snr = max(
            case.parameters.network_snr for case in corpus_cases
        )
        corpus_sidereal_epoch_count = len(
            {case.parameters.sidereal_epoch_index for case in corpus_cases}
        )
        if corpus_detectors != set(corpus_receipt.detectors):
            raw_aggregate_mismatches.append("validation_corpus.raw_detectors")
        if not math.isclose(corpus_f_min, corpus_receipt.f_min) or not math.isclose(
            corpus_f_max,
            corpus_receipt.f_max,
        ):
            raw_aggregate_mismatches.append("validation_corpus.raw_frequency_band")
        if not math.isclose(
            corpus_max_network_snr,
            corpus_receipt.max_network_snr,
        ):
            raw_aggregate_mismatches.append("validation_corpus.raw_max_network_snr")
        if corpus_sidereal_epoch_count != corpus_receipt.sidereal_epoch_count:
            raw_aggregate_mismatches.append("validation_corpus.raw_sidereal_epochs")
        if not any(case.parameters.detector_null for case in corpus_cases):
            raw_aggregate_mismatches.append("validation_corpus.raw_detector_nulls")
        if not any(case.parameters.prior_extreme for case in corpus_cases):
            raw_aggregate_mismatches.append("validation_corpus.raw_prior_extremes")

        expected_values = {
            "waveform_approximant": self.waveform.approximant,
            "detectors": self.data.detectors,
            "n_bins": heterodyne.n_bins,
            "time_dependent_response": self.likelihood.time_dependent_response,
            "finite_arm_response": self.likelihood.finite_arm_response,
            "source_revision": xg_source_revision(),
            "implementation_sha256": xg_implementation_sha256(),
            "runtime_environment_sha256": xg_runtime_environment_sha256(),
            "analysis_contract_sha256": self.xg_analysis_contract_sha256(),
            "input_files_sha256": input_file_hashes,
        }
        mismatches = raw_aggregate_mismatches + [
            name
            for name, expected in expected_values.items()
            if getattr(manifest, name) != expected
        ]
        if corpus_receipt.detectors != manifest.detectors:
            mismatches.append("validation_corpus.detectors")
        if not math.isclose(corpus_receipt.f_min, manifest.f_min) or not math.isclose(
            corpus_receipt.f_max,
            manifest.f_max,
        ):
            mismatches.append("validation_corpus.frequency_band")
        if not math.isclose(
            corpus_receipt.max_network_snr,
            manifest.max_network_snr,
        ):
            mismatches.append("validation_corpus.max_network_snr")
        if clock_receipt.implementation_sha256 == manifest.implementation_sha256:
            mismatches.append("clock_validation.independent_implementation")
        if clock_receipt.implementation_sha256 != clock_receipt.generator_source_sha256:
            mismatches.append("clock_validation.generator_source")
        if response_receipt.implementation_sha256 == manifest.implementation_sha256:
            mismatches.append("independent_response.implementation")
        if (
            response_receipt.implementation_sha256
            != response_receipt.generator_source_sha256
        ):
            mismatches.append("independent_response.generator_source")
        if (
            manifest.time_dependent_response
            and not response_receipt.includes_dynamic_delay
        ):
            mismatches.append("independent_response.dynamic_delay")
        if manifest.finite_arm_response and not response_receipt.includes_finite_arm:
            mismatches.append("independent_response.finite_arm")
        component_metrics = (
            clock_receipt.max_component_delta_log_l,
            response_receipt.max_component_delta_log_l,
            orbital_receipt.max_profiled_delta_log_l,
            compression_receipt.max_component_delta_log_l,
        )
        if any(
            value > manifest.max_component_delta_log_l for value in component_metrics
        ):
            mismatches.append("max_component_delta_log_l")
        if (
            response_receipt.max_combined_delta_log_l
            > manifest.max_combined_delta_log_l
            or compression_receipt.max_combined_delta_log_l
            > manifest.max_combined_delta_log_l
        ):
            mismatches.append("max_combined_delta_log_l")
        if compression_receipt.bin_edges_sha256 != manifest.bin_edges_sha256:
            mismatches.append("compression_validation.bin_edges_sha256")
        if compression_receipt.max_frozen_response_delta_log_l is not None and (
            manifest.max_frozen_response_delta_log_l is None
            or compression_receipt.max_frozen_response_delta_log_l
            > manifest.max_frozen_response_delta_log_l
        ):
            mismatches.append("compression_validation.frozen_response")
        if compression_receipt.timing_sigma_s != manifest.timing_sigma_s:
            mismatches.append("compression_validation.timing_sigma_s")
        if not math.isclose(manifest.f_min, self.likelihood.f_min) or not math.isclose(
            manifest.f_max,
            self.likelihood.f_max,
        ):
            mismatches.append("frequency_band")
        time_config = self.likelihood.time_marginalization
        if self.likelihood.time_dependent_response and time_config is not None:
            if compression_receipt.max_frozen_response_delta_log_l is None:
                mismatches.append(
                    "compression_validation.max_frozen_response_delta_log_l"
                )
            if compression_receipt.timing_sigma_s is None:
                mismatches.append("compression_validation.timing_sigma_s")
            if manifest.max_frozen_response_delta_log_l is None:
                mismatches.append("max_frozen_response_delta_log_l")
            if manifest.timing_sigma_s != time_config.timing_sigma_s:
                mismatches.append("timing_sigma_s")
        if mismatches:
            raise ValueError(
                "XG qualification manifest does not match the requested run: "
                + ", ".join(mismatches)
            )
        self._verified_xg_manifest = manifest
        self._verified_xg_manifest_path = manifest_path
        self._verified_xg_manifest_sha256 = expected_digest
        self._verified_xg_artifact_hashes = verified_artifact_hashes
        return self

    @model_validator(mode="after")
    def _validate_injection_frame_consistency(self) -> "PipelineConfig":
        if not isinstance(self.data, InjectionDataConfig):
            return self
        inj = set(self.data.injection_parameters)
        if "t_det" in inj and self.sampling.time_frame == "geocentric":
            raise ValueError(
                "injection_parameters uses 't_det' but [sampling].time_frame = "
                "'geocentric'. Use 't_c' in injection_parameters, or set "
                "time_frame to 'detector' or a specific detector name."
            )
        if inj & DETECTOR_SKY_PARAMS and self.sampling.sky_frame != "detector":
            raise ValueError(
                "injection_parameters uses detector-frame sky position "
                "('azimuth'/'zenith') but [sampling].sky_frame != 'detector'. "
                "Use 'ra'/'dec' in injection_parameters, or set sky_frame = 'detector'."
            )
        if inj & J_FRAME_SPIN_PARAMS and "phase_c" not in inj:
            raise ValueError(
                "injection_parameters uses J-frame spin angles but 'phase_c' is missing. "
                "SpinAnglesToCartesianSpinTransform requires 'phase_c' as a conditioning "
                "parameter. Add 'phase_c' to injection_parameters."
            )
        return self

    @model_validator(mode="after")
    def _validate_spin_parametrization(self) -> "PipelineConfig":
        prior_keys = frozenset(self.prior.root.keys())

        has_j_frame = bool(prior_keys & J_FRAME_SPIN_PARAMS)
        has_sphere_spin = any(
            isinstance(self.prior.root.get(label), UniformSphereSpec)
            or all(f"{label}_{s}" in prior_keys for s in ("mag", "theta", "phi"))
            for label in ("s1", "s2")
        )
        has_cartesian_spin = bool(prior_keys & CARTESIAN_SPIN_PARAMS)

        if sum([has_j_frame, has_sphere_spin, has_cartesian_spin]) > 1:
            raise ValueError(
                "Spin parametrizations are mutually exclusive. "
                "Found more than one of: J-frame angles, spherical per-spin, "
                f"Cartesian/aligned spins. Prior parameters: {sorted(prior_keys)}"
            )

        if has_j_frame:
            if "iota" in prior_keys:
                raise ValueError(
                    "J-frame spin angles produce 'iota' — 'iota' must not also appear in [prior]."
                )
            missing_j = J_FRAME_SPIN_PARAMS - prior_keys
            if missing_j:
                raise ValueError(
                    "J-frame spin parametrization requires all 7 parameters; "
                    f"missing from [prior]: {sorted(missing_j)}"
                )
            missing_mass = {"M_c", "q"} - prior_keys
            if missing_mass:
                raise ValueError(
                    f"SpinAnglesToCartesianSpinTransform requires {missing_mass} in [prior] "
                    "as conditioning parameters."
                )

        return self

    @model_validator(mode="after")
    def _validate_sky_time_parametrization(self) -> "PipelineConfig":
        prior_keys = frozenset(self.prior.root.keys())

        equatorial_present = prior_keys & EQUATORIAL_SKY_PARAMS
        if equatorial_present and not (EQUATORIAL_SKY_PARAMS <= prior_keys):
            missing = EQUATORIAL_SKY_PARAMS - prior_keys
            raise ValueError(
                f"[prior] must contain both 'ra' and 'dec' together; "
                f"missing: {sorted(missing)}"
            )
        detector_present = prior_keys & DETECTOR_SKY_PARAMS
        if detector_present and not (DETECTOR_SKY_PARAMS <= prior_keys):
            missing = DETECTOR_SKY_PARAMS - prior_keys
            raise ValueError(
                f"[prior] must contain both 'azimuth' and 'zenith' together; "
                f"missing: {sorted(missing)}"
            )

        has_equatorial_sky = bool(equatorial_present)
        has_detector_sky = bool(detector_present)
        if has_equatorial_sky and has_detector_sky:
            raise ValueError(
                "Sky parametrizations are mutually exclusive: "
                "cannot have both ra/dec and azimuth/zenith in [prior]."
            )
        if has_detector_sky and self.sampling.sky_frame == "geocentric":
            raise ValueError(
                "azimuth/zenith are in [prior] but sky_frame='geocentric' requests "
                "equatorial-sky sampling. Either remove azimuth/zenith from [prior] "
                "and use ra/dec, or set sky_frame='detector'."
            )

        has_geocentric_time = "t_c" in prior_keys
        has_detector_time = "t_det" in prior_keys
        if has_geocentric_time and has_detector_time:
            raise ValueError(
                "Time parametrizations are mutually exclusive: "
                "cannot have both t_c and t_det in [prior]."
            )
        if has_detector_time and self.sampling.time_frame == "geocentric":
            raise ValueError(
                "t_det is in [prior] but time_frame='geocentric' requests geocentric-time "
                "sampling. Either remove t_det from [prior] and use t_c, or set "
                "time_frame to 'detector' or a specific detector name."
            )

        return self

    @model_validator(mode="after")
    def _validate_sampling_ifo_consistency(self) -> "PipelineConfig":
        if (
            self.sampling.time_frame not in ("detector", "geocentric")
            and self.sampling.time_frame not in self.data.detectors
        ):
            raise ValueError(
                f"[sampling] time_frame={self.sampling.time_frame!r} is not in "
                f"data.detectors {self.data.detectors}"
            )

        prior_keys = frozenset(self.prior.root.keys())
        has_sky_params = bool(
            prior_keys & (EQUATORIAL_SKY_PARAMS | DETECTOR_SKY_PARAMS)
        )
        if (
            has_sky_params
            and self.sampling.sky_frame == "detector"
            and len(self.data.detectors) < 2
        ):
            raise ValueError(
                "Sky position sampling in detector frame requires at least 2 detectors; "
                f"got {self.data.detectors}"
            )

        return self

    @model_validator(mode="after")
    def _validate_ns_aw_constraints(self) -> "PipelineConfig":
        if self.sampler.type != "blackjax-ns-aw":
            return self

        prior_keys = frozenset(self.prior.root.keys())

        if "t_det" in prior_keys and not isinstance(
            self.prior.root["t_det"], UniformSpec
        ):
            raise ValueError(
                "NS AW sampler: the 't_det' prior must be 'uniform' for automatic "
                "conversion to 't_c'. Either use a uniform t_det prior or replace "
                "'t_det' with 't_c' in [prior]."
            )

        if (
            "t_c" in prior_keys
            and self.sampling.time_frame != "geocentric"
            and not isinstance(self.prior.root["t_c"], UniformSpec)
        ):
            raise ValueError(
                "NS AW sampler: the 't_c' prior must be 'uniform' for automatic "
                "conversion to 't_det'. Either use a uniform t_c prior, set "
                "[sampling] time_frame = 'geocentric' to sample t_c directly, or "
                "replace 't_c' with 't_det' in [prior]."
            )

        return self

    @model_validator(mode="after")
    def _validate_heterodyne_ref_data_type(self) -> "PipelineConfig":
        if (
            self.likelihood.heterodyne is not None
            and isinstance(
                self.likelihood.heterodyne.reference_parameters, CLIInjectionRefParams
            )
            and not isinstance(self.data, InjectionDataConfig)
        ):
            raise ValueError(
                "heterodyne.reference_parameters.type = 'injection' requires "
                "data.type = 'injection'"
            )
        return self
