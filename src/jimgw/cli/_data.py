import logging
from typing import Optional, assert_never

import jax
import numpy as np

from jimgw.cli._config import (
    DataConfig,
    FileDataConfig,
    GWOSCDataConfig,
    InjectionDataConfig,
)
from jimgw.cli._transforms import to_likelihood_space
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import (
    GroundBased2G,
    asd_file_dict,
    get_detector_preset,
)

logger = logging.getLogger(__name__)


def build_data(
    data_cfg: DataConfig,
    f_min: float,
    f_max: float,
    waveform=None,
    time_frame: str = "detector",
    time_dependent_response: bool = False,
    finite_arm_response: bool = False,
    orbital_motion_response: bool = False,
    orbital_reference_time: Optional[float] = None,
    orbital_validity_s: Optional[tuple[float, float]] = None,
    orbital_acceleration_over_c: Optional[tuple[float, float, float]] = None,
    orbital_jerk_over_c: Optional[tuple[float, float, float]] = None,
    seed: int = 0,
    input_provenance_sha256: Optional[dict[str, str]] = None,
) -> list[GroundBased2G]:
    """Construct a list of detectors populated with strain data and PSDs.

    For injection runs, *waveform* is required and is used to inject the signal.
    """
    preset = get_detector_preset()

    ifos: list[GroundBased2G] = []
    for name in data_cfg.detectors:
        val = preset[name]
        if isinstance(val, list):
            ifos.extend(val)  # ty: ignore[invalid-argument-type]
        else:
            ifos.append(val)

    if time_dependent_response and not getattr(
        waveform, "time_dependent_response", False
    ):
        raise ValueError(
            "time-dependent detector response requires a waveform emission-time cache"
        )
    for ifo in ifos:
        if finite_arm_response and ifo.arm_length_m is None:
            raise ValueError(
                f"finite-arm response for detector {ifo.name!r} requires "
                "arm-length metadata"
            )
        ifo.time_dependent_response = time_dependent_response
        ifo.finite_arm_response = finite_arm_response
        ifo.configure_orbital_motion_response(
            enabled=orbital_motion_response,
            reference_time=orbital_reference_time,
            validity_s=orbital_validity_s,
            acceleration_over_c=orbital_acceleration_over_c,
            jerk_over_c=orbital_jerk_over_c,
        )

    if isinstance(data_cfg, GWOSCDataConfig):
        _load_gwosc(ifos, data_cfg)
    elif isinstance(data_cfg, InjectionDataConfig):
        _load_injection(
            ifos,
            data_cfg,
            waveform,
            f_min=f_min,
            f_max=f_max,
            time_frame=time_frame,
            seed=seed,
            require_configured_psd=(
                time_dependent_response
                or finite_arm_response
                or orbital_motion_response
            ),
        )
    elif isinstance(data_cfg, FileDataConfig):
        _load_files(ifos, data_cfg, f_min=f_min, f_max=f_max)
    else:
        assert_never(data_cfg)

    for ifo in ifos:
        _validate_frequency_coverage(
            ifo.data.frequencies,
            f_min,
            f_max,
            f"{ifo.name} strain",
        )
        _validate_frequency_coverage(
            ifo.psd.frequencies,
            f_min,
            f_max,
            f"{ifo.name} PSD",
        )
        _validate_psd_values(ifo.psd, f_min, f_max, f"{ifo.name} PSD")
        logger.info(
            "%s: %.1f s @ %.0f Hz, PSD shape %s",
            ifo.name,
            ifo.data.duration,
            ifo.data.sampling_frequency,
            ifo.psd.values.shape,
        )

    provenance = (
        tuple(sorted(input_provenance_sha256.items()))
        if input_provenance_sha256 is not None
        else None
    )
    for ifo in ifos:
        ifo.input_provenance_sha256 = provenance

    return ifos


def _validate_frequency_coverage(
    frequencies,
    f_min: float,
    f_max: float,
    label: str,
) -> None:
    """Reject sensitivity or strain inputs that require extrapolation."""

    if len(frequencies) < 2:
        raise ValueError(f"{label} has fewer than two frequency samples")
    available_min = float(frequencies[0])
    available_max = float(frequencies[-1])
    if available_min > f_min or available_max < f_max:
        raise ValueError(
            f"{label} covers [{available_min}, {available_max}] Hz, not the "
            f"requested [{f_min}, {f_max}] Hz band"
        )


def _validate_psd_values(
    psd: PowerSpectrum,
    f_min: float,
    f_max: float,
    label: str,
) -> None:
    """Require finite, strictly positive PSD values over the analysis band."""

    frequencies = np.asarray(jax.device_get(psd.frequencies))
    values = np.asarray(jax.device_get(psd.values))
    in_band = (frequencies >= f_min) & (frequencies <= f_max)
    if not np.any(in_band):
        raise ValueError(f"{label} has no samples in the requested analysis band")
    invalid = ~np.isfinite(values[in_band]) | (values[in_band] <= 0.0)
    if np.any(invalid):
        raise ValueError(
            f"{label} must contain finite, strictly positive values over the "
            "analysis band"
        )


def _load_gwosc(ifos: list[GroundBased2G], cfg: GWOSCDataConfig) -> None:
    # Analysis segment: [trigger - duration + post_trigger, trigger + post_trigger]
    end = cfg.trigger_time + cfg.post_trigger_duration
    start = end - cfg.duration
    # PSD segment: immediately before the analysis segment
    psd_end = start
    psd_start = psd_end - cfg.psd_duration

    for ifo in ifos:
        logger.info("Fetching %s strain [%.1f, %.1f]", ifo.name, start, end)
        strain = Data.from_gwosc(ifo.name, start, end)
        ifo.set_data(strain)

        logger.info("Fetching %s PSD data [%.1f, %.1f]", ifo.name, psd_start, psd_end)
        psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
        nperseg = round(strain.duration * strain.sampling_frequency)
        ifo.set_psd(psd_data.to_psd(nperseg=nperseg))


def _load_injection(
    ifos: list[GroundBased2G],
    cfg: InjectionDataConfig,
    waveform,
    *,
    f_min: float,
    f_max: float,
    time_frame: str = "detector",
    seed: int = 0,
    require_configured_psd: bool = False,
) -> None:
    parameters = to_likelihood_space(
        cfg.injection_parameters,
        waveform_f_ref=waveform.f_ref,
        trigger_time=cfg.trigger_time,
        ifos=ifos,
        time_frame=time_frame,
    )

    if cfg.sampling_frequency / 2.0 < f_max:
        raise ValueError(
            "injection sampling_frequency has a Nyquist frequency below "
            f"f_max={f_max} Hz"
        )

    root_key = jax.random.key(seed)
    for detector_index, ifo in enumerate(ifos):
        family_name = "ET" if ifo.name.startswith("ET") else ifo.name
        sensitivity_file = cfg.psd_files.get(
            ifo.name,
            cfg.psd_files.get(family_name),
        )
        if sensitivity_file is not None:
            is_asd = cfg.psd_is_asd.get(
                ifo.name,
                cfg.psd_is_asd.get(family_name, False),
            )
            logger.info("Loading configured sensitivity for %s", ifo.name)
            ifo.set_psd(PowerSpectrum.from_file(str(sensitivity_file), is_asd=is_asd))
        elif require_configured_psd:
            raise ValueError(
                f"XG injection requires data.psd_files.{family_name}; built-in "
                "or downloaded sensitivity curves are not qualified"
            )
        elif ifo.name in asd_file_dict:
            logger.info("Loading design PSD for %s", ifo.name)
            ifo.load_and_set_psd()
        else:
            raise ValueError(
                f"No default ASD for detector {ifo.name!r}. Set "
                f"data.psd_files.{family_name} to a versioned PSD or ASD file. "
                f"Detectors with built-in defaults: {sorted(asd_file_dict)}."
            )

        _validate_frequency_coverage(
            ifo.psd.frequencies,
            f_min,
            f_max,
            f"{ifo.name} configured PSD",
        )
        _validate_psd_values(
            ifo.psd,
            f_min,
            f_max,
            f"{ifo.name} configured PSD",
        )

        logger.info("Injecting signal into %s", ifo.name)
        ifo.inject_signal(
            duration=cfg.duration,
            sampling_frequency=cfg.sampling_frequency,
            trigger_time=cfg.trigger_time,
            waveform_model=waveform,
            parameters=parameters,
            f_min=f_min,
            f_max=f_max,
            zero_noise=cfg.zero_noise,
            rng_key=jax.random.fold_in(root_key, detector_index),
            waveform_chunk_size=cfg.waveform_chunk_size,
        )


def _load_files(
    ifos: list[GroundBased2G],
    cfg: FileDataConfig,
    *,
    f_min: float,
    f_max: float,
) -> None:
    for ifo in ifos:
        family_name = "ET" if ifo.name.startswith("ET") else ifo.name
        if ifo.name not in cfg.strain_files:
            raise ValueError(
                f"strain_files requires a distinct {ifo.name} entry; detector "
                "strain cannot use a family-level fallback"
            )
        strain_path = cfg.strain_files[ifo.name]
        psd_path = cfg.psd_files.get(ifo.name)
        if psd_path is None:
            psd_path = cfg.psd_files[family_name]
        channel = cfg.strain_channels.get(
            ifo.name,
            cfg.strain_channels.get(family_name),
        )
        is_asd = cfg.psd_is_asd.get(
            ifo.name,
            cfg.psd_is_asd.get(family_name, False),
        )

        logger.info("Loading %s strain from %s", ifo.name, strain_path)
        strain = Data.from_file(str(strain_path), channel=channel)
        _validate_frequency_coverage(
            strain.frequencies,
            f_min,
            f_max,
            f"{ifo.name} configured strain",
        )

        logger.info("Loading %s PSD from %s", ifo.name, psd_path)
        psd = PowerSpectrum.from_file(str(psd_path), is_asd=is_asd)
        _validate_frequency_coverage(
            psd.frequencies,
            f_min,
            f_max,
            f"{ifo.name} configured PSD",
        )
        _validate_psd_values(
            psd,
            f_min,
            f_max,
            f"{ifo.name} configured PSD",
        )
        ifo.set_data(strain)
        ifo.set_psd(psd)
