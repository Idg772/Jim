import logging
import math
import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import requests
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Complex, Float, Key, jaxtyped

from jimgw.core.constants import (
    C_SI,
    DEG_TO_RAD,
    EARTH_SEMI_MAJOR_AXIS,
    EARTH_SEMI_MINOR_AXIS,
)
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.heterodyne_noise import (
    INDEXED_NOISE_ALGORITHM,
    compiled_indexed_noise,
)
from jimgw.core.single_event.native_storage import (
    allocate_native_strain,
    release_mapped_pages,
)
from jimgw.core.single_event.polarization import Polarization
from jimgw.core.single_event.time_dependent_response import (
    earth_orbital_curvature_delay,
    emission_gmst,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.utils import complex_inner_product, inner_product
from jimgw.typing import FloatLike, FloatScalar

logger = logging.getLogger(__name__)

# TODO: Need to expand this list. Currently it is only O3.
asd_file_dict = {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",
}


def finite_arm_transfer(
    frequency: FloatLike,
    direction_cosine: FloatLike,
    arm_length_m: Optional[float],
) -> Complex:
    """Return the round-trip transfer function of one interferometer arm.

    ``direction_cosine`` is the cosine between the direction towards the
    source and the arm, i.e. ``omega . arm`` when ``omega`` points from the
    geocenter towards the source (the wave propagates along ``-omega``). This
    is the convention of Essick, Vitale & Evans (2017) Eq. (5) and matches a
    direct photon-path integral of the round trip referenced to the
    beam-splitter arrival time.

    Passing ``propagation . arm`` (the opposite sign) evaluates the response
    of the antipodal sky position.  That error has been made twice in this
    project (a caller in 2026-09-02, the qualification oracle until
    2026-09-03), so the convention is pinned against a from-scratch
    photon-path integral in ``tests/unit/core/single_event/test_detector.py``
    and ``tests/unit/benchmarks/test_xg_response_oracle.py``; any
    implementation that disagrees with those tests is wrong, whatever it
    agrees with otherwise. Inputs follow JAX broadcasting rules, so a
    scalar or per-frequency direction cosine is supported. The normalized
    :func:`jax.numpy.sinc` convention makes the transfer tend to one as
    ``frequency * arm_length_m / C_SI`` tends to zero.

    Args:
        frequency: Frequency in Hz.
        direction_cosine: Cosine between the source direction and the arm.
        arm_length_m: Physical arm length in metres.

    Returns:
        The complex, frequency-dependent arm transfer function.

    Raises:
        ValueError: If the arm length is absent, non-finite, or non-positive.
    """
    if arm_length_m is None:
        raise ValueError("finite-arm response requires arm_length_m metadata")
    if not math.isfinite(arm_length_m) or arm_length_m <= 0:
        raise ValueError("arm_length_m must be finite and positive")

    x = jnp.asarray(frequency) * arm_length_m / C_SI
    mu = jnp.asarray(direction_cosine)

    phase_out = -jnp.pi * x * (1.0 - mu)
    phase_back = -jnp.pi * x * (3.0 - mu)
    phasor_out = jax.lax.complex(jnp.cos(phase_out), jnp.sin(phase_out))
    phasor_back = jax.lax.complex(jnp.cos(phase_back), jnp.sin(phase_back))
    return 0.5 * (
        phasor_out * jnp.sinc(x * (1.0 - mu)) + phasor_back * jnp.sinc(x * (1.0 + mu))
    )


def _same_grid(left, right) -> bool:
    """Compare two frequency grids without moving host arrays to a device."""
    if left is right:
        return True
    if left.shape != right.shape:
        return False
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.ctypes.data == right.ctypes.data
            and left.strides == right.strides
        ):
            return True
        return all(
            np.array_equal(
                np.asarray(left[start : start + 65536]),
                np.asarray(right[start : start + 65536]),
            )
            for start in range(0, len(left), 65536)
        )
    return bool(jnp.array_equal(left, right))


class Detector(ABC):
    """Base class for all detectors.

    Attributes:
        name (str): Name of the detector.
        data (Data): Detector data object.
        psd (PowerSpectrum): Power spectral density object.
        frequency_bounds (tuple[float, float]): Lower and upper frequency bounds.
    """

    name: str

    # NOTE: for some detectors (e.g. LISA, ET) data could be a list of Data
    # objects so this might be worth revisiting
    data: Data
    psd: PowerSpectrum
    input_provenance_sha256: Optional[tuple[tuple[str, str], ...]] = None

    frequency_bounds: tuple[float, float] = (0.0, jnp.inf)

    _sliced_frequencies: Float[Array, " n_sample"] = jnp.array([])
    _sliced_fd_data: Float[Array, " n_sample"] = jnp.array([])
    _sliced_psd: Float[Array, " n_sample"] = jnp.array([])

    @property
    def start_time(self) -> float:
        """GPS start time of the data segment."""
        return self.data.start_time

    @property
    def times(self) -> Float[Array, " n_sample"]:
        return self.data.times

    @property
    def frequencies(self) -> Float[Array, " n_sample"]:
        return self.data.frequencies

    @property
    def duration(self) -> FloatLike:
        return self.data.duration

    @property
    def frequency_mask(self) -> Bool[Array, " n_sample"]:
        f_min, f_max = self.frequency_bounds
        return (f_min <= self.frequencies) & (self.frequencies <= f_max)

    @abstractmethod
    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
        *,
        optimize: bool = True,
        finite_arm: Optional[bool] = None,
        apply_antenna: bool = True,
        include_data_epoch: bool = True,
    ) -> Complex[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the frequency domain.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.
            h_sky (dict[str, Float[Array, "n_sample"]]): Dictionary mapping polarization names
                to frequency-domain waveforms. The keys are polarization names (e.g., 'plus', 'cross')
                and values are complex strain arrays.
            params (dict): Dictionary of source parameters including:
                - ra (Float): Right ascension in radians
                - dec (Float): Declination in radians
                - psi (Float): Polarization angle in radians
                - trigger_time (Float): The trigger time in sec
                - t_c (Float): The difference between peak time and trigger time in sec
                - gmst (Float): The greenwich mean sidereal time at the trigger time in radian
            optimize: Use the real-angle phasor implementation.
            finite_arm: Override the detector's configured finite-arm response
                for this call. None uses the detector default.

        Returns:
            Complex[Array, "n_sample"]: Complex strain measured by the detector in frequency domain.
        """

    @abstractmethod
    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
    ) -> Float[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the time domain.

        Args:
            time: Array of time samples.
            h_sky: Dictionary mapping polarization names to time-domain waveforms.
            params: Dictionary of source parameters.

        Returns:
            Array of detector response in time domain.
        """

    def set_frequency_bounds(
        self, f_min: Optional[float] = None, f_max: Optional[float] = None
    ) -> None:
        """Set the frequency bounds for the detector.
        This also set the sliced frequencies, data and psd.

        Args:
            f_min: Minimum frequency.
            f_max: Maximum frequency.
        """
        bounds = list(self.frequency_bounds)
        if f_min is not None:
            bounds[0] = f_min
        if f_max is not None:
            bounds[1] = f_max
        self.frequency_bounds = (bounds[0], bounds[1])

        # Compute sliced frequencies, data and psd.
        data, freqs_1 = self.data.frequency_slice(*self.frequency_bounds)
        psd, freqs_2 = self.psd.frequency_slice(*self.frequency_bounds)

        assert _same_grid(freqs_1, freqs_2), (
            f"The {self.name} data and PSD must have same frequencies"
        )

        self._sliced_frequencies = freqs_1
        self._sliced_fd_data = data
        self._sliced_psd = psd

    def clear_data_and_psd(self) -> None:
        """Clear the data and PSD of the detector."""
        self.data = Data()
        self.psd = PowerSpectrum()
        self.frequency_bounds = (0.0, jnp.inf)
        self._sliced_frequencies = jnp.array([])
        self._sliced_fd_data = jnp.array([])
        self._sliced_psd = jnp.array([])
        self.optimal_snr = None
        self.match_filtered_snr = None
        self.input_provenance_sha256 = None

    @property
    def sliced_frequencies(self) -> Float[Array, " n_freq"]:
        """Get frequency-domain data slice based on frequency bounds.

        Returns:
            Float[Array, "n_sample"]: Sliced frequency-domain data.
            Float[Array, "n_sample"]: Frequency array.
        """
        return self._sliced_frequencies

    @property
    def sliced_fd_data(self) -> Complex[Array, " n_freq"]:
        """Get frequency-domain data slice based on frequency bounds.

        Returns:
            Complex[Array, "n_freq"]: Sliced frequency-domain data.
        """
        return self._sliced_fd_data

    @property
    def sliced_psd(self) -> Float[Array, " n_freq"]:
        """Get PSD slice based on frequency bounds.

        Returns:
            Float[Array, "n_freq"]: Sliced power spectral density.
        """
        return self._sliced_psd

    def __init__(self):
        if not jax.config.read("jax_enable_x64"):
            raise RuntimeError(
                "Detector requires JAX to run in 64-bit (float64) mode, "
                "but jax_enable_x64 is currently False.\n\n"
                "Please enable float64 before creating any Detector by putting at the very top of your script:\n"
                "    import jax\n"
                "    jax.config.update('jax_enable_x64', True)\n"
                "and then re-run."
            )


class GroundBased2G(Detector):
    """Object representing a ground-based detector.

    Contains information about the location and orientation of the detector on Earth,
    as well as actual strain data and the PSD of the associated noise.

    Attributes:
        name (str): Name of the detector.
        latitude (Float): Latitude of the detector in radians.
        longitude (Float): Longitude of the detector in radians.
        xarm_azimuth (Float): Azimuth of the x-arm in radians.
        yarm_azimuth (Float): Azimuth of the y-arm in radians.
        xarm_tilt (Float): Tilt of the x-arm in radians.
        yarm_tilt (Float): Tilt of the y-arm in radians.
        elevation (Float): Elevation of the detector in meters.
        polarization_mode (list[Polarization]): List of polarization modes (`pc` for plus and cross) to be used in
            computing antenna patterns; in the future, this could be expanded to
            include non-GR modes.
        data (Data): Array of Fourier-domain strain data.
        psd (PowerSpectrum): Power spectral density object.
    """

    polarization_mode: list[Polarization]
    data: Data
    psd: PowerSpectrum

    latitude: float = 0
    longitude: float = 0
    xarm_azimuth: float = 0
    yarm_azimuth: float = 0
    xarm_tilt: float = 0
    yarm_tilt: float = 0
    elevation: float = 0
    arm_length_m: Optional[float] = None
    finite_arm_response: bool = False
    time_dependent_response: bool = False
    orbital_motion_response: bool = False
    orbital_acceleration_over_c: Optional[tuple[float, float, float]] = None
    orbital_jerk_over_c: Optional[tuple[float, float, float]] = None
    orbital_reference_time: Optional[float] = None
    orbital_validity_s: Optional[tuple[float, float]] = None
    optimal_snr: Optional[FloatScalar] = None
    match_filtered_snr: Optional[Complex] = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __init__(
        self,
        name: str,
        latitude: float = 0,
        longitude: float = 0,
        elevation: float = 0,
        xarm_azimuth: float = 0,
        yarm_azimuth: float = 0,
        xarm_tilt: float = 0,
        yarm_tilt: float = 0,
        modes: str = "pc",
        *,
        arm_length_m: Optional[float] = None,
        finite_arm_response: bool = False,
        time_dependent_response: bool = False,
        orbital_motion_response: bool = False,
        orbital_acceleration_over_c: Optional[tuple[float, float, float]] = None,
        orbital_jerk_over_c: Optional[tuple[float, float, float]] = None,
        orbital_reference_time: Optional[float] = None,
        orbital_validity_s: Optional[tuple[float, float]] = None,
    ):
        """Initialize a ground-based detector.

        Args:
            name (str): Name of the detector.
            latitude (float, optional): Latitude of the detector in radians. Defaults to 0.
            longitude (float, optional): Longitude of the detector in radians. Defaults to 0.
            elevation (float, optional): Elevation of the detector in meters. Defaults to 0.
            xarm_azimuth (float, optional): Azimuth of the x-arm in radians. Defaults to 0.
            yarm_azimuth (float, optional): Azimuth of the y-arm in radians. Defaults to 0.
            xarm_tilt (float, optional): Tilt of the x-arm in radians. Defaults to 0.
            yarm_tilt (float, optional): Tilt of the y-arm in radians. Defaults to 0.
            modes (str, optional): Polarization modes. Defaults to "pc".
            arm_length_m (float, optional): Physical arm length in metres. Required
                when using the finite-arm response. Defaults to None.
            finite_arm_response (bool, optional): Use the frequency-dependent
                finite-arm response by default in :meth:`fd_response`. Defaults
                to False, preserving the long-wavelength response.
            time_dependent_response (bool, optional): Evaluate the detector
                orientation and delay at the waveform's frequency-dependent
                emission time. The waveform must provide a ``__tau__`` leaf.
                Defaults to False.
            orbital_motion_response (bool, optional): Add the nonlinear
                Earth-orbital delay at the waveform's emission time. This
                requires ``time_dependent_response`` and both orbital Taylor
                coefficient vectors. Defaults to False.
            orbital_acceleration_over_c (tuple[float, float, float], optional):
                Effective inertial Earth-centre acceleration coefficient divided
                by the speed of light, in ``s^-1``. It may be an
                ephemeris-fitted cubic-surrogate coefficient.
            orbital_jerk_over_c (tuple[float, float, float], optional): Inertial
                Earth-centre cubic coefficient divided by the speed of light,
                in ``s^-2``. It may be fitted over the declared validity window.
            orbital_reference_time (float, optional): GPS epoch at which the
                orbital coefficients were derived. It must equal the analysis
                trigger time when the response is enabled.
            orbital_validity_s (tuple[float, float], optional): Inclusive
                emission-offset interval, in seconds relative to
                ``orbital_reference_time``, qualified for the Taylor model.
        """
        super().__init__()

        self.name = name

        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth
        self.yarm_azimuth = yarm_azimuth
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt
        if arm_length_m is not None and (
            not math.isfinite(arm_length_m) or arm_length_m <= 0
        ):
            raise ValueError("arm_length_m must be finite and positive")
        if finite_arm_response and arm_length_m is None:
            raise ValueError("finite_arm_response=True requires arm_length_m metadata")
        self.arm_length_m = arm_length_m
        self.finite_arm_response = finite_arm_response
        self.time_dependent_response = time_dependent_response
        self.orbital_motion_response = orbital_motion_response
        self.orbital_acceleration_over_c = self._orbital_coefficient_tuple(
            "orbital_acceleration_over_c", orbital_acceleration_over_c
        )
        self.orbital_jerk_over_c = self._orbital_coefficient_tuple(
            "orbital_jerk_over_c", orbital_jerk_over_c
        )
        if orbital_reference_time is not None and not math.isfinite(
            orbital_reference_time
        ):
            raise ValueError("orbital_reference_time must be finite")
        self.orbital_reference_time = orbital_reference_time
        self.orbital_validity_s = self._orbital_validity_tuple(orbital_validity_s)
        self._validate_orbital_motion_configuration()
        self.input_provenance_sha256 = None

        self.polarization_mode = [Polarization(m) for m in modes]
        self.data = Data()
        self.psd = PowerSpectrum()

    @staticmethod
    def _orbital_coefficient_tuple(
        name: str,
        coefficient: Optional[tuple[float, float, float]],
    ) -> Optional[tuple[float, float, float]]:
        """Validate and normalize one inertial orbital Taylor coefficient."""

        if coefficient is None:
            return None
        try:
            array = np.asarray(coefficient, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain three finite values") from error
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain three finite values")
        return float(array[0]), float(array[1]), float(array[2])

    @staticmethod
    def _orbital_validity_tuple(
        validity: Optional[tuple[float, float]],
    ) -> Optional[tuple[float, float]]:
        """Validate and normalize the qualified emission-offset interval."""

        if validity is None:
            return None
        try:
            array = np.asarray(validity, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "orbital_validity_s must contain two ordered finite values"
            ) from error
        if (
            array.shape != (2,)
            or not np.all(np.isfinite(array))
            or not array[0] < 0.0 < array[1]
        ):
            raise ValueError(
                "orbital_validity_s must contain two ordered finite values "
                "bracketing zero"
            )
        return float(array[0]), float(array[1])

    def _validate_orbital_motion_configuration(self) -> None:
        """Fail closed when the optional orbital response is incomplete."""

        acceleration = self._orbital_coefficient_tuple(
            "orbital_acceleration_over_c", self.orbital_acceleration_over_c
        )
        jerk = self._orbital_coefficient_tuple(
            "orbital_jerk_over_c", self.orbital_jerk_over_c
        )
        validity = self._orbital_validity_tuple(self.orbital_validity_s)
        reference_time = self.orbital_reference_time
        if reference_time is not None and not math.isfinite(reference_time):
            raise ValueError("orbital_reference_time must be finite")
        metadata_present = (
            acceleration is not None,
            jerk is not None,
            reference_time is not None,
            validity is not None,
        )
        if any(metadata_present) and not all(metadata_present):
            raise ValueError(
                "orbital coefficients, reference time, and validity must be "
                "provided together"
            )
        if not self.orbital_motion_response:
            return
        if not self.time_dependent_response:
            raise ValueError(
                "orbital_motion_response=True requires time_dependent_response=True"
            )
        if not all(metadata_present):
            raise ValueError(
                "orbital_motion_response=True requires orbital coefficients, "
                "reference time, and validity"
            )

    def validate_orbital_motion_for_trigger(self, trigger_time: float) -> None:
        """Validate that orbital coefficients are anchored to ``trigger_time``."""

        self._validate_orbital_motion_configuration()
        if not self.orbital_motion_response:
            return
        if not math.isfinite(trigger_time):
            raise ValueError("trigger_time must be finite for orbital motion response")
        if trigger_time != self.orbital_reference_time:
            raise ValueError(
                "orbital_reference_time must exactly match the analysis trigger_time"
            )

    def configure_orbital_motion_response(
        self,
        *,
        enabled: bool,
        reference_time: Optional[float] = None,
        validity_s: Optional[tuple[float, float]] = None,
        acceleration_over_c: Optional[tuple[float, float, float]] = None,
        jerk_over_c: Optional[tuple[float, float, float]] = None,
    ) -> None:
        """Atomically configure the qualified Earth-orbital response.

        Calling this with ``enabled=False`` and no metadata clears a previous
        configuration.  Complete metadata may be installed while disabled,
        but partial metadata is always rejected.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        acceleration = self._orbital_coefficient_tuple(
            "orbital_acceleration_over_c", acceleration_over_c
        )
        jerk = self._orbital_coefficient_tuple("orbital_jerk_over_c", jerk_over_c)
        validity = self._orbital_validity_tuple(validity_s)
        if reference_time is not None and not math.isfinite(reference_time):
            raise ValueError("orbital_reference_time must be finite")
        metadata_present = (
            acceleration is not None,
            jerk is not None,
            reference_time is not None,
            validity is not None,
        )
        if any(metadata_present) and not all(metadata_present):
            raise ValueError(
                "orbital coefficients, reference time, and validity must be "
                "provided together"
            )
        if enabled and not self.time_dependent_response:
            raise ValueError(
                "orbital_motion_response=True requires time_dependent_response=True"
            )
        if enabled and not all(metadata_present):
            raise ValueError(
                "orbital_motion_response=True requires orbital coefficients, "
                "reference time, and validity"
            )

        self.orbital_motion_response = enabled
        self.orbital_acceleration_over_c = acceleration
        self.orbital_jerk_over_c = jerk
        self.orbital_reference_time = reference_time
        self.orbital_validity_s = validity

    @staticmethod
    def _get_arm(
        lat: float, lon: float, tilt: float, azimuth: float
    ) -> Float[Array, "3"]:
        """Construct detector-arm vectors in geocentric Cartesian coordinates.

        Args:
            lat (Float): Vertex latitude in radians.
            lon (Float): Vertex longitude in radians.
            tilt (Float): Arm tilt in radians.
            azimuth (Float): Arm azimuth in radians.

        Returns:
            Float[Array, "3"]: Detector arm vector in geocentric Cartesian coordinates.
        """
        e_lon = jnp.array([-jnp.sin(lon), jnp.cos(lon), 0])
        e_lat = jnp.array(
            [-jnp.sin(lat) * jnp.cos(lon), -jnp.sin(lat) * jnp.sin(lon), jnp.cos(lat)]
        )
        e_h = jnp.array(
            [jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)]
        )

        return (
            jnp.cos(tilt) * jnp.cos(azimuth) * e_lon
            + jnp.cos(tilt) * jnp.sin(azimuth) * e_lat
            + jnp.sin(tilt) * e_h
        )

    @property
    def arms(self) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
        """Get the detector arm vectors.

        Returns:
            tuple[Float[Array, "3"], Float[Array, "3"]]: A tuple containing:
                - x: X-arm vector in geocentric Cartesian coordinates
                - y: Y-arm vector in geocentric Cartesian coordinates
        """
        x = self._get_arm(
            self.latitude, self.longitude, self.xarm_tilt, self.xarm_azimuth
        )
        y = self._get_arm(
            self.latitude, self.longitude, self.yarm_tilt, self.yarm_azimuth
        )
        return x, y

    @property
    def tensor(self) -> Float[Array, "3 3"]:
        """Get the detector tensor defining the strain measurement.

        For a 2-arm differential-length detector, this is given by:

        $$

        D_{ij} = \\left(x_i x_j - y_i y_j\\right)/2

        $$

        for unit vectors $x$ and $y$ along the x and y arms.

        Returns:
            Float[Array, "3 3"]: The 3x3 detector tensor in geocentric coordinates.
        """
        # TODO: this could easily be generalized for other detector geometries
        arm1, arm2 = self.arms
        return 0.5 * (
            jnp.einsum("i,j->ij", arm1, arm1) - jnp.einsum("i,j->ij", arm2, arm2)
        )

    @property
    def vertex(self) -> Float[Array, "3"]:
        """Detector vertex coordinates in the reference celestial frame.

        Based on arXiv:gr-qc/0008066 Eqs. (B11-B13) except for a typo in the
        definition of the local radius; see Section 2.1 of LIGO-T980044-10.

        Returns:
            Float[Array, "3"]: Detector vertex coordinates in geocentric Cartesian coordinates.
        """
        # get detector and Earth parameters
        lat = self.latitude
        lon = self.longitude
        h = self.elevation
        major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
        # compute vertex location
        r = major**2 * (
            major**2 * jnp.cos(lat) ** 2 + minor**2 * jnp.sin(lat) ** 2
        ) ** (-0.5)
        x = (r + h) * jnp.cos(lat) * jnp.cos(lon)
        y = (r + h) * jnp.cos(lat) * jnp.sin(lon)
        z = ((minor / major) ** 2 * r + h) * jnp.sin(lat)
        return jnp.array([x, y, z])

    def fd_response(
        self,
        frequency: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict[str, Float],
        *,
        optimize: bool = True,
        finite_arm: Optional[bool] = None,
        apply_antenna: bool = True,
        include_data_epoch: bool = True,
    ) -> Complex[Array, " n_sample"]:
        """Modulate the waveform in the sky frame by the detector response in the frequency domain.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.
            h_sky (dict[str, Float[Array, "n_sample"]]): Dictionary mapping polarization names
                to frequency-domain waveforms. Keys are polarization names (e.g., 'plus', 'cross')
                and values are complex strain arrays.
            params (dict[str, Float]): Dictionary of source parameters containing:
                - ra (Float): Right ascension in radians
                - dec (Float): Declination in radians
                - psi (Float): Polarization angle in radians
                - trigger_time (Float): The trigger time in sec
                - t_c (Float): The difference between peak time and trigger time in sec
                - gmst (Float): The greenwich mean sidereal time at the trigger time in radian
            optimize: Use the real-angle phasor implementation.
            finite_arm: Override :attr:`finite_arm_response` for this call. None
                uses the detector default.
            apply_antenna: False returns the plus-polarization carrier with
                the same propagation phase, for reference summaries only.
            include_data_epoch: False omits the common data-epoch phase when
                constructing smooth ratios offline. Physical strain uses True.

        Returns:
            Array: Complex strain measured by the detector in frequency domain, obtained by
                  combining the antenna patterns for each polarization mode.
        """
        self._validate_orbital_motion_configuration()
        ra, dec, psi, gmst = params["ra"], params["dec"], params["psi"], params["gmst"]
        tau = h_sky.get("__tau__")
        if self.time_dependent_response:
            if tau is None:
                raise ValueError(
                    "time-dependent detector response requires a waveform "
                    "with a __tau__ emission-time leaf"
                )
            gmst = emission_gmst(gmst, params["t_c"], tau)
            h_sky = {
                polarization: strain
                for polarization, strain in h_sky.items()
                if polarization != "__tau__"
            }
        elif tau is not None:
            raise ValueError(
                "a waveform with a __tau__ emission-time leaf requires "
                "time_dependent_response=True"
            )

        use_finite_arm = self.finite_arm_response if finite_arm is None else finite_arm
        if use_finite_arm:
            antenna_pattern = self.frequency_dependent_antenna_pattern(
                ra, dec, psi, gmst, frequency
            )
            time_shift = (
                -self._source_projection(ra, dec, psi, gmst, self.vertex)[2] / C_SI
            )
        elif self.time_dependent_response:
            m, n, omega = self._wave_frame(ra, dec, psi, gmst)
            antenna_pattern = {}
            for polarization in self.polarization_mode:
                wave_tensor = polarization.tensor_from_basis(m, n)
                antenna_pattern[polarization.name] = jnp.einsum(
                    "ij,ij...->...", self.tensor, wave_tensor
                )
            time_shift = -jnp.einsum("i...,i->...", omega, self.vertex) / C_SI
        else:
            antenna_pattern = self.antenna_pattern(ra, dec, psi, gmst)
            time_shift = self.delay_from_geocenter(ra, dec, gmst)
        if self.orbital_motion_response:
            trigger_time = params["trigger_time"]
            if isinstance(trigger_time, (int, float, np.integer, np.floating)):
                self.validate_orbital_motion_for_trigger(float(trigger_time))
            emission_offset = params["t_c"] - tau
            acceleration = self.orbital_acceleration_over_c
            jerk = self.orbital_jerk_over_c
            validity = self.orbital_validity_s
            if acceleration is None or jerk is None or validity is None:
                raise RuntimeError("validated orbital response metadata is missing")
            orbital_delay = earth_orbital_curvature_delay(
                ra,
                dec,
                emission_offset,
                acceleration,
                jerk,
            )
            validity_min, validity_max = validity
            valid = (
                (jnp.asarray(trigger_time) == self.orbital_reference_time)
                & (emission_offset >= validity_min)
                & (emission_offset <= validity_max)
            )
            time_shift += jnp.where(valid, orbital_delay, jnp.nan)
        if include_data_epoch:
            time_shift += params["trigger_time"] - self.start_time + params["t_c"]
        else:
            time_shift += params["t_c"]

        if apply_antenna:
            h_detector = jax.tree_util.tree_map(
                lambda h, antenna: h * antenna,
                h_sky,
                antenna_pattern,
            )
            projected_strain = jnp.sum(
                jnp.stack(jax.tree_util.tree_leaves(h_detector)), axis=0
            )
        else:
            projected_strain = h_sky["p"]

        phase_angle = (-2.0 * jnp.pi) * frequency * time_shift
        if optimize:
            # Real-angle phasor: exp(-2πi f Δt) with a complex-typed
            # argument forces XLA's generic complex exp. cos/sin of the real
            # angle is the same rotation without that overhead.
            phase_shift = jax.lax.complex(jnp.cos(phase_angle), jnp.sin(phase_angle))
        else:
            # Diagnostic reference matching the pre-optimization formula.
            phase_shift = jnp.exp(1j * phase_angle)
        return projected_strain * phase_shift

    @staticmethod
    def _wave_frame(
        ra: FloatLike,
        dec: FloatLike,
        psi: FloatLike,
        gmst: FloatLike,
    ) -> tuple[Array, Array, Array]:
        """Construct vector-safe wave-frame basis vectors and source direction."""
        phi = jnp.asarray(ra) - jnp.mod(jnp.asarray(gmst), 2 * jnp.pi)
        theta = jnp.pi / 2 - jnp.asarray(dec)
        phi, theta, psi = jnp.broadcast_arrays(phi, theta, jnp.asarray(psi))

        u = jnp.stack(
            [
                jnp.cos(phi) * jnp.cos(theta),
                jnp.cos(theta) * jnp.sin(phi),
                -jnp.sin(theta),
            ]
        )
        v = jnp.stack([-jnp.sin(phi), jnp.cos(phi), jnp.zeros_like(phi)])
        m = -u * jnp.sin(psi) - v * jnp.cos(psi)
        n = -u * jnp.cos(psi) + v * jnp.sin(psi)
        omega = jnp.stack(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )
        return m, n, omega

    def frequency_dependent_antenna_pattern(
        self,
        ra: FloatLike,
        dec: FloatLike,
        psi: FloatLike,
        gmst: FloatLike,
        frequency: FloatLike,
    ) -> dict[str, Complex]:
        """Compute antenna patterns including the finite-arm transfer.

        ``frequency`` and sky coordinates may be scalars or broadcast-compatible
        arrays. At zero frequency this reduces to :meth:`antenna_pattern`.
        Arm-length metadata is deliberately mandatory so an XG calculation
        cannot silently fall back to the long-wavelength approximation.

        Raises:
            ValueError: If this detector has no valid arm-length metadata.
        """
        if self.arm_length_m is None:
            raise ValueError(
                f"finite-arm response for detector {self.name!r} requires "
                "arm_length_m metadata"
            )

        xarm, yarm = self.arms
        # ``omega`` points from the geocentre towards the source, and the
        # transfer function takes the cosine between that source direction
        # and the arm (the wave propagates along ``-omega``).
        #
        # The arm projection of a polarization tensor built from the wave
        # frame basis (m, n) is a scalar quadratic form in the arm's
        # components along m and n: for D_a = a (x) a / 2,
        # D_a : (m m - n n) = ((a.m)^2 - (a.n)^2) / 2 and
        # D_a : (m n + n m) = (a.m)(a.n).  Evaluating those dot products
        # per frequency keeps the whole response elementwise; materialising
        # the (3, 3, n_freq) tensors and contracting them was the dominant
        # memory traffic of every XG likelihood call.
        x_m, x_n, x_omega = self._source_projection(ra, dec, psi, gmst, xarm)
        y_m, y_n, y_omega = self._source_projection(ra, dec, psi, gmst, yarm)
        x_transfer = finite_arm_transfer(frequency, x_omega, self.arm_length_m)
        y_transfer = finite_arm_transfer(frequency, y_omega, self.arm_length_m)

        antenna_patterns = {}
        for polarization in self.polarization_mode:
            if polarization.name == "p":
                x_projection = 0.5 * (x_m * x_m - x_n * x_n)
                y_projection = 0.5 * (y_m * y_m - y_n * y_n)
            elif polarization.name == "c":
                x_projection = x_m * x_n
                y_projection = y_m * y_n
            else:
                m, n, _ = self._wave_frame(ra, dec, psi, gmst)
                wave_tensor = polarization.tensor_from_basis(m, n)
                xx = 0.5 * jnp.einsum("i,j->ij", xarm, xarm)
                yy = 0.5 * jnp.einsum("i,j->ij", yarm, yarm)
                x_projection = jnp.einsum("ij,ij...->...", xx, wave_tensor)
                y_projection = jnp.einsum("ij,ij...->...", yy, wave_tensor)
            antenna_patterns[polarization.name] = (
                x_projection * x_transfer - y_projection * y_transfer
            )
        return antenna_patterns

    @staticmethod
    def _source_projection(
        ra: FloatLike,
        dec: FloatLike,
        psi: FloatLike,
        gmst: FloatLike,
        vector: Float[Array, "3"],
    ) -> tuple[Array, Array, Array]:
        """Return ``(vector . m, vector . n, vector . omega)`` elementwise.

        Same wave-frame convention as :meth:`_wave_frame`, without stacking
        the basis vectors: every quantity is a scalar per broadcast element.
        """
        phi = jnp.asarray(ra) - jnp.mod(jnp.asarray(gmst), 2 * jnp.pi)
        theta = jnp.pi / 2 - jnp.asarray(dec)
        phi, theta, psi = jnp.broadcast_arrays(phi, theta, jnp.asarray(psi))
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        cos_psi, sin_psi = jnp.cos(psi), jnp.sin(psi)
        a0, a1, a2 = vector[0], vector[1], vector[2]
        dot_u = a0 * cos_phi * cos_theta + a1 * cos_theta * sin_phi - a2 * sin_theta
        dot_v = -a0 * sin_phi + a1 * cos_phi
        dot_omega = a0 * sin_theta * cos_phi + a1 * sin_theta * sin_phi + a2 * cos_theta
        dot_m = -dot_u * sin_psi - dot_v * cos_psi
        dot_n = -dot_u * cos_psi + dot_v * sin_psi
        return dot_m, dot_n, dot_omega

    def td_response(
        self,
        time: Float[Array, " n_sample"],
        h_sky: dict[str, Float[Array, " n_sample"]],
        params: dict,
    ) -> Array:
        """Modulate the waveform in the sky frame by the detector response in the time domain.

        Args:
            time: Array of time samples.
            h_sky: Dictionary mapping polarization names to time-domain waveforms.
            params: Dictionary of source parameters.

        Returns:
            Array of detector response in time domain.
        """
        raise NotImplementedError

    def delay_from_geocenter(
        self, ra: FloatScalar, dec: FloatScalar, gmst: FloatScalar
    ) -> FloatScalar:
        """Calculate time delay between two detectors in geocentric coordinates.

        Based on XLALArrivaTimeDiff in TimeDelay.c
        https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_delay__h.html

        Args:
            ra (Float): Right ascension of the source in radians.
            dec (Float): Declination of the source in radians.
            gmst (Float): Greenwich mean sidereal time in radians.

        Returns:
            Float: Time delay from Earth center in seconds.
        """
        delta_d = -self.vertex
        gmst = jnp.mod(gmst, 2 * jnp.pi)
        phi = ra - gmst
        theta = jnp.pi / 2 - dec
        omega = jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )
        return jnp.einsum("i...,i->...", omega, delta_d) / C_SI

    def antenna_pattern(
        self,
        ra: FloatScalar,
        dec: FloatScalar,
        psi: FloatScalar,
        gmst: FloatScalar,
    ) -> dict[str, Complex]:
        """Compute antenna patterns for polarizations at specified sky location.

        In the long-wavelength approximation, the antenna pattern for a
        given polarization is the dyadic product between the detector
        tensor and the corresponding polarization tensor.

        Args:
            ra (Float): Source right ascension in radians.
            dec (Float): Source declination in radians.
            psi (Float): Source polarization angle in radians.
            gmst (Float): Greenwich mean sidereal time (GMST) in radians.

        Returns:
            dict[str, Complex]: Dictionary mapping polarization names to their antenna pattern values.
        """
        detector_tensor = self.tensor

        antenna_patterns = {}
        for polarization in self.polarization_mode:
            wave_tensor = polarization.tensor_from_sky(ra, dec, psi, gmst)
            antenna_patterns[polarization.name] = jnp.einsum(
                "ij,ij...->...", detector_tensor, wave_tensor
            )

        return antenna_patterns

    @jaxtyped(typechecker=typechecker)
    def load_and_set_psd(self, psd_file: str = "", asd_file: str = "") -> PowerSpectrum:
        """Load power spectral density (PSD) from file or default GWTC-2 catalog,
            and set it to the detector.

        Supported formats: .npz, .txt, .dat, .csv.
        Pass ``asd_file`` (or ``is_asd=True`` via :meth:`PowerSpectrum.from_file`)
        when the file contains amplitude spectral density values (Hz⁻¹/²); they
        are squared internally to produce a PSD.

        Args:
            psd_file (str, optional): Path to a PSD file (Hz⁻¹). If empty, uses GWTC-2 ASD.
            asd_file (str, optional): Path to an ASD file (Hz⁻¹/²). Values are squared.

        Returns:
            PowerSpectrum: The loaded PSD, already set on the detector.
        """
        if psd_file:
            _loaded_psd = PowerSpectrum.from_file(psd_file, is_asd=False)
        elif asd_file:
            _loaded_psd = PowerSpectrum.from_file(asd_file, is_asd=True)
        else:
            logger.info("Grabbing GWTC-2 PSD for " + self.name)
            url = asd_file_dict[self.name]
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            fd, tmp_file_name = tempfile.mkstemp(
                suffix=".txt", prefix=f"jim_asd_{self.name}_"
            )
            try:
                with os.fdopen(fd, "wb") as _fh:
                    _fh.write(response.content)
                _loaded_psd = PowerSpectrum.from_file(tmp_file_name, is_asd=True)
            finally:
                os.unlink(tmp_file_name)
        _loaded_psd.name = f"{self.name}_psd"
        self.set_psd(_loaded_psd)
        return self.psd

    def _equal_data_psd_frequencies(self) -> Bool:
        """Check if the frequencies of the data and PSD match.
        A helper function for `set_data` and `set_psd`.

        Return:
            Bool: True if the frequencies match, False otherwise.
        """
        if self.psd.is_empty or self.data.is_empty:
            # In this case, we simply skip the check
            return True
        if self.psd.n_freq != self.data.n_freq:
            # Cannot proceed comparison, needs interpolation
            return False
        return _same_grid(self.psd.frequencies, self.data.frequencies)

    def set_data(self, data: Data | Array, **kws) -> None:
        """Add data to the detector.

        Args:
            data (Data | Array): Data to be added to the detector, either as a `Data` object
                or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `Data` constructor.

        Returns:
            None
        """
        self.input_provenance_sha256 = None
        if isinstance(data, Data):
            self.data = data
        else:
            self.data = Data(data, **kws)
        # Assert PSD frequencies agree with data
        if not ((self.psd is None) or self._equal_data_psd_frequencies()):
            self.psd = self.psd.interpolate(self.data.frequencies)

    def set_psd(self, psd: PowerSpectrum | Array, **kws) -> None:
        """Add PSD to the detector.

        Args:
            psd (PowerSpectrum | Array): PSD to be added to the detector, either as a `PowerSpectrum`
                object or as a timeseries array.
            **kws (dict): Additional keyword arguments to pass to `PowerSpectrum` constructor.

        Returns:
            None
        """
        self.input_provenance_sha256 = None
        if isinstance(psd, PowerSpectrum):
            self.psd = psd
        else:
            # not clear if we want to support this
            self.psd = PowerSpectrum(psd, **kws)
        # Assert PSD frequencies agree with data frequencies
        if not ((self.data is None) or self._equal_data_psd_frequencies()):
            self.psd = self.psd.interpolate(self.data.frequencies)

    def inject_signal(
        self,
        duration: float,
        sampling_frequency: float,
        trigger_time: float,
        waveform_model,
        parameters: dict[str, float],
        f_min: float,
        f_max: float,
        start_time: Optional[float] = None,
        zero_noise: bool = False,
        rng_key: Optional[Key] = None,
        waveform_chunk_size: int = 262_144,
        host_resident: bool = False,
        host_storage: str = "memory",
        noise_generation: str = "legacy",
    ) -> None:
        """Inject a signal into the detector data.

        Note: The power spectral density must be set beforehand.

        Args:
            duration (float): Duration of the data segment in seconds.
            sampling_frequency (float): Sampling frequency in Hz.
            trigger_time (float): GPS time of the event trigger. Used to stamp
                ``trigger_time`` and derive ``gmst`` for the waveform projection,
                mirroring the behavior of ``TransientLikelihoodFD``.
            waveform_model: The waveform model to be injected.
            parameters (dict): Dictionary of likelihood-space source parameters.
            f_min (float): Minimum frequency in Hz. The waveform is zeroed below
                this frequency.
            f_max (float): Maximum frequency in Hz. Should be set to the same
                value used in the likelihood.
            start_time (Optional[float], optional): GPS start time of the
                data buffer in seconds. If None, defaults to
                ``trigger_time - duration + 2.0`` (2 s of data after the trigger).
                Defaults to None.
            waveform_chunk_size: Maximum number of in-band frequency samples
                projected at once. This bounds temporary waveform and response
                arrays for long XG injections.
            host_resident: Keep the native Fourier series on the host, without
                materializing time-domain or window arrays.
            host_storage: ``memory`` retains the host array in RAM; ``mmap``
                uses a temporary file and releases clean mapped pages by chunk.
            noise_generation: ``legacy`` reproduces the original full-array
                noise draw. ``indexed-v1`` draws independently by absolute
                frequency index in bounded chunks, with a new seeded realization.

        Returns:
            None
        """
        # Derive start_time if not provided
        if start_time is None:
            start_time = trigger_time - duration + 2.0
            logger.info(
                "start_time not provided. Defaulting to trigger_time - duration + 2.0 = %.3f s.",
                start_time,
            )

        # Make a copy of the parameters to avoid modifying the original dictionary
        params = parameters.copy()
        if (
            isinstance(waveform_chunk_size, bool)
            or not isinstance(waveform_chunk_size, int)
            or waveform_chunk_size <= 0
        ):
            raise ValueError("waveform_chunk_size must be a positive integer")

        # Stamp trigger_time and gmst — mirrors TransientLikelihoodFD.evaluate()
        params["trigger_time"] = float(trigger_time)
        params["gmst"] = float(compute_gmst(trigger_time))

        if host_storage not in {"memory", "mmap"} or noise_generation not in {
            "legacy",
            "indexed-v1",
        }:
            raise ValueError("Invalid host storage or noise generation mode")
        if not host_resident and (
            host_storage != "memory" or noise_generation != "legacy"
        ):
            raise ValueError("mmap storage and indexed noise require host_resident")

        if host_resident:
            self._inject_signal_host_resident(
                duration=duration,
                sampling_frequency=sampling_frequency,
                start_time=start_time,
                waveform_model=waveform_model,
                params=params,
                f_min=f_min,
                f_max=f_max,
                zero_noise=zero_noise,
                rng_key=rng_key,
                waveform_chunk_size=waveform_chunk_size,
                host_storage=host_storage,
                noise_generation=noise_generation,
            )
            return

        # 1. Set empty data to initialize the detector
        n_times = int(jnp.round(duration * sampling_frequency))
        self.set_data(
            Data(
                name=f"{self.name}_empty",
                td=jnp.zeros(n_times),
                delta_t=1 / sampling_frequency,
                start_time=start_time,
            )
        )

        # Set frequency bounds before evaluating the waveform
        self.set_frequency_bounds(f_min, f_max)

        # 2. Compute the projected in-band strain in bounded chunks. The full
        # output array is required by Data.from_fd, but waveform polarizations,
        # emission clocks, and response intermediates never scale beyond one
        # configured chunk.
        projected_host = np.zeros(self.data.n_freq, dtype=np.complex128)
        if len(self.sliced_frequencies) == 0:
            raise ValueError(
                f"injection band [{f_min}, {f_max}] contains no frequency samples"
            )
        first_frequency_index = round(
            float(self.sliced_frequencies[0]) * float(self.duration)
        )
        native_prefix = self.sliced_frequencies[:2]
        for start in range(0, len(self.sliced_frequencies), waveform_chunk_size):
            stop = min(start + waveform_chunk_size, len(self.sliced_frequencies))
            chunk_frequencies = self.sliced_frequencies[start:stop]
            polarisations = waveform_model(
                jnp.concatenate((native_prefix, chunk_frequencies)), params
            )
            polarisations = jax.tree.map(lambda value: value[2:], polarisations)
            projected_chunk = self.fd_response(
                chunk_frequencies,
                polarisations,
                params,
            )
            projected_host[
                first_frequency_index + start : first_frequency_index + stop
            ] = np.asarray(jax.device_get(projected_chunk))
        projected_strain = jnp.asarray(projected_host)

        # 3. Set the new data
        strain_data = projected_strain
        if not zero_noise:
            if rng_key is None:
                seed = int(time.time())
                rng_key = jax.random.key(seed)
                logger.info(
                    "No rng_key provided for noise simulation. Using time-based key with seed=%d.",
                    seed,
                )
            noise = self.psd.simulate_data(rng_key)
            strain_data += jnp.where(self.frequency_mask, noise, 0.0 + 0.0j)

        self.set_data(
            Data.from_fd(
                name=f"{self.name}_injected",
                fd_strain=strain_data,
                frequencies=self.frequencies,
                start_time=self.data.start_time,
            )
        )

        # 4. Update the sliced data and psd with the (potentially) new frequency bounds
        self.set_frequency_bounds()
        masked_signal = projected_strain[self.frequency_mask]

        df = self.sliced_frequencies[1] - self.sliced_frequencies[0]
        _optimal_snr_sq = inner_product(
            masked_signal, masked_signal, self.sliced_psd, df
        )
        optimal_snr = _optimal_snr_sq**0.5
        match_filtered_snr = complex_inner_product(
            masked_signal, self.sliced_fd_data, self.sliced_psd, df
        )
        match_filtered_snr /= optimal_snr

        # Save as attributes
        self.optimal_snr = optimal_snr
        self.match_filtered_snr = match_filtered_snr

        logger.info(f"For detector {self.name}, the injected signal has:")
        logger.info(f"  - Optimal SNR: {optimal_snr:.4f}")
        logger.info(f"  - Match filtered SNR: {match_filtered_snr:.4f}")

    def _inject_signal_host_resident(
        self,
        *,
        duration: float,
        sampling_frequency: float,
        start_time: float,
        waveform_model,
        params: dict,
        f_min: float,
        f_max: float,
        zero_noise: bool,
        rng_key: Optional[Key],
        waveform_chunk_size: int,
        host_storage: str,
        noise_generation: str,
    ) -> None:
        """Project, add noise, accumulate SNR and persist one native chunk.

        Indexed noise avoids all full-length device arrays. The optional
        mapping releases clean pages after each write. Legacy noise remains
        available for reproducing prior seeded data, at its old memory cost.
        """
        n_times = round(duration * sampling_frequency)
        if n_times % 2:
            raise ValueError(
                "host-resident injection requires an even number of time samples"
            )
        buffer, owner = allocate_native_strain(n_times // 2 + 1, host_storage)
        self.set_data(
            Data.from_host_fd(
                buffer,
                delta_t=1.0 / sampling_frequency,
                start_time=start_time,
                name=f"{self.name}_injected",
            )
        )
        self.data._host_storage_owner = owner
        # A caller may provide an already-native device PSD, in which case
        # set_data correctly skips interpolation. Move that table in bounded
        # slices too, and reuse the Data grid instead of keeping a duplicate.
        if not isinstance(self.psd.values, np.ndarray):
            host_psd = np.empty(self.psd.n_freq, dtype=np.float64)
            for start in range(0, self.psd.n_freq, waveform_chunk_size):
                stop = min(start + waveform_chunk_size, self.psd.n_freq)
                host_psd[start:stop] = np.asarray(
                    jax.device_get(self.psd.values[start:stop])
                )
            self.psd = PowerSpectrum(host_psd, self.data.frequencies, self.psd.name)
        elif self.psd.frequencies is not self.data.frequencies:
            self.psd = PowerSpectrum(
                self.psd.values, self.data.frequencies, self.psd.name
            )
        self.set_frequency_bounds(f_min, f_max)
        n_band = len(self.sliced_frequencies)
        if n_band < 2:
            raise ValueError(
                f"injection band [{f_min}, {f_max}] requires at least two frequency samples"
            )
        first = int(np.searchsorted(self.frequencies, self.sliced_frequencies[0]))
        prefix = jnp.asarray(self.sliced_frequencies[:2])
        df = float(self.sliced_frequencies[1] - self.sliced_frequencies[0])

        def project_chunk(frequencies):
            polarisations = waveform_model(
                jnp.concatenate((prefix, frequencies)), params
            )
            polarisations = jax.tree.map(lambda value: value[2:], polarisations)
            return self.fd_response(frequencies, polarisations, params)

        project = (
            jax.jit(project_chunk)
            if noise_generation == "indexed-v1"
            else project_chunk
        )

        legacy_noise = None
        if not zero_noise:
            if rng_key is None:
                seed = int(time.time())
                rng_key = (
                    jax.random.key(seed, impl="threefry2x32")
                    if noise_generation == "indexed-v1"
                    else jax.random.key(seed)
                )
                logger.info(
                    "No rng_key provided for noise simulation. Using time-based key with seed=%d.",
                    seed,
                )
            if noise_generation == "legacy":
                legacy_noise = self.psd.simulate_data(rng_key)
        optimal_sq = 0.0
        cross = 0.0j
        for start in range(0, n_band, waveform_chunk_size):
            stop = min(start + waveform_chunk_size, n_band)
            frequencies = jnp.asarray(self.sliced_frequencies[start:stop])
            s = np.asarray(jax.device_get(project(frequencies)))
            p = self.sliced_psd[start:stop]
            if not np.all(np.isfinite(s)) or not np.all(np.isfinite(p) & (p > 0)):
                raise ValueError(
                    f"Non-finite injection or invalid in-band PSD for {self.name}"
                )
            optimal_sq += float(np.sum((s.real**2 + s.imag**2) / p))
            destination = buffer[first + start : first + stop]
            destination[:] = s
            if not zero_noise:
                if noise_generation == "indexed-v1":
                    noise = compiled_indexed_noise(rng_key, p, df, first + start)
                else:
                    noise = legacy_noise[first + start : first + stop]
                n = np.asarray(jax.device_get(noise))
                if not np.all(np.isfinite(n)):
                    raise ValueError(f"Non-finite native noise for {self.name}")
                cross += complex(np.sum(np.conj(s) * n / p))
                destination += n
            release_mapped_pages(destination, written=True)
        del legacy_noise
        optimal_sq *= 4.0 * df
        cross *= 4.0 * df
        optimal_snr = optimal_sq**0.5
        self.set_frequency_bounds()
        self.optimal_snr = optimal_snr
        self.match_filtered_snr = (
            (optimal_sq + cross) / optimal_snr if optimal_snr > 0 else complex(np.nan)
        )
        self.data_preparation_diagnostics = {
            "host_storage": host_storage,
            "noise_generation": noise_generation,
            "noise_algorithm": INDEXED_NOISE_ALGORITHM
            if noise_generation == "indexed-v1"
            else "legacy-jax-full-array",
            "zero_noise": zero_noise,
            "waveform_chunk_size": waveform_chunk_size,
            "full_length_device_noise": not zero_noise and noise_generation == "legacy",
            "native_fd_storage_bytes": buffer.nbytes,
        }
        logger.info(f"For detector {self.name}, the injected signal has:")
        logger.info(f"  - Optimal SNR: {optimal_snr:.4f}")
        logger.info(f"  - Match filtered SNR: {self.match_filtered_snr:.4f}")

    def get_whitened_frequency_domain_strain(
        self, frequency_series: Complex[Array, " n_freq"]
    ) -> Complex[Array, " n_freq"]:
        """Get the whitened frequency-domain strain.
        Args:
            frequency_series (Complex[Array, "n_freq"]): Array of frequency domain data/signal.
        Returns:
            Complex[Array, "n_freq"]: Whitened frequency-domain strain.
        """
        scaled_asd = jnp.sqrt(self.psd.values * self.duration / 4)
        return (frequency_series / scaled_asd) * self.frequency_mask

    def whitened_frequency_to_time_domain_strain(
        self, whitened_frequency_series: Complex[Array, " n_time // 2 + 1"]
    ) -> Float[Array, " n_time"]:
        """Get the whitened frequency-domain strain.
        Args:
            whitened_frequency_series (Complex[Array, "n_time // 2 + 1"]):
                Array of whitened frequency domain data/signal.
        Returns:
            Float[Array, "n_time"]: Whitened time-domain strain/signal.
        """
        freq_mask_ratio = len(self.frequency_mask) / jnp.sqrt(
            jnp.sum(self.frequency_mask)
        )
        return jnp.fft.irfft(whitened_frequency_series) * freq_mask_ratio

    @property
    def whitened_frequency_domain_data(self) -> Complex[Array, " n_sample"]:
        """Get the whitened frequency-domain data.

        Args:
            frequency (Float[Array, "n_sample"]): Array of frequency samples.

        Returns:
            Float[Array, "n_sample"]: Whitened frequency-domain data.
        """

        return self.get_whitened_frequency_domain_strain(self.data.fd)

    @property
    def whitened_time_domain_data(self) -> Float[Array, " n_sample"]:
        """Get the whitened time-domain data.

        Args:
            time (Float[Array, "n_sample"]): Array of time samples.

        Returns:
            Float[Array, "n_sample"]: Whitened time-domain data.
        """
        return self.whitened_frequency_to_time_domain_strain(
            self.whitened_frequency_domain_data
        )


def get_H1() -> GroundBased2G:
    """Return a [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instance for LIGO Hanford (H1)."""
    return GroundBased2G(
        "H1",
        latitude=(46 + 27.0 / 60 + 18.528 / 3600) * DEG_TO_RAD,
        longitude=-(119 + 24.0 / 60 + 27.5657 / 3600) * DEG_TO_RAD,
        xarm_azimuth=125.9994 * DEG_TO_RAD,
        yarm_azimuth=215.9994 * DEG_TO_RAD,
        xarm_tilt=-6.195e-4,
        yarm_tilt=1.25e-5,
        elevation=142.554,
        arm_length_m=4_000.0,
        modes="pc",
    )


def get_L1() -> GroundBased2G:
    """Return a [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instance for LIGO Livingston (L1)."""
    return GroundBased2G(
        "L1",
        latitude=(30 + 33.0 / 60 + 46.4196 / 3600) * DEG_TO_RAD,
        longitude=-(90 + 46.0 / 60 + 27.2654 / 3600) * DEG_TO_RAD,
        xarm_azimuth=197.7165 * DEG_TO_RAD,
        yarm_azimuth=287.7165 * DEG_TO_RAD,
        xarm_tilt=-3.121e-4,
        yarm_tilt=-6.107e-4,
        elevation=-6.574,
        arm_length_m=4_000.0,
        modes="pc",
    )


def get_V1() -> GroundBased2G:
    """Return a [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instance for Virgo (V1)."""
    return GroundBased2G(
        "V1",
        latitude=(43 + 37.0 / 60 + 53.0921 / 3600) * DEG_TO_RAD,
        longitude=(10 + 30.0 / 60 + 16.1878 / 3600) * DEG_TO_RAD,
        xarm_azimuth=70.5674 * DEG_TO_RAD,
        yarm_azimuth=160.5674 * DEG_TO_RAD,
        xarm_tilt=0,
        yarm_tilt=0,
        elevation=51.884,
        arm_length_m=3_000.0,
        modes="pc",
    )


def get_ET() -> list[GroundBased2G]:
    """Return a list of three [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instances for Einstein Telescope (ET).

    ET is modelled as a triangle of three interferometers at adjacent vertices,
    with arms rotated by 120° relative to each other. Vertex positions are
    propagated using the spherical forward-azimuth (haversine) formula with a
    latitude-dependent Earth radius derived from the WGS-84 ellipsoid.
    """
    name = "ET"
    latitude = (43 + 37.0 / 60 + 53.0921 / 3600) * DEG_TO_RAD
    longitude = (10 + 30.0 / 60 + 16.1878 / 3600) * DEG_TO_RAD
    xarm_azimuth = 70.5674 * DEG_TO_RAD
    yarm_azimuth = 130.5674 * DEG_TO_RAD
    xarm_tilt = 0
    yarm_tilt = 0
    elevation = 51.884
    length: float = 1e4  # arm length in metres

    a = EARTH_SEMI_MAJOR_AXIS / 1e3  # Numerical instability avoidance
    b = EARTH_SEMI_MINOR_AXIS / 1e3
    earth_approx_radius = (
        a
        * b
        / (jnp.sqrt(a**2 * jnp.sin(latitude) ** 2 + b**2 * jnp.cos(latitude) ** 2))
    )
    earth_approx_radius *= 1e3

    # Navigation bearing (clockwise from North) corresponding to xarm_azimuth
    # (counter-clockwise from East): brng = pi/2 - azimuth.
    # Both brng and the arm azimuths are incremented by 240° (4π/3) per vertex.
    brng = jnp.pi / 2 - xarm_azimuth

    ifos = []
    for i in range(3):
        ifos.append(
            GroundBased2G(
                f"{name}{i + 1}",
                latitude=float(latitude),
                longitude=float(longitude),
                xarm_azimuth=float(xarm_azimuth),
                yarm_azimuth=float(yarm_azimuth),
                elevation=elevation,
                xarm_tilt=xarm_tilt,
                yarm_tilt=yarm_tilt,
                arm_length_m=length,
            )
        )
        # Propagate to next vertex using the spherical forward-azimuth formula.
        # Coordinate update must precede arm rotation (uses current bearing).
        d = length / earth_approx_radius
        phi1 = latitude
        phi2 = jnp.arcsin(
            jnp.sin(phi1) * jnp.cos(d) + jnp.cos(phi1) * jnp.sin(d) * jnp.cos(brng)
        )
        longitude = longitude + jnp.arctan2(
            jnp.sin(brng) * jnp.sin(d) * jnp.cos(phi1),
            jnp.cos(d) - jnp.sin(phi1) * jnp.sin(phi2),
        )
        latitude = phi2
        # Rotate arms and bearing for the next detector vertex (240°, i.e. 4π/3, per vertex)
        xarm_azimuth += (4 / 3) * jnp.pi
        yarm_azimuth += (4 / 3) * jnp.pi
        brng += (4 / 3) * jnp.pi
    return ifos


def get_CE() -> GroundBased2G:
    """Return a [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G] instance for Cosmic Explorer (CE).

    CE shares the LIGO Hanford site geometry.
    """
    return GroundBased2G(
        "CE",
        latitude=(46 + 27.0 / 60 + 18.528 / 3600) * DEG_TO_RAD,
        longitude=-(119 + 24.0 / 60 + 27.5657 / 3600) * DEG_TO_RAD,
        xarm_azimuth=125.9994 * DEG_TO_RAD,
        yarm_azimuth=215.994 * DEG_TO_RAD,
        xarm_tilt=-6.195e-4,
        yarm_tilt=1.25e-5,
        elevation=142.554,
        arm_length_m=40_000.0,
        modes="pc",
    )


def get_CE_A() -> GroundBased2G:
    """Return the 40 km CE-A fiducial geometry with detector channel name CE.

    The Cosmic Explorer project recommends the intentionally offshore location
    46 degrees north, 125 degrees west for reproducible network calculations;
    this is not a selected construction site. Table 2 of
    https://arxiv.org/html/2307.10421v2 specifies the x arm at 260 degrees north
    of east. The orthogonal y arm at 350 degrees completes a right-handed
    (x, y, outward vertical) frame. See also https://cosmicexplorer.org/celocations.html.

    Zero elevation and arm tilts are explicit fiducial modeling conventions,
    not surveyed measurements. The legacy :func:`get_CE` remains at Hanford.
    """
    return GroundBased2G(
        "CE",
        latitude=46.0 * DEG_TO_RAD,
        longitude=-125.0 * DEG_TO_RAD,
        xarm_azimuth=260.0 * DEG_TO_RAD,
        yarm_azimuth=350.0 * DEG_TO_RAD,
        xarm_tilt=0.0,
        yarm_tilt=0.0,
        elevation=0.0,
        arm_length_m=40_000.0,
        modes="pc",
    )


def get_ET_Sardinia() -> list[GroundBased2G]:
    """Return a connected triangular ET at the published Sardinia fiducial site.

    The first corner is at 40 degrees 31 minutes north, 9 degrees 25 minutes
    east, with x/y azimuths 90/150 degrees north of east and nominal 10 km arms
    (https://arxiv.org/html/2307.10421v2, Table 2). Zero elevation is an explicit
    fiducial convention; this does not specify surveyed underground facilities.

    All corners lie on the zero-height Earth ellipsoid. Two corners are placed
    exactly 10 km in chord distance from the first, along its specified local
    azimuths. Their closing side is 3.1 mm shorter than the nominal 10 km arm
    length. Every directed arm points along an actual connecting ECEF chord;
    Earth curvature therefore produces small downward arm tilts and deviations
    below 0.000021 degrees from 60-degree opening angles.
    The finite-arm response uses the nominal 10 km for both arms; it does not
    resolve the closing side's 3.1 mm geometric difference.

    This new geometry does not reproduce the legacy Bilby-style pairing of
    rotated arms with independently propagated corners: those arms can point
    away from their neighboring corners. The historical :func:`get_ET` remains
    unchanged for existing fixtures.
    """
    latitude = (40.0 + 31.0 / 60) * DEG_TO_RAD
    longitude = (9.0 + 25.0 / 60) * DEG_TO_RAD
    length = 10_000.0
    major, minor = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
    ellipsoid_metric = 1.0 / np.asarray([major, major, minor]) ** 2
    eccentricity_squared = 1.0 - (minor / major) ** 2

    def local_frame(lat, lon):
        up = np.asarray(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        east = np.asarray([-np.sin(lon), np.cos(lon), 0.0])
        return east, np.cross(up, east), up

    normal_radius = major / np.sqrt(1.0 - eccentricity_squared * np.sin(latitude) ** 2)
    first = np.asarray(
        [
            normal_radius * np.cos(latitude) * np.cos(longitude),
            normal_radius * np.cos(latitude) * np.sin(longitude),
            normal_radius * (1.0 - eccentricity_squared) * np.sin(latitude),
        ]
    )
    east, north, up = local_frame(latitude, longitude)
    vertices = [first]
    for azimuth in (90.0 * DEG_TO_RAD, 150.0 * DEG_TO_RAD):
        horizontal = np.cos(azimuth) * east + np.sin(azimuth) * north
        tilt = -length / (2.0 * major)
        # Solve the short downward chord's intersection with the ellipsoid.
        # Only fixed preset constants enter this small host-side calculation.
        for _ in range(8):
            endpoint = first + length * (np.cos(tilt) * horizontal + np.sin(tilt) * up)
            derivative = length * (-np.sin(tilt) * horizontal + np.cos(tilt) * up)
            residual = np.dot(endpoint * ellipsoid_metric, endpoint) - 1.0
            tilt -= residual / (2.0 * np.dot(endpoint * ellipsoid_metric, derivative))
        vertices.append(
            first + length * (np.cos(tilt) * horizontal + np.sin(tilt) * up)
        )

    detectors = []
    for index, vertex in enumerate(vertices):
        # At zero geodetic height this ellipsoid-coordinate inverse is exact.
        lat = np.arctan2(
            vertex[2] / (1.0 - eccentricity_squared), np.hypot(*vertex[:2])
        )
        lon = np.arctan2(vertex[1], vertex[0])
        east, north, up = local_frame(lat, lon)
        angles = []
        for neighbor in ((index + 1) % 3, (index + 2) % 3):
            arm = vertices[neighbor] - vertex
            arm /= np.linalg.norm(arm)
            angles.append(
                (
                    float(
                        np.arctan2(np.dot(arm, north), np.dot(arm, east))
                        % (2.0 * np.pi)
                    ),
                    float(
                        np.arctan2(
                            np.dot(arm, up),
                            np.hypot(np.dot(arm, east), np.dot(arm, north)),
                        )
                    ),
                )
            )
        detectors.append(
            GroundBased2G(
                f"ET{index + 1}",
                latitude=float(lat),
                longitude=float(lon),
                elevation=0.0,
                xarm_azimuth=angles[0][0],
                yarm_azimuth=angles[1][0],
                xarm_tilt=angles[0][1],
                yarm_tilt=angles[1][1],
                arm_length_m=length,
                modes="pc",
            )
        )
    return detectors


def get_detector_preset(
    site_overrides: Optional[dict[str, str]] = None,
) -> dict[str, GroundBased2G | list[GroundBased2G]]:
    """Return a dictionary of pre-configured detector instances.

    Args:
        site_overrides: Explicit geometry labels keyed by detector channel.
            ``{"CE": "CE_A_fiducial_2023"}`` selects the published CE-A
            fiducial site, and ``{"ET": "ET_Sardinia_fiducial_2023"}`` selects
            a connected Sardinia triangle. Omitting overrides preserves all
            legacy geometries. Channel names remain CE and ET1/ET2/ET3.

    Returns:
        dict: Mapping of detector name to detector object(s).
            Keys are ``"H1"``, ``"L1"``, ``"V1"``, ``"CE"`` (single
            [`GroundBased2G`][jimgw.core.single_event.detector.GroundBased2G]) and ``"ET"`` (list of three).
    """
    overrides = site_overrides or {}
    site_builders = {
        "CE": {"CE_A_fiducial_2023": get_CE_A},
        "ET": {"ET_Sardinia_fiducial_2023": get_ET_Sardinia},
    }
    for detector, site in overrides.items():
        if detector not in site_builders or site not in site_builders[detector]:
            raise ValueError(f"Unsupported detector site override: {detector}={site!r}")
    detectors = {
        "H1": get_H1(),
        "L1": get_L1(),
        "V1": get_V1(),
        "ET": get_ET(),
        "CE": get_CE(),
    }
    for detector, site in overrides.items():
        detectors[detector] = site_builders[detector][site]()
    return detectors
