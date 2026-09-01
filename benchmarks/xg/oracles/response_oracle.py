"""Independent segmented time-domain response for XG qualification.

This module deliberately does not import Jim.  It evaluates a plane wave on a
rotating detector worldline and integrates both photon-path pieces of each
round-trip arm response in the time domain.  The companion frequency-domain
function is present only for convergence and convention checks.

The round-trip transfer follows Eq. (4) of Essick, Vitale, and Evans (2017),
with ``mu`` equal to propagation-direction dot arm and NumPy's normalized
``sinc`` convention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

C_SI = 299_792_458.0
SIDEREAL_DAY_S = 86_164.09053083288
SIDEREAL_OMEGA = 2.0 * np.pi / SIDEREAL_DAY_S


@dataclass(frozen=True)
class DetectorGeometry:
    """Earth-fixed detector geometry used by the independent oracle."""

    vertex_m: NDArray[np.float64]
    x_arm: NDArray[np.float64]
    y_arm: NDArray[np.float64]
    arm_length_m: float

    def __post_init__(self) -> None:
        for name in ("vertex_m", "x_arm", "y_arm"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite three-vector")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.arm_length_m) or self.arm_length_m <= 0.0:
            raise ValueError("arm_length_m must be finite and positive")
        for name in ("x_arm", "y_arm"):
            value = getattr(self, name)
            if not np.isclose(np.linalg.norm(value), 1.0, rtol=0.0, atol=1e-12):
                raise ValueError(f"{name} must be a unit vector")


def _rotate_about_z(
    vectors: NDArray[np.float64],
    angles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate one Earth-fixed vector into the inertial frame at each angle."""

    vector = np.asarray(vectors, dtype=np.float64)
    angle = np.asarray(angles, dtype=np.float64)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.stack(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
            np.broadcast_to(vector[2], angle.shape),
        ),
        axis=-1,
    )


def wave_frame(
    ra: float,
    dec: float,
    psi: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return inertial plus/cross tensors and the direction to the source."""

    values = np.asarray((ra, dec, psi), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("sky coordinates must be finite")
    source = np.asarray(
        (
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        )
    )
    u = np.asarray(
        (
            np.sin(dec) * np.cos(ra),
            np.sin(dec) * np.sin(ra),
            -np.cos(dec),
        )
    )
    v = np.asarray((-np.sin(ra), np.cos(ra), 0.0))
    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)
    plus = np.outer(m, m) - np.outer(n, n)
    cross = np.outer(m, n) + np.outer(n, m)
    return plus, cross, source


def finite_arm_transfer(
    frequency_hz: ArrayLike,
    mu: ArrayLike,
    arm_length_m: float,
) -> NDArray[np.complex128]:
    """Return the published round-trip arm transfer."""

    frequency, direction_cosine = np.broadcast_arrays(
        np.asarray(frequency_hz, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
    )
    x = frequency * float(arm_length_m) / C_SI
    outgoing = np.sinc(x * (1.0 - direction_cosine)) * np.exp(
        -1j * np.pi * x * (1.0 - direction_cosine)
    )
    returning = np.sinc(x * (1.0 + direction_cosine)) * np.exp(
        -1j * np.pi * x * (3.0 - direction_cosine)
    )
    return np.asarray(0.5 * (outgoing + returning), dtype=np.complex128)


def frequency_domain_response(
    frequency_hz: ArrayLike,
    emission_time_s: ArrayLike,
    *,
    geometry: DetectorGeometry,
    gmst_at_zero: float,
    ra: float,
    dec: float,
    psi: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate the adiabatic frequency-domain plus/cross response."""

    frequency, emission_time = np.broadcast_arrays(
        np.asarray(frequency_hz, dtype=np.float64),
        np.asarray(emission_time_s, dtype=np.float64),
    )
    plus, cross, source = wave_frame(ra, dec, psi)
    gmst = float(gmst_at_zero) + SIDEREAL_OMEGA * emission_time
    vertex = _rotate_about_z(geometry.vertex_m, gmst)
    x_arm = _rotate_about_z(geometry.x_arm, gmst)
    y_arm = _rotate_about_z(geometry.y_arm, gmst)
    delay = -np.einsum("...i,i->...", vertex, source) / C_SI

    def arm_response(
        arm: NDArray[np.float64],
        polarization: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        projection = 0.5 * np.einsum("...i,ij,...j->...", arm, polarization, arm)
        mu = -np.einsum("...i,i->...", arm, source)
        return projection * finite_arm_transfer(
            frequency,
            mu,
            geometry.arm_length_m,
        )

    phase = np.exp(-2j * np.pi * frequency * delay)
    response_plus = (arm_response(x_arm, plus) - arm_response(y_arm, plus)) * phase
    response_cross = (arm_response(x_arm, cross) - arm_response(y_arm, cross)) * phase
    return response_plus, response_cross


def _interp_complex(
    time_s: NDArray[np.float64],
    values: NDArray[np.complex128],
    query_s: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Linearly interpolate a complex series and use zero outside its support."""

    real = np.interp(query_s, time_s, values.real, left=0.0, right=0.0)
    imag = np.interp(query_s, time_s, values.imag, left=0.0, right=0.0)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def segmented_round_trip_response(
    time_s: ArrayLike,
    plus_strain: ArrayLike,
    cross_strain: ArrayLike,
    *,
    geometry: DetectorGeometry,
    gmst_at_zero: float,
    ra: float,
    dec: float,
    psi: float,
    arm_subsegments: int = 16,
) -> NDArray[np.complex128]:
    """Integrate the moving-worldline round-trip response in the time domain.

    The detector geometry is evaluated at reception time.  Each arm's two
    photon-path intervals are integrated independently with midpoint segments.
    Increasing ``arm_subsegments`` therefore supplies a direct convergence
    test for the finite-arm interpolation and light-path quadrature.
    """

    time = np.asarray(time_s, dtype=np.float64)
    plus_values = np.asarray(plus_strain, dtype=np.complex128)
    cross_values = np.asarray(cross_strain, dtype=np.complex128)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must be a strictly increasing one-dimensional grid")
    if plus_values.shape != time.shape or cross_values.shape != time.shape:
        raise ValueError("polarization series must match the time grid")
    if isinstance(arm_subsegments, bool) or arm_subsegments < 1:
        raise ValueError("arm_subsegments must be a positive integer")

    return segmented_round_trip_response_function(
        time,
        lambda query: _interp_complex(time, plus_values, query),
        lambda query: _interp_complex(time, cross_values, query),
        geometry=geometry,
        gmst_at_zero=gmst_at_zero,
        ra=ra,
        dec=dec,
        psi=psi,
        arm_subsegments=arm_subsegments,
    )


def segmented_round_trip_response_function(
    time_s: ArrayLike,
    plus_function: Callable[[NDArray[np.float64]], ArrayLike],
    cross_function: Callable[[NDArray[np.float64]], ArrayLike],
    *,
    geometry: DetectorGeometry,
    gmst_at_zero: float,
    ra: float,
    dec: float,
    psi: float,
    arm_subsegments: int = 16,
) -> NDArray[np.complex128]:
    """Integrate analytic polarization functions on the retarded light path.

    This form avoids a stored high-rate source series for hour-long signals.
    The callables are evaluated directly at every retarded midpoint, so a
    separate ``arm_subsegments`` refinement isolates photon-path quadrature
    error without conflating it with interpolation error.
    """

    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must be a strictly increasing one-dimensional grid")
    if not callable(plus_function) or not callable(cross_function):
        raise TypeError("polarization functions must be callable")
    if isinstance(arm_subsegments, bool) or arm_subsegments < 1:
        raise ValueError("arm_subsegments must be a positive integer")

    plus_tensor, cross_tensor, source = wave_frame(ra, dec, psi)
    gmst = float(gmst_at_zero) + SIDEREAL_OMEGA * time
    vertex = _rotate_about_z(geometry.vertex_m, gmst)
    x_arm = _rotate_about_z(geometry.x_arm, gmst)
    y_arm = _rotate_about_z(geometry.y_arm, gmst)
    vertex_delay = -np.einsum("...i,i->...", vertex, source) / C_SI

    def integrate_arm(arm: NDArray[np.float64]) -> NDArray[np.complex128]:
        plus_projection = 0.5 * np.einsum("...i,ij,...j->...", arm, plus_tensor, arm)
        cross_projection = 0.5 * np.einsum("...i,ij,...j->...", arm, cross_tensor, arm)
        mu = -np.einsum("...i,i->...", arm, source)
        first_width = geometry.arm_length_m * (1.0 - mu) / C_SI
        second_width = geometry.arm_length_m * (1.0 + mu) / C_SI
        first_average = np.zeros(time.shape, dtype=np.complex128)
        second_average = np.zeros(time.shape, dtype=np.complex128)
        for index in range(int(arm_subsegments)):
            fraction = (index + 0.5) / arm_subsegments
            first_retardation = fraction * first_width
            second_retardation = first_width + fraction * second_width
            for accumulator, retardation in (
                (first_average, first_retardation),
                (second_average, second_retardation),
            ):
                query = time - vertex_delay - retardation
                plus_values = np.asarray(plus_function(query), dtype=np.complex128)
                cross_values = np.asarray(cross_function(query), dtype=np.complex128)
                if plus_values.shape != time.shape or cross_values.shape != time.shape:
                    raise ValueError(
                        "polarization functions must return arrays matching time_s"
                    )
                accumulator += plus_projection * plus_values
                accumulator += cross_projection * cross_values
        return 0.5 * (first_average + second_average) / arm_subsegments

    return integrate_arm(x_arm) - integrate_arm(y_arm)
