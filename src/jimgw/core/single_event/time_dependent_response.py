"""Time-dependent long-wavelength response primitives for terrestrial detectors.

This module contains the inspectable, dense response primitives used by the XG
truth path.  It intentionally does not own waveform or sampler state.  The
caller supplies the intrinsic time-to-coalescence map and the detector's
Earth-fixed geometry.

The canonical convention is

``t_emit = trigger_time + t_c - tau``

and, for a GMST reference at the trigger time,

``gmst_emit = gmst_at_trigger + OMEGA_EARTH * (t_c - tau)``.

The detector phase follows :class:`GroundBased2G`: a positive geocentric time
or detector delay contributes ``exp(-2 pi i f dt)`` exactly once.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from jimgw.core.constants import C_SI, DAYSID_SI, MTSUN

SIDEREAL_ANGULAR_FREQUENCY = 2.0 * jnp.pi / DAYSID_SI
"""Earth's sidereal angular frequency in radians per second."""

SIDEREAL_FREQUENCY = 1.0 / DAYSID_SI
"""Earth's sidereal frequency in hertz."""

# These are the five non-uniform phases used by the Chen sideband
# construction.  They determine a trigonometric polynomial with harmonics
# k=0, +/-1, +/-2 without solving a linear system at run time.
SIDEREAL_COLLOCATION_PHASES = jnp.asarray(
    [0.0, jnp.pi / 4.0, jnp.pi / 2.0, jnp.pi, 3.0 * jnp.pi / 2.0]
)


class AntennaPatterns(NamedTuple):
    """Long-wavelength plus and cross antenna factors."""

    plus: Float[Array, "..."]
    cross: Float[Array, "..."]


class SiderealHarmonics(NamedTuple):
    """Real Fourier coefficients through the second sidereal harmonic."""

    constant: Array
    cos_one: Array
    sin_one: Array
    cos_two: Array
    sin_two: Array


class AntennaHarmonics(NamedTuple):
    """Five-harmonic representations of both tensor polarizations."""

    plus: SiderealHarmonics
    cross: SiderealHarmonics


class LongWavelengthResponse(NamedTuple):
    """Frequency-dependent response values before waveform projection."""

    antenna: AntennaPatterns
    delay: Float[Array, "..."]
    gmst: Float[Array, "..."]


def inertial_source_direction(
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
) -> Float[Array, "3 ..."]:
    """Return the geocentre-to-source unit vector in inertial coordinates.

    Right ascension is already an inertial angle, so this vector deliberately
    has no GMST dependence.  The Earth-fixed source direction used for the
    terrestrial detector geometry is a different object and remains in
    :func:`_sky_basis`.
    """

    ra, dec = jnp.broadcast_arrays(ra, dec)
    cos_dec = jnp.cos(dec)
    return jnp.stack(
        [cos_dec * jnp.cos(ra), cos_dec * jnp.sin(ra), jnp.sin(dec)],
        axis=0,
    )


def earth_orbital_curvature_delay(
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    emission_offset: Float[Array, "..."] | float,
    acceleration_over_c: Float[Array, "3"] | tuple[float, float, float],
    jerk_over_c: Float[Array, "3"] | tuple[float, float, float],
) -> Float[Array, "..."]:
    """Return the nonlinear Earth-orbital arrival-time residual.

    ``emission_offset`` is ``t_c - tau(f)`` relative to the trigger epoch.
    ``acceleration_over_c`` and ``jerk_over_c`` are inertial Earth-centre
    quadratic and cubic surrogate coefficients divided by the speed of light,
    in ``s^-1`` and ``s^-2`` respectively. They can be instantaneous
    ephemeris derivatives or coefficients fitted over a declared time window.
    Removing the constant position and linear velocity terms leaves the
    light-second displacement

    ``delta_r/c = 1/2 (a/c) dt^2 + 1/6 (j/c) dt^3``.

    The returned detector delay is ``-n . delta_r/c``, matching the sign of
    :func:`delay_from_geocenter`.  Consequently the frequency-domain response
    contains ``exp(+2 pi i f n . delta_r/c)``.

    The coefficient vectors must have exactly three inertial Cartesian
    components.  Finite-value validation belongs to the detector configuration
    boundary so this primitive remains JIT compatible.
    """

    acceleration_over_c = jnp.asarray(acceleration_over_c)
    jerk_over_c = jnp.asarray(jerk_over_c)
    if acceleration_over_c.shape != (3,):
        raise ValueError("acceleration_over_c must have exactly three components")
    if jerk_over_c.shape != (3,):
        raise ValueError("jerk_over_c must have exactly three components")

    ra, dec, emission_offset = jnp.broadcast_arrays(ra, dec, emission_offset)
    vector_shape = (3,) + (1,) * emission_offset.ndim
    acceleration_over_c = jnp.reshape(acceleration_over_c, vector_shape)
    jerk_over_c = jnp.reshape(jerk_over_c, vector_shape)
    displacement_over_c = (
        0.5 * acceleration_over_c * emission_offset**2
        + (1.0 / 6.0) * jerk_over_c * emission_offset**3
    )
    source_direction = inertial_source_direction(ra, dec)
    return -jnp.sum(source_direction * displacement_over_c, axis=0)


def time_to_coalescence_2pn(
    frequency: Float[Array, "..."],
    mass_1: Float[Array, ""] | float,
    mass_2: Float[Array, ""] | float,
    chi_1: Float[Array, ""] | float = 0.0,
    chi_2: Float[Array, ""] | float = 0.0,
    *,
    mode: int = 2,
) -> Float[Array, "..."]:
    """Return the non-negative 2PN time to coalescence for a waveform mode.

    The formula is Eq. (3.3) of Blanchet et al. (1995), including the aligned
    spin-orbit and spin-spin terms used by the Bilby-XG reference
    implementation.  Masses are detector-frame solar masses, frequencies are
    hertz, and spins are dimensionless components along the orbital angular
    momentum.  ``mode`` can be positive or negative; only ``abs(mode)`` sets
    the clock, and zero is invalid.

    The inspiral expression assumes positive masses and frequencies. Its
    post-inspiral continuation can become negative or turn upward near merger.
    The clock is therefore fixed at coalescence at and above the Schwarzschild
    ISCO frequency, and any earlier negative continuation is also fixed at
    zero. This prevents the response from moving after coalescence or rebounding
    at frequencies where the PN clock is outside its declared domain.
    """

    mode_number = abs(mode)
    if mode_number == 0:
        raise ValueError("mode must be nonzero")

    frequency_22 = 2.0 * jnp.asarray(frequency) / mode_number
    mass_1 = jnp.asarray(mass_1)
    mass_2 = jnp.asarray(mass_2)
    total_mass = mass_1 + mass_2
    eta = mass_1 * mass_2 / total_mass**2
    chirp_mass = (mass_1 * mass_2) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)

    total_mass_seconds = total_mass * MTSUN
    chirp_mass_seconds = chirp_mass * MTSUN
    x = jnp.pi * total_mass_seconds * frequency_22

    beta = (
        (113.0 * (mass_1 / total_mass) ** 2 + 75.0 * eta) * chi_1
        + (113.0 * (mass_2 / total_mass) ** 2 + 75.0 * eta) * chi_2
    ) / 12.0
    sigma = (721.0 - 247.0) * eta * chi_1 * chi_2 / 48.0

    tau_zero = (
        5.0
        / 256.0
        * chirp_mass_seconds
        * (jnp.pi * chirp_mass_seconds * frequency_22) ** (-8.0 / 3.0)
    )
    tau_two = (
        4.0 / 3.0 * (743.0 / 336.0 + 11.0 * eta / 4.0) * x ** (2.0 / 3.0) * tau_zero
    )
    tau_three = -8.0 / 5.0 * (4.0 * jnp.pi - beta) * x * tau_zero
    tau_four = (
        2.0
        * (
            3058673.0 / 1016064.0
            + 5429.0 * eta / 1008.0
            + 617.0 * eta**2 / 144.0
            - sigma
        )
        * x ** (4.0 / 3.0)
        * tau_zero
    )
    inspiral_tau = tau_zero + tau_two + tau_three + tau_four
    isco_frequency_22 = 1.0 / (6.0 ** (3.0 / 2.0) * jnp.pi * total_mass_seconds)
    return jnp.where(
        frequency_22 < isco_frequency_22,
        jnp.maximum(inspiral_tau, 0.0),
        0.0,
    )


def emission_time(
    trigger_time: Float[Array, "..."] | float,
    t_c: Float[Array, "..."] | float,
    tau: Float[Array, "..."] | float,
) -> Float[Array, "..."]:
    """Return the geocentric mode-emission epoch in GPS seconds."""

    return jnp.asarray(trigger_time) + jnp.asarray(t_c) - jnp.asarray(tau)


def linear_gmst(
    gmst_reference: Float[Array, "..."] | float,
    elapsed_time: Float[Array, "..."] | float,
    *,
    wrap: bool = True,
) -> Float[Array, "..."]:
    """Evolve one GMST reference with the constant sidereal rate.

    ``elapsed_time`` is relative to the epoch of ``gmst_reference``.  This
    avoids subtracting two large absolute GMST arguments for every sample.
    """

    angle = jnp.asarray(gmst_reference) + SIDEREAL_ANGULAR_FREQUENCY * jnp.asarray(
        elapsed_time
    )
    return jnp.mod(angle, 2.0 * jnp.pi) if wrap else angle


def emission_gmst(
    gmst_at_trigger: Float[Array, "..."] | float,
    t_c: Float[Array, "..."] | float,
    tau: Float[Array, "..."] | float,
    *,
    wrap: bool = True,
) -> Float[Array, "..."]:
    """Return GMST at ``trigger_time + t_c - tau``.

    ``gmst_at_trigger`` must refer to ``trigger_time``.  A positive ``tau``
    therefore moves both emission time and the unwrapped sidereal angle
    backwards.
    """

    return linear_gmst(gmst_at_trigger, jnp.asarray(t_c) - jnp.asarray(tau), wrap=wrap)


def _sky_basis(
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    psi: Float[Array, "..."] | float,
    gmst: Float[Array, "..."] | float,
) -> tuple[Array, Array, Array]:
    """Return polarization basis vectors and propagation direction."""

    ra, dec, psi, gmst = jnp.broadcast_arrays(ra, dec, psi, gmst)
    phi = ra - jnp.mod(gmst, 2.0 * jnp.pi)
    cos_dec = jnp.cos(dec)
    sin_dec = jnp.sin(dec)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    u = jnp.stack([sin_dec * cos_phi, sin_dec * sin_phi, -cos_dec], axis=0)
    v = jnp.stack([-sin_phi, cos_phi, jnp.zeros_like(phi)], axis=0)
    m = -u * jnp.sin(psi) - v * jnp.cos(psi)
    n = -u * jnp.cos(psi) + v * jnp.sin(psi)
    omega = jnp.stack([cos_dec * cos_phi, cos_dec * sin_phi, sin_dec], axis=0)
    return m, n, omega


def long_wavelength_antenna_patterns(
    detector_tensor: Float[Array, "3 3"],
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    psi: Float[Array, "..."] | float,
    gmst: Float[Array, "..."] | float,
) -> AntennaPatterns:
    """Evaluate vector-safe long-wavelength plus and cross antenna factors."""

    m, n, _ = _sky_basis(ra, dec, psi, gmst)
    plus_tensor = jnp.einsum("i...,j...->ij...", m, m) - jnp.einsum(
        "i...,j...->ij...", n, n
    )
    cross_tensor = jnp.einsum("i...,j...->ij...", m, n) + jnp.einsum(
        "i...,j...->ij...", n, m
    )
    return AntennaPatterns(
        plus=jnp.einsum("ij,ij...->...", detector_tensor, plus_tensor),
        cross=jnp.einsum("ij,ij...->...", detector_tensor, cross_tensor),
    )


def delay_from_geocenter(
    detector_vertex: Float[Array, "3"],
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    gmst: Float[Array, "..."] | float,
) -> Float[Array, "..."]:
    """Return the geocentre-to-detector arrival-time delay in seconds.

    The sign matches ``GroundBased2G.delay_from_geocenter`` and LAL's arrival
    time convention.
    """

    zeros = jnp.zeros_like(jnp.broadcast_arrays(ra, dec, gmst)[0])
    _, _, omega = _sky_basis(ra, dec, zeros, gmst)
    return -jnp.einsum("i...,i->...", omega, detector_vertex) / C_SI


def long_wavelength_response(
    detector_tensor: Float[Array, "3 3"],
    detector_vertex: Float[Array, "3"],
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    psi: Float[Array, "..."] | float,
    gmst: Float[Array, "..."] | float,
) -> LongWavelengthResponse:
    """Evaluate antenna factors and delay at explicit sidereal angles."""

    gmst = jnp.asarray(gmst)
    return LongWavelengthResponse(
        antenna=long_wavelength_antenna_patterns(detector_tensor, ra, dec, psi, gmst),
        delay=delay_from_geocenter(detector_vertex, ra, dec, gmst),
        gmst=gmst,
    )


def sidereal_harmonics_from_collocation(
    values: Array,
) -> SiderealHarmonics:
    """Recover harmonics 0, +/-1, +/-2 from five collocation values.

    The final dimension must contain values at
    :data:`SIDEREAL_COLLOCATION_PHASES`, in that order.  The returned series is
    ``a0 + a1c cos(x) + a1s sin(x) + a2c cos(2x) + a2s sin(2x)``.
    """

    values = jnp.asarray(values)
    if values.ndim == 0 or values.shape[-1] != 5:
        raise ValueError("collocation values must have a final dimension of length 5")

    r_one, r_two, r_three, r_four, r_five = (values[..., index] for index in range(5))
    root_two = jnp.sqrt(jnp.asarray(2.0, dtype=values.dtype))
    return SiderealHarmonics(
        constant=(r_one + r_three + r_four + r_five) / 4.0,
        cos_one=(r_one - r_four) / 2.0,
        sin_one=(r_three - r_five) / 2.0,
        cos_two=(r_one + r_four - r_three - r_five) / 4.0,
        sin_two=r_two
        + ((root_two - 1.0) * (r_four + r_five) - (root_two + 1.0) * (r_one + r_three))
        / 4.0,
    )


def evaluate_sidereal_harmonics(
    harmonics: SiderealHarmonics,
    phase: Float[Array, "..."] | float,
) -> Array:
    """Evaluate a five-harmonic representation at relative sidereal phases."""

    phase = jnp.asarray(phase)

    def align(coefficient: Array) -> Array:
        coefficient = jnp.asarray(coefficient)
        for _ in range(phase.ndim - coefficient.ndim):
            coefficient = coefficient[..., None]
        return coefficient

    return (
        align(harmonics.constant)
        + align(harmonics.cos_one) * jnp.cos(phase)
        + align(harmonics.sin_one) * jnp.sin(phase)
        + align(harmonics.cos_two) * jnp.cos(2.0 * phase)
        + align(harmonics.sin_two) * jnp.sin(2.0 * phase)
    )


def collocate_antenna_harmonics(
    detector_tensor: Float[Array, "3 3"],
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    psi: Float[Array, "..."] | float,
    gmst_reference: Float[Array, "..."] | float,
) -> AntennaHarmonics:
    """Construct exact long-wavelength antenna harmonics from five values."""

    gmst_reference = jnp.asarray(gmst_reference)
    phases = SIDEREAL_COLLOCATION_PHASES.astype(gmst_reference.dtype)
    gmst = gmst_reference[..., None] + phases
    patterns = long_wavelength_antenna_patterns(
        detector_tensor,
        jnp.asarray(ra)[..., None],
        jnp.asarray(dec)[..., None],
        jnp.asarray(psi)[..., None],
        gmst,
    )
    return AntennaHarmonics(
        plus=sidereal_harmonics_from_collocation(patterns.plus),
        cross=sidereal_harmonics_from_collocation(patterns.cross),
    )


def collocate_delay_harmonics(
    detector_vertex: Float[Array, "3"],
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    gmst_reference: Float[Array, "..."] | float,
) -> SiderealHarmonics:
    """Construct the raw geocentric delay's sidereal representation."""

    gmst_reference = jnp.asarray(gmst_reference)
    phases = SIDEREAL_COLLOCATION_PHASES.astype(gmst_reference.dtype)
    delay = delay_from_geocenter(
        detector_vertex,
        jnp.asarray(ra)[..., None],
        jnp.asarray(dec)[..., None],
        gmst_reference[..., None] + phases,
    )
    return sidereal_harmonics_from_collocation(delay)


def combine_sidereal_sidebands(
    harmonics: SiderealHarmonics,
    shifted_carriers: Complex[Array, "... 5"],
) -> Complex[Array, "..."]:
    """Apply antenna harmonics to five frequency-shifted carrier values.

    The final dimension of ``shifted_carriers`` has the order
    ``[h(f), h(f+Fsid), h(f-Fsid), h(f+2Fsid), h(f-2Fsid)]``.  The formula uses
    the ``exp(-2 pi i f t)`` Fourier-transform convention.  It is exact for
    multiplying a carrier by the five-harmonic long-wavelength antenna
    pattern; it does not make the nonlinear detector-delay phasor finite
    harmonic.
    """

    shifted_carriers = jnp.asarray(shifted_carriers)
    if shifted_carriers.ndim == 0 or shifted_carriers.shape[-1] != 5:
        raise ValueError("shifted carriers must have a final dimension of length 5")

    carrier, plus_one, minus_one, plus_two, minus_two = (
        shifted_carriers[..., index] for index in range(5)
    )

    def align(coefficient: Array) -> Array:
        coefficient = jnp.asarray(coefficient)
        for _ in range(carrier.ndim - coefficient.ndim):
            coefficient = coefficient[..., None]
        return coefficient

    return (
        align(harmonics.constant) * carrier
        + 0.5 * align(harmonics.cos_one) * (plus_one + minus_one)
        + 0.5j * align(harmonics.sin_one) * (plus_one - minus_one)
        + 0.5 * align(harmonics.cos_two) * (plus_two + minus_two)
        + 0.5j * align(harmonics.sin_two) * (plus_two - minus_two)
    )


def detector_time_phasor(
    frequency: Float[Array, "..."],
    time_shift: Float[Array, "..."] | float,
) -> Complex[Array, "..."]:
    """Return ``exp(-2 pi i f time_shift)`` without a generic complex exp."""

    angle = -2.0 * jnp.pi * jnp.asarray(frequency) * jnp.asarray(time_shift)
    return jax.lax.complex(jnp.cos(angle), jnp.sin(angle))


def project_long_wavelength_at_gmst(
    frequency: Float[Array, "..."],
    h_plus: Complex[Array, "..."],
    h_cross: Complex[Array, "..."],
    detector_tensor: Float[Array, "3 3"],
    detector_vertex: Float[Array, "3"],
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    psi: Float[Array, "..."] | float,
    gmst: Float[Array, "..."] | float,
    geocentric_time_from_start: Float[Array, "..."] | float,
) -> Complex[Array, "..."]:
    """Project a waveform with response values evaluated at explicit GMST."""

    response = long_wavelength_response(
        detector_tensor, detector_vertex, ra, dec, psi, gmst
    )
    projected = response.antenna.plus * h_plus + response.antenna.cross * h_cross
    return projected * detector_time_phasor(
        frequency, jnp.asarray(geocentric_time_from_start) + response.delay
    )


def project_dynamic_long_wavelength(
    frequency: Float[Array, "..."],
    h_plus: Complex[Array, "..."],
    h_cross: Complex[Array, "..."],
    tau: Float[Array, "..."],
    detector_tensor: Float[Array, "3 3"],
    detector_vertex: Float[Array, "3"],
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    psi: Float[Array, "..."] | float,
    gmst_at_trigger: Float[Array, "..."] | float,
    trigger_time: Float[Array, "..."] | float,
    start_time: Float[Array, "..."] | float,
    t_c: Float[Array, "..."] | float,
) -> Complex[Array, "..."]:
    """Project with antenna and delay evaluated at mode emission time."""

    gmst = emission_gmst(gmst_at_trigger, t_c, tau)
    geocentric_time = (
        jnp.asarray(trigger_time) - jnp.asarray(start_time) + jnp.asarray(t_c)
    )
    return project_long_wavelength_at_gmst(
        frequency,
        h_plus,
        h_cross,
        detector_tensor,
        detector_vertex,
        ra,
        dec,
        psi,
        gmst,
        geocentric_time,
    )


def project_static_long_wavelength(
    frequency: Float[Array, "..."],
    h_plus: Complex[Array, "..."],
    h_cross: Complex[Array, "..."],
    detector_tensor: Float[Array, "3 3"],
    detector_vertex: Float[Array, "3"],
    ra: Float[Array, "..."] | float,
    dec: Float[Array, "..."] | float,
    psi: Float[Array, "..."] | float,
    gmst_at_trigger: Float[Array, "..."] | float,
    trigger_time: Float[Array, "..."] | float,
    start_time: Float[Array, "..."] | float,
    t_c: Float[Array, "..."] | float,
) -> Complex[Array, "..."]:
    """Project with the current response frozen at the trigger GMST."""

    geocentric_time = (
        jnp.asarray(trigger_time) - jnp.asarray(start_time) + jnp.asarray(t_c)
    )
    return project_long_wavelength_at_gmst(
        frequency,
        h_plus,
        h_cross,
        detector_tensor,
        detector_vertex,
        ra,
        dec,
        psi,
        gmst_at_trigger,
        geocentric_time,
    )
