"""Paper-compatible precessing-tidal waveform for benchmark workloads.

The public ``ripplegw==0.3.0`` release exposes ``IMRPhenomPv2`` and
``IMRPhenomD_NRTidalv2`` separately, while Yallup et al. use the combined
``IMRPhenomPv2_NRTidalv2`` approximant.  This module composes Ripple's public
Pv2 carrier and NRTidalv2 primitives in the same order as LALSimulation:

1. construct the aligned-spin IMRPhenomD carrier, including neutron-star
   quadrupole-monopole terms;
2. add the NRTidalv2 amplitude, phase, and higher-order spin corrections;
3. apply the tidal merger taper and coalescence-time alignment; and
4. twist the corrected carrier into the precessing plus/cross polarizations.

It deliberately lives with the benchmark rather than Jim's public waveform
registry.  The implementation imports Ripple internals pinned by the benchmark
environment and must remain guarded by the independent LALSimulation parity
test before it is used for performance claims.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float
from ripplegw.constants import MTSUN
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.interfaces import DistanceScaledWaveform, FrequencyDomainWaveform
from ripplegw.typing import FloatLike
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    _get_merger_frequency,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD import get_IIb_raw_phase
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_utils import (
    get_coeffs,
    get_transition_frequencies,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomPv2 import PhenomPCoreTwistUp
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomPv2_utils import (
    ComputeNNLOanglecoeffs,
    SpinWeightedY,
    convert_spins,
    phP_get_transition_frequencies,
)

if __package__:
    from .paper_model_basis import (
        REQUIRED_SIXTH_EXPONENTS,
        FrequencyPowerBasis,
        amp_basis,
        amplitude_of_basis,
        phase_of_basis,
        phase_with_qm_correction_basis,
        phenomp_core_twist_up_basis,
        phenomp_twist_up_geometry_basis,
    )
else:  # pragma: no cover - exercised by the frozen benchmark harness
    from paper_model_basis import (
        REQUIRED_SIXTH_EXPONENTS,
        FrequencyPowerBasis,
        amp_basis,
        amplitude_of_basis,
        phase_of_basis,
        phase_with_qm_correction_basis,
        phenomp_core_twist_up_basis,
        phenomp_twist_up_geometry_basis,
    )

_PHENOMD_POLARIZATION_NORM = 2.0 * jnp.sqrt(5.0 / (64.0 * jnp.pi))
_TIME_ANCHORS = frozenset(("nrtidal-merger", "imrphenomd"))


def _apply_time_shift(
    carrier: Complex[Array, " n_freq"],
    frequency: Float[Array, " n_freq"],
    time_shift: FloatLike,
) -> Complex[Array, " n_freq"]:
    """Rotate by ``exp(-2*pi*i*f*dt)`` without a generic complex exp."""

    angle = (-2.0 * jnp.pi) * frequency * time_shift
    return carrier * jax.lax.complex(jnp.cos(angle), jnp.sin(angle))


def _phenomd_peak_time_shift(
    M_s: FloatLike,
    bbh_intrinsic: Float[Array, 4],
    coefficients: Float[Array, 19],
) -> FloatLike:
    """Return Ripple's IMRPhenomD peak-alignment shift in seconds.

    This is the affine time convention inherited by the coauthor combined
    Pv2+NRTidal implementation: the derivative is taken from the raw
    IMRPhenomD merger-ringdown phase at the BBH amplitude transition.  It is
    intentionally separate from the NRTidal merger derivative used by this
    benchmark's original reconstruction.
    """

    _, _, _, f4, f_rd, f_damp = get_transition_frequencies(
        bbh_intrinsic,
        coefficients[5],
        coefficients[6],
    )
    dimensionless_slope = jax.grad(get_IIb_raw_phase)(
        f4 * M_s,
        bbh_intrinsic,
        coefficients,
        f_rd,
        f_damp,
    )
    return -(M_s * dimensionless_slope) / (2.0 * jnp.pi)


def _carrier_and_geometry(
    frequency: Float[Array, " n_freq"],
    theta: Float[Array, 13],
    f_ref: float,
    *,
    no_taper: bool,
    time_anchor: str,
    basis: FrequencyPowerBasis | None = None,
) -> tuple[
    Complex[Array, " n_freq"],
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
    dict[str, FloatLike],
    list[Complex],
    FloatLike,
    FloatLike,
    FloatLike,
    FloatLike,
]:
    """Build the corrected co-precessing carrier and twist-up geometry."""

    if basis is None:
        basis = FrequencyPowerBasis.build(frequency, REQUIRED_SIXTH_EXPONENTS)

    (
        chirp_mass,
        eta,
        s1_x,
        s1_y,
        s1_z,
        s2_x,
        s2_y,
        s2_z,
        lambda_1,
        lambda_2,
        distance,
        phase_c,
        inclination,
    ) = theta
    primary_mass, secondary_mass = Mc_eta_to_ms(jnp.array([chirp_mass, eta]))

    # Ripple's Pv2 internals follow LAL's PhenomP convention: body 1 is the
    # lighter object and body 2 the heavier object.  The user-facing CBC
    # convention is the reverse, so swap masses, spins, and tides together.
    light_mass, heavy_mass = secondary_mass, primary_mass
    light_spin = (s2_x, s2_y, s2_z)
    heavy_spin = (s1_x, s1_y, s1_z)
    light_lambda, heavy_lambda = lambda_2, lambda_1

    (
        chi_light_l,
        chi_heavy_l,
        chi_p,
        theta_jn,
        alpha_0,
        phi_aligned,
        polarization_rotation,
    ) = convert_spins(
        light_mass,
        heavy_mass,
        f_ref,
        phase_c,
        inclination,
        *light_spin,
        *heavy_spin,
    )

    mass_ratio = heavy_mass / light_mass
    total_mass = light_mass + heavy_mass
    M_s = total_mass * MTSUN
    chi_eff = (light_mass * chi_light_l + heavy_mass * chi_heavy_l) / total_mass
    chi_l = (1.0 + mass_ratio) / mass_ratio * chi_eff
    symmetric_mass_ratio = light_mass * heavy_mass / total_mass**2
    pi_m = jnp.pi * total_mass * MTSUN

    angle_coefficients = ComputeNNLOanglecoeffs(mass_ratio, chi_l, chi_p)
    omega_ref = pi_m * f_ref
    omega_ref_cuberoot = omega_ref ** (1.0 / 3.0)
    alpha_offset = (
        angle_coefficients["alphacoeff1"] / omega_ref
        + angle_coefficients["alphacoeff2"] / omega_ref_cuberoot**2
        + angle_coefficients["alphacoeff3"] / omega_ref_cuberoot
        + angle_coefficients["alphacoeff4"] * jnp.log(omega_ref)
        + angle_coefficients["alphacoeff5"] * omega_ref_cuberoot
    )
    epsilon_offset = (
        angle_coefficients["epsiloncoeff1"] / omega_ref
        + angle_coefficients["epsiloncoeff2"] / omega_ref_cuberoot**2
        + angle_coefficients["epsiloncoeff3"] / omega_ref_cuberoot
        + angle_coefficients["epsiloncoeff4"] * jnp.log(omega_ref)
        + angle_coefficients["epsiloncoeff5"] * omega_ref_cuberoot
    )
    harmonics = [SpinWeightedY(theta_jn, 0.0, -2, 2, mode) for mode in range(-2, 3)]

    # IMRPhenomD expects the ordinary primary-first mass convention.
    bbh_intrinsic = jnp.array([heavy_mass, light_mass, chi_heavy_l, chi_light_l])
    tidal_intrinsic = jnp.array(
        [
            heavy_mass,
            light_mass,
            chi_heavy_l,
            chi_light_l,
            heavy_lambda,
            light_lambda,
        ]
    )
    coefficients = get_coeffs(bbh_intrinsic)
    transition_frequencies = phP_get_transition_frequencies(
        bbh_intrinsic,
        coefficients[5],
        coefficients[6],
        chi_p,
    )

    bbh_amplitude = amp_basis(
        basis,
        M_s,
        bbh_intrinsic,
        coefficients,
        transition_frequencies,
        D=distance,
    )
    corrected_amplitude = amplitude_of_basis(
        basis,
        M_s,
        tidal_intrinsic,
        jnp.array([distance, 0.0, phase_c]),
        bbh_amplitude,
        no_taper=no_taper,
    )
    corrected_amplitude /= _PHENOMD_POLARIZATION_NORM

    def carrier_phase_for_basis(candidate_basis):
        bbh_phase = phase_with_qm_correction_basis(
            candidate_basis,
            M_s,
            bbh_intrinsic,
            tidal_intrinsic,
            coefficients,
            transition_frequencies,
        )
        # LAL subtracts twice the orbital phase.  Ripple's ``convert_spins``
        # returns that orbital phase, while its Pv2 implementation stores the
        # doubled value in ``phic`` before assembling the carrier.
        return phase_of_basis(candidate_basis, M_s, tidal_intrinsic, bbh_phase) + (
            2.0 * phi_aligned
        )

    def carrier_phase(frequency_value):
        # The merger-alignment derivative is scalar and must remain with
        # respect to physical frequency. Building a scalar basis here keeps
        # that autodiff path independent of the cached vector basis.
        scalar_basis = FrequencyPowerBasis.build(
            frequency_value, REQUIRED_SIXTH_EXPONENTS
        )
        return carrier_phase_for_basis(scalar_basis)

    phase = carrier_phase_for_basis(basis)
    carrier = corrected_amplitude * (jnp.cos(phase) + 1.0j * jnp.sin(phase))

    if time_anchor == "nrtidal-merger":
        # LAL aligns the tidal waveform at the NRTidal merger frequency, not
        # the BBH ringdown frequency used by plain Pv2. Autodiff is the
        # differentiable analogue of LAL's local phase-spline derivative.
        merger_frequency = _get_merger_frequency(tidal_intrinsic)
        time_shift = jax.grad(carrier_phase)(merger_frequency) / (2.0 * jnp.pi)
    elif time_anchor == "imrphenomd":
        time_shift = _phenomd_peak_time_shift(M_s, bbh_intrinsic, coefficients)
    else:  # Defensive validation for callers of the functional interface.
        raise ValueError(
            f"unknown time anchor {time_anchor!r}; expected one of "
            f"{sorted(_TIME_ANCHORS)}"
        )
    carrier = _apply_time_shift(carrier, frequency, time_shift)

    return (
        carrier,
        symmetric_mass_ratio,
        chi_light_l,
        chi_heavy_l,
        chi_p,
        angle_coefficients,
        harmonics,
        alpha_offset,
        alpha_0,
        epsilon_offset,
        polarization_rotation,
    )


def gen_imrphenompv2_nrtidalv2_hphc(
    frequency: Float[Array, " n_freq"],
    theta: Float[Array, 13],
    f_ref: float,
    *,
    no_taper: bool = False,
    time_anchor: str = "nrtidal-merger",
    basis: FrequencyPowerBasis | None = None,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """Generate precessing NRTidalv2 plus/cross polarizations."""

    (
        carrier,
        eta,
        chi_light_l,
        chi_heavy_l,
        chi_p,
        angle_coefficients,
        harmonics,
        alpha_offset,
        alpha_0,
        epsilon_offset,
        polarization_rotation,
    ) = _carrier_and_geometry(
        frequency,
        theta,
        f_ref,
        no_taper=no_taper,
        time_anchor=time_anchor,
        basis=basis,
    )

    primary_mass, secondary_mass = Mc_eta_to_ms(theta[:2])
    total_mass = primary_mass + secondary_mass
    if basis is None:
        # Scalar jax.grad merger-alignment path (see _phenomd_peak_time_shift
        # and the "nrtidal-merger" branch of _carrier_and_geometry): keep the
        # stock ripple TwistUp verbatim rather than standing up a basis for a
        # single frequency value.
        hp, hc = PhenomPCoreTwistUp(
            frequency,
            carrier,
            eta,
            chi_light_l,
            chi_heavy_l,
            chi_p,
            total_mass,
            angle_coefficients,
            harmonics,
            alpha_offset - alpha_0,
            epsilon_offset,
        )
    else:
        geometry = phenomp_twist_up_geometry_basis(
            basis,
            total_mass,
            eta,
            chi_light_l,
            chi_heavy_l,
            chi_p,
            angle_coefficients,
        )
        hp, hc = phenomp_core_twist_up_basis(
            carrier,
            *geometry,
            harmonics,
            alpha_offset - alpha_0,
            epsilon_offset,
        )

    cosine = jnp.cos(2.0 * polarization_rotation)
    sine = jnp.sin(2.0 * polarization_rotation)
    return cosine * hp + sine * hc, cosine * hc - sine * hp


class RippleIMRPhenomPv2NRTidalv2(
    FrequencyDomainWaveform,
    DistanceScaledWaveform,
):
    """Benchmark-local JAX implementation of ``IMRPhenomPv2_NRTidalv2``."""

    def __init__(
        self,
        f_ref: float = 20.0,
        *,
        no_taper: bool = False,
        time_anchor: str = "nrtidal-merger",
    ) -> None:
        if time_anchor not in _TIME_ANCHORS:
            raise ValueError(
                f"unknown time anchor {time_anchor!r}; expected one of "
                f"{sorted(_TIME_ANCHORS)}"
            )
        self.f_ref = f_ref
        self.no_taper = no_taper
        self.time_anchor = time_anchor
        self._basis_memo: list[tuple[Array, FrequencyPowerBasis]] = []

    def _basis_for(self, frequency: Array) -> FrequencyPowerBasis | None:
        """Reuse powers for concrete grids; never retain transient tracers."""

        if isinstance(frequency, jax.core.Tracer):
            return None
        for candidate, basis in self._basis_memo:
            if candidate is frequency:
                return basis

        # A concrete grid may first reach the waveform while an outer JAX
        # transform is staging parameter-dependent work. Without an explicit
        # compile-time region, the basis operations inherit that ambient trace
        # and memoizing their results leaks DynamicJaxprTracers. Closed-over
        # grids are constants, so force their one-time powers to be evaluated
        # now; tracer arguments continue to use the inline path above.
        with jax.ensure_compile_time_eval():
            basis = FrequencyPowerBasis.build(frequency, REQUIRED_SIXTH_EXPONENTS)

        basis_leaves = jax.tree_util.tree_leaves(
            (
                basis.f,
                basis.log_f,
                basis._sixth_powers,
                basis._rational_powers,
            )
        )
        if any(isinstance(leaf, jax.core.Tracer) for leaf in basis_leaves):
            return None
        self._basis_memo.append((frequency, basis))
        return basis

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
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

    @property
    def cacheable_parameter_names(self) -> frozenset[str]:
        """Parameters reconstructed from the cached carrier and harmonics."""

        return frozenset(("d_L", "iota"))

    def build_waveform_cache(
        self,
        frequency: Float[Array, " n_freq"],
        params: Mapping[str, FloatLike],
    ) -> dict[str, object]:
        """Cache the expensive carrier and twist-up geometry at unit distance.

        Inclination affects only the cheap line-of-sight geometry and twist-up
        once the intrinsic precessing carrier has been constructed.  Keeping
        the raw alpha offset (rather than the inclination-specific difference)
        lets :meth:`waveform_from_cache` synthesize a new orientation exactly.

        In addition to the carrier, this also caches the four frequency
        arrays consumed by :func:`phenomp_core_twist_up_basis` --
        ``cexp_i_alpha_series``, ``cexp_m2i_epsilon_series``, ``cBetah``, and
        ``sBetah`` -- so that :meth:`waveform_from_cache` performs zero
        frequency-dependent transcendentals on a cache hit.  These arrays
        depend only on intrinsics (mass ratio, aligned/precessing spins,
        total mass), never on ``iota`` or ``d_L``.  At 259,584 bins they add
        ~12.5 MB per cached waveform (two complex128 arrays plus two
        float64 arrays); at 16 lanes per device that is ~200 MB, well inside
        the ~6 GB memory envelope.
        """

        theta = jnp.array([params[name] for name in self.parameter_names])
        theta = theta.at[10].set(1.0)
        basis = self._basis_for(frequency)
        (
            carrier,
            eta,
            chi_light_l,
            chi_heavy_l,
            chi_p,
            angle_coefficients,
            _harmonics,
            alpha_offset,
            _alpha_0,
            epsilon_offset,
            _polarization_rotation,
        ) = _carrier_and_geometry(
            frequency,
            theta,
            self.f_ref,
            no_taper=self.no_taper,
            time_anchor=self.time_anchor,
            basis=basis,
        )
        primary_mass, secondary_mass = Mc_eta_to_ms(theta[:2])
        total_mass = primary_mass + secondary_mass
        geometry_basis = basis
        if geometry_basis is None:
            # Traced frequency: _basis_for declines to memoize it, but the
            # vectorized geometry math is still correct and preferable to a
            # third, scalar-transcendental reimplementation. This mirrors
            # _carrier_and_geometry's own `basis is None` fallback (its one
            # local, unmemoized `FrequencyPowerBasis.build` call above).
            geometry_basis = FrequencyPowerBasis.build(
                frequency, REQUIRED_SIXTH_EXPONENTS
            )
        (
            cexp_i_alpha_series,
            cexp_m2i_epsilon_series,
            cBetah,
            sBetah,
        ) = phenomp_twist_up_geometry_basis(
            geometry_basis,
            total_mass,
            eta,
            chi_light_l,
            chi_heavy_l,
            chi_p,
            angle_coefficients,
        )
        return {
            "carrier_at_unit_distance": carrier,
            "eta": eta,
            "chi_light_l": chi_light_l,
            "chi_heavy_l": chi_heavy_l,
            "chi_p": chi_p,
            "angle_coefficients": angle_coefficients,
            "alpha_offset": alpha_offset,
            "epsilon_offset": epsilon_offset,
            "cexp_i_alpha_series": cexp_i_alpha_series,
            "cexp_m2i_epsilon_series": cexp_m2i_epsilon_series,
            "cBetah": cBetah,
            "sBetah": sBetah,
        }

    def waveform_from_cache(
        self,
        frequency: Float[Array, " n_freq"],
        params: Mapping[str, FloatLike],
        cache: Mapping[str, object],
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Reconstruct plus/cross polarizations for new ``iota`` and ``d_L``.

        Every frequency-dependent quantity needed by the twist-up --
        ``cexp_i_alpha_series``, ``cexp_m2i_epsilon_series``, ``cBetah``, and
        ``sBetah`` -- was already computed once in
        :meth:`build_waveform_cache`.  Only the scalar
        ``convert_spins``/``SpinWeightedY``/polarization-rotation work (which
        depends on the new ``iota``) and the ``1/d_L`` carrier rescale happen
        here, so a cache hit performs no frequency-dependent transcendentals.
        """

        primary_mass, secondary_mass = Mc_eta_to_ms(
            jnp.array([params["M_c"], params["eta"]])
        )
        converted = convert_spins(
            secondary_mass,
            primary_mass,
            self.f_ref,
            params["phase_c"],
            params["iota"],
            params["s2_x"],
            params["s2_y"],
            params["s2_z"],
            params["s1_x"],
            params["s1_y"],
            params["s1_z"],
        )
        theta_jn, alpha_0, polarization_rotation = (
            converted[3],
            converted[4],
            converted[6],
        )
        harmonics = [SpinWeightedY(theta_jn, 0.0, -2, 2, mode) for mode in range(-2, 3)]
        carrier = cache["carrier_at_unit_distance"] * (1.0 / params["d_L"])
        hp, hc = phenomp_core_twist_up_basis(
            carrier,
            cache["cexp_i_alpha_series"],
            cache["cexp_m2i_epsilon_series"],
            cache["cBetah"],
            cache["sBetah"],
            harmonics,
            cache["alpha_offset"] - alpha_0,
            cache["epsilon_offset"],
        )
        cosine = jnp.cos(2.0 * polarization_rotation)
        sine = jnp.sin(2.0 * polarization_rotation)
        return {
            "p": cosine * hp + sine * hc,
            "c": cosine * hc - sine * hp,
        }

    def __call__(
        self,
        frequency: Float[Array, " n_freq"],
        params: Mapping[str, FloatLike],
    ) -> dict[str, Complex[Array, " n_freq"]]:
        theta = jnp.array([params[name] for name in self.parameter_names])
        hp, hc = gen_imrphenompv2_nrtidalv2_hphc(
            frequency,
            theta,
            self.f_ref,
            no_taper=self.no_taper,
            time_anchor=self.time_anchor,
            basis=self._basis_for(frequency),
        )
        return {"p": hp, "c": hc}

    def __repr__(self) -> str:
        return (
            "RippleIMRPhenomPv2NRTidalv2("
            f"f_ref={self.f_ref}, no_taper={self.no_taper}, "
            f"time_anchor={self.time_anchor!r})"
        )


__all__ = [
    "RippleIMRPhenomPv2NRTidalv2",
    "gen_imrphenompv2_nrtidalv2_hphc",
]
