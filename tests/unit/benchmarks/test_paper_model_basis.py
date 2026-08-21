import jax
import jax.numpy as jnp
import numpy as np
from ripplegw.constants import MTSUN
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    Phase_with_qm_correction as ripple_phase_with_qm_correction,
)
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    _amplitude_of as ripple_tidal_amplitude_of,
)
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    _get_merger_frequency,
)
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    _phase_of as ripple_tidal_phase_of,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD import Amp as ripple_amp
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD import Phase as ripple_phase
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_utils import get_coeffs
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomPv2 import PhenomPCoreTwistUp
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomPv2_utils import (
    ComputeNNLOanglecoeffs,
    SpinWeightedY,
    convert_spins,
    phP_get_transition_frequencies,
)

from benchmarks.device_parallel_nss.paper_model_basis import (
    REQUIRED_SIXTH_EXPONENTS,
    FrequencyPowerBasis,
    amp_basis,
    amplitude_of_basis,
    phase_basis,
    phase_of_basis,
    phase_with_qm_correction_basis,
)


def test_basis_powers_match_direct_pow() -> None:
    frequency = jnp.linspace(20.0, 2048.0, 4097)
    exponents = (-10, -7, -2, 2, 4, 6, 7)
    basis = FrequencyPowerBasis.build(frequency, exponents)

    for exponent in exponents:
        np.testing.assert_allclose(
            np.asarray(basis.sixth_power(exponent)),
            np.asarray(frequency ** (exponent / 6.0)),
            rtol=1e-15,
        )
    np.testing.assert_allclose(
        np.asarray(basis.log_f), np.asarray(jnp.log(frequency)), rtol=1e-15
    )
    np.testing.assert_array_equal(np.asarray(basis.f), np.asarray(frequency))
    for numerator, denominator in ((3, 4), (289, 150)):
        np.testing.assert_allclose(
            np.asarray(basis.rational_power(numerator, denominator)),
            np.asarray(frequency ** (numerator / denominator)),
            rtol=1e-15,
        )


def test_basis_build_accepts_traced_scalars() -> None:
    """The merger-frequency gradient path builds a scalar traced basis."""

    def phase_like(frequency_scalar):
        basis = FrequencyPowerBasis.build(frequency_scalar, REQUIRED_SIXTH_EXPONENTS)
        return basis.sixth_power(2) + basis.log_f

    value, gradient = jax.value_and_grad(phase_like)(25.0)

    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))


def _bns_intrinsics(rng: np.random.Generator) -> jax.Array:
    heavy = float(rng.uniform(1.3, 1.6))
    light = float(rng.uniform(1.1, heavy))
    chi_heavy = float(rng.uniform(-0.05, 0.05))
    chi_light = float(rng.uniform(-0.05, 0.05))
    return jnp.asarray([heavy, light, chi_heavy, chi_light])


def _phenomd_inputs(
    frequency: jax.Array, theta: jax.Array
) -> tuple[FrequencyPowerBasis, jax.Array, jax.Array, tuple]:
    coefficients = get_coeffs(theta)
    transition_frequencies = phP_get_transition_frequencies(
        theta, coefficients[5], coefficients[6], 0.0
    )
    total_mass_seconds = (theta[0] + theta[1]) * MTSUN
    basis = FrequencyPowerBasis.build(frequency, REQUIRED_SIXTH_EXPONENTS)
    return basis, total_mass_seconds, coefficients, transition_frequencies


def _jaxpr_equations(jaxpr):
    """Yield equations recursively through call/custom-JVP sub-jaxprs."""

    raw_jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    for equation in raw_jaxpr.eqns:
        yield equation
        for parameter in equation.params.values():
            values = parameter if isinstance(parameter, (list, tuple)) else (parameter,)
            for value in values:
                nested = getattr(value, "jaxpr", value)
                if hasattr(nested, "eqns"):
                    yield from _jaxpr_equations(nested)


def _basis_leaves(basis: FrequencyPowerBasis) -> list[jax.Array]:
    return [
        basis.f,
        basis.log_f,
        *basis._sixth_powers.values(),
        *basis._rational_powers.values(),
    ]


def _array_power_equations(jaxpr):
    return [
        equation
        for equation in _jaxpr_equations(jaxpr)
        if equation.primitive.name in {"pow", "integer_pow"}
        and equation.invars[0].aval.shape != ()
    ]


def _equation_source_files(equation) -> set[str]:
    codes, _ = equation.source_info.traceback.raw_frames()
    return {code.co_filename for code in codes}


def _assert_only_upstream_pv2_array_powers(jaxpr) -> None:
    """Every remaining array power must be stock ripple PhenomPv2 math.

    Before the twist-up geometry was vendored (Task 7), the only caller of
    ripple's ``WignerdCoefficients`` (which does array-shaped ``s**2`` /
    ``(...)**0.5`` powers on its ``v`` argument) was ripple's own
    ``PhenomPCoreTwistUp``, so every array-power equation's traceback stayed
    entirely inside ``ripplegw/waveforms/cbc/IMRPhenomD/IMRPhenomPv2*``.
    ``phenomp_twist_up_geometry_basis`` (this benchmark's
    ``paper_model_basis.py``) now calls that same, unmodified
    ``WignerdCoefficients`` directly, so the frame attribution moved -- the
    traceback legitimately gains a ``paper_model_basis.py`` frame for the
    *same* stock math, rather than that math changing. The invariant this
    guards is therefore narrower than before: every array-power equation must
    still trace through ripple's upstream PhenomPv2 code (``IMRPhenomPv2.py``
    or ``IMRPhenomPv2_utils.py``, where ``WignerdCoefficients`` lives), and if
    it also passes through ``paper_model_basis.py`` that is only acceptable
    because the ripple frame is present too -- i.e. the power lives inside
    vendored-and-called stock code, not in basis-module code of its own.
    """

    array_powers = _array_power_equations(jaxpr)
    assert array_powers
    for equation in array_powers:
        source_files = _equation_source_files(equation)
        traces_through_ripple_pv2 = any(
            "ripplegw/waveforms/cbc/IMRPhenomD/IMRPhenomPv2" in path
            for path in source_files
        )
        assert traces_through_ripple_pv2
        traces_through_paper_model_basis = any(
            "/benchmarks/device_parallel_nss/paper_model_basis.py" in path
            for path in source_files
        )
        if traces_through_paper_model_basis:
            # Only acceptable when the same equation also traces through
            # ripple's upstream PhenomPv2 code, i.e. the power is stock
            # WignerdCoefficients math reached via the vendored geometry
            # function -- not a power the basis module introduces itself.
            assert traces_through_ripple_pv2


def test_phenomd_amp_and_phase_parity_against_ripple() -> None:
    frequency = jnp.linspace(20.0, 2048.0, 8193)
    rng = np.random.default_rng(11)

    for _ in range(5):
        theta = _bns_intrinsics(rng)
        basis, M_s, coefficients, transition_frequencies = _phenomd_inputs(
            frequency, theta
        )

        # Pinned ripple 0.3.0 takes physical Hz here and forms f*M_s itself.
        reference_amplitude = ripple_amp(
            frequency,
            theta,
            coefficients,
            transition_frequencies,
            D=40.0,
        )
        reference_phase = ripple_phase(
            frequency, theta, coefficients, transition_frequencies
        )
        actual_amplitude = amp_basis(
            basis,
            M_s,
            theta,
            coefficients,
            transition_frequencies,
            D=40.0,
        )
        actual_phase = phase_basis(
            basis, M_s, theta, coefficients, transition_frequencies
        )

        amplitude_scale = float(jnp.max(jnp.abs(reference_amplitude)))
        np.testing.assert_allclose(
            np.asarray(actual_amplitude),
            np.asarray(reference_amplitude),
            rtol=1e-10,
            atol=1e-10 * amplitude_scale,
        )
        np.testing.assert_allclose(
            np.asarray(actual_phase), np.asarray(reference_phase), rtol=1e-10
        )


def test_vendored_phenomd_functions_emit_no_array_pow() -> None:
    frequency = jnp.linspace(20.0, 2048.0, 4097)
    theta = _bns_intrinsics(np.random.default_rng(3))
    basis, M_s, coefficients, transition_frequencies = _phenomd_inputs(frequency, theta)

    jaxpr = jax.make_jaxpr(
        lambda: (
            amp_basis(
                basis,
                M_s,
                theta,
                coefficients,
                transition_frequencies,
                D=40.0,
            ),
            phase_basis(basis, M_s, theta, coefficients, transition_frequencies),
        )
    )()
    array_powers = _array_power_equations(jaxpr)

    assert array_powers == []


def test_vendored_tidal_carrier_emits_no_array_pow() -> None:
    """Cover the cached PhenomD+NRTidal layer (Pv2 twist-up is upstream code)."""

    frequency = jnp.linspace(20.0, 2048.0, 4097)
    theta_bbh = _bns_intrinsics(np.random.default_rng(7))
    basis, M_s, coefficients, transition_frequencies = _phenomd_inputs(
        frequency, theta_bbh
    )
    theta_tidal = jnp.concatenate((theta_bbh, jnp.asarray([300.0, 500.0])))

    def carrier_layers():
        bbh_amplitude = amp_basis(
            basis,
            M_s,
            theta_bbh,
            coefficients,
            transition_frequencies,
            D=40.0,
        )
        bbh_phase = phase_with_qm_correction_basis(
            basis,
            M_s,
            theta_bbh,
            theta_tidal,
            coefficients,
            transition_frequencies,
        )
        amplitude = amplitude_of_basis(
            basis,
            M_s,
            theta_tidal,
            jnp.asarray([40.0, 0.0, 0.3]),
            bbh_amplitude,
        )
        phase = phase_of_basis(basis, M_s, theta_tidal, bbh_phase)
        return amplitude, phase

    array_powers = _array_power_equations(jax.make_jaxpr(carrier_layers)())

    assert array_powers == []


_PHENOMD_POLARIZATION_NORM = 2.0 * jnp.sqrt(5.0 / (64.0 * jnp.pi))


def _reference_carrier_and_geometry(
    frequency: jax.Array,
    theta: jax.Array,
    f_ref: float,
    *,
    no_taper: bool,
):
    """Frozen pre-Task-6 generator body using untouched ripple primitives."""

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
        bbh_intrinsic, coefficients[5], coefficients[6], chi_p
    )

    bbh_amplitude = ripple_amp(
        frequency,
        bbh_intrinsic,
        coefficients,
        transition_frequencies,
        D=distance,
    )
    corrected_amplitude = ripple_tidal_amplitude_of(
        frequency,
        tidal_intrinsic,
        jnp.array([distance, 0.0, phase_c]),
        bbh_amplitude,
        no_taper=no_taper,
    )
    corrected_amplitude /= _PHENOMD_POLARIZATION_NORM

    def carrier_phase(frequency_value):
        bbh_phase = ripple_phase_with_qm_correction(
            frequency_value,
            bbh_intrinsic,
            tidal_intrinsic,
            coefficients,
            transition_frequencies,
        )
        return ripple_tidal_phase_of(frequency_value, tidal_intrinsic, bbh_phase) + (
            2.0 * phi_aligned
        )

    phase = carrier_phase(frequency)
    carrier = corrected_amplitude * (jnp.cos(phase) + 1.0j * jnp.sin(phase))

    merger_frequency = _get_merger_frequency(tidal_intrinsic)
    time_shift = jax.grad(carrier_phase)(merger_frequency) / (2.0 * jnp.pi)
    angle = (-2.0 * jnp.pi) * frequency * time_shift
    carrier *= jax.lax.complex(jnp.cos(angle), jnp.sin(angle))

    return (
        carrier,
        symmetric_mass_ratio,
        chi_light_l,
        chi_heavy_l,
        chi_p,
        angle_coefficients,
        harmonics,
        alpha_offset - alpha_0,
        epsilon_offset,
        polarization_rotation,
    )


def _reference_hphc(
    frequency: jax.Array,
    theta: jax.Array,
    f_ref: float,
    *,
    no_taper: bool = False,
) -> tuple[jax.Array, jax.Array]:
    (
        carrier,
        eta,
        chi_light_l,
        chi_heavy_l,
        chi_p,
        angle_coefficients,
        harmonics,
        alpha_offset,
        epsilon_offset,
        polarization_rotation,
    ) = _reference_carrier_and_geometry(frequency, theta, f_ref, no_taper=no_taper)

    primary_mass, secondary_mass = Mc_eta_to_ms(theta[:2])
    total_mass = primary_mass + secondary_mass
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
        alpha_offset,
        epsilon_offset,
    )

    cosine = jnp.cos(2.0 * polarization_rotation)
    sine = jnp.sin(2.0 * polarization_rotation)
    return cosine * hp + sine * hc, cosine * hc - sine * hp


def _example_waveform_params() -> dict[str, float]:
    return {
        "M_c": 1.2,
        "eta": 0.8 / 1.8**2,
        "s1_x": 0.012,
        "s1_y": -0.009,
        "s1_z": 0.020,
        "s2_x": -0.008,
        "s2_y": 0.006,
        "s2_z": -0.015,
        "lambda_1": 300.0,
        "lambda_2": 500.0,
        "d_L": 40.0,
        "phase_c": 0.3,
        "iota": 0.8,
    }


def test_full_waveform_parity_ripple_vs_basis() -> None:
    from benchmarks.device_parallel_nss.paper_model import (
        gen_imrphenompv2_nrtidalv2_hphc,
    )

    frequency = jnp.linspace(20.0, 2048.0, 8193)
    rng = np.random.default_rng(17)
    for _ in range(3):
        theta = jnp.asarray(
            [
                rng.uniform(1.18, 1.21),
                rng.uniform(0.2, 0.2499),
                *rng.uniform(-0.03, 0.03, 6),
                rng.uniform(0.0, 3000.0),
                rng.uniform(0.0, 3000.0),
                rng.uniform(10.0, 60.0),
                rng.uniform(0.0, 6.28),
                rng.uniform(0.0, 3.14),
            ]
        )
        hp_actual, hc_actual = gen_imrphenompv2_nrtidalv2_hphc(frequency, theta, 20.0)
        hp_reference, hc_reference = _reference_hphc(frequency, theta, 20.0)
        scale = float(jnp.max(jnp.abs(hp_reference)))

        np.testing.assert_allclose(
            np.asarray(hp_actual),
            np.asarray(hp_reference),
            rtol=1e-9,
            atol=1e-10 * scale,
        )
        np.testing.assert_allclose(
            np.asarray(hc_actual),
            np.asarray(hc_reference),
            rtol=1e-9,
            atol=1e-10 * scale,
        )


def test_waveform_class_memoizes_basis_per_frequency_grid() -> None:
    from benchmarks.device_parallel_nss.paper_model import (
        RippleIMRPhenomPv2NRTidalv2,
    )

    waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)
    frequency = jnp.linspace(20.0, 1024.0, 2049)
    parameters = _example_waveform_params()

    first = waveform(frequency, parameters)
    assert len(waveform._basis_memo) == 1
    second = waveform(frequency, parameters)
    assert len(waveform._basis_memo) == 1
    for polarization in first:
        np.testing.assert_array_equal(
            np.asarray(first[polarization]), np.asarray(second[polarization])
        )


def test_waveform_jit_first_closed_grid_memo_is_concrete_and_reusable() -> None:
    """A closed grid first seen under JIT must not leak staged basis values."""

    from benchmarks.device_parallel_nss.paper_model import (
        RippleIMRPhenomPv2NRTidalv2,
    )

    waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)
    frequency = jnp.linspace(20.0, 1024.0, 257)
    parameters = _example_waveform_params()

    first_transform = jax.jit(lambda values: waveform(frequency, values))
    first = first_transform(parameters)
    jax.block_until_ready(first["p"])

    assert len(waveform._basis_memo) == 1
    candidate, basis = waveform._basis_memo[0]
    assert candidate is frequency
    assert not any(isinstance(leaf, jax.core.Tracer) for leaf in _basis_leaves(basis))

    eager = waveform(frequency, parameters)
    independent_transform = jax.jit(lambda values: waveform(frequency, values))
    retraced = independent_transform(parameters)
    jax.block_until_ready(retraced["p"])
    for polarization in first:
        np.testing.assert_allclose(
            np.asarray(eager[polarization]),
            np.asarray(first[polarization]),
            rtol=1e-10,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(retraced[polarization]),
            np.asarray(first[polarization]),
            rtol=1e-10,
            atol=0.0,
        )

    cached_jaxpr = jax.make_jaxpr(lambda values: waveform(frequency, values))(
        parameters
    )
    _assert_only_upstream_pv2_array_powers(cached_jaxpr)

    # A separate waveform proves the first transformation itself also hoists
    # the closed-grid basis rather than relying on a previously eager memo.
    first_trace_waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)
    first_trace_jaxpr = jax.make_jaxpr(
        lambda values: first_trace_waveform(frequency, values)
    )(parameters)
    assert not any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in _basis_leaves(first_trace_waveform._basis_memo[0][1])
    )
    _assert_only_upstream_pv2_array_powers(first_trace_jaxpr)


def test_waveform_traced_frequency_argument_is_not_memoized() -> None:
    from benchmarks.device_parallel_nss.paper_model import (
        RippleIMRPhenomPv2NRTidalv2,
    )

    waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)
    frequency = jnp.linspace(20.0, 1024.0, 65)
    result = jax.jit(waveform)(frequency, _example_waveform_params())
    jax.block_until_ready(result["p"])

    assert waveform._basis_memo == []
