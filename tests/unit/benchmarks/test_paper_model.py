import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.device_parallel_nss.paper_model import (
    RippleIMRPhenomPv2NRTidalv2,
    _apply_time_shift,
)

EXPECTED_PARAMETER_NAMES = (
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

LALSIMULATION_PARITY_RTOL = 3e-4

PAPER_PARAMETERS = {
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

MASS_RATIO_EDGE_PARAMETERS = {
    "M_c": 1.18,
    "eta": 0.125 / 1.125**2,
    "s1_x": 0.030,
    "s1_y": -0.020,
    "s1_z": 0.030,
    "s2_x": -0.010,
    "s2_y": 0.040,
    "s2_z": -0.020,
    "lambda_1": 5000.0,
    "lambda_2": 100.0,
    "d_L": 75.0,
    "phase_c": 1.7,
    "iota": 1.4,
}


def _relative_waveform_change(
    baseline: dict[str, jax.Array], changed: dict[str, jax.Array]
) -> float:
    baseline_vector = jnp.concatenate([baseline["p"], baseline["c"]])
    changed_vector = jnp.concatenate([changed["p"], changed["c"]])
    return float(
        jnp.linalg.norm(changed_vector - baseline_vector)
        / jnp.linalg.norm(baseline_vector)
    )


def test_merger_alignment_phasor_matches_complex_exp_reference() -> None:
    """Freeze the pre-change ``carrier * exp(-2*pi*i*f*dt)`` formula."""

    rng = np.random.default_rng(9)
    frequency = jnp.linspace(20.0, 1024.0, 2049)
    carrier = jnp.asarray(
        rng.normal(size=frequency.size) + 1j * rng.normal(size=frequency.size)
    )
    time_shift = 0.0137
    reference = carrier * jnp.exp(-2.0j * jnp.pi * frequency * time_shift)

    result = _apply_time_shift(carrier, frequency, time_shift)

    np.testing.assert_allclose(
        np.asarray(result), np.asarray(reference), rtol=1e-13, atol=1e-14
    )


def test_parameter_schema_matches_the_paper_model() -> None:
    waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)

    assert waveform.parameter_names == EXPECTED_PARAMETER_NAMES


def test_waveform_output_is_finite_and_jittable() -> None:
    waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)
    frequencies = jnp.linspace(20.0, 1024.0, 257)

    eager = waveform(frequencies, PAPER_PARAMETERS)
    compiled = jax.jit(waveform)(frequencies, PAPER_PARAMETERS)

    assert set(eager) == {"p", "c"}
    for polarization in ("p", "c"):
        assert eager[polarization].shape == frequencies.shape
        assert jnp.iscomplexobj(eager[polarization])
        assert jnp.all(jnp.isfinite(eager[polarization]))
        assert jnp.any(jnp.abs(eager[polarization]) > 0.0)
        np.testing.assert_allclose(
            compiled[polarization], eager[polarization], rtol=1e-10, atol=0.0
        )


def test_waveform_scales_inversely_with_luminosity_distance() -> None:
    waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)
    frequencies = jnp.linspace(20.0, 1024.0, 257)
    nearby = waveform(frequencies, {**PAPER_PARAMETERS, "d_L": 40.0})
    distant = waveform(frequencies, {**PAPER_PARAMETERS, "d_L": 160.0})

    for polarization in ("p", "c"):
        np.testing.assert_allclose(
            nearby[polarization],
            4.0 * distant[polarization],
            rtol=1e-12,
            atol=0.0,
        )


@pytest.mark.parametrize("parameter_name", ["lambda_1", "lambda_2"])
def test_each_tidal_deformability_changes_the_waveform(parameter_name: str) -> None:
    waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)
    frequencies = jnp.linspace(20.0, 1024.0, 257)
    baseline = waveform(frequencies, PAPER_PARAMETERS)
    changed = waveform(
        frequencies,
        {**PAPER_PARAMETERS, parameter_name: PAPER_PARAMETERS[parameter_name] + 150.0},
    )

    assert _relative_waveform_change(baseline, changed) > 1e-3


@pytest.mark.parametrize("parameter_name", ["s1_x", "s1_y", "s2_x", "s2_y"])
def test_each_transverse_spin_component_changes_the_waveform(
    parameter_name: str,
) -> None:
    waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)
    frequencies = jnp.linspace(20.0, 1024.0, 257)
    baseline = waveform(frequencies, PAPER_PARAMETERS)
    changed = waveform(
        frequencies,
        {**PAPER_PARAMETERS, parameter_name: PAPER_PARAMETERS[parameter_name] + 0.01},
    )

    assert _relative_waveform_change(baseline, changed) > 1e-4


@pytest.mark.parametrize(
    "parameters",
    [PAPER_PARAMETERS, MASS_RATIO_EDGE_PARAMETERS],
    ids=["representative", "q-min-tidal-corner"],
)
def test_waveform_matches_lalsimulation_imrphenompv2_nrtidalv2(
    parameters: dict[str, float],
) -> None:
    lal = pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")

    delta_f = 1.0
    f_min = 20.0
    f_max = 2048.0
    f_ref = 20.0
    # LAL returns the exact f_max bin as zero by convention. Compare the common,
    # populated half-open grid [f_min, f_max) instead.
    frequencies = np.arange(f_min, f_max, delta_f)
    waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=f_ref)
    actual = waveform(jnp.asarray(frequencies), parameters)

    chirp_mass = parameters["M_c"]
    eta = parameters["eta"]
    total_mass = chirp_mass / eta ** (3.0 / 5.0)
    mass_difference = np.sqrt(1.0 - 4.0 * eta)
    primary_mass = 0.5 * total_mass * (1.0 + mass_difference)
    secondary_mass = 0.5 * total_mass * (1.0 - mass_difference)

    lal_parameters = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(
        lal_parameters, parameters["lambda_1"]
    )
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(
        lal_parameters, parameters["lambda_2"]
    )
    hp_lal, hc_lal = lalsimulation.SimInspiralChooseFDWaveform(
        primary_mass * lal.MSUN_SI,
        secondary_mass * lal.MSUN_SI,
        parameters["s1_x"],
        parameters["s1_y"],
        parameters["s1_z"],
        parameters["s2_x"],
        parameters["s2_y"],
        parameters["s2_z"],
        parameters["d_L"] * 1e6 * lal.PC_SI,
        parameters["iota"],
        parameters["phase_c"],
        0.0,
        0.0,
        0.0,
        delta_f,
        f_min,
        f_max,
        f_ref,
        lal_parameters,
        lalsimulation.GetApproximantFromString("IMRPhenomPv2_NRTidalv2"),
    )

    first_bin = round((f_min - hp_lal.f0) / hp_lal.deltaF)
    last_bin = first_bin + frequencies.size
    expected = {
        "p": np.asarray(hp_lal.data.data[first_bin:last_bin]),
        "c": np.asarray(hc_lal.data.data[first_bin:last_bin]),
    }

    for polarization in ("p", "c"):
        actual_polarization = np.asarray(actual[polarization])
        expected_polarization = expected[polarization]
        populated = (np.abs(actual_polarization) > 0.0) & (
            np.abs(expected_polarization) > 0.0
        )
        actual_populated = actual_polarization[populated]
        expected_populated = expected_polarization[populated]
        populated_frequencies = frequencies[populated]

        # Both phase and coalescence time are marginalized by the paper's
        # GW170817 likelihood. Remove the corresponding constant and linear
        # phase convention before checking the physical waveform shape.
        phase_difference = np.unwrap(np.angle(actual_populated / expected_populated))
        phase_line = np.polyfit(
            populated_frequencies,
            phase_difference,
            deg=1,
            w=np.abs(expected_populated),
        )
        aligned_actual = actual_populated * np.exp(
            -1.0j * np.polyval(phase_line, populated_frequencies)
        )
        relative_error = np.linalg.norm(aligned_actual - expected_populated) / (
            np.linalg.norm(expected_populated)
        )

        assert relative_error < LALSIMULATION_PARITY_RTOL
        assert abs(phase_line[0] / (2.0 * np.pi)) < 3e-5
        assert abs(phase_line[1]) < 2e-3
        np.testing.assert_allclose(
            np.abs(actual_populated),
            np.abs(expected_populated),
            rtol=1e-9,
            atol=0.0,
        )
