"""Shared full-band geometry versus the independent detector response."""

import copy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event.detector import get_CE_A, get_ET_Sardinia
from jimgw.core.single_event.time_dependent_response import time_to_coalescence_2pn
from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time
from jimgw.core.single_event.xg_evaluation import FastXGEvaluator
from jimgw.core.single_event.xg_network_response import NetworkResponse

jax.config.update("jax_enable_x64", True)
GPS = 1_300_000_000.0


@pytest.fixture(scope="module")
def setup():
    frequency = jnp.geomspace(2.0, 2048.0, 321)
    tau = time_to_coalescence_2pn(frequency, 1.4, 1.3, 0.015, -0.012)
    params = {
        "ra": 6.052874807646849,
        "dec": 0.17257877754217157,
        "psi": 1.6782976747256861,
        "t_c": 0.03512602655783986,
        "gmst": greenwich_mean_sidereal_time(GPS),
        "trigger_time": GPS,
    }
    detectors = [get_CE_A(), *get_ET_Sardinia()]
    for detector in detectors:
        detector.time_dependent_response = True
        detector.finite_arm_response = True
        detector.configure_orbital_motion_response(
            enabled=True,
            reference_time=GPS,
            validity_s=(-131072.125, 0.125),
            acceleration_over_c=(
                2.002108458580428e-11,
                -1.019481229165329e-12,
                -4.473979670828952e-13,
            ),
            jerk_over_c=(
                1.38632199910925e-20,
                3.84800286572565e-18,
                1.686090305174408e-18,
            ),
        )
    return detectors, frequency, params, tau


def baseline(detectors, frequency, params, tau):
    likelihood = SimpleNamespace(
        detectors=detectors,
        freq_grid_node_flat=frequency,
        reference_parameters=params,
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=jnp.linspace(-0.2, 0.2, 21),
    )
    return FastXGEvaluator(
        likelihood, lambda f, p: {"p": jnp.ones_like(f), "__tau__": tau}
    )


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    ("sky", "masses"),
    [
        ((0.0, 0.0, 0.0, -0.08), (1.4, 1.3)),
        ((3.1, np.pi / 2 - 1e-8, 2.1, 0.08), (1.8, 1.5)),
        ((2 * np.pi - 1e-8, -np.pi / 2 + 1e-8, np.pi, 0.0), (1.2, 1.2)),
    ],
)
def test_full_band_response_matches_independent_channels(setup, reuse, sky, masses):
    detectors, frequency, reference, _reference_tau = setup
    independent = baseline(*setup)
    network = NetworkResponse(*setup, reuse_opposite_arms=reuse)
    group = network.make_group(range(4), np.arange(frequency.size))
    params = {**reference, **dict(zip(("ra", "dec", "psi", "t_c"), sky, strict=True))}
    tau = time_to_coalescence_2pn(frequency, *masses, 0.015, -0.012)
    state = jax.jit(network.prepare)(params, tau)
    actual = jax.jit(group)(state)
    assert np.all(state["valid"])
    expected = [
        independent.response_modes(detector, params, tau) for detector in detectors
    ]
    for mode in ("p", "c"):
        np.testing.assert_allclose(
            actual[mode],
            jnp.stack([item[mode] for item in expected]),
            rtol=2e-11,
            atol=2e-11,
        )
    expected_delays = [
        detector.delay_from_geocenter(params["ra"], params["dec"], params["gmst"])
        for detector in detectors
    ]
    np.testing.assert_allclose(
        state["trigger_delays"], expected_delays, rtol=0, atol=8e-18
    )
    assert group.diagnostics["independent_arm_pairs"] == (5 if reuse else 8)
    assert group.diagnostics["reused_opposite_arms"] == (3 if reuse else 0)
    assert group.diagnostics["maximum_opposite_pair_error"] <= 8 * np.finfo(float).eps


def test_active_subsets_keep_distinct_vertex_delays_and_phase(setup):
    _detectors, frequency, reference, tau = setup
    network = NetworkResponse(*setup)
    state = network.prepare({**reference, "ra": 2.7, "t_c": -0.035}, tau)
    assert np.ptp(np.asarray(state["trigger_delays"])[1:]) > 1e-6
    indices = np.arange(2, frequency.size, 3)
    subset = network.make_group([1, 2, 3], indices)(state)
    full = network.make_group(range(4), np.arange(frequency.size))(state)
    for mode in ("p", "c"):
        np.testing.assert_allclose(
            subset[mode], full[mode][1:, indices], rtol=3e-14, atol=3e-14
        )


def test_unequal_orbital_contracts_remain_independent(setup):
    detectors, frequency, reference, tau = setup
    detectors = copy.deepcopy(detectors)
    detectors[2].orbital_acceleration_over_c = tuple(
        1.03 * np.asarray(detectors[2].orbital_acceleration_over_c)
    )
    detectors[3].orbital_motion_response = False
    network = NetworkResponse(detectors, frequency, reference, tau)
    assert network.diagnostics["orbital_contract_groups"] == 2
    params = {**reference, "ra": 1.2, "dec": -0.7, "t_c": -0.01}
    actual = network.make_group(range(4), np.arange(frequency.size))(
        network.prepare(params, tau * 1.05)
    )
    independent = baseline(detectors, frequency, reference, tau)
    expected = [
        independent.response_modes(detector, params, tau * 1.05)
        for detector in detectors
    ]
    for mode in ("p", "c"):
        np.testing.assert_allclose(
            actual[mode],
            jnp.stack([item[mode] for item in expected]),
            rtol=2e-11,
            atol=2e-11,
        )


@pytest.mark.parametrize("change", ["angle", "length"])
def test_perturbed_geometry_falls_back_for_the_affected_edge(setup, change):
    detectors, frequency, reference, tau = setup
    detectors = copy.deepcopy(detectors)
    if change == "angle":
        detectors[1].xarm_azimuth += 1e-7
        expected_pairs = 4
    else:
        detectors[1].arm_length_m += 1.0
        expected_pairs = 5
    network = NetworkResponse(detectors, frequency, reference, tau)
    group = network.make_group([1, 2, 3], np.arange(frequency.size))
    assert group.diagnostics["independent_arm_pairs"] == expected_pairs
    actual = group(network.prepare(reference, tau))
    independent = baseline(detectors, frequency, reference, tau)
    expected = [
        independent.response_modes(detector, reference, tau)
        for detector in detectors[1:]
    ]
    for mode in ("p", "c"):
        np.testing.assert_allclose(
            actual[mode],
            jnp.stack([item[mode] for item in expected]),
            rtol=2e-11,
            atol=2e-11,
        )


@pytest.mark.parametrize("bad_epoch", [False, True])
def test_full_clock_validity_cannot_be_hidden_by_node_trimming(setup, bad_epoch):
    detectors, frequency, reference, tau = setup
    detectors = copy.deepcopy(detectors)
    if bad_epoch:
        detectors[0].orbital_reference_time += 1.0
    else:
        detectors[0].orbital_validity_s = (-1000.0, 0.125)
    invalid = NetworkResponse(detectors, frequency, reference, tau)
    state = invalid.prepare(reference, tau)
    np.testing.assert_array_equal(state["valid"], [False, True, True, True])
    high_nodes = np.flatnonzero(np.asarray(frequency) > 100.0)
    assert np.all(np.asarray(reference["t_c"] - tau)[high_nodes] > -1000.0)
    actual = invalid.make_group([0], high_nodes)(state)
    assert np.all(np.isnan(actual["p"]))
    assert np.all(np.isnan(actual["c"]))


def test_nonfinite_clock_cannot_be_hidden_without_orbital_response(setup):
    detectors, frequency, reference, tau = setup
    detectors = copy.deepcopy(detectors)
    for detector in detectors:
        detector.orbital_motion_response = False
    network = NetworkResponse(detectors, frequency, reference, tau)
    state = network.prepare(reference, tau.at[0].set(jnp.nan))
    assert not np.any(state["valid"])
    actual = network.make_group([0], np.arange(1, frequency.size))(state)
    assert np.all(np.isnan(actual["p"]))
