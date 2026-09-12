"""Small native-data constructor checks for the optional JAX summary backend.

All arrays are synthetic; no physical long signal, external sensitivity file,
GPU, or qualification capability is used. Four logical CPU devices exercise
the same per-detector orchestration as an available multi-device host.
"""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event import likelihood as core_likelihood
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_CE, get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD

REFERENCE = {
    "amplitude": 1.3,
    "slope": 0.01,
    "phase_c": 0.2,
    "iota": 0.4,
    "ra": 1.375,
    "dec": -1.2108,
    "psi": 0.2,
    "t_c": 0.0,
}
EDGES = np.array([2.0, 3.0, 5.0, 8.0])
LOW = {"H1": 2.0, "L1": 3.0, "V1": 2.0, "CE": 3.0}
HIGH = {"H1": 8.0, "L1": 8.0, "V1": 6.0, "CE": 6.0}
ANCHORS = (-0.02, 0.0, 0.02)


class ToyWaveform:
    parameter_names = frozenset({"amplitude", "slope", "phase_c", "iota"})

    def __call__(self, frequencies, parameters):
        carrier = (
            parameters["amplitude"]
            * (1.0 + parameters["slope"] * frequencies)
            * jnp.exp(1j * (0.013 * frequencies**2 + parameters["phase_c"]))
        )
        inclination = jnp.cos(parameters["iota"])
        return {
            "p": carrier * (1 + inclination**2) / 2,
            "c": -1j * carrier * inclination,
        }


class NativeSpacingWaveform(ToyWaveform):
    """Expose an erroneous local/tail spacing convention in construction."""

    def __call__(self, frequencies, parameters):
        spacing = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0.0
        return {
            key: value * (1.0 + spacing)
            for key, value in super().__call__(frequencies, parameters).items()
        }


def _detectors():
    frequencies = jnp.arange(33, dtype=jnp.float64) / 4.0
    rng = np.random.default_rng(913)
    detectors = [get_H1(), get_L1(), get_V1(), get_CE()]
    for index, detector in enumerate(detectors):
        data = np.exp(0.07j * np.asarray(frequencies)) + 0.15 * (
            rng.normal(size=33) + 1j * rng.normal(size=33)
        )
        detector.set_data(
            Data.from_fd(jnp.asarray(data), frequencies, start_time=1126259460.4)
        )
        detector.set_psd(PowerSpectrum(1.0 + index / 3 + frequencies / 20, frequencies))
    return detectors


def _build(
    backend, *, anchors=ANCHORS, phase=False, waveform=None, chunk_size=12, edges=EDGES
):
    return HeterodynedTransientLikelihoodFD(
        detectors=_detectors(),
        waveform=ToyWaveform() if waveform is None else waveform,
        f_min=LOW,
        f_max=HIGH,
        trigger_time=1126259462.4,
        n_bins=3,
        frequency_bin_edges=edges,
        reference_parameters=REFERENCE,
        interpolation_order=3,
        phasor_moment_order=4,
        phasor_time_anchors=anchors,
        reference_chunk_size=chunk_size,
        summary_backend=backend,
        phase_marginalization=phase,
    )


def _independent_moments(likelihood, detector, waveform):
    """Dense boolean masks and explicit powers, independent of both reducers."""
    frequencies = detector.sliced_frequencies
    reference = np.asarray(
        detector.fd_response(
            frequencies,
            waveform(frequencies, likelihood.reference_parameters),
            likelihood.reference_parameters,
        )
    )
    f = np.asarray(frequencies)
    data = np.asarray(detector.sliced_fd_data)
    psd = np.asarray(detector.sliced_psd)
    order = likelihood.interpolation_order + likelihood.phasor_moment_order
    norm_order = 2 * likelihood.interpolation_order
    a = np.zeros((order + 1, 3), dtype=np.complex128)
    b = np.zeros((norm_order + 1, 3))
    anchors = likelihood.phasor_time_anchors or ()
    bank = np.zeros((len(anchors), order + 1, 3), dtype=np.complex128)
    edges = np.asarray(likelihood.freq_grid_edges)
    for index, (lo, hi) in enumerate(itertools.pairwise(edges)):
        selected = (f >= lo) & ((f <= hi) if index == 2 else (f < hi))
        u = (2 * f[selected] - lo - hi) / (hi - lo)
        product = data[selected] * reference[selected].conj() / psd[selected]
        norm = np.abs(reference[selected]) ** 2 / psd[selected]
        for degree in range(order + 1):
            a[degree, index] = np.sum(product * u**degree)
            for anchor_index, anchor in enumerate(anchors):
                bank[anchor_index, degree, index] = np.sum(
                    product * u**degree * np.exp(2j * np.pi * f[selected] * anchor)
                )
        for degree in range(norm_order + 1):
            b[degree, index] = np.sum(norm * u**degree)
    scale = 4.0 / float(detector.duration)
    return scale * np.concatenate((a, b)), scale * bank


def _assert_summaries(likelihood, waveform):
    for detector in likelihood.detectors:
        summary, bank = _independent_moments(likelihood, detector, waveform)
        actual = likelihood.summary_data[detector.name]
        assert actual.dtype == jnp.complex128
        np.testing.assert_allclose(actual, summary, rtol=3e-13, atol=3e-13)
        if likelihood.phasor_time_anchors is not None:
            actual_bank = likelihood.phasor_data_moments[detector.name]
            assert actual_bank.dtype == jnp.complex128
            np.testing.assert_allclose(actual_bank, bank, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("anchors", [None, ANCHORS])
@pytest.mark.parametrize("phase", [False, True])
def test_full_noisy_multidetector_constructors_match_native_sums_and_likelihoods(
    anchors, phase
):
    numpy_likelihood = _build("numpy", anchors=anchors, phase=phase)
    jax_likelihood = _build("jax", anchors=anchors, phase=phase)
    np.testing.assert_array_equal(jax_likelihood.freq_grid_edges, EDGES)
    assert jax_likelihood.bin_edges_sha256 == numpy_likelihood.bin_edges_sha256
    assert not jax_likelihood.identical_frequency_grids
    _assert_summaries(numpy_likelihood, ToyWaveform())
    _assert_summaries(jax_likelihood, ToyWaveform())
    for detector in jax_likelihood.detectors:
        name = detector.name
        assert float(detector.sliced_frequencies[0]) == LOW[name]
        assert float(detector.sliced_frequencies[-1]) == HIGH[name]
        np.testing.assert_allclose(
            jax_likelihood.summary_data[name],
            numpy_likelihood.summary_data[name],
            rtol=3e-13,
            atol=3e-13,
        )
    # H1 has 25 native points: two full chunks plus the inclusive final bin edge.
    h1 = jax_likelihood.summary_construction_diagnostics["H1"]
    assert h1["native_frequency_samples"] == 25
    assert h1["chunk_size"] == 12
    assert h1["chunks"] == 3
    for diagnostic in jax_likelihood.summary_construction_diagnostics.values():
        assert diagnostic["tiled_chunks"] == diagnostic["chunks"]
    for shift in (-0.015, 0.0, 0.018):
        parameters = {**REFERENCE, "t_c": shift, "amplitude": 1.15, "slope": 0.014}
        expected = numpy_likelihood.evaluate(parameters)
        actual = jax_likelihood.evaluate(parameters)
        cached = jax_likelihood.evaluate_from_waveform(
            parameters, jax_likelihood.generate_waveform(parameters)
        )
        assert np.isfinite(float(actual))
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(cached, expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("chunk_size", [1, 12])
def test_waveform_prefix_preserves_native_spacing_even_for_one_sample_tail(chunk_size):
    waveform = NativeSpacingWaveform()
    likelihood = _build("jax", waveform=waveform, chunk_size=chunk_size)
    _assert_summaries(likelihood, waveform)


def test_interior_irregular_frequency_uses_generic_fallback_with_correct_bin_membership():
    likelihood = _build("numpy")
    detector = likelihood.detectors[0]
    before = np.asarray(detector.sliced_frequencies).copy()
    # Change an interior point from the first bin to the second, retaining
    # both endpoints, native-spacing prefix, duration, noisy data and PSD.
    detector._sliced_frequencies = detector.sliced_frequencies.at[3].set(3.125)
    assert before[3] == 2.75
    np.testing.assert_array_equal(detector.sliced_frequencies[:2], before[:2])
    assert detector.sliced_frequencies[-1] == before[-1]
    summary, bank, diagnostics = likelihood._compute_reference_coefficients_jax(
        detector,
        ToyWaveform(),
        likelihood.freq_grid_edges,
        interpolation_order=7,
        norm_order=6,
        device=jax.local_devices()[0],
    )
    expected_summary, expected_bank = _independent_moments(
        likelihood, detector, ToyWaveform()
    )
    np.testing.assert_allclose(summary, expected_summary, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(bank, expected_bank, rtol=3e-13, atol=3e-13)
    assert diagnostics["chunks"] == 3
    assert diagnostics["tiled_chunks"] == 2


def test_narrow_bins_reduce_tile_width_to_bound_actual_padded_matrix_size(monkeypatch):
    budget = 128
    monkeypatch.setattr(core_likelihood, "_MAX_TILE_MATRIX_ELEMENTS", budget)
    monkeypatch.setattr(core_likelihood, "_MIN_TILE_SIZE", 1)
    # Narrow bins need many padded tile descriptors per chunk; the width is
    # reduced once for the whole stream so every chunk keeps one compiled
    # shape. The effective chunk size must also shrink when tile padding plus
    # the deterministic combine cannot fit at the requested size.
    likelihood = _build("jax", edges=np.array([2.0, 2.5, 3.0, 8.0]))
    _assert_summaries(likelihood, ToyWaveform())
    h1 = likelihood.summary_construction_diagnostics["H1"]
    assert 0 < h1["chunk_size"] < 12
    assert h1["requested_chunk_size"] == 12
    assert len(h1["tile_descriptor_shapes"]) == 1
    assert h1["tile_size"] < 12
    for diagnostic in likelihood.summary_construction_diagnostics.values():
        assert diagnostic["tiled_chunks"] == diagnostic["chunks"]
        assert diagnostic["max_tile_matrix_elements"] <= budget


def _check_four_cpu_devices():
    assert len(jax.local_devices()) == 4
    assert all(device.platform == "cpu" for device in jax.local_devices())
    likelihood = _build("jax")
    _assert_summaries(likelihood, ToyWaveform())
    diagnostics = likelihood.summary_construction_diagnostics
    assert {item["device"] for item in diagnostics.values()} == {
        str(device) for device in jax.local_devices()
    }
    destination = next(iter(likelihood.frequencies.devices()))
    for array in jax.tree.leaves(
        (likelihood.summary_data, likelihood.phasor_data_moments)
    ):
        assert array.devices() == {destination}
    value = float(jax.jit(likelihood.evaluate)({**REFERENCE, "t_c": 0.012}))
    assert np.isfinite(value)
    print(
        json.dumps(
            {
                "cpu_devices": 4,
                "detectors": sorted(diagnostics),
                "log_likelihood": value,
            }
        )
    )


def test_four_cpu_device_construction_and_gather_in_fresh_process():
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "OMP_NUM_THREADS": "1",
    }
    code = (
        "import runpy; "
        f"scope = runpy.run_path({str(Path(__file__).resolve())!r}); "
        "scope['_check_four_cpu_devices']()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["cpu_devices"] == 4
    assert receipt["detectors"] == ["CE", "H1", "L1", "V1"]
