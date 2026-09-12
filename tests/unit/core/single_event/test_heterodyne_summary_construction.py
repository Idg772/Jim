"""Construction-cost contracts for the native heterodyne summaries.

The device backend must compile one reduction shape per detector at the
configured chunk size, and the NumPy backend must survive a one-sample tail
chunk (the exact geometry of the 131072 s network fixture, whose bands are
one sample longer than a multiple of the chunk size).
"""

import jax
import jax.numpy as jnp
import numpy as np

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD

jax.config.update("jax_enable_x64", True)

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
EDGES = np.array([2.0, 2.5, 3.0, 4.0, 6.0, 10.0])
ANCHORS = (-0.02, 0.0, 0.02)
N_FREQ = 33  # 2..10 Hz at 0.25 Hz spacing, i.e. duration 4 s


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
    """A backend that reads the native spacing from its first two samples."""

    def __call__(self, frequencies, parameters):
        spacing = frequencies[1] - frequencies[0]
        return {
            key: value * (1.0 + spacing)
            for key, value in super().__call__(frequencies, parameters).items()
        }


def _detectors():
    frequencies = jnp.arange(8, 8 + N_FREQ, dtype=jnp.float64) / 4.0
    rng = np.random.default_rng(913)
    detectors = [get_H1(), get_L1()]
    for index, detector in enumerate(detectors):
        data = np.exp(0.07j * np.asarray(frequencies)) + 0.15 * (
            rng.normal(size=N_FREQ) + 1j * rng.normal(size=N_FREQ)
        )
        detector.set_data(
            Data.from_fd(jnp.asarray(data), frequencies, start_time=1126259460.4)
        )
        detector.set_psd(PowerSpectrum(1.0 + index / 3 + frequencies / 20, frequencies))
    return detectors


def _build(backend, *, chunk_size, waveform=None):
    return HeterodynedTransientLikelihoodFD(
        detectors=_detectors(),
        waveform=ToyWaveform() if waveform is None else waveform,
        f_min=2.0,
        f_max=10.0,
        trigger_time=1126259462.4,
        n_bins=5,
        frequency_bin_edges=EDGES,
        reference_parameters=REFERENCE,
        interpolation_order=3,
        phasor_moment_order=4,
        phasor_time_anchors=ANCHORS,
        reference_chunk_size=chunk_size,
        summary_backend=backend,
    )


def _summaries(likelihood):
    return {
        name: (
            np.asarray(likelihood.summary_data[name]),
            np.asarray(likelihood.phasor_data_moments[name]),
        )
        for name in likelihood.summary_data
    }


def _assert_close(left, right, tol):
    for name in left:
        for a, b in zip(left[name], right[name], strict=True):
            scale = np.abs(b).max()
            assert np.abs(a - b).max() <= tol * scale


def test_jax_backend_compiles_one_shape_per_detector_at_configured_chunk_size():
    # 33 samples in chunks of 12 -> 12, 12, 9 with different bin occupancy.
    likelihood = _build("jax", chunk_size=12)
    for name, diagnostics in likelihood.summary_construction_diagnostics.items():
        assert diagnostics["chunk_size"] == 12, name
        assert diagnostics["chunks"] == 3, name
        assert len(diagnostics["tile_descriptor_shapes"]) == 1, name
    _assert_close(
        _summaries(likelihood), _summaries(_build("numpy", chunk_size=33)), 1e-12
    )


def test_numpy_backend_survives_a_one_sample_tail_chunk():
    # 33 = 2 * 16 + 1: the last chunk holds a single native sample.
    tail = _build("numpy", chunk_size=16, waveform=NativeSpacingWaveform())
    whole = _build("numpy", chunk_size=11, waveform=NativeSpacingWaveform())
    for name in tail.summary_data:
        assert np.all(np.isfinite(np.asarray(tail.summary_data[name])))
    _assert_close(_summaries(tail), _summaries(whole), 1e-12)
