"""Small real-waveform checks for bounded host and mapped FD injection."""

import mmap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event.data import PowerSpectrum
from jimgw.core.single_event.detector import get_H1
from jimgw.core.single_event.heterodyne_noise import (
    INDEXED_NOISE_ALGORITHM,
    compiled_indexed_noise,
)
from jimgw.core.single_event.native_storage import mapped_region, release_mapped_pages
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2

jax.config.update("jax_enable_x64", True)

DURATION = 4.0
# Retain two complete OS pages for the explicit eviction/readback check,
# including macOS hosts with 16 KiB pages; only 17 frequencies carry signal.
SAMPLING_FREQUENCY = float(max(256, mmap.PAGESIZE // 16))
F_MIN, F_MAX = 20.0, 24.0  # 17 samples: chunks of eight leave a singleton tail.
FREQUENCIES = np.arange(int(DURATION * SAMPLING_FREQUENCY) // 2 + 1) / DURATION
PSD = 1e-46 * (1.0 + FREQUENCIES / 40.0) ** 2
PARAMETERS = {
    "M_c": 1.1802650981093186,
    "eta": 0.2499511169561633,
    "s1_z": -0.014169601016451747,
    "s2_z": -0.02350583516601907,
    "lambda_1": 483.1557044220429,
    "lambda_2": 771.0436351634714,
    "d_L": 200.0,
    "phase_c": 0.4,
    "iota": 1.3,
    "ra": 1.5,
    "dec": 0.3,
    "psi": 0.2,
    "t_c": 0.015,
}
KEY = jax.random.fold_in(jax.random.key(713), 2)


def _inject(*, host=True, storage="memory", noise="indexed-v1", chunk=8, zero=False):
    detector = get_H1()
    detector.set_psd(PowerSpectrum(jnp.asarray(PSD), jnp.asarray(FREQUENCIES)))
    detector.inject_signal(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        trigger_time=1_300_000_000.0,
        waveform_model=RippleIMRPhenomD_NRTidalv2(f_ref=20.0),
        parameters=PARAMETERS,
        f_min=F_MIN,
        f_max=F_MAX,
        zero_noise=zero,
        rng_key=KEY,
        waveform_chunk_size=chunk,
        host_resident=host,
        host_storage=storage,
        noise_generation=noise,
    )
    return detector


def _same_fd(actual, expected):
    # JIT fusion can change the waveform's floating-point phase arithmetic;
    # the random stream itself has stricter bitwise tests in test_heterodyne_noise.
    expected = np.asarray(expected)
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=2e-10 * np.max(np.abs(expected))
    )


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("chunk", [1, 8])
def test_real_nrtidal_legacy_device_and_host_preserve_singleton_tail(chunk, zero):
    device = _inject(host=False, noise="legacy", chunk=chunk, zero=zero)
    host = _inject(noise="legacy", chunk=chunk, zero=zero)
    expected = _inject(host=False, noise="legacy", chunk=17, zero=zero)
    assert len(host.sliced_fd_data) == 17
    assert np.all(np.isfinite(device.sliced_fd_data))
    assert np.all(np.isfinite(host.sliced_fd_data))
    _same_fd(device.data.fd, expected.data.fd)
    _same_fd(host.data.fd, expected.data.fd)
    for detector in (device, host, expected):
        np.testing.assert_array_equal(detector.psd.values, PSD)
        np.testing.assert_array_equal(detector.sliced_psd, PSD[80:97])
    assert host.data.time_domain_materialised is False
    assert float(host.optimal_snr) == pytest.approx(
        float(expected.optimal_snr), rel=2e-10
    )


@pytest.mark.parametrize("storage", ["memory", "mmap"])
def test_indexed_injection_never_calls_full_array_noise_and_matches_absolute_indices(
    monkeypatch, storage
):
    def forbidden_full_noise(*args, **kwargs):
        raise AssertionError("indexed injection must never simulate full-array noise")

    monkeypatch.setattr(PowerSpectrum, "simulate_data", forbidden_full_noise)
    noisy = _inject(storage=storage, chunk=8)
    signal = _inject(storage=storage, chunk=8, zero=True)
    expected_noise = np.asarray(compiled_indexed_noise(KEY, PSD[80:97], 0.25, 80))
    np.testing.assert_allclose(
        np.asarray(noisy.sliced_fd_data) - signal.sliced_fd_data,
        expected_noise,
        rtol=0,
        atol=8 * np.finfo(float).eps * np.max(np.abs(noisy.sliced_fd_data)),
    )
    assert isinstance(noisy.data.fd, np.ndarray)
    assert np.shares_memory(noisy.sliced_fd_data, noisy.data.fd)
    assert np.shares_memory(noisy.sliced_psd, noisy.psd.values)
    assert noisy.data.time_domain_materialised is False
    np.testing.assert_array_equal(noisy.psd.values, PSD)
    np.testing.assert_array_equal(noisy.data.fd[:80], 0)
    np.testing.assert_array_equal(noisy.data.fd[97:], 0)
    diagnostics = noisy.data_preparation_diagnostics
    assert diagnostics["noise_algorithm"] == INDEXED_NOISE_ALGORITHM
    assert diagnostics["full_length_device_noise"] is False


@pytest.mark.parametrize("chunk", [1, 5, 8, 17])
def test_indexed_memory_and_mmap_preserve_noise_across_chunk_boundaries(chunk):
    baseline = _inject(storage="memory", chunk=17)
    memory = _inject(storage="memory", chunk=chunk)
    mapped = _inject(storage="mmap", chunk=chunk)
    _same_fd(memory.data.fd, baseline.data.fd)
    np.testing.assert_array_equal(mapped.data.fd, memory.data.fd)
    for detector in (baseline, memory, mapped):
        assert detector.data.time_domain_materialised is False
        np.testing.assert_array_equal(detector.psd.values, PSD)


def test_mapped_band_views_survive_flush_page_eviction_and_file_readback():
    detector = _inject(storage="mmap")
    region = mapped_region(detector.data.fd)
    assert region is not None
    mapping, start, end = region
    assert end - start > 2 * mmap.PAGESIZE
    saved_band = detector.sliced_fd_data.copy()
    saved_fd = detector.data.fd.copy()
    advised = release_mapped_pages(detector.data.fd, written=True)
    if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        assert advised
    np.testing.assert_array_equal(detector.sliced_fd_data, saved_band)
    np.testing.assert_array_equal(detector.data.fd, saved_fd)
    owner = detector.data._host_storage_owner
    owner.seek(0)
    persisted = np.frombuffer(owner.read(), dtype=np.complex128)
    np.testing.assert_array_equal(persisted, saved_fd)
    assert np.shares_memory(detector.sliced_fd_data, detector.data.fd)
    assert detector.data.time_domain_materialised is False


def test_storage_and_indexed_modes_require_host_data():
    for options in ({"storage": "mmap", "noise": "legacy"}, {"noise": "indexed-v1"}):
        with pytest.raises(ValueError, match="require host_resident"):
            _inject(host=False, **options)
