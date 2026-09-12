"""Small real injections test native storage sharing and CLI readiness."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.cli import _data
from jimgw.cli._config import InjectionDataConfig
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.detector import GroundBased2G
from jimgw.core.single_event.native_storage import native_storage_accounting


class FlatWaveform:
    """Synthetic finite spectrum; not a physical four-second 2-Hz signal."""

    f_ref = 20.0

    def __call__(self, frequencies, parameters):
        del parameters
        carrier = jnp.ones_like(frequencies, dtype=jnp.complex128) * 1e-22
        return {"p": carrier, "c": -1j * carrier}


def _config(tmp_path, *, storage="memory", psd_min=2.0):
    psd = tmp_path / "et.npz"
    frequencies = np.arange(psd_min, 17.0)
    np.savez(psd, frequencies=frequencies, values=1e-44 * (1 + frequencies / 100))
    return InjectionDataConfig(
        detectors=["ET"],
        trigger_time=1126259462.4,
        duration=4.0,
        sampling_frequency=32.0,
        injection_parameters={"t_c": 0.0, "ra": 1.375, "dec": -1.2108, "psi": 0.2},
        psd_files={"ET": psd},
        waveform_chunk_size=13,
        host_resident_data=True,
        host_data_storage=storage,
        noise_generation="indexed-v1",
    )


def _build(config):
    return _data.build_data(
        config,
        f_min={"ET1": 2.0, "ET2": 3.0, "ET3": 4.0},
        f_max=12.0,
        waveform=FlatWaveform(),
        time_frame="geocentric",
        seed=17,
    )


@pytest.mark.parametrize("storage", ["memory", "mmap"])
def test_real_et_injections_share_grid_and_immutable_psd_but_not_strain(
    tmp_path, storage
):
    cfg = _config(tmp_path, storage=storage)
    noisy = _build(cfg)
    repeated = _build(cfg)
    clean = _build(cfg.model_copy(update={"zero_noise": True}))
    assert [detector.name for detector in noisy] == ["ET1", "ET2", "ET3"]
    for index, detector in enumerate(noisy):
        assert detector.data.frequencies is noisy[0].data.frequencies
        assert detector.psd is noisy[0].psd
        assert detector.psd.frequencies is detector.data.frequencies
        assert not detector.psd.values.flags.writeable
        assert not detector.psd.frequencies.flags.writeable
        assert np.shares_memory(detector.sliced_psd, detector.psd.values)
        assert np.shares_memory(detector.sliced_fd_data, detector.data.fd)
        assert detector.data._td is None
        assert detector.data._window is None
        assert detector.sliced_frequencies[0] == 2.0 + index
        assert detector.sliced_frequencies[-1] == 12.0
        np.testing.assert_array_equal(detector.data.fd, repeated[index].data.fd)
        if index:
            assert not np.shares_memory(detector.data.fd, noisy[0].data.fd)
        if storage == "mmap":
            assert detector.data._host_storage_owner is not None
    # Subtract each detector's own noiseless response before comparing the
    # noise on the shared band; differing antenna patterns are not evidence
    # of distinct noise realizations.
    common = (noisy[0].data.frequencies >= 4.0) & (noisy[0].data.frequencies <= 12.0)
    noises = [
        (detector.data.fd - signal.data.fd)[common]
        for detector, signal in zip(noisy, clean, strict=True)
    ]
    assert all(np.any(noise != noises[0]) for noise in noises[1:])
    _data.wait_for_data_ready(noisy)
    assert all(
        detector.data._td is None and detector.data._window is None
        for detector in noisy
    )
    n_frequencies = len(noisy[0].data.fd)
    accounting = native_storage_accounting(noisy)
    assert accounting == {
        "host_allocation_count": 2 + (3 if storage == "memory" else 0),
        "host_allocation_bytes": 2 * n_frequencies * 8
        + (3 * n_frequencies * 16 if storage == "memory" else 0),
        "mapped_file_count": 3 if storage == "mmap" else 0,
        "mapped_file_logical_bytes": 3 * n_frequencies * 16 if storage == "mmap" else 0,
        "device_array_object_count": 0,
        "device_array_logical_bytes": 0,
        "unclassified_array_object_count": 0,
    }


def test_original_psd_support_is_checked_before_reusing_native_cache(
    tmp_path, monkeypatch
):
    cfg = _config(tmp_path, psd_min=3.0)
    called = []
    original = GroundBased2G.inject_signal

    def record_injection(self, *args, **kwargs):
        called.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GroundBased2G, "inject_signal", record_injection)
    # ET1 can populate a native PSD cache with extrapolated values below 3 Hz.
    # ET2 still must fail against the original table's 3 Hz lower endpoint.
    with pytest.raises(ValueError, match="ET2 configured PSD covers"):
        _data.build_data(
            cfg,
            f_min={"ET1": 5.0, "ET2": 2.0, "ET3": 3.0},
            f_max=12.0,
            waveform=FlatWaveform(),
            time_frame="geocentric",
            seed=17,
        )
    assert called == ["ET1"]


class GuardedLazyData(Data):
    @property
    def td(self):
        raise AssertionError("readiness must not access lazy td")

    @property
    def window(self):
        raise AssertionError("readiness must not access lazy window")


def test_readiness_synchronizes_existing_buffers_without_lazy_property_reads(
    monkeypatch,
):
    data = GuardedLazyData.from_host_fd(np.ones(9, complex), delta_t=0.25)
    # Existing eager buffers still need synchronization even on a lazy-data
    # object. Populate the storage fields directly, keeping getter guards.
    td = jnp.sin(jnp.arange(16.0))
    window = jnp.cos(jnp.arange(16.0))
    data._td = td
    data._window = window
    sliced_fd = jnp.arange(9.0) + 1j
    sliced_psd = jnp.arange(9.0) + 1.0
    psd_frequencies = jnp.arange(9.0)
    psd_values = jnp.ones(9)
    ifo = SimpleNamespace(
        data=data,
        sliced_fd_data=sliced_fd,
        sliced_psd=sliced_psd,
        psd=SimpleNamespace(frequencies=psd_frequencies, values=psd_values),
    )
    original = jax.block_until_ready
    recorded = []

    def record_and_wait(value):
        recorded.extend(jax.tree.leaves(value))
        return original(value)

    monkeypatch.setattr(_data.jax, "block_until_ready", record_and_wait)
    _data.wait_for_data_ready([ifo])
    for expected in (
        data.fd,
        td,
        window,
        sliced_fd,
        sliced_psd,
        psd_frequencies,
        psd_values,
    ):
        assert any(value is expected for value in recorded)


def test_storage_accounting_counts_backing_buffers_without_lazy_reads():
    data = GuardedLazyData.from_host_fd(np.ones(9, complex), delta_t=0.25)
    window_scalar = np.array(1.0)
    data._window = np.broadcast_to(window_scalar, (16,))
    frequencies = np.arange(9.0)
    data._host_frequencies = frequencies
    values = np.ones(9)
    ifo = SimpleNamespace(
        data=data,
        psd=SimpleNamespace(frequencies=frequencies[1:], values=values[2:]),
    )
    accounting = native_storage_accounting([ifo, ifo])
    assert accounting == {
        "host_allocation_count": 4,
        "host_allocation_bytes": 9 * 16 + 8 + 2 * 9 * 8,
        "mapped_file_count": 0,
        "mapped_file_logical_bytes": 0,
        "device_array_object_count": 0,
        "device_array_logical_bytes": 0,
        "unclassified_array_object_count": 0,
    }
    assert data._td is None
    assert data._window.base is window_scalar


def test_storage_accounting_labels_device_objects_as_logical_bytes():
    data = GuardedLazyData.from_host_fd(np.ones(9, complex), delta_t=0.25)
    values = jnp.arange(9.0)
    data._td = values
    ifo = SimpleNamespace(
        data=data, psd=SimpleNamespace(frequencies=values, values=values)
    )
    accounting = native_storage_accounting([ifo, ifo])
    assert accounting["device_array_object_count"] == 1
    assert accounting["device_array_logical_bytes"] == values.nbytes
    assert accounting["host_allocation_count"] == 1
    assert accounting["host_allocation_bytes"] == 9 * 16
    assert accounting["unclassified_array_object_count"] == 0
