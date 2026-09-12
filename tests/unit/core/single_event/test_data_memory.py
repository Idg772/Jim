"""Host storage must avoid implicit allocations and preserve mutable data."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event import data as data_module
from jimgw.core.single_event.data import Data, PowerSpectrum


def test_materialized_arrays_does_not_compute_any_lazy_array(monkeypatch):
    fd = np.ones(33, dtype=np.complex128)
    data = Data.from_host_fd(fd, delta_t=1 / 16)

    def unexpected(*args, **kwargs):
        pytest.fail("Existing-buffer inspection invoked a lazy constructor")

    monkeypatch.setattr(np.fft, "irfft", unexpected)
    monkeypatch.setattr(np.fft, "rfftfreq", unexpected)
    monkeypatch.setattr(np, "ones", unexpected)
    arrays = data.materialized_arrays()
    jax.block_until_ready(arrays)
    assert len(arrays) == 1
    assert arrays[0] is fd
    assert data._td is None
    assert data._window is None
    assert data._host_frequencies is None


def test_materialized_arrays_returns_eager_buffers_by_identity():
    td = jnp.zeros(16)
    window = jnp.ones(16)
    data = Data(td, delta_t=0.25, window=window)
    arrays = data.materialized_arrays()
    assert len(arrays) == 3
    assert arrays[0] is data.fd
    assert arrays[1] is td
    assert arrays[2] is window
    assert data.time_domain_materialised


def test_materialized_arrays_includes_only_explicitly_created_host_buffers():
    data = Data.from_host_fd(np.ones(9, dtype=np.complex128), delta_t=0.25)
    frequencies = data.frequencies
    window = data.window
    arrays = data.materialized_arrays()
    assert len(arrays) == 3
    assert arrays[0] is data.fd
    assert arrays[1] is window
    assert arrays[2] is frequencies
    assert data._td is None


def test_host_window_is_readonly_scalar_broadcast():
    data = Data.from_host_fd(np.zeros(33, dtype=np.complex128), delta_t=0.25)
    window = data.window
    assert window.shape == (64,)
    assert window.strides == (0,)
    assert window.base.nbytes == np.dtype(np.float64).itemsize
    assert not window.flags.writeable
    np.testing.assert_array_equal(window * np.arange(64), np.arange(64))
    with pytest.raises(ValueError, match="read-only"):
        window[0] = 0.0


def test_mutable_fd_updates_are_visible_without_retaining_td():
    fd = np.zeros(9, dtype=np.complex128)
    data = Data.from_host_fd(fd, delta_t=0.25)
    initial = data.td
    fd[1] = 1.0 + 0.2j
    updated = data.td
    np.testing.assert_array_equal(initial, np.zeros(16))
    np.testing.assert_allclose(updated, np.fft.irfft(fd) / 0.25, rtol=0, atol=0)
    assert np.max(np.abs(updated - initial)) > 0.5
    assert data.fd is fd
    assert data._td is None
    assert not data.time_domain_materialised
    assert len(data.materialized_arrays()) == 1


def test_host_frequency_cache_tracks_changed_time_step():
    data = Data.from_host_fd(np.ones(9, dtype=np.complex128), delta_t=0.25)
    first = data.frequencies
    assert data.frequencies is first
    data.delta_t = 0.5
    second = data.frequencies
    np.testing.assert_array_equal(second, np.fft.rfftfreq(16, 0.5))
    assert second[-1] == 1.0
    assert first[-1] == 2.0
    assert data.frequencies is second


@pytest.mark.parametrize("n_time,delta_t", [(64, 1 / 16), (70, 0.3), (120, 0.05)])
def test_identical_host_native_grids_share_one_readonly_array(n_time, delta_t):
    first = Data.from_host_fd(np.zeros(n_time // 2 + 1, complex), delta_t)
    second = Data.from_host_fd(np.ones(n_time // 2 + 1, complex), delta_t)
    assert first.frequencies is second.frequencies
    np.testing.assert_array_equal(first.frequencies, np.fft.rfftfreq(n_time, delta_t))
    assert not first.frequencies.flags.writeable
    second.delta_t = 2 * delta_t
    assert second.frequencies is not first.frequencies
    np.testing.assert_array_equal(first.frequencies, np.fft.rfftfreq(n_time, delta_t))


def test_unused_shared_host_grid_is_not_retained_by_global_cache():
    data = Data.from_host_fd(np.zeros(18, complex), delta_t=0.173)
    grid_reference = weakref.ref(data.frequencies)
    assert grid_reference() is not None
    del data
    gc.collect()
    assert grid_reference() is None


def test_concurrent_host_grid_requests_allocate_one_grid(monkeypatch):
    data = [Data.from_host_fd(np.zeros(14, complex), delta_t=0.217) for _ in range(4)]
    original = np.arange
    calls = []

    def record_arange(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "arange", record_arange)
    with ThreadPoolExecutor(max_workers=4) as pool:
        grids = list(pool.map(lambda value: value.frequencies, data))
    assert len(calls) == 1
    assert all(grid is grids[0] for grid in grids)


def test_memmapped_fd_is_wrapped_without_copying(tmp_path):
    mapped = np.memmap(
        tmp_path / "strain.bin", dtype=np.complex128, mode="w+", shape=33
    )
    data = Data.from_host_fd(mapped, delta_t=0.25)
    assert np.shares_memory(data.fd, mapped)
    mapped[4] = 3 + 2j
    assert data.fd[4] == 3 + 2j
    assert data._td is None


@pytest.mark.parametrize(
    "frequencies",
    [
        [1.0, 3.0, 2.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [1.0, float("nan"), 2.0, 4.0],
    ],
)
def test_unsorted_host_psd_preserves_boolean_band_selection(frequencies):
    grid = np.array(frequencies)
    values = np.arange(grid.size, dtype=float) + 1.0
    psd = PowerSpectrum(values, grid)
    actual_values, actual_frequencies = psd.frequency_slice(1.5, 2.5)
    mask = (grid >= 1.5) & (grid <= 2.5)
    np.testing.assert_array_equal(actual_values, values[mask])
    np.testing.assert_array_equal(actual_frequencies, grid[mask])


def test_sorted_host_psd_keeps_views_and_inclusive_duplicate_endpoints():
    grid = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    values = np.arange(grid.size, dtype=float) + 1.0
    psd = PowerSpectrum(values, grid)
    actual_values, actual_frequencies = psd.frequency_slice(2.0, 3.0)
    np.testing.assert_array_equal(actual_values, values[1:4])
    np.testing.assert_array_equal(actual_frequencies, grid[1:4])
    assert np.shares_memory(actual_values, values)
    assert np.shares_memory(actual_frequencies, grid)


@pytest.mark.parametrize("bounds", [(float("nan"), 3.0), (1.0, float("nan"))])
def test_host_nan_band_bounds_keep_the_existing_empty_mask_result(bounds):
    data = Data.from_host_fd(np.ones(9, complex), delta_t=0.25)
    psd = PowerSpectrum(np.ones(9), data.frequencies)
    for source in (data, psd):
        values, frequencies = source.frequency_slice(*bounds)
        assert values.size == 0
        assert frequencies.size == 0


def test_mutating_host_psd_grid_invalidates_sorted_fast_path():
    grid = np.array([1.0, 2.0, 3.0, 4.0])
    psd = PowerSpectrum(np.array([10.0, 20.0, 30.0, 40.0]), grid)
    psd.frequency_slice(1.5, 2.5)
    grid[1:3] = [3.0, 2.0]
    values, frequencies = psd.frequency_slice(1.5, 2.5)
    np.testing.assert_array_equal(values, [30.0])
    np.testing.assert_array_equal(frequencies, [2.0])


def test_host_psd_ordering_check_catches_chunk_boundary_inversion():
    grid = np.arange(65_538, dtype=float)
    grid[65_536] = grid[65_535] - 0.5
    values = np.arange(grid.size, dtype=float)
    psd = PowerSpectrum(values, grid)
    actual_values, actual_frequencies = psd.frequency_slice(65_534.75, 65_535.25)
    mask = (grid >= 65_534.75) & (grid <= 65_535.25)
    np.testing.assert_array_equal(actual_values, values[mask])
    np.testing.assert_array_equal(actual_frequencies, grid[mask])


@pytest.mark.parametrize("kind", ["linear", "cubic"])
def test_host_psd_interpolation_uses_bounded_chunks_with_exact_scipy_values(
    monkeypatch, kind
):
    source_grid = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    source_values = 1 + np.sin(source_grid / 3) ** 2
    target = np.linspace(-1.0, 9.0, 262_145)
    target.setflags(write=False)
    original_interp1d = data_module.interp1d
    expected = original_interp1d(
        source_grid,
        source_values,
        kind=kind,
        fill_value=(source_values[0], source_values[-1]),
        bounds_error=False,
    )(target)
    calls = []

    def bounded_interpolator(*args, **kwargs):
        interpolate = original_interp1d(*args, **kwargs)

        def evaluate(frequencies):
            calls.append(len(frequencies))
            assert len(frequencies) <= 262_144
            return interpolate(frequencies)

        return evaluate

    monkeypatch.setattr(data_module, "interp1d", bounded_interpolator)
    psd = PowerSpectrum(source_values, source_grid).interpolate(target, kind=kind)
    assert calls == [262_144, 1]
    assert psd.frequencies is target
    np.testing.assert_array_equal(psd.values, expected)


def test_host_psd_interpolation_accepts_an_empty_target():
    psd = PowerSpectrum(np.array([1.0, 2.0]), np.array([0.0, 2.0]))
    target = np.array([], dtype=float)
    result = psd.interpolate(target)
    assert result.frequencies is target
    assert result.values.shape == (0,)
