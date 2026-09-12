"""Bounded native construction memory without changing noisy discrete sums."""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event import likelihood as module
from jimgw.core.single_event.heterodyne_moments import polynomial_moments
from jimgw.core.single_event.heterodyne_summary_tiles import NativeGridTilePlanner

jax.config.update("jax_enable_x64", True)


@dataclass
class _HostGrid:
    name: str
    grid: np.ndarray
    duration: float = 2.0

    def set_frequency_bounds(self, low, high):
        self.sliced_frequencies = self.grid[
            np.searchsorted(self.grid, low) : np.searchsorted(
                self.grid, high, side="right"
            )
        ]


class _BoundedNumpy:
    def __init__(self, limit):
        self.limit = limit

    def __getattr__(self, name):
        return getattr(np, name)

    def arange(self, *args, **kwargs):
        value = np.arange(*args, **kwargs)
        assert len(value) <= self.limit, "Host lattice proof allocated a full grid"
        return value

    def array_equal(self, first, second):
        assert len(first) <= self.limit, "Host containment proof compared a full grid"
        return np.array_equal(first, second)


@pytest.mark.parametrize("irregular", [False, True])
def test_host_nested_grid_is_actually_proved_in_bounded_chunks(monkeypatch, irregular):
    longest = np.arange(4, 29, dtype=float) / 2
    shorter = longest[3:20].copy()
    if irregular:
        longest[10] = np.nextafter(longest[10], np.inf)
    detectors = [_HostGrid("CE", shorter), _HostGrid("ET1", longest)]
    monkeypatch.setattr(module, "_GRID_PROOF_CHUNK_SIZE", 3)
    monkeypatch.setattr(module, "np", _BoundedNumpy(3))
    grid, identical, _ = module._set_and_merge_heterodyne_frequency_grids(
        detectors, 0, 20
    )
    assert isinstance(grid, np.ndarray)
    assert identical is False
    if irregular:
        np.testing.assert_array_equal(grid, np.unique(np.r_[longest, shorter]))
    else:
        assert grid is detectors[1].sliced_frequencies


def test_layout_scans_counts_without_materializing_chunk_plans(monkeypatch):
    def no_plan(*args, **kwargs):
        raise AssertionError("Metadata scan must not allocate TilePlan objects")

    monkeypatch.setattr(NativeGridTilePlanner, "plan", no_plan)
    planner, chunk, tiles, local, elements = module._summary_tile_layout(
        np.linspace(2, 10, 33), 128, 256, 1025, 64, 70
    )
    assert planner.tile_size > 0 and chunk == 64
    assert elements == tiles * (planner.tile_size * 70 + local)
    assert elements <= module._MAX_TILE_MATRIX_ELEMENTS


def test_floor_tile_size_cannot_override_memory_budget(monkeypatch):
    monkeypatch.setattr(module, "_MAX_TILE_MATRIX_ELEMENTS", 2260)
    planner, chunk, tiles, local, elements = module._summary_tile_layout(
        np.array([2.0, 10.0]), 128, 256, 1025, 1024, 70
    )
    assert chunk <= 32 and planner.tile_size <= 32
    assert elements == tiles * (planner.tile_size * 70 + local) <= 2260


def test_layout_counts_onehot_combine_and_rejects_impossible_budget(monkeypatch):
    monkeypatch.setattr(module, "_MAX_TILE_MATRIX_ELEMENTS", 512)
    planner, chunk, tiles, local, elements = module._summary_tile_layout(
        np.arange(33, dtype=float), 1, 0, 32, 32, 1
    )
    assert chunk < 32
    assert elements == tiles * (planner.tile_size + local) <= 512
    assert tiles * local > 0
    monkeypatch.setattr(module, "_MAX_TILE_MATRIX_ELEMENTS", 1)
    with pytest.raises(ValueError, match="cannot hold one native sample"):
        module._summary_tile_layout(np.array([0.0, 2.0]), 1, 0, 2, 2, 70)


@pytest.mark.parametrize(
    "first,count,expected_tiles,expected_local",
    [(655360, 267780097, 269, 32), (262144, 268173313, 342, 256)],
    ids=["CE-5Hz", "ET-2Hz"],
)
def test_full_network_layout_uses_exact_fixed_maximum_without_native_arrays(
    first, count, expected_tiles, expected_local
):
    # Only 1,025 bin edges and small descriptors are constructed here. The
    # physical-duration frequency/data/waveform vectors are never allocated.
    edges = np.asarray(
        module.HeterodynedTransientLikelihoodFD._make_binning_scheme(
            np.array([2.0, 2048.0]), 1024
        )
    )
    planner, chunk, tiles, local, elements = module._summary_tile_layout(
        edges, 131072.0, first, count, 131072, 70
    )
    assert chunk == 131072 and planner.tile_size == 512
    assert (tiles, local) == (expected_tiles, expected_local)
    assert tiles & (tiles - 1), "The fixed count need not be a power of two"
    assert elements == tiles * (512 * 70 + local)
    assert elements <= module._MAX_TILE_MATRIX_ELEMENTS
    full = planner.plan(first, chunk, padded_tiles=tiles, padded_local=local)
    # Both production bands have a one-sample inclusive tail at this size.
    assert (count - 1) % chunk == 0
    tail = planner.plan(first + count - 1, 1, padded_tiles=tiles, padded_local=local)
    assert full.starts.shape == tail.starts.shape == (expected_tiles,)
    assert full.local_ids.shape == tail.local_ids.shape == (expected_local,)
    assert full.covered_samples == chunk and tail.covered_samples == 1


def _stream(device_arrays, chunk_size):
    count, duration = 257, 128.0
    f = np.arange(256, 256 + count) / duration
    rng = np.random.default_rng(152)
    data = rng.normal(size=count) + 1j * rng.normal(size=count)
    psd = 1.0 + f / 10
    detector = SimpleNamespace(
        name="test",
        duration=duration,
        sliced_frequencies=jnp.asarray(f) if device_arrays else f,
        sliced_fd_data=jnp.asarray(data) if device_arrays else data,
        sliced_psd=jnp.asarray(psd) if device_arrays else psd,
    )
    likelihood = module.HeterodynedTransientLikelihoodFD.__new__(
        module.HeterodynedTransientLikelihoodFD
    )
    likelihood.reference_chunk_size = chunk_size
    likelihood.reference_parameters = {}
    likelihood.phasor_time_anchors = tuple(np.linspace(-0.2, 0.2, 21))
    likelihood._project_reference = lambda detector, frequencies, sky: sky["p"]
    waveform = lambda frequencies, parameters: {"p": jnp.exp(0.17j * frequencies)}
    edges = np.array([2.0, 4.0])
    return likelihood, detector, waveform, edges, f, data, psd


def test_non_power_of_two_fixed_tiles_preserve_noisy_moments_and_singleton_tail():
    likelihood, detector, waveform, _, f, data, psd = _stream(False, 128)
    # The first chunk intersects three bins, so its exact fixed maximum is
    # three tiles; later chunks and the singleton endpoint use inert padding.
    edges = np.array([2.0, 2.25, 2.5, 4.0])
    summary, bank, diagnostic = likelihood._compute_reference_coefficients_jax(
        detector,
        waveform,
        edges,
        interpolation_order=24,
        norm_order=16,
        device=jax.local_devices()[0],
    )
    assert diagnostic["tile_descriptor_shapes"] == [(128, 3)]
    assert diagnostic["chunks"] == diagnostic["tiled_chunks"] == 3
    assert diagnostic["native_frequency_samples"] == len(f) == 257
    assert diagnostic["compiled_executables"] == 1
    assert diagnostic["max_tile_matrix_elements"] <= module._MAX_TILE_MATRIX_ELEMENTS
    reference = np.exp(0.17j * f)
    a, b = polynomial_moments(f, data, psd, reference, edges, 24, 16)
    scale = 4.0 / detector.duration
    np.testing.assert_allclose(summary, scale * np.r_[a, b], rtol=3e-12, atol=1e-12)
    for index, anchor in enumerate(likelihood.phasor_time_anchors):
        expected, _ = polynomial_moments(
            f, data * np.exp(2j * np.pi * f * anchor), psd, reference, edges, 24, 0
        )
        np.testing.assert_allclose(
            bank[index], scale * expected, rtol=3e-12, atol=1e-12
        )


@pytest.mark.parametrize("device_arrays", [False, True])
def test_budgeted_chunks_preserve_all_moments_and_device_data(
    monkeypatch, device_arrays
):
    likelihood, detector, waveform, edges, f, data, psd = _stream(device_arrays, 256)
    original_get = jax.device_get

    def no_full_device_vector_get(value):
        if isinstance(value, jax.Array):
            assert value.ndim == 0, (
                "Device data/PSD/frequency vectors must not roundtrip to host"
            )
        return original_get(value)

    monkeypatch.setattr(jax, "device_get", no_full_device_vector_get)
    monkeypatch.setattr(module, "_MAX_TILE_MATRIX_ELEMENTS", 2260)
    summary, bank, diagnostic = likelihood._compute_reference_coefficients_jax(
        detector,
        waveform,
        edges,
        interpolation_order=24,
        norm_order=16,
        device=jax.local_devices()[0],
    )
    assert diagnostic["chunk_size"] == 32
    assert diagnostic["native_frequency_samples"] == len(f)
    assert diagnostic["max_tile_matrix_elements"] <= 2260
    assert diagnostic["tile_combine_elements"] > 0
    assert len(diagnostic["tile_descriptor_shapes"]) == 1
    assert diagnostic["compiled_executables"] == 1
    reference = np.exp(0.17j * f)
    a, b = polynomial_moments(f, data, psd, reference, edges, 24, 16)
    scale = 4.0 / detector.duration
    np.testing.assert_allclose(summary, scale * np.r_[a, b], rtol=3e-12, atol=1e-12)
    for index, anchor in enumerate(likelihood.phasor_time_anchors):
        expected, _ = polynomial_moments(
            f, data * np.exp(2j * np.pi * f * anchor), psd, reference, edges, 24, 0
        )
        np.testing.assert_allclose(
            bank[index], scale * expected, rtol=3e-12, atol=1e-12
        )


def test_fd_pages_are_released_only_after_completed_transfer_groups(monkeypatch):
    from jimgw.core.single_event import native_storage

    likelihood, detector, waveform, edges, *_ = _stream(False, 32)
    original_block = jax.block_until_ready
    completed = 0
    releases = []

    def synchronized(value):
        nonlocal completed
        result = original_block(value)
        completed += 1
        return result

    def release(view):
        assert completed > 0, "Mapped pages cannot be released during transfer"
        assert np.shares_memory(view, detector.sliced_fd_data)
        releases.append((completed, len(view)))

    monkeypatch.setattr(jax, "block_until_ready", synchronized)
    monkeypatch.setattr(native_storage, "release_mapped_pages", release)
    likelihood._compute_reference_coefficients_jax(
        detector,
        waveform,
        edges,
        interpolation_order=24,
        norm_order=16,
        device=jax.local_devices()[0],
    )
    assert releases == [(1, 32)] * 8 + [(2, 1)]
