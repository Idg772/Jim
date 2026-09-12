"""Bin-aligned FP64 tile reductions for discrete native heterodyne moments.

Planning uses only the bin edges and native grid metadata. Every retained
frequency sample is assigned once, including the final edge. Batched matrix
multiplication reduces samples within each tile; only tile results enter a
segmented reduction. No frequency-level atomic updates or [anchor,degree,N]
scratch plane is needed. Summation order differs from the NumPy reducer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TilePlan:
    """Small host descriptors; zero-length padding permits compiled reuse."""

    starts: np.ndarray
    lengths: np.ndarray
    bin_ids: np.ndarray
    tile_size: int
    sample_count: int
    n_bins: int
    active_tiles: int
    # Chunk-local bin table for the deterministic combine: ``tile_local[t]``
    # indexes ``local_ids``; padding tiles point past the table and padding
    # table slots hold ``n_bins`` so a dropping scatter ignores them.
    tile_local: np.ndarray = None
    local_ids: np.ndarray = None
    local_count: int = 0

    @property
    def padded_tiles(self):
        return len(self.starts)

    @property
    def covered_samples(self):
        return int(np.sum(self.lengths, dtype=np.int64))


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _native_boundary(edge, duration, *, right=False):
    """First native integer with i/duration >= edge (or > for right).

    Multiplication may round edge*duration across an integer. Correct the
    estimate against exactly the division that defines the native samples.
    This preserves searchsorted semantics at an edge and at its nextafter
    neighbours, including non-power-of-two durations.
    """
    scaled = edge * duration
    if not np.isfinite(scaled):
        raise ValueError("Native grid edge*duration is not finite")
    candidate = int(np.floor(scaled)) + 1 if right else int(np.ceil(scaled))
    before = (
        (lambda i: i / duration <= edge) if right else (lambda i: i / duration < edge)
    )
    while before(candidate):
        candidate += 1
    while not before(candidate - 1):
        candidate -= 1
    return candidate


class NativeGridTilePlanner:
    """Reuse corrected bin boundaries across a stream of native chunks.

    Only the initial construction performs scalar edge-to-integer conversion.
    Each plan clips the cached small boundary vector and visits nonempty bins.
    """

    def __init__(self, edges, duration, *, tile_size=512):
        edges = np.array(edges, dtype=np.float64, copy=True)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("edges must be finite and strictly increasing")
        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        self.tile_size = _integer(tile_size, "tile_size", 1)
        self.duration = duration
        self.n_bins = len(edges) - 1
        self.boundaries = np.asarray(
            [
                _native_boundary(float(edge), duration, right=i == self.n_bins)
                for i, edge in enumerate(edges)
            ],
            dtype=np.int64,
        )
        self.boundaries.setflags(write=False)

    def plan(
        self, first_native_index, sample_count, *, padded_tiles=None, padded_local=None
    ):
        """Return local chunk descriptors padded to a fixed tile count.

        Without ``padded_tiles`` the count is the next power of two of the
        active tiles; a caller planning a whole stream passes one common count
        so every chunk shares a compiled shape. ``padded_local`` likewise fixes
        the chunk-local bin table length.
        """
        first = _integer(first_native_index, "first_native_index", 0)
        count = _integer(sample_count, "sample_count", 1)
        if count > np.iinfo(np.int32).max:
            raise ValueError("A tile chunk must fit int32 local indices")
        if first + count > np.iinfo(np.int64).max:
            raise ValueError("Native grid indices must fit int64")
        # Clip before subtraction to avoid overflow for far outside edges.
        boundaries = np.clip(self.boundaries, first, first + count) - first
        occupied = np.flatnonzero(boundaries[1:] > boundaries[:-1])
        starts, lengths, bins = [], [], []
        for bin_id in occupied:
            begin, end = int(boundaries[bin_id]), int(boundaries[bin_id + 1])
            for start in range(begin, end, self.tile_size):
                starts.append(start)
                lengths.append(min(self.tile_size, end - start))
                bins.append(bin_id)
        active = len(starts)
        padded = 1 << max(0, active - 1).bit_length()
        if padded_tiles is not None:
            if padded_tiles < active:
                raise ValueError("padded_tiles is smaller than the active tile count")
            padded = int(padded_tiles)
        padding = padded - active
        local_index = {int(bin_id): i for i, bin_id in enumerate(occupied)}
        local_count = len(occupied)
        local_padded = 1 << max(0, local_count - 1).bit_length()
        if padded_local is not None:
            if padded_local < local_count:
                raise ValueError("padded_local is smaller than the occupied bin count")
            local_padded = int(padded_local)
        tile_local = [local_index[b] for b in bins] + [local_padded] * padding
        local_ids = [int(b) for b in occupied] + [self.n_bins] * (
            local_padded - local_count
        )
        return TilePlan(
            starts=np.asarray(starts + [0] * padding, dtype=np.int32),
            lengths=np.asarray(lengths + [0] * padding, dtype=np.int32),
            bin_ids=np.asarray(bins + [0] * padding, dtype=np.int32),
            tile_size=self.tile_size,
            sample_count=count,
            n_bins=self.n_bins,
            active_tiles=active,
            tile_local=np.asarray(tile_local, dtype=np.int32),
            local_ids=np.asarray(local_ids, dtype=np.int32),
            local_count=local_count,
        )


def plan_from_native_grid(
    edges, first_native_index, duration, sample_count, *, tile_size=512
):
    """One-shot plan for f[i]=(first_native_index+i)/duration.

    Reuse NativeGridTilePlanner for many chunks. The caller must verify the
    native-grid contract; arbitrary grids belong to the generic reducer.
    Input arrays may have tail padding: descriptors include real samples only.
    """
    return NativeGridTilePlanner(edges, duration, tile_size=tile_size).plan(
        first_native_index, sample_count
    )


@partial(jax.jit, static_argnames=("tile_size", "data_order", "norm_order"))
def compiled_summary_tiles(
    frequencies,
    data,
    psd,
    reference,
    edges,
    anchors,
    tile_starts,
    tile_lengths,
    tile_bins,
    *,
    tile_size,
    data_order,
    norm_order,
    tile_local=None,
    local_bins=None,
):
    """Return unnormalised (A_zero, B, A_anchors) from native noisy samples.

    With ``tile_local``/``local_bins`` (see ``TilePlan``) tile results are
    combined by a one-hot matrix product over the chunk's own bins and a
    unique-index scatter, which has a fixed reduction order on every backend.
    Without them the combine is a segmented sum, which may use atomics on
    accelerators.

    A has shape [data_order+1,bins], B [norm_order+1,bins], and the bank
    [anchors,data_order+1,bins]. All arithmetic and dot accumulation use
    float64/complex128. Zero-length tile padding is inert even when gathering
    an invalid out-of-band sample. Invalid in-band PSD produces NaNs.
    Descriptors are dynamic operands, so different chunks with the same
    padded tile count share a compiled executable.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("Native tile reduction requires jax_enable_x64=True")
    _integer(data_order, "data_order", 0)
    _integer(norm_order, "norm_order", 0)
    _integer(tile_size, "tile_size", 1)
    f = jnp.asarray(frequencies, dtype=jnp.float64)
    d = jnp.asarray(data, dtype=jnp.complex128)
    s = jnp.asarray(psd, dtype=jnp.float64)
    h = jnp.asarray(reference, dtype=jnp.complex128)
    bins = jnp.asarray(edges, dtype=jnp.float64)
    anchors = jnp.asarray(anchors, dtype=jnp.float64)
    if f.ndim != 1 or not f.size or any(value.shape != f.shape for value in (d, s, h)):
        raise ValueError("frequency/data/PSD/reference must be equal nonempty vectors")
    if bins.ndim != 1 or bins.size < 2 or anchors.ndim != 1:
        raise ValueError("edges and anchors must be vectors")
    if not (
        tile_starts.ndim == 1
        and tile_starts.shape == tile_lengths.shape == tile_bins.shape
    ):
        raise ValueError("tile descriptors must be equal-sized vectors")
    columns = jnp.arange(tile_size, dtype=jnp.int32)
    indices = tile_starts[:, None] + columns[None, :]
    valid = columns[None, :] < tile_lengths[:, None]
    indices = jnp.minimum(indices, f.size - 1)
    left, right = bins[tile_bins, None], bins[tile_bins + 1, None]
    ft = jnp.where(valid, f[indices], left)
    dt = jnp.where(valid, d[indices], 0)
    ht = jnp.where(valid, h[indices], 0)
    st = jnp.where(valid, s[indices], 1)
    st = jnp.where(jnp.isfinite(st) & (st > 0), st, jnp.nan)
    u = jnp.where(valid, (2 * ft - left - right) / (right - left), 0)
    max_order = max(data_order, norm_order)

    def power_step(power, _):
        return power * u, power

    _, powers = jax.lax.scan(power_step, jnp.ones_like(u), None, length=max_order + 1)
    # Both matrices have a tile batch dimension. Their contraction dimension
    # is only tile_size; no anchor*degree*frequency intermediate is formed.
    powers = jnp.transpose(powers, (1, 0, 2))
    times = jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), anchors))
    angle = 2 * jnp.pi * ft[:, None, :] * times[None, :, None]
    phase = jax.lax.complex(jnp.cos(angle), jnp.sin(angle))
    product = dt * ht.conj() / st
    weighted = phase * product[:, None, :]
    norm = (ht.real**2 + ht.imag**2) / st
    # Use one real FP64 GEMM for real/imaginary A planes and the norm plane.
    weights = jnp.concatenate((weighted.real, weighted.imag, norm[:, None, :]), axis=1)
    tile_values = jax.lax.dot_general(
        weights,
        powers,
        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float64,
    )
    if local_bins is None:
        reduced = jax.ops.segment_sum(
            tile_values,
            tile_bins,
            num_segments=bins.size - 1,
            indices_are_sorted=False,
        )
    else:
        onehot = (
            tile_local[None, :]
            == jnp.arange(local_bins.size, dtype=tile_local.dtype)[:, None]
        ).astype(jnp.float64)
        flat = tile_values.reshape(tile_values.shape[0], -1)
        local_sums = jnp.dot(onehot, flat, precision=jax.lax.Precision.HIGHEST)
        local_sums = local_sums.reshape((local_bins.size,) + tile_values.shape[1:])
        reduced = (
            jnp.zeros((bins.size - 1,) + tile_values.shape[1:], dtype=jnp.float64)
            .at[local_bins]
            .add(local_sums, mode="drop", unique_indices=True)
        )
    n_times = anchors.size + 1
    complex_a = jax.lax.complex(
        reduced[:, :n_times, : data_order + 1],
        reduced[:, n_times : 2 * n_times, : data_order + 1],
    )
    complex_a = jnp.transpose(complex_a, (1, 2, 0))
    return complex_a[0], reduced[:, -1, : norm_order + 1].T, complex_a[1:]


__all__ = [
    "NativeGridTilePlanner",
    "TilePlan",
    "compiled_summary_tiles",
    "plan_from_native_grid",
]
