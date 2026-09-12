import hashlib
import logging
import time
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms import CMA_ES
from jax.scipy.special import logsumexp
from jaxtyping import Array, Complex, Float
from ripplegw.interfaces import DistanceScaledWaveform, Waveform
from scipy.fft import fft as scipy_fft
from scipy.fft import next_fast_len

from jimgw.core.base import LikelihoodBase
from jimgw.core.constants import EARTH_RADIUS_LIGHT_S, MTSUN
from jimgw.core.prior import Prior, find_specific_prior
from jimgw.core.single_event.detector import Detector
from jimgw.core.single_event.heterodyne_moments import (
    polynomial_moments,
    validate_time_anchors,
)
from jimgw.core.single_event.marginalization_config import (
    DistanceMargConfig,
    HeterodyneTimeMargConfig,
    PhaseMargConfig,
    TimeMargConfig,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.utils import (
    FixedParameters,
    apply_fixed_parameters,
    complex_inner_product,
    inner_product,
)
from jimgw.core.transforms import NtoMTransform
from jimgw.core.utils import log_i0, round_up_to_power_of_two
from jimgw.typing import ComplexScalar, FloatLike, FloatScalar

logger = logging.getLogger(__name__)

_XG_PLAN_AUTHORITY = object()
_XG_QUALIFICATION_PLAN_AUTHORITY = object()


@dataclass(frozen=True)
class _VerifiedXGPlan:
    """Internal capability proving that an XG bin plan was verified."""

    bin_edges_sha256: str
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority is not _XG_PLAN_AUTHORITY:
            raise TypeError("XG plans must come from the verified pipeline builder")
        if len(self.bin_edges_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.bin_edges_sha256
        ):
            raise ValueError("XG bin edges must use a SHA-256 digest")


@dataclass(frozen=True)
class _QualificationXGPlan:
    """Internal capability restricted to deterministic qualification work."""

    bin_edges_sha256: str
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority is not _XG_QUALIFICATION_PLAN_AUTHORITY:
            raise TypeError(
                "XG qualification plans must come from the qualification builder"
            )
        if len(self.bin_edges_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.bin_edges_sha256
        ):
            raise ValueError("XG qualification bin edges must use a SHA-256 digest")


_LIKELIHOOD_OPTIMIZATION_AXES = frozenset(
    {"shared_frequency_grid", "detector_phasor", "real_inner_product"}
)
_MAX_DIRECT_SUM_PHASOR_ELEMENTS = 4_194_304
# Float64 elements of the bin-aligned tile matrices (powers, anchor weights,
# and the one-hot tile-combine matrix) for one native construction chunk:
# 16 Mi elements = 128 MiB; excludes waveform/compiler workspace and allocator
# pools. Tile widths below 32 make the GEMM degenerate; shrink chunks instead.
_MAX_TILE_MATRIX_ELEMENTS = 16 * 2**20
_MIN_TILE_SIZE = 32
_GRID_PROOF_CHUNK_SIZE = 262_144
# Degree of the per-bin polynomial ratio model.  Lobatto-node Vandermonde
# systems stay well conditioned up to this order.
_MAX_INTERPOLATION_ORDER = 8
# Taylor order of the analytic t_c phasor carried by the summary moments.
_MAX_PHASOR_MOMENT_ORDER = 16
_MAX_DIRECT_SUM_TIME_SAMPLES = 4_194_304
_MIN_RELATIVE_BIN_REFERENCE_RESPONSE = 1.0e-12


def _host_native_grid_is_uniform(grid, first_index, duration):
    """Prove the complete host lattice with at most one chunk of scratch."""
    for start in range(0, len(grid), _GRID_PROOF_CHUNK_SIZE):
        stop = min(start + _GRID_PROOF_CHUNK_SIZE, len(grid))
        part = grid[start:stop]
        expected = (np.arange(start, stop, dtype=grid.dtype) + first_index) / duration
        if not np.array_equal(part, expected) or np.any(part[1:] <= part[:-1]):
            return False
        if start and part[0] <= grid[start - 1]:
            return False
    return True


def _host_grid_is_contained(part, whole, offset):
    """Compare nested slices without a full-length equality mask."""
    for start in range(0, len(part), _GRID_PROOF_CHUNK_SIZE):
        stop = min(start + _GRID_PROOF_CHUNK_SIZE, len(part))
        if not np.array_equal(part[start:stop], whole[offset + start : offset + stop]):
            return False
    return True


@jax.jit
def _device_chunk_is_native(frequencies, first_index, duration, real_count):
    """Return one scalar; keep device frequency arrays on the device."""
    index = jnp.arange(frequencies.size, dtype=jnp.int64)
    expected = (index + first_index) / duration
    return jnp.all((index >= real_count) | (frequencies == expected))


def _summary_tile_layout(edges, duration, first_index, n_samples, chunk_size, planes):
    """Find one bounded executable shape without retaining stream descriptors.

    The metadata pass counts tiles directly from the small integer boundary
    vector. Execution regenerates only the current chunk's descriptors.
    """
    from jimgw.core.single_event.heterodyne_summary_tiles import NativeGridTilePlanner

    planners = {}
    while True:
        tile_size = min(512, chunk_size)
        tile_floor = min(_MIN_TILE_SIZE, chunk_size)
        while True:
            if tile_size not in planners:
                planners[tile_size] = NativeGridTilePlanner(
                    edges, duration, tile_size=tile_size
                )
            planner = planners[tile_size]
            max_tiles, max_local = 0, 0
            for start in range(0, n_samples, chunk_size):
                first = first_index + start
                stop = first + min(chunk_size, n_samples - start)
                lengths = np.diff(np.clip(planner.boundaries, first, stop))
                max_tiles = max(
                    max_tiles, int(np.sum((lengths + tile_size - 1) // tile_size))
                )
                max_local = max(max_local, int(np.count_nonzero(lengths)))
            # The whole-stream maximum already fixes one executable shape.
            # Rounding this batch dimension up can double padded arithmetic
            # and needlessly force smaller chunks under the matrix budget.
            padded_tiles = max(1, max_tiles)
            padded_local = 1 << max(0, max_local - 1).bit_length()
            matrix_elements = padded_tiles * (tile_size * planes + padded_local)
            if matrix_elements <= _MAX_TILE_MATRIX_ELEMENTS:
                return planner, chunk_size, padded_tiles, padded_local, matrix_elements
            if tile_size <= tile_floor:
                break
            tile_size = max(tile_floor, tile_size // 2)
        if chunk_size == 1:
            raise ValueError(
                "Tile matrix budget cannot hold one native sample and its combine matrix"
            )
        chunk_size = max(1, chunk_size // 2)


def _set_and_merge_heterodyne_frequency_grids(
    detectors: Sequence[Detector],
    f_min: float | dict[str, float],
    f_max: float | dict[str, float],
) -> tuple[Float[Array, " n_freq"], bool, FloatScalar]:
    """Apply detector bounds and return the canonical heterodyne frequency grid."""

    detector_frequencies = []
    for detector in detectors:
        detector_f_min = f_min[detector.name] if isinstance(f_min, dict) else f_min
        detector_f_max = f_max[detector.name] if isinstance(f_max, dict) else f_max
        detector.set_frequency_bounds(detector_f_min, detector_f_max)
        detector_frequencies.append(detector.sliced_frequencies)
    if not detector_frequencies:
        raise ValueError("heterodyne likelihood requires at least one detector")
    if any(len(frequencies) < 2 for frequencies in detector_frequencies):
        raise ValueError("Each detector frequency grid must contain at least 2 bins")

    grid_metadata = [
        (
            len(frequencies),
            float(jax.device_get(frequencies[0])),
            float(jax.device_get(frequencies[-1])),
            float(jax.device_get(frequencies[1] - frequencies[0])),
        )
        for frequencies in detector_frequencies
    ]
    spacings = [metadata[3] for metadata in grid_metadata]
    if not all(np.isclose(spacings[0], spacing) for spacing in spacings[1:]):
        raise ValueError("All detectors must have the same frequency spacing")

    first_grid = grid_metadata[0]
    if all(
        metadata[0] == first_grid[0]
        and metadata[1] == first_grid[1]
        and metadata[2] == first_grid[2]
        and metadata[3] == first_grid[3]
        for metadata in grid_metadata[1:]
    ):
        frequencies = detector_frequencies[0]
        identical_frequency_grids = True
    else:
        # Unequal analysis bounds often leave aligned slices of one native
        # Fourier grid (for example CE >= 5 Hz and ET >= 2 Hz). Prove that a
        # longest grid contains the others before reusing it. Metadata alone
        # cannot establish this: irregular interiors or a half-bin offset must
        # still take the general union path.
        longest_index = max(
            range(len(grid_metadata)), key=lambda i: grid_metadata[i][0]
        )
        longest = detector_frequencies[longest_index]
        n_longest, low, high, _ = grid_metadata[longest_index]
        nominal_spacing = (high - low) / (n_longest - 1)
        nested = np.isfinite(nominal_spacing) and nominal_spacing > 0.0
        offsets = []
        if nested:
            duration = float(
                getattr(detectors[longest_index], "duration", 1.0 / nominal_spacing)
            )
            nested = np.isfinite(duration) and duration > 0.0
        if nested:
            for size, start, stop, _ in grid_metadata:
                if not (
                    np.isfinite(start)
                    and np.isfinite(stop)
                    and low <= start <= stop <= high
                ):
                    nested = False
                    break
                offset = round((start - low) * duration)
                if offset < 0 or offset + size > n_longest:
                    nested = False
                    break
                offsets.append(offset)

        if nested and all(isinstance(g, np.ndarray) for g in detector_frequencies):
            # Host-resident grids: prove the lattice on the host, never moving a
            # full native grid to a device.
            is_native_uniform_grid = _host_native_grid_is_uniform
            is_contained_slice = _host_grid_is_contained

        elif nested:

            @jax.jit
            def is_native_uniform_grid(grid, first_index, duration):
                expected = (
                    jnp.arange(grid.size, dtype=grid.dtype) + first_index
                ) / duration
                return jnp.all(grid == expected) & jnp.all(grid[1:] > grid[:-1])

            @jax.jit
            def is_contained_slice(part, whole, offset):
                section = jax.lax.dynamic_slice_in_dim(whole, offset, part.size)
                return jnp.all(part == section)

        if nested:
            # These fused reductions return only scalars to the host. In
            # particular, neither the native grids nor their concatenation are
            # copied to NumPy. Exact lattice reconstruction also permits the
            # normal floating rounding of non-power-of-two durations.
            nested = bool(
                jax.device_get(
                    is_native_uniform_grid(longest, round(low * duration), duration)
                )
            )
            if nested:
                for index, (grid, offset) in enumerate(
                    zip(detector_frequencies, offsets, strict=True)
                ):
                    if index != longest_index and not bool(
                        jax.device_get(is_contained_slice(grid, longest, offset))
                    ):
                        nested = False
                        break
        if nested:
            frequencies = longest
        else:
            host_frequencies = [
                np.asarray(jax.device_get(frequencies))
                for frequencies in detector_frequencies
            ]
            union = np.unique(np.concatenate(host_frequencies))
            frequencies = (
                union
                if all(isinstance(g, np.ndarray) for g in detector_frequencies)
                else jnp.asarray(union)
            )
        identical_frequency_grids = False
    df = detector_frequencies[0][1] - detector_frequencies[0][0]
    return frequencies, identical_frequency_grids, df


@dataclass(frozen=True)
class _ZoomFFTPlan:
    """Static Bluestein factors for a local section of a zero-padded FFT."""

    input_chirp: np.ndarray
    kernel_fft: np.ndarray
    output_chirp: np.ndarray
    gather_indices: np.ndarray
    fft_size: int
    output_start: int


def _build_time_marginalization_fine_window(
    n_total: int,
    upsample_factor: int,
    duration: float,
    tc_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return coarse FFT candidates, their fine-time mask, and fine step."""

    n_fine = n_total * upsample_factor
    fine_step = duration / n_fine
    q_min = -(n_fine // 2)
    q_max = (n_fine - 1) // 2
    support_min = q_min * fine_step
    support_max = q_max * fine_step
    empty_candidates = np.empty(0, dtype=int)
    empty_mask = np.empty((upsample_factor, 0), dtype=bool)
    if (
        not tc_range[0] < tc_range[1]
        or tc_range[0] >= support_max
        or tc_range[1] <= support_min
    ):
        return empty_candidates, empty_mask, fine_step

    if tc_range[0] <= support_min:
        q_start = q_min
    else:
        q_start = max(q_min, int(np.floor(tc_range[0] / fine_step)) - 1)
    if tc_range[1] >= support_max:
        q_stop = q_max
    else:
        q_stop = min(q_max, int(np.ceil(tc_range[1] / fine_step)) + 1)
    nearby_q = np.arange(q_start, q_stop + 1, dtype=int)
    nearby_tc = nearby_q * fine_step
    valid_q = nearby_q[(nearby_tc > tc_range[0]) & (nearby_tc < tc_range[1])]
    fine_storage = np.where(valid_q >= 0, valid_q, valid_q + n_fine)
    fine_candidates = np.unique(fine_storage // upsample_factor)

    m_grid = (
        fine_candidates[None, :] * upsample_factor + np.arange(upsample_factor)[:, None]
    )
    q_grid = np.where(m_grid <= q_max, m_grid, m_grid - n_fine)
    fine_tc = q_grid * fine_step
    fine_mask = (fine_tc > tc_range[0]) & (fine_tc < tc_range[1])
    return fine_candidates, fine_mask, fine_step


def _build_time_marginalization_zoom_plan(
    n_total: int,
    upsample_factor: int,
    fine_candidates: np.ndarray,
) -> _ZoomFFTPlan:
    """Plan a ZoomFFT for the fine bins represented by the candidate grid.

    The phase-ramped implementation stores fine bin ``m`` at
    ``(m % upsample_factor, m // upsample_factor)``.  The requested bins form a
    local circular arc on the fine DFT grid.  Cutting that arc at its largest
    gap makes it one uniformly spaced section, which a Bluestein convolution
    can evaluate with two FFTs independent of the upsample factor.
    """

    if upsample_factor <= 1:
        raise ValueError("ZoomFFT planning requires upsample_factor > 1")
    if fine_candidates.size == 0:
        raise ValueError("ZoomFFT planning requires at least one candidate bin")

    n_fine = n_total * upsample_factor
    storage_grid = (
        fine_candidates[None, :] * upsample_factor + np.arange(upsample_factor)[:, None]
    )
    unique_storage = np.unique(storage_grid)
    # Cut the circular grid at its largest unused gap, including windows that
    # cross either the zero-time or signed-Nyquist storage seam.
    circular_gaps = np.diff(
        np.concatenate((unique_storage, unique_storage[:1] + n_fine))
    )
    cut_after = int(np.argmax(circular_gaps))
    q_start = int(unique_storage[(cut_after + 1) % unique_storage.size])
    relative_grid = np.remainder(storage_grid - q_start, n_fine)
    output_size = int(relative_grid.max()) + 1

    k = np.arange(max(n_total, output_size), dtype=np.int64)
    phase_period = 2 * n_fine
    # The quadratic Bluestein phases have this exact integer period.  Reducing
    # before exponentiation avoids precision loss from unnecessarily large angles.
    chirp = np.exp((-1j * np.pi / n_fine) * np.remainder(k**2, phase_period))
    input_k = k[:n_total]
    input_chirp = np.exp(
        (-1j * np.pi / n_fine)
        * np.remainder(input_k**2 + 2 * q_start * input_k, phase_period)
    )
    convolution_kernel = 1.0 / np.concatenate(
        (chirp[n_total - 1 : 0 : -1], chirp[:output_size])
    )
    fft_size = next_fast_len(n_total + output_size - 1)
    if fft_size is None:
        raise RuntimeError("Could not find a supported ZoomFFT convolution size")

    return _ZoomFFTPlan(
        input_chirp=input_chirp,
        kernel_fft=np.asarray(scipy_fft(convolution_kernel, n=fft_size)),
        output_chirp=chirp[:output_size],
        gather_indices=relative_grid.astype(np.int64),
        fft_size=fft_size,
        output_start=n_total - 1,
    )


class SingleEventLikelihood(LikelihoodBase):
    detectors: Sequence[Detector]
    waveform: Waveform
    fixed_parameters: FixedParameters
    trigger_time: float
    gmst: FloatScalar
    ref_dist: FloatLike

    @property
    def duration(self) -> FloatLike:
        """Duration of the data segment in seconds (taken from the first detector)."""
        return self.detectors[0].data.duration

    @property
    def detector_names(self) -> list[str]:
        """Names of the detectors used in this likelihood."""
        return [detector.name for detector in self.detectors]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[FixedParameters] = None,
    ) -> None:
        """
        Args:
            detectors (Sequence[Detector]): Detectors with initialized data and PSD.
            waveform (Waveform): Waveform model to evaluate.
            fixed_parameters (Optional[dict]): Parameters held constant during
                sampling. Values may be scalars or callables
                ``f(params) -> Float | dict``; callables are applied in insertion
                order. Defaults to None (no fixed parameters).

        Raises:
            ValueError: If any detector has uninitialized data or PSD.
        """
        # Check that all detectors have initialized data and PSD
        for detector in detectors:
            if detector.data.is_empty:
                raise ValueError(
                    f"Detector '{detector.name}' does not have initialized data. "
                    f"Please set data using detector.set_data() or detector.inject_signal() "
                    f"before initializing the likelihood."
                )
            if detector.psd.is_empty:
                raise ValueError(
                    f"Detector '{detector.name}' does not have initialized PSD. "
                    f"Please set PSD using detector.set_psd() or detector.load_and_set_psd() "
                    f"before initializing the likelihood."
                )

        self.detectors = detectors
        self.waveform = waveform
        self.fixed_parameters = fixed_parameters if fixed_parameters is not None else {}
        self.time_marginalization = False
        self.phase_marginalization = False
        self.distance_marginalization = False

    def _set_detector_frequency_bounds(
        self,
        f_min: float | dict[str, float],
        f_max: float | dict[str, float],
    ) -> list[Float[Array, " n_freq"]]:
        """Set per-detector frequency bounds and return the resulting grids."""
        detector_frequencies = []
        for detector in self.detectors:
            detector_f_min = f_min[detector.name] if isinstance(f_min, dict) else f_min
            detector_f_max = f_max[detector.name] if isinstance(f_max, dict) else f_max
            detector.set_frequency_bounds(detector_f_min, detector_f_max)
            detector_frequencies.append(detector.sliced_frequencies)
        return detector_frequencies

    def _prepare_parameters(self, params: dict[str, Float]) -> dict[str, Float]:
        """Add event metadata, marginalization defaults, and fixed parameters."""
        prepared_params = params.copy()
        prepared_params["trigger_time"] = self.trigger_time
        prepared_params["gmst"] = self.gmst
        if self.time_marginalization:
            prepared_params["t_c"] = 0.0
        if self.phase_marginalization:
            prepared_params["phase_c"] = 0.0
        if self.distance_marginalization:
            prepared_params["d_L"] = self.ref_dist
        apply_fixed_parameters(prepared_params, self.fixed_parameters)
        return prepared_params

    # --- direct evaluation ---

    def evaluate(self, params: dict[str, Float]) -> FloatScalar:
        """Prepare parameters and evaluate the likelihood.

        Constants are injected directly; callables receive the current params
        dict and may return a scalar or a dict (the matching key is extracted).
        Callables are applied in insertion order.
        """
        return self._evaluate(self._prepare_parameters(params))

    @abstractmethod
    def _evaluate(self, params: dict[str, Float]) -> FloatScalar:
        """Core likelihood evaluation method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    # --- waveform-cache evaluation ---

    def generate_waveform(self, params: dict[str, Float]) -> Any:
        """Generate a reusable waveform cache.

        Evaluated at unit distance when ``waveform_caches_distance`` is True,
        so ``d_L`` is not a cache dependency; otherwise ``d_L`` is a
        dependency like any other waveform parameter. Every other effective
        waveform input is always a dependency, regardless of distance support.

        Args:
            params (dict[str, Float]): Source parameters to build the cache from.

        Returns:
            Any: An opaque cache understood by this likelihood's
                `evaluate_from_waveform`.
        """
        return self._generate_waveform(self._prepare_parameters(params))

    @abstractmethod
    def _generate_waveform(self, params: dict[str, Float]) -> Any:
        """Build the waveform cache from already-prepared parameters."""
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate_from_waveform(
        self,
        params: dict[str, Float],
        waveform_cache: Any,
    ) -> FloatScalar:
        """Evaluate this likelihood from a waveform cache."""
        return self._evaluate_from_waveform(
            self._prepare_parameters(params), waveform_cache
        )

    @abstractmethod
    def _evaluate_from_waveform(
        self,
        params: dict[str, Float],
        waveform_cache: Any,
    ) -> FloatScalar:
        """Evaluate the likelihood from a generated waveform cache."""
        raise NotImplementedError("Subclasses must implement this method.")

    # --- cache machinery (distance-scaling optimization) ---

    @property
    def waveform_caches_distance(self) -> bool:
        """Whether a waveform cache can be reused across ``d_L`` changes.

        Derived from the waveform's type by default, so subclasses may
        override it to control the behavior directly. Consumed by the
        waveform-cache block-dependency inference (see
        ``jimgw.core.single_event.blocked_likelihood``): when True, ``d_L``
        is excluded from the cache's dependency set, so a ``d_L``-only
        proposal block reuses the cache instead of rebuilding it.
        """
        return isinstance(self.waveform, DistanceScaledWaveform)

    @property
    def waveform_cacheable_parameter_names(self) -> frozenset[str]:
        """Waveform inputs that may change without invalidating the cache.

        Distance-scaled waveforms always admit the existing unit-distance
        cache.  Waveforms with a finer analytic factorization may additionally
        declare ``cacheable_parameter_names`` and implement
        ``build_waveform_cache`` / ``waveform_from_cache``.  The latter is
        useful for orientation parameters whose cheap harmonic projection can
        be separated from an expensive intrinsic carrier.
        """

        declared_names: set[str] = set(
            getattr(self.waveform, "cacheable_parameter_names", ())
        )
        builder = getattr(self.waveform, "build_waveform_cache", None)
        reconstructor = getattr(self.waveform, "waveform_from_cache", None)
        has_builder = callable(builder)
        has_reconstructor = callable(reconstructor)
        if declared_names and not (has_builder and has_reconstructor):
            raise TypeError(
                "A waveform declaring cacheable_parameter_names must implement "
                "both build_waveform_cache and waveform_from_cache."
            )
        if has_builder != has_reconstructor:
            raise TypeError(
                "Custom waveform caching requires both build_waveform_cache "
                "and waveform_from_cache."
            )
        unknown_names = declared_names - set(self.waveform.parameter_names)
        if unknown_names:
            raise ValueError(
                "Cacheable parameter names "
                f"{sorted(unknown_names)} are not waveform parameters."
            )

        names = declared_names
        if self.waveform_caches_distance:
            names.add("d_L")
        return frozenset(names)

    @property
    def waveform_cache_dependency_parameter_names(self) -> frozenset[str]:
        """Likelihood-space inputs that invalidate the source cache.

        The normal carrier dependencies are the waveform inputs that cannot be
        reconstructed from its reusable cache.  A waveform can also declare
        ``emission_time_parameter_names`` when its cached payload contains an
        intrinsic frequency-to-time map.  Those timing inputs remain cache
        dependencies even when the carrier cache can otherwise reconstruct
        them cheaply.

        The blocked-likelihood adapter maps these likelihood-space names back
        through the configured transforms to price proposal blocks.
        """

        carrier_dependencies = set(self.waveform.parameter_names) - set(
            self.waveform_cacheable_parameter_names
        )
        timing_dependencies = set(
            getattr(self.waveform, "emission_time_parameter_names", ())
        )
        dependencies = carrier_dependencies | timing_dependencies
        if getattr(self, "phase_marginalization", False):
            dependencies.discard("phase_c")
        if getattr(self, "time_marginalization", False):
            dependencies.discard("t_c")
        if getattr(self, "distance_marginalization", False):
            dependencies.discard("d_L")
        return frozenset(dependencies)

    def _waveform_sky_for_cache(
        self,
        frequencies: Float[Array, " n_freq"],
        params: dict[str, Float],
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Evaluate sky-frame polarizations for a waveform cache.

        Evaluated at ``d_L = 1`` when ``waveform_caches_distance`` is True, so
        ``d_L`` is not a cache dependency; otherwise ``d_L`` is a dependency
        like any other waveform parameter.
        """
        custom_builder = getattr(self.waveform, "build_waveform_cache", None)
        if callable(custom_builder):
            return cast(
                dict[str, Complex[Array, " n_freq"]],
                custom_builder(frequencies, params),
            )

        # Not using `waveform_caches_distance` for passing type checks
        if isinstance(self.waveform, DistanceScaledWaveform):
            return self.waveform.at_unit_distance(frequencies, params)
        return self.waveform(frequencies, params)

    def _waveform_sky_from_cache(
        self,
        frequencies: Float[Array, " n_freq"],
        cached_polarizations: Any,
        params: dict[str, Float],
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Recover physical-distance polarizations from a cache entry.

        Rescales by ``1 / d_L`` when ``waveform_caches_distance`` is True;
        otherwise returns the cache unchanged, since ``d_L`` was already
        baked in by ``_waveform_sky_for_cache``.
        """
        custom_reconstructor = getattr(self.waveform, "waveform_from_cache", None)
        if callable(custom_reconstructor):
            return cast(
                dict[str, Complex[Array, " n_freq"]],
                custom_reconstructor(frequencies, params, cached_polarizations),
            )

        if not self.waveform_caches_distance:
            return cached_polarizations
        distance_scale = 1.0 / params["d_L"]
        return {
            polarization: strain * distance_scale
            for polarization, strain in cached_polarizations.items()
        }


class ZeroLikelihood(LikelihoodBase):
    """Trivial likelihood that always returns zero.

    Useful for prior-only sampling or debugging.
    """

    def __init__(self) -> None:
        pass

    def evaluate(self, params: dict[str, Float]) -> FloatScalar:
        """Return zero regardless of the parameters.

        Args:
            params (dict[str, Float]): Ignored.

        Returns:
            FloatScalar: Always 0.0.
        """
        return jnp.zeros(())


# ---------------------------------------------------------------------------
# Unified transient likelihood
# ---------------------------------------------------------------------------
class TransientLikelihoodFD(SingleEventLikelihood):
    """Frequency-domain transient gravitational wave likelihood.

    Supports optional analytic marginalization over coalescence time, phase,
    and/or luminosity distance via typed config objects.  Each marginalization
    mode is activated by passing the corresponding config object (or a plain
    dict shorthand) to the relevant parameter.

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        fixed_parameters: Parameters held constant during sampling.  Values
            may be constants or callables ``f(params) -> Float | dict``;
            callables are applied in insertion order.  See the likelihood
            tutorial for details and examples.
        f_min: Minimum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.
        f_max: Maximum frequency for likelihood evaluation.
            Can be a single float or a per-detector dictionary.
        trigger_time: GPS time of the event trigger.
        time_marginalization: If provided, marginalize over coalescence time
            ``t_c``.  Pass a [`TimeMargConfig`][jimgw.core.single_event.likelihood.TimeMargConfig]
            object, a plain dict (e.g. ``{"tc_range": (-0.1, 0.1)}``), or ``True``
            (shorthand for ``TimeMargConfig()``).  ``False`` or the default ``None``
            disables time marginalization.
        phase_marginalization: If provided, marginalize over coalescence phase
            ``phase_c``.  Pass a [`PhaseMargConfig`][jimgw.core.single_event.likelihood.PhaseMargConfig]
            object, a plain dict ``{}``, or ``True`` (shorthand for ``PhaseMargConfig()``).
            ``False`` or the default ``None`` disables phase marginalization.
        distance_marginalization: If provided, marginalize over luminosity
            distance ``d_L``.  Pass a [`DistanceMargConfig`][jimgw.core.single_event.likelihood.DistanceMargConfig]
            object or a plain dict (e.g. ``{"distance_prior": prior, "n_dist_points": 10000}``).
            ``False`` or the default ``None`` disables distance marginalization.
            ``True`` is not supported — ``distance_prior`` has no default; pass a
            dict or `DistanceMargConfig` instead.

    Example:
        >>> likelihood = TransientLikelihoodFD(
        ...     detectors, waveform,
        ...     f_min=20, f_max=1024, trigger_time=1234567890,
        ...     phase_marginalization=True,
        ...     time_marginalization={"tc_range": (-0.1, 0.1)},
        ... )
        >>> logL = likelihood.evaluate(params)
    """

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[FixedParameters] = None,
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = jnp.inf,
        trigger_time: float = 0,
        time_marginalization: Optional[Union[TimeMargConfig, dict, bool]] = None,
        phase_marginalization: Optional[Union[PhaseMargConfig, dict, bool]] = None,
        distance_marginalization: Optional[
            Union[DistanceMargConfig, dict, bool]
        ] = None,
        likelihood_optimizations: bool = True,
        likelihood_optimization_axes: Optional[Mapping[str, bool]] = None,
    ) -> None:
        super().__init__(detectors, waveform, fixed_parameters)
        if type(likelihood_optimizations) is not bool:
            raise TypeError("likelihood_optimizations must be a bool")
        self.likelihood_optimizations = likelihood_optimizations
        optimization_axes = {
            name: likelihood_optimizations for name in _LIKELIHOOD_OPTIMIZATION_AXES
        }
        if likelihood_optimization_axes is not None:
            unknown = set(likelihood_optimization_axes) - _LIKELIHOOD_OPTIMIZATION_AXES
            if unknown:
                raise ValueError(
                    "unknown likelihood optimization axes: "
                    + ", ".join(sorted(unknown))
                )
            invalid = [
                name
                for name, enabled in likelihood_optimization_axes.items()
                if type(enabled) is not bool
            ]
            if invalid:
                raise TypeError(
                    "likelihood optimization axes must be bools: "
                    + ", ".join(sorted(invalid))
                )
            optimization_axes.update(likelihood_optimization_axes)
        self.likelihood_optimization_axes = optimization_axes

        # --- frequency setup ---
        _frequencies = self._set_detector_frequency_bounds(f_min, f_max)

        assert all(
            jnp.isclose(
                _frequencies[0][1] - _frequencies[0][0],
                freq[1] - freq[0],
            )
            for freq in _frequencies
        ), "All detectors must have the same frequency spacing."

        self.df = _frequencies[0][1] - _frequencies[0][0]
        self.frequencies = jnp.unique(jnp.concatenate(_frequencies))
        # ``jnp.isin`` materialises an ``n_union x n_detector`` comparison
        # (one byte per pair): 62 GiB for a 259k-sample band and ~280 TB for
        # a 16.7M-sample XG band.  NumPy's sort-based ``isin`` needs only
        # ``O(n log n)`` host memory, and the masks are static anyway.
        host_frequencies = np.asarray(jax.device_get(self.frequencies))
        self.frequency_masks = [
            jnp.asarray(
                np.isin(
                    host_frequencies,
                    np.asarray(jax.device_get(detector.sliced_frequencies)),
                )
            )
            for detector in detectors
        ]
        # All-True masks mean every detector shares the union grid: the mask
        # gather is the identity and the scatter-add is a dense add. Detected
        # at trace time so the fast path emits no gather/scatter ops at all.
        self._identical_masks = optimization_axes["shared_frequency_grid"] and all(
            bool(jnp.all(mask)) for mask in self.frequency_masks
        )

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(self.trigger_time)

        # --- resolve marginalization inputs ---
        if isinstance(time_marginalization, dict):
            time_marginalization = TimeMargConfig(**time_marginalization)
        elif time_marginalization is True:
            time_marginalization = TimeMargConfig()
        elif not time_marginalization:
            time_marginalization = None

        if isinstance(phase_marginalization, dict):
            phase_marginalization = PhaseMargConfig(**phase_marginalization)
        elif phase_marginalization is True:
            phase_marginalization = PhaseMargConfig()
        elif not phase_marginalization:
            phase_marginalization = None

        if isinstance(distance_marginalization, dict):
            distance_marginalization = DistanceMargConfig(**distance_marginalization)
        elif not distance_marginalization:
            distance_marginalization = None
        elif distance_marginalization is True:
            raise ValueError(
                "distance_marginalization=True is not supported because "
                "`distance_prior` has no default.  Pass a dict with `distance_prior` "
                "or a DistanceMargConfig instance instead."
            )

        # --- marginalization flags ---
        self.time_marginalization = time_marginalization is not None
        self.phase_marginalization = phase_marginalization is not None
        self.distance_marginalization = distance_marginalization is not None

        response_is_time_dependent = bool(
            getattr(self.waveform, "time_dependent_response", False)
            or getattr(self.waveform, "response_is_time_dependent", False)
            or any(
                getattr(detector, "time_dependent_response", False)
                or getattr(detector, "response_is_time_dependent", False)
                for detector in self.detectors
            )
        )
        if self.time_marginalization and response_is_time_dependent:
            raise ValueError(
                "dense FFT time marginalization freezes the time-dependent "
                "detector response; sample t_c explicitly or use the guarded "
                "heterodyne direct-sum path"
            )

        if self.time_marginalization and self.distance_marginalization:
            raise NotImplementedError(
                "Joint time + distance marginalization is not yet supported."
            )

        if time_marginalization is not None:
            self._init_time_marginalization(time_marginalization)
        if self.phase_marginalization:
            self._init_phase_marginalization()
        if distance_marginalization is not None:
            self._init_distance_marginalization(distance_marginalization)

        self._install_time_marg_data_weights()

    def _install_time_marg_data_weights(self) -> None:
        """Precompute data weights for the time-marg identical-masks fast path.

        Hoists the per-tick `4 * conj(d) / S * df` and `1 / S` divides
        (computed once here, after every detector's frequency bounds are
        set) so the elementwise accumulation in `_likelihood` becomes
        multiply-only. Only the identical-masks time-marg fast path
        consumes these.

        Called from `__init__`. Third-party construction routines that
        bypass `__init__` (e.g. via `__new__`) must call this explicitly
        once `self.detectors`, `self.df`, `self.time_marginalization`, and
        `self._identical_masks` are set, or the `_likelihood` fast path
        falls back to its pre-precompute numerics (see `_likelihood`).
        """
        self._weighted_conj_data: list[Complex[Array, " n_freq"]] = []
        self._inverse_sliced_psd: list[Float[Array, " n_freq"]] = []
        if self.time_marginalization and self._identical_masks:
            self._weighted_conj_data = [
                4.0 * self.df * jnp.conj(ifo.sliced_fd_data) / ifo.sliced_psd
                for ifo in self.detectors
            ]
            self._inverse_sliced_psd = [1.0 / ifo.sliced_psd for ifo in self.detectors]

    # --- direct evaluation ---

    def _evaluate(self, params: dict[str, Float]) -> FloatScalar:
        waveform_sky = self.waveform(self.frequencies, params)
        return self._likelihood(params, waveform_sky)

    # --- waveform-cache evaluation ---

    def _generate_waveform(
        self, params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Generate reusable sky-frame waveform polarizations.

        Evaluated at unit distance when ``waveform_caches_distance`` is True,
        so ``d_L`` is not a cache dependency; otherwise ``d_L`` is a
        dependency like any other waveform parameter.
        """
        return self._waveform_sky_for_cache(self.frequencies, params)

    def _evaluate_from_waveform(
        self,
        params: dict[str, Float],
        waveform_cache: dict[str, Complex[Array, " n_freq"]],
    ) -> FloatScalar:
        """Core likelihood evaluation from a pre-generated waveform cache."""
        waveform_sky = self._waveform_sky_from_cache(
            self.frequencies, waveform_cache, params
        )
        return self._likelihood(params, waveform_sky)

    # --- shared likelihood core ---

    def _sliced_waveform_sky(
        self,
        waveform_sky: dict[str, Complex[Array, " n_freq"]],
        detector_index: int,
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Waveform restricted to one detector's frequency grid."""
        if self._identical_masks:
            return waveform_sky
        mask = self.frequency_masks[detector_index]
        return {key: waveform_sky[key][mask] for key in waveform_sky}

    def _real_inner_product(
        self,
        h1: Complex[Array, " n_freq"],
        h2: Complex[Array, " n_freq"],
        psd: Float[Array, " n_freq"],
    ) -> FloatScalar:
        """Select optimized or pre-optimization real reduction semantics."""

        if self.likelihood_optimization_axes["real_inner_product"]:
            return inner_product(h1, h2, psd, self.df)
        return complex_inner_product(h1, h2, psd, self.df).real

    def _likelihood(
        self,
        params: dict[str, Float],
        waveform_sky: dict[str, Complex[Array, " n_freq"]],
    ) -> FloatScalar:
        """Core likelihood computation from a physical-distance sky-frame waveform."""

        # --- choose accumulation type based on flags ---
        if self.time_marginalization:
            log_likelihood: FloatScalar = jnp.zeros(())
            if self._identical_masks:
                # One elementwise pass over the shared grid: the detector-summed
                # data integrand and the real |h|^2/S integrand. The barrier
                # stops XLA's reduce-fusion emitter (which caps kernels at 32
                # registers for occupancy) from swallowing this f64/complex
                # chain; it compiles as an unconstrained elementwise kernel and
                # the sums below stay trivially bandwidth-bound.
                n_freq = len(self.frequencies)
                complex_d_inner_h = jnp.zeros(n_freq, dtype=jnp.complex128)
                hh_over_psd = jnp.zeros(n_freq)
                # Third-party constructors that bypass `__init__` (e.g. via
                # `__new__`) may not have run `_install_time_marg_data_weights`;
                # fall back to the original explicit-division numerics rather
                # than raising on the missing precomputed weights.
                weighted_conj_data = getattr(self, "_weighted_conj_data", None)
                use_precomputed_weights = bool(weighted_conj_data)
                for i, ifo in enumerate(self.detectors):
                    h_dec = ifo.fd_response(
                        ifo.sliced_frequencies,
                        self._sliced_waveform_sky(waveform_sky, i),
                        params,
                        optimize=self.likelihood_optimization_axes["detector_phasor"],
                    )
                    if use_precomputed_weights:
                        complex_d_inner_h = complex_d_inner_h + (
                            h_dec * weighted_conj_data[i]
                        )
                        hh_over_psd = hh_over_psd + (
                            (h_dec.real**2 + h_dec.imag**2)
                            * self._inverse_sliced_psd[i]
                        )
                    else:
                        complex_d_inner_h = complex_d_inner_h + (
                            4
                            * h_dec
                            * jnp.conj(ifo.sliced_fd_data)
                            / ifo.sliced_psd
                            * self.df
                        )
                        hh_over_psd = hh_over_psd + (
                            (h_dec.real**2 + h_dec.imag**2) / ifo.sliced_psd
                        )
                complex_d_inner_h, hh_over_psd = jax.lax.optimization_barrier(
                    (complex_d_inner_h, hh_over_psd)
                )
                # sum_k (h_k|h_k)/2 = 2 df * sum(|h|^2/S)
                log_likelihood += -(2.0 * self.df) * jnp.sum(hh_over_psd)
            else:
                complex_d_inner_h = jnp.zeros(
                    len(self.frequencies), dtype=jnp.complex128
                )
                for i, ifo in enumerate(self.detectors):
                    psd = ifo.sliced_psd
                    waveform_sky_ifo = self._sliced_waveform_sky(waveform_sky, i)
                    h_dec = ifo.fd_response(
                        ifo.sliced_frequencies,
                        waveform_sky_ifo,
                        params,
                        optimize=self.likelihood_optimization_axes["detector_phasor"],
                    )
                    complex_d_inner_h = complex_d_inner_h.at[
                        self.frequency_masks[i]
                    ].add(4 * h_dec * jnp.conj(ifo.sliced_fd_data) / psd * self.df)
                    optimal_SNR = self._real_inner_product(h_dec, h_dec, psd)
                    log_likelihood += -optimal_SNR / 2

            if self.phase_marginalization:
                # joint time + phase marginalization
                log_likelihood += self._reduce_phase_time(complex_d_inner_h)
            else:
                # time only marginalization
                log_likelihood += self._reduce_time(complex_d_inner_h)
            return log_likelihood

        elif self.phase_marginalization or self.distance_marginalization:
            # Need complex or real accumulation across detectors
            complex_d_inner_h: ComplexScalar = jnp.zeros((), dtype=jnp.complex128)
            match_filter_snr: FloatScalar = jnp.zeros(())
            optimal_snr: FloatScalar = jnp.zeros(())

            for i, ifo in enumerate(self.detectors):
                psd = ifo.sliced_psd
                waveform_sky_ifo = self._sliced_waveform_sky(waveform_sky, i)
                h_dec = ifo.fd_response(
                    ifo.sliced_frequencies,
                    waveform_sky_ifo,
                    params,
                    optimize=self.likelihood_optimization_axes["detector_phasor"],
                )
                if self.phase_marginalization:
                    complex_d_inner_h += complex_inner_product(
                        h_dec, ifo.sliced_fd_data, psd, self.df
                    )
                else:
                    match_filter_snr += self._real_inner_product(
                        h_dec, ifo.sliced_fd_data, psd
                    )
                optimal_snr += self._real_inner_product(h_dec, h_dec, psd)

            if self.phase_marginalization and self.distance_marginalization:
                # joint phase + distance marginalization
                return self._reduce_phase_distance(complex_d_inner_h, optimal_snr)
            elif self.phase_marginalization:
                # phase only marginalization
                return self._reduce_phase(complex_d_inner_h, optimal_snr)
            else:
                # distance only marginalization
                return self._reduce_distance(match_filter_snr, optimal_snr)

        else:
            # No marginalization
            log_likelihood: FloatScalar = jnp.zeros(())
            for i, ifo in enumerate(self.detectors):
                psd = ifo.sliced_psd
                waveform_sky_ifo = self._sliced_waveform_sky(waveform_sky, i)
                h_dec = ifo.fd_response(
                    ifo.sliced_frequencies,
                    waveform_sky_ifo,
                    params,
                    optimize=self.likelihood_optimization_axes["detector_phasor"],
                )
                match_filter_SNR = self._real_inner_product(
                    h_dec, ifo.sliced_fd_data, psd
                )
                optimal_SNR = self._real_inner_product(h_dec, h_dec, psd)
                log_likelihood += match_filter_SNR - optimal_SNR / 2
            return log_likelihood

    # --- time marginalization helpers ---

    def _init_time_marginalization(self, config: TimeMargConfig) -> None:
        if "t_c" in self.fixed_parameters:
            raise ValueError("Cannot have t_c fixed while marginalizing over t_c")
        self.tc_range = config.tc_range
        fs = self.detectors[0].data.sampling_frequency
        duration = float(self.detectors[0].data.duration)
        self.tc_array = jnp.fft.fftfreq(int(duration * fs / 2), 1.0 / duration)
        tc_array = np.asarray(self.tc_array)
        tc_window = np.flatnonzero(
            (tc_array > self.tc_range[0]) & (tc_array < self.tc_range[1])
        )
        self._tc_window_indices = jnp.asarray(tc_window)
        self.tc_upsample = int(config.upsample_factor)
        fine_candidates: np.ndarray | None = None
        if self.tc_upsample == 1 and tc_window.size == 0:
            raise ValueError(
                f"time_marginalization tc_range {self.tc_range} contains no FFT "
                "time samples; widen the range."
            )
        if self.tc_upsample > 1:
            n_total = len(self.tc_array)
            fine_candidates, fine_mask, _ = _build_time_marginalization_fine_window(
                n_total,
                self.tc_upsample,
                duration,
                self.tc_range,
            )
            self._tc_fine_candidate_indices = jnp.asarray(fine_candidates)
            if not fine_mask.any():
                raise ValueError(
                    f"time_marginalization tc_range {self.tc_range} contains no "
                    "FFT time samples; widen the range."
                )
            self._tc_fine_mask = jnp.asarray(fine_mask)
        self.pad_low = jnp.zeros(int(self.frequencies[0] * duration))
        n_pad_high = int(
            (fs / 2.0 - 1.0 / duration - float(self.frequencies[-1])) * duration
        )
        self.pad_high = jnp.zeros(max(0, n_pad_high))
        padded_size = len(self.pad_low) + len(self.frequencies) + len(self.pad_high)
        if padded_size != len(self.tc_array):
            raise ValueError(
                "time_marginalization requires a one-sided frequency grid that "
                "excludes the Nyquist endpoint; lower f_max by one frequency bin"
            )
        if self.tc_upsample > 1:
            assert fine_candidates is not None
            zoom_plan = _build_time_marginalization_zoom_plan(
                padded_size,
                self.tc_upsample,
                fine_candidates,
            )
            self._tc_zoom_input_chirp = jnp.asarray(zoom_plan.input_chirp)
            self._tc_zoom_kernel_fft = jnp.asarray(zoom_plan.kernel_fft)
            self._tc_zoom_output_chirp = jnp.asarray(zoom_plan.output_chirp)
            self._tc_zoom_gather_indices = jnp.asarray(zoom_plan.gather_indices)
            self._tc_zoom_fft_size = zoom_plan.fft_size
            self._tc_zoom_output_start = zoom_plan.output_start

    def _windowed_fft(
        self, complex_d_inner_h: Float[Array, " n_freq"]
    ) -> Complex[Array, "upsample n_window"]:
        """Evaluate the tc-window matched filter on a fine sub-grid.

        For ``U > 1``, a local Bluestein/ZoomFFT returns the same band-limited
        samples as a ``U``-times zero-padded FFT without computing the unused
        remainder of that fine grid.
        """

        padded = jnp.concatenate((self.pad_low, complex_d_inner_h, self.pad_high))
        if self.tc_upsample == 1:
            fft_d_inner_h = jnp.fft.fft(padded, norm="backward")
            return fft_d_inner_h[self._tc_window_indices][None, :]

        transformed = jnp.fft.fft(
            padded * self._tc_zoom_input_chirp,
            n=self._tc_zoom_fft_size,
            norm="backward",
        )
        convolved = jnp.fft.ifft(
            transformed * self._tc_zoom_kernel_fft,
            n=self._tc_zoom_fft_size,
            norm="backward",
        )
        output_size = len(self._tc_zoom_output_chirp)
        local_window = jax.lax.dynamic_slice_in_dim(
            convolved,
            self._tc_zoom_output_start,
            output_size,
        )
        local_window *= self._tc_zoom_output_chirp
        return local_window[self._tc_zoom_gather_indices]

    def _reduce_time(self, complex_d_inner_h: Float[Array, " n_freq"]) -> FloatScalar:
        """FFT-based time marginalization (real part)."""
        if self.tc_upsample > 1:
            window = self._windowed_fft(complex_d_inner_h).real
            return logsumexp(jnp.where(self._tc_fine_mask, window, -jnp.inf)) - jnp.log(
                len(self.tc_array) * self.tc_upsample
            )

        complex_d_inner_h_positive_f = jnp.concatenate(
            (self.pad_low, complex_d_inner_h, self.pad_high)
        )
        fft_d_inner_h = jnp.fft.fft(complex_d_inner_h_positive_f, norm="backward")
        window = fft_d_inner_h.real[self._tc_window_indices]
        return logsumexp(window) - jnp.log(len(self.tc_array))

    # --- phase marginalization helpers ---

    def _init_phase_marginalization(self) -> None:
        if "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

    def _reduce_phase(
        self,
        complex_d_inner_h: complex | ComplexScalar,
        optimal_snr: FloatScalar,
    ) -> FloatScalar:
        """Phase marginalization via modified Bessel function (Thrane & Talbot 2019, Eq. 24)."""
        return -optimal_snr / 2 + log_i0(jnp.absolute(complex_d_inner_h))

    # --- distance marginalization helpers ---

    def _init_distance_marginalization(self, config: DistanceMargConfig) -> None:
        distance_prior = config.distance_prior
        n_dist_points = config.n_dist_points
        ref_dist = config.ref_dist

        if "d_L" in self.fixed_parameters:
            raise ValueError("Cannot have d_L fixed while marginalising over d_L")

        if list(distance_prior.parameter_names) != ["d_L"]:
            raise ValueError(
                f"distance_prior must be a 1D prior with parameter_names=['d_L'], "
                f"got parameter_names={list(distance_prior.parameter_names)}."
            )

        bounds = distance_prior.get_bounds()
        if bounds is None:
            raise ValueError(
                "The d_L sub-prior must have xmin and xmax attributes. "
                "Use a bounded prior such as PowerLawPrior or UniformPrior."
            )
        dist_min, dist_max = bounds

        if dist_min <= 0:
            raise ValueError(
                "The d_L prior's xmin must be > 0 (distance must be positive)"
            )
        if dist_max <= dist_min:
            raise ValueError("The d_L prior's xmax must be greater than xmin")

        if ref_dist is None:
            self.ref_dist = (dist_min + dist_max) / 2.0
        else:
            self.ref_dist = ref_dist

        distance_grid = jnp.linspace(dist_min, dist_max, n_dist_points)
        delta_d = (dist_max - dist_min) / (n_dist_points - 1)
        self.scaling = self.ref_dist / distance_grid

        log_prob_fn = jax.vmap(lambda d: distance_prior.log_prob({"d_L": d}))
        log_w = log_prob_fn(distance_grid) + jnp.log(delta_d)
        self.log_weights = log_w - logsumexp(log_w)

    def _reduce_distance(
        self, match_filter_snr: FloatScalar, optimal_snr: FloatScalar
    ) -> FloatScalar:
        """Distance marginalization using scaling + logsumexp."""
        log_integrand = (
            match_filter_snr * self.scaling
            - 0.5 * optimal_snr * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)

    # --- combined marginalization helpers ---

    def _reduce_phase_time(
        self, complex_d_inner_h: Float[Array, " n_freq"]
    ) -> FloatScalar:
        """FFT-based time + phase marginalization (Bessel-weighted FFT)."""
        if self.tc_upsample > 1:
            window = jnp.absolute(self._windowed_fft(complex_d_inner_h))
            return logsumexp(
                jnp.where(self._tc_fine_mask, log_i0(window), -jnp.inf)
            ) - jnp.log(len(self.tc_array) * self.tc_upsample)

        complex_d_inner_h_positive_f = jnp.concatenate(
            (self.pad_low, complex_d_inner_h, self.pad_high)
        )
        fft_d_inner_h = jnp.fft.fft(complex_d_inner_h_positive_f, norm="backward")
        window = jnp.absolute(fft_d_inner_h[self._tc_window_indices])
        return logsumexp(log_i0(window)) - jnp.log(len(self.tc_array))

    def _reduce_phase_distance(
        self,
        complex_d_inner_h: complex | ComplexScalar,
        optimal_snr: FloatScalar,
    ) -> FloatScalar:
        """Phase + distance marginalization (Thrane & Talbot 2019, Eq. 79)."""
        abs_kappa = jnp.absolute(complex_d_inner_h)
        log_integrand = (
            log_i0(abs_kappa * self.scaling)
            - 0.5 * optimal_snr * self.scaling**2
            + self.log_weights
        )
        return logsumexp(log_integrand)


# ---------------------------------------------------------------------------
# Heterodyned (relative-binning) likelihood
# ---------------------------------------------------------------------------
class HeterodynedTransientLikelihoodFD(SingleEventLikelihood):
    """Frequency-domain likelihood using the relative-binning (heterodyne) scheme.

    Optionally marginalizes over coalescence time by a direct sum on the
    relative-bin endpoints.  This keeps the production evaluation independent
    of the full data duration and does not use the dense FFT/ZoomFFT reduction
    implemented by :class:`TransientLikelihoodFD`.  Coalescence phase can be
    marginalized analytically at each time sample.  Luminosity distance remains
    sampled.

    Args:
        detectors: List of detector objects containing data and metadata.
        waveform: Waveform model to evaluate.
        fixed_parameters: Dictionary of fixed parameter values.  Each value
            may be a constant ``Float``, a callable returning a scalar, **or**
            a callable returning a ``dict`` (e.g. ``transform.backward``).
            See [`TransientLikelihoodFD`][jimgw.core.single_event.likelihood.TransientLikelihoodFD]
            for a detailed description and example.
        f_min: Minimum frequency for likelihood evaluation.
        f_max: Maximum frequency for likelihood evaluation.
        trigger_time: GPS time of the event trigger.
        n_bins: Number of frequency bins for relative binning.  Mutually
            exclusive with ``epsilon``; raises ``ValueError`` if both are set.
            When neither is set, ``epsilon=0.5`` is used as the default.
        epsilon: Maximum allowed phase change per bin (rad).  The bin count
            is set to ``max(1, int(total_phase / epsilon))``.  Mutually
            exclusive with ``n_bins``; raises ``ValueError`` if both are set.
            When neither is set, ``epsilon=0.5`` is used as the default.
        optimizer_popsize: Population size for the CMA-ES optimizer used
            when finding reference parameters automatically.  Defaults to 500.
        optimizer_n_steps: Maximum number of CMA-ES generations.  Defaults to 1000.
        optimizer_target: Optional log-likelihood value at which the CMA-ES
            search stops early, before ``optimizer_n_steps`` is reached.
            Defaults to None (always run the full ``optimizer_n_steps``).
        reference_parameters: Pre-computed reference parameters (dict).  If
            supplied, the optimizer is skipped entirely.
        reference_waveform: Optional waveform instance used to compute the
            reference waveform.  Defaults to ``waveform`` when not provided.
        prior: Prior distribution from which the initial CMA-ES mean is
            drawn.  Required when ``reference_parameters`` is not provided.
        likelihood_transforms: Transforms mapping sampling parameters to
            likelihood parameters (e.g. mass-ratio → symmetric mass-ratio).
        phase_marginalization: If provided, marginalize over coalescence phase
            ``phase_c``.  Pass a [`PhaseMargConfig`][jimgw.core.single_event.likelihood.PhaseMargConfig]
            object, a plain dict ``{}``, or ``True`` (shorthand for ``PhaseMargConfig()``).
            ``None`` or ``False`` (default) disables phase marginalization.
        time_marginalization: If provided, marginalize over coalescence time
            ``t_c`` on a uniform direct-sum grid. Pass a
            [`TimeMargConfig`][jimgw.core.single_event.likelihood.TimeMargConfig],
            a plain dictionary, or ``True``. The grid uses the same strict
            interval bounds and normalization as the dense likelihood.
        reference_chunk_size: Maximum number of dense frequency samples used
            at once while constructing the fixed reference summaries. This
            bounds host and device scratch memory independently of the number
            of relative bins.
        summary_backend: ``numpy`` retains the historical host reducer.
            ``jax`` constructs native polynomial phasor summaries on device,
            distributing independent detectors over available local devices.
        xg_evaluation_mode: ``auto`` (default) selects factored proposal and
            scalar-summary evaluation for supported anchored-carrier XG models,
            keeping the original native reference and moments. ``baseline``
            retains the supplied waveform and generic evaluation arithmetic.
            The resolved implementation is recorded in ``evaluation_diagnostics``.
        node_frequency_prefix: Optional two positive increasing frequencies
            retaining a prior node grid's waveform cutoff convention when a
            coarsened grid is reconstructed. This affects node reference and
            proposal evaluation; native summary construction retains its own
            original native-frequency prefix.
        native_probe_parameters: Optional bank of one to nine likelihood-space
            parameter dictionaries. With the JAX summary backend and supported
            stock waveform, accumulate independent native likelihoods from the
            same transferred chunks. Results are retained in
            ``native_probe_results``; no extra native traversal is performed.
            This adds bounded probe waveform workspace and compilation inside
            summary construction. The default ``None`` adds no probe reducer.
        xg_plan: Internal capability created either after complete production
            receipt verification or by the isolated deterministic qualification
            builder. The latter does not authorize normal production use.
    """

    n_bins: int
    epsilon: float
    reference_parameters: dict
    freq_grid_low: Float[Array, " n_valid"]
    freq_grid_high: Float[Array, " n_valid"]
    bin_widths: Float[Array, " n_valid"]
    waveform_low_ref: dict[str, Complex[Array, " n_valid"]]
    waveform_high_ref: dict[str, Complex[Array, " n_valid"]]
    summary_data: dict[str, Complex[Array, "4 n_valid"]]

    # Degree of the per-bin ratio polynomial.  A class-level default keeps
    # benchmark subclasses that bypass ``__init__`` on the classic linear path.
    interpolation_order: int = 1
    phasor_moment_order: int = 0

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[FixedParameters] = None,
        f_min: float | dict[str, float] = 0.0,
        f_max: float | dict[str, float] = jnp.inf,
        trigger_time: float = 0,
        n_bins: Optional[int] = None,
        epsilon: Optional[float] = None,
        optimizer_popsize: int = 500,
        optimizer_n_steps: int = 1000,
        optimizer_target: Optional[float] = None,
        reference_parameters: Optional[dict] = None,
        reference_waveform: Optional[Waveform] = None,
        prior: Optional[Prior] = None,
        likelihood_transforms: Optional[list[NtoMTransform]] = None,
        phase_marginalization: Optional[Union[PhaseMargConfig, dict, bool]] = None,
        time_marginalization: Optional[
            Union[HeterodyneTimeMargConfig, TimeMargConfig, dict, bool]
        ] = None,
        reference_chunk_size: int = 262_144,
        xg_plan: Optional[Union[_VerifiedXGPlan, _QualificationXGPlan]] = None,
        interpolation_order: int = 1,
        phasor_moment_order: int = 0,
        phasor_time_anchors: Optional[Sequence[float]] = None,
        phasor_approximation: str = "taylor",
        zero_noise_summary=None,
        reference_projection: str = "projected",
        frequency_bin_edges: Optional[Sequence[float]] = None,
        summary_backend: str = "numpy",
        xg_evaluation_mode: str = "auto",
        node_frequency_prefix: Optional[Sequence[float]] = None,
        native_probe_parameters: Optional[Sequence[dict]] = None,
    ):
        construction_started = time.perf_counter()
        self._native_probe_bank = None
        self.native_probe_parameters = None
        self.native_probe_parameters_sha256 = None
        self.native_probe_results = None
        if xg_evaluation_mode not in {"auto", "baseline"}:
            raise ValueError("xg_evaluation_mode must be auto or baseline")
        self.node_frequency_prefix = None
        if node_frequency_prefix is not None:
            try:
                prefix = np.asarray(node_frequency_prefix, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "node_frequency_prefix must contain two finite positive increasing frequencies"
                ) from error
            if (
                prefix.shape != (2,)
                or not np.all(np.isfinite(prefix))
                or not np.all(prefix > 0)
                or not prefix[0] < prefix[1]
            ):
                raise ValueError(
                    "node_frequency_prefix must contain two finite positive increasing frequencies"
                )
            self.node_frequency_prefix = tuple(map(float, prefix))
            self._xg_node_frequency_prefix = jnp.asarray(prefix)
        super().__init__(detectors, waveform, fixed_parameters)

        # --- coerce marginalization inputs ---
        if isinstance(phase_marginalization, dict):
            phase_marginalization = PhaseMargConfig(**phase_marginalization)
        elif phase_marginalization is True:
            phase_marginalization = PhaseMargConfig()
        elif not phase_marginalization:
            phase_marginalization = None
        self.phase_marginalization = phase_marginalization is not None

        if isinstance(time_marginalization, dict):
            time_marginalization = HeterodyneTimeMargConfig(**time_marginalization)
        elif time_marginalization is True:
            time_marginalization = HeterodyneTimeMargConfig()
        elif not time_marginalization:
            time_marginalization = None
        elif not isinstance(time_marginalization, HeterodyneTimeMargConfig):
            time_marginalization = HeterodyneTimeMargConfig(
                tc_range=time_marginalization.tc_range,
                upsample_factor=time_marginalization.upsample_factor,
            )
        self.time_marginalization = time_marginalization is not None

        if (
            isinstance(reference_chunk_size, bool)
            or not isinstance(reference_chunk_size, (int, np.integer))
            or reference_chunk_size <= 0
        ):
            raise ValueError("reference_chunk_size must be a positive integer")
        self.reference_chunk_size = int(reference_chunk_size)
        self.coefficient_builder = "numpy-segmented-v1"
        if (
            isinstance(interpolation_order, bool)
            or not isinstance(interpolation_order, (int, np.integer))
            or not 1 <= int(interpolation_order) <= _MAX_INTERPOLATION_ORDER
        ):
            raise ValueError(
                "interpolation_order must be an integer between 1 and "
                f"{_MAX_INTERPOLATION_ORDER}, got {interpolation_order!r}"
            )
        self.interpolation_order = int(interpolation_order)
        if self.node_frequency_prefix is not None and self.interpolation_order < 2:
            raise ValueError("node_frequency_prefix requires polynomial interpolation")
        if self.interpolation_order > 1 and time_marginalization is not None:
            raise ValueError(
                "interpolation_order above 1 does not support time "
                "marginalization: the direct-sum time grid assumes linear "
                "bin-edge coefficients"
            )
        if (
            isinstance(phasor_moment_order, bool)
            or not isinstance(phasor_moment_order, (int, np.integer))
            or not 0 <= int(phasor_moment_order) <= _MAX_PHASOR_MOMENT_ORDER
        ):
            raise ValueError(
                "phasor_moment_order must be an integer between 0 and "
                f"{_MAX_PHASOR_MOMENT_ORDER}, got {phasor_moment_order!r}"
            )
        self.phasor_moment_order = int(phasor_moment_order)
        if phasor_approximation not in {"taylor", "chebyshev"}:
            raise ValueError("phasor_approximation must be taylor or chebyshev")
        self.phasor_approximation = phasor_approximation
        if reference_projection not in {"projected", "carrier"}:
            raise ValueError("reference_projection must be projected or carrier")
        self.reference_projection = reference_projection
        if reference_projection == "carrier":
            from jimgw.core.single_event.dominant_mode import (
                DominantModeTimeCachedWaveform,
            )

            if self.interpolation_order < 2 or self.phasor_moment_order == 0:
                raise ValueError("carrier reference requires polynomial phasor moments")
            if not isinstance(waveform, DominantModeTimeCachedWaveform):
                DominantModeTimeCachedWaveform(waveform)
        self.zero_noise_summary = zero_noise_summary
        if summary_backend not in {"numpy", "jax"}:
            raise ValueError("summary_backend must be numpy or jax")
        if summary_backend == "jax" and (
            self.phasor_moment_order == 0 or zero_noise_summary is not None
        ):
            raise ValueError(
                "JAX summary construction requires native polynomial phasor moments"
            )
        if summary_backend == "jax" and not jax.config.jax_enable_x64:
            raise ValueError("JAX summary construction requires 64-bit precision")
        self.summary_backend = summary_backend
        if native_probe_parameters is not None and summary_backend != "jax":
            raise ValueError(
                "native probes require the existing JAX native summary stream"
            )
        self.summary_construction_diagnostics = {}
        if zero_noise_summary is not None and self.phasor_moment_order == 0:
            raise ValueError("zero_noise_summary requires polynomial phasor moments")
        if self.phasor_moment_order > 0:
            self.coefficient_builder = (
                zero_noise_summary.method
                if zero_noise_summary is not None
                else "jax-native-phasor-v1"
                if self.summary_backend == "jax"
                else "numpy-bincount-phasor-v2"
            )
        self.phasor_time_anchors = validate_time_anchors(phasor_time_anchors)
        if self.phasor_approximation == "chebyshev" and (
            self.phasor_moment_order != 16
            or self.phasor_time_anchors is None
            or len(self.phasor_time_anchors) < 2
            or zero_noise_summary is not None
        ):
            raise ValueError(
                "chebyshev phasor approximation requires native degree-16 moments "
                "and at least two time anchors"
            )
        self.phasor_data_moments: dict[str, Array] = {}
        if self.phasor_time_anchors is not None and self.phasor_moment_order == 0:
            raise ValueError("phasor_time_anchors require nonzero phasor_moment_order")
        if self.phasor_moment_order > 0 and self.interpolation_order == 1:
            raise ValueError(
                "phasor_moment_order requires interpolation_order above 1: the "
                "analytic t_c phasor dresses the polynomial-ratio moments"
            )

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(self.trigger_time)
        xg_response = bool(
            getattr(self.waveform, "time_dependent_response", False)
            or getattr(self.waveform, "response_is_time_dependent", False)
            or any(
                getattr(detector, "time_dependent_response", False)
                or getattr(detector, "response_is_time_dependent", False)
                or getattr(detector, "finite_arm_response", False)
                for detector in self.detectors
            )
        )
        production_plan = isinstance(xg_plan, _VerifiedXGPlan) and (
            xg_plan._authority is _XG_PLAN_AUTHORITY
        )
        qualification_plan = isinstance(xg_plan, _QualificationXGPlan) and (
            xg_plan._authority is _XG_QUALIFICATION_PLAN_AUTHORITY
        )
        if xg_plan is not None and not (production_plan or qualification_plan):
            raise TypeError("xg_plan must be an internally authorized XG plan")
        if xg_response and not reference_parameters:
            raise ValueError(
                "XG heterodyne likelihood requires fixed reference_parameters; "
                "iterative reference optimization is disabled"
            )
        if xg_response and n_bins is None:
            raise ValueError(
                "XG heterodyne likelihood requires an explicit, prequalified n_bins"
            )
        if xg_response and xg_plan is None:
            raise ValueError(
                "XG heterodyne likelihood requires a verified qualification "
                "manifest before construction"
            )
        if not xg_response and xg_plan is not None:
            raise ValueError("xg_plan is only valid for an XG detector response")
        if time_marginalization is not None:
            self._validate_direct_sum_time_resolution(time_marginalization)

        # --- phase marginalization flag ---
        if self.phase_marginalization and "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )
        if self.time_marginalization and "t_c" in self.fixed_parameters:
            raise ValueError("Cannot have t_c fixed while marginalizing over t_c")

        if n_bins is not None:
            if epsilon is not None:
                raise ValueError(
                    "'n_bins' and 'epsilon' are mutually exclusive; specify at most one."
                )
            if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
                raise ValueError(
                    f"'n_bins' must be a positive integer, got {n_bins!r}."
                )
            n_bins = int(n_bins)
            if n_bins <= 0:
                raise ValueError(
                    f"'n_bins' must be a positive integer, got {n_bins!r}."
                )
            if n_bins + 1 > _MAX_DIRECT_SUM_PHASOR_ELEMENTS:
                raise ValueError(
                    f"heterodyne n_bins={n_bins} exceeds the bounded bin limit "
                    f"{_MAX_DIRECT_SUM_PHASOR_ELEMENTS - 1}"
                )
        elif epsilon is None:
            epsilon = 0.5

        if epsilon is not None:
            if isinstance(epsilon, bool) or not np.isfinite(float(epsilon)):
                raise ValueError(
                    f"'epsilon' must be a positive number and finite, got {epsilon!r}."
                )
            epsilon = float(epsilon)
            if epsilon <= 0:
                raise ValueError(
                    f"'epsilon' must be a positive number and finite, got {epsilon!r}."
                )

        if likelihood_transforms is None:
            likelihood_transforms = []

        if reference_waveform is None:
            reference_waveform = waveform

        prepared_reference_parameters = None
        if reference_parameters:
            prepared_reference_parameters = reference_parameters.copy()
            apply_fixed_parameters(
                prepared_reference_parameters,
                self.fixed_parameters,
            )
            prepared_reference_parameters["trigger_time"] = self.trigger_time
            prepared_reference_parameters["gmst"] = self.gmst
            if self.phase_marginalization:
                prepared_reference_parameters.setdefault("phase_c", 0.0)
            if self.time_marginalization:
                prepared_reference_parameters.setdefault("t_c", 0.0)
            required_reference_parameters = set(self.waveform.parameter_names) | {
                "ra",
                "dec",
                "psi",
                "t_c",
            }
            missing_reference_parameters = required_reference_parameters - set(
                prepared_reference_parameters
            )
            if missing_reference_parameters:
                raise ValueError(
                    "heterodyne reference_parameters are incomplete; missing "
                    f"{sorted(missing_reference_parameters)}"
                )
        elif prior is None:
            raise ValueError(
                "Either reference parameters or parameter names must be provided"
            )

        # --- frequency setup (same as TransientLikelihoodFD) ---
        (
            self.frequencies,
            self.identical_frequency_grids,
            self.df,
        ) = _set_and_merge_heterodyne_frequency_grids(
            self.detectors,
            f_min,
            f_max,
        )

        if native_probe_parameters is not None:
            from jimgw.core.single_event.native_probe import NativeProbeBank

            self._native_probe_bank = NativeProbeBank(
                self, waveform, native_probe_parameters
            )
            self.native_probe_parameters = self._native_probe_bank.parameters
            self.native_probe_parameters_sha256 = self._native_probe_bank.parameter_hash

        # --- heterodyne setup ---
        logger.info("Initializing heterodyned likelihood..")

        if prepared_reference_parameters is not None:
            self.reference_parameters = prepared_reference_parameters
            logger.info(
                f"Found reference parameters, they are {self.reference_parameters}"
            )
        elif prior is not None:
            logger.info("No reference parameters are provided, finding it...")
            reference_parameters = self.maximize_likelihood(
                prior=prior,
                likelihood_transforms=likelihood_transforms,
                optimizer_popsize=optimizer_popsize,
                optimizer_n_steps=optimizer_n_steps,
                optimizer_target=optimizer_target,
            )
            self.reference_parameters = {
                key: float(value) for key, value in reference_parameters.items()
            }
            logger.info(f"The reference parameters are {self.reference_parameters}")
        logger.info("Constructing reference waveforms..")

        self.reference_parameters["trigger_time"] = self.trigger_time
        self.reference_parameters["gmst"] = self.gmst
        if self.phase_marginalization:
            self.reference_parameters.setdefault("phase_c", 0.0)
        if self.time_marginalization:
            self.reference_parameters.setdefault("t_c", 0.0)
        required_reference_parameters = set(self.waveform.parameter_names) | {
            "ra",
            "dec",
            "psi",
            "t_c",
        }
        missing_reference_parameters = required_reference_parameters - set(
            self.reference_parameters
        )
        if missing_reference_parameters:
            raise ValueError(
                "heterodyne reference_parameters are incomplete; missing "
                f"{sorted(missing_reference_parameters)}"
            )

        self.waveform_low_ref = {}
        self.waveform_high_ref = {}
        self.summary_data = {}

        if epsilon is not None:
            phase = HeterodynedTransientLikelihoodFD._max_phase_diff(
                jnp.asarray((self.frequencies[0], self.frequencies[-1])),
                self.frequencies[0],
                self.frequencies[-1],
            )
            n_bins = max(1, int(float(phase[-1]) / epsilon))
        assert isinstance(n_bins, int)
        if n_bins + 1 > _MAX_DIRECT_SUM_PHASOR_ELEMENTS:
            raise ValueError(
                f"heterodyne n_bins={n_bins} exceeds the bounded bin limit "
                f"{_MAX_DIRECT_SUM_PHASOR_ELEMENTS - 1}"
            )
        planned_edges, hpc_low, hpc_high = self._plan_fixed_reference_bin_edges(
            self.frequencies,
            n_bins,
            reference_waveform,
            self.reference_parameters,
            self.reference_chunk_size,
            **(
                {"frequency_bin_edges": frequency_bin_edges}
                if frequency_bin_edges is not None
                else {}
            ),
        )
        self._set_frequency_arrays(planned_edges)
        if self.phasor_time_anchors is not None and (
            len(self.phasor_time_anchors)
            * self.n_bins
            * (self.interpolation_order + self.phasor_moment_order + 1)
            > _MAX_DIRECT_SUM_PHASOR_ELEMENTS
        ):
            raise ValueError("phasor_time_anchors exceed the bounded moment-bank size")
        self.bin_edges_sha256 = self._bin_edges_sha256(
            self.freq_grid_edges,
            interpolation_order=self.interpolation_order,
            phasor_moment_order=self.phasor_moment_order,
            phasor_time_anchors=self.phasor_time_anchors,
            phasor_approximation=self.phasor_approximation,
            summary_builder_sha256=(
                zero_noise_summary.contract_sha256
                if zero_noise_summary is not None
                else None
            ),
            reference_projection=self.reference_projection,
        )
        if xg_plan is not None and self.bin_edges_sha256 != xg_plan.bin_edges_sha256:
            raise ValueError(
                "XG qualification bin edges do not match the retained reference support"
            )
        masked_freq_grid = self.freq_grid_edges

        for detector in self.detectors:
            waveform_low_ref = self._project_reference(
                detector, self.freq_grid_low, hpc_low
            )
            waveform_high_ref = self._project_reference(
                detector, self.freq_grid_high, hpc_high
            )
            self._validate_reference_projection(
                detector.name,
                hpc_low,
                waveform_low_ref,
                "low",
            )
            self._validate_reference_projection(
                detector.name,
                hpc_high,
                waveform_high_ref,
                "high",
            )
            self.waveform_low_ref[detector.name] = waveform_low_ref
            self.waveform_high_ref[detector.name] = waveform_high_ref

        self.waveform_node_ref: dict[str, Complex[Array, "n_node n_bins"]] = {}
        if self.interpolation_order > 1:
            node_frequencies = self.freq_grid_node_flat
            if self.node_frequency_prefix is not None:
                node_frequencies = jnp.concatenate(
                    (self._xg_node_frequency_prefix, node_frequencies)
                )
            hpc_nodes = reference_waveform(node_frequencies, self.reference_parameters)
            if self.node_frequency_prefix is not None:
                hpc_nodes = jax.tree.map(lambda value: value[2:], hpc_nodes)
            for detector in self.detectors:
                node_ref = self._project_reference(
                    detector, self.freq_grid_node_flat, hpc_nodes
                )
                self._validate_reference_projection(
                    detector.name,
                    hpc_nodes,
                    node_ref,
                    "node",
                )
                self.waveform_node_ref[detector.name] = jnp.reshape(
                    node_ref, (self.interpolation_order + 1, self.n_bins)
                )

        if time_marginalization is not None:
            self._init_direct_sum_time_marginalization(time_marginalization)

        # With phasor_moment_order M > 0 the data moments are needed up to
        # order K + M (the Taylor dressing of the t_c phasor mixes A_{k+m}
        # into A_k), while the reference-norm moments stay at order 2K.
        moment_order = self.interpolation_order + self.phasor_moment_order
        self.summary_moments: dict[str, tuple[Array, Array]] = {}
        device_summaries = None
        if self.summary_backend == "jax":
            devices = jax.local_devices()
            workers = min(len(self.detectors), len(devices))

            # Each worker owns one detector's independent reductions. Only
            # the small final summaries return to the likelihood's device.
            def construct(item):
                index, detector = item
                return self._compute_reference_coefficients_jax(
                    detector,
                    reference_waveform,
                    masked_freq_grid,
                    interpolation_order=moment_order,
                    norm_order=2 * self.interpolation_order,
                    device=devices[index % workers],
                )

            if workers > 1:

                def construct_on_device(items):
                    return [
                        (index, construct((index, detector)))
                        for index, detector in items
                    ]

                assignments = [
                    list(enumerate(self.detectors))[worker::workers]
                    for worker in range(workers)
                ]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    completed = [
                        item
                        for group in pool.map(construct_on_device, assignments)
                        for item in group
                    ]
                device_summaries = [value for _, value in sorted(completed)]
            else:
                device_summaries = [
                    construct(item) for item in enumerate(self.detectors)
                ]
        for detector_index, detector in enumerate(self.detectors):
            if zero_noise_summary is not None:
                moments = zero_noise_summary.build(self, detector, reference_waveform)
                a, b = jnp.asarray(moments.data), jnp.asarray(moments.norm)
                self.summary_moments[detector.name] = (a[0], b)
                self.summary_data[detector.name] = jnp.concatenate((a[0], b))
                if self.phasor_time_anchors is not None:
                    self.phasor_data_moments[detector.name] = a
                self.summary_construction_diagnostics[detector.name] = {
                    "evaluations": moments.evaluations,
                    "panels": moments.accepted_panels,
                    "estimated_moment_error": moments.estimated_error,
                }
                continue
            moment_kwargs = (
                {"norm_order": 2 * self.interpolation_order}
                if self.phasor_moment_order > 0
                else {}
            )
            if device_summaries is None:
                summary = self._compute_reference_coefficients(
                    detector,
                    reference_waveform,
                    masked_freq_grid,
                    interpolation_order=moment_order,
                    **moment_kwargs,
                )
            else:
                summary, bank, diagnostics = device_summaries[detector_index]
                # Host-resident grids have no device; summaries then live on
                # the default device like every other likelihood constant.
                destination = (
                    next(iter(self.frequencies.devices()))
                    if hasattr(self.frequencies, "devices")
                    else jax.devices()[0]
                )
                summary = jax.device_put(summary, destination)
                if self.phasor_time_anchors is not None:
                    self.phasor_data_moments[detector.name] = jax.device_put(
                        bank, destination
                    )
                self.summary_construction_diagnostics[detector.name] = diagnostics
            self.summary_data[detector.name] = summary
            if self.interpolation_order > 1:
                split = moment_order + 1
                self.summary_moments[detector.name] = (
                    summary[:split],
                    summary[split : split + 2 * self.interpolation_order + 1],
                )
        if self.phasor_moment_order > 0:
            if "t_c" not in self.reference_parameters:
                raise ValueError(
                    "phasor_moment_order requires t_c in the reference parameters"
                )
            self._phasor_reference_t_c = jnp.asarray(
                self.reference_parameters["t_c"], dtype=jnp.float64
            )
            self._phasor_reference_delay = {}
            reference = self.reference_parameters
            for detector in self.detectors:
                if all(key in reference for key in ("ra", "dec", "gmst")):
                    self._phasor_reference_delay[detector.name] = jnp.asarray(
                        detector.delay_from_geocenter(
                            reference["ra"], reference["dec"], reference["gmst"]
                        ),
                        dtype=jnp.float64,
                    )
                else:
                    self._phasor_reference_delay[detector.name] = jnp.zeros(())

        from jimgw.core.single_event.xg_evaluation import configure_xg_evaluation

        if self._native_probe_bank is not None:
            self.native_probe_results = self._native_probe_bank.results(
                self.summary_construction_diagnostics
            )
        configure_xg_evaluation(self, reference_waveform, mode=xg_evaluation_mode)
        jax.block_until_ready((self.summary_data, self.phasor_data_moments))
        self.construction_diagnostics = {
            "summary_backend": self.summary_backend,
            "evaluation": dict(self.evaluation_diagnostics),
            "wall_seconds_including_compilation": time.perf_counter()
            - construction_started,
            "detectors": dict(self.summary_construction_diagnostics),
        }

    # --- direct evaluation ---

    def build_extrinsic_summary(self, params, waveform_cache=None):
        """Build a conditional summary using the selected evaluation algebra."""
        from jimgw.core.single_event.heterodyne_extrinsics import (
            build_extrinsic_summary,
        )

        if getattr(self, "_xg_fast_evaluator", None) is not None:
            return self._xg_fast_evaluator.build_extrinsic_summary(
                params, waveform_cache
            )
        return build_extrinsic_summary(self, params, waveform_cache)

    def _project_reference(self, detector, frequencies, polarizations):
        options = (
            {"apply_antenna": False}
            if getattr(self, "reference_projection", "projected") == "carrier"
            else {}
        )
        return detector.fd_response(
            frequencies, polarizations, self.reference_parameters, **options
        )

    def _evaluate(self, params: dict[str, Float]) -> FloatScalar:
        if self.interpolation_order > 1:
            waveform_sky_nodes = self.waveform(self.freq_grid_node_flat, params)
            return self._polynomial_likelihood(params, waveform_sky_nodes)
        waveform_sky_low = self.waveform(self.freq_grid_low, params)
        waveform_sky_high = self.waveform(self.freq_grid_high, params)
        return self._likelihood(params, waveform_sky_low, waveform_sky_high)

    # --- waveform-cache evaluation ---

    def _generate_waveform(
        self, params: dict[str, Float]
    ) -> dict[str, dict[str, Complex[Array, " n_bins"]]]:
        """Generate bin-edge polarizations for cache reuse.

        Evaluated at unit distance when ``waveform_caches_distance`` is True,
        so ``d_L`` is not a cache dependency; otherwise ``d_L`` is a
        dependency like any other waveform parameter.
        """
        if self.interpolation_order > 1:
            return {
                "nodes": self._waveform_sky_for_cache(self.freq_grid_node_flat, params)
            }
        return {
            "low": self._waveform_sky_for_cache(self.freq_grid_low, params),
            "high": self._waveform_sky_for_cache(self.freq_grid_high, params),
        }

    def _evaluate_from_waveform(
        self,
        params: dict[str, Float],
        waveform_cache: dict[str, dict[str, Complex[Array, " n_bins"]]],
    ) -> FloatScalar:
        """Core likelihood evaluation from a pre-generated waveform cache."""
        if self.interpolation_order > 1:
            waveform_sky_nodes = self._waveform_sky_from_cache(
                self.freq_grid_node_flat, waveform_cache["nodes"], params
            )
            return self._polynomial_likelihood(params, waveform_sky_nodes)
        waveform_sky_low = self._waveform_sky_from_cache(
            self.freq_grid_low, waveform_cache["low"], params
        )
        waveform_sky_high = self._waveform_sky_from_cache(
            self.freq_grid_high, waveform_cache["high"], params
        )
        return self._likelihood(params, waveform_sky_low, waveform_sky_high)

    def _rigid_time_shift(self, detector: Detector, params: dict[str, Float]) -> Float:
        """Rigid arrival-time shift of ``params`` relative to the reference.

        ``t_c`` plus the geocentre delay of the detector at the trigger epoch
        for the parameter sky position, minus the same for the reference.
        The slow within-signal variation of the delay (rotation, orbit,
        emission-time GMST) stays in the node ratio.
        """
        t_c = jnp.asarray(params["t_c"], dtype=jnp.float64)
        time_shift = t_c - self._phasor_reference_t_c
        if "ra" in params and "dec" in params and "gmst" in params:
            delay = detector.delay_from_geocenter(
                params["ra"], params["dec"], params["gmst"]
            )
            reference_delay = self._phasor_reference_delay[detector.name]
            time_shift = time_shift + (delay - reference_delay)
        return time_shift

    # --- degree-K polynomial ratio core ---

    def _polynomial_likelihood(
        self,
        params: dict[str, Float],
        waveform_sky_nodes: dict[str, Complex[Array, " n_node_total"]],
    ) -> FloatScalar:
        """Contract a degree-K per-bin ratio polynomial with summary moments.

        Inside bin ``b`` the waveform ratio is modelled as
        ``r(u) = sum_k c_k u**k`` with ``u = (f - f_c) / half_width`` and the
        coefficients solved from the K+1 Lobatto nodes.  With the moments
        ``A_k = sum d h_ref* u**k / S`` and ``B_k = sum |h_ref|^2 u**k / S``
        the inner products are ``<d|h> = sum_k conj(c_k) A_k`` and
        ``<h|h> = sum_{k,l} c_k conj(c_l) B_{k+l}``.  Order 1 reproduces the
        classic r0/r1 relative-binning formulas; higher orders remove the
        second-order phasor curvature error that scales as (bins)^-2.
        """

        if getattr(self, "_xg_fast_evaluator", None) is not None:
            return self._xg_fast_evaluator.evaluate(params, waveform_sky_nodes)

        order = self.interpolation_order
        shape = (order + 1, self.n_bins)
        log_likelihood: FloatScalar = jnp.zeros(())
        complex_d_inner_h: ComplexScalar = jnp.zeros((), dtype=jnp.complex128)

        phasor_order = self.phasor_moment_order
        phasor_approximation = getattr(self, "phasor_approximation", "taylor")

        for detector in self.detectors:
            projected = detector.fd_response(
                self.freq_grid_node_flat, waveform_sky_nodes, params
            )
            ratio = (
                jnp.reshape(projected, shape) / self.waveform_node_ref[detector.name]
            )
            moments_a, moments_b = self.summary_moments[detector.name]
            if phasor_order > 0:
                # The rigid time shift (t_c plus the sky-dependent geocentre
                # delay at the trigger) multiplies the ratio by
                # exp(-2 pi i f dt).  Remove it at the nodes, leaving the
                # smooth, slowly winding remainder for the polynomial, and
                # put it back analytically in the data moments: with
                # f = f_c + w u inside a bin,
                # exp(2 pi i f dt) = exp(2 pi i f_c dt) sum_m (i theta)^m u^m / m!
                # with theta = 2 pi w dt, so A_k(dt) = e^{2 pi i f_c dt}
                # sum_m (i theta)^m / m! A_{k+m}.  <h|h> carries no phasor.
                time_shift = self._rigid_time_shift(detector, params)
                residual_shift = time_shift
                if self.phasor_time_anchors is not None:
                    anchors = jnp.asarray(self.phasor_time_anchors)
                    anchor_index = jnp.argmin(jnp.abs(anchors - time_shift))
                    moments_a = self.phasor_data_moments[detector.name][anchor_index]
                    valid = (time_shift >= anchors[0]) & (time_shift <= anchors[-1])
                    moments_a = jnp.where(valid, moments_a, jnp.nan)
                    residual_shift = time_shift - anchors[anchor_index]
                node_angle = (2.0 * jnp.pi) * self.freq_grid_nodes * time_shift
                node_phasor = jax.lax.complex(jnp.cos(node_angle), jnp.sin(node_angle))
                centre_angle = (2.0 * jnp.pi) * self.freq_grid_centres * residual_shift
                centre_phasor = jax.lax.complex(
                    jnp.cos(centre_angle), jnp.sin(centre_angle)
                )
                i_theta = (
                    1j * (2.0 * jnp.pi) * self.freq_grid_half_widths * residual_shift
                )
                taylor_terms = [jnp.ones_like(i_theta)]
                for m in range(1, phasor_order + 1):
                    if phasor_approximation == "taylor":
                        taylor_terms.append(taylor_terms[-1] * i_theta / m)
                    else:
                        taylor_terms.append(taylor_terms[-1] * i_theta)
                if phasor_approximation == "chebyshev":
                    taylor_terms = [
                        term * self._phasor_polynomial_coefficients[m]
                        for m, term in enumerate(taylor_terms)
                    ]
                ratio = ratio * node_phasor
                moments_a = centre_phasor * jnp.stack(
                    [
                        sum(
                            taylor_terms[m] * moments_a[k + m]
                            for m in range(phasor_order + 1)
                        )
                        for k in range(order + 1)
                    ]
                )
            coefficients = self._vandermonde_inverse @ ratio
            conj_coefficients = jnp.conj(coefficients)

            d_inner_h = jnp.sum(conj_coefficients * moments_a)
            h_inner_h: FloatScalar = jnp.zeros(())
            real_coefficients, imag_coefficients = coefficients.real, coefficients.imag
            for k in range(order + 1):
                h_inner_h += jnp.sum(
                    (real_coefficients[k] ** 2 + imag_coefficients[k] ** 2)
                    * moments_b[2 * k]
                ).real
                # B[k + m] is symmetric. Pair the two conjugate products
                # before multiplication, avoiding imaginary terms that cancel.
                for m in range(k):
                    h_inner_h += (
                        2
                        * jnp.sum(
                            (
                                real_coefficients[k] * real_coefficients[m]
                                + imag_coefficients[k] * imag_coefficients[m]
                            )
                            * moments_b[k + m]
                        ).real
                    )

            if self.phase_marginalization:
                complex_d_inner_h += d_inner_h
                log_likelihood += -0.5 * h_inner_h
            else:
                log_likelihood += (d_inner_h - 0.5 * h_inner_h).real

        if self.phase_marginalization:
            log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))

        return log_likelihood

    # --- shared likelihood core ---

    def _likelihood(
        self,
        params: dict[str, Float],
        waveform_sky_low: dict[str, Complex[Array, " n_bins"]],
        waveform_sky_high: dict[str, Complex[Array, " n_bins"]],
    ) -> FloatScalar:
        """Core likelihood computation from physical-distance bin-edge polarizations."""
        if self.time_marginalization:
            return self._time_marginalized_likelihood(
                params,
                waveform_sky_low,
                waveform_sky_high,
            )

        frequencies_low = self.freq_grid_low
        frequencies_high = self.freq_grid_high
        log_likelihood: FloatScalar = jnp.zeros(())

        complex_d_inner_h: ComplexScalar = jnp.zeros((), dtype=jnp.complex128)

        for detector in self.detectors:
            waveform_low = detector.fd_response(
                frequencies_low, waveform_sky_low, params
            )
            waveform_high = detector.fd_response(
                frequencies_high, waveform_sky_high, params
            )

            r_low = waveform_low / self.waveform_low_ref[detector.name]
            r_high = waveform_high / self.waveform_high_ref[detector.name]
            r0 = (r_low + r_high) / 2
            r1 = (r_high - r_low) / self.bin_widths

            _data = self.summary_data[detector.name]
            A0, A1, B0, B1 = _data[0], _data[1], _data[2], _data[3]

            if self.phase_marginalization:
                complex_d_inner_h += jnp.sum(A0 * r0.conj() + A1 * r1.conj())
                optimal_SNR = jnp.sum(
                    B0 * jnp.abs(r0) ** 2 + 2 * B1 * (r0 * r1.conj()).real
                )
                log_likelihood += -optimal_SNR.real / 2
            else:
                match_filter_SNR = jnp.sum(A0 * r0.conj() + A1 * r1.conj())
                optimal_SNR = jnp.sum(
                    B0 * jnp.abs(r0) ** 2 + 2 * B1 * (r0 * r1.conj()).real
                )
                log_likelihood += (match_filter_SNR - optimal_SNR / 2).real

        if self.phase_marginalization:
            log_likelihood += log_i0(jnp.absolute(complex_d_inner_h))

        return log_likelihood

    def _time_marginalized_likelihood(
        self,
        params: dict[str, Float],
        waveform_sky_low: dict[str, Complex[Array, " n_bins"]],
        waveform_sky_high: dict[str, Complex[Array, " n_bins"]],
    ) -> FloatScalar:
        """Evaluate relative-bin summaries on the direct-sum time grid."""

        network_low_coeff = jnp.zeros(self.n_bins, dtype=jnp.complex128)
        network_high_coeff = jnp.zeros(self.n_bins, dtype=jnp.complex128)
        optimal_snr: FloatScalar = jnp.zeros(())
        inverse_width = 1.0 / self.bin_widths

        for detector in self.detectors:
            waveform_low = detector.fd_response(
                self.freq_grid_low,
                waveform_sky_low,
                params,
            )
            waveform_high = detector.fd_response(
                self.freq_grid_high,
                waveform_sky_high,
                params,
            )
            r_low = waveform_low / self.waveform_low_ref[detector.name]
            r_high = waveform_high / self.waveform_high_ref[detector.name]

            A0, A1, B0, B1 = self.summary_data[detector.name]
            network_low_coeff += (0.5 * A0 - A1 * inverse_width) * r_low.conj()
            network_high_coeff += (0.5 * A0 + A1 * inverse_width) * r_high.conj()

            # A time translation has unit modulus. The relative-binning h-h
            # approximation therefore needs one reduction, not one per time.
            r0 = 0.5 * (r_low + r_high)
            r1 = (r_high - r_low) * inverse_width
            optimal_snr += jnp.sum(
                B0 * jnp.abs(r0) ** 2 + 2.0 * B1 * (r0 * r1.conj()).real
            ).real

        network_match = self._direct_sum_network_match(
            network_low_coeff,
            network_high_coeff,
        )
        if self.phase_marginalization:
            time_integrand = log_i0(jnp.absolute(network_match))
        else:
            time_integrand = network_match.real
        time_integrand = jnp.where(
            self._tc_valid_time_mask,
            time_integrand,
            -jnp.inf,
        )

        return (
            -0.5 * optimal_snr
            + logsumexp(time_integrand)
            - jnp.log(self._tc_normalization_count)
        )

    def _init_direct_sum_time_marginalization(
        self,
        config: HeterodyneTimeMargConfig,
    ) -> None:
        """Construct the duration-independent direct-sum time reduction."""

        duration = float(self.detectors[0].data.duration)
        sampling_frequency = float(self.detectors[0].data.sampling_frequency)
        n_total = int(duration * sampling_frequency / 2.0)
        upsample_factor = int(config.upsample_factor)
        fine_grid_size = n_total * upsample_factor
        fine_step = duration / fine_grid_size
        if config.normalization == "window":
            # Midpoint quadrature covers the exact declared uniform-prior
            # interval. Its spacing never exceeds the requested FFT-derived
            # resolution, and equal weights sum to one without an endpoint
            # or off-grid evidence offset.
            prior_width = config.tc_range[1] - config.tc_range[0]
            n_window = max(1, int(np.ceil(prior_width / fine_step)))
            window_step = prior_width / n_window
            tc_window = config.tc_range[0] + window_step * (np.arange(n_window) + 0.5)
        else:
            q_min = -(fine_grid_size // 2)
            q_max = (fine_grid_size - 1) // 2
            first = max(
                q_min,
                int(np.floor(config.tc_range[0] / fine_step)) + 1,
            )
            last = min(
                q_max,
                int(np.ceil(config.tc_range[1] / fine_step)) - 1,
            )
            tc_window = fine_step * np.arange(first, last + 1)

        if tc_window.size == 0:
            raise ValueError(
                f"time_marginalization tc_range {config.tc_range} contains no "
                "direct-sum time samples; widen the range"
            )

        self.tc_range = config.tc_range
        self.tc_upsample = upsample_factor
        self.tc_window = jnp.asarray(tc_window)
        self._tc_normalization_count = (
            fine_grid_size if config.normalization == "full_grid" else len(tc_window)
        )
        self.tc_normalization = config.normalization
        self.freeze_time_dependent_response = config.freeze_response
        n_phasor_frequencies = self.n_bins + 1
        if n_phasor_frequencies > _MAX_DIRECT_SUM_PHASOR_ELEMENTS:
            raise ValueError(
                f"direct-sum n_bins={self.n_bins} exceeds the bounded phasor "
                f"limit {_MAX_DIRECT_SUM_PHASOR_ELEMENTS}; reduce the bin plan"
            )
        self.tc_phasor_block_size = min(
            int(config.phasor_block_size),
            len(tc_window),
        )
        phasor_elements = self.tc_phasor_block_size * n_phasor_frequencies
        if phasor_elements > _MAX_DIRECT_SUM_PHASOR_ELEMENTS:
            maximum_block_size = max(
                1,
                _MAX_DIRECT_SUM_PHASOR_ELEMENTS // n_phasor_frequencies,
            )
            raise ValueError(
                "direct-sum phasor block would contain "
                f"{phasor_elements} elements; set phasor_block_size <= "
                f"{maximum_block_size} for {self.n_bins} bins"
            )
        n_time_blocks = int(np.ceil(len(tc_window) / self.tc_phasor_block_size))
        padded_size = n_time_blocks * self.tc_phasor_block_size
        padded_times = np.zeros(padded_size)
        padded_times[: len(tc_window)] = tc_window
        valid_times = np.arange(padded_size) < len(tc_window)
        self._tc_time_blocks = jnp.asarray(padded_times).reshape(
            n_time_blocks,
            self.tc_phasor_block_size,
        )
        self._tc_valid_time_mask = jnp.asarray(valid_times)

    def _validate_direct_sum_time_resolution(
        self,
        config: HeterodyneTimeMargConfig,
    ) -> None:
        """Reject an underresolved time grid before reference construction."""

        response_is_time_dependent = bool(
            getattr(self.waveform, "time_dependent_response", False)
            or getattr(self.waveform, "response_is_time_dependent", False)
            or any(
                getattr(detector, "time_dependent_response", False)
                or getattr(detector, "response_is_time_dependent", False)
                for detector in self.detectors
            )
        )
        if response_is_time_dependent and not config.freeze_response:
            raise ValueError(
                "direct-sum time marginalization shifts only the carrier; set "
                "freeze_response=true to acknowledge that a time-dependent "
                "response and its h-h term are frozen at t_c=0"
            )
        if response_is_time_dependent and config.normalization != "window":
            raise ValueError(
                "time-dependent direct-sum marginalization requires "
                "normalization='window'"
            )
        if response_is_time_dependent and config.timing_sigma_s is None:
            raise ValueError(
                "time-dependent direct-sum marginalization requires a finite "
                "timing_sigma_s from the frozen science envelope"
            )
        if response_is_time_dependent and not (
            config.tc_range[0] < 0.0 < config.tc_range[1]
        ):
            raise ValueError("the frozen response pivot t_c=0 must lie inside tc_range")

        duration = float(self.detectors[0].data.duration)
        if config.normalization == "window":
            support_min = -0.5 * duration
            support_max = 0.5 * duration
            if config.tc_range[0] < support_min or config.tc_range[1] > support_max:
                raise ValueError(
                    "window-normalized tc_range must lie inside the centered "
                    f"data duration [{support_min}, {support_max}] s"
                )
        sampling_frequency = float(self.detectors[0].data.sampling_frequency)
        n_total = int(duration * sampling_frequency / 2.0)
        fine_grid_size = n_total * int(config.upsample_factor)
        fine_step = duration / fine_grid_size
        if config.normalization == "window":
            n_window = int(
                np.ceil((config.tc_range[1] - config.tc_range[0]) / fine_step)
            )
        else:
            q_min = -(fine_grid_size // 2)
            q_max = (fine_grid_size - 1) // 2
            first = max(q_min, int(np.floor(config.tc_range[0] / fine_step)) + 1)
            last = min(q_max, int(np.ceil(config.tc_range[1] / fine_step)) - 1)
            n_window = max(0, last - first + 1)
        if n_window > _MAX_DIRECT_SUM_TIME_SAMPLES:
            raise ValueError(
                f"direct-sum time grid has {n_window} samples, above the bounded "
                f"limit {_MAX_DIRECT_SUM_TIME_SAMPLES}; narrow tc_range or reduce "
                "upsample_factor"
            )

        if config.timing_sigma_s is None:
            return
        maximum_step = config.timing_sigma_s / config.samples_per_timing_sigma
        if fine_step > maximum_step:
            required_upsample = int(np.ceil(duration / (n_total * maximum_step)))
            raise ValueError(
                "direct-sum time spacing "
                f"{fine_step:.6g} s does not resolve timing_sigma_s="
                f"{config.timing_sigma_s:.6g} s with "
                f"{config.samples_per_timing_sigma} samples per sigma; "
                f"set upsample_factor >= {required_upsample}"
            )

    def _direct_sum_network_match(
        self,
        network_low_coeff: Complex[Array, " n_bin"],
        network_high_coeff: Complex[Array, " n_bin"],
    ) -> Complex[Array, " n_time_padded"]:
        """Generate bounded phasor blocks and evaluate the network match."""

        edge_coefficients = jnp.concatenate(
            (
                network_low_coeff[:1],
                network_high_coeff[:-1] + network_low_coeff[1:],
                network_high_coeff[-1:],
            )
        )

        def evaluate_block(times: Float[Array, " block_size"]):
            return self._time_phasors(times, self.freq_grid_edges) @ edge_coefficients

        block_matches = jax.lax.map(evaluate_block, self._tc_time_blocks)
        return block_matches.reshape(-1)

    @staticmethod
    def _time_phasors(
        times: Float[Array, " n_time"],
        frequencies: Float[Array, " n_bin"],
    ) -> Complex[Array, "n_time n_bin"]:
        """Return conjugate time phasors for ``data * h.conj()`` summaries."""

        angle = (2.0 * jnp.pi) * times[:, None] * frequencies[None, :]
        return jax.lax.complex(jnp.cos(angle), jnp.sin(angle))

    # --- relative-binning setup helpers ---

    def _trim_reference_zero_edges(
        self,
        waveform_low: Mapping[str, Array],
        waveform_high: Mapping[str, Array],
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Trim outer bins whose source waveform vanishes at either endpoint."""

        edges, waveform_low, waveform_high = self._trim_reference_edge_arrays(
            self.freq_grid_edges,
            waveform_low,
            waveform_high,
        )
        self._set_frequency_arrays(edges)
        return waveform_low, waveform_high

    @staticmethod
    def _physical_waveform_amplitude(
        waveform: Mapping[str, Array],
    ) -> np.ndarray:
        """Return the summed host amplitude of physical polarization leaves."""

        physical = [
            np.abs(np.asarray(jax.device_get(value)))
            for name, value in waveform.items()
            if not name.startswith("__")
        ]
        if not physical:
            raise ValueError("reference waveform has no physical polarizations")
        return np.sum(np.stack(physical), axis=0)

    @classmethod
    def _trim_reference_edge_arrays(
        cls,
        frequency_edges: Float[Array, " n_bin+1"],
        waveform_low: Mapping[str, Array],
        waveform_high: Mapping[str, Array],
    ) -> tuple[Float[Array, " n_valid+1"], dict[str, Array], dict[str, Array]]:
        """Return the contiguous nonzero endpoint interval without mutating state."""

        valid = (cls._physical_waveform_amplitude(waveform_low) > 0) & (
            cls._physical_waveform_amplitude(waveform_high) > 0
        )
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            raise ValueError("reference waveform has no nonzero heterodyne bins")
        first = int(valid_indices[0])
        stop = int(valid_indices[-1]) + 1
        if not np.all(valid[first:stop]):
            raise ValueError(
                "reference waveform has an internal zero that cannot be represented "
                "by one contiguous heterodyne bin plan"
            )

        return (
            frequency_edges[first : stop + 1],
            {name: value[first:stop] for name, value in waveform_low.items()},
            {name: value[first:stop] for name, value in waveform_high.items()},
        )

    @staticmethod
    def _validate_reference_projection(
        detector_name: str,
        source_polarizations: Mapping[str, Array],
        projected_reference: Array,
        edge_name: str,
    ) -> None:
        """Reject detector-reference nulls that make waveform ratios unstable."""

        physical = [
            np.abs(np.asarray(jax.device_get(value)))
            for name, value in source_polarizations.items()
            if not name.startswith("__")
        ]
        if not physical:
            raise ValueError("reference waveform has no physical polarizations")
        source_scale = np.sum(np.stack(physical), axis=0)
        projected_amplitude = np.abs(np.asarray(jax.device_get(projected_reference)))
        relative_amplitude = np.divide(
            projected_amplitude,
            source_scale,
            out=np.zeros_like(projected_amplitude, dtype=float),
            where=source_scale > 0,
        )
        invalid = (~np.isfinite(relative_amplitude)) | (
            relative_amplitude <= _MIN_RELATIVE_BIN_REFERENCE_RESPONSE
        )
        if np.any(invalid):
            minimum = float(np.min(relative_amplitude))
            raise ValueError(
                f"heterodyne reference projection for detector {detector_name!r} "
                f"has a {edge_name}-edge response null (minimum relative "
                f"amplitude {minimum:.3g}); choose a non-null fixed reference"
            )

    @staticmethod
    def _make_binning_scheme(
        freqs: Float[Array, " n_freq"],
        n_bins: int,
        chi: float = 1.0,
    ) -> Float[Array, " n_bins+1"]:
        """Make ``n_bins`` frequency bins of equal phase change.

        ``n_bins`` must be a positive integer resolved by the caller
        (see :meth:`__init__`).
        """
        f_low = float(freqs[0])
        f_high = float(freqs[-1])
        if not np.isfinite(f_low) or not np.isfinite(f_high) or f_low >= f_high:
            raise ValueError(
                "heterodyne frequency support must be finite and increasing"
            )

        gamma = np.asarray((-5.0, -2.0, 3.0, 5.0, 7.0)) / 3.0
        f_star = np.where(gamma >= 0.0, f_high, f_low)

        def phase_envelope(frequency: np.ndarray) -> np.ndarray:
            scaled = frequency[:, None] / f_star[None, :]
            raw = (2.0 * np.pi * chi) * np.sum(
                np.power(scaled, gamma[None, :]) * np.sign(gamma)[None, :],
                axis=1,
            )
            return raw

        endpoint_phase = phase_envelope(np.asarray((f_low, f_high)))
        target_phase = np.linspace(
            endpoint_phase[0],
            endpoint_phase[1],
            n_bins + 1,
        )

        # The envelope derivative is strictly positive because every term has
        # sign(gamma) * gamma > 0. Vector bisection therefore gives the exact
        # continuous bin edges with O(n_bins) memory, independent of the dense
        # data duration.
        lower = np.full(n_bins + 1, f_low)
        upper = np.full(n_bins + 1, f_high)
        for _ in range(52):
            midpoint = 0.5 * (lower + upper)
            below_target = phase_envelope(midpoint) < target_phase
            lower = np.where(below_target, midpoint, lower)
            upper = np.where(below_target, upper, midpoint)

        f_bins = 0.5 * (lower + upper)
        f_bins[0] = f_low
        f_bins[-1] = f_high
        return jnp.asarray(f_bins)

    def _mask_and_set_frequency_arrays(
        self,
        waveform: dict[str, Complex[Array, " n_freq"]],
        frequencies: Float[Array, " n_freq"],
    ) -> Float[Array, " n_valid+1"]:
        """Set bin endpoints after removing zero-valued reference support."""

        physical_polarizations = [
            polarization
            for name, polarization in waveform.items()
            if not name.startswith("__")
        ]
        if not physical_polarizations:
            raise ValueError("reference waveform has no physical polarizations")
        h_amp = jnp.array([jnp.abs(p) for p in physical_polarizations]).sum(axis=0)
        _valid_frequencies = self.frequencies[h_amp > 0]
        return self._set_frequency_arrays_from_support(
            frequencies,
            (
                float(_valid_frequencies[0]),
                float(_valid_frequencies[-1]),
            ),
        )

    def _find_reference_frequency_support(
        self,
        reference_waveform: Waveform,
    ) -> tuple[float, float]:
        """Find the outer nonzero support with bounded endpoint scans."""

        return self._find_reference_frequency_support_on_grid(
            self.frequencies,
            reference_waveform,
            self.reference_parameters,
            self.reference_chunk_size,
        )

    @classmethod
    def _find_reference_frequency_support_on_grid(
        cls,
        frequencies: Float[Array, " n_freq"],
        reference_waveform: Waveform,
        reference_parameters: Mapping[str, Any],
        reference_chunk_size: int,
    ) -> tuple[float, float]:
        """Find a fixed reference's outer support without likelihood state."""

        n_frequencies = len(frequencies)

        def valid_chunk(start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
            chunk_frequencies = frequencies[start:stop]
            polarizations = reference_waveform(
                chunk_frequencies,
                reference_parameters,
            )
            amplitude = cls._physical_waveform_amplitude(polarizations)
            return (
                np.asarray(jax.device_get(chunk_frequencies)),
                np.flatnonzero(amplitude > 0),
            )

        first_frequency: float | None = None
        for start in range(0, n_frequencies, reference_chunk_size):
            stop = min(start + reference_chunk_size, n_frequencies)
            host_frequencies, valid = valid_chunk(start, stop)
            if valid.size:
                first_frequency = float(host_frequencies[valid[0]])
                break

        if first_frequency is None:
            raise ValueError("reference waveform is zero throughout the analysis band")

        last_frequency: float | None = None
        for stop in range(n_frequencies, 0, -reference_chunk_size):
            start = max(0, stop - reference_chunk_size)
            host_frequencies, valid = valid_chunk(start, stop)
            if valid.size:
                last_frequency = float(host_frequencies[valid[-1]])
                break

        if last_frequency is None:
            raise RuntimeError("failed to recover the final reference support sample")
        return first_frequency, last_frequency

    def _set_frequency_arrays_from_support(
        self,
        frequencies: Float[Array, " n_freq"],
        support: tuple[float, float],
    ) -> Float[Array, " n_valid+1"]:
        """Set static bin endpoints for a fixed nonzero support interval."""

        masked_frequencies_array = self._frequency_edges_from_support(
            frequencies,
            support,
        )
        self._set_frequency_arrays(masked_frequencies_array)
        return masked_frequencies_array

    @staticmethod
    def _frequency_edges_from_support(
        frequencies: Float[Array, " n_freq"],
        support: tuple[float, float],
    ) -> Float[Array, " n_valid+1"]:
        """Return planned edges within one fixed nonzero support interval."""

        host_frequencies = np.asarray(jax.device_get(frequencies))
        valid = (host_frequencies >= support[0]) & (host_frequencies <= support[1])
        masked_frequencies = host_frequencies[valid]
        if masked_frequencies.size < 2:
            raise ValueError(
                "reference waveform support contains fewer than two bin edges"
            )

        return jnp.asarray(masked_frequencies)

    def _set_frequency_arrays(
        self,
        frequency_edges: Float[Array, " n_valid+1"],
    ) -> None:
        """Store one already validated static edge array on the likelihood."""

        self.freq_grid_edges = frequency_edges
        self.freq_grid_low = frequency_edges[:-1]
        self.freq_grid_high = frequency_edges[1:]
        self.n_bins = len(frequency_edges) - 1
        self.bin_widths = self.freq_grid_high - self.freq_grid_low
        self._set_interpolation_nodes()

    @staticmethod
    def _lobatto_nodes(order: int) -> np.ndarray:
        """Return the K+1 Gauss-Lobatto nodes on [-1, 1] for a degree-K fit."""

        if order == 1:
            return np.asarray([-1.0, 1.0])
        legendre = np.zeros(order + 1)
        legendre[order] = 1.0
        interior = np.polynomial.legendre.legroots(
            np.polynomial.legendre.legder(legendre)
        )
        return np.concatenate(([-1.0], np.sort(np.real(interior)), [1.0]))

    def _set_interpolation_nodes(self) -> None:
        """Place the ratio-fit nodes in every bin and cache the Vandermonde solve."""

        order = int(getattr(self, "interpolation_order", 1))
        nodes = self._lobatto_nodes(order)
        self._vandermonde_inverse = jnp.asarray(
            np.linalg.inv(np.vander(nodes, order + 1, increasing=True))
        )
        centres = 0.5 * (self.freq_grid_low + self.freq_grid_high)
        half_widths = 0.5 * self.bin_widths
        self.freq_grid_centres = centres
        self.freq_grid_half_widths = half_widths
        self.freq_grid_nodes = (
            centres[None, :] + half_widths[None, :] * jnp.asarray(nodes)[:, None]
        )
        self.freq_grid_node_flat = jnp.reshape(self.freq_grid_nodes, (-1,))
        if getattr(self, "phasor_approximation", "taylor") == "chebyshev":
            from jimgw.core.single_event.heterodyne_phasor import (
                phasor_polynomial_coefficients,
            )

            coefficients, diagnostics = phasor_polynomial_coefficients(
                np.asarray(half_widths),
                self.phasor_time_anchors,
                self.phasor_moment_order,
                approximation="chebyshev",
            )
            self._phasor_polynomial_coefficients = jnp.asarray(coefficients)
            self.phasor_approximation_diagnostics = diagnostics

    @staticmethod
    def _bin_edges_sha256(
        frequency_edges: Float[Array, " n_bin+1"],
        *,
        interpolation_order: int = 1,
        phasor_moment_order: int = 0,
        phasor_time_anchors: Optional[Sequence[float]] = None,
        summary_builder_sha256: Optional[str] = None,
        reference_projection: str = "projected",
        phasor_approximation: str = "taylor",
    ) -> str:
        """Hash canonical little-endian float64 XG bin edges.

        Orders above 1 are folded into the digest so a qualification receipt
        binds the per-bin ratio model as well as the edges; order 1 keeps the
        historical digest unchanged.  A nonzero phasor moment order is bound
        the same way, since it changes which summary moments the likelihood
        contracts.
        """

        if (
            isinstance(interpolation_order, bool)
            or not isinstance(interpolation_order, (int, np.integer))
            or int(interpolation_order) < 1
        ):
            raise ValueError("interpolation_order must be a positive integer")
        bin_edges = np.asarray(jax.device_get(frequency_edges), dtype="<f8")
        bin_digest = hashlib.sha256()
        bin_digest.update(b"jimgw-xg-bin-edges-v1\0float64-le\0")
        bin_digest.update(bin_edges.tobytes(order="C"))
        if int(interpolation_order) != 1:
            bin_digest.update(
                f"\0interpolation-order={int(interpolation_order)}".encode("ascii")
            )
        if (
            isinstance(phasor_moment_order, bool)
            or not isinstance(phasor_moment_order, (int, np.integer))
            or int(phasor_moment_order) < 0
        ):
            raise ValueError("phasor_moment_order must be a non-negative integer")
        if int(phasor_moment_order) != 0:
            bin_digest.update(
                f"\0phasor-moment-order={int(phasor_moment_order)}".encode("ascii")
            )
        if phasor_approximation not in {"taylor", "chebyshev"}:
            raise ValueError("phasor_approximation must be taylor or chebyshev")
        if phasor_approximation == "chebyshev":
            from jimgw.core.single_event.heterodyne_phasor import (
                CHEBYSHEV_PHASOR_REVISION,
            )

            if int(phasor_moment_order) != 16:
                raise ValueError("chebyshev phasor approximation requires order 16")
            bin_digest.update(b"\0phasor-approximation\0")
            bin_digest.update(CHEBYSHEV_PHASOR_REVISION.encode("ascii"))
        anchors = validate_time_anchors(phasor_time_anchors)
        if anchors is not None:
            if phasor_moment_order == 0:
                raise ValueError(
                    "phasor_time_anchors require nonzero phasor_moment_order"
                )
            bin_digest.update(b"\0phasor-time-anchors-v1\0")
            bin_digest.update(np.asarray(anchors, dtype="<f8").tobytes())
        if summary_builder_sha256 is not None:
            if len(summary_builder_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in summary_builder_sha256
            ):
                raise ValueError("summary_builder_sha256 must be a SHA-256 digest")
            bin_digest.update(b"\0summary-builder-v1\0")
            bin_digest.update(summary_builder_sha256.encode("ascii"))
        if reference_projection not in {"projected", "carrier"}:
            raise ValueError("reference_projection must be projected or carrier")
        if reference_projection != "projected":
            bin_digest.update(b"\0carrier-reference-v1\0")
        return bin_digest.hexdigest()

    @classmethod
    def _plan_fixed_reference_bin_edges(
        cls,
        frequencies: Float[Array, " n_freq"],
        n_bins: int,
        reference_waveform: Waveform,
        reference_parameters: Mapping[str, Any],
        reference_chunk_size: int,
        *,
        frequency_bin_edges: Optional[Sequence[float]] = None,
    ) -> tuple[
        Float[Array, " n_valid+1"],
        dict[str, Array],
        dict[str, Array],
    ]:
        """Plan retained edges and endpoint waveforms without summary construction."""

        if frequency_bin_edges is None:
            frequency_edges = cls._make_binning_scheme(frequencies, n_bins=n_bins)
        else:
            edges = np.asarray(frequency_bin_edges, dtype=float)
            if (
                edges.shape != (n_bins + 1,)
                or np.any(~np.isfinite(edges))
                or np.any(np.diff(edges) <= 0)
                or edges[0] != float(frequencies[0])
                or edges[-1] != float(frequencies[-1])
            ):
                raise ValueError(
                    "frequency_bin_edges must be increasing, match n_bins and span the full frequency band"
                )
            frequency_edges = jnp.asarray(edges)
        support = cls._find_reference_frequency_support_on_grid(
            frequencies,
            reference_waveform,
            reference_parameters,
            reference_chunk_size,
        )
        frequency_edges = cls._frequency_edges_from_support(frequency_edges, support)
        waveform_low = reference_waveform(frequency_edges[:-1], reference_parameters)
        waveform_high = reference_waveform(frequency_edges[1:], reference_parameters)
        return cls._trim_reference_edge_arrays(
            frequency_edges,
            waveform_low,
            waveform_high,
        )

    @staticmethod
    def _max_phase_diff(
        freqs: Float[Array, " n_freq"],
        f_low: FloatLike,
        f_high: FloatLike,
        chi: float = 1.0,
    ) -> Float[Array, " n_freq"]:
        """
        Compute the cumulative phase difference used for bin construction.

        Uses 5 physically-motivated PN/IMR terms from arXiv:1806.08792:
        gamma ∈ {-5/3, -2/3, 1, 5/3, 7/3}, covering the dominant Newtonian
        chirp (0PN), spin-orbit (1.5PN), coalescence time, and phenomenological
        IMR contributions.  Each term is normalised by so that its individual
        contribution spans exactly ``chi * 2π`` rad across [f_low, f_high].
        The returned array starts at 0 (cumulative from f_low).

        See also Eq.(7) in arXiv:2302.05333.
        """
        gamma = jnp.array([-5.0, -2.0, 3.0, 5.0, 7.0]) / 3
        freq_2D = jax.lax.broadcast_in_dim(freqs, (freqs.size, gamma.size), [0])
        f_star = jnp.where(gamma >= 0, f_high, f_low)
        summand = (freq_2D / f_star) ** gamma * jnp.sign(gamma)
        dphi = 2 * jnp.pi * chi * jnp.sum(summand, axis=1)
        return dphi - dphi[0]

    @staticmethod
    def _compute_coefficients(
        detector: Detector,
        h_ref: Complex[Array, " n_freq"],
        f_bins: Float[Array, " n_valid+1"],
        *,
        interpolation_order: int = 1,
    ) -> Complex[Array, "n_summary n_valid"]:
        """Compute summaries with one bin-index array, never a dense mask."""

        summary = HeterodynedTransientLikelihoodFD._segmented_coefficient_sums(
            data=np.asarray(jax.device_get(detector.sliced_fd_data)),
            psd=np.asarray(jax.device_get(detector.sliced_psd)),
            frequencies=np.asarray(jax.device_get(detector.sliced_frequencies)),
            reference=np.asarray(jax.device_get(h_ref)),
            bins=np.asarray(jax.device_get(f_bins)),
            interpolation_order=interpolation_order,
        )
        return jnp.asarray((4.0 / float(detector.duration)) * summary)

    def _compute_reference_coefficients(
        self,
        detector: Detector,
        reference_waveform: Waveform,
        f_bins: Float[Array, " n_valid+1"],
        *,
        interpolation_order: int = 1,
        norm_order: Optional[int] = None,
    ) -> Complex[Array, "n_summary n_valid"]:
        """Stream fixed-reference summaries in bounded frequency chunks."""

        bins = np.asarray(jax.device_get(f_bins))
        n_summary = (
            int(interpolation_order) + int(norm_order) + 2
            if norm_order is not None
            else (4 if interpolation_order == 1 else 3 * int(interpolation_order) + 2)
        )
        n_frequencies = len(detector.sliced_frequencies)
        if n_frequencies <= self.reference_chunk_size and norm_order is None:
            frequencies = detector.sliced_frequencies
            polarizations = reference_waveform(
                frequencies,
                self.reference_parameters,
            )
            reference = self._project_reference(detector, frequencies, polarizations)
            if interpolation_order == 1:
                # Keep the classic call shape so overrides with the original
                # signature (benchmarks, test doubles) keep working.
                return self._compute_coefficients(detector, reference, f_bins)
            return self._compute_coefficients(
                detector,
                reference,
                f_bins,
                interpolation_order=interpolation_order,
            )

        summary = np.zeros((n_summary, len(bins) - 1), dtype=np.complex128)
        anchors = getattr(self, "phasor_time_anchors", None)
        bank = (
            None
            if anchors is None
            else np.zeros(
                (len(anchors), interpolation_order + 1, len(bins) - 1),
                dtype=np.complex128,
            )
        )

        # Two native samples are prepended to every waveform call so backends
        # that read the grid spacing from their first two inputs (ripple) see
        # the true spacing even on a one-sample tail chunk.
        prefix = jnp.asarray(np.asarray(detector.sliced_frequencies[:2]))

        def project_chunk(frequencies):
            sky = reference_waveform(
                jnp.concatenate((prefix, jnp.asarray(frequencies))),
                self.reference_parameters,
            )
            sky = jax.tree.map(lambda value: value[2:], sky)
            return self._project_reference(detector, frequencies, sky)

        # One compiled operation per full/tail shape, with frequencies as an
        # argument. Avoid thousands of eager operations and device round trips.
        project = jax.jit(project_chunk) if norm_order is not None else project_chunk
        for start in range(0, n_frequencies, self.reference_chunk_size):
            stop = min(start + self.reference_chunk_size, n_frequencies)
            frequencies = jnp.asarray(
                np.asarray(detector.sliced_frequencies[start:stop])
            )
            reference = project(frequencies)
            host_f = np.asarray(jax.device_get(frequencies))
            host_data = np.asarray(jax.device_get(detector.sliced_fd_data[start:stop]))
            host_psd = np.asarray(jax.device_get(detector.sliced_psd[start:stop]))
            host_reference = np.asarray(jax.device_get(reference))
            summary += self._segmented_coefficient_sums(
                data=host_data,
                psd=host_psd,
                frequencies=host_f,
                reference=host_reference,
                bins=bins,
                interpolation_order=interpolation_order,
                **({"norm_order": norm_order} if norm_order is not None else {}),
            )
            if bank is not None:
                for i, anchor in enumerate(anchors):
                    a, _ = polynomial_moments(
                        host_f,
                        host_data * np.exp(2j * np.pi * host_f * anchor),
                        host_psd,
                        host_reference,
                        bins,
                        interpolation_order,
                        0,
                    )
                    bank[i] += a
        if bank is not None:
            self.phasor_data_moments[detector.name] = jnp.asarray(
                (4.0 / float(detector.duration)) * bank
            )
        return jnp.asarray((4.0 / float(detector.duration)) * summary)

    def _compute_reference_coefficients_jax(
        self,
        detector,
        reference_waveform,
        f_bins,
        *,
        interpolation_order,
        norm_order,
        device,
    ):
        """Fuse reference projection and native overlap reductions on device.

        One compiled executable per detector and reducer: every chunk,
        including the tail, is padded to one memory-budgeted chunk size, and every
        tile descriptor to one count planned over the whole stream up front.
        Chunks are moved to the device from wherever the detector keeps them
        (host views or device arrays); the stream is never copied whole. Each
        chunk is checked where stored against the native grid ``k/duration``;
        native chunks use the bin-aligned GEMM reducer with a deterministic
        combine, others the generic segmented reducer, compiled only if used.
        The first two native frequencies are prepended to waveform calls so
        spacing-reading backends see the true spacing on a padded tail.

        Optional independent native probes consume these same transferred
        chunks in one additional compiled reducer per detector. Their small
        compensated sums and validity flags join synchronization before input
        pages are released; no second native-input traversal is made.
        """
        from jimgw.core.single_event.heterodyne_summary import compiled_summary_chunk
        from jimgw.core.single_event.heterodyne_summary_tiles import (
            compiled_summary_tiles,
        )
        from jimgw.core.single_event.native_storage import release_mapped_pages

        started = time.perf_counter()
        n_frequencies = len(detector.sliced_frequencies)
        anchors_host = self.phasor_time_anchors
        n_anchors = 0 if anchors_host is None else len(anchors_host)
        chunk_size = min(n_frequencies, self.reference_chunk_size)
        n_bins = len(f_bins) - 1
        duration = float(detector.duration)
        first_native_index = max(
            0, round(float(detector.sliced_frequencies[0]) * duration)
        )
        bins_host = np.asarray(f_bins)
        # Powers, real/imaginary anchor products and one norm plane.
        planes = max(interpolation_order, norm_order) + 2 * (n_anchors + 1) + 2
        planner, chunk_size, padded_tiles, padded_local, matrix_elements = (
            _summary_tile_layout(
                bins_host,
                duration,
                first_native_index,
                n_frequencies,
                chunk_size,
                planes,
            )
        )
        tile_size = planner.tile_size
        chunk_starts = range(0, n_frequencies, chunk_size)

        with jax.default_device(device):
            bins = jax.device_put(np.asarray(f_bins, dtype=np.float64), device)
            anchors = jax.device_put(
                np.asarray(
                    [] if anchors_host is None else anchors_host, dtype=np.float64
                ),
                device,
            )
            prefix = jax.device_put(detector.sliced_frequencies[:2], device)
            total = jnp.zeros(
                (interpolation_order + norm_order + 2, n_bins), dtype=jnp.complex128
            )
            bank = jnp.zeros(
                (n_anchors, interpolation_order + 1, n_bins), dtype=jnp.complex128
            )
            probe_bank = getattr(self, "_native_probe_bank", None)
            probe_state = (
                None if probe_bank is None else probe_bank.initial_state(device)
            )
            probe_reduce = (
                None
                if probe_bank is None
                else probe_bank.make_reducer(detector, device)
            )

            def project(frequencies):
                sky = reference_waveform(
                    jnp.concatenate((prefix, frequencies)), self.reference_parameters
                )
                sky = jax.tree.map(lambda value: value[2:], sky)
                return self._project_reference(detector, frequencies, sky)

            @jax.jit
            def accumulate_tiles(
                frequencies,
                data,
                psd,
                total,
                bank,
                starts,
                lengths,
                bin_ids,
                tile_local,
                local_ids,
            ):
                a, b, anchored = compiled_summary_tiles(
                    frequencies,
                    data,
                    psd,
                    project(frequencies),
                    bins,
                    anchors,
                    starts,
                    lengths,
                    bin_ids,
                    tile_size=tile_size,
                    data_order=interpolation_order,
                    norm_order=norm_order,
                    tile_local=tile_local,
                    local_bins=local_ids,
                )
                return total + jnp.concatenate((a, b), axis=0), bank + anchored

            @jax.jit
            def accumulate_segmented(frequencies, data, psd, total, bank):
                a, b, anchored = compiled_summary_chunk(
                    frequencies,
                    data,
                    psd,
                    project(frequencies),
                    bins,
                    anchors,
                    data_order=interpolation_order,
                    norm_order=norm_order,
                )
                return total + jnp.concatenate((a, b), axis=0), bank + anchored

            tiled_chunks = 0
            pending_fd_views = []
            for chunk_index, start in enumerate(chunk_starts):
                stop = min(start + chunk_size, n_frequencies)
                count = stop - start
                plan = planner.plan(
                    first_native_index + start,
                    count,
                    padded_tiles=padded_tiles,
                    padded_local=padded_local,
                )
                frequency_slice = detector.sliced_frequencies[start:stop]
                fd_slice = detector.sliced_fd_data[start:stop]
                if isinstance(fd_slice, np.ndarray):
                    pending_fd_views.append(fd_slice)
                if isinstance(frequency_slice, np.ndarray):
                    native = _host_native_grid_is_uniform(
                        frequency_slice, first_native_index + start, duration
                    )
                frequencies, data, psd = (
                    jax.device_put(value, device)
                    for value in (
                        frequency_slice,
                        fd_slice,
                        detector.sliced_psd[start:stop],
                    )
                )
                padding = chunk_size - count
                if padding:
                    frequencies = jnp.pad(
                        frequencies, (0, padding), constant_values=bins_host[-1] + 1.0
                    )
                    data = jnp.pad(data, (0, padding))
                    psd = jnp.pad(psd, (0, padding), constant_values=1.0)
                if not isinstance(frequency_slice, np.ndarray):
                    native = bool(
                        jax.device_get(
                            _device_chunk_is_native(
                                frequencies, first_native_index + start, duration, count
                            )
                        )
                    )
                if native:
                    descriptors = (
                        jax.device_put(value, device)
                        for value in (
                            plan.starts,
                            plan.lengths,
                            plan.bin_ids,
                            plan.tile_local,
                            plan.local_ids,
                        )
                    )
                    total, bank = accumulate_tiles(
                        frequencies, data, psd, total, bank, *descriptors
                    )
                    tiled_chunks += 1
                else:
                    total, bank = accumulate_segmented(
                        frequencies, data, psd, total, bank
                    )
                if probe_reduce is not None:
                    # Reuse the chunk already transferred for native moments.
                    # The oracle has its own stock waveform/response algebra
                    # and keeps real samples even outside the retained bins.
                    probe_state = probe_reduce(
                        frequencies, data, psd, start, count, probe_state
                    )
                # Bound outstanding transfers and temporaries while keeping
                # asynchronous dispatch within each small group of chunks.
                if (chunk_index + 1) % 8 == 0:
                    jax.block_until_ready((total, bank, probe_state))
                    for view in pending_fd_views:
                        release_mapped_pages(view)
                    pending_fd_views.clear()
            total, bank, probe_state = jax.block_until_ready((total, bank, probe_state))
            for view in pending_fd_views:
                release_mapped_pages(view)
            pending_fd_views.clear()
            finite = jnp.all(jnp.isfinite(total)) & jnp.all(jnp.isfinite(bank))
            if not bool(jax.device_get(finite)):
                raise ValueError(
                    f"Non-finite native heterodyne summaries for {detector.name}; "
                    "check in-band PSD, data and reference response"
                )
            scale = 4.0 / duration
            total, bank = jax.block_until_ready((total * scale, bank * scale))
        diagnostics = {
            "backend": "jax",
            "reducer": "bin-aligned-fp64-gemm-deterministic-combine-with-segment-fallback",
            "device": str(device),
            "native_frequency_samples": n_frequencies,
            "chunk_size": chunk_size,
            "chunks": len(chunk_starts),
            "tiled_chunks": tiled_chunks,
            "tile_size": tile_size,
            "tile_descriptor_shapes": [(tile_size, padded_tiles)],
            "padded_local_bins": padded_local,
            "max_padded_tile_slots": padded_tiles * tile_size,
            "max_tile_matrix_elements": matrix_elements,
            "tile_power_weight_elements": padded_tiles * tile_size * planes,
            "tile_combine_elements": padded_tiles * padded_local,
            "tile_matrix_element_budget": _MAX_TILE_MATRIX_ELEMENTS,
            "requested_chunk_size": self.reference_chunk_size,
            "compiled_executables": int(tiled_chunks > 0)
            + int(tiled_chunks < len(chunk_starts)),
            "time_anchors": n_anchors,
            "data_order": interpolation_order,
            "norm_order": norm_order,
            "wall_seconds_including_compilation": time.perf_counter() - started,
        }
        if probe_bank is not None:
            diagnostics["native_probe_compiled_executables"] = 1
            diagnostics["native_probe_count"] = probe_bank.count
            diagnostics["native_probe_max_waveform_samples"] = chunk_size + 2
            diagnostics["native_probe_parameter_frequency_elements"] = (
                probe_bank.count * (chunk_size + 2)
            )
            diagnostics["native_probes"] = probe_bank.channel_result(
                detector, probe_state, chunks=len(chunk_starts), chunk_size=chunk_size
            )
        return total, bank, diagnostics

    @staticmethod
    def _segmented_coefficient_sums(
        *,
        data: np.ndarray,
        psd: np.ndarray,
        frequencies: np.ndarray,
        reference: np.ndarray,
        bins: np.ndarray,
        interpolation_order: int = 1,
        norm_order: Optional[int] = None,
    ) -> np.ndarray:
        """Accumulate unnormalized bin summaries.

        Order 1 returns the classic ``A0, A1, B0, B1`` rows with raw frequency
        shifts.  Higher orders return ``A_0..A_K`` followed by ``B_0..B_2K``
        as moments of the normalized bin coordinate ``u`` in ``[-1, 1]``.
        """

        if norm_order is not None:
            a, b = polynomial_moments(
                frequencies, data, psd, reference, bins, interpolation_order, norm_order
            )
            return np.concatenate((a, b), axis=0)

        n_bins = len(bins) - 1
        if n_bins <= 0 or np.any(np.diff(bins) <= 0):
            raise ValueError("heterodyne frequency-bin edges must be increasing")

        indices = np.searchsorted(bins, frequencies, side="right") - 1
        indices = np.where(frequencies == bins[-1], n_bins - 1, indices)
        valid = (indices >= 0) & (indices < n_bins)
        indices = indices[valid]
        selected_frequencies = frequencies[valid]
        centers = 0.5 * (bins[:-1] + bins[1:])
        shifts = selected_frequencies - centers[indices]

        data_product = (data * reference.conj() / psd)[valid]
        self_product = (reference * reference.conj() / psd)[valid]

        def segmented_sum(values: np.ndarray) -> np.ndarray:
            result = np.zeros(n_bins, dtype=values.dtype)
            np.add.at(result, indices, values)
            return result

        if int(interpolation_order) != 1:
            order = int(interpolation_order)
            half_widths = 0.5 * (bins[1:] - bins[:-1])
            u = shifts / half_widths[indices]
            return np.stack(
                [segmented_sum(data_product * u**k) for k in range(order + 1)]
                + [segmented_sum(self_product * u**k) for k in range(2 * order + 1)]
            )

        return np.stack(
            (
                segmented_sum(data_product),
                segmented_sum(data_product * shifts),
                segmented_sum(self_product),
                segmented_sum(self_product * shifts),
            )
        )

    # --- reference-parameter optimization ---

    def maximize_likelihood(
        self,
        prior: Prior,
        likelihood_transforms: list[NtoMTransform],
        optimizer_popsize: int = 500,
        optimizer_n_steps: int = 1000,
        optimizer_target: Optional[float] = None,
    ):
        """Find the maximum-likelihood parameters using CMA-ES.

        Uses ``evosax.CMA_ES`` (Covariance Matrix Adaptation Evolution
        Strategy) to search the full parameter space.  The initial mean is
        drawn from the prior and the entire ask/tell loop is compiled with
        ``jax.lax.while_loop`` for speed, stopping once ``optimizer_n_steps``
        generations have run or ``optimizer_target`` has been reached,
        whichever comes first.

        Args:
            prior: Prior used to seed the initial CMA-ES mean.
            likelihood_transforms: Transforms mapping sampling parameters to
                likelihood parameters.
            optimizer_popsize: Population size for CMA-ES.
                Defaults to 500.
            optimizer_n_steps: Maximum number of CMA-ES generations.
                Defaults to 1000.
            optimizer_target: Optional log-likelihood value (same scale as
                `evaluate`) at which to stop early, before
                ``optimizer_n_steps`` is reached. Defaults to None, which
                always runs the full ``optimizer_n_steps`` generations.
        """
        parameter_names = list(prior.parameter_names)
        n_dim = len(parameter_names)

        # ------------------------------------------------------------------
        # Reconstruct f_min / f_max per detector from already-set bounds
        # ------------------------------------------------------------------
        f_min_dict = {d.name: d.frequency_bounds[0] for d in self.detectors}
        f_max_dict = {d.name: d.frequency_bounds[1] for d in self.detectors}

        # ------------------------------------------------------------------
        # Build the full (un-marginalized) TransientLikelihoodFD objective
        # ------------------------------------------------------------------
        full_likelihood = TransientLikelihoodFD(
            detectors=self.detectors,
            waveform=self.waveform,
            f_min=f_min_dict,
            f_max=f_max_dict,
            trigger_time=self.trigger_time,
        )

        # ------------------------------------------------------------------
        # Normalize the search space using the prior sample statistics so
        # that every dimension has unit variance before CMA-ES sees it.
        # CMA-ES then operates with std_init=1e-3 in a space where each
        # parameter already lives on a comparable scale.
        # ------------------------------------------------------------------
        n_init = max(optimizer_popsize, 1000)
        init_samples = prior.sample(jax.random.key(0), n_init)
        sample_matrix = jnp.column_stack(
            [init_samples[key] for key in parameter_names]
        )  # (n_init, n_dim)
        prior_mean = jnp.mean(sample_matrix, axis=0)
        prior_std = jnp.std(sample_matrix, axis=0)

        def _log_likelihood(z: Float[Array, " n_dim"]) -> FloatScalar:
            """Evaluate -logL for a single normalized parameter vector."""
            x = prior_mean + prior_std * z
            named_params = dict(zip(parameter_names, x, strict=True))
            prior_log_prob = prior.log_prob(named_params)
            for transform in likelihood_transforms:
                named_params = transform.forward(named_params)
            named_params = apply_fixed_parameters(named_params, self.fixed_parameters)
            return jnp.where(
                jnp.isfinite(prior_log_prob),
                -full_likelihood.evaluate(named_params),
                jnp.inf,
            )

        _log_likelihood_vmap = jax.vmap(_log_likelihood)

        # ------------------------------------------------------------------
        # Set up CMA-ES in normalized space: init_mean=0, std_init=1e-3
        # ------------------------------------------------------------------
        es = CMA_ES(population_size=optimizer_popsize, solution=jnp.zeros(n_dim))
        es_params = replace(es.default_params, std_init=1e-3)
        key = jax.random.key(42)
        state = es.init(key, jnp.zeros(n_dim), es_params)

        logger.info(
            f"Running evosax CMA-ES: "
            f"{n_dim}D, popsize={optimizer_popsize}, n_steps={optimizer_n_steps}"
            + (
                f", optimizer_target={optimizer_target}"
                if optimizer_target is not None
                else ""
            )
        )

        target_fitness = -jnp.inf if optimizer_target is None else -optimizer_target

        def _cond(carry):
            state, _ = carry
            return (state.generation_counter < optimizer_n_steps) & (
                state.best_fitness > target_fitness
            )

        def _step(carry):
            state, key = carry
            key, key_ask, key_tell = jax.random.split(key, 3)
            population, state = es.ask(key_ask, state, es_params)
            fitness = _log_likelihood_vmap(population)
            # Replace NaN/inf with a large penalty so CMA-ES state is never
            # corrupted by unphysical parameter samples (e.g. q < 0 → eta < 0
            # → waveform returns NaN).  Without this, jnp.argmin treats NaN as
            # the smallest value, best_solution never leaves its NaN initial
            # value, and the entire optimizer output is NaN.
            fitness = jnp.where(
                jnp.isfinite(fitness), fitness, jnp.finfo(jnp.float64).max
            )
            state, _ = es.tell(key_tell, population, fitness, state, es_params)
            return (state, key)

        state, _ = jax.lax.while_loop(_cond, _step, (state, key))

        best_fitness = float(state.best_fitness)
        logger.debug(
            f"CMA-ES finished after {int(state.generation_counter)} generations "
            f"(limit {optimizer_n_steps}), best_fitness={best_fitness:.4f}"
        )
        best_z = state.best_solution

        # ------------------------------------------------------------------
        # Convert best solution back to named parameters
        # ------------------------------------------------------------------
        best_x = prior_mean + prior_std * best_z
        named_params = dict(zip(parameter_names, best_x, strict=True))
        for transform in likelihood_transforms:
            named_params = transform.forward(named_params)
        named_params = apply_fixed_parameters(named_params, self.fixed_parameters)
        return named_params


# ---------------------------------------------------------------------------
# Multi-banded likelihood
# ---------------------------------------------------------------------------
class MultibandedTransientLikelihoodFD(SingleEventLikelihood):
    """Multi-banded likelihood for gravitational wave transient events.

    This implements the multi-banding method described in S. Morisaki, 2021, arXiv:2104.07813.
    The method divides the frequency range into bands with different resolutions,
    using coarser grids at higher frequencies to speed up likelihood evaluation.

    Attributes:
        reference_chirp_mass (Float): Reference chirp mass for determining frequency bands.
        reference_chirp_mass_in_second (Float): Geometrised reference chirp mass in time unit [second].
        highest_mode (int): Maximum magnetic number of GW moments (fixed to 2 for 22-mode).
        accuracy_factor (Float): Parameter L controlling approximation accuracy.
        time_offset (Float): Time offset for band construction.
        delta_f_end (Float): Frequency scale for high-frequency tapering.
        durations (Array): Durations of each band.
        fb_dfb (Array): Starting frequencies and taper widths for each band.
        linear_coeffs (dict): Pre-computed coefficients for (d|h) inner product.
        quadratic_coeffs (dict): Pre-computed coefficients for (h|h) inner product.

    Args:
        detectors (Sequence[Detector]): List of detector objects.
        waveform (Waveform): Waveform model to evaluate.
        fixed_parameters (Optional[dict]): Fixed parameters for the likelihood.
        f_min (Float | dict[str, Float]): Minimum frequency for likelihood
            evaluation, or a dict mapping detector name to per-detector Float.
        f_max (Float | dict[str, Float]): Maximum frequency for likelihood
            evaluation, or a dict mapping detector name to per-detector Float.
        trigger_time (Float): GPS time of the event trigger.
        highest_mode (int): Maximum magnetic number (default 2, for 22-mode only).
        accuracy_factor (Float): Accuracy parameter L (default 5.0).
        prior (Optional[Prior]): Combined prior object.  Needed when *reference_chirp_mass*,
            *time_offset*, or *delta_f_end* are ``None`` so they can be inferred
            automatically from the prior bounds.
        reference_chirp_mass (Optional[Float]): Reference chirp mass in solar masses.
            Use the minimum of your chirp-mass prior for maximum speedup.  When
            ``None``, the value is inferred from the ``M_c`` component of *prior*.
        time_offset (Optional[Float]): Time offset in seconds.  When ``None``,
            inferred from the ``t_c`` (or ``t_{ifo}``) prior range; falls back
            to 2.12 s with a warning when the prior is unavailable.
        delta_f_end (Optional[Float]): End frequency taper scale in Hz.  When
            ``None``, inferred from the ``t_c`` prior range; falls back to 53.0 Hz.
        max_banding_frequency (Optional[Float]): Upper limit on band starting frequency.
        min_banding_duration (Float): Minimum duration for bands.
    """

    highest_mode: int
    accuracy_factor: float
    reference_chirp_mass: float
    reference_chirp_mass_in_second: float
    time_offset: float
    delta_f_end: float
    max_banding_frequency: float
    min_banding_duration: float

    durations: Float[Array, " n_bands"]
    fb_dfb: Float[Array, "n_bands+1 2"]

    unique_frequencies: Float[Array, " n_unique"]
    unique_to_original: Array

    linear_coeffs: dict[str, Float[Array, " n_total_points"]]
    quadratic_coeffs: dict[str, Float[Array, " n_total_points"]]

    def __init__(
        self,
        detectors: Sequence[Detector],
        waveform: Waveform,
        fixed_parameters: Optional[FixedParameters] = None,
        f_min: float | dict[str, float] = 0,
        f_max: float | dict[str, float] = jnp.inf,
        trigger_time: float = 0,
        highest_mode: int = 2,
        accuracy_factor: float = 5.0,
        prior: Optional[Prior] = None,
        reference_chirp_mass: Optional[float] = None,
        time_offset: Optional[float] = None,
        delta_f_end: Optional[float] = None,
        max_banding_frequency: Optional[float] = None,
        min_banding_duration: float = 0.0,
    ):

        super().__init__(detectors, waveform, fixed_parameters)

        xg_response = bool(
            getattr(self.waveform, "time_dependent_response", False)
            or getattr(self.waveform, "response_is_time_dependent", False)
            or any(
                getattr(detector, "time_dependent_response", False)
                or getattr(detector, "response_is_time_dependent", False)
                or getattr(detector, "finite_arm_response", False)
                for detector in self.detectors
            )
        )
        if xg_response:
            raise ValueError(
                "XG detector responses are not qualified with the multiband likelihood"
            )

        logger.info("Initializing multi-banded likelihood...")

        reference_chirp_mass = self._resolve_reference_chirp_mass(
            reference_chirp_mass, prior
        )
        time_offset, delta_f_end = self._resolve_time_params(
            time_offset, delta_f_end, prior, float(trigger_time), detectors
        )
        self._validate_banding_params(
            reference_chirp_mass,
            highest_mode,
            accuracy_factor,
            time_offset,
            delta_f_end,
            min_banding_duration,
            max_banding_frequency,
        )

        self.reference_chirp_mass = reference_chirp_mass
        self.reference_chirp_mass_in_second = reference_chirp_mass * MTSUN
        self.highest_mode = highest_mode
        self.accuracy_factor = accuracy_factor
        self.time_offset = time_offset
        self.delta_f_end = delta_f_end
        self.min_banding_duration = min_banding_duration

        _f_mins = []
        _f_maxs = []
        for detector in detectors:
            f_min_ifo = f_min[detector.name] if isinstance(f_min, dict) else f_min
            f_max_ifo = f_max[detector.name] if isinstance(f_max, dict) else f_max
            detector.set_frequency_bounds(f_min_ifo, f_max_ifo)
            sliced = detector.sliced_frequencies
            _f_mins.append(float(sliced[0]))
            _f_maxs.append(float(sliced[-1]))

        self.minimum_frequency = min(_f_mins)
        self.maximum_frequency = max(_f_maxs)

        fmax_spa = (
            (15 / 968) ** (3 / 5)
            * (self.highest_mode / (2 * jnp.pi)) ** (8 / 5)
            / self.reference_chirp_mass_in_second
        )
        self.max_banding_frequency = (
            min(max_banding_frequency, fmax_spa)
            if max_banding_frequency is not None
            else fmax_spa
        )

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(trigger_time)

        self._setup_frequency_bands()
        self._setup_integers()
        self._setup_waveform_frequency_points()
        self._setup_linear_coefficients()
        self._setup_quadratic_coefficients()

        logger.info("Multi-banding setup complete with %d bands", self.n_bands)

    # --- direct evaluation ---

    def _evaluate(self, params: dict[str, Float]) -> FloatScalar:
        waveform_sky = self.waveform(self.unique_frequencies, params)
        return self._likelihood(params, waveform_sky)

    # --- waveform-cache evaluation ---

    def _generate_waveform(
        self, params: dict[str, Float]
    ) -> dict[str, Complex[Array, " n_freq"]]:
        """Generate polarizations at multiband frequencies for cache reuse.

        Evaluated at unit distance when ``waveform_caches_distance`` is True,
        so ``d_L`` is not a cache dependency; otherwise ``d_L`` is a
        dependency like any other waveform parameter.
        """
        return self._waveform_sky_for_cache(self.unique_frequencies, params)

    def _evaluate_from_waveform(
        self,
        params: dict[str, Float],
        waveform_cache: dict[str, Complex[Array, " n_freq"]],
    ) -> FloatScalar:
        """Core likelihood evaluation from a pre-generated waveform cache."""
        waveform_sky = self._waveform_sky_from_cache(
            self.unique_frequencies, waveform_cache, params
        )
        return self._likelihood(params, waveform_sky)

    # --- shared likelihood core ---

    def _likelihood(
        self,
        params: dict[str, Float],
        waveform_sky: dict[str, Complex[Array, " n_freq"]],
    ) -> FloatScalar:
        """Core likelihood computation from physical-distance multiband polarizations."""
        log_likelihood: FloatScalar = jnp.zeros(())

        for detector in self.detectors:
            # Get detector response at banded frequencies.
            h_det_unique = detector.fd_response(
                self.unique_frequencies, waveform_sky, params
            )
            strain = h_det_unique[self.unique_to_original]

            d_inner_h = jnp.sum(strain * self.linear_coeffs[detector.name])
            h_inner_h = jnp.sum(
                jnp.real(strain * jnp.conj(strain))
                * self.quadratic_coeffs[detector.name]
            )
            log_likelihood += jnp.real(d_inner_h) - h_inner_h / 2

        return log_likelihood

    # --- prior-inference and validation helpers ---

    def _resolve_reference_chirp_mass(
        self,
        reference_chirp_mass: Optional[Float],
        prior: Optional[Prior],
    ) -> float:
        """Return ``reference_chirp_mass``, inferring from the M_c prior minimum when not provided."""
        if reference_chirp_mass is not None:
            return reference_chirp_mass
        if prior is None:
            raise ValueError(
                "Either reference_chirp_mass or a prior with an M_c component must be provided."
            )
        mc_prior = find_specific_prior(prior, "M_c")
        mc_bounds = mc_prior.get_bounds() if mc_prior is not None else None
        if mc_bounds is None:
            raise ValueError(
                "reference_chirp_mass=None but no M_c prior found. "
                "Pass either reference_chirp_mass or a prior with an M_c component."
            )
        mc_min, _ = mc_bounds
        logger.info(
            "reference_chirp_mass inferred from M_c prior minimum: %.4f M_sun", mc_min
        )
        return mc_min

    def _resolve_time_params(
        self,
        time_offset: Optional[float],
        delta_f_end: Optional[float],
        prior: Optional[Prior],
        trigger_time: float,
        detectors: Sequence[Detector],
    ) -> tuple[float, float]:
        """Return ``(time_offset, delta_f_end)``, inferring from t_c prior bounds when not provided.

        Inference uses the geocentric coalescence time ``t_c`` only.
        Detector-frame time ``t_det`` is not supported because
        ``t_det = t_c + sky_delay(ra, dec)`` and the delay is sky-position-dependent,
        so ``t_c`` bounds cannot be derived from a ``t_det`` prior at setup time.
        Falls back to defaults (2.12 s, 53.0 Hz) when inference is not possible.
        """
        inferred_to: Optional[float] = None
        inferred_dfe: Optional[float] = None

        if prior is not None and (time_offset is None or delta_f_end is None):
            tc_prior = find_specific_prior(prior, "t_c")
            tc_bounds = tc_prior.get_bounds() if tc_prior is not None else None
            if tc_bounds is not None:
                tc_min, tc_max = tc_bounds
                t_end = min(
                    float(d.data.start_time) + float(d.data.duration) - trigger_time
                    for d in detectors
                )
                RE_S = EARTH_RADIUS_LIGHT_S
                denom = t_end - tc_max - RE_S

                if denom <= 0:
                    raise ValueError(
                        f"Cannot infer delta_f_end from t_c prior: "
                        f"t_end - xmax - s = {t_end:.4f} - {tc_max:.4f} - {RE_S:.6f} = {denom:.6f} <= 0. "
                        "Check that the t_c prior upper bound is well within the data segment."
                    )
                inferred_to = t_end - tc_min + RE_S
                inferred_dfe = 100.0 / denom

        if time_offset is None:
            if inferred_to is not None:
                time_offset = inferred_to
                logger.info("time_offset inferred from t_c prior: %.4f s", time_offset)
            else:
                time_offset = 2.12
                logger.warning(
                    "time_offset cannot be inferred from prior; using default 2.12 s"
                )

        if delta_f_end is None:
            if inferred_dfe is not None:
                delta_f_end = inferred_dfe
                logger.info("delta_f_end inferred from t_c prior: %.4f Hz", delta_f_end)
            else:
                delta_f_end = 53.0
                logger.warning(
                    "delta_f_end cannot be inferred from prior; using default 53.0 Hz"
                )

        return time_offset, delta_f_end

    def _validate_banding_params(
        self,
        reference_chirp_mass: float,
        highest_mode: int,
        accuracy_factor: float,
        time_offset: float,
        delta_f_end: float,
        min_banding_duration: float,
        max_banding_frequency: Optional[float],
    ) -> None:
        """Validate the related multi-banding configuration values."""
        if reference_chirp_mass <= 0:
            raise ValueError(
                f"reference_chirp_mass must be > 0, got {reference_chirp_mass}"
            )
        if highest_mode <= 0:
            raise ValueError(f"highest_mode must be > 0, got {highest_mode}")
        if accuracy_factor <= 0:
            raise ValueError(f"accuracy_factor must be > 0, got {accuracy_factor}")
        if time_offset < 0:
            raise ValueError(f"time_offset must be >= 0, got {time_offset}")
        if delta_f_end <= 0:
            raise ValueError(f"delta_f_end must be > 0, got {delta_f_end}")
        if min_banding_duration < 0:
            raise ValueError(
                f"min_banding_duration must be >= 0, got {min_banding_duration}"
            )
        if max_banding_frequency is not None and max_banding_frequency <= 0:
            raise ValueError(
                f"max_banding_frequency must be > 0, got {max_banding_frequency}"
            )

    # --- band structure ---

    @property
    def n_bands(self) -> int:
        """Number of frequency bands."""
        return len(self.durations)

    def _compute_tau_dtaudf(self, f: Float) -> tuple[Float, Float]:
        """Compute time-to-merger and its derivative using 0PN formula.

        Parameters
        ----------
        f : Float
            Input frequency in Hz.

        Returns
        -------
        tuple[Float, Float]
            (tau, dtaudf) where tau is time-to-merger in seconds and dtaudf is its derivative (negative, in seconds/Hz).
        """
        f_22 = 2 * f / self.highest_mode
        piMf = self.reference_chirp_mass_in_second * (
            jnp.pi * self.reference_chirp_mass_in_second * f_22
        ) ** (-8 / 3)
        tau = 5 / 256 * piMf
        dtaudf = -5 / 96 * piMf / f
        return tau, dtaudf

    def _find_starting_frequency(
        self, duration: float, f_now: float
    ) -> tuple[Optional[Float], Optional[Float]]:
        """Find starting frequency of next band via bisection search.

        Finds frequency satisfying conditions (10) and (51) of arXiv:2104.07813:
        - Time containment: tau(f) + L * sqrt(-dtau/df) < duration - time_offset
        - Smooth transition: f - 1/sqrt(-dtau/df) > f_now

        Parameters
        ----------
        duration : Float
            Duration of the next band.
        f_now : Float
            Starting frequency of current band.

        Returns
        -------
        tuple[Optional[Float], Optional[Float]]
            (fnext, dfnext) or (None, None) if no valid frequency exists.
        """

        def _is_above_fnext(f):
            tau, dtaudf = self._compute_tau_dtaudf(f)
            cond1 = (
                duration
                - self.time_offset
                - tau
                - self.accuracy_factor * jnp.sqrt(-dtaudf)
            ) > 0
            cond2 = f - 1.0 / jnp.sqrt(-dtaudf) - f_now > 0
            return cond1 and cond2

        fmin, fmax = f_now, self.max_banding_frequency

        if not _is_above_fnext(fmax):
            return None, None

        # Bisection search
        f = (fmin + fmax) / 2.0
        while fmax - fmin > 1e-2 / duration:
            f = (fmin + fmax) / 2.0
            if _is_above_fnext(f):
                fmax = f
            else:
                fmin = f

        _, dtaudf = self._compute_tau_dtaudf(f)
        return f, 1.0 / jnp.sqrt(-dtaudf)

    def _setup_frequency_bands(self) -> None:
        """Set up frequency bands with geometrically decreasing durations.

        Bands have durations T, T/2, T/4, ... where T is the original data duration.

        Sets:
            self.durations: Array of band durations
            self.fb_dfb: Array of [starting_freq, taper_width] for each band
        """
        original_duration = float(self.detectors[0].data.duration)

        durations_list = [original_duration]
        fb_dfb_list = [[self.minimum_frequency, 0.0]]

        dnext: float = original_duration / 2

        while dnext > max(self.time_offset, self.min_banding_duration):
            f_now, _ = fb_dfb_list[-1]
            fnext, dfnext = self._find_starting_frequency(dnext, f_now)

            if (
                fnext is not None
                and dfnext is not None
                and fnext < min(self.maximum_frequency, self.max_banding_frequency)
            ):
                durations_list.append(dnext)
                fb_dfb_list.append([fnext, dfnext])
                dnext /= 2
            else:
                break

        # Add final boundary
        fb_dfb_list.append(
            [self.maximum_frequency + self.delta_f_end, self.delta_f_end]
        )

        self.durations = jnp.array(durations_list)
        self.fb_dfb = jnp.array(fb_dfb_list)

        logger.info(
            f"Frequency range divided into {self.n_bands} bands with "
            f"intervals: {', '.join(['1/' + str(d) + ' Hz' for d in durations_list])}"
        )

    def _setup_integers(self) -> None:
        """Set up integer indices for each band.

        Sets:
            self.Nbs: Number of samples in downsampled data per band
            self.Mbs: Number of samples in shortened data per band
            self.Ks_Ke: Start/end frequency indices per band
        """
        original_duration = float(self.detectors[0].data.duration)
        durations = self.durations.tolist()
        fb_dfb = self.fb_dfb.tolist()

        Nbs_list = []
        Mbs_list = []
        Ks_Ke_list = []

        for b in range(self.n_bands):
            dnow = durations[b]
            f_now, dfnow = fb_dfb[b]
            fnext = fb_dfb[b + 1][0]

            Nb = max(
                round_up_to_power_of_two(int(2.0 * fnext * original_duration + 1)),
                2**b,
            )
            Nbs_list.append(Nb)
            Mbs_list.append(Nb // (2**b))
            Ks_Ke_list.append(
                [jnp.ceil((f_now - dfnow) * dnow), jnp.floor(fnext * dnow)]
            )

        self.Nbs = jnp.array(Nbs_list, dtype=jnp.int32)
        self.Mbs = jnp.array(Mbs_list, dtype=jnp.int32)
        self.Ks_Ke = jnp.array(Ks_Ke_list, dtype=jnp.int32)

    def _setup_waveform_frequency_points(self) -> None:
        """Set up frequency points where waveforms are evaluated.

        Creates banded frequency points and finds unique frequencies to avoid
        redundant waveform evaluations.

        Sets:
            self.banded_frequency_points: All frequency points across bands
            self.start_end_idxs: Start/end indices for each band
            self.unique_frequencies: Unique frequencies for waveform evaluation
            self.unique_to_original: Mapping from unique back to banded
        """
        durations = self.durations.tolist()
        Ks_Ke = self.Ks_Ke.tolist()

        band_freqs_list = []
        start_end_list = []
        start_idx = 0

        for b in range(self.n_bands):
            Ks, Ke = Ks_Ke[b]
            band_freqs = jnp.arange(Ks, Ke + 1) / durations[b]
            band_freqs_list.append(band_freqs)
            end_idx = start_idx + Ke - Ks
            start_end_list.append([start_idx, end_idx])
            start_idx = end_idx + 1

        banded_freq_array = jnp.concatenate(band_freqs_list)
        unique_freqs, idxs = jnp.unique(banded_freq_array, return_inverse=True)

        self.banded_frequency_points = banded_freq_array
        self.start_end_idxs = jnp.array(start_end_list, dtype=jnp.int32)
        self.unique_frequencies = unique_freqs
        self.unique_to_original = idxs.astype(jnp.int32)

    def _get_window_sequence(
        self, delta_f: float, start_idx: int, length: int, band: int
    ) -> Array:
        """Compute cosine-tapered window function for a frequency band.

        Window is 1 in band interior, with smooth cosine tapers at edges.

        Parameters
        ----------
        delta_f : Float
            Frequency interval.
        start_idx : int
            Starting frequency index (frequency = start_idx * delta_f).
        length : int
            Number of frequency points.
        band : int
            Band index.

        Returns
        -------
        Array
            Window sequence of given length.
        """

        f_now, dfnow = self.fb_dfb[band].tolist()
        fnext, dfnext = self.fb_dfb[band + 1].tolist()

        window = jnp.zeros(length)

        increase_start = max(
            0, min(length, int(jnp.floor((f_now - dfnow) / delta_f)) - start_idx + 1)
        )
        unity_start = max(0, min(length, int(jnp.ceil(f_now / delta_f)) - start_idx))
        decrease_start = max(
            0, min(length, int(jnp.floor((fnext - dfnext) / delta_f)) - start_idx + 1)
        )
        decrease_stop = max(0, min(length, int(jnp.ceil(fnext / delta_f)) - start_idx))

        window = window.at[unity_start:decrease_start].set(1.0)

        if increase_start < unity_start and dfnow > 0:
            frequencies = (
                jnp.arange(increase_start, unity_start) + start_idx
            ) * delta_f
            window = window.at[increase_start:unity_start].set(
                (1.0 + jnp.cos(jnp.pi * (frequencies - f_now) / dfnow)) / 2.0
            )

        if decrease_start < decrease_stop:
            frequencies = (
                jnp.arange(decrease_start, decrease_stop) + start_idx
            ) * delta_f
            window = window.at[decrease_start:decrease_stop].set(
                (1.0 - jnp.cos(jnp.pi * (frequencies - fnext) / dfnext)) / 2.0
            )

        return window

    def _setup_linear_coefficients(self) -> None:
        """Pre-compute coefficients for (d|h) inner product.

        For each band:
        1. Apply frequency mask and divide by PSD
        2. IFFT to time domain, take last M^(b) samples
        3. FFT back to get shortened data
        4. Multiply by window and normalization factor

        Sets:
            self.linear_coeffs: Dict mapping detector name to coefficient array
        """
        Ks_Ke = self.Ks_Ke.tolist()
        Nbs = self.Nbs.tolist()
        Mbs = self.Mbs.tolist()
        durations = self.durations.tolist()
        N = Nbs[-1]

        self.linear_coeffs = {}

        for detector in self.detectors:
            logger.info(f"Pre-computing linear coefficients for {detector.name}")
            data_fd = jnp.array(detector.data.fd)
            psd = jnp.array(detector.psd.values)
            freq_mask = jnp.array(detector.frequency_mask)

            valid_len = min(len(data_fd), N // 2 + 1)
            mask_valid = freq_mask[:valid_len]
            safe_psd = jnp.where(mask_valid, psd[:valid_len], 1.0)
            values = jnp.where(mask_valid, data_fd[:valid_len] / safe_psd, 0.0)
            fddata = jnp.zeros(N // 2 + 1, dtype=complex).at[:valid_len].set(values)

            band_coeffs = []

            for b in range(self.n_bands):
                Ks, Ke = Ks_Ke[b]
                Nb = Nbs[b]
                Mb = Mbs[b]
                db = durations[b]

                window = self._get_window_sequence(1.0 / db, Ks, Ke - Ks + 1, b)

                fddata_band = fddata[: Nb // 2 + 1].at[-1].set(0.0)

                tddata = jnp.fft.irfft(fddata_band)[-Mb:]
                fddata_shortened = jnp.fft.rfft(tddata)[Ks : Ke + 1]

                band_coeffs.append((4.0 / db) * window * jnp.conj(fddata_shortened))

            self.linear_coeffs[detector.name] = jnp.concatenate(band_coeffs)

    def _setup_quadratic_coefficients(self) -> None:
        """Pre-compute coefficients for (h|h) using linear interpolation.

        For each band and coarse frequency point, compute the weighted sum
        of 1/PSD values using linear interpolation weights.

        Sets:
            self.quadratic_coeffs: Dict mapping detector name to coefficient array
        """

        original_duration = float(self.detectors[0].data.duration)
        start_end_idxs = self.start_end_idxs.tolist()
        durations = self.durations.tolist()
        fb_dfb = self.fb_dfb.tolist()
        banded_frequency_points = self.banded_frequency_points.tolist()

        self.quadratic_coeffs = {}

        for detector in self.detectors:
            psd = jnp.array(detector.psd.values)
            freq_mask = jnp.array(detector.frequency_mask)

            band_coeffs = []

            for b in range(self.n_bands):
                logger.debug(f"Pre-computing quadratic coefficients for band {b}")

                start_idx, end_idx = start_end_idxs[b]
                banded_freqs = banded_frequency_points[start_idx : end_idx + 1]
                prefactor = 4 * durations[b] / original_duration

                f_now, dfnow = fb_dfb[b]
                fnext = fb_dfb[b + 1][0]
                start_idx_orig = int(jnp.ceil((f_now - dfnow) * original_duration))
                window_length = (
                    int(jnp.floor(fnext * original_duration)) - start_idx_orig + 1
                )

                window = self._get_window_sequence(
                    1.0 / original_duration, start_idx_orig, window_length, b
                )

                # Compute window / PSD
                end_idx_orig = min(start_idx_orig + len(window) - 1, len(psd) - 1)
                valid_len = end_idx_orig - start_idx_orig + 1

                local_mask = freq_mask[start_idx_orig : end_idx_orig + 1]
                psd_slice = psd[start_idx_orig : end_idx_orig + 1]
                safe_psd = jnp.where(local_mask, psd_slice, 1.0)
                window_over_psd = (
                    jnp.where(local_mask, 1.0 / safe_psd, 0.0) * window[:valid_len]
                )

                # Compute coefficients using linear interpolation
                n_coeff = len(banded_freqs)
                coeffs = jnp.zeros(n_coeff)

                for k in range(n_coeff - 1):
                    sum_start = (
                        start_idx_orig
                        if k == 0
                        else max(
                            start_idx_orig,
                            int(jnp.ceil(original_duration * banded_freqs[k])),
                        )
                    )
                    sum_end = (
                        end_idx_orig
                        if k == n_coeff - 2
                        else min(
                            end_idx_orig,
                            int(jnp.ceil(original_duration * banded_freqs[k + 1])) - 1,
                        )
                    )

                    freqs_in_sum = (
                        jnp.arange(sum_start, sum_end + 1) / original_duration
                    )
                    local_start = sum_start - start_idx_orig
                    local_end = sum_end - start_idx_orig + 1
                    wop = window_over_psd[local_start:local_end]

                    coeffs = coeffs.at[k].add(
                        prefactor * jnp.sum((banded_freqs[k + 1] - freqs_in_sum) * wop)
                    )
                    coeffs = coeffs.at[k + 1].add(
                        prefactor * jnp.sum((freqs_in_sum - banded_freqs[k]) * wop)
                    )

                band_coeffs.append(coeffs)

            self.quadratic_coeffs[detector.name] = jnp.concatenate(band_coeffs)


likelihood_presets = {
    "TransientLikelihoodFD": TransientLikelihoodFD,
    "HeterodynedTransientLikelihoodFD": HeterodynedTransientLikelihoodFD,
    "MultibandedTransientLikelihoodFD": MultibandedTransientLikelihoodFD,
}
