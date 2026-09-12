"""Versioned, chunk-independent Gaussian noise for native FD data.

The algorithm intentionally differs from the legacy full-array random draw.
Record ``INDEXED_NOISE_ALGORITHM`` with the seed, detector key derivation and
JAX version when recording a realization's provenance.
"""

from numbers import Integral, Real

import jax
import jax.numpy as jnp

INDEXED_NOISE_ALGORITHM = "indexed-threefry2x32-fold64-hi-lo-normal2-f64-v1"
_MAX_INDEX = 2**64 - 1


def _threefry_key(key):
    """Pin the generator even when the process-wide PRNG default differs."""
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        if key.shape != () or str(jax.random.key_impl(key)) != "threefry2x32":
            raise ValueError("indexed noise requires one Threefry2x32 key")
        words = jax.random.key_data(key)
    else:
        words = jnp.asarray(key)
        if words.shape != (2,) or words.dtype != jnp.uint32:
            raise ValueError("indexed noise requires a Threefry uint32[2] key")
    return jax.random.wrap_key_data(words, impl="threefry2x32")


@jax.jit
def _indexed_noise(key, psd, df, first_index, index_valid):
    count = psd.size
    indices = first_index + jnp.arange(count, dtype=jnp.uint64)

    def sample(index):
        high = jnp.asarray(index >> jnp.uint64(32), dtype=jnp.uint32)
        low = jnp.asarray(index & jnp.uint64(0xFFFFFFFF), dtype=jnp.uint32)
        indexed_key = jax.random.fold_in(jax.random.fold_in(key, high), low)
        return jax.random.normal(indexed_key, shape=(2,), dtype=jnp.float64)

    standard = jax.vmap(sample)(indices)
    metadata_valid = index_valid & jnp.isfinite(df) & (df > 0.0)
    if count:
        metadata_valid &= first_index <= jnp.uint64(_MAX_INDEX - (count - 1))
    valid = metadata_valid & jnp.isfinite(psd) & (psd > 0.0)
    variance = jnp.where(valid, psd / (4.0 * df), jnp.nan)
    scale = jnp.sqrt(variance)
    return jax.lax.complex(standard[:, 0] * scale, standard[:, 1] * scale)


def compiled_indexed_noise(key, psd_chunk, df, first_index):
    """Return complex128 native noise using O(chunk size) device storage.

    ``first_index`` is the absolute unsigned 64-bit frequency index, not the
    index relative to a detector's analysis band. For each index, fold its
    high 32 bits then low 32 bits into the supplied detector-specific
    Threefry2x32 key, and draw a float64 standard-normal pair with JAX's
    partitionable Threefry recipe explicitly enabled. Real and
    imaginary components each have variance ``psd_chunk / (4*df)``.

    Keys may be scalar typed Threefry keys or legacy uint32[2] Threefry keys.
    Other PRNG implementations are rejected: their batched-key behavior need
    not preserve this recipe across chunk sizes. The caller derives distinct
    detector keys, for example with ``jax.random.fold_in(root_key, detector_id)``.

    Positive finite PSD and df are required. Bad PSD elements become complex
    NaNs; invalid dynamic df/index metadata make the whole chunk NaN. Invalid
    Python scalar metadata raises before compilation. Index overflow is never
    accepted or silently wrapped. Empty chunks return shape (0,). This wrapper
    also works inside an outer JIT; all size-dependent allocations use only
    the supplied PSD chunk's length.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("indexed noise requires JAX 64-bit precision")
    if isinstance(first_index, Integral) and (
        isinstance(first_index, bool) or not 0 <= first_index <= _MAX_INDEX
    ):
        raise ValueError("first_index must be an unsigned 64-bit integer")
    if isinstance(df, Real) and (
        isinstance(df, bool) or not 0.0 < float(df) < float("inf")
    ):
        raise ValueError("df must be positive and finite")
    spectrum = jnp.asarray(psd_chunk)
    if spectrum.ndim != 1 or jnp.issubdtype(spectrum.dtype, jnp.complexfloating):
        raise ValueError("psd_chunk must be a real vector")
    if isinstance(first_index, Integral):
        if spectrum.size and first_index > _MAX_INDEX - (spectrum.size - 1):
            raise ValueError("noise chunk exceeds the unsigned 64-bit index range")
        index = jnp.asarray(first_index, dtype=jnp.uint64)
        index_valid = jnp.asarray(True)
    else:
        index = jnp.asarray(first_index)
        if index.ndim != 0 or not jnp.issubdtype(index.dtype, jnp.integer):
            raise ValueError("first_index must be an unsigned 64-bit integer")
        index_valid = index >= 0
        index = index.astype(jnp.uint64)
    spacing = jnp.asarray(df)
    if spacing.ndim != 0 or not (
        jnp.issubdtype(spacing.dtype, jnp.floating)
        or jnp.issubdtype(spacing.dtype, jnp.integer)
    ):
        raise ValueError("df must be a positive finite real scalar")
    # This flag changes random.normal's bit recipe even for a pinned typed
    # Threefry key. Keep the versioned realization independent of its caller's
    # process-wide setting, including when tracing inside a larger JIT.
    with jax.threefry_partitionable(True):
        return _indexed_noise(
            _threefry_key(key),
            spectrum.astype(jnp.float64),
            spacing.astype(jnp.float64),
            index,
            index_valid,
        )


__all__ = ["INDEXED_NOISE_ALGORITHM", "compiled_indexed_noise"]
