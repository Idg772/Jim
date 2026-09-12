"""Bounded indexed noise: exact chunk replay and finite-sample Gaussian checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event.heterodyne_noise import (
    INDEXED_NOISE_ALGORITHM,
    compiled_indexed_noise,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("first", [0, 2**32 - 5, 2**63 - 5, 2**64 - 257])
@pytest.mark.parametrize("chunk_size", [1, 17, 64])
def test_realization_is_bitwise_invariant_under_chunk_boundaries(first, chunk_size):
    count = 257
    spectrum = np.geomspace(1e-48, 1e-44, count)
    key = jax.random.fold_in(jax.random.key(2026), 3)
    complete = compiled_indexed_noise(key, spectrum, 1 / 131072.0, first)
    chunks = [
        compiled_indexed_noise(
            key, spectrum[start : start + chunk_size], 1 / 131072.0, first + start
        )
        for start in range(0, count, chunk_size)
    ]
    np.testing.assert_array_equal(np.concatenate(chunks), complete)
    assert complete.dtype == jnp.complex128


def test_versioned_recipe_and_exact_component_normalization():
    assert INDEXED_NOISE_ALGORITHM == "indexed-threefry2x32-fold64-hi-lo-normal2-f64-v1"
    key = jax.random.fold_in(jax.random.key(73), 2)
    first = 2**32 - 1
    spectrum = jnp.array([1.0, 4.0, 16.0])
    actual = compiled_indexed_noise(key, spectrum, 0.25, first)
    expected = []
    for offset in range(3):
        index = first + offset
        indexed_key = jax.random.fold_in(key, np.uint32(index >> 32))
        indexed_key = jax.random.fold_in(indexed_key, np.uint32(index & 0xFFFFFFFF))
        pair = jax.random.normal(indexed_key, (2,), dtype=jnp.float64)
        scale = jnp.sqrt(spectrum[offset] / (4 * 0.25))
        expected.append(jax.lax.complex(pair[0] * scale, pair[1] * scale))
    np.testing.assert_array_equal(actual, jnp.stack(expected))
    # Fourfold PSD and fourfold frequency spacing have opposite amplitude effects.
    np.testing.assert_array_equal(
        compiled_indexed_noise(key, spectrum * 4, 0.25, first), actual * 2
    )
    np.testing.assert_array_equal(
        compiled_indexed_noise(key, spectrum, 1.0, first), actual / 2
    )


def test_high_index_word_detector_keys_and_root_seed_are_distinct():
    key = jax.random.key(19)
    inputs = jnp.ones(128)
    streams = [
        compiled_indexed_noise(jax.random.fold_in(key, detector), inputs, 0.25, first)
        for detector, first in [(0, 0), (1, 0), (0, 2**32), (0, 2**63)]
    ]
    streams.append(compiled_indexed_noise(jax.random.key(20), inputs, 0.25, 0))
    for left in range(len(streams)):
        for right in range(left):
            assert not np.array_equal(streams[left], streams[right])


def test_gaussian_components_have_correct_means_variances_and_correlations():
    count = 65536  # A small CPU ensemble, never a native-duration data allocation.
    spectrum = np.geomspace(1e-48, 1e-44, count)
    df = 1 / 131072.0
    noise = np.asarray(compiled_indexed_noise(jax.random.key(9387), spectrum, df, 901))
    normalized = noise / np.sqrt(spectrum / (4 * df))
    real, imag = normalized.real, normalized.imag
    for component in (real, imag):
        assert abs(np.mean(component)) < 0.025
        assert abs(np.var(component) - 1.0) < 0.04
        assert abs(np.mean(component[:-1] * component[1:])) < 0.025
    assert abs(np.mean(real * imag)) < 0.025
    assert abs(np.mean(np.abs(normalized) ** 2) - 2.0) < 0.05


def test_nested_jit_and_scalar_replay_only_allocate_the_requested_small_chunk():
    @jax.jit
    def generate(key, psd, df, first):
        return compiled_indexed_noise(key, psd, df, first)

    first = 2**63 + 919
    key = jax.random.key(83)
    # A huge absolute offset still produces only seven samples. This would
    # be infeasible if generation allocated or advanced through [0, first].
    noise = generate(key, jnp.ones(7), jnp.asarray(0.25), jnp.uint64(first))
    assert noise.shape == (7,)
    for offset in range(7):
        sample = compiled_indexed_noise(key, jnp.ones(1), 0.25, first + offset)
        np.testing.assert_array_equal(sample[0], noise[offset])


def test_explicit_threefry_is_independent_of_process_default_and_accepts_legacy_keys():
    key = jax.random.key(8, impl="threefry2x32")
    legacy = jax.random.key_data(key)
    baseline = compiled_indexed_noise(key, jnp.ones(13), 0.25, 17)
    with jax.default_prng_impl("rbg"):
        result = compiled_indexed_noise(legacy, jnp.ones(13), 0.25, 17)
    np.testing.assert_array_equal(result, baseline)
    with pytest.raises(ValueError, match="Threefry"):
        compiled_indexed_noise(jax.random.key(8, impl="rbg"), jnp.ones(13), 0.25, 17)


def test_empty_chunks_are_complex128_without_random_samples():
    result = compiled_indexed_noise(jax.random.key(0), np.zeros(0), 0.25, 2**64 - 1)
    assert result.shape == (0,) and result.dtype == jnp.complex128


def test_versioned_recipe_pins_partitionable_flag_inside_and_outside_outer_jit():
    key = jax.random.key(91, impl="threefry2x32")
    spectrum = jnp.linspace(0.5, 2.0, 19)
    with jax.threefry_partitionable(True):
        expected = compiled_indexed_noise(key, spectrum, 0.25, 2**32 + 19)
    with jax.threefry_partitionable(False):
        actual = compiled_indexed_noise(key, spectrum, 0.25, 2**32 + 19)
        nested = jax.jit(compiled_indexed_noise)(
            key, spectrum, 0.25, jnp.uint64(2**32 + 19)
        )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(nested, expected)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_bad_psd_elements_propagate_nan_without_changing_valid_samples(bad):
    key = jax.random.key(22)
    valid = compiled_indexed_noise(key, np.ones(3), 0.25, 5)
    spectrum = np.array([1.0, bad, 1.0])
    actual = np.asarray(compiled_indexed_noise(key, spectrum, 0.25, 5))
    assert np.isnan(actual[1].real) and np.isnan(actual[1].imag)
    np.testing.assert_array_equal(actual[[0, 2]], np.asarray(valid)[[0, 2]])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan, True])
def test_invalid_python_df_raises(bad):
    with pytest.raises(ValueError, match="df must be"):
        compiled_indexed_noise(jax.random.key(0), np.ones(2), bad, 0)


@pytest.mark.parametrize("first", [-1, True, 1.5, 2**64])
def test_invalid_python_index_raises(first):
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        compiled_indexed_noise(jax.random.key(0), np.ones(2), 0.25, first)


def test_index_overflow_raises_eagerly_or_propagates_nan_when_traced():
    key = jax.random.key(0)
    with pytest.raises(ValueError, match="exceeds"):
        compiled_indexed_noise(key, np.ones(2), 0.25, 2**64 - 1)
    sample = compiled_indexed_noise(key, np.ones(1), 0.25, 2**64 - 1)
    assert np.all(np.isfinite(sample))
    dynamic = jax.jit(compiled_indexed_noise)
    for first in (jnp.uint64(2**64 - 1), jnp.int64(-1)):
        assert np.all(np.isnan(dynamic(key, jnp.ones(2), 0.25, first)))
    assert np.all(
        np.isnan(dynamic(key, jnp.ones(2), jnp.asarray(-0.25), jnp.uint64(0)))
    )
