"""Discrete device summaries against the existing NumPy moment constructor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event.heterodyne_moments import polynomial_moments
from jimgw.core.single_event.heterodyne_summary import compiled_summary_chunk

jax.config.update("jax_enable_x64", True)


def _oracle(f, data, psd, reference, edges, anchors, data_order, norm_order):
    a, b = polynomial_moments(f, data, psd, reference, edges, data_order, norm_order)
    anchored = [
        polynomial_moments(
            f,
            data * np.exp(2j * np.pi * f * anchor),
            psd,
            reference,
            edges,
            data_order,
            norm_order,
        )[0]
        for anchor in anchors
    ]
    bank = np.asarray(anchored, dtype=np.complex128).reshape(
        len(anchors), data_order + 1, len(edges) - 1
    )
    return a, b, bank


def _random_inputs(*, shuffle=False):
    rng = np.random.default_rng(916203)
    edges = np.array([2.0, 2.7, 4.0, 5.5, 8.0, 14.0])
    f = np.sort(np.r_[rng.uniform(1.0, 15.0, 263), edges, edges[-1]])
    if shuffle:
        rng.shuffle(f)
    data = rng.normal(size=f.size) + 1j * rng.normal(size=f.size)
    reference = rng.normal(size=f.size) + 1j * rng.normal(size=f.size)
    psd = rng.uniform(0.5, 3.0, size=f.size)
    return f, data, psd, reference, edges


@pytest.mark.parametrize("data_order,norm_order", [(0, 0), (3, 7), (7, 3), (24, 16)])
@pytest.mark.parametrize("anchors", [[], [-0.17, 0.0, 0.12]])
@pytest.mark.parametrize("shuffle", [False, True])
def test_noisy_complex_sums_match_numpy_for_independent_orders(
    data_order, norm_order, anchors, shuffle
):
    inputs = _random_inputs(shuffle=shuffle)
    expected = _oracle(*inputs, anchors, data_order, norm_order)
    actual = compiled_summary_chunk(
        *inputs, jnp.asarray(anchors), data_order=data_order, norm_order=norm_order
    )
    assert [x.dtype for x in actual] == [jnp.complex128, jnp.float64, jnp.complex128]
    for result, target in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, target, rtol=2e-13, atol=2e-13)
    if anchors:
        np.testing.assert_array_equal(actual[0], actual[2][1])


def test_bin_membership_includes_last_edge_once_and_leaves_empty_bins_zero():
    f = np.array([1.0, 2.0, 3.0, 3.0, 4.0, 6.0, 7.0])
    edges = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    data = np.arange(1, f.size + 1, dtype=np.float64)
    a, b, bank = compiled_summary_chunk(
        f,
        data,
        np.ones_like(f),
        np.ones_like(f),
        edges,
        np.array([0.0]),
        data_order=1,
        norm_order=1,
    )
    np.testing.assert_array_equal(a[0], [2, 7, 5, 6])
    np.testing.assert_array_equal(a[1], [-2, -7, -5, 6])
    np.testing.assert_array_equal(b[0], [1, 2, 1, 1])
    np.testing.assert_array_equal(bank[0], a)
    empty_a, empty_b, _ = compiled_summary_chunk(
        np.array([2.0, 6.0]),
        np.ones(2),
        np.ones(2),
        np.ones(2),
        edges,
        np.array([]),
        data_order=2,
        norm_order=2,
    )
    np.testing.assert_array_equal(empty_a[:, 1:3], np.zeros((3, 2)))
    np.testing.assert_array_equal(empty_b[:, 1:3], np.zeros((3, 2)))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_bad_in_band_psd_is_signalled_for_boundary_validation(bad):
    f = np.array([2.0, 3.0, 4.0])
    s = np.array([1.0, bad, 2.0])
    outputs = compiled_summary_chunk(
        f,
        np.ones(3),
        s,
        np.ones(3),
        np.array([2.0, 4.0]),
        np.array([0.1]),
        data_order=3,
        norm_order=2,
    )
    assert all(np.any(~np.isfinite(value)) for value in outputs)
    with pytest.raises(ValueError, match="in-band PSD"):
        polynomial_moments(f, np.ones(3), s, np.ones(3), [2.0, 4.0], 3, 2)


def test_out_of_band_nonfinite_samples_cannot_poison_valid_sums():
    f = np.array([-np.inf, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf, np.nan])
    selected = np.array([False, False, True, True, True, False, False, False])
    data = np.where(selected, 1.0 + 2.0j, complex(np.nan, np.inf))
    reference = np.where(selected, 2.0 - 1.0j, complex(np.inf, np.nan))
    psd = np.where(selected, 2.0, np.nan)
    edges, anchors = np.array([2.0, 3.0, 4.0]), np.array([-0.2, 0.0, 0.2])
    actual = compiled_summary_chunk(
        f, data, psd, reference, edges, anchors, data_order=4, norm_order=6
    )
    expected = _oracle(
        f[selected],
        data[selected],
        psd[selected],
        reference[selected],
        edges,
        anchors,
        4,
        6,
    )
    for result, target in zip(actual, expected, strict=True):
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, target, rtol=2e-14, atol=2e-14)


def test_empty_frequency_chunk_has_correct_shapes_and_zero_outputs():
    empty = np.zeros(0)
    a, b, bank = compiled_summary_chunk(
        empty,
        empty,
        empty,
        empty,
        np.array([2.0, 4.0, 7.0]),
        np.array([]),
        data_order=3,
        norm_order=5,
    )
    assert a.shape == (4, 2) and b.shape == (6, 2) and bank.shape == (0, 4, 2)
    assert np.count_nonzero(a) == np.count_nonzero(b) == 0


def test_nested_jit_chunk_accumulation_preserves_detector_cuts_and_phases():
    f, data, psd, reference, edges = _random_inputs()
    # These phases model an already-projected reference's carrier/data epoch.
    reference *= np.exp(-2j * np.pi * f * 1.973)
    detector_band = (f >= 3.0) & (f <= 12.0)
    f, data, psd, reference = (
        value[detector_band] for value in (f, data, psd, reference)
    )
    anchors = np.array([-0.1, 0.0, 0.1])
    expected = _oracle(f, data, psd, reference, edges, anchors, 8, 6)
    total = jax.tree.map(jnp.zeros_like, expected)

    @jax.jit
    def accumulate(frequencies, strain, spectrum, carrier, total):
        moments = compiled_summary_chunk(
            frequencies,
            strain,
            spectrum,
            carrier,
            edges,
            anchors,
            data_order=8,
            norm_order=6,
        )
        return jax.tree.map(jnp.add, total, moments)

    for start in range(0, f.size, 31):
        chunk = [value[start : start + 31] for value in (f, data, psd, reference)]
        missing = 31 - len(chunk[0])
        chunk = [
            np.pad(value, (0, missing), constant_values=fill)
            for value, fill in zip(
                chunk, [edges[-1] + 1, np.nan, np.nan, np.nan], strict=True
            )
        ]
        total = accumulate(*chunk, total)
    for result, target in zip(total, expected, strict=True):
        np.testing.assert_allclose(result, target, rtol=3e-13, atol=3e-13)


def test_large_cancellation_uses_absolute_discrete_sum_roundoff_budget():
    count = 4096
    f = np.repeat(np.linspace(2.0, 8.0, count // 2), 2)
    rng = np.random.default_rng(911)
    amplitude = 1e8 * np.exp(1j * rng.uniform(-np.pi, np.pi, count // 2))
    data = np.empty(count, dtype=np.complex128)
    data[::2] = amplitude
    data[1::2] = -amplitude + 1e-5 * (1.0 + 2.0j)
    psd, reference = np.ones(count), np.ones(count, dtype=np.complex128)
    edges, anchors = np.array([2.0, 4.0, 6.0, 8.0]), np.array([-0.09, 0.0, 0.09])
    expected = _oracle(f, data, psd, reference, edges, anchors, 24, 16)
    actual = compiled_summary_chunk(
        f, data, psd, reference, edges, anchors, data_order=24, norm_order=16
    )
    # Relative error in an almost-cancelled sum is meaningless. Multiplication
    # and reduction reordering are checked against the native absolute scale.
    budget = 8 * np.finfo(np.float64).eps * np.sum(np.abs(data))
    assert np.max(np.abs(expected[0])) < 0.1
    for result, target in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, target, rtol=0, atol=budget)


@pytest.mark.parametrize("order", [-1, 1.5, True])
def test_invalid_static_orders_fail_before_device_execution(order):
    with pytest.raises(ValueError, match="non-negative integers"):
        compiled_summary_chunk(
            np.ones(2),
            np.ones(2),
            np.ones(2),
            np.ones(2),
            np.array([0.0, 2.0]),
            np.array([]),
            data_order=order,
            norm_order=1,
        )
