"""Residual phasor approximation and independent noisy native contractions."""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import Polynomial

from jimgw.core.single_event.heterodyne_phasor import phasor_polynomial_coefficients
from jimgw.core.single_event.xg_evaluation import make_dresser
from jimgw.core.single_event.xg_network_evaluation import BatchedXGEvaluator

jax.config.update("jax_enable_x64", True)


def test_corrected_polynomial_covers_declared_support_and_improves_wide_bin():
    half = np.array([11.26, 37.15, 72.325, 137.68])
    anchors = np.linspace(-0.2, 0.2, 21)
    a, diagnostics = phasor_polynomial_coefficients(
        half, anchors, 16, approximation="chebyshev"
    )
    assert a.shape == (17, 4) and np.isrealobj(a)
    np.testing.assert_array_equal(a[0], 1)
    assert diagnostics["covered_residual_seconds"] > 0.5 * max(np.diff(anchors))
    assert diagnostics["likelihood_accuracy_qualified"] is False
    polynomial_error, taylor_error = [], []
    for b, width in enumerate(half):
        u = (
            np.linspace(-1, 1, 4097)
            * 2
            * np.pi
            * width
            * diagnostics["covered_residual_seconds"]
        )
        p = Polynomial(a[:, b] * 1j ** np.arange(17))
        truth = np.exp(1j * u)
        np.testing.assert_array_equal(p(-u), p(u).conj())
        assert p(0) == 1
        polynomial_error.append(max(abs(p(u) - truth)))
        taylor = Polynomial([1j**m / math.factorial(m) for m in range(17)])
        taylor_error.append(max(abs(taylor(u) - truth)))
    assert polynomial_error[0] < 1e-14
    assert polynomial_error[1] < 1e-12
    assert polynomial_error[2] < 6e-9
    assert polynomial_error[3] < 1.5e-4
    assert polynomial_error[2] < taylor_error[2] / 10_000


@pytest.mark.parametrize("order", [0, 1, 3, 16])
def test_taylor_coefficients_and_small_angle_limit(order):
    a, diagnostics = phasor_polynomial_coefficients([1e-15, 2], [-0.2, 0.2], order)
    np.testing.assert_array_equal(
        a, np.tile([1 / math.factorial(m) for m in range(order + 1)], (2, 1)).T
    )
    assert diagnostics["approximation"] == "taylor"
    if order == 16:
        b, diagnostics = phasor_polynomial_coefficients(
            [1e-15], [-0.2, 0.2], 16, approximation="chebyshev"
        )
        np.testing.assert_array_equal(b[:, 0], a[:, 0])
        assert diagnostics["small_angle_taylor_bins"] == 1


@pytest.mark.parametrize(
    ("half", "anchors", "order", "mode", "message"),
    [
        ([1], [0, 0.1], 16, "unknown", "phasor_approximation"),
        ([1], [0, 0.1], 8, "chebyshev", "order 16"),
        ([1], [0], 16, "chebyshev", "two anchors"),
        ([1], [0, 0], 16, "chebyshev", "strictly increasing"),
        ([1], [0.1, 0], 16, "chebyshev", "strictly increasing"),
        ([1], [0, np.nan], 16, "chebyshev", "finite"),
        ([0], [0, 0.1], 16, "chebyshev", "positive"),
        ([np.inf], [0, 0.1], 16, "chebyshev", "finite"),
        ([[1]], [0, 0.1], 16, "chebyshev", "one-dimensional"),
    ],
)
def test_invalid_support_or_representation_is_rejected(
    half, anchors, order, mode, message
):
    with pytest.raises(ValueError, match=message):
        phasor_polynomial_coefficients(half, anchors, order, approximation=mode)


@pytest.mark.parametrize("order", [16.0, True])
def test_noninteger_order_is_rejected(order):
    with pytest.raises(TypeError, match="integer"):
        phasor_polynomial_coefficients([1], [0, 0.1], order)


def test_serial_and_batched_horner_match_noisy_native_polynomial_contractions():
    rng = np.random.default_rng(901)
    half = np.array([11.26, 72.325])
    centres = np.array([40.0, 300.0])
    anchors = np.linspace(-0.04, 0.04, 5)
    x = np.linspace(-1, 1, 101, endpoint=False)
    frequencies = centres[:, None] + half[:, None] * x
    # Actual independent samples are compressed into moments, rather than
    # inventing arbitrary moments that might not arise from a native sum.
    native = rng.normal(size=frequencies.shape) + 1j * rng.normal(
        size=frequencies.shape
    )
    native += 3 * np.exp(1j * frequencies / 11)
    ratio_coeff = (rng.normal(size=(9, 2)) + 1j * rng.normal(size=(9, 2))) / (
        1 + np.arange(9)[:, None]
    ) ** 2
    ratio = np.stack([Polynomial(ratio_coeff[:, b])(x) for b in range(2)])
    data = np.stack(
        [
            np.stack(
                [
                    np.sum(
                        native * np.exp(2j * np.pi * anchor * frequencies) * x**k,
                        axis=1,
                    )
                    for k in range(25)
                ]
            )
            for anchor in anchors
        ]
    )
    likelihood = SimpleNamespace(
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_approximation="chebyshev",
        phasor_time_anchors=anchors,
        phasor_data_moments={"D": jnp.asarray(data)},
        freq_grid_centres=jnp.asarray(centres),
        freq_grid_half_widths=jnp.asarray(half),
        summary_moments={"D": (None, jnp.ones((17, 2)))},
        _rigid_time_shift=lambda detector, parameters: parameters["dt"],
    )
    dress = make_dresser(likelihood)
    a, _ = phasor_polynomial_coefficients(half, anchors, 16, approximation="chebyshev")
    evaluator = BatchedXGEvaluator.__new__(BatchedXGEvaluator)
    evaluator.order, evaluator.phasor_order = 8, 16
    evaluator.phasor_approximation = "chebyshev"
    evaluator.phasor_coefficients = jnp.asarray(a)
    group = SimpleNamespace(
        detector_indices=np.array([0]),
        bin_indices=np.arange(2),
        data_moments=jnp.asarray(data[None]),
        centres=jnp.asarray(centres),
        half_widths=jnp.asarray(half),
        phasor_coefficients=jnp.asarray(a),
    )
    detector = SimpleNamespace(name="D")
    compiled = jax.jit(lambda dt: dress(detector, {"dt": dt})[0])
    midpoints = (anchors[:-1] + anchors[1:]) / 2
    shifts = np.r_[
        anchors,
        np.nextafter(midpoints, -np.inf),
        midpoints,
        np.nextafter(midpoints, np.inf),
    ]
    errors, taylor_errors = [], []
    for dt in shifts:
        index = np.argmin(abs(anchors - dt))
        residual = dt - anchors[index]
        values = np.asarray(compiled(dt))
        batched = evaluator._dress(group, jnp.array([index]), jnp.array([residual]))[0]
        np.testing.assert_allclose(batched, values, rtol=0, atol=1e-11)
        phase_polynomial = np.stack(
            [
                Polynomial(a[:, b] * 1j ** np.arange(17))(
                    2 * np.pi * residual * half[b] * x
                )
                for b in range(2)
            ]
        )
        phase = np.exp(
            2j * np.pi * (anchors[index] * frequencies + residual * centres[:, None])
        )
        expected = np.sum(native * ratio.conj() * phase * phase_polynomial)
        actual = np.sum(ratio_coeff.conj() * values)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        exact = np.sum(native * ratio.conj() * np.exp(2j * np.pi * dt * frequencies))
        errors.append(abs(actual - exact))
        taylor = Polynomial([1j**m / math.factorial(m) for m in range(17)])
        taylor_value = np.sum(
            native
            * ratio.conj()
            * phase
            * taylor(2 * np.pi * residual * half[:, None] * x)
        )
        taylor_errors.append(abs(taylor_value - exact))
    assert max(errors) < 1e-5
    assert max(errors) < max(taylor_errors) / 1000
    for dt in [
        np.nextafter(anchors[0], -np.inf),
        np.nextafter(anchors[-1], np.inf),
        np.nan,
    ]:
        assert np.all(np.isnan(compiled(dt)))
