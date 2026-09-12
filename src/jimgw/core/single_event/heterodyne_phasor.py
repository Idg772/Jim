"""Static polynomial coefficients for anchored heterodyne phase integration.

The real coefficients ``a[m, b]`` represent ``sum a[m, b] * (1j*u)**m``.
Only construction uses Bessel functions. The polynomial retains the existing
moment degree and approximates the residual phase, not the waveform norm.
"""

import hashlib
import math

import numpy as np
from scipy.special import jv

CHEBYSHEV_PHASOR_REVISION = "corrected-chebyshev-bessel-v1"


def phasor_polynomial_coefficients(
    half_widths, time_anchors, order, *, approximation="taylor"
):
    """Return an ``(order+1, bins)`` real array and JSON-safe diagnostics.

    The Chebyshev polynomial is the degree-16 Jacobi--Anger series on each
    bin's full residual-phase support, with its constant corrected to one.
    Anchor selection must still reject shifts outside the anchor endpoints.
    Diagnostics describe support and representation, not likelihood accuracy.
    """
    if approximation not in {"taylor", "chebyshev"}:
        raise ValueError("phasor_approximation must be taylor or chebyshev")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)):
        raise TypeError("phasor order must be a nonnegative integer")
    if order < 0:
        raise ValueError("phasor order must be a nonnegative integer")
    if approximation == "chebyshev" and order != 16:
        raise ValueError("chebyshev phasor approximation requires order 16")
    widths = np.asarray(half_widths, dtype=np.float64)
    anchors = np.asarray(time_anchors, dtype=np.float64)
    if (
        widths.ndim != 1
        or not widths.size
        or not np.all(np.isfinite(widths))
        or np.any(widths <= 0)
    ):
        raise ValueError(
            "phasor half-widths must be finite, positive and one-dimensional"
        )
    if (
        anchors.ndim != 1
        or not anchors.size
        or not np.all(np.isfinite(anchors))
        or np.any(np.diff(anchors) <= 0)
    ):
        raise ValueError(
            "phasor anchors must be finite, nonempty and strictly increasing"
        )
    if approximation == "chebyshev" and anchors.size < 2:
        raise ValueError("chebyshev phasor approximation requires at least two anchors")

    residual = 0.5 * float(np.max(np.diff(anchors))) if anchors.size > 1 else 0.0
    # Subtracting a rounded selected anchor can add a few ulps. The slack is
    # for polynomial coverage only; it never enlarges accepted time support.
    slack = 8 * np.finfo(np.float64).eps * max(1.0, float(np.max(abs(anchors))))
    covered_residual = residual + slack
    theta = np.nextafter(2 * np.pi * widths * covered_residual, np.inf)
    if not np.all(np.isfinite(theta)):
        raise ValueError("phasor polynomial support must be finite")

    coefficients = np.broadcast_to(
        np.asarray([1 / math.factorial(m) for m in range(order + 1)])[:, None],
        (order + 1, widths.size),
    ).copy()
    small = theta < np.sqrt(np.finfo(np.float64).eps)
    if approximation == "chebyshev" and np.any(~small):
        # Transform T_m(x) to powers of x once; every bin shares this matrix.
        transform = np.zeros((order + 1, order + 1))
        for m in range(order + 1):
            basis = np.polynomial.chebyshev.cheb2poly(np.eye(order + 1)[m])
            transform[: basis.size, m] = basis
        degrees = np.arange(order + 1)
        phases = (1j**degrees)[:, None]
        angles = theta[~small]
        chebyshev = 2 * phases * jv(degrees[:, None], angles[None, :])
        chebyshev[0] *= 0.5
        powers = transform @ chebyshev
        coefficients[:, ~small] = (
            powers / phases / angles[None, :] ** degrees[:, None]
        ).real
        # This preserves exact anchor values and conjugate symmetry. The
        # corrected polynomial must be qualified as a whole by the selector.
        coefficients[0] = 1.0
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("nonfinite phasor polynomial coefficients")
    diagnostics = {
        "approximation": approximation,
        "revision": CHEBYSHEV_PHASOR_REVISION
        if approximation == "chebyshev"
        else "taylor-v1",
        "order": int(order),
        "bins": int(widths.size),
        "maximum_residual_seconds": residual,
        "support_slack_seconds": slack,
        "covered_residual_seconds": covered_residual,
        "minimum_theta_radians": float(np.min(theta)),
        "maximum_theta_radians": float(np.max(theta)),
        "small_angle_taylor_bins": int(np.count_nonzero(small))
        if approximation == "chebyshev"
        else 0,
        "coefficients_sha256": hashlib.sha256(coefficients.tobytes()).hexdigest(),
        "native_moment_degree_unchanged": True,
        "likelihood_accuracy_qualified": False,
    }
    return coefficients, diagnostics
