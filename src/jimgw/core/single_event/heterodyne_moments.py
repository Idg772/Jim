"""Bounded-memory construction of polynomial heterodyne summaries.

All inputs are samples of discrete overlaps: quadrature weights (including
4/T) belong to the caller. Data and norm degrees are deliberately independent.
"""

from __future__ import annotations

import numpy as np


def validate_time_anchors(anchors):
    """Canonical time-shift support, in seconds; never silently sort it."""
    if anchors is None:
        return None
    x = np.asarray(anchors, dtype=float)
    if (
        x.ndim != 1
        or len(x) < 2
        or not np.all(np.isfinite(x))
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError(
            "phasor_time_anchors must contain at least two finite, increasing times"
        )
    return tuple(float(v) for v in x)


def polynomial_moments(
    frequencies, data, psd, reference, edges, data_order, norm_order
):
    """Return unnormalised A and B moments on normalized bin coordinates.

    A complex data stream is accumulated exactly as a discrete sum; it is
    never interpolated. Each sample, including the final endpoint, belongs
    to one bin. Frequencies outside the supplied edges are ignored.
    """
    edges = np.asarray(edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("edges must be finite and strictly increasing")
    for order in (data_order, norm_order):
        if (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or order < 0
        ):
            raise ValueError("moment orders must be non-negative integers")
    f, d, s, h = map(np.asarray, (frequencies, data, psd, reference))
    if f.ndim != 1 or any(x.shape != f.shape for x in (d, s, h)):
        raise ValueError(
            "frequency, data, PSD and reference arrays must be equal-sized vectors"
        )
    index = np.searchsorted(edges, f, side="right") - 1
    index = np.where(f == edges[-1], len(edges) - 2, index)
    valid = (index >= 0) & (index < len(edges) - 1)
    index, f, d, s, h = (x[valid] for x in (index, f, d, s, h))
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("in-band PSD samples must be positive and finite")
    u = (2 * f - edges[index] - edges[index + 1]) / (edges[index + 1] - edges[index])
    # Use recurrence instead of reevaluating u**k, and bincount instead of
    # indexed scalar additions. Real and imaginary reductions stay separate.
    a = np.empty((data_order + 1, len(edges) - 1), dtype=np.complex128)
    b = np.empty((norm_order + 1, len(edges) - 1), dtype=np.float64)
    da, hb = d * h.conj() / s, (h.real**2 + h.imag**2) / s
    for k in range(max(data_order, norm_order) + 1):
        if k <= data_order:
            a[k] = np.bincount(
                index, weights=da.real, minlength=a.shape[1]
            ) + 1j * np.bincount(index, weights=da.imag, minlength=a.shape[1])
            da = da * u
        if k <= norm_order:
            b[k] = np.bincount(index, weights=hb, minlength=b.shape[1])
            hb = hb * u
    return a, b
