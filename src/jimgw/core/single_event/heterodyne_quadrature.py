"""Adaptive construction of smooth, zero-noise heterodyne moments.

This integrates deterministic signal/reference products, never interpolated
noise. It is an opt-in numerical building block: embedded quadrature errors
are estimates and do not constitute likelihood or posterior qualification.
"""

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise

import numpy as np
from scipy.linalg import eigh_tridiagonal

from jimgw.core.single_event.heterodyne_moments import validate_time_anchors


@dataclass(frozen=True)
class QuadratureMoments:
    data: np.ndarray
    norm: np.ndarray
    anchors: tuple[float, ...]
    evaluations: int
    accepted_panels: int
    estimated_error: float


def _native_left_index(frequency, spacing):
    """Index of the first native sample >= frequency, with float comparisons."""
    index = int(np.floor(frequency / spacing))
    if index * spacing < frequency:
        return index + 1
    if (index - 1) * spacing >= frequency:
        return index - 1
    return index


def _native_right_index(frequency, spacing):
    index = _native_left_index(frequency, spacing)
    return index if index * spacing <= frequency else index - 1


@lru_cache(maxsize=2048)
def discrete_gauss_rule(count, size):
    """Gaussian rule for equally weighted discrete points on [-1,1].

    The symmetric Hahn (alpha=beta=0) recurrence gives the off-diagonal
    Jacobi coefficients. See DLMF 18.22.2-3. Weights sum to one; callers
    multiply by count and Fourier spacing. Polynomial sums through degree
    2*size-1 are exact in exact arithmetic. Small grids are summed directly.
    """
    if count < 1 or size < 1:
        raise ValueError("quadrature count and size must be positive")
    if count <= size:
        x, w = np.zeros(size), np.zeros(size)
        x[:count] = np.linspace(-1.0, 1.0, count) if count > 1 else 0.0
        w[:count] = 1 / count
        return x, w
    k = np.arange(1, size, dtype=float)
    beta = k / (count - 1) * np.sqrt((count - k) * (count + k) / (4 * k * k - 1))
    x, v = eigh_tridiagonal(np.zeros(size), beta)
    return x, v[0] ** 2


def zero_noise_moments(
    signal,
    reference,
    psd,
    edges,
    order,
    phasor_order,
    *,
    anchors=None,
    atol=1e-10,
    rtol=1e-10,
    max_evaluations=2_000_000,
    max_depth=20,
    max_panel_width=None,
    native_delta_f=None,
    native_frequency_bounds=None,
    breakpoints=(),
):
    """Integrate A(anchor), B with embedded 16/32-point Gauss quadrature.

    Callbacks map an arbitrary frequency vector to complex detector strain
    (signal/reference) or a positive real PSD. They must not depend on grid
    spacing. Returned moments include the continuous-overlap factor 4.
    ``atol`` budgets moment error, not log-likelihood error. An independent
    dense-grid likelihood screen is required before using the result.
    ``max_panel_width`` can impose a conservative phase-resolution bound.
    With ``native_delta_f``, use discrete Gaussian quadrature of the native
    Fourier sum (origin zero) instead of replacing it by a continuous integral.
    Bins with few native points are summed exactly, including endpoints.
    ``native_frequency_bounds`` restricts sums to one detector's inclusive
    band while retaining the network's common bin-coordinate normalization.
    Known nonsmooth locations, such as PSD interpolation knots, can be supplied
    as ``breakpoints`` to avoid rediscovering them by repeated bisection.
    """
    edges = np.asarray(edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("edges must be finite and strictly increasing")
    if any(
        isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0
        for x in (order, phasor_order, max_depth)
    ):
        raise ValueError("orders and max_depth must be non-negative integers")
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol <= 0 or rtol < 0:
        raise ValueError(
            "quadrature tolerances must be finite; atol positive, rtol non-negative"
        )
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 48
    ):
        raise ValueError("max_evaluations must be an integer of at least 48")
    if max_panel_width is not None and (
        not np.isfinite(max_panel_width) or max_panel_width <= 0
    ):
        raise ValueError("max_panel_width must be finite and positive")
    if native_delta_f is not None and (
        not np.isfinite(native_delta_f) or native_delta_f <= 0
    ):
        raise ValueError("native_delta_f must be finite and positive")
    time_anchors = validate_time_anchors(anchors) if anchors is not None else (0.0,)
    native_limits = None
    if native_frequency_bounds is not None:
        limits = np.asarray(native_frequency_bounds, dtype=float)
        if (
            native_delta_f is None
            or limits.shape != (2,)
            or np.any(~np.isfinite(limits))
            or limits[1] < limits[0]
        ):
            raise ValueError(
                "native frequency bounds require finite ordered endpoints and native_delta_f"
            )
        native_limits = (
            _native_left_index(limits[0], native_delta_f),
            _native_right_index(limits[1], native_delta_f),
        )
    nb = len(edges) - 1
    data_degree, norm_degree = order + phasor_order, 2 * order
    data = np.zeros((len(time_anchors), data_degree + 1, nb), complex)
    norm = np.zeros((norm_degree + 1, nb))
    pending = []
    for b in range(nb):
        if native_delta_f is not None:
            first = _native_left_index(edges[b], native_delta_f)
            last = (
                _native_right_index(edges[b + 1], native_delta_f)
                if b == nb - 1
                else _native_left_index(edges[b + 1], native_delta_f) - 1
            )
            if native_limits is not None:
                first, last = max(first, native_limits[0]), min(last, native_limits[1])
            if last < first:
                continue
            count = (
                1
                if max_panel_width is None
                else min(
                    last - first + 1,
                    int(np.ceil((last - first + 1) * native_delta_f / max_panel_width)),
                )
            )
            if (len(pending) + count) * 48 > max_evaluations:
                raise RuntimeError(
                    "quadrature evaluation budget exceeded by initial panels"
                )
            cuts = np.linspace(first, last + 1, count + 1).astype(int)
            pending.extend(
                (b, lo * native_delta_f, (hi - 1) * native_delta_f, 0)
                for lo, hi in pairwise(cuts)
            )
            continue
        count = (
            1
            if max_panel_width is None
            else int(np.ceil((edges[b + 1] - edges[b]) / max_panel_width))
        )
        if (len(pending) + count) * 48 > max_evaluations:
            raise RuntimeError(
                "quadrature evaluation budget exceeded by initial panels"
            )
        bounds = np.linspace(edges[b], edges[b + 1], count + 1)
        pending.extend((b, lo, hi, 0) for lo, hi in pairwise(bounds))
    knots = np.asarray(breakpoints, dtype=float)
    if knots.ndim != 1 or np.any(~np.isfinite(knots)):
        raise ValueError("breakpoints must be a finite vector")
    if len(knots):
        split_panels = []
        knots = np.unique(knots)
        for b, lo, hi, depth in pending:
            local = knots[(knots > lo) & (knots < hi)]
            if native_delta_f is None:
                cuts = np.concatenate(([lo], local, [hi]))
                split_panels.extend(
                    (b, left, right, depth) for left, right in pairwise(cuts)
                )
            else:
                first, last = round(lo / native_delta_f), round(hi / native_delta_f)
                inner = np.ceil(local / native_delta_f).astype(np.int64)
                cuts = np.unique(np.concatenate(([first], inner, [last + 1])))
                split_panels.extend(
                    (b, left * native_delta_f, (right - 1) * native_delta_f, depth)
                    for left, right in pairwise(cuts)
                )
            if len(split_panels) * 48 > max_evaluations:
                raise RuntimeError(
                    "quadrature evaluation budget exceeded by breakpoints"
                )
        pending = split_panels
    rules = [np.polynomial.legendre.leggauss(n) for n in (16, 32)]
    evaluations = accepted = 0
    estimated_error = 0.0
    while pending:
        batch, pending = pending[:128], pending[128:]
        if evaluations + 48 * len(batch) > max_evaluations:
            raise RuntimeError(
                "quadrature evaluation budget exhausted before convergence"
            )
        bins = np.array([p[0] for p in batch])
        lo, hi = (np.array([p[i] for p in batch]) for i in (1, 2))
        estimates = []
        for x, w in rules:
            native_counts = None
            if native_delta_f is not None:
                native_counts = np.rint((hi - lo) / native_delta_f).astype(int) + 1
                discrete_rules = [
                    discrete_gauss_rule(int(n), len(x)) for n in native_counts
                ]
                x = np.stack([r[0] for r in discrete_rules])
                w = np.stack([r[1] for r in discrete_rules])
            f = (lo[:, None] + hi[:, None]) / 2 + (hi - lo)[:, None] / 2 * x
            shape = f.shape
            h, d, s = (
                np.asarray(fn(f.ravel())).reshape(shape)
                for fn in (reference, signal, psd)
            )
            if (
                np.any(~np.isfinite(h))
                or np.any(~np.isfinite(d))
                or np.any(~np.isfinite(s))
                or np.any(s <= 0)
            ):
                raise ValueError(
                    "quadrature callbacks returned non-finite strain or non-positive PSD"
                )
            u = (2 * f - edges[bins, None] - edges[bins + 1, None]) / (
                edges[bins + 1, None] - edges[bins, None]
            )
            weight = (
                2 * (hi - lo)[:, None] * w
                if native_counts is None
                else 4 * native_delta_f * native_counts[:, None] * w
            )
            da = (d * h.conj() / s * weight)[None, :, :] * np.exp(
                2j * np.pi * np.asarray(time_anchors)[:, None, None] * f
            )
            hb = abs(h) ** 2 / s * weight
            aa, bb = [], []
            for k in range(max(data_degree, norm_degree) + 1):
                if k <= data_degree:
                    aa.append(da.sum(axis=-1))
                    da = da * u
                if k <= norm_degree:
                    bb.append(hb.sum(axis=-1))
                    hb = hb * u
            estimates.append((np.stack(aa, axis=1), np.stack(bb)))
        evaluations += 48 * len(batch)
        a0, b0 = estimates[0]
        a1, b1 = estimates[1]
        measure = hi - lo if native_delta_f is None else hi - lo + native_delta_f
        allowance = atol * measure / (edges[-1] - edges[0])
        good = np.all(abs(a1 - a0) <= allowance + rtol * abs(a1), axis=(0, 1)) & np.all(
            abs(b1 - b0) <= allowance + rtol * abs(b1), axis=0
        )
        for i, (b, left, right, depth) in enumerate(batch):
            if good[i]:
                data[:, :, b] += a1[:, :, i]
                norm[:, b] += b1[:, i]
                accepted += 1
                estimated_error += max(
                    np.max(abs(a1[:, :, i] - a0[:, :, i])),
                    np.max(abs(b1[:, i] - b0[:, i])),
                )
            else:
                if depth >= max_depth:
                    raise RuntimeError(
                        "quadrature depth budget exhausted before convergence"
                    )
                if native_delta_f is None:
                    mid = (left + right) / 2
                    pending.extend(
                        [(b, left, mid, depth + 1), (b, mid, right, depth + 1)]
                    )
                else:
                    n = round((right - left) / native_delta_f) + 1
                    mid = left + (n // 2) * native_delta_f
                    pending.extend(
                        [
                            (b, left, mid - native_delta_f, depth + 1),
                            (b, mid, right, depth + 1),
                        ]
                    )
    return QuadratureMoments(
        data, norm, time_anchors, evaluations, accepted, float(estimated_error)
    )


class ZeroNoiseSummaryBuilder:
    """Construct native-sum moments directly from a declared noiseless signal.

    ``signal_parameters`` are physical injection parameters (including phase).
    ``psds`` maps detector names to original PSD knots and *power* values.
    This contract must only be used for data generated by this same signal,
    waveform and detector response. The CLI enforces zero-noise injection mode.
    No native noise samples are interpolated or discarded by this builder.
    """

    method = "discrete-zero-noise-gauss-v1"

    def __init__(self, signal_parameters, psds, **options):
        self.signal_parameters = dict(signal_parameters)
        self.psds = {
            name: (
                np.array(f, dtype=float, copy=True),
                np.array(s, dtype=float, copy=True),
            )
            for name, (f, s) in psds.items()
        }
        self.options = dict(options)

    @property
    def contract_sha256(self):
        payload = {
            "method": self.method,
            "signal": {k: float(v) for k, v in self.signal_parameters.items()},
            "options": self.options,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        for name, (f, s) in sorted(self.psds.items()):
            digest.update(name.encode() + b"\0")
            digest.update(np.asarray([len(f)], dtype="<i8").tobytes())
            digest.update(np.asarray(f, dtype="<f8").tobytes())
            digest.update(np.asarray(s, dtype="<f8").tobytes())
        return digest.hexdigest()

    def build(self, likelihood, detector, reference_waveform):
        import jax
        import jax.numpy as jnp

        frequencies = detector.sliced_frequencies
        prefix = frequencies[:2]

        # Ripple's cutoff convention reads the first two frequencies. Keep
        # their native spacing even though the integration nodes are irregular.
        def projected(model, parameters, *, reference=False):
            @jax.jit
            def function(f):
                pols = model(jnp.concatenate((prefix, f)), parameters)
                pols = jax.tree.map(lambda x: x[2:], pols)
                if reference:
                    return likelihood._project_reference(detector, f, pols)
                return detector.fd_response(f, pols, parameters)

            return lambda f: np.asarray(function(jnp.asarray(f)))

        parameters = {
            **self.signal_parameters,
            "trigger_time": likelihood.trigger_time,
            "gmst": likelihood.gmst,
        }
        f_psd, s_psd = self.psds[detector.name]
        if (
            f_psd.ndim != 1
            or s_psd.shape != f_psd.shape
            or len(f_psd) < 2
            or np.any(~np.isfinite(f_psd))
            or np.any(np.diff(f_psd) <= 0)
            or f_psd[0] > float(frequencies[0])
            or f_psd[-1] < float(frequencies[-1])
        ):
            raise ValueError("PSD knots must be ordered and cover the detector band")
        return zero_noise_moments(
            projected(likelihood.waveform, parameters),
            projected(
                reference_waveform, likelihood.reference_parameters, reference=True
            ),
            lambda f: np.interp(f, f_psd, s_psd),
            np.asarray(likelihood.freq_grid_edges),
            likelihood.interpolation_order,
            likelihood.phasor_moment_order,
            anchors=likelihood.phasor_time_anchors,
            native_delta_f=1 / float(detector.duration),
            native_frequency_bounds=(float(frequencies[0]), float(frequencies[-1])),
            breakpoints=f_psd,
            **self.options,
        )
