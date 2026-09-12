"""Device construction of discrete polynomial heterodyne summaries.

This baseline uses segmented reductions. It changes only summation order;
neither native samples nor the projected reference are approximated. The
caller owns frequency/anchor metadata validation and the final 4*df factor.
"""

from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("data_order", "norm_order"))
def compiled_summary_chunk(
    frequencies,
    data,
    psd,
    reference,
    edges,
    anchors,
    *,
    data_order,
    norm_order,
):
    """Return unnormalized ``(A0, B, Aanchors)`` for one native chunk.

    Shapes are ``(data_order+1, bins)``, ``(norm_order+1, bins)``, and
    ``(anchors, data_order+1, bins)``. ``Aanchors[a]`` includes the factor
    ``exp(+2j*pi*f*anchors[a])``; B is independent of the anchors. Empty
    anchors are supported. The final edge belongs to the final bin.

    Inputs outside the edges contribute zero, including nonfinite data,
    reference, and PSD there. Nonpositive or nonfinite *in-band* PSD values
    produce nonfinite summaries, which the construction boundary must reject.
    Edges must already be finite and strictly increasing, and anchors finite.
    The reference must already contain every desired carrier/epoch phase.

    The function can be nested inside a larger JIT containing the waveform
    and detector projection. Polynomial powers are advanced once per degree;
    scratch storage is O((anchors+1)*samples), with no anchor/degree/sample
    tensor. Segmented reductions may use atomics on accelerators; this is a
    correctness baseline, not a claim of optimal accelerator throughput.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("native heterodyne summaries require JAX 64-bit precision")
    for order in (data_order, norm_order):
        if isinstance(order, bool) or not isinstance(order, Integral) or order < 0:
            raise ValueError("moment orders must be non-negative integers")
    f = jnp.asarray(frequencies, dtype=jnp.float64)
    d = jnp.asarray(data, dtype=jnp.complex128)
    s = jnp.asarray(psd, dtype=jnp.float64)
    h = jnp.asarray(reference, dtype=jnp.complex128)
    e = jnp.asarray(edges, dtype=jnp.float64)
    times = jnp.asarray(anchors, dtype=jnp.float64)
    if f.ndim != 1 or any(value.shape != f.shape for value in (d, s, h)):
        raise ValueError(
            "frequency, data, PSD and reference must be equal-sized vectors"
        )
    if e.ndim != 1 or e.size < 2:
        raise ValueError("edges must be a vector containing at least two values")
    if times.ndim != 1:
        raise ValueError("anchors must be a vector")

    n_bins = e.size - 1
    index = jnp.searchsorted(e, f, side="right") - 1
    index = jnp.where(f == e[-1], n_bins - 1, index)
    valid = (index >= 0) & (index < n_bins) & jnp.isfinite(f)
    safe_index = jnp.clip(index, 0, n_bins - 1)
    left, right = e[safe_index], e[safe_index + 1]
    safe_f = jnp.where(valid, f, left)
    u = jnp.where(valid, (2 * safe_f - left - right) / (right - left), 0.0)
    d = jnp.where(valid, d, 0.0j)
    h = jnp.where(valid, h, 0.0j)
    valid_psd = jnp.isfinite(s) & (s > 0.0)
    s = jnp.where(valid, jnp.where(valid_psd, s, jnp.nan), 1.0)
    product = d * jnp.conj(h) / s
    norm = (h.real * h.real + h.imag * h.imag) / s
    angle = (2.0 * jnp.pi) * times[:, None] * safe_f[None, :]
    phasors = jax.lax.complex(jnp.cos(angle), jnp.sin(angle))
    products = jnp.concatenate((product[None, :], phasors * product[None, :]))

    # Invalid samples are zeros and can safely share an in-range scatter index.
    # No sorted-index promise is made: standalone chunks may be unsorted.
    def reduce_degree(power, _):
        a = jax.ops.segment_sum(
            (products * power[None, :]).T,
            safe_index,
            num_segments=n_bins,
        ).T
        b = jax.ops.segment_sum(norm * power, safe_index, num_segments=n_bins)
        return power * u, (a, b)

    _, (all_a, all_b) = jax.lax.scan(
        reduce_degree,
        jnp.ones_like(f),
        xs=None,
        length=max(data_order, norm_order) + 1,
    )
    a0 = all_a[: data_order + 1, 0, :]
    anchored = jnp.transpose(all_a[: data_order + 1, 1:, :], (1, 0, 2))
    return a0, all_b[: norm_order + 1], anchored
