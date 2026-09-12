"""Conditional scalar likelihoods for nonprecessing dominant-mode signals.

These summaries belong to one accepted intrinsic/sky/time state. They preserve
the detector's full response and are suitable for an extrinsic-only segment;
they do not replace the prior or sampling-coordinate Jacobian.
"""

import jax.numpy as jnp

from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.utils import log_i0


def build_extrinsic_summary(likelihood, params, waveform_cache=None):
    """Contract the current polynomial likelihood into two polarization modes.

    Supply an accepted waveform cache to avoid rebuilding the carrier. The
    caller must supply that cache from the same intrinsic state as ``params``.
    Unsupported precession/higher modes and distance/time marginalization
    are rejected. Returned arrays form an ordinary JAX pytree.
    """
    if (
        likelihood.interpolation_order < 2
        or likelihood.time_marginalization
        or likelihood.distance_marginalization
    ):
        raise ValueError(
            "extrinsic summaries require a polynomial likelihood without time/distance marginalization"
        )
    # Reuse the existing adapter's model-contract validation. No clock or
    # waveform is evaluated by this check.
    if not isinstance(likelihood.waveform, DominantModeTimeCachedWaveform):
        DominantModeTimeCachedWaveform(likelihood.waveform)
    p = likelihood._prepare_parameters(params)
    fixed = {**p, "psi": 0.0, "iota": 0.0, "d_L": 1.0}
    frequencies = likelihood.freq_grid_node_flat
    if waveform_cache is None:
        pols = likelihood.waveform(frequencies, fixed)
    else:
        if not {"iota", "d_L"} <= likelihood.waveform_cacheable_parameter_names:
            raise ValueError(
                "accepted waveform cache must reconstruct inclination and distance"
            )
        pols = likelihood._waveform_sky_from_cache(
            frequencies, waveform_cache["nodes"], fixed
        )
    h0 = pols["p"]
    z = jnp.zeros(2, dtype=jnp.complex128)
    gram = jnp.zeros((2, 2), dtype=jnp.complex128)
    order = likelihood.interpolation_order
    for detector in likelihood.detectors:
        a, b = likelihood.summary_moments[detector.name]
        phasor = jnp.ones_like(likelihood.freq_grid_nodes, dtype=jnp.complex128)
        if likelihood.phasor_moment_order:
            dt = likelihood._rigid_time_shift(detector, fixed)
            residual = dt
            if likelihood.phasor_time_anchors is not None:
                anchors = jnp.asarray(likelihood.phasor_time_anchors)
                index = jnp.argmin(abs(anchors - dt))
                a = likelihood.phasor_data_moments[detector.name][index]
                a = jnp.where((dt >= anchors[0]) & (dt <= anchors[-1]), a, jnp.nan)
                residual = dt - anchors[index]
            phasor = jnp.exp(2j * jnp.pi * likelihood.freq_grid_nodes * dt)
            theta = 2j * jnp.pi * likelihood.freq_grid_half_widths * residual
            terms = [jnp.ones_like(theta)]
            approximation = getattr(likelihood, "phasor_approximation", "taylor")
            for m in range(1, likelihood.phasor_moment_order + 1):
                terms.append(
                    terms[-1] * theta / m
                    if approximation == "taylor"
                    else terms[-1] * theta
                )
            if approximation == "chebyshev":
                terms = [
                    term * likelihood._phasor_polynomial_coefficients[m]
                    for m, term in enumerate(terms)
                ]
            a = jnp.exp(
                2j * jnp.pi * likelihood.freq_grid_centres * residual
            ) * jnp.stack(
                [
                    sum(term * a[k + m] for m, term in enumerate(terms))
                    for k in range(order + 1)
                ]
            )
        coefficients = []
        for mode in ("p", "c"):
            basis = {**pols, "p": jnp.zeros_like(h0), "c": jnp.zeros_like(h0)}
            basis[mode] = h0
            projected = detector.fd_response(frequencies, basis, fixed)
            ratio = (
                projected.reshape(order + 1, likelihood.n_bins)
                / likelihood.waveform_node_ref[detector.name]
            )
            coefficients.append(likelihood._vandermonde_inverse @ (ratio * phasor))
        c = jnp.stack(coefficients)
        z += jnp.sum(jnp.conj(c) * a, axis=(1, 2))
        degrees = jnp.arange(order + 1)
        norm_matrix = b[degrees[:, None] + degrees[None, :]]
        gram += jnp.einsum("ukb,vmb,kmb->uv", c, jnp.conj(c), norm_matrix)
    return {
        "overlap": z,
        "gram": gram,
        "fixed": {
            key: jnp.asarray(value)
            for key, value in p.items()
            if key not in {"psi", "iota", "d_L"}
        },
    }


def evaluate_extrinsic_summary(likelihood, params, summary):
    """Evaluate scalar contractions; stale conditional summaries return NaN."""
    p = likelihood._prepare_parameters(params)
    valid = jnp.asarray(True)
    for key, value in summary["fixed"].items():
        valid &= jnp.all(jnp.asarray(p[key]) == value)
    valid &= jnp.isfinite(p["d_L"]) & (p["d_L"] > 0)
    ci = jnp.cos(p["iota"])
    plus, cross = (1 + ci**2) / (2 * p["d_L"]), -1j * ci / p["d_L"]
    cp, sp = jnp.cos(2 * p["psi"]), jnp.sin(2 * p["psi"])
    weights = jnp.array([plus * cp - cross * sp, plus * sp + cross * cp])
    z = jnp.sum(jnp.conj(weights) * summary["overlap"])
    norm = jnp.einsum("u,uv,v->", weights, summary["gram"], jnp.conj(weights)).real
    result = (
        log_i0(abs(z)) if likelihood.phase_marginalization else z.real
    ) - 0.5 * norm
    return jnp.where(valid, result, jnp.nan)
