"""Factored evaluation of an existing anchored-carrier XG moment bank.

Native data, projected references, binning and moment construction are retained.
Only proposal evaluation changes: precomputed waveform powers, residual delay
phases, factored finite arms and even/odd evaluation of every phasor moment.
Taylor retains the original polynomial; optional corrected Chebyshev coefficients
change its approximation on the same support without increasing moment degree.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from jimgw.core.constants import C_SI
from jimgw.core.single_event.detector import GroundBased2G
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.heterodyne_phasor import phasor_polynomial_coefficients
from jimgw.core.single_event.time_dependent_response import (
    earth_orbital_curvature_delay,
    emission_gmst,
)
from jimgw.core.single_event.xg_waveform import (
    PrefixedBaselineWaveform,
    XGBasisCachedWaveform,
    supports_source,
)
from jimgw.core.utils import log_i0

XG_EVALUATION_REVISION = "factored-carrier-network-v2"


def residual_response(detector, params, tau):
    """Full dynamic antenna and delay relative to the trigger geocentre delay."""
    ra, dec, psi = params["ra"], params["dec"], params["psi"]
    gmst = emission_gmst(params["gmst"], params["t_c"], tau)
    delay = -detector._source_projection(ra, dec, psi, gmst, detector.vertex)[2] / C_SI
    delay -= detector.delay_from_geocenter(ra, dec, params["gmst"])
    if detector.orbital_motion_response:
        offset = params["t_c"] - tau
        orbit = earth_orbital_curvature_delay(
            ra,
            dec,
            offset,
            detector.orbital_acceleration_over_c,
            detector.orbital_jerk_over_c,
        )
        lo, hi = detector.orbital_validity_s
        valid = (
            (params["trigger_time"] == detector.orbital_reference_time)
            & (offset >= lo)
            & (offset <= hi)
        )
        delay += jnp.where(valid, orbit, jnp.nan)
    return gmst, delay


def make_factored_arm(frequency, arm_length_m):
    """Precompute frequency-only constants; combine the candidate phases."""
    x = np.asarray(frequency) * arm_length_m / C_SI
    angle = np.pi * x
    c1 = jnp.asarray(0.5 * np.exp(-1j * angle))
    c3 = jnp.asarray(0.5 * np.exp(-3j * angle))
    x, angle, frequency = map(jnp.asarray, (x, angle, frequency))

    def arm(mu, delay_difference):
        phase_angle = angle * mu - 2 * jnp.pi * frequency * delay_difference
        phase = jax.lax.complex(jnp.cos(phase_angle), jnp.sin(phase_angle))
        return phase * (c1 * jnp.sinc(x * (1 - mu)) + c3 * jnp.sinc(x * (1 + mu)))

    return arm


def make_dresser(likelihood):
    """Evaluate the selected anchored polynomial with even/odd Horner sums."""
    order = likelihood.interpolation_order
    phasor_order = likelihood.phasor_moment_order
    if order < 0 or phasor_order < 0:
        raise ValueError("Polynomial orders must be nonnegative")
    if likelihood.phasor_time_anchors is None:
        raise ValueError("Phasor dressing requires time anchors")
    anchors = jnp.asarray(likelihood.phasor_time_anchors)
    factorials = tuple(float(math.factorial(m)) for m in range(phasor_order + 1))
    approximation = getattr(likelihood, "phasor_approximation", "taylor")
    coefficients, diagnostics = phasor_polynomial_coefficients(
        likelihood.freq_grid_half_widths,
        likelihood.phasor_time_anchors,
        phasor_order,
        approximation=approximation,
    )
    coefficients = jnp.asarray(coefficients)

    def weighted(a, m):
        values = a[m : m + order + 1]
        if approximation == "taylor":
            return values / factorials[m]
        return values * coefficients[m]

    def dress(detector, parameters):
        dt = likelihood._rigid_time_shift(detector, parameters)
        index = jnp.argmin(jnp.abs(anchors - dt))
        a = likelihood.phasor_data_moments[detector.name][index]
        a = jnp.where((dt >= anchors[0]) & (dt <= anchors[-1]), a, jnp.nan)
        residual = dt - anchors[index]
        angle = 2 * jnp.pi * likelihood.freq_grid_centres * residual
        phasor = jax.lax.complex(jnp.cos(angle), jnp.sin(angle))
        theta = 2 * jnp.pi * likelihood.freq_grid_half_widths * residual

        x = -(theta**2)
        even_top = phasor_order // 2
        even = weighted(a, 2 * even_top)
        for degree in range(even_top - 1, -1, -1):
            even = even * x + weighted(a, 2 * degree)

        odd_top = (phasor_order - 1) // 2
        odd = jnp.zeros_like(even)
        if odd_top >= 0:
            odd = weighted(a, 2 * odd_top + 1)
            for degree in range(odd_top - 1, -1, -1):
                odd = odd * x + weighted(a, 2 * degree + 1)
        value = even + 1j * theta * odd
        return phasor * value, likelihood.summary_moments[detector.name][1].real

    dress.phasor_diagnostics = diagnostics
    return dress


class FastXGEvaluator:
    """Evaluate direct proposals and conditional polarization summaries."""

    def __init__(self, likelihood, reference_waveform):
        self.likelihood = likelihood
        self.frequency = likelihood.freq_grid_node_flat
        prefix = getattr(likelihood, "_xg_node_frequency_prefix", None)
        frequencies = (
            self.frequency
            if prefix is None
            else jnp.concatenate((prefix, self.frequency))
        )
        reference = reference_waveform(frequencies, likelihood.reference_parameters)
        if prefix is not None:
            reference = jax.tree.map(lambda value: value[2:], reference)
        self.reference_carrier = reference["p"]
        self.reference_delays = {
            detector.name: residual_response(
                detector, likelihood.reference_parameters, reference["__tau__"]
            )[1]
            for detector in likelihood.detectors
        }
        self.arms = {
            detector.name: make_factored_arm(self.frequency, detector.arm_length_m)
            for detector in likelihood.detectors
        }
        self.dress = make_dresser(likelihood)
        self.phasor_diagnostics = self.dress.phasor_diagnostics

    def response_modes(self, detector, params, tau):
        gmst, delay = residual_response(detector, params, tau)
        delta = delay - self.reference_delays[detector.name]
        xarm, yarm = detector.arms
        xm, xn, xmu = detector._source_projection(
            params["ra"], params["dec"], params["psi"], gmst, xarm
        )
        ym, yn, ymu = detector._source_projection(
            params["ra"], params["dec"], params["psi"], gmst, yarm
        )
        arm = self.arms[detector.name]
        tx, ty = arm(xmu, delta), arm(ymu, delta)
        return {
            "p": 0.5 * (xm**2 - xn**2) * tx - 0.5 * (ym**2 - yn**2) * ty,
            "c": xm * xn * tx - ym * yn * ty,
        }

    def coefficients(self, ratio):
        lk = self.likelihood
        return lk._vandermonde_inverse @ ratio.reshape(
            lk.interpolation_order + 1, lk.n_bins
        )

    def symmetric_norm(self, coefficients, moments):
        real, imag = coefficients.real, coefficients.imag
        value = jnp.zeros(())
        for k in range(self.likelihood.interpolation_order + 1):
            value += jnp.sum((real[k] ** 2 + imag[k] ** 2) * moments[2 * k])
            for m in range(k):
                value += 2 * jnp.sum(
                    (real[k] * real[m] + imag[k] * imag[m]) * moments[k + m]
                )
        return value

    def evaluate(self, params, polarizations):
        lk = self.likelihood
        overlap = jnp.zeros((), dtype=jnp.complex128)
        norm = jnp.zeros(())
        for detector in lk.detectors:
            response = self.response_modes(detector, params, polarizations["__tau__"])
            ratio = (
                response["p"] * polarizations["p"] + response["c"] * polarizations["c"]
            ) / self.reference_carrier
            coefficients = self.coefficients(ratio)
            a, b = self.dress(detector, params)
            overlap += jnp.sum(jnp.conj(coefficients) * a)
            norm += self.symmetric_norm(coefficients, b)
        match = log_i0(jnp.abs(overlap)) if lk.phase_marginalization else overlap.real
        return match - 0.5 * norm

    def build_extrinsic_summary(self, params, waveform_cache=None):
        lk = self.likelihood
        p = lk._prepare_parameters(params)
        fixed = {**p, "psi": 0.0, "iota": 0.0, "d_L": 1.0}
        if waveform_cache is None:
            polarizations = lk.waveform(self.frequency, fixed)
        else:
            polarizations = lk._waveform_sky_from_cache(
                self.frequency, waveform_cache["nodes"], fixed
            )
        overlap = jnp.zeros(2, dtype=jnp.complex128)
        gram = jnp.zeros((2, 2), dtype=jnp.complex128)
        order = lk.interpolation_order
        for detector in lk.detectors:
            response = self.response_modes(detector, fixed, polarizations["__tau__"])
            ratio = polarizations["p"] / self.reference_carrier
            c = jnp.stack(
                [self.coefficients(response[mode] * ratio) for mode in ("p", "c")]
            )
            a, b = self.dress(detector, fixed)
            overlap += jnp.sum(jnp.conj(c) * a, axis=(1, 2))
            diagonals = [self.symmetric_norm(c[mode], b) for mode in range(2)]
            degrees = jnp.arange(order + 1)
            off = jnp.einsum(
                "kb,mb,kmb->",
                c[0],
                jnp.conj(c[1]),
                b[degrees[:, None] + degrees[None, :]],
            )
            gram += jnp.array([[diagonals[0], off], [jnp.conj(off), diagonals[1]]])
        return {
            "overlap": overlap,
            "gram": gram,
            "fixed": {
                key: jnp.asarray(value)
                for key, value in p.items()
                if key not in {"psi", "iota", "d_L"}
            },
        }


def configure_xg_evaluation(likelihood, reference_waveform, mode="auto"):
    """Install the supported evaluator after reference and summary construction.

    This is also the reinstallation hook after changing a likelihood's node
    grid or moment bank. Baseline waveform and reference identities are retained
    so a copied likelihood never accidentally reuses closures from the old grid.
    Unsupported models and custom responses keep the supplied evaluator.
    """
    if mode not in {"auto", "baseline"}:
        raise ValueError("xg_evaluation_mode must be auto or baseline")
    lk = likelihood
    waveform = getattr(lk, "_baseline_waveform", lk.waveform)
    if getattr(lk, "_xg_node_frequency_prefix", None) is not None and not all(
        (
            type(model) is DominantModeTimeCachedWaveform
            and supports_source(model.source)
        )
        or getattr(model, "frequency_grid_independent", False) is True
        for model in (waveform, reference_waveform)
    ):
        raise ValueError(
            "node_frequency_prefix requires a supported stock waveform/reference "
            "or explicit frequency_grid_independent contracts"
        )
    lk._baseline_waveform = waveform
    lk._reference_waveform = reference_waveform
    lk.waveform = waveform
    lk._xg_fast_evaluator = None
    lk.xg_evaluation_mode = mode
    reason = None
    if mode == "baseline":
        reason = "explicit baseline"
    elif not jax.config.jax_enable_x64:
        reason = "64-bit precision is required"
    elif (
        lk.reference_projection != "carrier"
        or lk.interpolation_order < 2
        or lk.phasor_moment_order == 0
        or lk.phasor_time_anchors is None
        or lk.time_marginalization
        or lk.distance_marginalization
    ):
        reason = (
            "requires anchored carrier polynomial with no time/distance marginalization"
        )
    elif not all(
        type(model) is DominantModeTimeCachedWaveform
        and model.mode == 2
        and "_time_to_coalescence" not in vars(model)
        and supports_source(model.source)
        for model in (waveform, reference_waveform)
    ):
        reason = "unsupported waveform, reference or emission-clock adapter"
    elif not lk.detectors or not all(
        type(detector) is GroundBased2G
        and detector.time_dependent_response
        and detector.finite_arm_response
        and {mode.name for mode in detector.polarization_mode} == {"p", "c"}
        and "fd_response" not in vars(detector)
        and "frequency_dependent_antenna_pattern" not in vars(detector)
        and "_source_projection" not in vars(detector)
        and "delay_from_geocenter" not in vars(detector)
        for detector in lk.detectors
    ):
        reason = "requires standard time-dependent finite-arm detector responses"
    network_fallback_reason = None
    if reason is None:
        from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD

        standard_shift = (
            "_rigid_time_shift" not in vars(lk)
            and type(lk)._rigid_time_shift
            is HeterodynedTransientLikelihoodFD._rigid_time_shift
        )
        if len(lk.detectors) > 1 and standard_shift:
            from jimgw.core.single_event.xg_network_evaluation import BatchedXGEvaluator

            evaluator = BatchedXGEvaluator(lk, reference_waveform)
        else:
            evaluator = FastXGEvaluator(lk, reference_waveform)
            network_fallback_reason = (
                "single detector" if standard_shift else "custom rigid-time-shift hook"
            )
        lk.waveform = XGBasisCachedWaveform(
            waveform.source,
            frequency_prefix=getattr(lk, "_xg_node_frequency_prefix", None),
        )
        lk._xg_fast_evaluator = evaluator
    elif (
        getattr(lk, "_xg_node_frequency_prefix", None) is not None
        and type(waveform) is DominantModeTimeCachedWaveform
        and supports_source(waveform.source)
    ):
        lk.waveform = PrefixedBaselineWaveform(waveform, lk._xg_node_frequency_prefix)
    lk.evaluation_diagnostics = {
        "requested_mode": mode,
        "implementation": XG_EVALUATION_REVISION if reason is None else "baseline",
        "fallback_reason": reason,
        "waveform_nodes": int(lk.freq_grid_node_flat.size)
        if lk.interpolation_order > 1
        else None,
        "detector_channels": len(lk.detectors),
        "node_frequency_prefix": (
            np.asarray(lk._xg_node_frequency_prefix).tolist()
            if getattr(lk, "_xg_node_frequency_prefix", None) is not None
            else None
        ),
        "native_summaries_unchanged": True,
        "scalar_summary": "factored-two-mode" if reason is None else "generic-two-mode",
        "network_kernel": (
            getattr(
                lk._xg_fast_evaluator, "diagnostics", {"implementation": "per-channel"}
            )
            if reason is None
            else None
        ),
        "network_fallback_reason": network_fallback_reason,
        "phasor_approximation": getattr(lk, "phasor_approximation", "taylor"),
        "phasor_polynomial": getattr(lk._xg_fast_evaluator, "phasor_diagnostics", None),
    }
    return lk
