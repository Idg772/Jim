"""Batched proposal algebra for an existing anchored XG network moment bank.

Channels with identical nonzero-moment support share a response contraction.
The source stays on the complete node grid, including its emission clock and
reference frequency prefix. Only exactly zero detector/bin contributions are
removed; native quadrature, interpolation and phasor orders are unchanged.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jimgw.core.single_event.heterodyne_phasor import phasor_polynomial_coefficients
from jimgw.core.single_event.xg_network_response import NetworkResponse
from jimgw.core.utils import log_i0


@dataclass(frozen=True)
class _DetectorGroup:
    detector_indices: np.ndarray
    bin_indices: np.ndarray
    node_indices: np.ndarray
    data_moments: jax.Array
    norm_moments: jax.Array
    norm_matrix: jax.Array | None
    centres: jax.Array
    half_widths: jax.Array
    response: object
    phasor_coefficients: jax.Array


class BatchedXGEvaluator:
    """Share network response work and batch the small polynomial contractions.

    ``gram`` uses a constant real Hankel matrix and small matrix/vector
    contractions. Its candidate-dependent intermediates have at most detector,
    bin, degree and polarization axes, with no degree-by-degree outer product.
    ``triangular`` retains explicit symmetric degree-pair reductions as an
    alternative for timing and numerical comparison.
    """

    def __init__(
        self,
        likelihood,
        reference_waveform,
        *,
        trim_zero_bins=True,
        reuse_opposite_arms=True,
        norm_method="gram",
    ):
        if norm_method not in {"gram", "triangular"}:
            raise ValueError("norm_method must be gram or triangular")
        lk = self.likelihood = likelihood
        if (
            lk.reference_projection != "carrier"
            or lk.phasor_time_anchors is None
            or lk.interpolation_order < 2
            or lk.phasor_moment_order < 1
            or lk.time_marginalization
            or lk.distance_marginalization
        ):
            raise ValueError("Batched XG evaluation requires anchored carrier moments")
        if not lk.detectors or not jax.config.jax_enable_x64:
            raise ValueError("Batched XG evaluation requires detectors and float64")
        self.norm_method = norm_method
        self.order = lk.interpolation_order
        self.phasor_order = lk.phasor_moment_order
        self.factorials = tuple(
            float(math.factorial(m)) for m in range(self.phasor_order + 1)
        )
        self.phasor_approximation = getattr(lk, "phasor_approximation", "taylor")
        coefficients, self.phasor_diagnostics = phasor_polynomial_coefficients(
            lk.freq_grid_half_widths,
            lk.phasor_time_anchors,
            self.phasor_order,
            approximation=self.phasor_approximation,
        )
        self.phasor_coefficients = jnp.asarray(coefficients)
        self.frequency = lk.freq_grid_node_flat
        prefix = getattr(lk, "_xg_node_frequency_prefix", None)
        frequencies = (
            self.frequency
            if prefix is None
            else jnp.concatenate((prefix, self.frequency))
        )
        reference = reference_waveform(frequencies, lk.reference_parameters)
        if prefix is not None:
            reference = jax.tree.map(lambda value: value[2:], reference)
        self.reference_carrier = reference["p"]
        self.anchors = jnp.asarray(lk.phasor_time_anchors)
        self.reference_rigid_delays = jnp.stack(
            [lk._phasor_reference_delay[detector.name] for detector in lk.detectors]
        )
        self.response = NetworkResponse(
            lk.detectors,
            self.frequency,
            lk.reference_parameters,
            reference["__tau__"],
            reuse_opposite_arms=reuse_opposite_arms,
        )
        # Host inspection is confined to the retained moments, never native data.
        banks = []
        grouped = {}
        active_counts = []
        for index, detector in enumerate(lk.detectors):
            a = np.asarray(lk.phasor_data_moments[detector.name])
            b = np.asarray(lk.summary_moments[detector.name][1])
            if a.shape != (
                len(self.anchors),
                self.order + self.phasor_order + 1,
                lk.n_bins,
            ) or b.shape != (2 * self.order + 1, lk.n_bins):
                raise ValueError("Unexpected anchored XG moment shape")
            banks.append((a, b))
            active = (
                np.any(a != 0, axis=(0, 1)) | np.any(b != 0, axis=0)
                if trim_zero_bins
                else np.ones(lk.n_bins, dtype=bool)
            )
            active_counts.append(int(active.sum()))
            if np.any(active):
                key = active.tobytes()
                grouped.setdefault(key, (np.flatnonzero(active), []))[1].append(index)
        groups = []
        descriptions = []
        degrees = np.arange(self.order + 1)
        for bins, members in grouped.values():
            detectors = np.asarray(members, dtype=np.int32)
            bins = bins.astype(np.int32)
            nodes = (degrees[:, None] * lk.n_bins + bins[None, :]).ravel()
            a = jnp.asarray(np.stack([banks[d][0][..., bins] for d in members]))
            b = jnp.asarray(np.stack([banks[d][1][:, bins].real for d in members]))
            matrix = (
                jnp.transpose(
                    b[:, degrees[:, None] + degrees[None, :], :], (0, 3, 1, 2)
                )
                if norm_method == "gram"
                else None
            )
            response = self.response.make_group(detectors, nodes)
            groups.append(
                _DetectorGroup(
                    detectors,
                    bins,
                    nodes,
                    a,
                    b,
                    matrix,
                    lk.freq_grid_centres[bins],
                    lk.freq_grid_half_widths[bins],
                    response,
                    jnp.asarray(coefficients[:, bins]),
                )
            )
            descriptions.append(
                {
                    "detector_indices": detectors.tolist(),
                    "detectors": [lk.detectors[d].name for d in members],
                    "active_bins": len(bins),
                    "skipped_bins": lk.n_bins - len(bins),
                    "nodes_per_detector": len(nodes),
                }
            )
        self.groups = tuple(groups)
        self.diagnostics = {
            "implementation": "batched-anchored-network-v1",
            "trim_zero_bins": bool(trim_zero_bins),
            "reuse_opposite_arms": bool(reuse_opposite_arms),
            "norm_method": norm_method,
            "source_nodes": int(self.frequency.size),
            "full_detector_response_nodes": int(self.frequency.size)
            * len(lk.detectors),
            "active_detector_response_nodes": sum(
                len(group.detector_indices) * len(group.node_indices)
                for group in groups
            ),
            "active_bins_by_detector": dict(
                zip((d.name for d in lk.detectors), active_counts, strict=True)
            ),
            "entirely_zero_channels": [
                d.name
                for d, count in zip(lk.detectors, active_counts, strict=True)
                if count == 0
            ],
            "groups": descriptions,
            "response": self.response.diagnostics,
            "native_summaries_unchanged": True,
            "full_clock_validity": True,
            "phasor_polynomial": self.phasor_diagnostics,
        }

    def _prepare(self, params, polarizations):
        state = self.response.prepare(params, polarizations["__tau__"])
        dt = (
            params["t_c"]
            - self.likelihood._phasor_reference_t_c
            + (state["trigger_delays"] - self.reference_rigid_delays)
        )
        # These are global checks even when every moment of a channel is zero.
        # Otherwise skipping its nodes could silently turn an invalid proposal
        # (including NaN times zero in the serial path) into a finite result.
        valid = (
            jnp.all(state["valid"])
            & jnp.all(jnp.isfinite(state["delta"]))
            & jnp.all(jnp.isfinite(polarizations["__tau__"]))
            & jnp.all(jnp.isfinite(polarizations["p"]))
            & jnp.all(jnp.isfinite(polarizations["c"]))
            & jnp.all(jnp.isfinite(self.reference_carrier))
            & jnp.all(self.reference_carrier != 0)
            & jnp.all((dt >= self.anchors[0]) & (dt <= self.anchors[-1]))
        )
        for value in state["frame"]:
            valid &= jnp.all(jnp.isfinite(value))
        indices = jnp.argmin(jnp.abs(dt[:, None] - self.anchors), axis=1)
        residual = dt - self.anchors[indices]
        return state, indices, residual, valid

    def _coefficients(self, ratio, n_bins):
        return jnp.einsum(
            "ij,...jb->...ib",
            self.likelihood._vandermonde_inverse,
            ratio.reshape(*ratio.shape[:-1], self.order + 1, n_bins),
        )

    def _dress(self, group, indices, residual):
        detectors = group.detector_indices
        a = group.data_moments[jnp.arange(len(detectors)), indices[detectors]]
        residual = residual[detectors, None]
        angle = 2 * jnp.pi * group.centres * residual
        phasor = jax.lax.complex(jnp.cos(angle), jnp.sin(angle))
        theta = 2 * jnp.pi * group.half_widths * residual
        x = -(theta[:, None, :] ** 2)

        def weighted(m):
            values = a[:, m : m + self.order + 1]
            if self.phasor_approximation == "taylor":
                return values / self.factorials[m]
            return values * group.phasor_coefficients[m]

        top = self.phasor_order // 2
        even = weighted(2 * top)
        for degree in range(top - 1, -1, -1):
            even = even * x + weighted(2 * degree)
        top = (self.phasor_order - 1) // 2
        odd = weighted(2 * top + 1)
        for degree in range(top - 1, -1, -1):
            odd = odd * x + weighted(2 * degree + 1)
        return phasor[:, None, :] * (even + 1j * theta[:, None, :] * odd)

    def _gram(self, coefficients, group):
        """Return sum C_u conj(C_v) B, without a dynamic degree outer product."""
        if self.norm_method == "gram":
            c = jnp.transpose(coefficients, (0, 3, 2, 1))
            real = jnp.einsum("dbkm,dbmu->dbku", group.norm_matrix, c.real)
            imag = jnp.einsum("dbkm,dbmu->dbku", group.norm_matrix, c.imag)
            return jnp.einsum("dbku,dbkv->uv", c, jax.lax.complex(real, -imag))
        gram = jnp.zeros((coefficients.shape[1],) * 2, dtype=jnp.complex128)
        for k in range(self.order + 1):
            ck = coefficients[:, :, k]
            gram += jnp.einsum(
                "dub,dvb,db->uv", ck, jnp.conj(ck), group.norm_moments[:, 2 * k]
            )
            for m in range(k):
                cm = coefficients[:, :, m]
                b = group.norm_moments[:, k + m]
                gram += jnp.einsum("dub,dvb,db->uv", ck, jnp.conj(cm), b)
                gram += jnp.einsum("dub,dvb,db->uv", cm, jnp.conj(ck), b)
        return gram

    def evaluate(self, params, polarizations):
        state, indices, residual, valid = self._prepare(params, polarizations)
        overlap = jnp.zeros((), dtype=jnp.complex128)
        norm = jnp.zeros(())
        for group in self.groups:
            modes = group.response(state)
            nodes = group.node_indices
            ratio = (
                modes["p"] * polarizations["p"][nodes]
                + modes["c"] * polarizations["c"][nodes]
            ) / self.reference_carrier[nodes]
            c = self._coefficients(ratio, len(group.bin_indices))
            a = self._dress(group, indices, residual)
            overlap += jnp.sum(jnp.conj(c) * a)
            norm += self._gram(c[:, None], group)[0, 0].real
        match = (
            log_i0(jnp.abs(overlap))
            if self.likelihood.phase_marginalization
            else overlap.real
        )
        return jnp.where(valid, match - 0.5 * norm, jnp.nan)

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
        state, indices, residual, valid = self._prepare(fixed, polarizations)
        overlap = jnp.zeros(2, dtype=jnp.complex128)
        gram = jnp.zeros((2, 2), dtype=jnp.complex128)
        for group in self.groups:
            modes = group.response(state)
            nodes = group.node_indices
            carrier = polarizations["p"][nodes] / self.reference_carrier[nodes]
            ratio = jnp.stack((modes["p"], modes["c"]), axis=1) * carrier
            c = self._coefficients(ratio, len(group.bin_indices))
            a = self._dress(group, indices, residual)
            overlap += jnp.sum(jnp.conj(c) * a[:, None], axis=(0, 2, 3))
            gram += self._gram(c, group)
        # Match the fixed summary's Hermitian orientation explicitly. Diagonals
        # are real norms; any floating association residue is not a new phase.
        gram = jnp.array(
            [[gram[0, 0].real, gram[0, 1]], [jnp.conj(gram[0, 1]), gram[1, 1].real]]
        )
        return {
            "overlap": jnp.where(valid, overlap, jnp.nan),
            "gram": jnp.where(valid, gram, jnp.nan),
            "fixed": {
                key: jnp.asarray(value)
                for key, value in p.items()
                if key not in {"psi", "iota", "d_L"}
            },
        }
