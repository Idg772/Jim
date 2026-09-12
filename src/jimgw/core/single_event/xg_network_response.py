"""Shared emission frame and finite-arm projections for an XG network."""

import jax
import jax.numpy as jnp
import numpy as np

from jimgw.core.constants import C_SI
from jimgw.core.single_event.time_dependent_response import emission_gmst


def _project(frame, vectors):
    """Project fixed vectors using one candidate's already evaluated frame."""
    cp, sp, ct, st, cs, ss = frame
    a0, a1, a2 = (vectors[:, index, None] for index in range(3))
    u = a0 * cp * ct + a1 * ct * sp - a2 * st
    v = -a0 * sp + a1 * cp
    omega = a0 * st * cp + a1 * st * sp + a2 * ct
    return -u * ss - v * cs, -u * cs + v * ss, omega


def _frame(ra, dec, psi, gmst):
    phi = ra - jnp.mod(gmst, 2 * jnp.pi)
    theta = jnp.pi / 2 - dec
    return (
        jnp.cos(phi),
        jnp.sin(phi),
        jnp.cos(theta),
        jnp.sin(theta),
        jnp.cos(psi),
        jnp.sin(psi),
    )


class NetworkResponse:
    """Share source algebra while retaining every detector's physical response.

    ``prepare`` always checks the full source clock against every detector's
    orbital contract. A caller that skips zero-moment detector/bin groups must
    still require ``jnp.all(state['valid'])`` in its final result.

    Opposite-arm reuse is restricted to equal nominal lengths and unit vectors
    differing from exact opposition by at most eight float64 epsilons. This
    admits the roundoff from the connected ET preset's local-angle conversion;
    other directions keep independent projections and transfer functions.
    """

    def __init__(
        self,
        detectors,
        frequency,
        reference_parameters,
        reference_tau,
        *,
        reuse_opposite_arms=True,
    ):
        # Imported here so the legacy response remains the reference gauge and
        # the evaluator module can import this implementation without a cycle.
        from jimgw.core.single_event.xg_evaluation import residual_response

        self.detectors = tuple(detectors)
        self.frequency = jnp.asarray(frequency)
        self.reuse_opposite_arms = bool(reuse_opposite_arms)
        self.vertices = jnp.stack([detector.vertex for detector in detectors])
        self.arms = np.asarray([detector.arms for detector in detectors])
        self.lengths = np.asarray([detector.arm_length_m for detector in detectors])
        self.reference_delays = jnp.stack(
            [
                residual_response(detector, reference_parameters, reference_tau)[1]
                for detector in detectors
            ]
        )
        self._orbital_contracts = []
        self._orbital_indices = []
        for detector in detectors:
            if not detector.orbital_motion_response:
                self._orbital_indices.append(None)
                continue
            contract = (
                tuple(np.asarray(detector.orbital_acceleration_over_c).tolist()),
                tuple(np.asarray(detector.orbital_jerk_over_c).tolist()),
                float(detector.orbital_reference_time),
                tuple(detector.orbital_validity_s),
            )
            if contract not in self._orbital_contracts:
                self._orbital_contracts.append(contract)
            self._orbital_indices.append(self._orbital_contracts.index(contract))
        self.diagnostics = {
            "shared_emission_frame": True,
            "orbital_contract_groups": len(self._orbital_contracts),
            "opposite_arm_reuse": self.reuse_opposite_arms,
            "opposite_arm_tolerance": 8 * np.finfo(np.float64).eps,
            "groups": [],
        }

    def prepare(self, params, tau):
        """Prepare full-grid geometry and full-clock validity before trimming."""
        offset = params["t_c"] - tau
        clock_valid = jnp.all(jnp.isfinite(offset))
        gmst = emission_gmst(params["gmst"], params["t_c"], tau)
        frame = _frame(params["ra"], params["dec"], params["psi"], gmst)
        _, _, vertex_projection = _project(frame, self.vertices)
        trigger_frame = _frame(
            params["ra"], params["dec"], params["psi"], params["gmst"]
        )
        _, _, trigger_projection = _project(trigger_frame, self.vertices)
        trigger_delays = -trigger_projection[:, 0] / C_SI
        residual = -vertex_projection / C_SI - trigger_delays[:, None]

        orbital_delays = []
        orbital_valid = []
        if self._orbital_contracts:
            cos_dec = jnp.cos(params["dec"])
            source = jnp.stack(
                (
                    cos_dec * jnp.cos(params["ra"]),
                    cos_dec * jnp.sin(params["ra"]),
                    jnp.sin(params["dec"]),
                )
            )
            offset2, offset3 = offset**2, offset**3
            for acceleration, jerk, reference_time, (lo, hi) in self._orbital_contracts:
                displacement = (
                    0.5 * jnp.asarray(acceleration)[:, None] * offset2
                    + (1.0 / 6.0) * jnp.asarray(jerk)[:, None] * offset3
                )
                orbital_delays.append(-jnp.sum(source[:, None] * displacement, axis=0))
                orbital_valid.append(
                    (params["trigger_time"] == reference_time)
                    & jnp.all((offset >= lo) & (offset <= hi))
                )
        valid = []
        delays = []
        for index, contract_index in enumerate(self._orbital_indices):
            delay = residual[index]
            if contract_index is None:
                valid.append(clock_valid)
            else:
                delay = delay + orbital_delays[contract_index]
                valid.append(clock_valid & orbital_valid[contract_index])
            delays.append(delay)
        return {
            "frame": frame,
            "trigger_delays": trigger_delays,
            "delta": jnp.stack(delays) - self.reference_delays,
            "valid": jnp.stack(valid),
        }

    def make_group(self, detector_indices, node_indices):
        """Build a response callable for fixed detector and active-node subsets."""
        detector_indices = np.asarray(detector_indices, dtype=np.int32)
        node_indices = np.asarray(node_indices, dtype=np.int32)
        vectors = self.arms[detector_indices].reshape(-1, 3)
        lengths = np.repeat(self.lengths[detector_indices], 2)
        unique, unique_lengths, mapping, signs = [], [], [], []
        tolerance = self.diagnostics["opposite_arm_tolerance"]
        pair_errors = []
        for vector, length in zip(vectors, lengths, strict=True):
            match = None
            if (
                self.reuse_opposite_arms
                and abs(np.linalg.norm(vector) - 1) <= tolerance
            ):
                for index, (candidate, candidate_length) in enumerate(
                    zip(unique, unique_lengths, strict=True)
                ):
                    error = np.max(np.abs(vector + candidate))
                    if (
                        length == candidate_length
                        and abs(np.linalg.norm(candidate) - 1) <= tolerance
                        and error <= tolerance
                    ):
                        match = index
                        pair_errors.append(float(error))
                        break
            if match is None:
                match = len(unique)
                unique.append(vector)
                unique_lengths.append(length)
                signs.append(1.0)
            else:
                signs.append(-1.0)
            mapping.append(match)

        group_diagnostics = {
            "detector_indices": detector_indices.tolist(),
            "nodes": int(node_indices.size),
            "directed_arms": len(vectors),
            "independent_arm_pairs": len(unique),
            "reused_opposite_arms": len(pair_errors),
            "maximum_opposite_pair_error": max(pair_errors, default=0.0),
        }
        self.diagnostics["groups"].append(group_diagnostics)
        unique = jnp.asarray(np.asarray(unique).reshape(-1, 3))
        mapping = jnp.asarray(mapping)
        signs = jnp.asarray(signs)[:, None]
        detector_indices, node_indices = map(
            jnp.asarray, (detector_indices, node_indices)
        )
        frequency = self.frequency[node_indices]
        # Fixed frequency/length constants are built once per active-node group.
        x = jnp.asarray(np.asarray(unique_lengths)[:, None]) * frequency / C_SI
        angle = jnp.pi * x
        c1, c3 = 0.5 * jnp.exp(-1j * angle), 0.5 * jnp.exp(-3j * angle)
        n_detectors = len(vectors) // 2

        def response(state):
            frame = tuple(
                value[node_indices] if value.ndim else value for value in state["frame"]
            )
            dot_m, dot_n, mu = _project(frame, unique)
            sinc_minus, sinc_plus = jnp.sinc(x * (1 - mu)), jnp.sinc(x * (1 + mu))
            minus = jnp.where(signs > 0, sinc_minus[mapping], sinc_plus[mapping])
            plus = jnp.where(signs > 0, sinc_plus[mapping], sinc_minus[mapping])
            delta = state["delta"][detector_indices][:, node_indices]
            delta = jnp.repeat(delta, 2, axis=0)
            phase_angle = (
                angle[mapping] * (signs * mu[mapping]) - 2 * jnp.pi * frequency * delta
            )
            phase = jax.lax.complex(jnp.cos(phase_angle), jnp.sin(phase_angle))
            transfer = phase * (c1[mapping] * minus + c3[mapping] * plus)
            pp = 0.5 * (dot_m**2 - dot_n**2)
            pc = dot_m * dot_n
            pp = (pp[mapping] * transfer).reshape(n_detectors, 2, -1)
            pc = (pc[mapping] * transfer).reshape(n_detectors, 2, -1)
            valid = state["valid"][detector_indices, None]
            return {
                "p": jnp.where(valid, pp[:, 0] - pp[:, 1], jnp.nan),
                "c": jnp.where(valid, pc[:, 0] - pc[:, 1], jnp.nan),
            }

        response.diagnostics = group_diagnostics
        return response
