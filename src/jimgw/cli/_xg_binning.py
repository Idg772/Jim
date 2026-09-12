"""Fast, empirical pre-sampling grid selection for the XG network runner.

One native moment bank is constructed by the caller. Candidate grids only
translate that bank; noisy strain is never sampled or interpolated. All exact
candidate Lobatto nodes and independent reference check nodes share one sparse
waveform bank, evaluated in fixed-size batches to bound compilation and memory.
"""

from __future__ import annotations

import hashlib
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import i0e
from scipy.stats import qmc

from jimgw.cli._transforms import to_likelihood_space
from jimgw.cli._xg_timing import network_callback_weights
from jimgw.core.single_event.heterodyne_rebin import coarsen_moments, rebin_likelihood
from jimgw.core.single_event.heterodyne_selection import (
    legendre_lobatto_nodes,
    select_frozen_grid,
)
from jimgw.core.single_event.xg_evaluation import (
    XG_EVALUATION_REVISION,
    residual_response,
)


def _unit_to_prior(spec, values):
    if spec.type == "uniform":
        return spec.min + values * (spec.max - spec.min)
    if spec.type == "power_law":
        exponent = spec.alpha + 1
        if exponent == 0:
            return spec.min * (spec.max / spec.min) ** values
        return (
            spec.min**exponent + values * (spec.max**exponent - spec.min**exponent)
        ) ** (1 / exponent)
    if spec.type == "sine":
        return np.arccos(1 - 2 * values)
    if spec.type == "cosine":
        return np.arcsin(2 * values - 1)
    raise ValueError(
        f"automatic XG bin selection requires bounded supported priors; got {spec.type}"
    )


def parameter_banks(cfg, detectors, settings):
    """Prior-derived boundary/coupled cases plus disjoint scrambled Sobol banks."""
    if cfg.data.type != "injection":
        raise ValueError(
            "network automatic bin selection currently requires an injection reference"
        )
    reference = dict(cfg.data.injection_parameters)
    names = list(cfg.prior.root)
    specs = [cfg.prior.root[name] for name in names]
    if not all(name in reference for name in names):
        raise ValueError(
            "bin selection needs physical reference values for every prior parameter"
        )

    def physical(unit):
        points = [dict(reference) for _ in range(len(unit))]
        for column, (name, spec) in enumerate(zip(names, specs, strict=True)):
            for point, value in zip(
                points, _unit_to_prior(spec, unit[:, column]), strict=True
            ):
                point[name] = float(value)
        return points

    # Keep the reference and one-coordinate endpoints; the coupled cases catch
    # interactions including maximum total mass, time and loud amplitudes.
    corners = [reference]
    for name, spec in zip(names, specs, strict=True):
        for endpoint in _unit_to_prior(spec, np.asarray([0.0, 1.0])):
            corners.append({**reference, name: float(endpoint)})
    coupled = np.stack(
        [
            np.zeros(len(names)),
            np.ones(len(names)),
            np.arange(len(names)) % 2,
            1 - np.arange(len(names)) % 2,
        ]
    )
    corners.extend(physical(coupled))

    def sobol(count, seed):
        # Draw a power-of-two bank without scipy's non-power-of-two warning.
        unit = qmc.Sobol(len(names), scramble=True, seed=seed).random_base2(
            int(np.ceil(np.log2(count)))
        )[:count]
        return physical(unit)

    training = corners + sobol(settings.training_points, settings.seed)
    verification = [dict(reference)] + sobol(
        settings.verification_points, settings.seed + 1
    )

    def prepare(points):
        converted = []
        for point in points:
            value = to_likelihood_space(
                point,
                waveform_f_ref=cfg.waveform.f_ref,
                trigger_time=cfg.data.trigger_time,
                ifos=detectors,
                time_frame=cfg.sampling.time_frame,
            )
            if cfg.likelihood.phase_marginalization:
                value["phase_c"] = 0.0
            converted.append(value)
        return converted

    return prepare(training), prepare(verification)


def _bank_hash(points):
    serial = [{key: float(value) for key, value in p.items()} for p in points]
    return hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()


def export_native_moment_bank(fine, path, cfg):
    """Persist the small native moment bank for reproducible allocation trials."""
    from pathlib import Path

    started = time.perf_counter()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"native moment export already exists: {path}")
    metadata = {
        "schema": "xg-native-allocation-bank-v1",
        "n_bins": fine.n_bins,
        "interpolation_order": fine.interpolation_order,
        "phasor_moment_order": fine.phasor_moment_order,
        "phasor_approximation": getattr(fine, "phasor_approximation", "taylor"),
        "phase_marginalization": fine.phase_marginalization,
        "detectors": [d.name for d in fine.detectors],
        "reference_parameters": {
            k: float(v) for k, v in fine.reference_parameters.items()
        },
        "config": cfg.model_dump(mode="json"),
        "analysis_contract_sha256": cfg.xg_analysis_contract_sha256(),
        "input_files_sha256": cfg.xg_input_files_sha256(),
        "bin_edges_sha256": fine.bin_edges_sha256,
        "scope": "native noisy polynomial moments; no strain arrays; not an accuracy qualification",
    }
    settings = cfg.likelihood.heterodyne.bin_selection
    if settings is not None:
        training, verification = parameter_banks(cfg, fine.detectors, settings)
        metadata["parameter_banks"] = {
            name: [
                {k: float(v) for k, v in fine._prepare_parameters(p).items()}
                for p in points
            ]
            for name, points in (("training", training), ("verification", verification))
        }
        metadata["parameter_bank_hashes"] = {
            name: _bank_hash(points)
            for name, points in metadata["parameter_banks"].items()
        }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "edges": np.asarray(fine.freq_grid_edges),
        "anchors": np.asarray(fine.phasor_time_anchors),
        "prefix": np.asarray(
            getattr(fine, "_xg_node_frequency_prefix", fine.freq_grid_node_flat[:2])
        ),
    }
    for detector in fine.detectors:
        name = detector.name
        arrays[f"{name}_a"] = np.asarray(fine.summary_moments[name][0])
        arrays[f"{name}_b"] = np.asarray(fine.summary_moments[name][1])
        arrays[f"{name}_anchored_a"] = np.asarray(fine.phasor_data_moments[name])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "n_bins": fine.n_bins,
        "wall_seconds": time.perf_counter() - started,
    }


class SparseRatioBank:
    """Single compilation for bounded parameter/frequency blocks, no native reads."""

    def __init__(self, fine, training, verification, settings):
        self.fine = fine
        self.settings = settings
        self.points = {
            "training": [fine._prepare_parameters(p) for p in training],
            "verification": [fine._prepare_parameters(p) for p in verification],
        }
        self.values = {}
        self.frequencies = None
        self.calls = 0
        self.wall_seconds = 0.0
        self.waveform = fine.waveform
        self.reference_waveform = fine._reference_waveform
        self.prefix = jnp.asarray(
            getattr(fine, "_xg_node_frequency_prefix", fine.freq_grid_node_flat[:2])
        )
        reference = fine.reference_parameters

        def block(frequency, params):
            # The reference and proposal keep the same explicit cutoff-spacing
            # convention even though the shared node union has different order.
            full = jnp.concatenate((self.prefix, frequency))
            ref = jax.tree.map(
                lambda v: v[2:], self.reference_waveform(full, reference)
            )

            def single(p):
                pols = jax.tree.map(lambda v: v[2:], self.waveform(full, p))
                results = []
                for detector in fine.detectors:
                    gmst, delay = residual_response(detector, p, pols["__tau__"])
                    _, reference_delay = residual_response(
                        detector, reference, ref["__tau__"]
                    )
                    antenna = detector.frequency_dependent_antenna_pattern(
                        p["ra"], p["dec"], p["psi"], gmst, frequency
                    )
                    ratio = (antenna["p"] * pols["p"] + antenna["c"] * pols["c"]) / ref[
                        "p"
                    ]
                    results.append(
                        ratio
                        * jnp.exp(-2j * jnp.pi * frequency * (delay - reference_delay))
                    )
                return jnp.stack(results)

            return jax.vmap(single)(params)

        self._block = jax.jit(block)

    def __call__(self, frequencies, name):
        frequencies = np.asarray(frequencies, dtype=np.float64)
        if self.frequencies is not None and not np.array_equal(
            frequencies, self.frequencies
        ):
            raise ValueError(
                "all candidate ratios must share one frozen frequency bank"
            )
        self.frequencies = frequencies
        if name in self.values:
            return self.values[name]
        required = (
            sum(len(p) for p in self.points.values())
            * len(self.fine.detectors)
            * len(frequencies)
            * 16
        )
        if required > self.settings.max_bank_bytes:
            raise ValueError(
                f"sparse ratio bank needs {required} bytes, above max_bank_bytes="
                f"{self.settings.max_bank_bytes}; reduce the bounded candidate/panel settings"
            )
        started = time.perf_counter()
        points = self.points[name]
        batch, width = (
            self.settings.parameter_batch_size,
            self.settings.frequency_chunk_size,
        )
        result = np.empty(
            (len(points), len(self.fine.detectors), len(frequencies)),
            dtype=np.complex128,
        )
        for begin in range(0, len(points), batch):
            subset = points[begin : begin + batch]
            count = len(subset)
            subset = subset + [subset[-1]] * (batch - count)
            parameters = {
                key: jnp.asarray([p[key] for p in subset]) for key in subset[0]
            }
            for offset in range(0, len(frequencies), width):
                part = frequencies[offset : offset + width]
                padded = np.pad(part, (0, width - len(part)), mode="edge")
                output = np.asarray(self._block(jnp.asarray(padded), parameters))
                self.calls += 1
                result[begin : begin + count, :, offset : offset + len(part)] = output[
                    :count, :, : len(part)
                ]
        if not np.all(np.isfinite(result)):
            raise ValueError("nonfinite waveform ratio in pre-sampling bin selection")
        self.values[name] = result
        self.wall_seconds += time.perf_counter() - started
        return result


class MomentErrorOracle:
    """Score candidates against independent fine-grid polynomial reconstruction.

    Native moments include the actual noise realization. A fine Lobatto fit is
    compared with a fit at disjoint Gauss nodes before that fine reconstruction
    is used as the error reference. This is a finite-bank convergence check;
    the runner additionally checks the selected evaluator against its streaming
    native reference before entering nested sampling.
    """

    def __init__(self, fine, bank, tolerance):
        self.fine, self.bank, self.tolerance = fine, bank, tolerance
        self.edges = np.asarray(fine.freq_grid_edges)
        self.order = fine.interpolation_order
        self.inverse = np.linalg.inv(
            np.polynomial.polynomial.polyvander(
                legendre_lobatto_nodes(self.order), self.order
            )
        )
        self.references = {}
        self.reference_errors = {}
        self.moments = {}
        self.shifts = {}

    def _moments(self, edges):
        key = np.asarray(edges).tobytes()
        if key not in self.moments:
            self.moments[key] = [
                (
                    coarsen_moments(
                        self.edges, edges, self.fine.phasor_data_moments[d.name]
                    ),
                    coarsen_moments(
                        self.edges, edges, self.fine.summary_moments[d.name][1]
                    ),
                )
                for d in self.fine.detectors
            ]
        return self.moments[key]

    def score(self, edges, coefficients, name):
        """Coefficients have shape cases x detectors x bins x degree."""
        centres = (edges[:-1] + edges[1:]) / 2
        half = np.diff(edges) / 2
        z, norm = self.interval_statistics(
            centres, half, coefficients, self._moments(edges), name
        )
        return self.log_likelihood(z.sum(axis=-1), norm.sum(axis=-1))

    def log_likelihood(self, z, norm):
        if self.fine.phase_marginalization:
            amplitude = abs(z)
            log_overlap = np.log(i0e(amplitude)) + amplitude
        else:
            log_overlap = z.real
        result = log_overlap - 0.5 * norm
        if not np.all(np.isfinite(result)):
            raise ValueError("nonfinite likelihood in bin-selection moment contraction")
        return result

    def interval_statistics(self, centres, half, coefficients, moments, name):
        """Return signed per-case, per-interval overlap and norm contributions.

        Intervals need not form a partition. Native moment arrays are supplied
        by the catalogue cache; no native input or candidate JIT is accessed.
        """
        points = self.bank.points[name]
        total_z = np.zeros((len(points), len(half)), dtype=np.complex128)
        total_norm = np.zeros((len(points), len(half)))
        anchors = np.asarray(self.fine.phasor_time_anchors)
        degrees = np.arange(self.order + 1)
        approximation = getattr(self.fine, "phasor_approximation", "taylor")
        phase_coefficients = None
        if approximation != "taylor":
            from jimgw.core.single_event.heterodyne_phasor import (
                phasor_polynomial_coefficients,
            )

            phase_coefficients, _ = phasor_polynomial_coefficients(
                half,
                anchors,
                self.fine.phasor_moment_order,
                approximation=approximation,
            )
        for dindex, (detector, (stored_a, b)) in enumerate(
            zip(self.fine.detectors, moments, strict=True)
        ):
            # Only scalar geometry is evaluated here. No waveform or native
            # array is touched during the candidate search.
            shift_key = (name, detector.name)
            if shift_key not in self.shifts:
                self.shifts[shift_key] = np.asarray(
                    [float(self.fine._rigid_time_shift(detector, p)) for p in points]
                )
            shifts = self.shifts[shift_key]
            if np.any(shifts < anchors[0]) or np.any(shifts > anchors[-1]):
                raise ValueError(
                    "selection parameter bank exceeds declared time-anchor support"
                )
            indices = np.argmin(abs(shifts[:, None] - anchors), axis=1)
            residual = shifts - anchors[indices]
            a = stored_a[indices]
            theta = 2j * np.pi * residual[:, None] * half[None, :]
            term = np.ones_like(theta)
            dressed = np.zeros(
                (len(points), self.order + 1, len(half)), dtype=np.complex128
            )
            for m in range(self.fine.phasor_moment_order + 1):
                if m:
                    term *= theta / m if approximation == "taylor" else theta
                factor = (
                    term if phase_coefficients is None else term * phase_coefficients[m]
                )
                dressed += factor[:, None, :] * a[:, m : m + self.order + 1, :]
            dressed *= np.exp(2j * np.pi * residual[:, None] * centres[None, :])[
                :, None, :
            ]
            c = np.swapaxes(coefficients[:, dindex], -1, -2)
            total_z += np.einsum("ckb,ckb->cb", c.conj(), dressed, optimize=True)
            gram = b[degrees[:, None] + degrees[None, :]]
            total_norm += np.einsum(
                "ckb,cmb,kmb->cb", c, c.conj(), gram, optimize=True
            ).real
        if not np.all(np.isfinite(total_z)) or not np.all(np.isfinite(total_norm)):
            raise ValueError("nonfinite interval likelihood statistics")
        return total_z, total_norm

    def __call__(self, plan, node_ratios, name):
        if name not in self.references:
            ratios = self.bank.values[name]
            lobatto = ratios[..., plan.reference_node_indices]
            fine_coefficients = np.einsum("ij,...bj->...bi", self.inverse, lobatto)
            fine_logl = self.score(self.edges, fine_coefficients, name)
            checks, _ = np.polynomial.legendre.leggauss(2 * self.order + 7)
            check_values = ratios[..., plan.check_indices].reshape(
                len(ratios), len(self.fine.detectors), self.fine.n_bins, len(checks)
            )
            check_inverse = np.linalg.pinv(
                np.polynomial.polynomial.polyvander(checks, self.order)
            )
            check_coefficients = np.einsum(
                "ij,...bj->...bi", check_inverse, check_values
            )
            check_logl = self.score(self.edges, check_coefficients, name)
            difference = (fine_logl - fine_logl[0]) - (check_logl - check_logl[0])
            error = float(np.max(abs(difference)))
            self.reference_errors[name] = error
            if error > self.tolerance / 4:
                raise RuntimeError(
                    f"fine bin reference failed {name} convergence: {error:.6g} nats "
                    f"> {self.tolerance / 4:.6g}; increase reference_bins before sampling"
                )
            self.references[name] = check_logl
        coefficients = np.einsum("ij,...bj->...bi", self.inverse, node_ratios)
        candidate = self.score(plan.edges, coefficients, name)
        reference = self.references[name]
        return (candidate - candidate[0]) - (reference - reference[0])


class CandidateTimer:
    """Time unpadded production callbacks only after a grid passes accuracy.

    One executable per passing node shape contains rebuild, accepted-cache
    summary preparation and scalar hit branches. Inputs/moments are unchanged
    between synchronized timing samples. Native data are never revisited.
    """

    def __init__(self, cfg, fine, waveform, bank, settings):
        self.cfg, self.fine, self.waveform = cfg, fine, waveform
        self.bank, self.settings = bank, settings
        self.records = {}
        self.clones = {}
        self.compile_seconds = 0.0
        self.weights = network_callback_weights(cfg, fine)

    def clone(self, plan):
        from jimgw.core.single_event.likelihood import (
            _XG_QUALIFICATION_PLAN_AUTHORITY,
            _QualificationXGPlan,
        )

        key = self.plan_key(plan)
        if key not in self.clones:
            digest = self.fine._bin_edges_sha256(
                plan.edges,
                interpolation_order=plan.order,
                phasor_moment_order=self.fine.phasor_moment_order,
                phasor_time_anchors=self.fine.phasor_time_anchors,
                phasor_approximation=getattr(
                    self.fine, "phasor_approximation", "taylor"
                ),
                reference_projection=self.fine.reference_projection,
            )
            self.clones[key] = rebin_likelihood(
                self.fine,
                plan.edges,
                reference_waveform=self.waveform,
                xg_plan=_QualificationXGPlan(digest, _XG_QUALIFICATION_PLAN_AUTHORITY),
            )
        return self.clones[key]

    def plan_key(self, plan):
        return self.fine._bin_edges_sha256(
            plan.edges,
            interpolation_order=plan.order,
            phasor_moment_order=self.fine.phasor_moment_order,
            phasor_time_anchors=self.fine.phasor_time_anchors,
            reference_projection=self.fine.reference_projection,
            phasor_approximation=getattr(self.fine, "phasor_approximation", "taylor"),
        )

    def __call__(self, plan):
        lk = self.clone(plan)
        lanes = self.settings.timing_lanes or max(
            1,
            int(
                self.cfg.sampler.n_live
                * self.cfg.sampler.n_delete_frac
                / self.cfg.sampler.n_devices
            ),
        )
        if lanes > 512:
            raise ValueError("bin timing requires timing_lanes <=512 for this sampler")
        points = self.bank.points["training"]
        p = {
            key: jnp.asarray([points[i % len(points)][key] for i in range(lanes)])
            for key in points[0]
        }
        started = time.perf_counter()
        cache = jax.jit(jax.vmap(lk.generate_waveform))(p)
        jax.block_until_ready(cache)
        summary = {
            "overlap": jnp.zeros((lanes, 2), dtype=jnp.complex128),
            "gram": jnp.zeros((lanes, 2, 2), dtype=jnp.complex128),
            "fixed": {
                key: value
                for key, value in p.items()
                if key not in {"psi", "iota", "d_L"}
            },
        }
        from jimgw.core.single_event.heterodyne_extrinsics import (
            evaluate_extrinsic_summary,
        )

        def workload(kind, params, accepted, conditional):
            def rebuild(_):
                proposed = jax.vmap(lk.generate_waveform)(params)
                value = jax.vmap(lk.evaluate_from_waveform)(params, proposed)
                return value, conditional, proposed

            def prepare(_):
                prepared = jax.vmap(lk.build_extrinsic_summary)(params, accepted)
                return jnp.zeros(lanes, dtype=jnp.float64), prepared, accepted

            def hit(_):
                value = jax.vmap(lambda pp, ss: evaluate_extrinsic_summary(lk, pp, ss))(
                    params, conditional
                )
                return value, conditional, accepted

            return jax.lax.switch(kind, (rebuild, prepare, hit), None)

        compiled = jax.jit(workload).lower(jnp.int32(0), p, cache, summary).compile()
        _, summary, _ = compiled(jnp.int32(1), p, cache, summary)
        jax.block_until_ready(summary)
        compile_seconds = time.perf_counter() - started
        self.compile_seconds += compile_seconds
        for kind in range(3):
            jax.block_until_ready(compiled(jnp.int32(kind), p, cache, summary))
        elapsed = [[], [], []]
        # Rotate branch order so one callback does not always get first/last.
        for repeat in range(self.settings.timing_repeats):
            for kind in ((repeat + i) % 3 for i in range(3)):
                start = time.perf_counter()
                jax.block_until_ready(compiled(jnp.int32(kind), p, cache, summary))
                elapsed[kind].append(time.perf_counter() - start)
        medians = [float(np.median(values)) for values in elapsed]
        # Use the configured slice budget and contiguous scalar-hit segments.
        # Endpoint/shrink multiplicities still require a full sampler run.
        cost = float(np.dot(self.weights, medians))
        self.records[self.plan_key(plan)] = {
            "plan_sha256": self.plan_key(plan),
            "n_bins": plan.n_bins,
            "nodes": int(lk.freq_grid_node_flat.size),
            "lanes": lanes,
            "backend": jax.default_backend(),
            "compile_and_cache_seconds": compile_seconds,
            "rebuild_median_seconds": medians[0],
            "summary_prepare_median_seconds": medians[1],
            "scalar_hit_median_seconds": medians[2],
            "samples_seconds": elapsed,
            "callback_weights_rebuild_prepare_hit": list(self.weights),
            "weighted_callback_seconds": cost,
            "scope": "configured rebuild/summary-preparation/scalar-hit batched callbacks; not a full FSM transition",
            "phasor_approximation": getattr(lk, "phasor_approximation", "taylor"),
        }
        return cost


def select_network_binning(cfg, fine, detectors, waveform):
    """Return a frozen, rebound research candidate and its resolved config."""
    from jimgw.cli.xg_qualification import (
        _verify_qualification_candidate_binding,
        _verify_realized_candidate_inputs,
        bind_xg_qualification_candidate,
    )

    settings = cfg.likelihood.heterodyne.bin_selection
    if settings is None:
        raise ValueError("automatic bin selection settings are required")
    if not (
        fine.reference_projection == "carrier"
        and fine.phasor_time_anchors is not None
        and fine.phasor_moment_order > 0
        and fine.evaluation_diagnostics.get("implementation")
        in {"factored-carrier-arm-even-odd-v1", XG_EVALUATION_REVISION}
    ):
        raise ValueError(
            "automatic network bin selection requires the supported fast anchored-carrier evaluator"
        )
    started = time.perf_counter()
    training, verification = parameter_banks(cfg, detectors, settings)
    bank = SparseRatioBank(fine, training, verification, settings)
    oracle = MomentErrorOracle(fine, bank, settings.tolerance)
    timer = CandidateTimer(cfg, fine, waveform, bank, settings)
    counts = [n for n in settings.candidate_bins if n <= fine.n_bins]
    allocation = None
    if settings.method == "adaptive":
        from jimgw.cli._xg_allocation import select_adaptive_network_grid

        selection, allocation = select_adaptive_network_grid(
            fine, bank, oracle, timer, settings
        )
    else:
        selection = select_frozen_grid(
            fine.freq_grid_edges,
            lambda f: bank(f, "training"),
            lambda f: bank(f, "verification"),
            orders=(fine.interpolation_order,),
            bin_counts=counts,
            max_bins=fine.n_bins,
            error_budget=settings.tolerance * 0.75,
            likelihood_error=oracle,
            validated_cost=timer,
        )
    resolved = cfg.model_copy(deep=True)
    resolved.likelihood.heterodyne = resolved.likelihood.heterodyne.model_copy(
        update={
            "n_bins": selection.plan.n_bins,
            "frequency_bin_edges": selection.plan.edges.tolist(),
            "node_frequency_prefix": np.asarray(bank.prefix).tolist(),
            "epsilon": None,
        }
    )
    digest = fine._bin_edges_sha256(
        selection.plan.edges,
        interpolation_order=fine.interpolation_order,
        phasor_moment_order=fine.phasor_moment_order,
        phasor_time_anchors=fine.phasor_time_anchors,
        phasor_approximation=getattr(fine, "phasor_approximation", "taylor"),
        reference_projection=fine.reference_projection,
    )
    binding = bind_xg_qualification_candidate(resolved, digest)
    _verify_realized_candidate_inputs(binding, resolved, detectors, waveform)
    candidate = timer.clone(selection.plan)
    _verify_qualification_candidate_binding(binding, resolved)
    record = {
        "method": "shared-native-moments-adaptive-k8-v1"
        if allocation
        else "shared-native-moments-frozen-grid-v1",
        "allocation": allocation,
        "qualification": False,
        "validity_scope": "empirical prior-derived training and independent verification bank; native spot checks remain required before sampling",
        "cost_metric": (
            "measured batch latency weighted by configured rebuild/summary-preparation/scalar-hit counts; "
            + selection.cost_metric
        ),
        "callback_weights_rebuild_prepare_hit": list(timer.weights),
        "requested_tolerance_nats": settings.tolerance,
        "selection_tolerance_nats": settings.tolerance * 0.75,
        "reference_convergence_tolerance_nats": settings.tolerance / 4,
        "reference_convergence_error_nats": oracle.reference_errors,
        "maximum_training_error_nats": float(np.max(selection.training_error)),
        "maximum_verification_error_nats": float(np.max(selection.verification_error)),
        "training_cases": len(training),
        "verification_cases": len(verification),
        "training_bank_sha256": _bank_hash(training),
        "verification_bank_sha256": _bank_hash(verification),
        "prepared_parameter_banks": {
            name: [{k: float(v) for k, v in p.items()} for p in points]
            for name, points in bank.points.items()
        },
        "prepared_parameter_bank_sha256": {
            name: _bank_hash(points) for name, points in bank.points.items()
        },
        "reference_bins": fine.n_bins,
        "selected_bins": candidate.n_bins,
        "selected_nodes": int(candidate.freq_grid_node_flat.size),
        "candidate_diagnostics": list(selection.candidate_diagnostics),
        "sparse_frequency_count": selection.sparse_frequency_count,
        "ratio_batch_calls": bank.calls,
        "ratio_seconds_including_compilation": bank.wall_seconds,
        "native_summary_builds": 1,
        "native_samples_read_during_selection": 0,
        "candidate_likelihood_compiles_during_accuracy_search": 0,
        "timed_valid_candidates": timer.records,
        "valid_candidate_compilation_seconds": timer.compile_seconds,
        "wall_seconds_including_compilation": time.perf_counter() - started,
        "bin_edges_sha256": digest,
        "analysis_contract_sha256": binding.analysis_contract_sha256,
        "evaluation": candidate.evaluation_diagnostics,
        "phasor_approximation": getattr(candidate, "phasor_approximation", "taylor"),
        "phasor_approximation_diagnostics": getattr(
            candidate, "phasor_approximation_diagnostics", None
        ),
    }
    candidate.bin_selection_diagnostics = record
    return candidate, resolved, record
