"""Exact nested coarsening of discrete heterodyne moments.

Only the polynomial coordinate changes: native noisy data are never read or
interpolated. Stored anchor moments use exp(+2*pi*i*f*anchor), so they require
the same affine transform as unanchored moments, with no centre rephasing.
Likelihood clones are unqualified candidates and retain the fine reference
function, frequency support, detector bands, polynomial orders, and anchors.
"""

from __future__ import annotations

import copy
import math

import jax
import jax.numpy as jnp
import numpy as np


def _nested_edges(fine_edges, coarse_edges):
    fine, coarse = (
        np.asarray(jax.device_get(value), dtype=np.float64)
        for value in (fine_edges, coarse_edges)
    )
    for name, edges in (("fine", fine), ("coarse", coarse)):
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError(f"{name} edges must be finite and strictly increasing")
    if coarse[0] != fine[0] or coarse[-1] != fine[-1]:
        raise ValueError("coarsening must preserve the exact frequency support")
    indices = np.searchsorted(fine, coarse)
    if np.any(indices >= len(fine)) or not np.array_equal(fine[indices], coarse):
        raise ValueError("coarse edges must be an exact subset of fine edges")
    return fine, coarse, indices


def coarsen_moments(fine_edges, coarse_edges, moments):
    """Translate and sum ``[..., degree, fine_bin]`` into nested coarse bins.

    For a child bin, u_parent = alpha + beta*u_child. The degree-k parent
    moment is sum_j binom(k,j)*alpha**(k-j)*beta**j times its child moment,
    summed over children. Leading axes (including absolute-frequency time
    anchors) are preserved. Input and output use real64 or complex128.
    """
    fine, coarse, indices = _nested_edges(fine_edges, coarse_edges)
    values = np.asarray(jax.device_get(moments))
    if values.ndim < 2 or values.shape[-2] < 1 or values.shape[-1] != len(fine) - 1:
        raise ValueError("moments must have shape [..., degree, fine_bin]")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("moments must contain finite numeric values")
    values = values.astype(np.complex128 if np.iscomplexobj(values) else np.float64)
    if np.array_equal(fine, coarse):
        return values.copy()
    parent = np.searchsorted(indices[1:], np.arange(len(fine) - 1), side="right")
    child_center = (fine[:-1] + fine[1:]) / 2
    child_half = np.diff(fine) / 2
    parent_center = ((coarse[:-1] + coarse[1:]) / 2)[parent]
    parent_half = (np.diff(coarse) / 2)[parent]
    alpha, beta = (child_center - parent_center) / parent_half, child_half / parent_half
    count = values.shape[-2]
    transform = np.zeros((len(fine) - 1, count, count), dtype=np.float64)
    for k in range(count):
        for j in range(k + 1):
            transform[:, k, j] = math.comb(k, j) * alpha ** (k - j) * beta**j
    translated = np.einsum("...jf,fkj->...kf", values, transform, optimize=True)
    return np.add.reduceat(translated, indices[:-1], axis=-1)


def rebin_likelihood(fine, coarse_edges, *, reference_waveform=None, xg_plan=None):
    """Clone a polynomial native-moment likelihood without dense-grid access.

    XG clones require an internally authorized *qualification* capability for
    the new digest; a production receipt is never reused. The caller binds and
    validates the changed analysis contract before constructing that capability.
    Reference node evaluations retain the fine node grid's first two frequency
    values, preserving spacing-sensitive waveform cutoff conventions. Nested
    endpoint references are gathered directly from the original endpoint arrays.
    Core fast evaluators are reinstalled after the clone's arrays are replaced.
    Supported stock clock adapters preserve the prefix in both proposal and
    reference evaluation. Other models must explicitly declare
    ``frequency_grid_independent = True`` on both waveform objects; unknown
    grid-dependent custom sources are rejected.
    """
    from jimgw.core.single_event.likelihood import (
        _XG_QUALIFICATION_PLAN_AUTHORITY,
        HeterodynedTransientLikelihoodFD,
        _QualificationXGPlan,
    )

    if not isinstance(fine, HeterodynedTransientLikelihoodFD):
        raise TypeError("rebinning requires a heterodyned likelihood")
    if (
        fine.interpolation_order < 2
        or fine.time_marginalization
        or getattr(fine, "distance_marginalization", False)
        or getattr(fine, "zero_noise_summary", None) is not None
    ):
        raise ValueError(
            "rebinning supports native polynomial moments without time/distance marginalization"
        )
    if any(
        name in fine.__dict__
        for name in ("_polynomial_likelihood", "_research_extrinsic_summary")
    ):
        raise ValueError(
            "instance-bound research evaluators must be replaced by the core evaluator before rebinning"
        )
    old_edges, edges, indices = _nested_edges(fine.freq_grid_edges, coarse_edges)
    digest = fine._bin_edges_sha256(
        edges,
        interpolation_order=fine.interpolation_order,
        phasor_moment_order=fine.phasor_moment_order,
        phasor_time_anchors=fine.phasor_time_anchors,
        phasor_approximation=getattr(fine, "phasor_approximation", "taylor"),
        reference_projection=fine.reference_projection,
    )
    xg_response = bool(
        getattr(fine.waveform, "time_dependent_response", False)
        or getattr(fine.waveform, "response_is_time_dependent", False)
        or any(
            getattr(d, "time_dependent_response", False)
            or getattr(d, "response_is_time_dependent", False)
            or getattr(d, "finite_arm_response", False)
            for d in fine.detectors
        )
    )
    if (xg_plan is not None or xg_response) and not (
        isinstance(xg_plan, _QualificationXGPlan)
        and xg_plan._authority is _XG_QUALIFICATION_PLAN_AUTHORITY
        and xg_plan.bin_edges_sha256 == digest
    ):
        raise ValueError(
            "XG rebinning requires a matching authorized qualification plan"
        )
    source = (
        reference_waveform
        if reference_waveform is not None
        else getattr(fine, "_reference_waveform", None)
    )
    if source is None:
        raise ValueError("rebinning requires the original reference_waveform")
    from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
    from jimgw.core.single_event.xg_waveform import supports_source

    baseline = getattr(fine, "_baseline_waveform", fine.waveform)
    supported = all(
        type(model) is DominantModeTimeCachedWaveform
        and model.mode == 2
        and supports_source(model.source)
        for model in (baseline, source)
    )
    grid_independent = all(
        getattr(model, "frequency_grid_independent", False) is True
        for model in (baseline, source)
    )
    if not supported and not grid_independent:
        raise ValueError(
            "rebinning requires a supported stock waveform or an explicit "
            "frequency_grid_independent contract for candidate and reference"
        )

    clone = copy.copy(fine)
    clone.reference_parameters = dict(fine.reference_parameters)
    clone._reference_waveform = source
    clone._set_frequency_arrays(jnp.asarray(edges))
    clone.bin_edges_sha256 = digest
    clone._xg_node_frequency_prefix = jnp.asarray(
        getattr(fine, "_xg_node_frequency_prefix", fine.freq_grid_node_flat[:2])
    )
    if clone._xg_node_frequency_prefix.shape != (2,):
        raise ValueError("reference frequency prefix must contain exactly two values")
    host_prefix = np.asarray(clone._xg_node_frequency_prefix)
    if not np.all(np.isfinite(host_prefix)) or host_prefix[1] <= host_prefix[0]:
        raise ValueError("reference frequency prefix must be finite and increasing")
    clone.node_frequency_prefix = tuple(float(value) for value in host_prefix)
    frequency = clone.freq_grid_node_flat
    sky = source(
        jnp.concatenate((clone._xg_node_frequency_prefix, frequency)),
        clone.reference_parameters,
    )
    sky = jax.tree.map(lambda value: value[2:], sky)
    clone.waveform_node_ref = {}
    clone.waveform_low_ref = {}
    clone.waveform_high_ref = {}
    clone.summary_moments = {}
    clone.summary_data = {}
    clone.phasor_data_moments = {}
    for detector in clone.detectors:
        name = detector.name
        projected = clone._project_reference(detector, frequency, sky)
        clone._validate_reference_projection(name, sky, projected, "node")
        clone.waveform_node_ref[name] = jnp.reshape(
            projected, (clone.interpolation_order + 1, clone.n_bins)
        )
        clone.waveform_low_ref[name] = fine.waveform_low_ref[name][indices[:-1]]
        clone.waveform_high_ref[name] = fine.waveform_high_ref[name][indices[1:] - 1]
        a, b = fine.summary_moments[name]
        if a.shape != (
            fine.interpolation_order + fine.phasor_moment_order + 1,
            fine.n_bins,
        ) or b.shape != (2 * fine.interpolation_order + 1, fine.n_bins):
            raise ValueError(
                "moment bank does not match interpolation and phasor orders"
            )
        a, b = (
            jnp.asarray(coarsen_moments(old_edges, edges, value)) for value in (a, b)
        )
        clone.summary_moments[name] = (a, b)
        clone.summary_data[name] = jnp.concatenate((a, b))
        if fine.phasor_time_anchors is not None:
            bank = fine.phasor_data_moments[name]
            if bank.shape != (len(fine.phasor_time_anchors), a.shape[0], fine.n_bins):
                raise ValueError(
                    "phasor moment bank does not match anchors and degrees"
                )
            clone.phasor_data_moments[name] = jnp.asarray(
                coarsen_moments(old_edges, edges, bank)
            )
    clone.rebin_diagnostics = {
        "method": "exact-affine-moment-coarsening-v1",
        "source_bin_edges_sha256": fine.bin_edges_sha256,
        "bin_edges_sha256": digest,
        "source_n_bins": fine.n_bins,
        "n_bins": clone.n_bins,
        "native_samples_read": 0,
        "qualification": False,
        "node_frequency_prefix": np.asarray(clone._xg_node_frequency_prefix).tolist(),
    }
    clone.summary_construction_diagnostics = {
        name: {"method": "exact-affine-moment-coarsening-v1", "native_samples_read": 0}
        for name in clone.summary_moments
    }
    clone.construction_diagnostics = {
        "source_construction": dict(getattr(fine, "construction_diagnostics", {})),
        "rebin": dict(clone.rebin_diagnostics),
    }
    if hasattr(fine, "xg_evaluation_mode"):
        from jimgw.core.single_event.xg_evaluation import configure_xg_evaluation

        configure_xg_evaluation(clone, source, mode=fine.xg_evaluation_mode)
    return clone
