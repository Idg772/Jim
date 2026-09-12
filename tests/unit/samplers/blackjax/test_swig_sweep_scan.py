"""Exact local comparisons for the private, opt-in repeated-sweep scan."""

import json
import time
from collections import Counter
from functools import partial
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import StateWithLogLikelihood, init_state_strategy
from jax.extend import core
from jax.sharding import Mesh

from jimgw.samplers.blackjax._fsm import (
    SegmentSchedule,
    run_segment,
    slice_randoms_from_keys,
)
from jimgw.samplers.blackjax.swig import (
    BlackJAXSwiGSampler,
    CachedSliceState,
    _build_swig_constrained_step,
    _run_swig_sweep_scan,
)
from jimgw.samplers.config import BlackJAXSwiGConfig


def _equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def _jaxpr_counts(function, *args):
    counts = Counter()

    def visit(value):
        if isinstance(value, core.Jaxpr):
            for equation in value.eqns:
                counts[equation.primitive.name] += 1
                visit(equation.params)
        elif isinstance(value, core.ClosedJaxpr):
            visit(value.jaxpr)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(jax.make_jaxpr(function)(*args))
    return {
        "equations": sum(counts.values()),
        "while": counts["while"],
        "scan": counts["scan"],
    }


def _with_visible_final_cache(step, rolled, key, state, threshold, factors):
    """Expose an existing private boundary output during tracing, without callbacks.

    No cache is rebuilt for this probe. The wrapper returns the final segment's
    actual accepted cache as an additional compiled output for direct comparison.
    """
    captured = []

    def capture_segment(*args, **kwargs):
        result = run_segment(*args, **kwargs)
        captured.append(result[0].cache)
        return result

    def capture_scan(*args, **kwargs):
        result = _run_swig_sweep_scan(*args, **kwargs)
        captured.append(result[0].cache)
        return result

    target = "_run_swig_sweep_scan" if rolled else "run_segment"
    wrapper = capture_scan if rolled else capture_segment
    with patch(f"jimgw.samplers.blackjax.swig.{target}", wrapper):
        result = step(key, state, threshold, block_covariance_factors=factors)
    return result, captured[-1]


def _psd_weighted_cache_bound(likelihood, params, left_cache, right_cache):
    """Bound this cache perturbation using the actual discrete PSD moments.

    B moments contain 4 df sum(|h_ref|^2 u^k / PSD). Absolute moment
    contractions upper-bound the squared norms without a cancellation-prone
    subtraction or norm clamping. The finite Taylor phasor's absolute envelope
    handles the distinct data-overlap approximation. This is a local bound for
    the polynomial model and the tested cache perturbation, not a native-model
    or posterior qualification.
    """
    assert likelihood.phasor_moment_order > 0
    params = likelihood._prepare_parameters(params)
    frequencies = likelihood.freq_grid_node_flat
    sky = [
        likelihood._waveform_sky_from_cache(frequencies, cache["nodes"], params)
        for cache in (left_cache, right_cache)
    ]
    order = likelihood.interpolation_order
    degrees = np.arange(order + 1)
    vandermonde = np.asarray(likelihood._vandermonde_inverse)
    data_norm = norm_upper = delta_norm_upper = delta_taylor_norm_upper = 0.0
    for detector in likelihood.detectors:
        reference = np.asarray(likelihood.waveform_node_ref[detector.name])
        ratios = [
            np.asarray(
                detector.fd_response(frequencies, polarizations, params)
            ).reshape(order + 1, likelihood.n_bins)
            / reference
            for polarizations in sky
        ]
        dt = float(likelihood._rigid_time_shift(detector, params))
        phasor = np.exp(2j * np.pi * np.asarray(likelihood.freq_grid_nodes) * dt)
        coefficient = vandermonde @ (ratios[0] * phasor)
        delta = vandermonde @ ((ratios[1] - ratios[0]) * phasor)
        b = np.abs(
            np.asarray(likelihood.summary_moments[detector.name][1])[
                degrees[:, None] + degrees[None, :]
            ]
        )
        old_per_bin = np.einsum(
            "kb,lb,klb->b", np.abs(coefficient), np.abs(coefficient), b
        )
        delta_per_bin = np.einsum("kb,lb,klb->b", np.abs(delta), np.abs(delta), b)
        residual = dt
        if likelihood.phasor_time_anchors is not None:
            anchors = np.asarray(likelihood.phasor_time_anchors)
            assert anchors[0] <= dt <= anchors[-1]
            residual -= anchors[np.argmin(abs(anchors - dt))]
        theta = 2 * np.pi * np.asarray(likelihood.freq_grid_half_widths) * abs(residual)
        term = np.ones_like(theta)
        envelope = term.copy()
        for m in range(1, likelihood.phasor_moment_order + 1):
            term = term * theta / m
            envelope += term
        norm_upper += float(np.sum(old_per_bin))
        delta_norm_upper += float(np.sum(delta_per_bin))
        delta_taylor_norm_upper += float(np.sum(delta_per_bin * envelope**2))
        data_norm += float(
            4
            / detector.duration
            * np.sum(
                abs(np.asarray(detector.sliced_fd_data)) ** 2
                / np.asarray(detector.sliced_psd)
            )
        )
    overlap_bound = np.sqrt(data_norm * delta_taylor_norm_upper)
    half_norm_bound = np.sqrt(norm_upper * delta_norm_upper) + 0.5 * delta_norm_upper
    return {
        "data_norm_squared": data_norm,
        "waveform_norm_squared_upper": norm_upper,
        "delta_waveform_norm_squared_upper": delta_norm_upper,
        "delta_taylor_waveform_norm_squared_upper": delta_taylor_norm_upper,
        "overlap_change_bound": float(overlap_bound),
        "half_norm_change_bound": float(half_norm_bound),
        "phase_marginalized_loglikelihood_change_bound": float(
            overlap_bound + half_norm_bound
        ),
    }


@pytest.mark.parametrize(
    "case,cap",
    [
        ("ordinary", 100),
        ("periodic", 100),
        ("nonconvex", 100),
        ("nan", 2),
        ("ordinary", 1),
    ],
)
def test_segment_scan_preserves_full_cache_and_diagnostics(case, cap):
    def prior(x):
        return jnp.where(jnp.all(abs(x) < 3), -jnp.sum(x * x) / 8, -jnp.inf)

    def build(x):
        return {"x": x[0], "powers": jnp.array([x[0] ** 2, x[0] ** 3])}

    def score(x, cache):
        value = -cache["powers"][0] - x[1] ** 2
        if case == "nonconvex":
            value = -((cache["powers"][0] - 0.7) ** 2) - x[1] ** 2
        if case == "nan":
            value = jnp.where(x[0] > 0.6, jnp.nan, value)
        return value

    def wrap(x):
        return x.at[1].set((x[1] + 2) % 4 - 2) if case == "periodic" else x

    def chain(key, position, threshold, rolled):
        state = CachedSliceState(
            position,
            prior(position),
            score(position, build(position)),
            threshold,
            build(position),
        )
        segments = []
        for sweep in range(8):
            for rebuild, length in ((True, 9), (False, 3)):
                keys = jax.random.split(
                    jax.random.fold_in(key, sweep * 2 + int(not rebuild)), length
                )
                _, level, left, budget, shrink = slice_randoms_from_keys(keys)
                direction = jnp.tile(
                    jnp.array([0.4, 0.0]) if rebuild else jnp.array([0.0, 0.4]),
                    (length, 1),
                )
                segments.append(
                    (rebuild, SegmentSchedule(direction, level, left, budget, shrink))
                )

        def one(current, rebuild, schedule):
            def evaluate(x, cache):
                cache = build(x) if rebuild else cache
                return prior(x), score(x, cache), cache

            return run_segment(
                schedule,
                current,
                threshold,
                eval_candidate=evaluate,
                wrap_position=wrap,
                max_expansions=10,
                max_shrinkage=cap,
            )

        if rolled:
            final, infos = _run_swig_sweep_scan(segments, state, 8, one)
            return final, infos[0]
        infos = []
        for mode, schedule in segments:
            state, info = one(state, mode, schedule)
            infos.append(info)
        return state, jax.tree.map(lambda *parts: jnp.concatenate(parts), *infos)

    keys = jax.random.split(jax.random.key(71), 8)
    positions = jnp.stack(
        (jnp.linspace(-0.4, 0.4, 8), jnp.linspace(-0.3, 0.3, 8)), axis=1
    )
    thresholds = jnp.linspace(-2.0, -0.7, 8)
    ordinary = jax.jit(jax.vmap(lambda k, x, t: chain(k, x, t, False)))(
        keys, positions, thresholds
    )
    rolled = jax.jit(jax.vmap(lambda k, x, t: chain(k, x, t, True)))(
        keys, positions, thresholds
    )
    _equal(ordinary, rolled)
    assert ordinary[1].num_expansions.shape == (8, 96)
    assert np.asarray(ordinary[1].num_expansions).sum() > 0
    if cap == 1:
        assert np.any(~np.asarray(ordinary[1].is_accepted))


def _builder_options():
    def prior(x):
        return jnp.where(jnp.all(abs(x) < 3), -0.5 * jnp.sum(x * x), -jnp.inf)

    def build(x):
        return x[:9] ** 2

    def evaluate(x, cache):
        return -jnp.sum(cache) - jnp.sum(x[9:] ** 2)

    return {
        "log_prior_fn": prior,
        "build_cache": build,
        "log_likelihood_from_cache_fn": evaluate,
        "prepare_hit_summary": lambda x, cache: -jnp.sum(cache),
        "log_likelihood_from_hit_summary": lambda x, summary: (
            summary - jnp.sum(x[9:] ** 2)
        ),
        "rebuild_required_by_block": {
            tuple(range(7)): True,
            (7, 8): True,
            (9,): False,
            (10, 11): False,
        },
        "num_gibbs_sweeps": 8,
        "num_inner_steps_per_dim": 1,
        "max_steps": 10,
        "max_shrinkage": 100,
        "periodic": {9: (-3.0, 3.0)},
        "n_dims": 12,
        "per_slice_info": True,
        "bracket_mode": "stepping-out",
    }


def test_actual_block_schedule_and_scalar_cache_preserve_transition(tmp_path):
    options = _builder_options()
    x = jnp.linspace(-0.2, 0.2, 12)
    threshold = jnp.asarray(-4.0)
    state = StateWithLogLikelihood(
        x,
        options["log_prior_fn"](x),
        options["log_likelihood_from_cache_fn"](x, options["build_cache"](x)),
        threshold,
    )
    keys = jax.random.split(jax.random.key(27), 8)
    factors = tuple(jnp.eye(n) * 0.35 for n in (7, 2, 1, 2))
    widths = tuple(jnp.asarray(w) for w in (0.8, 1.0, 0.7, 1.2))
    results, evidence = [], {}
    for name, rolled in (("unrolled", False), ("scan", True)):
        step = _build_swig_constrained_step(**options, _scan_sweeps=rolled)
        fn = jax.vmap(
            lambda key, step=step: step(
                key,
                state,
                threshold,
                block_covariance_factors=factors,
                block_widths=widths,
            )
        )
        counts = _jaxpr_counts(fn, keys)
        started = time.perf_counter()
        compiled = jax.jit(fn).lower(keys).compile()
        counts["lower_compile_seconds"] = time.perf_counter() - started
        results.append(jax.block_until_ready(compiled(keys)))
        evidence[name] = counts
    _equal(*results)
    assert results[0][1].num_expansions.shape == (8, 96)
    assert evidence["unrolled"]["while"] == 16
    assert evidence["scan"]["while"] == 2
    assert evidence["scan"]["equations"] < evidence["unrolled"]["equations"]
    (tmp_path / "graph-evidence.json").write_text(json.dumps(evidence, indent=2))
    print("sweep-scan evidence", json.dumps(evidence))


@pytest.mark.parametrize(
    "extra",
    [
        {"num_de_jumps": 1},
        {"resolved_bridge_blocks": (((9, 10), False),)},
        {"rebuild_required_by_block": {(0,): True, (1,): True}},
    ],
)
def test_private_scan_rejects_schedules_outside_current_scope(extra):
    with pytest.raises(ValueError, match="private sweep scan requires"):
        _build_swig_constrained_step(**(_builder_options() | extra), _scan_sweeps=True)


def test_successive_replicated_nss_steps_preserve_width_and_covariance_adaptation():
    options = _builder_options()
    groups = options["rebuild_required_by_block"]
    config = BlackJAXSwiGConfig(
        blocks=[[f"x{i}" for i in indices] for indices in groups],
        n_live=32,
        n_delete_frac=0.125,
        num_gibbs_sweeps=8,
        num_inner_steps_per_dim=1,
        adaptive_slice_widths=True,
        periodic_wrapped_covariance=True,
        scalar_extrinsic_cache=True,
        bracket_mode="stepping-out",
        max_steps=10,
        max_shrinkage=100,
    )
    prior = options["log_prior_fn"]
    score = lambda x: options["log_likelihood_from_cache_fn"](
        x, options["build_cache"](x)
    )
    sampler = BlackJAXSwiGSampler(
        n_dims=12,
        log_prior_fn=prior,
        log_likelihood_fn=score,
        log_posterior_fn=lambda x: prior(x) + score(x),
        config=config,
        periodic=options["periodic"],
        **{
            key: options[key]
            for key in (
                "rebuild_required_by_block",
                "build_cache",
                "log_likelihood_from_cache_fn",
                "prepare_hit_summary",
                "log_likelihood_from_hit_summary",
            )
        },
    )
    positions = jax.random.normal(jax.random.key(220), (32, 12)) * 0.25
    initial = adaptive_init(
        positions,
        init_state_fn=jax.vmap(
            partial(init_state_strategy, logprior_fn=prior, loglikelihood_fn=score)
        ),
        update_inner_kernel_params_fn=sampler._fsm_update_inner_kernel_params_fn,
    )
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("replacement",))
    kernels = []
    for rolled in (False, True):
        with patch(
            "jimgw.samplers.blackjax.swig._build_swig_constrained_step",
            partial(_build_swig_constrained_step, _scan_sweeps=rolled),
        ):
            kernels.append(jax.jit(sampler._build_nested_sampler(4, mesh=mesh).step))
    states = [initial, initial]
    extensions = 0
    for key in jax.random.split(jax.random.key(221), 3):
        pairs = [
            jax.block_until_ready(kernel(key, state))
            for kernel, state in zip(kernels, states, strict=True)
        ]
        _equal(*pairs)
        states = [pair[0] for pair in pairs]
        extensions += int(jnp.sum(pairs[0][1].update_info.num_expansions))
    assert extensions > 0
    assert any(
        float(width) != 1.0 for width in states[0].inner_kernel_params["block_widths"]
    )
