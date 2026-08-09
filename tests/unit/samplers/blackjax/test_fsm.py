"""Bitwise tests for the FSM slice-segment scheduler."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.mcmc.slice import SliceInfo
from blackjax.mcmc.slice import build_kernel as build_slice_kernel
from blackjax.ns.base import NSState, StateWithLogLikelihood
from blackjax.ns.base import init_state_strategy as _init_state_strategy
from blackjax.ns.nss import slice_constrained_step
from jax.sharding import Mesh

from jimgw.samplers.blackjax._fsm import (
    SegmentSchedule,
    run_segment,
    slice_randoms_from_keys,
)
from jimgw.samplers.blackjax._slice import stepping_out_cached
from jimgw.samplers.blackjax.nss import (
    _build_nss_fsm_constrained_step,
    _sample_direction_from_covariance_factor,
)
from jimgw.samplers.blackjax.sharding import update_with_mcmc_take_last_replicated
from jimgw.samplers.blackjax.swig import (
    CachedSliceState,
    _build_swig_constrained_step,
    _build_swig_constrained_step_lockstep,
)
from jimgw.samplers.periodic import to_prior_space_proposal

N_DIMS = 3


def _log_prior(position):
    return -0.5 * jnp.sum(position**2)


def _log_likelihood(position):
    return -jnp.sum((position - 0.2) ** 2) * 3.0


def _make_directions(prop_keys):
    # Position-independent direction draw, one per slice (unit normals here;
    # production callers use sample_direction_from_covariance).
    return jax.vmap(lambda k: jax.random.normal(k, (N_DIMS,)))(prop_keys)


def _reference_chain(
    chain_key,
    position,
    loglikelihood_0,
    n_slices,
    max_expansions,
    max_shrinkage,
):
    """Current implementation: one BlackJAX slice kernel per slice, scanned."""
    kernel = build_slice_kernel(
        interval=stepping_out_cached,
        max_expansions=max_expansions,
        max_shrinkage=max_shrinkage,
    )
    slice_keys = jax.random.split(chain_key, n_slices)
    prop_keys = jax.vmap(lambda k: jax.random.split(k)[0])(slice_keys)
    directions = _make_directions(prop_keys)

    state = CachedSliceState(
        position=position,
        logdensity=_log_prior(position),
        loglikelihood=_log_likelihood(position),
        loglikelihood_birth=jnp.asarray(loglikelihood_0),
        cache=(),
    )

    def one_slice(carry, inp):
        key, direction = inp

        def proposal_generator(direction_key, pos, logdensity_fn):
            del direction_key, logdensity_fn

            def slice_fn(t):
                x = pos + t * direction
                candidate = CachedSliceState(
                    position=x,
                    logdensity=_log_prior(x),
                    loglikelihood=_log_likelihood(x),
                    loglikelihood_birth=jnp.asarray(loglikelihood_0),
                    cache=(),
                )
                return candidate, candidate.loglikelihood > loglikelihood_0

            return slice_fn

        new_state, info = kernel(key, carry, None, proposal_generator)
        return new_state, info

    return jax.lax.scan(one_slice, state, (slice_keys, directions))


def _fsm_chain(
    chain_key,
    position,
    loglikelihood_0,
    n_slices,
    max_expansions,
    max_shrinkage,
):
    slice_keys = jax.random.split(chain_key, n_slices)
    prop_keys, level_u, bracket_u, bracket_v, shrink_key_data = slice_randoms_from_keys(
        slice_keys
    )
    schedule = SegmentSchedule(
        directions=_make_directions(prop_keys),
        level_u=level_u,
        bracket_u=bracket_u,
        bracket_v=bracket_v,
        shrink_key_data=shrink_key_data,
    )
    state = CachedSliceState(
        position=position,
        logdensity=_log_prior(position),
        loglikelihood=_log_likelihood(position),
        loglikelihood_birth=jnp.asarray(loglikelihood_0),
        cache=(),
    )

    def eval_candidate(pos, cache):
        return _log_prior(pos), _log_likelihood(pos), cache

    return run_segment(
        schedule,
        state,
        loglikelihood_0,
        eval_candidate=eval_candidate,
        wrap_position=lambda x: x,
        max_expansions=max_expansions,
        max_shrinkage=max_shrinkage,
    )


@pytest.mark.parametrize("max_expansions,max_shrinkage", [(10, 100), (1, 100), (10, 2)])
def test_run_segment_matches_blackjax_scan_bitwise(max_expansions, max_shrinkage):
    n_lanes, n_slices = 4096, 5
    keys = jax.random.split(jax.random.key(7), n_lanes)
    positions = jnp.linspace(-1.5, 1.5, n_lanes * N_DIMS).reshape(n_lanes, N_DIMS)
    # Heterogeneous per-lane contours force very different expansion/shrink counts.
    thresholds = jnp.linspace(-40.0, -1.0, n_lanes)

    ref_state, ref_info = jax.jit(
        jax.vmap(
            lambda k, p, l0: _reference_chain(
                k,
                p,
                l0,
                n_slices,
                max_expansions,
                max_shrinkage,
            )
        )
    )(keys, positions, thresholds)
    fsm_state, fsm_info = jax.jit(
        jax.vmap(
            lambda k, p, l0: _fsm_chain(
                k,
                p,
                l0,
                n_slices,
                max_expansions,
                max_shrinkage,
            )
        )
    )(keys, positions, thresholds)

    for a, b in zip(
        jax.tree.leaves(fsm_state), jax.tree.leaves(ref_state), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fsm_info.is_accepted, ref_info.is_accepted)
    np.testing.assert_array_equal(fsm_info.num_expansions, ref_info.num_expansions)
    np.testing.assert_array_equal(fsm_info.num_shrink, ref_info.num_shrink)
    np.testing.assert_array_equal(fsm_info.bracket_left, ref_info.bracket_left)
    np.testing.assert_array_equal(fsm_info.bracket_right, ref_info.bracket_right)


def test_run_segment_rejects_prngs_without_map_invariant_draws():
    with pytest.raises(ValueError, match="requires Threefry PRNG keys"):
        _fsm_chain(
            jax.random.key(17, impl="rbg"),
            jnp.zeros((N_DIMS,)),
            jnp.asarray(-2.0),
            3,
            10,
            100,
        )


# Toy cache with SwiG semantics: rebuild slices refresh the cache from the
# intrinsic coordinates; cache-hit slices must reuse it exactly.
_REBUILD_BY_BLOCK = {
    (0, 1): True,
    (2,): True,
    (3,): False,
    (4, 5): False,
}
_N_DIMS = 6


def _toy_log_prior(position):
    return -0.5 * jnp.sum(position**2)


def _toy_build_cache(position):
    return jnp.tanh(position[:3]) * 2.0


def _toy_log_likelihood_from_cache(position, cache):
    return -jnp.sum((cache - 0.3) ** 2) - jnp.sum((position[3:] - 0.1) ** 2)


def _swig_builders(per_slice_info=False):
    common = {
        "log_prior_fn": _toy_log_prior,
        "build_cache": _toy_build_cache,
        "log_likelihood_from_cache_fn": _toy_log_likelihood_from_cache,
        "rebuild_required_by_block": _REBUILD_BY_BLOCK,
        "num_gibbs_sweeps": 2,
        "num_inner_steps_per_dim": 1,
        "max_steps": 10,
        "max_shrinkage": 100,
        "periodic": {4: (0.0, 2.0)},
        "n_dims": _N_DIMS,
    }
    lockstep = _build_swig_constrained_step_lockstep(**common)
    fsm = _build_swig_constrained_step(**common, per_slice_info=per_slice_info)
    return lockstep, fsm


def _block_covariances():
    return (
        jnp.asarray([[1.0, 0.35], [0.35, 0.6]]),
        jnp.asarray([[0.4]]),
        jnp.asarray([[0.7]]),
        jnp.asarray([[0.8, -0.2], [-0.2, 0.5]]),
    )


def _block_covariance_factors():
    return tuple(jnp.linalg.cholesky(cov) for cov in _block_covariances())


def _particle_state(position, loglikelihood_0):
    from blackjax.ns.base import StateWithLogLikelihood

    return StateWithLogLikelihood(
        position=position,
        logdensity=_toy_log_prior(position),
        loglikelihood=_toy_log_likelihood_from_cache(
            position, _toy_build_cache(position)
        ),
        loglikelihood_birth=jnp.asarray(loglikelihood_0),
    )


def test_swig_fsm_constrained_step_matches_lockstep_bitwise():
    lockstep, fsm = _swig_builders()
    n_lanes = 64
    keys = jax.random.split(jax.random.key(3), n_lanes)
    positions = jax.random.normal(jax.random.key(4), (n_lanes, _N_DIMS)) * 0.5
    thresholds = jnp.linspace(-30.0, -2.0, n_lanes)
    covariance_factors = _block_covariance_factors()

    def run(step_fn):
        def one(key, pos, l0):
            return step_fn(
                key,
                _particle_state(pos, l0),
                l0,
                block_covariance_factors=covariance_factors,
            )

        return jax.jit(jax.vmap(one))(keys, positions, thresholds)

    ref_state, ref_info = run(lockstep)
    new_state, new_info = run(fsm)
    for a, b in zip(
        jax.tree.leaves(new_state), jax.tree.leaves(ref_state), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(new_info.is_accepted, ref_info.is_accepted)
    np.testing.assert_array_equal(new_info.num_expansions, ref_info.num_expansions)
    np.testing.assert_array_equal(new_info.num_shrink, ref_info.num_shrink)


def test_swig_factor_direction_has_mahalanobis_norm_two():
    covariance = jnp.asarray([[1.5, 0.4], [0.4, 0.8]])
    covariance_factor = jnp.linalg.cholesky(covariance)
    direction = _sample_direction_from_covariance_factor(
        jax.random.key(27),
        jnp.zeros(2),
        covariance_factor,
    )

    whitened_direction = jnp.linalg.solve(covariance_factor, direction)
    np.testing.assert_allclose(
        whitened_direction @ whitened_direction,
        4.0,
        rtol=1e-6,
    )


def test_swig_legacy_covariance_path_matches_lockstep_bitwise():
    lockstep, fsm = _swig_builders()
    key = jax.random.key(29)
    position = jnp.linspace(-0.4, 0.4, _N_DIMS)
    threshold = jnp.asarray(-25.0)
    state = _particle_state(position, threshold)
    block_covariances = _block_covariances()

    reference_result = jax.jit(
        lambda: lockstep(
            key,
            state,
            threshold,
            block_covariances,
        )
    )()
    legacy_result = jax.jit(
        lambda: fsm(
            key,
            state,
            threshold,
            block_covariances,
        )
    )()

    for actual, expected in zip(
        jax.tree.leaves(legacy_result),
        jax.tree.leaves(reference_result),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_swig_fsm_per_slice_info_shapes_and_totals():
    _, fsm_scalar = _swig_builders(per_slice_info=False)
    _, fsm_vector = _swig_builders(per_slice_info=True)
    key = jax.random.key(11)
    pos = jnp.zeros(_N_DIMS)
    l0 = jnp.asarray(-25.0)
    covariance_factors = _block_covariance_factors()
    _, info_s = jax.jit(
        lambda: fsm_scalar(
            key,
            _particle_state(pos, l0),
            l0,
            block_covariance_factors=covariance_factors,
        )
    )()
    _, info_v = jax.jit(
        lambda: fsm_vector(
            key,
            _particle_state(pos, l0),
            l0,
            block_covariance_factors=covariance_factors,
        )
    )()
    n_slices_total = 2 * (2 + 1 + 1 + 2)
    assert info_v.num_expansions.shape == (n_slices_total,)
    assert info_v.num_shrink.shape == (n_slices_total,)
    np.testing.assert_array_equal(info_v.num_expansions.sum(), info_s.num_expansions)
    np.testing.assert_array_equal(info_v.num_shrink.sum(), info_s.num_shrink)


def test_swig_fsm_hit_segments_never_reference_build_cache():
    marker_calls = []

    def marked_build_cache(position):
        marker_calls.append(None)
        return _toy_build_cache(position)

    fsm = _build_swig_constrained_step(
        log_prior_fn=_toy_log_prior,
        build_cache=marked_build_cache,
        log_likelihood_from_cache_fn=_toy_log_likelihood_from_cache,
        rebuild_required_by_block=_REBUILD_BY_BLOCK,
        num_gibbs_sweeps=2,
        num_inner_steps_per_dim=1,
        max_steps=10,
        max_shrinkage=100,
        periodic=None,
        n_dims=_N_DIMS,
    )
    key = jax.random.key(0)
    l0 = jnp.asarray(-25.0)
    covariance_factors = _block_covariance_factors()
    jax.make_jaxpr(
        lambda: fsm(
            key,
            _particle_state(jnp.zeros(_N_DIMS), l0),
            l0,
            block_covariance_factors=covariance_factors,
        )
    )()
    # Two sweeps x one rebuild segment per sweep, plus the chain-entry build.
    assert len(marker_calls) == 3


def test_swig_fsm_lowers_to_one_while_per_segment():
    _, fsm = _swig_builders()
    n_lanes = 16
    keys = jax.random.split(jax.random.key(0), n_lanes)
    positions = jnp.zeros((n_lanes, _N_DIMS))
    thresholds = jnp.full((n_lanes,), -25.0)
    covariance_factors = _block_covariance_factors()

    def batched(keys, positions, thresholds):
        return jax.vmap(
            lambda key, position, threshold: fsm(
                key,
                _particle_state(position, threshold),
                threshold,
                block_covariance_factors=covariance_factors,
            )
        )(keys, positions, thresholds)

    text = jax.jit(batched).lower(keys, positions, thresholds).as_text()
    main_text = text.split("func.func private", 1)[0]
    # Two sweeps x (one rebuild segment + one cache-hit segment). Incidental
    # PRNG and linear-algebra loops lower into private helper functions.
    assert main_text.count("stablehlo.while") == 4


def test_swig_fsm_does_not_factor_covariance_inside_slice_path():
    _, fsm = _swig_builders()
    key = jax.random.key(31)
    threshold = jnp.asarray(-25.0)
    state = _particle_state(jnp.zeros(_N_DIMS), threshold)
    covariance_factors = _block_covariance_factors()

    jaxpr = jax.make_jaxpr(
        lambda: fsm(
            key,
            state,
            threshold,
            block_covariance_factors=covariance_factors,
        )
    )()

    assert "cholesky" not in str(jaxpr).lower()


_NSS_PERIODIC = {1: (0.0, 2.0)}
_NSS_DIMS = 3


def _toy_nss_loglik(position):
    return -jnp.sum((position - 0.15) ** 2) * 4.0


def _build_nss_factor_reference_step():
    init_state_fn = partial(
        _init_state_strategy,
        logprior_fn=_toy_log_prior,
        loglikelihood_fn=_toy_nss_loglik,
    )
    kernel = build_slice_kernel(
        interval=stepping_out_cached,
        max_expansions=10,
        max_shrinkage=100,
    )
    proposal = to_prior_space_proposal(
        _NSS_PERIODIC,
        _NSS_DIMS,
        _sample_direction_from_covariance_factor,
    )
    return slice_constrained_step(init_state_fn, kernel, proposal)


def _nss_reference_chain(
    step,
    chain_key,
    state,
    loglikelihood_0,
    covariance_factor,
    n_inner_steps,
):
    keys = jax.random.split(chain_key, n_inner_steps)

    def body(carry, key):
        return step(key, carry, loglikelihood_0, cov=covariance_factor)

    return jax.lax.scan(body, state, keys)


def test_nss_fsm_constrained_step_matches_factor_scan_bitwise():
    n_lanes, n_inner_steps = 512, 8
    reference_step = _build_nss_factor_reference_step()
    fsm_step = _build_nss_fsm_constrained_step(
        log_prior_fn=_toy_log_prior,
        log_likelihood_fn=_toy_nss_loglik,
        periodic=_NSS_PERIODIC,
        n_dims=_NSS_DIMS,
    )
    chain_keys = jax.random.split(jax.random.key(21), n_lanes)
    positions = jax.random.normal(jax.random.key(22), (n_lanes, _NSS_DIMS)) * 0.4
    thresholds = jnp.linspace(-20.0, -1.0, n_lanes)
    covariance = jnp.asarray(
        [
            [0.5, 0.12, -0.04],
            [0.12, 0.35, 0.08],
            [-0.04, 0.08, 0.25],
        ]
    )
    covariance_factor = jnp.linalg.cholesky(covariance)

    def particle(position, loglikelihood_0):
        return StateWithLogLikelihood(
            position=position,
            logdensity=_toy_log_prior(position),
            loglikelihood=_toy_nss_loglik(position),
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
        )

    ref_state, ref_info = jax.jit(
        jax.vmap(
            lambda key, position, threshold: _nss_reference_chain(
                reference_step,
                key,
                particle(position, threshold),
                threshold,
                covariance_factor,
                n_inner_steps,
            )
        )
    )(chain_keys, positions, thresholds)
    fsm_state, fsm_info = jax.jit(
        jax.vmap(
            lambda key, position, threshold: fsm_step(
                jax.random.split(key, n_inner_steps),
                particle(position, threshold),
                threshold,
                covariance_factor=covariance_factor,
            )
        )
    )(chain_keys, positions, thresholds)

    for actual, expected in zip(
        jax.tree.leaves(fsm_state),
        jax.tree.leaves(ref_state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        jax.tree.leaves(fsm_info),
        jax.tree.leaves(ref_info),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_nss_fsm_legacy_covariance_matches_precomputed_factor():
    n_inner_steps = 5
    fsm_step = _build_nss_fsm_constrained_step(
        log_prior_fn=_toy_log_prior,
        log_likelihood_fn=_toy_nss_loglik,
        periodic=_NSS_PERIODIC,
        n_dims=_NSS_DIMS,
    )
    covariance = jnp.asarray(
        [
            [0.5, 0.12, -0.04],
            [0.12, 0.35, 0.08],
            [-0.04, 0.08, 0.25],
        ]
    )
    covariance_factor = jnp.linalg.cholesky(covariance)
    keys = jax.random.split(jax.random.key(33), n_inner_steps)
    threshold = jnp.asarray(-12.0)
    state = StateWithLogLikelihood(
        position=jnp.asarray([0.1, 0.2, 0.3]),
        logdensity=_toy_log_prior(jnp.asarray([0.1, 0.2, 0.3])),
        loglikelihood=_toy_nss_loglik(jnp.asarray([0.1, 0.2, 0.3])),
        loglikelihood_birth=threshold,
    )

    factor_result = jax.jit(
        lambda: fsm_step(
            keys,
            state,
            threshold,
            covariance_factor=covariance_factor,
        )
    )()
    legacy_result = jax.jit(lambda: fsm_step(keys, state, threshold, covariance))()

    for actual, expected in zip(
        jax.tree.leaves(legacy_result),
        jax.tree.leaves(factor_result),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_nss_fsm_does_not_factor_covariance_inside_slice_path():
    fsm_step = _build_nss_fsm_constrained_step(
        log_prior_fn=_toy_log_prior,
        log_likelihood_fn=_toy_nss_loglik,
        periodic=_NSS_PERIODIC,
        n_dims=_NSS_DIMS,
    )
    covariance_factor = jnp.linalg.cholesky(jnp.eye(_NSS_DIMS) * 0.3)
    keys = jax.random.split(jax.random.key(35), 5)
    threshold = jnp.asarray(-12.0)
    position = jnp.asarray([0.1, 0.2, 0.3])
    state = StateWithLogLikelihood(
        position=position,
        logdensity=_toy_log_prior(position),
        loglikelihood=_toy_nss_loglik(position),
        loglikelihood_birth=threshold,
    )

    jaxpr = jax.make_jaxpr(
        lambda: fsm_step(
            keys,
            state,
            threshold,
            covariance_factor=covariance_factor,
        )
    )()

    assert "cholesky" not in str(jaxpr).lower()


def test_replicated_update_can_fold_inner_steps_without_changing_info_shape():
    n_inner_steps = 4
    mesh = Mesh(np.asarray(jax.devices()[:1]), axis_names=("replacement",))
    particles = StateWithLogLikelihood(
        position=jnp.arange(4.0).reshape(2, 2),
        logdensity=jnp.zeros(2),
        loglikelihood=jnp.arange(1.0, 3.0),
        loglikelihood_birth=jnp.full(2, -jnp.inf),
    )

    def folded_step(keys, state, loglikelihood_0):
        del loglikelihood_0
        per_slice = jnp.arange(keys.shape[0])
        return state, SliceInfo(
            is_accepted=jnp.ones(keys.shape[0], dtype=bool),
            num_expansions=per_slice,
            num_shrink=per_slice + 1,
            bracket_left=-per_slice,
            bracket_right=per_slice,
        )

    update = update_with_mcmc_take_last_replicated(
        folded_step,
        n_inner_steps=n_inner_steps,
        n_delete=1,
        mesh=mesh,
        fold_inner_steps=True,
    )
    _, info = update(jax.random.key(0), NSState(particles), jnp.asarray(0.0))

    for leaf in jax.tree.leaves(info):
        assert leaf.shape == (1, n_inner_steps)
