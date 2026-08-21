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
    SegmentInfo,
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
    _COMPLEMENTARY_DE_KEY_DOMAIN,
    CachedSliceState,
    _apply_complementary_de_proposals,
    _build_swig_constrained_step,
    _build_swig_constrained_step_lockstep,
    _ComplementaryDEProposalSchedule,
    _prepare_complementary_de_proposals,
    _sample_signed_permuted_covariance_basis,
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


def _swig_builders(per_slice_info=False, num_slice_steps_by_block=None):
    common = {
        "log_prior_fn": _toy_log_prior,
        "build_cache": _toy_build_cache,
        "log_likelihood_from_cache_fn": _toy_log_likelihood_from_cache,
        "rebuild_required_by_block": _REBUILD_BY_BLOCK,
        "num_gibbs_sweeps": 2,
        "num_inner_steps_per_dim": 1,
        "num_slice_steps_by_block": num_slice_steps_by_block,
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


def test_swig_covariance_basis_uses_each_signed_factor_column_once():
    covariance_factor = jnp.asarray(
        [
            [1.2, 0.0, 0.0],
            [0.3, 0.8, 0.0],
            [-0.2, 0.1, 0.5],
        ]
    )
    directions = _sample_signed_permuted_covariance_basis(
        jax.random.key(41), covariance_factor
    )

    whitened = jnp.linalg.solve(covariance_factor, directions.T).T
    np.testing.assert_array_equal(jnp.sum(jnp.abs(whitened) > 0.0, axis=1), 1)
    np.testing.assert_array_equal(jnp.sum(jnp.abs(whitened) > 0.0, axis=0), 1)
    np.testing.assert_allclose(jnp.max(jnp.abs(whitened), axis=1), 2.0)


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


def test_swig_default_block_budget_matches_the_legacy_dimension_budget_bitwise():
    _, default_budget = _swig_builders()
    _, explicit_legacy_budget = _swig_builders(num_slice_steps_by_block=(2, 1, 1, 2))
    key = jax.random.key(32)
    position = jnp.linspace(-0.4, 0.4, _N_DIMS)
    threshold = jnp.asarray(-25.0)
    state = _particle_state(position, threshold)
    covariance_factors = _block_covariance_factors()

    def run(step):
        return jax.jit(
            lambda: step(
                key,
                state,
                threshold,
                block_covariance_factors=covariance_factors,
            )
        )()

    expected = run(default_budget)
    actual = run(explicit_legacy_budget)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


_H2_REBUILD_BY_BLOCK = {
    tuple(range(8)): True,
    (8,): True,
    (9,): True,
    (10, 11): False,
    (12,): False,
    (13, 14): False,
}
_H2_N_DIMS = 15


def _h2_builders(
    *,
    per_slice_info=False,
    direction_mode="covariance-basis-8d",
    num_slice_steps_by_block=None,
    build_cache=None,
    log_likelihood_from_cache_fn=None,
    block_kernel_modes=None,
    periodic=None,
    resolved_complementary_de_jump_block=None,
):
    build_cache = build_cache or (lambda position: jnp.tanh(position[:10]))
    log_likelihood_from_cache_fn = log_likelihood_from_cache_fn or (
        lambda position, cache: (
            -jnp.sum((cache - 0.2) ** 2) - jnp.sum((position[10:] + 0.1) ** 2)
        )
    )
    common = {
        "log_prior_fn": _toy_log_prior,
        "build_cache": build_cache,
        "log_likelihood_from_cache_fn": log_likelihood_from_cache_fn,
        "rebuild_required_by_block": _H2_REBUILD_BY_BLOCK,
        "num_gibbs_sweeps": 1,
        "num_inner_steps_per_dim": 1,
        "num_slice_steps_by_block": num_slice_steps_by_block,
        "max_steps": 4,
        "max_shrinkage": 30,
        "periodic": periodic,
        "n_dims": _H2_N_DIMS,
        "direction_mode": direction_mode,
        "block_kernel_modes": block_kernel_modes,
        "resolved_complementary_de_jump_block": (resolved_complementary_de_jump_block),
    }
    return (
        _build_swig_constrained_step_lockstep(**common),
        _build_swig_constrained_step(**common, per_slice_info=per_slice_info),
    )


@pytest.mark.parametrize("attempts", [4, 8], ids=["h5-cde4", "h6-cde8"])
def test_swig_complementary_de_uses_fixed_strict_parent_excluded_complement(attempts):
    modes = (
        "slice",
        "periodic-uniform-independence",
        "periodic-uniform-independence",
        "slice",
        "periodic-uniform-independence",
        "slice",
    )
    _, fsm = _h2_builders(
        direction_mode="covariance",
        block_kernel_modes=modes,
        periodic={8: (-jnp.pi, jnp.pi), 9: (-jnp.pi, jnp.pi), 12: (0.0, jnp.pi)},
        resolved_complementary_de_jump_block=(tuple(range(8)), True, attempts),
    )
    threshold = jnp.asarray(-20.0)
    live_positions = jax.random.normal(jax.random.key(74), (12, _H2_N_DIMS)) * 0.05
    live_loglikelihoods = jnp.ones((12,))
    live_loglikelihoods = live_loglikelihoods.at[jnp.asarray([1, 7])].set(threshold)
    parent_index = jnp.asarray(4, dtype=jnp.int32)

    _, info = jax.jit(
        lambda: fsm(
            jax.random.key(75),
            _h2_particle_state(live_positions[parent_index], threshold),
            threshold,
            block_covariance_factors=_h2_covariance_factors(),
            live_positions=live_positions,
            live_loglikelihoods=live_loglikelihoods,
            parent_index=parent_index,
        )
    )()

    donors = np.asarray(info.complementary_de_donor_indices_by_attempt)
    assert donors.shape == (attempts, 2)
    assert np.all(donors[:, 0] != donors[:, 1])
    assert not np.any(donors == int(parent_index))
    assert np.all(np.asarray(live_loglikelihoods)[donors] > float(threshold))
    np.testing.assert_array_equal(
        info.complementary_de_donor_policy_violations_by_attempt,
        np.zeros((attempts,), dtype=bool),
    )
    np.testing.assert_array_equal(info.complementary_de_complement_size, 9)
    np.testing.assert_array_equal(info.complementary_de_parent_index, parent_index)
    np.testing.assert_array_equal(info.num_complementary_de_attempts, attempts)


@pytest.mark.parametrize("attempts", [1, 2, 3, 5, 6, 7, 9])
def test_swig_complementary_de_builders_reject_unregistered_attempt_budgets(attempts):
    modes = (
        "slice",
        "periodic-uniform-independence",
        "periodic-uniform-independence",
        "slice",
        "periodic-uniform-independence",
        "slice",
    )

    with pytest.raises(ValueError, match="exactly four or eight attempts"):
        _h2_builders(
            direction_mode="covariance",
            block_kernel_modes=modes,
            periodic={
                8: (-jnp.pi, jnp.pi),
                9: (-jnp.pi, jnp.pi),
                12: (0.0, jnp.pi),
            },
            resolved_complementary_de_jump_block=(
                tuple(range(8)),
                True,
                attempts,
            ),
        )


def test_complementary_de_redraws_ordered_pairs_from_one_frozen_complement():
    key = jax.random.key(78)
    live_positions = jnp.arange(90, dtype=jnp.float32).reshape(10, 9) / 100.0
    threshold = jnp.asarray(0.0)
    live_loglikelihoods = jnp.asarray(
        [1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )
    parent_index = jnp.asarray(5, dtype=jnp.int32)
    schedule = _prepare_complementary_de_proposals(
        key,
        live_positions=live_positions,
        live_loglikelihoods=live_loglikelihoods,
        loglikelihood_0=threshold,
        parent_index=parent_index,
        parameter_indices=tuple(range(8)),
        attempts=4,
    )

    eligible = (live_loglikelihoods > threshold) & (
        jnp.arange(live_positions.shape[0]) != parent_index
    )
    expected_pairs = []
    expected_key_data = []
    for attempt_index in range(4):
        operation_key = jax.random.fold_in(
            key, _COMPLEMENTARY_DE_KEY_DOMAIN + attempt_index
        )
        pair_key, _ = jax.random.split(operation_key)
        expected_pairs.append(
            jax.random.choice(
                pair_key,
                live_positions.shape[0],
                shape=(2,),
                replace=False,
                p=eligible.astype(jnp.float32) / eligible.sum(),
            )
        )
        expected_key_data.append(jax.random.key_data(operation_key))

    expected_pairs = jnp.stack(expected_pairs)
    np.testing.assert_array_equal(schedule.donor_indices, expected_pairs)
    np.testing.assert_array_equal(
        schedule.operation_key_data, jnp.stack(expected_key_data)
    )
    assert np.unique(np.asarray(schedule.operation_key_data), axis=0).shape[0] == 4
    for attempt_index, (donor_a, donor_b) in enumerate(np.asarray(expected_pairs)):
        expected_displacement = np.zeros((9,), dtype=np.float32)
        expected_displacement[:8] = np.asarray(
            live_positions[donor_a, :8] - live_positions[donor_b, :8]
        )
        np.testing.assert_array_equal(
            schedule.displacement[attempt_index], expected_displacement
        )
        reverse_displacement = np.asarray(
            live_positions[donor_b, :8] - live_positions[donor_a, :8]
        )
        np.testing.assert_array_equal(reverse_displacement, -expected_displacement[:8])


def test_eight_attempt_complementary_de_appends_to_four_attempt_schedule():
    key = jax.random.key(780)
    live_positions = jnp.arange(108, dtype=jnp.float32).reshape(12, 9) / 100.0
    threshold = jnp.asarray(0.0)
    live_loglikelihoods = jnp.asarray(
        [1.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )
    kwargs = {
        "live_positions": live_positions,
        "live_loglikelihoods": live_loglikelihoods,
        "loglikelihood_0": threshold,
        "parent_index": jnp.asarray(5, dtype=jnp.int32),
        "parameter_indices": tuple(range(8)),
    }

    four = _prepare_complementary_de_proposals(key, attempts=4, **kwargs)
    eight = _prepare_complementary_de_proposals(key, attempts=8, **kwargs)

    for four_values, eight_values in (
        (four.operation_key_data, eight.operation_key_data),
        (four.donor_indices, eight.donor_indices),
        (four.donor_policy_violations, eight.donor_policy_violations),
        (four.displacement, eight.displacement),
    ):
        np.testing.assert_array_equal(eight_values[:4], four_values)
    np.testing.assert_array_equal(eight.complement_size, four.complement_size)
    assert eight.operation_key_data.shape == (8, 2)
    assert eight.donor_indices.shape == (8, 2)
    assert eight.displacement.shape == (8, 9)
    expected_appended_keys = jnp.stack(
        [
            jax.random.key_data(
                jax.random.fold_in(key, _COMPLEMENTARY_DE_KEY_DOMAIN + index)
            )
            for index in range(4, 8)
        ]
    )
    np.testing.assert_array_equal(eight.operation_key_data[4:], expected_appended_keys)


def test_complementary_de_fails_closed_when_strict_complement_is_too_small():
    live_positions = jnp.arange(36, dtype=jnp.float32).reshape(4, 9) / 100.0
    threshold = jnp.asarray(1.0)
    schedule = _prepare_complementary_de_proposals(
        jax.random.key(79),
        live_positions=live_positions,
        live_loglikelihoods=jnp.asarray([1.0, 2.0, 1.0, 0.0]),
        loglikelihood_0=threshold,
        parent_index=jnp.asarray(0, dtype=jnp.int32),
        parameter_indices=tuple(range(8)),
        attempts=4,
    )

    np.testing.assert_array_equal(schedule.complement_size, 1)
    np.testing.assert_array_equal(
        schedule.donor_indices, np.full((4, 2), -1, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        schedule.donor_policy_violations, np.ones((4,), dtype=bool)
    )
    np.testing.assert_array_equal(schedule.displacement, np.zeros((4, 9)))


def test_rejected_complementary_de_restores_h4_endpoint_and_cache_bitwise():
    position = jnp.linspace(0.1, 0.9, 9)
    initial = CachedSliceState(
        position=position,
        logdensity=jnp.asarray(0.0),
        loglikelihood=jnp.asarray(1.0),
        loglikelihood_birth=jnp.asarray(0.0),
        cache=position[:8] ** 2,
    )
    schedule = _ComplementaryDEProposalSchedule(
        operation_key_data=jnp.stack(
            [
                jax.random.key_data(jax.random.fold_in(jax.random.key(80), index))
                for index in range(4)
            ]
        ),
        displacement=jnp.full((4, 9), 0.01),
        donor_indices=jnp.asarray([[1, 2], [2, 1], [3, 4], [4, 3]]),
        donor_policy_violations=jnp.zeros((4,), dtype=bool),
        complement_size=jnp.asarray(8, dtype=jnp.int32),
    )

    def tied_candidate(candidate, cache):
        del cache
        return jnp.asarray(0.0), jnp.asarray(0.0), candidate[:8] ** 2

    actual, info = _apply_complementary_de_proposals(
        schedule,
        initial,
        jnp.asarray(0.0),
        eval_candidate=tied_candidate,
        wrap_position=lambda candidate: candidate,
    )

    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(initial), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    np.testing.assert_array_equal(info.acceptances, np.zeros((4,), dtype=bool))


def test_accepted_complementary_de_cache_matches_rebuild_and_next_cache_hit():
    position = jnp.linspace(0.1, 0.5, 9)
    initial = CachedSliceState(
        position=position,
        logdensity=jnp.asarray(0.0),
        loglikelihood=jnp.asarray(-1.0),
        loglikelihood_birth=jnp.asarray(-10.0),
        cache=position[:8] ** 2,
    )
    displacements = jnp.zeros((4, 9)).at[:, 0].set(0.01)
    schedule = _ComplementaryDEProposalSchedule(
        operation_key_data=jnp.stack(
            [
                jax.random.key_data(jax.random.fold_in(jax.random.key(81), index))
                for index in range(4)
            ]
        ),
        displacement=displacements,
        donor_indices=jnp.asarray([[1, 2], [2, 1], [3, 4], [4, 3]]),
        donor_policy_violations=jnp.zeros((4,), dtype=bool),
        complement_size=jnp.asarray(8, dtype=jnp.int32),
    )
    evaluations = []

    def rebuild_candidate(candidate, cache):
        del cache
        evaluations.append(candidate)
        rebuilt = candidate[:8] ** 2
        return jnp.asarray(0.0), -jnp.sum((rebuilt - 0.25) ** 2), rebuilt

    actual, info = _apply_complementary_de_proposals(
        schedule,
        initial,
        jnp.asarray(-10.0),
        eval_candidate=rebuild_candidate,
        wrap_position=lambda candidate: candidate,
    )

    expected_cache = actual.position[:8] ** 2
    assert len(evaluations) == 4
    np.testing.assert_array_equal(info.acceptances, np.ones((4,), dtype=bool))
    np.testing.assert_array_equal(actual.cache, expected_cache)
    np.testing.assert_array_equal(
        actual.loglikelihood, -jnp.sum((expected_cache - 0.25) ** 2)
    )
    next_position = actual.position.at[8].add(0.125)
    cache_hit_loglikelihood = (
        -jnp.sum((actual.cache - 0.25) ** 2) - next_position[8] ** 2
    )
    rebuilt_loglikelihood = (
        -jnp.sum((next_position[:8] ** 2 - 0.25) ** 2) - next_position[8] ** 2
    )
    np.testing.assert_array_equal(cache_hit_loglikelihood, rebuilt_loglikelihood)


def test_complementary_de_mh_uses_full_coupled_prior_ratio():
    position = jnp.zeros((9,)).at[8].set(1.0)
    initial = CachedSliceState(
        position=position,
        logdensity=jnp.asarray(0.0),
        loglikelihood=jnp.asarray(0.0),
        loglikelihood_birth=jnp.asarray(-1.0),
        cache=position[:8] ** 2,
    )
    operation_key = jax.random.key(2)
    schedule = _ComplementaryDEProposalSchedule(
        operation_key_data=jax.random.key_data(operation_key)[None, :],
        displacement=jnp.zeros((1, 9)).at[0, 0].set(0.1),
        donor_indices=jnp.asarray([[1, 2]]),
        donor_policy_violations=jnp.zeros((1,), dtype=bool),
        complement_size=jnp.asarray(8, dtype=jnp.int32),
    )

    def coupled_prior_candidate(candidate, cache):
        return -candidate[0] * candidate[8], jnp.asarray(0.0), cache

    actual, info = _apply_complementary_de_proposals(
        schedule,
        initial,
        jnp.asarray(-1.0),
        eval_candidate=coupled_prior_candidate,
        wrap_position=lambda candidate: candidate,
    )

    _, accept_key = jax.random.split(operation_key)
    expected_acceptance = jnp.log(jax.random.uniform(accept_key)) < -0.1
    np.testing.assert_array_equal(expected_acceptance, False)
    np.testing.assert_array_equal(info.acceptances, expected_acceptance[None])
    np.testing.assert_array_equal(actual.position, initial.position)


@pytest.mark.parametrize("attempts", [4, 8], ids=["h5-cde4", "h6-cde8"])
def test_swig_complementary_de_fsm_matches_lockstep_pathwise(attempts):
    modes = (
        "slice",
        "periodic-uniform-independence",
        "periodic-uniform-independence",
        "slice",
        "periodic-uniform-independence",
        "slice",
    )
    lockstep, fsm = _h2_builders(
        direction_mode="covariance",
        block_kernel_modes=modes,
        periodic={8: (-jnp.pi, jnp.pi), 9: (-jnp.pi, jnp.pi), 12: (0.0, jnp.pi)},
        resolved_complementary_de_jump_block=(tuple(range(8)), True, attempts),
    )
    live_positions = jax.random.normal(jax.random.key(76), (16, _H2_N_DIMS)) * 0.05
    live_loglikelihoods = jnp.ones((16,))
    parent_indices = jnp.asarray([0, 2, 4, 6, 8, 10, 12, 14], dtype=jnp.int32)
    keys = jax.random.split(jax.random.key(77), parent_indices.size)
    thresholds = jnp.full((parent_indices.size,), -20.0)
    factors = _h2_covariance_factors()

    def run(step):
        return jax.jit(
            jax.vmap(
                lambda key, parent_index, threshold: step(
                    key,
                    _h2_particle_state(live_positions[parent_index], threshold),
                    threshold,
                    block_covariance_factors=factors,
                    live_positions=live_positions,
                    live_loglikelihoods=live_loglikelihoods,
                    parent_index=parent_index,
                )
            )
        )(keys, parent_indices, thresholds)

    expected = run(lockstep)
    actual = run(fsm)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_forced_reject_h5_reproduces_h4_endpoint_bitwise():
    modes = (
        "slice",
        "periodic-uniform-independence",
        "periodic-uniform-independence",
        "slice",
        "periodic-uniform-independence",
        "slice",
    )
    common = {
        "direction_mode": "covariance",
        "block_kernel_modes": modes,
        "periodic": {
            8: (-jnp.pi, jnp.pi),
            9: (-jnp.pi, jnp.pi),
            12: (0.0, jnp.pi),
        },
    }
    _, h4 = _h2_builders(**common)
    _, h5 = _h2_builders(
        **common,
        resolved_complementary_de_jump_block=(tuple(range(8)), True, 4),
    )
    position = jnp.linspace(-0.1, 0.1, _H2_N_DIMS)
    threshold = jnp.asarray(-20.0)
    live_positions = jnp.broadcast_to(position, (12, _H2_N_DIMS))
    live_positions = live_positions.at[:, 0].add(
        100.0 * jnp.arange(12, dtype=position.dtype)
    )
    live_loglikelihoods = jnp.ones((12,))
    parent_index = jnp.asarray(0, dtype=jnp.int32)
    state = _h2_particle_state(position, threshold)
    factors = _h2_covariance_factors()
    key = jax.random.key(83)

    def run(step):
        return jax.jit(
            lambda: step(
                key,
                state,
                threshold,
                block_covariance_factors=factors,
                live_positions=live_positions,
                live_loglikelihoods=live_loglikelihoods,
                parent_index=parent_index,
            )
        )()

    h4_state, h4_info = run(h4)
    h5_state, h5_info = run(h5)
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(h4_state), jax.tree.leaves(h5_state), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
    for field in (
        "is_accepted",
        "num_expansions",
        "num_shrink",
        "num_periodic_uniform_independence_attempts",
        "num_periodic_uniform_independence_acceptances",
        "num_periodic_uniform_independence_attempts_by_block",
        "num_periodic_uniform_independence_acceptances_by_block",
    ):
        np.testing.assert_array_equal(getattr(h5_info, field), getattr(h4_info, field))
    np.testing.assert_array_equal(
        h5_info.complementary_de_acceptances_by_attempt,
        np.zeros((4,), dtype=bool),
    )


def _h2_covariance_factors():
    return tuple(
        jnp.diag(jnp.linspace(0.4, 1.1, len(indices)))
        for indices in _H2_REBUILD_BY_BLOCK
    )


def _h2_particle_state(position, threshold):
    cache = jnp.tanh(position[:10])
    return StateWithLogLikelihood(
        position=position,
        logdensity=_toy_log_prior(position),
        loglikelihood=-jnp.sum((cache - 0.2) ** 2)
        - jnp.sum((position[10:] + 0.1) ** 2),
        loglikelihood_birth=jnp.asarray(threshold),
    )


def test_swig_covariance_basis_fsm_matches_lockstep_bitwise():
    lockstep, fsm = _h2_builders()
    keys = jax.random.split(jax.random.key(43), 8)
    positions = jax.random.normal(jax.random.key(44), (8, _H2_N_DIMS)) * 0.2
    thresholds = jnp.linspace(-40.0, -3.0, 8)
    factors = _h2_covariance_factors()

    def run(step):
        return jax.jit(
            jax.vmap(
                lambda key, position, threshold: step(
                    key,
                    _h2_particle_state(position, threshold),
                    threshold,
                    block_covariance_factors=factors,
                )
            )
        )(keys, positions, thresholds)

    expected = run(lockstep)
    actual = run(fsm)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_swig_explicit_block_budget_fsm_matches_lockstep_bitwise():
    lockstep, fsm = _h2_builders(
        direction_mode="covariance",
        num_slice_steps_by_block=(5, 1, 1, 2, 1, 2),
    )
    keys = jax.random.split(jax.random.key(48), 8)
    positions = jax.random.normal(jax.random.key(49), (8, _H2_N_DIMS)) * 0.2
    thresholds = jnp.linspace(-40.0, -3.0, 8)
    factors = _h2_covariance_factors()

    def run(step):
        return jax.jit(
            jax.vmap(
                lambda key, position, threshold: step(
                    key,
                    _h2_particle_state(position, threshold),
                    threshold,
                    block_covariance_factors=factors,
                )
            )
        )(keys, positions, thresholds)

    expected = run(lockstep)
    actual = run(fsm)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_swig_periodic_uniform_independence_fsm_matches_lockstep_pathwise():
    modes = (
        "slice",
        "periodic-uniform-independence",
        "periodic-uniform-independence",
        "slice",
        "periodic-uniform-independence",
        "slice",
    )
    lockstep, fsm = _h2_builders(
        direction_mode="covariance",
        block_kernel_modes=modes,
        periodic={8: (-jnp.pi, jnp.pi), 9: (-jnp.pi, jnp.pi), 12: (0.0, jnp.pi)},
    )
    keys = jax.random.split(jax.random.key(52), 16)
    positions = jax.random.normal(jax.random.key(53), (16, _H2_N_DIMS)) * 0.2
    thresholds = jnp.linspace(-40.0, -3.0, 16)
    factors = _h2_covariance_factors()

    def run(step):
        return jax.jit(
            jax.vmap(
                lambda key, position, threshold: step(
                    key,
                    _h2_particle_state(position, threshold),
                    threshold,
                    block_covariance_factors=factors,
                )
            )
        )(keys, positions, thresholds)

    expected = run(lockstep)
    actual = run(fsm)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


@pytest.mark.parametrize(
    "complementary_de_attempts",
    [None, 4, 8],
    ids=["h4", "h5-cde4", "h6-cde8"],
)
def test_swig_periodic_uniform_independence_keeps_two_segments_and_12_slices(
    complementary_de_attempts,
):
    modes = (
        "slice",
        "periodic-uniform-independence",
        "periodic-uniform-independence",
        "slice",
        "periodic-uniform-independence",
        "slice",
    )
    _, fsm = _h2_builders(
        per_slice_info=True,
        direction_mode="covariance",
        block_kernel_modes=modes,
        periodic={8: (-jnp.pi, jnp.pi), 9: (-jnp.pi, jnp.pi), 12: (0.0, jnp.pi)},
        resolved_complementary_de_jump_block=(
            (tuple(range(8)), True, complementary_de_attempts)
            if complementary_de_attempts is not None
            else None
        ),
    )
    keys = jax.random.split(jax.random.key(54), 4)
    live_positions = jax.random.normal(jax.random.key(55), (16, _H2_N_DIMS)) * 0.2
    positions = live_positions[:4]
    live_loglikelihoods = jnp.ones((16,))
    parent_indices = jnp.arange(4, dtype=jnp.int32)
    thresholds = jnp.linspace(-20.0, -3.0, 4)
    factors = _h2_covariance_factors()

    def batched(keys, positions, thresholds, parent_indices):
        return jax.vmap(
            lambda key, position, threshold, parent_index: fsm(
                key,
                _h2_particle_state(position, threshold),
                threshold,
                block_covariance_factors=factors,
                live_positions=live_positions,
                live_loglikelihoods=live_loglikelihoods,
                parent_index=parent_index,
            )
        )(keys, positions, thresholds, parent_indices)

    lowered = jax.jit(batched).lower(keys, positions, thresholds, parent_indices)
    _, info = lowered.compile()(keys, positions, thresholds, parent_indices)
    assert info.num_expansions.shape == (4, 12)
    assert info.num_shrink.shape == (4, 12)
    np.testing.assert_array_equal(
        info.num_periodic_uniform_independence_attempts,
        jnp.full((4,), 3),
    )
    total_updates = info.num_expansions.shape[1] + int(
        np.asarray(info.num_periodic_uniform_independence_attempts)[0]
    )
    if complementary_de_attempts is not None:
        np.testing.assert_array_equal(
            info.num_complementary_de_attempts,
            jnp.full((4,), complementary_de_attempts),
        )
        total_updates += int(np.asarray(info.num_complementary_de_attempts)[0])
    assert total_updates == (
        15 if complementary_de_attempts is None else 15 + complementary_de_attempts
    )
    if complementary_de_attempts == 8:
        assert total_updates == 23
    main_text = lowered.as_text().split("func.func private", 1)[0]
    assert main_text.count("stablehlo.while") == 2


def test_swig_explicit_all_slice_modes_match_no_feature_bitwise():
    _, default_step = _h2_builders()
    _, explicit_step = _h2_builders(block_kernel_modes=("slice",) * 6)
    key = jax.random.key(56)
    threshold = jnp.asarray(-10.0)
    position = jnp.linspace(-0.2, 0.2, _H2_N_DIMS)
    state = _h2_particle_state(position, threshold)
    factors = _h2_covariance_factors()

    expected = jax.jit(
        lambda: default_step(
            key,
            state,
            threshold,
            block_covariance_factors=factors,
        )
    )()
    actual = jax.jit(
        lambda: explicit_step(
            key,
            state,
            threshold,
            block_covariance_factors=factors,
        )
    )()
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_swig_periodic_uniform_independence_fsm_calls_likelihood_once():
    evaluations = []

    def log_likelihood_from_cache(position, cache):
        del cache
        jax.debug.callback(lambda value: evaluations.append(float(value)), position[0])
        return jnp.asarray(0.0)

    step = _build_swig_constrained_step(
        log_prior_fn=lambda position: jnp.log(2.0 * position[0]),
        build_cache=lambda position: position[0],
        log_likelihood_from_cache_fn=log_likelihood_from_cache,
        rebuild_required_by_block={(0,): False},
        num_gibbs_sweeps=1,
        num_inner_steps_per_dim=1,
        max_steps=4,
        max_shrinkage=30,
        periodic={0: (0.0, 1.0)},
        n_dims=1,
        block_kernel_modes=("periodic-uniform-independence",),
    )
    position = jnp.asarray([0.5])
    state = StateWithLogLikelihood(
        position=position,
        logdensity=jnp.log(jnp.asarray(1.0)),
        loglikelihood=jnp.asarray(0.0),
        loglikelihood_birth=jnp.asarray(-1.0),
    )
    _, info = jax.jit(
        lambda: step(
            jax.random.key(57),
            state,
            jnp.asarray(-1.0),
            block_covariance_factors=(jnp.ones((1, 1)),),
        )
    )()
    jax.block_until_ready(info.num_periodic_uniform_independence_acceptances)

    assert len(evaluations) == 1
    np.testing.assert_array_equal(info.num_periodic_uniform_independence_attempts, 1)
    np.testing.assert_array_equal(info.num_expansions, 0)
    np.testing.assert_array_equal(info.num_shrink, 0)


def test_swig_covariance_basis_changes_only_the_8d_block(monkeypatch):
    captured = []

    def capture_segment(
        schedule,
        state,
        loglikelihood_0,
        **kwargs,
    ):
        del loglikelihood_0, kwargs
        captured.append(schedule.directions)
        n_slices = schedule.directions.shape[0]
        zeros = jnp.zeros((n_slices,), dtype=int)
        return state, SegmentInfo(
            is_accepted=jnp.ones((n_slices,), dtype=bool),
            num_expansions=zeros,
            num_shrink=zeros,
            bracket_left=jnp.zeros((n_slices,)),
            bracket_right=jnp.zeros((n_slices,)),
        )

    monkeypatch.setattr(
        "jimgw.samplers.blackjax.swig.run_segment",
        capture_segment,
    )
    key = jax.random.key(45)
    threshold = jnp.asarray(-20.0)
    state = _h2_particle_state(jnp.zeros(_H2_N_DIMS), threshold)
    factors = _h2_covariance_factors()

    def directions_for(direction_mode):
        captured.clear()
        _, step = _h2_builders(direction_mode=direction_mode)
        step(
            key,
            state,
            threshold,
            block_covariance_factors=factors,
        )
        return jnp.concatenate(tuple(captured))

    basis_directions = directions_for("covariance-basis-8d")
    covariance_directions = directions_for("covariance")

    assert basis_directions.shape == (15, 15)
    whitened = jnp.linalg.solve(factors[0], basis_directions[:8, :8].T).T
    np.testing.assert_array_equal(jnp.sum(jnp.abs(whitened) > 0.0, axis=1), 1)
    np.testing.assert_array_equal(jnp.sum(jnp.abs(whitened) > 0.0, axis=0), 1)
    np.testing.assert_allclose(jnp.max(jnp.abs(whitened), axis=1), 2.0)
    np.testing.assert_array_equal(basis_directions[8:], covariance_directions[8:])


def test_swig_covariance_basis_keeps_exact_15_slice_10_plus_5_pricing():
    cache_calls = []
    likelihood_calls = []

    def counted_build_cache(position):
        jax.debug.callback(lambda _: cache_calls.append(None), position[0])
        return jnp.tanh(position[:10])

    def counted_log_likelihood(position, cache):
        jax.debug.callback(lambda _: likelihood_calls.append(None), position[0])
        return -jnp.sum((cache - 0.2) ** 2) - jnp.sum((position[10:] + 0.1) ** 2)

    _, fsm = _h2_builders(
        per_slice_info=True,
        build_cache=counted_build_cache,
        log_likelihood_from_cache_fn=counted_log_likelihood,
    )
    threshold = jnp.asarray(-30.0)
    state = _h2_particle_state(jnp.zeros(_H2_N_DIMS), threshold)
    result = jax.jit(
        lambda: fsm(
            jax.random.key(46),
            state,
            threshold,
            block_covariance_factors=_h2_covariance_factors(),
        )
    )()
    jax.block_until_ready(result)
    info = jax.device_get(result[1])

    evaluations_per_slice = (
        np.asarray(info.num_expansions) + np.asarray(info.num_shrink) + 2
    )
    assert evaluations_per_slice.shape == (15,)
    assert len(likelihood_calls) == int(evaluations_per_slice.sum())
    assert len(cache_calls) == 1 + int(evaluations_per_slice[:10].sum())
    assert int(evaluations_per_slice[:10].size) == 10
    assert int(evaluations_per_slice[10:].size) == 5


def test_swig_explicit_block_budget_keeps_exact_12_slice_7_plus_5_pricing():
    cache_calls = []
    likelihood_calls = []

    def counted_build_cache(position):
        jax.debug.callback(lambda _: cache_calls.append(None), position[0])
        return jnp.tanh(position[:10])

    def counted_log_likelihood(position, cache):
        jax.debug.callback(lambda _: likelihood_calls.append(None), position[0])
        return -jnp.sum((cache - 0.2) ** 2) - jnp.sum((position[10:] + 0.1) ** 2)

    _, fsm = _h2_builders(
        per_slice_info=True,
        direction_mode="covariance",
        num_slice_steps_by_block=(5, 1, 1, 2, 1, 2),
        build_cache=counted_build_cache,
        log_likelihood_from_cache_fn=counted_log_likelihood,
    )
    threshold = jnp.asarray(-30.0)
    state = _h2_particle_state(jnp.zeros(_H2_N_DIMS), threshold)
    result = jax.jit(
        lambda: fsm(
            jax.random.key(47),
            state,
            threshold,
            block_covariance_factors=_h2_covariance_factors(),
        )
    )()
    jax.block_until_ready(result)
    info = jax.device_get(result[1])

    evaluations_per_slice = (
        np.asarray(info.num_expansions) + np.asarray(info.num_shrink) + 2
    )
    assert evaluations_per_slice.shape == (12,)
    assert len(likelihood_calls) == int(evaluations_per_slice.sum())
    assert len(cache_calls) == 1 + int(evaluations_per_slice[:7].sum())
    assert int(evaluations_per_slice[:7].size) == 7
    assert int(evaluations_per_slice[7:].size) == 5


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

