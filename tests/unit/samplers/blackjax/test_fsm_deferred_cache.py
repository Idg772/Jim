"""Deferred pure-R caches preserve the ordinary segment's external contract."""

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.samplers.blackjax._deferred_cache import run_deferred_rebuild_segment
from jimgw.samplers.blackjax._fsm import (
    SegmentSchedule,
    run_segment,
    slice_randoms_from_keys,
)
from jimgw.samplers.blackjax.swig import CachedSliceState


def _equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        left, right = np.asarray(left), np.asarray(right)
        assert left.shape == right.shape and left.dtype == right.dtype
        np.testing.assert_array_equal(
            left.reshape(-1).view(np.uint8), right.reshape(-1).view(np.uint8)
        )


def _cache(position):
    return {
        "carrier": jax.lax.complex(position * 2, -position),
        "clock": position**2,
        "nested": (position + 3, jnp.asarray(7, jnp.int32)),
    }


def _evaluate(position, ignored_cache):
    del ignored_cache
    # Quantization makes the scheduling assertions independent of reassociation.
    lp = -jnp.floor(32 * jnp.sum(position**2)) / 64
    ll = -jnp.floor(32 * jnp.sum((position - 0.2) ** 2)) / 16
    return lp, ll, _cache(position)


def _schedule(key, n_slices):
    prop, level, bracket, budget, shrink = slice_randoms_from_keys(
        jax.random.split(key, n_slices)
    )
    directions = jax.vmap(lambda k: jax.random.normal(k, (3,)))(prop)
    budget = budget.at[::3].set(0.0).at[1::3].set(1 - 2**-40)
    return SegmentSchedule(directions, level, bracket, budget, shrink)


def _fixture(n_slices, lanes=12):
    schedules = jax.vmap(partial(_schedule, n_slices=n_slices))(
        jax.random.split(jax.random.key(17), lanes)
    )
    position = jnp.linspace(-0.4, 0.4, lanes * 3).reshape(lanes, 3)
    lp, ll, cache = jax.vmap(_evaluate, in_axes=(0, None))(position, None)
    threshold = jnp.linspace(-2.0, -0.2, lanes)
    state = CachedSliceState(position, lp, ll, threshold, cache)
    return schedules, state, threshold


def _deferred(schedule, state, threshold, **options):
    return run_deferred_rebuild_segment(
        schedule,
        state,
        threshold,
        run_segment_fn=run_segment,
        build_cache=_cache,
        cache_independent_rebuild=True,
        **options,
    )


@pytest.mark.parametrize("n_slices", [1, 3, 7, 9])
@pytest.mark.parametrize("caps", [(1, 1), (10, 100)])
def test_vmap_state_brackets_counts_and_cache_match(n_slices, caps):
    """Explicit R routing accepts lengths other than the historical nine."""
    with jax.enable_x64():
        inputs = _fixture(n_slices)
        options = {
            "eval_candidate": _evaluate,
            "wrap_position": lambda p: p,
            "max_expansions": caps[0],
            "max_shrinkage": caps[1],
            "width": 0.7,
        }
        baseline = jax.jit(jax.vmap(lambda *args: run_segment(*args, **options)))(
            *inputs
        )
        deferred = jax.jit(jax.vmap(lambda *args: _deferred(*args, **options)))(*inputs)
        _equal(deferred, baseline)
        assert np.any(np.asarray(deferred[1].is_accepted))
        if caps[1] == 1:
            assert np.any(~np.asarray(deferred[1].is_accepted))


def test_partial_acceptance_and_no_accept_lanes_preserve_original_cache():
    with jax.enable_x64():
        schedule, state, threshold = _fixture(7)
        threshold = threshold.at[::2].set(jnp.inf)
        original_cache = jax.tree.map(lambda x: x + 11, state.cache)
        state = state._replace(cache=original_cache)
        options = {
            "eval_candidate": _evaluate,
            "wrap_position": lambda p: (p + 1) % 2 - 1,
            "max_expansions": 10,
            "max_shrinkage": 2,
        }
        inputs = schedule, state, threshold
        baseline = jax.jit(jax.vmap(lambda *args: run_segment(*args, **options)))(
            *inputs
        )
        deferred = jax.jit(jax.vmap(lambda *args: _deferred(*args, **options)))(*inputs)
        _equal(deferred, baseline)
        accepted = np.asarray(deferred[1].is_accepted)
        assert not accepted[::2].any()
        assert accepted[1::2].any() and (~accepted[1::2]).any()
        _equal(
            jax.tree.map(lambda x: x[::2], deferred[0].cache),
            jax.tree.map(lambda x: x[::2], original_cache),
        )


@pytest.mark.parametrize("accept", [False, True])
def test_same_request_random_stream_and_one_cache_only_restoration(accept):
    """Runtime callback spies prove no extra prior/likelihood evaluation."""
    with jax.enable_x64():
        schedule, state, threshold = jax.tree.map(lambda x: x[0], _fixture(3, lanes=1))
        threshold = jnp.asarray(-10.0 if accept else jnp.inf)
        baseline_positions, deferred_positions, restored_positions = [], [], []

        def recording_evaluator(log, position, cache):
            jax.debug.callback(
                lambda p: log.append(np.array(p)), position, ordered=True
            )
            return _evaluate(position, cache)

        def recording_builder(position):
            jax.debug.callback(
                lambda p: restored_positions.append(np.array(p)), position, ordered=True
            )
            return _cache(position)

        options = {
            "wrap_position": lambda p: p,
            "max_expansions": 10,
            "max_shrinkage": 3,
        }
        baseline = jax.jit(
            lambda: run_segment(
                schedule,
                state,
                threshold,
                eval_candidate=partial(recording_evaluator, baseline_positions),
                **options,
            )
        )()
        deferred = jax.jit(
            lambda: run_deferred_rebuild_segment(
                schedule,
                state,
                threshold,
                run_segment_fn=run_segment,
                eval_candidate=partial(recording_evaluator, deferred_positions),
                build_cache=recording_builder,
                cache_independent_rebuild=True,
                **options,
            )
        )()
        jax.block_until_ready((baseline, deferred))
        jax.effects_barrier()
        _equal(deferred, baseline)
        np.testing.assert_array_equal(deferred_positions, baseline_positions)
        info = deferred[1]
        assert len(deferred_positions) == int(
            jnp.sum(info.num_expansions + info.num_shrink + 2)
        )
        assert len(restored_positions) == int(accept)
        if accept:
            np.testing.assert_array_equal(restored_positions[0], deferred[0].position)


def test_inner_runner_sees_only_scalar_dummy_and_receives_options():
    with jax.enable_x64():
        schedule, state, threshold = jax.tree.map(lambda x: x[0], _fixture(3, lanes=1))

        def inspected_runner(schedule, state, threshold, *, eval_candidate, **options):
            assert state.cache.shape == () and state.cache.dtype == jnp.int8
            lp, ll, cache = eval_candidate(state.position, state.cache)
            _equal((lp, ll), _evaluate(state.position, None)[:2])
            assert cache.shape == () and cache.dtype == jnp.int8
            assert options["width"] == 0.3
            return run_segment(
                schedule, state, threshold, eval_candidate=eval_candidate, **options
            )

        result = run_deferred_rebuild_segment(
            schedule,
            state,
            threshold,
            run_segment_fn=inspected_runner,
            eval_candidate=_evaluate,
            build_cache=_cache,
            cache_independent_rebuild=True,
            wrap_position=lambda p: p,
            max_expansions=10,
            max_shrinkage=100,
            width=0.3,
        )
        assert jnp.any(result[1].is_accepted)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"cache_independent_rebuild": False}, "pure cache-independent"),
        ({"cache_independent_rebuild": 1}, "pure cache-independent"),
        ({"shrink_only": True}, "ordinary stepping-out"),
        ({"max_expansions": 0}, "caps must be positive"),
        ({"max_shrinkage": 0}, "caps must be positive"),
        ({"build_cache": None}, "must be callable"),
        ({"run_segment_fn": None}, "must be callable"),
    ],
)
def test_unsupported_contracts_are_rejected(changes, message):
    with jax.enable_x64():
        schedule, state, threshold = jax.tree.map(lambda x: x[0], _fixture(1, lanes=1))
        options = {
            "run_segment_fn": run_segment,
            "eval_candidate": _evaluate,
            "build_cache": _cache,
            "cache_independent_rebuild": True,
            "wrap_position": lambda p: p,
            "max_expansions": 10,
            "max_shrinkage": 100,
        }
        with pytest.raises((ValueError, TypeError), match=message):
            run_deferred_rebuild_segment(
                schedule, state, threshold, **(options | changes)
            )


@pytest.mark.parametrize(
    "schedule,message",
    [
        (SimpleNamespace(is_independence=True), "regular slice schedule"),
        (SegmentSchedule(*(jnp.empty((0,)) for _ in range(5))), "at least one"),
    ],
)
def test_hybrid_and_empty_schedules_are_rejected(schedule, message):
    with pytest.raises((ValueError, TypeError), match=message):
        run_deferred_rebuild_segment(
            schedule,
            None,
            None,
            run_segment_fn=run_segment,
            eval_candidate=_evaluate,
            build_cache=_cache,
            cache_independent_rebuild=True,
            wrap_position=lambda p: p,
            max_expansions=10,
            max_shrinkage=100,
        )
