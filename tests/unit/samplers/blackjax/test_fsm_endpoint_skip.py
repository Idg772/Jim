"""Production endpoint skipping preserves the complete FSM transition."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.samplers.blackjax._fsm import (
    SegmentSchedule,
    _Carry,
    run_segment,
    slice_randoms_from_keys,
)
from jimgw.samplers.blackjax.swig import CachedSliceState


def _assert_identical(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        left, right = np.asarray(left), np.asarray(right)
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(
            left.reshape(-1).view(np.uint8), right.reshape(-1).view(np.uint8)
        )


def _cache(position, accepted_moves):
    return {
        "position": position,
        "outer": jnp.outer(position, position),
        "phase": jax.lax.complex(jnp.cos(position), jnp.sin(position)),
        "accepted_moves": accepted_moves,
    }


def _prior(position):
    return jnp.where(
        jnp.abs(position[2]) < 1.25, -0.5 * jnp.sum(position[1:] ** 2), -jnp.inf
    )


def _likelihood(position):
    return -3.0 * jnp.sum((position[1:] - 0.2) ** 2) + jnp.cos(3 * position[0])


def _evaluate(position, cache):
    return (
        _prior(position),
        _likelihood(position),
        _cache(position, cache["accepted_moves"] + 1),
    )


def _schedule(key, n_slices, scale):
    prop, level, bracket_u, bracket_v, shrink_key = slice_randoms_from_keys(
        jax.random.split(key, n_slices)
    )
    directions = jax.vmap(lambda k: jax.random.normal(k, (3,)))(prop) * scale
    # Exercise both initially exhausted sides and ordinary budget allocations.
    bracket_v = bracket_v.at[::3].set(0.0).at[1::3].set(1.0 - 2.0**-40)
    return SegmentSchedule(directions, level, bracket_u, bracket_v, shrink_key)


@pytest.mark.parametrize("vectorized", [False, True], ids=["scalar", "vmap"])
@pytest.mark.parametrize(
    "max_expansions,max_shrinkage,shrink_only",
    [
        (0, 1, False),
        (1, 2, False),
        (2, 2, False),
        (10, 100, False),
        (10, 1, False),
        (10, 100, True),
    ],
)
def test_complete_transition_matches_baseline(
    vectorized, max_expansions, max_shrinkage, shrink_only
):
    with jax.enable_x64():
        keys = jax.random.split(jax.random.key(31), 16)
        positions = jax.random.uniform(keys[0], (16, 3), minval=-0.9, maxval=0.9)
        scales = jnp.geomspace(0.002, 40.0, 16)
        drops = jnp.geomspace(1e-6, 20.0, 16)

        def run(skip):
            def one(key, position, scale, drop):
                threshold = _likelihood(position) - drop
                state = CachedSliceState(
                    position,
                    _prior(position),
                    _likelihood(position),
                    threshold,
                    _cache(position, jnp.asarray(0)),
                )

                def wrap(x):
                    return x.at[0].set((x[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi)

                return run_segment(
                    _schedule(key, 12, scale),
                    state,
                    threshold,
                    eval_candidate=_evaluate,
                    wrap_position=wrap,
                    max_expansions=max_expansions,
                    max_shrinkage=max_shrinkage,
                    width=1.3,
                    shrink_only=shrink_only,
                    skip_exhausted_endpoints=skip,
                )

            if vectorized:
                return jax.jit(jax.vmap(one))(keys, positions, scales, drops)
            return jax.jit(one)(keys[-1], positions[-1], scales[-1], drops[-1])

        actual, expected = run(True), run(False)
        _assert_identical(actual, expected)
        np.testing.assert_array_equal(
            actual[0].cache["accepted_moves"],
            np.asarray(actual[1].is_accepted).sum(axis=-1),
        )


def test_internal_carry_preserves_random_keys(monkeypatch):
    original_while = jax.lax.while_loop
    with jax.enable_x64():
        position = jnp.zeros(3)
        state = CachedSliceState(
            position,
            _prior(position),
            _likelihood(position),
            jnp.asarray(0.0),
            _cache(position, jnp.asarray(0)),
        )

        def run(skip):
            def one(key):
                captured = []

                def observe(cond, body, initial):
                    result = original_while(cond, body, initial)
                    if isinstance(initial, _Carry):
                        captured.append(result)
                    return result

                with monkeypatch.context() as patch:
                    patch.setattr(jax.lax, "while_loop", observe)
                    result = run_segment(
                        _schedule(key, 24, 3.0),
                        state,
                        0.0,
                        eval_candidate=_evaluate,
                        wrap_position=lambda x: x,
                        max_expansions=10,
                        max_shrinkage=3,
                        skip_exhausted_endpoints=skip,
                    )
                assert len(captured) == 1
                return result, captured[0]

            return jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(918), 8))

        _assert_identical(run(True), run(False))


def _count_evaluations(skip, *, max_expansions, flat=True, shrink_only=False):
    calls = []
    with jax.enable_x64():
        n_slices = 4 if flat else 1
        schedule = _schedule(jax.random.key(47), n_slices, 1.0)._replace(
            directions=jnp.ones((n_slices, 3)),
            bracket_u=jnp.full((n_slices,), 0.5),
        )
        if not flat:
            schedule = schedule._replace(
                level_u=jnp.full((n_slices,), jnp.exp(-0.1)),
                bracket_v=jnp.full((n_slices,), 0.45),
            )
        position = jnp.zeros(3)
        state = CachedSliceState(
            position,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(-1.0),
            _cache(position, jnp.asarray(0)),
        )

        def evaluate(x, cache):
            jax.debug.callback(lambda _: calls.append(1), x, ordered=True)
            prior = jnp.asarray(0.0) if flat else -jnp.sum(x**2)
            return prior, jnp.asarray(0.0), _cache(x, cache["accepted_moves"] + 1)

        result = jax.jit(
            lambda: run_segment(
                schedule,
                state,
                -1.0,
                eval_candidate=evaluate,
                wrap_position=lambda x: x,
                max_expansions=max_expansions,
                max_shrinkage=100,
                width=4.0,
                shrink_only=shrink_only,
                **({} if skip is None else {"skip_exhausted_endpoints": skip}),
            )
        )()
        jax.block_until_ready(result)
        jax.effects_barrier()
    return result, len(calls)


@pytest.mark.parametrize("max_expansions", [0, 1, 2, 10])
def test_skips_only_exhausted_predicate_calls(max_expansions):
    actual, actual_calls = _count_evaluations(True, max_expansions=max_expansions)
    expected, expected_calls = _count_evaluations(False, max_expansions=max_expansions)
    _assert_identical(actual, expected)
    assert expected_calls - actual_calls == 8
    assert actual_calls == int(
        np.sum(actual[1].num_expansions) + np.sum(actual[1].num_shrink)
    )


@pytest.mark.parametrize("flat,shrink_only", [(False, False), (True, True)])
def test_required_endpoints_and_shrink_only_calls_are_unchanged(flat, shrink_only):
    kwargs = {"max_expansions": 10, "flat": flat, "shrink_only": shrink_only}
    actual, actual_calls = _count_evaluations(True, **kwargs)
    expected, expected_calls = _count_evaluations(False, **kwargs)
    _assert_identical(actual, expected)
    assert actual_calls == expected_calls


def test_default_retains_baseline_calls():
    default, default_calls = _count_evaluations(None, max_expansions=1)
    explicit, explicit_calls = _count_evaluations(False, max_expansions=1)
    _assert_identical(default, explicit)
    assert default_calls == explicit_calls == 12
