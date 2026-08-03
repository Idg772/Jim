"""Pathwise tests for Jim's vectorized slice-sampling helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.mcmc.slice import SliceState, stepping_out
from blackjax.mcmc.slice import build_kernel as build_slice_kernel

from jimgw.samplers.blackjax._slice import stepping_out_cached


@pytest.mark.parametrize("max_expansions", (0, 1, 10))
def test_cached_stepping_out_matches_blackjax_when_vmapped(max_expansions):
    keys = jax.random.split(jax.random.key(0), 4096)
    limits = jnp.linspace(-0.25, 8.0, keys.shape[0])

    def run(interval):
        def one(key, limit):
            def in_slice(t):
                return jnp.abs(t) < limit

            left, right, count, accept = interval(key, in_slice, 1.0, max_expansions)
            return left, right, count, accept(0.0)

        return jax.jit(jax.vmap(one))(keys, limits)

    expected = run(stepping_out)
    actual = run(stepping_out_cached)
    for expected_leaf, actual_leaf in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_cached_interval_preserves_full_slice_transition_pathwise():
    keys = jax.random.split(jax.random.key(1), 4096)
    positions = jnp.linspace(-2.0, 2.0, keys.shape[0])

    def logdensity(position):
        return -0.5 * position**2

    states = SliceState(position=positions, logdensity=jax.vmap(logdensity)(positions))

    def proposal_generator(direction_key, position, logdensity_fn):
        del logdensity_fn
        direction = jax.random.normal(direction_key)

        def slice_fn(t):
            candidate_position = position + t * direction
            candidate = SliceState(
                candidate_position,
                logdensity(candidate_position),
            )
            return candidate, jnp.abs(candidate_position) < 1.75

        return slice_fn

    def run(interval):
        kernel = build_slice_kernel(
            interval=interval,
            max_expansions=10,
            max_shrinkage=3,
        )
        return jax.jit(
            jax.vmap(lambda key, state: kernel(key, state, None, proposal_generator))
        )(keys, states)

    expected = run(stepping_out)
    actual = run(stepping_out_cached)
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
