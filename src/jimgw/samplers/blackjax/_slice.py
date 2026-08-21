"""Jim-specific slice-sampling execution helpers."""

from collections.abc import Callable

import jax
import jax.numpy as jnp


def stepping_out_cached(rng_key, in_slice: Callable, width: float, max_expansions: int):
    """BlackJAX stepping-out with its expensive predicate cached in the carry.

    JAX's batching rule masks a vmapped ``while_loop`` by re-evaluating its
    condition in the body.  BlackJAX's condition contains ``in_slice`` and can
    therefore execute a waveform likelihood twice per vector step.  Carrying
    the last boolean keeps the loop condition cheap while preserving the scalar
    endpoint sequence, random splits, brackets, and expansion count exactly.
    """
    u_key, jk_key = jax.random.split(rng_key)
    left = -width * jax.random.uniform(u_key)
    right = left + width

    v = jax.random.uniform(jk_key)
    j = jnp.floor(max_expansions * v).astype(int)
    k = (max_expansions - 1) - j

    def expand(endpoint, remaining, delta):
        inside = in_slice(endpoint)

        def condition(carry):
            _, remaining, inside = carry
            return inside & (remaining > 0)

        def body(carry):
            endpoint, remaining, _ = carry
            endpoint = endpoint + delta
            remaining = remaining - 1
            return endpoint, remaining, in_slice(endpoint)

        return jax.lax.while_loop(
            condition,
            body,
            (endpoint, remaining, inside),
        )

    # Preserve BlackJAX's call order: finish the left side before evaluating
    # the initial right endpoint.
    left, jl, _ = expand(left, j, -width)
    right, kr, _ = expand(right, k, width)
    num_expansions = (j - jl) + (k - kr)
    accept_fn = lambda _: jnp.asarray(True)
    return left, right, num_expansions, accept_fn


def shrink_only_bracket(rng_key, in_slice: Callable, width: float, max_expansions: int):
    """Fixed-width bracket with no endpoint evaluations (Neal 2003 §4.1).

    Consumes the same first uniform as ``stepping_out_cached`` so the FSM
    scheduler's pre-drawn ``bracket_u`` reproduces this bracket exactly.
    """
    del in_slice, max_expansions
    u_key, _ = jax.random.split(rng_key)
    left = -width * jax.random.uniform(u_key)
    right = left + width
    accept_fn = lambda _: jnp.asarray(True)
    return left, right, jnp.asarray(0), accept_fn
