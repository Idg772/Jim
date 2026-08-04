"""FSM scheduler for vectorized slice sampling.

Implements the finite-state-machine execution of Dance et al.,
"Efficiently Vectorized MCMC on Modern Accelerators" (arXiv:2503.17405),
specialized to Jim's nested slice transition. A static *segment* (a
contiguous run of slices sharing one waveform-rebuild class) runs as one
``lax.while_loop`` whose body evaluates the likelihood exactly once per
lane per tick. Under ``vmap`` the while-loop batching rule masks finished
lanes, so each lane advances through its own (slice, phase) sequence
instead of waiting at per-ordinal loop barriers.

Bitwise contract: for the same per-slice key sequence this reproduces the
BlackJAX slice kernel (with Jim's cached stepping-out) exactly: the same
endpoint states, expansion and shrink counts, accept flags, and recorded
brackets. All uniforms are pre-drawn from BlackJAX's key-split tree; JAX's
counter-based Threefry PRNG makes draw *order* irrelevant. RBG-family keys
are intentionally unsupported because their non-standard ``vmap`` semantics
are incompatible with asynchronous lane advancement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

_LEFT, _RIGHT, _SHRINK, _DONE = 0, 1, 2, 3


class SegmentSchedule(NamedTuple):
    """Pre-drawn per-slice randomness for one segment, in execution order."""

    directions: Array
    level_u: Array
    bracket_u: Array
    bracket_v: Array
    shrink_key_data: Array


class SegmentInfo(NamedTuple):
    """Per-slice diagnostics matching BlackJAX ``SliceInfo`` semantics."""

    is_accepted: Array
    num_expansions: Array
    num_shrink: Array
    bracket_left: Array
    bracket_right: Array


def slice_randoms_from_keys(slice_keys):
    """Split per-slice keys exactly as the BlackJAX slice kernel does.

    Returns ``(prop_keys, level_u, bracket_u, bracket_v, shrink_key_data)``.
    Callers draw directions from ``prop_keys`` and hand the remaining draws to
    :class:`SegmentSchedule`.
    """

    key_impl = str(jax.random.key_impl(slice_keys))
    if key_impl != "threefry2x32":
        raise ValueError(
            "The FSM scheduler requires Threefry PRNG keys; "
            f"{key_impl!r} has batching semantics that cannot preserve "
            "BlackJAX slice paths under asynchronous lane advancement."
        )

    def one(key):
        prop_key, slice_key = jax.random.split(key)
        level_key, interval_key, shrink_key = jax.random.split(slice_key, 3)
        u_key, jk_key = jax.random.split(interval_key)
        return (
            prop_key,
            jax.random.uniform(level_key),
            jax.random.uniform(u_key),
            jax.random.uniform(jk_key),
            jax.random.key_data(shrink_key),
        )

    return jax.vmap(one)(slice_keys)


class _Carry(NamedTuple):
    slice_idx: Array
    phase: Array
    state: Any
    level: Array
    left: Array
    right: Array
    j_remaining: Array
    k_remaining: Array
    shrink_key_data: Array
    n_expand: Array
    n_shrink: Array
    info: SegmentInfo


def _slice_entry(schedule, idx, logdensity, max_expansions, width):
    """Bundle initialization of the slice at ``idx``."""
    level = logdensity + jnp.log(schedule.level_u[idx])
    left = -width * schedule.bracket_u[idx]
    right = left + width
    j = jnp.floor(max_expansions * schedule.bracket_v[idx]).astype(int)
    k = (max_expansions - 1) - j
    return level, left, right, j, k, schedule.shrink_key_data[idx]


def run_segment(
    schedule: SegmentSchedule,
    state,
    loglikelihood_0,
    *,
    eval_candidate: Callable,
    wrap_position: Callable,
    max_expansions: int,
    max_shrinkage: int,
    width: float = 1.0,
):
    """Run every slice of one segment for a single chain.

    ``eval_candidate(position, cache) -> (logdensity, loglikelihood, cache)``
    is the segment's only likelihood site. ``state`` carries ``position``,
    ``logdensity``, ``loglikelihood``, ``loglikelihood_birth``, and ``cache``.
    """
    assert max_shrinkage >= 1

    n_slices = schedule.level_u.shape[0]
    zero_i = jnp.zeros((n_slices,), dtype=jnp.result_type(int))
    info0 = SegmentInfo(
        is_accepted=jnp.zeros((n_slices,), dtype=bool),
        num_expansions=zero_i,
        num_shrink=zero_i,
        bracket_left=jnp.zeros((n_slices,)),
        bracket_right=jnp.zeros((n_slices,)),
    )
    level, left, right, j, k, shrink_key_data = _slice_entry(
        schedule, 0, state.logdensity, max_expansions, width
    )
    carry = _Carry(
        slice_idx=jnp.asarray(0),
        phase=jnp.asarray(_LEFT),
        state=state,
        level=level,
        left=left,
        right=right,
        j_remaining=j,
        k_remaining=k,
        shrink_key_data=shrink_key_data,
        n_expand=jnp.asarray(0),
        n_shrink=jnp.asarray(0),
        info=info0,
    )

    def cond(c):
        return c.phase != _DONE

    def body(c):
        key = jax.random.wrap_key_data(
            c.shrink_key_data,
            impl="threefry2x32",
        )
        next_key, subkey = jax.random.split(key)
        t_shrink = c.left + jax.random.uniform(subkey) * (c.right - c.left)
        in_left = c.phase == _LEFT
        in_right = c.phase == _RIGHT
        in_shrink = c.phase == _SHRINK
        shrink_key_data = jnp.where(
            in_shrink,
            jax.random.key_data(next_key),
            c.shrink_key_data,
        )
        t = jnp.where(
            in_left,
            c.left,
            jnp.where(
                in_right,
                c.right,
                jnp.where(in_shrink, t_shrink, 0.0),
            ),
        )

        direction = schedule.directions[c.slice_idx]
        position = wrap_position(c.state.position + t * direction)
        logdensity, loglikelihood, cache = eval_candidate(position, c.state.cache)
        inside = (logdensity >= c.level) & (loglikelihood > loglikelihood_0)

        expand_left = in_left & inside & (c.j_remaining > 0)
        to_right = in_left & ~expand_left
        expand_right = in_right & inside & (c.k_remaining > 0)
        to_shrink = in_right & ~expand_right

        found = in_shrink & inside
        n_shrink = c.n_shrink + in_shrink.astype(c.n_shrink.dtype)
        exhausted = in_shrink & ~inside & (n_shrink >= max_shrinkage)
        slice_done = found | exhausted
        keep_shrinking = in_shrink & ~slice_done
        n_expand = c.n_expand + (expand_left | expand_right).astype(c.n_expand.dtype)

        left = jnp.where(
            expand_left,
            c.left - width,
            jnp.where(keep_shrinking & (t < 0.0), t, c.left),
        )
        right = jnp.where(
            expand_right,
            c.right + width,
            jnp.where(keep_shrinking & (t >= 0.0), t, c.right),
        )

        candidate = c.state._replace(
            position=position,
            logdensity=logdensity,
            loglikelihood=loglikelihood,
            cache=cache,
        )
        new_state = jax.tree.map(
            lambda new, old: jnp.where(found, new, old), candidate, c.state
        )

        info = c.info._replace(
            bracket_left=jnp.where(
                to_shrink,
                c.info.bracket_left.at[c.slice_idx].set(left),
                c.info.bracket_left,
            ),
            bracket_right=jnp.where(
                to_shrink,
                c.info.bracket_right.at[c.slice_idx].set(right),
                c.info.bracket_right,
            ),
        )
        info = jax.tree.map(
            lambda arr, val: jnp.where(slice_done, arr.at[c.slice_idx].set(val), arr),
            info,
            SegmentInfo(
                is_accepted=found,
                num_expansions=n_expand,
                num_shrink=n_shrink,
                bracket_left=info.bracket_left[c.slice_idx],
                bracket_right=info.bracket_right[c.slice_idx],
            ),
        )

        next_idx = c.slice_idx + slice_done.astype(c.slice_idx.dtype)
        segment_done = slice_done & (next_idx >= n_slices)
        entry_idx = jnp.minimum(next_idx, n_slices - 1)
        e_level, e_left, e_right, e_j, e_k, e_key_data = _slice_entry(
            schedule,
            entry_idx,
            new_state.logdensity,
            max_expansions,
            width,
        )
        advance = slice_done & ~segment_done

        phase = jnp.where(
            segment_done,
            _DONE,
            jnp.where(
                advance,
                _LEFT,
                jnp.where(
                    to_right,
                    _RIGHT,
                    jnp.where(
                        to_shrink | keep_shrinking,
                        _SHRINK,
                        c.phase,
                    ),
                ),
            ),
        )
        return _Carry(
            slice_idx=jnp.where(advance, next_idx, c.slice_idx),
            phase=phase,
            state=new_state,
            level=jnp.where(advance, e_level, c.level),
            left=jnp.where(advance, e_left, left),
            right=jnp.where(advance, e_right, right),
            j_remaining=jnp.where(
                advance,
                e_j,
                c.j_remaining - expand_left.astype(c.j_remaining.dtype),
            ),
            k_remaining=jnp.where(
                advance,
                e_k,
                c.k_remaining - expand_right.astype(c.k_remaining.dtype),
            ),
            shrink_key_data=jnp.where(advance, e_key_data, shrink_key_data),
            n_expand=jnp.where(advance, 0, n_expand),
            n_shrink=jnp.where(advance, 0, n_shrink),
            info=info,
        )

    final = jax.lax.while_loop(cond, body, carry)
    return final.state, final.info
