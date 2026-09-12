"""Materialize a pure rebuild segment's source cache at its final position.

Routing is the caller's responsibility: this helper must only wrap explicit
rebuild segments. Hit callbacks, hybrid operations and observers of intermediate
caches cannot run inside such a segment. This changes cache materialization,
not the prescribed stepping-out or shrinking operations.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jimgw.samplers.blackjax._fsm import SegmentSchedule


def run_deferred_rebuild_segment(
    schedule: SegmentSchedule,
    state,
    loglikelihood_0,
    *,
    run_segment_fn: Callable,
    eval_candidate: Callable,
    build_cache: Callable,
    cache_independent_rebuild: bool,
    wrap_position: Callable,
    max_expansions: int,
    max_shrinkage: int,
    width: float = 1.0,
    shrink_only: bool = False,
):
    """Run ordinary R slices with scalar internal cache and restore once.

    ``cache_independent_rebuild=True`` declares an explicit caller contract:
    ``eval_candidate(position, cache)`` is pure, ignores its incoming cache,
    and returns a cache identical to the deterministic ``build_cache(position)``.
    Neither callback may observe state history or lane-specific hit summaries.
    The declaration cannot be verified from arbitrary Python callables.

    Every proposal keeps the original prior and likelihood calculation. Its
    cache output is discarded; only a scalar int8 dummy enters the inner loop.
    If any slice accepts, one call to ``build_cache`` restores the final cache,
    without reevaluating prior or likelihood. Otherwise the entering cache is
    returned exactly. Under outer ``vmap``, JAX can execute the final builder for
    unchanged lanes too, but still selects their original entering cache.

    Slice count does not determine routing: even a three-slice segment can be
    an explicitly identified R segment. The supplied runner must preserve the
    ordinary ``(state, SegmentInfo)`` contract and not observe private caches.
    """
    if cache_independent_rebuild is not True:
        raise ValueError(
            "Deferred caches require pure cache-independent rebuild callbacks"
        )
    if shrink_only:
        raise ValueError("Deferred caches require ordinary stepping-out")
    if not isinstance(schedule, SegmentSchedule):
        raise TypeError(
            "Deferred caches require a regular slice schedule, not hybrid operations"
        )
    if schedule.level_u.ndim != 1 or schedule.level_u.shape[0] < 1:
        raise ValueError("Deferred caches require at least one regular rebuild slice")
    if max_expansions < 1 or max_shrinkage < 1:
        raise ValueError("Expansion and shrinking caps must be positive")
    if not all(
        callable(callback) for callback in (run_segment_fn, eval_candidate, build_cache)
    ):
        raise TypeError("Deferred runner, evaluator and cache builder must be callable")

    dummy = jnp.asarray(0, dtype=jnp.int8)

    def score(position, ignored_cache):
        del ignored_cache
        score_dummy = jnp.asarray(0, dtype=jnp.int8)
        log_prior, log_likelihood, _ = eval_candidate(position, score_dummy)
        return log_prior, log_likelihood, score_dummy

    final, info = run_segment_fn(
        schedule,
        state._replace(cache=dummy),
        loglikelihood_0,
        eval_candidate=score,
        wrap_position=wrap_position,
        max_expansions=max_expansions,
        max_shrinkage=max_shrinkage,
        width=width,
        shrink_only=False,
    )
    restored = jax.lax.cond(
        jnp.any(info.is_accepted),
        lambda position: build_cache(position),
        lambda _: state.cache,
        final.position,
    )
    return final._replace(cache=restored), info
