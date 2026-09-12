import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.base import StateWithLogLikelihood

from jimgw.samplers.blackjax.swig import _build_swig_constrained_step


def test_conditional_segment_summary_preserves_full_seeded_transition():
    prior = lambda x: jnp.where(jnp.all(abs(x) < 3), 0.0, -jnp.inf)
    build = lambda x: x[0] ** 2
    evaluate = lambda x, c: -c - x[1] ** 2
    options = {
        "log_prior_fn": prior,
        "build_cache": build,
        "log_likelihood_from_cache_fn": evaluate,
        "rebuild_required_by_block": {(0,): True, (1,): False},
        "num_gibbs_sweeps": 3,
        "num_inner_steps_per_dim": 2,
        "max_steps": 4,
        "max_shrinkage": 30,
        "periodic": None,
        "n_dims": 2,
    }
    regular = _build_swig_constrained_step(**options)
    compressed = _build_swig_constrained_step(
        **options,
        prepare_hit_summary=lambda x, c: -c,
        log_likelihood_from_hit_summary=lambda x, s: s - x[1] ** 2,
    )
    x = jnp.array([0.2, 0.1])
    state = StateWithLogLikelihood(x, prior(x), evaluate(x, build(x)), -2.0)
    keys = jax.random.split(jax.random.key(73), 16)
    cov = (jnp.eye(1), jnp.eye(1))
    a = jax.jit(jax.vmap(lambda k: regular(k, state, -2.0, cov)))(keys)
    b = jax.jit(jax.vmap(lambda k: compressed(k, state, -2.0, cov)))(keys)
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(left, right)
