"""Tests for cache-aware Nested Slice within Gibbs."""

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jimgw.samplers.blackjax.swig import BlackJAXSwiGSampler
from jimgw.samplers.config import BlackJAXSwiGConfig


def _log_prior(position):
    return jnp.where(jnp.all((position >= 0.0) & (position <= 1.0)), 0.0, -jnp.inf)


def _build_cache(position):
    return position[0] ** 2


def _log_likelihood_from_cache(position, cache):
    return -40.0 * ((cache - 0.25) ** 2 + (position[1] - 0.5) ** 2)


def _log_likelihood(position):
    return _log_likelihood_from_cache(position, _build_cache(position))


def _make_sampler(
    checkpoint_dir: Optional[Path] = None,
) -> BlackJAXSwiGSampler:
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        n_live=24,
        n_delete_frac=0.25,
        termination_dlogz=1.5,
        max_steps=4,
        max_shrinkage=30,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
    )
    return BlackJAXSwiGSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
        rebuild_required_by_block={(0,): True, (1,): False},
        build_cache=_build_cache,
        log_likelihood_from_cache_fn=_log_likelihood_from_cache,
    )


def test_swig_cached_likelihood_remains_consistent():
    sampler = _make_sampler()
    initial = jax.random.uniform(jax.random.key(1), (24, 2))
    sampler.sample(jax.random.key(2), initial)
    result = sampler.get_samples()
    expected = jax.vmap(_log_likelihood)(jnp.asarray(result["samples"]))
    np.testing.assert_allclose(result["log_likelihood"], expected, rtol=1e-10)


def test_swig_does_not_store_cache_on_live_particles():
    sampler = _make_sampler()
    initial = jax.random.uniform(jax.random.key(3), (24, 2))
    sampler.sample(jax.random.key(4), initial)
    assert not hasattr(sampler._final_state.particles, "cache")


def test_swig_has_a_distinct_sampler_name():
    assert _make_sampler().sampler_name == "BlackJAX SwiG"


def test_swig_forwards_n_devices_to_internal_nss_config():
    """`n_devices` must reach the internally-built `BlackJAXNSSConfig` that
    `_sample` actually reads — a value forwarded by hand across two Pydantic
    models is easy to silently drop.
    """
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        n_live=24,
        n_delete_frac=0.25,
        termination_dlogz=1.5,
        n_devices=2,
    )
    sampler = BlackJAXSwiGSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
        rebuild_required_by_block={(0,): True, (1,): False},
        build_cache=_build_cache,
        log_likelihood_from_cache_fn=_log_likelihood_from_cache,
    )
    assert sampler._config.n_devices == 2


def test_swig_fsm_update_params_factors_each_block_once(monkeypatch):
    config = BlackJAXSwiGConfig(
        blocks=[["first", "third"], ["second"]],
        n_live=24,
    )
    sampler = BlackJAXSwiGSampler(
        n_dims=3,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
        rebuild_required_by_block={(0, 2): True, (1,): False},
        build_cache=_build_cache,
        log_likelihood_from_cache_fn=_log_likelihood_from_cache,
    )
    positions = jnp.asarray(
        [
            [-2.0, -1.0, 0.0],
            [-1.0, 0.5, 1.0],
            [0.0, 2.0, 1.5],
            [1.0, 1.0, 3.0],
            [2.0, -2.0, 4.0],
        ]
    )
    state = SimpleNamespace(particles=SimpleNamespace(position=positions))
    covariance = jnp.cov(positions, ddof=0, rowvar=False)

    original_cholesky = jnp.linalg.cholesky
    factorized_shapes = []

    def recording_cholesky(matrix):
        factorized_shapes.append(matrix.shape)
        return original_cholesky(matrix)

    monkeypatch.setattr(jnp.linalg, "cholesky", recording_cholesky)
    covariance_params = sampler._update_inner_kernel_params_fn(
        jax.random.key(0),
        state,
        None,
    )
    assert set(covariance_params) == {"block_covariances"}
    assert factorized_shapes == []
    for actual, indices in zip(
        covariance_params["block_covariances"],
        ((0, 2), (1,)),
        strict=True,
    ):
        expected = covariance[jnp.ix_(jnp.asarray(indices), jnp.asarray(indices))]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    params = sampler._fsm_update_inner_kernel_params_fn(
        jax.random.key(0),
        state,
        None,
    )

    assert set(params) == {"block_covariance_factors"}
    assert factorized_shapes == [(2, 2), (1, 1)]
    for factor, indices in zip(
        params["block_covariance_factors"],
        ((0, 2), (1,)),
        strict=True,
    ):
        expected = covariance[jnp.ix_(jnp.asarray(indices), jnp.asarray(indices))]
        np.testing.assert_allclose(factor @ factor.T, expected, rtol=1e-6)
        np.testing.assert_array_equal(factor, jnp.tril(factor))


def test_swig_checkpoint_records_sampler_name(tmp_path, monkeypatch):
    sampler = _make_sampler(checkpoint_dir=tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pkl"
    original_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None
            if self == checkpoint_path
            else original_unlink(self, missing_ok=missing_ok)
        ),
    )
    sampler.sample(
        jax.random.key(5),
        jax.random.uniform(jax.random.key(6), (24, 2)),
    )
    monkeypatch.setattr(Path, "unlink", original_unlink)

    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    assert checkpoint["sampler_name"] == sampler.sampler_name
    checkpoint_path.unlink()
