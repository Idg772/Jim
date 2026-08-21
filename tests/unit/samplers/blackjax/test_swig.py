"""Tests for cache-aware Nested Slice within Gibbs."""

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.ns.adaptive import AdaptiveNSState

from jimgw.samplers.blackjax import swig
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
    **config_overrides,
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
        **config_overrides,
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


def _make_complementary_de_sampler(
    *, attempts: int = 4, n_devices: int = 1
) -> BlackJAXSwiGSampler:
    intrinsic_names = [f"intrinsic_{index}" for index in range(8)]

    def log_prior(position):
        return jnp.where(jnp.all((position >= 0.0) & (position <= 1.0)), 0.0, -jnp.inf)

    def build_cache(position):
        return position[:8] ** 2

    def log_likelihood_from_cache(position, cache):
        return -jnp.sum((cache - 0.25) ** 2) - (position[8] - 0.5) ** 2

    def log_likelihood(position):
        return log_likelihood_from_cache(position, build_cache(position))

    config = BlackJAXSwiGConfig(
        blocks=[intrinsic_names, ["phase"]],
        block_kernel_modes=["slice", "periodic-uniform-independence"],
        complementary_de_jump_block={
            "parameters": intrinsic_names,
            "attempts": attempts,
        },
        num_inner_steps_per_dim=1,
        num_gibbs_sweeps=1,
        n_live=24,
        n_delete_frac=0.25,
        termination_dlogz=1.5,
        max_steps=4,
        max_shrinkage=30,
        n_devices=n_devices,
    )
    return BlackJAXSwiGSampler(
        n_dims=9,
        log_prior_fn=log_prior,
        log_likelihood_fn=log_likelihood,
        log_posterior_fn=lambda x: log_prior(x) + log_likelihood(x),
        config=config,
        periodic={8: (0.0, 1.0)},
        rebuild_required_by_block={tuple(range(8)): True, (8,): False},
        build_cache=build_cache,
        log_likelihood_from_cache_fn=log_likelihood_from_cache,
        resolved_complementary_de_jump_block=(tuple(range(8)), True, attempts),
    )


@pytest.mark.parametrize("attempts", [4, 8], ids=["h5-cde4", "h6-cde8"])
def test_complementary_de_sampler_reports_exact_work_and_consistent_cache(attempts):
    sampler = _make_complementary_de_sampler(attempts=attempts)
    initial = jax.random.uniform(jax.random.key(81), (24, 9))
    sampler.sample(jax.random.key(82), initial)
    result = sampler.get_samples()
    expected = jax.vmap(sampler._log_likelihood_fn)(jnp.asarray(result["samples"]))
    np.testing.assert_allclose(result["log_likelihood"], expected, rtol=1e-10)

    diagnostics = sampler.get_diagnostics()
    replacements = 6 * diagnostics["n_iterations"]
    assert diagnostics["n_complementary_de_attempts"] == attempts * replacements
    assert (
        diagnostics["n_likelihood_evaluations_complementary_de"]
        == attempts * replacements
    )
    assert (
        diagnostics["n_likelihood_evaluations_complementary_de_waveform_rebuild"]
        == attempts * replacements
    )
    assert diagnostics["n_likelihood_evaluations_complementary_de_cache_hit"] == 0
    assert diagnostics["n_complementary_de_donor_policy_violations"] == 0
    assert diagnostics["complementary_de_attempts_history"].shape == (replacements,)
    assert diagnostics["complementary_de_acceptances_history"].shape == (replacements,)
    assert diagnostics["complementary_de_donor_policy_violations_history"].shape == (
        replacements,
    )
    assert diagnostics["complementary_de_complement_size_history"].shape == (
        replacements,
    )
    assert diagnostics["complementary_de_attempts_by_block_history"].shape == (
        replacements,
        1,
    )
    assert diagnostics["complementary_de_acceptances_by_block_history"].shape == (
        replacements,
        1,
    )
    assert diagnostics["complementary_de_acceptances_by_attempt_history"].shape == (
        replacements,
        attempts,
    )
    assert diagnostics[
        "complementary_de_donor_policy_violations_by_attempt_history"
    ].shape == (replacements, attempts)
    parent_history = diagnostics["complementary_de_parent_index_history"]
    donor_history = diagnostics["complementary_de_donor_indices_by_attempt_history"]
    assert parent_history.shape == (replacements,)
    assert donor_history.shape == (replacements, attempts, 2)
    assert diagnostics["complementary_de_position_before_by_attempt_history"].shape == (
        replacements,
        attempts,
        9,
    )
    assert diagnostics[
        "complementary_de_proposal_position_by_attempt_history"
    ].shape == (replacements, attempts, 9)
    assert not np.any(donor_history == parent_history[:, None, None])
    assert diagnostics["complementary_de_complement_size_history"].size == replacements
    assert diagnostics["complementary_de_blocks"] == [
        {
            "parameters": [f"intrinsic_{index}" for index in range(8)],
            "requires_waveform_rebuild": True,
            "attempts_per_replacement": attempts,
            "gamma": 1.0,
            "placement": "after-target-block",
            "n_attempts": attempts * replacements,
            "n_acceptances": diagnostics["n_complementary_de_acceptances"],
            "n_donor_policy_violations": 0,
            "acceptance_rate": diagnostics["complementary_de_acceptance_rate"],
        }
    ]
    assert diagnostics["n_slice_updates"] == 8 * replacements
    assert diagnostics["n_periodic_uniform_independence_attempts"] == replacements
    assert (
        diagnostics["n_slice_updates"]
        + diagnostics["n_periodic_uniform_independence_attempts"]
        + diagnostics["n_complementary_de_attempts"]
        == (9 + attempts) * replacements
    )


def test_complementary_de_sampler_accepts_eight_attempt_budget():
    sampler = _make_complementary_de_sampler(attempts=8)

    assert sampler._resolved_complementary_de_jump_block == (
        tuple(range(8)),
        True,
        8,
    )


def test_complementary_de_constructor_rejects_mismatched_resolved_target():
    intrinsic_names = [f"intrinsic_{index}" for index in range(8)]
    config = BlackJAXSwiGConfig(
        blocks=[intrinsic_names, ["phase"]],
        block_kernel_modes=["slice", "periodic-uniform-independence"],
        complementary_de_jump_block={
            "parameters": intrinsic_names,
            "attempts": 4,
        },
        num_gibbs_sweeps=1,
    )

    with pytest.raises(ValueError, match="configured target block"):
        BlackJAXSwiGSampler(
            n_dims=9,
            log_prior_fn=_log_prior,
            log_likelihood_fn=_log_likelihood,
            log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
            config=config,
            periodic={8: (0.0, 1.0)},
            rebuild_required_by_block={tuple(range(8)): True, (8,): False},
            build_cache=_build_cache,
            log_likelihood_from_cache_fn=_log_likelihood_from_cache,
            resolved_complementary_de_jump_block=(tuple(range(1, 9)), True, 4),
        )


def test_complementary_de_constructor_rejects_mismatched_resolved_attempts():
    intrinsic_names = [f"intrinsic_{index}" for index in range(8)]
    config = BlackJAXSwiGConfig(
        blocks=[intrinsic_names, ["phase"]],
        block_kernel_modes=["slice", "periodic-uniform-independence"],
        complementary_de_jump_block={
            "parameters": intrinsic_names,
            "attempts": 4,
        },
        num_gibbs_sweeps=1,
    )

    with pytest.raises(ValueError, match="attempts do not match"):
        BlackJAXSwiGSampler(
            n_dims=9,
            log_prior_fn=_log_prior,
            log_likelihood_fn=_log_likelihood,
            log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
            config=config,
            periodic={8: (0.0, 1.0)},
            rebuild_required_by_block={tuple(range(8)): True, (8,): False},
            build_cache=_build_cache,
            log_likelihood_from_cache_fn=_log_likelihood_from_cache,
            resolved_complementary_de_jump_block=(tuple(range(8)), True, 8),
        )


def test_swig_cached_likelihood_remains_consistent():
    sampler = _make_sampler()
    initial = jax.random.uniform(jax.random.key(1), (24, 2))
    sampler.sample(jax.random.key(2), initial)
    result = sampler.get_samples()
    expected = jax.vmap(_log_likelihood)(jnp.asarray(result["samples"]))
    np.testing.assert_allclose(result["log_likelihood"], expected, rtol=1e-10)
    diagnostics = sampler.get_diagnostics()
    assert diagnostics["n_de_jump_attempts"] == 0
    assert diagnostics["n_de_jump_acceptances"] == 0
    assert diagnostics["de_jump_acceptance_rate"] is None
    assert diagnostics["n_likelihood_evaluations_de_jumps"] == 0
    assert diagnostics["n_likelihood_evaluations"] == (
        diagnostics["n_likelihood_evaluations_stepping_out"]
        + diagnostics["n_likelihood_evaluations_shrinking"]
    )


@pytest.mark.parametrize("num_gibbs_sweeps", [1, 2])
def test_periodic_uniform_independence_reports_block_aligned_work(
    num_gibbs_sweeps: int,
):
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["phase"]],
        block_kernel_modes=["slice", "periodic-uniform-independence"],
        num_gibbs_sweeps=num_gibbs_sweeps,
        n_live=24,
        n_delete_frac=0.25,
        termination_dlogz=1.5,
        max_steps=4,
        max_shrinkage=30,
    )
    sampler = BlackJAXSwiGSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
        periodic={1: (0.0, 1.0)},
        rebuild_required_by_block={(0,): True, (1,): False},
        build_cache=_build_cache,
        log_likelihood_from_cache_fn=_log_likelihood_from_cache,
    )

    sampler.sample(
        jax.random.key(70),
        jax.random.uniform(jax.random.key(71), (24, 2)),
    )
    diagnostics = sampler.get_diagnostics()
    assert not any("complementary_de" in key for key in diagnostics)
    replacements = 6 * diagnostics["n_iterations"]
    expected_attempts = num_gibbs_sweeps * replacements
    assert diagnostics["n_periodic_uniform_independence_attempts"] == expected_attempts
    assert (
        diagnostics["n_likelihood_evaluations_periodic_uniform_independence"]
        == expected_attempts
    )
    attempts = diagnostics["periodic_uniform_independence_attempts_by_block_history"]
    acceptances = diagnostics[
        "periodic_uniform_independence_acceptances_by_block_history"
    ]
    assert attempts.shape == acceptances.shape
    assert attempts.shape[-1] == 1
    np.testing.assert_array_equal(attempts, np.full_like(attempts, num_gibbs_sweeps))
    block = diagnostics["periodic_uniform_independence_blocks"][0]
    assert block["parameters"] == ["phase"]
    assert block["requires_waveform_rebuild"] is False
    assert block["attempts_per_replacement"] == num_gibbs_sweeps
    assert block["n_attempts"] == expected_attempts
    assert block["n_acceptances"] == int(acceptances.sum())
    assert diagnostics["n_likelihood_evaluations"] == (
        diagnostics["n_likelihood_evaluations_stepping_out"]
        + diagnostics["n_likelihood_evaluations_shrinking"]
        + expected_attempts
    )
    assert diagnostics["n_slice_updates"] == expected_attempts
    assert diagnostics["n_likelihood_evaluations_physical"] == (
        diagnostics["n_likelihood_evaluations"] + 2 * expected_attempts
    )


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


def test_periodic_uniform_independence_requires_declared_periodic_bounds():
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["phase"]],
        block_kernel_modes=["slice", "periodic-uniform-independence"],
        num_gibbs_sweeps=1,
        n_live=24,
    )

    with pytest.raises(ValueError, match="declared periodic bounds"):
        BlackJAXSwiGSampler(
            n_dims=2,
            log_prior_fn=_log_prior,
            log_likelihood_fn=_log_likelihood,
            log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
            config=config,
            periodic=None,
            rebuild_required_by_block={(0,): True, (1,): False},
            build_cache=_build_cache,
            log_likelihood_from_cache_fn=_log_likelihood_from_cache,
        )


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


def test_pre_fsm_lockstep_selects_reference_builder_and_covariances(monkeypatch):
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        scheduler="pre-fsm-lockstep",
        n_live=24,
        n_delete_frac=0.25,
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
    selected = {}

    def lockstep_builder(**kwargs):
        selected["builder"] = kwargs
        return lambda *args, **kwargs: None

    def reject_fsm_builder(**kwargs):
        raise AssertionError(f"FSM builder selected: {kwargs}")

    def replicated_builder(
        constrained_step,
        *,
        n_inner_steps,
        update_inner_kernel_params_fn,
        n_delete,
        mesh,
    ):
        selected["update"] = update_inner_kernel_params_fn
        return lambda *args, **kwargs: None

    monkeypatch.setattr(swig, "_build_swig_constrained_step_lockstep", lockstep_builder)
    monkeypatch.setattr(swig, "_build_swig_constrained_step", reject_fsm_builder)
    monkeypatch.setattr(
        swig,
        "build_replicated_from_mcmc_kernel",
        replicated_builder,
    )

    mesh = object()
    sampler._build_nested_sampler(6, mesh=mesh)
    positions = jnp.asarray([[0.1, 0.2], [0.3, 0.4], [0.8, 0.6]])
    state = SimpleNamespace(particles=SimpleNamespace(position=positions))
    params = sampler._inner_kernel_params_fn_for_mesh(mesh)(
        jax.random.key(0), state, None
    )

    assert selected["builder"]
    assert selected["builder"]["num_slice_steps_by_block"] is None
    assert selected["update"] is sampler._fsm_update_inner_kernel_params_fn
    assert set(params) == {"block_covariances"}
    assert sampler.sampler_name == "BlackJAX SwiG pre-FSM lockstep"


def test_swig_forwards_explicit_block_slice_budget_to_fsm_builder(monkeypatch):
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        num_gibbs_sweeps=1,
        num_slice_steps_by_block=[3, 1],
        n_live=24,
        n_delete_frac=0.25,
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
    selected = {}

    def fsm_builder(**kwargs):
        selected.update(kwargs)
        return lambda *args, **kwargs: None

    def from_mcmc_builder(*args, **kwargs):
        return lambda *args, **kwargs: None

    monkeypatch.setattr(swig, "_build_swig_constrained_step", fsm_builder)
    monkeypatch.setattr(swig, "build_from_mcmc_kernel", from_mcmc_builder)

    sampler._build_nested_sampler(6)

    assert selected["num_slice_steps_by_block"] == [3, 1]


def _make_de_mix_sampler(
    *,
    resolved_de_jump_blocks=(),
    **config_overrides,
) -> BlackJAXSwiGSampler:
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        n_live=24,
        n_delete_frac=0.25,
        termination_dlogz=1.5,
        max_steps=4,
        max_shrinkage=30,
        direction_mode="de-mix",
        **config_overrides,
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
        resolved_de_jump_blocks=resolved_de_jump_blocks,
    )


def test_de_mix_direction_is_a_live_pair_difference_restricted_to_block():
    live = jnp.asarray(
        [
            [0.0, 10.0, -1.0],
            [1.0, 20.0, -2.0],
            [4.0, 40.0, -8.0],
            [9.0, 90.0, -27.0],
        ]
    )
    block = jnp.asarray((0, 2))
    factor = jnp.linalg.cholesky(jnp.eye(2))
    direction = swig._sample_de_mix_direction(
        jax.random.key(7),
        live,
        jnp.asarray([1.0, 1.0, 1.0, 1.0]),
        jnp.asarray([-100.0, -100.0, -100.0]),
        -jnp.inf,
        block,
        1.0,  # always take the DE branch
        swig._sample_direction_from_covariance_factor,
        factor,
        jnp.zeros(2),
    )
    differences = jnp.stack(
        [
            (live[a] - live[b])[block]
            for a in range(live.shape[0])
            for b in range(live.shape[0])
            if a != b
        ]
    )
    assert direction.shape == (2,)
    matches = jnp.all(jnp.isclose(differences, direction[None, :]), axis=1)
    assert bool(jnp.any(matches))


def test_de_mix_pair_excludes_parent_and_points_below_the_active_contour():
    live = jnp.asarray(
        [
            [0.0, 10.0],  # current parent
            [1.0, 20.0],  # batch-dead at the contour
            [4.0, 40.0],
            [9.0, 90.0],
        ]
    )
    direction = swig._sample_de_mix_direction(
        jax.random.key(4),
        live,
        jnp.asarray([2.0, 0.0, 3.0, 4.0]),
        live[0],
        0.0,
        jnp.asarray((0, 1)),
        1.0,
        swig._sample_direction_from_covariance_factor,
        jnp.eye(2),
        jnp.zeros(2),
    )

    survivor_difference = live[2] - live[3]
    assert bool(
        jnp.allclose(direction, survivor_difference)
        | jnp.allclose(direction, -survivor_difference)
    )


def test_de_mix_update_params_carry_live_positions():
    sampler = _make_de_mix_sampler()
    positions = jnp.asarray([[0.1, 0.2], [0.3, 0.4], [0.8, 0.6]])
    loglikelihood = jnp.asarray([-3.0, -2.0, -1.0])
    state = SimpleNamespace(
        particles=SimpleNamespace(
            position=positions,
            loglikelihood=loglikelihood,
        )
    )
    covariance_params = sampler._update_inner_kernel_params_fn(None, state, None)
    factor_params = sampler._fsm_update_inner_kernel_params_fn(None, state, None)
    assert set(covariance_params) == {
        "block_covariances",
        "live_positions",
        "live_loglikelihoods",
    }
    assert set(factor_params) == {
        "block_covariance_factors",
        "live_positions",
        "live_loglikelihoods",
    }
    np.testing.assert_array_equal(covariance_params["live_positions"], positions)
    np.testing.assert_array_equal(
        covariance_params["live_loglikelihoods"], loglikelihood
    )


def test_de_mix_sampler_keeps_cached_likelihood_consistent():
    sampler = _make_de_mix_sampler()
    initial = jax.random.uniform(jax.random.key(11), (24, 2))
    sampler.sample(jax.random.key(12), initial)
    result = sampler.get_samples()
    expected = jax.vmap(_log_likelihood)(jnp.asarray(result["samples"]))
    np.testing.assert_allclose(result["log_likelihood"], expected, rtol=1e-10)


def test_de_jump_applies_a_live_pair_displacement_when_in_contour():
    live = jnp.asarray([[1.0, 2.0], [0.0, 0.0]])
    key = jax.random.key(0)
    start = swig.CachedSliceState(
        position=jnp.asarray([0.25, 0.25]),
        logdensity=jnp.asarray(0.0),
        loglikelihood=jnp.asarray(1.0),
        loglikelihood_birth=jnp.asarray(-jnp.inf),
        cache=jnp.asarray(0.0),
    )
    moved, num_acceptances = swig._apply_de_jumps(
        key,
        start,
        jnp.asarray(0.0),
        live,
        1,
        lambda position, cache: (jnp.asarray(0.0), jnp.asarray(2.0), cache),
        lambda position: position,
    )
    jump_key = jax.random.split(key, 1)[0]
    pair_key, _ = jax.random.split(jump_key)
    pair = jax.random.choice(pair_key, live.shape[0], shape=(2,), replace=False)
    expected = start.position + live[pair[0]] - live[pair[1]]
    np.testing.assert_array_equal(moved.position, expected)
    np.testing.assert_allclose(moved.loglikelihood, 2.0)
    np.testing.assert_array_equal(num_acceptances, 1)


def test_de_jump_rejects_out_of_contour_proposals():
    live = jnp.asarray([[1.0, 2.0], [0.0, 0.0]])
    start = swig.CachedSliceState(
        position=jnp.asarray([0.25, 0.25]),
        logdensity=jnp.asarray(0.0),
        loglikelihood=jnp.asarray(1.0),
        loglikelihood_birth=jnp.asarray(-jnp.inf),
        cache=jnp.asarray(0.0),
    )
    unmoved, num_acceptances = swig._apply_de_jumps(
        jax.random.key(0),
        start,
        jnp.asarray(0.0),
        live,
        3,
        lambda position, cache: (jnp.asarray(0.0), jnp.asarray(-5.0), cache),
        lambda position: position,
    )
    np.testing.assert_array_equal(unmoved.position, start.position)
    np.testing.assert_allclose(unmoved.loglikelihood, 1.0)
    np.testing.assert_array_equal(num_acceptances, 0)


def test_periodic_uniform_independence_uses_one_call_and_preserves_rejected_cache():
    start = swig.CachedSliceState(
        position=jnp.asarray([0.25, 0.75]),
        logdensity=jnp.asarray(0.0),
        loglikelihood=jnp.asarray(1.0),
        loglikelihood_birth=jnp.asarray(-1.0),
        cache=jnp.asarray([7.0, 8.0]),
    )
    evaluated = []

    def reject_candidate(position, cache):
        evaluated.append((position, cache))
        return jnp.asarray(-jnp.inf), jnp.asarray(2.0), jnp.asarray([9.0, 10.0])

    actual, accepted = swig._apply_periodic_uniform_independence(
        jax.random.key(14),
        start,
        jnp.asarray(0.0),
        parameter_index=1,
        lower=0.0,
        upper=2.0,
        eval_candidate=reject_candidate,
    )

    assert len(evaluated) == 1
    assert not bool(accepted)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(start), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_periodic_uniform_independence_uses_full_nonuniform_prior_ratio():
    key = jax.random.key(19)
    start = swig.CachedSliceState(
        position=jnp.asarray([0.8]),
        logdensity=jnp.log(jnp.asarray(1.6)),
        loglikelihood=jnp.asarray(1.0),
        loglikelihood_birth=jnp.asarray(0.0),
        cache=jnp.asarray(0.8),
    )

    def evaluate(position, cache):
        del cache
        return jnp.log(2.0 * position[0]), jnp.asarray(1.0), position[0]

    actual, accepted = swig._apply_periodic_uniform_independence(
        key,
        start,
        jnp.asarray(0.0),
        parameter_index=0,
        lower=0.0,
        upper=1.0,
        eval_candidate=evaluate,
    )

    proposal_key, accept_key = jax.random.split(key)
    proposed = jax.random.uniform(proposal_key)
    expected_accept = jnp.log(jax.random.uniform(accept_key)) < jnp.log(
        proposed / start.position[0]
    )
    assert bool(accepted) == bool(expected_accept)
    np.testing.assert_array_equal(
        actual.position,
        jnp.where(expected_accept, proposed, start.position[0])[None],
    )


def test_periodic_uniform_independence_targets_nonuniform_periodic_prior():
    initial = swig.CachedSliceState(
        position=jnp.asarray([0.5]),
        logdensity=jnp.log(jnp.asarray(1.0)),
        loglikelihood=jnp.asarray(0.0),
        loglikelihood_birth=jnp.asarray(-1.0),
        cache=jnp.asarray(0.5),
    )

    def evaluate(position, cache):
        del cache
        return jnp.log(2.0 * position[0]), jnp.asarray(0.0), position[0]

    def transition(state, key):
        state, _ = swig._apply_periodic_uniform_independence(
            key,
            state,
            jnp.asarray(-1.0),
            parameter_index=0,
            lower=0.0,
            upper=1.0,
            eval_candidate=evaluate,
        )
        return state, state.position[0]

    keys = jax.random.split(jax.random.key(73), 40_000)
    _, draws = jax.jit(lambda: jax.lax.scan(transition, initial, keys))()
    # The normalized target density is p(x)=2x on the periodic support [0,1).
    assert float(jnp.mean(draws[2_000:])) == pytest.approx(2.0 / 3.0, abs=0.01)
    assert float(jnp.mean(draws[2_000:] <= 0.5)) == pytest.approx(0.25, abs=0.015)


def test_de_jump_can_target_only_selected_coordinates():
    live = jnp.asarray([[0.4, 0.3, 0.2], [0.1, 0.2, 0.0]])
    start = swig.CachedSliceState(
        position=jnp.asarray([0.25, 0.25, 0.25]),
        logdensity=jnp.asarray(0.0),
        loglikelihood=jnp.asarray(1.0),
        loglikelihood_birth=jnp.asarray(-jnp.inf),
        cache=jnp.asarray(0.0),
    )

    moved, num_acceptances = swig._apply_de_jumps(
        jax.random.key(0),
        start,
        jnp.asarray(0.0),
        live,
        1,
        lambda position, cache: (jnp.asarray(0.0), jnp.asarray(2.0), cache),
        lambda position: position,
        (1, 2),
    )

    np.testing.assert_allclose(moved.position[0], start.position[0])
    differences = jnp.stack(
        [
            (live[0] - live[1])[jnp.asarray((1, 2))],
            (live[1] - live[0])[jnp.asarray((1, 2))],
        ]
    )
    moved_difference = (
        moved.position[jnp.asarray((1, 2))] - start.position[jnp.asarray((1, 2))]
    )
    assert bool(jnp.any(jnp.all(jnp.isclose(differences, moved_difference), axis=1)))
    np.testing.assert_array_equal(num_acceptances, 1)


def test_de_jump_sampler_keeps_cached_likelihood_consistent():
    sampler = _make_de_mix_sampler(num_de_jumps=2)
    initial = jax.random.uniform(jax.random.key(21), (24, 2))
    sampler.sample(jax.random.key(22), initial)
    result = sampler.get_samples()
    expected = jax.vmap(_log_likelihood)(jnp.asarray(result["samples"]))
    np.testing.assert_allclose(result["log_likelihood"], expected, rtol=1e-10)
    diagnostics = sampler.get_diagnostics()
    expected_attempts = 2 * 6 * diagnostics["n_iterations"]
    assert diagnostics["n_de_jump_attempts"] == expected_attempts
    assert diagnostics["n_likelihood_evaluations_de_jumps"] == expected_attempts
    assert 0 <= diagnostics["n_de_jump_acceptances"] <= expected_attempts
    assert diagnostics["de_jump_acceptance_rate"] == (
        diagnostics["n_de_jump_acceptances"] / expected_attempts
    )
    assert diagnostics["n_likelihood_evaluations"] == (
        diagnostics["n_likelihood_evaluations_stepping_out"]
        + diagnostics["n_likelihood_evaluations_shrinking"]
        + expected_attempts
    )


def test_full_and_targeted_de_jump_accounting_and_cache_consistency():
    sampler = _make_de_mix_sampler(
        num_de_jumps=2,
        de_jump_blocks=[{"parameters": ["slow", "fast"], "attempts": 1}],
        resolved_de_jump_blocks=(((0, 1), True, 1),),
    )
    initial = jax.random.uniform(jax.random.key(31), (24, 2))
    sampler.sample(jax.random.key(32), initial)
    result = sampler.get_samples()
    expected = jax.vmap(_log_likelihood)(jnp.asarray(result["samples"]))
    np.testing.assert_allclose(result["log_likelihood"], expected, rtol=1e-10)

    diagnostics = sampler.get_diagnostics()
    replacements = 6 * diagnostics["n_iterations"]
    assert diagnostics["n_de_jump_attempts"] == 3 * replacements
    assert diagnostics["n_likelihood_evaluations_de_jumps"] == 3 * replacements
    assert diagnostics["n_likelihood_evaluations_de_jumps_waveform_rebuild"] == (
        3 * replacements
    )
    assert diagnostics["n_likelihood_evaluations_de_jumps_cache_hit"] == 0
    assert len(diagnostics["targeted_de_jump_blocks"]) == 1
    targeted = diagnostics["targeted_de_jump_blocks"][0]
    assert targeted["parameters"] == ["slow", "fast"]
    assert targeted["requires_waveform_rebuild"] is True
    assert targeted["attempts_per_replacement"] == 1
    assert targeted["n_attempts"] == replacements
    assert 0 <= targeted["n_acceptances"] <= replacements
    assert targeted["acceptance_rate"] == targeted["n_acceptances"] / replacements


def test_covariance_mode_leaves_update_param_keys_unchanged():
    sampler = _make_sampler()
    positions = jnp.asarray([[0.1, 0.2], [0.3, 0.4], [0.8, 0.6]])
    state = SimpleNamespace(particles=SimpleNamespace(position=positions))
    assert set(sampler._update_inner_kernel_params_fn(None, state, None)) == {
        "block_covariances"
    }
    assert set(sampler._fsm_update_inner_kernel_params_fn(None, state, None)) == {
        "block_covariance_factors"
    }


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

