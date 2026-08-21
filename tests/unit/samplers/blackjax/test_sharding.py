"""Multi-device tests for BlackJAX NSS and SwiG."""

from __future__ import annotations

import pickle
import re
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import StateWithLogLikelihood, init_state_strategy
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler
from jimgw.samplers.blackjax.sharding import (
    _pack_particles,
    _particle_dtypes,
    _unpack_particles,
    make_live_mesh,
    place_key,
    place_replicated_state,
)
from jimgw.samplers.blackjax.swig import BlackJAXSwiGSampler
from jimgw.samplers.config import BlackJAXNSSConfig, BlackJAXSwiGConfig

_HAS_FOUR_DEVICES = jax.local_device_count() >= 4


def _log_prior(position):
    return jnp.where(jnp.all((position >= 0.0) & (position <= 1.0)), 0.0, -jnp.inf)


def _build_cache(position):
    return position[0] ** 2


def _log_likelihood_from_cache(position, cache):
    return -20.0 * ((cache - 0.25) ** 2 + (position[1] - 0.5) ** 2)


def _log_likelihood(position):
    return _log_likelihood_from_cache(position, _build_cache(position))


@pytest.fixture(scope="module", params=(2, 4))
def nss_transition_results(request):
    """Run matching one- and multi-device transitions from one frozen state."""
    n_devices = request.param
    if jax.local_device_count() < n_devices:
        pytest.skip(f"requires {n_devices} local JAX devices")

    n_live = 8
    n_delete = 4
    config = BlackJAXNSSConfig(
        n_live=n_live,
        n_delete_frac=n_delete / n_live,
        num_inner_steps_per_dim=1,
        termination_dlogz=2.0,
        n_devices=n_devices,
    )
    sampler = BlackJAXNSSSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
    )
    positions = jax.random.uniform(jax.random.key(10), (n_live, 2))
    single_init = partial(
        init_state_strategy,
        logprior_fn=_log_prior,
        loglikelihood_fn=_log_likelihood,
    )
    one_device_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._update_inner_kernel_params_fn,
    )
    fsm_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._fsm_update_inner_kernel_params_fn,
    )
    mesh = make_live_mesh(n_devices, n_live, n_delete)
    assert mesh is not None

    keys = jax.random.split(jax.random.key(11), 5)
    one_device_step = (
        jax.jit(sampler._build_nested_sampler(n_delete).step)
        .lower(keys[0], one_device_state)
        .compile()
    )
    replicated_state = place_replicated_state(fsm_state, mesh)
    replicated_key = place_key(keys[0], mesh)
    four_device_step = jax.jit(sampler._build_nested_sampler(n_delete, mesh).step)
    executable = four_device_step.lower(replicated_key, replicated_state).compile()

    one_device_results = []
    sharded_results = []
    for key in keys:
        one_device_result = one_device_step(key, one_device_state)
        sharded_result = executable(place_key(key, mesh), replicated_state)
        one_device_results.append(one_device_result)
        sharded_results.append(sharded_result)
        one_device_state = one_device_result[0]
        replicated_state = sharded_result[0]

    return n_devices, one_device_results, sharded_results, executable.as_text()


@pytest.fixture(scope="module")
def swig_transition_results():
    """Compile matching SwiG transitions to cover its cache-aware chain path."""
    if not _HAS_FOUR_DEVICES:
        pytest.skip("run with XLA_FLAGS=--xla_force_host_platform_device_count=4")

    n_live = 8
    n_delete = 4
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        num_de_jumps=2,
        de_jump_blocks=[{"parameters": ["slow", "fast"], "attempts": 1}],
        n_live=n_live,
        n_delete_frac=n_delete / n_live,
        num_gibbs_sweeps=1,
        max_steps=3,
        max_shrinkage=20,
        termination_dlogz=2.0,
        n_devices=4,
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
        resolved_de_jump_blocks=(((0, 1), True, 1),),
    )
    positions = jax.random.uniform(jax.random.key(20), (n_live, 2))
    single_init = partial(
        init_state_strategy,
        logprior_fn=_log_prior,
        loglikelihood_fn=_log_likelihood,
    )
    one_device_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._update_inner_kernel_params_fn,
    )
    fsm_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._fsm_update_inner_kernel_params_fn,
    )
    mesh = make_live_mesh(4, n_live, n_delete)
    assert mesh is not None

    key = jax.random.key(21)
    one_device_result = jax.jit(sampler._build_nested_sampler(n_delete).step)(
        key, one_device_state
    )
    replicated_state = place_replicated_state(fsm_state, mesh)
    replicated_key = place_key(key, mesh)
    sharded_step = jax.jit(sampler._build_nested_sampler(n_delete, mesh).step)
    executable = sharded_step.lower(replicated_key, replicated_state).compile()
    sharded_result = executable(replicated_key, replicated_state)
    return one_device_result, sharded_result, executable.as_text()


def test_make_live_mesh_rejects_unavailable_devices():
    with pytest.raises(ValueError, match="only .* local JAX devices"):
        make_live_mesh(jax.local_device_count() + 1, 16, 4)


def test_make_live_mesh_rejects_indivisible_n_live(monkeypatch):
    # Fake enough devices to clear the availability check without needing
    # real hardware, so the n_live divisibility check below it is reachable.
    monkeypatch.setattr(jax, "local_devices", lambda: [object(), object()])
    with pytest.raises(ValueError, match="n_live=11 must be divisible by n_devices=2"):
        make_live_mesh(2, 11, 4)


def test_make_live_mesh_rejects_indivisible_n_delete(monkeypatch):
    monkeypatch.setattr(jax, "local_devices", lambda: [object(), object()])
    with pytest.raises(ValueError, match="n_delete=3 must be divisible by n_devices=2"):
        make_live_mesh(2, 12, 3)


def test_replicated_steps_are_pathwise_equivalent_and_have_target_placement(
    nss_transition_results,
):
    _, one_device_results, sharded_results, _ = nss_transition_results
    for one_device_result, sharded_result in zip(
        one_device_results, sharded_results, strict=True
    ):
        one_device_host = jax.device_get(one_device_result)
        sharded_host = jax.device_get(sharded_result)
        one_device_state, one_device_info = one_device_host
        sharded_state, sharded_info = sharded_host
        for expected_tree, actual_tree in (
            (one_device_state.particles, sharded_state.particles),
            (one_device_state.integrator, sharded_state.integrator),
            (one_device_info, sharded_info),
        ):
            for expected, actual in zip(
                jax.tree.leaves(expected_tree),
                jax.tree.leaves(actual_tree),
                strict=True,
            ):
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=1e-14,
                    atol=1e-14,
                    equal_nan=True,
                )

        covariance = one_device_state.inner_kernel_params["cov"]
        covariance_factor = sharded_state.inner_kernel_params["covariance_factor"]
        np.testing.assert_allclose(
            covariance_factor @ covariance_factor.T,
            covariance,
            rtol=1e-14,
            atol=1e-14,
        )

    state, info = sharded_results[-1]
    for leaf in jax.tree.leaves(state):
        assert leaf.sharding.is_fully_replicated
    for leaf in jax.tree.leaves(info):
        assert not leaf.sharding.is_fully_replicated
        assert leaf.sharding.spec == P("replacement")


def test_replicated_step_lowers_to_one_endpoint_all_gather(nss_transition_results):
    _, _, _, hlo = nss_transition_results

    assert len(re.findall(r"\ball-gather\(", hlo)) == 1
    assert "all-reduce(" not in hlo
    assert "collective-permute(" not in hlo


def test_nss_fsm_outer_step_migrates_legacy_covariance_params():
    n_live = 8
    n_delete = 4
    config = BlackJAXNSSConfig(
        n_live=n_live,
        n_delete_frac=n_delete / n_live,
        num_inner_steps_per_dim=1,
        termination_dlogz=2.0,
    )
    sampler = BlackJAXNSSSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
    )
    positions = jax.random.uniform(jax.random.key(12), (n_live, 2))
    single_init = partial(
        init_state_strategy,
        logprior_fn=_log_prior,
        loglikelihood_fn=_log_likelihood,
    )
    legacy_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._update_inner_kernel_params_fn,
    )
    assert set(legacy_state.inner_kernel_params) == {"cov"}

    mesh = Mesh(np.asarray(jax.devices()[:1]), axis_names=("replacement",))
    state = place_replicated_state(legacy_state, mesh)
    key = place_key(jax.random.key(13), mesh)
    new_state, _ = jax.jit(sampler._build_nested_sampler(n_delete, mesh).step)(
        key,
        state,
    )

    assert set(new_state.inner_kernel_params) == {"covariance_factor"}
    covariance_factor = new_state.inner_kernel_params["covariance_factor"]
    covariance = jnp.cov(new_state.particles.position, ddof=0, rowvar=False)
    np.testing.assert_allclose(
        covariance_factor @ covariance_factor.T,
        covariance,
        rtol=1e-14,
        atol=1e-14,
    )


def test_nss_checkpoint_params_are_retargeted_for_selected_topology():
    n_live = 8
    config = BlackJAXNSSConfig(
        n_live=n_live,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=1,
        termination_dlogz=2.0,
    )
    sampler = BlackJAXNSSSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
    )
    positions = jax.random.uniform(jax.random.key(14), (n_live, 2))
    single_init = partial(
        init_state_strategy,
        logprior_fn=_log_prior,
        loglikelihood_fn=_log_likelihood,
    )
    covariance_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._update_inner_kernel_params_fn,
    )
    factor_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._fsm_update_inner_kernel_params_fn,
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), axis_names=("replacement",))

    for source, topology in ((factor_state, None), (covariance_state, mesh)):
        retargeted = sampler._normalise_inner_kernel_params_for_mesh(source, topology)
        covariance = jnp.cov(retargeted.particles.position, ddof=0, rowvar=False)
        if topology is None:
            assert set(retargeted.inner_kernel_params) == {"cov"}
            np.testing.assert_allclose(
                retargeted.inner_kernel_params["cov"], covariance, rtol=1e-14
            )
        else:
            assert set(retargeted.inner_kernel_params) == {"covariance_factor"}
            factor = retargeted.inner_kernel_params["covariance_factor"]
            np.testing.assert_allclose(
                factor @ factor.T, covariance, rtol=1e-14, atol=1e-14
            )


def test_swig_checkpoint_params_are_retargeted_for_selected_topology():
    n_live = 8
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        n_live=n_live,
        n_delete_frac=0.5,
        num_gibbs_sweeps=1,
        termination_dlogz=2.0,
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
    positions = jax.random.uniform(jax.random.key(15), (n_live, 2))
    single_init = partial(
        init_state_strategy,
        logprior_fn=_log_prior,
        loglikelihood_fn=_log_likelihood,
    )
    covariance_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._update_inner_kernel_params_fn,
    )
    factor_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._fsm_update_inner_kernel_params_fn,
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), axis_names=("replacement",))

    nonmesh = sampler._normalise_inner_kernel_params_for_mesh(factor_state, None)
    sharded = sampler._normalise_inner_kernel_params_for_mesh(covariance_state, mesh)
    assert set(nonmesh.inner_kernel_params) == {"block_covariances"}
    assert set(sharded.inner_kernel_params) == {"block_covariance_factors"}
    for covariance, factor in zip(
        nonmesh.inner_kernel_params["block_covariances"],
        sharded.inner_kernel_params["block_covariance_factors"],
        strict=True,
    ):
        np.testing.assert_allclose(
            factor @ factor.T, covariance, rtol=1e-14, atol=1e-14
        )


def test_particle_pack_round_trip_preserves_mixed_dtypes():
    particles = StateWithLogLikelihood(
        position=jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
        logdensity=jnp.arange(3, dtype=jnp.float16),
        loglikelihood=jnp.arange(3, dtype=jnp.float32),
        loglikelihood_birth=jnp.arange(3, dtype=jnp.bfloat16),
    )

    unpacked = _unpack_particles(
        _pack_particles(particles), 2, _particle_dtypes(particles)
    )
    for expected, actual in zip(
        jax.tree.leaves(particles), jax.tree.leaves(unpacked), strict=True
    ):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not _HAS_FOUR_DEVICES,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
def test_distributed_initialisation_evaluates_each_likelihood_once():
    evaluated = []

    def counted_log_likelihood(position):
        jax.debug.callback(lambda marker: evaluated.append(float(marker)), position[0])
        return _log_likelihood(position)

    config = BlackJAXNSSConfig(
        n_live=8,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=1,
        termination_dlogz=1e6,
        n_devices=4,
    )
    sampler = BlackJAXNSSSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=counted_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + counted_log_likelihood(x),
        config=config,
    )
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    markers = jnp.arange(config.n_live, dtype=dtype) / config.n_live
    initial = jnp.stack((markers, jnp.full_like(markers, 0.5)), axis=1)

    class InitialisationComplete(RuntimeError):
        pass

    class StopBeforeFirstStep:
        @staticmethod
        def step(rng_key, state):
            del rng_key, state
            raise InitialisationComplete

    sampler._build_nested_sampler = lambda *args, **kwargs: StopBeforeFirstStep()  # type: ignore[method-assign]
    with pytest.raises(InitialisationComplete):
        sampler.sample(jax.random.key(30), initial)

    np.testing.assert_array_equal(
        np.sort(np.asarray(evaluated)),
        np.asarray(markers),
    )


def test_replicated_swig_step_is_pathwise_equivalent(swig_transition_results):
    one_device_result, sharded_result, _ = swig_transition_results
    one_device_state, one_device_info = jax.device_get(one_device_result)
    sharded_state, sharded_info = jax.device_get(sharded_result)
    for expected_tree, actual_tree in (
        (one_device_state.particles, sharded_state.particles),
        (one_device_state.integrator, sharded_state.integrator),
        (one_device_info, sharded_info),
    ):
        for expected, actual in zip(
            jax.tree.leaves(expected_tree),
            jax.tree.leaves(actual_tree),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual, expected, rtol=1e-14, atol=1e-14, equal_nan=True
            )

    covariances = one_device_state.inner_kernel_params["block_covariances"]
    factors = sharded_state.inner_kernel_params["block_covariance_factors"]
    for covariance, factor in zip(covariances, factors, strict=True):
        np.testing.assert_allclose(
            factor @ factor.T,
            covariance,
            rtol=1e-14,
            atol=1e-14,
        )

    state, info = sharded_result
    assert all(leaf.sharding.is_fully_replicated for leaf in jax.tree.leaves(state))
    assert all(leaf.sharding.spec == P("replacement") for leaf in jax.tree.leaves(info))


@pytest.mark.skipif(
    not _HAS_FOUR_DEVICES,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
@pytest.mark.parametrize(
    (
        "direction_mode",
        "num_slice_steps_by_block",
        "block_kernel_modes",
        "periodic",
        "complementary_de_attempts",
    ),
    [
        pytest.param(
            "covariance-basis-8d",
            None,
            None,
            None,
            None,
            id="h2-covariance-basis",
        ),
        pytest.param(
            "covariance",
            [5, 1, 1, 2, 1, 2],
            None,
            None,
            None,
            id="h3-explicit-block-budget",
        ),
        pytest.param(
            "covariance",
            None,
            [
                "slice",
                "periodic-uniform-independence",
                "periodic-uniform-independence",
                "slice",
                "periodic-uniform-independence",
                "slice",
            ],
            {8: (0.0, 1.0), 9: (0.0, 1.0), 12: (0.0, 1.0)},
            None,
            id="h4-periodic-uniform-independence",
        ),
        pytest.param(
            "covariance",
            None,
            [
                "slice",
                "periodic-uniform-independence",
                "periodic-uniform-independence",
                "slice",
                "periodic-uniform-independence",
                "slice",
            ],
            {8: (0.0, 1.0), 9: (0.0, 1.0), 12: (0.0, 1.0)},
            4,
            id="h5-complementary-de",
        ),
        pytest.param(
            "covariance",
            None,
            [
                "slice",
                "periodic-uniform-independence",
                "periodic-uniform-independence",
                "slice",
                "periodic-uniform-independence",
                "slice",
            ],
            {8: (0.0, 1.0), 9: (0.0, 1.0), 12: (0.0, 1.0)},
            8,
            id="h6-complementary-de",
        ),
    ],
)
def test_swig_fixed_slice_budget_is_pathwise_equivalent_on_d1_and_d4(
    direction_mode,
    num_slice_steps_by_block,
    block_kernel_modes,
    periodic,
    complementary_de_attempts,
):
    block_indices = (
        tuple(range(8)),
        (8,),
        (9,),
        (10, 11),
        (12,),
        (13, 14),
    )
    rebuild_required_by_block = {
        indices: block < 3 for block, indices in enumerate(block_indices)
    }
    block_names = [
        [f"parameter_{index}" for index in indices] for indices in block_indices
    ]
    n_live = 16
    n_delete = 4
    config = BlackJAXSwiGConfig(
        blocks=block_names,
        direction_mode=direction_mode,
        num_slice_steps_by_block=num_slice_steps_by_block,
        block_kernel_modes=block_kernel_modes,
        complementary_de_jump_block=(
            {"parameters": block_names[0], "attempts": complementary_de_attempts}
            if complementary_de_attempts is not None
            else None
        ),
        n_live=n_live,
        n_delete_frac=n_delete / n_live,
        num_gibbs_sweeps=1,
        num_inner_steps_per_dim=1,
        max_steps=3,
        max_shrinkage=20,
        termination_dlogz=2.0,
        n_devices=4,
    )

    def build_cache(position):
        return position[:10] ** 2

    def log_likelihood_from_cache(position, cache):
        return -jnp.sum((cache - 0.25) ** 2) - jnp.sum((position[10:] - 0.5) ** 2)

    def log_likelihood(position):
        return log_likelihood_from_cache(position, build_cache(position))

    sampler = BlackJAXSwiGSampler(
        n_dims=15,
        log_prior_fn=_log_prior,
        log_likelihood_fn=log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + log_likelihood(x),
        config=config,
        periodic=periodic,
        rebuild_required_by_block=rebuild_required_by_block,
        build_cache=build_cache,
        log_likelihood_from_cache_fn=log_likelihood_from_cache,
        resolved_complementary_de_jump_block=(
            (block_indices[0], True, complementary_de_attempts)
            if complementary_de_attempts is not None
            else None
        ),
    )
    positions = jax.random.uniform(jax.random.key(40), (n_live, 15))
    single_init = partial(
        init_state_strategy,
        logprior_fn=_log_prior,
        loglikelihood_fn=log_likelihood,
    )
    one_device_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._update_inner_kernel_params_fn,
    )
    sharded_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._fsm_update_inner_kernel_params_fn,
    )
    mesh = make_live_mesh(4, n_live, n_delete)
    assert mesh is not None
    key = jax.random.key(41)

    one_device_result = jax.jit(sampler._build_nested_sampler(n_delete).step)(
        key, one_device_state
    )
    replicated_state = place_replicated_state(sharded_state, mesh)
    replicated_key = place_key(key, mesh)
    sharded_result = jax.jit(sampler._build_nested_sampler(n_delete, mesh).step)(
        replicated_key, replicated_state
    )

    one_device_host = jax.device_get(one_device_result)
    sharded_host = jax.device_get(sharded_result)
    for expected_tree, actual_tree in (
        (one_device_host[0].particles, sharded_host[0].particles),
        (one_device_host[0].integrator, sharded_host[0].integrator),
        (one_device_host[1], sharded_host[1]),
    ):
        for expected, actual in zip(
            jax.tree.leaves(expected_tree),
            jax.tree.leaves(actual_tree),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-14,
                atol=1e-14,
                equal_nan=True,
            )

    if complementary_de_attempts is not None:
        info = sharded_host[1].update_info
        parents = np.asarray(info.complementary_de_parent_index).reshape(n_delete)
        donors = np.asarray(info.complementary_de_donor_indices_by_attempt).reshape(
            n_delete, complementary_de_attempts, 2
        )
        assert parents.shape == (n_delete,)
        assert donors.shape == (n_delete, complementary_de_attempts, 2)
        _, dead_idx = jax.lax.top_k(-one_device_state.particles.loglikelihood, n_delete)
        threshold = one_device_state.particles.loglikelihood[dead_idx].max()
        _, inner_key = jax.random.split(key)
        choice_key, _ = jax.random.split(inner_key)
        survivor_weights = (
            one_device_state.particles.loglikelihood > threshold
        ).astype(jnp.float32)
        expected_parents = jax.random.choice(
            choice_key,
            n_live,
            shape=(n_delete,),
            p=survivor_weights / survivor_weights.sum(),
            replace=True,
        )
        np.testing.assert_array_equal(parents, expected_parents)
        assert not np.any(donors == parents[:, None, None])
        incoming_positions = np.asarray(one_device_state.particles.position)
        incoming_loglikelihoods = np.asarray(one_device_state.particles.loglikelihood)
        assert np.all(incoming_loglikelihoods[donors] > float(threshold))
        positions_before = np.asarray(
            info.complementary_de_position_before_by_attempt
        ).reshape(n_delete, complementary_de_attempts, 15)
        proposal_positions = np.asarray(
            info.complementary_de_proposal_position_by_attempt
        ).reshape(n_delete, complementary_de_attempts, 15)
        expected_displacements = (
            incoming_positions[donors[:, :, 0]] - incoming_positions[donors[:, :, 1]]
        )
        np.testing.assert_allclose(
            proposal_positions[:, :, :8] - positions_before[:, :, :8],
            expected_displacements[:, :, :8],
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_array_equal(
            proposal_positions[:, :, 8:], positions_before[:, :, 8:]
        )


def test_replicated_swig_step_has_one_endpoint_all_gather(swig_transition_results):
    _, _, hlo = swig_transition_results
    assert len(re.findall(r"\ball-gather\(", hlo)) == 1
    assert "all-reduce(" not in hlo
    assert "collective-permute(" not in hlo


@pytest.mark.skipif(
    not _HAS_FOUR_DEVICES,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
def test_nss_runs_sharded_and_preserves_diagnostics():
    config = BlackJAXNSSConfig(
        n_live=16,
        n_delete_frac=0.25,
        num_inner_steps_per_dim=1,
        termination_dlogz=2.0,
        n_devices=4,
    )
    sampler = BlackJAXNSSSampler(
        n_dims=2,
        log_prior_fn=_log_prior,
        log_likelihood_fn=_log_likelihood,
        log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
        config=config,
    )

    initial = jax.random.uniform(jax.random.key(0), (16, 2))
    sampler.sample(jax.random.key(1), initial)
    diagnostics = sampler.get_diagnostics()

    assert diagnostics["n_iterations"] > 0
    assert diagnostics["n_likelihood_evaluations"] > 0
    stepping_out = diagnostics["n_stepping_out_history"]
    assert stepping_out.shape[-1] == 2
    assert stepping_out.shape[0] % config.n_devices == 0


@pytest.mark.skipif(
    not _HAS_FOUR_DEVICES,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
def test_swig_runs_sharded_with_consistent_cache():
    config = BlackJAXSwiGConfig(
        blocks=[["slow"], ["fast"]],
        num_de_jumps=2,
        de_jump_blocks=[{"parameters": ["slow", "fast"], "attempts": 1}],
        n_live=16,
        n_delete_frac=0.25,
        num_gibbs_sweeps=1,
        max_steps=3,
        max_shrinkage=20,
        termination_dlogz=2.0,
        n_devices=4,
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
        resolved_de_jump_blocks=(((0, 1), True, 1),),
    )

    initial = jax.random.uniform(jax.random.key(2), (16, 2))
    sampler.sample(jax.random.key(3), initial)
    result = sampler.get_samples()
    expected = jax.vmap(_log_likelihood)(jnp.asarray(result["samples"]))

    np.testing.assert_allclose(result["log_likelihood"], expected, rtol=2e-6)
    diagnostics = sampler.get_diagnostics()
    assert diagnostics["n_likelihood_evaluations"] > 0
    expected_attempts = 3 * 4 * diagnostics["n_iterations"]
    assert diagnostics["n_de_jump_attempts"] == expected_attempts
    assert diagnostics["targeted_de_jump_blocks"][0]["n_attempts"] == (
        4 * diagnostics["n_iterations"]
    )
    assert not hasattr(sampler._final_state.particles, "cache")


@pytest.mark.skipif(
    not _HAS_FOUR_DEVICES,
    reason="run with XLA_FLAGS=--xla_force_host_platform_device_count=4",
)
def test_sharded_checkpoint_is_host_backed_and_resumable(tmp_path, monkeypatch):
    # This test exercises checkpoint state, not the process-global compilation
    # cache. Keeping the cache disabled avoids leaving JAX workers with a path
    # under pytest's soon-to-be-removed temporary directory.
    monkeypatch.setattr(
        BlackJAXNSSConfig,
        "configure_jax_cache",
        lambda self: None,
    )
    config = BlackJAXNSSConfig(
        n_live=16,
        n_delete_frac=0.25,
        num_inner_steps_per_dim=1,
        termination_dlogz=2.0,
        n_devices=4,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1e-9,
    )

    def make_sampler(sampler_config=config):
        return BlackJAXNSSSampler(
            n_dims=2,
            log_prior_fn=_log_prior,
            log_likelihood_fn=_log_likelihood,
            log_posterior_fn=lambda x: _log_prior(x) + _log_likelihood(x),
            config=sampler_config,
        )

    checkpoint = tmp_path / "checkpoint.pkl"
    initial = jax.random.uniform(jax.random.key(4), (16, 2))
    baseline_config = config.model_copy(
        update={"checkpoint_dir": None, "checkpoint_interval": 0.0}
    )
    uninterrupted = make_sampler(baseline_config)
    uninterrupted.sample(jax.random.key(5), initial)

    original_write_checkpoint = BlackJAXNSSConfig.write_checkpoint

    def write_then_interrupt(self, data, label):
        original_write_checkpoint(self, data, label)
        raise RuntimeError("simulated interruption after checkpoint")

    monkeypatch.setattr(
        BlackJAXNSSConfig,
        "write_checkpoint",
        write_then_interrupt,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        make_sampler().sample(jax.random.key(5), initial)

    with checkpoint.open("rb") as stream:
        saved = pickle.load(stream)
    assert isinstance(saved["state"].particles.position, np.ndarray)
    assert saved["n_iter"] == 1

    monkeypatch.setattr(
        BlackJAXNSSConfig,
        "write_checkpoint",
        original_write_checkpoint,
    )

    resumed = make_sampler()
    resumed.sample(jax.random.key(999), initial)
    assert (
        resumed.get_diagnostics()["n_iterations"]
        == uninterrupted.get_diagnostics()["n_iterations"]
    )
    for expected, actual in zip(
        jax.tree.leaves(uninterrupted._final_state),
        jax.tree.leaves(resumed._final_state),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual, expected, rtol=1e-14, atol=1e-14, equal_nan=True
        )
    assert not checkpoint.exists()
