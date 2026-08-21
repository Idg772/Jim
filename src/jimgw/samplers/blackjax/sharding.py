"""Multi-device execution for BlackJAX nested sampling.

The live population is deliberately replicated: Jim's flat live-particle record
is tiny compared with the waveform and likelihood workspace used by each
replacement chain.  Only the replacement-chain axis is sharded, and completed
endpoints are packed into one homogeneous buffer before they are exchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.adaptive import AdaptiveNSState
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.base import NSInfo, StateWithLogLikelihood
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.integrator import update_integrator
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

_REPLACEMENT_AXIS = "replacement"


def make_live_mesh(n_devices: int, n_live: int, n_delete: int) -> Optional[Mesh]:
    """Build a single-host mesh for sharding replacement-chain work.

    ``n_live`` divisibility is retained for distributed initialisation.  The
    steady-state live population itself is replicated.
    """
    if n_devices == 1:
        return None

    devices = jax.local_devices()
    if n_devices > len(devices):
        raise ValueError(
            f"n_devices={n_devices} requested, but only {len(devices)} local "
            "JAX devices are available."
        )
    if n_live % n_devices:
        raise ValueError(f"n_live={n_live} must be divisible by n_devices={n_devices}.")
    if n_delete % n_devices:
        raise ValueError(
            f"n_delete={n_delete} must be divisible by n_devices={n_devices}."
        )

    return Mesh(np.asarray(devices[:n_devices]), axis_names=(_REPLACEMENT_AXIS,))


def replacement_sharding(mesh: Mesh) -> NamedSharding:
    """Return the sharding used for replacement IDs and initial live work."""
    return NamedSharding(mesh, P(_REPLACEMENT_AXIS))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Return the fully replicated sharding for compact sampler state."""
    return NamedSharding(mesh, P())


def place_replicated_state(state: AdaptiveNSState, mesh: Mesh) -> AdaptiveNSState:
    """Replicate the live particles, adaptation parameters, and integrator."""
    replicated = replicated_sharding(mesh)
    return state._replace(
        particles=jax.tree.map(
            lambda x: jax.device_put(x, replicated), state.particles
        ),
        inner_kernel_params=jax.tree.map(
            lambda x: jax.device_put(x, replicated), state.inner_kernel_params
        ),
        integrator=jax.tree.map(
            lambda x: jax.device_put(x, replicated), state.integrator
        ),
    )


def place_key(rng_key, mesh: Mesh):
    """Replicate a sampler PRNG key over the mesh."""
    return jax.device_put(rng_key, replicated_sharding(mesh))


def _particle_arrays(
    particles: StateWithLogLikelihood,
) -> tuple[Array, Array, Array, Array]:
    """Narrow BlackJAX's generic particle annotation to Jim's dense arrays."""
    return (
        cast(Array, particles.position),
        cast(Array, particles.logdensity),
        cast(Array, particles.loglikelihood),
        cast(Array, particles.loglikelihood_birth),
    )


def _validate_packable_particles(particles: StateWithLogLikelihood) -> None:
    """Validate Jim's dense particle representation at trace time."""
    position, *scalar_fields = _particle_arrays(particles)
    if position.ndim != 2:
        raise TypeError(
            "Multi-device nested sampling requires a dense particle position "
            "array with shape (n_particles, n_dims)."
        )

    if any(
        field.ndim != 1 or field.shape[0] != position.shape[0]
        for field in scalar_fields
    ):
        raise TypeError(
            "Particle log-density, likelihood, and birth-likelihood fields must "
            "be one-dimensional and share the position's leading axis."
        )


def _pack_particles(particles: StateWithLogLikelihood):
    """Pack a flat Jim particle record into one collective payload."""
    _validate_packable_particles(particles)
    position, logdensity, loglikelihood, loglikelihood_birth = _particle_arrays(
        particles
    )
    packed_dtype = jnp.result_type(
        position.dtype,
        logdensity.dtype,
        loglikelihood.dtype,
        loglikelihood_birth.dtype,
    )
    return jnp.concatenate(
        (
            position.astype(packed_dtype),
            logdensity[:, None].astype(packed_dtype),
            loglikelihood[:, None].astype(packed_dtype),
            loglikelihood_birth[:, None].astype(packed_dtype),
        ),
        axis=1,
    )


def _particle_dtypes(particles: StateWithLogLikelihood) -> tuple[Any, Any, Any, Any]:
    position, logdensity, loglikelihood, loglikelihood_birth = _particle_arrays(
        particles
    )
    return (
        position.dtype,
        logdensity.dtype,
        loglikelihood.dtype,
        loglikelihood_birth.dtype,
    )


def _unpack_particles(
    packed, n_dims: int, dtypes: tuple[Any, Any, Any, Any]
) -> StateWithLogLikelihood:
    """Restore BlackJAX's particle state from a packed endpoint buffer."""
    return StateWithLogLikelihood(
        position=packed[:, :n_dims].astype(dtypes[0]),
        logdensity=packed[:, n_dims].astype(dtypes[1]),
        loglikelihood=packed[:, n_dims + 1].astype(dtypes[2]),
        loglikelihood_birth=packed[:, n_dims + 2].astype(dtypes[3]),
    )


def replicate_initial_particles(
    particles: StateWithLogLikelihood, mesh: Mesh
) -> StateWithLogLikelihood:
    """Gather distributed initial particle records once into each replica."""
    position, _, _, _ = _particle_arrays(particles)
    dtypes = _particle_dtypes(particles)
    particle_specs = jax.tree.map(lambda _: P(_REPLACEMENT_AXIS), particles)

    def gather(local_particles):
        packed = _pack_particles(local_particles)
        return jax.lax.all_gather(packed, _REPLACEMENT_AXIS, tiled=True)

    packed = jax.shard_map(
        gather,
        mesh=mesh,
        in_specs=(particle_specs,),
        out_specs=P(),
        check_vma=False,
    )(particles)
    return _unpack_particles(packed, position.shape[-1], dtypes)


def _shard_by_replacement_id(mesh: Mesh, value):
    """Retain growing per-replacement history without replicating it."""
    replicated_specs = jax.tree.map(lambda _: P(), value)
    replacement_specs = jax.tree.map(lambda _: P(_REPLACEMENT_AXIS), value)
    n_devices = mesh.shape[_REPLACEMENT_AXIS]
    n_replacements = jax.tree.leaves(value)[0].shape[0]
    if n_replacements % n_devices:
        raise ValueError(
            f"replacement history length {n_replacements} must be divisible by "
            f"mesh size {n_devices}."
        )
    local_size = n_replacements // n_devices

    def take_local_shard(replicated_value):
        start = jax.lax.axis_index(_REPLACEMENT_AXIS) * local_size
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start, local_size, axis=0),
            replicated_value,
        )

    return jax.shard_map(
        take_local_shard,
        mesh=mesh,
        in_specs=(replicated_specs,),
        out_specs=replacement_specs,
        check_vma=False,
    )(value)


def update_with_mcmc_take_last_parent_index(
    constrained_step_fn: Callable,
    n_inner_steps: int,
    n_delete: int,
) -> Callable:
    """D1 MCMC replacement update that preserves each original parent index."""

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        choice_key, sample_key = jax.random.split(rng_key)
        particles = state.particles
        weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
        weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
        start_idx = jax.random.choice(
            choice_key,
            len(weights),
            shape=(n_delete,),
            p=weights / weights.sum(),
            replace=True,
        )
        start_states = jax.tree.map(lambda x: x[start_idx], particles)
        sample_keys = jax.random.split(sample_key, n_delete)

        def run_chain(key, chain_state, parent_index):
            keys = jax.random.split(key, n_inner_steps)

            def body_fn(current_state, step_key):
                return constrained_step_fn(
                    step_key,
                    current_state,
                    loglikelihood_0,
                    parent_index=parent_index,
                    **step_parameters,
                )

            return jax.lax.scan(body_fn, chain_state, keys)

        return jax.vmap(run_chain)(sample_keys, start_states, start_idx)

    return update_function


def build_from_mcmc_kernel_with_parent_index(
    constrained_step_fn: Callable,
    n_inner_steps: int,
    update_inner_kernel_params_fn: Callable,
    n_delete: int,
) -> Callable:
    """Build an adaptive D1 NS kernel with lane-local original parent indices."""
    inner_kernel = update_with_mcmc_take_last_parent_index(
        constrained_step_fn,
        n_inner_steps=n_inner_steps,
        n_delete=n_delete,
    )
    delete_fn = partial(default_delete_fn, num_delete=n_delete)
    return build_adaptive_kernel(
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )


def update_with_mcmc_take_last_replicated(
    constrained_step_fn: Callable,
    n_inner_steps: int,
    n_delete: int,
    mesh: Mesh,
    fold_inner_steps: bool = False,
    pass_parent_index: bool = False,
) -> Callable:
    """Run sharded constrained chains and gather only packed endpoints."""

    def update_function(rng_key, state, loglikelihood_0, **step_parameters):
        particles = state.particles
        position, _, loglikelihood, _ = _particle_arrays(particles)
        dtypes = _particle_dtypes(particles)
        survivor_weights = (loglikelihood > loglikelihood_0).astype(jnp.float32)
        survivor_weights = jnp.where(
            survivor_weights.sum() > 0,
            survivor_weights,
            jnp.ones_like(survivor_weights),
        )

        choice_key, sample_key = jax.random.split(rng_key)
        start_idx = jax.random.choice(
            choice_key,
            loglikelihood.shape[0],
            shape=(n_delete,),
            p=survivor_weights / survivor_weights.sum(),
            replace=True,
        )
        start_states = jax.tree.map(lambda x: x[start_idx], particles)
        sample_keys = jax.random.split(sample_key, n_delete)

        state_specs = jax.tree.map(lambda _: P(_REPLACEMENT_AXIS), start_states)
        parameter_specs = jax.tree.map(lambda _: P(), step_parameters)
        n_dims = position.shape[-1]

        if pass_parent_index:

            def run_local_chains_with_parent(
                keys, states, parent_indices, threshold, parameters
            ):
                def run_chain(key, chain_state, parent_index):
                    keys = jax.random.split(key, n_inner_steps)
                    if fold_inner_steps:
                        return constrained_step_fn(
                            keys,
                            chain_state,
                            threshold,
                            parent_index=parent_index,
                            **parameters,
                        )

                    def body_fn(current_state, step_key):
                        return constrained_step_fn(
                            step_key,
                            current_state,
                            threshold,
                            parent_index=parent_index,
                            **parameters,
                        )

                    return jax.lax.scan(body_fn, chain_state, keys)

                endpoints, update_info = jax.vmap(run_chain)(
                    keys, states, parent_indices
                )
                packed_endpoints = _pack_particles(endpoints)
                packed_endpoints = jax.lax.all_gather(
                    packed_endpoints, _REPLACEMENT_AXIS, tiled=True
                )
                return packed_endpoints, update_info

            packed_endpoints, update_info = jax.shard_map(
                run_local_chains_with_parent,
                mesh=mesh,
                in_specs=(
                    P(_REPLACEMENT_AXIS),
                    state_specs,
                    P(_REPLACEMENT_AXIS),
                    P(),
                    parameter_specs,
                ),
                out_specs=(P(), P(_REPLACEMENT_AXIS)),
                check_vma=False,
            )(
                sample_keys,
                start_states,
                start_idx,
                loglikelihood_0,
                step_parameters,
            )
            return _unpack_particles(packed_endpoints, n_dims, dtypes), update_info

        def run_local_chains(keys, states, threshold, parameters):
            def run_chain(key, chain_state):
                keys = jax.random.split(key, n_inner_steps)
                if fold_inner_steps:
                    return constrained_step_fn(
                        keys,
                        chain_state,
                        threshold,
                        **parameters,
                    )

                def body_fn(current_state, step_key):
                    return constrained_step_fn(
                        step_key,
                        current_state,
                        threshold,
                        **parameters,
                    )

                return jax.lax.scan(body_fn, chain_state, keys)

            endpoints, update_info = jax.vmap(run_chain)(keys, states)
            packed_endpoints = _pack_particles(endpoints)
            packed_endpoints = jax.lax.all_gather(
                packed_endpoints, _REPLACEMENT_AXIS, tiled=True
            )
            return packed_endpoints, update_info

        packed_endpoints, update_info = jax.shard_map(
            run_local_chains,
            mesh=mesh,
            in_specs=(P(_REPLACEMENT_AXIS), state_specs, P(), parameter_specs),
            out_specs=(P(), P(_REPLACEMENT_AXIS)),
            check_vma=False,
        )(sample_keys, start_states, loglikelihood_0, step_parameters)
        return _unpack_particles(packed_endpoints, n_dims, dtypes), update_info

    return update_function


def build_replicated_from_mcmc_kernel(
    constrained_step_fn: Callable,
    n_inner_steps: int,
    update_inner_kernel_params_fn: Callable,
    n_delete: int,
    mesh: Mesh,
    fold_inner_steps: bool = False,
    pass_parent_index: bool = False,
) -> Callable:
    """Build Jim's fused adaptive NS step for a replicated live population.

    Reimplementing the small BlackJAX orchestration seam is intentional.  Using
    BlackJAX's generic delete/scatter wrapper on a sharded population causes
    compiler-inserted gathers, reductions, and permutes around global indexing.
    Here every replica plans and commits the same batch locally, while only the
    expensive replacement chains are distributed.
    """
    inner_kernel = update_with_mcmc_take_last_replicated(
        constrained_step_fn,
        n_inner_steps=n_inner_steps,
        n_delete=n_delete,
        mesh=mesh,
        fold_inner_steps=fold_inner_steps,
        pass_parent_index=pass_parent_index,
    )

    def kernel(rng_key, state: AdaptiveNSState):
        _, dead_idx = jax.lax.top_k(-state.particles.loglikelihood, n_delete)
        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)
        loglikelihood_0 = dead_particles.loglikelihood.max()

        inner_kernel_update_key, inner_key = jax.random.split(rng_key)
        new_particles, update_info = inner_kernel(
            inner_key,
            state,
            loglikelihood_0,
            **state.inner_kernel_params,
        )

        updated_particles = jax.tree.map(
            lambda live, replacements: live.at[dead_idx].set(replacements),
            state.particles,
            new_particles,
        )
        state_with_new_particles = state._replace(particles=updated_particles)

        # Current Jim adaptation callbacks depend only on the updated live set.
        # Keep the complete callback contract while performing all state-derived
        # calculations on replicated inputs.
        callback_info = NSInfo(dead_particles, update_info)
        new_inner_kernel_params = update_inner_kernel_params_fn(
            inner_kernel_update_key,
            state_with_new_particles,
            callback_info,
            state.inner_kernel_params,
        )
        new_integrator = update_integrator(
            state.integrator,
            updated_particles,
            dead_particles,
        )

        new_state = AdaptiveNSState(
            particles=updated_particles,
            inner_kernel_params=new_inner_kernel_params,
            integrator=new_integrator,
        )
        returned_dead_particles = _shard_by_replacement_id(mesh, dead_particles)
        return new_state, NSInfo(returned_dead_particles, update_info)

    return kernel
