"""Runtime-isolated sampler variants for the post-JIT SwiG ablation.

The benchmark keeps the GW170817 likelihood and every scientific setting fixed.
Only the four sampler implementation axes named in :class:`AblationVariant`
change.  Patches are process-local; the comparison driver launches every cell
in a fresh Python interpreter so variants cannot leak into one another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Topology = Literal["legacy-sharded-live", "replicated-live"]
Interval = Literal["stock-stepping-out", "cached-stepping-out"]
Scheduler = Literal["lockstep", "fsm"]
DirectionParameter = Literal["covariance", "cholesky-factor"]


@dataclass(frozen=True)
class AblationVariant:
    """One independently selectable sampler implementation cell."""

    name: str
    topology: Topology
    interval: Interval
    scheduler: Scheduler
    direction_parameter: DirectionParameter

    def report(self) -> dict[str, str]:
        return asdict(self)


VARIANTS = (
    AblationVariant(
        "legacy-stock-lockstep-cov",
        "legacy-sharded-live",
        "stock-stepping-out",
        "lockstep",
        "covariance",
    ),
    AblationVariant(
        "legacy-cached-lockstep-cov",
        "legacy-sharded-live",
        "cached-stepping-out",
        "lockstep",
        "covariance",
    ),
    AblationVariant(
        "replicated-stock-lockstep-cov",
        "replicated-live",
        "stock-stepping-out",
        "lockstep",
        "covariance",
    ),
    AblationVariant(
        "replicated-cached-lockstep-cov",
        "replicated-live",
        "cached-stepping-out",
        "lockstep",
        "covariance",
    ),
    AblationVariant(
        "replicated-cached-fsm-cov",
        "replicated-live",
        "cached-stepping-out",
        "fsm",
        "covariance",
    ),
    AblationVariant(
        "replicated-cached-fsm-factor",
        "replicated-live",
        "cached-stepping-out",
        "fsm",
        "cholesky-factor",
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
VARIANT_NAMES = tuple(VARIANT_BY_NAME)


def variant(name: str) -> AblationVariant:
    try:
        return VARIANT_BY_NAME[name]
    except KeyError as error:
        raise ValueError(
            f"unknown sampler ablation variant {name!r}; expected one of "
            f"{VARIANT_NAMES}"
        ) from error


def _install_stock_interval() -> None:
    """Use BlackJAX's uncached reference stepping-out implementation."""

    from blackjax.mcmc.slice import stepping_out

    from jimgw.samplers.blackjax import swig

    # The lockstep builder resolves this module global when it is constructed.
    swig.stepping_out_cached = stepping_out


def _install_fsm_covariance_parameters() -> None:
    """Keep the FSM scheduler while supplying covariance matrices."""

    from jimgw.samplers.blackjax import swig

    swig.BlackJAXSwiGSampler._fsm_update_inner_kernel_params_fn = property(
        lambda self: self._update_block_covariances
    )


def _install_legacy_sharded_live_topology() -> None:
    """Restore the pre-replication sharded-live orchestration seam.

    This is the topology from Jim revision 86335bdb, adapted only to the
    current mesh-axis name and builder signature.  The constrained kernel and
    likelihood remain those in the current working tree.
    """

    from collections.abc import Callable

    import jax
    import jax.numpy as jnp
    from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jimgw.samplers.blackjax import nss, sharding, swig

    axis = sharding._REPLACEMENT_AXIS

    def place_sharded_state(state, mesh):
        live = NamedSharding(mesh, P(axis))
        replicated = NamedSharding(mesh, P())
        return state._replace(
            particles=jax.tree.map(
                lambda value: jax.device_put(value, live), state.particles
            ),
            inner_kernel_params=jax.tree.map(
                lambda value: jax.device_put(value, replicated),
                state.inner_kernel_params,
            ),
            integrator=jax.tree.map(
                lambda value: jax.device_put(value, replicated), state.integrator
            ),
        )

    def keep_initial_particles_sharded(particles, mesh):
        del mesh
        return particles

    def all_gather(mesh, value, in_spec):
        def gather(local_value):
            return jax.lax.all_gather(local_value, axis, tiled=True)

        return jax.shard_map(
            gather,
            mesh=mesh,
            in_specs=in_spec,
            out_specs=P(),
            check_vma=False,
        )(value)

    def delete_fn_sharded(mesh, n_delete: int) -> Callable:
        def delete_fn(state):
            global_log_likelihood = all_gather(
                mesh, state.particles.loglikelihood, P(axis)
            )
            _, dead_idx = jax.lax.top_k(-global_log_likelihood, n_delete)
            return dead_idx, dead_idx

        return delete_fn

    def update_with_mcmc_take_last_sharded(
        constrained_step_fn: Callable,
        n_inner_steps: int,
        n_delete: int,
        mesh,
        *,
        fold_inner_steps: bool = False,
    ) -> Callable:
        live = NamedSharding(mesh, P(axis))

        def update_function(rng_key, state, loglikelihood_0, **step_parameters):
            particles = state.particles
            particle_specs = jax.tree.map(
                lambda value: P(axis) if value.ndim > 0 else P(), particles
            )
            global_particles = jax.shard_map(
                lambda values: jax.tree.map(
                    lambda value: jax.lax.all_gather(value, axis, tiled=True), values
                ),
                mesh=mesh,
                in_specs=(particle_specs,),
                out_specs=jax.tree.map(lambda _: P(), particles),
                check_vma=False,
            )(particles)
            global_log_likelihood = global_particles.loglikelihood
            survivor_weights = (global_log_likelihood > loglikelihood_0).astype(
                jnp.float32
            )
            survivor_weights = jnp.where(
                survivor_weights.sum() > 0,
                survivor_weights,
                jnp.ones_like(survivor_weights),
            )

            choice_key, sample_key = jax.random.split(rng_key)
            start_idx = jax.random.choice(
                choice_key,
                global_log_likelihood.shape[0],
                shape=(n_delete,),
                p=survivor_weights / survivor_weights.sum(),
                replace=True,
            )
            start_states = jax.tree.map(
                lambda value: jax.device_put(value[start_idx], live), global_particles
            )
            sample_keys = jax.device_put(jax.random.split(sample_key, n_delete), live)
            state_specs = jax.tree.map(
                lambda value: P(axis) if value.ndim > 0 else P(), start_states
            )
            parameter_specs = jax.tree.map(lambda _: P(), step_parameters)

            def run_local_chains(keys, states, threshold, parameters):
                def run_chain(key, chain_state):
                    keys = jax.random.split(key, n_inner_steps)
                    if fold_inner_steps:
                        return constrained_step_fn(
                            keys, chain_state, threshold, **parameters
                        )

                    def body_fn(current_state, step_key):
                        return constrained_step_fn(
                            step_key, current_state, threshold, **parameters
                        )

                    return jax.lax.scan(body_fn, chain_state, keys)

                return jax.vmap(run_chain)(keys, states)

            return jax.shard_map(
                run_local_chains,
                mesh=mesh,
                in_specs=(P(axis), state_specs, P(), parameter_specs),
                out_specs=(state_specs, P(axis)),
                check_vma=False,
            )(sample_keys, start_states, loglikelihood_0, step_parameters)

        return update_function

    def build_sharded_from_mcmc_kernel(
        constrained_step_fn: Callable,
        n_inner_steps: int,
        update_inner_kernel_params_fn: Callable,
        n_delete: int,
        mesh,
        fold_inner_steps: bool = False,
    ) -> Callable:
        inner_kernel = update_with_mcmc_take_last_sharded(
            constrained_step_fn,
            n_inner_steps=n_inner_steps,
            n_delete=n_delete,
            mesh=mesh,
            fold_inner_steps=fold_inner_steps,
        )
        return build_adaptive_kernel(
            delete_fn_sharded(mesh, n_delete),
            inner_kernel,
            update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        )

    # nss.py imported these functions directly; swig.py imported the builder.
    nss.replicate_initial_particles = keep_initial_particles_sharded
    nss.place_replicated_state = place_sharded_state
    swig.build_replicated_from_mcmc_kernel = build_sharded_from_mcmc_kernel


def install_sampler_ablation(name: str) -> dict[str, str]:
    """Install one variant and return its serialisable design metadata."""

    selected = variant(name)
    if (
        selected.scheduler == "lockstep"
        and selected.direction_parameter != "covariance"
    ):
        raise ValueError("lockstep variants must use covariance parameters")
    if selected.interval == "stock-stepping-out":
        _install_stock_interval()
    if selected.topology == "legacy-sharded-live":
        _install_legacy_sharded_live_topology()
    if selected.scheduler == "fsm" and selected.direction_parameter == "covariance":
        _install_fsm_covariance_parameters()
    return selected.report()
