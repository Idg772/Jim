"""Cache-aware Nested Slice within Gibbs (SwiG) sampling."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
from blackjax import SamplingAlgorithm
from blackjax.mcmc.slice import SliceInfo
from blackjax.mcmc.slice import build_kernel as build_slice_kernel
from blackjax.ns.adaptive import AdaptiveNSState
from blackjax.ns.from_mcmc import build_kernel as build_from_mcmc_kernel
from blackjax.ns.nss import sample_direction_from_covariance
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from jax.sharding import Mesh
from jaxtyping import Array, Float

from jimgw.samplers.blackjax._fsm import (
    SegmentSchedule,
    run_segment,
    slice_randoms_from_keys,
)
from jimgw.samplers.blackjax._slice import stepping_out_cached
from jimgw.samplers.blackjax.nss import (
    BlackJAXNSSSampler,
    _sample_direction_from_covariance_factor,
)
from jimgw.samplers.blackjax.sharding import build_replicated_from_mcmc_kernel
from jimgw.samplers.config import BlackJAXNSSConfig, BlackJAXSwiGConfig
from jimgw.samplers.periodic import _build_masks_arrays
from jimgw.typing import FloatScalar


class CachedSliceState(NamedTuple):
    """Ephemeral slice state; caches are never stored on the live particles."""

    position: Float[Array, " n_dims"]
    logdensity: FloatScalar
    loglikelihood: FloatScalar
    loglikelihood_birth: FloatScalar
    cache: object


def _resolve_block_direction_parameters(
    block_covariances,
    block_covariance_factors,
):
    """Select the legacy covariance or optimized factor direction path."""
    if block_covariance_factors is None:
        if block_covariances is None:
            raise ValueError(
                "Specify either block_covariance_factors or block_covariances."
            )
        return block_covariances, sample_direction_from_covariance
    if block_covariances is not None:
        raise ValueError(
            "Specify only one of block_covariance_factors and block_covariances."
        )
    return block_covariance_factors, _sample_direction_from_covariance_factor


def _build_swig_constrained_step_lockstep(
    *,
    log_prior_fn: Callable,
    build_cache: Callable,
    log_likelihood_from_cache_fn: Callable,
    rebuild_required_by_block: dict[tuple[int, ...], bool],
    num_gibbs_sweeps: int,
    num_inner_steps_per_dim: int,
    max_steps: int,
    max_shrinkage: int,
    periodic: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
) -> Callable:
    """Reference scan implementation for FSM pathwise-equivalence tests."""
    slice_kernel = build_slice_kernel(
        interval=stepping_out_cached,
        max_expansions=max_steps,
        max_shrinkage=max_shrinkage,
    )
    periodic_mask, periodic_lower, periodic_period = _build_masks_arrays(
        periodic, n_dims
    )

    def wrap_periodic_position(position):
        return jnp.where(
            periodic_mask,
            periodic_lower + jnp.mod(position - periodic_lower, periodic_period),
            position,
        )

    def constrained_step(
        rng_key,
        state,
        loglikelihood_0,
        block_covariances=None,
        *,
        block_covariance_factors=None,
    ):
        block_direction_parameters, sample_block_direction = (
            _resolve_block_direction_parameters(
                block_covariances,
                block_covariance_factors,
            )
        )
        cache = build_cache(state.position)
        cached_state = CachedSliceState(
            position=state.position,
            logdensity=state.logdensity,
            loglikelihood=state.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
            cache=cache,
        )
        accepted = jnp.asarray(True)
        num_expansions = jnp.asarray(0)
        num_shrink = jnp.asarray(0)

        for _ in range(num_gibbs_sweeps):
            for (parameter_indices, requires_rebuild), direction_parameter in zip(
                rebuild_required_by_block.items(),
                block_direction_parameters,
                strict=True,
            ):
                parameter_index_array = jnp.asarray(parameter_indices)
                n_steps = num_inner_steps_per_dim * len(parameter_indices)

                def one_slice(
                    carry,
                    key,
                    parameter_index_array=parameter_index_array,
                    direction_parameter=direction_parameter,
                    requires_rebuild=requires_rebuild,
                ):
                    current, all_accepted, expansions, shrink = carry

                    def proposal_generator(direction_key, position, logdensity_fn):
                        del logdensity_fn
                        block_position = position[parameter_index_array]
                        block_direction = sample_block_direction(
                            direction_key,
                            block_position,
                            direction_parameter,
                        )
                        direction = (
                            jnp.zeros_like(position)
                            .at[parameter_index_array]
                            .set(block_direction)
                        )

                        def slice_fn(t):
                            proposed = wrap_periodic_position(position + t * direction)
                            logprior = log_prior_fn(proposed)
                            proposed_cache = (
                                build_cache(proposed)
                                if requires_rebuild
                                else current.cache
                            )
                            loglikelihood = log_likelihood_from_cache_fn(
                                proposed, proposed_cache
                            )
                            proposed_state = CachedSliceState(
                                position=proposed,
                                logdensity=logprior,
                                loglikelihood=loglikelihood,
                                loglikelihood_birth=jnp.asarray(loglikelihood_0),
                                cache=proposed_cache,
                            )
                            return proposed_state, loglikelihood > loglikelihood_0

                        return slice_fn

                    new_state, info = slice_kernel(
                        key, current, None, proposal_generator
                    )
                    return (
                        new_state,
                        all_accepted & info.is_accepted,
                        expansions + info.num_expansions,
                        shrink + info.num_shrink,
                    ), None

                keys = jax.random.split(rng_key, n_steps + 1)
                rng_key = keys[0]
                (cached_state, accepted, num_expansions, num_shrink), _ = jax.lax.scan(
                    one_slice,
                    (cached_state, accepted, num_expansions, num_shrink),
                    keys[1:],
                )

        final_state = state._replace(
            position=cached_state.position,
            logdensity=cached_state.logdensity,
            loglikelihood=cached_state.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
        )
        info = SliceInfo(
            is_accepted=accepted,
            num_expansions=num_expansions,
            num_shrink=num_shrink,
            bracket_left=jnp.zeros(n_dims),
            bracket_right=jnp.zeros(n_dims),
        )
        return final_state, info

    return constrained_step


def _build_swig_constrained_step(
    *,
    log_prior_fn: Callable,
    build_cache: Callable,
    log_likelihood_from_cache_fn: Callable,
    rebuild_required_by_block: dict[tuple[int, ...], bool],
    num_gibbs_sweeps: int,
    num_inner_steps_per_dim: int,
    max_steps: int,
    max_shrinkage: int,
    periodic: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
    per_slice_info: bool = False,
) -> Callable:
    """Run a SwiG transition with one FSM loop per static cache segment."""
    periodic_mask, periodic_lower, periodic_period = _build_masks_arrays(
        periodic, n_dims
    )

    def wrap_periodic_position(position):
        return jnp.where(
            periodic_mask,
            periodic_lower + jnp.mod(position - periodic_lower, periodic_period),
            position,
        )

    def make_eval(requires_rebuild):
        if requires_rebuild:

            def eval_candidate(position, cache):
                del cache
                new_cache = build_cache(position)
                return (
                    log_prior_fn(position),
                    log_likelihood_from_cache_fn(position, new_cache),
                    new_cache,
                )

        else:

            def eval_candidate(position, cache):
                return (
                    log_prior_fn(position),
                    log_likelihood_from_cache_fn(position, cache),
                    cache,
                )

        return eval_candidate

    def constrained_step(
        rng_key,
        state,
        loglikelihood_0,
        block_covariances=None,
        *,
        block_covariance_factors=None,
    ):
        block_direction_parameters, sample_block_direction = (
            _resolve_block_direction_parameters(
                block_covariances,
                block_covariance_factors,
            )
        )
        cache = build_cache(state.position)
        cached_state = CachedSliceState(
            position=state.position,
            logdensity=state.logdensity,
            loglikelihood=state.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
            cache=cache,
        )

        slice_records = []
        key = rng_key
        for _ in range(num_gibbs_sweeps):
            for (parameter_indices, requires_rebuild), direction_parameter in zip(
                rebuild_required_by_block.items(),
                block_direction_parameters,
                strict=True,
            ):
                parameter_index_array = jnp.asarray(parameter_indices)
                n_steps = num_inner_steps_per_dim * len(parameter_indices)
                keys = jax.random.split(key, n_steps + 1)
                key = keys[0]
                (
                    prop_keys,
                    level_u,
                    bracket_u,
                    bracket_v,
                    shrink_key_data,
                ) = slice_randoms_from_keys(keys[1:])
                # Do not vmap these floating-point draws: batched direction
                # normalization changes x64 rounding relative to the reference
                # scan. Static unrolling preserves the bitwise path and adds no
                # preprocessing loop ahead of the scheduler segments.
                block_directions = jnp.stack(
                    [
                        cast(
                            Array,
                            sample_block_direction(
                                prop_keys[i],
                                jnp.zeros_like(state.position[parameter_index_array]),
                                direction_parameter,
                            ),
                        )
                        for i in range(n_steps)
                    ]
                )
                directions = (
                    jnp.zeros((n_steps, n_dims), dtype=state.position.dtype)
                    .at[:, parameter_index_array]
                    .set(block_directions)
                )
                for slice_idx in range(n_steps):
                    slice_records.append(
                        (
                            requires_rebuild,
                            directions[slice_idx],
                            level_u[slice_idx],
                            bracket_u[slice_idx],
                            bracket_v[slice_idx],
                            shrink_key_data[slice_idx],
                        )
                    )

        segments = []
        run_start = 0
        for i in range(1, len(slice_records) + 1):
            if (
                i == len(slice_records)
                or slice_records[i][0] != slice_records[run_start][0]
            ):
                chunk = slice_records[run_start:i]
                segments.append(
                    (
                        chunk[0][0],
                        SegmentSchedule(
                            directions=jnp.stack([record[1] for record in chunk]),
                            level_u=jnp.stack([record[2] for record in chunk]),
                            bracket_u=jnp.stack([record[3] for record in chunk]),
                            bracket_v=jnp.stack([record[4] for record in chunk]),
                            shrink_key_data=jnp.stack([record[5] for record in chunk]),
                        ),
                    )
                )
                run_start = i

        segment_infos = []
        for requires_rebuild, schedule in segments:
            cached_state, segment_info = run_segment(
                schedule,
                cached_state,
                loglikelihood_0,
                eval_candidate=make_eval(requires_rebuild),
                wrap_position=wrap_periodic_position,
                max_expansions=max_steps,
                max_shrinkage=max_shrinkage,
            )
            segment_infos.append(segment_info)

        accepted = jnp.all(
            jnp.concatenate([info.is_accepted for info in segment_infos])
        )
        num_expansions = jnp.concatenate(
            [info.num_expansions for info in segment_infos]
        )
        num_shrink = jnp.concatenate([info.num_shrink for info in segment_infos])
        if not per_slice_info:
            num_expansions = num_expansions.sum()
            num_shrink = num_shrink.sum()

        final_state = state._replace(
            position=cached_state.position,
            logdensity=cached_state.logdensity,
            loglikelihood=cached_state.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
        )
        info = SliceInfo(
            is_accepted=accepted,
            num_expansions=num_expansions,
            num_shrink=num_shrink,
            bracket_left=jnp.zeros(n_dims),
            bracket_right=jnp.zeros(n_dims),
        )
        return final_state, info

    return constrained_step


class BlackJAXSwiGSampler(BlackJAXNSSSampler):
    """Nested Slice within Gibbs using cache-aware slices over named blocks."""

    _swig_config: BlackJAXSwiGConfig

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: BlackJAXSwiGConfig,
        periodic: Optional[dict[int, tuple[float, float]]] = None,
        rebuild_required_by_block: Optional[dict[tuple[int, ...], bool]] = None,
        build_cache: Optional[Callable] = None,
        log_likelihood_from_cache_fn: Optional[Callable] = None,
    ) -> None:
        if (
            rebuild_required_by_block is None
            or build_cache is None
            or log_likelihood_from_cache_fn is None
        ):
            raise ValueError("BlackJAXSwiGSampler requires cache callbacks.")
        if periodic is not None and not isinstance(periodic, dict):
            raise TypeError("Cache-aware sampling requires dict-form periodic bounds.")

        nss_config = BlackJAXNSSConfig(
            n_live=config.n_live,
            n_delete_frac=config.n_delete_frac,
            num_inner_steps_per_dim=config.num_inner_steps_per_dim,
            termination_dlogz=config.termination_dlogz,
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_interval=config.checkpoint_interval,
            n_devices=config.n_devices,
        )
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=nss_config,
            periodic=periodic,
        )
        self._swig_config = config
        self._rebuild_required_by_block = rebuild_required_by_block
        self._build_cache = build_cache
        self._log_likelihood_from_cache_fn = log_likelihood_from_cache_fn
        self._periodic = periodic

        def block_covariances(state):
            covariance = jnp.atleast_2d(
                particles_covariance_matrix(state.particles.position)
            )
            return tuple(
                covariance[
                    jnp.ix_(
                        jnp.asarray(parameter_indices),
                        jnp.asarray(parameter_indices),
                    )
                ]
                for parameter_indices in self._rebuild_required_by_block
            )

        def update_block_covariances(rng_key, state, info, params=None):
            del rng_key, info, params
            return {"block_covariances": block_covariances(state)}

        def update_block_covariance_factors(rng_key, state, info, params=None):
            del rng_key, info, params
            covariance_factors = tuple(
                jnp.linalg.cholesky(covariance)
                for covariance in block_covariances(state)
            )
            return {"block_covariance_factors": covariance_factors}

        self._update_block_covariances = update_block_covariances
        self._update_block_covariance_factors = update_block_covariance_factors

    @property
    def sampler_name(self) -> str:
        if self._swig_config.scheduler == "pre-fsm-lockstep":
            return "BlackJAX SwiG pre-FSM lockstep"
        return "BlackJAX SwiG"

    @property
    def _update_inner_kernel_params_fn(self) -> Callable:
        return self._update_block_covariances

    @property
    def _fsm_update_inner_kernel_params_fn(self) -> Callable:
        if self._swig_config.scheduler == "pre-fsm-lockstep":
            return self._update_block_covariances
        return self._update_block_covariance_factors

    def _normalise_inner_kernel_params_for_mesh(
        self, state: AdaptiveNSState, mesh: Optional[Mesh]
    ) -> AdaptiveNSState:
        """Convert SwiG checkpoint block parameters for the selected path."""
        try:
            params = state.inner_kernel_params
        except AttributeError as error:
            raise ValueError(
                "checkpoint state has no inner-kernel parameters"
            ) from error
        keys = set(params)
        use_covariance_factors = (
            mesh is not None and self._swig_config.scheduler == "fsm"
        )
        if not use_covariance_factors:
            if keys == {"block_covariances"}:
                return state
            if keys == {"block_covariance_factors"}:
                return state._replace(
                    inner_kernel_params=self._update_inner_kernel_params_fn(
                        None, state, None
                    )
                )
        else:
            if keys == {"block_covariance_factors"}:
                return state
            if keys == {"block_covariances"}:
                return state._replace(
                    inner_kernel_params=self._fsm_update_inner_kernel_params_fn(
                        None, state, None
                    )
                )
        expected = (
            "block_covariance_factors"
            if use_covariance_factors
            else "block_covariances"
        )
        raise ValueError(
            "checkpoint has incompatible SwiG inner-kernel parameters: "
            f"expected {expected!r}, found {sorted(keys)!r}"
        )

    def _build_nested_sampler(
        self, n_delete: int, mesh: Optional[Mesh] = None
    ) -> SamplingAlgorithm:
        constrained_step_builder = (
            _build_swig_constrained_step_lockstep
            if self._swig_config.scheduler == "pre-fsm-lockstep"
            else _build_swig_constrained_step
        )
        constrained_step = constrained_step_builder(
            log_prior_fn=self._log_prior_fn,
            build_cache=self._build_cache,
            log_likelihood_from_cache_fn=self._log_likelihood_from_cache_fn,
            rebuild_required_by_block=self._rebuild_required_by_block,
            num_gibbs_sweeps=self._swig_config.num_gibbs_sweeps,
            num_inner_steps_per_dim=self._swig_config.num_inner_steps_per_dim,
            max_steps=self._swig_config.max_steps,
            max_shrinkage=self._swig_config.max_shrinkage,
            periodic=self._periodic,
            n_dims=self.n_dims,
        )
        if mesh is None:
            kernel = build_from_mcmc_kernel(
                constrained_step,
                num_inner_steps=1,
                update_inner_kernel_params_fn=self._update_block_covariances,
                num_delete=n_delete,
            )
        else:
            kernel = build_replicated_from_mcmc_kernel(
                constrained_step,
                n_inner_steps=1,
                update_inner_kernel_params_fn=self._fsm_update_inner_kernel_params_fn,
                n_delete=n_delete,
                mesh=mesh,
            )

        # `nested_sampler.init` is never called (state init happens in
        # `_batched_nss_init`); BlackJAX still requires SamplingAlgorithm.init
        # to type as returning a State.
        return SamplingAlgorithm(
            lambda position, rng_key=None: position,  # type: ignore[return-value]
            kernel,
        )
