"""Cache-aware Nested Slice within Gibbs (SwiG) sampling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
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
    HybridSegmentSchedule,
    SegmentSchedule,
    run_hybrid_segment,
    run_segment,
    slice_randoms_from_keys,
)
from jimgw.samplers.blackjax._slice import shrink_only_bracket, stepping_out_cached
from jimgw.samplers.blackjax.nss import (
    BlackJAXNSSSampler,
    _sample_direction_from_covariance_factor,
)
from jimgw.samplers.blackjax.sharding import (
    build_from_mcmc_kernel_with_parent_index,
    build_replicated_from_mcmc_kernel,
)
from jimgw.samplers.config import (
    SUPPORTED_COMPLEMENTARY_DE_ATTEMPTS,
    BlackJAXNSSConfig,
    BlackJAXSwiGConfig,
)
from jimgw.samplers.periodic import _build_masks_arrays
from jimgw.typing import FloatScalar


class CachedSliceState(NamedTuple):
    """Ephemeral slice state; caches are never stored on the live particles."""

    position: Float[Array, " n_dims"]
    logdensity: FloatScalar
    loglikelihood: FloatScalar
    loglikelihood_birth: FloatScalar
    cache: object


class SwiGInfo(NamedTuple):
    """Slice diagnostics extended with fixed-cost DE-jump telemetry."""

    is_accepted: object
    num_expansions: object
    num_shrink: object
    bracket_left: object
    bracket_right: object
    num_de_jump_attempts: object
    num_de_jump_acceptances: object


class TargetedSwiGInfo(NamedTuple):
    """DE diagnostics with per-target-group fixed-cost telemetry."""

    is_accepted: object
    num_expansions: object
    num_shrink: object
    bracket_left: object
    bracket_right: object
    num_de_jump_attempts: object
    num_de_jump_acceptances: object
    num_targeted_de_jump_attempts: object
    num_targeted_de_jump_acceptances: object


class PeriodicIndependenceSwiGInfo(NamedTuple):
    """Slice diagnostics with fixed-cost periodic independence telemetry."""

    is_accepted: object
    num_expansions: object
    num_shrink: object
    bracket_left: object
    bracket_right: object
    num_periodic_uniform_independence_attempts: object
    num_periodic_uniform_independence_acceptances: object
    num_periodic_uniform_independence_attempts_by_block: object
    num_periodic_uniform_independence_acceptances_by_block: object


class PeriodicIndependenceComplementaryDESwiGInfo(NamedTuple):
    """H4 diagnostics extended with fixed-complement DE-MH telemetry."""

    is_accepted: object
    num_expansions: object
    num_shrink: object
    bracket_left: object
    bracket_right: object
    num_periodic_uniform_independence_attempts: object
    num_periodic_uniform_independence_acceptances: object
    num_periodic_uniform_independence_attempts_by_block: object
    num_periodic_uniform_independence_acceptances_by_block: object
    num_complementary_de_attempts: object
    num_complementary_de_acceptances: object
    num_complementary_de_attempts_by_block: object
    num_complementary_de_acceptances_by_block: object
    complementary_de_acceptances_by_attempt: object
    complementary_de_donor_indices_by_attempt: object
    complementary_de_donor_policy_violations_by_attempt: object
    complementary_de_complement_size: object
    complementary_de_parent_index: object
    complementary_de_position_before_by_attempt: object
    complementary_de_proposal_position_by_attempt: object


class _ComplementaryDEProposalSchedule(NamedTuple):
    """Frozen donor complement and per-attempt proposal randomness."""

    operation_key_data: Array
    displacement: Array
    donor_indices: Array
    donor_policy_violations: Array
    complement_size: Array


class _ComplementaryDEAttemptInfo(NamedTuple):
    """Per-attempt results for one frozen complementary-DE group."""

    acceptances: Array
    donor_indices: Array
    donor_policy_violations: Array
    complement_size: Array
    position_before: Array
    proposal_position: Array


ResolvedDEJumpBlock = tuple[tuple[int, ...], bool, int]

_COMPLEMENTARY_DE_KEY_DOMAIN = 0xCDE40000


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


def _resolve_num_slice_steps_by_block(
    rebuild_required_by_block: dict[tuple[int, ...], bool],
    num_inner_steps_per_dim: int,
    num_slice_steps_by_block: Optional[Sequence[int]],
) -> tuple[int, ...]:
    """Resolve the static number of slice updates assigned to each block."""
    if num_slice_steps_by_block is None:
        return tuple(
            num_inner_steps_per_dim * len(parameter_indices)
            for parameter_indices in rebuild_required_by_block
        )
    counts = tuple(num_slice_steps_by_block)
    if len(counts) != len(rebuild_required_by_block):
        raise ValueError(
            "num_slice_steps_by_block must contain exactly one count per block"
        )
    if any(count < 1 for count in counts):
        raise ValueError("num_slice_steps_by_block counts must all be positive")
    return counts


def _slice_to_block_map(
    block_step_counts: tuple[int, ...], num_gibbs_sweeps: int
) -> tuple[int, ...]:
    """Block index of every scheduled slice update, in schedule order."""
    per_sweep = [
        block_index
        for block_index, count in enumerate(block_step_counts)
        for _ in range(count)
    ]
    return tuple(per_sweep) * num_gibbs_sweeps


def _updated_block_widths(
    block_widths,
    num_expansions,
    num_shrink,
    *,
    slice_to_block: tuple[int, ...],
    n_blocks: int,
    rate: float,
    target_expansions: float,
    target_shrinks: float,
    shrink_only: bool,
):
    """Multiplicative per-block width update from per-slice counters.

    Counters arrive with the scheduled-slice axis last; every leading axis
    (lanes, deletions) is reduced by a full-axis sum so the result is
    replicated identically on every device under GSPMD.
    """
    n_slices = len(slice_to_block)
    exp_totals = jnp.asarray(num_expansions, float).reshape(-1, n_slices)
    shr_totals = jnp.asarray(num_shrink, float).reshape(-1, n_slices)
    n_visits_per_slice = exp_totals.shape[0]
    map_array = jnp.asarray(slice_to_block)
    slice_counts = jnp.zeros(n_blocks).at[map_array].add(jnp.ones(n_slices))
    exp_by_block = jnp.zeros(n_blocks).at[map_array].add(exp_totals.sum(axis=0))
    shr_by_block = jnp.zeros(n_blocks).at[map_array].add(shr_totals.sum(axis=0))
    visits = slice_counts * n_visits_per_slice
    mean_expansions = exp_by_block / visits
    mean_shrinks = shr_by_block / visits
    shrink_excess = (mean_shrinks - target_shrinks) / max(target_shrinks, 1.0)
    if shrink_only:
        delta = -shrink_excess
    else:
        expansion_excess = (mean_expansions - target_expansions) / max(
            target_expansions, 1.0
        )
        delta = expansion_excess - shrink_excess
    log_widths = jnp.log(jnp.asarray(block_widths, float)) + rate * delta
    log_widths = jnp.clip(log_widths, jnp.log(1e-3), jnp.log(1e3))
    widths = jnp.exp(log_widths)
    return tuple(widths[i] for i in range(n_blocks))


def _resolve_block_kernel_modes(
    *,
    rebuild_required_by_block: dict[tuple[int, ...], bool],
    block_step_counts: tuple[int, ...],
    block_kernel_modes: Optional[Sequence[str]],
    num_gibbs_sweeps: int,
    num_inner_steps_per_dim: int,
    has_explicit_block_budget: bool,
    periodic: Optional[dict[int, tuple[float, float]]],
    direction_mode: str,
    num_de_jumps: int,
    resolved_de_jump_blocks: tuple[ResolvedDEJumpBlock, ...],
) -> tuple[str, ...]:
    """Validate and resolve the per-block kernel selection fail-closed."""
    if block_kernel_modes is None:
        return ("slice",) * len(rebuild_required_by_block)
    modes = tuple(block_kernel_modes)
    if len(modes) != len(rebuild_required_by_block):
        raise ValueError("block_kernel_modes must contain exactly one mode per block")
    unsupported = sorted(set(modes) - {"slice", "periodic-uniform-independence"})
    if unsupported:
        raise ValueError(f"Unsupported block kernel modes: {unsupported}")
    if "periodic-uniform-independence" not in modes:
        return modes
    if num_inner_steps_per_dim != 1:
        raise ValueError(
            "periodic-uniform-independence requires exactly one update per block"
        )
    if has_explicit_block_budget:
        raise ValueError(
            "periodic-uniform-independence cannot be combined with "
            "num_slice_steps_by_block"
        )
    if direction_mode != "covariance" or num_de_jumps or resolved_de_jump_blocks:
        raise ValueError(
            "periodic-uniform-independence cannot be combined with DE or "
            "non-covariance direction modes"
        )
    for (parameter_indices, _), count, mode in zip(
        rebuild_required_by_block.items(), block_step_counts, modes, strict=True
    ):
        if mode != "periodic-uniform-independence":
            continue
        if len(parameter_indices) != 1 or count != 1:
            raise ValueError(
                "periodic-uniform-independence requires a singleton block and "
                "exactly one update"
            )
        parameter_index = parameter_indices[0]
        if periodic is None or parameter_index not in periodic:
            raise ValueError(
                "periodic-uniform-independence requires declared periodic bounds"
            )
        lower, upper = periodic[parameter_index]
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError(
                "periodic-uniform-independence requires finite, increasing "
                "periodic bounds"
            )
    return modes


def _sample_signed_permuted_covariance_basis(rng_key, covariance_factor):
    """Return the factor columns once each, in random signed order.

    The rows are slice directions.  In whitened coordinates they are a signed
    permutation of the coordinate axes with norm two, matching the scale of
    the established covariance-direction draw without introducing a Haar/QR
    rotation.
    """
    permutation_key, sign_key = jax.random.split(rng_key)
    n_dims = covariance_factor.shape[-1]
    permutation = jax.random.permutation(permutation_key, n_dims)
    signs = jnp.where(
        jax.random.bernoulli(sign_key, shape=(n_dims,)),
        jnp.asarray(1.0, dtype=covariance_factor.dtype),
        jnp.asarray(-1.0, dtype=covariance_factor.dtype),
    )
    return (2.0 * covariance_factor[:, permutation] * signs).T


def _sample_de_mix_direction(
    rng_key,
    live_positions,
    live_loglikelihoods,
    parent_position,
    loglikelihood_constraint,
    parameter_index_array,
    de_fraction,
    sample_block_direction,
    direction_parameter,
    block_template,
):
    """Mix DE-over-live-pairs and covariance-chord slice directions.

    A differential-evolution direction is the difference of two distinct live
    points restricted to the block.  When the live set spans several basins,
    cross-basin pairs point through a companion basin's core and the pair
    separation sets the slice bracket scale, so minority basins keep a
    replenishment rate proportional to their live population instead of dying
    by covariance collapse.  Covariance chords keep within-basin exploration
    isotropic in the Mahalanobis metric.
    """
    mode_key, pair_key, chord_key = jax.random.split(rng_key, 3)
    chord_direction = sample_block_direction(
        chord_key, block_template, direction_parameter
    )
    eligible = (live_loglikelihoods > loglikelihood_constraint) & ~jnp.all(
        live_positions == parent_position,
        axis=1,
    )
    use_de = jax.random.bernoulli(mode_key, de_fraction) & (eligible.sum() >= 2)

    def sample_de_direction(_):
        pair = jax.random.choice(
            pair_key,
            live_positions.shape[0],
            shape=(2,),
            replace=False,
            p=eligible.astype(live_positions.dtype),
        )
        return (live_positions[pair[0]] - live_positions[pair[1]])[
            parameter_index_array
        ]

    return jax.lax.cond(
        use_de,
        sample_de_direction,
        lambda _: chord_direction,
        operand=None,
    )


def _prepare_complementary_de_proposals(
    rng_key,
    *,
    live_positions,
    live_loglikelihoods,
    loglikelihood_0,
    parent_index,
    parameter_indices: tuple[int, ...],
    attempts: int,
) -> _ComplementaryDEProposalSchedule:
    """Freeze one parent's strict-survivor donor complement and draw pairs."""
    n_live = live_positions.shape[0]
    live_indices = jnp.arange(n_live, dtype=jnp.int32)
    parent_index = jnp.asarray(parent_index, dtype=jnp.int32)
    parent_valid = (parent_index >= 0) & (parent_index < n_live)
    eligible = (live_loglikelihoods > loglikelihood_0) & (live_indices != parent_index)
    complement_size = eligible.sum(dtype=jnp.int32)
    can_draw = parent_valid & (complement_size >= 2)
    target_indices = jnp.asarray(parameter_indices, dtype=jnp.int32)

    operation_keys = tuple(
        jax.random.fold_in(rng_key, _COMPLEMENTARY_DE_KEY_DOMAIN + attempt_index)
        for attempt_index in range(attempts)
    )
    displacements = []
    donor_indices = []
    violations = []
    for operation_key in operation_keys:
        pair_key, _ = jax.random.split(operation_key)

        def draw_pair(_, pair_key=pair_key):
            weights = eligible.astype(live_positions.dtype)
            return jax.random.choice(
                pair_key,
                n_live,
                shape=(2,),
                replace=False,
                p=weights / weights.sum(),
            ).astype(jnp.int32)

        pair = jax.lax.cond(
            can_draw,
            draw_pair,
            lambda _: jnp.full((2,), -1, dtype=jnp.int32),
            operand=None,
        )
        safe_pair = jnp.clip(pair, 0, n_live - 1)
        valid_pair = (
            can_draw
            & (pair[0] != pair[1])
            & (pair[0] != parent_index)
            & (pair[1] != parent_index)
            & eligible[safe_pair[0]]
            & eligible[safe_pair[1]]
        )
        difference = live_positions[safe_pair[0]] - live_positions[safe_pair[1]]
        displacement = (
            jnp.zeros_like(difference)
            .at[target_indices]
            .set(difference[target_indices])
        )
        displacements.append(jnp.where(valid_pair, displacement, 0.0))
        donor_indices.append(pair)
        violations.append(~valid_pair)

    return _ComplementaryDEProposalSchedule(
        operation_key_data=jnp.stack(
            [jax.random.key_data(key) for key in operation_keys]
        ),
        displacement=jnp.stack(displacements),
        donor_indices=jnp.stack(donor_indices),
        donor_policy_violations=jnp.stack(violations),
        complement_size=complement_size,
    )


def _apply_complementary_de_proposals(
    schedule: _ComplementaryDEProposalSchedule,
    cached_state: CachedSliceState,
    loglikelihood_0,
    *,
    eval_candidate: Callable,
    wrap_position: Callable,
) -> tuple[CachedSliceState, _ComplementaryDEAttemptInfo]:
    """Apply a fixed schedule of gamma-one complementary DE-MH attempts."""
    acceptances = []
    positions_before = []
    proposal_positions = []
    for attempt_index in range(schedule.displacement.shape[0]):
        operation_key = jax.random.wrap_key_data(
            schedule.operation_key_data[attempt_index],
            impl="threefry2x32",
        )
        _, accept_key = jax.random.split(operation_key)
        position_before = cached_state.position
        proposal = wrap_position(position_before + schedule.displacement[attempt_index])
        logdensity, loglikelihood, cache = eval_candidate(proposal, cached_state.cache)
        candidate = cached_state._replace(
            position=proposal,
            logdensity=logdensity,
            loglikelihood=loglikelihood,
            cache=cache,
        )
        accepted = (
            ~schedule.donor_policy_violations[attempt_index]
            & (loglikelihood > loglikelihood_0)
            & (
                jnp.log(jax.random.uniform(accept_key))
                < logdensity - cached_state.logdensity
            )
        )
        cached_state = jax.tree.map(
            lambda new, old, accepted=accepted: jnp.where(accepted, new, old),
            candidate,
            cached_state,
        )
        acceptances.append(accepted)
        positions_before.append(position_before)
        proposal_positions.append(proposal)

    return cached_state, _ComplementaryDEAttemptInfo(
        acceptances=jnp.stack(acceptances),
        donor_indices=schedule.donor_indices,
        donor_policy_violations=schedule.donor_policy_violations,
        complement_size=schedule.complement_size,
        position_before=jnp.stack(positions_before),
        proposal_position=jnp.stack(proposal_positions),
    )


def _apply_de_jumps(
    rng_key,
    cached_state,
    loglikelihood_0,
    live_positions,
    num_de_jumps,
    eval_candidate,
    wrap_position,
    parameter_indices=None,
):
    """Metropolis differential-evolution jumps over live-point pairs.

    A full pair displacement ``x + (x_a - x_b)`` lands in a companion basin's
    core whenever the pair spans basins, with no slice shrinkage to prune the
    disconnected segment; same-basin pairs act as small in-basin perturbations.
    The pair-difference proposal is symmetric, so accepting in-contour moves
    with a prior-ratio Metropolis test targets the constrained prior exactly.
    When ``parameter_indices`` is provided, the symmetric displacement is
    projected onto that fixed coordinate group. Each attempt costs one
    likelihood evaluation.
    """
    num_acceptances = jnp.asarray(0, dtype=jnp.int32)
    for jump_key in jax.random.split(rng_key, num_de_jumps):
        pair_key, accept_key = jax.random.split(jump_key)
        pair = jax.random.choice(
            pair_key, live_positions.shape[0], shape=(2,), replace=False
        )
        if parameter_indices is None:
            # Preserve the established full-space operation order exactly.
            proposal = wrap_position(
                cached_state.position
                + live_positions[pair[0]]
                - live_positions[pair[1]]
            )
        else:
            displacement = live_positions[pair[0]] - live_positions[pair[1]]
            parameter_index_array = jnp.asarray(parameter_indices)
            displacement = (
                jnp.zeros_like(displacement)
                .at[parameter_index_array]
                .set(displacement[parameter_index_array])
            )
            proposal = wrap_position(cached_state.position + displacement)
        logdensity, loglikelihood, cache = eval_candidate(proposal, cached_state.cache)
        candidate = CachedSliceState(
            position=proposal,
            logdensity=logdensity,
            loglikelihood=loglikelihood,
            loglikelihood_birth=cached_state.loglikelihood_birth,
            cache=cache,
        )
        accept = (loglikelihood > loglikelihood_0) & (
            jnp.log(jax.random.uniform(accept_key))
            < logdensity - cached_state.logdensity
        )
        cached_state = jax.tree.map(
            lambda new, old, accept=accept: jnp.where(accept, new, old),
            candidate,
            cached_state,
        )
        num_acceptances = num_acceptances + accept.astype(jnp.int32)
    return cached_state, num_acceptances


def _apply_periodic_uniform_independence(
    rng_key,
    cached_state,
    loglikelihood_0,
    *,
    parameter_index: int,
    lower: float,
    upper: float,
    eval_candidate: Callable,
):
    """Apply one exact uniform-independence MH update on a periodic support."""
    proposal_key, accept_key = jax.random.split(rng_key)
    proposed_value = jax.random.uniform(
        proposal_key,
        minval=jnp.asarray(lower, dtype=cached_state.position.dtype),
        maxval=jnp.asarray(upper, dtype=cached_state.position.dtype),
    )
    proposal = cached_state.position.at[parameter_index].set(proposed_value)
    logdensity, loglikelihood, cache = eval_candidate(proposal, cached_state.cache)
    candidate = CachedSliceState(
        position=proposal,
        logdensity=logdensity,
        loglikelihood=loglikelihood,
        loglikelihood_birth=cached_state.loglikelihood_birth,
        cache=cache,
    )
    accepted = (loglikelihood > loglikelihood_0) & (
        jnp.log(jax.random.uniform(accept_key)) < logdensity - cached_state.logdensity
    )
    return (
        jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old),
            candidate,
            cached_state,
        ),
        accepted,
    )


def _build_swig_constrained_step_lockstep(
    *,
    log_prior_fn: Callable,
    build_cache: Callable,
    log_likelihood_from_cache_fn: Callable,
    rebuild_required_by_block: dict[tuple[int, ...], bool],
    num_gibbs_sweeps: int,
    num_inner_steps_per_dim: int,
    num_slice_steps_by_block: Optional[Sequence[int]] = None,
    max_steps: int,
    max_shrinkage: int,
    periodic: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
    direction_mode: str = "covariance",
    de_fraction: float = 0.5,
    num_de_jumps: int = 0,
    resolved_de_jump_blocks: tuple[ResolvedDEJumpBlock, ...] = (),
    block_kernel_modes: Optional[Sequence[str]] = None,
    resolved_complementary_de_jump_block: Optional[ResolvedDEJumpBlock] = None,
    bracket_mode: str = "stepping-out",
) -> Callable:
    """Reference scan implementation for FSM pathwise-equivalence tests."""
    if bracket_mode not in ("stepping-out", "shrink-only"):
        raise ValueError(f"Unsupported bracket_mode: {bracket_mode!r}")
    block_step_counts = _resolve_num_slice_steps_by_block(
        rebuild_required_by_block,
        num_inner_steps_per_dim,
        num_slice_steps_by_block,
    )
    resolved_block_kernel_modes = _resolve_block_kernel_modes(
        rebuild_required_by_block=rebuild_required_by_block,
        block_step_counts=block_step_counts,
        block_kernel_modes=block_kernel_modes,
        num_gibbs_sweeps=num_gibbs_sweeps,
        num_inner_steps_per_dim=num_inner_steps_per_dim,
        has_explicit_block_budget=num_slice_steps_by_block is not None,
        periodic=periodic,
        direction_mode=direction_mode,
        num_de_jumps=num_de_jumps,
        resolved_de_jump_blocks=resolved_de_jump_blocks,
    )
    if resolved_complementary_de_jump_block is not None:
        complementary_indices, complementary_rebuild, complementary_attempts = (
            resolved_complementary_de_jump_block
        )
        if complementary_indices not in rebuild_required_by_block:
            raise ValueError(
                "complementary DE parameters must match one resolved slice block"
            )
        if (
            not complementary_rebuild
            or not rebuild_required_by_block[complementary_indices]
        ):
            raise ValueError(
                "complementary DE must target a waveform-rebuild slice block"
            )
        if complementary_attempts not in SUPPORTED_COMPLEMENTARY_DE_ATTEMPTS:
            raise ValueError("complementary DE requires exactly four or eight attempts")
    slice_kernel = build_slice_kernel(
        interval=(
            stepping_out_cached
            if bracket_mode == "stepping-out"
            else shrink_only_bracket
        ),
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
        live_positions=None,
        live_loglikelihoods=None,
        parent_index=None,
        block_widths=None,
    ):
        root_key = rng_key
        uses_covariance_factors = block_covariance_factors is not None
        block_direction_parameters, sample_block_direction = (
            _resolve_block_direction_parameters(
                block_covariances,
                block_covariance_factors,
            )
        )
        if block_widths is not None and len(block_widths) != len(
            rebuild_required_by_block
        ):
            raise ValueError("block_widths must contain exactly one factor per block")
        if (
            direction_mode == "de-mix"
            or num_de_jumps > 0
            or resolved_de_jump_blocks
            or resolved_complementary_de_jump_block is not None
        ) and live_positions is None:
            raise ValueError("DE moves require live_positions parameters")
        if (
            direction_mode == "de-mix"
            or resolved_complementary_de_jump_block is not None
        ) and live_loglikelihoods is None:
            raise ValueError("DE survivor selection requires live_loglikelihoods")
        if resolved_complementary_de_jump_block is not None and parent_index is None:
            raise ValueError("complementary DE requires the original parent_index")
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
        num_de_jump_acceptances = jnp.asarray(0, dtype=jnp.int32)
        periodic_independence_acceptances = []
        complementary_schedule = None
        complementary_info = None
        if resolved_complementary_de_jump_block is not None:
            complementary_schedule = _prepare_complementary_de_proposals(
                root_key,
                live_positions=live_positions,
                live_loglikelihoods=live_loglikelihoods,
                loglikelihood_0=loglikelihood_0,
                parent_index=parent_index,
                parameter_indices=resolved_complementary_de_jump_block[0],
                attempts=resolved_complementary_de_jump_block[2],
            )

        for _ in range(num_gibbs_sweeps):
            for block_index, (
                (parameter_indices, requires_rebuild),
                direction_parameter,
                n_steps,
                block_kernel_mode,
            ) in enumerate(
                zip(
                    rebuild_required_by_block.items(),
                    block_direction_parameters,
                    block_step_counts,
                    resolved_block_kernel_modes,
                    strict=True,
                )
            ):
                parameter_index_array = jnp.asarray(parameter_indices)
                use_covariance_basis = (
                    direction_mode == "covariance-basis-8d"
                    and len(parameter_indices) == 8
                )

                if block_kernel_mode == "periodic-uniform-independence":
                    keys = jax.random.split(rng_key, n_steps + 1)
                    rng_key = keys[0]
                    parameter_index = parameter_indices[0]
                    assert periodic is not None
                    lower, upper = periodic[parameter_index]

                    if requires_rebuild:

                        def independence_eval(position, cache):
                            del cache
                            new_cache = build_cache(position)
                            return (
                                log_prior_fn(position),
                                log_likelihood_from_cache_fn(position, new_cache),
                                new_cache,
                            )

                    else:

                        def independence_eval(position, cache):
                            return (
                                log_prior_fn(position),
                                log_likelihood_from_cache_fn(position, cache),
                                cache,
                            )

                    cached_state, independence_accepted = (
                        _apply_periodic_uniform_independence(
                            keys[1],
                            cached_state,
                            loglikelihood_0,
                            parameter_index=parameter_index,
                            lower=lower,
                            upper=upper,
                            eval_candidate=independence_eval,
                        )
                    )
                    periodic_independence_acceptances.append(independence_accepted)
                    continue

                def one_slice(
                    carry,
                    slice_input,
                    parameter_index_array=parameter_index_array,
                    direction_parameter=direction_parameter,
                    requires_rebuild=requires_rebuild,
                    use_covariance_basis=use_covariance_basis,
                    block_index=block_index,
                ):
                    if use_covariance_basis:
                        key, basis_direction = slice_input
                    else:
                        key = slice_input
                    current, all_accepted, expansions, shrink = carry

                    def proposal_generator(direction_key, position, logdensity_fn):
                        del logdensity_fn
                        block_position = position[parameter_index_array]
                        if use_covariance_basis:
                            block_direction = basis_direction
                            if block_widths is not None:
                                block_direction = (
                                    block_direction * block_widths[block_index]
                                )
                        elif direction_mode in (
                            "covariance",
                            "covariance-basis-8d",
                        ):
                            block_direction = sample_block_direction(
                                direction_key,
                                block_position,
                                direction_parameter,
                            )
                            if block_widths is not None:
                                block_direction = (
                                    block_direction * block_widths[block_index]
                                )
                        else:
                            if block_widths is not None:
                                raise ValueError(
                                    "block_widths is not supported with de-mix "
                                    "directions"
                                )
                            block_direction = _sample_de_mix_direction(
                                direction_key,
                                live_positions,
                                live_loglikelihoods,
                                state.position,
                                loglikelihood_0,
                                parameter_index_array,
                                de_fraction,
                                sample_block_direction,
                                direction_parameter,
                                block_position,
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
                scan_inputs = keys[1:]
                if use_covariance_basis:
                    prop_keys, *_ = slice_randoms_from_keys(scan_inputs)
                    covariance_factor = (
                        direction_parameter
                        if uses_covariance_factors
                        else jnp.linalg.cholesky(direction_parameter)
                    )
                    basis_directions = _sample_signed_permuted_covariance_basis(
                        prop_keys[0], covariance_factor
                    )
                    scan_inputs = (scan_inputs, basis_directions)
                (cached_state, accepted, num_expansions, num_shrink), _ = jax.lax.scan(
                    one_slice,
                    (cached_state, accepted, num_expansions, num_shrink),
                    scan_inputs,
                )
                if (
                    resolved_complementary_de_jump_block is not None
                    and parameter_indices == resolved_complementary_de_jump_block[0]
                ):
                    assert complementary_schedule is not None

                    def complementary_eval(position, cache):
                        del cache
                        new_cache = build_cache(position)
                        return (
                            log_prior_fn(position),
                            log_likelihood_from_cache_fn(position, new_cache),
                            new_cache,
                        )

                    cached_state, complementary_info = (
                        _apply_complementary_de_proposals(
                            complementary_schedule,
                            cached_state,
                            loglikelihood_0,
                            eval_candidate=complementary_eval,
                            wrap_position=wrap_periodic_position,
                        )
                    )

        if num_de_jumps > 0:

            def jump_eval(position, cache):
                del cache
                new_cache = build_cache(position)
                return (
                    log_prior_fn(position),
                    log_likelihood_from_cache_fn(position, new_cache),
                    new_cache,
                )

            cached_state, num_de_jump_acceptances = _apply_de_jumps(
                rng_key,
                cached_state,
                loglikelihood_0,
                live_positions,
                num_de_jumps,
                jump_eval,
                wrap_periodic_position,
            )

        targeted_de_jump_acceptances = []
        for group_index, (
            parameter_indices,
            requires_rebuild,
            attempts,
        ) in enumerate(resolved_de_jump_blocks):
            if requires_rebuild:

                def targeted_jump_eval(position, cache):
                    del cache
                    new_cache = build_cache(position)
                    return (
                        log_prior_fn(position),
                        log_likelihood_from_cache_fn(position, new_cache),
                        new_cache,
                    )

            else:

                def targeted_jump_eval(position, cache):
                    return (
                        log_prior_fn(position),
                        log_likelihood_from_cache_fn(position, cache),
                        cache,
                    )

            cached_state, group_acceptances = _apply_de_jumps(
                jax.random.fold_in(rng_key, group_index + 1),
                cached_state,
                loglikelihood_0,
                live_positions,
                attempts,
                targeted_jump_eval,
                wrap_periodic_position,
                parameter_indices,
            )
            targeted_de_jump_acceptances.append(group_acceptances)

        final_state = state._replace(
            position=cached_state.position,
            logdensity=cached_state.logdensity,
            loglikelihood=cached_state.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
        )
        slice_info = {
            "is_accepted": accepted,
            "num_expansions": num_expansions,
            "num_shrink": num_shrink,
            "bracket_left": jnp.zeros(n_dims),
            "bracket_right": jnp.zeros(n_dims),
        }
        if complementary_info is not None:
            independence_acceptances = jnp.stack(periodic_independence_acceptances)
            complementary_attempts = jnp.asarray(
                complementary_info.acceptances.size, dtype=jnp.int32
            )
            complementary_acceptances = complementary_info.acceptances.sum(
                dtype=jnp.int32
            )
            info = PeriodicIndependenceComplementaryDESwiGInfo(
                **slice_info,
                num_periodic_uniform_independence_attempts=jnp.asarray(
                    len(periodic_independence_acceptances), dtype=jnp.int32
                ),
                num_periodic_uniform_independence_acceptances=(
                    independence_acceptances.sum(dtype=jnp.int32)
                ),
                num_periodic_uniform_independence_attempts_by_block=jnp.ones(
                    independence_acceptances.shape, dtype=jnp.int32
                ),
                num_periodic_uniform_independence_acceptances_by_block=(
                    independence_acceptances.astype(jnp.int32)
                ),
                num_complementary_de_attempts=complementary_attempts,
                num_complementary_de_acceptances=complementary_acceptances,
                num_complementary_de_attempts_by_block=jnp.asarray(
                    [complementary_attempts], dtype=jnp.int32
                ),
                num_complementary_de_acceptances_by_block=jnp.asarray(
                    [complementary_acceptances], dtype=jnp.int32
                ),
                complementary_de_acceptances_by_attempt=(
                    complementary_info.acceptances
                ),
                complementary_de_donor_indices_by_attempt=(
                    complementary_info.donor_indices
                ),
                complementary_de_donor_policy_violations_by_attempt=(
                    complementary_info.donor_policy_violations
                ),
                complementary_de_complement_size=(complementary_info.complement_size),
                complementary_de_parent_index=jnp.asarray(
                    parent_index, dtype=jnp.int32
                ),
                complementary_de_position_before_by_attempt=(
                    complementary_info.position_before
                ),
                complementary_de_proposal_position_by_attempt=(
                    complementary_info.proposal_position
                ),
            )
        elif periodic_independence_acceptances:
            independence_acceptances = jnp.stack(periodic_independence_acceptances)
            info = PeriodicIndependenceSwiGInfo(
                **slice_info,
                num_periodic_uniform_independence_attempts=jnp.asarray(
                    len(periodic_independence_acceptances), dtype=jnp.int32
                ),
                num_periodic_uniform_independence_acceptances=(
                    independence_acceptances.sum(dtype=jnp.int32)
                ),
                num_periodic_uniform_independence_attempts_by_block=jnp.ones(
                    independence_acceptances.shape, dtype=jnp.int32
                ),
                num_periodic_uniform_independence_acceptances_by_block=(
                    independence_acceptances.astype(jnp.int32)
                ),
            )
        elif resolved_de_jump_blocks:
            targeted_attempts = jnp.asarray(
                [attempts for _, _, attempts in resolved_de_jump_blocks],
                dtype=jnp.int32,
            )
            targeted_acceptances = jnp.stack(targeted_de_jump_acceptances)
            info = TargetedSwiGInfo(
                **slice_info,
                num_de_jump_attempts=(
                    jnp.asarray(num_de_jumps, dtype=jnp.int32) + targeted_attempts.sum()
                ),
                num_de_jump_acceptances=(
                    num_de_jump_acceptances + targeted_acceptances.sum()
                ),
                num_targeted_de_jump_attempts=targeted_attempts,
                num_targeted_de_jump_acceptances=targeted_acceptances,
            )
        elif num_de_jumps > 0:
            info = SwiGInfo(
                **slice_info,
                num_de_jump_attempts=jnp.asarray(num_de_jumps, dtype=jnp.int32),
                num_de_jump_acceptances=num_de_jump_acceptances,
            )
        else:
            info = SliceInfo(**slice_info)
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
    num_slice_steps_by_block: Optional[Sequence[int]] = None,
    max_steps: int,
    max_shrinkage: int,
    periodic: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
    per_slice_info: bool = False,
    direction_mode: str = "covariance",
    de_fraction: float = 0.5,
    num_de_jumps: int = 0,
    resolved_de_jump_blocks: tuple[ResolvedDEJumpBlock, ...] = (),
    block_kernel_modes: Optional[Sequence[str]] = None,
    resolved_complementary_de_jump_block: Optional[ResolvedDEJumpBlock] = None,
    bracket_mode: str = "stepping-out",
) -> Callable:
    """Run a SwiG transition with one FSM loop per static cache segment."""
    if bracket_mode not in ("stepping-out", "shrink-only"):
        raise ValueError(f"Unsupported bracket_mode: {bracket_mode!r}")
    shrink_only = bracket_mode == "shrink-only"
    block_step_counts = _resolve_num_slice_steps_by_block(
        rebuild_required_by_block,
        num_inner_steps_per_dim,
        num_slice_steps_by_block,
    )
    resolved_block_kernel_modes = _resolve_block_kernel_modes(
        rebuild_required_by_block=rebuild_required_by_block,
        block_step_counts=block_step_counts,
        block_kernel_modes=block_kernel_modes,
        num_gibbs_sweeps=num_gibbs_sweeps,
        num_inner_steps_per_dim=num_inner_steps_per_dim,
        has_explicit_block_budget=num_slice_steps_by_block is not None,
        periodic=periodic,
        direction_mode=direction_mode,
        num_de_jumps=num_de_jumps,
        resolved_de_jump_blocks=resolved_de_jump_blocks,
    )
    if shrink_only and any(mode != "slice" for mode in resolved_block_kernel_modes):
        raise ValueError(
            "shrink-only brackets cannot combine with non-slice block_kernel_modes"
        )
    uses_periodic_independence = (
        "periodic-uniform-independence" in resolved_block_kernel_modes
    )
    if resolved_complementary_de_jump_block is not None:
        complementary_indices, complementary_rebuild, complementary_attempts = (
            resolved_complementary_de_jump_block
        )
        if not uses_periodic_independence:
            raise ValueError(
                "complementary DE requires the periodic hybrid FSM schedule"
            )
        if complementary_indices not in rebuild_required_by_block:
            raise ValueError(
                "complementary DE parameters must match one resolved slice block"
            )
        if (
            not complementary_rebuild
            or not rebuild_required_by_block[complementary_indices]
        ):
            raise ValueError(
                "complementary DE must target a waveform-rebuild slice block"
            )
        if complementary_attempts not in SUPPORTED_COMPLEMENTARY_DE_ATTEMPTS:
            raise ValueError("complementary DE requires exactly four or eight attempts")
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
        live_positions=None,
        live_loglikelihoods=None,
        parent_index=None,
        block_widths=None,
    ):
        uses_covariance_factors = block_covariance_factors is not None
        block_direction_parameters, sample_block_direction = (
            _resolve_block_direction_parameters(
                block_covariances,
                block_covariance_factors,
            )
        )
        if block_widths is not None and len(block_widths) != len(
            rebuild_required_by_block
        ):
            raise ValueError("block_widths must contain exactly one factor per block")
        if (
            direction_mode == "de-mix"
            or num_de_jumps > 0
            or resolved_de_jump_blocks
            or resolved_complementary_de_jump_block is not None
        ) and live_positions is None:
            raise ValueError("DE moves require live_positions parameters")
        if (
            direction_mode == "de-mix"
            or resolved_complementary_de_jump_block is not None
        ) and live_loglikelihoods is None:
            raise ValueError("DE survivor selection requires live_loglikelihoods")
        if resolved_complementary_de_jump_block is not None and parent_index is None:
            raise ValueError("complementary DE requires the original parent_index")
        cache = build_cache(state.position)
        cached_state = CachedSliceState(
            position=state.position,
            logdensity=state.logdensity,
            loglikelihood=state.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
            cache=cache,
        )

        if uses_periodic_independence:
            if str(jax.random.key_impl(rng_key)) != "threefry2x32":
                raise ValueError(
                    "The FSM scheduler requires Threefry PRNG keys for mixed "
                    "block kernels."
                )
            operation_records = []
            complementary_insertion_index = None
            key = rng_key
            for _ in range(num_gibbs_sweeps):
                for block_index, (
                    (parameter_indices, requires_rebuild),
                    direction_parameter,
                    n_steps,
                    block_kernel_mode,
                ) in enumerate(
                    zip(
                        rebuild_required_by_block.items(),
                        block_direction_parameters,
                        block_step_counts,
                        resolved_block_kernel_modes,
                        strict=True,
                    )
                ):
                    parameter_index_array = jnp.asarray(parameter_indices)
                    keys = jax.random.split(key, n_steps + 1)
                    key = keys[0]
                    if block_kernel_mode == "periodic-uniform-independence":
                        parameter_index = parameter_indices[0]
                        assert periodic is not None
                        lower, upper = periodic[parameter_index]
                        operation_records.append(
                            {
                                "requires_rebuild": requires_rebuild,
                                "is_independence": True,
                                "parameter_index": parameter_index,
                                "proposal_lower": lower,
                                "proposal_upper": upper,
                                "operation_key_data": jax.random.key_data(keys[1]),
                                "direction": jnp.zeros_like(state.position),
                                "level_u": jnp.asarray(0.5, dtype=state.position.dtype),
                                "bracket_u": jnp.asarray(
                                    0.5, dtype=state.position.dtype
                                ),
                                "bracket_v": jnp.asarray(
                                    0.5, dtype=state.position.dtype
                                ),
                                "shrink_key_data": jax.random.key_data(keys[1]),
                            }
                        )
                        continue

                    (
                        prop_keys,
                        level_u,
                        bracket_u,
                        bracket_v,
                        shrink_key_data,
                    ) = slice_randoms_from_keys(keys[1:])
                    block_directions = jnp.stack(
                        [
                            cast(
                                Array,
                                sample_block_direction(
                                    prop_keys[i],
                                    jnp.zeros_like(
                                        state.position[parameter_index_array]
                                    ),
                                    direction_parameter,
                                ),
                            )
                            for i in range(n_steps)
                        ]
                    )
                    if block_widths is not None:
                        block_directions = block_directions * block_widths[block_index]
                    directions = (
                        jnp.zeros((n_steps, n_dims), dtype=state.position.dtype)
                        .at[:, parameter_index_array]
                        .set(block_directions)
                    )
                    for slice_idx in range(n_steps):
                        operation_records.append(
                            {
                                "requires_rebuild": requires_rebuild,
                                "is_independence": False,
                                "parameter_index": 0,
                                "proposal_lower": 0.0,
                                "proposal_upper": 1.0,
                                "operation_key_data": jax.random.key_data(
                                    keys[slice_idx + 1]
                                ),
                                "direction": directions[slice_idx],
                                "level_u": level_u[slice_idx],
                                "bracket_u": bracket_u[slice_idx],
                                "bracket_v": bracket_v[slice_idx],
                                "shrink_key_data": shrink_key_data[slice_idx],
                            }
                        )
                    if (
                        resolved_complementary_de_jump_block is not None
                        and parameter_indices == resolved_complementary_de_jump_block[0]
                    ):
                        complementary_insertion_index = len(operation_records)

            complementary_schedule = None
            if resolved_complementary_de_jump_block is not None:
                if complementary_insertion_index is None:
                    raise ValueError(
                        "complementary DE target was not scheduled as a slice block"
                    )
                complementary_schedule = _prepare_complementary_de_proposals(
                    rng_key,
                    live_positions=live_positions,
                    live_loglikelihoods=live_loglikelihoods,
                    loglikelihood_0=loglikelihood_0,
                    parent_index=parent_index,
                    parameter_indices=resolved_complementary_de_jump_block[0],
                    attempts=resolved_complementary_de_jump_block[2],
                )
                complementary_records = []
                for attempt_index in range(resolved_complementary_de_jump_block[2]):
                    operation_key_data = complementary_schedule.operation_key_data[
                        attempt_index
                    ]
                    complementary_records.append(
                        {
                            "requires_rebuild": True,
                            "is_independence": False,
                            "is_complementary_de": True,
                            "parameter_index": 0,
                            "proposal_lower": 0.0,
                            "proposal_upper": 1.0,
                            "operation_key_data": operation_key_data,
                            "direction": jnp.zeros_like(state.position),
                            "complementary_de_displacement": (
                                complementary_schedule.displacement[attempt_index]
                            ),
                            "complementary_de_donor_indices": (
                                complementary_schedule.donor_indices[attempt_index]
                            ),
                            "complementary_de_donor_policy_violation": (
                                complementary_schedule.donor_policy_violations[
                                    attempt_index
                                ]
                            ),
                            "level_u": jnp.asarray(0.5, dtype=state.position.dtype),
                            "bracket_u": jnp.asarray(0.5, dtype=state.position.dtype),
                            "bracket_v": jnp.asarray(0.5, dtype=state.position.dtype),
                            "shrink_key_data": operation_key_data,
                        }
                    )
                operation_records[
                    complementary_insertion_index:complementary_insertion_index
                ] = complementary_records

            hybrid_segments = []
            run_start = 0
            for i in range(1, len(operation_records) + 1):
                if (
                    i == len(operation_records)
                    or operation_records[i]["requires_rebuild"]
                    != operation_records[run_start]["requires_rebuild"]
                ):
                    chunk = operation_records[run_start:i]
                    slice_indices = []
                    independence_indices = []
                    complementary_de_indices = []
                    n_slices = 0
                    n_independence = 0
                    n_complementary_de = 0
                    for record in chunk:
                        is_complementary_de = record.get("is_complementary_de", False)
                        is_slice = (
                            not record["is_independence"] and not is_complementary_de
                        )
                        slice_indices.append(n_slices if is_slice else 0)
                        independence_indices.append(
                            n_independence if record["is_independence"] else 0
                        )
                        complementary_de_indices.append(
                            n_complementary_de if is_complementary_de else 0
                        )
                        n_slices += is_slice
                        n_independence += record["is_independence"]
                        n_complementary_de += is_complementary_de
                    complementary_de_records = [
                        record
                        for record in chunk
                        if record.get("is_complementary_de", False)
                    ]
                    hybrid_segments.append(
                        (
                            chunk[0]["requires_rebuild"],
                            HybridSegmentSchedule(
                                is_independence=jnp.asarray(
                                    [record["is_independence"] for record in chunk]
                                ),
                                is_complementary_de=jnp.asarray(
                                    [
                                        record.get("is_complementary_de", False)
                                        for record in chunk
                                    ]
                                ),
                                parameter_index=jnp.asarray(
                                    [record["parameter_index"] for record in chunk],
                                    dtype=jnp.int32,
                                ),
                                proposal_lower=jnp.asarray(
                                    [record["proposal_lower"] for record in chunk],
                                    dtype=state.position.dtype,
                                ),
                                proposal_upper=jnp.asarray(
                                    [record["proposal_upper"] for record in chunk],
                                    dtype=state.position.dtype,
                                ),
                                operation_key_data=jnp.stack(
                                    [record["operation_key_data"] for record in chunk]
                                ),
                                slice_info_index=jnp.asarray(
                                    slice_indices, dtype=jnp.int32
                                ),
                                independence_info_index=jnp.asarray(
                                    independence_indices, dtype=jnp.int32
                                ),
                                complementary_de_info_index=jnp.asarray(
                                    complementary_de_indices, dtype=jnp.int32
                                ),
                                complementary_de_displacement=jnp.stack(
                                    [
                                        record.get(
                                            "complementary_de_displacement",
                                            jnp.zeros_like(state.position),
                                        )
                                        for record in chunk
                                    ]
                                ),
                                complementary_de_donor_indices=jnp.stack(
                                    [
                                        record["complementary_de_donor_indices"]
                                        for record in complementary_de_records
                                    ]
                                    or [jnp.full((2,), -1, dtype=jnp.int32)]
                                ),
                                complementary_de_donor_policy_violation=jnp.stack(
                                    [
                                        record[
                                            "complementary_de_donor_policy_violation"
                                        ]
                                        for record in complementary_de_records
                                    ]
                                    or [jnp.asarray(False)]
                                ),
                                directions=jnp.stack(
                                    [record["direction"] for record in chunk]
                                ),
                                level_u=jnp.stack(
                                    [record["level_u"] for record in chunk]
                                ),
                                bracket_u=jnp.stack(
                                    [record["bracket_u"] for record in chunk]
                                ),
                                bracket_v=jnp.stack(
                                    [record["bracket_v"] for record in chunk]
                                ),
                                shrink_key_data=jnp.stack(
                                    [record["shrink_key_data"] for record in chunk]
                                ),
                                n_slices=n_slices,
                                n_independence=n_independence,
                                n_complementary_de=n_complementary_de,
                                complementary_de_complement_size=(
                                    complementary_schedule.complement_size
                                    if n_complementary_de
                                    and complementary_schedule is not None
                                    else jnp.asarray(0, dtype=jnp.int32)
                                ),
                            ),
                        )
                    )
                    run_start = i

            hybrid_infos = []
            for requires_rebuild, schedule in hybrid_segments:
                cached_state, hybrid_info = run_hybrid_segment(
                    schedule,
                    cached_state,
                    loglikelihood_0,
                    eval_candidate=make_eval(requires_rebuild),
                    wrap_position=wrap_periodic_position,
                    max_expansions=max_steps,
                    max_shrinkage=max_shrinkage,
                )
                hybrid_infos.append(hybrid_info)

            slice_infos = [
                info.slice_info
                for info in hybrid_infos
                if info.slice_info.is_accepted.size
            ]
            accepted = (
                jnp.all(jnp.concatenate([info.is_accepted for info in slice_infos]))
                if slice_infos
                else jnp.asarray(True)
            )
            num_expansions = (
                jnp.concatenate([info.num_expansions for info in slice_infos])
                if slice_infos
                else jnp.zeros((0,), dtype=jnp.int32)
            )
            num_shrink = (
                jnp.concatenate([info.num_shrink for info in slice_infos])
                if slice_infos
                else jnp.zeros((0,), dtype=jnp.int32)
            )
            if not per_slice_info:
                num_expansions = num_expansions.sum()
                num_shrink = num_shrink.sum()
            independence_acceptances = jnp.concatenate(
                [
                    info.independence_acceptances
                    for info in hybrid_infos
                    if info.independence_acceptances.size
                ]
            )
            final_state = state._replace(
                position=cached_state.position,
                logdensity=cached_state.logdensity,
                loglikelihood=cached_state.loglikelihood,
                loglikelihood_birth=jnp.asarray(loglikelihood_0),
            )
            if resolved_complementary_de_jump_block is not None:
                complementary_infos = [
                    info
                    for info in hybrid_infos
                    if info.complementary_de_acceptances.size
                ]
                complementary_acceptances = jnp.concatenate(
                    [info.complementary_de_acceptances for info in complementary_infos]
                )
                complementary_donor_indices = jnp.concatenate(
                    [
                        info.complementary_de_donor_indices
                        for info in complementary_infos
                    ]
                )
                complementary_policy_violations = jnp.concatenate(
                    [
                        info.complementary_de_donor_policy_violations
                        for info in complementary_infos
                    ]
                )
                complementary_position_before = jnp.concatenate(
                    [
                        info.complementary_de_position_before
                        for info in complementary_infos
                    ]
                )
                complementary_proposal_position = jnp.concatenate(
                    [
                        info.complementary_de_proposal_position
                        for info in complementary_infos
                    ]
                )
                complementary_attempts = jnp.asarray(
                    complementary_acceptances.size, dtype=jnp.int32
                )
                complementary_acceptance_count = complementary_acceptances.sum(
                    dtype=jnp.int32
                )
                return final_state, PeriodicIndependenceComplementaryDESwiGInfo(
                    is_accepted=accepted,
                    num_expansions=num_expansions,
                    num_shrink=num_shrink,
                    bracket_left=jnp.zeros(n_dims),
                    bracket_right=jnp.zeros(n_dims),
                    num_periodic_uniform_independence_attempts=jnp.asarray(
                        independence_acceptances.size, dtype=jnp.int32
                    ),
                    num_periodic_uniform_independence_acceptances=(
                        independence_acceptances.sum(dtype=jnp.int32)
                    ),
                    num_periodic_uniform_independence_attempts_by_block=jnp.ones(
                        independence_acceptances.shape, dtype=jnp.int32
                    ),
                    num_periodic_uniform_independence_acceptances_by_block=(
                        independence_acceptances.astype(jnp.int32)
                    ),
                    num_complementary_de_attempts=complementary_attempts,
                    num_complementary_de_acceptances=(complementary_acceptance_count),
                    num_complementary_de_attempts_by_block=jnp.asarray(
                        [complementary_attempts], dtype=jnp.int32
                    ),
                    num_complementary_de_acceptances_by_block=jnp.asarray(
                        [complementary_acceptance_count], dtype=jnp.int32
                    ),
                    complementary_de_acceptances_by_attempt=(complementary_acceptances),
                    complementary_de_donor_indices_by_attempt=(
                        complementary_donor_indices
                    ),
                    complementary_de_donor_policy_violations_by_attempt=(
                        complementary_policy_violations
                    ),
                    complementary_de_complement_size=(
                        complementary_infos[0].complementary_de_complement_size
                    ),
                    complementary_de_parent_index=jnp.asarray(
                        parent_index, dtype=jnp.int32
                    ),
                    complementary_de_position_before_by_attempt=(
                        complementary_position_before
                    ),
                    complementary_de_proposal_position_by_attempt=(
                        complementary_proposal_position
                    ),
                )
            return final_state, PeriodicIndependenceSwiGInfo(
                is_accepted=accepted,
                num_expansions=num_expansions,
                num_shrink=num_shrink,
                bracket_left=jnp.zeros(n_dims),
                bracket_right=jnp.zeros(n_dims),
                num_periodic_uniform_independence_attempts=jnp.asarray(
                    independence_acceptances.size, dtype=jnp.int32
                ),
                num_periodic_uniform_independence_acceptances=(
                    independence_acceptances.sum(dtype=jnp.int32)
                ),
                num_periodic_uniform_independence_attempts_by_block=jnp.ones(
                    independence_acceptances.shape, dtype=jnp.int32
                ),
                num_periodic_uniform_independence_acceptances_by_block=(
                    independence_acceptances.astype(jnp.int32)
                ),
            )

        slice_records = []
        key = rng_key
        for _ in range(num_gibbs_sweeps):
            for block_index, (
                (parameter_indices, requires_rebuild),
                direction_parameter,
                n_steps,
            ) in enumerate(
                zip(
                    rebuild_required_by_block.items(),
                    block_direction_parameters,
                    block_step_counts,
                    strict=True,
                )
            ):
                parameter_index_array = jnp.asarray(parameter_indices)
                use_covariance_basis = (
                    direction_mode == "covariance-basis-8d"
                    and len(parameter_indices) == 8
                )
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
                if use_covariance_basis:
                    covariance_factor = (
                        direction_parameter
                        if uses_covariance_factors
                        else jnp.linalg.cholesky(direction_parameter)
                    )
                    block_directions = _sample_signed_permuted_covariance_basis(
                        prop_keys[0], covariance_factor
                    )
                    if block_widths is not None:
                        block_directions = block_directions * block_widths[block_index]
                elif direction_mode in (
                    "covariance",
                    "covariance-basis-8d",
                ):
                    block_directions = jnp.stack(
                        [
                            cast(
                                Array,
                                sample_block_direction(
                                    prop_keys[i],
                                    jnp.zeros_like(
                                        state.position[parameter_index_array]
                                    ),
                                    direction_parameter,
                                ),
                            )
                            for i in range(n_steps)
                        ]
                    )
                    if block_widths is not None:
                        block_directions = block_directions * block_widths[block_index]
                else:
                    if block_widths is not None:
                        raise ValueError(
                            "block_widths is not supported with de-mix directions"
                        )
                    block_directions = jnp.stack(
                        [
                            _sample_de_mix_direction(
                                prop_keys[i],
                                live_positions,
                                live_loglikelihoods,
                                state.position,
                                loglikelihood_0,
                                parameter_index_array,
                                de_fraction,
                                sample_block_direction,
                                direction_parameter,
                                jnp.zeros_like(state.position[parameter_index_array]),
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
                shrink_only=shrink_only,
            )
            segment_infos.append(segment_info)

        num_de_jump_acceptances = jnp.asarray(0, dtype=jnp.int32)
        if num_de_jumps > 0:
            # Full-space jumps touch every cache-relevant dimension, so they
            # always rebuild.  Jump evaluations are a fixed num_de_jumps per
            # replacement and are not folded into the slice expansion/shrink
            # counters.
            cached_state, num_de_jump_acceptances = _apply_de_jumps(
                key,
                cached_state,
                loglikelihood_0,
                live_positions,
                num_de_jumps,
                make_eval(True),
                wrap_periodic_position,
            )

        targeted_de_jump_acceptances = []
        for group_index, (
            parameter_indices,
            requires_rebuild,
            attempts,
        ) in enumerate(resolved_de_jump_blocks):
            cached_state, group_acceptances = _apply_de_jumps(
                jax.random.fold_in(key, group_index + 1),
                cached_state,
                loglikelihood_0,
                live_positions,
                attempts,
                make_eval(requires_rebuild),
                wrap_periodic_position,
                parameter_indices,
            )
            targeted_de_jump_acceptances.append(group_acceptances)

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
        slice_info = {
            "is_accepted": accepted,
            "num_expansions": num_expansions,
            "num_shrink": num_shrink,
            "bracket_left": jnp.zeros(n_dims),
            "bracket_right": jnp.zeros(n_dims),
        }
        if resolved_de_jump_blocks:
            targeted_attempts = jnp.asarray(
                [attempts for _, _, attempts in resolved_de_jump_blocks],
                dtype=jnp.int32,
            )
            targeted_acceptances = jnp.stack(targeted_de_jump_acceptances)
            info = TargetedSwiGInfo(
                **slice_info,
                num_de_jump_attempts=(
                    jnp.asarray(num_de_jumps, dtype=jnp.int32) + targeted_attempts.sum()
                ),
                num_de_jump_acceptances=(
                    num_de_jump_acceptances + targeted_acceptances.sum()
                ),
                num_targeted_de_jump_attempts=targeted_attempts,
                num_targeted_de_jump_acceptances=targeted_acceptances,
            )
        elif num_de_jumps > 0:
            info = SwiGInfo(
                **slice_info,
                num_de_jump_attempts=jnp.asarray(num_de_jumps, dtype=jnp.int32),
                num_de_jump_acceptances=num_de_jump_acceptances,
            )
        else:
            info = SliceInfo(**slice_info)
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
        resolved_de_jump_blocks: tuple[ResolvedDEJumpBlock, ...] = (),
        resolved_complementary_de_jump_block: Optional[ResolvedDEJumpBlock] = None,
    ) -> None:
        if (
            rebuild_required_by_block is None
            or build_cache is None
            or log_likelihood_from_cache_fn is None
        ):
            raise ValueError("BlackJAXSwiGSampler requires cache callbacks.")
        if periodic is not None and not isinstance(periodic, dict):
            raise TypeError("Cache-aware sampling requires dict-form periodic bounds.")
        block_kernel_modes = tuple(
            config.block_kernel_modes or ("slice",) * len(config.blocks)
        )
        if config.block_kernel_modes is not None:
            if len(rebuild_required_by_block) != len(block_kernel_modes):
                raise ValueError(
                    "Resolved cache blocks must align with block_kernel_modes."
                )
            for (parameter_indices, _), mode in zip(
                rebuild_required_by_block.items(), block_kernel_modes, strict=True
            ):
                if mode != "periodic-uniform-independence":
                    continue
                if len(parameter_indices) != 1:
                    raise ValueError(
                        "periodic-uniform-independence requires a resolved singleton block"
                    )
                parameter_index = parameter_indices[0]
                if periodic is None or parameter_index not in periodic:
                    raise ValueError(
                        "periodic-uniform-independence requires declared periodic bounds"
                    )
                lower, upper = periodic[parameter_index]
                if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                    raise ValueError(
                        "periodic-uniform-independence requires finite, increasing "
                        "periodic bounds"
                    )
        if len(resolved_de_jump_blocks) != len(config.de_jump_blocks):
            raise ValueError(
                "Resolved DE jump blocks must match the named sampler configuration."
            )
        for (parameter_indices, _, attempts), configured in zip(
            resolved_de_jump_blocks, config.de_jump_blocks, strict=True
        ):
            if not parameter_indices or any(
                index < 0 or index >= n_dims for index in parameter_indices
            ):
                raise ValueError("Resolved DE jump block contains invalid indices.")
            if attempts != configured.attempts:
                raise ValueError(
                    "Resolved DE jump attempts do not match the sampler configuration."
                )
        configured_complementary = config.complementary_de_jump_block
        if (configured_complementary is None) != (
            resolved_complementary_de_jump_block is None
        ):
            raise ValueError(
                "Resolved complementary DE block must match the sampler configuration."
            )
        if resolved_complementary_de_jump_block is not None:
            parameter_indices, requires_rebuild, attempts = (
                resolved_complementary_de_jump_block
            )
            assert configured_complementary is not None
            target_block_index = config.blocks.index(
                configured_complementary.parameters
            )
            expected_indices, expected_requires_rebuild = tuple(
                rebuild_required_by_block.items()
            )[target_block_index]
            if (
                parameter_indices != expected_indices
                or requires_rebuild != expected_requires_rebuild
            ):
                raise ValueError(
                    "Resolved complementary DE block does not match its configured "
                    "target block and cache price."
                )
            if attempts != configured_complementary.attempts:
                raise ValueError(
                    "Resolved complementary DE attempts do not match the sampler "
                    "configuration."
                )
            if (
                len(parameter_indices) != 8
                or any(index < 0 or index >= n_dims for index in parameter_indices)
                or not requires_rebuild
                or attempts not in SUPPORTED_COMPLEMENTARY_DE_ATTEMPTS
            ):
                raise ValueError(
                    "Resolved complementary DE block violates the supported fixed-work "
                    "contract."
                )

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
        self._block_kernel_modes = block_kernel_modes
        self._resolved_de_jump_blocks = tuple(resolved_de_jump_blocks)
        self._resolved_complementary_de_jump_block = (
            resolved_complementary_de_jump_block
        )
        uses_live_positions = bool(
            config.direction_mode == "de-mix"
            or config.num_de_jumps > 0
            or self._resolved_de_jump_blocks
            or self._resolved_complementary_de_jump_block is not None
        )
        uses_live_loglikelihoods = bool(
            config.direction_mode == "de-mix"
            or self._resolved_complementary_de_jump_block is not None
        )

        # Widths MUST be 0-d jnp arrays, never Python floats: they live inside
        # state.inner_kernel_params, whose every leaf gets a P() spec via
        # jax.tree.map at sharding.py:309 and is device_put by
        # place_replicated_state — Python-float leaves break that contract.
        adaptive_widths = config.adaptive_slice_widths
        block_step_counts = _resolve_num_slice_steps_by_block(
            rebuild_required_by_block,
            config.num_inner_steps_per_dim,
            config.num_slice_steps_by_block,
        )
        slice_to_block = _slice_to_block_map(block_step_counts, config.num_gibbs_sweeps)
        n_blocks = len(rebuild_required_by_block)
        initial_widths = tuple(jnp.asarray(1.0, dtype=float) for _ in range(n_blocks))
        self._initial_block_widths = initial_widths

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
            del rng_key
            parameters = {"block_covariances": block_covariances(state)}
            if uses_live_positions:
                parameters["live_positions"] = state.particles.position
            if uses_live_loglikelihoods:
                parameters["live_loglikelihoods"] = state.particles.loglikelihood
            if adaptive_widths:
                update_info = getattr(info, "update_info", None)
                previous_widths = (params or {}).get("block_widths")
                if update_info is None or previous_widths is None:
                    parameters["block_widths"] = initial_widths
                else:
                    parameters["block_widths"] = _updated_block_widths(
                        previous_widths,
                        update_info.num_expansions,
                        update_info.num_shrink,
                        slice_to_block=slice_to_block,
                        n_blocks=n_blocks,
                        rate=config.width_adaptation_rate,
                        target_expansions=config.width_target_expansions,
                        target_shrinks=config.width_target_shrinks,
                        shrink_only=config.bracket_mode == "shrink-only",
                    )
            else:
                del info, params
            return parameters

        def update_block_covariance_factors(rng_key, state, info, params=None):
            del rng_key
            covariance_factors = tuple(
                jnp.linalg.cholesky(covariance)
                for covariance in block_covariances(state)
            )
            parameters = {"block_covariance_factors": covariance_factors}
            if uses_live_positions:
                parameters["live_positions"] = state.particles.position
            if uses_live_loglikelihoods:
                parameters["live_loglikelihoods"] = state.particles.loglikelihood
            if adaptive_widths:
                update_info = getattr(info, "update_info", None)
                previous_widths = (params or {}).get("block_widths")
                if update_info is None or previous_widths is None:
                    parameters["block_widths"] = initial_widths
                else:
                    parameters["block_widths"] = _updated_block_widths(
                        previous_widths,
                        update_info.num_expansions,
                        update_info.num_shrink,
                        slice_to_block=slice_to_block,
                        n_blocks=n_blocks,
                        rate=config.width_adaptation_rate,
                        target_expansions=config.width_target_expansions,
                        target_shrinks=config.width_target_shrinks,
                        shrink_only=config.bracket_mode == "shrink-only",
                    )
            else:
                del info, params
            return parameters

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
        use_covariance_factors = (
            mesh is not None and self._swig_config.scheduler == "fsm"
        )
        if (
            self._swig_config.direction_mode == "de-mix"
            or self._swig_config.num_de_jumps > 0
            or self._resolved_de_jump_blocks
            or self._resolved_complementary_de_jump_block is not None
        ):
            # All inner-kernel parameters are derived from the live particles;
            # rebuild them so live_positions is present regardless of which
            # direction mode wrote the checkpoint.
            rebuild = (
                self._fsm_update_inner_kernel_params_fn
                if use_covariance_factors
                else self._update_inner_kernel_params_fn
            )
            result = state._replace(inner_kernel_params=rebuild(None, state, None))
        else:
            keys = set(params) - {"live_positions", "block_widths"}
            if not use_covariance_factors:
                if keys == {"block_covariances"}:
                    result = state
                elif keys == {"block_covariance_factors"}:
                    result = state._replace(
                        inner_kernel_params=self._update_inner_kernel_params_fn(
                            None, state, None
                        )
                    )
                else:
                    result = None
            else:
                if keys == {"block_covariance_factors"}:
                    result = state
                elif keys == {"block_covariances"}:
                    result = state._replace(
                        inner_kernel_params=self._fsm_update_inner_kernel_params_fn(
                            None, state, None
                        )
                    )
                else:
                    result = None
            if result is None:
                expected = (
                    "block_covariance_factors"
                    if use_covariance_factors
                    else "block_covariances"
                )
                raise ValueError(
                    "checkpoint has incompatible SwiG inner-kernel parameters: "
                    f"expected {expected!r}, found {sorted(keys)!r}"
                )
        normalised_params = dict(result.inner_kernel_params)
        if self._swig_config.adaptive_slice_widths:
            normalised_params["block_widths"] = params.get(
                "block_widths", self._initial_block_widths
            )
        else:
            normalised_params.pop("block_widths", None)
        result = result._replace(inner_kernel_params=normalised_params)
        return result

    def _get_diagnostics(self) -> dict[str, Any]:
        """Include fixed-cost DE jumps in SwiG likelihood-work accounting."""
        diagnostics = super()._get_diagnostics()
        update_info: Any = self._final_state.update_info
        if hasattr(update_info, "num_de_jump_attempts"):
            attempts_history = np.asarray(update_info.num_de_jump_attempts)
            acceptances_history = np.asarray(update_info.num_de_jump_acceptances)
        else:
            attempts_history = np.zeros_like(
                np.asarray(update_info.is_accepted), dtype=np.int32
            )
            acceptances_history = np.zeros_like(attempts_history)
        total_attempts = int(np.sum(attempts_history, dtype=np.int64))
        total_acceptances = int(np.sum(acceptances_history, dtype=np.int64))

        diagnostics.update(
            {
                "n_likelihood_evaluations": (
                    diagnostics["n_likelihood_evaluations"] + total_attempts
                ),
                "n_likelihood_evaluations_de_jumps": total_attempts,
                "n_de_jump_attempts": total_attempts,
                "n_de_jump_acceptances": total_acceptances,
                "de_jump_acceptance_rate": (
                    total_acceptances / total_attempts if total_attempts else None
                ),
                "de_jump_attempts_history": attempts_history,
                "de_jump_acceptances_history": acceptances_history,
            }
        )
        if hasattr(
            update_info,
            "num_periodic_uniform_independence_attempts",
        ):
            independence_attempts_history = np.asarray(
                update_info.num_periodic_uniform_independence_attempts
            )
            independence_acceptances_history = np.asarray(
                update_info.num_periodic_uniform_independence_acceptances
            )
            raw_attempts_by_block_history = np.asarray(
                update_info.num_periodic_uniform_independence_attempts_by_block
            )
            raw_acceptances_by_block_history = np.asarray(
                update_info.num_periodic_uniform_independence_acceptances_by_block
            )
            independence_attempts = int(
                np.sum(independence_attempts_history, dtype=np.int64)
            )
            independence_acceptances = int(
                np.sum(independence_acceptances_history, dtype=np.int64)
            )
            independence_block_configs = [
                (configured_block, resolved_block)
                for configured_block, resolved_block, mode in zip(
                    self._swig_config.blocks,
                    self._rebuild_required_by_block.items(),
                    self._block_kernel_modes,
                    strict=True,
                )
                if mode == "periodic-uniform-independence"
            ]
            n_independence_blocks = len(independence_block_configs)
            expected_attempt_axis = (
                self._swig_config.num_gibbs_sweeps * n_independence_blocks
            )
            if raw_attempts_by_block_history.shape[-1] != expected_attempt_axis:
                raise RuntimeError(
                    "periodic-independence telemetry has an incompatible attempt axis"
                )
            grouped_shape = (
                *raw_attempts_by_block_history.shape[:-1],
                self._swig_config.num_gibbs_sweeps,
                n_independence_blocks,
            )
            attempts_by_block_history = raw_attempts_by_block_history.reshape(
                grouped_shape
            ).sum(axis=-2, dtype=np.int64)
            acceptances_by_block_history = raw_acceptances_by_block_history.reshape(
                grouped_shape
            ).sum(axis=-2, dtype=np.int64)
            attempts_by_block = np.asarray(
                np.sum(
                    attempts_by_block_history.reshape(-1, n_independence_blocks),
                    axis=0,
                    dtype=np.int64,
                )
            ).reshape(-1)
            acceptances_by_block = np.asarray(
                np.sum(
                    acceptances_by_block_history.reshape(-1, n_independence_blocks),
                    axis=0,
                    dtype=np.int64,
                )
            ).reshape(-1)
            independence_blocks = []
            rebuild_attempts = 0
            for (
                (configured_block, (_, requires_rebuild)),
                block_attempts,
                block_acceptances,
            ) in zip(
                independence_block_configs,
                attempts_by_block,
                acceptances_by_block,
                strict=True,
            ):
                n_block_attempts = int(block_attempts)
                n_block_acceptances = int(block_acceptances)
                if requires_rebuild:
                    rebuild_attempts += n_block_attempts
                independence_blocks.append(
                    {
                        "parameters": list(configured_block),
                        "requires_waveform_rebuild": requires_rebuild,
                        "attempts_per_replacement": (
                            self._swig_config.num_gibbs_sweeps
                        ),
                        "n_attempts": n_block_attempts,
                        "n_acceptances": n_block_acceptances,
                        "acceptance_rate": (
                            n_block_acceptances / n_block_attempts
                            if n_block_attempts
                            else None
                        ),
                    }
                )
            diagnostics.update(
                {
                    "n_likelihood_evaluations": (
                        diagnostics["n_likelihood_evaluations"] + independence_attempts
                    ),
                    "n_likelihood_evaluations_periodic_uniform_independence": (
                        independence_attempts
                    ),
                    "n_likelihood_evaluations_periodic_uniform_independence_waveform_rebuild": (
                        rebuild_attempts
                    ),
                    "n_likelihood_evaluations_periodic_uniform_independence_cache_hit": (
                        independence_attempts - rebuild_attempts
                    ),
                    "n_periodic_uniform_independence_attempts": (independence_attempts),
                    "n_periodic_uniform_independence_acceptances": (
                        independence_acceptances
                    ),
                    "periodic_uniform_independence_acceptance_rate": (
                        independence_acceptances / independence_attempts
                        if independence_attempts
                        else None
                    ),
                    "periodic_uniform_independence_attempts_history": (
                        independence_attempts_history
                    ),
                    "periodic_uniform_independence_acceptances_history": (
                        independence_acceptances_history
                    ),
                    "periodic_uniform_independence_attempts_by_block_history": (
                        attempts_by_block_history
                    ),
                    "periodic_uniform_independence_acceptances_by_block_history": (
                        acceptances_by_block_history
                    ),
                    "periodic_uniform_independence_blocks": independence_blocks,
                }
            )
        if hasattr(update_info, "num_complementary_de_attempts"):
            configured_complementary = self._swig_config.complementary_de_jump_block
            resolved_complementary = self._resolved_complementary_de_jump_block
            if configured_complementary is None or resolved_complementary is None:
                raise RuntimeError(
                    "complementary-DE telemetry was produced without a configured "
                    "complementary DE block"
                )
            attempts_per_replacement = configured_complementary.attempts
            # Canonical diagnostics are indexed by replacement. JAX retains a
            # singleton lane axis for D=1/M=1 on most fields, while the scalar
            # parent index is already flat, so normalize the complete family.
            complementary_attempts_history = np.asarray(
                update_info.num_complementary_de_attempts
            ).reshape(-1)
            complementary_acceptances_history = np.asarray(
                update_info.num_complementary_de_acceptances
            ).reshape(-1)
            complementary_attempts_by_block_history = np.asarray(
                update_info.num_complementary_de_attempts_by_block
            ).reshape(-1, 1)
            complementary_acceptances_by_block_history = np.asarray(
                update_info.num_complementary_de_acceptances_by_block
            ).reshape(-1, 1)
            complementary_acceptances_by_attempt_history = np.asarray(
                update_info.complementary_de_acceptances_by_attempt
            ).reshape(-1, attempts_per_replacement)
            complementary_donor_indices_by_attempt_history = np.asarray(
                update_info.complementary_de_donor_indices_by_attempt
            ).reshape(-1, attempts_per_replacement, 2)
            complementary_donor_policy_violations_by_attempt_history = np.asarray(
                update_info.complementary_de_donor_policy_violations_by_attempt
            ).reshape(-1, attempts_per_replacement)
            complementary_complement_size_history = np.asarray(
                update_info.complementary_de_complement_size
            ).reshape(-1)
            complementary_parent_index_history = np.asarray(
                update_info.complementary_de_parent_index
            ).reshape(-1)
            complementary_position_before_by_attempt_history = np.asarray(
                update_info.complementary_de_position_before_by_attempt
            ).reshape(-1, attempts_per_replacement, self.n_dims)
            complementary_proposal_position_by_attempt_history = np.asarray(
                update_info.complementary_de_proposal_position_by_attempt
            ).reshape(-1, attempts_per_replacement, self.n_dims)
            complementary_attempts = int(
                np.sum(complementary_attempts_history, dtype=np.int64)
            )
            complementary_acceptances = int(
                np.sum(complementary_acceptances_history, dtype=np.int64)
            )
            complementary_donor_policy_violations = int(
                np.sum(
                    complementary_donor_policy_violations_by_attempt_history,
                    dtype=np.int64,
                )
            )
            complementary_attempts_by_block = np.asarray(
                np.sum(
                    complementary_attempts_by_block_history.reshape(-1, 1),
                    axis=0,
                    dtype=np.int64,
                )
            ).reshape(-1)
            complementary_acceptances_by_block = np.asarray(
                np.sum(
                    complementary_acceptances_by_block_history.reshape(-1, 1),
                    axis=0,
                    dtype=np.int64,
                )
            ).reshape(-1)
            _, complementary_requires_rebuild, _ = resolved_complementary
            complementary_blocks = []
            for block_attempts, block_acceptances in zip(
                complementary_attempts_by_block,
                complementary_acceptances_by_block,
                strict=True,
            ):
                n_block_attempts = int(block_attempts)
                n_block_acceptances = int(block_acceptances)
                complementary_blocks.append(
                    {
                        "parameters": list(configured_complementary.parameters),
                        "requires_waveform_rebuild": (complementary_requires_rebuild),
                        "attempts_per_replacement": (configured_complementary.attempts),
                        "gamma": 1.0,
                        "placement": "after-target-block",
                        "n_attempts": n_block_attempts,
                        "n_acceptances": n_block_acceptances,
                        "n_donor_policy_violations": (
                            complementary_donor_policy_violations
                        ),
                        "acceptance_rate": (
                            n_block_acceptances / n_block_attempts
                            if n_block_attempts
                            else None
                        ),
                    }
                )
            complementary_rebuild_attempts = (
                complementary_attempts if complementary_requires_rebuild else 0
            )
            diagnostics.update(
                {
                    "n_likelihood_evaluations": (
                        diagnostics["n_likelihood_evaluations"] + complementary_attempts
                    ),
                    "n_likelihood_evaluations_complementary_de": (
                        complementary_attempts
                    ),
                    "n_likelihood_evaluations_complementary_de_waveform_rebuild": (
                        complementary_rebuild_attempts
                    ),
                    "n_likelihood_evaluations_complementary_de_cache_hit": (
                        complementary_attempts - complementary_rebuild_attempts
                    ),
                    "n_complementary_de_attempts": complementary_attempts,
                    "n_complementary_de_acceptances": complementary_acceptances,
                    "n_complementary_de_donor_policy_violations": (
                        complementary_donor_policy_violations
                    ),
                    "complementary_de_acceptance_rate": (
                        complementary_acceptances / complementary_attempts
                        if complementary_attempts
                        else None
                    ),
                    "complementary_de_attempts_history": (
                        complementary_attempts_history
                    ),
                    "complementary_de_acceptances_history": (
                        complementary_acceptances_history
                    ),
                    "complementary_de_donor_policy_violations_history": np.sum(
                        complementary_donor_policy_violations_by_attempt_history,
                        axis=-1,
                        dtype=np.int32,
                    ),
                    "complementary_de_attempts_by_block_history": (
                        complementary_attempts_by_block_history
                    ),
                    "complementary_de_acceptances_by_block_history": (
                        complementary_acceptances_by_block_history
                    ),
                    "complementary_de_acceptances_by_attempt_history": (
                        complementary_acceptances_by_attempt_history
                    ),
                    "complementary_de_donor_indices_by_attempt_history": (
                        complementary_donor_indices_by_attempt_history
                    ),
                    "complementary_de_donor_policy_violations_by_attempt_history": (
                        complementary_donor_policy_violations_by_attempt_history
                    ),
                    "complementary_de_complement_size_history": (
                        complementary_complement_size_history
                    ),
                    "complementary_de_parent_index_history": (
                        complementary_parent_index_history
                    ),
                    "complementary_de_position_before_by_attempt_history": (
                        complementary_position_before_by_attempt_history
                    ),
                    "complementary_de_proposal_position_by_attempt_history": (
                        complementary_proposal_position_by_attempt_history
                    ),
                    "complementary_de_blocks": complementary_blocks,
                }
            )
        if self._resolved_de_jump_blocks:
            targeted_attempts_history = np.asarray(
                update_info.num_targeted_de_jump_attempts
            )
            targeted_acceptances_history = np.asarray(
                update_info.num_targeted_de_jump_acceptances
            )
            n_groups = len(self._resolved_de_jump_blocks)
            targeted_attempts = np.asarray(
                np.sum(
                    targeted_attempts_history.reshape(-1, n_groups),
                    axis=0,
                    dtype=np.int64,
                )
            ).reshape(-1)
            targeted_acceptances = np.asarray(
                np.sum(
                    targeted_acceptances_history.reshape(-1, n_groups),
                    axis=0,
                    dtype=np.int64,
                )
            ).reshape(-1)
            targeted_stats = []
            rebuild_attempts = total_attempts - int(targeted_attempts.sum())
            for configured, resolved, attempts, acceptances in zip(
                self._swig_config.de_jump_blocks,
                self._resolved_de_jump_blocks,
                targeted_attempts,
                targeted_acceptances,
                strict=True,
            ):
                _, requires_rebuild, _ = resolved
                n_attempts = int(attempts)
                n_acceptances = int(acceptances)
                if requires_rebuild:
                    rebuild_attempts += n_attempts
                targeted_stats.append(
                    {
                        "parameters": list(configured.parameters),
                        "requires_waveform_rebuild": requires_rebuild,
                        "attempts_per_replacement": configured.attempts,
                        "n_attempts": n_attempts,
                        "n_acceptances": n_acceptances,
                        "acceptance_rate": (
                            n_acceptances / n_attempts if n_attempts else None
                        ),
                    }
                )
            diagnostics.update(
                {
                    "targeted_de_jump_blocks": targeted_stats,
                    "targeted_de_jump_attempts_history": targeted_attempts_history,
                    "targeted_de_jump_acceptances_history": (
                        targeted_acceptances_history
                    ),
                    "n_likelihood_evaluations_de_jumps_waveform_rebuild": (
                        rebuild_attempts
                    ),
                    "n_likelihood_evaluations_de_jumps_cache_hit": (
                        total_attempts - rebuild_attempts
                    ),
                }
            )
        block_step_counts = _resolve_num_slice_steps_by_block(
            self._rebuild_required_by_block,
            self._swig_config.num_inner_steps_per_dim,
            self._swig_config.num_slice_steps_by_block,
        )
        slice_updates_per_replacement = self._swig_config.num_gibbs_sweeps * sum(
            count
            for count, mode in zip(
                block_step_counts, self._block_kernel_modes, strict=True
            )
            if mode == "slice"
        )
        n_replacements = np.asarray(update_info.is_accepted).size
        n_slice_updates = slice_updates_per_replacement * n_replacements
        # Stepping-out slice updates evaluate both bracket endpoints (2 evals
        # each); shrink-only bracket mode never evaluates an endpoint, so the
        # endpoint term must vanish under that mode.
        n_endpoint_evals_per_slice_update = (
            0 if self._swig_config.bracket_mode == "shrink-only" else 2
        )
        diagnostics.update(
            {
                "n_slice_updates": n_slice_updates,
                "n_likelihood_evaluations_physical": (
                    diagnostics["n_likelihood_evaluations"]
                    + n_endpoint_evals_per_slice_update * n_slice_updates
                ),
            }
        )
        return diagnostics

    def _build_nested_sampler(
        self, n_delete: int, mesh: Optional[Mesh] = None
    ) -> SamplingAlgorithm:
        is_fsm_builder = self._swig_config.scheduler != "pre-fsm-lockstep"
        constrained_step_builder = cast(
            Callable[..., Any],
            (
                _build_swig_constrained_step
                if is_fsm_builder
                else _build_swig_constrained_step_lockstep
            ),
        )
        constrained_step_kwargs: dict[str, Any] = {
            "log_prior_fn": self._log_prior_fn,
            "build_cache": self._build_cache,
            "log_likelihood_from_cache_fn": self._log_likelihood_from_cache_fn,
            "rebuild_required_by_block": self._rebuild_required_by_block,
            "num_gibbs_sweeps": self._swig_config.num_gibbs_sweeps,
            "num_inner_steps_per_dim": self._swig_config.num_inner_steps_per_dim,
            "num_slice_steps_by_block": self._swig_config.num_slice_steps_by_block,
            "max_steps": self._swig_config.max_steps,
            "max_shrinkage": self._swig_config.max_shrinkage,
            "periodic": self._periodic,
            "n_dims": self.n_dims,
            "direction_mode": self._swig_config.direction_mode,
            "de_fraction": self._swig_config.de_fraction,
            "num_de_jumps": self._swig_config.num_de_jumps,
            "resolved_de_jump_blocks": self._resolved_de_jump_blocks,
            "block_kernel_modes": self._block_kernel_modes,
            "resolved_complementary_de_jump_block": (
                self._resolved_complementary_de_jump_block
            ),
            "bracket_mode": self._swig_config.bracket_mode,
        }
        # per_slice_info only exists on the FSM builder's signature; passing
        # it to the lockstep builder would raise a TypeError.
        if is_fsm_builder:
            constrained_step_kwargs["per_slice_info"] = (
                self._swig_config.adaptive_slice_widths
            )
        constrained_step = constrained_step_builder(**constrained_step_kwargs)
        if mesh is None:
            if self._resolved_complementary_de_jump_block is None:
                kernel = build_from_mcmc_kernel(
                    constrained_step,
                    num_inner_steps=1,
                    update_inner_kernel_params_fn=self._update_block_covariances,
                    num_delete=n_delete,
                )
            else:
                kernel = build_from_mcmc_kernel_with_parent_index(
                    constrained_step,
                    n_inner_steps=1,
                    update_inner_kernel_params_fn=self._update_block_covariances,
                    n_delete=n_delete,
                )
        else:
            if self._resolved_complementary_de_jump_block is None:
                kernel = build_replicated_from_mcmc_kernel(
                    constrained_step,
                    n_inner_steps=1,
                    update_inner_kernel_params_fn=(
                        self._fsm_update_inner_kernel_params_fn
                    ),
                    n_delete=n_delete,
                    mesh=mesh,
                )
            else:
                kernel = build_replicated_from_mcmc_kernel(
                    constrained_step,
                    n_inner_steps=1,
                    update_inner_kernel_params_fn=(
                        self._fsm_update_inner_kernel_params_fn
                    ),
                    n_delete=n_delete,
                    mesh=mesh,
                    pass_parent_index=True,
                )

        # `nested_sampler.init` is never called (state init happens in
        # `_batched_nss_init`); BlackJAX still requires SamplingAlgorithm.init
        # to type as returning a State.
        return SamplingAlgorithm(
            lambda position, rng_key=None: position,  # type: ignore[return-value]
            kernel,
        )
