"""BlackJAX Nested Slice Sampling (NSS)."""

import logging
import pickle
import shutil
import time
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
from anesthetic.samples import NestedSamples
from blackjax import SamplingAlgorithm, nss
from blackjax.mcmc.slice import SliceInfo
from blackjax.ns.adaptive import AdaptiveNSState
from blackjax.ns.adaptive import init as _ns_adaptive_init
from blackjax.ns.base import NSInfo
from blackjax.ns.base import init_state_strategy as _init_state_strategy
from blackjax.ns.nss import (
    live_covariance,
    sample_direction_from_covariance,
)
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from jax.flatten_util import ravel_pytree
from jax.sharding import Mesh
from jaxtyping import Array, Float, Key
from scipy.special import logsumexp

from jimgw.samplers.base import Sampler
from jimgw.samplers.blackjax._fsm import (
    SegmentSchedule,
    run_segment,
    slice_randoms_from_keys,
)
from jimgw.samplers.blackjax.sharding import (
    build_replicated_from_mcmc_kernel,
    make_live_mesh,
    place_key,
    place_replicated_state,
    replacement_sharding,
    replicate_initial_particles,
)
from jimgw.samplers.config import BlackJAXNSSConfig
from jimgw.samplers.periodic import _build_masks_arrays, to_prior_space_proposal

logger = logging.getLogger(__name__)


class _FsmParticleState(NamedTuple):
    """Particle state extended with an empty cache for the generic FSM runner."""

    position: object
    logdensity: object
    loglikelihood: object
    loglikelihood_birth: object
    cache: object


def _sample_direction_from_covariance_factor(
    rng_key, position, covariance_factor: Array
):
    """Draw a direction with Mahalanobis norm two from a precomputed factor."""
    _, unravel_fn = ravel_pytree(position)
    standard_direction = jax.random.normal(
        rng_key,
        shape=(covariance_factor.shape[-1],),
        dtype=covariance_factor.dtype,
    )
    direction = (
        2.0
        * (covariance_factor @ standard_direction)
        / jnp.linalg.norm(standard_direction)
    )
    return unravel_fn(direction)


def _live_covariance_factor(rng_key, state, info, params=None):
    """Factor the live covariance once for the next outer NS-FSM step."""
    del rng_key, info, params
    covariance = jnp.atleast_2d(particles_covariance_matrix(state.particles.position))
    return {"covariance_factor": jnp.linalg.cholesky(covariance)}


def _resolve_covariance_factor(cov, covariance_factor):
    """Resolve optimized parameters, accepting legacy covariance checkpoints."""
    if covariance_factor is None:
        if cov is None:
            raise ValueError("Specify either covariance_factor or cov.")
        return jnp.linalg.cholesky(cov)
    if cov is not None:
        raise ValueError("Specify only one of covariance_factor and cov.")
    return covariance_factor


def _build_nss_fsm_constrained_step(
    *,
    log_prior_fn: Callable,
    log_likelihood_fn: Callable,
    periodic: Optional[dict[int, tuple[float, float]]],
    n_dims: int,
    max_expansions: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Build an NSS transition that folds every inner slice into one segment."""
    mask, lower, period = _build_masks_arrays(periodic, n_dims)

    def wrap_position(position):
        return jnp.where(mask, lower + jnp.mod(position - lower, period), position)

    def eval_candidate(position, cache):
        return log_prior_fn(position), log_likelihood_fn(position), cache

    def constrained_step(
        keys,
        state,
        loglikelihood_0,
        cov=None,
        *,
        covariance_factor=None,
    ):
        covariance_factor = _resolve_covariance_factor(cov, covariance_factor)
        prop_keys, level_u, bracket_u, bracket_v, shrink_key_data = (
            slice_randoms_from_keys(keys)
        )
        direction_template = jnp.zeros_like(state.position)
        # A vmap changes x64 direction-normalization rounding relative to the
        # reference scan. Keep these static draws unrolled so the folded segment
        # remains bitwise pathwise-equivalent without a preprocessing loop.
        directions = jnp.stack(
            tuple(
                cast(
                    Array,
                    _sample_direction_from_covariance_factor(
                        prop_keys[slice_idx],
                        direction_template,
                        covariance_factor,
                    ),
                )
                for slice_idx in range(prop_keys.shape[0])
            )
        )
        schedule = SegmentSchedule(
            directions=directions,
            level_u=level_u,
            bracket_u=bracket_u,
            bracket_v=bracket_v,
            shrink_key_data=shrink_key_data,
        )
        entry = _FsmParticleState(
            position=state.position,
            logdensity=state.logdensity,
            loglikelihood=state.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
            cache=(),
        )
        final, segment_info = run_segment(
            schedule,
            entry,
            loglikelihood_0,
            eval_candidate=eval_candidate,
            wrap_position=wrap_position,
            max_expansions=max_expansions,
            max_shrinkage=max_shrinkage,
        )
        new_state = state._replace(
            position=final.position,
            logdensity=final.logdensity,
            loglikelihood=final.loglikelihood,
            loglikelihood_birth=jnp.asarray(loglikelihood_0),
        )
        info = SliceInfo(
            is_accepted=segment_info.is_accepted,
            num_expansions=segment_info.num_expansions,
            num_shrink=segment_info.num_shrink,
            bracket_left=segment_info.bracket_left,
            bracket_right=segment_info.bracket_right,
        )
        return new_state, info

    return constrained_step


def _termination_reached(integrator, termination_dlogz: float) -> bool:
    """Single-sync, NaN-safe nested-sampling termination check.

    ``dlogz = logaddexp(0, logZ_live - logZ)`` estimates the remaining
    evidence fraction still held by the live points; it is always >= 0, so
    the only non-finite values it can take are ``+inf`` or ``NaN``.
    ``+inf`` is the *expected* value before the first iteration --
    ``blackjax``'s integrator initialises ``logZ = -inf`` with a finite
    ``logZ_live``, so ``logZ_live - logZ = +inf`` at start-up -- and must
    **not** terminate the loop.  ``NaN`` is an indeterminate ``inf - inf``
    (e.g. ``logZ_live`` and ``logZ`` simultaneously ``-inf``: no live
    evidence *and* no dead evidence), which cannot shrink further, so it is
    treated as terminal to avoid an infinite loop.  The previous
    ``bool(jnp.isfinite(dlogz) and dlogz < threshold)`` implementation
    silently never terminated on NaN: Python's ``and`` short-circuits on the
    falsy left-hand array, returning it (which is falsy) without ever
    evaluating the finite-and-below-threshold comparison, and it forced two
    device syncs (one per ``bool()``) instead of one.
    """
    dlogz = jnp.logaddexp(0, integrator.logZ_live - integrator.logZ)
    is_nan = jnp.isnan(dlogz)
    terminate = bool(is_nan | (dlogz < termination_dlogz))
    if terminate and bool(is_nan):
        logger.warning(
            "Nested sampling termination criterion dlogz is NaN "
            "(logZ_live=%s, logZ=%s); terminating the run to avoid an "
            "infinite loop.",
            integrator.logZ_live,
            integrator.logZ,
        )
    return terminate


def _finalise_on_host(state, dead: list) -> NSInfo:
    """Host-numpy equivalent of ``blackjax.ns.utils.finalise``.

    Avoids tracing one variadic device concatenate over the whole dead
    history (n_iter operands x ~24 leaves) at the end of every run: pulling
    the (already small) dead-point history and final live state to the host
    once and concatenating with numpy is value-identical and orders of
    magnitude cheaper than compiling blackjax's device-side ``finalise``.
    """
    if not dead:
        raise ValueError("finalise requires at least one dead-point batch")
    host_dead = jax.device_get(dead)
    host_particles = jax.device_get(state.particles)
    update_info = jax.tree.map(
        lambda *leaves: np.concatenate([np.asarray(leaf) for leaf in leaves], axis=0),
        *[info.update_info for info in host_dead],
    )
    particles = jax.tree.map(
        lambda *leaves: np.concatenate([np.asarray(leaf) for leaf in leaves], axis=0),
        *([info.particles for info in host_dead] + [host_particles]),
    )
    return NSInfo(particles, update_info)


# Module-level alias so ``_sample`` resolves ``finalise`` dynamically through
# the module's global namespace at call time (not an early-bound reference).
# ``tests/unit/samplers/blackjax/test_swig.py`` monkeypatches
# ``jimgw.samplers.blackjax.nss.finalise`` to spy on the live
# ``AdaptiveNSState`` just before it is combined with the dead-point history
# (the state itself is discarded once finalise runs); keeping the call site
# spelled ``finalise(...)`` preserves that hook while defaulting to the
# faster host-numpy implementation.
finalise = _finalise_on_host


class BlackJAXNSSSampler(Sampler):
    """BlackJAX Nested Slice Sampler (NSS).

    NSS combines nested sampling with an adaptive slice-sampling inner kernel.
    It works directly in the sampling space defined by ``sample_transforms``
    (no unit-cube constraint required).  Operates on flat arrays of shape
    ``(n_dims,)``; the NSS kernel is pytree-generic.

    Configure via [`BlackJAXNSSConfig`][jimgw.samplers.config.BlackJAXNSSConfig].

    Args:
        n_dims: Dimension of the sampling space.
        log_prior_fn: Log-prior callable ``(arr,) -> float``.
        log_likelihood_fn: Log-likelihood callable ``(arr,) -> float``.
        log_posterior_fn: Log-posterior callable ``(arr,) -> float``.
        config: Optional ``BlackJAXNSSConfig``; defaults to all-default values.
        periodic: Optional periodic-parameter spec in index space,
            ``dict[int, (lo, hi)]`` where the key is the dimension index and
            the value is the ``(lower, upper)`` period bounds.  ``None`` means
            no periodic parameters.  Provided by Jim after resolving names.
    """

    _config: BlackJAXNSSConfig
    _proposal: Callable
    _final_state: NSInfo
    _nested_samples: NestedSamples
    _n_iterations: int

    def __init__(
        self,
        *,
        n_dims: int,
        log_prior_fn: Callable,
        log_likelihood_fn: Callable,
        log_posterior_fn: Callable,
        config: Optional[BlackJAXNSSConfig] = None,
        periodic: Optional[dict[int, tuple[float, float]]] = None,
    ) -> None:
        if config is None:
            config = BlackJAXNSSConfig()
        super().__init__(
            n_dims=n_dims,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )
        self._proposal = to_prior_space_proposal(
            periodic, n_dims, sample_direction_from_covariance
        )
        self._periodic_index = periodic

    @property
    def sampler_name(self) -> str:
        return "BlackJAX NSS"

    @property
    def _update_inner_kernel_params_fn(self) -> Callable:
        return live_covariance

    @property
    def _fsm_update_inner_kernel_params_fn(self) -> Callable:
        return _live_covariance_factor

    def _inner_kernel_params_fn_for_mesh(self, mesh: Optional[Mesh]) -> Callable:
        if mesh is None:
            return self._update_inner_kernel_params_fn
        return self._fsm_update_inner_kernel_params_fn

    def _normalise_inner_kernel_params_for_mesh(
        self, state: AdaptiveNSState, mesh: Optional[Mesh]
    ) -> AdaptiveNSState:
        """Convert checkpoint parameters to the representation a path expects."""
        try:
            params = state.inner_kernel_params
        except AttributeError as error:
            raise ValueError(
                "checkpoint state has no inner-kernel parameters"
            ) from error
        keys = set(params)
        if mesh is None:
            if keys == {"cov"}:
                return state
            if keys == {"covariance_factor"}:
                return state._replace(
                    inner_kernel_params=self._update_inner_kernel_params_fn(
                        None, state, None
                    )
                )
        else:
            if keys == {"covariance_factor"}:
                return state
            if keys == {"cov"}:
                return state._replace(
                    inner_kernel_params=self._fsm_update_inner_kernel_params_fn(
                        None, state, None
                    )
                )
        expected = "cov" if mesh is None else "covariance_factor"
        raise ValueError(
            "checkpoint has incompatible inner-kernel parameters: "
            f"expected {expected!r}, found {sorted(keys)!r}"
        )

    def _build_nested_sampler(self, n_delete: int, mesh: Optional[Mesh] = None):
        config = self._config
        num_inner_steps = config.num_inner_steps_per_dim * self.n_dims
        if mesh is not None:
            constrained_step = _build_nss_fsm_constrained_step(
                log_prior_fn=self._log_prior_fn,
                log_likelihood_fn=self._log_likelihood_fn,
                periodic=self._periodic_index,
                n_dims=self.n_dims,
            )
            kernel = build_replicated_from_mcmc_kernel(
                constrained_step,
                n_inner_steps=num_inner_steps,
                update_inner_kernel_params_fn=self._fsm_update_inner_kernel_params_fn,
                n_delete=n_delete,
                mesh=mesh,
                fold_inner_steps=True,
            )
            # `nested_sampler.init` is never called (state init happens in
            # `_batched_nss_init`); BlackJAX still requires SamplingAlgorithm.init
            # to type as returning a State.
            return SamplingAlgorithm(
                lambda position, rng_key=None: position,  # type: ignore[return-value]
                kernel,
            )
        return nss(
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
            num_delete=n_delete,
            num_inner_steps=num_inner_steps,
            proposal=self._proposal,
        )

    def _sample(
        self,
        rng_key: Key,
        initial_position: Float[Array, "n_live n_dims"],
    ) -> None:
        """Run the BlackJAX NSS sampler.

        If ``config.checkpoint_dir`` is set, a ``checkpoint.pkl`` is written
        atomically after each nested-sampling iteration (subject to
        ``config.checkpoint_interval``) and the sampler resumes from the
        checkpoint if one already exists at that path.

        Args:
            rng_key: JAX PRNG key.
            initial_position: Starting live points in the sampling space,
                shape ``(n_live, n_dims)``.  Must match ``config.n_live``.
                Ignored when resuming from a checkpoint.

        Raises:
            ValueError: If ``initial_position`` shape does not match
                ``(n_live, n_dims)``.
        """
        config = self._config
        n_live = config.n_live
        n_delete = int(n_live * config.n_delete_frac)
        mesh = make_live_mesh(config.n_devices, n_live, n_delete)
        ckpt_path = (
            config.checkpoint_dir / "checkpoint.pkl"
            if config.checkpoint_dir is not None
            else None
        )
        config.configure_jax_cache()
        _method_t0 = time.perf_counter()
        phase_seconds: dict[str, float | None] = {
            "init_total": None,
            "likelihood_jit": None,
            "initial_likelihood_eval": None,
            "sampler_kernel_jit": None,
            "ns_loop": None,
            "finalise": None,
        }

        def _validated_initial_particles(pos):
            arr = jnp.asarray(pos)
            if arr.ndim != 2 or arr.shape != (n_live, self.n_dims):
                raise ValueError(
                    f"initial_position must have shape ({n_live}, {self.n_dims}), "
                    f"got {arr.shape}."
                )
            return arr

        nested_sampler = self._build_nested_sampler(n_delete, mesh)
        update_inner_kernel_params_fn = self._inner_kernel_params_fn_for_mesh(mesh)

        # Bypass BlackJAX's jax.vmap(init_state_fn) to avoid peak-memory OOM.
        # A full vmap over all live particles materialises O(n_live) concurrent
        # intermediate buffers, which can exceed available GPU memory for expensive
        # likelihoods. lax.map with n_delete particles per batch bounds peak memory
        # to n_delete/n_live of the full-vmap cost at no extra computation.
        _single_init_fn = partial(
            _init_state_strategy,
            logprior_fn=self._log_prior_fn,
            loglikelihood_fn=self._log_likelihood_fn,
        )

        def _batched_nss_init(positions):
            _init_t0 = time.perf_counter()
            if mesh is not None:
                positions = jax.device_put(positions, replacement_sharding(mesh))

            def _batched_fn(pos):
                return jax.lax.map(_single_init_fn, pos, batch_size=n_delete)

            if mesh is None:
                _adaptive_init_t0 = time.perf_counter()
                state = _ns_adaptive_init(
                    positions,
                    init_state_fn=_batched_fn,
                    update_inner_kernel_params_fn=update_inner_kernel_params_fn,
                )
                phase_seconds["init_adaptive_init"] = (
                    time.perf_counter() - _adaptive_init_t0
                )
                phase_seconds["init_total"] = time.perf_counter() - _init_t0
                return state

            # Evaluate each initial likelihood exactly once on a sharded live
            # batch, then exchange only the compact particle records.  All
            # steady-state sampler data is replicated after this one-time step.
            # AOT-compile so the one-off likelihood JIT cost is measured
            # separately from the evaluation itself; arXiv:2607.28265 quotes
            # the likelihood and sampler-kernel compiles separately from the
            # sampling time.
            _jit_t0 = time.perf_counter()
            _compiled_init = (
                jax.jit(_batched_fn, out_shardings=replacement_sharding(mesh))
                .lower(positions)
                .compile()
            )
            phase_seconds["likelihood_jit"] = time.perf_counter() - _jit_t0
            _eval_t0 = time.perf_counter()
            initial_particles = jax.block_until_ready(_compiled_init(positions))
            phase_seconds["initial_likelihood_eval"] = time.perf_counter() - _eval_t0
            _replication_t0 = time.perf_counter()
            initial_particles = replicate_initial_particles(initial_particles, mesh)
            phase_seconds["init_particle_replication"] = (
                time.perf_counter() - _replication_t0
            )
            _adaptive_init_t0 = time.perf_counter()
            state = _ns_adaptive_init(
                initial_particles.position,
                init_state_fn=lambda _: initial_particles,
                update_inner_kernel_params_fn=update_inner_kernel_params_fn,
            )
            phase_seconds["init_adaptive_init"] = (
                time.perf_counter() - _adaptive_init_t0
            )
            _placement_t0 = time.perf_counter()
            state = place_replicated_state(state, mesh)
            phase_seconds["init_state_placement"] = time.perf_counter() - _placement_t0
            phase_seconds["init_total"] = time.perf_counter() - _init_t0
            return state

        # Resume from checkpoint if one exists.
        if (
            ckpt_path is not None
            and config.checkpoint_interval > 0
            and ckpt_path.exists()
        ):
            _initial_rng_key = rng_key
            try:
                with open(ckpt_path, "rb") as _f:
                    _ckpt = pickle.load(_f)
                self._validate_checkpoint(_ckpt)
                state = _ckpt["state"]
                state = self._normalise_inner_kernel_params_for_mesh(state, mesh)
                dead = _ckpt["dead"]
                rng_key = _ckpt["rng_key"]
                if mesh is not None:
                    state = place_replicated_state(state, mesh)
                    rng_key = place_key(rng_key, mesh)
                n_iter = _ckpt["n_iter"]
                self._prev_elapsed = float(_ckpt["elapsed_time"])
                logger.info(
                    "%s: resumed from checkpoint at n_iter=%d (%s)",
                    self.sampler_name,
                    n_iter,
                    ckpt_path,
                )
            except (
                OSError,
                EOFError,
                KeyError,
                TypeError,
                ValueError,
                pickle.UnpicklingError,
            ) as _e:
                logger.warning(
                    "%s: incompatible or corrupt checkpoint at %s (%s) — starting fresh.",
                    self.sampler_name,
                    ckpt_path,
                    _e,
                )
                rng_key = _initial_rng_key
                state = _batched_nss_init(
                    _validated_initial_particles(initial_position)
                )
                dead = []
                n_iter = 0
                self._prev_elapsed = 0.0
        else:
            state = _batched_nss_init(_validated_initial_particles(initial_position))
            dead = []
            n_iter = 0

        if mesh is not None:
            rng_key = place_key(rng_key, mesh)

        def _terminate(state: AdaptiveNSState) -> bool:
            return _termination_reached(state.integrator, config.termination_dlogz)

        # Compile the outer sampler kernel explicitly so its one-off cost is
        # measured independently of steady-state nested-sampling work.  This
        # mirrors the paper's timing convention, which quotes likelihood and
        # sampler-kernel JIT separately from sampling wall time.  The fallback
        # preserves benchmark observers that intentionally wrap ``jax.jit``
        # with a plain callable to time the lazy first invocation themselves.
        jitted_step_fn = jax.jit(nested_sampler.step)
        step_fn: Callable[[Any, Any], Any]
        lower_step = getattr(jitted_step_fn, "lower", None)
        if callable(lower_step):
            _sampler_jit_t0 = time.perf_counter()
            compile_step = getattr(lower_step(rng_key, state), "compile", None)
            if not callable(compile_step):
                raise TypeError("lowered sampler kernel is not compilable")
            step_fn = cast(Callable[[Any, Any], Any], compile_step())
            phase_seconds["sampler_kernel_jit"] = time.perf_counter() - _sampler_jit_t0
        else:
            step_fn = cast(Callable[[Any, Any], Any], jitted_step_fn)
        _last_ckpt_t = time.perf_counter()

        _loop_t0 = time.perf_counter()
        while not _terminate(state):
            rng_key, subkey = jax.random.split(rng_key)
            state, dead_info = step_fn(subkey, state)
            dead.append(dead_info)
            n_iter += 1
            if (
                ckpt_path is not None
                and config.checkpoint_interval > 0
                and time.perf_counter() - _last_ckpt_t >= config.checkpoint_interval
            ):
                _last_ckpt_t = config.write_checkpoint(
                    {
                        "state": jax.device_get(state),
                        "dead": jax.device_get(dead),
                        "rng_key": jax.device_get(rng_key),
                        "n_iter": n_iter,
                        "sampler_name": self.sampler_name,
                        "elapsed_time": self._prev_elapsed
                        + (time.perf_counter() - _method_t0),
                    },
                    self.sampler_name,
                )
        phase_seconds["ns_loop"] = time.perf_counter() - _loop_t0

        _finalise_t0 = time.perf_counter()
        final_state = finalise(state, dead)  # type: ignore[arg-type]  # AdaptiveNSState structurally satisfies NSState (.particles field)
        self._final_state = final_state
        self._n_iterations = n_iter

        # Build anesthetic NestedSamples for use in get_samples() and get_diagnostics().
        particles_sample = np.array(self._final_state.particles.position)
        log_likelihood = np.array(self._final_state.particles.loglikelihood)
        logL_birth = np.array(self._final_state.particles.loglikelihood_birth)
        logL_birth = np.where(np.isnan(logL_birth), -np.inf, logL_birth)
        self._nested_samples = NestedSamples(
            particles_sample,
            logL=log_likelihood,
            logL_birth=logL_birth,
            logzero=np.nan,
            dtype=np.float64,
        )
        phase_seconds["finalise"] = time.perf_counter() - _finalise_t0
        self._phase_seconds = phase_seconds
        if ckpt_path is not None:
            ckpt_path.unlink(missing_ok=True)
        if config.checkpoint_dir is not None:
            shutil.rmtree(config.checkpoint_dir / "jax_cache", ignore_errors=True)
            jax.config.update("jax_compilation_cache_dir", None)

    def get_samples(self) -> dict[str, np.ndarray]:
        """Return equally-weighted posterior samples via anesthetic's ``posterior_points``.

        Uses `NestedSamples.posterior_points` to
        resample the nested dead-point collection to a set of truly equal-weight
        samples (rows duplicated proportional to integer weights).

        Returns:
            Dict with keys ``"samples"`` (shape ``(n, n_dims)``) and
            ``"log_likelihood"`` (shape ``(n,)``).
        """
        if not self._sampled:
            raise RuntimeError("get_samples() called before sample()")
        posterior = self._nested_samples.posterior_points()
        samples = np.asarray(posterior.iloc[:, : self.n_dims])
        log_L = np.asarray(posterior["logL"])
        return {"samples": samples, "log_likelihood": log_L}

    def get_weighted_samples(self) -> dict[str, np.ndarray]:
        """Return the original nested points and normalized posterior log-weights.

        Unlike [`get_samples`][jimgw.samplers.blackjax.nss.BlackJAXNSSSampler.get_samples],
        this method performs no posterior resampling.  The weights are the
        nested-sampling quadrature weights from `NestedSamples.logw`, normalized
        in log space using 64-bit arithmetic.

        Returns:
            Dict with ``"samples"`` (shape ``(n, n_dims)``), death-contour
            ``"log_likelihood"`` and birth-contour
            ``"log_likelihood_birth"`` (both shape ``(n,)``), and normalized
            ``"log_weights"`` (shape ``(n,)``). All arrays retain the original
            nested-point ordering.
        """
        if not self._sampled:
            raise RuntimeError("get_weighted_samples() called before sample()")

        samples = np.asarray(self._nested_samples.iloc[:, : self.n_dims])
        log_likelihood = np.asarray(self._nested_samples["logL"])
        log_likelihood_birth = np.asarray(self._nested_samples["logL_birth"])
        log_weights = np.asarray(self._nested_samples.logw(), dtype=np.float64)
        log_weights = log_weights - logsumexp(log_weights)
        return {
            "samples": samples,
            "log_likelihood": log_likelihood,
            "log_likelihood_birth": log_likelihood_birth,
            "log_weights": log_weights,
        }

    def _get_diagnostics(self) -> dict[str, Any]:
        """Return NSS run diagnostics.

        Returns a dict with the following keys:

        * ``"n_likelihood_evaluations"`` — total likelihood calls.
        * ``"n_iterations"`` — total nested-sampling iterations.
        * ``"n_stepping_out_history"`` — stepping-out evaluations per iteration.
        * ``"n_shrinking_history"`` — shrinking evaluations per iteration.
        * ``"n_likelihood_evaluations_stepping_out"`` — total stepping-out evaluations.
        * ``"n_likelihood_evaluations_shrinking"`` — total shrinking evaluations.
        * ``"acceptance_history"`` — per-iteration acceptance flag.
        * ``"log_Z"`` — log Bayesian evidence (anesthetic mean estimate).
        * ``"log_Z_error"`` — standard deviation of log Z from 100 bootstrap samples.
        """
        if not self._sampled:
            raise RuntimeError("get_diagnostics() called before sample()")
        ui: Any = (
            self._final_state.update_info
        )  # SliceInfo — blackjax stubs type this as base NamedTuple
        total_steps = int(jnp.sum(ui.num_expansions))
        total_shrink = int(jnp.sum(ui.num_shrink))

        log_Z = np.asarray(self._nested_samples.logZ()).item()
        log_Z_error = np.std(np.asarray(self._nested_samples.logZ(nsamples=100))).item()

        return {
            "n_likelihood_evaluations": total_steps + total_shrink,
            "n_iterations": self._n_iterations,
            "n_stepping_out_history": np.asarray(ui.num_expansions),
            "n_shrinking_history": np.asarray(ui.num_shrink),
            "n_likelihood_evaluations_stepping_out": total_steps,
            "n_likelihood_evaluations_shrinking": total_shrink,
            "acceptance_history": np.asarray(ui.is_accepted),
            "log_Z": log_Z,
            "log_Z_error": log_Z_error,
            "sample_phase_seconds": dict(getattr(self, "_phase_seconds", {})) or None,
        }
