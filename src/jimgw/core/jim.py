import logging
from collections.abc import Sequence
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key
from ripplegw.interfaces import Waveform

from jimgw._logging import ensure_logger_handler
from jimgw.core.base import LikelihoodBase
from jimgw.core.folding import (
    ResolvedFoldSymmetry,
    UnfoldedWeightedSamples,
    _fold_to_fundamental,
    _folded_log_likelihood_from_cache,
    _folded_log_prior,
    _in_fundamental_domain,
    _unfold_weighted_samples,
)
from jimgw.core.prior import Prior
from jimgw.core.single_event.blocked_likelihood import (
    _build_rebuild_required_by_block,
    _resolve_rebuild_required_by_parameter_groups,
    _validate_parameter_blocks,
    _validate_parameter_groups,
)
from jimgw.core.single_event.likelihood import SingleEventLikelihood
from jimgw.core.transforms import BijectiveTransform, NtoMTransform
from jimgw.samplers import FoldSymmetryConfig, Sampler, SamplerConfig, build_sampler
from jimgw.typing import FloatScalar

logger = logging.getLogger(__name__)

# Number of prior draws used to verify the posterior at construction time.
# More than half returning NaN is treated as a hard error; any non-zero count
# triggers a warning.
_NAN_TEST_POINTS = 10
_NAN_FAIL_THRESHOLD = 5

# Fixed key used for downsampling in get_samples.
_DOWNSAMPLE_KEY: Key = jax.random.key(42)


class Jim:
    """Master class for gravitational-wave parameter estimation.

    Wires together a [`LikelihoodBase`][jimgw.core.base.LikelihoodBase], a
    [`Prior`][jimgw.core.prior.Prior], optional parameter transforms, and a
    pluggable JAX [`Sampler`][jimgw.samplers.base.Sampler] selected via a typed
    ``sampler_config`` object.
    """

    likelihood: LikelihoodBase
    prior: Prior
    sample_transforms: Sequence[BijectiveTransform]
    likelihood_transforms: Sequence[NtoMTransform]
    sampling_parameter_names: tuple[str, ...]
    prior_parameter_names: tuple[str, ...]
    likelihood_parameter_names: tuple[str, ...]
    marginalized_parameter_names: tuple[str, ...]
    sampler: Sampler

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sampler_config: SamplerConfig,
        *,
        sample_transforms: Sequence[BijectiveTransform] = (),
        likelihood_transforms: Sequence[NtoMTransform] = (),
        periodic: Optional[list[str] | dict[str, tuple[float, float]]] = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        """Initialise Jim and build the internal sampler.

        Args:
            likelihood: The likelihood to evaluate.
            prior: The prior distribution.
            sampler_config: Pydantic config selecting and configuring the
                sampler backend (e.g. [`FlowMCConfig`][jimgw.samplers.config.FlowMCConfig]).
            sample_transforms: Bijective transforms applied in the sampling
                space (reversed when retrieving posterior samples).
            likelihood_transforms: Transforms applied to reach the likelihood
                parameter space from the prior parameter space.
            periodic: Periodic sampling-space parameters.  For most samplers,
                pass a ``dict`` mapping parameter name to ``(lo, hi)`` bounds
                (e.g. ``{"phase_c": (0.0, 6.2832)}``).  For the BlackJAX
                NS AW sampler (unit-cube space), pass a ``list`` of parameter
                names (bounds are implicit as ``[0, 1]``).
            seed: Integer random seed. The key for the sampling run is derived
                from this seed at construction time, so `sample` is
                reproducible regardless of any intermediate operations (sanity
                checks, initial-position draws, etc.).
            verbose: Enable DEBUG-level logging for all ``jimgw`` components.
                At ``False`` (default) INFO-level messages are always shown.
                Pass ``True`` to also see per-step diagnostics and
                backend-specific progress output (e.g. flowMC training loss).
        """
        if sampler_config.type == "flowmc":
            ensure_logger_handler("flowMC", logging.INFO)
        if verbose:
            logging.getLogger("jimgw").setLevel(logging.DEBUG)
            if sampler_config.type == "flowmc":
                logging.getLogger("flowMC").setLevel(logging.DEBUG)

        self.prior_parameter_names = prior.parameter_names
        self.sampling_parameter_names = self.prior_parameter_names
        for transform in sample_transforms:
            self.sampling_parameter_names = transform.propagate_name(
                self.sampling_parameter_names
            )
        self.likelihood_parameter_names = self.prior_parameter_names
        for transform in likelihood_transforms:
            self.likelihood_parameter_names = transform.propagate_name(
                self.likelihood_parameter_names
            )
        if isinstance(likelihood, SingleEventLikelihood):
            self.marginalized_parameter_names = tuple(
                parameter_name
                for flag_name, parameter_name in (
                    ("time_marginalization", "t_c"),
                    ("phase_marginalization", "phase_c"),
                    ("distance_marginalization", "d_L"),
                )
                if getattr(likelihood, flag_name, False)
            )
        else:
            self.marginalized_parameter_names = ()

        self._validate_problem(
            likelihood,
            prior,
            sample_transforms,
            likelihood_transforms,
            sampler_config,
        )
        self._validate_normalized_prior(prior, sampler_config)
        self._setup_problem(
            likelihood,
            prior,
            sample_transforms,
            likelihood_transforms,
            sampler_config,
        )
        root_key: Key = jax.random.key(seed)

        # Reserve _sampler_key immediately so sampling is reproducible even if
        # sanity checks or other internal splits consume _rng_key first.
        self._rng_key, self._sampler_key = jax.random.split(root_key)
        self._sampler_config = sampler_config

        # Resolve periodic parameter names → dimension indices.
        if periodic:
            names = self.sampling_parameter_names

            unknown = [name for name in periodic if name not in names]
            if unknown:
                raise ValueError(
                    f"Periodic parameter(s) {unknown} not found in "
                    f"sampling parameters {self.sampling_parameter_names}."
                )

            if isinstance(periodic, list):
                # NS AW style: list[str] → list[int].
                if sampler_config.type != "blackjax-ns-aw":
                    raise ValueError(
                        "List-form periodic (names without bounds) is only supported for "
                        "the 'blackjax-ns-aw' sampler. For other samplers pass a dict "
                        "mapping parameter names to (lo, hi) bounds, e.g. "
                        '{"phase_c": (0.0, 6.2832)}.'
                    )
                periodic_resolved = [names.index(name) for name in periodic]
            elif isinstance(periodic, dict):
                # dict[str, (lo, hi)] → dict[int, (lo, hi)]
                periodic_resolved = {
                    names.index(name): bounds for name, bounds in periodic.items()
                }
        else:
            periodic_resolved = None

        if (
            sampler_config.type == "blackjax-swig"
            and sampler_config.fold_symmetry is not None
        ):
            self._setup_fold_symmetry(
                likelihood,
                sample_transforms,
                likelihood_transforms,
                sampler_config.fold_symmetry,
                periodic,
            )

        self.sampler = build_sampler(
            sampler_config,
            n_dims=len(self.sampling_parameter_names),
            log_prior_fn=self._log_prior_fn,
            log_likelihood_fn=self._log_likelihood_fn,
            log_posterior_fn=self._log_posterior_fn,
            periodic=periodic_resolved,
            **self._sampler_backend_kwargs,
        )
        self._verify_posterior()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_problem(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: Sequence[BijectiveTransform],
        likelihood_transforms: Sequence[NtoMTransform],
        sampler_config: SamplerConfig,
    ) -> None:
        """Validate that the prior and likelihood parameter spaces are compatible.

        Args:
            likelihood: The likelihood to evaluate.
            prior: The prior distribution.
            sample_transforms: Bijective transforms from prior space to sampling space.
            likelihood_transforms: Transforms from prior space to likelihood space.
            sampler_config: Configuration for the selected sampler.

        Raises:
            ValueError: If any transform (sample or likelihood) produces a parameter
                that already exists in the current parameter space but is not consumed
                by that same transform; if prior parameters overlap with fixed
                parameters; if analytically marginalized parameters appear in the
                prior, sampling, or likelihood parameter spaces; if prior parameters
                are not consumed by the likelihood; or if the likelihood requires
                parameters not provided by the prior or fixed_parameters.
            TypeError: If a cache-aware sampler is paired with a likelihood that
                does not support waveform caching.
        """
        if sampler_config.type == "blackjax-swig" and not isinstance(
            likelihood, SingleEventLikelihood
        ):
            raise TypeError(
                "The selected cache-aware sampler requires a waveform-cache-capable "
                "single-event likelihood."
            )

        current_sampling_names = self.prior_parameter_names
        for transform in sample_transforms:
            consumed = set(transform.name_mapping[0])
            produced = set(transform.name_mapping[1])
            overwritten = (produced & set(current_sampling_names)) - consumed
            if overwritten:
                raise ValueError(
                    f"Sample transform {transform!r} produces parameter(s) "
                    f"{sorted(overwritten)} that already exist in the parameter "
                    "space but are not consumed by this transform. Remove the "
                    "prior on these parameters or remove the conflicting transform."
                )
            current_sampling_names = transform.propagate_name(current_sampling_names)
        if isinstance(likelihood, SingleEventLikelihood):
            marginalized_names = set(self.marginalized_parameter_names)
            if sampler_config.type == "blackjax-swig":
                _validate_parameter_blocks(
                    sampler_config.blocks,
                    parameter_names=self.sampling_parameter_names,
                )
                _validate_parameter_groups(
                    [block.parameters for block in sampler_config.de_jump_blocks],
                    parameter_names=self.sampling_parameter_names,
                )
                _validate_parameter_groups(
                    sampler_config.bridge_blocks,
                    parameter_names=self.sampling_parameter_names,
                )
                if sampler_config.complementary_de_jump_block is not None:
                    _validate_parameter_groups(
                        [sampler_config.complementary_de_jump_block.parameters],
                        parameter_names=self.sampling_parameter_names,
                    )

            current_likelihood_names = self.prior_parameter_names
            for transform in likelihood_transforms:
                consumed = set(transform.name_mapping[0])
                produced = set(transform.name_mapping[1])
                overwritten = (produced & set(current_likelihood_names)) - consumed
                if overwritten:
                    raise ValueError(
                        f"Likelihood transform {transform!r} produces parameter(s) "
                        f"{sorted(overwritten)} that already exist in the parameter "
                        "space but are not consumed by this transform. Remove the "
                        "prior on these parameters or remove the conflicting transform."
                    )
                current_likelihood_names = transform.propagate_name(
                    current_likelihood_names
                )

            invalid_marginalized_names = sorted(
                marginalized_names
                & (
                    set(self.prior_parameter_names)
                    | set(self.sampling_parameter_names)
                    | set(self.likelihood_parameter_names)
                )
            )
            if invalid_marginalized_names:
                raise ValueError(
                    f"Marginalized parameter(s) {invalid_marginalized_names} "
                    "must not appear in the prior, sampling, or likelihood "
                    "parameter spaces."
                )

            if likelihood.fixed_parameters:
                overlap = set(self.likelihood_parameter_names) & set(
                    likelihood.fixed_parameters.keys()
                )
                if overlap:
                    raise ValueError(
                        f"Prior defines parameter(s) {sorted(overlap)} that are "
                        "also in fixed_parameters. Either remove them from the prior "
                        "or from fixed_parameters."
                    )

            # Waveforms that publish a `parameter_names` attribute can be
            # cross-checked against the prior.
            wf_param_names = getattr(likelihood.waveform, "parameter_names", None)
            if isinstance(likelihood.waveform, Waveform) and wf_param_names is not None:
                consumed: set[str] = set(wf_param_names)
                consumed |= {"ra", "dec", "psi", "t_c"}
                consumed -= marginalized_names

                provided = set(self.likelihood_parameter_names)
                if likelihood.fixed_parameters:
                    provided |= set(likelihood.fixed_parameters.keys())

                unused = provided - consumed
                if unused:
                    raise ValueError(
                        f"Prior defines parameter(s) {sorted(unused)} that are not "
                        "consumed by the likelihood. Remove them from the prior or "
                        "add appropriate likelihood_transforms."
                    )
                missing = consumed - provided
                if missing:
                    raise ValueError(
                        f"Likelihood requires parameter(s) {sorted(missing)} that are "
                        "not provided by the prior or fixed_parameters. Add them to "
                        "the prior or to fixed_parameters."
                    )

    def _setup_problem(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: Sequence[BijectiveTransform],
        likelihood_transforms: Sequence[NtoMTransform],
        sampler_config: SamplerConfig,
    ) -> None:
        """Wire together the likelihood, prior, and transforms.

        Constructs ``_log_prior_fn``, ``_log_likelihood_fn``, and
        ``_log_posterior_fn`` — flat-array callables used by the sampler.
        Validation is performed separately by ``_validate_problem`` before this
        method is called.

        Args:
            likelihood: The likelihood to evaluate.
            prior: The prior distribution.
            sample_transforms: Bijective transforms from prior space to sampling space.
            likelihood_transforms: Transforms from prior space to likelihood space.
            sampler_config: Configuration for the selected sampler.
        """
        self.likelihood = likelihood
        self.prior = prior
        self.sample_transforms = sample_transforms
        self.likelihood_transforms = likelihood_transforms

        if not sample_transforms:
            logger.info(
                "No sample transforms provided. Using prior parameters as sampling parameters."
            )
        else:
            logger.info("Using sample transforms.")
            logger.debug(f"Sampling parameter names = {self.sampling_parameter_names}")

        if not likelihood_transforms:
            logger.info(
                "No likelihood transforms provided. Using prior parameters as likelihood parameters."
            )
        else:
            logger.debug(
                f"Using {len(likelihood_transforms)} likelihood transform(s): {[type(t).__name__ for t in likelihood_transforms]}"
            )

        # Build sampling-space callables. These operate on flat arrays of shape
        # (n_dims,) and are injected into the sampler.
        names = self.sampling_parameter_names

        def _sampling_array_to_likelihood_parameters(
            arr: Float[Array, " n_dims"],
        ) -> dict[str, Float]:
            named = dict(zip(names, arr, strict=True))
            for transform in reversed(sample_transforms):
                named, _ = transform.inverse(named)
            for transform in likelihood_transforms:
                named = transform.forward(named)
            return named

        def _log_prior_fn(arr: Float[Array, " n_dims"]) -> FloatScalar:
            named = dict(zip(names, arr, strict=True))
            jac: FloatScalar = jnp.zeros(())
            for transform in reversed(sample_transforms):
                named, j = transform.inverse(named)
                jac += j
            return prior.log_prob(named) + jac

        def _log_likelihood_fn(arr: Float[Array, " n_dims"]) -> FloatScalar:
            named = _sampling_array_to_likelihood_parameters(arr)
            return likelihood.evaluate(named)

        def _log_posterior_fn(arr: Float[Array, " n_dims"]) -> FloatScalar:
            named = dict(zip(names, arr, strict=True))
            jac: FloatScalar = jnp.zeros(())
            for transform in reversed(sample_transforms):
                named, j = transform.inverse(named)
                jac = jac + j
            log_prior = prior.log_prob(named) + jac
            for transform in likelihood_transforms:
                named = transform.forward(named)
            return likelihood.evaluate(named) + log_prior

        self._log_prior_fn = _log_prior_fn
        self._log_likelihood_fn = _log_likelihood_fn
        self._log_posterior_fn = _log_posterior_fn
        self._sampler_backend_kwargs: dict[str, Any] = {}

        if sampler_config.type == "blackjax-swig":
            # `_validate_problem` has already established the waveform-cache API.
            assert isinstance(likelihood, SingleEventLikelihood)
            self._rebuild_required_by_block = _build_rebuild_required_by_block(
                likelihood,
                sampler_config.blocks,
                parameter_names=self.sampling_parameter_names,
                sample_transforms=sample_transforms,
                likelihood_transforms=likelihood_transforms,
            )
            resolved_bridge_blocks = _resolve_rebuild_required_by_parameter_groups(
                likelihood,
                sampler_config.bridge_blocks,
                parameter_names=self.sampling_parameter_names,
                sample_transforms=sample_transforms,
                likelihood_transforms=likelihood_transforms,
            )
            rebuild_bridge_blocks = [
                block
                for block, (_, requires_rebuild) in zip(
                    sampler_config.bridge_blocks,
                    resolved_bridge_blocks,
                    strict=True,
                )
                if requires_rebuild
            ]
            if rebuild_bridge_blocks:
                raise ValueError(
                    "Bridge blocks must be cache-resident; waveform rebuilds are "
                    f"required by {rebuild_bridge_blocks}."
                )
            resolved_de_jump_blocks = _resolve_rebuild_required_by_parameter_groups(
                likelihood,
                [block.parameters for block in sampler_config.de_jump_blocks],
                parameter_names=self.sampling_parameter_names,
                sample_transforms=sample_transforms,
                likelihood_transforms=likelihood_transforms,
            )
            complementary_de_blocks = (
                [sampler_config.complementary_de_jump_block.parameters]
                if sampler_config.complementary_de_jump_block is not None
                else []
            )
            resolved_complementary_de_blocks = (
                _resolve_rebuild_required_by_parameter_groups(
                    likelihood,
                    complementary_de_blocks,
                    parameter_names=self.sampling_parameter_names,
                    sample_transforms=sample_transforms,
                    likelihood_transforms=likelihood_transforms,
                )
            )

            def build_cache(position):
                return likelihood.generate_waveform(
                    _sampling_array_to_likelihood_parameters(position)
                )

            def log_likelihood_from_cache_fn(position, cache):
                return likelihood.evaluate_from_waveform(
                    _sampling_array_to_likelihood_parameters(position), cache
                )

            self._sampler_backend_kwargs = {
                "rebuild_required_by_block": self._rebuild_required_by_block,
                "build_cache": build_cache,
                "log_likelihood_from_cache_fn": log_likelihood_from_cache_fn,
            }
            if sampler_config.scalar_extrinsic_cache:
                from types import SimpleNamespace

                from jimgw.core.single_event.dominant_mode import (
                    DominantModeTimeCachedWaveform,
                )
                from jimgw.core.single_event.heterodyne_extrinsics import (
                    evaluate_extrinsic_summary,
                )
                from jimgw.core.single_event.likelihood import (
                    HeterodynedTransientLikelihoodFD,
                )

                if (
                    not isinstance(likelihood, HeterodynedTransientLikelihoodFD)
                    or likelihood.interpolation_order < 2
                    or likelihood.time_marginalization
                    or likelihood.distance_marginalization
                    or sampler_config.scheduler != "fsm"
                    or sampler_config.fold_symmetry is not None
                ):
                    raise ValueError(
                        "scalar_extrinsic_cache requires an unfolded polynomial "
                        "FSM likelihood without time/distance marginalization"
                    )
                if not isinstance(likelihood.waveform, DominantModeTimeCachedWaveform):
                    DominantModeTimeCachedWaveform(likelihood.waveform)
                if not {"iota", "d_L"} <= likelihood.waveform_cacheable_parameter_names:
                    raise ValueError(
                        "scalar extrinsic cache requires cached inclination/distance"
                    )
                # Infer dependencies through the actual sampling transforms.
                # A hit that moves sky/time or a coupled intrinsic is unsafe.
                contract = SimpleNamespace(
                    waveform_cache_dependency_parameter_names=(
                        (
                            set(self.likelihood_parameter_names)
                            | set(likelihood.fixed_parameters)
                        )
                        - {"psi", "iota", "d_L"}
                    ),
                    fixed_parameters=likelihood.fixed_parameters,
                )
                groups = (
                    list(sampler_config.blocks)
                    + list(sampler_config.bridge_blocks)
                    + [block.parameters for block in sampler_config.de_jump_blocks]
                )
                scalar_groups = _resolve_rebuild_required_by_parameter_groups(
                    contract,
                    groups,
                    parameter_names=self.sampling_parameter_names,
                    sample_transforms=sample_transforms,
                    likelihood_transforms=likelihood_transforms,
                )
                waveform_groups = _resolve_rebuild_required_by_parameter_groups(
                    likelihood,
                    groups,
                    parameter_names=self.sampling_parameter_names,
                    sample_transforms=sample_transforms,
                    likelihood_transforms=likelihood_transforms,
                )
                if any(
                    slow and not rebuild
                    for (_, slow), (_, rebuild) in zip(
                        scalar_groups, waveform_groups, strict=True
                    )
                ):
                    raise ValueError(
                        "scalar_extrinsic_cache hit groups may vary only psi, iota "
                        "and d_L after applying sampling transforms"
                    )

                def prepare_hit_summary(position, cache):
                    return likelihood.build_extrinsic_summary(
                        _sampling_array_to_likelihood_parameters(position),
                        cache,
                    )

                def log_likelihood_from_hit_summary(position, summary):
                    return evaluate_extrinsic_summary(
                        likelihood,
                        _sampling_array_to_likelihood_parameters(position),
                        summary,
                    )

                self._sampler_backend_kwargs.update(
                    prepare_hit_summary=prepare_hit_summary,
                    log_likelihood_from_hit_summary=log_likelihood_from_hit_summary,
                    # The validated polynomial adapter's rebuild callback
                    # constructs its cache solely from the supplied position.
                    # No intermediate accepted cache is consumed inside R.
                    cache_independent_rebuild=True,
                )
            if resolved_bridge_blocks:
                self._sampler_backend_kwargs["resolved_bridge_blocks"] = (
                    resolved_bridge_blocks
                )
            if resolved_de_jump_blocks:
                self._sampler_backend_kwargs["resolved_de_jump_blocks"] = tuple(
                    (indices, requires_rebuild, block.attempts)
                    for (indices, requires_rebuild), block in zip(
                        resolved_de_jump_blocks,
                        sampler_config.de_jump_blocks,
                        strict=True,
                    )
                )
            if resolved_complementary_de_blocks:
                assert sampler_config.complementary_de_jump_block is not None
                indices, requires_rebuild = resolved_complementary_de_blocks[0]
                self._sampler_backend_kwargs["resolved_complementary_de_jump_block"] = (
                    indices,
                    requires_rebuild,
                    sampler_config.complementary_de_jump_block.attempts,
                )

    def _setup_fold_symmetry(
        self,
        likelihood: SingleEventLikelihood,
        sample_transforms: Sequence[BijectiveTransform],
        likelihood_transforms: Sequence[NtoMTransform],
        fold_config: FoldSymmetryConfig,
        periodic: Optional[list[str] | dict[str, tuple[float, float]]],
    ) -> None:
        """Validate and install the cache-aware eight-image quotient target."""

        fold_names = (fold_config.cos_iota, fold_config.azimuth, fold_config.psi)
        unknown_names = [
            name for name in fold_names if name not in self.sampling_parameter_names
        ]
        if unknown_names:
            raise ValueError(
                f"Fold symmetry parameter(s) {unknown_names} are not sampling "
                f"parameters {self.sampling_parameter_names}."
            )

        if not likelihood.phase_marginalization:
            raise ValueError("fold_symmetry requires active phase marginalization")

        detectors = tuple(likelihood.detectors)
        if len(detectors) != 3:
            raise ValueError(
                "fold_symmetry requires exactly three detector sites "
                f"(got {len(detectors)})"
            )
        try:
            vertices = np.stack(
                tuple(
                    np.asarray(detector.vertex, dtype=float) for detector in detectors
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "fold_symmetry detector vertices must each be finite 3-vectors"
            ) from exc
        if vertices.shape != (3, 3):
            raise ValueError(
                "fold_symmetry detector vertices must each be finite 3-vectors"
            )
        if not np.all(np.isfinite(vertices)):
            raise ValueError("fold_symmetry detector vertices must be finite")
        if np.unique(vertices, axis=0).shape[0] != 3:
            raise ValueError("fold_symmetry detector vertices must be distinct")
        baselines = vertices[1:] - vertices[0]
        if np.linalg.matrix_rank(baselines) != 2:
            raise ValueError("fold_symmetry detector vertices must not be collinear")

        if periodic is None or isinstance(periodic, list):
            raise ValueError(
                "fold_symmetry requires explicit periodic bounds for azimuth and psi"
            )
        expected_periods = {
            fold_config.azimuth: (0.0, 2.0 * np.pi),
            fold_config.psi: (0.0, np.pi),
        }
        for name, expected_bounds in expected_periods.items():
            bounds = periodic.get(name)
            try:
                actual_bounds = tuple(float(value) for value in bounds)  # type: ignore[union-attr]
            except (TypeError, ValueError):
                actual_bounds = ()
            if actual_bounds != expected_bounds:
                raise ValueError(
                    f"fold_symmetry requires periodic {name!r} bounds "
                    f"{expected_bounds}, got {bounds!r}"
                )

        resolved_fold_group = _resolve_rebuild_required_by_parameter_groups(
            likelihood,
            [list(fold_names)],
            parameter_names=self.sampling_parameter_names,
            sample_transforms=sample_transforms,
            likelihood_transforms=likelihood_transforms,
        )
        _, requires_rebuild = resolved_fold_group[0]
        if requires_rebuild:
            resolved_coordinates = _resolve_rebuild_required_by_parameter_groups(
                likelihood,
                [[name] for name in fold_names],
                parameter_names=self.sampling_parameter_names,
                sample_transforms=sample_transforms,
                likelihood_transforms=likelihood_transforms,
            )
            rebuild_names = [
                name
                for name, (_, coordinate_requires_rebuild) in zip(
                    fold_names, resolved_coordinates, strict=True
                )
                if coordinate_requires_rebuild
            ]
            raise ValueError(
                "Fold symmetry coordinates must be cache-resident; waveform "
                f"rebuilds are required by {rebuild_names}."
            )

        fold = ResolvedFoldSymmetry(
            indices=tuple(
                self.sampling_parameter_names.index(name) for name in fold_names
            ),
            azimuth_reflection_center=float(fold_config.azimuth_reflection_center),
        )
        self._resolved_fold_symmetry = fold
        self._base_log_prior_fn = self._log_prior_fn
        self._base_log_likelihood_fn = self._log_likelihood_fn
        self._base_log_posterior_fn = self._log_posterior_fn
        self._base_build_cache = self._sampler_backend_kwargs["build_cache"]
        self._base_log_likelihood_from_cache_fn = self._sampler_backend_kwargs[
            "log_likelihood_from_cache_fn"
        ]

        base_log_prior_fn = self._base_log_prior_fn
        base_build_cache = self._base_build_cache
        base_log_likelihood_from_cache_fn = self._base_log_likelihood_from_cache_fn

        def folded_log_prior(position):
            return _folded_log_prior(position, fold, base_log_prior_fn)

        def folded_log_likelihood_from_cache(position, cache):
            return _folded_log_likelihood_from_cache(
                position,
                cache,
                fold,
                base_log_prior_fn,
                base_log_likelihood_from_cache_fn,
            )

        def folded_log_likelihood(position):
            cache = base_build_cache(position)
            return folded_log_likelihood_from_cache(position, cache)

        def folded_log_posterior(position):
            return folded_log_prior(position) + folded_log_likelihood(position)

        self._log_prior_fn = folded_log_prior
        self._log_likelihood_fn = folded_log_likelihood
        self._log_posterior_fn = folded_log_posterior
        self._sampler_backend_kwargs["log_likelihood_from_cache_fn"] = (
            folded_log_likelihood_from_cache
        )

    def _verify_posterior(self) -> None:
        """Draw test points from the prior and verify the posterior is not mostly NaN.

        Raises:
            ValueError: If more than ``_NAN_FAIL_THRESHOLD`` out of
                ``_NAN_TEST_POINTS`` test points return NaN posterior values.
        """
        self._rng_key, check_key = jax.random.split(self._rng_key)
        check_positions = self._draw_initial_positions(check_key, _NAN_TEST_POINTS)
        log_posteriors = jax.vmap(self._log_posterior_fn)(check_positions)
        n_nan = int(jnp.sum(jnp.isnan(log_posteriors)))
        if n_nan > _NAN_FAIL_THRESHOLD:
            raise ValueError(
                f"The posterior returned NaN for {n_nan}/{_NAN_TEST_POINTS} test "
                "points sampled from the prior. Check your likelihood and "
                "transforms for correctness."
            )
        elif n_nan > 0:
            logger.warning(
                "%d/%d test points sampled from the prior returned NaN posterior "
                "values. This may indicate issues at the boundaries of your prior.",
                n_nan,
                _NAN_TEST_POINTS,
            )

    def _validate_normalized_prior(
        self, prior: Prior, sampler_config: SamplerConfig
    ) -> None:
        """Raise if a normalization-requiring sampler is paired with an unnormalized prior.

        Args:
            prior: The prior to check.
            sampler_config: The sampler configuration to check against.

        Raises:
            ValueError: If an evidence-computing sampler is paired with an
                unnormalized prior.
        """
        if (
            sampler_config.type in ("blackjax-nss", "blackjax-swig", "blackjax-smc")
            and not prior.is_normalized
        ):
            raise ValueError(
                f"{type(sampler_config).__name__} computes Bayesian evidence and "
                "therefore requires a normalized prior (∫ exp(log_prob(x)) dx = 1). "
                "If your custom prior is normalized, override the is_normalized "
                "property to return True."
            )

    def _draw_initial_positions(self, key: Key, n: int) -> Float[Array, "n n_dims"]:
        """Sample ``n`` initial positions from the prior in sampling space.

        Args:
            key: JAX PRNG key.
            n: Number of positions to draw.

        Returns:
            Array of shape ``(n, n_dims)`` in sampling space.

        Raises:
            ValueError: If any drawn position contains non-finite values.
        """
        initial = self.prior.sample(key, n)
        for transform in self.sample_transforms:
            initial = jax.vmap(transform.forward)(initial)
        arr = jnp.array([initial[name] for name in self.sampling_parameter_names]).T
        if not jnp.all(jnp.isfinite(arr)):
            raise ValueError(
                "Initial positions contain non-finite values (NaN or inf). "
                "Check your priors and transforms for validity."
            )
        return self._canonicalize_fold_positions(arr)

    def _canonicalize_fold_positions(self, positions):
        """Map fold-enabled initial positions to the fundamental domain."""

        fold = getattr(self, "_resolved_fold_symmetry", None)
        if fold is None:
            return positions

        arr = jnp.asarray(positions)
        if arr.ndim == 1:
            canonical = _fold_to_fundamental(arr, fold)
            in_domain = _in_fundamental_domain(canonical, fold)
        elif arr.ndim == 2:
            canonical = jax.vmap(lambda point: _fold_to_fundamental(point, fold))(arr)
            in_domain = jax.vmap(lambda point: _in_fundamental_domain(point, fold))(
                canonical
            )
        else:
            raise ValueError(
                "Folded initial positions must have shape (n_dims,) or "
                "(n_positions, n_dims)."
            )
        if not bool(jnp.all(in_domain)):
            raise RuntimeError(
                "Fold canonicalization produced a position outside the "
                "fundamental domain."
            )
        return canonical

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_name(self, x: Float[Array, " n_dims"]) -> dict[str, Float]:
        """Convert a flat sampling-space array to a named dict."""
        return dict(zip(self.sampling_parameter_names, x, strict=True))

    def samples_to_prior_space(
        self,
        sample_array: Float[Array, "n_samples n_dims"] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Convert a batch of sampling-space positions to named prior-space arrays.

        Args:
            sample_array: Array of shape ``(n_samples, n_dims)`` in sampling space.

        Returns:
            Dict mapping prior parameter names to 1-D numpy arrays without
            changing the input row order.
        """
        named = jax.vmap(self.add_name)(jnp.array(sample_array))
        for transform in reversed(self.sample_transforms):
            named = jax.vmap(transform.backward)(named)
        return {name: np.array(named[name]) for name in self.prior_parameter_names}

    def evaluate_prior(self, params: Float[Array, " n_dims"]) -> Float:
        """Log-prior in the sampling space (with Jacobian corrections from sample_transforms)."""
        return self._log_prior_fn(params)

    def evaluate_posterior(self, params: Float[Array, " n_dims"]) -> Float:
        """Log-posterior in the sampling space."""
        return self._log_posterior_fn(params)

    def sample_initial_positions(
        self,
        n_points: int,
        rng_key: Optional[Key] = None,
    ) -> Float[Array, "n_points n_dims"]:
        """Draw ``n_points`` initial positions from the prior in sampling space.

        Args:
            n_points: Number of positions to draw.
            rng_key: Optional explicit PRNG key. If ``None``, Jim's internal
                auxiliary key is advanced automatically.

        Returns:
            Array of shape ``(n_points, n_dims)`` in sampling space.
        """
        if rng_key is None:
            self._rng_key, rng_key = jax.random.split(self._rng_key)
        return self._draw_initial_positions(rng_key, n_points)

    def sample(
        self,
        initial_position: Optional[Float[Array, "n_chains n_dims"]] = None,
    ) -> None:
        """Run the sampler.

        The sampling key is pre-reserved at construction time from ``seed``,
        so results are reproducible regardless of any calls made before this
        method (e.g. the construction-time posterior verification).

        Args:
            initial_position: Starting positions in sampling space, or
                ``None`` (default) to draw them from the prior. The
                expected shape depends on the backend:

                - flowMC: ``(n_chains, n_dims)`` or ``(n_dims,)`` (broadcast
                  to all chains).
                - BlackJAX NS AW / NSS: exactly ``(n_live, n_dims)``.
                - BlackJAX SMC: exactly ``(n_particles, n_dims)``.

                The concrete sampler validates the shape and raises
                ``ValueError`` on mismatch.
        """
        if initial_position is None:
            cfg = self._sampler_config
            counts = {
                attr: getattr(cfg, attr)
                for attr in ("n_chains", "n_live", "n_particles")
                if hasattr(cfg, attr)
            }
            if len(counts) != 1:
                raise TypeError(
                    f"Cannot determine number of initial positions from "
                    f"{type(cfg).__name__}: expected exactly one of n_chains, "
                    f"n_live, n_particles, found {list(counts)}"
                )
            n = next(iter(counts.values()))
            self._rng_key, init_key = jax.random.split(self._rng_key)
            initial_position = self._draw_initial_positions(init_key, n)
        elif hasattr(self, "_resolved_fold_symmetry"):
            initial_position = self._canonicalize_fold_positions(initial_position)
        self.sampler.sample(self._sampler_key, initial_position)

    def get_samples(
        self,
        n_samples: int = 0,
    ) -> dict[str, np.ndarray]:
        """Retrieve posterior samples in prior space, optionally downsampled.

        Calls [`Sampler.get_samples`][jimgw.samplers.base.Sampler.get_samples] on the
        underlying sampler, which returns equally-weighted posterior samples.
        Pass ``n_samples`` to further downsample.

        Args:
            n_samples: Target number of samples.  If 0 (default) returns all
            available samples, otherwise downsample uniformly without replacement.

        Returns:
            Dict mapping prior parameter names to 1-D numpy arrays in prior
            space, plus an extra item containing the log-likelihood values.
        """
        if hasattr(self, "_resolved_fold_symmetry"):
            raise NotImplementedError(
                "Folded equally weighted samples must be unfolded before conversion "
                "to prior space."
            )
        result = self.sampler.get_samples()
        sample_array = result["samples"]  # (n, n_dims) in sampling space
        log_likelihood = result["log_likelihood"]  # (n,)
        n_available = sample_array.shape[0]

        if n_samples > 0:
            if n_samples > n_available:
                logger.warning(
                    "Requested %d samples but only %d available. Returning all available samples.",
                    n_samples,
                    n_available,
                )
                n_samples = n_available
            if n_samples < n_available:
                indices = np.array(
                    jax.random.choice(
                        _DOWNSAMPLE_KEY, n_available, shape=(n_samples,), replace=False
                    )
                )
                sample_array = sample_array[indices]
                log_likelihood = log_likelihood[indices]

        out = self.samples_to_prior_space(sample_array)
        out["log_likelihood"] = np.asarray(log_likelihood)
        return out

    def get_weighted_samples(
        self,
        space: Literal["prior", "sampling"] = "prior",
    ) -> dict[str, np.ndarray]:
        """Retrieve original weighted posterior samples in the requested space.

        This is the non-resampling counterpart to [`get_samples`][jimgw.core.jim.Jim.get_samples]
        for sampler backends that retain their weighted posterior collection.
        By default, sample-space transforms are reversed exactly as they are for
        `get_samples`. Pass ``space="sampling"`` to retain the backend's raw
        sampling-space positions. The original nested-point ordering and aligned
        normalized log weights are preserved in either space.

        Args:
            space: Return named prior-space arrays (default) or the backend's raw
                ``"samples"`` array in sampling space.

        Returns:
            Dict mapping prior parameter names to 1-D numpy arrays, plus
            ``"log_likelihood"`` and ``"log_weights"``. Nested-sampling
            backends also provide the aligned ``"log_likelihood_birth"``
            field. The log weights are normalized such that
            ``scipy.special.logsumexp(log_weights) == 0``.

        Raises:
            ValueError: If ``space`` is not ``"prior"`` or ``"sampling"``.
            NotImplementedError: If the configured sampler does not expose
                weighted posterior samples.
        """
        if space not in ("prior", "sampling"):
            raise ValueError("space must be 'prior' or 'sampling'")
        if space == "prior" and hasattr(self, "_resolved_fold_symmetry"):
            raise NotImplementedError(
                "Folded weighted samples must be unfolded in sampling space before "
                "conversion to prior space."
            )

        result = self.sampler.get_weighted_samples()
        if space == "sampling":
            return {name: np.asarray(values) for name, values in result.items()}
        out = self.samples_to_prior_space(result["samples"])
        out["log_likelihood"] = np.asarray(result["log_likelihood"])
        if "log_likelihood_birth" in result:
            out["log_likelihood_birth"] = np.asarray(result["log_likelihood_birth"])
        out["log_weights"] = np.asarray(result["log_weights"])
        return out

    def unfold_weighted_samples(
        self,
        positions: Float[Array, "n_points n_dims"] | np.ndarray,
        log_weights: Float[Array, " n_points"] | np.ndarray,
        *,
        batch_size: Optional[int] = None,
    ) -> UnfoldedWeightedSamples:
        """Expand folded weighted points into their eight base-target images.

        The input positions must be the raw sampling-space output from
        ``get_weighted_samples(space="sampling")``. The returned image rows
        remain in sampling space; callers must reverse sample transforms only
        after this expansion.
        """

        if not hasattr(self, "_resolved_fold_symmetry"):
            raise RuntimeError(
                "unfold_weighted_samples requires an enabled fold_symmetry"
            )
        return _unfold_weighted_samples(
            positions,
            log_weights,
            self._resolved_fold_symmetry,
            self._base_log_prior_fn,
            self._base_log_likelihood_from_cache_fn,
            self._base_build_cache,
            batch_size=batch_size,
        )

    def get_diagnostics(self) -> dict[str, Any]:
        """Return run-level diagnostics from the most recent `sample` call.

        Returns:
            Plain dict of backend-specific diagnostics.
        """
        return self.sampler.get_diagnostics()
