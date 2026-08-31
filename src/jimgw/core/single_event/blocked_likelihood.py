"""Waveform-cache support for single-event likelihoods."""

from collections.abc import Sequence

from jimgw.core.single_event.likelihood import SingleEventLikelihood
from jimgw.core.transforms import (
    BijectiveTransform,
    ConditionalBijectiveTransform,
    NtoMTransform,
)


def _validate_parameter_blocks(
    blocks: Sequence[Sequence[str]],
    *,
    parameter_names: tuple[str, ...],
) -> None:
    """Validate that proposal blocks exactly cover valid sampling parameters."""
    flattened_names = [name for block in blocks for name in block]
    unknown_names = sorted(set(flattened_names) - set(parameter_names))
    missing_names = sorted(set(parameter_names) - set(flattened_names))
    if unknown_names:
        raise ValueError(
            "Block parameter(s) "
            f"{unknown_names} are not sampling parameters {parameter_names}."
        )
    if missing_names:
        raise ValueError(
            f"Parameter blocks do not cover sampling parameter(s) {missing_names}."
        )


def _validate_parameter_groups(
    groups: Sequence[Sequence[str]],
    *,
    parameter_names: tuple[str, ...],
) -> None:
    """Validate named proposal subsets without requiring a partition."""
    flattened_names = [name for group in groups for name in group]
    unknown_names = sorted(set(flattened_names) - set(parameter_names))
    if unknown_names:
        raise ValueError(
            "Proposal group parameter(s) "
            f"{unknown_names} are not sampling parameters {parameter_names}."
        )


def _build_rebuild_required_by_block(
    likelihood: SingleEventLikelihood,
    blocks: Sequence[Sequence[str]],
    *,
    parameter_names: tuple[str, ...],
    sample_transforms: Sequence[BijectiveTransform],
    likelihood_transforms: Sequence[NtoMTransform],
) -> dict[tuple[int, ...], bool]:
    """Map each proposal block to whether it requires rebuilding."""
    waveform_sampling_dependencies = _infer_waveform_sampling_dependencies(
        likelihood,
        parameter_names,
        sample_transforms,
        likelihood_transforms,
    )
    return {
        tuple(parameter_names.index(name) for name in block): bool(
            set(block).intersection(waveform_sampling_dependencies)
        )
        for block in blocks
    }


def _resolve_rebuild_required_by_parameter_groups(
    likelihood: SingleEventLikelihood,
    groups: Sequence[Sequence[str]],
    *,
    parameter_names: tuple[str, ...],
    sample_transforms: Sequence[BijectiveTransform],
    likelihood_transforms: Sequence[NtoMTransform],
) -> tuple[tuple[tuple[int, ...], bool], ...]:
    """Resolve optional named proposal groups to indices and cache prices."""
    waveform_sampling_dependencies = _infer_waveform_sampling_dependencies(
        likelihood,
        parameter_names,
        sample_transforms,
        likelihood_transforms,
    )
    return tuple(
        (
            tuple(parameter_names.index(name) for name in group),
            bool(set(group).intersection(waveform_sampling_dependencies)),
        )
        for group in groups
    )


def _infer_waveform_sampling_dependencies(
    likelihood: SingleEventLikelihood,
    parameter_names: tuple[str, ...],
    sample_transforms: Sequence[BijectiveTransform],
    likelihood_transforms: Sequence[NtoMTransform],
) -> set[str]:
    """Map source-cache inputs back to sampling-space parameters.

    A likelihood may publish ``waveform_cache_dependency_parameter_names``
    when its reusable payload contains more than the waveform carrier.  The
    XG response path uses this contract for the intrinsic emission-time map.
    Older likelihoods and lightweight test doubles retain the waveform-only
    inference below.
    """
    all_sampling_parameters = set(parameter_names)
    dependencies = {name: {name} for name in parameter_names}

    for transforms, reverse in (
        (reversed(sample_transforms), True),
        (iter(likelihood_transforms), False),
    ):
        for transform in transforms:
            from_names, to_names = transform.name_mapping
            consumed_names = to_names if reverse else from_names
            produced_names = from_names if reverse else to_names
            conditional_names = (
                transform.conditional_names
                if isinstance(transform, ConditionalBijectiveTransform)
                else ()
            )
            input_names = (*consumed_names, *conditional_names)
            input_dependencies: set[str] = set()
            for name in input_names:
                input_dependencies.update(
                    dependencies.get(name, all_sampling_parameters)
                )
            for name in consumed_names:
                dependencies.pop(name, None)
            for name in produced_names:
                dependencies[name] = input_dependencies.copy()

    declared_dependency_names = getattr(
        likelihood, "waveform_cache_dependency_parameter_names", None
    )
    dependencies_are_explicit = declared_dependency_names is not None
    if declared_dependency_names is None:
        declared_cacheable_names = getattr(
            likelihood, "waveform_cacheable_parameter_names", None
        )
        if declared_cacheable_names is None:
            # Lightweight test doubles and third-party likelihoods may expose
            # the pre-validation duck-typed surface only. Real
            # SingleEventLikelihood instances use the validated property.
            cacheable_parameter_names: set[str] = set(
                getattr(likelihood.waveform, "cacheable_parameter_names", ())
            )
            if likelihood.waveform_caches_distance:
                cacheable_parameter_names.add("d_L")
        else:
            cacheable_parameter_names = set(declared_cacheable_names)
        cache_dependency_parameter_names = (
            set(likelihood.waveform.parameter_names) - cacheable_parameter_names
        )
    else:
        cache_dependency_parameter_names = set(declared_dependency_names)

    waveform_sampling_dependencies: set[str] = set()
    for parameter_name in cache_dependency_parameter_names:
        if parameter_name in likelihood.fixed_parameters:
            if callable(likelihood.fixed_parameters[parameter_name]):
                waveform_sampling_dependencies.update(all_sampling_parameters)
            continue
        sampling_dependencies = dependencies.get(parameter_name)
        if sampling_dependencies is None:
            if dependencies_are_explicit:
                raise ValueError(
                    "declared source-cache dependency "
                    f"{parameter_name!r} is not fixed, sampled, or produced by a "
                    "configured transform"
                )
            continue
        waveform_sampling_dependencies.update(sampling_dependencies)
    return waveform_sampling_dependencies
