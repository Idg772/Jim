"""Bounded schedule weights for the XG candidate callback timing proxy."""

from types import SimpleNamespace

from jimgw.cli._prior import build_prior
from jimgw.cli._transforms import (
    infer_likelihood_transforms,
    infer_sample_transforms,
)
from jimgw.core.single_event.blocked_likelihood import (
    _build_rebuild_required_by_block,
    _resolve_rebuild_required_by_parameter_groups,
    _validate_parameter_blocks,
)
from jimgw.samplers.blackjax.swig import _resolve_num_slice_steps_by_block


def network_callback_weights(cfg, fine) -> tuple[float, float, float]:
    """Return scheduled (rebuild, scalar-summary, scalar-hit) weights.

    Resolve the same transformed block dependencies and slice counts as Jim.
    Each contiguous cache-hit segment prepares one scalar summary, including
    segments joined across sweep boundaries. This is a callback timing proxy,
    not a prediction of total FSM runtime: stepping/shrink multiplicities,
    inactive-lane work and the entry waveform cache are excluded.

    Only metadata and detector geometry are used; no Jim, native data, sampler
    transition or compiled likelihood is constructed here.
    """
    sampler = cfg.sampler
    if sampler.type != "blackjax-swig" or sampler.scheduler != "fsm":
        raise ValueError("XG callback timing requires blackjax-swig scheduler='fsm'")
    if not sampler.scalar_extrinsic_cache:
        raise ValueError("XG callback timing requires scalar_extrinsic_cache")
    unsupported = {
        "bridge_blocks": bool(sampler.bridge_blocks),
        "num_de_jumps": bool(sampler.num_de_jumps),
        "de_jump_blocks": bool(sampler.de_jump_blocks),
        "complementary_de_jump_block": sampler.complementary_de_jump_block is not None,
        "fold_symmetry": sampler.fold_symmetry is not None,
        "likelihood_screen_fraction": bool(
            getattr(sampler, "likelihood_screen_fraction", 0.0)
        ),
        "block_kernel_modes": any(
            mode != "slice" for mode in (sampler.block_kernel_modes or ())
        ),
    }
    active = [name for name, enabled in unsupported.items() if enabled]
    if active:
        raise ValueError(
            "XG callback timing supports ordinary slice blocks only; unsupported "
            + ", ".join(active)
        )

    prior_names = tuple(build_prior(cfg.prior).parameter_names)
    prior_params = frozenset(prior_names)
    sample_transforms = infer_sample_transforms(
        prior_params,
        cfg.data.trigger_time,
        fine.detectors,
        cfg.sampling,
        prior_cfg=cfg.prior,
    )
    likelihood_transforms = infer_likelihood_transforms(
        prior_params,
        cfg.data.trigger_time,
        fine.detectors,
        cfg.sampling,
        cfg.waveform.f_ref,
        cfg.likelihood.phase_marginalization,
    )
    sampling_names = prior_names
    for transform in sample_transforms:
        sampling_names = transform.propagate_name(sampling_names)
    likelihood_names = prior_names
    for transform in likelihood_transforms:
        likelihood_names = transform.propagate_name(likelihood_names)
    _validate_parameter_blocks(sampler.blocks, parameter_names=sampling_names)
    resolver_kwargs = {
        "parameter_names": sampling_names,
        "sample_transforms": sample_transforms,
        "likelihood_transforms": likelihood_transforms,
    }
    rebuild_by_block = _build_rebuild_required_by_block(
        fine, sampler.blocks, **resolver_kwargs
    )
    scalar_contract = SimpleNamespace(
        waveform_cache_dependency_parameter_names=(
            (set(likelihood_names) | set(fine.fixed_parameters))
            - {"psi", "iota", "d_L"}
        ),
        fixed_parameters=fine.fixed_parameters,
    )
    scalar_groups = _resolve_rebuild_required_by_parameter_groups(
        scalar_contract, sampler.blocks, **resolver_kwargs
    )
    rebuild_flags = tuple(rebuild_by_block.values())
    if any(
        slow and not rebuild
        for (_, slow), rebuild in zip(scalar_groups, rebuild_flags, strict=True)
    ):
        raise ValueError(
            "scalar_extrinsic_cache hit groups may vary only psi, iota and d_L "
            "after applying sampling transforms"
        )
    counts = _resolve_num_slice_steps_by_block(
        rebuild_by_block,
        sampler.num_inner_steps_per_dim,
        sampler.num_slice_steps_by_block,
    )
    sweeps = sampler.num_gibbs_sweeps
    rebuilds = sweeps * sum(
        count for count, rebuild in zip(counts, rebuild_flags, strict=True) if rebuild
    )
    hits = sweeps * sum(
        count
        for count, rebuild in zip(counts, rebuild_flags, strict=True)
        if not rebuild
    )
    summary_starts = sum(
        not rebuild and (index == 0 or rebuild_flags[index - 1])
        for index, rebuild in enumerate(rebuild_flags)
    )
    summaries = sweeps * summary_starts
    if rebuild_flags and not rebuild_flags[0] and not rebuild_flags[-1]:
        summaries -= sweeps - 1
    return float(rebuilds), float(summaries), float(hits)
