import logging
import math
from collections.abc import Sequence

from jimgw.core.jim import Jim
from jimgw.core.transforms import BijectiveTransform, NtoMTransform

logger = logging.getLogger(__name__)

_CLI_CHECKPOINT_INTERVAL = 600.0  # 10 minutes

# Sampling coordinates that live on a circle, with their canonical support.
# A declared ``[prior]`` entry with explicit ``min``/``max`` overrides it.
_PERIODIC_SUPPORTS: dict[str, tuple[float, float]] = {
    "ra": (0.0, 2.0 * math.pi),
    "azimuth": (0.0, 2.0 * math.pi),
    "psi": (0.0, math.pi),
    "phase_c": (0.0, 2.0 * math.pi),
    "s1_phi": (0.0, 2.0 * math.pi),
    "s2_phi": (0.0, 2.0 * math.pi),
}


def infer_periodic_bounds(
    prior_cfg, sampling_parameter_names: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Return ``{name: (lo, hi)}`` for every periodic sampling coordinate.

    Bounds come from the declared prior support when the coordinate is a
    prior parameter and from the canonical circle otherwise (for example
    ``azimuth`` produced by the detector-frame sky transform).

    A prior parameter is periodic only when its declared support spans the
    full canonical period.  A restricted range (for example an RA window
    used as a referee run) is a plain bounded coordinate: wrapping it would
    teleport proposals across the window, so it is left out of the result.
    """
    specs = prior_cfg.root
    bounds: dict[str, tuple[float, float]] = {}
    for name in sampling_parameter_names:
        if name not in _PERIODIC_SUPPORTS:
            continue
        canonical_lo, canonical_hi = _PERIODIC_SUPPORTS[name]
        lo, hi = canonical_lo, canonical_hi
        spec = specs.get(name)
        if spec is not None and hasattr(spec, "min") and hasattr(spec, "max"):
            lo, hi = float(spec.min), float(spec.max)
        if not hi > lo:
            raise ValueError(f"Invalid periodic bounds for {name!r}: ({lo}, {hi})")
        period = canonical_hi - canonical_lo
        if not math.isclose(hi - lo, period, rel_tol=1e-9, abs_tol=1e-12):
            logger.info(
                "Prior for %s spans [%g, %g], not its full period %g; "
                "treating it as a bounded, non-periodic coordinate",
                name,
                lo,
                hi,
                period,
            )
            continue
        bounds[name] = (lo, hi)
    return bounds


def _with_checkpoint(sampler_config, output_dir):
    """Return a copy of *sampler_config* with CLI checkpoint defaults applied.

    Only fields the user did not explicitly set are filled in, so explicit
    values in the config file are always respected.  The CLI defaults are:
    checkpoint every ``_CLI_CHECKPOINT_INTERVAL`` seconds, writing to
    ``{output_dir}/checkpoint.pkl``.
    """
    explicitly_set = sampler_config.model_fields_set
    update = {}
    if "checkpoint_dir" not in explicitly_set:
        update["checkpoint_dir"] = output_dir
    if "checkpoint_interval" not in explicitly_set:
        update["checkpoint_interval"] = _CLI_CHECKPOINT_INTERVAL
    if not update:
        return sampler_config
    merged = sampler_config.model_dump() | update
    return sampler_config.__class__.model_validate(merged)


def build_jim(
    likelihood,
    prior,
    sample_transforms: Sequence[BijectiveTransform],
    likelihood_transforms: Sequence[NtoMTransform],
    cfg,
    verbose: bool = False,
):
    """Wire together Jim from the fully-built components."""
    sampler_config = _with_checkpoint(cfg.sampler, cfg.output.dir)
    periodic = None
    if getattr(cfg.sampler, "periodic_wrapped_covariance", False):
        # SwiG's wrapped covariance refuses to run without declared periodic
        # bounds; derive them from the sampling coordinates exactly as the
        # benchmark drivers do.
        sampling_names = tuple(prior.parameter_names)
        for transform in sample_transforms:
            sampling_names = tuple(transform.propagate_name(sampling_names))
        periodic = infer_periodic_bounds(cfg.prior, sampling_names)
        if not periodic:
            raise ValueError(
                "sampler.periodic_wrapped_covariance requires at least one "
                "periodic sampling parameter "
                f"({', '.join(sorted(_PERIODIC_SUPPORTS))}); none found in "
                f"{sampling_names}"
            )
    jim = Jim(
        likelihood=likelihood,
        prior=prior,
        sampler_config=sampler_config,
        sample_transforms=sample_transforms,
        likelihood_transforms=likelihood_transforms,
        periodic=periodic,
        seed=cfg.seed,
        verbose=verbose,
    )
    logger.info("Built Jim (sampler=%s, seed=%d)", cfg.sampler.type, cfg.seed)
    return jim
