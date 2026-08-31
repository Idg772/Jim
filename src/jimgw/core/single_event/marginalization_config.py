"""Pydantic configuration models for likelihood analytic marginalizations."""

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from jimgw.core.prior import Prior


class PhaseMargConfig(BaseModel):
    """Configuration sentinel for phase marginalization (no parameters)."""

    model_config = {"extra": "forbid"}


class TimeMargConfig(BaseModel):
    """Configuration for time marginalization."""

    model_config = {"extra": "forbid"}
    tc_range: tuple[float, float] = (-0.1, 0.1)
    upsample_factor: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_tc_range(self) -> "TimeMargConfig":
        lower, upper = self.tc_range
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("tc_range must contain finite increasing bounds")
        return self


class HeterodyneTimeMargConfig(TimeMargConfig):
    """Direct-sum time marginalization for relative binning.

    ``phasor_block_size`` bounds the temporary time-by-bin phasor array used by
    the compiled likelihood. ``freeze_response`` explicitly opts in to holding
    a response that declares time dependence fixed at ``t_c = 0`` while the
    carrier is shifted across the quadrature grid. Static detector responses do
    not require that approximation. ``timing_sigma_s`` records the smallest
    phase-profiled timing width in the frozen science envelope. When provided,
    construction requires at least ``samples_per_timing_sigma`` grid points per
    width instead of accepting an inherited upsample factor blindly.
    ``normalization='full_grid'`` preserves the historical full-segment time
    prior. ``normalization='window'`` uses equal-weight midpoint quadrature
    over ``tc_range`` as the complete uniform-prior support and is the required
    XG production convention.
    """

    phasor_block_size: int = Field(default=64, ge=1)
    freeze_response: bool = False
    timing_sigma_s: Optional[float] = Field(default=None, gt=0.0)
    samples_per_timing_sigma: int = Field(default=4, ge=2)
    normalization: Literal["full_grid", "window"] = "full_grid"

    @model_validator(mode="after")
    def _validate_timing_sigma(self) -> "HeterodyneTimeMargConfig":
        if self.timing_sigma_s is not None and not math.isfinite(self.timing_sigma_s):
            raise ValueError("timing_sigma_s must be finite")
        return self


class DistanceMargConfig(BaseModel):
    """Configuration for distance marginalization."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}
    distance_prior: Prior  # required — no default
    n_dist_points: int = Field(default=10000, ge=2)
    ref_dist: Optional[float] = Field(default=None, gt=0.0)
