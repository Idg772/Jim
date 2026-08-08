"""Pydantic configuration models for likelihood analytic marginalizations."""

from typing import Optional

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
    jitter_time: bool = False

    @model_validator(mode="after")
    def _validate_jitter_upsampling_exclusion(self) -> "TimeMargConfig":
        if self.jitter_time and self.upsample_factor > 1:
            raise ValueError("jitter_time cannot be combined with upsample_factor > 1")
        return self


class DistanceMargConfig(BaseModel):
    """Configuration for distance marginalization."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}
    distance_prior: Prior  # required — no default
    n_dist_points: int = Field(default=10000, ge=2)
    ref_dist: Optional[float] = Field(default=None, gt=0.0)
