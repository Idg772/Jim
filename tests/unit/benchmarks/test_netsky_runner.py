"""Focused NETSKY campaign-runner contract tests."""

import copy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign.run_injection import (
    _build_sampler_config,
    _detector_plane_azimuth,
)


@dataclass(frozen=True)
class _Site:
    vertex: np.ndarray


class _CurrentConfig:
    model_fields: ClassVar[dict[str, object]] = {
        name: object()
        for name in (
            "scheduler",
            "direction_mode",
            "num_de_jumps",
            "adaptive_slice_widths",
            "bracket_mode",
            "width_adaptation_rate",
            "width_target_expansions",
            "width_target_shrinks",
            "num_slice_steps_by_block",
            "bridge_blocks",
            "periodic_wrapped_covariance",
            "fold_symmetry",
        )
    }

    def __init__(self, **values: object) -> None:
        self.values = values


def _sites() -> list[_Site]:
    return [
        _Site(np.asarray([0.0, 0.0, 0.0])),
        _Site(np.asarray([3.0, 0.0, 0.0])),
        _Site(np.asarray([0.0, 4.0, 1.0])),
    ]


def _netsky_config() -> dict[str, object]:
    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config.update(
        {
            "blocking_scheme": common.NETSKY_SCHEME,
            "blocks": copy.deepcopy(common.NETSKY_BLOCKS),
            "bridge_blocks": copy.deepcopy(common.NETSKY_BRIDGE_BLOCKS),
            "periodic_wrapped_covariance": True,
            "num_slice_steps_by_block": [4, 3, 3, 1, 4],
            "fold_symmetry": {},
        }
    )
    return config


def test_netsky_sampler_config_completes_fold_geometry_and_passthrough() -> None:
    sites = _sites()

    built = _build_sampler_config(_netsky_config(), _CurrentConfig, ifos=sites)

    assert built.values["bridge_blocks"] == common.NETSKY_BRIDGE_BLOCKS
    assert built.values["periodic_wrapped_covariance"] is True
    assert built.values["num_slice_steps_by_block"] == [4, 3, 3, 1, 4]
    assert built.values["fold_symmetry"] == {
        "cos_iota": "cos_iota",
        "azimuth": "azimuth",
        "psi": "psi",
        "azimuth_reflection_center": pytest.approx(_detector_plane_azimuth(sites)),
    }


def test_fold_config_requires_current_api_and_detector_geometry() -> None:
    class _PinnedConfig:
        model_fields: ClassVar[dict[str, object]] = {"scheduler": object()}

        def __init__(self, **values: object) -> None:
            self.values = values

    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config["fold_symmetry"] = {}
    with pytest.raises(RuntimeError, match="fold_symmetry"):
        _build_sampler_config(config, _PinnedConfig, ifos=_sites())
    with pytest.raises(ValueError, match="detector"):
        _build_sampler_config(config, _CurrentConfig)
