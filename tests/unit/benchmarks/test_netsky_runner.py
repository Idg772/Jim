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
    _fold_projection_accounting,
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


def test_fold_projection_accounting_separates_sampler_and_unfold_work() -> None:
    accounting = _fold_projection_accounting(
        {"group_order": 8, "folded_points": 11},
        {"n_likelihood_evaluations_physical": 17},
    )

    assert accounting == {
        "images_per_folded_target_callback": 8,
        "sampler_folded_target_callbacks": 17,
        "sampler_true_image_projections": 136,
        "retained_folded_points_unfolded": 11,
        "unfold_true_image_projections": 88,
        "total_true_image_projections": 224,
        "sampler_callback_counter": ("diagnostics.n_likelihood_evaluations_physical"),
    }


@pytest.mark.parametrize(
    ("telemetry", "diagnostics"),
    [
        (
            {"group_order": 7, "folded_points": 11},
            {"n_likelihood_evaluations_physical": 17},
        ),
        (
            {"group_order": 8, "folded_points": -1},
            {"n_likelihood_evaluations_physical": 17},
        ),
        (
            {"group_order": 8, "folded_points": 11},
            {"n_likelihood_evaluations_physical": None},
        ),
    ],
)
def test_fold_projection_accounting_rejects_invalid_counters(
    telemetry: dict[str, object],
    diagnostics: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="projection accounting"):
        _fold_projection_accounting(telemetry, diagnostics)
