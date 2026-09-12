"""Configured callback weights without constructing data or a sampler."""

from types import SimpleNamespace

import pytest

from jimgw.cli._config import PriorConfig, SamplingConfig
from jimgw.cli._xg_timing import network_callback_weights
from jimgw.core.single_event.detector import get_CE_A, get_ET_Sardinia
from jimgw.samplers.config import BlackJAXSwiGConfig
from tests.xg_fixtures import network_config


@pytest.fixture
def setup():
    raw = network_config()
    cfg = SimpleNamespace(
        sampler=BlackJAXSwiGConfig.model_validate(raw["sampler"]),
        prior=PriorConfig.model_validate(raw["prior"]),
        sampling=SamplingConfig.model_validate(raw["sampling"]),
        data=SimpleNamespace(trigger_time=raw["data"]["trigger_time"]),
        waveform=SimpleNamespace(f_ref=raw["waveform"]["f_ref"]),
        likelihood=SimpleNamespace(
            phase_marginalization=raw["likelihood"]["phase_marginalization"]
        ),
    )
    # These detectors have geometry only: no strain, PSD or native moments.
    fine = SimpleNamespace(
        detectors=[get_CE_A(), *get_ET_Sardinia()],
        fixed_parameters={"phase_c": 0.0},
        waveform_cache_dependency_parameter_names=frozenset(
            {"M_c", "eta", "s1_z", "s2_z", "lambda_1", "lambda_2"}
        ),
    )
    return cfg, fine


def test_actual_network_defaults(setup):
    cfg, fine = setup
    assert network_callback_weights(cfg, fine) == (72.0, 8.0, 24.0)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"num_slice_steps_by_block": [2, 4, 1, 5]}, (48.0, 8.0, 48.0)),
        ({"num_inner_steps_per_dim": 2}, (144.0, 8.0, 48.0)),
        ({"num_gibbs_sweeps": 1}, (9.0, 1.0, 3.0)),
        ({"num_gibbs_sweeps": 1000000}, (9000000.0, 1000000.0, 3000000.0)),
    ],
)
def test_configured_counts_are_resolved_without_expanding_schedule(
    setup, updates, expected
):
    cfg, fine = setup
    cfg.sampler = cfg.sampler.model_copy(update=updates)
    assert network_callback_weights(cfg, fine) == expected


@pytest.mark.parametrize(
    ("order", "summaries"),
    [([2, 0, 1, 3], 9.0), ([0, 2, 1, 3], 16.0), ([2, 3, 0, 1], 8.0)],
)
def test_contiguous_hit_segments_include_sweep_boundaries(setup, order, summaries):
    cfg, fine = setup
    cfg.sampler = cfg.sampler.model_copy(
        update={
            "blocks": [cfg.sampler.blocks[index] for index in order],
            "fsm_sweep_unroll": None,
        }
    )
    assert network_callback_weights(cfg, fine) == (72.0, summaries, 24.0)


def test_physical_chirp_mass_exposes_unsafe_scalar_sky_hits(setup):
    cfg, fine = setup
    cfg.sampling = cfg.sampling.model_copy(update={"chirp_mass_coordinate": "M_c"})
    cfg.sampler = cfg.sampler.model_copy(
        update={
            "blocks": [
                ["M_c" if name == "M_hat" else name for name in block]
                for block in cfg.sampler.blocks
            ]
        }
    )
    with pytest.raises(ValueError, match="hit groups may vary only psi, iota and d_L"):
        network_callback_weights(cfg, fine)


def test_blocks_are_validated_in_transformed_sampling_space(setup):
    cfg, fine = setup
    cfg.sampler = cfg.sampler.model_copy(
        update={"blocks": [*cfg.sampler.blocks[:-1], ["iota", "d_L"]]}
    )
    with pytest.raises(ValueError, match="not sampling parameters"):
        network_callback_weights(cfg, fine)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"type": "flowmc"}, "blackjax-swig"),
        ({"scheduler": "nested"}, "scheduler='fsm'"),
        ({"scalar_extrinsic_cache": False}, "scalar_extrinsic_cache"),
        ({"bridge_blocks": [["psi"]]}, "bridge_blocks"),
        ({"num_de_jumps": 1}, "num_de_jumps"),
        ({"de_jump_blocks": [object()]}, "de_jump_blocks"),
        ({"complementary_de_jump_block": object()}, "complementary_de_jump_block"),
        ({"fold_symmetry": object()}, "fold_symmetry"),
        (
            {
                "block_kernel_modes": [
                    "slice",
                    "slice",
                    "periodic-uniform-independence",
                    "slice",
                ]
            },
            "block_kernel_modes",
        ),
    ],
)
def test_unsupported_schedules_fail_before_likelihood_access(setup, updates, message):
    cfg, _ = setup
    cfg.sampler = cfg.sampler.model_copy(update={**updates, "fsm_sweep_unroll": None})
    with pytest.raises(ValueError, match=message):
        network_callback_weights(cfg, object())
