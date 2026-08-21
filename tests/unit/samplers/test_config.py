import warnings

import jax
import numpy as np
import pytest
from pydantic import ValidationError

from jimgw.samplers.config import (
    BaseSamplerConfig,
    BlackJAXNSAWConfig,
    BlackJAXNSSConfig,
    BlackJAXSMCConfig,
    BlackJAXSwiGConfig,
    DEJumpBlockConfig,
    FlowMCConfig,
    FoldSymmetryConfig,
    GRWConfig,
    HMCConfig,
    MALAConfig,
    ParallelTemperingConfig,
    SamplerConfig,
)


def test_discriminated_union_dispatch_flowmc():
    cfg = FlowMCConfig.model_validate({"type": "flowmc", "n_chains": 500})
    assert isinstance(cfg, FlowMCConfig)
    assert cfg.n_chains == 500


def test_discriminated_union_dispatch_ns_aw():
    cfg = BlackJAXNSAWConfig.model_validate({"type": "blackjax-ns-aw", "n_live": 2000})
    assert isinstance(cfg, BlackJAXNSAWConfig)
    assert cfg.n_live == 2000


def test_sampler_config_union_from_dict():
    from pydantic import TypeAdapter

    ta = TypeAdapter(SamplerConfig)
    cfg = ta.validate_python({"type": "flowmc"})
    assert isinstance(cfg, FlowMCConfig)

    cfg2 = ta.validate_python({"type": "blackjax-ns-aw"})
    assert isinstance(cfg2, BlackJAXNSAWConfig)

    cfg3 = ta.validate_python({"type": "blackjax-nss"})
    assert isinstance(cfg3, BlackJAXNSSConfig)

    cfg4 = ta.validate_python({"type": "blackjax-smc"})
    assert isinstance(cfg4, BlackJAXSMCConfig)

    cfg5 = ta.validate_python({"type": "blackjax-swig", "blocks": [["x"], ["y"]]})
    assert isinstance(cfg5, BlackJAXSwiGConfig)


def test_swig_blocks_must_be_nonempty_and_unique():
    with pytest.raises(ValidationError, match="at least one"):
        BlackJAXSwiGConfig(blocks=[])
    with pytest.raises(ValidationError, match="empty"):
        BlackJAXSwiGConfig(blocks=[["x"], []])
    with pytest.raises(ValidationError, match="multiple blocks"):
        BlackJAXSwiGConfig(blocks=[["x"], ["x"]])


def test_swig_sampling_defaults():
    config = BlackJAXSwiGConfig(blocks=[["x"]])
    assert config.num_gibbs_sweeps == 2
    assert config.num_slice_steps_by_block is None
    assert config.termination_dlogz == pytest.approx(0.1)
    assert config.n_devices == 1
    assert config.scheduler == "fsm"
    assert config.de_jump_blocks == []
    assert config.complementary_de_jump_block is None
    assert config.block_kernel_modes is None
    assert config.periodic_wrapped_covariance is False
    assert config.bridge_blocks == []
    assert config.fold_symmetry is None


def test_fold_symmetry_config_validates_names_and_finite_center():
    fold = FoldSymmetryConfig(
        cos_iota="cos_iota",
        azimuth="azimuth",
        psi="psi",
        azimuth_reflection_center=0.25,
    )
    assert fold.cos_iota == "cos_iota"

    for field in ("cos_iota", "azimuth", "psi"):
        values = {
            "cos_iota": "cos_iota",
            "azimuth": "azimuth",
            "psi": "psi",
            "azimuth_reflection_center": 0.25,
        }
        values[field] = ""
        with pytest.raises(ValidationError, match="cannot be empty"):
            FoldSymmetryConfig(**values)

    with pytest.raises(ValidationError, match="must be distinct"):
        FoldSymmetryConfig(
            cos_iota="angle",
            azimuth="angle",
            psi="psi",
            azimuth_reflection_center=0.25,
        )

    for center in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValidationError, match="must be finite"):
            FoldSymmetryConfig(
                cos_iota="cos_iota",
                azimuth="azimuth",
                psi="psi",
                azimuth_reflection_center=center,
            )


def test_swig_fold_symmetry_requires_fsm_scheduler():
    fold = {
        "cos_iota": "cos_iota",
        "azimuth": "azimuth",
        "psi": "psi",
        "azimuth_reflection_center": 0.25,
    }
    config = BlackJAXSwiGConfig(blocks=[["x"]], fold_symmetry=fold)
    assert config.fold_symmetry == FoldSymmetryConfig(**fold)

    with pytest.raises(ValidationError, match="fold_symmetry requires scheduler='fsm'"):
        BlackJAXSwiGConfig(
            blocks=[["x"]],
            fold_symmetry=fold,
            scheduler="pre-fsm-lockstep",
        )


def test_swig_bridge_blocks_may_overlap_primary_but_require_unique_members():
    config = BlackJAXSwiGConfig(
        blocks=[["a"], ["b"]],
        bridge_blocks=[["a", "b"]],
    )
    assert config.bridge_blocks == [["a", "b"]]

    with pytest.raises(ValidationError, match="cannot contain empty"):
        BlackJAXSwiGConfig(
            blocks=[["a"], ["b"]],
            bridge_blocks=[[]],
        )
    with pytest.raises(ValidationError, match="duplicate parameters"):
        BlackJAXSwiGConfig(
            blocks=[["a"], ["b"]],
            bridge_blocks=[["a", "a"]],
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"direction_mode": "de-mix"}, "covariance directions"),
        ({"scheduler": "pre-fsm-lockstep"}, "scheduler='fsm'"),
        (
            {
                "block_kernel_modes": [
                    "periodic-uniform-independence",
                    "slice",
                ]
            },
            "periodic-uniform-independence",
        ),
        ({"adaptive_slice_widths": True}, "fixed slice widths"),
        (
            {"adaptive_slice_widths": True, "bracket_mode": "shrink-only"},
            "fixed slice widths",
        ),
    ],
)
def test_swig_bridge_blocks_require_plain_fixed_width_covariance_fsm(overrides, match):
    with pytest.raises(ValidationError, match=match):
        BlackJAXSwiGConfig(
            blocks=[["a"], ["b"]],
            bridge_blocks=[["a", "b"]],
            **overrides,
        )


def test_swig_bridge_blocks_reject_complementary_de_schedule():
    intrinsic = [f"intrinsic_{index}" for index in range(8)]
    with pytest.raises(ValidationError, match="complementary DE"):
        BlackJAXSwiGConfig(
            blocks=[intrinsic, ["phase"]],
            bridge_blocks=[[intrinsic[0], "phase"]],
            block_kernel_modes=["slice", "periodic-uniform-independence"],
            complementary_de_jump_block={
                "parameters": intrinsic,
                "attempts": 4,
            },
            num_gibbs_sweeps=1,
        )


def test_swig_accepts_one_complementary_de_block_inside_its_slice_segment():
    intrinsic = [f"intrinsic_{index}" for index in range(8)]
    config = BlackJAXSwiGConfig(
        blocks=[intrinsic, ["spin_phi"], ["fast_0", "fast_1"]],
        num_gibbs_sweeps=1,
        block_kernel_modes=[
            "slice",
            "periodic-uniform-independence",
            "slice",
        ],
        complementary_de_jump_block={"parameters": intrinsic, "attempts": 4},
    )

    assert config.complementary_de_jump_block == DEJumpBlockConfig(
        parameters=intrinsic,
        attempts=4,
    )


def test_swig_accepts_eight_complementary_de_attempts():
    intrinsic = [f"intrinsic_{index}" for index in range(8)]
    config = BlackJAXSwiGConfig(
        blocks=[intrinsic, ["spin_phi"], ["fast_0", "fast_1"]],
        num_gibbs_sweeps=1,
        block_kernel_modes=[
            "slice",
            "periodic-uniform-independence",
            "slice",
        ],
        complementary_de_jump_block={"parameters": intrinsic, "attempts": 8},
    )

    assert config.complementary_de_jump_block == DEJumpBlockConfig(
        parameters=intrinsic,
        attempts=8,
    )


def test_swig_complementary_de_must_match_one_complete_slice_block():
    with pytest.raises(ValidationError, match="complete slice block"):
        BlackJAXSwiGConfig(
            blocks=[["intrinsic_0", "intrinsic_1"], ["fast"]],
            num_gibbs_sweeps=1,
            complementary_de_jump_block={
                "parameters": ["intrinsic_0"],
                "attempts": 4,
            },
        )


@pytest.mark.parametrize("attempts", [1, 2, 3, 5, 6, 7, 9])
def test_swig_complementary_de_rejects_unregistered_attempt_budgets(attempts):
    intrinsic = [f"intrinsic_{index}" for index in range(8)]
    with pytest.raises(ValidationError, match="exactly four or eight attempts"):
        BlackJAXSwiGConfig(
            blocks=[intrinsic, ["spin_phi"], ["fast_0", "fast_1"]],
            num_gibbs_sweeps=1,
            block_kernel_modes=[
                "slice",
                "periodic-uniform-independence",
                "slice",
            ],
            complementary_de_jump_block={
                "parameters": intrinsic,
                "attempts": attempts,
            },
        )


def test_swig_complementary_de_short_kernel_mode_list_fails_as_validation_error():
    intrinsic = [f"intrinsic_{index}" for index in range(8)]
    with pytest.raises(
        ValidationError, match="block_kernel_modes must contain exactly one mode"
    ):
        BlackJAXSwiGConfig(
            blocks=[["phase"], intrinsic, ["fast"]],
            block_kernel_modes=["periodic-uniform-independence"],
            num_gibbs_sweeps=1,
            complementary_de_jump_block={
                "parameters": intrinsic,
                "attempts": 4,
            },
        )


@pytest.mark.parametrize(
    "override",
    [
        {"num_gibbs_sweeps": 2},
        {"num_inner_steps_per_dim": 2},
        {"scheduler": "pre-fsm-lockstep"},
        {"direction_mode": "de-mix"},
        {"num_de_jumps": 1},
        {"de_jump_blocks": [{"parameters": ["fast_0"], "attempts": 1}]},
        {"num_slice_steps_by_block": [8, 1, 2]},
        {
            "block_kernel_modes": [
                "periodic-uniform-independence",
                "periodic-uniform-independence",
                "slice",
            ]
        },
    ],
)
def test_swig_complementary_de_fails_closed_outside_frozen_hybrid_schedule(
    override,
):
    intrinsic = [f"intrinsic_{index}" for index in range(8)]
    kwargs = {
        "blocks": [intrinsic, ["spin_phi"], ["fast_0", "fast_1"]],
        "num_gibbs_sweeps": 1,
        "block_kernel_modes": [
            "slice",
            "periodic-uniform-independence",
            "slice",
        ],
        "complementary_de_jump_block": {
            "parameters": intrinsic,
            "attempts": 4,
        },
    }
    kwargs.update(override)

    with pytest.raises(ValidationError, match="complementary_de_jump_block"):
        BlackJAXSwiGConfig(**kwargs)


def test_swig_accepts_periodic_uniform_independence_for_singleton_blocks():
    config = BlackJAXSwiGConfig(
        blocks=[["intrinsic_0", "intrinsic_1"], ["spin_phi"], ["ridge_0", "ridge_1"]],
        num_gibbs_sweeps=1,
        block_kernel_modes=[
            "slice",
            "periodic-uniform-independence",
            "slice",
        ],
    )

    assert config.block_kernel_modes == [
        "slice",
        "periodic-uniform-independence",
        "slice",
    ]


def test_swig_block_kernel_modes_require_one_mode_per_block():
    with pytest.raises(ValidationError, match="exactly one mode per block"):
        BlackJAXSwiGConfig(
            blocks=[["slow"], ["periodic"]],
            num_gibbs_sweeps=1,
            block_kernel_modes=["slice"],
        )


def test_swig_periodic_uniform_independence_requires_singleton_block():
    with pytest.raises(ValidationError, match="singleton"):
        BlackJAXSwiGConfig(
            blocks=[["periodic_0", "periodic_1"]],
            num_gibbs_sweeps=1,
            block_kernel_modes=["periodic-uniform-independence"],
        )


def test_swig_periodic_uniform_independence_allows_multiple_gibbs_sweeps():
    config = BlackJAXSwiGConfig(
        blocks=[["periodic"]],
        num_gibbs_sweeps=2,
        block_kernel_modes=["periodic-uniform-independence"],
    )

    assert config.num_gibbs_sweeps == 2


@pytest.mark.parametrize(
    "work_budget",
    [
        {"num_inner_steps_per_dim": 2},
    ],
)
def test_swig_periodic_uniform_independence_rejects_multiple_block_updates(
    work_budget,
):
    with pytest.raises(ValidationError, match="exactly one update per block"):
        BlackJAXSwiGConfig(
            blocks=[["periodic"]],
            num_gibbs_sweeps=1,
            block_kernel_modes=["periodic-uniform-independence"],
            **work_budget,
        )


def test_swig_periodic_uniform_independence_rejects_explicit_block_budget():
    with pytest.raises(ValidationError, match="cannot be combined"):
        BlackJAXSwiGConfig(
            blocks=[["periodic"]],
            num_gibbs_sweeps=1,
            block_kernel_modes=["periodic-uniform-independence"],
            num_slice_steps_by_block=[1],
        )


@pytest.mark.parametrize(
    "incompatible",
    [
        {"direction_mode": "de-mix"},
        {"direction_mode": "covariance-basis-8d"},
        {"num_de_jumps": 1},
        {"de_jump_blocks": [{"parameters": ["periodic"]}]},
    ],
)
def test_swig_periodic_uniform_independence_rejects_de_modes(incompatible):
    with pytest.raises(ValidationError, match="cannot be combined with DE"):
        BlackJAXSwiGConfig(
            blocks=[["periodic"]],
            num_gibbs_sweeps=1,
            block_kernel_modes=["periodic-uniform-independence"],
            **incompatible,
        )


def test_swig_accepts_explicit_slice_steps_for_each_block():
    config = BlackJAXSwiGConfig(
        blocks=_covariance_basis_blocks(),
        num_gibbs_sweeps=1,
        num_slice_steps_by_block=[5, 1, 1, 2, 1, 2],
    )

    assert config.num_slice_steps_by_block == [5, 1, 1, 2, 1, 2]
    assert sum(config.num_slice_steps_by_block) == 12
    assert config.direction_mode == "covariance"
    assert config.num_de_jumps == 0


def test_swig_explicit_slice_steps_require_one_count_per_block():
    with pytest.raises(ValidationError, match="exactly one count per block"):
        BlackJAXSwiGConfig(
            blocks=[["slow"], ["fast"]],
            num_slice_steps_by_block=[1],
        )


def test_swig_explicit_slice_steps_must_be_positive():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        BlackJAXSwiGConfig(
            blocks=[["slow"], ["fast"]],
            num_slice_steps_by_block=[1, 0],
        )


def test_swig_explicit_slice_steps_reject_a_second_per_dimension_budget():
    with pytest.raises(ValidationError, match="requires num_inner_steps_per_dim=1"):
        BlackJAXSwiGConfig(
            blocks=[["slow"], ["fast"]],
            num_inner_steps_per_dim=2,
            num_slice_steps_by_block=[1, 1],
        )


def test_swig_explicit_slice_steps_reject_covariance_basis_mode():
    with pytest.raises(ValidationError, match="cannot be combined"):
        BlackJAXSwiGConfig(
            blocks=_covariance_basis_blocks(),
            direction_mode="covariance-basis-8d",
            num_gibbs_sweeps=1,
            num_slice_steps_by_block=[8, 1, 1, 2, 1, 2],
        )


def _covariance_basis_blocks():
    intrinsic = [f"intrinsic_{index}" for index in range(8)]
    return [
        intrinsic,
        ["spin_phi_1"],
        ["spin_phi_2"],
        ["sky_1", "sky_2"],
        ["polarization"],
        ["ridge_1", "ridge_2"],
    ]


def test_swig_accepts_covariance_basis_mode_for_one_unique_8d_block():
    config = BlackJAXSwiGConfig(
        blocks=_covariance_basis_blocks(),
        direction_mode="covariance-basis-8d",
        num_gibbs_sweeps=1,
        num_inner_steps_per_dim=1,
    )

    assert config.direction_mode == "covariance-basis-8d"


def test_swig_covariance_basis_mode_requires_an_8d_block():
    with pytest.raises(ValidationError, match="exactly one 8D block"):
        BlackJAXSwiGConfig(
            blocks=[["slow"], ["fast"]],
            direction_mode="covariance-basis-8d",
            num_gibbs_sweeps=1,
            num_inner_steps_per_dim=1,
        )


def test_swig_covariance_basis_mode_requires_one_gibbs_sweep():
    with pytest.raises(ValidationError, match="num_gibbs_sweeps=1"):
        BlackJAXSwiGConfig(
            blocks=_covariance_basis_blocks(),
            direction_mode="covariance-basis-8d",
            num_gibbs_sweeps=2,
            num_inner_steps_per_dim=1,
        )


def test_swig_covariance_basis_mode_requires_one_update_per_dimension():
    with pytest.raises(ValidationError, match="num_inner_steps_per_dim=1"):
        BlackJAXSwiGConfig(
            blocks=_covariance_basis_blocks(),
            direction_mode="covariance-basis-8d",
            num_gibbs_sweeps=1,
            num_inner_steps_per_dim=2,
        )


def test_swig_covariance_basis_mode_requires_exactly_15_updates():
    intrinsic = [f"intrinsic_{index}" for index in range(8)]
    with pytest.raises(ValidationError, match="15 total slice updates"):
        BlackJAXSwiGConfig(
            blocks=[intrinsic, ["fast"]],
            direction_mode="covariance-basis-8d",
            num_gibbs_sweeps=1,
            num_inner_steps_per_dim=1,
        )


def test_swig_named_de_jump_blocks_are_typed_and_validated():
    config = BlackJAXSwiGConfig(
        blocks=[["iota"], ["d_L"]],
        de_jump_blocks=[{"parameters": ["iota", "d_L"], "attempts": 1}],
    )

    assert config.de_jump_blocks == [
        DEJumpBlockConfig(parameters=["iota", "d_L"], attempts=1)
    ]
    with pytest.raises(ValidationError, match="cannot be empty"):
        BlackJAXSwiGConfig(
            blocks=[["iota"]],
            de_jump_blocks=[{"parameters": [], "attempts": 1}],
        )
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        BlackJAXSwiGConfig(
            blocks=[["iota"]],
            de_jump_blocks=[{"parameters": ["iota"], "attempts": 0}],
        )
    with pytest.raises(ValidationError, match="distinct parameter groups"):
        BlackJAXSwiGConfig(
            blocks=[["iota"], ["d_L"]],
            de_jump_blocks=[
                {"parameters": ["iota", "d_L"]},
                {"parameters": ["d_L", "iota"]},
            ],
        )


def test_swig_scheduler_is_validated():
    config = BlackJAXSwiGConfig(blocks=[["x"]], scheduler="pre-fsm-lockstep")
    assert config.scheduler == "pre-fsm-lockstep"
    with pytest.raises(ValidationError, match="scheduler"):
        BlackJAXSwiGConfig(blocks=[["x"]], scheduler="unknown")


def test_sampler_configs_do_not_advertise_cache_capabilities():
    assert not hasattr(FlowMCConfig(), "cache_blocks")
    assert not hasattr(BlackJAXSwiGConfig(blocks=[["x"]]), "cache_blocks")


@pytest.mark.parametrize(
    ("config_cls", "config_kwargs"),
    [
        (BlackJAXNSAWConfig, {}),
        (BlackJAXNSSConfig, {}),
        (BlackJAXSwiGConfig, {"blocks": [["x"]]}),
    ],
)
@pytest.mark.parametrize("termination_dlogz", [0.0, -1.0])
def test_nested_sampling_termination_dlogz_must_be_positive(
    config_cls, config_kwargs, termination_dlogz
):
    with pytest.raises(ValidationError, match="greater than 0"):
        config_cls(**config_kwargs, termination_dlogz=termination_dlogz)


@pytest.mark.parametrize("config_cls", [BlackJAXNSSConfig, BlackJAXSwiGConfig])
def test_nested_sampler_sharding_requires_divisible_particle_counts(config_cls):
    kwargs = {"blocks": [["x"]]} if config_cls is BlackJAXSwiGConfig else {}
    with pytest.raises(ValidationError, match="n_live must be divisible by n_devices"):
        config_cls(n_live=10, n_delete_frac=0.4, n_devices=4, **kwargs)
    with pytest.raises(
        ValidationError, match="n_delete must be divisible by n_devices"
    ):
        config_cls(n_live=12, n_delete_frac=0.25, n_devices=2, **kwargs)


@pytest.mark.parametrize("config_cls", [BlackJAXNSSConfig, BlackJAXSwiGConfig])
def test_nested_sampler_sharding_requires_positive_device_count(config_cls):
    kwargs = {"blocks": [["x"]]} if config_cls is BlackJAXSwiGConfig else {}
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        config_cls(n_devices=0, **kwargs)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        FlowMCConfig(unknown_field=42)


def test_n_delete_frac_validator():
    with pytest.raises(ValidationError):
        BlackJAXNSAWConfig(n_delete_frac=0.0)
    with pytest.raises(ValidationError):
        BlackJAXNSAWConfig(n_delete_frac=1.0)
    cfg = BlackJAXNSAWConfig(n_delete_frac=0.5)
    assert cfg.n_delete_frac == 0.5


def test_base_config_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        BaseSamplerConfig[str](type="test", unknown_field=True)


def test_base_config_requires_sampler_type():
    with pytest.raises(ValidationError, match="type"):
        BaseSamplerConfig[str]()


# ---------------------------------------------------------------------------
# B1: FlowMC kernel/PT warning validator
# ---------------------------------------------------------------------------


def test_flowmc_pt_off_by_default():
    cfg = FlowMCConfig()
    assert cfg.parallel_tempering is None


def test_flowmc_pt_on_with_config():
    cfg = FlowMCConfig(parallel_tempering=ParallelTemperingConfig(n_temperatures=3))
    assert cfg.parallel_tempering is not None
    assert cfg.parallel_tempering.n_temperatures == 3


def test_flowmc_pt_on_with_true():
    cfg = FlowMCConfig(parallel_tempering=True)
    assert cfg.parallel_tempering is not None
    assert cfg.parallel_tempering.n_temperatures == 5  # default


def test_flowmc_pt_on_with_dict():
    cfg = FlowMCConfig(parallel_tempering={"n_temperatures": 8})
    assert cfg.parallel_tempering is not None
    assert cfg.parallel_tempering.n_temperatures == 8


def test_flowmc_pt_off_with_false():
    cfg = FlowMCConfig(parallel_tempering=False)
    assert cfg.parallel_tempering is None


def test_flowmc_pt_off_with_none():
    cfg = FlowMCConfig(parallel_tempering=None)
    assert cfg.parallel_tempering is None


def test_flowmc_irrelevant_kernel_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FlowMCConfig(local_kernel="MALA", hmc=HMCConfig(step_size=0.5))
    assert any("hmc" in str(warning.message).lower() for warning in w)


def test_flowmc_irrelevant_parallel_tempering_warns():
    # No warning expected: passing PT config enables it, nothing is ignored.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FlowMCConfig(parallel_tempering=ParallelTemperingConfig(n_temperatures=10))
    pt_warnings = [x for x in w if "parallel_tempering" in str(x.message).lower()]
    assert len(pt_warnings) == 0


def test_flowmc_no_spurious_warning_when_kernel_matches():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FlowMCConfig(local_kernel="HMC", hmc=HMCConfig(step_size=0.5))
    kernel_warnings = [x for x in w if "hmc" in str(x.message).lower()]
    assert len(kernel_warnings) == 0


# ---------------------------------------------------------------------------
# B2: BlackJAXSMCConfig temperature ladder validator
# ---------------------------------------------------------------------------


def test_smc_config_defaults():
    cfg = BlackJAXSMCConfig()
    assert cfg.persistent_sampling is True
    assert cfg.temperature_ladder is None


def test_smc_temperature_ladder_valid():
    cfg = BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 1.0])
    assert cfg.temperature_ladder == [0.0, 0.5, 1.0]


def test_smc_temperature_ladder_must_start_at_zero():
    with pytest.raises(ValidationError, match=r"start at 0\.0"):
        BlackJAXSMCConfig(temperature_ladder=[0.1, 0.5, 1.0])


def test_smc_temperature_ladder_must_end_at_one():
    with pytest.raises(ValidationError, match=r"end at 1\.0"):
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 0.9])


def test_smc_temperature_ladder_must_be_increasing():
    with pytest.raises(ValidationError, match="increasing"):
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.8, 0.5, 1.0])


def test_smc_temperature_ladder_warns_ess():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 1.0], target_ess=5000)
    assert any("ESS" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# B2b: BlackJAXSMCConfig ESS XOR validator
# ---------------------------------------------------------------------------


def test_smc_default_ess_fraction():
    cfg = BlackJAXSMCConfig()
    assert cfg.target_ess_fraction == 0.9
    assert cfg.target_ess is None


def test_smc_ess_fraction_set():
    cfg = BlackJAXSMCConfig(target_ess_fraction=0.3)
    assert cfg.target_ess_fraction == 0.3
    assert cfg.target_ess is None


def test_smc_absolute_ess_set():
    cfg = BlackJAXSMCConfig(target_ess=1000)
    assert cfg.target_ess == 1000
    assert cfg.target_ess_fraction is None


def test_smc_both_ess_raises():
    with pytest.raises(ValidationError, match="exactly one"):
        BlackJAXSMCConfig(target_ess_fraction=0.9, target_ess=1000)


def test_smc_fraction_zero_raises():
    with pytest.raises(ValidationError):
        BlackJAXSMCConfig(target_ess_fraction=0.0)


def test_smc_fraction_above_one_in_persistent_ok():
    cfg = BlackJAXSMCConfig(target_ess_fraction=1.5, persistent_sampling=True)
    assert cfg.target_ess_fraction == 1.5


def test_smc_fraction_above_one_in_tempered_raises():
    with pytest.raises(ValidationError, match=r"1\.0"):
        BlackJAXSMCConfig(target_ess_fraction=1.5, persistent_sampling=False)


def test_smc_absolute_ess_above_n_particles_in_tempered_raises():
    with pytest.raises(ValidationError):
        BlackJAXSMCConfig(target_ess=5000, n_particles=2000, persistent_sampling=False)


def test_smc_absolute_ess_above_n_particles_in_persistent_ok():
    cfg = BlackJAXSMCConfig(target_ess=5000, n_particles=2000, persistent_sampling=True)
    assert cfg.target_ess == 5000


def test_smc_fraction_warns_with_fixed_ladder():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BlackJAXSMCConfig(temperature_ladder=[0.0, 0.5, 1.0], target_ess_fraction=0.3)
    assert any(
        "target_ess_fraction" in str(x.message) or "ESS" in str(x.message) for x in w
    )


# ---------------------------------------------------------------------------
# C: New kernel sub-config features (array step sizes, condition_matrix)
# ---------------------------------------------------------------------------


def test_mala_step_size_scalar():
    cfg = MALAConfig(step_size=1e-2)
    assert cfg.step_size == 1e-2


def test_mala_step_size_array():
    arr = np.array([1e-2, 2e-2, 3e-2])
    cfg = MALAConfig(step_size=arr)
    np.testing.assert_array_equal(cfg.step_size, arr)


def test_grw_step_size_array():
    arr = np.array([5e-3, 1e-2])
    cfg = GRWConfig(step_size=arr)
    np.testing.assert_array_equal(cfg.step_size, arr)


def test_hmc_condition_matrix_scalar():
    cfg = HMCConfig(condition_matrix=2.0)
    assert cfg.condition_matrix == 2.0


def test_hmc_condition_matrix_array():
    arr = np.array([1.0, 2.0, 0.5])
    cfg = HMCConfig(condition_matrix=arr)
    np.testing.assert_array_equal(cfg.condition_matrix, arr)


def test_hmc_defaults():
    cfg = HMCConfig()
    assert cfg.step_size == 2e-3
    assert cfg.condition_matrix == 1.0
    assert cfg.n_leapfrog_steps == 10


# ---------------------------------------------------------------------------
# D: Config classes no longer have a periodic field
# ---------------------------------------------------------------------------


def test_flowmc_config_has_no_periodic_field():
    assert not hasattr(FlowMCConfig(), "periodic")


def test_blackjax_ns_aw_config_has_no_periodic_field():
    assert not hasattr(BlackJAXNSAWConfig(), "periodic")


def test_blackjax_nss_config_has_no_periodic_field():
    assert not hasattr(BlackJAXNSSConfig(), "periodic")


def test_blackjax_smc_config_has_no_periodic_field():
    assert not hasattr(BlackJAXSMCConfig(), "periodic")


def test_flowmc_config_rejects_periodic_field():
    with pytest.raises(ValidationError):
        FlowMCConfig(periodic={"phase_c": (0.0, 6.28)})


def test_smc_resolve_target_ess_fraction():
    cfg = BlackJAXSMCConfig(target_ess_fraction=0.4)
    assert cfg._resolve_target_ess_fraction() == pytest.approx(0.4)

    cfg2 = BlackJAXSMCConfig(target_ess=1000, n_particles=2000)
    assert cfg2._resolve_target_ess_fraction() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# E: checkpoint_dir / checkpoint_interval validators
# ---------------------------------------------------------------------------


def test_checkpoint_dir_accepts_string(tmp_path):
    from pathlib import Path

    cfg = BlackJAXNSAWConfig(checkpoint_dir=str(tmp_path))
    assert cfg.checkpoint_dir == tmp_path
    assert isinstance(cfg.checkpoint_dir, Path)


def test_checkpoint_interval_negative_raises():
    with pytest.raises(ValidationError):
        FlowMCConfig(checkpoint_interval=-1.0)


def test_checkpoint_interval_without_dir_raises():
    with pytest.raises(ValidationError, match="checkpoint_dir must be set"):
        BlackJAXNSAWConfig(checkpoint_interval=600.0)


def test_checkpoint_interval_with_dir_ok(tmp_path):
    cfg = BlackJAXNSAWConfig(checkpoint_dir=tmp_path, checkpoint_interval=600.0)
    assert cfg.checkpoint_dir == tmp_path
    assert cfg.checkpoint_interval == 600.0


# ---------------------------------------------------------------------------
# F: configure_jax_cache
# ---------------------------------------------------------------------------


def test_configure_jax_cache_sets_dir(tmp_path):
    original = getattr(jax.config, "jax_compilation_cache_dir", None)
    try:
        BlackJAXNSAWConfig(
            checkpoint_dir=tmp_path, checkpoint_interval=60.0
        ).configure_jax_cache()
        assert (tmp_path / "jax_cache").is_dir()
        assert getattr(jax.config, "jax_compilation_cache_dir", None) == str(
            tmp_path / "jax_cache"
        )
    finally:
        jax.config.update("jax_compilation_cache_dir", original)


def test_configure_jax_cache_noop_when_no_dir():
    original = getattr(jax.config, "jax_compilation_cache_dir", None)
    try:
        BlackJAXNSAWConfig().configure_jax_cache()
        assert getattr(jax.config, "jax_compilation_cache_dir", None) == original
    finally:
        jax.config.update("jax_compilation_cache_dir", original)


def test_swig_adaptive_width_defaults():
    config = BlackJAXSwiGConfig(blocks=[["x"]])
    assert config.adaptive_slice_widths is False
    assert config.width_adaptation_rate == 0.25
    assert config.width_target_expansions == 1.0
    assert config.width_target_shrinks == 3.0
    assert config.bracket_mode == "stepping-out"


def test_swig_adaptive_width_validation():
    with pytest.raises(ValidationError):
        BlackJAXSwiGConfig(blocks=[["x"]], width_adaptation_rate=0.0)
    with pytest.raises(ValidationError):
        BlackJAXSwiGConfig(blocks=[["x"]], width_target_shrinks=0.0)
    # shrink-only requires adaptation on
    with pytest.raises(ValidationError):
        BlackJAXSwiGConfig(blocks=[["x"]], bracket_mode="shrink-only")
    # widths require the covariance direction mode
    with pytest.raises(ValidationError):
        BlackJAXSwiGConfig(
            blocks=[["x"]], adaptive_slice_widths=True, direction_mode="de-mix"
        )
    # widths require the FSM scheduler
    with pytest.raises(ValidationError):
        BlackJAXSwiGConfig(
            blocks=[["x"]], adaptive_slice_widths=True, scheduler="pre-fsm-lockstep"
        )
    # widths exclude mixed block kernel modes
    with pytest.raises(ValidationError):
        BlackJAXSwiGConfig(
            blocks=[["x"]],
            adaptive_slice_widths=True,
            block_kernel_modes=["periodic-uniform-independence"],
        )
    ok = BlackJAXSwiGConfig(
        blocks=[["x"]], adaptive_slice_widths=True, bracket_mode="shrink-only"
    )
    assert ok.bracket_mode == "shrink-only"
