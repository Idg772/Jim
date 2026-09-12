"""CE FSM optimizations preserve ordinary production SwiG transitions."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import init_state_strategy
from jax.sharding import Mesh

from jimgw.samplers.blackjax.swig import BlackJAXSwiGSampler
from jimgw.samplers.config import BlackJAXSwiGConfig
from tests.unit.samplers.blackjax.test_swig_sweep_scan import _builder_options, _equal


def _config(mode="auto", **changes):
    options = _builder_options()
    return BlackJAXSwiGConfig(
        **{
            "blocks": [
                [f"x{i}" for i in group]
                for group in options["rebuild_required_by_block"]
            ],
            "n_live": 32,
            "n_delete_frac": 0.125,
            "num_gibbs_sweeps": 8,
            "scalar_extrinsic_cache": True,
            "fsm_optimizations": mode,
            **changes,
        }
    )


def _sampler(config, *, certified=True, options=None, **backend_changes):
    options = _builder_options() if options is None else options
    prior = options["log_prior_fn"]

    def score(x):
        return options["log_likelihood_from_cache_fn"](x, options["build_cache"](x))

    return BlackJAXSwiGSampler(
        n_dims=12,
        log_prior_fn=prior,
        log_likelihood_fn=score,
        log_posterior_fn=lambda x: prior(x) + score(x),
        config=config,
        periodic=options["periodic"],
        cache_independent_rebuild=certified,
        **{
            name: options[name]
            for name in (
                "rebuild_required_by_block",
                "build_cache",
                "log_likelihood_from_cache_fn",
                "prepare_hit_summary",
                "log_likelihood_from_hit_summary",
            )
        },
        **backend_changes,
    )


@pytest.mark.parametrize("mode", ["auto", "baseline", "ce-fast"])
def test_config_roundtrip_and_default(mode):
    config = _config(mode)
    assert config.fsm_optimizations == mode
    assert BlackJAXSwiGConfig.model_validate_json(config.model_dump_json()) == config
    assert BlackJAXSwiGConfig(blocks=[["x"]]).fsm_optimizations == "auto"


@pytest.mark.parametrize("mode", ["fast", "CE-fast", True, 1, None])
def test_unknown_mode_rejected(mode):
    with pytest.raises(ValueError, match="fsm_optimizations"):
        _config(mode)


@pytest.mark.parametrize(
    "changes",
    [
        {"scalar_extrinsic_cache": False},
        {"scheduler": "pre-fsm-lockstep"},
        {"direction_mode": "de-mix"},
        {"bracket_mode": "shrink-only", "adaptive_slice_widths": True},
        {"num_de_jumps": 1},
        {"bridge_blocks": [["x9"]]},
    ],
)
def test_forced_mode_rejects_unsupported_config(changes):
    with pytest.raises(ValueError, match="fsm_optimizations"):
        _config("ce-fast", **changes)


@pytest.mark.parametrize("certified", [False, None, 1])
def test_forced_mode_requires_explicit_cache_contract(certified):
    with pytest.raises(ValueError, match="certified independent positional"):
        _sampler(_config("ce-fast"), certified=certified)


@pytest.mark.parametrize("mode", ["auto", "ce-fast", "baseline"])
def test_eligible_adapter_resolves_requested_mode(mode):
    diagnostics = _sampler(_config(mode)).fsm_optimization_diagnostics
    enabled = mode != "baseline"
    assert diagnostics["requested"] == mode
    assert diagnostics["implementation"] == ("ce-fast-v1" if enabled else "baseline")
    assert diagnostics["skip_exhausted_endpoints"] is enabled
    assert diagnostics["defer_rebuild_cache"] is enabled
    assert diagnostics["fallback_reason"] is None


@pytest.mark.parametrize("reason", ["uncertified", "no-summary", "hit-last", "de-mix"])
def test_auto_falls_back_for_unsupported_adapter(reason):
    options = _builder_options()
    changes = {}
    if reason == "no-summary":
        options |= {
            "prepare_hit_summary": None,
            "log_likelihood_from_hit_summary": None,
        }
    elif reason == "hit-last":
        options["rebuild_required_by_block"] = dict(
            options["rebuild_required_by_block"]
        ) | {(10, 11): True}
    elif reason == "de-mix":
        changes["direction_mode"] = "de-mix"
    sampler = _sampler(
        _config("auto", **changes), certified=reason != "uncertified", options=options
    )
    diagnostics = sampler.fsm_optimization_diagnostics
    assert diagnostics["implementation"] == "baseline"
    assert not diagnostics["skip_exhausted_endpoints"]
    assert not diagnostics["defer_rebuild_cache"]
    assert diagnostics["fallback_reason"]


def test_three_adaptive_outer_steps_match_baseline():
    """Real ordinary outer steps update covariance, widths, counters and particles."""
    options = _builder_options()
    samplers = [
        _sampler(
            _config(
                mode,
                fsm_sweep_unroll=1,
                adaptive_slice_widths=True,
                periodic_wrapped_covariance=True,
            )
        )
        for mode in ("baseline", "ce-fast")
    ]
    assert samplers[0].fsm_optimization_diagnostics["implementation"] == "baseline"
    assert samplers[1].fsm_optimization_diagnostics["implementation"] == "ce-fast-v1"
    positions = jax.random.normal(jax.random.key(220), (32, 12)) * 0.25
    initial = adaptive_init(
        positions,
        init_state_fn=jax.vmap(
            partial(
                init_state_strategy,
                logprior_fn=options["log_prior_fn"],
                loglikelihood_fn=samplers[0]._log_likelihood_fn,
            )
        ),
        update_inner_kernel_params_fn=samplers[0]._fsm_update_inner_kernel_params_fn,
    )
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("replacement",))
    key = jax.random.key(221)
    kernels = [
        jax.jit(sampler._build_nested_sampler(4, mesh=mesh).step)
        .lower(key, initial)
        .compile()
        for sampler in samplers
    ]
    states = [initial, initial]
    expansions = 0
    for key in jax.random.split(jax.random.key(222), 3):
        key_before = np.asarray(jax.random.key_data(key)).copy()
        results = [
            jax.block_until_ready(kernel(key, state))
            for kernel, state in zip(kernels, states, strict=True)
        ]
        _equal(*results)
        np.testing.assert_array_equal(jax.random.key_data(key), key_before)
        states = [result[0] for result in results]
        expansions += int(jnp.sum(results[0][1].update_info.num_expansions))
    assert expansions > 0
    assert any(
        float(width) != 1.0 for width in states[0].inner_kernel_params["block_widths"]
    )
