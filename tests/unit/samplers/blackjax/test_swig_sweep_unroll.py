"""Configured sweep rolling preserves the prescribed FSM transition."""

import json
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import StateWithLogLikelihood, init_state_strategy
from jax.sharding import Mesh

from jimgw.samplers.blackjax.swig import (
    BlackJAXSwiGSampler,
    _build_swig_constrained_step,
)
from jimgw.samplers.config import BlackJAXSwiGConfig
from tests.unit.samplers.blackjax.test_swig_sweep_scan import (
    _builder_options,
    _equal,
    _jaxpr_counts,
)


def _config(unroll=None, **changes):
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
            "fsm_sweep_unroll": unroll,
            **changes,
        }
    )


def _sampler(config, options=None):
    options = _builder_options() if options is None else options
    prior = options["log_prior_fn"]
    score = lambda x: options["log_likelihood_from_cache_fn"](
        x, options["build_cache"](x)
    )
    return BlackJAXSwiGSampler(
        n_dims=12,
        log_prior_fn=prior,
        log_likelihood_fn=score,
        log_posterior_fn=lambda x: prior(x) + score(x),
        config=config,
        periodic=options["periodic"],
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
    )


@pytest.mark.parametrize("unroll", [None, 1, 2, 3, 4, 8])
def test_sweep_unroll_config_is_explicit_and_roundtrips(unroll):
    config = _config(unroll)
    assert config.fsm_sweep_unroll == unroll
    assert BlackJAXSwiGConfig.model_validate_json(config.model_dump_json()) == config
    assert BlackJAXSwiGConfig(blocks=[["x"], ["y"]]).fsm_sweep_unroll is None


@pytest.mark.parametrize("unroll", [0, -1, 9, True, 1.0, "2"])
def test_sweep_unroll_rejects_invalid_counts(unroll):
    with pytest.raises(ValueError, match="fsm_sweep_unroll"):
        _config(unroll)


@pytest.mark.parametrize(
    "changes",
    [
        {"scheduler": "pre-fsm-lockstep"},
        {"scalar_extrinsic_cache": False},
        {"direction_mode": "de-mix"},
        {"bracket_mode": "shrink-only", "adaptive_slice_widths": True},
        {"num_de_jumps": 1},
        {"bridge_blocks": [["x9", "x10"]]},
        {
            "block_kernel_modes": [
                "slice",
                "slice",
                "periodic-uniform-independence",
                "slice",
            ]
        },
    ],
)
def test_sweep_unroll_rejects_unqualified_execution_modes(changes):
    with pytest.raises(ValueError, match="fsm_sweep_unroll"):
        _config(1, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"prepare_hit_summary": None, "log_likelihood_from_hit_summary": None},
        {
            "rebuild_required_by_block": {
                tuple(range(7)): True,
                (7, 8): True,
                (9,): False,
                (10, 11): True,
            }
        },
    ],
)
def test_resolved_cache_schedule_rejected_before_sampling(changes):
    with pytest.raises(ValueError, match="fsm_sweep_unroll"):
        _sampler(_config(1), _builder_options() | changes)


def test_private_sweep_scan_remains_compatible():
    options = _builder_options()
    x = jnp.linspace(-0.2, 0.2, 12)
    threshold = jnp.asarray(-4.0)
    state = StateWithLogLikelihood(
        x,
        options["log_prior_fn"](x),
        options["log_likelihood_from_cache_fn"](x, options["build_cache"](x)),
        threshold,
    )
    factors = tuple(jnp.eye(n) * 0.35 for n in (7, 2, 1, 2))
    key = jax.random.key(27)
    private = _build_swig_constrained_step(**options, _scan_sweeps=True)
    configured = _build_swig_constrained_step(**options, fsm_sweep_unroll=1)
    _equal(
        jax.jit(
            lambda k: private(k, state, threshold, block_covariance_factors=factors)
        )(key),
        jax.jit(
            lambda k: configured(k, state, threshold, block_covariance_factors=factors)
        )(key),
    )
    with pytest.raises(ValueError, match="fsm_sweep_unroll"):
        _build_swig_constrained_step(**options, _scan_sweeps=True, fsm_sweep_unroll=2)


def test_configured_outer_steps_preserve_adaptation_and_diagnostics(tmp_path):
    """Compile the real outer sampler through its normal production adapter."""
    options = _builder_options()
    prior = options["log_prior_fn"]
    score = lambda x: options["log_likelihood_from_cache_fn"](
        x, options["build_cache"](x)
    )
    samplers = [
        _sampler(
            _config(
                unroll, adaptive_slice_widths=True, periodic_wrapped_covariance=True
            )
        )
        for unroll in (None, 1, 2, 4)
    ]
    positions = jax.random.normal(jax.random.key(220), (32, 12)) * 0.25
    initial = adaptive_init(
        positions,
        init_state_fn=jax.vmap(
            partial(init_state_strategy, logprior_fn=prior, loglikelihood_fn=score)
        ),
        update_inner_kernel_params_fn=samplers[0]._fsm_update_inner_kernel_params_fn,
    )
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("replacement",))
    key = jax.random.key(221)
    kernels, evidence = [], {}
    for unroll, sampler in zip((None, 1, 2, 4), samplers, strict=True):
        step = sampler._build_nested_sampler(4, mesh=mesh).step
        counts = _jaxpr_counts(step, key, initial)
        started = time.perf_counter()
        lowered = jax.jit(step).lower(key, initial)
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
        compiled = lowered.compile()
        counts["lower_compile_seconds"] = time.perf_counter() - started
        counts["stablehlo_characters"] = len(stablehlo)
        kernels.append(compiled)
        evidence[str(unroll)] = counts
    states = [initial] * len(kernels)
    extensions = 0
    for key in jax.random.split(jax.random.key(222), 3):
        results = [
            jax.block_until_ready(kernel(key, state))
            for kernel, state in zip(kernels, states, strict=True)
        ]
        for candidate in results[1:]:
            _equal(results[0], candidate)
        states = [result[0] for result in results]
        extensions += int(jnp.sum(results[0][1].update_info.num_expansions))
    assert extensions > 0
    assert any(
        float(width) != 1 for width in states[0].inner_kernel_params["block_widths"]
    )
    assert evidence["1"]["equations"] < evidence["None"]["equations"]
    assert evidence["1"]["while"] < evidence["None"]["while"]
    (tmp_path / "configured-sweep-compile.json").write_text(
        json.dumps(evidence, indent=2)
    )
    print("configured sweep compile evidence", json.dumps(evidence))


def test_physical_network_sweep_unroll_preserves_eight_sweep_transition(tmp_path):
    """Real full-band response and scalar-cache callbacks, bounded quadrature."""
    from jimgw.core.prior import (
        CombinePrior,
        CosinePrior,
        PowerLawPrior,
        SinePrior,
        UniformPrior,
    )
    from jimgw.core.single_event.heterodyne_extrinsics import evaluate_extrinsic_summary
    from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
    from tests.unit.core.single_event.test_xg_network_evaluation import (
        PARAMETERS,
        _make_native_network,
        _options,
        _read_psd_tables,
    )

    detectors, waveform = _make_native_network(_read_psd_tables(tmp_path))
    edges = np.r_[2.0, 4.0, np.geomspace(8.0, 2048.0, 25)]
    likelihood = HeterodynedTransientLikelihoodFD(
        **_options(detectors, waveform, edges, True)
    )
    intervals = {
        "M_c": (1.18, 1.1807),
        "eta": (2 / 9, 0.25),
        "s1_z": (-0.05, 0.05),
        "s2_z": (-0.05, 0.05),
        "lambda_1": (0.0, 1000.0),
        "lambda_2": (0.0, 1000.0),
        "t_c": (-0.1, 0.1),
        "ra": (0.0, 2 * np.pi),
        "psi": (0.0, np.pi),
    }
    prior = CombinePrior(
        [
            *(UniformPrior(lo, hi, [name]) for name, (lo, hi) in intervals.items()),
            CosinePrior(["dec"]),
            SinePrior(["iota"]),
            PowerLawPrior(1.0, 1000.0, 2.0, ["d_L"]),
        ]
    )
    blocks = [
        ["M_c", "eta", "lambda_1", "lambda_2", "s1_z", "s2_z", "t_c"],
        ["ra", "dec"],
        ["psi"],
        ["iota", "d_L"],
    ]
    names = prior.parameter_names

    def parameters(position):
        return dict(zip(names, position, strict=True))

    def log_prior(position):
        return prior.log_prob(parameters(position))

    def log_likelihood(position):
        return likelihood.evaluate(parameters(position))

    # Physical-coordinate sky moves need no new source waveform. Conservatively
    # rebuilding that block reproduces the production transformed XG R9/H3
    # service schedule while keeping this test independent of those transforms.
    groups = {
        tuple(names.index(name) for name in block): index < 2
        for index, block in enumerate(blocks)
    }
    samplers = [
        BlackJAXSwiGSampler(
            n_dims=12,
            log_prior_fn=log_prior,
            log_likelihood_fn=log_likelihood,
            log_posterior_fn=lambda position: (
                log_prior(position) + log_likelihood(position)
            ),
            build_cache=lambda position: likelihood.generate_waveform(
                parameters(position)
            ),
            log_likelihood_from_cache_fn=lambda position, cache: (
                likelihood.evaluate_from_waveform(parameters(position), cache)
            ),
            prepare_hit_summary=lambda position, cache: (
                likelihood.build_extrinsic_summary(parameters(position), cache)
            ),
            log_likelihood_from_hit_summary=lambda position, summary: (
                evaluate_extrinsic_summary(likelihood, parameters(position), summary)
            ),
            rebuild_required_by_block=groups,
            periodic={
                names.index("ra"): (0.0, 2 * np.pi),
                names.index("psi"): (0.0, np.pi),
            },
            config=BlackJAXSwiGConfig(
                blocks=blocks,
                scalar_extrinsic_cache=True,
                num_gibbs_sweeps=8,
                fsm_sweep_unroll=unroll,
                adaptive_slice_widths=True,
                n_live=32,
            ),
        )
        for unroll in (None, 1, 2, 4)
    ]
    position = jnp.asarray([PARAMETERS[name] for name in names])
    score = log_likelihood(position)
    threshold = score - 10.0
    state = StateWithLogLikelihood(position, log_prior(position), score, threshold)
    scales = {
        "M_c": 1e-9,
        "eta": 1e-6,
        "lambda_1": 0.05,
        "lambda_2": 0.05,
        "s1_z": 1e-6,
        "s2_z": 1e-6,
        "t_c": 1e-7,
        "ra": 1e-5,
        "dec": 1e-5,
        "psi": 2e-4,
        "iota": 2e-4,
        "d_L": 0.001,
    }
    factors = tuple(
        jnp.diag(jnp.asarray([scales[name] for name in block])) for block in blocks
    )
    keys = jax.random.split(jax.random.key(192), 2)
    results, evidence = [], {}
    for unroll, sampler in zip((None, 1, 2, 4), samplers, strict=True):
        step = sampler._build_constrained_step()
        fn = jax.vmap(
            lambda key, step=step: step(
                key,
                state,
                threshold,
                block_covariance_factors=factors,
            )
        )
        counts = _jaxpr_counts(fn, keys)
        started = time.perf_counter()
        lowered = jax.jit(fn).lower(keys)
        counts["stablehlo_characters"] = len(
            str(lowered.compiler_ir(dialect="stablehlo"))
        )
        compiled = lowered.compile()
        counts["lower_compile_seconds"] = time.perf_counter() - started
        results.append(jax.block_until_ready(compiled(keys)))
        evidence[str(unroll)] = counts
    expected_state, expected_info = results[0]
    errors = []
    for actual_state, actual_info in results[1:]:
        np.testing.assert_array_equal(actual_state.position, expected_state.position)
        np.testing.assert_array_equal(
            actual_state.logdensity, expected_state.logdensity
        )
        np.testing.assert_array_equal(
            actual_state.loglikelihood_birth, expected_state.loglikelihood_birth
        )
        _equal(actual_info, expected_info)
        error = float(
            np.max(abs(actual_state.loglikelihood - expected_state.loglikelihood))
        )
        errors.append(error)
        assert error < 5e-4
    assert expected_info.num_expansions.shape == (2, 96)
    assert np.asarray(expected_info.num_expansions).sum() > 0
    assert evidence["1"]["while"] < evidence["None"]["while"]
    evidence["maximum_score_error_nats"] = max(errors)
    evidence["scope"] = (
        "CPU constrained transitions, noisy 2--2048 Hz CE-A/ET, df=2 Hz; not a full inference timing"
    )
    (tmp_path / "physical-configured-sweep-compile.json").write_text(
        json.dumps(evidence, indent=2)
    )
    print("physical configured sweep evidence", json.dumps(evidence))
