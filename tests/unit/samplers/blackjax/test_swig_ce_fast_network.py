"""Combined CE scheduling optimizations on the transformed CE+ET sampler.

The full 2--2048 Hz band uses coarse 2 Hz native quadrature and deterministic
positive analytic test PSDs. This exercises the actual long emission clock,
finite-arm/orbital geometry, K8/M16, 21 anchors and 256-bin execution shape;
it does not qualify native-grid accuracy, posterior mixing or GPU performance.
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.base import StateWithLogLikelihood

from jimgw.cli._config import PipelineConfig
from jimgw.cli._jim import build_jim
from jimgw.cli._prior import build_prior
from jimgw.cli._transforms import infer_likelihood_transforms, infer_sample_transforms
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.samplers.blackjax import swig
from tests.unit.core.single_event.test_xg_network_evaluation import (
    _make_native_network,
    _options,
    _read_psd_tables,
)
from tests.xg_fixtures import network_config


def _same_tree(actual, expected, *, atol=0.0):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        left, right = np.asarray(left), np.asarray(right)
        assert left.shape == right.shape and left.dtype == right.dtype
        if atol and left.dtype.kind in "fc":
            np.testing.assert_allclose(left, right, rtol=0, atol=atol)
        else:
            np.testing.assert_array_equal(left, right)


def _same_source_cache(actual, expected):
    """Scale tolerances to carrier amplitude and the long emission clock."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    errors = {"maximum_clock_error_seconds": 0.0, "maximum_carrier_relative_error": 0.0}
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        left, right = np.asarray(left), np.asarray(right)
        assert left.shape == right.shape and left.dtype == right.dtype
        if left.dtype.kind == "c":
            # Absolute tolerances on ~1e-20 strain would otherwise accept zero.
            np.testing.assert_allclose(left, right, rtol=1e-8, atol=1e-34)
            relative = abs(left - right) / np.maximum(abs(right), 1e-26)
            errors["maximum_carrier_relative_error"] = max(
                errors["maximum_carrier_relative_error"], float(np.max(relative))
            )
        else:
            # Separate compilations can reassociate a ~78,000 second clock by
            # a few ULPs; the full likelihood is checked independently below.
            np.testing.assert_allclose(left, right, rtol=2e-14, atol=1e-12)
            errors["maximum_clock_error_seconds"] = max(
                errors["maximum_clock_error_seconds"], float(np.max(abs(left - right)))
            )
    return errors


def _problem(tmp_path):
    raw = network_config()
    tables = _read_psd_tables(tmp_path)
    for name, table in tables.items():
        path = tmp_path / f"{name}-analytic-psd.txt"
        np.savetxt(path, table)
        raw["data"]["psd_files"][name] = str(path)
    raw["data"]["sampling_frequency"] = 4096.0
    raw["likelihood"]["f_max"] = 2048.0
    edges = np.r_[2.0, 4.0, np.geomspace(8.0, 2048.0, 255)]
    heterodyne = raw["likelihood"]["heterodyne"]
    heterodyne.pop("bin_selection", None)
    heterodyne.update(
        n_bins=256, frequency_bin_edges=edges.tolist(), reference_chunk_size=256
    )
    raw["sampler"].update(
        n_devices=1,
        n_live=32,
        fsm_sweep_unroll=1,
        fsm_optimizations="auto",
        checkpoint_interval=0.0,
    )
    raw["output"]["dir"] = str(tmp_path / "unused")
    cfg = PipelineConfig.model_validate(raw, context={"prepare_xg_qualification": True})
    detectors, waveform = _make_native_network(tables)
    likelihood = HeterodynedTransientLikelihoodFD(
        **_options(detectors, waveform, edges, True)
    )
    prior = build_prior(cfg.prior)
    names = frozenset(prior.parameter_names)
    transforms = infer_sample_transforms(
        names, cfg.data.trigger_time, detectors, cfg.sampling, prior_cfg=cfg.prior
    )
    likelihood_transforms = infer_likelihood_transforms(
        names,
        cfg.data.trigger_time,
        detectors,
        cfg.sampling,
        cfg.waveform.f_ref,
        phase_marginalization=True,
    )
    return cfg, likelihood, prior, transforms, likelihood_transforms


def test_ce_fast_preserves_transformed_full_band_eight_sweep_network(
    tmp_path, monkeypatch, record_property
):
    with jax.enable_x64():
        cfg, likelihood, prior, transforms, likelihood_transforms = _problem(tmp_path)
        assert likelihood.interpolation_order == 8
        assert likelihood.phasor_moment_order == 16
        assert likelihood._xg_fast_evaluator.diagnostics["source_nodes"] == 2304
        jims = {}
        for mode in ("baseline", "auto", "ce-fast"):
            sampler_config = cfg.sampler.model_copy(update={"fsm_optimizations": mode})
            this_cfg = cfg.model_copy(update={"sampler": sampler_config})
            jims[mode] = build_jim(
                likelihood, prior, transforms, likelihood_transforms, this_cfg
            )
            sampler = jims[mode].sampler
            flags = sampler.fsm_optimization_diagnostics
            enabled = mode != "baseline"
            assert flags["skip_exhausted_endpoints"] is enabled
            assert flags["defer_rebuild_cache"] is enabled
            assert flags["implementation"] == ("ce-fast-v1" if enabled else "baseline")
            assert tuple(sampler._rebuild_required_by_block.values()) == (
                True,
                True,
                False,
                False,
            )
            assert [len(block) for block in sampler._rebuild_required_by_block] == [
                7,
                2,
                1,
                2,
            ]
            assert sampler._swig_config.fsm_sweep_unroll == 1
            assert sampler._swig_config.num_gibbs_sweeps == 8
            assert (
                jims[mode]._sampler_backend_kwargs["cache_independent_rebuild"] is True
            )

        jim = jims["baseline"]
        assert {"M_hat", "t_det", "d_hat", "cos_iota", "q"} <= set(
            jim.sampling_parameter_names
        )
        initial_points = []
        for change in ({}, {"ra": 2e-5, "t_c": 1e-7, "d_L": 0.001, "psi": 1e-4}):
            named = {
                name: jnp.asarray(
                    cfg.data.injection_parameters[name] + change.get(name, 0.0)
                )
                for name in prior.parameter_names
            }
            for transform in transforms:
                named = transform.forward(named)
            initial_points.append(
                jnp.asarray([named[name] for name in jim.sampling_parameter_names])
            )
        positions = jnp.stack(initial_points)
        assert not np.array_equal(positions[0], positions[1])
        logprior = jax.jit(jax.vmap(jim._log_prior_fn))(positions)
        loglikelihood = jax.jit(jax.vmap(jim._log_likelihood_fn))(positions)
        assert np.all(np.isfinite(logprior)) and np.all(np.isfinite(loglikelihood))
        thresholds = loglikelihood - jnp.asarray([10.0, 100.0])
        states = StateWithLogLikelihood(positions, logprior, loglikelihood, thresholds)
        scales = {
            "M_hat": 1e-9,
            "q": 2e-5,
            "lambda_1": 0.05,
            "lambda_2": 0.05,
            "s1_z": 1e-6,
            "s2_z": 1e-6,
            "t_det": 1e-7,
            "ra": 1e-5,
            "dec": 1e-5,
            "psi": 2e-4,
            "cos_iota": 2e-4,
            "d_hat": 1e-4,
        }
        # d_hat carries the network response normalization. Set its local scale
        # relative to this fixture rather than assuming a design-PSD magnitude.
        scales["d_hat"] = (
            float(abs(positions[0, jim.sampling_parameter_names.index("d_hat")])) * 5e-5
        )
        factors = tuple(
            jnp.diag(jnp.asarray([scales[name] for name in block]))
            for block in cfg.sampler.blocks
        )
        keys = jax.random.split(jax.random.key(32091), 2)

        # The public transition returns neither its cache nor its real segment
        # brackets. Observe that existing internal boundary in this test only;
        # no scheduler state or callback value is modified.
        original_scan = swig._run_swig_sweep_scan
        captures, timings, results = {}, {}, {}
        for mode in ("baseline", "auto"):
            captures[mode] = []

            def observed_scan(*args, _mode=mode, **kwargs):
                final, infos = original_scan(*args, **kwargs)
                jax.debug.callback(
                    lambda value: captures[_mode].append(
                        jax.tree.map(np.asarray, value)
                    ),
                    (final, infos),
                    ordered=True,
                )
                return final, infos

            monkeypatch.setattr(swig, "_run_swig_sweep_scan", observed_scan)
            step = jims[mode].sampler._build_constrained_step()
            fn = jax.jit(
                jax.vmap(
                    lambda key, state, threshold, _step=step: _step(
                        key, state, threshold, block_covariance_factors=factors
                    )
                )
            )
            started = time.perf_counter()
            compiled = fn.lower(keys, states, thresholds).compile()
            compile_seconds = time.perf_counter() - started
            print(f"{mode} lower/compile: {compile_seconds:.3f}s", flush=True)
            results[mode] = jax.block_until_ready(compiled(keys, states, thresholds))
            jax.effects_barrier()
            timings[mode] = {"lower_compile_seconds": compile_seconds}
            assert len(captures[mode]) == 2

        expected, actual = results["baseline"], results["auto"]
        _same_tree(actual[1], expected[1])
        np.testing.assert_array_equal(actual[0].position, expected[0].position)
        np.testing.assert_array_equal(actual[0].logdensity, expected[0].logdensity)
        np.testing.assert_array_equal(
            actual[0].loglikelihood_birth, expected[0].loglikelihood_birth
        )
        np.testing.assert_allclose(
            actual[0].loglikelihood, expected[0].loglikelihood, rtol=0, atol=5e-4
        )
        assert actual[1].num_expansions.shape == (2, 96)
        assert np.asarray(actual[1].num_expansions).sum() > 0
        assert np.all(np.asarray(actual[1].is_accepted))
        cache_errors = []
        for baseline, optimized in zip(
            captures["baseline"], captures["auto"], strict=True
        ):
            base_state, base_info = baseline
            fast_state, fast_info = optimized
            _same_tree(fast_info, base_info)
            cache_errors.append(_same_source_cache(fast_state.cache, base_state.cache))
            # These are actual nontrivial brackets, not the public placeholders.
            assert any(np.any(np.asarray(info.bracket_right) > 0) for info in fast_info)
        fresh = jax.jit(jax.vmap(jim._log_likelihood_fn))(actual[0].position)
        error = float(jnp.max(abs(fresh - actual[0].loglikelihood)))
        assert np.all(np.isfinite(fresh))
        assert error < 5e-4
        fresh_caches = jax.jit(jax.vmap(jims["auto"].sampler._build_cache))(
            actual[0].position
        )
        for lane in range(2):
            final_cache = captures["auto"][lane][0].cache
            rebuilt = jax.tree.map(lambda value, _lane=lane: value[_lane], fresh_caches)
            cache_errors.append(_same_source_cache(final_cache, rebuilt))
        receipt = {
            "scope": __doc__,
            "lanes": 2,
            "sweeps": 8,
            "slices_per_lane": 96,
            "bins": 256,
            "nodes": 2304,
            "maximum_saved_fresh_error_nats": error,
            "source_cache_errors": cache_errors,
            "lower_compile_seconds_diagnostic_CPU_only": timings,
            "optimization_diagnostics": jims[
                "auto"
            ].sampler.fsm_optimization_diagnostics,
        }
        (tmp_path / "ce-fast-network-parity.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        record_property("maximum_saved_fresh_error_nats", error)
        print("ce-fast network parity", json.dumps(receipt))
