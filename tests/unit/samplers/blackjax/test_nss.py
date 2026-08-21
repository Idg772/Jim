"""Smoke test: BlackJAXNSSSampler on a 2-D Gaussian."""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import init_state_strategy

blackjax = pytest.importorskip("blackjax")

from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.samplers.blackjax.nss import BlackJAXNSSSampler
from jimgw.samplers.config import BlackJAXNSSConfig

_SIGMA = 0.05
_MU = 0.5


class _GaussianLikelihood:
    def evaluate(self, params: dict) -> float:
        x = params["x"]
        y = params["y"]
        return -0.5 * ((x - _MU) ** 2 + (y - _MU) ** 2) / _SIGMA**2


def _make_sampler(n_live: int = 100) -> BlackJAXNSSSampler:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    config = BlackJAXNSSConfig(
        n_live=n_live,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=5,
        termination_dlogz=0.5,
    )
    parameter_names = prior.parameter_names  # ("x", "y")

    def log_prior_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return prior.log_prob(named)

    def log_likelihood_fn(arr):
        named = dict(zip(parameter_names, arr, strict=True))
        return likelihood.evaluate(named)

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    return BlackJAXNSSSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )


def test_nss_construction():
    sampler = _make_sampler()
    assert sampler.n_dims == 2


def test_nss_fsm_update_params_returns_one_cholesky_factor(monkeypatch):
    sampler = _make_sampler(n_live=5)
    positions = jnp.asarray(
        [
            [-2.0, -1.0],
            [-1.0, 0.5],
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, -2.0],
        ]
    )
    state = SimpleNamespace(particles=SimpleNamespace(position=positions))
    covariance = jnp.cov(positions, ddof=0, rowvar=False)

    original_cholesky = jnp.linalg.cholesky
    factorized_shapes = []

    def recording_cholesky(matrix):
        factorized_shapes.append(matrix.shape)
        return original_cholesky(matrix)

    monkeypatch.setattr(jnp.linalg, "cholesky", recording_cholesky)
    params = sampler._fsm_update_inner_kernel_params_fn(
        jax.random.key(0),
        state,
        None,
    )

    assert set(params) == {"covariance_factor"}
    assert factorized_shapes == [(2, 2)]
    factor = params["covariance_factor"]
    np.testing.assert_allclose(factor @ factor.T, covariance, rtol=1e-6)
    np.testing.assert_array_equal(factor, jnp.tril(factor))

    legacy_params = sampler._update_inner_kernel_params_fn(
        jax.random.key(1),
        state,
        None,
    )
    assert set(legacy_params) == {"cov"}


def test_nss_factor_checkpoint_resumes_on_nonmesh_path(tmp_path, monkeypatch):
    monkeypatch.setattr(BlackJAXNSSConfig, "configure_jax_cache", lambda self: None)
    n_live = 8

    def log_prior(position):
        return jnp.where(jnp.all((position >= 0.0) & (position <= 1.0)), 0.0, -jnp.inf)

    def log_likelihood(position):
        return -20.0 * jnp.sum((position - 0.5) ** 2)

    config = BlackJAXNSSConfig(
        n_live=n_live,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=1,
        termination_dlogz=1e6,
        checkpoint_dir=tmp_path,
        checkpoint_interval=60.0,
    )
    sampler = BlackJAXNSSSampler(
        n_dims=2,
        log_prior_fn=log_prior,
        log_likelihood_fn=log_likelihood,
        log_posterior_fn=lambda x: log_prior(x) + log_likelihood(x),
        config=config,
    )
    positions = jax.random.uniform(jax.random.key(20), (n_live, 2))
    single_init = partial(
        init_state_strategy,
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
    )
    factor_state = adaptive_init(
        positions,
        init_state_fn=jax.vmap(single_init),
        update_inner_kernel_params_fn=sampler._fsm_update_inner_kernel_params_fn,
    )
    checkpoint = {
        "state": jax.device_get(factor_state),
        "dead": [],
        "rng_key": jax.device_get(jax.random.key(21)),
        "n_iter": 0,
        "sampler_name": sampler.sampler_name,
        "elapsed_time": 17.0,
    }
    with (tmp_path / "checkpoint.pkl").open("wb") as stream:
        pickle.dump(checkpoint, stream)

    sampler.sample(jax.random.key(999), jnp.zeros_like(positions))

    assert sampler._prev_elapsed == 17.0
    assert sampler.get_diagnostics()["n_iterations"] > 0
    assert not (tmp_path / "checkpoint.pkl").exists()


def test_nss_get_samples_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_samples()


def test_nss_get_weighted_samples_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_weighted_samples()


def _init_pos(n_live: int, seed: int = 99) -> jax.Array:
    return jax.random.uniform(jax.random.key(seed), (n_live, 2))


def test_nss_sample_and_get_samples():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(0), _init_pos(100))
    result = sampler.get_samples()
    assert isinstance(result, dict)
    assert "samples" in result
    assert "log_likelihood" in result


def test_nss_samples_fields():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(1), _init_pos(100))
    result = sampler.get_samples()

    assert isinstance(result["samples"], np.ndarray)
    assert result["samples"].ndim == 2
    assert result["samples"].shape[1] == 2
    n = result["samples"].shape[0]
    assert n > 0
    assert result["log_likelihood"].shape == (n,)


def test_nss_samples_in_prior_support():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(2), _init_pos(100))
    result = sampler.get_samples()

    assert np.all(result["samples"][:, 0] >= 0.0) and np.all(
        result["samples"][:, 0] <= 1.0
    )
    assert np.all(result["samples"][:, 1] >= 0.0) and np.all(
        result["samples"][:, 1] <= 1.0
    )


def test_nss_diagnostics_before_sample_raises():
    sampler = _make_sampler()
    with pytest.raises(RuntimeError, match="before sample"):
        sampler.get_diagnostics()


def test_nss_diagnostics():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(4), _init_pos(100))
    diag = sampler.get_diagnostics()

    assert isinstance(diag, dict)
    assert diag["n_iterations"] > 0
    assert diag["n_stepping_out_history"] is not None
    assert diag["n_shrinking_history"] is not None
    assert diag["n_likelihood_evaluations_stepping_out"] is not None
    assert diag["n_likelihood_evaluations_shrinking"] is not None
    assert diag["n_likelihood_evaluations"] == (
        diag["n_likelihood_evaluations_stepping_out"]
        + diag["n_likelihood_evaluations_shrinking"]
    )
    assert "log_Z" in diag
    assert "log_Z_error" in diag
    assert np.isfinite(diag["log_Z"])
    assert "sampling_time" in diag
    assert diag["sampling_time"] >= 0.0


def test_nss_sample_phase_seconds():
    sampler = _make_sampler()
    sampler.sample(jax.random.key(4), _init_pos(100))
    diag = sampler.get_diagnostics()

    phases = diag["sample_phase_seconds"]
    assert set(phases) == {
        "init_total",
        "init_adaptive_init",
        "likelihood_jit",
        "initial_likelihood_eval",
        "sampler_kernel_jit",
        "ns_loop",
        "finalise",
    }
    assert phases["init_total"] is not None and phases["init_total"] > 0.0
    assert (
        phases["init_adaptive_init"] is not None and phases["init_adaptive_init"] >= 0.0
    )
    assert phases["ns_loop"] is not None and phases["ns_loop"] > 0.0
    assert phases["finalise"] is not None and phases["finalise"] >= 0.0

    # The compile/eval split exists only on the sharded (mesh) path; on the
    # single-device CPU path both must be None together.
    jit_s = phases["likelihood_jit"]
    eval_s = phases["initial_likelihood_eval"]
    assert (jit_s is None) == (eval_s is None)
    if jit_s is not None:
        assert jit_s > 0.0 and eval_s > 0.0
        assert jit_s + eval_s <= phases["init_total"] + 1e-6

    sampler_jit_s = phases["sampler_kernel_jit"]
    assert sampler_jit_s is not None and sampler_jit_s > 0.0

    # Phases must fit inside the overall reported sampling time (small
    # slack for the un-timed terminate checks between steps).
    accounted = (
        phases["init_total"] + sampler_jit_s + phases["ns_loop"] + phases["finalise"]
    )
    assert accounted <= diag["sampling_time"] + 0.5


def test_nss_checkpoint_file_created(tmp_path, monkeypatch):
    """Checkpoint .pkl is written during sampling and cleaned up on success."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names
    config = BlackJAXNSSConfig(
        n_live=100,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=5,
        termination_dlogz=0.5,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1e-9,
    )

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = BlackJAXNSSSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    ckpt_path = tmp_path / "checkpoint.pkl"
    _orig_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None if self == ckpt_path else _orig_unlink(self, missing_ok=missing_ok)
        ),
    )
    sampler.sample(jax.random.key(42), _init_pos(100))
    monkeypatch.setattr(Path, "unlink", _orig_unlink)
    assert ckpt_path.exists(), "Checkpoint was never written"
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    assert "elapsed_time" in ckpt
    assert ckpt["elapsed_time"] >= 0.0
    assert ckpt["sampler_name"] == sampler.sampler_name
    ckpt_path.unlink()


def test_nss_falls_back_to_fresh_run_on_foreign_checkpoint(tmp_path):
    """A checkpoint written by a different sampler is treated like a corrupt
    one: NSS logs a warning and starts fresh rather than raising.

    Unlike flowMC (which validates the checkpoint before entering its resume
    try/except and so raises), NSS/NS AW/SMC validate *inside* the same
    try/except that already catches corrupt-checkpoint errors, so a foreign
    ``sampler_name`` is swallowed the same way.
    """
    config = BlackJAXNSSConfig(
        n_live=20,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=5,
        termination_dlogz=2.0,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1e-9,
    )
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = BlackJAXNSSSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    ckpt_path = tmp_path / "checkpoint.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({"sampler_name": "BlackJAX SwiG"}, f)

    sampler.sample(jax.random.key(0), _init_pos(20))
    result = sampler.get_samples()
    assert "samples" in result


def test_nss_checkpoint_failure_restores_caller_rng_key(tmp_path):
    """A checkpoint that fails *after* its rng_key is read falls back to the
    caller-supplied key, not the partially-loaded checkpoint's key.
    """
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names

    def _make(checkpoint_dir=None):
        config = BlackJAXNSSConfig(
            n_live=20,
            n_delete_frac=0.5,
            num_inner_steps_per_dim=5,
            termination_dlogz=0.5,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
        )

        def log_prior_fn(arr):
            return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

        def log_likelihood_fn(arr):
            return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

        def log_posterior_fn(arr):
            return log_prior_fn(arr) + log_likelihood_fn(arr)

        return BlackJAXNSSSampler(
            n_dims=len(parameter_names),
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )

    caller_key = jax.random.key(7)

    reference = _make(checkpoint_dir=None)
    reference.sample(caller_key, _init_pos(20))
    log_z_reference = reference.get_diagnostics()["log_Z"]

    sampler = _make(checkpoint_dir=tmp_path)
    ckpt_path = tmp_path / "checkpoint.pkl"
    # Valid enough to pass `_validate_checkpoint` and overwrite `rng_key` with
    # a decoy key, but missing "n_iter" so loading fails right after.
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {
                "sampler_name": sampler.sampler_name,
                "state": None,
                "dead": None,
                "rng_key": jax.random.key(999),
            },
            f,
        )

    sampler.sample(caller_key, _init_pos(20))

    assert sampler.get_diagnostics()["log_Z"] == pytest.approx(
        log_z_reference, rel=1e-6
    )


def test_nss_resume_gives_same_result(tmp_path, monkeypatch):
    """A run resumed from a crashed checkpoint gives the same log_Z as an uninterrupted run."""
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names

    def _make(checkpoint_dir=None):
        config = BlackJAXNSSConfig(
            n_live=100,
            n_delete_frac=0.5,
            num_inner_steps_per_dim=5,
            termination_dlogz=0.5,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=1e-9 if checkpoint_dir is not None else 0.0,
        )

        def log_prior_fn(arr):
            return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

        def log_likelihood_fn(arr):
            return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

        def log_posterior_fn(arr):
            return log_prior_fn(arr) + log_likelihood_fn(arr)

        return BlackJAXNSSSampler(
            n_dims=len(parameter_names),
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            log_posterior_fn=log_posterior_fn,
            config=config,
        )

    s_a = _make(checkpoint_dir=None)
    s_a.sample(jax.random.key(0), _init_pos(100))
    log_z_a = s_a.get_diagnostics()["log_Z"]

    # Run B: suppress deletion of the checkpoint file only (simulates a crash leaving it behind).
    ckpt_path = tmp_path / "checkpoint.pkl"
    _orig_unlink = Path.unlink
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (
            None if self == ckpt_path else _orig_unlink(self, missing_ok=missing_ok)
        ),
    )
    s_b = _make(checkpoint_dir=tmp_path)
    s_b.sample(jax.random.key(0), _init_pos(100))
    monkeypatch.setattr(Path, "unlink", _orig_unlink)
    assert ckpt_path.exists(), "Checkpoint was never written"

    # Run C: resumes from B's checkpoint → same log_Z. Deletes checkpoint on success.
    s_c = _make(checkpoint_dir=tmp_path)
    s_c.sample(jax.random.key(0), _init_pos(100))

    assert s_c.get_diagnostics()["log_Z"] == pytest.approx(log_z_a, rel=1e-6)
    assert not (tmp_path / "checkpoint.pkl").exists(), "Checkpoint was not cleaned up"


def test_termination_reached_is_nan_safe_and_single_valued(caplog):
    """Task 10: ``_termination_reached`` is a single-sync, NaN-safe predicate.

    The previous ``bool(jnp.isfinite(dlogz) and dlogz < threshold)`` never
    terminated when ``dlogz`` was NaN, because Python's ``and`` returns the
    falsy left-hand array without evaluating the right-hand comparison.

    ``dlogz`` is only ever non-finite as ``+inf`` (the expected value at
    t=0, since blackjax's integrator initialises ``logZ=-inf`` with a
    finite ``logZ_live``) or ``NaN`` (indeterminate ``inf - inf``, e.g. both
    ``logZ_live`` and ``logZ`` simultaneously ``-inf``).  Only the NaN case
    must terminate; treating ``+inf`` as terminal would end every run after
    zero iterations.
    """
    from jimgw.samplers.blackjax.nss import _termination_reached

    # t=0 initial state: logZ=-inf, logZ_live finite -> dlogz=+inf.
    # Must NOT terminate (or every real run would stop after zero steps).
    initial_state = SimpleNamespace(logZ_live=jnp.array(-3.2), logZ=jnp.array(-jnp.inf))
    with caplog.at_level("WARNING"):
        result = _termination_reached(initial_state, termination_dlogz=0.5)
    assert result is False
    assert isinstance(result, bool)
    assert not caplog.records

    # logZ_live == logZ == -inf -> logaddexp(0, nan) -> dlogz is NaN.
    caplog.clear()
    nan_integrator = SimpleNamespace(
        logZ_live=jnp.array(-jnp.inf), logZ=jnp.array(-jnp.inf)
    )
    with caplog.at_level("WARNING"):
        result = _termination_reached(nan_integrator, termination_dlogz=0.5)
    assert result is True  # NaN dlogz now terminates (semantics change)
    assert isinstance(result, bool)
    assert any("nan" in rec.message.lower() for rec in caplog.records)

    # Finite dlogz well above the threshold: must not terminate.
    caplog.clear()
    non_terminal = SimpleNamespace(logZ_live=jnp.array(0.0), logZ=jnp.array(-100.0))
    with caplog.at_level("WARNING"):
        result = _termination_reached(non_terminal, termination_dlogz=0.5)
    assert result is False
    assert isinstance(result, bool)
    assert not caplog.records

    # Finite dlogz below the threshold: terminates, no warning.
    caplog.clear()
    terminal = SimpleNamespace(logZ_live=jnp.array(-100.0), logZ=jnp.array(0.0))
    with caplog.at_level("WARNING"):
        result = _termination_reached(terminal, termination_dlogz=0.5)
    assert result is True
    assert isinstance(result, bool)
    assert not caplog.records


def test_finalise_on_host_matches_blackjax_finalise():
    """Task 11: ``_finalise_on_host`` is value-identical to blackjax's finalise."""
    from blackjax.mcmc.slice import SliceInfo
    from blackjax.ns.base import NSInfo, StateWithLogLikelihood
    from blackjax.ns.utils import finalise as blackjax_finalise

    from jimgw.samplers.blackjax.nss import _finalise_on_host

    rng = np.random.default_rng(0)

    def _make_particles(n: int) -> StateWithLogLikelihood:
        return StateWithLogLikelihood(
            position=jnp.asarray(rng.normal(size=(n, 3))),
            logdensity=jnp.asarray(rng.normal(size=(n,))),
            loglikelihood=jnp.asarray(rng.normal(size=(n,))),
            loglikelihood_birth=jnp.asarray(rng.normal(size=(n,))),
        )

    def _make_update_info(n: int) -> SliceInfo:
        return SliceInfo(
            is_accepted=jnp.asarray(rng.integers(0, 2, size=(n,))).astype(bool),
            num_expansions=jnp.asarray(rng.integers(0, 5, size=(n,))),
            num_shrink=jnp.asarray(rng.integers(0, 5, size=(n,))),
            bracket_left=jnp.asarray(rng.normal(size=(n, 3))),
            bracket_right=jnp.asarray(rng.normal(size=(n, 3))),
        )

    n_delete = 2
    dead = [
        NSInfo(
            particles=_make_particles(n_delete),
            update_info=_make_update_info(n_delete),
        )
        for _ in range(3)
    ]
    state = SimpleNamespace(particles=_make_particles(4))

    expected = blackjax_finalise(state, dead)
    actual = _finalise_on_host(state, dead)

    expected_leaves, expected_tree = jax.tree.flatten(expected)
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    assert expected_tree == actual_tree
    assert len(expected_leaves) == len(actual_leaves)
    for expected_leaf, actual_leaf in zip(expected_leaves, actual_leaves, strict=True):
        assert isinstance(actual_leaf, np.ndarray)
        assert np.array_equal(np.asarray(expected_leaf), actual_leaf)

    with pytest.raises(ValueError, match="finalise requires"):
        _finalise_on_host(state, [])


def test_nss_init_phase_seconds_keys_on_sharded_run():
    """Task 12: mesh-path init sub-phase timers appear with non-negative values."""
    if jax.local_device_count() < 2:
        pytest.skip(
            "requires >=2 local JAX devices "
            "(e.g. XLA_FLAGS=--xla_force_host_platform_device_count=2)"
        )

    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["x"]),
            UniformPrior(0.0, 1.0, parameter_names=["y"]),
        ]
    )
    likelihood = _GaussianLikelihood()
    parameter_names = prior.parameter_names
    config = BlackJAXNSSConfig(
        n_live=8,
        n_delete_frac=0.5,
        num_inner_steps_per_dim=1,
        termination_dlogz=2.0,
        n_devices=2,
    )

    def log_prior_fn(arr):
        return prior.log_prob(dict(zip(parameter_names, arr, strict=True)))

    def log_likelihood_fn(arr):
        return likelihood.evaluate(dict(zip(parameter_names, arr, strict=True)))

    def log_posterior_fn(arr):
        return log_prior_fn(arr) + log_likelihood_fn(arr)

    sampler = BlackJAXNSSSampler(
        n_dims=len(parameter_names),
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        log_posterior_fn=log_posterior_fn,
        config=config,
    )
    sampler.sample(jax.random.key(7), _init_pos(8))
    phases = sampler.get_diagnostics()["sample_phase_seconds"]

    for key in (
        "init_adaptive_init",
        "init_particle_replication",
        "init_state_placement",
    ):
        assert key in phases, f"missing phase key {key!r}: {sorted(phases)}"
        assert phases[key] is not None
        assert phases[key] >= 0.0

    # Pre-existing keys are untouched by the new instrumentation.
    for key in (
        "init_total",
        "likelihood_jit",
        "initial_likelihood_eval",
        "sampler_kernel_jit",
        "ns_loop",
        "finalise",
    ):
        assert key in phases
