"""Focused Jim/config integration tests for quotient folding."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jimgw.core.jim as jim_module
from jimgw.core.folding import (
    _build_fold_images,
    _fold_to_fundamental,
    _folded_log_likelihood,
    _folded_log_likelihood_from_cache,
    _folded_log_posterior,
    _folded_log_prior,
    _in_fundamental_domain,
)
from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.likelihood import SingleEventLikelihood
from jimgw.samplers.config import BlackJAXSwiGConfig, FoldSymmetryConfig

_FOLD_NAMES = ("cos_iota", "azimuth", "psi")
_PARAMETER_NAMES = ("slow", *_FOLD_NAMES)
_PERIODIC = {"azimuth": (0.0, 2.0 * np.pi), "psi": (0.0, np.pi)}


@dataclass(frozen=True)
class _DetectorSite:
    """Only the detector-site surface used by fold validation."""

    name: str
    vertex: jax.Array


class _CacheResidentWaveform:
    """Minimal waveform declaration for cache-dependency inference."""

    parameter_names = _PARAMETER_NAMES

    def __init__(self, cacheable_parameter_names: tuple[str, ...]) -> None:
        self.cacheable_parameter_names = cacheable_parameter_names

    def build_waveform_cache(self, frequencies: Any, params: dict[str, Any]) -> Any:
        del frequencies, params
        raise AssertionError("The lightweight likelihood owns its cache in this test")

    def waveform_from_cache(
        self,
        frequencies: Any,
        cache: Any,
        params: dict[str, Any],
    ) -> Any:
        del frequencies, cache, params
        raise AssertionError("The lightweight likelihood owns its cache in this test")


class _ToySingleEventLikelihood(SingleEventLikelihood):
    """Finite analytic likelihood with observable cache/projection calls."""

    def __init__(
        self,
        *,
        cacheable_parameter_names: tuple[str, ...] = _FOLD_NAMES,
        phase_marginalization: bool = True,
        detectors: tuple[_DetectorSite, ...] | None = None,
    ) -> None:
        # Deliberately avoid SingleEventLikelihood.__init__: real detector data
        # and PSDs are irrelevant to this Jim wiring test.
        self.waveform = _CacheResidentWaveform(cacheable_parameter_names)
        self.fixed_parameters: dict[str, Any] = {}
        self.trigger_time = 0.0
        self.gmst = jnp.asarray(0.0)
        self.ref_dist = 1.0
        self.time_marginalization = False
        self.phase_marginalization = phase_marginalization
        self.distance_marginalization = False
        self.detectors = detectors or _valid_detector_sites()
        self.cache_build_positions: list[np.ndarray] = []
        self.projection_positions: list[np.ndarray] = []

    @staticmethod
    def _true_log_likelihood(params: dict[str, Any], cache: Any) -> jax.Array:
        return (
            0.25 * jnp.asarray(cache)
            + 0.4 * params["cos_iota"]
            + jnp.cos(params["azimuth"])
            + 0.2 * jnp.sin(2.0 * params["psi"])
        )

    def _evaluate(self, params: dict[str, Any]) -> jax.Array:
        return self._true_log_likelihood(params, params["slow"])

    def _generate_waveform(self, params: dict[str, Any]) -> jax.Array:
        position = jnp.stack(tuple(params[name] for name in _PARAMETER_NAMES))
        jax.debug.callback(self.cache_build_positions.append, position)
        return jnp.asarray(params["slow"])

    def _evaluate_from_waveform(
        self,
        params: dict[str, Any],
        waveform_cache: Any,
    ) -> jax.Array:
        position = jnp.stack(tuple(params[name] for name in _PARAMETER_NAMES))
        jax.debug.callback(self.projection_positions.append, position)
        return self._true_log_likelihood(params, waveform_cache)

    def clear_call_history(self) -> None:
        self.cache_build_positions.clear()
        self.projection_positions.clear()


class _CapturingSampler:
    """Sampler boundary double that records exactly what Jim wires into it."""

    def __init__(self, **kwargs: Any) -> None:
        self._log_prior_fn = kwargs["log_prior_fn"]
        self._log_likelihood_fn = kwargs["log_likelihood_fn"]
        self._log_posterior_fn = kwargs["log_posterior_fn"]
        self._build_cache = kwargs["build_cache"]
        self._log_likelihood_from_cache_fn = kwargs["log_likelihood_from_cache_fn"]
        self.periodic = kwargs["periodic"]
        self.sampled_positions: list[np.ndarray] = []
        self.get_samples_calls = 0
        self.get_weighted_samples_calls = 0
        self.raw_weighted_samples = {
            "samples": np.asarray([[0.2, 0.5, 1.0, 0.3]]),
            "log_likelihood": np.asarray([-1.2]),
            "log_likelihood_birth": np.asarray([-2.0]),
            "log_weights": np.asarray([0.0]),
        }

    def sample(self, rng_key: jax.Array, initial_position: jax.Array) -> None:
        del rng_key
        self.sampled_positions.append(np.asarray(initial_position))

    def get_samples(self) -> dict[str, np.ndarray]:
        self.get_samples_calls += 1
        return {
            "samples": self.raw_weighted_samples["samples"],
            "log_likelihood": self.raw_weighted_samples["log_likelihood"],
        }

    def get_weighted_samples(self) -> dict[str, np.ndarray]:
        self.get_weighted_samples_calls += 1
        return self.raw_weighted_samples


@pytest.fixture
def capturing_sampler(monkeypatch: pytest.MonkeyPatch) -> list[_CapturingSampler]:
    built: list[_CapturingSampler] = []

    def build_sampler(config: Any, **kwargs: Any) -> _CapturingSampler:
        del config
        sampler = _CapturingSampler(**kwargs)
        built.append(sampler)
        return sampler

    monkeypatch.setattr(jim_module, "build_sampler", build_sampler)
    return built


def _valid_detector_sites() -> tuple[_DetectorSite, ...]:
    return (
        _DetectorSite("A", jnp.asarray([0.0, 0.0, 0.0])),
        _DetectorSite("B", jnp.asarray([3.0, 0.0, 0.0])),
        _DetectorSite("C", jnp.asarray([0.0, 4.0, 1.0])),
    )


def _prior() -> CombinePrior:
    prior = CombinePrior(
        [
            UniformPrior(0.0, 1.0, parameter_names=["slow"]),
            UniformPrior(-1.0, 1.0, parameter_names=["cos_iota"]),
            UniformPrior(0.0, 2.0 * np.pi, parameter_names=["azimuth"]),
            UniformPrior(0.0, np.pi, parameter_names=["psi"]),
        ]
    )
    assert prior.is_normalized
    return prior


def _fold_config(**overrides: Any) -> FoldSymmetryConfig:
    values = {
        "cos_iota": "cos_iota",
        "azimuth": "azimuth",
        "psi": "psi",
        "azimuth_reflection_center": 0.37,
    }
    values.update(overrides)
    return FoldSymmetryConfig(**values)


def _swig_config(
    *,
    fold_symmetry: FoldSymmetryConfig | None,
    scheduler: str = "fsm",
) -> BlackJAXSwiGConfig:
    return BlackJAXSwiGConfig(
        blocks=[["slow"], ["cos_iota", "azimuth", "psi"]],
        fold_symmetry=fold_symmetry,
        scheduler=scheduler,
        n_live=4,
        n_delete_frac=0.5,
        num_gibbs_sweeps=1,
        max_steps=2,
        max_shrinkage=2,
    )


def _folded_jim(
    *,
    likelihood: _ToySingleEventLikelihood | None = None,
    fold_config: FoldSymmetryConfig | None = None,
    periodic: dict[str, tuple[float, float]] | None = _PERIODIC,
) -> Jim:
    return Jim(
        likelihood=likelihood or _ToySingleEventLikelihood(),
        prior=_prior(),
        sampler_config=_swig_config(
            fold_symmetry=fold_config if fold_config is not None else _fold_config()
        ),
        periodic=periodic,
    )


def _sampling_array(prior: CombinePrior, key: jax.Array, n: int) -> jax.Array:
    named = prior.sample(key, n)
    return jnp.stack(tuple(named[name] for name in _PARAMETER_NAMES), axis=1)


def test_fold_symmetry_config_validates_its_coordinate_contract() -> None:
    config = _fold_config()
    assert (config.cos_iota, config.azimuth, config.psi) == _FOLD_NAMES
    assert config.azimuth_reflection_center == pytest.approx(0.37)

    for overrides, message in (
        ({"cos_iota": ""}, "empty"),
        ({"azimuth": "cos_iota"}, "distinct"),
        ({"azimuth_reflection_center": np.nan}, "finite"),
        ({"azimuth_reflection_center": np.inf}, "finite"),
    ):
        with pytest.raises(ValueError) as exc_info:
            _fold_config(**overrides)
        assert message in str(exc_info.value).lower()

    with pytest.raises(ValueError):
        FoldSymmetryConfig(
            **{
                **_fold_config().model_dump(),
                "unexpected_coordinate": "phase_c",
            }
        )


def test_fold_symmetry_is_default_off_and_requires_fsm() -> None:
    assert _swig_config(fold_symmetry=None).fold_symmetry is None

    with pytest.raises(ValueError) as exc_info:
        _swig_config(
            fold_symmetry=_fold_config(),
            scheduler="pre-fsm-lockstep",
        )
    assert "fsm" in str(exc_info.value).lower()


@pytest.mark.parametrize("coordinate", _FOLD_NAMES)
def test_jim_rejects_fold_coordinate_names_outside_sampling_space(
    coordinate: str,
    capturing_sampler: list[_CapturingSampler],
) -> None:
    del capturing_sampler
    with pytest.raises(ValueError) as exc_info:
        _folded_jim(fold_config=_fold_config(**{coordinate: "missing"}))
    message = str(exc_info.value).lower()
    assert "missing" in message
    assert "sampling" in message


@pytest.mark.parametrize("missing_cache_coordinate", _FOLD_NAMES)
def test_jim_requires_every_fold_coordinate_to_be_cache_resident(
    missing_cache_coordinate: str,
    capturing_sampler: list[_CapturingSampler],
) -> None:
    del capturing_sampler
    cacheable = tuple(name for name in _FOLD_NAMES if name != missing_cache_coordinate)
    likelihood = _ToySingleEventLikelihood(cacheable_parameter_names=cacheable)

    with pytest.raises(ValueError) as exc_info:
        _folded_jim(likelihood=likelihood)
    message = str(exc_info.value).lower()
    assert missing_cache_coordinate in message
    assert "cache" in message


def test_jim_requires_phase_marginalization_for_folding(
    capturing_sampler: list[_CapturingSampler],
) -> None:
    del capturing_sampler
    likelihood = _ToySingleEventLikelihood(phase_marginalization=False)

    with pytest.raises(ValueError) as exc_info:
        _folded_jim(likelihood=likelihood)
    assert "phase" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    ("detectors", "message"),
    [
        (_valid_detector_sites()[:2], "exactly"),
        (
            (
                _DetectorSite("A", jnp.asarray([0.0, 0.0, 0.0])),
                _DetectorSite("B", jnp.asarray([1.0, 0.0, 0.0])),
                _DetectorSite("C", jnp.asarray([2.0, 0.0, 0.0])),
            ),
            "collinear",
        ),
        (
            (
                *_valid_detector_sites()[:2],
                _DetectorSite("C", jnp.asarray([0.0, np.nan, 1.0])),
            ),
            "finite",
        ),
    ],
)
def test_jim_rejects_invalid_fold_detector_geometry(
    detectors: tuple[_DetectorSite, ...],
    message: str,
    capturing_sampler: list[_CapturingSampler],
) -> None:
    del capturing_sampler
    likelihood = _ToySingleEventLikelihood(detectors=detectors)

    with pytest.raises(ValueError) as exc_info:
        _folded_jim(likelihood=likelihood)
    assert message in str(exc_info.value).lower()


@pytest.mark.parametrize(
    ("periodic", "coordinate"),
    [
        (None, "periodic"),
        ({"azimuth": (0.0, 2.0 * np.pi)}, "psi"),
        (
            {"azimuth": (-np.pi, np.pi), "psi": (0.0, np.pi)},
            "azimuth",
        ),
        (
            {"azimuth": (0.0, 2.0 * np.pi), "psi": (0.0, 2.0 * np.pi)},
            "psi",
        ),
    ],
)
def test_jim_requires_canonical_periods_for_fold_coordinates(
    periodic: dict[str, tuple[float, float]] | None,
    coordinate: str,
    capturing_sampler: list[_CapturingSampler],
) -> None:
    del capturing_sampler
    with pytest.raises(ValueError) as exc_info:
        _folded_jim(periodic=periodic)
    assert coordinate in str(exc_info.value).lower()


def test_folded_jim_preserves_base_callbacks_and_matches_pure_helpers(
    capturing_sampler: list[_CapturingSampler],
) -> None:
    jim = _folded_jim()
    sampler = capturing_sampler[-1]
    fold = jim._resolved_fold_symmetry
    assert fold.indices == (1, 2, 3)
    assert fold.azimuth_reflection_center == pytest.approx(0.37)
    assert sampler.periodic == {2: _PERIODIC["azimuth"], 3: _PERIODIC["psi"]}
    position = _fold_to_fundamental(
        jnp.asarray([0.4, -0.6, 5.2, 2.4]),
        fold,
    )

    assert jim._base_log_prior_fn is not jim._log_prior_fn
    assert jim._base_log_likelihood_fn is not jim._log_likelihood_fn
    assert jim._base_log_posterior_fn is not jim._log_posterior_fn
    assert sampler._log_prior_fn is jim._log_prior_fn
    assert sampler._log_likelihood_fn is jim._log_likelihood_fn
    assert sampler._log_posterior_fn is jim._log_posterior_fn
    assert sampler._build_cache is jim._base_build_cache
    assert (
        jim._sampler_backend_kwargs["log_likelihood_from_cache_fn"]
        is sampler._log_likelihood_from_cache_fn
    )

    named_position = jim.add_name(position)
    base_prior = jim.prior.log_prob(named_position)
    base_likelihood = jim.likelihood.evaluate(named_position)
    assert jim._base_log_prior_fn(position) == pytest.approx(float(base_prior))
    assert jim._base_log_likelihood_fn(position) == pytest.approx(
        float(base_likelihood)
    )
    assert jim._base_log_posterior_fn(position) == pytest.approx(
        float(base_prior + base_likelihood)
    )

    assert jim._log_prior_fn(position) == pytest.approx(
        float(_folded_log_prior(position, fold, jim._base_log_prior_fn))
    )
    assert jim._log_likelihood_fn(position) == pytest.approx(
        float(
            _folded_log_likelihood(
                position,
                fold,
                jim._base_log_prior_fn,
                jim._base_log_likelihood_fn,
            )
        )
    )
    assert jim._log_posterior_fn(position) == pytest.approx(
        float(
            _folded_log_posterior(
                position,
                fold,
                jim._base_log_prior_fn,
                jim._base_log_likelihood_fn,
            )
        )
    )

    cache = jim._base_build_cache(position)
    assert sampler._log_likelihood_from_cache_fn(position, cache) == pytest.approx(
        float(
            _folded_log_likelihood_from_cache(
                position,
                cache,
                fold,
                jim._base_log_prior_fn,
                jim._base_log_likelihood_from_cache_fn,
            )
        )
    )


def test_no_cache_folded_likelihood_builds_one_cache_for_eight_projections(
    capturing_sampler: list[_CapturingSampler],
) -> None:
    jim = _folded_jim()
    del capturing_sampler
    likelihood = jim.likelihood
    assert isinstance(likelihood, _ToySingleEventLikelihood)
    position = _fold_to_fundamental(
        jnp.asarray([0.3, -0.7, 4.8, 2.6]),
        jim._resolved_fold_symmetry,
    )
    expected_images = _build_fold_images(position, jim._resolved_fold_symmetry)
    likelihood.clear_call_history()

    value = jim._log_likelihood_fn(position)
    jax.block_until_ready(value)

    assert len(likelihood.cache_build_positions) == 1
    assert len(likelihood.projection_positions) == 8
    np.testing.assert_allclose(likelihood.cache_build_positions[0], position)
    np.testing.assert_allclose(
        np.stack(likelihood.projection_positions),
        expected_images,
        atol=1.0e-6,
    )


def test_folded_initial_draws_are_the_canonicalized_prior_pushforward(
    capturing_sampler: list[_CapturingSampler],
) -> None:
    jim = _folded_jim()
    del capturing_sampler
    key = jax.random.key(104)
    raw_positions = _sampling_array(jim.prior, key, 256)
    expected = jax.vmap(
        lambda point: _fold_to_fundamental(point, jim._resolved_fold_symmetry)
    )(raw_positions)

    actual = jim.sample_initial_positions(256, rng_key=key)

    np.testing.assert_allclose(actual, expected, atol=1.0e-6)
    assert bool(
        jnp.all(
            jax.vmap(
                lambda point: _in_fundamental_domain(point, jim._resolved_fold_symmetry)
            )(actual)
        )
    )


def test_folded_jim_canonicalizes_explicit_and_default_sampler_starts(
    capturing_sampler: list[_CapturingSampler],
) -> None:
    jim = _folded_jim()
    sampler = capturing_sampler[-1]
    supplied = jnp.asarray(
        [
            [0.2, -0.8, 1.0, 2.8],
            [0.4, 0.6, 5.7, 1.9],
            [0.6, -0.3, 3.2, 0.7],
            [0.8, 0.9, 2.1, 2.2],
        ]
    )
    expected = jax.vmap(
        lambda point: _fold_to_fundamental(point, jim._resolved_fold_symmetry)
    )(supplied)

    jim.sample(initial_position=supplied)
    np.testing.assert_allclose(sampler.sampled_positions[-1], expected, atol=1.0e-6)

    jim.sample()
    default_positions = jnp.asarray(sampler.sampled_positions[-1])
    assert default_positions.shape == (4, 4)
    assert bool(
        jnp.all(
            jax.vmap(
                lambda point: _in_fundamental_domain(point, jim._resolved_fold_symmetry)
            )(default_positions)
        )
    )


def test_default_off_preserves_callback_identity_and_initialization(
    capturing_sampler: list[_CapturingSampler],
) -> None:
    likelihood = _ToySingleEventLikelihood(
        phase_marginalization=False,
        detectors=_valid_detector_sites()[:2],
    )
    jim = Jim(
        likelihood=likelihood,
        prior=_prior(),
        sampler_config=_swig_config(fold_symmetry=None),
        periodic=None,
    )
    sampler = capturing_sampler[-1]

    assert sampler._log_prior_fn is jim._log_prior_fn
    assert sampler._log_likelihood_fn is jim._log_likelihood_fn
    assert sampler._log_posterior_fn is jim._log_posterior_fn
    assert sampler._build_cache is jim._sampler_backend_kwargs["build_cache"]
    assert (
        sampler._log_likelihood_from_cache_fn
        is jim._sampler_backend_kwargs["log_likelihood_from_cache_fn"]
    )
    for attribute in (
        "_resolved_fold_symmetry",
        "_base_log_prior_fn",
        "_base_log_likelihood_fn",
        "_base_log_posterior_fn",
        "_base_build_cache",
        "_base_log_likelihood_from_cache_fn",
    ):
        assert not hasattr(jim, attribute)

    key = jax.random.key(91)
    expected = _sampling_array(jim.prior, key, 32)
    actual = jim.sample_initial_positions(32, rng_key=key)
    np.testing.assert_array_equal(actual, expected)


def test_folded_extraction_fails_closed_until_unfold_is_available(
    capturing_sampler: list[_CapturingSampler],
) -> None:
    jim = _folded_jim()
    sampler = capturing_sampler[-1]

    with pytest.raises((NotImplementedError, RuntimeError)) as samples_exc:
        jim.get_samples()
    assert "unfold" in str(samples_exc.value).lower()

    with pytest.raises((NotImplementedError, RuntimeError)) as weighted_exc:
        jim.get_weighted_samples(space="prior")
    assert "unfold" in str(weighted_exc.value).lower()

    raw = jim.get_weighted_samples(space="sampling")
    assert raw.keys() == sampler.raw_weighted_samples.keys()
    for name, values in sampler.raw_weighted_samples.items():
        np.testing.assert_array_equal(raw[name], values)
