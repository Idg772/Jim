"""End-to-end evidence regression for the SwiG quotient-fold target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.integration

blackjax = pytest.importorskip("blackjax")

from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.single_event.likelihood import SingleEventLikelihood
from jimgw.samplers.config import BlackJAXSwiGConfig, FoldSymmetryConfig

_PARAMETER_NAMES = ("cos_iota", "azimuth", "psi")
_PERIODIC = {"azimuth": (0.0, 2.0 * np.pi), "psi": (0.0, np.pi)}


@dataclass(frozen=True)
class _DetectorSite:
    name: str
    vertex: jax.Array


class _FullyCacheableWaveform:
    """Minimal declaration used by Jim's cache-dependency inference."""

    parameter_names = _PARAMETER_NAMES
    cacheable_parameter_names = frozenset(_PARAMETER_NAMES)

    def build_waveform_cache(self, frequencies: Any, params: dict[str, Any]) -> Any:
        del frequencies, params
        return jnp.zeros(())

    def waveform_from_cache(
        self,
        frequencies: Any,
        params: dict[str, Any],
        cache: Any,
    ) -> Any:
        del frequencies, params
        return cache


class _AnalyticSingleEventLikelihood(SingleEventLikelihood):
    """Cheap non-symmetric target with a valid waveform-cache surface."""

    def __init__(self) -> None:
        # Detector data and PSDs are irrelevant to this analytic integration test.
        self.waveform = _FullyCacheableWaveform()
        self.fixed_parameters: dict[str, Any] = {}
        self.trigger_time = 0.0
        self.gmst = jnp.zeros(())
        self.ref_dist = 1.0
        self.time_marginalization = False
        self.phase_marginalization = True
        self.distance_marginalization = False
        self.detectors = (
            _DetectorSite("A", jnp.asarray([0.0, 0.0, 0.0])),
            _DetectorSite("B", jnp.asarray([3.0, 0.0, 0.0])),
            _DetectorSite("C", jnp.asarray([0.0, 4.0, 1.0])),
        )

    @staticmethod
    def _analytic_log_likelihood(params: dict[str, Any]) -> jax.Array:
        return (
            0.7 * params["cos_iota"]
            + 0.2 * jnp.cos(params["azimuth"])
            + 0.1 * jnp.sin(2.0 * params["psi"])
        )

    def _evaluate(self, params: dict[str, Any]) -> jax.Array:
        return self._analytic_log_likelihood(params)

    def _generate_waveform(self, params: dict[str, Any]) -> jax.Array:
        del params
        return jnp.zeros(())

    def _evaluate_from_waveform(
        self,
        params: dict[str, Any],
        waveform_cache: Any,
    ) -> jax.Array:
        del waveform_cache
        return self._analytic_log_likelihood(params)


def _prior() -> CombinePrior:
    # The inclination support is deliberately not invariant under c -> -c.
    return CombinePrior(
        [
            UniformPrior(-1.0, 0.25, parameter_names=["cos_iota"]),
            UniformPrior(0.0, 2.0 * np.pi, parameter_names=["azimuth"]),
            UniformPrior(0.0, np.pi, parameter_names=["psi"]),
        ]
    )


def _config(*, folded: bool) -> BlackJAXSwiGConfig:
    fold = (
        FoldSymmetryConfig(
            cos_iota="cos_iota",
            azimuth="azimuth",
            psi="psi",
            azimuth_reflection_center=0.37,
        )
        if folded
        else None
    )
    return BlackJAXSwiGConfig(
        blocks=[list(_PARAMETER_NAMES)],
        fold_symmetry=fold,
        n_live=96,
        n_delete_frac=0.25,
        num_gibbs_sweeps=1,
        num_inner_steps_per_dim=1,
        max_steps=5,
        max_shrinkage=30,
        termination_dlogz=0.1,
        n_devices=1,
    )


def _run(*, folded: bool) -> tuple[float, float]:
    jim = Jim(
        likelihood=_AnalyticSingleEventLikelihood(),
        prior=_prior(),
        sampler_config=_config(folded=folded),
        periodic=_PERIODIC,
        seed=1701,
    )
    jim.sample()
    diagnostics = jim.get_diagnostics()
    return float(diagnostics["log_Z"]), float(diagnostics["log_Z_error"])


def test_folded_and_base_swig_evidence_agree_for_noninvariant_support() -> None:
    prior = _prior()
    common = {"azimuth": jnp.asarray(0.4), "psi": jnp.asarray(0.3)}
    assert jnp.isfinite(
        prior.log_prob({**common, "cos_iota": jnp.asarray(-0.75)})
    )
    assert jnp.isneginf(
        prior.log_prob({**common, "cos_iota": jnp.asarray(0.75)})
    )

    base_log_z, base_error = _run(folded=False)
    folded_log_z, folded_error = _run(folded=True)

    c_factor = (np.exp(0.7 * 0.25) - np.exp(-0.7)) / (0.7 * 1.25)
    exact_log_z = np.log(c_factor) + np.log(np.i0(0.2)) + np.log(np.i0(0.1))
    assert abs(base_log_z - exact_log_z) <= 3.0 * base_error
    assert abs(folded_log_z - exact_log_z) <= 3.0 * folded_error

    combined_error = np.hypot(base_error, folded_error)
    assert combined_error > 0.0
    assert abs(folded_log_z - base_log_z) <= 3.0 * combined_error
