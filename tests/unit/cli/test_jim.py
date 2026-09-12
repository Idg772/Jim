"""Unit tests for the CLI Jim wiring."""

import math
from types import SimpleNamespace

import pytest

from jimgw.cli import _jim
from jimgw.cli._config import PipelineConfig, PriorConfig
from jimgw.cli._jim import build_jim, infer_periodic_bounds

_RAW = {
    "data": {
        "type": "gwosc",
        "detectors": ["H1", "L1"],
        "trigger_time": 1126259462.4,
        "duration": 4.0,
        "psd_duration": 1024.0,
    },
    "waveform": {"approximant": "IMRPhenomXAS"},
    "prior": {
        "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
        "ra": {"type": "uniform", "min": 0.0, "max": 2.0 * math.pi},
        "dec": {"type": "cosine"},
        "psi": {"type": "uniform", "min": 0.0, "max": math.pi},
    },
    "likelihood": {"f_min": 20.0, "f_max": 1024.0},
    "output": {"dir": "tests/tmp/test"},
}


def _swig_config(**sampler_extra) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            **_RAW,
            "sampler": {
                "type": "blackjax-swig",
                "blocks": [["M_c"], ["ra", "dec"], ["psi"]],
                "n_live": 4,
                "n_delete_frac": 0.5,
                **sampler_extra,
            },
        }
    )


class _Prior:
    parameter_names = ("M_c", "ra", "dec", "psi")


class _Rename:
    """Sample transform stub renaming one parameter (like t_c -> t_det)."""

    def __init__(self, source: str, target: str):
        self._source = source
        self._target = target

    def propagate_name(self, names):
        return tuple(self._target if name == self._source else name for name in names)


def test_infer_periodic_bounds_reads_declared_prior_support():
    prior_cfg = PriorConfig.model_validate(_RAW["prior"])
    bounds = infer_periodic_bounds(prior_cfg, ("M_c", "ra", "dec", "psi", "t_det"))
    assert bounds == {"ra": (0.0, 2.0 * math.pi), "psi": (0.0, math.pi)}


def test_infer_periodic_bounds_skips_restricted_window_prior():
    raw = {**_RAW["prior"], "ra": {"type": "uniform", "min": 2.90, "max": 3.05}}
    prior_cfg = PriorConfig.model_validate(raw)
    bounds = infer_periodic_bounds(prior_cfg, ("M_c", "ra", "dec", "psi", "t_det"))
    assert bounds == {"psi": (0.0, math.pi)}


def test_infer_periodic_bounds_keeps_shifted_full_period_prior():
    raw = {**_RAW["prior"], "ra": {"type": "uniform", "min": -math.pi, "max": math.pi}}
    prior_cfg = PriorConfig.model_validate(raw)
    bounds = infer_periodic_bounds(prior_cfg, ("M_c", "ra", "dec", "psi"))
    assert bounds["ra"] == (-math.pi, math.pi)


def test_infer_periodic_bounds_uses_canonical_support_for_derived_azimuth():
    prior_cfg = PriorConfig.model_validate(_RAW["prior"])
    bounds = infer_periodic_bounds(prior_cfg, ("M_c", "azimuth", "zenith", "psi"))
    assert bounds["azimuth"] == (0.0, 2.0 * math.pi)
    assert bounds["psi"] == (0.0, math.pi)
    assert "zenith" not in bounds


def test_build_jim_declares_periodic_bounds_for_wrapped_covariance(monkeypatch):
    captured = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(_jim, "Jim", _stub)
    cfg = _swig_config(periodic_wrapped_covariance=True)

    build_jim(
        likelihood=object(),
        prior=_Prior(),
        sample_transforms=[_Rename("M_c", "log_M_c")],
        likelihood_transforms=[],
        cfg=cfg,
    )

    assert captured["periodic"] == {
        "ra": (0.0, 2.0 * math.pi),
        "psi": (0.0, math.pi),
    }


def test_build_jim_leaves_periodic_unset_without_wrapped_covariance(monkeypatch):
    captured = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(_jim, "Jim", _stub)
    cfg = _swig_config()

    build_jim(
        likelihood=object(),
        prior=_Prior(),
        sample_transforms=[],
        likelihood_transforms=[],
        cfg=cfg,
    )

    assert captured.get("periodic") is None


def test_build_jim_rejects_wrapped_covariance_without_periodic_parameters(monkeypatch):
    monkeypatch.setattr(_jim, "Jim", lambda **kwargs: SimpleNamespace())
    cfg = PipelineConfig.model_validate(
        {
            **_RAW,
            "prior": {"M_c": {"type": "uniform", "min": 10.0, "max": 80.0}},
            "sampler": {
                "type": "blackjax-swig",
                "blocks": [["M_c"]],
                "n_live": 4,
                "n_delete_frac": 0.5,
                "periodic_wrapped_covariance": True,
            },
        }
    )

    class _MassOnly:
        parameter_names = ("M_c",)

    with pytest.raises(ValueError, match="periodic"):
        build_jim(
            likelihood=object(),
            prior=_MassOnly(),
            sample_transforms=[],
            likelihood_transforms=[],
            cfg=cfg,
        )
