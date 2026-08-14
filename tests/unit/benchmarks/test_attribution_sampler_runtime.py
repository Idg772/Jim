import pytest

from benchmarks.device_parallel_nss.sampler_ablation import variant


def test_attribution_chain_exposes_legacy_and_production_sampler_endpoints() -> None:
    legacy = variant("legacy-stock-lockstep-cov")
    production = variant("replicated-cached-fsm-factor")

    assert legacy.report() == {
        "name": "legacy-stock-lockstep-cov",
        "topology": "legacy-sharded-live",
        "interval": "stock-stepping-out",
        "scheduler": "lockstep",
        "direction_parameter": "covariance",
    }
    assert production.report() == {
        "name": "replicated-cached-fsm-factor",
        "topology": "replicated-live",
        "interval": "cached-stepping-out",
        "scheduler": "fsm",
        "direction_parameter": "cholesky-factor",
    }


def test_unknown_attribution_sampler_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sampler ablation variant"):
        variant("not-a-cell")
