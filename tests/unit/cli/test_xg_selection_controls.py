"""Configuration and fail-closed orchestration for pre-sampling grid selection.

These tests never allocate native network data or run the sampler. Numerical
moment/scorer parity is covered by separate tests; mocks here record which
frozen configuration reaches each package construction boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from jimgw.cli import _xg_binning
from jimgw.cli import xg_qualification as qualification
from jimgw.cli._config import (
    CLIHeterodynedConfig,
    CLIXGBinSelectionConfig,
    PipelineConfig,
)
from tests.xg_fixtures import network_config

SELECTED_EDGES = [2.0, 5.0, 32.0, 64.0]
SELECTED_PREFIX = [2.0, 2.0001]


@pytest.fixture
def cfg(tmp_path):
    raw = network_config()
    psd = tmp_path / "psd.npz"
    np.savez(psd, frequencies=[0.0, 2.0, 5.0, 64.0], values=np.ones(4))
    raw["data"]["psd_files"] = {"CE": str(psd), "ET": str(psd)}
    raw["output"]["dir"] = str(tmp_path / "run")
    return PipelineConfig.model_validate(
        raw, context={"prepare_xg_qualification": True}
    )


def test_default_network_requests_fast_empirical_selection_without_changing_sampler(
    cfg,
):
    h = cfg.likelihood.heterodyne
    assert h.xg_evaluation_mode == "auto"
    assert h.bin_selection is not None
    assert h.bin_selection.tolerance == 0.05
    assert h.bin_selection.reference_bins == 80
    assert h.bin_selection.candidate_bins == [40, 80]
    assert h.frequency_bin_edges is None
    assert (
        cfg.sampler.n_live,
        cfg.sampler.num_gibbs_sweeps,
        cfg.sampler.n_devices,
    ) == (32, 8, 1)
    assert cfg.sampler.bracket_mode == "stepping-out"
    assert cfg.verified_xg_manifest is None


@pytest.mark.parametrize(
    "change",
    [
        {"tolerance": 0},
        {"tolerance": -0.01},
        {"tolerance": np.nan},
        {"tolerance": np.inf},
        {"tolerance": 0.051},
        {"reference_bins": 1},
        {"reference_bins": 8193},
        {"reference_bins": True},
        {"candidate_bins": []},
        {"candidate_bins": [2, 2]},
        {"candidate_bins": [0, 2]},
        {"candidate_bins": [True, 2]},
        {"candidate_bins": [2.5]},
        {"candidate_bins": list(range(1, 18))},
        {"reference_bins": 8, "candidate_bins": [9, 10]},
        {"training_points": 3},
        {"verification_points": 257},
        {"frequency_chunk_size": 31},
        {"parameter_batch_size": 0},
        {"max_bank_bytes": 1048575},
        {"timing_lanes": 513},
        {"timing_repeats": 2},
        {"seed": -1},
    ],
)
def test_selection_config_rejects_invalid_or_unbounded_controls(change):
    with pytest.raises(ValidationError):
        CLIXGBinSelectionConfig.model_validate(change)


def test_bounded_fixture_can_filter_candidate_counts_above_reference_cap():
    settings = CLIXGBinSelectionConfig(reference_bins=8, candidate_bins=[4, 8, 16])
    assert settings.reference_bins == 8
    assert settings.candidate_bins == [4, 8, 16]


@pytest.mark.parametrize(
    "prefix",
    [[], [2.0], [2.0, 3.0, 4.0], [2.0, 2.0], [3.0, 2.0], [0.0, 2.0], [2.0, np.nan]],
)
def test_saved_node_prefix_requires_two_positive_finite_increasing_frequencies(prefix):
    with pytest.raises(ValidationError, match="node_frequency_prefix"):
        CLIHeterodynedConfig(n_bins=2, node_frequency_prefix=prefix)


def test_pinned_grid_serialization_retains_source_frequency_prefix():
    settings = CLIHeterodynedConfig(
        n_bins=3,
        frequency_bin_edges=SELECTED_EDGES,
        node_frequency_prefix=[2.0, 2.0001],
    )
    restored = CLIHeterodynedConfig.model_validate_json(settings.model_dump_json())
    assert restored.node_frequency_prefix == [2.0, 2.0001]
    assert restored.frequency_bin_edges == SELECTED_EDGES


def _patch_package_construction(monkeypatch, *, failure=None):
    events = []
    fine = SimpleNamespace(
        summary_data={"CE": np.zeros((1, 1))},
        phasor_data_moments={"CE": np.zeros((1, 1, 1))},
    )
    chosen = object()

    def plan(cfg, detectors, waveform):
        h = cfg.likelihood.heterodyne
        events.append(("plan", h.n_bins, h.frequency_bin_edges))
        if failure == "plan":
            raise ValueError("realized detector metadata changed")
        return "a" * 64

    def bind(cfg, digest):
        events.append(("bind", cfg.likelihood.heterodyne.n_bins, digest))
        return object()

    def build(binding, cfg, detectors, waveform, prior, transforms, **kwargs):
        events.append(("build", cfg.likelihood.heterodyne.n_bins, kwargs))
        if failure == "build":
            raise ValueError("qualification binding changed")
        return fine

    def select(cfg, candidate, detectors, waveform):
        assert candidate is fine
        events.append(("select", cfg.likelihood.heterodyne.n_bins))
        if failure == "select":
            raise RuntimeError("frozen grid failed independent verification")
        resolved = cfg.model_copy(deep=True)
        resolved.likelihood.heterodyne = resolved.likelihood.heterodyne.model_copy(
            update={
                "n_bins": 3,
                "frequency_bin_edges": SELECTED_EDGES,
                "node_frequency_prefix": SELECTED_PREFIX,
                "epsilon": None,
            }
        )
        return chosen, resolved, {"selected_bins": 3, "qualification": False}

    monkeypatch.setattr(qualification, "plan_xg_qualification_bin_edges", plan)
    monkeypatch.setattr(qualification, "bind_xg_qualification_candidate", bind)
    monkeypatch.setattr(qualification, "build_xg_qualification_candidate", build)
    monkeypatch.setattr(_xg_binning, "select_network_binning", select)
    return events, fine, chosen


def test_package_builds_one_fine_bank_and_returns_frozen_copy(cfg, monkeypatch):
    cfg.likelihood.heterodyne.n_bins = 40
    before = cfg.model_dump()
    events, _, chosen = _patch_package_construction(monkeypatch)
    probes = [{"M_c": 1.18}]
    candidate, resolved, selection = (
        qualification.build_selected_xg_qualification_candidate(
            cfg, [], "waveform", "prior", [], native_probe_parameters=probes
        )
    )
    assert candidate is chosen
    assert [event[0] for event in events] == ["plan", "bind", "build", "select"]
    assert events[0] == ("plan", 80, None)
    assert events[2] == ("build", 80, {"native_probe_parameters": probes})
    assert events[3] == ("select", 40)
    assert cfg.model_dump() == before
    assert resolved.likelihood.heterodyne.frequency_bin_edges == SELECTED_EDGES
    assert resolved.likelihood.heterodyne.node_frequency_prefix == SELECTED_PREFIX
    assert resolved.sampler == cfg.sampler
    assert resolved.verified_xg_manifest is None
    assert selection == {"selected_bins": 3, "qualification": False}


@pytest.mark.parametrize("pinned", [False, True])
def test_package_keeps_explicit_or_unselected_grid_without_search(
    cfg, monkeypatch, pinned
):
    if pinned:
        cfg.likelihood.heterodyne = cfg.likelihood.heterodyne.model_copy(
            update={"n_bins": 3, "frequency_bin_edges": SELECTED_EDGES}
        )
    else:
        cfg.likelihood.heterodyne.bin_selection = None
    events, fine, _ = _patch_package_construction(monkeypatch)
    candidate, resolved, selection = (
        qualification.build_selected_xg_qualification_candidate(
            cfg, [], "waveform", "prior", []
        )
    )
    assert candidate is fine
    assert resolved is not cfg
    assert resolved.model_dump() == cfg.model_dump()
    assert selection is None
    assert [event[0] for event in events] == ["plan", "bind", "build"]


@pytest.mark.parametrize(
    ("failure", "error", "expected"),
    [
        ("plan", ValueError, ["plan"]),
        ("build", ValueError, ["plan", "bind", "build"]),
        ("select", RuntimeError, ["plan", "bind", "build", "select"]),
    ],
)
def test_package_propagates_gate_failures_without_mutating_request(
    cfg, monkeypatch, failure, error, expected
):
    before = cfg.model_dump()
    events, _, _ = _patch_package_construction(monkeypatch, failure=failure)
    with pytest.raises(error):
        qualification.build_selected_xg_qualification_candidate(
            cfg, [], "waveform", "prior", []
        )
    assert [event[0] for event in events] == expected
    assert cfg.model_dump() == before


def test_package_requires_candidate_preflight_before_planning(cfg, monkeypatch):
    events, _, _ = _patch_package_construction(monkeypatch)
    cfg._xg_qualification_preflight = False
    with pytest.raises(ValueError, match="prepare_xg_qualification"):
        qualification.build_selected_xg_qualification_candidate(
            cfg, [], "waveform", "prior", []
        )
    assert events == []


def test_fixture_is_fresh_and_still_requires_production_qualification():
    raw = network_config()
    raw["likelihood"]["heterodyne"]["bin_selection"]["candidate_bins"].append(16)
    assert network_config()["likelihood"]["heterodyne"]["bin_selection"][
        "candidate_bins"
    ] == [40, 80]
    with pytest.raises(ValidationError, match="qualification_manifest"):
        PipelineConfig.model_validate(network_config())
