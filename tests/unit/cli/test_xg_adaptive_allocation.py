"""Independent NumPy checks of native-moment adaptive grid selection.

No real waveform, native detector construction, sampler, or JAX compilation is
used. Discrete noisy frequency samples define the reference integrals directly.
"""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from jimgw.cli import _xg_binning
from jimgw.cli._config import CLIHeterodynedConfig, CLIXGBinSelectionConfig
from jimgw.cli._xg_allocation import catalogue_moments, select_adaptive_network_grid
from jimgw.core.single_event.heterodyne_allocation import build_interval_catalogue
from jimgw.core.single_event.heterodyne_selection import (
    FrozenGridPlan,
    FrozenGridSelection,
)
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD


def _native_problem(edges, *, scale=1.0, phase_marginalization=True):
    edges = np.asarray(edges, dtype=float)
    anchors = np.array([-0.02, 0.0, 0.02])
    rng = np.random.default_rng(908)
    local = (np.arange(31) + 0.37) / 31
    frequencies = (edges[:-1, None] + np.diff(edges)[:, None] * local).ravel()
    weights = np.repeat(np.diff(edges) * scale / len(local), len(local))
    detector_data = [
        1.0
        + 0.4 * rng.normal(size=len(frequencies))
        + 0.4j * rng.normal(size=len(frequencies))
        for _ in range(2)
    ]
    detectors = [SimpleNamespace(name=f"D{i}", offset=i * 0.001) for i in range(2)]
    fine_a, fine_b, anchored = {}, {}, {}
    for detector, data in zip(detectors, detector_data, strict=True):
        a = np.empty((25, len(edges) - 1), dtype=complex)
        b = np.empty((17, len(edges) - 1))
        stored = np.empty((len(anchors), 25, len(edges) - 1), dtype=complex)
        for bin_index, (lo, hi) in enumerate(pairwise(edges)):
            take = (frequencies >= lo) & (frequencies < hi)
            u = (2 * frequencies[take] - lo - hi) / (hi - lo)
            powers = u[None] ** np.arange(25)[:, None]
            a[:, bin_index] = powers @ (weights[take] * data[take])
            b[:, bin_index] = powers[:17] @ weights[take]
            for k, anchor in enumerate(anchors):
                stored[k, :, bin_index] = powers @ (
                    weights[take]
                    * data[take]
                    * np.exp(2j * np.pi * frequencies[take] * anchor)
                )
        fine_a[detector.name], fine_b[detector.name] = a, b
        anchored[detector.name] = stored
    fine = SimpleNamespace(
        freq_grid_edges=edges,
        n_bins=len(edges) - 1,
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=anchors,
        phasor_approximation="taylor",
        phasor_data_moments=anchored,
        summary_moments={d.name: (fine_a[d.name], fine_b[d.name]) for d in detectors},
        detectors=detectors,
        phase_marginalization=phase_marginalization,
        reference_projection="carrier",
        reference_parameters={"M_c": 1.2, "phase_c": 0.0},
        freq_grid_node_flat=FrozenGridPlan(edges, 8, 0).nodes.ravel(),
        _rigid_time_shift=lambda d, p: p.get("shift", 0.0) + d.offset,
        _bin_edges_sha256=HeterodynedTransientLikelihoodFD._bin_edges_sha256,
        _prepare_parameters=lambda p: dict(p),
    )
    fine.bin_edges_sha256 = fine._bin_edges_sha256(
        edges,
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=anchors,
        reference_projection="carrier",
    )
    return fine, frequencies, weights, detector_data


def test_catalogue_moments_match_direct_discrete_parent_integrals_with_anchors():
    edges = np.array([2, 2.04, 2.12, 2.19, 2.43, 2.51, 2.9, 3.7, 4])
    fine, frequencies, weights, data = _native_problem(edges)
    cat = build_interval_catalogue(
        edges, (0, 2, 4, 6, 8), extra_partitions=[(0, 3, 5, 8)]
    )
    a = np.stack([fine.phasor_data_moments[d.name] for d in fine.detectors])
    b = np.stack([fine.summary_moments[d.name][1] for d in fine.detectors])
    translated_a = catalogue_moments(cat, a)
    translated_b = catalogue_moments(cat, b)
    assert translated_a.shape == (2, 3, 25, cat.n_intervals)
    for interval, (start, stop) in enumerate(cat.intervals):
        lo, hi = edges[[start, stop]]
        take = (frequencies >= lo) & (frequencies < hi)
        u = (2 * frequencies[take] - lo - hi) / (hi - lo)
        powers = u[None] ** np.arange(25)[:, None]
        expected_b = powers[:17] @ weights[take]
        for d in range(2):
            np.testing.assert_allclose(
                translated_b[d, :, interval], expected_b, rtol=3e-12, atol=1e-12
            )
            for k, anchor in enumerate(fine.phasor_time_anchors):
                expected_a = powers @ (
                    weights[take]
                    * data[d][take]
                    * np.exp(2j * np.pi * frequencies[take] * anchor)
                )
                np.testing.assert_allclose(
                    translated_a[d, k, :, interval], expected_a, rtol=3e-12, atol=1e-12
                )


def test_catalogue_moments_promote_integer_input_before_affine_translation():
    cat = build_interval_catalogue([0.0, 1.0, 3.0], [0, 1, 2], refinement_depth=0)
    # One point of unit mass at the centre of each fine interval.
    fine = np.array([[1, 1], [0, 0], [0, 0]], dtype=np.int64)
    result = catalogue_moments(cat, fine)
    root = cat.roots[0]
    points = np.array([0.5, 2.0])
    local = (points - 1.5) / 1.5
    expected = np.array([np.sum(local**k) for k in range(3)])
    assert result.dtype == np.float64
    np.testing.assert_allclose(result[:, root], expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("phase_marginalization", [False, True])
def test_interval_statistics_match_direct_native_sums_and_whole_oracle_score(
    phase_marginalization,
):
    edges = np.array([2, 2.04, 2.12, 2.19, 2.43, 2.51, 2.9, 3.7, 4])
    fine, frequencies, weights, data = _native_problem(
        edges, phase_marginalization=phase_marginalization
    )
    cat = build_interval_catalogue(
        edges, (0, 2, 4, 6, 8), extra_partitions=[(0, 3, 5, 8)]
    )
    points = [{"shift": value} for value in (-0.009, 0.0, 0.008)]
    bank = SimpleNamespace(points={"training": points})
    oracle = _xg_binning.MomentErrorOracle(fine, bank, 0.05)
    rng = np.random.default_rng(301)
    coefficients = 0.03 * (
        rng.normal(size=(3, 2, cat.n_intervals, 9))
        + 1j * rng.normal(size=(3, 2, cat.n_intervals, 9))
    )
    coefficients[..., 0] += 1
    stored = [
        (
            catalogue_moments(cat, fine.phasor_data_moments[d.name]),
            catalogue_moments(cat, fine.summary_moments[d.name][1]),
        )
        for d in fine.detectors
    ]
    bounds = edges[cat.intervals]
    centres, half = bounds.mean(axis=1), np.diff(bounds, axis=1)[:, 0] / 2
    z, norm = oracle.interval_statistics(
        centres, half, coefficients, stored, "training"
    )
    expected_z, expected_q = np.zeros_like(z), np.zeros_like(norm)
    for t, (lo, hi) in enumerate(bounds):
        take = (frequencies >= lo) & (frequencies < hi)
        u = (2 * frequencies[take] - lo - hi) / (hi - lo)
        for j, point in enumerate(points):
            for d, detector in enumerate(fine.detectors):
                ratio = np.polynomial.polynomial.polyval(u, coefficients[j, d, t])
                expected_z[j, t] += np.sum(
                    weights[take]
                    * data[d][take]
                    * ratio.conj()
                    * np.exp(
                        2j
                        * np.pi
                        * frequencies[take]
                        * (point["shift"] + detector.offset)
                    )
                )
                expected_q[j, t] += np.sum(weights[take] * abs(ratio) ** 2)
    np.testing.assert_allclose(z, expected_z, rtol=3e-12, atol=1e-12)
    np.testing.assert_allclose(norm, expected_q, rtol=3e-12, atol=1e-12)
    chosen = (0, 3, 5, 8)
    lookup = {tuple(pair): t for t, pair in enumerate(cat.intervals)}
    selected = [lookup[(a, b)] for a, b in pairwise(chosen)]
    whole = oracle.score(edges[list(chosen)], coefficients[:, :, selected], "training")
    np.testing.assert_allclose(
        whole,
        oracle.log_likelihood(z[:, selected].sum(1), norm[:, selected].sum(1)),
        rtol=3e-12,
        atol=1e-12,
    )


class _RatioBank:
    def __init__(self, fine, events, *, mode="curved"):
        self.fine, self.events, self.mode = fine, events, mode
        self.values, self.calls = {}, []
        self.points = {
            "training": [{"amplitude": a, "shift": 0.0} for a in (0, 1, 0.85)],
            "verification": [{"amplitude": a, "shift": 0.0} for a in (0, 0.94, 1.04)],
        }

    def __call__(self, f, name):
        self.events.append(name)
        self.calls.append((name, f.copy()))
        if self.mode == "flat" or (
            self.mode == "holdout_failure" and name == "training"
        ):
            feature = np.zeros_like(f, dtype=complex)
        elif self.mode == "holdout_failure":
            feature = 10 * np.maximum(f - 0.5, 0)
        else:
            feature = 0.8 * np.exp(-(((f - 0.12) / 0.02) ** 2)) * np.exp(100j * f)
        values = np.stack([1 + p["amplitude"] * feature for p in self.points[name]])
        self.values[name] = np.repeat(values[:, None], len(self.fine.detectors), axis=1)
        return self.values[name]


class _Timer:
    plan_key = _xg_binning.CandidateTimer.plan_key

    def __init__(self, fine, events, *, result=None):
        self.fine, self.events, self.result = fine, events, result
        self.plans, self.records = [], {}

    def __call__(self, plan):
        self.events.append("timer")
        self.plans.append(plan)
        value = float(plan.n_bins) if self.result is None else self.result
        self.records[self.plan_key(plan)] = {"n_bins": plan.n_bins, "cost": value}
        return value


def _settings(**overrides):
    values = {
        "allocation_base_bins": 4,
        "allocation_refinement_depth": 2,
        "candidate_bins": [2, 4, 8, 16, 32],
        "candidate_edge_indices": [[0, 2, 4, 8, 16, 32]],
        "allocation_max_intervals": 256,
        "allocation_max_proposals": 64,
        "tolerance": 0.05,
        "max_timed_candidates": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_noisy_nonuniform_selection_matches_complete_native_likelihood_gate():
    fine, _, _, _ = _native_problem(np.linspace(0, 1, 33), scale=100)
    events = []
    bank, timer = _RatioBank(fine, events), _Timer(fine, events)
    oracle = _xg_binning.MomentErrorOracle(fine, bank, 0.05)
    selected, diagnostics = select_adaptive_network_grid(
        fine, bank, oracle, timer, _settings()
    )
    assert np.max(selected.training_error) <= 0.0375
    assert np.max(selected.verification_error) <= 0.0375
    assert not np.allclose(
        np.diff(selected.plan.edges), np.diff(selected.plan.edges)[0]
    )
    uniform_records = [
        r
        for r in selected.candidate_diagnostics
        if np.allclose(
            np.diff(r["frequency_bin_edges"]), np.diff(r["frequency_bin_edges"])[0]
        )
    ]
    assert {r["n_bins"] for r in uniform_records}.issuperset({2, 4, 8, 16, 32})
    assert selected.plan.n_bins < min(
        r["n_bins"] for r in uniform_records if r["training_passed"]
    )
    assert [name for name, _ in bank.calls] == ["training", "verification"]
    np.testing.assert_array_equal(bank.calls[0][1], bank.calls[1][1])
    assert 1 <= len(timer.plans) <= 3
    assert diagnostics["compiled_finalists"] == len(timer.plans)
    assert events[-1] == "verification"
    # Independently reconstruct the winner through the ordinary complete oracle.
    ratios = bank.values["verification"][..., selected.plan.node_indices]
    errors = oracle(selected.plan, ratios, "verification")
    np.testing.assert_allclose(abs(errors), selected.verification_error, atol=1e-12)


def test_holdout_failure_does_not_time_or_choose_another_partition():
    fine, _, _, _ = _native_problem(np.linspace(0, 1, 33), scale=100)
    events = []
    bank = _RatioBank(fine, events, mode="holdout_failure")
    timer = _Timer(fine, events)
    oracle = _xg_binning.MomentErrorOracle(fine, bank, 0.05)
    with pytest.raises(
        RuntimeError, match="frozen grid failed independent verification"
    ):
        select_adaptive_network_grid(fine, bank, oracle, timer, _settings())
    assert events == ["training", "timer", "timer", "timer", "verification"]
    assert len(timer.plans) == 3
    assert min(p.n_bins for p in timer.plans) == 1


@pytest.mark.parametrize("invalid_cost", [np.nan, np.inf, -1.0])
def test_invalid_candidate_timing_fails_before_consuming_holdout(invalid_cost):
    fine, _, _, _ = _native_problem(np.linspace(0, 1, 33))
    events = []
    bank = _RatioBank(fine, events, mode="flat")
    timer = _Timer(fine, events, result=invalid_cost)
    oracle = _xg_binning.MomentErrorOracle(fine, bank, 0.05)
    with pytest.raises(ValueError, match="cost|finite|nonnegative"):
        select_adaptive_network_grid(fine, bank, oracle, timer, _settings())
    assert "verification" not in events


def test_equal_count_layout_keys_and_default_three_shape_limit_are_distinct():
    fine, _, _, _ = _native_problem(np.linspace(0, 1, 33))
    events = []
    bank, timer = _RatioBank(fine, events, mode="flat"), _Timer(fine, events)
    oracle = _xg_binning.MomentErrorOracle(fine, bank, 0.05)
    settings = _settings(candidate_edge_indices=[[0, 8, 32], [0, 16, 32]])
    result, diagnostics = select_adaptive_network_grid(
        fine, bank, oracle, timer, settings
    )
    same_count = [r for r in result.candidate_diagnostics if r["n_bins"] == 2]
    assert len(same_count) >= 2
    assert len({r["plan_sha256"] for r in same_count}) == len(same_count)
    assert diagnostics["compiled_finalists"] == len(timer.plans) == 3
    assert len({p.n_bins for p in timer.plans}) == 3
    first = FrozenGridPlan(np.array([0, 0.25, 1]), 8, 0)
    second = FrozenGridPlan(np.array([0, 0.5, 1]), 8, 0)
    assert timer.plan_key(first) != timer.plan_key(second)
    old = timer.plan_key(first)
    fine.phasor_approximation = "chebyshev"
    assert timer.plan_key(first) != old


def test_native_bank_export_roundtrips_moments_prefix_provenance_and_parameters(
    tmp_path, monkeypatch
):
    fine, _, _, _ = _native_problem([2, 2.1, 2.7, 4])
    fine._xg_node_frequency_prefix = np.array([2.0, 2.0001])
    points = ([{"M_c": 1.2}], [{"M_c": 1.21}])
    monkeypatch.setattr(_xg_binning, "parameter_banks", lambda *args: points)
    cfg = SimpleNamespace(
        likelihood=SimpleNamespace(heterodyne=SimpleNamespace(bin_selection=object())),
        model_dump=lambda **kwargs: {"data": {"zero_noise": False}},
        xg_analysis_contract_sha256=lambda: "a" * 64,
        xg_input_files_sha256=lambda: {"noise": "b" * 64},
    )
    path = tmp_path / "native-moments.npz"
    receipt = _xg_binning.export_native_moment_bank(fine, path, cfg)
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["bytes"] == path.stat().st_size
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        np.testing.assert_array_equal(archive["prefix"], fine._xg_node_frequency_prefix)
        np.testing.assert_array_equal(archive["edges"], fine.freq_grid_edges)
        for detector in fine.detectors:
            np.testing.assert_array_equal(
                archive[f"{detector.name}_anchored_a"],
                fine.phasor_data_moments[detector.name],
            )
            np.testing.assert_array_equal(
                archive[f"{detector.name}_b"], fine.summary_moments[detector.name][1]
            )
        assert not any("strain" in name for name in archive.files)
    assert metadata["analysis_contract_sha256"] == "a" * 64
    assert metadata["config"]["data"]["zero_noise"] is False
    assert metadata["parameter_banks"]["verification"] == points[1]
    assert metadata["parameter_bank_hashes"]["verification"] == _xg_binning._bank_hash(
        points[1]
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _xg_binning.export_native_moment_bank(fine, path, cfg)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "order,anchors",
    [
        (8, [-0.02, 0.02]),
        (16, None),
        (16, []),
        (16, [0.0]),
        (16, [0.0, 0.0]),
        (16, [0.02, -0.02]),
        (16, [-0.02, np.inf]),
    ],
)
def test_chebyshev_config_requires_degree_16_and_valid_anchor_support(order, anchors):
    with pytest.raises(ValidationError):
        CLIHeterodynedConfig(
            interpolation_order=8,
            phasor_moment_order=order,
            phasor_time_anchors=anchors,
            phasor_approximation="chebyshev",
        )


def test_resolved_phasor_and_allocation_config_roundtrip_exact_floats():
    edges = [2.0, float(np.nextafter(5.0, np.inf)), 64.0]
    prefix = [2.0, 2.0001]
    h = CLIHeterodynedConfig(
        n_bins=2,
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=[-0.02, 0.0, 0.02],
        phasor_approximation="chebyshev",
        frequency_bin_edges=edges,
        node_frequency_prefix=prefix,
        bin_selection=CLIXGBinSelectionConfig(
            method="adaptive",
            reference_bins=32,
            candidate_bins=[4, 16, 32],
            candidate_edge_indices=[[0, 3, 9, 32]],
        ),
    )
    restored = CLIHeterodynedConfig.model_validate_json(h.model_dump_json())
    assert restored.frequency_bin_edges == edges
    assert restored.node_frequency_prefix == prefix
    assert restored.phasor_approximation == "chebyshev"
    assert restored.phasor_time_anchors == h.phasor_time_anchors
    assert restored.bin_selection.method == "adaptive"
    assert restored.bin_selection.candidate_edge_indices == [[0, 3, 9, 32]]


def test_legacy_taylor_bin_hashes_match_frozen_pre_port_source():
    # Goldens independently obtained by extracting and executing only the hash
    # method from xg-five-minute-gpu-20260911/network-source.tar.gz, before this
    # phasor/allocation port. Tests do not need the benchmark archive installed.
    digest = HeterodynedTransientLikelihoodFD._bin_edges_sha256
    edges = [2.0, 5.0, 64.0, 2048.0]
    assert (
        digest(edges)
        == "e6e367cbe6067eae48b130c8138886509bbffda3aca7e56b350fb619bde0dae4"
    )
    kwargs = {
        "interpolation_order": 8,
        "phasor_moment_order": 16,
        "phasor_time_anchors": [-0.02, 0.0, 0.02],
        "reference_projection": "carrier",
    }
    expected = "8053f9ce087cad1f34fc918d14894862e85a716b57f4ce7ef2a0d62cf39afe74"
    assert digest(edges, **kwargs) == expected
    assert digest(edges, phasor_approximation="taylor", **kwargs) == expected
    assert digest(edges, phasor_approximation="chebyshev", **kwargs) != expected


@pytest.mark.parametrize("method", ["uniform", "adaptive"])
def test_selection_wrapper_preserves_method_phasor_exact_edges_and_original_prefix(
    monkeypatch, method
):
    from jimgw.cli import _xg_allocation, xg_qualification

    fine, _, _, _ = _native_problem(np.linspace(2, 64, 17))
    fine.phasor_approximation = "chebyshev"
    fine.evaluation_diagnostics = {"implementation": "factored-carrier-arm-even-odd-v1"}
    h = CLIHeterodynedConfig(
        n_bins=4,
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=[-0.02, 0.0, 0.02],
        phasor_approximation="chebyshev",
        reference_projection="carrier",
        bin_selection=CLIXGBinSelectionConfig(
            method=method, reference_bins=16, candidate_bins=[4, 16]
        ),
    )
    cfg = SimpleNamespace(likelihood=SimpleNamespace(heterodyne=h))
    cfg.model_copy = lambda deep=False: SimpleNamespace(
        likelihood=SimpleNamespace(heterodyne=h.model_copy(deep=deep))
    )
    prefix = np.array([2.0, 2.0001])
    points = [{"M_c": 1.2}, {"M_c": 1.21}]
    bank = SimpleNamespace(
        prefix=prefix,
        points={"training": points, "verification": points},
        calls=0,
        wall_seconds=0.0,
    )
    edges = fine.freq_grid_edges[[0, 3, 9, 16]]
    selected = FrozenGridSelection(
        plan=FrozenGridPlan(edges, 8, 1),
        training_error=np.zeros(2),
        verification_error=np.zeros(2),
        training_residual=None,
        verification_residual=None,
        candidate_diagnostics=(),
        sparse_frequency_count=80,
        error_metric="test",
        frequencies=np.arange(80),
        cost_metric="test",
    )
    candidate = SimpleNamespace(
        n_bins=3,
        freq_grid_node_flat=selected.plan.nodes.ravel(),
        evaluation_diagnostics=fine.evaluation_diagnostics,
        phasor_approximation="chebyshev",
    )
    timer = SimpleNamespace(
        clone=lambda plan: candidate,
        weights=(1, 2, 3),
        records={},
        compile_seconds=0.0,
    )
    monkeypatch.setattr(_xg_binning, "parameter_banks", lambda *args: (points, points))
    monkeypatch.setattr(_xg_binning, "SparseRatioBank", lambda *args: bank)
    monkeypatch.setattr(
        _xg_binning,
        "MomentErrorOracle",
        lambda *args: SimpleNamespace(reference_errors={}),
    )
    monkeypatch.setattr(_xg_binning, "CandidateTimer", lambda *args: timer)
    monkeypatch.setattr(
        _xg_binning, "select_frozen_grid", lambda *args, **kwargs: selected
    )
    monkeypatch.setattr(
        _xg_allocation,
        "select_adaptive_network_grid",
        lambda *args: (selected, {"test": True}),
    )
    bound_configs = []

    def bind(resolved, digest):
        bound_configs.append(resolved)
        return SimpleNamespace(analysis_contract_sha256="a" * 64)

    monkeypatch.setattr(xg_qualification, "bind_xg_qualification_candidate", bind)
    monkeypatch.setattr(
        xg_qualification, "_verify_realized_candidate_inputs", lambda *args: None
    )
    monkeypatch.setattr(
        xg_qualification, "_verify_qualification_candidate_binding", lambda *args: None
    )
    _, resolved, record = _xg_binning.select_network_binning(
        cfg, fine, fine.detectors, object()
    )
    frozen = resolved.likelihood.heterodyne
    assert frozen.frequency_bin_edges == edges.tolist()
    assert frozen.node_frequency_prefix == prefix.tolist()
    assert frozen.phasor_approximation == "chebyshev"
    assert frozen.phasor_time_anchors == h.phasor_time_anchors
    assert frozen.bin_selection.method == method
    assert frozen.n_bins == 3 and frozen.epsilon is None
    assert bound_configs == [resolved]
    assert record["method"] == (
        "shared-native-moments-adaptive-k8-v1"
        if method == "adaptive"
        else "shared-native-moments-frozen-grid-v1"
    )
    assert cfg.likelihood.heterodyne.frequency_bin_edges is None
    assert cfg.likelihood.heterodyne.n_bins == 4
