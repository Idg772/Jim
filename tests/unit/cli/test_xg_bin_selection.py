"""Bounded noisy-network tests for selection before sampler construction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.cli import _xg_binning as selection
from jimgw.cli._config import PipelineConfig
from jimgw.cli._data import build_data
from jimgw.cli._prior import build_prior
from jimgw.cli.xg_qualification import (
    bind_xg_qualification_candidate,
    build_xg_qualification_candidate,
    plan_xg_qualification_bin_edges,
)
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.heterodyne_moments import polynomial_moments
from jimgw.core.single_event.heterodyne_selection import (
    FrozenGridPlan,
    select_frozen_grid,
)
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from tests.xg_fixtures import network_config


def synthetic_moment_problem():
    frequencies = np.linspace(1.0, 3.0, 701)
    random = np.random.default_rng(441)
    data = (
        2.0
        + random.normal(size=len(frequencies))
        + 1j * random.normal(size=len(frequencies))
    )
    edges = np.asarray([1.0, 2.0, 3.0])
    a, b = polynomial_moments(
        frequencies,
        data,
        np.ones_like(frequencies),
        np.ones_like(frequencies),
        edges,
        data_order=10,
        norm_order=16,
    )
    fine = SimpleNamespace(
        freq_grid_edges=edges,
        n_bins=2,
        interpolation_order=8,
        phasor_moment_order=2,
        phasor_time_anchors=[0.0],
        phasor_data_moments={"test": a[None]},
        summary_moments={"test": (a, b)},
        detectors=[SimpleNamespace(name="test")],
        phase_marginalization=True,
        _rigid_time_shift=lambda d, p: 0.0,
    )
    return fine, frequencies, data


def test_noisy_native_moment_oracle_matches_direct_complex_overlap():
    from scipy.special import i0e

    fine, native_f, native_data = synthetic_moment_problem()
    bank = SimpleNamespace(
        values={}, points={"training": [{}, {}], "verification": [{}, {}]}
    )

    def ratios(f, name):
        value = np.stack([np.ones_like(f), 1 + 0.1 * f + 0.05j * f**2])[:, None, :]
        bank.values[name] = value
        return value

    oracle = selection.MomentErrorOracle(fine, bank, 0.05)
    result = select_frozen_grid(
        fine.freq_grid_edges,
        lambda f: ratios(f, "training"),
        lambda f: ratios(f, "verification"),
        orders=(8,),
        bin_counts=[1, 2],
        max_bins=2,
        error_budget=0.0375,
        likelihood_error=oracle,
    )
    assert result.plan.n_bins == 1
    values = bank.values["verification"][..., result.plan.node_indices]
    coefficients = np.einsum("ij,...bj->...bi", oracle.inverse, values)
    actual = oracle.score(result.plan.edges, coefficients, "verification")
    exact_ratios = np.stack(
        [np.ones_like(native_f), 1 + 0.1 * native_f + 0.05j * native_f**2]
    )
    overlap = np.sum(native_data * exact_ratios.conj(), axis=-1)
    expected = (
        np.log(i0e(abs(overlap)))
        + abs(overlap)
        - 0.5 * np.sum(abs(exact_ratios) ** 2, axis=-1)
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
    assert max(oracle.reference_errors.values()) < 1e-8


def test_unresolved_fine_reference_fails_before_verification():
    fine, _, _ = synthetic_moment_problem()
    bank = SimpleNamespace(
        values={}, points={"training": [{}, {}], "verification": [{}, {}]}
    )

    def training(f):
        bank.values["training"] = np.stack([np.ones_like(f), 2 * np.exp(80j * f)])[
            :, None, :
        ]
        return bank.values["training"]

    def holdout(f):
        pytest.fail("fine-reference failure must stop before holdout")

    with pytest.raises(
        RuntimeError, match="fine bin reference failed training convergence"
    ):
        select_frozen_grid(
            fine.freq_grid_edges,
            training,
            holdout,
            orders=(8,),
            max_bins=2,
            error_budget=0.0375,
            likelihood_error=selection.MomentErrorOracle(fine, bank, 0.05),
        )


def test_moment_oracle_holdout_rejects_selected_coarse_grid_without_retuning():
    fine, _, _ = synthetic_moment_problem()
    bank = SimpleNamespace(
        values={}, points={"training": [{}, {}], "verification": [{}, {}]}
    )

    def ratios(f, name):
        proposed = (
            np.ones_like(f) if name == "training" else 1 + 10 * np.maximum(f - 2, 0)
        )
        bank.values[name] = np.stack([np.ones_like(f), proposed])[:, None, :]
        return bank.values[name]

    oracle = selection.MomentErrorOracle(fine, bank, 0.05)
    with pytest.raises(
        RuntimeError, match="frozen grid failed independent verification"
    ):
        select_frozen_grid(
            fine.freq_grid_edges,
            lambda f: ratios(f, "training"),
            lambda f: ratios(f, "verification"),
            orders=(8,),
            max_bins=2,
            error_budget=0.0375,
            likelihood_error=oracle,
        )
    assert max(oracle.reference_errors.values()) < 1e-7


@pytest.fixture(scope="module")
def tiny_network(tmp_path_factory):
    raw = network_config()
    raw["data"]["injection_parameters"].update(d_L=200.0, phase_c=0.0)
    inputs = tmp_path_factory.mktemp("selection-inputs")
    frequencies = np.linspace(0.0, 64.0, 513)
    for name, scale in (("CE", 1e-46), ("ET", 2e-46)):
        psd = inputs / f"{name}-psd.npz"
        values = scale * (1 + (8 / np.maximum(frequencies, 1)) ** 4)
        np.savez(psd, frequencies=frequencies, values=values)
        raw["data"]["psd_files"][name] = str(psd)
    response = raw["likelihood"]
    # This tiny FD quadrature tests algebra at local parameter points; it does
    # not represent a resolved 2 Hz signal or qualify the broad physical prior.
    widths = {
        "M_c": 2e-7,
        "q": 2e-4,
        "s1_z": 1e-4,
        "s2_z": 1e-4,
        "lambda_1": 0.1,
        "lambda_2": 0.1,
        "d_L": 0.1,
        "iota": 1e-3,
        "ra": 1e-3,
        "dec": 1e-3,
        "psi": 1e-3,
        "t_c": 1e-5,
    }
    for name, width in widths.items():
        centre = raw["data"]["injection_parameters"][name]
        raw["prior"][name] = {
            "type": "uniform",
            "min": centre - width,
            "max": centre + width,
        }
    raw["prior"]["iota"] = {"type": "sine"}
    response["heterodyne"].update(
        n_bins=80,
        reference_chunk_size=256,
        bin_selection={
            "method": "uniform",
            "reference_bins": 80,
            "candidate_bins": [80],
            "training_points": 4,
            "verification_points": 4,
            "frequency_chunk_size": 256,
            "parameter_batch_size": 4,
            "timing_lanes": 2,
            "timing_repeats": 3,
        },
    )
    raw["output"]["dir"] = str(tmp_path_factory.mktemp("selection-network"))
    cfg = PipelineConfig.model_validate(raw, context={"prepare_xg_qualification": True})
    waveform = DominantModeTimeCachedWaveform(RippleIMRPhenomD_NRTidalv2(f_ref=20.0))
    lc = cfg.likelihood
    detectors = build_data(
        cfg.data,
        f_min={"CE": 5.0, "ET1": 2.0, "ET2": 2.0, "ET3": 2.0},
        f_max=64.0,
        waveform=waveform,
        time_frame=cfg.sampling.time_frame,
        time_dependent_response=True,
        finite_arm_response=True,
        orbital_motion_response=lc.orbital_motion_response,
        orbital_reference_time=lc.orbital_reference_time,
        orbital_validity_s=lc.orbital_validity_s,
        orbital_acceleration_over_c=lc.orbital_acceleration_over_c,
        orbital_jerk_over_c=lc.orbital_jerk_over_c,
        seed=cfg.seed,
        input_provenance_sha256=cfg.xg_input_files_sha256(),
    )
    digest = plan_xg_qualification_bin_edges(cfg, detectors, waveform)
    binding = bind_xg_qualification_candidate(cfg, digest)
    fine = build_xg_qualification_candidate(
        binding, cfg, detectors, waveform, build_prior(cfg.prior), []
    )
    return cfg, fine, detectors, waveform


def test_parameter_banks_are_deterministic_and_holdout_is_disjoint(tiny_network):
    cfg, _, detectors, _ = tiny_network
    settings = cfg.likelihood.heterodyne.bin_selection
    train, holdout = selection.parameter_banks(cfg, detectors, settings)
    again_train, again_holdout = selection.parameter_banks(cfg, detectors, settings)
    assert train == again_train
    assert holdout == again_holdout
    assert train[0] == holdout[0]
    assert selection._bank_hash(train) != selection._bank_hash(holdout)
    assert all(point not in train for point in holdout[1:])
    assert len(train) == 1 + 2 * len(cfg.prior.root) + 4 + settings.training_points
    assert len(holdout) == 1 + settings.verification_points


def test_noisy_network_selection_reuses_moments_and_matches_compiled_scorer(
    tiny_network, monkeypatch, record_property
):
    cfg, fine, detectors, waveform = tiny_network
    settings = cfg.likelihood.heterodyne.bin_selection
    train, verify = selection.parameter_banks(cfg, detectors, settings)
    bank = selection.SparseRatioBank(fine, train, verify, settings)
    oracle = selection.MomentErrorOracle(fine, bank, settings.tolerance)
    result = select_frozen_grid(
        fine.freq_grid_edges,
        lambda f: bank(f, "training"),
        lambda f: bank(f, "verification"),
        orders=(8,),
        bin_counts=settings.candidate_bins,
        max_bins=fine.n_bins,
        error_budget=settings.tolerance * 0.75,
        likelihood_error=oracle,
    )
    # Feed the same finished bank into integration; any second native build
    # must fail while ordinary reference waveform evaluation remains allowed.
    import jimgw.cli.xg_qualification as qualification

    def forbidden(*args, **kwargs):
        pytest.fail("selection must not reconstruct the native summary bank")

    monkeypatch.setattr(qualification, "build_xg_qualification_candidate", forbidden)
    monkeypatch.setattr(selection, "SparseRatioBank", lambda *args: bank)
    candidate, resolved, record = selection.select_network_binning(
        cfg, fine, detectors, waveform
    )
    assert candidate.n_bins == result.plan.n_bins
    assert candidate.n_bins <= 80
    assert (
        resolved.likelihood.heterodyne.frequency_bin_edges == result.plan.edges.tolist()
    )
    assert record["native_summary_builds"] == 1
    assert record["native_samples_read_during_selection"] == 0
    assert record["candidate_likelihood_compiles_during_accuracy_search"] == 0
    assert len(record["timed_valid_candidates"]) == 1
    timed = next(iter(record["timed_valid_candidates"].values()))
    assert timed["n_bins"] == 80
    assert timed["lanes"] == 2
    assert record["valid_candidate_compilation_seconds"] > 0
    assert record["maximum_verification_error_nats"] <= settings.tolerance * 0.75
    assert (
        max(record["reference_convergence_error_nats"].values())
        <= settings.tolerance / 4
    )
    np.testing.assert_array_equal(
        candidate._xg_node_frequency_prefix, fine.freq_grid_node_flat[:2]
    )
    ratios = bank.values["verification"][..., result.plan.node_indices]
    coefficients = np.einsum("ij,...bj->...bi", oracle.inverse, ratios)
    expected = oracle.score(result.plan.edges, coefficients, "verification")
    evaluate = jax.jit(jax.vmap(candidate.evaluate))
    points = {
        name: jnp.asarray([point[name] for point in verify]) for name in verify[0]
    }
    actual = np.asarray(evaluate(points))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
    record_property("selection_diagnostics", json.dumps(record))
    record_property(
        "maximum_scorer_parity_error", float(np.max(abs(actual - expected)))
    )


def test_sparse_ratio_chunks_retain_original_cutoff_prefix(monkeypatch):
    def waveform(frequency, params):
        amplitude = params["x"] + frequency[1] - frequency[0]
        return {
            "p": jnp.full_like(frequency, amplitude),
            "c": jnp.zeros_like(frequency),
            "__tau__": jnp.zeros_like(frequency),
        }

    class Detector:
        def frequency_dependent_antenna_pattern(self, ra, dec, psi, gmst, frequency):
            return {"p": jnp.ones_like(frequency), "c": jnp.zeros_like(frequency)}

    reference = {"x": 1.0, "ra": 0.0, "dec": 0.0, "psi": 0.0}
    proposal = {**reference, "x": 2.0}
    fine = SimpleNamespace(
        waveform=waveform,
        _reference_waveform=waveform,
        freq_grid_node_flat=jnp.asarray([2.0, 3.0, 4.0]),
        _xg_node_frequency_prefix=jnp.asarray([2.0, 2.01]),
        reference_parameters=reference,
        detectors=[Detector()],
        _prepare_parameters=lambda p: p,
    )
    monkeypatch.setattr(
        selection, "residual_response", lambda d, p, tau: (0.0, jnp.zeros_like(tau))
    )
    settings = SimpleNamespace(
        parameter_batch_size=2, frequency_chunk_size=32, max_bank_bytes=1048576
    )
    bank = selection.SparseRatioBank(fine, [proposal], [proposal], settings)
    frequencies = np.geomspace(2.0, 64.0, 69)
    actual = bank(frequencies, "training")
    np.testing.assert_allclose(actual, (2.0 + 0.01) / (1.0 + 0.01), rtol=0, atol=1e-14)
    assert bank.calls == 3
    assert bank(frequencies, "training") is actual
    assert bank.calls == 3
    with pytest.raises(ValueError, match="one frozen frequency bank"):
        bank(frequencies + 0.01, "training")
    small = SimpleNamespace(
        parameter_batch_size=2, frequency_chunk_size=32, max_bank_bytes=1
    )
    guarded = selection.SparseRatioBank(fine, [proposal], [proposal], small)
    with pytest.raises(ValueError, match="max_bank_bytes"):
        guarded(frequencies, "training")
    assert guarded.calls == 0


@pytest.mark.parametrize("method", ["uniform", "adaptive"])
def test_coarsened_physical_network_matches_shared_moment_scorer(
    tiny_network, monkeypatch, method
):
    cfg, fine, detectors, waveform = tiny_network
    cfg = cfg.model_copy(deep=True)
    cfg.likelihood.heterodyne.bin_selection = (
        cfg.likelihood.heterodyne.bin_selection.model_copy(
            update={
                "method": method,
                "candidate_bins": [20, 40, 80],
                "allocation_base_bins": 20,
            }
        )
    )
    observed = []
    oracle_type = selection.MomentErrorOracle

    def retain_oracle(*args):
        oracle = oracle_type(*args)
        observed.append(oracle)
        return oracle

    # Actual timing has its own test above. Deterministic costs exercise the
    # network coarsening path without compiling all candidate timing kernels.
    monkeypatch.setattr(
        selection.CandidateTimer, "__call__", lambda timer, plan: float(plan.n_bins)
    )
    monkeypatch.setattr(selection, "MomentErrorOracle", retain_oracle)
    candidate, _, record = selection.select_network_binning(
        cfg, fine, detectors, waveform
    )
    if method == "uniform":
        assert candidate.n_bins == 20
    else:
        assert candidate.n_bins < fine.n_bins
        assert record["allocation"]["compiled_finalists"] <= 3
    assert record["maximum_verification_error_nats"] < 0.0375
    oracle = observed[0]
    plan = FrozenGridPlan(np.asarray(candidate.freq_grid_edges), 8, 0.0)
    indices = np.searchsorted(oracle.bank.frequencies, plan.nodes)
    values = oracle.bank.values["verification"][..., indices]
    coefficients = np.einsum("ij,...bj->...bi", oracle.inverse, values)
    expected = oracle.score(plan.edges, coefficients, "verification")
    verify = oracle.bank.points["verification"]
    points = {
        name: jnp.asarray([point[name] for point in verify]) for name in verify[0]
    }
    actual = np.asarray(jax.jit(jax.vmap(candidate.evaluate))(points))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(
        candidate._xg_node_frequency_prefix, fine.freq_grid_node_flat[:2]
    )
