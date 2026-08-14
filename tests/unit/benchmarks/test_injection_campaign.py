import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign import (
    prepare_baseline_diagnostic as baseline_diagnostic,
)
from benchmarks.injection_campaign import prepare_fsm_diagnostic as fsm_diagnostic
from benchmarks.injection_campaign import prepare_historical_stress as historical_stress
from benchmarks.injection_campaign import run_campaign as run_campaign_module
from benchmarks.injection_campaign.plot_pp import (
    _remediation_eligibility,
    aggregate_and_plot,
    load_rank_rows,
)
from benchmarks.injection_campaign.plot_snr import aggregate_and_plot_snr
from benchmarks.injection_campaign.plot_timing import (
    aggregate_and_plot_timing,
    load_timings,
)
from benchmarks.injection_campaign.prepare_campaign import prepare_campaign
from benchmarks.injection_campaign.run_campaign import _is_complete
from benchmarks.injection_campaign.run_injection import (
    _analysis_components,
    _build_sampler_config,
    _implementation_report,
    _marginalized_parameters,
    _paper_convention_timing,
    _rank_truth_coordinates,
    _sampled_parameters,
    _time_marginalization_settings,
    _transient_likelihood_kwargs,
    _weighted_samples,
)


def _write_noise_curves(directory: Path) -> None:
    directory.mkdir()
    frequencies = np.asarray([1.0, 10.0, 20.0, 100.0, 1024.0, 4096.0])
    ligo_psd = 1.0e-46 * (1.0 + (100.0 / frequencies) ** 2)
    virgo_psd = 2.0e-46 * (1.0 + (100.0 / frequencies) ** 2)
    np.savetxt(
        directory / "aLIGO_O4_high_asd.txt",
        np.column_stack([frequencies, np.sqrt(ligo_psd)]),
    )
    np.savetxt(
        directory / "AdV_psd.txt",
        np.column_stack([frequencies, virgo_psd]),
    )


def _prepared_campaign(
    tmp_path: Path,
    n_injections: int = 4,
    *,
    phase_marginalization: bool = True,
) -> Path:
    curves = tmp_path / "curves"
    _write_noise_curves(curves)
    campaign = tmp_path / "campaign"
    prepare_campaign(
        campaign,
        n_injections=n_injections,
        catalogue_size=max(n_injections, 8),
        seed=1234,
        noise_curves_dir=curves,
        config_overrides={"phase_marginalization": phase_marginalization},
    )
    return campaign


def _write_pp_fixture_result(
    campaign: Path,
    manifest: dict[str, object],
    injection_id: int,
    *,
    legacy_rank_coordinates: bool = False,
    array_overrides: dict[str, np.ndarray] | None = None,
    summary_overrides: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Write a compact result whose ranks are derived from its frozen inputs."""

    row = common.read_catalogue(campaign / "catalogue.csv")[injection_id]
    truth = {
        name: row[name]
        for name in (*common.PARAMETERS, *common.MARGINALIZED_PARAMETERS)
    }
    config = manifest["config"]
    assert isinstance(config, dict)
    rank_truth = _rank_truth_coordinates(
        truth,
        common.PARAMETERS,
        phase_marginalization=bool(config["phase_marginalization"]),
    )
    override_lengths = {
        int(np.asarray(values).size) for values in (array_overrides or {}).values()
    }
    if len(override_lengths) > 1:
        raise ValueError("fixture array overrides must have one sample count")
    sample_count = next(iter(override_lengths), 2)
    offsets = np.linspace(-1.0e-6, 1.0e-6, sample_count)
    arrays = {
        name: np.asarray(rank_truth[name] + offsets) for name in common.PARAMETERS
    }
    arrays.update(array_overrides or {})
    log_weights = np.log(np.full(sample_count, 1.0 / sample_count))
    posterior_arrays = {
        **arrays,
        "log_likelihood": np.asarray([-2.0, -1.0]),
        "log_weights": log_weights,
    }
    directory = common.result_dir(campaign, injection_id)
    directory.mkdir(parents=True, exist_ok=True)
    posterior_path = directory / "posterior.npz"
    common.atomic_savez_compressed(posterior_path, posterior_arrays)
    stored_truth = (
        {name: float(truth[name]) for name in common.PARAMETERS}
        if legacy_rank_coordinates
        else rank_truth
    )
    summary: dict[str, object] = {
        "config_sha256": manifest["config_sha256"],
        "injection_id": injection_id,
        "truth": truth,
        "seeds": {"noise": row["noise_seed"], "sampler": row["sampler_seed"]},
        "posterior_samples": len(log_weights),
        "posterior": {
            "path": "posterior.npz",
            "sha256": common.file_sha256(posterior_path),
        },
        "ranks": {
            name: common.posterior_rank(arrays[name], stored_truth[name], log_weights)
            for name in common.PARAMETERS
        },
    }
    if not legacy_rank_coordinates:
        summary["rank_truth"] = rank_truth
    summary.update(summary_overrides or {})
    summary_path = directory / "summary.json"
    common.atomic_write_json(summary_path, summary)
    return summary_path, summary


def test_catalogue_is_deterministic_and_inside_the_recovery_prior() -> None:
    first = common.generate_catalogue(8, 42)
    second = common.generate_catalogue(1000, 42)[:8]

    assert first == second
    assert len({row["noise_seed"] for row in first}) == 8
    assert len({row["sampler_seed"] for row in first}) == 8
    for index, row in enumerate(first):
        assert row["injection_id"] == index
        assert 1.5 <= row["M_c"] <= 2.5
        assert 0.5 <= row["q"] <= 1.0
        assert 0.0 <= row["s1_mag"] <= 0.05
        assert 0.0 <= row["s2_mag"] <= 0.05
        assert 0.0 <= row["s1_theta"] <= np.pi
        assert 0.0 <= row["s2_theta"] <= np.pi
        assert 0.0 <= row["iota"] <= np.pi
        assert 30.0 <= row["d_L"] <= 150.0
        assert -np.pi / 2 <= row["dec"] <= np.pi / 2
        assert -0.1 <= row["t_c"] <= 0.1


def test_default_config_matches_paper_sharded_pp_protocol() -> None:
    config = common.DEFAULT_CONFIG

    assert common.PAPER_CATALOGUE_SIZE == 1000
    assert common.PAPER_PP_RECOVERIES == 100
    assert len(common.PARAMETERS) == 15
    assert "t_c" in common.PARAMETERS
    assert "d_L" not in common.PARAMETERS
    assert common.MARGINALIZED_PARAMETERS == ("phase_c", "d_L")
    assert config["paper_configuration"] == "Sharded"
    assert config["prior"]["M_c"]["range"] == [1.5, 2.5]
    assert config["prior"]["q"]["range"] == [0.5, 1.0]
    assert config["prior"]["d_L"]["range_mpc"] == [30.0, 150.0]
    assert config["prior"]["t_c"]["range_seconds"] == [-0.1, 0.1]
    assert config["phase_marginalization"] is True
    assert config["time_marginalization"] is False
    assert config["distance_marginalization"]["n_grid_points"] == 10000
    assert config["blocks"][-1] == ["t_c"]
    assert config["n_live"] == 512
    assert config["n_delete"] == 64
    assert config["num_gibbs_sweeps"] == 1
    assert config["n_devices"] == 4
    assert config["termination_log_z_live_minus_dead"] == -3.0
    assert config["trigger_time_gps"] == 1187008882.0
    assert config["segment_center_offset_seconds"] == -2.0
    assert config["segment_start_offset_seconds"] == -66.0
    assert config["f_min_hz"] == 20.0
    assert config["f_max_hz"] == 2048.0
    assert config["carrier_time_anchor"] == "imrphenomd"
    assert config["psd"]["H1"] == {
        "source": "aLIGO_O4_high_asd.txt",
        "source_quantity": "ASD",
    }
    assert config["timing"]["excluded_one_off_phases"] == [
        "likelihood_jit",
        "sampler_kernel_jit",
    ]


def test_recovery_components_sample_exactly_the_15_paper_parameters() -> None:
    import jax.numpy as jnp

    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1

    components = _analysis_components(
        common.DEFAULT_CONFIG,
        jnp,
        [get_H1(), get_L1(), get_V1()],
    )

    assert components["prior"].parameter_names == common.PARAMETERS
    assert components["distance_prior"].parameter_names == ("d_L",)
    sampling_names = components["prior"].parameter_names
    for transform in components["sample_transforms"]:
        sampling_names = transform.propagate_name(sampling_names)
    assert set(sampling_names) == (set(common.PARAMETERS) - {"ra", "dec"}) | {
        "zenith",
        "azimuth",
    }
    assert "t_c" not in components["periodic"]


def test_recovery_components_select_the_configured_carrier_time_anchor() -> None:
    import jax.numpy as jnp

    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1

    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config["carrier_time_anchor"] = "imrphenomd"
    components = _analysis_components(
        config,
        jnp,
        [get_H1(), get_L1(), get_V1()],
    )

    assert components["waveform"].time_anchor == "imrphenomd"


def test_recovery_components_can_sample_h1_arrival_time_before_detector_sky() -> None:
    import jax.numpy as jnp

    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
    from jimgw.core.single_event.transforms import (
        GeocentricArrivalTimeToDetectorArrivalTimeTransform,
        SkyFrameToDetectorFrameSkyPositionTransform,
    )

    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config["time_sampling_frame"] = "H1"
    components = _analysis_components(
        config,
        jnp,
        [get_H1(), get_L1(), get_V1()],
    )

    assert [type(transform) for transform in components["sample_transforms"]] == [
        GeocentricArrivalTimeToDetectorArrivalTimeTransform,
        SkyFrameToDetectorFrameSkyPositionTransform,
    ]
    sampling_names = components["prior"].parameter_names
    for transform in components["sample_transforms"]:
        sampling_names = transform.propagate_name(sampling_names)
    assert set(sampling_names) == (set(common.PARAMETERS) - {"ra", "dec", "t_c"}) | {
        "zenith",
        "azimuth",
        "t_det",
    }

    config["time_sampling_frame"] = "missing"
    with pytest.raises(ValueError, match="unknown detector"):
        _analysis_components(config, jnp, [get_H1(), get_L1(), get_V1()])


def test_phase_marginalized_spin_azimuth_rank_truth_uses_sampled_gauge() -> None:
    truth = {
        "M_c": 2.0,
        "s1_phi": 5.8,
        "s2_phi": 0.4,
        "phase_c": 1.0,
    }

    gauged = _rank_truth_coordinates(
        truth,
        ("M_c", "s1_phi", "s2_phi"),
        phase_marginalization=True,
    )
    ungauged = _rank_truth_coordinates(
        truth,
        ("M_c", "s1_phi", "s2_phi"),
        phase_marginalization=False,
    )

    assert gauged["M_c"] == 2.0
    assert gauged["s1_phi"] == pytest.approx((5.8 + 1.0) % (2.0 * np.pi))
    assert gauged["s2_phi"] == pytest.approx(1.4)
    assert ungauged == {"M_c": 2.0, "s1_phi": 5.8, "s2_phi": 0.4}


def test_appendix_a_parameter_treatment_samples_distance_and_marginalizes_time():
    import jax.numpy as jnp

    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1

    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config["time_marginalization"] = {
        "tc_range_seconds": [-0.1, 0.1],
        "upsample_factor": 32,
    }
    config["distance_marginalization"] = False
    config["likelihood_f_max_hz"] = 2048.0 - 1.0 / 128.0
    config["blocks"][-1] = ["d_L"]
    components = _analysis_components(
        config,
        jnp,
        [get_H1(), get_L1(), get_V1()],
    )

    expected_sampled = tuple(name for name in common.PARAMETERS if name != "t_c") + (
        "d_L",
    )
    assert _sampled_parameters(config) == expected_sampled
    assert _marginalized_parameters(config) == ("phase_c", "t_c")
    assert components["prior"].parameter_names == expected_sampled
    likelihood_kwargs = _transient_likelihood_kwargs(config, components)
    assert likelihood_kwargs["f_max"] == pytest.approx(2048.0 - 1.0 / 128.0)
    assert likelihood_kwargs["time_marginalization"] == {
        "tc_range": (-0.1, 0.1),
        "upsample_factor": 32,
    }
    assert likelihood_kwargs["distance_marginalization"] is None


def test_attribution_likelihood_axes_are_forwarded_without_collapsing() -> None:
    config = copy.deepcopy(common.DEFAULT_CONFIG)
    axes = {
        "shared_frequency_grid": True,
        "detector_phasor": False,
        "real_inner_product": True,
    }
    config["likelihood_implementation"] = "baseline"
    config["likelihood_optimization_axes"] = axes

    kwargs = _transient_likelihood_kwargs(
        config,
        {"distance_prior": object()},
    )

    assert kwargs["likelihood_optimizations"] is False
    assert kwargs["likelihood_optimization_axes"] == axes
    assert kwargs["likelihood_optimization_axes"] is not axes


@pytest.mark.parametrize("upsample_factor", [0, True, 1.9, "32"])
def test_time_marginalization_requires_an_exact_positive_upsample_factor(
    upsample_factor: object,
) -> None:
    config = copy.deepcopy(common.DEFAULT_CONFIG)
    config["time_marginalization"] = {
        "tc_range_seconds": [-0.1, 0.1],
        "upsample_factor": upsample_factor,
    }

    with pytest.raises(ValueError, match="exact positive integer"):
        _time_marginalization_settings(config)


def test_prepare_campaign_round_trip_and_input_integrity(tmp_path: Path) -> None:
    campaign = _prepared_campaign(tmp_path)

    manifest = common.load_manifest(campaign)
    catalogue = common.read_catalogue(campaign / "catalogue.csv")

    assert manifest["n_injections"] == 4
    assert manifest["catalogue_size"] == 8
    assert manifest["config"]["n_devices"] == 4
    assert len(catalogue) == 8
    assert (campaign / "status.csv").is_file()
    with np.load(campaign / "inputs/psd/aLIGO-O4-high.npz") as archive:
        assert archive["frequencies"].shape == (262145,)
        assert np.all(np.isfinite(archive["values"]))
        assert np.all(archive["values"] > 0)
        assert archive["source_quantity"].item() == "ASD"
        frequency_index = int(20.0 * 128.0)
        expected_psd = 1.0e-46 * (1.0 + (100.0 / 20.0) ** 2)
        assert archive["values"][frequency_index] == pytest.approx(expected_psd)

    catalogue_path = campaign / "catalogue.csv"
    catalogue_path.write_text(catalogue_path.read_text() + "\n")
    with pytest.raises(ValueError, match="catalogue hash mismatch"):
        common.load_manifest(campaign)


def test_prepare_refuses_to_replace_an_existing_campaign(tmp_path: Path) -> None:
    campaign = _prepared_campaign(tmp_path)
    curves = tmp_path / "curves"

    with pytest.raises(FileExistsError, match="campaign already exists"):
        prepare_campaign(
            campaign,
            n_injections=4,
            catalogue_size=8,
            seed=1234,
            noise_curves_dir=curves,
        )


def test_prepare_baseline_diagnostic_reindexes_exact_rows_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    source = _prepared_campaign(tmp_path, n_injections=100)
    source_manifest = common.load_manifest(source)
    source_rows = common.read_catalogue(source / "catalogue.csv")
    output = tmp_path / "baseline-diagnostic"

    manifest = baseline_diagnostic.prepare_baseline_diagnostic(source, output)

    expected_source_ids = (97, 12, 40, 41, 32)
    assert baseline_diagnostic.DEFAULT_SOURCE_IDS == expected_source_ids
    assert manifest["selection"]["source_injection_ids"] == list(expected_source_ids)
    assert manifest["n_injections"] == 5
    assert manifest["catalogue_size"] == 5
    selected_rows = common.read_catalogue(output / "catalogue.csv")
    assert [row["injection_id"] for row in selected_rows] == list(range(5))
    for diagnostic_id, source_id in enumerate(expected_source_ids):
        assert selected_rows[diagnostic_id] == {
            **source_rows[source_id],
            "injection_id": diagnostic_id,
        }

    mapping = manifest["catalogue"]["provenance"]["mapping"]
    assert [entry["diagnostic_id"] for entry in mapping] == list(range(5))
    assert [entry["source_injection_id"] for entry in mapping] == list(
        expected_source_ids
    )
    assert [entry["noise_seed"] for entry in mapping] == [
        source_rows[source_id]["noise_seed"] for source_id in expected_source_ids
    ]
    assert [entry["sampler_seed"] for entry in mapping] == [
        source_rows[source_id]["sampler_seed"] for source_id in expected_source_ids
    ]
    assert (
        manifest["catalogue"]["provenance"]["source_catalogue_sha256"]
        == source_manifest["catalogue"]["sha256"]
    )
    assert (
        manifest["catalogue"]["provenance"]["source_config_sha256"]
        == source_manifest["config_sha256"]
    )

    source_inputs = {
        path.relative_to(source / "inputs"): common.file_sha256(path)
        for path in (source / "inputs").rglob("*")
        if path.is_file()
    }
    diagnostic_inputs = {
        path.relative_to(output / "inputs"): common.file_sha256(path)
        for path in (output / "inputs").rglob("*")
        if path.is_file()
    }
    assert diagnostic_inputs == source_inputs
    assert manifest["psd"] == source_manifest["psd"]

    config = manifest["config"]
    assert config["campaign"] == (
        "paper-baseline-code-high-res-d1-m3-pathology-diagnostic"
    )
    assert config["paper_configuration"] == "High-Res"
    assert "sampler_scheduler" not in config
    assert config["n_devices"] == 1
    assert config["num_gibbs_sweeps"] == 3
    diagnostic = manifest["baseline_diagnostic"]
    assert diagnostic["implementation_revision"] == (
        "86335bdb1e7ef6191937dd17b2ca53edbb1d899f"
    )
    assert diagnostic["implementation_tree_sha256"] == (
        "09085b4d427cfbbb9b379228b2b5d6cb781042687c64f0207740d1af3354c141"
    )
    assert diagnostic["paper_configuration"] == "High-Res"
    assert diagnostic["configuration_semantics"] == (
        "Table II High-Res settings (D=1, M=3) executed with the pinned "
        "paper-baseline implementation"
    )
    assert diagnostic["paper_timing_available"] is False
    assert diagnostic["changed_variables"]["paper_configuration"] == {
        "source": "Sharded",
        "diagnostic": "High-Res",
    }
    assert diagnostic["changed_variables"]["n_devices"] == {
        "source": 4,
        "diagnostic": 1,
    }
    assert diagnostic["changed_variables"]["num_gibbs_sweeps"] == {
        "source": 1,
        "diagnostic": 3,
    }
    assert common.load_manifest(output) == manifest
    assert [row["status"] for row in common.status_rows(output, 5)] == ["pending"] * 5


def test_prepare_fsm_diagnostic_changes_only_m_and_pins_candidate(
    tmp_path: Path,
) -> None:
    source = _prepared_campaign(tmp_path, n_injections=100)
    source_manifest = common.load_manifest(source)
    source_rows = common.read_catalogue(source / "catalogue.csv")
    for source_id in fsm_diagnostic.DEFAULT_SOURCE_IDS:
        summary = common.result_dir(source, source_id) / "summary.json"
        summary.parent.mkdir(parents=True)
        common.atomic_write_json(
            summary,
            {"ranks": {"q": 0.1, "ra": 0.2, "t_c": 0.3}},
        )
    output = tmp_path / "fsm-diagnostic"
    revision = "a" * 40
    tree_sha256 = "b" * 64

    manifest = fsm_diagnostic.prepare_fsm_diagnostic(
        source,
        output,
        implementation_revision=revision,
        implementation_tree_sha256=tree_sha256,
    )

    assert manifest["selection"]["source_injection_ids"] == [97, 12, 41, 32]
    assert manifest["n_injections"] == manifest["catalogue_size"] == 4
    selected_rows = common.read_catalogue(output / "catalogue.csv")
    for diagnostic_id, source_id in enumerate((97, 12, 41, 32)):
        assert selected_rows[diagnostic_id] == {
            **source_rows[source_id],
            "injection_id": diagnostic_id,
        }

    config = manifest["config"]
    source_config = source_manifest["config"]
    assert config["n_devices"] == source_config["n_devices"] == 4
    assert source_config["num_gibbs_sweeps"] == 1
    assert config["num_gibbs_sweeps"] == 3
    assert config["sampler_scheduler"] == "fsm"
    assert config["paper_configuration"] == "Sharded-M3 diagnostic"
    assert manifest["reproduction_scope"]["pp_calibration_eligible"] is False
    assert manifest["psd"] == source_manifest["psd"]
    assert {
        path.relative_to(source / "inputs"): common.file_sha256(path)
        for path in (source / "inputs").rglob("*")
        if path.is_file()
    } == {
        path.relative_to(output / "inputs"): common.file_sha256(path)
        for path in (output / "inputs").rglob("*")
        if path.is_file()
    }
    pin = manifest["implementation_diagnostic"]
    assert pin["implementation_label"] == "candidate"
    assert pin["implementation_revision"] == revision
    assert pin["implementation_tree_sha256"] == tree_sha256
    assert common.load_manifest(output) == manifest
    assert [row["status"] for row in common.status_rows(output, 4)] == ["pending"] * 4


def test_sampler_config_supports_current_and_pinned_baseline_scheduler_apis() -> None:
    class CurrentConfig:
        model_fields: ClassVar[dict[str, object]] = {"scheduler": object()}

        def __init__(self, **values: object) -> None:
            self.values = values

    class PinnedBaselineConfig:
        model_fields: ClassVar[dict[str, object]] = {}

        def __init__(self, **values: object) -> None:
            self.values = values

    config = dict(common.DEFAULT_CONFIG)
    config.pop("sampler_scheduler", None)
    config["n_devices"] = 1
    config["num_gibbs_sweeps"] = 3

    current = _build_sampler_config(config, CurrentConfig)
    baseline = _build_sampler_config(config, PinnedBaselineConfig)

    assert current.values["scheduler"] == "fsm"
    assert "scheduler" not in baseline.values
    for built in (current, baseline):
        assert built.values["n_devices"] == 1
        assert built.values["num_gibbs_sweeps"] == 3
        assert built.values["blocks"] == common.DEFAULT_CONFIG["blocks"]
        assert built.values["n_live"] == common.DEFAULT_CONFIG["n_live"]


def test_candidate_implementation_pin_is_checked_before_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "candidate"
    module = root / "src/jimgw/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    revision = "a" * 40
    tree_sha256 = "b" * 64
    manifest = {
        "implementation_diagnostic": {
            "implementation_label": "candidate",
            "implementation_revision": revision,
            "implementation_tree_sha256": tree_sha256,
        }
    }

    class CurrentConfig:
        model_fields: ClassVar[dict[str, object]] = {"scheduler": object()}

    monkeypatch.setenv("JIM_IMPLEMENTATION_LABEL", "candidate")
    monkeypatch.setenv("JIM_IMPLEMENTATION_REVISION", revision)
    monkeypatch.setenv("JIM_IMPLEMENTATION_TREE_SHA256", tree_sha256)
    monkeypatch.setenv("JIM_IMPLEMENTATION_ROOT", str(root))

    report = _implementation_report(
        manifest,
        SimpleNamespace(__file__=str(module)),
        CurrentConfig,
    )

    assert report["label"] == "candidate"
    assert report["tree_sha256"] == tree_sha256
    monkeypatch.setenv("JIM_IMPLEMENTATION_TREE_SHA256", "c" * 64)
    with pytest.raises(RuntimeError, match="requires the pinned implementation"):
        _implementation_report(
            manifest,
            SimpleNamespace(__file__=str(module)),
            CurrentConfig,
        )


def test_weighted_samples_recovers_direct_ns_weights_from_pinned_baseline() -> None:
    import jax
    import jax.numpy as jnp

    sample_array = np.asarray([[0.2, 1.2], [0.4, 1.4], [0.8, 1.8]])
    log_likelihood = np.asarray([-3.0, -2.0, -1.0])
    log_likelihood_birth = np.asarray([-np.inf, -3.0, -2.0])
    raw_log_weights = np.log(np.asarray([2.0, 3.0, 5.0]))

    class ILoc:
        def __getitem__(self, index: tuple[slice, slice]) -> np.ndarray:
            rows, columns = index
            return sample_array[rows, columns]

    class NestedSamples:
        iloc = ILoc()

        def __getitem__(self, name: str) -> np.ndarray:
            return {
                "logL": log_likelihood,
                "logL_birth": log_likelihood_birth,
            }[name]

        def logw(self) -> np.ndarray:
            return raw_log_weights

    jim = SimpleNamespace(
        sampler=SimpleNamespace(_nested_samples=NestedSamples(), n_dims=2),
        add_name=lambda row: {"q": row[0], "t_c": row[1]},
        sample_transforms=[],
        prior_parameter_names=("q", "t_c"),
    )

    weighted = _weighted_samples(jim, jax, jnp)

    np.testing.assert_allclose(weighted["q"], sample_array[:, 0])
    np.testing.assert_allclose(weighted["t_c"], sample_array[:, 1])
    np.testing.assert_allclose(weighted["log_likelihood"], log_likelihood)
    np.testing.assert_allclose(weighted["log_likelihood_birth"], log_likelihood_birth)
    np.testing.assert_allclose(
        weighted["log_weights"], np.log(np.asarray([0.2, 0.3, 0.5]))
    )
    assert np.exp(weighted["log_weights"]).sum() == pytest.approx(1.0)


def test_weighted_samples_uses_current_direct_weighted_api() -> None:
    expected = {
        "q": np.asarray([0.4, 0.6]),
        "log_likelihood": np.asarray([-2.0, -1.0]),
        "log_weights": np.log(np.asarray([0.25, 0.75])),
    }
    jim = SimpleNamespace(get_weighted_samples=lambda: expected)

    weighted = _weighted_samples(jim, None, None)

    assert set(weighted) == set(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(weighted[name], values)


def test_historical_stress_mapping_preserves_quantiles_and_seeds() -> None:
    source = {
        name: str(value) for name, value in common.generate_catalogue(1, 7)[0].items()
    }
    source.update(
        {
            "injection_id": "0",
            "noise_seed": "3833912172",
            "sampler_seed": "2532440150",
            "M_c": "1.1998190676308333",
            "q": "0.5426643258069528",
            "d_L": "11.522859203888984",
            "t_c": "0.005132786507252289",
        }
    )

    mapped = historical_stress.map_historical_row(source, preflight_id=4)

    assert mapped["injection_id"] == 4
    assert mapped["noise_seed"] == 3833912172
    assert mapped["sampler_seed"] == 2532440150
    assert mapped["M_c"] == pytest.approx(2.1606355876944465)
    assert mapped["q"] == pytest.approx(0.7386653290325444)
    assert mapped["d_L"] == pytest.approx(33.950873282935746)
    assert mapped["t_c"] == pytest.approx(0.01710928835750762)
    for name in set(common.CATALOGUE_FIELDS) - {
        "injection_id",
        "noise_seed",
        "sampler_seed",
        "M_c",
        "q",
        "d_L",
        "t_c",
    }:
        assert mapped[name] == pytest.approx(float(source[name]))


def test_prepare_historical_stress_preserves_pinned_order_and_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curves = tmp_path / "curves"
    _write_noise_curves(curves)
    legacy_rows = common.generate_catalogue(100, 91)
    for row in legacy_rows:
        row["M_c"] = 1.18 + (float(row["M_c"]) - 1.5) * 0.03
        row["q"] = 0.125 + (float(row["q"]) - 0.5) * 1.75
        row["t_c"] = float(row["t_c"]) * 0.3
        row["d_L"] = 50.0
    source = tmp_path / "legacy.csv"
    common.atomic_write_csv(
        source,
        legacy_rows,
        historical_stress.LEGACY_CATALOGUE_FIELDS,
    )
    monkeypatch.setattr(
        historical_stress,
        "HISTORICAL_SOURCE_SHA256",
        common.file_sha256(source),
    )

    campaign = tmp_path / "historical-stress"
    manifest = historical_stress.prepare_historical_stress(
        campaign,
        source_catalogue=source,
        noise_curves_dir=curves,
    )

    provenance = manifest["catalogue"]["provenance"]
    assert [entry["historical_id"] for entry in provenance["mapping"]] == list(
        historical_stress.HISTORICAL_IDS
    )
    assert provenance["source_catalogue_sha256"] == common.file_sha256(source)
    assert manifest["config"]["campaign"].endswith("historical-stress")
    assert manifest["config"]["timing"]["selected_events"].startswith("targeted")
    assert manifest["reproduction_scope"]["pp_calibration_eligible"] is False
    assert len(common.read_catalogue(campaign / "catalogue.csv")) == 10

    baseline_campaign = tmp_path / "historical-stress-pre-fsm-lockstep"
    baseline_manifest = historical_stress.prepare_historical_stress(
        baseline_campaign,
        source_catalogue=source,
        noise_curves_dir=curves,
        sampler_scheduler="pre-fsm-lockstep",
    )
    assert baseline_manifest["config"]["n_devices"] == 4
    assert baseline_manifest["config"]["sampler_scheduler"] == "pre-fsm-lockstep"
    assert (
        baseline_manifest["config"]["campaign"]
        == (historical_stress.SAMPLER_SCHEDULERS["pre-fsm-lockstep"])
    )
    assert baseline_manifest["catalogue"]["sha256"] == manifest["catalogue"]["sha256"]


def test_explicit_stress_catalogue_is_labelled_non_calibration(
    tmp_path: Path,
) -> None:
    curves = tmp_path / "curves"
    _write_noise_curves(curves)
    rows = common.generate_catalogue(2, 17)
    campaign = tmp_path / "stress"

    manifest = prepare_campaign(
        campaign,
        n_injections=2,
        catalogue_size=2,
        seed=17,
        noise_curves_dir=curves,
        catalogue_rows=rows,
        catalogue_provenance={"kind": "targeted-test"},
        campaign_name="targeted-stress",
    )

    assert manifest["config"]["campaign"] == "targeted-stress"
    assert manifest["selection"]["rule"] == "explicit ordered stress catalogue"
    assert manifest["catalogue"]["provenance"] == {"kind": "targeted-test"}
    assert manifest["catalogue"]["generator"] == "explicit ordered rows"
    assert manifest["master_seed"] is None
    assert manifest["reproduction_scope"]["iid_prior_predictive_catalogue"] is False
    assert manifest["reproduction_scope"]["pp_calibration_eligible"] is False
    assert common.load_manifest(campaign)["config_sha256"] == manifest["config_sha256"]
    with pytest.raises(ValueError, match="targeted, non-iid"):
        aggregate_and_plot(campaign)
    with pytest.raises(ValueError, match="targeted, non-iid"):
        load_timings(campaign)


def test_explicit_catalogue_requires_ordered_contiguous_ids(tmp_path: Path) -> None:
    curves = tmp_path / "curves"
    _write_noise_curves(curves)
    rows = common.generate_catalogue(2, 17)
    rows[1]["injection_id"] = 7

    with pytest.raises(ValueError, match="contiguous and ordered"):
        prepare_campaign(
            tmp_path / "campaign",
            n_injections=2,
            catalogue_size=2,
            seed=17,
            noise_curves_dir=curves,
            catalogue_rows=rows,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("injection_id", 0.5, "exact integer"),
        ("noise_seed", 2**32, "uint32"),
        ("q", np.nan, "finite and numeric"),
        ("q", 0.49, "outside"),
    ],
)
def test_explicit_catalogue_rejects_invalid_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    curves = tmp_path / "curves"
    _write_noise_curves(curves)
    rows = common.generate_catalogue(1, 17)
    rows[0][field] = value

    with pytest.raises(ValueError, match=message):
        prepare_campaign(
            tmp_path / "campaign",
            n_injections=1,
            catalogue_size=1,
            seed=17,
            noise_curves_dir=curves,
            catalogue_rows=rows,
        )


def test_posterior_rank_uses_direct_log_weights_and_strict_indicator() -> None:
    samples = np.asarray([0.0, 1.0, 1.0, 2.0])
    log_weights = np.log(np.asarray([0.1, 0.2, 0.3, 0.4]))

    assert common.posterior_rank(samples, 1.0, log_weights) == pytest.approx(0.1)
    assert common.posterior_rank(samples, 1.5, log_weights) == pytest.approx(0.6)
    assert common.posterior_rank(samples, 1.5, log_weights + 1000.0) == pytest.approx(
        0.6
    )
    with pytest.raises(ValueError, match="non-empty"):
        common.posterior_rank(np.asarray([]), 1.0, np.asarray([]))
    with pytest.raises(ValueError, match="same shape"):
        common.posterior_rank(samples, 1.0, log_weights[:-1])
    with pytest.raises(ValueError, match=r"NaN or \+inf"):
        common.posterior_rank(samples, 1.0, np.asarray([0.0, 0.0, np.nan, 0.0]))


def test_posterior_rank_clips_floating_point_endpoint_overshoot() -> None:
    log_weights = np.asarray(
        [
            -17.33996714867339,
            -34.01224552250915,
            -42.08341774096238,
            -11.458201625536885,
            -16.431282532221385,
            -3.865712980185554,
            -36.391781899261666,
            -28.184252761292306,
            -25.26192052062462,
            -44.93151222591784,
            -24.529154940058444,
            -45.2785581842991,
            -7.442390930842279,
            -42.38516108126124,
            -15.423003996225503,
            -30.684440119490286,
        ]
    )
    samples = np.zeros(log_weights.size)
    samples[2] = 2.0

    assert common.posterior_rank(samples, 1.0, log_weights) == 1.0


def test_pp_loader_clips_only_machine_precision_rank_overshoot(
    tmp_path: Path,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=1, phase_marginalization=False)
    manifest = common.load_manifest(campaign)
    row = common.read_catalogue(campaign / "catalogue.csv")[0]
    summary_path, summary = _write_pp_fixture_result(
        campaign,
        manifest,
        0,
        array_overrides={
            "t_c": np.asarray([float(row["t_c"]) - 2.0e-6, float(row["t_c"]) - 1e-6])
        },
    )
    ranks = summary["ranks"]
    assert isinstance(ranks, dict)
    ranks["t_c"] = 1.0 + np.finfo(float).eps
    common.atomic_write_json(summary_path, summary)

    _, rows = load_rank_rows(campaign)
    assert rows[0]["t_c"] == 1.0

    ranks["t_c"] = 1.0 + 1.0e-8
    common.atomic_write_json(summary_path, summary)
    with pytest.raises(ValueError, match="stored rank mismatch for t_c"):
        load_rank_rows(campaign)


def test_pp_loader_corrects_legacy_phase_gauge_ranks_from_posterior(
    tmp_path: Path,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=1)
    manifest = common.load_manifest(campaign)
    summary_path, summary = _write_pp_fixture_result(
        campaign,
        manifest,
        0,
        legacy_rank_coordinates=True,
        array_overrides={
            "s1_phi": np.asarray([0.1, 0.4, 0.8, 1.2]),
            "s2_phi": np.asarray([0.2, 0.5, 1.0, 1.8]),
        },
    )

    _, rows = load_rank_rows(campaign)

    truth = summary["truth"]
    assert isinstance(truth, dict)
    expected_truth = _rank_truth_coordinates(
        truth,
        common.PARAMETERS,
        phase_marginalization=True,
    )
    with np.load(summary_path.parent / "posterior.npz") as posterior:
        for name in ("s1_phi", "s2_phi"):
            assert rows[0][name] == pytest.approx(
                common.posterior_rank(
                    posterior[name], expected_truth[name], posterior["log_weights"]
                )
            )
    assert rows[0]["_legacy_phase_gauge_corrected_parameters"] == [
        "s1_phi",
        "s2_phi",
    ]

    summary["rank_truth"] = {}
    common.atomic_write_json(summary_path, summary)
    with pytest.raises(ValueError, match="invalid rank truth inventory"):
        load_rank_rows(campaign)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("truth", "truth mismatch for phase_c"),
        ("seed", "result seeds do not match the catalogue"),
        ("rank", "stored rank mismatch for ra"),
    ],
)
def test_pp_loader_binds_truth_seeds_and_all_ranks_to_frozen_inputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=1)
    manifest = common.load_manifest(campaign)
    summary_path, summary = _write_pp_fixture_result(campaign, manifest, 0)

    if mutation == "truth":
        truth = summary["truth"]
        assert isinstance(truth, dict)
        truth["phase_c"] = float(truth["phase_c"]) + 0.1
    elif mutation == "seed":
        seeds = summary["seeds"]
        assert isinstance(seeds, dict)
        seeds["sampler"] = int(seeds["sampler"]) + 1
    else:
        ranks = summary["ranks"]
        assert isinstance(ranks, dict)
        ranks["ra"] = 0.123456789
    common.atomic_write_json(summary_path, summary)

    with pytest.raises(ValueError, match=message):
        load_rank_rows(campaign)


def test_remediation_eligibility_requires_full_corrected_m1_protocol() -> None:
    manifest = {
        "n_injections": common.PAPER_PP_RECOVERIES,
        "config": copy.deepcopy(common.DEFAULT_CONFIG),
    }

    eligible, reasons = _remediation_eligibility(manifest, is_complete=True)
    assert eligible is True
    assert reasons == []

    manifest["config"]["carrier_time_anchor"] = "nrtidal-merger"
    manifest["config"]["num_gibbs_sweeps"] = 2
    eligible, reasons = _remediation_eligibility(manifest, is_complete=True)
    assert eligible is False
    assert "config.carrier_time_anchor must equal 'imrphenomd'" in reasons
    assert "config.num_gibbs_sweeps must equal 1" in reasons

    manifest["config"] = copy.deepcopy(common.DEFAULT_CONFIG)
    manifest["n_injections"] = 5
    eligible, reasons = _remediation_eligibility(manifest, is_complete=False)
    assert eligible is False
    assert "selected recovery set is incomplete" in reasons
    assert "requires exactly 100 selected recoveries" in reasons


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("waveform", "ToyWaveform"),
        ("sampling_frequency_hz", 2048.0),
        ("detectors", ["H1", "L1"]),
        ("sampler_scheduler", "pre-fsm-lockstep"),
    ],
)
def test_remediation_eligibility_rejects_material_methodology_changes(
    field: str,
    value: object,
) -> None:
    manifest = {
        "n_injections": common.PAPER_PP_RECOVERIES,
        "config": copy.deepcopy(common.DEFAULT_CONFIG),
    }
    manifest["config"][field] = value

    eligible, reasons = _remediation_eligibility(manifest, is_complete=True)

    assert eligible is False
    assert any(reason.startswith(f"config.{field} must equal") for reason in reasons)


def test_remediation_eligibility_pins_priors_but_allows_valid_reblocking() -> None:
    manifest = {
        "n_injections": common.PAPER_PP_RECOVERIES,
        "config": copy.deepcopy(common.DEFAULT_CONFIG),
    }
    manifest["config"]["prior"]["q"]["range"] = [0.25, 1.0]

    eligible, reasons = _remediation_eligibility(manifest, is_complete=True)

    assert eligible is False
    assert any(reason.startswith("config.prior must equal") for reason in reasons)

    manifest["config"] = copy.deepcopy(common.DEFAULT_CONFIG)
    manifest["config"]["blocks"] = [
        *common.DEFAULT_CONFIG["blocks"][:3],
        ["iota", "zenith", "azimuth", "psi", "t_c"],
    ]
    eligible, reasons = _remediation_eligibility(manifest, is_complete=True)
    assert eligible is True
    assert reasons == []


def test_paper_timing_excludes_exactly_the_two_one_off_jit_phases() -> None:
    timing = _paper_convention_timing(
        100.0,
        {
            "likelihood_jit": 20.0,
            "sampler_kernel_jit": 30.0,
            "ns_loop": 40.0,
        },
    )

    assert timing["likelihood_jit_seconds"] == pytest.approx(20.0)
    assert timing["sampler_jit_seconds"] == pytest.approx(30.0)
    assert timing["post_jit_sampling_seconds"] == pytest.approx(50.0)
    with pytest.raises(RuntimeError, match="sampler_kernel_jit"):
        _paper_convention_timing(100.0, {"likelihood_jit": 20.0})


def test_status_and_pp_outputs_are_derived_from_compact_summaries(
    tmp_path: Path,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=5, phase_marginalization=False)
    manifest = common.load_manifest(campaign)
    for injection_id in range(5):
        _write_pp_fixture_result(
            campaign,
            manifest,
            injection_id,
            summary_overrides={
                "timing_seconds": {
                    "sample_call": injection_id + 20.0,
                    "total": injection_id + 30.0,
                    "paper_convention": {
                        "likelihood_jit_seconds": 2.0,
                        "sampler_jit_seconds": 3.0,
                        "post_jit_sampling_seconds": injection_id + 15.0,
                    },
                },
            },
        )

    rows = common.refresh_status(campaign, 5)
    report = aggregate_and_plot(campaign)
    timing_report = aggregate_and_plot_timing(
        campaign, pdf_output=tmp_path / "figure-3.pdf"
    )

    assert {row["status"] for row in rows} == {"complete"}
    assert all(row["attempts"] == 1 for row in rows)
    assert report["completed_injections"] == 5
    assert report["diagnostic_status"] == "complete"
    assert len(report["per_parameter"]) == len(common.PARAMETERS)
    assert report["methodology"]["combined_test"]["number_of_pvalues"] == 15
    assert report["remediation_assessment"]["excluded_parameters"] == []
    assert report["remediation_assessment"]["exclusion_reason"] is None
    assert report["remediation_assessment"]["combined_test"]["number_of_pvalues"] == 15
    assert report["remediation_assessment"]["eligible"] is False
    assert report["remediation_assessment"]["passes"] is None
    assert (
        "requires exactly 100 selected recoveries"
        in report["remediation_assessment"]["ineligibility_reasons"]
    )
    assert [band["sigma"] for band in report["methodology"]["confidence_bands"]] == [
        1,
        2,
        3,
    ]
    assert timing_report["completed_injections"] == 5
    assert timing_report["timing_definition"].startswith("jim.sample wall time")
    assert timing_report["x_scale"] == "linear"
    assert timing_report["figure_3_summary"]["measurement"] == "Sharded"
    assert timing_report["figure_3_summary"]["median_seconds"] == pytest.approx(17.0)
    for relative in report["outputs"]:
        assert (campaign / relative).is_file()
    assert (campaign / "pp/pp-combined.png").stat().st_size > 0
    assert (campaign / "pp/pp-grid.png").stat().st_size > 0
    assert (campaign / "timing/figure-3-equivalent.png").stat().st_size > 0
    assert (campaign / "timing/figure-3-summary.csv").is_file()
    with (campaign / "timing/per-injection-timing.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        timing_rows = list(csv.DictReader(stream))
    assert [float(row["post_jit_sampling_seconds"]) for row in timing_rows] == [
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
    ]
    assert (campaign / "timing/report.json").is_file()
    assert (tmp_path / "figure-3.pdf").stat().st_size > 0


def test_pp_publication_requires_the_complete_selected_leading_set(
    tmp_path: Path,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=2, phase_marginalization=False)
    manifest = common.load_manifest(campaign)
    _write_pp_fixture_result(campaign, manifest, 0)

    with pytest.raises(ValueError, match="incomplete selected leading-ID set"):
        aggregate_and_plot(campaign)

    report = aggregate_and_plot(campaign, allow_partial=True)
    assert report["diagnostic_status"] == "partial-exploratory"
    assert report["selection"]["included_injection_ids"] == [0]
    assert report["selection"]["missing_injection_ids"] == [1]
    assert report["selection"]["complete"] is False
    assert report["remediation_assessment"]["eligible"] is False
    assert report["remediation_assessment"]["passes"] is None


def test_figure_3_timing_requires_complete_post_jit_measurements(
    tmp_path: Path,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=2)
    manifest = common.load_manifest(campaign)
    directory = common.result_dir(campaign, 0)
    directory.mkdir(parents=True)
    common.atomic_write_json(
        directory / "summary.json",
        {
            "config_sha256": manifest["config_sha256"],
            "injection_id": 0,
            "timing_seconds": {
                "sample_call": 100.0,
                "total": 120.0,
                "paper_convention": {
                    "likelihood_jit_seconds": 20.0,
                    "sampler_jit_seconds": 30.0,
                    "post_jit_sampling_seconds": 50.0,
                },
            },
        },
    )

    with pytest.raises(ValueError, match="incomplete selected leading-ID set"):
        load_timings(campaign)

    _, rows = load_timings(campaign, allow_partial=True)
    assert rows[0]["post_jit_sampling_seconds"] == pytest.approx(50.0)

    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["timing_seconds"].pop("paper_convention")
    common.atomic_write_json(summary_path, summary)
    with pytest.raises(TypeError, match="missing Figure 3 post-JIT timing"):
        load_timings(campaign, allow_partial=True)


def test_figure_7_snr_plot_uses_the_complete_evaluated_selection(
    tmp_path: Path,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=4)
    manifest = common.load_manifest(campaign)
    network_snrs = (10.0, 20.0, 30.0, 50.0)
    for injection_id, network_snr in enumerate(network_snrs):
        directory = common.result_dir(campaign, injection_id)
        directory.mkdir(parents=True)
        common.atomic_write_json(
            directory / "summary.json",
            {
                "config_sha256": manifest["config_sha256"],
                "injection_id": injection_id,
                "network": {
                    "optimal_snr": network_snr,
                    "optimal_snr_by_detector": {
                        "H1": network_snr / 2.0,
                        "L1": network_snr / 2.0,
                        "V1": network_snr / 3.0,
                    },
                },
            },
        )

    report = aggregate_and_plot_snr(campaign)

    assert report["diagnostic_status"] == "complete"
    assert report["evaluated_injections"] == 4
    assert report["selected_injections"] == 4
    assert report["frozen_catalogue_size"] == 8
    assert report["full_catalogue_evaluated"] is False
    assert report["summary"] == {
        "median": 25.0,
        "mean": 27.5,
        "minimum": 10.0,
        "maximum": 50.0,
    }
    for relative in report["outputs"]:
        assert (campaign / relative).is_file()
    assert (campaign / "snr/figure-7-equivalent.png").stat().st_size > 0


def test_completed_result_must_match_campaign_hash(tmp_path: Path) -> None:
    directory = tmp_path / "result"
    directory.mkdir()
    (directory / "posterior.npz").write_bytes(b"posterior")
    posterior_sha256 = common.file_sha256(directory / "posterior.npz")
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "config_sha256": "right",
                "posterior": {"sha256": posterior_sha256},
            }
        )
    )

    assert _is_complete(directory, "right") is True
    assert _is_complete(directory, "wrong") is False
    (directory / "posterior.npz").write_bytes(b"corrupt")
    assert _is_complete(directory, "right") is False


def test_run_campaign_plot_dispatches_pp_and_figure_3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=1)
    manifest = common.load_manifest(campaign)
    row = common.read_catalogue(campaign / manifest["catalogue"]["path"])[0]
    directory = common.result_dir(campaign, 0)
    directory.mkdir(parents=True)
    common.atomic_savez_compressed(
        directory / "posterior.npz", {"M_c": np.asarray([2.0])}
    )
    common.atomic_write_json(
        directory / "summary.json",
        {
            "config_sha256": manifest["config_sha256"],
            "injection_id": 0,
            "truth": {
                name: row[name]
                for name in (*common.PARAMETERS, *common.MARGINALIZED_PARAMETERS)
            },
            "seeds": {
                "noise": row["noise_seed"],
                "sampler": row["sampler_seed"],
            },
            "posterior": {"sha256": common.file_sha256(directory / "posterior.npz")},
        },
    )
    dispatched: list[str] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        dispatched.append(command[2])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_campaign_module.subprocess, "run", fake_run)
    args = run_campaign_module._parse_args([str(campaign), "--plot"])

    assert run_campaign_module.run_campaign(args) == 0
    assert dispatched == [
        "benchmarks.injection_campaign.plot_pp",
        "benchmarks.injection_campaign.plot_timing",
    ]


def test_runpod_provisioner_has_a_deleting_cost_guard() -> None:
    repository = Path(__file__).resolve().parents[3]
    script = (
        repository / "benchmarks/injection_campaign/runpod/provision.sh"
    ).read_text()

    assert "--terminate-after" in script
    assert "--stop-after" not in script
    assert "launch=false" in script
    assert "--gpu-count 4" in script


def test_pod_runner_excludes_only_regenerable_cache_from_results() -> None:
    repository = Path(__file__).resolve().parents[3]
    script = (
        repository / "benchmarks/injection_campaign/runpod/run_on_pod.sh"
    ).read_text()

    assert "benchmarks.injection_campaign.run_campaign" in script
    assert '--exclude="$(basename "$output_dir")/.jax-cache"' in script
    assert "--plot" in script
    assert 'range_arguments=(--start "$start")' in script
    assert 'range_arguments+=(--stop "$stop")' in script
    assert 'UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"' in script
    assert "/workspace/.cache/uv" not in script
    assert "export NCCL_NVLS_ENABLE=0" in script
    assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in script
    assert 'manifest.get("config", {}).get("n_devices")' in script
    assert 'if [[ "$gpu_count" -ne "$expected_gpu_count" ]]' in script


def test_curated_runpod_package_includes_campaign_import_closure() -> None:
    repository = Path(__file__).resolve().parents[3]
    script = (
        repository / "benchmarks/device_parallel_nss/runpod/package_workspace.sh"
    ).read_text()

    for relative in (
        "benchmarks/injection_campaign/common.py",
        "benchmarks/injection_campaign/prepare_campaign.py",
        "benchmarks/injection_campaign/run_injection.py",
        "benchmarks/injection_campaign/run_campaign.py",
        "benchmarks/injection_campaign/plot_pp.py",
        "benchmarks/injection_campaign/plot_timing.py",
        "benchmarks/injection_campaign/runpod/run_on_pod.sh",
    ):
        assert f'"{relative}"' in script
