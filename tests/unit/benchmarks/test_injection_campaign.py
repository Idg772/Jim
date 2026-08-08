import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign.plot_pp import aggregate_and_plot
from benchmarks.injection_campaign.plot_timing import aggregate_and_plot_timing
from benchmarks.injection_campaign.prepare_campaign import prepare_campaign
from benchmarks.injection_campaign.run_campaign import _is_complete
from benchmarks.injection_campaign.run_injection import _time_marginalization_config
from jimgw.core.prior import CombinePrior, UniformPrior


def _write_noise_curves(directory: Path) -> None:
    directory.mkdir()
    frequencies = np.asarray([1.0, 10.0, 20.0, 100.0, 1024.0, 4096.0])
    for name, scale in (
        ("aLIGO_ZERO_DET_high_P_psd.txt", 1.0e-46),
        ("AdV_psd.txt", 2.0e-46),
    ):
        values = scale * (1.0 + (100.0 / frequencies) ** 2)
        np.savetxt(directory / name, np.column_stack([frequencies, values]))


def _prepared_campaign(tmp_path: Path, n_injections: int = 4) -> Path:
    curves = tmp_path / "curves"
    _write_noise_curves(curves)
    campaign = tmp_path / "campaign"
    prepare_campaign(
        campaign,
        n_injections=n_injections,
        seed=1234,
        noise_curves_dir=curves,
    )
    return campaign


def test_campaign_config_samples_coalescence_time() -> None:
    assert common.DEFAULT_CONFIG["sample_coalescence_time"] is True
    assert common.DEFAULT_CONFIG["coalescence_time_range_seconds"] == [-0.03, 0.03]
    assert "time_marginalization_tc_range_seconds" not in common.DEFAULT_CONFIG
    assert "time_marginalization_upsample_factor" not in common.DEFAULT_CONFIG
    assert "time_marginalization_jitter_time" not in common.DEFAULT_CONFIG


def test_campaign_config_puts_tc_in_the_mass_swig_block() -> None:
    blocks = common.DEFAULT_CONFIG["blocks"]
    flattened = [name for block in blocks for name in block]

    assert blocks[0] == ["M_c", "q", "lambda_1", "lambda_2", "t_c"]
    assert all("time_jitter" not in block for block in blocks)
    assert len(flattened) == 16
    assert len(set(flattened)) == 16


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({}, {"upsample_factor": 1, "jitter_time": False}),
        (
            {"time_marginalization_upsample_factor": 32},
            {"upsample_factor": 32, "jitter_time": False},
        ),
        (
            {
                "time_marginalization_upsample_factor": 1,
                "time_marginalization_jitter_time": True,
            },
            {"upsample_factor": 1, "jitter_time": True},
        ),
    ],
)
def test_time_marginalization_config_preserves_stored_campaign_behavior(
    stored: dict[str, object], expected: dict[str, object]
) -> None:
    config = {
        "time_marginalization_tc_range_seconds": [-0.03, 0.03],
        **stored,
    }

    assert _time_marginalization_config(config) == {
        "tc_range": (-0.03, 0.03),
        **expected,
    }


def test_time_marginalization_config_rejects_non_boolean_jitter_flag() -> None:
    with pytest.raises(TypeError, match="jitter.*boolean"):
        _time_marginalization_config(
            {
                "time_marginalization_tc_range_seconds": [-0.03, 0.03],
                "time_marginalization_jitter_time": "false",
            }
        )


def test_time_jitter_extends_recovery_prior_and_periodic_bounds() -> None:
    from benchmarks.injection_campaign.run_injection import (
        _recovery_sampling_components,
    )

    physical_names = tuple(f"physical_{index:02d}" for index in range(15))
    original_prior = CombinePrior(
        [UniformPrior(0.0, 1.0, parameter_names=[name]) for name in physical_names]
    )
    original_periodic = {physical_names[0]: (0.0, 1.0)}
    likelihood = SimpleNamespace(
        jitter_time=True,
        time_jitter_bounds=(-0.00025, 0.00025),
    )

    prior, periodic = _recovery_sampling_components(
        original_prior, original_periodic, likelihood, {}
    )

    assert prior.parameter_names == (*physical_names, "time_jitter")
    assert len(prior.parameter_names) == 16
    assert prior.base_prior[-1].parameter_names == ("time_jitter",)
    assert prior.base_prior[-1].xmin == likelihood.time_jitter_bounds[0]
    assert prior.base_prior[-1].xmax == likelihood.time_jitter_bounds[1]
    assert periodic == {
        physical_names[0]: (0.0, 1.0),
        "time_jitter": likelihood.time_jitter_bounds,
    }
    assert original_prior.parameter_names == physical_names
    assert original_periodic == {physical_names[0]: (0.0, 1.0)}


def test_recovery_components_leave_non_jitter_campaigns_unchanged() -> None:
    from benchmarks.injection_campaign.run_injection import (
        _recovery_sampling_components,
    )

    original_prior = CombinePrior(
        [UniformPrior(0.0, 1.0, parameter_names=["physical"])]
    )
    original_periodic = {"physical": (0.0, 1.0)}

    prior, periodic = _recovery_sampling_components(
        original_prior,
        original_periodic,
        SimpleNamespace(jitter_time=False),
        {},
    )

    assert prior is original_prior
    assert periodic == original_periodic
    assert "time_jitter" not in prior.parameter_names
    assert "time_jitter" not in periodic


def test_likelihood_time_settings_selects_sampled_coalescence_time() -> None:
    from benchmarks.injection_campaign.run_injection import _likelihood_time_settings

    assert _likelihood_time_settings({"sample_coalescence_time": True}) is None

    legacy = {
        "time_marginalization_tc_range_seconds": [-0.03, 0.03],
        "time_marginalization_jitter_time": True,
    }
    assert _likelihood_time_settings(legacy) == {
        "tc_range": (-0.03, 0.03),
        "upsample_factor": 1,
        "jitter_time": True,
    }


def test_sampled_tc_extends_recovery_prior_without_periodic_bounds() -> None:
    from benchmarks.injection_campaign.run_injection import (
        _recovery_sampling_components,
    )

    physical_names = tuple(f"physical_{index:02d}" for index in range(15))
    original_prior = CombinePrior(
        [UniformPrior(0.0, 1.0, parameter_names=[name]) for name in physical_names]
    )
    original_periodic = {physical_names[0]: (0.0, 1.0)}
    config = {
        "sample_coalescence_time": True,
        "coalescence_time_range_seconds": [-0.03, 0.03],
    }

    prior, periodic = _recovery_sampling_components(
        original_prior,
        original_periodic,
        SimpleNamespace(jitter_time=False),
        config,
    )

    assert prior.parameter_names == (*physical_names, "t_c")
    assert len(prior.parameter_names) == 16
    assert prior.base_prior[-1].parameter_names == ("t_c",)
    assert prior.base_prior[-1].xmin == -0.03
    assert prior.base_prior[-1].xmax == 0.03
    assert periodic == original_periodic
    assert "t_c" not in periodic
    assert original_prior.parameter_names == physical_names


def test_ranked_parameters_include_tc_only_when_sampled() -> None:
    from benchmarks.injection_campaign.run_injection import _ranked_parameters

    assert _ranked_parameters({}) == common.PARAMETERS
    assert _ranked_parameters({"sample_coalescence_time": True}) == (
        *common.PARAMETERS,
        "t_c",
    )


def test_prior_ppf_and_cdf_are_inverse() -> None:
    grid = np.linspace(1e-6, 1.0 - 1e-6, 101)
    for name in (*common.PARAMETERS, *common.NUISANCE_PARAMETERS):
        values = np.asarray([common.prior_ppf(name, u) for u in grid])
        back = np.asarray([common.prior_cdf(name, v) for v in values])
        np.testing.assert_allclose(back, grid, atol=1e-12)


def test_stratified_catalogue_covers_every_stratum_once() -> None:
    n = 16
    rows = common.generate_catalogue(n, 7)
    for name in (*common.PARAMETERS, *common.NUISANCE_PARAMETERS):
        u = np.sort([common.prior_cdf(name, row[name]) for row in rows])
        assert np.array_equal(np.floor(u * n).astype(int), np.arange(n))


def test_iid_catalogue_mode_differs_from_stratified() -> None:
    stratified = common.generate_catalogue(8, 42)
    iid = common.generate_catalogue(8, 42, stratified=False)
    assert stratified != iid
    assert [row["noise_seed"] for row in stratified] == [
        row["noise_seed"] for row in iid
    ]


def test_catalogue_is_deterministic_and_inside_the_recovery_prior() -> None:
    first = common.generate_catalogue(8, 42)
    second = common.generate_catalogue(8, 42)

    assert first == second
    assert len({row["noise_seed"] for row in first}) == 8
    assert len({row["sampler_seed"] for row in first}) == 8
    for index, row in enumerate(first):
        assert row["injection_id"] == index
        assert 1.18 <= row["M_c"] <= 1.21
        assert 0.125 <= row["q"] <= 1.0
        assert 0.0 <= row["s1_mag"] <= 0.05
        assert 0.0 <= row["s2_mag"] <= 0.05
        assert 0.0 <= row["s1_theta"] <= np.pi
        assert 0.0 <= row["s2_theta"] <= np.pi
        assert 0.0 <= row["iota"] <= np.pi
        assert 1.0 <= row["d_L"] <= 75.0
        assert -np.pi / 2 <= row["dec"] <= np.pi / 2
        assert -0.03 <= row["t_c"] <= 0.03


def test_prepare_campaign_round_trip_and_input_integrity(tmp_path: Path) -> None:
    campaign = _prepared_campaign(tmp_path)

    manifest = common.load_manifest(campaign)
    catalogue = common.read_catalogue(campaign / "catalogue.csv")

    assert manifest["n_injections"] == 4
    assert manifest["config"]["n_devices"] == 4
    assert manifest["config"]["sample_coalescence_time"] is True
    assert manifest["config"]["coalescence_time_range_seconds"] == [-0.03, 0.03]
    assert manifest["config"]["blocks"][0] == [
        "M_c",
        "q",
        "lambda_1",
        "lambda_2",
        "t_c",
    ]
    assert len(catalogue) == 4
    assert set(catalogue[0]) == set(common.CATALOGUE_FIELDS)
    assert "time_jitter" not in catalogue[0]
    assert "t_c" in catalogue[0]
    assert (campaign / "status.csv").is_file()
    with np.load(campaign / "inputs/psd/aLIGO-design.npz") as archive:
        assert archive["frequencies"].shape == (262145,)
        assert np.all(np.isfinite(archive["values"]))
        assert np.all(archive["values"] > 0)

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
            seed=1234,
            noise_curves_dir=curves,
        )


def test_posterior_rank_is_tie_safe() -> None:
    assert common.posterior_rank(np.asarray([0.0, 1.0, 1.0, 2.0]), 1.0) == 0.5
    with pytest.raises(ValueError, match="non-empty"):
        common.posterior_rank(np.asarray([]), 1.0)


def test_status_and_pp_outputs_are_derived_from_compact_summaries(
    tmp_path: Path,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=5)
    manifest = common.load_manifest(campaign)
    completed_ids = (1, 3, 4)
    for injection_id in completed_ids:
        directory = common.result_dir(campaign, injection_id)
        directory.mkdir(parents=True)
        ranks = {
            name: ((injection_id + parameter_index / len(common.PARAMETERS)) % 5) / 5
            for parameter_index, name in enumerate(common.PARAMETERS)
        }
        ranks["time_jitter"] = 0.5
        common.atomic_savez_compressed(
            directory / "posterior.npz", {"M_c": np.asarray([1.19, 1.20])}
        )
        posterior_sha256 = common.file_sha256(directory / "posterior.npz")
        common.atomic_write_json(
            directory / "summary.json",
            {
                "config_sha256": manifest["config_sha256"],
                "injection_id": injection_id,
                "posterior_samples": 2,
                "posterior": {"sha256": posterior_sha256},
                "timing_seconds": {
                    "sample_call": injection_id + 1.0,
                    "total": injection_id + 2.0,
                },
                "ranks": ranks,
            },
        )

    rows = common.refresh_status(campaign, 5)
    report = aggregate_and_plot(campaign)
    timing_report = aggregate_and_plot_timing(
        campaign, pdf_output=tmp_path / "figure-3.pdf"
    )

    assert {row["injection_id"] for row in rows if row["status"] == "complete"} == set(
        completed_ids
    )
    assert all(
        row["attempts"] == (1 if row["injection_id"] in completed_ids else 0)
        for row in rows
    )
    assert report["completed_injections"] == len(completed_ids)
    assert timing_report["completed_injections"] == len(completed_ids)
    for relative in report["outputs"]:
        assert (campaign / relative).is_file()
    assert (campaign / "pp/pp-combined.png").stat().st_size > 0
    assert (campaign / "pp/pp-grid.png").stat().st_size > 0
    assert (campaign / "timing/figure-3-equivalent.png").stat().st_size > 0
    assert (campaign / "timing/figure-3-summary.csv").is_file()
    assert (tmp_path / "figure-3.pdf").stat().st_size > 0

    with (campaign / "pp/summary.csv").open(newline="", encoding="utf-8") as stream:
        summary_rows = {row["parameter"]: row for row in csv.DictReader(stream)}
    assert set(summary_rows) == set(common.PARAMETERS)
    assert "time_jitter" not in summary_rows
    catalogue = common.read_catalogue(campaign / "catalogue.csv")
    # M_c is PARAMETERS[0], so its fake rank for injection i is (i % 5) / 5.
    expected_residual = float(
        np.mean(
            [
                (injection_id % 5) / 5
                - common.prior_cdf("M_c", catalogue[injection_id]["M_c"])
                for injection_id in completed_ids
            ]
        )
    )
    mc_row = summary_rows["M_c"]
    assert {
        "truth_ks_statistic",
        "truth_ks_pvalue",
        "residual_mean",
        "residual_stderr",
    } <= set(mc_row)
    np.testing.assert_allclose(
        float(mc_row["residual_mean"]), expected_residual, atol=1e-12
    )


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
    assert "export NCCL_NVLS_ENABLE=0" in script
    assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in script


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
