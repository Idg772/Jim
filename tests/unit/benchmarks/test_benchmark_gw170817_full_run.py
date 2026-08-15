import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.device_parallel_nss import benchmark_gw170817_full_run as benchmark
from benchmarks.device_parallel_nss import (
    benchmark_gw170817_likelihood_lanes as lane_benchmark,
)


def _bundle_arrays(
    *,
    event: str = "GW170817",
    workload: str = benchmark.ALIGNED_WORKLOAD,
) -> dict[str, object]:
    manifest = benchmark._data_manifest(workload)
    manifest["event"] = event
    analysis_start, _, _, _ = benchmark._data_window(workload)
    # Keep Nyquist above the benchmark's 20 Hz lower cutoff so the
    # event-specific whitening guard exercises a non-empty frequency band.
    n_time = 8192
    delta_t = benchmark.DURATION / n_time
    frequencies = np.fft.rfftfreq(n_time, delta_t)
    arrays: dict[str, object] = {
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True))
    }
    for name in benchmark.IFO_NAMES:
        arrays.update(
            {
                f"{name}_strain_td": np.zeros(n_time),
                f"{name}_strain_delta_t": np.asarray(delta_t),
                f"{name}_strain_start_time": np.asarray(analysis_start),
                f"{name}_psd_values": np.ones(frequencies.size),
                f"{name}_psd_frequencies": frequencies,
            }
        )
    return arrays


def test_frozen_bundle_round_trip_and_reuse(tmp_path: Path) -> None:
    data_file = tmp_path / "gw170817.npz"
    benchmark._atomic_save_npz(data_file, _bundle_arrays())

    manifest, arrays = benchmark._read_bundle(data_file)
    prepared = benchmark._prepare_data(data_file)

    assert manifest["event"] == "GW170817"
    assert arrays["H1_strain_td"].shape == (8192,)
    assert prepared["reused_existing"] is True
    assert prepared["sha256"] == benchmark._sha256(data_file)
    assert prepared["sample_counts"] == {"H1": 8192, "L1": 8192, "V1": 8192}


def test_data_manifest_pins_cleaned_analysis_and_psd_sources() -> None:
    manifest = benchmark._data_manifest(benchmark.PAPER_WORKLOAD)

    assert manifest["analysis_strain_dataset"] == "GW170817-v2"
    assert manifest["analysis_strain_release"] == "O1_O2-Preliminary"
    assert manifest["analysis_strain_product"] == "LOSC_CLN_16_V1"
    assert manifest["analysis_strain_glitch_mitigation"] == (
        "L1 glitch removed by the GWOSC release"
    )
    assert manifest["psd_strain_dataset"] == "GW170817-v2"
    assert manifest["psd_strain_product"] == "LOSC_CLN_16_V1"
    assert manifest["psd_strain_source_urls"] == {
        ifo: benchmark._cleaned_strain_url(ifo) for ifo in benchmark.IFO_NAMES
    }
    assert manifest["psd_duration_seconds"] == pytest.approx(1778.43)
    assert manifest["psd_average"] == "median"
    assert manifest["psd_median_bias_correction"] == ("scipy.signal.welch built-in")
    assert manifest["psd_postprocessing"] == ("running-log-median-with-line-protection")
    assert manifest["psd_smoothing_width_hz"] == 1.0
    assert manifest["psd_line_protection_ratio"] == 2.0
    assert manifest["gwosc_sample_rate_hz"] == 16384
    assert manifest["time_marginalization_fft_sample_rate_hz"] == 4096
    assert manifest["gwosc_format"] == "hdf5"
    assert manifest["analysis_strain_source_urls"] == {
        ifo: benchmark._cleaned_strain_url(ifo) for ifo in benchmark.IFO_NAMES
    }


def test_known_l1_glitch_guard_rejects_a_loud_transient() -> None:
    arrays = _bundle_arrays(workload=benchmark.PAPER_WORKLOAD)
    start = float(arrays["L1_strain_start_time"])
    delta_t = float(arrays["L1_strain_delta_t"])
    glitch_index = round((benchmark.L1_GLITCH_GPS - start) / delta_t)
    arrays["L1_strain_td"][glitch_index] = 1_000.0

    with pytest.raises(SystemExit, match="known GW170817 L1 glitch window"):
        benchmark._validate_known_l1_glitch_window(arrays)


def test_known_l1_glitch_guard_accepts_clean_data() -> None:
    benchmark._validate_known_l1_glitch_window(
        _bundle_arrays(workload=benchmark.PAPER_WORKLOAD)
    )


def test_known_l1_glitch_guard_is_native_sample_rate_invariant() -> None:
    peaks = []
    for sample_rate_hz in (4096, 16384):
        n_time = sample_rate_hz
        delta_t = 1.0 / sample_rate_hz
        sample_times = np.arange(n_time) * delta_t
        frequencies = np.fft.rfftfreq(n_time, delta_t)
        peaks.append(
            benchmark._validate_known_l1_glitch_window(
                {
                    "L1_strain_td": np.sin(2.0 * np.pi * 100.0 * sample_times),
                    "L1_strain_delta_t": delta_t,
                    "L1_strain_start_time": benchmark.L1_GLITCH_GPS - 0.5,
                    "L1_psd_values": np.ones(frequencies.size),
                    "L1_psd_frequencies": frequencies,
                }
            )
        )

    assert peaks[1] == pytest.approx(peaks[0], rel=2e-8)


def test_frozen_bundle_rejects_known_l1_glitch(tmp_path: Path) -> None:
    arrays = _bundle_arrays(workload=benchmark.PAPER_WORKLOAD)
    start = float(arrays["L1_strain_start_time"])
    delta_t = float(arrays["L1_strain_delta_t"])
    glitch_index = round((benchmark.L1_GLITCH_GPS - start) / delta_t)
    arrays["L1_strain_td"][glitch_index] = 1_000.0
    data_file = tmp_path / "raw-l1.npz"
    benchmark._atomic_save_npz(data_file, arrays)

    with pytest.raises(SystemExit, match="known GW170817 L1 glitch window"):
        benchmark._read_bundle(data_file, benchmark.PAPER_WORKLOAD)


def test_prepare_data_fetches_pinned_strain_products(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jimgw.core.single_event.data import Data

    calls: list[tuple[str, float, float, dict[str, object]]] = []
    n_time = 8192
    delta_t = benchmark.DURATION / n_time
    frequencies = np.fft.rfftfreq(n_time, delta_t)
    raw_psd = np.ones(frequencies.size)
    raw_psd[256] = 0.25
    raw_psd[512] = 3.0

    class FakeData:
        def __init__(self, ifo: str, start: float) -> None:
            self.name = ifo
            self.td = np.zeros(n_time)
            self.delta_t = delta_t
            self.start_time = start
            self.duration = benchmark.DURATION
            self.sampling_frequency = 1.0 / delta_t

        def to_psd(self, *, nperseg: int, average: str) -> SimpleNamespace:
            assert nperseg == n_time
            assert average == "median"
            return SimpleNamespace(
                values=raw_psd,
                frequencies=frequencies,
            )

    def fake_from_gwosc(
        ifo: str,
        start: float,
        end: float,
        **kwargs: object,
    ) -> FakeData:
        calls.append((ifo, start, end, kwargs))
        return FakeData(ifo, start)

    monkeypatch.setattr(Data, "from_gwosc", staticmethod(fake_from_gwosc))

    data_file = tmp_path / "clean-data.npz"
    prepared = benchmark._prepare_data(data_file, benchmark.PAPER_WORKLOAD)

    analysis_start, analysis_end, psd_start, psd_end = benchmark._data_window(
        benchmark.PAPER_WORKLOAD
    )
    expected_common = {
        "sample_rate": benchmark.GWOSC_PAPER_SAMPLE_RATE_HZ,
        "format": benchmark.GWOSC_FORMAT,
    }
    assert calls == [
        item
        for ifo in benchmark.IFO_NAMES
        for item in (
            (
                ifo,
                analysis_start,
                analysis_end,
                {
                    **expected_common,
                    "dataset": benchmark.GWOSC_ANALYSIS_STRAIN_DATASET,
                },
            ),
            (
                ifo,
                psd_start,
                psd_end,
                {
                    **expected_common,
                    "dataset": benchmark.GWOSC_ANALYSIS_STRAIN_DATASET,
                },
            ),
        )
    ]
    assert prepared["reused_existing"] is False
    assert prepared["manifest"]["analysis_strain_dataset"] == "GW170817-v2"
    with np.load(data_file) as archive:
        for ifo in benchmark.IFO_NAMES:
            smoothed = archive[f"{ifo}_psd_values"]
            assert smoothed[256] == pytest.approx(1.0)
            assert smoothed[512] == pytest.approx(3.0)


def test_aligned_workload_keeps_the_historical_4khz_products() -> None:
    manifest = benchmark._data_manifest(benchmark.ALIGNED_WORKLOAD)

    assert manifest["analysis_strain_product"] == "LOSC_CLN_4_V1"
    assert manifest["psd_strain_product"] == "O2_4KHZ_R1"
    assert manifest["gwosc_sample_rate_hz"] == 4096
    assert manifest["psd_postprocessing"] is None
    assert manifest["psd_smoothing_width_hz"] is None
    assert manifest["psd_line_protection_ratio"] is None
    assert manifest["analysis_strain_source_urls"] == {
        ifo: benchmark._cleaned_strain_url(
            ifo,
            product=benchmark.GWOSC_ALIGNED_ANALYSIS_STRAIN_PRODUCT,
        )
        for ifo in benchmark.IFO_NAMES
    }


def test_16khz_and_legacy_grids_have_the_same_likelihood_slice() -> None:
    legacy = np.fft.rfftfreq(
        int(benchmark.DURATION * benchmark.GWOSC_ALIGNED_SAMPLE_RATE_HZ),
        1.0 / benchmark.GWOSC_ALIGNED_SAMPLE_RATE_HZ,
    )
    native = np.fft.rfftfreq(
        int(benchmark.DURATION * benchmark.GWOSC_PAPER_SAMPLE_RATE_HZ),
        1.0 / benchmark.GWOSC_PAPER_SAMPLE_RATE_HZ,
    )

    legacy_slice = legacy[(legacy >= benchmark.F_MIN) & (legacy <= benchmark.F_MAX)]
    native_slice = native[(native >= benchmark.F_MIN) & (native <= benchmark.F_MAX)]

    np.testing.assert_array_equal(native_slice, legacy_slice)
    assert native_slice.size == 259_584
    assert native_slice[0] == benchmark.F_MIN
    assert native_slice[-1] == benchmark.F_MAX


def test_native_strain_is_projected_after_fft_without_in_band_decimation() -> None:
    import jax
    import jax.numpy as jnp

    from jimgw.core.single_event.data import Data, PowerSpectrum

    jax.config.update("jax_enable_x64", True)
    native_sample_rate_hz = 8192
    n_time = native_sample_rate_hz
    delta_t = 1.0 / native_sample_rate_hz
    times = np.arange(n_time) * delta_t
    td = np.sin(2.0 * np.pi * 100.0 * times)
    frequencies = np.fft.rfftfreq(n_time, delta_t)
    arrays = {
        "H1_strain_td": td,
        "H1_strain_delta_t": np.asarray(delta_t),
        "H1_strain_start_time": np.asarray(0.0),
        "H1_psd_values": np.ones(frequencies.size),
        "H1_psd_frequencies": frequencies,
    }
    native = Data(jnp.asarray(td), delta_t=delta_t, name="H1")
    native_fd = native.fft()

    projected, psd = benchmark._likelihood_inputs_from_bundle(
        "H1",
        arrays,
        Data=Data,
        PowerSpectrum=PowerSpectrum,
        jnp=jnp,
    )

    assert float(projected.sampling_frequency) == 4096.0
    assert projected.n_time == 4096
    assert projected.has_fd is True
    np.testing.assert_array_equal(
        np.asarray(projected.fd), np.asarray(native_fd[: projected.n_freq])
    )
    assert float(psd.frequencies[-1]) == 2048.0


def test_raw_psd_format_v5_bundle_is_rejected(tmp_path: Path) -> None:
    arrays = _bundle_arrays(workload=benchmark.PAPER_WORKLOAD)
    manifest = json.loads(str(arrays["manifest_json"].item()))
    manifest["format_version"] = 5
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    data_file = tmp_path / "v5-data.npz"
    benchmark._atomic_save_npz(data_file, arrays)

    with pytest.raises(
        SystemExit, match="incompatible manifest fields: format_version"
    ):
        benchmark._read_bundle(data_file, benchmark.PAPER_WORKLOAD)


def test_frozen_bundle_rejects_wrong_workload(tmp_path: Path) -> None:
    data_file = tmp_path / "wrong-event.npz"
    benchmark._atomic_save_npz(data_file, _bundle_arrays(event="GW150914"))

    with pytest.raises(SystemExit, match="incompatible manifest fields: event"):
        benchmark._read_bundle(data_file)


def test_data_windows_are_selected_by_workload() -> None:
    aligned = benchmark._data_manifest(benchmark.ALIGNED_WORKLOAD)
    paper = benchmark._data_manifest(benchmark.PAPER_WORKLOAD)

    assert aligned["analysis_start_gps"] == pytest.approx(benchmark.GPS - 126.0)
    assert aligned["analysis_end_gps"] == pytest.approx(benchmark.GPS + 2.0)
    assert paper["analysis_start_gps"] == pytest.approx(benchmark.GPS - 64.0)
    assert paper["analysis_end_gps"] == pytest.approx(benchmark.GPS + 64.0)
    assert aligned["psd_end_gps"] == aligned["analysis_start_gps"]
    assert paper["psd_end_gps"] == paper["analysis_start_gps"]
    assert aligned["psd_duration_seconds"] == pytest.approx(2048.0)
    assert aligned["psd_strain_dataset"] == "O2"
    assert aligned["psd_average"] == "mean"
    assert paper["psd_start_gps"] == benchmark.GWOSC_EVENT_FILE_START_GPS
    assert paper["psd_duration_seconds"] == pytest.approx(1778.43)


def test_frozen_bundle_cannot_be_reused_across_workloads(tmp_path: Path) -> None:
    data_file = tmp_path / "paper-data.npz"
    benchmark._atomic_save_npz(
        data_file,
        _bundle_arrays(workload=benchmark.PAPER_WORKLOAD),
    )

    benchmark._read_bundle(data_file, benchmark.PAPER_WORKLOAD)
    with pytest.raises(SystemExit, match="incompatible manifest fields: workload"):
        benchmark._read_bundle(data_file, benchmark.ALIGNED_WORKLOAD)


def test_config_fingerprint_covers_seed() -> None:
    first = benchmark._config_report(seed=1, n_devices=4)
    repeated = benchmark._config_report(seed=1, n_devices=4)
    second = benchmark._config_report(seed=2, n_devices=4)

    assert first["sha256"] == repeated["sha256"]
    assert first["sha256"] != second["sha256"]
    assert first["n_live"] == 512
    assert first["n_delete"] == 64
    assert first["num_gibbs_sweeps"] == 1
    assert first["termination_dlogz"] == pytest.approx(0.0485873516)


def test_paper_workload_matches_the_full_15d_gw170817_specification() -> None:
    config = benchmark._config_report(
        seed=0,
        n_devices=4,
        workload=benchmark.PAPER_WORKLOAD,
    )

    assert config["workload"] == "paper-15d"
    assert config["waveform"] == "IMRPhenomPv2_NRTidalv2"
    assert config["sampled_dimensions"] == 15
    assert config["priors"]["q"]["range"] == [0.125, 1.0]
    assert config["priors"]["d_L"]["range_mpc"] == [1.0, 75.0]
    assert config["blocks"] == [list(block) for block in benchmark.PAPER_BLOCKS]
    assert sum(map(len, config["blocks"])) == 15
    assert config["time_marginalization_fft_sample_rate_hz"] == 4096


def test_workloads_have_distinct_config_fingerprints() -> None:
    aligned = benchmark._config_report(0, 4, benchmark.ALIGNED_WORKLOAD)
    paper = benchmark._config_report(0, 4, benchmark.PAPER_WORKLOAD)

    assert aligned["sha256"] != paper["sha256"]


def test_cli_accepts_paper_workload(tmp_path: Path) -> None:
    args = benchmark._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--workload",
            benchmark.PAPER_WORKLOAD,
        ]
    )

    assert args.workload == benchmark.PAPER_WORKLOAD


def test_cli_accepts_posterior_sample_output(tmp_path: Path) -> None:
    samples_file = tmp_path / "posterior-samples.npz"

    args = benchmark._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--samples-output",
            str(samples_file),
        ]
    )

    assert args.samples_output == samples_file


def test_write_posterior_samples_round_trip(tmp_path: Path) -> None:
    samples_file = tmp_path / "posterior-samples.npz"
    samples = {
        "M_c": np.asarray([1.19, 1.20]),
        "q": np.asarray([0.8, 0.9]),
        "log_likelihood": np.asarray([100.0, 101.0]),
    }

    artifact = benchmark._write_posterior_samples(samples_file, samples)

    assert artifact["path"] == str(samples_file)
    assert artifact["count"] == 2
    assert artifact["fields"] == ["M_c", "q", "log_likelihood"]
    assert artifact["space"] == "prior"
    assert artifact["weighting"] == "equal"
    assert artifact["sha256"] == benchmark._sha256(samples_file)
    with np.load(samples_file, allow_pickle=False) as saved:
        assert saved.files == artifact["fields"]
        for name, values in samples.items():
            np.testing.assert_array_equal(saved[name], values)


def test_write_nested_samples_round_trip(tmp_path: Path) -> None:
    nested_file = tmp_path / "nested-samples.npz"
    samples = {
        "M_c": np.asarray([1.19, 1.20]),
        "q": np.asarray([0.8, 0.9]),
        "log_likelihood": np.asarray([100.0, 101.0]),
        "log_likelihood_birth": np.asarray([-np.inf, 99.0]),
        "log_weights": np.log(np.asarray([0.25, 0.75])),
    }

    artifact = benchmark._write_posterior_samples(
        nested_file,
        samples,
        weighting="normalized nested-sampling log weights",
    )

    assert artifact["path"] == str(nested_file)
    assert artifact["count"] == 2
    assert artifact["fields"] == list(samples)
    assert artifact["weighting"] == "normalized nested-sampling log weights"
    assert artifact["sha256"] == benchmark._sha256(nested_file)
    with np.load(nested_file, allow_pickle=False) as saved:
        assert saved.files == artifact["fields"]
        for name, values in samples.items():
            np.testing.assert_array_equal(saved[name], values)
        assert np.exp(saved["log_weights"]).sum() == pytest.approx(1.0)


def test_write_posterior_samples_rejects_inconsistent_lengths(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="inconsistent lengths"):
        benchmark._write_posterior_samples(
            tmp_path / "posterior-samples.npz",
            {
                "M_c": np.asarray([1.19, 1.20]),
                "log_likelihood": np.asarray([100.0]),
            },
        )


def test_likelihood_lane_cli_accepts_the_same_paper_workload(tmp_path: Path) -> None:
    args = lane_benchmark._parse_args(
        [
            "--data-file",
            str(tmp_path / "data.npz"),
            "--output",
            str(tmp_path / "lanes.json"),
            "--workload",
            benchmark.PAPER_WORKLOAD,
        ]
    )

    assert args.workload == benchmark.PAPER_WORKLOAD


def test_paper_blocks_rebuild_only_for_waveform_shape_parameters() -> None:
    import jax.numpy as jnp

    from jimgw.core.single_event.blocked_likelihood import (
        _build_rebuild_required_by_block,
    )
    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1

    components = benchmark._analysis_components(
        benchmark.PAPER_WORKLOAD,
        jnp,
        [get_H1(), get_L1(), get_V1()],
    )
    parameter_names = (
        "M_c",
        "q",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "iota",
        "lambda_1",
        "lambda_2",
        "d_L",
        "zenith",
        "azimuth",
        "psi",
    )
    likelihood = SimpleNamespace(
        waveform=components["waveform"],
        fixed_parameters={"phase_c": 0.0},
        waveform_caches_distance=True,
    )

    rebuild = _build_rebuild_required_by_block(
        likelihood,
        benchmark.PAPER_BLOCKS,
        parameter_names=parameter_names,
        sample_transforms=components["sample_transforms"],
        likelihood_transforms=components["likelihood_transforms"],
    )

    assert list(rebuild.values()) == [True, True, True, True, False, False, False]


def test_cli_accepts_candidate_one_device_sweep(tmp_path: Path) -> None:
    args = benchmark._parse_args(
        ["--data-file", str(tmp_path / "data.npz"), "--n-devices", "1"]
    )

    assert args.n_devices == 1


def test_cli_rejects_telemetry_without_profile(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        benchmark._parse_args(
            [
                "--data-file",
                str(tmp_path / "data.npz"),
                "--telemetry-output",
                str(tmp_path / "telemetry.dmon"),
            ]
        )


def test_normalise_per_slice_array_removes_inner_step_axis() -> None:
    values = np.arange(3 * 11).reshape(3, 1, 11)

    normalised = benchmark._normalise_per_slice_array(values, 11)

    assert normalised.shape == (3, 11)
    np.testing.assert_array_equal(normalised, values[:, 0, :])


def test_install_per_slice_diagnostics_forces_production_builder_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jimgw.samplers.blackjax import swig

    calls: list[dict[str, object]] = []
    constrained_step = object()

    def production_builder(**kwargs: object) -> object:
        calls.append(kwargs)
        return constrained_step

    monkeypatch.setattr(swig, "_build_swig_constrained_step", production_builder)

    assert benchmark._install_per_slice_swig_diagnostics() is True
    result = swig._build_swig_constrained_step(
        marker="production",
        per_slice_info=False,
    )

    assert result is constrained_step
    assert calls == [{"marker": "production", "per_slice_info": True}]


def test_install_per_slice_diagnostics_rejects_legacy_builder_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jimgw.samplers.blackjax import swig

    def legacy_builder(*, marker: str) -> str:
        return marker

    monkeypatch.setattr(swig, "_build_swig_constrained_step", legacy_builder)

    assert benchmark._install_per_slice_swig_diagnostics() is False
    assert swig._build_swig_constrained_step is legacy_builder


def test_outer_step_report_separates_first_step() -> None:
    observer = benchmark._OuterStepObserver(
        jax=None,
        jnp=None,
        profile_dir=None,
        profile_warmup_steps=10,
        profile_steps=15,
        telemetry_output=None,
        max_outer_steps=None,
    )
    observer.host_perf_counter_start[:] = [10.0, 20.0, 30.0]
    observer.host_perf_counter_end[:] = [15.0, 22.0, 33.0]

    report = observer.report()

    assert report["duration_seconds"] == [5.0, 2.0, 3.0]
    assert report["first_step_seconds_including_jit"] == 5.0
    assert report["steady_state_seconds"]["median"] == 2.5


def test_derive_paper_convention_subtracts_both_jit_costs() -> None:
    result = benchmark._derive_paper_convention(100.0, 30.0, {"likelihood_jit": 20.0})

    assert result["post_jit_sampling_seconds"] == pytest.approx(50.0)
    assert result["likelihood_jit_seconds"] == pytest.approx(20.0)
    assert result["sampler_jit_seconds"] == pytest.approx(30.0)
    assert "2607.28265" in result["note"]


def test_derive_paper_convention_without_phases_matches_legacy_post_jit() -> None:
    result = benchmark._derive_paper_convention(100.0, 30.0, None)

    assert result["post_jit_sampling_seconds"] == pytest.approx(70.0)
    assert result["likelihood_jit_seconds"] is None


def test_derive_paper_convention_tolerates_missing_estimates() -> None:
    result = benchmark._derive_paper_convention(100.0, None, {"likelihood_jit": None})

    assert result["post_jit_sampling_seconds"] == pytest.approx(100.0)
    assert result["likelihood_jit_seconds"] is None
    assert result["sampler_jit_seconds"] is None
