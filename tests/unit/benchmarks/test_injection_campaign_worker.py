import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign import run_campaign as run_campaign_module
from benchmarks.injection_campaign import run_injection as run_injection_module
from benchmarks.injection_campaign.prepare_campaign import prepare_campaign
from benchmarks.injection_campaign.runpod import upload_and_run


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


def _prepared_campaign(tmp_path: Path, n_injections: int = 3) -> Path:
    curves = tmp_path / "curves"
    _write_noise_curves(curves)
    campaign = tmp_path / "campaign"
    prepare_campaign(
        campaign,
        n_injections=n_injections,
        catalogue_size=max(n_injections, 8),
        seed=1234,
        noise_curves_dir=curves,
    )
    return campaign


def _write_complete_result(campaign: Path, injection_id: int) -> None:
    manifest = common.load_manifest(campaign)
    row = common.read_catalogue(campaign / manifest["catalogue"]["path"])[injection_id]
    directory = common.result_dir(campaign, injection_id)
    directory.mkdir(parents=True, exist_ok=True)
    posterior = directory / "posterior.npz"
    common.atomic_savez_compressed(posterior, {"sample": np.asarray([injection_id])})
    common.atomic_write_json(
        directory / "summary.json",
        {
            "config_sha256": manifest["config_sha256"],
            "injection_id": injection_id,
            "truth": {
                name: row[name]
                for name in (*common.PARAMETERS, *common.MARGINALIZED_PARAMETERS)
            },
            "seeds": {
                "noise": row["noise_seed"],
                "sampler": row["sampler_seed"],
            },
            "posterior": {"sha256": common.file_sha256(posterior)},
        },
    )


def test_sparse_selection_is_unique_in_range_and_ordered() -> None:
    args = run_campaign_module._parse_args(
        ["campaign", "--injection-id", "7", "--injection-id", "0"]
    )

    assert run_campaign_module._selected_injection_ids(args, 8) == [7, 0]

    duplicate = run_campaign_module._parse_args(
        ["campaign", "--injection-id", "7", "--injection-id", "7"]
    )
    with pytest.raises(SystemExit, match="must be unique"):
        run_campaign_module._selected_injection_ids(duplicate, 8)

    mixed = run_campaign_module._parse_args(
        ["campaign", "--start", "1", "--injection-id", "7"]
    )
    with pytest.raises(SystemExit, match="cannot be combined"):
        run_campaign_module._selected_injection_ids(mixed, 8)

    out_of_range = run_campaign_module._parse_args(["campaign", "--injection-id", "8"])
    with pytest.raises(SystemExit, match="outside the frozen campaign"):
        run_campaign_module._selected_injection_ids(out_of_range, 8)


def test_long_lived_worker_failure_is_resumable_and_skips_valid_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _prepared_campaign(tmp_path)
    runtime = SimpleNamespace(initialization_seconds=0.01)
    calls: list[int] = []
    fail_zero = True

    def fake_prepare(*_: Any, **__: Any) -> SimpleNamespace:
        return runtime

    def fake_run(args: argparse.Namespace, *, runtime: object) -> dict[str, Any]:
        nonlocal fail_zero
        assert runtime is not None
        calls.append(args.injection_id)
        if args.injection_id == 0 and fail_zero:
            fail_zero = False
            raise RuntimeError("controlled worker failure")
        _write_complete_result(campaign, args.injection_id)
        return {"injection_id": args.injection_id}

    monkeypatch.setattr(run_injection_module, "prepare_injection_runtime", fake_prepare)
    monkeypatch.setattr(run_injection_module, "run_injection", fake_run)
    first = run_campaign_module._parse_args(
        [
            str(campaign),
            "--long-lived-worker",
            "--retry-count",
            "0",
            "--injection-id",
            "0",
            "--injection-id",
            "2",
        ]
    )

    assert run_campaign_module.run_campaign(first) == 1
    assert calls == [0, 2]
    assert (common.result_dir(campaign, 0) / "failure.json").is_file()
    assert run_campaign_module._is_complete(
        common.result_dir(campaign, 2),
        common.load_manifest(campaign)["config_sha256"],
    )

    second = run_campaign_module._parse_args(
        [
            str(campaign),
            "--long-lived-worker",
            "--retry-count",
            "0",
            "--injection-id",
            "0",
            "--injection-id",
            "2",
        ]
    )
    assert run_campaign_module.run_campaign(second) == 0
    assert calls == [0, 2, 0]
    assert not (common.result_dir(campaign, 0) / "failure.json").exists()


class _FakeRandom:
    @staticmethod
    def key(seed: int) -> tuple[str, int]:
        return ("key", int(seed))

    @staticmethod
    def fold_in(key: tuple[str, int], index: int) -> tuple[str, int, int]:
        return (*key, index)


class _FakeJax:
    __version__ = "test"
    random = _FakeRandom()


class _FakePowerSpectrum:
    @classmethod
    def from_file(cls, _: str) -> object:
        return object()


class _FakeDetector:
    def __init__(self, name: str, instances: list["_FakeDetector"]) -> None:
        self.name = name
        self.optimal_snr = {"H1": 3.0, "L1": 4.0, "V1": 12.0}[name]
        self.match_filtered_snr = complex(self.optimal_snr)
        self.noise_key: object | None = None
        instances.append(self)

    def set_psd(self, _: object) -> None:
        pass

    def inject_signal(self, **kwargs: Any) -> None:
        self.noise_key = kwargs["rng_key"]


class _FakeLikelihood:
    detector_groups: ClassVar[list[tuple[_FakeDetector, ...]]] = []

    def __init__(self, detectors: list[_FakeDetector], **_: Any) -> None:
        self.detectors = detectors
        self.detector_groups.append(tuple(detectors))


class _FakeJim:
    seeds: ClassVar[list[int]] = []

    def __init__(self, _: object, __: object, *, seed: int, **___: Any) -> None:
        self.seed = seed
        self.seeds.append(seed)

    def sample_initial_positions(self, n_live: int) -> np.ndarray:
        return np.zeros((n_live, 1))

    def sample(self, _: np.ndarray) -> None:
        pass

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "sample_phase_seconds": {
                "likelihood_jit": 0.0,
                "sampler_kernel_jit": 0.0,
            },
            "n_iterations": 2,
            "n_likelihood_evaluations": 3,
            "log_Z": -1.0,
            "log_Z_error": 0.1,
        }

    def get_weighted_samples(self) -> dict[str, np.ndarray]:
        samples = {name: np.asarray([0.0, 1.0]) for name in common.PARAMETERS}
        return {
            **samples,
            "log_likelihood": np.asarray([-2.0, -1.0]),
            "log_likelihood_birth": np.asarray([-np.inf, -2.0]),
            "log_weights": np.log(np.asarray([0.5, 0.5])),
        }


def _fake_runtime(
    campaign: Path,
    instances: list[_FakeDetector],
) -> run_injection_module.InjectionRuntime:
    manifest = common.load_manifest(campaign)

    def detector_factory(name: str) -> Any:
        return lambda: _FakeDetector(name, instances)

    return run_injection_module.InjectionRuntime(
        campaign_dir=campaign.resolve(),
        manifest=manifest,
        catalogue=common.read_catalogue(campaign / manifest["catalogue"]["path"]),
        config=manifest["config"],
        cache_dir=campaign / ".jax-cache",
        simulate_cpu=True,
        jax=_FakeJax(),
        jnp=SimpleNamespace(),
        jaxlib=SimpleNamespace(__version__="test"),
        blackjax=SimpleNamespace(__version__="test"),
        jimgw=SimpleNamespace(__version__="test"),
        Jim=_FakeJim,
        PowerSpectrum=_FakePowerSpectrum,
        detector_factories=tuple(detector_factory(name) for name in ("H1", "L1", "V1")),
        TransientLikelihoodFD=_FakeLikelihood,
        BlackJAXSwiGConfig=SimpleNamespace,
        devices={"backend": "cpu", "local_count": 4},
        constant_handling={"assertion": "effective"},
        cache_diagnostics=None,
        initialization_seconds=0.01,
    )


def _fake_injection_args(campaign: Path, injection_id: int) -> argparse.Namespace:
    return argparse.Namespace(
        campaign_dir=campaign,
        injection_id=injection_id,
        jax_compilation_cache_dir=campaign / ".jax-cache",
        simulate_cpu=True,
        verbose=False,
        force=False,
        jax_cache_diagnostics=False,
    )


@pytest.fixture
def lightweight_science_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_injection_module,
        "_analysis_components",
        lambda *_: {
            "waveform": object(),
            "prior": object(),
            "distance_prior": object(),
            "sample_transforms": [],
            "likelihood_transforms": [],
            "periodic": {},
        },
    )
    monkeypatch.setattr(
        run_injection_module, "_implementation_report", lambda *_: {"label": "test"}
    )
    monkeypatch.setattr(
        run_injection_module, "_build_sampler_config", lambda *_: object()
    )
    _FakeLikelihood.detector_groups.clear()
    _FakeJim.seeds.clear()


def _scientific_projection(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
        for key in (
            "config_sha256",
            "injection_id",
            "truth",
            "seeds",
            "network",
            "ranks",
            "rank_truth",
            "parameter_treatment",
            "diagnostics",
        )
    }


def test_one_shot_and_shared_runtime_use_the_same_event_science_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lightweight_science_path: None,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=1)
    instances: list[_FakeDetector] = []
    runtime = _fake_runtime(campaign, instances)
    args = _fake_injection_args(campaign, 0)

    shared_summary = run_injection_module.run_injection(args, runtime=runtime)
    with np.load(common.result_dir(campaign, 0) / "posterior.npz") as posterior:
        shared_posterior = {name: posterior[name].copy() for name in posterior.files}
    (common.result_dir(campaign, 0) / "summary.json").unlink()
    (common.result_dir(campaign, 0) / "posterior.npz").unlink()
    monkeypatch.setattr(
        run_injection_module,
        "prepare_injection_runtime",
        lambda *_args, **_kws: runtime,
    )

    one_shot_summary = run_injection_module.run_injection(args)

    assert _scientific_projection(one_shot_summary) == _scientific_projection(
        shared_summary
    )
    with np.load(common.result_dir(campaign, 0) / "posterior.npz") as posterior:
        assert set(posterior.files) == set(shared_posterior)
        for name, expected in shared_posterior.items():
            np.testing.assert_array_equal(posterior[name], expected)
    assert shared_summary["execution"]["mode"] == "long-lived-worker"
    assert one_shot_summary["execution"]["mode"] == "one-shot"


@pytest.mark.parametrize(
    "mutation",
    [
        "injection_id",
        "truth",
        "noise_seed",
        "sampler_seed",
        "posterior_hash",
    ],
)
def test_existing_result_short_circuit_recomputes_stale_identity_or_payload(
    tmp_path: Path,
    lightweight_science_path: None,
    mutation: str,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=2)
    instances: list[_FakeDetector] = []
    runtime = _fake_runtime(campaign, instances)
    args = _fake_injection_args(campaign, 0)
    expected = run_injection_module.run_injection(args, runtime=runtime)
    assert len(instances) == 3

    summary_path = common.result_dir(campaign, 0) / "summary.json"
    stale = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "injection_id":
        stale["injection_id"] = 1
    elif mutation == "truth":
        stale["truth"]["q"] = float(stale["truth"]["q"]) + 0.01
    elif mutation == "noise_seed":
        stale["seeds"]["noise"] = int(stale["seeds"]["noise"]) + 1
    elif mutation == "sampler_seed":
        stale["seeds"]["sampler"] = int(stale["seeds"]["sampler"]) + 1
    else:
        stale["posterior"]["sha256"] = "0" * 64
    common.atomic_write_json(summary_path, stale)

    repaired = run_injection_module.run_injection(args, runtime=runtime)

    assert len(instances) == 6
    assert _scientific_projection(repaired) == _scientific_projection(expected)
    row = runtime.catalogue[0]
    assert run_campaign_module._is_complete(
        summary_path.parent,
        runtime.manifest["config_sha256"],
        injection_id=0,
        catalogue_row=row,
    )


def test_existing_result_short_circuit_keeps_campaign_hash_fail_closed(
    tmp_path: Path,
    lightweight_science_path: None,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=1)
    instances: list[_FakeDetector] = []
    runtime = _fake_runtime(campaign, instances)
    args = _fake_injection_args(campaign, 0)
    run_injection_module.run_injection(args, runtime=runtime)
    summary_path = common.result_dir(campaign, 0) / "summary.json"
    stale = json.loads(summary_path.read_text(encoding="utf-8"))
    stale["config_sha256"] = "0" * 64
    common.atomic_write_json(summary_path, stale)

    with pytest.raises(SystemExit, match="different campaign hash"):
        run_injection_module.run_injection(args, runtime=runtime)

    assert len(instances) == 3


def test_shared_runtime_recreates_event_objects_and_keeps_seed_streams_isolated(
    tmp_path: Path,
    lightweight_science_path: None,
) -> None:
    campaign = _prepared_campaign(tmp_path, n_injections=2)
    instances: list[_FakeDetector] = []
    runtime = _fake_runtime(campaign, instances)

    first = run_injection_module.run_injection(
        _fake_injection_args(campaign, 0), runtime=runtime
    )
    second = run_injection_module.run_injection(
        _fake_injection_args(campaign, 1), runtime=runtime
    )

    assert len(_FakeLikelihood.detector_groups) == 2
    assert set(map(id, _FakeLikelihood.detector_groups[0])).isdisjoint(
        map(id, _FakeLikelihood.detector_groups[1])
    )
    assert _FakeJim.seeds == [first["seeds"]["sampler"], second["seeds"]["sampler"]]
    first_noise_keys = [
        detector.noise_key for detector in _FakeLikelihood.detector_groups[0]
    ]
    second_noise_keys = [
        detector.noise_key for detector in _FakeLikelihood.detector_groups[1]
    ]
    assert first_noise_keys != second_noise_keys


def test_cache_diagnostics_are_compact_and_measure_inventory_delta(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    diagnostics = run_injection_module._JaxCompilerDiagnostics(cache_dir)
    before = diagnostics.begin_event()
    diagnostics.emit(
        logging.LogRecord(
            "jax._src.compiler",
            logging.WARNING,
            __file__,
            1,
            "Persistent compilation cache hit for '%s' with key %r",
            ("jit_step", "key"),
            None,
        )
    )
    (cache_dir / "entry").write_bytes(b"cache")

    report = diagnostics.finish_event(before)

    assert report["compiler_events"] == {"persistent_cache_hits": 1}
    assert report["modules"] == ["jit_step"]
    assert report["cache_delta"] == {"files": 1, "bytes": 5}


def test_preimport_constant_environment_uses_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", raising=False)
    monkeypatch.delenv("JAX_EMBEDDED_CONSTANTS_MAX_BYTES", raising=False)

    report = run_injection_module._configure_jax_constant_environment()

    assert report["simplified_jaxpr_constants_environment"] == "True"
    assert report["embedded_constants_max_bytes_environment"] == "32"


def test_remote_command_preserves_sparse_selection_order() -> None:
    args = SimpleNamespace(
        seed=123,
        retry_count=0,
        start=0,
        stop=None,
        injection_ids=[7, 0],
        plot=False,
        implementation="candidate",
    )

    command = upload_and_run._build_run_command(
        args,
        "/workspace/campaign",
        n_injections=8,
        catalogue_size=8,
        frozen=True,
    )

    selections = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--injection-id"
    ]
    assert selections == ["7", "0"]
    assert "--start" not in command


def test_pod_worker_sets_constant_handling_before_jax_and_enables_diagnostics() -> None:
    repository = Path(__file__).resolve().parents[3]
    script = (
        repository / "benchmarks/injection_campaign/runpod/run_on_pod.sh"
    ).read_text(encoding="utf-8")

    constant_export = script.index("export JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS")
    first_jax_process = script.index("import jax")
    assert constant_export < first_jax_process
    assert "export JAX_EMBEDDED_CONSTANTS_MAX_BYTES" in script
    assert "--long-lived-worker --jax-cache-diagnostics" in script
    assert 'range_arguments+=(--injection-id "$injection_id")' in script
