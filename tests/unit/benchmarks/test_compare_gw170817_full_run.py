import importlib
import sys
from pathlib import Path

import pytest

from benchmarks.device_parallel_nss import compare_gw170817_full_run as comparison

DATA_SHA256 = "a" * 64


def _runner_report(
    implementation_root: Path,
    *,
    label: str = "candidate",
    revision: str = "deadbeef",
    workload: str = comparison.PAPER_WORKLOAD,
    sampled_dimensions: int = 15,
    sample_seconds: float = 2.0,
) -> dict[str, object]:
    config: dict[str, object] = {
        "seed": 0,
        "workload": workload,
        "sampled_dimensions": sampled_dimensions,
        "n_devices": comparison.DEVICE_COUNT,
    }
    config["sha256"] = comparison._sha256_json(config)
    return {
        "schema_version": 1,
        "benchmark": comparison.RUNNER_BENCHMARK,
        "devices": {
            "backend": "cpu",
            "local_count": comparison.DEVICE_COUNT,
            "requested_count": comparison.DEVICE_COUNT,
            "global_count": comparison.DEVICE_COUNT,
            "devices": [{} for _ in range(comparison.DEVICE_COUNT)],
        },
        "data": {"sha256": DATA_SHA256},
        "environment": {"jax": "test"},
        "implementation": {
            "label": label,
            "revision": revision,
            "module_file": str(implementation_root / "src/jimgw/__init__.py"),
        },
        "config": config,
        "timing_seconds": {"sample_call": sample_seconds, "total": 3.0},
        "results": {"log_Z": 1.0, "n_likelihood_evaluations": 10},
    }


def test_python_executable_path_must_preserve_virtualenv_symlink(
    tmp_path: Path,
) -> None:
    managed_interpreter = tmp_path / "managed-python"
    managed_interpreter.touch()
    virtualenv_interpreter = tmp_path / "venv-python"
    virtualenv_interpreter.symlink_to(managed_interpreter)

    selected = virtualenv_interpreter.expanduser().absolute()

    assert selected == virtualenv_interpreter
    assert selected.is_symlink()
    assert selected.resolve() == managed_interpreter
    assert comparison.DEVICE_COUNT == 4


def test_runner_environment_disables_unsupported_h200_nvls(tmp_path: Path) -> None:
    environment = comparison._runner_environment(
        implementation_root=tmp_path,
        seed=2,
        simulate_cpu=False,
    )

    assert environment["NCCL_NVLS_ENABLE"] == "0"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert environment["JAX_ENABLE_COMPILATION_CACHE"] == "1"


def test_cli_defaults_to_historical_workload(tmp_path: Path) -> None:
    args = comparison._parse_args(["--output-dir", str(tmp_path / "output")])

    assert args.workload == comparison.ALIGNED_WORKLOAD


def test_cli_accepts_paper_workload(tmp_path: Path) -> None:
    args = comparison._parse_args(
        [
            "--output-dir",
            str(tmp_path / "output"),
            "--workload",
            comparison.PAPER_WORKLOAD,
        ]
    )

    assert args.workload == comparison.PAPER_WORKLOAD


def test_scientific_runner_command_forwards_workload(tmp_path: Path) -> None:
    command = comparison._implementation_runner_command(
        python=tmp_path / "python",
        runner_script=tmp_path / "runner.py",
        data_file=tmp_path / "data.npz",
        workload=comparison.PAPER_WORKLOAD,
        seed=2,
        implementation_root=tmp_path / "implementation",
        implementation_label="candidate",
        implementation_revision="deadbeef",
        output=tmp_path / "report.json",
    )

    workload_index = command.index("--workload")
    assert command[workload_index + 1] == comparison.PAPER_WORKLOAD
    assert command.count("--workload") == 1


def test_freeze_harness_copies_and_hashes_paper_model_modules(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    runner = source_dir / "runner.py"
    paper_model = source_dir / "paper_model.py"
    paper_model_basis = source_dir / "paper_model_basis.py"
    runner.write_text("from paper_model import Waveform\n")
    paper_model.write_text("from paper_model_basis import Waveform\n")
    paper_model_basis.write_text("Waveform = object()\n")

    frozen_runner, manifest = comparison._freeze_harness(
        runner,
        tmp_path / "harness",
    )

    frozen_model = frozen_runner.with_name("paper_model.py")
    frozen_basis = frozen_runner.with_name("paper_model_basis.py")
    assert frozen_runner.read_bytes() == runner.read_bytes()
    assert frozen_model.read_bytes() == paper_model.read_bytes()
    assert frozen_basis.read_bytes() == paper_model_basis.read_bytes()
    assert manifest["sha256"] == comparison._sha256_file(frozen_runner)
    assert manifest["companions"] == [
        {
            "module": "paper_model",
            "source": str(paper_model),
            "frozen_copy": str(frozen_model),
            "sha256": comparison._sha256_file(frozen_model),
        },
        {
            "module": "paper_model_basis",
            "source": str(paper_model_basis),
            "frozen_copy": str(frozen_basis),
            "sha256": comparison._sha256_file(frozen_basis),
        },
    ]


def test_frozen_paper_model_imports_basis_from_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = Path(comparison.__file__).parent
    frozen_runner, _ = comparison._freeze_harness(
        source_dir / "benchmark_gw170817_full_run.py",
        tmp_path / "harness",
    )
    monkeypatch.syspath_prepend(str(frozen_runner.parent))
    previous_modules = {
        name: sys.modules.pop(name, None)
        for name in ("paper_model", "paper_model_basis")
    }
    importlib.invalidate_caches()

    try:
        frozen_model = importlib.import_module("paper_model")
        frozen_basis = importlib.import_module("paper_model_basis")
        assert frozen_model.FrequencyPowerBasis is frozen_basis.FrequencyPowerBasis
    finally:
        sys.modules.pop("paper_model", None)
        sys.modules.pop("paper_model_basis", None)
        sys.modules.update(
            {name: module for name, module in previous_modules.items() if module}
        )


def test_validate_report_accepts_matching_paper_workload(tmp_path: Path) -> None:
    report = _runner_report(tmp_path)

    comparison._validate_report(
        report,
        expected_backend="cpu",
        expected_data_sha256=DATA_SHA256,
        expected_implementation_root=tmp_path,
        expected_label="candidate",
        expected_revision="deadbeef",
        expected_seed=0,
        expected_workload=comparison.PAPER_WORKLOAD,
    )


def test_validate_report_accepts_schema_v2(tmp_path: Path) -> None:
    report = _runner_report(tmp_path)
    report["schema_version"] = 2

    comparison._validate_report(
        report,
        expected_backend="cpu",
        expected_data_sha256=DATA_SHA256,
        expected_implementation_root=tmp_path,
        expected_label="candidate",
        expected_revision="deadbeef",
        expected_seed=0,
        expected_workload=comparison.PAPER_WORKLOAD,
    )


@pytest.mark.parametrize(
    ("workload", "sampled_dimensions", "message"),
    [
        (comparison.ALIGNED_WORKLOAD, 15, "config.workload"),
        (comparison.PAPER_WORKLOAD, 11, "config.sampled_dimensions"),
    ],
)
def test_validate_report_rejects_wrong_workload_shape(
    tmp_path: Path,
    workload: str,
    sampled_dimensions: int,
    message: str,
) -> None:
    report = _runner_report(
        tmp_path,
        workload=workload,
        sampled_dimensions=sampled_dimensions,
    )

    with pytest.raises(RuntimeError, match=message):
        comparison._validate_report(
            report,
            expected_backend="cpu",
            expected_data_sha256=DATA_SHA256,
            expected_implementation_root=tmp_path,
            expected_label="candidate",
            expected_revision="deadbeef",
            expected_seed=0,
            expected_workload=comparison.PAPER_WORKLOAD,
        )


def test_summary_and_markdown_record_workload(tmp_path: Path) -> None:
    baseline = _runner_report(
        tmp_path,
        label="baseline",
        revision="baseline-revision",
        sample_seconds=2.0,
    )
    candidate = _runner_report(
        tmp_path,
        label="candidate",
        revision="candidate-revision",
        sample_seconds=1.0,
    )
    summary = comparison._build_summary(
        [baseline, candidate],
        baseline_label="baseline",
        candidate_label="candidate",
        revisions={
            "baseline": "baseline-revision",
            "candidate": "candidate-revision",
        },
        seeds=[0],
        data_sha256=DATA_SHA256,
        workload=comparison.PAPER_WORKLOAD,
    )

    output = tmp_path / "summary.md"
    comparison._write_markdown(
        output,
        summary,
        baseline_label="baseline",
        candidate_label="candidate",
    )

    assert summary["workload"] == comparison.PAPER_WORKLOAD
    assert summary["invariants"]["sampled_dimensions"] == 15
    markdown = output.read_text()
    assert f"Workload: `{comparison.PAPER_WORKLOAD}`" in markdown
    assert "Sampled dimensions: `15`" in markdown
    assert "synthetic-injection catalogue" in markdown


def test_summary_records_per_run_and_aggregate_post_jit_seconds(
    tmp_path: Path,
) -> None:
    baseline = _runner_report(
        tmp_path,
        label="baseline",
        revision="baseline-revision",
        sample_seconds=100.0,
    )
    baseline["timing_seconds"]["jit_compile_estimate"] = 30.0
    candidate = _runner_report(
        tmp_path,
        label="candidate",
        revision="candidate-revision",
        sample_seconds=80.0,
    )
    candidate["schema_version"] = 2
    candidate["timing_seconds"]["paper_convention"] = {
        "post_jit_sampling_seconds": 55.0
    }

    summary = comparison._build_summary(
        [baseline, candidate],
        baseline_label="baseline",
        candidate_label="candidate",
        revisions={
            "baseline": "baseline-revision",
            "candidate": "candidate-revision",
        },
        seeds=[0],
        data_sha256=DATA_SHA256,
        workload=comparison.PAPER_WORKLOAD,
    )

    assert summary["pairs"][0]["baseline_sample_call_seconds"] == pytest.approx(100.0)
    assert summary["pairs"][0]["baseline_post_jit_sample_seconds"] == pytest.approx(
        70.0
    )
    assert summary["pairs"][0]["candidate_post_jit_sample_seconds"] == pytest.approx(
        55.0
    )
    assert summary["groups"]["baseline"]["post_jit_sample_seconds"][
        "samples"
    ] == pytest.approx([70.0])
    assert summary["groups"]["candidate"]["sample_call_seconds"][
        "median"
    ] == pytest.approx(80.0)
    assert summary["groups"]["candidate"]["post_jit_sample_seconds"][
        "median"
    ] == pytest.approx(55.0)

    output = tmp_path / "post-jit-summary.md"
    comparison._write_markdown(
        output,
        summary,
        baseline_label="baseline",
        candidate_label="candidate",
    )
    markdown = output.read_text()
    assert "Post-JIT median [s]" in markdown
    assert "Baseline post-JIT [s]" in markdown
    assert "| baseline | `baseline-rev` | 1 | 100.000 | 70.000 |" in markdown
    assert "| candidate | `candidate-re` | 1 | 80.000 | 55.000 |" in markdown


def test_post_jit_sample_seconds_prefers_paper_convention_block() -> None:
    report = {
        "timing_seconds": {
            "sample_call": 100.0,
            "jit_compile_estimate": 30.0,
            "paper_convention": {"post_jit_sampling_seconds": 55.0},
        }
    }

    assert comparison._post_jit_sample_seconds(report) == pytest.approx(55.0)


def test_post_jit_sample_seconds_falls_back_to_legacy_subtraction() -> None:
    report = {
        "timing_seconds": {
            "sample_call": 100.0,
            "jit_compile_estimate": 30.0,
        }
    }

    assert comparison._post_jit_sample_seconds(report) == pytest.approx(70.0)


def test_post_jit_sample_seconds_falls_back_to_sample_call() -> None:
    report = {"timing_seconds": {"sample_call": 100.0}}

    assert comparison._post_jit_sample_seconds(report) == pytest.approx(100.0)


def test_post_jit_sample_seconds_handles_absent_timing() -> None:
    assert comparison._post_jit_sample_seconds({}) is None
