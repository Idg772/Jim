"""Focused packaging contract for NETSKY campaign dependencies."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.device_parallel_nss import analyze_gw170817_sampler_output
from benchmarks.device_parallel_nss.runpod import upload_and_run
from benchmarks.injection_campaign import common as campaign_common

GW170817_POSITION_FIELDS = (
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
    "ra",
    "dec",
    "psi",
)


def _run_pod_parser(
    tmp_path: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[3]
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"
    return subprocess.run(
        [
            "bash",
            str(repository / "benchmarks/device_parallel_nss/runpod/run_on_pod.sh"),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _netsky_artifact_report(tmp_path: Path) -> dict[str, Any]:
    posterior_path = tmp_path / "posterior.npz"
    posterior = {
        **{
            name: np.asarray([float(index), float(index + 1)])
            for index, name in enumerate(GW170817_POSITION_FIELDS)
        },
        "log_likelihood": np.asarray([100.0, 101.0]),
        "log_weights": np.log(np.asarray([0.4, 0.6])),
    }
    np.savez(posterior_path, **posterior)
    folded_path = tmp_path / "posterior-folded-nested-diagnostics.npz"
    folded = {
        "log_likelihood": np.asarray([10.0, 11.0]),
        "log_likelihood_birth": np.asarray([-np.inf, 9.0]),
    }
    np.savez(folded_path, **folded)
    return {
        "config": {
            "blocking_scheme": "netsky",
            "num_gibbs_sweeps": 2,
            "blocks": campaign_common.NETSKY_BLOCKS,
            "bridge_blocks": campaign_common.NETSKY_BRIDGE_BLOCKS,
            "periodic_wrapped_covariance": True,
            "fold_symmetry": {
                "cos_iota": "cos_iota",
                "azimuth": "azimuth",
                "psi": "psi",
                "azimuth_reflection_center": 1.25,
            },
        },
        "results": {
            "posterior_artifact": {
                "path": str(posterior_path),
                "sha256": upload_and_run._sha256_file(posterior_path),
                "bytes": posterior_path.stat().st_size,
                "format": "npz",
                "space": "prior",
                "weighting": campaign_common.UNFOLDED_POSTERIOR_WEIGHTING,
                "count": 2,
                "fields": list(posterior),
                "schema_version": 2,
                "weight_effective_size_semantics": (
                    campaign_common.POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
                ),
            },
            "folded_nested_diagnostics": {
                "path": str(folded_path),
                "sha256": upload_and_run._sha256_file(folded_path),
                "bytes": folded_path.stat().st_size,
                "format": "npz",
                "space": "folded sampling-space target",
                "weighting": "not applicable: folded nested-sampling contours",
                "count": 2,
                "fields": list(folded),
                "semantics": campaign_common.FOLDED_TARGET_SEMANTICS,
            },
            "posterior_weight_effective_size": 1.92,
            "posterior_weight_effective_size_semantics": (
                campaign_common.POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
            ),
            "quotient_fold": {"group_order": 8, "folded_points": 2},
        },
    }


def test_workspace_package_includes_folded_result_helpers() -> None:
    repository = Path(__file__).resolve().parents[3]
    package_script = (
        repository / "benchmarks/device_parallel_nss/runpod/package_workspace.sh"
    ).read_text(encoding="utf-8")

    assert '"benchmarks/injection_campaign/folded_results.py"' in package_script


def test_netsky_validator_uses_real_gw170817_posterior_fields() -> None:
    assert upload_and_run.GW170817_PAPER_15D_PARAMETERS == GW170817_POSITION_FIELDS
    assert "d_L" in upload_and_run.GW170817_PAPER_15D_PARAMETERS
    assert "t_c" not in upload_and_run.GW170817_PAPER_15D_PARAMETERS


def test_upload_cli_locks_netsky_to_m2_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_and_run.py",
            "pod-123",
            "workspace.tar.gz",
            "--candidate-only",
            "--workload",
            "paper-15d",
            "--blocking-scheme",
            "netsky",
        ],
    )

    args = upload_and_run._parse_args()
    command = upload_and_run._build_run_command(args, "/remote/results")

    assert args.num_gibbs_sweeps == 2
    assert args.direction_mode == "covariance"
    assert command[command.index("--num-gibbs-sweeps") + 1] == "2"
    assert command[command.index("--direction-mode") + 1] == "covariance"


def test_upload_cli_rejects_noncanonical_netsky_sweep_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_and_run.py",
            "pod-123",
            "workspace.tar.gz",
            "--candidate-only",
            "--workload",
            "paper-15d",
            "--blocking-scheme",
            "netsky",
            "--num-gibbs-sweeps",
            "1",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        upload_and_run._parse_args()


def test_upload_cli_rejects_non_covariance_netsky_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_and_run.py",
            "pod-123",
            "workspace.tar.gz",
            "--candidate-only",
            "--workload",
            "paper-15d",
            "--blocking-scheme",
            "netsky",
            "--direction-mode",
            "de-mix",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        upload_and_run._parse_args()


def test_upload_cli_rejects_netsky_sampler_options_for_legacy_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_and_run.py",
            "pod-123",
            "workspace.tar.gz",
            "--candidate-only",
            "--num-gibbs-sweeps",
            "2",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        upload_and_run._parse_args()


def test_pod_runner_accepts_canonical_netsky_defaults(tmp_path: Path) -> None:
    result = _run_pod_parser(
        tmp_path,
        "--candidate-only",
        "--workload",
        "paper-15d",
        "--blocking-scheme",
        "netsky",
    )

    assert result.returncode == 1
    assert "Expected four GPUs" in result.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--num-gibbs-sweeps", "1"), "fixes --num-gibbs-sweeps at 2"),
        (("--direction-mode", "de-mix"), "requires --direction-mode covariance"),
    ],
)
def test_pod_runner_rejects_noncanonical_netsky_sampler_options(
    tmp_path: Path,
    arguments: tuple[str, str],
    message: str,
) -> None:
    result = _run_pod_parser(
        tmp_path,
        "--candidate-only",
        "--workload",
        "paper-15d",
        "--blocking-scheme",
        "netsky",
        *arguments,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_candidate_run_mode_includes_netsky_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_and_run.py",
            "pod-123",
            "workspace.tar.gz",
            "--candidate-only",
            "--candidate-seed",
            "2",
            "--workload",
            "paper-15d",
            "--blocking-scheme",
            "netsky",
        ],
    )

    args = upload_and_run._parse_args()

    assert upload_and_run._run_mode(args) == "candidate-netsky-seed2"


def test_netsky_pilot_and_three_seed_modes_are_collision_safe() -> None:
    modes = [upload_and_run._candidate_run_mode("netsky", seed) for seed in range(3)]

    assert modes == [
        "candidate-netsky-seed0",
        "candidate-netsky-seed1",
        "candidate-netsky-seed2",
    ]
    assert len(set(modes)) == 3


@pytest.mark.parametrize(
    ("blocking_scheme", "expected"),
    (("paper", "candidate"), ("all-slow", "candidate-all-slow")),
)
def test_candidate_run_mode_preserves_legacy_paths(
    monkeypatch: pytest.MonkeyPatch,
    blocking_scheme: str,
    expected: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_and_run.py",
            "pod-123",
            "workspace.tar.gz",
            "--candidate-only",
            "--candidate-seed",
            "2",
            "--workload",
            "paper-15d",
            "--blocking-scheme",
            blocking_scheme,
        ],
    )

    args = upload_and_run._parse_args()

    assert upload_and_run._run_mode(args) == expected


def test_netsky_artifact_verifier_checks_physical_and_folded_products(
    tmp_path: Path,
) -> None:
    report = _netsky_artifact_report(tmp_path)

    verification = upload_and_run.validate_netsky_result_artifacts(report)

    assert verification == {
        "posterior_samples": 2,
        "posterior_sha256": report["results"]["posterior_artifact"]["sha256"],
        "folded_nested_samples": 2,
        "folded_nested_sha256": report["results"]["folded_nested_diagnostics"][
            "sha256"
        ],
    }


def test_netsky_artifact_verifier_rejects_a_changed_folded_product(
    tmp_path: Path,
) -> None:
    report = _netsky_artifact_report(tmp_path)
    folded_path = Path(report["results"]["folded_nested_diagnostics"]["path"])
    with folded_path.open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        upload_and_run.validate_netsky_result_artifacts(report)


def test_downloaded_netsky_archive_contains_both_verified_products(
    tmp_path: Path,
) -> None:
    report = _netsky_artifact_report(tmp_path)
    verification = upload_and_run.validate_netsky_result_artifacts(report)
    report_path = tmp_path / "candidate-paper-15d-netsky-g4-seed0.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    verification_path = tmp_path / "artifact-verification.json"
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    archive = tmp_path / "results.tar.gz"
    root = "pod-123-candidate-netsky-seed0"
    with tarfile.open(archive, "w:gz") as package:
        package.add(report_path, arcname=f"{root}/{report_path.name}")
        package.add(
            Path(report["results"]["posterior_artifact"]["path"]),
            arcname=f"{root}/posterior/posterior.npz",
        )
        package.add(
            Path(report["results"]["folded_nested_diagnostics"]["path"]),
            arcname=f"{root}/posterior/posterior-folded-nested-diagnostics.npz",
        )
        package.add(
            verification_path,
            arcname=f"{root}/artifact-verification.json",
        )

    transported = upload_and_run._verify_netsky_results_archive(
        archive,
        expected_root=root,
    )

    assert transported == verification


def test_pod_runner_records_netsky_folded_artifact_verification() -> None:
    repository = Path(__file__).resolve().parents[3]
    runner = (
        repository / "benchmarks/device_parallel_nss/runpod/run_on_pod.sh"
    ).read_text(encoding="utf-8")

    assert "validate_netsky_result_artifacts" in runner
    assert "verification.update(netsky_verification)" in runner


@pytest.mark.parametrize("local_parent", ("", "posterior"))
def test_analyzer_relocates_remote_folded_artifact_beside_report(
    tmp_path: Path,
    local_parent: str,
) -> None:
    report_path = tmp_path / "extracted" / "run.json"
    report_path.parent.mkdir()
    folded_path = report_path.parent / local_parent / "seed0-folded.npz"
    folded_path.parent.mkdir(exist_ok=True)
    np.savez(
        folded_path,
        log_likelihood=np.asarray([1.0]),
        log_likelihood_birth=np.asarray([-np.inf]),
    )
    reports = [
        {
            "config": {"blocking_scheme": "netsky"},
            "results": {
                "folded_nested_diagnostics": {
                    "path": "/workspace/results/posterior/seed0-folded.npz"
                }
            },
        }
    ]

    resolved = analyze_gw170817_sampler_output._resolve_folded_paths(
        [report_path],
        reports,
        explicit_paths=None,
    )

    assert resolved == [folded_path.resolve()]
