from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import pytest

from benchmarks.xg import run_qualification
from benchmarks.xg.runpod import workflow
from benchmarks.xg.runpod.upload_and_run import (
    LAUNCH_RECEIPT_KIND,
    _load_launch_receipt,
    _parse_connection,
    _validate_pod_metadata,
)

REPOSITORY = Path(__file__).resolve().parents[3]
RUNPOD_DIR = REPOSITORY / "benchmarks/xg/runpod"


def test_qualification_driver_cli_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "campaign.toml"
    bundle_dir = tmp_path / "bundle"
    observed: list[tuple[Path, Path, int]] = []
    monkeypatch.setattr(
        run_qualification,
        "run_campaign",
        lambda config_path, output_path, *, n_devices: observed.append(
            (config_path, output_path, n_devices)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qualification.py",
            "--config",
            str(config),
            "--bundle-dir",
            str(bundle_dir),
            "--n-devices",
            "4",
        ],
    )

    run_qualification.main()

    assert observed == [(config.resolve(), bundle_dir.resolve(), 4)]


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fake_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".gitignore").write_text("benchmark-results/\n")
    for relative in workflow.REQUIRED_TRACKED_PATHS:
        path = repository.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == workflow.CONFIG_PATH:
            content = "committed-config\n"
        elif path.suffix == ".sh":
            content = "#!/usr/bin/env bash\nset -euo pipefail\n"
        else:
            content = f"fixture for {relative.as_posix()}\n"
        path.write_text(content)
    for relative in workflow.VERIFIED_INPUT_SHA256:
        destination = repository.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY.joinpath(*relative.parts), destination)
    (repository / "order/branch/file.txt").parent.mkdir(parents=True)
    (repository / "order/branch/file.txt").write_text("nested\n")
    (repository / "order-branch.txt").write_text("sibling\n")
    _git(repository, "init")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=XG Test",
        "-c",
        "user.email=xg-test@example.invalid",
        "commit",
        "-m",
        "Create immutable fixture",
    )
    return repository


def _member_bytes(archive: Path, relative: Path) -> bytes:
    verified = workflow.verify_workspace_archive(archive)
    member_name = (verified.root / relative.as_posix()).as_posix()
    with tarfile.open(archive, mode="r:*") as package:
        member = package.getmember(member_name)
        stream = package.extractfile(member)
        assert stream is not None
        return stream.read()


def _install_workspace_manifest(root: Path) -> None:
    manifest = {
        "schema_version": workflow.SCHEMA_VERSION,
        "package": workflow.PACKAGE_KIND,
        "capture": workflow.CAPTURE_DESCRIPTION,
        "source_revision": "a" * 40,
        "git_metadata_included": False,
        "image": workflow.IMAGE,
        "image_sha256": workflow.IMAGE_SHA256,
        "gpu_id": workflow.GPU_ID,
        "gpu_count": workflow.GPU_COUNT,
        "guard_minutes": workflow.GUARD_MINUTES,
        "python_version": workflow.PYTHON_VERSION,
        "uv_version": workflow.UV_VERSION,
        "config_path": workflow.CONFIG_PATH.as_posix(),
        "config_sha256": "b" * 64,
        "uv_lock_sha256": "c" * 64,
        "qualification_entrypoint": workflow.QUALIFICATION_ENTRYPOINT.as_posix(),
        "psd_path": workflow.PSD_PATH.as_posix(),
        "psd_sha256": workflow.PSD_SHA256,
        "verified_inputs": [
            {"path": path.as_posix(), "sha256": digest}
            for path, digest in sorted(
                workflow.VERIFIED_INPUT_SHA256.items(),
                key=lambda item: item[0].as_posix(),
            )
        ],
        "payload_tree_sha256": "d" * 64,
        "file_count": 0,
        "files": [],
    }
    path = root.joinpath(*workflow.PACKAGE_MANIFEST_PATH.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(workflow._canonical_json_bytes(manifest))


def _launch_receipt_payload(
    workspace: workflow.VerifiedArchive,
    started: datetime,
) -> dict[str, object]:
    def timestamp(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    return {
        "schema_version": 1,
        "kind": LAUNCH_RECEIPT_KIND,
        "pod_id": "pod-fixture-1",
        "image": workflow.IMAGE,
        "gpu_id": workflow.GPU_ID,
        "gpu_count": workflow.GPU_COUNT,
        "guard_minutes": workflow.GUARD_MINUTES,
        "launch_started_at": timestamp(started),
        "terminate_after": timestamp(
            started + timedelta(minutes=workflow.GUARD_MINUTES)
        ),
        "workspace_archive_sha256": workspace.sha256,
    }


def test_workspace_package_is_deterministic_committed_head_plus_verified_psd(
    tmp_path: Path,
) -> None:
    repository = _fake_repository(tmp_path)
    repository.joinpath(*workflow.CONFIG_PATH.parts).write_text("dirty-config\n")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_receipt = workflow.build_workspace_archive(repository, first)
    second_receipt = workflow.build_workspace_archive(repository, second)

    assert first_receipt.sha256 == second_receipt.sha256
    assert first_receipt.manifest["source_revision"] == _git(
        repository, "rev-parse", "HEAD"
    )
    assert first_receipt.manifest["capture"] == (
        "git archive of committed HEAD plus verified CE PSD and DE405 ephemerides"
    )
    assert first_receipt.manifest["image"] == workflow.IMAGE
    assert first_receipt.manifest["gpu_count"] == 4
    assert first_receipt.manifest["guard_minutes"] == 120
    assert first_receipt.manifest["psd_sha256"] == workflow.PSD_SHA256
    assert _member_bytes(first, Path(workflow.CONFIG_PATH.as_posix())) == (
        b"committed-config\n"
    )
    assert _member_bytes(first, Path(workflow.PSD_PATH.as_posix())) == (
        REPOSITORY.joinpath(*workflow.PSD_PATH.parts).read_bytes()
    )
    assert _member_bytes(first, Path(workflow.EARTH_EPHEMERIS_PATH.as_posix())) == (
        REPOSITORY.joinpath(*workflow.EARTH_EPHEMERIS_PATH.parts).read_bytes()
    )
    assert _member_bytes(first, Path(workflow.SUN_EPHEMERIS_PATH.as_posix())) == (
        REPOSITORY.joinpath(*workflow.SUN_EPHEMERIS_PATH.parts).read_bytes()
    )
    assert all(
        ".git" not in entry["path"].split("/")
        for entry in first_receipt.manifest["files"]
    )


def test_workspace_verifier_rejects_payload_tampering(tmp_path: Path) -> None:
    repository = _fake_repository(tmp_path)
    archive = tmp_path / "workspace.tar.gz"
    receipt = workflow.build_workspace_archive(repository, archive)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    workflow._extract_checked_archive(archive, extracted)
    root = extracted.joinpath(*receipt.root.parts)
    root.joinpath(*workflow.CONFIG_PATH.parts).write_text("tampered\n")
    tampered = tmp_path / "tampered.tar.gz"
    workflow._write_deterministic_archive(root, tampered)

    with pytest.raises(ValueError, match="inventory|SHA-256"):
        workflow.verify_workspace_archive(tampered)
    with pytest.raises(ValueError, match="inventory|SHA-256"):
        workflow.verify_extracted_workspace(root)


def test_workspace_package_rejects_wrong_psd(tmp_path: Path) -> None:
    repository = _fake_repository(tmp_path)
    repository.joinpath(*workflow.PSD_PATH.parts).write_text("wrong PSD\n")

    with pytest.raises(ValueError, match="XG input SHA-256 mismatch"):
        workflow.build_workspace_archive(repository, tmp_path / "workspace.tar.gz")


def test_workspace_package_rejects_wrong_ephemeris(tmp_path: Path) -> None:
    repository = _fake_repository(tmp_path)
    repository.joinpath(*workflow.EARTH_EPHEMERIS_PATH.parts).write_bytes(b"wrong")

    with pytest.raises(ValueError, match="XG input SHA-256 mismatch"):
        workflow.build_workspace_archive(repository, tmp_path / "workspace.tar.gz")


def test_result_archive_preserves_and_verifies_failure_status(tmp_path: Path) -> None:
    root = tmp_path / "jim-xg-results"
    root.mkdir()
    (root / "qualification.log").write_text("failed threshold\n")
    _install_workspace_manifest(root)
    workflow.write_result_manifest(root, 7)
    archive = tmp_path / "result.tar.gz"
    workflow._write_deterministic_archive(root, archive)

    verified = workflow.verify_result_archive(archive)

    assert verified.exit_status == 7
    assert verified.manifest["package"] == workflow.RESULT_KIND
    assert verified.manifest["file_count"] == 3
    assert verified.manifest["source_revision"] == "a" * 40


def test_result_verifier_rejects_special_archive_member(tmp_path: Path) -> None:
    archive = tmp_path / "special.tar.gz"
    with tarfile.open(archive, mode="w:gz") as package:
        root = tarfile.TarInfo("result")
        root.type = tarfile.DIRTYPE
        package.addfile(root)
        link = tarfile.TarInfo("result/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        package.addfile(link)

    with pytest.raises(ValueError, match="forbidden special member"):
        workflow.verify_result_archive(archive)


def test_result_verifier_rejects_inventory_tampering(tmp_path: Path) -> None:
    root = tmp_path / "result"
    root.mkdir()
    (root / "science.log").write_text("complete\n")
    _install_workspace_manifest(root)
    complete = root.joinpath(*workflow.COMPLETE_PATH.parts)
    complete.touch()
    workflow.write_result_manifest(root, 0)
    (root / "science.log").write_text("changed after manifest\n")
    archive = tmp_path / "tampered-result.tar.gz"
    workflow._write_deterministic_archive(root, archive)

    with pytest.raises(ValueError, match="inventory|SHA-256"):
        workflow.verify_result_archive(archive)


def test_remote_publish_function_produces_a_verified_archive(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    result_root.mkdir()
    _install_workspace_manifest(result_root)
    result_root.joinpath(*workflow.COMPLETE_PATH.parts).touch()
    result_archive = tmp_path / "published.tar.gz"
    script = (RUNPOD_DIR / "run_on_pod.sh").read_text()
    function_start = script.index("publish_results() {")
    function_end = script.index("\n}\ntrap ", function_start) + 2
    publish_function = script[function_start:function_end]
    publish_script = f"""set -euo pipefail
repository="$XG_TEST_REPOSITORY"
output_root="$XG_TEST_OUTPUT_ROOT"
result_archive="$XG_TEST_RESULT_ARCHIVE"
{publish_function}
publish_results 0
"""

    result = subprocess.run(
        ["bash", "-c", publish_script],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "XG_TEST_REPOSITORY": str(REPOSITORY),
            "XG_TEST_OUTPUT_ROOT": str(result_root),
            "XG_TEST_RESULT_ARCHIVE": str(result_archive),
        },
    )

    assert result.returncode == 0, result.stderr
    assert workflow.verify_result_archive(result_archive).exit_status == 0
    assert Path(f"{result_archive}.sha256").is_file()


def test_success_result_requires_completion_marker(tmp_path: Path) -> None:
    root = tmp_path / "result"
    root.mkdir()
    _install_workspace_manifest(root)

    with pytest.raises(ValueError, match="completion marker"):
        workflow.write_result_manifest(root, 0)


def test_provisioner_is_dry_by_default_and_has_no_resource_overrides() -> None:
    provision = RUNPOD_DIR / "provision.sh"
    result = subprocess.run(
        ["bash", str(provision)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    assert workflow.IMAGE in output
    assert "--gpu-id NVIDIA\\ H200" in output
    assert "--gpu-count 4" in output
    assert "Automatic deletion deadline:" in output
    assert "(120 minutes)" in output
    assert "Dry run only" in output

    rejected = subprocess.run(
        ["bash", str(provision), "--gpu-id", "NVIDIA H100 PCIe"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2

    launch_without_package = subprocess.run(
        ["bash", str(provision), "--launch"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert launch_without_package.returncode == 2
    assert "requires the immutable --archive" in launch_without_package.stderr


def test_remote_runner_orders_every_fail_closed_gate_before_sampling() -> None:
    script = (RUNPOD_DIR / "run_on_pod.sh").read_text()

    qualification = script.index('"$qualification_entrypoint"')
    immutable_recheck = script.index("post-qualification-package.json")
    assembly = script.index("jim-xg-qualify assemble")
    pre_science_recheck = script.index("pre-science-package.json")
    science = script.index("jim-run --verbose")
    post_science_recheck = script.index("post-science-package.json")
    completion = script.index('touch "$output_root/.runpod/complete"')
    assert (
        qualification
        < immutable_recheck
        < assembly
        < pre_science_recheck
        < science
        < post_science_recheck
        < completion
    )
    assert "--n-devices 4" in script
    assert "55m" in script
    assert "30m" in script
    assert "trap 'publish_results $?' EXIT" in script
    assert "Expected exactly four JAX CUDA devices" in script
    assert "compile(source.read_bytes()" in script
    assert 'archive_python="$UV_PROJECT_ENVIRONMENT/bin/python"' in script
    assert "Qualification manifest changed during science sampling" in script
    assert workflow.EARTH_EPHEMERIS_PATH in workflow.VERIFIED_INPUT_SHA256
    assert workflow.SUN_EPHEMERIS_PATH in workflow.VERIFIED_INPUT_SHA256


def test_launch_receipt_binds_package_and_fixed_deletion_guard(tmp_path: Path) -> None:
    workspace = workflow.VerifiedArchive(
        path=tmp_path / "workspace.tar.gz",
        sha256="a" * 64,
        root=PurePosixPath("workspace"),
        manifest={},
    )
    started = datetime(2026, 9, 1, 12, tzinfo=UTC)
    receipt_path = tmp_path / "launch.json"
    receipt_path.write_text(
        json.dumps(_launch_receipt_payload(workspace, started)),
        encoding="utf-8",
    )

    receipt = _load_launch_receipt(
        receipt_path,
        workspace,
        now=started + timedelta(minutes=10),
    )

    assert receipt.pod_id == "pod-fixture-1"
    assert receipt.workspace_archive_sha256 == workspace.sha256
    assert receipt.terminate_after == started + timedelta(minutes=120)

    tampered = _launch_receipt_payload(workspace, started)
    tampered["guard_minutes"] = 119
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="guard_minutes"):
        _load_launch_receipt(
            receipt_path,
            workspace,
            now=started + timedelta(minutes=10),
        )


def test_live_pod_metadata_requires_exact_paid_launch() -> None:
    payload = {
        "id": "pod-fixture-1",
        "imageName": workflow.IMAGE,
        "gpuCount": workflow.GPU_COUNT,
        "desiredStatus": "RUNNING",
        "runtimeStatus": "running",
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "machine": {
            "secureCloud": True,
            "gpuId": workflow.GPU_ID,
        },
    }

    _validate_pod_metadata(payload, "pod-fixture-1")

    payload["imageName"] = "runpod/pytorch:mutable"
    with pytest.raises(ValueError, match="imageName"):
        _validate_pod_metadata(payload, "pod-fixture-1")


def test_parse_connection_pins_explicit_identity_and_port(tmp_path: Path) -> None:
    identity = tmp_path / "id_ed25519"
    identity.write_text("fixture")
    payload = json.dumps(
        {"sshCommand": f"ssh root@example.invalid -p 22022 -i {identity}"}
    )

    connection = _parse_connection(payload)

    assert connection.target == "root@example.invalid"
    assert connection.port == "22022"
    assert connection.identity == identity.resolve()


def test_parse_connection_rejects_missing_identity(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    payload = json.dumps(
        {"nested": {"ssh_command": f"ssh root@example.invalid -p 22 -i {missing}"}}
    )

    with pytest.raises(ValueError, match="identity does not exist"):
        _parse_connection(payload)


def test_archive_path_rejects_parent_escape() -> None:
    with pytest.raises(ValueError, match="unsafe path"):
        workflow._safe_archive_path("result/../outside")


def test_manifest_loader_rejects_non_object_json(tmp_path: Path) -> None:
    archive = tmp_path / "manifest.tar.gz"
    payload = b"[]\n"
    with tarfile.open(archive, mode="w:gz") as package:
        root = tarfile.TarInfo("root")
        root.type = tarfile.DIRTYPE
        package.addfile(root)
        member = tarfile.TarInfo("root/.runpod/package-manifest.json")
        member.size = len(payload)
        package.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="JSON object"):
        workflow.verify_workspace_archive(archive)
