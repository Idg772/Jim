import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from benchmarks.device_parallel_nss.runpod import upload_and_run

BASELINE_REVISION = "86335bdb1e7ef6191937dd17b2ca53edbb1d899f"
CANDIDATE_REVISION = "c" * 40
BASELINE_ROOT = ".runpod/implementations/paper-baseline"


def _inventory_entry(path: str, data: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _tree_sha256(path: str, data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    aggregate = hashlib.sha256()
    aggregate.update(path.encode())
    aggregate.update(b"\0")
    aggregate.update(digest)
    return aggregate.hexdigest()


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


def _write_valid_package(
    path: Path,
    *,
    candidate_tree_sha256: str | None = None,
    extra_members: tuple[tarfile.TarInfo, ...] = (),
    extra_files: tuple[tuple[str, bytes], ...] = (),
) -> None:
    candidate_path = "src/jimgw/__init__.py"
    candidate_data = b"candidate\n"
    baseline_path = "src/jimgw/__init__.py"
    baseline_data = b"baseline\n"
    manifest = {
        "schema_version": 1,
        "package": "jim-gw170817-runpod",
        "provenance": {"git_metadata_included": False},
        "candidate": {
            "revision": CANDIDATE_REVISION,
            "tree_sha256": candidate_tree_sha256
            or _tree_sha256(candidate_path, candidate_data),
            "file_count": 1,
            "files": [_inventory_entry(candidate_path, candidate_data)],
        },
        "baseline": {
            "revision": BASELINE_REVISION,
            "root": BASELINE_ROOT,
            "tree_sha256": _tree_sha256(baseline_path, baseline_data),
            "file_count": 1,
            "files": [_inventory_entry(baseline_path, baseline_data)],
        },
    }
    with tarfile.open(path, mode="w:gz") as archive:
        _add_bytes(archive, candidate_path, candidate_data)
        _add_bytes(
            archive,
            f"{BASELINE_ROOT}/{baseline_path}",
            baseline_data,
        )
        _add_bytes(
            archive,
            upload_and_run.PACKAGE_MANIFEST_PATH.as_posix(),
            (json.dumps(manifest) + "\n").encode(),
        )
        for name, data in extra_files:
            _add_bytes(archive, name, data)
        for member in extra_members:
            archive.addfile(member)


def test_verify_workspace_archive_accepts_curated_package(tmp_path: Path) -> None:
    archive = tmp_path / "workspace.tar.gz"
    _write_valid_package(archive)

    manifest = upload_and_run._verify_workspace_archive(archive)

    assert manifest["candidate"]["revision"] == CANDIDATE_REVISION
    assert manifest["baseline"]["revision"] == BASELINE_REVISION


def test_verify_workspace_archive_rejects_false_tree_digest(tmp_path: Path) -> None:
    archive = tmp_path / "workspace.tar.gz"
    _write_valid_package(archive, candidate_tree_sha256="0" * 64)

    with pytest.raises(ValueError, match="candidate tree SHA-256 mismatch"):
        upload_and_run._verify_workspace_archive(archive)


def test_ssh_options_pin_a_per_run_known_hosts_file(tmp_path: Path) -> None:
    connection = upload_and_run.SSHConnection(
        target="root@example.test",
        port="22022",
        identity=tmp_path / "identity",
    )
    known_hosts = tmp_path / "known_hosts"

    first_probe = connection.ssh_options(known_hosts, accept_new=True)
    subsequent = connection.ssh_options(known_hosts)
    scp = connection.scp_options(known_hosts)

    assert "StrictHostKeyChecking=accept-new" in first_probe
    assert "StrictHostKeyChecking=yes" in subsequent
    assert "StrictHostKeyChecking=yes" in scp
    assert f"UserKnownHostsFile={known_hosts}" in first_probe
    assert f"UserKnownHostsFile={known_hosts}" in subsequent
    assert f"UserKnownHostsFile={known_hosts}" in scp
    assert all(
        "/dev/null" not in option for option in (*first_probe, *subsequent, *scp)
    )


@pytest.mark.parametrize(
    "name",
    (
        ".git/config",
        "candidate/.git/objects/pack",
        "/absolute/path",
        "src/../outside",
    ),
)
def test_verify_workspace_archive_rejects_unsafe_paths(
    tmp_path: Path,
    name: str,
) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    _write_valid_package(archive, extra_files=((name, b"unsafe\n"),))

    with pytest.raises(ValueError, match="unsafe path|forbidden Git metadata"):
        upload_and_run._verify_workspace_archive(archive)


@pytest.mark.parametrize(
    "member_type",
    (
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        tarfile.FIFOTYPE,
    ),
)
def test_verify_workspace_archive_rejects_special_members(
    tmp_path: Path,
    member_type: bytes,
) -> None:
    member = tarfile.TarInfo("special-member")
    member.type = member_type
    if member_type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
        member.linkname = "src/jimgw/__init__.py"
    archive = tmp_path / "special.tar.gz"
    _write_valid_package(archive, extra_members=(member,))

    with pytest.raises(ValueError, match="forbidden special member"):
        upload_and_run._verify_workspace_archive(archive)


def test_verify_workspace_archive_rejects_uninventoried_file(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "unrelated.tar.gz"
    _write_valid_package(
        archive,
        extra_files=(("tmp/unrelated-output.json", b"not curated\n"),),
    )

    with pytest.raises(ValueError, match="outside the curated inventory"):
        upload_and_run._verify_workspace_archive(archive)


def test_package_workspace_includes_integration_test_import_closure() -> None:
    repository = Path(__file__).resolve().parents[3]
    package_script = (
        repository / "benchmarks/device_parallel_nss/runpod/package_workspace.sh"
    ).read_text()

    for required_path in (
        "tests/__init__.py",
        "tests/integration/__init__.py",
        "tests/integration/_helpers.py",
        "tests/integration/test_sampler_sharding.py",
    ):
        assert f'"{required_path}"' in package_script


def test_package_workspace_includes_paper_model_basis_and_tests() -> None:
    repository = Path(__file__).resolve().parents[3]
    package_script = (
        repository / "benchmarks/device_parallel_nss/runpod/package_workspace.sh"
    ).read_text()
    pod_runner = (
        repository / "benchmarks/device_parallel_nss/runpod/run_on_pod.sh"
    ).read_text()

    assert '"benchmarks/device_parallel_nss/paper_model_basis.py"' in package_script
    assert '"tests/unit/benchmarks/test_paper_model_basis.py"' in package_script
    assert "tests/unit/benchmarks/test_paper_model_basis.py" in pod_runner


def test_candidate_only_runner_makes_gpu_hlo_optional() -> None:
    repository = Path(__file__).resolve().parents[3]
    runner = (
        repository / "benchmarks/device_parallel_nss/runpod/run_on_pod.sh"
    ).read_text()

    assert '--samples-output "$samples_file"' in runner
    assert '--profile-dir "$profile_dir"' in runner
    assert '--telemetry-output "$telemetry_file"' in runner
    assert 'slice_arguments=(--slice-data-output "$slice_file")' in runner
    assert 'if [[ "$no_hlo" == false ]]; then' in runner
    assert '"XLA_FLAGS=--xla_dump_to=$hlo_dir --xla_dump_hlo_as_text"' in runner
    assert 'verification["hlo_capture"] = False' in runner
    assert 'path.name.endswith("after_optimizations.txt")' in runner
    assert 'hlo_manifest_path = hlo_path / "hlo-manifest.json"' in runner
    assert '"optimized_hlo_files": hlo_manifest[' in runner


def test_upload_cli_accepts_no_hlo_for_candidate_only(
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
            "--no-hlo",
        ],
    )

    args = upload_and_run._parse_args()

    assert args.candidate_only is True
    assert args.no_hlo is True
    assert upload_and_run._build_run_command(args, "/remote/results")[-2:] == [
        "--candidate-only",
        "--no-hlo",
    ]


def test_upload_cli_keeps_hlo_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["upload_and_run.py", "pod-123", "workspace.tar.gz", "--candidate-only"],
    )

    args = upload_and_run._parse_args()

    assert args.no_hlo is False
    assert "--no-hlo" not in upload_and_run._build_run_command(args, "/remote/results")


def test_upload_cli_accepts_frozen_data_for_candidate_only(
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
            "--data-file",
            "gw170817.npz",
        ],
    )

    args = upload_and_run._parse_args()

    assert args.data_file == Path("gw170817.npz")


def test_upload_cli_rejects_frozen_data_for_baseline_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_and_run.py",
            "pod-123",
            "workspace.tar.gz",
            "--original-sharded-only",
            "--data-file",
            "gw170817.npz",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        upload_and_run._parse_args()


def test_upload_cli_rejects_no_hlo_without_candidate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_and_run.py",
            "pod-123",
            "workspace.tar.gz",
            "--original-sharded-only",
            "--no-hlo",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        upload_and_run._parse_args()
