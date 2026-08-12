import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign.runpod import upload_and_run


def _csv_bytes(fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _add_bytes(package: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    package.addfile(member, io.BytesIO(payload))


def _valid_archive_payloads() -> dict[str, bytes]:
    catalogue_rows = []
    for injection_id in range(2):
        row: dict[str, object] = {
            field: float(injection_id) for field in common.CATALOGUE_FIELDS
        }
        row.update(
            {
                "injection_id": injection_id,
                "noise_seed": 100 + injection_id,
                "sampler_seed": 200 + injection_id,
            }
        )
        catalogue_rows.append(row)
    catalogue = _csv_bytes(common.CATALOGUE_FIELDS, catalogue_rows)
    status = _csv_bytes(
        common.STATUS_FIELDS,
        [
            {
                "injection_id": injection_id,
                "status": "pending",
                "attempts": 0,
                "runtime_seconds": "",
                "posterior_samples": "",
                "summary": "",
                "posterior": "",
                "error": "",
            }
            for injection_id in range(2)
        ],
    )
    psd = b"frozen-psd"
    manifest: dict[str, object] = {
        "schema_version": common.SCHEMA_VERSION,
        "n_injections": 2,
        "catalogue_size": 2,
        "catalogue": {
            "path": "catalogue.csv",
            "bytes": len(catalogue),
            "sha256": hashlib.sha256(catalogue).hexdigest(),
        },
        "psd": {
            "files": {
                "inputs/psd/design.npz": {
                    "bytes": len(psd),
                    "sha256": hashlib.sha256(psd).hexdigest(),
                }
            }
        },
    }
    manifest["config_sha256"] = common.canonical_sha256(manifest)
    return {
        "manifest.json": (json.dumps(manifest) + "\n").encode(),
        "catalogue.csv": catalogue,
        "status.csv": status,
        "inputs/psd/design.npz": psd,
    }


def _write_campaign_archive(
    path: Path,
    *,
    payloads: dict[str, bytes] | None = None,
    extra_members: tuple[tarfile.TarInfo, ...] = (),
) -> None:
    content = _valid_archive_payloads() if payloads is None else payloads
    with tarfile.open(path, mode="w:gz") as package:
        root = tarfile.TarInfo("historical-stress")
        root.type = tarfile.DIRTYPE
        package.addfile(root)
        for relative, payload in content.items():
            _add_bytes(package, f"historical-stress/{relative}", payload)
        for member in extra_members:
            package.addfile(member)


def test_frozen_campaign_archive_is_verified_without_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "campaign-input.tar.gz"
    payloads = _valid_archive_payloads()
    _write_campaign_archive(archive, payloads=payloads)

    frozen = upload_and_run._verify_frozen_campaign_archive(archive)

    assert frozen.root.as_posix() == "historical-stress"
    assert frozen.manifest["n_injections"] == 2
    assert (
        frozen.manifest_sha256 == hashlib.sha256(payloads["manifest.json"]).hexdigest()
    )
    assert (
        frozen.catalogue_sha256 == hashlib.sha256(payloads["catalogue.csv"]).hexdigest()
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("historical-stress/results/injection-000/summary.json", "results/cache"),
        ("historical-stress/.jax-cache/kernel", "results/cache"),
        ("another-root/file", "exactly one root"),
    ],
)
def test_frozen_campaign_archive_rejects_non_input_data(
    tmp_path: Path,
    name: str,
    expected: str,
) -> None:
    archive = tmp_path / "campaign-input.tar.gz"
    extra = tarfile.TarInfo(name)
    extra.size = 0
    _write_campaign_archive(archive, extra_members=(extra,))

    with pytest.raises(ValueError, match=expected):
        upload_and_run._verify_frozen_campaign_archive(archive)


def test_frozen_campaign_archive_rejects_special_members(tmp_path: Path) -> None:
    archive = tmp_path / "campaign-input.tar.gz"
    link = tarfile.TarInfo("historical-stress/inputs/psd/link.npz")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    _write_campaign_archive(archive, extra_members=(link,))

    with pytest.raises(ValueError, match="forbidden special member"):
        upload_and_run._verify_frozen_campaign_archive(archive)


def test_frozen_campaign_archive_rejects_changed_catalogue(tmp_path: Path) -> None:
    archive = tmp_path / "campaign-input.tar.gz"
    payloads = _valid_archive_payloads()
    payloads["catalogue.csv"] += b"\n"
    _write_campaign_archive(archive, payloads=payloads)

    with pytest.raises(ValueError, match="catalogue hash mismatch"):
        upload_and_run._verify_frozen_campaign_archive(archive)


def test_candidate_diagnostic_must_pin_packaged_candidate_tree() -> None:
    revision = "a" * 40
    tree_sha256 = "b" * 64
    manifest = {
        "implementation_diagnostic": {
            "implementation_label": "candidate",
            "implementation_revision": revision,
            "implementation_tree_sha256": tree_sha256,
        }
    }
    package = {
        "candidate": {"revision": revision, "tree_sha256": tree_sha256},
    }

    upload_and_run._validate_frozen_implementation_pin(
        manifest,
        package,
        "candidate",
    )

    package["candidate"]["tree_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="does not pin the packaged candidate"):
        upload_and_run._validate_frozen_implementation_pin(
            manifest,
            package,
            "candidate",
        )


def test_staged_command_requires_and_preserves_frozen_campaign() -> None:
    args = SimpleNamespace(
        seed=123,
        retry_count=0,
        start=2,
        stop=10,
        plot=False,
    )

    command = upload_and_run._build_run_command(
        args,
        "/workspace/campaign",
        n_injections=10,
        catalogue_size=10,
        frozen=True,
    )

    assert command[command.index("--start") + 1] == "2"
    assert command[command.index("--stop") + 1] == "10"
    assert "--require-existing-campaign" in command
    assert "--no-plot" in command


def test_staged_upload_without_frozen_campaign_is_refused() -> None:
    with pytest.raises(SystemExit, match="require --campaign-input"):
        upload_and_run.main(
            ["pod-123", "workspace.tar.gz", "--start", "0", "--stop", "2"]
        )


def test_pod_runner_can_require_frozen_input_and_disable_plots() -> None:
    repository = Path(__file__).resolve().parents[3]
    script = (
        repository / "benchmarks/injection_campaign/runpod/run_on_pod.sh"
    ).read_text(encoding="utf-8")

    assert "--require-existing-campaign" in script
    assert "refusing remote regeneration" in script
    assert "--no-plot" in script
    assert "plot_arguments+=(--plot)" in script
    assert "Candidate likelihood preflight passed" in script
    assert "candidate likelihood preflight implementation pin mismatch" in script
