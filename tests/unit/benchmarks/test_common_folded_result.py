from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.injection_campaign import common


def _write_completed_result(directory: Path) -> tuple[str, Path]:
    directory.mkdir()
    config_sha256 = "a" * 64
    posterior_path = directory / "posterior.npz"
    posterior_path.write_bytes(b"posterior payload")
    common.atomic_write_json(
        directory / "summary.json",
        {
            "config_sha256": config_sha256,
            "posterior": {
                "path": "posterior.npz",
                "sha256": common.file_sha256(posterior_path),
                "bytes": posterior_path.stat().st_size,
            },
        },
    )
    return config_sha256, posterior_path


def test_completed_folded_result_requires_declared_diagnostic_artifact(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "result"
    config_sha256, _ = _write_completed_result(directory)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["folded_nested_diagnostics"] = {
        "path": "folded_nested_diagnostics.npz",
        "sha256": "b" * 64,
        "bytes": 17,
    }
    common.atomic_write_json(summary_path, summary)

    with pytest.raises(ValueError, match="folded diagnostic artifact is missing"):
        common.validate_completed_result(directory, config_sha256)


def test_completed_folded_result_validates_companion_path_hash_and_bytes(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "result"
    config_sha256, _ = _write_completed_result(directory)
    diagnostic_path = directory / "folded_nested_diagnostics.npz"
    diagnostic_path.write_bytes(b"folded diagnostic payload")
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["folded_nested_diagnostics"] = {
        "path": "folded_nested_diagnostics.npz",
        "sha256": common.file_sha256(diagnostic_path),
        "bytes": diagnostic_path.stat().st_size,
    }
    common.atomic_write_json(summary_path, summary)

    assert common.validate_completed_result(directory, config_sha256) == summary

    diagnostic_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="folded diagnostic hash mismatch"):
        common.validate_completed_result(directory, config_sha256)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"path": "../folded_nested_diagnostics.npz"},
        {"path": "other.npz"},
    ],
)
def test_completed_folded_result_rejects_invalid_companion_metadata(
    tmp_path: Path,
    metadata: object,
) -> None:
    directory = tmp_path / "result"
    config_sha256, _ = _write_completed_result(directory)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["folded_nested_diagnostics"] = metadata
    common.atomic_write_json(summary_path, summary)

    with pytest.raises(
        (TypeError, ValueError),
        match="folded diagnostic metadata|folded diagnostic path",
    ):
        common.validate_completed_result(directory, config_sha256)
