"""Focused packaging contract for NETSKY campaign dependencies."""

from pathlib import Path


def test_workspace_package_includes_folded_result_helpers() -> None:
    repository = Path(__file__).resolve().parents[3]
    package_script = (
        repository / "benchmarks/device_parallel_nss/runpod/package_workspace.sh"
    ).read_text(encoding="utf-8")

    assert '"benchmarks/injection_campaign/folded_results.py"' in package_script
