import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.device_parallel_nss import summarize_gw170817_diagnostics as summary


def test_slot_cost_models_independent_device_lockstep() -> None:
    # One outer iteration, four chains split two-per-device, one slice.
    counts = np.asarray([[[1], [3], [2], [2]]])

    cost = summary._slot_cost(counts, n_devices=2)

    assert cost["fsm_slots"] == 8
    assert cost["lockstep_slots"] == 10
    assert cost["lockstep_to_fsm_ratio"] == pytest.approx(1.25)


def test_combined_slot_cost_treats_expansion_and_shrink_as_separate_loops() -> None:
    expansions = np.asarray([[[1], [4], [2], [2]]])
    shrink = np.asarray([[[5], [1], [3], [3]]])

    cost = summary._combine_slot_cost(expansions, shrink, n_devices=2)

    assert cost["combined"]["fsm_slots"] == 21
    assert cost["combined"]["lockstep_slots"] == 28
    assert cost["combined"]["lockstep_to_fsm_ratio"] == pytest.approx(28 / 21)


def test_slice_summary_reads_harness_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "slices.npz"
    np.savez(
        artifact,
        num_expansions=np.asarray([[1, 2], [3, 4], [2, 2], [1, 5]]),
        num_shrink=np.ones((4, 2), dtype=int),
        chain_index=np.asarray([0, 1, 0, 1]),
        slice_block_index=np.asarray([0, 1]),
        slice_requires_rebuild=np.asarray([True, False]),
        slice_block_parameter_names=np.asarray(["x", "y"]),
        n_devices=np.asarray(1),
    )
    report = {"results": {"per_slice_update_info": {"path": str(artifact)}}}

    result = summary._slice_summary(report)

    assert result["shape"] == [2, 2, 2]
    assert set(result["by_cache_class"]) == {"waveform_rebuild", "cache_hit"}
    assert result["by_block"]["0"]["block_parameters"] == ["x"]


def test_full_report_loader_rejects_mixed_workloads(tmp_path: Path) -> None:
    report = {
        "benchmark": "gw170817-full-swig-4gpu",
        "config": {"workload": "aligned-11d", "sampled_dimensions": 11},
        "results": {"early_stopped_for_cache_probe": False},
    }
    (tmp_path / "report.json").write_text(json.dumps(report))

    with pytest.raises(RuntimeError, match="uses workload"):
        summary._load_full_reports(tmp_path, "paper-15d")


def test_microbenchmark_summary_rejects_wrong_workload() -> None:
    report = {
        "workload": {"name": "aligned-11d", "sampled_dimensions": 11},
        "results": {},
    }

    with pytest.raises(RuntimeError, match="likelihood-lane workload"):
        summary._microbenchmark_summary(report, "paper-15d")


def test_persistent_cache_summary_rejects_wrong_workload(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"workload": "aligned-11d"}))

    with pytest.raises(RuntimeError, match="persistent-cache workload"):
        summary._persistent_cache_summary(path, "paper-15d")
