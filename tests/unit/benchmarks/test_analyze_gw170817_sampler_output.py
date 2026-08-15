"""Tests for the GW170817 weighted sampler-output analysis."""

import heapq
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.device_parallel_nss import analyze_gw170817_sampler_output as analysis


def _synthetic_nested_run(
    *,
    random_seed: int,
    replacement_scale: float,
    n_live: int = 64,
    n_replacements: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    death = list(rng.exponential(size=n_live))
    birth = [-np.inf] * n_live
    active = [(value, index) for index, value in enumerate(death)]
    heapq.heapify(active)
    for _ in range(n_replacements):
        contour, _ = heapq.heappop(active)
        replacement_death = contour + replacement_scale * rng.exponential()
        index = len(death)
        death.append(replacement_death)
        birth.append(contour)
        heapq.heappush(active, (replacement_death, index))
    return np.asarray(death), np.asarray(birth)


def test_insertion_rank_ks_accepts_a_perfect_nested_run() -> None:
    death, birth = _synthetic_nested_run(random_seed=12, replacement_scale=1.0)

    result = analysis.insertion_rank_diagnostic(
        death,
        birth,
        window_size=1000,
        random_seed=34,
    )

    assert result.global_p_value > 0.01
    assert result.worst_window_p_value > 0.01


def test_insertion_rank_ks_rejects_a_biased_nested_run() -> None:
    death, birth = _synthetic_nested_run(random_seed=12, replacement_scale=0.01)

    result = analysis.insertion_rank_diagnostic(
        death,
        birth,
        window_size=1000,
        random_seed=34,
    )

    assert result.global_p_value < 1e-20
    assert result.worst_window_p_value < 1e-20


def _write_report_triplet(
    root: Path,
    *,
    schemes: list[str],
    blocks: list[list[str]],
) -> tuple[list[Path], list[Path]]:
    nested_paths: list[Path] = []
    report_paths: list[Path] = []
    for seed, scheme in enumerate(schemes):
        nested_path = root / f"nested-seed{seed}.npz"
        nested_path.write_bytes(f"nested-{seed}".encode())
        report_path = root / f"report-seed{seed}.json"
        report_path.write_text(
            json.dumps(
                {
                    "config": {
                        "seed": seed,
                        "workload": "paper-15d",
                        "sampled_dimensions": 15,
                        "phase_marginalization": True,
                        "distance_marginalization": False,
                        "blocking_scheme": scheme,
                        "blocks": blocks,
                    },
                    "data": {"sha256": "frozen-bundle"},
                    "results": {
                        "nested_artifact": {
                            "weighting": analysis.WEIGHTING,
                            "sha256": analysis._sha256(nested_path),
                        }
                    },
                }
            )
        )
        nested_paths.append(nested_path)
        report_paths.append(report_path)
    return report_paths, nested_paths


def test_reports_reject_mixed_blocking_schemes(tmp_path: Path) -> None:
    report_paths, nested_paths = _write_report_triplet(
        tmp_path,
        schemes=["all-slow", "all-slow", "paper"],
        blocks=[list(block) for block in analysis.ALL_SLOW_BLOCKS],
    )

    with pytest.raises(ValueError, match="different blocking schemes"):
        analysis._reports(report_paths, nested_paths)


def test_reports_reject_wrong_all_slow_blocks(tmp_path: Path) -> None:
    report_paths, nested_paths = _write_report_triplet(
        tmp_path,
        schemes=["all-slow"] * 3,
        blocks=[list(block) for block in analysis.PAPER_BLOCKS],
    )

    with pytest.raises(ValueError, match="wrong blocks for all-slow"):
        analysis._reports(report_paths, nested_paths)
