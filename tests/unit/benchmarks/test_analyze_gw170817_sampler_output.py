"""Tests for the GW170817 weighted sampler-output analysis."""

import heapq

import numpy as np

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
