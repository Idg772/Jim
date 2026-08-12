from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign import plot_pp as plot_module


@pytest.mark.parametrize(
    ("combined_pvalue", "expected_passes"),
    [(0.05, False), (0.0500000001, True)],
    ids=("equal-to-alpha-fails", "above-alpha-passes"),
)
def test_remediation_requires_fisher_p_above_alpha_over_all_parameters_including_q(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    combined_pvalue: float,
    expected_passes: bool,
) -> None:
    campaign = tmp_path / "campaign"
    manifest = {
        "n_injections": common.PAPER_PP_RECOVERIES,
        "config_sha256": "frozen-config",
        "config": copy.deepcopy(common.DEFAULT_CONFIG),
    }
    rank_markers = {
        name: (index + 1) / (len(common.PARAMETERS) + 1)
        for index, name in enumerate(common.PARAMETERS)
    }
    rows = [
        {
            "injection_id": injection_id,
            **rank_markers,
            "_legacy_phase_gauge_corrected_parameters": [],
        }
        for injection_id in range(common.PAPER_PP_RECOVERIES)
    ]
    parameter_pvalues = {
        name: (0.01 if name == "q" else 0.5 + index / 100.0)
        for index, name in enumerate(common.PARAMETERS)
    }
    marker_to_name = {marker: name for name, marker in rank_markers.items()}
    combined_inputs: list[list[float]] = []

    monkeypatch.setattr(
        plot_module,
        "load_rank_rows",
        lambda *_args, **_kwargs: (manifest, rows),
    )

    def fake_kstest(ranks: np.ndarray, *_args: object, **_kwargs: object) -> object:
        name = marker_to_name[float(ranks[0])]
        return SimpleNamespace(statistic=0.1, pvalue=parameter_pvalues[name])

    def fake_combine_pvalues(values: list[float], *, method: str) -> object:
        assert method == "fisher"
        combined_inputs.append(list(values))
        return SimpleNamespace(statistic=42.0, pvalue=combined_pvalue)

    def fake_draw_combined(output_path: Path, **_kwargs: object) -> None:
        output_path.write_bytes(b"test image")

    monkeypatch.setattr(plot_module, "kstest", fake_kstest)
    monkeypatch.setattr(plot_module, "combine_pvalues", fake_combine_pvalues)
    monkeypatch.setattr(plot_module, "_draw_paper_style_combined", fake_draw_combined)
    monkeypatch.setattr(plot_module, "file_sha256", lambda _path: "0" * 64)

    report = plot_module.aggregate_and_plot(campaign)

    expected_pvalues = [parameter_pvalues[name] for name in common.PARAMETERS]
    assert combined_inputs == [expected_pvalues, expected_pvalues]
    assessment = report["remediation_assessment"]
    assert assessment["criterion"] == (
        "Fisher-combined exact KS p-value > 0.05 over all 15 sampled "
        "parameters, including q"
    )
    assert assessment["alpha"] == 0.05
    assert assessment["excluded_parameters"] == []
    assert assessment["parameters"] == list(common.PARAMETERS)
    assert "q" in assessment["parameters"]
    assert assessment["combined_test"]["number_of_pvalues"] == 15
    assert assessment["combined_test"]["degrees_of_freedom"] == 30
    assert assessment["combined_test"]["pvalue"] == combined_pvalue
    assert assessment["eligible"] is True
    assert assessment["passes"] is expected_passes
