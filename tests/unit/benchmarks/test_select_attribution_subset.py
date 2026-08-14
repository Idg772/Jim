import json
from pathlib import Path

from benchmarks.injection_campaign.select_attribution_subset import select_subset

CAMPAIGN = Path("campaign-results/corrected-anchor-fsm-d4-m1-pp-20260810")


def test_subset_is_deterministic_deciles():
    result = select_subset(CAMPAIGN, n=10)
    assert len(result["injection_ids"]) == 10
    assert result["injection_ids"] == sorted(set(result["injection_ids"]))
    # Median of decile representatives must sit within 10% of the
    # population median (44.664 s) by construction.
    assert abs(result["mapping_factor"] - 1.0) < 0.10
    # Re-running returns the identical selection.
    assert select_subset(CAMPAIGN, n=10) == result


def test_subset_uses_lower_middle_event_from_each_decile():
    ranked_summaries = []
    for summary_path in CAMPAIGN.glob("results/injection-*/summary.json"):
        summary = json.loads(summary_path.read_text())
        seconds = summary["timing_seconds"]["paper_convention"][
            "post_jit_sampling_seconds"
        ]
        ranked_summaries.append((float(seconds), int(summary["injection_id"])))
    ranked_summaries.sort()

    expected_ids = sorted(injection_id for _, injection_id in ranked_summaries[4::10])

    assert select_subset(CAMPAIGN, n=10)["injection_ids"] == expected_ids


def test_shape_comparison_uses_paper_reference():
    shape = select_subset(CAMPAIGN, n=10)["shape_comparison"]
    assert shape["paper"] == {"min": 258.0, "median": 306.0, "max": 383.0}
    assert 0 < shape["ours"]["min"] < shape["ours"]["median"] < shape["ours"]["max"]
