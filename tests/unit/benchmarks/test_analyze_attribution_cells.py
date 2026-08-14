import json
from pathlib import Path

from benchmarks.injection_campaign.analyze_attribution_cells import analyze

CELLS = (
    "00-legacy",
    "01-stepping-cache",
    "02-replicated-topology",
    "03-fsm-scheduler",
    "04-shared-frequency-grid",
    "05-real-angle-phasor",
    "06-real-inner-product",
    "07-cholesky-factor",
    "08-production",
)


def _write_synthetic_cells(root: Path, times: list[float]) -> None:
    (root / "subset.json").write_text(
        json.dumps({"mapping_factor": 0.99}), encoding="utf-8"
    )
    for cell, base_time in zip(CELLS, times, strict=True):
        for injection_id, event_scale in enumerate((1.0, 1.5, 2.0)):
            result_dir = root / cell / "results" / f"injection-{injection_id:03d}"
            result_dir.mkdir(parents=True)
            summary = {
                "injection_id": injection_id,
                "timing_seconds": {
                    "paper_convention": {
                        "post_jit_sampling_seconds": base_time * event_scale,
                    },
                    "sample_phases": {"ns_loop": base_time * event_scale * 0.8},
                },
            }
            (result_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )


def test_recovers_known_factors(tmp_path):
    times = [100.0, 80.0, 79.0, 50.0, 40.0, 33.0, 31.0, 30.0, 30.0]
    _write_synthetic_cells(tmp_path, times)

    result = analyze(tmp_path)

    chain = result["paper_convention"]["chain_factors"]
    assert abs(chain["01-stepping-cache"]["geomean"] - 100.0 / 80.0) < 1e-9
    assert abs(result["paper_convention"]["F_impl"]["geomean"] - 100.0 / 30.0) < 1e-9


def test_reports_emulation_fidelity_gate(tmp_path):
    _write_synthetic_cells(
        tmp_path,
        [100.0, 80.0, 79.0, 50.0, 40.0, 33.0, 31.0, 30.0, 25.0],
    )

    fidelity = analyze(tmp_path)["paper_convention"]["emulation_fidelity"]

    assert fidelity["within_five_percent"] is False
    assert "does not reproduce production" in fidelity["caveat"]
