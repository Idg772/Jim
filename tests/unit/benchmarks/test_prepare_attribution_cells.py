import json
from pathlib import Path

from benchmarks.injection_campaign.common import load_manifest, read_catalogue
from benchmarks.injection_campaign.prepare_attribution_cells import prepare_cells

TEMPLATES = Path("campaign-results/pp-speedup-attribution-cells-20260813")
SOURCE = Path("campaign-results/corrected-anchor-fsm-d4-m1-pp-20260810")
SUBSET_PATH = Path("campaign-results/pp-residual-cells-20260814/subset.json")
SUBSET = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))["injection_ids"]


def test_prepare_cells_yields_nine_valid_campaigns(tmp_path):
    dirs = prepare_cells(TEMPLATES, SOURCE, SUBSET, tmp_path)
    assert [directory.name for directory in dirs] == [
        "00-legacy",
        "01-stepping-cache",
        "02-replicated-topology",
        "03-fsm-scheduler",
        "04-shared-frequency-grid",
        "05-real-angle-phasor",
        "06-real-inner-product",
        "07-cholesky-factor",
        "08-production",
    ]
    source_rows = {
        row["injection_id"]: row for row in read_catalogue(SOURCE / "catalogue.csv")
    }
    for cell_dir in dirs:
        manifest = load_manifest(cell_dir)
        rows = read_catalogue(cell_dir / "catalogue.csv")
        assert len(rows) == 10
        for new_id, row in enumerate(rows):
            assert row["injection_id"] == new_id
            source = source_rows[SUBSET[new_id]]
            for field in row:
                if field != "injection_id":
                    assert row[field] == source[field]
        assert manifest["selection"]["source_injection_ids"] == SUBSET


def test_production_cell_has_no_monkeypatch(tmp_path):
    directories = prepare_cells(TEMPLATES, SOURCE, SUBSET, tmp_path)
    production = json.loads(
        (directories[-1] / "manifest.json").read_text(encoding="utf-8")
    )["config"]

    assert "sampler_ablation_variant" not in production
    assert production["sampler_scheduler"] == "fsm"
    assert production["likelihood_implementation"] == "optimized"
    assert production["likelihood_optimization_axes"] == {
        "shared_frequency_grid": True,
        "detector_phasor": True,
        "real_inner_product": True,
    }


def test_cells_share_the_reviewed_template_implementation_pin(tmp_path):
    directories = prepare_cells(TEMPLATES, SOURCE, SUBSET, tmp_path)
    template_pin = json.loads(
        (TEMPLATES / "00-legacy" / "manifest.json").read_text(encoding="utf-8")
    )["implementation_diagnostic"]
    expected = {
        name: template_pin[name]
        for name in (
            "implementation_label",
            "implementation_revision",
            "implementation_tree_sha256",
        )
    }

    for directory in directories:
        pin = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))[
            "implementation_diagnostic"
        ]
        assert {name: pin[name] for name in expected} == expected


def test_cells_can_be_repinned_to_the_final_workspace_package(tmp_path):
    revision = "a" * 40
    tree_sha256 = "b" * 64
    directories = prepare_cells(
        TEMPLATES,
        SOURCE,
        SUBSET,
        tmp_path,
        implementation_revision=revision,
        implementation_tree_sha256=tree_sha256,
    )

    for directory in directories:
        pin = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))[
            "implementation_diagnostic"
        ]
        assert pin["implementation_label"] == "candidate"
        assert pin["implementation_revision"] == revision
        assert pin["implementation_tree_sha256"] == tree_sha256
