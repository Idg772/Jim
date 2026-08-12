import copy
from pathlib import Path

import numpy as np
import pytest

from benchmarks.injection_campaign import common
from benchmarks.injection_campaign.prepare_campaign import prepare_campaign
from benchmarks.injection_campaign.prepare_time_anchor_diagnostic import (
    DIAGNOSTIC_TIME_ANCHOR,
    SOURCE_TIME_ANCHOR,
    prepare_time_anchor_diagnostic,
)
from benchmarks.injection_campaign.runpod.upload_and_run import _parse_args


def _write_noise_curves(directory: Path) -> None:
    directory.mkdir()
    frequencies = np.asarray([1.0, 10.0, 20.0, 100.0, 1024.0, 4096.0])
    ligo_psd = 1.0e-46 * (1.0 + (100.0 / frequencies) ** 2)
    virgo_psd = 2.0e-46 * (1.0 + (100.0 / frequencies) ** 2)
    np.savetxt(
        directory / "aLIGO_O4_high_asd.txt",
        np.column_stack([frequencies, np.sqrt(ligo_psd)]),
    )
    np.savetxt(
        directory / "AdV_psd.txt",
        np.column_stack([frequencies, virgo_psd]),
    )


def _source_campaign(tmp_path: Path) -> Path:
    curves = tmp_path / "curves"
    _write_noise_curves(curves)
    source = tmp_path / "source"
    prepare_campaign(
        source,
        n_injections=4,
        catalogue_size=4,
        seed=1234,
        noise_curves_dir=curves,
        config_overrides={"carrier_time_anchor": SOURCE_TIME_ANCHOR},
    )
    manifest = common.load_manifest(source)
    for injection_id in (1, 3):
        result = common.result_dir(source, injection_id)
        result.mkdir(parents=True)
        posterior = result / "posterior.npz"
        common.atomic_savez_compressed(
            posterior,
            {
                "q": np.asarray([0.6]),
                "log_weights": np.asarray([0.0]),
            },
        )
        common.atomic_write_json(
            result / "summary.json",
            {
                "config_sha256": manifest["config_sha256"],
                "injection_id": injection_id,
                "posterior": {"sha256": common.file_sha256(posterior)},
                "ranks": {
                    "q": 0.1,
                    "ra": 0.2,
                    "dec": 0.3,
                    "t_c": 0.4,
                },
            },
        )
    return source


def test_time_anchor_preparer_freezes_an_exact_current_code_ab_pair(
    tmp_path: Path,
) -> None:
    source = _source_campaign(tmp_path)
    source_manifest = common.load_manifest(source)
    source_rows = common.read_catalogue(source / "catalogue.csv")
    outputs = {}

    for anchor in (SOURCE_TIME_ANCHOR, DIAGNOSTIC_TIME_ANCHOR):
        output = tmp_path / anchor
        outputs[anchor] = prepare_time_anchor_diagnostic(
            source,
            output,
            implementation_revision="a" * 40,
            implementation_tree_sha256="b" * 64,
            source_ids=(3, 1),
            carrier_time_anchor=anchor,
        )
        rows = common.read_catalogue(output / "catalogue.csv")
        assert [row["injection_id"] for row in rows] == [0, 1]
        assert [row["noise_seed"] for row in rows] == [
            source_rows[3]["noise_seed"],
            source_rows[1]["noise_seed"],
        ]
        assert [row["sampler_seed"] for row in rows] == [
            source_rows[3]["sampler_seed"],
            source_rows[1]["sampler_seed"],
        ]
        assert outputs[anchor]["config"]["carrier_time_anchor"] == anchor
        assert outputs[anchor]["config"]["n_devices"] == 4
        assert outputs[anchor]["config"]["num_gibbs_sweeps"] == 1
        assert (
            outputs[anchor]["config"]["blocks"] == source_manifest["config"]["blocks"]
        )
        assert outputs[anchor]["reproduction_scope"]["pp_calibration_eligible"] is False
        common.load_manifest(output)

    local_config = copy.deepcopy(outputs[SOURCE_TIME_ANCHOR]["config"])
    author_config = copy.deepcopy(outputs[DIAGNOSTIC_TIME_ANCHOR]["config"])
    for config in (local_config, author_config):
        config.pop("campaign")
        config.pop("paper_configuration")
        config["timing"].pop("selected_events")
    local_anchor = local_config.pop("carrier_time_anchor")
    author_anchor = author_config.pop("carrier_time_anchor")
    assert local_anchor == SOURCE_TIME_ANCHOR
    assert author_anchor == DIAGNOSTIC_TIME_ANCHOR
    assert local_config == author_config


def test_upload_run_label_is_validated_and_defaults_to_legacy_paths() -> None:
    assert _parse_args(["pod", "workspace.tar.gz"]).run_label is None
    assert (
        _parse_args(
            ["pod", "workspace.tar.gz", "--run-label", "author-anchor"]
        ).run_label
        == "author-anchor"
    )


@pytest.mark.parametrize("value", ["Author", "author/control", "a" * 33])
def test_upload_run_label_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(SystemExit):
        _parse_args(["pod", "workspace.tar.gz", "--run-label", value])
