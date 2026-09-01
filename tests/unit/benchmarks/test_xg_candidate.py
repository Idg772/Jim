import hashlib
import tomllib
from pathlib import Path

import pytest

from benchmarks.xg.prepare_inputs import (
    CE_PSD_SHA256,
    EARTH_EPHEMERIS_SHA256,
    SOURCE_REVISION,
    SUN_EPHEMERIS_SHA256,
)
from jimgw.cli._config import PipelineConfig
from jimgw.cli._transforms import infer_sample_transforms
from jimgw.core.single_event.detector import get_CE

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "benchmarks" / "xg" / "xg-ce-4096-65536.toml"


def _candidate_raw(tmp_path: Path) -> dict:
    raw = tomllib.loads(CONFIG.read_text())
    raw["data"]["psd_files"]["CE"] = str(tmp_path / "CE_psd.txt")
    Path(raw["data"]["psd_files"]["CE"]).write_text("5 1e-44\n5000 1e-48\n")
    for field, filename in (
        ("orbital_earth_ephemeris_file", "earth.dat.gz"),
        ("orbital_sun_ephemeris_file", "sun.dat.gz"),
    ):
        path = tmp_path / filename
        path.write_bytes(filename.encode())
        raw["likelihood"][field] = str(path)
    return raw


def test_xg_candidate_freezes_requested_sampler_shape(tmp_path: Path) -> None:
    raw = _candidate_raw(tmp_path)

    cfg = PipelineConfig.model_validate(
        raw,
        context={"prepare_xg_qualification": True},
    )

    assert cfg.likelihood.heterodyne is not None
    assert cfg.likelihood.heterodyne.n_bins == 65_536
    assert cfg.sampler.n_live == 4_096
    assert int(cfg.sampler.n_live * cfg.sampler.n_delete_frac) == 512
    assert cfg.sampler.n_devices == 4
    assert cfg.likelihood.phase_marginalization
    assert cfg.likelihood.time_marginalization is None
    assert cfg.likelihood.distance_marginalization is None
    assert cfg.likelihood.orbital_motion_response
    assert cfg.likelihood.orbital_reference_time == cfg.data.trigger_time
    assert cfg.sampler.fold_symmetry is None
    assert cfg.sampling.inclination_coordinate == "cos_iota"
    assert ["cos_iota", "d_L"] in cfg.sampler.blocks

    sample_transforms = infer_sample_transforms(
        frozenset(cfg.prior.root),
        cfg.data.trigger_time,
        [get_CE()],
        cfg.sampling,
        prior_cfg=cfg.prior,
    )
    sample_names = set(cfg.prior.root)
    for transform in sample_transforms:
        inputs, outputs = transform.name_mapping
        sample_names.difference_update(inputs)
        sample_names.update(outputs)

    primary_names = [name for block in cfg.sampler.blocks for name in block]
    assert len(primary_names) == len(set(primary_names))
    assert set(primary_names) == sample_names
    assert all(set(block) <= sample_names for block in cfg.sampler.bridge_blocks)


def test_xg_candidate_binds_orbital_epoch_and_ephemerides(tmp_path: Path) -> None:
    raw = _candidate_raw(tmp_path)
    cfg = PipelineConfig.model_validate(
        raw,
        context={"prepare_xg_qualification": True},
    )

    hashes = cfg.xg_input_files_sha256()
    assert hashes["ephemeris:earth"] == hashlib.sha256(b"earth.dat.gz").hexdigest()
    assert hashes["ephemeris:sun"] == hashlib.sha256(b"sun.dat.gz").hexdigest()

    raw["likelihood"]["orbital_reference_time"] += 1.0
    with pytest.raises(
        ValueError,
        match="orbital_reference_time must equal data.trigger_time",
    ):
        PipelineConfig.model_validate(
            raw,
            context={"prepare_xg_qualification": True},
        )


def test_public_ce_psd_provenance_is_pinned() -> None:
    assert SOURCE_REVISION == "5c15b707e1b9c90d0ef2f36d4b378124f31074a8"
    assert len(CE_PSD_SHA256) == hashlib.sha256().digest_size * 2
    assert len(EARTH_EPHEMERIS_SHA256) == hashlib.sha256().digest_size * 2
    assert len(SUN_EPHEMERIS_SHA256) == hashlib.sha256().digest_size * 2
