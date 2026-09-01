import hashlib
import tomllib
from pathlib import Path

from benchmarks.xg.prepare_inputs import CE_PSD_SHA256, SOURCE_REVISION
from jimgw.cli._config import PipelineConfig
from jimgw.cli._transforms import infer_sample_transforms
from jimgw.core.single_event.detector import get_CE

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "benchmarks" / "xg" / "xg-ce-4096-65536.toml"


def test_xg_candidate_freezes_requested_sampler_shape(tmp_path: Path) -> None:
    raw = tomllib.loads(CONFIG.read_text())
    raw["data"]["psd_files"]["CE"] = str(tmp_path / "CE_psd.txt")
    Path(raw["data"]["psd_files"]["CE"]).write_text("5 1e-44\n5000 1e-48\n")

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


def test_public_ce_psd_provenance_is_pinned() -> None:
    assert SOURCE_REVISION == "5c15b707e1b9c90d0ef2f36d4b378124f31074a8"
    assert len(CE_PSD_SHA256) == hashlib.sha256().digest_size * 2
