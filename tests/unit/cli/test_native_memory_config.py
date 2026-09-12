"""Noise realization is scientific provenance; host storage is placement."""

import copy
import hashlib
import json
import tomllib

import pytest
import tomli_w
from pydantic import ValidationError

from jimgw.cli._config import PipelineConfig


def _raw(tmp_path):
    return {
        "seed": 19,
        "data": {
            "type": "injection",
            "detectors": ["H1"],
            "trigger_time": 1_300_000_000.0,
            "duration": 4.0,
            "sampling_frequency": 256.0,
            "injection_parameters": {"M_c": 1.18, "eta": 0.249, "d_L": 200.0},
            "zero_noise": False,
        },
        "waveform": {"approximant": "IMRPhenomD_NRTidalv2", "f_ref": 20.0},
        "prior": {"M_c": {"type": "uniform", "min": 1.17, "max": 1.19}},
        "sampling": {"time_frame": "geocentric", "sky_frame": "geocentric"},
        "likelihood": {"f_min": 20.0, "f_max": 24.0},
        "sampler": {"type": "flowmc"},
        "output": {"dir": str(tmp_path / "output")},
    }


def _legacy_digest(cfg):
    # Independently reconstruct the schema before native-memory/noise fields.
    payload = {
        "contract_schema_version": 1,
        "float_precision": "float64",
        "seed": cfg.seed,
        **{
            field: getattr(cfg, field).model_dump(mode="json")
            for field in (
                "data",
                "waveform",
                "prior",
                "sampling",
                "likelihood",
                "sampler",
            )
        },
    }
    for field in (
        "detector_sites",
        "host_resident_data",
        "host_data_storage",
        "noise_generation",
    ):
        payload["data"].pop(field)
    for field in ("detector_f_min", "detector_f_max"):
        payload["likelihood"].pop(field)
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def test_default_legacy_noise_retains_prior_scientific_contract(tmp_path):
    raw = _raw(tmp_path)
    implicit = PipelineConfig.model_validate(raw)
    assert implicit.data.host_resident_data is False
    assert implicit.data.host_data_storage == "memory"
    assert implicit.data.noise_generation == "legacy"
    old_digest = _legacy_digest(implicit)
    assert implicit.xg_analysis_contract_sha256() == old_digest
    for host, storage in [(False, "memory"), (True, "memory"), (True, "mmap")]:
        explicit = copy.deepcopy(raw)
        explicit["data"].update(
            host_resident_data=host,
            host_data_storage=storage,
            noise_generation="legacy",
        )
        assert (
            PipelineConfig.model_validate(explicit).xg_analysis_contract_sha256()
            == old_digest
        )


def test_indexed_noise_changes_hash_but_storage_placement_does_not(tmp_path):
    raw = _raw(tmp_path)
    legacy = PipelineConfig.model_validate(raw).xg_analysis_contract_sha256()
    raw["data"].update(host_resident_data=True, noise_generation="indexed-v1")
    indexed_memory = PipelineConfig.model_validate(raw)
    assert indexed_memory.xg_analysis_contract_sha256() != legacy
    raw["data"]["host_data_storage"] = "mmap"
    indexed_mapped = PipelineConfig.model_validate(raw)
    assert (
        indexed_mapped.xg_analysis_contract_sha256()
        == indexed_memory.xg_analysis_contract_sha256()
    )
    # Storage mode remains visible in serialized configuration even though it
    # is excluded from scientific hashing; the indexed realization remains explicit.
    resolved = indexed_mapped.model_dump(mode="json", exclude_none=True)
    assert resolved["data"]["noise_generation"] == "indexed-v1"
    assert resolved["data"]["host_data_storage"] == "mmap"
    restored = PipelineConfig.model_validate(tomllib.loads(tomli_w.dumps(resolved)))
    assert (
        restored.xg_analysis_contract_sha256()
        == indexed_mapped.xg_analysis_contract_sha256()
    )
    assert restored.data.noise_generation == "indexed-v1"
    assert restored.data.host_data_storage == "mmap"


@pytest.mark.parametrize(
    "change",
    [{"host_data_storage": "mmap"}, {"noise_generation": "indexed-v1"}],
)
def test_memory_modes_require_host_resident_data(tmp_path, change):
    raw = _raw(tmp_path)
    raw["data"].update(change)
    with pytest.raises(ValidationError, match="require host_resident_data"):
        PipelineConfig.model_validate(raw)


@pytest.mark.parametrize(
    "change",
    [{"noise_generation": "indexed"}, {"host_data_storage": "disk"}],
)
def test_unknown_memory_and_noise_algorithm_names_are_rejected(tmp_path, change):
    raw = _raw(tmp_path)
    raw["data"].update(host_resident_data=True, **change)
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(raw)
