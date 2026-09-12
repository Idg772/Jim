"""Network config and builder wiring, without long native-grid allocations.

Constructor-boundary tests isolate argument forwarding. They do not qualify
the response, compression, or a sampling run.
"""

import copy
import hashlib
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import tomli_w
from pydantic import ValidationError

from jimgw import cli
from jimgw.cli import _data, _likelihood, _waveform, xg_qualification
from jimgw.cli._config import InjectionDataConfig, PipelineConfig
from jimgw.core.single_event import likelihood as core_likelihood
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_CE_A, get_ET_Sardinia

SITES = {"CE": "CE_A_fiducial_2023", "ET": "ET_Sardinia_fiducial_2023"}
LOW = {"CE": 5.0, "ET1": 2.0, "ET2": 3.0, "ET3": 2.0}
HIGH = {"CE": 2048.0, "ET1": 1024.0, "ET2": 1024.0, "ET3": 512.0}
REFERENCE = {
    "M_c": 1.18,
    "eta": 0.249,
    "s1_z": 0.0,
    "s2_z": 0.0,
    "d_L": 20.0,
    "phase_c": 0.0,
    "iota": 0.4,
    "ra": 1.0,
    "dec": 0.2,
    "psi": 0.3,
    "t_c": 0.0,
}


def _raw(tmp_path):
    psd = tmp_path / "sensitivity.npz"
    np.savez(psd, frequencies=[0.0, 2.0, 5.0, 2048.0], values=np.ones(4))
    return {
        "data": {
            "type": "injection",
            "detectors": ["CE", "ET"],
            "detector_sites": dict(SITES),
            "trigger_time": 1_300_000_000.0,
            "duration": 131072.0,
            "sampling_frequency": 4096.0,
            "injection_parameters": REFERENCE.copy(),
            "psd_files": {"CE": str(psd), "ET": str(psd)},
        },
        "waveform": {"approximant": "IMRPhenomD", "f_ref": 20.0},
        "prior": {"M_c": {"type": "uniform", "min": 1.17, "max": 1.19}},
        "sampling": {"time_frame": "CE", "sky_frame": "geocentric"},
        "likelihood": {
            "f_min": 2.0,
            "f_max": 2048.0,
            "detector_f_min": {"CE": 5.0, "ET": 2.0, "ET2": 3.0},
            "detector_f_max": {"ET": 1024.0, "ET3": 512.0},
        },
        "sampler": {"type": "flowmc"},
        "output": {"dir": str(tmp_path / "output")},
    }


def _provided_heterodyne(raw):
    raw["likelihood"]["heterodyne"] = {
        "n_bins": 8,
        "reference_parameters": {"type": "provided", "values": REFERENCE.copy()},
    }


def test_family_and_channel_overrides_resolve_inside_global_envelope(tmp_path):
    cfg = PipelineConfig.model_validate(_raw(tmp_path))
    assert cfg.data.detector_sites == SITES
    assert cfg.likelihood.frequency_bounds(cfg.data.detectors) == (LOW, HIGH)
    assert cfg.likelihood.frequency_bounds(["CE", "ET1", "ET2", "ET3"]) == (LOW, HIGH)
    assert (cfg.likelihood.f_min, cfg.likelihood.f_max) == (2.0, 2048.0)


def test_omitted_overrides_inherit_global_bounds_and_legacy_sites(tmp_path):
    raw = _raw(tmp_path)
    raw["data"].pop("detector_sites")
    raw["likelihood"].pop("detector_f_min")
    raw["likelihood"].pop("detector_f_max")
    cfg = PipelineConfig.model_validate(raw)
    assert cfg.data.detector_sites == {}
    assert cfg.likelihood.frequency_bounds(cfg.data.detectors) == (2.0, 2048.0)
    cfg.likelihood.detector_f_min = {"CE": 5.0}
    assert cfg.likelihood.frequency_bounds(cfg.data.detectors) == (
        {"CE": 5.0, "ET1": 2.0, "ET2": 2.0, "ET3": 2.0},
        2048.0,
    )


@pytest.mark.parametrize(
    "sites",
    [
        {"CE": "CE_A_typo"},
        {"ET": "ET_Sardinia_typo"},
        {"CE": "ET_Sardinia_fiducial_2023"},
        {"ET1": "ET_Sardinia_fiducial_2023"},
        {"H1": "CE_A_fiducial_2023"},
    ],
)
def test_unknown_or_unrequested_site_labels_are_rejected(tmp_path, sites):
    raw = _raw(tmp_path)
    raw["data"]["detector_sites"] = sites
    with pytest.raises(ValidationError, match="site|detector_sites"):
        PipelineConfig.model_validate(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("detector_f_min", {"L1": 5.0}),
        ("detector_f_max", {"ET4": 1000.0}),
        ("detector_f_min", {"CE": 1.0}),
        ("detector_f_min", {"CE": 2048.0}),
        ("detector_f_min", {"ET2": 1200.0}),
        ("detector_f_min", {"ET": float("nan")}),
        ("detector_f_max", {"CE": 4096.0}),
        ("detector_f_max", {"ET": 2.0}),
        ("detector_f_max", {"CE": float("inf")}),
        ("f_min", 2048.0),
        ("f_max", float("inf")),
    ],
)
def test_unknown_outside_empty_and_nonfinite_bands_are_rejected(tmp_path, field, value):
    raw = _raw(tmp_path)
    raw["likelihood"][field] = value
    with pytest.raises(ValidationError, match="Frequency|frequency|finite|f_min"):
        PipelineConfig.model_validate(raw)


@pytest.mark.parametrize("field", ["f_min", "f_max"])
def test_global_envelope_is_required_even_with_detector_overrides(tmp_path, field):
    raw = _raw(tmp_path)
    raw["likelihood"].pop(field)
    with pytest.raises(ValidationError, match="Field required"):
        PipelineConfig.model_validate(raw)


def test_missing_et_sensitivity_is_rejected_by_input_binding(tmp_path):
    raw = _raw(tmp_path)
    raw["data"]["psd_files"].pop("ET")
    cfg = PipelineConfig.model_validate(raw)
    with pytest.raises(ValueError, match="sensitivity file is missing for ET1"):
        cfg.xg_input_files_sha256()


def test_sites_and_cutoff_overrides_are_bound_by_analysis_digest(tmp_path):
    raw = _raw(tmp_path)
    baseline = PipelineConfig.model_validate(raw).xg_analysis_contract_sha256()
    for field, replacement in (("sites", {}), ("cutoff", {"CE": 6.0})):
        changed = copy.deepcopy(raw)
        if field == "sites":
            changed["data"]["detector_sites"] = replacement
        else:
            changed["likelihood"]["detector_f_min"] = replacement
        assert (
            PipelineConfig.model_validate(changed).xg_analysis_contract_sha256()
            != baseline
        )


def test_data_and_qualification_metadata_realize_the_same_selected_sites(
    tmp_path, monkeypatch
):
    cfg = PipelineConfig.model_validate(_raw(tmp_path))
    observed = {}

    def tiny_injection(ifos, data_cfg, waveform, **kwargs):
        del data_cfg, waveform
        observed.update(kwargs)
        for detector in ifos:
            detector.set_data(Data(jnp.zeros(16), 1 / 4096.0, window=jnp.ones(16)))
            detector.set_psd(PowerSpectrum(jnp.ones(9), jnp.arange(9) * 256.0))

    monkeypatch.setattr(_data, "_load_injection", tiny_injection)
    low, high = cfg.likelihood.frequency_bounds(cfg.data.detectors)
    built = _data.build_data(cfg.data, low, high, waveform=SimpleNamespace())
    planned = xg_qualification._configured_ifos(cfg)
    expected = [get_CE_A(), *get_ET_Sardinia()]
    assert [detector.name for detector in built] == ["CE", "ET1", "ET2", "ET3"]
    assert observed["f_min"] == LOW
    assert observed["f_max"] == HIGH
    for actual, metadata, selected in zip(built, planned, expected, strict=True):
        np.testing.assert_array_equal(actual.vertex, selected.vertex)
        np.testing.assert_array_equal(metadata.vertex, selected.vertex)
        np.testing.assert_array_equal(actual.arms, selected.arms)
        np.testing.assert_array_equal(metadata.arms, selected.arms)


class _BuilderReached(Exception):
    pass


def test_cli_forwards_resolved_cutoffs_to_data_builder(tmp_path, monkeypatch):
    path = tmp_path / "network.toml"
    path.write_text(tomli_w.dumps(_raw(tmp_path)))
    observed = {}

    def capture(data_cfg, **kwargs):
        observed.update(kwargs)
        assert isinstance(data_cfg, InjectionDataConfig)
        assert data_cfg.detector_sites == SITES
        raise _BuilderReached

    monkeypatch.setattr(_data, "build_data", capture)
    monkeypatch.setattr(_waveform, "build_waveform", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(cli, "_log_versions", lambda sampler: None)
    with pytest.raises(_BuilderReached):
        cli.run(path)
    assert observed["f_min"] == LOW
    assert observed["f_max"] == HIGH


@pytest.mark.parametrize("kind", ["dense", "heterodyne", "multiband"])
def test_likelihood_builders_receive_resolved_channel_cutoffs(
    tmp_path, monkeypatch, kind
):
    raw = _raw(tmp_path)
    if kind == "heterodyne":
        _provided_heterodyne(raw)
    elif kind == "multiband":
        raw["likelihood"]["multiband"] = {"reference_chirp_mass": 1.18}
    cfg = PipelineConfig.model_validate(raw)
    ifos = xg_qualification._configured_ifos(cfg)
    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        raise _BuilderReached

    constructor = {
        "dense": "TransientLikelihoodFD",
        "heterodyne": "HeterodynedTransientLikelihoodFD",
        "multiband": "MultibandedTransientLikelihoodFD",
    }[kind]
    monkeypatch.setattr(_likelihood, constructor, capture)
    with pytest.raises(_BuilderReached):
        _likelihood.build_likelihood(
            cfg, ifos, SimpleNamespace(), SimpleNamespace(parameter_names=[]), []
        )
    assert observed["f_min"] == LOW
    assert observed["f_max"] == HIGH
    assert observed["detectors"] is ifos


def test_qualification_planner_receives_resolved_channel_cutoffs(tmp_path, monkeypatch):
    raw = _raw(tmp_path)
    _provided_heterodyne(raw)
    cfg = PipelineConfig.model_validate(raw)
    ifos = xg_qualification._configured_ifos(cfg)
    observed = {}

    def capture(detectors, low, high):
        observed.update(detectors=detectors, low=low, high=high)
        raise _BuilderReached

    # This boundary test isolates cutoff forwarding; independent qualification
    # enforcement is covered in test_xg_qualification_candidate.py.
    monkeypatch.setattr(
        xg_qualification, "_validate_qualification_candidate_config", lambda cfg: None
    )
    monkeypatch.setattr(
        xg_qualification, "_verify_realized_qualification_inputs", lambda *a, **k: None
    )
    monkeypatch.setattr(
        core_likelihood, "_set_and_merge_heterodyne_frequency_grids", capture
    )
    with pytest.raises(_BuilderReached):
        xg_qualification.plan_xg_qualification_bin_edges(
            cfg, ifos, SimpleNamespace(parameter_names=[])
        )
    assert observed == {"detectors": ifos, "low": LOW, "high": HIGH}


def test_qualification_candidate_receives_resolved_channel_cutoffs(
    tmp_path, monkeypatch
):
    raw = _raw(tmp_path)
    _provided_heterodyne(raw)
    cfg = PipelineConfig.model_validate(raw)
    ifos = xg_qualification._configured_ifos(cfg)
    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        raise _BuilderReached

    monkeypatch.setattr(
        xg_qualification, "_verify_qualification_candidate_binding", lambda *a: None
    )
    monkeypatch.setattr(
        xg_qualification, "_verify_realized_candidate_inputs", lambda *a: None
    )
    monkeypatch.setattr(core_likelihood, "HeterodynedTransientLikelihoodFD", capture)
    binding = SimpleNamespace(planned_bin_edges_sha256="a" * 64)
    with pytest.raises(_BuilderReached):
        xg_qualification.build_xg_qualification_candidate(
            binding, cfg, ifos, SimpleNamespace(), SimpleNamespace(), []
        )
    assert observed["f_min"] == LOW
    assert observed["f_max"] == HIGH
    assert observed["detectors"] is ifos


def test_empty_network_defaults_preserve_the_pre_network_contract_schema(tmp_path):
    raw = _raw(tmp_path)
    raw["data"].pop("detector_sites")
    raw["likelihood"].pop("detector_f_min")
    raw["likelihood"].pop("detector_f_max")
    cfg = PipelineConfig.model_validate(raw)
    # Construct the payload using exactly the pre-network schema's fields.
    legacy = {
        "contract_schema_version": 1,
        "float_precision": "float64",
        "seed": cfg.seed,
        **{
            name: getattr(cfg, name).model_dump(mode="json")
            for name in (
                "data",
                "waveform",
                "prior",
                "sampling",
                "likelihood",
                "sampler",
            )
        },
    }
    legacy["data"].pop("detector_sites")
    legacy["data"].pop("host_resident_data")  # storage placement, not science
    legacy["data"].pop("host_data_storage")
    assert legacy["data"].pop("noise_generation") == "legacy"
    legacy["likelihood"].pop("detector_f_min")
    legacy["likelihood"].pop("detector_f_max")
    expected = hashlib.sha256(
        json.dumps(
            legacy, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assert cfg.xg_analysis_contract_sha256() == expected
