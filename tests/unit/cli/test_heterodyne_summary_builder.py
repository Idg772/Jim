"""Native-summary backend configuration and constructor-boundary wiring.

These tests allocate only a tiny PSD fixture and detector metadata. They do
not build a native frequency grid or qualify either numerical backend.
"""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from jimgw.cli import _likelihood, xg_qualification
from jimgw.cli._config import CLIHeterodynedConfig, PipelineConfig
from jimgw.core.single_event import likelihood as core_likelihood

REFERENCE = {
    "M_c": 30.0,
    "eta": 0.24,
    "s1_z": 0.01,
    "s2_z": -0.02,
    "d_L": 400.0,
    "phase_c": 0.1,
    "iota": 0.4,
    "ra": 1.375,
    "dec": -1.2108,
    "psi": 0.2,
    "t_c": 0.0,
}


def _raw(tmp_path, *, backend=None, xg=False):
    psd = tmp_path / "sensitivity.npz"
    np.savez(psd, frequencies=[0.0, 20.0, 128.0], values=np.ones(3))
    heterodyne = {
        "n_bins": 8,
        "reference_chunk_size": 31,
        "interpolation_order": 8,
        "phasor_moment_order": 16,
        "phasor_time_anchors": [-0.02, 0.0, 0.02],
        "reference_parameters": {"type": "provided", "values": REFERENCE.copy()},
    }
    if backend is not None:
        heterodyne["summary_backend"] = backend
    return {
        "data": {
            "type": "injection",
            "detectors": ["H1"],
            "trigger_time": 1_126_259_462.4,
            "duration": 4.0,
            "sampling_frequency": 256.0,
            "injection_parameters": REFERENCE.copy(),
            "zero_noise": False,
            "psd_files": {"H1": str(psd)},
        },
        "waveform": {"approximant": "IMRPhenomD", "f_ref": 20.0},
        "prior": {"M_c": {"type": "uniform", "min": 25.0, "max": 35.0}},
        "sampling": {"time_frame": "geocentric", "sky_frame": "geocentric"},
        "likelihood": {
            "f_min": 20.0,
            "f_max": 128.0,
            "time_dependent_response": xg,
            "finite_arm_response": xg,
            "heterodyne": heterodyne,
        },
        "sampler": {"type": "flowmc"},
        "output": {"dir": str(tmp_path / "output")},
    }


@pytest.mark.parametrize("options", [{}, {"interpolation_order": 8}])
def test_legacy_numpy_backend_remains_default_for_classic_and_polynomial(options):
    assert CLIHeterodynedConfig(**options).summary_backend == "numpy"


@pytest.mark.parametrize("backend", ["gpu", "auto", "JAX", None, True])
def test_unknown_summary_backends_are_rejected(backend):
    with pytest.raises(ValidationError, match="summary_backend"):
        CLIHeterodynedConfig(summary_backend=backend)


@pytest.mark.parametrize("order", [1, 8])
def test_jax_requires_nonzero_phasor_moment_order(order):
    with pytest.raises(ValidationError, match="native polynomial phasor moments"):
        CLIHeterodynedConfig(summary_backend="jax", interpolation_order=order)


def test_jax_rejects_zero_noise_quadrature():
    with pytest.raises(ValidationError, match="cannot be combined"):
        CLIHeterodynedConfig(
            summary_backend="jax",
            interpolation_order=8,
            phasor_moment_order=16,
            zero_noise_quadrature={},
        )


@pytest.mark.parametrize("anchors", [None, [-0.02, 0.0, 0.02]])
def test_jax_accepts_native_polynomial_moments_with_or_without_anchor_bank(anchors):
    cfg = CLIHeterodynedConfig(
        summary_backend="jax",
        interpolation_order=8,
        phasor_moment_order=16,
        phasor_time_anchors=anchors,
    )
    assert cfg.summary_backend == "jax"
    assert cfg.phasor_time_anchors == anchors


def test_numpy_still_accepts_the_existing_zero_noise_summary_path():
    cfg = CLIHeterodynedConfig(
        interpolation_order=8,
        phasor_moment_order=16,
        zero_noise_quadrature={},
    )
    assert cfg.summary_backend == "numpy"
    assert cfg.zero_noise_quadrature is not None


def test_default_numpy_preserves_the_pre_backend_contract_schema(tmp_path):
    raw = _raw(tmp_path)
    cfg = PipelineConfig.model_validate(raw)
    # Independently reconstruct the prior normalized schema, excluding the new
    # field and existing non-scientific qualification-manifest locations.
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
    payload["data"].pop("detector_sites")
    # Storage placement is not part of the scientific plan.
    payload["data"].pop("host_resident_data")
    payload["data"].pop("host_data_storage")
    assert payload["data"].pop("noise_generation") == "legacy"
    for field in ("detector_f_min", "detector_f_max"):
        payload["likelihood"].pop(field)
    for field in (
        "summary_backend",
        "qualification_manifest",
        "qualification_manifest_sha256",
    ):
        payload["likelihood"]["heterodyne"].pop(field)
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assert cfg.xg_analysis_contract_sha256() == digest
    explicit_numpy = copy.deepcopy(raw)
    explicit_numpy["likelihood"]["heterodyne"]["summary_backend"] = "numpy"
    assert (
        PipelineConfig.model_validate(explicit_numpy).xg_analysis_contract_sha256()
        == digest
    )
    explicit_jax = copy.deepcopy(raw)
    explicit_jax["likelihood"]["heterodyne"]["summary_backend"] = "jax"
    assert (
        PipelineConfig.model_validate(explicit_jax).xg_analysis_contract_sha256()
        != digest
    )


@pytest.mark.parametrize("before,after", [("numpy", "jax"), ("jax", "numpy")])
def test_changing_backend_invalidates_an_existing_candidate_binding(
    tmp_path, before, after
):
    cfg = PipelineConfig.model_validate(
        _raw(tmp_path, backend=before, xg=True),
        context={"prepare_xg_qualification": True},
    )
    binding = xg_qualification.bind_xg_qualification_candidate(cfg, "a" * 64)
    xg_qualification._verify_qualification_candidate_binding(binding, cfg)
    cfg.likelihood.heterodyne.summary_backend = after
    with pytest.raises(ValueError, match="changed after binding: analysis contract"):
        xg_qualification.build_xg_qualification_candidate(
            binding, cfg, [], SimpleNamespace(), SimpleNamespace(), []
        )


class _ConstructorReached(Exception):
    pass


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("builder", ["cli", "qualification"])
def test_backend_and_summary_shape_controls_reach_both_builders(
    tmp_path, monkeypatch, backend, builder
):
    qualification = builder == "qualification"
    cfg = PipelineConfig.model_validate(
        _raw(tmp_path, backend=backend, xg=qualification),
        context={"prepare_xg_qualification": qualification},
    )
    ifos = xg_qualification._configured_ifos(cfg)
    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        raise _ConstructorReached

    if qualification:
        binding = xg_qualification.bind_xg_qualification_candidate(cfg, "a" * 64)
        # This test stops at the constructor, so detector data need not be
        # realized. Real config/input/metadata binding checks remain enabled.
        monkeypatch.setattr(
            xg_qualification, "_verify_realized_candidate_inputs", lambda *args: None
        )
        monkeypatch.setattr(
            core_likelihood, "HeterodynedTransientLikelihoodFD", capture
        )
        with pytest.raises(_ConstructorReached):
            xg_qualification.build_xg_qualification_candidate(
                binding, cfg, ifos, SimpleNamespace(), SimpleNamespace(), []
            )
    else:
        monkeypatch.setattr(_likelihood, "HeterodynedTransientLikelihoodFD", capture)
        with pytest.raises(_ConstructorReached):
            _likelihood.build_likelihood(
                cfg, ifos, SimpleNamespace(), SimpleNamespace(parameter_names=[]), []
            )
    assert observed["summary_backend"] == backend
    assert observed["reference_chunk_size"] == 31
    assert observed["interpolation_order"] == 8
    assert observed["phasor_moment_order"] == 16
    assert observed["phasor_time_anchors"] == [-0.02, 0.0, 0.02]
    assert observed["zero_noise_summary"] is None
    assert observed["detectors"] is ifos
