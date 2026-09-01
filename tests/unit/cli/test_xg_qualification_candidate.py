from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from jimgw.cli._config import (
    CLIInjectionRefParams,
    CLIProvidedRefParams,
    InjectionDataConfig,
    PipelineConfig,
)
from jimgw.cli._data import build_data
from jimgw.cli._likelihood import build_likelihood
from jimgw.cli._prior import build_prior
from jimgw.cli._transforms import infer_likelihood_transforms, to_likelihood_space
from jimgw.cli._waveform import build_waveform
from jimgw.cli.xg_qualification import (
    bind_xg_qualification_candidate,
    build_xg_qualification_candidate,
    plan_xg_qualification_bin_edges,
)
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.marginalization_config import PhaseMargConfig

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures"

REFERENCE_PARAMETERS = {
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


def _raw_config(tmp_path: Path, *, copy_inputs: bool = False) -> dict:
    strain_path = FIXTURES_DIR / "GW150914_strain_H1.npz"
    psd_path = FIXTURES_DIR / "GW150914_psd_H1.npz"
    if copy_inputs:
        copied_strain = tmp_path / "strain.npz"
        copied_psd = tmp_path / "psd.npz"
        shutil.copyfile(strain_path, copied_strain)
        shutil.copyfile(psd_path, copied_psd)
        strain_path = copied_strain
        psd_path = copied_psd
    return {
        "seed": 17,
        "data": {
            "type": "file",
            "detectors": ["H1"],
            "trigger_time": 1_126_259_462.4,
            "strain_files": {"H1": str(strain_path)},
            "psd_files": {"H1": str(psd_path)},
        },
        "waveform": {"approximant": "IMRPhenomD", "f_ref": 20.0},
        "prior": {
            "M_c": {"type": "uniform", "min": 25.0, "max": 35.0},
            "eta": {"type": "uniform", "min": 0.20, "max": 0.25},
            "s1_z": {"type": "uniform", "min": -0.1, "max": 0.1},
            "s2_z": {"type": "uniform", "min": -0.1, "max": 0.1},
            "d_L": {"type": "uniform", "min": 100.0, "max": 1_000.0},
            "phase_c": {"type": "uniform", "min": 0.0, "max": 2.0 * math.pi},
            "iota": {"type": "sine"},
            "ra": {"type": "uniform", "min": 0.0, "max": 2.0 * math.pi},
            "dec": {"type": "cosine"},
            "psi": {"type": "uniform", "min": 0.0, "max": math.pi},
            "t_c": {"type": "uniform", "min": -0.05, "max": 0.05},
        },
        "sampling": {"time_frame": "geocentric", "sky_frame": "geocentric"},
        "likelihood": {
            "f_min": 20.0,
            "f_max": 128.0,
            "time_dependent_response": True,
            "finite_arm_response": True,
            "heterodyne": {
                "n_bins": 16,
                "reference_parameters": {
                    "type": "provided",
                    "values": REFERENCE_PARAMETERS,
                },
            },
        },
        "sampler": {"type": "flowmc"},
        "output": {"dir": str(tmp_path / "output")},
    }


def _preflight(raw: dict) -> PipelineConfig:
    return PipelineConfig.model_validate(
        raw,
        context={"prepare_xg_qualification": True},
    )


def _injection_raw_config(tmp_path: Path) -> dict:
    raw = _raw_config(tmp_path)
    injection_parameters = dict(REFERENCE_PARAMETERS)
    injection_parameters.pop("eta")
    injection_parameters["q"] = 0.8
    raw["data"] = {
        "type": "injection",
        "detectors": ["H1"],
        "trigger_time": 1_126_259_462.4,
        "duration": 4.0,
        "sampling_frequency": 512.0,
        "injection_parameters": injection_parameters,
        "zero_noise": True,
        "psd_files": {"H1": str(FIXTURES_DIR / "GW150914_psd_H1.npz")},
        "waveform_chunk_size": 31,
    }
    raw["likelihood"]["heterodyne"]["reference_parameters"] = {"type": "injection"}
    return raw


def _static_baseline(cfg: PipelineConfig) -> HeterodynedTransientLikelihoodFD:
    source_waveform = build_waveform(cfg.waveform)
    static_ifos = build_data(
        cfg.data,
        f_min=cfg.likelihood.f_min,
        f_max=cfg.likelihood.f_max,
        waveform=source_waveform,
    )
    heterodyne = cfg.likelihood.heterodyne
    assert heterodyne is not None
    reference_cfg = heterodyne.reference_parameters
    if isinstance(reference_cfg, CLIProvidedRefParams):
        reference_parameters = dict(reference_cfg.values)
    elif isinstance(reference_cfg, CLIInjectionRefParams):
        assert isinstance(cfg.data, InjectionDataConfig)
        reference_parameters = to_likelihood_space(
            cfg.data.injection_parameters,
            waveform_f_ref=cfg.waveform.f_ref,
            trigger_time=cfg.data.trigger_time,
            ifos=static_ifos,
            time_frame=cfg.sampling.time_frame,
        )
    else:  # pragma: no cover - qualification preflight rejects this case
        raise TypeError("test baseline requires a fixed reference")
    return HeterodynedTransientLikelihoodFD(
        detectors=static_ifos,
        waveform=source_waveform,
        fixed_parameters=cfg.likelihood.fixed_parameters or None,
        f_min=cfg.likelihood.f_min,
        f_max=cfg.likelihood.f_max,
        trigger_time=cfg.data.trigger_time,
        n_bins=heterodyne.n_bins,
        reference_parameters=reference_parameters,
        phase_marginalization=(
            PhaseMargConfig() if cfg.likelihood.phase_marginalization else None
        ),
        reference_chunk_size=heterodyne.reference_chunk_size,
    )


def _realized_components(cfg: PipelineConfig, input_hashes: dict[str, str]):
    waveform = DominantModeTimeCachedWaveform(build_waveform(cfg.waveform))
    ifos = build_data(
        cfg.data,
        f_min=cfg.likelihood.f_min,
        f_max=cfg.likelihood.f_max,
        waveform=waveform,
        time_frame=cfg.sampling.time_frame,
        time_dependent_response=cfg.likelihood.time_dependent_response,
        finite_arm_response=cfg.likelihood.finite_arm_response,
        seed=cfg.seed,
        input_provenance_sha256=input_hashes,
    )
    prior = build_prior(cfg.prior)
    transforms = infer_likelihood_transforms(
        frozenset(prior.parameter_names),
        cfg.data.trigger_time,
        ifos,
        cfg.sampling,
        cfg.waveform.f_ref,
        phase_marginalization=cfg.likelihood.phase_marginalization,
    )
    return ifos, waveform, prior, transforms


def test_qualification_candidate_builds_real_xg_likelihood_but_production_stays_locked(
    tmp_path: Path,
) -> None:
    cfg = _preflight(_raw_config(tmp_path))
    ifos, waveform, prior, transforms = _realized_components(
        cfg,
        cfg.xg_input_files_sha256(),
    )
    planned_digest = plan_xg_qualification_bin_edges(cfg, ifos, waveform)
    static_baseline = _static_baseline(cfg)
    assert planned_digest == static_baseline.bin_edges_sha256
    binding = bind_xg_qualification_candidate(
        cfg,
        planned_digest,
    )

    with pytest.raises(ValueError, match="qualification receipt"):
        build_likelihood(cfg, ifos, waveform, prior, transforms)

    likelihood = build_xg_qualification_candidate(
        binding,
        cfg,
        ifos,
        waveform,
        prior,
        transforms,
    )
    assert likelihood.bin_edges_sha256 == planned_digest
    assert bool(jnp.isfinite(likelihood.evaluate(REFERENCE_PARAMETERS)))
    assert cfg.verified_xg_manifest is None


def test_bin_planner_matches_static_baseline_after_support_trimming(
    tmp_path: Path,
) -> None:
    raw = _raw_config(tmp_path)
    raw["likelihood"]["f_max"] = 2_048.0
    raw["likelihood"]["heterodyne"]["n_bins"] = 64
    cfg = _preflight(raw)
    ifos, waveform, _, _ = _realized_components(
        cfg,
        cfg.xg_input_files_sha256(),
    )

    planned_digest = plan_xg_qualification_bin_edges(cfg, ifos, waveform)
    static_baseline = _static_baseline(cfg)

    assert static_baseline.n_bins < 64
    assert planned_digest == static_baseline.bin_edges_sha256


def test_bin_planner_resolves_injection_reference_in_likelihood_space(
    tmp_path: Path,
) -> None:
    cfg = _preflight(_injection_raw_config(tmp_path))
    ifos, waveform, prior, transforms = _realized_components(
        cfg,
        cfg.xg_input_files_sha256(),
    )

    planned_digest = plan_xg_qualification_bin_edges(cfg, ifos, waveform)
    static_baseline = _static_baseline(cfg)
    binding = bind_xg_qualification_candidate(cfg, planned_digest)
    candidate = build_xg_qualification_candidate(
        binding,
        cfg,
        ifos,
        waveform,
        prior,
        transforms,
    )

    assert planned_digest == static_baseline.bin_edges_sha256
    assert candidate.bin_edges_sha256 == planned_digest


def test_candidate_rejects_config_mutation_after_binding(tmp_path: Path) -> None:
    cfg = _preflight(_raw_config(tmp_path))
    binding = bind_xg_qualification_candidate(cfg, "b" * 64)
    cfg.data.trigger_time += 1.0

    with pytest.raises(ValueError, match="analysis contract"):
        build_xg_qualification_candidate(
            binding,
            cfg,
            [],
            object(),  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            [],
        )


def test_candidate_rejects_input_mutation_after_binding(tmp_path: Path) -> None:
    raw = _raw_config(tmp_path, copy_inputs=True)
    cfg = _preflight(raw)
    binding = bind_xg_qualification_candidate(cfg, "b" * 64)
    psd_path = Path(raw["data"]["psd_files"]["H1"])
    with psd_path.open("ab") as output:
        output.write(b"changed")

    with pytest.raises(ValueError, match="input files"):
        build_xg_qualification_candidate(
            binding,
            cfg,
            [],
            object(),  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            [],
        )


def test_candidate_rejects_realized_geometry_mutation(tmp_path: Path) -> None:
    cfg = _preflight(_raw_config(tmp_path))
    binding = bind_xg_qualification_candidate(cfg, "b" * 64)
    ifos, waveform, prior, transforms = _realized_components(
        cfg,
        binding.input_files_sha256,
    )
    assert ifos[0].arm_length_m is not None
    ifos[0].arm_length_m += 1.0

    with pytest.raises(ValueError, match="realized detector metadata"):
        build_xg_qualification_candidate(
            binding,
            cfg,
            ifos,
            waveform,
            prior,
            transforms,
        )


def test_candidate_rejects_realized_bin_edge_mismatch(tmp_path: Path) -> None:
    cfg = _preflight(_raw_config(tmp_path))
    binding = bind_xg_qualification_candidate(cfg, "0" * 64)
    ifos, waveform, prior, transforms = _realized_components(
        cfg,
        binding.input_files_sha256,
    )

    with pytest.raises(ValueError, match="bin edges do not match"):
        build_xg_qualification_candidate(
            binding,
            cfg,
            ifos,
            waveform,
            prior,
            transforms,
        )


def test_candidate_binding_requires_explicit_preflight_context(tmp_path: Path) -> None:
    raw = _raw_config(tmp_path)
    raw["likelihood"].pop("heterodyne")
    cfg = PipelineConfig.model_validate(raw)

    with pytest.raises(ValueError, match="prepare_xg_qualification"):
        bind_xg_qualification_candidate(cfg, "b" * 64)


def test_candidate_binding_requires_x64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _preflight(_raw_config(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(config=SimpleNamespace(jax_enable_x64=False)),
    )

    with pytest.raises(ValueError, match="64-bit precision"):
        bind_xg_qualification_candidate(cfg, "b" * 64)
