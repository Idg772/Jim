"""Build qualification candidates and verify frozen XG evidence bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import typer
from pydantic import ValidationError

from jimgw.cli._config import (
    CLIInjectionRefParams,
    CLIProvidedRefParams,
    InjectionDataConfig,
    PipelineConfig,
    XGClockValidationReceipt,
    XGCompressionValidationReceipt,
    XGIndependentResponseReceipt,
    XGOrbitalValidationReceipt,
    XGQualificationManifest,
    XGValidationCorpusReceipt,
    _file_sha256,
    xg_implementation_sha256,
    xg_runtime_environment_sha256,
    xg_source_revision,
)
from jimgw.cli._likelihood import detector_metadata_sha256
from jimgw.core.single_event.detector import get_detector_preset

if TYPE_CHECKING:
    from ripplegw.interfaces import Waveform

    from jimgw.core.prior import CombinePrior
    from jimgw.core.single_event.detector import GroundBased2G
    from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD
    from jimgw.core.transforms import NtoMTransform

app = typer.Typer(
    name="jim-xg-qualify",
    add_completion=False,
    help="Assemble a verified manifest from completed XG qualification evidence.",
)

_RECEIPT_FILES = {
    "validation_corpus": (
        "validation-corpus.json",
        XGValidationCorpusReceipt,
    ),
    "clock_validation": (
        "clock-validation.json",
        XGClockValidationReceipt,
    ),
    "independent_response": (
        "independent-response.json",
        XGIndependentResponseReceipt,
    ),
    "orbital_validation": (
        "orbital-validation.json",
        XGOrbitalValidationReceipt,
    ),
    "compression_validation": (
        "compression-validation.json",
        XGCompressionValidationReceipt,
    ),
}

_XG_QUALIFICATION_BINDING_AUTHORITY = object()


@dataclass(frozen=True)
class XGQualificationCandidateBinding:
    """Immutable preflight contract for one qualification-only likelihood."""

    analysis_contract_sha256: str
    detector_metadata_sha256: str
    planned_bin_edges_sha256: str
    _input_files: tuple[tuple[str, str], ...]
    _authority: object = field(repr=False, compare=False)

    @property
    def input_files_sha256(self) -> dict[str, str]:
        """Return a defensive copy suitable for ``build_data`` provenance."""

        return dict(self._input_files)


def _validate_qualification_candidate_config(cfg: PipelineConfig) -> None:
    import jax

    if not cfg.is_xg_qualification_preflight:
        raise ValueError(
            "XG qualification candidates require a PipelineConfig validated with "
            "prepare_xg_qualification"
        )
    if cfg.verified_xg_manifest is not None:
        raise ValueError("a production-qualified pipeline is not a candidate preflight")
    if not jax.config.jax_enable_x64:
        raise ValueError("XG qualification candidates require JAX 64-bit precision")

    likelihood = cfg.likelihood
    heterodyne = likelihood.heterodyne
    if heterodyne is None:
        raise ValueError("XG qualification candidates require heterodyne likelihood")
    if not likelihood.time_dependent_response or not likelihood.finite_arm_response:
        raise ValueError(
            "XG qualification candidates require dynamic and finite-arm response"
        )
    if heterodyne.n_bins is None:
        raise ValueError("XG qualification candidates require an explicit n_bins")
    if not isinstance(
        heterodyne.reference_parameters,
        (CLIProvidedRefParams, CLIInjectionRefParams),
    ):
        raise TypeError(
            "XG qualification candidates require fixed provided or injection "
            "reference parameters"
        )
    if (
        heterodyne.qualification_manifest is not None
        or heterodyne.qualification_manifest_sha256 is not None
    ):
        raise ValueError(
            "XG qualification candidates cannot carry a production manifest"
        )


def _qualification_reference_parameters(
    cfg: PipelineConfig,
    ifos: list[GroundBased2G],
) -> dict[str, float]:
    """Resolve the configured fixed reference into likelihood space."""

    from jimgw.cli._transforms import to_likelihood_space

    heterodyne = cfg.likelihood.heterodyne
    assert heterodyne is not None
    reference_cfg = heterodyne.reference_parameters
    if isinstance(reference_cfg, CLIProvidedRefParams):
        return dict(reference_cfg.values)
    if isinstance(reference_cfg, CLIInjectionRefParams):
        if not isinstance(cfg.data, InjectionDataConfig):
            raise TypeError("injection reference parameters require injection data")
        return to_likelihood_space(
            cfg.data.injection_parameters,
            waveform_f_ref=cfg.waveform.f_ref,
            trigger_time=cfg.data.trigger_time,
            ifos=ifos,
            time_frame=cfg.sampling.time_frame,
        )
    raise TypeError("qualification candidate reference parameters are not fixed")


def _verify_realized_qualification_inputs(
    cfg: PipelineConfig,
    ifos: list[GroundBased2G],
    waveform: Waveform,
    *,
    expected_detector_metadata_sha256: str,
    expected_input_files_sha256: dict[str, str],
) -> None:
    """Verify realized data, geometry, response mode, and waveform identity."""

    from jimgw.cli._likelihood import _validate_xg_detector_inputs
    from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform

    if detector_metadata_sha256(ifos) != expected_detector_metadata_sha256:
        raise ValueError(
            "realized detector metadata does not match the qualification contract"
        )
    _validate_xg_detector_inputs(ifos, expected_input_files_sha256)
    for ifo in ifos:
        if bool(ifo.time_dependent_response) != cfg.likelihood.time_dependent_response:
            raise ValueError("realized dynamic response flag changed after binding")
        if bool(ifo.finite_arm_response) != cfg.likelihood.finite_arm_response:
            raise ValueError("realized finite-arm response flag changed after binding")

    if not isinstance(waveform, DominantModeTimeCachedWaveform):
        raise TypeError(
            "qualification candidate requires DominantModeTimeCachedWaveform"
        )
    source_waveform = waveform.source
    if type(source_waveform).__name__ != cfg.waveform.approximant:
        raise TypeError("realized waveform does not match the qualification candidate")
    realized_f_ref = getattr(waveform, "f_ref", None)
    if realized_f_ref is None or float(realized_f_ref) != cfg.waveform.f_ref:
        raise ValueError("realized waveform reference frequency changed after binding")


def plan_xg_qualification_bin_edges(
    cfg: PipelineConfig,
    ifos: list[GroundBased2G],
    reference_waveform: Waveform,
) -> str:
    """Plan the exact retained XG edge digest without building a likelihood.

    The helper evaluates only the bounded reference-support scan and the
    configured endpoint waveforms. It creates no detector summaries, candidate
    capability, receipt, or production manifest. The qualification candidate
    independently realizes and checks this digest during construction.
    """

    from jimgw.core.single_event.likelihood import (
        HeterodynedTransientLikelihoodFD,
        _set_and_merge_heterodyne_frequency_grids,
    )
    from jimgw.core.single_event.time_utils import (
        greenwich_mean_sidereal_time as compute_gmst,
    )
    from jimgw.core.single_event.utils import apply_fixed_parameters

    _validate_qualification_candidate_config(cfg)
    likelihood_cfg = cfg.likelihood
    heterodyne = likelihood_cfg.heterodyne
    assert heterodyne is not None
    assert heterodyne.n_bins is not None

    analysis_contract_sha256 = cfg.xg_analysis_contract_sha256()
    input_files_sha256 = cfg.xg_input_files_sha256()
    metadata_sha256 = detector_metadata_sha256(_configured_ifos(cfg))
    _verify_realized_qualification_inputs(
        cfg,
        ifos,
        reference_waveform,
        expected_detector_metadata_sha256=metadata_sha256,
        expected_input_files_sha256=input_files_sha256,
    )

    reference_parameters = _qualification_reference_parameters(cfg, ifos)
    apply_fixed_parameters(reference_parameters, likelihood_cfg.fixed_parameters)
    reference_parameters["trigger_time"] = cfg.data.trigger_time
    reference_parameters["gmst"] = compute_gmst(cfg.data.trigger_time)
    if likelihood_cfg.phase_marginalization:
        reference_parameters.setdefault("phase_c", 0.0)
    if likelihood_cfg.time_marginalization is not None:
        reference_parameters.setdefault("t_c", 0.0)
    required_parameters = set(reference_waveform.parameter_names) | {
        "ra",
        "dec",
        "psi",
        "t_c",
    }
    missing_parameters = required_parameters - set(reference_parameters)
    if missing_parameters:
        raise ValueError(
            "heterodyne reference_parameters are incomplete; missing "
            f"{sorted(missing_parameters)}"
        )

    frequencies, _, _ = _set_and_merge_heterodyne_frequency_grids(
        ifos,
        *likelihood_cfg.frequency_bounds([ifo.name for ifo in ifos]),
    )
    frequency_edges, _, _ = (
        HeterodynedTransientLikelihoodFD._plan_fixed_reference_bin_edges(
            frequencies,
            heterodyne.n_bins,
            reference_waveform,
            reference_parameters,
            heterodyne.reference_chunk_size,
            **(
                {"frequency_bin_edges": heterodyne.frequency_bin_edges}
                if heterodyne.frequency_bin_edges is not None
                else {}
            ),
        )
    )
    from jimgw.cli._likelihood import build_zero_noise_summary

    summary_builder = build_zero_noise_summary(cfg, ifos, reference_waveform)
    planned_digest = HeterodynedTransientLikelihoodFD._bin_edges_sha256(
        frequency_edges,
        interpolation_order=heterodyne.interpolation_order,
        phasor_moment_order=heterodyne.phasor_moment_order,
        phasor_time_anchors=heterodyne.phasor_time_anchors,
        phasor_approximation=heterodyne.phasor_approximation,
        summary_builder_sha256=(
            summary_builder.contract_sha256 if summary_builder is not None else None
        ),
        reference_projection=heterodyne.reference_projection,
    )

    if cfg.xg_analysis_contract_sha256() != analysis_contract_sha256:
        raise ValueError("the XG analysis contract changed while planning bins")
    if cfg.xg_input_files_sha256() != input_files_sha256:
        raise ValueError("the XG inputs changed while planning bins")
    if detector_metadata_sha256(_configured_ifos(cfg)) != metadata_sha256:
        raise ValueError("XG detector metadata changed while planning bins")
    _verify_realized_qualification_inputs(
        cfg,
        ifos,
        reference_waveform,
        expected_detector_metadata_sha256=metadata_sha256,
        expected_input_files_sha256=input_files_sha256,
    )
    return planned_digest


def bind_xg_qualification_candidate(
    cfg: PipelineConfig,
    planned_bin_edges_sha256: str,
) -> XGQualificationCandidateBinding:
    """Freeze the exact contract a qualification candidate must realize.

    The supplied edge digest must come from the independent, immutable bin
    planner used by the compression campaign.  This function does not plan
    bins, create evidence, or issue a production likelihood capability.
    """

    _validate_qualification_candidate_config(cfg)
    if len(planned_bin_edges_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in planned_bin_edges_sha256
    ):
        raise ValueError("planned XG bin edges must use a SHA-256 digest")

    analysis_contract_sha256 = cfg.xg_analysis_contract_sha256()
    input_files_sha256 = cfg.xg_input_files_sha256()
    metadata_sha256 = detector_metadata_sha256(_configured_ifos(cfg))
    if analysis_contract_sha256 != cfg.xg_analysis_contract_sha256():
        raise ValueError("the XG analysis contract changed while binding the candidate")
    if input_files_sha256 != cfg.xg_input_files_sha256():
        raise ValueError("the XG inputs changed while binding the candidate")
    if metadata_sha256 != detector_metadata_sha256(_configured_ifos(cfg)):
        raise ValueError("XG detector metadata changed while binding the candidate")

    return XGQualificationCandidateBinding(
        analysis_contract_sha256=analysis_contract_sha256,
        detector_metadata_sha256=metadata_sha256,
        planned_bin_edges_sha256=planned_bin_edges_sha256,
        _input_files=tuple(sorted(input_files_sha256.items())),
        _authority=_XG_QUALIFICATION_BINDING_AUTHORITY,
    )


def _verify_qualification_candidate_binding(
    binding: XGQualificationCandidateBinding,
    cfg: PipelineConfig,
) -> dict[str, str]:
    if (
        not isinstance(binding, XGQualificationCandidateBinding)
        or binding._authority is not _XG_QUALIFICATION_BINDING_AUTHORITY
    ):
        raise TypeError(
            "qualification binding must come from bind_xg_qualification_candidate"
        )
    _validate_qualification_candidate_config(cfg)
    current_inputs = cfg.xg_input_files_sha256()
    mismatches = []
    if cfg.xg_analysis_contract_sha256() != binding.analysis_contract_sha256:
        mismatches.append("analysis contract")
    if current_inputs != binding.input_files_sha256:
        mismatches.append("input files")
    if (
        detector_metadata_sha256(_configured_ifos(cfg))
        != binding.detector_metadata_sha256
    ):
        mismatches.append("configured detector metadata")
    if mismatches:
        raise ValueError(
            "XG qualification candidate changed after binding: " + ", ".join(mismatches)
        )
    return current_inputs


def _verify_realized_candidate_inputs(
    binding: XGQualificationCandidateBinding,
    cfg: PipelineConfig,
    ifos: list[GroundBased2G],
    waveform: Waveform,
) -> None:
    _verify_realized_qualification_inputs(
        cfg,
        ifos,
        waveform,
        expected_detector_metadata_sha256=binding.detector_metadata_sha256,
        expected_input_files_sha256=binding.input_files_sha256,
    )


def build_xg_qualification_candidate(
    binding: XGQualificationCandidateBinding,
    cfg: PipelineConfig,
    ifos: list[GroundBased2G],
    waveform: Waveform,
    prior: CombinePrior,
    likelihood_transforms: list[NtoMTransform],
    *,
    native_probe_parameters: Sequence[Mapping[str, Any]] | None = None,
) -> HeterodynedTransientLikelihoodFD:
    """Construct the actual response likelihood for deterministic measurement.

    This qualification-only route uses a distinct internal capability.  It
    cannot be supplied through ``build_likelihood`` or the normal ``jim-run``
    pipeline, and its output does not constitute a production receipt.
    Optional native probes are accumulated during the same data traversal as
    the reference moments; they do not change those moments or qualify the grid.
    """

    from jimgw.cli._likelihood import (
        _validate_xg_detector_inputs,
        build_zero_noise_summary,
    )
    from jimgw.core.single_event.likelihood import (
        _XG_QUALIFICATION_PLAN_AUTHORITY,
        HeterodynedTransientLikelihoodFD,
        _QualificationXGPlan,
    )
    from jimgw.core.single_event.marginalization_config import (
        HeterodyneTimeMargConfig,
        PhaseMargConfig,
    )

    _verify_qualification_candidate_binding(binding, cfg)
    _verify_realized_candidate_inputs(binding, cfg, ifos, waveform)
    likelihood_cfg = cfg.likelihood
    heterodyne = likelihood_cfg.heterodyne
    assert heterodyne is not None
    assert heterodyne.n_bins is not None

    reference_parameters = _qualification_reference_parameters(cfg, ifos)

    phase_marginalization = (
        PhaseMargConfig() if likelihood_cfg.phase_marginalization else None
    )
    time_cfg = likelihood_cfg.time_marginalization
    time_marginalization = (
        HeterodyneTimeMargConfig(
            tc_range=time_cfg.tc_range,
            upsample_factor=time_cfg.upsample_factor,
            phasor_block_size=time_cfg.phasor_block_size,
            freeze_response=time_cfg.freeze_response,
            timing_sigma_s=time_cfg.timing_sigma_s,
            samples_per_timing_sigma=time_cfg.samples_per_timing_sigma,
            normalization=time_cfg.normalization,
        )
        if time_cfg is not None
        else None
    )

    f_min, f_max = likelihood_cfg.frequency_bounds([ifo.name for ifo in ifos])
    likelihood = HeterodynedTransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        fixed_parameters=(
            likelihood_cfg.fixed_parameters if likelihood_cfg.fixed_parameters else None
        ),
        f_min=f_min,
        f_max=f_max,
        trigger_time=cfg.data.trigger_time,
        n_bins=heterodyne.n_bins,
        reference_parameters=reference_parameters,
        prior=prior,
        likelihood_transforms=likelihood_transforms,
        phase_marginalization=phase_marginalization,
        time_marginalization=time_marginalization,
        reference_chunk_size=heterodyne.reference_chunk_size,
        summary_backend=heterodyne.summary_backend,
        xg_evaluation_mode=heterodyne.xg_evaluation_mode,
        node_frequency_prefix=heterodyne.node_frequency_prefix,
        native_probe_parameters=native_probe_parameters,
        interpolation_order=heterodyne.interpolation_order,
        phasor_moment_order=heterodyne.phasor_moment_order,
        phasor_time_anchors=heterodyne.phasor_time_anchors,
        phasor_approximation=heterodyne.phasor_approximation,
        zero_noise_summary=build_zero_noise_summary(cfg, ifos, waveform),
        reference_projection=heterodyne.reference_projection,
        frequency_bin_edges=heterodyne.frequency_bin_edges,
        xg_plan=_QualificationXGPlan(
            binding.planned_bin_edges_sha256,
            _XG_QUALIFICATION_PLAN_AUTHORITY,
        ),
    )

    _verify_qualification_candidate_binding(binding, cfg)
    _verify_realized_candidate_inputs(binding, cfg, ifos, waveform)
    _validate_xg_detector_inputs(ifos, binding.input_files_sha256)
    if likelihood.bin_edges_sha256 != binding.planned_bin_edges_sha256:
        raise ValueError("realized XG bin edges do not match the qualification binding")
    return likelihood


def build_selected_xg_qualification_candidate(
    cfg: PipelineConfig,
    ifos: list[GroundBased2G],
    waveform: Waveform,
    prior: CombinePrior,
    likelihood_transforms: list[NtoMTransform],
    *,
    native_probe_parameters: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[HeterodynedTransientLikelihoodFD, PipelineConfig, dict[str, Any] | None]:
    """Build one native bank and freeze an empirical qualification grid.

    An explicit edge layout is rebuilt as declared. Otherwise, configured bin
    selection coarsens one fine native bank and returns its exact resolved
    configuration. The input configuration is never mutated. Native probes,
    when supplied, use likelihood-space parameters just as in
    ``build_xg_qualification_candidate``.

    All planning, input and capability checks remain those of the qualification
    candidate route. Empirical selection does not issue a production receipt or
    change eligibility for ``jim-run``; independent native validation is still
    required before a caller samples the candidate.
    """
    import jax

    _validate_qualification_candidate_config(cfg)
    heterodyne = cfg.likelihood.heterodyne
    assert heterodyne is not None
    automatic = (
        heterodyne.bin_selection is not None and heterodyne.frequency_bin_edges is None
    )
    construction_cfg = cfg.model_copy(deep=True)
    if automatic:
        assert heterodyne.bin_selection is not None
        construction_cfg.likelihood.heterodyne = heterodyne.model_copy(
            update={
                "n_bins": heterodyne.bin_selection.reference_bins,
                "epsilon": None,
            }
        )
    digest = plan_xg_qualification_bin_edges(construction_cfg, ifos, waveform)
    binding = bind_xg_qualification_candidate(construction_cfg, digest)
    candidate = build_xg_qualification_candidate(
        binding,
        construction_cfg,
        ifos,
        waveform,
        prior,
        likelihood_transforms,
        native_probe_parameters=native_probe_parameters,
    )
    jax.block_until_ready((candidate.summary_data, candidate.phasor_data_moments))
    if not automatic:
        return candidate, construction_cfg, None

    from jimgw.cli._xg_binning import select_network_binning

    # The selector verifies and rebinds the selected edges against the original
    # requested contract, not the temporary fine-grid construction config.
    return select_network_binning(cfg, candidate, ifos, waveform)


@app.callback()
def main() -> None:
    """Manage frozen XG qualification evidence."""


def _load_receipts(bundle_dir: Path) -> dict[str, object]:
    receipts = {}
    for receipt_name, (filename, model) in _RECEIPT_FILES.items():
        path = bundle_dir / filename
        try:
            receipts[receipt_name] = model.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot load qualification receipt {path}") from error
    return receipts


def _configured_ifos(cfg: PipelineConfig):
    presets = (
        get_detector_preset(site_overrides=cfg.data.detector_sites)
        if cfg.data.detector_sites
        else get_detector_preset()
    )
    ifos = []
    for detector_name in cfg.data.detectors:
        preset = presets[detector_name]
        if isinstance(preset, list):
            ifos.extend(preset)
        else:
            ifos.append(preset)
    for ifo in ifos:
        ifo.time_dependent_response = cfg.likelihood.time_dependent_response
        ifo.finite_arm_response = cfg.likelihood.finite_arm_response
        ifo.configure_orbital_motion_response(
            enabled=cfg.likelihood.orbital_motion_response,
            reference_time=cfg.likelihood.orbital_reference_time,
            validity_s=cfg.likelihood.orbital_validity_s,
            acceleration_over_c=cfg.likelihood.orbital_acceleration_over_c,
            jerk_over_c=cfg.likelihood.orbital_jerk_over_c,
        )
    return ifos


def build_xg_qualification_manifest(
    cfg: PipelineConfig,
    bundle_dir: Path,
) -> XGQualificationManifest:
    """Build a manifest from completed receipts without running science jobs."""

    import jax

    if not jax.config.jax_enable_x64:
        raise ValueError("XG qualification requires JAX 64-bit precision")

    heterodyne = cfg.likelihood.heterodyne
    if heterodyne is None or not (
        cfg.likelihood.time_dependent_response
        or cfg.likelihood.finite_arm_response
        or cfg.likelihood.orbital_motion_response
    ):
        raise ValueError("XG qualification requires an XG heterodyne config")
    if heterodyne.n_bins is None:
        raise ValueError("XG qualification requires an explicit n_bins")

    receipts = _load_receipts(bundle_dir)
    corpus = receipts["validation_corpus"]
    compression = receipts["compression_validation"]
    assert isinstance(corpus, XGValidationCorpusReceipt)
    assert isinstance(compression, XGCompressionValidationReceipt)
    time_config = cfg.likelihood.time_marginalization
    frozen_response_budget = (
        0.01
        if cfg.likelihood.time_dependent_response and time_config is not None
        else None
    )
    timing_sigma_s = time_config.timing_sigma_s if time_config is not None else None

    receipt_fields = {}
    for receipt_name, (filename, _) in _RECEIPT_FILES.items():
        receipt_fields[f"{receipt_name}_file"] = filename
        receipt_fields[f"{receipt_name}_sha256"] = _file_sha256(bundle_dir / filename)

    return XGQualificationManifest(
        schema_version=1,
        source_revision=xg_source_revision(),
        implementation_sha256=xg_implementation_sha256(),
        runtime_environment_sha256=xg_runtime_environment_sha256(),
        analysis_contract_sha256=cfg.xg_analysis_contract_sha256(),
        input_files_sha256=cfg.xg_input_files_sha256(),
        detector_metadata_sha256=detector_metadata_sha256(_configured_ifos(cfg)),
        bin_edges_sha256=compression.bin_edges_sha256,
        waveform_approximant=cfg.waveform.approximant,
        detectors=cfg.data.detectors,
        f_min=cfg.likelihood.f_min,
        f_max=cfg.likelihood.f_max,
        n_bins=heterodyne.n_bins,
        time_dependent_response=cfg.likelihood.time_dependent_response,
        finite_arm_response=cfg.likelihood.finite_arm_response,
        orbital_motion_response=cfg.likelihood.orbital_motion_response,
        max_network_snr=corpus.max_network_snr,
        max_component_delta_log_l=0.01,
        max_combined_delta_log_l=0.05,
        max_frozen_response_delta_log_l=frozen_response_budget,
        timing_sigma_s=timing_sigma_s,
        float_precision="float64",
        **receipt_fields,
    )


def _manifest_bytes(manifest: XGQualificationManifest) -> bytes:
    return (
        json.dumps(
            manifest.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()


@app.command()
def assemble(
    config: Annotated[Path, typer.Argument(help="Predeclared XG TOML config.")],
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing the five completed receipts."),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Manifest output path."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing manifest."),
    ] = False,
) -> None:
    """Assemble a manifest and verify the complete evidence bundle."""

    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        with config.open("rb") as config_file:
            raw = tomllib.load(config_file)
        cfg = PipelineConfig.model_validate(
            raw,
            context={"prepare_xg_qualification": True},
        )
        resolved_bundle = bundle_dir.resolve()
        output_path = (output or (resolved_bundle / "xg-qualification.json")).resolve()
        if output_path.parent != resolved_bundle:
            raise ValueError("the qualification manifest must stay inside bundle_dir")
        if output_path.exists() and not overwrite:
            raise ValueError(f"qualification manifest already exists: {output_path}")

        manifest = build_xg_qualification_manifest(cfg, resolved_bundle)
        candidate_path = resolved_bundle / (
            f".{output_path.name}.{uuid.uuid4().hex}.candidate"
        )
        candidate_bytes = _manifest_bytes(manifest)
        candidate_path.write_bytes(candidate_bytes)
        try:
            validation_raw = copy.deepcopy(raw)
            heterodyne = validation_raw["likelihood"]["heterodyne"]
            heterodyne["qualification_manifest"] = str(candidate_path)
            heterodyne["qualification_manifest_sha256"] = hashlib.sha256(
                candidate_bytes
            ).hexdigest()
            PipelineConfig.model_validate(validation_raw)
            candidate_path.replace(output_path)
        except Exception:
            candidate_path.unlink(missing_ok=True)
            raise
    except (OSError, ValueError, ValidationError, tomllib.TOMLDecodeError) as error:
        typer.echo(f"XG qualification assembly failed: {error}", err=True)
        raise typer.Exit(code=2) from error

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    typer.echo(f"Wrote {output_path}")
    typer.echo(f"qualification_manifest_sha256 = {digest}")


if __name__ == "__main__":
    app()
