"""Assemble and verify a frozen XG qualification evidence bundle."""

import copy
import hashlib
import json
import tomllib
import uuid
from pathlib import Path
from typing import Annotated, Optional

import typer
from pydantic import ValidationError

from jimgw.cli._config import (
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
    presets = get_detector_preset()
    ifos = []
    for detector_name in cfg.data.detectors:
        preset = presets[detector_name]
        if isinstance(preset, list):
            ifos.extend(preset)
        else:
            ifos.append(preset)
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
        cfg.likelihood.time_dependent_response or cfg.likelihood.finite_arm_response
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
