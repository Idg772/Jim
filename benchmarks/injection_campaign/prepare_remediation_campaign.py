"""Freeze a paired blocking-remediation campaign.

Legacy schemes retain the complete leading-100 M=1 source and change only its
block partition and labels. The netsky arm instead consumes one complete batch
from the frozen D=4/M=2 fast-ridge P-P baseline and records its additional
bridge, covariance, and quotient-fold configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.injection_campaign.common import (
    NETSKY_BRIDGE_BLOCKS,
    NETSKY_SCHEME,
    PAPER_PP_RECOVERIES,
    SCHEMA_VERSION,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_manifest,
    publication_eligible,
    read_catalogue,
    refresh_status,
    result_dir,
)
from benchmarks.injection_campaign.merge_staged_results import _validate_result
from benchmarks.injection_campaign.prepare_blocking_diagnostic import (
    SCHEMES as M1_SCHEMES,
)
from benchmarks.injection_campaign.prepare_blocking_diagnostic import blocks_for_scheme

SCHEMES = (*M1_SCHEMES, NETSKY_SCHEME)
DEFAULT_SCHEME = "all-slow-time"
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_FAST_RIDGE_SOURCE_BLOCKS = [
    ["M_c", "q", "lambda_1", "lambda_2"],
    ["s1_mag", "s1_theta", "s1_phi"],
    ["s2_mag", "s2_theta", "s2_phi"],
    ["zenith", "azimuth"],
    ["psi"],
    ["cos_iota", "d_hat"],
]
_FAST_RIDGE_TIME_MARGINALIZATION = {
    "tc_range_seconds": [-0.03, 0.03],
    "upsample_factor": 1,
}
_NETSKY_SAMPLING_PARAMETERIZATION = {
    "distance_transform": ("log_d_hat=log(d_L/(M_c^(5/6)*R_net(ra,dec,psi,iota)))"),
    "sky_transform": (
        "(ra,dec) -> detector-frame (cos_zenith,azimuth) at trigger time"
    ),
    "joint_block": ["azimuth", "cos_iota", "psi", "log_d_hat"],
    "physical_output_parameters": ["ra", "dec", "iota", "psi", "d_L"],
    "sampling_space_parameters": [
        "cos_zenith",
        "azimuth",
        "cos_iota",
        "psi",
        "log_d_hat",
    ],
    "match": "Roulet-style network-frame blocking with Jim's R_net coordinate",
    "fold_model_limitation": (
        "aligned/weak-precession proxy: cos_iota is not cos(theta_JN) and the "
        "full precessing phi_JL handedness term is omitted; keep folding "
        "default-off outside the validated low-spin campaign"
    ),
}


def _is_expected_netsky_source(config: dict[str, Any]) -> bool:
    """Return whether config is the frozen D=4/M=2 fast-ridge source arm."""

    prior = config.get("prior")
    return (
        config.get("blocking_scheme") == "fast-ridge"
        and config.get("blocks") == _FAST_RIDGE_SOURCE_BLOCKS
        and config.get("direction_mode") == "covariance"
        and config.get("adaptive_slice_widths") is True
        and config.get("bracket_mode") == "shrink-only"
        and type(config.get("num_de_jumps")) is int
        and config.get("num_de_jumps") == 0
        and config.get("phase_marginalization") is True
        and config.get("time_marginalization") == _FAST_RIDGE_TIME_MARGINALIZATION
        and config.get("distance_marginalization") is False
        and type(config.get("n_devices")) is int
        and config.get("n_devices") == 4
        and type(config.get("num_gibbs_sweeps")) is int
        and config.get("num_gibbs_sweeps") == 2
        and config.get("sampler_scheduler") == "fsm"
        and config.get("width_adaptation_rate") == 0.25
        and config.get("width_target_expansions") == 1.0
        and config.get("width_target_shrinks") == 3.0
        and config.get("detectors") == ["H1", "L1", "V1"]
        and config.get("waveform") == "IMRPhenomPv2_NRTidalv2"
        and isinstance(prior, dict)
        and prior.get("phase_c")
        == {
            "distribution": "uniform",
            "range_radians": [0.0, "2pi"],
        }
        and prior.get("psi")
        == {
            "distribution": "uniform",
            "range_radians": [0.0, "pi"],
        }
        and prior.get("spin_magnitudes")
        == {"distribution": "uniform", "range": [0.0, 0.05]}
    )


def _revision(value: str) -> str:
    if _REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase 40-character Git SHA")
    return value


def _tree_sha256(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_campaign", type=Path)
    parser.add_argument("output_campaign", type=Path)
    parser.add_argument("--scheme", choices=SCHEMES, default=DEFAULT_SCHEME)
    parser.add_argument("--implementation-revision", type=_revision, required=True)
    parser.add_argument(
        "--implementation-tree-sha256", type=_tree_sha256, required=True
    )
    return parser.parse_args(argv)


def prepare_remediation_campaign(
    source_campaign: Path,
    output_campaign: Path,
    *,
    scheme: str,
    implementation_revision: str,
    implementation_tree_sha256: str,
) -> dict[str, Any]:
    """Copy an eligible paired source into the requested remediation arm."""

    if _REVISION_RE.fullmatch(implementation_revision) is None:
        raise ValueError("implementation_revision must be a lowercase Git SHA")
    if _SHA256_RE.fullmatch(implementation_tree_sha256) is None:
        raise ValueError("implementation_tree_sha256 must be a lowercase SHA-256")
    if scheme not in SCHEMES:
        raise ValueError(
            f"unknown remediation scheme {scheme!r}; expected {list(SCHEMES)}"
        )
    is_netsky = scheme == NETSKY_SCHEME

    source_campaign = source_campaign.expanduser().resolve()
    output_campaign = output_campaign.expanduser().resolve()
    if output_campaign.exists():
        raise FileExistsError(f"remediation campaign already exists: {output_campaign}")

    source_manifest = load_manifest(source_campaign)
    if not publication_eligible(source_manifest):
        raise ValueError("source campaign is not an iid P-P calibration campaign")
    source_recoveries = source_manifest.get("n_injections")
    source_catalogue_size = source_manifest.get("catalogue_size")
    if is_netsky and (
        type(source_recoveries) is not int
        or source_recoveries < 1
        or type(source_catalogue_size) is not int
        or source_catalogue_size != source_recoveries
    ):
        raise ValueError("netsky source campaign must be a complete non-empty batch")
    if not is_netsky and int(source_recoveries or -1) != PAPER_PP_RECOVERIES:
        raise ValueError(
            f"source campaign must select exactly {PAPER_PP_RECOVERIES} recoveries"
        )
    selection = source_manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("start_inclusive") != 0:
        raise ValueError("source campaign must use the leading contiguous selection")
    expected_stop = source_recoveries if is_netsky else PAPER_PP_RECOVERIES
    if selection.get("stop_exclusive") != expected_stop:
        if is_netsky:
            raise ValueError("netsky source selection must cover its complete batch")
        raise ValueError("source campaign selection does not end at recovery 100")

    source_config = source_manifest.get("config")
    if not isinstance(source_config, dict):
        raise TypeError("source campaign has no valid configuration")
    if is_netsky and not _is_expected_netsky_source(source_config):
        raise ValueError(
            "source campaign is not the expected fast-ridge, time-marginalized, "
            "sampled-d_L FSM D=4, M=2 campaign"
        )
    if not is_netsky and (
        source_config.get("n_devices") != 4
        or source_config.get("num_gibbs_sweeps") != 1
        or source_config.get("sampler_scheduler", "fsm") != "fsm"
    ):
        raise ValueError("source campaign is not the expected FSM D=4, M=1 campaign")
    source_blocks = source_config.get("blocks")
    remediation_blocks = blocks_for_scheme(source_blocks, scheme)

    catalogue = read_catalogue(
        source_campaign / str(source_manifest["catalogue"]["path"])
    )
    if len(catalogue) != int(source_manifest.get("catalogue_size", -1)):
        raise ValueError("source catalogue size does not match its manifest")
    selected_recoveries = source_recoveries if is_netsky else PAPER_PP_RECOVERIES
    assert isinstance(selected_recoveries, int)
    for injection_id in range(selected_recoveries):
        _validate_result(
            result_dir(source_campaign, injection_id),
            injection_id,
            source_manifest,
            catalogue,
        )

    output_campaign.mkdir(parents=True)
    try:
        catalogue_relative = Path(str(source_manifest["catalogue"]["path"]))
        catalogue_destination = output_campaign / catalogue_relative
        catalogue_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_campaign / catalogue_relative, catalogue_destination)
        shutil.copytree(source_campaign / "inputs", output_campaign / "inputs")

        config = copy.deepcopy(source_config)
        if is_netsky:
            config["campaign"] = "pp-netsky-m2-paired-remediation"
            config["paper_configuration"] = "D=4/M=2 netsky paired arm"
        else:
            config["campaign"] = f"paper-fig2a-m1-{scheme}-remediation"
            config["paper_configuration"] = f"Sharded M=1 ({scheme} blocks)"
        config["blocks"] = remediation_blocks
        if is_netsky:
            config.update(
                {
                    "adaptive_slice_widths": False,
                    "blocking_scheme": NETSKY_SCHEME,
                    "bridge_blocks": copy.deepcopy(NETSKY_BRIDGE_BLOCKS),
                    "bracket_mode": "stepping-out",
                    "periodic_wrapped_covariance": True,
                    "num_gibbs_sweeps": 2,
                    "fold_symmetry": {},
                    "fold_unfold_batch_size": 1,
                }
            )

        changed_variables: dict[str, Any] = {
            "blocks": {
                "source": copy.deepcopy(source_blocks),
                "remediation": copy.deepcopy(remediation_blocks),
            }
        }
        if is_netsky:
            changed_variables.update(
                {
                    "adaptive_slice_widths": {
                        "source": source_config.get("adaptive_slice_widths"),
                        "remediation": False,
                    },
                    "blocking_scheme": {
                        "source": source_config.get("blocking_scheme"),
                        "remediation": NETSKY_SCHEME,
                    },
                    "bridge_blocks": {
                        "source": copy.deepcopy(source_config.get("bridge_blocks", [])),
                        "remediation": copy.deepcopy(NETSKY_BRIDGE_BLOCKS),
                    },
                    "bracket_mode": {
                        "source": source_config.get("bracket_mode"),
                        "remediation": "stepping-out",
                    },
                    "periodic_wrapped_covariance": {
                        "source": bool(
                            source_config.get("periodic_wrapped_covariance", False)
                        ),
                        "remediation": True,
                    },
                    "fold_symmetry": {
                        "source": copy.deepcopy(source_config.get("fold_symmetry")),
                        "remediation": {},
                    },
                    "fold_unfold_batch_size": {
                        "source": source_config.get("fold_unfold_batch_size"),
                        "remediation": 1,
                    },
                }
            )

        manifest = {
            key: copy.deepcopy(value)
            for key, value in source_manifest.items()
            if key
            not in {"config_sha256", "created_at_utc", "implementation_diagnostic"}
        }
        manifest.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "config": config,
                "catalogue": {
                    **copy.deepcopy(source_manifest["catalogue"]),
                    "sha256": file_sha256(catalogue_destination),
                    "bytes": catalogue_destination.stat().st_size,
                },
                "implementation_diagnostic": {
                    "implementation_label": "candidate",
                    "implementation_revision": implementation_revision,
                    "implementation_tree_sha256": implementation_tree_sha256,
                    "sampler_scheduler": "fsm",
                    "configuration_semantics": (
                        "Paired D=4/M=2 netsky remediation of the complete "
                        "fast-ridge real-PE-matched P-P source batch. Catalogue "
                        "rows, truths, seeds, PSDs, priors, likelihood, live set, "
                        "stopping rule, and nuisance treatment remain frozen; the "
                        "recorded netsky blocking, bridge, fixed-width stepping-out "
                        "covariance, and quotient-fold configuration selects this arm."
                        if is_netsky
                        else (
                            "Paired full Fig. 2a remediation: D=4, M=1, the complete "
                            "iid catalogue and leading-100 selection, truths, seeds, "
                            "PSDs, priors, likelihood, live set, stopping rule, and "
                            "all other scientific settings match the source. Only "
                            "the block partition and presentation labels change."
                        )
                    ),
                    "source_campaign_config_sha256": source_manifest["config_sha256"],
                    "changed_variables": changed_variables,
                },
                "blocking_remediation": {
                    "scheme": scheme,
                    "paired_source_campaign": source_campaign.name,
                    "paired_source_manifest_sha256": file_sha256(
                        source_campaign / "manifest.json"
                    ),
                    "acceptance_test": (
                        "Fisher-combined exact KS p-value over all 15 sampled "
                        "parameters, including q, must exceed 0.05"
                    ),
                },
            }
        )
        if is_netsky:
            manifest["sampling_parameterization"] = copy.deepcopy(
                _NETSKY_SAMPLING_PARAMETERIZATION
            )
        manifest["config_sha256"] = canonical_sha256(manifest)
        atomic_write_json(output_campaign / "manifest.json", manifest)
        refresh_status(output_campaign, selected_recoveries)
        load_manifest(output_campaign)
        return manifest
    except BaseException:
        shutil.rmtree(output_campaign, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = prepare_remediation_campaign(
        args.source_campaign,
        args.output_campaign,
        scheme=args.scheme,
        implementation_revision=args.implementation_revision,
        implementation_tree_sha256=args.implementation_tree_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
