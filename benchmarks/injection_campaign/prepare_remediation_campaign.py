"""Freeze a paired 100-event M=1 blocking-remediation campaign.

The output retains the source campaign's complete iid catalogue, leading-100
selection, truths, seeds, PSDs, priors, likelihood settings, and sampler
settings.  Only the named block partition and presentation labels change.
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
    SCHEMES,
    blocks_for_scheme,
)

DEFAULT_SCHEME = "all-slow-time"
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


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
    """Copy an eligible leading-100 campaign and change only its block partition."""

    if _REVISION_RE.fullmatch(implementation_revision) is None:
        raise ValueError("implementation_revision must be a lowercase Git SHA")
    if _SHA256_RE.fullmatch(implementation_tree_sha256) is None:
        raise ValueError("implementation_tree_sha256 must be a lowercase SHA-256")

    source_campaign = source_campaign.expanduser().resolve()
    output_campaign = output_campaign.expanduser().resolve()
    if output_campaign.exists():
        raise FileExistsError(f"remediation campaign already exists: {output_campaign}")

    source_manifest = load_manifest(source_campaign)
    if not publication_eligible(source_manifest):
        raise ValueError("source campaign is not an iid P-P calibration campaign")
    if int(source_manifest.get("n_injections", -1)) != PAPER_PP_RECOVERIES:
        raise ValueError(
            f"source campaign must select exactly {PAPER_PP_RECOVERIES} recoveries"
        )
    selection = source_manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("start_inclusive") != 0:
        raise ValueError("source campaign must use the leading contiguous selection")
    if selection.get("stop_exclusive") != PAPER_PP_RECOVERIES:
        raise ValueError("source campaign selection does not end at recovery 100")

    source_config = source_manifest.get("config")
    if not isinstance(source_config, dict):
        raise TypeError("source campaign has no valid configuration")
    if (
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
    for injection_id in range(PAPER_PP_RECOVERIES):
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
        config["campaign"] = f"paper-fig2a-m1-{scheme}-remediation"
        config["paper_configuration"] = f"Sharded M=1 ({scheme} blocks)"
        config["blocks"] = remediation_blocks

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
                        "Paired full Fig. 2a remediation: D=4, M=1, the complete "
                        "iid catalogue and leading-100 selection, truths, seeds, "
                        "PSDs, priors, likelihood, live set, stopping rule, and all "
                        "other scientific settings match the source. Only the block "
                        "partition and presentation labels change."
                    ),
                    "source_campaign_config_sha256": source_manifest["config_sha256"],
                    "changed_variables": {
                        "blocks": {
                            "source": copy.deepcopy(source_blocks),
                            "remediation": copy.deepcopy(remediation_blocks),
                        }
                    },
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
        manifest["config_sha256"] = canonical_sha256(manifest)
        atomic_write_json(output_campaign / "manifest.json", manifest)
        refresh_status(output_campaign, PAPER_PP_RECOVERIES)
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
