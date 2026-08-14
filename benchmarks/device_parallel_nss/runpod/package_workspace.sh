#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$script_dir/../../.." && pwd)"
baseline_revision="86335bdb1e7ef6191937dd17b2ca53edbb1d899f"
candidate_revision="$(git -C "$repository" rev-parse HEAD)"
candidate_short="${candidate_revision:0:12}"
output="${1:-/private/tmp/jim-nss-benchmark-${candidate_short}.tar.gz}"
staging="$(mktemp -d /private/tmp/jim-nss-package.XXXXXX)"

cleanup() {
  if [[ -n "${staging:-}" && -d "$staging" ]]; then
    rm -rf -- "$staging"
  fi
}
trap cleanup EXIT

git -C "$repository" rev-parse --verify "${baseline_revision}^{commit}" >/dev/null

# Copy only the current candidate source, package metadata, benchmark harness,
# and tests exercised by run_on_pod.sh. `git ls-files --others` is deliberately
# scoped to src/: unrelated workspace artifacts never enter the staging tree.
python3 - "$repository" "$staging" <<'PY'
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
staging = Path(sys.argv[2]).resolve()

source_result = subprocess.run(
    [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "src",
    ],
    cwd=repository,
    check=True,
    capture_output=True,
)
source_paths = {
    Path(raw.decode())
    for raw in source_result.stdout.split(b"\0")
    if raw
}
required_paths = {
    Path(path)
    for path in (
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "benchmarks/device_parallel_nss/benchmark_gw170817_full_run.py",
        "benchmarks/device_parallel_nss/benchmark_gw170817_likelihood_lanes.py",
        "benchmarks/device_parallel_nss/compare_gw170817_full_run.py",
        "benchmarks/device_parallel_nss/paper_model.py",
        "benchmarks/device_parallel_nss/paper_model_basis.py",
        "benchmarks/device_parallel_nss/sampler_ablation.py",
        "benchmarks/device_parallel_nss/summarize_gw170817_diagnostics.py",
        "benchmarks/injection_campaign/__init__.py",
        "benchmarks/injection_campaign/common.py",
        "benchmarks/injection_campaign/evaluate_time_marginalization_diagnostic.py",
        "benchmarks/injection_campaign/evaluate_historical_stress.py",
        "benchmarks/injection_campaign/prepare_campaign.py",
        "benchmarks/injection_campaign/prepare_baseline_diagnostic.py",
        "benchmarks/injection_campaign/prepare_fsm_diagnostic.py",
        "benchmarks/injection_campaign/prepare_historical_stress.py",
        "benchmarks/injection_campaign/prepare_time_anchor_diagnostic.py",
        "benchmarks/injection_campaign/prepare_time_marginalization_diagnostic.py",
        "benchmarks/injection_campaign/probe_likelihood.py",
        "benchmarks/injection_campaign/run_injection.py",
        "benchmarks/injection_campaign/run_campaign.py",
        "benchmarks/injection_campaign/plot_pp.py",
        "benchmarks/injection_campaign/plot_timing.py",
        "benchmarks/injection_campaign/README.md",
        "benchmarks/injection_campaign/runpod/provision.sh",
        "benchmarks/injection_campaign/runpod/run_on_pod.sh",
        "benchmarks/injection_campaign/runpod/upload_and_run.py",
        "benchmarks/device_parallel_nss/runpod/FSM_VALIDATION.md",
        "benchmarks/device_parallel_nss/runpod/README.md",
        "benchmarks/device_parallel_nss/runpod/package_workspace.sh",
        "benchmarks/device_parallel_nss/runpod/run_on_pod.sh",
        "benchmarks/device_parallel_nss/runpod/upload_and_run.py",
        "tests/conftest.py",
        "tests/__init__.py",
        "tests/integration/__init__.py",
        "tests/integration/_helpers.py",
        "tests/integration/test_sampler_sharding.py",
        "tests/unit/benchmarks/test_benchmark_gw170817_full_run.py",
        "tests/unit/benchmarks/test_compare_gw170817_full_run.py",
        "tests/unit/benchmarks/test_paper_model.py",
        "tests/unit/benchmarks/test_paper_model_basis.py",
        "tests/unit/benchmarks/test_time_anchor_diagnostic.py",
        "tests/unit/benchmarks/test_time_marginalization_diagnostic.py",
        "tests/unit/samplers/blackjax/test_sharding.py",
        "tests/unit/samplers/blackjax/test_slice.py",
    )
}

for relative in sorted(source_paths | required_paths):
    source = repository / relative
    if not source.is_file():
        raise SystemExit(f"required package file is missing: {relative}")
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
PY

baseline_root="$staging/.runpod/implementations/paper-baseline"
baseline_archive="$staging/.runpod/paper-baseline.tar"
mkdir -p "$baseline_root"
git -C "$repository" archive \
  --format=tar \
  --output="$baseline_archive" \
  "$baseline_revision" \
  src/jimgw
tar -xf "$baseline_archive" -C "$baseline_root"
rm -f -- "$baseline_archive"

# The manifest hashes the exact curated candidate and baseline trees. It is
# intentionally written inside the package, so upload and pod-side scripts can
# validate provenance without access to the original Git object database.
python3 - \
  "$repository" \
  "$staging" \
  "$candidate_revision" \
  "$baseline_revision" <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

repository = Path(sys.argv[1]).resolve()
staging = Path(sys.argv[2]).resolve()
candidate_revision = sys.argv[3]
baseline_revision = sys.argv[4]
baseline_root = Path(".runpod/implementations/paper-baseline")


def tree_manifest(root: Path, *, excluded_prefix: Path | None = None) -> tuple[list[dict[str, object]], str]:
    files: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        if excluded_prefix is not None and relative.is_relative_to(excluded_prefix):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative.as_posix(), "sha256": digest, "bytes": path.stat().st_size})
        aggregate.update(relative.as_posix().encode())
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
    return files, aggregate.hexdigest()


candidate_files, candidate_tree_sha256 = tree_manifest(
    staging,
    excluded_prefix=Path(".runpod"),
)
baseline_files, baseline_tree_sha256 = tree_manifest(staging / baseline_root)
status = subprocess.run(
    ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    cwd=repository,
    check=True,
    capture_output=True,
).stdout
manifest = {
    "schema_version": 1,
    "package": "jim-gw170817-runpod",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "provenance": {
        "generator": "benchmarks/device_parallel_nss/runpod/package_workspace.sh",
        "archive_layout": "curated working-tree candidate plus exported baseline src/jimgw",
        "git_metadata_included": False,
    },
    "candidate": {
        "revision": candidate_revision,
        "root": ".",
        "capture": "curated current working tree",
        "working_tree_dirty": bool(status),
        "working_tree_status_sha256": hashlib.sha256(status).hexdigest(),
        "tree_sha256": candidate_tree_sha256,
        "file_count": len(candidate_files),
        "files": candidate_files,
    },
    "baseline": {
        "revision": baseline_revision,
        "root": baseline_root.as_posix(),
        "capture": f"git archive {baseline_revision} -- src/jimgw",
        "tree_sha256": baseline_tree_sha256,
        "file_count": len(baseline_files),
        "files": baseline_files,
    },
}
manifest_path = staging / ".runpod/package-manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

mkdir -p "$(dirname "$output")"
COPYFILE_DISABLE=1 tar -C "$staging" -czf "$output" .
echo "$output"
