#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$script_dir/../../.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir=""
workload="aligned-11d"
candidate_only=false
original_sharded_only=false
no_hlo=false
seed=0
paper_baseline="86335bdb1e7ef6191937dd17b2ca53edbb1d899f"

usage() {
  cat <<'EOF'
Usage: run_on_pod.sh [--output-dir PATH] [--workload aligned-11d|paper-15d] [--seed N] [--candidate-only [--no-hlo]|--original-sharded-only]

The legacy positional output directory remains supported. The workload defaults
to aligned-11d so existing invocations retain their historical behaviour.
--candidate-only runs one four-GPU candidate analysis with profiling, telemetry,
per-slice diagnostics, posterior-sample persistence, and GPU HLO capture.
--no-hlo disables GPU HLO capture for --candidate-only while retaining all other
profiling, telemetry, diagnostics, and posterior artifacts.
--original-sharded-only runs the same analysis against the pinned paper-style
sharded-live-state revision, without running the candidate.
--seed selects the non-negative candidate or original-sharded sampler seed.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --output-dir)
      if [[ "$#" -lt 2 ]]; then
        echo "--output-dir requires a path" >&2
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    --workload)
      if [[ "$#" -lt 2 ]]; then
        echo "--workload requires aligned-11d or paper-15d" >&2
        exit 2
      fi
      workload="$2"
      shift 2
      ;;
    --candidate-only)
      candidate_only=true
      shift
      ;;
    --seed)
      if [[ "$#" -lt 2 ]]; then
        echo "--seed requires a non-negative integer" >&2
        exit 2
      fi
      seed="$2"
      shift 2
      ;;
    --no-hlo)
      no_hlo=true
      shift
      ;;
    --original-sharded-only)
      original_sharded_only=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$output_dir" ]]; then
        echo "Output directory was provided more than once" >&2
        usage >&2
        exit 2
      fi
      output_dir="$1"
      shift
      ;;
  esac
done

case "$workload" in
  aligned-11d|paper-15d) ;;
  *)
    echo "Unknown workload: $workload" >&2
    usage >&2
    exit 2
    ;;
esac

if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "--seed requires a non-negative integer" >&2
  exit 2
fi

if [[ "$candidate_only" == true && "$original_sharded_only" == true ]]; then
  echo "--candidate-only and --original-sharded-only are mutually exclusive" >&2
  exit 2
fi

if [[ "$no_hlo" == true && "$candidate_only" == false ]]; then
  echo "--no-hlo requires --candidate-only" >&2
  exit 2
fi

package_manifest="$repository/.runpod/package-manifest.json"
package_mode=false
packaged_baseline_revision=""
packaged_baseline_root=""
if [[ -d "$repository/.git" ]]; then
  candidate_revision="$(git -C "$repository" rev-parse HEAD)"
elif [[ -f "$package_manifest" ]]; then
  package_mode=true
  IFS=$'\t' read -r \
    candidate_revision \
    packaged_baseline_revision \
    packaged_baseline_root < <(
      python3 - "$package_manifest" "$paper_baseline" <<'PY'
import json
import re
import sys
from pathlib import Path, PurePosixPath

path = Path(sys.argv[1])
expected_baseline = sys.argv[2]
manifest = json.loads(path.read_text())
if manifest.get("schema_version") != 1 or manifest.get("package") != "jim-gw170817-runpod":
    raise SystemExit(f"unsupported RunPod package manifest: {path}")
candidate_revision = manifest.get("candidate", {}).get("revision")
baseline_revision = manifest.get("baseline", {}).get("revision")
baseline_root = manifest.get("baseline", {}).get("root")
if not isinstance(candidate_revision, str) or re.fullmatch(r"[0-9a-f]{40}", candidate_revision) is None:
    raise SystemExit("package manifest has an invalid candidate revision")
if baseline_revision != expected_baseline:
    raise SystemExit(
        f"package baseline is {baseline_revision!r}, expected {expected_baseline}"
    )
if not isinstance(baseline_root, str):
    raise SystemExit("package manifest has no baseline root")
root_path = PurePosixPath(baseline_root)
if root_path.is_absolute() or ".." in root_path.parts:
    raise SystemExit("package manifest has an unsafe baseline root")
print(candidate_revision, baseline_revision, baseline_root, sep="\t")
PY
    )
  if [[ ! -d "$repository/$packaged_baseline_root/src/jimgw" ]]; then
    echo "Packaged baseline root is incomplete: $packaged_baseline_root" >&2
    exit 1
  fi
else
  echo "Workspace has neither .git nor .runpod/package-manifest.json" >&2
  exit 1
fi

if [[ "$package_mode" == true && "$candidate_only" == false && "$original_sharded_only" == false ]]; then
  echo "The minimized git-free package requires --candidate-only or --original-sharded-only." >&2
  echo "The legacy paired matrix remains available from a full Git checkout." >&2
  exit 2
fi

output_dir="${output_dir:-/workspace/jim-gw170817-results/$timestamp}"

export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"

gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count + 0}')"
if [[ "$gpu_count" -lt 4 ]]; then
  echo "Expected four GPUs, but nvidia-smi found $gpu_count" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.11.2/install.sh | sh
fi
uv --version
uv python install 3.11
uv sync --directory "$repository" --frozen --extra cuda \
  --group test --group cross-validation --python 3.11

JAX_PLATFORMS=cuda uv run --directory "$repository" --no-sync python - <<'PY'
import jax

devices = jax.devices()
print(f"jax={jax.__version__} backend={jax.default_backend()} devices={devices}")
if jax.default_backend() != "gpu" or len(devices) != 4:
    raise SystemExit("JAX did not initialise all four GPUs")
PY

XLA_FLAGS=--xla_force_host_platform_device_count=4 \
JAX_PLATFORMS=cpu \
uv run --directory "$repository" --no-sync pytest \
  tests/unit/benchmarks/test_paper_model.py \
  tests/unit/benchmarks/test_paper_model_basis.py \
  tests/unit/benchmarks/test_benchmark_gw170817_full_run.py \
  tests/unit/benchmarks/test_compare_gw170817_full_run.py \
  tests/unit/samplers/blackjax/test_sharding.py \
  tests/unit/samplers/blackjax/test_slice.py \
  tests/integration/test_sampler_sharding.py \
  -q --tb=short

mkdir -p "$output_dir"
if [[ "$package_mode" == true ]]; then
  cp "$package_manifest" "$output_dir/package-manifest.json"
fi

if [[ "$candidate_only" == true || "$original_sharded_only" == true ]]; then
  if [[ "$original_sharded_only" == true ]]; then
    implementation_label="paper-baseline"
    if [[ "$package_mode" == true ]]; then
      implementation_revision="$packaged_baseline_revision"
      implementation_root="$repository/$packaged_baseline_root"
      packaged_baseline_output="$output_dir/implementations/original-sharded"
      mkdir -p "$(dirname "$packaged_baseline_output")"
      cp -R "$implementation_root" "$packaged_baseline_output"
    else
      implementation_revision="$(git -C "$repository" rev-parse --verify "${paper_baseline}^{commit}")"
      implementation_root="$output_dir/implementations/original-sharded"
      implementation_archive="$output_dir/implementations/original-sharded.tar"
      mkdir -p "$implementation_root"
      git -C "$repository" archive \
        --format=tar \
        --output="$implementation_archive" \
        "$implementation_revision"
      tar -xf "$implementation_archive" -C "$implementation_root"
    fi
    run_stem="original-sharded-${workload}-g4-seed${seed}"
  else
    implementation_label="ours"
    implementation_revision="$candidate_revision"
    implementation_root="$repository"
    run_stem="candidate-${workload}-g4-seed${seed}"
  fi

  data_file="$output_dir/data/gw170817.npz"
  report_file="$output_dir/${run_stem}.json"
  samples_file="$output_dir/posterior/${run_stem}.npz"
  nested_file="$output_dir/nested/${run_stem}.npz"
  slice_file="$output_dir/per-slice/${run_stem}.npz"
  hlo_dir="$output_dir/gpu-hlo/${run_stem}"
  slice_arguments=()
  nested_arguments=()
  if [[ "$candidate_only" == true ]]; then
    slice_arguments=(--slice-data-output "$slice_file")
    nested_arguments=(--nested-output "$nested_file")
  fi
  profile_dir="$output_dir/profiles/${run_stem}"
  telemetry_file="$output_dir/telemetry/${run_stem}.dmon"
  mkdir -p \
    "$(dirname "$data_file")" \
    "$(dirname "$samples_file")" \
    "$(dirname "$nested_file")" \
    "$(dirname "$slice_file")" \
    "$profile_dir" \
    "$(dirname "$telemetry_file")"
  if [[ "$no_hlo" == false ]]; then
    mkdir -p "$hlo_dir"
  fi

  JAX_PLATFORMS=cpu \
  uv run --directory "$repository" --no-sync python \
    benchmarks/device_parallel_nss/benchmark_gw170817_full_run.py \
    --prepare-data \
    --data-file "$data_file" \
    --workload "$workload" \
    --implementation-root "$implementation_root" \
    --implementation-label "$implementation_label" \
    --implementation-revision "$implementation_revision" \
    --output "$output_dir/data-prepare.json"

  if [[ "$package_mode" == true && "$original_sharded_only" == true ]]; then
    python3 - "$output_dir/data-prepare.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if report.get("data", {}).get("reused_existing") is not True:
    raise SystemExit(
        "original-sharded run did not reuse the candidate's frozen data bundle"
    )
PY
  fi

  benchmark_environment=(
    "CUDA_VISIBLE_DEVICES=0,1,2,3"
    "JAX_PLATFORMS=cuda"
    "JAX_ENABLE_COMPILATION_CACHE=0"
    "NCCL_NVLS_ENABLE=0"
    "PYTHONHASHSEED=0"
    "XLA_PYTHON_CLIENT_PREALLOCATE=false"
  )
  if [[ "$no_hlo" == false ]]; then
    benchmark_environment+=(
      "XLA_FLAGS=--xla_dump_to=$hlo_dir --xla_dump_hlo_as_text"
    )
  fi
  env "${benchmark_environment[@]}" \
  uv run --directory "$repository" --no-sync python \
    benchmarks/device_parallel_nss/benchmark_gw170817_full_run.py \
    --data-file "$data_file" \
    --workload "$workload" \
    --seed "$seed" \
    --n-devices 4 \
    --implementation-root "$implementation_root" \
    --implementation-label "$implementation_label" \
    --implementation-revision "$implementation_revision" \
    --profile-dir "$profile_dir" \
    --profile-warmup-steps 10 \
    --profile-steps 15 \
    --telemetry-output "$telemetry_file" \
    "${slice_arguments[@]}" \
    --samples-output "$samples_file" \
    "${nested_arguments[@]}" \
    --output "$report_file"

  uv run --directory "$repository" --no-sync python - \
    "$output_dir" "$workload" "$run_stem" \
    "$implementation_label" "$implementation_revision" "$no_hlo" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()
workload = sys.argv[2]
run_stem = sys.argv[3]
implementation_label = sys.argv[4]
implementation_revision = sys.argv[5]
no_hlo = sys.argv[6] == "true"
report_path = root / f"{run_stem}.json"
report = json.loads(report_path.read_text())
artifact = report["results"]["posterior_artifact"]
samples_path = Path(artifact["path"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

digest = sha256_file(samples_path)
if digest != artifact["sha256"]:
    raise SystemExit("posterior artifact SHA-256 mismatch")
with np.load(samples_path, allow_pickle=False) as samples:
    if set(samples.files) != set(artifact["fields"]):
        raise SystemExit("posterior artifact fields do not match the report")
    counts = {name: int(samples[name].shape[0]) for name in samples.files}
    if set(counts.values()) != {artifact["count"]} or artifact["count"] < 1:
        raise SystemExit("posterior artifact sample counts are invalid")
    if not all(samples[name].ndim == 1 for name in samples.files):
        raise SystemExit("posterior artifact arrays must be one-dimensional")
    if not all(np.all(np.isfinite(samples[name])) for name in samples.files):
        raise SystemExit("posterior artifact contains non-finite values")

profile = report["timing_seconds"]["outer_step"]["profile"]
profile_path = Path(profile["directory"])
if len(profile["captured_step_indices"]) != profile["requested_steps"]:
    raise SystemExit("profiler did not capture every requested steady step")
if not profile_path.is_dir() or not any(path.is_file() for path in profile_path.rglob("*")):
    raise SystemExit("profiler trace directory is empty")
if report["config"]["workload"] != workload:
    raise SystemExit("runner report recorded the wrong workload")
if workload == "paper-15d" and report["config"]["sampled_dimensions"] != 15:
    raise SystemExit("paper workload did not report 15 sampled dimensions")
if report["implementation"]["label"] != implementation_label:
    raise SystemExit("runner report recorded the wrong implementation label")
if report["implementation"]["revision"] != implementation_revision:
    raise SystemExit("runner report recorded the wrong implementation revision")

verification = {
    "workload": workload,
    "posterior_samples": artifact["count"],
    "posterior_fields": artifact["fields"],
    "posterior_sha256": artifact["sha256"],
    "profiled_steps": len(profile["captured_step_indices"]),
    "profile_files": sum(path.is_file() for path in profile_path.rglob("*")),
    "implementation_label": implementation_label,
    "implementation_revision": implementation_revision,
}
if no_hlo:
    verification["hlo_capture"] = False
else:
    hlo_path = root / "gpu-hlo" / run_stem
    hlo_files = sorted(path for path in hlo_path.rglob("*") if path.is_file())
    optimized_hlo_files = [
        path for path in hlo_files if path.name.endswith("after_optimizations.txt")
    ]
    if not hlo_files:
        raise SystemExit("GPU HLO dump directory is empty")
    if not optimized_hlo_files:
        raise SystemExit("GPU HLO dump has no optimized text module")
    hlo_manifest = {
        "root": str(hlo_path),
        "files": [
            {
                "path": path.relative_to(hlo_path).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in hlo_files
        ],
        "file_count": len(hlo_files),
        "optimized_text_module_count": len(optimized_hlo_files),
        "total_bytes": sum(path.stat().st_size for path in hlo_files),
    }
    hlo_manifest_path = hlo_path / "hlo-manifest.json"
    hlo_manifest_path.write_text(
        json.dumps(hlo_manifest, indent=2, sort_keys=True) + "\n"
    )
    verification.update(
        {
            "hlo_files": hlo_manifest["file_count"],
            "optimized_hlo_files": hlo_manifest["optimized_text_module_count"],
            "hlo_bytes": hlo_manifest["total_bytes"],
            "hlo_manifest": str(hlo_manifest_path),
        }
    )
(root / "artifact-verification.json").write_text(
    json.dumps(verification, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(verification, sort_keys=True))
PY

  archive="${output_dir}.tar.gz"
  tar -C "$(dirname "$output_dir")" -czf "$archive" "$(basename "$output_dir")"
  echo "Single-implementation benchmark complete: $output_dir"
  echo "Download bundle: $archive"
  exit 0
fi

paired_dir="$output_dir/paired-four-gpu"

uv run --directory "$repository" --no-sync python \
  benchmarks/device_parallel_nss/compare_gw170817_full_run.py \
  --output-dir "$paired_dir" \
  --workload "$workload" \
  --seeds 0 \
  --profile-dir "$output_dir/profiles" \
  --profile-warmup-steps 10 \
  --profile-steps 15 \
  --slice-data-dir "$output_dir/per-slice" \
  --telemetry-dir "$output_dir/telemetry" \
  --jax-compilation-cache-dir "$output_dir/jax-compilation-cache" \
  --persistent-cache-probe

mkdir -p "$output_dir/candidate-device-sweep"
for device_count in 1 2; do
  if [[ "$device_count" -eq 1 ]]; then
    visible_devices="0"
  else
    visible_devices="0,1"
  fi
  CUDA_VISIBLE_DEVICES="$visible_devices" \
  JAX_PLATFORMS=cuda \
  JAX_ENABLE_COMPILATION_CACHE=0 \
  NCCL_NVLS_ENABLE=0 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --directory "$repository" --no-sync python \
    benchmarks/device_parallel_nss/benchmark_gw170817_full_run.py \
    --data-file "$paired_dir/data/gw170817.npz" \
    --workload "$workload" \
    --seed 0 \
    --n-devices "$device_count" \
    --implementation-root "$repository" \
    --implementation-label ours \
    --implementation-revision "$candidate_revision" \
    --slice-data-output "$output_dir/per-slice/ours-g${device_count}-seed0-sweep.npz" \
    --output "$output_dir/candidate-device-sweep/ours-g${device_count}-seed0.json"
done

CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=cuda \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --directory "$repository" --no-sync python \
  benchmarks/device_parallel_nss/benchmark_gw170817_likelihood_lanes.py \
  --data-file "$paired_dir/data/gw170817.npz" \
  --workload "$workload" \
  --implementation-root "$repository" \
  --implementation-label ours \
  --implementation-revision "$candidate_revision" \
  --lanes 16 32 64 128 \
  --warmup 2 \
  --repeats 10 \
  --output "$output_dir/likelihood-lane-sweep.json"

uv run --directory "$repository" --no-sync python \
  benchmarks/device_parallel_nss/summarize_gw170817_diagnostics.py \
  --paired-dir "$paired_dir" \
  --sweep-dir "$output_dir/candidate-device-sweep" \
  --microbenchmark "$output_dir/likelihood-lane-sweep.json" \
  --workload "$workload" \
  --output-dir "$output_dir/diagnostic-summary"

archive="${output_dir}.tar.gz"
tar -C "$(dirname "$output_dir")" -czf "$archive" "$(basename "$output_dir")"
echo "Benchmark complete: $output_dir"
echo "Download bundle: $archive"
