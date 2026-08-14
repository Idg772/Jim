#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$script_dir/../../.." && pwd)"
output_dir="/workspace/jim-injection-campaign"
n_injections=100
catalogue_size=1000
seed=260728265
retry_count=2
start=0
stop=""
injection_ids=()
require_existing_campaign=false
plot=true
implementation="candidate"
fresh_processes=false

usage() {
  echo "Usage: $0 [--output-dir PATH] [--n-injections N] [--catalogue-size N] [--seed N] [--retry-count N] [--start N] [--stop N] [--injection-id N ...] [--require-existing-campaign] [--no-plot] [--implementation candidate|paper-baseline] [--fresh-processes]"
}

while (($#)); do
  case "$1" in
    --output-dir) output_dir="$2"; shift 2 ;;
    --n-injections) n_injections="$2"; shift 2 ;;
    --catalogue-size) catalogue_size="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --retry-count) retry_count="$2"; shift 2 ;;
    --start) start="$2"; shift 2 ;;
    --stop) stop="$2"; shift 2 ;;
    --injection-id) injection_ids+=("$2"); shift 2 ;;
    --require-existing-campaign) require_existing_campaign=true; shift ;;
    --no-plot) plot=false; shift ;;
    --implementation) implementation="$2"; shift 2 ;;
    --fresh-processes) fresh_processes=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$implementation" != "candidate" && "$implementation" != "paper-baseline" ]]; then
  echo "--implementation must be candidate or paper-baseline" >&2
  exit 2
fi
if [[ "$implementation" == "paper-baseline" && "$require_existing_campaign" != true ]]; then
  echo "paper-baseline runs require a frozen --require-existing-campaign input" >&2
  exit 2
fi

for value in "$n_injections" "$catalogue_size" "$retry_count" "$start"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "counts must be non-negative integers" >&2
    exit 2
  fi
done
if [[ "$n_injections" -lt 1 ]]; then
  echo "--n-injections must be positive" >&2
  exit 2
fi
if [[ "$catalogue_size" -lt 1 ]]; then
  echo "--catalogue-size must be positive" >&2
  exit 2
fi
if [[ "$n_injections" -gt "$catalogue_size" ]]; then
  echo "--n-injections cannot exceed --catalogue-size" >&2
  exit 2
fi
if [[ -n "$stop" && ! "$stop" =~ ^[1-9][0-9]*$ ]]; then
  echo "--stop must be a positive integer" >&2
  exit 2
fi
if [[ -n "$stop" && "$start" -ge "$stop" ]]; then
  echo "--start must be less than --stop" >&2
  exit 2
fi
if ((${#injection_ids[@]})); then
  if [[ "$start" -ne 0 || -n "$stop" ]]; then
    echo "--injection-id cannot be combined with --start/--stop" >&2
    exit 2
  fi
  declare -A seen_injection_ids=()
  for injection_id in "${injection_ids[@]}"; do
    if ! [[ "$injection_id" =~ ^[0-9]+$ ]]; then
      echo "--injection-id values must be non-negative integers" >&2
      exit 2
    fi
    if [[ -n "${seen_injection_ids[$injection_id]:-}" ]]; then
      echo "--injection-id values must be unique" >&2
      exit 2
    fi
    seen_injection_ids[$injection_id]=1
  done
fi
if [[ "$require_existing_campaign" == true && ! -f "$output_dir/manifest.json" ]]; then
  echo "Frozen campaign manifest is missing; refusing remote regeneration: $output_dir/manifest.json" >&2
  exit 1
fi

expected_gpu_count=4
if [[ -f "$output_dir/manifest.json" ]]; then
  expected_gpu_count="$(python3 - "$output_dir/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
value = manifest.get("config", {}).get("n_devices")
if isinstance(value, bool) or not isinstance(value, int) or value < 1:
    raise SystemExit("campaign manifest has an invalid config.n_devices")
print(value)
PY
)"
fi

export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
# Keep uv's transactional cache on the container filesystem. Runpod's mounted
# /workspace volume can span filesystem boundaries, which makes uv's atomic
# temporary-file rename fail with EXDEV while installing large CUDA wheels.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
# H200's NVLS path is not supported by this JAX/NCCL workload, and the
# injection is built on GPU 0 before the live set is sharded. Avoid reserving
# most of GPU 0 up front so all four devices retain room for the sampler.
export NCCL_NVLS_ENABLE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Hoist large closed-over detector arrays out of HLO so event values do not
# poison persistent compilation-cache keys. Keep JAX's documented 32-byte
# threshold for embedding small constants, preserving default numerics.
export JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="${JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS:-True}"
export JAX_EMBEDDED_CONSTANTS_MAX_BYTES="${JAX_EMBEDDED_CONSTANTS_MAX_BYTES:-32}"

package_manifest="$repository/.runpod/package-manifest.json"
if [[ ! -f "$package_manifest" ]]; then
  echo "Workspace package manifest is missing: $package_manifest" >&2
  exit 1
fi
read -r implementation_revision implementation_root implementation_tree_sha256 < <(
  python3 - "$package_manifest" "$implementation" <<'PY'
import json
import sys
from pathlib import PurePosixPath

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
section_name = "baseline" if sys.argv[2] == "paper-baseline" else "candidate"
section = manifest.get(section_name)
if not isinstance(section, dict):
    raise SystemExit(f"package has no {section_name} implementation")
revision = section.get("revision")
root = section.get("root")
tree_sha256 = section.get("tree_sha256")
if (
    not isinstance(revision, str)
    or not isinstance(root, str)
    or not isinstance(tree_sha256, str)
    or len(tree_sha256) != 64
):
    raise SystemExit(f"package {section_name} metadata is invalid")
path = PurePosixPath(root)
if path.is_absolute() or ".." in path.parts:
    raise SystemExit(f"package {section_name} root is unsafe")
print(revision, root, tree_sha256)
PY
)
implementation_root="$repository/$implementation_root"
if [[ "$implementation" == "paper-baseline" ]]; then
  if [[ ! -d "$implementation_root/src/jimgw" ]]; then
    echo "Packaged paper baseline is incomplete: $implementation_root" >&2
    exit 1
  fi
  export PYTHONPATH="$implementation_root/src:$repository${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$repository/src:$repository${PYTHONPATH:+:$PYTHONPATH}"
fi
export JIM_IMPLEMENTATION_LABEL="$implementation"
export JIM_IMPLEMENTATION_REVISION="$implementation_revision"
export JIM_IMPLEMENTATION_ROOT="$implementation_root"
export JIM_IMPLEMENTATION_TREE_SHA256="$implementation_tree_sha256"

gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count + 0}')"
if [[ "$gpu_count" -ne "$expected_gpu_count" ]]; then
  echo "Expected exactly $expected_gpu_count GPUs, but nvidia-smi found $gpu_count" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.11.2/install.sh | sh
fi
uv python install 3.11
uv sync --directory "$repository" --frozen --extra cuda \
  --group cross-validation --python 3.11

uv run --directory "$repository" --no-sync python - "$implementation" "$implementation_root" <<'PY'
import pathlib
import sys

import jimgw
from jimgw.samplers.config import BlackJAXSwiGConfig

label = sys.argv[1]
root = pathlib.Path(sys.argv[2]).resolve()
module = pathlib.Path(jimgw.__file__).resolve()
module.relative_to(root)
fields = getattr(BlackJAXSwiGConfig, "model_fields", {})
if label == "paper-baseline" and "scheduler" in fields:
    raise SystemExit("paper baseline unexpectedly exposes candidate scheduler API")
print(f"Implementation: label={label} module={module}")
PY

JAX_PLATFORMS=cuda uv run --directory "$repository" --no-sync python - "$expected_gpu_count" <<'PY'
import sys

import jax

expected = int(sys.argv[1])
devices = jax.devices()
if len(devices) != expected or any(device.platform != "gpu" for device in devices):
    raise SystemExit(f"Expected {expected} JAX GPUs, got: {devices}")
print("JAX devices:", devices)
PY

if [[ "$implementation" == "paper-baseline" ]]; then
  candidate_probe="/tmp/jim-candidate-likelihood-probe.json"
  baseline_probe="/tmp/jim-paper-baseline-likelihood-probe.json"
  JAX_PLATFORMS=cuda \
  PYTHONPATH="$repository/src:$repository" \
  JIM_IMPLEMENTATION_LABEL=candidate \
  JIM_IMPLEMENTATION_ROOT="$repository" \
    uv run --directory "$repository" --no-sync python -m \
      benchmarks.injection_campaign.probe_likelihood \
      "$output_dir" 0 > "$candidate_probe"
  JAX_PLATFORMS=cuda \
  PYTHONPATH="$implementation_root/src:$repository" \
    uv run --directory "$repository" --no-sync python -m \
      benchmarks.injection_campaign.probe_likelihood \
      "$output_dir" 0 > "$baseline_probe"
  python3 - "$candidate_probe" "$baseline_probe" "$repository" "$implementation_root" <<'PY'
import json
import math
import pathlib
import sys

candidate = json.load(open(sys.argv[1], encoding="utf-8"))
baseline = json.load(open(sys.argv[2], encoding="utf-8"))
for field in ("config_sha256", "injection_id", "noise_seed", "sampler_seed"):
    if candidate[field] != baseline[field]:
        raise SystemExit(f"likelihood preflight differs in {field}")
if not candidate["finite"] or not baseline["finite"]:
    raise SystemExit("likelihood preflight produced a non-finite value")
candidate_root = pathlib.Path(sys.argv[3]).resolve()
baseline_root = pathlib.Path(sys.argv[4]).resolve()
pathlib.Path(candidate["jimgw_module"]).resolve().relative_to(candidate_root)
pathlib.Path(baseline["jimgw_module"]).resolve().relative_to(baseline_root)
for detector, value in candidate["optimal_snr_by_detector"].items():
    if not math.isclose(
        value,
        baseline["optimal_snr_by_detector"][detector],
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise SystemExit(f"likelihood preflight SNR differs for {detector}")
delta = baseline["log_likelihood_at_truth"] - candidate["log_likelihood_at_truth"]
if not math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1.0e-6):
    raise SystemExit(f"likelihood preflight mismatch: delta_logL={delta:.12g}")
print(
    "Likelihood preflight passed: "
    f"candidate={candidate['log_likelihood_at_truth']:.12f} "
    f"baseline={baseline['log_likelihood_at_truth']:.12f} "
    f"delta={delta:.3g}"
)
PY
else
  candidate_probe="/tmp/jim-candidate-likelihood-probe.json"
  JAX_PLATFORMS=cuda \
    uv run --directory "$repository" --no-sync python -m \
      benchmarks.injection_campaign.probe_likelihood \
      "$output_dir" 0 > "$candidate_probe"
  python3 - \
    "$candidate_probe" \
    "$output_dir/manifest.json" \
    "$package_manifest" \
    "$repository" \
    "$expected_gpu_count" <<'PY'
import json
import pathlib
import sys

probe = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
package = json.load(open(sys.argv[3], encoding="utf-8"))
candidate_root = pathlib.Path(sys.argv[4]).resolve()
expected_gpu_count = int(sys.argv[5])
pin = manifest.get("implementation_diagnostic")
candidate = package.get("candidate")
if not isinstance(pin, dict) or not isinstance(candidate, dict):
    raise SystemExit("candidate likelihood preflight has no implementation pin")
if (
    pin.get("implementation_label") != "candidate"
    or pin.get("implementation_revision") != candidate.get("revision")
    or pin.get("implementation_tree_sha256") != candidate.get("tree_sha256")
):
    raise SystemExit("candidate likelihood preflight implementation pin mismatch")
if probe["config_sha256"] != manifest.get("config_sha256"):
    raise SystemExit("candidate likelihood preflight config hash mismatch")
if probe["injection_id"] != 0:
    raise SystemExit("candidate likelihood preflight used the wrong injection")
if not probe["finite"]:
    raise SystemExit("candidate likelihood preflight produced a non-finite value")
if probe["backend"] != "gpu" or probe["device_count"] != expected_gpu_count:
    raise SystemExit("candidate likelihood preflight used the wrong GPU topology")
pathlib.Path(probe["jimgw_module"]).resolve().relative_to(candidate_root)
print(
    "Candidate likelihood preflight passed: "
    f"logL={probe['log_likelihood_at_truth']:.12f} "
    f"snr={probe['optimal_snr_by_detector']}"
)
PY
fi

if [[ ! -f "$output_dir/manifest.json" ]]; then
  uv run --directory "$repository" --no-sync python -m \
    benchmarks.injection_campaign.prepare_campaign \
    "$output_dir" --n-injections "$n_injections" \
    --catalogue-size "$catalogue_size" --seed "$seed"
fi

set +e
range_arguments=()
if ((${#injection_ids[@]})); then
  for injection_id in "${injection_ids[@]}"; do
    range_arguments+=(--injection-id "$injection_id")
  done
else
  range_arguments=(--start "$start")
  if [[ -n "$stop" ]]; then
    range_arguments+=(--stop "$stop")
  fi
fi
plot_arguments=()
if [[ "$plot" == true ]]; then
  plot_arguments+=(--plot)
fi
worker_arguments=(--long-lived-worker)
if [[ "$fresh_processes" == true ]]; then
  worker_arguments=()
fi
JAX_PLATFORMS=cuda uv run --directory "$repository" --no-sync python -m \
  benchmarks.injection_campaign.run_campaign \
  "$output_dir" --retry-count "$retry_count" \
  "${worker_arguments[@]}" --jax-cache-diagnostics \
  "${range_arguments[@]}" "${plot_arguments[@]}"
campaign_status=$?
set -e

# Always make the completed/partial scientific outputs downloadable. The
# persistent compilation cache is deliberately excluded because it is large
# and can be regenerated.
results_archive="${output_dir}.tar.gz"
tar -C "$(dirname "$output_dir")" \
  --exclude="$(basename "$output_dir")/.jax-cache" \
  -czf "$results_archive" "$(basename "$output_dir")"
sha256sum "$results_archive"
exit "$campaign_status"
