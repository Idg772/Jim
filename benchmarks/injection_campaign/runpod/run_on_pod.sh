#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$script_dir/../../.." && pwd)"
output_dir="/workspace/jim-injection-campaign"
n_injections=100
seed=260728265
retry_count=2
start=0
stop=""

usage() {
  echo "Usage: $0 [--output-dir PATH] [--n-injections N] [--seed N] [--retry-count N] [--start N] [--stop N]"
}

while (($#)); do
  case "$1" in
    --output-dir) output_dir="$2"; shift 2 ;;
    --n-injections) n_injections="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --retry-count) retry_count="$2"; shift 2 ;;
    --start) start="$2"; shift 2 ;;
    --stop) stop="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in "$n_injections" "$retry_count" "$start"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "counts must be non-negative integers" >&2
    exit 2
  fi
done
if [[ "$n_injections" -lt 1 ]]; then
  echo "--n-injections must be positive" >&2
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

export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
# H200's NVLS path is not supported by this JAX/NCCL workload, and the
# injection is built on GPU 0 before the live set is sharded. Avoid reserving
# most of GPU 0 up front so all four devices retain room for the sampler.
export NCCL_NVLS_ENABLE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

gpu_count="$(nvidia-smi -L | awk '/^GPU / {count++} END {print count + 0}')"
if [[ "$gpu_count" -ne 4 ]]; then
  echo "Expected exactly four GPUs, but nvidia-smi found $gpu_count" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.11.2/install.sh | sh
fi
uv python install 3.11
uv sync --directory "$repository" --frozen --extra cuda \
  --group cross-validation --python 3.11

JAX_PLATFORMS=cuda uv run --directory "$repository" --no-sync python - <<'PY'
import jax

devices = jax.devices()
if len(devices) != 4 or any(device.platform != "gpu" for device in devices):
    raise SystemExit(f"Expected four JAX GPUs, got: {devices}")
print("JAX devices:", devices)
PY

if [[ ! -f "$output_dir/manifest.json" ]]; then
  uv run --directory "$repository" --no-sync python -m \
    benchmarks.injection_campaign.prepare_campaign \
    "$output_dir" --n-injections "$n_injections" --seed "$seed"
fi

set +e
range_arguments=(--start "$start")
if [[ -n "$stop" ]]; then
  range_arguments+=(--stop "$stop")
fi
JAX_PLATFORMS=cuda uv run --directory "$repository" --no-sync python -m \
  benchmarks.injection_campaign.run_campaign \
  "$output_dir" --retry-count "$retry_count" "${range_arguments[@]}" --plot
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
