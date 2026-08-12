#!/usr/bin/env bash

set -euo pipefail

gpu_id="NVIDIA H200"
image="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
cloud_type="SECURE"
min_cuda_version="12.8"
data_center_ids=""
guard_minutes=720
# Preserve the historical --gpu-count 4 default while allowing diagnostic
# campaigns to request an exact smaller topology explicitly.
gpu_count=4
pod_name="jim-swig-100-injection-campaign"
launch=false

usage() {
  echo "Usage: $0 [--launch] [--name NAME] [--gpu-id ID] [--gpu-count N] [--image IMAGE] [--cloud-type TYPE] [--min-cuda-version VERSION] [--data-center-ids IDS] [--guard-minutes N]"
}

while (($#)); do
  case "$1" in
    --launch) launch=true; shift ;;
    --name) pod_name="$2"; shift 2 ;;
    --gpu-id) gpu_id="$2"; shift 2 ;;
    --gpu-count) gpu_count="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --cloud-type) cloud_type="$2"; shift 2 ;;
    --min-cuda-version) min_cuda_version="$2"; shift 2 ;;
    --data-center-ids) data_center_ids="$2"; shift 2 ;;
    --guard-minutes) guard_minutes="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "$guard_minutes" =~ ^[1-9][0-9]*$ ]]; then
  echo "--guard-minutes must be a positive integer" >&2
  exit 2
fi
if ! [[ "$gpu_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "--gpu-count must be a positive integer" >&2
  exit 2
fi
if ! [[ "$min_cuda_version" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "--min-cuda-version must look like MAJOR.MINOR" >&2
  exit 2
fi
if [[ -n "$data_center_ids" && ! "$data_center_ids" =~ ^[A-Za-z0-9-]+(,[A-Za-z0-9-]+)*$ ]]; then
  echo "--data-center-ids must be a comma-separated Runpod data-center list" >&2
  exit 2
fi
if ! [[ "$pod_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "--name must contain only letters, digits, dots, underscores, and hyphens" >&2
  exit 2
fi

terminate_at="$({ python3 - "$guard_minutes" <<'PY'
from datetime import datetime, timedelta, timezone
import sys

deadline = datetime.now(timezone.utc) + timedelta(minutes=int(sys.argv[1]))
print(deadline.replace(microsecond=0).isoformat().replace("+00:00", "Z"))
PY
} )"

command=(
  runpodctl pod create
  --name "$pod_name"
  --image "$image"
  --gpu-id "$gpu_id"
  --gpu-count "$gpu_count"
  --cloud-type "$cloud_type"
  --min-cuda-version "$min_cuda_version"
  --container-disk-in-gb 40
  --volume-in-gb 30
  --volume-mount-path /workspace
  --ports 22/tcp
  --terminate-after "$terminate_at"
)
if [[ -n "$data_center_ids" ]]; then
  command+=(--data-center-ids "$data_center_ids")
fi

printf 'Cost-guard deadline: %s\n' "$terminate_at" >&2
printf 'Command:' >&2
printf ' %q' "${command[@]}" >&2
printf '\n' >&2

if [[ "$launch" != true ]]; then
  echo "Dry run only. Re-run with --launch after configuring Runpod and SSH." >&2
  exit 0
fi

runpodctl user >/dev/null
"${command[@]}"
