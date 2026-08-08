#!/usr/bin/env bash

set -euo pipefail

gpu_id="NVIDIA H200"
image="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
cloud_type="SECURE"
guard_minutes=360
launch=false

usage() {
  echo "Usage: $0 [--launch] [--gpu-id ID] [--image IMAGE] [--cloud-type TYPE] [--guard-minutes N]"
}

while (($#)); do
  case "$1" in
    --launch) launch=true; shift ;;
    --gpu-id) gpu_id="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --cloud-type) cloud_type="$2"; shift 2 ;;
    --guard-minutes) guard_minutes="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "$guard_minutes" =~ ^[1-9][0-9]*$ ]]; then
  echo "--guard-minutes must be a positive integer" >&2
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
  --name jim-swig-100-injection-campaign
  --image "$image"
  --gpu-id "$gpu_id"
  --gpu-count 4
  --cloud-type "$cloud_type"
  --min-cuda-version 12.8
  --container-disk-in-gb 40
  --volume-in-gb 30
  --volume-mount-path /workspace
  --ports 22/tcp
  --terminate-after "$terminate_at"
)

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
