#!/usr/bin/env bash

set -euo pipefail

image="runpod/pytorch@sha256:60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f"
gpu_id="NVIDIA H200"
gpu_count=4
guard_minutes=120
max_gpu_cost_usd=40.0
minimum_balance_reserve_usd=10.0
pod_name="jim-xg-ce-4096-65536"
archive=""
receipt=""
launch=false
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $0 [--launch --archive PATH] [--receipt PATH] [--name NAME]"
}

while (($#)); do
  case "$1" in
    --launch)
      launch=true
      shift
      ;;
    --name)
      if (($# < 2)); then
        echo "--name requires a value" >&2
        exit 2
      fi
      pod_name="$2"
      shift 2
      ;;
    --archive)
      if (($# < 2)); then
        echo "--archive requires a value" >&2
        exit 2
      fi
      archive="$2"
      shift 2
      ;;
    --receipt)
      if (($# < 2)); then
        echo "--receipt requires a value" >&2
        exit 2
      fi
      receipt="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "$pod_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "--name must contain only letters, digits, dots, underscores, and hyphens" >&2
  exit 2
fi

read -r launch_started_at terminate_at < <(python3 - "$guard_minutes" <<'PY'
from datetime import datetime, timedelta, timezone
import sys

started = datetime.now(timezone.utc).replace(microsecond=0)
deadline = started + timedelta(minutes=int(sys.argv[1]))
timestamp = lambda value: value.isoformat().replace("+00:00", "Z")
print(timestamp(started), timestamp(deadline))
PY
)

command=(
  runpodctl pod create
  --name "$pod_name"
  --image "$image"
  --gpu-id "$gpu_id"
  --gpu-count "$gpu_count"
  --cloud-type SECURE
  --min-cuda-version 12.8
  --container-disk-in-gb 40
  --volume-in-gb 30
  --volume-mount-path /workspace
  --ports 22/tcp
  --terminate-after "$terminate_at"
  --wait
  --wait-timeout 10m
)

printf 'Immutable image: %s\n' "$image" >&2
printf 'Automatic deletion deadline: %s (%d minutes)\n' "$terminate_at" "$guard_minutes" >&2
printf 'Command:' >&2
printf ' %q' "${command[@]}" >&2
printf '\n' >&2

if [[ "$launch" != true ]]; then
  echo "Dry run only. Re-run with --launch after inspecting the command." >&2
  exit 0
fi

if [[ -z "$archive" ]]; then
  echo "--launch requires the immutable --archive that will be uploaded" >&2
  exit 2
fi
python3 "$script_dir/workflow.py" verify-package "$archive" >/dev/null
workspace_sha256="$(sha256sum "$archive" | awk '{print $1}')"
receipt="$(python3 - "$archive" "$receipt" <<'PY'
import sys
from pathlib import Path

archive = Path(sys.argv[1]).expanduser().resolve()
requested = sys.argv[2]
receipt = (
    Path(requested).expanduser().resolve()
    if requested
    else Path(f"{archive}.runpod-launch.json")
)
candidate = receipt.with_name(receipt.name + ".candidate")
if not receipt.parent.is_dir():
    raise SystemExit(f"Launch receipt directory does not exist: {receipt.parent}")
if receipt.exists() or receipt.is_symlink() or candidate.exists() or candidate.is_symlink():
    raise SystemExit(f"Refusing to overwrite a launch receipt: {receipt}")
print(receipt)
PY
)"

if [[ "$(runpodctl version)" != runpodctl\ 2.11.0-* ]]; then
  echo "This frozen launcher requires runpodctl 2.11.0" >&2
  exit 1
fi

gpu_json="$(runpodctl gpu list --output json)"
account_json="$(runpodctl user --output json)"
python3 - \
  "$gpu_json" \
  "$account_json" \
  "$gpu_id" \
  "$gpu_count" \
  "$guard_minutes" \
  "$max_gpu_cost_usd" \
  "$minimum_balance_reserve_usd" <<'PY'
import json
import math
import sys

gpu_rows = json.loads(sys.argv[1])
account = json.loads(sys.argv[2])
gpu_id = sys.argv[3]
gpu_count = int(sys.argv[4])
guard_minutes = int(sys.argv[5])
maximum_cost = float(sys.argv[6])
reserve = float(sys.argv[7])

matches = [row for row in gpu_rows if row.get("gpuId") == gpu_id]
if len(matches) != 1 or matches[0].get("available") is not True:
    raise SystemExit(f"The frozen GPU type is unavailable: {gpu_id}")
unit_price = matches[0].get("securePricePerHr")
if isinstance(unit_price, bool) or not isinstance(unit_price, (int, float)):
    raise SystemExit("Runpod did not report a secure-cloud H200 price")
projected = float(unit_price) * gpu_count * guard_minutes / 60.0
if not math.isfinite(projected) or projected > maximum_cost:
    raise SystemExit(
        f"Projected GPU ceiling ${projected:.2f} exceeds ${maximum_cost:.2f}"
    )
balance = account.get("clientBalance")
if isinstance(balance, bool) or not isinstance(balance, (int, float)):
    raise SystemExit("Runpod did not report the account balance")
if float(balance) - projected < reserve:
    raise SystemExit(
        f"Balance would fall below the ${reserve:.2f} reserve at the hard deadline"
    )
current_spend = account.get("currentSpendPerHr")
if isinstance(current_spend, bool) or not isinstance(current_spend, (int, float)):
    raise SystemExit("Runpod did not report the current hourly spend")
if float(current_spend) != 0.0:
    raise SystemExit("Another paid Runpod resource is active; refusing an ambiguous cap")
print(f"Maximum H200 GPU charge before automatic deletion: ${projected:.2f}")
PY

pod_payload="$("${command[@]}")"
python3 - \
  "$pod_payload" \
  "$receipt" \
  "$launch_started_at" \
  "$terminate_at" \
  "$image" \
  "$gpu_id" \
  "$gpu_count" \
  "$guard_minutes" \
  "$workspace_sha256" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

pod = json.loads(sys.argv[1])
receipt = Path(sys.argv[2])
pod_id = pod.get("id")
if not isinstance(pod_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", pod_id) is None:
    raise SystemExit("Runpod did not return a valid pod id; the deletion guard remains active")
payload = {
    "schema_version": 1,
    "kind": "jim-xg-runpod-launch",
    "pod_id": pod_id,
    "image": sys.argv[5],
    "gpu_id": sys.argv[6],
    "gpu_count": int(sys.argv[7]),
    "guard_minutes": int(sys.argv[8]),
    "launch_started_at": sys.argv[3],
    "terminate_after": sys.argv[4],
    "workspace_archive_sha256": sys.argv[9],
}
candidate = receipt.with_name(receipt.name + ".candidate")
try:
    candidate.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.link(candidate, receipt)
finally:
    candidate.unlink(missing_ok=True)
print(receipt)
PY
