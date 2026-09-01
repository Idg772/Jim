#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$script_dir/../../.." && pwd)"
revision="$(git -C "$repository" rev-parse --verify HEAD^{commit})"
output="${1:-/private/tmp/jim-xg-${revision:0:12}.tar.gz}"

python3 "$script_dir/workflow.py" package \
  --repository "$repository" \
  --output "$output"
