#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository="$(cd "$script_dir/../../.." && pwd)"
output_root=""
result_archive=""

usage() {
  echo "Usage: $0 --output-root /workspace/NAME --result-archive /workspace/NAME.tar.gz"
}

while (($#)); do
  case "$1" in
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --result-archive)
      result_archive="$2"
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

case "$output_root" in
  /workspace/jim-xg-results/*) ;;
  *) echo "--output-root must be below /workspace/jim-xg-results" >&2; exit 2 ;;
esac
case "$result_archive" in
  /workspace/jim-xg-result-*.tar.gz) ;;
  *) echo "--result-archive must be a jim-xg-result archive below /workspace" >&2; exit 2 ;;
esac
if [[ -e "$output_root" || -e "$result_archive" || -e "${result_archive}.partial" ]]; then
  echo "Refusing to overwrite a remote XG result path" >&2
  exit 1
fi

mkdir -p "$output_root/.runpod"
export PYTHONDONTWRITEBYTECODE=1

publish_results() {
  local status="$1"
  local archive_python="python3"
  trap - EXIT
  set +e
  if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" \
    && -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    archive_python="$UV_PROJECT_ENVIRONMENT/bin/python"
  fi
  printf '%s\n' "$status" > "$output_root/.runpod/exit-status"
  if [[ ! -e "$output_root/.runpod/package-manifest.json" \
    && -f "$repository/.runpod/package-manifest.json" ]]; then
    cp "$repository/.runpod/package-manifest.json" \
      "$output_root/.runpod/package-manifest.json"
  fi
  "$archive_python" "$repository/benchmarks/xg/runpod/workflow.py" \
    manifest-result "$output_root" --exit-status "$status" >/dev/null
  if [[ "$?" -eq 0 ]]; then
    "$archive_python" "$repository/benchmarks/xg/runpod/workflow.py" \
      archive-result "$output_root" --output "${result_archive}.partial" \
      >/dev/null
    if [[ "$?" -eq 0 ]]; then
      mv "${result_archive}.partial" "$result_archive"
      sha256sum "$result_archive" > "${result_archive}.sha256"
    fi
  fi
  exit "$status"
}
trap 'publish_results $?' EXIT

cp "$repository/.runpod/package-manifest.json" \
  "$output_root/.runpod/package-manifest.json"
package_manifest="$repository/.runpod/package-manifest.json"
package_manifest_sha256="$(sha256sum "$package_manifest" | awk '{print $1}')"

verify_packaged_source() {
  local receipt="$1"
  local observed_manifest_sha256
  observed_manifest_sha256="$(sha256sum "$package_manifest" | awk '{print $1}')"
  if [[ "$observed_manifest_sha256" != "$package_manifest_sha256" ]]; then
    echo "The packaged workspace manifest changed during the XG workflow" >&2
    return 1
  fi
  python3 "$repository/benchmarks/xg/runpod/workflow.py" \
    verify-extracted "$repository" > "$receipt"
}

verify_packaged_source "$output_root/.runpod/package-verification.json"

read -r source_revision < <(
  python3 - "$package_manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["source_revision"])
PY
)

gpu_names="$output_root/.runpod/gpu-names.txt"
nvidia-smi --query-gpu=name --format=csv,noheader > "$gpu_names"
python3 - "$gpu_names" <<'PY'
import sys
from pathlib import Path

names = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if len(names) != 4:
    raise SystemExit(f"Expected exactly four GPUs, found {len(names)}: {names}")
if any("H200" not in name for name in names):
    raise SystemExit(f"Expected four H200 GPUs, found: {names}")
PY
nvidia-smi \
  --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv > "$output_root/.runpod/nvidia-smi.csv"

export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
export UV_CACHE_DIR="/root/.cache/uv"
export UV_PROJECT_ENVIRONMENT="/root/jim-xg-venv-${source_revision:0:12}"
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=10
export UV_CONCURRENT_DOWNLOADS=4
if ! command -v uv >/dev/null 2>&1 \
  || [[ "$(uv --version)" != "uv 0.11.2"* ]]; then
  curl -LsSf "https://astral.sh/uv/0.11.2/install.sh" | sh
  hash -r
fi
if [[ "$(uv --version)" != "uv 0.11.2"* ]]; then
  echo "The frozen XG runtime requires uv 0.11.2" >&2
  exit 1
fi
uv python install 3.12 2>&1 | tee "$output_root/.runpod/uv-python.log"
SETUPTOOLS_SCM_PRETEND_VERSION="0.0.dev0+g${source_revision:0:12}" \
  uv sync --directory "$repository" --frozen --extra cuda \
    --group cross-validation --group test --python 3.12 \
    2>&1 | tee "$output_root/.runpod/uv-sync.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR="/root/jim-xg-jax-cache-${source_revision:0:12}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=True
export JAX_EMBEDDED_CONSTANTS_MAX_BYTES=32
export NCCL_NVLS_ENABLE=0
export PYTHONHASHSEED=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run --directory "$repository" --no-sync python - \
  "$source_revision" "$output_root/.runpod/runtime.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import jax
import numpy
import scipy

jax.config.update("jax_enable_x64", True)
expected_revision = sys.argv[1]
devices = jax.devices()
if len(devices) != 4 or any(device.platform != "gpu" for device in devices):
    raise SystemExit(f"Expected exactly four JAX CUDA devices, found: {devices}")
version = importlib.metadata.version("jimgw")
if expected_revision[:12] not in version:
    raise SystemExit(
        f"Installed Jim version {version!r} does not bind revision {expected_revision}"
    )
payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "jimgw": version,
    "jax": jax.__version__,
    "jaxlib": importlib.metadata.version("jaxlib"),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "ripplegw": importlib.metadata.version("ripplegw"),
    "equinox": importlib.metadata.version("equinox"),
    "lalsuite": importlib.metadata.version("lalsuite"),
    "astropy": importlib.metadata.version("astropy"),
    "bilby": importlib.metadata.version("bilby"),
    "gwpy": importlib.metadata.version("gwpy"),
    "jax_enable_x64": bool(jax.config.jax_enable_x64),
    "devices": [str(device) for device in devices],
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

uv run --directory "$repository" --no-sync pytest -q -p no:cacheprovider \
  tests/unit/benchmarks/test_xg_*.py \
  tests/unit/cli/test_xg_builder.py \
  tests/unit/core/single_event/test_time_dependent_response.py \
  tests/unit/core/single_event/test_xg_cache_integration.py \
  2>&1 | tee "$output_root/.runpod/preflight-tests.log"

config="$repository/benchmarks/xg/xg-ce-4096-65536.toml"
qualification_entrypoint="$repository/benchmarks/xg/run_qualification.py"
qualification_dir="$output_root/qualification"
qualified_config="$output_root/xg-ce-4096-65536.qualified.toml"
science_output="$output_root/science-output"
mkdir "$qualification_dir"

uv run --directory "$repository" --no-sync python - "$qualification_entrypoint" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1])
compile(source.read_bytes(), str(source), "exec")
PY

timeout --foreground --signal=TERM --kill-after=2m 55m \
  uv run --directory "$repository" --no-sync python "$qualification_entrypoint" \
    --config "$config" \
    --bundle-dir "$qualification_dir" \
    --n-devices 4 \
  2>&1 | tee "$output_root/qualification.log"

# The evidence generator is not allowed to rewrite any packaged implementation,
# configuration, or oracle source before assembly.
verify_packaged_source "$output_root/.runpod/post-qualification-package.json"

timeout --foreground --signal=TERM --kill-after=30s 5m \
  uv run --directory "$repository" --no-sync jim-xg-qualify assemble \
    "$config" "$qualification_dir" \
    --output "$qualification_dir/xg-qualification.json" \
  2>&1 | tee "$output_root/qualification-assembly.log"

manifest="$qualification_dir/xg-qualification.json"
manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
uv run --directory "$repository" --no-sync python - \
  "$config" "$qualified_config" "$manifest" "$manifest_sha256" "$science_output" <<'PY'
import re
import sys
import tomllib
from pathlib import Path

import tomli_w

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
manifest = Path(sys.argv[3])
digest = sys.argv[4]
output = Path(sys.argv[5])
if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise SystemExit("Qualification manifest digest is invalid")
with source.open("rb") as stream:
    raw = tomllib.load(stream)
heterodyne = raw["likelihood"]["heterodyne"]
if "qualification_manifest" in heterodyne or "qualification_manifest_sha256" in heterodyne:
    raise SystemExit("Frozen input config already contains active qualification fields")
heterodyne["qualification_manifest"] = str(manifest.resolve())
heterodyne["qualification_manifest_sha256"] = digest
raw["output"]["dir"] = str(output.resolve())
raw["output"]["overwrite"] = False
with destination.open("wb") as stream:
    tomli_w.dump(raw, stream)
PY

# Manifest assembly and science construction must see precisely the same source,
# Python environment, and container platform. Do not install or edit anything here.
verify_packaged_source "$output_root/.runpod/pre-science-package.json"

timeout --foreground --signal=TERM --kill-after=2m 30m \
  uv run --directory "$repository" --no-sync jim-run --verbose "$qualified_config" \
  2>&1 | tee "$output_root/science-run.log"

uv run --directory "$repository" --no-sync python - \
  "$science_output" "$manifest" "$manifest_sha256" <<'PY'
import hashlib
import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
expected_manifest = Path(sys.argv[2]).resolve()
expected_manifest_sha256 = sys.argv[3]
required = ("samples.npz", "diagnostics.json", "config.final.toml")
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"Science output is missing required artifacts: {missing}")
with (root / "config.final.toml").open("rb") as stream:
    config = tomllib.load(stream)
sampler = config["sampler"]
heterodyne = config["likelihood"]["heterodyne"]
configured_manifest = Path(heterodyne.get("qualification_manifest", "")).resolve()
configured_digest = heterodyne.get("qualification_manifest_sha256")
if configured_manifest != expected_manifest or configured_digest != expected_manifest_sha256:
    raise SystemExit("Science output does not bind the qualification manifest")
observed_manifest_sha256 = hashlib.sha256(expected_manifest.read_bytes()).hexdigest()
if observed_manifest_sha256 != expected_manifest_sha256:
    raise SystemExit("Qualification manifest changed during science sampling")
if sampler.get("type") != "blackjax-swig":
    raise SystemExit("Science output used the wrong sampler")
if sampler.get("n_live") != 4096 or sampler.get("n_devices") != 4:
    raise SystemExit("Science output did not preserve the frozen live set")
if heterodyne.get("n_bins") != 65536:
    raise SystemExit("Science output did not preserve the frozen bin count")
if sampler.get("fold_symmetry") is not None:
    raise SystemExit("Science output unexpectedly used sky folding")
if config["likelihood"].get("distance_marginalization") is not None:
    raise SystemExit("Science output unexpectedly marginalized luminosity distance")
expected_blocks = [
    ["M_c", "q", "lambda_1", "lambda_2", "s1_z", "s2_z", "t_det"],
    ["ra", "dec"],
    ["psi"],
    ["cos_iota", "d_L"],
]
expected_bridge_blocks = [
    ["ra", "dec", "psi", "cos_iota", "d_L", "t_det"],
]
if sampler.get("blocks") != expected_blocks:
    raise SystemExit("Science output did not preserve the frozen primary blocks")
if sampler.get("bridge_blocks") != expected_bridge_blocks:
    raise SystemExit("Science output did not preserve the frozen bridge block")
posterior_names = set(config["prior"]) | {"log_likelihood"}
with np.load(root / "samples.npz", allow_pickle=False) as samples:
    if not samples.files:
        raise SystemExit("Science output contains no posterior fields")
    missing_fields = sorted(posterior_names - set(samples.files))
    if missing_fields:
        raise SystemExit(f"Science posterior is missing fields: {missing_fields}")
    counts = {name: samples[name].shape[0] for name in samples.files}
    if len(set(counts.values())) != 1 or next(iter(counts.values())) < 1:
        raise SystemExit(f"Science posterior lengths are invalid: {counts}")
    if any(not np.all(np.isfinite(samples[name])) for name in samples.files):
        raise SystemExit("Science posterior contains non-finite values")
diagnostics = json.loads((root / "diagnostics.json").read_text())
if not isinstance(diagnostics, dict):
    raise SystemExit("Science diagnostics are invalid")


def require_finite_json(value, location="diagnostics"):
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_json(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_finite_json(child, f"{location}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise SystemExit(f"Science diagnostics are non-finite at {location}")


require_finite_json(diagnostics)
PY

verify_packaged_source "$output_root/.runpod/post-science-package.json"
if [[ "$(sha256sum "$manifest" | awk '{print $1}')" != "$manifest_sha256" ]]; then
  echo "The qualification manifest changed before result publication" >&2
  exit 1
fi
touch "$output_root/.runpod/complete"
