# FSM scheduler GPU validation

This runbook is the release gate for the finite-state-machine slice scheduler.
Run it on one four-H200 Secure Cloud pod with seed 0. The statistical,
collective, memory, and timing gates below must all pass before the GPU gate in
`docs/development/replicated_live_state_nested_sampling.md` is marked complete.

All frozen references in this document are for the historical
`aligned-11d` workload. Every scientific command pins that selector explicitly.
They do **not** validate `paper-15d`: the 15D workload has 15 slices (11
waveform-rebuild and 4 cache-hit slices across 7 blocks) and requires fresh
counter, evidence, timing, and memory references.

The paired run deliberately does **not** request per-slice diagnostics. The
pre-FSM baseline builder at `1c05a7b7` has no `per_slice_info` keyword, whereas
the current benchmark obtains those counters from the production FSM builder.
Collect the candidate counters in the separate direct run below and compare
them with the retained local reference artifact.

## Frozen references and acceptance thresholds

| Item | Required value |
| --- | --- |
| Pre-FSM baseline | `1c05a7b745082e4ca79a982ad34a65987743a7b6` |
| Analysis window | historical `aligned-11d`, trigger -126 s to +2 s |
| Frozen input SHA-256 | `3db1c5f97add6dc4dc1533014dbc790447c45fdace82f5671322043758c5cd1a` |
| Local reference counter artifact | `benchmark-results/device-parallel-nss/y3bj26f7xqys6e-four-gpu/per-slice/00-01-ours-g4-seed0.npz` |
| Reference counter artifact SHA-256 | `2a0467f797e3469801bafed300b4c26c5bab40fa36b0573a6ae3d9ad4bb0afc7` |
| Counter shape | `(16768, 11)` (`262 * 64` chains by 11 slices) |
| Expansion / shrink totals | `208865` / `918178` |
| Outer iterations | `262` |
| Likelihood evaluations | `1127043` |
| Reference log Z | `606.5114049113298` |
| Maximum allowed absolute log-Z difference | `3.2e-11` |
| Pre-FSM steady outer-step median | `1.286605888 s` |
| FSM steady outer-step gate | at most `1.03 s` (at least `1.25x`) |
| Previous candidate steady-window framebuffer maximum | `7419 MiB` |

The `3.2e-11` log-Z tolerance is the largest difference recorded by the prior
paired repository run. It replaces the unsupported `3e-13` figure in the
original implementation plan.

## 1. Local preflight and packaging

The comparison scripts export implementations with `git archive`; staged or
uncommitted implementation changes are not benchmarked. Commit the candidate
and verify that it differs from the baseline before packaging:

```bash
baseline=1c05a7b745082e4ca79a982ad34a65987743a7b6
candidate="$(git rev-parse HEAD)"
test "$candidate" != "$baseline"
git diff --check
git diff --quiet
git diff --cached --quiet
test -z "$(git ls-files --others --exclude-standard -- src/jimgw)"
git cat-file -e "$candidate:src/jimgw/samplers/blackjax/_fsm.py"

uv run pytest tests/unit/samplers/blackjax/test_fsm.py -v
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
uv run pytest \
  tests/unit/samplers/blackjax/test_sharding.py \
  tests/integration/test_sampler_sharding.py \
  tests/unit/benchmarks/ \
  -v

archive="$(bash benchmarks/device_parallel_nss/runpod/package_workspace.sh)"
printf '%s\n' "$archive"
```

Inspect the provision command and its price before launching:

```bash
bash benchmarks/device_parallel_nss/runpod/provision.sh
bash benchmarks/device_parallel_nss/runpod/provision.sh --launch
```

Do not invoke `upload_and_run.py` for this gate: that helper dispatches the
broader `run_on_pod.sh` study, whose paired comparison still defaults to the
paper baseline rather than `1c05a7b7`. Instead, obtain the connection fields
and transfer the archive directly (replace the placeholders with the
connection values printed by `runpodctl ssh info` and the archive path printed
by `package_workspace.sh`):

```bash
pod_id=<pod-id>
runpodctl ssh info "$pod_id"

ssh_target=<user@host>
ssh_port=<port>
ssh_identity=<absolute-private-key-path>
local_archive=<archive-path-printed-by-package_workspace.sh>

scp -P "$ssh_port" -i "$ssh_identity" \
  "$local_archive" "$ssh_target:/workspace/jim-fsm-workspace.tar.gz"
ssh -p "$ssh_port" -i "$ssh_identity" "$ssh_target" \
  'mkdir -p /workspace/Jim && tar --no-same-owner -xzf /workspace/jim-fsm-workspace.tar.gz -C /workspace/Jim'
```

## 2. Prepare the pod

From the extracted repository on the pod:

```bash
repository="$(pwd)"
validation_root="/workspace/jim-fsm-validation/$(date -u +%Y%m%dT%H%M%SZ)"
baseline=1c05a7b745082e4ca79a982ad34a65987743a7b6
candidate="$(git -C "$repository" rev-parse HEAD)"

git -C "$repository" diff --quiet
git -C "$repository" diff --cached --quiet
test -z "$(git -C "$repository" ls-files --others --exclude-standard -- src/jimgw)"
git -C "$repository" cat-file -e "$candidate:src/jimgw/samplers/blackjax/_fsm.py"

export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export NCCL_NVLS_ENABLE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.11.2/install.sh | sh
fi
uv python install 3.11
uv sync --directory "$repository" --frozen --extra cuda --group test --python 3.11

JAX_PLATFORMS=cuda uv run --directory "$repository" --no-sync python - <<'PY'
import jax

devices = jax.devices()
print(f"jax={jax.__version__} backend={jax.default_backend()} devices={devices}")
if jax.default_backend() != "gpu" or len(devices) != 4:
    raise SystemExit("JAX did not initialise all four GPUs")
PY
```

## 3. Paired timing and profile run

Run the pre-FSM revision and the committed FSM candidate against the same
freshly frozen data, seed, host, and device topology. Keep per-slice output off
for this pair so the historical builder remains runnable.

```bash
paired_dir="$validation_root/paired-four-gpu"

uv run --directory "$repository" --no-sync python \
  benchmarks/device_parallel_nss/compare_gw170817_full_run.py \
  --repository "$repository" \
  --baseline-ref "$baseline" \
  --candidate-ref "$candidate" \
  --baseline-label pre-fsm \
  --candidate-label fsm \
  --workload aligned-11d \
  --output-dir "$paired_dir" \
  --seeds 0 \
  --profile-dir "$validation_root/profiles" \
  --profile-warmup-steps 10 \
  --profile-steps 15 \
  --telemetry-dir "$validation_root/telemetry"

sha256sum "$paired_dir/data/gw170817.npz"
```

The data hash must equal the frozen-input hash in the table above. Stop if it
does not; iteration, counter, and evidence comparisons would not be valid.

## 4. Candidate-only per-slice run

Reuse the exact paired input and run only the FSM candidate with production
per-slice diagnostics enabled:

```bash
mkdir -p \
  "$validation_root/per-slice" \
  "$validation_root/candidate-diagnostics" \
  "$validation_root/gpu-hlo"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
JAX_PLATFORMS=cuda \
JAX_ENABLE_COMPILATION_CACHE=0 \
XLA_FLAGS="--xla_dump_to=$validation_root/gpu-hlo --xla_dump_hlo_as_text" \
uv run --directory "$repository" --no-sync python \
  benchmarks/device_parallel_nss/benchmark_gw170817_full_run.py \
  --data-file "$paired_dir/data/gw170817.npz" \
  --workload aligned-11d \
  --seed 0 \
  --n-devices 4 \
  --implementation-root "$repository" \
  --implementation-label fsm \
  --implementation-revision "$candidate" \
  --slice-data-output "$validation_root/per-slice/fsm-g4-seed0.npz" \
  --output "$validation_root/candidate-diagnostics/fsm-g4-seed0.json"

sha256sum "$validation_root/per-slice/fsm-g4-seed0.npz"
```

The NPZ hash should equal the reference artifact hash in the table. The
array-level comparison below is authoritative if ZIP-container metadata ever
makes the file hashes differ.

After downloading the result bundle to the original repository, compare the
counter arrays directly:

```bash
uv run python - \
  benchmark-results/device-parallel-nss/y3bj26f7xqys6e-four-gpu/per-slice/00-01-ours-g4-seed0.npz \
  /path/to/downloaded/fsm-g4-seed0.npz <<'PY'
from pathlib import Path
import sys

import numpy as np

reference_path, candidate_path = map(Path, sys.argv[1:])
with np.load(reference_path, allow_pickle=False) as reference, np.load(
    candidate_path, allow_pickle=False
) as candidate:
    for name in ("num_expansions", "num_shrink"):
        expected = np.asarray(reference[name])
        actual = np.asarray(candidate[name])
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise SystemExit(
                f"{name}: expected {expected.dtype} {expected.shape}, "
                f"got {actual.dtype} {actual.shape}"
            )
        if expected.tobytes(order="C") != actual.tobytes(order="C"):
            raise SystemExit(f"{name}: arrays are not byte-identical")
        print(f"{name}: byte-identical {actual.shape} total={int(actual.sum())}")
PY
```

## 5. Statistical and timing gates

Quoted post-JIT sampling times follow `timing_seconds.paper_convention` (see
`benchmarks/device_parallel_nss/README.md`, "Timing conventions"); state cold
vs warm cache for every quoted number.

Check both paired reports and the candidate-only diagnostic report:

```bash
uv run --directory "$repository" --no-sync python - \
  "$paired_dir/raw" \
  "$validation_root/candidate-diagnostics/fsm-g4-seed0.json" <<'PY'
import json
from pathlib import Path
import statistics
import sys

from benchmarks.device_parallel_nss.summarize_gw170817_diagnostics import (
    _steady_samples,
)

raw_dir = Path(sys.argv[1])
diagnostic_path = Path(sys.argv[2])
reports = [
    json.loads(path.read_text())
    for path in sorted(raw_dir.glob("*.json"))
    if path.name.startswith("00-")
]
reports.append(json.loads(diagnostic_path.read_text()))

reference_logz = 606.5114049113298
for report in reports:
    label = report["implementation"]["label"]
    results = report["results"]
    assert results["n_iterations"] == 262, (label, results["n_iterations"])
    assert results["n_likelihood_evaluations"] == 1_127_043, (
        label,
        results["n_likelihood_evaluations"],
    )
    difference = abs(results["log_Z"] - reference_logz)
    assert difference <= 3.2e-11, (label, results["log_Z"], difference)
    print(label, "iterations/evals/logZ", results["n_iterations"],
          results["n_likelihood_evaluations"], results["log_Z"], difference)

paired = reports[:2]
medians = {
    report["implementation"]["label"]: statistics.median(_steady_samples(report))
    for report in paired
}
candidate_median = medians["fsm"]
speedup = medians["pre-fsm"] / candidate_median
print("steady medians", medians, "speedup", speedup)
assert candidate_median <= 1.03, candidate_median
PY
```

The absolute `1.03 s` gate protects against a coincidentally slow or fast
paired baseline. Report the same-host paired speedup as context; it is not a
second hard gate because host-to-host variance can move the freshly measured
baseline even when the candidate meets the declared historical threshold.

## 6. HLO and collective census

Run the repository's fixed-state structural benchmark against both revisions:

```bash
uv run --directory "$repository" --no-sync python \
  benchmarks/device_parallel_nss/compare_outer_step.py \
  --repository "$repository" \
  --baseline-ref "$baseline" \
  --candidate-ref "$candidate" \
  --baseline-label pre-fsm \
  --candidate-label fsm \
  --output-dir "$validation_root/collective-census" \
  --device-counts 4 \
  --repeats 1 \
  --n-live 512 \
  --n-delete 64 \
  --dims 11 \
  --inner-steps-per-dim 1 \
  --warmup 2 \
  --iterations 5 \
  --dtype float64

python - "$validation_root/collective-census/summary.json" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
counts = summary["groups"]["fsm"]["4"]["collective_counts"]
print(counts)
assert counts["all-gather"] == [1]
assert counts["all-reduce"] == [0]
assert counts["collective-permute"] == [0]
PY
```

The candidate-only run in step 4 also dumped the real-GPU HLO. Locate the
compiled outer-step module and retain a concise loop/collective extract with
the result bundle:

```bash
find "$validation_root/gpu-hlo" -name '*after_optimizations.txt' -print0 \
  | xargs -0 grep -l 'all-gather' \
  | tee "$validation_root/gpu-hlo/outer-step-modules.txt"
test -s "$validation_root/gpu-hlo/outer-step-modules.txt"

while IFS= read -r hlo; do
  printf '\n===== %s =====\n' "$hlo"
  grep -nE 'while|all-gather|all-reduce|collective-permute' "$hlo" || true
done < "$validation_root/gpu-hlo/outer-step-modules.txt" \
  > "$validation_root/gpu-hlo/outer-step-census.txt"
```

Inspect the selected GPU module, not merely the CPU StableHLO. The production
GW170817 configuration has one Gibbs sweep, so require two likelihood-bearing
scheduler whiles (one rebuild segment and one cache-hit segment), no nested
slice whiles inside either scheduler body, and no waveform-rebuild payload in
the cache-hit body. Incidental PRNG and linear-algebra whiles are expected and
must not be mistaken for scheduler loops. The CPU StableHLO test remains the
machine-checked structural reference: its two-sweep fixture has exactly four
top-level scheduler loops.

## 7. Profile and memory review

The harness captures `nvidia-smi dmon -s pucvmet` once per second during the
15-step profile window. Its `fb` column is framebuffer memory in MiB. Record the
maximum for every candidate GPU and require the overall sampled maximum to be
at most `7419 MiB`:

```bash
python - "$validation_root/telemetry/00-01-fsm-g4-seed0.dmon" <<'PY'
from collections import defaultdict
from pathlib import Path
import sys

path = Path(sys.argv[1])
header = None
maximum = defaultdict(int)
for raw_line in path.read_text(errors="replace").splitlines():
    line = raw_line.strip()
    if line.startswith("#"):
        candidate = line.lstrip("# ").split()
        if "gpu" in candidate and "fb" in candidate:
            header = candidate
    elif line and header is not None:
        fields = line.split()
        if len(fields) == len(header):
            row = dict(zip(header, fields, strict=True))
            maximum[row["gpu"]] = max(maximum[row["gpu"]], int(row["fb"]))
print("framebuffer MiB maxima", dict(maximum))
assert maximum and max(maximum.values()) <= 7419
PY
```

This is a one-second telemetry proxy, not an instantaneous allocator peak. The
checked-in diagnostics summarizer currently reports SM activity, memory-bus
activity, and power but does not parse `fb`; record this limitation with the
result.

Open both generated JAX traces in Perfetto or TensorBoard and report the
profiled-window decomposition for fused kernels, NCCL, FFT, and memcpy. There
is currently no checked-in script that computes per-kernel means or a launch
census from the trace JSON, so this part of the gate is a manual review. If the
FSM median falls between `1.03 s` and `1.29 s`, extract and record the rebuild
and cache-hit scheduler-kernel means before diagnosing scheduler-tick overhead.
Compare them with the calibrated pre-FSM launch costs (`7.95 ms` per rebuild
launch and `6.39 ms` per cache-hit launch); the performance gate still remains
failed above `1.03 s`.

## 8. Archive, download, and close the pod

```bash
archive="${validation_root}.tar.gz"
tar -C "$(dirname "$validation_root")" -czf "$archive" \
  "$(basename "$validation_root")"
sha256sum "$archive"
printf 'Download bundle: %s\n' "$archive"
```

Download and verify the archive with the same SSH fields used during upload,
then delete the pod using the provisioning workflow's pod ID:

```bash
remote_archive=<absolute-path-printed-by-step-8>
local_results=benchmark-results/device-parallel-nss/fsm-validation.tar.gz
mkdir -p "$(dirname "$local_results")"
scp -P "$ssh_port" -i "$ssh_identity" \
  "$ssh_target:$remote_archive" "$local_results"
sha256sum "$local_results"
runpodctl pod delete "$pod_id"
```

On a complete pass, update the development document with the measured values
and mark its GPU gates complete.
