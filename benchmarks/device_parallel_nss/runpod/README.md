# RunPod benchmark: GW170817 sharding diagnostics

This workflow compares the paper-style sharded-live-state implementation at
`86335bdb1e7ef6191937dd17b2ca53edbb1d899f` with the current replicated executor,
then measures the candidate on one, two, and four visible GPUs. It supports the
historical aligned-spin workload and the paper's full 15-parameter GW170817
model; the historical workload remains the default.

## What it measures

The benchmark freezes one workload-selected public GW170817 input and performs
four converged BlackJAX SwiG analyses plus one short cache probe:

| Run | Seed | Implementation | Visible GPUs | Purpose |
| ---: | ---: | --- | ---: | --- |
| 1 | 0 | Paper baseline | 4 | Profile, timings, per-slice counters, telemetry |
| 2 | 0 | Candidate | 4 | Profile, timings, per-slice counters, telemetry |
| 3 | 0 | Candidate | 4 | One-step fresh-process persistent-cache hit |
| 4 | 0 | Candidate | 1 | Device-scaling control |
| 5 | 0 | Candidate | 2 | Device-scaling control |

The candidate's four-GPU point from run 2 completes the 1/2/4-GPU sweep. A
separate one-GPU microbenchmark measures one real vmap'd likelihood slot at
16, 32, 64, and 128 lanes for waveform-rebuild and cache-hit paths.

Both workloads use H1/L1/V1, 128 seconds, 20--(2048-1/128) Hz, float64, 512
live points, 64 deletions, one Gibbs sweep, and the paper-equivalent stopping
value `termination_dlogz=0.0485873516`. The upper edge is the final bin of the
fixed 4096 Hz analysis and time-marginalization grid. Within a
selected workload, the data and PSD arrays are downloaded once, stored in one
NPZ file, hashed, and reused byte-for-byte by both revisions. Analysis strain is
pinned to GWOSC's glitch-mitigated `GW170817-v2` product. The historical
`aligned-11d` arm stays at 4096 Hz; `paper-15d` uses the native 16384 Hz
`LOSC_CLN_16_V1` product while retaining the exact 20--2048 likelihood grid.
Its native windowed FFT is projected to the fixed analysis grid only after the
transform, rather than being time-domain decimated.
For `paper-15d`, the median-Welch PSD uses that same CLN event file's complete
1778.43-second pre-analysis prefix; SciPy supplies the standard median bias
correction. A 1 Hz running log-median removes finite-segment bin noise while
preserving bins above twice the local trend as instrumental lines. The
historical aligned workload retains its run-wide O2 PSD without this
post-processing.
Native-array preparation and setup are larger, but the timed sampler kernel's
frequency slice, time-correlation FFT, and padding remain unchanged.
Bundle preparation also whitens L1 with the stored PSD and rejects the
documented pre-merger glitch window if its peak exceeds the event-specific
guard. Each implementation/seed combination runs in a fresh process.

## Workload boundary

`--workload aligned-11d` is the default and preserves the previously measured
benchmark exactly: real GW170817 data, aligned-spin `IMRPhenomD_NRTidalv2`, 11
sampled parameters, six SwiG blocks, and the historical strain interval ending
two seconds after the trigger.

`--workload paper-15d` selects the paper's full-resolution GW170817 analysis:
the precessing-tidal `IMRPhenomPv2_NRTidalv2` model, 15 sampled parameters, the
paper's event-specific priors, time/phase marginalization, and seven SwiG
blocks. It follows the paper's stated real-event window literally, centring the
128-second strain segment on the trigger. Because Ripple does not publish that
combined approximant, this checkout provides a benchmark-local composition of
its Pv2 and NRTidalv2 primitives. An independent LALSimulation parity test
guards the composition. The RunPod preflight installs the cross-validation
dependencies and requires this test to pass before starting any paid scientific
run.

The `paper-15d` option reproduces the model used for the paper's real GW170817
timings; it does not recreate the unreleased 1,000-event synthetic injection
catalogue, its design-sensitivity noise realizations, or the paper's heterodyned
likelihood. The aligned and paper modes intentionally use different public
strain windows and preceding PSD intervals; a manifest mismatch prevents
cross-mode artifact reuse. Within either mode, baseline and candidate share the
same frozen bytes and exercise the full-resolution likelihood. Keep these
boundaries in any performance claim, and do not claim exact posterior or
evidence reproduction without the authors' unreleased preprocessing details.

## Hardware and cost guard

Use one Secure Cloud pod with four `NVIDIA H200` SXM GPUs. At the live check on
3 August 2026, four-card CUDA 12.8 capacity was low and cost $17.56/hour in
total. The observed aligned-11d end-to-end run used 65.9 billed minutes and cost
$19.28; allow roughly 60--80 minutes and $18--24 for that workload. The
provisioner applies an 80-minute automatic deletion deadline, giving a $23.41
GPU ceiling at that historical price. The paper-15d end-to-end cost has not yet
been characterized; first run a bounded smoke test and set `--guard-minutes`
deliberately before a full run. Recheck capacity and price immediately before
launch.

## Observed aligned-11d result (3 August 2026)

On four H200 SXM GPUs, the three-seed median `sample_call` time fell from
521.419 seconds for the paper-style baseline to 450.282 seconds for the
candidate. That is a 1.158x speedup, or 13.64% less sampler time. Paired
speedups were 1.135x, 1.151x, and 1.158x (geometric mean 1.148x).

For each shared seed, the two revisions produced the same iteration count and
likelihood-evaluation count. Their log-evidence values agreed within 3.2e-11
and their reported ESS values agreed within 4.1e-8.
This confirms that the timing improvement did not come from doing less sampler
work or changing the stopping trajectory.

## Authentication

The SSH orchestrator uses `runpodctl`, so configure its API key outside this
repository and register a public SSH key:

```bash
export RUNPOD_API_KEY='...'
runpodctl user
runpodctl ssh list-keys
```

If necessary:

```bash
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub
```

## Package, launch, and run

Create the minimized workspace archive:

```bash
bash benchmarks/device_parallel_nss/runpod/package_workspace.sh
```

The archive excludes `.git`, ignored outputs, and unrelated working-tree files.
It contains only the current candidate source and benchmark prerequisites, an
exported `src/jimgw` snapshot of the pinned paper baseline, and a machine-readable
`.runpod/package-manifest.json` with both revisions and per-file/tree hashes. The
upload helper validates the complete inventory and rejects unsafe paths, links,
device members, or any `.git` component before contacting a pod.

Inspect the exact launch command without creating a pod:

```bash
bash benchmarks/device_parallel_nss/runpod/provision.sh
```

Launch after confirming the live price and account balance:

```bash
bash benchmarks/device_parallel_nss/runpod/provision.sh --launch
```

Then upload the printed archive, execute one candidate analysis over SSH, and
download the result bundle:

```bash
python benchmarks/device_parallel_nss/runpod/upload_and_run.py \
  <pod-id> /private/tmp/jim-nss-benchmark-<revision>.tar.gz \
  --candidate-only
```

The workload remains backwards-compatible with `aligned-11d`; select the full
paper model explicitly:

```bash
python benchmarks/device_parallel_nss/runpod/upload_and_run.py \
  <pod-id> /private/tmp/jim-nss-benchmark-<revision>.tar.gz \
  --workload paper-15d \
  --candidate-only
```

The minimized package requires either `--candidate-only` or
`--original-sharded-only`. The legacy paired comparison, device sweep, and lane
sweep remain available when `run_on_pod.sh` is invoked from a full Git checkout.

For one four-GPU candidate run with no baseline or device sweep, retain the
profiler trace, complete equally weighted posterior sample set, and text GPU
HLO dumps with:

```bash
python benchmarks/device_parallel_nss/runpod/upload_and_run.py \
  <pod-id> /private/tmp/jim-nss-benchmark-<revision>.tar.gz \
  --workload paper-15d \
  --candidate-only
```

This mode writes `posterior/candidate-paper-15d-g4-seed0.npz`, containing all
15 parameters in prior space plus `log_likelihood`, and records its SHA-256,
fields, dtypes, and sample count in the JSON report. It writes the compiler
output under `gpu-hlo/candidate-paper-15d-g4-seed0/`, including an
`hlo-manifest.json` with a SHA-256 and byte count for every dumped file. The
runner requires at least one optimized text HLO module and verifies the NPZ,
HLO, and profiler artifacts before packaging the result bundle.

To run the same candidate FSM benchmark and retain the complete profiling,
telemetry, per-slice, and posterior pipeline without compiler dumps, add
`--no-hlo`:

```bash
python benchmarks/device_parallel_nss/runpod/upload_and_run.py \
  <pod-id> /private/tmp/jim-nss-benchmark-<revision>.tar.gz \
  --workload paper-15d \
  --candidate-only \
  --data-file benchmark-results/device-parallel-nss/gw170817-cln16-median/data/gw170817.npz \
  --no-hlo
```

This mode does not set XLA dump flags, create `gpu-hlo/`, or require HLO files
during artifact verification. The result bundle records `hlo_capture: false`
in `artifact-verification.json`; all non-HLO checks and artifacts are unchanged.
When `--data-file` is supplied, the upload helper verifies its SHA-256 before
and after the run and the pod's preparation phase validates and reuses that
exact frozen bundle instead of downloading a new copy.

Run the identical single-analysis workflow against only the pinned original
paper-style sharded-live-state revision with:

```bash
python benchmarks/device_parallel_nss/runpod/upload_and_run.py \
  <pod-id> /private/tmp/jim-nss-benchmark-<revision>.tar.gz \
  --workload paper-15d \
  --original-sharded-only
```

This mode writes
`posterior/original-sharded-paper-15d-g4-seed0.npz` and retains the exported
baseline source and package manifest alongside the timing, profile, telemetry,
and per-slice artifacts. The candidate implementation is not executed. The
candidate-only result also retains the same package manifest. Run candidate-only
first on the same pod: original-sharded mode requires its frozen data bundle,
copies it into the baseline result tree, verifies the copy's SHA-256, and asserts
that preparation reports `reused_existing=true`. This makes the two analyses use
byte-identical corrected strain and PSD inputs.

The helper streams its own progress in the local terminal. The benchmark runs
through SSH, so the RunPod web console's container log can remain empty after
boot; that does not mean the process is idle. The helper downloads results into
`benchmark-results/device-parallel-nss/`.

Delete the pod as soon as the result archive is safe locally:

```bash
runpodctl pod delete <pod-id>
```

The automatic deletion deadline is a failsafe, not the normal teardown path.

## Result bundle

The archive contains:

- `paired-four-gpu/hardware.json`: GPU metadata, NVLink status, and topology;
- `paired-four-gpu/data/gw170817.npz`: the immutable strain/PSD input;
- `paired-four-gpu/harness/`: the exact runner used for both implementations;
- `paired-four-gpu/raw/`: reports and stderr/stdout logs for every paired run;
- `profiles/`: TensorBoard/Perfetto-compatible JAX traces for 15 steady outer
  steps from both four-GPU implementations;
- `posterior/`: equally weighted posterior NPZ artifacts when sample output is
  requested (always present in `--candidate-only` mode);
- `gpu-hlo/`: text compiler dumps and a hashed `hlo-manifest.json` for each
  single-implementation run, omitted for candidate runs using `--no-hlo`;
- `per-slice/`: per-chain, per-slice `num_expansions` and `num_shrink` NPZs;
- `telemetry/`: one-second `nvidia-smi dmon` samples captured only during each
  profiler window;
- `jax-compilation-cache/` and `persistent-cache-summary.json`: the cold/warm
  fresh-process cache evidence;
- `candidate-device-sweep/`: complete one- and two-GPU candidate reports (the
  paired candidate report is the four-GPU point);
- `likelihood-lane-sweep.json`: 16/32/64/128-lane real-likelihood timings;
- `diagnostic-summary/`: derived lockstep/FSM slot ratios, scaling efficiency,
  lane-cost curves, and telemetry summaries;
- `paired-four-gpu/manifest.json`: revisions, run order, seeds, hashes, and
  commands;
- `paired-four-gpu/summary.json` and `summary.md`: paired sample-call timings
  plus evidence diagnostics.

The H200 host used for this run requires `NCCL_NVLS_ENABLE=0` because its
NVSwitch Fabric Manager rejects NCCL's NVLink SHARP multicast allocation.
Ordinary NCCL/NVLink collectives remain enabled, and both revisions use the
same setting; it is recorded in each raw report.

Every raw full-run report contains host `time.perf_counter()` start/end values
for every synchronized outer step. The first step includes lazy JIT; later
steps provide the steady-state distribution and stragglers. Evidence estimates
and likelihood-evaluation counts must remain scientifically consistent before
making a performance claim.
