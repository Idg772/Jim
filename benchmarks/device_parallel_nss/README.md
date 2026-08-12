# Device-parallel NSS outer-step benchmark

This benchmark isolates Jim's device-parallel nested-sampling executor. It
repeatedly applies one compiled outer transition to the same seeded live state
with a deliberately cheap likelihood. That makes communication, replacement
chain vectorisation, and host/device latency visible without waveform cost
masking them.

Run the paper-sized structural workload on four simulated CPU devices:

```bash
uv run python benchmarks/device_parallel_nss/benchmark_outer_step.py \
  --n-live 512 \
  --n-delete 64 \
  --dims 15 \
  --iterations 100 \
  --n-devices 4 \
  --simulate-cpu
```

On a machine with four real GPUs or TPUs, omit `--simulate-cpu`:

```bash
uv run python benchmarks/device_parallel_nss/benchmark_outer_step.py \
  --n-live 512 --n-delete 64 --dims 15 --iterations 100 --n-devices 4
```

Use a separate process for each point in a 1/2/4-device scaling comparison;
JAX fixes its visible device topology at process start. Add `--dtype float64`
to match an x64 analysis. `--inner-steps-per-dim` and `--seed` are also
available; run `--help` for all options.

The script writes one JSON report to stdout containing:

- environment, dependency, backend, and device metadata;
- setup, lowering, compilation, and synchronised warmed-step timings;
- a post-SPMD HLO census of all-gather, all-reduce, and collective-permute
  instructions, including the matching instruction text;
- every live-state and returned-history leaf's shape and JAX sharding.

For the replicated multi-device executor, the structural target is one packed
endpoint all-gather, no all-reduces or collective-permutes, fully replicated
live/adaptation/integrator state, and replacement-ID-sharded dead history and
diagnostics.

Simulated CPU timings are useful for deterministic regression checks, not as a
claim about GPU scaling. This cheap fixed-state seam also does not replace the
full-resolution and compressed gravitational-wave benchmarks: those should use
the same `n_live`, `n_delete`, precision, inner work, and stopping rule when
comparing implementations.

## Four-GPU full workflows

`benchmark_gw170817_full_run.py` runs the repository's full-resolution
GW170817 SwiG workflow against an immutable, workload-selected local strain/PSD
bundle. The paired orchestrator, `compare_gw170817_full_run.py`, exports the
baseline and candidate implementations, then alternates three shared-seed pairs
on exactly four GPUs:

```bash
uv run python benchmarks/device_parallel_nss/compare_gw170817_full_run.py \
  --workload aligned-11d \
  --output-dir /tmp/jim-gw170817-4gpu
```

The historical `aligned-11d` workload remains the default. To benchmark the
paper's full GW170817 parameter-estimation model, select `paper-15d`:

```bash
uv run python benchmarks/device_parallel_nss/compare_gw170817_full_run.py \
  --workload paper-15d \
  --output-dir /tmp/jim-gw170817-paper-15d-4gpu
```

It fixes H1/L1/V1, 128 seconds, 20--(2048-1/128) Hz, float64, 512 live points,
64 deletions, one Gibbs sweep, and `termination_dlogz=0.0485873516`. The upper
edge is fixed at the final bin of the legacy 4096 Hz analysis grid, while the
native strain extends to 8192 Hz. Reports include
the frozen-data and configuration hashes, total and sampler wall times,
iterations, likelihood evaluations, evidence, posterior sample count, and ESS.

`paper-15d` matches the paper's stated real-GW170817 model: precessing-tidal
`IMRPhenomPv2_NRTidalv2`; two isotropic three-dimensional spins with magnitudes
up to 0.05; two tidal deformabilities; full sky, inclination, polarization,
and distance; phase and time marginalization; and the paper's seven SwiG
blocks. The benchmark-local JAX waveform is independently checked against
LALSimulation, with affine-phase-aligned mismatches of order `1e-9` across
validation points drawn from the paper prior.

The data-window choice is also workload-specific. `aligned-11d` preserves the
historical Jim interval from 126 seconds before to 2 seconds after the precise
trigger and its 2048-second run-wide O2 mean-Welch PSD. `paper-15d` approximates
the published real-event setup and uses the trigger-centred
interval from 64 seconds before to 64 seconds after it. Its PSD starts from a
median-Welch estimate of the same cleaned event file's complete pre-analysis
prefix, GPS 1187007040--1187008818.43 (1778.43 seconds). SciPy applies the
standard median-periodogram bias correction. A 1 Hz running log-median removes
finite-segment bin noise; bins above twice the local trend retain their Welch
values so instrumental lines remain down-weighted. Analysis and PSD strain are
both explicitly fetched from GWOSC's native 16384 Hz `GW170817-v2`
(`LOSC_CLN_16_V1`) release, in which the L1 glitch was removed. This keeps the
stored product's high-frequency anti-aliasing transition outside the 20--2048
Hz likelihood band. The time-marginalization correlation FFT remains pinned to
4096 Hz: the native windowed FFT is projected onto 0--2048 Hz only after it is
formed, with no time-domain decimation. Thus the likelihood slice (259,584
bins), padding, and sampler-kernel shapes are unchanged; preparation, bundle
I/O, and setup still process larger native arrays.
The manifest records these products and all four endpoints, and preparation
rejects a bundle whose whitened L1 data retain the known pre-merger glitch. An
aligned bundle therefore cannot be silently reused for `paper-15d`, nor can an
older bundle prepared from the default glitch-containing event release.

This does not recreate the paper's separate synthetic-injection catalogue:
its exact catalogue, coloured-noise seeds, and data-preparation code have not
been released. Each paired comparison freezes one workload-specific public
GW170817 strain/PSD bundle and reuses it byte-for-byte for the baseline and
candidate. Exact posterior or evidence agreement with the publication still
depends on the authors' unreleased preprocessing details.

For historical context, the three-seed four-H200 **aligned-11d** run on 3
August 2026 measured median `sample_call` times of 521.419 seconds for the
paper-style baseline and 450.282 seconds for the candidate: a 1.158x speedup
(13.64% less sampler time), with matched iterations, likelihood evaluations,
log evidence, and ESS for every paired seed. `paper-15d` needs fresh timing and
scientific goldens; those 11D numbers must not be used as its validation.

### Observed PSD-correction check (7 August 2026)

A fresh seed-0 `paper-15d` run on four H200s held the 16 kHz cleaned strain,
likelihood, waveform, priors, sampler settings, and initial positions fixed and
changed only the PSD post-processing described above. The resulting equal-weight
posterior contained 2,382 samples; the sampler reported posterior ESS 4,220 and
terminated after 227 outer iterations. The configuration SHA-256 is
`567fab881d67077c2c01e2940dc7da4ad6ac557f820f3ead1b6ff3448b516de6`, the
frozen-data SHA-256 is
`a502f1d077618b94c8bf8d9820d10f850c8764e18a0f0e7e18663e14ac75104f`, and the
posterior artifact SHA-256 is
`377807f995842752da438a2f15c210bb80980463c9ad102fc682b3e24f67091f`.

The effective tidal deformability 5th/50th/95th percentiles moved from
`[180, 718, 1548]` in the previous raw-16 kHz run to `[172, 410, 899]` after
smoothing. The mass-ratio median remained `0.913 -> 0.915`, while the effective
spin median moved only `+0.0031 -> -0.0018`. This fresh-run result confirms that
finite-segment Welch-bin noise was materially tilting the soft tidal direction;
it is not merely an importance-reweighting artifact. It does not establish that
this ad hoc smoother is equivalent to the paper's on-source BayesWave PSD or its
calibration marginalization.

## Baseline-versus-candidate comparison

`compare_outer_step.py` runs this seam against exported source snapshots of the
paper-style baseline (`86335bdb`) and the candidate revision. Each point runs in
a fresh process so JAX sees exactly the requested topology. The default matrix
uses one and four devices only, five shared seeds, float64, 20 warmups, and 1,000
timed steps:

```bash
uv run python benchmarks/device_parallel_nss/compare_outer_step.py \
  --simulate-cpu \
  --output-dir /tmp/jim-nss-comparison
```

On a four-GPU host, omit `--simulate-cpu`. The runner masks the same host down
to GPU 0 for the one-device control and captures `nvidia-smi` metadata, NVLink
status, topology, raw reports, and derived speedup/efficiency tables.

The complete cost-guarded RunPod workflow is in
[`runpod/README.md`](runpod/README.md).

## Timing conventions

`timing_seconds.paper_convention.post_jit_sampling_seconds` is the sampling
wall time excluding **both** one-off JIT compilation costs, matching
arXiv:2607.28265 (Table 3), which excludes "the two one-off compilation
costs — the likelihood and the sampler-kernel JIT" from quoted sampling
times:

- `sampler_jit_seconds` — first outer step minus the steady-state median
  (identical to the legacy `jit_compile_estimate`, which is retained
  unchanged).
- `likelihood_jit_seconds` — measured directly: the sampler AOT-compiles
  the initial batched likelihood evaluation
  (`jax.jit(...).lower(...).compile()`) and times the compile separately
  from the evaluation. Reported as `null` for sampler revisions without
  phase timing, in which case it is not subtracted.

Per-phase raw numbers are in `timing_seconds.sample_phases`
(`init_total`, `likelihood_jit`, `initial_likelihood_eval`, `ns_loop`,
`sampler_kernel_jit`, `finalise`). The injection-campaign runner uses the
direct `sampler_kernel_jit` measurement; the full-run comparison keeps its
first-step observer for compatibility with historical revisions that do not
expose that phase.

### Warm-cache protocol

To quote warm-start numbers, pass `--jax-compilation-cache-dir <dir>` and
run the benchmark twice with the same directory: the first (prime) run
populates the persistent XLA cache and is discarded; the second (warm) run
is the quotable one. On a warm run both JIT fields should collapse to a
few seconds. Cold-run JIT fields must be reported alongside whenever the
release protocol requires cold starts, and every published table must say
which convention (cold or warm) it uses.
