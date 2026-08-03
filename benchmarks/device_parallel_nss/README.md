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
