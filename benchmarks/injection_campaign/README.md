# FSM/SwiG 100-injection P–P campaign

This directory turns the measured `paper-15d` benchmark into a reproducible,
resumable injection-recovery campaign. Each recovery uses the benchmark's
15-dimensional precessing-tidal waveform and physical prior, plus an
explicitly sampled coalescence time `t_c` in the covariance-shaped
mass/tidal SwiG block. The resulting 16-dimensional sampler uses 512 live
points, 64 deletions per outer step, one Gibbs sweep across seven blocks,
and four-way device sharding. Phase is injected but analytically
marginalized; `t_c` is sampled (uniform, non-periodic, ±30 ms around the
trigger), so the P–P test covers the 15 physical parameters plus `t_c`.

The default noise curves are the Bilby-packaged Advanced LIGO zero-detuned
high-power and Advanced Virgo design PSDs. `prepare_campaign` interpolates and
hashes them onto the exact 128 s / 4096 Hz analysis grid. There is no network
access during a recovery.

## Local or already-provisioned GPU host

Prepare the immutable inputs once:

```bash
uv run --group cross-validation python -m \
  benchmarks.injection_campaign.prepare_campaign \
  campaign-results/paper-15d-100 --n-injections 100 --seed 260728265
```

Validate what would run:

```bash
uv run python -m benchmarks.injection_campaign.run_campaign \
  campaign-results/paper-15d-100 --dry-run
```

Run or resume all recoveries on exactly four visible GPUs:

```bash
JAX_PLATFORMS=cuda uv run --extra cuda python -m \
  benchmarks.injection_campaign.run_campaign \
  campaign-results/paper-15d-100 --retry-count 2 --plot
```

To split the work into explicit ranges, add `--start 0 --stop 25`, etc. A job
is skipped only when both its summary and posterior exist and its manifest hash
matches. Each job runs in its own process, so a failed recovery cannot poison
the next one's GPU state. Successful logs are deleted by default; failed logs
are retained. The shared `.jax-cache` is reusable but excluded from Runpod
downloads.

Regenerate the P–P products at any time from completed summaries:

```bash
uv run python -m benchmarks.injection_campaign.plot_pp \
  campaign-results/paper-15d-100
```

Generate a Figure 3-style timing distribution and summary table:

```bash
uv run python -m benchmarks.injection_campaign.plot_timing \
  campaign-results/paper-15d-100 \
  --pdf-output output/pdf/fsm-swig-injection-timing.pdf
```

The sampler-call series is the closest available analogue of the paper's
sampling wall time. The campaign did not separately record likelihood and
sampler-kernel JIT phases, so the generated figure marks that convention
explicitly instead of claiming the paper's post-JIT metric. It also includes
the operational end-to-end process time for context.

The compact tracking artifacts are:

- `manifest.json`: frozen scientific configuration, input hashes, and storage policy.
- `catalogue.csv`: truths and independent noise/sampler seeds.
- `status.csv`: one row per injection with completion, timing, and artifact paths.
- `results/injection-NNN/summary.json`: diagnostics, SNRs, timings, and ranks.
- `results/injection-NNN/posterior.npz`: compressed equal-weight posterior only.
- `pp/ranks.csv`, `pp/summary.csv`, `pp-combined.png`, and `pp-grid.png`.
- `timing/figure-3-summary.csv` and `timing/figure-3-equivalent.png`.

Calibration-specific configuration and outputs:

- `sample_coalescence_time` / `coalescence_time_range_seconds`: new campaigns
  sample `t_c` directly and build the likelihood without FFT time
  marginalization. The earlier FFT-grid treatments (deterministic
  `upsample_factor`, sampled one-cell `time_jitter`) produced a wrapped
  "comb" posterior at high SNR — alignment slabs separated by up to
  ~165 logL — whose teeth local slice moves cross only by luck; sampled
  `t_c` restores the smooth t_c–mass ridge that covariance-shaped
  proposals follow natively, and matches the reference paper's own P-P
  configuration. `t_c` joins the P-P outputs as a sixteenth parameter.
- Older manifests without `sample_coalescence_time` retain their stored
  behavior exactly (FFT marginalization with their recorded
  `upsample_factor`/`jitter_time` settings), including under `--force`.
- New catalogues use an independently shuffled Latin hypercube for every parameter, so each marginal covers every prior stratum once.
- `pp/summary.csv` reports truth-draw KS statistics and rank-minus-truth-quantile residual means and standard errors.

Profiles, HLO, per-slice arrays, successful logs, and copied strain buffers are
deliberately not retained.

## Runpod

The provisioner is dry-run by default and sets a six-hour deletion guard:

```bash
benchmarks/injection_campaign/runpod/provision.sh
benchmarks/injection_campaign/runpod/provision.sh --launch
```

Create the curated working-tree archive (this includes dirty/untracked campaign
work, not `.git`, caches, or benchmark results):

```bash
benchmarks/device_parallel_nss/runpod/package_workspace.sh \
  /private/tmp/jim-injection-campaign.tar.gz
```

Then upload, execute, and download the result archive:

```bash
uv run python -m benchmarks.injection_campaign.runpod.upload_and_run \
  POD_ID /private/tmp/jim-injection-campaign.tar.gz \
  --download-dir campaign-results
```

The downloader retrieves partial results even when one or more recoveries fail.
After verifying the local archive, delete the pod rather than merely stopping it;
the provision-time `--terminate-after` deadline is the cost backstop.

## Scientific scope

This is a prior-predictive sampler-calibration campaign, not a reproduction of
an unreleased injection catalogue. Truths are drawn from the recovery prior and
noise is freshly drawn from the stated design PSD for every detector/injection.
The manifest and catalogue hashes make that distinction auditable.
