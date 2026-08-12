# Sharded FSM/SwiG paper-methodology P–P campaign

This directory runs the **Sharded** injection test described in Sections III–V
of [arXiv:2607.28265v1](https://arxiv.org/abs/2607.28265), using the FSM
execution path of the SwiG sampler. By default it freezes a deterministic
catalogue of 1000 independent draws from the paper's Table I priors, then
recovers catalogue entries 0–99 for the main-text, 100-injection P–P test.

The frozen workload matches the published calibration methodology:

- IMRPhenomPv2_NRTidalv2 injection and recovery waveforms on the uncompressed
  128 s, 4096 Hz grid over 20–2048 Hz.
- H1, L1 and V1 with independent coloured-Gaussian noise per event and
  detector. H1/L1 use Bilby's `aLIGO_O4_high_asd.txt` (squared to a PSD) and V1
  uses `AdV_psd.txt`; preparation interpolates linearly in PSD and records the
  source and frozen-array hashes.
- The paper's literal injection-segment convention at reference GPS
  `1187008882`: the 128 s segment starts at `T - 66 s`, is centred at
  `T - 2 s`, and ends at `T + 62 s`. The sampled coalescence-time offset is
  relative to `T`.
- Phase and luminosity distance are analytically marginalised, while
  coalescence time is sampled. The P–P test therefore covers the same 15
  directly sampled parameters as the paper and excludes the two marginalised
  parameters.
- The carrier phase uses the IMRPhenomD peak-time anchor used by the author
  implementation. With phase marginalisation, spin azimuth ranks use the
  retained gauge coordinate `beta_i = (s_i_phi + phase_c) mod 2 pi`; summaries
  preserve both the physical catalogue truth and the transformed rank truth.
- The common Sharded settings: 512 live points, 64 deletions per outer
  iteration, one Gibbs sweep, four devices, and the published
  `log Z_live - log Z_dead < -3` stopping convention. The paper control uses
  seven physical proposal blocks. The successful remediation retains these
  settings but uses the homogeneous three-block `all-slow-time` partition
  documented below.
- Credible levels are computed directly from the original nested-sampling
  weights as `sum(w[theta < theta_true]) / sum(w)`. There is no equal-weight
  resampling step.

## Table I priors

Injection truths and recovery priors are identical:

| Parameter | Range | Prior |
| --- | --- | --- |
| `M_c` | `[1.5, 2.5] M_sun` | uniform |
| `q` | `[0.5, 1]` | uniform |
| `|s1|`, `|s2|` | `[0, 0.05]` | uniform |
| spin polar angles | `cos(theta_si) in [-1, 1]` | isotropic |
| spin azimuths | `[0, 2 pi)` | uniform |
| `iota` | `[0, pi]` | `sin(iota)` |
| `lambda_1`, `lambda_2` | `[0, 5000]` | uniform |
| `ra` | `[0, 2 pi)` | uniform |
| `dec` | `[-pi/2, pi/2]` | `cos(dec)` |
| `psi` | `[0, pi)` | uniform |
| `t_c` | `[-0.1, 0.1] s` | uniform, sampled |
| `phase_c` | `[0, 2 pi)` | uniform, marginalised |
| `d_L` | `[30, 150] Mpc` | proportional to `d_L^2`, marginalised |

## Local or already-provisioned four-GPU host

Prepare the immutable inputs once. The explicit counts below are also the
defaults: 1000 truths are written to `catalogue.csv`, while the manifest selects
only the first 100 for recovery.

```bash
uv run --group cross-validation python -m \
  benchmarks.injection_campaign.prepare_campaign \
  campaign-results/paper-sharded-pp \
  --catalogue-size 1000 --n-injections 100 --seed 260728265
```

Preparation refuses to replace an existing manifest. Use a new directory to
change the seed, catalogue size, or recovery count.

## Full all-slow-time remediation

The completed Figure 2(a) remediation is
`campaign-results/all-slow-time-fsm-d4-m1-pp-20260811`. It uses one homogeneous
proposal partition for every recovery:

```text
[M_c, q, lambda_1, lambda_2,
 s1_mag, s1_theta, s1_phi,
 s2_mag, s2_theta, s2_phi,
 iota, t_c]
[zenith, azimuth]
[psi]
```

Freeze a new paired campaign from a complete corrected-anchor source with:

```bash
uv run python -m benchmarks.injection_campaign.prepare_remediation_campaign \
  campaign-results/corrected-anchor-fsm-d4-m1-pp-20260810 \
  campaign-results/all-slow-time-fsm-d4-m1-pp-REPRO \
  --scheme all-slow-time \
  --implementation-revision 7b450d15207703f464665817486096c58e44194e \
  --implementation-tree-sha256 \
    0f80108078b8b20c761ac4218ca6c484637dd64d8f1e652b30dd8d173c1fc8f7
```

Preparation requires a publication-eligible leading-100 source and applies
the full staged-result validator to all 100 source recoveries. It copies the
catalogue and PSD inputs, changes only the block partition and presentation
labels in the scientific config, creates no result payloads, and pins the
source manifest plus candidate Git revision and curated implementation tree.

The frozen production config SHA-256 is
`fa1c5b34230b68ac1a9b22f91f5011ecbef800baa2e635e6e24d5dc63cb36a29`.
Its paired source-manifest file SHA-256 is
`7a5a746ac8dc3e8d6c675c63c9a2511d1dfe270de1c16d209272c7e4f4d5f720`,
and its catalogue SHA-256 is
`5b26098abcc6dceb7437a1626632b54c1f708f346b29c9abbff93a3266b1a9ce`.
All 100 final results use the pinned tree above, the long-lived worker, and a
real four-device `NVIDIA A100-SXM4-80GB` backend.

The complete report is eligible and passes the frozen rule: Fisher statistic
35.8643494248485, 30 degrees of freedom, and all-15 p-value
0.21255202454037572. The q value is 0.5729477560230845; no parameter is
excluded. The historical corrected-anchor paper-block control instead failed
the same all-15 rule with 0.025311472645006664; its earlier q-exempt assessment
is superseded. See `PP_REMEDIATION.md` for the full decision record and artifact
hashes.

Validate the selected jobs without running them:

```bash
uv run python -m benchmarks.injection_campaign.run_campaign \
  campaign-results/paper-sharded-pp --dry-run
```

Run or resume all selected recoveries on exactly four visible GPUs:

```bash
JAX_PLATFORMS=cuda uv run --extra cuda python -m \
  benchmarks.injection_campaign.run_campaign \
  campaign-results/paper-sharded-pp --retry-count 2 --plot
```

To split the work into explicit ranges, add `--start 0 --stop 25`, etc. These
ranges address the selected recovery IDs, not all frozen catalogue rows. A job
is skipped only after its summary and posterior are bound to the frozen
campaign: the config hash, posterior path/hash and recorded byte count, exact
injection ID, both seeds, and all 17 sampled and marginalised truth fields must
match. A completed payload with another config hash fails closed; another stale
or corrupt payload is recomputed.

The local default starts one process per event. `--long-lived-worker`, which is
always used by the pod runner, instead shares process-level imports, the JAX
device client, catalogue and compilation-cache instrumentation. It reconstructs
the detectors, likelihood, Jim object, sampler state and PRNG streams for every
event. Successful logs are deleted by default; failed logs are retained. The
shared `.jax-cache` is reusable but excluded from Runpod result downloads.

To run the paper's Appendix B-sized stress test, create a separate campaign
with both `--catalogue-size 1000` and `--n-injections 1000`. Merely freezing
1000 truths while recovering the first 100 remains an `N = 100` calibration
test.

Regenerate the P–P products at any time from completed summaries:

```bash
uv run python -m benchmarks.injection_campaign.plot_pp \
  campaign-results/paper-sharded-pp
```

This command deliberately refuses to publish the paper result until all 100
selected recovery IDs are present. It binds config, IDs, truths and seeds to the
frozen catalogue, verifies posterior hashes, and recomputes every rank from the
weighted posterior before publishing. Full receipt acceptance is the stronger
`merge_staged_results` validation described below. For troubleshooting only,
`--allow-partial` emits products labelled `partial-exploratory`; they are not a
substitute for the main P–P result.

Generate the operational timing distribution and summary table:

```bash
uv run python -m benchmarks.injection_campaign.plot_timing \
  campaign-results/paper-sharded-pp \
  --pdf-output output/pdf/fsm-swig-injection-timing.pdf
```

This Figure 3 product uses the paper's post-JIT convention. For every recovery,
the sampler records the full `jim.sample` wall time and directly measures the
one-off likelihood and sampler-kernel compilations. The plotted value is
`sample_call - likelihood_jit - sampler_kernel_jit`; initial likelihood
evaluation, nested iterations, termination checks, and finalisation remain in
the sampling budget. No warm-up injection is discarded. The publication plot
requires all 100 selected IDs by default; `--allow-partial` is explicitly
exploratory. Inclusive sampler-call, both JIT phases, and end-to-end time remain
available in the audit CSV but are not mixed into the Figure 3 histogram.

Generate the paper Figure 7-style optimal network-SNR distribution for the
evaluated recovery set:

```bash
uv run python -m benchmarks.injection_campaign.plot_snr \
  campaign-results/paper-sharded-pp
```

The frozen catalogue contains parameters and seeds for 1000 events, while only
the selected 100 recoveries have evaluated waveform SNRs. The plot and report
therefore identify their population as the selected recovery set rather than
claiming to represent all 1000 unevaluated catalogue rows.

The compact tracking artifacts are:

- `manifest.json`: frozen scientific configuration, selection rule, input
  hashes, reproduction scope, and storage policy.
- `catalogue.csv`: all 1000 frozen truths and independent noise/sampler seeds.
- `status.csv`: an operational projection with one row per selected recovery;
  it is regenerated from result directories and is not resume or scientific
  acceptance authority.
- `results/injection-NNN/summary.json`: diagnostics, SNRs, timings, weighted
  ranks and rank coordinates, execution/device inventory, implementation
  label/revision/tree/module, insertion-index diagnostic, and posterior
  metadata.
- `results/injection-NNN/posterior.npz`: compressed parameter arrays,
  death- and birth-contour log likelihoods, and normalized original
  nested-sampling log weights.
- `pp/ranks.csv`, `pp/summary.csv`, `pp-combined.png`, and `pp-grid.png`.
- `pp/report.json`: completeness, 15 exact two-sided one-sample KS tests, the
  all-parameter Fisher-combined p-value, the exact binomial-band convention,
  remediation eligibility, and artifact hashes. The remediation contract has
  `excluded_parameters: []`: a complete eligible 100-event campaign passes
  only when the Fisher-combined p-value over all 15 sampled parameters,
  including q, is strictly greater than 0.05.
- `timing/figure-3-summary.csv` and `timing/figure-3-equivalent.png`: the
  post-JIT Sharded distribution and median/minimum/maximum.
- `timing/per-injection-timing.csv` and `timing/report.json`: raw post-JIT,
  compilation, inclusive sampler-call, and end-to-end timings plus definitions.
- `snr/figure-7-equivalent.png`, `snr/network-snr.csv`, and `snr/report.json`:
  the Figure 7-style optimal network-SNR distribution for evaluated recoveries.

Profiles, HLO, per-slice arrays, successful logs, copied strain buffers, and the
regenerable compilation cache are deliberately not retained.

## Historical mode-loss stress gate

Before spending the full 100-event budget, the known difficult cases can be
run as a separate ten-event regression gate:

```bash
uv run --group cross-validation python -m \
  benchmarks.injection_campaign.prepare_historical_stress \
  campaign-results/paper-sharded-pp-historical-stress
```

The source order is historical IDs `0, 82, 88, 75, 23, 58, 15, 81, 36, 96`.
The first two are sentinels; run preflight IDs `[0, 2)` and evaluate them before
continuing with `[2, 10)`. The legacy truths are outside the current paper
priors, so `M_c`, `q`, `d_L`, and `t_c` are moved to the same quantiles of the
paper priors while angles, spins, tides, phase, noise seeds, and sampler seeds
are preserved. These are therefore historical **quantile analogues**, not the
same likelihood points.

Evaluate a staged prefix or the completed gate with:

```bash
uv run python -m benchmarks.injection_campaign.evaluate_historical_stress \
  campaign-results/paper-sharded-pp-historical-stress --allow-partial
```

Every case must have `|z_q| <= 2.5`, with an immediate hard stop above `3`,
and neither the directly weighted `q` rank nor `t_c` rank may be exactly zero
or one. The historically narrow cases must also clear their prior-width-scaled
`q` posterior-width floors. The report verifies frozen hashes, truths, seeds,
four H200 GPU devices, and the post-JIT timing arithmetic. The old runtime-ratio
gate is explicitly not reused because the priors, PSD, segment convention,
sampled-time model, hardware, and timing definition changed.

This selected population is a one-way veto and risk-reduction check. Its
manifest marks it non-IID and publication-ineligible; the P-P and Figure 3
commands refuse to publish products from it. A pass permits the full campaign
but does not prove recovery of the exact legacy likelihood modes.

## Runpod

The provisioner is dry-run by default and creates a four-GPU pod with a
twelve-hour **deletion** guard:

```bash
benchmarks/injection_campaign/runpod/provision.sh
benchmarks/injection_campaign/runpod/provision.sh --launch
```

The deletion deadline is a cost backstop, not a runtime forecast. Inspect the
printed GPU type, price, command and deadline before adding `--launch`; report
actual hardware and timing provenance with every timing result.

Create the curated working-tree archive. It includes dirty and untracked
campaign work, but excludes `.git`, caches, and benchmark results:

```bash
benchmarks/device_parallel_nss/runpod/package_workspace.sh \
  /private/tmp/jim-injection-campaign.tar.gz
```

For a split remediation run, upload the curated workspace together with a
transport-only frozen campaign-input archive. The latter must contain exactly
one campaign root with pristine `manifest.json`, `catalogue.csv`, `status.csv`
and `inputs/`, but no results or caches:

```bash
uv run python -m benchmarks.injection_campaign.runpod.upload_and_run \
  POD_ID /private/tmp/jim-injection-campaign.tar.gz \
  --campaign-input /private/tmp/jim-remediation-input.tar.gz \
  --download-dir campaign-results/runpod-remediation-receipts \
  --run-label allslowtime --start 0 --stop 25
```

Use another non-overlapping `--start/--stop` range, or repeated
`--injection-id`, for each lane. Staged selections require `--campaign-input`.
Before execution, the uploader validates the curated workspace inventory and
tree, canonical campaign manifest, pristine status, catalogue and PSD hashes
and byte counts, and the manifest's implementation pin. It verifies uploaded
and remote frozen hashes. The downloader retrieves a cumulative result archive
even after a failed recovery, checks it against the remote SHA-256, and then
publishes it atomically. An identical existing receipt is accepted
idempotently; a differing file, symlink or non-file destination is never
overwritten. The receipt is transport evidence, not yet scientific acceptance.

Preflight and merge one or more receipts into the original frozen campaign:

```bash
uv run python -m benchmarks.injection_campaign.merge_staged_results \
  CAMPAIGN_DIR RECEIPT_1.tar.gz RECEIPT_2.tar.gz --dry-run

uv run python -m benchmarks.injection_campaign.merge_staged_results \
  CAMPAIGN_DIR RECEIPT_1.tar.gz RECEIPT_2.tar.gz
```

The merge path safely extracts regular files without trusting tar paths. It
requires byte-identical immutable manifest/catalogue/PSD inputs, a real D=4
GPU inventory, the pinned implementation, exact IDs/truths/seeds, posterior
path/hash/bytes/field order, numeric shapes and finiteness, normalized weights
and ESS, recomputed ranks, birth-likelihood ordering, recomputed insertion-index
metadata, and consistent post-JIT timing. It rejects archive and destination
collisions, stages copies before atomic installation, revalidates the merged
results, and regenerates `status.csv`.

Every manifest marked `blocking_remediation` requires both
`log_likelihood_birth` and the complete matching insertion-index diagnostic.
Legacy results without that manifest marker may omit birth likelihoods only
when they also omit insertion metadata. The insertion diagnostic is an
integrity/convergence record; it is not an extra threshold in the all-15 Fisher
decision. After receipts are safe locally, delete the pod rather than merely
stopping it; the provision-time deadline remains the backstop.

## Reproduction scope and limitations

This is a reproduction of the **published Sharded methodology**, not of the
paper's unpublished random realisation. The paper does not publish its
event-level 1000-row catalogue or its noise and sampler seeds. This campaign
therefore freezes a new deterministic iid catalogue from Table I. It can test
coverage under the same prior-predictive experiment, but it cannot reproduce
the paper's individual event ranks, SNR sample, KS values, Fisher-combined
p-value, or exact P–P curves.

The paper's internal catalogue also stored every injection's per-detector
optimal SNR. Here those SNRs are computed from the identical injected signal
and recorded in `summary.json` when a selected recovery runs, rather than being
precomputed for all 1000 rows in `catalogue.csv`. This bookkeeping difference
does not change the injections, likelihood, sampler, or calibration statistic.

## Matched blocking diagnostic

`prepare_blocking_diagnostic` builds a publication-ineligible D=4, M=1
diagnostic from completed rows of a corrected-anchor campaign. By default it
selects source IDs 10, 19, and 40 and creates three matched sampler-seed
replicates per source. Prepare each scheme in its own output directory, using
the same pinned candidate revision and curated-tree digest:

```bash
uv run python -m benchmarks.injection_campaign.prepare_blocking_diagnostic \
  SOURCE_CAMPAIGN OUTPUT_CAMPAIGN \
  --scheme paper \
  --implementation-revision GIT_SHA \
  --implementation-tree-sha256 CURATED_TREE_SHA256
```

The supported partitions are:

| Scheme | Change from the seven paper blocks |
| --- | --- |
| `paper` | Retain all seven blocks. |
| `mass-time` | Merge mass/tidal parameters with `t_c`. |
| `all-intrinsic` | Merge mass/tidal parameters and both spin blocks. |
| `all-slow` | Add `iota` to the all-intrinsic block. |
| `all-slow-time` | Add `t_c` to the all-slow block. |
| `sky-time` | Merge `[zenith, azimuth]` with `t_c`. |
| `fast-extrinsic` | Merge sky, `psi`, and `t_c`. |
| `full-extrinsic` | Add `iota` to the fast-extrinsic block. |
| `detector-time-fast-extrinsic` | Transform geocentric time to H1 arrival time and merge `[zenith, azimuth, psi, t_det]`. |

Seeds are derived without the scheme name, so equal replicate indices are
paired comparisons. Evaluate complete results—or a clearly labelled prefix—with:

```bash
uv run python -m benchmarks.injection_campaign.evaluate_blocking_diagnostic \
  OUTPUT_CAMPAIGN --allow-partial
```

Early sky/time and extrinsic-only trials were source-dependent and did not
establish a general remedy. The subsequent intrinsic/slow-block investigation
selected `all-slow-time` for a separate, homogeneous 100-event campaign. That
complete production campaign, rather than any hand-selected diagnostic row,
provides the passing calibration result recorded above. The generic
`prepare_campaign` command still reproduces the paper's seven blocks;
`prepare_remediation_campaign` defaults to `all-slow-time`, and the diagnostic
preparer defaults to `sky-time`. Targeted diagnostics remain
publication-ineligible and cannot support a P–P claim by themselves.

Timing is not automatically a like-for-like paper reproduction: hardware may
differ from the paper's four GH200 superchips. The campaign records both JIT
costs separately, and the Figure 3 post-JIT metric subtracts them from the full
sampler call. Scientific claims should be based on the weighted-rank
calibration products; timing claims must state the actual hardware and timing
convention separately.
