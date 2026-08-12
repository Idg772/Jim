# P–P campaign remediation record

## Target and decision rule

The reference population is the Figure 2(a) Sharded setup in
arXiv:2607.28265v1: 100 prior-predictive injections, D=4, M=1, 512 live
points, 64 deletions, the FSM scheduler, and termination at
`log Z_live - log Z_dead < -3`. The paper does not publish its event
catalogue, noise seeds, or sampler seeds, so its exact curves and p-values are
not reproducible.

The local acceptance rule is deliberately strict and was frozen before the
production result was evaluated:

- all leading injection IDs 0–99 must be complete and valid;
- each of the 15 sampled parameters receives an exact, two-sided, one-sample
  Kolmogorov–Smirnov test against `Uniform(0, 1)`;
- Fisher's method combines all 15 KS p-values, with 30 degrees of freedom; and
- the combined p-value must be strictly greater than 0.05.

There are no excluded parameters. In particular, q remains in the combined
statistic even though the known q pathology in the original work was explicitly
allowed to remain unresolved. An individual parameter is not a separate
failure gate.

## Corrections and blocking remedy

Two implementation errors were corrected before testing the blocking remedy:

1. The carrier waveform had used the NRTidal-merger time anchor. Injection and
   recovery now use the IMRPhenomD peak-time anchor, matching the author
   implementation, and recovery requires the frozen manifest value.
2. With phase marginalisation, the retained spin-azimuth coordinate is
   `beta_i = (s_i_phi + phase_c) mod 2 pi`. Production summaries preserve both
   the physical catalogue truth and transformed rank truth. P–P regeneration
   recomputes ranks from the weighted posterior and fails closed on a mismatch.

The paired corrected-anchor campaign showed that those corrections alone were
not sufficient. The production remedy therefore changed the proposal
partition, while retaining D=4, M=1 and every other scientific configuration
field, to one homogeneous `all-slow-time` scheme for all 100 recoveries:

```text
[M_c, q, lambda_1, lambda_2,
 s1_mag, s1_theta, s1_phi,
 s2_mag, s2_theta, s2_phi,
 iota, t_c]
[zenith, azimuth]
[psi]
```

No paper-block or event-wise fallback result is represented as part of this
homogeneous campaign.

## Final homogeneous result

The completed production campaign is
`campaign-results/all-slow-time-fsm-d4-m1-pp-20260811`.

| Statistic | Result |
| --- | ---: |
| Complete recoveries | 100 / 100 |
| Fisher statistic | 35.8643494248485 |
| Fisher degrees of freedom | 30 |
| All-15 Fisher-combined exact KS p-value | 0.21255202454037572 |
| Remediation threshold | strictly greater than 0.05 |
| Excluded parameters | none |
| Eligible | true |
| Remediation decision | **pass** |

The individual q KS p-value is 0.5729477560230845. The smallest individual
value is 0.05651324361786969 for `s2_mag`; it remains above 0.05, although the
frozen decision rule uses the all-parameter Fisher value rather than an
individual minimum.

The report records `legacy_summaries_corrected_from_weighted_posteriors = 0`:
all 100 production summaries carry the current spin-azimuth rank-coordinate
metadata.

## Historical corrected-anchor control

The paired source campaign is
`campaign-results/corrected-anchor-fsm-d4-m1-pp-20260810`. It used the paper's
seven proposal blocks with the corrected carrier anchor and completed all 100
recoveries. Its all-15 Fisher-combined exact KS p-value was
0.025311472645006664, so it **failed** the same strict 0.05 rule. This is a
historical paper-block control, not the final remediation result.

That source report also contains an earlier q-exempt assessment: its 14-value
Fisher p-value is 0.08546032502510702 and its own `passes` field is therefore
true. That superseded assessment is not the final all-15 decision rule and is
not being represented as a successful control here.

The final manifest pins that control exactly:

- source config SHA-256:
  `742f2365ba9ecdfcf8f2e4bb3c15d21283a38f9ceff56603499ad5fb68a48184`;
- source manifest file SHA-256:
  `7a5a746ac8dc3e8d6c675c63c9a2511d1dfe270de1c16d209272c7e4f4d5f720`;
- catalogue SHA-256:
  `5b26098abcc6dceb7437a1626632b54c1f708f346b29c9abbff93a3266b1a9ce`.

The source and production catalogue and PSD files are byte-identical, and all
likelihood, prior, waveform, data, D=4, M=1, FSM, live-set, deletion and
stopping fields match. Only `blocks`, campaign identity, and the descriptive
paper-configuration label differ within the scientific configuration.

The curated implementation trees are not byte-identical. Both identify Git
revision `7b450d15207703f464665817486096c58e44194e`, while the production
campaign pins tree
`0f80108078b8b20c761ac4218ca6c484637dd64d8f1e652b30dd8d173c1fc8f7`.
The intervening changes add the long-lived worker, JAX cache/constant handling,
birth-likelihood and insertion diagnostics, and corrected spin-azimuth rank
metadata. The sampling transitions, likelihood, prior and waveform code used
by the paired campaigns are unchanged. The result should therefore be
described as the complete production protocol, not attributed to a block-list
text change without its pinned implementation provenance.

All production recoveries report the same implementation revision and tree,
use a real four-device `NVIDIA A100-SXM4-80GB` GPU backend, and use the
long-lived worker. The worker shares process-level JAX/runtime infrastructure
only; detectors, likelihood, Jim, sampler state, and PRNG streams are rebuilt
for every event.

## Diagnostic progression

Small matched diagnostics were publication-ineligible scheme-selection tools,
not substitutes for the 100-event calibration. Early sky/time and extrinsic
merges were source-dependent. Later intrinsic/slow-block trials motivated the
homogeneous `all-slow-time` production test. The production result above, not a
selected diagnostic row or pooled repeated run, is the acceptance evidence.

The supported diagnostic schemes are documented in `README.md`. Repeated or
overlapping parameter blocks remain invalid because the sampler requires a
disjoint partition.

## Validation and receipt chain

The result is protected at several distinct boundaries:

1. `prepare_remediation_campaign` requires the complete paired 100-row source
   campaign and runs the full staged-result validator over every source result
   before freezing an empty remediation campaign. It copies the catalogue and
   PSDs, records their hashes, and pins the source manifest plus candidate Git
   revision and curated-tree digest.
2. Resume checks bind a purported completion to its frozen config hash,
   posterior path/hash and recorded byte count, exact injection ID, both seeds,
   and all 17 physical and marginalised truth fields. A wrong campaign hash
   fails closed; another stale or corrupt payload is recomputed.
3. The Runpod uploader validates the curated workspace inventory/tree and the
   frozen input archive before upload, verifies the remote manifest and
   catalogue hashes, checks the downloaded archive against the remote digest,
   and publishes receipts atomically without overwriting a differing file.
4. `merge_staged_results` safely extracts each receipt and checks immutable
   manifest/catalogue/PSD identity, real D=4 GPU inventory, implementation pin,
   IDs, truths, seeds, posterior path/hash/bytes and field inventory, array
   shapes and finiteness, normalized weights and ESS, stored ranks, timing
   arithmetic, and collisions before atomically installing results and
   rebuilding `status.csv`.
5. The `blocking_remediation` manifest marker requires every posterior to
   include `log_likelihood_birth` and every summary to include the matching
   insertion-index diagnostic. The merge validator recomputes it from the
   stored death and birth likelihoods and requires the metadata to agree. This
   is a provenance/integrity diagnostic, not an additional threshold in the
   frozen Fisher acceptance rule.
6. `plot_pp` requires the complete leading-100 selection, rebinds every summary
   to catalogue truths and seeds, verifies posterior hashes, recomputes all
   ranks from the original normalized nested-sampling weights, and then applies
   the all-15 decision rule.

`status.csv` is an operational projection regenerated from validated result
directories; it is not itself scientific acceptance evidence.

## Reproduction artifacts

Core frozen artifacts:

- Final manifest:
  `campaign-results/all-slow-time-fsm-d4-m1-pp-20260811/manifest.json`
  - config SHA-256:
    `fa1c5b34230b68ac1a9b22f91f5011ecbef800baa2e635e6e24d5dc63cb36a29`
  - file SHA-256:
    `e6d0351655bcdd28cf902dd76eb0c10de41c53625143716b0e662a1e2c3b6918`
- Machine-readable final report:
  `campaign-results/all-slow-time-fsm-d4-m1-pp-20260811/pp/report.json`
  - file SHA-256:
    `9f98e5ea948d04c0fa3a4858586ba5a9806dde5a14c0fcb255f3d59f767612be`
- Final combined plot:
  `campaign-results/all-slow-time-fsm-d4-m1-pp-20260811/pp/pp-combined.png`
  - SHA-256:
    `ea7d1a350a48d30705cfb84389688e0338111c02d2042005b4e955a338124721`
- Final grid plot:
  `campaign-results/all-slow-time-fsm-d4-m1-pp-20260811/pp/pp-grid.png`
  - SHA-256:
    `0ae15bc2a687ad36d86b99a1bca57a0bde43c05201ffef2d9839826666332db6`
- Final ranks and summary tables:
  - `pp/ranks.csv`:
    `d0b0fee2205e0b508abf554f302393996ceb6399ec62499e8452a63cb020644b`
  - `pp/summary.csv`:
    `f6fc12d8a2ef7c377f93f4ebd3efcbb4234d10a70f275119626b0ebcda93defb`

Immutable transport receipts are under
`campaign-results/runpod-allslowtime-receipts-20260811`:

| Receipt | SHA-256 |
| --- | --- |
| `75ymj4y9l8mcho-allslowtime-campaign-000-003.tar.gz` | `4da94a4dbf35f1d41ec36223f0a1844098475af4386020d057b5d5a5e977d896` |
| `75ymj4y9l8mcho-allslowtime-campaign-003-035.tar.gz` | `872a172ad3267ae220ff381cd6a328685c909346d16cd347cb14034e90769c9c` |
| `hym8e8pa1kwx5a-allslowtime-campaign-035-068.tar.gz` | `eaeccdc035cc4f1d04a5b265cd391c24879e4aed838906ba79d694c8c19711fe` |
| `75ymj4y9l8mcho-allslowtime-campaign-068-092.tar.gz` | `ed82a37b5660853d5542d99fb5e1a4d7dd72de4baa17e45302649bfe3d73823c` |
| `n8ds6ps6t7063g-allslowtime-campaign-092-100.tar.gz` | `ace99f0904f0759f85af5603fe62bd7c79e4eb4d47f5e0f8a52da2fac97db4a3` |

These are immutable evidence files. This record intentionally makes no claim
about current pods, account balance, or other live infrastructure state.
