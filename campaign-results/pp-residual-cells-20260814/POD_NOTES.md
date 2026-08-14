# P-P Residual Attribution Pod Notes

## Stage 1 pod

- Pod ID: `33vwj8o1gd4ssh`
- Launched: `2026-08-14T15:20:34Z`
- Cloud/GPU: Runpod Community Cloud, 4 x `NVIDIA A100-SXM4-80GB`
- Price at launch: US$5.56/hour total
- Image: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- NVIDIA driver: `595.71.05`
- Automatic termination: `2026-08-15T01:20:18Z` (10-hour cost guard)
- GPU UUIDs:
  - `GPU-e3450bfa-f876-24c4-42ff-ddd8d5966bcc`
  - `GPU-c264f067-21e3-cb50-23df-99e274d27df5`
  - `GPU-4463a58c-3d00-18a3-0391-890a198ca804`
  - `GPU-187daf72-49e9-4844-3db8-308a658f02d2`

The Runpod catalogue contained no GH200 SKU on 2026-08-14. The plan's
GH200-vs-current-class hardware arm therefore cannot be run on Runpod without
revising the hardware target or using another provider.

## Immutable workload

- Git revision: `db1768b6a0fbf7a0e645c2bd106878dc142777c0`
- Implementation tree SHA-256: `8d17e860fca3754075915dba48782b0c8f52f95d0b5291b544b777b97b8b66d8`
- Workspace archive: `jim-pp-residual-db1768b6.tar.gz`
- Workspace archive SHA-256: `cf349e009c4ea885d4f6fb7cbb2a1b3e4c06572e2415be1f67c9d7d029ffb942`
- Frozen environment: JAX `0.10.2`, jaxlib `0.10.2`, BlackJAX `1.6.2`

## Endpoint smoke gate

The paired injection-000 smoke used fresh one-shot processes and the frozen
00-legacy and 08-production inputs. The downloaded receipts are:

- `campaign-results/pp-residual-receipts-20260814/33vwj8o1gd4ssh-00-legacy-campaign-ids1-5feceb66ffc8.tar.gz`
- `campaign-results/pp-residual-receipts-20260814/33vwj8o1gd4ssh-08-production-campaign-ids1-5feceb66ffc8.tar.gz`

| Metric | 00-legacy | 08-production | Comparison |
|---|---:|---:|---:|
| Nested-sampling iterations | 247 | 247 | identical |
| Likelihood evaluations | 1,145,211 | 1,145,211 | identical |
| `log_Z` | 253.41912696056625 | 253.4191269605658 | relative delta `1.7944476265093704e-15` |
| Paper-convention post-JIT sampling | 293.116474712966 s | 108.70856240810826 s | `F_impl = 2.6963513105118873` |
| `sample_phases.ns_loop` | 281.5287032679189 s | 95.75208021700382 s | ratio `2.9401836767398453` |

The exact go/no-go gate required identical iteration and likelihood-evaluation
counts, relative `log_Z` difference at most `1e-6`, and a first
paper-convention `F_impl` in the expected 2.5--5 range. All checks passed, so
the Stage 1 full chain was authorized to continue.

## Stage 1 full-chain result

All nine cells completed all ten events: 90/90 recovery envelopes passed
`validate_completed_result`. Across every cell, each paired event had identical
nested-sampling iteration and likelihood-evaluation counts. The largest
cross-cell evidence drift was `1.4097167877480388e-11` absolute and
`6.549887255355215e-14` relative, both well inside the `1e-6`
relative gate. All required timing fields were finite and positive.

The paper-convention full-chain factor was
`F_impl = 2.8293923767064753`, with paired-event bootstrap 95% interval
`[2.7916932427041337, 2.866393235974976]`. The production-fidelity
endpoint was `1.0008846182533366`
`[0.9971758748372767, 1.003725937879268]`, so no endpoint caveat applies.

## Later-stage feasibility

The literal local D=1 device-filter test passed only its unavailable-device
guard. The substantive simulated-device pre-gate was therefore rerun with
`XLA_FLAGS=--xla_force_host_platform_device_count=4`; all three
pathwise-equivalence cases passed. The hardware arm nevertheless remained
blocked because the exact GH200 target was absent. No substitute accelerator
was used.

The period-stack CPU gate tested JAX/jaxlib `0.8.2`, `0.8.3`,
`0.9.0`, `0.9.0.1`, `0.9.1`, `0.9.2`,
`0.10.0`, and `0.10.1`. Each failed the mandated bitwise FSM gate
at the same two covariance elements by one ULP
(`2.22044605e-16`); the Stage-1 `0.10.2` control passed all 15
tests. Because the first passing candidate is the current stack, no identifying
old/new GPU environment arm was launched.

## Billing and teardown

- Pod uptime: 25,369 seconds (7.047 hours)
- Account-balance delta: US$39.3568241001
- Initial/final balances: US$93.344931405 / US$53.9881073049
- Result receipts: 11 archives in
  `campaign-results/pp-residual-receipts-20260814/`
- Teardown: pod `33vwj8o1gd4ssh` deleted after all receipts were merged
  and validated
- Final Runpod inventory check: zero pods

The Stage-1 spend stayed below the plan's US$55 total ceiling. Tasks 6 and 7
incurred no GPU spend.
