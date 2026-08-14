# P-P Residual Attribution: Stage-1 Result and Open Closure

Date: 2026-08-14

Status: **the implementation chain is measured; the end-to-end audit is not
closed**. Stage 1 identifies a joint external residual of 2.29735x, but the
specified GH200 hardware factor and a distinct period-stack environment factor
could not be measured under the plan's scientific gates. The population term
is therefore not identifiable and the 1.25x closure criterion is not met.

## Headline result

The primary timing metric is the paper-convention post-JIT sampling time. Ten
timing-stratified events were run through nine cumulative cells on one
4 x A100-SXM4-80GB pod, with the same event rows, seeds, PSDs, and frozen
revision in every cell. All 90 recoveries passed the result-envelope and
cross-cell trajectory gates.

```text
observed endpoint       = 6.851098
F_impl                  = 2.8293923767064753
carrier anchor          = 1.054
measured internal       = F_impl * 1.054
                        = 2.982179565048625
joint external residual = 6.851098 / measured internal
                        = 2.2973459010635704
```

Thus the measured identity is:

```text
6.851098 = 2.982179565048625 * (H * E * P_leftover)
H * E * P_leftover = 2.2973459010635704
```

Neither `H` nor `E` was assigned a placeholder value. As a
result, `P_leftover = external / (H * E)` cannot be evaluated. The
2.29735x number is a joint external residual, not a measured population factor.

## Measured ledger

Intervals below are 95% percentile intervals from 1,000 paired-event bootstrap
resamples over the ten selected events. They do not include the endpoint's
timing-convention ambiguity, carrier-anchor uncertainty, pod-to-pod variance,
or the unmeasured hardware/environment terms.

| Ledger row | Factor | 95% interval | Evidence class |
|---|---:|---:|---|
| 01 stepping cache | 1.231549 | [1.226968, 1.235701] | Paired, same-pod, 10 events |
| 02 replicated topology | 1.000759 | [0.999371, 1.001872] | Paired, same-pod, 10 events |
| 03 FSM scheduler | 1.631783 | [1.622623, 1.641911] | Paired, same-pod, 10 events |
| 04 shared frequency grid | 1.016220 | [1.015024, 1.017503] | Paired, same-pod, 10 events |
| 05 real angle phasor | 1.242723 | [1.238878, 1.246609] | Paired, same-pod, 10 events |
| 06 real inner product | 1.114254 | [1.112283, 1.116492] | Paired, same-pod, 10 events |
| 07 Cholesky factor | 0.998894 | [0.996539, 1.001315] | Paired, same-pod, 10 events |
| 08 production fidelity | 1.000885 | [0.997176, 1.003726] | Paired, same-pod, 10 events |
| Full implementation chain, `F_impl` | **2.829392** | **[2.791693, 2.866393]** | Paired, same-pod, 10 events |
| Carrier anchor | 1.054000 | Not supplied | Fixed prior anchor required by the spec; not remeasured here |
| Internal subtotal, `F_impl x 1.054` | **2.982180** | **[2.942445, 3.021178]** | Derived from paired `F_impl`; carrier treated as fixed |
| Joint external residual | **2.297346** | **[2.267691, 2.328369]** | Algebraic inverse of paired `F_impl` |
| Hardware, `H` | Unmeasured | -- | Exact GH200 target absent from Runpod catalogue |
| Environment, `E` | Unmeasured | -- | No distinct older stack passed the mandatory bitwise gate |
| Population, `P_leftover` | Not identifiable | -- | Requires measured `H` and `E` |

The endpoint fidelity cell is within the required 5% band, so the chain needs
no emulation caveat. The independent nested-sampling-loop metric gives
`F_impl = 2.961536`, bootstrap interval
`[2.943135, 2.981970]`; its joint external residual is 2.194838
`[2.179798, 2.208561]`. The paper-convention metric remains primary.

The result falls in the plan's lowest decision band:
`F_impl x 1.054 = 2.98218 <= 3.5`. The P-P-native chain therefore does
not support the ledger-understatement hypothesis. It leaves a slightly larger
external residual than the superseded cross-study 3.32x code ledger did.

## Scientific gate

- 90/90 result envelopes passed `validate_completed_result`.
- Every paired event had identical nested-sampling iteration and
  likelihood-evaluation counts in all nine cells.
- Maximum absolute `log_Z` drift was
  `1.4097167877480388e-11`; maximum relative drift was
  `6.549887255355215e-14`, far below `1e-6`.
- Paper-convention times were finite and positive, spanning
  96.91993026598357--365.7859551620204 seconds.
- The frozen implementation revision was
  `db1768b6a0fbf7a0e645c2bd106878dc142777c0`; the packaged workspace
  SHA-256 was
  `cf349e009c4ea885d4f6fb7cbb2a1b3e4c06572e2415be1f67c9d7d029ffb942`.

## Hardware factor

The Runpod catalogue contained no GH200 SKU on 2026-08-14. Substituting H100,
H200, or another accelerator would change the requested estimand, so neither
the paired D=1 replay nor the GH200 lanes crosscheck was run.

The local trajectory pre-gate itself is ready: the literal `-k device`
test passed one unavailable-device guard but did not exercise equivalence; the
supplemental forced-four-device pathwise gate passed all three cases. The
planned event labels were also audited before spend: after timing
stratification, the actual fastest/median/slowest representatives are reindexed
IDs 003, 006 (or 008 for the other central representative), and 001, not the
plan text's 000/004/009.

There is also a sign erratum in Task 6. To preserve both the multiplicative
closure identity and “`H > 1` means the Stage-1 reference hardware is
faster,” the factor must be defined as:

```text
H = T_GH200 / T_stage1_reference
```

The inverse ratio printed in the plan cannot have that interpretation. No
numeric `H` is reported, so the erratum does not alter a measurement.

## Environment factor

The nominal period stack, JAX/jaxlib 0.8.2 with BlackJAX 1.6.2, has a declared
dependency conflict because BlackJAX requires jaxlib 0.9.0 or newer. Forced
runtime tests of 0.8.2 and 0.8.3, followed by resolver-compatible 0.9.0,
0.9.0.1, 0.9.1, 0.9.2, 0.10.0, and 0.10.1, all failed the mandatory FSM
bitwise gate at the same two covariance elements by one ULP
(`2.22044605e-16`). The likelihood suite passed 125 tests under both
0.8.2 and 0.9.0. The Stage-1 JAX/jaxlib 0.10.2 control passed all 15 FSM tests.

Because the first passing stack is the current stack itself, a GPU replay
would not identify an old/new environment effect. No GPU environment arm was
launched and `E` remains unmeasured. Full details are in
`pp-residual-cells-20260814/env-arm/STACK_NOTES.md`.

## Counting-convention consistency

Only cells 01--03 belong in the sampler-work comparison. Their directly paired
combined factor is:

```text
T_00 / T_03 = 2.0111473000930618
95% paired-event bootstrap interval = [1.9949400519958347, 2.029718823198627]
```

The point and interval lie wholly inside the retained trace's reconstructed
stock-to-production physical-lane-slot band of 1.743--2.179x. Applied to the
previous current P-P estimate of 2,236,161 physical slots, the point estimate
predicts about 4,497,249 old slots, within the 4.063--5.078 million
reconstruction. It is 12.4% above the rounded four-million prose proxy, so the
proper conclusion is “consistent with the physical-slot reconstruction,” not
“exactly reproduces four million.”

The approximately-four-million count therefore remains compatible with a
physical lane-slot reading. It is already represented by the stepping-cache
and FSM wall-time factors and must not be multiplied into the ledger again.

## Population mapping and limitation

The selected source IDs were 11, 12, 31, 38, 39, 56, 66, 90, 93, and 94. The
subset median was 44.82364556007087 seconds versus the 100-event population
median of 44.664371909573674 seconds, giving a population/subset mapping of
0.9964466600494655.

That close median alignment does not establish population equivalence. Our
population's max/min timing spread is 1.913617, materially wider than the
paper's published 1.484496 spread. The mapping is descriptive and is not
silently multiplied into the paired implementation factors or assigned as
`P_leftover`.

## Audit rules and remaining evidence

This report corrects the three attribution errors in the superseded waterfall:

1. no exponent is fitted to force the factors to the observed endpoint;
2. no hardware factor is inferred from specification-sheet bandwidth; and
3. unexplained residual is reported explicitly rather than erased.

The measured evidence currently supports only
`H x E x P_leftover = 2.297346`
`[2.267691, 2.328369]`. This is above the 1.25x closure target, and
`P_leftover` cannot be isolated while `H` and `E` are
unmeasured.

Evidence that would move the audit is:

- an exact GH200-96 replay paired against the Stage-1 reference class, or a
  full 4 x GH200 replay;
- an older JAX stack that passes the preregistered bitwise gate, or an
  explicitly revised protocol for the one-ULP mismatch; and
- the paper's raw catalogue, timing artifacts, and counter definitions, which
  would directly constrain the remaining population/counting ambiguity.

## Pod and cost

Stage 1 used Runpod pod `33vwj8o1gd4ssh`, Community Cloud,
4 x NVIDIA A100-SXM4-80GB at US$5.56/hour. It ran for 25,369 seconds
(7.047 hours). The recorded account-balance delta was US$39.3568241001, below
the US$55 plan ceiling. The pod was deleted after all receipts were downloaded,
merged, and validated; the final Runpod inventory contained zero pods.
