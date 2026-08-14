# Period-Stack Environment Arm

Status: **not measured**. No GPU environment replay was launched because no
distinct older stack passed the plan's mandatory CPU equivalence gate.

## Selection

The period rule selected JAX/jaxlib `0.8.2`: it was the newest release
before the six-month cutoff implied by a July 2026 submission month.
BlackJAX was held at `1.6.2`. The scratch environments used Python
`3.11.15`.

The exact `jax==0.8.2`, `jaxlib==0.8.2`,
`blackjax==1.6.2` request does not resolve: BlackJAX 1.6.2 declares
`jaxlib>=0.9.0`. To distinguish a resolver issue from an actual runtime
failure, 0.8.2 and the next release, 0.8.3, were also tested with JAX/jaxlib
force-overridden without dependencies. No CUDA plugin was selected or
installed because the CPU pre-gate failed before GPU compatibility became
relevant.

## CPU compatibility gates

| JAX/jaxlib | Resolver status | FSM gate | Likelihood gate |
|---|---|---|---|
| 0.8.2 | Conflicts with BlackJAX's declared floor | Failed after 6 passes | 125 passed |
| 0.8.3 | Conflicts with BlackJAX's declared floor | Failed after 6 passes | Not run after FSM failure |
| 0.9.0 | Compatible | Failed after 6 passes | 125 passed |
| 0.9.0.1 | Compatible | Failed after 6 passes | Not run after FSM failure |
| 0.9.1 | Compatible | Failed after 6 passes | Not run after FSM failure |
| 0.9.2 | Compatible | Failed after 6 passes | Not run after FSM failure |
| 0.10.0 | Compatible | Failed after 6 passes | Not run after FSM failure |
| 0.10.1 | Compatible | Failed after 6 passes | Not run after FSM failure |
| 0.10.2 | Stage-1 control | 15 passed | Already qualified by the Stage-1 workload |

Every failing version stopped in
`test_swig_legacy_covariance_path_matches_lockstep_bitwise`. Two of six
covariance elements differed by `2.22044605e-16`; although this is only
one ULP, the plan requires bitwise trajectory equivalence, so the failure is
disqualifying. The first passing candidate is JAX/jaxlib `0.10.2`, which
is the Stage-1 stack itself and therefore supplies no old/new contrast.

Representative commands:

```bash
/private/tmp/jim-period-jax-082/bin/python -m pytest \
  tests/unit/samplers/blackjax/test_fsm.py -x -q
/private/tmp/jim-period-jax-090/bin/python -m pytest \
  tests/unit/core/single_event/test_likelihood.py -x -q
```

## Decision

Running 00-legacy and 08-production on GPUs under `0.10.2` and comparing
them with Stage 1's `0.10.2` results would measure pod-to-pod noise, not
a period-stack effect. The GPU arm was therefore not launched and
`E` remains unmeasured. A future environment measurement needs either:

1. an older stack that passes the preregistered bitwise gate, or
2. an explicitly revised scientific protocol that accepts the one-ULP
   trajectory difference and redefines the estimand before any GPU spend.
