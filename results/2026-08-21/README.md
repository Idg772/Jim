# 100-injection P-P campaign (2026-08-21)

Results for the N=100 BNS injection probability-probability campaign.
Configuration: Sharded real-PE fast-ridge, M=2 Gibbs sweeps, D=4 devices, joint
(cos iota, d_hat) slice block, analytic phi_c and t_c marginalization, adaptive
slice widths with shrink-only brackets. Paper reference: arXiv:2607.28265v1.

Result: the Fisher-combined exact one-sample KS test gives p = 0.318 over 15
parameters. The post-JIT sampling time has median 45.2 s per injection on 4x
H200 (paper timing convention, Figure 3 / Table III).

## Files

- `catalogue.csv` — the 100 injection truths and seeds.
- `index.json` — per-injection provenance: batch, seeds, posterior and summary
  paths with SHA-256 checksums.
- `manifest.json` — the pooled campaign configuration and provenance.
- `pp/ranks.csv` — per-injection posterior rank of the truth for the 15 sampled
  parameters.
- `pp/summary.csv`, `pp/report.json` — per-parameter KS tests and the
  Fisher-combined result.
- `timing.csv` — per-injection post-JIT sampling, sample-call, and total wall
  times.
- `pp-timing.png`, `pp-timing.pdf` — the P-P plot in the paper style and the
  post-JIT timing histogram on linear axes.
- `make_figure.py` — builds `timing.csv` and the figure from the files above.
  The timing extraction reads the per-injection summaries from the local batch
  directories listed in `index.json`; those large raw outputs stay out of
  version control.
