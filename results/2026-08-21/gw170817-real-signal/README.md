# GW170817 real-signal results, seeds 0-2

Three independent-seed runs of the sampler on the real GW170817 data.
Configuration: paper 15d workload, fast-ridge blocking, zero
differential-evolution jumps, adaptive slice widths with shrink-only
brackets, M=2 Gibbs sweeps, covariance directions, n_live=512, 4 devices.
Analytic phi_c and t_c marginalization. Run date: 2026-08-20.

| Seed | log_Z | n_iterations | nested ESS |
| ---- | ----- | ------------ | ---------- |
| 0 | 503.18 +/- 0.24 | 235 | 3957 |
| 1 | 503.55 +/- 0.20 | 235 | 4079 |
| 2 | 503.40 +/- 0.23 | 234 | 3904 |

## Files

- `seed-{0,1,2}/candidate-...-seedN.json` — the full run report:
  configuration, timing, work accounting, and evidence.
- `seed-{0,1,2}/posterior/*.npz` — equal-weight posterior draws.
- `seed-{0,1,2}/nested/*.npz` — the dead-point history with
  normalized nested-sampling log weights (used for the corner plots).
- `seed-{0,1,2}/package-manifest.json`, `artifact-verification.json`,
  `data-prepare.json` — provenance and checksums.
- `corner-seed{0,1,2}.png` — that seed's posterior only.
- `corner-overlay.png` — seeds 0-2 over the equal-seed combined
  posterior. The 15D order, serif styling, automatic posterior axes, orange fill,
  blue contours, full-prior chirp-mass inset, and full-sky inset follow
  Figure 4 of arXiv:2607.28265.
- `make_corner_plots.py` — rebuilds the corner plots from the files
  above with the normalized nested-sampling weights directly. Each seed
  contributes one third of the pooled posterior mass.

Source: extracted from the pod tarballs in
`campaign-results/gw170817-real-event-fast-ridge-adaptive-shrink-node-m1-m2-seeds0-2-20260820/m2/`
(telemetry, profiler traces, and the strain-data bundle stay in the
tarballs only).
