# CE XG tracer run

This directory freezes the first engineering candidate for the current
dominant-mode, aligned-spin XG response path. It uses one 40 km CE detector,
5--2048 Hz, an 8192 s zero-noise injection, sampled detector-arrival time,
sampled luminosity distance, phase marginalization, 65,536 relative bins,
4,096 live points, 512 simultaneous deletions, two Gibbs sweeps, and no sky
folding. Inclination is sampled in the exact uniform `cos_iota` coordinate.
The trigger is a physical GPS epoch inside the pinned DE405 span. Earth-orbit
curvature uses an affine-removed cubic surrogate fitted over the complete
emission-time support and checked against the full ephemeris.

The injection coordinates are adapted from the public Licence-to-Bin example
at revision `5c15b707e1b9c90d0ef2f36d4b378124f31074a8`. Its precessing spins are
projected onto the orbital axis because the current Jim dynamic-response cache
supports dominant nonprecessing waveforms only. The Euclidean `d_L^2` prior is
also an explicit approximation to the paper's source-frame-volume prior. This
is therefore an aligned tracer, not a reproduction of the published signal.

Prepare the immutable CE PSD and Earth/Sun ephemerides:

```console
uv run python benchmarks/xg/prepare_inputs.py
```

The compressed path is intentionally locked until the five independent
evidence receipts described in `docs/xg_qualification.md` have been generated
for this exact analysis and runtime. After those jobs finish, assemble the
manifest:

```console
uv run jim-xg-qualify assemble \
  benchmarks/xg/xg-ce-4096-65536.toml \
  benchmark-results/xg-ce-4096-65536/qualification
```

Add the emitted manifest path and digest to `[likelihood.heterodyne]`, then run
on exactly four GPUs:

```console
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run jim-run --verbose benchmarks/xg/xg-ce-4096-65536.toml
```

Do not create placeholder receipts or bypass the XG plan token. A missing or
failed qualification result blocks this run.
