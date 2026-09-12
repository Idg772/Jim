# CE and ET-D inference with adaptive bins

These results use CE from 5 Hz and three ET-D channels from 2 Hz, through
2048 Hz. A 131072-second segment contains a signal of about 21.6 hours.
The waveform is IMRPhenomD_NRTidalv2. The data contain Gaussian noise.
The detector response includes rotation, orbital motion, and finite arm
length. The sampler uses 4096 live points, 512 replacements per round, and
eight Gibbs sweeps. Phase is marginalized; 12 physical parameters are sampled.

The adaptive run selected 128 bins with degree-eight waveform interpolation
and degree-16 Chebyshev phasor moments before sampling. It used 1152 source
nodes and 4041 active detector entries. On four H200 GPUs, the sampling loop
took 220.865 seconds; the loop and final processing took 224.384 seconds.
The observed sample call took 350.390 seconds. The measured interval from
heterodyning start to sampler return took 524.845 seconds, including first-use
compilation. No complete warm-worker pipeline time was measured.

The retained 256-bin and 1024-bin Taylor runs used four H100 SXM GPUs. Their
loop times were 481.573 and 902.502 seconds. Hardware and implementation
changes prevent attribution of this time difference to the allocator alone.
The matched fixed-grid H200 trial timed out without a completed posterior
or sampling time. `fixed256-h200-status.json` records that limit.

The adaptive posterior contains all 12 injected sampled parameters inside
their marginal 95% intervals, and 11 inside their 90% intervals. The saved
comparison includes effective spin and tidal parameters, sky position, and
the polarization branches. Individual spin shifts must be assessed with
their correlation and effective spin. The three pre-sampling native checks
pass the 0.05-nat tolerance. These finite checks and one-event posterior
comparisons do not establish population coverage or prior-wide accuracy.
The reported weight ESS is not a measure of chain mixing.

Files follow the dated result layout on `NS-FSM`:

- `catalogue.csv` contains the injection values and seeds for each run.
- `timing.csv` separates loop, final processing, sample-call, and pipeline time.
- `index.json` records source paths, sizes, and SHA-256 checksums.
- `manifest.json` states the scope, timing definitions, and storage rules.
- Each run folder contains its configuration, native checks, and timing records.
- `nested/` contains all weighted points, including log weights and likelihoods.
- `posterior/` contains the saved posterior draws.
- `posterior-comparison/` contains the comparison statistics and plots.
- `posterior-recovery/` contains the injection recovery statistics and plots.
- `make_figure.py` creates the timing figure from `timing.csv`.

The old 1024-bin attempts 5 and 8 have identical physical parameter arrays.
Their likelihood and weight arrays differ. Both outputs are retained; they
do not count as independent runs. All four retained
network runs share noise seed 2 and sampler seed 2. NPZ files use lossless
ZIP compression; every NPY member is unchanged. Historical configurations
and provenance records retain their original paths and options. They are
records of those runs, not portable launch configurations. Large strain and
moment files, compiler files, per-slice traces, and pod tools are omitted.
