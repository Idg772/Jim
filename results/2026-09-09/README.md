# CE inference from 5 Hz

These results use one CE detector, Gaussian noise, and an 8192-second data
segment from 5 to 2048 Hz. The waveform is IMRPhenomD_NRTidalv2. The detector
response includes rotation, orbital motion, and finite arm length. The runs
use four H100 SXM GPUs, 4096 live points, and 512 replacements per round.
Phase is marginalized; 12 physical parameters are sampled.

The main run uses eight Gibbs sweeps and deferred cache updates. Its sampling
loop took 63.580 seconds. The loop and final processing took 65.635 seconds.
The observed sample call took 353.430 seconds, including 284.642 seconds of
explicit outer-step compilation. Data preparation and likelihood construction
are outside that call. No complete warm-worker pipeline time was measured.

The endpoint control gives the same values for every saved array. Its loop
took 66.665 seconds. The two six-sweep runs use different sampler seeds and
the same noise. Their loops took about 54 seconds. These runs test sensitivity
to sampler settings; they are not a population calibration study.

The native posterior checks cover earlier two- and six-sweep outputs, not
the final eight-sweep posterior. The check record contains 71 rows and 51
unique physical points. The largest centered likelihood error is about
1.47e-6 nats. Exact agreement between the two eight-sweep executions does
not establish independent-chain mixing.

Files follow the dated result layout on `NS-FSM`:

- `catalogue.csv` contains the injection values and seeds for each run.
- `timing.csv` separates loop, final processing, sample-call, and pipeline time.
- `index.json` records source paths, sizes, and SHA-256 checksums.
- `manifest.json` states the scope, timing definitions, and storage rules.
- Each run folder contains its original configuration and timing record.
- `nested/` contains all weighted points, including log weights and likelihoods.
- `posterior/` contains the saved posterior draws.
- `comparison/` contains the paired execution and recovery checks.
- `validation/` contains the earlier native checks and recovery summaries.
- `make_figure.py` creates the timing figure from `timing.csv`.

The endpoint control arrays are omitted because their values equal the main
run arrays; `comparison/duplicate-arrays.json` records this check. Retained
NPZ files use lossless ZIP compression. Every NPY member is unchanged.
Historical configurations and provenance records retain their original paths.
The noise seed is 1 for all four runs; the seed in each final configuration
is the sampler seed. Large strain files, compiler files, per-slice traces,
and pod tools are not part of this result package.
