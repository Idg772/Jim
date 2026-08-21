"""Run a frozen-data GW170817 SwiG benchmark workload.

This is a full, converged, four-device BlackJAX SwiG run, not the cheap
outer-step microbenchmark in this directory.  It deliberately uses a frozen
local data bundle so baseline and candidate revisions see byte-identical
strain and PSD inputs and never perform network I/O during a timed run.

``--workload aligned-11d`` preserves the repository's historical aligned-spin
analogue.  ``--workload paper-15d`` uses the paper's full precessing-tidal
GW170817 parameter space, priors, marginalizations, and seven SwiG blocks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

# The Runpod harness executes this file directly. Make the repository's
# namespace package importable in that mode as well as under pytest/module use.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.device_parallel_nss.sampler_ablation import (
    VARIANT_NAMES as SAMPLER_ABLATION_VARIANTS,
)

SCHEMA_VERSION = 2
BENCHMARK_NAME = "gw170817-full-swig-4gpu"
DATA_FORMAT_VERSION = 6

GPS = 1187008882.43
DURATION = 128.0
ALIGNED_START = GPS + 2.0 - DURATION
# Preserve the epoch of the original 4096 Hz analysis window exactly.  Asking
# GWPy for the nominal trigger-centred decimal epoch on the native 16384 Hz
# grid would floor to the next 16 kHz tick and shift the data by about 61 us.
PAPER_NOMINAL_START = GPS - DURATION / 2.0
PAPER_START = 1187008818.429931640625
# Backwards-compatible aliases for the historical aligned-spin workload.
START = ALIGNED_START
END = START + DURATION
PSD_START = START - 2048.0
PSD_END = START
F_MIN = 20.0
# Keep the historical 4096 Hz analysis edge even when paper-15d is built from
# native 16384 Hz strain.  The endpoint below preserves the exact legacy
# frequency grid and its time-marginalization FFT shape.
NOMINAL_F_MAX = 2048.0
F_MAX = NOMINAL_F_MAX - 1.0 / DURATION
IFO_NAMES = ("H1", "L1", "V1")

# GW170817-v3 and the O2 run data contain the loud L1 transient at
# GPS 1187008881.389.  The event-specific v2 products have that glitch removed.
# Keep the selectors and source metadata explicit: GWPy otherwise resolves an
# unversioned event request to the highest available event version, which is
# the contaminated v3 product, and defaults to the 4096 Hz stored product.
GWOSC_ANALYSIS_STRAIN_DATASET = "GW170817-v2"
GWOSC_ANALYSIS_STRAIN_RELEASE = "O1_O2-Preliminary"
GWOSC_ALIGNED_ANALYSIS_STRAIN_PRODUCT = "LOSC_CLN_4_V1"
GWOSC_PAPER_ANALYSIS_STRAIN_PRODUCT = "LOSC_CLN_16_V1"
GWOSC_ANALYSIS_STRAIN_DOCUMENTATION_URL = (
    "https://gwosc.org/eventapi/html/O1_O2-Preliminary/GW170817/v2/"
)
GWOSC_ANALYSIS_STRAIN_API_BASE_URL = (
    "https://gwosc.org/eventapi/json/O1_O2-Preliminary/GW170817/v2"
)
GWOSC_EVENT_FILE_START_GPS = 1187007040.0
GWOSC_EVENT_FILE_DURATION_SECONDS = 2048.0
GWOSC_PSD_STRAIN_DATASET = "O2"
GWOSC_PSD_STRAIN_PRODUCT = "O2_4KHZ_R1"
GWOSC_ALIGNED_SAMPLE_RATE_HZ = 4096
GWOSC_PAPER_SAMPLE_RATE_HZ = 16384
TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ = 4096
# Backwards-compatible aliases for callers of the historical aligned workload.
GWOSC_ANALYSIS_STRAIN_PRODUCT = GWOSC_ALIGNED_ANALYSIS_STRAIN_PRODUCT
GWOSC_SAMPLE_RATE_HZ = GWOSC_ALIGNED_SAMPLE_RATE_HZ
GWOSC_FORMAT = "hdf5"

L1_GLITCH_GPS = 1187008881.389
L1_GLITCH_WINDOW_HALF_WIDTH_SECONDS = 0.2
L1_GLITCH_MAX_WHITENED_AMPLITUDE = 8.0
DATA_TUKEY_ROLL_OFF_SECONDS = 0.4
PSD_SMOOTHING_METHOD = "running-log-median-with-line-protection"
PSD_SMOOTHING_WIDTH_HZ = 1.0
PSD_LINE_PROTECTION_RATIO = 2.0

N_DEVICES = 4
SUPPORTED_DEVICE_COUNTS = (1, 2, 4)
N_LIVE = 512
N_DELETE = 64
N_DELETE_FRAC = N_DELETE / N_LIVE
NUM_INNER_STEPS_PER_DIM = 1
NUM_GIBBS_SWEEPS = 1
TERMINATION_DLOGZ = 0.0485873516

ALIGNED_WORKLOAD = "aligned-11d"
PAPER_WORKLOAD = "paper-15d"
WORKLOAD_CHOICES = (ALIGNED_WORKLOAD, PAPER_WORKLOAD)
PAPER_BLOCKING_SCHEME = "paper"
ALL_SLOW_BLOCKING_SCHEME = "all-slow"
IOTA_DL_BLOCKING_SCHEME = "iota-dl"
FAST_RIDGE_BLOCKING_SCHEME = "fast-ridge"
FAST_RIDGE_INTRINSIC_BLOCKING_SCHEME = "fast-ridge-intrinsic"
FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME = "fast-ridge-intrinsic-5step"
FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE = (5, 1, 1, 2, 1, 2)
FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME = "fast-ridge-intrinsic-periodic-mh"
FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME = (
    "fast-ridge-intrinsic-periodic-mh-cde4"
)
FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME = (
    "fast-ridge-intrinsic-periodic-mh-cde8"
)
FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES = (
    "slice",
    "periodic-uniform-independence",
    "periodic-uniform-independence",
    "slice",
    "periodic-uniform-independence",
    "slice",
)
FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK = {
    "total_updates": 15,
    "total_slice_updates": 12,
    "waveform_rebuild_slice_updates": 8,
    "cache_hit_slice_updates": 4,
    "periodic_independence_attempts": 3,
    "waveform_rebuild_periodic_independence_attempts": 2,
    "cache_hit_periodic_independence_attempts": 1,
    "cache_segments": 2,
}
FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCK = {
    "parameters": (
        "M_c",
        "q",
        "lambda_1",
        "lambda_2",
        "s1_mag",
        "s1_theta",
        "s2_mag",
        "s2_theta",
    ),
    "attempts": 4,
}
FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_FIXED_WORK = {
    **FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK,
    "total_updates": 19,
    "complementary_de_attempts": 4,
    "waveform_rebuild_complementary_de_attempts": 4,
    "cache_hit_complementary_de_attempts": 0,
    "complementary_de_gamma": 1.0,
    "complementary_de_insert_after_block": 0,
}
FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCK = {
    **FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCK,
    "attempts": 8,
}
FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_FIXED_WORK = {
    **FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK,
    "total_updates": 23,
    "complementary_de_attempts": 8,
    "waveform_rebuild_complementary_de_attempts": 8,
    "cache_hit_complementary_de_attempts": 0,
    "complementary_de_gamma": 1.0,
    "complementary_de_insert_after_block": 0,
}
COMPLEMENTARY_DE_BLOCK_BY_SCHEME = {
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME: (
        FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCK
    ),
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME: (
        FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCK
    ),
}
COMPLEMENTARY_DE_FIXED_WORK_BY_SCHEME = {
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME: (
        FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_FIXED_WORK
    ),
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME: (
        FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_FIXED_WORK
    ),
}
COMPLEMENTARY_DE_BLOCKING_SCHEMES = frozenset(COMPLEMENTARY_DE_BLOCK_BY_SCHEME)
PERIODIC_MH_BLOCKING_SCHEMES = frozenset(
    {
        FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME,
        *COMPLEMENTARY_DE_BLOCKING_SCHEMES,
    }
)
BLOCKING_SCHEME_CHOICES = (
    PAPER_BLOCKING_SCHEME,
    ALL_SLOW_BLOCKING_SCHEME,
    IOTA_DL_BLOCKING_SCHEME,
    FAST_RIDGE_BLOCKING_SCHEME,
    FAST_RIDGE_INTRINSIC_BLOCKING_SCHEME,
    FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME,
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME,
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME,
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME,
)

ALIGNED_BLOCKS = (
    ("M_c", "q", "lambda_1", "lambda_2"),
    ("s1_z", "s2_z"),
    ("iota",),
    ("d_L",),
    ("zenith", "azimuth"),
    ("psi",),
)

PAPER_BLOCKS = (
    ("M_c", "q", "lambda_1", "lambda_2"),
    ("s1_mag", "s1_theta", "s1_phi"),
    ("s2_mag", "s2_theta", "s2_phi"),
    ("iota",),
    ("zenith", "azimuth"),
    ("psi",),
    ("d_L",),
)

PAPER_ALL_SLOW_BLOCKS = (
    (
        "M_c",
        "q",
        "lambda_1",
        "lambda_2",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
        "iota",
    ),
    ("zenith", "azimuth"),
    ("psi",),
    ("d_L",),
)

PAPER_IOTA_DL_BLOCKS = (
    ("M_c", "q", "lambda_1", "lambda_2"),
    ("s1_mag", "s1_theta", "s1_phi"),
    ("s2_mag", "s2_theta", "s2_phi"),
    ("iota", "d_L"),
    ("zenith", "azimuth"),
    ("psi",),
)

PAPER_FAST_RIDGE_BLOCKS = (
    ("M_c", "q", "lambda_1", "lambda_2"),
    ("s1_mag", "s1_theta", "s1_phi"),
    ("s2_mag", "s2_theta", "s2_phi"),
    ("zenith", "azimuth"),
    ("psi",),
    ("cos_iota", "d_hat"),
)

# Couple the non-periodic intrinsic coordinates that control the observed
# q--chi_eff manifold while leaving the two spin azimuths as periodic
# singletons.  The first three blocks are all waveform-rebuild priced and are
# contiguous, so this preserves the fast-ridge kernel's ten slow plus five fast
# slice updates and the same two FSM cache segments.
PAPER_FAST_RIDGE_INTRINSIC_BLOCKS = (
    (
        "M_c",
        "q",
        "lambda_1",
        "lambda_2",
        "s1_mag",
        "s1_theta",
        "s2_mag",
        "s2_theta",
    ),
    ("s1_phi",),
    ("s2_phi",),
    ("zenith", "azimuth"),
    ("psi",),
    ("cos_iota", "d_hat"),
)

# Backwards-compatible aliases used by the likelihood-lane microbenchmark.
BLOCKS = ALIGNED_BLOCKS

ALIGNED_LIMITATIONS = (
    (
        "This is the closest full workflow checked into Jim, not an exact paper "
        "reproduction."
    ),
    (
        "It uses aligned-spin IMRPhenomD_NRTidalv2 rather than the paper's "
        "precessing-tidal IMRPhenomPv2_NRTidalv2 model."
    ),
    (
        "It samples 11 parameters in 6 SwiG blocks rather than 15 parameters in "
        "7 blocks."
    ),
    (
        "It analyses public GW170817 strain with PSDs estimated from preceding "
        "GWOSC data rather than a synthetic design-sensitivity injection. The "
        "analysis strain is pinned to the glitch-mitigated GW170817-v2 product."
    ),
    "Time and phase are analytically marginalized; distance is sampled.",
    (
        "The likelihood ends at 2048-1/128 Hz, the last bin represented in "
        "its fixed 4096 Hz time-marginalization FFT grid."
    ),
)

PAPER_LIMITATIONS = (
    (
        "This reproduces the paper's full-resolution 15-parameter GW170817 "
        "model, priors, marginalizations, and seven SwiG blocks."
    ),
    (
        "IMRPhenomPv2_NRTidalv2 is a benchmark-local JAX composition of "
        "Ripple's Pv2 and NRTidalv2 primitives, guarded by an independent "
        "LALSimulation waveform-parity test."
    ),
    (
        "The paper's exact data-preparation code is not public; this benchmark "
        "uses frozen, glitch-mitigated GW170817-v2 strain and a median-Welch "
        "PSD estimated from the same LOSC_CLN_16_V1 event product's complete "
        "pre-analysis prefix, then removes finite-segment bin noise with a "
        "line-protected 1 Hz running log-median."
    ),
    "Time and phase are analytically marginalized; distance is sampled.",
    (
        "The native strain extends to 8192 Hz, but the likelihood ends at "
        "2048-1/128 Hz and keeps the legacy 4096 Hz time-marginalization grid."
    ),
)

PAPER_ALL_SLOW_LIMITATIONS = (
    (
        "This preserves the paper's full-resolution 15-parameter GW170817 "
        "model, priors, and marginalizations, but deliberately replaces its "
        "seven-block sampler partition with the requested four-block all-slow "
        "partition."
    ),
    *PAPER_LIMITATIONS[1:],
)

PAPER_IOTA_DL_LIMITATIONS = (
    (
        "This preserves the paper's full-resolution 15-parameter GW170817 "
        "model, priors, and marginalizations, but merges its iota and d_L "
        "singleton blocks into one joint block so slice directions can track "
        "the distance-inclination amplitude ridge."
    ),
    *PAPER_LIMITATIONS[1:],
)

PAPER_FAST_RIDGE_LIMITATIONS = (
    (
        "This preserves the paper's full-resolution 15-parameter GW170817 "
        "model, priors, and marginalizations, but samples inclination and "
        "distance through a joint fast (cos_iota, d_hat) ridge block. d_hat "
        "is the network-SNR-weighted distance of arXiv:2207.03508."
    ),
    *PAPER_LIMITATIONS[1:],
)

PAPER_FAST_RIDGE_INTRINSIC_LIMITATIONS = (
    (
        "This preserves the exact fast (cos_iota, d_hat) ridge block and "
        "jointly slices the eight non-periodic mass, tide, spin-magnitude, "
        "and spin-polar coordinates. The two periodic spin azimuths remain "
        "singletons. The schedule still contains ten waveform-rebuild and "
        "five cache-hit slice updates."
    ),
    *PAPER_LIMITATIONS[1:],
)

PAPER_FAST_RIDGE_INTRINSIC_5STEP_LIMITATIONS = (
    (
        "This preserves the fast-ridge-intrinsic blocks but uses five random "
        "covariance directions in the eight-dimensional intrinsic block. "
        "The remaining blocks retain one update per dimension, giving seven "
        "waveform-rebuild and five cache-hit slice updates per replacement."
    ),
    *PAPER_LIMITATIONS[1:],
)

PAPER_FAST_RIDGE_INTRINSIC_PERIODIC_MH_LIMITATIONS = (
    (
        "This preserves the fast-ridge-intrinsic blocks and their default "
        "dimension-based work budget, but replaces the s1_phi, s2_phi, and "
        "psi singleton slices with one-call prior-corrected uniform "
        "independence updates on their periodic supports. The fixed work is "
        "twelve slices plus three independence attempts in two cache segments."
    ),
    *PAPER_LIMITATIONS[1:],
)

PAPER_FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_LIMITATIONS = (
    (
        "This preserves the fast-ridge-intrinsic periodic-MH schedule and "
        "adds four gamma-one complementary live-point differential-evolution "
        "Metropolis attempts on the eight-dimensional intrinsic block."
    ),
    *PAPER_LIMITATIONS[1:],
)
PAPER_FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_LIMITATIONS = (
    (
        "This preserves the fast-ridge-intrinsic periodic-MH schedule and "
        "adds eight gamma-one complementary live-point differential-evolution "
        "Metropolis attempts on the eight-dimensional intrinsic block."
    ),
    *PAPER_LIMITATIONS[1:],
)

LIMITATIONS = ALIGNED_LIMITATIONS


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _de_jump_block(value: str) -> dict[str, Any]:
    """Parse ``name[,name...]:attempts`` into a sampler config value."""
    names_text, separator, attempts_text = value.rpartition(":")
    names = names_text.split(",") if separator else []
    if not names or any(not name.isidentifier() for name in names):
        raise argparse.ArgumentTypeError(
            "must use name[,name...]:attempts with valid parameter names"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("parameter names must be unique")
    attempts = _positive_int(attempts_text)
    return {"parameters": names, "attempts": attempts}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help=(
            "Frozen .npz input bundle. A full run only reads this file. Use "
            "--prepare-data once to create it from GWOSC."
        ),
    )
    parser.add_argument(
        "--workload",
        choices=WORKLOAD_CHOICES,
        default=ALIGNED_WORKLOAD,
        help=(
            "Benchmark model: the historical 11D aligned-spin analogue or "
            "the paper's full 15D precessing-tidal GW170817 analysis."
        ),
    )
    parser.add_argument(
        "--blocking-scheme",
        choices=BLOCKING_SCHEME_CHOICES,
        default=PAPER_BLOCKING_SCHEME,
        help=(
            "Sampler partition. all-slow groups masses, tides, both spin "
            "spheres, and iota into one 11D paper-workload block; iota-dl "
            "keeps the paper partition but merges the iota and d_L "
            "singletons into one joint block; fast-ridge uses the joint "
            "cacheable (cos_iota, d_hat) coordinates; fast-ridge-intrinsic "
            "also couples the eight non-periodic intrinsic coordinates while "
            "keeping both spin azimuths as periodic singletons; "
            "fast-ridge-intrinsic-5step uses the same blocks with five random "
            "covariance directions in the eight-dimensional block; "
            "fast-ridge-intrinsic-periodic-mh instead keeps the default "
            "dimension budget and replaces the three periodic singleton "
            "slices with one-call uniform independence updates; the cde4 and "
            "cde8 variants additionally insert four or eight gamma-one "
            "complementary-live DE-MH attempts after the eight-dimensional "
            "intrinsic block."
        ),
    )
    parser.add_argument(
        "--adaptive-slice-widths",
        action="store_true",
        help=(
            "Adapt each block's slice-direction width online from its "
            "stepping-out expansion / shrinkage history instead of using a "
            "fixed unit width. Requires the fsm scheduler and covariance "
            "direction mode."
        ),
    )
    parser.add_argument(
        "--bracket-mode",
        choices=("stepping-out", "shrink-only"),
        default="stepping-out",
        help=(
            "Slice-sampling bracket construction. shrink-only skips "
            "stepping-out expansion entirely (Neal 2003 Sec. 4.1) and "
            "requires --adaptive-slice-widths."
        ),
    )
    parser.add_argument(
        "--width-adaptation-rate",
        type=float,
        default=0.25,
        help="Log-width step size per replacement for --adaptive-slice-widths.",
    )
    parser.add_argument(
        "--width-target-expansions",
        type=float,
        default=1.0,
        help="Target mean stepping-out expansions per slice for the width controller.",
    )
    parser.add_argument(
        "--width-target-shrinks",
        type=float,
        default=3.0,
        help="Target mean shrinkage steps per slice for the width controller.",
    )
    parser.add_argument(
        "--direction-mode",
        choices=("covariance", "de-mix", "covariance-basis-8d"),
        default="covariance",
        help=(
            "Slice-direction proposal. de-mix replaces a de_fraction of the "
            "block covariance chords with differences of random live-point "
            "pairs so cross-basin moves stay reachable; covariance-basis-8d "
            "uses each signed, randomly permuted Cholesky column exactly once "
            "in the unique eight-dimensional block."
        ),
    )
    parser.add_argument(
        "--de-fraction",
        type=float,
        default=0.5,
        help="Fraction of de-mix slice directions drawn from live-point pairs.",
    )
    parser.add_argument(
        "--num-de-jumps",
        type=_nonnegative_int,
        default=0,
        help=(
            "Metropolis differential-evolution jumps per replacement (full "
            "live-pair displacements; one likelihood evaluation each) so "
            "cross-basin transitions bypass slice shrinkage."
        ),
    )
    parser.add_argument(
        "--de-jump-block",
        action="append",
        type=_de_jump_block,
        default=[],
        metavar="PARAM[,PARAM...]:ATTEMPTS",
        help=(
            "Repeatable named-coordinate DE move group. Unlike --blocking-scheme, "
            "this does not alter the slice partition."
        ),
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help=(
            "Fetch GW170817 strain and PSD source data, write --data-file, "
            "report its SHA-256, and exit without sampling. If a valid bundle "
            "already exists it is reused without fetching."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-gibbs-sweeps",
        type=_positive_int,
        default=NUM_GIBBS_SWEEPS,
        help="Paper-notation M: complete Gibbs sweeps per replacement.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=_nonnegative_int,
        default=None,
        help=(
            "Optional independent sampler-trajectory seed. --seed continues "
            "to determine the initial live points; omitting this option "
            "preserves the historical shared-seed path exactly."
        ),
    )
    parser.add_argument(
        "--n-devices",
        type=int,
        choices=SUPPORTED_DEVICE_COUNTS,
        default=N_DEVICES,
        help=(
            "Visible-device count. Four is the primary comparison; one and two "
            "are supported for the candidate scaling sweep."
        ),
    )
    parser.add_argument(
        "--implementation-root",
        type=Path,
        default=None,
        help=(
            "Optional checkout containing src/jimgw. When omitted, import "
            "resolution is controlled by PYTHONPATH/the active environment."
        ),
    )
    parser.add_argument(
        "--implementation-label",
        default="candidate",
        help="Human-readable implementation label stored in the report.",
    )
    parser.add_argument(
        "--implementation-revision",
        default=None,
        help="Known Git revision for a source export without .git metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also atomically write the JSON report to this path.",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help=(
            "Write a JAX profiler trace for --profile-steps steady outer steps "
            "after --profile-warmup-steps."
        ),
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=_nonnegative_int,
        default=10,
        help="Outer steps to complete before starting the profiler trace.",
    )
    parser.add_argument(
        "--profile-steps",
        type=_positive_int,
        default=15,
        help="Number of consecutive outer steps to include in the trace.",
    )
    parser.add_argument(
        "--slice-data-output",
        type=Path,
        default=None,
        help=(
            "Write per-chain, per-slice num_expansions and num_shrink arrays "
            "to an NPZ artifact."
        ),
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=None,
        help=(
            "Write all equally weighted posterior samples in prior space, "
            "including log_likelihood, to an NPZ artifact."
        ),
    )
    parser.add_argument(
        "--nested-output",
        type=Path,
        default=None,
        help=(
            "Write the full weighted nested-point collection in prior space, "
            "including birth/death log-likelihoods and normalized log weights, "
            "to an NPZ artifact."
        ),
    )
    parser.add_argument(
        "--retain-per-slice-info",
        action="store_true",
        help=(
            "Retain per-slice counters in the compiled kernel without writing "
            "an NPZ. Used to make a cache probe's HLO match an instrumented run."
        ),
    )
    parser.add_argument(
        "--jax-compilation-cache-dir",
        type=Path,
        default=None,
        help="Persistent JAX compilation-cache directory for this implementation.",
    )
    parser.add_argument(
        "--telemetry-output",
        type=Path,
        default=None,
        help=(
            "Capture nvidia-smi dmon telemetry during the profiled steady-state "
            "window. Requires --profile-dir."
        ),
    )
    parser.add_argument(
        "--max-outer-steps",
        type=_positive_int,
        default=None,
        help=(
            "Benchmark-only early stop after this many outer steps. Intended for "
            "persistent-cache probes, not scientific runs."
        ),
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help=(
            "Measure the unobstructed AOT-compiled sampling loop only. This "
            "disables the outer-step observer and skips posterior extraction."
        ),
    )
    parser.add_argument(
        "--ablation-variant",
        choices=SAMPLER_ABLATION_VARIANTS,
        default=None,
        help="Process-local sampler implementation cell used by --timing-only.",
    )
    parser.add_argument(
        "--simulate-cpu",
        action="store_true",
        help=(
            "Expose four logical CPU devices for smoke tests. Results from "
            "this mode are not GPU benchmark measurements."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.blocking_scheme != PAPER_BLOCKING_SCHEME
        and args.workload != PAPER_WORKLOAD
    ):
        parser.error("non-paper --blocking-scheme requires --workload paper-15d")
    if (
        args.direction_mode == "covariance-basis-8d"
        and args.blocking_scheme != FAST_RIDGE_INTRINSIC_BLOCKING_SCHEME
    ):
        parser.error(
            "--direction-mode covariance-basis-8d requires "
            "--blocking-scheme fast-ridge-intrinsic"
        )
    if (
        args.blocking_scheme == FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME
        and args.direction_mode != "covariance"
    ):
        parser.error(
            "--blocking-scheme fast-ridge-intrinsic-5step requires "
            "--direction-mode covariance"
        )
    if (
        args.blocking_scheme == FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME
        and args.direction_mode != "covariance"
    ):
        parser.error(
            "--blocking-scheme fast-ridge-intrinsic-periodic-mh requires "
            "--direction-mode covariance"
        )
    if (
        args.blocking_scheme in COMPLEMENTARY_DE_BLOCKING_SCHEMES
        and args.direction_mode != "covariance"
    ):
        parser.error(
            f"--blocking-scheme {args.blocking_scheme} requires --direction-mode "
            "covariance"
        )
    if (
        args.blocking_scheme in COMPLEMENTARY_DE_BLOCKING_SCHEMES
        and args.n_devices != 1
    ):
        parser.error(f"--blocking-scheme {args.blocking_scheme} requires --n-devices 1")
    if args.blocking_scheme in COMPLEMENTARY_DE_BLOCKING_SCHEMES and (
        args.num_de_jumps != 0 or args.de_jump_block
    ):
        parser.error(
            f"--blocking-scheme {args.blocking_scheme} fixes the complementary-DE "
            "schedule and forbids legacy DE moves"
        )
    if args.telemetry_output is not None and args.profile_dir is None:
        parser.error("--telemetry-output requires --profile-dir")
    if args.timing_only != (args.ablation_variant is not None):
        parser.error("--timing-only and --ablation-variant must be used together")
    if args.timing_only:
        incompatible = {
            "--prepare-data": args.prepare_data,
            "--profile-dir": args.profile_dir is not None,
            "--slice-data-output": args.slice_data_output is not None,
            "--samples-output": args.samples_output is not None,
            "--nested-output": args.nested_output is not None,
            "--retain-per-slice-info": args.retain_per_slice_info,
            "--jax-compilation-cache-dir": args.jax_compilation_cache_dir is not None,
            "--telemetry-output": args.telemetry_output is not None,
            "--max-outer-steps": args.max_outer_steps is not None,
            "--verbose": args.verbose,
        }
        selected = [flag for flag, enabled in incompatible.items() if enabled]
        if selected:
            parser.error("--timing-only is incompatible with " + ", ".join(selected))
    return args


def _select_implementation(root: Path | None) -> None:
    if root is None:
        return
    resolved = root.expanduser().resolve()
    source = resolved / "src"
    if not (source / "jimgw").is_dir():
        raise SystemExit(
            f"--implementation-root must contain src/jimgw; got {resolved}"
        )
    sys.path.insert(0, str(source))


def _configure_cpu_simulation(n_devices: int) -> None:
    desired = f"--xla_force_host_platform_device_count={n_devices}"
    existing = os.environ.get("XLA_FLAGS", "").split()
    configured = [
        token
        for token in existing
        if token.startswith("--xla_force_host_platform_device_count=")
    ]
    if configured and configured != [desired]:
        raise SystemExit(
            "XLA_FLAGS already contains a conflicting host device count: "
            + " ".join(configured)
        )
    if not configured:
        os.environ["XLA_FLAGS"] = " ".join([*existing, desired]).strip()
    configured_platforms = os.environ.get("JAX_PLATFORMS")
    if configured_platforms and configured_platforms.split(",", 1)[0] != "cpu":
        raise SystemExit(
            "--simulate-cpu conflicts with JAX_PLATFORMS=" + configured_platforms
        )
    os.environ["JAX_PLATFORMS"] = "cpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _data_window(workload: str) -> tuple[float, float, float, float]:
    if workload == ALIGNED_WORKLOAD:
        analysis_start = ALIGNED_START
        psd_start = analysis_start - 2048.0
    elif workload == PAPER_WORKLOAD:
        analysis_start = PAPER_START
        # The cleaned event product begins here.  Its complete prefix before
        # the trigger-centred analysis window is 1778.43 seconds, conventionally
        # described as 1778 seconds.  Keeping the PSD inside this file avoids
        # mixing the CLN analysis strain with the run-wide O2 product.
        psd_start = GWOSC_EVENT_FILE_START_GPS
    else:
        raise ValueError(f"unknown workload: {workload}")
    analysis_end = analysis_start + DURATION
    psd_end = analysis_start
    return analysis_start, analysis_end, psd_start, psd_end


def _analysis_configuration(workload: str) -> dict[str, Any]:
    """Return the pinned analysis-strain product for a workload."""

    if workload == ALIGNED_WORKLOAD:
        product = GWOSC_ALIGNED_ANALYSIS_STRAIN_PRODUCT
        sample_rate_hz = GWOSC_ALIGNED_SAMPLE_RATE_HZ
    elif workload == PAPER_WORKLOAD:
        product = GWOSC_PAPER_ANALYSIS_STRAIN_PRODUCT
        sample_rate_hz = GWOSC_PAPER_SAMPLE_RATE_HZ
    else:
        raise ValueError(f"unknown workload: {workload}")
    return {
        "dataset": GWOSC_ANALYSIS_STRAIN_DATASET,
        "product": product,
        "sample_rate_hz": sample_rate_hz,
        "source_urls": {
            ifo: _cleaned_strain_url(ifo, product=product) for ifo in IFO_NAMES
        },
    }


def _psd_configuration(workload: str) -> dict[str, Any]:
    """Return workload-specific PSD product and Welch aggregation settings."""

    if workload == ALIGNED_WORKLOAD:
        return {
            "dataset": GWOSC_PSD_STRAIN_DATASET,
            "product": GWOSC_PSD_STRAIN_PRODUCT,
            "sample_rate_hz": GWOSC_ALIGNED_SAMPLE_RATE_HZ,
            "source_urls": {},
            "average": "mean",
            "median_bias_correction": None,
            "postprocessing": None,
            "smoothing_width_hz": None,
            "line_protection_ratio": None,
        }
    if workload == PAPER_WORKLOAD:
        analysis = _analysis_configuration(workload)
        return {
            "dataset": analysis["dataset"],
            "product": analysis["product"],
            "sample_rate_hz": analysis["sample_rate_hz"],
            "source_urls": analysis["source_urls"],
            "average": "median",
            # scipy.signal.welch divides the segment median by its standard
            # FINDCHIRP bias factor.  Do not apply a second correction here.
            "median_bias_correction": "scipy.signal.welch built-in",
            "postprocessing": PSD_SMOOTHING_METHOD,
            "smoothing_width_hz": PSD_SMOOTHING_WIDTH_HZ,
            "line_protection_ratio": PSD_LINE_PROTECTION_RATIO,
        }
    raise ValueError(f"unknown workload: {workload}")


def _cleaned_strain_url(
    ifo: str,
    *,
    product: str = GWOSC_PAPER_ANALYSIS_STRAIN_PRODUCT,
) -> str:
    if ifo not in IFO_NAMES:
        raise ValueError(f"unknown interferometer: {ifo}")
    return (
        f"{GWOSC_ANALYSIS_STRAIN_API_BASE_URL}/{ifo[0]}-{ifo}_"
        f"{product}-{GWOSC_EVENT_FILE_START_GPS:.0f}-"
        f"{GWOSC_EVENT_FILE_DURATION_SECONDS:.0f}.hdf5"
    )


def _data_manifest(workload: str = ALIGNED_WORKLOAD) -> dict[str, Any]:
    analysis_start, analysis_end, psd_start, psd_end = _data_window(workload)
    analysis_configuration = _analysis_configuration(workload)
    psd_configuration = _psd_configuration(workload)
    return {
        "format_version": DATA_FORMAT_VERSION,
        "workload": workload,
        "event": "GW170817",
        "source": "GWOSC public strain with an explicitly pinned clean event product",
        "detectors": list(IFO_NAMES),
        "gps": GPS,
        "duration_seconds": DURATION,
        "analysis_start_gps": analysis_start,
        "analysis_end_gps": analysis_end,
        "psd_start_gps": psd_start,
        "psd_end_gps": psd_end,
        "psd_duration_seconds": psd_end - psd_start,
        "f_min_hz": F_MIN,
        "nominal_f_max_hz": NOMINAL_F_MAX,
        "likelihood_f_max_hz": F_MAX,
        "psd_estimator": "scipy.signal.welch",
        "psd_nperseg": "analysis_n_time",
        "psd_average": psd_configuration["average"],
        "psd_median_bias_correction": psd_configuration["median_bias_correction"],
        "psd_postprocessing": psd_configuration["postprocessing"],
        "psd_smoothing_width_hz": psd_configuration["smoothing_width_hz"],
        "psd_line_protection_ratio": psd_configuration["line_protection_ratio"],
        "analysis_strain_dataset": analysis_configuration["dataset"],
        "analysis_strain_release": GWOSC_ANALYSIS_STRAIN_RELEASE,
        "analysis_strain_product": analysis_configuration["product"],
        "analysis_strain_glitch_mitigation": ("L1 glitch removed by the GWOSC release"),
        "analysis_strain_documentation_url": (GWOSC_ANALYSIS_STRAIN_DOCUMENTATION_URL),
        "analysis_strain_source_urls": analysis_configuration["source_urls"],
        "psd_strain_dataset": psd_configuration["dataset"],
        "psd_strain_product": psd_configuration["product"],
        "psd_strain_source_urls": psd_configuration["source_urls"],
        "gwosc_sample_rate_hz": analysis_configuration["sample_rate_hz"],
        "time_marginalization_fft_sample_rate_hz": (
            TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ
        ),
        "gwosc_format": GWOSC_FORMAT,
        "known_l1_glitch_gps": L1_GLITCH_GPS,
        "known_l1_glitch_window_half_width_seconds": (
            L1_GLITCH_WINDOW_HALF_WIDTH_SECONDS
        ),
        "known_l1_glitch_max_whitened_amplitude": (L1_GLITCH_MAX_WHITENED_AMPLITUDE),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "gwpy_version": _package_version("gwpy"),
    }


def _required_data_keys() -> set[str]:
    keys = {"manifest_json"}
    for name in IFO_NAMES:
        keys.update(
            {
                f"{name}_strain_td",
                f"{name}_strain_delta_t",
                f"{name}_strain_start_time",
                f"{name}_psd_values",
                f"{name}_psd_frequencies",
            }
        )
    return keys


def _read_bundle(
    path: Path,
    workload: str = ALIGNED_WORKLOAD,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not path.is_file():
        raise SystemExit(
            f"Frozen data file does not exist: {path}. "
            "Create it first with --prepare-data."
        )
    if path.suffix.lower() != ".npz":
        raise SystemExit(f"--data-file must use the .npz extension; got {path}")

    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(_required_data_keys() - set(archive.files))
            if missing:
                raise ValueError("missing keys: " + ", ".join(missing))
            manifest = json.loads(str(archive["manifest_json"].item()))
            arrays = {
                key: np.array(archive[key], copy=True)
                for key in archive.files
                if key != "manifest_json"
            }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid frozen data bundle {path}: {error}") from error

    expected = _data_manifest(workload)
    invariant_keys = (
        "format_version",
        "workload",
        "event",
        "detectors",
        "gps",
        "duration_seconds",
        "analysis_start_gps",
        "analysis_end_gps",
        "psd_start_gps",
        "psd_end_gps",
        "psd_duration_seconds",
        "f_min_hz",
        "nominal_f_max_hz",
        "likelihood_f_max_hz",
        "psd_estimator",
        "psd_nperseg",
        "psd_average",
        "psd_median_bias_correction",
        "psd_postprocessing",
        "psd_smoothing_width_hz",
        "psd_line_protection_ratio",
        "analysis_strain_dataset",
        "analysis_strain_release",
        "analysis_strain_product",
        "analysis_strain_glitch_mitigation",
        "analysis_strain_documentation_url",
        "analysis_strain_source_urls",
        "psd_strain_dataset",
        "psd_strain_product",
        "psd_strain_source_urls",
        "gwosc_sample_rate_hz",
        "time_marginalization_fft_sample_rate_hz",
        "gwosc_format",
        "known_l1_glitch_gps",
        "known_l1_glitch_window_half_width_seconds",
        "known_l1_glitch_max_whitened_amplitude",
    )
    mismatched = [key for key in invariant_keys if manifest.get(key) != expected[key]]
    if mismatched:
        raise SystemExit(
            f"Frozen data bundle {path} has incompatible manifest fields: "
            + ", ".join(mismatched)
        )

    for name in IFO_NAMES:
        td = arrays[f"{name}_strain_td"]
        delta_t = float(arrays[f"{name}_strain_delta_t"])
        start_time = float(arrays[f"{name}_strain_start_time"])
        psd = arrays[f"{name}_psd_values"]
        frequencies = arrays[f"{name}_psd_frequencies"]
        if td.ndim != 1 or psd.ndim != 1 or frequencies.ndim != 1:
            raise SystemExit(f"Frozen {name} arrays must all be one-dimensional")
        if not np.isfinite(delta_t) or delta_t <= 0:
            raise SystemExit(f"Frozen {name} strain has invalid delta_t={delta_t}")
        # GWPy crops on the source sample grid, so a fractional requested GPS
        # start can be rounded down by less than one sample interval.
        if abs(start_time - expected["analysis_start_gps"]) > delta_t:
            raise SystemExit(
                f"Frozen {name} strain starts at {start_time}, expected "
                f"{expected['analysis_start_gps']}"
            )
        if not np.isclose(td.size * delta_t, DURATION, rtol=0.0, atol=1e-9):
            raise SystemExit(
                f"Frozen {name} strain duration is {td.size * delta_t}, "
                f"expected {DURATION}"
            )
        if not np.all(np.isfinite(td)):
            raise SystemExit(f"Frozen {name} strain contains non-finite values")
        if not np.all(np.isfinite(psd)) or np.any(psd <= 0.0):
            raise SystemExit(f"Frozen {name} PSD must be finite and positive")
        expected_frequencies = np.fft.rfftfreq(td.size, delta_t)
        if psd.shape != frequencies.shape or not np.array_equal(
            frequencies, expected_frequencies
        ):
            raise SystemExit(
                f"Frozen {name} PSD grid does not exactly match the strain grid"
            )
    _validate_known_l1_glitch_window(arrays)
    return manifest, arrays


def _validate_known_l1_glitch_window(arrays: dict[str, Any]) -> float:
    """Reject L1 data containing the documented pre-merger GW170817 glitch.

    The strain is whitened with the exact PSD and frequency band that will be
    used by this benchmark.  This is deliberately an event-specific guard,
    rather than a generic Gaussianity assertion: clean public strain still
    contains ordinary non-stationarity, while the raw L1 glitch is separated
    from the clean release by more than an order of magnitude at this seam.

    Returns:
        Maximum absolute whitened amplitude in the known glitch window.
    """

    td = np.asarray(arrays["L1_strain_td"], dtype=np.float64)
    delta_t = float(arrays["L1_strain_delta_t"])
    start_time = float(arrays["L1_strain_start_time"])
    psd = np.asarray(arrays["L1_psd_values"], dtype=np.float64)
    frequencies = np.asarray(arrays["L1_psd_frequencies"], dtype=np.float64)

    if td.ndim != 1 or psd.shape != frequencies.shape:
        raise SystemExit("Cannot validate the known GW170817 L1 glitch window")
    expected_frequencies = np.fft.rfftfreq(td.size, delta_t)
    if psd.shape != expected_frequencies.shape or not np.array_equal(
        frequencies, expected_frequencies
    ):
        raise SystemExit("Cannot validate the known GW170817 L1 glitch window")

    from scipy.signal.windows import tukey

    duration = td.size * delta_t
    alpha = min(1.0, 2.0 * DATA_TUKEY_ROLL_OFF_SECONDS / duration)
    strain_fd = np.fft.rfft(td * tukey(td.size, alpha=alpha)) * delta_t
    native_sample_rate_hz = 1.0 / delta_t
    guard_sample_rate_hz = min(
        native_sample_rate_hz, TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ
    )
    guard_n_time_float = duration * guard_sample_rate_hz
    guard_n_time = round(guard_n_time_float)
    if not np.isclose(guard_n_time_float, guard_n_time, rtol=0.0, atol=1e-9):
        raise SystemExit("Cannot validate the known GW170817 L1 glitch window")
    guard_delta_t = 1.0 / guard_sample_rate_hz
    guard_frequencies = np.fft.rfftfreq(guard_n_time, guard_delta_t)
    guard_f_max = min(F_MAX, guard_sample_rate_hz / 2.0 - 1.0 / duration)
    source_mask = (frequencies >= F_MIN) & (frequencies <= guard_f_max)
    guard_mask = (guard_frequencies >= F_MIN) & (guard_frequencies <= guard_f_max)
    if not np.any(source_mask) or not np.array_equal(
        frequencies[source_mask], guard_frequencies[guard_mask]
    ):
        raise SystemExit(
            "Cannot validate the known GW170817 L1 glitch window: "
            "the strain grid does not include the analysis band"
        )
    whitened_fd = np.zeros(guard_frequencies.size, dtype=strain_fd.dtype)
    whitening_scale = np.sqrt(psd[source_mask] * guard_delta_t / 2.0)
    whitened_fd[guard_mask] = strain_fd[source_mask] / whitening_scale
    whitened_td = np.fft.irfft(whitened_fd, n=guard_n_time)

    times = start_time + np.arange(guard_n_time, dtype=np.float64) * guard_delta_t
    glitch_window = np.abs(times - L1_GLITCH_GPS) <= L1_GLITCH_WINDOW_HALF_WIDTH_SECONDS
    if not np.any(glitch_window):
        raise SystemExit(
            "Cannot validate the known GW170817 L1 glitch window: "
            "the strain grid has no sample in the guard interval"
        )
    peak = float(np.max(np.abs(whitened_td[glitch_window])))
    if not np.isfinite(peak) or peak > L1_GLITCH_MAX_WHITENED_AMPLITUDE:
        raise SystemExit(
            "Frozen L1 strain fails the known GW170817 L1 glitch window guard: "
            f"max |z|={peak:.3f}, limit={L1_GLITCH_MAX_WHITENED_AMPLITUDE:.1f}. "
            f"Recreate it from {GWOSC_ANALYSIS_STRAIN_DATASET}."
        )
    return peak


def _atomic_save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".npz":
        raise SystemExit(f"--data-file must use the .npz extension; got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _line_protected_log_median_psd(
    values: np.ndarray,
    frequencies: np.ndarray,
    *,
    width_hz: float = PSD_SMOOTHING_WIDTH_HZ,
    line_protection_ratio: float = PSD_LINE_PROTECTION_RATIO,
) -> np.ndarray:
    """Remove finite-Welch-segment bin noise without erasing spectral lines.

    The broad-band trend is a running median in log PSD. Bins whose original
    Welch value exceeds twice that local trend remain untouched so narrow
    instrumental lines continue to be down-weighted by the likelihood.
    """

    values = np.asarray(values, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 1 or frequencies.ndim != 1 or values.shape != frequencies.shape:
        raise ValueError("PSD values and frequencies must be same-length 1D arrays")
    if values.size < 3 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("PSD values must contain at least three finite positive bins")
    frequency_steps = np.diff(frequencies)
    if (
        not np.all(np.isfinite(frequencies))
        or np.any(frequency_steps <= 0.0)
        or not np.allclose(
            frequency_steps,
            frequency_steps[0],
            rtol=1e-10,
            atol=0.0,
        )
    ):
        raise ValueError("PSD frequencies must be finite, increasing, and uniform")
    if not np.isfinite(width_hz) or width_hz <= 0.0:
        raise ValueError("PSD smoothing width must be finite and positive")
    if not np.isfinite(line_protection_ratio) or line_protection_ratio <= 1.0:
        raise ValueError(
            "PSD line-protection ratio must be finite and greater than one"
        )

    window_bins = max(3, round(width_hz / frequency_steps[0]))
    if window_bins % 2 == 0:
        window_bins += 1
    smooth_trend = np.exp(
        median_filter(np.log(values), size=window_bins, mode="reflect")
    )
    return np.where(
        values > line_protection_ratio * smooth_trend,
        values,
        smooth_trend,
    )


def _prepare_data(
    path: Path,
    workload: str = ALIGNED_WORKLOAD,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        manifest, loaded_arrays = _read_bundle(resolved, workload)
        return {
            "path": str(resolved),
            "sha256": _sha256(resolved),
            "bytes": resolved.stat().st_size,
            "reused_existing": True,
            "manifest": manifest,
            "sample_counts": {
                name: int(loaded_arrays[f"{name}_strain_td"].size) for name in IFO_NAMES
            },
        }

    import jax

    jax.config.update("jax_enable_x64", True)

    from jimgw.core.single_event.data import Data

    analysis_start, analysis_end, psd_start, psd_end = _data_window(workload)
    analysis_configuration = _analysis_configuration(workload)
    psd_configuration = _psd_configuration(workload)
    arrays: dict[str, Any] = {}
    for name in IFO_NAMES:
        strain = Data.from_gwosc(
            name,
            analysis_start,
            analysis_end,
            dataset=analysis_configuration["dataset"],
            sample_rate=analysis_configuration["sample_rate_hz"],
            format=GWOSC_FORMAT,
        )
        psd_source = Data.from_gwosc(
            name,
            psd_start,
            psd_end,
            dataset=psd_configuration["dataset"],
            sample_rate=psd_configuration["sample_rate_hz"],
            format=GWOSC_FORMAT,
        )
        psd = psd_source.to_psd(
            nperseg=int(DURATION * psd_source.sampling_frequency),
            average=psd_configuration["average"],
        )
        psd_values = np.asarray(psd.values)
        if psd_configuration["postprocessing"] == PSD_SMOOTHING_METHOD:
            psd_values = _line_protected_log_median_psd(
                psd_values,
                np.asarray(psd.frequencies),
                width_hz=psd_configuration["smoothing_width_hz"],
                line_protection_ratio=psd_configuration["line_protection_ratio"],
            )
        arrays.update(
            {
                f"{name}_strain_td": np.asarray(strain.td),
                f"{name}_strain_delta_t": np.asarray(float(strain.delta_t)),
                f"{name}_strain_start_time": np.asarray(strain.start_time),
                f"{name}_psd_values": psd_values,
                f"{name}_psd_frequencies": np.asarray(psd.frequencies),
            }
        )

    _validate_known_l1_glitch_window(arrays)
    arrays["manifest_json"] = np.asarray(
        json.dumps(_data_manifest(workload), sort_keys=True)
    )

    _atomic_save_npz(resolved, arrays)
    manifest, loaded = _read_bundle(resolved, workload)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "reused_existing": False,
        "manifest": manifest,
        "sample_counts": {
            name: int(loaded[f"{name}_strain_td"].size) for name in IFO_NAMES
        },
    }


def _git_metadata(
    repository: Path | None, revision_override: str | None
) -> dict[str, Any]:
    def git(root: Path, *arguments: str) -> str | None:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    if repository is None:
        return {
            "repository": None,
            "revision": revision_override,
            "dirty": None,
        }
    root = repository.expanduser().resolve()
    if not (root / ".git").exists():
        return {
            "repository": str(root),
            "revision": revision_override,
            "dirty": None,
        }
    status = git(root, "status", "--porcelain")
    return {
        "repository": str(root),
        "revision": revision_override or git(root, "rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _device_metadata(jax: Any, n_devices: int) -> dict[str, Any]:
    devices = jax.local_devices()
    if len(devices) != n_devices:
        raise SystemExit(
            f"Benchmark requires exactly {n_devices} visible local devices; "
            f"JAX sees {len(devices)}. Mask the process to four GPUs."
        )
    return {
        "backend": jax.default_backend(),
        "requested_count": n_devices,
        "local_count": jax.local_device_count(),
        "global_count": jax.device_count(),
        "process_count": jax.process_count(),
        "devices": [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
            }
            for device in devices
        ],
    }


def _workload_spec(
    workload: str,
    blocking_scheme: str = PAPER_BLOCKING_SCHEME,
) -> dict[str, Any]:
    if blocking_scheme not in BLOCKING_SCHEME_CHOICES:
        raise ValueError(
            f"unknown blocking scheme {blocking_scheme!r}; "
            f"expected one of {BLOCKING_SCHEME_CHOICES}"
        )
    if workload == ALIGNED_WORKLOAD:
        if blocking_scheme != PAPER_BLOCKING_SCHEME:
            raise ValueError("non-standard blocking requires the paper-15d workload")
        return {
            "waveform": "IMRPhenomD_NRTidalv2",
            "sampled_dimensions": 11,
            "blocks": ALIGNED_BLOCKS,
            "mass_ratio_range": (0.5, 1.0),
            "distance_range_mpc": (30.0, 150.0),
            "spin_parameterization": "aligned Cartesian z components",
            "limitations": ALIGNED_LIMITATIONS,
        }
    if workload == PAPER_WORKLOAD:
        if blocking_scheme == ALL_SLOW_BLOCKING_SCHEME:
            blocks = PAPER_ALL_SLOW_BLOCKS
            limitations = PAPER_ALL_SLOW_LIMITATIONS
        elif blocking_scheme == IOTA_DL_BLOCKING_SCHEME:
            blocks = PAPER_IOTA_DL_BLOCKS
            limitations = PAPER_IOTA_DL_LIMITATIONS
        elif blocking_scheme == FAST_RIDGE_BLOCKING_SCHEME:
            blocks = PAPER_FAST_RIDGE_BLOCKS
            limitations = PAPER_FAST_RIDGE_LIMITATIONS
        elif blocking_scheme == FAST_RIDGE_INTRINSIC_BLOCKING_SCHEME:
            blocks = PAPER_FAST_RIDGE_INTRINSIC_BLOCKS
            limitations = PAPER_FAST_RIDGE_INTRINSIC_LIMITATIONS
        elif blocking_scheme == FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME:
            blocks = PAPER_FAST_RIDGE_INTRINSIC_BLOCKS
            limitations = PAPER_FAST_RIDGE_INTRINSIC_5STEP_LIMITATIONS
        elif blocking_scheme == FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME:
            blocks = PAPER_FAST_RIDGE_INTRINSIC_BLOCKS
            limitations = PAPER_FAST_RIDGE_INTRINSIC_PERIODIC_MH_LIMITATIONS
        elif blocking_scheme == (FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME):
            blocks = PAPER_FAST_RIDGE_INTRINSIC_BLOCKS
            limitations = PAPER_FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_LIMITATIONS
        elif blocking_scheme == (FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME):
            blocks = PAPER_FAST_RIDGE_INTRINSIC_BLOCKS
            limitations = PAPER_FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_LIMITATIONS
        else:
            blocks = PAPER_BLOCKS
            limitations = PAPER_LIMITATIONS
        return {
            "waveform": "IMRPhenomPv2_NRTidalv2",
            "sampled_dimensions": 15,
            "blocks": blocks,
            "mass_ratio_range": (0.125, 1.0),
            "distance_range_mpc": (1.0, 75.0),
            "spin_parameterization": (
                "two isotropic spin spheres with magnitudes in [0, 0.05]"
            ),
            "limitations": limitations,
        }
    raise ValueError(
        f"unknown workload {workload!r}; expected one of {WORKLOAD_CHOICES}"
    )


def _config_report(
    seed: int,
    n_devices: int,
    workload: str = ALIGNED_WORKLOAD,
    blocking_scheme: str = PAPER_BLOCKING_SCHEME,
    direction_mode: str = "covariance",
    de_fraction: float = 0.5,
    num_de_jumps: int = 0,
    de_jump_blocks: list[dict[str, Any]] | None = None,
    adaptive_slice_widths: bool = False,
    bracket_mode: str = "stepping-out",
    width_adaptation_rate: float = 0.25,
    width_target_expansions: float = 1.0,
    width_target_shrinks: float = 3.0,
    sampler_seed: int | None = None,
) -> dict[str, Any]:
    spec = _workload_spec(workload, blocking_scheme)
    canonical_de_jump_blocks = [
        {
            "parameters": list(block["parameters"]),
            "attempts": int(block["attempts"]),
        }
        for block in (de_jump_blocks or [])
    ]
    config: dict[str, Any] = {
        "seed": seed,
        "workload": workload,
        "blocking_scheme": blocking_scheme,
        "event": "GW170817",
        "detectors": list(IFO_NAMES),
        "duration_seconds": DURATION,
        "f_min_hz": F_MIN,
        "nominal_f_max_hz": NOMINAL_F_MAX,
        "likelihood_f_max_hz": F_MAX,
        "waveform": spec["waveform"],
        "waveform_f_ref_hz": 20.0,
        "sampled_dimensions": spec["sampled_dimensions"],
        "priors": {
            "M_c": {"distribution": "uniform", "range": [1.18, 1.21]},
            "q": {
                "distribution": "uniform",
                "range": list(spec["mass_ratio_range"]),
            },
            "spins": spec["spin_parameterization"],
            "iota": "isotropic",
            "lambda_1": {"distribution": "uniform", "range": [0.0, 5000.0]},
            "lambda_2": {"distribution": "uniform", "range": [0.0, 5000.0]},
            "d_L": {
                "distribution": "power-law",
                "alpha": 2.0,
                "range_mpc": list(spec["distance_range_mpc"]),
            },
            "sky": "isotropic",
            "psi": {"distribution": "uniform", "range_radians": [0.0, "pi"]},
        },
        "phase_marginalization": True,
        "time_marginalization_tc_range_seconds": [-0.03, 0.03],
        "time_marginalization_fft_sample_rate_hz": (
            TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ
        ),
        "distance_marginalization": False,
        "n_devices": n_devices,
        "n_live": N_LIVE,
        "n_delete": N_DELETE,
        "n_delete_frac": N_DELETE_FRAC,
        "num_inner_steps_per_dim": NUM_INNER_STEPS_PER_DIM,
        "num_gibbs_sweeps": NUM_GIBBS_SWEEPS,
        "termination_dlogz": TERMINATION_DLOGZ,
        "blocks": [list(block) for block in spec["blocks"]],
        "dtype": "float64",
        "paper_notation": {
            "D_devices": n_devices,
            "m_live_points": N_LIVE,
            "k_deleted_points": N_DELETE,
            "M_gibbs_sweeps": NUM_GIBBS_SWEEPS,
        },
    }
    if sampler_seed is not None:
        config["sampler_seed"] = sampler_seed
    if direction_mode != "covariance" or num_de_jumps > 0 or canonical_de_jump_blocks:
        # Only non-default proposal settings enter the config (and its hash)
        # so historical covariance-mode reports stay byte-identical.
        config["direction_mode"] = direction_mode
        config["de_fraction"] = de_fraction
        config["num_de_jumps"] = num_de_jumps
        config["de_jump_blocks"] = canonical_de_jump_blocks
    if adaptive_slice_widths:
        # Only non-default width-controller settings enter the config (and
        # its hash) so historical fixed-width reports stay byte-identical.
        config["adaptive_slice_widths"] = adaptive_slice_widths
        config["bracket_mode"] = bracket_mode
        config["width_adaptation_rate"] = width_adaptation_rate
        config["width_target_expansions"] = width_target_expansions
        config["width_target_shrinks"] = width_target_shrinks
    if blocking_scheme == FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME:
        config["num_slice_steps_by_block"] = list(FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE)
    if blocking_scheme == FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME:
        config["block_kernel_modes"] = list(
            FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
        )
        config["fixed_work"] = dict(FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK)
    if blocking_scheme in COMPLEMENTARY_DE_BLOCKING_SCHEMES:
        complementary_de_block = COMPLEMENTARY_DE_BLOCK_BY_SCHEME[blocking_scheme]
        config["block_kernel_modes"] = list(
            FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
        )
        config["complementary_de_jump_block"] = {
            "parameters": list(complementary_de_block["parameters"]),
            "attempts": complementary_de_block["attempts"],
        }
        config["fixed_work"] = dict(
            COMPLEMENTARY_DE_FIXED_WORK_BY_SCHEME[blocking_scheme]
        )
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["sha256"] = hashlib.sha256(encoded).hexdigest()
    return config


def _likelihood_inputs_from_bundle(
    name: str,
    arrays: Mapping[str, np.ndarray],
    *,
    Data: Any,
    PowerSpectrum: Any,
    jnp: Any,
) -> tuple[Any, Any]:
    """Project native bundle arrays onto the fixed 4096 Hz analysis grid.

    The native time series is windowed and Fourier transformed before the
    high-frequency bins are discarded.  This is a frequency-domain projection,
    not a time-domain decimation, so it retains the 16 kHz product throughout
    the 20--2048 Hz likelihood band without introducing a 4 kHz anti-aliasing
    transition.  Returning a fixed-FD 4096 Hz ``Data`` object also keeps the
    historical time-marginalization FFT shape in older comparison revisions.
    """

    strain = Data(
        td=jnp.asarray(arrays[f"{name}_strain_td"]),
        delta_t=float(arrays[f"{name}_strain_delta_t"]),
        start_time=float(arrays[f"{name}_strain_start_time"]),
        name=name,
    )
    native_sample_rate_hz = float(strain.sampling_frequency)
    duration = float(strain.duration)
    target_sample_rate_hz = TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ
    tolerance = 1e-10 * target_sample_rate_hz
    if native_sample_rate_hz < target_sample_rate_hz - tolerance:
        raise SystemExit(
            f"Frozen {name} strain sample rate {native_sample_rate_hz:g} Hz is "
            f"below the fixed analysis rate {target_sample_rate_hz:g} Hz"
        )
    if native_sample_rate_hz > target_sample_rate_hz + tolerance:
        n_projected_frequencies = int(duration * target_sample_rate_hz / 2.0) + 1
        native_fd = strain.fft()
        projected_frequencies = strain.frequencies[:n_projected_frequencies]
        if not np.isclose(
            float(projected_frequencies[-1]),
            target_sample_rate_hz / 2.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise SystemExit(
                f"Frozen {name} strain cannot be projected onto the fixed "
                f"{target_sample_rate_hz:g} Hz analysis grid"
            )
        strain = Data.from_fd(
            native_fd[:n_projected_frequencies],
            projected_frequencies,
            start_time=strain.start_time,
            name=name,
        )

    psd_values = arrays[f"{name}_psd_values"]
    psd_frequencies = arrays[f"{name}_psd_frequencies"]
    n_psd_frequencies = int(duration * target_sample_rate_hz / 2.0) + 1
    if len(psd_frequencies) < n_psd_frequencies or not np.isclose(
        float(psd_frequencies[n_psd_frequencies - 1]),
        target_sample_rate_hz / 2.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise SystemExit(
            f"Frozen {name} PSD does not contain the fixed "
            f"{target_sample_rate_hz:g} Hz analysis grid"
        )
    psd = PowerSpectrum(
        values=jnp.asarray(psd_values[:n_psd_frequencies]),
        frequencies=jnp.asarray(psd_frequencies[:n_psd_frequencies]),
        name=f"{name}_psd",
    )
    return strain, psd


def _analysis_components(
    workload: str,
    jnp: Any,
    ifos: list[Any],
    *,
    blocking_scheme: str = PAPER_BLOCKING_SCHEME,
) -> dict[str, Any]:
    """Construct the waveform, priors, transforms, and periods for a workload."""

    from jimgw.core.prior import (
        CombinePrior,
        CosinePrior,
        PowerLawPrior,
        SinePrior,
        UniformPrior,
        UniformSpherePrior,
    )
    from jimgw.core.single_event.transforms import (
        DistanceToSNRWeightedDistanceTransform,
        MassRatioToSymmetricMassRatioTransform,
        SkyFrameToDetectorFrameSkyPositionTransform,
        SphereSpinToCartesianSpinTransform,
    )
    from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
    from jimgw.core.transforms import CosineTransform

    spec = _workload_spec(workload, blocking_scheme)
    if workload == PAPER_WORKLOAD:
        if __package__:
            from .paper_model import RippleIMRPhenomPv2NRTidalv2
        else:
            from paper_model import RippleIMRPhenomPv2NRTidalv2

        waveform = RippleIMRPhenomPv2NRTidalv2(f_ref=20.0)
        spin_priors = [
            UniformSpherePrior(parameter_names=["s1"], max_mag=0.05),
            UniformSpherePrior(parameter_names=["s2"], max_mag=0.05),
        ]
        spin_transforms = [
            SphereSpinToCartesianSpinTransform("s1"),
            SphereSpinToCartesianSpinTransform("s2"),
        ]
    else:
        waveform = RippleIMRPhenomD_NRTidalv2(f_ref=20.0)
        spin_priors = [
            UniformPrior(-0.05, 0.05, parameter_names=["s1_z"]),
            UniformPrior(-0.05, 0.05, parameter_names=["s2_z"]),
        ]
        spin_transforms = []

    prior = CombinePrior(
        [
            UniformPrior(1.18, 1.21, parameter_names=["M_c"]),
            UniformPrior(*spec["mass_ratio_range"], parameter_names=["q"]),
            *spin_priors,
            SinePrior(parameter_names=["iota"]),
            UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
            UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
            PowerLawPrior(
                *spec["distance_range_mpc"],
                2.0,
                parameter_names=["d_L"],
            ),
            UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
            CosinePrior(parameter_names=["dec"]),
            UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        ]
    )
    periodic = {
        "psi": (0.0, float(jnp.pi)),
        "azimuth": (0.0, 2 * float(jnp.pi)),
    }
    if workload == PAPER_WORKLOAD:
        periodic.update(
            {
                "s1_phi": (0.0, 2 * float(jnp.pi)),
                "s2_phi": (0.0, 2 * float(jnp.pi)),
            }
        )
    sample_transforms = []
    if blocking_scheme in {
        FAST_RIDGE_BLOCKING_SCHEME,
        FAST_RIDGE_INTRINSIC_BLOCKING_SCHEME,
        FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME,
        *PERIODIC_MH_BLOCKING_SCHEMES,
    }:
        # Build d_hat while the physical sky position and inclination are
        # still present. Reverse application restores sky, then iota, then
        # d_L, so the conditional inverse sees every quantity it needs.
        sample_transforms.extend(
            [
                DistanceToSNRWeightedDistanceTransform(
                    trigger_time=GPS,
                    ifos=ifos,
                ),
                CosineTransform((["iota"], ["cos_iota"])),
            ]
        )
    sample_transforms.append(
        SkyFrameToDetectorFrameSkyPositionTransform(
            trigger_time=GPS,
            ifos=ifos,
        )
    )

    return {
        "spec": spec,
        "waveform": waveform,
        "prior": prior,
        "sample_transforms": sample_transforms,
        "likelihood_transforms": [
            MassRatioToSymmetricMassRatioTransform,
            *spin_transforms,
        ],
        "periodic": periodic,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        return float(array.item())
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    scalar = _safe_float(value)
    return int(scalar) if scalar is not None else None


def _json_safe(value: Any) -> Any:
    """Convert diagnostic arrays/scalars into JSON-safe builtins."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _pytree_sha256(jax: Any, value: Any) -> str:
    """Hash the logical host value of a pytree, independent of sharding."""

    leaves, tree = jax.tree_util.tree_flatten(jax.device_get(value))
    digest = hashlib.sha256(str(tree).encode())
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(array.shape).encode())
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _derive_paper_convention(
    sample_call_seconds: float,
    jit_estimate: float | None,
    sample_phases: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Post-JIT sampling time under the arXiv:2607.28265 convention.

    The paper excludes both one-off compilation costs from quoted sampling
    times and reports them separately (Table 3: likelihood + sampler-kernel
    JIT). ``likelihood_jit`` is measured directly (AOT compile) when the
    sampler revision provides phase timings; otherwise it is None and not
    subtracted, which reduces to the legacy sample_call - jit_estimate.
    """

    likelihood_jit: float | None = None
    sampler_jit: float | None = None
    sampler_jit_source: str | None = None
    if sample_phases is not None:
        raw = sample_phases.get("likelihood_jit")
        if raw is not None:
            likelihood_jit = float(raw)
        raw = sample_phases.get("sampler_kernel_jit")
        if raw is not None:
            sampler_jit = float(raw)
            sampler_jit_source = "direct_aot_phase"
    if sampler_jit is None and jit_estimate is not None:
        sampler_jit = jit_estimate
        sampler_jit_source = "first_step_minus_steady_median"
    post_jit = sample_call_seconds
    if sampler_jit is not None:
        post_jit -= sampler_jit
    if likelihood_jit is not None:
        post_jit -= likelihood_jit
    return {
        "likelihood_jit_seconds": likelihood_jit,
        "sampler_jit_seconds": sampler_jit,
        "sampler_jit_source": sampler_jit_source,
        "post_jit_sampling_seconds": post_jit,
        "note": (
            "Sampling time excluding both one-off JIT costs (likelihood + "
            "sampler kernel), matching arXiv:2607.28265 Table 3. "
            "sampler_jit_seconds prefers the directly measured AOT sampler "
            "compile and otherwise falls back to first-step-minus-steady-median; "
            "likelihood_jit_seconds is the AOT-measured compile of the "
            "initial batched likelihood evaluation, or None when the "
            "sampler revision does not report it."
        ),
    }


class _TelemetryCapture:
    """Capture one-second GPU telemetry during the steady-state trace window."""

    def __init__(self, output: Path | None) -> None:
        self.output = output.expanduser().resolve() if output is not None else None
        self._stream: Any = None
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if self.output is None:
            return
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise RuntimeError("--telemetry-output requires nvidia-smi")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output.open("w", encoding="utf-8")
        command = [executable, "dmon", "-s", "pucvmet", "-d", "1", "-o", "DT"]
        self._stream.write("# command: " + " ".join(command) + "\n")
        self._stream.flush()
        self._process = subprocess.Popen(
            command,
            stdout=self._stream,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        process = self._process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
        self._process = None
        self._stream = None


class _OuterStepObserver:
    """Observe the JIT-compiled outer kernel without changing sampler source."""

    def __init__(
        self,
        *,
        jax: Any,
        jnp: Any,
        profile_dir: Path | None,
        profile_warmup_steps: int,
        profile_steps: int,
        telemetry_output: Path | None,
        max_outer_steps: int | None,
    ) -> None:
        self.jax = jax
        self.jnp = jnp
        self.profile_dir = (
            profile_dir.expanduser().resolve() if profile_dir is not None else None
        )
        self.profile_warmup_steps = profile_warmup_steps
        self.profile_steps = profile_steps
        self.max_outer_steps = max_outer_steps
        self.telemetry = _TelemetryCapture(telemetry_output)
        self.host_perf_counter_start: list[float] = []
        self.host_perf_counter_end: list[float] = []
        self.compiled_kernel_count = 0
        self.trace_started = False
        self.trace_stopped = False
        self.profiled_step_indices: list[int] = []

    def _start_trace(self) -> None:
        if self.profile_dir is None or self.trace_started:
            return
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.jax.profiler.start_trace(str(self.profile_dir))
        self.telemetry.start()
        self.trace_started = True

    def _stop_trace(self) -> None:
        if not self.trace_started or self.trace_stopped:
            return
        try:
            # Profiler serialization can take several seconds after the last
            # device step.  Stop dmon first so its samples cover only the
            # requested steady-state window, not that host-side drain.
            self.telemetry.stop()
        finally:
            try:
                self.jax.profiler.stop_trace()
            finally:
                self.trace_stopped = True

    def _force_termination(self, result: Any) -> Any:
        state, dead_info = result
        integrator = state.integrator
        terminated_integrator = integrator._replace(
            logZ_live=integrator.logZ - self.jnp.asarray(1000.0)
        )
        return state._replace(integrator=terminated_integrator), dead_info

    def wrap(self, compiled: Callable) -> Callable:
        self.compiled_kernel_count += 1

        def observed(*args: Any, **kwargs: Any) -> Any:
            step_index = len(self.host_perf_counter_start)
            if self.profile_dir is not None and step_index == self.profile_warmup_steps:
                self._start_trace()

            started = time.perf_counter()
            with self.jax.profiler.TraceAnnotation(
                "nested_sampler_outer_step", step_num=step_index
            ):
                result = compiled(*args, **kwargs)
                self.jax.block_until_ready(result)
            ended = time.perf_counter()
            self.host_perf_counter_start.append(started)
            self.host_perf_counter_end.append(ended)

            if self.trace_started and not self.trace_stopped:
                self.profiled_step_indices.append(step_index)
                if len(self.profiled_step_indices) >= self.profile_steps:
                    self._stop_trace()

            if (
                self.max_outer_steps is not None
                and step_index + 1 >= self.max_outer_steps
            ):
                result = self._force_termination(result)
            return result

        return observed

    def close(self) -> None:
        self._stop_trace()

    def report(self) -> dict[str, Any]:
        durations = [
            ended - started
            for started, ended in zip(
                self.host_perf_counter_start,
                self.host_perf_counter_end,
                strict=True,
            )
        ]
        steady = durations[1:]
        return {
            "outer_steps": len(durations),
            "host_perf_counter_start": self.host_perf_counter_start,
            "host_perf_counter_end": self.host_perf_counter_end,
            "duration_seconds": durations,
            "first_step_seconds_including_jit": durations[0] if durations else None,
            "steady_state_seconds": {
                "count": len(steady),
                "minimum": min(steady) if steady else None,
                "median": float(np.median(steady)) if steady else None,
                "mean": float(np.mean(steady)) if steady else None,
                "maximum": max(steady) if steady else None,
                "standard_deviation": float(np.std(steady)) if steady else None,
            },
            "profile": {
                "directory": str(self.profile_dir)
                if self.profile_dir is not None
                else None,
                "warmup_steps": self.profile_warmup_steps,
                "requested_steps": self.profile_steps,
                "captured_step_indices": self.profiled_step_indices,
                "trace_started": self.trace_started,
                "trace_stopped": self.trace_stopped,
                "telemetry_output": (
                    str(self.telemetry.output)
                    if self.telemetry.output is not None
                    else None
                ),
            },
            "compiled_kernel_count": self.compiled_kernel_count,
            "max_outer_steps": self.max_outer_steps,
        }


@contextlib.contextmanager
def _observe_outer_step_jit(jax: Any, observer: _OuterStepObserver):
    """Wrap only BlackJAX's function named ``kernel`` when it is JIT-compiled."""

    original_jit = jax.jit

    def observing_jit(fun: Callable | None = None, *jit_args: Any, **jit_kwargs: Any):
        if fun is None:
            return lambda actual: observing_jit(actual, *jit_args, **jit_kwargs)
        compiled = original_jit(fun, *jit_args, **jit_kwargs)
        if getattr(fun, "__name__", None) == "kernel":
            return observer.wrap(compiled)
        return compiled

    jax.jit = observing_jit
    try:
        yield
    finally:
        jax.jit = original_jit
        observer.close()


def _install_per_slice_swig_diagnostics() -> bool:
    """Enable per-slice counters when the selected revision supports them."""

    from jimgw.samplers.blackjax import swig

    original = swig._build_swig_constrained_step
    parameters = inspect.signature(original).parameters.values()
    supports_flag = "per_slice_info" in inspect.signature(original).parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if not supports_flag:
        return False

    def with_per_slice_info(**kwargs):
        kwargs["per_slice_info"] = True
        return original(**kwargs)

    swig._build_swig_constrained_step = with_per_slice_info
    return True


def _normalise_per_slice_array(value: Any, n_slices: int) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    if array.ndim != 2 or array.shape[1] != n_slices:
        raise RuntimeError(
            "expected per-slice update_info with shape (chains, "
            f"{n_slices}); got {array.shape}"
        )
    return array


def _write_slice_data(path: Path, jim: Any, n_devices: int) -> dict[str, Any]:
    update_info = jim.sampler._final_state.update_info
    rebuild_by_block = jim.sampler._rebuild_required_by_block
    block_kernel_modes = tuple(
        getattr(
            jim.sampler,
            "_block_kernel_modes",
            ("slice",) * len(rebuild_by_block),
        )
    )
    if len(block_kernel_modes) != len(rebuild_by_block):
        raise RuntimeError(
            "per-slice block kernel modes do not align with cache blocks"
        )
    unsupported_modes = sorted(
        set(block_kernel_modes) - {"slice", "periodic-uniform-independence"}
    )
    if unsupported_modes:
        raise RuntimeError(
            "unsupported block kernel modes in per-slice diagnostics: "
            f"{unsupported_modes}"
        )
    slice_block_indices: list[int] = []
    slice_requires_rebuild: list[bool] = []
    slice_slot_in_block: list[int] = []
    slice_gibbs_sweep: list[int] = []
    slice_block_parameter_names: list[str] = []
    block_parameter_indices = list(rebuild_by_block)
    max_block_dimensions = max(map(len, block_parameter_indices))
    padded_parameter_indices: list[list[int]] = []
    for gibbs_sweep in range(NUM_GIBBS_SWEEPS):
        for block_index, ((indices, requires_rebuild), kernel_mode) in enumerate(
            zip(
                rebuild_by_block.items(),
                block_kernel_modes,
                strict=True,
            )
        ):
            if kernel_mode != "slice":
                continue
            parameter_names = ",".join(
                jim.sampling_parameter_names[index] for index in indices
            )
            n_block_slots = NUM_INNER_STEPS_PER_DIM * len(indices)
            for slot_in_block in range(n_block_slots):
                slice_block_indices.append(block_index)
                slice_requires_rebuild.append(requires_rebuild)
                slice_slot_in_block.append(slot_in_block)
                slice_gibbs_sweep.append(gibbs_sweep)
                slice_block_parameter_names.append(parameter_names)
                padded_parameter_indices.append(
                    [*indices, *([-1] * (max_block_dimensions - len(indices)))]
                )

    n_slices = len(slice_block_indices)
    num_expansions = _normalise_per_slice_array(update_info.num_expansions, n_slices)
    num_shrink = _normalise_per_slice_array(update_info.num_shrink, n_slices)
    if num_expansions.shape != num_shrink.shape:
        raise RuntimeError("per-slice expansion and shrink arrays differ in shape")
    if num_expansions.shape[0] % N_DELETE:
        raise RuntimeError("per-slice history is not divisible into outer iterations")

    n_iterations = num_expansions.shape[0] // N_DELETE
    arrays = {
        "num_expansions": num_expansions,
        "num_shrink": num_shrink,
        "outer_iteration": np.repeat(np.arange(n_iterations), N_DELETE),
        "chain_index": np.tile(np.arange(N_DELETE), n_iterations),
        "slice_block_index": np.asarray(slice_block_indices),
        "slice_slot_in_block": np.asarray(slice_slot_in_block),
        "slice_gibbs_sweep": np.asarray(slice_gibbs_sweep),
        "slice_block_parameter_indices": np.asarray(padded_parameter_indices),
        "slice_block_parameter_names": np.asarray(slice_block_parameter_names),
        "slice_requires_rebuild": np.asarray(slice_requires_rebuild),
        "n_devices": np.asarray(n_devices),
        "lanes_per_device": np.asarray(N_DELETE // n_devices),
    }
    _atomic_save_npz(path, arrays)
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "shape": list(num_expansions.shape),
        "n_iterations": n_iterations,
        "n_delete": N_DELETE,
        "n_slices": n_slices,
        "totals": {
            "num_expansions": int(num_expansions.sum()),
            "num_shrink": int(num_shrink.sum()),
        },
    }


def _write_posterior_samples(
    path: Path,
    samples: dict[str, np.ndarray],
    *,
    weighting: str = "equal",
) -> dict[str, Any]:
    """Atomically persist an aligned mapping of samples in prior space."""

    if not samples:
        raise RuntimeError("posterior sample mapping is empty")

    arrays = {name: np.asarray(values) for name, values in samples.items()}
    invalid_shapes = {
        name: list(values.shape) for name, values in arrays.items() if values.ndim != 1
    }
    if invalid_shapes:
        raise RuntimeError(
            "posterior sample arrays must be one-dimensional; got "
            + json.dumps(invalid_shapes, sort_keys=True)
        )

    sample_counts = {name: int(values.shape[0]) for name, values in arrays.items()}
    if len(set(sample_counts.values())) != 1:
        raise RuntimeError(
            "posterior sample arrays have inconsistent lengths: "
            + json.dumps(sample_counts, sort_keys=True)
        )

    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".npz":
        raise SystemExit(f"sample output must use the .npz extension; got {resolved}")
    _atomic_save_npz(resolved, arrays)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "format": "npz",
        "space": "prior",
        "weighting": weighting,
        "count": next(iter(sample_counts.values())),
        "fields": list(arrays),
        "dtypes": {name: str(values.dtype) for name, values in arrays.items()},
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    total_started = time.perf_counter()

    import jax
    import jax.numpy as jnp
    import jaxlib

    jax.config.update("jax_enable_x64", True)
    compilation_cache_dir: Path | None = None
    if args.jax_compilation_cache_dir is not None:
        resolved_cache_dir = Path(args.jax_compilation_cache_dir).expanduser().resolve()
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        compilation_cache_dir = resolved_cache_dir
        jax.config.update("jax_compilation_cache_dir", str(resolved_cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    import blackjax
    import ripplegw

    import jimgw
    from jimgw.core.jim import Jim
    from jimgw.core.single_event.data import Data, PowerSpectrum
    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
    from jimgw.core.single_event.likelihood import TransientLikelihoodFD
    from jimgw.samplers.config import BlackJAXSwiGConfig, DEJumpBlockConfig

    ablation: dict[str, str] | None = None
    if args.ablation_variant is not None:
        from benchmarks.device_parallel_nss.sampler_ablation import (
            install_sampler_ablation,
        )

        ablation = install_sampler_ablation(args.ablation_variant)

    if (
        args.slice_data_output is not None or args.retain_per_slice_info
    ) and not _install_per_slice_swig_diagnostics():
        raise SystemExit(
            "the selected implementation does not expose per-slice SwiG "
            "diagnostics; rerun without --slice-data-output and "
            "--retain-per-slice-info"
        )

    devices = _device_metadata(jax, args.n_devices)
    if not args.simulate_cpu and devices["backend"] != "gpu":
        raise SystemExit(
            "This is a GPU benchmark, but JAX selected backend "
            f"{devices['backend']!r}. Use --simulate-cpu only for smoke tests."
        )

    data_started = time.perf_counter()
    data_path = args.data_file.expanduser().resolve()
    data_sha256 = _sha256(data_path) if data_path.is_file() else None
    manifest, arrays = _read_bundle(data_path, args.workload)
    ifos = [get_H1(), get_L1(), get_V1()]
    for ifo in ifos:
        name = ifo.name
        strain, psd = _likelihood_inputs_from_bundle(
            name,
            arrays,
            Data=Data,
            PowerSpectrum=PowerSpectrum,
            jnp=jnp,
        )
        ifo.set_data(strain)
        ifo.set_psd(psd)
    data_load_seconds = time.perf_counter() - data_started

    setup_started = time.perf_counter()
    components = _analysis_components(
        args.workload,
        jnp,
        ifos,
        blocking_scheme=args.blocking_scheme,
    )
    workload_spec = components["spec"]
    likelihood = TransientLikelihoodFD(
        ifos,
        waveform=components["waveform"],
        trigger_time=GPS,
        f_min=F_MIN,
        f_max=F_MAX,
        phase_marginalization=True,
        time_marginalization={"tc_range": (-0.03, 0.03)},
    )
    complementary_de_config: dict[str, Any] = {}
    if args.blocking_scheme in COMPLEMENTARY_DE_BLOCKING_SCHEMES:
        complementary_de_block = COMPLEMENTARY_DE_BLOCK_BY_SCHEME[args.blocking_scheme]
        complementary_de_config["complementary_de_jump_block"] = DEJumpBlockConfig(
            parameters=list(complementary_de_block["parameters"]),
            attempts=complementary_de_block["attempts"],
        )
    sampler_config = BlackJAXSwiGConfig(
        blocks=[list(block) for block in workload_spec["blocks"]],
        n_live=N_LIVE,
        n_delete_frac=N_DELETE_FRAC,
        num_inner_steps_per_dim=NUM_INNER_STEPS_PER_DIM,
        num_gibbs_sweeps=NUM_GIBBS_SWEEPS,
        termination_dlogz=TERMINATION_DLOGZ,
        n_devices=args.n_devices,
        scheduler=(
            "pre-fsm-lockstep"
            if ablation is not None and ablation["scheduler"] == "lockstep"
            else "fsm"
        ),
        direction_mode=args.direction_mode,
        num_slice_steps_by_block=(
            list(FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE)
            if args.blocking_scheme == FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME
            else None
        ),
        block_kernel_modes=(
            list(FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES)
            if args.blocking_scheme in PERIODIC_MH_BLOCKING_SCHEMES
            else None
        ),
        de_fraction=args.de_fraction,
        num_de_jumps=args.num_de_jumps,
        de_jump_blocks=args.de_jump_block,
        adaptive_slice_widths=args.adaptive_slice_widths,
        bracket_mode=args.bracket_mode,
        width_adaptation_rate=args.width_adaptation_rate,
        width_target_expansions=args.width_target_expansions,
        width_target_shrinks=args.width_target_shrinks,
        **complementary_de_config,
    )
    sampler_seed = args.seed if args.sampler_seed is None else args.sampler_seed
    jim = Jim(
        likelihood,
        components["prior"],
        sample_transforms=components["sample_transforms"],
        likelihood_transforms=components["likelihood_transforms"],
        periodic=components["periodic"],
        sampler_config=sampler_config,
        seed=sampler_seed,
        verbose=args.verbose,
    )
    problem_setup_seconds = time.perf_counter() - setup_started

    positions_started = time.perf_counter()
    initial_positions = jim.sample_initial_positions(
        N_LIVE, rng_key=jax.random.key(args.seed)
    )
    initial_positions_host = np.asarray(jax.device_get(initial_positions))
    initial_positions_sha256 = hashlib.sha256(
        initial_positions_host.tobytes(order="C")
    ).hexdigest()
    initial_positions_seconds = time.perf_counter() - positions_started

    sample_started = time.perf_counter()
    observer: _OuterStepObserver | None = None
    if args.timing_only:
        jim.sample(initial_positions)
    else:
        observer = _OuterStepObserver(
            jax=jax,
            jnp=jnp,
            profile_dir=args.profile_dir,
            profile_warmup_steps=args.profile_warmup_steps,
            profile_steps=args.profile_steps,
            telemetry_output=args.telemetry_output,
            max_outer_steps=args.max_outer_steps,
        )
        with _observe_outer_step_jit(jax, observer):
            jim.sample(initial_positions)
    sample_call_seconds = time.perf_counter() - sample_started
    if observer is not None and observer.compiled_kernel_count != 1:
        raise RuntimeError(
            "expected to observe exactly one JIT-compiled outer kernel, got "
            f"{observer.compiled_kernel_count}"
        )

    extraction_started = time.perf_counter()
    diagnostics = jim.get_diagnostics()
    samples = {} if args.timing_only else jim.get_samples()
    posterior_count = int(next(iter(samples.values())).shape[0]) if samples else None
    nested_samples = getattr(jim.sampler, "_nested_samples", None)
    try:
        posterior_ess = (
            _safe_float(nested_samples.neff())
            if nested_samples is not None and not args.timing_only
            else None
        )
    except (AttributeError, TypeError, ValueError):
        posterior_ess = None
    slice_data = (
        _write_slice_data(args.slice_data_output, jim, args.n_devices)
        if args.slice_data_output is not None
        else None
    )
    posterior_artifact = (
        _write_posterior_samples(args.samples_output, samples)
        if args.samples_output is not None
        else None
    )
    nested_artifact = (
        _write_posterior_samples(
            args.nested_output,
            jim.get_weighted_samples(),
            weighting="normalized nested-sampling log weights",
        )
        if args.nested_output is not None
        else None
    )
    result_extraction_seconds = time.perf_counter() - extraction_started
    nested_state_sha256 = _pytree_sha256(jax, jim.sampler._final_state)

    step_timing = observer.report() if observer is not None else None
    first_step = (
        step_timing["first_step_seconds_including_jit"]
        if step_timing is not None
        else None
    )
    steady_median = (
        step_timing["steady_state_seconds"]["median"]
        if step_timing is not None
        else None
    )
    jit_estimate = (
        first_step - steady_median
        if first_step is not None and steady_median is not None
        else None
    )
    sample_phases = diagnostics.get("sample_phase_seconds")
    if args.timing_only:
        required_phases = ("likelihood_jit", "sampler_kernel_jit", "ns_loop")
        missing_phases = [
            name
            for name in required_phases
            if not isinstance(sample_phases, Mapping) or sample_phases.get(name) is None
        ]
        if missing_phases:
            raise RuntimeError(
                "timing-only run did not directly measure phases: "
                + ", ".join(missing_phases)
            )

    inferred_root = Path(jimgw.__file__).resolve().parents[2]
    implementation_root = args.implementation_root or inferred_root
    timing = {
        "total": time.perf_counter() - total_started,
        "data_load_and_hash": data_load_seconds,
        "problem_setup": problem_setup_seconds,
        "initial_positions": initial_positions_seconds,
        "sample_call": sample_call_seconds,
        "result_extraction": result_extraction_seconds,
        "jit_compile_estimate": jit_estimate,
        "jit_note": (
            "timing-only runs AOT-compile both kernels and measure ns_loop "
            "directly without an observer. Other runs estimate sampler JIT as "
            "first observed step minus the later synchronized-step median."
        ),
        "sample_phases": sample_phases,
        "paper_convention": _derive_paper_convention(
            sample_call_seconds, jit_estimate, sample_phases
        ),
        "sampler_reported": _safe_float(diagnostics.get("sampling_time")),
        "outer_step": step_timing,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "timing_only": bool(args.timing_only),
        "ablation": ablation,
        "implementation": {
            "label": args.implementation_label,
            "module_file": str(Path(jimgw.__file__).resolve()),
            "package_version": getattr(jimgw, "__version__", None),
            **_git_metadata(implementation_root, args.implementation_revision),
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "operating_system": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "blackjax": getattr(blackjax, "__version__", None),
            "ripplegw": getattr(ripplegw, "__version__", None),
            "nccl_nvls_enable": os.environ.get("NCCL_NVLS_ENABLE"),
            "jax_compilation_cache_dir": (
                str(compilation_cache_dir)
                if compilation_cache_dir is not None
                else None
            ),
        },
        "devices": devices,
        "data": {
            "path": str(data_path),
            "sha256": data_sha256,
            "bytes": data_path.stat().st_size,
            "manifest": manifest,
            "sample_counts": {
                name: int(arrays[f"{name}_strain_td"].size) for name in IFO_NAMES
            },
        },
        "config": {
            **_config_report(
                args.seed,
                args.n_devices,
                args.workload,
                args.blocking_scheme,
                direction_mode=args.direction_mode,
                de_fraction=args.de_fraction,
                num_de_jumps=args.num_de_jumps,
                de_jump_blocks=args.de_jump_block,
                adaptive_slice_widths=args.adaptive_slice_widths,
                bracket_mode=args.bracket_mode,
                width_adaptation_rate=args.width_adaptation_rate,
                width_target_expansions=args.width_target_expansions,
                width_target_shrinks=args.width_target_shrinks,
                sampler_seed=args.sampler_seed,
            ),
            "initial_positions_sha256": initial_positions_sha256,
        },
        "timing_seconds": timing,
        "results": {
            "n_iterations": _safe_int(diagnostics.get("n_iterations")),
            "n_likelihood_evaluations": _safe_int(
                diagnostics.get("n_likelihood_evaluations")
            ),
            "n_likelihood_evaluations_physical": _safe_int(
                diagnostics.get("n_likelihood_evaluations_physical")
            ),
            "n_slice_updates": _safe_int(diagnostics.get("n_slice_updates")),
            "n_periodic_uniform_independence_attempts": _safe_int(
                diagnostics.get("n_periodic_uniform_independence_attempts")
            ),
            "n_periodic_uniform_independence_acceptances": _safe_int(
                diagnostics.get("n_periodic_uniform_independence_acceptances")
            ),
            "periodic_uniform_independence_acceptance_rate": _safe_float(
                diagnostics.get("periodic_uniform_independence_acceptance_rate")
            ),
            "n_likelihood_evaluations_periodic_uniform_independence": _safe_int(
                diagnostics.get(
                    "n_likelihood_evaluations_periodic_uniform_independence"
                )
            ),
            "n_likelihood_evaluations_periodic_uniform_independence_waveform_rebuild": _safe_int(
                diagnostics.get(
                    "n_likelihood_evaluations_periodic_uniform_independence_waveform_rebuild"
                )
            ),
            "n_likelihood_evaluations_periodic_uniform_independence_cache_hit": _safe_int(
                diagnostics.get(
                    "n_likelihood_evaluations_periodic_uniform_independence_cache_hit"
                )
            ),
            "periodic_uniform_independence_blocks": _json_safe(
                diagnostics.get("periodic_uniform_independence_blocks")
            ),
            "periodic_uniform_independence_attempts_by_block_history": _json_safe(
                diagnostics.get(
                    "periodic_uniform_independence_attempts_by_block_history"
                )
            ),
            "periodic_uniform_independence_acceptances_by_block_history": (
                _json_safe(
                    diagnostics.get(
                        "periodic_uniform_independence_acceptances_by_block_history"
                    )
                )
            ),
            **(
                {
                    "n_likelihood_evaluations_complementary_de": _safe_int(
                        diagnostics.get("n_likelihood_evaluations_complementary_de")
                    ),
                    "n_likelihood_evaluations_complementary_de_waveform_rebuild": _safe_int(
                        diagnostics.get(
                            "n_likelihood_evaluations_complementary_de_waveform_rebuild"
                        )
                    ),
                    "n_likelihood_evaluations_complementary_de_cache_hit": _safe_int(
                        diagnostics.get(
                            "n_likelihood_evaluations_complementary_de_cache_hit"
                        )
                    ),
                    "n_complementary_de_attempts": _safe_int(
                        diagnostics.get("n_complementary_de_attempts")
                    ),
                    "n_complementary_de_acceptances": _safe_int(
                        diagnostics.get("n_complementary_de_acceptances")
                    ),
                    "n_complementary_de_donor_policy_violations": _safe_int(
                        diagnostics.get("n_complementary_de_donor_policy_violations")
                    ),
                    "complementary_de_acceptance_rate": _safe_float(
                        diagnostics.get("complementary_de_acceptance_rate")
                    ),
                    "complementary_de_blocks": _json_safe(
                        diagnostics.get("complementary_de_blocks")
                    ),
                    "complementary_de_attempts_history": _json_safe(
                        diagnostics.get("complementary_de_attempts_history")
                    ),
                    "complementary_de_acceptances_history": _json_safe(
                        diagnostics.get("complementary_de_acceptances_history")
                    ),
                    "complementary_de_attempts_by_block_history": _json_safe(
                        diagnostics.get("complementary_de_attempts_by_block_history")
                    ),
                    "complementary_de_acceptances_by_block_history": _json_safe(
                        diagnostics.get("complementary_de_acceptances_by_block_history")
                    ),
                    "complementary_de_donor_policy_violations_history": _json_safe(
                        diagnostics.get(
                            "complementary_de_donor_policy_violations_history"
                        )
                    ),
                    "complementary_de_acceptances_by_attempt_history": _json_safe(
                        diagnostics.get(
                            "complementary_de_acceptances_by_attempt_history"
                        )
                    ),
                    "complementary_de_donor_indices_by_attempt_history": _json_safe(
                        diagnostics.get(
                            "complementary_de_donor_indices_by_attempt_history"
                        )
                    ),
                    "complementary_de_donor_policy_violations_by_attempt_history": _json_safe(
                        diagnostics.get(
                            "complementary_de_donor_policy_violations_by_attempt_history"
                        )
                    ),
                    "complementary_de_complement_size_history": _json_safe(
                        diagnostics.get("complementary_de_complement_size_history")
                    ),
                    "complementary_de_parent_index_history": _json_safe(
                        diagnostics.get("complementary_de_parent_index_history")
                    ),
                    "complementary_de_position_before_by_attempt_history": _json_safe(
                        diagnostics.get(
                            "complementary_de_position_before_by_attempt_history"
                        )
                    ),
                    "complementary_de_proposal_position_by_attempt_history": _json_safe(
                        diagnostics.get(
                            "complementary_de_proposal_position_by_attempt_history"
                        )
                    ),
                    "complementary_de_sampling_parameter_names": list(
                        jim.sampling_parameter_names
                    ),
                }
                if args.blocking_scheme in COMPLEMENTARY_DE_BLOCKING_SCHEMES
                else {}
            ),
            "n_likelihood_evaluations_de_jumps": _safe_int(
                diagnostics.get("n_likelihood_evaluations_de_jumps")
            ),
            "n_de_jump_attempts": _safe_int(diagnostics.get("n_de_jump_attempts")),
            "n_de_jump_acceptances": _safe_int(
                diagnostics.get("n_de_jump_acceptances")
            ),
            "de_jump_acceptance_rate": _safe_float(
                diagnostics.get("de_jump_acceptance_rate")
            ),
            "targeted_de_jump_blocks": diagnostics.get("targeted_de_jump_blocks"),
            "n_likelihood_evaluations_de_jumps_waveform_rebuild": _safe_int(
                diagnostics.get("n_likelihood_evaluations_de_jumps_waveform_rebuild")
            ),
            "n_likelihood_evaluations_de_jumps_cache_hit": _safe_int(
                diagnostics.get("n_likelihood_evaluations_de_jumps_cache_hit")
            ),
            "log_Z": _safe_float(diagnostics.get("log_Z")),
            "log_Z_error": _safe_float(diagnostics.get("log_Z_error")),
            "posterior_samples": posterior_count,
            "posterior_artifact": posterior_artifact,
            "nested_artifact": nested_artifact,
            "posterior_effective_sample_size": posterior_ess,
            "posterior_ess_source": (
                "anesthetic.NestedSamples.neff" if posterior_ess is not None else None
            ),
            "nested_state_sha256": nested_state_sha256,
            "per_slice_update_info": slice_data,
            "early_stopped_for_cache_probe": args.max_outer_steps is not None,
        },
        "limitations": list(workload_spec["limitations"]),
        "simulated_cpu": bool(args.simulate_cpu),
    }


def _atomic_write_report(path: Path, report_text: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            dir=resolved.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(report_text)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, resolved)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _emit_report(report: dict[str, Any], output: Path | None) -> None:
    report_text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if output is not None:
        _atomic_write_report(output, report_text)
    print(report_text, flush=True)


def main(argv: list[str] | None = None) -> None:
    global NUM_GIBBS_SWEEPS

    args = _parse_args(argv)
    NUM_GIBBS_SWEEPS = args.num_gibbs_sweeps
    _select_implementation(args.implementation_root)
    if args.prepare_data:
        started = time.perf_counter()
        data = _prepare_data(args.data_file, args.workload)
        _emit_report(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": BENCHMARK_NAME,
                "operation": "prepare-data",
                "data": data,
                "timing_seconds": {"total": time.perf_counter() - started},
            },
            args.output,
        )
        return
    if args.simulate_cpu:
        _configure_cpu_simulation(args.n_devices)
    _emit_report(_run(args), args.output)


if __name__ == "__main__":
    main()
