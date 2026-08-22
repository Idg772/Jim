"""Run a fail-closed likelihood A/B for the paper-15d GW170817 model.

Both cells use the same configured device count, FSM nested sampler, blocks,
slice-direction mode, Gibbs-sweep count, seed, frozen data, and initial-position
RNG.  The historical default remains ``M=1``.
The only intended difference is the likelihood evaluation:

* ``full`` uses :class:`TransientLikelihoodFD`; and
* ``heterodyne5000`` uses the benchmark-local direct-sum time-marginalized
  relative-binning likelihood with 5000 requested bins and a frozen reference.

The paper used different device counts for its full and heterodyned runs. This
controlled pair instead holds the requested device count fixed across arms and
is not a claim to reproduce that allocation. Each cell runs in a fresh process
so process-local likelihood and sampler patches cannot leak.

The default five-step prefix is a fail-fast execution gate.  Select
``--mode full`` explicitly for scientific nested-sampling runs.  The separate
``--preflight-only`` mode performs no sampling: it constructs both likelihoods
on the frozen data and checks the reference value, approximation error, and
direct/cache/JIT parity before any remote compute is purchased.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.device_parallel_nss.likelihood_pair_preflight_contract import (
    preflight_probe_families as _preflight_probe_families,
)
from benchmarks.injection_campaign.common import (
    FOLDED_TARGET_SEMANTICS,
    NETSKY_BLOCKS,
    NETSKY_BRIDGE_BLOCKS,
    NETSKY_SCHEME,
    POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS,
    UNFOLDED_POSTERIOR_WEIGHTING,
)

SCHEMA_VERSION = 1
PREFIX_STEPS = 5
WORKLOAD = "paper-15d"
BLOCKING_SCHEME = "paper"
DIRECTION_MODE = "covariance"
N_DEVICES = 4
N_LIVE = 512
N_DELETE = 64
N_BINS_REQUESTED = 5000
NUM_INNER_STEPS_PER_DIM = 1
NUM_GIBBS_SWEEPS = 1
NETSKY_NUM_GIBBS_SWEEPS = 2
NETSKY_FIXED_WORK = {
    "total_updates": 32,
    "total_slice_updates": 32,
    "waveform_rebuild_slice_updates": 20,
    "cache_hit_slice_updates": 12,
    "primary_slice_updates": 30,
    "bridge_slice_updates": 2,
}
TERMINATION_DLOGZ = 0.0485873516
WAVEFORM = "IMRPhenomPv2_NRTidalv2"
WAVEFORM_F_REF_HZ = 20.0
CARRIER_TIME_ANCHOR = "imrphenomd"
TC_RANGE_SECONDS = (-0.03, 0.03)
DETECTORS = ("H1", "L1", "V1")
DURATION_SECONDS = 128.0
F_MIN_HZ = 20.0
NOMINAL_F_MAX_HZ = 2048.0
LIKELIHOOD_F_MAX_HZ = 2047.9921875
TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ = 4096
TIME_UPSAMPLE_FACTOR = 8
TIME_MARGINALIZATION_DESCRIPTION = "matched U=8 dense sub-grid direct sum"
FSM_VARIANT = "replicated-cached-fsm-cov"
MAX_REFERENCE_ABS_LOGL_DELTA = 0.10
MAX_STRESS_ABS_LOGL_DELTA = 0.25
PARITY_ATOL = 1.0e-6
REFERENCE_SOURCE_ATOL = 5.0e-3
CANONICAL_DATA_SHA256 = (
    "a502f1d077618b94c8bf8d9820d10f850c8764e18a0f0e7e18663e14ac75104f"
)
MIN_BETA_LOGL_DROP = 1.0
MAX_RELATIVE_BINNING_BETA = 0.01
PROBE_SYSTEMATIC_POINTS_PER_FILE = 12
MIN_INDEPENDENT_PROBE_FILES = 3
MAX_FIDUCIAL_LOGL_GAP = 0.10
MAX_U8_U16_ABS_LOGL_DELTA = 0.005
MAX_Q_TIME_U8_U16_LOGL_DROP = 200.0
Q_TIME_GRID_VALUES = (0.125, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_REFERENCE_FILE = Path(__file__).with_name("gw170817_heterodyne_reference.json")
DIRECTION_MODE_CHOICES = ("covariance", "covariance-basis-8d")
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
PAPER_BLOCKS = (
    ("M_c", "q", "lambda_1", "lambda_2"),
    ("s1_mag", "s1_theta", "s1_phi"),
    ("s2_mag", "s2_theta", "s2_phi"),
    ("iota",),
    ("zenith", "azimuth"),
    ("psi",),
    ("d_L",),
)
FAST_RIDGE_BLOCKS = (
    ("M_c", "q", "lambda_1", "lambda_2"),
    ("s1_mag", "s1_theta", "s1_phi"),
    ("s2_mag", "s2_theta", "s2_phi"),
    ("zenith", "azimuth"),
    ("psi",),
    ("cos_iota", "d_hat"),
)
FAST_RIDGE_INTRINSIC_BLOCKS = (
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
FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCK = {
    "parameters": FAST_RIDGE_INTRINSIC_BLOCKS[0],
    "attempts": 4,
}
FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_POLICY = {
    "attempts_per_replacement": 4,
    "gamma": 1.0,
    "placement": "after-target-block",
    "donor_pair_ordered": True,
    "donors_redrawn_each_attempt": True,
    "donor_source": "strict-live-survivors-pre-batch-snapshot",
    "exclude_original_parent_index": True,
    "exclude_batch_dead_points": True,
    "exclude_evolving_endpoints": True,
    "exclude_simultaneous_newborns": True,
    "contour_acceptance": "strict-greater-than",
    "cache_policy": "waveform-rebuild-transactional-rollback",
    "prior_ratio": "complete-transformed-space-prior",
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
    "parameters": FAST_RIDGE_INTRINSIC_BLOCKS[0],
    "attempts": 8,
}
FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_POLICY = {
    **FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_POLICY,
    "attempts_per_replacement": 8,
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
COMPLEMENTARY_DE_POLICY_BY_SCHEME = {
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME: (
        FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_POLICY
    ),
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME: (
        FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_POLICY
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
PERIODIC_MH_BLOCKING_SCHEMES = {
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME,
    *COMPLEMENTARY_DE_BLOCK_BY_SCHEME,
}
BLOCKS_BY_SCHEME = {
    "paper": PAPER_BLOCKS,
    "fast-ridge": FAST_RIDGE_BLOCKS,
    FAST_RIDGE_INTRINSIC_BLOCKING_SCHEME: FAST_RIDGE_INTRINSIC_BLOCKS,
    FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME: FAST_RIDGE_INTRINSIC_BLOCKS,
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME: (FAST_RIDGE_INTRINSIC_BLOCKS),
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME: (
        FAST_RIDGE_INTRINSIC_BLOCKS
    ),
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME: (
        FAST_RIDGE_INTRINSIC_BLOCKS
    ),
    NETSKY_SCHEME: NETSKY_BLOCKS,
}
FIXED_WORK_SCHEDULE_BY_SCHEME = {
    "paper": {
        "block_sizes": (4, 3, 3, 1, 2, 1, 1),
        "rebuild_required_by_block": (True, True, True, False, False, False, False),
    },
    "fast-ridge": {
        "block_sizes": (4, 3, 3, 2, 1, 2),
        "rebuild_required_by_block": (True, True, True, False, False, False),
    },
    FAST_RIDGE_INTRINSIC_BLOCKING_SCHEME: {
        "block_sizes": (8, 1, 1, 2, 1, 2),
        "rebuild_required_by_block": (True, True, True, False, False, False),
    },
    FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME: {
        "block_sizes": (8, 1, 1, 2, 1, 2),
        "rebuild_required_by_block": (True, True, True, False, False, False),
        "num_slice_steps_by_block": FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE,
    },
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME: {
        "block_sizes": (8, 1, 1, 2, 1, 2),
        "rebuild_required_by_block": (True, True, True, False, False, False),
        "block_kernel_modes": FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES,
    },
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME: {
        "block_sizes": (8, 1, 1, 2, 1, 2),
        "rebuild_required_by_block": (True, True, True, False, False, False),
        "block_kernel_modes": FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES,
        "complementary_de_jump_block": (FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCK),
    },
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME: {
        "block_sizes": (8, 1, 1, 2, 1, 2),
        "rebuild_required_by_block": (True, True, True, False, False, False),
        "block_kernel_modes": FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES,
        "complementary_de_jump_block": (FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCK),
    },
    NETSKY_SCHEME: {
        "block_sizes": (4, 3, 3, 1, 4),
        "rebuild_required_by_block": (True, True, True, False, False),
        "bridge_blocks": NETSKY_BRIDGE_BLOCKS,
        "bridge_rebuild_required_by_block": (False,),
    },
}
POSITION_FIELDS = (
    "M_c",
    "q",
    "s1_mag",
    "s1_theta",
    "s1_phi",
    "s2_mag",
    "s2_theta",
    "s2_phi",
    "iota",
    "lambda_1",
    "lambda_2",
    "d_L",
    "ra",
    "dec",
    "psi",
)
WEIGHTED_FIELDS = (
    *POSITION_FIELDS,
    "log_likelihood",
    "log_likelihood_birth",
    "log_weights",
)
PRIMARY_DE_COUNTERS = (
    "n_likelihood_evaluations_de_jumps",
    "n_de_jump_attempts",
    "n_de_jump_acceptances",
)
CLASSIFIED_DE_COUNTERS = (
    "n_likelihood_evaluations_de_jumps_waveform_rebuild",
    "n_likelihood_evaluations_de_jumps_cache_hit",
)


@dataclass(frozen=True)
class LikelihoodArm:
    """One process-isolated likelihood cell."""

    kind: str
    label_suffix: str
    compressed: bool

    @property
    def label(self) -> str:
        """Return a provenance label for the configured device count."""

        return f"jim-paper-15d-d{N_DEVICES}-fsm-{self.label_suffix}"


FULL = LikelihoodArm(
    kind="full",
    label_suffix="full",
    compressed=False,
)
HETERODYNE = LikelihoodArm(
    kind="heterodyne5000",
    label_suffix="heterodyne5000",
    compressed=True,
)
# Run the new/cheaper path first so an invalid compression cell fails before
# paying for a full-resolution comparison.
ARMS = (HETERODYNE, FULL)
ARM_BY_KIND = {arm.kind: arm for arm in ARMS}


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _gpu_devices(value: str, n_devices: int) -> str:
    parts = value.split(",")
    if len(parts) != n_devices:
        raise argparse.ArgumentTypeError(
            f"expected exactly {n_devices} comma-separated GPU indices"
        )
    try:
        indices = [int(part) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPU indices must be integers") from error
    if any(index < 0 for index in indices) or len(set(indices)) != n_devices:
        raise argparse.ArgumentTypeError(
            "GPU indices must be non-negative and pairwise distinct"
        )
    return ",".join(str(index) for index in indices)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=DEFAULT_REFERENCE_FILE,
        help="Frozen full-likelihood point used by the 5000-bin cell.",
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        required=True,
        help=(
            "Output prefix. Mode, arm, artifact kind, and extension are added; "
            "existing targets are never overwritten."
        ),
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=2,
        help="Seed 2 is the adversarial seed selected before this A/B.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=_nonnegative_int,
        default=None,
        help=(
            "Optional independent sampler-trajectory seed. The initial live "
            "points remain pinned by --seed; omission preserves the historical "
            "paired path exactly."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("prefix", "full"),
        default="prefix",
        help="prefix runs five outer steps; full runs to the fixed termination.",
    )
    parser.add_argument(
        "--num-gibbs-sweeps",
        type=_positive_int,
        default=None,
        help="Complete blocked-kernel sweeps per replacement (paper notation M).",
    )
    parser.add_argument(
        "--blocking-scheme",
        choices=tuple(BLOCKS_BY_SCHEME),
        default="paper",
        help=(
            "Paper singleton blocks, the cacheable joint fast-ridge block, or "
            "the fixed-budget fast ridge with one coupled non-periodic "
            "intrinsic block. fast-ridge-intrinsic-5step keeps those blocks "
            "but uses five random covariance steps in the 8D block; "
            "fast-ridge-intrinsic-periodic-mh replaces the three periodic "
            "singleton slices with one-call uniform independence updates; "
            "the cde4 and cde8 variants add four or eight fixed "
            "complementary-live DE-MH moves."
        ),
    )
    parser.add_argument(
        "--n-devices",
        type=int,
        choices=(1, 4),
        default=4,
        help="One economical GPU or the multi-device referee layout.",
    )
    parser.add_argument(
        "--direction-mode",
        choices=DIRECTION_MODE_CHOICES,
        default="covariance",
        help=(
            "Fixed slice-direction law shared by every likelihood arm. "
            "covariance-basis-8d is valid only with fast-ridge-intrinsic."
        ),
    )
    parser.add_argument(
        "--gpu-devices",
        default=None,
        help="Comma-separated physical GPU indices exposed to each fresh process.",
    )
    parser.add_argument(
        "--implementation-revision",
        default=None,
        help=(
            "Exact source-tree identifier for curated exports without .git; "
            "forwarded into every arm report."
        ),
    )
    parser.add_argument(
        "--simulate-cpu",
        action="store_true",
        help="Expose four logical CPU devices for a smoke test only.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Construct and compare both likelihoods without running the sampler; "
            "writes PREFIX.preflight.json."
        ),
    )
    parser.add_argument(
        "--probe-file",
        type=Path,
        action="append",
        default=[],
        help=(
            "Repeatable prior-space weighted NPZ used only by --preflight-only. "
            "At least three distinct cross-mode probes are required for a pass."
        ),
    )
    parser.add_argument(
        "--_cell-kind",
        choices=tuple(ARM_BY_KIND),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.num_gibbs_sweeps is None:
        args.num_gibbs_sweeps = (
            NETSKY_NUM_GIBBS_SWEEPS if args.blocking_scheme == NETSKY_SCHEME else 1
        )
    if args.gpu_devices is None:
        args.gpu_devices = ",".join(str(index) for index in range(args.n_devices))
    try:
        args.gpu_devices = _gpu_devices(args.gpu_devices, args.n_devices)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.preflight_only and args._cell_kind is not None:
        parser.error("--preflight-only cannot be combined with an internal cell")
    if args.probe_file and not args.preflight_only:
        parser.error("--probe-file requires --preflight-only")
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
        args.blocking_scheme == FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME
        and args.n_devices != 1
    ):
        parser.error(
            "--blocking-scheme fast-ridge-intrinsic-periodic-mh requires --n-devices 1"
        )
    if (
        args.blocking_scheme == (FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME)
        and args.direction_mode != "covariance"
    ):
        parser.error(
            "--blocking-scheme fast-ridge-intrinsic-periodic-mh-cde4 requires "
            "--direction-mode covariance"
        )
    if (
        args.blocking_scheme == (FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME)
        and args.n_devices != 1
    ):
        parser.error(
            "--blocking-scheme fast-ridge-intrinsic-periodic-mh-cde4 requires "
            "--n-devices 1"
        )
    if (
        args.blocking_scheme == (FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME)
        and args.direction_mode != "covariance"
    ):
        parser.error(
            "--blocking-scheme fast-ridge-intrinsic-periodic-mh-cde8 requires "
            "--direction-mode covariance"
        )
    if (
        args.blocking_scheme == (FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE8_BLOCKING_SCHEME)
        and args.n_devices != 1
    ):
        parser.error(
            "--blocking-scheme fast-ridge-intrinsic-periodic-mh-cde8 requires "
            "--n-devices 1"
        )
    if args.blocking_scheme == NETSKY_SCHEME and (
        args.num_gibbs_sweeps != NETSKY_NUM_GIBBS_SWEEPS
    ):
        parser.error("--blocking-scheme netsky requires --num-gibbs-sweeps 2")
    if args.num_gibbs_sweeps != 1 and args.blocking_scheme not in {
        FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME,
        NETSKY_SCHEME,
    }:
        parser.error(
            "--num-gibbs-sweeps other than 1 is scoped to "
            "fast-ridge-intrinsic-periodic-mh"
        )
    return args


def _configure_runtime(args: argparse.Namespace) -> None:
    """Apply one fail-closed sampler layout before constructing any cell."""

    global BLOCKING_SCHEME, DIRECTION_MODE, N_DEVICES, NUM_GIBBS_SWEEPS, PAPER_BLOCKS
    global FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK
    BLOCKING_SCHEME = args.blocking_scheme
    DIRECTION_MODE = args.direction_mode
    N_DEVICES = args.n_devices
    NUM_GIBBS_SWEEPS = args.num_gibbs_sweeps
    PAPER_BLOCKS = BLOCKS_BY_SCHEME[args.blocking_scheme]
    FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK = {
        "total_updates": 15 * NUM_GIBBS_SWEEPS,
        "total_slice_updates": 12 * NUM_GIBBS_SWEEPS,
        "waveform_rebuild_slice_updates": 8 * NUM_GIBBS_SWEEPS,
        "cache_hit_slice_updates": 4 * NUM_GIBBS_SWEEPS,
        "periodic_independence_attempts": 3 * NUM_GIBBS_SWEEPS,
        "waveform_rebuild_periodic_independence_attempts": (2 * NUM_GIBBS_SWEEPS),
        "cache_hit_periodic_independence_attempts": NUM_GIBBS_SWEEPS,
        "cache_segments": 2,
    }
    from benchmarks.device_parallel_nss import benchmark_gw170817_full_run as benchmark

    benchmark.NUM_GIBBS_SWEEPS = NUM_GIBBS_SWEEPS
    benchmark.FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK = dict(
        FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK
    )
    benchmark.PAPER_FAST_RIDGE_INTRINSIC_PERIODIC_MH_LIMITATIONS = (
        (
            "This preserves the fast-ridge-intrinsic blocks and their default "
            "dimension-based work budget, but replaces the s1_phi, s2_phi, and "
            "psi singleton slices with prior-corrected uniform independence "
            f"updates on their periodic supports. With M={NUM_GIBBS_SWEEPS}, the "
            f"fixed work is {12 * NUM_GIBBS_SWEEPS} slices plus "
            f"{3 * NUM_GIBBS_SWEEPS} independence attempts in two cache segments."
        ),
        *benchmark.PAPER_LIMITATIONS[1:],
    )


def _benchmark_name() -> str:
    return f"gw170817-full-swig-{N_DEVICES}gpu"


def _pair_name() -> str:
    suffix = "" if DIRECTION_MODE == "covariance" else f"-{DIRECTION_MODE}"
    return f"jim-paper-15d-d{N_DEVICES}-fsm-likelihood-pair{suffix}"


def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_paths(
    prefix: Path,
    mode: str,
    arm: LikelihoodArm | None = None,
) -> dict[str, Path]:
    base = prefix.expanduser().resolve()
    if arm is None:
        return {
            "pair": Path(f"{base}.{mode}.likelihood-pair.json"),
            "preflight": Path(f"{base}.preflight.json"),
        }
    stem = f"{base}.{mode}.{arm.label}"
    paths = {
        "report": Path(f"{stem}.report.json"),
        "weighted": Path(f"{stem}.weighted.npz"),
    }
    if BLOCKING_SCHEME == NETSKY_SCHEME:
        paths["folded"] = Path(f"{stem}.folded-nested-diagnostics.npz")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _config_sha256(config: Mapping[str, Any]) -> str:
    """Hash runner-owned config, excluding derived/report-only fields."""

    payload = {
        key: value
        for key, value in config.items()
        if key not in {"sha256", "initial_positions_sha256"}
    }
    return _canonical_sha256(payload)


def _source_manifest(repository: Path) -> dict[str, Any]:
    roots = (
        repository / "src" / "jimgw",
        repository
        / "benchmarks"
        / "device_parallel_nss"
        / "benchmark_gw170817_full_run.py",
        repository / "benchmarks" / "device_parallel_nss" / "sampler_ablation.py",
        repository / "benchmarks" / "device_parallel_nss" / "paper_model.py",
        repository / "benchmarks" / "device_parallel_nss" / "paper_model_basis.py",
        repository / "benchmarks" / "device_parallel_nss" / "paper_heterodyne.py",
        repository
        / "benchmarks"
        / "device_parallel_nss"
        / "preflight_gw170817_likelihood_pair.py",
        repository
        / "benchmarks"
        / "device_parallel_nss"
        / "likelihood_pair_preflight_contract.py",
        repository / "benchmarks" / "injection_campaign" / "common.py",
        repository / "benchmarks" / "injection_campaign" / "folded_results.py",
        repository / "benchmarks" / "injection_campaign" / "run_injection.py",
        Path(__file__).resolve(),
    )
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(path for path in root.rglob("*.py") if path.is_file())
        elif root.is_file():
            paths.append(root)
        else:
            raise RuntimeError(f"required source path does not exist: {root}")

    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(repository).as_posix()
        digest = _sha256(path)
        files.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
    return {
        "sha256": aggregate.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"JSON document is not an object: {path}")
    return value


def _finite_float_mapping(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{name} fields are {actual!r}, expected {expected!r}")
    invalid = [
        key
        for key, item in value.items()
        if isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
    ]
    if invalid:
        raise ValueError(f"{name} has non-finite/non-numeric fields: {invalid}")


def _load_reference(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"frozen heterodyne reference does not exist: {resolved}")
    reference = _load_json(resolved)
    if reference.get("schema_version") != 1:
        raise ValueError("heterodyne reference schema_version must be 1")
    contract = reference.get("waveform_contract")
    expected_contract = {
        "model": WAVEFORM,
        "f_ref_hz": WAVEFORM_F_REF_HZ,
        "time_anchor": CARRIER_TIME_ANCHOR,
    }
    if contract != expected_contract:
        raise ValueError(
            f"heterodyne waveform contract {contract!r} != {expected_contract!r}"
        )

    likelihood_fields = {
        "M_c",
        "eta",
        "s1_x",
        "s1_y",
        "s1_z",
        "s2_x",
        "s2_y",
        "s2_z",
        "iota",
        "lambda_1",
        "lambda_2",
        "d_L",
        "ra",
        "dec",
        "psi",
        "t_c",
        "phase_c",
    }
    sampling_fields = set(POSITION_FIELDS)
    _finite_float_mapping(
        reference.get("likelihood_parameters"),
        likelihood_fields,
        "likelihood_parameters",
    )
    _finite_float_mapping(
        reference.get("sampling_parameters"),
        sampling_fields,
        "sampling_parameters",
    )
    likelihood = reference["likelihood_parameters"]
    sampling = reference["sampling_parameters"]
    if likelihood["t_c"] != 0.0 or likelihood["phase_c"] != 0.0:
        raise ValueError("marginalized reference t_c and phase_c must both be zero")
    expected_eta = sampling["q"] / (1.0 + sampling["q"]) ** 2
    if not math.isclose(likelihood["eta"], expected_eta, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("reference q and eta are inconsistent")
    for prefix in ("s1", "s2"):
        magnitude = sampling[f"{prefix}_mag"]
        theta = sampling[f"{prefix}_theta"]
        phi = sampling[f"{prefix}_phi"]
        expected_xyz = (
            magnitude * math.sin(theta) * math.cos(phi),
            magnitude * math.sin(theta) * math.sin(phi),
            magnitude * math.cos(theta),
        )
        actual_xyz = tuple(likelihood[f"{prefix}_{axis}"] for axis in "xyz")
        if not all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-14)
            for actual, expected in zip(actual_xyz, expected_xyz, strict=True)
        ):
            raise ValueError(f"reference {prefix} spherical/Cartesian spins disagree")
    if not math.isfinite(float(reference.get("source_log_likelihood", math.nan))):
        raise ValueError("reference source_log_likelihood must be finite")
    for name in ("corrected_u1_log_likelihood", "selection_u16_log_likelihood"):
        if not math.isfinite(float(reference.get(name, math.nan))):
            raise ValueError(f"reference {name} must be finite")
    source_sha = reference.get("source_sha256")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("reference source_sha256 is not a lowercase SHA-256")
    if not isinstance(reference.get("source_path"), str):
        raise TypeError("reference source_path must be a string")
    return reference, _sha256(resolved)


def _key_hashes(seed: int, sampler_seed: int | None = None) -> dict[str, str]:
    import jax

    root_key = jax.random.key(seed)
    sampler_root_key = jax.random.key(seed if sampler_seed is None else sampler_seed)
    _, sampler_key = jax.random.split(sampler_root_key)

    def digest(key: Any) -> str:
        array = np.asarray(jax.device_get(jax.random.key_data(key)))
        return hashlib.sha256(array.tobytes(order="C")).hexdigest()

    return {
        "initial_positions_rng_key_sha256": digest(root_key),
        "sampler_rng_key_sha256": digest(sampler_key),
    }


def _bin_metadata(likelihood: Any) -> dict[str, Any]:
    low = np.asarray(likelihood.freq_grid_low, dtype="<f8")
    high = np.asarray(likelihood.freq_grid_high, dtype="<f8")
    if low.ndim != 1 or high.shape != low.shape or low.size == 0:
        raise RuntimeError("heterodyne bin-edge arrays have invalid shapes")
    if not np.array_equal(low[1:], high[:-1]):
        raise RuntimeError("heterodyne bins do not share identical adjacent edges")
    edges = np.ascontiguousarray(np.concatenate((low, high[-1:])), dtype="<f8")
    if np.any(np.diff(edges) <= 0.0):
        raise RuntimeError("heterodyne bin edges are not strictly increasing")
    requested = int(getattr(likelihood, "requested_n_bins", -1))
    realized = int(likelihood.n_bins)
    if (
        requested != N_BINS_REQUESTED
        or realized != N_BINS_REQUESTED
        or realized != low.size
    ):
        raise RuntimeError(
            "paper likelihood requires exactly 5000 requested and realized bins; "
            f"got requested={requested}, realized={realized}"
        )
    denominator_floors: dict[str, float] = {}
    for detector_name, values in likelihood.waveform_low_ref.items():
        low_values = np.asarray(values)
        high_values = np.asarray(likelihood.waveform_high_ref[detector_name])
        minimum = float(min(np.min(np.abs(low_values)), np.min(np.abs(high_values))))
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise RuntimeError(
                f"reference waveform denominator vanishes for {detector_name}"
            )
        denominator_floors[detector_name] = minimum
    return {
        "requested_bins": requested,
        "realized_bins": realized,
        "bin_edges_count": int(edges.size),
        "bin_edges_dtype": "float64-little-endian",
        "bin_edges_sha256": hashlib.sha256(edges.tobytes(order="C")).hexdigest(),
        "first_edge_hz": float(edges[0]),
        "last_edge_hz": float(edges[-1]),
        "reference_denominator_min_abs_by_detector": denominator_floors,
    }


def _time_metadata(likelihood: Any, *, compressed: bool) -> dict[str, Any]:
    if compressed:
        window = np.asarray(likelihood.tc_window, dtype="<f8")
        normalization_count = int(likelihood._tc_normalization_count)
    else:
        coarse_count = len(likelihood.tc_array)
        normalization_count = coarse_count * int(likelihood.tc_upsample)
        candidates = np.asarray(
            likelihood._tc_fine_candidate_indices,
            dtype=np.int64,
        )
        fine_mask = np.asarray(likelihood._tc_fine_mask, dtype=bool)
        storage = (
            candidates[None, :] * int(likelihood.tc_upsample)
            + np.arange(int(likelihood.tc_upsample))[:, None]
        )
        q_max = (normalization_count - 1) // 2
        signed = np.where(storage <= q_max, storage, storage - normalization_count)
        fine_times = signed * (DURATION_SECONDS / normalization_count)
        window = np.asarray(np.sort(fine_times[fine_mask]), dtype="<f8")
    tc_window_count = len(window)
    expected_normalization = int(
        DURATION_SECONDS
        * TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ
        / 2.0
        * TIME_UPSAMPLE_FACTOR
    )
    if normalization_count != expected_normalization:
        raise RuntimeError(
            f"time normalization count {normalization_count} != "
            f"{expected_normalization}"
        )
    if tuple(float(value) for value in likelihood.tc_range) != TC_RANGE_SECONDS:
        raise RuntimeError("likelihood did not retain the fixed t_c interval")
    if int(likelihood.tc_upsample) != TIME_UPSAMPLE_FACTOR or tc_window_count <= 0:
        raise RuntimeError("likelihood did not use the required U=8 time grid")
    fine_step = DURATION_SECONDS / expected_normalization
    first = math.floor(TC_RANGE_SECONDS[0] / fine_step) + 1
    last = math.ceil(TC_RANGE_SECONDS[1] / fine_step) - 1
    expected_window = np.asarray(
        fine_step * np.arange(first, last + 1),
        dtype="<f8",
    )
    if not np.array_equal(window, expected_window):
        raise RuntimeError("likelihood time window does not match the pinned U=8 grid")
    return {
        "tc_range_seconds": list(TC_RANGE_SECONDS),
        "upsample_factor": TIME_UPSAMPLE_FACTOR,
        "window_point_count": int(tc_window_count),
        "window_sha256": hashlib.sha256(
            np.ascontiguousarray(window).tobytes(order="C")
        ).hexdigest(),
        "normalization_point_count": normalization_count,
        "resolution_seconds": DURATION_SECONDS / normalization_count,
        "limitation": (
            "U=8 is a converged corrective sub-grid time marginalization: the stock "
            "public-proxy U=1 grid causes parameter-dependent logL errors"
        ),
    }


def _coefficient_builder_metadata(likelihood: Any) -> dict[str, Any]:
    method = type(likelihood)._compute_coefficients
    source = inspect.getsource(method).encode()
    return {
        "qualified_name": f"{method.__module__}.{method.__qualname__}",
        "algorithm": getattr(likelihood, "coefficient_builder", None),
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }


def _likelihood_metadata(
    arm: LikelihoodArm,
    likelihood: Any,
    reference_sha256: str,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "kind": arm.kind,
        "class": f"{type(likelihood).__module__}.{type(likelihood).__qualname__}",
        "phase_marginalization": "analytic network log-I0",
        "time_marginalization": TIME_MARGINALIZATION_DESCRIPTION,
        "time_grid": _time_metadata(likelihood, compressed=arm.compressed),
        "distance_marginalization": False,
        "reference_json_sha256": reference_sha256,
        "reference_used": arm.compressed,
    }
    if arm.compressed:
        common.update(
            {
                "compression": "relative-binning linear bin-edge expansion",
                "bins": _bin_metadata(likelihood),
                "coefficient_builder": _coefficient_builder_metadata(likelihood),
            }
        )
    else:
        common.update(
            {
                "compression": None,
                "bins": None,
                "coefficient_builder": None,
            }
        )
    return common


def _install_scientific_target(
    benchmark: Any,
    arm: LikelihoodArm,
    reference: Mapping[str, Any],
    reference_sha256: str,
) -> Callable[[], None]:
    """Patch only the fresh cell process and return a fail-closed verifier."""

    original_analysis = benchmark._analysis_components
    original_config = benchmark._config_report
    used = {"analysis": 0, "likelihood": 0, "config": 0}
    state: dict[str, Any] = {}

    def anchored_analysis(
        workload: str,
        jnp: Any,
        ifos: list[Any],
        *,
        blocking_scheme: str = BLOCKING_SCHEME,
    ) -> dict[str, Any]:
        components = original_analysis(
            workload,
            jnp,
            ifos,
            blocking_scheme=blocking_scheme,
        )
        if workload != WORKLOAD or blocking_scheme != BLOCKING_SCHEME:
            raise RuntimeError(
                "likelihood pair requires paper-15d with the paper block partition"
            )
        from benchmarks.device_parallel_nss.paper_model import (
            RippleIMRPhenomPv2NRTidalv2,
        )

        waveform = RippleIMRPhenomPv2NRTidalv2(
            f_ref=WAVEFORM_F_REF_HZ,
            time_anchor=CARRIER_TIME_ANCHOR,
        )
        if (
            waveform.f_ref != WAVEFORM_F_REF_HZ
            or waveform.time_anchor != CARRIER_TIME_ANCHOR
        ):
            raise RuntimeError("paper waveform did not retain its explicit anchor")
        used["analysis"] += 1
        return {**components, "waveform": waveform}

    benchmark._analysis_components = anchored_analysis

    import jimgw.core.single_event.likelihood as likelihood_module

    full_type = likelihood_module.TransientLikelihoodFD

    def likelihood_factory(*args: Any, **kwargs: Any) -> Any:
        if used["likelihood"]:
            raise RuntimeError("expected exactly one likelihood construction per cell")
        expected_kwargs = {
            "trigger_time": benchmark.GPS,
            "f_min": benchmark.F_MIN,
            "f_max": benchmark.F_MAX,
            "phase_marginalization": True,
        }
        for name, expected in expected_kwargs.items():
            if kwargs.get(name) != expected:
                raise RuntimeError(
                    f"likelihood factory got {name}={kwargs.get(name)!r}, "
                    f"expected {expected!r}"
                )
        input_time_config = kwargs.get("time_marginalization")
        if input_time_config != {"tc_range": TC_RANGE_SECONDS}:
            raise RuntimeError(
                "full runner changed its stock time-marginalization input: "
                f"{input_time_config!r}"
            )
        # The stock public-proxy grid is U=1.  Frozen-data adversarial checks
        # found parameter-dependent 1--11 nat errors at that resolution.  Force
        # the same converged U=8 grid in both A/B cells at this single seam.
        kwargs = {
            **kwargs,
            "time_marginalization": {
                "tc_range": TC_RANGE_SECONDS,
                "upsample_factor": TIME_UPSAMPLE_FACTOR,
            },
        }
        waveform = kwargs.get("waveform")
        if (
            getattr(waveform, "f_ref", None) != WAVEFORM_F_REF_HZ
            or getattr(waveform, "time_anchor", None) != CARRIER_TIME_ANCHOR
        ):
            raise RuntimeError("likelihood factory received the wrong waveform anchor")
        if arm.compressed:
            from benchmarks.device_parallel_nss.paper_heterodyne import (
                PaperTimeMarginalizedHeterodynedLikelihoodFD,
            )

            likelihood = PaperTimeMarginalizedHeterodynedLikelihoodFD(
                *args,
                **kwargs,
                n_bins=N_BINS_REQUESTED,
                reference_parameters=dict(reference["likelihood_parameters"]),
            )
        else:
            likelihood = full_type(*args, **kwargs)
        if (
            not likelihood.phase_marginalization
            or not likelihood.time_marginalization
            or likelihood.distance_marginalization
        ):
            raise RuntimeError("likelihood marginalization contract was not retained")
        used["likelihood"] += 1
        state["likelihood"] = likelihood
        state["metadata"] = _likelihood_metadata(
            arm,
            likelihood,
            reference_sha256,
        )
        return likelihood

    likelihood_module.TransientLikelihoodFD = likelihood_factory

    def config_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if "metadata" not in state:
            raise RuntimeError("config was generated before the likelihood existed")
        config = original_config(*args, **kwargs)
        if config.get("workload") != WORKLOAD:
            raise RuntimeError("config does not describe paper-15d")
        if config.get("blocking_scheme") != BLOCKING_SCHEME:
            raise RuntimeError("config does not describe the paper block partition")
        config["carrier_time_anchor"] = CARRIER_TIME_ANCHOR
        config["likelihood"] = state["metadata"]
        config["sha256"] = _config_sha256(config)
        used["config"] += 1
        return config

    benchmark._config_report = config_report

    def verify() -> None:
        if used != {"analysis": 1, "likelihood": 1, "config": 1}:
            raise RuntimeError(
                f"scientific target hooks were not used exactly once: {used}"
            )

    return verify


def _pairing_metadata(
    args: argparse.Namespace,
    arm: LikelihoodArm,
    source: Mapping[str, Any],
    data_sha256: str,
    reference: Mapping[str, Any],
    reference_sha256: str,
) -> dict[str, Any]:
    return {
        "experiment": f"D={N_DEVICES} full-vs-heterodyne5000 likelihood A/B",
        "workload": WORKLOAD,
        "blocking_scheme": BLOCKING_SCHEME,
        "arm_label": arm.label,
        "likelihood_kind": arm.kind,
        "n_devices": N_DEVICES,
        "mode": args.mode,
        "prefix_steps": PREFIX_STEPS if args.mode == "prefix" else None,
        "source_tree_sha256": source["sha256"],
        "data_sha256": data_sha256,
        "reference_json_sha256": reference_sha256,
        "reference_source_path": reference["source_path"],
        "reference_source_sha256": reference["source_sha256"],
        "reference_source_log_likelihood": reference["source_log_likelihood"],
        "reference_corrected_u1_log_likelihood": reference[
            "corrected_u1_log_likelihood"
        ],
        "reference_selection_u16_log_likelihood": reference[
            "selection_u16_log_likelihood"
        ],
        **_key_hashes(args.seed, sampler_seed=args.sampler_seed),
        **(
            {
                "trajectory_seed_override": {
                    "initial_positions_seed": args.seed,
                    "sampler_seed": args.sampler_seed,
                }
            }
            if args.sampler_seed is not None
            else {}
        ),
        "proposal": {
            "direction_parameter": "covariance",
            "direction_mode": DIRECTION_MODE,
            **(
                {"num_slice_steps_by_block": list(FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE)}
                if BLOCKING_SCHEME == FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME
                else {}
            ),
            **(
                {
                    "block_kernel_modes": list(
                        FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
                    ),
                    "fixed_work": dict(FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK),
                }
                if BLOCKING_SCHEME == FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME
                else {}
            ),
            **(
                {
                    "block_kernel_modes": list(
                        FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
                    ),
                    "complementary_de_jump_block": {
                        "parameters": list(
                            COMPLEMENTARY_DE_BLOCK_BY_SCHEME[BLOCKING_SCHEME][
                                "parameters"
                            ]
                        ),
                        "attempts": COMPLEMENTARY_DE_BLOCK_BY_SCHEME[BLOCKING_SCHEME][
                            "attempts"
                        ],
                    },
                    "complementary_de_policy": dict(
                        COMPLEMENTARY_DE_POLICY_BY_SCHEME[BLOCKING_SCHEME]
                    ),
                    "fixed_work": dict(
                        COMPLEMENTARY_DE_FIXED_WORK_BY_SCHEME[BLOCKING_SCHEME]
                    ),
                }
                if BLOCKING_SCHEME in COMPLEMENTARY_DE_BLOCK_BY_SCHEME
                else {}
            ),
            **(
                {
                    "bridge_blocks": [list(block) for block in NETSKY_BRIDGE_BLOCKS],
                    "periodic_wrapped_covariance": True,
                    "fixed_work": dict(NETSKY_FIXED_WORK),
                }
                if BLOCKING_SCHEME == NETSKY_SCHEME
                else {}
            ),
            "num_de_jumps": 0,
            "de_jump_blocks": [],
        },
        "paper_notation": {
            "D_devices": N_DEVICES,
            "m_live_points": N_LIVE,
            "k_deleted_points": N_DELETE,
            "M_gibbs_sweeps": NUM_GIBBS_SWEEPS,
        },
        "waveform": {
            "model": WAVEFORM,
            "f_ref_hz": WAVEFORM_F_REF_HZ,
            "time_anchor": CARRIER_TIME_ANCHOR,
        },
    }


def _assert_fixed_sampler_contract(benchmark: Any) -> None:
    from benchmarks.device_parallel_nss.sampler_ablation import variant

    selected = variant(FSM_VARIANT).report()
    expected_variant = {
        "topology": "replicated-live",
        "interval": "cached-stepping-out",
        "scheduler": "fsm",
        "direction_parameter": "covariance",
    }
    for name, expected in expected_variant.items():
        if selected.get(name) != expected:
            raise RuntimeError(
                f"{FSM_VARIANT} violates fixed {name}: "
                f"{selected.get(name)!r} != {expected!r}"
            )
    if (
        benchmark.NUM_INNER_STEPS_PER_DIM != NUM_INNER_STEPS_PER_DIM
        or benchmark.NUM_GIBBS_SWEEPS != NUM_GIBBS_SWEEPS
        or benchmark.N_LIVE != N_LIVE
        or benchmark.N_DELETE != N_DELETE
        or benchmark.TERMINATION_DLOGZ != TERMINATION_DLOGZ
    ):
        raise RuntimeError("full runner constants no longer match the frozen A/B")
    if BLOCKING_SCHEME == NETSKY_SCHEME:
        expected_netsky_constants = {
            "NETSKY_SCHEME": NETSKY_SCHEME,
            "NETSKY_BLOCKS": NETSKY_BLOCKS,
            "NETSKY_BRIDGE_BLOCKS": NETSKY_BRIDGE_BLOCKS,
            "NETSKY_NUM_GIBBS_SWEEPS": NETSKY_NUM_GIBBS_SWEEPS,
            "NETSKY_FIXED_WORK": NETSKY_FIXED_WORK,
        }
        for name, expected in expected_netsky_constants.items():
            if getattr(benchmark, name, None) != expected:
                raise RuntimeError(
                    f"full runner {name} no longer matches the NETSKY referee"
                )
    spec = benchmark._workload_spec(WORKLOAD, BLOCKING_SCHEME)
    expected_spec = {
        "waveform": WAVEFORM,
        "sampled_dimensions": 15,
        "blocks": PAPER_BLOCKS,
        "mass_ratio_range": (0.125, 1.0),
        "distance_range_mpc": (1.0, 75.0),
    }
    for name, expected in expected_spec.items():
        if spec.get(name) != expected:
            raise RuntimeError(
                f"paper workload violates fixed {name}: "
                f"{spec.get(name)!r} != {expected!r}"
            )


def _configure_cell_artifacts(
    benchmark_args: argparse.Namespace,
    paths: Mapping[str, Path],
    *,
    mode: str,
) -> None:
    """Request the scientific products owned by one likelihood cell."""

    benchmark_args.timing_only = False
    benchmark_args.nested_output = paths["weighted"]
    if BLOCKING_SCHEME == NETSKY_SCHEME:
        benchmark_args.folded_nested_output = paths["folded"]
    benchmark_args.samples_output = None
    benchmark_args.max_outer_steps = PREFIX_STEPS if mode == "prefix" else None


def _run_cell(args: argparse.Namespace, arm: LikelihoodArm) -> None:
    from benchmarks.device_parallel_nss import benchmark_gw170817_full_run as benchmark

    _assert_fixed_sampler_contract(benchmark)
    paths = _artifact_paths(args.prefix, args.mode, arm)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise SystemExit(
            "refusing to overwrite existing cell artifacts: " + ", ".join(existing)
        )

    data_path = args.data_file.expanduser().resolve()
    data_sha256 = _sha256(data_path)
    if data_sha256 != CANONICAL_DATA_SHA256:
        raise SystemExit(
            f"internal cell rejected non-canonical frozen data: {data_sha256}"
        )
    reference, reference_sha256 = _load_reference(args.reference_file)
    parse_argv = [
        "--data-file",
        str(data_path),
        "--workload",
        WORKLOAD,
        "--blocking-scheme",
        BLOCKING_SCHEME,
        "--seed",
        str(args.seed),
        "--n-devices",
        str(N_DEVICES),
        "--direction-mode",
        DIRECTION_MODE,
        "--num-de-jumps",
        "0",
        "--implementation-label",
        arm.label,
        "--output",
        str(paths["report"]),
        "--timing-only",
        "--ablation-variant",
        FSM_VARIANT,
    ]
    if args.sampler_seed is not None:
        parse_argv.extend(["--sampler-seed", str(args.sampler_seed)])
    if args.implementation_revision is not None:
        parse_argv.extend(["--implementation-revision", args.implementation_revision])
    if args.simulate_cpu:
        parse_argv.append("--simulate-cpu")
    benchmark_args = benchmark._parse_args(parse_argv)

    # The benchmark CLI restricts ablations to timing mode, while its execution
    # seam supports scientific extraction.  Change only this parsed flag and
    # explicitly request the weighted nested-point artifact.
    _configure_cell_artifacts(benchmark_args, paths, mode=args.mode)

    benchmark._select_implementation(benchmark_args.implementation_root)
    if benchmark_args.simulate_cpu:
        benchmark._configure_cpu_simulation(N_DEVICES)
    verify_target = _install_scientific_target(
        benchmark,
        arm,
        reference,
        reference_sha256,
    )
    source = _source_manifest(_repository())
    report = benchmark._run(benchmark_args)
    verify_target()
    if report["data"]["sha256"] != data_sha256:
        raise RuntimeError("runner data hash changed inside a likelihood cell")
    report["benchmark"] = _benchmark_name()
    report["paired_likelihood_referee"] = _pairing_metadata(
        args,
        arm,
        source,
        data_sha256,
        reference,
        reference_sha256,
    )
    benchmark._emit_report(report, paths["report"])


def _get_path(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for component in dotted_path.split("."):
        current = current[component]
    return current


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right,
        sort_keys=True,
        allow_nan=False,
    )


def _first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left:
                return {"path": child, "left": "<missing>", "right": right[key]}
            if key not in right:
                return {"path": child, "left": left[key], "right": "<missing>"}
            if not _same_json(left[key], right[key]):
                return _first_difference(left[key], right[key], child)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {"path": f"{path}.length", "left": len(left), "right": len(right)}
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            if not _same_json(left_item, right_item):
                return _first_difference(left_item, right_item, f"{path}[{index}]")
    return {"path": path, "left": left, "right": right}


def _validate_likelihood_metadata(
    value: Any,
    arm: LikelihoodArm,
    reference_sha256: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["config.likelihood is missing or not an object"]
    errors: list[str] = []
    expected = {
        "kind": arm.kind,
        "class": (
            "benchmarks.device_parallel_nss.paper_heterodyne."
            "PaperTimeMarginalizedHeterodynedLikelihoodFD"
            if arm.compressed
            else "jimgw.core.single_event.likelihood.TransientLikelihoodFD"
        ),
        "reference_used": arm.compressed,
        "reference_json_sha256": reference_sha256,
        "distance_marginalization": False,
        "phase_marginalization": "analytic network log-I0",
        "time_marginalization": TIME_MARGINALIZATION_DESCRIPTION,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            errors.append(
                f"config.likelihood.{name}: {value.get(name)!r} != {expected_value!r}"
            )
    time_grid = value.get("time_grid")
    expected_normalization = int(
        DURATION_SECONDS
        * TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ
        / 2.0
        * TIME_UPSAMPLE_FACTOR
    )
    if not isinstance(time_grid, Mapping):
        errors.append("config.likelihood.time_grid is missing")
    else:
        if time_grid.get("tc_range_seconds") != list(TC_RANGE_SECONDS):
            errors.append("config.likelihood.time_grid has the wrong t_c range")
        if time_grid.get("upsample_factor") != TIME_UPSAMPLE_FACTOR:
            errors.append("config.likelihood.time_grid has the wrong upsample factor")
        if time_grid.get("normalization_point_count") != expected_normalization:
            errors.append("config.likelihood.time_grid has the wrong normalization")
        if (
            not isinstance(time_grid.get("window_point_count"), int)
            or time_grid.get("window_point_count", 0) <= 0
        ):
            errors.append("config.likelihood.time_grid has no window points")
        fine_step = DURATION_SECONDS / expected_normalization
        first = math.floor(TC_RANGE_SECONDS[0] / fine_step) + 1
        last = math.ceil(TC_RANGE_SECONDS[1] / fine_step) - 1
        expected_window = np.asarray(
            fine_step * np.arange(first, last + 1),
            dtype="<f8",
        )
        expected_window_sha = hashlib.sha256(
            np.ascontiguousarray(expected_window).tobytes(order="C")
        ).hexdigest()
        if time_grid.get("window_point_count") != len(expected_window):
            errors.append("config.likelihood.time_grid has the wrong point count")
        if time_grid.get("window_sha256") != expected_window_sha:
            errors.append("config.likelihood.time_grid has the wrong window hash")
    if arm.compressed:
        bins = value.get("bins")
        if not isinstance(bins, Mapping):
            errors.append("compressed likelihood has no bin metadata")
        else:
            realized = bins.get("realized_bins")
            if bins.get("requested_bins") != N_BINS_REQUESTED:
                errors.append("compressed likelihood did not request 5000 bins")
            if realized != N_BINS_REQUESTED:
                errors.append("compressed likelihood did not realize all 5000 bins")
            if bins.get("bin_edges_count") != (
                realized + 1 if isinstance(realized, int) else None
            ):
                errors.append("compressed likelihood edge count is inconsistent")
            edge_sha = bins.get("bin_edges_sha256")
            if not isinstance(edge_sha, str) or len(edge_sha) != 64:
                errors.append("compressed likelihood has no valid bin-edge SHA-256")
        builder = value.get("coefficient_builder")
        if (
            not isinstance(builder, Mapping)
            or builder.get("algorithm") != "numpy-segmented-v1"
            or not str(builder.get("qualified_name", "")).endswith(
                "PaperTimeMarginalizedHeterodynedLikelihoodFD._compute_coefficients"
            )
        ):
            errors.append("compressed likelihood coefficient builder is not pinned")
        elif (
            not isinstance(builder.get("source_sha256"), str)
            or len(builder["source_sha256"]) != 64
        ):
            errors.append("compressed coefficient builder has no valid source hash")
        denominator_floors = (
            bins.get("reference_denominator_min_abs_by_detector", {})
            if isinstance(bins, Mapping)
            else {}
        )
        if set(denominator_floors) != set(DETECTORS) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in denominator_floors.values()
        ):
            errors.append("compressed reference denominator floors are invalid")
    elif any(
        value.get(name) is not None
        for name in ("compression", "bins", "coefficient_builder")
    ):
        errors.append("full likelihood unexpectedly reports compression metadata")
    return errors


def _validate_periodic_independence_telemetry(
    results: Mapping[str, Any],
    *,
    iterations: int,
) -> list[str]:
    """Validate the exact H4 block/history/count accounting in one arm."""

    errors: list[str] = []
    replacements = iterations * N_DELETE
    expected_blocks = (("s1_phi", True), ("s2_phi", True), ("psi", False))
    attempts = np.asarray(
        results.get("periodic_uniform_independence_attempts_by_block_history")
    )
    acceptances = np.asarray(
        results.get("periodic_uniform_independence_acceptances_by_block_history")
    )
    histories_valid = (
        attempts.shape == acceptances.shape
        and attempts.ndim >= 2
        and attempts.shape[-1] == len(expected_blocks)
        and attempts.size == replacements * len(expected_blocks)
        and np.issubdtype(attempts.dtype, np.integer)
        and np.issubdtype(acceptances.dtype, np.integer)
        and np.all(attempts == NUM_GIBBS_SWEEPS)
        and np.all((0 <= acceptances) & (acceptances <= NUM_GIBBS_SWEEPS))
    )
    if not histories_valid:
        return ["results.periodic_uniform_independence block histories are invalid"]
    attempts_flat = attempts.reshape(-1, len(expected_blocks))
    acceptances_flat = acceptances.reshape(-1, len(expected_blocks))
    per_block_attempts = attempts_flat.sum(axis=0, dtype=np.int64)
    per_block_acceptances = acceptances_flat.sum(axis=0, dtype=np.int64)
    raw_blocks = results.get("periodic_uniform_independence_blocks")
    if not isinstance(raw_blocks, list) or len(raw_blocks) != len(expected_blocks):
        errors.append("results.periodic_uniform_independence_blocks is invalid")
    else:
        for index, (parameter, requires_rebuild) in enumerate(expected_blocks):
            raw = raw_blocks[index]
            expected_attempts = int(per_block_attempts[index])
            expected_acceptances = int(per_block_acceptances[index])
            expected_rate = expected_acceptances / expected_attempts
            expected = {
                "parameters": [parameter],
                "requires_waveform_rebuild": requires_rebuild,
                "attempts_per_replacement": NUM_GIBBS_SWEEPS,
                "n_attempts": expected_attempts,
                "n_acceptances": expected_acceptances,
            }
            if not isinstance(raw, Mapping):
                errors.append(f"results periodic block {parameter} is not an object")
                continue
            for name, expected_value in expected.items():
                if raw.get(name) != expected_value:
                    errors.append(
                        f"results periodic block {parameter}.{name}: "
                        f"{raw.get(name)!r} != {expected_value!r}"
                    )
            rate = raw.get("acceptance_rate")
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or not math.isclose(
                    float(rate), expected_rate, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                errors.append(
                    f"results periodic block {parameter}.acceptance_rate drifted"
                )
    total_attempts = int(per_block_attempts.sum())
    total_acceptances = int(per_block_acceptances.sum())
    expected_counts = {
        "n_periodic_uniform_independence_attempts": total_attempts,
        "n_periodic_uniform_independence_acceptances": total_acceptances,
        "n_likelihood_evaluations_periodic_uniform_independence": total_attempts,
        "n_likelihood_evaluations_periodic_uniform_independence_waveform_rebuild": (
            2 * replacements
        ),
        "n_likelihood_evaluations_periodic_uniform_independence_cache_hit": (
            replacements
        ),
    }
    for name, expected in expected_counts.items():
        if results.get(name) != expected:
            errors.append(f"results.{name}: {results.get(name)!r} != {expected!r}")
    aggregate_rate = results.get("periodic_uniform_independence_acceptance_rate")
    expected_rate = total_acceptances / total_attempts
    if (
        isinstance(aggregate_rate, bool)
        or not isinstance(aggregate_rate, (int, float))
        or not math.isfinite(float(aggregate_rate))
        or not math.isclose(
            float(aggregate_rate), expected_rate, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        errors.append("results.periodic_uniform_independence_acceptance_rate drifted")
    return errors


def _validate_complementary_de_telemetry(
    results: Mapping[str, Any],
    *,
    iterations: int,
    attempts_per_replacement: int,
) -> list[str]:
    """Validate a cDE-family production history and donor-policy proof."""

    errors: list[str] = []
    replacements = iterations * N_DELETE
    total_attempts = replacements * attempts_per_replacement
    strict_canonical_shapes = attempts_per_replacement == 8

    attempts = np.asarray(results.get("complementary_de_attempts_history"))
    acceptances = np.asarray(results.get("complementary_de_acceptances_history"))
    attempts_by_block = np.asarray(
        results.get("complementary_de_attempts_by_block_history")
    )
    acceptances_by_block = np.asarray(
        results.get("complementary_de_acceptances_by_block_history")
    )
    violations = np.asarray(
        results.get("complementary_de_donor_policy_violations_history")
    )
    acceptances_by_attempt = np.asarray(
        results.get("complementary_de_acceptances_by_attempt_history")
    )
    donors = np.asarray(
        results.get("complementary_de_donor_indices_by_attempt_history")
    )
    violations_by_attempt = np.asarray(
        results.get("complementary_de_donor_policy_violations_by_attempt_history")
    )
    complement_sizes = np.asarray(
        results.get("complementary_de_complement_size_history")
    )
    parent_indices = np.asarray(results.get("complementary_de_parent_index_history"))
    positions_before = np.asarray(
        results.get("complementary_de_position_before_by_attempt_history")
    )
    proposals = np.asarray(
        results.get("complementary_de_proposal_position_by_attempt_history")
    )

    scalar_histories_valid = (
        attempts.size == replacements
        and (not strict_canonical_shapes or attempts.shape == (replacements,))
        and acceptances.shape == attempts.shape
        and violations.shape == attempts.shape
        and complement_sizes.shape == attempts.shape
        and parent_indices.shape == attempts.shape
        and attempts_by_block.shape == attempts.shape + (1,)
        and acceptances_by_block.shape == attempts.shape + (1,)
        and np.issubdtype(attempts.dtype, np.integer)
        and np.issubdtype(acceptances.dtype, np.integer)
        and (
            np.issubdtype(violations.dtype, np.integer)
            or np.issubdtype(violations.dtype, np.bool_)
        )
        and np.issubdtype(complement_sizes.dtype, np.integer)
        and np.issubdtype(parent_indices.dtype, np.integer)
        and np.issubdtype(attempts_by_block.dtype, np.integer)
        and np.issubdtype(acceptances_by_block.dtype, np.integer)
        and np.all(attempts == attempts_per_replacement)
        and np.all((0 <= acceptances) & (acceptances <= attempts_per_replacement))
        and np.all(violations == 0)
        and np.all((447 <= complement_sizes) & (complement_sizes < N_LIVE))
        and np.all((0 <= parent_indices) & (parent_indices < N_LIVE))
        and np.array_equal(attempts_by_block[..., 0], attempts)
        and np.array_equal(acceptances_by_block[..., 0], acceptances)
    )
    if not scalar_histories_valid:
        errors.append("results.complementary_de scalar histories are invalid")

    per_attempt_valid = (
        acceptances_by_attempt.size == total_attempts
        and (
            not strict_canonical_shapes
            or acceptances_by_attempt.shape == (replacements, attempts_per_replacement)
        )
        and violations_by_attempt.shape == acceptances_by_attempt.shape
        and (
            np.issubdtype(acceptances_by_attempt.dtype, np.integer)
            or np.issubdtype(acceptances_by_attempt.dtype, np.bool_)
        )
        and (
            np.issubdtype(violations_by_attempt.dtype, np.integer)
            or np.issubdtype(violations_by_attempt.dtype, np.bool_)
        )
        and np.all((acceptances_by_attempt == 0) | (acceptances_by_attempt == 1))
        and np.all(violations_by_attempt == 0)
    )
    if violations_by_attempt.size and np.any(violations_by_attempt != 0):
        errors.append("results.complementary_de donor-violation histories are nonzero")
    if per_attempt_valid:
        acceptance_matrix = acceptances_by_attempt.reshape(
            replacements, attempts_per_replacement
        )
        violation_matrix = violations_by_attempt.reshape(
            replacements, attempts_per_replacement
        )
        if scalar_histories_valid and not np.array_equal(
            acceptance_matrix.sum(axis=1), acceptances.reshape(-1)
        ):
            errors.append(
                "results.complementary_de acceptance histories are inconsistent"
            )
        if scalar_histories_valid and not np.array_equal(
            violation_matrix.sum(axis=1), violations.reshape(-1)
        ):
            errors.append(
                "results.complementary_de donor-violation histories are inconsistent"
            )
    else:
        errors.append("results.complementary_de per-attempt histories are invalid")

    donor_history_valid = (
        donors.size == total_attempts * 2
        and (
            not strict_canonical_shapes
            or donors.shape == (replacements, attempts_per_replacement, 2)
        )
        and np.issubdtype(donors.dtype, np.integer)
    )
    if donor_history_valid:
        donor_pairs = donors.reshape(replacements, attempts_per_replacement, 2)
        donor_history_valid = bool(
            np.all((0 <= donor_pairs) & (donor_pairs < N_LIVE))
            and np.all(donor_pairs[..., 0] != donor_pairs[..., 1])
            and scalar_histories_valid
            and np.all(donor_pairs != parent_indices.reshape(replacements, 1, 1))
        )
    if not donor_history_valid:
        errors.append("results.complementary_de donor histories are invalid")

    parameter_names = results.get("complementary_de_sampling_parameter_names")
    positions_valid = (
        isinstance(parameter_names, list)
        and len(parameter_names) == 15
        and len(set(parameter_names)) == 15
        and set(FAST_RIDGE_INTRINSIC_BLOCKS[0]).issubset(parameter_names)
        and positions_before.shape == proposals.shape
        and (
            not strict_canonical_shapes
            or positions_before.shape
            == (replacements, attempts_per_replacement, len(parameter_names))
        )
        and positions_before.size == total_attempts * len(parameter_names)
        and np.issubdtype(positions_before.dtype, np.floating)
        and np.issubdtype(proposals.dtype, np.floating)
        and np.all(np.isfinite(positions_before))
        and np.all(np.isfinite(proposals))
    )
    if not positions_valid:
        errors.append("results.complementary_de position histories are invalid")

    total_acceptances = (
        int(acceptances_by_attempt.astype(np.int64).sum())
        if per_attempt_valid
        else None
    )
    expected_counts = {
        "n_likelihood_evaluations_complementary_de": total_attempts,
        "n_likelihood_evaluations_complementary_de_waveform_rebuild": total_attempts,
        "n_likelihood_evaluations_complementary_de_cache_hit": 0,
        "n_complementary_de_attempts": total_attempts,
        "n_complementary_de_acceptances": total_acceptances,
        "n_complementary_de_donor_policy_violations": 0,
    }
    for name, expected in expected_counts.items():
        if results.get(name) != expected:
            errors.append(f"results.{name}: {results.get(name)!r} != {expected!r}")

    rate = results.get("complementary_de_acceptance_rate")
    expected_rate = (
        total_acceptances / total_attempts if total_acceptances is not None else None
    )
    if (
        expected_rate is None
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or not math.isclose(float(rate), expected_rate, rel_tol=0.0, abs_tol=1e-12)
    ):
        errors.append("results.complementary_de_acceptance_rate drifted")

    blocks = results.get("complementary_de_blocks")
    if not isinstance(blocks, list) or len(blocks) != 1:
        errors.append("results.complementary_de_blocks is invalid")
    else:
        block = blocks[0]
        expected_block = {
            "parameters": list(FAST_RIDGE_INTRINSIC_BLOCKS[0]),
            "requires_waveform_rebuild": True,
            "attempts_per_replacement": attempts_per_replacement,
            "n_attempts": total_attempts,
            "n_acceptances": total_acceptances,
            "acceptance_rate": expected_rate,
            "gamma": 1.0,
            "placement": "after-target-block",
            "n_donor_policy_violations": 0,
        }
        if not isinstance(block, Mapping):
            errors.append("results complementary-DE block is not an object")
        else:
            for name, expected in expected_block.items():
                actual = block.get(name)
                if isinstance(expected, float):
                    matched = (
                        not isinstance(actual, bool)
                        and isinstance(actual, (int, float))
                        and math.isfinite(float(actual))
                        and math.isclose(
                            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                        )
                    )
                else:
                    matched = actual == expected
                if not matched:
                    errors.append(
                        f"results complementary-DE block.{name}: "
                        f"{actual!r} != {expected!r}"
                    )
    return errors


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_gap_quantiles(value: Any, *, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping) or set(value) != {"p05", "p50", "p95"}:
        return [f"{path} must contain exactly p05, p50, and p95"]
    quantiles = [value[name] for name in ("p05", "p50", "p95")]
    if any(not _is_finite_number(item) or float(item) < 0.0 for item in quantiles):
        return [f"{path} values must be finite and nonnegative"]
    if not float(quantiles[0]) <= float(quantiles[1]) <= float(quantiles[2]):
        return [f"{path} values are not ordered"]
    return []


def _validate_netsky_report_semantics(report: Mapping[str, Any]) -> list[str]:
    """Validate NETSKY configuration, telemetry, and accounting semantics."""

    errors: list[str] = []
    config_value = report.get("config")
    config = config_value if isinstance(config_value, Mapping) else {}
    results_value = report.get("results")
    results = results_value if isinstance(results_value, Mapping) else {}
    fold_config_value = config.get("fold_symmetry")
    fold_config = fold_config_value if isinstance(fold_config_value, Mapping) else {}
    expected_fold_names = {
        "cos_iota": "cos_iota",
        "azimuth": "azimuth",
        "psi": "psi",
    }
    if set(fold_config) != {*expected_fold_names, "azimuth_reflection_center"}:
        errors.append("config.fold_symmetry is not a completed fold configuration")
    else:
        for name, expected in expected_fold_names.items():
            if fold_config.get(name) != expected:
                errors.append(
                    f"config.fold_symmetry.{name}: "
                    f"{fold_config.get(name)!r} != {expected!r}"
                )
        center = fold_config.get("azimuth_reflection_center")
        if not _is_finite_number(center):
            errors.append(
                "config.fold_symmetry.azimuth_reflection_center is not finite"
            )

    posterior_artifact = results.get("posterior_artifact")
    if not isinstance(posterior_artifact, Mapping):
        errors.append("results.posterior_artifact is not an object")
        posterior_artifact = {}
    posterior_count = results.get("posterior_samples")
    if (
        isinstance(posterior_count, bool)
        or not isinstance(posterior_count, int)
        or posterior_count <= 0
    ):
        errors.append("results.posterior_samples is not a positive integer")
    elif posterior_artifact.get("count") != posterior_count:
        errors.append(
            "results.posterior_samples does not match posterior_artifact.count"
        )

    effective_size = results.get("posterior_weight_effective_size")
    if not _is_finite_number(effective_size) or float(effective_size) <= 0.0:
        errors.append(
            "results.posterior_weight_effective_size is not positive and finite"
        )

    quotient_value = results.get("quotient_fold")
    if not isinstance(quotient_value, Mapping):
        errors.append("results.quotient_fold is not an object")
        return errors
    quotient = quotient_value
    if not _same_json(quotient.get("completed_config"), fold_config):
        errors.append("results.quotient_fold.completed_config does not match config")
    model_limitation = quotient.get("model_limitation")
    limitations = report.get("limitations")
    if (
        not isinstance(model_limitation, str)
        or not model_limitation
        or not isinstance(limitations, list)
        or model_limitation not in limitations
    ):
        errors.append("results.quotient_fold.model_limitation is not preserved")
    batch_size = quotient.get("batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        errors.append("results.quotient_fold.batch_size is not a positive integer")
    if quotient.get("group_order") != 8:
        errors.append("results.quotient_fold.group_order must be 8")

    folded_metadata = results.get("folded_nested_diagnostics")
    folded_count = (
        folded_metadata.get("count") if isinstance(folded_metadata, Mapping) else None
    )
    folded_points = quotient.get("folded_points")
    if (
        isinstance(folded_points, bool)
        or not isinstance(folded_points, int)
        or folded_points < 1
        or isinstance(folded_count, bool)
        or not isinstance(folded_count, int)
        or folded_points > folded_count
    ):
        errors.append(
            "results.quotient_fold.folded_points is not a valid retained subset"
        )
    elif (
        isinstance(posterior_count, int)
        and not isinstance(posterior_count, bool)
        and not folded_points <= posterior_count <= 8 * folded_points
    ):
        errors.append(
            "expanded posterior count must lie between one and eight supported "
            "images per retained folded point"
        )

    entropy = quotient.get("normalized_conditional_image_entropy")
    if (
        not _is_finite_number(entropy)
        or not -1.0e-12 <= float(entropy) <= 1.0 + 1.0e-12
    ):
        errors.append("normalized conditional image entropy is outside [0, 1]")
    masses = quotient.get("image_sector_posterior_masses")
    valid_masses = (
        isinstance(masses, list)
        and len(masses) == 8
        and all(_is_finite_number(value) and float(value) >= 0.0 for value in masses)
    )
    if not valid_masses:
        errors.append("image-sector posterior masses are invalid")
    else:
        mass_values = [float(value) for value in masses]
        if not math.isclose(sum(mass_values), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
            errors.append("image-sector posterior masses are not normalized")
        expected_nonidentity = quotient.get("expected_nonidentity_mass")
        if not _is_finite_number(expected_nonidentity) or not math.isclose(
            float(expected_nonidentity),
            1.0 - mass_values[0],
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            errors.append("expected non-identity mass does not match sector masses")
    zero_support = quotient.get("zero_support_image_fraction")
    if not _is_finite_number(zero_support) or not 0.0 <= float(zero_support) <= 1.0:
        errors.append("zero-support image fraction is outside [0, 1]")

    gaps_value = quotient.get("supported_image_log_likelihood_gaps")
    if not isinstance(gaps_value, Mapping) or set(gaps_value) != {
        "within_orbit_span_weighted_quantiles",
        "identity_absolute_gap_weighted_quantiles",
    }:
        errors.append("supported-image likelihood-gap telemetry is invalid")
    else:
        errors.extend(
            _validate_gap_quantiles(
                gaps_value["within_orbit_span_weighted_quantiles"],
                path=(
                    "results.quotient_fold.supported_image_log_likelihood_gaps."
                    "within_orbit_span_weighted_quantiles"
                ),
            )
        )
        errors.extend(
            _validate_gap_quantiles(
                gaps_value["identity_absolute_gap_weighted_quantiles"],
                path=(
                    "results.quotient_fold.supported_image_log_likelihood_gaps."
                    "identity_absolute_gap_weighted_quantiles"
                ),
            )
        )

    physical_callbacks = results.get("n_likelihood_evaluations_physical")
    accounting_value = quotient.get("projection_accounting")
    if not isinstance(accounting_value, Mapping):
        errors.append("results.quotient_fold.projection_accounting is not an object")
    elif (
        isinstance(physical_callbacks, bool)
        or not isinstance(physical_callbacks, int)
        or physical_callbacks < 0
        or isinstance(folded_points, bool)
        or not isinstance(folded_points, int)
        or folded_points < 0
    ):
        errors.append("NETSKY projection inputs are not nonnegative integers")
    else:
        expected_accounting = {
            "images_per_folded_target_callback": 8,
            "sampler_folded_target_callbacks": physical_callbacks,
            "sampler_true_image_projections": 8 * physical_callbacks,
            "retained_folded_points_unfolded": folded_points,
            "unfold_true_image_projections": 8 * folded_points,
            "total_true_image_projections": 8 * (physical_callbacks + folded_points),
            "sampler_callback_counter": ("results.n_likelihood_evaluations_physical"),
        }
        if not _same_json(accounting_value, expected_accounting):
            errors.append("results.quotient_fold.projection_accounting drifted")

    timing_value = report.get("timing_seconds")
    timing = timing_value if isinstance(timing_value, Mapping) else {}
    postprocessing = timing.get("fold_unfold_postprocessing")
    extraction = timing.get("result_extraction")
    if not _is_finite_number(postprocessing) or float(postprocessing) <= 0.0:
        errors.append("fold/unfold postprocessing time is not positive and finite")
    if not _is_finite_number(extraction) or (
        _is_finite_number(postprocessing) and float(extraction) < float(postprocessing)
    ):
        errors.append("result extraction time does not contain fold/unfold time")
    return errors


def _validate_arm_report(
    report: Mapping[str, Any],
    arm: LikelihoodArm,
    mode: str,
    *,
    data_sha256: str,
    reference_sha256: str,
    source_sha256: str,
    simulate_cpu: bool,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "benchmark": _benchmark_name(),
        "timing_only": False,
        "ablation.name": FSM_VARIANT,
        "ablation.topology": "replicated-live",
        "ablation.interval": "cached-stepping-out",
        "ablation.scheduler": "fsm",
        "ablation.direction_parameter": "covariance",
        "implementation.label": arm.label,
        "devices.requested_count": N_DEVICES,
        "devices.local_count": N_DEVICES,
        "data.sha256": data_sha256,
        "data.manifest.workload": WORKLOAD,
        "data.manifest.detectors": list(DETECTORS),
        "data.manifest.duration_seconds": DURATION_SECONDS,
        "data.manifest.f_min_hz": F_MIN_HZ,
        "data.manifest.nominal_f_max_hz": NOMINAL_F_MAX_HZ,
        "data.manifest.likelihood_f_max_hz": LIKELIHOOD_F_MAX_HZ,
        "data.manifest.analysis_strain_product": "LOSC_CLN_16_V1",
        "data.manifest.gwosc_sample_rate_hz": 16384,
        "config.workload": WORKLOAD,
        "config.blocking_scheme": BLOCKING_SCHEME,
        "config.waveform": WAVEFORM,
        "config.waveform_f_ref_hz": WAVEFORM_F_REF_HZ,
        "config.carrier_time_anchor": CARRIER_TIME_ANCHOR,
        "config.sampled_dimensions": 15,
        "config.blocks": [list(block) for block in PAPER_BLOCKS],
        "config.priors.M_c.range": [1.18, 1.21],
        "config.priors.q.range": [0.125, 1.0],
        "config.priors.d_L.range_mpc": [1.0, 75.0],
        "config.phase_marginalization": True,
        "config.time_marginalization_tc_range_seconds": list(TC_RANGE_SECONDS),
        "config.time_marginalization_fft_sample_rate_hz": (
            TIME_MARGINALIZATION_FFT_SAMPLE_RATE_HZ
        ),
        "config.distance_marginalization": False,
        "config.n_devices": N_DEVICES,
        "config.n_live": N_LIVE,
        "config.n_delete": N_DELETE,
        "config.num_inner_steps_per_dim": NUM_INNER_STEPS_PER_DIM,
        "config.num_gibbs_sweeps": NUM_GIBBS_SWEEPS,
        "config.termination_dlogz": TERMINATION_DLOGZ,
        "config.dtype": "float64",
        "config.paper_notation.D_devices": N_DEVICES,
        "config.paper_notation.M_gibbs_sweeps": NUM_GIBBS_SWEEPS,
        "paired_likelihood_referee.workload": WORKLOAD,
        "paired_likelihood_referee.blocking_scheme": BLOCKING_SCHEME,
        "paired_likelihood_referee.arm_label": arm.label,
        "paired_likelihood_referee.likelihood_kind": arm.kind,
        "paired_likelihood_referee.n_devices": N_DEVICES,
        "paired_likelihood_referee.mode": mode,
        "paired_likelihood_referee.source_tree_sha256": source_sha256,
        "paired_likelihood_referee.data_sha256": data_sha256,
        "paired_likelihood_referee.reference_json_sha256": reference_sha256,
        "paired_likelihood_referee.proposal.direction_parameter": "covariance",
        "paired_likelihood_referee.proposal.direction_mode": DIRECTION_MODE,
        "paired_likelihood_referee.proposal.num_de_jumps": 0,
        "paired_likelihood_referee.proposal.de_jump_blocks": [],
        "paired_likelihood_referee.paper_notation.D_devices": N_DEVICES,
        "paired_likelihood_referee.paper_notation.M_gibbs_sweeps": (NUM_GIBBS_SWEEPS),
        "simulated_cpu": simulate_cpu,
        **{f"results.{counter}": 0 for counter in PRIMARY_DE_COUNTERS},
    }
    if BLOCKING_SCHEME == NETSKY_SCHEME:
        expected.update(
            {
                "config.bridge_blocks": [list(block) for block in NETSKY_BRIDGE_BLOCKS],
                "config.periodic_wrapped_covariance": True,
                "config.fixed_work": dict(NETSKY_FIXED_WORK),
                "paired_likelihood_referee.proposal.bridge_blocks": [
                    list(block) for block in NETSKY_BRIDGE_BLOCKS
                ],
                "paired_likelihood_referee.proposal.periodic_wrapped_covariance": (
                    True
                ),
                "paired_likelihood_referee.proposal.fixed_work": dict(
                    NETSKY_FIXED_WORK
                ),
                "results.posterior_artifact.space": "prior",
                "results.posterior_artifact.weighting": (UNFOLDED_POSTERIOR_WEIGHTING),
                "results.folded_nested_diagnostics.space": (
                    "folded sampling-space target"
                ),
                "results.folded_nested_diagnostics.weighting": (
                    "not applicable: folded nested-sampling contours"
                ),
                "results.folded_nested_diagnostics.semantics": (
                    FOLDED_TARGET_SEMANTICS
                ),
                "results.posterior_weight_effective_size_semantics": (
                    POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS
                ),
            }
        )
    else:
        expected.update(
            {
                "results.nested_artifact.space": "prior",
                "results.nested_artifact.weighting": (
                    "normalized nested-sampling log weights"
                ),
            }
        )
    if DIRECTION_MODE != "covariance":
        expected["config.direction_mode"] = DIRECTION_MODE
    if BLOCKING_SCHEME == FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME:
        expected["config.num_slice_steps_by_block"] = list(
            FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE
        )
        expected["paired_likelihood_referee.proposal.num_slice_steps_by_block"] = list(
            FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE
        )
    if BLOCKING_SCHEME in PERIODIC_MH_BLOCKING_SCHEMES:
        expected["config.block_kernel_modes"] = list(
            FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
        )
        expected["config.fixed_work"] = dict(
            FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK
        )
        expected["paired_likelihood_referee.proposal.block_kernel_modes"] = list(
            FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
        )
        expected["paired_likelihood_referee.proposal.fixed_work"] = dict(
            FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK
        )
    if BLOCKING_SCHEME in COMPLEMENTARY_DE_BLOCK_BY_SCHEME:
        complementary_de_block = COMPLEMENTARY_DE_BLOCK_BY_SCHEME[BLOCKING_SCHEME]
        complementary_de_jump_block = {
            "parameters": list(complementary_de_block["parameters"]),
            "attempts": complementary_de_block["attempts"],
        }
        expected["config.block_kernel_modes"] = list(
            FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
        )
        expected["config.complementary_de_jump_block"] = complementary_de_jump_block
        expected["config.fixed_work"] = dict(
            COMPLEMENTARY_DE_FIXED_WORK_BY_SCHEME[BLOCKING_SCHEME]
        )
        expected["paired_likelihood_referee.proposal.block_kernel_modes"] = list(
            FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
        )
        expected["paired_likelihood_referee.proposal.complementary_de_jump_block"] = (
            complementary_de_jump_block
        )
        expected["paired_likelihood_referee.proposal.fixed_work"] = dict(
            COMPLEMENTARY_DE_FIXED_WORK_BY_SCHEME[BLOCKING_SCHEME]
        )
    for path, expected_value in expected.items():
        try:
            actual = _get_path(report, path)
        except (KeyError, TypeError):
            errors.append(f"missing {path}")
            continue
        if not _same_json(actual, expected_value):
            errors.append(f"{path}: {actual!r} != {expected_value!r}")

    for counter in CLASSIFIED_DE_COUNTERS:
        try:
            actual = _get_path(report, f"results.{counter}")
        except (KeyError, TypeError):
            errors.append(f"missing results.{counter}")
            continue
        if actual not in (None, 0):
            errors.append(f"results.{counter}: {actual!r} is neither None nor 0")

    config = report.get("config")
    if isinstance(config, Mapping):
        if config.get("sha256") != _config_sha256(config):
            errors.append("config.sha256 does not match the canonical config")
        errors.extend(
            _validate_likelihood_metadata(
                config.get("likelihood"),
                arm,
                reference_sha256,
            )
        )
    else:
        errors.append("missing config")

    early = report.get("results", {}).get("early_stopped_for_cache_probe")
    iterations = report.get("results", {}).get("n_iterations")
    evaluations = report.get("results", {}).get("n_likelihood_evaluations")
    log_z = report.get("results", {}).get("log_Z")
    log_z_error = report.get("results", {}).get("log_Z_error")
    results_value = report.get("results", {})
    results = results_value if isinstance(results_value, Mapping) else {}
    nested_artifact_name = (
        "folded_nested_diagnostics"
        if BLOCKING_SCHEME == NETSKY_SCHEME
        else "nested_artifact"
    )
    nested_artifact = results.get(nested_artifact_name, {})
    nested_count = (
        nested_artifact.get("count") if isinstance(nested_artifact, Mapping) else None
    )
    outer_steps = (
        report.get("timing_seconds", {}).get("outer_step", {}).get("outer_steps")
    )
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        errors.append(f"results.n_iterations is not a positive integer: {iterations!r}")
    if (
        isinstance(evaluations, bool)
        or not isinstance(evaluations, int)
        or evaluations <= 0
    ):
        errors.append(
            "results.n_likelihood_evaluations is not a positive integer: "
            f"{evaluations!r}"
        )
    if (
        isinstance(nested_count, bool)
        or not isinstance(nested_count, int)
        or nested_count <= 0
    ):
        errors.append(
            f"results.{nested_artifact_name}.count is not a positive integer: "
            f"{nested_count!r}"
        )
    elif isinstance(iterations, int) and not isinstance(iterations, bool):
        expected_nested_count = N_LIVE + iterations * N_DELETE
        if nested_count != expected_nested_count:
            errors.append(
                f"results.{nested_artifact_name}.count {nested_count} != "
                f"n_live + n_iterations*n_delete ({expected_nested_count})"
            )
        if (
            isinstance(evaluations, int)
            and not isinstance(evaluations, bool)
            and evaluations < nested_count
        ):
            errors.append(
                f"results.n_likelihood_evaluations {evaluations} is below "
                f"nested point count {nested_count}"
            )
    if BLOCKING_SCHEME == NETSKY_SCHEME:
        if results.get("nested_artifact") is not None:
            errors.append("results.nested_artifact is legacy-only for NETSKY")
        errors.extend(_validate_netsky_report_semantics(report))
    if (
        not isinstance(log_z, (int, float))
        or isinstance(log_z, bool)
        or not math.isfinite(float(log_z))
    ):
        errors.append(f"results.log_Z is not finite: {log_z!r}")
    if (
        not isinstance(log_z_error, (int, float))
        or isinstance(log_z_error, bool)
        or not math.isfinite(float(log_z_error))
        or float(log_z_error) <= 0.0
    ):
        errors.append(
            f"results.log_Z_error is not positive and finite: {log_z_error!r}"
        )
    if (
        isinstance(outer_steps, bool)
        or not isinstance(outer_steps, int)
        or outer_steps <= 0
    ):
        errors.append(
            f"timing_seconds.outer_step.outer_steps is not positive: {outer_steps!r}"
        )
    elif isinstance(iterations, int) and outer_steps != iterations:
        errors.append(f"outer-step count {outer_steps} != n_iterations {iterations}")
    if (
        BLOCKING_SCHEME in PERIODIC_MH_BLOCKING_SCHEMES
        and isinstance(iterations, int)
        and not isinstance(iterations, bool)
        and iterations > 0
    ):
        results = report.get("results", {})
        replacements = iterations * N_DELETE
        expected_h4_counts = {
            "n_slice_updates": 12 * NUM_GIBBS_SWEEPS * replacements,
            "n_periodic_uniform_independence_attempts": (
                3 * NUM_GIBBS_SWEEPS * replacements
            ),
            "n_likelihood_evaluations_periodic_uniform_independence": (
                3 * NUM_GIBBS_SWEEPS * replacements
            ),
        }
        for name, expected_count in expected_h4_counts.items():
            actual = results.get(name)
            if actual != expected_count:
                errors.append(f"results.{name}: {actual!r} != {expected_count!r}")
        acceptances = results.get("n_periodic_uniform_independence_acceptances")
        attempts = expected_h4_counts["n_periodic_uniform_independence_attempts"]
        if (
            isinstance(acceptances, bool)
            or not isinstance(acceptances, int)
            or not 0 <= acceptances <= attempts
        ):
            errors.append(
                "results.n_periodic_uniform_independence_acceptances is outside "
                f"[0, {attempts}]: {acceptances!r}"
            )
        rate = results.get("periodic_uniform_independence_acceptance_rate")
        expected_rate = (
            acceptances / attempts
            if isinstance(acceptances, int) and not isinstance(acceptances, bool)
            else None
        )
        if (
            expected_rate is None
            or isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(float(rate))
            or not math.isclose(
                float(rate), expected_rate, rel_tol=0.0, abs_tol=1.0e-12
            )
        ):
            errors.append(
                "results.periodic_uniform_independence_acceptance_rate does not "
                f"match attempts/acceptances: {rate!r}"
            )
        physical = results.get("n_likelihood_evaluations_physical")
        slice_updates = expected_h4_counts["n_slice_updates"]
        expected_physical = (
            evaluations + 2 * slice_updates
            if isinstance(evaluations, int) and not isinstance(evaluations, bool)
            else None
        )
        if physical != expected_physical:
            errors.append(
                "results.n_likelihood_evaluations_physical: "
                f"{physical!r} != {expected_physical!r}"
            )
        if isinstance(results, Mapping):
            errors.extend(
                _validate_periodic_independence_telemetry(
                    results,
                    iterations=iterations,
                )
            )
            if BLOCKING_SCHEME in COMPLEMENTARY_DE_BLOCK_BY_SCHEME:
                attempts_per_replacement = int(
                    COMPLEMENTARY_DE_BLOCK_BY_SCHEME[BLOCKING_SCHEME]["attempts"]
                )
                expected_cde_attempts = attempts_per_replacement * replacements
                for name, expected_count in {
                    "n_likelihood_evaluations_complementary_de": (
                        expected_cde_attempts
                    ),
                    "n_likelihood_evaluations_complementary_de_waveform_rebuild": (
                        expected_cde_attempts
                    ),
                    "n_likelihood_evaluations_complementary_de_cache_hit": 0,
                    "n_complementary_de_attempts": expected_cde_attempts,
                    "n_complementary_de_donor_policy_violations": 0,
                }.items():
                    if results.get(name) != expected_count:
                        errors.append(
                            f"results.{name}: {results.get(name)!r} != "
                            f"{expected_count!r}"
                        )
                errors.extend(
                    _validate_complementary_de_telemetry(
                        results,
                        iterations=iterations,
                        attempts_per_replacement=attempts_per_replacement,
                    )
                )
        else:
            errors.append("results is not an object")
    if mode == "prefix":
        if (
            early is not True
            or iterations != PREFIX_STEPS
            or outer_steps != PREFIX_STEPS
        ):
            errors.append(
                "prefix cell must stop after exactly five outer steps; "
                f"got early={early!r}, iterations={iterations!r}, "
                f"outer_steps={outer_steps!r}"
            )
    elif early is not False:
        errors.append(f"full cell unexpectedly reports early stop: {early!r}")
    return errors


def _validate_netsky_posterior_artifact(
    path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if not path.is_file():
        return {"passed": False, "errors": [f"missing weighted artifact: {path}"]}
    digest = _sha256(path)
    artifact_value = report.get("results", {}).get("posterior_artifact", {})
    artifact = artifact_value if isinstance(artifact_value, Mapping) else {}
    if artifact is not artifact_value:
        errors.append("results.posterior_artifact is not an object")
    expected_metadata = {
        "path": str(path.expanduser().resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "format": "npz",
        "space": "prior",
        "weighting": UNFOLDED_POSTERIOR_WEIGHTING,
        "schema_version": 2,
        "weight_effective_size_semantics": (POSTERIOR_WEIGHT_EFFECTIVE_SIZE_SEMANTICS),
    }
    for name, expected in expected_metadata.items():
        if artifact.get(name) != expected:
            errors.append(
                f"posterior artifact report {name}={artifact.get(name)!r}, "
                f"expected {expected!r}"
            )
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = tuple(archive.files)
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        return {"passed": False, "sha256": digest, "errors": [str(error)]}
    expected_fields = {*POSITION_FIELDS, "log_likelihood", "log_weights"}
    if set(fields) != expected_fields:
        errors.append(
            f"posterior fields {sorted(fields)!r} != {sorted(expected_fields)!r}"
        )
    if "log_likelihood_birth" in arrays:
        errors.append("unfolded posterior must not contain log_likelihood_birth")
    if artifact.get("fields") != list(fields):
        errors.append("posterior artifact report fields do not match the NPZ")
    shapes = {name: values.shape for name, values in arrays.items()}
    if any(len(shape) != 1 for shape in shapes.values()):
        errors.append(f"posterior arrays are not all one-dimensional: {shapes!r}")
    counts = {shape[0] for shape in shapes.values() if len(shape) == 1}
    count = next(iter(counts)) if len(counts) == 1 else None
    if count is None or count <= 0:
        errors.append(f"posterior arrays do not share a positive length: {shapes!r}")
    elif artifact.get("count") != count:
        errors.append(
            f"posterior artifact report count {artifact.get('count')!r} != {count}"
        )
    actual_dtypes = {name: str(values.dtype) for name, values in arrays.items()}
    if artifact.get("dtypes") != actual_dtypes:
        errors.append(
            f"posterior artifact report dtypes {artifact.get('dtypes')!r} != "
            f"{actual_dtypes!r}"
        )
    invalid_numeric = [
        name
        for name, values in arrays.items()
        if not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ]
    if invalid_numeric:
        errors.append(f"posterior artifact has non-real fields: {invalid_numeric}")
    if expected_fields.issubset(arrays) and count is not None and count > 0:
        invalid_finite = [
            name
            for name in sorted(expected_fields)
            if not np.all(np.isfinite(arrays[name]))
        ]
        if invalid_finite:
            errors.append(
                "posterior artifact has non-finite physical/likelihood/weight "
                f"fields: {invalid_finite}"
            )
        log_weights = arrays["log_weights"]
        if np.all(np.isfinite(log_weights)):
            maximum = float(np.max(log_weights))
            weights = np.exp(log_weights - maximum)
            total = math.exp(maximum) * float(weights.sum())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-8):
                errors.append(
                    f"unfolded posterior log weights are not normalized: sum={total}"
                )
            else:
                normalized = math.exp(maximum) * weights
                effective_size = float(1.0 / np.sum(normalized**2))
                reported = report.get("results", {}).get(
                    "posterior_weight_effective_size"
                )
                if not _is_finite_number(reported) or not math.isclose(
                    float(reported),
                    effective_size,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-10,
                ):
                    errors.append(
                        "posterior weight effective size does not match log weights"
                    )
        q = arrays["q"]
        distance = arrays["d_L"]
        if np.any((q < 0.125) | (q > 1.0)):
            errors.append("weighted q samples leave the fixed prior support")
        if np.any((distance < 1.0) | (distance > 75.0)):
            errors.append("weighted d_L samples leave the fixed prior support")
    return {
        "passed": not errors,
        "sha256": digest,
        "count": count,
        "fields": list(fields),
        "errors": errors,
    }


def _validate_folded_nested_diagnostics(
    path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if not path.is_file():
        return {
            "passed": False,
            "errors": [f"missing folded nested diagnostics: {path}"],
        }
    digest = _sha256(path)
    metadata_value = report.get("results", {}).get("folded_nested_diagnostics", {})
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    if metadata is not metadata_value:
        errors.append("results.folded_nested_diagnostics is not an object")
    expected_fields = ("log_likelihood", "log_likelihood_birth")
    expected_metadata = {
        "path": str(path.expanduser().resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "format": "npz",
        "space": "folded sampling-space target",
        "weighting": "not applicable: folded nested-sampling contours",
        "fields": list(expected_fields),
        "semantics": FOLDED_TARGET_SEMANTICS,
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            errors.append(
                f"folded diagnostic report {name}={metadata.get(name)!r}, "
                f"expected {expected!r}"
            )
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = tuple(archive.files)
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        return {"passed": False, "sha256": digest, "errors": [str(error)]}
    if fields != expected_fields:
        errors.append(f"folded fields {list(fields)!r} != {list(expected_fields)!r}")
    shapes = {name: values.shape for name, values in arrays.items()}
    if any(len(shape) != 1 for shape in shapes.values()):
        errors.append(f"folded arrays are not one-dimensional: {shapes!r}")
    counts = {shape[0] for shape in shapes.values() if len(shape) == 1}
    count = next(iter(counts)) if len(counts) == 1 else None
    if count is None or count <= 0:
        errors.append(f"folded arrays do not share a positive length: {shapes!r}")
    else:
        if metadata.get("count") != count:
            errors.append(
                f"folded diagnostic report count {metadata.get('count')!r} != {count}"
            )
        iterations = report.get("results", {}).get("n_iterations")
        if isinstance(iterations, int) and not isinstance(iterations, bool):
            expected_count = N_LIVE + iterations * N_DELETE
            if count != expected_count:
                errors.append(
                    f"folded diagnostic count {count} != "
                    f"n_live + n_iterations*n_delete ({expected_count})"
                )
    actual_dtypes = {name: str(values.dtype) for name, values in arrays.items()}
    if metadata.get("dtypes") != actual_dtypes:
        errors.append(
            f"folded diagnostic report dtypes {metadata.get('dtypes')!r} != "
            f"{actual_dtypes!r}"
        )
    if set(expected_fields).issubset(arrays):
        death = arrays["log_likelihood"]
        birth = arrays["log_likelihood_birth"]
        if not np.issubdtype(death.dtype, np.number) or np.issubdtype(
            death.dtype, np.complexfloating
        ):
            errors.append("folded death likelihood is not a real numeric array")
        if not np.issubdtype(birth.dtype, np.number) or np.issubdtype(
            birth.dtype, np.complexfloating
        ):
            errors.append("folded birth likelihood is not a real numeric array")
        if not np.all(np.isfinite(death)):
            errors.append("folded death likelihoods contain non-finite values")
        if np.any(np.isnan(birth)) or np.any(np.isposinf(birth)):
            errors.append("folded birth likelihoods contain NaN or +inf")
        initial = np.isneginf(birth)
        if int(np.count_nonzero(initial)) != N_LIVE:
            errors.append(
                "folded diagnostics must contain exactly n_live initial -inf "
                "birth likelihoods"
            )
        elif not np.all(initial[:N_LIVE]) or np.any(initial[N_LIVE:]):
            errors.append("folded initial -inf births are not the first n_live rows")
        finite_birth = np.isfinite(birth)
        if np.any(death[finite_birth] <= birth[finite_birth]):
            errors.append(
                "folded birth likelihoods must be strictly below death likelihoods"
            )
        try:
            from jimgw.samplers.diagnostics import insertion_index_diagnostic

            expected_insertion = insertion_index_diagnostic(
                death,
                birth,
                n_live=N_LIVE,
            )
        except ValueError as error:
            errors.append(f"folded insertion-index inputs are invalid: {error}")
        else:
            stored = report.get("results", {}).get("insertion_index_diagnostic")
            if not isinstance(stored, Mapping) or set(stored) != set(
                expected_insertion
            ):
                errors.append("folded insertion-index diagnostic inventory is invalid")
            else:
                for name, expected in expected_insertion.items():
                    actual = stored[name]
                    if isinstance(expected, float):
                        matches = _is_finite_number(actual) and math.isclose(
                            float(actual),
                            expected,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-10,
                        )
                    else:
                        matches = actual == expected
                    if not matches:
                        errors.append(
                            f"folded insertion-index diagnostic mismatch for {name}"
                        )
    return {
        "passed": not errors,
        "sha256": digest,
        "count": count,
        "fields": list(fields),
        "errors": errors,
    }


def _validate_weighted_artifact(
    path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    if BLOCKING_SCHEME == NETSKY_SCHEME:
        return _validate_netsky_posterior_artifact(path, report)
    errors: list[str] = []
    if not path.is_file():
        return {"passed": False, "errors": [f"missing weighted artifact: {path}"]}
    digest = _sha256(path)
    nested_value = report.get("results", {}).get("nested_artifact", {})
    nested = nested_value if isinstance(nested_value, Mapping) else {}
    if nested is not nested_value:
        errors.append("results.nested_artifact is not an object")
    if nested.get("sha256") != digest:
        errors.append("weighted artifact SHA-256 does not match its report")
    expected_static_metadata = {
        "path": str(path.expanduser().resolve()),
        "bytes": path.stat().st_size,
        "format": "npz",
        "space": "prior",
        "weighting": "normalized nested-sampling log weights",
        "fields": list(WEIGHTED_FIELDS),
    }
    for name, expected in expected_static_metadata.items():
        if nested.get(name) != expected:
            errors.append(
                f"weighted artifact report {name}={nested.get(name)!r}, "
                f"expected {expected!r}"
            )
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = tuple(archive.files)
            if set(fields) != set(WEIGHTED_FIELDS):
                errors.append(
                    f"weighted fields {sorted(fields)!r} != {sorted(WEIGHTED_FIELDS)!r}"
                )
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        return {"passed": False, "sha256": digest, "errors": [str(error)]}
    shapes = {name: values.shape for name, values in arrays.items()}
    if any(len(shape) != 1 for shape in shapes.values()):
        errors.append(f"weighted arrays are not all one-dimensional: {shapes!r}")
    counts = {shape[0] for shape in shapes.values() if len(shape) == 1}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        errors.append(f"weighted arrays do not share a positive length: {shapes!r}")
    else:
        count = next(iter(counts))
        if nested.get("count") != count:
            errors.append(
                f"weighted artifact report count {nested.get('count')!r} != {count}"
            )
    actual_dtypes = {name: str(values.dtype) for name, values in arrays.items()}
    if nested.get("dtypes") != actual_dtypes:
        errors.append(
            f"weighted artifact report dtypes {nested.get('dtypes')!r} != "
            f"{actual_dtypes!r}"
        )
    if set(WEIGHTED_FIELDS).issubset(arrays):
        finite_fields = set(WEIGHTED_FIELDS) - {"log_likelihood_birth"}
        invalid = [
            name
            for name in sorted(finite_fields)
            if not np.all(np.isfinite(arrays[name]))
        ]
        if invalid:
            errors.append(f"weighted artifact has non-finite fields: {invalid}")
        births = arrays["log_likelihood_birth"]
        if np.any(np.isnan(births)) or np.any(np.isposinf(births)):
            errors.append("birth likelihoods contain NaN or +inf")
        if int(np.count_nonzero(np.isneginf(births))) != N_LIVE:
            errors.append(
                "weighted artifact must contain exactly n_live initial -inf "
                "birth likelihoods"
            )
        finite_births = np.isfinite(births)
        deaths = arrays["log_likelihood"]
        if np.any(deaths[finite_births] <= births[finite_births]):
            errors.append(
                "finite birth likelihoods must be strictly below death likelihoods"
            )
        log_weights = arrays["log_weights"]
        if np.all(np.isfinite(log_weights)):
            maximum = float(np.max(log_weights))
            total = math.exp(maximum) * float(np.exp(log_weights - maximum).sum())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
                errors.append(f"nested log weights are not normalized: sum={total}")
        q = arrays["q"]
        distance = arrays["d_L"]
        if np.any((q < 0.125) | (q > 1.0)):
            errors.append("weighted q samples leave the fixed prior support")
        if np.any((distance < 1.0) | (distance > 75.0)):
            errors.append("weighted d_L samples leave the fixed prior support")
    return {
        "passed": not errors,
        "sha256": digest,
        "count": next(iter(counts)) if len(counts) == 1 else None,
        "fields": list(fields),
        "errors": errors,
    }


def _scientific_config_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(report.get("config", {}))
    config.pop("sha256", None)
    config.pop("likelihood", None)
    return config


def compare_pair(
    reports: Mapping[str, Mapping[str, Any]],
    weighted_paths: Mapping[str, Path],
    *,
    folded_paths: Mapping[str, Path] | None = None,
    mode: str,
    data_sha256: str,
    reference_sha256: str,
    source_sha256: str,
    simulate_cpu: bool,
) -> dict[str, Any]:
    validations: dict[str, Any] = {}
    for arm in ARMS:
        report = reports[arm.kind]
        errors = _validate_arm_report(
            report,
            arm,
            mode,
            data_sha256=data_sha256,
            reference_sha256=reference_sha256,
            source_sha256=source_sha256,
            simulate_cpu=simulate_cpu,
        )
        artifact = _validate_weighted_artifact(weighted_paths[arm.kind], report)
        folded = (
            _validate_folded_nested_diagnostics(
                folded_paths[arm.kind],
                report,
            )
            if BLOCKING_SCHEME == NETSKY_SCHEME and folded_paths is not None
            else None
        )
        if BLOCKING_SCHEME == NETSKY_SCHEME and folded is None:
            folded = {
                "passed": False,
                "errors": ["missing folded diagnostics path for NETSKY"],
            }
        validations[arm.kind] = {
            "passed": (
                not errors
                and artifact["passed"]
                and (folded is None or folded["passed"])
            ),
            "report_errors": errors,
            "weighted_artifact": artifact,
            **({"folded_nested_diagnostics": folded} if folded is not None else {}),
        }

    full_report = reports[FULL.kind]
    compressed_report = reports[HETERODYNE.kind]
    metadata_exclusions = {"arm_label", "likelihood_kind"}
    projections = {
        "schema_version": (
            full_report.get("schema_version"),
            compressed_report.get("schema_version"),
        ),
        "benchmark": (
            full_report.get("benchmark"),
            compressed_report.get("benchmark"),
        ),
        "data": (full_report.get("data"), compressed_report.get("data")),
        "scientific_config_except_likelihood": (
            _scientific_config_projection(full_report),
            _scientific_config_projection(compressed_report),
        ),
        "environment": (
            full_report.get("environment"),
            compressed_report.get("environment"),
        ),
        "devices": (
            full_report.get("devices"),
            compressed_report.get("devices"),
        ),
        "implementation_source": (
            {
                key: value
                for key, value in full_report.get("implementation", {}).items()
                if key != "label"
            },
            {
                key: value
                for key, value in compressed_report.get("implementation", {}).items()
                if key != "label"
            },
        ),
        "pairing_contract": (
            {
                key: value
                for key, value in full_report.get(
                    "paired_likelihood_referee", {}
                ).items()
                if key not in metadata_exclusions
            },
            {
                key: value
                for key, value in compressed_report.get(
                    "paired_likelihood_referee", {}
                ).items()
                if key not in metadata_exclusions
            },
        ),
    }
    checks: list[dict[str, Any]] = []
    for name, (left, right) in projections.items():
        passed = _same_json(left, right)
        checks.append(
            {
                "name": name,
                "passed": passed,
                "first_difference": (
                    None if passed else _first_difference(left, right, path=name)
                ),
            }
        )

    strict_pass = all(value["passed"] for value in validations.values()) and all(
        check["passed"] for check in checks
    )
    first_failure: Any = None
    for arm in ARMS:
        if not validations[arm.kind]["passed"]:
            first_failure = {"arm": arm.kind, **validations[arm.kind]}
            break
    if first_failure is None:
        first_failure = next((check for check in checks if not check["passed"]), None)
    return {
        "strict_pass": strict_pass,
        "criterion": (
            (
                "Each unfolded posterior and separate folded-target diagnostic "
                "must be internally valid and "
                if BLOCKING_SCHEME == NETSKY_SCHEME
                else "Both artifacts must be internally valid and "
            )
            + "all scientific setup, "
            f"RNG, data, source, D={N_DEVICES} FSM, M={NUM_GIBBS_SWEEPS}, "
            "and proposal fields "
            "must match. "
            "Likelihood metadata and stochastic scientific outputs are expected "
            "to differ and are not compared for equality."
        ),
        "arm_validation": validations,
        "pairing_checks": checks,
        "first_failure": first_failure,
    }


def _cell_command(args: argparse.Namespace, arm: LikelihoodArm) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-file",
        str(args.data_file.expanduser().resolve()),
        "--reference-file",
        str(args.reference_file.expanduser().resolve()),
        "--prefix",
        str(args.prefix.expanduser().resolve()),
        "--seed",
        str(args.seed),
        "--mode",
        args.mode,
        "--num-gibbs-sweeps",
        str(args.num_gibbs_sweeps),
        "--blocking-scheme",
        args.blocking_scheme,
        "--direction-mode",
        args.direction_mode,
        "--n-devices",
        str(args.n_devices),
        "--gpu-devices",
        args.gpu_devices,
    ]
    if args.sampler_seed is not None:
        command.extend(["--sampler-seed", str(args.sampler_seed)])
    if args.implementation_revision is not None:
        command.extend(["--implementation-revision", args.implementation_revision])
    if args.simulate_cpu:
        command.append("--simulate-cpu")
    command.extend(["--_cell-kind", arm.kind])
    return command


def _validate_inputs_before_cells(
    data_path: Path,
    reference_path: Path,
) -> tuple[dict[str, Any], str, str]:
    if not data_path.is_file():
        raise SystemExit(f"frozen data bundle does not exist: {data_path}")
    data_sha256 = _sha256(data_path)
    if data_sha256 != CANONICAL_DATA_SHA256:
        raise SystemExit(
            "likelihood pair requires the canonical frozen data SHA-256 "
            f"{CANONICAL_DATA_SHA256}; got {data_sha256}"
        )
    reference, reference_sha256 = _load_reference(reference_path)
    from benchmarks.device_parallel_nss import benchmark_gw170817_full_run as benchmark

    # Read and validate the complete frozen manifest before starting a costly
    # cell.  This intentionally does not create/fetch data.
    benchmark._read_bundle(data_path, WORKLOAD)
    return reference, reference_sha256, data_sha256


def _run_pair(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data_file.expanduser().resolve()
    reference_path = args.reference_file.expanduser().resolve()
    reference, reference_sha256, data_sha256 = _validate_inputs_before_cells(
        data_path,
        reference_path,
    )
    all_paths = {
        "pair": _artifact_paths(args.prefix, args.mode)["pair"],
        **{
            f"{arm.kind}.{kind}": path
            for arm in ARMS
            for kind, path in _artifact_paths(args.prefix, args.mode, arm).items()
        },
    }
    existing = [str(path) for path in all_paths.values() if path.exists()]
    if existing:
        raise SystemExit(
            "refusing to overwrite existing likelihood-pair artifacts: "
            + ", ".join(existing)
        )
    for path in all_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    repository = _repository()
    source = _source_manifest(repository)
    reports: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        print(f"running {args.mode} likelihood cell {arm.label}", flush=True)
        environment = os.environ.copy()
        if not args.simulate_cpu:
            environment["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
        subprocess.run(
            _cell_command(args, arm),
            cwd=repository,
            env=environment,
            check=True,
        )
        after_arm = _source_manifest(repository)
        if after_arm["sha256"] != source["sha256"]:
            raise RuntimeError(
                f"source changed while running {arm.label}: "
                f"{source['sha256']} -> {after_arm['sha256']}"
            )
        report_path = _artifact_paths(args.prefix, args.mode, arm)["report"]
        reports[arm.kind] = _load_json(report_path)

    weighted_paths = {
        arm.kind: _artifact_paths(args.prefix, args.mode, arm)["weighted"]
        for arm in ARMS
    }
    folded_paths = (
        {
            arm.kind: _artifact_paths(args.prefix, args.mode, arm)["folded"]
            for arm in ARMS
        }
        if BLOCKING_SCHEME == NETSKY_SCHEME
        else None
    )
    comparison = compare_pair(
        reports,
        weighted_paths,
        folded_paths=folded_paths,
        mode=args.mode,
        data_sha256=data_sha256,
        reference_sha256=reference_sha256,
        source_sha256=source["sha256"],
        simulate_cpu=bool(args.simulate_cpu),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "name": _pair_name(),
        "scientific_scope": {
            "workload": WORKLOAD,
            "sampled_dimensions": 15,
            "waveform": WAVEFORM,
            "carrier_time_anchor": CARRIER_TIME_ANCHOR,
            "data_status": (
                "frozen public-data reconstruction; the paper's exact private "
                "data-preparation code is unavailable"
            ),
            "paper_relation": (
                "The paper reported different device counts for its full and "
                "heterodyned runs. This controlled pair holds "
                f"D={N_DEVICES} fixed across both arms. The matched U=8 time "
                "grid corrects the demonstrably under-resolved public-proxy "
                "U=1 grid, but the paper's private time grid is unknown."
            ),
            "only_intended_difference": (
                "full-resolution likelihood versus 5000-requested-bin "
                "relative-binning likelihood"
            ),
        },
        "run": {
            "mode": args.mode,
            "prefix_steps": PREFIX_STEPS if args.mode == "prefix" else None,
            "seed": args.seed,
            **(
                {"sampler_seed": args.sampler_seed}
                if args.sampler_seed is not None
                else {}
            ),
            "n_devices": N_DEVICES,
            "num_gibbs_sweeps": NUM_GIBBS_SWEEPS,
            "direction_mode": DIRECTION_MODE,
            "blocking_scheme": BLOCKING_SCHEME,
            **(
                {"num_slice_steps_by_block": list(FAST_RIDGE_INTRINSIC_5STEP_SCHEDULE)}
                if BLOCKING_SCHEME == FAST_RIDGE_INTRINSIC_5STEP_BLOCKING_SCHEME
                else {}
            ),
            **(
                {
                    "block_kernel_modes": list(
                        FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
                    ),
                    "fixed_work": dict(FAST_RIDGE_INTRINSIC_PERIODIC_MH_FIXED_WORK),
                }
                if BLOCKING_SCHEME == FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME
                else {}
            ),
            **(
                {
                    "block_kernel_modes": list(
                        FAST_RIDGE_INTRINSIC_PERIODIC_MH_KERNEL_MODES
                    ),
                    "complementary_de_jump_block": {
                        "parameters": list(
                            COMPLEMENTARY_DE_BLOCK_BY_SCHEME[BLOCKING_SCHEME][
                                "parameters"
                            ]
                        ),
                        "attempts": COMPLEMENTARY_DE_BLOCK_BY_SCHEME[BLOCKING_SCHEME][
                            "attempts"
                        ],
                    },
                    "complementary_de_policy": dict(
                        COMPLEMENTARY_DE_POLICY_BY_SCHEME[BLOCKING_SCHEME]
                    ),
                    "fixed_work": dict(
                        COMPLEMENTARY_DE_FIXED_WORK_BY_SCHEME[BLOCKING_SCHEME]
                    ),
                }
                if BLOCKING_SCHEME in COMPLEMENTARY_DE_BLOCK_BY_SCHEME
                else {}
            ),
            **(
                {
                    "bridge_blocks": [list(block) for block in NETSKY_BRIDGE_BLOCKS],
                    "periodic_wrapped_covariance": True,
                    "fixed_work": dict(NETSKY_FIXED_WORK),
                    "posterior_weighting": UNFOLDED_POSTERIOR_WEIGHTING,
                    "folded_target_semantics": FOLDED_TARGET_SEMANTICS,
                }
                if BLOCKING_SCHEME == NETSKY_SCHEME
                else {}
            ),
            "simulate_cpu": bool(args.simulate_cpu),
            "gpu_devices": None if args.simulate_cpu else args.gpu_devices,
            "data_file": str(data_path),
            "data_sha256": data_sha256,
            "reference_file": str(reference_path),
            "reference_json_sha256": reference_sha256,
            "reference_source_path": reference["source_path"],
            "reference_source_sha256": reference["source_sha256"],
        },
        "source": source,
        "arms": {
            arm.kind: {
                "label": arm.label,
                "variant": FSM_VARIANT,
                **{
                    kind: str(path)
                    for kind, path in _artifact_paths(
                        args.prefix,
                        args.mode,
                        arm,
                    ).items()
                },
            }
            for arm in ARMS
        },
        "comparison": comparison,
    }


def _stress_points(reference: Mapping[str, Any]) -> list[tuple[str, dict[str, float]]]:
    base = {
        key: float(value) for key, value in reference["likelihood_parameters"].items()
    }
    intrinsic = {**base}
    intrinsic.update(
        {
            "M_c": base["M_c"] + 2.0e-4,
            "eta": base["eta"] - 5.0e-4,
            "lambda_1": base["lambda_1"] + 20.0,
            "lambda_2": base["lambda_2"] - 20.0,
        }
    )
    precession = {**base}
    precession.update(
        {
            "s1_x": base["s1_x"] + 8.0e-4,
            "s1_y": base["s1_y"] - 8.0e-4,
            "s2_x": base["s2_x"] + 8.0e-4,
            "s2_y": base["s2_y"] + 8.0e-4,
        }
    )
    extrinsic = {**base}
    extrinsic.update(
        {
            "d_L": base["d_L"] * 1.03,
            "iota": base["iota"] + 0.03,
            "ra": base["ra"] + 0.02,
            "dec": base["dec"] + 0.02,
            "psi": base["psi"] + 0.02,
        }
    )
    return [
        ("reference", base),
        ("intrinsic-small", intrinsic),
        ("precession-small", precession),
        ("extrinsic-small", extrinsic),
    ]


def _sampling_point_to_likelihood(
    arrays: Mapping[str, np.ndarray],
    index: int,
) -> dict[str, float]:
    sampling = {name: float(arrays[name][index]) for name in POSITION_FIELDS}
    return _sampling_mapping_to_likelihood(sampling)


def _sampling_mapping_to_likelihood(
    sampling: Mapping[str, float],
) -> dict[str, float]:
    """Convert one physical prior-space sample to likelihood coordinates."""

    q = sampling["q"]
    result = {
        name: sampling[name]
        for name in (
            "M_c",
            "iota",
            "lambda_1",
            "lambda_2",
            "d_L",
            "ra",
            "dec",
            "psi",
        )
    }
    result["eta"] = q / (1.0 + q) ** 2
    for prefix in ("s1", "s2"):
        magnitude = sampling[f"{prefix}_mag"]
        theta = sampling[f"{prefix}_theta"]
        phi = sampling[f"{prefix}_phi"]
        result[f"{prefix}_x"] = magnitude * math.sin(theta) * math.cos(phi)
        result[f"{prefix}_y"] = magnitude * math.sin(theta) * math.sin(phi)
        result[f"{prefix}_z"] = magnitude * math.cos(theta)
    result["t_c"] = 0.0
    result["phase_c"] = 0.0
    return result


def _load_probe_bank(
    paths: Sequence[Path],
) -> tuple[list[tuple[str, dict[str, float]]], list[dict[str, Any]], set[str]]:
    points: list[tuple[str, dict[str, float]]] = []
    manifests: list[dict[str, Any]] = []
    convergence_names: set[str] = set()
    resolved_paths = [path.expanduser().resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("preflight probe files must be pairwise distinct")
    for file_index, path in enumerate(resolved_paths):
        if not path.is_file():
            raise SystemExit(f"preflight probe file does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = set(WEIGHTED_FIELDS) - set(archive.files)
            if missing:
                raise ValueError(f"probe {path} is missing fields {sorted(missing)}")
            arrays = {name: np.asarray(archive[name]) for name in WEIGHTED_FIELDS}
        counts = {len(values) for values in arrays.values() if values.ndim == 1}
        if (
            len(counts) != 1
            or any(values.ndim != 1 for values in arrays.values())
            or next(iter(counts), 0) <= 0
        ):
            raise ValueError(f"probe {path} does not contain aligned 1D arrays")
        if any(
            not np.all(np.isfinite(arrays[name]))
            for name in set(WEIGHTED_FIELDS) - {"log_likelihood_birth"}
        ):
            raise ValueError(f"probe {path} contains non-finite required values")

        log_weights = arrays["log_weights"]
        maximum_weight = float(np.max(log_weights))
        weights = np.exp(log_weights - maximum_weight)
        weights /= weights.sum()
        cdf = np.cumsum(weights)
        quantiles = (
            np.arange(PROBE_SYSTEMATIC_POINTS_PER_FILE, dtype=np.float64) + 0.5
        ) / PROBE_SYSTEMATIC_POINTS_PER_FILE
        systematic = np.searchsorted(cdf, quantiles, side="left")
        top_index = int(np.argmax(arrays["log_likelihood"]))
        selected = list(dict.fromkeys([*systematic.tolist(), top_index]))
        if len(selected) < PROBE_SYSTEMATIC_POINTS_PER_FILE // 2:
            raise ValueError(f"probe {path} has too few distinct systematic points")

        labels: list[str] = []
        top_name = ""
        for selected_index, sample_index in enumerate(selected):
            suffix = (
                "top-logL"
                if sample_index == top_index
                else f"systematic-{selected_index:02d}"
            )
            name = f"probe-{file_index:02d}-{path.stem}-{suffix}"
            points.append(
                (
                    name,
                    _sampling_point_to_likelihood(arrays, int(sample_index)),
                )
            )
            labels.append(name)
            if sample_index == top_index:
                top_name = name
        if not top_name:
            raise RuntimeError(f"probe {path} did not retain its top-logL point")
        convergence_names.add(top_name)
        manifests.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "stored_count": next(iter(counts)),
                "selected_count": len(selected),
                "selected_names": labels,
                "stored_top_log_likelihood": float(arrays["log_likelihood"][top_index]),
                "stored_top_q": float(arrays["q"][top_index]),
                "stored_log_likelihood_is_gate": False,
            }
        )
    return points, manifests, convergence_names


def _evaluate_likelihood_paths(
    jax: Any,
    likelihood: Any,
    points: Sequence[tuple[str, dict[str, float]]],
    *,
    parity_names: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    parity_names = parity_names or {name for name, _ in points}
    jitted_direct = jax.jit(likelihood.evaluate)
    jitted_cached = jax.jit(likelihood.evaluate_from_waveform)
    results: dict[str, dict[str, float]] = {}
    for name, params in points:
        values = {"direct": likelihood.evaluate(params)}
        if name in parity_names:
            cache = likelihood.generate_waveform(params)
            values.update(
                {
                    "cache": likelihood.evaluate_from_waveform(params, cache),
                    "jit_direct": jitted_direct(params),
                    "jit_cache": jitted_cached(params, cache),
                }
            )
        host = {
            key: float(np.asarray(jax.device_get(value)))
            for key, value in values.items()
        }
        if not all(math.isfinite(value) for value in host.values()):
            raise RuntimeError(f"non-finite {name} likelihood path values: {host}")
        results[name] = host
    return results


def _foreign_cache_ridge_points(
    reference: Mapping[str, Any],
) -> tuple[dict[str, float], list[tuple[str, dict[str, float]]]]:
    """Build cache-parent/target pairs that differ only in fast coordinates."""

    parent = {
        name: float(value) for name, value in reference["likelihood_parameters"].items()
    }
    targets = []
    for name, iota, distance in (
        ("foreign-cache-face-on", 0.05, 15.0),
        ("foreign-cache-edge-on", math.pi / 2.0, 45.0),
        ("foreign-cache-face-away", math.pi - 0.05, 70.0),
    ):
        target = dict(parent)
        target.update({"iota": iota, "d_L": distance})
        targets.append((name, target))
    return parent, targets


def _periodic_move_points(
    reference: Mapping[str, Any],
) -> tuple[
    dict[str, float],
    list[tuple[str, dict[str, float]]],
    list[tuple[str, dict[str, float]]],
]:
    """Build changed-angle probes matching H4's three periodic updates."""

    parent = {
        name: float(value) for name, value in reference["likelihood_parameters"].items()
    }
    points: list[tuple[str, dict[str, float]]] = []
    for label, delta in (("s1", 1.1), ("s2", -0.9)):
        point = dict(parent)
        x_name = f"{label}_x"
        y_name = f"{label}_y"
        radius = math.hypot(parent[x_name], parent[y_name])
        if radius <= 0.0:
            raise ValueError(
                f"periodic preflight requires nonzero {label} transverse spin"
            )
        phi = math.atan2(parent[y_name], parent[x_name]) + delta
        point[x_name] = radius * math.cos(phi)
        point[y_name] = radius * math.sin(phi)
        points.append((f"periodic-move-{label}_phi", point))
    psi_point = dict(parent)
    psi_point["psi"] = (parent["psi"] + 0.5 * math.pi) % math.pi
    if math.isclose(psi_point["psi"], parent["psi"], rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("periodic preflight psi probe did not move")
    points.append(("periodic-move-psi", psi_point))
    return parent, points, [("periodic-move-psi", psi_point)]


def _netsky_polarization_identity_points(
    reference: Mapping[str, Any],
) -> tuple[dict[str, float], list[tuple[str, dict[str, float]]]]:
    """Build the base point and exact half-period polarization image."""

    parent = {
        name: float(value) for name, value in reference["likelihood_parameters"].items()
    }
    image = dict(parent)
    image["psi"] = (parent["psi"] + 0.5 * math.pi) % math.pi
    if math.isclose(image["psi"], parent["psi"], rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("NETSKY polarization image did not move")
    return parent, [
        ("polarization-base", dict(parent)),
        ("polarization-half-period", image),
    ]


def _netsky_polarization_identity_check(
    full_values: Mapping[str, Mapping[str, float]],
    compressed_values: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Prove the P image through direct, shared-cache, and JIT paths."""

    expected_names = {"polarization-base", "polarization-half-period"}
    expected_paths = {
        "direct",
        "foreign_cache",
        "jit_direct",
        "jit_foreign_cache",
    }
    per_likelihood: dict[str, Any] = {}
    maxima: list[float] = []
    for likelihood_name, values in (
        (FULL.kind, full_values),
        (HETERODYNE.kind, compressed_values),
    ):
        if set(values) != expected_names:
            raise ValueError(f"{likelihood_name} polarization probes are incomplete")
        if any(set(values[name]) != expected_paths for name in expected_names):
            raise ValueError(
                f"{likelihood_name} polarization likelihood paths are incomplete"
            )
        reference_value = values["polarization-base"]["direct"]
        deltas = {
            name: {
                path: value - reference_value for path, value in values[name].items()
            }
            for name in sorted(expected_names)
        }
        maximum = max(
            abs(delta) for paths in deltas.values() for delta in paths.values()
        )
        maxima.append(maximum)
        per_likelihood[likelihood_name] = {
            "path_minus_base_direct": deltas,
            "max_abs_delta": maximum,
        }
    maximum = max(maxima)
    return {
        "passed": maximum <= PARITY_ATOL,
        "abs_tolerance": PARITY_ATOL,
        "observed_max_abs_delta": maximum,
        "identity": "psi -> (psi + pi/2) mod pi",
        "cache_policy": "one waveform cache built at the base point",
        "per_likelihood": per_likelihood,
    }


def _periodic_move_cache_parity_check(
    full_values: Mapping[str, Mapping[str, float]],
    compressed_values: Mapping[str, Mapping[str, float]],
    full_foreign_values: Mapping[str, Mapping[str, float]],
    compressed_foreign_values: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Fail closed on direct/cache/JIT parity for each actual H4 angle move."""

    own_names = {
        "periodic-move-s1_phi",
        "periodic-move-s2_phi",
        "periodic-move-psi",
    }
    expected_own_paths = {"direct", "cache", "jit_direct", "jit_cache"}
    expected_foreign_paths = {
        "direct",
        "foreign_cache",
        "jit_direct",
        "jit_foreign_cache",
    }
    per_likelihood: dict[str, Any] = {}
    errors: list[float] = []
    for likelihood_name, values, foreign_values in (
        (FULL.kind, full_values, full_foreign_values),
        (HETERODYNE.kind, compressed_values, compressed_foreign_values),
    ):
        if set(values) != own_names:
            raise ValueError(
                f"{likelihood_name} periodic move probes do not cover H4 angles"
            )
        if set(foreign_values) != {"periodic-move-psi"}:
            raise ValueError(
                f"{likelihood_name} periodic foreign-cache probes must cover psi"
            )
        per_point: dict[str, Any] = {}
        for name in sorted(own_names):
            paths = values[name]
            if set(paths) != expected_own_paths:
                raise ValueError(f"{likelihood_name} {name} cache paths are incomplete")
            direct = paths["direct"]
            deltas = {path: value - direct for path, value in paths.items()}
            maximum = max(abs(delta) for delta in deltas.values())
            errors.append(maximum)
            per_point[name] = {
                "path_minus_direct": deltas,
                "max_abs_delta": maximum,
            }
        foreign_paths = foreign_values["periodic-move-psi"]
        if set(foreign_paths) != expected_foreign_paths:
            raise ValueError(
                f"{likelihood_name} psi foreign-cache paths are incomplete"
            )
        direct = foreign_paths["direct"]
        deltas = {path: value - direct for path, value in foreign_paths.items()}
        maximum = max(abs(delta) for delta in deltas.values())
        errors.append(maximum)
        per_point["periodic-move-psi-foreign-cache"] = {
            "path_minus_direct": deltas,
            "max_abs_delta": maximum,
        }
        per_likelihood[likelihood_name] = per_point
    maximum = max(errors)
    return {
        "passed": maximum <= PARITY_ATOL,
        "abs_tolerance": PARITY_ATOL,
        "observed_max_abs_delta": maximum,
        "expected_cache_policy": {
            "s1_phi": "waveform-rebuild",
            "s2_phi": "waveform-rebuild",
            "psi": "cache-hit",
        },
        "per_likelihood_and_move": per_likelihood,
    }


def _complementary_de_proposal_point(
    reference: Mapping[str, Any],
    *,
    blocking_scheme: str = FAST_RIDGE_INTRINSIC_PERIODIC_MH_CDE4_BLOCKING_SCHEME,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Build one deterministic gamma-one cDE-family proposal at fixed ``d_hat``."""

    if blocking_scheme not in COMPLEMENTARY_DE_POLICY_BY_SCHEME:
        raise ValueError(f"unsupported complementary-DE scheme {blocking_scheme!r}")

    parent = {
        name: float(value) for name, value in reference["sampling_parameters"].items()
    }
    delta = {
        "M_c": 1.0e-4,
        "q": 1.0e-2,
        "lambda_1": 5.0,
        "lambda_2": -7.0,
        "s1_mag": 1.0e-3,
        "s1_theta": 2.0e-2,
        "s2_mag": -1.0e-3,
        "s2_theta": -2.0e-2,
    }
    intrinsic = FAST_RIDGE_INTRINSIC_BLOCKS[0]
    donor_a = dict(parent)
    donor_b = dict(parent)
    for name in intrinsic:
        donor_a[name] = parent[name] + 0.5 * delta[name]
        donor_b[name] = parent[name] - 0.5 * delta[name]
    for donor in (donor_a, donor_b):
        donor["d_L"] = parent["d_L"] * (donor["M_c"] / parent["M_c"]) ** (5.0 / 6.0)
    proposal = dict(parent)
    for name in intrinsic:
        proposal[name] = parent[name] + donor_a[name] - donor_b[name]
    proposal["d_L"] = parent["d_L"] * (proposal["M_c"] / parent["M_c"]) ** (5.0 / 6.0)
    return _sampling_mapping_to_likelihood(proposal), {
        "parameters": list(intrinsic),
        "parent": parent,
        "donor_a": donor_a,
        "donor_b": donor_b,
        "proposal": proposal,
        "fixed_sampling_coordinates": ["d_hat"],
        "physical_distance_relation": (
            "d_L_new=d_L_parent*(M_c_new/M_c_parent)^(5/6) at fixed R_net and d_hat"
        ),
        "policy": dict(COMPLEMENTARY_DE_POLICY_BY_SCHEME[blocking_scheme]),
    }


def _complementary_de_proposal_cache_parity_check(
    full_values: Mapping[str, Mapping[str, float]],
    compressed_values: Mapping[str, Mapping[str, float]],
    proposal_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on direct/cache/JIT parity for a cDE-family proposal."""

    expected_names = {"complementary-de-proposal"}
    expected_paths = {"direct", "cache", "jit_direct", "jit_cache"}
    per_likelihood: dict[str, Any] = {}
    maxima: list[float] = []
    for likelihood_name, values in (
        (FULL.kind, full_values),
        (HETERODYNE.kind, compressed_values),
    ):
        if set(values) != expected_names:
            raise ValueError(
                f"{likelihood_name} complementary-DE probes are incomplete"
            )
        paths = values["complementary-de-proposal"]
        if set(paths) != expected_paths:
            raise ValueError(
                f"{likelihood_name} complementary-DE cache paths are incomplete"
            )
        direct = paths["direct"]
        deltas = {path: value - direct for path, value in paths.items()}
        maximum = max(abs(delta) for delta in deltas.values())
        maxima.append(maximum)
        per_likelihood[likelihood_name] = {
            "path_minus_direct": deltas,
            "max_abs_delta": maximum,
        }
    maximum = max(maxima)
    return {
        "passed": maximum <= PARITY_ATOL,
        "abs_tolerance": PARITY_ATOL,
        "observed_max_abs_delta": maximum,
        "proposal_contract": copy.deepcopy(dict(proposal_contract)),
        "expected_cache_policy": "waveform-rebuild-transactional-rollback",
        "per_likelihood": per_likelihood,
    }


def _evaluate_foreign_cache_paths(
    jax: Any,
    likelihood: Any,
    parent: Mapping[str, float],
    targets: Sequence[tuple[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Compare direct evaluation with one cache built at another ridge point."""

    cache = likelihood.generate_waveform(dict(parent))
    jitted_direct = jax.jit(likelihood.evaluate)
    jitted_cached = jax.jit(likelihood.evaluate_from_waveform)
    results: dict[str, dict[str, float]] = {}
    for name, params in targets:
        values = {
            "direct": likelihood.evaluate(params),
            "foreign_cache": likelihood.evaluate_from_waveform(params, cache),
            "jit_direct": jitted_direct(params),
            "jit_foreign_cache": jitted_cached(params, cache),
        }
        host = {
            key: float(np.asarray(jax.device_get(value)))
            for key, value in values.items()
        }
        if not all(math.isfinite(value) for value in host.values()):
            raise RuntimeError(f"non-finite {name} foreign-cache values: {host}")
        results[name] = host
    return results


def _q_time_grid_points(
    reference: Mapping[str, Any],
) -> list[tuple[str, dict[str, float]]]:
    """Stress q-dependent carrier timing across the complete q prior support."""

    base = {
        name: float(value) for name, value in reference["likelihood_parameters"].items()
    }
    points = []
    for q in Q_TIME_GRID_VALUES:
        point = dict(base)
        point["eta"] = q / (1.0 + q) ** 2
        point["t_c"] = 0.0
        points.append((f"q-time-grid-{q:.3f}", point))
    return points


def _evaluate_direct(
    jax: Any,
    likelihood: Any,
    points: Sequence[tuple[str, dict[str, float]]],
) -> dict[str, float]:
    values = {
        name: float(np.asarray(jax.device_get(likelihood.evaluate(params))))
        for name, params in points
    }
    invalid = {
        name: value for name, value in values.items() if not math.isfinite(value)
    }
    if invalid:
        raise RuntimeError(f"non-finite direct likelihood values: {invalid}")
    return values


def _u8_u16_point_bank_check(
    points: Sequence[tuple[str, dict[str, float]]],
    full_u8_values: Mapping[str, Mapping[str, float]],
    u16_values: Mapping[str, float],
    *,
    local_names: set[str],
) -> dict[str, Any]:
    """Require U8/U16 convergence on every local and selected probe point."""

    names = [name for name, _ in points]
    if not names or len(set(names)) != len(names):
        raise ValueError("U8/U16 point bank must be non-empty with unique names")
    expected_names = set(names)
    if set(full_u8_values) != expected_names:
        raise ValueError("full-U8 values do not cover the complete point bank")
    if set(u16_values) != expected_names:
        raise ValueError("U16 values do not cover the complete point bank")
    if not local_names <= expected_names:
        raise ValueError("local point names are not a subset of the point bank")

    per_point: dict[str, Any] = {}
    errors: list[float] = []
    for name in names:
        full_u8 = full_u8_values[name]["direct"]
        u16 = u16_values[name]
        delta = u16 - full_u8
        errors.append(abs(delta))
        per_point[name] = {
            "source": "local" if name in local_names else "independent-probe",
            "full_u8_log_likelihood": full_u8,
            "u16_log_likelihood": u16,
            "u16_minus_full_u8": delta,
        }

    maximum = max(errors)
    local_count = sum(name in local_names for name in names)
    probe_count = len(names) - local_count
    return {
        "passed": maximum <= MAX_U8_U16_ABS_LOGL_DELTA,
        "maximum_abs_log_likelihood_delta": MAX_U8_U16_ABS_LOGL_DELTA,
        "observed_max_abs_delta": maximum,
        "point_count": len(names),
        "local_point_count": local_count,
        "independent_probe_point_count": probe_count,
        "per_point": per_point,
        "note": (
            "Fatal U8/U16 convergence covers every local stress coordinate and "
            "every selected coordinate from each independent probe file, "
            "including systematic and top-logL points."
        ),
    }


def _foreign_cache_parity_check(
    full_values: Mapping[str, Mapping[str, float]],
    compressed_values: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Summarize foreign-parent cache parity and fail closed on either arm."""

    per_likelihood: dict[str, dict[str, Any]] = {}
    all_errors: list[float] = []
    for likelihood_name, values_by_target in (
        (FULL.kind, full_values),
        (HETERODYNE.kind, compressed_values),
    ):
        if not values_by_target:
            raise ValueError(
                f"foreign-cache parity has no {likelihood_name} target values"
            )
        per_target: dict[str, Any] = {}
        for target_name, paths in values_by_target.items():
            direct = paths["direct"]
            deltas = {path_name: value - direct for path_name, value in paths.items()}
            maximum = max(abs(delta) for delta in deltas.values())
            all_errors.append(maximum)
            per_target[target_name] = {
                "path_minus_direct": deltas,
                "max_abs_delta": maximum,
            }
        per_likelihood[likelihood_name] = per_target
    maximum = max(all_errors)
    return {
        "passed": maximum <= PARITY_ATOL,
        "abs_tolerance": PARITY_ATOL,
        "observed_max_abs_delta": maximum,
        "per_likelihood_and_target": per_likelihood,
    }


def _q_time_grid_regression_check(
    points: Sequence[tuple[str, dict[str, float]]],
    full_u8_values: Mapping[str, Mapping[str, float]],
    compressed_u8_values: Mapping[str, Mapping[str, float]],
    u16_values: Mapping[str, float],
    *,
    bank_max_log_likelihood: float,
) -> dict[str, Any]:
    """Check cache, compression, and relevant U8/U16 behavior across q."""

    if len(points) != len(Q_TIME_GRID_VALUES):
        raise ValueError(
            "q-time grid does not match the frozen complete-q grid: "
            f"{len(points)} != {len(Q_TIME_GRID_VALUES)}"
        )
    if not math.isfinite(bank_max_log_likelihood):
        raise ValueError("q-time grid bank maximum must be finite")

    expected_paths = {"direct", "cache", "jit_direct", "jit_cache"}
    expected_names = [f"q-time-grid-{q:.3f}" for q in Q_TIME_GRID_VALUES]
    observed_names = [name for name, _ in points]
    if observed_names != expected_names:
        raise ValueError(
            "q-time grid names/order do not match the frozen complete-q grid"
        )
    for likelihood_name, values in (
        (FULL.kind, full_u8_values),
        (HETERODYNE.kind, compressed_u8_values),
    ):
        if set(values) != set(expected_names):
            raise ValueError(
                f"q-time grid {likelihood_name} values do not match the frozen grid"
            )
        for name in expected_names:
            if set(values[name]) != expected_paths:
                raise ValueError(
                    f"q-time grid {likelihood_name} paths for {name} must be "
                    f"{sorted(expected_paths)}"
                )
    if set(u16_values) != set(expected_names):
        raise ValueError("q-time grid U16 values do not match the frozen grid")

    comparison_maximum = max(
        bank_max_log_likelihood,
        *(full_u8_values[name]["direct"] for name in expected_names),
    )
    per_point: dict[str, Any] = {}
    parity_errors: list[float] = []
    beta_values: list[float] = []
    near_max_errors: list[float] = []
    compression_passes: list[bool] = []
    all_u8_u16_errors: list[float] = []
    relevant_u8_u16_errors: list[float] = []
    relevant_names: list[str] = []
    excluded_names: list[str] = []
    for q, (name, params) in zip(Q_TIME_GRID_VALUES, points, strict=True):
        expected_eta = q / (1.0 + q) ** 2
        if not math.isclose(params["eta"], expected_eta, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"q-time grid eta does not match q at {name}")
        if params["t_c"] != 0.0:
            raise ValueError(f"q-time grid t_c must remain zero at {name}")

        full_paths = dict(full_u8_values[name])
        compressed_paths = dict(compressed_u8_values[name])
        full_u8 = full_paths["direct"]
        compressed_u8 = compressed_paths["direct"]
        full_path_errors = {path: value - full_u8 for path, value in full_paths.items()}
        compressed_path_errors = {
            path: value - compressed_u8 for path, value in compressed_paths.items()
        }
        full_parity = max(abs(value) for value in full_path_errors.values())
        compressed_parity = max(abs(value) for value in compressed_path_errors.values())
        parity_errors.extend((full_parity, compressed_parity))

        log_likelihood_drop = comparison_maximum - full_u8
        compression_delta = compressed_u8 - full_u8
        compression_abs_delta = abs(compression_delta)
        beta: float | None = None
        if log_likelihood_drop >= MIN_BETA_LOGL_DROP:
            beta = compression_abs_delta / log_likelihood_drop
            beta_values.append(beta)
            compression_pass = beta <= MAX_RELATIVE_BINNING_BETA
        else:
            near_max_errors.append(compression_abs_delta)
            compression_pass = compression_abs_delta <= MAX_REFERENCE_ABS_LOGL_DELTA
        compression_passes.append(compression_pass)

        u16 = u16_values[name]
        u8_u16_delta = u16 - full_u8
        u8_u16_abs_delta = abs(u8_u16_delta)
        all_u8_u16_errors.append(u8_u16_abs_delta)
        within_relevance_window = log_likelihood_drop <= MAX_Q_TIME_U8_U16_LOGL_DROP
        if within_relevance_window:
            relevant_names.append(name)
            relevant_u8_u16_errors.append(u8_u16_abs_delta)
        else:
            excluded_names.append(name)

        per_point[name] = {
            "q": q,
            "eta": params["eta"],
            "t_c_seconds": params["t_c"],
            FULL.kind: full_paths,
            HETERODYNE.kind: compressed_paths,
            "full_path_minus_direct": full_path_errors,
            "full_path_max_abs_delta": full_parity,
            "heterodyne_path_minus_direct": compressed_path_errors,
            "heterodyne_path_max_abs_delta": compressed_parity,
            "heterodyne_minus_full_u8_log_likelihood": compression_delta,
            "full_u8_log_likelihood_drop_from_comparison_max": log_likelihood_drop,
            "relative_binning_beta": beta,
            "compression_accuracy_passed": compression_pass,
            "u16_log_likelihood": u16,
            "u16_minus_full_u8": u8_u16_delta,
            "within_u8_u16_relevance_window": within_relevance_window,
        }

    parity_maximum = max(parity_errors)
    beta_maximum = max(beta_values) if beta_values else None
    near_max_error = max(near_max_errors) if near_max_errors else None
    relevant_maximum = max(relevant_u8_u16_errors) if relevant_u8_u16_errors else None
    return {
        "comparison_max_log_likelihood": comparison_maximum,
        "per_point": per_point,
        "checks": {
            "q_time_grid_direct_cache_jit_parity": {
                "passed": parity_maximum <= PARITY_ATOL,
                "abs_tolerance": PARITY_ATOL,
                "observed_max_abs_delta": parity_maximum,
            },
            "q_time_grid_compression_accuracy": {
                "passed": all(compression_passes),
                "near_max_maximum_abs_log_likelihood_delta": (
                    MAX_REFERENCE_ABS_LOGL_DELTA
                ),
                "minimum_log_likelihood_drop_for_beta": MIN_BETA_LOGL_DROP,
                "maximum_relative_binning_beta": MAX_RELATIVE_BINNING_BETA,
                "observed_near_max_max_abs_delta": near_max_error,
                "observed_max_beta": beta_maximum,
                "near_max_point_count": len(near_max_errors),
                "beta_point_count": len(beta_values),
            },
            "q_time_grid_u8_u16_relevant_convergence": {
                "passed": (
                    bool(relevant_u8_u16_errors)
                    and relevant_maximum is not None
                    and relevant_maximum <= MAX_U8_U16_ABS_LOGL_DELTA
                ),
                "maximum_abs_log_likelihood_delta": MAX_U8_U16_ABS_LOGL_DELTA,
                "maximum_full_u8_log_likelihood_drop": (MAX_Q_TIME_U8_U16_LOGL_DROP),
                "observed_max_abs_delta_relevant": relevant_maximum,
                "observed_max_abs_delta_all_points": max(all_u8_u16_errors),
                "relevant_point_count": len(relevant_names),
                "relevant_point_names": relevant_names,
                "excluded_point_count": len(excluded_names),
                "excluded_point_names": excluded_names,
                "gate_revision_status": "post-failure exploratory",
            },
        },
        "note": (
            "The likelihood marginalizes coalescence time on the matched dense "
            "grid; t_c is fixed at zero in the parameter dictionaries while q "
            "stresses q-dependent carrier placement. Full/compressed U8 cache "
            "parity and compression accuracy remain fatal over every q point. "
            "After observing an isolated U8/U16 discrepancy 824 nat below the "
            "likelihood maximum, the U8/U16 fatal window was revised "
            "post-failure and is exploratory rather than a predeclared "
            "confirmation gate. Every point and the all-point maximum remain "
            "reported."
        ),
    }


def _waveform_cache_dependency_check(
    full: Any,
    compressed: Any,
) -> dict[str, Any]:
    """Require q/eta and time to stay out of the fast ridge cache contract."""

    expected = frozenset(("d_L", "iota"))
    per_likelihood: dict[str, Any] = {}
    passed = True
    for name, likelihood in ((FULL.kind, full), (HETERODYNE.kind, compressed)):
        cacheable = frozenset(likelihood.waveform_cacheable_parameter_names)
        waveform_parameters = frozenset(likelihood.waveform.parameter_names)
        likelihood_passed = (
            cacheable == expected
            and "eta" in waveform_parameters
            and "eta" not in cacheable
            and "t_c" not in cacheable
        )
        passed = passed and likelihood_passed
        per_likelihood[name] = {
            "passed": likelihood_passed,
            "cacheable_parameter_names": sorted(cacheable),
            "eta_is_waveform_parameter": "eta" in waveform_parameters,
            "eta_is_cacheable": "eta" in cacheable,
            "t_c_is_cacheable": "t_c" in cacheable,
        }
    return {
        "passed": passed,
        "expected_cacheable_parameter_names": sorted(expected),
        "per_likelihood": per_likelihood,
        "note": (
            "Sampling q is transformed to waveform eta, which must rebuild the "
            "carrier. Coalescence time remains marginalized and is not part of "
            "the fast iota/d_L cache contract."
        ),
    }


def _fixed_work_cache_schedule_check(
    full: Any,
    compressed: Any,
    components: Mapping[str, Any],
    *,
    blocking_scheme: str,
) -> dict[str, Any]:
    """Resolve and prove the candidate's fixed-work cache schedule."""

    expected_schedule = FIXED_WORK_SCHEDULE_BY_SCHEME.get(blocking_scheme)
    if expected_schedule is None or blocking_scheme not in BLOCKS_BY_SCHEME:
        return {
            "passed": False,
            "blocking_scheme": blocking_scheme,
            "supported_blocking_schemes": sorted(FIXED_WORK_SCHEDULE_BY_SCHEME),
            "error": "unsupported blocking scheme for fixed-work proof",
        }

    configured_slice_steps = expected_schedule.get("num_slice_steps_by_block")
    configured_kernel_modes = expected_schedule.get("block_kernel_modes")
    configured_complementary_de = expected_schedule.get("complementary_de_jump_block")
    configured_bridge_blocks = expected_schedule.get("bridge_blocks", ())
    expected_bridge_rebuild = list(
        expected_schedule.get("bridge_rebuild_required_by_block", ())
    )
    resolved_kernel_modes = list(
        configured_kernel_modes
        if configured_kernel_modes is not None
        else ("slice",) * len(expected_schedule["block_sizes"])
    )
    resolved_slice_steps = list(
        configured_slice_steps
        if configured_slice_steps is not None
        else expected_schedule["block_sizes"]
    )
    complementary_de_attempts = (
        int(configured_complementary_de["attempts"])
        if configured_complementary_de is not None
        else 0
    )
    complementary_de_parameters = (
        tuple(configured_complementary_de["parameters"])
        if configured_complementary_de is not None
        else ()
    )
    complementary_de_block_indexes = [
        index
        for index, block in enumerate(BLOCKS_BY_SCHEME[blocking_scheme])
        if tuple(block) == complementary_de_parameters
    ]
    complementary_de_insert_after_block = (
        complementary_de_block_indexes[0]
        if len(complementary_de_block_indexes) == 1
        else None
    )
    expected_gibbs_sweeps = (
        NUM_GIBBS_SWEEPS
        if blocking_scheme
        in {FAST_RIDGE_INTRINSIC_PERIODIC_MH_BLOCKING_SCHEME, NETSKY_SCHEME}
        else 1
    )
    bridge_slice_updates = expected_gibbs_sweeps * len(configured_bridge_blocks)
    waveform_rebuild_bridge_updates = expected_gibbs_sweeps * sum(
        expected_bridge_rebuild
    )
    cache_hit_bridge_updates = bridge_slice_updates - waveform_rebuild_bridge_updates
    total_updates = (
        expected_gibbs_sweeps * (sum(resolved_slice_steps) + complementary_de_attempts)
        + bridge_slice_updates
    )
    total_slice_updates = (
        expected_gibbs_sweeps
        * sum(
            steps
            for steps, mode in zip(
                resolved_slice_steps,
                resolved_kernel_modes,
                strict=True,
            )
            if mode == "slice"
        )
        + bridge_slice_updates
    )
    periodic_independence_attempts = expected_gibbs_sweeps * sum(
        steps
        for steps, mode in zip(
            resolved_slice_steps,
            resolved_kernel_modes,
            strict=True,
        )
        if mode == "periodic-uniform-independence"
    )
    waveform_rebuild_slice_updates = (
        expected_gibbs_sweeps
        * sum(
            steps
            for steps, rebuild, mode in zip(
                resolved_slice_steps,
                expected_schedule["rebuild_required_by_block"],
                resolved_kernel_modes,
                strict=True,
            )
            if rebuild and mode == "slice"
        )
        + waveform_rebuild_bridge_updates
    )
    cache_hit_slice_updates = (
        expected_gibbs_sweeps
        * sum(
            steps
            for steps, rebuild, mode in zip(
                resolved_slice_steps,
                expected_schedule["rebuild_required_by_block"],
                resolved_kernel_modes,
                strict=True,
            )
            if not rebuild and mode == "slice"
        )
        + cache_hit_bridge_updates
    )
    rebuild_periodic_attempts = expected_gibbs_sweeps * sum(
        steps
        for steps, rebuild, mode in zip(
            resolved_slice_steps,
            expected_schedule["rebuild_required_by_block"],
            resolved_kernel_modes,
            strict=True,
        )
        if rebuild and mode == "periodic-uniform-independence"
    )
    cache_hit_periodic_attempts = periodic_independence_attempts - (
        rebuild_periodic_attempts
    )
    waveform_rebuild_complementary_de_attempts = (
        expected_gibbs_sweeps * complementary_de_attempts
        if complementary_de_insert_after_block is not None
        and expected_schedule["rebuild_required_by_block"][
            complementary_de_insert_after_block
        ]
        else 0
    )
    cache_hit_complementary_de_attempts = (
        expected_gibbs_sweeps * complementary_de_attempts
        - waveform_rebuild_complementary_de_attempts
    )
    cache_segments = 1 + sum(
        left != right
        for left, right in zip(
            expected_schedule["rebuild_required_by_block"],
            expected_schedule["rebuild_required_by_block"][1:],
        )
    )
    expected = {
        "blocks": [list(block) for block in BLOCKS_BY_SCHEME[blocking_scheme]],
        "block_sizes": list(expected_schedule["block_sizes"]),
        "rebuild_required_by_block": list(
            expected_schedule["rebuild_required_by_block"]
        ),
        "total_dimensions": 15,
        "num_gibbs_sweeps": expected_gibbs_sweeps,
        "num_inner_steps_per_dim": 1,
        "num_slice_steps_by_block": (
            list(configured_slice_steps) if configured_slice_steps is not None else None
        ),
        "block_kernel_modes": resolved_kernel_modes,
        "resolved_slice_steps_by_block": resolved_slice_steps,
        "total_updates": total_updates,
        "total_slice_updates": total_slice_updates,
        "waveform_rebuild_slice_updates": waveform_rebuild_slice_updates,
        "cache_hit_slice_updates": cache_hit_slice_updates,
        "periodic_independence_attempts": periodic_independence_attempts,
        "waveform_rebuild_periodic_independence_attempts": (rebuild_periodic_attempts),
        "cache_hit_periodic_independence_attempts": cache_hit_periodic_attempts,
        "complementary_de_attempts": complementary_de_attempts,
        "waveform_rebuild_complementary_de_attempts": (
            waveform_rebuild_complementary_de_attempts
        ),
        "cache_hit_complementary_de_attempts": (cache_hit_complementary_de_attempts),
        "complementary_de_jump_block": (
            {
                "parameters": list(complementary_de_parameters),
                "attempts": complementary_de_attempts,
            }
            if configured_complementary_de is not None
            else None
        ),
        "complementary_de_gamma": 1.0 if configured_complementary_de else None,
        "complementary_de_insert_after_block": (complementary_de_insert_after_block),
        "cache_segments": cache_segments,
        "direction_mode": DIRECTION_MODE,
    }
    if configured_bridge_blocks:
        expected.update(
            {
                "bridge_blocks": [list(block) for block in configured_bridge_blocks],
                "bridge_rebuild_required_by_block": expected_bridge_rebuild,
                "bridge_slice_updates": bridge_slice_updates,
                "waveform_rebuild_bridge_slice_updates": (
                    waveform_rebuild_bridge_updates
                ),
                "cache_hit_bridge_slice_updates": cache_hit_bridge_updates,
            }
        )
    try:
        from jimgw.core.single_event.blocked_likelihood import (
            _build_rebuild_required_by_block,
            _infer_waveform_sampling_dependencies,
            _resolve_rebuild_required_by_parameter_groups,
            _validate_parameter_blocks,
        )
        from jimgw.samplers.config import BlackJAXSwiGConfig, DEJumpBlockConfig

        blocks = tuple(tuple(block) for block in components["spec"]["blocks"])
        sampler_config = BlackJAXSwiGConfig(
            blocks=[list(block) for block in blocks],
            num_gibbs_sweeps=NUM_GIBBS_SWEEPS,
            num_inner_steps_per_dim=NUM_INNER_STEPS_PER_DIM,
            direction_mode=DIRECTION_MODE,
            num_slice_steps_by_block=(
                list(configured_slice_steps)
                if configured_slice_steps is not None
                else None
            ),
            block_kernel_modes=(
                list(configured_kernel_modes)
                if configured_kernel_modes is not None
                else None
            ),
            bridge_blocks=[list(block) for block in configured_bridge_blocks],
            complementary_de_jump_block=(
                DEJumpBlockConfig(
                    parameters=list(complementary_de_parameters),
                    attempts=complementary_de_attempts,
                )
                if configured_complementary_de is not None
                else None
            ),
            num_de_jumps=0,
        )
        sampling_parameter_names = tuple(components["prior"].parameter_names)
        sample_transforms = tuple(components["sample_transforms"])
        likelihood_transforms = tuple(components["likelihood_transforms"])
        for transform in sample_transforms:
            sampling_parameter_names = transform.propagate_name(
                sampling_parameter_names
            )
        _validate_parameter_blocks(
            blocks,
            parameter_names=sampling_parameter_names,
        )
        q_block_indexes = [index for index, block in enumerate(blocks) if "q" in block]
        if len(q_block_indexes) != 1:
            raise ValueError(
                f"expected exactly one q block, observed {q_block_indexes}"
            )
        q_block_index = q_block_indexes[0]

        rebuild_by_likelihood: dict[str, list[bool]] = {}
        bridge_rebuild_by_likelihood: dict[str, list[bool]] = {}
        per_likelihood: dict[str, dict[str, Any]] = {}
        likelihoods = ((FULL.kind, full), (HETERODYNE.kind, compressed))
        for name, likelihood in likelihoods:
            rebuild_by_block = _build_rebuild_required_by_block(
                likelihood,
                blocks,
                parameter_names=sampling_parameter_names,
                sample_transforms=sample_transforms,
                likelihood_transforms=likelihood_transforms,
            )
            flags = list(rebuild_by_block.values())
            dependencies = _infer_waveform_sampling_dependencies(
                likelihood,
                sampling_parameter_names,
                sample_transforms,
                likelihood_transforms,
            )
            resolved_bridges = _resolve_rebuild_required_by_parameter_groups(
                likelihood,
                sampler_config.bridge_blocks,
                parameter_names=sampling_parameter_names,
                sample_transforms=sample_transforms,
                likelihood_transforms=likelihood_transforms,
            )
            bridge_flags = [
                requires_rebuild for _, requires_rebuild in resolved_bridges
            ]
            cacheable = frozenset(likelihood.waveform_cacheable_parameter_names)
            waveform_parameters = frozenset(likelihood.waveform.parameter_names)
            rebuild_by_likelihood[name] = flags
            bridge_rebuild_by_likelihood[name] = bridge_flags
            per_likelihood[name] = {
                "waveform_sampling_dependencies": sorted(dependencies),
                "eta_is_waveform_parameter": "eta" in waveform_parameters,
                "eta_is_cacheable": "eta" in cacheable,
                "q_requires_waveform_rebuild": "q" in dependencies,
                "q_block_requires_waveform_rebuild": flags[q_block_index],
                "t_c_is_sampling_parameter": "t_c" in sampling_parameter_names,
                "t_c_is_waveform_parameter": "t_c" in waveform_parameters,
                "t_c_is_fixed_parameter": "t_c" in likelihood.fixed_parameters,
                "t_c_is_marginalized": bool(likelihood.time_marginalization),
            }

        block_sizes = [len(block) for block in blocks]
        reference_flags = rebuild_by_likelihood[FULL.kind]
        reference_bridge_flags = bridge_rebuild_by_likelihood[FULL.kind]
        configured_steps = sampler_config.num_slice_steps_by_block
        kernel_modes = list(
            sampler_config.block_kernel_modes or ("slice",) * len(blocks)
        )
        resolved_steps = list(
            configured_steps
            if configured_steps is not None
            else (NUM_INNER_STEPS_PER_DIM * block_size for block_size in block_sizes)
        )
        resolved_complementary_de = sampler_config.complementary_de_jump_block
        observed_complementary_attempts = (
            resolved_complementary_de.attempts
            if resolved_complementary_de is not None
            else 0
        )
        observed_complementary_parameters = (
            tuple(resolved_complementary_de.parameters)
            if resolved_complementary_de is not None
            else ()
        )
        observed_complementary_indexes = [
            index
            for index, block in enumerate(blocks)
            if tuple(block) == observed_complementary_parameters
        ]
        observed_complementary_insert_after_block = (
            observed_complementary_indexes[0]
            if len(observed_complementary_indexes) == 1
            else None
        )
        observed_bridge_slice_updates = NUM_GIBBS_SWEEPS * len(
            sampler_config.bridge_blocks
        )
        observed_rebuild_bridge_updates = NUM_GIBBS_SWEEPS * sum(reference_bridge_flags)
        observed_cache_bridge_updates = (
            observed_bridge_slice_updates - observed_rebuild_bridge_updates
        )
        total_updates = (
            NUM_GIBBS_SWEEPS * (sum(resolved_steps) + observed_complementary_attempts)
            + observed_bridge_slice_updates
        )
        total_slice_updates = (
            NUM_GIBBS_SWEEPS
            * sum(
                steps
                for steps, mode in zip(resolved_steps, kernel_modes, strict=True)
                if mode == "slice"
            )
            + observed_bridge_slice_updates
        )
        rebuild_slice_updates = (
            NUM_GIBBS_SWEEPS
            * sum(
                steps
                for steps, rebuild, mode in zip(
                    resolved_steps,
                    reference_flags,
                    kernel_modes,
                    strict=True,
                )
                if rebuild and mode == "slice"
            )
            + observed_rebuild_bridge_updates
        )
        cache_slice_updates = (
            NUM_GIBBS_SWEEPS
            * sum(
                steps
                for steps, rebuild, mode in zip(
                    resolved_steps,
                    reference_flags,
                    kernel_modes,
                    strict=True,
                )
                if not rebuild and mode == "slice"
            )
            + observed_cache_bridge_updates
        )
        periodic_attempts = NUM_GIBBS_SWEEPS * sum(
            steps
            for steps, mode in zip(resolved_steps, kernel_modes, strict=True)
            if mode == "periodic-uniform-independence"
        )
        rebuild_periodic_attempts = NUM_GIBBS_SWEEPS * sum(
            steps
            for steps, rebuild, mode in zip(
                resolved_steps,
                reference_flags,
                kernel_modes,
                strict=True,
            )
            if rebuild and mode == "periodic-uniform-independence"
        )
        cache_hit_periodic_attempts = periodic_attempts - rebuild_periodic_attempts
        rebuild_complementary_attempts = (
            observed_complementary_attempts
            if observed_complementary_insert_after_block is not None
            and reference_flags[observed_complementary_insert_after_block]
            else 0
        )
        cache_hit_complementary_attempts = (
            observed_complementary_attempts - rebuild_complementary_attempts
        )
        cache_segments = 1 + sum(
            left != right for left, right in pairwise(reference_flags)
        )
        observed = {
            "blocks": [list(block) for block in blocks],
            "sampling_parameter_names": list(sampling_parameter_names),
            "block_sizes": block_sizes,
            "rebuild_required_by_block": rebuild_by_likelihood,
            "total_dimensions": len(sampling_parameter_names),
            "num_gibbs_sweeps": NUM_GIBBS_SWEEPS,
            "num_inner_steps_per_dim": NUM_INNER_STEPS_PER_DIM,
            "num_slice_steps_by_block": (
                list(configured_steps) if configured_steps is not None else None
            ),
            "block_kernel_modes": kernel_modes,
            "resolved_slice_steps_by_block": resolved_steps,
            "total_updates": total_updates,
            "total_slice_updates": total_slice_updates,
            "waveform_rebuild_slice_updates": rebuild_slice_updates,
            "cache_hit_slice_updates": cache_slice_updates,
            "periodic_independence_attempts": periodic_attempts,
            "waveform_rebuild_periodic_independence_attempts": (
                rebuild_periodic_attempts
            ),
            "cache_hit_periodic_independence_attempts": (cache_hit_periodic_attempts),
            "complementary_de_attempts": observed_complementary_attempts,
            "waveform_rebuild_complementary_de_attempts": (
                rebuild_complementary_attempts
            ),
            "cache_hit_complementary_de_attempts": (cache_hit_complementary_attempts),
            "complementary_de_jump_block": (
                {
                    "parameters": list(observed_complementary_parameters),
                    "attempts": observed_complementary_attempts,
                }
                if resolved_complementary_de is not None
                else None
            ),
            "complementary_de_gamma": (
                1.0 if resolved_complementary_de is not None else None
            ),
            "complementary_de_insert_after_block": (
                observed_complementary_insert_after_block
            ),
            "cache_segments": cache_segments,
            "direction_mode": sampler_config.direction_mode,
            "q_block_index": q_block_index,
            "per_likelihood": per_likelihood,
        }
        if sampler_config.bridge_blocks:
            observed.update(
                {
                    "bridge_blocks": [
                        list(block) for block in sampler_config.bridge_blocks
                    ],
                    "bridge_rebuild_required_by_block": bridge_rebuild_by_likelihood,
                    "bridge_slice_updates": observed_bridge_slice_updates,
                    "waveform_rebuild_bridge_slice_updates": (
                        observed_rebuild_bridge_updates
                    ),
                    "cache_hit_bridge_slice_updates": observed_cache_bridge_updates,
                }
            )
        fixed_work_matches = all(
            (
                observed["blocks"] == expected["blocks"],
                observed["block_sizes"] == expected["block_sizes"],
                observed["total_dimensions"] == expected["total_dimensions"],
                observed["num_gibbs_sweeps"] == expected["num_gibbs_sweeps"],
                observed["num_inner_steps_per_dim"]
                == expected["num_inner_steps_per_dim"],
                observed["num_slice_steps_by_block"]
                == expected["num_slice_steps_by_block"],
                observed["block_kernel_modes"] == expected["block_kernel_modes"],
                observed["resolved_slice_steps_by_block"]
                == expected["resolved_slice_steps_by_block"],
                observed["total_updates"] == expected["total_updates"],
                observed["total_slice_updates"] == expected["total_slice_updates"],
                observed["waveform_rebuild_slice_updates"]
                == expected["waveform_rebuild_slice_updates"],
                observed["cache_hit_slice_updates"]
                == expected["cache_hit_slice_updates"],
                observed["periodic_independence_attempts"]
                == expected["periodic_independence_attempts"],
                observed["waveform_rebuild_periodic_independence_attempts"]
                == expected["waveform_rebuild_periodic_independence_attempts"],
                observed["cache_hit_periodic_independence_attempts"]
                == expected["cache_hit_periodic_independence_attempts"],
                observed["complementary_de_attempts"]
                == expected["complementary_de_attempts"],
                observed["waveform_rebuild_complementary_de_attempts"]
                == expected["waveform_rebuild_complementary_de_attempts"],
                observed["cache_hit_complementary_de_attempts"]
                == expected["cache_hit_complementary_de_attempts"],
                observed["complementary_de_jump_block"]
                == expected["complementary_de_jump_block"],
                observed["complementary_de_gamma"]
                == expected["complementary_de_gamma"],
                observed["complementary_de_insert_after_block"]
                == expected["complementary_de_insert_after_block"],
                observed["cache_segments"] == expected["cache_segments"],
                observed["direction_mode"] == expected["direction_mode"],
                all(
                    flags == expected["rebuild_required_by_block"]
                    for flags in rebuild_by_likelihood.values()
                ),
                (
                    not configured_bridge_blocks
                    or all(
                        flags == expected["bridge_rebuild_required_by_block"]
                        for flags in bridge_rebuild_by_likelihood.values()
                    )
                ),
                all(
                    observed.get(name) == expected.get(name)
                    for name in (
                        "bridge_blocks",
                        "bridge_slice_updates",
                        "waveform_rebuild_bridge_slice_updates",
                        "cache_hit_bridge_slice_updates",
                    )
                ),
            )
        )
        dependencies_match = all(
            item["eta_is_waveform_parameter"]
            and not item["eta_is_cacheable"]
            and item["q_requires_waveform_rebuild"]
            and item["q_block_requires_waveform_rebuild"]
            and not item["t_c_is_sampling_parameter"]
            and not item["t_c_is_waveform_parameter"]
            and not item["t_c_is_fixed_parameter"]
            and item["t_c_is_marginalized"]
            for item in per_likelihood.values()
        )
        return {
            "passed": fixed_work_matches and dependencies_match,
            "blocking_scheme": blocking_scheme,
            "expected": expected,
            "observed": observed,
            "note": (
                "Block cache prices are resolved by the same production "
                "dependency inference used by Jim. Sampling q maps to waveform "
                "eta and must rebuild; t_c is absent because time is marginalized."
            ),
        }
    except (
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return {
            "passed": False,
            "blocking_scheme": blocking_scheme,
            "expected": expected,
            "error": f"{type(error).__name__}: {error}",
        }


def _construct_preflight_likelihoods(
    data_path: Path,
    reference: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from benchmarks.device_parallel_nss import benchmark_gw170817_full_run as benchmark
    from benchmarks.device_parallel_nss.paper_heterodyne import (
        PaperTimeMarginalizedHeterodynedLikelihoodFD,
    )
    from benchmarks.device_parallel_nss.paper_model import (
        RippleIMRPhenomPv2NRTidalv2,
    )
    from benchmarks.device_parallel_nss.preflight_gw170817_likelihood_pair import (
        _construct_full_likelihood_identical_grid,
    )
    from jimgw.core.single_event.data import Data, PowerSpectrum
    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1

    _assert_fixed_sampler_contract(benchmark)
    _, arrays = benchmark._read_bundle(data_path, WORKLOAD)
    ifos = [get_H1(), get_L1(), get_V1()]
    for ifo in ifos:
        strain, psd = benchmark._likelihood_inputs_from_bundle(
            ifo.name,
            arrays,
            Data=Data,
            PowerSpectrum=PowerSpectrum,
            jnp=jnp,
        )
        ifo.set_data(strain)
        ifo.set_psd(psd)
    components = benchmark._analysis_components(
        WORKLOAD,
        jnp,
        ifos,
        blocking_scheme=BLOCKING_SCHEME,
    )
    waveform = RippleIMRPhenomPv2NRTidalv2(
        f_ref=WAVEFORM_F_REF_HZ,
        time_anchor=CARRIER_TIME_ANCHOR,
    )
    common = {
        "waveform": waveform,
        "trigger_time": benchmark.GPS,
        "f_min": benchmark.F_MIN,
        "f_max": benchmark.F_MAX,
        "phase_marginalization": True,
    }
    full = _construct_full_likelihood_identical_grid(
        detectors=ifos,
        **common,
        tc_range=TC_RANGE_SECONDS,
        upsample_factor=TIME_UPSAMPLE_FACTOR,
    )
    compressed = PaperTimeMarginalizedHeterodynedLikelihoodFD(
        ifos,
        **common,
        time_marginalization={
            "tc_range": TC_RANGE_SECONDS,
            "upsample_factor": TIME_UPSAMPLE_FACTOR,
        },
        n_bins=N_BINS_REQUESTED,
        reference_parameters=dict(reference["likelihood_parameters"]),
    )
    u16_full = _construct_full_likelihood_identical_grid(
        detectors=ifos,
        **common,
        tc_range=TC_RANGE_SECONDS,
        upsample_factor=16,
    )
    source_u1_full = _construct_full_likelihood_identical_grid(
        detectors=ifos,
        **common,
        tc_range=TC_RANGE_SECONDS,
        upsample_factor=1,
    )
    fixed_work_check = _fixed_work_cache_schedule_check(
        full,
        compressed,
        components,
        blocking_scheme=BLOCKING_SCHEME,
    )
    return jax, full, compressed, u16_full, source_u1_full, fixed_work_check


def _produce_optional_preflight_matrices(
    jax: Any,
    full: Any,
    compressed: Any,
    reference: Mapping[str, Any],
    *,
    blocking_scheme: str,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Produce raw and checked periodic/cDE matrices required by a scheme."""

    run_periodic, run_complementary_de = _preflight_probe_families(blocking_scheme)
    periodic_report: dict[str, Any] | None = None
    periodic_check: dict[str, Any] | None = None
    if run_periodic:
        periodic_parent, own_points, foreign_points = _periodic_move_points(reference)
        full_values = _evaluate_likelihood_paths(jax, full, own_points)
        compressed_values = _evaluate_likelihood_paths(jax, compressed, own_points)
        full_foreign = _evaluate_foreign_cache_paths(
            jax,
            full,
            periodic_parent,
            foreign_points,
        )
        compressed_foreign = _evaluate_foreign_cache_paths(
            jax,
            compressed,
            periodic_parent,
            foreign_points,
        )
        periodic_check = _periodic_move_cache_parity_check(
            full_values,
            compressed_values,
            full_foreign,
            compressed_foreign,
        )
        periodic_report = {
            "parent_parameters": periodic_parent,
            "points": [
                {
                    "name": name,
                    "parameters": parameters,
                    FULL.kind: full_values[name],
                    HETERODYNE.kind: compressed_values[name],
                }
                for name, parameters in own_points
            ],
            "psi_foreign_cache": {
                FULL.kind: full_foreign["periodic-move-psi"],
                HETERODYNE.kind: compressed_foreign["periodic-move-psi"],
            },
        }

    complementary_de_report: dict[str, Any] | None = None
    complementary_de_check: dict[str, Any] | None = None
    if run_complementary_de:
        point, contract = _complementary_de_proposal_point(
            reference,
            blocking_scheme=blocking_scheme,
        )
        points = [("complementary-de-proposal", point)]
        full_values = _evaluate_likelihood_paths(jax, full, points)
        compressed_values = _evaluate_likelihood_paths(jax, compressed, points)
        complementary_de_check = _complementary_de_proposal_cache_parity_check(
            full_values,
            compressed_values,
            contract,
        )
        complementary_de_report = {
            "proposal_contract": contract,
            "point": {
                "name": "complementary-de-proposal",
                "likelihood_parameters": point,
                FULL.kind: full_values["complementary-de-proposal"],
                HETERODYNE.kind: compressed_values["complementary-de-proposal"],
            },
        }
    return (
        periodic_report,
        periodic_check,
        complementary_de_report,
        complementary_de_check,
    )


def _run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = _artifact_paths(args.prefix, args.mode)["preflight"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing preflight: {output}")
    data_path = args.data_file.expanduser().resolve()
    reference_path = args.reference_file.expanduser().resolve()
    reference, reference_sha256, data_sha256 = _validate_inputs_before_cells(
        data_path,
        reference_path,
    )
    source = _source_manifest(_repository())
    if args.simulate_cpu:
        from benchmarks.device_parallel_nss import (
            benchmark_gw170817_full_run as benchmark,
        )

        benchmark._configure_cpu_simulation(1)
    (
        jax,
        full,
        compressed,
        u16_full,
        source_u1_full,
        fixed_work_check,
    ) = _construct_preflight_likelihoods(data_path, reference)
    local_points = _stress_points(reference)
    probe_points, probe_manifests, probe_top_names = _load_probe_bank(args.probe_file)
    points = [*local_points, *probe_points]
    parity_names = {"reference", *probe_top_names}
    full_values = _evaluate_likelihood_paths(
        jax,
        full,
        points,
        parity_names=parity_names,
    )
    compressed_values = _evaluate_likelihood_paths(
        jax,
        compressed,
        points,
        parity_names=parity_names,
    )
    foreign_cache_parent, foreign_cache_targets = _foreign_cache_ridge_points(reference)
    full_foreign_cache_values = _evaluate_foreign_cache_paths(
        jax,
        full,
        foreign_cache_parent,
        foreign_cache_targets,
    )
    compressed_foreign_cache_values = _evaluate_foreign_cache_paths(
        jax,
        compressed,
        foreign_cache_parent,
        foreign_cache_targets,
    )
    foreign_cache_check = _foreign_cache_parity_check(
        full_foreign_cache_values,
        compressed_foreign_cache_values,
    )
    (
        periodic_move_report,
        periodic_move_check,
        complementary_de_report,
        complementary_de_check,
    ) = _produce_optional_preflight_matrices(
        jax,
        full,
        compressed,
        reference,
        blocking_scheme=BLOCKING_SCHEME,
    )
    netsky_polarization_report: dict[str, Any] | None = None
    netsky_polarization_check: dict[str, Any] | None = None
    if BLOCKING_SCHEME == NETSKY_SCHEME:
        polarization_parent, polarization_points = _netsky_polarization_identity_points(
            reference
        )
        full_polarization_values = _evaluate_foreign_cache_paths(
            jax,
            full,
            polarization_parent,
            polarization_points,
        )
        compressed_polarization_values = _evaluate_foreign_cache_paths(
            jax,
            compressed,
            polarization_parent,
            polarization_points,
        )
        netsky_polarization_check = _netsky_polarization_identity_check(
            full_polarization_values,
            compressed_polarization_values,
        )
        netsky_polarization_report = {
            "parent_parameters": polarization_parent,
            "points": [
                {
                    "name": name,
                    "parameters": parameters,
                    FULL.kind: full_polarization_values[name],
                    HETERODYNE.kind: compressed_polarization_values[name],
                }
                for name, parameters in polarization_points
            ],
        }
    cache_dependency_check = _waveform_cache_dependency_check(full, compressed)
    q_time_grid_points = _q_time_grid_points(reference)
    q_time_grid_full_u8_values = _evaluate_likelihood_paths(
        jax,
        full,
        q_time_grid_points,
    )
    q_time_grid_compressed_u8_values = _evaluate_likelihood_paths(
        jax,
        compressed,
        q_time_grid_points,
    )
    q_time_grid_u16_values = _evaluate_direct(jax, u16_full, q_time_grid_points)
    u16_values = _evaluate_direct(jax, u16_full, points)
    source_u1_value = _evaluate_direct(
        jax,
        source_u1_full,
        [local_points[0]],
    )["reference"]
    full_direct = {name: values["direct"] for name, values in full_values.items()}
    maximum_full = max(full_direct.values())
    q_time_grid_check = _q_time_grid_regression_check(
        q_time_grid_points,
        q_time_grid_full_u8_values,
        q_time_grid_compressed_u8_values,
        q_time_grid_u16_values,
        bank_max_log_likelihood=maximum_full,
    )
    point_reports: list[dict[str, Any]] = []
    all_parity_errors: list[float] = []
    beta_values: list[float] = []
    near_max_errors: list[float] = []
    local_names = {name for name, _ in local_points}
    for name, _ in points:
        full_paths = full_values[name]
        compressed_paths = compressed_values[name]
        full_parity = max(
            abs(value - full_paths["direct"]) for value in full_paths.values()
        )
        compressed_parity = max(
            abs(value - compressed_paths["direct"])
            for value in compressed_paths.values()
        )
        approximation_delta = compressed_paths["direct"] - full_paths["direct"]
        all_parity_errors.extend((full_parity, compressed_parity))
        log_likelihood_drop = maximum_full - full_paths["direct"]
        beta: float | None = None
        if log_likelihood_drop >= MIN_BETA_LOGL_DROP:
            beta = abs(approximation_delta) / log_likelihood_drop
            beta_values.append(beta)
        else:
            near_max_errors.append(abs(approximation_delta))
        point_reports.append(
            {
                "name": name,
                "source": "local" if name in local_names else "independent-probe",
                "full": full_paths,
                "heterodyne5000": compressed_paths,
                "full_path_max_abs_delta": full_parity,
                "heterodyne_path_max_abs_delta": compressed_parity,
                "heterodyne_minus_full_log_likelihood": approximation_delta,
                "full_log_likelihood_drop_from_bank_max": log_likelihood_drop,
                "relative_binning_beta": beta,
            }
        )
    source_reference_delta = source_u1_value - float(
        reference["corrected_u1_log_likelihood"]
    )
    reference_approximation = abs(
        compressed_values["reference"]["direct"] - full_values["reference"]["direct"]
    )
    stress_approximation = max(
        abs(compressed_values[name]["direct"] - full_values[name]["direct"])
        for name, _ in local_points
        if name != "reference"
    )
    u8_u16_point_bank_check = _u8_u16_point_bank_check(
        points,
        full_values,
        u16_values,
        local_names=local_names,
    )
    selection_u16_delta = u16_values["reference"] - float(
        reference["selection_u16_log_likelihood"]
    )
    fiducial_gap = maximum_full - full_values["reference"]["direct"]
    beta_p99 = float(np.quantile(beta_values, 0.99)) if beta_values else None
    beta_max = max(beta_values) if beta_values else None
    near_max_error = max(near_max_errors) if near_max_errors else None
    top_q_values = [item["stored_top_q"] for item in probe_manifests]
    top_q_span = max(top_q_values) - min(top_q_values) if top_q_values else 0.0
    checks = {
        "canonical_data_pinned": {
            "passed": data_sha256 == CANONICAL_DATA_SHA256,
            "expected_sha256": CANONICAL_DATA_SHA256,
            "observed_sha256": data_sha256,
        },
        "independent_cross_mode_probe_bank": {
            "passed": (
                len(probe_manifests) >= MIN_INDEPENDENT_PROBE_FILES
                and top_q_span >= 0.05
            ),
            "minimum_distinct_files": MIN_INDEPENDENT_PROBE_FILES,
            "observed_distinct_files": len(probe_manifests),
            "minimum_stored_top_q_span": 0.05,
            "observed_stored_top_q_span": top_q_span,
            "note": (
                "Stored likelihood values are not compared because these probes "
                "used an older carrier anchor; only their coordinates are reused."
            ),
        },
        "source_u1_reference_reproduced": {
            "passed": abs(source_reference_delta) <= REFERENCE_SOURCE_ATOL,
            "abs_tolerance": REFERENCE_SOURCE_ATOL,
            "delta": source_reference_delta,
            "stock_u1_log_likelihood": source_u1_value,
            "corrected_u8_log_likelihood": full_values["reference"]["direct"],
        },
        "selection_u16_reference_reproduced": {
            "passed": abs(selection_u16_delta) <= REFERENCE_SOURCE_ATOL,
            "abs_tolerance": REFERENCE_SOURCE_ATOL,
            "delta": selection_u16_delta,
            "stock_u16_log_likelihood": u16_values["reference"],
        },
        "direct_cache_jit_parity": {
            "passed": max(all_parity_errors) <= PARITY_ATOL,
            "abs_tolerance": PARITY_ATOL,
            "max_abs_delta": max(all_parity_errors),
        },
        "waveform_cache_dependency_contract": cache_dependency_check,
        "fixed_work_cache_schedule": fixed_work_check,
        "foreign_parent_ridge_cache_parity": foreign_cache_check,
        "reference_compression_accuracy": {
            "passed": reference_approximation <= MAX_REFERENCE_ABS_LOGL_DELTA,
            "max_abs_log_likelihood_delta": MAX_REFERENCE_ABS_LOGL_DELTA,
            "observed_abs_delta": reference_approximation,
        },
        "local_stress_compression_accuracy": {
            "passed": stress_approximation <= MAX_STRESS_ABS_LOGL_DELTA,
            "max_abs_log_likelihood_delta": MAX_STRESS_ABS_LOGL_DELTA,
            "observed_abs_delta": stress_approximation,
        },
        "source_paper_relative_binning_beta": {
            "passed": (
                bool(beta_values)
                and beta_p99 is not None
                and beta_p99 <= MAX_RELATIVE_BINNING_BETA
                and beta_max is not None
                and beta_max <= MAX_RELATIVE_BINNING_BETA
            ),
            "minimum_log_likelihood_drop": MIN_BETA_LOGL_DROP,
            "maximum_beta": MAX_RELATIVE_BINNING_BETA,
            "observed_p99": beta_p99,
            "observed_max": beta_max,
            "point_count": len(beta_values),
        },
        "near_max_absolute_compression_accuracy": {
            "passed": (
                bool(near_max_errors)
                and near_max_error is not None
                and near_max_error <= MAX_REFERENCE_ABS_LOGL_DELTA
            ),
            "maximum_abs_log_likelihood_delta": MAX_REFERENCE_ABS_LOGL_DELTA,
            "observed_max_abs_delta": near_max_error,
            "point_count": len(near_max_errors),
        },
        "fiducial_within_point_one_nat_of_bank_max": {
            "passed": fiducial_gap <= MAX_FIDUCIAL_LOGL_GAP,
            "maximum_gap": MAX_FIDUCIAL_LOGL_GAP,
            "observed_gap": fiducial_gap,
            "bank_max_log_likelihood": maximum_full,
            "fiducial_log_likelihood": full_values["reference"]["direct"],
        },
        "u8_u16_time_grid_convergence": u8_u16_point_bank_check,
        **q_time_grid_check["checks"],
    }
    if periodic_move_check is not None:
        checks["periodic_move_direct_cache_jit_parity"] = periodic_move_check
    if complementary_de_check is not None:
        checks["complementary_de_proposal_direct_cache_jit_parity"] = (
            complementary_de_check
        )
    if netsky_polarization_check is not None:
        checks["netsky_polarization_half_period_identity"] = netsky_polarization_check
    strict_pass = all(check["passed"] for check in checks.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "name": "jim-paper-15d-likelihood-preflight",
        "strict_pass": strict_pass,
        "scientific_run": False,
        "data": {"path": str(data_path), "sha256": data_sha256},
        "reference": {
            "path": str(reference_path),
            "json_sha256": reference_sha256,
            "source_path": reference["source_path"],
            "source_sha256": reference["source_sha256"],
            "source_log_likelihood": reference["source_log_likelihood"],
            "corrected_u1_log_likelihood": reference["corrected_u1_log_likelihood"],
            "selection_u16_log_likelihood": reference["selection_u16_log_likelihood"],
        },
        "independent_probe_files": probe_manifests,
        "waveform": {
            "model": WAVEFORM,
            "f_ref_hz": WAVEFORM_F_REF_HZ,
            "time_anchor": CARRIER_TIME_ANCHOR,
        },
        "full_likelihood": _likelihood_metadata(FULL, full, reference_sha256),
        "heterodyne_likelihood": _likelihood_metadata(
            HETERODYNE,
            compressed,
            reference_sha256,
        ),
        "stress_set": (
            "Frozen reference, three deterministic local perturbations, and "
            "deterministic weighted-systematic coordinates plus the top-logL "
            "coordinate from each independent probe artifact. This is a "
            "construction/approximation gate, not sampler validation."
        ),
        "points": point_reports,
        "foreign_parent_ridge_cache": {
            "parent_parameters": foreign_cache_parent,
            "targets": [
                {
                    "name": name,
                    "parameters": params,
                    FULL.kind: full_foreign_cache_values[name],
                    HETERODYNE.kind: compressed_foreign_cache_values[name],
                }
                for name, params in foreign_cache_targets
            ],
        },
        "periodic_move_cache_parity": periodic_move_report,
        "complementary_de_proposal_cache_parity": complementary_de_report,
        "netsky_polarization_identity": netsky_polarization_report,
        "q_time_grid": {
            "comparison_max_log_likelihood": q_time_grid_check[
                "comparison_max_log_likelihood"
            ],
            "per_point": q_time_grid_check["per_point"],
            "note": q_time_grid_check["note"],
        },
        "checks": checks,
        "source": source,
        "jax": {
            "version": jax.__version__,
            "backend": jax.default_backend(),
            "enable_x64": bool(jax.config.jax_enable_x64),
        },
    }
    after = _source_manifest(_repository())
    if after["sha256"] != source["sha256"]:
        raise RuntimeError("source changed during likelihood preflight")
    _atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_runtime(args)
    if args._cell_kind is not None:
        _run_cell(args, ARM_BY_KIND[args._cell_kind])
        return
    if args.preflight_only:
        report = _run_preflight(args)
        if not report["strict_pass"]:
            raise SystemExit(2)
        return
    summary = _run_pair(args)
    pair_path = _artifact_paths(args.prefix, args.mode)["pair"]
    _atomic_write_json(pair_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    if not summary["comparison"]["strict_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
