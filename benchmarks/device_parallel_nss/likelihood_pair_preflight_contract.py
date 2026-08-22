"""Shared optional-probe contract for likelihood-pair preflights."""

from __future__ import annotations

H4_BLOCKING_SCHEME = "fast-ridge-intrinsic-periodic-mh"
H5_BLOCKING_SCHEME = "fast-ridge-intrinsic-periodic-mh-cde4"
H6_BLOCKING_SCHEME = "fast-ridge-intrinsic-periodic-mh-cde8"
PERIODIC_BLOCKING_SCHEMES = {H4_BLOCKING_SCHEME, H5_BLOCKING_SCHEME, H6_BLOCKING_SCHEME}
COMPLEMENTARY_DE_BLOCKING_SCHEMES = {H5_BLOCKING_SCHEME, H6_BLOCKING_SCHEME}


def preflight_probe_families(blocking_scheme: str) -> tuple[bool, bool]:
    """Return whether periodic and complementary-DE probes are required."""

    periodic = blocking_scheme in PERIODIC_BLOCKING_SCHEMES
    complementary_de = blocking_scheme in COMPLEMENTARY_DE_BLOCKING_SCHEMES
    return periodic, complementary_de
