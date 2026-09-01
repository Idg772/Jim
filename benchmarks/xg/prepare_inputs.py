"""Prepare immutable public inputs for the CE XG tracer run."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

SOURCE_REVISION = "5c15b707e1b9c90d0ef2f36d4b378124f31074a8"
CE_PSD_URL = (
    "https://raw.githubusercontent.com/NirGutt/gwRombusX/"
    f"{SOURCE_REVISION}/bilby_rom/gw/detector/noise_curves/CE_psd.txt"
)
CE_PSD_SHA256 = "a8934610ae6395a86129a70bf913d2b3469a477f872fec18c626b49dc7aa3f49"
DEFAULT_OUTPUT = Path("benchmark-results/xg-ce-4096-65536/inputs/CE_psd.txt")
LALSUITE_EPHEMERIS_BASE_URL = (
    "https://git.ligo.org/lscsoft/lalsuite/-/raw/master/lalpulsar/lib"
)
EARTH_EPHEMERIS_NAME = "earth00-40-DE405.dat.gz"
EARTH_EPHEMERIS_SHA256 = (
    "4995647b2c47617c90804ad0bc814ce42b426f1e5015a90cf939bcdd0c20ea67"
)
SUN_EPHEMERIS_NAME = "sun00-40-DE405.dat.gz"
SUN_EPHEMERIS_SHA256 = (
    "0b132dc5a712ebc16661a10cb88409e2577c16723c64f98e9b9b4d265510700f"
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _download_verified(url: str, output: Path, expected_sha256: str) -> Path:
    """Download one immutable input and reject any byte-level drift."""

    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    digest = _sha256(content)
    if digest != expected_sha256:
        raise RuntimeError(
            f"input digest mismatch: expected {expected_sha256}, received {digest}"
        )

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = output.with_suffix(f"{output.suffix}.candidate")
    candidate.write_bytes(content)
    candidate.replace(output)
    return output


def prepare_ce_psd(output: Path = DEFAULT_OUTPUT) -> Path:
    """Download the pinned CE PSD and reject any byte-level drift."""

    return _download_verified(CE_PSD_URL, output, CE_PSD_SHA256)


def prepare_ephemerides(input_dir: Path = DEFAULT_OUTPUT.parent) -> tuple[Path, Path]:
    """Download the pinned DE405 Earth and Sun ephemerides."""

    earth = _download_verified(
        f"{LALSUITE_EPHEMERIS_BASE_URL}/{EARTH_EPHEMERIS_NAME}",
        input_dir / EARTH_EPHEMERIS_NAME,
        EARTH_EPHEMERIS_SHA256,
    )
    sun = _download_verified(
        f"{LALSUITE_EPHEMERIS_BASE_URL}/{SUN_EPHEMERIS_NAME}",
        input_dir / SUN_EPHEMERIS_NAME,
        SUN_EPHEMERIS_SHA256,
    )
    return earth, sun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = prepare_ce_psd(args.output)
    earth, sun = prepare_ephemerides(output.parent)
    for path, digest in (
        (output, CE_PSD_SHA256),
        (earth, EARTH_EPHEMERIS_SHA256),
        (sun, SUN_EPHEMERIS_SHA256),
    ):
        print(f"Wrote {path}")
        print(f"sha256 = {digest}")


if __name__ == "__main__":
    main()
