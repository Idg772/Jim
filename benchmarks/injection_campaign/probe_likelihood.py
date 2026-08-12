"""Evaluate one frozen injection truth without constructing or running a sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.injection_campaign.common import load_manifest, read_catalogue
from benchmarks.injection_campaign.run_injection import (
    _analysis_components,
    _injection_parameters,
    _transient_likelihood_kwargs,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("injection_id", type=int)
    return parser.parse_args(argv)


def probe(campaign_dir: Path, injection_id: int) -> dict[str, object]:
    campaign_dir = campaign_dir.expanduser().resolve()
    manifest = load_manifest(campaign_dir)
    catalogue = read_catalogue(campaign_dir / manifest["catalogue"]["path"])
    if not 0 <= injection_id < int(manifest["n_injections"]):
        raise ValueError("injection_id is outside the selected campaign")
    truth = catalogue[injection_id]
    config = manifest["config"]

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import jimgw
    from jimgw.core.single_event.data import PowerSpectrum
    from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
    from jimgw.core.single_event.likelihood import TransientLikelihoodFD

    ifos = [get_H1(), get_L1(), get_V1()]
    for ifo in ifos:
        relative = manifest["psd"]["detector_files"][ifo.name]
        ifo.set_psd(PowerSpectrum.from_file(str(campaign_dir / relative)))
    components = _analysis_components(config, jnp, ifos)
    parameters = _injection_parameters(truth, components["likelihood_transforms"])
    duration = float(config["duration_seconds"])
    start_offset = float(config["segment_start_offset_seconds"])
    noise_key = jax.random.key(truth["noise_seed"])
    for detector_index, ifo in enumerate(ifos):
        ifo.inject_signal(
            duration=duration,
            sampling_frequency=float(config["sampling_frequency_hz"]),
            trigger_time=float(config["trigger_time_gps"]),
            waveform_model=components["waveform"],
            parameters=parameters,
            f_min=float(config["f_min_hz"]),
            f_max=float(config["f_max_hz"]),
            start_time=float(config["trigger_time_gps"]) + start_offset,
            zero_noise=False,
            rng_key=jax.random.fold_in(noise_key, detector_index),
        )
    likelihood = TransientLikelihoodFD(
        ifos,
        waveform=components["waveform"],
        **_transient_likelihood_kwargs(config, components),
    )
    value = likelihood.evaluate(parameters)
    value.block_until_ready()
    return {
        "config_sha256": manifest["config_sha256"],
        "injection_id": injection_id,
        "noise_seed": truth["noise_seed"],
        "sampler_seed": truth["sampler_seed"],
        "log_likelihood_at_truth": float(value),
        "optimal_snr_by_detector": {ifo.name: float(ifo.optimal_snr) for ifo in ifos},
        "jimgw_module": str(Path(jimgw.__file__).resolve()),
        "backend": jax.default_backend(),
        "device_count": jax.local_device_count(),
        "finite": bool(np.isfinite(float(value))),
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    print(json.dumps(probe(args.campaign_dir, args.injection_id), sort_keys=True))


if __name__ == "__main__":
    main()
