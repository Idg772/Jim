"""Independent native probe sums on chunks already loaded for heterodyning.

No moment interpolation or fast network response enters these sums. The stock
waveform and detector's independent complex exponential response evaluate every
real native sample, including final endpoints and samples outside bin support.
"""

import hashlib
import json
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import i0e

from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.time_utils import greenwich_mean_sidereal_time
from jimgw.core.single_event.utils import apply_fixed_parameters
from jimgw.core.single_event.xg_waveform import supports_source

MAX_NATIVE_PROBES = 9


def prepare_native_probe_parameters(
    points, *, trigger_time, phase_marginalization, fixed_parameters=None
):
    """Normalize a bounded bank exactly as the independent native reference does."""
    if not 1 <= len(points) <= MAX_NATIVE_PROBES:
        raise ValueError(f"native probe bank requires 1..{MAX_NATIVE_PROBES} points")
    fixed_parameters = fixed_parameters or {}
    if phase_marginalization and "phase_c" in fixed_parameters:
        raise ValueError("native probes cannot fix and marginalize phase_c")
    result = []
    for point in points:
        if not isinstance(point, Mapping):
            raise TypeError("native probe parameters must be mappings")
        prepared = dict(point)
        prepared["trigger_time"] = trigger_time
        prepared["gmst"] = greenwich_mean_sidereal_time(trigger_time)
        if phase_marginalization:
            prepared["phase_c"] = 0.0
        apply_fixed_parameters(prepared, fixed_parameters)
        normalized = {}
        for key, value in prepared.items():
            array = np.asarray(value)
            if (
                not isinstance(key, str)
                or array.ndim != 0
                or not np.isrealobj(array)
                or not np.isfinite(array)
            ):
                raise ValueError("native probe parameters must be finite real scalars")
            normalized[key] = float(array)
        result.append(normalized)
    if any(set(point) != set(result[0]) for point in result):
        raise ValueError("native probe points must have identical parameter keys")
    return result


def native_probe_parameters_sha256(parameters):
    encoded = json.dumps(
        parameters, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


class NativeProbeBank:
    """Immutable probe inputs and bounded independent reducers for each detector."""

    def __init__(self, likelihood, waveform, points):
        if not (
            type(waveform) is DominantModeTimeCachedWaveform
            and supports_source(waveform.source)
        ):
            raise ValueError(
                "native probes require the supported stock waveform and clock"
            )
        if likelihood.time_marginalization or likelihood.distance_marginalization:
            raise ValueError("native probes support only fixed or marginalized phase")
        self.waveform = waveform
        self.parameters = prepare_native_probe_parameters(
            points,
            trigger_time=likelihood.trigger_time,
            phase_marginalization=likelihood.phase_marginalization,
            fixed_parameters=likelihood.fixed_parameters,
        )
        self.parameter_hash = native_probe_parameters_sha256(self.parameters)
        self.phase_marginalization = likelihood.phase_marginalization
        self.count = len(self.parameters)
        required = set(waveform.parameter_names) | {"ra", "dec", "psi", "t_c"}
        if required - set(self.parameters[0]):
            raise ValueError(
                "native probe parameters omit required waveform/response inputs"
            )
        self.detectors = tuple(likelihood.detectors)
        self.bands = {}
        prefixes = []
        for detector in self.detectors:
            f, d, psd = (
                detector.sliced_frequencies,
                detector.sliced_fd_data,
                detector.sliced_psd,
            )
            if (
                any(array.ndim != 1 for array in (f, d, psd))
                or len(f) < 2
                or len(f) != len(d)
                or len(f) != len(psd)
            ):
                raise ValueError(
                    "native probes require matching one-dimensional bands with at least two samples"
                )
            first_two = np.asarray(jax.device_get(f[:2]), dtype=np.float64)
            last = float(jax.device_get(f[-1]))
            df = float(first_two[1] - first_two[0])
            if not np.all(np.isfinite(first_two)) or not math.isfinite(last) or df <= 0:
                raise ValueError("native probe frequency metadata are invalid")
            prefixes.append(first_two)
            self.bands[detector.name] = {
                "frequency_min": float(first_two[0]),
                "frequency_max": last,
                "df": df,
                "native_samples": len(f),
            }
        if len(self.bands) != len(self.detectors):
            raise ValueError("native probes require unique detector names")
        self.prefix = np.unique(np.concatenate(prefixes))[:2]
        spacing = self.prefix[1] - self.prefix[0]
        for band in self.bands.values():
            offset = (band["frequency_min"] - self.prefix[0]) / band["df"]
            if not np.isclose(
                band["df"], spacing, rtol=1e-10, atol=0
            ) or not np.isclose(offset, round(offset), rtol=0, atol=1e-6):
                raise ValueError(
                    "native probes require aligned native grids with equal spacing"
                )

    def initial_state(self, device):
        return jax.device_put(
            (
                np.zeros((self.count, 3)),
                np.zeros((self.count, 3)),
                np.ones(self.count, dtype=bool),
            ),
            device,
        )

    def make_reducer(self, detector, device):
        """Compile once per detector; padded tails share the same executable."""
        prefix = jax.device_put(self.prefix, device)
        parameters = jax.device_put(
            {
                key: np.asarray([point[key] for point in self.parameters])
                for key in self.parameters[0]
            },
            device,
        )
        band = self.bands[detector.name]
        df = band["df"]

        @jax.jit
        def accumulate(frequency, data, psd, start, count, state):
            total, compensation, previous_valid = state
            real_sample = jnp.arange(frequency.size) < count
            expected = band["frequency_min"] + (start + jnp.arange(frequency.size)) * df
            tolerance = (
                32 * np.finfo(np.float64).eps * jnp.maximum(1, jnp.abs(expected))
            )
            valid_samples = jnp.all(
                jnp.where(
                    real_sample,
                    jnp.isfinite(frequency)
                    & (jnp.abs(frequency - expected) <= tolerance)
                    & jnp.isfinite(data)
                    & jnp.isfinite(psd)
                    & (psd > 0),
                    True,
                )
            )
            # Repeat a real sample for padding so each detector stays inside
            # its own orbital contract. The separate prefix still pins the
            # network waveform spacing; the count mask excludes every repeat.
            f = jnp.where(real_sample, frequency, frequency[0])
            d = jnp.where(real_sample, data, 0.0j)
            s = jnp.where(real_sample, psd, 1.0)

            def evaluate(point):
                sky = self.waveform(jnp.concatenate((prefix, f)), point)
                sky = jax.tree.map(lambda value: value[2:], sky)
                h = detector.fd_response(f, sky, point, optimize=False)
                valid = jnp.all(jnp.where(real_sample, jnp.isfinite(h), True))
                h = jnp.where(real_sample, h, 0.0j)
                z = (4.0 * df) * jnp.sum(jnp.conj(h) * d / s)
                q = (4.0 * df) * jnp.sum((h.real**2 + h.imag**2) / s)
                values = jnp.stack((z.real, z.imag, q))
                return values, valid & jnp.all(jnp.isfinite(values))

            values, valid = jax.vmap(evaluate)(parameters)
            adjusted = values - compensation
            updated = total + adjusted
            compensation = (updated - total) - adjusted
            return updated, compensation, previous_valid & valid & valid_samples

        return accumulate

    def channel_result(self, detector, state, *, chunks, chunk_size):
        values, _, valid = jax.device_get(state)
        if not np.all(valid) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"invalid native probe samples or waveform for {detector.name}"
            )
        return {
            **self.bands[detector.name],
            "chunks": chunks,
            "chunk_size": chunk_size,
            "sums": np.asarray(values).tolist(),
        }

    def results(self, diagnostics):
        results = []
        for index in range(self.count):
            total = np.zeros(3)
            channels = {}
            for detector in self.detectors:
                channel = dict(diagnostics[detector.name]["native_probes"])
                values = np.asarray(channel.pop("sums"))[index]
                total += values
                channel["complex_overlap"] = {
                    "real": float(values[0]),
                    "imag": float(values[1]),
                }
                channel["waveform_norm"] = float(values[2])
                channels[detector.name] = channel
            magnitude = math.hypot(total[0], total[1])
            match = (
                float(np.log(i0e(magnitude)) + magnitude)
                if self.phase_marginalization
                else float(total[0])
            )
            value = match - 0.5 * float(total[2])
            if not math.isfinite(value):
                raise ValueError("nonfinite coherent native probe likelihood")
            results.append(
                {
                    "log_likelihood": value,
                    "complex_overlap": {
                        "real": float(total[0]),
                        "imag": float(total[1]),
                    },
                    "waveform_norm": float(total[2]),
                    "marginalization": "phase"
                    if self.phase_marginalization
                    else "fixed",
                    "native_prefix": self.prefix.tolist(),
                    "detector_phasor_optimized": False,
                    "channels": channels,
                    "qualification": False,
                    "fused_native_summary_stream": True,
                    "native_samples_reread": 0,
                }
            )
        return results
