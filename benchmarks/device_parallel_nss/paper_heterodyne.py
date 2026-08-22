"""Paper-style time-marginalized relative-binning likelihood.

The public Jim heterodyne supports phase marginalization but not the direct-sum
coalescence-time marginalization used for the paper's GW170817 result.  This
benchmark-local subclass supplies that missing reduction without changing the
public likelihood API.  It intentionally supports only the paper comparison:

* relative binning inherited from :class:`HeterodynedTransientLikelihoodFD`;
* a uniform, explicitly upsampled time grid evaluated by direct summation;
* optional analytic phase marginalization; and
* no distance marginalization (``d_L`` remains sampled).

The time-dependent network matched filter is evaluated as two dense
matrix-vector products after summing detector coefficients. This applies the
same linear bin-edge approximation used by the parent likelihood, with the common
``exp(-2 pi i f t_c)`` detector phasor applied at every marginalization point.

The parent coefficient builder materializes a ``(n_bins, n_frequencies)``
mask.  At the paper's 5,000-bin/128-second resolution that temporary is much
larger than the likelihood itself.  This benchmark subclass computes the same
four per-bin sums with a one-dimensional bin assignment instead.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, Complex, Float

from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    SingleEventLikelihood,
)
from jimgw.core.single_event.marginalization_config import (
    PhaseMargConfig,
    TimeMargConfig,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.utils import apply_fixed_parameters
from jimgw.core.utils import log_i0
from jimgw.typing import FloatScalar


class PaperTimeMarginalizedHeterodynedLikelihoodFD(HeterodynedTransientLikelihoodFD):
    """Relative-binning likelihood with direct-sum ``t_c`` marginalization.

    ``time_marginalization`` is required. Its uniform grid, strict interval
    bounds, and ``upsample_factor`` match
    :class:`~jimgw.core.single_event.likelihood.TransientLikelihoodFD`. The
    normalization uses the full upsampled FFT-grid size, so evidence offsets
    are directly comparable between compressed and uncompressed arms.
    """

    def __init__(
        self,
        detectors: Any,
        waveform: Any,
        fixed_parameters: Any = None,
        f_min: Any = 0.0,
        f_max: Any = jnp.inf,
        trigger_time: float = 0.0,
        n_bins: int | None = None,
        epsilon: float | None = None,
        optimizer_popsize: int = 500,
        optimizer_n_steps: int = 1000,
        optimizer_target: float | None = None,
        reference_parameters: dict[str, Any] | None = None,
        reference_waveform: Any = None,
        prior: Any = None,
        likelihood_transforms: Any = None,
        phase_marginalization: PhaseMargConfig | dict[str, Any] | bool = False,
        *,
        time_marginalization: TimeMargConfig | dict[str, Any] | bool,
    ) -> None:
        if isinstance(time_marginalization, dict):
            time_config = TimeMargConfig(**time_marginalization)
        elif time_marginalization is True:
            time_config = TimeMargConfig()
        elif isinstance(time_marginalization, TimeMargConfig):
            time_config = time_marginalization
        else:
            raise ValueError(
                "paper heterodyne requires an explicit time_marginalization config"
            )
        # The paid comparison must use a frozen, reviewable reference. Do not
        # silently invoke the parent's stochastic CMA-ES setup path.
        del optimizer_popsize, optimizer_n_steps, optimizer_target
        if reference_parameters is None:
            raise ValueError("paper heterodyne requires frozen reference_parameters")
        if prior is not None or likelihood_transforms not in (None, []):
            raise ValueError(
                "paper heterodyne does not optimize a reference at construction"
            )

        # Reproduce the parent initialization with host-side frequency-grid
        # bookkeeping. JAX's generic `isin` setup can allocate prohibitive
        # intermediates for three identical 259k-point GW170817 grids.
        SingleEventLikelihood.__init__(
            self,
            detectors,
            waveform,
            fixed_parameters,
        )
        if isinstance(phase_marginalization, dict):
            phase_config = PhaseMargConfig(**phase_marginalization)
        elif phase_marginalization is True:
            phase_config = PhaseMargConfig()
        elif phase_marginalization:
            phase_config = phase_marginalization
        else:
            phase_config = None
        self.phase_marginalization = phase_config is not None
        if self.phase_marginalization and "phase_c" in self.fixed_parameters:
            raise ValueError(
                "Cannot have phase_c fixed while marginalizing over phase_c"
            )

        detector_frequencies = self._set_detector_frequency_bounds(f_min, f_max)
        host_frequencies = [
            np.asarray(jax.device_get(values)) for values in detector_frequencies
        ]
        spacings = [values[1] - values[0] for values in host_frequencies]
        if not all(np.isclose(spacings[0], spacing) for spacing in spacings[1:]):
            raise ValueError("All detectors must have the same frequency spacing")
        self.df = detector_frequencies[0][1] - detector_frequencies[0][0]
        if all(
            np.array_equal(host_frequencies[0], values)
            for values in host_frequencies[1:]
        ):
            self.frequencies = detector_frequencies[0]
            self.frequency_masks = [
                jnp.ones(len(self.frequencies), dtype=bool) for _ in detectors
            ]
            self.identical_frequency_grids = True
        else:
            merged = np.unique(np.concatenate(host_frequencies))
            self.frequencies = jnp.asarray(merged)
            self.frequency_masks = [
                jnp.asarray(np.isin(merged, values)) for values in host_frequencies
            ]
            self.identical_frequency_grids = False

        self.trigger_time = trigger_time
        self.gmst = compute_gmst(trigger_time)
        self.reference_parameters = reference_parameters.copy()
        apply_fixed_parameters(self.reference_parameters, self.fixed_parameters)
        self.reference_parameters["trigger_time"] = self.trigger_time
        self.reference_parameters["gmst"] = self.gmst
        if reference_waveform is None:
            reference_waveform = waveform

        if n_bins is not None:
            if epsilon is not None:
                raise ValueError("n_bins and epsilon are mutually exclusive")
            if n_bins <= 0:
                raise ValueError("n_bins must be positive")
        elif epsilon is None:
            epsilon = 0.5
        if epsilon is not None:
            if epsilon <= 0:
                raise ValueError("epsilon must be positive")
            phase = self._max_phase_diff(
                self.frequencies,
                self.frequencies[0],
                self.frequencies[-1],
            )
            n_bins = max(1, int(float(phase[-1]) / epsilon))
        if n_bins is None:
            raise RuntimeError("failed to resolve heterodyne bin count")
        self.requested_n_bins = n_bins
        requested_grid = self._make_binning_scheme(self.frequencies, n_bins=n_bins)
        reference_sky = reference_waveform(
            self.frequencies,
            self.reference_parameters,
        )
        masked_grid = self._mask_and_set_frequency_arrays(
            reference_sky,
            requested_grid,
        )
        reference_low = reference_waveform(
            self.freq_grid_low,
            self.reference_parameters,
        )
        reference_high = reference_waveform(
            self.freq_grid_high,
            self.reference_parameters,
        )
        self.waveform_low_ref = {}
        self.waveform_high_ref = {}
        self.summary_data = {}
        for index, detector in enumerate(self.detectors):
            detector_reference_sky = {
                key: value[self.frequency_masks[index]]
                for key, value in reference_sky.items()
            }
            detector_reference = detector.fd_response(
                detector.sliced_frequencies,
                detector_reference_sky,
                self.reference_parameters,
            )
            self.waveform_low_ref[detector.name] = detector.fd_response(
                self.freq_grid_low,
                reference_low,
                self.reference_parameters,
            )
            self.waveform_high_ref[detector.name] = detector.fd_response(
                self.freq_grid_high,
                reference_high,
                self.reference_parameters,
            )
            self.summary_data[detector.name] = self._compute_coefficients(
                detector,
                detector_reference,
                masked_grid,
            )

        if "t_c" in self.fixed_parameters:
            raise ValueError("Cannot have t_c fixed while marginalizing over t_c")

        duration = float(self.detectors[0].data.duration)
        sampling_frequency = float(self.detectors[0].data.sampling_frequency)
        n_total = int(duration * sampling_frequency / 2.0)
        tc_array = np.fft.fftfreq(n_total, 1.0 / duration)
        if time_config.upsample_factor == 1:
            tc_window = tc_array[
                (tc_array > time_config.tc_range[0])
                & (tc_array < time_config.tc_range[1])
            ]
        else:
            fine_step = duration / (n_total * time_config.upsample_factor)
            first = math.floor(time_config.tc_range[0] / fine_step) + 1
            last = math.ceil(time_config.tc_range[1] / fine_step) - 1
            tc_window = fine_step * np.arange(first, last + 1)
        if tc_window.size == 0:
            raise ValueError(
                f"time_marginalization tc_range {time_config.tc_range} contains "
                "no direct-sum time samples; widen the range"
            )

        self.time_marginalization = True
        self.tc_range = time_config.tc_range
        self.tc_upsample = time_config.upsample_factor
        self.tc_array = jnp.asarray(tc_array)
        self.tc_window = jnp.asarray(tc_window)
        self._tc_normalization_count = n_total * time_config.upsample_factor
        self._tc_phase_low = self._time_phasors(self.tc_window, self.freq_grid_low)
        self._tc_phase_high = self._time_phasors(self.tc_window, self.freq_grid_high)
        self.coefficient_builder = "numpy-segmented-v1"

    @staticmethod
    def _compute_coefficients(
        detector: Any,
        h_ref: Complex[Array, " n_freq"],
        f_bins: Float[Array, " n_valid_plus_one"],
    ) -> Complex[Array, "four n_valid"]:
        """Compute parent-equivalent bin summaries without a dense mask.

        ``searchsorted(..., side="right")`` reproduces the parent's half-open
        internal intervals: a frequency exactly on a shared edge belongs to
        the bin on its right.  The final edge is then included explicitly, as
        in the parent implementation.
        """

        data = np.asarray(jax.device_get(detector.sliced_fd_data))
        psd = np.asarray(jax.device_get(detector.sliced_psd))
        freqs = np.asarray(jax.device_get(detector.sliced_frequencies))
        reference = np.asarray(jax.device_get(h_ref))
        bins = np.asarray(jax.device_get(f_bins))
        n_bins = len(bins) - 1
        if n_bins <= 0 or np.any(np.diff(bins) <= 0):
            raise ValueError("heterodyne frequency-bin edges must be increasing")

        indices = np.searchsorted(bins, freqs, side="right") - 1
        indices = np.where(freqs == bins[-1], n_bins - 1, indices)
        valid = (indices >= 0) & (indices < n_bins)
        indices = indices[valid]
        selected_frequencies = freqs[valid]
        centers = 0.5 * (bins[:-1] + bins[1:])
        shifts = selected_frequencies - centers[indices]

        data_product = (data * reference.conj() / psd)[valid]
        self_product = (reference * reference.conj() / psd)[valid]

        def segmented_sum(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            result = np.zeros(n_bins, dtype=values.dtype)
            np.add.at(result, indices, values)
            return result

        summary = np.stack(
            (
                segmented_sum(data_product),
                segmented_sum(data_product * shifts),
                segmented_sum(self_product),
                segmented_sum(self_product * shifts),
            )
        )
        return jnp.asarray((4.0 / float(detector.duration)) * summary)

    @staticmethod
    def _time_phasors(
        times: Float[Array, " n_time"],
        frequencies: Float[Array, " n_bin"],
    ) -> Complex[Array, "n_time n_bin"]:
        """Conjugate detector phasors used by ``data * h.conj()`` summaries."""

        angle = (2.0 * jnp.pi) * times[:, None] * frequencies[None, :]
        return jax.lax.complex(jnp.cos(angle), jnp.sin(angle))

    def _likelihood(
        self,
        params: dict[str, Float],
        waveform_sky_low: dict[str, Complex[Array, " n_bins"]],
        waveform_sky_high: dict[str, Complex[Array, " n_bins"]],
    ) -> FloatScalar:
        """Evaluate the binned likelihood over the direct-sum time grid."""

        network_low_coeff = jnp.zeros(self.n_bins, dtype=jnp.complex128)
        network_high_coeff = jnp.zeros(self.n_bins, dtype=jnp.complex128)
        optimal_snr: FloatScalar = jnp.zeros(())

        for detector in self.detectors:
            waveform_low = detector.fd_response(
                self.freq_grid_low, waveform_sky_low, params
            )
            waveform_high = detector.fd_response(
                self.freq_grid_high, waveform_sky_high, params
            )
            r_low = waveform_low / self.waveform_low_ref[detector.name]
            r_high = waveform_high / self.waveform_high_ref[detector.name]

            A0, A1, B0, B1 = self.summary_data[detector.name]
            inverse_width = 1.0 / self.bin_widths
            low_coeff = (0.5 * A0 - A1 * inverse_width) * r_low.conj()
            high_coeff = (0.5 * A0 + A1 * inverse_width) * r_high.conj()
            network_low_coeff += low_coeff
            network_high_coeff += high_coeff

            # The exact h-h term is invariant under a coalescence-time shift.
            # Evaluate its usual relative-binning approximation once at t_c=0.
            r0 = (r_low + r_high) / 2.0
            r1 = (r_high - r_low) * inverse_width
            optimal_snr += jnp.sum(
                B0 * jnp.abs(r0) ** 2 + 2.0 * B1 * (r0 * r1.conj()).real
            ).real

        network_match = self._tc_phase_low @ network_low_coeff
        network_match += self._tc_phase_high @ network_high_coeff

        if self.phase_marginalization:
            time_integrand = log_i0(jnp.absolute(network_match))
        else:
            time_integrand = network_match.real
        return (
            -optimal_snr / 2.0
            + logsumexp(time_integrand)
            - jnp.log(self._tc_normalization_count)
        )


__all__ = ["PaperTimeMarginalizedHeterodynedLikelihoodFD"]
