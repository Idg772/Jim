# MIT License
#
# Copyright (c) 2022 Adam Coogan, Thomas Edwards
# Copyright (c) 2025 GW JAX Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Fixed-grid waveform carrier and exact mode-2 emission clock for XG.

These factorizations preserve the supported Ripple 0.3.0 waveform formulas,
including its frequency-spacing cutoff, time anchor and tidal taper. They
reassociate floating-point operations and must not be treated as bitwise equal
to the baseline waveform.
"""

from importlib.metadata import version

import jax
import jax.numpy as jnp
from ripplegw.constants import MTSUN
from ripplegw.conversions import Mc_eta_to_ms
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    IMRPhenomD_NRTidalv2,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD import get_IIb_raw_phase
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_QNMdata import fM_CUT
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_utils import (
    get_coeffs,
    get_transition_frequencies,
)

from jimgw.core.single_event._nrtidalv2_basis import (
    REQUIRED_SIXTH_EXPONENTS,
    FrequencyPowerBasis,
    amp_basis,
    amplitude_of_basis,
    phase_of_basis,
    phase_with_qm_correction_basis,
)
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.transform_utils import Mc_eta_to_m1_m2

SUPPORTED_RIPPLE_VERSION = "0.3.0"


class PrefixedBaselineWaveform(DominantModeTimeCachedWaveform):
    """Keep stock arithmetic and a prior node grid's waveform cutoff spacing."""

    def __init__(self, waveform, frequency_prefix):
        super().__init__(waveform.source, mode=waveform.mode)
        self._original_waveform = waveform
        self.frequency_prefix = jnp.asarray(frequency_prefix)

    def __call__(self, frequency, params):
        full = self._original_waveform(
            jnp.concatenate((self.frequency_prefix, frequency)), params
        )
        return jax.tree.map(lambda value: value[2:], full)

    def build_waveform_cache(self, frequency, params):
        full = self._original_waveform.build_waveform_cache(
            jnp.concatenate((self.frequency_prefix, frequency)), params
        )
        return jax.tree.map(lambda value: value[2:], full)


def supports_source(source):
    """Check the exact backend contract for the factored carrier formulas."""
    return (
        version("ripplegw") == SUPPORTED_RIPPLE_VERSION
        and type(source) is IMRPhenomD_NRTidalv2
        and not source.use_lambda_tildes
        and not source.no_taper
    )


def basis_waveform(frequency, f_ref, clock):
    """Build the fixed-grid aligned-spin carrier with the stock time anchor.

    Assembly follows ripplegw 0.3.0 _bbh_amp_psi and
    gen_IMRPhenomD_NRTidalv2_hphc, including the stock cutoff convention.
    No Pv2 spin conversion or alternative time anchor is introduced.
    """
    basis = FrequencyPowerBasis.build(frequency, REQUIRED_SIXTH_EXPONENTS)
    reference = FrequencyPowerBasis.build(
        jnp.asarray([f_ref]), REQUIRED_SIXTH_EXPONENTS
    )

    def waveform(p):
        m1, m2 = Mc_eta_to_ms(jnp.array([p["M_c"], p["eta"]]))
        bbh = jnp.array([m1, m2, p["s1_z"], p["s2_z"]])
        intrinsic = jnp.concatenate((bbh, jnp.array([p["lambda_1"], p["lambda_2"]])))
        extrinsic = jnp.array([p["d_L"], 0.0, p["phase_c"]])
        mass = (m1 + m2) * MTSUN
        coeff = get_coeffs(bbh)
        transitions = get_transition_frequencies(bbh, coeff[5], coeff[6])
        _, _, _, f4, f_rd, f_damp = transitions
        t0 = jax.grad(get_IIb_raw_phase)(f4 * mass, bbh, coeff, f_rd, f_damp)
        phase = phase_with_qm_correction_basis(
            basis, mass, bbh, intrinsic, coeff, transitions
        )
        phase_ref = phase_with_qm_correction_basis(
            reference, mass, bbh, intrinsic, coeff, transitions
        )[0]
        phase -= t0 * (frequency * mass - f_ref * mass) + phase_ref
        phase += -2 * p["phase_c"]
        spacing = frequency[1] - frequency[0]
        cutoff = jnp.floor(fM_CUT / mass / spacing) * spacing
        phase = phase * jnp.heaviside(cutoff - frequency, 0.0) + (
            2 * jnp.pi * jnp.heaviside(frequency - cutoff, 1.0)
        )
        amplitude = amp_basis(basis, mass, bbh, coeff, transitions, D=p["d_L"])
        amplitude = amplitude_of_basis(basis, mass, intrinsic, extrinsic, amplitude)
        phase = phase_of_basis(basis, mass, intrinsic, phase)
        h0 = amplitude * jax.lax.complex(jnp.cos(phase), jnp.sin(phase))
        cosine = jnp.cos(p["iota"])
        return {
            "p": h0 * (0.5 * (1 + cosine**2)),
            "c": -1j * h0 * cosine,
            "__tau__": clock(frequency, p),
        }

    return waveform


def clock_coefficients(p):
    """Four scalar frequency coefficients of Jim's mode-2 2PN clock."""
    m1, m2 = Mc_eta_to_m1_m2(p["M_c"], p["eta"])
    total = m1 + m2
    eta = m1 * m2 / total**2
    chirp = (m1 * m2) ** (3.0 / 5) / total ** (1.0 / 5)
    mass = total * MTSUN
    mc = chirp * MTSUN
    x = jnp.pi * mass
    beta = (
        (113 * (m1 / total) ** 2 + 75 * eta) * p["s1_z"]
        + (113 * (m2 / total) ** 2 + 75 * eta) * p["s2_z"]
    ) / 12
    sigma = (721 - 247) * eta * p["s1_z"] * p["s2_z"] / 48
    t0 = 5.0 / 256 * mc * (jnp.pi * mc) ** (-8.0 / 3)
    t2 = 4.0 / 3 * (743.0 / 336 + 11 * eta / 4) * x ** (2.0 / 3) * t0
    t3 = -8.0 / 5 * (4 * jnp.pi - beta) * x * t0
    t4 = (
        2
        * (3058673.0 / 1016064 + 5429 * eta / 1008 + 617 * eta**2 / 144 - sigma)
        * x ** (4.0 / 3)
        * t0
    )
    return jnp.stack((t0, t2, t3, t4)), 1 / (6**1.5 * jnp.pi * mass)


def reconstruct_clock(frequency, frequency_powers, coefficients):
    """Reconstruct the exact four-term 2PN clock with its original ISCO guard."""
    terms, isco = coefficients
    tau = jnp.sum(terms[:, None] * frequency_powers, axis=0)
    return jnp.where(frequency < isco, jnp.maximum(tau, 0.0), 0.0)


class XGBasisCachedWaveform(DominantModeTimeCachedWaveform):
    """Reuse grid powers and cache one unit-distance carrier plus its clock.

    Only the stock aligned NRTidalv2 backend is supported. Other likelihoods
    retain their supplied waveform, including custom emission-clock adapters.
    """

    def __init__(
        self, source, *, factor_clock=True, compact=True, frequency_prefix=None
    ):
        if not supports_source(source):
            raise ValueError(
                "The factored XG carrier requires Ripple 0.3.0 stock "
                "IMRPhenomD_NRTidalv2 with lambda_1/lambda_2 and its taper"
            )
        super().__init__(source)
        self.factor_clock = factor_clock
        self.compact = compact
        self.frequency_prefix = frequency_prefix
        self._grid_factories = []

    def _make_factory(self, f):
        if self.frequency_prefix is not None:
            f = jnp.concatenate((jnp.asarray(self.frequency_prefix), f))
        if self.factor_clock:
            powers = (
                f[None, :] ** jnp.array([-8.0 / 3, -2.0, -5.0 / 3, -4.0 / 3])[:, None]
            )

            def clock(frequency, p):
                return reconstruct_clock(frequency, powers, clock_coefficients(p))
        else:
            clock = self._time_to_coalescence
        factory = basis_waveform(f, self.f_ref, clock)
        if self.frequency_prefix is None:
            return factory

        def without_prefix(params):
            return jax.tree.map(lambda value: value[2:], factory(params))

        return without_prefix

    def _factory(self, f):
        if isinstance(f, jax.core.Tracer):
            return self._make_factory(f)
        for old_f, factory in self._grid_factories:
            if f is old_f:
                return factory
        with jax.ensure_compile_time_eval():
            factory = self._make_factory(f)
        self._grid_factories.append((f, factory))
        return factory

    def __call__(self, frequency, params):
        # Use the same unit-distance carrier algebra in both entry points.
        # Direct physical-distance fusion and cached fusion produced different
        # likelihood values in the standalone compiled XG contraction screen.
        return self.waveform_from_cache(
            frequency, params, self.build_waveform_cache(frequency, params)
        )

    def build_waveform_cache(self, frequency, params):
        p = {**params, "d_L": 1.0, "iota": 0.0}
        pols = self._factory(frequency)(p)
        if self.compact:
            return {"carrier": pols["p"], "__tau__": pols["__tau__"]}
        return {
            "__source_cache__": {"p": pols["p"], "c": pols["c"]},
            "__tau__": pols["__tau__"],
        }

    def waveform_from_cache(self, frequency, params, cache):
        del frequency
        h = cache["carrier"] if self.compact else cache["__source_cache__"]["p"]
        ci = jnp.cos(params["iota"])
        return {
            "p": h * (0.5 * (1 + ci**2) / params["d_L"]),
            "c": h * (-1j * ci / params["d_L"]),
            "__tau__": cache["__tau__"],
        }
