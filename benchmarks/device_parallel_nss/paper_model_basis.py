"""Frequency-power basis for the benchmark-local Pv2+NRTidalv2 waveform.

The paper workload evaluates the waveform on a frequency grid that is fixed
for the entire run, but ripple's series evaluate ``fM_s ** (k/3)``-style
powers per call -- 112 f64 pow call sites per bin-lane element in the
compiled GPU kernel. Because ``(M*f)^p = M^p*f^p`` reassociates floating
point, XLA cannot hoist the frequency part itself. This module precomputes
``f^(n/6)`` and ``log(f)`` once per grid; the vendored waveform functions
(also in this module) consume them and reduce the per-tick series to
multiply-adds plus per-lane scalar powers.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from fractions import Fraction

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float
from ripplegw.constants import EULERGAMMA, MPC, MRSUN, MTSUN, PI, C
from ripplegw.typing import FloatLike
from ripplegw.utils.tidal import get_kappa, get_quadparam_octparam
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomD_NRTidalv2 import (
    _get_merger_frequency,
    get_planck_taper,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_QNMdata import fM_CUT
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_utils import (
    get_delta0,
    get_delta1,
    get_delta2,
    get_delta3,
    get_delta4,
)
from ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomPv2_utils import WignerdCoefficients
from ripplegw.waveforms.cbc.Taylor.TaylorF2 import (
    get_4PNQM2SCoeff,
    get_4PNQM2SOCoeff,
    get_6PNQM2SCoeff,
)


@dataclass(frozen=True)
class FrequencyPowerBasis:
    """Frequency arrays reused by every waveform evaluation on one grid."""

    f: Array
    log_f: Array
    _sixth_powers: dict[int, Array] = field(repr=False)
    _rational_powers: dict[Fraction, Array] = field(repr=False)

    @classmethod
    def build(cls, f: Array, sixth_exponents: Collection[int]) -> FrequencyPowerBasis:
        """Build the complete basis cached for a concrete waveform grid."""

        return cls._build(f, sixth_exponents, REQUIRED_RATIONAL_EXPONENTS)

    @classmethod
    def _build(
        cls,
        f: Array,
        sixth_exponents: Collection[int],
        rational_exponents: Collection[Fraction],
    ) -> FrequencyPowerBasis:
        f = jnp.asarray(f)
        return cls(
            f=f,
            log_f=jnp.log(f),
            _sixth_powers={
                int(n): f ** (n / 6.0) for n in sorted(set(sixth_exponents))
            },
            _rational_powers={
                exponent: f ** float(exponent) for exponent in rational_exponents
            },
        )

    def sixth_power(self, n: int) -> Array:
        """Return the precomputed ``f**(n/6)``."""

        return self._sixth_powers[n]

    def rational_power(self, numerator: int, denominator: int) -> Array:
        """Return a cached non-sixth rational power of the frequency grid."""

        return self._rational_powers[Fraction(numerator, denominator)]


# Grown below as each vendored function's frequency powers are factored.
REQUIRED_SIXTH_EXPONENTS: frozenset[int] = frozenset(
    {
        -18,
        -10,
        -8,
        -7,
        -6,
        -4,
        -2,
        2,
        4,
        6,
        8,
        10,
        12,
        13,
        14,
        16,
        18,
        24,
    }
)

# ripplegw 0.3.0 contains two powers outside the advertised sixth-power
# lattice: PhenomD's fM_s**(3/4), and NRTidalv2's
# ((pi*M_s*f)**(2/3))**2.89 == (pi*M_s*f)**(289/150). Caching these exact
# rational powers avoids the plan's invalid integer-floor rewrite.
REQUIRED_RATIONAL_EXPONENTS: frozenset[Fraction] = frozenset(
    {Fraction(3, 4), Fraction(289, 150)}
)

_INSPIRAL_PHASE_SIXTH_EXPONENTS = frozenset({-10, -8, -6, -4, -2, 2, 4, 8, 10, 12})
_IIA_PHASE_SIXTH_EXPONENTS = frozenset({-18})
_IIB_PHASE_SIXTH_EXPONENTS = frozenset({-6})
_IIB_PHASE_RATIONAL_EXPONENTS = frozenset({Fraction(3, 4)})
_INSPIRAL_AMP_SIXTH_EXPONENTS = frozenset({4, 8, 10, 12, 14, 16, 18})


def _scalar_basis_from_fm(
    fM_s: FloatLike,
    M_s: FloatLike,
    sixth_exponents: Collection[int],
    rational_exponents: Collection[Fraction] = (),
) -> FrequencyPowerBasis:
    """Build the cheap scalar basis used by PhenomD's matching gradients."""

    return FrequencyPowerBasis._build(fM_s / M_s, sixth_exponents, rational_exponents)


def _fm_power(basis: FrequencyPowerBasis, M_s: FloatLike, sixth_exponent: int) -> Array:
    """Return ``(M_s*f)**(sixth_exponent/6)`` with cached frequency power."""

    return (M_s ** (sixth_exponent / 6.0)) * basis.sixth_power(sixth_exponent)


def _pi_fm_power(
    basis: FrequencyPowerBasis, M_s: FloatLike, sixth_exponent: int
) -> Array:
    """Return ``(pi*M_s*f)**(sixth_exponent/6)`` from the cache."""

    return ((PI * M_s) ** (sixth_exponent / 6.0)) * basis.sixth_power(sixth_exponent)


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::get_inspiral_phase; only frequency
# powers are factored onto FrequencyPowerBasis.
def _get_inspiral_phase_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
) -> Float[Array, " n_freq"]:
    """Calculate the inspiral phase for the IMRPhenomD waveform."""

    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    theta_M_s = m1_s + m2_s
    eta = m1_s * m2_s / (theta_M_s**2.0)

    m1M = m1_s / theta_M_s
    m2M = m2_s / theta_M_s

    phi0 = 1.0
    phi1 = 0.0
    phi2 = 5.0 * (74.3 / 8.4 + 11.0 * eta) / 9.0
    phi3 = -16.0 * PI + (
        m1M * (25.0 + 38.0 / 3.0 * m1M) * chi1 + m2M * (25.0 + 38.0 / 3.0 * m2M) * chi2
    )
    phi4 = 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0
    phi4 += (
        (247.0 / 4.8 * eta) * chi1 * chi2
        + (-721.0 / 4.8 * eta) * chi1 * chi2
        + ((-720.0 / 9.6 * m1M * m1M) + (1.0 / 9.6 * m1M * m1M)) * chi1 * chi1
        + ((-720.0 / 9.6 * m2M * m2M) + (1.0 / 9.6 * m2M * m2M)) * chi2 * chi2
        + ((240.0 / 9.6 * m1M * m1M) + (-7.0 / 9.6 * m1M * m1M)) * chi1 * chi1
        + ((240.0 / 9.6 * m2M * m2M) + (-7.0 / 9.6 * m2M * m2M)) * chi2 * chi2
    )
    phi5 = 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta) * PI
    phi5 += (
        -m1M
        * (
            1391.5 / 8.4
            - m1M * (1.0 - m1M) * 10.0 / 3.0
            + m1M * (1276.0 / 8.1 + m1M * (1.0 - m1M) * 170.0 / 9.0)
        )
    ) * chi1 + (
        -m2M
        * (
            1391.5 / 8.4
            - m2M * (1.0 - m2M) * 10.0 / 3.0
            + m2M * (1276.0 / 8.1 + m2M * (1.0 - m2M) * 170.0 / 9.0)
        )
    ) * chi2
    phi5_log = (5.0 / 3.0) * (772.9 / 8.4 - 13.0 * eta) * PI
    phi5_log += 3.0 * (
        (
            -m1M
            * (
                1391.5 / 8.4
                - m1M * (1.0 - m1M) * 10.0 / 3.0
                + m1M * (1276.0 / 8.1 + m1M * (1.0 - m1M) * 170.0 / 9.0)
            )
        )
        * chi1
        + (
            -m2M
            * (
                1391.5 / 8.4
                - m2M * (1.0 - m2M) * 10.0 / 3.0
                + m2M * (1276.0 / 8.1 + m2M * (1.0 - m2M) * 170.0 / 9.0)
            )
        )
        * chi2
    )

    phi6 = (
        (
            11583.231236531 / 4.694215680
            - 640.0 / 3.0 * PI * PI
            - 684.8 / 2.1 * EULERGAMMA
        )
        + eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * PI * PI)
        + eta * eta * 76.055 / 1.728
        - eta * eta * eta * 127.825 / 1.296
        + (-684.8 / 2.1) * jnp.log(4.0)
    )
    phi6 += (PI * m1M * (1490.0 / 3.0 + m1M * 260.0)) * chi1 + (
        PI * m2M * (1490.0 / 3.0 + m2M * 260.0)
    ) * chi2
    phi6_log = -684.8 / 2.1

    phi7 = PI * (
        770.96675 / 2.54016 + 378.515 / 1.512 * eta - 740.45 / 7.56 * eta * eta
    )
    phi7 += (
        m1M
        * (
            -17097.8035 / 4.8384
            + eta * 28764.25 / 6.72
            + eta * eta * 47.35 / 1.44
            + m1M
            * (
                -7189.233785 / 1.524096
                + eta * 458.555 / 3.024
                - eta * eta * 534.5 / 7.2
            )
        )
    ) * chi1 + (
        m2M
        * (
            -17097.8035 / 4.8384
            + eta * 28764.25 / 6.72
            + eta * eta * 47.35 / 1.44
            + m2M
            * (
                -7189.233785 / 1.524096
                + eta * 458.555 / 3.024
                - eta * eta * 534.5 / 7.2
            )
        )
    ) * chi2

    fM_s = M_s * basis.f
    log_v = (jnp.log(PI * M_s) + basis.log_f) / 3.0
    pi_fm_one_third = _pi_fm_power(basis, M_s, 2)
    phi_TF2 = (
        phi0 * _pi_fm_power(basis, M_s, -10)
        + phi1 * _pi_fm_power(basis, M_s, -8)
        + phi2 * _pi_fm_power(basis, M_s, -6)
        + phi3 * _pi_fm_power(basis, M_s, -4)
        + phi4 * _pi_fm_power(basis, M_s, -2)
        + phi5_log * log_v
        + phi5
        + phi6_log * log_v * pi_fm_one_third
        + phi6 * pi_fm_one_third
        + phi7 * _pi_fm_power(basis, M_s, 4)
    ) * (3.0 / (128.0 * eta)) - PI / 4.0
    phi_Ins = (
        phi_TF2
        + (
            coeffs[7] * fM_s
            + (3.0 / 4.0) * coeffs[8] * _fm_power(basis, M_s, 8)
            + (3.0 / 5.0) * coeffs[9] * _fm_power(basis, M_s, 10)
            + (1.0 / 2.0) * coeffs[10] * _fm_power(basis, M_s, 12)
        )
        / eta
    )
    return phi_Ins


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::get_IIa_raw_phase; only frequency
# powers are factored onto FrequencyPowerBasis.
def _get_iia_raw_phase_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
) -> Float[Array, " n_freq"]:
    m1, m2, _, _ = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    theta_M_s = m1_s + m2_s
    eta = m1_s * m2_s / (theta_M_s**2.0)
    fM_s = M_s * basis.f

    phi_iia_raw = (
        coeffs[11] * fM_s
        + coeffs[12] * (jnp.log(M_s) + basis.log_f)
        - coeffs[13] * _fm_power(basis, M_s, -18) / 3.0
    ) / eta
    return phi_iia_raw


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::get_IIb_raw_phase; only frequency
# powers are factored onto FrequencyPowerBasis.
def _get_iib_raw_phase_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
    f_RD: FloatLike,
    f_damp: FloatLike,
    Rholm: float = 1.0,
    Taulm: float = 1.0,
) -> Float[Array, " n_freq"]:
    m1, m2, _, _ = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    theta_M_s = m1_s + m2_s
    eta = m1_s * m2_s / (theta_M_s**2.0)

    fM_s = M_s * basis.f
    f_RDM_s = f_RD * theta_M_s
    f_dampM_s = f_damp * theta_M_s
    fm_three_quarters = (M_s ** (3.0 / 4.0)) * basis.rational_power(3, 4)

    phi_iib_raw = (
        coeffs[14] * fM_s
        - coeffs[15] * _fm_power(basis, M_s, -6)
        + 4.0 * coeffs[16] * fm_three_quarters / 3.0
        + coeffs[17]
        * Rholm
        * jnp.arctan((fM_s - coeffs[18] * f_RDM_s) / (Rholm * f_dampM_s * Taulm))
    ) / eta
    return phi_iib_raw


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::get_Amp0; only frequency powers are
# factored onto FrequencyPowerBasis.
def _get_amp0_basis(
    basis: FrequencyPowerBasis, M_s: FloatLike, eta: FloatLike
) -> Float[Array, " n_freq"]:
    return (
        (2.0 / 3.0 * eta) ** (1.0 / 2.0)
        * _fm_power(basis, M_s, -7)
        * PI ** (-1.0 / 6.0)
    )


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::get_inspiral_Amp; only frequency
# powers are factored onto FrequencyPowerBasis.
def _get_inspiral_amp_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
) -> Float[Array, " n_freq"]:
    m1, m2, chi1, chi2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    theta_M_s = m1_s + m2_s
    eta = m1_s * m2_s / (theta_M_s**2.0)
    eta2 = eta * eta
    eta3 = eta * eta2

    Seta = jnp.sqrt(jnp.abs(1.0 - 4.0 * eta))
    SetaPlus1 = 1.0 + Seta
    chi12 = chi1 * chi1
    chi22 = chi2 * chi2

    A0 = 1.0
    A2 = ((-969.0 + 1804.0 * eta) * PI ** (2.0 / 3.0)) / 672.0
    A3 = (
        (
            chi1 * (81.0 * SetaPlus1 - 44.0 * eta)
            + chi2 * (81.0 - 81.0 * Seta - 44.0 * eta)
        )
        * PI
    ) / 48.0
    A4 = (
        (
            -27312085.0
            - 10287648.0 * chi22
            - 10287648.0 * chi12 * SetaPlus1
            + 10287648.0 * chi22 * Seta
            + 24.0
            * (
                -1975055.0
                + 857304.0 * chi12
                - 994896.0 * chi1 * chi2
                + 857304.0 * chi22
            )
            * eta
            + 35371056.0 * eta2
        )
        * (PI ** (4.0 / 3.0))
    ) / 8.128512e6
    A5 = (
        (PI ** (5.0 / 3.0))
        * (
            chi2
            * (
                -285197.0 * (-1 + Seta)
                + 4 * (-91902.0 + 1579.0 * Seta) * eta
                - 35632.0 * eta2
            )
            + chi1
            * (
                285197.0 * SetaPlus1
                - 4.0 * (91902.0 + 1579.0 * Seta) * eta
                - 35632.0 * eta2
            )
            + 42840.0 * (-1.0 + 4.0 * eta) * PI
        )
    ) / 32256.0
    A6 = (
        -(
            (PI**2.0)
            * (
                -336.0
                * (
                    -3248849057.0
                    + 2943675504.0 * chi12
                    - 3339284256.0 * chi1 * chi2
                    + 2943675504.0 * chi22
                )
                * eta2
                - 324322727232.0 * eta3
                - 7.0
                * (
                    -177520268561.0
                    + 107414046432.0 * chi22
                    + 107414046432.0 * chi12 * SetaPlus1
                    - 107414046432.0 * chi22 * Seta
                    + 11087290368.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) * PI
                )
                + 12.0
                * eta
                * (
                    -545384828789.0
                    - 176491177632.0 * chi1 * chi2
                    + 202603761360.0 * chi22
                    + 77616.0 * chi12 * (2610335.0 + 995766.0 * Seta)
                    - 77287373856.0 * chi22 * Seta
                    + 5841690624.0 * (chi1 + chi2) * PI
                    + 21384760320.0 * (PI**2.0)
                )
            )
        )
        / 6.0085960704e10
    )
    A7 = coeffs[0]
    A8 = coeffs[1]
    A9 = coeffs[2]

    return (
        A0
        + A2 * _fm_power(basis, M_s, 4)
        + A3 * (M_s * basis.f)
        + A4 * _fm_power(basis, M_s, 8)
        + A5 * _fm_power(basis, M_s, 10)
        + A6 * _fm_power(basis, M_s, 12)
        + A7 * _fm_power(basis, M_s, 14)
        + A8 * _fm_power(basis, M_s, 16)
        + A9 * _fm_power(basis, M_s, 18)
    )


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::get_IIb_Amp; only frequency powers
# are factored onto FrequencyPowerBasis.
def _get_iib_amp_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
    f_RD: FloatLike,
    f_damp: FloatLike,
) -> Float[Array, " n_freq"]:
    m1, m2, _, _ = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    theta_M_s = m1_s + m2_s
    gamma1 = coeffs[4]
    gamma2 = coeffs[5]
    gamma3 = coeffs[6]
    fDM = f_damp * theta_M_s
    fRD = f_RD * theta_M_s

    fDMgamma3 = fDM * gamma3
    fminfRD = M_s * basis.f - fRD
    return (
        jnp.exp(-(fminfRD) * gamma2 / fDMgamma3)
        * (fDMgamma3 * gamma1)
        / (fminfRD * fminfRD + fDMgamma3 * fDMgamma3)
    )


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::get_IIa_Amp; only frequency powers
# are factored onto FrequencyPowerBasis.
def _get_iia_amp_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
    f1: FloatLike,
    f3: FloatLike,
    f_RD: FloatLike,
    f_damp: FloatLike,
) -> Float[Array, " n_freq"]:
    f2 = (f1 + f3) / 2

    def inspiral_amp_at_fm(fM_s):
        scalar_basis = _scalar_basis_from_fm(fM_s, M_s, _INSPIRAL_AMP_SIXTH_EXPONENTS)
        return _get_inspiral_amp_basis(scalar_basis, M_s, theta, coeffs)

    def iib_amp_at_fm(fM_s):
        scalar_basis = _scalar_basis_from_fm(fM_s, M_s, ())
        return _get_iib_amp_basis(scalar_basis, M_s, theta, coeffs, f_RD, f_damp)

    v1, d1 = jax.value_and_grad(inspiral_amp_at_fm)(f1 * M_s)
    v3, d3 = jax.value_and_grad(iib_amp_at_fm)(f3 * M_s)

    delta0 = get_delta0(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta1 = get_delta1(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta2 = get_delta2(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta3 = get_delta3(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)
    delta4 = get_delta4(f1 * M_s, f2 * M_s, f3 * M_s, v1, coeffs[3], v3, d1, d3)

    return (
        delta0
        + delta1 * (M_s * basis.f)
        + delta2 * _fm_power(basis, M_s, 12)
        + delta3 * _fm_power(basis, M_s, 18)
        + delta4 * _fm_power(basis, M_s, 24)
    )


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::IMRPhenDAmplitude; only frequency
# powers are factored onto FrequencyPowerBasis.
def _imrphenomd_amplitude_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
    transition_frequencies: tuple[
        FloatLike, FloatLike, FloatLike, FloatLike, FloatLike, FloatLike
    ],
) -> Float[Array, " n_freq"]:
    _, _, f3, f4, f_RD, f_damp = transition_frequencies
    amp_ins = _get_inspiral_amp_basis(basis, M_s, theta, coeffs)
    amp_iia = _get_iia_amp_basis(basis, M_s, theta, coeffs, f3, f4, f_RD, f_damp)
    amp_iib = _get_iib_amp_basis(basis, M_s, theta, coeffs, f_RD, f_damp)

    f = basis.f
    fcut_true = jnp.floor(fM_CUT / M_s / (f[1] - f[0])) * (f[1] - f[0])
    return (
        amp_ins * jnp.heaviside(f3 - f, 0.5)
        + jnp.heaviside(f - f3, 0.5) * amp_iia * jnp.heaviside(f4 - f, 0.5)
        + jnp.heaviside(f - f4, 0.5) * amp_iib * jnp.heaviside(fcut_true - f, 0.0)
        + 0.0 * jnp.heaviside(f - fcut_true, 1.0)
    )


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::IMRPhenDAmplitude_NoCut; only
# frequency powers are factored onto FrequencyPowerBasis.
def _imrphenomd_amplitude_no_cut_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
    transition_frequencies: tuple[
        FloatLike, FloatLike, FloatLike, FloatLike, FloatLike, FloatLike
    ],
) -> Float[Array, " n_freq"]:
    _, _, f3, f4, f_RD, f_damp = transition_frequencies
    amp_ins = _get_inspiral_amp_basis(basis, M_s, theta, coeffs)
    amp_iia = _get_iia_amp_basis(basis, M_s, theta, coeffs, f3, f4, f_RD, f_damp)
    amp_iib = _get_iib_amp_basis(basis, M_s, theta, coeffs, f_RD, f_damp)
    f = basis.f
    return (
        amp_ins * jnp.heaviside(f3 - f, 0.5)
        + jnp.heaviside(f - f3, 0.5) * amp_iia * jnp.heaviside(f4 - f, 0.5)
        + jnp.heaviside(f - f4, 0.5) * amp_iib
    )


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::Amp; only frequency powers are
# factored onto FrequencyPowerBasis.
def amp_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
    transition_frequencies: tuple[
        FloatLike, FloatLike, FloatLike, FloatLike, FloatLike, FloatLike
    ],
    D: FloatLike = 1,
) -> Float[Array, " n_freq"]:
    m1, m2, _, _ = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    theta_M_s = m1_s + m2_s
    eta = m1_s * m2_s / (theta_M_s**2.0)

    amp = _imrphenomd_amplitude_basis(basis, M_s, theta, coeffs, transition_frequencies)
    amp0 = _get_amp0_basis(basis, M_s, eta) * (2.0 * jnp.sqrt(5.0 / (64.0 * PI)))
    dist_s = (D * MPC) / C
    return amp0 * amp * (theta_M_s**2.0) / dist_s


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomD.py::Phase; only frequency powers are
# factored onto FrequencyPowerBasis.
def phase_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 4],
    coeffs: Float[Array, 19],
    transition_freqs: tuple[
        FloatLike, FloatLike, FloatLike, FloatLike, FloatLike, FloatLike
    ],
    Rholm: float = 1.0,
    Taulm: float = 1.0,
) -> Float[Array, " n_freq"]:
    f1, f2, _, _, f_RD, f_damp = transition_freqs
    phi_ins = _get_inspiral_phase_basis(basis, M_s, theta, coeffs)

    def inspiral_phase_at_fm(fM_s):
        scalar_basis = _scalar_basis_from_fm(fM_s, M_s, _INSPIRAL_PHASE_SIXTH_EXPONENTS)
        return _get_inspiral_phase_basis(scalar_basis, M_s, theta, coeffs)

    def iia_raw_phase_at_fm(fM_s):
        scalar_basis = _scalar_basis_from_fm(fM_s, M_s, _IIA_PHASE_SIXTH_EXPONENTS)
        return _get_iia_raw_phase_basis(scalar_basis, M_s, theta, coeffs)

    phi_ins_f1, dphi_ins_f1 = jax.value_and_grad(inspiral_phase_at_fm)(f1 * M_s)
    phi_iia_f1, dphi_iia_f1 = jax.value_and_grad(iia_raw_phase_at_fm)(f1 * M_s)

    beta1_correction = dphi_ins_f1 - dphi_iia_f1
    beta0 = phi_ins_f1 - beta1_correction * (f1 * M_s) - phi_iia_f1
    phi_iia = (
        _get_iia_raw_phase_basis(basis, M_s, theta, coeffs)
        + beta1_correction * (M_s * basis.f)
        + beta0
    )

    def iia_phase_at_fm(fM_s):
        return iia_raw_phase_at_fm(fM_s) + beta1_correction * fM_s

    def iib_raw_phase_at_fm(fM_s):
        scalar_basis = _scalar_basis_from_fm(
            fM_s,
            M_s,
            _IIB_PHASE_SIXTH_EXPONENTS,
            _IIB_PHASE_RATIONAL_EXPONENTS,
        )
        return _get_iib_raw_phase_basis(
            scalar_basis, M_s, theta, coeffs, f_RD, f_damp, Rholm, Taulm
        )

    phi_iia_f2, dphi_iia_f2 = jax.value_and_grad(iia_phase_at_fm)(f2 * M_s)
    phi_iib_f2, dphi_iib_f2 = jax.value_and_grad(iib_raw_phase_at_fm)(f2 * M_s)

    a1_correction = dphi_iia_f2 - dphi_iib_f2
    a0 = phi_iia_f2 + beta0 - a1_correction * (f2 * M_s) - phi_iib_f2
    phi_iib = (
        _get_iib_raw_phase_basis(basis, M_s, theta, coeffs, f_RD, f_damp, Rholm, Taulm)
        + a0
        + a1_correction * (M_s * basis.f)
    )

    f = basis.f
    return (
        phi_ins * jnp.heaviside(f1 - f, 0.5)
        + jnp.heaviside(f - f1, 0.5) * phi_iia * jnp.heaviside(f2 - f, 0.5)
        + phi_iib * jnp.heaviside(f - f2, 0.5)
    )


# Vendored from ripplegw 0.3.0 waveforms/cbc/IMRPhenom_NRTidal/
# IMRPhenomD_NRTidalv2.py::get_amp0_lal; scalar-only source is unchanged.
def _get_amp0_lal(M: FloatLike, distance: FloatLike) -> FloatLike:
    return 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M * MRSUN * M * MTSUN / distance


# Vendored from ripplegw 0.3.0 waveforms/cbc/IMRPhenom_NRTidal/
# IMRPhenomD_NRTidalv2.py::get_tidal_amplitude; only frequency powers are
# factored onto FrequencyPowerBasis.
def _get_tidal_amplitude_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 6],
    kappa: FloatLike,
    distance: FloatLike = 1,
) -> Float[Array, " n_freq"]:
    m1, m2, _, _, _, _ = theta
    M = m1 + m2
    distance *= MPC

    x = _pi_fm_power(basis, M_s, 4)
    x_2p89 = ((PI * M_s) ** (289.0 / 150.0)) * basis.rational_power(289, 150)
    x_4 = _pi_fm_power(basis, M_s, 16)
    x_13over4 = _pi_fm_power(basis, M_s, 13)

    n1 = 4.157407407407407
    n289 = 2519.111111111111
    d = 13477.8073677
    num = 1.0 + n1 * x + n289 * x_2p89
    den = 1.0 + d * x_4
    poly = num / den

    prefac = -9.0 * kappa
    ampT = prefac * x_13over4 * poly
    amp0 = _get_amp0_lal(M, distance)
    ampT *= amp0 * 2 * jnp.sqrt(PI / 5)
    return ampT


# Vendored from ripplegw 0.3.0 waveforms/cbc/IMRPhenom_NRTidal/
# IMRPhenomD_NRTidalv2.py::get_tidal_phase; only frequency powers are factored
# onto FrequencyPowerBasis.
def _get_tidal_phase_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 6],
    kappa: FloatLike,
) -> Float[Array, " n_freq"]:
    m1, m2, _, _, _, _ = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    theta_M_s = m1_s + m2_s
    X1 = m1_s / theta_M_s
    X2 = m2_s / theta_M_s

    x = _pi_fm_power(basis, M_s, 4)
    x_2 = _pi_fm_power(basis, M_s, 8)
    x_3 = _pi_fm_power(basis, M_s, 12)
    x_3over2 = _pi_fm_power(basis, M_s, 6)
    x_5over2 = _pi_fm_power(basis, M_s, 10)

    c_Newt = 2.4375
    n_1 = -12.615214237993088
    n_3over2 = 19.0537346970349
    n_2 = -21.166863146081035
    n_5over2 = 90.55082156324926
    n_3 = -60.25357801943598
    d_1 = -15.111207827736678
    d_3over2 = 22.195327350624694
    d_2 = 8.064109635305156

    num = (
        1.0
        + (n_1 * x)
        + (n_3over2 * x_3over2)
        + (n_2 * x_2)
        + (n_5over2 * x_5over2)
        + (n_3 * x_3)
    )
    den = 1.0 + (d_1 * x) + (d_3over2 * x_3over2) + (d_2 * x_2)
    ratio = num / den

    psi_T = -kappa * c_Newt / (X1 * X2) * x_5over2
    psi_T *= ratio
    return psi_T


# Vendored from ripplegw 0.3.0 waveforms/cbc/IMRPhenom_NRTidal/
# IMRPhenomD_NRTidalv2.py::get_spin_phase_correction; only frequency powers
# are factored onto FrequencyPowerBasis.
def _get_spin_phase_correction_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 6],
) -> Float[Array, " n_freq"]:
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    theta_M_s = m1_s + m2_s
    eta = m1_s * m2_s / (theta_M_s**2.0)

    X1 = m1_s / theta_M_s
    X1sq = X1 * X1
    chi1_sq = chi1 * chi1
    X2 = m2_s / theta_M_s
    X2sq = X2 * X2
    chi2_sq = chi2 * chi2

    quadparam1, octparam1 = get_quadparam_octparam(lambda1)
    quadparam2, octparam2 = get_quadparam_octparam(lambda2)
    quadparam1 -= 1
    quadparam2 -= 1
    octparam1 -= 1
    octparam2 -= 1

    SS_3p5 = (
        -400.0 * PI * quadparam1 * chi1_sq * X1sq
        - 400.0 * PI * quadparam2 * chi2_sq * X2sq
    )
    SSS_3p5 = (
        10.0
        * ((X1sq + 308.0 / 3.0 * X1) * chi1 + (X2sq - 89.0 / 3.0 * X2) * chi2)
        * quadparam1
        * X1sq
        * chi1_sq
        + 10.0
        * ((X2sq + 308.0 / 3.0 * X2) * chi2 + (X1sq - 89.0 / 3.0 * X1) * chi1)
        * quadparam2
        * X2sq
        * chi2_sq
        - 440.0 * octparam1 * X1 * X1sq * chi1_sq * chi1
        - 440.0 * octparam2 * X2 * X2sq * chi2_sq * chi2
    )

    prefac = 3.0 / (128.0 * eta)
    return prefac * (SS_3p5 + SSS_3p5) * _pi_fm_power(basis, M_s, 4)


# Vendored from ripplegw 0.3.0 waveforms/cbc/IMRPhenom_NRTidal/
# IMRPhenomD_NRTidalv2.py::get_qm_phase_correction; only frequency powers are
# factored onto FrequencyPowerBasis.
def _get_qm_phase_correction_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta: Float[Array, 6],
) -> Float[Array, " n_freq"]:
    m1, m2, chi1, chi2, lambda1, lambda2 = theta
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    theta_M_s = m1_s + m2_s
    eta = m1_s * m2_s / (theta_M_s**2.0)

    X1 = m1_s / theta_M_s
    X2 = m2_s / theta_M_s
    quadparam1, _ = get_quadparam_octparam(lambda1)
    quadparam2, _ = get_quadparam_octparam(lambda2)
    dquadmon1 = quadparam1 - 1.0
    dquadmon2 = quadparam2 - 1.0

    delta_phi4 = (
        get_4PNQM2SOCoeff(X1) + get_4PNQM2SCoeff(X1)
    ) * dquadmon1 * chi1 * chi1 + (
        get_4PNQM2SOCoeff(X2) + get_4PNQM2SCoeff(X2)
    ) * dquadmon2 * chi2 * chi2
    delta_phi6 = (
        get_6PNQM2SCoeff(X1) * dquadmon1 * chi1 * chi1
        + get_6PNQM2SCoeff(X2) * dquadmon2 * chi2 * chi2
    )

    v = _pi_fm_power(basis, M_s, 2)
    prefac = 3.0 / (128.0 * eta)
    return prefac * (delta_phi4 / v + delta_phi6 * v)


# Vendored from ripplegw 0.3.0 waveforms/cbc/IMRPhenom_NRTidal/
# IMRPhenomD_NRTidalv2.py::Phase_with_qm_correction; only frequency powers
# are factored onto FrequencyPowerBasis.
def phase_with_qm_correction_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta_bbh: Float[Array, 4],
    theta_intrinsic: Float[Array, 6],
    coeffs: Float[Array, 19],
    transition_freqs: tuple[
        FloatLike, FloatLike, FloatLike, FloatLike, FloatLike, FloatLike
    ],
) -> Float[Array, " n_freq"]:
    f1, f2, _, _, f_RD, f_damp = transition_freqs

    def inspiral_phase_for_basis(candidate_basis):
        return _get_inspiral_phase_basis(
            candidate_basis, M_s, theta_bbh, coeffs
        ) + _get_qm_phase_correction_basis(candidate_basis, M_s, theta_intrinsic)

    phi_ins = inspiral_phase_for_basis(basis)

    def inspiral_phase_at_fm(fM_s):
        return inspiral_phase_for_basis(
            _scalar_basis_from_fm(fM_s, M_s, _INSPIRAL_PHASE_SIXTH_EXPONENTS)
        )

    def iia_raw_phase_at_fm(fM_s):
        return _get_iia_raw_phase_basis(
            _scalar_basis_from_fm(fM_s, M_s, _IIA_PHASE_SIXTH_EXPONENTS),
            M_s,
            theta_bbh,
            coeffs,
        )

    phi_ins_f1, dphi_ins_f1 = jax.value_and_grad(inspiral_phase_at_fm)(f1 * M_s)
    phi_iia_f1, dphi_iia_f1 = jax.value_and_grad(iia_raw_phase_at_fm)(f1 * M_s)
    beta1_correction = dphi_ins_f1 - dphi_iia_f1
    beta0 = phi_ins_f1 - beta1_correction * (f1 * M_s) - phi_iia_f1

    phi_iia = (
        _get_iia_raw_phase_basis(basis, M_s, theta_bbh, coeffs)
        + beta1_correction * (M_s * basis.f)
        + beta0
    )

    def iia_phase_at_fm(fM_s):
        return iia_raw_phase_at_fm(fM_s) + beta1_correction * fM_s

    def iib_raw_phase_at_fm(fM_s):
        return _get_iib_raw_phase_basis(
            _scalar_basis_from_fm(
                fM_s,
                M_s,
                _IIB_PHASE_SIXTH_EXPONENTS,
                _IIB_PHASE_RATIONAL_EXPONENTS,
            ),
            M_s,
            theta_bbh,
            coeffs,
            f_RD,
            f_damp,
        )

    phi_iia_f2, dphi_iia_f2 = jax.value_and_grad(iia_phase_at_fm)(f2 * M_s)
    phi_iib_f2, dphi_iib_f2 = jax.value_and_grad(iib_raw_phase_at_fm)(f2 * M_s)
    a1_correction = dphi_iia_f2 - dphi_iib_f2
    a0 = phi_iia_f2 + beta0 - a1_correction * (f2 * M_s) - phi_iib_f2

    phi_iib = (
        _get_iib_raw_phase_basis(basis, M_s, theta_bbh, coeffs, f_RD, f_damp)
        + a0
        + a1_correction * (M_s * basis.f)
    )

    f = basis.f
    return (
        phi_ins * jnp.heaviside(f1 - f, 0.5)
        + jnp.heaviside(f - f1, 0.5) * phi_iia * jnp.heaviside(f2 - f, 0.5)
        + phi_iib * jnp.heaviside(f - f2, 0.5)
    )


# Vendored from ripplegw 0.3.0 waveforms/cbc/IMRPhenom_NRTidal/
# IMRPhenomD_NRTidalv2.py::_amplitude_of; only frequency powers are factored
# onto FrequencyPowerBasis. The unchanged Planck taper remains imported from
# the pinned ripple module.
def amplitude_of_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta_intrinsic: Float[Array, 6],
    theta_extrinsic: Float[Array, 3],
    bbh_amp: Float[Array, " n_freq"],
    no_taper: bool = False,
) -> Float[Array, " n_freq"]:
    kappa = get_kappa(theta=theta_intrinsic)
    tidal_amplitude = _get_tidal_amplitude_basis(
        basis,
        M_s,
        theta_intrinsic,
        kappa,
        distance=theta_extrinsic[0],
    )

    if no_taper:
        taper = jnp.ones_like(basis.f)
    else:
        merger_frequency = _get_merger_frequency(theta_intrinsic, kappa)
        taper = get_planck_taper(basis.f, merger_frequency)
    return taper * (bbh_amp + tidal_amplitude)


# Vendored from ripplegw 0.3.0 waveforms/cbc/IMRPhenom_NRTidal/
# IMRPhenomD_NRTidalv2.py::_phase_of; only frequency powers are factored onto
# FrequencyPowerBasis.
def phase_of_basis(
    basis: FrequencyPowerBasis,
    M_s: FloatLike,
    theta_intrinsic: Float[Array, 6],
    bbh_psi: Float[Array, " n_freq"],
) -> Float[Array, " n_freq"]:
    kappa = get_kappa(theta=theta_intrinsic)
    tidal_phase = _get_tidal_phase_basis(basis, M_s, theta_intrinsic, kappa)
    spin_phase = _get_spin_phase_correction_basis(basis, M_s, theta_intrinsic)
    return -(bbh_psi + tidal_phase + spin_phase)


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomPv2.py::PhenomPCoreTwistUp (ripplegw
# 0.3.0), split into an intrinsics-only geometry half and a core application
# half so that Tasks 7-8 can cache the geometry across extrinsic-only
# re-evaluations. Three mechanical rewrites relative to the stock body:
# (1) the omega/omega_cbrt frequency powers are read off the basis instead of
# recomputed (``pi_m_s = jnp.pi * MTSUN * total_mass`` is the per-lane scalar;
# ``1/omega = pi_m_s**-1 * basis.sixth_power(-6)``,
# ``1/omega_cbrt2 = pi_m_s**(-2/3) * basis.sixth_power(-4)``,
# ``1/omega_cbrt = pi_m_s**(-1/3) * basis.sixth_power(-2)``,
# ``omega_cbrt = pi_m_s**(1/3) * basis.sixth_power(2)``,
# ``logomega = jnp.log(pi_m_s) + basis.log_f``); (2) every unit phasor uses
# ``jax.lax.complex(jnp.cos(x), jnp.sin(x))`` instead of ``jnp.exp(1j * x)``;
# (3) the scalar alphaoffset/epsilonoffset are factored out of the frequency
# series via ``exp(i(A - a0)) = exp(iA) * exp(-i*a0)`` and
# ``exp(-2i(E - e0)) = exp(-2iE) * exp(2i*e0)``, so the geometry function
# returns the un-offset series phasors and the core function multiplies in
# the scalar offset phasors. Everything else (q/m1/m2/Sperp/SL, the five
# Wigner products, T2m/Tm2m, eps_phase_hP, hp/hc assembly) is copied verbatim,
# with ``* 0.5`` written for the trailing ``/ 2.0``.
def phenomp_twist_up_geometry_basis(
    basis: FrequencyPowerBasis,
    total_mass: FloatLike,
    eta: FloatLike,
    chi1_l: FloatLike,
    chi2_l: FloatLike,
    chip: FloatLike,
    angcoeffs: dict[str, FloatLike],
) -> tuple[Array, Array, Array, Array]:
    """Intrinsics-only frequency arrays feeding ``PhenomPCoreTwistUp``.

    Returns ``(cexp_i_alpha_series, cexp_m2i_epsilon_series, cBetah,
    sBetah)`` -- the un-offset alpha/epsilon phasor series and the Wigner-d
    half-angle coefficients. None of these depend on alphaoffset,
    epsilonoffset, or Y2m.
    """

    q = (1.0 + jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0)) - 2.0 * eta) / (2.0 * eta)
    m1 = 1.0 / (1.0 + q)  # Mass of the smaller BH for unit total mass M=1.
    m2 = q / (1.0 + q)  # Mass of the larger BH for unit total mass M=1.
    Sperp = chip * (
        m2 * m2
    )  # Dimensionfull spin component in the orbital plane. S_perp = S_2_perp

    SL = chi1_l * m1 * m1 + chi2_l * m2 * m2  # Dimensionfull aligned spin.

    pi_m_s = jnp.pi * MTSUN * total_mass
    inv_omega = (pi_m_s**-1) * basis.sixth_power(-6)
    inv_omega_cbrt2 = (pi_m_s ** (-2.0 / 3.0)) * basis.sixth_power(-4)
    inv_omega_cbrt = (pi_m_s ** (-1.0 / 3.0)) * basis.sixth_power(-2)
    omega_cbrt = (pi_m_s ** (1.0 / 3.0)) * basis.sixth_power(2)
    logomega = jnp.log(pi_m_s) + basis.log_f

    alpha = (
        angcoeffs["alphacoeff1"] * inv_omega
        + angcoeffs["alphacoeff2"] * inv_omega_cbrt2
        + angcoeffs["alphacoeff3"] * inv_omega_cbrt
        + angcoeffs["alphacoeff4"] * logomega
        + angcoeffs["alphacoeff5"] * omega_cbrt
    )

    epsilon = (
        angcoeffs["epsiloncoeff1"] * inv_omega
        + angcoeffs["epsiloncoeff2"] * inv_omega_cbrt2
        + angcoeffs["epsiloncoeff3"] * inv_omega_cbrt
        + angcoeffs["epsiloncoeff4"] * logomega
        + angcoeffs["epsiloncoeff5"] * omega_cbrt
    )

    cexp_i_alpha_series = jax.lax.complex(jnp.cos(alpha), jnp.sin(alpha))
    cexp_m2i_epsilon_series = jax.lax.complex(
        jnp.cos(2.0 * epsilon), -jnp.sin(2.0 * epsilon)
    )

    cBetah, sBetah = WignerdCoefficients(omega_cbrt, SL, eta, Sperp)

    return cexp_i_alpha_series, cexp_m2i_epsilon_series, cBetah, sBetah


# Vendored from ripplegw 0.3.0
# waveforms/cbc/IMRPhenomD/IMRPhenomPv2.py::PhenomPCoreTwistUp; see the
# provenance note on ``phenomp_twist_up_geometry_basis`` above for the full
# rewrite rationale -- this half applies the scalar alphaoffset/epsilonoffset
# and Y2m to the geometry series and assembles hp/hc.
def phenomp_core_twist_up_basis(
    hPhenom: Complex[Array, " n_freq"],
    cexp_i_alpha_series: Array,
    cexp_m2i_epsilon_series: Array,
    cBetah: FloatLike,
    sBetah: FloatLike,
    Y2m: list,
    alphaoffset: FloatLike,
    epsilonoffset: FloatLike,
) -> tuple[Complex[Array, " n_freq"], Complex[Array, " n_freq"]]:
    """Apply the scalar extrinsic offsets and assemble hp/hc.

    Numerically equivalent to
    ``ripplegw...IMRPhenomPv2.PhenomPCoreTwistUp`` given the geometry
    produced by ``phenomp_twist_up_geometry_basis`` for the same intrinsics.
    """

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah2 * cBetah
    cBetah4 = cBetah3 * cBetah
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah2 * sBetah
    sBetah4 = sBetah3 * sBetah

    Y2mA = jnp.array(Y2m)  # need to pass Y2m in a 5-component list
    hp_sum = 0
    hc_sum = 0

    # exp(i(A - a0)) = exp(iA) * exp(-i*a0)
    alpha_offset_phasor = jax.lax.complex(jnp.cos(alphaoffset), -jnp.sin(alphaoffset))
    # exp(-2i(E - e0)) = exp(-2iE) * exp(2i*e0)
    epsilon_offset_phasor = jax.lax.complex(
        jnp.cos(2.0 * epsilonoffset), jnp.sin(2.0 * epsilonoffset)
    )

    cexp_i_alpha = cexp_i_alpha_series * alpha_offset_phasor
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = jnp.conj(cexp_i_alpha)  # exp(-i*alpha) = conj(exp(i*alpha))
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    T2m = (
        cexp_2i_alpha * cBetah4 * Y2mA[0]
        - cexp_i_alpha * 2 * cBetah3 * sBetah * Y2mA[1]
        + 1 * jnp.sqrt(6) * sBetah2 * cBetah2 * Y2mA[2]
        - cexp_mi_alpha * 2 * cBetah * sBetah3 * Y2mA[3]
        + cexp_m2i_alpha * sBetah4 * Y2mA[4]
    )
    Tm2m = (
        cexp_m2i_alpha * sBetah4 * jnp.conjugate(Y2mA[0])
        + cexp_mi_alpha * 2 * cBetah * sBetah3 * jnp.conjugate(Y2mA[1])
        + 1 * jnp.sqrt(6) * sBetah2 * cBetah2 * jnp.conjugate(Y2mA[2])
        + cexp_i_alpha * 2 * cBetah3 * sBetah * jnp.conjugate(Y2mA[3])
        + cexp_2i_alpha * cBetah4 * jnp.conjugate(Y2mA[4])
    )
    hp_sum = T2m + Tm2m
    hc_sum = 1j * (T2m - Tm2m)
    cexp_m2i_epsilon = cexp_m2i_epsilon_series * epsilon_offset_phasor
    eps_phase_hP = cexp_m2i_epsilon * hPhenom * 0.5

    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum

    return hp, hc


__all__ = [
    "REQUIRED_RATIONAL_EXPONENTS",
    "REQUIRED_SIXTH_EXPONENTS",
    "FrequencyPowerBasis",
    "amp_basis",
    "amplitude_of_basis",
    "phase_basis",
    "phase_of_basis",
    "phase_with_qm_correction_basis",
    "phenomp_core_twist_up_basis",
    "phenomp_twist_up_geometry_basis",
]
