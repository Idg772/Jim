from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp

from benchmarks.device_parallel_nss.paper_heterodyne import (
    PaperTimeMarginalizedHeterodynedLikelihoodFD,
)
from benchmarks.device_parallel_nss.preflight_gw170817_likelihood_pair import (
    _construct_full_likelihood_identical_grid,
)
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import (
    HeterodynedTransientLikelihoodFD,
    TransientLikelihoodFD,
)
from jimgw.core.single_event.waveform import RippleIMRPhenomD

FIXTURES_DIR = Path(__file__).parents[2] / "fixtures"
GPS = 1126259462.4
F_MIN = 20.0
F_MAX = 1024.0
TC_RANGE = (-0.03, 0.03)


def _problem():
    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        ifo.set_data(
            Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
        )
        ifo.set_psd(
            PowerSpectrum.from_file(str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz"))
        )
    return ifos, RippleIMRPhenomD(f_ref=20.0)


def _parameters() -> dict[str, float]:
    return {
        "M_c": 30.0,
        "eta": 0.249,
        "s1_z": 0.0,
        "s2_z": 0.0,
        "d_L": 400.0,
        "phase_c": 0.0,
        "t_c": 0.0,
        "iota": 0.0,
        "ra": 1.375,
        "dec": -1.2108,
        "psi": 0.0,
    }


def _heterodyne(*, n_bins: int = 5000, phase_marginalization: bool = True):
    ifos, waveform = _problem()
    return PaperTimeMarginalizedHeterodynedLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        n_bins=n_bins,
        reference_parameters=_parameters(),
        phase_marginalization=phase_marginalization,
        time_marginalization={"tc_range": TC_RANGE},
    )


def test_direct_sum_grid_matches_full_time_marginalization_grid() -> None:
    heterodyne = _heterodyne(n_bins=64)
    ifos, waveform = _problem()
    full = TransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        phase_marginalization=True,
        time_marginalization={"tc_range": TC_RANGE},
    )

    np.testing.assert_array_equal(heterodyne.tc_array, full.tc_array)
    np.testing.assert_array_equal(
        heterodyne.tc_window,
        full.tc_array[full._tc_window_indices],
    )
    assert heterodyne._tc_normalization_count == len(full.tc_array)


def test_preflight_full_constructor_matches_stock_full() -> None:
    ifos, waveform = _problem()
    stock = TransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        phase_marginalization=True,
        time_marginalization={"tc_range": TC_RANGE},
    )
    efficient = _construct_full_likelihood_identical_grid(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        phase_marginalization=True,
        tc_range=TC_RANGE,
        upsample_factor=1,
    )
    np.testing.assert_array_equal(efficient.frequencies, stock.frequencies)
    np.testing.assert_array_equal(efficient.tc_array, stock.tc_array)
    np.testing.assert_array_equal(
        efficient.evaluate(_parameters()),
        stock.evaluate(_parameters()),
    )


def test_reference_point_matches_full_phase_time_likelihood() -> None:
    heterodyne = _heterodyne()
    ifos, waveform = _problem()
    full = TransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        phase_marginalization=True,
        time_marginalization={"tc_range": TC_RANGE},
    )

    compressed = float(heterodyne.evaluate(_parameters()))
    uncompressed = float(full.evaluate(_parameters()))
    assert compressed == pytest.approx(uncompressed, abs=0.05)


def test_segmented_coefficients_match_parent_dense_builder() -> None:
    heterodyne = _heterodyne(n_bins=64)
    params = heterodyne.reference_parameters
    detector = heterodyne.detectors[0]
    reference_sky = heterodyne.waveform(detector.sliced_frequencies, params)
    reference = detector.fd_response(
        detector.sliced_frequencies,
        reference_sky,
        params,
    )
    edges = jnp.concatenate((heterodyne.freq_grid_low, heterodyne.freq_grid_high[-1:]))

    frequencies = detector.sliced_frequencies[None, :]
    left = edges[:-1, None]
    right = edges[1:, None]
    centers = 0.5 * (left + right)
    mask = (frequencies >= left) & (frequencies < right)
    mask = mask.at[-1].set(mask[-1] | (frequencies[0] == edges[-1]))
    shifts = (frequencies - centers) * mask
    data_product = detector.sliced_fd_data * reference.conj() / detector.sliced_psd
    self_product = reference * reference.conj() / detector.sliced_psd
    dense = (4.0 / detector.duration) * jnp.stack(
        (
            jnp.sum(data_product[None, :] * mask, axis=1),
            jnp.sum(data_product[None, :] * shifts, axis=1),
            jnp.sum(self_product[None, :] * mask, axis=1),
            jnp.sum(self_product[None, :] * shifts, axis=1),
        )
    )
    segmented = heterodyne._compute_coefficients(detector, reference, edges)
    np.testing.assert_allclose(segmented, dense, rtol=2e-13, atol=2e-13)
    assert heterodyne.coefficient_builder == "numpy-segmented-v1"


@pytest.mark.parametrize("phase_marginalization", [False, True])
def test_supported_core_matches_benchmark_direct_sum(
    phase_marginalization: bool,
) -> None:
    benchmark_likelihood = _heterodyne(
        n_bins=128,
        phase_marginalization=phase_marginalization,
    )
    ifos, waveform = _problem()
    supported = HeterodynedTransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        n_bins=128,
        reference_parameters=_parameters(),
        phase_marginalization=phase_marginalization,
        time_marginalization={
            "tc_range": TC_RANGE,
            "phasor_block_size": 11,
        },
    )

    np.testing.assert_allclose(
        supported.evaluate(_parameters()),
        benchmark_likelihood.evaluate(_parameters()),
        rtol=2e-13,
        atol=2e-13,
    )


@pytest.mark.parametrize("phase_marginalization", [False, True])
def test_window_normalized_direct_sum_matches_explicit_time_quadrature(
    phase_marginalization: bool,
) -> None:
    ifos, waveform = _problem()
    compressed = HeterodynedTransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        n_bins=5000,
        reference_parameters=_parameters(),
        phase_marginalization=phase_marginalization,
        time_marginalization={
            "tc_range": TC_RANGE,
            "normalization": "window",
            "phasor_block_size": 11,
        },
    )
    full_ifos, full_waveform = _problem()
    full = TransientLikelihoodFD(
        detectors=full_ifos,
        waveform=full_waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        phase_marginalization=phase_marginalization,
    )
    params = _parameters()
    explicit_values = jax.vmap(lambda tc: full.evaluate({**params, "t_c": tc}))(
        compressed.tc_window
    )
    explicit_quadrature = logsumexp(explicit_values) - jnp.log(
        len(compressed.tc_window)
    )

    assert float(compressed.evaluate(params)) == pytest.approx(
        float(explicit_quadrature),
        abs=0.05,
    )


def test_cached_and_jitted_evaluations_match_direct() -> None:
    heterodyne = _heterodyne(n_bins=128)
    params = _parameters()
    cache = heterodyne.generate_waveform(params)
    direct = heterodyne.evaluate(params)

    np.testing.assert_allclose(
        heterodyne.evaluate_from_waveform(params, cache),
        direct,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jax.jit(heterodyne.evaluate)(params),
        direct,
        rtol=1e-12,
        atol=1e-12,
    )


def test_sampled_tc_is_ignored_by_marginalized_likelihood() -> None:
    heterodyne = _heterodyne(n_bins=128)
    params = _parameters()
    shifted = {**params, "t_c": 0.019}
    np.testing.assert_array_equal(
        heterodyne.evaluate(params),
        heterodyne.evaluate(shifted),
    )


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (False, ValueError),
        (None, ValueError),
    ],
)
def test_invalid_time_marginalization_is_rejected(config, error) -> None:
    ifos, waveform = _problem()
    with pytest.raises(error):
        PaperTimeMarginalizedHeterodynedLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=F_MIN,
            f_max=F_MAX,
            trigger_time=GPS,
            n_bins=64,
            reference_parameters=_parameters(),
            phase_marginalization=True,
            time_marginalization=config,
        )


def test_upsampled_time_grid_has_matching_normalization() -> None:
    ifos, waveform = _problem()
    heterodyne = PaperTimeMarginalizedHeterodynedLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        n_bins=64,
        reference_parameters=_parameters(),
        phase_marginalization=True,
        time_marginalization={"tc_range": TC_RANGE, "upsample_factor": 4},
    )
    assert heterodyne.tc_upsample == 4
    assert heterodyne._tc_normalization_count == 4 * len(heterodyne.tc_array)
    assert len(heterodyne.tc_window) > 4 * 50
    assert np.all(np.asarray(heterodyne.tc_window) > TC_RANGE[0])
    assert np.all(np.asarray(heterodyne.tc_window) < TC_RANGE[1])


def test_upsampled_reference_matches_full_likelihood() -> None:
    ifos, waveform = _problem()
    heterodyne = PaperTimeMarginalizedHeterodynedLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        n_bins=5000,
        reference_parameters=_parameters(),
        phase_marginalization=True,
        time_marginalization={"tc_range": TC_RANGE, "upsample_factor": 4},
    )
    full = TransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=F_MIN,
        f_max=F_MAX,
        trigger_time=GPS,
        phase_marginalization=True,
        time_marginalization={"tc_range": TC_RANGE, "upsample_factor": 4},
    )
    assert float(heterodyne.evaluate(_parameters())) == pytest.approx(
        float(full.evaluate(_parameters())),
        abs=0.05,
    )
