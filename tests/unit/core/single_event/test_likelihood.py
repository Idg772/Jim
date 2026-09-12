import logging
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import logsumexp
from scipy.fft import next_fast_len

from jimgw.core.constants import EARTH_RADIUS_LIGHT_S
from jimgw.core.jim import Jim
from jimgw.core.prior import CombinePrior, GaussianPrior, PowerLawPrior, UniformPrior
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.dominant_mode import DominantModeTimeCachedWaveform
from jimgw.core.single_event.likelihood import (
    _XG_PLAN_AUTHORITY,
    HeterodynedTransientLikelihoodFD,
    MultibandedTransientLikelihoodFD,
    TransientLikelihoodFD,
    ZeroLikelihood,
    _build_time_marginalization_fine_window,
    _build_time_marginalization_zoom_plan,
    _VerifiedXGPlan,
)
from jimgw.core.single_event.time_utils import (
    greenwich_mean_sidereal_time as compute_gmst,
)
from jimgw.core.single_event.transforms import (
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    MassRatioToSymmetricMassRatioTransform,
)
from jimgw.core.single_event.utils import complex_inner_product, inner_product
from jimgw.core.single_event.waveform import RippleIMRPhenomD
from jimgw.core.utils import log_i0
from jimgw.samplers.config import BlackJAXSwiGConfig
from tests.utils import assert_all_finite, common_keys_allclose

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"


def test_custom_waveform_cache_requires_both_protocol_hooks() -> None:
    likelihood = object.__new__(TransientLikelihoodFD)
    likelihood.waveform = SimpleNamespace(
        parameter_names=("iota",),
        cacheable_parameter_names=frozenset({"iota"}),
        build_waveform_cache=lambda frequencies, params: None,
    )

    with pytest.raises(TypeError, match="build_waveform_cache.*waveform_from_cache"):
        _ = likelihood.waveform_cacheable_parameter_names


def test_custom_waveform_cache_rejects_unknown_parameter_names() -> None:
    likelihood = object.__new__(TransientLikelihoodFD)
    likelihood.waveform = SimpleNamespace(
        parameter_names=("iota",),
        cacheable_parameter_names=frozenset({"not_a_parameter"}),
        build_waveform_cache=lambda frequencies, params: None,
        waveform_from_cache=lambda frequencies, params, cache: None,
    )

    with pytest.raises(ValueError, match="not waveform parameters"):
        _ = likelihood.waveform_cacheable_parameter_names


def test_source_cache_dependencies_include_emission_time_inputs() -> None:
    likelihood = object.__new__(TransientLikelihoodFD)
    likelihood.waveform = SimpleNamespace(
        parameter_names=("M_c", "iota"),
        cacheable_parameter_names=frozenset({"iota"}),
        emission_time_parameter_names=frozenset({"M_c", "chi_eff"}),
        build_waveform_cache=lambda frequencies, params: None,
        waveform_from_cache=lambda frequencies, params, cache: None,
    )

    assert likelihood.waveform_cache_dependency_parameter_names == frozenset(
        {"M_c", "chi_eff"}
    )


def test_reference_support_ignores_reserved_timing_metadata() -> None:
    likelihood = object.__new__(HeterodynedTransientLikelihoodFD)
    likelihood.reference_chunk_size = 3
    likelihood.frequencies = jnp.arange(1.0, 9.0)
    likelihood.reference_parameters = {}

    def reference(frequencies, params):
        del params
        physical = jnp.where(
            (frequencies >= 3.0) & (frequencies <= 6.0),
            jnp.ones_like(frequencies, dtype=jnp.complex128),
            jnp.zeros_like(frequencies, dtype=jnp.complex128),
        )
        return {"p": physical, "c": 1j * physical, "__tau__": frequencies**-1}

    support = likelihood._find_reference_frequency_support(reference)
    assert support == (3.0, 6.0)

    waveform = reference(likelihood.frequencies, {})
    masked = likelihood._mask_and_set_frequency_arrays(
        waveform,
        likelihood.frequencies,
    )
    np.testing.assert_array_equal(masked, jnp.arange(3.0, 7.0))


def test_upsampled_fine_window_handles_signed_wrap_and_strict_bounds():
    candidates, fine_mask, fine_step = _build_time_marginalization_fine_window(
        n_total=3,
        upsample_factor=2,
        duration=3.0,
        tc_range=(-1.6, -1.4),
    )

    assert fine_step == 0.5
    np.testing.assert_array_equal(candidates, np.asarray([1]))
    np.testing.assert_array_equal(fine_mask, np.asarray([[False], [True]]))

    fine_step = 1.0 / 10.0
    candidates, fine_mask, _ = _build_time_marginalization_fine_window(
        n_total=2,
        upsample_factor=5,
        duration=1.0,
        tc_range=(-3 * fine_step, -1.5 * fine_step),
    )
    np.testing.assert_array_equal(candidates, np.asarray([1]))
    np.testing.assert_array_equal(
        fine_mask,
        np.asarray([[False], [False], [False], [True], [False]]),
    )


@pytest.mark.parametrize("tc_range", [(2.2, 2.4), (-2.4, -2.2)])
def test_zoom_plan_stays_local_at_signed_support_boundary(tc_range):
    n_total = 5
    upsample = 3
    candidates, _, _ = _build_time_marginalization_fine_window(
        n_total=n_total,
        upsample_factor=upsample,
        duration=5.0,
        tc_range=tc_range,
    )
    plan = _build_time_marginalization_zoom_plan(
        n_total,
        upsample,
        candidates,
    )

    rng = np.random.default_rng(17)
    values = rng.normal(size=n_total) + 1j * rng.normal(size=n_total)
    convolved = np.fft.ifft(
        np.fft.fft(values * plan.input_chirp, n=plan.fft_size) * plan.kernel_fft
    )
    local = (
        convolved[plan.output_start : plan.output_start + len(plan.output_chirp)]
        * plan.output_chirp
    )
    actual = local[plan.gather_indices]

    fine_fft = np.fft.fft(values, n=n_total * upsample)
    storage = candidates[None, :] * upsample + np.arange(upsample)[:, None]
    expected = fine_fft[storage]

    assert len(plan.output_chirp) == upsample
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize(
    "tc_range",
    [(1e300, 1e301), (np.nan, 0.0), (0.0, np.nan), (1.0, -1.0)],
)
def test_upsampled_fine_window_rejects_invalid_ranges(tc_range):
    candidates, fine_mask, _ = _build_time_marginalization_fine_window(
        n_total=3,
        upsample_factor=2,
        duration=3.0,
        tc_range=tc_range,
    )

    assert candidates.size == 0
    assert fine_mask.shape == (2, 0)


def test_inner_product_matches_real_part_of_complex_inner_product():
    rng = np.random.default_rng(2)
    n = 1024
    h1 = jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))
    h2 = jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))
    psd = jnp.asarray(rng.uniform(0.5, 2.0, size=n))
    df = 0.25
    expected = complex_inner_product(h1, h2, psd, df).real
    result = inner_product(h1, h2, psd, df)
    assert jnp.iscomplexobj(result) is False
    np.testing.assert_allclose(float(result), float(expected), rtol=1e-13)


@pytest.fixture
def detectors_and_waveform():
    gps = 1126259462.4
    fmin = 20.0
    fmax = 1024.0
    ifos = [get_H1(), get_L1()]
    for ifo in ifos:
        data = Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
        ifo.set_data(data)
        psd = PowerSpectrum.from_file(
            str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
        )
        ifo.set_psd(psd)
    waveform = RippleIMRPhenomD(f_ref=20.0)
    return ifos, waveform, fmin, fmax, gps


def example_params():
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


def _fixture_xg_plan(ifos, waveform, fmin, fmax, gps, *, n_bins):
    baseline = HeterodynedTransientLikelihoodFD(
        detectors=ifos,
        waveform=waveform,
        f_min=fmin,
        f_max=fmax,
        trigger_time=gps,
        n_bins=n_bins,
        reference_parameters=example_params(),
    )
    return _VerifiedXGPlan(baseline.bin_edges_sha256, _XG_PLAN_AUTHORITY)


def _waveform_cache_prior() -> CombinePrior:
    bounds = {
        "M_c": (20.0, 40.0),
        "q": (0.5, 1.0),
        "s1_z": (-0.05, 0.05),
        "s2_z": (-0.05, 0.05),
        "iota": (0.1, 3.0),
        "ra": (0.0, 2.0 * jnp.pi),
        "dec": (-1.5, 1.5),
        "psi": (0.0, jnp.pi),
        "t_c": (-0.05, 0.05),
    }
    return CombinePrior(
        [
            UniformPrior(lo, hi, parameter_names=[name])
            for name, (lo, hi) in bounds.items()
        ]
    )


class TestZeroLikelihood:
    def test_initialization_and_evaluation(self, detectors_and_waveform):
        likelihood = ZeroLikelihood()
        assert isinstance(likelihood, ZeroLikelihood)
        assert likelihood.evaluate(example_params()) == 0.0


class TestTransientLikelihoodFD:
    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def make_d_L_prior(xmin: float = 100.0, xmax: float = 5000.0) -> PowerLawPrior:
        return PowerLawPrior(xmin=xmin, xmax=xmax, alpha=2.0, parameter_names=["d_L"])

    @staticmethod
    def params_without_d_L() -> dict:
        return {
            "M_c": 30.0,
            "eta": 0.249,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "phase_c": 0.0,
            "t_c": 0.0,
            "iota": 0.0,
            "ra": 1.375,
            "dec": -1.2108,
            "psi": 0.0,
        }

    @staticmethod
    def params_without_d_L_phase() -> dict:
        return {
            "M_c": 30.0,
            "eta": 0.249,
            "s1_z": 0.0,
            "s2_z": 0.0,
            "t_c": 0.0,
            "iota": 0.0,
            "ra": 1.375,
            "dec": -1.2108,
            "psi": 0.0,
        }

    # ── Initialization ────────────────────────────────────────────────────────

    def test_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        assert isinstance(likelihood, TransientLikelihoodFD)
        assert likelihood.frequencies[0] == fmin
        assert likelihood.frequencies[-1] == fmax
        assert likelihood.trigger_time == gps
        assert hasattr(likelihood, "gmst")

    def test_dynamic_response_rejects_dense_time_marginalization(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        timed_waveform = DominantModeTimeCachedWaveform(waveform)
        for detector in ifos:
            detector.time_dependent_response = True

        with pytest.raises(ValueError, match="sample t_c explicitly"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=timed_waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={"tc_range": (-0.03, 0.03)},
            )

    def test_dynamic_response_alias_rejects_dense_time_marginalization(
        self,
        detectors_and_waveform,
        monkeypatch,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        monkeypatch.setattr(
            waveform,
            "response_is_time_dependent",
            True,
            raising=False,
        )

        with pytest.raises(ValueError, match="sample t_c explicitly"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={"tc_range": (-0.03, 0.03)},
            )

    def test_identical_masks_detected_for_shared_grid(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        assert likelihood._identical_masks is True
        for mask in likelihood.frequency_masks:
            assert bool(jnp.all(mask))

    def test_baseline_likelihood_ablation_matches_optimized_active_path(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        kwargs = {
            "detectors": ifos,
            "waveform": waveform,
            "f_min": fmin,
            "f_max": fmax,
            "trigger_time": gps,
            "phase_marginalization": True,
        }
        optimized = TransientLikelihoodFD(**kwargs, likelihood_optimizations=True)
        baseline = TransientLikelihoodFD(**kwargs, likelihood_optimizations=False)
        assert optimized._identical_masks is True
        assert baseline._identical_masks is False
        np.testing.assert_allclose(
            np.asarray(optimized.evaluate(example_params())),
            np.asarray(baseline.evaluate(example_params())),
            rtol=1e-12,
        )

    def test_identical_mask_time_marg_fast_path_matches_slow_path(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        kwargs = {
            "detectors": ifos,
            "waveform": waveform,
            "f_min": fmin,
            "f_max": fmax,
            "trigger_time": gps,
            "time_marginalization": {},
            "phase_marginalization": True,
        }
        fast = TransientLikelihoodFD(**kwargs)
        slow = TransientLikelihoodFD(**kwargs)
        # Force the per-detector gather/scatter path on one instance. The
        # detector-summed fast path reassociates additions, so equivalence is
        # numerical rather than bit-exact.
        slow._identical_masks = False
        params = example_params()
        np.testing.assert_allclose(
            np.asarray(fast.evaluate(params)),
            np.asarray(slow.evaluate(params)),
            rtol=1e-12,
        )

    def test_time_marg_fast_path_matches_per_detector_reference(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        assert likelihood._identical_masks
        params = example_params()
        result = float(likelihood.evaluate(params))

        # Reference: the mixed-grid slow path, which keeps the pre-change
        # per-detector semantics (scatter accumulation + per-detector
        # inner_product) and shares Tasks 1-2's pure-function changes, so it
        # matches the pre-change value to ~1 ulp. Forcing it exercises the
        # full old code shape end-to-end.
        reference = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        reference._identical_masks = False
        np.testing.assert_allclose(
            result, float(reference.evaluate(params)), rtol=1e-10
        )

    def test_time_marg_weighted_data_matches_reference_division_path(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        assert likelihood._identical_masks

        # Precomputed per-detector weight lists exist, one entry per detector.
        assert len(likelihood._weighted_conj_data) == len(likelihood.detectors)
        assert len(likelihood._inverse_sliced_psd) == len(likelihood.detectors)

        rng = np.random.default_rng(20260819)
        base_params = example_params()
        for _ in range(3):
            params = dict(base_params)
            params["t_c"] = float(rng.uniform(-0.05, 0.05))
            params["phase_c"] = float(rng.uniform(0.0, 2.0 * np.pi))
            params["d_L"] = float(rng.uniform(200.0, 600.0))

            result = float(likelihood.evaluate(params))

            # Reference: the pre-change explicit-division accumulation,
            # `4 * h * conj(d) / S * df` and `|h|^2 / S`, computed inline
            # here rather than via the (now precomputed-weight) fast path.
            # Uses the same prepared params (trigger_time/gmst injected,
            # t_c zeroed for marginalization) that `evaluate` uses internally.
            prepared_params = likelihood._prepare_parameters(params)
            waveform_sky = likelihood.waveform(likelihood.frequencies, prepared_params)
            n_freq = len(likelihood.frequencies)
            complex_d_inner_h = jnp.zeros(n_freq, dtype=jnp.complex128)
            hh_over_psd = jnp.zeros(n_freq)
            for i, ifo in enumerate(likelihood.detectors):
                h_dec = ifo.fd_response(
                    ifo.sliced_frequencies,
                    likelihood._sliced_waveform_sky(waveform_sky, i),
                    prepared_params,
                    optimize=likelihood.likelihood_optimization_axes["detector_phasor"],
                )
                complex_d_inner_h = complex_d_inner_h + (
                    4
                    * h_dec
                    * jnp.conj(ifo.sliced_fd_data)
                    / ifo.sliced_psd
                    * likelihood.df
                )
                hh_over_psd = hh_over_psd + (
                    (h_dec.real**2 + h_dec.imag**2) / ifo.sliced_psd
                )
            reference_logl = -(2.0 * likelihood.df) * jnp.sum(hh_over_psd)
            reference_logl = reference_logl + likelihood._reduce_time(complex_d_inner_h)
            reference = float(reference_logl)

            assert abs(result - reference) < 1e-9

    def test_time_marg_fast_path_emits_optimization_barrier(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        jaxpr = jax.make_jaxpr(likelihood.evaluate)(example_params())
        barriers = [
            eqn
            for eqn in jaxpr.jaxpr.eqns
            if eqn.primitive.name == "optimization_barrier"
        ]
        assert len(barriers) == 1

        n_freq = len(likelihood.frequencies)
        expected_avals = [
            ((n_freq,), jnp.dtype(jnp.complex128)),
            ((n_freq,), jnp.dtype(jnp.float64)),
        ]
        barrier = barriers[0]
        assert [(var.aval.shape, var.aval.dtype) for var in barrier.invars] == (
            expected_avals
        )
        assert [(var.aval.shape, var.aval.dtype) for var in barrier.outvars] == (
            expected_avals
        )

    def test_identical_mask_fast_path_plain_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        kwargs = {
            "detectors": ifos,
            "waveform": waveform,
            "f_min": fmin,
            "f_max": fmax,
            "trigger_time": gps,
        }
        fast = TransientLikelihoodFD(**kwargs)
        slow = TransientLikelihoodFD(**kwargs)
        slow._identical_masks = False
        params = example_params()
        np.testing.assert_array_equal(
            np.asarray(fast.evaluate(params)), np.asarray(slow.evaluate(params))
        )

    def test_cached_waveform_matches_full_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        assert jnp.allclose(
            likelihood.evaluate(params),
            likelihood.evaluate_from_waveform(params, cache),
        )

    def test_cached_waveform_restores_physical_distance_scaling(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        physical_waveform = waveform(
            likelihood.frequencies, likelihood._prepare_parameters(params)
        )

        for polarization, strain in physical_waveform.items():
            assert jnp.allclose(cache[polarization] / params["d_L"], strain)

    def test_cached_waveform_supports_cache_reusing_changes(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        moved = {
            **params,
            "d_L": params["d_L"] * 1.2,
            "ra": params["ra"] + 0.1,
            "psi": 0.2,
            "t_c": 0.01,
        }
        assert jnp.allclose(
            likelihood.evaluate(moved),
            likelihood.evaluate_from_waveform(moved, cache),
        )

    def test_evaluate_bypasses_waveform_cache_machinery(
        self, detectors_and_waveform, monkeypatch
    ):
        # evaluate() must generate the waveform directly at the true d_L; the
        # waveform-cache machinery (unit-distance normalization when supported,
        # rescaling on the way out) is reserved for the SwiG cache path
        # (generate_waveform()/evaluate_from_waveform()) and must not add
        # overhead to every other sampler's likelihood calls.
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )

        def _unexpected(*args, **kwargs):
            raise AssertionError("evaluate() must not use the waveform-cache machinery")

        monkeypatch.setattr(likelihood, "_waveform_sky_for_cache", _unexpected)
        monkeypatch.setattr(likelihood, "_waveform_sky_from_cache", _unexpected)

        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_waveform_cache_infers_blocks_after_transform_and_marginalization(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["ra", "dec"],
            ["psi"],
            ["t_c"],
        ]
        jim = Jim(
            likelihood,
            _waveform_cache_prior(),
            BlackJAXSwiGConfig(blocks=blocks, n_live=8, n_delete_frac=0.25),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
            periodic=[],
        )
        assert jim.sampler._rebuild_required_by_block == {
            (0, 1): True,
            (2, 3): True,
            (4,): True,
            (5, 6): False,
            (7,): False,
            (8,): False,
        }

    def test_waveform_cache_distance_block_reuses_cache_with_time_marginalization(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            time_marginalization={"tc_range": (-0.03, 0.03)},
        )
        prior = CombinePrior(
            [
                base_prior
                for base_prior in _waveform_cache_prior().base_prior
                if base_prior.parameter_names != ("t_c",)
            ]
            + [UniformPrior(100.0, 1000.0, parameter_names=["d_L"])]
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["d_L"],
            ["ra", "dec"],
            ["psi"],
        ]
        jim = Jim(
            likelihood,
            prior,
            BlackJAXSwiGConfig(
                blocks=blocks,
                de_jump_blocks=[
                    {"parameters": ["iota", "d_L"], "attempts": 1},
                    {"parameters": ["d_L"], "attempts": 1},
                ],
                n_live=8,
                n_delete_frac=0.25,
            ),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
        )
        assert jim.sampler._rebuild_required_by_block == {
            (0, 1): True,
            (2, 3): True,
            (4,): True,
            (8,): False,
            (5, 6): False,
            (7,): False,
        }
        assert jim.sampler._resolved_de_jump_blocks == (
            ((4, 8), True, 1),
            ((8,), False, 1),
        )

    def test_waveform_cache_rejects_non_sampling_parameter_in_blocks(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["ra", "dec"],
            ["psi"],
            ["t_c"],
            ["d_L"],
        ]
        with pytest.raises(ValueError, match="d_L.*not sampling parameters"):
            Jim(
                likelihood,
                _waveform_cache_prior(),
                BlackJAXSwiGConfig(blocks=blocks, n_live=8, n_delete_frac=0.25),
                likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
            )

    def test_uninitialized_data_raises(self):
        gps = 1126259462.4
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_psd(
                PowerSpectrum.from_file(
                    str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
                )
            )
        with pytest.raises(ValueError, match="does not have initialized data"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
            )

    def test_partially_initialized_data_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        extra = get_H1()
        extra.set_psd(
            PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{extra.name}.npz")
            )
        )
        with pytest.raises(ValueError, match=r"H1.*does not have initialized data"):
            TransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    def test_uninitialized_psd_raises(self):
        gps = 1126259462.4
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_data(
                Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
            )
        with pytest.raises(ValueError, match="does not have initialized PSD"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
            )

    def test_partially_initialized_psd_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        extra = get_H1()
        extra.set_data(
            Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{extra.name}.npz"))
        )
        with pytest.raises(ValueError, match="H1.*does not have initialized PSD"):
            TransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        params = example_params()
        ll = likelihood.evaluate(params)
        assert jnp.isfinite(ll)
        ll_jit = jax.jit(likelihood.evaluate)(params)
        assert jnp.isfinite(ll_jit)
        assert jnp.allclose(ll, ll_jit)
        ll_diff = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
        ).evaluate(params)
        assert jnp.isfinite(ll_diff)
        assert jnp.allclose(ll, ll_diff, atol=1e-2)

    # ── Time marginalization ───────────────────────────────────────────────────

    def test_time_marg_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        assert isinstance(likelihood, TransientLikelihoodFD)
        assert hasattr(likelihood, "tc_range")
        assert hasattr(likelihood, "tc_array")
        assert hasattr(likelihood, "pad_low")
        assert hasattr(likelihood, "pad_high")
        assert likelihood.tc_range == (-0.1, 0.1)

    def test_time_marg_custom_tc_range(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        custom_range = (-0.05, 0.05)
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={"tc_range": custom_range},
        )
        assert likelihood.tc_range == custom_range

    def test_time_marg_upsample_factor_validation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={"upsample_factor": 4},
        )
        assert likelihood.tc_upsample == 4
        with pytest.raises(ValueError):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={"upsample_factor": 0},
            )

    def test_time_marg_rejects_included_nyquist_endpoint(self, detectors_and_waveform):
        ifos, waveform, fmin, _, gps = detectors_and_waveform
        nyquist = float(ifos[0].data.sampling_frequency) / 2.0
        with pytest.raises(ValueError, match="excludes the Nyquist endpoint"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=nyquist,
                trigger_time=gps,
                time_marginalization={"upsample_factor": 4},
            )

    def test_upsampled_reduction_matches_zero_padded_fft(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        upsample = 4
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={"upsample_factor": upsample},
        )
        rng = np.random.default_rng(0)
        n_freq = len(likelihood.frequencies)
        d_inner_h = jnp.asarray(rng.normal(size=n_freq) + 1j * rng.normal(size=n_freq))

        padded = jnp.concatenate((likelihood.pad_low, d_inner_h, likelihood.pad_high))
        n_total = padded.size
        fine_fft = jnp.fft.fft(
            jnp.concatenate(
                (padded, jnp.zeros((upsample - 1) * n_total, dtype=padded.dtype))
            ),
            norm="backward",
        )
        duration = float(likelihood.detectors[0].data.duration)
        fine_tc = jnp.fft.fftfreq(n_total * upsample, 1.0 / duration)
        in_window = (fine_tc > likelihood.tc_range[0]) & (
            fine_tc < likelihood.tc_range[1]
        )
        norm = jnp.log(n_total * upsample)

        ref_time = logsumexp(jnp.where(in_window, fine_fft.real, -jnp.inf)) - norm
        np.testing.assert_allclose(
            float(likelihood._reduce_time(d_inner_h)), float(ref_time), rtol=1e-12
        )
        np.testing.assert_allclose(
            float(jax.jit(likelihood._reduce_time)(d_inner_h)),
            float(ref_time),
            rtol=1e-12,
        )
        ref_phase_time = (
            logsumexp(jnp.where(in_window, log_i0(jnp.absolute(fine_fft)), -jnp.inf))
            - norm
        )
        np.testing.assert_allclose(
            float(likelihood._reduce_phase_time(d_inner_h)),
            float(ref_phase_time),
            rtol=1e-12,
        )

    def test_upsampled_windowed_fft_compiles_without_offset_loop(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={
                "tc_range": (-0.03, 0.03),
                "upsample_factor": 32,
            },
        )
        d_inner_h = jnp.ones(len(likelihood.frequencies), dtype=jnp.complex128)

        stablehlo = str(
            jax.jit(likelihood._windowed_fft)
            .lower(d_inner_h)
            .compiler_ir(dialect="stablehlo")
        )

        assert "stablehlo.while" not in stablehlo
        assert stablehlo.count("stablehlo.fft") == 2
        assert (
            len(likelihood._tc_zoom_output_chirp)
            <= int(jnp.sum(likelihood._tc_fine_mask)) + 2 * likelihood.tc_upsample
        )
        assert likelihood._tc_zoom_fft_size == next_fast_len(
            len(likelihood.tc_array) + len(likelihood._tc_zoom_output_chirp) - 1
        )
        assert likelihood._tc_zoom_fft_size < 2 * len(likelihood.tc_array)

        coarse = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={"upsample_factor": 1},
        )
        coarse_hlo = str(
            jax.jit(coarse._windowed_fft)
            .lower(d_inner_h)
            .compiler_ir(dialect="stablehlo")
        )
        assert "stablehlo.while" not in coarse_hlo
        assert coarse_hlo.count("stablehlo.fft") == 1

    @pytest.mark.parametrize(
        ("upsample", "tc_range"),
        [
            (2, (-0.035, 0.013)),
            (3, (-0.021, 0.017)),
            (5, (0.004, 0.027)),
            (8, (-0.029, -0.003)),
            (32, (-0.03, 0.03)),
        ],
    )
    def test_upsampled_windowed_fft_matches_phase_ramp_oracle(
        self, detectors_and_waveform, upsample, tc_range
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={
                "tc_range": tc_range,
                "upsample_factor": upsample,
            },
        )
        rng = np.random.default_rng(11 + upsample)
        d_inner_h = jnp.asarray(
            rng.normal(size=len(likelihood.frequencies))
            + 1j * rng.normal(size=len(likelihood.frequencies))
        )

        padded = jnp.concatenate((likelihood.pad_low, d_inner_h, likelihood.pad_high))
        duration = float(likelihood.detectors[0].data.duration)
        frequencies = jnp.arange(padded.size) / duration
        fine_step = duration / (padded.size * upsample)

        def phase_ramp(offset):
            shifted = padded * jnp.exp(-2j * jnp.pi * frequencies * offset)
            return jnp.fft.fft(shifted)[likelihood._tc_fine_candidate_indices]

        reference = jax.lax.map(phase_ramp, fine_step * jnp.arange(upsample))
        actual = likelihood._windowed_fft(d_inner_h)

        np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=2e-11)

    @pytest.mark.parametrize("bounds", [(0.2, 0.3), (-0.3, -0.2)])
    def test_upsampled_time_marg_accepts_fine_only_window(
        self, detectors_and_waveform, bounds
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        duration = float(ifos[0].data.duration)
        n_total = int(duration * ifos[0].data.sampling_frequency / 2)
        dt = duration / n_total
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={
                "tc_range": (bounds[0] * dt, bounds[1] * dt),
                "upsample_factor": 4,
            },
        )

        assert likelihood._tc_window_indices.size == 0
        assert int(jnp.sum(likelihood._tc_fine_mask)) == 1
        d_inner_h = jnp.ones(len(likelihood.frequencies), dtype=jnp.complex128)
        padded = jnp.concatenate((likelihood.pad_low, d_inner_h, likelihood.pad_high))
        fine_fft = jnp.fft.fft(
            jnp.concatenate((padded, jnp.zeros(3 * padded.size, dtype=padded.dtype)))
        )
        storage = (
            likelihood._tc_fine_candidate_indices[None, :] * likelihood.tc_upsample
            + jnp.arange(likelihood.tc_upsample)[:, None]
        )
        expected_window = fine_fft[storage][likelihood._tc_fine_mask]
        normalization = jnp.log(padded.size * likelihood.tc_upsample)

        np.testing.assert_allclose(
            likelihood._windowed_fft(d_inner_h)[likelihood._tc_fine_mask],
            expected_window,
            rtol=2e-11,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            likelihood._reduce_time(d_inner_h),
            logsumexp(expected_window.real) - normalization,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            likelihood._reduce_phase_time(d_inner_h),
            logsumexp(log_i0(jnp.absolute(expected_window))) - normalization,
            rtol=1e-12,
        )

    def test_upsampling_removes_grid_alignment_swing(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        def build(upsample):
            return TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={"upsample_factor": upsample},
            )

        coarse, fine = build(1), build(32)
        freqs = coarse.frequencies
        duration = float(coarse.detectors[0].data.duration)
        dt = duration / len(coarse.tc_array)
        amplitude = 50.0 / len(freqs)

        def marginalised_value(likelihood, t0):
            z = amplitude * jnp.exp(2j * jnp.pi * freqs * t0)
            return float(likelihood._reduce_phase_time(z))

        offsets = np.linspace(0.0, dt, 9)
        coarse_values = np.asarray([marginalised_value(coarse, t0) for t0 in offsets])
        fine_values = np.asarray([marginalised_value(fine, t0) for t0 in offsets])
        assert np.ptp(coarse_values) > 1.0
        assert np.ptp(fine_values) < 0.05

    def test_windowed_time_reduction_matches_masked_logsumexp(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        assert likelihood._tc_window_indices.size > 0

        rng = np.random.default_rng(0)
        n = len(likelihood.frequencies)
        d_inner_h = jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))

        padded = jnp.concatenate((likelihood.pad_low, d_inner_h, likelihood.pad_high))
        fft_ref = jnp.fft.fft(padded, norm="backward")
        in_window = (likelihood.tc_array > likelihood.tc_range[0]) & (
            likelihood.tc_array < likelihood.tc_range[1]
        )
        norm = jnp.log(len(likelihood.tc_array))

        ref_time = logsumexp(jnp.where(in_window, fft_ref.real, -jnp.inf)) - norm
        np.testing.assert_allclose(
            float(likelihood._reduce_time(d_inner_h)), float(ref_time), rtol=1e-12
        )

        ref_phase_time = (
            logsumexp(jnp.where(in_window, log_i0(jnp.absolute(fft_ref)), -jnp.inf))
            - norm
        )
        np.testing.assert_allclose(
            float(likelihood._reduce_phase_time(d_inner_h)),
            float(ref_phase_time),
            rtol=1e-12,
        )

    def test_empty_tc_window_raises_at_construction(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="tc_range"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={"tc_range": (1e-9, 2e-9)},
            )

    def test_time_marg_fixed_t_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have t_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={},
                fixed_parameters={"t_c": 0.0},
            )

    def test_time_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_time_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_time_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_time_marg_geq_base(self, detectors_and_waveform):
        """Time-marginalized log-likelihood is >= base - log(N_total).

        _reduce_time returns logsumexp(tc_range terms) - log(N_total), where
        N_total = len(tc_array) is the full FFT size.  Since t_c=0 lies within
        the default tc_range (-0.1, 0.1), the t=0 term is always included in
        the logsumexp, giving the tight lower bound marg ≥ base - log(N_total).
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
        )
        base = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        marg_result = marg.evaluate(example_params())
        base_result = base.evaluate(example_params())
        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result - jnp.log(len(marg.tc_array))

    # ── Phase marginalization ──────────────────────────────────────────────────

    def test_phase_marg_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_phase_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_phase_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_marg_geq_base(self, detectors_and_waveform):
        """Phase-marginalized likelihood must be >= base (I_0(x) >= 1)."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        base = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        marg_result = marg.evaluate(example_params())
        base_result = base.evaluate(example_params())
        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result

    # ── Phase + time marginalization ───────────────────────────────────────────

    def test_phase_time_marg_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={},
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_phase_time_marg_fixed_t_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have t_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                time_marginalization={},
                phase_marginalization=True,
                fixed_parameters={"t_c": 0.0},
            )

    def test_phase_time_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_time_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_phase_time_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_time_marg_geq_base(self, detectors_and_waveform):
        """Phase+time marg log-likelihood is >= base - log(N_total).

        Same reasoning as test_time_marg_geq_base: the time marginalisation
        normalises by len(tc_array), so the result is offset by -log(N_total)
        relative to the point estimate.  The t=0 term is always included,
        giving marg ≥ base - log(N_total).
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        base = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        marg_result = marg.evaluate(example_params())
        base_result = base.evaluate(example_params())
        assert jnp.isfinite(marg_result)
        assert marg_result >= base_result - jnp.log(len(marg.tc_array))

    def test_phase_time_marg_geq_phase_only(self, detectors_and_waveform):
        """Phase+time marg log-likelihood is >= phase-only - log(N_total).

        The time marginalisation normalises by len(tc_array), so the result
        is offset by -log(N_total).  The t=0 term is always included in the
        logsumexp, giving pt ≥ p - log(N_total).
        """
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        pt = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            time_marginalization={},
            phase_marginalization=True,
        )
        p = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        pt_result = pt.evaluate(example_params())
        p_result = p.evaluate(example_params())
        assert jnp.isfinite(pt_result)
        assert pt_result >= p_result - jnp.log(len(pt.tc_array))

    # ── Distance marginalization ───────────────────────────────────────────────

    def test_dist_marg_no_prior_raises(self):
        from pydantic import ValidationError

        from jimgw.core.single_event.likelihood import DistanceMargConfig

        with pytest.raises(ValidationError):
            DistanceMargConfig()  # type: ignore[call-arg]

    def test_dist_marg_fixed_d_L_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have d_L fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                distance_marginalization={"distance_prior": self.make_d_L_prior()},
                fixed_parameters={"d_L": 400.0},
            )

    def test_dist_marg_prior_missing_d_L_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(
            ValueError, match="must be a 1D prior with parameter_names="
        ):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                distance_marginalization={
                    "distance_prior": UniformPrior(
                        xmin=10.0, xmax=100.0, parameter_names=["M_c"]
                    )
                },
            )

    def test_dist_marg_n_dist_points_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="greater than or equal to 2"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                distance_marginalization={
                    "distance_prior": self.make_d_L_prior(),
                    "n_dist_points": 1,
                },
            )

    def test_dist_marg_negative_ref_dist_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="greater than 0"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                distance_marginalization={
                    "distance_prior": self.make_d_L_prior(),
                    "ref_dist": -100.0,
                },
            )

    def test_dist_marg_power_law_prior(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert isinstance(likelihood, TransientLikelihoodFD)

    def test_dist_marg_uniform_prior(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": UniformPrior(
                    xmin=100.0, xmax=5000.0, parameter_names=["d_L"]
                )
            },
        )
        assert isinstance(likelihood, TransientLikelihoodFD)

    def test_dist_marg_default_ref_dist(self, detectors_and_waveform):
        """Default ref_dist is the midpoint of [xmin, xmax]."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": self.make_d_L_prior(xmin=200.0, xmax=1000.0)
            },
        )
        assert jnp.isclose(likelihood.ref_dist, (200.0 + 1000.0) / 2.0)

    def test_dist_marg_custom_ref_dist(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": self.make_d_L_prior(),
                "ref_dist": 500.0,
            },
        )
        assert jnp.isclose(likelihood.ref_dist, 500.0)

    def test_dist_marg_log_weights_normalized(self, detectors_and_waveform):
        from jax.scipy.special import logsumexp

        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert jnp.isclose(logsumexp(likelihood.log_weights), 0.0, atol=1e-5)

    def test_dist_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        result = likelihood.evaluate(self.params_without_d_L())
        assert jnp.isfinite(result)

    def test_dist_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L()
        result = likelihood.evaluate(params)
        result_jit = jax.jit(likelihood.evaluate)(params)
        assert jnp.allclose(result, result_jit)

    def test_dist_marg_matches_base_near_true_distance(self, detectors_and_waveform):
        """With a narrow prior around d_L=400 the marginalized value ≈ point estimate."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_d_L = 400.0
        marg = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": UniformPrior(
                    xmin=true_d_L - 1.0, xmax=true_d_L + 1.0, parameter_names=["d_L"]
                ),
                "ref_dist": true_d_L,
            },
        )
        base = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        marg_result = marg.evaluate(self.params_without_d_L())
        base_result = base.evaluate(example_params())
        assert jnp.isfinite(marg_result)
        assert jnp.abs(marg_result - base_result) < 1.0

    def test_dist_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert jnp.isfinite(likelihood.evaluate(self.params_without_d_L()))

    # ── Phase + distance marginalization ──────────────────────────────────────

    def test_phase_dist_marg_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                phase_marginalization=True,
                distance_marginalization={"distance_prior": self.make_d_L_prior()},
                fixed_parameters={"phase_c": 0.0},
            )

    def test_phase_dist_marg_fixed_d_L_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have d_L fixed"):
            TransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                phase_marginalization=True,
                distance_marginalization={"distance_prior": self.make_d_L_prior()},
                fixed_parameters={"d_L": 400.0},
            )

    def test_phase_dist_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        result = likelihood.evaluate(self.params_without_d_L_phase())
        assert jnp.isfinite(result)

    def test_phase_dist_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        params = self.params_without_d_L_phase()
        result = likelihood.evaluate(params)
        result_jit = jax.jit(likelihood.evaluate)(params)
        assert jnp.allclose(result, result_jit)

    def test_phase_dist_marg_geq_distance_marginalized(self, detectors_and_waveform):
        """Phase+distance marginalization must be >= distance-only."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        d_prior = self.make_d_L_prior()
        pd = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": d_prior},
        )
        d = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            distance_marginalization={
                "distance_prior": d_prior,
                "ref_dist": pd.ref_dist,
            },
        )
        params = self.params_without_d_L_phase()
        pd_result = pd.evaluate(params)
        d_result = d.evaluate({**params, "phase_c": 0.0})
        assert jnp.isfinite(pd_result)
        assert pd_result >= d_result

    def test_phase_dist_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
            distance_marginalization={"distance_prior": self.make_d_L_prior()},
        )
        assert jnp.isfinite(likelihood.evaluate(self.params_without_d_L_phase()))

    # ── Callable fixed parameters ──────────────────────────────────────────────

    def test_callable_constant_matches_constant(self, detectors_and_waveform):
        """A callable returning a constant gives the same result as passing the constant."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        const = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": 0.0, "s2_z": 0.0},
        )
        callable_ = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        params = {
            k: v for k, v in example_params().items() if k not in ("s1_z", "s2_z")
        }
        assert jnp.allclose(
            const.evaluate(dict(params)), callable_.evaluate(dict(params))
        )

    def test_callable_reads_sampled_params(self, detectors_and_waveform):
        """A callable can derive its value from other sampled parameters."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": lambda p: p["_s1_z_raw"]},
        )
        params = example_params()
        params["_s1_z_raw"] = 0.0
        ref = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
        )
        result = likelihood.evaluate(dict(params))
        assert_all_finite(result)
        assert jnp.allclose(result, ref.evaluate(example_params()))

    def test_callable_does_not_mutate_input(self, detectors_and_waveform):
        """evaluate() must not mutate the caller's params dict."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": lambda p: 0.5},
        )
        params = example_params()
        keys_before = set(params.keys())
        values_before = {k: float(v) for k, v in params.items()}
        likelihood.evaluate(params)
        assert set(params.keys()) == keys_before
        for k, v in values_before.items():
            assert float(params[k]) == v

    def test_callable_jit_compatible(self, detectors_and_waveform):
        """Callable fixed_parameters must work under jax.jit."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        lambda_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        params = example_params()
        result = lambda_likelihood.evaluate(dict(params))
        result_jit = jax.jit(lambda_likelihood.evaluate)(dict(params))
        assert jnp.isfinite(result_jit)
        assert jnp.allclose(result, result_jit)

        transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time=gps, ifo=ifos[0]
        )
        transform_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"t_c": transform.backward},
        )
        params_with_tdet = {**example_params(), "t_det": 0.0}
        result_tr = transform_likelihood.evaluate(dict(params_with_tdet))
        result_tr_jit = jax.jit(transform_likelihood.evaluate)(dict(params_with_tdet))
        assert jnp.isfinite(result_tr_jit)
        assert jnp.allclose(result_tr, result_tr_jit)

    def test_callable_insertion_order_chaining(self, detectors_and_waveform):
        """Later callables in fixed_parameters see values written by earlier ones."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        seen_s1_z = []

        def set_nonzero_s1_z(p):
            return 0.5

        def read_s1_z_for_s2(p):
            seen_s1_z.append(float(p["s1_z"]))
            return float(p["s1_z"])

        likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"s1_z": set_nonzero_s1_z, "s2_z": read_s1_z_for_s2},
        )
        likelihood.evaluate(dict(example_params()))
        assert len(seen_s1_z) == 1
        assert seen_s1_z[0] == 0.5

    def test_callable_transform_backward(self, detectors_and_waveform):
        """transform.backward and an equivalent lambda must give the same result."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        transform = GeocentricArrivalTimeToDetectorArrivalTimeTransform(
            trigger_time=gps, ifo=ifos[0]
        )
        gmst = compute_gmst(gps)
        t_det_fixed = 0.0

        lambda_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={
                "t_c": lambda p: (
                    t_det_fixed - ifos[0].delay_from_geocenter(p["ra"], p["dec"], gmst)
                )
            },
        )
        transform_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters={"t_c": transform.backward},
        )
        params = {**example_params(), "t_det": t_det_fixed}
        result_lambda = lambda_likelihood.evaluate(dict(params))
        result_transform = transform_likelihood.evaluate(dict(params))
        assert jnp.isfinite(result_lambda)
        assert jnp.isfinite(result_transform)
        assert jnp.allclose(result_lambda, result_transform, atol=1e-6)


class TestHeterodynedTransientLikelihoodFD:
    # ── Initialization ────────────────────────────────────────────────────────

    def test_initialization_and_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ref_params = example_params()
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
        )
        assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)
        params = example_params()
        result = likelihood.evaluate(params)
        assert jnp.isfinite(result)
        assert jnp.allclose(result, base.evaluate(params))

    def test_initialization_stores_attributes(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
        )
        assert hasattr(likelihood, "freq_grid_low")
        assert hasattr(likelihood, "freq_grid_high")
        assert hasattr(likelihood, "bin_widths")
        for det in ifos:
            for attr in ("summary_data", "waveform_low_ref", "waveform_high_ref"):
                obj = getattr(likelihood, attr)
                assert det.name in obj
                assert jnp.isfinite(obj[det.name]).all()

    @pytest.mark.parametrize("phase_marginalization", [False, True])
    def test_direct_sum_time_marginalization_is_cached_and_jittable(
        self,
        detectors_and_waveform,
        phase_marginalization,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=128,
            reference_parameters=example_params(),
            phase_marginalization=phase_marginalization,
            time_marginalization={
                "tc_range": (-0.03, 0.03),
                "upsample_factor": 4,
                "phasor_block_size": 17,
            },
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        direct = likelihood.evaluate(params)

        np.testing.assert_allclose(
            likelihood.evaluate_from_waveform(params, cache),
            direct,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            jax.jit(likelihood.evaluate)(params),
            direct,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            likelihood.evaluate({**params, "t_c": 0.02}),
            direct,
        )

    def test_direct_sum_generates_bounded_phasor_blocks(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=64,
            reference_parameters=example_params(),
            time_marginalization={
                "tc_range": (-0.03, 0.03),
                "upsample_factor": 4,
                "phasor_block_size": 13,
            },
        )
        rng = np.random.default_rng(41)
        low_coeff = jnp.asarray(
            rng.normal(size=likelihood.n_bins) + 1j * rng.normal(size=likelihood.n_bins)
        )
        high_coeff = jnp.asarray(
            rng.normal(size=likelihood.n_bins) + 1j * rng.normal(size=likelihood.n_bins)
        )

        blocked = likelihood._direct_sum_network_match(low_coeff, high_coeff)
        expected = (
            likelihood._time_phasors(
                likelihood.tc_window,
                likelihood.freq_grid_low,
            )
            @ low_coeff
        )
        expected += (
            likelihood._time_phasors(
                likelihood.tc_window,
                likelihood.freq_grid_high,
            )
            @ high_coeff
        )
        np.testing.assert_allclose(
            np.asarray(blocked)[: len(likelihood.tc_window)],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        assert not hasattr(likelihood, "_tc_phase_low")
        assert not hasattr(likelihood, "_tc_phase_high")
        assert not hasattr(likelihood, "tc_array")
        assert likelihood._tc_time_blocks.size < (
            len(likelihood.tc_window) + likelihood.tc_phasor_block_size
        )
        assert likelihood._tc_valid_time_mask.size == likelihood._tc_time_blocks.size

    def test_direct_sum_rejects_underresolved_timing_width(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        with pytest.raises(ValueError, match="does not resolve timing_sigma_s"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=32,
                reference_parameters=example_params(),
                time_marginalization={
                    "tc_range": (-0.03, 0.03),
                    "upsample_factor": 1,
                    "timing_sigma_s": 2.0e-5,
                    "samples_per_timing_sigma": 4,
                },
            )

    def test_direct_sum_rejects_unbounded_time_grid_before_reference_work(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        reference_calls = []

        def counted_reference(frequencies, params):
            reference_calls.append(len(frequencies))
            return waveform(frequencies, params)

        with pytest.raises(ValueError, match="above the bounded limit"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                reference_waveform=counted_reference,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=32,
                reference_parameters=example_params(),
                time_marginalization={
                    "tc_range": (-0.03, 0.03),
                    "upsample_factor": 100_000_000,
                    "normalization": "window",
                },
            )

        assert reference_calls == []

    def test_direct_sum_can_normalize_over_explicit_time_prior(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=32,
            reference_parameters=example_params(),
            time_marginalization={
                "tc_range": (-0.03, 0.03),
                "upsample_factor": 4,
                "normalization": "window",
            },
        )

        assert likelihood.tc_normalization == "window"
        assert likelihood._tc_normalization_count == len(likelihood.tc_window)
        prior_width = 0.06
        requested_step = 2.0 / (
            float(ifos[0].data.sampling_frequency) * likelihood.tc_upsample
        )
        expected_count = int(np.ceil(prior_width / requested_step))
        midpoint_step = prior_width / expected_count
        assert len(likelihood.tc_window) == expected_count
        assert float(likelihood.tc_window[0]) == pytest.approx(
            -0.03 + 0.5 * midpoint_step
        )
        assert float(likelihood.tc_window[-1]) == pytest.approx(
            0.03 - 0.5 * midpoint_step
        )

    def test_window_normalization_rejects_prior_outside_data_support(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        duration = float(ifos[0].data.duration)

        with pytest.raises(ValueError, match="centered data duration"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=32,
                reference_parameters=example_params(),
                time_marginalization={
                    "tc_range": (-0.03, 0.5 * duration + 0.01),
                    "normalization": "window",
                },
            )

    def test_direct_sum_requires_explicit_frozen_dynamic_response(
        self,
        detectors_and_waveform,
        monkeypatch,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        xg_plan = _fixture_xg_plan(ifos, waveform, fmin, fmax, gps, n_bins=32)
        monkeypatch.setattr(
            type(ifos[0]),
            "response_is_time_dependent",
            True,
            raising=False,
        )
        kwargs = {
            "detectors": ifos,
            "waveform": waveform,
            "f_min": fmin,
            "f_max": fmax,
            "trigger_time": gps,
            "n_bins": 32,
            "reference_parameters": example_params(),
            "xg_plan": xg_plan,
        }

        with pytest.raises(ValueError, match="freeze_response=true"):
            HeterodynedTransientLikelihoodFD(
                **kwargs,
                time_marginalization={"tc_range": (-0.03, 0.03)},
            )

        likelihood = HeterodynedTransientLikelihoodFD(
            **kwargs,
            time_marginalization={
                "tc_range": (-0.03, 0.03),
                "freeze_response": True,
                "timing_sigma_s": 0.01,
                "normalization": "window",
            },
        )
        assert likelihood.freeze_time_dependent_response is True

    @pytest.mark.parametrize(
        ("time_config", "message"),
        [
            (
                {
                    "tc_range": (-0.03, 0.03),
                    "freeze_response": True,
                    "timing_sigma_s": 0.01,
                },
                "normalization='window'",
            ),
            (
                {
                    "tc_range": (-0.03, 0.03),
                    "freeze_response": True,
                    "normalization": "window",
                },
                "finite timing_sigma_s",
            ),
            (
                {
                    "tc_range": (0.01, 0.03),
                    "freeze_response": True,
                    "timing_sigma_s": 0.01,
                    "normalization": "window",
                },
                "pivot t_c=0",
            ),
        ],
    )
    def test_dynamic_direct_sum_requires_complete_physics_contract(
        self,
        detectors_and_waveform,
        monkeypatch,
        time_config,
        message,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        xg_plan = _fixture_xg_plan(ifos, waveform, fmin, fmax, gps, n_bins=32)
        monkeypatch.setattr(
            type(ifos[0]),
            "response_is_time_dependent",
            True,
            raising=False,
        )

        with pytest.raises(ValueError, match=message):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=32,
                reference_parameters=example_params(),
                time_marginalization=time_config,
                xg_plan=xg_plan,
            )

    def test_direct_sum_accepts_dominant_mode_timing_cache(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        xg_plan = _fixture_xg_plan(ifos, waveform, fmin, fmax, gps, n_bins=64)
        timed_waveform = DominantModeTimeCachedWaveform(waveform)
        for ifo in ifos:
            ifo.time_dependent_response = True

        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=timed_waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=64,
            reference_parameters=example_params(),
            phase_marginalization=True,
            time_marginalization={
                "tc_range": (-0.03, 0.03),
                "freeze_response": True,
                "timing_sigma_s": 0.01,
                "normalization": "window",
            },
            xg_plan=xg_plan,
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)

        assert "__tau__" in cache["low"]
        assert "__tau__" in cache["high"]
        np.testing.assert_allclose(
            likelihood.evaluate_from_waveform(params, cache),
            likelihood.evaluate(params),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_xg_heterodyne_rejects_iterative_reference_and_implicit_bins(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        timed_waveform = DominantModeTimeCachedWaveform(waveform)
        for ifo in ifos:
            ifo.time_dependent_response = True

        with pytest.raises(ValueError, match="fixed reference_parameters"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=timed_waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=32,
                prior=_waveform_cache_prior(),
            )

        with pytest.raises(ValueError, match="explicit, prequalified n_bins"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=timed_waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_parameters=example_params(),
            )

        with pytest.raises(ValueError, match="verified qualification manifest"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=timed_waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=32,
                reference_parameters=example_params(),
            )

    def test_rejected_xg_setup_does_not_slice_dense_frequency_data(
        self,
        detectors_and_waveform,
        monkeypatch,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        timed_waveform = DominantModeTimeCachedWaveform(waveform)
        for ifo in ifos:
            ifo.time_dependent_response = True

        def fail_if_called(*args, **kwargs):
            del args, kwargs
            pytest.fail("frequency slicing ran before the XG setup gate")

        monkeypatch.setattr(
            HeterodynedTransientLikelihoodFD,
            "_set_detector_frequency_bounds",
            fail_if_called,
        )

        with pytest.raises(ValueError, match="fixed reference_parameters"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=timed_waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=32,
                prior=_waveform_cache_prior(),
            )

    def test_oversized_bin_plan_fails_before_frequency_slicing(
        self,
        detectors_and_waveform,
        monkeypatch,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        def fail_if_called(*args, **kwargs):
            del args, kwargs
            pytest.fail("frequency slicing ran before the bin limit gate")

        monkeypatch.setattr(
            HeterodynedTransientLikelihoodFD,
            "_set_detector_frequency_bounds",
            fail_if_called,
        )

        with pytest.raises(ValueError, match="bounded bin limit"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=4_194_304,
                reference_parameters=example_params(),
            )

    def test_reference_summary_construction_is_chunked(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        chunk_size = 127
        call_sizes = []

        def counted_reference(frequencies, params):
            call_sizes.append(len(frequencies))
            return waveform(frequencies, params)

        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            reference_waveform=counted_reference,
            reference_chunk_size=chunk_size,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=32,
            reference_parameters=example_params(),
        )

        assert jnp.isfinite(likelihood.evaluate(example_params()))
        # Each chunk carries two native prefix samples so spacing-reading
        # waveform backends see the true grid spacing on every call.
        assert max(call_sizes) <= chunk_size + 2
        assert len(call_sizes) > len(ifos) + 2

    def test_no_reference_params_and_no_prior_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
            )

    def test_incomplete_reference_parameters_fail_before_waveform_construction(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform

        with pytest.raises(ValueError, match=r"incomplete.*dec"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=32,
                reference_parameters={
                    key: value
                    for key, value in example_params().items()
                    if key != "dec"
                },
            )

    def test_evaluate_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_cached_waveform_matches_full_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=32,
            reference_parameters=example_params(),
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        assert jnp.allclose(
            likelihood.evaluate_from_waveform(params, cache),
            likelihood.evaluate(params),
        )

    def test_cached_waveform_supports_cache_reusing_changes(
        self, detectors_and_waveform
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=32,
            reference_parameters=example_params(),
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)
        projected = {
            **params,
            "d_L": params["d_L"] * 1.2,
            "psi": params["psi"] + 0.1,
            "t_c": 0.002,
        }
        assert jnp.allclose(
            likelihood.evaluate_from_waveform(projected, cache),
            likelihood.evaluate(projected),
        )

    def test_evaluate_bypasses_waveform_cache_machinery(
        self, detectors_and_waveform, monkeypatch
    ):
        # evaluate() must generate the waveform directly at the true d_L; the
        # waveform-cache machinery (unit-distance normalization when supported,
        # rescaling on the way out) is reserved for the SwiG cache path
        # (generate_waveform()/evaluate_from_waveform()) and must not add
        # overhead to every other sampler's likelihood calls.
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=32,
            reference_parameters=example_params(),
        )

        def _unexpected(*args, **kwargs):
            raise AssertionError("evaluate() must not use the waveform-cache machinery")

        monkeypatch.setattr(likelihood, "_waveform_sky_for_cache", _unexpected)
        monkeypatch.setattr(likelihood, "_waveform_sky_from_cache", _unexpected)

        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_waveform_cache_uses_heterodyned_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            fixed_parameters={"d_L": 400.0},
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=32,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["ra", "dec"],
            ["psi"],
            ["t_c"],
        ]
        jim = Jim(
            likelihood,
            _waveform_cache_prior(),
            BlackJAXSwiGConfig(blocks=blocks, n_live=8, n_delete_frac=0.25),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
        )
        assert jim.sampler._rebuild_required_by_block == {
            (0, 1): True,
            (2, 3): True,
            (4,): True,
            (5, 6): False,
            (7,): False,
            (8,): False,
        }

    def test_evaluate_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
        )
        # Per-detector frequency bounds do not change the reference waveform
        # support, so the relative-binning grid still starts at the global f_min.
        assert jnp.isclose(likelihood.freq_grid_low[0], fmin)
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_binning_edges_invert_phase_envelope_without_dense_grid(self):
        frequencies = jnp.linspace(5.0, 2_048.0, 1_000_001)
        n_bins = 257

        edges = HeterodynedTransientLikelihoodFD._make_binning_scheme(
            frequencies,
            n_bins,
        )
        edge_phase = HeterodynedTransientLikelihoodFD._max_phase_diff(
            edges,
            edges[0],
            edges[-1],
        )
        expected_phase = jnp.linspace(0.0, edge_phase[-1], n_bins + 1)

        np.testing.assert_allclose(edge_phase, expected_phase, rtol=2e-12, atol=2e-12)
        assert jnp.all(jnp.diff(edges) > 0)

    def test_reference_projection_null_is_rejected(self):
        source = {
            "p": jnp.ones(3, dtype=jnp.complex128),
            "c": 1j * jnp.ones(3, dtype=jnp.complex128),
            "__tau__": jnp.arange(3.0),
        }

        with pytest.raises(ValueError, match="response null"):
            HeterodynedTransientLikelihoodFD._validate_reference_projection(
                "CE",
                source,
                jnp.asarray((1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j)),
                "low",
            )

    def test_qualified_bin_edge_digest_checks_final_retained_plan(
        self,
        detectors_and_waveform,
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ifos[0].finite_arm_response = True

        with pytest.raises(ValueError, match="bin edges do not match"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=32,
                reference_parameters=example_params(),
                xg_plan=_VerifiedXGPlan("0" * 64, _XG_PLAN_AUTHORITY),
            )

    @pytest.mark.parametrize(
        ("n_bins", "epsilon", "match"),
        [
            (32, 0.5, "mutually exclusive"),
            (0, None, "positive integer"),
            (True, None, "positive integer"),
            (None, 0.0, "positive number"),
            (None, np.inf, "positive number"),
        ],
    )
    def test_invalid_binning_parameters_raise(
        self, detectors_and_waveform, n_bins, epsilon, match
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match=match):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                n_bins=n_bins,
                epsilon=epsilon,
                reference_parameters=example_params(),
            )

    def test_maximize_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_params = example_params()
        for ifo in ifos:
            ifo.inject_signal(
                duration=4.0,
                sampling_frequency=fmax * 2,
                trigger_time=gps,
                waveform_model=waveform,
                parameters=true_params,
                f_min=fmin,
                f_max=fmax,
                zero_noise=True,
            )
        base = TransientLikelihoodFD(
            detectors=ifos, waveform=waveform, f_min=fmin, f_max=fmax, trigger_time=gps
        )
        ll_injected = float(base.evaluate(true_params))
        fixed_parameters = {
            k: v for k, v in true_params.items() if k not in ("M_c", "eta")
        }
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            fixed_parameters=fixed_parameters,
            prior=CombinePrior(
                [
                    UniformPrior(25.0, 35.0, parameter_names=["M_c"]),
                    UniformPrior(0.125, 1.0, parameter_names=["q"]),
                ]
            ),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
            optimizer_popsize=10,
            optimizer_n_steps=50,
        )
        result = likelihood.reference_parameters.copy()
        expected_keys = {
            "M_c",
            "eta",
            "s1_z",
            "s2_z",
            "d_L",
            "t_c",
            "phase_c",
            "iota",
            "psi",
            "ra",
            "dec",
            "trigger_time",
            "gmst",
        }
        assert set(result.keys()) == expected_keys
        for val in result.values():
            assert jnp.isfinite(val)
        assert jnp.isfinite(likelihood.evaluate(result))
        assert jnp.isclose(float(base.evaluate(result)), ll_injected)
        common_keys_allclose(result, true_params)

    def test_maximize_likelihood_stops_early_at_optimizer_target(
        self, detectors_and_waveform, caplog
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        true_params = example_params()
        for ifo in ifos:
            ifo.inject_signal(
                duration=4.0,
                sampling_frequency=fmax * 2,
                trigger_time=gps,
                waveform_model=waveform,
                parameters=true_params,
                f_min=fmin,
                f_max=fmax,
                zero_noise=True,
            )
        fixed_parameters = {
            k: v for k, v in true_params.items() if k not in ("M_c", "eta")
        }
        # A trivially low target is satisfied by the first generation's best
        # fitness, so the loop must exit long before optimizer_n_steps.
        with caplog.at_level(
            logging.DEBUG, logger="jimgw.core.single_event.likelihood"
        ):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                fixed_parameters=fixed_parameters,
                prior=CombinePrior(
                    [
                        UniformPrior(25.0, 35.0, parameter_names=["M_c"]),
                        UniformPrior(0.125, 1.0, parameter_names=["q"]),
                    ]
                ),
                likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
                optimizer_popsize=10,
                optimizer_n_steps=50,
                optimizer_target=-1e10,
            )
        [finished_message] = [
            record.message
            for record in caplog.records
            if "CMA-ES finished after" in record.message
        ]
        generations = int(finished_message.split("after ")[1].split(" generations")[0])
        assert generations < 50

    def test_low_frequency_reference_cutoff_does_not_reindex_summary_data(
        self, detectors_and_waveform, monkeypatch
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        reference_fmin = 80.0
        requested_n_bins = 32

        def fake_compute_coefficients(likelihood, detector, h_ref, f_bins):
            freqs = detector.sliced_frequencies
            freqs_broadcast = freqs[None, :]
            left_bounds = f_bins[:-1][:, None]
            right_bounds = f_bins[1:][:, None]

            mask = (freqs_broadcast >= left_bounds) & (freqs_broadcast < right_bounds)

            n_freqs = len(freqs)
            n_bins = len(f_bins) - 1
            assert n_freqs > len(f_bins), f"{n_freqs = }, {len(f_bins) = }"
            assert likelihood.n_bins == n_bins, f"{likelihood.n_bins = }, {n_bins = }"
            assert n_bins <= requested_n_bins
            assert freqs[0] == fmin
            assert f_bins[0] >= reference_fmin
            assert mask.shape == (n_bins, n_freqs), (
                f"{mask.shape = }, expected: ({n_bins}, {n_freqs})"
            )
            coeffs = jnp.arange(n_bins, dtype=jnp.float64)
            return jnp.array([coeffs + nn * 100 for nn in range(4)])

        monkeypatch.setattr(
            HeterodynedTransientLikelihoodFD,
            "_compute_coefficients",
            fake_compute_coefficients,
        )

        def reference_waveform(frequencies, params):
            waveform_sky = waveform(frequencies, params)
            mask = frequencies >= reference_fmin
            return {
                polarization: jnp.where(mask, strain, jnp.zeros_like(strain))
                for polarization, strain in waveform_sky.items()
            }

        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            reference_waveform=reference_waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            n_bins=requested_n_bins,
            reference_parameters=example_params(),
        )
        assert likelihood.n_bins < requested_n_bins
        assert likelihood.freq_grid_low[0] >= reference_fmin

        expected = jnp.arange(likelihood.n_bins, dtype=jnp.float64)
        expected_arr = jnp.array([expected + nn * 100 for nn in range(4)])
        for detector in ifos:
            assert jnp.array_equal(likelihood.summary_data[detector.name], expected_arr)

    # ── Phase marginalization ──────────────────────────────────────────────────

    def test_phase_marg_fixed_phase_c_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match="Cannot have phase_c fixed"):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_parameters=example_params(),
                phase_marginalization=True,
                fixed_parameters={"phase_c": 0.0},
            )

    def test_phase_marg_no_reference_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError):
            HeterodynedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                phase_marginalization=True,
            )

    def test_phase_marg_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        assert isinstance(likelihood, HeterodynedTransientLikelihoodFD)
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_marg_jit_matches(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        params = example_params()
        assert jnp.allclose(
            likelihood.evaluate(params), jax.jit(likelihood.evaluate)(params)
        )

    def test_phase_marg_different_fmin(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
            phase_marginalization=True,
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_phase_marg_matches_base_at_ref_params(self, detectors_and_waveform):
        """At reference params the het phase-marg should closely match non-het phase-marg."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ref_params = example_params()
        phase_likelihood = TransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            phase_marginalization=True,
        )
        het_phase_likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=ref_params,
            phase_marginalization=True,
        )
        params = example_params()
        het_result = het_phase_likelihood.evaluate(params)
        phase_result = phase_likelihood.evaluate(params)
        assert jnp.isfinite(het_result)
        assert jnp.allclose(het_result, phase_result, atol=1e-1)

    # ── Callable fixed parameters ──────────────────────────────────────────────

    def test_callable_fixed_parameter(self, detectors_and_waveform):
        """Callable fixed_parameters must work in HeterodynedTransientLikelihoodFD."""
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_parameters=example_params(),
            fixed_parameters={"s1_z": lambda p: 0.0, "s2_z": lambda p: 0.0},
        )
        assert jnp.isfinite(likelihood.evaluate(example_params()))


class TestMultibandedTransientLikelihoodFD:
    def test_rejects_unqualified_xg_response(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        ifos[0].finite_arm_response = True

        with pytest.raises(ValueError, match="not qualified with the multiband"):
            MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_chirp_mass=20.0,
            )

    # ── Initialization ────────────────────────────────────────────────────────

    def test_infers_banding_parameters_from_prior(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        mc_prior = UniformPrior(15.0, 30.0, parameter_names=["M_c"])
        tc_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
        prior = CombinePrior([CombinePrior([mc_prior]), tc_prior])

        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            prior=prior,
        )

        t_end = min(
            float(ifo.data.start_time) + float(ifo.data.duration) - gps for ifo in ifos
        )
        expected_time_offset = t_end - tc_prior.xmin + EARTH_RADIUS_LIGHT_S
        expected_delta_f_end = 100.0 / (t_end - tc_prior.xmax - EARTH_RADIUS_LIGHT_S)

        assert likelihood.reference_chirp_mass == mc_prior.xmin
        assert jnp.isclose(likelihood.time_offset, expected_time_offset)
        assert jnp.isclose(likelihood.delta_f_end, expected_delta_f_end)

    def test_unbounded_time_prior_uses_defaults(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        prior = CombinePrior([GaussianPrior(0.0, 1.0, parameter_names=["t_c"])])

        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            prior=prior,
            reference_chirp_mass=20.0,
        )

        assert likelihood.time_offset == 2.12
        assert likelihood.delta_f_end == 53.0

    @pytest.mark.parametrize(
        ("prior", "match"),
        [
            (None, "Either reference_chirp_mass or a prior"),
            (
                CombinePrior([UniformPrior(-0.1, 0.1, parameter_names=["t_c"])]),
                "no M_c prior found",
            ),
            (
                CombinePrior([GaussianPrior(20.0, 1.0, parameter_names=["M_c"])]),
                "no M_c prior found",
            ),
        ],
    )
    def test_missing_reference_chirp_mass_raises(
        self, detectors_and_waveform, prior, match
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        with pytest.raises(ValueError, match=match):
            MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                prior=prior,
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"reference_chirp_mass": 0.0}, "reference_chirp_mass"),
            ({"highest_mode": 0}, "highest_mode"),
            ({"accuracy_factor": 0.0}, "accuracy_factor"),
            ({"time_offset": -1.0}, "time_offset"),
            ({"delta_f_end": 0.0}, "delta_f_end"),
            ({"min_banding_duration": -1.0}, "min_banding_duration"),
            ({"max_banding_frequency": 0.0}, "max_banding_frequency"),
        ],
    )
    def test_invalid_banding_parameters_raise(
        self, detectors_and_waveform, kwargs, match
    ):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        params = {"reference_chirp_mass": 20.0, **kwargs}
        with pytest.raises(ValueError, match=match):
            MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                **params,
            )

    def test_frequency_band_helpers_create_multiple_bands(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
            time_offset=0.0,
        )

        tau, dtaudf = likelihood._compute_tau_dtaudf(100.0)
        fnext, dfnext = likelihood._find_starting_frequency(4.0, fmin)
        no_fnext, no_dfnext = likelihood._find_starting_frequency(
            4.0, likelihood.max_banding_frequency
        )

        assert tau > 0
        assert dtaudf < 0
        assert fnext is not None
        assert dfnext is not None
        assert fmin < fnext < likelihood.max_banding_frequency
        assert dfnext > 0
        assert (no_fnext, no_dfnext) == (None, None)
        assert likelihood.n_bands > 1

    def test_initialization(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        assert isinstance(likelihood, MultibandedTransientLikelihoodFD)
        assert likelihood.minimum_frequency == fmin
        assert likelihood.maximum_frequency == fmax
        assert likelihood.trigger_time == gps
        assert hasattr(likelihood, "gmst")
        assert likelihood.reference_chirp_mass == 20.0
        assert likelihood.time_offset == 2.12
        assert likelihood.delta_f_end == 53.0
        assert likelihood.n_bands > 0

    def test_band_setup(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        assert len(likelihood.unique_frequencies) > 0
        assert len(likelihood.unique_to_original) > 0
        assert len(likelihood.unique_frequencies) <= len(likelihood.unique_to_original)
        for ifo in ifos:
            assert ifo.name in likelihood.linear_coeffs
            assert ifo.name in likelihood.quadratic_coeffs
            assert len(likelihood.linear_coeffs[ifo.name]) == len(
                likelihood.unique_to_original
            )

    def test_uninitialized_data_raises(self):
        gps = 1126259462.4
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_psd(
                PowerSpectrum.from_file(
                    str(FIXTURES_DIR / f"GW150914_psd_{ifo.name}.npz")
                )
            )
        with pytest.raises(ValueError, match="does not have initialized data"):
            MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
                reference_chirp_mass=20.0,
            )

    def test_partially_initialized_data_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        extra = get_H1()
        extra.set_psd(
            PowerSpectrum.from_file(
                str(FIXTURES_DIR / f"GW150914_psd_{extra.name}.npz")
            )
        )
        with pytest.raises(ValueError, match=r"H1.*does not have initialized data"):
            MultibandedTransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_chirp_mass=20.0,
            )

    def test_uninitialized_psd_raises(self):
        gps = 1126259462.4
        ifos = [get_H1(), get_L1()]
        for ifo in ifos:
            ifo.set_data(
                Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{ifo.name}.npz"))
            )
        with pytest.raises(ValueError, match="does not have initialized PSD"):
            MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=RippleIMRPhenomD(f_ref=20.0),
                f_min=20.0,
                f_max=1024.0,
                trigger_time=gps,
                reference_chirp_mass=20.0,
            )

    def test_partially_initialized_psd_raises(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        extra = get_H1()
        extra.set_data(
            Data.from_file(str(FIXTURES_DIR / f"GW150914_strain_{extra.name}.npz"))
        )
        with pytest.raises(ValueError, match=r"H1.*does not have initialized PSD"):
            MultibandedTransientLikelihoodFD(
                detectors=ifos + [extra],
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                reference_chirp_mass=20.0,
            )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def test_cached_waveform_matches_direct_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        params = example_params()
        cache = likelihood.generate_waveform(params)

        assert jnp.allclose(
            likelihood.evaluate(params),
            likelihood.evaluate_from_waveform(params, cache),
        )
        assert jnp.allclose(
            likelihood.evaluate_from_waveform(params, cache),
            jax.jit(likelihood.evaluate_from_waveform)(params, cache),
        )

        projected = {
            **params,
            "d_L": params["d_L"] * 1.2,
            "psi": params["psi"] + 0.1,
            "t_c": 0.002,
        }
        assert jnp.allclose(
            likelihood.evaluate(projected),
            likelihood.evaluate_from_waveform(projected, cache),
        )

    def test_evaluate_bypasses_waveform_cache_machinery(
        self, detectors_and_waveform, monkeypatch
    ):
        # evaluate() must generate the waveform directly at the true d_L; the
        # waveform-cache machinery (unit-distance normalization when supported,
        # rescaling on the way out) is reserved for the SwiG cache path
        # (generate_waveform()/evaluate_from_waveform()) and must not add
        # overhead to every other sampler's likelihood calls.
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )

        def _unexpected(*args, **kwargs):
            raise AssertionError("evaluate() must not use the waveform-cache machinery")

        monkeypatch.setattr(likelihood, "_waveform_sky_for_cache", _unexpected)
        monkeypatch.setattr(likelihood, "_waveform_sky_from_cache", _unexpected)

        assert jnp.isfinite(likelihood.evaluate(example_params()))

    def test_waveform_cache_uses_multibanded_likelihood(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            fixed_parameters={"d_L": 400.0, "phase_c": 0.0},
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        blocks = [
            ["M_c", "q"],
            ["s1_z", "s2_z"],
            ["iota"],
            ["ra", "dec"],
            ["psi"],
            ["t_c"],
        ]
        jim = Jim(
            likelihood,
            _waveform_cache_prior(),
            BlackJAXSwiGConfig(blocks=blocks, n_live=8, n_delete_frac=0.25),
            likelihood_transforms=[MassRatioToSymmetricMassRatioTransform],
        )
        assert jim.sampler._rebuild_required_by_block == {
            (0, 1): True,
            (2, 3): True,
            (4,): True,
            (5, 6): False,
            (7,): False,
            (8,): False,
        }

    def test_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        params = example_params()
        ll = likelihood.evaluate(params)
        assert jnp.isfinite(ll)
        ll_jit = jax.jit(likelihood.evaluate)(params)
        assert jnp.isfinite(ll_jit)
        assert jnp.allclose(ll, ll_jit)
        ll_diff = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min={"H1": fmin, "L1": fmin + 1.0},
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        ).evaluate(params)
        assert jnp.isfinite(ll_diff)

    def test_evaluate_does_not_mutate_params(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        likelihood = MultibandedTransientLikelihoodFD(
            detectors=ifos,
            waveform=waveform,
            f_min=fmin,
            f_max=fmax,
            trigger_time=gps,
            reference_chirp_mass=20.0,
        )
        params = example_params()
        keys_before = set(params.keys())
        values_before = {k: float(v) for k, v in params.items()}
        likelihood.evaluate(params)
        assert set(params.keys()) == keys_before
        for k, v in values_before.items():
            assert float(params[k]) == v

    @pytest.mark.slow
    def test_accuracy_factor_evaluation(self, detectors_and_waveform):
        ifos, waveform, fmin, fmax, gps = detectors_and_waveform
        params = example_params()
        for acc in [1.0, 5.0, 10.0]:
            likelihood = MultibandedTransientLikelihoodFD(
                detectors=ifos,
                waveform=waveform,
                f_min=fmin,
                f_max=fmax,
                trigger_time=gps,
                accuracy_factor=acc,
                reference_chirp_mass=20.0,
            )
            assert jnp.isfinite(likelihood.evaluate(params)), (
                f"Not finite for accuracy_factor={acc}"
            )
