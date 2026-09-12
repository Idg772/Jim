"""Unit tests for transform inference logic."""

import pytest
from pydantic import ValidationError

from jimgw.cli._config import PipelineConfig, PriorConfig, SamplingConfig, UniformSpec
from jimgw.cli._prior import adapt_prior_for_ns_time

_MINIMAL_DATA = {
    "type": "gwosc",
    "detectors": ["H1", "L1"],
    "trigger_time": 1126259462.4,
    "duration": 4.0,
    "psd_duration": 1024.0,
}

_MINIMAL_PRIOR = {
    "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
    "q": {"type": "uniform", "min": 0.125, "max": 1.0},
}


def _make_pipeline_cfg(
    prior_raw=None,
    sky_frame="detector",
    time_frame="detector",
    inclination_coordinate="iota",
    distance_coordinate="d_L",
    chirp_mass_coordinate="M_c",
):
    """Build a minimal PipelineConfig, merging prior_raw on top of _MINIMAL_PRIOR."""
    return PipelineConfig.model_validate(
        {
            "data": _MINIMAL_DATA,
            "waveform": {"approximant": "IMRPhenomXAS"},
            "prior": {**_MINIMAL_PRIOR, **(prior_raw or {})},
            "likelihood": {"f_min": 20.0, "f_max": 1024.0},
            "sampler": {"type": "flowmc"},
            "output": {"dir": "tests/tmp/test"},
            "sampling": {
                "sky_frame": sky_frame,
                "time_frame": time_frame,
                "inclination_coordinate": inclination_coordinate,
                "distance_coordinate": distance_coordinate,
                "chirp_mass_coordinate": chirp_mass_coordinate,
            },
        }
    )


def _make_ifos():
    from jimgw.core.single_event.detector import get_detector_preset

    preset = get_detector_preset()
    return [preset["H1"], preset["L1"]]


TRIGGER_TIME = 1126259462.4


def _infer(
    prior_params,
    sky_frame="detector",
    time_frame="detector",
    inclination_coordinate="iota",
    distance_coordinate="d_L",
    chirp_mass_coordinate="M_c",
):

    from jimgw.cli._transforms import (
        infer_likelihood_transforms,
        infer_sample_transforms,
    )

    ifos = _make_ifos()
    cfg = SamplingConfig(
        sky_frame=sky_frame,
        time_frame=time_frame,
        inclination_coordinate=inclination_coordinate,
        distance_coordinate=distance_coordinate,
        chirp_mass_coordinate=chirp_mass_coordinate,
    )  # type: ignore[arg-type]
    prior_cfg = PriorConfig.model_validate({})
    sample_t = infer_sample_transforms(
        frozenset(prior_params),
        TRIGGER_TIME,
        ifos,
        cfg,
        prior_cfg=prior_cfg,
    )
    lh_t = infer_likelihood_transforms(
        frozenset(prior_params),
        TRIGGER_TIME,
        ifos,
        cfg,
        20.0,
    )
    return sample_t, lh_t


# ---------------------------------------------------------------------------
# Standard CBC case (the GW150914 default)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gw150914_default_transforms():
    params = {
        "M_c",
        "q",
        "s1_z",
        "s2_z",
        "iota",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "ra",
        "dec",
    }
    sample_t, lh_t = _infer(params)

    sample_names = [type(t).__name__ for t in sample_t]
    assert "GeocentricArrivalTimeToDetectorArrivalTimeTransform" in sample_names
    assert "SkyFrameToDetectorFrameSkyPositionTransform" in sample_names
    assert len(sample_t) == 2

    # Only q→eta in likelihood space; reverse sky/time handled by reversed sample transforms
    assert len(lh_t) == 1
    assert "q" in str(lh_t[0].name_mapping)


# ---------------------------------------------------------------------------
# Geocentric sky (no sky sample transform)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_geocentric_sky_no_sky_sample_transform():
    params = {
        "M_c",
        "q",
        "iota",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "ra",
        "dec",
        "s1_z",
        "s2_z",
    }
    sample_t, _ = _infer(params, sky_frame="geocentric")

    sample_names = [type(t).__name__ for t in sample_t]
    assert "SkyFrameToDetectorFrameSkyPositionTransform" not in sample_names
    assert "GeocentricArrivalTimeToDetectorArrivalTimeTransform" in sample_names


@pytest.mark.slow
def test_cos_iota_sample_coordinate_round_trips_to_physical_iota():
    sample_t, _ = _infer(
        {"M_c", "q", "iota"},
        inclination_coordinate="cos_iota",
    )

    transforms = [
        transform
        for transform in sample_t
        if type(transform).__name__ == "CosineTransform"
    ]
    assert len(transforms) == 1
    transformed, _ = transforms[0].transform({"iota": 1.2})
    recovered, _ = transforms[0].inverse(transformed)
    assert recovered["iota"] == pytest.approx(1.2)


def test_cos_iota_sample_coordinate_requires_sine_iota_prior():
    with pytest.raises(ValidationError, match="type='sine'"):
        _make_pipeline_cfg(
            prior_raw={
                "iota": {"type": "uniform", "min": 0.0, "max": 3.14159},
            },
            inclination_coordinate="cos_iota",
        )


def test_cos_iota_sample_coordinate_requires_iota_prior():
    with pytest.raises(ValidationError, match="physical 'iota'"):
        _make_pipeline_cfg(inclination_coordinate="cos_iota")


def test_cos_iota_sample_coordinate_rejects_duplicate_prior_coordinate():
    with pytest.raises(ValidationError, match="also contains 'cos_iota'"):
        _make_pipeline_cfg(
            prior_raw={
                "iota": {"type": "sine"},
                "cos_iota": {"type": "uniform", "min": -1.0, "max": 1.0},
            },
            inclination_coordinate="cos_iota",
        )


def test_cos_iota_unit_cube_transform_is_applied_once():
    from jimgw.cli._transforms import infer_sample_transforms

    sampling_cfg = SamplingConfig(inclination_coordinate="cos_iota")
    prior_cfg = PriorConfig.model_validate({"iota": {"type": "sine"}})
    transforms = infer_sample_transforms(
        frozenset({"iota"}),
        TRIGGER_TIME,
        _make_ifos(),
        sampling_cfg,
        unit_cube=True,
        prior_cfg=prior_cfg,
    )

    mappings = [transform.name_mapping for transform in transforms]
    assert mappings.count((["iota"], ["cos_iota"])) == 1
    assert mappings.count((["cos_iota"], ["cos_iota_unit"])) == 1


# ---------------------------------------------------------------------------
# Geocentric time frame (t_c sampled directly)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_geocentric_time_no_time_sample_transform():
    params = {
        "M_c",
        "q",
        "iota",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "ra",
        "dec",
        "s1_z",
        "s2_z",
    }
    sample_t, _ = _infer(params, time_frame="geocentric")

    sample_names = [type(t).__name__ for t in sample_t]
    assert "GeocentricArrivalTimeToDetectorArrivalTimeTransform" not in sample_names
    assert "SkyFrameToDetectorFrameSkyPositionTransform" in sample_names


# ---------------------------------------------------------------------------
# Detector-frame sky prior (azimuth/zenith in prior)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_detector_frame_sky_prior():
    params = {
        "M_c",
        "q",
        "iota",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "azimuth",
        "zenith",
        "s1_z",
        "s2_z",
    }
    sample_t, lh_t = _infer(params)

    sample_names = [type(t).__name__ for t in sample_t]
    # No sky sample transform (already in detector frame)
    assert "SkyFrameToDetectorFrameSkyPositionTransform" not in sample_names
    # Reverse sky transform must appear in likelihood transforms
    lh_repr = [repr(t) for t in lh_t]
    assert any("zenith" in r or "azimuth" in r for r in lh_repr)


# ---------------------------------------------------------------------------
# Detector-frame time prior (t_det in prior)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_detector_frame_time_prior():
    params = {
        "M_c",
        "q",
        "iota",
        "d_L",
        "t_det",
        "phase_c",
        "psi",
        "ra",
        "dec",
        "s1_z",
        "s2_z",
    }
    sample_t, lh_t = _infer(params)

    sample_names = [type(t).__name__ for t in sample_t]
    # No time sample transform
    assert "GeocentricArrivalTimeToDetectorArrivalTimeTransform" not in sample_names
    # Reverse time transform must appear in likelihood transforms
    lh_repr = [repr(t) for t in lh_t]
    assert any("t_det" in r or "t_c" in r for r in lh_repr)


# ---------------------------------------------------------------------------
# J-frame spin angles
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_j_frame_spin_angles():
    params = {
        "M_c",
        "q",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "ra",
        "dec",
        "theta_jn",
        "phi_jl",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "a_1",
        "a_2",
    }
    sample_t, lh_t = _infer(params)

    # Spin physics transforms are in likelihood_transforms (prior → likelihood space),
    # not in sample_transforms — so the sampler explores in J-frame angle space.
    sample_names = [type(t).__name__ for t in sample_t]
    assert "SpinAnglesToCartesianSpinTransform" not in sample_names

    lh_names = [type(t).__name__ for t in lh_t]
    assert "SpinAnglesToCartesianSpinTransform" in lh_names


def test_j_frame_iota_conflict():
    with pytest.raises(ValidationError, match="iota"):
        _make_pipeline_cfg(
            prior_raw={
                "theta_jn": {"type": "uniform", "min": 0.0, "max": 3.14159},
                "phi_jl": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "tilt_1": {"type": "sine"},
                "tilt_2": {"type": "sine"},
                "phi_12": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "a_1": {"type": "uniform", "min": 0.0, "max": 0.99},
                "a_2": {"type": "uniform", "min": 0.0, "max": 0.99},
                "iota": {"type": "sine"},  # conflict
                "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                "q": {"type": "uniform", "min": 0.125, "max": 1.0},
            }
        )


def test_j_frame_partial_params_rejected():
    with pytest.raises(ValidationError, match="missing"):
        _make_pipeline_cfg(
            prior_raw={
                "theta_jn": {"type": "uniform", "min": 0.0, "max": 3.14159},
                "phi_jl": {"type": "uniform", "min": 0.0, "max": 6.28318},
                # only 2 of 7 J-frame params
                "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
                "q": {"type": "uniform", "min": 0.125, "max": 1.0},
            }
        )


# ---------------------------------------------------------------------------
# Spherical per-spin
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sphere_spin_transform():
    params = {
        "M_c",
        "q",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "iota",
        "ra",
        "dec",
        "s1_mag",
        "s1_theta",
        "s1_phi",
        "s2_mag",
        "s2_theta",
        "s2_phi",
    }
    sample_t, lh_t = _infer(params)

    # Sphere spin physics transforms are in likelihood_transforms, not sample_transforms.
    sample_names = [type(t).__name__ for t in sample_t]
    assert "SphereSpinToCartesianSpinTransform" not in sample_names

    lh_names = [type(t).__name__ for t in lh_t]
    assert lh_names.count("SphereSpinToCartesianSpinTransform") == 2


# ---------------------------------------------------------------------------
# Cartesian spins — no transform needed
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cartesian_spins_no_transform():
    params = {
        "M_c",
        "q",
        "d_L",
        "t_c",
        "phase_c",
        "psi",
        "iota",
        "ra",
        "dec",
        "s1_x",
        "s1_y",
        "s1_z",
        "s2_x",
        "s2_y",
        "s2_z",
    }
    sample_t, _ = _infer(params)

    sample_names = [type(t).__name__ for t in sample_t]
    assert "SphereSpinToCartesianSpinTransform" not in sample_names
    assert "SpinAnglesToCartesianSpinTransform" not in sample_names


# ---------------------------------------------------------------------------
# Validation: mutually exclusive groups
# ---------------------------------------------------------------------------


def test_spin_groups_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _make_pipeline_cfg(
            prior_raw={
                "s1_z": {"type": "uniform", "min": -0.99, "max": 0.99},
                "s2_z": {"type": "uniform", "min": -0.99, "max": 0.99},
                "s1_mag": {"type": "uniform", "min": 0.0, "max": 0.99},
                "s1_theta": {"type": "sine"},
                "s1_phi": {"type": "uniform", "min": 0.0, "max": 6.28318},
            }
        )


def test_sky_groups_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _make_pipeline_cfg(
            prior_raw={
                "ra": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "dec": {"type": "cosine"},
                "azimuth": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "zenith": {"type": "sine"},
            }
        )


def test_time_groups_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _make_pipeline_cfg(
            prior_raw={
                "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
                "t_det": {"type": "uniform", "min": -0.1, "max": 0.1},
            }
        )


def test_t_det_geocentric_time_frame_raises():
    with pytest.raises(ValidationError, match="t_det"):
        _make_pipeline_cfg(
            prior_raw={"t_det": {"type": "uniform", "min": -0.1, "max": 0.1}},
            time_frame="geocentric",
        )


def test_detector_sky_geocentric_sky_frame_raises():
    with pytest.raises(ValidationError, match="azimuth"):
        _make_pipeline_cfg(
            prior_raw={
                "azimuth": {"type": "uniform", "min": 0.0, "max": 6.28318},
                "zenith": {"type": "sine"},
            },
            sky_frame="geocentric",
        )


# ---------------------------------------------------------------------------
# adapt_prior_for_ns_time
# ---------------------------------------------------------------------------


def _make_prior_cfg(params: dict):
    """Build a PriorConfig from a dict of {name: spec_dict}."""
    return PriorConfig.model_validate(params)


def test_adapt_ns_time_converts_t_c_to_t_det():
    """t_c in prior + detector time_frame → replaced by t_det with widened GPS bounds."""
    cfg = SamplingConfig(time_frame="detector")
    prior_cfg = _make_prior_cfg(
        {
            "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
            "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
            "d_L": {"type": "power_law", "min": 1.0, "max": 2000.0, "alpha": 2.0},
        }
    )

    result = adapt_prior_for_ns_time(prior_cfg, cfg)

    assert result is not None
    assert "t_c" not in result.root
    assert "t_det" in result.root
    spec = result.root["t_det"]
    assert isinstance(spec, UniformSpec)
    assert spec.min == prior_cfg.root["t_c"].min
    assert spec.max == prior_cfg.root["t_c"].max
    # Insertion order preserved: t_det sits where t_c was
    assert list(result.root.keys()) == ["M_c", "t_det", "d_L"]


def test_adapt_ns_time_geocentric_no_conversion():
    """time_frame='geocentric' → no conversion; t_c sampled directly is already exact."""
    cfg = SamplingConfig(time_frame="geocentric")
    prior_cfg = _make_prior_cfg(
        {
            "t_c": {"type": "uniform", "min": -0.1, "max": 0.1},
        }
    )

    result = adapt_prior_for_ns_time(prior_cfg, cfg)

    assert result is None  # no change needed


def test_adapt_ns_time_t_det_in_prior_no_conversion():
    """User already put t_det in prior → no conversion needed."""
    cfg = SamplingConfig(time_frame="detector")
    lo = TRIGGER_TIME - 0.15
    hi = TRIGGER_TIME + 0.15
    prior_cfg = _make_prior_cfg(
        {
            "t_det": {"type": "uniform", "min": lo, "max": hi},
        }
    )

    result = adapt_prior_for_ns_time(prior_cfg, cfg)

    assert result is None  # t_det already in prior, no change


def test_adapt_ns_time_t_det_geocentric():
    """t_det in prior + geocentric sampling → adapted to widened t_c prior."""
    cfg = SamplingConfig(time_frame="geocentric")
    lo = TRIGGER_TIME - 0.15
    hi = TRIGGER_TIME + 0.15
    prior_cfg = _make_prior_cfg({"t_det": {"type": "uniform", "min": lo, "max": hi}})

    result = adapt_prior_for_ns_time(prior_cfg, cfg)

    assert result is not None
    assert "t_c" in result.root
    assert "t_det" not in result.root
    spec = result.root["t_c"]
    assert isinstance(spec, UniformSpec)
    assert spec.min == lo
    assert spec.max == hi


# ---------------------------------------------------------------------------
# SNR-weighted distance sampling coordinate
# ---------------------------------------------------------------------------

_D_HAT_PRIOR = {
    "iota": {"type": "sine"},
    "d_L": {"type": "power_law", "min": 1.0, "max": 1000.0, "alpha": 2.0},
    "ra": {"type": "uniform", "min": 0.0, "max": 6.283185307179586},
    "dec": {"type": "cosine"},
    "psi": {"type": "uniform", "min": 0.0, "max": 3.141592653589793},
}


@pytest.mark.slow
def test_d_hat_sample_coordinate_is_built_before_cos_iota_and_round_trips():
    sample_t, _ = _infer(
        {"M_c", "q", "iota", "d_L", "ra", "dec", "psi"},
        sky_frame="geocentric",
        inclination_coordinate="cos_iota",
        distance_coordinate="d_hat",
    )
    names = [type(transform).__name__ for transform in sample_t]
    assert "DistanceToSNRWeightedDistanceTransform" in names
    # d_hat conditions on physical iota, so it must be built while iota exists.
    assert names.index("DistanceToSNRWeightedDistanceTransform") < names.index(
        "CosineTransform"
    )

    physical = {
        "M_c": 30.0,
        "q": 0.8,
        "iota": 1.2,
        "d_L": 400.0,
        "ra": 1.375,
        "dec": -1.2108,
        "psi": 0.2,
    }
    forward = dict(physical)
    for transform in sample_t:
        forward = transform.forward(forward)
    assert "d_hat" in forward and "d_L" not in forward
    assert "cos_iota" in forward and "iota" not in forward
    backward = dict(forward)
    for transform in reversed(sample_t):
        backward = transform.backward(backward)
    assert backward["d_L"] == pytest.approx(400.0)
    assert backward["iota"] == pytest.approx(1.2)


def test_d_hat_sample_coordinate_requires_conditioning_priors():
    with pytest.raises(ValidationError, match="distance_coordinate='d_hat'"):
        _make_pipeline_cfg(
            prior_raw={"d_L": _D_HAT_PRIOR["d_L"]},
            distance_coordinate="d_hat",
        )


def test_d_hat_sample_coordinate_accepts_full_extrinsic_prior():
    cfg = _make_pipeline_cfg(
        prior_raw=_D_HAT_PRIOR,
        sky_frame="geocentric",
        inclination_coordinate="cos_iota",
        distance_coordinate="d_hat",
    )
    assert cfg.sampling.distance_coordinate == "d_hat"


# ---------------------------------------------------------------------------
# Doppler-dressed chirp mass
# ---------------------------------------------------------------------------


def test_doppler_dressed_chirp_mass_transform_is_added():
    sample_t, _ = _infer(
        {"M_c", "q", "ra", "dec"},
        sky_frame="geocentric",
        chirp_mass_coordinate="M_hat",
    )
    names = [type(t).__name__ for t in sample_t]
    assert names.count("ChirpMassToDopplerDressedChirpMassTransform") == 1


def test_doppler_dressed_chirp_mass_absent_by_default():
    sample_t, _ = _infer({"M_c", "q", "ra", "dec"}, sky_frame="geocentric")
    names = [type(t).__name__ for t in sample_t]
    assert "ChirpMassToDopplerDressedChirpMassTransform" not in names


def test_doppler_dressed_chirp_mass_precedes_sky_transform():
    """The reversed chain must restore ra/dec before the dressing consumes them,
    so the dressing has to sit before the sky transform in the forward list."""
    sample_t, _ = _infer(
        {"M_c", "q", "ra", "dec"},
        sky_frame="detector",
        chirp_mass_coordinate="M_hat",
    )
    names = [type(t).__name__ for t in sample_t]
    assert names.index("ChirpMassToDopplerDressedChirpMassTransform") < names.index(
        "SkyFrameToDetectorFrameSkyPositionTransform"
    )


def test_doppler_dressed_chirp_mass_follows_distance_transform():
    """d_hat's inverse consumes M_c, so in the reversed chain the dressing must
    run first: it therefore sits after the distance transform going forward."""
    sample_t, _ = _infer(
        {"M_c", "q", "ra", "dec", "psi", "iota", "d_L"},
        sky_frame="geocentric",
        distance_coordinate="d_hat",
        chirp_mass_coordinate="M_hat",
    )
    names = [type(t).__name__ for t in sample_t]
    assert names.index("DistanceToSNRWeightedDistanceTransform") < names.index(
        "ChirpMassToDopplerDressedChirpMassTransform"
    )


def test_full_stack_round_trips_with_dressed_chirp_mass():
    """Physical -> sampling -> physical must be the identity for the whole chain
    (dressing composed with d_hat, cos_iota, detector time and detector sky)."""
    sample_t, _ = _infer(
        {"M_c", "q", "ra", "dec", "psi", "iota", "d_L", "t_c"},
        sky_frame="detector",
        time_frame="detector",
        inclination_coordinate="cos_iota",
        distance_coordinate="d_hat",
        chirp_mass_coordinate="M_hat",
    )
    physical = {
        "M_c": 1.1802650981093186,
        "q": 0.97,
        "ra": 2.96479778676665,
        "dec": 0.17257877754217157,
        "psi": 1.678,
        "iota": 2.016,
        "d_L": 20.0,
        "t_c": 0.035,
    }
    point = dict(physical)
    for transform in sample_t:
        point, _ = transform.transform(point)
    assert "M_hat" in point and "M_c" not in point
    for transform in reversed(sample_t):
        point, _ = transform.inverse(point)
    for name, value in physical.items():
        assert point[name] == pytest.approx(value, rel=1e-10, abs=1e-12), name


def test_dressed_chirp_mass_requires_sky_priors():
    with pytest.raises(ValidationError, match="requires physical"):
        _make_pipeline_cfg(chirp_mass_coordinate="M_hat")


def test_dressed_chirp_mass_rejects_duplicate_prior_coordinate():
    with pytest.raises(ValidationError, match="already declares 'M_hat'"):
        _make_pipeline_cfg(
            prior_raw={
                "ra": {"type": "uniform", "min": 0.0, "max": 6.283185307179586},
                "dec": {"type": "cosine"},
                "M_hat": {"type": "uniform", "min": 1.0, "max": 2.0},
            },
            chirp_mass_coordinate="M_hat",
        )
