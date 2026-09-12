"""Published XG site selections preserve channel names and legacy fixtures."""

import numpy as np
import pytest

from jimgw.core.constants import EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
from jimgw.core.single_event.detector import (
    get_CE,
    get_CE_A,
    get_detector_preset,
    get_ET,
    get_ET_Sardinia,
)


def _geometry(detector):
    return np.asarray(
        [
            detector.latitude,
            detector.longitude,
            detector.elevation,
            detector.xarm_azimuth,
            detector.yarm_azimuth,
            detector.xarm_tilt,
            detector.yarm_tilt,
            detector.arm_length_m,
        ]
    )


def test_ce_a_published_values_and_directed_orthogonal_arms():
    detector = get_CE_A()
    assert detector.name == "CE"
    np.testing.assert_allclose(
        np.rad2deg(
            [
                detector.latitude,
                detector.longitude,
                detector.xarm_azimuth,
                detector.yarm_azimuth,
            ]
        ),
        [46.0, -125.0, 260.0, 350.0],
        rtol=0,
        atol=1e-13,
    )
    assert detector.arm_length_m == 40_000.0
    assert detector.elevation == detector.xarm_tilt == detector.yarm_tilt == 0.0
    x, y = map(np.asarray, detector.arms)
    up = np.asarray(
        [
            np.cos(detector.latitude) * np.cos(detector.longitude),
            np.cos(detector.latitude) * np.sin(detector.longitude),
            np.sin(detector.latitude),
        ]
    )
    np.testing.assert_allclose([x @ x, y @ y, x @ y], [1.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(np.cross(x, y), up, atol=1e-15)
    # Directed arms matter to the finite-arm transfer: x points slightly west
    # of south, while y points slightly south of east.
    east = np.asarray([-np.sin(detector.longitude), np.cos(detector.longitude), 0])
    north = np.cross(up, east)
    assert x @ east < 0 and x @ north < 0
    assert y @ east > 0 and y @ north < 0


def test_ce_a_vertex_uses_zero_height_geodetic_coordinates():
    detector = get_CE_A()
    latitude, longitude = np.deg2rad([46.0, -125.0])
    a, b = EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS
    eccentricity_squared = 1 - (b / a) ** 2
    normal_radius = a / np.sqrt(1 - eccentricity_squared * np.sin(latitude) ** 2)
    expected = np.asarray(
        [
            normal_radius * np.cos(latitude) * np.cos(longitude),
            normal_radius * np.cos(latitude) * np.sin(longitude),
            normal_radius * (1 - eccentricity_squared) * np.sin(latitude),
        ]
    )
    np.testing.assert_allclose(detector.vertex, expected, rtol=0, atol=2e-9)


def test_ce_a_override_retains_channel_and_leaves_other_presets_unchanged():
    defaults = get_detector_preset()
    selected = get_detector_preset({"CE": "CE_A_fiducial_2023"})
    assert selected.keys() == defaults.keys()
    assert selected["CE"].name == "CE"
    np.testing.assert_array_equal(_geometry(selected["CE"]), _geometry(get_CE_A()))
    for name in ("H1", "L1", "V1"):
        np.testing.assert_array_equal(
            _geometry(selected[name]), _geometry(defaults[name])
        )
    for first, second in zip(selected["ET"], defaults["ET"], strict=True):
        np.testing.assert_array_equal(_geometry(first), _geometry(second))


def test_default_ce_geometry_remains_hanford():
    detector = get_CE()
    expected_angles = [
        46 + 27 / 60 + 18.528 / 3600,
        -(119 + 24 / 60 + 27.5657 / 3600),
        125.9994,
        215.994,
    ]
    np.testing.assert_allclose(
        np.rad2deg(
            [
                detector.latitude,
                detector.longitude,
                detector.xarm_azimuth,
                detector.yarm_azimuth,
            ]
        ),
        expected_angles,
        rtol=0,
        atol=1e-13,
    )
    assert detector.elevation == 142.554
    assert detector.xarm_tilt == -6.195e-4
    assert detector.yarm_tilt == 1.25e-5
    np.testing.assert_array_equal(
        _geometry(get_detector_preset()["CE"]), _geometry(detector)
    )
    np.testing.assert_array_equal(
        _geometry(get_detector_preset({})["CE"]), _geometry(detector)
    )


def test_sardinia_reference_corner_and_channel_names():
    detectors = get_ET_Sardinia()
    assert [detector.name for detector in detectors] == ["ET1", "ET2", "ET3"]
    first = detectors[0]
    np.testing.assert_allclose(
        np.rad2deg(
            [first.latitude, first.longitude, first.xarm_azimuth, first.yarm_azimuth]
        ),
        [40 + 31 / 60, 9 + 25 / 60, 90.0, 150.0],
        rtol=0,
        atol=3e-11,
    )
    assert all(detector.elevation == 0.0 for detector in detectors)
    assert all(detector.arm_length_m == 10_000.0 for detector in detectors)
    assert detectors[1].latitude > first.latitude
    assert detectors[2].longitude < first.longitude


def test_sardinia_arms_point_to_their_actual_neighboring_corners():
    detectors = get_ET_Sardinia()
    vertices = [np.asarray(detector.vertex) for detector in detectors]
    for index, detector in enumerate(detectors):
        for arm, neighbor in zip(
            detector.arms, ((index + 1) % 3, (index + 2) % 3), strict=True
        ):
            chord = vertices[neighbor] - vertices[index]
            distance = np.linalg.norm(chord)
            assert abs(distance - 10_000.0) < 0.004
            np.testing.assert_allclose(arm, chord / distance, rtol=0, atol=5e-13)
        # Surface-to-surface straight arms point downward relative to each
        # corner's own ellipsoid tangent plane; zero tilts would miss corners.
        assert -0.0008 < detector.xarm_tilt < -0.0007
        assert -0.0008 < detector.yarm_tilt < -0.0007


def test_sardinia_triangle_is_right_handed_and_static_tensors_cancel():
    detectors = get_ET_Sardinia()
    for detector in detectors:
        x, y = map(np.asarray, detector.arms)
        opening = np.rad2deg(np.arccos(np.clip(x @ y, -1.0, 1.0)))
        assert abs(opening - 60.0) < 0.000021
        lat, lon = detector.latitude, detector.longitude
        up = np.asarray(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        assert np.cross(x, y) @ up > 0.86
    # Each shared physical arm appears once with each sign in the three
    # detector tensors. This is the long-wavelength geometric null relation;
    # it does not assert a null after frequency-dependent propagation delays.
    np.testing.assert_allclose(
        sum(np.asarray(d.tensor) for d in detectors), 0.0, atol=1e-12
    )


def test_both_site_selections_are_explicit_and_preserve_legacy_et():
    overrides = {"CE": "CE_A_fiducial_2023", "ET": "ET_Sardinia_fiducial_2023"}
    selected = get_detector_preset(overrides)
    np.testing.assert_array_equal(_geometry(selected["CE"]), _geometry(get_CE_A()))
    for actual, expected in zip(selected["ET"], get_ET_Sardinia(), strict=True):
        np.testing.assert_array_equal(_geometry(actual), _geometry(expected))
    for actual, expected in zip(get_detector_preset()["ET"], get_ET(), strict=True):
        np.testing.assert_array_equal(_geometry(actual), _geometry(expected))
    assert overrides == {"CE": "CE_A_fiducial_2023", "ET": "ET_Sardinia_fiducial_2023"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"CE": "unknown"},
        {"H1": "CE_A_fiducial_2023"},
        {"missing": "CE_A_fiducial_2023"},
        {"CE": "ET_Sardinia_fiducial_2023"},
        {"ET": "CE_A_fiducial_2023"},
        {"ET": "unknown"},
    ],
)
def test_unknown_or_mismatched_site_override_is_rejected(overrides):
    with pytest.raises(ValueError, match="Unsupported detector site override"):
        get_detector_preset(overrides)
