from types import SimpleNamespace

import numpy as np
import pytest

from jimgw.cli._config import CLIHeterodynedConfig
from jimgw.cli._likelihood import build_zero_noise_summary
from jimgw.core.single_event.likelihood import HeterodynedTransientLikelihoodFD


def test_compressed_construction_requires_declared_zero_noise_injection():
    heterodyne = CLIHeterodynedConfig(
        n_bins=256,
        interpolation_order=8,
        phasor_moment_order=16,
        zero_noise_quadrature={"atol": 1e-6, "rtol": 1e-9},
    )
    cfg = SimpleNamespace(
        likelihood=SimpleNamespace(heterodyne=heterodyne),
        data=SimpleNamespace(zero_noise=False),
    )
    with pytest.raises(ValueError, match="zero-noise injection"):
        build_zero_noise_summary(cfg, [], None)


def test_summary_builder_is_bound_into_bin_digest():
    digest = HeterodynedTransientLikelihoodFD._bin_edges_sha256
    edges = np.array([2.0, 10.0, 100.0])
    assert digest(edges) != digest(edges, summary_builder_sha256="a" * 64)
    assert digest(edges, summary_builder_sha256="a" * 64) != digest(
        edges, summary_builder_sha256="b" * 64
    )
    assert digest(edges) != digest(edges, reference_projection="carrier")


def test_custom_frequency_grid_is_validated_before_construction():
    with pytest.raises(ValueError, match="frequency_bin_edges"):
        CLIHeterodynedConfig(n_bins=2, frequency_bin_edges=[2.0, 4.0])
    with pytest.raises(ValueError, match="frequency_bin_edges"):
        CLIHeterodynedConfig(n_bins=2, frequency_bin_edges=[2.0, 4.0, 3.0])
    plan = HeterodynedTransientLikelihoodFD._plan_fixed_reference_bin_edges
    waveform = lambda f, p: {"p": np.ones_like(f, dtype=complex)}
    edges, _, _ = plan(
        np.array([2.0, 3.0, 4.0]),
        2,
        waveform,
        {},
        16,
        frequency_bin_edges=[2.0, 2.5, 4.0],
    )
    np.testing.assert_array_equal(edges, [2.0, 2.5, 4.0])
    with pytest.raises(ValueError, match="full frequency band"):
        plan(
            np.array([2.0, 3.0, 4.0]),
            2,
            waveform,
            {},
            16,
            frequency_bin_edges=[2.5, 3.0, 4.0],
        )


@pytest.mark.parametrize("options", [{"atol": float("nan")}, {"max_evaluations": 0}])
def test_invalid_quadrature_options_fail(options):
    with pytest.raises(ValueError):
        CLIHeterodynedConfig(
            interpolation_order=8,
            phasor_moment_order=16,
            zero_noise_quadrature=options,
        )
