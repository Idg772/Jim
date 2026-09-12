"""Exact, bounded-storage union construction for nested detector bands."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event import likelihood as module

jax.config.update("jax_enable_x64", True)


@dataclass
class _DetectorGrid:
    name: str
    frequencies: jax.Array
    duration: float

    def set_frequency_bounds(self, low, high):
        self.sliced_frequencies = self.frequencies[
            (self.frequencies >= low) & (self.frequencies <= high)
        ]


class _NumpyUnionAudit:
    """Instrument this module's NumPy calls without changing global NumPy."""

    def __init__(self, *, allow_union):
        self.allow_union = allow_union
        self.calls = []

    def __getattr__(self, name):
        return getattr(np, name)

    def concatenate(self, values):
        assert self.allow_union, "Nested grids must not allocate a host concatenation"
        self.calls.append("concatenate")
        return np.concatenate(values)

    def unique(self, values):
        assert self.allow_union, "Nested grids must not sort a host union"
        self.calls.append("unique")
        return np.unique(values)


@pytest.mark.parametrize("duration", [8.0, 10.0, 80_000.0])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_network_reuses_exact_longest_native_grid(
    monkeypatch, duration, reversed_order
):
    # The longest test has only 480001 samples, not a day-long 4096 Hz allocation.
    native = jnp.arange(int(6 * duration) + 1, dtype=jnp.float64) / duration
    detectors = [
        _DetectorGrid(name, native, duration) for name in ("CE", "ET1", "ET2", "ET3")
    ]
    if reversed_order:
        detectors.reverse()
    audit = _NumpyUnionAudit(allow_union=False)
    monkeypatch.setattr(module, "np", audit)
    low = {"CE": 5.0, "ET1": 2.0, "ET2": 2.0, "ET3": 2.0}
    high = {"CE": 6.0, "ET1": 6.0, "ET2": 5.0, "ET3": 6.0}

    result, identical, df = module._set_and_merge_heterodyne_frequency_grids(
        detectors, low, high
    )

    longest = max(detectors, key=lambda detector: detector.sliced_frequencies.size)
    assert result is longest.sliced_frequencies
    assert identical is False
    np.testing.assert_array_equal(result, native[(native >= 2.0) & (native <= 6.0)])
    assert float(df) == float(
        detectors[0].sliced_frequencies[1] - detectors[0].sliced_frequencies[0]
    )
    assert audit.calls == []


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([2.0, 2.5, 3.0], [5.0, 5.5, 6.0]),  # disjoint bands
        ([2.0, 2.5, 3.0, 3.5], [3.0, 3.5, 4.0]),  # non-nested overlap
        ([2.0, 2.5, 3.0, 3.5], [2.25, 2.75, 3.25]),  # half-bin offset
        ([2.0, 2.5, 3.0, 3.5], [2.0, 2.5, 3.5]),  # internal gap
        ([2.0, 2.5, 3.01, 3.5], [2.5, 3.0]),  # irregular longest
        ([2.0, 2.5, 3.0, 3.5], [2.0, 2.500001, 3.000002]),  # tolerated df difference
        ([2.0, 2.5, 3.000000000000001, 3.5], [2.0, 2.5]),  # tiny interior irregularity
    ],
)
def test_non_nested_or_irregular_grids_preserve_general_union(
    monkeypatch, first, second
):
    detectors = [
        _DetectorGrid(name, jnp.asarray(values), 2.0)
        for name, values in (("CE", first), ("ET1", second))
    ]
    audit = _NumpyUnionAudit(allow_union=True)
    monkeypatch.setattr(module, "np", audit)

    result, identical, df = module._set_and_merge_heterodyne_frequency_grids(
        detectors, 0.0, 10.0
    )

    np.testing.assert_array_equal(result, np.unique(np.concatenate([first, second])))
    assert identical is False
    assert float(df) == first[1] - first[0]
    assert audit.calls == ["concatenate", "unique"]


def test_identical_grid_branch_is_preserved(monkeypatch):
    native = jnp.arange(10, dtype=jnp.float64) / 2
    detectors = [_DetectorGrid(name, native, 2.0) for name in ("CE", "ET1")]
    audit = _NumpyUnionAudit(allow_union=False)
    monkeypatch.setattr(module, "np", audit)

    result, identical, df = module._set_and_merge_heterodyne_frequency_grids(
        detectors, 2.0, 4.0
    )

    assert result is detectors[0].sliced_frequencies
    assert identical is True
    assert float(df) == 0.5
    assert audit.calls == []


def test_mismatched_native_spacing_remains_an_error():
    detectors = [
        _DetectorGrid("CE", jnp.arange(10) / 2.0, 2.0),
        _DetectorGrid("ET1", jnp.arange(10) / 4.0, 4.0),
    ]
    with pytest.raises(ValueError, match="same frequency spacing"):
        module._set_and_merge_heterodyne_frequency_grids(detectors, 0.0, 10.0)


@pytest.mark.parametrize(
    "detectors", [[], [_DetectorGrid("CE", jnp.asarray([2.0]), 2.0)]]
)
def test_empty_and_single_bin_guards_are_preserved(detectors):
    with pytest.raises(ValueError, match="at least"):
        module._set_and_merge_heterodyne_frequency_grids(detectors, 0.0, 10.0)
