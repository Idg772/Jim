"""Dominant-mode waveform adapter for time-dependent detector responses."""

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, cast

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float
from ripplegw.interfaces import (
    DistanceScaledWaveform,
    FrequencyDomainWaveform,
    Waveform,
)

from jimgw.core.single_event.time_dependent_response import (
    time_to_coalescence_2pn,
)
from jimgw.core.single_event.transform_utils import Mc_eta_to_m1_m2
from jimgw.typing import FloatLike

TIME_TO_COALESCENCE_KEY = "__tau__"
"""Reserved waveform-output leaf containing the non-negative emission clock."""

_SOURCE_CACHE_KEY = "__source_cache__"
_REQUIRED_TIMING_PARAMETERS = frozenset(("M_c", "eta", "s1_z", "s2_z"))
_KNOWN_HIGHER_MODE_CLASSES = frozenset(
    (
        "IMRPhenomHM",
        "IMRPhenomXHM",
        "IMRPhenomXPHM",
    )
)
_KNOWN_PRECESSING_CLASSES = frozenset(
    (
        "IMRPhenomPv2",
        "IMRPhenomXP",
        "IMRPhenomXPHM",
        "RippleIMRPhenomPv2NRTidalv2",
    )
)
_KNOWN_DOMINANT_MODE_CLASSES = frozenset(
    (
        "TaylorF2",
        "IMRPhenomD",
        "IMRPhenomD_NRTidalv2",
        "IMRPhenomXAS",
        "IMRPhenomXAS_NRTidalv3",
    )
)


class DominantModeTimeCachedWaveform(
    FrequencyDomainWaveform,
    DistanceScaledWaveform,
):
    """Add a reusable dominant-mode 2PN clock to an FD waveform.

    The wrapped waveform must produce only the already-combined ``p`` and
    ``c`` polarizations of a non-precessing, dominant-mode source.  The
    adapter adds :data:`TIME_TO_COALESCENCE_KEY` to direct and reconstructed
    outputs.  A time-dependent detector consumes that leaf before projecting
    the two physical polarizations.

    Existing custom waveform caches are delegated to without changing their
    payload.  Otherwise the wrapped waveform is cached at unit distance.  In
    either case, reconstruction copies the cached clock unchanged: distance
    and any delegated inclination reconstruction apply only to ``p`` and
    ``c``.
    """

    source: Waveform
    mode: int
    _f_ref: FloatLike
    time_dependent_response: ClassVar[bool] = True
    waveform_metadata: ClassVar[dict[str, Any]] = {
        "domain": "FD",
        "is_precessing": False,
        "source_type": "cbc",
    }

    def __init__(self, source: Waveform, *, mode: int = 2) -> None:
        if abs(mode) != 2:
            raise ValueError("the dominant-mode adapter requires abs(mode) == 2")
        if not hasattr(source, "f_ref"):
            raise ValueError("the wrapped frequency-domain waveform must define f_ref")

        parameter_names = set(source.parameter_names)
        required_parameters = _REQUIRED_TIMING_PARAMETERS | {"d_L", "iota"}
        missing_parameters = required_parameters - parameter_names
        if missing_parameters:
            raise ValueError(
                "the dominant-mode time cache requires waveform parameters "
                f"{sorted(required_parameters)}; missing {sorted(missing_parameters)}"
            )

        metadata = getattr(source, "waveform_metadata", {})
        class_name = type(source).__name__
        if metadata.get("is_precessing", False) or class_name in (
            _KNOWN_PRECESSING_CLASSES | _KNOWN_HIGHER_MODE_CLASSES
        ):
            raise ValueError(
                f"{class_name} does not expose a dominant-mode response-time "
                "decomposition"
            )
        if not (
            metadata.get("dominant_mode_only", False)
            or class_name in _KNOWN_DOMINANT_MODE_CLASSES
        ):
            raise ValueError(
                f"{class_name} does not explicitly declare dominant-mode-only "
                "output; set waveform_metadata['dominant_mode_only'] = True only "
                "for a verified p/c dominant-mode backend"
            )

        builder = getattr(source, "build_waveform_cache", None)
        reconstructor = getattr(source, "waveform_from_cache", None)
        if callable(builder) != callable(reconstructor):
            raise TypeError(
                "the wrapped waveform must implement both build_waveform_cache "
                "and waveform_from_cache"
            )
        declared_cacheable = set(getattr(source, "cacheable_parameter_names", ()))
        if declared_cacheable and not callable(builder):
            raise TypeError(
                "a wrapped waveform that declares cacheable_parameter_names must "
                "implement its custom cache methods"
            )
        unknown_cacheable = declared_cacheable - parameter_names
        if unknown_cacheable:
            raise ValueError(
                "wrapped cacheable parameters are not waveform parameters: "
                f"{sorted(unknown_cacheable)}"
            )
        if callable(builder) and "d_L" not in declared_cacheable:
            raise TypeError(
                "a wrapped custom waveform cache must explicitly declare d_L cacheable"
            )
        if not callable(builder) and not isinstance(source, DistanceScaledWaveform):
            raise TypeError(
                "the wrapped waveform must implement DistanceScaledWaveform or "
                "a custom cache that reconstructs d_L"
            )

        self.source = source
        self.mode = mode
        self._f_ref = cast(Any, source).f_ref
        self._has_custom_source_cache = callable(builder)
        fallback_cacheable = {"d_L", "iota"} if not callable(builder) else {"d_L"}
        self._cacheable_parameter_names = frozenset(
            declared_cacheable | fallback_cacheable
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the wrapped waveform's parameter contract."""

        return self.source.parameter_names

    @property
    def f_ref(self) -> FloatLike:
        """Return the wrapped waveform's configured reference frequency."""

        return self._f_ref

    @property
    def cacheable_parameter_names(self) -> frozenset[str]:
        """Return parameters reconstructed without rebuilding source state."""

        return self._cacheable_parameter_names

    @property
    def emission_time_parameter_names(self) -> frozenset[str]:
        """Return intrinsic inputs that invalidate the cached emission clock."""

        return _REQUIRED_TIMING_PARAMETERS

    def _time_to_coalescence(
        self,
        frequency: Float[Array, " n_freq"],
        params: Mapping[str, FloatLike],
    ) -> Float[Array, " n_freq"]:
        mass_1, mass_2 = Mc_eta_to_m1_m2(
            jnp.asarray(params["M_c"]), jnp.asarray(params["eta"])
        )
        return time_to_coalescence_2pn(
            frequency,
            mass_1,
            mass_2,
            params["s1_z"],
            params["s2_z"],
            mode=self.mode,
        )

    @staticmethod
    def _validate_polarizations(output: Mapping[str, Any]) -> None:
        keys = set(output)
        if keys != {"p", "c"}:
            raise ValueError(
                "the dominant-mode adapter requires exactly the p/c waveform "
                f"polarizations; received {sorted(keys)}"
            )

    def __call__(
        self,
        frequency: Float[Array, " n_freq"],
        params: Mapping[str, FloatLike],
    ) -> dict[str, Complex[Array, " n_freq"] | Float[Array, " n_freq"]]:
        """Evaluate physical polarizations and their positive emission clock."""

        polarizations = self.source(frequency, params)
        self._validate_polarizations(polarizations)
        return {
            "p": polarizations["p"],
            "c": polarizations["c"],
            TIME_TO_COALESCENCE_KEY: self._time_to_coalescence(frequency, params),
        }

    def build_waveform_cache(
        self,
        frequency: Float[Array, " n_freq"],
        params: Mapping[str, FloatLike],
    ) -> dict[str, Any]:
        """Build source state and the intrinsic clock in one atomic payload."""

        if self._has_custom_source_cache:
            builder = cast(
                Callable[[Float[Array, " n_freq"], Mapping[str, FloatLike]], Any],
                cast(Any, self.source).build_waveform_cache,
            )
            source_cache = builder(frequency, params)
        else:
            distance_scaled_source = cast(DistanceScaledWaveform, self.source)
            face_on_params = dict(params)
            face_on_params["iota"] = 0.0
            source_cache = distance_scaled_source.at_unit_distance(
                frequency,
                face_on_params,
            )
            self._validate_polarizations(source_cache)
        return {
            _SOURCE_CACHE_KEY: source_cache,
            TIME_TO_COALESCENCE_KEY: self._time_to_coalescence(frequency, params),
        }

    def waveform_from_cache(
        self,
        frequency: Float[Array, " n_freq"],
        params: Mapping[str, FloatLike],
        cache: Mapping[str, Any],
    ) -> dict[str, Complex[Array, " n_freq"] | Float[Array, " n_freq"]]:
        """Reconstruct polarizations while retaining the cached clock exactly."""

        missing_keys = {_SOURCE_CACHE_KEY, TIME_TO_COALESCENCE_KEY} - set(cache)
        if missing_keys:
            raise ValueError(
                f"dominant-mode source cache is missing {sorted(missing_keys)}"
            )

        source_cache = cache[_SOURCE_CACHE_KEY]
        if self._has_custom_source_cache:
            reconstructor = cast(
                Callable[
                    [Float[Array, " n_freq"], Mapping[str, FloatLike], Any],
                    Mapping[str, Any],
                ],
                cast(Any, self.source).waveform_from_cache,
            )
            polarizations = reconstructor(frequency, params, source_cache)
        else:
            distance_scale = 1.0 / params["d_L"]
            cos_iota = jnp.cos(params["iota"])
            polarizations = {
                "p": source_cache["p"] * (0.5 * (1.0 + cos_iota**2) * distance_scale),
                "c": source_cache["c"] * (cos_iota * distance_scale),
            }
        self._validate_polarizations(polarizations)
        return {
            "p": polarizations["p"],
            "c": polarizations["c"],
            TIME_TO_COALESCENCE_KEY: cache[TIME_TO_COALESCENCE_KEY],
        }


__all__ = ["TIME_TO_COALESCENCE_KEY", "DominantModeTimeCachedWaveform"]
