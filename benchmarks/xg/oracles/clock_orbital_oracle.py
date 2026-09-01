"""Independent clock and orbital-error primitives for XG qualification.

This module deliberately does not import Jim.  Callers inject a complex
frequency-domain carrier, ephemeris position/velocity samples, and waveform
tangents.  The routines here then provide the numerical operations needed by
the clock and orbital qualification generators without sharing production
response code.

The inner-product weights are expected to include the one-sided quadrature
factor (for example ``4 * df / PSD``).  Positions and velocities are unit
neutral as long as they use consistent units; LALPulsar's light-seconds and
``v/c`` are therefore accepted directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

TWO_PI = 2.0 * np.pi


ComplexCarrier = Callable[[NDArray[np.float64]], ArrayLike]


@dataclass(frozen=True)
class StationaryTimeEstimate:
    """Converged phase-derivative estimates for one frequency array."""

    frequency_hz: NDArray[np.float64]
    stationary_time_s: NDArray[np.float64]
    phase_derivative_rad_per_hz: NDArray[np.float64]
    estimated_abs_error_s: NDArray[np.float64]
    final_step_hz: NDArray[np.float64]
    refinement_count: NDArray[np.int64]


@dataclass(frozen=True)
class ResponseImpact:
    """Zero-noise response loss after normalization to a target SNR."""

    delta_log_l: float
    target_snr: float
    unscaled_reference_snr: float
    amplitude_scale: float


@dataclass(frozen=True)
class FisherProfile:
    """Linear tangent-space profile of an omitted waveform contribution."""

    parameter_names: tuple[str, ...]
    unprofiled_delta_log_l: float
    profiled_delta_log_l: float
    parameter_bias: dict[str, float]
    parameter_sigma: dict[str, float]
    parameter_bias_sigma: dict[str, float]
    fisher_matrix: NDArray[np.float64]
    covariance_matrix: NDArray[np.float64]
    projected_residual: NDArray[np.complex128]
    target_snr: float
    amplitude_scale: float
    rank: int
    condition_number: float


def _as_finite_frequency_array(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim == 0:
        frequency = frequency.reshape(1)
    if frequency.ndim != 1 or frequency.size == 0:
        raise ValueError("frequency_hz must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency_hz must contain finite positive values")
    return frequency


def _evaluate_carrier(
    carrier: ComplexCarrier,
    frequencies: NDArray[np.float64],
) -> NDArray[np.complex128]:
    values = np.asarray(carrier(frequencies), dtype=np.complex128)
    if values.shape != frequencies.shape:
        raise ValueError("carrier output must have the same shape as its input")
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError("carrier returned non-finite values")
    if np.any(np.abs(values) == 0.0):
        raise ValueError("carrier must be nonzero at derivative stencil points")
    return values


def adaptive_stationary_time(
    carrier: ComplexCarrier,
    frequency_hz: ArrayLike,
    *,
    initial_step_hz: ArrayLike | None = None,
    absolute_tolerance_s: float = 1.0e-8,
    relative_tolerance: float = 1.0e-10,
    minimum_refinements: int = 4,
    maximum_refinements: int = 30,
    fourier_phase_sign: float = -1.0,
) -> StationaryTimeEstimate:
    """Recover stationary time from an adaptively differentiated FD carrier.

    For the Fourier convention ``h(f) = A(f) exp[-2 pi i f t + ...]``, the
    stationary time is ``-d arg(h) / (2 pi df)``; this is the default
    ``fourier_phase_sign=-1``.  A central derivative is evaluated at steps
    ``h`` and ``h/2`` and fourth-order Richardson extrapolation is applied.
    Refinement continues until the fourth-order Richardson truncation estimate
    derived from consecutive extrapolates meets the requested tolerance.

    Differentiating the complex carrier and taking ``Im(h' / h)`` avoids a
    global phase unwrap and makes every evaluation local to the requested
    frequency.  A deliberately coarse initial step is safe: it is repeatedly
    halved rather than being accepted from a single wrapped phase difference.
    """

    frequency = _as_finite_frequency_array(frequency_hz)
    numeric_controls = np.asarray(
        (absolute_tolerance_s, relative_tolerance, fourier_phase_sign),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric_controls)):
        raise ValueError("derivative controls must be finite")
    if absolute_tolerance_s <= 0.0 or relative_tolerance < 0.0:
        raise ValueError("derivative tolerances must be nonnegative with atol > 0")
    if abs(fourier_phase_sign) != 1.0:
        raise ValueError("fourier_phase_sign must be either -1 or +1")
    if (
        isinstance(minimum_refinements, bool)
        or isinstance(maximum_refinements, bool)
        or not isinstance(minimum_refinements, (int, np.integer))
        or not isinstance(maximum_refinements, (int, np.integer))
        or minimum_refinements < 1
        or maximum_refinements < minimum_refinements
    ):
        raise ValueError("invalid derivative refinement bounds")

    if initial_step_hz is None:
        step = np.minimum(0.01 * frequency, 0.1)
    else:
        step = np.broadcast_to(
            np.asarray(initial_step_hz, dtype=np.float64), frequency.shape
        ).copy()
    if (
        not np.all(np.isfinite(step))
        or np.any(step <= 0.0)
        or np.any(step >= frequency)
    ):
        raise ValueError(
            "initial_step_hz must be finite, positive, and below frequency"
        )

    center = _evaluate_carrier(carrier, frequency)
    stationary_time = np.empty_like(frequency)
    phase_derivative = np.empty_like(frequency)
    estimated_error = np.empty_like(frequency)
    final_step = np.empty_like(frequency)
    refinement_count = np.empty(frequency.shape, dtype=np.int64)

    for index, (frequency_value, center_value, initial_step) in enumerate(
        zip(frequency, center, step, strict=True)
    ):
        local_step = float(initial_step)
        previous_time: float | None = None
        converged = False

        for refinement in range(1, maximum_refinements + 1):
            stencil = np.asarray(
                (
                    frequency_value - local_step,
                    frequency_value + local_step,
                    frequency_value - 0.5 * local_step,
                    frequency_value + 0.5 * local_step,
                ),
                dtype=np.float64,
            )
            if stencil[0] <= 0.0:
                local_step *= 0.5
                continue
            values = _evaluate_carrier(carrier, stencil)
            derivative_h = (values[1] - values[0]) / (2.0 * local_step)
            derivative_half = (values[3] - values[2]) / local_step
            derivative_richardson = (4.0 * derivative_half - derivative_h) / 3.0

            phase_value = float(np.imag(derivative_richardson / center_value))
            time_value = fourier_phase_sign * phase_value / TWO_PI
            successive_change = (
                np.inf if previous_time is None else abs(time_value - previous_time)
            )
            # The leading error in a Richardson-extrapolated central
            # derivative scales as h**4.  Consecutive extrapolates therefore
            # differ by 15 times the finer estimate's truncation error.
            error_value = successive_change / 15.0
            tolerance = absolute_tolerance_s + relative_tolerance * abs(time_value)
            if refinement >= minimum_refinements and error_value <= tolerance:
                stationary_time[index] = time_value
                phase_derivative[index] = phase_value
                estimated_error[index] = error_value
                final_step[index] = 0.5 * local_step
                refinement_count[index] = refinement
                converged = True
                break

            previous_time = time_value
            local_step *= 0.5

        if not converged:
            raise RuntimeError(
                "phase derivative did not converge at "
                f"{frequency_value:.17g} Hz after {maximum_refinements} refinements"
            )

    return StationaryTimeEstimate(
        frequency_hz=frequency.copy(),
        stationary_time_s=stationary_time,
        phase_derivative_rad_per_hz=phase_derivative,
        estimated_abs_error_s=estimated_error,
        final_step_hz=final_step,
        refinement_count=refinement_count,
    )


def time_to_coalescence_from_stationary_time(
    stationary_time_s: ArrayLike,
    *,
    coalescence_time_s: float = 0.0,
    frequency_hz: ArrayLike | None = None,
    cutoff_frequency_hz: float | None = None,
) -> NDArray[np.float64]:
    """Convert stationary epochs to a nonnegative pre-coalescence clock.

    Values at or above ``cutoff_frequency_hz`` are fixed at coalescence.  The
    cutoff arguments must either both be supplied or both be omitted.
    """

    stationary_time = np.asarray(stationary_time_s, dtype=np.float64)
    if not np.all(np.isfinite(stationary_time)) or not np.isfinite(coalescence_time_s):
        raise ValueError("clock inputs must be finite")
    if (frequency_hz is None) != (cutoff_frequency_hz is None):
        raise ValueError(
            "frequency_hz and cutoff_frequency_hz must be supplied together"
        )

    clock = np.maximum(float(coalescence_time_s) - stationary_time, 0.0)
    if cutoff_frequency_hz is not None:
        cutoff = float(cutoff_frequency_hz)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("cutoff_frequency_hz must be finite and positive")
        frequency = np.broadcast_to(
            np.asarray(frequency_hz, dtype=np.float64), stationary_time.shape
        )
        if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
            raise ValueError("frequency_hz must contain finite positive values")
        clock = np.where(frequency < cutoff, clock, 0.0)
    return np.asarray(clock, dtype=np.float64)


def noise_weighted_inner_product(
    left: ArrayLike,
    right: ArrayLike,
    weights: ArrayLike,
) -> float:
    """Return ``Re sum(conj(left) * right * weights)`` on a flattened network."""

    left_array = np.asarray(left, dtype=np.complex128)
    right_array = np.asarray(right, dtype=np.complex128)
    if left_array.shape != right_array.shape or left_array.size == 0:
        raise ValueError("inner-product operands must have one equal nonempty shape")
    weight_array = np.broadcast_to(
        np.asarray(weights, dtype=np.float64), left_array.shape
    )
    if (
        not np.all(np.isfinite(left_array.real))
        or not np.all(np.isfinite(left_array.imag))
        or not np.all(np.isfinite(right_array.real))
        or not np.all(np.isfinite(right_array.imag))
        or not np.all(np.isfinite(weight_array))
        or np.any(weight_array < 0.0)
    ):
        raise ValueError("inner-product arrays must be finite and weights nonnegative")
    return float(
        np.real(
            np.vdot(
                left_array.reshape(-1),
                (right_array * weight_array).reshape(-1),
            )
        )
    )


def response_impact_delta_log_l(
    reference_response: ArrayLike,
    candidate_response: ArrayLike,
    weights: ArrayLike,
    *,
    target_snr: float,
) -> ResponseImpact:
    """Return the zero-noise likelihood loss caused by a response difference.

    Both responses are scaled by the same factor so the reference has exactly
    ``target_snr``.  The result is the stable residual expression
    ``0.5 * ||h_candidate - h_reference||^2`` rather than a subtraction of two
    order-SNR-squared likelihood values.
    """

    target = float(target_snr)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target_snr must be finite and positive")
    reference = np.asarray(reference_response, dtype=np.complex128)
    candidate = np.asarray(candidate_response, dtype=np.complex128)
    reference_norm = noise_weighted_inner_product(reference, reference, weights)
    if reference_norm <= 0.0:
        raise ValueError("reference response must have positive weighted norm")
    unscaled_snr = float(np.sqrt(reference_norm))
    scale = target / unscaled_snr
    difference = scale * (candidate - reference)
    delta_log_l = 0.5 * noise_weighted_inner_product(
        difference,
        difference,
        weights,
    )
    return ResponseImpact(
        delta_log_l=max(0.0, float(delta_log_l)),
        target_snr=target,
        unscaled_reference_snr=unscaled_snr,
        amplitude_scale=scale,
    )


def remove_affine_orbital_motion(
    time_s: ArrayLike,
    position: ArrayLike,
    velocity: ArrayLike,
    *,
    reference_index: int = 0,
) -> NDArray[np.float64]:
    """Remove constant position and line-of-sight-velocity degeneracies.

    ``position`` must have shape ``(n, 3)``.  ``velocity`` may contain one
    three-vector per time sample or a single three-vector.  Only the velocity
    at ``reference_index`` defines the affine term, matching a local expansion
    of a full ephemeris about the declared reference epoch.
    """

    time = np.asarray(time_s, dtype=np.float64)
    positions = np.asarray(position, dtype=np.float64)
    velocities = np.asarray(velocity, dtype=np.float64)
    if time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time)):
        raise ValueError("time_s must be a nonempty finite one-dimensional array")
    if positions.shape != (time.size, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("position must be a finite array with shape (n_time, 3)")
    if (
        isinstance(reference_index, bool)
        or not isinstance(reference_index, (int, np.integer))
        or not -time.size <= reference_index < time.size
    ):
        raise ValueError("reference_index is outside the time grid")
    reference_index = int(reference_index) % time.size
    if velocities.shape == (3,) and np.all(np.isfinite(velocities)):
        reference_velocity = velocities
    elif velocities.shape == positions.shape and np.all(np.isfinite(velocities)):
        reference_velocity = velocities[reference_index]
    else:
        raise ValueError("velocity must have shape (3,) or (n_time, 3)")

    elapsed = time - time[reference_index]
    affine = positions[reference_index] + elapsed[:, None] * reference_velocity
    return np.asarray(positions - affine, dtype=np.float64)


def project_orbital_delay(
    position_residual: ArrayLike,
    source_direction: ArrayLike,
) -> NDArray[np.float64]:
    """Project a position residual onto one unit source direction."""

    residual = np.asarray(position_residual, dtype=np.float64)
    direction = np.asarray(source_direction, dtype=np.float64)
    if residual.ndim != 2 or residual.shape[1] != 3:
        raise ValueError("position_residual must have shape (n_time, 3)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("position_residual must be finite")
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("source_direction must be a finite three-vector")
    if not np.isclose(np.linalg.norm(direction), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("source_direction must be a unit vector")
    return np.asarray(residual @ direction, dtype=np.float64)


def profile_waveform_residual(
    reference_waveform: ArrayLike,
    waveform_difference: ArrayLike,
    tangents: ArrayLike,
    weights: ArrayLike,
    *,
    parameter_names: Sequence[str],
    target_snr: float,
    singular_value_rcond: float = 1.0e-12,
    require_full_rank: bool = True,
) -> FisherProfile:
    """Project an omitted waveform contribution off all parameter tangents.

    The linearized best fit solves

    ``min_delta 0.5 * ||waveform_difference - J delta||^2``.

    The reference, difference, and tangents are first scaled together so the
    reference waveform has the requested network SNR.  The returned parameter
    bias is therefore invariant under normalization, while posterior sigmas,
    bias-in-sigma, and likelihood loss have the target-SNR scaling required by
    a qualification receipt.
    """

    reference = np.asarray(reference_waveform, dtype=np.complex128)
    difference = np.asarray(waveform_difference, dtype=np.complex128)
    tangent_array = np.asarray(tangents, dtype=np.complex128)
    names = tuple(parameter_names)
    if reference.shape != difference.shape or reference.size == 0:
        raise ValueError("reference and waveform difference must have equal shape")
    if (
        tangent_array.ndim != reference.ndim + 1
        or tangent_array.shape[1:] != reference.shape
    ):
        raise ValueError("tangents must have shape (n_parameter, *waveform_shape)")
    if (
        len(names) != tangent_array.shape[0]
        or not names
        or any(not name for name in names)
    ):
        raise ValueError("parameter_names must name every tangent")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique")
    target = float(target_snr)
    rcond = float(singular_value_rcond)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target_snr must be finite and positive")
    if not np.isfinite(rcond) or rcond <= 0.0 or rcond >= 1.0:
        raise ValueError("singular_value_rcond must lie strictly between zero and one")
    if not np.all(np.isfinite(tangent_array.real)) or not np.all(
        np.isfinite(tangent_array.imag)
    ):
        raise ValueError("tangents must be finite")

    reference_norm = noise_weighted_inner_product(reference, reference, weights)
    if reference_norm <= 0.0:
        raise ValueError("reference waveform must have positive weighted norm")
    scale = target / np.sqrt(reference_norm)
    scaled_difference = np.asarray(scale * difference, dtype=np.complex128)
    scaled_tangents = np.asarray(scale * tangent_array, dtype=np.complex128)
    weight_array = np.broadcast_to(
        np.asarray(weights, dtype=np.float64), reference.shape
    ).reshape(-1)
    tangent_matrix = scaled_tangents.reshape((len(names), -1))
    difference_vector = scaled_difference.reshape(-1)

    fisher = np.real(
        np.einsum(
            "pi,i,qi->pq",
            tangent_matrix.conj(),
            weight_array,
            tangent_matrix,
            optimize=True,
        )
    )
    fisher = np.asarray(0.5 * (fisher + fisher.T), dtype=np.float64)
    forcing = np.real(
        np.einsum(
            "pi,i,i->p",
            tangent_matrix.conj(),
            weight_array,
            difference_vector,
            optimize=True,
        )
    )

    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    largest = float(eigenvalues[-1])
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("Fisher matrix has no positive direction")
    threshold = rcond * largest
    if eigenvalues[0] < -threshold:
        raise ValueError("Fisher matrix is not positive semidefinite")
    retained = eigenvalues > threshold
    rank = int(np.count_nonzero(retained))
    if require_full_rank and rank != len(names):
        raise ValueError(f"Fisher matrix is rank deficient ({rank} < {len(names)})")

    inverse_eigenvalues = np.zeros_like(eigenvalues)
    inverse_eigenvalues[retained] = 1.0 / eigenvalues[retained]
    covariance = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
    covariance = np.asarray(0.5 * (covariance + covariance.T), dtype=np.float64)
    bias = np.asarray(covariance @ forcing, dtype=np.float64)
    projected = np.asarray(
        difference_vector - bias @ tangent_matrix,
        dtype=np.complex128,
    )
    unprofiled_delta = 0.5 * noise_weighted_inner_product(
        scaled_difference,
        scaled_difference,
        weights,
    )
    profiled_delta = 0.5 * noise_weighted_inner_product(
        projected,
        projected,
        weight_array,
    )
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if require_full_rank and np.any(sigma <= 0.0):
        raise ValueError("full-rank Fisher matrix produced a nonpositive sigma")
    bias_sigma = np.divide(
        np.abs(bias),
        sigma,
        out=np.full_like(bias, np.inf),
        where=sigma > 0.0,
    )
    retained_values = eigenvalues[retained]
    condition_number = float(largest / np.min(retained_values))

    return FisherProfile(
        parameter_names=names,
        unprofiled_delta_log_l=max(0.0, float(unprofiled_delta)),
        profiled_delta_log_l=max(0.0, float(profiled_delta)),
        parameter_bias={
            name: float(value) for name, value in zip(names, bias, strict=True)
        },
        parameter_sigma={
            name: float(value) for name, value in zip(names, sigma, strict=True)
        },
        parameter_bias_sigma={
            name: float(value) for name, value in zip(names, bias_sigma, strict=True)
        },
        fisher_matrix=fisher,
        covariance_matrix=covariance,
        projected_residual=projected.reshape(reference.shape),
        target_snr=target,
        amplitude_scale=float(scale),
        rank=rank,
        condition_number=condition_number,
    )


def fisher_projected_bias_sigma(
    fisher_matrix: ArrayLike,
    parameter_shift: ArrayLike,
    *,
    singular_value_rcond: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Project a parameter shift onto identifiable Fisher eigenmodes.

    A rank-deficient single-detector Fisher matrix has no data-defined sigma in
    its null space. Dividing coordinate shifts by the diagonal of a
    pseudoinverse therefore assigns arbitrary optimizer drift an artificial,
    and sometimes infinite, significance. This routine reports the invariant
    displacement ``sqrt(lambda) * v.T @ delta`` only for retained eigenmodes;
    null-space motion is intentionally excluded from a data-constrained bias.
    """

    fisher = np.asarray(fisher_matrix, dtype=np.float64)
    shift = np.asarray(parameter_shift, dtype=np.float64)
    rcond = float(singular_value_rcond)
    if (
        fisher.ndim != 2
        or fisher.shape[0] != fisher.shape[1]
        or fisher.shape[0] == 0
        or shift.shape != (fisher.shape[0],)
    ):
        raise ValueError("Fisher matrix and parameter shift have incompatible shapes")
    if (
        not np.all(np.isfinite(fisher))
        or not np.all(np.isfinite(shift))
        or not np.isfinite(rcond)
        or rcond <= 0.0
        or rcond >= 1.0
    ):
        raise ValueError("Fisher projection inputs must be finite and rcond valid")
    fisher = np.asarray(0.5 * (fisher + fisher.T), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    largest = float(eigenvalues[-1])
    if largest <= 0.0:
        raise ValueError("Fisher matrix has no positive direction")
    threshold = rcond * largest
    if eigenvalues[0] < -threshold:
        raise ValueError("Fisher matrix is not positive semidefinite")
    retained = eigenvalues > threshold
    if not np.any(retained):
        raise ValueError("Fisher projection retained no identifiable direction")
    coordinates = eigenvectors[:, retained].T @ shift
    return np.asarray(
        np.abs(coordinates) * np.sqrt(eigenvalues[retained]),
        dtype=np.float64,
    )


__all__ = [
    "FisherProfile",
    "ResponseImpact",
    "StationaryTimeEstimate",
    "adaptive_stationary_time",
    "fisher_projected_bias_sigma",
    "noise_weighted_inner_product",
    "profile_waveform_residual",
    "project_orbital_delay",
    "remove_affine_orbital_motion",
    "response_impact_delta_log_l",
    "time_to_coalescence_from_stationary_time",
]
