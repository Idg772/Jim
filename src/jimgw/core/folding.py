"""Pure quotient-folding algebra for the network-sky extrinsic coordinates.

The three commuting involutions act on a flat sampling-space position whose
resolved coordinates are ``(cos_iota, azimuth, psi)``.  The element number is
the three-bit mask ``H | S | P`` and image order is therefore deterministic.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float


class ResolvedFoldSymmetry(NamedTuple):
    """Sampling-space indices and detector-plane reflection centre."""

    indices: tuple[int, int, int]
    azimuth_reflection_center: float


class FoldedTargetValues(NamedTuple):
    """Quotient-target values derived from one complete image table."""

    log_prior: Array
    log_likelihood: Array
    log_posterior: Array
    log_branch_probabilities: Array


class UnfoldedWeightedSamples(NamedTuple):
    """Fixed eight-image expansion of a weighted folded collection."""

    positions: Array
    log_weights: Array
    true_log_likelihoods: Array
    base_log_priors: Array
    log_branch_probabilities: Array


def _apply_fold_element(
    position: Float[Array, " n_dims"],
    element: int,
    fold: ResolvedFoldSymmetry,
) -> Float[Array, " n_dims"]:
    """Apply one statically selected element of the eight-image fold group."""

    if element < 0 or element >= 8:
        raise ValueError("fold element must be an integer in [0, 8)")

    c_idx, phi_idx, psi_idx = fold.indices
    image = jnp.asarray(position)
    if element & 1:  # H: coupled raw-coordinate handedness map.
        image = image.at[c_idx].set(-image[c_idx])
        image = image.at[phi_idx].set(jnp.mod(image[phi_idx] + jnp.pi, 2.0 * jnp.pi))
    if element & 2:  # S: reflection through the detector plane.
        image = image.at[phi_idx].set(
            jnp.mod(
                2.0 * fold.azimuth_reflection_center - image[phi_idx],
                2.0 * jnp.pi,
            )
        )
    if element & 4:  # P: polarization half-period.
        image = image.at[psi_idx].set(jnp.mod(image[psi_idx] + 0.5 * jnp.pi, jnp.pi))
    return image


def _build_fold_images(
    position: Float[Array, " n_dims"],
    fold: ResolvedFoldSymmetry,
) -> Float[Array, "8 n_dims"]:
    """Return all fold images in deterministic bit-mask order."""

    return jnp.stack(
        tuple(_apply_fold_element(position, element, fold) for element in range(8))
    )


def _normalize_fold_periods(
    position: Float[Array, " n_dims"],
    fold: ResolvedFoldSymmetry,
) -> Float[Array, " n_dims"]:
    """Normalize the two periodic fold coordinates to their half-open periods."""

    _, phi_idx, psi_idx = fold.indices
    normalized = jnp.asarray(position)
    normalized = normalized.at[phi_idx].set(jnp.mod(normalized[phi_idx], 2.0 * jnp.pi))
    normalized = normalized.at[psi_idx].set(jnp.mod(normalized[psi_idx], jnp.pi))
    return normalized


def _centered_hatted_azimuth(
    position: Float[Array, " n_dims"],
    fold: ResolvedFoldSymmetry,
) -> Float[Array, ""]:
    """Return delta_hat in ``[0, 2 pi)`` for a normalized position."""

    c_idx, phi_idx, _ = fold.indices
    c = position[c_idx]
    phi_hat = jnp.mod(
        position[phi_idx] + jnp.pi * (c >= 0.0),
        2.0 * jnp.pi,
    )
    return jnp.mod(
        phi_hat - fold.azimuth_reflection_center,
        2.0 * jnp.pi,
    )


def _in_fundamental_domain(
    position: Float[Array, " n_dims"],
    fold: ResolvedFoldSymmetry,
) -> Array:
    """Return whether a position represents its orbit's fundamental sector."""

    normalized = _normalize_fold_periods(position, fold)
    c_idx, _, psi_idx = fold.indices
    c = normalized[c_idx]
    delta_hat = _centered_hatted_azimuth(normalized, fold)
    inclination_sector = jnp.logical_or(
        jnp.logical_and(c > 0.0, delta_hat <= jnp.pi),
        jnp.logical_and(c == 0.0, delta_hat <= 0.5 * jnp.pi),
    )
    return jnp.logical_and(normalized[psi_idx] < 0.5 * jnp.pi, inclination_sector)


def _fold_to_fundamental(
    position: Float[Array, " n_dims"],
    fold: ResolvedFoldSymmetry,
) -> Float[Array, " n_dims"]:
    """Choose the deterministic representative of a fold-group orbit."""

    normalized = _normalize_fold_periods(position, fold)
    c_idx, _, psi_idx = fold.indices
    c = normalized[c_idx]

    handed = _apply_fold_element(normalized, 1, fold)
    positive_c = jnp.where(c < 0.0, handed, normalized)
    reflected = _apply_fold_element(positive_c, 2, fold)
    positive_c = jnp.where(
        _centered_hatted_azimuth(positive_c, fold) > jnp.pi,
        reflected,
        positive_c,
    )

    # H and S both preserve c == 0.  Enumerate that four-image null-stratum
    # orbit and let argmin provide the element-index part of the lexicographic
    # (delta_hat, element_index) tie-break.
    null_images = jnp.stack(
        tuple(_apply_fold_element(normalized, element, fold) for element in range(4))
    )
    null_deltas = jnp.stack(
        tuple(_centered_hatted_azimuth(image, fold) for image in null_images)
    )
    null_representative = null_images[jnp.argmin(null_deltas)]
    representative = jnp.where(c == 0.0, null_representative, positive_c)

    polarized = _apply_fold_element(representative, 4, fold)
    return jnp.where(
        representative[psi_idx] >= 0.5 * jnp.pi,
        polarized,
        representative,
    )


def _folded_target_from_image_values(
    log_priors: Float[Array, " images"],
    true_log_likelihoods: Float[Array, " images"],
) -> FoldedTargetValues:
    """Reduce base-image values to the exact normalized quotient target.

    Unsupported images are identified by non-finite base log-prior values and
    contribute zero density.  Crucially, the folded prior is the *sum* of image
    priors; there is no division by the group order.
    """

    log_priors = jnp.asarray(log_priors)
    true_log_likelihoods = jnp.asarray(true_log_likelihoods)
    supported = jnp.isfinite(log_priors)
    supported_log_priors = jnp.where(supported, log_priors, -jnp.inf)
    log_joint_terms = jnp.where(
        supported,
        log_priors + true_log_likelihoods,
        -jnp.inf,
    )
    folded_log_prior = logsumexp(supported_log_priors)
    folded_log_posterior = logsumexp(log_joint_terms)

    safe_log_prior = jnp.where(jnp.isfinite(folded_log_prior), folded_log_prior, 0.0)
    folded_log_likelihood = jnp.where(
        jnp.isfinite(folded_log_prior),
        folded_log_posterior - safe_log_prior,
        -jnp.inf,
    )
    safe_log_posterior = jnp.where(
        jnp.isfinite(folded_log_posterior), folded_log_posterior, 0.0
    )
    log_branch_probabilities = jnp.where(
        jnp.isfinite(folded_log_posterior),
        log_joint_terms - safe_log_posterior,
        -jnp.inf,
    )
    return FoldedTargetValues(
        log_prior=folded_log_prior,
        log_likelihood=folded_log_likelihood,
        log_posterior=folded_log_posterior,
        log_branch_probabilities=log_branch_probabilities,
    )


def _evaluate_fold_images(
    position: Float[Array, " n_dims"],
    fold: ResolvedFoldSymmetry,
    base_log_prior_fn: Callable[[Array], Array],
    base_log_likelihood_fn: Callable[[Array], Array],
) -> FoldedTargetValues:
    """Evaluate the base target on every image and reduce it to the quotient."""

    images = _build_fold_images(position, fold)
    log_priors = jnp.stack(tuple(base_log_prior_fn(image) for image in images))
    true_log_likelihoods = jnp.stack(
        tuple(base_log_likelihood_fn(image) for image in images)
    )
    return _folded_target_from_image_values(log_priors, true_log_likelihoods)


def _folded_log_prior(
    position: Float[Array, " n_dims"],
    fold: ResolvedFoldSymmetry,
    base_log_prior_fn: Callable[[Array], Array],
) -> Array:
    """Evaluate the normalized quotient prior on the fundamental domain."""

    images = _build_fold_images(position, fold)
    log_priors = jnp.stack(tuple(base_log_prior_fn(image) for image in images))
    supported_log_priors = jnp.where(jnp.isfinite(log_priors), log_priors, -jnp.inf)
    log_prior = logsumexp(supported_log_priors)
    return jnp.where(_in_fundamental_domain(position, fold), log_prior, -jnp.inf)


def _folded_log_likelihood(
    position: Float[Array, " n_dims"],
    fold: ResolvedFoldSymmetry,
    base_log_prior_fn: Callable[[Array], Array],
    base_log_likelihood_fn: Callable[[Array], Array],
) -> Array:
    """Evaluate the prior-weighted conditional image likelihood."""

    return _evaluate_fold_images(
        position,
        fold,
        base_log_prior_fn,
        base_log_likelihood_fn,
    ).log_likelihood


def _folded_log_likelihood_from_cache(
    position: Float[Array, " n_dims"],
    cache: Any,
    fold: ResolvedFoldSymmetry,
    base_log_prior_fn: Callable[[Array], Array],
    base_log_likelihood_from_cache_fn: Callable[[Array, Any], Array],
) -> Array:
    """Evaluate the folded likelihood using one caller-supplied waveform cache."""

    return _evaluate_fold_images(
        position,
        fold,
        base_log_prior_fn,
        lambda image: base_log_likelihood_from_cache_fn(image, cache),
    ).log_likelihood


def _folded_log_posterior(
    position: Float[Array, " n_dims"],
    fold: ResolvedFoldSymmetry,
    base_log_prior_fn: Callable[[Array], Array],
    base_log_likelihood_fn: Callable[[Array], Array],
) -> Array:
    """Evaluate the quotient posterior numerator on the fundamental domain."""

    log_posterior = _evaluate_fold_images(
        position,
        fold,
        base_log_prior_fn,
        base_log_likelihood_fn,
    ).log_posterior
    return jnp.where(_in_fundamental_domain(position, fold), log_posterior, -jnp.inf)


def _unfold_weighted_samples_static(
    positions: Float[Array, "n_points n_dims"],
    log_weights: Float[Array, " n_points"],
    fold: ResolvedFoldSymmetry,
    base_log_prior_fn: Callable[[Array], Array],
    base_log_likelihood_from_cache_fn: Callable[[Array, Any], Array],
    build_cache: Callable[[Array], Any],
    *,
    batch_size: int | None = None,
) -> UnfoldedWeightedSamples:
    """JIT-compatible fixed-shape eight-image expansion kernel.

    The returned arrays have fixed size ``n_points * 8`` and retain zero-mass
    branches as ``-inf`` log weights. Host-side input filtering and validation
    belong to `_unfold_weighted_samples`.
    """

    positions = jnp.asarray(positions)
    log_weights = jnp.asarray(log_weights)
    if positions.ndim != 2:
        raise ValueError("positions must have shape (n_points, n_dims)")
    if log_weights.ndim != 1 or log_weights.shape[0] != positions.shape[0]:
        raise ValueError("log_weights must have shape (n_points,)")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive when provided")

    def unfold_one(inputs):
        position, log_weight = inputs
        cache = build_cache(position)
        images = _build_fold_images(position, fold)
        image_log_priors = jnp.stack(
            tuple(base_log_prior_fn(image) for image in images)
        )
        true_log_likelihoods = jnp.stack(
            tuple(base_log_likelihood_from_cache_fn(image, cache) for image in images)
        )
        target = _folded_target_from_image_values(
            image_log_priors,
            true_log_likelihoods,
        )
        return (
            images,
            log_weight + target.log_branch_probabilities,
            true_log_likelihoods,
            image_log_priors,
            target.log_branch_probabilities,
        )

    if batch_size is None:
        images, split_weights, likelihoods, priors, branch_probabilities = jax.lax.map(
            unfold_one, (positions, log_weights)
        )
    else:
        images, split_weights, likelihoods, priors, branch_probabilities = jax.lax.map(
            unfold_one,
            (positions, log_weights),
            batch_size=batch_size,
        )
    n_dims = positions.shape[1]
    return UnfoldedWeightedSamples(
        positions=images.reshape((-1, n_dims)),
        log_weights=split_weights.reshape((-1,)),
        true_log_likelihoods=likelihoods.reshape((-1,)),
        base_log_priors=priors.reshape((-1,)),
        log_branch_probabilities=branch_probabilities.reshape((-1,)),
    )


def _unfold_weighted_samples(
    positions: Float[Array, "n_points n_dims"],
    log_weights: Float[Array, " n_points"],
    fold: ResolvedFoldSymmetry,
    base_log_prior_fn: Callable[[Array], Array],
    base_log_likelihood_from_cache_fn: Callable[[Array, Any], Array],
    build_cache: Callable[[Array], Any],
    *,
    batch_size: int | None = None,
) -> UnfoldedWeightedSamples:
    """Safely unfold a host-side weighted nested-sampling collection.

    Exactly zero-mass nested rows (``log_weight == -inf``) are discarded before
    the static kernel builds waveform caches. Unsupported image branches remain
    present in the returned eight-image tables with ``-inf`` weights.
    """

    host_positions = np.asarray(positions)
    host_log_weights = np.asarray(log_weights)
    if host_positions.ndim != 2:
        raise ValueError("positions must have shape (n_points, n_dims)")
    if (
        host_log_weights.ndim != 1
        or host_log_weights.shape[0] != host_positions.shape[0]
    ):
        raise ValueError("log_weights must have shape (n_points,)")
    if np.any(np.isnan(host_log_weights)) or np.any(np.isposinf(host_log_weights)):
        raise ValueError("log_weights may be finite or -inf, but not NaN or +inf")

    retained = ~np.isneginf(host_log_weights)
    if not np.any(retained):
        raise ValueError("at least one finite-weight row is required for unfolding")
    retained_positions = jnp.asarray(host_positions[retained])
    retained_log_weights = jnp.asarray(host_log_weights[retained])
    unfolded = _unfold_weighted_samples_static(
        retained_positions,
        retained_log_weights,
        fold,
        base_log_prior_fn,
        base_log_likelihood_from_cache_fn,
        build_cache,
        batch_size=batch_size,
    )

    n_retained = int(retained_log_weights.shape[0])
    branch_probabilities = np.asarray(unfolded.log_branch_probabilities).reshape(
        n_retained, 8
    )
    impossible_rows = ~np.any(np.isfinite(branch_probabilities), axis=1)
    if np.any(impossible_rows):
        bad_rows = np.flatnonzero(retained)[impossible_rows].tolist()
        raise ValueError(
            "finite-weight row has no finite base posterior branch: "
            f"input row(s) {bad_rows}"
        )

    input_log_total = float(logsumexp(retained_log_weights))
    output_log_total = float(logsumexp(unfolded.log_weights))
    if not np.isclose(output_log_total, input_log_total, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(
            "deterministic unfolding did not preserve total log weight: "
            f"input={input_log_total}, output={output_log_total}"
        )
    return unfolded
