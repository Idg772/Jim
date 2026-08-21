"""Behavioral tests for quotient-folding algebra and targets."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.folding import (
    ResolvedFoldSymmetry,
    _apply_fold_element,
    _build_fold_images,
    _fold_to_fundamental,
    _folded_log_likelihood,
    _folded_log_likelihood_from_cache,
    _folded_log_posterior,
    _folded_log_prior,
    _folded_target_from_image_values,
    _in_fundamental_domain,
    _unfold_weighted_samples,
    _unfold_weighted_samples_static,
)


def _fold() -> ResolvedFoldSymmetry:
    return ResolvedFoldSymmetry(
        indices=(1, 2, 3),
        azimuth_reflection_center=0.37,
    )


def test_raw_fold_group_has_eight_unit_jacobian_images() -> None:
    fold = _fold()
    position = jnp.asarray([9.0, -0.42, 1.13, 0.29, -4.0])
    images = _build_fold_images(position, fold)

    assert images.shape == (8, 5)
    np.testing.assert_allclose(images[0], position)
    assert len({tuple(np.asarray(image).round(12)) for image in images}) == 8

    # The handedness operation couples the raw azimuth and inclination sign.
    handed = _apply_fold_element(position, 1, fold)
    assert handed[fold.indices[0]] == -position[fold.indices[0]]
    assert handed[fold.indices[1]] == pytest.approx(
        float(jnp.mod(position[fold.indices[1]] + jnp.pi, 2.0 * jnp.pi))
    )

    def hatted_azimuth(point: jax.Array) -> jax.Array:
        c = point[fold.indices[0]]
        phi = point[fold.indices[1]]
        return jnp.mod(phi + jnp.pi * (c >= 0), 2.0 * jnp.pi)

    np.testing.assert_allclose(hatted_azimuth(handed), hatted_azimuth(position))

    for left, right in itertools.product(range(8), repeat=2):
        composed = _apply_fold_element(
            _apply_fold_element(position, left, fold), right, fold
        )
        expected = _apply_fold_element(position, left ^ right, fold)
        np.testing.assert_allclose(composed, expected, atol=1.0e-12)

    for element in range(8):
        jacobian = jax.jacfwd(
            lambda point, element=element: _apply_fold_element(point, element, fold)
        )(position)
        assert abs(float(jnp.linalg.det(jacobian))) == pytest.approx(1.0)


def test_fold_to_fundamental_is_orbit_invariant_with_exact_boundary_rules() -> None:
    fold = _fold()
    position = jnp.asarray([9.0, -0.42, 5.41, 2.72, -4.0])
    images = _build_fold_images(position, fold)
    representatives = jax.vmap(lambda image: _fold_to_fundamental(image, fold))(images)

    np.testing.assert_allclose(
        representatives,
        jnp.broadcast_to(representatives[0], representatives.shape),
        atol=1.0e-12,
    )
    representative = representatives[0]
    assert bool(_in_fundamental_domain(representative, fold))
    np.testing.assert_allclose(
        _fold_to_fundamental(representative, fold), representative, atol=1.0e-12
    )
    # Away from stabilizer boundaries, exactly one orbit image belongs to F.
    assert (
        int(jnp.sum(jax.vmap(lambda x: _in_fundamental_domain(x, fold))(images))) == 1
    )

    c_idx, phi_idx, psi_idx = fold.indices

    # The detector-plane fixed boundary delta_hat == pi is included in F.
    fixed_boundary = position.at[c_idx].set(0.4)
    fixed_boundary = fixed_boundary.at[phi_idx].set(
        jnp.mod(fold.azimuth_reflection_center, 2.0 * jnp.pi)
    )
    fixed_boundary = fixed_boundary.at[psi_idx].set(0.1)
    assert bool(_in_fundamental_domain(fixed_boundary, fold))
    np.testing.assert_allclose(
        _fold_to_fundamental(fixed_boundary, fold), fixed_boundary, atol=1.0e-12
    )

    # On c == 0 all H/S images select the same lexicographically first image,
    # whose centered hatted azimuth lies in [0, pi/2].
    null_position = position.at[c_idx].set(0.0).at[psi_idx].set(0.2)
    null_images = jnp.stack(
        tuple(_apply_fold_element(null_position, element, fold) for element in range(4))
    )
    null_representatives = jax.vmap(lambda image: _fold_to_fundamental(image, fold))(
        null_images
    )
    np.testing.assert_allclose(
        null_representatives,
        jnp.broadcast_to(null_representatives[0], null_representatives.shape),
        atol=1.0e-12,
    )
    assert bool(_in_fundamental_domain(null_representatives[0], fold))

    null_boundary = null_position.at[phi_idx].set(
        jnp.mod(
            fold.azimuth_reflection_center + 0.5 * jnp.pi - jnp.pi,
            2.0 * jnp.pi,
        )
    )
    assert bool(_in_fundamental_domain(null_boundary, fold))
    null_boundary_images = jnp.stack(
        tuple(_apply_fold_element(null_boundary, element, fold) for element in range(4))
    )
    null_boundary_representatives = jax.vmap(
        lambda image: _fold_to_fundamental(image, fold)
    )(null_boundary_images)
    np.testing.assert_allclose(
        null_boundary_representatives,
        jnp.broadcast_to(
            null_boundary_representatives[0], null_boundary_representatives.shape
        ),
        atol=1.0e-12,
    )

    # Full-period endpoints normalize before sector selection; the half-period
    # polarization boundary is assigned to the lower sector by P.
    endpoint = position.at[c_idx].set(0.7)
    endpoint = endpoint.at[phi_idx].set(2.0 * jnp.pi)
    endpoint = endpoint.at[psi_idx].set(jnp.pi)
    endpoint_folded = _fold_to_fundamental(endpoint, fold)
    assert endpoint_folded[phi_idx] >= 0.0
    assert endpoint_folded[phi_idx] < 2.0 * jnp.pi
    assert endpoint_folded[psi_idx] == pytest.approx(0.0)
    assert bool(_in_fundamental_domain(endpoint_folded, fold))

    half_period = endpoint.at[psi_idx].set(0.5 * jnp.pi)
    assert _fold_to_fundamental(half_period, fold)[psi_idx] == pytest.approx(0.0)


def test_quotient_target_preserves_evidence_with_noninvariant_support() -> None:
    # D=[-2, 1] is deliberately not invariant under x -> -x.
    def log_prior(x: jax.Array) -> jax.Array:
        return jnp.where(
            jnp.logical_and(x >= -2.0, x <= 1.0),
            -jnp.log(3.0),
            -jnp.inf,
        )

    def folded_values(x: jax.Array):
        images = jnp.asarray([x, -x])
        return _folded_target_from_image_values(
            jax.vmap(log_prior)(images),
            0.7 * images,
        )

    full_grid = jnp.linspace(-2.0, 1.0, 40_001)
    original_density = jnp.exp(jax.vmap(log_prior)(full_grid))
    original_evidence = jnp.trapezoid(
        original_density * jnp.exp(0.7 * full_grid), full_grid
    )

    folded_grid = jnp.linspace(0.0, 2.0, 40_001)
    folded = jax.vmap(folded_values)(folded_grid)
    folded_prior_density = jnp.exp(folded.log_prior)
    folded_evidence = jnp.trapezoid(jnp.exp(folded.log_posterior), folded_grid)

    assert float(jnp.trapezoid(folded_prior_density, folded_grid)) == pytest.approx(
        1.0, abs=5.0e-5
    )
    assert float(folded_evidence) == pytest.approx(float(original_evidence), abs=5.0e-5)

    # Conditional branch probabilities unfold the quotient posterior back to
    # the positive and negative halves of the original posterior exactly.
    unfolded_branch_density = jnp.exp(
        folded.log_posterior[:, None]
        - jnp.log(folded_evidence)
        + folded.log_branch_probabilities
    )
    unfolded_branch_masses = jnp.trapezoid(unfolded_branch_density, folded_grid, axis=0)
    positive_grid = jnp.linspace(0.0, 1.0, 20_001)
    negative_grid = jnp.linspace(-2.0, 0.0, 40_001)
    expected_positive_mass = (
        jnp.trapezoid(
            jnp.exp(jax.vmap(log_prior)(positive_grid) + 0.7 * positive_grid),
            positive_grid,
        )
        / original_evidence
    )
    expected_negative_mass = (
        jnp.trapezoid(
            jnp.exp(jax.vmap(log_prior)(negative_grid) + 0.7 * negative_grid),
            negative_grid,
        )
        / original_evidence
    )
    assert unfolded_branch_masses[0] == pytest.approx(
        float(expected_positive_mass), abs=5.0e-5
    )
    assert unfolded_branch_masses[1] == pytest.approx(
        float(expected_negative_mass), abs=5.0e-5
    )
    assert float(jnp.sum(unfolded_branch_masses)) == pytest.approx(1.0, abs=5.0e-5)

    # There is no division by the group order in the folded prior.
    point = folded_values(jnp.asarray(1.5))
    assert float(jnp.exp(point.log_prior)) == pytest.approx(1.0 / 3.0)

    # The rejected likelihood-only /2 construction loses evidence because only
    # one reflected image is supported on part of D.
    full_images = jnp.stack((full_grid, -full_grid), axis=1)
    full_image_priors = jax.vmap(jax.vmap(log_prior))(full_images)
    full_image_log_likelihoods = 0.7 * full_images
    supported_terms = jnp.where(
        jnp.isfinite(full_image_priors),
        full_image_priors + full_image_log_likelihoods,
        -jnp.inf,
    )
    bad_log_likelihood = (
        jax.scipy.special.logsumexp(supported_terms, axis=1)
        - jax.vmap(log_prior)(full_grid)
        - jnp.log(2.0)
    )
    bad_evidence = jnp.trapezoid(
        original_density * jnp.exp(bad_log_likelihood), full_grid
    )
    assert abs(float(bad_evidence - original_evidence)) > 0.05

    invariant = _folded_target_from_image_values(
        jnp.full(8, -jnp.log(8.0)),
        jnp.linspace(-1.0, 1.0, 8),
    )
    expected_mean_likelihood = jax.scipy.special.logsumexp(
        jnp.linspace(-1.0, 1.0, 8)
    ) - jnp.log(8.0)
    assert invariant.log_likelihood == pytest.approx(float(expected_mean_likelihood))


def test_folded_callbacks_evaluate_all_images_and_mask_the_domain() -> None:
    fold = _fold()
    c_idx, phi_idx, psi_idx = fold.indices
    representative = jnp.asarray([2.0, 0.5, 0.0, 0.2, -1.0])
    # For c > 0, phi=center-pi gives delta_hat=0 and is inside F.
    representative = representative.at[phi_idx].set(
        jnp.mod(fold.azimuth_reflection_center - jnp.pi, 2.0 * jnp.pi)
    )

    def base_log_prior(position: jax.Array) -> jax.Array:
        c = position[c_idx]
        return jnp.where(
            jnp.logical_and(c >= -1.0, c <= 0.25),
            -jnp.log(1.25),
            -jnp.inf,
        )

    def base_log_likelihood(position: jax.Array) -> jax.Array:
        return (
            0.4 * position[c_idx]
            + jnp.cos(position[phi_idx])
            + 0.2 * jnp.sin(2.0 * position[psi_idx])
        )

    images = _build_fold_images(representative, fold)
    image_priors = jax.vmap(base_log_prior)(images)
    image_likelihoods = jax.vmap(base_log_likelihood)(images)
    expected = _folded_target_from_image_values(image_priors, image_likelihoods)

    folded_prior = jax.jit(lambda x: _folded_log_prior(x, fold, base_log_prior))
    folded_likelihood = jax.jit(
        lambda x: _folded_log_likelihood(x, fold, base_log_prior, base_log_likelihood)
    )
    folded_posterior = jax.jit(
        lambda x: _folded_log_posterior(x, fold, base_log_prior, base_log_likelihood)
    )

    assert folded_prior(representative) == pytest.approx(float(expected.log_prior))
    assert folded_likelihood(representative) == pytest.approx(
        float(expected.log_likelihood)
    )
    assert folded_posterior(representative) == pytest.approx(
        float(expected.log_posterior)
    )

    cache = jnp.asarray(0.73)

    def cached_log_likelihood(position: jax.Array, value: jax.Array) -> jax.Array:
        return base_log_likelihood(position) + value

    cached = jax.jit(
        lambda x: _folded_log_likelihood_from_cache(
            x,
            cache,
            fold,
            base_log_prior,
            cached_log_likelihood,
        )
    )
    assert cached(representative) == pytest.approx(
        float(expected.log_likelihood + cache)
    )

    outside = _apply_fold_element(representative, 1, fold)
    assert not bool(_in_fundamental_domain(outside, fold))
    assert jnp.isneginf(folded_prior(outside))
    assert jnp.isneginf(folded_posterior(outside))


def test_deterministic_unfolding_splits_each_weight_over_aligned_images() -> None:
    fold = _fold()
    c_idx, phi_idx, psi_idx = fold.indices
    phi = jnp.mod(fold.azimuth_reflection_center - jnp.pi, 2.0 * jnp.pi)
    positions = jnp.asarray(
        [
            [2.0, 0.5, phi, 0.2, -1.0],
            [3.0, 0.1, phi, 0.3, -2.0],
        ]
    )
    input_log_weights = jnp.log(jnp.asarray([0.3, 0.7]))

    def base_log_prior(position: jax.Array) -> jax.Array:
        c = position[c_idx]
        return jnp.where(
            jnp.logical_and(c >= -1.0, c <= 0.25),
            -jnp.log(1.25),
            -jnp.inf,
        )

    def build_cache(position: jax.Array) -> jax.Array:
        # This depends only on a coordinate untouched by the image group.
        return position[0]

    def true_log_likelihood(position: jax.Array, cache: jax.Array) -> jax.Array:
        return (
            0.4 * position[c_idx]
            + jnp.cos(position[phi_idx])
            + 0.2 * jnp.sin(2.0 * position[psi_idx])
            + 0.05 * cache
        )

    unfold = jax.jit(
        lambda x, w: _unfold_weighted_samples_static(
            x,
            w,
            fold,
            base_log_prior,
            true_log_likelihood,
            build_cache,
            batch_size=1,
        )
    )
    unfolded = unfold(positions, input_log_weights)

    expected_images = jax.vmap(lambda x: _build_fold_images(x, fold))(positions)
    np.testing.assert_allclose(
        unfolded.positions.reshape(2, 8, 5), expected_images, atol=1.0e-12
    )

    expected_branch_probabilities = []
    expected_likelihoods = []
    for position, images in zip(positions, expected_images, strict=True):
        cache = build_cache(position)
        image_priors = jax.vmap(base_log_prior)(images)
        image_likelihoods = jax.vmap(
            lambda image, cache=cache: true_log_likelihood(image, cache)
        )(images)
        expected = _folded_target_from_image_values(image_priors, image_likelihoods)
        expected_branch_probabilities.append(expected.log_branch_probabilities)
        expected_likelihoods.append(image_likelihoods)
    expected_branch_probabilities = jnp.stack(expected_branch_probabilities)
    expected_likelihoods = jnp.stack(expected_likelihoods)

    np.testing.assert_allclose(
        unfolded.log_branch_probabilities.reshape(2, 8),
        expected_branch_probabilities,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        unfolded.true_log_likelihoods.reshape(2, 8),
        expected_likelihoods,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        unfolded.base_log_priors.reshape(2, 8),
        jax.vmap(jax.vmap(base_log_prior))(expected_images),
        atol=1.0e-12,
    )
    output_weights = unfolded.log_weights.reshape(2, 8)
    np.testing.assert_allclose(
        jax.scipy.special.logsumexp(output_weights, axis=1), input_log_weights
    )
    assert float(jax.scipy.special.logsumexp(unfolded.log_weights)) == pytest.approx(
        0.0, abs=1.0e-12
    )

    first_image_priors = jax.vmap(base_log_prior)(expected_images[0])
    assert jnp.all(jnp.isneginf(output_weights[0][~jnp.isfinite(first_image_priors)]))


def test_unfolding_drops_zero_weight_rows_before_building_caches() -> None:
    fold = _fold()
    positions = jnp.asarray(
        [
            [1.0, 0.2, 0.3, 0.1, -1.0],
            [999.0, 0.2, 0.3, 0.1, -2.0],
            [3.0, 0.2, 0.3, 0.1, -3.0],
        ]
    )
    log_weights = jnp.asarray([jnp.log(0.4), -jnp.inf, jnp.log(0.6)])
    cache_inputs: list[float] = []

    def base_log_prior(position: jax.Array) -> jax.Array:
        return jnp.where(position[fold.indices[0]] < 0.0, 0.0, -jnp.inf)

    def build_cache(position: jax.Array) -> jax.Array:
        jax.debug.callback(
            lambda value: cache_inputs.append(float(value)),
            position[0],
            ordered=True,
        )
        return position[0]

    def true_log_likelihood(position: jax.Array, cache: jax.Array) -> jax.Array:
        return position[1] + 0.01 * cache

    unfolded = _unfold_weighted_samples(
        positions,
        log_weights,
        fold,
        base_log_prior,
        true_log_likelihood,
        build_cache,
        batch_size=1,
    )
    jax.block_until_ready(unfolded.log_weights)

    assert unfolded.positions.shape == (16, 5)
    assert cache_inputs == [1.0, 3.0]
    assert 999.0 not in cache_inputs
    output_weights = unfolded.log_weights.reshape(2, 8)
    assert jnp.all(jnp.sum(jnp.isneginf(output_weights), axis=1) == 4)
    assert float(jax.scipy.special.logsumexp(unfolded.log_weights)) == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_unfolding_rejects_finite_row_with_no_posterior_branch_without_nans() -> None:
    fold = _fold()
    positions = jnp.asarray([[1.0, 0.2, 0.3, 0.1, -1.0]])
    log_weights = jnp.asarray([0.0])

    def base_log_prior(position: jax.Array) -> jax.Array:
        del position
        return jnp.asarray(0.0)

    def build_cache(position: jax.Array) -> jax.Array:
        return position[0]

    def impossible_log_likelihood(position: jax.Array, cache: jax.Array) -> jax.Array:
        del position, cache
        return jnp.asarray(-jnp.inf)

    static_unfold = jax.jit(
        lambda x, w: _unfold_weighted_samples_static(
            x,
            w,
            fold,
            base_log_prior,
            impossible_log_likelihood,
            build_cache,
            batch_size=1,
        )
    )
    impossible = static_unfold(positions, log_weights)
    assert jnp.all(jnp.isneginf(impossible.log_branch_probabilities))
    assert not bool(jnp.any(jnp.isnan(impossible.log_branch_probabilities)))
    assert not bool(jnp.any(jnp.isnan(impossible.log_weights)))

    with pytest.raises(ValueError, match="no finite base posterior branch"):
        _unfold_weighted_samples(
            positions,
            log_weights,
            fold,
            base_log_prior,
            impossible_log_likelihood,
            build_cache,
            batch_size=1,
        )


@pytest.mark.parametrize(
    ("positions", "log_weights", "message"),
    [
        (jnp.zeros(5), jnp.zeros(1), "positions must have shape"),
        (jnp.zeros((1, 5)), jnp.zeros((1, 1)), "log_weights must have shape"),
        (jnp.zeros((2, 5)), jnp.zeros(1), "log_weights must have shape"),
        (jnp.zeros((1, 5)), jnp.asarray([jnp.nan]), "finite or -inf"),
        (jnp.zeros((1, 5)), jnp.asarray([jnp.inf]), "finite or -inf"),
        (jnp.zeros((1, 5)), jnp.asarray([-jnp.inf]), "finite-weight row"),
    ],
    ids=(
        "positions-rank",
        "weights-rank",
        "weights-length",
        "weights-nan",
        "weights-positive-infinity",
        "no-finite-weight-row",
    ),
)
def test_host_unfolding_rejects_invalid_inputs_before_cache_builds(
    positions: jax.Array,
    log_weights: jax.Array,
    message: str,
) -> None:
    fold = _fold()
    cache_inputs: list[float] = []

    def base_log_prior(position: jax.Array) -> jax.Array:
        del position
        return jnp.asarray(0.0)

    def build_cache(position: jax.Array) -> jax.Array:
        jax.debug.callback(
            lambda value: cache_inputs.append(float(value)),
            position[0],
            ordered=True,
        )
        return position[0]

    def true_log_likelihood(position: jax.Array, cache: jax.Array) -> jax.Array:
        return position[1] + cache

    with pytest.raises(ValueError, match=message):
        _unfold_weighted_samples(
            positions,
            log_weights,
            fold,
            base_log_prior,
            true_log_likelihood,
            build_cache,
            batch_size=1,
        )

    assert cache_inputs == []
