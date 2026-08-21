"""Tests for deterministic quotient-fold result telemetry."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.injection_campaign.folded_results import (
    extract_unfolded_weighted_posterior,
    unfolded_posterior_telemetry,
)


def test_unfolded_telemetry_uses_posterior_mass_and_base_prior_support() -> None:
    input_log_weights = np.log(np.asarray([0.4, 0.6]))
    log_branch_probabilities = np.full((2, 8), -np.inf)
    log_branch_probabilities[0, 0] = 0.0
    log_branch_probabilities[1, :2] = np.log([0.25, 0.75])
    base_log_priors = np.zeros((2, 8))
    base_log_priors[0, 3:5] = -np.inf
    base_log_priors[1, 7] = -np.inf
    true_log_likelihoods = np.asarray(
        [
            [0.0, 0.5, 1.0, 50.0, 50.0, 1.5, 2.0, 2.5],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0],
        ]
    )

    telemetry = unfolded_posterior_telemetry(
        input_log_weights,
        base_log_priors,
        log_branch_probabilities,
        true_log_likelihoods,
    )

    entropy = -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
    assert telemetry["normalized_conditional_image_entropy"] == pytest.approx(
        0.6 * entropy / np.log(8.0)
    )
    assert telemetry["image_sector_posterior_masses"] == pytest.approx(
        [0.55, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert telemetry["expected_nonidentity_mass"] == pytest.approx(0.45)
    assert telemetry["zero_support_image_fraction"] == pytest.approx(3.0 / 16.0)
    assert telemetry["folded_points"] == 2
    assert telemetry["group_order"] == 8
    gaps = telemetry["supported_image_log_likelihood_gaps"]
    assert gaps["within_orbit_span_weighted_quantiles"] == pytest.approx(
        {"p05": 2.5, "p50": 6.0, "p95": 6.0}
    )
    assert gaps["identity_absolute_gap_weighted_quantiles"]["p95"] == pytest.approx(
        6.0
    )


@pytest.mark.parametrize(
    ("input_log_weights", "base_log_priors", "branch_probabilities", "message"),
    [
        (np.zeros((1, 1)), np.zeros((1, 8)), np.zeros((1, 8)), "one-dimensional"),
        (np.zeros(2), np.zeros((1, 8)), np.zeros((1, 8)), "one row per"),
        (np.zeros(1), np.zeros((1, 7)), np.zeros((1, 7)), "eight images"),
        (
            np.asarray([np.nan]),
            np.zeros((1, 8)),
            np.zeros((1, 8)),
            "invalid",
        ),
        (
            np.asarray([0.0]),
            np.zeros((1, 8)),
            np.full((1, 8), -np.inf),
            "normalized",
        ),
    ],
)
def test_unfolded_telemetry_rejects_invalid_tables(
    input_log_weights: np.ndarray,
    base_log_priors: np.ndarray,
    branch_probabilities: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        unfolded_posterior_telemetry(
            input_log_weights,
            base_log_priors,
            branch_probabilities,
        )


def test_folded_extraction_unfolds_sampling_space_before_reverse_transforms() -> None:
    calls: list[str] = []
    raw_positions = np.asarray([[10.0, 1.0], [20.0, 2.0]])
    raw_log_weights = np.log(np.asarray([0.4, 0.6]))
    folded_death = np.asarray([1.5, 2.5])
    folded_birth = np.asarray([-np.inf, 1.0])
    images = np.arange(32.0).reshape(16, 2)
    branch_probabilities = np.full((2, 8), -np.inf)
    branch_probabilities[0, :2] = np.log([0.25, 0.75])
    branch_probabilities[1, 0] = 0.0
    split_weights = (raw_log_weights[:, None] + branch_probabilities).reshape(-1)
    true_log_likelihoods = np.linspace(-2.0, 2.0, 16)
    base_log_priors = np.zeros(16)
    base_log_priors[[3, 7, 15]] = -np.inf

    class _Jim:
        def get_weighted_samples(self, *, space: str) -> dict[str, np.ndarray]:
            calls.append(f"weighted:{space}")
            return {
                "samples": raw_positions,
                "log_likelihood": folded_death,
                "log_likelihood_birth": folded_birth,
                "log_weights": raw_log_weights,
            }

        def unfold_weighted_samples(
            self,
            positions: np.ndarray,
            log_weights: np.ndarray,
            *,
            batch_size: int | None,
        ) -> SimpleNamespace:
            calls.append(f"unfold:{batch_size}")
            np.testing.assert_array_equal(positions, raw_positions)
            np.testing.assert_array_equal(log_weights, raw_log_weights)
            return SimpleNamespace(
                positions=images,
                log_weights=split_weights,
                true_log_likelihoods=true_log_likelihoods,
                base_log_priors=base_log_priors,
                log_branch_probabilities=branch_probabilities.reshape(-1),
            )

        def samples_to_prior_space(
            self, positions: np.ndarray
        ) -> dict[str, np.ndarray]:
            calls.append("reverse")
            np.testing.assert_array_equal(positions, images[[0, 1, 8]])
            return {"physical_x": positions[:, 0] + 100.0}

    expected_insertion = {"minimum_p_value": 0.75}

    def insertion_diagnostic(
        death: np.ndarray,
        birth: np.ndarray,
        *,
        n_live: int,
    ) -> dict[str, float]:
        calls.append(f"insertion:{n_live}")
        np.testing.assert_array_equal(death, folded_death)
        np.testing.assert_array_equal(birth, folded_birth)
        return expected_insertion

    extracted = extract_unfolded_weighted_posterior(
        _Jim(),
        n_live=2,
        batch_size=3,
        insertion_diagnostic_fn=insertion_diagnostic,
    )

    assert calls == ["weighted:sampling", "insertion:2", "unfold:3", "reverse"]
    assert extracted.insertion_diagnostic is expected_insertion
    np.testing.assert_array_equal(extracted.folded_log_likelihood, folded_death)
    np.testing.assert_array_equal(extracted.folded_log_likelihood_birth, folded_birth)
    np.testing.assert_array_equal(
        extracted.true_log_likelihood,
        true_log_likelihoods[[0, 1, 8]],
    )
    np.testing.assert_allclose(
        extracted.log_weights,
        split_weights[[0, 1, 8]],
    )
    assert extracted.samples.keys() == {"physical_x"}
    assert extracted.telemetry["image_sector_posterior_masses"] == pytest.approx(
        [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert "supported_image_log_likelihood_gaps" in extracted.telemetry
