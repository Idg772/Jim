"""Tests for deterministic quotient-fold result telemetry."""

import numpy as np
import pytest

from benchmarks.injection_campaign.folded_results import (
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

    telemetry = unfolded_posterior_telemetry(
        input_log_weights,
        base_log_priors,
        log_branch_probabilities,
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
