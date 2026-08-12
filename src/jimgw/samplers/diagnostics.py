"""Public diagnostics for sampler output."""

from typing import TypeAlias

import numpy as np
from anesthetic.utils import compute_insertion_indexes, insertion_p_value
from numpy.typing import ArrayLike

InsertionIndexDiagnostic: TypeAlias = dict[str, float | int | str]


def insertion_index_diagnostic(
    log_likelihood: ArrayLike,
    log_likelihood_birth: ArrayLike,
    *,
    n_live: int,
) -> InsertionIndexDiagnostic:
    """Test nested-sampling insertion indexes against a discrete uniform law.

    Initial live points are identified by a birth likelihood of ``-inf`` and
    excluded.  Every remaining point represents a constrained-prior
    replacement and should have an insertion index uniformly distributed over
    ``0, ..., n_live - 1`` when replacement sampling is calibrated.

    Args:
        log_likelihood: Death-contour log likelihoods in nested-sample order.
        log_likelihood_birth: Aligned birth-contour log likelihoods.
        n_live: Constant number of live points used during replacement.

    Returns:
        JSON-serializable discrete Kolmogorov--Smirnov diagnostic.

    Raises:
        ValueError: If the arrays are invalid or contain no replacement points.
    """

    if type(n_live) is not int or n_live < 1:
        raise ValueError("n_live must be a positive integer")
    death = np.asarray(log_likelihood, dtype=np.float64)
    birth = np.asarray(log_likelihood_birth, dtype=np.float64)
    if death.ndim != 1 or birth.ndim != 1 or death.shape != birth.shape:
        raise ValueError(
            "log_likelihood and log_likelihood_birth must be aligned 1D arrays"
        )
    if death.size == 0:
        raise ValueError("insertion-index inputs must not be empty")
    if not np.all(np.isfinite(death)):
        raise ValueError("log_likelihood must contain only finite values")
    if np.any(np.isnan(birth)) or np.any(np.isposinf(birth)):
        raise ValueError("log_likelihood_birth must not contain NaN or +inf")

    replacement = np.isfinite(birth)
    if not np.any(replacement):
        raise ValueError("insertion-index inputs contain no replacement points")
    if np.any(death[replacement] <= birth[replacement]):
        raise ValueError(
            "replacement log likelihoods must exceed their birth likelihoods"
        )

    indexes = compute_insertion_indexes(death, birth)[replacement]
    if np.any(indexes < 0) or np.any(indexes >= n_live):
        raise ValueError("computed insertion indexes fall outside the live set")
    result = insertion_p_value(indexes, n_live)
    return {
        "method": "discrete-uniform-kolmogorov-smirnov",
        "n_live": n_live,
        "sample_size": int(result["sample_size"]),
        "statistic": float(result["D"]),
        "p_value": float(result["p-value"]),
        "index_min": int(np.min(indexes)),
        "index_max": int(np.max(indexes)),
        "index_mean": float(np.mean(indexes)),
        "expected_index_mean": (n_live - 1) / 2.0,
    }
