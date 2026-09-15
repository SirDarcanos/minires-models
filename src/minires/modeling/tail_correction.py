"""Bounded convex correction of the frozen run-011 closest continuous anchor."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

VERSION = "minires-tail-focused-correction-v1"
FEATURES = ("anchor_g", "neural_minus_xgboost_g", "absolute_disagreement_g")
CONTRACT = {
    "version": VERSION,
    "features": list(FEATURES),
    "anchor": "float64_0.8_times_neural_plus_(1.0_minus_0.8)_times_xgboost",
    "feature_clip": 1.0,
    "scale_floor": 1e-12,
    "coefficient_l1_bound_g": 2.0,
    "ordinary_squared_error_weight": 0.1,
    "tail_squared_excess_weight": 4.0,
    "tail_margin_g": 4.5,
    "anchor_departure_weight": 1.0,
    "coefficient_squared_penalty": 0.01,
    "iterations": 2000,
    "optimizer": "zero_initialized_euclidean_l1_projected_gradient",
    "step": "1/(2*5.1*frobenius_design_squared/n+0.02)",
    "target_residual_clipping": False,
    "correction_scales": [0.0, 1.0],
}
FLOAT32_MAXIMUM = 3.4028234663852886e38


def features(base_predictions: Sequence[Sequence[float]]) -> tuple[Any, Any]:
    """Return the exact float64 convex anchor and three prediction-only inputs."""
    import numpy as np

    matrix = np.asarray(base_predictions, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != 2 or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()
            or np.any(np.abs(matrix) > FLOAT32_MAXIMUM)):
        raise ValueError("invalid_tail_correction_predictions")
    anchor = 0.8 * matrix[0] + (1.0 - 0.8) * matrix[1]
    difference = matrix[0] - matrix[1]
    design = np.column_stack((anchor, difference, np.abs(difference)))
    if not np.isfinite(design).all() or np.any(np.abs(design) > FLOAT32_MAXIMUM):
        raise ValueError("invalid_tail_correction_predictions")
    return anchor, design


def fit_state(
    base_predictions: Sequence[Sequence[float]], targets: Sequence[float],
) -> dict[str, Any]:
    """Fit only on supplied training-OOF predictions and unclipped training targets."""
    import numpy as np

    anchor, matrix = features(base_predictions)
    target = np.asarray(targets, dtype=np.float64)
    if (target.shape != anchor.shape or not np.isfinite(target).all()
            or np.any(np.abs(target) > FLOAT32_MAXIMUM)):
        raise ValueError("invalid_tail_correction_training_data")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    design = np.column_stack((np.ones(len(anchor)), np.clip((matrix - means) / scales, -1, 1)))
    coefficients = np.zeros(len(FEATURES) + 1)
    step = 1.0 / (2.0 * 5.1 * float(np.sum(design * design)) / len(anchor) + 0.02)
    for _ in range(2000):
        correction = design @ coefficients
        error = anchor + correction - target
        gradient = (design.T @ (0.2 * error + 8.0 * np.sign(error)
                    * np.maximum(np.abs(error) - 4.5, 0.0) + 2.0 * correction)
                    / len(anchor) + 0.02 * coefficients)
        proposed = coefficients - step * gradient
        if np.sum(np.abs(proposed)) > 2.0:
            ordered = np.sort(np.abs(proposed))[::-1]
            cumulative = np.cumsum(ordered)
            ranks = np.arange(1, len(ordered) + 1)
            rho = np.nonzero(ordered - (cumulative - 2.0) / ranks > 0)[0][-1]
            threshold = (cumulative[rho] - 2.0) / (rho + 1)
            proposed = np.sign(proposed) * np.maximum(np.abs(proposed) - threshold, 0.0)
            # Roundoff must not invalidate the global departure bound.
            proposed *= min(1.0, 2.0 / float(np.sum(np.abs(proposed))))
        coefficients = proposed
    state = {
        "contract": copy.deepcopy(CONTRACT), "feature_means": means.tolist(),
        "feature_scales": scales.tolist(), "coefficients": coefficients.tolist(),
    }
    if not valid_state(state):
        raise ValueError("invalid_tail_correction_state")
    return state


def valid_state(state: Mapping[str, Any]) -> bool:
    """Accept only finite state with the exact declared optimizer and global bound."""
    try:
        means, scales, coefficients = (
            state["feature_means"], state["feature_scales"], state["coefficients"]
        )
        return (
            set(state) == {"contract", "feature_means", "feature_scales", "coefficients"}
            and state["contract"] == CONTRACT
            and isinstance(means, list) and len(means) == 3
            and isinstance(scales, list) and len(scales) == 3
            and isinstance(coefficients, list) and len(coefficients) == 4
            and all(isinstance(v, (float, int)) and not isinstance(v, bool)
                    and math.isfinite(v) and abs(v) <= FLOAT32_MAXIMUM
                    for v in (*means, *scales, *coefficients))
            and all(v > 0 for v in scales)
            and math.fsum(abs(v) for v in coefficients) <= 2.0
        )
    except (KeyError, TypeError, ValueError):
        return False


def predict(
    state: Mapping[str, Any], base_predictions: Sequence[Sequence[float]], scale: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Preserve the continuous anchor with exact identity or a globally bounded correction."""
    import numpy as np

    if scale not in (0.0, 1.0) or not valid_state(state):
        raise ValueError("invalid_tail_correction_state")
    anchor, matrix = features(base_predictions)
    if scale == 0.0:
        return tuple(anchor.tolist()), (0.0,) * len(anchor)
    standardized = np.clip(
        (matrix - np.asarray(state["feature_means"])) / np.asarray(state["feature_scales"]), -1, 1
    )
    design = np.column_stack((np.ones(len(anchor)), standardized))
    correction = np.clip(design @ np.asarray(state["coefficients"]), -2.0, 2.0)
    prediction = anchor + correction
    if (not np.isfinite(prediction).all() or not np.isfinite(correction).all()
            or np.any(np.abs(prediction) > FLOAT32_MAXIMUM)):
        raise ValueError("invalid_tail_correction_predictions")
    return tuple(prediction.tolist()), tuple(correction.tolist())
