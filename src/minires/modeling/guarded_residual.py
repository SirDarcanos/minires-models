"""Pure numerical contract for guarded training-OOF residual correction."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


VERSION = "minires-guarded-residual-stacking-v1"
FEATURES = (
    "anchor_g",
    "base_1_minus_anchor_g",
    "base_2_minus_anchor_g",
    "base_3_minus_anchor_g",
    "base_4_minus_anchor_g",
    "spread_g",
)
RIDGE_PENALTY = 1.0
FEATURE_CLIP = 4.0
RESIDUAL_BOUND_G = 2.0
CORRECTION_SCALES = (0.0, 0.5, 1.0)
FLOAT32_MAXIMUM = 3.4028234663852886e38


def residual_features(
    base_predictions: Sequence[Sequence[float]], expected_rows: int,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return a fixed mean anchor and prediction-only residual features."""
    if len(base_predictions) != 4 or expected_rows <= 0:
        raise ValueError("invalid_guarded_residual_prediction_matrix")
    columns = tuple(tuple(float(value) for value in column) for column in base_predictions)
    if any(len(column) != expected_rows for column in columns) or not all(
        math.isfinite(value) and abs(value) <= FLOAT32_MAXIMUM
        for column in columns for value in column
    ):
        raise ValueError("invalid_guarded_residual_prediction_matrix")
    import numpy as np

    anchors: list[float] = []
    rows: list[tuple[float, ...]] = []
    for values in zip(*columns):
        anchor = math.fsum(values) / 4.0
        cast_anchor = float(np.float32(anchor))
        row = (
            cast_anchor,
            *(value - cast_anchor for value in values),
            max(values) - min(values),
        )
        if not all(math.isfinite(value) and abs(value) <= FLOAT32_MAXIMUM for value in row):
            raise ValueError("invalid_guarded_residual_prediction_matrix")
        cast = tuple(float(value) for value in np.asarray(row, dtype=np.float32))
        if not all(math.isfinite(value) for value in cast):
            raise ValueError("invalid_guarded_residual_prediction_matrix")
        anchors.append(cast_anchor)
        rows.append(cast)
    return tuple(anchors), tuple(rows)


def fit_state(
    features: Sequence[Sequence[float]], targets: Sequence[float], anchors: Sequence[float],
) -> dict[str, Any]:
    """Fit one deterministic bounded ridge residual state from training OOF rows."""
    import numpy as np

    matrix = np.asarray(features, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    anchor_values = np.asarray(anchors, dtype=np.float64)
    if (
        matrix.ndim != 2 or matrix.shape[1] != len(FEATURES) or matrix.shape[0] == 0
        or target_values.shape != (matrix.shape[0],)
        or anchor_values.shape != (matrix.shape[0],)
        or not np.isfinite(matrix).all() or not np.isfinite(target_values).all()
        or not np.isfinite(anchor_values).all()
    ):
        raise ValueError("invalid_guarded_residual_training_data")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    standardized = np.clip((matrix - means) / scales, -FEATURE_CLIP, FEATURE_CLIP)
    design = np.column_stack((np.ones(matrix.shape[0]), standardized))
    residual = np.clip(target_values - anchor_values, -RESIDUAL_BOUND_G, RESIDUAL_BOUND_G)
    penalty: Any = np.zeros(
        (len(FEATURES), len(FEATURES) + 1), dtype=np.float64
    )
    penalty[:, 1:] = math.sqrt(RIDGE_PENALTY * matrix.shape[0]) * np.eye(len(FEATURES))
    augmented_design = np.vstack((design, penalty))
    augmented_target = np.concatenate((residual, np.zeros(len(FEATURES))))
    coefficients, _, _, _ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)
    state = {
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_penalty": RIDGE_PENALTY,
        "feature_clip": FEATURE_CLIP,
        "residual_bound_g": RESIDUAL_BOUND_G,
    }
    if not valid_numeric_state(state):
        raise ValueError("invalid_guarded_residual_training_data")
    return state


def valid_numeric_state(state: Mapping[str, Any]) -> bool:
    """Validate the canonical analytical residual state."""
    expected = {
        "feature_means", "feature_scales", "coefficients", "ridge_penalty",
        "feature_clip", "residual_bound_g",
    }
    try:
        means = state["feature_means"]
        scales = state["feature_scales"]
        coefficients = state["coefficients"]
        values = (*means, *scales, *coefficients)
        return (
            set(state) == expected
            and isinstance(means, list) and len(means) == len(FEATURES)
            and isinstance(scales, list) and len(scales) == len(FEATURES)
            and isinstance(coefficients, list) and len(coefficients) == len(FEATURES) + 1
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and abs(float(value)) <= FLOAT32_MAXIMUM for value in values)
            and all(float(value) > 0.0 for value in scales)
            and state["ridge_penalty"] == RIDGE_PENALTY
            and state["feature_clip"] == FEATURE_CLIP
            and state["residual_bound_g"] == RESIDUAL_BOUND_G
        )
    except (KeyError, TypeError):
        return False


def predict(
    state: Mapping[str, Any], features: Sequence[Sequence[float]],
    anchors: Sequence[float], correction_scale: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Apply the exact identity or bounded additive residual correction."""
    import numpy as np

    if correction_scale not in CORRECTION_SCALES or not valid_numeric_state(state):
        raise ValueError("invalid_guarded_residual_state")
    matrix = np.asarray(features, dtype=np.float64)
    anchor_values = np.asarray(anchors, dtype=np.float64)
    if (
        matrix.ndim != 2 or matrix.shape[1] != len(FEATURES)
        or anchor_values.shape != (matrix.shape[0],)
        or not np.isfinite(matrix).all() or not np.isfinite(anchor_values).all()
        or np.any(np.abs(matrix) > FLOAT32_MAXIMUM)
        or np.any(np.abs(anchor_values) > FLOAT32_MAXIMUM)
    ):
        raise ValueError("invalid_guarded_residual_state")
    if correction_scale == 0.0:
        return tuple(float(value) for value in anchors), (0.0,) * len(anchors)
    means = np.asarray(state["feature_means"], dtype=np.float64)
    scales = np.asarray(state["feature_scales"], dtype=np.float64)
    coefficients = np.asarray(state["coefficients"], dtype=np.float64)
    standardized = np.clip((matrix - means) / scales, -FEATURE_CLIP, FEATURE_CLIP)
    design = np.column_stack((np.ones(matrix.shape[0]), standardized))
    raw = design @ coefficients
    corrections = correction_scale * np.clip(raw, -RESIDUAL_BOUND_G, RESIDUAL_BOUND_G)
    predictions = anchor_values + corrections
    if (
        not np.isfinite(raw).all()
        or not np.isfinite(corrections).all()
        or not np.isfinite(predictions).all()
        or np.any(np.abs(raw) > FLOAT32_MAXIMUM)
        or np.any(np.abs(corrections) > FLOAT32_MAXIMUM)
        or np.any(np.abs(predictions) > FLOAT32_MAXIMUM)
    ):
        raise ValueError("invalid_guarded_residual_state")
    return (
        tuple(float(value) for value in predictions),
        tuple(float(value) for value in corrections),
    )


def shift_summary(left: Sequence[float], right: Sequence[float]) -> dict[str, float | int]:
    """Return source-neutral aggregate shift evidence for paired predictions."""
    if len(left) != len(right) or not left:
        raise ValueError("invalid_guarded_residual_shift_evidence")
    differences = [float(r) - float(l) for l, r in zip(left, right)]
    if not all(math.isfinite(value) for value in (*left, *right, *differences)):
        raise ValueError("invalid_guarded_residual_shift_evidence")
    absolute = sorted(abs(value) for value in differences)

    def nearest_rank(fraction: float) -> float:
        return absolute[max(0, math.ceil(fraction * len(absolute)) - 1)]

    return {
        "count": len(differences),
        "mean_signed_shift_g": math.fsum(differences) / len(differences),
        "mean_absolute_shift_g": math.fsum(absolute) / len(absolute),
        "root_mean_square_shift_g": math.sqrt(
            math.fsum(value * value for value in differences) / len(differences)
        ),
        "median_absolute_shift_g": nearest_rank(0.5),
        "p95_absolute_shift_g": nearest_rank(0.95),
        "maximum_absolute_shift_g": absolute[-1],
    }
