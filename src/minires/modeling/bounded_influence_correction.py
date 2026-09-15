"""Versioned bounded-influence numerical correction; no production loader registration."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence, TypedDict, TypeGuard

from .tail_correction import FEATURES, FLOAT32_MAXIMUM, features

VERSION = "minires-bounded-influence-correction-v1"
CONTRACT: dict[str, Any] = {
    "version": VERSION,
    "features": list(FEATURES),
    "anchor": "float64_0.8_times_neural_plus_(1.0_minus_0.8)_times_xgboost",
    "feature_clip": 1.0,
    "scale_floor": 1e-12,
    "intercept": True,
    "coefficient_l1_bound_g": 2.0,
    "huber_definition": "H_d(z)=z^2_if_abs(z)<=d_else_2*d*abs(z)-d^2",
    "ordinary_huber_weight": 0.1,
    "ordinary_huber_delta_g": 5.0,
    "excess_huber_weight": 4.0,
    "excess_huber_delta_g": 2.5,
    "tail_margin_g": 4.5,
    "anchor_departure_weight": 1.0,
    "coefficient_squared_penalty": 0.01,
    "iterations": 2000,
    "optimizer": "zero_initialized_euclidean_l1_projected_gradient",
    "step": "1/(2*5.1*frobenius_design_squared/n+0.02)",
    "prediction_derivative_bound": 21.0,
    "target_residual_clipping": False,
    "row_weighting_or_exclusion": False,
    "correction_scales": [0.0, 1.0],
}


class BoundedInfluenceState(TypedDict):
    contract: dict[str, Any]
    feature_means: list[float]
    feature_scales: list[float]
    coefficients: list[float]


def _vector(values: Sequence[float]) -> Any:
    import numpy as np

    vector = np.asarray(values, dtype=np.float64)
    if (vector.ndim != 1 or not vector.size or not np.isfinite(vector).all()
            or np.any(np.abs(vector) > FLOAT32_MAXIMUM)):
        raise ValueError("invalid_bounded_influence_vector")
    return vector


def prediction_loss_components(errors: Sequence[float]) -> tuple[Any, Any]:
    """Weighted ordinary/excess Huber losses per row, excluding both fit penalties."""
    import numpy as np

    absolute = np.abs(_vector(errors))
    excess = np.maximum(absolute - 4.5, 0.0)
    # This equivalent form avoids squaring the unbounded part of a Huber loss.
    ordinary_core = np.minimum(absolute, 5.0)
    excess_core = np.minimum(excess, 2.5)
    return (0.1 * (ordinary_core ** 2 + 10.0 * (absolute - ordinary_core)),
            4.0 * (excess_core ** 2 + 5.0 * (excess - excess_core)))


def prediction_derivative(errors: Sequence[float]) -> Any:
    import numpy as np

    error = _vector(errors)
    return (0.2 * np.clip(error, -5.0, 5.0)
            + 8.0 * np.sign(error) * np.minimum(np.maximum(np.abs(error) - 4.5, 0.0), 2.5))


def _project(proposed: Any) -> Any:
    import numpy as np

    if np.sum(np.abs(proposed)) > 2.0:
        ordered = np.sort(np.abs(proposed))[::-1]
        cumulative = np.cumsum(ordered)
        ranks = np.arange(1, len(ordered) + 1)
        rho = np.nonzero(ordered - (cumulative - 2.0) / ranks > 0)[0][-1]
        threshold = (cumulative[rho] - 2.0) / (rho + 1)
        proposed = np.sign(proposed) * np.maximum(np.abs(proposed) - threshold, 0.0)
        proposed *= min(1.0, 2.0 / math.fsum(abs(float(v)) for v in proposed))
    return proposed


def fit_state(base_predictions: Sequence[Sequence[float]], targets: Sequence[float]) -> BoundedInfluenceState:
    """Fit exclusively on supplied training OOF predictions, never clip targets."""
    import numpy as np

    anchor, matrix = features(base_predictions)
    target = _vector(targets)
    if target.shape != anchor.shape:
        raise ValueError("invalid_bounded_influence_training_data")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    design = np.column_stack((np.ones(len(anchor)), np.clip((matrix - means) / scales, -1, 1)))
    coefficients = np.zeros(4)
    # Prediction curvature <= .2+8; departure adds 2. Thus Hessian norm is
    # <= 10.2*||D||_F^2/n + .02 (ridge), even across the C1 Huber boundaries.
    step = 1.0 / (2.0 * 5.1 * float(np.sum(design * design)) / len(anchor) + 0.02)
    for _ in range(2000):
        correction = design @ coefficients
        error = anchor + correction - target
        gradient = (design.T @ (prediction_derivative(error) + 2.0 * correction)
                    / len(anchor) + 0.02 * coefficients)
        coefficients = _project(coefficients - step * gradient)
    state: BoundedInfluenceState = {
        "contract": copy.deepcopy(CONTRACT), "feature_means": means.tolist(),
        "feature_scales": scales.tolist(), "coefficients": coefficients.tolist(),
    }
    if not valid_state(state):
        raise ValueError("invalid_bounded_influence_state")
    return state


def _same_contract(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(_same_contract(value[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(_same_contract(v, e) for v, e in zip(value, expected))
    return bool(value == expected)


def valid_state(state: object) -> TypeGuard[BoundedInfluenceState]:
    """Fail closed, including contract type tampering (bool is not numeric state)."""
    try:
        if not isinstance(state, Mapping) or set(state) != {
            "contract", "feature_means", "feature_scales", "coefficients",
        }:
            return False
        if not _same_contract(state["contract"], CONTRACT):
            return False
        means, scales, coefficients = state["feature_means"], state["feature_scales"], state["coefficients"]
        return (all(isinstance(v, list) and len(v) == n
                    for v, n in ((means, 3), (scales, 3), (coefficients, 4)))
                and all(type(v) in (float, int) and math.isfinite(v) and abs(v) <= FLOAT32_MAXIMUM
                        for v in (*means, *scales, *coefficients))
                and all(v > 0 for v in scales)
                and math.fsum(abs(v) for v in coefficients) <= 2.0)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def predict(state: object, base_predictions: Sequence[Sequence[float]], scale: float
            ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    import numpy as np

    if type(scale) not in (int, float) or scale not in (0.0, 1.0) or not valid_state(state):
        raise ValueError("invalid_bounded_influence_state")
    anchor, matrix = features(base_predictions)
    if scale == 0.0:
        return tuple(anchor.tolist()), (0.0,) * len(anchor)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            standardized = np.clip((matrix - np.asarray(state["feature_means"]))
                                   / np.asarray(state["feature_scales"]), -1, 1)
            design = np.column_stack((np.ones(len(anchor)), standardized))
            correction = np.clip(design @ np.asarray(state["coefficients"]), -2.0, 2.0)
            prediction = _vector(anchor + correction)
    except FloatingPointError:
        raise ValueError("invalid_bounded_influence_predictions") from None
    return tuple(prediction.tolist()), tuple(correction.tolist())
