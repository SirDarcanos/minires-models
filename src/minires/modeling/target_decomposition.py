"""Bounding-box target decomposition with gram-valued reconstruction.

The modeled factor is derived from sliced resin mass labels. It is not physical
occupied-union volume and is deliberately not bounded to one.
"""
from __future__ import annotations

import math
from typing import Sequence

from ..ingestion import CanonicalRow
from .learned import _matrix
from .legacy import LEGACY_FEATURES

VERSION = "minires-bounding-box-target-decomposition-v1"
TARGET_UNIT = "bounding_box_occupancy_factor"
DENSITY_G_PER_ML = 1.1
DENSITY_G_PER_MM3 = DENSITY_G_PER_ML / 1000.0
BOUNDING_BOX_VOLUME_INDEX = LEGACY_FEATURES.index("bbox_area")


def occupancy_training_matrix(
    rows: Sequence[CanonicalRow],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Return legacy model inputs and direct unbounded occupancy-factor labels."""
    features, masses = _matrix(rows)
    if not rows or len(rows) != len(features):
        raise ValueError("invalid_target_decomposition_data")
    factors = []
    for row, values, mass in zip(rows, features, masses):
        density = row.metadata.get("resin_density_g_per_ml")
        bounding_box_volume = values[BOUNDING_BOX_VOLUME_INDEX]
        if (
            isinstance(density, bool)
            or not isinstance(density, (int, float))
            or not math.isfinite(float(density))
            or float(density) != DENSITY_G_PER_ML
            or not math.isfinite(bounding_box_volume)
            or bounding_box_volume <= 0.0
            or not math.isfinite(mass)
            or mass < 0.0
        ):
            raise ValueError("invalid_target_decomposition_data")
        factor = mass / (bounding_box_volume * DENSITY_G_PER_MM3)
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError("invalid_target_decomposition_data")
        factors.append(factor)
    return features, tuple(factors)


def reconstruct_mass_predictions(
    features: Sequence[tuple[float, ...]], factors: Sequence[float],
) -> tuple[float, ...]:
    """Reconstruct finite nonnegative sliced resin mass predictions in grams."""
    if not features or len(features) != len(factors):
        raise ValueError("invalid_target_decomposition_prediction")
    predictions = []
    for row, factor in zip(features, factors):
        try:
            bounding_box_volume = float(row[BOUNDING_BOX_VOLUME_INDEX])
            value = float(factor)
        except (IndexError, TypeError, ValueError, OverflowError):
            raise ValueError("invalid_target_decomposition_prediction") from None
        mass = value * bounding_box_volume * DENSITY_G_PER_MM3
        if (
            not math.isfinite(value) or value < 0.0
            or not math.isfinite(bounding_box_volume) or bounding_box_volume <= 0.0
            or not math.isfinite(mass) or mass < 0.0
        ):
            raise ValueError("invalid_target_decomposition_prediction")
        predictions.append(mass)
    return tuple(predictions)
