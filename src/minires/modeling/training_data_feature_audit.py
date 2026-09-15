"""Bounded training-only audit of data contracts, targets, and base error regimes.

The public seam deliberately has no validation or held-out-test input.  It retains
all accepted training rows, fits only the two frozen anchor components, and emits
aggregate private evidence rather than row-level values or model artifacts.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Callable, Mapping, Sequence

from ..evaluation import EvaluationConfig
from ..ingestion import CanonicalRow, Dataset, InputError, fingerprint, load_records, normalize, number
from ..private_io import PrivateArgumentParser, write_private_json
from ..source_identity import code_fingerprint
from . import tail_correction
from . import tail_focused_search as tail
from . import training_stability as stability
from . import tuning as t
from .definitions import validate_component_preprocessing

VERSION = "minires-training-data-feature-audit-v1"
PREDECLARED_OUTPUT_ROOT = stability.PROJECT_ROOT / "private" / "candidate-tuning" / "run-019"
MAXIMUM_MODEL_FITS = 8
MAXIMUM_ELAPSED_SECONDS = 7200.0
MODEL_SEEDS = stability.MODEL_SEEDS
OUTER_SPLIT_SEEDS = stability.OUTER_SPLIT_SEEDS
FOLDS = stability.FOLDS
SERIOUS_ERROR_G = 5.0
RECONCILIATION_REL_TOLERANCE = 1e-9
RECONCILIATION_ABS_TOLERANCE = 1e-9
SIMILAR_DISTANCE_MAXIMUM = 0.10
MINIMUM_SIMILAR_RELATIONSHIPS = 100
MINIMUM_EXACT_GROUP_ROWS = 20
HETEROGENEOUS_TARGET_DELTA_G = 5.0
HETEROGENEOUS_FRACTION_MINIMUM = 0.10
REGIME_QUANTILES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
MINIMUM_REGIME_SUPPORT = 100
REGIME_ENRICHMENT_RATIO = 2.0
REGIME_MINIMUM_ABSOLUTE_EXCESS = 0.01
EXPECTED_DENSITY_G_PER_ML = 1.1
EXPECTED_LAYER_HEIGHT_MM = 0.05
EXPECTED_SLICER_ADDED_SUPPORTS = False

CANONICAL_ALIAS_PAIRS = (
    ("file_size_kib", "kb", "file_size"),
    ("volume_mm3", "volume", "volume"),
    ("surface_area_mm2", "surface_area", "surface_area"),
    ("bounding_box_x_mm", "bbox_x", "bounding_box_x"),
    ("bounding_box_y_mm", "bbox_y", "bounding_box_y"),
    ("bounding_box_z_mm", "bbox_z", "bounding_box_z"),
    ("bounding_box_volume_mm3", "bbox_area", "bounding_box_volume"),
    ("mesh_mass_at_unit_density", "mass", "mesh_mass_at_unit_density"),
    ("euler_characteristic", "euler_number", "euler"),
    ("mesh_scale_mm", "scale", "mesh_scale"),
    ("surface_to_volume_ratio_per_mm", "surface_volume_ratio", "surface_volume_ratio"),
    ("sliced_resin_mass_g", "weight", "sliced_resin_mass"),
)
CANONICAL_GEOMETRY_FIELDS = (
    "volume_mm3", "surface_area_mm2", "bounding_box_x_mm", "bounding_box_y_mm",
    "bounding_box_z_mm", "bounding_box_volume_mm3", "euler_number",
)
REGIME_VIEWS = ("mesh_volume", "surface_compactness", "bounding_box_fill", "aspect_ratio")
EULER_BINS = ("negative", "zero", "positive")
QUANTILE_BINS = ("q1", "q2", "q3", "q4", "q5")

DECISION_RULES = {
    "priority": ["investigate_data_contracts", "obtain_better_inputs", "improve_base_predictor", "inconclusive_collect_evidence"],
    "investigate_data_contracts": "any_explicit_alias_geometry_or_slicing_contract_contradiction_or_invalid_explicit_value",
    "obtain_better_inputs": (
        "no_data_contract_trigger_and_either_exact_or_similar_geometry_target_heterogeneity_trigger; "
        "or_missing_required_slicing_attributes_have_support_at_least_100_and_strict_above5g_rate_at_least_2x_overall_with_at_least_0.01_absolute_excess_in_all_four_cells"
    ),
    "improve_base_predictor": (
        "neither_prior_route_triggers_and_the_same_predeclared_geometry_view_bin_has_support_at_least_100_and_strict_above5g_rate_at_least_2x_overall_with_at_least_0.01_absolute_excess_in_all_four_cells"
    ),
    "otherwise": "inconclusive_collect_evidence; no_model_run_warranted",
}


@dataclass(frozen=True)
class TrainingDataFeatureAuditResult:
    status: str
    blockers: tuple[str, ...]
    evidence: Mapping[str, Any]
    resource_use: Mapping[str, Any]


def _plan(raw: str, normalized: str, dependencies: Mapping[str, str]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "kind": "training_only_data_target_and_frozen_base_feature_audit",
        "questions": [
            "do explicit unit geometry or slicing declarations contradict one another",
            "do equal or predeclared-similar canonical geometries have materially different sliced resin mass targets",
            "does the frozen base anchor show stable supported severe-error enrichment in a predeclared geometry regime",
        ],
        "what_no_fitting_can_answer": "explicit contract reconciliation slicing-attribute coverage and target heterogeneity among exact or deterministically similar canonical geometries",
        "what_requires_fitting": "honest strict_above5g attribution for the frozen base anchor on outer-held training rows",
        "input_fingerprint": raw,
        "normalized_input_fingerprint": normalized,
        "code_fingerprint": code_fingerprint(),
        "dependency_versions": dict(dependencies),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "required_environment": {
            "python": "3.13",
            **t.CROSS_FITTED_GATE_DEPENDENCY_VERSIONS,
            "scikit-learn": t.CROSS_FITTED_GATE_SCIKIT_LEARN_VERSION,
        },
        "expected_training_sha256": t.GUARDED_RESIDUAL_DEVELOPMENT_CHECKSUMS["train.jsonl"],
        "normalization": {"seed": stability.NORMALIZATION_SEED, "volume_unit": "mm3", "scope_confirmed": True, "contract": "legacy"},
        "committed_provenance_limit": "historical_measurements_were_reused_as_authoritative_without_mesh_probing_or_slicing; harmonized rows_do_not_identify_origin; per_row_historical_conditions_cannot_be_verified",
        "contract_audit": {
            "canonical_alias_pairs": [list(pair[:2]) for pair in CANONICAL_ALIAS_PAIRS],
            "numeric_reconciliation": {"relative_tolerance": RECONCILIATION_REL_TOLERANCE, "absolute_tolerance": RECONCILIATION_ABS_TOLERANCE},
            "geometry_identity": "bounding_box_volume_equals_x_times_y_times_z; mesh_volume_does_not_exceed_bounding_box_volume; surface_ratio_equals_surface_area_divided_by_volume",
            "required_slicing_attributes": {
                "volume_unit": "mm3",
                "resin_density_g_per_ml": EXPECTED_DENSITY_G_PER_ML,
                "layer_height_mm": EXPECTED_LAYER_HEIGHT_MM,
                "slicer_added_supports": EXPECTED_SLICER_ADDED_SUPPORTS,
                "scope_confirmed": True,
            },
            "missing_unknown_is_not_contradiction": True,
            "target_to_mesh_volume_ratio_interpretation": "not_a_defect_test",
            "row_policy": "preserve_every_row_no_repair_removal_imputation_or_replacement",
        },
        "target_similarity": {
            "ordered_canonical_geometry": list(CANONICAL_GEOMETRY_FIELDS),
            "transforms": "log_positive_geometry_and_signed_log1p_absolute_euler",
            "robust_scale": "median_center_and_max(1.4826*MAD,IQR/1.349,1e-12); zero_scale_becomes_1",
            "nearest_relation": "each_row_to_lowest_index_nearest_nonidentical_geometry_by_rms_robust_standardized_distance",
            "similar_distance_maximum": SIMILAR_DISTANCE_MAXIMUM,
            "exact_group_minimum_total_rows": MINIMUM_EXACT_GROUP_ROWS,
            "similar_relation_minimum_count": MINIMUM_SIMILAR_RELATIONSHIPS,
            "material_target_delta_g": HETEROGENEOUS_TARGET_DELTA_G,
            "material_fraction_minimum": HETEROGENEOUS_FRACTION_MINIMUM,
            "all_relationships_reported": True,
        },
        "frozen_anchor": {
            "formula": "float64_0.8_times_neural_plus_(1.0_minus_0.8)_times_xgboost",
            "base_candidates": [asdict(candidate) for candidate in tail.base_candidates()],
            "fixed_training_counts": dict(tail.FIXED_COUNTS),
            "outer_split_seeds": list(OUTER_SPLIT_SEEDS),
            "model_seeds": list(MODEL_SEEDS),
            "fold_assignment": "stable-record-identity-sha256-round-robin-v1",
            "outer_holdout": "rank_modulo_5_equals_zero",
            "historical_selection": "conditional_on_reused_validation_selected_pair_and_counts",
        },
        "geometry_regimes": {
            "quantile_views": list(REGIME_VIEWS),
            "quantiles": list(REGIME_QUANTILES),
            "quantile_edges": "linear_quantiles_from_each_cells_outer_fitting_partition_only",
            "euler_bins": list(EULER_BINS),
            "support": MINIMUM_REGIME_SUPPORT,
            "enrichment_ratio": REGIME_ENRICHMENT_RATIO,
            "minimum_absolute_rate_excess": REGIME_MINIMUM_ABSOLUTE_EXCESS,
            "stability": "same_view_and_relative_bin_must_meet_all_conditions_in_all_four_cells",
            "multiplicity": "report_all_23_view_bins; descriptive_screen_only; no_p_values_or_selected_thresholds",
        },
        "decision_rules": copy.deepcopy(DECISION_RULES),
        "fit_allocation": {"outer_partition_neural_network": 4, "outer_partition_xgboost": 4},
        "maximum_fits": MAXIMUM_MODEL_FITS,
        "maximum_elapsed_seconds": MAXIMUM_ELAPSED_SECONDS,
        "validation_input": "unavailable_to_this_interface",
        "held_out_test_input": "unavailable_to_this_interface",
        "source_group_use": "none_in_fitting_analysis_or_output",
        "selection_lock_promotion": "forbidden_audit_only",
        "stop_rule": "one_attempt_stops_regardless_of_outcome_without_retry_recycling_or_continuation",
    }


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=RECONCILIATION_REL_TOLERANCE, abs_tol=RECONCILIATION_ABS_TOLERANCE)


def _contract_target_audit(raw_rows: Sequence[Mapping[str, Any]], rows: Sequence[CanonicalRow]) -> dict[str, Any]:
    alias: dict[str, dict[str, int]] = {}
    explicit_contradictions = 0
    for canonical, legacy, label in CANONICAL_ALIAS_PAIRS:
        counts = {"both_missing": 0, "one_missing": 0, "matching": 0, "invalid_explicit": 0, "contradictory": 0}
        for raw in raw_rows:
            left_raw, right_raw = raw.get(canonical), raw.get(legacy)
            if left_raw is None and right_raw is None:
                counts["both_missing"] += 1
                continue
            if left_raw is None or right_raw is None:
                counts["one_missing"] += 1
                continue
            left, right = number(left_raw), number(right_raw)
            if left is None or right is None or not math.isfinite(left) or not math.isfinite(right):
                counts["invalid_explicit"] += 1
                explicit_contradictions += 1
            elif _close(left, right):
                counts["matching"] += 1
            else:
                counts["contradictory"] += 1
                explicit_contradictions += 1
        alias[label] = counts

    geometry = {
        "bounding_box_product": {"unavailable": 0, "matching": 0, "invalid_explicit": 0, "contradictory": 0},
        "mesh_within_bounding_box": {"unavailable": 0, "matching": 0, "invalid_explicit": 0, "contradictory": 0},
        "surface_volume_ratio": {"unavailable": 0, "matching": 0, "invalid_explicit": 0, "contradictory": 0},
    }
    for raw in raw_rows:
        dimensions = [number(raw.get(name)) for name in ("bounding_box_x_mm", "bounding_box_y_mm", "bounding_box_z_mm")]
        bbox = number(raw.get("bounding_box_volume_mm3"))
        if bbox is None or any(value is None for value in dimensions):
            geometry["bounding_box_product"]["unavailable"] += 1
        else:
            bbox_value = float(bbox)
            dimension_values = [float(value) for value in dimensions if value is not None]
            if not all(math.isfinite(value) and value > 0 for value in [bbox_value, *dimension_values]):
                geometry["bounding_box_product"]["invalid_explicit"] += 1
                explicit_contradictions += 1
            elif _close(bbox_value, math.prod(dimension_values)):
                geometry["bounding_box_product"]["matching"] += 1
            else:
                geometry["bounding_box_product"]["contradictory"] += 1
                explicit_contradictions += 1
        mesh_volume = number(raw.get("volume_mm3"))
        if mesh_volume is None or bbox is None:
            geometry["mesh_within_bounding_box"]["unavailable"] += 1
        elif not all(math.isfinite(value) and value > 0 for value in (mesh_volume, bbox)):
            geometry["mesh_within_bounding_box"]["invalid_explicit"] += 1
            explicit_contradictions += 1
        elif mesh_volume <= bbox or _close(mesh_volume, bbox):
            geometry["mesh_within_bounding_box"]["matching"] += 1
        else:
            geometry["mesh_within_bounding_box"]["contradictory"] += 1
            explicit_contradictions += 1
        surface, volume = number(raw.get("surface_area_mm2")), mesh_volume
        ratio = number(raw.get("surface_to_volume_ratio_per_mm"))
        if surface is None or volume is None or ratio is None:
            geometry["surface_volume_ratio"]["unavailable"] += 1
        elif not all(math.isfinite(float(value)) for value in (surface, volume, ratio)) or float(volume) <= 0:
            geometry["surface_volume_ratio"]["invalid_explicit"] += 1
            explicit_contradictions += 1
        elif _close(float(ratio), float(surface) / float(volume)):
            geometry["surface_volume_ratio"]["matching"] += 1
        else:
            geometry["surface_volume_ratio"]["contradictory"] += 1
            explicit_contradictions += 1

    slicing: dict[str, dict[str, int]] = {
        key: {"missing_unknown": 0, "matching": 0, "invalid_explicit": 0, "contradictory": 0}
        for key in (
            "slicing_conditions", "volume_unit", "resin_density_g_per_ml",
            "layer_height_mm", "slicer_added_supports", "scope_confirmed",
        )
    }
    missing_masks: list[bool] = []
    for raw in raw_rows:
        raw_conditions = raw.get("slicing_conditions")
        conditions: Mapping[str, Any] = {}
        malformed_conditions = False
        if raw_conditions is None:
            slicing["slicing_conditions"]["missing_unknown"] += 1
        elif isinstance(raw_conditions, Mapping):
            slicing["slicing_conditions"]["matching"] += 1
            conditions = raw_conditions
        elif isinstance(raw_conditions, str):
            try:
                decoded_conditions = json.loads(raw_conditions)
            except json.JSONDecodeError:
                decoded_conditions = None
            if isinstance(decoded_conditions, Mapping):
                slicing["slicing_conditions"]["matching"] += 1
                conditions = decoded_conditions
            else:
                malformed_conditions = True
        else:
            malformed_conditions = True
        if malformed_conditions:
            slicing["slicing_conditions"]["invalid_explicit"] += 1
            explicit_contradictions += 1
        values = {
            "volume_unit": raw.get("volume_unit"),
            "resin_density_g_per_ml": raw.get("resin_density_g_per_ml"),
            "layer_height_mm": conditions.get("layer_height_mm"),
            "slicer_added_supports": conditions.get("slicer_added_supports"),
            "scope_confirmed": raw.get("scope_confirmed"),
        }
        missing = raw_conditions is None or malformed_conditions
        for key, expected in (
            ("volume_unit", "mm3"),
            ("resin_density_g_per_ml", EXPECTED_DENSITY_G_PER_ML),
            ("layer_height_mm", EXPECTED_LAYER_HEIGHT_MM),
            ("slicer_added_supports", EXPECTED_SLICER_ADDED_SUPPORTS),
            ("scope_confirmed", True),
        ):
            raw_value = values[key]
            if raw_value is None:
                slicing[key]["missing_unknown"] += 1
                missing = True
            elif isinstance(expected, str):
                if not isinstance(raw_value, str) or not raw_value:
                    slicing[key]["invalid_explicit"] += 1
                    explicit_contradictions += 1
                elif raw_value == expected:
                    slicing[key]["matching"] += 1
                else:
                    slicing[key]["contradictory"] += 1
                    explicit_contradictions += 1
            elif isinstance(expected, bool):
                if not isinstance(raw_value, bool):
                    slicing[key]["invalid_explicit"] += 1
                    explicit_contradictions += 1
                elif raw_value is expected:
                    slicing[key]["matching"] += 1
                else:
                    slicing[key]["contradictory"] += 1
                    explicit_contradictions += 1
            else:
                value = number(raw_value)
                if value is None or not math.isfinite(value) or value <= 0:
                    slicing[key]["invalid_explicit"] += 1
                    explicit_contradictions += 1
                elif _close(value, expected):
                    slicing[key]["matching"] += 1
                else:
                    slicing[key]["contradictory"] += 1
                    explicit_contradictions += 1
        missing_masks.append(missing)

    targets = [_required_target(row) for row in rows]
    geometries = [tuple(_required_feature(row, name) for name in CANONICAL_GEOMETRY_FIELDS) for row in rows]
    exact: dict[tuple[float, ...], list[float]] = {}
    for geometry_key, target in zip(geometries, targets):
        exact.setdefault(geometry_key, []).append(target)
    repeated = [values for values in exact.values() if len(values) > 1]
    repeated_rows = sum(len(values) for values in repeated)
    heterogeneous = [values for values in repeated if max(values) - min(values) > HETEROGENEOUS_TARGET_DELTA_G]
    exact_summary: dict[str, Any] = {
        "repeated_group_count": len(repeated),
        "repeated_group_rows": repeated_rows,
        "heterogeneous_group_count": len(heterogeneous),
        "rows_in_heterogeneous_groups": sum(len(values) for values in heterogeneous),
        "support_sufficient": repeated_rows >= MINIMUM_EXACT_GROUP_ROWS,
    }
    exact_summary["heterogeneous_row_fraction"] = (
        exact_summary["rows_in_heterogeneous_groups"] / repeated_rows if repeated_rows else 0.0
    )
    exact_summary["trigger"] = bool(
        exact_summary["support_sufficient"]
        and exact_summary["heterogeneous_row_fraction"] >= HETEROGENEOUS_FRACTION_MINIMUM
    )
    similar_summary = _nearest_target_summary(geometries, targets)
    return {
        "record_count": len(rows),
        "alias_reconciliation": alias,
        "geometry_reconciliation": geometry,
        "slicing_attribute_coverage": slicing,
        "per_row_measurement_origin_available": False,
        "historical_measurement_contract_status": "unknown_not_a_contradiction",
        "target_to_mesh_volume_ratio_used_as_defect_test": False,
        "explicit_contradiction_count": explicit_contradictions,
        "data_contract_trigger": explicit_contradictions > 0,
        "target_similarity": {"exact_geometry": exact_summary, "nearest_geometry": similar_summary},
        "target_heterogeneity_trigger": bool(exact_summary["trigger"] or similar_summary["trigger"]),
        "missing_required_slicing_attribute_count": sum(missing_masks),
        "missing_required_slicing_attribute_mask": missing_masks,
    }


def _nearest_target_summary(geometries: Sequence[tuple[float, ...]], targets: Sequence[float]) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(geometries, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(targets) or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("invalid_similarity_geometry")
    transformed = np.empty_like(values)
    transformed[:, :6] = np.log(values[:, :6])
    transformed[:, 6] = np.sign(values[:, 6]) * np.log1p(np.abs(values[:, 6]))
    median = np.median(transformed, axis=0)
    mad = np.median(np.abs(transformed - median), axis=0) * 1.4826
    q1, q3 = np.quantile(transformed, (0.25, 0.75), axis=0, method="linear")
    scales = np.maximum(mad, (q3 - q1) / 1.349)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    standardized = (transformed - median) / scales
    norms = np.sum(standardized * standardized, axis=1)
    nearest = np.full(len(values), -1, dtype=int)
    nearest_distance = np.full(len(values), np.inf)
    block = 256
    for start in range(0, len(values), block):
        stop = min(start + block, len(values))
        distances = norms[start:stop, None] + norms[None, :] - 2.0 * standardized[start:stop] @ standardized.T
        distances = np.maximum(distances, 0.0)
        for local, index in enumerate(range(start, stop)):
            distances[local, index] = np.inf
            distances[local, distances[local] <= 1e-24] = np.inf
        choices = np.argmin(distances, axis=1)
        selected = distances[np.arange(stop - start), choices]
        finite = np.isfinite(selected)
        nearest[start:stop][finite] = choices[finite]
        nearest_distance[start:stop][finite] = np.sqrt(selected[finite] / values.shape[1])
    similar = nearest_distance <= SIMILAR_DISTANCE_MAXIMUM
    deltas = np.asarray([abs(float(targets[i]) - float(targets[j])) if j >= 0 else math.inf
                         for i, j in enumerate(nearest)], dtype=np.float64)
    count = int(np.sum(similar))
    heterogeneous = int(np.sum(similar & (deltas > HETEROGENEOUS_TARGET_DELTA_G)))
    bins = {
        "at_most_2g": int(np.sum(similar & (deltas <= 2.0))),
        "above_2g_at_most_5g": int(np.sum(similar & (deltas > 2.0) & (deltas <= 5.0))),
        "above_5g": heterogeneous,
    }
    fraction = heterogeneous / count if count else 0.0
    return {
        "nonidentical_nearest_relationship_count": int(np.sum(nearest >= 0)),
        "similar_relationship_count": count,
        "target_delta_bins": bins,
        "heterogeneous_fraction": fraction,
        "support_sufficient": count >= MINIMUM_SIMILAR_RELATIONSHIPS,
        "trigger": bool(count >= MINIMUM_SIMILAR_RELATIONSHIPS and fraction >= HETEROGENEOUS_FRACTION_MINIMUM),
    }


def _validate_frozen_fit_data(rows: Sequence[CanonicalRow]) -> None:
    for base in tail.base_candidates():
        features, targets = t.candidate_prediction_matrix(rows, base)
        if (
            len(features) != len(rows)
            or len(targets) != len(rows)
            or not t._valid_float32_matrix(features)
            or not all(
                math.isfinite(value) and abs(value) <= t.FLOAT32_MAXIMUM
                for value in targets
            )
        ):
            raise ValueError("invalid_frozen_fit_data")


def _required_feature(row: CanonicalRow, name: str) -> float:
    value = row.features.get(name)
    if value is None or not math.isfinite(value):
        raise ValueError("invalid_training_geometry")
    return float(value)


def _required_target(row: CanonicalRow) -> float:
    value = row.sliced_resin_mass_g
    if value is None or not math.isfinite(value):
        raise ValueError("invalid_training_target")
    return float(value)


def _regime_values(rows: Sequence[CanonicalRow]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {key: [] for key in (*REGIME_VIEWS, "euler")}
    for row in rows:
        volume = _required_feature(row, "volume_mm3")
        surface = _required_feature(row, "surface_area_mm2")
        dimensions = [_required_feature(row, name) for name in ("bounding_box_x_mm", "bounding_box_y_mm", "bounding_box_z_mm")]
        bbox = _required_feature(row, "bounding_box_volume_mm3")
        result["mesh_volume"].append(math.log(volume))
        result["surface_compactness"].append(surface / volume ** (2.0 / 3.0))
        result["bounding_box_fill"].append(volume / bbox)
        result["aspect_ratio"].append(max(dimensions) / min(dimensions))
        result["euler"].append(_required_feature(row, "euler_number"))
    if any(not math.isfinite(value) for values in result.values() for value in values):
        raise ValueError("invalid_geometry_regime")
    return result


def _summarize_cell_regimes(fitting: Sequence[CanonicalRow], held: Sequence[CanonicalRow], targets: Sequence[float], predictions: Sequence[float], missing: Sequence[bool]) -> dict[str, Any]:
    import numpy as np

    if not held or len(held) != len(targets) or len(held) != len(predictions) or len(held) != len(missing):
        raise ValueError("invalid_audit_cell")
    errors = np.abs(np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64))
    if not np.isfinite(errors).all():
        raise ValueError("invalid_audit_predictions")
    serious = errors > SERIOUS_ERROR_G
    overall_rate = float(np.mean(serious))
    fit_values, held_values = _regime_values(fitting), _regime_values(held)
    views: dict[str, Any] = {}
    for view in REGIME_VIEWS:
        edges = np.quantile(np.asarray(fit_values[view]), REGIME_QUANTILES, method="linear")
        assignments = np.searchsorted(edges[1:-1], np.asarray(held_values[view]), side="right")
        groups: dict[str, Any] = {}
        for index, label in enumerate(QUANTILE_BINS):
            mask = assignments == index
            groups[label] = _error_group(mask, errors, serious, overall_rate)
        views[view] = {"fitting_quantile_edges": edges.tolist(), "bins": groups}
    euler = np.asarray(held_values["euler"])
    views["euler"] = {"bins": {
        "negative": _error_group(euler < 0, errors, serious, overall_rate),
        "zero": _error_group(euler == 0, errors, serious, overall_rate),
        "positive": _error_group(euler > 0, errors, serious, overall_rate),
    }}
    missing_array = np.asarray(missing, dtype=bool)
    return {
        "overall": {"count": len(errors), "mae_g": float(np.mean(errors)), "above_5g_count": int(np.sum(serious)), "above_5g_rate": overall_rate},
        "views": views,
        "missing_required_slicing_attributes": _error_group(missing_array, errors, serious, overall_rate),
    }


def _error_group(mask: Any, errors: Any, serious: Any, overall_rate: float) -> dict[str, Any]:
    import numpy as np

    count = int(np.sum(mask))
    serious_count = int(np.sum(serious[mask]))
    rate = serious_count / count if count else 0.0
    ratio = rate / overall_rate if overall_rate > 0 else (math.inf if rate > 0 else 0.0)
    supported = count >= MINIMUM_REGIME_SUPPORT
    enriched = bool(supported and rate - overall_rate >= REGIME_MINIMUM_ABSOLUTE_EXCESS and ratio >= REGIME_ENRICHMENT_RATIO)
    return {
        "count": count,
        "mae_contribution_g": float(np.sum(errors[mask]) / len(errors)),
        "above_5g_count": serious_count,
        "above_5g_rate": rate,
        "rate_ratio_to_cell": ratio,
        "absolute_rate_excess": rate - overall_rate,
        "support_sufficient": supported,
        "enriched": enriched,
    }


def _stable_enriched_regimes(cells: Mapping[str, Mapping[str, Any]]) -> list[str]:
    candidates = [f"{view}:{label}" for view in REGIME_VIEWS for label in QUANTILE_BINS]
    candidates += [f"euler:{label}" for label in EULER_BINS]
    return [candidate for candidate in candidates if all(
        cell["error_regimes"]["views"][candidate.split(":")[0]]["bins"][candidate.split(":")[1]]["enriched"]
        for cell in cells.values()
    )]


def _decision(contract: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stable_regimes = _stable_enriched_regimes(cells) if len(cells) == 4 else []
    missing_enriched = bool(len(cells) == 4 and all(
        cell["error_regimes"]["missing_required_slicing_attributes"]["enriched"] for cell in cells.values()
    ))
    if contract["data_contract_trigger"]:
        route = "investigate_data_contracts"
    elif contract["target_heterogeneity_trigger"] or missing_enriched:
        route = "obtain_better_inputs"
    elif stable_regimes:
        route = "improve_base_predictor"
    else:
        route = "inconclusive_collect_evidence"
    return {
        "route": route,
        "priority_applied": list(DECISION_RULES["priority"]),
        "data_contract_trigger": bool(contract["data_contract_trigger"]),
        "target_heterogeneity_trigger": bool(contract["target_heterogeneity_trigger"]),
        "missing_slicing_attributes_stably_error_enriched": missing_enriched,
        "stable_supported_enriched_geometry_regimes": stable_regimes,
        "candidate_or_model_run_authorized": False,
    }


def run_training_data_feature_audit(
    training_records: Dataset,
    config: EvaluationConfig,
    *,
    runtime: t.CandidateRuntime,
    output_root: str | Path,
    clock: Callable[[], float] = time.monotonic,
) -> TrainingDataFeatureAuditResult:
    """Run one create-only audit and stop; no result selects or changes a model."""
    if (config.seed != stability.NORMALIZATION_SEED or config.volume_unit != "mm3" or config.scope_confirmed is not True):
        raise InputError("invalid_training_data_feature_audit_configuration")
    output = Path(output_root)
    if "private" not in output.resolve().parts:
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None

    started = clock()
    cpu_started = time.process_time()
    fit_count = 0
    cells: dict[str, Any] = {}
    active_cell: str | None = None
    blockers: tuple[str, ...] = ()
    status = "blocked"
    plan = _plan("unavailable", "unavailable", runtime.dependency_versions)
    contract: dict[str, Any] = {}

    def check_deadline() -> None:
        now = clock()
        if not math.isfinite(started) or not math.isfinite(now) or now < started:
            raise RuntimeError("training_data_feature_audit_invalid_clock")
        if now - started >= MAXIMUM_ELAPSED_SECONDS:
            raise RuntimeError("training_data_feature_audit_deadline_reached")

    def before_fit() -> None:
        nonlocal fit_count
        check_deadline()
        if fit_count >= MAXIMUM_MODEL_FITS:
            raise RuntimeError("training_data_feature_audit_fit_limit_reached")
        fit_count += 1

    try:
        check_deadline()
        loaded, raw_fingerprint = load_records(training_records)
        if not loaded or any(not isinstance(item, Mapping) for item in loaded):
            raise ValueError("invalid_training_records")
        raw_rows = [item for item in loaded if isinstance(item, Mapping)]
        training = normalize(raw_rows, config, contract="legacy")
        identities = [str(row.metadata.get("record_identity", "")) for row in training]
        geometry_names = CANONICAL_GEOMETRY_FIELDS
        if (not t._valid_candidate_feature_data(training, t.LEGACY_FEATURES)
                or any(row.outcome != "included" or not identity for row, identity in zip(training, identities))
                or len(identities) != len(set(identities))
                or any(row.features[name] is None for row in training for name in geometry_names)):
            raise ValueError("invalid_training_records")
        plan = _plan(raw_fingerprint, fingerprint([asdict(row) for row in training]), runtime.dependency_versions)
        write_private_json(output / "audit-plan.json", plan)
        _validate_frozen_fit_data(training)
        contract = _contract_target_audit(raw_rows, training)
        missing_all = contract.pop("missing_required_slicing_attribute_mask")
        for split_seed in OUTER_SPLIT_SEEDS:
            outer = t._cross_fit_assignments(training, split_seed, FOLDS)
            fitting = [row for row, fold in zip(training, outer) if fold != 0]
            held = [row for row, fold in zip(training, outer) if fold == 0]
            held_missing = [value for value, fold in zip(missing_all, outer) if fold == 0]
            if not fitting or not held:
                raise ValueError("invalid_training_partition")
            _, targets = t.candidate_prediction_matrix(held, tail.base_candidates()[0])
            for model_seed in MODEL_SEEDS:
                active_cell = stability._cell_key(split_seed, model_seed)
                columns: list[tuple[float, ...]] = []
                first_fit = fit_count
                for base, count in zip(tail.base_candidates(), (87, 1091)):
                    fit_x, fit_y = t.candidate_prediction_matrix(fitting, base)
                    held_x, _ = t.candidate_prediction_matrix(held, base)
                    before_fit()
                    fitted = t._stack_refit(runtime, base, model_seed, fit_x, fit_y, count)
                    check_deadline()
                    validate_component_preprocessing(t._locked_model_specification(base, tail.FIXED_COUNTS), fitted.preprocessing_state)
                    columns.append(t._predict(fitted.predictor, held_x))
                    check_deadline()
                anchor, _ = tail_correction.features(columns)
                error_regimes = _summarize_cell_regimes(fitting, held, targets, anchor.tolist(), held_missing)
                if fit_count - first_fit != 2 or error_regimes["overall"]["count"] != len(held):
                    raise ValueError("incomplete_audit_cell")
                cells[active_cell] = {
                    "outer_split_seed": split_seed,
                    "model_seed": model_seed,
                    "fitting_record_count": len(fitting),
                    "held_out_record_count": len(held),
                    "fits": 2,
                    "error_regimes": error_regimes,
                }
                active_cell = None
        if len(cells) != 4 or fit_count != MAXIMUM_MODEL_FITS:
            raise ValueError("incomplete_audit")
        check_deadline()
        status = "completed"
    except Exception as error:
        reason = str(error)
        blockers = (reason if reason in {
            "training_data_feature_audit_invalid_clock",
            "training_data_feature_audit_deadline_reached",
            "training_data_feature_audit_fit_limit_reached",
        } else "training_data_feature_audit_runtime_failed",)

    elapsed = clock() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = 0.0
        status, blockers = "blocked", ("training_data_feature_audit_invalid_clock",)
    elif elapsed >= MAXIMUM_ELAPSED_SECONDS:
        status, blockers = "blocked", ("training_data_feature_audit_deadline_reached",)
    resources = {
        "elapsed_seconds": elapsed,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "fits_started": fit_count,
        "fits_in_completed_cells": 2 * len(cells),
        "unused_fit_capacity": MAXIMUM_MODEL_FITS - fit_count,
        "maximum_fits": MAXIMUM_MODEL_FITS,
        "maximum_elapsed_seconds": MAXIMUM_ELAPSED_SECONDS,
    }
    decision = _decision(contract, cells) if status == "completed" else {
        "route": "blocked_no_decision", "candidate_or_model_run_authorized": False,
    }
    evidence = {
        "version": VERSION,
        "status": status,
        "blockers": list(blockers),
        "contract_and_target_audit": contract,
        "cells": cells,
        "runtime_failed_cell": active_cell,
        "uncompleted_cells": [stability._cell_key(split, seed) for split in OUTER_SPLIT_SEEDS for seed in MODEL_SEEDS if stability._cell_key(split, seed) not in cells],
        "decision": decision,
        "decision_rules": copy.deepcopy(DECISION_RULES),
        "resource_use": resources,
        "validation_labels_used": False,
        "held_out_test_accessed": False,
        "source_groups_used": False,
        "rows_removed_or_repaired": 0,
        "candidate_selected": False,
        "lock_created": False,
        "interpretation": "training_only_descriptive_audit_conditional_on_historical_validation_selected_anchor_and_counts; multiple_regime_views_are_not_independent_tests; no_causal_or_validation_claim",
    }
    if not (output / "audit-plan.json").exists():
        write_private_json(output / "audit-plan.json", plan)
    write_private_json(output / "training-data-feature-audit.json", evidence)
    files = (output / "audit-plan.json", output / "training-data-feature-audit.json")
    write_private_json(output / "manifest.json", {
        "version": VERSION,
        "create_only": True,
        "artifacts": {path.name: sha256(path.read_bytes()).hexdigest() for path in files},
        "validation_labels_used": False,
        "held_out_test_accessed": False,
        "source_groups_used": False,
        "publication_performed": False,
        "candidate_selected": False,
        "lock_created": False,
    })
    return TrainingDataFeatureAuditResult(status, blockers, evidence, resources)


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(description="Private bounded training-only data and feature audit.")
    parser.add_argument("--training-records", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--volume-unit", default="mm3")
    parser.add_argument("--scope-confirmed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_root.resolve() != PREDECLARED_OUTPUT_ROOT.resolve():
            raise InputError("training_data_feature_audit_output_root_mismatch")
        if args.output_root.exists():
            raise InputError("private_output_directory_unavailable")
        stability._verify_predeclared_training_artifact(args.training_records)
        runtime = t.TensorflowXGBoostCandidateRuntime()
        t._verify_predeclared_environment(runtime.dependency_versions)
        result = run_training_data_feature_audit(
            args.training_records,
            EvaluationConfig(None, args.volume_unit, True if args.scope_confirmed else None, seed=stability.NORMALIZATION_SEED),
            runtime=runtime,
            output_root=args.output_root,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (ImportError, RuntimeError, OSError, ValueError, TypeError):
        raise SystemExit("training_data_feature_audit_failed") from None
    print(json.dumps({"status": result.status, "blockers": result.blockers, "resource_use": result.resource_use}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
