"""Governed bounding-box target-decomposition candidate lifecycle."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import copy
import json
import math
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

from ..ingestion import CanonicalRow, fingerprint
from ..private_io import create_private_file, write_private_json
from . import target_decomposition as target
from . import tuning as t

PLAN_KIND = "bounding_box_target_decomposition"
FAMILY = "bounding_box_target_decomposition"
VERSION = target.VERSION
MAXIMUM_FITS = 24
MAXIMUM_VALIDATION_CANDIDATE_EVALUATIONS = 4
FIXED_COUNTS = {"neural_network_epochs": 87, "xgboost_trees": 1091}
SCORED_KINDS = ("decomposed_ensemble", "closest_anchor_control")
OUTER_SPLIT_SEEDS = (101, 202)
MODEL_SEEDS = (41, 42)
ENSEMBLE_NEURAL_WEIGHT = 0.8

TARGET_CONTRACT = {
    "version": VERSION,
    "representation": target.TARGET_UNIT,
    "formula": "sliced_resin_mass_g/(bounding_box_volume_mm3*0.0011_g_per_mm3)",
    "inverse": "factor*bounding_box_volume_mm3*0.0011_g_per_mm3",
    "density_g_per_ml": target.DENSITY_G_PER_ML,
    "bounded_to_one": False,
    "zero_target": "factor_zero",
    "invalid_prediction": "block_without_clipping",
    "interpretation": "label_derived_training_representation_not_occupied_union_volume",
}
HONEST_CONTRACT = {
    "assignment_version": "stable-record-identity-sha256-round-robin-v1",
    "outer_split_seeds": list(OUTER_SPLIT_SEEDS),
    "model_seeds": list(MODEL_SEEDS),
    "outer_holdout": "rank_modulo_5_equals_zero",
    "fit_partition": "other_four_training_folds_only",
    "source_or_family_metadata_used": False,
    "aggregate_mae": "sum_absolute_error_across_all_four_cells_divided_by_total_row_cell_count",
    "intervention_qualification": (
        "nonincreasing_above5g_count_each_cell_and_lower_aggregate_above5g_count_"
        "and_nonincreasing_aggregate_mae"
    ),
    "routing": "decomposition_qualification_routes_both_fixed_candidates_to_validation",
    "selection": "training_evidence_does_not_select_between_candidates",
}


def base_candidates() -> tuple[t.Candidate, ...]:
    """Return raw and decomposed copies of the exact closest base contracts."""
    from . import tail_focused_search

    neural, xgboost = tail_focused_search.base_candidates()
    decomposed = []
    for base in (neural, xgboost):
        parameters = copy.deepcopy(base.parameters)
        parameters["target_representation"] = target.TARGET_UNIT
        decomposed.append(t.Candidate(
            f"decomposed-{base.candidate_id}", base.family, parameters,
            t.LEGACY_FEATURES, VERSION,
        ))
    return neural, xgboost, *decomposed


def candidate(scored_kind: str) -> t.Candidate:
    if scored_kind not in SCORED_KINDS:
        raise ValueError("invalid_candidate_configuration")
    parameters = {
        "scored_kind": scored_kind,
        "base_candidates": tuple(asdict(item) for item in base_candidates()),
        "fixed_training_counts": dict(FIXED_COUNTS),
        "ensemble_neural_weight": ENSEMBLE_NEURAL_WEIGHT,
        "ensemble_formula": "float64_0.8_neural_plus_parenthesized_1_minus_0.8_xgboost",
        "target_contract": copy.deepcopy(TARGET_CONTRACT),
        "honest_contract": copy.deepcopy(HONEST_CONTRACT),
        "validation_use": "conditional_scoring_eligibility_ranking_and_locking_only",
    }
    return t.Candidate(
        f"bbox-target-{scored_kind}-{fingerprint(parameters)[:16]}",
        FAMILY, t._freeze_json_lists(parameters), t.LEGACY_FEATURES, VERSION,
    )


def rules() -> tuple[dict[str, Any], ...]:
    return tuple({
        "construction": PLAN_KIND,
        "candidate_id": candidate(kind).candidate_id,
        "scored_kind": kind,
        **candidate(kind).parameters,
    } for kind in SCORED_KINDS)


def validate_candidate(declared: t.Candidate) -> None:
    try:
        expected = candidate(str(declared.parameters["scored_kind"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid_candidate_configuration") from None
    if declared != expected:
        raise ValueError("invalid_candidate_configuration")


def _fixed_base_specification(base: t.Candidate):
    from .definitions import candidate_model_specification

    parameters = copy.deepcopy(base.parameters)
    representation = parameters.pop("target_representation", None)
    if base.family == "neural_network":
        parameters["maximum_epochs"] = FIXED_COUNTS["neural_network_epochs"]
    else:
        parameters["n_estimators"] = FIXED_COUNTS["xgboost_trees"]
    return candidate_model_specification(
        base.family, parameters,
        identity_namespace=(VERSION if representation else "minires-model-definition-v1"),
        output_unit=(target.TARGET_UNIT if representation else "g"),
    )


def specification(declared: t.Candidate):
    """Resolve one scored candidate to its exact fixed-count model specification."""
    from .definitions import ensemble_model_specification

    validate_candidate(declared)
    raw_neural, raw_xgboost, decomposed_neural, decomposed_xgboost = base_candidates()
    bases = (
        (decomposed_neural, decomposed_xgboost)
        if declared.parameters["scored_kind"] == "decomposed_ensemble"
        else (raw_neural, raw_xgboost)
    )
    return ensemble_model_specification(
        _fixed_base_specification(bases[0]), _fixed_base_specification(bases[1]),
        ENSEMBLE_NEURAL_WEIGHT,
    )


def _ensemble(neural: Sequence[float], xgboost: Sequence[float]) -> tuple[float, ...]:
    if len(neural) != len(xgboost):
        raise ValueError("invalid_target_decomposition_prediction")
    values = tuple(
        ENSEMBLE_NEURAL_WEIGHT * float(left)
        + (1.0 - ENSEMBLE_NEURAL_WEIGHT) * float(right)
        for left, right in zip(neural, xgboost)
    )
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid_target_decomposition_prediction")
    return values


def _fit_bases(
    runtime: t.CandidateRuntime, training: Sequence[CanonicalRow],
    evaluation: Sequence[CanonicalRow], model_seed: int,
    start_fit: Callable[[str, int | None, int, int, t.Candidate], dict[str, Any]],
    check_deadline: Callable[[], None], *, stage: str,
    outer_split_seed: int | None,
) -> tuple[list[t.LockedFit], tuple[tuple[float, ...], ...]]:
    fits: list[t.LockedFit] = []
    columns: list[tuple[float, ...]] = []
    for base_index, (base, count) in enumerate(
        zip(base_candidates(), (87, 1091, 87, 1091)), 1,
    ):
        decomposed = base.parameters.get("target_representation") == target.TARGET_UNIT
        if decomposed:
            train_x, train_y = target.occupancy_training_matrix(training)
            evaluation_x, _ = target.occupancy_training_matrix(evaluation)
        else:
            train_x, train_y = t.candidate_prediction_matrix(training, base)
            evaluation_x, _ = t.candidate_prediction_matrix(evaluation, base)
        attempt = start_fit(stage, outer_split_seed, model_seed, base_index, base)
        try:
            fitted = t._stack_refit(runtime, base, model_seed, train_x, train_y, count)
        except Exception:
            attempt["status"] = "failed"
            attempt["bounded_failure_reason"] = "candidate_runtime_failed"
            raise RuntimeError("candidate_runtime_failed") from None
        attempt["status"] = "completed"
        check_deadline()
        predictions = t._predict(fitted.predictor, evaluation_x)
        check_deadline()
        fits.append(fitted)
        columns.append(predictions)
    return fits, tuple(columns)


def _predictions(
    features: Sequence[tuple[float, ...]],
    columns: Sequence[Sequence[float]],
) -> dict[str, tuple[float, ...]]:
    if len(columns) != 4:
        raise ValueError("invalid_target_decomposition_prediction")
    raw_neural, raw_xgboost, decomposed_neural, decomposed_xgboost = columns
    factor = _ensemble(decomposed_neural, decomposed_xgboost)
    return {
        "decomposed_ensemble": target.reconstruct_mass_predictions(features, factor),
        "closest_anchor_control": _ensemble(raw_neural, raw_xgboost),
    }


def _honest_metrics(targets: Sequence[float], predictions: Sequence[float]) -> dict[str, Any]:
    if not targets or len(targets) != len(predictions):
        raise ValueError("invalid_target_decomposition_evidence")
    errors = [abs(float(prediction) - float(target))
              for prediction, target in zip(predictions, targets)]
    if not all(math.isfinite(value) for value in errors):
        raise ValueError("invalid_target_decomposition_evidence")
    return {
        "count": len(errors),
        "absolute_error_sum_g": math.fsum(errors),
        "mae_g": math.fsum(errors) / len(errors),
        "above_5g_count": sum(value > 5.0 for value in errors),
    }


def _route_decision(cells: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    anchor_kind = "closest_anchor_control"
    decisions: dict[str, Any] = {}
    for kind in ("decomposed_ensemble",):
        candidate_cells = [cell["metrics"][kind] for cell in cells.values()]
        anchor_cells = [cell["metrics"][anchor_kind] for cell in cells.values()]
        candidate_count = sum(int(item["count"]) for item in candidate_cells)
        anchor_count = sum(int(item["count"]) for item in anchor_cells)
        candidate_error = math.fsum(float(item["absolute_error_sum_g"])
                                    for item in candidate_cells)
        anchor_error = math.fsum(float(item["absolute_error_sum_g"])
                                 for item in anchor_cells)
        decision = {
            "nonincreasing_above_5g_count_every_cell": all(
                candidate_item["above_5g_count"] <= anchor_item["above_5g_count"]
                for candidate_item, anchor_item in zip(candidate_cells, anchor_cells)
            ),
            "aggregate_above_5g_count": sum(int(item["above_5g_count"])
                                             for item in candidate_cells),
            "anchor_aggregate_above_5g_count": sum(int(item["above_5g_count"])
                                                    for item in anchor_cells),
            "aggregate_mae_g": candidate_error / candidate_count,
            "anchor_aggregate_mae_g": anchor_error / anchor_count,
        }
        decision["qualified"] = (
            decision["nonincreasing_above_5g_count_every_cell"]
            and decision["aggregate_above_5g_count"]
            < decision["anchor_aggregate_above_5g_count"]
            and decision["aggregate_mae_g"] <= decision["anchor_aggregate_mae_g"]
        )
        decisions[kind] = decision
    return decisions, any(item["qualified"] for item in decisions.values())


def _candidate_run(
    kind: str, seed: int, training_count: int, validation: Sequence[CanonicalRow],
    predictions: Sequence[float], fitted: t.LockedFit,
) -> t.CandidateRun:
    declared = candidate(kind)
    reports = t._explicit_validation_source_reports(training_count, validation, predictions)
    metrics = t._candidate_metrics(reports)
    eligible = t._tail_eligible(metrics)
    prediction_fingerprint = fingerprint({
        "candidate_id": declared.candidate_id,
        "model_seed": seed,
        "predictions_g": [float(value) for value in predictions],
    })
    scored_evidence_fingerprint = fingerprint({
        "scored_state_fingerprint": fitted.metadata["scored_state_fingerprint"],
        "prediction_fingerprint": prediction_fingerprint,
        "metrics": metrics,
        "source_reports_fingerprint": fingerprint(reports),
    })
    return t.CandidateRun(
        declared, seed, "completed",
        () if eligible else ("development_serious_error_gate_failed",),
        eligible, metrics, reports,
        ({"fixed_training_counts": dict(FIXED_COUNTS),
          "validation_labels_used_for_fitting": False,
          "scored_kind": kind,
          "scored_state_fingerprint": fitted.metadata["scored_state_fingerprint"],
          "prediction_fingerprint": prediction_fingerprint,
          "source_reports_fingerprint": fingerprint(reports),
          "metrics_fingerprint": fingerprint(metrics),
          "scored_evidence_fingerprint": scored_evidence_fingerprint},),
        t._resources(0.0, 0.0),
    )


def _fitted_state(
    kind: str, fits: Sequence[t.LockedFit], model_seed: int,
) -> t.LockedFit:
    indices = (2, 3) if kind == "decomposed_ensemble" else (0, 1)
    selected = [fits[index] for index in indices]
    bases = _selected_base_candidates(candidate(kind))
    artifacts: dict[str, bytes] = {}
    preprocessing: dict[str, Any] = {}
    inventory: dict[str, list[str]] = {}
    component_fits: list[dict[str, Any]] = []
    for index, (base, fitted) in enumerate(zip(bases, selected), 1):
        preprocessing[f"base_{index}"] = copy.deepcopy(fitted.preprocessing_state)
        inventory[f"base_{index}"] = sorted(fitted.artifacts)
        count_name = "selected_epochs" if base.family == "neural_network" else "selected_trees"
        expected_count = (
            FIXED_COUNTS["neural_network_epochs"]
            if base.family == "neural_network" else FIXED_COUNTS["xgboost_trees"]
        )
        fitted_fingerprint = fitted.metadata.get("fitted_parameter_fingerprint")
        if (
            fitted.metadata.get("seed") != model_seed
            or fitted.metadata.get(count_name) != expected_count
            or not isinstance(fitted_fingerprint, str)
            or len(fitted_fingerprint) != 64
        ):
            raise ValueError("invalid_locked_candidate_artifact")
        artifact_checksums: dict[str, str] = {}
        for name, content in fitted.artifacts.items():
            if not name or Path(name).name != name or not isinstance(content, bytes):
                raise ValueError("invalid_locked_candidate_artifact")
            artifacts[f"base-{index:02d}-{name}"] = content
            artifact_checksums[name] = sha256(content).hexdigest()
        component_fits.append({
            "base_candidate_id": base.candidate_id,
            "model_seed": model_seed,
            "fixed_count_name": count_name,
            "fixed_count": expected_count,
            "fitted_parameter_fingerprint": fitted_fingerprint,
            "artifact_sha256": dict(sorted(artifact_checksums.items())),
            "parameter_artifact_binding": fingerprint({
                "fitted_parameter_fingerprint": fitted_fingerprint,
                "artifact_sha256": dict(sorted(artifact_checksums.items())),
            }),
        })
    scored_state_fingerprint = fingerprint(component_fits)

    def predictor(rows: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
        combined = _ensemble(
            t._predict(selected[0].predictor, rows),
            t._predict(selected[1].predictor, rows),
        )
        return (
            target.reconstruct_mass_predictions(rows, combined)
            if kind == "decomposed_ensemble" else combined
        )
    return t.LockedFit(
        predictor, preprocessing, artifacts,
        {"seed": model_seed, "artifact_inventory": inventory,
         "component_fits": component_fits,
         "scored_state_fingerprint": scored_state_fingerprint,
         "scored_kind": kind},
    )


def _fit_accounting(
    attempts: Sequence[Mapping[str, Any]], validation_evaluation_count: int,
) -> dict[str, Any]:
    return {
        "maximum_base_fit_attempts": MAXIMUM_FITS,
        "attempted_base_fits": len(attempts),
        "completed_base_fits": sum(item.get("status") == "completed" for item in attempts),
        "failed_base_fits": sum(item.get("status") == "failed" for item in attempts),
        "planned_honest_base_fits": 16,
        "planned_conditional_production_base_fits": 8,
        "attempts": copy.deepcopy(list(attempts)),
        "completed_validation_candidate_evaluations": validation_evaluation_count,
        "maximum_validation_candidate_evaluations": (
            MAXIMUM_VALIDATION_CANDIDATE_EVALUATIONS
        ),
        "budget_recycling": False,
    }


def develop(
    output: Path, plan: t.SearchPlan, runtime: t.CandidateRuntime,
    training: Sequence[CanonicalRow], validation: Sequence[CanonicalRow],
    training_fingerprint: str, validation_fingerprint: str,
    started: float, cpu_started: float, deadline: float, clock: Callable[[], float],
) -> t.TuningResult:
    fit_attempts: list[dict[str, Any]] = []

    def check_deadline() -> None:
        if clock() >= deadline:
            raise RuntimeError("candidate_search_deadline_reached")

    def start_fit(
        stage: str, outer_split_seed: int | None, model_seed: int,
        base_index: int, base: t.Candidate,
    ) -> dict[str, Any]:
        if len(fit_attempts) >= MAXIMUM_FITS:
            raise RuntimeError("candidate_fit_limit_reached")
        check_deadline()
        attempt = {
            "attempt_number": len(fit_attempts) + 1,
            "stage": stage,
            "outer_split_seed": outer_split_seed,
            "model_seed": model_seed,
            "base_index": base_index,
            "base_candidate_id": base.candidate_id,
            "status": "started",
        }
        fit_attempts.append(attempt)
        return attempt

    honest: dict[str, Any] = {
        "version": VERSION, "contract": copy.deepcopy(HONEST_CONTRACT), "cells": {},
        "interventions": {}, "route_to_validation": False,
    }
    first: list[t.CandidateRun] = []
    second: list[t.CandidateRun] = []
    validation_evaluation_count = 0
    combined: list[dict[str, Any]] = []
    blockers: list[str] = []
    locked = None
    lock_directory = output / "locked-candidate"
    status = "blocked"
    try:
        for outer_seed in OUTER_SPLIT_SEEDS:
            assignments = t._cross_fit_assignments(training, outer_seed, 5)
            fitted_rows = [row for row, fold in zip(training, assignments) if fold != 0]
            held_rows = [row for row, fold in zip(training, assignments) if fold == 0]
            if not fitted_rows or not held_rows:
                raise ValueError("invalid_cross_fit_partition")
            held_features, held_targets = t.candidate_prediction_matrix(
                held_rows, base_candidates()[0]
            )
            for model_seed in MODEL_SEEDS:
                _, columns = _fit_bases(
                    runtime, fitted_rows, held_rows, model_seed, start_fit,
                    check_deadline, stage="honest_training",
                    outer_split_seed=outer_seed,
                )
                cell_predictions = _predictions(held_features, columns)
                honest["cells"][f"{outer_seed}:{model_seed}"] = {
                    "outer_split_seed": outer_seed,
                    "model_seed": model_seed,
                    "fitting_record_count": len(fitted_rows),
                    "held_out_record_count": len(held_rows),
                    "metrics": {
                        kind: _honest_metrics(held_targets, values)
                        for kind, values in cell_predictions.items()
                    },
                }
        honest["interventions"], honest["route_to_validation"] = _route_decision(
            honest["cells"]
        )
        check_deadline()
        if not honest["route_to_validation"]:
            status = "training_evidence_rejected"
            blockers.append("target_decomposition_training_prerequisite_failed")
        else:
            seed42_fits: dict[str, t.LockedFit] = {}
            for model_seed, runs in ((41, first), (42, second)):
                fits, columns = _fit_bases(
                    runtime, training, validation, model_seed, start_fit,
                    check_deadline, stage="conditional_production",
                    outer_split_seed=None,
                )
                validation_features, _ = t.candidate_prediction_matrix(
                    validation, base_candidates()[0]
                )
                cell_predictions = _predictions(validation_features, columns)
                for kind in SCORED_KINDS:
                    if validation_evaluation_count >= MAXIMUM_VALIDATION_CANDIDATE_EVALUATIONS:
                        raise RuntimeError("candidate_run_limit_reached")
                    check_deadline()
                    scored_fit = _fitted_state(kind, fits, model_seed)
                    runs.append(_candidate_run(
                        kind, model_seed, len(training), validation,
                        cell_predictions[kind], scored_fit,
                    ))
                    validation_evaluation_count += 1
                    if model_seed == 42:
                        seed42_fits[candidate(kind).candidate_id] = scored_fit
                    check_deadline()
            check_deadline()
            combined = t._combine_seed_results(first, second)
            eligible = [item for item in combined if item["eligible"]]
            check_deadline()
            if not eligible:
                status = "completed_no_candidate"
                blockers.append("no_eligible_candidate")
            else:
                selected = min(eligible, key=t._combined_rank_key)
                declared = next(candidate(kind) for kind in SCORED_KINDS
                                if candidate(kind).candidate_id == selected["candidate_id"])
                honest["status"] = "completed"
                honest["blockers"] = []
                honest["uncompleted_cells"] = []
                honest["fit_accounting"] = _fit_accounting(
                    fit_attempts, validation_evaluation_count,
                )
                check_deadline()
                try:
                    seed42_run = next(
                        run for run in second
                        if run.candidate.candidate_id == declared.candidate_id
                    )
                    locked = _lock(
                        lock_directory, declared,
                        seed42_fits[declared.candidate_id], seed42_run, selected, plan,
                        training, validation, training_fingerprint,
                        validation_fingerprint, honest,
                    )
                except Exception:
                    raise RuntimeError("candidate_refit_or_lock_failed") from None
                check_deadline()
                status = "completed"
        check_deadline()
    except Exception as error:
        locked = None
        status = "blocked"
        reason = str(error)
        blockers = [reason if reason in {
            "candidate_fit_limit_reached", "candidate_search_deadline_reached",
            "candidate_run_limit_reached", "candidate_refit_or_lock_failed",
        } else "candidate_runtime_failed"]
        if lock_directory.exists():
            try:
                shutil.rmtree(lock_directory)
            except OSError:
                blockers = ["candidate_refit_or_lock_failed"]
    planned_cells = [
        f"{outer_seed}:{model_seed}"
        for outer_seed in OUTER_SPLIT_SEEDS for model_seed in MODEL_SEEDS
    ]
    honest["status"] = status
    honest["blockers"] = list(blockers)
    honest["uncompleted_cells"] = [
        key for key in planned_cells if key not in honest["cells"]
    ]
    honest["fit_accounting"] = _fit_accounting(
        fit_attempts, validation_evaluation_count,
    )
    write_private_json(output / "honest-training-evidence.json", honest)
    attempted = len(fit_attempts)
    completed_fits = sum(item.get("status") == "completed" for item in fit_attempts)
    failed_fits = sum(item.get("status") == "failed" for item in fit_attempts)
    result = t.TuningResult(
        status, tuple(blockers), len(first) + len(second),
        {"neural_network": 0, "xgboost": 0,
         "ensemble": len(first), "second_seed": len(second), "control": 0},
        plan, None, tuple(first), tuple(second), tuple(combined), locked,
        {**t._resources(max(0.0, clock() - started), time.process_time() - cpu_started),
         "model_fit_attempts": attempted, "completed_model_fits": completed_fits,
         "failed_model_fits": failed_fits, "maximum_model_fit_attempts": MAXIMUM_FITS,
         "validation_candidate_evaluations": validation_evaluation_count,
         "maximum_validation_candidate_evaluations": MAXIMUM_VALIDATION_CANDIDATE_EVALUATIONS},
    )
    t._write_tuning_outputs(output, result)
    return result


def _selected_base_candidates(declared: t.Candidate) -> tuple[t.Candidate, ...]:
    validate_candidate(declared)
    raw_neural, raw_xgboost, decomposed_neural, decomposed_xgboost = base_candidates()
    return (
        (decomposed_neural, decomposed_xgboost)
        if declared.parameters["scored_kind"] == "decomposed_ensemble"
        else (raw_neural, raw_xgboost)
    )


def _split_evidence(identities: Sequence[str]) -> dict[str, Any]:
    cells = {}
    for outer_seed in OUTER_SPLIT_SEEDS:
        assignments = list(t._cross_fit_assignments_for_identities(identities, outer_seed, 5))
        for model_seed in MODEL_SEEDS:
            cells[f"{outer_seed}:{model_seed}"] = {
                "outer_split_seed": outer_seed, "model_seed": model_seed,
                "outer_assignments": assignments,
            }
    value = {"training_record_identities": list(identities), "cells": cells}
    return {**value, "fingerprint": fingerprint(value)}


def _feature_contract() -> dict[str, Any]:
    return {
        "ordered_features": list(t.LEGACY_FEATURES), "dtype": "float32",
        "transformation_version": VERSION,
        "base_transformation_version": t.TRANSFORMATION_VERSION,
        "target_contract": copy.deepcopy(TARGET_CONTRACT),
    }


def _data_usage() -> dict[str, str]:
    return {
        "fitting": "training_records_only",
        "preprocessing": "training_records_only",
        "early_stopping": "not_used_fixed_training_counts",
        "honest_evaluation": "training_outer_holdout_only_before_validation",
        "candidate_selection": "conditional_validation_scoring_only",
        "candidate_locking": "training_and_validation_contract_only",
    }


def _lock(
    directory: Path, declared: t.Candidate, fitted: t.LockedFit,
    seed42_run: t.CandidateRun, selected: Mapping[str, Any], plan: t.SearchPlan,
    training: Sequence[CanonicalRow], validation: Sequence[CanonicalRow],
    training_fingerprint: str, validation_fingerprint: str,
    honest: Mapping[str, Any],
) -> t.LockedCandidate:
    identities = [str(row.metadata["record_identity"]) for row in training]
    contract = {
        "version": t.TUNING_VERSION,
        "development_contract": "explicit_train_validation",
        "candidate": asdict(declared),
        "model_specification": specification(declared).to_dict(),
        "selection_seeds": [41, 42], "seed_weighting": "equal_weight_each_seed",
        "lock_prediction": "exact_seed42_training_only_state_already_scored",
        "seed42_scored_fit_metadata": copy.deepcopy(list(seed42_run.fit_metadata)),
        "seed42_scored_metrics": copy.deepcopy(seed42_run.metrics),
        "seed42_source_reports_fingerprint": fingerprint(seed42_run.source_reports),
        "selected_combined_development_evidence": copy.deepcopy(dict(selected)),
        "search_plan": plan.to_dict(),
        "development_evidence": {
            "search_plan_id": plan.plan_id,
            "search_plan_fingerprint": fingerprint(plan.to_dict()),
            "training_input_fingerprint": training_fingerprint,
            "validation_input_fingerprint": validation_fingerprint,
            "normalized_training_fingerprint": fingerprint([asdict(row) for row in training]),
            "normalized_validation_fingerprint": fingerprint([asdict(row) for row in validation]),
            "partition_identity_fingerprint": plan.source_allocation_fingerprint,
            "test_input_attestation": "no_test_argument_or_path_available",
            "validation_grouping_contract": t.EXPLICIT_DEVELOPMENT_EVIDENCE_VERSION,
            "honest_training_fingerprint": fingerprint(honest),
        },
        "development_source_groups": sorted({str(row.metadata["anonymous_source_group"])
                                             for row in (*training, *validation)}),
        "development_data_usage": _data_usage(),
        "feature_contract": _feature_contract(),
        "features": list(t.LEGACY_FEATURES), "fixed_training_counts": dict(FIXED_COUNTS),
        "target_contract": copy.deepcopy(TARGET_CONTRACT),
        "honest_contract": copy.deepcopy(HONEST_CONTRACT), "split_evidence": _split_evidence(identities),
        "eligibility_rule": t._eligibility_gates(), "ranking_rule": list(plan.ranking_rule),
        "dependency_versions": dict(plan.dependency_versions),
        "dependency_environment": {"python": platform.python_version(), "platform": platform.platform(),
                                   "versions": dict(plan.dependency_versions)},
        "code_fingerprint": plan.code_fingerprint,
        "refit_partition": "training_records_only", "refit_record_count": len(training),
        "validation_record_count": len(validation), "runtime_metadata": fitted.metadata,
        "test_input_accessed": False, "final_test_access": False, "output_unit": "g",
        "classification": "internal_advisory_human_review_required",
    }
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_private_json(directory / "candidate-contract.json", contract)
    write_private_json(directory / "preprocessing-state.json", fitted.preprocessing_state)
    write_private_json(directory / "honest-training-evidence.json", honest)
    for name, content in sorted(fitted.artifacts.items()):
        with create_private_file(directory / name) as stream:
            stream.write(content)
    manifest = {
        "version": t.TUNING_VERSION, "create_only": True,
        "files": {path.name: sha256(path.read_bytes()).hexdigest()
                  for path in sorted(directory.iterdir()) if path.is_file()},
        "locked_before_final_assessment": True, "test_input_accessed": False,
    }
    write_private_json(directory / "lock-manifest.json", manifest)
    blockers, _, _ = t.verify_locked_candidate_files(directory, plan.dependency_versions)
    if blockers:
        raise ValueError("invalid_target_decomposition_lock")
    return t.LockedCandidate(declared, fitted.predictor, directory, manifest, contract)


def valid_contract(contract: Mapping[str, Any]) -> bool:
    try:
        declared = candidate(str(contract["candidate"]["parameters"]["scored_kind"]))
        plan = contract["search_plan"]
        expected_plan = t.generate_search_plan(
            t.SearchLimits.for_plan(41, PLAN_KIND),
            input_fingerprint=plan["input_fingerprint"],
            code_fingerprint=plan["code_fingerprint"],
            dependency_versions=plan["dependency_versions"],
            normalized_input_fingerprint=plan["normalized_input_fingerprint"],
            source_allocation_fingerprint=plan["source_allocation_fingerprint"],
            configuration_fingerprint=plan["configuration_fingerprint"],
            code_configuration_fingerprint=plan["code_configuration_fingerprint"],
        )
        selected = contract["selected_combined_development_evidence"]
        identities = contract["split_evidence"]["training_record_identities"]
        evidence = contract["development_evidence"]
        sources = contract["development_source_groups"]
        runtime_metadata = contract["runtime_metadata"]
        seed42_metadata = contract["seed42_scored_fit_metadata"]
        return (
            json.loads(json.dumps(contract["candidate"])) == json.loads(json.dumps(asdict(declared)))
            and contract["model_specification"] == specification(declared).to_dict()
            and contract["search_plan"] == json.loads(json.dumps(expected_plan.to_dict()))
            and contract["version"] == t.TUNING_VERSION
            and contract["development_contract"] == "explicit_train_validation"
            and contract["selection_seeds"] == [41, 42]
            and contract["lock_prediction"] == "exact_seed42_training_only_state_already_scored"
            and runtime_metadata["seed"] == 42
            and runtime_metadata["scored_kind"] == declared.parameters["scored_kind"]
            and runtime_metadata["scored_state_fingerprint"]
            == fingerprint(runtime_metadata["component_fits"])
            and isinstance(seed42_metadata, list) and len(seed42_metadata) == 1
            and set(seed42_metadata[0]) == {
                "fixed_training_counts", "validation_labels_used_for_fitting",
                "scored_kind", "scored_state_fingerprint", "prediction_fingerprint",
                "source_reports_fingerprint", "metrics_fingerprint",
                "scored_evidence_fingerprint",
            }
            and seed42_metadata[0]["fixed_training_counts"] == FIXED_COUNTS
            and seed42_metadata[0]["validation_labels_used_for_fitting"] is False
            and seed42_metadata[0]["scored_kind"] == declared.parameters["scored_kind"]
            and seed42_metadata[0]["scored_state_fingerprint"]
            == runtime_metadata["scored_state_fingerprint"]
            and seed42_metadata[0]["source_reports_fingerprint"]
            == contract["seed42_source_reports_fingerprint"]
            and seed42_metadata[0]["metrics_fingerprint"]
            == fingerprint(contract["seed42_scored_metrics"])
            and seed42_metadata[0]["scored_evidence_fingerprint"] == fingerprint({
                "scored_state_fingerprint": runtime_metadata["scored_state_fingerprint"],
                "prediction_fingerprint": seed42_metadata[0]["prediction_fingerprint"],
                "metrics": contract["seed42_scored_metrics"],
                "source_reports_fingerprint": contract["seed42_source_reports_fingerprint"],
            })
            and t._valid_candidate_metrics(contract["seed42_scored_metrics"])
            and t._tail_eligible(contract["seed42_scored_metrics"])
            and contract["fixed_training_counts"] == FIXED_COUNTS
            and contract["target_contract"] == TARGET_CONTRACT
            and contract["honest_contract"] == HONEST_CONTRACT
            and contract["feature_contract"] == _feature_contract()
            and contract["features"] == list(t.LEGACY_FEATURES)
            and contract["development_data_usage"] == _data_usage()
            and isinstance(identities, list) and len(identities) >= 6
            and all(isinstance(value, str) and value for value in identities)
            and len(set(identities)) == len(identities)
            and contract["split_evidence"] == _split_evidence(identities)
            and contract["refit_record_count"] == len(identities)
            and contract["refit_partition"] == "training_records_only"
            and contract["validation_record_count"] > 0
            and contract["test_input_accessed"] is False
            and contract["final_test_access"] is False
            and contract["output_unit"] == "g"
            and contract["eligibility_rule"] == t._eligibility_gates()
            and contract["ranking_rule"] == list(expected_plan.ranking_rule)
            and contract["dependency_versions"] == expected_plan.dependency_versions
            and contract["dependency_environment"]["versions"] == expected_plan.dependency_versions
            and contract["code_fingerprint"] == expected_plan.code_fingerprint
            and isinstance(sources, list) and bool(sources)
            and sources == sorted(set(sources))
            and evidence["search_plan_id"] == expected_plan.plan_id
            and evidence["search_plan_fingerprint"] == fingerprint(plan)
            and evidence["partition_identity_fingerprint"] == expected_plan.source_allocation_fingerprint
            and evidence["test_input_attestation"] == "no_test_argument_or_path_available"
            and evidence["validation_grouping_contract"] == t.EXPLICIT_DEVELOPMENT_EVIDENCE_VERSION
            and all(isinstance(evidence[key], str) and evidence[key] for key in (
                "training_input_fingerprint", "validation_input_fingerprint",
                "normalized_training_fingerprint", "normalized_validation_fingerprint",
                "honest_training_fingerprint",
            ))
            and selected["candidate_id"] == declared.candidate_id
            and selected["eligible"] is True
            and selected["seed_results"] == [41, 42]
            and selected["seed_eligibility"] == [True, True]
            and selected["family"] == FAMILY
            and selected["blockers"] == []
            and selected["equal_seed_weight"] == 0.5
            and t._valid_candidate_metrics(selected["metrics"])
            and t._tail_eligible(selected["metrics"])
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _valid_honest_metric(metric: Any, expected_count: int) -> bool:
    if not isinstance(metric, Mapping) or set(metric) != {
        "count", "absolute_error_sum_g", "mae_g", "above_5g_count",
    }:
        return False
    count = metric["count"]
    absolute_error = metric["absolute_error_sum_g"]
    mae = metric["mae_g"]
    above = metric["above_5g_count"]
    return (
        isinstance(count, int) and not isinstance(count, bool)
        and count == expected_count and count > 0
        and isinstance(above, int) and not isinstance(above, bool)
        and 0 <= above <= count
        and isinstance(absolute_error, (int, float))
        and not isinstance(absolute_error, bool)
        and math.isfinite(float(absolute_error)) and float(absolute_error) >= 0.0
        and isinstance(mae, (int, float)) and not isinstance(mae, bool)
        and math.isfinite(float(mae)) and float(mae) >= 0.0
        and math.isclose(
            float(mae), float(absolute_error) / count,
            rel_tol=1e-12, abs_tol=1e-12,
        )
    )


def _expected_completed_attempts() -> list[dict[str, Any]]:
    bases = base_candidates()
    expected: list[dict[str, Any]] = []
    for outer_seed in OUTER_SPLIT_SEEDS:
        for model_seed in MODEL_SEEDS:
            for base_index, base in enumerate(bases, 1):
                expected.append({
                    "attempt_number": len(expected) + 1,
                    "stage": "honest_training",
                    "outer_split_seed": outer_seed,
                    "model_seed": model_seed,
                    "base_index": base_index,
                    "base_candidate_id": base.candidate_id,
                    "status": "completed",
                })
    for model_seed in MODEL_SEEDS:
        for base_index, base in enumerate(bases, 1):
            expected.append({
                "attempt_number": len(expected) + 1,
                "stage": "conditional_production",
                "outer_split_seed": None,
                "model_seed": model_seed,
                "base_index": base_index,
                "base_candidate_id": base.candidate_id,
                "status": "completed",
            })
    return expected


def _valid_honest_evidence(honest: Any, contract: Mapping[str, Any]) -> bool:
    if not isinstance(honest, Mapping):
        return False
    expected_keys = {
        f"{outer_seed}:{model_seed}"
        for outer_seed in OUTER_SPLIT_SEEDS for model_seed in MODEL_SEEDS
    }
    cells = honest.get("cells")
    identities = contract["split_evidence"]["training_record_identities"]
    if (
        honest.get("version") != VERSION
        or honest.get("contract") != HONEST_CONTRACT
        or honest.get("route_to_validation") is not True
        or honest.get("status") != "completed"
        or honest.get("blockers") != []
        or honest.get("uncompleted_cells") != []
        or not isinstance(cells, Mapping)
        or set(cells) != expected_keys
    ):
        return False
    for outer_seed in OUTER_SPLIT_SEEDS:
        assignments = t._cross_fit_assignments_for_identities(identities, outer_seed, 5)
        fitting_count = sum(fold != 0 for fold in assignments)
        held_count = sum(fold == 0 for fold in assignments)
        if not fitting_count or not held_count:
            return False
        for model_seed in MODEL_SEEDS:
            cell = cells[f"{outer_seed}:{model_seed}"]
            if (
                not isinstance(cell, Mapping)
                or set(cell) != {
                    "outer_split_seed", "model_seed", "fitting_record_count",
                    "held_out_record_count", "metrics",
                }
                or cell["outer_split_seed"] != outer_seed
                or cell["model_seed"] != model_seed
                or cell["fitting_record_count"] != fitting_count
                or cell["held_out_record_count"] != held_count
                or not isinstance(cell["metrics"], Mapping)
                or set(cell["metrics"]) != set(SCORED_KINDS)
                or not all(
                    _valid_honest_metric(cell["metrics"][kind], held_count)
                    for kind in SCORED_KINDS
                )
            ):
                return False
    decisions, route = _route_decision(cells)
    accounting = honest.get("fit_accounting")
    expected_attempts = _expected_completed_attempts()
    return (
        route is True
        and honest.get("interventions") == decisions
        and isinstance(accounting, Mapping)
        and set(accounting) == {
            "maximum_base_fit_attempts", "attempted_base_fits",
            "completed_base_fits", "failed_base_fits",
            "planned_honest_base_fits",
            "planned_conditional_production_base_fits", "attempts",
            "completed_validation_candidate_evaluations",
            "maximum_validation_candidate_evaluations", "budget_recycling",
        }
        and accounting["maximum_base_fit_attempts"] == MAXIMUM_FITS
        and accounting["attempted_base_fits"] == MAXIMUM_FITS
        and accounting["completed_base_fits"] == MAXIMUM_FITS
        and accounting["failed_base_fits"] == 0
        and accounting["planned_honest_base_fits"] == 16
        and accounting["planned_conditional_production_base_fits"] == 8
        and accounting["attempts"] == expected_attempts
        and accounting["completed_validation_candidate_evaluations"]
        == MAXIMUM_VALIDATION_CANDIDATE_EVALUATIONS
        and accounting["maximum_validation_candidate_evaluations"]
        == MAXIMUM_VALIDATION_CANDIDATE_EVALUATIONS
        and accounting["budget_recycling"] is False
    )


def verify_state_files(
    directory: Path, contract: Mapping[str, Any], preprocessing: Mapping[str, Any],
) -> bool:
    try:
        from .definitions import (
            ModelKind, validate_component_preprocessing,
        )

        honest = json.loads((directory / "honest-training-evidence.json").read_text())
        declared = t._candidate_from_dict(contract["candidate"])
        bases = _selected_base_candidates(declared)
        resolved = specification(declared)
        specifications = (
            (resolved,)
            if resolved.model_kind is not ModelKind.ENSEMBLE
            else (resolved.ensemble.neural_network, resolved.ensemble.xgboost)
        )
        runtime_metadata = contract["runtime_metadata"]
        inventory = runtime_metadata["artifact_inventory"]
        component_fits = runtime_metadata["component_fits"]
        expected_files = {
            "candidate-contract.json", "preprocessing-state.json",
            "honest-training-evidence.json", "lock-manifest.json",
        }
        if (
            len(specifications) != len(bases)
            or not isinstance(component_fits, list)
            or len(component_fits) != len(bases)
            or runtime_metadata.get("scored_state_fingerprint")
            != fingerprint(component_fits)
            or set(inventory) != {f"base_{index}" for index in range(1, len(bases) + 1)}
            or set(preprocessing) != {f"base_{index}" for index in range(1, len(bases) + 1)}
        ):
            return False
        for index, (base, base_specification) in enumerate(zip(bases, specifications), 1):
            names = inventory[f"base_{index}"]
            component = component_fits[index - 1]
            count_name = (
                "selected_epochs" if base.family == "neural_network" else "selected_trees"
            )
            expected_count = (
                FIXED_COUNTS["neural_network_epochs"]
                if base.family == "neural_network" else FIXED_COUNTS["xgboost_trees"]
            )
            if (
                not isinstance(names, list) or not names
                or names != sorted(set(names))
                or any(
                    not isinstance(name, str) or not name
                    or Path(name).name != name
                    for name in names
                )
                or not isinstance(component, Mapping)
                or set(component) != {
                    "base_candidate_id", "model_seed", "fixed_count_name",
                    "fixed_count", "fitted_parameter_fingerprint", "artifact_sha256",
                    "parameter_artifact_binding",
                }
                or component["base_candidate_id"] != base.candidate_id
                or component["model_seed"] != 42
                or component["fixed_count_name"] != count_name
                or component["fixed_count"] != expected_count
                or not isinstance(component["fitted_parameter_fingerprint"], str)
                or len(component["fitted_parameter_fingerprint"]) != 64
            ):
                return False
            expected_checksums = {
                name: sha256((directory / f"base-{index:02d}-{name}").read_bytes()).hexdigest()
                for name in names
            }
            if (
                component["artifact_sha256"] != expected_checksums
                or component["parameter_artifact_binding"] != fingerprint({
                    "fitted_parameter_fingerprint": component["fitted_parameter_fingerprint"],
                    "artifact_sha256": expected_checksums,
                })
            ):
                return False
            validate_component_preprocessing(
                base_specification, preprocessing[f"base_{index}"]
            )
            expected_files.update(f"base-{index:02d}-{name}" for name in names)
        return (
            {path.name for path in directory.iterdir()} == expected_files
            and _valid_honest_evidence(honest, contract)
            and fingerprint(honest)
            == contract["development_evidence"]["honest_training_fingerprint"]
        )
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return False


def load(models: Any, directory: Path, contract: Mapping[str, Any]):
    preprocessing = json.loads((directory / "preprocessing-state.json").read_text())
    declared = t._candidate_from_dict(contract["candidate"])
    base_predictors = []
    for index, base in enumerate(_selected_base_candidates(declared), 1):
        base_specification = (
            specification(declared) if len(_selected_base_candidates(declared)) == 1
            else (specification(declared).ensemble.neural_network
                  if index == 1 else specification(declared).ensemble.xgboost)
        )
        names = contract["runtime_metadata"]["artifact_inventory"][f"base_{index}"]
        artifacts = {name: (directory / f"base-{index:02d}-{name}").read_bytes()
                     for name in names}
        base_predictors.append(models.load_verified_component(
            base_specification, artifacts, preprocessing[f"base_{index}"]
        ))
    def predictor(rows: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
        combined = _ensemble(
            t._predict(base_predictors[0], rows),
            t._predict(base_predictors[1], rows),
        )
        return (
            target.reconstruct_mass_predictions(rows, combined)
            if declared.parameters["scored_kind"] == "decomposed_ensemble"
            else combined
        )
    return predictor
