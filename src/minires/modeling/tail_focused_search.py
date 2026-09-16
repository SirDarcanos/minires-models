"""The fixed tail-correction lifecycle, reached through candidate tuning's public seam.

The honest stage is a pre-validation gate, not a score on meta-training OOF rows.
No base fit, including nested fits, receives an early-stopping partition.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Callable, Mapping, Sequence

from . import tail_correction as numeric
from . import tuning as t
from .definitions import (
    BatchPredictor, ModelRuntime, TailCorrectionModelSpecification,
    COMPONENT_PREPROCESSING_VERSION, validate_component_preprocessing,
)
from ..ingestion import CanonicalRow, fingerprint
from ..private_io import create_private_file, write_private_json

PLAN_KIND = "tail_focused_correction"
FAMILY = "tail_focused_correction"
MAXIMUM_FITS = 52
FIXED_COUNTS = {"neural_network_epochs": 87, "xgboost_trees": 1091}
HONEST_CONTRACT = {
    "assignment_version": "stable-record-identity-sha256-round-robin-v1",
    "outer_holdout": "rank_modulo_5_equals_zero",
    "inner_folds": 5,
    "seed_order": [41, 42],
    "stage_order": "both_honest_seeds_before_any_production_fit_or_validation_score",
    "qualification": "each_seed_strictly_lower_prediction_loss_and_nonincreasing_mae_and_above5g_count",
    "prediction_loss": "mean(0.1*error_squared+4*max(abs(error)-4.5,0)_squared)",
    "historical_selection": "conditional_on_reused_validation_selected_pair_and_counts",
    "rejection": "stop_whole_round_without_validation_scoring_or_budget_recycling",
    "lock_prediction": "selected_seed42_training_only_state_not_equal_seed_prediction_average",
}


def base_candidates() -> tuple[t.Candidate, ...]:
    """Original closest run-011 contracts; selected counts are separate, not ceilings."""
    return (
        t.Candidate("neu-04-1a110172", "neural_network", {
            "activation": "mish", "batch_size": 256, "dropout": 0.1,
            "early_stopping_patience": 8, "l2": 1e-5, "layers": (256, 128, 64),
            "learning_rate": 0.003, "loss": "huber", "maximum_epochs": 100,
            "optimizer": "adam", "validation_selection": t.TAIL_ALIGNED_VALIDATION_SELECTION,
        }),
        t.Candidate("xgb-05-cee2625d", "xgboost", {
            "colsample_bytree": 0.9, "early_stopping_rounds": 50, "gamma": 0.2,
            "learning_rate": 0.05, "max_depth": 9, "min_child_weight": 10.0,
            "n_estimators": 1200, "n_jobs": 1, "objective": "reg:squarederror",
            "reg_alpha": 0.1, "reg_lambda": 10.0, "subsample": 0.9,
            "validation_selection": t.TAIL_ALIGNED_VALIDATION_SELECTION,
        }),
    )


def candidate(scale: float) -> t.Candidate:
    if scale not in (0.0, 1.0):
        raise ValueError("invalid_candidate_configuration")
    parameters = {
        "base_candidates": tuple(asdict(base) for base in base_candidates()),
        "fixed_training_counts": dict(FIXED_COUNTS), "correction_scale": scale,
        "numeric_contract": copy.deepcopy(numeric.CONTRACT),
        "honest_contract": copy.deepcopy(HONEST_CONTRACT),
        "base_runtime_contracts": [
            {**copy.deepcopy(t.CANDIDATE_RUNTIME_CONTRACT[base.family]),
             "early_stopping_partition": "none_fixed_training_count",
             "fixed_training_count": count}
            for base, count in zip(base_candidates(), (87, 1091))
        ],
        "base_preprocessing_state_contract": COMPONENT_PREPROCESSING_VERSION,
        "base_model_specifications": [
            t._locked_model_specification(base, FIXED_COUNTS).to_dict()
            for base in base_candidates()
        ],
    }
    frozen_parameters = t._freeze_json_lists(parameters)
    return t.Candidate("tail-correction-" + fingerprint(parameters)[:16], FAMILY,
                       frozen_parameters, t.LEGACY_FEATURES, numeric.VERSION)


def rules() -> tuple[dict[str, Any], ...]:
    return tuple({"construction": PLAN_KIND, "candidate_id": candidate(scale).candidate_id,
                  **candidate(scale).parameters} for scale in (0.0, 1.0))


def _fit_stage(
    runtime: t.CandidateRuntime, training: Sequence[CanonicalRow],
    evaluation: Sequence[CanonicalRow], seed: int, before_fit: Callable[[], None],
    check_deadline: Callable[[], None], *, assignment_seed: int | None = None,
) -> tuple[dict[str, Any], list[t.LockedFit], list[tuple[float, ...]], dict[str, Any]]:
    assignments = t._cross_fit_assignments(training, seed if assignment_seed is None else assignment_seed, 5)
    oof_columns: list[tuple[float, ...]] = []
    full_columns: list[tuple[float, ...]] = []
    evaluation_columns: list[tuple[float, ...]] = []
    base_fits: list[t.LockedFit] = []
    for base, count in zip(base_candidates(), (87, 1091)):
        oof = [math.nan] * len(training)
        for fold in range(5):
            held = [i for i, value in enumerate(assignments) if value == fold]
            fitted_indices = [i for i, value in enumerate(assignments) if value != fold]
            if not held or not fitted_indices:
                raise ValueError("invalid_cross_fit_partition")
            fit_x, fit_y = t.candidate_prediction_matrix([training[i] for i in fitted_indices], base)
            held_x, _ = t.candidate_prediction_matrix([training[i] for i in held], base)
            before_fit()
            fitted = t._stack_refit(runtime, base, seed, fit_x, fit_y, count)
            check_deadline()
            validate_component_preprocessing(
                t._locked_model_specification(base, FIXED_COUNTS), fitted.preprocessing_state
            )
            predictions = t._predict(fitted.predictor, held_x)
            for index, prediction in zip(held, predictions):
                oof[index] = prediction
        numeric.features((oof, oof))  # Complete finite, float32-representable coverage.
        oof_columns.append(tuple(oof))
        train_x, train_y = t.candidate_prediction_matrix(training, base)
        evaluate_x, _ = t.candidate_prediction_matrix(evaluation, base)
        before_fit()
        fitted = t._stack_refit(runtime, base, seed, train_x, train_y, count)
        check_deadline()
        validate_component_preprocessing(
            t._locked_model_specification(base, FIXED_COUNTS), fitted.preprocessing_state
        )
        base_fits.append(fitted)
        full_columns.append(t._predict(fitted.predictor, train_x))
        check_deadline()
        evaluation_columns.append(t._predict(fitted.predictor, evaluate_x))
    _, train_y = t.candidate_prediction_matrix(training, base_candidates()[0])
    before_fit()
    state = numeric.fit_state(oof_columns, train_y)
    check_deadline()
    oof_anchor, _ = numeric.features(oof_columns)
    full_anchor, _ = numeric.features(full_columns)
    evaluation_anchor, _ = numeric.features(evaluation_columns)
    check_deadline()
    oof_corrected = numeric.predict(state, oof_columns, 1.0)[0]
    check_deadline()
    full_corrected = numeric.predict(state, full_columns, 1.0)[0]
    shift = {
        "base_prediction_shift": {
            f"base_{index}": t._guarded_residual_shift_summary(oof, full)
            for index, (oof, full) in enumerate(zip(oof_columns, full_columns), 1)
        },
        "anchor_shift": t._guarded_residual_shift_summary(oof_anchor.tolist(), full_anchor.tolist()),
        "anchor_distributions": {
            "training_oof": t._prediction_summary(oof_anchor.tolist()),
            "full_fit_training": t._prediction_summary(full_anchor.tolist()),
            "evaluation": t._prediction_summary(evaluation_anchor.tolist()),
        },
        "evaluation_labels_used": False,
        "corrected_prediction_shift": t._guarded_residual_shift_summary(
            oof_corrected, full_corrected,
        ),
    }
    check_deadline()
    return state, base_fits, evaluation_columns, shift


def _metrics(targets: Sequence[float], predictions: Sequence[float]) -> dict[str, Any]:
    if not targets or len(targets) != len(predictions):
        raise ValueError("invalid_honest_evidence")
    errors = [abs(float(target) - float(prediction)) for target, prediction in zip(targets, predictions)]
    result = {
        "count": len(errors), "mae_g": math.fsum(errors) / len(errors),
        "above_5g_count": sum(error > 5.0 for error in errors),
        "prediction_loss": math.fsum(0.1 * error ** 2 + 4.0 * max(error - 4.5, 0.0) ** 2
                                     for error in errors) / len(errors),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("invalid_honest_evidence")
    return result


def _qualified(anchor: Mapping[str, Any], corrected: Mapping[str, Any]) -> bool:
    return (corrected["prediction_loss"] < anchor["prediction_loss"]
            and corrected["mae_g"] <= anchor["mae_g"]
            and corrected["above_5g_count"] <= anchor["above_5g_count"])


def develop(
    output: Path, plan: t.SearchPlan, runtime: t.CandidateRuntime,
    training: Sequence[CanonicalRow], validation: Sequence[CanonicalRow],
    training_fingerprint: str, validation_fingerprint: str,
    started: float, cpu_started: float, deadline: float, clock: Callable[[], float],
) -> t.TuningResult:
    fit_count = 0

    def check_deadline() -> None:
        if clock() >= deadline:
            raise RuntimeError("candidate_search_deadline_reached")

    def before_fit() -> None:
        nonlocal fit_count
        if fit_count >= MAXIMUM_FITS:
            raise RuntimeError("candidate_fit_limit_reached")
        check_deadline()
        fit_count += 1

    honest: dict[str, Any] = {"version": numeric.VERSION, "contract": HONEST_CONTRACT, "seeds": {}}
    blockers: list[str] = []
    status = "blocked"
    first: list[t.CandidateRun] = []
    second: list[t.CandidateRun] = []
    combined: list[dict[str, Any]] = []
    locked = None
    shift_evidence: dict[str, Any] = {"version": numeric.VERSION, "seeds": {}}
    try:
        for seed in (41, 42):
            outer = t._cross_fit_assignments(training, seed, 5)
            fitted_rows = [row for row, fold in zip(training, outer) if fold != 0]
            held_rows = [row for row, fold in zip(training, outer) if fold == 0]
            state, _, columns, shift = _fit_stage(
                runtime, fitted_rows, held_rows, seed, before_fit, check_deadline
            )
            check_deadline()
            _, targets = t.candidate_prediction_matrix(held_rows, base_candidates()[0])
            anchor_predictions = numeric.predict(state, columns, 0.0)[0]
            check_deadline()
            corrected_predictions = numeric.predict(state, columns, 1.0)[0]
            check_deadline()
            anchor = _metrics(targets, anchor_predictions)
            corrected = _metrics(targets, corrected_predictions)
            honest["seeds"][str(seed)] = {
                "anchor": anchor, "corrected": corrected, "qualified": _qualified(anchor, corrected),
                "fitting_record_count": len(fitted_rows), "held_out_record_count": len(held_rows),
                "prediction_distributions": {
                    "anchor": t._prediction_summary(anchor_predictions),
                    "corrected": t._prediction_summary(corrected_predictions),
                },
                "oof_full_fit_shift": shift,
            }
        check_deadline()
        if not all(item["qualified"] for item in honest["seeds"].values()):
            blockers.append("honest_training_correction_gate_failed")
            status = "training_evidence_rejected"
        else:
            second_fits: dict[str, t.LockedFit] = {}
            for seed, runs in ((41, first), (42, second)):
                state, bases, columns, shift = _fit_stage(
                    runtime, training, validation, seed, before_fit, check_deadline
                )
                shift["corrections"] = {}
                shift_evidence["seeds"][str(seed)] = shift
                for scale in (0.0, 1.0):
                    declared = candidate(scale)
                    check_deadline()
                    predictions, corrections = numeric.predict(state, columns, scale)
                    check_deadline()
                    reports = t._explicit_validation_source_reports(len(training), validation, predictions)
                    metrics = t._candidate_metrics(reports)
                    check_deadline()
                    eligible = t._tail_eligible(metrics)
                    metadata = {
                        "fixed_training_counts": dict(FIXED_COUNTS),
                        "correction_scale": scale,
                        "maximum_absolute_correction_g": max(abs(value) for value in corrections),
                        "correction_fit_partition": "training_oof_predictions_only",
                        "validation_labels_used_for_fitting": False,
                        "honest_training_qualified_both_seeds": True,
                    }
                    runs.append(t.CandidateRun(
                        declared, seed, "completed",
                        () if eligible else ("development_serious_error_gate_failed",),
                        eligible, metrics, reports, (metadata,), t._resources(0.0, 0.0),
                    ))
                    shift["corrections"][declared.candidate_id] = t._prediction_summary(corrections)
                    if seed == 42:
                        second_fits[declared.candidate_id] = _fitted_state(bases, state, scale)
            check_deadline()
            combined = t._combine_seed_results(first, second)
            check_deadline()
            eligible_results = [item for item in combined if item["eligible"]]
            if not eligible_results:
                blockers.append("no_eligible_candidate")
                status = "completed_no_candidate"
            else:
                selected = min(eligible_results, key=t._combined_rank_key)
                declared = next(candidate(scale) for scale in (0.0, 1.0)
                                if candidate(scale).candidate_id == selected["candidate_id"])
                check_deadline()
                try:
                    locked = _lock(
                        output / "locked-candidate", declared, second_fits[declared.candidate_id],
                        selected, plan, training, validation, training_fingerprint,
                        validation_fingerprint, honest, shift_evidence,
                    )
                except Exception:
                    raise RuntimeError("candidate_refit_or_lock_failed") from None
                status = "completed"
        if locked is None:
            check_deadline()
    except Exception as error:
        status = "blocked"
        locked = None
        reason = str(error)
        blockers = []
        blockers.append(reason if reason in {"candidate_fit_limit_reached", "candidate_search_deadline_reached",
                                             "candidate_refit_or_lock_failed"}
                        else "candidate_runtime_failed")
    write_private_json(output / "honest-training-evidence.json", honest)
    write_private_json(output / "oof-full-fit-shift.json", shift_evidence)
    result = t.TuningResult(
        status, tuple(blockers), len(first) + len(second),
        {"neural_network": 0, "xgboost": 0, "ensemble": len(first), "second_seed": len(second), "control": 0},
        plan, None, tuple(first), tuple(second), tuple(combined), locked,
        {**t._resources(max(0.0, clock() - started), time.process_time() - cpu_started),
         "model_fits": fit_count, "maximum_model_fits": MAXIMUM_FITS},
    )
    t._write_tuning_outputs(output, result)
    return result


def _predictor(
    bases: Sequence[BatchPredictor], state: Mapping[str, Any], scale: float,
) -> BatchPredictor:
    if len(bases) != 2 or not numeric.valid_state(state):
        raise ValueError("invalid_tail_correction_state")
    def predict(rows: Sequence[tuple[float, ...]]) -> Sequence[float]:
        return numeric.predict(state, [t._predict(base, rows) for base in bases], scale)[0]
    return predict


def _fitted_state(bases: Sequence[t.LockedFit], state: Mapping[str, Any], scale: float) -> t.LockedFit:
    artifacts: dict[str, bytes] = {}
    preprocessing: dict[str, Any] = {"numeric_state": copy.deepcopy(dict(state))}
    inventory: dict[str, list[str]] = {}
    for index, fitted in enumerate(bases, 1):
        preprocessing[f"base_{index}"] = copy.deepcopy(fitted.preprocessing_state)
        inventory[f"base_{index}"] = sorted(fitted.artifacts)
        for name, content in fitted.artifacts.items():
            if not name or Path(name).name != name or not isinstance(content, bytes):
                raise ValueError("invalid_locked_candidate_artifact")
            artifacts[f"base-{index:02d}-{name}"] = content
    return t.LockedFit(_predictor([base.predictor for base in bases], state, scale),
                       preprocessing, artifacts, {"seed": 42, "artifact_inventory": inventory})


def _assignments(identities: Sequence[str], seed: int) -> list[int]:
    return list(t._cross_fit_assignments_for_identities(identities, seed, 5))


def _split_evidence(identities: Sequence[str]) -> dict[str, Any]:
    seeds = {}
    for seed in (41, 42):
        outer = _assignments(identities, seed)
        fitted_indices = [i for i, fold in enumerate(outer) if fold != 0]
        seeds[str(seed)] = {
            "outer_assignments": outer,
            "honest_inner_training_indices": fitted_indices,
            "honest_inner_assignments": _assignments([identities[i] for i in fitted_indices], seed),
            "production_oof_assignments": _assignments(identities, seed),
        }
    evidence = {"training_record_identities": list(identities), "seeds": seeds}
    return {**evidence, "fingerprint": fingerprint(evidence)}


def specification(
    declared: t.Candidate, fixed_counts: Mapping[str, int | float],
) -> TailCorrectionModelSpecification:
    scale = declared.parameters.get("correction_scale")
    if (not isinstance(scale, (int, float)) or isinstance(scale, bool)
            or dict(fixed_counts) != FIXED_COUNTS or declared != candidate(scale)):
        raise ValueError("invalid_model_specification")
    neural, xgboost = base_candidates()
    return TailCorrectionModelSpecification(
        (t._locked_model_specification(neural, FIXED_COUNTS),
         t._locked_model_specification(xgboost, FIXED_COUNTS)),
        float(declared.parameters["correction_scale"]),
    )


def _feature_contract() -> dict[str, Any]:
    return {
        "ordered_features": list(t.LEGACY_FEATURES), "dtype": "float32",
        "transformation_version": numeric.VERSION,
        "base_transformation_version": t.TRANSFORMATION_VERSION,
        "numeric_contract": copy.deepcopy(numeric.CONTRACT),
    }


def _data_usage() -> dict[str, str]:
    return {
        "fitting": "training_records_only", "preprocessing": "training_records_only",
        "early_stopping": "not_used_fixed_training_counts",
        "correction_fitting": "training_oof_predictions_only",
        "honest_evaluation": "training_outer_holdout_only_before_validation",
        "candidate_selection": "validation_scoring_only_after_honest_qualification",
    }


def _lock(
    directory: Path, declared: t.Candidate, fitted: t.LockedFit,
    selected: Mapping[str, Any], plan: t.SearchPlan,
    training: Sequence[CanonicalRow], validation: Sequence[CanonicalRow],
    training_fingerprint: str, validation_fingerprint: str,
    honest: Mapping[str, Any], shift: Mapping[str, Any],
) -> t.LockedCandidate:
    identities = [str(row.metadata["record_identity"]) for row in training]
    contract = {
        "version": t.TUNING_VERSION, "development_contract": "explicit_train_validation",
        "candidate": asdict(declared),
        "model_specification": specification(declared, FIXED_COUNTS).to_dict(),
        "selection_seeds": [41, 42], "seed_weighting": "equal_weight_each_seed",
        "lock_prediction": HONEST_CONTRACT["lock_prediction"],
        "selected_combined_development_evidence": copy.deepcopy(dict(selected)),
        "search_plan": plan.to_dict(),
        "development_evidence": {
            "search_plan_id": plan.plan_id, "search_plan_fingerprint": fingerprint(plan.to_dict()),
            "training_input_fingerprint": training_fingerprint,
            "validation_input_fingerprint": validation_fingerprint,
            "normalized_training_fingerprint": fingerprint([asdict(row) for row in training]),
            "normalized_validation_fingerprint": fingerprint([asdict(row) for row in validation]),
            "partition_identity_fingerprint": plan.source_allocation_fingerprint,
            "test_input_attestation": "no_test_argument_or_path_available",
            "validation_grouping_contract": t.EXPLICIT_DEVELOPMENT_EVIDENCE_VERSION,
            "honest_training_fingerprint": fingerprint(honest),
            "oof_full_fit_shift_fingerprint": fingerprint(shift),
        },
        "development_source_groups": sorted({str(row.metadata["anonymous_source_group"])
                                             for row in (*training, *validation)}),
        "development_data_usage": _data_usage(),
        "feature_contract": _feature_contract(),
        "features": list(t.LEGACY_FEATURES), "fixed_training_counts": dict(FIXED_COUNTS),
        "honest_contract": copy.deepcopy(HONEST_CONTRACT), "split_evidence": _split_evidence(identities),
        "eligibility_rule": t._eligibility_gates(), "ranking_rule": list(plan.ranking_rule),
        "dependency_versions": dict(plan.dependency_versions),
        "dependency_environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": dict(plan.dependency_versions),
        },
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
    write_private_json(directory / "oof-full-fit-shift.json", shift)
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
        raise ValueError("invalid_tail_correction_lock")
    return t.LockedCandidate(declared, fitted.predictor, directory, manifest, contract)


def valid_contract(contract: Mapping[str, Any]) -> bool:
    """Reconstruct every fixed choice, rather than trusting self-consistent JSON."""
    try:
        declared = candidate(contract["candidate"]["parameters"]["correction_scale"])
        normalized = json.loads(json.dumps(contract))
        evidence = contract["development_evidence"]
        plan = contract["search_plan"]
        expected_plan = t.generate_search_plan(
            t.SearchLimits.for_plan(41, PLAN_KIND), input_fingerprint=plan["input_fingerprint"],
            code_fingerprint=plan["code_fingerprint"], dependency_versions=plan["dependency_versions"],
            normalized_input_fingerprint=plan["normalized_input_fingerprint"],
            source_allocation_fingerprint=plan["source_allocation_fingerprint"],
            configuration_fingerprint=plan["configuration_fingerprint"],
            code_configuration_fingerprint=plan["code_configuration_fingerprint"],
        )
        identities = contract["split_evidence"]["training_record_identities"]
        selected = contract["selected_combined_development_evidence"]
        return (
            normalized["candidate"] == json.loads(json.dumps(asdict(declared)))
            and normalized["model_specification"] == specification(declared, FIXED_COUNTS).to_dict()
            and normalized["search_plan"] == json.loads(json.dumps(expected_plan.to_dict()))
            and contract["version"] == t.TUNING_VERSION
            and contract["development_contract"] == "explicit_train_validation"
            and contract["output_unit"] == "g"
            and contract["selection_seeds"] == [41, 42]
            and contract["seed_weighting"] == "equal_weight_each_seed"
            and contract["lock_prediction"] == HONEST_CONTRACT["lock_prediction"]
            and contract["runtime_metadata"]["seed"] == 42
            and contract["fixed_training_counts"] == FIXED_COUNTS
            and contract["feature_contract"] == _feature_contract()
            and contract["features"] == list(t.LEGACY_FEATURES)
            and contract["development_data_usage"] == _data_usage()
            and contract["honest_contract"] == HONEST_CONTRACT
            and isinstance(identities, list) and len(identities) >= 7
            and all(isinstance(value, str) and value for value in identities)
            and len(set(identities)) == len(identities)
            and contract["split_evidence"] == _split_evidence(identities)
            and contract["refit_record_count"] == len(identities)
            and contract["refit_partition"] == "training_records_only"
            and contract["validation_record_count"] > 0
            and contract["test_input_accessed"] is False and contract["final_test_access"] is False
            and contract["eligibility_rule"] == t._eligibility_gates()
            and contract["ranking_rule"] == list(expected_plan.ranking_rule)
            and contract["dependency_versions"] == expected_plan.dependency_versions
            and contract["dependency_environment"]["versions"] == expected_plan.dependency_versions
            and contract["code_fingerprint"] == expected_plan.code_fingerprint
            and evidence["search_plan_id"] == expected_plan.plan_id
            and evidence["search_plan_fingerprint"] == fingerprint(plan)
            and evidence["partition_identity_fingerprint"] == expected_plan.source_allocation_fingerprint
            and evidence["test_input_attestation"] == "no_test_argument_or_path_available"
            and evidence["validation_grouping_contract"] == t.EXPLICIT_DEVELOPMENT_EVIDENCE_VERSION
            and all(isinstance(evidence[key], str) and evidence[key] for key in (
                "training_input_fingerprint", "validation_input_fingerprint",
                "normalized_training_fingerprint", "normalized_validation_fingerprint"))
            and isinstance(contract["development_source_groups"], list)
            and bool(contract["development_source_groups"])
            and contract["development_source_groups"] == sorted(set(contract["development_source_groups"]))
            and selected["candidate_id"] == declared.candidate_id and selected["eligible"] is True
            and selected["family"] == FAMILY and selected["blockers"] == []
            and selected["seed_results"] == [41, 42] and selected["seed_eligibility"] == [True, True]
            and selected["equal_seed_weight"] == 0.5
            and t._valid_candidate_metrics(selected["metrics"]) and t._tail_eligible(selected["metrics"])
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _valid_summary(summary: Mapping[str, Any], count: int, *, shift: bool = False) -> bool:
    keys = ({"count", "mean_signed_shift_g", "mean_absolute_shift_g", "root_mean_square_shift_g",
             "median_absolute_shift_g", "p95_absolute_shift_g", "maximum_absolute_shift_g"}
            if shift else {"count", "mean_g", "standard_deviation_g", "minimum_g", "median_g", "maximum_g"})
    if (set(summary) != keys or summary["count"] != count
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) for value in summary.values())):
        return False
    if shift:
        return all(value >= 0 for key, value in summary.items() if key != "mean_signed_shift_g")
    return (summary["standard_deviation_g"] >= 0
            and summary["minimum_g"] <= summary["median_g"] <= summary["maximum_g"])


def _valid_shift(shift: Mapping[str, Any], training_count: int, evaluation_count: int) -> bool:
    return (
        shift["evaluation_labels_used"] is False
        and set(shift["base_prediction_shift"]) == {"base_1", "base_2"}
        and all(_valid_summary(item, training_count, shift=True)
                for item in shift["base_prediction_shift"].values())
        and _valid_summary(shift["anchor_shift"], training_count, shift=True)
        and _valid_summary(shift["corrected_prediction_shift"], training_count, shift=True)
        and set(shift["anchor_distributions"]) == {"training_oof", "full_fit_training", "evaluation"}
        and all(_valid_summary(item, evaluation_count if key == "evaluation" else training_count)
                for key, item in shift["anchor_distributions"].items())
    )


def verify_state_files(directory: Path, contract: Mapping[str, Any], preprocessing: Mapping[str, Any]) -> bool:
    try:
        honest = json.loads((directory / "honest-training-evidence.json").read_text())
        shift = json.loads((directory / "oof-full-fit-shift.json").read_text())
        inventory = contract["runtime_metadata"]["artifact_inventory"]
        expected_files = {"candidate-contract.json", "preprocessing-state.json", "lock-manifest.json",
                          "honest-training-evidence.json", "oof-full-fit-shift.json"}
        if set(inventory) != {"base_1", "base_2"}:
            return False
        for index in (1, 2):
            names = inventory[f"base_{index}"]
            if (not isinstance(names, list) or not names or names != sorted(set(names))
                    or any(not isinstance(name, str) or not name or Path(name).name != name for name in names)):
                return False
            expected_files.update(f"base-{index:02d}-{name}" for name in names)
        if {path.name for path in directory.iterdir()} != expected_files:
            return False
        if (set(preprocessing) != {"base_1", "base_2", "numeric_state"}
                or not all(isinstance(value, Mapping) for value in preprocessing.values())
                or not numeric.valid_state(preprocessing["numeric_state"])
                or honest["contract"] != HONEST_CONTRACT or honest["version"] != numeric.VERSION
                or set(honest["seeds"]) != {"41", "42"}
                or shift["version"] != numeric.VERSION or set(shift["seeds"]) != {"41", "42"}
                or fingerprint(honest) != contract["development_evidence"]["honest_training_fingerprint"]
                or fingerprint(shift) != contract["development_evidence"]["oof_full_fit_shift_fingerprint"]):
            return False
        for index, base in enumerate(base_candidates(), 1):
            validate_component_preprocessing(
                t._locked_model_specification(base, FIXED_COUNTS), preprocessing[f"base_{index}"]
            )
        for seed in ("41", "42"):
            item = honest["seeds"][seed]
            training_count = contract["refit_record_count"]
            validation_count = contract["validation_record_count"]
            held_count = contract["split_evidence"]["seeds"][seed]["outer_assignments"].count(0)
            fitted_count = training_count - held_count
            if (item["qualified"] is not True or not _qualified(item["anchor"], item["corrected"])
                    or item["held_out_record_count"] != held_count
                    or item["fitting_record_count"] != fitted_count
                    or not _valid_shift(item["oof_full_fit_shift"], fitted_count, held_count)
                    or set(item["prediction_distributions"]) != {"anchor", "corrected"}
                    or not all(_valid_summary(summary, held_count)
                               for summary in item["prediction_distributions"].values())
                    or not _valid_shift(shift["seeds"][seed], training_count, validation_count)):
                return False
            corrections = shift["seeds"][seed]["corrections"]
            if set(corrections) != {candidate(scale).candidate_id for scale in (0.0, 1.0)}:
                return False
            for scale in (0.0, 1.0):
                summary = corrections[candidate(scale).candidate_id]
                if (not _valid_summary(summary, validation_count)
                        or summary["minimum_g"] < -2.0 * scale or summary["maximum_g"] > 2.0 * scale):
                    return False
            for metrics in (item["anchor"], item["corrected"]):
                if (set(metrics) != {"count", "mae_g", "above_5g_count", "prediction_loss"}
                        or not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                                   for v in metrics.values()) or metrics["count"] != held_count
                        or not isinstance(metrics["above_5g_count"], int)
                        or metrics["above_5g_count"] > held_count):
                    return False
        return True
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return False


def load(models: ModelRuntime, directory: Path, contract: Mapping[str, Any]) -> BatchPredictor:
    preprocessing = json.loads((directory / "preprocessing-state.json").read_text())
    bases = []
    declared = t._candidate_from_dict(contract["candidate"])
    resolved = specification(declared, contract["fixed_training_counts"])
    for index, base in enumerate(resolved.bases, 1):
        names = contract["runtime_metadata"]["artifact_inventory"][f"base_{index}"]
        artifacts = {name: (directory / f"base-{index:02d}-{name}").read_bytes() for name in names}
        bases.append(models.load_verified_component(base, artifacts, preprocessing[f"base_{index}"]))
    return _predictor(bases, preprocessing["numeric_state"],
                      contract["candidate"]["parameters"]["correction_scale"])
