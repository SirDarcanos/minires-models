"""Four-cell training prerequisite only: no candidate/validation continuation."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from ..evaluation import EvaluationConfig
from ..ingestion import CanonicalRow, Dataset, InputError, fingerprint, load_records, normalize
from ..private_io import PrivateArgumentParser, write_private_json
from . import bounded_influence_correction as numeric
from . import correction_transition as transition
from . import tail_focused_search as tail
from . import training_stability as stability
from . import tuning as t
from .definitions import validate_component_preprocessing

VERSION = "minires-bounded-influence-training-prerequisite-v1"
PREDECLARED_OUTPUT_ROOT = stability.PROJECT_ROOT / "private" / "candidate-tuning" / "run-017"
MAXIMUM_MODEL_FITS = 52
MAXIMUM_ELAPSED_SECONDS = 7200.0
QUALIFICATION_CONTRACT = {
    "per_cell": "strictly_lower_new_unpenalized_prediction_loss_and_nonincreasing_mae_and_strict_above5g_count",
    "overall": "all_four_complete_valid_cells_must_pass; no_seed_selection_or_averaging",
    "prediction_loss": "mean(0.1*H_5(error)+4*H_2.5(max(abs(error)-4.5,0)))",
    "success": "training_prerequisite_supported_only; not_validation_eligible",
    "continuation": "none_even_if_all_four_pass; no_production_validation_ranking_selection_lock_or_promotion",
    "metric_failure": "complete_all_remaining_cells_without_replacement",
    "invalid_runtime_deadline": "block_preserve_completed_cells_and_partial_fit_accounting_no_replacement",
}


def _plan(raw: str, normalized: str, dependencies: Mapping[str, str]) -> dict[str, Any]:
    plan = transition._plan(raw, normalized, dependencies)
    # Reuse only the historical base/runtime/preprocessing descriptors, not the old
    # correction contract or its conditional-production honest gate.
    base_contract = copy.deepcopy(dict(tail.candidate(1.0).parameters))
    for key in ("numeric_contract", "honest_contract", "correction_scale"):
        base_contract.pop(key)
    plan.pop("accounting_contract")
    plan.pop("frozen_model_contract")
    plan.update({
        "version": VERSION, "kind": "training_only_bounded_influence_prerequisite",
        "hypothesis": "bounded_ordinary_and_excess_influence_improves_all_four_training_holdouts",
        "correction_contract": copy.deepcopy(numeric.CONTRACT),
        "frozen_base_contract": base_contract,
        "qualification_contract": copy.deepcopy(QUALIFICATION_CONTRACT),
        "old_diagnostic_accounting_contract": copy.deepcopy(transition.ACCOUNTING_CONTRACT),
        "new_loss_accounting": "ordinary_and_excess_components_by_total_transition_and_anchor_bin; contributions_divide_by_cell_count; conservation_tolerance_1e-12",
        "normalization": {"seed": 17, "volume_unit": "mm3", "scope_confirmed": True, "contract": "legacy"},
        "selection": "forbidden_training_prerequisite_only",
        "locking": "forbidden_training_prerequisite_only",
        "stop_rule": "one_attempt_stop_even_if_supported_no_continuation_retry_or_budget_recycling",
    })
    return plan


def _fit_stage(
    runtime: t.CandidateRuntime, training: Sequence[CanonicalRow],
    evaluation: Sequence[CanonicalRow], seed: int, before_fit: Callable[[], None],
    check_deadline: Callable[[], None], *, assignment_seed: int,
) -> tuple[numeric.BoundedInfluenceState, list[tuple[float, ...]], dict[str, Any]]:
    """Small dedicated stage: twelve frozen base fits and exactly one NEW correction.

    The old stage hardcodes its numeric contract. Keeping this stage separate
    avoids changing old lock semantics or fitting/discarding an old correction.
    """
    assignments = t._cross_fit_assignments(training, assignment_seed, 5)
    oof_columns: list[tuple[float, ...]] = []
    full_columns: list[tuple[float, ...]] = []
    evaluation_columns: list[tuple[float, ...]] = []
    for base, count in zip(tail.base_candidates(), (87, 1091)):
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
            validate_component_preprocessing(t._locked_model_specification(base, tail.FIXED_COUNTS),
                                             fitted.preprocessing_state)
            predictions = t._predict(fitted.predictor, held_x)
            check_deadline()
            for index, prediction in zip(held, predictions):
                oof[index] = prediction
        numeric.features((oof, oof))
        oof_columns.append(tuple(oof))
        train_x, train_y = t.candidate_prediction_matrix(training, base)
        evaluate_x, _ = t.candidate_prediction_matrix(evaluation, base)
        before_fit()
        fitted = t._stack_refit(runtime, base, seed, train_x, train_y, count)
        check_deadline()
        validate_component_preprocessing(t._locked_model_specification(base, tail.FIXED_COUNTS),
                                         fitted.preprocessing_state)
        full_columns.append(t._predict(fitted.predictor, train_x))
        check_deadline()
        evaluation_columns.append(t._predict(fitted.predictor, evaluate_x))
        check_deadline()
    _, train_y = t.candidate_prediction_matrix(training, tail.base_candidates()[0])
    before_fit()
    state = numeric.fit_state(oof_columns, train_y)
    check_deadline()
    oof_anchor, _ = numeric.features(oof_columns)
    full_anchor, _ = numeric.features(full_columns)
    evaluation_anchor, _ = numeric.features(evaluation_columns)
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
        "corrected_prediction_shift": t._guarded_residual_shift_summary(oof_corrected, full_corrected),
    }
    check_deadline()
    return state, evaluation_columns, shift


def _new_loss_accounting(targets: Sequence[float], anchor: Sequence[float], corrected: Sequence[float]) -> dict[str, Any]:
    """New qualification loss, explicitly separate from run-016's OLD diagnostics."""
    import numpy as np

    vectors = np.asarray((targets, anchor, corrected), dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != 3 or not vectors.shape[1]:
        raise ValueError("invalid_qualification_vectors")
    for vector in vectors:
        numeric._vector(vector)
    errors = vectors[1:] - vectors[0]
    components = [numeric.prediction_loss_components(error) for error in errors]
    n = len(targets)
    transitions = 2 * (np.abs(errors[0]) > 5).astype(int) + (np.abs(errors[1]) > 5).astype(int)
    bins = sum((np.abs(errors[0]) > edge).astype(int) for edge in (4, 5, 7))

    def group(mask: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"count": int(np.sum(mask))}
        for index, model in enumerate(("anchor", "corrected")):
            ordinary, excess = (math.fsum(values[mask].tolist()) / n for values in components[index])
            result[model] = {"ordinary_loss_contribution": ordinary, "excess_loss_contribution": excess,
                             "prediction_loss_contribution": ordinary + excess}
        result["delta"] = {key: result["corrected"][key] - result["anchor"][key] for key in result["anchor"]}
        return result

    total = group(np.ones(n, dtype=bool))
    result = {"total": total,
              "transitions": {key: group(transitions == i) for i, key in enumerate(transition.TRANSITIONS)},
              "anchor_error_bins": {key: group(bins == i) for i, key in enumerate(transition.BINS)}}
    for partition in (result["transitions"], result["anchor_error_bins"]):
        if sum(item["count"] for item in partition.values()) != n:
            raise ValueError("incomplete_new_loss_accounting")
        for model in ("anchor", "corrected", "delta"):
            for key in total[model]:
                if not math.isclose(math.fsum(item[model][key] for item in partition.values()),
                                    total[model][key], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("nonconserving_new_loss_accounting")
    transition._finite_tree(result)
    return result


def _qualification(paired: Mapping[str, Any], new_loss: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {
        "strictly_lower_new_prediction_loss": new_loss["total"]["corrected"]["prediction_loss_contribution"]
        < new_loss["total"]["anchor"]["prediction_loss_contribution"],
        "nonincreasing_mae": paired["total"]["delta"]["mae_contribution_g"] <= 0,
        "nonincreasing_above_5g_count": paired["total"]["delta"]["above_5g_count"] <= 0,
    }
    return {"conditions": conditions, "qualified": all(conditions.values())}


@dataclass(frozen=True)
class BoundedInfluenceTrainingResult:
    status: str
    blockers: tuple[str, ...]
    evidence: Mapping[str, Any]
    resource_use: Mapping[str, Any]


def run_bounded_influence_training(
    training_records: Dataset, config: EvaluationConfig, *, runtime: t.CandidateRuntime,
    output_root: str | Path, clock: Callable[[], float] = time.monotonic,
) -> BoundedInfluenceTrainingResult:
    """Qualify all four honest training cells, then STOP regardless of the outcome."""
    if (config.seed != stability.NORMALIZATION_SEED or config.volume_unit != "mm3"
            or config.scope_confirmed is not True):
        raise InputError("invalid_bounded_influence_configuration")
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
    status = "blocked"
    blockers: tuple[str, ...] = ()
    plan = _plan("unavailable", "unavailable", runtime.dependency_versions)

    def check_deadline() -> None:
        now = clock()
        if not math.isfinite(now) or not math.isfinite(started) or now < started:
            raise RuntimeError("bounded_influence_invalid_clock")
        if now - started >= MAXIMUM_ELAPSED_SECONDS:
            raise RuntimeError("bounded_influence_deadline_reached")

    def before_fit() -> None:
        nonlocal fit_count
        check_deadline()
        if fit_count >= MAXIMUM_MODEL_FITS:
            raise RuntimeError("bounded_influence_fit_limit_reached")
        fit_count += 1

    try:
        check_deadline()
        loaded, raw = load_records(training_records)
        training = normalize(loaded, config, contract="legacy")
        identities = [str(row.metadata.get("record_identity", "")) for row in training]
        if (not training or not t._valid_candidate_feature_data(training, t.LEGACY_FEATURES)
                or any(row.outcome != "included" or not identity
                       or not row.metadata.get("anonymous_source_group")
                       for row, identity in zip(training, identities))
                or len(identities) != len(set(identities))):
            raise ValueError("invalid_training_records")
        plan = _plan(raw, fingerprint([asdict(row) for row in training]), runtime.dependency_versions)
        write_private_json(output / "prerequisite-plan.json", plan)
        for split_seed in stability.OUTER_SPLIT_SEEDS:
            outer = t._cross_fit_assignments(training, split_seed, stability.FOLDS)
            fitted = [row for row, fold in zip(training, outer) if fold != 0]
            held = [row for row, fold in zip(training, outer) if fold == 0]
            if not fitted or not held:
                raise ValueError("invalid_training_partition")
            _, targets = t.candidate_prediction_matrix(held, tail.base_candidates()[0])
            for seed in stability.MODEL_SEEDS:
                active_cell = stability._cell_key(split_seed, seed)
                first_fit = fit_count
                state, columns, shift = _fit_stage(
                    runtime, fitted, held, seed, before_fit, check_deadline,
                    assignment_seed=stability.INNER_SPLIT_SEEDS[split_seed],
                )
                check_deadline()
                anchor = numeric.predict(state, columns, 0.0)[0]
                check_deadline()
                corrected = numeric.predict(state, columns, 1.0)[0]
                check_deadline()
                paired = transition.summarize_correction_transitions(targets, anchor, corrected)
                new_loss = _new_loss_accounting(targets, anchor, corrected)
                if fit_count - first_fit != 13 or paired["count"] != len(held):
                    raise ValueError("incomplete_cell")
                transition._validate_shift(shift, len(fitted), len(held))
                check_deadline()
                cells[active_cell] = {
                    "outer_split_seed": split_seed, "model_seed": seed,
                    "inner_split_seed": stability.INNER_SPLIT_SEEDS[split_seed],
                    "fitting_record_count": len(fitted), "held_out_record_count": len(held),
                    "fits": fit_count - first_fit,
                    "paired_transitions_with_old_diagnostic_loss": paired,
                    "new_qualification_loss": new_loss,
                    "qualification": _qualification(paired, new_loss),
                    "oof_full_fit_shift": shift,
                }
                active_cell = None
        if len(cells) != 4 or fit_count != MAXIMUM_MODEL_FITS:
            raise ValueError("incomplete_prerequisite")
        check_deadline()
        if all(cell["qualification"]["qualified"] for cell in cells.values()):
            status = "training_prerequisite_supported"
        else:
            status, blockers = "training_evidence_rejected", ("four_cell_training_prerequisite_failed",)
    except Exception as error:
        # Third-party failures must retain completed cells without publishing raw errors.
        reason = str(error)
        blockers = (reason if reason in {
            "bounded_influence_deadline_reached", "bounded_influence_fit_limit_reached",
            "bounded_influence_invalid_clock",
        } else "bounded_influence_runtime_failed",)
    elapsed = clock() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = 0.0
        status, blockers = "blocked", ("bounded_influence_invalid_clock",)
    elif elapsed >= MAXIMUM_ELAPSED_SECONDS:
        status, blockers = "blocked", ("bounded_influence_deadline_reached",)
    resources = {"elapsed_seconds": elapsed, "process_cpu_seconds": time.process_time() - cpu_started,
                 "fits_started": fit_count, "fits_in_completed_cells": 13 * len(cells),
                 "unused_fit_capacity": MAXIMUM_MODEL_FITS - fit_count,
                 "maximum_fits": MAXIMUM_MODEL_FITS, "maximum_elapsed_seconds": MAXIMUM_ELAPSED_SECONDS}
    evidence = {
        "version": VERSION, "status": status, "blockers": list(blockers), "cells": cells,
        "runtime_failed_cell": active_cell,
        "metric_failed_cells": [key for key, cell in cells.items() if not cell["qualification"]["qualified"]],
        "uncompleted_cells": [stability._cell_key(split, seed)
                              for split in stability.OUTER_SPLIT_SEEDS for seed in stability.MODEL_SEEDS
                              if stability._cell_key(split, seed) not in cells],
        "resource_use": resources, "validation_labels_used": False, "held_out_test_accessed": False,
        "validation_eligible": False, "production_continuation": False,
        "qualification_contract": copy.deepcopy(QUALIFICATION_CONTRACT),
        "interpretation": "conditional_on_historical_validation_selected_anchor_and_counts; within_cell_paired; different_splits_not_paired_or_causal; training_prerequisite_only",
    }
    transition._finite_tree(evidence)
    if not (output / "prerequisite-plan.json").exists():
        write_private_json(output / "prerequisite-plan.json", plan)
    write_private_json(output / "bounded-influence-evidence.json", evidence)
    files = (output / "prerequisite-plan.json", output / "bounded-influence-evidence.json")
    write_private_json(output / "manifest.json", {
        "version": VERSION, "create_only": True,
        "artifacts": {path.name: sha256(path.read_bytes()).hexdigest() for path in files},
        "validation_labels_used": False, "held_out_test_accessed": False, "publication_performed": False,
        "validation_eligible": False, "production_continuation": False,
    })
    return BoundedInfluenceTrainingResult(status, blockers, evidence, resources)


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(description="Private bounded-influence training prerequisite only.")
    parser.add_argument("--training-records", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--volume-unit", default="mm3")
    parser.add_argument("--scope-confirmed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_root.resolve() != PREDECLARED_OUTPUT_ROOT.resolve():
            raise InputError("bounded_influence_output_root_mismatch")
        if args.output_root.exists():
            raise InputError("private_output_directory_unavailable")
        stability._verify_predeclared_training_artifact(args.training_records)
        runtime = t.TensorflowXGBoostCandidateRuntime()
        t._verify_predeclared_environment(runtime.dependency_versions)
        result = run_bounded_influence_training(
            args.training_records,
            EvaluationConfig(None, args.volume_unit, True if args.scope_confirmed else None,
                             seed=stability.NORMALIZATION_SEED),
            runtime=runtime, output_root=args.output_root,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (ImportError, RuntimeError, OSError, ValueError, TypeError):
        raise SystemExit("bounded_influence_training_failed") from None
    print(json.dumps({"status": result.status, "blockers": result.blockers,
                      "resource_use": result.resource_use}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
