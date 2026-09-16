"""Bounded training-only paired accounting for the unchanged tail correction.

This is observational instrumentation, not a candidate, estimator, or gate.
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
from ..ingestion import Dataset, InputError, fingerprint, load_records, normalize
from ..private_io import PrivateArgumentParser, write_private_json
from . import tail_correction as numeric
from . import tail_focused_search as tail
from . import training_stability as stability
from . import tuning as t

VERSION = "minires-correction-transition-diagnostic-v1"
PREDECLARED_OUTPUT_ROOT = stability.PROJECT_ROOT / "private" / "candidate-tuning" / "run-016"
MAXIMUM_MODEL_FITS = 52
MAXIMUM_ELAPSED_SECONDS = 7200.0
TRANSITIONS = ("stable_nonserious", "harm", "repair", "persistent_serious")
BINS = ("le4", "gt4_le5", "gt5_le7", "gt7")
ACCOUNTING_CONTRACT = {
    "residual": "prediction_minus_target; negative_underestimate; positive_overestimate",
    "serious": "absolute_error_strictly_greater_than_5; exactly_plus_or_minus_5_is_nonserious",
    "transitions": dict(zip(TRANSITIONS, ("not_above5_to_not_above5", "not_above5_to_above5",
                                          "above5_to_not_above5", "above5_to_above5"))),
    "anchor_absolute_error_bins": dict(zip(BINS, ("[0,4]", "(4,5]", "(5,7]", "(7,infinity)"))),
    "prediction_loss": "0.1*error_squared+4*max(abs(error)-4.5,0)_squared; no_fit_penalties",
    "contributions": "sums_divided_by_entire_cell_count_not_subgroup_count; delta_corrected_minus_anchor",
    "correction_direction": "toward_if_anchor_residual_times_correction_negative; away_if_positive; otherwise_neutral",
    "oracle": "label_aware_theoretical_only; minimum_abs_error=max(anchor_abs_error-2,0); not_an_estimator",
    "conservation": "counts_exact; floating_sums_rel_tol_1e-12_abs_tol_1e-12",
    "interpretation_rules": {
        "near_threshold_harm_offsets_repairs": "harm_count>=repair_count_and_harm_count>0_and_more_than_half_of_harms_have_anchor_abs_error_in_(4,5]",
        "severe_tails_dominate_excess_loss": "anchor_gt7_weighted_excess_loss_contribution_strictly_exceeds_half_of_total_anchor_weighted_excess_loss",
        "bound_cannot_repair_gt7": "theorem_for_all_rows; gt7_oracle_serious_count_equals_gt7_count; exactly_7_can_reach_5",
        "scope": "per_cell_descriptive_flags_only; no_causal_claim_or_automatic_intervention",
    },
}


def _finite_tree(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, (float, int)) and not math.isfinite(value):
        raise ValueError("nonfinite_correction_transition_evidence")


def summarize_correction_transitions(
    targets: Sequence[float], anchor: Sequence[float], corrected: Sequence[float],
) -> dict[str, Any]:
    """Reduce paired vectors immediately to fixed source-neutral aggregate evidence."""
    import numpy as np

    matrix = np.asarray((targets, anchor, corrected), dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != 3 or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()):
        raise ValueError("invalid_correction_transition_vectors")
    n = matrix.shape[1]
    # Python arithmetic and fsum fail on overflow instead of silently emitting infinity.
    errors = [(float(a) - float(y), float(c) - float(y), float(c) - float(a))
              for y, a, c in matrix.T]
    _finite_tree(errors)
    # Addition may round by a few ulps; do not mistake that for changing the model bound.
    if any(abs(d) > 2.0 + 4 * max(math.ulp(float(a)), math.ulp(float(c)))
           for (_, _, d), a, c in zip(errors, matrix[1], matrix[2])):
        raise ValueError("correction_bound_exceeded")
    transition = [TRANSITIONS[2 * int(abs(a) > 5) + int(abs(c) > 5)] for a, c, _ in errors]
    bins = [BINS[sum(abs(a) > edge for edge in (4, 5, 7))] for a, _, _ in errors]

    def group(indices: Sequence[int]) -> dict[str, Any]:
        pairs = [errors[i] for i in indices]
        result: dict[str, Any] = {"count": len(pairs)}
        for column, name in enumerate(("anchor", "corrected")):
            values = [pair[column] for pair in pairs]
            ordinary = math.fsum(0.1 * e ** 2 for e in values) / n
            excess = math.fsum(4.0 * max(abs(e) - 4.5, 0.0) ** 2 for e in values) / n
            result[name] = {
                "mae_contribution_g": math.fsum(abs(e) for e in values) / n,
                "ordinary_loss_contribution": ordinary,
                "excess_loss_contribution": excess,
                "prediction_loss_contribution": ordinary + excess,
                "above_5g_count": sum(abs(e) > 5 for e in values),
                "residual_sign_counts": {"negative": sum(e < 0 for e in values),
                                         "zero": sum(e == 0 for e in values),
                                         "positive": sum(e > 0 for e in values)},
            }
        result["delta"] = {key: result["corrected"][key] - result["anchor"][key]
                           for key in ("mae_contribution_g", "ordinary_loss_contribution",
                                       "excess_loss_contribution", "prediction_loss_contribution",
                                       "above_5g_count")}
        corrections = [d for _, _, d in pairs]
        result["correction"] = {
            "signed_sum_g": math.fsum(corrections),
            "absolute_sum_g": math.fsum(abs(d) for d in corrections),
            "maximum_absolute_g": max((abs(d) for d in corrections), default=0.0),
            "negative_count": sum(d < 0 for d in corrections),
            "zero_count": sum(d == 0 for d in corrections),
            "positive_count": sum(d > 0 for d in corrections),
            "toward_count": sum((a < 0 < d) or (d < 0 < a) for a, _, d in pairs),
            "away_count": sum((a < 0 and d < 0) or (a > 0 and d > 0) for a, _, d in pairs),
            "neutral_count": sum(a == 0 or d == 0 for a, _, d in pairs),
            "absolute_error_improved_count": sum(abs(c) < abs(a) for a, c, _ in pairs),
            "absolute_error_worsened_count": sum(abs(c) > abs(a) for a, c, _ in pairs),
            "absolute_error_unchanged_count": sum(abs(c) == abs(a) for a, c, _ in pairs),
        }
        result["transition_counts"] = {key: sum(transition[i] == key for i in indices) for key in TRANSITIONS}
        oracle_errors = [max(abs(a) - 2.0, 0.0) for a, _, _ in pairs]
        result["oracle"] = {
            "minimum_mae_contribution_g": math.fsum(oracle_errors) / n,
            "minimum_prediction_loss_contribution": math.fsum(
                0.1 * e ** 2 + 4 * max(e - 4.5, 0.0) ** 2 for e in oracle_errors) / n,
            # Compare the original error, avoiding subtraction roundoff at seven.
            "unavoidable_above_5g_count": sum(abs(a) > 7 for a, _, _ in pairs),
            "repairable_serious_count": sum(5 < abs(a) <= 7 for a, _, _ in pairs),
        }
        return result

    total = group(list(range(n)))
    transitions = {key: group([i for i in range(n) if transition[i] == key]) for key in TRANSITIONS}
    anchor_bins = {key: group([i for i in range(n) if bins[i] == key]) for key in BINS}
    for partition in (transitions, anchor_bins):
        if sum(item["count"] for item in partition.values()) != n:
            raise ValueError("incomplete_correction_transition_evidence")
        for model in ("anchor", "corrected", "delta"):
            for key in total["delta"]:
                expected = total[model][key]
                observed = math.fsum(item[model][key] for item in partition.values())
                if not math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("nonconserving_correction_transition_evidence")
    harms, repairs = transitions["harm"]["count"], transitions["repair"]["count"]
    if total["delta"]["above_5g_count"] != harms - repairs:
        raise ValueError("nonconserving_correction_transition_evidence")
    result = {
        "count": n, "total": total, "transitions": transitions, "anchor_error_bins": anchor_bins,
        "descriptive_flags": {
            "near_threshold_harm_offsets_repairs": harms >= repairs and harms > 0
            and 2 * anchor_bins["gt4_le5"]["transition_counts"]["harm"] > harms,
            "severe_tails_dominate_excess_loss": anchor_bins["gt7"]["anchor"]["excess_loss_contribution"]
            > 0.5 * total["anchor"]["excess_loss_contribution"],
            "bound_cannot_repair_gt7": anchor_bins["gt7"]["oracle"]["unavoidable_above_5g_count"]
            == anchor_bins["gt7"]["count"],
        },
    }
    _finite_tree(result)
    return result


def _validate_shift(shift: Mapping[str, Any], fitting_count: int, held_count: int) -> None:
    """Require the existing complete aggregate schema, rejecting extra private fields."""
    if (set(shift) != {"base_prediction_shift", "anchor_shift", "corrected_prediction_shift",
                       "anchor_distributions", "evaluation_labels_used"}
            or shift["evaluation_labels_used"] is not False
            or set(shift["base_prediction_shift"]) != {"base_1", "base_2"}
            or set(shift["anchor_distributions"]) != {"training_oof", "full_fit_training", "evaluation"}):
        raise ValueError("incomplete_shift_evidence")
    shift_keys = set(t._guarded_residual_shift_summary([0.0], [0.0]))
    distribution_keys = set(t._prediction_summary([0.0]))
    summaries = [(value, shift_keys, fitting_count) for value in (
        *shift["base_prediction_shift"].values(), shift["anchor_shift"], shift["corrected_prediction_shift"]
    )]
    summaries.extend((value, distribution_keys, held_count if name == "evaluation" else fitting_count)
                     for name, value in shift["anchor_distributions"].items())
    for summary, keys, count in summaries:
        if (set(summary) != keys or summary["count"] != count
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in summary.values())):
            raise ValueError("invalid_shift_evidence")


@dataclass(frozen=True)
class CorrectionTransitionDiagnosticResult:
    status: str
    blockers: tuple[str, ...]
    evidence: Mapping[str, Any]
    resource_use: Mapping[str, Any]


def _plan(raw: str, normalized: str, dependencies: Mapping[str, str]) -> dict[str, Any]:
    plan = copy.deepcopy(stability._plan(raw, normalized))
    plan.update({
        "version": VERSION, "kind": "training_only_correction_transition_diagnostic",
        "hypothesis": "paired_accounting_of_repairs_harms_and_loss_not_causal_inference",
        "accounting_contract": copy.deepcopy(ACCOUNTING_CONTRACT),
        "frozen_model_contract": copy.deepcopy(tail.candidate(1.0).parameters),
        "dependency_versions": dict(dependencies), "python_version": platform.python_version(),
        "required_environment": {"python": "3.13", **t.CROSS_FITTED_GATE_DEPENDENCY_VERSIONS,
                                 "scikit-learn": t.CROSS_FITTED_GATE_SCIKIT_LEARN_VERSION},
        "platform": platform.platform(),
        "expected_training_sha256": t.GUARDED_RESIDUAL_DEVELOPMENT_CHECKSUMS["train.jsonl"],
        "fit_allocation": {"inner_oof_bases": 40, "outer_partition_bases": 8, "corrections": 4},
        "historical_selection": "conditional_on_reused_validation_selected_anchor_and_counts",
        "stop_rule": "one_attempt_no_retries_capacity_recycling_or_intervention_selection",
    })
    return plan


def run_correction_transition_diagnostic(
    training_records: Dataset, config: EvaluationConfig, *, runtime: t.CandidateRuntime,
    output_root: str | Path, clock: Callable[[], float] = time.monotonic,
) -> CorrectionTransitionDiagnosticResult:
    """Execute four fixed training cells; never select, qualify, or lock a model."""
    if (config.seed != stability.NORMALIZATION_SEED or config.volume_unit != "mm3"
            or config.scope_confirmed is not True):
        raise InputError("invalid_correction_transition_configuration")
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
            raise RuntimeError("correction_transition_invalid_clock")
        if now - started >= MAXIMUM_ELAPSED_SECONDS:
            raise RuntimeError("correction_transition_deadline_reached")

    def before_fit() -> None:
        nonlocal fit_count
        check_deadline()
        if fit_count >= MAXIMUM_MODEL_FITS:
            raise RuntimeError("correction_transition_fit_limit_reached")
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
        write_private_json(output / "diagnostic-plan.json", plan)
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
                state, _, columns, shift = tail._fit_stage(
                    runtime, fitted, held, seed, before_fit, check_deadline,
                    assignment_seed=stability.INNER_SPLIT_SEEDS[split_seed],
                )
                check_deadline()
                anchor = numeric.predict(state, columns, 0.0)[0]
                check_deadline()
                corrected = numeric.predict(state, columns, 1.0)[0]
                check_deadline()
                paired = summarize_correction_transitions(targets, anchor, corrected)
                if fit_count - first_fit != 13 or paired["count"] != len(held):
                    raise ValueError("incomplete_cell")
                _validate_shift(shift, len(fitted), len(held))
                check_deadline()
                cells[active_cell] = {
                    "outer_split_seed": split_seed, "model_seed": seed,
                    "inner_split_seed": stability.INNER_SPLIT_SEEDS[split_seed],
                    "fitting_record_count": len(fitted), "held_out_record_count": len(held),
                    "fits": fit_count - first_fit, "paired_evidence": paired,
                    "oof_full_fit_shift": shift,
                }
                active_cell = None
        if len(cells) != 4 or fit_count != MAXIMUM_MODEL_FITS:
            raise ValueError("incomplete_diagnostic")
        check_deadline()
        status = "completed"
    except (InputError, RuntimeError, ValueError, TypeError, OverflowError, OSError, KeyError) as error:
        reason = str(error)
        blockers = (reason if reason in {
            "correction_transition_deadline_reached", "correction_transition_fit_limit_reached",
            "correction_transition_invalid_clock",
        } else "correction_transition_runtime_failed",)
    elapsed = clock() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = 0.0
        status, blockers = "blocked", ("correction_transition_invalid_clock",)
    elif elapsed >= MAXIMUM_ELAPSED_SECONDS:
        status, blockers = "blocked", ("correction_transition_deadline_reached",)
    resources = {"elapsed_seconds": elapsed, "process_cpu_seconds": time.process_time() - cpu_started,
                 "fits_started": fit_count, "fits_in_completed_cells": 13 * len(cells),
                 "unused_fit_capacity": MAXIMUM_MODEL_FITS - fit_count,
                 "maximum_fits": MAXIMUM_MODEL_FITS, "maximum_elapsed_seconds": MAXIMUM_ELAPSED_SECONDS}
    evidence = {
        "version": VERSION, "status": status, "blockers": list(blockers), "cells": cells,
        "failed_cell": active_cell,
        "uncompleted_cells": [stability._cell_key(split, seed)
                              for split in stability.OUTER_SPLIT_SEEDS for seed in stability.MODEL_SEEDS
                              if stability._cell_key(split, seed) not in cells],
        "resource_use": resources, "validation_labels_used": False, "held_out_test_accessed": False,
        "interpretation": "conditional_on_historical_validation_selected_anchor; within_cell_paired; different_splits_not_paired_or_causal; no_intervention_selected",
    }
    _finite_tree(evidence)
    if not (output / "diagnostic-plan.json").exists():
        write_private_json(output / "diagnostic-plan.json", plan)
    write_private_json(output / "correction-transition-evidence.json", evidence)
    files = (output / "diagnostic-plan.json", output / "correction-transition-evidence.json")
    write_private_json(output / "manifest.json", {
        "version": VERSION, "create_only": True,
        "artifacts": {path.name: sha256(path.read_bytes()).hexdigest() for path in files},
        "validation_labels_used": False, "held_out_test_accessed": False, "publication_performed": False,
    })
    return CorrectionTransitionDiagnosticResult(status, blockers, evidence, resources)


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(description="Private training-only correction-transition diagnostic.")
    parser.add_argument("--training-records", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--volume-unit", default="mm3")
    parser.add_argument("--scope-confirmed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_root.resolve() != PREDECLARED_OUTPUT_ROOT.resolve():
            raise InputError("correction_transition_output_root_mismatch")
        if args.output_root.exists():
            raise InputError("private_output_directory_unavailable")
        stability._verify_predeclared_training_artifact(args.training_records)
        runtime = t.TensorflowXGBoostCandidateRuntime()
        t._verify_predeclared_environment(runtime.dependency_versions)
        result = run_correction_transition_diagnostic(
            args.training_records,
            EvaluationConfig(None, args.volume_unit, True if args.scope_confirmed else None,
                             seed=stability.NORMALIZATION_SEED),
            runtime=runtime, output_root=args.output_root,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (ImportError, RuntimeError, OSError, ValueError, TypeError):
        raise SystemExit("correction_transition_diagnostic_failed") from None
    print(json.dumps({"status": result.status, "blockers": result.blockers,
                      "resource_use": result.resource_use}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
