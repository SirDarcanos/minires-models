"""Fixed training-only descriptive bins of the frozen correction's actual inputs.

Severe cohorts have only 22–26 historical rows: three-dimensional bins are sparse,
not extra samples. Associations cannot establish a deployable gate or no signal.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from ..evaluation import EvaluationConfig
from ..ingestion import Dataset, InputError, fingerprint, load_records, normalize
from ..private_io import PrivateArgumentParser, write_private_json
from . import bounded_influence_correction as numeric
from . import bounded_influence_training as prerequisite
from . import correction_transition as transition
from . import tail_focused_search as tail
from . import training_stability as stability
from . import tuning as t

VERSION = "minires-bounded-influence-feature-signal-v1"
PREDECLARED_OUTPUT_ROOT = stability.PROJECT_ROOT / "private" / "candidate-tuning" / "run-018"
MAXIMUM_MODEL_FITS = 52
MAXIMUM_ELAPSED_SECONDS = 7200.0
FEATURE_BINS = ("low", "middle", "high")
MINIMUM_SUPPORT = 20
INTERPRETATION = (
    "training_only_descriptive; conditional_on_historically_validation_selected_anchor_and_counts; "
    "cohorts_nested_do_not_sum; marginal_views_repeat_rows; overlapping_holdouts_not_independent; "
    "historical_gt7_cohorts_22_to_26_make_joint_bins_sparse_not_extra_samples; "
    "empty_or_insufficient_support_is_not_evidence_of_no_signal; "
    "associations_not_predictive_or_causal_proof; cannot_prove_out_of_sample_gating_or_rule_out_nonlinear_signal; "
    "no_automatic_gate_selection_qualification_or_continuation"
)
ACCOUNTING_CONTRACT = {
    "features": list(numeric.FEATURES),
    "representation": "exact_frozen_state_training_oof_means_scales_then_clip_to_[-1,1]; no_intercept_bin",
    "feature_bins": dict(zip(FEATURE_BINS, ("[-1,-1/3]", "(-1/3,1/3]", "(1/3,1]"))),
    "boundary_arithmetic": "float64_edges_-1.0/3.0_and_1.0/3.0; equality_belongs_to_lower_bin",
    "views": "each_of_three_marginals_has_all_three_bins; joint_has_all_27_in_feature_order; include_empty",
    "cohorts": {"all": "all_outer_held_rows", "anchor_gt5": "abs(anchor-target)>5",
                "anchor_gt7": "abs(anchor-target)>7"},
    "cohort_overlap": "nested_not_disjoint; never_sum_across_cohorts_or_marginal_views",
    "target_use": "evaluation_outcomes_and_cohorts_only; never_bins_or_preprocessing",
    "support": "count>=20_fixed_engineering_descriptive_threshold; retain_all_counts; empty_or_insufficient_not_no_signal",
    "residual": "prediction_minus_target; negative_underestimate; positive_overestimate; zero_exact",
    "serious": "abs(residual)>5; exactly_plus_or_minus_5_nonserious",
    "transitions": copy.deepcopy(transition.ACCOUNTING_CONTRACT["transitions"]),
    "prediction_loss": "new_bounded_influence_0.1*H_5(e)+4*H_2.5(max(abs(e)-4.5,0)); no_fit_penalties",
    "huber": "H_d(z)=z^2_if_abs(z)<=d_else_2*d*abs(z)-d^2",
    "contributions": "MAE_and_ordinary_excess_total_loss_sums_divided_by_entire_held_cell_n_not_subgroup; delta_corrected_minus_anchor",
    "correction": "actual_corrected_minus_anchor; signed_absolute_sums_maximum_magnitude_and_sign_counts; bound_2g_with_four_ulp_roundoff",
    "direction": "toward_opposite_nonzero_signs_of_anchor_error_and_correction; away_same_nonzero_signs; neutral_either_zero; toward_not_improved_due_to_overshoot",
    "outcomes": "abs_corrected_error_less_greater_equal_abs_anchor_error_improved_worsened_unchanged",
    "conservation": "within_each_cohort_each_marginal_and_joint_conserves_counts_exactly_and_contributions_rel_abs_tol_1e-12; reconcile_run017_accounting",
    "forbidden": "adaptive_quantiles_direction_selection_best_bin_ranking_threshold_optimization_extra_features_or_fits_learned_gate",
}


def _plan(raw: str, normalized: str, dependencies: Mapping[str, str]) -> dict[str, Any]:
    plan = prerequisite._plan(raw, normalized, dependencies)
    plan.pop("qualification_contract")
    plan.update({
        "version": VERSION, "kind": "training_only_descriptive_feature_signal_diagnostic",
        "hypothesis": "describe_fixed_input_associations_with_helped_and_hurt_errors_without_selecting_a_gate",
        "accounting_contract": copy.deepcopy(ACCOUNTING_CONTRACT),
        "new_loss_accounting": "descriptive_only_reuse_run017_numeric_accounting_no_qualification",
        "selection": "forbidden_descriptive_only", "locking": "forbidden_descriptive_only",
        "qualification": "forbidden_descriptive_only",
        "interpretation": INTERPRETATION,
        "stop_rule": "one_attempt_stop_regardless_outcomes_no_continuation_retry_or_budget_recycling",
    })
    return plan


def _feature_bin_indices(state: object, columns: Sequence[Sequence[float]]) -> Any:
    """Target-free exact prediction preprocessing; never refit statistics on held rows."""
    import numpy as np

    if not numeric.valid_state(state):
        raise ValueError("invalid_feature_signal_state")
    _, matrix = numeric.features(columns)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            standardized = np.clip((matrix - np.asarray(state["feature_means"]))
                                   / np.asarray(state["feature_scales"]), -1, 1)
    except FloatingPointError:
        raise ValueError("invalid_feature_signal_preprocessing") from None
    if not np.isfinite(standardized).all():
        raise ValueError("invalid_feature_signal_preprocessing")
    return np.searchsorted(np.asarray([-1.0 / 3.0, 1.0 / 3.0]), standardized, side="left")


def _check_partition(total: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> None:
    """Conserve every additive field, including sign/outcome/transition counts."""
    for key, expected in total.items():
        if key in ("support_sufficient", "support_status"):
            continue
        values = [group[key] for group in groups]
        if isinstance(expected, Mapping):
            _check_partition(expected, values)
        elif type(expected) is int:
            if sum(values) != expected:
                raise ValueError("nonconserving_feature_signal_counts")
        else:
            observed = max(values, default=0.0) if key == "maximum_absolute_g" else math.fsum(values)
            if not math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("nonconserving_feature_signal_contributions")


def summarize_feature_signal(
    state: object, columns: Sequence[Sequence[float]], targets: Sequence[float],
) -> dict[str, Any]:
    """Immediately reduce held predictions/inputs to fixed source-neutral aggregates."""
    import numpy as np

    bin_indices = _feature_bin_indices(state, columns)
    anchor = np.asarray(numeric.predict(state, columns, 0.0)[0])
    corrected = np.asarray(numeric.predict(state, columns, 1.0)[0])
    target = numeric._vector(targets)
    if target.shape != anchor.shape:
        raise ValueError("invalid_feature_signal_targets")
    # Existing run-017 accounting also checks the actual paired departure bound.
    paired = transition.summarize_correction_transitions(target.tolist(), anchor.tolist(), corrected.tolist())
    new_loss = prerequisite._new_loss_accounting(target.tolist(), anchor.tolist(), corrected.tolist())
    errors = np.asarray((anchor - target, corrected - target))
    absolute = np.abs(errors)
    components = [numeric.prediction_loss_components(error) for error in errors]
    correction = corrected - anchor
    n = len(target)
    transitions = 2 * (absolute[0] > 5).astype(int) + (absolute[1] > 5).astype(int)

    def group(mask: Any) -> dict[str, Any]:
        count = int(np.sum(mask))
        result: dict[str, Any] = {
            "count": count, "support_sufficient": count >= MINIMUM_SUPPORT,
            "support_status": "empty" if count == 0 else "sufficient" if count >= MINIMUM_SUPPORT else "insufficient",
        }
        for index, model in enumerate(("anchor", "corrected")):
            ordinary, excess = (math.fsum(values[mask].tolist()) / n for values in components[index])
            values = errors[index][mask]
            result[model] = {
                "mae_contribution_g": math.fsum(absolute[index][mask].tolist()) / n,
                "ordinary_loss_contribution": ordinary, "excess_loss_contribution": excess,
                "prediction_loss_contribution": ordinary + excess,
                "above_5g_count": int(np.sum(absolute[index][mask] > 5)),
                "residual_sign_counts": {"negative": int(np.sum(values < 0)), "zero": int(np.sum(values == 0)),
                                         "positive": int(np.sum(values > 0))},
            }
        result["delta"] = {key: result["corrected"][key] - result["anchor"][key]
                           for key in ("mae_contribution_g", "ordinary_loss_contribution",
                                       "excess_loss_contribution", "prediction_loss_contribution", "above_5g_count")}
        a, c, d = errors[0][mask], errors[1][mask], correction[mask]
        result["correction"] = {
            "signed_sum_g": math.fsum(d.tolist()), "absolute_sum_g": math.fsum(np.abs(d).tolist()),
            "maximum_absolute_g": float(np.max(np.abs(d))) if count else 0.0,
            "negative_count": int(np.sum(d < 0)), "zero_count": int(np.sum(d == 0)),
            "positive_count": int(np.sum(d > 0)),
            "toward_count": int(np.sum(((a < 0) & (d > 0)) | ((a > 0) & (d < 0)))),
            "away_count": int(np.sum(((a < 0) & (d < 0)) | ((a > 0) & (d > 0)))),
            "neutral_count": int(np.sum((a == 0) | (d == 0))),
            "absolute_error_improved_count": int(np.sum(np.abs(c) < np.abs(a))),
            "absolute_error_worsened_count": int(np.sum(np.abs(c) > np.abs(a))),
            "absolute_error_unchanged_count": int(np.sum(np.abs(c) == np.abs(a))),
        }
        result["transition_counts"] = {key: int(np.sum(transitions[mask] == index))
                                       for index, key in enumerate(transition.TRANSITIONS)}
        if result["delta"]["above_5g_count"] != result["transition_counts"]["harm"] - result["transition_counts"]["repair"]:
            raise ValueError("nonconserving_feature_signal_transitions")
        return result

    cohorts = {}
    for name, mask, baseline_bins in (
        ("all", np.ones(n, dtype=bool), None),
        ("anchor_gt5", absolute[0] > 5, ("gt5_le7", "gt7")),
        ("anchor_gt7", absolute[0] > 7, ("gt7",)),
    ):
        total = group(mask)
        marginal = {feature: {key: group(mask & (bin_indices[:, index] == bin_index))
                              for bin_index, key in enumerate(FEATURE_BINS)}
                    for index, feature in enumerate(numeric.FEATURES)}
        joint = {"|".join(FEATURE_BINS[i] for i in indices):
                 group(mask & np.all(bin_indices == np.asarray(indices), axis=1))
                 for indices in product(range(3), repeat=3)}
        for partition in (*marginal.values(), joint):
            _check_partition(total, list(partition.values()))
        # Independent baseline arithmetic retains run-017's exact new loss and old
        # non-loss summaries. Never reinterpret its old squared loss as Huber loss.
        old_groups = ([paired["total"]] if baseline_bins is None else
                      [paired["anchor_error_bins"][key] for key in baseline_bins])
        new_groups = ([new_loss["total"]] if baseline_bins is None else
                      [new_loss["anchor_error_bins"][key] for key in baseline_bins])
        baselines = []
        for old_group, new_group in zip(old_groups, new_groups):
            baseline = copy.deepcopy(old_group)
            baseline.pop("oracle")
            for model in ("anchor", "corrected", "delta"):
                baseline[model].update(new_group[model])
            baselines.append(baseline)
        _check_partition(total, baselines)
        cohorts[name] = {"total": total, "marginal": marginal, "joint": joint}
    result = {"count": n, "contribution_denominator": n, "cohorts": cohorts,
              "paired_transitions_with_old_diagnostic_loss": paired, "new_descriptive_loss": new_loss}
    transition._finite_tree(result)
    return result


@dataclass(frozen=True)
class FeatureSignalDiagnosticResult:
    status: str
    blockers: tuple[str, ...]
    evidence: Mapping[str, Any]
    resource_use: Mapping[str, Any]


def run_feature_signal_diagnostic(
    training_records: Dataset, config: EvaluationConfig, *, runtime: t.CandidateRuntime,
    output_root: str | Path, clock: Callable[[], float] = time.monotonic,
) -> FeatureSignalDiagnosticResult:
    """Execute four fixed training cells; never select, qualify, or lock a model."""
    if (config.seed != stability.NORMALIZATION_SEED or config.volume_unit != "mm3"
            or config.scope_confirmed is not True):
        raise InputError("invalid_feature_signal_configuration")
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
            raise RuntimeError("feature_signal_invalid_clock")
        if now - started >= MAXIMUM_ELAPSED_SECONDS:
            raise RuntimeError("feature_signal_deadline_reached")

    def before_fit() -> None:
        nonlocal fit_count
        check_deadline()
        if fit_count >= MAXIMUM_MODEL_FITS:
            raise RuntimeError("feature_signal_fit_limit_reached")
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
                state, columns, shift = prerequisite._fit_stage(
                    runtime, fitted, held, seed, before_fit, check_deadline,
                    assignment_seed=stability.INNER_SPLIT_SEEDS[split_seed],
                )
                check_deadline()
                summary = summarize_feature_signal(state, columns, targets)
                if fit_count - first_fit != 13 or summary["count"] != len(held):
                    raise ValueError("incomplete_cell")
                transition._validate_shift(shift, len(fitted), len(held))
                check_deadline()
                cells[active_cell] = {
                    "outer_split_seed": split_seed, "model_seed": seed,
                    "inner_split_seed": stability.INNER_SPLIT_SEEDS[split_seed],
                    "fitting_record_count": len(fitted), "held_out_record_count": len(held),
                    "fits": fit_count - first_fit, "feature_signal": summary,
                    "oof_full_fit_shift": shift,
                }
                active_cell = None
        if len(cells) != 4 or fit_count != MAXIMUM_MODEL_FITS:
            raise ValueError("incomplete_diagnostic")
        check_deadline()
        status = "completed"
    except Exception as error:
        # Preserve completed cells without exposing third-party error details.
        reason = str(error)
        blockers = (reason if reason in {
            "feature_signal_deadline_reached", "feature_signal_fit_limit_reached",
            "feature_signal_invalid_clock",
        } else "feature_signal_runtime_failed",)
    elapsed = clock() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        elapsed = 0.0
        status, blockers = "blocked", ("feature_signal_invalid_clock",)
    elif elapsed >= MAXIMUM_ELAPSED_SECONDS:
        status, blockers = "blocked", ("feature_signal_deadline_reached",)
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
        "interpretation": INTERPRETATION,
        "validation_eligible": False, "production_continuation": False,
        "selection_performed": False, "qualification_performed": False, "locking_performed": False,
    }
    transition._finite_tree(evidence)
    if not (output / "diagnostic-plan.json").exists():
        write_private_json(output / "diagnostic-plan.json", plan)
    write_private_json(output / "feature-signal-evidence.json", evidence)
    files = (output / "diagnostic-plan.json", output / "feature-signal-evidence.json")
    write_private_json(output / "manifest.json", {
        "version": VERSION, "create_only": True,
        "artifacts": {path.name: sha256(path.read_bytes()).hexdigest() for path in files},
        "validation_labels_used": False, "held_out_test_accessed": False, "publication_performed": False,
        "selection_performed": False, "qualification_performed": False, "locking_performed": False,
        "validation_eligible": False, "production_continuation": False,
    })
    return FeatureSignalDiagnosticResult(status, blockers, evidence, resources)


def build_parser() -> argparse.ArgumentParser:
    parser = PrivateArgumentParser(description="Private training-only feature-signal diagnostic.")
    parser.add_argument("--training-records", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--volume-unit", default="mm3")
    parser.add_argument("--scope-confirmed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_root.resolve() != PREDECLARED_OUTPUT_ROOT.resolve():
            raise InputError("feature_signal_output_root_mismatch")
        if args.output_root.exists():
            raise InputError("private_output_directory_unavailable")
        stability._verify_predeclared_training_artifact(args.training_records)
        runtime = t.TensorflowXGBoostCandidateRuntime()
        t._verify_predeclared_environment(runtime.dependency_versions)
        result = run_feature_signal_diagnostic(
            args.training_records,
            EvaluationConfig(None, args.volume_unit, True if args.scope_confirmed else None,
                             seed=stability.NORMALIZATION_SEED),
            runtime=runtime, output_root=args.output_root,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (ImportError, RuntimeError, OSError, ValueError, TypeError):
        raise SystemExit("feature_signal_diagnostic_failed") from None
    print(json.dumps({"status": result.status, "blockers": result.blockers,
                      "resource_use": result.resource_use}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
