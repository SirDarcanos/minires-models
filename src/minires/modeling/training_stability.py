"""A finite training-only diagnosis of split and model-seed sensitivity.

This module deliberately has no validation or held-out-test input.  It holds the
completed tail-focused round's frozen anchor and correction algorithm fixed while
crossing independently assigned training holdouts with model random seeds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from ..evaluation import EvaluationConfig
from ..ingestion import Dataset, InputError, fingerprint, load_records, normalize
from ..private_io import write_private_json
from ..source_identity import code_fingerprint
from . import tail_correction as numeric
from . import tail_focused_search as tail
from . import tuning as t

VERSION = "minires-training-stability-diagnostic-v1"
MODEL_SEEDS = (41, 42)
OUTER_SPLIT_SEEDS = (101, 202)
INNER_SPLIT_SEEDS = {101: 1101, 202: 1202}
FOLDS = 5
# Each crossed cell has ten inner-OOF base fits, two outer-partition base
# fits, and one deterministic numerical correction fit.
MAXIMUM_MODEL_FITS = 52
MAXIMUM_ELAPSED_SECONDS = 7200.0
NORMALIZATION_SEED = 17
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDECLARED_OUTPUT_ROOT = PROJECT_ROOT / "private" / "candidate-tuning" / "run-015"


@dataclass(frozen=True)
class TrainingStabilityDiagnosticResult:
    """Aggregate diagnostic evidence; it cannot select, rank, or lock a candidate."""

    status: str
    blockers: tuple[str, ...]
    evidence: Mapping[str, Any]
    resource_use: Mapping[str, Any]


def _metric_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mae_g": float(right["mae_g"]) - float(left["mae_g"]),
        "above_5g_count": int(right["above_5g_count"]) - int(left["above_5g_count"]),
        "prediction_loss": float(right["prediction_loss"]) - float(left["prediction_loss"]),
    }


def _plan(input_fingerprint: str, normalized_input_fingerprint: str) -> dict[str, Any]:
    bases = tail.base_candidates()
    return {
        "version": VERSION,
        "kind": "training_only_crossed_split_model_seed_diagnostic",
        "hypothesis": (
            "the frozen closest anchor and correction vary materially with model initialization "
            "or with deterministic training-holdout assignment"
        ),
        "input_fingerprint": input_fingerprint,
        "normalized_input_fingerprint": normalized_input_fingerprint,
        "code_fingerprint": code_fingerprint(),
        "model_seeds": list(MODEL_SEEDS),
        "outer_split_seeds": list(OUTER_SPLIT_SEEDS),
        "inner_split_seeds": {str(key): value for key, value in INNER_SPLIT_SEEDS.items()},
        "fold_assignment": "stable-record-identity-sha256-round-robin-v1",
        "outer_holdout": "rank_modulo_5_equals_zero",
        "inner_folds": FOLDS,
        "base_candidates": [asdict(candidate) for candidate in bases],
        "fixed_training_counts": dict(tail.FIXED_COUNTS),
        "correction_contract": numeric.CONTRACT,
        "maximum_fits": MAXIMUM_MODEL_FITS,
        "maximum_elapsed_seconds": MAXIMUM_ELAPSED_SECONDS,
        "validation_input": "unavailable_to_this_interface",
        "held_out_test_input": "unavailable_to_this_interface",
        "selection": "none_diagnostic_only",
        "locking": "forbidden_diagnostic_only",
        "publication": "none_private_aggregate_evidence_only",
    }


def _cell_key(split_seed: int, model_seed: int) -> str:
    return f"split-{split_seed}-model-{model_seed}"


def _build_contrasts(
    cells: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_seed_contrasts: dict[str, Any] = {}
    for split_seed in OUTER_SPLIT_SEEDS:
        first = cells[_cell_key(split_seed, MODEL_SEEDS[0])]
        second = cells[_cell_key(split_seed, MODEL_SEEDS[1])]
        model_seed_contrasts[f"split-{split_seed}"] = {
            "comparison": "model_seed_42_minus_model_seed_41_on_identical_outer_holdout",
            "anchor": _metric_difference(first["anchor"], second["anchor"]),
            "corrected": _metric_difference(first["corrected"], second["corrected"]),
        }
    split_contrasts: dict[str, Any] = {}
    for model_seed in MODEL_SEEDS:
        first = cells[_cell_key(OUTER_SPLIT_SEEDS[0], model_seed)]
        second = cells[_cell_key(OUTER_SPLIT_SEEDS[1], model_seed)]
        split_contrasts[f"model-{model_seed}"] = {
            "comparison": "split_202_minus_split_101_on_different_outer_holdouts_not_a_paired_row_comparison",
            "anchor": _metric_difference(first["anchor"], second["anchor"]),
            "corrected": _metric_difference(first["corrected"], second["corrected"]),
        }
    return model_seed_contrasts, split_contrasts


def run_training_stability_diagnostic(
    training_records: Dataset,
    config: EvaluationConfig,
    *,
    runtime: t.CandidateRuntime,
    output_root: str | Path,
    clock: Callable[[], float] = time.monotonic,
) -> TrainingStabilityDiagnosticResult:
    """Run exactly four training-only crossed cells using the frozen tail correction.

    The public seam takes one development artifact only.  It writes aggregate,
    create-only private evidence and never scores validation data, selects a
    candidate, or creates a lock.
    """
    if (config.seed != NORMALIZATION_SEED or config.volume_unit != "mm3"
            or config.scope_confirmed is not True):
        raise InputError("invalid_training_stability_configuration")
    output = Path(output_root)
    if "private" not in output.resolve().parts:
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None

    loaded, input_fingerprint = load_records(training_records)
    training = normalize(loaded, config, contract="legacy")
    if not t._valid_candidate_feature_data(training, t.LEGACY_FEATURES):
        raise InputError("invalid_candidate_feature_data")
    identities = [str(row.metadata.get("record_identity", "")) for row in training]
    if (not training or any(row.outcome != "included" or not identity
                            or not row.metadata.get("anonymous_source_group")
                            for row, identity in zip(training, identities))
            or len(identities) != len(set(identities))):
        raise InputError("invalid_training_stability_records")
    normalized_fingerprint = fingerprint([asdict(row) for row in training])
    plan = _plan(input_fingerprint, normalized_fingerprint)
    write_private_json(output / "diagnostic-plan.json", plan)

    started = clock()
    cpu_started = time.process_time()
    deadline = started + MAXIMUM_ELAPSED_SECONDS
    fit_count = 0

    def check_deadline() -> None:
        if clock() >= deadline:
            raise RuntimeError("training_stability_deadline_reached")

    def before_fit() -> None:
        nonlocal fit_count
        if fit_count >= MAXIMUM_MODEL_FITS:
            raise RuntimeError("training_stability_fit_limit_reached")
        check_deadline()
        fit_count += 1

    cells: dict[str, Any] = {}
    blockers: tuple[str, ...] = ()
    status = "completed"
    try:
        for split_seed in OUTER_SPLIT_SEEDS:
            outer = t._cross_fit_assignments(training, split_seed, FOLDS)
            fitted_rows = [row for row, fold in zip(training, outer) if fold != 0]
            held_rows = [row for row, fold in zip(training, outer) if fold == 0]
            if not fitted_rows or not held_rows:
                raise ValueError("invalid_training_stability_partition")
            _, targets = t.candidate_prediction_matrix(held_rows, tail.base_candidates()[0])
            for model_seed in MODEL_SEEDS:
                state, _, columns, shift = tail._fit_stage(
                    runtime, fitted_rows, held_rows, model_seed, before_fit, check_deadline,
                    assignment_seed=INNER_SPLIT_SEEDS[split_seed],
                )
                anchor_predictions = numeric.predict(state, columns, 0.0)[0]
                corrected_predictions = numeric.predict(state, columns, 1.0)[0]
                cells[_cell_key(split_seed, model_seed)] = {
                    "outer_split_seed": split_seed,
                    "inner_split_seed": INNER_SPLIT_SEEDS[split_seed],
                    "model_seed": model_seed,
                    "fitting_record_count": len(fitted_rows),
                    "held_out_record_count": len(held_rows),
                    "anchor": tail._metrics(targets, anchor_predictions),
                    "corrected": tail._metrics(targets, corrected_predictions),
                    "correction_delta": _metric_difference(
                        tail._metrics(targets, anchor_predictions),
                        tail._metrics(targets, corrected_predictions),
                    ),
                    "qualified": None,
                    "validation_labels_used": False,
                    "held_out_test_accessed": False,
                    "oof_full_fit_shift": shift,
                }
        check_deadline()
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        status = "blocked"
        reason = str(error)
        blockers = (reason if reason in {
            "training_stability_fit_limit_reached", "training_stability_deadline_reached",
        } else "training_stability_runtime_failed",)

    model_seed_contrasts: dict[str, Any] = {}
    split_contrasts: dict[str, Any] = {}
    if status == "completed":
        model_seed_contrasts, split_contrasts = _build_contrasts(cells)
    evidence = {
        "version": VERSION,
        "status": status,
        "blockers": list(blockers),
        "cells": cells,
        "model_seed_contrasts": model_seed_contrasts,
        "split_contrasts": split_contrasts,
        "interpretation": (
            "descriptive training-holdout evidence only; it does not establish a cause, "
            "select a candidate, or evaluate unchanged validation eligibility gates"
        ),
        "validation_labels_used": False,
        "held_out_test_accessed": False,
    }
    resource_use = {
        "elapsed_seconds": max(0.0, clock() - started),
        "process_cpu_seconds": time.process_time() - cpu_started,
        "fits": fit_count,
        "maximum_fits": MAXIMUM_MODEL_FITS,
    }
    write_private_json(output / "training-stability-evidence.json", evidence)
    files = (output / "diagnostic-plan.json", output / "training-stability-evidence.json")
    write_private_json(output / "manifest.json", {
        "version": VERSION,
        "create_only": True,
        "artifacts": {path.name: sha256(path.read_bytes()).hexdigest() for path in files},
        "validation_labels_used": False,
        "held_out_test_accessed": False,
        "publication_performed": False,
    })
    return TrainingStabilityDiagnosticResult(status, blockers, evidence, resource_use)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the private training-only crossed split/model-seed stability diagnostic."
    )
    parser.add_argument("--training-records", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--volume-unit", default="mm3")
    parser.add_argument("--scope-confirmed", action="store_true")
    return parser


def _verify_predeclared_training_artifact(training_records: Path) -> None:
    data_root = Path(__file__).resolve().parents[3] / "data"
    expected = data_root / "train.jsonl"
    if training_records.resolve() != expected.resolve():
        raise InputError("training_stability_artifact_checksum_mismatch")
    try:
        manifest = json.loads((data_root / "manifest.json").read_text())
        checksum = t.GUARDED_RESIDUAL_DEVELOPMENT_CHECKSUMS["train.jsonl"]
        if (manifest["artifacts"].get("train.jsonl") != checksum
                or sha256(expected.read_bytes()).hexdigest() != checksum):
            raise ValueError("checksum mismatch")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        raise InputError("training_stability_artifact_checksum_mismatch") from None


def _verify_predeclared_output_root(output_root: Path) -> None:
    if output_root.resolve() != PREDECLARED_OUTPUT_ROOT.resolve():
        raise InputError("training_stability_output_root_mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _verify_predeclared_training_artifact(args.training_records)
        _verify_predeclared_output_root(args.output_root)
        try:
            runtime: t.CandidateRuntime = t.TensorflowXGBoostCandidateRuntime()
        except ImportError:
            runtime = t._BlockedCandidateRuntime("candidate_tuning_dependencies_required")
        except RuntimeError:
            runtime = t._BlockedCandidateRuntime("candidate_tuning_runtime_unavailable")
        t._verify_predeclared_environment(runtime.dependency_versions)
        result = run_training_stability_diagnostic(
            args.training_records,
            EvaluationConfig(None, args.volume_unit, True if args.scope_confirmed else None,
                             seed=NORMALIZATION_SEED),
            runtime=runtime, output_root=args.output_root,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, ValueError, TypeError):
        raise SystemExit("training_stability_diagnostic_failed") from None
    print(json.dumps({
        "status": result.status,
        "blockers": result.blockers,
        "resource_use": result.resource_use,
        "validation_labels_used": False,
        "held_out_test_accessed": False,
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
